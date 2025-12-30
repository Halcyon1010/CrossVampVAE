import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math
import lpips
# 假设你已经有了之前的 BiCrossAttnFlow 定义
# from models.flow import BiCrossAttnFlow 

class ResBlock(nn.Module):
    """
    Simple ResBlock: Conv-GN-SiLU-Conv-GN + skip
    """
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, out_ch),
        )
        self.act = nn.SiLU()
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        return self.act(h + self.shortcut(x))


class SelfAttention(nn.Module):
    """
    Lightweight spatial self-attention at low resolution
    """
    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x).view(B, C, H * W).permute(0, 2, 1)   # [B, N, C]
        k = self.k(x).view(B, C, H * W)                    # [B, C, N]
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(C), dim=-1)  # [B, N, N]
        v = self.v(x).view(B, C, H * W)                    # [B, C, N]
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        out = self.proj(out)
        return x + self.gamma * out

class CrossAttnBlock(nn.Module):
    """
    Target tokens attend to condition tokens (cross-attn), then FFN.
    q_tokens: [B, Tq, D]
    kv_tokens: [B, Tk, D]
    """
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)

        # Q from target, K/V from condition
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = q_tokens + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnSTNet(nn.Module):
    """
    Predict (s, t) for target vector conditioned on cond vector via cross-attention.

    target: [B, Dt]  -> split into seq_len tokens
    cond:   [B, Dc]  -> split into seq_len tokens
    output: s,t each [B, Dt]
    """
    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        seq_len: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert target_dim % seq_len == 0, f"target_dim ({target_dim}) must be divisible by seq_len ({seq_len})"
        assert cond_dim % seq_len == 0, f"cond_dim ({cond_dim}) must be divisible by seq_len ({seq_len})"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.seq_len = seq_len
        self.t_tok = target_dim // seq_len
        self.c_tok = cond_dim // seq_len

        self.t_in = nn.Linear(self.t_tok, embed_dim)
        self.c_in = nn.Linear(self.c_tok, embed_dim)

        # Learnable positional embeddings
        self.t_pos = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.c_pos = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        self.blocks = nn.ModuleList(
            [CrossAttnBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )

        self.t_out = nn.Linear(embed_dim, self.t_tok)

        # Map flattened target back to (s,t)
        self.out_proj = nn.Linear(target_dim, 2 * target_dim)

        # Critical: identity init so flow starts near identity (stable training)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Small init for positions
        nn.init.normal_(self.t_pos, std=0.02)
        nn.init.normal_(self.c_pos, std=0.02)

    def forward(self, target: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = target.size(0)
        t = target.view(B, self.seq_len, self.t_tok)
        c = cond.view(B, self.seq_len, self.c_tok)

        t = self.t_in(t) + self.t_pos
        c = self.c_in(c) + self.c_pos

        for blk in self.blocks:
            t = blk(t, c)

        t = self.t_out(t).contiguous().view(B, self.target_dim)
        st = self.out_proj(t)
        s, tt = st.chunk(2, dim=1)
        return s, tt


# ============================================================
# 2) Bi-directional cross-attention affine coupling (invertible)
# ============================================================
class BiCrossAttnCoupling(nn.Module):
    """
    Two-step bi-directional coupling:
      Step A: update b conditioned on a
      Step B: update a conditioned on updated b

    Each step is affine coupling => triangular Jacobian => exact log-det.
    """
    def __init__(
        self,
        dim: int,
        seq_len: int = 8,
        embed_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "latent dim must be even for half split"
        self.dim = dim
        self.half = dim // 2

        self.st_b = CrossAttnSTNet(
            target_dim=self.half,
            cond_dim=self.half,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=heads,
            num_layers=layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.st_a = CrossAttnSTNet(
            target_dim=self.half,
            cond_dim=self.half,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=heads,
            num_layers=layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Scale gate to prevent exp(s) explosion
        self.scale = nn.Parameter(torch.zeros(1))

    def _affine(self, z: torch.Tensor, s: torch.Tensor, t: torch.Tensor, reverse: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.scale * torch.tanh(s)
        if not reverse:
            z_new = z * (torch.exp(s) + 1e-3) + t  # 加个 epsilon 防止乘 0
            log_det = s.sum(dim=1, keepdim=True)
        else:
            z_new = (z - t) * torch.exp(-s)
            log_det = -s.sum(dim=1, keepdim=True)
        return z_new, log_det

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = x[:, :self.half], x[:, self.half:]
        log_det = x.new_zeros(x.size(0), 1)


        if not reverse:
            # Step A: b | a
            s_b, t_b = self.st_b(b, a)
            b, inc = self._affine(b, s_b, t_b, reverse=False)
            log_det += inc

            # Step B: a | b (updated)
            s_a, t_a = self.st_a(a, b)
            a, inc = self._affine(a, s_a, t_a, reverse=False)
            log_det += inc
        else:
            # Inverse in reverse order
            # Undo Step B
            s_a, t_a = self.st_a(a, b)
            a, inc = self._affine(a, s_a, t_a, reverse=True)
            log_det += inc

            # Undo Step A (now use recovered a)
            s_b, t_b = self.st_b(b, a)
            b, inc = self._affine(b, s_b, t_b, reverse=True)
            log_det += inc

        return torch.cat([a, b], dim=1), log_det


class BiCrossAttnFlow(nn.Module):
    """
    Stack multiple bi-directional coupling layers with fixed random permutations.
    """
    def __init__(
        self,
        dim: int,
        length: int,
        seq_len: int = 8,
        embed_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BiCrossAttnCoupling(
                    dim=dim,
                    seq_len=seq_len,
                    embed_dim=embed_dim,
                    heads=heads,
                    layers=layers,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(length)
            ]
        )

        perms, invs = [], []
        for _ in range(length):
            p = torch.randperm(dim)
            ip = torch.empty_like(p)
            ip[p] = torch.arange(dim)
            perms.append(p)
            invs.append(ip)

        self.register_buffer("perms", torch.stack(perms))      # [L, dim]
        self.register_buffer("inv_perms", torch.stack(invs))   # [L, dim]

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = x.new_zeros(x.size(0), 1)
        L = len(self.layers)

        if not reverse:
            for i in range(L):
                x = x[:, self.perms[i]]
                x, inc = self.layers[i](x, reverse=False)
                log_det += inc
        else:
            for i in reversed(range(L)):
                x, inc = self.layers[i](x, reverse=True)
                log_det += inc
                x = x[:, self.inv_perms[i]]

        return x, log_det



    

class CrossFlowVampVAE(nn.Module):
    """
    A fair ablation-ready VampVAE with optional BiCrossAttnFlow posterior.
    - flow_length = 0 => No-flow baseline (same encoder/decoder/prior/loss)
    - flow_length > 0 => Flow version (only difference is posterior transformation)
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        img_size: int = 32,
        num_components: int = 50,

        # Flow params
        flow_length: int = 8,
        flow_seq_len: int = 8,
        flow_embed_dim: int = 128,
        flow_heads: int = 4,
        flow_layers: int = 2,
        flow_dropout: float = 0.0,
        flow_mlp_ratio: float = 4.0,

        # Loss / recon
        recon_loss: str = "l1",   # "l1" or "mse"
        lpips_weight: float = 100.0,

        # Architecture knobs (match your no-flow ResVampVAE)
        use_attn_in_enc: bool = True,
        use_attn_in_dec: bool = True,
        gn_groups: int = 8,

        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_components = num_components
        self.recon_loss_type = recon_loss
        self.lpips_weight = lpips_weight
        self.use_attn_in_enc = use_attn_in_enc
        self.use_attn_in_dec = use_attn_in_dec

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        self.hidden_dims = list(hidden_dims)
        self.last_h = self.hidden_dims[-1]

        # -------------------------
        # Encoder: Downsample + ResBlock (+ optional attention at deepest)
        # -------------------------
        enc_modules = []
        c_in = in_channels
        for i, h in enumerate(self.hidden_dims):
            enc_modules.append(
                nn.Sequential(
                    nn.Conv2d(c_in, h, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(gn_groups, h),
                    nn.SiLU(),
                    ResBlock(h, h, groups=gn_groups),
                )
            )
            if self.use_attn_in_enc and (i == len(self.hidden_dims) - 1):
                enc_modules.append(SelfAttention(h))
            c_in = h
        self.encoder = nn.Sequential(*enc_modules)

        # Infer feature map size and flat dim robustly
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            enc_out = self.encoder(dummy)
            self.enc_shape = enc_out.shape[1:]  # [C, H, W]
            self.flat_dim = int(enc_out.numel())

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # -------------------------
        # VampPrior pseudo inputs (match data range [-1,1])
        # -------------------------
        self.pseudo_img_shape = (in_channels, img_size, img_size)
        flat_img_dim = in_channels * img_size * img_size
        self.register_buffer("pseudo_id", torch.eye(num_components))
        self.embed_pseudo = nn.Sequential(
            nn.Linear(num_components, flat_img_dim),
            nn.Hardtanh(-1.0, 1.0),   # IMPORTANT: match tanh/Normalize -> [-1,1]
        )

        # -------------------------
        # Optional Flow 
        # -------------------------
        if flow_length > 0:
            self.flow_module = BiCrossAttnFlow(
                dim=latent_dim,
                length=flow_length,
                seq_len=flow_seq_len,
                embed_dim=flow_embed_dim,
                heads=flow_heads,
                layers=flow_layers,
                dropout=flow_dropout,
                mlp_ratio=flow_mlp_ratio,
            )
        else:
            self.flow_module = None

        # -------------------------
        # Decoder: Linear -> reshape -> Upsample blocks (+ optional attention early)
        # -------------------------
        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)

        dec_dims = list(reversed(self.hidden_dims))  # e.g. [512,256,128,64]
        dec_modules = []
        for i in range(len(dec_dims) - 1):
            in_ch = dec_dims[i]
            out_ch = dec_dims[i + 1]
            dec_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                    nn.GroupNorm(gn_groups, out_ch),
                    nn.SiLU(),
                    ResBlock(out_ch, out_ch, groups=gn_groups),
                )
            )
            # add attention near the beginning of decoding (lowest res)
            if self.use_attn_in_dec and i == 0:
                dec_modules.append(SelfAttention(out_ch))
        self.decoder = nn.Sequential(*dec_modules)

        # final upsample to original resolution
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(dec_dims[-1], dec_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.GroupNorm(gn_groups, dec_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(dec_dims[-1], in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # -------------------------
        # LPIPS 
        # -------------------------
        self.lpips_loss = lpips.LPIPS(net="vgg").to(device)
        self.lpips_loss.eval()
        for p in self.lpips_loss.parameters():
            p.requires_grad = False

        # constants
        self._log_2pi = math.log(2.0 * math.pi)

    # -------------------------
    # core
    # -------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, -10.0, 10.0)  # stability (align with your other VAE)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.view(z.size(0), *self.enc_shape)   # robust reshape
        h = self.decoder(h)
        return self.final_layer(h)

    def log_normal_diag(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # returns [B]
        return -0.5 * (self._log_2pi + logvar + (x - mu).pow(2) / torch.exp(logvar)).sum(dim=1)

    def forward(self, x: torch.Tensor, **kwargs):
        # 1) encode -> z0
        mu, log_var = self.encode(x)
        z0 = self.reparameterize(mu, log_var)

        # 2) flow -> zk, log_det
        if self.flow_module is not None:
            zk, log_det_flow = self.flow_module(z0, reverse=False)  # log_det_flow: [B, 1]
        else:
            zk = z0
            log_det_flow = x.new_zeros(x.size(0), 1)

        # 3) decode from zk
        recons = self.decode(zk)

        # IMPORTANT: return 7 values for flow-loss
        return [recons, x, mu, log_var, z0, zk, log_det_flow]

    # -------------------------
    # Loss (Vamp + optional flow)
    # -------------------------
    def loss_function(self, *args, **kwargs) -> dict:
        """
        Args order (flow version):
        recons, input, mu, log_var, z0, zk, log_det_flow
        Return keys MUST match your trainer:
        'loss', 'Reconstruction_Loss', 'LPIPS_Loss', 'KLD_Raw', 'KLD_Optim'
        """
        self.lpips_loss.eval()

        recons, input, mu, log_var, z0, zk, log_det_flow = args
        kld_weight = kwargs.get("M_N", 1.0)

        # -------------------------
        # LPIPS (keep your behavior)
        # -------------------------
        recons_clamped = torch.clamp(recons, -1.0, 1.0)
        input_clamped  = torch.clamp(input,  -1.0, 1.0)
        p_loss = self.lpips_loss(recons_clamped, input_clamped, normalize=False).mean()
        p_loss = torch.relu(p_loss)

        # -------------------------
        # Pixel recon
        # -------------------------
        pixel_loss = F.l1_loss(recons, input, reduction="none").view(input.size(0), -1).sum(1).mean()
        total_recons_loss = pixel_loss + p_loss * 100 # keep consistent with your baseline

        # -------------------------
        # VampPrior log p(zk)
        # -------------------------
        pseudo_imgs = self.embed_pseudo(self.pseudo_id).view(-1, *self.pseudo_img_shape)
        prior_mu, prior_log_var = self.encode(pseudo_imgs)

        prior_log_var = prior_log_var.clamp(min=-6.0, max=6.0)

        B, D = zk.shape
        z_expand = zk.unsqueeze(1)                 # [B,1,D]
        prior_mu_exp = prior_mu.unsqueeze(0)       # [1,K,D]
        prior_log_var_exp = prior_log_var.unsqueeze(0)

        LOG_2PI = math.log(2.0 * math.pi)

        log_prob_components = -0.5 * (
            LOG_2PI +
            prior_log_var_exp +
            (z_expand - prior_mu_exp).pow(2) / torch.exp(prior_log_var_exp)
        ).sum(dim=2)                                # [B,K]

        log_p_zk = torch.logsumexp(log_prob_components, dim=1) - math.log(float(self.num_components))  # [B]

        # -------------------------
        # log q(zk|x) via change-of-variables
        # log q0(z0|x)
        # -------------------------
        log_q_z0 = -0.5 * (
            LOG_2PI +
            log_var +
            (z0 - mu).pow(2) / torch.exp(log_var)
        ).sum(dim=1)                                # [B]

        # log qk(zk|x) = log q0(z0|x) - log_det
        log_det = log_det_flow.squeeze(1) if log_det_flow.dim() == 2 else log_det_flow
        log_q_zk = log_q_z0 - log_det               # [B]

        # -------------------------
        # KL raw & free-bits hinge
        # -------------------------
        kld_raw = (log_q_zk - log_p_zk).mean()

        free_bits_per_dim = 0.5
        target_kl_threshold = free_bits_per_dim * self.latent_dim
        kld_loss = torch.max(
            kld_raw - target_kl_threshold,
            torch.tensor(0.0, device=zk.device)
        )

        # -------------------------
        # Total
        # -------------------------
        loss = total_recons_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "Reconstruction_Loss": pixel_loss.detach(),
            "LPIPS_Loss": p_loss.detach(),
            "KLD_Raw": kld_raw.detach(),
            "KLD_Optim": kld_loss.detach(),
        }

    # -------------------------
    # Sampling from VampPrior
    # -------------------------
    @torch.no_grad()
    def sample(self, num_samples: int, current_device: torch.device):
        pseudo_imgs = self.embed_pseudo(self.pseudo_id).view(-1, *self.pseudo_img_shape)
        prior_mu, prior_logvar = self.encode(pseudo_imgs)

        idx = torch.randint(0, self.num_components, (num_samples,), device=current_device)
        mu = prior_mu[idx]
        logvar = prior_logvar[idx]
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z)
