import argparse
import os
os.environ["TORCH_HOME"] = "/mnt/afs/250010063/torch_cache"
from pathlib import Path
from typing import Any, Dict, Tuple
from torchvision.utils import make_grid, save_image
import torch
import torchvision
import yaml
from torchvision import transforms
from models.crossVAE import InvertibleVAE
from utils import TxtLogger
#from models import *  # assumes vae_models is defined here
from Proposed_VAE import MSSSIMImprovedVAE
from models.vae_flow import PlanarVAE
from models.vamp_flow import CrossFlowVampVAE
from models.ResVamp import ResVampVAE
import torch.nn as nn
import PIL.Image as Image
# -----------------------------
# Args / Config
# -----------------------------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CIFAR-10 real images and generate 50k samples for FID.")

    # Paths
    parser.add_argument("--data_dir", type=str, default="/mnt/afs/250010063/DL4/data",
                        help="Dataset root directory (CIFAR-10 will be downloaded here).")
    parser.add_argument("--save_dir", type=str, default="/mnt/afs/250010063/DL4/result",
                        help="Directory to save logs.")
    parser.add_argument("--resume", type=str, default=r"/mnt/afs/250010063/DL4/result/VampResNet_CIFAR10_CrossFlow_no_flow/best_fid.pth",
                        help="Path to checkpoint containing model state_dict (expects key 'model').")
    parser.add_argument("--config", "-c", dest="filename", metavar="FILE", default=r"/mnt/afs/250010063/DL4/configs/vae.yaml",
                        help="Path to YAML config file.")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # Export options
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of images to generate for FID.")
    parser.add_argument("--gen_batch_size", type=int, default=100,
                        help="Batch size for image generation.")
    parser.add_argument("--real_dirname", type=str, default="real_cifar",
                        help="Subfolder name to save real images.")
    parser.add_argument("--fake_dirname", type=str, default="fake_cifar",
                        help="Subfolder name to save generated images.")

    # DataLoader (only used for CIFAR download; kept for compatibility)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers.")
    parser.add_argument("--prior_ckpt", type=str, default=r'/mnt/afs/250010063/DL4/result/prior_out/prior_ep050.pth',
                    help="Path to prior checkpoint (.pth) containing key 'prior' and cfg (H/W/seq_len...).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--recon_dirname", type=str, default="recon_cifar",
                        help="Subfolder name to save reconstructed images.")
    parser.add_argument("--do_recon", default=False,
                        help="If set, reconstruct CIFAR-10 train images and save for FID.")
    parser.add_argument("--recon_batch_size", type=int, default=256,
                        help="Batch size for reconstruction.")

    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Model / Data
# -----------------------------
def get_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and device_str.startswith("cuda"):
        return torch.device(device_str)
    return torch.device("cpu")

class TransformerPrior(nn.Module):
    def __init__(self, vocab_size, seq_len,
                 d_model=512, n_head=8, n_layer=8, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.bos_id = vocab_size - 1
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(block, n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)


        mask = torch.full((seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf，其余为 0
        self.register_buffer("causal_mask", mask)

    def forward(self, idx):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.transformer(x, mask=self.causal_mask[:L, :L])
        x = self.ln(x)
        return self.head(x)   # [B, L, K]

    @torch.no_grad()
    def sample(self, num_samples, device, temperature=1.0, top_k=64):
        self.eval()
        # idx = torch.zeros(num_samples, self.seq_len, device=device, dtype=torch.long)

        idx = torch.full(
                        (num_samples, self.seq_len),
                        fill_value=self.bos_id,
                        device=device,
                        dtype=torch.long
                    )

        for t in range(self.seq_len):
            logits = self(idx)[:, t, :]
            logits = self(idx)[:, t, :]  # [B, K+1]
            logits[:, self.bos_id] = -1e10  # 禁止采样 BOS
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                thr = v[:, [-1]]
                logits = torch.where(logits < thr, torch.full_like(logits, -1e10), logits)

            probs = F.softmax(logits, dim=-1)
            idx[:, t] = torch.multinomial(probs, 1).squeeze(1)
            
        return idx

def build_model(config: Dict[str, Any], device: torch.device, ckpt_path: str = "") -> torch.nn.Module:
    model_name = config["model_params"]["name"]
    model = vae_models[model_name](**config["model_params"]).to(device)

    if ckpt_path:
        weights = torch.load(ckpt_path, map_location=device)
        state = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
        model.load_state_dict(state)

    model.eval()
    return model

def build_prior(device: torch.device, prior_ckpt: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(prior_ckpt, map_location=device)
    if "cfg" not in ckpt or "prior" not in ckpt:
        raise ValueError("prior_ckpt must contain keys: 'prior' and 'cfg'.")

    cfg = ckpt["cfg"]
    prior = TransformerPrior(
        vocab_size=cfg["vocab_size"]+1,
        seq_len=cfg["seq_len"],
        d_model=cfg.get("d_model", 512),
        n_head=cfg.get("n_head", 8),
        n_layer=cfg.get("n_layer", 8),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    prior.load_state_dict(ckpt["prior"], strict=True)
    prior.eval()
    return prior, cfg

@torch.no_grad()
def generate_fake_images_vqvae_with_prior(vqvae: torch.nn.Module,
                                          prior: torch.nn.Module,
                                          prior_cfg: Dict[str, Any],
                                          device: torch.device,
                                          fake_dir: Path,
                                          num_samples: int,
                                          batch_size: int,
                                          temperature: float,
                                          top_k: int) -> None:
    """
    Unconditional generation for VQ-VAE using a learned autoregressive prior:
      indices ~ p(z) -> decode_from_indices(indices_hw) -> image
    Assumes vqvae.decode_from_indices returns images in [-1, 1].
    """
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(fake_dir.glob("*.png")))
    if existing > 0:
        print(f"[Fake] Warning: {fake_dir} already contains {existing} png files. "
              f"Consider clearing it to avoid mixing runs.")

    H, W = int(prior_cfg["H"]), int(prior_cfg["W"])
    L = int(prior_cfg["seq_len"])
    if H * W != L:
        raise ValueError(f"Invalid prior cfg: H*W={H*W} != seq_len={L}")

    idx = 0
    steps = (num_samples + batch_size - 1) // batch_size
    for _ in range(steps):
        cur_bs = min(batch_size, num_samples - idx)

        idx_seq = prior.sample(cur_bs, device=device, temperature=temperature, top_k=top_k)  # [B,L]
        inds_hw = idx_seq.view(cur_bs, H, W).long()

        samples = vqvae.decode_from_indices(inds_hw)            # [-1,1]
        samples = torch.clamp((samples + 1) / 2, 0, 1)          # [0,1]

        for i in range(cur_bs):
            save_image(samples[i], fake_dir / f"{idx:05d}.png")
            idx += 1

    print(f"[Fake] Saved {num_samples} generated images.")

def get_cifar10_train(data_dir: str) -> torchvision.datasets.CIFAR10:
    ds = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    return ds


# -----------------------------
# Export Real / Generate Fake
# -----------------------------
def export_real_images(dataset: torchvision.datasets.CIFAR10, real_dir: Path) -> None:
    """
    Export CIFAR-10 training set (50k) to a folder for pytorch-fid.
    """
    real_dir.mkdir(parents=True, exist_ok=True)

    # If already exported, skip (fast path)
    existing = len(list(real_dir.glob("*.png")))
    if existing >= 50000:
        print(f"[Real] Found {existing} images in {real_dir}. Skip exporting.")
        return

    to_pil = transforms.ToPILImage()
    for i, (img, _) in enumerate(dataset):
        out_path = real_dir / f"{i:05d}.png"
        if out_path.exists():
            continue
        to_pil(img).save(out_path)

    print("[Real] Saved 50,000 real CIFAR-10 images.")


@torch.no_grad()
def generate_fake_images(model: torch.nn.Module,
                         device: torch.device,
                         fake_dir: Path,
                         num_samples: int,
                         batch_size: int,
                         latent_dim: int) -> None:
    """
    Generate num_samples images from a VAE-like model by sampling z ~ N(0, I) and decoding.
    Assumes model.decode(z) returns images in [-1, 1].
    """
    #model.train()
    model.eval()
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Optional warning if folder not empty
    existing = len(list(fake_dir.glob("*.png")))
    if existing > 0:
        print(f"[Fake] Warning: {fake_dir} already contains {existing} png files. "
              f"Consider clearing it to avoid mixing runs.")

    idx = 0
    steps = (num_samples + batch_size - 1) // batch_size
    for _ in range(steps):
        cur_bs = min(batch_size, num_samples - idx)
        # z = torch.randn(cur_bs, latent_dim, device=device)
        with torch.no_grad():
            samples = model.sample(cur_bs, device)                    # expected [-1, 1]
            samples = torch.clamp((samples + 1) / 2, 0, 1)  # -> [0, 1]
            
            for i in range(cur_bs):
                save_image(to_uint8(samples[i]), fake_dir / f"{idx:05d}.png")
                idx += 1

    print(f"[Fake] Saved {num_samples} generated images.")

def to_uint8(x: torch.Tensor) -> torch.Tensor:
        # Assumes input x is in range [-1, 1] from Tanh
        # x = (x + 1.0) / 2.0
        # x = x.clamp(0, 1)
        return (x * 255.0).round().to(torch.uint8)
@torch.no_grad()
def recon_cifar10_train(vqvae: torch.nn.Module,
                        dataset: torchvision.datasets.CIFAR10,
                        device: torch.device,
                        recon_dir: Path,
                        batch_size: int = 256,
                        max_images: int = 50000) -> None:
    """
    Reconstruct CIFAR-10 train images using VQ-VAE and save reconstructions as PNGs.
    Output images are saved in [0,1] range as uint8 PNG via torchvision.save_image.

    This function tries the following model APIs in order:
      1) vqvae.encode_to_indices(x) -> [B,H,W] indices
      2) vqvae.encode(x) returning either indices or (z_q, indices, ...) depending on your implementation
      3) vqvae(x) returning recon directly (as common autoencoder forward)
    And then uses:
      - vqvae.decode_from_indices(indices) if indices are available
      - otherwise uses recon output directly
    """
    recon_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(recon_dir.glob("*.png")))
    if existing >= min(50000, max_images):
        print(f"[Recon] Found {existing} images in {recon_dir}. Skip reconstruction.")
        return

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,   # 为了可复现和避免多进程开销；你也可改为 args.num_workers
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    vqvae.eval()
    saved = 0
    global_idx = 0

    for x, _ in loader:
        if saved >= max_images:
            break

        x = x.to(device)  # CIFAR ToTensor(): [0,1]

        # 统一到你生成/解码的规范：很多 VQ-VAE 训练使用 [-1,1]
        # 如果你的模型期望 [0,1]，把下面这行注释掉即可。
        x_in = x * 2.0 - 1.0  # -> [-1,1]

        recon = None

        # ---- Path A: indices -> decode_from_indices ----
        if hasattr(vqvae, "encode_to_indices"):
            inds = vqvae.encode_to_indices(x_in)  # [B,H,W]
            recon = vqvae.decode_from_indices(inds)

        else:
            # 尝试 encode
            if hasattr(vqvae, "encode"):
                enc_out = vqvae.encode(x_in)

                # 情况1：直接返回 indices [B,H,W]
                if torch.is_tensor(enc_out) and enc_out.dim() == 3 and enc_out.dtype in (torch.int32, torch.int64):
                    inds = enc_out.long()
                    recon = vqvae.decode_from_indices(inds)

                # 情况2：返回 tuple/list，尝试从中找 indices
                elif isinstance(enc_out, (tuple, list)):
                    inds = None
                    for t in enc_out:
                        if torch.is_tensor(t) and t.dim() == 3 and t.dtype in (torch.int32, torch.int64):
                            inds = t.long()
                            break
                    if inds is not None and hasattr(vqvae, "decode_from_indices"):
                        recon = vqvae.decode_from_indices(inds)

            # ---- Path B: forward 直接给 recon ----
            if recon is None:
                try:
                    out = vqvae(x_in)
                    # 常见：forward 返回 recon 或 (recon, loss_dict/extra)
                    if torch.is_tensor(out):
                        recon = out
                    elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                        recon = out[0]
                except Exception:
                    recon = None

        if recon is None:
            raise RuntimeError(
                "Cannot reconstruct: model must implement one of:\n"
                "  - encode_to_indices(x) + decode_from_indices(indices)\n"
                "  - encode(x) returning indices + decode_from_indices(indices)\n"
                "  - forward(x) returning recon directly\n"
                "Please adapt recon_cifar10_train() to your VQ-VAE implementation."
            )

        # recon expected in [-1,1] (following your generation path)
        recon = torch.clamp((recon + 1) / 2, 0, 1)

        bsz = recon.size(0)
        for i in range(bsz):
            if saved >= max_images:
                break
            save_image(recon[i], recon_dir / f"{global_idx:05d}.png")
            saved += 1
            global_idx += 1

    print(f"[Recon] Saved {saved} reconstructed images to {recon_dir}.")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = get_args()
    set_seed(args.seed)

    device = get_device(args.device)

    # Logger
    logger = TxtLogger(args.save_dir, "export_fid_images")
    logger.log(f"Args: {vars(args)}")

    # Environment info
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda version: {torch.version.cuda}")
        print(f"current device: {torch.cuda.current_device()}")

    # Load config + build model
    config = load_config(args.filename)
    # model = MSSSIMImprovedVAE(
    #                             in_channels=3,
    #                             latent_dim=128,
    #                             hidden_dims=[64,128,256,512],
    #                             input_size=32,
    #                             attn_resolutions=(4,),
    #                             attn_heads=8,
    #                             groups=16,
    #                             out_act='tanh',
    #                             mssim_window=11,
    #                             mssim_img_range=2.0,   # tanh
    #                             mssim_eps=1e-6,
    #                         ).to(device)

    # model = VampVAE(
    #     in_channels=3,
    #     latent_dim=128,
    #     num_components=100,
    #     img_size=32,
    #     hidden_dims=[64, 128, 256, 512], # ResNet 结构可以更深
    #     device=device
    # ).to(device)
    model = CrossFlowVampVAE(
        in_channels=3,
        latent_dim=128,
        num_components=100,
        img_size=32,
        hidden_dims=[64, 128, 256, 512], # ResNet 结构可以更深
        flow_length=0,                 # 关键：开启 flow
        flow_embed_dim=128,
        flow_heads=4,
        device=device
    ).to(device)
    ckpt = torch.load(args.resume, map_location=device)
    state = ckpt["model"]

    # 1) drop cached window buffers that are lazily built
    for k in list(state.keys()):
        if k.endswith("mssim_loss._window") or k.endswith("mssim._window") or k.endswith("_window"):
            state.pop(k, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    # model.load_state_dict(weights['model'], strict=True)
    model.eval()
    # model = build_model(config, device, args.resume)
    # prior, prior_cfg = build_prior(device, args.prior_ckpt)
    # Determine latent_dim robustly
    # latent_dim = getattr(model, "embedding_dim", None)
    # if latent_dim is None:
    #     latent_dim = config["model_params"].get("embedding_dim", None)
    # if latent_dim is None:
    #     raise ValueError("Cannot determine latent_dim. Expected model.latent_dim or config['model_params']['latent_dim'].")

    # Prepare dirs
    real_dir = Path(args.data_dir) / args.real_dirname
    fake_dir = Path(args.data_dir) / args.fake_dirname
    recon_dir = Path(args.data_dir) / args.recon_dirname
    # Export real + generate fake
    trainset = get_cifar10_train(args.data_dir)
    export_real_images(trainset, real_dir)

    if args.do_recon:
        recon_cifar10_train(
            vqvae=model,
            dataset=trainset,
            device=device,
            recon_dir=recon_dir,
            batch_size=args.recon_batch_size,
            max_images=50000
        )

        print("\nRun Recon FID with official pytorch-fid:")
        print(f"python -m pytorch_fid {real_dir} {recon_dir}")

#     generate_fake_images_vqvae_with_prior(
#     vqvae=model,
#     prior=prior,
#     prior_cfg=prior_cfg,
#     device=device,
#     fake_dir=fake_dir,
#     num_samples=args.num_samples,
#     batch_size=args.gen_batch_size,
#     temperature=args.temperature,
#     top_k=args.top_k
# )
    generate_fake_images(model, device, fake_dir, args.num_samples, args.gen_batch_size, 128)
    #generate_smallFig(real_dir, args)
    generate_smallFig(fake_dir, args)
    # Print official FID command
    print("\nRun FID with official pytorch-fid:")
    print(f"python -m pytorch_fid {real_dir} {fake_dir}")

def generate_smallFig(dir, args):
    img_tensor = []
    for file in os.listdir(dir):
        if len(img_tensor)<16:
            
            if file.endswith('.png'):
                img = Image.open(os.path.join(dir, file)).convert('RGB')
                img = transforms.ToTensor()(img)
                img_tensor.append(img)
    img_tensor = torch.stack(img_tensor)  # Shape: (N, C, H, W)
    grid = make_grid((img_tensor*255).to(torch.uint8), nrow=4, padding=2)
    save_image(grid, os.path.join(args.save_dir, f"gen.png"), nrow=4)

if __name__ == "__main__":
    main()
