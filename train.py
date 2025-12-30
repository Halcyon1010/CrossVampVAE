import os
os.environ["TORCH_HOME"] = "/mnt/afs/250010063/torch_cache"
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from models.vamp_flow import CrossFlowVampVAE
# ==============================================================================
# 导入模型 (请确保文件路径正确)
# ==============================================================================


# ==============================================================================
# 配置参数
# ==============================================================================
def get_args():
    parser = argparse.ArgumentParser(description="Trainer for ResNet-Attention VampVAE")
    
    # 实验基础配置
    parser.add_argument("--exp_name", type=str, default="ResVampVAE_CIFAR10", help="实验名称")
    parser.add_argument("--data_dir", type=str, default="/mnt/afs/250010063/DL4/data", help="数据集路径")
    parser.add_argument("--save_dir", type=str, default="/mnt/afs/250010063/DL4/result", help="结果保存路径")
    parser.add_argument("--resume", type=str, default=r"", help="断点续训路径 (last.pth)")
    parser.add_argument("--seed", type=int, default=42)
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1000, help="总 Epoch 数")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    # 模型参数 (必须与定义一致)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_components", type=int, default=100, help="VampPrior 伪输入数量")
    parser.add_argument("--img_size", type=int, default=32)
    
    # 关键策略
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Beta 预热的 Epoch 数")
    parser.add_argument("--beta_max", type=float, default=0.2, help="KL 权重最大值")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    
    # FID 配置
    parser.add_argument("--fid_every", type=int, default=10, help="每多少 Epoch 测一次 FID")
    parser.add_argument("--fid_samples", type=int, default=2000, help="测试 FID 时生成的样本数")
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ==============================================================================
# 辅助函数
# ==============================================================================

def visualize_pseudo_inputs(model, save_path, epoch):
    """
    可视化 VampPrior 学到的 K 个伪输入。
    这些图像代表了 Latent Space 中的 K 个‘锚点’。
    """
    model.eval()
    with torch.no_grad():
        # 获取伪输入 [K, C, H, W]
        pseudo_imgs = model.embed_pseudo(model.pseudo_id).view(-1, *model.pseudo_img_shape)
        
        # 反归一化: [-1, 1] -> [0, 1]
        vis_imgs = (pseudo_imgs + 1) / 2.0
        vis_imgs = torch.clamp(vis_imgs, 0, 1)
        
        # 只取前 64 个展示
        n_show = min(64, model.num_components)
        grid = make_grid(vis_imgs[:n_show], nrow=8, padding=2)
        save_image((grid*255).to(torch.uint8), os.path.join(save_path, f"pseudo_epoch_{epoch}.png"))

@torch.no_grad()
def calculate_fid(model, dataloader, device, num_samples=2000):
    """
    计算 FID 分数
    """
    model.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    
    print(f"--- Calculating FID ({num_samples} samples) ---")
    
    # 1. 真实图片统计 (Real)
    # 遍历一部分验证集即可
    count = 0
    for x, _ in dataloader:
        x = x.to(device)
        # Tanh [-1, 1] -> [0, 1]
        x = (x + 1) / 2.0
        x = x.clamp(0, 1)
        fid.update((x*255).to(torch.uint8), real=True)
        count += x.size(0)
        if count >= num_samples: break
            
    # 2. 生成图片统计 (Fake)
    # 使用 VampVAE 特有的混合采样
    remaining = num_samples
    while remaining > 0:
        batch = min(100, remaining)
        # 注意: 这里的 sample 签名是 (num_samples, current_device)
        samples = model.sample(num_samples=batch, current_device=device)
        
        # Tanh [-1, 1] -> [0, 1]
        samples = (samples + 1) / 2.0
        samples = samples.clamp(0, 1)
        
        fid.update((samples*255).to(torch.uint8), real=False)
        remaining -= batch
        
    try:
        score = fid.compute().item()
        fid.reset()
        return score
    except Exception as e:
        print(f"FID Error: {e}")
        return float('inf')

@torch.no_grad()
def validate_loss(model, dataloader, device, beta):
    model.eval()
    total_loss = 0
    for x, _ in dataloader:
        x = x.to(device)
        results = model(x)
        # 验证时也传入当前的 beta
        loss_dict = model.loss_function(*results, M_N=beta)
        total_loss += loss_dict['loss'].item()
    return total_loss / len(dataloader)

# ==============================================================================
# 主训练循环
# ==============================================================================
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径设置
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    sample_dir = os.path.join(exp_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    print(f"🚀 Experiment: {args.exp_name}")
    print(f"📂 Saving to: {exp_dir}")

    # 数据集 (归一化到 -1, 1)
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    
    # 3. 归一化 (保持不变)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
    
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    
    model = CrossFlowVampVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_components=args.num_components,
        img_size=args.img_size,
        hidden_dims=[64, 128, 256, 512], # ResNet 结构可以更深
        flow_length=8,                 # 关键：开启 flow
        flow_embed_dim=args.latent_dim,
        flow_heads=4,
        device=device
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # History 记录
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], 
        "recon": [], "kld": [], "fid": [], "beta": [],'lpips':[]
    }
    
    
    start_epoch = 0
    best_fid = float('inf')
    
    # 断点续训
    if args.resume and os.path.exists(args.resume):
        print(f"♻️ Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_fid = ckpt.get('best_fid', float('inf'))
        if 'history' in ckpt: history = ckpt['history']

    # --- Loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}")
        
        epoch_loss = 0
        epoch_pixel = 0    # 改名: 像素损失
        epoch_lpips = 0    # 新增: 感知损失
        epoch_kld_opt = 0  # 实际优化的 KL
        epoch_kld_raw = 0  # 真实的 KL (观察是否有 posterior collapse)
        
        # 计算当前 Epoch 的 Beta (Warmup)
        # 简单的线性 Warmup: epoch 0 -> 0, epoch warmup -> beta_max
        beta_progress = min(1.0, epoch / max(1, args.warmup_epochs))
        current_beta = args.beta_max * beta_progress
        
        for x, _ in progress_bar:
            x = x.to(device)
            optimizer.zero_grad()
            
            #with torch.amp.autocast(device_type='cuda'):
            results = model(x)
            # results: [recons, input, mu, log_var, z]
            # 注意：loss_function 参数名为 M_N 表示 beta
            loss_dict = model.loss_function(*results, M_N=current_beta)
            loss = loss_dict['loss']
            # optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # scaler.step(optimizer)
            # scaler.update()
            
            # --- 更新统计数据 ---
            # loss_dict 的 key 必须与 loss_function 返回的一致
            epoch_loss += loss.item()
            epoch_pixel += loss_dict['Reconstruction_Loss'].item()
            epoch_lpips += loss_dict['LPIPS_Loss'].item()
            epoch_kld_opt += loss_dict['KLD_Optim'].item()
            epoch_kld_raw += loss_dict['KLD_Raw'].item()
            
            # --- 进度条打印 ---
            progress_bar.set_postfix({
                'L': f"{loss.item():.2f}",       # Total Loss
                'Pix': f"{loss_dict['Reconstruction_Loss'].item():.1f}", # Pixel L1
                'LPIPS': f"{loss_dict['LPIPS_Loss'].item():.3f}",        # 重点监控!
                'KL_R': f"{loss_dict['KLD_Raw'].item():.1f}",            # Raw KL
                'KL_O': f"{loss_dict['KLD_Optim'].item():.1f}"           # Optimized KL
            })

        # 计算平均值
        avg_loss = epoch_loss / len(train_loader)
        avg_pixel = epoch_pixel / len(train_loader)
        avg_lpips = epoch_lpips / len(train_loader)
        avg_kld_opt = epoch_kld_opt / len(train_loader)
        avg_kld_raw = epoch_kld_raw / len(train_loader)
        
        
        # 打印 Epoch 总结
        
        # 验证集 Loss
        val_loss = validate_loss(model, val_loader, device, current_beta)
        
        # FID 计算
        fid_score = float('nan')
        if (epoch + 1) % args.fid_every == 0 or (epoch + 1) == args.epochs:
            fid_score = calculate_fid(model, val_loader, device, num_samples=args.fid_samples)
            if fid_score < best_fid:
                best_fid = fid_score
                save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_fid': best_fid,
                'history': history,
            }
                torch.save(save_dict, os.path.join(exp_dir, "best_fid.pth"))
                print(f"🔥 New Best FID: {best_fid:.2f}")

        scheduler.step()
        
        # 更新 History (请确保你初始化 history 字典时包含了这些 key)
        # 建议在初始化 history 时加上: "lpips": [], "kld_raw": []
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['recon'].append(avg_pixel) # 记录像素损失
        history['lpips'].append(avg_lpips) # 记录 LPIPS
        history['kld'].append(avg_kld_raw) # 记录原始 KL 更能反映模型状态
        history['fid'].append(fid_score)
        history['beta'].append(current_beta)
        print(f"Ep {epoch+1} | Loss: {avg_loss:.2f} | LPIPS: {avg_lpips:.3f} | KL_Raw: {avg_kld_raw:.1f} | FID: {fid_score:.2f}")

        # 保存 CSV
        pd.DataFrame(history).to_csv(os.path.join(exp_dir, "history.csv"), index=False)
        
        # 可视化与保存
        if (epoch + 1) % 5 == 0:
            # 1. 采样图片
            model.eval()
            with torch.no_grad():
                samples = model.sample(64, device)
                samples = (samples + 1) / 2.0 # 反归一化
                samples = samples.clamp(0, 1)
                grid = make_grid((samples*255).to(torch.uint8), nrow=8, padding=2)
                save_image(grid, os.path.join(sample_dir, f"gen_epoch_{epoch}.png"), nrow=8)
            
            # 2. 伪输入可视化 (查看 VampPrior 的锚点)
            visualize_pseudo_inputs(model, sample_dir, epoch)

        # 保存 Last Checkpoint
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_fid': best_fid,
            'history': history,
        }
        torch.save(save_dict, os.path.join(exp_dir, "last.pth"))
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.2f} | Val: {val_loss:.2f} | FID: {fid_score:.2f}")

if __name__ == "__main__":
    args = get_args()
    train(args)