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
from utils import TxtLogger
from models.vamp_flow import CrossFlowVampVAE
import torch.nn as nn
import PIL.Image as Image
from models.vamp_flow import CrossFlowVampVAE
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
            
            for i in range(cur_bs):
                save_image(to_uint8(samples[i]), fake_dir / f"{idx:05d}.png")
                idx += 1

    print(f"[Fake] Saved {num_samples} generated images.")

def to_uint8(x: torch.Tensor) -> torch.Tensor:
        # Assumes input x is in range [-1, 1] from Tanh
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        return (x * 255.0).round().to(torch.uint8)

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

    model = CrossFlowVampVAE(
        in_channels=3,
        latent_dim=128,
        num_components=100,
        img_size=32,
        hidden_dims=[64, 128, 256, 512], # ResNet 结构可以更深
        flow_length=8,                 # 关键：开启 flow
        flow_embed_dim=128,
        flow_heads=4,
        device=device
    ).to(device)
    ckpt = torch.load(args.resume, map_location=device)
    state = ckpt["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    
    # Prepare dirs
    real_dir = Path(args.data_dir) / args.real_dirname
    fake_dir = Path(args.data_dir) / args.fake_dirname
    recon_dir = Path(args.data_dir) / args.recon_dirname
    # Export real + generate fake
    trainset = get_cifar10_train(args.data_dir)
    export_real_images(trainset, real_dir)

    generate_fake_images(model, device, fake_dir, args.num_samples, args.gen_batch_size, 128)
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
