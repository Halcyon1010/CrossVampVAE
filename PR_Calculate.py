# precision_recall_pca.py
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.decomposition import PCA


# -------------------------
# Dataset
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, folder: str):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in folder: {folder}")

        # Inception-V3 eval protocol (resize to 299)
        self.tf = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


# -------------------------
# Inception Feature Extractor (robust for torchvision versions)
# -------------------------
class InceptionFeature(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
            transform_input=False,
        )
        net.fc = nn.Identity()
        self.net = net.eval()

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, tuple):  # defensive
            out = out[0]
        return out


@torch.no_grad()
def extract_features(folder: str, device: torch.device, batch_size=64, num_workers=4) -> np.ndarray:
    ds = ImageFolderDataset(folder)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model = InceptionFeature().to(device)
    feats = []

    for x in tqdm(loader, desc=f"Extracting features from {folder}"):
        x = x.to(device, non_blocking=True)
        f = model(x)  # [B,2048]
        feats.append(f.cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    return feats


def chunked_1nn(tree: cKDTree, queries: np.ndarray, chunk: int = 5000, tag: str = ""):
    dists_all, idx_all = [], []
    n = queries.shape[0]
    n_chunks = (n + chunk - 1) // chunk
    for ci in range(n_chunks):
        s = ci * chunk
        e = min((ci + 1) * chunk, n)
        dists, idx = tree.query(queries[s:e], k=1)
        dists_all.append(dists)
        idx_all.append(idx)
        print(f"[PR]{tag} 1NN chunk {ci+1}/{n_chunks} ({e}/{n})")
    return np.concatenate(dists_all), np.concatenate(idx_all)


def compute_precision_recall(real_feats: np.ndarray,
                             fake_feats: np.ndarray,
                             k: int = 3,
                             chunk: int = 5000) -> tuple[float, float]:
    """
    real_feats: [Nr, D]
    fake_feats: [Ng, D]
    k: kNN radius uses (k+1) because nearest neighbor includes itself
    """
    print("[PR] Build KDTree (real)...")
    real_tree = cKDTree(real_feats)
    print("[PR] Build KDTree (fake)...")
    fake_tree = cKDTree(fake_feats)

    print(f"[PR] Compute radii with k={k} ...")
    # distance to k-th neighbor (exclude itself by using k+1, take last)
    real_r = real_tree.query(real_feats, k=k + 1)[0][:, -1]
    fake_r = fake_tree.query(fake_feats, k=k + 1)[0][:, -1]

    # Precision: fake in real manifold
    print("[PR] Precision: fake -> real 1NN ...")
    dist_fr, idx_fr = chunked_1nn(real_tree, fake_feats, chunk=chunk, tag="(P)")
    precision = float(np.mean(dist_fr <= real_r[idx_fr]))

    # Recall: real covered by fake manifold
    print("[PR] Recall: real -> fake 1NN ...")
    dist_rf, idx_rf = chunked_1nn(fake_tree, real_feats, chunk=chunk, tag="(R)")
    recall = float(np.mean(dist_rf <= fake_r[idx_rf]))

    return precision, recall


def main(args):
    device = torch.device(args.device)

    # 1) extract 2048-d inception features
    real_feats = extract_features(args.real_dir, device,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
    fake_feats = extract_features(args.fake_dir, device,
                                  batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"Real feats: {real_feats.shape}")
    print(f"Fake feats: {fake_feats.shape}")

    # 2) PCA (optional but recommended)
    if args.pca_dim > 0:
        pca_dim = args.pca_dim
        if pca_dim >= real_feats.shape[1]:
            raise ValueError(f"pca_dim ({pca_dim}) must be < feature dim ({real_feats.shape[1]})")

        print(f"[PCA] Fitting PCA on real feats -> dim={pca_dim} ...")
        pca = PCA(n_components=pca_dim, random_state=0, svd_solver="randomized")
        real_feats = pca.fit_transform(real_feats)
        fake_feats = pca.transform(fake_feats)
        print(f"[PCA] Done. New shapes: real {real_feats.shape}, fake {fake_feats.shape}")

    # 3) compute precision/recall
    precision, recall = compute_precision_recall(
        real_feats, fake_feats, k=args.k, chunk=args.chunk
    )
    print(f"\nPrecision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, default=r"/mnt/afs/250010063/DL4/data/real_cifar")
    parser.add_argument("--fake_dir", type=str, default=r"/mnt/afs/250010063/DL4/data/fake_cifar_v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)

    # PR params
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=5000)

    # PCA params (set 0 to disable)
    parser.add_argument("--pca_dim", type=int, default=256)

    args = parser.parse_args()
    main(args)
