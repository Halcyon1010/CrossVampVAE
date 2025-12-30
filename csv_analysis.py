#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# =========================================================
# Basic utils
# =========================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_images(root):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root, f"*{ext}")))
    return sorted(paths)


def load_gray(path):
    img = Image.open(path).convert("L")
    return np.asarray(img).astype(np.float32) / 255.0


def conv2d(x, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    xpad = np.pad(x, ((ph, ph), (pw, pw)), mode="reflect")
    H, W = x.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(xpad[i:i+kh, j:j+kw] * k)
    return out


# =========================================================
# Metrics
# =========================================================
def laplacian_variance(gray):
    k = np.array([[0,  1, 0],
                  [1, -4, 1],
                  [0,  1, 0]], dtype=np.float32)
    y = conv2d(gray, k)
    return float(y.var())


def tenengrad(gray):
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]], dtype=np.float32)
    gx = conv2d(gray, kx)
    gy = conv2d(gray, ky)
    return float((gx**2 + gy**2).mean())


def fft_highfreq_ratio(gray, r0=0.25):
    H, W = gray.shape
    F = np.fft.fftshift(np.fft.fft2(gray))
    P = np.abs(F) ** 2

    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    r_thr = r0 * (min(H, W) / 2.0)

    low = P[rr <= r_thr].sum()
    high = P[rr >  r_thr].sum()
    return float(high / (low + high + 1e-12))


# =========================================================
# Evaluation
# =========================================================
def eval_folder(folder):
    paths = list_images(folder)
    records = []

    for p in paths:
        g = load_gray(p)
        records.append({
            "lap_var": laplacian_variance(g),
            "tenengrad": tenengrad(g),
            "hf_ratio": fft_highfreq_ratio(g),
        })

    return pd.DataFrame(records)


def summarize(df):
    """
    Summarize numeric columns only.
    """
    num_df = df.select_dtypes(include=["number"])

    return pd.DataFrame({
        "mean": num_df.mean(),
        "median": num_df.median(),
        "p90": num_df.quantile(0.9),
    })


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gen", type=str, default=r'/mnt/afs/250010063/DL4/data/fake_cifar')
    ap.add_argument("--out_dir", type=str, default=r"/mnt/afs/250010063/DL4/result/single_image_quality")
    ap.add_argument("--tag", type=str, default=r"VAE")
    args = ap.parse_args()

    # ---------- Per-folder evaluation with cache ----------

    df_noflow = eval_folder_with_cache(
        img_dir=args.gen,
        out_dir=args.out_dir,
        tag=args.tag
    )

    # ---------- Statistics ----------

    stat_noflow = summarize(df_noflow)

    stat = pd.concat(
        [stat_noflow],
        axis=1,
        keys=[args.tag]
    )

    stat_path = os.path.join(args.out_dir, args.tag+"_statistics.csv")
    stat.to_csv(stat_path)
    print(f"Saved statistics to {stat_path}")


def eval_folder_with_cache(img_dir, out_dir, tag):
    """
    Evaluate a folder with caching.
    If cache exists, load it; otherwise compute and save.
    """
    os.makedirs(out_dir, exist_ok=True)

    cache_path = os.path.join(out_dir, f"cache_{tag}.csv")

    if os.path.exists(cache_path):
        print(f"[Cache] Loading cached results for {tag} from {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        print(f"[Compute] Evaluating images in {img_dir}")
        df = eval_folder(img_dir)
        df.to_csv(cache_path, index=False)
        print(f"[Cache] Saved cache to {cache_path}")

    return df

if __name__ == "__main__":
    main()
