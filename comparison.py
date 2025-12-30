#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Cache discovery & loading
# =========================================================
def find_cache_files(cache_dir: str, pattern: str = "cache_*.csv"):
    paths = sorted(glob.glob(os.path.join(cache_dir, pattern)))
    return paths


def infer_tag_from_cache_name(path: str) -> str:
    """
    cache_<tag>.csv -> <tag>
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    if name.startswith("cache_"):
        return name[len("cache_"):]
    return name


def load_cache_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize possible column name variants
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["lap_var", "laplacian_var", "laplacian_variance"]:
            col_map[c] = "lap_var"
        elif cl in ["tenengrad", "tenen", "tenengrad_score"]:
            col_map[c] = "tenengrad"
        elif cl in ["hf_ratio", "high_freq_ratio", "high_frequency_ratio", "hf"]:
            col_map[c] = "hf_ratio"

    df = df.rename(columns=col_map)

    required = ["lap_var", "tenengrad", "hf_ratio"]
    missing = [m for m in required if m not in df.columns]
    if missing:
        raise ValueError(
            f"Cache file {path} missing columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep only numeric metric columns
    out = df[required].copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna()

    return out


# =========================================================
# Summary
# =========================================================
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize numeric columns only (same as your style).
    Output: rows=metrics, cols=stats
    """
    num_df = df.select_dtypes(include=["number"])
    return pd.DataFrame({
        "mean": num_df.mean(),
        "median": num_df.median(),
        "p90": num_df.quantile(0.9),
    })


# =========================================================
# Plotting
# =========================================================
def sample_for_plot(x: np.ndarray, max_points: int, seed: int):
    """
    Only used for plotting (speed); does NOT affect summary.
    """
    if max_points <= 0 or x.shape[0] <= max_points:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_points, replace=False)
    return x[idx]


def plot_box_from_cache(
    dfs, labels, out_dir,
    showfliers=False,
    max_points=20000,
    seed=42,
):
    """
    For each metric:
      - one boxplot figure
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["lap_var", "tenengrad", "hf_ratio"]

    for m in metrics:
        # ---------- Boxplot ----------
        plt.figure(figsize=(7, 4.5))
        data_for_box = []
        for df in dfs:
            x = df[m].to_numpy()
            x = x[np.isfinite(x)]
            x = sample_for_plot(x, max_points=max_points, seed=seed)
            data_for_box.append(x)

        plt.boxplot(data_for_box, labels=labels, showfliers=showfliers, whis=(5, 95))
        plt.ylabel(m)
        plt.title(f"{m} boxplot (whis=5-95%, sampled <= {max_points}/model)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{m}_box_compare.png"), dpi=300)
        plt.close()

# =========================================================

def plot_delta_to_real_multi_models(
    tag_to_df: dict,
    labels: list,
    metric: str,
    out_dir: str,
    max_points: int = 50000,
    seed: int = 42,
):
    """
    One figure per metric.
    Y-axis: metric(model) - mean(metric(real))
    X-axis: model names (excluding 'real')
    """
    if "real" not in tag_to_df:
        raise KeyError("tag_to_df must contain 'real'")

    os.makedirs(out_dir, exist_ok=True)

    df_real = tag_to_df["real"]
    real_mean = df_real[metric].mean()

    data = []
    model_names = []

    for lb in labels:
        if lb == "real":
            continue
        df = tag_to_df[lb]
        x = df[metric].to_numpy()
        x = x[np.isfinite(x)]
        x = sample_for_plot(x, max_points=max_points, seed=seed)
        delta = x - real_mean

        data.append(delta)
        model_names.append(lb)

    plt.figure(figsize=(1.2 * len(model_names) + 2, 4.8))

    parts = plt.violinplot(
        data,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )

    # 美化
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.6)

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)

    plt.xticks(
        ticks=range(1, len(model_names) + 1),
        labels=model_names,
        rotation=25,
        ha="right",
    )
    plt.ylabel(f"{metric} − mean(real)")
    plt.title(f"Δ-to-Real Distribution ({metric})")

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{metric}_delta_violin.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[OK] Saved {save_path}")

# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default=r'/mnt/afs/250010063/DL4/result/single_image_quality',
                    help="Directory containing cache_*.csv (already computed 50k metrics)")
    ap.add_argument("--out_dir", type=str, default=r'/mnt/afs/250010063/DL4/result/compare',
                    help="Directory to save summary and plots")
    ap.add_argument("--pattern", type=str, default="cache_*.csv")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--density", action="store_true", help="Use density in histogram")
    ap.add_argument("--max_points", type=int, default=50000,
                    help="Max samples per model for plotting (speed). 0 = no sampling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_order", type=str, default="VAE,BetaVAE,IWAE,MSSIM-VAE,CrossVampVAE,real",
                    help="Optional comma-separated order, e.g. 'real,flow,noflow,VAE,BetaVAE'")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cache_paths = find_cache_files(args.cache_dir, args.pattern)
    if len(cache_paths) == 0:
        raise FileNotFoundError(f"No cache files found in {args.cache_dir} with pattern {args.pattern}")

    # Load all caches
    tag_to_df = {}
    for p in cache_paths:
        tag = infer_tag_from_cache_name(p)
        df = load_cache_csv(p)
        tag_to_df[tag] = df
        print(f"[OK] Loaded {tag}: {df.shape[0]} rows from {p}")

    # Order
    if args.model_order.strip():
        labels = [x.strip() for x in args.model_order.split(",") if x.strip()]
        # keep only existing
        labels = [x for x in labels if x in tag_to_df]
        # append any remaining not listed
        rest = [k for k in sorted(tag_to_df.keys()) if k not in labels]
        labels = labels + rest
    else:
        labels = sorted(tag_to_df.keys())

    dfs = [tag_to_df[k] for k in labels]

    # Summary table (all models)
    stats_list = []
    for lb, df in zip(labels, dfs):
        s = summarize(df)  # rows=metric, cols=mean/median/p90
        stats_list.append(s)

    stat = pd.concat(stats_list, axis=1, keys=labels)
    stat_path = os.path.join(args.out_dir, "summary_statistics_all_models.csv")
    stat.to_csv(stat_path)
    print(f"[OK] Saved summary statistics: {stat_path}")

    
    # Plots
    plot_hist_box_from_cache(
        dfs=dfs,
        labels=labels,
        out_dir=args.out_dir,
        bins=args.bins,
        density=args.density,
        showfliers=False,
        max_points=args.max_points,
        seed=args.seed,
    )
    print(f"[OK] Saved hist/box comparison plots to: {args.out_dir}")

    # =========================================================
    # Delta-to-Real plots (Real as reference)
    # =========================================================
    if "real" not in tag_to_df:
        raise KeyError(
            "Could not find 'real' in cache tags. "
            "Please ensure you have cache_real.csv (tag should be 'real'). "
            f"Found tags: {sorted(tag_to_df.keys())}"
        )

    metrics = ["lap_var"]

    for m in metrics:
        plot_delta_to_real_multi_models(
            tag_to_df=tag_to_df,
            labels=labels,
            metric=m,
            out_dir=delta_out_dir,
            max_points=args.max_points,
            seed=args.seed,
        )

if __name__ == "__main__":
    main()
