"""
EOS-04 SAR Flood Detection — Stage 7: Visualization & Report
=============================================================
Generates:
  1. Backscatter distribution (before vs during, HH + HV)
  2. Otsu threshold histogram
  3. Training loss / IoU curves
  4. Final flood map overlay (RGB composite + flood mask)
  5. Change detection comparison panel
  6. Summary report (text)

Usage:
    python 07_visualize.py --out_dir outputs/figures
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import rasterio
from rasterio.plot import show
from skimage.filters import threshold_otsu
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROC  = "data/processed"
FEAT  = "data/features"
FIGS  = "outputs/figures"


def read(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile


def percentile_stretch(arr, low=2, high=98):
    lo, hi = np.nanpercentile(arr, [low, high])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


# ── 1. Backscatter distributions ─────────────────────────────────────────────
def plot_backscatter_distributions(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EOS-04 SAR Backscatter Distribution (σ⁰ dB)\n"
                 "Before: 20-Jan-2023 | During flood: 26-Jul-2023",
                 fontsize=12)

    colors = {"before": "#3B8BD4", "during": "#E85D24"}
    for ax, pol in zip(axes, ["HH", "HV"]):
        for date in ["before", "during"]:
            arr, _ = read(f"{PROC}/{date}_sigma0_{pol}_dB.tif")
            vals = arr[np.isfinite(arr)].ravel()
            vals = vals[::10]  # subsample for speed
            ax.hist(vals, bins=120, alpha=0.55, density=True,
                    color=colors[date], label=date.capitalize())

            # Otsu threshold on during image
            if date == "during":
                t = threshold_otsu(vals)
                ax.axvline(t, color="red", linestyle="--", linewidth=1.2,
                           label=f"Otsu = {t:.1f} dB")

        ax.set_title(f"Polarisation: {pol}")
        ax.set_xlabel("σ⁰ (dB)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "backscatter_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path}")


# ── 2. Change detection panel ─────────────────────────────────────────────────
def plot_change_detection(out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EOS-04 MRS SAR — Flood Change Detection\n"
                 "Path: UTM Zone 44N | Incidence 37.9° | HH & HV Polarisations",
                 fontsize=12)

    keys = [
        ("before_sigma0_HH_dB", "Before HH σ⁰ (dB)"),
        ("during_sigma0_HH_dB", "During HH σ⁰ (dB)"),
        ("change_HH_dB",        "Δσ⁰ HH (during − before)"),
        ("before_sigma0_HV_dB", "Before HV σ⁰ (dB)"),
        ("during_sigma0_HV_dB", "During HV σ⁰ (dB)"),
        ("change_HV_dB",        "Δσ⁰ HV (during − before)"),
    ]

    for ax, (key, title) in zip(axes.ravel(), keys):
        arr, _ = read(f"{PROC}/{key}.tif")
        if "change" in key.lower():
            # Diverging colormap centred at 0 for change
            vmax = float(np.nanpercentile(np.abs(arr), 98))
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            arr_s = percentile_stretch(arr)
            im = ax.imshow(arr_s, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(out_dir, "change_detection_panel.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path}")


# ── 3. Flood map overlay ──────────────────────────────────────────────────────
def plot_flood_map_overlay(out_dir, flood_path="outputs/flood_maps/flood_ensemble.tif"):
    if not os.path.exists(flood_path):
        log.warning(f"Flood map not found: {flood_path}")
        return

    # Use during_HH as background
    bg, _ = read(f"{PROC}/during_sigma0_HH_dB.tif")
    fl, _ = read(flood_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("EOS-04 Flood Extent Map — 26-Jul-2023\n"
                 "Scene centre: 17.32°N 80.99°E | UTM Zone 44N", fontsize=12)

    # Left: backscatter only
    axes[0].imshow(percentile_stretch(bg), cmap="gray", origin="upper")
    axes[0].set_title("σ⁰_HH During Flood (26-Jul-2023)")
    axes[0].axis("off")

    # Right: backscatter + flood overlay
    axes[1].imshow(percentile_stretch(bg), cmap="gray", origin="upper")
    flood_rgba = np.zeros((*fl.shape, 4), dtype=np.float32)
    flood_rgba[fl == 1] = [0.0, 0.4, 1.0, 0.6]   # semi-transparent blue for flood
    axes[1].imshow(flood_rgba, origin="upper")
    axes[1].set_title("Flood Inundation Extent (ensemble)")
    axes[1].axis("off")

    # Legend
    patch = mpatches.Patch(color=(0, 0.4, 1.0), alpha=0.7, label="Flood inundation")
    axes[1].legend(handles=[patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "flood_map_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path}")


# ── 4. Training curves ────────────────────────────────────────────────────────
def plot_training_curves(out_dir, history_json="outputs/history.json"):
    if not os.path.exists(history_json):
        log.warning(f"history.json not found at {history_json}")
        return

    with open(history_json) as f:
        h = json.load(f)

    epochs    = [e["epoch"] for e in h]
    t_loss    = [e["train_loss"] for e in h]
    v_loss    = [e["val_loss"] for e in h]
    iou       = [e["iou"] for e in h]
    f1        = [e["f1"] for e in h]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DL Training History — EOS-04 Flood Detection")

    axes[0].plot(epochs, t_loss, label="Train loss", color="#E85D24")
    axes[0].plot(epochs, v_loss, label="Val loss",   color="#3B8BD4")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (Dice+BCE)")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, iou, label="Val IoU", color="#1D9E75", linewidth=2)
    axes[1].plot(epochs, f1,  label="Val F1",  color="#7F77DD", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path}")


# ── 5. Summary text report ────────────────────────────────────────────────────
def write_text_report(out_dir, flood_report_path="outputs/flood_maps/flood_report.json"):
    lines = [
        "=" * 70,
        "EOS-04 SAR FLOOD DETECTION REPORT",
        "=" * 70,
        "",
        "MISSION METADATA",
        "-" * 40,
        "Satellite      : EOS-04 (RISAT-1A)",
        "Sensor         : SAR C-band, MRS mode",
        "Polarisations  : HH + HV (dual-pol)",
        "Pixel spacing  : 18 m × 18 m (UTM Zone 44N, WGS84)",
        "Incidence angle: 37.91°",
        "DEM correction : Applied (Copernicus 30 m)",
        "",
        "ACQUISITIONS",
        "-" * 40,
        "Before  : 20-Jan-2023 00:32:06 UTC  (Orbit 5141, Cycle 18)",
        "During  : 26-Jul-2023 00:31:51 UTC  (Orbit 7968, Cycle 29)",
        "Scene   : 17.32°N 80.99°E | ~190×188 km swath",
        "",
        "METHODOLOGY",
        "-" * 40,
        "1. Preprocessing  : DN→σ⁰ calibration, noise bias removal,",
        "                    Refined Lee speckle filter (7×7, ENL=2)",
        "2. Features       : σ⁰_HH/HV, CR, RFDI, Δσ⁰, GLCM texture",
        "3. Classical      : Otsu auto-threshold on σ⁰_HH",
        "4. ML classifiers : Random Forest, XGBoost (GPU), SVM",
        "5. Deep learning  : U-Net / Attention U-Net / SegFormer / DeepLabV3+",
        "                    DGX DDP (8×A100), FP16, patch 512×512",
        "                    Loss: Dice + BCE | LR: cosine with warmup",
        "6. Post-process   : Morphological open/close, sieve filter",
        "7. Ensemble       : Weighted vote (DL×2 + Otsu×1)",
        "",
    ]

    if os.path.exists(flood_report_path):
        with open(flood_report_path) as f:
            rep = json.load(f)
        s = rep.get("flood_stats", {})
        lines += [
            "FLOOD EXTENT RESULTS",
            "-" * 40,
            f"Model          : {rep.get('model', 'N/A')}",
            f"Flood pixels   : {s.get('flood_pixels', 'N/A'):,}",
            f"Flood area     : {s.get('flood_area_ha', 'N/A'):,.1f} ha",
            f"Scene coverage : {s.get('flood_pct', 'N/A'):.2f}%",
            "",
        ]
        if "overall_accuracy" in s:
            lines += [
                "ACCURACY (vs reference)",
                "-" * 40,
                f"Overall Accuracy : {s['overall_accuracy']:.4f}",
                f"Flood IoU        : {s['iou_flood']:.4f}",
                f"Flood F1         : {s['f1_flood']:.4f}",
                f"Precision        : {s['precision_flood']:.4f}",
                f"Recall           : {s['recall_flood']:.4f}",
                f"Kappa            : {s['kappa']:.4f}",
                "",
            ]

    lines += ["=" * 70]
    report_str = "\n".join(lines)

    path = os.path.join(out_dir, "flood_detection_report.txt")
    with open(path, "w") as f:
        f.write(report_str)
    print(report_str)
    log.info(f"Report saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    plot_backscatter_distributions(args.out_dir)
    plot_change_detection(args.out_dir)
    plot_flood_map_overlay(args.out_dir, args.flood_path)
    plot_training_curves(args.out_dir, args.history_json)
    write_text_report(args.out_dir, args.flood_report)

    log.info(f"All figures saved to {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("EOS-04 Visualization")
    p.add_argument("--out_dir",      default=FIGS)
    p.add_argument("--flood_path",   default="outputs/flood_maps/flood_ensemble.tif")
    p.add_argument("--history_json", default="outputs/history.json")
    p.add_argument("--flood_report", default="outputs/flood_maps/flood_report.json")
    args = p.parse_args()
    main(args)
