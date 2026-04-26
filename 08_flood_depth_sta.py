"""
EOS-04 SAR Flood Detection — STA Flood Depth Estimation
========================================================
Estimates flood depth using the STA (SAR Topographic Analysis) technique.

Method:
  1. Reproject Copernicus 30m DEM to match flood map CRS (UTM Zone 44N)
  2. Compute Water Surface Elevation (WSE) per connected flood region
  3. Depth = WSE - terrain elevation
  4. Classify into 14 depth categories (0 to >6.5m)
  5. Generate coloured depth map figure

Inputs:
  data/dem_raw/output_hh.tif          Copernicus GLO-30 DEM (WGS84)
  outputs/flood_maps/flood_final.tif  Final flood mask (0=non-flood, 1=flood, 255=nodata)

Outputs:
  outputs/flood_depth/flood_depth_m.tif           Continuous depth in metres
  outputs/flood_depth/flood_depth_classified.tif  14-class depth categories
  outputs/flood_depth/flood_depth_map.png         Publication figure
  outputs/flood_depth/depth_statistics.json       Area statistics per class

Usage:
  python 08_flood_depth_sta.py
"""

import os
import json
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter, label as ndlabel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DEM_RAW    = "data/dem_raw/output_hh.tif"
FLOOD_MASK = "outputs/flood_maps/flood_final.tif"
OUT_DIR    = "outputs/flood_depth"

# ── Depth classification bins (metres) ───────────────────────────────────────
BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 9999]
DEPTH_LABELS = [
    "0 - 0.5", "0.5 - 1", "1 - 1.5", "1.5 - 2",
    "2 - 2.5", "2.5 - 3", "3 - 3.5", "3.5 - 4",
    "4 - 4.5", "4.5 - 5", "5 - 5.5", "5.5 - 6",
    "6 - 6.5", "> 6.5"
]
# Colour scheme matching reference figure (yellow → orange → red → blue)
DEPTH_COLORS = [
    "#FFFFF0",  # 0 - 0.5 m  (very light yellow)
    "#FFF5CC",  # 0.5 - 1 m
    "#FFE099",  # 1 - 1.5 m
    "#FFCA66",  # 1.5 - 2 m
    "#FFB133",  # 2 - 2.5 m
    "#FF9900",  # 2.5 - 3 m  (orange)
    "#FF6600",  # 3 - 3.5 m
    "#FF3300",  # 3.5 - 4 m
    "#CC0000",  # 4 - 4.5 m  (red)
    "#990000",  # 4.5 - 5 m
    "#ADD8E6",  # 5 - 5.5 m  (light blue)
    "#6495ED",  # 5.5 - 6 m
    "#0000FF",  # 6 - 6.5 m  (blue)
    "#00004D",  # > 6.5 m    (dark blue)
]


# ── Step 1: Reproject DEM ─────────────────────────────────────────────────────
def reproject_dem(dem_path, ref_path, out_path):
    log.info("Step 1: Reprojecting DEM to match flood map CRS...")

    with rasterio.open(dem_path) as src:
        dem_data      = src.read(1).astype(np.float32)
        src_crs       = src.crs
        src_transform = src.transform
        log.info(f"  DEM source CRS   : {src_crs}")
        log.info(f"  DEM shape        : {dem_data.shape}")
        log.info(f"  Elevation range  : {np.nanmin(dem_data):.1f} to {np.nanmax(dem_data):.1f} m")

    with rasterio.open(ref_path) as ref:
        ref_profile   = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs       = ref.crs
        ref_shape     = (ref.height, ref.width)

    dem_reproj = np.zeros(ref_shape, dtype=np.float32)
    reproject(
        source=dem_data,
        destination=dem_reproj,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear
    )

    log.info(f"  Reprojected shape: {dem_reproj.shape}")
    log.info(f"  UTM elev range   : {dem_reproj.min():.1f} to {dem_reproj.max():.1f} m")

    p = ref_profile.copy()
    p.update(dtype="float32", count=1, nodata=-9999)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(dem_reproj[np.newaxis])
    log.info(f"  Saved: {out_path}")

    return dem_reproj, ref_profile


# ── Step 2: Load flood mask ───────────────────────────────────────────────────
def load_flood_mask(flood_path):
    log.info("Step 2: Loading flood mask...")
    with rasterio.open(flood_path) as src:
        flood = src.read(1).astype(np.float32)
    flood_binary = (flood == 1)
    log.info(f"  Flood pixels: {flood_binary.sum():,}")
    return flood, flood_binary


# ── Step 3: STA depth estimation ──────────────────────────────────────────────
def compute_depth_sta(dem_reproj, flood, flood_binary):
    log.info("Step 3: Computing flood depth using STA technique...")

    # Smooth DEM to reduce noise
    dem_smooth = gaussian_filter(dem_reproj, sigma=3)

    # Label connected flood regions
    labeled, n_regions = ndlabel(flood_binary)
    log.info(f"  Connected flood regions: {n_regions:,}")

    depth_map = np.zeros_like(dem_smooth, dtype=np.float32)

    processed = 0
    for region_id in range(1, n_regions + 1):
        region_mask = labeled == region_id
        region_size = region_mask.sum()

        # Skip very small regions (noise)
        if region_size < 5:
            continue

        # WSE = 90th percentile elevation in region
        # (water surface sits at the high end of the flooded terrain)
        region_elev = dem_smooth[region_mask]
        wse = np.percentile(region_elev, 90)

        # Depth = WSE - local terrain elevation (clipped to >= 0)
        depth = np.maximum(wse - dem_smooth[region_mask], 0)
        depth_map[region_mask] = depth
        processed += 1

        if processed % 5000 == 0:
            log.info(f"  Processed {processed:,} / {n_regions:,} regions...")

    # Apply flood mask — zero outside flood, nodata for boundary
    depth_map[~flood_binary]  = 0
    depth_map[flood == 255]   = -9999

    valid_depths = depth_map[flood_binary]
    log.info(f"  Depth range : {valid_depths.min():.2f} to {valid_depths.max():.2f} m")
    log.info(f"  Mean depth  : {valid_depths.mean():.2f} m")
    log.info(f"  Median depth: {np.median(valid_depths):.2f} m")

    return depth_map


# ── Step 4: Classify depth ────────────────────────────────────────────────────
def classify_depth(depth_map, flood_binary):
    log.info("Step 4: Classifying depth into categories...")

    classified = np.zeros_like(depth_map, dtype=np.uint8)
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        mask = flood_binary & (depth_map >= lo) & (depth_map < hi)
        classified[mask] = i + 1  # classes 1-14

    return classified


# ── Step 5: Save outputs ──────────────────────────────────────────────────────
def save_outputs(depth_map, classified, ref_profile, flood_binary):
    log.info("Step 5: Saving output files...")

    # Continuous depth GeoTIFF
    p = ref_profile.copy()
    p.update(dtype="float32", count=1, nodata=-9999)
    depth_path = f"{OUT_DIR}/flood_depth_m.tif"
    with rasterio.open(depth_path, "w", **p) as dst:
        dst.write(depth_map[np.newaxis])
    log.info(f"  Saved: {depth_path}")

    # Classified depth GeoTIFF
    p2 = ref_profile.copy()
    p2.update(dtype="uint8", count=1, nodata=0)
    class_path = f"{OUT_DIR}/flood_depth_classified.tif"
    with rasterio.open(class_path, "w", **p2) as dst:
        dst.write(classified[np.newaxis])
    log.info(f"  Saved: {class_path}")

    # Statistics JSON
    total_flood = flood_binary.sum()
    stats = {"total_flood_ha": float(total_flood * 18 * 18 / 10000), "classes": []}

    log.info("\n=== DEPTH STATISTICS ===")
    for i, lbl in enumerate(DEPTH_LABELS):
        n   = int((classified == (i + 1)).sum())
        ha  = n * 18 * 18 / 10000
        pct = n / total_flood * 100 if total_flood > 0 else 0
        log.info(f"  {lbl:>10} m : {ha:>10,.1f} ha  ({pct:.1f}%)")
        stats["classes"].append({
            "label": f"{lbl} m",
            "pixels": n,
            "area_ha": round(ha, 2),
            "pct": round(pct, 2)
        })
    log.info(f"  {'Total':>10}   : {total_flood*18*18/10000:>10,.1f} ha")

    stats_path = f"{OUT_DIR}/depth_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"\n  Saved: {stats_path}")

    return stats


# ── Step 6: Visualisation ─────────────────────────────────────────────────────
def visualise(depth_map, classified, flood_binary, stats):
    log.info("Step 6: Generating flood depth map figure...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 10),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    # ── Left panel: depth map ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")

    # Build RGBA display image
    display = np.zeros((*classified.shape, 4), dtype=np.float32)

    # Non-flood = dark background
    display[~flood_binary] = [0.05, 0.05, 0.15, 1.0]

    # Depth classes
    for i, hex_col in enumerate(DEPTH_COLORS):
        mask = classified == (i + 1)
        if not mask.any():
            continue
        r = int(hex_col[1:3], 16) / 255
        g = int(hex_col[3:5], 16) / 255
        b = int(hex_col[5:7], 16) / 255
        display[mask] = [r, g, b, 0.9]

    ax.imshow(display, origin="upper", aspect="equal", interpolation="nearest")
    ax.set_title(
        "EOS-04 SAR Flood Depth — Krishna River Basin\n"
        "Andhra Pradesh | 26 July 2023 | STA Technique",
        color="white", fontsize=13, fontweight="bold", pad=10
    )
    ax.axis("off")

    # Coordinate labels
    ax.text(0.01, 0.01, "80.1°E", transform=ax.transAxes,
            color="white", fontsize=8, va="bottom")
    ax.text(0.99, 0.01, "81.9°E", transform=ax.transAxes,
            color="white", fontsize=8, va="bottom", ha="right")
    ax.text(0.01, 0.99, "18.2°N", transform=ax.transAxes,
            color="white", fontsize=8, va="top")
    ax.text(0.01, 0.01, "16.5°N", transform=ax.transAxes,
            color="white", fontsize=8, va="bottom")

    # ── Right panel: legend + stats ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    ax2.axis("off")

    ax2.text(0.5, 0.97, "Flood Depth (metres)",
             transform=ax2.transAxes, color="white",
             fontsize=12, fontweight="bold", ha="center", va="top")

    # Legend patches
    y_start = 0.90
    step    = 0.055
    for i, (lbl, col) in enumerate(zip(DEPTH_LABELS, DEPTH_COLORS)):
        y = y_start - i * step
        patch = mpatches.FancyBboxPatch(
            (0.05, y - 0.02), 0.18, 0.04,
            boxstyle="round,pad=0.01",
            facecolor=col, edgecolor="white", linewidth=0.5,
            transform=ax2.transAxes
        )
        ax2.add_patch(patch)

        # Area stat
        ha_val = stats["classes"][i]["area_ha"] if i < len(stats["classes"]) else 0
        ax2.text(0.28, y, f"{lbl} m",
                 transform=ax2.transAxes, color="white",
                 fontsize=9, va="center")
        ax2.text(0.95, y, f"{ha_val:,.0f} ha",
                 transform=ax2.transAxes, color="#aaaaaa",
                 fontsize=8, va="center", ha="right")

    # Summary stats
    valid = depth_map[flood_binary]
    y_stats = y_start - len(DEPTH_LABELS) * step - 0.04
    ax2.text(0.5, y_stats,
             f"Total flood area: {stats['total_flood_ha']:,.0f} ha\n"
             f"Mean depth: {valid.mean():.2f} m\n"
             f"Max depth:  {valid.max():.2f} m\n"
             f"Pixel size: 18 m × 18 m",
             transform=ax2.transAxes, color="white",
             fontsize=9, ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#2a2a4a", edgecolor="#555577"))

    plt.tight_layout(pad=0.5)
    out_path = f"{OUT_DIR}/flood_depth_map.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    log.info(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Check inputs exist
    for f in [DEM_RAW, FLOOD_MASK]:
        if not os.path.exists(f):
            log.error(f"File not found: {f}")
            return

    # Run pipeline
    dem_reproj, ref_profile = reproject_dem(
        DEM_RAW, FLOOD_MASK,
        "data/processed/dem_utm.tif"
    )
    flood, flood_binary = load_flood_mask(FLOOD_MASK)
    depth_map  = compute_depth_sta(dem_reproj, flood, flood_binary)
    classified = classify_depth(depth_map, flood_binary)
    stats      = save_outputs(depth_map, classified, ref_profile, flood_binary)
    visualise(depth_map, classified, flood_binary, stats)

    log.info("\n=== STA Flood Depth Estimation Complete ===")
    log.info(f"Outputs saved in: {OUT_DIR}/")
    log.info("  flood_depth_m.tif          — continuous depth (metres)")
    log.info("  flood_depth_classified.tif — 14-class depth map")
    log.info("  flood_depth_map.png        — publication figure")
    log.info("  depth_statistics.json      — area per depth class")


if __name__ == "__main__":
    main()
