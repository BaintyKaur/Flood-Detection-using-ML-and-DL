"""
EOS-04 SAR Flood Detection — Stage 1: Preprocessing & Calibration
==================================================================
Converts raw DN GeoTIFFs to calibrated σ⁰ (sigma-naught) in dB,
applies speckle filtering, and saves stacked multi-date arrays.

Metadata used:
  Calibration_Constant_HH/HV = 68.664
  Image_Noise_Bias_HH = 13783.756 (Jan), 14173.494 (Jul)
  Image_Noise_Bias_HV = 12172.063 (Jan), 12088.514 (Jul)
  RTC_Apply_Flag = 1  (terrain correction already applied)
  IncidenceAngle = 37.91°
  OutputPixelSpacing = 18.0 m

Input layout expected:
  data/
    before/HH.tif   (20-JAN-2023)
    before/HV.tif
    during/HH.tif   (26-JUL-2023)
    during/HV.tif

Outputs (data/processed/):
  before_sigma0_HH.tif, before_sigma0_HV.tif
  during_sigma0_HH.tif, during_sigma0_HV.tif
  stack_4ch.tif  → [σ⁰_HH_before, σ⁰_HV_before, σ⁰_HH_during, σ⁰_HV_during]
  change_HH.tif, change_HV.tif  → Δσ⁰ (during - before)
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Metadata constants ────────────────────────────────────────────────────────
CAL_CONST_HH = 68.664          # same for both dates
CAL_CONST_HV = 68.664
NOISE_BIAS = {
    "before": {"HH": 13783.756, "HV": 12172.063},   # 20-JAN-2023
    "during": {"HH": 14173.494, "HV": 12088.514},   # 26-JUL-2023
}
INCIDENCE_ANGLE_DEG = 37.91126
PIXEL_SPACING_M     = 18.0
NODATA_DN           = 0        # EOS-04 fill value


# ── Calibration: DN → σ⁰ (linear) ───────────────────────────────────────────
def dn_to_sigma0_linear(dn_arr: np.ndarray, noise_bias: float, cal_const: float) -> np.ndarray:
    """
    EOS-04 L2B calibration formula:
        σ⁰ = (DN² - noise_bias) / 10^(cal_const / 10)
    Returns linear σ⁰; invalid pixels (DN=0 or result≤0) → NaN.
    """
    dn = dn_arr.astype(np.float32)
    mask = dn == NODATA_DN
    sigma0 = (dn ** 2 - noise_bias) / (10 ** (cal_const / 10.0))
    sigma0[mask] = np.nan
    sigma0[sigma0 <= 0] = np.nan
    return sigma0


def sigma0_to_db(sigma0: np.ndarray) -> np.ndarray:
    """Convert linear σ⁰ → dB: 10 * log10(σ⁰)."""
    db = np.full_like(sigma0, np.nan)
    valid = np.isfinite(sigma0) & (sigma0 > 0)
    db[valid] = 10.0 * np.log10(sigma0[valid])
    return db


# ── Refined Lee speckle filter ────────────────────────────────────────────────
def refined_lee_filter(img: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Simplified Refined Lee speckle filter on linear σ⁰.
    Works in linear domain for statistical validity; output in linear.
    ENL (Equivalent Number of Looks) assumed from metadata: RangeLooks=2, AzimuthLooks=1 → ENL≈2.
    """
    ENL = 2.0
    valid = np.isfinite(img)
    img_c = np.where(valid, img, 0.0)

    # Local mean and variance via uniform filter
    mean_local = uniform_filter(img_c, size=window)
    mean_sq    = uniform_filter(img_c ** 2, size=window)
    var_local  = mean_sq - mean_local ** 2
    var_local  = np.maximum(var_local, 0)

    # Variance of the noise (based on ENL)
    var_noise = (mean_local ** 2) / ENL

    # Lee weight
    weight = var_local / (var_local + var_noise + 1e-10)

    filtered = mean_local + weight * (img_c - mean_local)
    filtered[~valid] = np.nan
    return filtered.astype(np.float32)


# ── I/O helpers ───────────────────────────────────────────────────────────────
def read_tif(path: str):
    """Return (array, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        return arr, src.profile.copy()


def write_tif(path: str, arr: np.ndarray, profile: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr[np.newaxis, :, :])
    log.info(f"Saved: {path}")


# ── Per-band processing pipeline ─────────────────────────────────────────────
def process_band(
    tif_path: str,
    out_path_linear: str,
    out_path_db: str,
    date_key: str,   # "before" or "during"
    pol: str,        # "HH" or "HV"
    profile: dict,
    apply_filter: bool = True,
):
    """Full calibration + speckle filter for one band/date."""
    dn, _ = read_tif(tif_path)
    log.info(f"Processing {date_key}/{pol}: DN range [{dn.min():.0f}, {dn.max():.0f}]")

    noise = NOISE_BIAS[date_key][pol]
    cal   = CAL_CONST_HH if pol == "HH" else CAL_CONST_HV

    sigma0 = dn_to_sigma0_linear(dn, noise, cal)
    if apply_filter:
        sigma0 = refined_lee_filter(sigma0, window=7)
    sigma0_db = sigma0_to_db(sigma0)

    write_tif(out_path_linear, sigma0,    profile)
    write_tif(out_path_db,     sigma0_db, profile)

    pct = np.nanpercentile(sigma0_db, [5, 50, 95])
    log.info(f"  σ⁰_dB  p5={pct[0]:.2f}  median={pct[1]:.2f}  p95={pct[2]:.2f}")
    return sigma0_db


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_root: str = "data", out_root: str = "data/processed"):
    os.makedirs(out_root, exist_ok=True)

    results = {}  # key → σ⁰_dB array

    for date_key in ("before", "during"):
        for pol in ("HH", "HV"):
            in_path = os.path.join(data_root, date_key, f"{pol}.tif")
            _, profile = read_tif(in_path)

            out_lin = os.path.join(out_root, f"{date_key}_sigma0_{pol}_linear.tif")
            out_db  = os.path.join(out_root, f"{date_key}_sigma0_{pol}_dB.tif")

            db = process_band(in_path, out_lin, out_db, date_key, pol, profile)
            results[f"{date_key}_{pol}"] = db

    # ── Change detection: Δσ⁰ = during - before ─────────────────────────────
    for pol in ("HH", "HV"):
        delta = results[f"during_{pol}"] - results[f"before_{pol}"]
        _, profile = read_tif(os.path.join(data_root, "before", f"{pol}.tif"))
        write_tif(os.path.join(out_root, f"change_{pol}_dB.tif"), delta, profile)
        log.info(f"Change {pol}: mean Δ={np.nanmean(delta):.2f} dB, std={np.nanstd(delta):.2f} dB")

    # ── 4-channel stack ──────────────────────────────────────────────────────
    # Channels: [σ⁰_HH_before, σ⁰_HV_before, σ⁰_HH_during, σ⁰_HV_during]
    _, profile = read_tif(os.path.join(data_root, "before", "HH.tif"))
    stack_keys = ["before_HH", "before_HV", "during_HH", "during_HV"]
    stack = np.stack([results[k] for k in stack_keys], axis=0)  # (4, H, W)

    stack_path = os.path.join(out_root, "stack_4ch.tif")
    p4 = profile.copy()
    p4.update(count=4, dtype="float32", nodata=np.nan)
    with rasterio.open(stack_path, "w", **p4) as dst:
        dst.write(stack)
    log.info(f"4-channel stack saved: {stack_path}  shape={stack.shape}")

    # ── Normalization stats (for DL input) ───────────────────────────────────
    stats = {}
    for i, key in enumerate(stack_keys):
        ch = stack[i]
        valid = ch[np.isfinite(ch)]
        stats[key] = {"mean": float(valid.mean()), "std": float(valid.std())}
        log.info(f"  {key}: mean={stats[key]['mean']:.3f} std={stats[key]['std']:.3f}")

    import json
    with open(os.path.join(out_root, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Normalization stats saved to norm_stats.json")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_root",  default="data/processed")
    args = parser.parse_args()
    main(args.data_root, args.out_root)
