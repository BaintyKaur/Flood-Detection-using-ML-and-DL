"""
Patch — run this instead of the broken section of 01_preprocess.py
Completes the pipeline from where it crashed:
  - Computes change maps (resampling before→during to match shapes)
  - Saves 4-channel stack
  - Saves norm_stats.json
"""

import os
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling

PROC = "data/processed"

def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile.copy()

def write_tif(path, arr, profile):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr[np.newaxis, :, :])
    print(f"Saved: {path}")

def resample_to_match(src_path, ref_path):
    """Resample src array to match the shape and transform of ref."""
    with rasterio.open(ref_path) as ref:
        ref_shape  = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs    = ref.crs
        ref_profile = ref.profile.copy()

    with rasterio.open(src_path) as src:
        # Read and resample in one step using rasterio
        data = src.read(
            1,
            out_shape=ref_shape,
            resampling=Resampling.bilinear
        ).astype(np.float32)

    return data, ref_profile

print("=== Completing preprocessing (shape-mismatch fix) ===")

# Load during images as the reference shape
results = {}
for pol in ["HH", "HV"]:
    during_arr, during_profile = read_tif(f"{PROC}/during_sigma0_{pol}_dB.tif")
    results[f"during_{pol}"] = during_arr
    print(f"during_{pol} shape: {during_arr.shape}")

# Resample before images to match during shape
for pol in ["HH", "HV"]:
    before_arr, _ = resample_to_match(
        f"{PROC}/before_sigma0_{pol}_dB.tif",
        f"{PROC}/during_sigma0_{pol}_dB.tif"
    )
    results[f"before_{pol}"] = before_arr
    print(f"before_{pol} resampled to: {before_arr.shape}")

    # Overwrite the before file with resampled version so shapes match permanently
    _, during_profile = read_tif(f"{PROC}/during_sigma0_{pol}_dB.tif")
    write_tif(f"{PROC}/before_sigma0_{pol}_dB.tif", before_arr, during_profile)

# Also resample before linear files
for pol in ["HH", "HV"]:
    before_lin, _ = resample_to_match(
        f"{PROC}/before_sigma0_{pol}_linear.tif",
        f"{PROC}/during_sigma0_{pol}_linear.tif"
    )
    _, during_profile = read_tif(f"{PROC}/during_sigma0_{pol}_linear.tif")
    write_tif(f"{PROC}/before_sigma0_{pol}_linear.tif", before_lin, during_profile)

# Compute change maps: during - before
_, ref_profile = read_tif(f"{PROC}/during_sigma0_HH_dB.tif")
for pol in ["HH", "HV"]:
    delta = results[f"during_{pol}"] - results[f"before_{pol}"]
    write_tif(f"{PROC}/change_{pol}_dB.tif", delta, ref_profile)
    print(f"Change {pol}: mean={np.nanmean(delta):.2f} dB  std={np.nanstd(delta):.2f} dB")

# Build 4-channel stack: [HH_before, HV_before, HH_during, HV_during]
stack_keys = ["before_HH", "before_HV", "during_HH", "during_HV"]
stack = np.stack([results[f"{d}_{p}"] for d, p in
                  [("before","HH"),("before","HV"),("during","HH"),("during","HV")]],
                 axis=0)  # (4, H, W)

stack_path = f"{PROC}/stack_4ch.tif"
p4 = ref_profile.copy()
p4.update(count=4, dtype="float32", nodata=np.nan)
with rasterio.open(stack_path, "w", **p4) as dst:
    dst.write(stack)
print(f"4-channel stack saved: {stack_path}  shape={stack.shape}")

# Compute and save normalisation stats
stats = {}
for i, (d, p) in enumerate([("before","HH"),("before","HV"),("during","HH"),("during","HV")]):
    ch = stack[i]
    valid = ch[np.isfinite(ch)]
    key = f"{d}_{p}"
    stats[key] = {"mean": float(valid.mean()), "std": float(valid.std())}
    print(f"  {key}: mean={stats[key]['mean']:.3f}  std={stats[key]['std']:.3f}")

with open(f"{PROC}/norm_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("Normalization stats saved.")

print("\n=== Preprocessing complete ===")
print("Files in data/processed/:")
for f in sorted(os.listdir(PROC)):
    size_mb = os.path.getsize(f"{PROC}/{f}") / 1e6
    print(f"  {f:45s}  {size_mb:.1f} MB")
