"""
EOS-04 SAR Flood Detection — Stage 5+6: Inference + Post-processing
====================================================================
1. Full-scene sliding-window inference (DL model)
2. Morphological clean-up
3. Ensemble: vote across Otsu + ML + DL predictions
4. Accuracy assessment (if reference mask available)
5. Export flood map GeoTIFF + inundation statistics

Usage:
    python 06_inference.py \
        --model_path outputs/checkpoints/best_model.pt \
        --model_name unet \
        --stack_path data/processed/stack_4ch.tif \
        --out_dir    outputs/flood_maps

    # With reference for validation:
        --reference  data/labels/flood_mask.tif
"""

import os
import argparse
import json
import logging
import numpy as np
import torch
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from scipy.ndimage import binary_opening, binary_closing, label as ndlabel
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from dataset import SARFloodInferenceDataset
from models  import get_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Sliding-window inference ──────────────────────────────────────────────────
def run_inference(
    model: torch.nn.Module,
    stack_path: str,
    stats_json: str,
    device: torch.device,
    patch_size: int = 512,
    overlap:    int = 64,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Returns full-scene probability map (float32, [0,1]).
    Uses overlap-tile averaging to reduce boundary artefacts.
    """
    ds = SARFloodInferenceDataset(
        stack_path, stats_json, patch_size=patch_size, overlap=overlap
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    H, W = ds.H, ds.W
    prob_map = np.zeros((H, W), dtype=np.float64)
    count    = np.zeros((H, W), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for imgs, row_starts, col_starts in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast():
                logits = model(imgs).squeeze(1)  # (B, H_p, W_p)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)

            for b in range(probs.shape[0]):
                r = int(row_starts[b])
                c = int(col_starts[b])
                prob_map[r:r+patch_size, c:c+patch_size] += probs[b]
                count   [r:r+patch_size, c:c+patch_size] += 1.0

    count = np.maximum(count, 1.0)
    return (prob_map / count).astype(np.float32)


# ── Post-processing ───────────────────────────────────────────────────────────
def postprocess_mask(
    prob_map: np.ndarray,
    threshold: float  = 0.5,
    min_area_px: int  = 25,        # ~0.8 ha at 18m pixel
    morph_open: int   = 3,         # removes speckle noise in flood map
    morph_close: int  = 5,         # fills small holes inside flood regions
) -> np.ndarray:
    """
    1. Threshold probability map
    2. Binary opening (remove isolated pixels)
    3. Binary closing (fill small holes)
    4. Sieve filter (remove small objects)
    Returns binary mask (uint8).
    """
    binary = (prob_map >= threshold).astype(np.uint8)

    struct_open  = np.ones((morph_open,  morph_open),  dtype=bool)
    struct_close = np.ones((morph_close, morph_close), dtype=bool)

    cleaned = binary_opening( binary.astype(bool), structure=struct_open)
    cleaned = binary_closing(cleaned,               structure=struct_close)

    # Remove objects smaller than min_area_px
    labeled, n_feat = ndlabel(cleaned)
    sizes = np.bincount(labeled.ravel())
    keep  = sizes >= min_area_px
    keep[0] = False  # background
    cleaned = keep[labeled].astype(np.uint8)

    return cleaned


# ── Ensemble voting ───────────────────────────────────────────────────────────
def ensemble_vote(maps: list, weights: list = None) -> np.ndarray:
    """
    Weighted majority vote across N binary flood maps.
    maps: list of (H, W) uint8 arrays
    weights: list of floats (defaults to uniform)
    Returns binary flood map (uint8).
    """
    if weights is None:
        weights = [1.0] * len(maps)
    total_w = sum(weights)
    score   = np.zeros_like(maps[0], dtype=np.float32)
    for m, w in zip(maps, weights):
        score += m.astype(np.float32) * w
    return (score / total_w >= 0.5).astype(np.uint8)


# ── Accuracy assessment ───────────────────────────────────────────────────────
def accuracy_assessment(pred: np.ndarray, ref: np.ndarray) -> dict:
    """
    Full accuracy assessment ignoring nodata (255).
    """
    valid = ref != 255
    p = pred[valid].astype(np.int32)
    t = ref[valid].astype(np.int32)

    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())

    total = tp + fp + fn + tn
    oa    = (tp + tn) / (total + 1e-6)
    iou   = tp / (tp + fp + fn + 1e-6)
    f1    = 2 * tp / (2 * tp + fp + fn + 1e-6)
    prec  = tp / (tp + fp + 1e-6)
    rec   = tp / (tp + fn + 1e-6)

    # Cohen's Kappa
    po   = oa
    pe_1 = ((tp + fp) / total) * ((tp + fn) / total)
    pe_0 = ((tn + fn) / total) * ((tn + fp) / total)
    pe   = pe_1 + pe_0
    kappa = (po - pe) / (1 - pe + 1e-6)

    metrics = {
        "overall_accuracy": round(oa, 4),
        "iou_flood":        round(iou, 4),
        "f1_flood":         round(f1, 4),
        "precision_flood":  round(prec, 4),
        "recall_flood":     round(rec, 4),
        "kappa":            round(kappa, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    log.info("=== Accuracy Assessment ===")
    for k, v in metrics.items():
        log.info(f"  {k}: {v}")
    return metrics


# ── Inundation statistics ─────────────────────────────────────────────────────
def compute_flood_stats(flood_mask: np.ndarray, pixel_spacing_m: float = 18.0) -> dict:
    n_flood = int((flood_mask == 1).sum())
    area_ha  = n_flood * (pixel_spacing_m ** 2) / 10000.0
    pct      = n_flood / (flood_mask.size) * 100

    stats = {
        "flood_pixels":   n_flood,
        "flood_area_ha":  round(area_ha, 2),
        "flood_pct":      round(pct, 3),
        "pixel_spacing_m": pixel_spacing_m,
        "total_pixels":   int(flood_mask.size),
    }
    log.info(f"Flood extent: {n_flood:,} pixels | {area_ha:,.1f} ha | {pct:.2f}%")
    return stats


# ── Export ────────────────────────────────────────────────────────────────────
def export_flood_tif(flood_mask: np.ndarray, profile: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    p = profile.copy(); p.update(dtype="uint8", count=1, nodata=255)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(flood_mask[np.newaxis])
    log.info(f"Flood map saved: {out_path}")


def export_flood_vector(flood_mask: np.ndarray, profile: dict,
                         out_path: str, crs=None):
    """Export flood polygons as GeoJSON."""
    transform = profile["transform"]
    crs_out   = crs or profile.get("crs")

    polygons = []
    for geom, val in shapes(flood_mask.astype(np.uint8), transform=transform):
        if int(val) == 1:
            polygons.append(shape(geom))

    if polygons:
        gdf = gpd.GeoDataFrame({"geometry": polygons}, crs=crs_out)
        gdf.to_file(out_path.replace(".tif", ".geojson"), driver="GeoJSON")
        log.info(f"Flood vector saved: {out_path.replace('.tif', '.geojson')} ({len(polygons)} polygons)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Inference on {device}")

    stats_json = os.path.join(os.path.dirname(args.stack_path), "norm_stats.json")

    # ── DL Inference ──────────────────────────────────────────────────────────
    model = get_model(args.model_name, in_channels=4).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    log.info(f"Model loaded: {args.model_path}")

    log.info("Running sliding-window inference ...")
    prob_map = run_inference(
        model, args.stack_path, stats_json, device,
        patch_size=args.patch_size, overlap=64, batch_size=args.batch_size
    )

    # Save probability map
    with rasterio.open(args.stack_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="float32", nodata=-9999)
    prob_path = os.path.join(args.out_dir, "flood_prob_DL.tif")
    with rasterio.open(prob_path, "w", **profile) as dst:
        dst.write(prob_map[np.newaxis])
    log.info(f"Probability map: {prob_path}")

    # Post-process
    dl_mask = postprocess_mask(prob_map, threshold=0.5)
    export_flood_tif(dl_mask, profile, os.path.join(args.out_dir, "flood_DL.tif"))
    export_flood_vector(dl_mask, profile,   os.path.join(args.out_dir, "flood_DL.tif"))

    # ── Load Otsu mask if available ────────────────────────────────────────────
    maps, weights, labels = [dl_mask], [2.0], ["DL"]

    otsu_path = "data/features/otsu_flood_HH.tif"
    if os.path.exists(otsu_path):
        with rasterio.open(otsu_path) as src:
            otsu_mask = src.read(1)
        otsu_clean = postprocess_mask(otsu_mask.astype(np.float32), threshold=0.5)
        maps.append(otsu_clean); weights.append(1.0); labels.append("Otsu")

    # ── Ensemble ───────────────────────────────────────────────────────────────
    if len(maps) > 1:
        log.info(f"Ensemble: {labels}")
        ensemble_mask = ensemble_vote(maps, weights)
        export_flood_tif(ensemble_mask, profile,
                          os.path.join(args.out_dir, "flood_ensemble.tif"))
        export_flood_vector(ensemble_mask, profile,
                             os.path.join(args.out_dir, "flood_ensemble.tif"))
        final_mask = ensemble_mask
    else:
        final_mask = dl_mask

    # ── Stats ──────────────────────────────────────────────────────────────────
    stats = compute_flood_stats(final_mask, pixel_spacing_m=18.0)

    # ── Accuracy assessment ───────────────────────────────────────────────────
    if args.reference and os.path.exists(args.reference):
        with rasterio.open(args.reference) as src:
            ref_mask = src.read(1)
        acc = accuracy_assessment(final_mask, ref_mask)
        stats.update(acc)

    # Save report
    report = {"model": args.model_name, "flood_stats": stats}
    with open(os.path.join(args.out_dir, "flood_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved to flood_report.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser("EOS-04 Flood Map Inference")
    p.add_argument("--model_path",  default="outputs/checkpoints/best_model.pt")
    p.add_argument("--model_name",  default="unet")
    p.add_argument("--stack_path",  default="data/processed/stack_4ch.tif")
    p.add_argument("--out_dir",     default="outputs/flood_maps")
    p.add_argument("--patch_size",  type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--reference",   default="", help="Reference mask for accuracy assessment")
    args = p.parse_args()
    main(args)
