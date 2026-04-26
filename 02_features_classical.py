"""
EOS-04 SAR Flood Detection — Stage 2+3: Features & Classical Methods
=====================================================================
Computes all SAR-derived features and runs threshold + ML classifiers.

Features computed:
  1. σ⁰_HH, σ⁰_HV  (both dates, dB)
  2. Cross-polarization ratio  CR = σ⁰_HH / σ⁰_HV  (linear domain)
  3. Radar Forest Degradation Index  RFDI = (HH - HV) / (HH + HV)
  4. Temporal change  Δσ⁰_HH, Δσ⁰_HV  (during - before, dB)
  5. GLCM texture (contrast, correlation, energy, homogeneity)
  6. Local mean and local std (window 7×7)

Classical methods:
  A. Otsu thresholding on σ⁰_HH_during
  B. Otsu on combined change (Δσ⁰_HH + Δσ⁰_HV)

ML classifiers (require labelled samples — see sample_labels.csv):
  Random Forest, XGBoost, SVM  with cross-validation
"""

import os
import json
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu
from skimage.feature import graycomatrix, graycoprops
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROC = "data/processed"
FEAT = "data/features"


# ── I/O ───────────────────────────────────────────────────────────────────────
def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile.copy()


def write_tif(path, arr, profile):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p = profile.copy(); p.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr[np.newaxis])


# ── Feature 1-4: Backscatter + ratio + RFDI + change ─────────────────────────
def compute_basic_features(proc_dir=PROC):
    feats = {}
    for date in ("before", "during"):
        for pol in ("HH", "HV"):
            arr, _ = read_tif(f"{proc_dir}/{date}_sigma0_{pol}_dB.tif")
            feats[f"{date}_{pol}_db"] = arr

    # Load linear versions for ratio (ratio in dB domain = subtraction)
    for date in ("before", "during"):
        hh, _ = read_tif(f"{proc_dir}/{date}_sigma0_HH_linear.tif")
        hv, _ = read_tif(f"{proc_dir}/{date}_sigma0_HV_linear.tif")

        with np.errstate(divide="ignore", invalid="ignore"):
            cr   = np.where(hv != 0, hh / hv, np.nan)
            rfdi = np.where((hh + hv) != 0, (hh - hv) / (hh + hv), np.nan)

        feats[f"{date}_CR"]   = cr
        feats[f"{date}_RFDI"] = rfdi

    # Change features
    feats["delta_HH"] = feats["during_HH_db"] - feats["before_HH_db"]
    feats["delta_HV"] = feats["during_HV_db"] - feats["before_HV_db"]
    feats["delta_CR"] = feats["during_CR"]     - feats["before_CR"]

    return feats


# ── Feature 5: Local stats (mean, std) ────────────────────────────────────────
def local_stats(arr: np.ndarray, window: int = 7):
    valid = np.where(np.isfinite(arr), arr, 0.0)
    lmean  = uniform_filter(valid, size=window)
    lmean2 = uniform_filter(valid ** 2, size=window)
    lstd   = np.sqrt(np.maximum(lmean2 - lmean ** 2, 0))
    mask = ~np.isfinite(arr)
    lmean[mask] = np.nan; lstd[mask] = np.nan
    return lmean, lstd


# ── Feature 6: GLCM texture (tile-based for large images) ────────────────────
def glcm_texture_tile(arr_db: np.ndarray, tile_size: int = 256, levels: int = 64):
    """
    Compute GLCM features tile-by-tile to fit in memory.
    Returns dict with keys: contrast, correlation, energy, homogeneity.
    Each is a full-size array with tile-level values (nearest-neighbour upsampled).
    NOTE: For speed on DGX, use tile_size=512 and reduce levels to 32.
    """
    H, W = arr_db.shape
    props = ["contrast", "correlation", "energy", "homogeneity"]
    out = {p: np.full((H, W), np.nan, dtype=np.float32) for p in props}

    # Quantize to [0, levels-1]
    vmin, vmax = np.nanpercentile(arr_db, [2, 98])
    q = ((arr_db - vmin) / (vmax - vmin + 1e-8) * (levels - 1)).astype(np.uint8)
    q = np.clip(q, 0, levels - 1)

    for row in range(0, H, tile_size):
        for col in range(0, W, tile_size):
            r1, r2 = row, min(row + tile_size, H)
            c1, c2 = col, min(col + tile_size, W)
            patch = q[r1:r2, c1:c2]
            if patch.size < 4:
                continue
            try:
                glcm = graycomatrix(patch, distances=[1], angles=[0], levels=levels,
                                    symmetric=True, normed=True)
                for p in props:
                    val = float(graycoprops(glcm, p)[0, 0])
                    out[p][r1:r2, c1:c2] = val
            except Exception:
                pass
    return out


# ── Classical: Otsu thresholding ──────────────────────────────────────────────
def otsu_flood_map(arr_db: np.ndarray, profile: dict, out_path: str,
                   label: str = "Otsu"):
    """
    Flood pixels have LOWER backscatter (specular reflection).
    Threshold separates water (low) from land (high).
    """
    valid = arr_db[np.isfinite(arr_db)]
    thresh = threshold_otsu(valid)
    flood_map = (arr_db < thresh).astype(np.uint8)
    flood_map[~np.isfinite(arr_db)] = 255  # nodata

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    p = profile.copy(); p.update(dtype="uint8", count=1, nodata=255)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(flood_map[np.newaxis])

    n_flood = int((flood_map == 1).sum())
    area_ha  = n_flood * (18 * 18) / 10000
    log.info(f"[{label}] thresh={thresh:.2f} dB | flood pixels={n_flood} | area={area_ha:.1f} ha")
    return flood_map, thresh


# ── ML classifiers ────────────────────────────────────────────────────────────
def build_feature_matrix(feats: dict, row_idx, col_idx):
    """Extract per-pixel feature vectors for labelled sample locations."""
    keys = [
        "during_HH_db", "during_HV_db",
        "before_HH_db", "before_HV_db",
        "delta_HH",     "delta_HV",
        "during_CR",    "during_RFDI",
    ]
    X = np.column_stack([feats[k][row_idx, col_idx] for k in keys])
    return X, keys


def load_samples(sample_csv: str, feats: dict):
    """
    Load labelled pixel samples from CSV.
    Expected columns: row, col, label  (label: 1=flood, 0=non-flood)
    """
    import pandas as pd
    df = pd.read_csv(sample_csv)
    rows = df["row"].values
    cols = df["col"].values
    y    = df["label"].values
    X, feat_names = build_feature_matrix(feats, rows, cols)

    # Drop NaN rows
    mask = np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], feat_names


def train_ml_classifiers(X, y, feat_names, out_dir="models"):
    """
    Train RF, XGBoost, SVM with 5-fold stratified CV.
    Saves trained models and SHAP feature importance plot.
    Designed for DGX: RF and XGBoost use all CPU cores (n_jobs=-1).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    import xgboost as xgb
    import joblib
    import shap

    os.makedirs(out_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=42
            ))
        ]),
        "XGBoost": Pipeline([
            ("clf", xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                tree_method="gpu_hist",     # DGX GPU acceleration
                device="cuda",
                n_jobs=-1, random_state=42, eval_metric="logloss",
                use_label_encoder=False
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                        class_weight="balanced", probability=True))
        ]),
    }

    results = {}
    for name, pipe in classifiers.items():
        log.info(f"Training {name} ...")
        scores = cross_val_score(pipe, X, y, cv=skf,
                                  scoring="f1", n_jobs=-1 if name != "SVM" else 1)
        log.info(f"  {name} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
        pipe.fit(X, y)
        joblib.dump(pipe, f"{out_dir}/{name}.joblib")
        results[name] = {"cv_f1_mean": float(scores.mean()),
                         "cv_f1_std":  float(scores.std())}

        # SHAP feature importance (RF and XGBoost only)
        if name in ("RandomForest", "XGBoost"):
            clf = pipe.named_steps["clf"]
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X[:500])  # sample 500 for speed
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            importance = np.abs(shap_vals).mean(0)
            log.info(f"  SHAP importance: { {feat_names[i]: round(float(importance[i]),4) for i in np.argsort(importance)[::-1]} }")

    with open(f"{out_dir}/ml_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Full feature stack for DL input ──────────────────────────────────────────
def save_full_feature_stack(feats: dict, profile: dict, out_dir=FEAT):
    """
    Saves an 8-channel GeoTIFF used as input to DL models.
    Channels:
      0: σ⁰_HH_during  1: σ⁰_HV_during
      2: σ⁰_HH_before  3: σ⁰_HV_before
      4: Δσ⁰_HH        5: Δσ⁰_HV
      6: CR_during      7: RFDI_during
    """
    os.makedirs(out_dir, exist_ok=True)
    keys = [
        "during_HH_db", "during_HV_db",
        "before_HH_db", "before_HV_db",
        "delta_HH",     "delta_HV",
        "during_CR",    "during_RFDI",
    ]
    stack = np.stack([feats[k] for k in keys], axis=0).astype(np.float32)
    p = profile.copy(); p.update(count=8, dtype="float32", nodata=np.nan)
    path = f"{out_dir}/feature_stack_8ch.tif"
    with rasterio.open(path, "w", **p) as dst:
        dst.write(stack)
    log.info(f"8-channel feature stack: {path}  shape={stack.shape}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== Stage 2+3: Feature Engineering & Classical Methods ===")
    os.makedirs(FEAT, exist_ok=True)

    feats = compute_basic_features()

    # Read profile from any processed file
    _, profile = read_tif(f"{PROC}/during_sigma0_HH_dB.tif")

    # Local stats on during_HH
    lmean, lstd = local_stats(feats["during_HH_db"])
    feats["during_HH_lmean"] = lmean
    feats["during_HH_lstd"]  = lstd

    # Save individual feature TIFs
    for name, arr in feats.items():
        write_tif(f"{FEAT}/{name}.tif", arr, profile)

    # Otsu flood maps
    _, thresh_hh = otsu_flood_map(
        feats["during_HH_db"], profile,
        f"{FEAT}/otsu_flood_HH.tif", "Otsu σ⁰_HH"
    )
    combined = feats["delta_HH"] + feats["delta_HV"]
    _, thresh_comb = otsu_flood_map(
        combined, profile,
        f"{FEAT}/otsu_flood_combined_change.tif", "Otsu combined Δ"
    )

    # 8-channel feature stack for DL
    save_full_feature_stack(feats, profile, FEAT)

    # ML classifiers (only if labelled samples exist)
    sample_csv = "data/sample_labels.csv"
    if os.path.exists(sample_csv):
        X, y, feat_names = load_samples(sample_csv, feats)
        log.info(f"Loaded {X.shape[0]} labelled samples (flood={y.sum()}, non-flood={(y==0).sum()})")
        train_ml_classifiers(X, y, feat_names, out_dir="models")
    else:
        log.warning(f"No labelled samples at {sample_csv}; skipping ML classifiers.")
        log.warning("Create sample_labels.csv with columns: row,col,label")

    log.info("Stage 2+3 complete.")


if __name__ == "__main__":
    main()
