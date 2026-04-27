# EOS-04 SAR Flood Detection Pipeline 🛰️🌊

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.3-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DGX](https://img.shields.io/badge/Platform-NVIDIA%20DGX%20A100-76B900.svg)](https://www.nvidia.com/en-us/data-center/dgx-a100/)

---

## 📌 Overview

A complete end-to-end deep learning pipeline for **flood inundation mapping** using **EOS-04 (RISAT-1A) C-band SAR imagery**. This project detects and maps flood extent over the **Krishna River Basin, Andhra Pradesh, India** by comparing pre-flood (January 2023) and during-flood (July 2023) satellite acquisitions.

The pipeline integrates classical remote sensing methods (backscatter thresholding, Otsu segmentation) with state-of-the-art deep learning (U-Net semantic segmentation) and flood depth estimation using the **STA (SAR Topographic Analysis)** technique — all trained and executed on an **NVIDIA DGX A100 server**.

---

## 🛰️ Study Area & Data

| Parameter | Details |
|-----------|---------|
| **Satellite** | EOS-04 (RISAT-1A), ISRO |
| **Sensor** | SAR C-band, MRS Mode |
| **Polarisations** | HH + HV (Dual-pol) |
| **Pixel Spacing** | 18 m × 18 m |
| **Projection** | UTM Zone 44N, WGS84 |
| **Incidence Angle** | 37.91° |
| **Scene Centre** | 17.32°N, 80.99°E |
| **Scene Size** | ~190 × 188 km |
| **Pre-flood Date** | 20 January 2023 (Orbit 5141) |
| **Flood Date** | 26 July 2023 (Orbit 7968) |
| **DEM** | Copernicus GLO-30 (30m) |
| **Processing Level** | L2B ARD (RTC applied) |

---

## 🔬 Methodology

```
Raw EOS-04 DN GeoTIFF
        │
        ▼
┌─────────────────────────────────────────┐
│  STAGE 1 — Preprocessing & Calibration │
│  • DN → σ⁰ (sigma-naught) conversion   │
│  • Noise bias removal (HH/HV)          │
│  • Refined Lee speckle filter (7×7)    │
│  • Log transform → dB scale            │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  STAGE 2 — Feature Engineering         │
│  • σ⁰_HH, σ⁰_HV (before + during)     │
│  • Cross-pol ratio CR = HH/HV          │
│  • RFDI (Radar Forest Degradation)     │
│  • Temporal change Δσ⁰ (during−before) │
│  • GLCM texture features               │
│  • Local mean and std (7×7 window)     │
└────────────────────┬────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────┐
│  STAGE 3A        │  │  STAGE 3B            │
│  Classical       │  │  ML Classifiers      │
│  • Otsu HH       │  │  • Random Forest     │
│  • Otsu Δ change │  │  • XGBoost (GPU)     │
│                  │  │  • SVM               │
└────────┬─────────┘  └──────────┬───────────┘
         └──────────┬────────────┘
                    ▼
┌─────────────────────────────────────────┐
│  STAGE 4 — Deep Learning (DGX A100)    │
│  • U-Net semantic segmentation         │
│  • 4-channel input: HH/HV before+after │
│  • Patch size: 512×512                 │
│  • Loss: Dice + BCE (pos_weight=5)     │
│  • Optimiser: AdamW + OneCycleLR       │
│  • Mixed precision FP16 (AMP)          │
│  • GPU: NVIDIA A100 80GB               │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  STAGE 5 — Post-processing             │
│  • Sliding-window inference            │
│  • Morphological open/close            │
│  • Sieve filter (min 50 pixels)        │
│  • Ensemble: DL + Otsu voting          │
│  • SAR boundary masking                │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  STAGE 6 — Flood Depth (STA)           │
│  • Copernicus DEM reprojection         │
│  • Water Surface Elevation per region  │
│  • Depth = WSE − terrain elevation     │
│  • 14-class depth map (0 to >6.5m)    │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  STAGE 7 — Output                      │
│  • Flood extent GeoTIFF + GeoJSON      │
│  • Flood depth GeoTIFF                 │
│  • Accuracy assessment                 │
│  • QGIS visualisation                  │
└─────────────────────────────────────────┘
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Flood Area Detected** | 377,573 ha (3,775 km²) |
| **Scene Coverage** | 10.77% flooded |
| **Overall Accuracy** | 96.78% |
| **IoU (Flood class)** | 0.9405 |
| **F1 Score** | 0.9642 |
| **Precision** | 91.7% |
| **Recall** | 87.1% |
| **Cohen's Kappa** | 0.8747 |
| **Flood Polygons** | 10,796 objects |
| **Training Epochs** | 50 |
| **Best Epoch** | 38 |

### Flood Depth Distribution

| Depth Range | Area (ha) | Coverage |
|-------------|-----------|---------|
| 0 – 0.5 m | 53,229 | 13.6% |
| 0.5 – 1 m | 14,026 | 3.6% |
| 1 – 1.5 m | 13,041 | 3.3% |
| 1.5 – 2 m | 10,714 | 2.7% |
| 2 – 2.5 m | 8,794 | 2.2% |
| > 2.5 m | 292,470 | 74.6% |

---

## 🗂️ Project Structure

```
EOS4/
├── 01_preprocess.py          # DN → σ⁰ calibration + speckle filter
├── 02_features_classical.py  # Feature engineering + Otsu thresholding
├── 03_dataset.py             # PyTorch Dataset (patch sampling)
├── 04_models.py              # U-Net, Attention U-Net, SegFormer, DeepLabV3+
├── 05_train.py               # DGX training loop (AMP, DDP-ready)
├── 06_inference.py           # Sliding-window inference + ensemble
├── 07_visualize.py           # Figures + training curves
├── 08_flood_depth_sta.py     # STA flood depth estimation
├── fix_preprocess.py         # Shape mismatch fix (before/during)
├── dataset.py                # Symlink → 03_dataset.py
├── models.py                 # Symlink → 04_models.py
├── requirements.txt          # Python dependencies
├── run_pipeline.sh           # Full pipeline shell script
│
├── data/
│   ├── before/
│   │   ├── HH.tif            # Pre-flood HH polarisation (20-Jan-2023)
│   │   └── HV.tif            # Pre-flood HV polarisation
│   ├── during/
│   │   ├── HH.tif            # Flood HH polarisation (26-Jul-2023)
│   │   └── HV.tif            # Flood HV polarisation
│   ├── dem_raw/
│   │   └── output_hh.tif     # Copernicus GLO-30 DEM
│   ├── labels/
│   │   └── flood_mask.tif    # Training labels (from Otsu)
│   └── processed/            # Calibrated σ⁰, stacks, features
│
└── outputs/
    ├── unet/
    │   ├── checkpoints/
    │   │   ├── best_model.pt # Best U-Net weights (IoU=0.9405)
    │   │   └── last.pt
    │   └── history.json      # Training curves
    ├── flood_maps/
    │   ├── flood_final.tif   # Final flood extent mask
    │   ├── flood_final.geojson
    │   └── flood_report.json
    ├── flood_depth/
    │   ├── flood_depth_m.tif
    │   ├── flood_depth_classified.tif
    │   └── flood_depth_map.png
    └── figures/
        └── *.png
```

---

## ⚙️ Installation

### Requirements
- Python 3.9+
- NVIDIA GPU (tested on A100 80GB)
- CUDA 12.x

### Setup

```bash
# Create virtual environment
python3 -m venv ~/flood_env
source ~/flood_env/bin/activate

# Install geospatial packages (conda recommended for GDAL)
conda install -c conda-forge gdal rasterio geopandas -y

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages
pip install transformers timm scikit-learn xgboost shap \
            scikit-image scipy matplotlib pandas joblib \
            wandb tensorboard sentencepiece
```

---

## 🚀 Usage

### Step 1 — Preprocess
```bash
python 01_preprocess.py --data_root data --out_root data/processed
```

### Step 2 — Feature Engineering
```bash
python 02_features_classical.py
```

### Step 3 — Train U-Net
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python 05_train.py \
    --model unet --batch_size 4 --epochs 50

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 05_train.py \
    --model unet --batch_size 4 --epochs 50
```

### Step 4 — Inference
```bash
python 06_inference.py \
    --model_name unet \
    --model_path outputs/unet/checkpoints/best_model.pt \
    --stack_path data/processed/stack_4ch.tif \
    --out_dir outputs/flood_maps
```

### Step 5 — Flood Depth
```bash
python 08_flood_depth_sta.py
```

### Full pipeline
```bash
bash run_pipeline.sh
```

---

## 🖥️ Hardware

Trained and executed on **NVIDIA DGX A100** server:

| Component | Specification |
|-----------|--------------|
| GPUs | 8 × NVIDIA A100 SXM4 80GB |
| GPU Memory | 640 GB total |
| CUDA Version | 12.3 |
| Driver | 535.129.03 |
| Interconnect | NVLink |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.5.1+cu121 | Deep learning |
| rasterio | 1.4.3 | Raster I/O |
| GDAL | 3.7 | Geospatial processing |
| scikit-learn | 1.6.1 | ML classifiers |
| XGBoost | 2.1.4 | Gradient boosting |
| transformers | ≥4.35 | SegFormer |
| timm | ≥0.9.8 | DeepLabV3+ backbone |
| scipy | ≥1.11 | Morphological operations |
| geopandas | 1.0.1 | Vector export |

---

## 📍 Study Area

The study covers the **Krishna River Basin** in Andhra Pradesh and Telangana, India — one of the most flood-prone river systems in peninsular India. The July 2023 event was associated with the Southwest Monsoon causing widespread inundation of agricultural land, particularly paddy fields in the Krishna delta region near the Bay of Bengal.

**Scene extent:**
- North: 18.17°N
- South: 16.46°N
- East: 81.87°E
- West: 80.10°E

---

## 🔑 Key Findings

- U-Net achieved **IoU = 0.9405** on the flood detection task
- **377,573 hectares** (3,775 km²) of land was inundated
- The **Krishna River delta** showed the most severe flooding
- **Flooded paddy fields** accounted for the majority of agricultural inundation
- The STA technique estimated flood depths ranging from **0 to >6.5 metres**
- Deep learning significantly outperformed classical Otsu thresholding in complex terrain

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{eos04_flood_2023,
  title   = {EOS-04 SAR Flood Detection Pipeline: 
             Krishna River Basin Flood Mapping using Deep Learning},
  author  = {Bainty Kaur and Team},
  year    = {2026},
  school  = {KUDSIT},
  note    = {Capstone Project — Remote Sensing and Deep Learning}
}
```

---

## 👥 Team

**SODS Capstone Project — Semester 2**
KUDSIT DGX Server | Remote Sensing & Deep Learning

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- **ISRO / NRSC** for EOS-04 SAR data
- **Copernicus Programme** for GLO-30 DEM
- **KUDSIT** for DGX A100 compute access
- **Anthropic Claude** for pipeline development assistance
