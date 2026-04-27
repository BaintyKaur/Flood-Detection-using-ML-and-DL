#!/usr/bin/env bash
# ============================================================
# EOS-04 SAR Flood Detection — Full Pipeline Run Script (DGX)
# ============================================================
# Prerequisites:
#   - data/before/HH.tif, data/before/HV.tif  (20-Jan-2023)
#   - data/during/HH.tif, data/during/HV.tif  (26-Jul-2023)
#   - data/labels/flood_mask.tif               (binary reference; 1=flood, 0=non-flood)
#   - [optional] data/sample_labels.csv        (row,col,label for ML classifiers)

set -e
export PYTHONPATH="$PWD:$PYTHONPATH"

# ── Stage 1: Preprocessing & calibration ─────────────────────────────────────
echo "=== Stage 1: Preprocessing ==="
python 01_preprocess.py \
    --data_root data \
    --out_root  data/processed

# ── Stage 2+3: Feature engineering & classical methods ───────────────────────
echo "=== Stage 2+3: Features & Thresholding ==="
python 02_features_classical.py

# ── Stage 4: DL Training on DGX (8×A100) ─────────────────────────────────────
echo "=== Stage 4: Deep Learning Training ==="

# U-Net (fastest, good baseline)
torchrun --nproc_per_node=8 05_train.py \
    --model      unet \
    --batch_size 8 \
    --epochs     50 \
    --patch_size 512 \
    --n_train    4000 \
    --n_val      800 \
    --lr         1e-4 \
    --pos_weight 5.0 \
    --output_dir outputs/unet

# Attention U-Net (better small flood objects)
torchrun --nproc_per_node=8 05_train.py \
    --model      attention_unet \
    --batch_size 8 \
    --epochs     50 \
    --output_dir outputs/attention_unet

# SegFormer-B2 (best for complex scenes, ~27M params)
torchrun --nproc_per_node=8 05_train.py \
    --model      segformer \
    --batch_size 4 \
    --epochs     50 \
    --lr         6e-5 \
    --output_dir outputs/segformer

# ── Stage 5+6: Inference + post-processing ────────────────────────────────────
echo "=== Stage 5+6: Inference ==="

for MODEL in unet attention_unet segformer; do
    python 06_inference.py \
        --model_name  $MODEL \
        --model_path  outputs/$MODEL/checkpoints/best_model.pt \
        --stack_path  data/processed/stack_4ch.tif \
        --out_dir     outputs/flood_maps/$MODEL \
        --reference   data/labels/flood_mask.tif
done

# ── Stage 7: Visualisation ────────────────────────────────────────────────────
echo "=== Stage 7: Visualisation ==="
python 07_visualize.py \
    --out_dir     outputs/figures \
    --flood_path  outputs/flood_maps/unet/flood_ensemble.tif

echo ""
echo "Pipeline complete. Results in outputs/"
echo "  outputs/figures/       — all plots"
echo "  outputs/flood_maps/    — flood GeoTIFFs + GeoJSON vectors"
echo "  outputs/*/history.json — training curves"
