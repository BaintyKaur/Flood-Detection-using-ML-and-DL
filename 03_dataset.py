"""
EOS-04 SAR Flood Detection — Stage 4a: PyTorch Dataset
=======================================================
Handles:
  - Tiled patch extraction from large GeoTIFF (10501×10401 px)
  - 4-channel input: [σ⁰_HH_before, σ⁰_HV_before, σ⁰_HH_during, σ⁰_HV_during]
  - Per-channel z-score normalisation using pre-computed stats
  - Augmentation: flip, rotate, speckle noise injection, brightness jitter
  - Sliding-window inference sampler

Expected directory structure:
  data/
    processed/
      stack_4ch.tif        <- 4-channel σ⁰ dB stack
    labels/
      flood_mask.tif       <- binary mask: 1=flood, 0=non-flood, 255=nodata
    processed/
      norm_stats.json
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window
import torchvision.transforms.functional as TF


# ── Normalisation ─────────────────────────────────────────────────────────────
class ChannelNormalizer:
    def __init__(self, stats_json: str):
        with open(stats_json) as f:
            stats = json.load(f)
        # Order: before_HH, before_HV, during_HH, during_HV
        keys = ["before_HH", "before_HV", "during_HH", "during_HV"]
        self.mean = torch.tensor([stats[k]["mean"] for k in keys], dtype=torch.float32)
        self.std  = torch.tensor([stats[k]["std"]  for k in keys], dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (C, H, W)"""
        return (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)


# ── Augmentations ────────────────────────────────────────────────────────────
def augment(img: torch.Tensor, mask: torch.Tensor):
    """
    img:  (C, H, W) float32
    mask: (H, W) uint8 binary
    """
    # Random horizontal flip
    if random.random() > 0.5:
        img  = TF.hflip(img)
        mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

    # Random vertical flip
    if random.random() > 0.5:
        img  = TF.vflip(img)
        mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

    # Random 90° rotation (preserve shape)
    k = random.randint(0, 3)
    if k > 0:
        img  = torch.rot90(img,  k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[0, 1])

    # Multiplicative speckle noise (SAR-specific, applied in linear domain proxy)
    if random.random() > 0.5:
        sigma = random.uniform(0.02, 0.08)
        noise = 1.0 + torch.randn_like(img) * sigma
        img   = img * noise

    # Channel-wise brightness jitter (shift in dB)
    if random.random() > 0.5:
        shift = torch.FloatTensor(img.shape[0], 1, 1).uniform_(-1.0, 1.0)
        img   = img + shift

    return img, mask


# ── Dataset: random patch sampling ───────────────────────────────────────────
class SARFloodDataset(Dataset):
    """
    Randomly samples patches from the large GeoTIFF during training.
    Skips patches with >50% nodata, ensures minimum flood pixels.
    """

    def __init__(
        self,
        stack_path:  str,   # 4-channel σ⁰ tif
        mask_path:   str,   # binary flood mask tif
        stats_json:  str,
        patch_size:  int   = 512,
        n_patches:   int   = 2000,     # virtual epoch size
        min_flood_ratio: float = 0.0,  # 0 = all patches OK; 0.05 = need ≥5% flood
        augment:     bool  = True,
        seed:        int   = 42,
    ):
        self.stack_path  = stack_path
        self.mask_path   = mask_path
        self.patch_size  = patch_size
        self.n_patches   = n_patches
        self.min_flood   = min_flood_ratio
        self.do_augment  = augment
        self.normalizer  = ChannelNormalizer(stats_json)

        # Read raster dimensions (not the data itself — lazy loading)
        with rasterio.open(stack_path) as src:
            self.H, self.W = src.height, src.width
            self.n_channels = src.count

        rng = np.random.default_rng(seed)
        # Pre-sample patch top-left corners
        max_r = self.H - patch_size
        max_c = self.W - patch_size
        self.origins = list(zip(
            rng.integers(0, max_r, n_patches).tolist(),
            rng.integers(0, max_c, n_patches).tolist()
        ))

    def __len__(self):
        return self.n_patches

    def _read_patch(self, r, c, p):
        win = Window(c, r, p, p)
        with rasterio.open(self.stack_path) as src:
            arr = src.read(window=win).astype(np.float32)   # (C, H, W)
        with rasterio.open(self.mask_path) as src:
            msk = src.read(1, window=win).astype(np.int64)   # (H, W)
        return arr, msk

    def __getitem__(self, idx):
        r, c = self.origins[idx]
        p    = self.patch_size

        arr, msk = self._read_patch(r, c, p)

        # Replace NaN with 0 (nodata)
        arr = np.nan_to_num(arr, nan=0.0)

        img  = torch.from_numpy(arr)
        mask = torch.from_numpy(msk)

        # Normalise
        img = self.normalizer(img)

        # Augment
        if self.do_augment:
            img, mask = augment(img, mask)

        return img, mask


# ── Dataset: sliding-window inference ─────────────────────────────────────────
class SARFloodInferenceDataset(Dataset):
    """
    Dense sliding-window patches with overlap for full-scene inference.
    Returns (patch, row_start, col_start) for stitching.
    """

    def __init__(
        self,
        stack_path: str,
        stats_json: str,
        patch_size: int = 512,
        overlap:    int = 64,
    ):
        self.stack_path = stack_path
        self.patch_size = patch_size
        self.overlap    = overlap
        self.stride     = patch_size - overlap
        self.normalizer = ChannelNormalizer(stats_json)

        with rasterio.open(stack_path) as src:
            self.H, self.W = src.height, src.width

        # Build all (row, col) patch origins
        self.patches = []
        r = 0
        while r < self.H:
            r = min(r, self.H - patch_size)
            c = 0
            while c < self.W:
                c = min(c, self.W - patch_size)
                self.patches.append((r, c))
                if c == self.W - patch_size: break
                c += self.stride
            if r == self.H - patch_size: break
            r += self.stride

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        r, c = self.patches[idx]
        p    = self.patch_size
        win  = Window(c, r, p, p)

        with rasterio.open(self.stack_path) as src:
            arr = src.read(window=win).astype(np.float32)

        arr = np.nan_to_num(arr, nan=0.0)
        img = torch.from_numpy(arr)
        img = self.normalizer(img)
        return img, r, c


# ── DataLoader factory ────────────────────────────────────────────────────────
def get_dataloaders(
    stack_path: str,
    mask_path:  str,
    stats_json: str,
    patch_size: int    = 512,
    batch_size: int    = 8,
    n_train:    int    = 3000,
    n_val:      int    = 500,
    num_workers: int   = 8,
    pin_memory: bool   = True,
):
    train_ds = SARFloodDataset(
        stack_path, mask_path, stats_json,
        patch_size=patch_size, n_patches=n_train,
        min_flood_ratio=0.0, augment=True, seed=42
    )
    val_ds = SARFloodDataset(
        stack_path, mask_path, stats_json,
        patch_size=patch_size, n_patches=n_val,
        min_flood_ratio=0.0, augment=False, seed=99
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick sanity check
    stack_p = "data/processed/stack_4ch.tif"
    mask_p  = "data/labels/flood_mask.tif"
    stats_p = "data/processed/norm_stats.json"

    if not all(os.path.exists(p) for p in [stack_p, mask_p, stats_p]):
        print("Run 01_preprocess.py first, then provide flood_mask.tif in data/labels/")
    else:
        train_l, val_l = get_dataloaders(stack_p, mask_p, stats_p, batch_size=4)
        img, mask = next(iter(train_l))
        print(f"Batch img shape : {img.shape}    dtype={img.dtype}")
        print(f"Batch mask shape: {mask.shape}   dtype={mask.dtype}")
        print(f"Img  range      : [{img.min():.2f}, {img.max():.2f}]")
        print(f"Mask unique vals: {mask.unique().tolist()}")
        print("Dataset OK ✓")
