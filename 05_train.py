"""
EOS-04 SAR Flood Detection — Training Script (Clean Rewrite)
=============================================================
Fixes:
  - grad_fn error (loss always differentiable)
  - AMP deprecation warnings (updated API)
  - Works reliably on single GPU on shared server
  - Works with torchrun for multi-GPU if needed

Single GPU run:
    python 05_train.py --model unet --batch_size 4 --epochs 50

Multi-GPU run:
    torchrun --nproc_per_node=4 05_train.py --model unet --batch_size 4 --epochs 50
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── Loss ──────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p = probs.reshape(probs.size(0), -1)
        t = targets.float().reshape(targets.size(0), -1)
        inter = (p * t).sum(dim=1)
        dice  = (2 * inter + self.smooth) / (p.sum(dim=1) + t.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.dice = DiceLoss()

    def forward(self, logits, masks):
        # masks: (B, H, W)  values 0=non-flood 1=flood 255=nodata
        valid = (masks != 255)

        # Always return a differentiable tensor even if no valid pixels
        if not valid.any():
            return logits.sum() * 0.0

        # Expand valid mask to match logits shape
        l_valid = logits[valid]
        t_valid = masks[valid].float()

        bce_loss  = self.bce(l_valid, t_valid).mean()

        # Dice on valid pixels only — reshape to (1, N) for dice
        dice_loss = self.dice(
            l_valid.unsqueeze(0).unsqueeze(0),
            t_valid.unsqueeze(0).unsqueeze(0)
        )

        return 0.5 * bce_loss + 0.5 * dice_loss


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(logits, masks, threshold=0.5):
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).long()
        valid = masks != 255
        if not valid.any():
            return {"iou": 0.0, "f1": 0.0, "oa": 0.0}
        p = preds[valid].cpu().numpy()
        t = masks[valid].cpu().numpy()
        tp = int(((p == 1) & (t == 1)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        tn = int(((p == 0) & (t == 0)).sum())
        iou = tp / (tp + fp + fn + 1e-6)
        f1  = 2 * tp / (2 * tp + fp + fn + 1e-6)
        oa  = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        return {"iou": round(iou, 4), "f1": round(f1, 4), "oa": round(oa, 4)}


# ── Training epoch ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    n = 0
    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            logits = model(imgs).squeeze(1)   # (B, H, W)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n += 1

        if batch_idx % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            log.info(f"Ep{epoch:03d} [{batch_idx}/{len(loader)}] "
                     f"loss={loss.item():.4f} lr={lr:.2e}")

    return total_loss / max(n, 1)


# ── Validation epoch ──────────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    agg = {"iou": 0.0, "f1": 0.0, "oa": 0.0}
    n = 0
    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        m = compute_metrics(logits, masks)
        for k in agg:
            agg[k] += m[k]
        n += 1
    return total_loss / max(n, 1), {k: round(v / max(n, 1), 4) for k, v in agg.items()}


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # ── DDP detection ─────────────────────────────────────────────────────────
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = local_rank == 0
    else:
        world_size = 1
        is_main    = True
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main:
        log.info(f"Device: {device} | GPUs: {world_size} | "
                 f"model: {args.model} | batch: {args.batch_size} | "
                 f"epochs: {args.epochs}")
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    # Import here so PYTHONPATH issues show clearly
    try:
        from dataset import SARFloodDataset
        from models  import get_model
    except ImportError as e:
        log.error(f"Import error: {e}")
        log.error("Make sure dataset.py and models.py exist in the same folder as this script")
        log.error(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
        sys.exit(1)

    stack_path = os.path.join(args.data_root, "stack_4ch.tif")
    mask_path  = args.mask_path
    stats_json = os.path.join(args.data_root, "norm_stats.json")

    for f in [stack_path, mask_path, stats_json]:
        if not os.path.exists(f):
            log.error(f"File not found: {f}")
            sys.exit(1)

    train_ds = SARFloodDataset(
        stack_path, mask_path, stats_json,
        patch_size=args.patch_size, n_patches=args.n_train,
        augment=True, seed=42
    )
    val_ds = SARFloodDataset(
        stack_path, mask_path, stats_json,
        patch_size=args.patch_size, n_patches=args.n_val,
        augment=False, seed=99
    )

    if ddp:
        from torch.utils.data import DistributedSampler
        from torch.nn.parallel import DistributedDataParallel as DDP
        train_sampler = DistributedSampler(train_ds)
        val_sampler   = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler   = None

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(args.model, in_channels=4).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        log.info(f"Model parameters: {total_params:.1f}M")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion  = CombinedLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = 5 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler('cuda')

    # ── Training loop ─────────────────────────────────────────────────────────
    best_iou  = 0.0
    history   = []
    ckpt_dir  = Path(args.output_dir) / "checkpoints"

    for epoch in range(1, args.epochs + 1):
        if ddp and train_sampler:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device, epoch
        )
        val_loss, val_m = val_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        if is_main:
            log.info(
                f"Epoch {epoch:03d}/{args.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"IoU={val_m['iou']:.4f}  F1={val_m['f1']:.4f}  "
                f"OA={val_m['oa']:.4f}  [{elapsed:.0f}s]"
            )

            m_state = model.module.state_dict() if ddp else model.state_dict()

            # Save last checkpoint
            torch.save({
                "epoch": epoch, "model": m_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_iou": best_iou,
            }, ckpt_dir / "last.pt")

            # Save best
            if val_m["iou"] > best_iou:
                best_iou = val_m["iou"]
                torch.save(m_state, ckpt_dir / "best_model.pt")
                log.info(f"  *** New best IoU={best_iou:.4f} saved ***")

            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": val_loss, **val_m})
            with open(Path(args.output_dir) / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    if is_main:
        log.info(f"Training complete. Best IoU = {best_iou:.4f}")

    if ddp:
        dist.destroy_process_group()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="unet",
                   choices=["unet","attention_unet","segformer","deeplabv3p"])
    p.add_argument("--data_root",   default="data/processed")
    p.add_argument("--mask_path",   default="data/labels/flood_mask.tif")
    p.add_argument("--output_dir",  default="outputs/unet")
    p.add_argument("--patch_size",  type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--n_train",     type=int,   default=4000)
    p.add_argument("--n_val",       type=int,   default=800)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--pos_weight",  type=float, default=5.0)
    main(p.parse_args())
