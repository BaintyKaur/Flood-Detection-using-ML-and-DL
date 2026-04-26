"""
EOS-04 SAR Flood Detection — Stage 4b: DL Model Architectures
==============================================================
All models:
  - in_channels = 4  (σ⁰_HH_before, σ⁰_HV_before, σ⁰_HH_during, σ⁰_HV_during)
  - out_classes  = 1  (binary flood / non-flood logit)

Available:
  1. UNet             — custom lightweight baseline
  2. AttentionUNet    — UNet + attention gates
  3. get_segformer()  — HuggingFace SegFormer-B2 with 4-ch patch embed
  4. get_deeplabv3p() — DeepLabV3+ (timm ResNet50 backbone, 4-ch conv1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared building blocks ────────────────────────────────────────────────────
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )
    def forward(self, x): return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up   = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if shapes differ (possible when input ≠ 2^n)
        dh = x2.shape[2] - x1.shape[2]
        dw = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ── 1. Vanilla UNet ───────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    Standard encoder-decoder UNet.
    in_channels=4, bilinear upsampling, ~31M parameters with base_ch=64.
    """
    def __init__(self, in_channels=4, base_ch=64, bilinear=True):
        super().__init__()
        b = base_ch
        self.inc   = DoubleConv(in_channels, b)
        self.down1 = Down(b,    b*2)
        self.down2 = Down(b*2,  b*4)
        self.down3 = Down(b*4,  b*8)
        self.down4 = Down(b*8,  b*16)
        self.up1   = Up(b*16 + b*8, b*8, bilinear)
        self.up2   = Up(b*8  + b*4, b*4, bilinear)
        self.up3   = Up(b*4  + b*2, b*2, bilinear)
        self.up4   = Up(b*2  + b,   b,   bilinear)
        self.out   = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.out(x)


# ── 2. Attention Gate U-Net ───────────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al., 2018).
    g = gating signal (from decoder), x = skip connection.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g1 to match x1 if needed
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    U-Net with attention gates on skip connections.
    """
    def __init__(self, in_channels=4, base_ch=64):
        super().__init__()
        b = base_ch
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b,   b*2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b*2, b*4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b*4, b*8))
        self.bot  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b*8, b*16))

        self.ag4 = AttentionGate(b*16, b*8,  b*8)
        self.ag3 = AttentionGate(b*8,  b*4,  b*4)
        self.ag2 = AttentionGate(b*4,  b*2,  b*2)
        self.ag1 = AttentionGate(b*2,  b,    b)

        self.up4 = Up(b*16+b*8, b*8)
        self.up3 = Up(b*8 +b*4, b*4)
        self.up2 = Up(b*4 +b*2, b*2)
        self.up1 = Up(b*2 +b,   b)

        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bot(e4)

        e4 = self.ag4(b,  e4); d4 = self.up4(b,  e4)
        e3 = self.ag3(d4, e3); d3 = self.up3(d4, e3)
        e2 = self.ag2(d3, e2); d2 = self.up2(d3, e2)
        e1 = self.ag1(d2, e1); d1 = self.up1(d2, e1)
        return self.out(d1)


# ── 3. SegFormer (HuggingFace transformers) ───────────────────────────────────
def get_segformer(in_channels=4, pretrained=True):
    """
    SegFormer-B2 with the patch embedding modified for 4-ch SAR input.
    Mix-Transformer encoder + MLP decoder.
    No pretrained weights for 4-ch, so we init conv1 from scratch
    (or average RGB channels and duplicate).
    """
    from transformers import SegformerForSemanticSegmentation, SegformerConfig

    config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    config.num_labels           = 1
    config.id2label             = {0: "flood"}
    config.label2id             = {"flood": 0}
    config.ignore_mismatched_sizes = True

    model = SegformerForSemanticSegmentation(config)

    # Replace first patch embedding conv: 3→4 channels
    old_conv = model.segformer.encoder.patch_embeddings[0].proj
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    if pretrained:
        # Average the 3 RGB weights into 4 channels
        with torch.no_grad():
            w3 = old_conv.weight.data          # (out, 3, kH, kW)
            avg = w3.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)
            new_conv.weight.data = avg.repeat(1, in_channels, 1, 1)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()
    model.segformer.encoder.patch_embeddings[0].proj = new_conv

    # Wrap to produce (B, 1, H', W') logit
    class SegFormerWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
        def forward(self, x):
            out = self.model(pixel_values=x)
            logits = out.logits   # (B, 1, H/4, W/4)
            # Upsample to input size
            logits = F.interpolate(logits, size=x.shape[2:],
                                   mode="bilinear", align_corners=False)
            return logits

    return SegFormerWrapper(model)


# ── 4. DeepLabV3+ (timm backbone) ────────────────────────────────────────────
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, dilations=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvBnRelu(in_ch, out_ch, kernel=1, padding=0),  # 1×1
        ] + [
            ConvBnRelu(in_ch, out_ch, kernel=3,
                       padding=d, dilation=d) for d in dilations
        ])
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            ConvBnRelu((len(dilations) + 2) * out_ch, out_ch, kernel=1, padding=0),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = [conv(x) for conv in self.convs]
        gp = F.interpolate(self.global_avg(x), size=size,
                           mode="bilinear", align_corners=False)
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))


def get_deeplabv3p(in_channels=4, pretrained_backbone=True):
    """
    DeepLabV3+ with ResNet50 backbone (timm).
    Modifies conv1 for 4-ch input.
    """
    import timm

    class DeepLabV3Plus(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = timm.create_model(
                "resnet50", pretrained=pretrained_backbone,
                features_only=True, out_indices=[1, 4]
            )
            # Patch conv1 for 4-ch input
            old = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False
            )
            if pretrained_backbone:
                with torch.no_grad():
                    w = old.weight.data              # (64, 3, 7, 7)
                    avg = w.mean(dim=1, keepdim=True) # (64, 1, 7, 7)
                    new_conv.weight.data = avg.repeat(1, in_channels, 1, 1)
            backbone.conv1 = new_conv
            self.backbone = backbone

            # ASPP on layer4 features (2048 ch for ResNet50)
            self.aspp     = ASPP(2048, 256)
            # Low-level features from layer1 (256 ch)
            self.low_proj = ConvBnRelu(256, 48, kernel=1, padding=0)
            # Decoder
            self.decoder  = nn.Sequential(
                ConvBnRelu(256 + 48, 256),
                ConvBnRelu(256, 256),
                nn.Conv2d(256, 1, 1),
            )

        def forward(self, x):
            feats = self.backbone(x)
            low, high = feats[0], feats[1]

            high = self.aspp(high)
            high = F.interpolate(high, size=low.shape[2:],
                                  mode="bilinear", align_corners=False)
            low  = self.low_proj(low)
            cat  = torch.cat([high, low], dim=1)
            out  = self.decoder(cat)
            out  = F.interpolate(out, size=x.shape[2:],
                                  mode="bilinear", align_corners=False)
            return out

    return DeepLabV3Plus()


# ── Model registry ────────────────────────────────────────────────────────────
def get_model(name: str, in_channels: int = 4) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(in_channels=in_channels)
    elif name == "attention_unet":
        return AttentionUNet(in_channels=in_channels)
    elif name == "segformer":
        return get_segformer(in_channels=in_channels)
    elif name == "deeplabv3p":
        return get_deeplabv3p(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from unet | attention_unet | segformer | deeplabv3p")


if __name__ == "__main__":
    print("Testing model shapes (batch=2, 4ch, 512×512)...")
    x = torch.randn(2, 4, 512, 512)
    for name in ["unet", "attention_unet"]:
        m = get_model(name)
        m.eval()
        with torch.no_grad():
            y = m(x)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  {name:20s}  out={y.shape}  params={params:.1f}M")
    print("Model check OK ✓  (SegFormer/DeepLabV3+ require HuggingFace/timm)")
