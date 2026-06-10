"""DCAN: Deep Contour-Aware Network for gland segmentation.

Architecture from Chen et al., CVPR 2016 (Sec. 2.1-2.3 / Fig. 2):
- VGG-16 encoder converted to a **DeepLab-LargeFOV atrous backbone** (output
  stride 8): the last two max-pools are made stride-1 and the following
  convolutions are dilated, so deep features stay at 1/8 resolution for sharp
  boundaries (the paper initialised its downsampling path from DeepLab).
- **Multi-level contextual fusion**: classifier heads tap features from three
  encoder depths (different receptive fields) and fuse their score maps by
  summation, with deep supervision on each.
- Two branches (object + contour) share the encoder.

`output_stride=8` is the paper-faithful, high-accuracy setting (needs a capable
GPU). `output_stride=32` keeps the original 2 GB-friendly backbone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# --------------------------------------------------------------------------- #
# Atrous (DeepLab-LargeFOV) encoder construction
# --------------------------------------------------------------------------- #
def _build_encoder(pretrained: bool, output_stride: int):
    """Split VGG-16-BN into 5 stages; optionally make it atrous (stride 8).

    Returns (enc1..enc5, deep_channels, deep_stride).
    """
    vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
    features = list(vgg.features.children())

    enc1 = nn.Sequential(*features[:7])     # → /2,  64
    enc2 = nn.Sequential(*features[7:14])   # → /4,  128
    enc3 = nn.Sequential(*features[14:24])  # → /8,  256
    enc4 = nn.Sequential(*features[24:34])  # → /16, 512  (block4 convs + pool4)
    enc5 = nn.Sequential(*features[34:44])  # → /32, 512  (block5 convs + pool5)

    if output_stride == 8:
        # DeepLab-LargeFOV: keep /8 by removing the last two pooling strides and
        # dilating the subsequent convolutions to preserve the receptive field.
        # pool4 (last layer of enc4): stride 2 -> 1 (kernel 3, pad 1 preserves size)
        enc4[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # block5 convs (in enc5): dilation 2
        for m in enc5.modules():
            if isinstance(m, nn.Conv2d):
                m.dilation = (2, 2)
                m.padding = (2, 2)
        # pool5 (last layer of enc5): stride 2 -> 1
        enc5[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        deep_stride = 8
    elif output_stride == 32:
        deep_stride = 32
    else:
        raise ValueError(f"output_stride must be 8 or 32, got {output_stride}")

    return enc1, enc2, enc3, enc4, enc5, deep_stride


def _bilinear_weights(channels: int, ksize: int) -> torch.Tensor:
    """Bilinear-interpolation kernel for initialising a ConvTranspose2d.

    We let the network *learn* the upsampling (deconvolution) but start it from a
    plain bilinear-interpolation filter — a strong, artefact-free initial guess
    that the paper also uses. `filt` is the standard 2-D triangular (bilinear)
    window; it is copied onto the diagonal of the weight tensor so each output
    channel is upsampled only from its own input channel (a per-channel resize),
    leaving cross-channel mixing to be learned from this identity-like start.
    """
    factor = (ksize + 1) // 2
    center = factor - 1 if ksize % 2 == 1 else factor - 0.5
    og = np.ogrid[:ksize, :ksize]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    w = np.zeros((channels, channels, ksize, ksize), dtype=np.float32)
    for i in range(channels):
        w[i, i] = filt  # diagonal only -> channel i upsamples from channel i
    return torch.from_numpy(w)


class _LearnedUpsampler(nn.Module):
    """Upsample score maps by `factor` via stacked bilinear-initialised deconv."""

    def __init__(self, num_classes: int, factor: int):
        super().__init__()
        # Upsample by `factor` using a stack of ×2 deconvolutions (factor=8 -> 3
        # layers). Each step doubles H,W; stacking ×2 stages is more stable and
        # learns sharper edges than a single large-kernel deconv.
        layers = []
        n = 0
        f = factor
        while f > 1:
            deconv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
            deconv.weight.data.copy_(_bilinear_weights(num_classes, 4))  # start = bilinear ×2
            layers.append(deconv)
            f //= 2
            n += 1
        self.up = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = self.up(x)
        # Deconv factors are powers of two; if the input size isn't an exact
        # multiple, snap the result to the exact target size with a bilinear pass.
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


class _Head(nn.Module):
    """Per-level classifier head producing a score map at the feature resolution.

    One head is attached to each tapped encoder depth. It (1) `transform`s the
    (256/512/1024-channel) encoder feature down to a common `head_dim` with a 3×3
    conv + GroupNorm + ReLU (GroupNorm, not BatchNorm, so the head is robust to
    small batch sizes), then (2) `cls`-ifies it with Dropout + a 1×1 conv into
    `num_classes` (2: background / foreground) score logits at the same /8 grid.
    """

    def __init__(self, in_ch: int, head_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_ch, head_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, head_dim), head_dim),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(nn.Dropout2d(0.5), nn.Conv2d(head_dim, num_classes, kernel_size=1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.cls(self.transform(feat))  # -> (B, num_classes, H/8, W/8)


class _MultiLevelBranch(nn.Module):
    """Multi-level contextual fusion branch (paper Fig. 2 / Sec. 2.1).

    Heads tap three encoder depths (all at the deep stride for the atrous
    backbone). Their score maps are summed (sum-fusion) and the fused map is
    upsampled to the input resolution; each head is also a deeply-supervised
    auxiliary output. Returns (main_logits, [aux...]) at the input resolution.
    """

    def __init__(self, in_chs: tuple[int, ...], deep_stride: int, head_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.heads = nn.ModuleList([_Head(c, head_dim, num_classes) for c in in_chs])
        self.upsampler = _LearnedUpsampler(num_classes, factor=deep_stride)

    def forward(self, feats: list[torch.Tensor], target_size: tuple[int, int]):
        # One score map per tapped depth (different receptive fields = multi-level
        # context). This is the "multi-level contextual fusion" of paper §2.1.
        score_maps = [head(f) for head, f in zip(self.heads, feats)]
        sizes = {tuple(s.shape[-2:]) for s in score_maps}
        if len(sizes) == 1:
            # Atrous backbone (output_stride 8): every tap is already at /8, so we
            # SUM the score maps first (cheap, at /8) and learn-upsample once.
            fused = score_maps[0]
            for s in score_maps[1:]:
                fused = fused + s
            main = self.upsampler(fused, target_size)
        else:
            # Legacy /32 backbone: taps live at different strides, so each score
            # map must be resized to full resolution before they can be summed.
            ups = [F.interpolate(s, size=target_size, mode="bilinear", align_corners=False) for s in score_maps]
            main = ups[0]
            for u in ups[1:]:
                main = main + u
        # Deep supervision: every individual head map (not just the fused one) is
        # upsampled and returned as an auxiliary output so each depth gets a direct
        # gradient. The training loss weights these by w_a (see losses.py).
        aux = [F.interpolate(s, size=target_size, mode="bilinear", align_corners=False) for s in score_maps]
        return main, aux


class DCAN(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, head_dim: int = 64, output_stride: int = 8):
        super().__init__()
        self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, deep_stride = _build_encoder(
            pretrained, output_stride
        )

        # fc6-equivalent bottleneck. For the atrous backbone use the LargeFOV
        # dilation (12); GroupNorm keeps it batch-size agnostic.
        dilation = 12 if output_stride == 8 else 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),
        )

        # The two branches share the encoder but have independent heads/upsamplers.
        # They tap the SAME three depths (enc3=256, enc4=512, bottleneck=1024).
        # obj_branch predicts the gland body; contour_branch predicts gland edges.
        # The contour branch is the only thing the FCN baseline drops.
        in_chs = (256, 512, 1024)  # enc3, enc4, bottleneck
        self.obj_branch = _MultiLevelBranch(in_chs, deep_stride, head_dim, num_classes)
        self.contour_branch = _MultiLevelBranch(in_chs, deep_stride, head_dim, num_classes)

    def freeze_encoder_bn(self) -> None:
        """Keep pretrained encoder BatchNorm layers in eval mode (for small batches)."""
        for enc in (self.enc1, self.enc2, self.enc3, self.enc4, self.enc5):
            for m in enc.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, x: torch.Tensor):
        """input image -> object & contour logits (the full forward pass).

        Returns a dict: obj_main / cnt_main are the fused full-resolution logits
        used at inference; obj_aux / cnt_aux are the per-head deep-supervision
        outputs used only by the training loss.
        """
        target_size = (x.shape[2], x.shape[3])  # remember input H,W to upsample back to
        # --- shared encoder (downsampling path) ---
        x = self.enc1(x)          # /2,  64
        x = self.enc2(x)          # /4,  128
        f3 = self.enc3(x)         # /8,  256  <- tapped
        f4 = self.enc4(f3)        # /8,  512  (atrous: stays /8)  <- tapped
        f5 = self.enc5(f4)        # /8,  512  (atrous: stays /8)
        fb = self.bottleneck(f5)  # /8, 1024  (fc6, dilation 12)  <- tapped
        feats = [f3, f4, fb]

        # --- two parallel decoder branches (same features, separate weights) ---
        obj_main, obj_aux = self.obj_branch(feats, target_size)      # gland body
        cnt_main, cnt_aux = self.contour_branch(feats, target_size)  # gland edges
        return {"obj_main": obj_main, "obj_aux": obj_aux, "cnt_main": cnt_main, "cnt_aux": cnt_aux}
