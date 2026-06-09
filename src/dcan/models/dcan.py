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
    """Bilinear-interpolation kernel for initialising a ConvTranspose2d."""
    factor = (ksize + 1) // 2
    center = factor - 1 if ksize % 2 == 1 else factor - 0.5
    og = np.ogrid[:ksize, :ksize]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    w = np.zeros((channels, channels, ksize, ksize), dtype=np.float32)
    for i in range(channels):
        w[i, i] = filt
    return torch.from_numpy(w)


class _LearnedUpsampler(nn.Module):
    """Upsample score maps by `factor` via stacked bilinear-initialised deconv."""

    def __init__(self, num_classes: int, factor: int):
        super().__init__()
        layers = []
        n = 0
        f = factor
        while f > 1:
            deconv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
            deconv.weight.data.copy_(_bilinear_weights(num_classes, 4))
            layers.append(deconv)
            f //= 2
            n += 1
        self.up = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


class _Head(nn.Module):
    """Per-level classifier head producing a score map at the feature resolution."""

    def __init__(self, in_ch: int, head_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_ch, head_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(16, head_dim), head_dim),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(nn.Dropout2d(0.5), nn.Conv2d(head_dim, num_classes, kernel_size=1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.cls(self.transform(feat))


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
        score_maps = [head(f) for head, f in zip(self.heads, feats)]
        sizes = {tuple(s.shape[-2:]) for s in score_maps}
        if len(sizes) == 1:
            # Atrous backbone: all heads share the deep stride -> sum then learned upsample.
            fused = score_maps[0]
            for s in score_maps[1:]:
                fused = fused + s
            main = self.upsampler(fused, target_size)
        else:
            # Mixed resolutions (output_stride 32): upsample each to full res, then sum.
            ups = [F.interpolate(s, size=target_size, mode="bilinear", align_corners=False) for s in score_maps]
            main = ups[0]
            for u in ups[1:]:
                main = main + u
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
        """Returns dict with keys: obj_main, obj_aux, cnt_main, cnt_aux."""
        target_size = (x.shape[2], x.shape[3])
        x = self.enc1(x)
        x = self.enc2(x)
        f3 = self.enc3(x)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        fb = self.bottleneck(f5)
        feats = [f3, f4, fb]

        obj_main, obj_aux = self.obj_branch(feats, target_size)
        cnt_main, cnt_aux = self.contour_branch(feats, target_size)
        return {"obj_main": obj_main, "obj_aux": obj_aux, "cnt_main": cnt_main, "cnt_aux": cnt_aux}
