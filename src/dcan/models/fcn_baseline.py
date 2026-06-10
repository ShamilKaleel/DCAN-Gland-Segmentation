"""Object-only FCN baseline (CUMedVision1-style ablation).

Identical encoder and multi-level contextual fusion as DCAN, but with only the
object branch (no contour prediction), so the ablation isolates the effect of
the contour branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dcan.models.dcan import _build_encoder, _MultiLevelBranch


class FCNBaseline(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, head_dim: int = 64, output_stride: int = 8):
        super().__init__()
        self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, deep_stride = _build_encoder(
            pretrained, output_stride
        )
        dilation = 12 if output_stride == 8 else 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),
        )
        self.obj_branch = _MultiLevelBranch((256, 512, 1024), deep_stride, head_dim, num_classes)

    def freeze_encoder_bn(self) -> None:
        for enc in (self.enc1, self.enc2, self.enc3, self.enc4, self.enc5):
            for m in enc.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, x: torch.Tensor):
        # Same encoder + multi-level fusion as DCAN, but a SINGLE (object) branch —
        # no contour prediction, so there is nothing to fuse-subtract and touching
        # glands stay merged. This is exactly the ablation that isolates the value
        # of DCAN's contour branch.
        target_size = (x.shape[2], x.shape[3])
        x = self.enc1(x)
        x = self.enc2(x)
        f3 = self.enc3(x)         # /8, 256  (tapped)
        f4 = self.enc4(f3)        # /8, 512  (tapped)
        f5 = self.enc5(f4)        # /8, 512
        fb = self.bottleneck(f5)  # /8, 1024 (tapped)
        obj_main, obj_aux = self.obj_branch([f3, f4, fb], target_size)
        return {"obj_main": obj_main, "obj_aux": obj_aux}
