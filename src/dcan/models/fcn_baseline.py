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
        target_size = (x.shape[2], x.shape[3])
        x = self.enc1(x)
        x = self.enc2(x)
        f3 = self.enc3(x)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        fb = self.bottleneck(f5)
        obj_main, obj_aux = self.obj_branch([f3, f4, fb], target_size)
        return {"obj_main": obj_main, "obj_aux": obj_aux}
