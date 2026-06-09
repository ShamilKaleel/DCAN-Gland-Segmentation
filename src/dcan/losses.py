"""Loss functions with deep supervision and auxiliary weight scheduling."""

from __future__ import annotations

import torch
import torch.nn as nn


class DCANLoss(nn.Module):
    """Combined object + contour loss with deep-supervision auxiliary losses.

    L = L_o + L_c + w_a * (sum(aux_o_i) + sum(aux_c_i))
    """

    def __init__(
        self,
        obj_weight: torch.Tensor | None = None,
        cnt_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.obj_ce = nn.CrossEntropyLoss(weight=obj_weight)
        self.cnt_ce = nn.CrossEntropyLoss(weight=cnt_weight)

    def forward(
        self,
        outputs: dict,
        obj_target: torch.Tensor,
        cnt_target: torch.Tensor,
        w_a: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        l_obj = self.obj_ce(outputs["obj_main"], obj_target)
        l_cnt = self.cnt_ce(outputs["cnt_main"], cnt_target)

        aux_obj = sum(self.obj_ce(a, obj_target) for a in outputs["obj_aux"])
        aux_cnt = sum(self.cnt_ce(a, cnt_target) for a in outputs["cnt_aux"])

        total = l_obj + l_cnt + w_a * (aux_obj + aux_cnt)
        return {
            "total": total,
            "obj_main": l_obj,
            "cnt_main": l_cnt,
            "aux_obj": aux_obj,
            "aux_cnt": aux_cnt,
        }


class FCNBaselineLoss(nn.Module):
    def __init__(self, obj_weight: torch.Tensor | None = None):
        super().__init__()
        self.obj_ce = nn.CrossEntropyLoss(weight=obj_weight)

    def forward(
        self,
        outputs: dict,
        obj_target: torch.Tensor,
        w_a: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        l_obj = self.obj_ce(outputs["obj_main"], obj_target)
        aux_obj = sum(self.obj_ce(a, obj_target) for a in outputs["obj_aux"])
        total = l_obj + w_a * aux_obj
        return {"total": total, "obj_main": l_obj, "aux_obj": aux_obj}


class AuxWeightScheduler:
    """Decay w_a by factor 0.1 every `step_iters` iterations, down to `min_w`."""

    def __init__(self, init_w: float = 1.0, decay: float = 0.1, step_iters: int = 10000, min_w: float = 1e-3):
        self.init_w = init_w
        self.decay = decay
        self.step_iters = step_iters
        self.min_w = min_w

    def __call__(self, iteration: int) -> float:
        n = iteration // self.step_iters
        w = self.init_w * (self.decay ** n)
        return max(w, self.min_w)
