# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch.nn as nn

from rsl_rl.models import MLPModel


class ZeroInitMLPModel(MLPModel):
    """RSL-RL 5 MLP actor with a zero-initialized Gaussian mean output."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        linears = [module for module in self.mlp.modules() if isinstance(module, nn.Linear)]
        if not linears:
            raise TypeError(f"Expected an MLP with at least one Linear layer, got {type(self.mlp)}")
        nn.init.zeros_(linears[-1].weight)
        nn.init.zeros_(linears[-1].bias)
