# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA PhysicsNeMo @ c8f68428840a559f4b86ef56e96000b80a8d9594,
# physicsnemo/nn/module/gnn_layers/mesh_graph_mlp.py (MeshGraphMLP), licensed under the
# Apache License 2.0 (see LICENSE-Apache-2.0.txt in this directory).
#
# Changes: plain torch.nn.LayerNorm (no Transformer Engine), no nvFuser recompute path,
# no profiling decorators. Layer order and parameter layout are unchanged, so a state dict
# from the original loads into this module.
"""Residual-block MLP of MeshGraphNets: Linear, act, [Linear, act]×(h−1), Linear, LayerNorm."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class MeshGraphMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int | None = 1,
        activation_fn: nn.Module | None = None,
        norm_type: str | None = "LayerNorm",
    ):
        super().__init__()
        act = activation_fn if activation_fn is not None else nn.SiLU()
        if hidden_layers is not None:
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), act]
            self.hidden_layers = hidden_layers
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), act]
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.norm_type = norm_type
            if norm_type is not None:
                if norm_type != "LayerNorm":
                    raise ValueError(f"norm_type {norm_type!r}: only LayerNorm is supported")
                layers.append(nn.LayerNorm(output_dim))
            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
