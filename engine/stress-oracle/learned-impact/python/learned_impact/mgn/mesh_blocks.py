# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA PhysicsNeMo @ c8f68428840a559f4b86ef56e96000b80a8d9594,
# physicsnemo/nn/module/gnn_layers/mesh_edge_block.py (MeshEdgeBlock) and
# mesh_node_block.py (MeshNodeBlock), Apache License 2.0 (LICENSE-Apache-2.0.txt).
#
# Changes: the graph is plain (src, dst) index tensors instead of a PyG/DGL object, and
# aggregation is `index_add_` (no torch_scatter), so the blocks run on MPS. Concatenation
# order and residuals are unchanged: edge `e' = e + MLP([e, h_src, h_dst])`, node
# `h' = h + MLP([Σ_in e', h])`. `TypedNodeBlock` and the global terms are ours (the plan's
# separate sums per edge type, plus a GraphNet global).
"""MeshGraphNets edge and node blocks on index tensors."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .mesh_graph_mlp import MeshGraphMLP


def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum rows of `src` into `dim_size` buckets (index_add_, deterministic on CPU)."""
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    if index.numel():
        out.index_add_(0, index, src)
    return out


def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    s = scatter_sum(src, index, dim_size)
    n = scatter_sum(torch.ones_like(index, dtype=src.dtype).unsqueeze(-1), index, dim_size)
    return s / n.clamp(min=1)


class MeshEdgeBlock(nn.Module):
    def __init__(self, input_dim_nodes: int = 512, input_dim_edges: int = 512,
                 output_dim: int = 512, hidden_dim: int = 512, hidden_layers: int = 1,
                 activation_fn: nn.Module | None = None, norm_type: str = "LayerNorm",
                 extra_dim: int = 0):
        super().__init__()
        # PhysicsNeMo's MeshGraphEdgeMLPConcat is a MeshGraphMLP on [e, src, dst]; the
        # attribute path `edge_mlp.model` matches the original state dict.
        self.edge_mlp = MeshGraphMLP(input_dim_edges + 2 * input_dim_nodes + extra_dim,
                                     output_dim, hidden_dim, hidden_layers, activation_fn,
                                     norm_type)

    def forward(self, efeat: Tensor, nfeat: Tensor, src: Tensor, dst: Tensor,
                extra: Tensor | None = None) -> tuple[Tensor, Tensor]:
        parts = [efeat, nfeat[src], nfeat[dst]]
        if extra is not None:
            parts.append(extra)
        return self.edge_mlp(torch.cat(parts, dim=-1)) + efeat, nfeat


class MeshNodeBlock(nn.Module):
    def __init__(self, aggregation: str = "sum", input_dim_nodes: int = 512,
                 input_dim_edges: int = 512, output_dim: int = 512, hidden_dim: int = 512,
                 hidden_layers: int = 1, activation_fn: nn.Module | None = None,
                 norm_type: str = "LayerNorm"):
        super().__init__()
        if aggregation not in ("sum", "mean"):
            raise ValueError(aggregation)
        self.aggregation = aggregation
        self.node_mlp = MeshGraphMLP(input_dim_nodes + input_dim_edges, output_dim, hidden_dim,
                                     hidden_layers, activation_fn, norm_type)

    def forward(self, efeat: Tensor, nfeat: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
        agg = (scatter_sum if self.aggregation == "sum" else scatter_mean)(
            efeat, dst, nfeat.shape[0])
        return efeat, self.node_mlp(torch.cat([agg, nfeat], dim=-1)) + nfeat


class TypedNodeBlock(nn.Module):
    """Node update with one sum per edge type and the global latent:
    `h' = h + MLP([Σ_type1 e', …, Σ_typeK e', h, g[batch]])`."""

    def __init__(self, n_edge_types: int, width: int, hidden_layers: int = 1,
                 activation_fn: nn.Module | None = None):
        super().__init__()
        self.node_mlp = MeshGraphMLP((n_edge_types + 2) * width, width, width, hidden_layers,
                                     activation_fn)

    def forward(self, efeats: list[Tensor], dsts: list[Tensor], nfeat: Tensor,
                gfeat_per_node: Tensor) -> Tensor:
        n = nfeat.shape[0]
        aggs = [scatter_sum(e, d, n) for e, d in zip(efeats, dsts)]
        return self.node_mlp(torch.cat([*aggs, nfeat, gfeat_per_node], dim=-1)) + nfeat


class GlobalBlock(nn.Module):
    """GraphNet global update (Battaglia 2018): `g' = g + MLP([g, mean_nodes h])`."""

    def __init__(self, width: int, hidden_layers: int = 1,
                 activation_fn: nn.Module | None = None):
        super().__init__()
        self.mlp = MeshGraphMLP(2 * width, width, width, hidden_layers, activation_fn)

    def forward(self, gfeat: Tensor, nfeat: Tensor, node_graph: Tensor) -> Tensor:
        pooled = scatter_mean(nfeat, node_graph, gfeat.shape[0])
        return self.mlp(torch.cat([gfeat, pooled], dim=-1)) + gfeat
