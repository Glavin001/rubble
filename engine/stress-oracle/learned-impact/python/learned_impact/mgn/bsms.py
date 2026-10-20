# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA PhysicsNeMo @ c8f68428840a559f4b86ef56e96000b80a8d9594,
# physicsnemo/nn/module/gnn_layers/bsms.py, Apache License 2.0 (LICENSE-Apache-2.0.txt).
# That file is in turn modified from BSMS-GNN (https://github.com/Eydcao/BSMS-GNN,
# Cao et al., ICML 2023), also Apache License 2.0, whose notice it carries.
#
# Changes:
# - `degree(index, num_nodes)` honours `num_nodes` (the original ignores it and sizes the
#   output by `max(index) + 1`, which breaks when the last nodes have no out-edges, e.g. a
#   chunk isolated by broken bonds), and `cal_ew` clamps the degree at 1, so an isolated
#   node yields finite weights instead of inf;
# - aggregation through `index_add_` (no torch_scatter);
# - `WeightedEdgeConv.forward` no longer scales the gathered tensor in place;
# - `GraphMessagePassing.forward_fiber` takes precomputed edge-local fibre features instead
#   of absolute positions, plus an optional per-node extra input (the global latent), for
#   the invariant model. `forward(x, g, pos)` keeps the original behaviour.
# Only the 2-D (unbatched) layout is kept.
"""BSMS-GNN bi-stride message passing, pooling and unpooling layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .mesh_blocks import scatter_sum
from .mesh_graph_mlp import MeshGraphMLP


def degree(index: Tensor, num_nodes: int, dtype: torch.dtype = torch.float) -> Tensor:
    out = torch.zeros((num_nodes,), dtype=dtype, device=index.device)
    if index.numel():
        out.index_add_(0, index, torch.ones_like(index, dtype=dtype))
    return out


class GraphMessagePassing(nn.Module):
    def __init__(self, latent_dim: int, hidden_layer: int, pos_dim: int, extra_dim: int = 0):
        super().__init__()
        self.mlp_node = MeshGraphMLP(2 * latent_dim + extra_dim, latent_dim, latent_dim,
                                     hidden_layer)
        self.mlp_edge = MeshGraphMLP(2 * latent_dim + pos_dim + 1, latent_dim, latent_dim,
                                     hidden_layer)
        self.pos_dim = pos_dim

    def forward(self, x: Tensor, g: Tensor, pos: Tensor) -> Tensor:
        i, j = g[0], g[1]
        d = pos[i] - pos[j]
        fiber = torch.cat([d, torch.norm(d, dim=-1, keepdim=True)], dim=-1)
        return self.forward_fiber(x, g, fiber)

    def forward_fiber(self, x: Tensor, g: Tensor, fiber: Tensor,
                      extra: Tensor | None = None) -> Tensor:
        i, j = g[0], g[1]
        edge_embedding = self.mlp_edge(torch.cat([fiber, x[i], x[j]], dim=-1))
        aggr_out = scatter_sum(edge_embedding, j, x.shape[0])
        parts = [x, aggr_out] if extra is None else [x, aggr_out, extra]
        return self.mlp_node(torch.cat(parts, dim=-1)) + x


class WeightedEdgeConv(nn.Module):
    def forward(self, x: Tensor, g: Tensor, ew: Tensor, aggragating: bool = True) -> Tensor:
        i, j = g[0], g[1]
        weighted_info = (x[i] if aggragating else x[j]) * ew.unsqueeze(-1)
        target_index = j if aggragating else i
        return scatter_sum(weighted_info, target_index, x.shape[0])

    @torch.no_grad()
    def cal_ew(self, w: Tensor, g: Tensor) -> tuple[Tensor, Tensor]:
        deg = degree(g[0], num_nodes=w.shape[0], dtype=w.dtype).clamp(min=1)
        normed_w = w.squeeze(-1) / deg
        i, j = g[0], g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = scatter_sum(w_to_send, j, normed_w.shape[0]) + eps
        ec = w_to_send / aggr_w[j]
        return ec, aggr_w


class Unpool(nn.Module):
    def forward(self, h: Tensor, pre_node_num: int, idx: Tensor) -> Tensor:
        new_h = h.new_zeros([pre_node_num, h.shape[-1]])
        new_h[idx] = h
        return new_h


class BistrideGraphMessagePassing(nn.Module):
    """The original BSMS U-processor (kept for the PhysicsNeMo parity fixture)."""

    def __init__(self, unet_depth: int, latent_dim: int, hidden_layer: int, pos_dim: int):
        super().__init__()
        self.bottom_gmp = GraphMessagePassing(latent_dim, hidden_layer, pos_dim)
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.unet_depth = unet_depth
        self.edge_conv = WeightedEdgeConv()
        for _ in range(unet_depth):
            self.down_gmps.append(GraphMessagePassing(latent_dim, hidden_layer, pos_dim))
            self.up_gmps.append(GraphMessagePassing(latent_dim, hidden_layer, pos_dim))
            self.unpools.append(Unpool())

    def forward(self, h: Tensor, m_ids: list[Tensor], m_gs: list[Tensor], pos: Tensor) -> Tensor:
        down_outs, down_ps, cts = [], [], []
        w = pos.new_ones((pos.shape[-2], 1))
        for i in range(self.unet_depth):
            h = self.down_gmps[i](h, m_gs[i], pos)
            down_outs.append(h)
            down_ps.append(pos)
            ew, w = self.edge_conv.cal_ew(w, m_gs[i])
            h = self.edge_conv(h, m_gs[i], ew)
            pos = self.edge_conv(pos, m_gs[i], ew)
            cts.append(ew)
            h = h[m_ids[i]]
            pos = pos[m_ids[i]]
            w = w[m_ids[i]]
        h = self.bottom_gmp(h, m_gs[self.unet_depth], pos)
        for i in range(self.unet_depth):
            d = self.unet_depth - i - 1
            g, idx = m_gs[d], m_ids[d]
            h = self.unpools[i](h, down_outs[d].shape[-2], idx)
            h = self.edge_conv(h, g, cts[d], aggragating=False)
            h = self.up_gmps[i](h, g, down_ps[d])
            h = h.add(down_outs[d])
        return h
