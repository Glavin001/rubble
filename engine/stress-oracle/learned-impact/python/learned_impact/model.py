"""Lean MGN-T-style residual network (smoke / train scaffolding)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import CHUNK_STATE_DIM, JOINT_STATE_DIM, K, LOCAL6_DIM


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MessagePassing(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.edge = MLP(3 * h, h, h)
        self.node = MLP(2 * h, h, h)
        self.norm = nn.LayerNorm(h)

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        row_ptr: torch.Tensor,
        col: torch.Tensor,
        endpoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Undirected edge features updated from endpoint nodes.
        i, j = endpoints[:, 0], endpoints[:, 1]
        e_in = torch.cat([node[i], node[j], edge], dim=-1)
        edge = edge + self.edge(e_in)
        # Aggregate both directions into nodes (CSR over bidirectional cols).
        n, h = node.shape
        agg = torch.zeros(n, h, device=node.device, dtype=node.dtype)
        # Build directed edge features matching CSR order: for each undirected edge, both ends.
        # Simpler: scatter from undirected endpoints twice.
        agg.index_add_(0, i, edge)
        agg.index_add_(0, j, edge)
        node = self.norm(node + self.node(torch.cat([node, agg], dim=-1)))
        return node, edge


class PhysicsAttention(nn.Module):
    """Soft tokens over nodes (P tokens), multi-head attention, mix back."""

    def __init__(self, h: int, p: int, heads: int = 4):
        super().__init__()
        self.p = p
        # heads must divide h
        self.heads = max(1, min(heads, h))
        while h % self.heads != 0 and self.heads > 1:
            self.heads -= 1
        self.assign = nn.Linear(h, p)
        self.qkv = nn.Linear(h, 3 * h)
        self.out = nn.Linear(h, h)
        self.ffn = nn.Sequential(
            nn.Linear(h, 4 * h),
            nn.SiLU(),
            nn.Linear(4 * h, h),
        )
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)

    def forward(self, node: torch.Tensor) -> torch.Tensor:
        # Soft assignment N×P → tokens P×H
        w = F.softmax(self.assign(node), dim=0)  # over nodes
        tokens = w.transpose(0, 1) @ node  # P×H
        qkv = self.qkv(tokens).view(self.p, 3, self.heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # heads: P×heads×d
        scale = (q.shape[-1]) ** -0.5
        attn = torch.einsum("phd,qhd->hpq", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        mixed = torch.einsum("hpq,qhd->phd", attn, v).reshape(self.p, -1)
        tokens = self.norm1(tokens + self.out(mixed))
        tokens = self.norm2(tokens + self.ffn(tokens))
        # Broadcast tokens back to nodes via soft weights.
        return node + w @ tokens


class LearnedImpactNet(nn.Module):
    """Encode → 2×MP → attention → 2×MP → bond/chunk heads."""

    def __init__(self, h: int = 48, p: int = 64, node_in: int | None = None, edge_in: int | None = None):
        super().__init__()
        # chunk: mass(1)+inertia(3)+half(3)+state(12)+loads(K*6)+grav(3)
        if node_in is None:
            node_in = 1 + 3 + 3 + CHUNK_STATE_DIM + K * LOCAL6_DIM + 3
        if edge_in is None:
            edge_in = JOINT_STATE_DIM + LOCAL6_DIM  # joint + qs force
        self.h = h
        self.p = p
        self.encode_n = MLP(node_in, h, h)
        self.encode_e = MLP(edge_in, h, h)
        self.mp = nn.ModuleList([MessagePassing(h) for _ in range(4)])
        self.attn = PhysicsAttention(h, p)
        self.bond_fail = nn.Linear(h, K + 1)  # 0..K
        self.bond_util = nn.Linear(h, 1)
        self.chunk_delta = nn.Linear(h, CHUNK_STATE_DIM)

    def encode_inputs(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        loads = batch["chunk_load_bins"].reshape(batch["chunk_load_bins"].shape[0], -1)
        node_in = torch.cat(
            [
                batch["chunk_mass"][:, None],
                batch["chunk_inertia"],
                batch["chunk_half_extents"],
                batch["chunk_state"],
                loads,
                batch["chunk_gravity_local"],
            ],
            dim=-1,
        )
        edge_in = torch.cat([batch["bond_joint_state"], batch["bond_qs_force"]], dim=-1)
        return self.encode_n(node_in), self.encode_e(edge_in)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        node, edge = self.encode_inputs(batch)
        row_ptr, col, endpoints = batch["csr_row_ptr"], batch["csr_col"], batch["bond_endpoints"]
        node, edge = self.mp[0](node, edge, row_ptr, col, endpoints)
        node, edge = self.mp[1](node, edge, row_ptr, col, endpoints)
        node = self.attn(node)
        node, edge = self.mp[2](node, edge, row_ptr, col, endpoints)
        node, edge = self.mp[3](node, edge, row_ptr, col, endpoints)
        return {
            "bond_fail_logits": self.bond_fail(edge),
            "bond_peak_util": self.bond_util(edge).squeeze(-1),
            "chunk_state_delta": self.chunk_delta(node),
        }


class BondMLPBaseline(nn.Module):
    """Per-bond MLP on joint state + QS force (no message passing / attention)."""

    def __init__(self, h: int = 48, edge_in: int | None = None):
        super().__init__()
        if edge_in is None:
            edge_in = JOINT_STATE_DIM + LOCAL6_DIM
        self.h = h
        self.p = 0
        self.encode = MLP(edge_in, h, h)
        self.bond_fail = nn.Linear(h, K + 1)
        self.bond_util = nn.Linear(h, 1)
        # Chunk head: weak baseline — predict zeros via a bias-only path from mean edge.
        self.chunk_delta = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, CHUNK_STATE_DIM))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        edge_in = torch.cat([batch["bond_joint_state"], batch["bond_qs_force"]], dim=-1)
        edge = self.encode(edge_in)
        n = batch["chunk_state"].shape[0]
        pooled = (edge.mean(dim=0, keepdim=True) if edge.shape[0]
                  else edge.new_zeros((1, self.h))).expand(n, -1)
        return {
            "bond_fail_logits": self.bond_fail(edge),
            "bond_peak_util": self.bond_util(edge).squeeze(-1),
            "chunk_state_delta": self.chunk_delta(pooled),
        }
