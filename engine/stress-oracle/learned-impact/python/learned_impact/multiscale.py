"""Topology-aware latent communication; the physical graph is never decimated.

Matching uses stable node IDs, never coordinates or tensor row order. Pooling
cannot join disconnected components. The CPU hierarchy builder is deliberately
measured separately from device inference; it is not a resident GPU algorithm.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .model import MLP


@dataclass(frozen=True)
class Level:
    parent: torch.Tensor
    endpoints: torch.Tensor
    active: torch.Tensor
    size: int

    def to(self, device):
        return Level(self.parent.to(device), self.endpoints.to(device), self.active.to(device), self.size)


def build_hierarchy(node_ids, endpoints, connected, levels=8):
    """Greedy connected matching, with no physical edge/contact truncation."""
    if levels < 0 or node_ids.ndim != 1 or node_ids.dtype != torch.long:
        raise ValueError('invalid hierarchy IDs/levels')
    if endpoints.dtype != torch.long or endpoints.shape != (len(connected), 2):
        raise ValueError('invalid hierarchy endpoints')
    if connected.dtype != torch.bool or connected.ndim != 1:
        raise ValueError('invalid connectivity mask')
    ids = node_ids.detach().cpu().tolist()
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate node identity')
    edges = endpoints.detach().cpu()[connected.detach().cpu()].tolist()
    if any(a == b or min(a, b) < 0 or max(a, b) >= len(ids) for a, b in edges):
        raise ValueError('invalid connected edge')
    result = []
    for _ in range(levels):
        adjacent = [[] for _ in ids]
        for a, b in edges:
            adjacent[a].append(b)
            adjacent[b].append(a)
        parent = [-1] * len(ids)
        coarse_ids = []
        for a in sorted(range(len(ids)), key=ids.__getitem__):
            if parent[a] >= 0:
                continue
            candidates = [b for b in adjacent[a] if parent[b] < 0]
            b = min(candidates, key=ids.__getitem__) if candidates else a
            parent[a] = parent[b] = len(coarse_ids)
            coarse_ids.append(min(ids[a], ids[b]))
        if len(coarse_ids) == len(ids):
            break
        coarse_edges = sorted({tuple(sorted((parent[a], parent[b])))
                               for a, b in edges if parent[a] != parent[b]})
        sizes = [0] * len(coarse_ids)
        for p in parent:
            sizes[p] += 1
        active = [size > 1 for size in sizes]
        for a, b in coarse_edges:
            active[a] = active[b] = True
        result.append(Level(torch.tensor(parent, dtype=torch.long),
                            torch.tensor(coarse_edges, dtype=torch.long).reshape(-1, 2),
                            torch.tensor(active, dtype=torch.bool),
                            len(coarse_ids)))
        ids, edges = coarse_ids, coarse_edges
    return result


class HierarchyCache:
    """One-entry exact topology cache, invalidated by every identity/edge change."""
    def __init__(self):
        self.key = None
        self.value = None
        self.builds = 0

    def get(self, node_ids, endpoints, connected, levels):
        tensors = (node_ids, endpoints, connected)
        if self.key is None or self.key[0] != levels or any(
                a.device != b.device or not torch.equal(a, b)
                for a, b in zip(tensors, self.key[1:])):
            value = build_hierarchy(*tensors, levels)
            self.value = [level.to(node_ids.device) for level in value]
            self.key = (levels, *(x.detach().clone() for x in tensors))
            self.builds += 1
        return self.value


def mean_pool(node, parent, size):
    output = node.new_zeros(size, node.shape[-1])
    output.index_add_(0, parent, node)
    count = node.new_zeros(size)
    count.index_add_(0, parent, node.new_ones(len(node)))
    return output / count.clamp_min(1)[:, None]


class BondBlock(nn.Module):
    """Update every cached edge, communicate only over connected material."""
    def __init__(self, width):
        super().__init__()
        self.edge = MLP(3 * width, width, width)
        self.node = MLP(2 * width, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, node, edge, endpoints, connected):
        a, b = endpoints.unbind(-1)
        edge = edge + self.edge(torch.cat((node[a], node[b], edge), -1))
        active = edge * connected[:, None]
        aggregate = torch.zeros_like(node)
        aggregate.index_add_(0, a, active)
        aggregate.index_add_(0, b, active)
        return self.norm(node + self.node(torch.cat((node, aggregate), -1))), edge


class CoarseBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.message = MLP(2 * width, width, width)
        self.update = MLP(2 * width, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, node, endpoints):
        a, b = endpoints.unbind(-1)
        # Symmetric construction avoids dependence on coarse edge orientation.
        message = self.message(torch.cat((node[a] + node[b], (node[a] - node[b]).abs()), -1))
        aggregate = torch.zeros_like(node)
        aggregate.index_add_(0, a, message)
        aggregate.index_add_(0, b, message)
        return self.norm(node + self.update(torch.cat((node, aggregate), -1)))


class MultiscaleProcessor(nn.Module):
    """Shared weights across scales; fine topology and edge outputs are retained."""
    def __init__(self, width):
        super().__init__()
        self.pre = BondBlock(width)
        self.coarse = CoarseBlock(width)
        self.prolong = MLP(2 * width, width, width)
        self.post = BondBlock(width)

    def forward(self, node, edge, endpoints, connected, hierarchy):
        node, edge = self.pre(node, edge, endpoints, connected)
        saved = []
        for level in hierarchy:
            saved.append(node)
            node = mean_pool(node, level.parent, level.size)
            node = torch.where(level.active[:, None], self.coarse(node, level.endpoints), node)
        for level, fine in zip(reversed(hierarchy), reversed(saved)):
            correction = self.prolong(torch.cat((fine, node[level.parent]), -1))
            node = fine + correction * level.active[level.parent, None]
        return self.post(node, edge, endpoints, connected)


@dataclass(frozen=True)
class EdgeMemory:
    scene: str
    next_time: float
    ids: torch.Tensor
    values: torch.Tensor

    def detach(self):
        return EdgeMemory(self.scene, self.next_time, self.ids, self.values.detach())


def align_memory(state, ids, *, scene, time, width, like):
    """Differentiable stable-ID remap; retired IDs disappear, new IDs start at zero."""
    if ids.dtype != torch.long or ids.ndim != 1 or len(torch.unique(ids)) != len(ids):
        raise ValueError('invalid or duplicate bond identity')
    result = like.new_zeros(len(ids), width)
    if state is None:
        return result
    if not math.isfinite(time) or not math.isfinite(state.next_time) or state.scene != scene or abs(state.next_time - time) > 1e-9:
        raise ValueError('memory scene/clock mismatch; start a new sequence explicitly')
    if state.values.shape != (len(state.ids), width) or not torch.isfinite(state.values).all():
        raise ValueError('invalid recurrent state')
    if len(torch.unique(state.ids)) != len(state.ids):
        raise ValueError('duplicate stored bond identity')
    if len(state.ids):
        sorted_ids, order = torch.sort(state.ids)
        position = torch.searchsorted(sorted_ids, ids)
        safe = position.clamp_max(len(sorted_ids) - 1)
        found = (position < len(sorted_ids)) & (sorted_ids[safe] == ids)
        result = torch.where(found[:, None], state.values[order[safe]], result)
    return result


class RecurrentGraphProcessor(nn.Module):
    def __init__(self, width=48, memory=16, recurrent=True):
        super().__init__()
        if width < 2 or memory < 1:
            raise ValueError('invalid recurrent dimensions')
        self.width, self.memory_width, self.recurrent = width, memory, recurrent
        self.processor = MultiscaleProcessor(width)
        self.memory_in = nn.Linear(memory, width, bias=False)
        self.cell = nn.GRUCell(width, memory)
        self.memory_out = nn.Linear(memory, width, bias=False)

    def forward(self, node, edge, endpoints, connected, hierarchy, ids,
                *, scene, time, dt, state=None):
        if not scene or not math.isfinite(time) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('invalid sequence identity/clock')
        previous = align_memory(state, ids, scene=scene, time=time,
                                width=self.memory_width, like=edge)
        if not self.recurrent:
            previous = torch.zeros_like(previous)
        node, edge = self.processor(node, edge + self.memory_in(previous),
                                    endpoints, connected, hierarchy)
        memory = self.cell(edge, previous)
        if self.recurrent:
            edge = edge + self.memory_out(memory)
        else:
            memory = torch.zeros_like(memory)
        return node, edge, EdgeMemory(scene, time + dt, ids.clone(), memory)
