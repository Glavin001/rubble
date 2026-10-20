# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA PhysicsNeMo @ c8f68428840a559f4b86ef56e96000b80a8d9594,
# physicsnemo/datapipes/gnn/bsms.py (BistrideMultiLayerGraph, Graph), Apache License 2.0
# (LICENSE-Apache-2.0.txt), itself modified from BSMS-GNN
# (https://github.com/Eydcao/BSMS-GNN, Apache License 2.0).
#
# Changes:
# - breadth-first search on a `collections.deque` (the original pops list heads: O(n²));
#   clusters from one labelling pass;
# - no scipy: the two-hop join (A+I)² is enumerated directly. The edge set and its order
#   (lexicographic in the coarse ids) are the same as the original's CSR product;
# - each coarse edge records its *support*: the fine paths (direct, or through one middle
#   node) it was joined from. `live_edges` uses it to mask a coarse edge once no fine path
#   beneath it is intact, so no level carries messages across a crack (ours; BSMS has no
#   masking);
# - the hierarchy is built once per pristine structure; `restrict` maps it onto the chunks
#   present in a call.
"""Bi-stride multi-level graph (BSMS-GNN pooling hierarchy) with crack masking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

_INF = 1 + 1e10


def adjacency(edges: np.ndarray, n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(edges[0].tolist(), edges[1].tolist()):
        adj[a].append(b)
    return adj


def bfs_dist(adj: list[list[int]], seed: int) -> np.ndarray:
    res = np.full(len(adj), _INF)
    res[seed] = 0
    q = deque([seed])
    while q:
        f = q.popleft()
        d = res[f] + 1
        for n in adj[f]:
            if res[n] == _INF:
                res[n] = d
                q.append(n)
    return res


def find_clusters(adj: list[list[int]]) -> list[list[int]]:
    label = np.full(len(adj), -1, dtype=np.int64)
    clusters: list[list[int]] = []
    for s in range(len(adj)):
        if label[s] >= 0:
            continue
        c = len(clusters)
        label[s] = c
        q = deque([s])
        members = [s]
        while q:
            f = q.popleft()
            for n in adj[f]:
                if label[n] < 0:
                    label[n] = c
                    q.append(n)
                    members.append(n)
        clusters.append(sorted(members))
    return clusters


def nearest_center_seed(pos: np.ndarray, clusters: list[list[int]]) -> list[int]:
    seeds = []
    for c in clusters:
        center = np.mean(pos[c], axis=0)
        dist = np.linalg.norm(pos[c] - center[None, :], 2, axis=-1)
        seeds.append(c[int(np.argmin(dist))])
    return seeds


@dataclass
class Level:
    """Level l+1 built from level l. `ids` index level-l nodes; `edges` are level-(l+1)
    directed edges in level-(l+1) ids. Edge `k` is supported by fine paths
    `(sup_a[s], sup_b[s])` for `s` with `sup_owner[s] == k` (level-l edge indices; a direct
    edge has `sup_a == sup_b`)."""

    ids: np.ndarray
    edges: np.ndarray
    sup_owner: np.ndarray
    sup_a: np.ndarray
    sup_b: np.ndarray


def bstride_selection(edges: np.ndarray, pos: np.ndarray, n: int) -> Level:
    adj = adjacency(edges, n)
    clusters = find_clusters(adj)
    seeds = nearest_center_seed(pos, clusters)
    kept: set[int] = set()
    for seed, _c in zip(seeds, clusters):
        dist = bfs_dist(adj, seed)
        even = {i for i, d in enumerate(dist.tolist()) if d != _INF and d % 2 == 0}
        odd = {i for i, d in enumerate(dist.tolist()) if d != _INF and d % 2 == 1}
        kept |= even if (len(even) <= len(odd) or not odd) else odd
    ids = np.array(sorted(kept), dtype=np.int64)
    new_id = -np.ones(n, dtype=np.int64)
    new_id[ids] = np.arange(len(ids))
    # level-l edge index for (a, b)
    eidx = {(a, b): k for k, (a, b) in enumerate(zip(edges[0].tolist(), edges[1].tolist()))}
    support: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a in ids.tolist():
        for b in adj[a]:  # direct (A·I and I·A terms)
            if new_id[b] >= 0 and b != a:
                support.setdefault((a, b), []).append((eidx[(a, b)], eidx[(a, b)]))
        for m in adj[a]:  # two hops through m (A·A term)
            for b in adj[m]:
                if b != a and new_id[b] >= 0:
                    support.setdefault((a, b), []).append((eidx[(a, m)], eidx[(m, b)]))
    keys = sorted(support, key=lambda ab: (new_id[ab[0]], new_id[ab[1]]))
    e_new = np.array([[new_id[a] for a, _ in keys], [new_id[b] for _, b in keys]],
                     dtype=np.int64).reshape(2, -1)
    owner, sa, sb = [], [], []
    for k, ab in enumerate(keys):
        for p, q in support[ab]:
            owner.append(k)
            sa.append(p)
            sb.append(q)
    return Level(ids=ids, edges=e_new, sup_owner=np.array(owner, dtype=np.int64),
                 sup_a=np.array(sa, dtype=np.int64), sup_b=np.array(sb, dtype=np.int64))


@dataclass
class Hierarchy:
    """Per pristine structure: level-0 directed edges plus `levels` coarse levels."""

    n_nodes: int
    edges0: np.ndarray  # [2, E0] directed
    edge0_bond: np.ndarray  # [E0] pristine bond index of each directed edge
    pos: np.ndarray  # [N, 3] rest positions (seed choice only)
    levels: list[Level]


def build_hierarchy(n_nodes: int, bond_ends: np.ndarray, pos: np.ndarray,
                    num_levels: int) -> Hierarchy:
    """`bond_ends` [B, 2] undirected pristine bonds (local chunk indices)."""
    b = np.asarray(bond_ends, dtype=np.int64)
    edges0 = np.stack([np.concatenate([b[:, 0], b[:, 1]]), np.concatenate([b[:, 1], b[:, 0]])])
    edge0_bond = np.concatenate([np.arange(len(b)), np.arange(len(b))])
    levels: list[Level] = []
    g, p, n = edges0, np.asarray(pos, dtype=np.float64), n_nodes
    for _ in range(num_levels):
        lv = bstride_selection(g, p, n)
        levels.append(lv)
        g, p, n = lv.edges, p[lv.ids], len(lv.ids)
    return Hierarchy(n_nodes=n_nodes, edges0=edges0, edge0_bond=edge0_bond, pos=pos,
                     levels=levels)


def live_edges(h: Hierarchy, bond_live: np.ndarray, node_present: np.ndarray
               ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per level, the live mask of its directed edges and the present mask of its nodes.

    Level 0: a directed edge is live iff its bond is intact and both ends are present.
    Level l+1: an edge is live iff some supporting fine path is entirely live."""
    live = bond_live[h.edge0_bond] & node_present[h.edges0[0]] & node_present[h.edges0[1]]
    edge_masks, node_masks = [live], [node_present.copy()]
    present = node_present
    for lv in h.levels:
        path = live[lv.sup_a] & live[lv.sup_b]
        nxt = np.zeros(lv.edges.shape[1], dtype=bool)
        np.logical_or.at(nxt, lv.sup_owner, path)
        present = present[lv.ids]
        nxt &= present[lv.edges[0]] & present[lv.edges[1]]
        edge_masks.append(nxt)
        node_masks.append(present.copy())
        live = nxt
    return edge_masks, node_masks
