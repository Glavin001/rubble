"""Collate calls into one batched graph (CSR-style concatenation, as the engine will).

Node layout: for each call in order, its chunks then its impactors. Directed edge twins
stay adjacent, so undirected edge `k` of the batch is rows `2k, 2k+1` of each edge type.
The BSMS hierarchy is built once per pristine structure (cached) and masked per call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from ..mgn.bsms_graph import Hierarchy, build_hierarchy, live_edges
from . import schema as S

SCALE_INDEX = {k: i for i, k in enumerate(S.SCALE_KEYS)}


class HierarchyCache:
    def __init__(self, num_levels: int):
        self.num_levels = num_levels
        self._cache: dict[tuple[str, int], Hierarchy] = {}

    def get(self, event_id: str, sid: int, st: dict[str, np.ndarray]) -> Hierarchy:
        key = (event_id, sid)
        if key not in self._cache:
            self._cache[key] = build_hierarchy(len(st["chunk_key"]), st["bond_ends"],
                                               st["chunk_rest_pos"], self.num_levels)
        return self._cache[key]


@dataclass
class Batch:
    n_calls: int
    t: dict[str, Tensor] = field(default_factory=dict)
    edges: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    # hierarchy: per level l ≥ 0 its directed edges (compact ids); per l ≥ 1 the pooling
    # ids into level l−1 and the batch chunk index of each node
    h_edges: list[Tensor] = field(default_factory=list)
    h_ids: list[Tensor] = field(default_factory=list)
    h_chunk: list[Tensor] = field(default_factory=list)
    calls: list[S.Call] = field(default_factory=list)

    def to(self, device: torch.device | str) -> "Batch":
        dev = torch.device(device)

        def mv(x):
            if not isinstance(x, Tensor):
                return x
            if dev.type == "mps" and x.dtype == torch.float64:  # MPS has no float64
                x = x.float()
            return x.to(dev)

        return Batch(self.n_calls, {k: mv(v) for k, v in self.t.items()},
                     {k: {n: mv(a) for n, a in d.items()} for k, d in self.edges.items()},
                     [mv(x) for x in self.h_edges], [mv(x) for x in self.h_ids],
                     [mv(x) for x in self.h_chunk], self.calls)

    def clone_inputs(self) -> "Batch":
        b = Batch(self.n_calls, dict(self.t), {k: dict(d) for k, d in self.edges.items()},
                  self.h_edges, self.h_ids, self.h_chunk, self.calls)
        for k in ("chunk_x", "imp_x"):
            b.t[k] = self.t[k].clone()
        for d in b.edges.values():
            d["x"] = d["x"].clone()
        return b


def _f(a: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _i(a: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64))


def collate(calls: list[S.Call], structures: dict[str, dict[int, dict[str, np.ndarray]]],
            cache: HierarchyCache) -> Batch:
    """`structures[event_id]` is the event's pristine structure graph."""
    b = Batch(n_calls=len(calls), calls=list(calls))
    L = cache.num_levels
    acc: dict[str, list] = {}

    def add(name: str, arr):
        acc.setdefault(name, []).append(arr)

    e_acc: dict[str, dict[str, list]] = {t: {} for t in S.EDGE_TYPES}
    h_edges: list[list[np.ndarray]] = [[] for _ in range(L + 1)]
    h_ids: list[list[np.ndarray]] = [[] for _ in range(L + 1)]
    h_chunk: list[list[np.ndarray]] = [[] for _ in range(L + 1)]
    h_count = [0] * (L + 1)
    node_off = chunk_off = imp_off = frag_off = 0
    ens_rows = []
    for gi, c in enumerate(calls):
        a = c.arrays
        nc, ni = c.counts["chunks"], c.counts["impactors"]
        add("chunk_x", a["chunk_feat"])
        add("chunk_qs", a["chunk_qs_feat"])
        add("imp_x", a["impactor_feat"])
        add("imp_qs", a["impactor_qs_feat"])
        add("glob_x", a["global_feat"])
        add("chunk_node", node_off + np.arange(nc))
        add("imp_node", node_off + nc + np.arange(ni))
        add("chunk_call", np.full(nc, gi))
        add("imp_call", np.full(ni, gi))
        add("node_call", np.full(nc + ni, gi))
        for k in ("mass", "inertia_body", "q_world", "pos", "v", "w"):
            add(f"chunk_{k}", a[f"chunk_{k}"])
            add(f"imp_{k}", a[f"impactor_{k}"])
        add("chunk_anchored", a["chunk_anchored"])
        for k in ("v_end", "w_end", "dx", "drot"):
            add(f"y_chunk_{k}", a[f"y_chunk_{k}"])
            add(f"y_imp_{k}", a[f"y_impactor_{k}"])
        add("y_chunk_fragment", a["y_chunk_fragment"].astype(np.int64) + frag_off)
        frag_off += int(a["y_chunk_fragment"].max(initial=-1)) + 1
        add("y_imp_regime", a["y_impactor_regime"])
        has_pf = "y_impactor_perforate_freq" in a
        add("y_imp_perf_freq", a["y_impactor_perforate_freq"] if has_pf else np.zeros(ni))
        add("y_imp_has_freq", np.full(ni, has_pf))
        if "ens_impactor_v_end" in a:
            ens_rows.append((imp_off, a["ens_impactor_v_end"]))
        nb = c.counts["bond"]
        for k in ("peak_util", "broke", "break_bin", "fail_mode", "ddamage", "dissipated"):
            add(f"y_bond_{k}", a[f"y_bond_{k}"])
        add("y_bond_support_path", a.get("y_bond_support_path", np.zeros(nb)))
        has_bf = "y_bond_break_freq" in a
        add("y_bond_freq", a["y_bond_break_freq"] if has_bf else np.zeros(nb))
        add("y_bond_has_freq", np.full(nb, has_bf))
        add("bond_call", np.full(nb, gi))
        add("bond_key", a["bond_key"])
        add("bond_qs", a["bond_qs_feat"])
        for base in ("qs", "punch"):
            for k in ("bond_broke", "bond_peak_util", "chunk_v_end", "impactor_v_end"):
                name = f"base_{base}_{k}"
                if name in a:
                    add(name, a[name])
        add("scales", np.array([[c.scales[k] for k in S.SCALE_KEYS]]))
        en = c.energy
        add("energy", np.array([[en["KE_in"], en["KE_out"], en["W_gravity"], en["W_external"],
                                 en["E_elastic_end"], en["D_bond"],
                                 en.get("D_bond_joint", 0.0), en.get("D_bond_damping", 0.0)]]))
        add("energy_has_split", np.array([("D_bond_joint" in en) and ("D_bond_damping" in en)]))
        add("call_anchored", np.array([c.anchored]))
        add("gravity", np.array([c.gravity], dtype=np.float64))
        for t in S.EDGE_TYPES:
            d = e_acc[t]
            d.setdefault("src", []).append(a[f"{t}_src"].astype(np.int64) + node_off)
            d.setdefault("dst", []).append(a[f"{t}_dst"].astype(np.int64) + node_off)
            d.setdefault("x", []).append(a[f"{t}_feat"])
            d.setdefault("call", []).append(np.full(len(a[f"{t}_src"]), gi))
            d.setdefault("frame", []).append(a[f"{t}_frame"])

        # ---- hierarchy: level 0 = this call's chunks (batch chunk order) --------------
        ck = a["chunk_key"]
        pos_of_key = {int(k): i for i, k in enumerate(ck.tolist())}
        bond_keys = set(a["bond_key"].tolist())
        for sid, st in sorted(structures[c.event_id].items()):
            hier = cache.get(c.event_id, sid, st)
            call_idx = np.array([pos_of_key.get(int(k), -1) for k in st["chunk_key"].tolist()])
            present = call_idx >= 0
            if not present.any():
                continue
            bond_live = np.array([int(k) in bond_keys for k in st["bond_key"].tolist()])
            emask, nmask = live_edges(hier, bond_live, present)
            # compact id of each pristine node at each level (−1 if absent)
            comp = [np.where(present, call_idx + chunk_off, -1)]
            chunk_of = [np.where(present, call_idx + chunk_off, -1)]
            e0 = hier.edges0[:, emask[0]]
            h_edges[0].append(np.stack([comp[0][e0[0]], comp[0][e0[1]]]))
            for li, lv in enumerate(hier.levels, start=1):
                pres = nmask[li]
                cid = -np.ones(len(lv.ids), dtype=np.int64)
                cid[pres] = h_count[li] + np.arange(int(pres.sum()))
                h_count[li] += int(pres.sum())
                prev = comp[li - 1][lv.ids]  # level-(l−1) compact id of each kept node
                h_ids[li].append(prev[pres])
                ch = chunk_of[li - 1][lv.ids]
                h_chunk[li].append(ch[pres])
                el = lv.edges[:, emask[li]]
                h_edges[li].append(np.stack([cid[el[0]], cid[el[1]]]))
                comp.append(cid)
                chunk_of.append(ch)
        node_off += nc + ni
        chunk_off += nc
        imp_off += ni
    for k, v in acc.items():
        if k.startswith("base_") and len(v) != len(calls):
            continue  # a baseline is used only when every call in the batch has it
        arr = np.concatenate(v)
        b.t[k] = _i(arr) if arr.dtype.kind in "iub" else _f(arr)
    for k in ("chunk_anchored", "call_anchored", "y_bond_has_freq", "y_imp_has_freq",
              "energy_has_split"):
        b.t[k] = b.t[k].bool()
    for k in ("y_bond_support_path",):
        b.t[k] = b.t[k].float()
    b.t["energy"] = torch.from_numpy(np.concatenate(acc["energy"]).astype(np.float64))
    b.t["scales"] = torch.from_numpy(np.concatenate(acc["scales"]).astype(np.float64))
    b.t["gravity"] = torch.from_numpy(np.concatenate(acc["gravity"]).astype(np.float64))
    b.t["n_nodes"] = torch.tensor(node_off)
    for t in S.EDGE_TYPES:
        d = e_acc[t]
        b.edges[t] = {
            "src": _i(np.concatenate(d["src"])), "dst": _i(np.concatenate(d["dst"])),
            "x": _f(np.concatenate(d["x"])), "call": _i(np.concatenate(d["call"])),
            "frame": _f(np.concatenate(d["frame"])),
        }
    # ensemble members' impactor velocities, padded [Ni, M, 3] with a mask
    m_max = max((r[1].shape[0] for r in ens_rows), default=0)
    ens = np.zeros((imp_off, m_max, 3), np.float32)
    ens_mask = np.zeros((imp_off, m_max), bool)
    for off, arr in ens_rows:
        ens[off:off + arr.shape[1], :arr.shape[0]] = np.transpose(arr, (1, 0, 2))
        ens_mask[off:off + arr.shape[1], :arr.shape[0]] = True
    b.t["ens_imp_v_end"] = _f(ens)
    b.t["ens_imp_mask"] = torch.from_numpy(ens_mask)
    b.h_edges = [_i(np.concatenate(x, axis=1)) if x else torch.zeros(2, 0, dtype=torch.long)
                 for x in h_edges]
    b.h_ids = [torch.zeros(0, dtype=torch.long)] + [
        _i(np.concatenate(x)) if x else torch.zeros(0, dtype=torch.long) for x in h_ids[1:]]
    b.h_chunk = [torch.arange(chunk_off)] + [
        _i(np.concatenate(x)) if x else torch.zeros(0, dtype=torch.long) for x in h_chunk[1:]]
    return b


def collate_events(events: list[S.Event], cache: HierarchyCache,
                   calls: list[S.Call] | None = None) -> Batch:
    st = {e.manifest["event_id"]: e.structures for e in events}
    return collate(calls if calls is not None else [c for e in events for c in e.calls], st,
                   cache)
