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
    n_fragments: int = 0  # distinct y_chunk_fragment ids (host-side, so losses need no sync)

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
                     [mv(x) for x in self.h_chunk], self.calls, self.n_fragments)

    def clone_inputs(self) -> "Batch":
        b = Batch(self.n_calls, dict(self.t), {k: dict(d) for k, d in self.edges.items()},
                  self.h_edges, self.h_ids, self.h_chunk, self.calls, self.n_fragments)
        for k in ("chunk_x", "imp_x"):
            b.t[k] = self.t[k].clone()
        for d in b.edges.values():
            d["x"] = d["x"].clone()
        return b


def _f(a: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _i(a: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64))


# call_piece's cache: id(call) → (weak reference, fingerprint, levels, piece). Kept off the
# Call itself (copies of a call's __dict__ must not carry it); the fingerprint (the identity of
# every array the piece reads) makes a call whose arrays were replaced recompute.
_PIECES: dict[int, tuple] = {}
_PIECE_KEYS = ("bond_key", "chunk_key", "y_bond_break_time", "y_chunk_fragment", "y_chunk_def_end",
               "fs_bond_in", "fs_chunk_cluster_out", "fs_cluster_out")


def _fingerprint(c: S.Call) -> tuple:
    a = c.arrays
    return (id(a), id(c.fs), *(id(a.get(k)) for k in _PIECE_KEYS))


def clear_piece_cache() -> None:
    _PIECES.clear()


def call_piece(c: S.Call, structures: dict[str, dict[int, dict[str, np.ndarray]]],
               cache: HierarchyCache) -> dict:
    """Everything `collate` derives from one call alone, with call-local indices: the `/5`
    full-state encodings, the cluster grouping, the break-time fraction and the hierarchy's
    level ids and edges. A pure function of the call (and its event's structures), so it is
    computed once and cached on the call; `collate` then only offsets and concatenates
    (bit-identical to computing it every batch). Noise and pushforward make new calls, which
    get their own pieces."""
    import weakref

    fp = _fingerprint(c)
    hit = _PIECES.get(id(c))
    if hit is not None and hit[0]() is c and hit[1] == fp and hit[2] == cache.num_levels:
        return hit[3]
    from . import fullstate as FS

    a = c.arrays
    nc, ni, nb = c.counts["chunks"], c.counts["impactors"], c.counts["bond"]
    p: dict = {"nc": nc, "ni": ni, "nb": nb}
    p["full_state"] = c.fs is not None and "fs_bond_in" in a
    if p["full_state"]:
        bx, by, bm = FS.bond_arrays(c, c.fs["cols"], c.fs["scale"], c.fs["scale_names"])
        hs = FS.hidden_scale(c)
        yd = a["y_chunk_def_end"] if "y_chunk_def_end" in a else np.zeros((nc, 12), np.float32)
        ci, co = FS.chunk_arrays(c, c.fs["cols"], float(np.linalg.norm(c.gravity)))
        p["fs"] = {"bx": bx, "by": by, "bm": bm,
                   "hid": (yd.astype(np.float64) / hs).astype(np.float32), "ci": ci, "co": co}
        if "fs_chunk_cluster_out" in a and "fs_cluster_out" in a:
            cg = FS.cluster_groups(c, a["fs_chunk_cluster_out"])
            cy, cact = FS.cluster_targets(c, cg)
            p["clu"] = {"of_chunk": cg["of_chunk"].astype(np.int64), "x": cg["x"], "key": cg["key"],
                        "y": cy, "act": cact}
    bt = a.get("y_bond_break_time")
    p["break_frac"] = ((np.asarray(bt, np.float64) / float(c.scales["dt"])).astype(np.float32)
                       if bt is not None else np.full(nb, np.nan, np.float32))
    p["n_frag"] = int(a["y_chunk_fragment"].max(initial=-1)) + 1
    # hierarchy, call-local: level-0 ids are the call's chunk positions; level-l ids count
    # from 0 within the call (collate adds the batch offsets where an id is ≥ 0)
    L = cache.num_levels
    hier_rows = []  # per structure: (level-0 edges, [(ids, chunk, edges) per level ≥ 1])
    h_count = [0] * (L + 1)
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
        comp = [np.where(present, call_idx, -1)]
        chunk_of = [np.where(present, call_idx, -1)]
        e0 = hier.edges0[:, emask[0]]
        lv_rows = []
        for li, lv in enumerate(hier.levels, start=1):
            pres = nmask[li]
            cid = -np.ones(len(lv.ids), dtype=np.int64)
            cid[pres] = h_count[li] + np.arange(int(pres.sum()))
            h_count[li] += int(pres.sum())
            prev = comp[li - 1][lv.ids]
            ch = chunk_of[li - 1][lv.ids]
            el = lv.edges[:, emask[li]]
            lv_rows.append((prev[pres], ch[pres], np.stack([cid[el[0]], cid[el[1]]])))
            comp.append(cid)
            chunk_of.append(ch)
        hier_rows.append((np.stack([comp[0][e0[0]], comp[0][e0[1]]]), lv_rows))
    p["hier"], p["h_count"] = hier_rows, h_count
    if len(_PIECES) > 4 * 65536:  # drop the entries of calls that are gone
        for k in [k for k, v in _PIECES.items() if v[0]() is None]:
            del _PIECES[k]
    try:
        _PIECES[id(c)] = (weakref.ref(c), fp, L, p)
    except TypeError:  # not weak-referenceable: no cache
        pass
    return p


def _off(x: np.ndarray, o: int) -> np.ndarray:
    """Batch offset for a call-local id array, keeping −1 (absent) as is."""
    return np.where(x >= 0, x + o, x) if o else x


def collate(calls: list[S.Call], structures: dict[str, dict[int, dict[str, np.ndarray]]],
            cache: HierarchyCache) -> Batch:
    """`structures[event_id]` is the event's pristine structure graph. Bit-identical to
    building every array per batch (tests/collate_reference.py): the call-local work is
    `call_piece`, cached on each call."""
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
    pieces = [call_piece(c, structures, cache) for c in calls]
    # full chunk inertia only when every call carries it (else the diagonal fallback)
    full_inertia = all("chunk_inertia_struct" in c.arrays and "chunk_body_q" in c.arrays
                       for c in calls)
    full_state = bool(calls) and all(p["full_state"] for p in pieces)
    clu_off = 0
    for gi, (c, p) in enumerate(zip(calls, pieces)):
        a = c.arrays
        nc, ni, nb = p["nc"], p["ni"], p["nb"]
        if full_state:
            f = p["fs"]
            add("y_bond_fs_start", f["bx"][:, :len(_fs_bond_fields())])
            add("y_bond_fs", f["by"])
            add("y_bond_fs_mode", f["bm"])
            e_acc["bond"].setdefault("fs", []).append(np.repeat(f["bx"], 2, 0))  # both twins
            add("y_chunk_hid", f["hid"])
            add("chunk_fs_in", f["ci"])
            add("y_chunk_fs", f["co"])
        if full_state and "clu" in p:
            # frame-end bodies, grouped as the oracle's clusters (training targets; a live
            # call has none: inference groups by the decoded fragments in state_io)
            cl = p["clu"]
            add("clu_of_chunk", cl["of_chunk"] + clu_off)
            add("clu_x", cl["x"])
            add("clu_call", np.full(len(cl["key"]), gi))
            add("clu_key", cl["key"])
            add("y_clu", cl["y"])
            add("y_clu_act", cl["act"])
            clu_off += len(cl["key"])
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
        if full_inertia:  # ChunkData.inertia (structure body frame) and the body rotation
            add("chunk_inertia_struct", a["chunk_inertia_struct"])
            add("chunk_body_q", a["chunk_body_q"])
        for k in ("v_end", "w_end", "dx", "drot"):
            add(f"y_chunk_{k}", a[f"y_chunk_{k}"])
            add(f"y_imp_{k}", a[f"y_impactor_{k}"])
        add("y_chunk_fragment", a["y_chunk_fragment"].astype(np.int64) + frag_off)
        frag_off += p["n_frag"]
        add("y_imp_regime", a["y_impactor_regime"])
        has_pf = "y_impactor_perforate_freq" in a
        add("y_imp_perf_freq", a["y_impactor_perforate_freq"] if has_pf else np.zeros(ni))
        add("y_imp_has_freq", np.full(ni, has_pf))
        if "ens_impactor_v_end" in a:
            ens_rows.append((imp_off, a["ens_impactor_v_end"]))
        for k in ("peak_util", "fail_margin", "broke", "break_bin", "fail_mode", "ddamage",
                  "dissipated"):
            add(f"y_bond_{k}", a[f"y_bond_{k}"])
        add("y_bond_support_path", a.get("y_bond_support_path", np.zeros(nb)))
        # exact break time ÷ dt (NaN where unknown or unbroken); masked in the loss
        add("y_bond_break_frac", p["break_frac"])
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
        add("energy", np.array([[en.get("KE_in", np.nan), en["KE_out"], en["W_gravity"], en["W_external"],
                                 en["E_elastic_end"], en["D_bond"],
                                 en.get("D_bond_joint", 0.0), en.get("D_bond_damping", 0.0)]]))
        add("energy_has_split", np.array([("D_bond_joint" in en) and ("D_bond_damping" in en)]))
        # the joint law's frame-start stored energy (an input-side quantity; NaN if absent):
        # the decode's energy envelope
        es, ki = en.get("E_elastic_start"), en.get("KE_in")
        add("e_stored_start", np.array([float(es) if es is not None else np.nan]))
        add("e_ke_in", np.array([float(ki) if ki is not None else np.nan]))  # the builder's f64
        # the event's energy budget left before this call (schema.energy_budget; live calls:
        # the host's); NaN without a ledger: the decode's per-frame bound applies instead
        ec = en.get("E_contact_start")  # the contacts' stored energy at frame start (P0)
        add("e_contact_start", np.array([float(ec) if ec is not None else 0.0]))
        rb = en.get("E_budget_remaining")
        add("e_budget_remaining", np.array([float(rb) if rb is not None else np.nan]))
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
        for e0, lv_rows in p["hier"]:
            h_edges[0].append(_off(e0, chunk_off))
            for li, (ids, ch, el) in enumerate(lv_rows, start=1):
                h_ids[li].append(_off(ids, chunk_off if li == 1 else h_count[li - 1]))
                h_chunk[li].append(_off(ch, chunk_off))
                h_edges[li].append(_off(el, h_count[li]))
        for li in range(1, L + 1):
            h_count[li] += p["h_count"][li]
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
    b.t["e_stored_start"] = torch.from_numpy(np.concatenate(acc["e_stored_start"]).astype(np.float64))
    b.t["e_ke_in"] = torch.from_numpy(np.concatenate(acc["e_ke_in"]).astype(np.float64))
    b.t["e_contact_start"] = torch.from_numpy(np.concatenate(acc["e_contact_start"]).astype(np.float64))
    b.t["e_budget_remaining"] = torch.from_numpy(np.concatenate(acc["e_budget_remaining"]).astype(np.float64))
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
        if "fs" in d:  # `/5` full-state bond inputs, per directed twin
            b.edges[t]["fs"] = _f(np.concatenate(d["fs"]))
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
    b.n_fragments = frag_off
    return add_valid_masks(b)


def _fs_bond_fields():
    from . import fullstate as FS
    return FS.BOND_FIELDS


def collate_events(events: list[S.Event], cache: HierarchyCache,
                   calls: list[S.Call] | None = None) -> Batch:
    st = {e.manifest["event_id"]: e.structures for e in events}
    return collate(calls if calls is not None else [c for e in events for c in e.calls], st,
                   cache)


# ---- shape bucketing -------------------------------------------------------------------
# MPS compiles a kernel graph per op and per tensor shape: every new batch shape costs about
# 1 s (measured on P1's calls: 1,167 ms/step with new shapes against 132–158 ms with
# repeated ones). `pad_batch` rounds every dimension up to a geometric bucket so a few
# shapes repeat. Padding lives in extra calls (index ≥ len(b.calls)) that are anchored and
# masked out of every loss (`*_valid`); metrics only iterate `b.calls`, so they never see it.
PAD_SCALES = {"L": 1.0, "m": 1.0, "v": 1.0, "c": 1.0, "f_t": 1.0, "E": 1.0, "GA": 1.0,
              "dt": 1.0 / 60.0, "E_in": 1.0}


def bucket(n: int, growth: float = 1.25, minimum: int = 64) -> int:
    """Smallest `ceil(minimum · growth^k)` ≥ n."""
    size = float(minimum)
    while int(np.ceil(size)) < n:
        size *= growth
    return int(np.ceil(size))


def _pad_rows(x: Tensor, n: int, value=0) -> Tensor:
    if n <= 0:
        return x
    pad = torch.full((n,) + tuple(x.shape[1:]), value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], 0)


def add_valid_masks(b: Batch) -> Batch:
    t = b.t
    if "chunk_valid" not in t:
        t["chunk_valid"] = torch.ones(t["chunk_node"].shape[0], dtype=torch.bool)
        t["imp_valid"] = torch.ones(t["imp_node"].shape[0], dtype=torch.bool)
        t["bond_valid"] = torch.ones(t["bond_call"].shape[0], dtype=torch.bool)
        t["call_valid"] = torch.ones(b.n_calls, dtype=torch.bool)
    return b


def pad_batch(b: Batch, growth: float = 1.25, call_multiple: int = 8) -> Batch:
    """Pad a collated batch (in place) to bucketed sizes. Exact: real entries are
    untouched, and every loss and metric ignores the padding."""
    add_valid_masks(b)
    t = b.t
    G, Nc, Ni = b.n_calls, t["chunk_node"].shape[0], t["imp_node"].shape[0]
    N = int(t["n_nodes"])
    Eb = t["bond_call"].shape[0]
    G2 = int(np.ceil((G + 1) / call_multiple) * call_multiple)
    pc = bucket(Nc + 2, growth) - Nc
    pi = (bucket(Ni, growth, 16) - Ni) if Ni else 16
    pb = bucket(Eb, growth) - Eb
    pg, pcall = G2 - G, G  # padding calls; the first one holds every padding entity
    # ---- calls
    t["glob_x"] = _pad_rows(t["glob_x"], pg)
    sc = torch.tensor([[PAD_SCALES[k] for k in S.SCALE_KEYS]], dtype=t["scales"].dtype)
    t["scales"] = torch.cat([t["scales"], sc.expand(pg, -1)], 0)
    t["energy"] = _pad_rows(t["energy"], pg)
    t["gravity"] = _pad_rows(t["gravity"], pg)
    t["energy_has_split"] = _pad_rows(t["energy_has_split"], pg, False)
    t["e_stored_start"] = _pad_rows(t["e_stored_start"], pg, float("nan"))
    t["e_ke_in"] = _pad_rows(t["e_ke_in"], pg, float("nan"))
    t["e_budget_remaining"] = _pad_rows(t["e_budget_remaining"], pg, float("nan"))
    t["e_contact_start"] = _pad_rows(t["e_contact_start"], pg, 0.0)
    t["call_anchored"] = _pad_rows(t["call_anchored"], pg, True)
    t["call_valid"] = _pad_rows(t["call_valid"], pg, False)
    # ---- nodes: padding chunks then padding impactors, after every real node
    c_nodes = torch.arange(N, N + pc)
    i_nodes = torch.arange(N + pc, N + pc + pi)
    t["node_call"] = torch.cat([t["node_call"], torch.full((pc + pi,), pcall, dtype=torch.long)])
    t["n_nodes"] = torch.tensor(N + pc + pi)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for pre, n, nodes in (("chunk", pc, c_nodes), ("imp", pi, i_nodes)):
        t[f"{pre}_node"] = torch.cat([t[f"{pre}_node"], nodes])
        t[f"{pre}_call"] = _pad_rows(t[f"{pre}_call"], n, pcall)
        t[f"{pre}_valid"] = _pad_rows(t[f"{pre}_valid"], n, False)
        t[f"{pre}_x"] = _pad_rows(t[f"{pre}_x"], n)
        t[f"{pre}_qs"] = _pad_rows(t[f"{pre}_qs"], n)
        t[f"{pre}_mass"] = _pad_rows(t[f"{pre}_mass"], n, 1.0)
        t[f"{pre}_inertia_body"] = _pad_rows(t[f"{pre}_inertia_body"], n, 1.0)
        t[f"{pre}_q_world"] = torch.cat([t[f"{pre}_q_world"], ident.expand(n, 4)], 0)
        if pre == "chunk" and "chunk_inertia_struct" in t:
            unit = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=t["chunk_inertia_struct"].dtype)
            t["chunk_inertia_struct"] = torch.cat([t["chunk_inertia_struct"], unit.expand(n, 6)], 0)
            t["chunk_body_q"] = torch.cat([t["chunk_body_q"], ident.expand(n, 4)
                                           .to(t["chunk_body_q"].dtype)], 0)
        for k in ("pos", "v", "w"):
            t[f"{pre}_{k}"] = _pad_rows(t[f"{pre}_{k}"], n)
        for k in ("v_end", "w_end", "dx", "drot"):
            t[f"y_{pre}_{k}"] = _pad_rows(t[f"y_{pre}_{k}"], n)
    t["chunk_anchored"] = _pad_rows(t["chunk_anchored"], pc, True)
    for k in ("chunk_fs_in", "y_chunk_fs", "y_chunk_hid"):
        if k in t:
            t[k] = _pad_rows(t[k], pc)
    if "clu_of_chunk" in t:  # padding chunks form one padding body (masked out)
        n_cl = t["clu_x"].shape[0]
        t["clu_valid"] = torch.cat([torch.ones(n_cl, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)])
        t["clu_of_chunk"] = torch.cat([t["clu_of_chunk"], torch.full((pc,), n_cl, dtype=torch.long)])
        for k, fill in (("clu_x", 0.0), ("y_clu", 0.0)):
            t[k] = _pad_rows(t[k], 1, fill)
        t["clu_call"] = _pad_rows(t["clu_call"], 1, pcall)
        t["clu_key"] = _pad_rows(t["clu_key"], 1, -1)
        t["y_clu_act"] = _pad_rows(t["y_clu_act"], 1, 0)
    f0 = b.n_fragments
    t["y_chunk_fragment"] = torch.cat([t["y_chunk_fragment"], torch.arange(f0, f0 + pc)])
    b.n_fragments = f0 + pc
    t["y_imp_regime"] = _pad_rows(t["y_imp_regime"], pi, S.REGIME_UNDECIDED)
    t["y_imp_perf_freq"] = _pad_rows(t["y_imp_perf_freq"], pi)
    t["y_imp_has_freq"] = _pad_rows(t["y_imp_has_freq"], pi, False)
    t["ens_imp_v_end"] = _pad_rows(t["ens_imp_v_end"], pi)
    t["ens_imp_mask"] = _pad_rows(t["ens_imp_mask"], pi, False)
    for k in [k for k in t if k.startswith("base_")]:
        n = pc if "_chunk_" in k else pi if "_impactor_" in k else pb
        t[k] = _pad_rows(t[k], n)
    # ---- bonds (undirected targets and directed twins): padding chunk 0 ↔ padding chunk 1
    a, c_ = int(c_nodes[0]), int(c_nodes[1])
    for k in [k for k in t if k.startswith("y_bond_")]:
        t[k] = _pad_rows(t[k], pb)
    t["bond_call"] = _pad_rows(t["bond_call"], pb, pcall)
    t["bond_key"] = _pad_rows(t["bond_key"], pb, -1)
    t["bond_valid"] = _pad_rows(t["bond_valid"], pb, False)
    t["bond_qs"] = _pad_rows(t["bond_qs"], 2 * pb)
    for et, d in b.edges.items():
        n_und = pb if et == "bond" else (bucket(d["src"].shape[0] // 2, growth, 16)
                                        - d["src"].shape[0] // 2)
        twin_s = torch.tensor([a, c_] * n_und, dtype=torch.long)
        twin_d = torch.tensor([c_, a] * n_und, dtype=torch.long)
        d["src"] = torch.cat([d["src"], twin_s])
        d["dst"] = torch.cat([d["dst"], twin_d])
        d["x"] = _pad_rows(d["x"], 2 * n_und)
        if "fs" in d:
            d["fs"] = _pad_rows(d["fs"], 2 * n_und)
        d["call"] = _pad_rows(d["call"], 2 * n_und, pcall)
        d["frame"] = torch.cat([d["frame"], torch.eye(3).expand(n_und, 3, 3)], 0)
    # ---- hierarchy: level 0 is chunk-indexed; deeper levels pad onto padding nodes above
    pad_prev = torch.arange(Nc, Nc + pc)  # level-0 compact ids of the padding chunks
    for li in range(len(b.h_edges)):
        e = b.h_edges[li]
        if li > 0:
            n_real = b.h_ids[li].shape[0]
            n_pad = bucket(n_real + 2, growth) - n_real
            reps = pad_prev[torch.arange(n_pad) % len(pad_prev)]
            b.h_ids[li] = torch.cat([b.h_ids[li], reps])
            b.h_chunk[li] = torch.cat([b.h_chunk[li], torch.full((n_pad,), Nc, dtype=torch.long)])
            pad_prev = torch.arange(n_real, n_real + n_pad)
        else:
            b.h_chunk[0] = torch.arange(Nc + pc)
        p0, p1 = int(pad_prev[0]), int(pad_prev[1])
        n_e = bucket(e.shape[1] + 2, growth) - e.shape[1]
        pad_e = torch.tensor([[p0, p1] * (n_e // 2) + [p0] * (n_e % 2),
                              [p1, p0] * (n_e // 2) + [p1] * (n_e % 2)], dtype=torch.long)
        b.h_edges[li] = torch.cat([e, pad_e], 1)
    b.n_calls = G2
    return b
