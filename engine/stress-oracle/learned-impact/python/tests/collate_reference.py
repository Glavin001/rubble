"""A verbatim copy of `batch.collate` before the per-call piece cache (2026-10-22): the
parity reference for the cached path. Never used outside tests."""
# ruff: noqa
import numpy as np
import torch

from learned_impact.outcome import schema as S
from learned_impact.mgn.bsms_graph import live_edges
from learned_impact.outcome.batch import Batch, HierarchyCache, _f, _i, add_valid_masks


def collate_reference(calls: list[S.Call], structures: dict[str, dict[int, dict[str, np.ndarray]]],
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
    # full chunk inertia only when every call carries it (else the diagonal fallback)
    full_inertia = all("chunk_inertia_struct" in c.arrays and "chunk_body_q" in c.arrays
                       for c in calls)
    from learned_impact.outcome import fullstate as FS
    full_state = bool(calls) and all(c.fs is not None and "fs_bond_in" in c.arrays for c in calls)
    clu_off = 0
    for gi, c in enumerate(calls):
        a = c.arrays
        nc, ni = c.counts["chunks"], c.counts["impactors"]
        if full_state:
            bx, by, bm = FS.bond_arrays(c, c.fs["cols"], c.fs["scale"], c.fs["scale_names"])
            add("y_bond_fs_start", bx[:, :len(FS.BOND_FIELDS)])
            add("y_bond_fs", by)
            add("y_bond_fs_mode", bm)
            e_acc["bond"].setdefault("fs", []).append(np.repeat(bx, 2, 0))  # both twins
            hs = FS.hidden_scale(c)  # frame-end hidden state, encoded like the def_* inputs
            yd = a["y_chunk_def_end"] if "y_chunk_def_end" in a else np.zeros((nc, 12), np.float32)
            add("y_chunk_hid", (yd.astype(np.float64) / hs).astype(np.float32))
            ci, co = FS.chunk_arrays(c, c.fs["cols"], float(np.linalg.norm(c.gravity)))
            add("chunk_fs_in", ci)
            add("y_chunk_fs", co)
        if full_state and "fs_chunk_cluster_out" in a and "fs_cluster_out" in a:
            # frame-end bodies, grouped as the oracle's clusters (training targets; a live
            # call has none: inference groups by the decoded fragments in state_io)
            cg = FS.cluster_groups(c, a["fs_chunk_cluster_out"])
            cy, cact = FS.cluster_targets(c, cg)
            add("clu_of_chunk", cg["of_chunk"].astype(np.int64) + clu_off)
            add("clu_x", cg["x"])
            add("clu_call", np.full(len(cg["key"]), gi))
            add("clu_key", cg["key"])
            add("y_clu", cy)
            add("y_clu_act", cact)
            clu_off += len(cg["key"])
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
        frag_off += int(a["y_chunk_fragment"].max(initial=-1)) + 1
        add("y_imp_regime", a["y_impactor_regime"])
        has_pf = "y_impactor_perforate_freq" in a
        add("y_imp_perf_freq", a["y_impactor_perforate_freq"] if has_pf else np.zeros(ni))
        add("y_imp_has_freq", np.full(ni, has_pf))
        if "ens_impactor_v_end" in a:
            ens_rows.append((imp_off, a["ens_impactor_v_end"]))
        nb = c.counts["bond"]
        for k in ("peak_util", "fail_margin", "broke", "break_bin", "fail_mode", "ddamage",
                  "dissipated"):
            add(f"y_bond_{k}", a[f"y_bond_{k}"])
        add("y_bond_support_path", a.get("y_bond_support_path", np.zeros(nb)))
        # exact break time ÷ dt (NaN where unknown or unbroken); masked in the loss
        bt = a.get("y_bond_break_time")
        add("y_bond_break_frac", (np.asarray(bt, np.float64) / float(c.scales["dt"])).astype(np.float32)
            if bt is not None else np.full(nb, np.nan, np.float32))
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


