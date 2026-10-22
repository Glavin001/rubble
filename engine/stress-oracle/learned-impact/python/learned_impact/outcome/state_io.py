"""`learned-impact-decoded-state/1`: the model's decoded outcome of one call, keyed by stable
keys (DECODED_STATE.md). It is what a DAgger restart (P1 recorder) and the closed-loop
rollout harness apply at window end, with the island's contacts filtered for that frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import decode as D
from . import schema as S
from .batch import Batch

VERSION = "learned-impact-decoded-state/1"


def _vec(x) -> list[float]:
    return [float(v) for v in x]


def decoded_states(b: Batch, out: dict[str, torch.Tensor], dec: dict,
                   meta: dict[str, Any] | None = None, model=None) -> list[dict]:
    """One decoded-state document per call of the batch (float64 values, world axes)."""
    t = b.t
    cc, ic, bc = (t["chunk_call"].cpu().numpy(), t["imp_call"].cpu().numpy(),
                  t["bond_call"].cpu().numpy())
    cv, cw = dec["chunk_v_end"].detach().cpu().numpy(), dec["chunk_w_end"].detach().cpu().numpy()
    cdx, cdr = dec["chunk_dx"].detach().cpu().numpy(), dec["chunk_drot"].detach().cpu().numpy()
    iv, iw = dec["imp_v_end"].detach().cpu().numpy(), dec["imp_w_end"].detach().cpu().numpy()
    idx, idr = dec["imp_dx"].detach().cpu().numpy(), dec["imp_drot"].detach().cpu().numpy()
    broke = dec["bond_broke"].cpu().numpy()
    damage = dec["bond_damage_end"].detach().cpu().numpy()
    tbin = out["bond_time_logits"].detach().cpu().numpy().argmax(-1) + 1
    p_perf = torch.sigmoid(out["imp_perf_logit"]).detach().cpu().numpy()
    active = dec["energy_bound_active"].cpu().numpy()
    infeasible = dec["energy_bound_infeasible"].cpu().numpy()
    free = dec["free"].cpu().numpy()
    anchored = t["chunk_anchored"].cpu().numpy()
    frag = D.fragments(b, dec["bond_broke"]) if (model is not None and "bond_fs" in out) else None
    docs = []
    for g, c in enumerate(b.calls):
        a = c.arrays
        sc, si, sb = cc == g, ic == g, bc == g
        bkeys = a["bond_key"].tolist()
        doc = {
            "schema_version": VERSION,
            "calls_schema": S.SCHEMA_VERSION,
            "event_id": c.event_id, "call_index": c.call_index, "frame": c.frame,
            "island": c.island,
            "meta": meta or {},
            "flags": {"free": bool(free[g]), "energy_bound_active": bool(active[g]),
                      "energy_bound_infeasible": bool(infeasible[g])},
            "chunks": [
                {"key": int(k), "anchored": bool(an), "v": _vec(v), "w": _vec(w),
                 "dx": _vec(dx), "drot": _vec(dr)}
                for k, an, v, w, dx, dr in zip(a["chunk_key"].tolist(), anchored[sc], cv[sc],
                                               cw[sc], cdx[sc], cdr[sc])],
            "impactors": [
                {"key": int(k), "v": _vec(v), "w": _vec(w), "dx": _vec(dx), "drot": _vec(dr),
                 "p_perforate": float(p)}
                for k, v, w, dx, dr, p in zip(a["impactor_key"].tolist(), iv[si], iw[si],
                                              idx[si], idr[si], p_perf[si])],
            "bonds_broken": [{"key": int(k), "bin": int(tb)} for k, x, tb in
                             zip(bkeys, broke[sb], tbin[sb]) if x],
            "bond_damage": [{"key": int(k), "damage": float(d)} for k, x, d in
                            zip(bkeys, broke[sb], damage[sb]) if not x],
        }
        if "bond_fs" in out and c.fs is not None:
            fs = _full_state_out(c, out, broke[sb], damage[sb], np.nonzero(sb)[0], np.nonzero(sc)[0])
            if model is not None and getattr(model, "cluster_head", None) is not None:
                fs["cluster"] = _clusters_out(c, out, frag[sc], np.nonzero(sc)[0], g, model)
            # P0's host format: per kind, keys, field-path columns and flat row-major rows;
            # JSON has no infinity, so ±inf and NaN (None) go as strings
            doc["full_state_out"] = [
                {"kind": kind, "keys": tab["keys"], "columns": tab["columns"],
                 "rows": [_json_num(x) for row in tab["values"] for x in row]}
                for kind, tab in fs.items()]
        docs.append(doc)
    return docs


def _json_num(x: float):
    if np.isnan(x):
        return "nan"
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(x)


@torch.no_grad()
def _clusters_out(c: S.Call, out: dict, frag_rows: np.ndarray, c_rows: np.ndarray, gi: int,
                  model) -> dict:
    """Frame-end bodies from the decoded fragments: the model's cluster head on each
    fragment's chunks, decoded to raw world columns, keyed by the body's lowest chunk key."""
    from . import fullstate as FS

    grp = FS.cluster_groups(c, frag_rows)
    ch = out["_chunk_h"][torch.as_tensor(c_rows)]
    g = out["_glob"][gi:gi + 1]
    n = len(grp["key"])
    co = model.cluster_out(ch, g, torch.as_tensor(grp["of_chunk"], dtype=torch.long),
                           torch.zeros(n, dtype=torch.long), torch.as_tensor(grp["x"])).cpu().numpy()
    cols, vals = FS.cluster_raw(c, grp, co[:, :13].astype(np.float64), co[:, 13:15])
    return {"keys": grp["key"].tolist(), "columns": cols,
            "values": [[float(x) for x in row] for row in vals]}


def _full_state_out(c: S.Call, out: dict, broke, damage, b_rows, c_rows) -> dict:
    """`/5` output: the predicted full-state columns in raw units, by P1's field paths and the
    entities' keys (the host writes them back through the full-state unshred path)."""
    from . import fullstate as FS

    a = c.arrays
    names = c.fs["scale_names"]
    keys = a["bond_key"].astype(np.int64)
    sc_rows = np.stack([c.fs["scale"].get(int(k), np.full(len(names), np.nan)) for k in keys]) \
        if len(keys) else np.zeros((0, len(names)))
    sc = {n: sc_rows[:, i] for i, n in enumerate(names)}
    dt = float(c.scales["dt"])
    enc = out["bond_fs"].detach().cpu().numpy()[b_rows].astype(np.float64)
    mode = out["bond_fs_mode_logits"].detach().cpu().numpy()[b_rows].argmax(-1)
    cols, vals = [], []
    for i, (f, how, _) in enumerate(FS.BOND_FIELDS):
        cols.append(f)
        vals.append(FS.decode_raw(enc[:, i], how, sc, dt))
    cols += list(FS.BOND_MODE) + ["alive", "joint.damage"]
    vals += [(mode == 1).astype(float), (mode == 2).astype(float), (~broke).astype(float),
             np.where(broke, 1.0, damage)]
    table = np.stack(vals, 1) if len(keys) else np.zeros((0, len(cols)))
    m = a["chunk_mass"].astype(np.float64)
    g = max(float(np.linalg.norm(c.gravity)), 1e-6)
    L = float(c.scales["L"])
    rs = np.concatenate([np.repeat((m * g)[:, None], 3, 1), np.repeat((m * g * L)[:, None], 3, 1)], 1)
    rx = out["chunk_fs"].detach().cpu().numpy()[c_rows].astype(np.float64) * rs
    return {"bond": {"keys": keys.tolist(), "columns": cols,
                     "values": [[float(x) for x in row] for row in table]},
            "chunk": {"keys": a["chunk_key"].astype(np.int64).tolist(), "columns": list(FS.CHUNK_FIELDS),
                      "values": [[float(x) for x in row] for row in rx]}}


def validate(doc: dict, call: S.Call | None = None) -> None:
    if doc.get("schema_version") != VERSION:
        raise S.SchemaError(f"decoded state version {doc.get('schema_version')!r}")
    for part in ("chunks", "impactors"):
        for n in doc[part]:
            for k in ("v", "w", "dx", "drot"):
                if len(n[k]) != 3 or not all(np.isfinite(n[k])):
                    raise S.SchemaError(f"{part} {n['key']}: bad {k}")
    if call is not None:
        a = call.arrays
        if sorted(n["key"] for n in doc["chunks"]) != sorted(a["chunk_key"].tolist()):
            raise S.SchemaError("chunk keys differ from the call")
        if sorted(n["key"] for n in doc["impactors"]) != sorted(a["impactor_key"].tolist()):
            raise S.SchemaError("impactor keys differ from the call")
        live = set(a["bond_key"].tolist())
        b = [x["key"] for x in doc["bonds_broken"]] + [x["key"] for x in doc["bond_damage"]]
        if sorted(b) != sorted(live):
            raise S.SchemaError("bond keys must partition the call's live bonds")
        for x in doc["bonds_broken"]:
            if not 1 <= x["bin"] <= S.N_TIME_BINS:
                raise S.SchemaError(f"bond {x['key']}: bin {x['bin']}")


def write(path: Path | str, doc: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc))  # repr floats: exact f64 round trip
    tmp.replace(path)  # atomic: a polling reader never sees half a file
    return path


def read(path: Path | str, call: S.Call | None = None) -> dict:
    doc = json.loads(Path(path).read_text())
    validate(doc, call)
    return doc


def from_targets(c: S.Call) -> dict:
    """The oracle's own outcome of a call as a decoded state (the perfect model; used to
    check that a consumer reproduces P0's ceiling)."""
    a = c.arrays
    broke = a["y_bond_broke"] > 0
    d0 = a["bond_feat"][0::2, S.col("bond_feat", "damage")]
    return {
        "schema_version": VERSION, "calls_schema": S.SCHEMA_VERSION, "event_id": c.event_id,
        "call_index": c.call_index, "frame": c.frame, "island": c.island,
        "meta": {"model": "oracle"},
        "flags": {"free": not c.anchored, "energy_bound_active": False,
                  "energy_bound_infeasible": False},
        "chunks": [{"key": int(k), "anchored": bool(an), "v": _vec(v), "w": _vec(w),
                    "dx": _vec(dx), "drot": _vec(dr)} for k, an, v, w, dx, dr in zip(
            a["chunk_key"].tolist(), a["chunk_anchored"], a["y_chunk_v_end"], a["y_chunk_w_end"],
            a["y_chunk_dx"], a["y_chunk_drot"])],
        "impactors": [{"key": int(k), "v": _vec(v), "w": _vec(w), "dx": _vec(dx),
                       "drot": _vec(dr), "p_perforate": float(r == S.REGIMES["perforate"])}
                      for k, v, w, dx, dr, r in zip(
            a["impactor_key"].tolist(), a["y_impactor_v_end"], a["y_impactor_w_end"],
            a["y_impactor_dx"], a["y_impactor_drot"], a["y_impactor_regime"])],
        "bonds_broken": [{"key": int(k), "bin": int(b_)} for k, x, b_ in zip(
            a["bond_key"].tolist(), broke, a["y_bond_break_bin"]) if x],
        "bond_damage": [{"key": int(k), "damage": float(d + dd)} for k, x, d, dd in zip(
            a["bond_key"].tolist(), broke, d0, a["y_bond_ddamage"]) if not x],
    }
