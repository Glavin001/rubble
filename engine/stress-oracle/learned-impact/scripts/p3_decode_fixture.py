"""The decode parity fixture for the engine's Slang decode (P3, `learned.slang`).

Runs P2's PyTorch decode (`learned_impact.outcome.decode`, the deployed rule) on fixed
inputs: P2's exported parity calls (`export/parity/<k>/call` + `outputs.npz`) and variants
of them that exercise every branch (energy bound on a free call, an anchored call, a call
made non-free by gravity, extra breaks). Writes one JSON file with the decode's inputs
(raw node arrays, scales, gravity, the network's outputs) and its outputs, which
`stress-gpu/tests/learned_decode.rs` replays on the GPU. Generated once and committed; the
test never imports P2's code.

    P2=<P2 worktree>/engine/stress-oracle/learned-impact
    $P2/python/.venv/bin/python learned-impact/scripts/p3_decode_fixture.py \
        --python $P2/python --export $P2/runs/v3/sweep-all3/export \
        --out crates/stress-gpu/tests/fixtures/learned_decode.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True, help="P2's learned-impact/python")
    ap.add_argument("--export", required=True, help="an export directory with parity/<k>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.python)
    import torch
    from learned_impact.outcome import decode as D
    from learned_impact.outcome import schema as S
    from learned_impact.outcome.batch import HierarchyCache, collate

    exp = Path(a.export)
    manifest = json.loads((exp / "weights.json").read_text())
    calls, outs, structures = [], [], {}
    for k in sorted(int(p.name) for p in (exp / "parity").iterdir()):
        d = exp / "parity" / str(k)
        ev = S.load_event(d / "call")
        structures[ev.manifest["event_id"]] = ev.structures
        calls.append(ev.calls[0])
        outs.append({n: v for n, v in np.load(d / "outputs.npz").items()})
    base = len(calls)

    def variant(i, f_call=None, f_out=None):
        c = copy.deepcopy(calls[i])
        o = {n: v.copy() for n, v in outs[i].items()}
        if f_call:
            f_call(c)
        if f_out:
            f_out(o)
        calls.append(c)
        outs.append(o)

    def fast(o, s):
        o["chunk_out"][:, 0:3] *= s
        o["chunk_out"][:, 6:9] *= s
        o["chunk_out"][:, 3:6] *= s

    variant(0, None, lambda o: fast(o, 40.0))  # free, energy bound active

    def anchor(c):
        c.arrays["chunk_anchored"] = c.arrays["chunk_anchored"].copy()
        c.arrays["chunk_anchored"][:4] = 1
        c.anchored = True

    variant(min(2, base - 1), anchor, lambda o: fast(o, 20.0))  # anchored: scaled whole

    def gravity(c):
        c.gravity = [0.0, 0.0, -9.81]
        g = c.arrays["global_feat"].copy()
        g[:, S.col("global_feat", "gravity_mag")] = 1.0
        c.arrays["global_feat"] = g

    def more_breaks(o):
        o["bond_log_q"] = o["bond_log_q"] + 0.3

    variant(base - 1, gravity, more_breaks)  # not free (gravity), more breaks

    cache = HierarchyCache(manifest["model_config"]["levels"])
    b = collate(calls, structures, cache)
    out = {n: torch.from_numpy(np.concatenate([o[n] for o in outs])) for n in outs[0]}
    dcfg = D.DecodeConfig(tau=manifest["decode"]["tau"], temperature=manifest["decode"]["temperature"],
                          energy_tolerance=manifest["decode"]["energy_tolerance"])
    dec = D.decode(out, b, dcfg)
    t = b.t

    def l(x):
        return np.asarray(x, dtype=np.float64).reshape(-1).tolist()

    nc_all = t["chunk_mass"].shape[0]
    doc = {
        "schema": "learned-impact-decode-fixture/1",
        "source": {"export": str(exp), "checkpoint": manifest.get("checkpoint")},
        "decode": {"tau": dcfg.tau, "energy_tolerance": dcfg.energy_tolerance},
        "calls": [],
    }
    tbin = out["bond_time_logits"].numpy().argmax(-1) + 1
    d0 = b.edges["bond"]["x"][0::2, S.col("bond_feat", "damage")].numpy()
    free = dec["free"].numpy()
    for g, c in enumerate(calls):
        sc = (t["chunk_call"] == g).numpy()
        si = (t["imp_call"] == g).numpy()
        sb = (t["bond_call"] == g).numpy()
        ci = np.flatnonzero(sc)
        ii = np.flatnonzero(si)
        doc["calls"].append({
            "scales": {k: float(c.scales[k]) for k in S.SCALE_KEYS},
            "gravity": l(c.gravity),
            "free": bool(free[g]),
            "external": False,
            "chunks": {
                "mass": l(t["chunk_mass"][sc]), "inertia": l(t["chunk_inertia_body"][sc]),
                "q": l(t["chunk_q_world"][sc]), "pos": l(t["chunk_pos"][sc]),
                "v": l(t["chunk_v"][sc]), "w": l(t["chunk_w"][sc]),
                "anchored": [int(x) for x in t["chunk_anchored"][sc].numpy()],
                "out": l(out["chunk_out"][sc]),
            },
            "impactors": {
                "mass": l(t["imp_mass"][si]), "inertia": l(t["imp_inertia_body"][si]),
                "q": l(t["imp_q_world"][si]), "pos": l(t["imp_pos"][si]),
                "v": l(t["imp_v"][si]), "w": l(t["imp_w"][si]),
                "out": l(out["imp_out"][si]),
            },
            "bonds": {
                "damage0": l(d0[sb]), "log_q": l(out["bond_log_q"][sb]),
                "ddamage": l(out["bond_ddamage"][sb]), "bin": [int(x) for x in tbin[sb]],
            },
            "expected": {
                "chunk_v": l(dec["chunk_v_end"][ci]), "chunk_w": l(dec["chunk_w_end"][ci]),
                "chunk_dx": l(dec["chunk_dx"][ci]), "chunk_drot": l(dec["chunk_drot"][ci]),
                "imp_v": l(dec["imp_v_end"][ii]), "imp_w": l(dec["imp_w_end"][ii]),
                "imp_dx": l(dec["imp_dx"][ii]), "imp_drot": l(dec["imp_drot"][ii]),
                "broke": [int(x) for x in dec["bond_broke"][sb].numpy()],
                "damage": l(dec["bond_damage_end"][sb]),
                "energy_bound_active": bool(dec["energy_bound_active"][g]),
                "energy_bound_infeasible": bool(dec["energy_bound_infeasible"][g]),
            },
        })
    assert sum(len(c["chunks"]["mass"]) for c in doc["calls"]) == nc_all
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc))
    summary = [(len(c["chunks"]["mass"]), c["free"], c["expected"]["energy_bound_active"],
                sum(c["expected"]["broke"])) for c in doc["calls"]]
    print(a.out, summary)


if __name__ == "__main__":
    main()
