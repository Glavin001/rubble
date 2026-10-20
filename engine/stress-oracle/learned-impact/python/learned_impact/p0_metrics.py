"""P0 metrics: the chaos floor of an oracle ensemble and how a candidate run sits in it.

Inputs are `learned_p0` result files (one JSON object per run; see
`crates/stress-rubble/examples/learned_p0.rs`). Metrics follow the plan's
"Objectives" table, restricted to what P0 needs:

- ram exit velocity (along the approach, `ram_vy`);
- fragment count, largest-fragment mass fraction, broken bonds;
- the fragment partition (chunk-weighted adjusted Rand index, members against each
  other and a candidate against the members);
- fragment sizes (1-Wasserstein distance between mass-weighted log10 fragment-mass
  distributions);
- per-bond break frequency over the ensemble.

The P2 gate rules used here (plan, "P2: model"): a candidate passes a scene when
|v - median| <= max(IQR, 0.05 v0) for the ram, and its fragment count and largest
fraction fall inside the ensemble's [min, max]. The skill ratio is the candidate's mean
distance to the members over a typical member's mean distance to the others (<= 1.25
passes; reported, with the absolute distances, since an exact-agreement ensemble makes the
ratio undefined).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def load(path) -> dict:
    with open(path) as f:
        return json.loads(f.readline())


def partition(run: dict) -> np.ndarray:
    """Every chunk's fragment label, walls concatenated (labels made unique per wall)."""
    out = []
    for k, w in enumerate(run["end"]["walls"]):
        out += [(k, p) if p >= 0 else (k, -1) for p in w["partition"]]
    labels = {}
    return np.array([labels.setdefault(x, len(labels)) for x in out])


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two labelings of the same chunks (each chunk weight 1: the
    walls here are uniform grids, so this is the mass-weighted ARI)."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def fragment_masses(run: dict) -> np.ndarray:
    m = []
    for w in run["end"]["walls"]:
        per_chunk = w["mass"] / len(w["partition"])
        labels, counts = np.unique([p for p in w["partition"] if p >= 0], return_counts=True)
        m += list(counts * per_chunk)
    return np.array(m)


def w1_log_mass(a: dict, b: dict) -> float:
    """1-Wasserstein distance between the mass-weighted distributions of log10 fragment mass."""
    from scipy.stats import wasserstein_distance

    ma, mb = fragment_masses(a), fragment_masses(b)
    return float(wasserstein_distance(np.log10(ma), np.log10(mb), u_weights=ma, v_weights=mb))


def broken_set(run: dict) -> set:
    return {(k, b) for k, w in enumerate(run["end"]["walls"]) for b in w["broken"]}


def bond_count(run: dict) -> int:
    return sum(w["bonds"] for w in run["end"]["walls"])


def summary(vals) -> dict:
    v = np.asarray(vals, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "min": float(v.min()), "max": float(v.max()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def distances(a: dict, b: dict) -> dict:
    return {
        "ram": abs(a["end"]["ram_vy"] - b["end"]["ram_vy"]),
        "partition": 1.0 - ari(partition(a), partition(b)),
        "sizes": w1_log_mass(a, b),
        "bonds": len(broken_set(a) ^ broken_set(b)),
    }


def ensemble(members: list[dict]) -> dict:
    """The chaos floor of one scene's ensemble."""
    n = len(members)
    v0 = members[0]["e0"]
    nb = bond_count(members[0])
    freq = {}
    for m in members:
        for k in broken_set(m):
            freq[k] = freq.get(k, 0) + 1
    p = np.array(list(freq.values())) / n if freq else np.zeros(0)
    # Typical member-to-rest distance per metric: each member's mean distance to the others,
    # then the median over members.
    d = [[distances(members[i], members[j]) for j in range(n) if j != i] for i in range(n)]
    typical = {k: float(np.median([np.mean([x[k] for x in row]) for row in d])) for k in d[0][0]}
    consensus = {k for k, c in freq.items() if c / n >= 0.5}
    return {
        "n": n,
        "ram_vy": summary([m["end"]["ram_vy"] for m in members]),
        "fragments": summary([m["end"]["fragments"] for m in members]),
        "largest_fraction": summary([m["end"]["largest_fraction"] for m in members]),
        "broken_bonds": summary([m["end"]["broken_bonds"] for m in members]),
        "bonds": nb,
        "bonds_ever_broken": len(freq),
        "bonds_always_broken": int((p == 1).sum()),
        "bonds_uncertain_0.1_0.9": int(((p > 0.1) & (p < 0.9)).sum()),
        "consensus_bonds": len(consensus),
        "typical_member_distance": typical,
        "_freq": freq,
        "_consensus": consensus,
    }


def candidate_vs(ens: dict, members: list[dict], run: dict, v0: float) -> dict:
    """Where a candidate run sits in an ensemble, and the P2 gate's verdict for it."""
    e = run["end"]
    r, f, lf = ens["ram_vy"], ens["fragments"], ens["largest_fraction"]
    ram_tol = max(r["iqr"], 0.05 * v0)
    ram_ok = abs(e["ram_vy"] - r["median"]) <= ram_tol
    frag_ok = f["min"] <= e["fragments"] <= f["max"]
    largest_ok = lf["min"] - 1e-9 <= e["largest_fraction"] <= lf["max"] + 1e-9
    d = [distances(run, m) for m in members]
    mean = {k: float(np.mean([x[k] for x in d])) for k in d[0]}
    skill = {k: (mean[k] / ens["typical_member_distance"][k]) if ens["typical_member_distance"][k] > 0 else (0.0 if mean[k] == 0 else math.inf) for k in mean}
    # Bond precision / recall against the ensemble consensus.
    b = broken_set(run)
    cons = ens["_consensus"]
    tp = len(b & cons)
    prec = tp / len(b) if b else (1.0 if not cons else 0.0)
    rec = tp / len(cons) if cons else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {
        "ram_vy": e["ram_vy"],
        "fragments": e["fragments"],
        "largest_fraction": e["largest_fraction"],
        "broken_bonds": e["broken_bonds"],
        "ram_ok": bool(ram_ok),
        "ram_tolerance": ram_tol,
        "fragments_ok": bool(frag_ok),
        "largest_ok": bool(largest_ok),
        "within_spread": bool(ram_ok and frag_ok and largest_ok),
        "mean_distance": mean,
        "skill_ratio": skill,
        "bond_precision": prec,
        "bond_recall": rec,
        "bond_f1": f1,
    }


V0 = {"free_wall": 40.0, "wall_row": 40.0, "free_wall_late": 40.0, "wall_row_late": 40.0, "free_wall_late2": 40.0, "wall_row_late2": 40.0, "free_wall_v25": 25.0, "free_wall_v60": 60.0, "free_wall_edge": 40.0, "thick_wall": 40.0, "b5_v20": 20.0}


def v0_of(run: dict) -> float:
    """The scene's initial ram speed."""
    return V0[run["scene"]]


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_ensembles(out_dir, modes=("kin", "kin_weibull")):
    out_dir = Path(out_dir)
    runs = {}
    for p in sorted(out_dir.glob("*.json")):
        d = load(p)
        name = p.stem
        scene = d["scene"]
        if name.endswith("_ref"):
            runs.setdefault(scene, {})["ref"] = d
            continue
        mode = d["perturbation"]["mode"]
        runs.setdefault(scene, {}).setdefault(mode, []).append(d)
    return runs
