"""P0 step 6: a per-bond gradient-boosted tree baseline on the chaos-floor ensembles.

    python -m learned_impact.gbdt_baseline BONDS.jsonl CHAOS_OUT_DIR [JSON]

`BONDS.jsonl` is `learned_p0 bonds` (each scene's bond and chunk geometry and ram). Every
ensemble member gives one row per bond: impact-relative, scale-free features (no bond ids,
no absolute positions) and whether the bond broke. Two evaluations:

- **leave one scene out**: train on the other scenes' members, predict the held-out scene's
  (the generalisation a graph model would have to beat);
- **in-scene** (5-fold over members of one scene): the same scene seen in training, an
  optimistic bound.

Each predicted break set (probability >= 0.5) gives fragments (connected components of the
unbroken bonds) and the largest fraction, scored against the ensemble as `candidate_vs`
does; pooled bond precision/recall/F1 against the ensemble consensus. A per-bond tree has
no velocity output, so the ram speed is not part of its verdict.
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .p0_metrics import load_ensembles

FEATURES = ["depth", "depth_frac", "lateral", "lateral_over_face", "edge_dist", "normal_along", "area", "wall_index", "wall_thickness", "anchored", "v0", "mass_ratio", "energy_ratio", "weibull"]


def scene_geometry(rec):
    """Per bond of every wall: geometry in the ram's frame of approach."""
    ram = rec["ram"]
    p0 = np.array(ram["position"], float)
    v = np.array(ram["velocity"], float)
    d = v / np.linalg.norm(v)
    face = 0.1  # the M5 ram's half extent (a 0.2 m box); b5's 1000 kg ram is the same shape class
    if "half_extents" in ram["shape"]:
        nums = [float(x) for x in ram["shape"].split("[")[1].split("]")[0].split(",")]
        face = min(nums)
    walls = []
    first_front = None
    for k, w in enumerate(rec["walls"]):
        centers = np.array([c["center"] for c in w["chunks"]], float)
        half = np.array([c["half"] for c in w["chunks"]], float)
        along = (centers - p0) @ d
        front = float(np.min(along - np.abs(half @ np.abs(d))))
        back = float(np.max(along + np.abs(half @ np.abs(d))))
        if first_front is None:
            first_front = front
        lat_pts = (centers - p0) - np.outer(along, d)
        lo, hi = lat_pts.min(0) - half.max(), lat_pts.max(0) + half.max()
        rows = []
        for b in w["bonds"]:
            c = np.array(b["centroid"], float)
            r = c - p0
            a = float(r @ d)
            lat = r - a * d
            edge = float(np.min(np.concatenate([lat - lo, hi - lat])[np.abs(np.concatenate([d, d])) < 0.5]))
            anchored = w["chunks"][b["a"]]["support"] or w["chunks"][b["b"]]["support"]
            rows.append({"depth": a - front, "depth_frac": (a - front) / max(back - front, 1e-9), "lateral": float(np.linalg.norm(lat)), "lat_vec": lat, "face": face,
                         "lateral_over_face": float(np.linalg.norm(lat)) / face, "edge_dist": edge, "normal_along": abs(float(np.dot(b["normal"], d))),
                         "area": b["area"], "wall_index": k, "wall_thickness": back - front, "anchored": float(anchored),
                         "ga": b["g_tension"] * b["area"], "a": b["a"], "b": b["b"]})
        walls.append({"rows": rows, "mass": float(sum(c["mass"] for c in w["chunks"])), "chunks": len(w["chunks"])})
    return {"ram_mass": ram["mass"], "walls": walls}


def member_rows(geom, run, mode):
    """Feature matrix and labels of one ensemble member."""
    pert = run["perturbation"] or {}
    off = np.array(pert.get("offset_m", [0, 0, 0]), float)
    fs = pert.get("speed_factor", 1.0)
    v0 = run_v0(run) * fs
    total_mass = sum(w["mass"] for w in geom["walls"])
    e0 = 0.5 * geom["ram_mass"] * v0 * v0
    ga = sum(r["ga"] for w in geom["walls"] for r in w["rows"])
    X, y = [], []
    for k, (w, wo) in enumerate(zip(geom["walls"], run["end"]["walls"])):
        broken = set(wo["broken"])
        for i, r in enumerate(w["rows"]):
            # The member's impact point moved across the face by `off`: lateral distances shift.
            lat = float(np.linalg.norm(r["lat_vec"] - off))
            X.append([r["depth"], r["depth_frac"], lat, lat / r["face"], r["edge_dist"], r["normal_along"], r["area"], r["wall_index"], r["wall_thickness"], r["anchored"],
                      v0, geom["ram_mass"] / total_mass, e0 / ga, 1.0 if mode == "kin_weibull" else 0.0])
            y.append(1 if i in broken else 0)
    return np.array(X), np.array(y)


V0S = {"free_wall": 40.0, "wall_row": 40.0, "free_wall_v25": 25.0, "free_wall_v60": 60.0, "free_wall_edge": 40.0, "thick_wall": 40.0, "b5_v20": 20.0}


def run_v0(run):
    return V0S[run["scene"]]


def fragments_of(geom, broken_mask):
    """Fragment count and largest mass fraction from a predicted break mask (walls concatenated)."""
    count, largest, total = 0, 0.0, 0.0
    at = 0
    for w in geom["walls"]:
        n = w["chunks"]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for r in w["rows"]:
            if not broken_mask[at]:
                a, b = find(r["a"]), find(r["b"])
                if a != b:
                    parent[a] = b
            at += 1
        roots = [find(i) for i in range(n)]
        sizes = np.unique(roots, return_counts=True)[1]
        count += len(sizes)
        largest = max(largest, sizes.max() * w["mass"] / n)
        total += w["mass"]
    return count, largest / total


def evaluate(geom, ens_runs, prob):
    """Scores of a predicted break probability per bond against one scene's kin ensemble."""
    pred = prob >= 0.5
    frag, largest = fragments_of(geom, pred)
    fr = [r["end"]["fragments"] for r in ens_runs]
    lf = [r["end"]["largest_fraction"] for r in ens_runs]
    freq = np.zeros(len(prob))
    for r in ens_runs:
        at = 0
        for w, wo in zip(geom["walls"], r["end"]["walls"]):
            br = set(wo["broken"])
            for i in range(len(w["rows"])):
                freq[at] += i in br
                at += 1
    freq /= len(ens_runs)
    cons = freq >= 0.5
    tp = int((pred & cons).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int(cons.sum()), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    brier = float(np.mean((prob - freq) ** 2))
    return {"fragments": int(frag), "largest_fraction": float(largest), "fragments_ok": bool(min(fr) <= frag <= max(fr)), "largest_ok": bool(min(lf) - 1e-9 <= largest <= max(lf) + 1e-9),
            "ensemble_fragments": [min(fr), float(np.median(fr)), max(fr)], "bond_precision": prec, "bond_recall": rec, "bond_f1": f1, "brier_vs_frequency": brier,
            "predicted_breaks": int(pred.sum()), "consensus_breaks": int(cons.sum())}


def main():
    geoms = {}
    for line in open(sys.argv[1]):
        rec = json.loads(line)
        geoms[rec["scene"]] = scene_geometry(rec)
    runs = load_ensembles(sys.argv[2])
    scenes = [s for s in V0S if s in runs]
    data = {s: {m: [member_rows(geoms[s], r, m) for r in runs[s].get(m, [])] for m in ("kin", "kin_weibull")} for s in scenes}
    report = {"leave_one_scene_out": {}, "in_scene": {}}
    clf = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0)
    for held in scenes:
        Xtr = np.vstack([x for s in scenes if s != held for m in data[s] for (x, _) in data[s][m]])
        ytr = np.concatenate([y for s in scenes if s != held for m in data[s] for (_, y) in data[s][m]])
        model = clf().fit(Xtr, ytr)
        # One prediction per member (they differ only in the speed factor); averaged.
        prob = np.mean([model.predict_proba(x)[:, 1] for (x, _) in data[held]["kin"]], axis=0)
        r = evaluate(geoms[held], runs[held]["kin"], prob)
        report["leave_one_scene_out"][held] = r
        print(f"LOSO {held:15s} frag {r['fragments']:4d} (ens {r['ensemble_fragments']}) {'ok' if r['fragments_ok'] else 'out'} | largest {r['largest_fraction']:.3f} {'ok' if r['largest_ok'] else 'out'} | bond P {r['bond_precision']:.2f} R {r['bond_recall']:.2f} F1 {r['bond_f1']:.2f} | brier {r['brier_vs_frequency']:.3f}")
    for s in scenes:
        rows = data[s]["kin"] + data[s]["kin_weibull"]
        n = len(data[s]["kin"])
        probs = np.zeros_like(data[s]["kin"][0][1], dtype=float)
        folds = 4
        for f in range(folds):
            test = [i for i in range(n) if i % folds == f]
            train = [rows[i] for i in range(len(rows)) if not (i < n and i % folds == f)]
            model = clf().fit(np.vstack([x for x, _ in train]), np.concatenate([y for _, y in train]))
            for i in test:
                probs += model.predict_proba(data[s]["kin"][i][0])[:, 1]
        probs /= n
        r = evaluate(geoms[s], runs[s]["kin"], probs)
        report["in_scene"][s] = r
        print(f"in-scene {s:15s} frag {r['fragments']:4d} (ens {r['ensemble_fragments']}) {'ok' if r['fragments_ok'] else 'out'} | largest {r['largest_fraction']:.3f} {'ok' if r['largest_ok'] else 'out'} | bond P {r['bond_precision']:.2f} R {r['bond_recall']:.2f} F1 {r['bond_f1']:.2f} | brier {r['brier_vs_frequency']:.3f}")
    if len(sys.argv) > 3:
        json.dump(report, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
