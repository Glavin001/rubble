"""Closed-loop full-event evaluation (remaining work, step 2 of the plan's "Goal and definition
of done"): every validation wall event of P1's dataset run whole, from its first routed frame
until it settles, through P0's coupled harness (`learned_p0 event`), under each candidate:

  ceiling  the oracle's own frame through the call boundary (window-end, reduced state, the
           island's contacts filtered): what a perfect model achieves; equal to the echo /
           targets provider (bit-identical, `closed-loop-harness-2026-10-21.md`)
  ceiling_full  (diagnostic) the same with the full state carried between calls (hidden
           deformation kept): separates what the reduced state loses
  qs       CPU `blast_parity` (P1's QS baseline), standalone
  punch    the host's predictive punch, the engine taking each frame with its contacts
  model    P2's checkpoint (`--checkpoint`), through `stress_learned::provider_client`

and scored against the event's oracle ensemble (P1's recorded members) with the acceptance
metrics: impactor exit speed and regime, partition (1 - ARI), fragment sizes (W1 of mass-
weighted log10 mass), fragment count and largest fraction in the ensemble's range, fragment
motion (mass-weighted RMS chunk velocity and position differences 0.1 s and 0.5 s after the
first contact, and at the end), support survival (anchored mass retained), and the physics
checks (KE <= E_in + W_g + 5% E0, momentum where it is conserved, damage monotone, finite
state, fallbacks and energy-bound activations).

Skill: a candidate's mean distance to the ensemble's members over a typical member's distance
to the others; the calibrated bar is max(1.25, the same ratio for a held-out oracle member).

    python -m learned_impact.closed_loop_eval --runs P1_RUNS/p1-bulk-v1-free-t1 P1_RUNS/p1-bulk-v1-anchored-t1 \\
        --out ../runs/cl-eval [--candidates ceiling,qs,punch[,model]] [--checkpoint best.pt --python-dir DIR]
        [--max-ensembles 16] [--jobs 6] [--bin ../../target-p0/release/examples/learned_p0]

Writes `<out>/runs/<candidate>/<event>.json` (resumable), `<out>/report.json` and
`<out>/report.md`; one progress line per finished run (run it under `scripts/job.py`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .outcome_audit import load_event

HERE = Path(__file__).resolve().parent
DEFAULT_BIN = HERE.parents[2] / "target-p0" / "release" / "examples" / "learned_p0"
SKILL_MAX = 1.25
METRICS = ("ram", "partition", "sizes", "motion_v_0.1", "motion_v_0.5", "motion_x_0.5", "support")


# ------------------------------------------------------------------ discovery


def validation_ensembles(runs: list[Path], split: str, max_ensembles: int | None) -> dict[str, list[Path]]:
    """Ensemble id -> member event dirs (sorted by member), for events of `split`."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for run in runs:
        for ctx_path in sorted((run / "events").glob("*/context.json")):
            ctx = json.loads(ctx_path.read_text())
            if ctx.get("split") != split:
                continue
            ens = ctx.get("ensemble") or {}
            if not ens.get("id"):
                continue
            groups[ens["id"]].append((ens.get("member", 0), ctx_path.parent))
    out = {}
    per_family: Counter = Counter()
    for eid in sorted(groups):
        members = [p for _, p in sorted(groups[eid])]
        fam = json.loads((members[0] / "context.json").read_text())["family"]
        if max_ensembles is not None and per_family[fam] >= max_ensembles:
            continue
        per_family[fam] += 1
        out[eid] = members
    return out


def sample_frames(ev) -> list[int]:
    """Frames 0.1 s and 0.5 s after the first contact (6 and 30 frames at 1/60), and the end."""
    phases = [f.get("phase") for f in ev.frames]
    first = next((i + 1 for i, p in enumerate(phases) if p and p != "pre_contact"), 1)
    n = ev.n_frames
    return [min(first + 6, n), min(first + 30, n), n]


# ------------------------------------------------------------------ truth and runs


def oracle_state(ev, samples: list[int]) -> dict:
    """The recorded member as the scorer reads candidates."""
    rows = ev.state_rows()
    last = rows[ev.n_frames]
    cl = ev.array("state_chunk_cluster")[last]
    active = ev.array("state_chunk_active")[last].astype(bool)
    anchored = ev.array("state_chunk_anchored")[last].astype(bool)
    pos, vel = ev.array("state_chunk_pos"), ev.array("state_chunk_vel")
    return {
        "cluster": np.where(active, cl, -1),
        "anchored": anchored & active,
        "speed": [i["speed_along_v0"] for i in ev.outcome["impactors"]],
        "regime": [i["regime"] for i in ev.outcome["impactors"]],
        "fragments": ev.outcome["fragments"],
        "largest": ev.outcome["largest_fraction"],
        "samples": {f: (pos[rows[f]], vel[rows[f]]) for f in samples if f in rows},
    }


def candidate_state(run: dict, samples: list[int]) -> dict:
    r = run["run"]
    by = {s["frame"]: (np.array(s["chunk_pos"], np.float32), np.array(s["chunk_vel"], np.float32)) for s in r["samples"]}
    cl = np.array(r["chunk_cluster"])
    return {
        "cluster": cl,
        "anchored": np.array(r["chunk_anchored"], bool),
        "speed": [i["speed_along_v0"] for i in r["impactors"]],
        "regime": [i["regime"] for i in r["impactors"]],
        "fragments": r["fragments"],
        "largest": None,
        "samples": {f: by[f] for f in samples if f in by},
        "checks": r["checks"],
        "extra": run.get("extra", {}),
    }


def candidate_args(name: str, a, out: Path, event: str) -> tuple[list[str], dict]:
    deb = {"STRESS_DEBRIS_ROUTING": "1"}
    if name == "ceiling":
        return ["--variant", "provider", "--provider", "oracle"], deb
    if name == "ceiling_full":
        # Diagnostic: the oracle's frame with the full (hidden) state carried between calls.
        return ["--variant", "provider", "--provider", "oracle_full"], deb
    if name == "qs":
        return ["--variant", "qs"], {}
    if name == "punch":
        return ["--variant", "provider", "--provider", "punch"], deb
    if name == "model":
        d = out / "provider" / event
        args = ["--variant", "provider", "--provider", "model", "--checkpoint", str(a.checkpoint), "--dir", str(d), "--timeout", str(a.timeout), "--device", a.device]
        if a.python_dir:
            args += ["--python-dir", str(a.python_dir)]
        return args, deb
    raise SystemExit(f"unknown candidate {name}")


# ------------------------------------------------------------------ distances


def ari(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score

    keep = (a >= 0) & (b >= 0)
    return float(adjusted_rand_score(a[keep], b[keep]))


def w1_log_mass(a: np.ndarray, b: np.ndarray, mass: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    def masses(c):
        out = []
        for k in np.unique(c[c >= 0]):
            out.append(mass[c == k].sum())
        return np.array(out)

    ma, mb = masses(a), masses(b)
    return float(wasserstein_distance(np.log10(ma), np.log10(mb), u_weights=ma, v_weights=mb))


def rms(a: np.ndarray, b: np.ndarray, mass: np.ndarray) -> float:
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    d = ((a[ok] - b[ok]) ** 2).sum(1)
    return float(np.sqrt((d * mass[ok]).sum() / max(mass[ok].sum(), 1e-12)))


def distance(x: dict, y: dict, mass: np.ndarray, samples: list[int], anchored_family: bool) -> dict:
    d = {
        # A non-finite run (null in its JSON) is infinitely far.
        "ram": float(np.mean([abs(p - q) if p is not None and q is not None else math.inf for p, q in zip(x["speed"], y["speed"])])),
        "partition": 1.0 - ari(x["cluster"], y["cluster"]),
        "sizes": w1_log_mass(x["cluster"], y["cluster"], mass),
    }
    for label, f, kind in (("motion_v_0.1", samples[0], 1), ("motion_v_0.5", samples[1], 1), ("motion_x_0.5", samples[1], 0)):
        if f in x["samples"] and f in y["samples"]:
            d[label] = rms(x["samples"][f][kind], y["samples"][f][kind], mass)
    if anchored_family:
        tot = mass.sum()
        d["support"] = abs(mass[x["anchored"]].sum() / tot - mass[y["anchored"]].sum() / tot)
    return d


def ensemble_scores(truth: list[dict], cands: dict[str, list[dict | None]], mass: np.ndarray, samples: list[list[int]], anchored_family: bool) -> dict:
    """Skill per metric for each candidate and for a held-out oracle member (calibration)."""
    n = len(truth)
    # Every pair of members (each member's own sample frames apply to both).
    dmat = [[distance(truth[i], truth[j], mass, samples[i], anchored_family) if i != j else None for j in range(n)] for i in range(n)]
    metrics = [m for m in METRICS if all(m in dmat[0][j] for j in range(1, n))]

    def typical(members: list[int], m: str) -> float:
        vals = [np.mean([dmat[i][j][m] for j in members if j != i]) for i in members]
        return float(np.median(vals))

    out: dict = {"metrics": metrics, "members": n}
    loo = {}
    for m in metrics:
        ratios = []
        for i in range(n):
            rest = [j for j in range(n) if j != i]
            t = typical(rest, m)
            dm = np.mean([dmat[i][j][m] for j in rest])
            ratios.append(dm / t if t > 0 else (0.0 if dm == 0 else math.inf))
        loo[m] = float(np.mean(ratios))
    out["oracle_member"] = loo
    t_all = {m: typical(list(range(n)), m) for m in metrics}
    out["typical"] = t_all
    frag = [t["fragments"] for t in truth]
    largest = [t["largest"] for t in truth]
    regimes = Counter(r for t in truth for r in t["regime"][:1])
    majority, share = regimes.most_common(1)[0]
    out["regime_majority"] = majority if share / n >= 0.8 else None
    for name, runs in cands.items():
        per = defaultdict(list)
        regime_hits, in_range, physics = [], [], []
        for i, c in enumerate(runs):
            if c is None:
                continue
            ds = [distance(c, truth[j], mass, samples[i], anchored_family) for j in range(n)]
            for m in metrics:
                if all(m in x for x in ds):
                    per[m].append(np.mean([x[m] for x in ds]))
            if out["regime_majority"]:
                regime_hits.append(c["regime"][0] == out["regime_majority"])
            in_range.append(min(frag) <= c["fragments"] <= max(frag))
            ch = c["checks"]
            physics.append({
                "ke_ok": ch["ke_excess_over_e0"] <= 0.05,
                "momentum_ok": ch["momentum_rel"] is None or ch["momentum_rel"] <= 1e-3,
                "damage_monotone": ch["damage_drop"] <= 1e-12,
                "finite": ch["nonfinite"] == 0,
                "fallbacks": c["extra"].get("fallbacks", 0),
                "energy_bound_active": c["extra"].get("energy_bound_active", 0),
            })
        skill = {m: (float(np.mean(per[m])) / t_all[m] if t_all[m] > 0 else (0.0 if np.mean(per[m]) == 0 else math.inf)) for m in metrics if per[m]}
        out[name] = {
            "runs": sum(c is not None for c in runs),
            "skill": skill,
            "pass_calibrated": {m: skill[m] <= max(SKILL_MAX, loo[m]) for m in skill},
            "regime_agreement": float(np.mean(regime_hits)) if regime_hits else None,
            "fragments_in_range": float(np.mean(in_range)) if in_range else None,
            "physics": {k: (float(np.mean([p[k] for p in physics])) if physics else None) for k in ("ke_ok", "momentum_ok", "damage_monotone", "finite")},
            "fallbacks": int(sum(p["fallbacks"] for p in physics)),
            "energy_bound_active": int(sum(p["energy_bound_active"] for p in physics)),
        }
    return out


# ------------------------------------------------------------------ report


def aggregate(per_ensemble: dict, families: dict[str, str], candidates: list[str]) -> dict:
    """Per family: mean skill per metric (over ensembles), the calibration, pass shares."""
    agg: dict = {}
    for fam in sorted(set(families.values())):
        ens = [e for eid, e in per_ensemble.items() if families[eid] == fam]
        if not ens:
            continue
        metrics = [m for m in METRICS if all(m in e["metrics"] for e in ens)]
        row = {"ensembles": len(ens), "oracle_member": {m: float(np.mean([e["oracle_member"][m] for e in ens])) for m in metrics}}
        for c in candidates:
            got = [e[c] for e in ens if c in e and e[c]["runs"]]
            if not got:
                continue
            skill = {m: float(np.mean([g["skill"][m] for g in got if m in g["skill"]])) for m in metrics}
            bar = {m: max(SKILL_MAX, row["oracle_member"][m]) for m in metrics}
            ra = [g["regime_agreement"] for g in got if g["regime_agreement"] is not None]
            row[c] = {
                "runs": int(sum(g["runs"] for g in got)),
                "skill": skill,
                "bar": bar,
                "pass": {m: skill[m] <= bar[m] for m in metrics},
                "regime_agreement": float(np.mean(ra)) if ra else None,
                "fragments_in_range": float(np.mean([g["fragments_in_range"] for g in got])),
                "physics": {k: float(np.mean([g["physics"][k] for g in got])) for k in ("ke_ok", "momentum_ok", "damage_monotone", "finite")},
                "fallbacks": int(sum(g["fallbacks"] for g in got)),
                "energy_bound_active": int(sum(g["energy_bound_active"] for g in got)),
            }
        agg[fam] = row
    return agg


def markdown(agg: dict, meta: dict) -> str:
    lines = ["# Closed-loop full-event evaluation", "", f"Generated {meta['generated']}. {meta['note']}", ""]
    for fam, row in agg.items():
        lines.append(f"## {fam} ({row['ensembles']} ensembles)")
        lines.append("")
        metrics = list(row["oracle_member"])
        lines.append("| Candidate | runs | " + " | ".join(metrics) + " | regime agree | fragments in range | KE ok | momentum ok | damage monotone | finite | fallbacks |")
        lines.append("|---|---|" + "---|" * len(metrics) + "---|---|---|---|---|---|---|")
        lines.append("| *held-out oracle member (bar = max(1.25, this))* | | " + " | ".join(f"{row['oracle_member'][m]:.2f}" for m in metrics) + " | | | | | | | |")
        for c, r in row.items():
            if c in ("ensembles", "oracle_member"):
                continue
            cell = lambda m: f"{r['skill'][m]:.2f}{'' if r['pass'][m] else ' ✗'}"
            ra = "-" if r["regime_agreement"] is None else f"{r['regime_agreement']:.0%}"
            p = r["physics"]
            lines.append(f"| {c} | {r['runs']} | " + " | ".join(cell(m) for m in metrics) + f" | {ra} | {r['fragments_in_range']:.0%} | {p['ke_ok']:.0%} | {p['momentum_ok']:.0%} | {p['damage_monotone']:.0%} | {p['finite']:.0%} | {r['fallbacks']} |")
        lines.append("")
    lines.append("Skill = mean distance to the ensemble's members over a typical member's (lower is better; ✗ = above the calibrated bar).")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--candidates", default="ceiling,qs,punch")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--python-dir", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-ensembles", type=int, default=None, help="per family")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--note", default="")
    a = p.parse_args(argv)
    cands = a.candidates.split(",")
    if "model" in cands and not a.checkpoint:
        raise SystemExit("--candidates model needs --checkpoint")
    ensembles = validation_ensembles(a.runs, a.split, a.max_ensembles)
    members = [m for ms in ensembles.values() for m in ms]
    print(f"[cl-eval] {len(ensembles)} ensembles, {len(members)} members, candidates {cands}", flush=True)
    events = {m.name: load_event(m) for m in members}
    samples = {name: sample_frames(ev) for name, ev in events.items()}
    jobs = [(c, m) for c in cands for m in members if not (a.out / "runs" / c / f"{m.name}.json").exists()]
    print(f"[cl-eval] {len(jobs)} runs to do ({len(cands) * len(members) - len(jobs)} already done)", flush=True)
    t0 = time.time()
    done = failed = 0

    def one(c: str, m: Path):
        args, env = candidate_args(c, a, a.out, m.name)
        cmd = [str(a.bin), "event", "--event-dir", str(m), "--samples", ",".join(map(str, samples[m.name]))] + args
        t = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **env))
        line = next((l for l in res.stdout.splitlines() if l.startswith("{")), None)
        dest = a.out / "runs" / c
        dest.mkdir(parents=True, exist_ok=True)
        if res.returncode != 0 or line is None:
            (dest / f"{m.name}.err").write_text(res.stderr[-20000:])
            return c, m, None, time.time() - t
        tmp = dest / f"{m.name}.json.tmp"
        tmp.write_text(line + "\n")
        tmp.replace(dest / f"{m.name}.json")
        return c, m, json.loads(line), time.time() - t

    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(one, c, m) for c, m in jobs]
        for fu in as_completed(futs):
            c, m, d, dt = fu.result()
            n = done + failed + 1
            if d is None:
                failed += 1
                summary = "FAILED"
            else:
                done += 1
                r = d["run"]
                summary = f"exit {r['impactors'][0]['speed_along_v0']:.2f} m/s {r['impactors'][0]['regime']} fragments {r['fragments']} ke {r['checks']['ke_excess_over_e0']:+.3f}"
            el = time.time() - t0
            print(f"[cl-eval] {n}/{len(jobs)} | {c} {m.name}: {summary} | {dt:.0f} s | ETA {el / n * (len(jobs) - n) / 60:.0f} min | failures {failed}", flush=True)
    # Score.
    per_ensemble, families = {}, {}
    for eid, ms in ensembles.items():
        evs = [events[m.name] for m in ms]
        fam = evs[0].context["family"]
        families[eid] = fam
        mass = evs[0].array("static_chunk_mass").reshape(-1).astype(np.float64)
        smp = [samples[m.name] for m in ms]
        # Every member's sample frames, so a candidate of member i compares with member j there.
        union = sorted({f for x in smp for f in x})
        truth = [oracle_state(ev, union) for ev in evs]
        cruns = {}
        for c in cands:
            lst = []
            for m, s in zip(ms, smp):
                f = a.out / "runs" / c / f"{m.name}.json"
                lst.append(candidate_state(json.loads(f.read_text()), s) if f.exists() else None)
            cruns[c] = lst
        per_ensemble[eid] = ensemble_scores(truth, cruns, mass, smp, fam == "anchored_wall")
    agg = aggregate(per_ensemble, families, cands)
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": [str(r) for r in a.runs], "split": a.split, "candidates": cands, "checkpoint": str(a.checkpoint) if a.checkpoint else None,
            "max_ensembles": a.max_ensembles, "failures": failed, "note": a.note or f"Split {a.split}; candidates {', '.join(cands)}; debris-reach routing on for the coupled candidates."}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps({"meta": meta, "families": agg, "ensembles": per_ensemble}, indent=1, default=float))
    (a.out / "report.md").write_text(markdown(agg, meta))
    print(markdown(agg, meta))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
