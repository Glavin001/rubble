"""Closed-loop full-event evaluation (remaining work, step 2 of the plan's "Goal and definition
of done"): every validation wall event of P1's dataset run whole, from its first routed frame
until it settles, through P0's coupled harness (`learned_p0 event`), under each candidate:

  ceiling  the oracle's own frame through the call boundary (window-end, reduced state, the
           island's contacts filtered): what a perfect model achieves; equal to the echo /
           targets provider (bit-identical, `closed-loop-harness-2026-10-21.md`)
  ceiling_full  (diagnostic) the same with the full state carried between calls (hidden
           deformation kept): separates what the reduced state loses
  ceiling_def  the ceiling with each chunk's decoded pose and velocity kept relative to its
           body's rigid fit (`STRESS_DECODED_DEFORMATION=1`) instead of the rigid fit alone:
           the elastic deformation of a still-bonded (anchored, prestressed) wall survives
           the call, from the decoded state alone
  a_ceiling  harness A: the standalone oracle with routed islands through the provider, the
           oracle's island step (bit for bit the oracle; the sanity column)
  a_ceiling_decoded / a_ceiling_def  harness A with the oracle's outcome through the /3
           decoded state (reduced / with deformation): the /3 boundary's own ceiling
  a_ceiling_full2  harness A with the oracle's outcome through decoded state /2
  a_model_start  a /5 checkpoint with STRESS_FULL_BASE=start (no oracle values)
  a_model / a_model_def  harness A with the checkpoint (/3 decoded state; `_def` keeps the
           decoded deformation)
  coupled  (diagnostic) the oracle's own coupled frame, no call boundary: what the coupling
           with rubble alone costs
  qs       CPU `blast_parity` (P1's QS baseline), standalone
  punch    the host's predictive punch, the engine taking each frame with its contacts
  model    P2's checkpoint (`--checkpoint`), through `stress_learned::provider_client`
  model_def  the same with `STRESS_DECODED_DEFORMATION=1`
  <any>_tiered  with the deployed tiered routing (`STRESS_ROUTING_TIERED=1`)
  <any>_hbN  with the hand-back after N quiet routed frames (`STRESS_HANDBACK_FRAMES=N`)

and scored against the event's oracle ensemble (P1's recorded members) with the acceptance
metrics: impactor exit speed and regime, partition (1 - ARI), fragment sizes (W1 of mass-
weighted ln mass), fragment count and largest fraction in the ensemble's range, fragment
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
import re
import shutil
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
    """Ensemble id -> member event dirs (sorted by member), for events of `split` (a comma list).
    The sealed split is never read."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for run in runs:
        for ctx_path in sorted((run / "events").glob("*/context.json")):
            ctx = json.loads(ctx_path.read_text())
            if ctx.get("split") not in split.split(","):
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


# The rubble policy of every coupled candidate: production's (`blast_parity_routed`: 4
# iterations, one step a frame), the planning session's reference decision of 2026-10-22.
ENGINE_POLICY = os.environ.get("LEARNED_P0_ENGINE_POLICY", "production")


def candidate_args(name: str, a, out: Path, event: str) -> tuple[list[str], dict]:
    # `<base>_hbN`: the hand-back (`STRESS_HANDBACK_FRAMES=N`): an island quiet for N
    # routed frames (no decoded break, no fast impactor contact) leaves the routing for the
    # quasi-static path until a fast contact routes it again.
    # `<base>_tiered`: the deployed routing's tiered rule (`STRESS_ROUTING_TIERED=1`,
    # `blast_parity_routed`'s: only a fast impactor contact seeds an island).
    if name.endswith("_tiered"):
        args, env = candidate_args(name[: -len("_tiered")], a, out, event)
        if "--dir" in args:
            args[args.index("--dir") + 1] = str(out / "provider" / name / event)
        return args, {**env, "STRESS_ROUTING_TIERED": "1"}
    m = re.fullmatch(r"(.+)_hb(\d+)", name)
    if m:
        args, env = candidate_args(m.group(1), a, out, event)
        if "--dir" in args:
            args[args.index("--dir") + 1] = str(out / "provider" / name / event)
        return args, {**env, "STRESS_HANDBACK_FRAMES": m.group(2)}
    # Debris-reach routing, and the engine's resting ground overlap handed to the oracle's
    # contact as an offset (`World::handed_ground_overlaps`; without it a body the engine
    # rested in the ground is launched when routed, up to 1000 E0).
    deb = {"STRESS_DEBRIS_ROUTING": "1", "STRESS_HANDED_INDENTS": "1", "STRESS_P0_ENGINE_POLICY": ENGINE_POLICY}
    if name == "ceiling":
        return ["--variant", "provider", "--provider", "oracle"], deb
    if name == "ceiling_full":
        # Diagnostic: the oracle's frame with the full (hidden) state carried between calls.
        return ["--variant", "provider", "--provider", "oracle_full"], deb
    if name == "ceiling_def":
        return ["--variant", "provider", "--provider", "oracle"], {**deb, "STRESS_DECODED_DEFORMATION": "1"}
    if name == "ceiling_def_nojoint":
        # Diagnostic: kept deformation, but each surviving bond's joint history reset at the
        # call boundary (only its damage carried).
        return ["--variant", "provider", "--provider", "oracle"], {**deb, "STRESS_DECODED_DEFORMATION": "1", "STRESS_DECODED_RESET_JOINTS": "1"}
    if name == "qs":
        return ["--variant", "qs"], {}
    # Harness A (the standalone oracle, routed islands through the provider; no rubble):
    if name.startswith("a_"):
        base = {"STRESS_DEBRIS_ROUTING": "1"}
        if name == "a_ceiling":
            # the oracle's island step: the standalone oracle bit for bit (sanity)
            return ["--variant", "provider", "--provider", "oracle", "--host", "a"], base
        if name in ("a_ceiling_decoded", "a_ceiling_def"):
            # the oracle's outcome through the /3 decoded state (reduced, or with deformation)
            env = {**base, "STRESS_DECODED_DEFORMATION": "1"} if name == "a_ceiling_def" else base
            return ["--variant", "provider", "--provider", "oracle_decoded", "--host", "a"], env
        if name == "a_ceiling_full2":
            # the oracle's outcome through decoded state /2 (the full-state contract's ceiling;
            # STRESS_FULL_PREDICTED limits it to a model's predicted columns)
            return ["--variant", "provider", "--provider", "oracle_full2", "--host", "a"], base
        if name in ("a_model", "a_model_def", "a_model_start"):
            d = out / "provider" / name / event
            args = ["--variant", "provider", "--provider", "model", "--host", "a", "--checkpoint", str(a.checkpoint), "--dir", str(d), "--timeout", str(a.timeout), "--device", a.device]
            if a.python_dir:
                args += ["--python-dir", str(a.python_dir)]
            env = {**base, "STRESS_DECODED_DEFORMATION": "1"} if name == "a_model_def" else base
            if name == "a_model_start":
                # a /5 model through decoded state /2 with the unpredicted fields from the frame
                # start (no oracle information); a_model takes them from STRESS_FULL_BASE
                env = {**base, "STRESS_FULL_BASE": "start"}
            return args, env
    if name == "coupled":
        # Diagnostic: the oracle's own coupled frame, no call boundary (the coupling alone).
        return ["--variant", "coupled"], deb
    if name == "punch":
        return ["--variant", "provider", "--provider", "punch"], deb
    if name in ("model", "model_def"):
        d = out / "provider" / name / event
        args = ["--variant", "provider", "--provider", "model", "--checkpoint", str(a.checkpoint), "--dir", str(d), "--timeout", str(a.timeout), "--device", a.device]
        if a.python_dir:
            args += ["--python-dir", str(a.python_dir)]
        return args, ({**deb, "STRESS_DECODED_DEFORMATION": "1"} if name == "model_def" else deb)
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
    return float(wasserstein_distance(np.log(ma), np.log(mb), u_weights=ma, v_weights=mb))


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


def tolerances(v0: float) -> dict:
    """Physical tolerance per metric: the floor of the member spread is tolerance / 1.25
    (P2's pre-registered gates-v1: ram 0.05 v0, partition 0.05, sizes 0.1 on ln mass; ours in
    the same spirit: fragment velocities 0.05 v0, positions 0.05 v0 over the 0.5 s since first
    contact, anchored mass fraction 0.05)."""
    return {"ram": 0.05 * v0, "partition": 0.05, "sizes": 0.1, "motion_v_0.1": 0.05 * v0, "motion_v_0.5": 0.05 * v0,
            "motion_x_0.5": 0.05 * v0 * 0.5, "support": 0.05}


def _skill_in(D: np.ndarray, d_c: float, pool: list[int], floor: float) -> float:
    """A unit at mean distance `d_c` from the members `pool`, over their own typical spread."""
    d_m = np.mean([D[j, [k for k in pool if k != j]].mean() for j in pool])
    return float(d_c / max(d_m, floor))


def loo_gates(D: np.ndarray, tol: float) -> tuple[float, float]:
    """Each oracle member held out and scored against the others by the identical rule (P2's
    `metrics._loo_gates`): mean pass and mean skill / bar."""
    m = len(D)
    floor = max(tol / SKILL_MAX, 1e-12)
    passes, sob = [], []
    for i in range(m):
        others = [k for k in range(m) if k != i]
        s_ = _skill_in(D, D[i, others].mean(), others, floor)
        h = [_skill_in(D, D[j, [k for k in others if k != j]].mean(), [k for k in others if k != j], floor) for j in others] if len(others) >= 3 else []
        bar = max(SKILL_MAX, float(np.mean(h))) if h else SKILL_MAX
        passes.append(s_ <= bar)
        sob.append(s_ / bar)
    return float(np.mean(passes)), float(np.mean(sob)), [[bool(p), float(r)] for p, r in zip(passes, sob)]


def ensemble_scores(truth: list[dict], cands: dict[str, list[dict | None]], mass: np.ndarray, samples: list[list[int]], anchored_family: bool, v0: float) -> dict:
    """P2's gate (gates-v1) per metric: each candidate run (started from member i) scored
    against every member, skill = mean distance / max(the members' typical distance,
    tolerance / 1.25), passing at skill <= max(1.25, the mean held-out member skill); the
    held-out members' own pass rate and skill / bar by the identical rule."""
    n = len(truth)
    # Every pair of members (each member's own sample frames apply to both).
    dmat = [[distance(truth[i], truth[j], mass, samples[i], anchored_family) if i != j else None for j in range(n)] for i in range(n)]
    metrics = [m for m in METRICS if all(m in dmat[0][j] for j in range(1, n))]
    tol = tolerances(v0)
    D = {}
    for m in metrics:
        x = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = 0.5 * (dmat[i][j][m] + dmat[j][i][m])
        D[m] = x
    out: dict = {"metrics": metrics, "members": n, "v0": v0, "tolerance": {m: tol[m] for m in metrics}}
    out["oracle_member"] = {m: dict(zip(("pass_rate", "skill_over_bar", "units"), loo_gates(D[m], tol[m]))) for m in metrics}
    everyone = list(range(n))
    bars, floors = {}, {}
    for m in metrics:
        floors[m] = max(tol[m] / SKILL_MAX, 1e-12)
        h = [_skill_in(D[m], D[m][j, [k for k in everyone if k != j]].mean(), [k for k in everyone if k != j], floors[m]) for j in everyone] if n >= 3 else []
        bars[m] = max(SKILL_MAX, float(np.mean(h))) if h else SKILL_MAX
    out["bar"] = bars
    frag = [t["fragments"] for t in truth]
    regimes = Counter(r for t in truth for r in t["regime"][:1])
    majority, share = regimes.most_common(1)[0]
    out["regime_majority"] = majority if share / n >= 0.8 else None
    for name, runs in cands.items():
        units = defaultdict(list)
        regime_hits, in_range, physics = [], [], []
        for i, c in enumerate(runs):
            if c is None:
                continue
            ds = [distance(c, truth[j], mass, samples[i], anchored_family) for j in range(n)]
            for m in metrics:
                if all(m in x for x in ds):
                    sk = _skill_in(D[m], float(np.mean([x[m] for x in ds])), everyone, floors[m])
                    units[m].append((sk <= bars[m], sk / bars[m]))
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
        out[name] = {
            "runs": sum(c is not None for c in runs),
            "units": {m: [[bool(p), float(r)] for p, r in u] for m, u in units.items()},
            "pass_rate": {m: float(np.mean([p for p, _ in u])) for m, u in units.items()},
            "skill_over_bar": {m: float(np.mean([r for _, r in u])) for m, u in units.items()},
            "regime_agreement": float(np.mean(regime_hits)) if regime_hits else None,
            "fragments_in_range": float(np.mean(in_range)) if in_range else None,
            "physics": {k: (float(np.mean([p[k] for p in physics])) if physics else None) for k in ("ke_ok", "momentum_ok", "damage_monotone", "finite")},
            "fallbacks": int(sum(p["fallbacks"] for p in physics)),
            "energy_bound_active": int(sum(p["energy_bound_active"] for p in physics)),
        }
    return out


V11_PASS_MARGIN = 0.05  # pass-rate shortfall, absolute
V11_SOB_MARGIN = 0.05  # mean skill / bar, relative to the held-out members'


def gate_v11(per: list, n_boot: int = 2000, seed: int = 0) -> dict:
    """gates-v1.1 (pre-registered by the planning session, 2026-10-21): a family metric is
    accepted when the one-sided 95% bootstrap bound (resampling whole ensembles) of
    (held-out member pass rate - candidate pass rate) is <= 5 percentage points AND that of
    (candidate mean skill/bar - member mean skill/bar) / member mean skill/bar is <= 5%.
    `per`: per ensemble, (candidate units [[pass, skill/bar], ...], member stats)."""
    per = [(np.array([u[0] for u in us], float), np.array([u[1] for u in us], float), ms) for us, ms in per if len(us)]
    if not per:
        return {}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        pick = [per[k] for k in rng.integers(0, len(per), len(per))]
        cp = np.mean(np.concatenate([x[0] for x in pick]))
        cs = np.mean(np.concatenate([x[1] for x in pick]))
        mp = np.mean([x[2]["pass_rate"] for x in pick])
        msb = np.mean([x[2]["skill_over_bar"] for x in pick])
        d.append((mp - cp, (cs - msb) / max(msb, 1e-12)))
    d = np.array(d)
    d[~np.isfinite(d)] = np.inf  # a non-finite run is infinitely worse
    hi = np.quantile(d, 0.95, axis=0, method="higher")
    return {"pass_shortfall_ub95": float(hi[0]), "sob_excess_ub95": float(hi[1]), "accept": bool(hi[0] <= V11_PASS_MARGIN and hi[1] <= V11_SOB_MARGIN)}


def aggregate(per_ensemble: dict, families: dict[str, str], candidates: list[str]) -> dict:
    """Per family and metric (P2's family acceptance): accept <=> the candidate's pass rate >=
    the held-out members' AND its mean skill / bar <= theirs, over the family's units (a
    candidate run each; the members by ensemble)."""
    agg: dict = {}
    for fam in sorted(set(families.values())):
        ens = [e for eid, e in per_ensemble.items() if families[eid] == fam]
        if not ens:
            continue
        metrics = [m for m in METRICS if all(m in e["metrics"] for e in ens)]
        member = {m: {k: float(np.mean([e["oracle_member"][m][k] for e in ens])) for k in ("pass_rate", "skill_over_bar")} for m in metrics}
        row = {"ensembles": len(ens), "oracle_member": member}
        for c in candidates:
            got = [e[c] for e in ens if c in e and e[c]["runs"]]
            if not got:
                continue
            units = {m: [u for g in got for u in g["units"].get(m, [])] for m in metrics}
            ens_c = [e for e in ens if c in e and e[c]["runs"]]
            ci = {m: gate_v11([(e[c]["units"].get(m, []), e["oracle_member"][m]) for e in ens_c]) for m in metrics}
            pr = {m: float(np.mean([p for p, _ in units[m]])) if units[m] else float("nan") for m in metrics}
            sob = {m: float(np.mean([r for _, r in units[m]])) if units[m] else float("nan") for m in metrics}
            acc = {m: bool(pr[m] >= member[m]["pass_rate"] and sob[m] <= member[m]["skill_over_bar"]) for m in metrics}
            acc11 = {m: ci[m]["accept"] for m in metrics if ci.get(m)}
            ra = [g["regime_agreement"] for g in got if g["regime_agreement"] is not None]
            row[c] = {
                "runs": int(sum(g["runs"] for g in got)),
                "pass_rate": pr,
                "skill_over_bar": sob,
                "accept": acc,
                "accept_all": all(acc.values()),
                "ci": ci,
                "accept_v11": acc11,
                "accept_all_v11": bool(acc11) and all(acc11.values()),
                "regime_agreement": float(np.mean(ra)) if ra else None,
                "fragments_in_range": float(np.mean([g["fragments_in_range"] for g in got])),
                "physics": {k: float(np.mean([g["physics"][k] for g in got])) for k in ("ke_ok", "momentum_ok", "damage_monotone", "finite")},
                "fallbacks": int(sum(g["fallbacks"] for g in got)),
                "energy_bound_active": int(sum(g["energy_bound_active"] for g in got)),
            }
        agg[fam] = row
    return agg


# ------------------------------------------------------------------ report


def markdown(agg: dict, meta: dict) -> str:
    lines = ["# Closed-loop full-event evaluation", "", f"Generated {meta['generated']}. {meta['note']}", ""]
    for fam, row in agg.items():
        lines.append(f"## {fam} ({row['ensembles']} ensembles)")
        lines.append("")
        if "ceiling_full" in row and not row["ceiling_full"]["accept_all_v11"]:
            bad = ", ".join(m for m, ok in row["ceiling_full"]["accept_v11"].items() if not ok)
            lines.append(f"**Boundary problem: the reference ceiling (`ceiling_full`, the oracle's full state through the call boundary) fails gates-v1.1 on {bad}.** A model cannot be accepted where the perfect model is not; fix the boundary (or check the gate's margin on more ensembles) before training on this family.")
            lines.append("")
        metrics = list(row["oracle_member"])
        lines.append("| Candidate | runs | accepted v1 | accepted v1.1 | " + " | ".join(metrics) + " | regime agree | fragments in range | KE ok | momentum ok | damage monotone | finite | fallbacks |")
        lines.append("|---|---|---|---|" + "---|" * len(metrics) + "---|---|---|---|---|---|---|")
        om = row["oracle_member"]
        lines.append("| *held-out oracle member (reference)* | | | | " + " | ".join(f"{om[m]['pass_rate']:.0%} / {om[m]['skill_over_bar']:.2f}" for m in metrics) + " | | | | | | | |")
        for c, r in row.items():
            if c in ("ensembles", "oracle_member"):
                continue
            cell = lambda m: f"{r['pass_rate'][m]:.0%} / {r['skill_over_bar'][m]:.2f}{'' if r['accept'][m] else ' ✗'}{'' if r['accept_v11'].get(m, False) else ' ✗¹'}"
            ra = "-" if r["regime_agreement"] is None else f"{r['regime_agreement']:.0%}"
            p = r["physics"]
            lines.append(f"| {c} | {r['runs']} | {'yes' if r['accept_all'] else 'no'} | {'yes' if r['accept_all_v11'] else 'no'} | " + " | ".join(cell(m) for m in metrics) + f" | {ra} | {r['fragments_in_range']:.0%} | {p['ke_ok']:.0%} | {p['momentum_ok']:.0%} | {p['damage_monotone']:.0%} | {p['finite']:.0%} | {r['fallbacks']} |")
        lines.append("")
    lines.append("Cells: pass rate / mean skill ÷ bar (P2's gates-v1 rule). Skill = a run's mean distance to the ensemble's members over the members' typical distance, floored at tolerance / 1.25; it passes at skill <= max(1.25, the mean held-out member skill). A metric is accepted (no ✗) when the pass rate is at least the held-out members' and the mean skill ÷ bar at most theirs. ✗¹: fails gates-v1.1 (the one-sided 95% bootstrap bound, resampling ensembles, of the pass-rate shortfall against the held-out members exceeds 5 points, or of the mean skill ÷ bar excess 5%). Tolerances: ram 0.05 v0, partition (1 - ARI) 0.05, sizes (W1, mass-weighted ln mass) 0.1, fragment velocities 0.05 v0, positions 0.05 v0 × 0.5 s, anchored mass fraction 0.05.")
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
    p.add_argument("--no-reference", action="store_true", help="omit the standing ceiling_full column (diagnostic runs of other candidates only)")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-ensembles", type=int, default=None, help="per family")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--note", default="")
    p.add_argument("--dump-island-states", action="store_true", help="a_model*: write each routed frame's exact island start state (<out>/states/<cand>/<member>/<frame>.msgpack, index.jsonl)")
    p.add_argument("--members", type=Path, help="JSON list of member names: run only these (scored against their whole ensembles)")
    a = p.parse_args(argv)
    cands = a.candidates.split(",")
    if "sealed" in a.split.split(","):
        raise SystemExit("the sealed split is never evaluated here")
    # The reference ceiling (`ceiling_full`: the oracle's full state through the call
    # boundary) is a standing column: a family where it fails the gate has a boundary
    # problem, not a training problem.
    if "ceiling_full" not in cands and not a.no_reference:
        cands = ["ceiling_full"] + cands
    if "model" in cands and not a.checkpoint:
        raise SystemExit("--candidates model needs --checkpoint")
    ensembles = validation_ensembles(a.runs, a.split, a.max_ensembles)
    if a.members:
        only = set(json.loads(a.members.read_text()))
        ensembles = {k: ms for k, ms in ensembles.items() if any(m.name in only for m in ms)}
        missing = only - {m.name for ms in ensembles.values() for m in ms}
        if missing:
            raise SystemExit(f"--members not in the {a.split} ensembles of --runs: {sorted(missing)}")
    members = [m for ms in ensembles.values() for m in ms]
    run_members = [m for m in members if m.name in only] if a.members else members
    print(f"[cl-eval] {len(ensembles)} ensembles, {len(members)} members, candidates {cands}", flush=True)
    events = {m.name: load_event(m) for m in members}
    samples = {name: sample_frames(ev) for name, ev in events.items()}
    jobs = [(c, m) for c in cands for m in run_members if not (a.out / "runs" / c / f"{m.name}.json").exists()]
    print(f"[cl-eval] {len(jobs)} runs to do ({len(cands) * len(run_members) - len(jobs)} already done)", flush=True)
    t0 = time.time()
    done = failed = 0

    def one(c: str, m: Path):
        args, env = candidate_args(c, a, a.out, m.name)
        if a.dump_island_states and c.startswith("a_model"):
            # The rollout's own island states (DAgger): <out>/states/<cand>/<member>/
            d = a.out / "states" / c / m.name
            shutil.rmtree(d, ignore_errors=True)
            env = {**env, "STRESS_DUMP_ISLAND_STATES": str(d.resolve())}
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
        scene = json.loads((evs[0].path / "scene.json").read_text())
        v0 = max(float(np.linalg.norm(i["velocity"])) for i in scene["impactors"])
        per_ensemble[eid] = ensemble_scores(truth, cruns, mass, smp, fam == "anchored_wall", v0)
    agg = aggregate(per_ensemble, families, cands)
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": [str(r) for r in a.runs], "split": a.split, "candidates": cands, "checkpoint": str(a.checkpoint) if a.checkpoint else None,
            "max_ensembles": a.max_ensembles, "failures": failed, "note": a.note or f"Split {a.split}; candidates {', '.join(cands)}; debris-reach routing and the handed ground overlaps on for the coupled candidates."}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps({"meta": meta, "families": agg, "ensembles": per_ensemble}, indent=1, default=float))
    (a.out / "report.md").write_text(markdown(agg, meta))
    print(markdown(agg, meta))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
