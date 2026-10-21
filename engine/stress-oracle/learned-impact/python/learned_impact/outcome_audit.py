"""Read and audit `learned-impact-frames/1` datasets (plan P1).

    python -m learned_impact.outcome_audit RUN_DIR [RUN_DIR ...] --out REPORT.md [--json REPORT.json]

A run directory is a generator job (`runs/<job>/`, with `events/`, `summaries.jsonl`,
`failures.jsonl`) or a plain directory of event directories. Every event is checked for
integrity (shapes, causality of the frame arrays, the stopping rule) and the dataset report
gives: a histogram of event durations, frames per phase, the share of events still breaking
in their last 0.5 s (must be 0), the truncated share, cost per event by family, and failures.

numpy only; no torch.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "learned-impact-frames/1"
# Recorder revision 1 wrote the same layout under the name P2 now uses for its call records.
LEGACY_SCHEMAS = {"learned-impact-outcome/1"}
PHASES = ["pre_contact", "contact", "secondary", "separation", "settling"]
DURATION_BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, float("inf")]


@dataclass
class OutcomeEvent:
    path: Path
    context: dict
    outcome: dict
    frames: list
    catalog: dict
    _cache: dict = field(default_factory=dict)

    def array(self, name: str) -> np.ndarray:
        if name not in self._cache:
            e = self.catalog[name]
            dtype = {"f32": "<f4", "i32": "<i4"}[e["dtype"]]
            data = np.fromfile(self.path / e["path"], dtype=dtype)
            n = int(np.prod(e["shape"])) if e["shape"] else 1
            if data.size != n:
                raise ValueError(f"{self.path.name}/{name}: {data.size} values, catalog says {e['shape']}")
            self._cache[name] = data.reshape(e["shape"])
        return self._cache[name]

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def load_event(path: Path) -> OutcomeEvent:
    path = Path(path)
    read = lambda n: json.loads((path / n).read_text())
    ctx = read("context.json")
    if ctx.get("schema") != SCHEMA and not (ctx.get("schema") in LEGACY_SCHEMAS and ctx.get("recorder_revision") == 1):
        raise ValueError(f"{path.name}: schema {ctx.get('schema')!r} != {SCHEMA}")
    return OutcomeEvent(path, ctx, read("outcome.json"), read("frames.json"), read("catalog.json")["arrays"])


def event_dirs(run: Path) -> list[Path]:
    run = Path(run)
    root = run / "events" if (run / "events").is_dir() else run
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "outcome.json").exists())


def check_event(ev: OutcomeEvent) -> list[str]:
    """Integrity problems of one event (empty: none)."""
    bad = []
    F, N, E = ev.n_frames, ev.context["n_chunks"], ev.context["n_bonds"]
    I = ev.context["n_impactors"]
    want = {
        "state_chunk_pos": (F + 1, N, 3), "state_bond_connected": (F + 1, E), "state_impactor": (F + 1, I, 13),
        "frame_bond_break_bin": (F, E), "frame_bond_peak_util": (F, E), "frame_bond_qs_prior_util": (F, E),
        "frame_chunk_routed": (F, N), "frame_phase": (F,), "frame_impactor_contact": (F, I, 4),
        "static_chunk_mass": (1, N), "static_bond_endpoints": (1, E, 2),
    }
    if "state_bond_energy" in ev.catalog:
        want.update({"state_bond_energy": (F + 1, E, 3), "state_bond_mode": (F + 1, E), "state_chunk_q_world": (F + 1, N, 4)})
    if "qs_bond_broken" in ev.catalog:
        want.update({"qs_bond_broken": (F, E), "qs_chunk_vel": (F, N, 3), "qs_impactor": (F, I, 13)})
    for name, shape in want.items():
        if name not in ev.catalog:
            bad.append(f"missing {name}")
        elif tuple(ev.catalog[name]["shape"]) != shape:
            bad.append(f"{name} shape {ev.catalog[name]['shape']} != {list(shape)}")
    if bad:
        return bad
    conn = ev.array("state_bond_connected")
    bins = ev.array("frame_bond_break_bin")
    broke = bins >= 0
    # A bond breaks in a frame only if it was connected at the frame's start, and is then
    # disconnected at its end; every disconnection is a recorded break.
    if np.any(broke & (conn[:-1] == 0)):
        bad.append("a break recorded on a bond already disconnected at the frame start")
    lost = (conn[:-1] == 1) & (conn[1:] == 0)
    if not np.array_equal(lost, broke):
        bad.append(f"breaks ({int(broke.sum())}) != disconnections ({int(lost.sum())})")
    if np.any((bins >= 8) | (bins < -1)):
        bad.append("break bin out of range")
    if int(broke.sum()) != ev.outcome["broken_bonds"]:
        bad.append("outcome broken_bonds != frame breaks")
    phase = ev.array("frame_phase")
    if [PHASES[p] for p in phase] != [f["phase"] for f in ev.frames]:
        bad.append("frame_phase != frames.json phases")
    if "qs_bond_broken" in ev.catalog and np.any((ev.array("qs_bond_broken") == 1) & (conn[:-1] == 0)):
        bad.append("QS branch breaks a bond disconnected at the frame start")
    prior = ev.array("frame_bond_qs_prior_util")
    if np.any((prior >= 0) & (conn[:-1] == 0)):
        bad.append("QS prior on a disconnected bond")
    if not ev.outcome["truncated"]:
        quiet = ev.context["stop_rule"]["quiet_frames"]
        if broke[-quiet:].any():
            bad.append(f"breaks within the last {quiet} frames of a finished event")
    if "state_bond_energy" in ev.catalog:
        en = ev.array("state_bond_energy")
        # Dissipation only accumulates (f32 round-off allowed).
        if np.any(np.diff(en[:, :, :2], axis=0) < -1e-6 * np.maximum(1.0, np.abs(en[1:, :, :2]))):
            bad.append("per-bond dissipation decreased")
    if "frame_bond_dissipated" in ev.catalog:
        fd = ev.array("frame_bond_dissipated")
        if fd.shape != (F, E, 2):
            bad.append(f"frame_bond_dissipated shape {list(fd.shape)}")
        elif np.any(fd < -1e-6 * max(1.0, float(np.abs(fd).max()))):
            bad.append("negative per-bond frame dissipation")
    for name, e in ev.catalog.items():
        if e["dtype"] == "f32" and not np.all(np.isfinite(ev.array(name))):
            bad.append(f"{name}: nonfinite")
    return bad


def histogram(values, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append({"lo": lo, "hi": hi, "count": int(sum(lo <= v < hi for v in values))})
    return out


def audit(runs: list[Path]) -> dict:
    events, problems, failures = [], {}, []
    for run in runs:
        run = Path(run)
        for p in event_dirs(run):
            ev = load_event(p)
            events.append(ev)
            bad = check_event(ev)
            if bad:
                problems[p.name] = bad
        f = run / "failures.jsonl"
        if f.exists():
            failures += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    by_family = defaultdict(list)
    for ev in events:
        by_family[f"{ev.context['family']}/t{ev.context['tier']}"].append(ev)
    durations = [ev.outcome["sim_seconds"] for ev in events]
    phases = Counter()
    for ev in events:
        phases.update(f["phase"] for f in ev.frames)
    # Finished events only: a truncated one may still be breaking (it is excluded anyway).
    still_breaking = [ev.path.name for ev in events if ev.outcome["breaks_last_half_second"] > 0 and not ev.outcome["truncated"]]
    truncated = [ev.path.name for ev in events if ev.outcome["truncated"]]
    fam = {}
    for k, evs in sorted(by_family.items()):
        cost = np.array([ev.outcome["cost"]["event"] for ev in evs])
        parts = {c: float(np.mean([ev.outcome["cost"].get(c, 0.0) for ev in evs])) for c in ["oracle", "qs_branch", "punch_branch", "qs_prior", "overhead"]}
        sim = np.array([ev.outcome["sim_seconds"] for ev in evs])
        chunks = np.array([ev.context["n_chunks"] for ev in evs])
        breaks = np.array([ev.outcome["broken_bonds"] for ev in evs])
        regimes = Counter(i["regime"] for ev in evs for i in ev.outcome["impactors"])
        size = sum(sum(f.stat().st_size for f in ev.path.rglob("*") if f.is_file()) for ev in evs)
        fam[k] = {
            "events": len(evs),
            "cost_s": {"mean": float(cost.mean()), "median": float(np.median(cost)), "p90": float(np.quantile(cost, 0.9)), "max": float(cost.max()), "parts_mean": parts},
            "cost_per_chunk_sim_s": float(np.median(cost / (chunks * np.maximum(sim, 1e-9)))),
            "sim_s": {"mean": float(sim.mean()), "median": float(np.median(sim)), "max": float(sim.max())},
            "chunks": {"min": int(chunks.min()), "median": float(np.median(chunks)), "max": int(chunks.max())},
            "no_break_share": float(np.mean(breaks == 0)),
            "breaks_median": float(np.median(breaks)),
            "regimes": dict(regimes),
            "truncated": sum(ev.outcome["truncated"] for ev in evs),
            "bytes_per_event": size / len(evs),
            "frames_per_phase": dict(Counter(f["phase"] for ev in evs for f in ev.frames)),
        }
    splits = Counter(str(ev.context.get("split")) for ev in events)
    ens = defaultdict(list)
    for ev in events:
        e = ev.context.get("ensemble") or {}
        if e.get("id"):
            ens[e["id"]].append(e)
    complete = sum(1 for v in ens.values() if len(v) == v[0]["size"])
    oracle = sorted({json.dumps(ev.context["oracle_config"], sort_keys=True) for ev in events})
    return {
        "events": len(events),
        "oracle_configs": [json.loads(o) for o in oracle],
        "integrity_problems": problems,
        "failures": failures,
        "duration_histogram_s": histogram(durations, DURATION_BINS),
        "frames_per_phase": {p: phases.get(p, 0) for p in PHASES},
        "still_breaking_last_half_second": still_breaking,
        "still_breaking_share": len(still_breaking) / max(len(events), 1),
        "truncated": truncated,
        "truncated_share": len(truncated) / max(len(events), 1),
        "families": fam,
        "splits": dict(splits),
        "ensembles": {"groups": len(ens), "complete": complete, "members": sum(len(v) for v in ens.values())},
    }


def markdown(r: dict) -> str:
    L = [f"# Impact-outcome dataset audit", "", f"Events: **{r['events']}**; failures: **{len(r['failures'])}**; integrity problems: **{len(r['integrity_problems'])}**.", ""]
    for o in r["oracle_configs"]:
        L.append(f"Oracle configuration: `{o['name']}` (provisional: {o['provisional']}; {o['source']}).")
    L += ["", "| Check | Value | Rule |", "|---|---|---|",
          f"| Events still breaking in their last 0.5 s | {len(r['still_breaking_last_half_second'])} ({100 * r['still_breaking_share']:.1f}%) | must be 0 |",
          f"| Truncated at the cap | {len(r['truncated'])} ({100 * r['truncated_share']:.1f}%) | excluded from outcome metrics |",
          f"| Ensembles (complete / groups, members) | {r['ensembles']['complete']} / {r['ensembles']['groups']}, {r['ensembles']['members']} | all members recorded |",
          f"| Events per split | {', '.join(f'{k} {v}' for k, v in sorted(r['splits'].items()))} | layout groups held out whole |", "",
          "## Event durations (simulated s)", "", "| Range | Events |", "|---|---|"]
    for b in r["duration_histogram_s"]:
        if b["count"]:
            L.append(f"| [{b['lo']:g}, {b['hi']:g}) | {b['count']} |")
    L += ["", "## Frames per phase", "", "| Phase | Frames | Share |", "|---|---|---|"]
    tot = max(sum(r["frames_per_phase"].values()), 1)
    for p, n in r["frames_per_phase"].items():
        L.append(f"| {p} | {n} | {100 * n / tot:.1f}% |")
    L += ["", "## Cost per event by family (wall-clock s, one thread per event)", "",
          "| Family | Events | Chunks (median) | Sim s (median) | Mean | Median | p90 | Max | Oracle / QS / punch / prior (mean) | No-break share | Regimes | MB/event |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, f in r["families"].items():
        c, p = f["cost_s"], f["cost_s"]["parts_mean"]
        L.append(f"| {k} | {f['events']} | {f['chunks']['median']:.0f} | {f['sim_s']['median']:.2f} | {c['mean']:.1f} | {c['median']:.1f} | {c['p90']:.1f} | {c['max']:.1f} | "
                 f"{p['oracle']:.1f} / {p['qs_branch']:.1f} / {p['punch_branch']:.1f} / {p['qs_prior']:.2f} | {100 * f['no_break_share']:.0f}% | "
                 f"{', '.join(f'{a} {b}' for a, b in sorted(f['regimes'].items()))} | {f['bytes_per_event'] / 1e6:.1f} |")
    if r["failures"]:
        L += ["", "## Failures", "", "| Event | Error |", "|---|---|"]
        for f in r["failures"]:
            L.append(f"| {f['event_id']} | {str(f['error'])[:200]} |")
    if r["integrity_problems"]:
        L += ["", "## Integrity problems", ""]
        for k, v in r["integrity_problems"].items():
            L.append(f"- `{k}`: {'; '.join(v)}")
    if r["still_breaking_last_half_second"]:
        L += ["", "Still breaking in the last 0.5 s: " + ", ".join(f"`{e}`" for e in r["still_breaking_last_half_second"])]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    r = audit(a.runs)
    text = markdown(r)
    if a.out:
        a.out.write_text(text)
    else:
        print(text)
    if a.json:
        a.json.write_text(json.dumps(r, indent=1))
    return 1 if r["integrity_problems"] or r["still_breaking_last_half_second"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
