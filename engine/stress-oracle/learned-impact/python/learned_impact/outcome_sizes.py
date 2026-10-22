"""Bytes per recorded event, split by what revision 8 added (planning's size table):

    python -m learned_impact.outcome_sizes RUN_DIR [RUN_DIR ...] [--json OUT]

`base`: what the revision-7 recorder writes for an event with every frame kept (its arrays
and JSON files); `snapshots`: the exact World restart snapshots; `f64`: the f64 copies of
state, exact break times and impactor impulses. Sizes are on-disk (compressed) bytes.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

NEW_ARRAYS = {"break_i32", "break_f64", "frame_impactor_impulse", "frame_impactor_peak_force"}


def event_sizes(ev: Path) -> dict:
    cat = json.loads((ev / "catalog.json").read_text())["arrays"]
    out = {"base": 0, "snapshots": 0, "f64": 0}
    for name, e in cat.items():
        b = (ev / e["path"]).stat().st_size
        out["f64" if name.endswith("_f64") or name in NEW_ARRAYS else "base"] += b
    for f in ev.iterdir():
        if f.is_file():
            out["base"] += f.stat().st_size
    snaps = list((ev / "snapshots").glob("*.zst")) if (ev / "snapshots").exists() else []
    out["snapshots"] = sum(f.stat().st_size for f in snaps)
    ctx = json.loads((ev / "context.json").read_text())
    outcome = json.loads((ev / "outcome.json").read_text())
    out["n_snapshots"] = len(snaps)
    out["snapshot_raw"] = sum(s.get("bytes", 0) for s in ctx.get("snapshots", {}).get("boundaries", []))
    out["frames"] = outcome.get("frames", 0)
    out["wall_s"] = outcome.get("cost", {}).get("event", 0.0)
    out["overhead_s"] = outcome.get("cost", {}).get("overhead", 0.0)
    out["chunks"] = ctx.get("n_chunks", 0)
    out["family"] = f"{ctx.get('family')}/t{ctx.get('tier')}"
    return out


def table(runs: list[Path]) -> dict:
    fam: dict[str, list[dict]] = {}
    for run in runs:
        for ev in sorted((run / "events").iterdir()):
            if ev.is_dir() and (ev / "catalog.json").exists():
                s = event_sizes(ev)
                fam.setdefault(s["family"], []).append(s)
    rows = {}
    for k, evs in sorted(fam.items()):
        mean = lambda key: statistics.fmean(e[key] for e in evs)
        rows[k] = {"events": len(evs), "frames": mean("frames"), "chunks": mean("chunks"),
                   "base": mean("base"), "snapshots": mean("snapshots"), "f64": mean("f64"),
                   "n_snapshots": mean("n_snapshots"), "snapshot_raw": mean("snapshot_raw"),
                   "wall_s": mean("wall_s"), "overhead_s": mean("overhead_s")}
    return rows


def markdown(rows: dict) -> str:
    mb = lambda x: f"{x / 1e6:.2f}"
    lines = ["| family | events | frames | chunks | today (MB) | + snapshots (MB) | + f64 state (MB) | snapshots per event | snapshot ratio (zstd) | wall s (overhead s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in rows.items():
        ratio = r["snapshot_raw"] / r["snapshots"] if r["snapshots"] else 0.0
        lines.append(f"| {k} | {r['events']} | {r['frames']:.0f} | {r['chunks']:.0f} | {mb(r['base'])} | {mb(r['base'] + r['snapshots'])} | "
                     f"{mb(r['base'] + r['snapshots'] + r['f64'])} | {r['n_snapshots']:.1f} | {ratio:.1f}x | {r['wall_s']:.1f} ({r['overhead_s']:.1f}) |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = table(a.runs)
    print(markdown(rows))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
