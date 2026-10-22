"""Reclaim the frames of superseded recordings (user decision, 2026-10-22), keeping what
regenerates them and what describes them.

    python -m learned_impact.outcome_reclaim JOB_DIR ... [--keep-file KEEP] [--splits train] [--apply]

For each event of each job (not listed in `--keep-file`, one event id or ensemble id per
line; only `--splits` if given) it removes `arrays/`, `catalog.json`, `scene.json` and
`replay.json`. It keeps `context.json`, `outcome.json`, `frames.json` and the job's
summaries, and writes `reclaimed.json` with what went and how to regenerate it. The job's
`recipes.json` is zstd-compressed to `recipes.json.zst`, verified byte for byte before the
original is removed. Without `--apply` it only reports. Every recipe regenerates its event
with the current recorder (full state included):

    zstd -dc JOB/recipes.json.zst > /tmp/recipes.json
    learned_impact_outcome_record --recipes /tmp/recipes.json --event EVENT_ID --out DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HEAVY_FILES = ("catalog.json", "scene.json", "replay.json")
HEAVY_DIRS = ("arrays",)


def _blocks(p: Path) -> int:
    if p.is_file():
        return p.stat().st_blocks * 512
    return sum(f.stat().st_blocks * 512 for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def _keep_ids(path: Path | None) -> set[str]:
    if not path:
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def reclaim(jobs: list[Path], keep: set[str], splits: set[str] | None, apply: bool) -> dict:
    report = {"jobs": {}, "freed_bytes": 0}
    for job in jobs:
        freed, events, kept = 0, 0, 0
        for ev in sorted((job / "events").iterdir()):
            ctx_p = ev / "context.json"
            if not ctx_p.exists() or (ev / "reclaimed.json").exists():
                continue
            ctx = json.loads(ctx_p.read_text())
            ens = (ctx.get("parameters", {}).get("ensemble") or {}).get("id")
            if ev.name in keep or (ens and ens in keep):
                kept += 1
                continue
            if splits is not None and ctx.get("split") not in splits:
                continue
            targets = [ev / f for f in HEAVY_FILES] + [ev / d for d in HEAVY_DIRS]
            b = sum(_blocks(t) for t in targets)
            freed += b
            events += 1
            if apply:
                (ev / "reclaimed.json").write_text(json.dumps({
                    "removed": [t.name for t in targets if t.exists()], "bytes": b,
                    "reason": "superseded pre-full-state recording (recorder revision <= 7); user decision 2026-10-22",
                    "regenerate": f"zstd -dc {job.name}/recipes.json.zst > /tmp/r.json && learned_impact_outcome_record --recipes /tmp/r.json --event {ev.name} --out DIR",
                }, indent=1))
                for t in targets:
                    if t.is_dir():
                        shutil.rmtree(t)
                    elif t.exists():
                        t.unlink()
        rec = job / "recipes.json"
        if apply and rec.exists() and events:
            import zstandard
            raw = rec.read_bytes()
            z = zstandard.ZstdCompressor(level=19).compress(raw)
            if zstandard.ZstdDecompressor().decompress(z, max_output_size=len(raw) + 1) != raw:
                raise RuntimeError(f"{rec}: compressed copy does not round-trip")
            (job / "recipes.json.zst").write_bytes(z)
            freed += rec.stat().st_blocks * 512 - _blocks(job / "recipes.json.zst")
            rec.unlink()
        report["jobs"][job.name] = {"events_reclaimed": events, "events_kept": kept, "freed_bytes": freed}
        report["freed_bytes"] += freed
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", nargs="+", type=Path)
    ap.add_argument("--keep-file", type=Path)
    ap.add_argument("--splits")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    splits = set(a.splits.split(",")) if a.splits else None
    r = reclaim(a.jobs, _keep_ids(a.keep_file), splits, a.apply)
    for k, v in r["jobs"].items():
        print(f"{k}: {v['events_reclaimed']} reclaimed, {v['events_kept']} kept, {v['freed_bytes'] / 1e9:.2f} GB")
    print(f"{'freed' if a.apply else 'would free'} {r['freed_bytes'] / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
