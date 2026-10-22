"""Harness-A rollouts of TRAINING events for DAgger (P2's relabelling).

A checkpoint is rolled out in harness A (the standalone oracle with every routed island's
frame from the model, `learned_p0 event --host a`) on a sample of training events, and every
routed frame's exact island start state is dumped (`STRESS_DUMP_ISLAND_STATES`). The states
are then relabelled by the oracle (`learned_impact_relabel`, `state` jobs). Only training
events: a rollout of a validation or test event would carry its layout into training (the
runner refuses any other split).

    python -m learned_impact.dagger_rollouts --events LIST --n 120 --checkpoint best.pt \\
        --python-dir DIR --bin learned_p0 --out ../runs/dagger-rollouts-<name> [--jobs 8]

The environment carries the harness's decode settings (STRESS_FULL_STATE_COLUMNS, ...).
Writes `<out>/states/<event>/` (index.jsonl and <frame>.msgpack), `<out>/runs/<event>.json`
(the run's summary) and `<out>/rollouts.json` (event dir, states dir, split group, frames).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sample(events: list[Path], n: int) -> list[Path]:
    """`n` events spread evenly over the families (the runs' directories), in order."""
    by: dict[str, list[Path]] = defaultdict(list)
    for e in events:
        by[e.parent.parent.name].append(e)
    out: list[Path] = []
    groups = [sorted(v) for _, v in sorted(by.items())]
    i = 0
    while len(out) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(out) < n:
                # spread within a family: stride through it
                stride = max(1, len(g) * len(groups) // max(n, 1))
                j = i * stride
                if j < len(g):
                    out.append(g[j])
        i += 1
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=Path, required=True, help="a list of training event dirs, one per line")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--python-dir", type=Path, required=True)
    p.add_argument("--bin", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--timeout", type=float, default=300.0)
    a = p.parse_args(argv)
    events = [Path(l.strip()) for l in a.events.read_text().splitlines() if l.strip()]
    chosen = sample(events, a.n)
    for e in chosen:
        split = json.loads((e / "context.json").read_text()).get("split")
        if split != "train":
            raise SystemExit(f"{e}: split {split!r}; rollouts for DAgger are of training events only")
    out = a.out.resolve()
    (out / "runs").mkdir(parents=True, exist_ok=True)
    print(f"[dagger-rollouts] {len(chosen)} training events of {len(events)}", flush=True)

    def one(e: Path):
        states = out / "states" / e.name
        if (out / "runs" / f"{e.name}.json").exists():
            return e, "cached", 0.0
        d = out / "provider" / e.name
        cmd = [str(a.bin), "event", "--event-dir", str(e), "--variant", "provider", "--provider", "model", "--host", "a",
               "--checkpoint", str(a.checkpoint.resolve()), "--python-dir", str(a.python_dir.resolve()), "--dir", str(d),
               "--timeout", str(a.timeout), "--device", a.device]
        env = {**os.environ, "STRESS_DEBRIS_ROUTING": "1", "STRESS_DUMP_ISLAND_STATES": str(states)}
        t = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = next((l for l in res.stdout.splitlines() if l.startswith("{")), None)
        if res.returncode != 0 or line is None:
            (out / "runs" / f"{e.name}.err").write_text(res.stderr[-20000:])
            return e, "FAILED", time.time() - t
        (out / "runs" / f"{e.name}.json").write_text(line + "\n")
        return e, "ok", time.time() - t

    rows, failed, t0 = [], 0, time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(one, e) for e in chosen]
        for k, fu in enumerate(as_completed(futs)):
            e, status, dt = fu.result()
            failed += status == "FAILED"
            el = time.time() - t0
            print(f"[dagger-rollouts] {k + 1}/{len(chosen)} {e.name}: {status} | {dt:.0f} s | ETA {el / (k + 1) * (len(chosen) - k - 1) / 60:.0f} min | failures {failed}", flush=True)
    for e in chosen:
        idx = out / "states" / e.name / "index.jsonl"
        frames = [json.loads(l)["frame"] for l in idx.read_text().splitlines()] if idx.exists() else []
        ctx = json.loads((e / "context.json").read_text())
        rows.append({"event": str(e), "states": str(out / "states" / e.name), "split": ctx.get("split"), "split_group": ctx.get("split_group"), "frames": frames})
    (out / "rollouts.json").write_text(json.dumps({"checkpoint": str(a.checkpoint), "events": rows}, indent=1))
    print(f"[dagger-rollouts] {sum(len(r['frames']) for r in rows)} routed frames dumped; {out / 'rollouts.json'}", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
