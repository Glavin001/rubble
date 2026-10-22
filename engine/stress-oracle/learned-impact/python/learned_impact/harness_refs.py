"""Compact harness-A references for recorded ensembles (the closed-loop evaluation's truth).

Harness A re-simulates each member from its scene and scores candidates against the
recorded members. A reference keeps exactly what that needs, so the recorded frames
(`arrays/`) can be reclaimed:

- per member, verbatim: `scene.json`, `context.json`, `outcome.json`, `frames.json` (the
  re-simulation's inputs; the sample frames come from the phases);
- per member, `reference.npz`: the scorer's arrays at the ensemble's sample frames and the
  last frame (`state_chunk_pos`, `state_chunk_vel`, `state_chunk_cluster`,
  `state_chunk_active`, `state_chunk_anchored`, their dtypes kept, `rows`: the frames
  stored) and `static_chunk_mass`.

A reference root has the recorded run's layout (`<root>/events/<member>/`), so discovery,
`learned_p0 event --event-dir` and the relabel work on it unchanged; `closed_loop_eval`
reads `reference.npz` when present (`RefEvent`).

    python -m learned_impact.harness_refs --runs RUN [RUN ...] --split val,test --out ROOT
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .outcome_audit import load_event

STATE = ["state_chunk_pos", "state_chunk_vel", "state_chunk_cluster", "state_chunk_active", "state_chunk_anchored"]
KEEP = ["scene.json", "context.json", "outcome.json", "frames.json"]


@dataclass
class RefEvent:
    """A reference member as `closed_loop_eval` reads events (`OutcomeEvent`'s interface)."""
    path: Path
    context: dict
    outcome: dict
    frames: list
    arrays: dict

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def state_rows(self) -> dict[int, int]:
        return {int(f): i for i, f in enumerate(self.arrays["rows"])}


def load_ref(path: Path) -> RefEvent:
    path = Path(path)
    read = lambda n: json.loads((path / n).read_text())
    with np.load(path / "reference.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return RefEvent(path, read("context.json"), read("outcome.json"), read("frames.json"), arrays)


def load_any(path: Path):
    """A reference member, or a recorded event."""
    return load_ref(path) if (Path(path) / "reference.npz").exists() else load_event(path)


def main(argv=None) -> int:
    from .closed_loop_eval import sample_frames, validation_ensembles
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--split", default="val,test")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if "sealed" in a.split.split(","):
        raise SystemExit("the sealed split is never read")
    ensembles = validation_ensembles(a.runs, a.split, None)
    n = 0
    for eid, members in ensembles.items():
        evs = [load_event(m) for m in members]
        union = sorted({f for ev in evs for f in sample_frames(ev)})
        for m, ev in zip(members, evs):
            rows = ev.state_rows()
            keep = sorted(set(f for f in union if f in rows) | {ev.n_frames})
            dst = a.out / "events" / m.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in KEEP:
                shutil.copy2(m / f, dst / f)
            arrays = {k: ev.array(k)[[rows[f] for f in keep]] for k in STATE}
            arrays["static_chunk_mass"] = ev.array("static_chunk_mass")
            arrays["rows"] = np.array(keep, np.int64)
            np.savez(dst / "reference.npz", **arrays)
            n += 1
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "index.json").write_text(json.dumps({"source_runs": [str(r) for r in a.runs], "split": a.split, "ensembles": {e: [m.name for m in ms] for e, ms in ensembles.items()}}, indent=1))
    print(f"[refs] {len(ensembles)} ensembles, {n} members -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
