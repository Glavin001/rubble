"""Calls on demand for training events (the 20 GB budget's lever, planning decision of
2026-10-21).

Evaluation events (validation, test, sealed and every ensemble member) have their calls built
once and cached permanently (`learned_impact_outcome_calls --eval-only`). Training events
have theirs built when a trainer first asks for them, by the same Rust builder (the single
source of features), into a bounded cache that evicts the least recently used:

    cache = CallCache("runs/calls-cache", max_gb=2.0, builder="runs/bin/learned_impact_outcome_calls")
    cache.prefetch(event_dirs)          # one builder run, in parallel, for a batch of events
    call_dir = cache.path(event_dir)    # built if missing or stale; marked as used

A cached call directory is stale when its event changed (the event's `outcome.json`
fingerprint and the builder's size and modification time are recorded beside it), so a
rebuilt builder or regenerated event never serves old calls.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class CallCache:
    def __init__(self, root, max_gb: float, builder, jobs: int = 4):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * 1e9)
        self.builder = Path(builder)
        self.jobs = jobs

    def _key(self, event: Path) -> dict:
        outcome = json.loads((event / "outcome.json").read_text())
        st = self.builder.stat()
        return {"event": str(event.resolve()), "fingerprint": outcome.get("fingerprint"),
                "builder": [st.st_size, int(st.st_mtime)]}

    def _fresh(self, event: Path) -> bool:
        d = self.root / event.name
        stamp = d / ".source.json"
        return (d / "manifest.json").exists() and stamp.exists() and json.loads(stamp.read_text()) == self._key(event)

    def _build(self, events: list[Path]) -> None:
        if not events:
            return
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            r = subprocess.run([str(self.builder), "--out", tmp, "--jobs", str(self.jobs), *map(str, events)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"call builder failed: {r.stderr[-2000:]}")
            for e in events:
                src, dst = Path(tmp) / e.name, self.root / e.name
                if dst.exists():
                    shutil.rmtree(dst)
                os.replace(src, dst)
                (dst / ".source.json").write_text(json.dumps(self._key(e)))
        self._evict(keep={e.name for e in events})

    def _touch(self, d: Path) -> None:
        (d / ".used").write_text(str(time.time()))

    def _evict(self, keep: set[str]) -> None:
        entries = []
        for d in self.root.iterdir():
            if d.is_dir() and (d / "manifest.json").exists():
                used = d / ".used"
                entries.append((used.stat().st_mtime if used.exists() else 0.0, d))
        total = sum(_dir_bytes(d) for _, d in entries)
        for _, d in sorted(entries, key=lambda x: x[0]):
            if total <= self.max_bytes:
                break
            if d.name in keep:
                continue
            total -= _dir_bytes(d)
            shutil.rmtree(d)

    def prefetch(self, events) -> None:
        """Build the calls of every stale or missing event, in one builder run."""
        events = [Path(e) for e in events]
        self._build([e for e in events if not self._fresh(e)])
        for e in events:
            self._touch(self.root / e.name)

    def path(self, event) -> Path:
        """The call directory of `event` (built if missing or stale), marked as used."""
        event = Path(event)
        if not self._fresh(event):
            self._build([event])
        d = self.root / event.name
        self._touch(d)
        return d

    def bytes(self) -> int:
        return _dir_bytes(self.root)
