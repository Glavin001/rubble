"""Calls on demand for training events (the 20 GB budget's lever, planning decision of
2026-10-21).

Evaluation events (validation, test, sealed) have their calls built once and cached
permanently (`learned_impact_outcome_calls --eval-only`). Training events, ensemble members
included, have theirs built when a trainer first asks for them, by the same Rust builder (the single
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
import threading
import time
from pathlib import Path


def _dir_bytes(p: Path) -> int:
    """Disk space (allocated blocks, as `du` and the generator's budget guard count)."""
    return sum(f.stat().st_blocks * 512 for f in p.rglob("*") if f.is_file())


class CallCache:
    def __init__(self, root, max_gb: float, builder, jobs: int = 4):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * 1e9)
        self.builder = Path(builder)
        self.jobs = jobs
        self._lock = threading.RLock()
        self.pinned: set[str] = set()  # built ahead and not yet consumed: never evicted

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
        with self._lock:
            self._build_locked(events)

    def _build_locked(self, events: list[Path]) -> None:
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
            if d.name in keep or d.name in self.pinned:
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


class CallStream:
    """Training events' call directories in a given order, built ahead in a background
    thread so the trainer never waits on the builder once the stream is primed:

        stream = CallStream(cache, train_events, ahead=64, batch=16)
        for call_dir in stream:      # call_dir stays on disk until the next item is asked for
            load(call_dir)

    At most `ahead` built-but-unconsumed events are pinned in the cache (`max_gb` must hold
    them plus the one being read). One builder run per `batch` events, `cache.jobs` threads.
    """

    def __init__(self, cache: CallCache, events, ahead: int = 64, batch: int = 16):
        self.cache, self.events = cache, [Path(e) for e in events]
        self.ahead, self.batch = max(ahead, batch), batch
        self._ready: dict[int, Path] = {}
        self._cv = threading.Condition()
        self._consumed = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            i = 0
            while i < len(self.events):
                with self._cv:
                    self._cv.wait_for(lambda: i - self._consumed < self.ahead)
                chunk = self.events[i:i + self.batch]
                with self.cache._lock:
                    self.cache.pinned.update(e.name for e in chunk)
                self.cache._build([e for e in chunk if not self.cache._fresh(e)])
                with self._cv:
                    for k, e in enumerate(chunk):
                        self._ready[i + k] = self.cache.root / e.name
                    self._cv.notify_all()
                i += len(chunk)
        except BaseException as e:  # surfaced to the consumer
            with self._cv:
                self._error = e
                self._cv.notify_all()

    def __iter__(self):
        prev = None
        for k in range(len(self.events)):
            with self._cv:
                self._cv.wait_for(lambda: k in self._ready or self._error is not None)
                if self._error is not None:
                    raise RuntimeError("call stream builder failed") from self._error
                d = self._ready.pop(k)
                self._consumed = k + 1
                self._cv.notify_all()
            with self.cache._lock:
                if prev is not None:
                    self.cache.pinned.discard(prev.name)
                self.cache._touch(d)
            prev = d
            yield d
        if prev is not None:
            with self.cache._lock:
                self.cache.pinned.discard(prev.name)

