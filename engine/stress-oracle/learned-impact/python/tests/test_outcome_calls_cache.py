"""The on-demand call cache builds once, rebuilds when the event changes, and evicts the
least recently used beyond its budget (with a stand-in builder: no Rust needed)."""
import json
import os
import stat
import time

from learned_impact.outcome_calls_cache import CallCache

FAKE = """#!/usr/bin/env python3
import os, sys, json
args = sys.argv[1:]
out = args[args.index("--out") + 1]
evs = [a for a in args if a not in ("--out", out, "--jobs") and not a.isdigit()]
log = os.environ["FAKE_LOG"]
with open(log, "a") as f:
    f.write(" ".join(os.path.basename(e) for e in evs) + "\\n")
for e in evs:
    d = os.path.join(out, os.path.basename(e))
    os.makedirs(d)
    open(os.path.join(d, "manifest.json"), "w").write("{}")
    open(os.path.join(d, "arrays.bin.zst"), "wb").write(b"x" * 1000)
"""


def make(tmp_path, name, fp="a"):
    d = tmp_path / "events" / name
    d.mkdir(parents=True)
    (d / "outcome.json").write_text(json.dumps({"fingerprint": fp}))
    return d


def test_builds_once_rebuilds_on_change_and_evicts(tmp_path, monkeypatch):
    builder = tmp_path / "builder"
    builder.write_text(FAKE)
    builder.chmod(builder.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    cache = CallCache(tmp_path / "cache", max_gb=2.5 * 16e3 / 1e9, builder=builder)  # room for two events (4 KB blocks x 4 files)
    a, b, c = make(tmp_path, "a"), make(tmp_path, "b"), make(tmp_path, "c")
    assert (cache.path(a) / "manifest.json").exists()
    cache.path(a)
    assert log.read_text().splitlines() == ["a"], "a fresh cached call is not rebuilt"
    time.sleep(0.01)
    cache.prefetch([b])
    time.sleep(0.01)
    cache.path(a)  # a is now the most recently used
    time.sleep(0.01)
    cache.path(c)  # over budget: b, the least recently used, goes
    names = sorted(p.name for p in (tmp_path / "cache").iterdir() if p.is_dir())
    assert names == ["a", "c"]
    (a / "outcome.json").write_text(json.dumps({"fingerprint": "changed"}))
    cache.path(a)
    assert log.read_text().splitlines()[-1] == "a", "a changed event is rebuilt"


def test_a_stream_builds_ahead_in_order_within_the_budget(tmp_path, monkeypatch):
    from learned_impact.outcome_calls_cache import CallStream
    builder = tmp_path / "builder"
    builder.write_text(FAKE)
    builder.chmod(builder.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    # room for 4 events (one 1000-byte file each: 4 KB blocks with the manifest and stamps)
    cache = CallCache(tmp_path / "cache", max_gb=4 * 16e3 / 1e9, builder=builder)
    evs = [make(tmp_path, f"e{i:02d}") for i in range(10)]
    seen = []
    for d in CallStream(cache, evs, ahead=2, batch=2):
        assert (d / "manifest.json").exists(), "a yielded call is on disk"
        seen.append(d.name)
    assert seen == [e.name for e in evs]
    built = [line.split() for line in log.read_text().splitlines()]
    assert sum(len(b) for b in built) == 10 and all(len(b) <= 2 for b in built), "built once each, in batches"
    assert cache.bytes() <= 6 * 16e3, "the cache stays bounded"
