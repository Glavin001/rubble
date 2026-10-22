"""Cache-backed training data: chunked, read into memory, within the cache's disk bound."""

import shutil
import textwrap

import numpy as np

from learned_impact.outcome import schema as S
from learned_impact.outcome.synthetic import make_event
from learned_impact.outcome.train import _cached_train_events

# A stand-in for P1's CallCache (its API: prefetch, path, bytes): "building" copies a
# prebuilt calls dir; the LRU keeps at most max_gb, sparing the chunk just built.
FAKE = textwrap.dedent('''
    import shutil
    from pathlib import Path

    def _b(p):
        return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())

    class CallCache:
        def __init__(self, root, max_gb, builder, jobs=4):
            self.root, self.src, self.max = Path(root), Path(builder), int(max_gb * 1e9)
            self.root.mkdir(parents=True, exist_ok=True)
            self.order = []

        def prefetch(self, events):
            names = [Path(e).name for e in events]
            for n in names:
                if not (self.root / n).exists():
                    shutil.copytree(self.src / n, self.root / n)
                    self.order.append(n)
            for n in list(self.order):
                if self.bytes() <= self.max:
                    break
                if n not in names:
                    shutil.rmtree(self.root / n)
                    self.order.remove(n)

        def path(self, event):
            return self.root / Path(event).name

        def bytes(self):
            return _b(self.root)
''')


def _setup(tmp_path):
    built = tmp_path / "built"
    for i in range(6):
        m, cs, st = make_event(300 + i, split="train")
        S.write_event(built / m["event_id"], m, cs, st)
    one = max(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in built.iterdir())
    mod = tmp_path / "fake_cache.py"
    mod.write_text(FAKE)
    recs = sorted(str(tmp_path / "rec" / p.name) for p in built.iterdir())
    return built, one, mod, recs


def test_a_cache_holding_one_pass_is_kept(tmp_path):
    built, one, mod, recs = _setup(tmp_path)
    root = tmp_path / "cache"
    cfg = {"module": str(mod), "root": str(root), "builder": str(built), "chunk": 2,
           "max_gb": 9 * one / 1e9, "headroom_gb": 2.1 * one / 1e9}
    _cached_train_events(recs, cfg)
    assert len(list(root.iterdir())) == 6


def test_chunked_cache_stays_bounded_and_loads_into_memory(tmp_path):
    built = tmp_path / "built"
    for i in range(6):
        m, cs, st = make_event(300 + i, split="train")
        S.write_event(built / m["event_id"], m, cs, st)
    one = max(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in built.iterdir())
    mod = tmp_path / "fake_cache.py"
    mod.write_text(FAKE)
    recs = sorted(str(tmp_path / "rec" / p.name) for p in built.iterdir())
    root = tmp_path / "cache"
    # room for two events plus one chunk (two events) of headroom: a pass never fits
    cfg = {"module": str(mod), "root": str(root), "builder": str(built), "chunk": 2,
           "max_gb": 4.2 * one / 1e9, "headroom_gb": 2.1 * one / 1e9}
    events, _ = _cached_train_events(recs, cfg)
    assert [e.manifest["event_id"] for e in events] == [p.rsplit("/", 1)[1] for p in recs]
    assert not list(root.iterdir())  # smaller than one pass: cleared after loading
    ref = S.load_event(built / events[0].manifest["event_id"])
    for k, v in ref.calls[0].arrays.items():
        np.testing.assert_array_equal(events[0].calls[0].arrays[k], v)
