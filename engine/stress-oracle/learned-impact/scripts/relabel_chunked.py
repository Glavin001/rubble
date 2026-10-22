"""Relabel a DAgger round's jobs in chunks of events, keeping only the calls: P0's binary also
writes each job's start/end island states (`<i>_start/_end.msgpack`, ~6× the calls), which
training does not read; they are deleted after each chunk so the round's disk use stays at
its calls (DAgger output counts toward the shared data budget).

    python ../scripts/relabel_chunked.py --round DIR --bin RELABEL [--full-state-columns M] [--events-per-chunk 10]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learned_impact.outcome.dagger import relabel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--round", required=True)
ap.add_argument("--bin", required=True)
ap.add_argument("--full-state-columns", default=None)
ap.add_argument("--events-per-chunk", type=int, default=10)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
rd = Path(a.round)
jobs = json.loads((rd / "jobs.json").read_text())
events = list(dict.fromkeys(j["event"] for j in jobs))
for k in range(0, len(events), a.events_per_chunk):
    part = set(events[k:k + a.events_per_chunk])
    cd = rd / "chunks" / f"{k // a.events_per_chunk:03d}"
    if (cd / "done").exists():
        continue
    cd.mkdir(parents=True, exist_ok=True)
    (cd / "jobs.json").write_text(json.dumps([j for j in jobs if j["event"] in part], indent=1))
    out = relabel(cd, Path(a.bin), a.threads, None, a.full_state_columns)
    for f in out.glob("*.msgpack"):
        f.unlink()
    (cd / "done").write_text("")
    n = sum(1 for _ in out.glob("*_call"))
    print(f"chunk {cd.name}: {len(part)} events, {n} calls", flush=True)
print("RELABEL_DONE", flush=True)
