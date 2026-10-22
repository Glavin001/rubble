"""DAgger with the oracle's island step as the expert (planning session, 2026-10-22).

The model's own states leave the oracle's trajectory. Pushforward may only train on them while
they stay on the oracle's topology (`train._successors`). Everywhere else the correct target
of the next frame is the oracle's step *from the model's state*. P0's harness-A relabel
(`learned_impact_relabel`, stress_learned::relabel) gives exactly that: it replays the event to
the frame, imposes the decoded island state, steps the oracle's island one frame, and writes
the call (inputs built by the Rust call builder from that state) with the oracle's targets.

One round:
1. `export`: run a checkpoint over the training calls (teacher-forced inputs). For every call
   with a successor whose prediction leaves the oracle's topology (a different break set) or
   whose chunk velocities are off by more than `tol`·v, write its decoded frame-end state
   (`learned-impact-decoded-state/1`, the provider's format) and a relabel job
   {event, frame = the parent call's frame, decoded, parent_call, split_group}; P0's
   relabel applies the decoded state and writes frame + 1's call.
2. `relabel`: run P0's CLI (`learned_impact_relabel`, stress-learned) on the jobs: one
   `learned-impact-outcome/4` call per job with `source: "dagger"`. Its regime target is -1
   (undecided: a one-frame relabel cannot know the event's final regime), which the loss
   masks.
3. Train with them as a weighted source: `--dataset <train>@w,<dagger dir>@w'`. They inherit
   the parent's split group, so splits stay leak-free.

    python -m learned_impact.outcome.dagger export --checkpoint best.pt --dataset <train calls> \
        --recordings <p1 runs root> --out <round dir> [--max-jobs N]
    python -m learned_impact.outcome.dagger relabel --round <round dir> --bin <learned_impact_relabel>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from . import decode as D
from . import state_io as IO
from .batch import SCALE_INDEX, HierarchyCache, collate
from .train import Data, _batches, decode_config, load_model


def recording_dirs(root: Path) -> dict[str, Path]:
    """event_id → P1's recording dir (`<root>/*/events/<event_id>`)."""
    return {p.name: p for p in root.glob("*/events/*") if p.is_dir()}


def diverged(c, dec_rows: dict, tol: float) -> bool:
    oracle_broke = {int(k) for k, x in zip(c.arrays["bond_key"].tolist(),
                                           (c.arrays["y_bond_broke"] > 0).tolist()) if x}
    return dec_rows["broke"] != oracle_broke or dec_rows["dv_max"] > tol * c.scales["v"]


@torch.no_grad()
def export(ckpt: Path, data: Data, recordings: dict[str, Path], out: Path, tol: float = 0.05,
           max_jobs: int = 0, device: str = "cpu") -> dict:
    model, ck = load_model(ckpt, device)
    model.eval()
    dcfg = decode_config(ck)
    cache = HierarchyCache(model.cfg.levels)
    calls = [c for c in data.train if (c.event_id, c.call_index) in data.successor]
    (out / "states").mkdir(parents=True, exist_ok=True)
    jobs, n_seen, missing = [], 0, set()
    for bc in _batches(calls, 20000, 64):
        b = collate(bc, data.structures, cache)
        o = model(b.to(device))
        o = {k: v.cpu() for k, v in o.items()}
        dec = D.decode(o, b, dcfg)
        docs = IO.decoded_states(b, o, dec, {"checkpoint": str(ckpt), "dagger": True})
        t = b.t
        cc = t["chunk_call"].numpy()
        dv = np.linalg.norm(dec["chunk_v_end"].numpy() - t["y_chunk_v_end"].numpy(), axis=1)
        bcall = t["bond_call"].numpy()
        broke = dec["bond_broke"].numpy()
        for g, c in enumerate(bc):
            n_seen += 1
            rows = {"broke": {int(k) for k, x in zip(c.arrays["bond_key"].tolist(),
                                                     broke[bcall == g].tolist()) if x},
                    "dv_max": float(dv[cc == g].max()) if (cc == g).any() else 0.0}
            if not diverged(c, rows, tol):
                continue
            rec = recordings.get(c.event_id)
            if rec is None:
                missing.add(c.event_id)
                continue
            sp = out / "states" / f"{c.event_id}__{c.call_index:04d}.json"
            IO.write(sp, docs[g])
            # P0's job: `decoded` is the model's decoded state of frame F, applied as harness A
            # applies it; the relabelled call is frame F+1's (inputs from that state, the
            # oracle's targets)
            jobs.append({"event": str(rec), "frame": int(c.frame), "decoded": str(sp),
                         "parent_call": int(c.call_index), "parent_event_id": c.event_id,
                         "split_group": c.split_group})
            if max_jobs and len(jobs) >= max_jobs:
                break
        if max_jobs and len(jobs) >= max_jobs:
            break
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1))
    summary = {"checkpoint": str(ckpt), "calls_with_successor": n_seen, "jobs": len(jobs),
               "diverged_rate": len(jobs) / max(n_seen, 1), "tol": tol,
               "events_without_recording": sorted(missing)}
    (out / "export.json").write_text(json.dumps(summary, indent=1))
    return summary


def relabel(round_dir: Path, binary: Path, threads: int = 8) -> Path:
    """P0's oracle island step on every job; the calls land in `<round>/calls`."""
    out = round_dir / "calls"
    out.mkdir(parents=True, exist_ok=True)
    with open(round_dir / "relabel.log", "w") as log:
        subprocess.run([str(binary), "--jobs", str(round_dir / "jobs.json"), "--out", str(out),
                        "--threads", str(threads)], stdout=log, stderr=subprocess.STDOUT, check=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--dataset", required=True)
    e.add_argument("--recordings", required=True, help="P1's runs root (…/runs)")
    e.add_argument("--out", required=True)
    e.add_argument("--tol", type=float, default=0.05)
    e.add_argument("--max-jobs", type=int, default=0)
    e.add_argument("--device", default="cpu")
    r = sub.add_parser("relabel")
    r.add_argument("--round", required=True)
    r.add_argument("--bin", required=True)
    r.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        s = export(Path(a.checkpoint), Data(a.dataset), recording_dirs(Path(a.recordings)),
                   Path(a.out), a.tol, a.max_jobs, a.device)
        print(json.dumps(s, indent=1))
    else:
        print(relabel(Path(a.round), Path(a.bin), a.threads))


if __name__ == "__main__":
    main()
