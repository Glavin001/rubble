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


# A relabel is admitted only if the oracle's own frame balances energy: the model's state
# may interpenetrate bodies, and the oracle then resolves the overlap by launching them
# (bulk v1, 10 memorised events: 43 of 779 relabels gained up to 400× their input energy).
# Such a target teaches nothing physical.
ADMIT_ENERGY_TOL = 0.05


def admissible(c) -> bool:
    e = c.energy
    gain = (e["KE_out"] + e.get("E_elastic_end", 0.0) - e["KE_in"] - e.get("E_elastic_start", 0.0)
            - e.get("W_gravity", 0.0) - e.get("W_external", 0.0))
    return gain <= ADMIT_ENERGY_TOL * c.scales["E_in"]


def load_relabelled(root: str | Path) -> tuple[list, dict, dict]:
    """The admissible relabelled calls under `root`, their structures, and the counts."""
    from . import schema as S

    calls, structures, n = [], {}, {"relabelled": 0, "admitted": 0}
    for p in S.find_events(root):
        e = S.load_event(p)
        for c in e.calls:
            if not c.routed:
                continue
            n["relabelled"] += 1
            if admissible(c):
                calls.append(c)
                structures[e.manifest["event_id"]] = e.structures
                n["admitted"] += 1
    return calls, structures, n


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
        docs = IO.decoded_states(b, o, dec, {"checkpoint": str(ckpt), "dagger": True}, model=model)
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


def export_from_rollouts(rollouts: Path, recordings: dict[str, Path], split_groups: dict[str, str],
                         out: Path, every: int = 1) -> dict:
    """Relabel jobs from a harness-A model rollout's dumped island states (true DAgger: the
    model's own states, which teacher-forced predictions never reach). `rollouts` holds
    `<event>/index.jsonl` (lines {frame, state, …}) and the states (P0's exact IslandState at
    the start of each routed frame); each becomes the job {event, frame F, state}: the
    oracle's frame-F call from the model's state, with the oracle's targets."""
    jobs, missing = [], set()
    for ev_dir in sorted(p for p in rollouts.iterdir() if (p / "index.jsonl").exists()):
        eid = ev_dir.name
        rec = recordings.get(eid)
        if rec is None:
            missing.add(eid)
            continue
        rows = [json.loads(x) for x in (ev_dir / "index.jsonl").read_text().splitlines() if x]
        for r in rows[::every]:
            jobs.append({"event": str(rec), "frame": int(r["frame"]),
                         "state": str(ev_dir / r["state"]), "parent_call": None,
                         "parent_event_id": eid, "split_group": split_groups.get(eid, "")})
    out.mkdir(parents=True, exist_ok=True)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1))
    summary = {"rollouts": str(rollouts), "jobs": len(jobs), "events_without_recording": sorted(missing)}
    (out / "export.json").write_text(json.dumps(summary, indent=1))
    return summary


def rollouts_from_json(path: Path) -> tuple[Path, dict[str, Path], dict[str, str]]:
    """P0's `rollouts.json` ({events: [{event, states, split, split_group, …}]}) → the states
    root, event_id → recording dir and event_id → split group. Only training rollouts are
    relabelled for training: any other split raises."""
    d = json.loads(Path(path).read_text())
    recs, groups, roots = {}, {}, set()
    for r in d["events"]:
        if r.get("split") != "train":
            raise ValueError(f"{r['event']}: split {r.get('split')!r}: only training rollouts")
        eid = Path(r["event"]).name
        recs[eid], groups[eid] = Path(r["event"]), r["split_group"]
        roots.add(Path(r["states"]).parent)
    if len(roots) != 1:
        raise ValueError(f"{path}: states under {len(roots)} roots")
    return roots.pop(), recs, groups


def relabel(round_dir: Path, binary: Path, threads: int = 8, calls: str | None = None,
            full_state_columns: str | None = None) -> Path:
    """P0's oracle island step on every job; the calls land in `<round>/calls`. A job the
    oracle or the call builder rejects (a nonfinite state) produces no call and is listed
    in `<round>/calls/failures.jsonl` (rev 8)."""
    out = round_dir / "calls"
    out.mkdir(parents=True, exist_ok=True)
    with open(round_dir / "relabel.log", "w") as log:
        subprocess.run([str(binary), "--jobs", str(round_dir / "jobs.json"), "--out", str(out),
                        "--threads", str(threads), *(["--calls", calls] if calls else []),
                        *(["--full-state-columns", full_state_columns] if full_state_columns else [])],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
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
    r.add_argument("--calls", default=None, help="v3 for /3-trained checkpoints (P0's rev-5 binary)")
    r.add_argument("--full-state-columns", default=None,
                   help="a /5 calls manifest whose full-state columns the relabelled calls use (rev 8)")
    x = sub.add_parser("export-rollouts")
    x.add_argument("--rollouts", default=None, help="<run>/states/<candidate>")
    x.add_argument("--rollouts-json", default=None,
                   help="P0's rollouts.json (states, recordings and split groups; training only)")
    x.add_argument("--recordings", default=None)
    x.add_argument("--dataset", default=None, help="calls (for each event's split group)")
    x.add_argument("--out", required=True)
    x.add_argument("--every", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        s = export(Path(a.checkpoint), Data(a.dataset), recording_dirs(Path(a.recordings)),
                   Path(a.out), a.tol, a.max_jobs, a.device)
        print(json.dumps(s, indent=1))
    elif a.cmd == "export-rollouts" and a.rollouts_json:
        root, recs, groups = rollouts_from_json(Path(a.rollouts_json))
        print(json.dumps(export_from_rollouts(root, recs, groups, Path(a.out), a.every), indent=1))
    elif a.cmd == "export-rollouts":
        groups = {}
        for ev in Path(a.rollouts).iterdir():
            m = Path(a.dataset) / ev.name / "manifest.json"
            if m.exists():
                groups[ev.name] = json.loads(m.read_text())["split_group"]
        print(json.dumps(export_from_rollouts(Path(a.rollouts), recording_dirs(Path(a.recordings)),
                                              groups, Path(a.out), a.every), indent=1))
    else:
        print(relabel(Path(a.round), Path(a.bin), a.threads, a.calls, a.full_state_columns))


if __name__ == "__main__":
    main()
