"""Existing training runs → offline Weights & Biases runs (the runs that predate tracking).

    python -m learned_impact.wandb_backfill <run dir or a dir of runs> [...] [--mode offline]

A run dir holds config.json and history.jsonl and/or progress.log (train.py's outputs; a
pilot's or viability's root holds several, found recursively). Each becomes one W&B run
(project rubble-learned-impact, name = the run dir's name, group = its parent's, tag
"backfill"), logged through the same translation as live tracking (tracking.flatten):
history.jsonl's records, plus what only progress.log has for older runs (the gradient
group shares, |g|, steps/s), the final status as the summary, and report-*.json as
artifacts. Runs already tracked live (wandb-run.json) or backfilled (wandb-backfill.json)
are skipped unless --force.

Offline runs land under <run dir>/wandb/; after `wandb login`, upload them with
`wandb sync <run dir>/wandb/offline-run-*` (or `wandb sync --sync-all` from the run dir).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .outcome import tracking as TR

# (term names hold brackets, e.g. fs_joint.plastic.lin[0]: each list ends at its "] " + next field)
STEP = re.compile(r"step (\d+)/\d+ ep (\d+) loss=([-\d.eE+naif]+) \[(.*?)\] \|g\|=([-\d.eE+naif]+)"
                  r" gshare\[(.*?)\] lr=([-\d.eE+]+) ([\d.]+) steps/s")
SHARES = re.compile(r"group gradient shares \(effective, mean of \d+ measurements\) (.*?) → multipliers (.*)$")
VAL = re.compile(r"val @ (\d+): (.*?)(?: \(best\))? best=")
KV = re.compile(r"([\w.\[\]/-]+)=([-\d.eE+]+|nan|inf)")


def _kv(s: str) -> dict:
    out = {}
    for k, v in KV.findall(s):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def progress_records(path: Path) -> list[dict]:
    """train_running / eval records from progress.log (older runs' only record)."""
    recs, last = [], 0
    if not path.exists():
        return recs
    for line in path.read_text(errors="replace").splitlines():
        m = SHARES.search(line)
        if m:  # no step on the line: the balancer measured just after the last logged step
            recs.append({"kind": "group_shares", "step": last, "shares": _kv(m.group(1)),
                         "mult": _kv(m.group(2))})
            continue
        m = STEP.search(line)
        if m:
            step, ep, loss, terms, g, gshare, lr, rate = m.groups()
            gs = _kv(gshare.split(" λeff[")[0])
            last = int(step)
            recs.append({"kind": "train_running", "step": int(step), "epoch": int(ep),
                         "loss": float(loss), "terms": _kv(terms), "grad_norm": float(g),
                         "gshare": gs, "lr": float(lr), "steps_per_s": float(rate)})
            continue
        m = VAL.search(line)
        if m:
            recs.append({"kind": "eval", "step": int(m.group(1)), "metrics": _kv(m.group(2))})
    return recs


def run_records(run: Path) -> list[dict]:
    hist = []
    if (run / "history.jsonl").exists():
        for line in (run / "history.jsonl").read_text().splitlines():
            try:
                hist.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    prog = progress_records(run / "progress.log")
    if not hist:
        return prog
    # history.jsonl first; progress.log adds what older histories lack (shares, |g|, steps/s)
    extra = {r["step"]: r for r in prog if r["kind"] == "train_running"}
    for r in hist:
        if r.get("kind") == "train_running" and r.get("step") in extra:
            e = extra[r["step"]]
            for k in ("gshare", "grad_norm", "steps_per_s"):
                r.setdefault(k, e[k])
    if not any(r.get("kind") == "group_shares" for r in hist):
        hist += [r for r in prog if r["kind"] == "group_shares"]
    return hist


def find_runs(roots: list[Path]) -> list[Path]:
    runs = set()
    for root in roots:
        for p in [root, *root.rglob("*")]:
            if p.is_dir() and (p / "config.json").exists() and (
                    (p / "history.jsonl").exists() or (p / "progress.log").exists()):
                if "wandb" not in p.parts:
                    runs.add(p)
    return sorted(runs)


def backfill(run: Path, mode: str = "offline", force: bool = False) -> str | None:
    if not force and ((run / "wandb-run.json").exists() or (run / "wandb-backfill.json").exists()):
        return None
    cfg = json.loads((run / "config.json").read_text())
    m = cfg.get("model", {})
    tags = ["backfill", f"variant-{m.get('variant', '?')}", f"w{m.get('width', '?')}",
            f"seed{cfg.get('seed', '?')}", *(["full-state"] if m.get("full_state") else []),
            *(["kinematic"] if cfg.get("loss", {}).get("kinematic") else []),
            *(["fine-tune"] if cfg.get("init_checkpoint") else []), *cfg.get("wandb_tags", [])]
    tr = TR.Tracker(mode, run, cfg, tags, job_type="train", marker=False)
    if tr.run is None:
        return None
    for rec in sorted(run_records(run), key=lambda r: (r.get("step") or 0)):
        tr.record(rec)
    st = run / "status.json"
    if st.exists():
        s = json.loads(st.read_text())
        tr.summary({k: s[k] for k in ("best_score", "best_step", "step") if isinstance(s.get(k), (int, float))})
        tr.summary({"finished": float(s.get("state") == "done")})
    for rep in sorted(run.glob("report*.json")):
        try:
            tr.summary({f"report/{rep.stem}": json.loads(rep.read_text())})
        except json.JSONDecodeError:
            pass
        tr.artifact(rep, f"{run.parent.name}-{run.name}-{rep.stem}")
    rid = tr.run.id
    tr.finish(0)
    (run / "wandb-backfill.json").write_text(json.dumps({"id": rid, "mode": mode}))
    return rid


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--mode", default="offline", choices=("offline", "online"))
    ap.add_argument("--force", action="store_true", help="also runs already tracked or backfilled")
    a = ap.parse_args(argv)
    runs = find_runs([Path(r) for r in a.runs])
    done = 0
    for r in runs:
        rid = backfill(r, a.mode, a.force)
        print(f"{r}: {'run ' + rid if rid else 'skipped (tracked already)'}", flush=True)
        done += rid is not None
    print(f"{done} of {len(runs)} runs backfilled ({a.mode}); upload: wandb login, then "
          f"wandb sync <run dir>/wandb/offline-run-*")


if __name__ == "__main__":
    main()
