"""Weights & Biases tracking for the trainer (optional: `pip install .[tracking]`).

Every record the trainer writes to history.jsonl is also logged to W&B (one translation,
`flatten`, shared with the backfill), plus the run's config, tags and final summary.
Tracking never stops a run: a W&B error disables it with one warning.

Modes (`--wandb`): "off"; "offline" (the default when wandb is installed): the run is
written under `<run dir>/wandb/` and uploaded later with `wandb sync <run dir>/wandb/offline-run-*`
after `wandb login`; "online" (needs `wandb login` first). No API key is ever read or
written here: W&B's own login handles it.
"""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

PROJECT = "rubble-learned-impact"
MODES = ("auto", "off", "offline", "online")


def available() -> bool:
    try:
        import wandb  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def resolve_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"--wandb {mode!r}: one of {MODES}")
    if mode == "auto":  # LEARNED_IMPACT_WANDB overrides the default (the tests set off)
        env = os.environ.get("LEARNED_IMPACT_WANDB", "")
        if env in ("off", "offline", "online"):
            return resolve_mode(env)
        return "offline" if available() else "off"
    if mode != "off" and not available():
        warnings.warn("wandb is not installed (pip install .[tracking]): tracking off")
        return "off"
    return mode


def git_sha(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short=8", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _numeric_leaves(d, prefix: str, out: dict) -> None:
    if isinstance(d, bool):
        out[prefix] = float(d)
    elif isinstance(d, (int, float)):
        out[prefix] = float(d)
    elif isinstance(d, dict):
        for k, v in d.items():
            _numeric_leaves(v, f"{prefix}/{k}" if prefix else str(k), out)


def flatten(rec: dict) -> tuple[int | None, dict]:
    """A history.jsonl record → (step, W&B metrics). Kinds: train_running (the running
    normalised loss terms, LR, gradient norm and group shares, balancer multipliers,
    steps/s), group_shares (the balancer's windowed shares), eval (validation metrics
    including per-family, gate and clamp counts; loss curves on fixed train/val samples)."""
    kind, step, out = rec.get("kind"), rec.get("step"), {}
    if kind == "train_running":
        out["train/loss"] = rec.get("loss")
        _numeric_leaves(rec.get("terms", {}), "train", out)
        for k in ("lr", "epoch", "grad_norm", "steps_per_s"):
            if k in rec:
                out[k] = rec[k]
        _numeric_leaves(rec.get("gshare", {}), "gshare", out)
        _numeric_leaves(rec.get("mult", {}), "mult", out)
    elif kind == "group_shares":
        _numeric_leaves(rec.get("shares", {}), "gshare_window", out)
        _numeric_leaves(rec.get("mult", {}), "mult", out)
    elif kind == "eval":
        out["val/score"] = rec.get("score")
        _numeric_leaves(rec.get("metrics", {}), "val", out)
        for name in ("train", "val"):
            if isinstance(rec.get(name), dict):
                _numeric_leaves(rec[name], f"curve/{name}", out)
        if "epoch" in rec:
            out["epoch"] = rec["epoch"]
    return step, {k: v for k, v in out.items() if v is not None}


def data_revision(data) -> str:
    """schema/builder/recorder revision of the run's data (the first event's manifest)."""
    for e in getattr(data, "events", []) or []:
        m = e.manifest
        return (f"{str(m.get('schema_version', '?')).rsplit('/', 1)[-1]}"
                f"-b{m.get('builder_revision', '?')}-{m.get('recorder_revision', '?')}")
    return "unknown"


def run_group(run_dir: Path) -> str:
    """The run's parent path below the nearest `runs/` ancestor (v7, v7/scaling,
    dagger-n10/r1), else its parent's name."""
    p = Path(run_dir).resolve()
    parts = p.parts
    if "runs" in parts[:-1]:
        i = len(parts) - 1 - list(reversed(parts[:-1])).index("runs") - 1
        rel = parts[i + 1:-1]
        if rel:
            return "/".join(rel)
    return p.parent.name


class Tracker:
    """One W&B run; every method is a no-op when off or after a W&B error."""

    def __init__(self, mode: str, run_dir: Path, config: dict | None = None,
                 tags: list[str] | None = None, group: str | None = None,
                 job_type: str = "train", name: str | None = None, resume_id: str | None = None,
                 marker: bool = True):
        self.run = None
        self.mode = resolve_mode(mode)
        if self.mode == "off":
            return
        try:
            import wandb

            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                project=PROJECT, name=name or run_dir.name, group=group or run_group(run_dir),
                job_type=job_type, tags=[t for t in (tags or []) if t][:64], config=config or {},
                dir=str(run_dir), mode=self.mode, id=resume_id, resume="allow" if resume_id else None,
                settings=wandb.Settings(console="off", silent=True))
            if marker:  # the live training run's id (the backfill skips tracked runs)
                (run_dir / "wandb-run.json").write_text(json.dumps(
                    {"id": self.run.id, "mode": self.mode, "project": PROJECT, "dir": str(run_dir / "wandb")}))
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"wandb tracking off: {e!r}")
            self.run = None

    def _guard(self, f, *a, **kw):
        if self.run is None:
            return
        try:
            f(*a, **kw)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"wandb tracking off: {e!r}")
            self.run = None

    def record(self, rec: dict) -> None:
        step, m = flatten(rec)
        if m:
            self._guard(lambda: self.run.log(m, step=step))

    def summary(self, d: dict) -> None:
        flat: dict = {}
        _numeric_leaves(d, "", flat)
        self._guard(lambda: self.run.summary.update(flat))

    def artifact(self, path: Path, name: str, kind: str = "report") -> None:
        def go():
            import wandb

            art = wandb.Artifact(name.replace("/", "-"), type=kind)
            art.add_file(str(path))
            self.run.log_artifact(art)
        if Path(path).exists():
            self._guard(go)

    def finish(self, exit_code: int = 0) -> None:
        self._guard(lambda: self.run.finish(exit_code=exit_code))
        self.run = None


class HistoryWriter:
    """history.jsonl, with every record also sent to the tracker."""

    def __init__(self, path: Path, tracker: Tracker):
        self.f = open(path, "a", buffering=1)
        self.tracker = tracker

    def write_record(self, rec: dict) -> None:
        self.f.write(json.dumps(rec) + "\n")
        self.tracker.record(rec)

    def close(self) -> None:
        self.f.close()
