"""Closed-loop model provider for P0's harness (`learned_p0 ceiling --variant provider
--provider files --dir D`, PROVIDER_PROTOCOL.md on P0's branch).

For each `D/req/<id>/ready` without an answer it reads `request.json`, runs the model on the
request's call record, decodes, and writes `D/resp/<id>.json`
(`learned-impact-decoded-state/1`) atomically. It stops when `D/done` appears.

    python -m learned_impact.outcome.provider --dir D --checkpoint ../runs/<job>/best.pt
    python -m learned_impact.outcome.provider --dir D --mode echo      # answer oracle.json
    python -m learned_impact.outcome.provider --dir D --mode targets   # answer the record's
                                                                       # own targets

`model` needs `call_record` (the Rust call builder); with `call_record: null` it fails loudly
rather than guess features. `echo` needs the harness's `--echo-oracle`. `targets` needs a
call record carrying targets (a recorded frame): the from_targets self-check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from . import decode as D
from . import schema as S
from . import state_io as IO
from .batch import HierarchyCache, collate


class ProviderError(RuntimeError):
    pass


class Provider:
    def __init__(self, root: Path | str, mode: str = "model", checkpoint: str | None = None,
                 device: str = "cpu", log=print):
        self.root = Path(root)
        self.mode = mode
        self.log = log
        self.model = None
        self.n = 0
        self.t_model = 0.0
        if mode == "model":
            if not checkpoint:
                raise ProviderError("--checkpoint is required in model mode")
            from .train import load_model

            self.model, ck = load_model(checkpoint, device)
            self.model.eval()
            self.device = torch.device(device)
            tc = ck["train_config"]
            self.dcfg = D.DecodeConfig(temperature=ck.get("temperature", 1.0),
                                       energy_tolerance=tc["decode"]["energy_tolerance"],
                                       tau=tc["decode"].get("tau", 1.0),
                                       break_mode=tc["decode"].get("break_mode", "threshold"))
            self.cache = HierarchyCache(self.model.cfg.levels)
            sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            self.meta = {"checkpoint": str(checkpoint), "sha256": sha,
                         "decode": {"tau": self.dcfg.tau, "break_mode": self.dcfg.break_mode,
                                    "energy_tolerance": self.dcfg.energy_tolerance,
                                    "temperature": self.dcfg.temperature}}
        elif mode not in ("echo", "targets"):
            raise ProviderError(f"mode {mode!r}")

    def _load_call(self, req_dir: Path, req: dict, targets: bool) -> tuple[S.Call, dict]:
        rec = req.get("call_record")
        if rec is None:
            raise ProviderError(
                f"{req_dir.name}: call_record is null; the model needs the Rust call builder's "
                "record (outcome_calls.rs). Not guessing features from `state`.")
        ev = S.load_event(req_dir / rec, require_targets=targets)
        if len(ev.calls) != 1:
            raise ProviderError(f"{req_dir.name}: call record has {len(ev.calls)} calls, not 1")
        c = ev.calls[0]
        keys = req["keys"]
        if (sorted(c.arrays["chunk_key"].tolist()) != sorted(keys["chunks"])
                or sorted(c.arrays["impactor_key"].tolist()) != sorted(keys["impactors"])
                or sorted(c.arrays["bond_key"].tolist()) != sorted(keys["bonds"])):
            raise ProviderError(f"{req_dir.name}: call record keys differ from the request")
        return c, ev.structures

    def answer(self, req_dir: Path) -> dict:
        req = json.loads((req_dir / "request.json").read_text())
        if self.mode == "echo":
            p = req_dir / "oracle.json"
            if not p.exists():
                raise ProviderError(f"{req_dir.name}: no oracle.json (run the harness with "
                                    "--echo-oracle)")
            return json.loads(p.read_text())
        if self.mode == "targets":
            c, _ = self._load_call(req_dir, req, targets=True)
            return IO.from_targets(c)
        c, st = self._load_call(req_dir, req, targets=False)
        c = S.with_dummy_targets(c)
        t0 = time.perf_counter()
        b = collate([c], {c.event_id: st}, self.cache)
        with torch.no_grad():
            out = self.model(b.to(self.device))
        out = {k: v.cpu() for k, v in out.items()}
        dec = D.decode(out, b, self.dcfg)
        doc = IO.decoded_states(b, out, dec, dict(self.meta))[0]
        self.t_model += time.perf_counter() - t0
        IO.validate(doc, c)
        return doc

    def run(self, poll_s: float = 0.002, idle_timeout_s: float | None = None) -> int:
        req_root, resp_root = self.root / "req", self.root / "resp"
        resp_root.mkdir(parents=True, exist_ok=True)
        last_beat = last_work = time.time()
        while True:
            pending = sorted(p.parent for p in req_root.glob("*/ready")
                             if not (resp_root / f"{p.parent.name}.json").exists())
            for d in pending:
                doc = self.answer(d)
                IO.write(resp_root / f"{d.name}.json", doc)
                self.n += 1
                last_work = time.time()
            if (self.root / "done").exists() and not pending:
                self.log(f"[provider] done: {self.n} calls, model time {self.t_model:.2f} s")
                return self.n
            now = time.time()
            if now - last_beat >= 30:
                self.log(f"[provider] alive: {self.n} calls answered, waiting on {self.root}")
                last_beat = now
            if idle_timeout_s is not None and now - last_work > idle_timeout_s:
                raise ProviderError(f"no request for {idle_timeout_s} s and no done file")
            time.sleep(poll_s)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", default="model", choices=["model", "echo", "targets"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--idle-timeout", type=float, default=None)
    a = ap.parse_args(argv)
    Provider(a.dir, a.mode, a.checkpoint, a.device).run(idle_timeout_s=a.idle_timeout)


if __name__ == "__main__":
    main()
