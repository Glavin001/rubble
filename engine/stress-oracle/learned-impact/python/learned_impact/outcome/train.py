"""Trainer for the impact-outcome model (plan, "Training procedure").

AdamW with warm-up + cosine decay, gradient clipping, fixed seeds, the phase-quota sampler,
GNS noise on carried columns, carried-state pushforward, zero-predictor loss normalisation,
per-term gradient norms, and checkpoint selection by the validation skill score (never by
loss). Progress goes to `runs/<job>/progress.log` (a line at least every 30 s) and
`runs/<job>/status.json` (rewritten every 30 s).

    python -m learned_impact.outcome.train --dataset <dir> --out ../runs/<job> --steps 2000
    python -m learned_impact.outcome.train --synthetic 64 --out ../runs/syn-smoke --steps 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import decode as D
from . import losses as Lo
from . import metrics as M
from . import pushforward as P
from . import schema as S
from .batch import Batch, HierarchyCache, collate
from .model import ImpactOutcomeModel, ModelConfig, count_parameters

DEFAULT_QUOTAS = {"contact": 0.35, "secondary": 0.20, "separation": 0.25, "quiet": 0.20}
PHASE_BUCKET = {"contact": "contact", "blast": "contact", "secondary": "secondary",
                "separation": "separation", "settling": "separation", "quiet": "quiet",
                "pre_contact": "quiet"}


@dataclass
class TrainConfig:
    dataset: str = ""
    out: str = "runs/outcome"
    steps: int = 1000
    batch_calls: int = 64  # upper bound on calls per batch
    # Batches are packed to a node + undirected-edge budget, not a fixed call count: the
    # step is launch-latency bound on MPS, so filling it is nearly free (measured: 8 calls of
    # 220 chunks = 37 ms/step). One call over the budget still forms its own batch.
    batch_budget: int = 20000
    lr: float = 1e-3
    min_lr: float = 1e-6
    warmup: int = 50
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    phase_quotas: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_QUOTAS))
    include_unrouted: bool = False  # variant C's non-impact frames; A and B ignore them
    noise_sigma: float = 0.0  # GNS noise on carried columns (σ tuned on validation)
    pushforward_prob: float = 0.0
    pushforward_steps: int = 2  # frames chained per pushforward (plan: 2–4)
    pushforward_topology: bool = True  # successors take the model's breaks, not the oracle's
    eval_every: int = 200
    log_every_s: float = 30.0
    log_every_steps: int = 25
    grad_norm_every: int = 100
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: Lo.LossConfig = field(default_factory=Lo.LossConfig)
    decode: D.DecodeConfig = field(default_factory=D.DecodeConfig)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["decode"]["dtype"] = str(self.decode.dtype)
        return d


# ---- progress -----------------------------------------------------------------------
class Progress:
    def __init__(self, out: Path, total: int):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.log = open(out / "progress.log", "a", buffering=1)
        self.total = total
        self.t0 = time.time()
        self.last_status = 0.0
        self.status = {"state": "starting", "step": 0, "total": total, "last_error": None,
                       "best_score": None, "best_step": None}

    def line(self, msg: str) -> None:
        s = f"[train {time.strftime('%H:%M:%S')}] {msg}"
        print(s, flush=True)
        self.log.write(s + "\n")
        self.log.flush()

    def update(self, force: bool = False, **kw) -> None:
        self.status.update(kw)
        now = time.time()
        if force or now - self.last_status >= 30.0:
            step = self.status.get("step", 0)
            rate = step / max(now - self.t0, 1e-9)
            self.status.update(last_update=time.strftime("%Y-%m-%dT%H:%M:%S"),
                               steps_per_s=rate,
                               eta_s=(self.total - step) / rate if rate > 0 else None)
            tmp = self.out / "status.json.tmp"
            tmp.write_text(json.dumps(self.status, indent=1, default=str))
            tmp.replace(self.out / "status.json")
            self.last_status = now


# ---- data ---------------------------------------------------------------------------
class Data:
    def __init__(self, dataset: str, include_unrouted: bool = False):
        self.events = [S.load_event(p) for p in S.find_events(dataset)]
        if not self.events:
            raise S.SchemaError(f"no events under {dataset}")
        self.structures = {e.manifest["event_id"]: e.structures for e in self.events}
        keep = (lambda c: True) if include_unrouted else (lambda c: c.routed)
        self.train = [c for e in self.events if e.manifest["split"] == "train"
                      for c in e.calls if keep(c)]
        self.val = [c for e in self.events if e.manifest["split"] == "val"
                    for c in e.calls if keep(c)]
        g_train = {e.manifest["split_group"] for e in self.events if e.manifest["split"] == "train"}
        g_other = {e.manifest["split_group"] for e in self.events if e.manifest["split"] != "train"}
        leak = g_train & g_other
        if leak:
            raise S.SchemaError(f"split_group in train and val/test: {sorted(leak)[:5]}")
        self.successor = {}
        idx = {(c.event_id, c.call_index): c for c in self.train}
        for c in self.train:
            if c.prev_call is not None and (c.event_id, c.prev_call) in idx:
                self.successor[(c.event_id, c.prev_call)] = c
        # per-family positive rate of the break labels (the prior correction)
        pos, tot = defaultdict(float), defaultdict(float)
        for c in self.train:
            pos[c.family] += float(c.arrays["y_bond_broke"].sum())
            tot[c.family] += float(len(c.arrays["y_bond_broke"]))
        self.priors = {f: max(pos[f] / max(tot[f], 1), 1e-4) for f in tot}


class PhaseQuotaSampler:
    """Samples calls by phase with set quotas plus every transition call, per epoch."""

    def __init__(self, calls: list[S.Call], quotas: dict[str, float], seed: int):
        self.rng = random.Random(seed)
        self.buckets: dict[str, list[S.Call]] = defaultdict(list)
        for c in calls:
            self.buckets[PHASE_BUCKET[c.phase]].append(c)
        self.transitions = [c for c in calls if c.transition]
        present = {k: v for k, v in quotas.items() if self.buckets.get(k)}
        z = sum(present.values()) or 1.0
        self.quotas = {k: v / z for k, v in present.items()}
        self.n = len(calls)
        self.epoch_counts: Counter = Counter()

    def epoch(self) -> list[S.Call]:
        out = list(self.transitions)
        for k, q in self.quotas.items():
            b = self.buckets[k]
            want = int(round(q * self.n))
            if want <= len(b):
                out.extend(self.rng.sample(b, want))
            else:
                out.extend(self.rng.choices(b, k=want))
        self.rng.shuffle(out)
        self.epoch_counts = Counter(PHASE_BUCKET[c.phase] for c in out)
        return out


def call_size(c: S.Call) -> int:
    k = c.counts
    return (k["chunks"] + k["impactors"] + k["bond"] + k["cracked"] + k["impactor_contact"]
            + k["structure_contact"])


def take_batch(queue: list[S.Call], budget: int, max_calls: int
               ) -> tuple[list[S.Call], list[S.Call]]:
    """Greedy packing in queue order (the sampler already shuffled): never drops or splits
    a call; one oversized call forms its own batch."""
    used, n = 0, 0
    for c in queue:
        sz = call_size(c)
        if n and (used + sz > budget or n >= max_calls):
            break
        used += sz
        n += 1
    return queue[:n], queue[n:]


def _batches(calls: list[S.Call], budget: int, max_calls: int) -> list[list[S.Call]]:
    out, rest = [], list(calls)
    while rest:
        b, rest = take_batch(rest, budget, max_calls)
        out.append(b)
    return out


def _produce(sampler: "PhaseQuotaSampler", cfg: "TrainConfig", coll, nrng):
    """Sampling, noise and collation, in order (deterministic for a seed)."""
    queue: list[S.Call] = []
    epoch = 0
    while True:
        counts = None
        if sum(call_size(c) for c in queue) < cfg.batch_budget:
            queue.extend(sampler.epoch())
            epoch += 1
            counts = dict(sampler.epoch_counts)
        calls, queue = take_batch(queue, cfg.batch_budget, cfg.batch_calls)
        if cfg.noise_sigma > 0:
            calls = [P.add_noise(c, cfg.noise_sigma, nrng) for c in calls]
        yield calls, coll(calls), epoch, counts


class Prefetch:
    """Runs a generator in one background thread, `depth` items ahead (collation is numpy
    and overlaps the GPU step). Errors re-raise in the caller."""

    def __init__(self, make_gen, depth: int = 2):
        import queue
        import threading

        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self.stop = threading.Event()

        def run():
            try:
                for item in make_gen():
                    while not self.stop.is_set():
                        try:
                            self.q.put(("ok", item), timeout=0.5)
                            break
                        except queue.Full:
                            continue
                    if self.stop.is_set():
                        return
            except BaseException as e:  # noqa: BLE001
                self.q.put(("err", e))

        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def next(self):
        kind, item = self.q.get()
        if kind == "err":
            raise item
        return item

    def close(self):
        self.stop.set()


def make_optimizer(params, cfg: "TrainConfig", dev: torch.device):
    """Fused AdamW where the backend has it (one kernel for all parameters instead of
    several per tensor: the optimiser was about a third of an MPS step), else foreach."""
    try:
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=True)


def _lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def _predictions(b: Batch, dec: dict) -> list[dict[str, dict[int, np.ndarray]]]:
    """Per call, keyed predicted frame-end state (input form of `rewrite_carried`)."""
    t = b.t
    cc = t["chunk_call"].cpu().numpy()
    ic = t["imp_call"].cpu().numpy()
    bc = t["bond_call"].cpu().numpy()
    cv, cw = dec["chunk_v_end"].cpu().numpy(), dec["chunk_w_end"].cpu().numpy()
    iv, iw = dec["imp_v_end"].cpu().numpy(), dec["imp_w_end"].cpu().numpy()
    dmg = dec["bond_damage_end"].detach().cpu().numpy()
    peak = np.exp(dec["_log_r"].detach().cpu().numpy())
    broke = dec["bond_broke"].cpu().numpy()
    out = []
    for g, c in enumerate(b.calls):
        a = c.arrays
        s_c, s_i, s_b = cc == g, ic == g, bc == g
        out.append({
            "chunk_v": dict(zip(a["chunk_key"].tolist(), cv[s_c])),
            "chunk_w": dict(zip(a["chunk_key"].tolist(), cw[s_c])),
            "imp_v": dict(zip(a["impactor_key"].tolist(), iv[s_i])),
            "imp_w": dict(zip(a["impactor_key"].tolist(), iw[s_i])),
            "bond_damage": dict(zip(a["bond_key"].tolist(), dmg[s_b])),
            "bond_peak": dict(zip(a["bond_key"].tolist(), peak[s_b])),
            "broke": {int(k) for k, x in zip(a["bond_key"].tolist(), broke[s_b]) if x},
        })
    return out


def grad_norms(terms: dict[str, torch.Tensor], params: list[torch.nn.Parameter]) -> dict:
    out = {}
    for k, v in terms.items():
        if not v.requires_grad:
            out[k] = 0.0
            continue
        gs = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        out[k] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
    return out


def save(path: Path, model, cfg: TrainConfig, norms, priors, step, metrics, temperature=1.0):
    torch.save({"model": model.state_dict(), "model_config": cfg.model.to_dict(),
                "train_config": cfg.to_dict(), "norms": norms, "priors": priors, "step": step,
                "metrics": metrics, "temperature": temperature,
                "schema_version": S.SCHEMA_VERSION}, path)


def load_model(path: str | Path, device="cpu") -> tuple[ImpactOutcomeModel, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ImpactOutcomeModel(ModelConfig(**ck["model_config"]))
    m.load_state_dict(ck["model"])
    return m.to(device), ck


def train(cfg: TrainConfig, data: Data | None = None) -> dict:
    out = Path(cfg.out)
    prog = Progress(out, cfg.steps)
    try:
        return _train(cfg, data, out, prog)
    except Exception as e:  # loud failure, recorded in status.json
        prog.line(f"FAILED: {e!r}")
        prog.update(force=True, state="failed", last_error=traceback.format_exc())
        raise


def _train(cfg: TrainConfig, data: Data | None, out: Path, prog: Progress) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    dev = pick_device(cfg.device)
    data = data or Data(cfg.dataset, cfg.include_unrouted)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    cache = HierarchyCache(cfg.model.levels)
    model = ImpactOutcomeModel(cfg.model).to(dev)
    params = [p for p in model.parameters()]
    balancer = None
    if cfg.loss.balance == "uncertainty":
        balancer = Lo.UncertaintyWeights().to(dev)
    elif cfg.loss.balance != "fixed":
        raise ValueError(f"loss.balance {cfg.loss.balance!r}")
    opt = make_optimizer(params + (list(balancer.parameters()) if balancer else []), cfg, dev)
    monitor = Lo.DominanceMonitor(cfg.loss.dominance_warn, cfg.loss.dominance_patience)
    sampler = PhaseQuotaSampler(data.train, cfg.phase_quotas, cfg.seed)
    nrng = np.random.default_rng(cfg.seed)
    prog.line(f"device={dev} params={count_parameters(model)} train_calls={len(data.train)} "
              f"val_calls={len(data.val)} variant={cfg.model.variant} width={cfg.model.width} "
              f"rounds={cfg.model.rounds} priors={ {k: round(v, 4) for k, v in data.priors.items()} }")
    coll = lambda calls: collate(calls, data.structures, cache)  # noqa: E731
    norm_batches = [coll(bc) for bc in
                    _batches(data.train, cfg.batch_budget, cfg.batch_calls)[:64]]
    norms = Lo.zero_baselines(norm_batches, cfg.loss, data.priors)
    prog.line("zero-predictor baselines: " + " ".join(f"{k}={v:.4g}" for k, v in norms.items()))
    val_batches = [coll(bc) for bc in _batches(data.val, cfg.batch_budget, cfg.batch_calls)]
    best = {"score": float("inf"), "step": None, "metrics": None}
    last_log = time.time()
    acc: dict[str, list[float]] = defaultdict(list)
    epoch = 0
    gnorm_last: dict[str, float] = {}
    prog.update(force=True, state="running")
    batches = Prefetch(lambda: _produce(sampler, cfg, coll, nrng), depth=2)
    for step in range(cfg.steps):
        calls, b, epoch, epoch_counts = batches.next()
        if epoch_counts is not None:
            prog.line(f"epoch {epoch}: phase counts {epoch_counts}")
        for g_ in opt.param_groups:
            g_["lr"] = _lr(step, cfg)
        b = b.to(dev)
        o = model(b)
        terms = Lo.loss_terms(o, b, cfg.loss, data.priors, cfg.decode)
        loss, normed = Lo.total(terms, norms, cfg.loss)
        if balancer is not None:
            loss = balancer(normed, cfg.loss.weights)
        if cfg.pushforward_prob > 0 and random.random() < cfg.pushforward_prob:
            # Pushforward over up to `pushforward_steps` frames: each step's input is the
            # model's own decoded outcome of the previous frame (stopped gradient between
            # steps, Brandstetter et al. 2022), its targets are the oracle's.
            chain_calls, chain_b, chain_o = calls, b, o
            pf_losses = []
            for _k in range(cfg.pushforward_steps):
                pairs = [(i, data.successor[(c.event_id, c.call_index)])
                         for i, c in enumerate(chain_calls)
                         if (c.event_id, c.call_index) in data.successor]
                if not pairs:
                    break
                with torch.no_grad():
                    bc = chain_b if dev.type == "cpu" else chain_b.to("cpu")
                    oc = {k: v.detach().cpu() for k, v in chain_o.items()}
                    dec = D.decode(oc, bc, cfg.decode)
                    dec["_log_r"] = oc["bond_log_r"]
                    preds = _predictions(bc, dec)
                succ = [P.push(chain_calls[i], s_, preds[i],
                               preds[i]["broke"] if cfg.pushforward_topology else None)
                        for i, s_ in pairs]
                bs = coll(succ).to(dev)
                os_ = model(bs)
                ts = Lo.loss_terms(os_, bs, cfg.loss, data.priors, cfg.decode)
                ls, ns = Lo.total(ts, norms, cfg.loss)
                if balancer is not None:
                    ls = balancer(ns, cfg.loss.weights)
                pf_losses.append(ls)
                chain_calls, chain_b, chain_o = succ, bs, os_
            if pf_losses:
                ls = sum(pf_losses) / len(pf_losses)
                loss = loss + ls
                acc["pushforward"].append(float(ls.detach()))
                acc["pushforward_depth"].append(float(len(pf_losses)))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: "
                                     f"{ {k: float(v) for k, v in terms.items()} }")
        opt.zero_grad(set_to_none=True)
        if cfg.grad_norm_every and step % cfg.grad_norm_every == 0:
            eff = balancer.effective(cfg.loss.weights) if balancer else cfg.loss.weights
            gnorm_last = grad_norms({k: eff.get(k, 1.0) * v for k, v in normed.items()}, params)
            shares, flagged = monitor.update(gnorm_last)
            if flagged:
                prog.line(f"WARNING gradient dominance: {flagged} carry "
                          f"{ {k: round(shares[k], 3) for k in flagged} } of the summed "
                          f"per-term gradient norm (V34 failure mode); check the targets and λ")
                prog.update(force=True, dominance_warning={k: shares[k] for k in flagged})
        loss.backward()
        total_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
        opt.step()
        for k, v in normed.items():
            acc[k].append(float(v.detach()))
        acc["loss"].append(float(loss.detach()))
        acc["grad_norm"].append(total_norm)
        now = time.time()
        if (now - last_log >= cfg.log_every_s or step % cfg.log_every_steps == 0
                or step == cfg.steps - 1):
            rate = (step + 1) / max(now - prog.t0, 1e-9)
            eta = (cfg.steps - step - 1) / max(rate, 1e-9)
            terms_s = " ".join(f"{k}={np.mean(v):.3f}" for k, v in acc.items()
                               if k not in ("loss", "grad_norm"))
            tot_g = sum(gnorm_last.values()) or 1.0
            gn = " ".join(f"{k}={v / tot_g:.2f}" for k, v in gnorm_last.items())
            if balancer is not None:
                gn += " λeff[" + " ".join(f"{k}={v:.2f}" for k, v in
                                         balancer.effective(cfg.loss.weights).items()) + "]"
            prog.line(f"step {step + 1}/{cfg.steps} ep {epoch} loss={np.mean(acc['loss']):.4f} "
                      f"[{terms_s}] |g|={np.mean(acc['grad_norm']):.3g} gshare[{gn}] "
                      f"lr={_lr(step, cfg):.2e} {rate:.2f} steps/s ETA {eta / 60:.1f} min")
            acc.clear()
            last_log = now
        prog.update(step=step + 1, epoch=epoch)
        if val_batches and ((step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1):
            res = M.evaluate(model, val_batches, cfg.decode, data.priors)
            msg = " ".join(f"{k}={v:.3f}" for k, v in sorted(res.metrics.items()))
            improved = res.score < best["score"]
            if improved:
                best = {"score": res.score, "step": step + 1, "metrics": res.metrics}
                save(out / "best.pt", model, cfg, norms, data.priors, step + 1, res.metrics)
            prog.line(f"val @ {step + 1}: {msg} {'(best)' if improved else ''} "
                      f"best={best['score']:.3f}@{best['step']}")
            prog.update(force=True, best_score=best["score"], best_step=best["step"])
    batches.close()
    # validation-only temperature for p_b, stored with the last checkpoint
    temperature = 1.0
    if val_batches:
        logits, tg = [], []
        with torch.no_grad():
            for vb in val_batches:
                o = model(vb.to(dev))
                sh = Lo.prior_logit_shift(vb, data.priors)
                lg = o["bond_break_logit"].cpu() + (sh.cpu() if sh is not None else 0)
                logits.append(lg.numpy())
                tg.append(np.where(vb.t["y_bond_has_freq"].numpy(), vb.t["y_bond_freq"].numpy(),
                                   (vb.t["y_bond_broke"].numpy() > 0).astype(float)))
        temperature = M.fit_temperature(np.concatenate(logits), np.concatenate(tg))
        prog.line(f"temperature (validation) = {temperature:.3f}")
    save(out / "last.pt", model, cfg, norms, data.priors, cfg.steps, best["metrics"], temperature)
    prog.update(force=True, state="done", step=cfg.steps)
    return {"best": best, "norms": norms, "temperature": temperature}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="generate N synthetic events into <out>/data first (no accuracy claims)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-calls", type=int, default=64, help="max calls per batch")
    ap.add_argument("--batch-budget", type=int, default=20000,
                    help="nodes + undirected edges per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--rounds", default="2,1,2,1,2")
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--reference", action="store_true", help="W64, rounds 3,2,2,2,3")
    ap.add_argument("--variant", default="A", choices=["A", "B"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--pushforward", type=float, default=0.0, help="probability per step")
    ap.add_argument("--pushforward-steps", type=int, default=2)
    ap.add_argument("--pushforward-oracle-topology", action="store_true",
                    help="keep the oracle's live bonds in successors (the old behaviour)")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--energy-tolerance", type=float, default=0.05)
    ap.add_argument("--break-mode", default="threshold", choices=["threshold", "sample"])
    ap.add_argument("--balance", default="fixed", choices=["fixed", "uncertainty"])
    a = ap.parse_args(argv)
    dataset = a.dataset
    if a.synthetic:
        from .synthetic import write_dataset
        dataset = str(Path(a.out) / "data")
        write_dataset(dataset, a.synthetic, ensembles=max(1, a.synthetic // 16))
        from .ensemble import aggregate_dataset
        aggregate_dataset(dataset)
    if not dataset:
        ap.error("--dataset or --synthetic is required")
    mc = ModelConfig.reference() if a.reference else ModelConfig(
        width=a.width, rounds=tuple(int(x) for x in a.rounds.split(",")), levels=a.levels)
    mc.variant = a.variant
    seeds = [int(s) for s in a.seeds.split(",")]
    data = Data(dataset)
    for s in seeds:
        out = Path(a.out) / (f"seed{s}" if len(seeds) > 1 else "")
        cfg = TrainConfig(dataset=dataset, out=str(out), steps=a.steps,
                          batch_calls=a.batch_calls, batch_budget=a.batch_budget, lr=a.lr, seed=s, device=a.device,
                          noise_sigma=a.noise, pushforward_prob=a.pushforward,
                          pushforward_steps=a.pushforward_steps,
                          pushforward_topology=not a.pushforward_oracle_topology,
                          eval_every=a.eval_every, model=dataclasses.replace(mc),
                          decode=D.DecodeConfig(energy_tolerance=a.energy_tolerance,
                                                break_mode=a.break_mode))
        cfg.loss.energy_tolerance = a.energy_tolerance
        cfg.loss.balance = a.balance
        r = train(cfg, data)
        print(json.dumps({"seed": s, "best_score": r["best"]["score"],
                          "best_step": r["best"]["step"]}))


if __name__ == "__main__":
    main()
