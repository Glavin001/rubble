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
import shutil
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import decode as D
from . import tracking as TR
from . import losses as Lo
from . import metrics as M
from . import pushforward as P
from . import schema as S
from .batch import SCALE_INDEX, Batch, HierarchyCache, collate, pad_batch
from .model import ImpactOutcomeModel, ModelConfig, count_parameters

DEFAULT_QUOTAS = {"contact": 0.35, "secondary": 0.20, "separation": 0.25, "quiet": 0.20}
PHASE_BUCKET = {"contact": "contact", "blast": "contact", "secondary": "secondary",
                "separation": "separation", "settling": "separation", "quiet": "quiet",
                "pre_contact": "quiet"}


@dataclass
class TrainConfig:
    dataset: str = ""
    val_dataset: str = ""  # validation from this dataset's val split instead (eval ensembles)
    train_events: str = ""  # a file listing P1 recording dirs: training calls via CallCache
    call_cache: dict = field(default_factory=dict)  # {module, root, builder, max_gb, jobs, chunk}
    out: str = "runs/outcome"
    steps: int = 1000
    batch_calls: int = 64  # upper bound on calls per batch
    # Batches are packed to a node + undirected-edge budget, not a fixed call count: the
    # step is launch-latency bound on MPS, so filling it is nearly free (measured: 8 calls of
    # 220 chunks = 37 ms/step). One call over the budget still forms its own batch.
    batch_budget: int = 20000
    select_tau: bool = True  # choose the break threshold on validation at the end
    # Weights & Biases (tracking.py): auto (offline when wandb is installed) | off | offline |
    # online; extra tags (e.g. "scaling", "overfit", "dagger-round")
    wandb: str = "auto"
    wandb_tags: tuple = ()
    # coverage tiers (outcome.coverage) JSON: selection (best step, τ, temperature) on
    # validation tiers 1+2 only; tier 3 (extrapolation) evaluated and logged, never selected on
    tiers: str = ""
    curve_calls: int = 4000  # calls per fixed sample (train and val) for the loss curves
    init_checkpoint: str = ""  # start from these weights (and their input stats): DAgger rounds
    # `/5`: the training data's full-state column set (P1's manifest.full_state_columns): a
    # live call must write exactly these columns and static split (P0's provider)
    full_state_columns: dict = field(default_factory=dict)
    # Pad every batch to bucketed shapes (exact; MPS recompiles per shape: ~1 s a new one)
    pad_shapes: bool = True
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
    # Pushforward samples are kept only on the oracle's topology (same breaks) with chunk
    # velocities within this fraction of v of the oracle's (`_successors`).
    pushforward_tol: float = 0.05
    eval_every: int = 200
    log_every_s: float = 30.0
    log_every_steps: int = 25
    grad_norm_every: int = 50
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
def _read_list(path: str) -> list[str]:
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def _cached_train_events(event_dirs: list, cfg: dict) -> tuple[list[S.Event], int]:
    """P1's recorded training events → their calls through P1's CallCache (module imported
    from `cfg["module"]`, P1's learned_impact/outcome_calls_cache.py), built in chunks of
    `chunk` events (one parallel builder run each) and read into memory, so the cache's disk
    use stays within `max_gb` (GB, everything included: the LRU holds `max_gb` less one
    chunk's headroom, the builder's staging dir the chunk) even when one pass is larger;
    events still cached are reused."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("outcome_calls_cache", cfg["module"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    evs = [Path(e) for e in event_dirs]
    chunk = int(cfg.get("chunk", 32))
    total_gb = float(cfg.get("max_gb", 1.5))
    headroom_gb = float(cfg.get("headroom_gb", 0.25))  # one chunk (32 events ≈ 0.16 GB) staged
    if total_gb <= headroom_gb:
        raise ValueError(f"call cache max_gb {total_gb} must exceed the {headroom_gb} GB chunk headroom")
    cache = mod.CallCache(cfg["root"], max_gb=total_gb - headroom_gb, builder=cfg["builder"],
                          jobs=int(cfg.get("jobs", 8)))
    t0, events, peak = time.time(), [], 0
    for i in range(0, len(evs), chunk):
        part = evs[i:i + chunk]
        cache.prefetch(part)
        events += [S.load_event(cache.path(e), in_memory=True) for e in part]
        peak = max(peak, cache.bytes())
    # A cache smaller than one pass is never hit by the next run (LRU over a sequential
    # pass evicts each event before it comes round again): clear it, the calls are in memory.
    cleared = not all((Path(cfg["root"]) / e.name / "manifest.json").exists() for e in evs)
    if cleared:
        for e in evs:
            d = Path(cfg["root"]) / e.name
            if (d / "manifest.json").exists():
                shutil.rmtree(d)
    print(f"[data] {len(evs)} training events' calls ready in {time.time() - t0:.0f} s "
          f"(cache peak {peak / 1e9:.2f} GB of {total_gb} GB"
          f"{'; smaller than one pass, cleared' if cleared else ''})", flush=True)
    from .ensemble import aggregate
    groups: dict[str, list[S.Event]] = defaultdict(list)
    for e in events:
        if e.manifest.get("ensemble_id"):
            for c in e.calls:
                c.arrays = dict(c.arrays)
            groups[e.manifest["ensemble_id"]].append(e)
    return events, sum(aggregate(g) for g in groups.values())


def _split_of(path) -> str:
    return json.loads((Path(path) / "manifest.json").read_text()).get("split", "")


def complete_ensembles(paths: list) -> list:
    """`paths` without the members of an incomplete ensemble (fewer members on disk than its
    `ensemble_size`): its frequencies would be estimated from too few members."""
    man = {p: json.loads((Path(p) / "manifest.json").read_text()) for p in paths}
    have: Counter = Counter(m["ensemble_id"] for m in man.values() if m.get("ensemble_id"))
    short = {e for e, n in have.items()
             if any(m.get("ensemble_id") == e and n < int(m.get("ensemble_size", n)) for m in man.values())}
    if short:
        print(f"[data] {len(short)} incomplete ensembles left out: {sorted(short)}", flush=True)
    return [p for p in paths if man[p].get("ensemble_id") not in short]


def _load_events(dataset: str, splits: set[str] | None = None) -> tuple[list[S.Event], int]:
    """Events under `dataset`, only those of `splits` when given: the manifest is read
    first, so other splits (the sealed set above all) are never loaded."""
    paths = [p for p in S.find_events(dataset) if splits is None or _split_of(p) in splits]
    paths = complete_ensembles(paths)
    events = [S.load_event(p) for p in paths]
    if not events:
        raise S.SchemaError(f"no events under {dataset}")
    # Ensemble soft labels (break frequency, perforation frequency, members' exit
    # velocities), aggregated in memory: the dataset on disk is never rewritten.
    from .ensemble import aggregate

    groups: dict[str, list[S.Event]] = defaultdict(list)
    for e in events:
        if e.manifest.get("ensemble_id"):
            groups[e.manifest["ensemble_id"]].append(e)
    aligned = 0
    for evs in groups.values():
        for e in evs:
            for c in e.calls:
                c.arrays = dict(c.arrays)
        aligned += aggregate(evs)
    return events, aligned


def holdout_ballistic_speeds(events: list, positions: tuple) -> list[str]:
    """The in-distribution ballistic test (planning, 2026-10-22): in each *training* ballistic
    family, the ensembles at these positions of its speed order (ensemble ids …_v<NNN>_…)
    become validation, in memory, each its own split group. They share the layout with the
    family's training speeds by design: the test is generalisation to unseen speeds of a
    seen layout. Returns the moved ensemble ids."""
    import re

    fam: dict[str, set] = defaultdict(set)
    for e in events:
        m = e.manifest
        if m.get("ensemble_id") and m["family"].startswith("bl_") and m["split"] == "train":
            fam[m["family"]].add(m["ensemble_id"])
    speed = lambda eid: int(re.search(r"_v(\d+)_", eid).group(1))  # noqa: E731
    moved = {eid for ids in fam.values() for i, eid in enumerate(sorted(ids, key=speed)) if i in positions}
    for e in events:
        m = e.manifest
        if m.get("ensemble_id") in moved:
            m["split"], m["split_group"] = "val", m["ensemble_id"]
            for c in e.calls:
                c.split_group = m["ensemble_id"]
    print(f"[data] in-distribution ballistic val: {len(moved)} training ensembles held out "
          f"(speed positions {sorted(positions)} of {', '.join(f'{f} {len(i)}' for f, i in sorted(fam.items()))})",
          flush=True)
    return sorted(moved)


def holdout_tier1_configs(events: list, frac: float = 0.1, min_configs: int = 3) -> list[str]:
    """Tier-1 validation in memory (planning, 2026-10-22: no oracle time or disk): for each
    training layout (manifest layout_family) with ≥ `min_configs` configurations, about `frac`
    of them (at least one; a whole configuration with all its ensemble members), chosen by a
    stable hash, become validation with their own split group. Returns the moved
    configuration ids (ensemble id, or event id for a single run)."""
    import hashlib

    configs: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in events:
        m = e.manifest
        if m["split"] != "train" or m.get("source", "standalone") == "dagger":
            continue
        configs[m.get("layout_family", "")][m.get("ensemble_id") or m["event_id"]].append(e)
    moved = []
    h = lambda x: hashlib.sha256(x.encode()).hexdigest()  # noqa: E731
    for lay, cfg in sorted(configs.items()):
        if not lay or len(cfg) < min_configs:
            continue
        k = max(1, int(frac * len(cfg) + 0.5))
        for cid in sorted(cfg, key=h)[:k]:
            for e in cfg[cid]:
                e.manifest["split"], e.manifest["split_group"] = "val", f"tier1:{cid}"
                for c in e.calls:
                    c.split_group = f"tier1:{cid}"
            moved.append(cid)
    n_lay = sum(1 for lay, cfg in configs.items() if lay and len(cfg) >= min_configs)
    print(f"[data] tier-1 val: {len(moved)} training configurations held out from {n_lay} layouts "
          f"with ≥ {min_configs} configurations", flush=True)
    return sorted(moved)


class Data:
    """Train calls from `dataset`'s train split; validation calls from its val split, or,
    with `val_dataset`, from that dataset's val split instead (the bulk evaluation
    ensembles). Layout groups may not cross from train into any other split of either."""

    def __init__(self, dataset: str, include_unrouted: bool = False,
                 val_dataset: str | None = None, train_events: list | None = None,
                 call_cache: dict | None = None, extra_train: str = "",
                 holdout_ballistic: tuple = (), holdout_tier1: float = 0.0,
                 exclude_families: tuple = ()):
        # `dataset` may list several roots, `path[@weight],…`: their training calls are
        # mixed by `MixedSampler` with the given shares of every epoch (default: in
        # proportion to their sizes, i.e. plain concatenation).
        self.events, self.ensemble_aligned_calls, self.source_weights = [], 0, {}
        specs = parse_datasets(dataset)
        if val_dataset and any(Path(p).resolve() == Path(val_dataset).resolve() for p, _ in specs):
            val_dataset = None  # one directory with every split (calls-v5): its own val split
        if any(w is not None for _, w in specs) and any(w is None for _, w in specs):
            raise ValueError(f"--dataset {dataset!r}: weight every source or none")
        if train_events:
            # Training calls built on demand from P1's recordings (CallCache: the same Rust
            # builder, byte-identical calls; built once, cached persistently, LRU-bounded);
            # `dataset` adds its resident events (validation, test and resident training
            # ensembles); an event in both raises (duplicate ids).
            self.events, self.ensemble_aligned_calls = _cached_train_events(train_events, call_cache or {})
            specs = [(p, w) for p, w in specs]
        for path, weight in specs:
            evs, n = _load_events(path)  # all splits: resident training ensembles too
            ids = {e.manifest["event_id"] for e in self.events}
            dup = ids & {e.manifest["event_id"] for e in evs}
            if dup:
                raise S.SchemaError(f"event ids in two training datasets: {sorted(dup)[:5]}")
            self.events += evs
            self.ensemble_aligned_calls += n
            if weight is not None:
                for src in {e.manifest.get("source", "standalone") for e in evs}:
                    self.source_weights[src] = self.source_weights.get(src, 0.0) + weight
        val_events = self.events
        if val_dataset:
            val_events, n = _load_events(val_dataset, {"val"})
            self.ensemble_aligned_calls += n
            ids = {e.manifest["event_id"] for e in self.events}
            dup = ids & {e.manifest["event_id"] for e in val_events}
            if dup:
                raise S.SchemaError(f"event ids in both datasets: {sorted(dup)[:5]}")
            self.events = self.events + val_events
        if exclude_families:  # keep a run's data fixed when a dataset grows (e.g. the v7 seeds)
            n0 = len(self.events)
            self.events = [e for e in self.events if e.manifest["family"] not in exclude_families]
            print(f"[data] {n0 - len(self.events)} events of excluded families {sorted(exclude_families)} left out",
                  flush=True)
        if holdout_ballistic:
            holdout_ballistic_speeds(self.events, holdout_ballistic)
        if holdout_tier1:
            holdout_tier1_configs(self.events, holdout_tier1)
        self.structures = {e.manifest["event_id"]: e.structures for e in self.events}
        keep = (lambda c: True) if include_unrouted else (lambda c: c.routed)
        from_val = {id(e) for e in val_events} if val_dataset else set()
        train_events = [e for e in self.events if e.manifest["split"] == "train"
                        and id(e) not in from_val]
        self.train = [c for e in train_events for c in e.calls if keep(c)]
        self.val = [c for e in val_events if e.manifest["split"] == "val"
                    for c in e.calls if keep(c)]
        if extra_train:
            # DAgger relabels (P0's oracle island step from the model's own states): training
            # only, admitted by the oracle's energy balance, mixed in as source "dagger" with
            # the given share of each epoch (the rest stays the recorded data's)
            from .dagger import load_relabelled
            share = 0.25
            for spec_ in extra_train.split(","):
                path_, _, w_ = spec_.partition("@")
                share = float(w_) if w_ else share
                calls_, st_, n_ = load_relabelled(path_)
                held = {e.manifest["split_group"] for e in self.events if e.manifest["split"] != "train"}
                leak = {c.split_group for c in calls_} & held
                if leak:  # relabels of validation/test rollouts would train on held-out layouts
                    raise S.SchemaError(f"DAgger calls from held-out split groups: {sorted(leak)[:5]}")
                for c in calls_:
                    c.source = "dagger"
                self.train += calls_
                self.structures.update(st_)
                print(f"[data] + {n_['admitted']} of {n_['relabelled']} relabelled calls from {path_} "
                      f"({n_['energy_gain']} gain energy, {n_['outside_envelope']} leave the energy envelope)", flush=True)
            others = {c.source for c in self.train if c.source != "dagger"}
            self.source_weights = {"dagger": share, **{o: (1 - share) / max(len(others), 1) for o in others}}
        g_train = {e.manifest["split_group"] for e in train_events}
        g_other = {e.manifest["split_group"] for e in self.events
                   if e.manifest["split"] != "train" or id(e) in from_val}
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


def parse_datasets(spec: str) -> list[tuple[str, float | None]]:
    """`path[@weight],path[@weight]` → [(path, weight or None)]."""
    out = []
    for part in filter(None, (x.strip() for x in spec.split(","))):
        path, _, w = part.rpartition("@") if "@" in part else (part, "", "")
        out.append((path, float(w) if w else None))
    return out


class MixedSampler:
    """One `PhaseQuotaSampler` per data source (`Call.source`), each epoch drawing every
    source's share of the training calls: `weights[source]` normalised, or proportional to
    the sources' sizes when unweighted (identical to a single sampler over all calls)."""

    def __init__(self, calls: list[S.Call], quotas: dict[str, float], seed: int,
                 weights: dict[str, float] | None = None):
        by: dict[str, list[S.Call]] = defaultdict(list)
        for c in calls:
            by[c.source].append(c)
        self.samplers = {s: PhaseQuotaSampler(cs, quotas, seed + i)
                         for i, (s, cs) in enumerate(sorted(by.items()))}
        n = len(calls)
        w = {s: (weights or {}).get(s, len(by[s]) / n if not weights else 0.0) for s in by}
        z = sum(w.values()) or 1.0
        self.share = {s: v / z for s, v in w.items()}
        self.n = n
        self.rng = random.Random(seed)
        self.epoch_counts: Counter = Counter()
        self.source_counts: Counter = Counter()

    def epoch(self) -> list[S.Call]:
        if len(self.samplers) == 1:
            sm = next(iter(self.samplers.values()))
            out = sm.epoch()
            self.epoch_counts = sm.epoch_counts
            self.source_counts = Counter(c.source for c in out)
            return out
        out = []
        for s, sm in self.samplers.items():
            want = int(round(self.share[s] * self.n))
            pool = sm.epoch()
            out += (self.rng.sample(pool, want) if want <= len(pool)
                    else pool + self.rng.choices(pool, k=want - len(pool)))
        self.rng.shuffle(out)
        self.epoch_counts = Counter(PHASE_BUCKET[c.phase] for c in out)
        self.source_counts = Counter(c.source for c in out)
        return out


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
            if len(getattr(sampler, "source_counts", ())) > 1:
                counts["sources"] = dict(sampler.source_counts)
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


def input_stats(calls: list[S.Call], variant: str, full_state: bool = False) -> dict:
    """Per-column mean and std of every input array over the training calls (float64), in
    the model's input order (features, variant B's QS frame, `/5` full state)."""
    from . import fullstate as FS

    def fs_chunk(c):
        return FS.chunk_arrays(c, c.fs["cols"], float(np.linalg.norm(c.gravity)))[0]

    def fs_bond(c):
        return np.repeat(FS.bond_arrays(c, c.fs["cols"], c.fs["scale"], c.fs["scale_names"])[0], 2, 0)

    def cat(name, *extra):
        xs = [np.concatenate([c.arrays[name], *[e(c) if callable(e) else c.arrays[e] for e in extra]], 1)
              for c in calls if len(c.arrays[name])]
        if not xs:  # no rows of this input in the training calls: identity standardisation
            d = sum((e(calls[0]) if callable(e) else calls[0].arrays[e]).shape[1] for e in extra) \
                + calls[0].arrays[name].shape[1]
            return torch.zeros(d), torch.ones(d)
        x = np.concatenate(xs).astype(np.float64)
        return torch.tensor(x.mean(0), dtype=torch.float32), torch.tensor(x.std(0), dtype=torch.float32)

    b = variant == "B"
    ch = (["chunk_qs_feat"] if b else []) + ([fs_chunk] if full_state else [])
    out = {"chunk": cat("chunk_feat", *ch),
           "imp": cat("impactor_feat", *(["impactor_qs_feat"] if b else [])),
           "glob": cat("global_feat")}
    for t in S.EDGE_TYPES:
        if any(len(c.arrays[f"{t}_feat"]) for c in calls):
            ex = (["bond_qs_feat"] if (t == "bond" and b) else []) + \
                 ([fs_bond] if (t == "bond" and full_state) else [])
            out[f"edge_{t}"] = cat(f"{t}_feat", *ex)
    return out


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
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


_PF_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pushforward")


@torch.no_grad()
def _successors(chain_calls, bc, oc, data, cfg, coll):
    """The next pushforward batch from CPU copies of a batch and its outputs, or None.

    Only samples whose prediction stays on the oracle's topology are pushed forward: the
    model breaks exactly the bonds the oracle breaks in this call (so the fragment partition
    is the same), and every chunk's frame-end velocity is within `pushforward_tol`·v of the
    oracle's. The target of the successor is the oracle's next call, which follows from the
    oracle's state; where the model's state differs in topology it describes a different
    wall, and the sample is skipped (counted; DAgger relabels such states). Returns
    (successors, batch, n_candidates, n_skipped)."""
    cand = [(i, data.successor[(c.event_id, c.call_index)]) for i, c in enumerate(chain_calls)
            if (c.event_id, c.call_index) in data.successor]
    if not cand:
        return None
    dec = D.decode(oc, bc, cfg.decode)
    dec["_log_r"] = oc["bond_log_r"]
    dec["_log_q"] = oc["bond_log_q"]
    preds = _predictions(bc, dec)
    t = bc.t
    cc = t["chunk_call"].numpy()
    vref = t["scales"][:, SCALE_INDEX["v"]].numpy()
    dv = np.linalg.norm(dec["chunk_v_end"].numpy() - t["y_chunk_v_end"].numpy(), axis=1)
    pairs = []
    for i, s_ in cand:
        c = chain_calls[i]
        oracle_broke = {int(k) for k, x in zip(c.arrays["bond_key"].tolist(),
                                               (c.arrays["y_bond_broke"] > 0).tolist()) if x}
        sel = cc == i
        close = (not sel.any()) or float(dv[sel].max()) <= cfg.pushforward_tol * vref[i]
        if preds[i]["broke"] == oracle_broke and close:
            pairs.append((i, s_))
    if not pairs:
        return [], None, len(cand), len(cand)
    succ = [P.push(chain_calls[i], s_, preds[i], None) for i, s_ in pairs]
    return succ, coll(succ), len(cand), len(cand) - len(pairs)


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
    margin = np.exp(dec["_log_q"].detach().cpu().numpy())
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
            "bond_margin": dict(zip(a["bond_key"].tolist(), margin[s_b])),
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


def save(path: Path, model, cfg: TrainConfig, norms, priors, step, metrics, temperature=1.0,
         tau: float | None = None):
    """`priors` is stored only when training used the prior reweighting (the inference-time
    logit correction is valid only then); `tau` is the validation-chosen break threshold."""
    torch.save({"model": model.state_dict(), "model_config": cfg.model.to_dict(),
                "train_config": cfg.to_dict(), "norms": norms,
                "priors": priors if cfg.loss.break_reweight == "prior" else None, "step": step,
                "metrics": metrics, "temperature": temperature,
                "tau": cfg.decode.tau if tau is None else tau,
                "schema_version": S.SCHEMA_VERSION}, path)


# The grid must bracket the optimum: the bulk v1 retrain chose 1.0, its old upper end,
# with the score still falling there.
TAU_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)


def select_tau(model, val_batches, cfg: "TrainConfig", priors, prog=None) -> tuple[float, dict]:
    """The break threshold τ (on the failure margin q̂) with the best validation selection
    score: a validation-only operating point, as the plan allows for τ."""
    scores = {}
    for tau in TAU_GRID:
        dc = dataclasses.replace(cfg.decode, tau=tau)
        scores[tau] = M.evaluate(model, val_batches, dc, priors).score
        if prog:
            prog.line(f"tau {tau}: validation score {scores[tau]:.3f}")
    best = min(scores, key=scores.get)
    return best, scores


def decode_config(ck: dict) -> D.DecodeConfig:
    """The deployed decode of a checkpoint: τ, temperature, break mode and seed as stored."""
    tc = ck["train_config"]["decode"]
    return D.DecodeConfig(temperature=ck.get("temperature", 1.0),
                          energy_tolerance=tc["energy_tolerance"],
                          tau=ck.get("tau", tc.get("tau", 1.0)),
                          break_mode=ck.get("break_mode", tc.get("break_mode", "threshold")),
                          sample_seed=int(ck.get("sample_seed", tc.get("sample_seed", 0))))


def load_model(path: str | Path, device="cpu") -> tuple[ImpactOutcomeModel, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = dict(ck["model_config"])
    mc.setdefault("monotone_margin", False)  # checkpoints before the monotone margin head
    # full-state checkpoints from before the flag: a cluster head exists iff its weights do
    mc.setdefault("cluster_head", any(k.startswith("cluster_head.") for k in ck["model"]))
    # break-time / hidden heads: present iff the head's output width includes them
    from . import fullstate as FS
    from . import schema as S_
    bw = [v for k, v in ck["model"].items() if k.startswith("bond_head.") and k.endswith("weight")][-1]
    cw = [v for k, v in ck["model"].items() if k.startswith("chunk_head.") and k.endswith("weight")][-1]
    fs = bool(mc.get("full_state"))
    mc.setdefault("break_time_head", bw.shape[0] == 4 + S_.N_TIME_BINS + (FS.N_BOND_OUT if fs else 0) + 1)
    mc.setdefault("hidden_head", cw.shape[0] == 12 + (FS.N_CHUNK_OUT if fs else 0) + 12)
    m = ImpactOutcomeModel(ModelConfig(**mc))
    m.load_state_dict(ck["model"])
    return m.to(device), ck


def train(cfg: TrainConfig, data: Data | None = None) -> dict:
    out = Path(cfg.out)
    prog = Progress(out, cfg.steps)
    run: dict = {}
    try:
        res = _train(cfg, data, out, prog, run)
        if "tracker" in run:
            run["tracker"].finish(0)
        return res
    except Exception as e:  # loud failure, recorded in status.json
        prog.line(f"FAILED: {e!r}")
        prog.update(force=True, state="failed", last_error=traceback.format_exc())
        if "tracker" in run:
            run["tracker"].summary({"failed": 1})
            run["tracker"].finish(1)
        raise


def _tracker(cfg: TrainConfig, data: Data, out: Path) -> "TR.Tracker":
    mc = cfg.model
    tags = [f"variant-{mc.variant}", f"w{mc.width}", f"rounds-{sum(mc.rounds)}", f"seed{cfg.seed}",
            f"data-{TR.data_revision(data)}", f"git-{TR.git_sha(Path(__file__).parent)}",
            *(["full-state"] if mc.full_state else []), *(["break-time-head"] if mc.break_time_head else []),
            *(["hidden-head"] if mc.hidden_head else []), *(["kinematic"] if cfg.loss.kinematic else []),
            *(["dagger"] if any(c.source == "dagger" for c in data.train) else []),
            *(["fine-tune"] if cfg.init_checkpoint else []), *cfg.wandb_tags]
    config = {**cfg.to_dict(), "n_train_calls": len(data.train), "n_val_calls": len(data.val),
              "n_events": len(getattr(data, "events", []) or [])}
    return TR.Tracker(cfg.wandb, out, config, tags)


def _train(cfg: TrainConfig, data: Data | None, out: Path, prog: Progress, run: dict) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    dev = pick_device(cfg.device)
    data = data or Data(cfg.dataset, cfg.include_unrouted, cfg.val_dataset or None,
                        _read_list(cfg.train_events) if cfg.train_events else None, cfg.call_cache)
    if cfg.model.full_state and not cfg.full_state_columns:
        cols = [e.manifest.get("full_state_columns") for e in data.events if e.manifest.get("full_state_columns")]
        if cols:
            cfg.full_state_columns = cols[0]
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    tracker = run["tracker"] = _tracker(cfg, data, out)
    if tracker.run is not None:
        prog.line(f"wandb: {tracker.mode} run {tracker.run.id} ({TR.PROJECT}) under {out / 'wandb'}")
    cache = HierarchyCache(cfg.model.levels)
    model = ImpactOutcomeModel(cfg.model)
    if cfg.init_checkpoint:
        # fine-tuning (DAgger rounds): the previous weights *and* its input standardisation,
        # so the inputs keep the scale the weights were trained on
        init = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"])
        prog.line(f"initialised from {cfg.init_checkpoint} (step {init.get('step')})")
    elif cfg.model.input_norm:
        model.set_input_stats(input_stats(data.train, cfg.model.variant, cfg.model.full_state))
    model = model.to(dev)
    params = [p for p in model.parameters()]
    balancer = None
    group_bal = None
    share_acc: dict[str, float] = {}
    share_n = 0
    if cfg.loss.balance == "uncertainty":
        balancer = Lo.UncertaintyWeights().to(dev)
    elif cfg.loss.balance == "groups":
        group_bal = Lo.GroupBalancer()
    elif cfg.loss.balance != "fixed":
        raise ValueError(f"loss.balance {cfg.loss.balance!r}")
    opt = make_optimizer(params + (list(balancer.parameters()) if balancer else []), cfg, dev)
    monitor = Lo.DominanceMonitor(cfg.loss.dominance_warn, cfg.loss.dominance_patience)
    sampler = MixedSampler(data.train, cfg.phase_quotas, cfg.seed,
                           getattr(data, "source_weights", None) or None)
    nrng = np.random.default_rng(cfg.seed)
    prog.line(f"device={dev} params={count_parameters(model)} train_calls={len(data.train)} "
              f"val_calls={len(data.val)} variant={cfg.model.variant} width={cfg.model.width} "
              f"rounds={cfg.model.rounds} priors={ {k: round(v, 4) for k, v in data.priors.items()} }")
    def coll(calls):
        b = collate(calls, data.structures, cache)
        return pad_batch(b) if cfg.pad_shapes else b
    if cfg.decode.energy_envelope is not None:
        # The energy envelope is the decode's hard constraint, so it must never exclude what
        # the oracle itself does: every recorded training frame is checked (DAgger relabels
        # are admitted by it already). An excess is an error, never a silent clamp.
        rec = [c for c in data.train if c.source != "dagger"]
        bad = []
        for bc in _batches(rec, cfg.batch_budget, cfg.batch_calls):
            n = D.label_envelope_clamps(collate(bc, data.structures, cache), cfg.decode.energy_envelope)
            bad += [(c.event_id, c.call_index) for c, k in zip(bc, n.tolist()) if k]
        prog.line(f"energy envelope (tol {cfg.decode.energy_envelope}): {len(bad)} of {len(rec)} "
                  f"recorded training calls outside")
        if bad:
            raise S.SchemaError(f"the energy envelope (decode.ENERGY_ENVELOPE_TOL = "
                                f"{cfg.decode.energy_envelope}) excludes {len(bad)} recorded frames, "
                                f"e.g. {bad[:5]}: widen it (it must hold for the oracle)")
    norm_batches = [coll(bc) for bc in
                    _batches(data.train, cfg.batch_budget, cfg.batch_calls)[:64]]
    norms = Lo.zero_baselines(norm_batches, cfg.loss, data.priors)
    prog.line("zero-predictor baselines: " + " ".join(f"{k}={v:.4g}" for k, v in norms.items()))
    val_sel, val_t3, tier = data.val, [], {}
    if cfg.tiers:
        from .coverage import event_tier
        tmap = json.loads(Path(cfg.tiers).read_text())
        tier = {id(c): event_tier(tmap, c.event_id, "val") for c in data.val}
        val_sel = [c for c in data.val if tier[id(c)] in (1, 2)]
        val_t3 = [c for c in data.val if tier[id(c)] == 3]
        prog.line(f"validation tiers: {sum(tier[id(c)] == 1 for c in data.val)} tier-1, "
                  f"{sum(tier[id(c)] == 2 for c in data.val)} tier-2 calls select; {len(val_t3)} tier-3 "
                  f"calls are reported only")
    val_batches = [coll(bc) for bc in _batches(val_sel, cfg.batch_budget, cfg.batch_calls)]
    t3_batches = [coll(bc) for bc in _batches(val_t3, cfg.batch_budget, cfg.batch_calls)]
    # Loss curves (history.jsonl): per-term loss in eval mode on fixed, equally sized samples
    # of validation and of *training* calls, so the two curves compare like with like (the
    # running train loss is a noisy minibatch mean with pushforward and noise).
    rs = random.Random(cfg.seed + 7)
    n_curve = min(len(data.val), cfg.curve_calls) if data.val else 0
    curve = {}
    if n_curve:
        for name, pool in (("val", data.val), ("train", data.train)):
            sample = rs.sample(pool, min(n_curve, len(pool)))
            curve[name] = [coll(bc) for bc in _batches(sample, cfg.batch_budget, cfg.batch_calls)]
    hist = TR.HistoryWriter(out / "history.jsonl", tracker)

    def curve_losses() -> dict:
        model.eval()
        res_ = {}
        with torch.no_grad():
            for name, bl in curve.items():
                sums: dict[str, float] = defaultdict(float)
                tot = 0.0
                for cb in bl:
                    cb = cb.to(dev)
                    tm = Lo.loss_terms(model(cb), cb, cfg.loss, data.priors, cfg.decode)
                    ls_, nm_ = Lo.total(tm, norms, cfg.loss)
                    for k, v in nm_.items():
                        sums[k] += float(v)
                    tot += float(ls_)
                res_[name] = {"loss": tot / len(bl), "terms": {k: v / len(bl) for k, v in sums.items()}}
        model.train()
        return res_
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
        if group_bal is not None:
            gw = group_bal.weights(cfg.loss.weights)
            loss = sum(gw.get(k, 1.0) * v for k, v in normed.items())
        # Pushforward (Brandstetter et al. 2022): each step's input is the model's own decoded
        # outcome of the previous frame (gradient stopped between steps), its targets the
        # oracle's. The first successor batch (decode, push, collate: CPU work) is built on a
        # worker thread while the device runs the main loss's backward; the chain's loss is
        # back-propagated after it. The gradient is the same sum as one backward of both.
        do_pf = cfg.pushforward_prob > 0 and random.random() < cfg.pushforward_prob
        first = None
        if do_pf:
            with torch.no_grad():
                bc0 = b if dev.type == "cpu" else b.to("cpu")
                oc0 = {k: v.detach().cpu() for k, v in o.items()}
            first = _PF_POOL.submit(_successors, calls, bc0, oc0, data, cfg, coll)
        opt.zero_grad(set_to_none=True)
        if cfg.grad_norm_every and step % cfg.grad_norm_every == 0:
            eff = balancer.effective(cfg.loss.weights) if balancer else (
                group_bal.weights(cfg.loss.weights) if group_bal else cfg.loss.weights)
            if group_bal is not None:
                # per group, not per term: 5 extra backward passes instead of ~40
                gsum: dict[str, torch.Tensor] = {}
                for k, v in normed.items():
                    gk = group_bal.group_of.get(k, k)
                    gsum[gk] = gsum.get(gk, 0) + eff.get(k, 1.0) * v
                gnorm_last = grad_norms(gsum, params)
                gshares = group_bal.update(gnorm_last)
                for gk, gv in gshares.items():  # window average of the effective shares
                    share_acc[gk] = share_acc.get(gk, 0.0) + gv
                share_n += 1
            else:
                gnorm_last = grad_norms({k: eff.get(k, 1.0) * v for k, v in normed.items()}, params)
            if group_bal is not None:
                if step % (cfg.grad_norm_every * 10) == 0 or step <= 5 * cfg.grad_norm_every:
                    hist.write_record({"kind": "group_shares", "step": step + 1,
                                       "shares": {g: v / max(share_n, 1) for g, v in share_acc.items()},
                                       "mult": dict(group_bal.mult)})
                    prog.line(f"group gradient shares (effective, mean of {share_n} measurements) "
                              + " ".join(f"{g}={v / max(share_n, 1):.2f}" for g, v in share_acc.items())
                              + " → multipliers " + " ".join(f"{g}={v:.3g}" for g, v in group_bal.mult.items()))
                    share_acc, share_n = {}, 0
            shares, flagged = monitor.update(gnorm_last)
            if flagged:
                prog.line(f"WARNING gradient dominance: {flagged} carry "
                          f"{ {k: round(shares[k], 3) for k in flagged} } of the summed "
                          f"per-term gradient norm (V34 failure mode); check the targets and λ")
                prog.update(force=True, dominance_warning={k: shares[k] for k in flagged})
        loss.backward()
        if first is not None:
            pf_losses = []
            nxt = first.result()
            for _k in range(cfg.pushforward_steps):
                if nxt is None:
                    break
                succ, bs, n_cand, n_skip = nxt
                acc["pushforward_skip_rate"].append(n_skip / max(n_cand, 1))
                if bs is None:
                    break
                bs = bs.to(dev)
                os_ = model(bs)
                ts = Lo.loss_terms(os_, bs, cfg.loss, data.priors, cfg.decode)
                ls, ns = Lo.total(ts, norms, cfg.loss)
                if balancer is not None:
                    ls = balancer(ns, cfg.loss.weights)
                if group_bal is not None:
                    gw = group_bal.weights(cfg.loss.weights)
                    ls = sum(gw.get(k, 1.0) * v for k, v in ns.items())
                pf_losses.append(ls)
                if _k + 1 < cfg.pushforward_steps:
                    with torch.no_grad():
                        nxt = _successors(succ, bs if dev.type == "cpu" else bs.to("cpu"),
                                          {k: v.detach().cpu() for k, v in os_.items()},
                                          data, cfg, coll)
            if pf_losses:
                ls = sum(pf_losses) / len(pf_losses)
                ls.backward()
                loss = loss.detach() + ls.detach()
                acc["pushforward"].append(ls.detach())
                acc["pushforward_depth"].append(float(len(pf_losses)))
        total_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        # Logged values stay on the device until a log line: a float() per term per step
        # would sync MPS every step and stop the CPU preparing the next batch meanwhile.
        for k, v in normed.items():
            acc[k].append(v.detach())
        acc["loss"].append(loss.detach())
        acc["grad_norm"].append(total_norm.detach())
        now = time.time()
        if (now - last_log >= cfg.log_every_s or step % cfg.log_every_steps == 0
                or step == cfg.steps - 1):
            acc = {k: [float(x) for x in (torch.stack([torch.as_tensor(y, dtype=torch.float32)
                                                       .to(dev) for y in v]).cpu().tolist())]
                   for k, v in acc.items()}
            if not np.isfinite(acc["loss"]).all():
                raise FloatingPointError(f"non-finite loss by step {step + 1}: "
                                         f"{ {k: v[-1] for k, v in acc.items()} }")
            rate = (step + 1) / max(now - prog.t0, 1e-9)
            eta = (cfg.steps - step - 1) / max(rate, 1e-9)
            terms_s = " ".join(f"{k}={np.mean(v):.3f}" for k, v in acc.items()
                               if k not in ("loss", "grad_norm"))
            tot_g = sum(gnorm_last.values()) or 1.0
            gn = " ".join(f"{k}={v / tot_g:.2f}" for k, v in gnorm_last.items())
            if balancer is not None:
                gn += " λeff[" + " ".join(f"{k}={v:.2f}" for k, v in
                                         balancer.effective(cfg.loss.weights).items()) + "]"
            hist.write_record({"kind": "train_running", "step": step + 1, "epoch": epoch,
                               "loss": float(np.mean(acc["loss"])),
                               "terms": {k: float(np.mean(v)) for k, v in acc.items()
                                         if k not in ("loss", "grad_norm")},
                               "lr": _lr(step, cfg), "grad_norm": float(np.mean(acc["grad_norm"])),
                               "gshare": {k: float(v / tot_g) for k, v in gnorm_last.items()},
                               **({"mult": {g: float(v) for g, v in group_bal.mult.items()}}
                                  if group_bal is not None else {}),
                               "steps_per_s": rate})
            prog.line(f"step {step + 1}/{cfg.steps} ep {epoch} loss={np.mean(acc['loss']):.4f} "
                      f"[{terms_s}] |g|={np.mean(acc['grad_norm']):.3g} gshare[{gn}] "
                      f"lr={_lr(step, cfg):.2e} {rate:.2f} steps/s ETA {eta / 60:.1f} min")
            acc = defaultdict(list)
            last_log = now
        prog.update(step=step + 1, epoch=epoch)
        if val_batches and ((step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1):
            res = M.evaluate(model, val_batches, cfg.decode,
                             data.priors if cfg.loss.break_reweight == "prior" else None)
            cl = curve_losses()
            hist.write_record({"kind": "eval", "step": step + 1, "epoch": epoch, **cl,
                               "score": res.score, "metrics": res.metrics})
            if t3_batches:  # extrapolation: reported, never selected on
                r3 = M.evaluate(model, t3_batches, cfg.decode,
                                data.priors if cfg.loss.break_reweight == "prior" else None)
                hist.write_record({"kind": "eval_tier3", "step": step + 1, "score": r3.score,
                                   "metrics": r3.metrics})
                prog.line(f"val tier 3 @ {step + 1} (not selected on): score {r3.score:.3f}")
            if cl:
                prog.line(f"loss @ {step + 1}: train-sample {cl['train']['loss']:.4f} "
                          f"val-sample {cl['val']['loss']:.4f}")
            msg = " ".join(f"{k}={v:.3f}" for k, v in sorted(res.metrics.items()))
            improved = res.score < best["score"]
            if improved:
                best = {"score": res.score, "step": step + 1, "metrics": res.metrics}
                save(out / "best.pt", model, cfg, norms, data.priors, step + 1, res.metrics)
            prog.line(f"val @ {step + 1}: {msg} {'(best)' if improved else ''} "
                      f"best={best['score']:.3f}@{best['step']}")
            prog.update(force=True, best_score=best["score"], best_step=best["step"])
    batches.close()
    hist.close()
    # validation-only temperature for p_b, stored with the last checkpoint
    shift_priors = data.priors if cfg.loss.break_reweight == "prior" else None
    temperature = 1.0
    if val_batches:
        logits, tg = [], []
        with torch.no_grad():
            for vb in val_batches:
                o = model(vb.to(dev))
                sh = Lo.prior_logit_shift(vb, shift_priors)
                lg = o["bond_break_logit"].cpu() + (sh.cpu() if sh is not None else 0)
                logits.append(lg.numpy())
                tg.append(np.where(vb.t["y_bond_has_freq"].numpy(), vb.t["y_bond_freq"].numpy(),
                                   (vb.t["y_bond_broke"].numpy() > 0).astype(float)))
        temperature = M.fit_temperature(np.concatenate(logits), np.concatenate(tg))
        prog.line(f"temperature (validation) = {temperature:.3f}")
    save(out / "last.pt", model, cfg, norms, data.priors, cfg.steps, best["metrics"], temperature)
    tau = cfg.decode.tau
    if val_batches and best["step"] is not None and cfg.select_tau:
        ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        tau, scores = select_tau(model, val_batches, cfg, shift_priors, prog)
        prog.line(f"tau (validation) = {tau} (scores {scores})")
        save(out / "best.pt", model, cfg, norms, data.priors, ck["step"], ck["metrics"],
             temperature, tau)
    if cfg.tiers and val_batches and best["step"] is not None:
        # the per-tier report of the best checkpoint at its τ, from the calls in memory (the
        # in-memory tier-1 hold-outs exist only here); never one blended average
        ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        from .coverage import TIER_NAMES
        dc = decode_config(ck)
        rep = {"checkpoint": str(out / "best.pt"), "tau": dc.tau, "tiers": {}}
        for t in (1, 2, 3):
            ct = [c for c in data.val if tier.get(id(c)) == t]
            if not ct:
                continue
            bt = [coll(bc) for bc in _batches(ct, cfg.batch_budget, cfg.batch_calls)]
            rt = M.three_way(model, bt, dc, shift_priors)
            rep["tiers"][str(t)] = {"name": TIER_NAMES[t], "n_calls": len(ct),
                                    "n_events": len({c.event_id for c in ct}),
                                    "selection_score": rt["selection_score"], "families": rt["families"],
                                    "acceptance": rt["acceptance"]}
            prog.line(f"tier {t} ({TIER_NAMES[t]}): {len(ct)} calls, score {rt['selection_score']:.3f}, "
                      f"accepted families {[f for f, a in rt['acceptance'].items() if a['accept']]}")
        (out / "report-val-tiers.json").write_text(json.dumps(rep, indent=1, default=float))
        tracker.summary({"tiers": {k: {"selection_score": v["selection_score"]} for k, v in rep["tiers"].items()}})
        tracker.artifact(out / "report-val-tiers.json", f"{out.parent.name}-{out.name}-report-val-tiers")
    prog.update(force=True, state="done", step=cfg.steps)
    tracker.summary({"best_score": best["score"], "best_step": best["step"] or 0,
                     "temperature": temperature, "tau": tau, "best": best["metrics"] or {}})
    return {"best": best, "norms": norms, "temperature": temperature, "tau": tau}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="",
                    help="training calls: path[@weight],… (several sources mixed by weight)")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="generate N synthetic events into <out>/data first (no accuracy claims)")
    ap.add_argument("--val-dataset", default="",
                    help="validate on this dataset's val split (the bulk evaluation ensembles)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-calls", type=int, default=64, help="max calls per batch")
    ap.add_argument("--batch-budget", type=int, default=20000,
                    help="nodes + undirected edges per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--rounds", default="3,2,2,2,3")
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--reference", action="store_true", help="W64, rounds 3,2,2,2,3 (the default)")
    ap.add_argument("--large", action="store_true", help="W128, rounds 4,3,2,3,4")
    ap.add_argument("--cost", action="store_true", help="W32, rounds 2,1,2,1,2 (P0's cost point)")
    ap.add_argument("--variant", default="B", choices=["A", "B"])
    ap.add_argument("--full-state", action="store_true",
                    help="/5: the oracle's full dynamic bond and chunk state as inputs and heads")
    ap.add_argument("--train-events", default="", help="file of P1 recording dirs: training calls via CallCache")
    ap.add_argument("--extra-train", default="", help="DAgger relabel dirs, dir[@share],… (training only)")
    ap.add_argument("--call-cache-module", default="", help="P1's outcome_calls_cache.py")
    ap.add_argument("--call-cache", default="", help="cache root")
    ap.add_argument("--call-builder", default="", help="P1's learned_impact_outcome_calls")
    ap.add_argument("--cache-gb", type=float, default=1.5,
                    help="total disk bound of the call cache (counts toward P1's 30 GB budget)")
    ap.add_argument("--break-time-head", action="store_true",
                    help="a continuous break time per bond (the decode's break order)")
    ap.add_argument("--hidden-head", action="store_true",
                    help="the chunks' frame-end hidden state (u, theta, v, w) as a head")
    ap.add_argument("--tiers", default="", help="coverage tiers JSON (outcome.coverage): select on "
                    "validation tiers 1+2, report tier 3 separately")
    ap.add_argument("--exclude-families", default="", help="comma-separated families left out of "
                    "every split (to keep a run series' data fixed as a dataset grows)")
    ap.add_argument("--holdout-tier1", type=float, default=0.0,
                    help="hold out about this share of the configurations of every training layout "
                         "with ≥ 3 as tier-1 validation, in memory (e.g. 0.1)")
    ap.add_argument("--holdout-ballistic", default="",
                    help="speed positions (0-based, e.g. 1,3) of each training ballistic family's "
                         "ensembles moved to validation: the in-distribution ballistic test")
    ap.add_argument("--kinematic", action="store_true",
                    help="the soft kinematic-consistency loss term (displacements outside the "
                         "velocity trapezoid's envelope, decode.trapezoid_excess)")
    ap.add_argument("--init-checkpoint", default="",
                    help="start from these weights and input statistics (fine-tuning, DAgger)")
    ap.add_argument("--drop-features", default="",
                    help="comma-separated <array>:<column> inputs held at 0 (ablation), e.g. "
                         "bond_feat:log_qs_hertz_util")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--wandb", default="auto", choices=("auto", "off", "offline", "online"),
                    help="Weights & Biases tracking (auto: offline when wandb is installed; runs land in "
                         "<run dir>/wandb, `wandb sync` uploads them after `wandb login`)")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--pushforward", type=float, default=0.0, help="probability per step")
    ap.add_argument("--pushforward-steps", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--energy-tolerance", type=float, default=0.05)
    ap.add_argument("--break-mode", default="threshold", choices=["threshold", "sample"])
    ap.add_argument("--balance", default="groups", choices=["fixed", "uncertainty", "groups"],
                    help="groups: keep the term groups' gradient shares near Lo.GROUP_TARGET")
    ap.add_argument("--no-input-norm", action="store_true",
                    help="disable the fixed per-column input standardisation")
    ap.add_argument("--w-break", type=float, default=None, help="λ for break and margin")
    ap.add_argument("--reweight", default=None, choices=["prior", "none", "focal"],
                    help="break-loss class reweighting")
    ap.add_argument("--break-margin", type=float, default=0.2,
                    help="broken bonds must reach log q̂ ≥ this (0 = the plain censored hinge)")
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
    mc = (ModelConfig.large() if a.large else ModelConfig.cost() if a.cost
          else ModelConfig.reference() if a.reference else ModelConfig(
              width=a.width, rounds=tuple(int(x) for x in a.rounds.split(",")), levels=a.levels))
    mc.variant = a.variant
    mc.drop_features = tuple(x for x in a.drop_features.split(",") if x)
    mc.full_state = a.full_state
    mc.break_time_head = a.break_time_head
    mc.hidden_head = a.hidden_head
    mc.__post_init__()
    seeds = [int(s) for s in a.seeds.split(",")]
    cc = {"module": a.call_cache_module, "root": a.call_cache, "builder": a.call_builder,
          "max_gb": a.cache_gb} if a.train_events else {}
    data = Data(dataset, val_dataset=a.val_dataset or None,
                train_events=_read_list(a.train_events) if a.train_events else None, call_cache=cc,
                extra_train=a.extra_train,
                holdout_ballistic=tuple(int(x) for x in a.holdout_ballistic.split(",") if x),
                holdout_tier1=a.holdout_tier1,
                exclude_families=tuple(x for x in a.exclude_families.split(",") if x))
    for s in seeds:
        out = Path(a.out) / (f"seed{s}" if len(seeds) > 1 else "")
        cfg = TrainConfig(dataset=dataset, val_dataset=a.val_dataset, out=str(out), steps=a.steps,
                          train_events=a.train_events, call_cache=cc,
                          init_checkpoint=a.init_checkpoint,
                          batch_calls=a.batch_calls, batch_budget=a.batch_budget, lr=a.lr, seed=s, device=a.device,
                          noise_sigma=a.noise, pushforward_prob=a.pushforward,
                          pushforward_steps=a.pushforward_steps,
                          eval_every=a.eval_every, model=dataclasses.replace(mc), wandb=a.wandb,
                          tiers=a.tiers,
                          decode=D.DecodeConfig(energy_tolerance=a.energy_tolerance,
                                                break_mode=a.break_mode))
        cfg.loss.energy_tolerance = a.energy_tolerance
        cfg.loss.kinematic = a.kinematic
        cfg.loss.balance = a.balance
        cfg.loss.break_margin = a.break_margin
        cfg.model.input_norm = not a.no_input_norm
        if a.w_break is not None:
            cfg.loss.weights["break"] = a.w_break
            cfg.loss.weights["margin"] = a.w_break
        if a.reweight is not None:
            cfg.loss.break_reweight = a.reweight
        r = train(cfg, data)
        print(json.dumps({"seed": s, "best_score": r["best"]["score"],
                          "best_step": r["best"]["step"]}))


if __name__ == "__main__":
    main()
