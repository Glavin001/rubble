"""One trainer's step, split into its parts (planning, 2026-10-22: the Modal trial kept the GPU
25–38% busy): collate (the reference per-batch collate or the cached pieces), padding,
host→device, forward, loss, backward, optimiser, and with pushforward the successor rebuild
(device→host copy, decode, push, collate). Then the end-to-end rate with collation on the
background thread, as in training. The device is synchronised around each part, so the parts
add up; the overlapped rate is the real one.

    python ../scripts/profile_step.py <calls dir> --device mps [--events 40] [--steps 40]
        [--pushforward 1.0] [--width 64] --out profile.json
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learned_impact.outcome import losses as Lo, schema as S  # noqa: E402
from learned_impact.outcome.batch import HierarchyCache, clear_piece_cache, collate, pad_batch  # noqa: E402
from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig  # noqa: E402
from learned_impact.outcome.train import (Prefetch, TrainConfig, _batches, _successors,  # noqa: E402
                                          make_optimizer, pick_device)
from tests.collate_reference import collate_reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--device", default="auto")
ap.add_argument("--events", type=int, default=40)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--warmup", type=int, default=8)
ap.add_argument("--pushforward", type=float, default=0.0)
ap.add_argument("--width", type=int, default=64)
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = pick_device(a.device)
sync = (torch.mps.synchronize if dev.type == "mps" else torch.cuda.synchronize if dev.type == "cuda"
        else (lambda: None))

ps = [p for p in S.find_events(a.dataset) if json.loads((Path(p) / "manifest.json").read_text())["split"] == "val"]
random.seed(0)
ps = random.sample(ps, min(a.events, len(ps)))
evs = [S.load_event(p, in_memory=True) for p in ps]
st = {e.manifest["event_id"]: e.structures for e in evs}
calls = [c for e in evs for c in e.calls if c.routed]
succ = {}
for e in evs:
    rc = sorted((c for c in e.calls if c.routed), key=lambda c: c.call_index)
    for x, y in zip(rc, rc[1:]):
        if y.call_index == x.call_index + 1:
            succ[(x.event_id, x.call_index)] = y
data = SimpleNamespace(successor=succ, priors=None, structures=st)
mc = ModelConfig(width=a.width, full_state=True, break_time_head=True, hidden_head=True)
if a.width == 128:
    mc = ModelConfig.large()
    mc.full_state, mc.break_time_head, mc.hidden_head = True, True, True
cfg = TrainConfig(model=mc, pushforward_prob=a.pushforward)
cfg.loss.kinematic = True
cache = HierarchyCache(mc.levels)


def batches(n):
    out = []
    while len(out) < n:
        random.shuffle(calls)
        out += _batches(calls, cfg.batch_budget, cfg.batch_calls)
    return out[:n]


bl = batches(a.warmup + a.steps)
torch.manual_seed(0)
model = ImpactOutcomeModel(mc).to(dev)
opt = make_optimizer(list(model.parameters()), cfg, dev)
norms = Lo.zero_baselines([pad_batch(collate(bc, st, cache)) for bc in bl[:4]], cfg.loss)


def step(bc, coll, t):
    t0 = time.perf_counter()
    b = coll(bc)
    t1 = time.perf_counter()
    b = pad_batch(b)
    t2 = time.perf_counter()
    bd = b.to(dev)
    sync()
    t3 = time.perf_counter()
    o = model(bd)
    sync()
    t4 = time.perf_counter()
    loss, _ = Lo.total(Lo.loss_terms(o, bd, cfg.loss, None, cfg.decode), norms, cfg.loss)
    sync()
    t5 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    sync()
    t6 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sync()
    t7 = time.perf_counter()
    for k, v in (("collate", t1 - t0), ("pad", t2 - t1), ("to_device", t3 - t2), ("forward", t4 - t3),
                 ("loss", t5 - t4), ("backward", t6 - t5), ("optimiser", t7 - t6)):
        t[k].append(v)
    if a.pushforward > 0 and random.random() < a.pushforward:
        u0 = time.perf_counter()
        with torch.no_grad():
            bc0 = b
            oc0 = {k: v.detach().cpu() for k, v in o.items()}
        u1 = time.perf_counter()
        r = _successors(bc, bc0, oc0, data, cfg, lambda cs: pad_batch(coll(cs)))
        u2 = time.perf_counter()
        t["pf_to_host"].append(u1 - u0)
        t["pf_rebuild"].append(u2 - u1)  # decode + push + collate of the successors
        if r is not None and r[1] is not None:
            bs = r[1].to(dev)
            os_ = model(bs)
            ls, _ = Lo.total(Lo.loss_terms(os_, bs, cfg.loss, None, cfg.decode), norms, cfg.loss)
            ls.backward()
            sync()
            t["pf_device"].append(time.perf_counter() - u2)


report = {"device": str(dev), "width": a.width, "events": len(evs), "calls": len(calls),
          "steps": a.steps, "pushforward": a.pushforward, "modes": {}}
for name, coll in (("reference", lambda bc: collate_reference(bc, st, cache)),
                   ("cached", lambda bc: collate(bc, st, cache))):
    clear_piece_cache()
    if name == "cached":  # pieces built once (an epoch); timed steps reuse them
        for bc in bl:
            collate(bc, st, cache)
    t = defaultdict(list)
    for i, bc in enumerate(bl):
        tt = defaultdict(list) if i < a.warmup else t
        step(bc, coll, tt)
    ms = {k: 1000 * float(np.mean(v)) for k, v in t.items()}
    ms["sum_ms"] = sum(ms.values())
    # overlapped: collation on the background thread, as train.py does
    pf = Prefetch(lambda: ((bc, pad_batch(coll(bc))) for bc in bl), depth=2)
    wait = []
    sync()
    t0 = time.perf_counter()
    for i in range(len(bl)):
        w0 = time.perf_counter()
        bc, b = pf.next()
        wait.append(time.perf_counter() - w0)
        bd = b.to(dev)
        o = model(bd)
        loss, _ = Lo.total(Lo.loss_terms(o, bd, cfg.loss, None, cfg.decode), norms, cfg.loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync()
    dt = time.perf_counter() - t0
    pf.close()
    ms["overlapped_ms_per_step"] = 1000 * dt / len(bl)
    ms["overlapped_wait_ms"] = 1000 * float(np.mean(wait[a.warmup:]))
    report["modes"][name] = ms
    print(name, json.dumps({k: round(v, 1) for k, v in ms.items()}), flush=True)
if a.out:
    Path(a.out).write_text(json.dumps(report, indent=1))
