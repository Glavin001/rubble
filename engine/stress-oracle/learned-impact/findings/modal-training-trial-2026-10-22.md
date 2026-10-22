# Modal training trial: the unchanged trainer on cloud CUDA (2026-10-22)

**Question.** Can P2's PyTorch trainer (`python -m learned_impact.outcome.train`) run on
Modal GPUs, and what do speed and cost look like? Is the viability training affordable
there within about $8?

**Viability training:** W64/12 and W128/16 × 3 seeds, plus 25/50/100% data-scaling runs,
each about 10k steps.

**Answer.**

- **It works.** The CUDA step-1 loss matches the CPU loss to 1e-7.
- **Total spend: $0.72 billed** ($0.733 on the ledger) of the $2.00 cap.
- **The step is host-bound, not GPU-bound.**
  - A single trainer keeps an L4 only 25–38% busy.
  - W128 runs at almost the speed of W64.
  - An L40S is faster mainly because its host CPU is faster.
- **The cheap way to train is to pack 3–4 trainers into one container on one GPU.** This is lossless: each run is unchanged, it just shares the GPU.
  - It cuts the cost per run-step by about 2.5×, to **$0.045–0.049 per 1k steps** for W64 with pushforward on an L4.
  - That is about **$0.50 per 10k-step W64 run** on sub100-sized data.
- **The viability set's training steps alone fit in about $5.** Two full-scale costs were not measured here and could push the total past $8:
  - evaluation on the full 49k-call validation set;
  - the memory needed for the full dataset.
- **Recommendation:** first run one ~$0.35 full-data calibration container (see the end), then launch.

## Setup

**Branch.** `learned-impact-modal`, cut from `claude/learned-impact-p2-model-7cf306`, which is unchanged.

**Code.**
- `modal/pack_data.py`: one deterministic `.tar.gz` and its sha256.
- `modal/modal_job.py`: the container body.
  - Verifies the hash and unpacks.
  - Checks the step-1 parity, and skips training on a mismatch.
  - Runs the unchanged trainer CLI under a hard timeout.
  - Writes timestamped output and samples GPU utilisation.
  - Can run N concurrent trainers.
- `modal/train_modal.py`: the Modal app.
  - Pinned image.
  - `learned-impact` Volume, with upload-once checked by hash.
  - Entrypoints `probe`, `upload`, `trial`, `launch` and `pull`.
  - A budget guard and the ledger `findings/modal-spend.md`, with `CAP_USD = 2.0`.
- `python/tests/test_modal_job.py` runs the exact container body on CPU: pack, unpack, parity, train and evaluate.
  - It also checks that a wrong reference stops the run before training.

**Image.**
- `debian:bookworm-slim` + Python 3.12.
- `torch==2.14.1` (the PyPI wheel is the CUDA build, `2.14.1+cu130`, cuDNN 9.24), `numpy==2.5.4`, `zstandard==0.25.0`.
- `gcc` + `libc6-dev`. Torch 2.14 sends an autograd bmm outer product to a Triton kernel, and Triton compiles its driver shim with the system C compiler. Without it the first `grad_norms` call fails with "Failed to find C compiler".
- The package is added as the last layer.
- The torch layer built once in about 60 s and is cached since then. Later code changes rebuild only the thin layer, in 2–3 s.

**Modal host.** Driver 610.57.04.
- L4: 23 GB.
- L40S: 46 GB.

**Data (sub100).** A stratified real subset, copied to a scratch directory. The sources were only read.
- 100 train events from P1's `calls-v3-train`: 71 free-wall t1, 20 anchored-wall t1 and 9 free-wall t2. That is 2,297 routed calls.
- 24 val events from `calls-v3-eval`: three complete 8-member ensembles, one per family. That is 956 calls.
- No split group is shared between the two sets.
- Archive: `sub100.tar.gz`, 136 MB, sha256 `41ec5592…0c62cd`.
- Uploaded once, in 19 s. Later `upload` calls check the hash and skip the upload.
- The calls are `/3`, so they are read with `LEARNED_IMPACT_ACCEPT_V3=1` (`--accept-v3`).
- Batches are packed to the 20,000 node + edge budget, which gives 38.3 calls per step on average.

**Model and trainer.**
- Trainer defaults: W64 with rounds 3,2,2,2,3 (12 rounds) and 2 levels, which is the `--reference` config. W128/16 is `--large`.
- `--pushforward 0.3`, the bulk/viability recipe. Runs marked "no-pf" use the default 0.
- 300 steps, with `--eval-every 100`.
- This gave 50–115 s of steady training per job.

## Commands (from the repo root)

```bash
LI=engine/stress-oracle/learned-impact; M=$LI/modal/train_modal.py
python3 $LI/modal/subset_data.py <P1>/engine/stress-oracle/learned-impact/runs <scratch>/sub100 100 3
#   (re-running it and pack_data.py reproduces sha256 41ec5592…0c62cd exactly)
python3 $LI/modal/pack_data.py --train <scratch>/sub100/train --val <scratch>/sub100/val --name sub100
# local, free: the CPU reference on the archive's own contents (devcheck.step1_losses, 64 calls,
# seed 0, default W64/12) → $LI/runs/modal/sub100-step1-cpu.json; plus a CPU smoke run of
# modal_job.trial on a 5+8-event "tiny" archive (8 steps, pushforward 0.3: rc 0, parity OK)
engine/tools/modal/modal.sh run $M::probe
engine/tools/modal/modal.sh run $M::upload --name sub100
engine/tools/modal/modal.sh run $M::trial --name sub100 --gpu L4   --run l4-w64-pf   --steps 300 --eval-every 100 --max-seconds 300 --accept-v3 --extra "--pushforward 0.3"
engine/tools/modal/modal.sh run $M::trial --name sub100 --gpu L40S --run l40s-w64-pf --steps 300 --eval-every 100 --max-seconds 300 --accept-v3 --extra "--pushforward 0.3"
engine/tools/modal/modal.sh run $M::trial --name sub100 --gpu L4   --run l4-w64-nopf --steps 300 --eval-every 100 --max-seconds 300 --accept-v3
engine/tools/modal/modal.sh run $M::trial --name sub100 --gpu L40S --run l40s-w64-nopf --steps 300 --eval-every 100 --max-seconds 300 --accept-v3
engine/tools/modal/modal.sh run $M::trial --name sub100 --gpu L4   --run l4-w128-pf  --steps 300 --eval-every 100 --max-seconds 300 --accept-v3 --extra "--large --pushforward 0.3"
engine/tools/modal/modal.sh run $M::trial --name sub100 --gpu L4   --run l4-w64-pf-x3 --steps 300 --eval-every 100 --max-seconds 400 --accept-v3 --extra "--pushforward 0.3" --parallel 3 --cores 6 --mem-gib 16
#   … the same with --gpu L40S (l40s-w64-pf-x3), --extra "--large --pushforward 0.3" on both GPUs
#   (l4-w128-pf-x3, l40s-w128-pf-x3), and --parallel 4 --cores 8 --mem-gib 24 on the L4 (l4-w64-pf-x4)
```

Every run's outputs were pulled back to `$LI/runs/modal/<run>/` (gitignored):
- `best.pt`, `last.pt`;
- `history.jsonl`, `progress.log`, `status.json`, `config.json`;
- `stdout.log`, with each line prefixed by seconds since launch;
- `gpu_util.csv`;
- `trial.json` / `trial-local.json`.

Containers with `--parallel` put each trainer's outputs in `<run>/<i>/`.

## Parity (CPU vs CUDA, fixed weights and data)

The check is the step-1 loss of the seed-0 W64/12 model on the first 64 routed calls of sub100's training split.

- **CPU (local):** total 125.615463.
- **CUDA, all 10 GPU containers:** 125.615448 or 125.615463, i.e. within 1.2e-7 relative.
- **Per term:** all 11 terms are within 4e-7 relative. The tolerance is 2e-3.

## Measured speed

*Steady* is the rate over log intervals that contain no evaluation, after the first 25
steps. Calls/s = steps/s × 38.3 base calls per step; pushforward's successor calls are not counted.

**$/s** is the container's reserved rate: GPU + cores × $0.0000131 + GiB × $0.00000222.

| run | GPU | trainers × config | cores / GiB | steps/s steady (each) | aggregate steps/s | calls/s | GPU busy | $/s |
|---|---|---|---|---|---|---|---|---|
| l4-w64-nopf | L4 | 1 × W64 | 4 / 8 | 6.04 | 6.04 | 231 | 33% | 0.000292 |
| l40s-w64-nopf | L40S | 1 × W64 | 4 / 8 | 8.91 | 8.91 | 341 | 34% | 0.000612 |
| l4-w64-pf | L4 | 1 × W64 pf | 4 / 8 | 2.58 | 2.58 | 99 | n/a | 0.000292 |
| l40s-w64-pf | L40S | 1 × W64 pf | 4 / 8 | 4.13 | 4.13 | 158 | 25% | 0.000612 |
| l4-w128-pf | L4 | 1 × W128 pf | 4 / 8 | 2.35 | 2.35 | 90 | 38% | 0.000292 |
| l4-w64-pf-x3 | L4 | 3 × W64 pf | 6 / 16 | 2.28, 2.21, 2.34 | 6.83 | 262 | 77% | 0.000336 |
| l4-w64-pf-x4 | L4 | 4 × W64 pf | 8 / 24 | 2.07, 1.98, 2.08, 2.23 | 8.36 | 320 | 92% | 0.000380 |
| l40s-w64-pf-x3 | L40S | 3 × W64 pf | 6 / 16 | 3.90, 3.70, 3.98 | 11.58 | 443 | 84% | 0.000656 |
| l4-w128-pf-x3 | L4 | 3 × W128 pf | 6 / 16 | 1.72, 1.65, 1.74 | 5.11 | 196 | 91% | 0.000336 |
| l40s-w128-pf-x3 | L40S | 3 × W128 pf | 6 / 16 | 2.75, 2.64, 2.76 | 8.15 | 312 | 91% | 0.000656 |

"GPU busy" is nvidia-smi's utilisation: the share of time any kernel was running. It is not SM occupancy.

**Comparison with the Mac.** Local M3 Max MPS does about 2.2 steps/s on bulk W32 with pushforward, and about 4.3 steps/s on a 1-event W64 overfit. So one cloud trainer is roughly as fast as the Mac on an L4, and only about 1.9× faster on an L40S. The cloud's gain is parallel runs, not per-run speed.

**Fixed time per job (sub100).**

| phase | L4 | L40S |
|---|---|---|
| Container cold start (call → function entry) | 2.6–10.5 s (median ≈ 4 s) | same |
| Hash check + unpack, 136 MB | 1.3 s | 0.8 s |
| Parity | 2 s | 1.4 s |
| Trainer imports + data load (2,297 + 956 calls) | 13–15 s | 9 s |
| Zero baselines + val/curve batch collation, then step 1 | 12 s | 7 s |
| First 25 steps (Triton JIT of the bmm kernel + new padded shapes), pf | 8 s | 5 s |
| One evaluation every 100 steps, extra time | 1–5 s | 1–3 s |
| Final evaluation + temperature + 10-value τ selection | 25–35 s | 16–23 s |
| Copy outputs to the Volume | 1.2–1.5 s (7.7 s on the first write) | same |

Image build: about 60 s once, then 2–3 s for the code layer. Data upload: 19 s once.

## Cost per 1k steps and per 10k-step W64 run

These figures are from the measured steady rates and the $/s above. A 10k-step run adds the fixed time (about 90 s) and 100 evaluation passes over sub100's validation set. The costs scale linearly.

| GPU, packing | W64 pf: $ / 1k run-steps | W64 pf: $ per 10k run | wall per 10k run | W128 pf: $ / 1k run-steps | W64 no-pf: $ / 1k |
|---|---|---|---|---|---|
| L4, 1 trainer | $0.113 | $1.16 | 1.08 h | $0.124 | $0.048 |
| L40S, 1 trainer | $0.148 | $1.54 | 0.67 h | n/a | $0.069 |
| **L4, 3 trainers** | **$0.049** | **$0.50** | 1.22 h | **$0.066** | n/a |
| L4, 4 trainers | $0.045 | $0.46 | 1.33 h | n/a | n/a |
| L40S, 3 trainers | $0.057 | $0.59 | 0.72 h | $0.080 | n/a |

## Bottlenecks

1. **Per-step host time, not GPU arithmetic.**
   - One trainer keeps the GPU 25–38% busy.
   - W128/16 (about 4× the FLOPs of W64 per round) runs at 2.35 steps/s against 2.58 for W64.
   - Speed follows the host CPU. The L40S host runs the CPU-only start-up 1.5× faster (9.0 vs 13.4 s), and its steps/s are 1.47× (no-pf) to 1.60× (pf) the L4's.
   - The step is many small kernels: 12 message-passing rounds of small MLPs over about 20k padded nodes and edges, plus per-term losses. So it is limited by Python and kernel-launch latency.
2. **Pushforward's CPU work.**
   - At p = 0.3, pushforward raises the mean step from 166 ms to 387 ms on the L4. That is about 0.74 s per pushforward step.
   - Each one copies the outputs to the host, decodes on the CPU (`decode`, `_predictions`, `P.push`) and collates the successors. All of that runs on the main thread, then two more forward passes run.
   - This is the largest single cost in the recipe.
3. **Not data loading.**
   - The `Prefetch` thread collates ahead.
   - One trainer used about 1.4 cores on average (152 CPU-s over 107 s), so 4 reserved cores were never the limit.
   - With packing, 3 trainers used about 4.9 of the 6 cores.
4. **Packing fills the idle GPU.**
   - With 3 trainers each one slows only 5–12%, and GPU busy goes to 77–91%.
   - With 4 trainers on the L4 each slows 20%, which is close to the GPU's limit.
   - W128 × 3 already reaches 91% busy.

## Viability training on Modal within about $8: recommendation

**Plan.** 8 runs, packed:
- W64/12 seeds 0–2. Seed 0 is also the 100% scaling point: same config, same data, so no extra run is needed.
- W128/16 seeds 0–2.
- 25% and 50% scaling runs (W64).

Pack them as two 4-trainer containers on the L4, or as three 3-trainer containers. All runs are 10k steps with `--pushforward 0.3` and `--eval-every 1000`.

**What sub100 does not show.**
- **Full validation set.** The full `calls-v3-eval` val split is 1,059 events and 49.3k calls, 52× sub100's.
  - The trainer evaluates all of it at each evaluation, and 11 more times at the end (temperature + 10 τ values).
  - At the measured 1.4–2.0 ms per call on the L4, that adds roughly 1,900–2,600 s per run, which runs concurrently inside a packed container.
  - For comparison, on the Mac it took about 18 of the bulk W32 retrain's 94 min.
- **Full-data memory.** It is unknown.
  - sub100 needed 1.3 GB RSS just to load.
  - Full train + val is about 18× larger on disk, and the 49k validation calls are kept pre-collated.
  - Plan on 16–24 GiB per trainer until it is measured. `trial` now records `child_maxrss_gb`.
- **Load and collation.** Expect about 5 min before step 1.

**Estimate on the L4.**

| | 3-trainer containers | GPU time | at 16 GiB per trainer | at 24 GiB per trainer |
|---|---|---|---|---|
| Training steps alone | W64 ≈ 4,400 s, W128 ≈ 5,900 s | about 4.2 GPU-hours | ≈ $4.9 | n/a |
| Adding full-val evaluation, load and memory | n/a | n/a | ≈ $8–9 | ≈ $9–11 |

The L40S finishes about 1.7× sooner but costs about 15–20% more.

**Recommendation.**

1. **Pack and the L4.** Use the L4 with 3–4 trainers per container. Do not run one trainer per GPU: that costs 2.3× more for the same results.
2. **Calibrate first, before committing $8:**
   - One L4 container, 3 W64 trainers on the full archive.
   - 300 steps, `--eval-every 150`, with `child_maxrss_gb` recorded.
   - This measures load time, RSS and the full-val evaluation time directly.
   - About 15 min, about $0.35, after a one-time upload of about 2.5 GB.
   - Then size `--mem-gib` and `max_seconds` from it and `launch` the plan.
3. **If the calibrated total exceeds $8, cut cost losslessly first:**
   - Profile and speed up the CPU-side decode in evaluation and pushforward. The results stay identical; the work is on the host thread.
   - Use 4-way packing.

   Changing the recipe would also cut it: fewer evaluations, a validation subset for checkpoint selection, a smaller pushforward probability, or fewer seeds. Those change results and are your decision. For reference, no-pf is 2.3× cheaper per step.
4. **Raise the cap.** `CAP_USD` in `train_modal.py` is 2.0 for this trial. Raise it deliberately, for example to 10.0, before the viability launch. `launch` refuses any plan whose worst case would pass the cap.
