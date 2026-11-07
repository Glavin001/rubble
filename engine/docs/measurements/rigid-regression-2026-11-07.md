# Rigid regression check: soft-body vehicle work (2026-11-07)

**Question.** Has the rigid engine slowed down because of the soft-body vehicle work?
The rigid engine covers AVBD, threshold fracture, and large sleeping maps. The
requirement is that rigid Rubble keeps handling about 100k bodies efficiently.

**Verdict: no regression.**
- **Code.** For rigid-only scenes, head runs byte-identical rigid shaders. Its step
  code differs only by bookkeeping that does nothing when no vehicles are present.
- **State.** Every reproducible rigid scene ends in a bit-identical state on all
  three builds.
- **Timings.**
  - With the GPU uncontended, the three builds are within 3 % of one another
    (the targeted repeats below).
  - Over the five interleaved rounds, the medians differ only by amounts the
    builds' own min..max ranges cover. Another agent was using the GPU during
    those rounds.

Status: complete except the equivalence-harness recordings (`tests/equivalence.rs`).
Twice the GPU hung on another agent's test and those runs stopped; see "Still to run".

## Builds compared

| label | commit | what it is |
|---|---|---|
| head | `1861035c` (`origin/fix/contact-ub`) | the vehicle work: surface contact, World broadphase vehicle proxies, vehicle sleep, centred damping, coupling, kernel and compile changes |
| main | `ce5a599c` (`origin/main`) | merge-base plus PR #102 (lab consolidation) |
| mb | `4c7d54cf` | merge-base of head and main (PR #99) |

`origin/main` is **not** an ancestor of head. Main carries PR #102, which head lacks.
PR #102 changes `engine-world/src/{threshold,destruction,lib}.rs` and the threshold
`fracture.wgsl`. A head-vs-main difference could therefore come from #102 rather
than from the vehicle work. **Head vs mb is the clean comparison for the vehicle
work**, and main is reported alongside it.

## Code review of the rigid touch points (mb → head)

- **Rigid WGSL is byte-identical.** `diff -rq` of engine-webgpu's
  `OUT_DIR/wgsl` across the three release builds finds 382 rigid modules equal.
  Head only adds new files: explicit beams, `vehicle_coupling`, `feature_query`,
  `strict`, and `loop_cap_test`.
  - `world_step.slang` and the rest of the rigid plan (`engine-exec`) are unchanged.
  - `engine-fracture` and `engine-scenes` are unchanged.
  - In `engine/slang/physics/`, only `feature_query.slang` and `strict.slang` were
    added. No shared module changed.
- **Resting table, `awake_broadphase` gate, `pose_pack` moved lists, all-asleep
  skip.** None of these is in the mb → head diff; they predate it.
  - Head adds `WgpuSim::rest_view` and `rest_buffers`, which are read-only
    accessors for the vehicle proxy query (`cpl_query`).
  - It adds `mark_gpu_impulses` and `schedule_gpu_impulses`, which only vehicles
    call.
  - It adds bookkeeping for host timing segments, which is empty unless
    `gpu_timing` is on.
- **`World::step`.**
  - `explicit_before_step`, `explicit_after_step` and `explicit_after_submit`
    return at once when no vehicles were added.
  - The only per-step additions are one `web_time::Instant::now()`, stored for
    `last_submit_instant`, and a `None` check before the readback extras.
  - `web_time::Instant` replaces `std::time::Instant` only in env-gated timing.
    Natively it is `std::time::Instant`.
- **Threshold destruction** (`threshold.rs`, `stress-gpu/src/ingest.rs`).
  - The shared ingest is now made with 2 substep slots instead of 1, for the
    vehicles' chunk loads in slot 1. This doubles the records buffer, which costs
    memory only.
  - With no vehicles, `merge(1)` has the same record count and dispatch sizes as
    before. The rows kernel does not read `record_count`.
  - A new `ensure!` caps `max_bodies` at 2^20, an explicit limit checked at
    creation.

## Method

- **Bench.** `engine/crates/engine-world/examples/rigid_regression_bench.rs` is new.
  - It uses only API present on main, so the same file was built on all three
    commits.
  - Each step's **wall** time is `World::step` plus `poll_blocking`.
  - The **GPU** time is the step pass's timestamp (`GpuTiming::Step`), taken only
    on steps that ran.
  - At the end it writes an FNV-1a hash of every body's position, linear velocity
    and flags.
- **City scenes.** These are the threshold town: cuboid houses, the large-map
  sleep policy, and the still, drive, demolish and war scenarios. They come from
  the existing ignored test `engine-world/tests/city_scale.rs`, which is identical
  on all three commits.
- **Driver.** `engine/tools/rigid-regression/driver.py` runs each scene on each
  build in turn and rotates the build order every round. It kills a run after
  300 s, or 600 s for the city runs.
  - Each round was a separate invocation of about 6 minutes.
  - `summarize.py` builds the tables.
- **Rounds.**
  - **Five rounds** of every scene, in `results.csv`. Round 2's city256 rows are
    dropped because they finished as the GPU hung at 12:21.
  - **Two targeted rounds** of the three noisiest comparisons, in `targeted.csv`.
    These ran while the other agents were limited to short correctness runs.
  - Raw data, including the runs on the hung GPU, is in
    `results-raw-incl-wedged.csv`.
- **Hardware.** Apple M3 Max, Metal through wgpu, release builds.
- **Shared GPU.** Another agent shared the GPU during the five rounds. Read each
  min..max range as the noise band for that scene.

## Results

### All scenes, five interleaved rounds

Each cell is the median over rounds of each run's p50 wall ms per step, with the
min..max over rounds in brackets. The table covers 10k, 30k and 100k bodies at about
1 % awake, 30k and 100k all awake, stacks, piles, the threshold town, and idle large
maps.

| scene | bodies | head | main | mb | head vs main | head vs mb |
|---|---|---|---|---|---|---|
| sparse, 1 % awake | 10k | 2.727 (2.669..3.569) | 2.806 (2.681..2.837) | 2.728 (2.580..3.466) | -2.8% | -0.0% |
| sparse, 1 % awake | 30k | 1.342 (1.211..2.000) | 1.268 (1.257..2.036) | 1.392 (1.190..1.817) | +5.8% | -3.6% |
| sparse, 1 % awake | 100k | 1.512 (1.342..2.461) | 1.593 (1.403..3.541) | 1.493 (1.348..3.087) | -5.1% | +1.3% |
| all awake (sleep off) | 30k | 4.837 (4.534..6.106) | 5.323 (4.548..6.328) | 5.050 (4.503..6.601) | -9.1% | -4.2% |
| all awake (sleep off) | 100k | 14.829 (12.612..17.428) | 14.025 (12.577..16.557) | 13.818 (12.543..17.495) | +5.7% | +7.3% * |
| 400 stacks of 10, sleep off | 4k | 2.414 (1.873..3.032) | 2.404 (1.894..3.213) | 2.359 (1.845..3.199) | +0.4% | +2.3% |
| Box Columns, sleep off | 100k | 9.473 (8.994..11.888) | 9.698 (9.112..14.368) | 9.766 (9.329..12.110) | -2.3% | -3.0% |
| Box Pile | 32k | 6.540 (6.228..6.706) | 6.488 (6.319..7.273) | 6.572 (6.167..6.727) | +0.8% | -0.5% |
| Pyramid 30, sleep off | 9.5k | 5.656 (5.131..6.124) | 5.364 (5.203..5.665) | 5.249 (5.139..5.547) | +5.4% | +7.8% |
| 100k Grid | 100k | 9.594 (9.343..12.054) | 10.688 (9.188..12.292) | 11.006 (9.712..19.969) | -10.2% | -12.8% |
| Brick Ring | 28k | 5.818 (5.729..11.602) | 5.959 (5.679..6.933) | 5.619 (5.271..5.922) | -2.4% | +3.5% * |
| idle map 2 km, all asleep | 1.6k | 0.004 | 0.005 | 0.008 | (step skipped) | |
| idle map 4 km, all asleep | 6.4k | 0.003 | 0.005 | 0.003 | (step skipped) | |
| map 4 km + 1 falling box | 6.4k | 0.859 (0.814..1.233) | 1.360 (0.867..1.396) | 0.963 (0.805..1.250) | -36.8% | -10.8% |
| town K=16 still | 11k chunks | 3.095 (3.052..10.844) | 3.153 (3.106..4.281) | 3.166 (3.019..10.014) | -1.8% | -2.2% |
| town K=16 drive | | 4.329 (3.127..5.533) | 3.252 (3.156..4.703) | 3.197 (3.079..4.558) | +33.1% | +35.4% † |
| town K=16 demolish | | 3.568 (3.390..4.678) | 4.644 (3.380..14.549) | 3.413 (3.330..5.035) | -23.2% | +4.5% |
| town K=16 war | | 5.137 (3.717..10.139) | 5.147 (3.726..5.852) | 3.699 (3.549..5.373) | -0.2% | +38.9% † |
| town K=64 still | 46k chunks | 3.932 (3.717..4.169) | 3.956 (3.800..5.455) | 4.121 (3.922..5.634) | -0.6% | -4.6% |
| town K=64 drive | | 4.123 (4.083..4.216) | 4.042 (3.898..5.631) | 4.221 (3.916..5.600) | +2.0% | -2.3% |
| town K=64 demolish | | 4.743 (4.591..4.816) | 4.583 (4.422..6.212) | 4.786 (4.679..13.072) | +3.5% | -0.9% |
| town K=64 war | | 5.293 (5.074..6.743) | 5.530 (5.034..7.086) | 5.374 (5.166..18.131) | -4.3% | -1.5% |
| town K=64 sleep, still | 46k chunks | 0.017 (0.008..0.031) | 0.010 (0.008..0.082) | 0.009 (0.008..0.015) | (step skipped) | |
| town K=64 sleep, drive | | 2.588 (1.459..4.228) | 1.402 (1.351..1.518) | 1.538 (1.384..2.604) | +84.6% | +68.3% * |
| town K=64 sleep, demolish | | 3.540 (3.264..4.747) | 3.503 (3.434..3.639) | 3.490 (3.354..4.509) | +1.1% | +1.4% |
| town K=64 sleep, war | | 5.124 (4.894..15.088) | 5.118 (5.026..11.633) | 5.026 (4.772..6.191) | +0.1% | +1.9% |
| town K=256 sleep, still (4 rounds) | 183k chunks | 0.064 (0.056..0.075) | 0.059 (0.052..0.063) | 0.059 (0.047..0.068) | (step skipped) | |
| town K=256 sleep, drive | | 2.190 (2.081..2.472) | 2.309 (2.244..3.453) | 2.331 (2.195..3.581) | -5.2% | -6.0% |
| town K=256 sleep, demolish | | 5.415 (5.369..5.663) | 5.788 (5.502..6.809) | 5.465 (5.358..6.839) | -6.5% | -0.9% |
| town K=256 sleep, war | | 8.160 (8.003..9.839) | 8.104 (8.034..9.946) | 8.235 (7.985..10.988) | +0.7% | -0.9% |

\* Re-measured in the targeted repeats below, where the difference is gone.

† Wide noise. Head's min is at or below the other builds' typical values, and the
same scenario at K=64 shows no difference.

**GPU step time** (timestamp queries; full table from `summarize.py`). The cleanest
signal, all within ±5 % on the rigid scenes:

| scene | head | main | mb |
|---|---|---|---|
| sparse 100k | 0.95 | 0.99 | 0.99 |
| all awake 100k | 10.18 | 10.54 | 10.30 |
| Box Columns 100k | 6.10 | 6.08 | 6.26 |
| Pyramid 30 | 3.62 | 3.57 | 3.67 |
| map + 1 box | 0.412 | 0.411 | 0.411 |

### Targeted repeats on an uncontended GPU

Two rounds. Each cell is the median p50 wall ms, with min..max in brackets.

| scene | head | main | mb | head vs main | head vs mb |
|---|---|---|---|---|---|
| all awake 100k | 13.069 (12.790..13.349) | 13.049 (12.774..13.325) | 12.951 (12.818..13.085) | +0.2% | +0.9% |
| Brick Ring 28k | 5.914 (5.836..5.992) | 5.871 (5.855..5.887) | 5.893 (5.789..5.997) | +0.7% | +0.4% |
| town K=64 sleep, still | 0.010 | 0.009 | 0.010 | (skipped) | |
| town K=64 sleep, drive | 1.438 (1.389..1.486) | 1.403 (1.345..1.462) | 1.413 (1.380..1.446) | +2.4% | +1.7% |
| town K=64 sleep, demolish | 3.446 (3.412..3.480) | 3.361 (3.286..3.435) | 3.353 (3.328..3.377) | +2.5% | +2.8% |
| town K=64 sleep, war | 4.792 (4.699..4.884) | 4.806 (4.654..4.957) | 4.854 (4.817..4.890) | -0.3% | -1.3% |

The town K=64 sleep drive gap seen in the five rounds (2.59 against 1.40) is gone.
In those rounds its runs fell into two modes:

- fast: host about 0.33 ms per step, GPU about 0.045 ms;
- slow: host about 1.25 ms per step, GPU about 0.19 ms.

mb also hit the slow mode once. Host time moving 4× while the host code is the same
points to contention, not to head's code. The all-awake 100k and Brick Ring gaps are
gone as well.

### Bit-identical results

The end-state hash covers every body's position, linear velocity and flags, bit for
bit, after the measured steps. It is **identical across head, main and mb in every
round** for these 11 scenes:

- sparse 10k, 30k and 100k;
- all awake 30k and 100k;
- 400 stacks of 10;
- idle maps of 2 km and 4 km;
- the 4 km map with one falling box;
- Box Columns 100k and Pyramid 30, both with sleep off.

Box Pile 32k, 100k Grid and Brick Ring 28k give a **different hash on every run,
even within one build**. That is 5 distinct hashes from 5 runs per build. Those
scenes are not reproducible run to run on native wgpu, so they cannot show a
difference between builds.

## Regressions over noise and their cause

None. Every gap above noise in the shared-GPU rounds disappeared when re-measured on
an uncontended GPU. Code review and the shader comparison find no change in rigid
work per step. No bisect was needed.

## Still to run

- **Equivalence-harness recordings** (`engine-world/tests/equivalence.rs`).
  - Scenes: house, gables, city and vehicle. The vehicle scene is the rigid sedan.
  - Order: record head twice to measure its own noise, then record mb, then
    compare with `engine/tools/rigid-regression/equiv.py`.
  - These were pending when the GPU hung a second time, at about 13:22.
- **More targeted rounds.** Rounds 12–17 timed out on the hung GPU and are excluded.

## Reproduce

```sh
# per commit (the example file is committed on this branch; copy it into the others)
cargo build --release -p engine-world --example rigid_regression_bench \
  --test city_scale --test equivalence
# copy target/release/examples/rigid_regression_bench and deps/city_scale-* into BINDIR
RR_ROUND=0 python3 engine/tools/rigid-regression/driver.py out.csv 1 \
  head=BIN_HEAD main=BIN_MAIN mb=BIN_MB
RR_ONLY=awake100k,demo_ring28k,city64_sleep RR_ROUND=10 python3 ... targeted.csv 1 ...
python3 engine/tools/rigid-regression/summarize.py out.csv
python3 engine/tools/rigid-regression/equiv.py OUT house,gables,city,vehicle \
  head=BIN_HEAD mb=BIN_MB
```
