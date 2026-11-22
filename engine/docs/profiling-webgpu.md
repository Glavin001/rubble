# Profiling WebGPU: where a browser frame's milliseconds go

One command answers "where does every millisecond of a browser frame go": the host WASM
(Rust), the page's JS, submit, the GPU per step and per phase, each kernel alone, readback
and drawing. The same Rust runs natively on wgpu-on-Metal, so a browser number has a native
twin. This page is the how-to, the baseline numbers from an M3 Max, and the ranked list of
lossless optimisation opportunities they point to.

Sibling: `docs/profiling-metal.md` (the `metal-profiling` session: engine-metal, Xcode and
Instruments). This page covers `engine-webgpu` (wgpu, browser and native), the WASM
bindings and the pages.

## Baseline (M3 Max, 2026-11-22)

`main` at d5a1d480 plus this branch. Release builds. Chrome is the system Chrome,
headless, visible, rAF 60 Hz, `crossOriginIsolated`. Runs are in `out/webgpu-profile/`
(`tools/webgpu-profile/baseline.sh`).

| Run | GPU per step | Host per frame | Notes |
|---|---|---|---|
| Ring110k, browser (`b-ring`, sedan, `sleep=0`) | 14.8 ms (rigid 11.6, car frame 2.5, coupling plan 0.56) | JS 3.0 ms, of which `world.poll` 2.37 | GPU-bound at 60 Hz |
| Ring110k, native wgpu (`n-ring110k`, no car) | 13.28 ms (replay whole step 13.19) | — | 110,440 awake |
| 8-car lab, browser (`b-lab8`, mountain, ledges) | 16.3 ms (car frame 15.5, rigid 0.31) | JS 0.57 ms | GPU-bound: the vehicle frame |
| 8-car lab, native wgpu (`n-lab8`) | 15.8 ms | — | rigid replay whole step 0.55 ms |

**Ring110k by kernel** (native replay at step 60, in-sequence medians, 13.19 ms whole step):

| Kernel | Dispatches | ms | Share |
|---|---|---|---|
| `solve_primal_color_compact` | 40 | 6.20 | 47 % |
| `solve_dual_compact` | 4 | 1.89 | 14 % |
| `narrowphase_fresh_copy_persistent` | 1 | 1.45 | 11 % |
| `persist_list_copy_persistent` | 1 | 0.94 | 7 % |
| `broadphase_query` | 1 | 0.81 | 6 % |
| `solve_primal_sparse_compact` | 4 (4 workgroups each) | 0.59 | 4.5 % |
| `narrowphase_defer_copy_persistent` | 1 | 0.55 | 4 % |
| everything else (40 dispatches) | | ~1.3 | ~10 % |

## Opportunities (ranked by evidence, none adopted)

These are leads with their evidence. Each still needs a shader-profiler capture to name its
limiter, and the scenario matrix to admit any change.

1. **The primal colour solve is half the rigid step**: 6.2 ms of 13.2 on ring110k, in 40
   dispatches (about 10 colours × 4 iterations). It is the same picture on native Metal at
   9k–32k bodies (42–57 %, `profiling-metal.md`). Each dispatch writes at 4–10 GB/s, far
   below the memory system, so it is likely latency-, barrier- or occupancy-bound rather
   than bandwidth-bound. Next step: a capture of the colour passes for limiter and occupancy.
2. **`world.poll` takes 2.4 ms per browser frame on the ring** (80 % of the frame's JS),
   against 0.03 ms on the 8-car lab. `world.poll.device` is 0, so it is host work that grows
   with body count, not a GPU wait. Next step: split the span to find which part of poll
   scales with the 110k bodies.
3. **`solve_primal_sparse_compact` runs 4 workgroups on a 40-core GPU**: 0.15 ms per
   iteration, 0.6 ms per step. This is a near-serial tail of the colouring.
4. **Empty and tiny dispatches in small rigid steps.** On the 8-car lab's rigid step (0.55
   ms), the isolated work sums to 0.16 ms while in sequence it takes 0.51 ms. 42 dispatches
   had zero workgroups and still cost a launch and a barrier each.
5. **Chrome's GPU process** (Chrome trace, 2026-11-19): wire decode about 1.9 ms per ring
   step, `Queue::Submit` 30 ms/s, presentation blocked 255 ms/s behind queued compute, and
   one Tint → MSL compile (18 ms) in the middle of the run (a stutter: warm the pipelines).
6. **The vehicle frame dominates vehicle scenes** (15.5 of 16.3 ms for 8 cars on the
   mountain). The explicit plan is timed per segment by `timing=phases`. The rigid replay
   does not cover it.


## The tools

| Tool | What it answers | Where |
|---|---|---|
| Plan labels | Captures (Chrome GPU traces, Xcode, Instruments) name every pass `step:<phase>` and every dispatch `phase/kernel` | `WgpuSim::set_debug_labels`, `RUBBLE_GPU_LABELS=1`, the panel's *labels* box, harness `--labels` |
| Host spans | The host's own time per frame: `world.step` and its parts, `sim.encode_dispatches`, `world.poll` | `engine_webgpu::host_profile`; `RUBBLE_HOST_PROFILE=1` natively, `setHostProfile` in the browser |
| GPU per step / phase | The step's GPU time from timestamps (World `gpuTiming` `step` or `phases`) | existing; read by the panel and the harness |
| Kernel replay | Every dispatch of one step timed **alone**, from its own real input, plus the whole step in three shapes from the same state | `WgpuSim::kernel_replay`; the panel's *Kernel replay* button; harness `--replay N`; native `RUBBLE_KERNEL_REPLAY_AT=N` |
| Browser harness | Headless Chrome on this machine's GPU: visibility and rAF asserted, spans, GPU times, CDP trace, replay → `summary.json`, `table.txt`, `trace.json` | `tools/webgpu-profile/run.sh` |
| Profiler panel | The same breakdown live on the page (a UI toggle, never a URL flag) | `profiler.mjs`, the *Profiler* button on `vehicle-lab.html` (next to *Copy perf report*) and `brick-ring-drive.html` (in the controls) |
| Diff | Before/after of two runs, with the scenario matrix's noise threshold; a verdict | `tools/webgpu-profile/diff.mjs` |
| Baseline | The scenario set below, browser and native, under one GPU-lock acquisition | `tools/webgpu-profile/baseline.sh` |

### Build

```bash
cd engine
cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
wasm-pack build crates/engine-world-wasm --release --target web --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
cargo build --release -p engine-world --example vehicle_lab --example brick_ring_drive_probe
cargo test --release -p engine-world --test kernel_replay --no-run
```

### One browser run

```bash
engine/tools/webgpu-profile/run.sh --page vehicle-lab.html --query 'mode=mountain&car=sedan&res=low&cars=8&terrain=ledges&timing=step' --name lab8 --replay 15
```

It takes the machine's GPU lock (`gpu-run.sh`: the flock `~/.rubble/gpu.lock` through
`scripts/with-gpu-lock.py`, which every session's GPU jobs queue on, and which kills the
whole process group after the limit), starts the system Chrome headless with unquantised
timestamps (`--enable-webgpu-developer-features`, Dawn's `timestamp_quantization` off,
`allow_unsafe_apis`; none of these changes the page's work), opens the page, and:

1. **asserts** the page is `visible` with rAF at 30 Hz or more (a hidden or throttled page
   runs rAF at about 1 Hz and every number would be wrong);
2. lets it run `--warm` seconds, then records `--secs` seconds of frames: the rAF
   callbacks' JS, the Rust host spans (also as `performance.measure`, so the trace shows
   them under User Timing), the page's named spans (`page.draw`) and the steps' GPU times;
3. records the last `--trace` seconds as a Chrome trace (CDP `Tracing`: `gpu`, `dawn`,
   `v8`, `blink.user_timing`, `toplevel`, `devtools.timeline` and the disabled-by-default
   Dawn and timeline categories) and tabulates each process's busiest threads;
4. with `--replay N`, holds the page's frame loop and runs the kernel replay;
5. writes `out/webgpu-profile/<name>/summary.json` (every number), `table.txt` and
   `trace.json` (open it in `ui.perfetto.dev` or `chrome://tracing`).

Options: `--page`, `--query`, `--name`, `--secs` (10), `--warm` (5), `--trace` (3; 0 off),
`--replay` (0 off), `--labels`, `--size WxH`, `--out`, `--startLimit`.

The harness is the repo's raw-CDP pattern (as `vehicle-lab-record.mjs`) on the system
Chrome, not Playwright's bundled Chromium: the bundled build's headless WebGPU is not this
machine's Metal GPU path, and the numbers must be the ones a user's Chrome gives.

### JSPI and double counting

The World's host work runs inside wasm exports. Some waits (a blocking readback, the kernel
replay) suspend the wasm stack to the event loop through JSPI (`engine-gpu-wait`). A span
that suspended includes event-loop time that is not its own work: the span records the
number of suspensions (`engine_gpu_wait::YIELDS` across it) and the profiler files it under
`<name> (suspended)` and `suspended in GPU waits`, never inside the frame's JS. The old
`stress_phases()` double counted for exactly this reason. In the real-time loops measured
here no step or poll suspends (they never wait), so the suspended rows are empty.

### Kernel replay

`WgpuSim::kernel_replay(KernelReplayOptions { reps, .. })`, between steps, outside the
simulation loop:

1. copies every buffer of the step aside (the state to restore at the end);
2. makes the next step's host writes and plan exactly as `step_encode` would, and copies the
   state again (the step's input);
3. from that input, `reps` times each: the whole plan in **one pass** (the shipped shape), a
   **pass per phase** (`GpuTiming::Phases`' shape), a **pass per dispatch** in plan order;
4. **isolated**: for each dispatch `k`, the buffers it binds are restored to the state just
   before it and it runs alone in a timed pass, `reps` times; its last run's output becomes
   dispatch `k + 1`'s input, so each kernel sees exactly the input the real step gives it,
   indirect workgroup counts included (read back as they were before it ran);
5. an **empty pass** (a one-workgroup no-op), the floor every single-pass time includes;
6. restores every buffer. No host bookkeeping moved; the next real step runs as if the
   replay never happened. `tests/kernel_replay.rs` checks this: every buffer's digest
   before and after is the same.

Every timed pass runs alone in its own submission after the restore has finished (see
*Timestamps on Apple GPUs* below). The report: per dispatch its phase, kernel, workgroups,
isolated median/p90/min, in-sequence time, share of the whole step; and the step's shapes.
It is the rigid step's plan (`rigid_plan`); the explicit vehicles' kernels are already
timed one dispatch per segment by `GpuTiming::Phases` (`explicit_vehicles.rs`).

In the browser the same Rust runs: `WebWorld.kernelReplayRun` is called through
`WebAssembly.promising` (it waits on the GPU with `engine_gpu_wait::until`), with the
page's rAF loop held by `profiler.mjs` so nothing else borrows the World.

### Timestamps on Apple GPUs

Pass timestamps on Metal (wgpu natively, Dawn in Chrome) have three traps. The replay
works around each:

- **A pass's begin tick is taken when Metal starts the pass,** which can be before the work
  encoded ahead of it in the same command buffer has finished. A restore blit followed by a
  timed pass in one submission would charge part of the blit to the kernel. So every restore
  is submitted and waited on alone, and the timed pass follows in a submission of its own.
- **A resolve can read samples before they land,** in the same command buffer and also in a
  later submission that is queued straight behind a long pass. The timed submission is waited
  on before the resolve is encoded. Without that wait, a 13 ms whole-step pass read 0 on most
  reps and `tests/kernel_replay.rs` failed. Earlier reports of the per-phase span (18.7 ms
  against 13.3 ms for the ring) were this race, not real idle time.
- **Some passes write no timestamp at all.** A reused query slot would then keep an older
  pass's tick and stretch every span computed over it. This happened on the ring: a 16.7 ms
  pass-per-phase span over 11.3 ms of passes with no gaps. Slots are never reused (a fresh
  query set when one is full), zero ticks are skipped, and the report counts them as
  `timed passes that wrote no timestamp`. A non-zero count means some rows have fewer
  samples than `reps`.

With timestamps read correctly, splitting a step costs little. On ring110k natively:
13.19 ms in one pass, a 13.86 ms span with a pass per phase (0.01 ms idle between phase
passes), and 15.74 ms with a pass per dispatch. In Chrome: 11.63 / 11.67 / 11.82 ms. Take the
total from the single pass and attribution from the replay's rows.

### Native twin

```bash
cd engine
RUBBLE_HOST_PROFILE=1 RUBBLE_KERNEL_REPLAY_AT=360 RUBBLE_KERNEL_REPLAY_OUT=../out/n-lab8.json \
  tools/webgpu-profile/gpu-run.sh -- target/release/examples/vehicle_lab --mode gpu --measure mountain-perf --cars 8 --terrain ledges --timing step --secs 10
```

`RUBBLE_KERNEL_REPLAY_AT=N` runs the replay once at step `N` (`engine_webgpu::replay::env_hook`,
called by `vehicle_lab --measure mountain-perf` and `brick_ring_drive_probe` between steps),
prints its table and writes the JSON with the host spans' medians.

### Diff

```bash
node engine/tools/webgpu-profile/diff.mjs out/webgpu-profile/A/summary.json out/webgpu-profile/B/summary.json
```

Every median of both runs (frame JS, each host span, GPU step and segments, each replayed
kernel by `phase/kernel#n`), the delta, and `faster` / `slower` / `same` per row with the
scenario matrix's threshold (5 % **and** 0.05 ms). The verdict is `slower` if any row is
slower. `--json` for agents. A lossless change still has to pass the scenario matrix and
the bit-for-bit ladder (`AGENTS.md`); the diff is the evidence of *where* it saved time.
