---
name: profiling-gpu-time
description: Find where Rubble's GPU and host time actually goes, per kernel, on native Metal (Apple Silicon) and in the browser (WebGPU/Dawn), and turn it into evidence for a lossless optimization. Use before any performance work, when choosing what to optimize, when a step or frame is slower than expected, when reading per-phase or per-kernel timings, or when GPU marks and wall time disagree.
---

# Profiling GPU time

The goal is to know, at the lowest level, where every microsecond of a step goes, so
optimization works from evidence instead of guesses. The tools measure. They never change
the physics, and they are off by default.

Full how-to and baseline numbers:
- `engine/docs/profiling-metal.md`: native Metal hosts (`MetalSim` rigid, `MetalExplicitBeams` vehicles).
- `engine/docs/profiling-webgpu.md`: wgpu, browser and native, the World, the pages.

Read `gpu-safe-runs` first. Every GPU run on the Mac goes through the machine's one lock:
`engine/scripts/gpu-lock.sh` or `tools/webgpu-profile/gpu-run.sh`, both of which wrap
`scripts/with-gpu-lock.py`. Other sessions queue on the same lock and a wait can take
45 minutes or more, so run profiles in the background and wait on the PID. Outputs go to
the repo's gitignored `out/`, never `/tmp`.

## Pick the tool by the question

| Question | Tool | Command (from `engine/`) |
|---|---|---|
| What does one step cost, and how much of it is host time? | Lockstep step timing | `scripts/metal-profile.sh --scene "rigid:Brick Ring (9k)" --no-replay` |
| Which kernel costs what? | **Kernel replay**: each dispatch alone on the real step's input | Metal: `scripts/metal-profile.sh --scene …`. wgpu: `RUBBLE_KERNEL_REPLAY_AT=N` on `vehicle_lab` / `brick_ring_drive_probe`, or the page's *Kernel replay* button |
| Why is this kernel slow? (limiter, occupancy, per-line cost) | Xcode capture and shader profiler | `RUBBLE_CAPTURE_STEP=N RUBBLE_METAL_DEBUG_LIB=1 scripts/metal-profile.sh …`, then `open out/…gputrace` |
| Where is the GPU idle, and where does the host wait? | Instruments Metal System Trace, summarized to text | `scripts/metal-trace.sh --name X -- <binary …>` writes `out/metal-trace/X/summary.txt` |
| What does a browser frame cost: JS, WASM host, Dawn, GPU, draw? | Browser harness (headless system Chrome) | `tools/webgpu-profile/run.sh --page … --query … --name X --replay 15` |
| Did my change help, and where? | Diff two runs | `node tools/webgpu-profile/diff.mjs A/summary.json B/summary.json`; Metal: compare the `.json` files |
| Is the change allowed in? | Scenario matrix and bit-for-bit ladder | `tools/scenario-matrix/run.sh SWITCH` (`AGENTS.md`) |

## Workflow

1. **Pick a representative state with work in it.** Profile at 10k+ awake rigid bodies
   (ring110k with `sleep=0` or `BR_CAR=none`) and at the vehicle counts that matter (8, 32,
   64 cars). A step where every body is asleep is skipped, and the replay says "nothing to
   replay": move N later or turn sleep off.
2. **Baseline the whole step** (lockstep or `timing=step`). This number is the truth; every
   breakdown below must add up to it.
3. **Run the kernel replay** and read the per-kernel table. Check that the isolated sum is
   close to the whole step (within a few %). If it is far off, the breakdown is wrong;
   find out why before reading it.
4. **For the top kernels, take an Xcode capture** with the debug libraries. Read the
   limiter, occupancy, ALU and memory utilisation, and the per-line cost on the `.slang`
   source.
5. **Look for gaps**: GPU idle between command buffers, host waits (Instruments summary,
   Chrome trace), host spans (`RUBBLE_HOST_PROFILE=1`, or the page's profiler).
6. **Form one hypothesis** that names the kernel or gap, the cause and the evidence. Make
   one lossless change, re-run the same profile on the same binary setup, then diff.
7. **Gate it**: scenario matrix (no row slower, bit for bit with bit-exact on), plus
   `vehicle_parity` after Metal kernel edits. A lossy idea is measured and reported only;
   adopting it is the user's decision (`AGENTS.md`).

## Reading the numbers

**Never use pass-per-phase or pass-per-dispatch timings for absolute cost.** These include
`RUBBLE_GPU_TIMING`, `timing=phases`, `step_profiled` and GPU marks. Each pass boundary
drains the GPU, and the idle time lands on the next phase. Measured on ring110k (native
wgpu-on-Metal, all bricks awake):

| How the step is timed | GPU ms |
|---|---|
| One pass (what ships) | 13.3 |
| A pass per phase | 18.7 span |
| A pass per dispatch | 20.0 span |
| Kernel replay, isolated sum | 13.8 |

Phase timings rank phases at best. Cost that grows with world size has been blamed on the
wrong phase this way before.

**Kernel replay.**
- `iso ms` is the median of M runs of that dispatch alone, from the exact input the real
  step gives it (indirect counts included).
- `in-seq` is the same dispatch timed inside the sequence.
- The empty-pass or launch floor (about 2–3 µs) is included in every isolated time.
- `isolated sum − n × floor` estimates the step without launch costs. What the whole step
  takes beyond that is barriers and drains between dependent dispatches.
- The share of each kernel depends on the state: profile the moment you care about, and
  compare runs at the same step.

**Many repeated small dispatches** (solver colours × iterations: 4 iterations of 5–6 colour
passes plus a dual pass) are usually barrier- or occupancy-bound, not ALU-bound. Compare
each one's workgroups with the 40 GPU cores on an M3 Max, and read occupancy in the capture
before touching the math.

**Bandwidth.** Metal's `changed` / `>=GB/s` columns are lower bounds: they count only
written lines and see no reads. Compare them against about 400 GB/s (M3 Max). For a
memory-bound kernel the shader profiler's limiter and memory utilisation are the real answer.

**Host and browser.**
- Browser host spans need `crossOriginIsolated` (COOP/COEP). Otherwise Chrome rounds
  `performance.now()` to 0.1 ms. The harness asserts this, and also asserts the page is
  visible with rAF ≥ 30 Hz. A hidden pane runs rAF at about 1 Hz.
- Spans that suspended through JSPI are filed under "(suspended)", never in the frame's JS.
- In the Chrome trace, the GPU process pays wire decode (about 1.9 ms per step on the ring)
  and `Queue::Submit`, and presentation waits behind queued compute.
- A shader compile in the middle of a run (Tint → MSL, 18 ms) is a stutter: warm the
  pipelines before the scene starts.
- First measured browser frame (ring, 60 Hz): 3.1 ms of JS per rAF frame, 2.4 ms of it in
  `world.poll`. Check host spans before assuming a page is GPU-bound.

**Overheads of the tools themselves.**
- Labels (`RUBBLE_METAL_LABELS=1`, `RUBBLE_GPU_LABELS=1`) roughly double the host's
  encode time and add no GPU time. Use them for captures and traces, not for host timing.
- Debug libraries produce bit-identical results with the same GPU time (checked on 8 cars).
- A CPU read of a shared buffer just before a dispatch adds 0.02–0.03 ms of GPU time. Keep
  host work equal on both sides of an A/B.

**Reproducibility.**
- Native rigid steps (`MetalSim`, native wgpu) are not reproducible run to run: lockstep
  worlds drift apart.
- Compare kernels on one saved state (the replay does this) or with the matrix's fixtures,
  never two free-running worlds.
- The noise threshold is 5 % **and** 0.05 ms (`diff.mjs`, the matrix).
- One-car rows can change level between processes.

## Report

Every number needs:
- the scene, body and car counts, awake count, and the step N measured;
- the backend and device (M3 Max, native Metal, wgpu-on-Metal or Chrome version);
- the commit, release or fast profile, and whether bit-exact was on;
- for browser runs, visibility, rAF rate and `crossOriginIsolated`;
- median and p90 over M reps, and the whole step next to the breakdown;
- the path of the run's `out/` JSON, so the next agent can diff against it.

Lead with the decision or deviation, then the top kernels or gaps and their evidence
(limiter, occupancy, bandwidth, launch or barrier share).
