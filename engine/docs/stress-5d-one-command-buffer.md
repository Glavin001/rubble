# Plan 5d: the stress stage in rubble's frame, one command buffer, one read

Status: **in progress** (WIP commit 08efa0c, opt-in since: `STRESS_FRAME_SUBMITS=one`; the default submits each part on its own, as before). Parent charter:
`engine/docs/stress-baseline-goal.md` (Goal: "resident: one submission and one status read
per tick"; M5 step 1). Original plan entry: `engine/docs/stress-qs-baseline-plan.md` 5d.

This document exists because that requirement was reported as met when it was not. Read the
**Rules** section before reporting progress.

## 1. The requirement

On every tick of a resident frame, the stress stage and rubble's step run as:

1. **One command buffer**, one `queue.submit` (or the host's equivalent), holding rubble's
   substeps, the contact ingest, the stress cascade, the GPU splits, the routing check and
   the punch;
2. **One status read** (one host map of device memory) after it;
3. **No other host↔device traffic that waits on the GPU** during the tick (no
   `device.poll` and no map, except that one read);
4. **The stress stage described in a backend-neutral plan**: an ordered command list, like
   `engine-exec` `rigid_plan::Cmd`, which any host replays. No host hand-records stress
   dispatches;
5. **Every GPU kernel is Slang**: no WGSL, MSL or CUDA written by hand;
6. **Lossless and deterministic**: every recorded digest unchanged, and runs repeat bit for
   bit.

### Definitions (no other reading counts)

- **Tick:** one call of `GpuWorld::step_frame` that takes the resident path
  (`resident_now()` true). This includes split frames, GPU-punch frames, frames whose
  cascade disconnects islands, and frames after an aborted GPU split.
- **Submission:** one call to `wgpu::Queue::submit` (or Metal `commit`, CUDA stream
  launch-and-record) made during the tick, by **any** code: rubble, the ingest, the
  solver, the split path, diagnostics excepted only when their env switch is set.
- **Read:** one `map_async` and wait, or any `gpu.read*` / `read_ranges` call.
- **Allowed exceptions:**
  - frame 0;
  - a tick in which the host adds or removes bodies on purpose (`add_impactor`, scene
    edits);
  - runs with a diagnostic env switch (`STRESS_GPU_DET_*`, `*_TIMING`, `*_TRACE` that
    reads). The default build's separate submissions are the state being replaced, not an exception.

  Nothing else is an exception. In particular "rare fallback paths" are **not** exceptions:
  they either stay within one submission and one read or are removed.

## 2. Where it stands (measured, not claimed)

| Item | Status |
|---|---|
| Kernels in Slang | **Met** (`stress-gpu/shaders/*.slang`, generated MSL and WGSL checked in) |
| One status read per resident tick | **Met** on the common path (frame-mode GPU splits included) |
| One submission per resident tick | **Not met. Counted (W1):** 7 per tick by default, 2 with `STRESS_FRAME_SUBMITS=one` (`wall_ram_late2`, `blast_parity_routed`, GPU punch; section 3) |
| Backend-neutral plan | **Not met.** Stress dispatches are hand-recorded in wgpu: `solver.rs`, `ingest.rs`, `resident.rs`, `routing.rs`, `gpu_pass.rs`, `gpu_split.rs` |
| Apple Metal, web WebGPU, Linux CUDA (your requirement) | **Not met.** The stress stage runs only on native wgpu. `engine-metal` replays `rigid_plan` (rubble only); the web host runs rubble only; `engine-cuda` is a source-emitting stub with no runtime |
| Lossless | WIP checked on the quick tier only (46728bf1, both modes) |
| Determinism | Suites repeat; the 36-shot barrage has a latent GPU race (charter log, "Barrage nondeterminism") |
| Speed | WIP single-buffer mode is slower on the quick tier: median 8.1 against 7.3 ms, p95 14.5 against 12.0. The host's follow of the last frame's splits no longer overlaps the engine's GPU work |

## 3. Known extra submissions and reads on the resident path (to remove)

**Counted by W1** (`engine_gpu_io`, `GpuWorld::last_tick_io`, `STRESS_BENCH_IO=1`) on
`wall_ram_late2` under `blast_parity_routed` with `STRESS_IMPACT_PUNCH=gpu`, one engine
substep, every frame after the first (digest c55cf1b5 in both modes):

| Mode | Submissions per tick | Reads | Waits |
|---|---|---|---|
| Default | 7: `stress-rubble` `resident.rs` ingest start; rubble's step (`engine-webgpu` `gpu.rs`); `resident.rs` substep ingest and merge; `resident_frame.rs` the frame encoder; `solver.rs` the cascade; `resident_frame.rs` the read's own copy (`read_ranges`) | 1 | 0 |
| `STRESS_FRAME_SUBMITS=one` | 2: `solver.rs` the cascade (holding everything before it); the read's own copy | 1 | 0 |

So the "5 submissions" stated before W1 was a hand count, and it missed two: the frame
encoder and the read's copy. The read helper submits its copy: W2 must record the copy into
the frame's command buffer and map it after that one submission.

The list below was found by reading the code at 08efa0c. These paths are not on the frames
above. W2 must count each one on a scene that takes it.

1. `stress-rubble` `step_resident`: a submission between engine substeps when `n > 1`.
   rubble writes its host uniforms with `queue.write_buffer` once per step, and those
   writes land at a submission.
2. `solver.rs` `resident_cascade_with`: the gate-carry copy on a layout change
   (`gpu.queue.submit` of its own encoder).
3. `resident_frame.rs`: fallback cascade windows after an aborted frame split. Each window
   has its own submission and read.
4. `resident_frame.rs`: the host split path (`split_resident_with`, `sync`,
   `sync_chunk_state`). Reads and uploads when the GPU split did not run.
5. `RubbleEngine::resident_substeps`: the CCD bound reads bodies (`needs_ccd_bodies`). Off
   under `blast_parity` (no CCD substeps) but reachable.
6. `frame_log_range`: "a longer log is read separately" (a second read when the split log
   exceeds 256 KB).
7. Any `gpu.read` reached from `flush_pending`, `adopt_from_log` or `check_colliders`
   outside diagnostics.
8. `prepare_routing`, `set_impactors`, punch impactor upload: buffer creation and writes
   only (allowed: no submission), to be confirmed.

## 4. Work items, in order

Each item ends with a commit, the guard and the item's own test. Do not start the next
item with the previous one's test failing.

### W1. Count, and fail on, submissions and reads

- A per-tick counter on `Gpu` (`submits`, `reads`), incremented by every submit and read the
  stress side makes. `WgpuSim` gets the same counter, for rubble's submits.
  `GpuWorld::last_tick_io() -> (submits, reads)`.
- `stress_bench` JSON: per-frame `submits` and `reads` arrays, and their maxima after
  frame 0.
- `STRESS_STRICT_IO=1`: a resident tick with more than 1 submission or 1 read returns an
  error naming the call sites (a `#[track_caller]` location recorded with each count).
- **Done when:** the counter matches a hand count on `wall_ram_late2` (one engine substep).
  The bench prints per-frame counts. The full list in section 3 is confirmed or corrected
  in this file.
- **Status: done.**
  - The counter is the crate `engine-gpu-io`: every `queue.submit`, `map_async` and blocking
    poll in `engine-webgpu`, `stress-gpu` and `stress-rubble` records its call site. It is a
    shared crate, not a counter on `Gpu` and on `WgpuSim`, so both engines' calls land in
    one tick.
  - `GpuWorld::last_tick_io()` returns the counts and sites, and `last_tick_resident()`
    says whether the tick was resident.
  - Bench JSON `io`: per-frame `submits`, `reads`, `waits`, `resident`, the resident maxima
    after frame 0, and the sites of the worst frame.
  - `STRESS_STRICT_IO=1` exempts frame 0 and the tick after `add_impactor`.
  - Test: `stress-rubble/tests/frame_io.rs`.
  - Section 3 is corrected with the measured counts.

### W2. One submission and one read on every resident tick

Remove every item in section 3 that is on the resident path:

- **(1)** rubble's per-substep host uniforms move to a per-substep slot (a uniform array
  indexed by substep, written once before the frame) or are written by a GPU kernel. Then
  `step_encode` for every substep goes into the same encoder.
- **(2)** the gate-carry copy is encoded into the frame encoder.
- **(3, 4)** an aborted GPU split must not fall back to host work inside the tick. Either the
  capacities (island headroom, bodies, colliders) are explicit engine-wide settings that
  fail with a clear error when exceeded (AGENTS.md), or the split continues next tick on
  the GPU. Decide with the user before choosing.
- **(5)** under any profile that the resident path runs, the CCD bound comes from the
  previous tick's one read (already the design: `resident_status`); confirm it.
- **(6)** the log read is sized so that one read always covers it, or overflow fails with a
  clear error.
- **Done when:** with `STRESS_STRICT_IO=1`, these complete with no error, and the
  per-frame maxima after frame 0 are **submits 1, reads 1**:
  - `stress_bench --config blast_parity` on vibe-town quick and standard;
  - the 36-shot barrage (`STRESS_RUBBLE_MAX_BODIES=65536 STRESS_RUBBLE_MAX_COLLIDERS=262144`);
  - `wall_ram_late2` and `wall_row2_late2` under `blast_parity_routed` with
    `STRESS_IMPACT_PUNCH=gpu`;
  - a CCD scene with more than one engine substep (to be added: a fast thin bar into a
    wall).

### W3. Speed parity with the many-submission mode

- The host's follow of the last frame's splits (`flush_pending`) runs after the frame's
  submission and before its read, as it did with separate submissions. Its writes must not
  be needed by this tick's GPU work: prove it, or move them to the next tick's encoder.
- **Done when**, on a quiet M3 Max, one job at a time:
  - vibe-town quick and standard medians and p95s are within 3 % of the default (separate submissions),
    or better;
  - every M6 target still holds (standard 11.46 / 37.88 / 48.24 ms, quick 11.38 / 15.32 /
    35.5 ms).

### W4. The stress stage as a backend-neutral plan

- A command list type for the stress stage, in `engine-exec` or a sibling crate with no
  wgpu dependency. Per command:
  - module and entry (the Slang file and entry point);
  - its bindings, as buffer ids into a host-owned buffer table;
  - its parameter block, as bytes;
  - a direct group count or an indirect (buffer id, offset);
  - buffer copies and clears;
  - pass boundaries.
- `encode_stress_frame(inputs) -> Vec<StressCmd>` produces the whole tick's stress work:
  ingest, cascade passes, split passes, routing check, punch. It interleaves with
  `rigid_plan::encode_step`'s commands into **one** list for the tick.
- The wgpu host replays the list; `solver.rs`, `ingest.rs`, `resident.rs`, `routing.rs`,
  `gpu_pass.rs` and `gpu_split.rs` no longer call `begin_compute_pass` on the resident path.
- **Done when:**
  - `grep` finds no `begin_compute_pass` or `dispatch_workgroups` reachable from
    `resident_frame` outside the plan's replayer;
  - every recorded digest is unchanged (guard all identical);
  - W2's counts still hold.

### W5. Three platforms build and run the same plan: Apple Metal, web WebGPU, Linux CUDA

Your requirement of 2026-10-18. Each platform replays the **same** tick command list (W4), with
kernels compiled from the **same** Slang sources (Slang to MSL, WGSL and CUDA C++; nothing
hand-written per backend). Today: rubble's own step has a Metal host (`engine-metal`, replays
`rigid_plan`) and a browser host (wasm, `web/`). `engine-cuda` is a stub that emits and
validates CUDA sources with no runtime. The stress stage runs only on native wgpu.

- **Apple Metal** (macOS, M3 Max): a host that replays the tick's list on Metal directly
  (`engine-metal`), not through wgpu.
  - Builds: `cargo build --release` of the Metal host with the stress stage.
  - Runs: `wall_ram_late2`, `wall_row2_late2` (`blast_parity_routed`, GPU punch) and
    vibe-town quick and standard (`blast_parity`), each tick one command buffer
    (`commit`) and one read, by W1's counter.
  - Agrees: the same digests as native wgpu on Metal (both drive the same Apple GPU), or
    else the first differing stage found and the cause fixed.
  - Speed: M6 targets met.
- **Web WebGPU** (browser, wasm): the stress stage built to wasm, its kernels from the
  Slang-generated WGSL, replaying the tick's list through the browser's WebGPU.
  - Builds: `wasm-pack`/`cargo build --target wasm32-unknown-unknown` of the stress stage
    plus rubble's web host, and `npm run build` in `web/`.
  - Runs: `wall_ram_late2` and vibe-town quick in Chrome on the M3 Max, each tick one
    `queue.submit` and one read (counter exposed to the page and logged).
  - Agrees: ram speed and fragments within 1 % and 1 fragment of native, broken bonds
    within 1 %. Bit identity is not required: the browser's WGSL compiler is not Metal's
    MSL path.
  - Speed: frame times reported (no target until measured; the user decides it).
- **Linux CUDA** (NVIDIA GPU, Linux): a real CUDA runtime host in `engine-cuda` (driver
  binding, buffers, module loading, launches, one stream per tick, one device-to-host copy
  for the status), replaying rubble's step plan and the stress list, kernels from Slang's
  CUDA target.
  - Builds: on Linux, with the CUDA toolkit, `cargo build --release -p engine-cuda` with
    its runtime feature. On machines without CUDA, the source-emitting build still
    passes, as today.
  - Runs: the same scenes as Metal on an NVIDIA GPU, each tick one stream submission and
    one read, by W1's counter (the CUDA host counts its own).
  - Agrees: ram speed and fragments within 1 % and 1 fragment of native, broken bonds
    within 1 %; deterministic run to run on that GPU (identical digests).
  - **Needs an NVIDIA Linux machine to verify.** If none is available to this work, W5's
    CUDA part is reported as **built, not run**, and is not done.
- **Done when:** all three platforms meet their Builds, Runs and Agrees above, with the
  commands and numbers logged in the charter.

### W6. Determinism

- The barrage race (charter log) is found and fixed.
- **Done when:** six runs of the barrage give one digest, and the guard is all identical.

## 5. Acceptance: 5d is done when, and only when

All of the following, each with its evidence recorded in the charter log (command,
numbers, commit):

1. **W1** counters exist and are tested.
2. **W2:** every listed scene runs under `STRESS_STRICT_IO=1` with submits 1 and reads 1
   per resident tick.
3. **W3:** speed within 3 % of the many-submission mode, and all M6 targets met.
4. **W4:** no hand-recorded stress dispatch on the resident path, and the guard all
   identical.
5. **W5:** Apple Metal, web WebGPU and Linux CUDA each build, run the acceptance scenes at
   one submission and one read per tick, and agree with native as W5 states. CUDA verified
   on an NVIDIA Linux GPU: built but not run is **not done**.
6. **W6:** the barrage is deterministic over six runs.
7. All stress-gpu and stress-rubble test suites pass.

Anything less is reported as **not done**, with the failing item named.

## 6. Rules (to not repeat the mistake)

- **Never write "one submission" or "one read" without the W1 counter's output for the
  stated scene.** Report the counter, not the design.
- Report what is measured, per frame kind (quiet, split, punch, fallback), not the common
  case.
- Each work item is committed with its test. A WIP commit says what is unverified.
- The guard (`python3 scripts/destruction_peak.py guard --no-build`) runs before every
  commit that touches a shader or the resident path.
- Perf runs one at a time on a quiet machine.
- When a requirement cannot be met as written, say so and ask. Don't redefine it.
