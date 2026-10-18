# Plan 5d: the stress stage in rubble's frame, one command buffer, one read

Status: **in progress** (WIP commit 08efa0c, 2026-10-18). Parent charter:
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
    reads, `STRESS_FRAME_SUBMITS=many`).

  Nothing else is an exception. In particular "rare fallback paths" are **not** exceptions:
  they either stay within one submission and one read or are removed.

## 2. Where it stands (measured, not claimed)

| Item | Status |
|---|---|
| Kernels in Slang | **Met** (`stress-gpu/shaders/*.slang`, generated MSL and WGSL checked in) |
| One status read per resident tick | **Met** on the common path (frame-mode GPU splits included) |
| One submission per resident tick | **Not met.** Before 08efa0c: 5 (ingest start, rubble step, substep ingest, ingest merge, cascade). After 08efa0c: 1 on frames with one engine substep and no fallback; more on the paths in 3. Not yet counted by instrumentation |
| Backend-neutral plan | **Not met.** Stress dispatches are hand-recorded in wgpu: `solver.rs`, `ingest.rs`, `resident.rs`, `routing.rs`, `gpu_pass.rs`, `gpu_split.rs` |
| Second host replays it | **Not met.** `engine-metal` replays `rigid_plan` (rubble only); `engine-cuda` is a source-emitting stub with no runtime |
| Lossless | WIP checked on the quick tier only (46728bf1, both modes) |
| Determinism | Suites repeat; the 36-shot barrage has a latent GPU race (charter log, "Barrage nondeterminism") |
| Speed | WIP single-buffer mode is slower on the quick tier: median 8.1 against 7.3 ms, p95 14.5 against 12.0. The host's follow of the last frame's splits no longer overlaps the engine's GPU work |

## 3. Known extra submissions and reads on the resident path (to remove)

Found by reading the code at 08efa0c; work item W1 must confirm the list by counting.

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
  - vibe-town quick and standard medians and p95s are within 3 % of `STRESS_FRAME_SUBMITS=many`,
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

### W5. A second host replays the same plan

- `engine-metal` replays the tick's list (rubble already replays `rigid_plan` there),
  with the stress kernels from the same Slang sources.
- **Done when:** `wall_ram_late2` and the vibe-town quick tier run on the Metal host with
  one command buffer and one read per tick, and the results agree with wgpu's. Either the
  same digests, or a stated tolerance agreed with the user if bit identity across backends
  is not achievable.
- CUDA: `engine-cuda` has no runtime (it emits and validates sources). Replaying on CUDA
  needs that runtime first. **This is a decision for the user:** in scope for 5d, or a
  follow-up. Until decided, W5's acceptance is Metal, plus Slang emitting CUDA source for
  every stress kernel without errors.

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
5. **W5:** Metal replays the plan with agreeing results (CUDA as decided by the user).
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
