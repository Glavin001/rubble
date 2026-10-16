---
name: W4 async readback elimination
overview: Finish M2 by landing W4 (eliminate the three remaining post-solve GPU->CPU readbacks via GPU-persisted warm-start, demoted debug APIs, and an async snapshot ring), then W5/M2(e) (single-encoder frame, delete readback path, zero-readback assert). Written for handoff — includes measurement methodology, expectations, and assumptions.
todos:
  - id: qw-timing
    content: "Quick win 0: split solve_ms timing buckets so readbacks report separately from solve"
    status: completed
  - id: qw-overlap
    content: "Quick win 1: overlap the three independent post-solve awaits (joint await)"
    status: completed
  - id: w4a-warm
    content: "W4a-1: stop CPU warm-contact round trip; trust GPU prev_contacts on clean frames"
    status: completed
  - id: w4a-events
    content: "W4a-2: make events opt-in and last_contacts an explicit blocking debug/query API; delete hot-path contact download"
    status: completed
  - id: w4a-pairmap
    content: "W4a-3 (optional/perf): CAS pair map replacing O(new x prev) warm-start match"
    status: completed
  - id: w4b-ring
    content: "W4b: async snapshot ring for body states; keep blocking sync() for tests"
    status: completed
  - id: w4b-web
    content: "W4b: wire snapshot ring through wasm + demo render loop; verify E2E green"
    status: completed
  - id: w5-encoder
    content: "W5: single-encoder frame; delete readback paths; zero-readback debug assert"
    status: completed
  - id: w5-measure
    content: Refresh perf baselines + browser re-measurement to confirm projections
    status: completed
isProject: false
---

# Finish M2: W4 + W5 (eliminate the last hot-path readbacks)

## Context you need before touching anything

Read [docs/production-plan.md](docs/production-plan.md) — especially §2 D2/D3, §3 M2, and the execution log at the bottom (W1/W2/W3 entries). This plan implements the remaining M2 stages (W4, W5). Branch: `claude/rubble-physics-production-plan-jmj42u`.

The engine is a GPU AVBD rigid-body solver. One `World::step()` runs: predict → broadphase (LBVH) → narrowphase → GPU contact sort → warm-start → adjacency → coloring → solve → extract. After W1–W3, everything up to and including the solve runs with **zero** CPU awaits on web. What remains are three post-solve readbacks (details below).

### Key files

- [crates/rubble3d/src/gpu/mod.rs](crates/rubble3d/src/gpu/mod.rs) — the GPU pipeline. The wasm step entry is `step_with_contacts_async` (~line 2251); native timed path is `step_with_contacts_timed` (~line 1760); native untimed `step_with_contacts` (~line 1706).
- [crates/rubble3d/src/lib.rs](crates/rubble3d/src/lib.rs) — the `World` API layer. `step_async` (~line 1160) and `step` (~line 1040) call the pipeline, then feed `contact_persistence`, `collision_events`, `last_contacts`, and copy `results` into `self.states`.
- [crates/rubble-gpu/src/buffer.rs](crates/rubble-gpu/src/buffer.rs) — `GpuBuffer::download/download_async`, `GpuAtomicCounter::read/read_async`. Every one of these is a submit + `map_async` + await round trip.
- [crates/rubble-wasm/src/lib.rs](crates/rubble-wasm/src/lib.rs) — wasm bindings. `PhysicsWorld3D::step()` (~line 510) awaits `world.step_async()`; `copy_transforms_into` (~line 561) reads `World` state for rendering.
- [web/src/3d/main.ts](web/src/3d/main.ts) — the demo. `loop_()` awaits `world.step()`, then `syncTransformCache()` + render. Overlay text comes from `last_step_overlay_text`.
- [crates/rubble3d/src/gpu/warmstart_wgsl.rs](crates/rubble3d/src/gpu/warmstart_wgsl.rs) — `WARMSTART_MATCH_WGSL` kernel matching new contacts against `prev_contacts`.

## The measured problem (2026-07-05, 137-body Pyramid scene, M3 Max, Chrome)

Step 45 ms → ~16 ms after W1–W3. The remaining ~15 ms is three **serialized** GPU→CPU round trips in `step_with_contacts_async`, each ~3–5 ms of `mapAsync` latency (NOT bandwidth — the data is ~100 KB; Chrome resolves buffer maps on device-timeline ticks):

1. `self.contact_count.read_async(...)` — frame-end contact count (gpu/mod.rs ~line 2302)
2. `self.download_contacts_exact_async(total)` — full contact buffer download (~line 2306)
3. `self.body_states.download_async(...)` — body states for rendering (~line 2325)

Readbacks 1–2 are timed inside `solve_ms`, so the overlay's "Solve 10.6 ms" is mostly readback latency, not solve work. The persistent-kernel path (W3) is confirmed taken at 137 bodies (`num_bodies <= PERSISTENT_MAX_BODIES` = 256, gpu/mod.rs line 1558); actual solve compute is sub-ms.

### Why does the contact download exist at all? (trace before changing)

`World::step_async` (lib.rs ~1160) does this every frame:

```
prev = contact_persistence.prev_contacts()      // CPU Vec from LAST frame's download
warm = Some(prev)                               // re-UPLOADED to GPU prev_contacts
(results, new_contacts) = pipeline.step_with_contacts_async(..., warm, ...)
events = contact_persistence.update(&new_contacts)   // CPU pair-set diff -> Started/Ended
last_contacts = new_contacts.clone()                 // debug/test API
```

So contacts are downloaded each frame only to be (a) re-uploaded next frame as warm-start input, (b) diffed on CPU for collision events, (c) stored for `World::last_contacts()` (used by narrowphase-validation tests only).

**Critical observation:** the GPU already keeps solved contacts! `swap_contact_buffers` (gpu/mod.rs ~line 3890) does a GPU-side `copy_buffer_to_buffer` of solved contacts → `prev_contacts` after every solve. The CPU round trip (download → `contact_persistence` → re-upload via `upload_warm_contacts`) re-uploads bit-identical data over it. The download can be removed without the CAS pair map, by trusting the GPU-side copy.

**Body-index caveat (the reason the CPU path was safe):** contacts store *compacted* body indices (alive bodies in insertion order, see `alive_indices` in lib.rs). If bodies are added/removed between frames, the compaction remap shifts indices and GPU `prev_contacts` becomes stale/wrong. `World` already tracks exactly this with the `gpu_dirty` flag (lib.rs ~line 583, set by add/remove/set_* mutations). Rule: **on `gpu_dirty` frames, invalidate warm-start** (`prev_contact_count = 0`) instead of trusting stale indices. Losing warm-start for one frame after a mutation is acceptable (it self-heals next frame).

## How to measure (do this before AND after every stage)

### Browser (the number the user cares about)

1. Build + serve: `npm --prefix web run dev` (rebuilds wasm via `predev`), open `http://localhost:5173/src/3d/index.html`, Pyramid scene (default, 137 bodies).
2. **The tab must be visible and focused.** Backgrounded/occluded tabs throttle `requestAnimationFrame` to ~1 FPS and inflate mapAsync latency — measurements from a hidden tab are garbage. (This bit us: first sample showed 85 ms.)
3. Read the overlay (top-left, `#timings` element), or from DevTools console:

```js
// steps/sec over 4s + overlay breakdown
const s0 = window.__rubble_test.stepCount; const t0 = performance.now();
setTimeout(() => {
  const dt = (performance.now() - t0) / 1000;
  console.log('steps/sec:', (window.__rubble_test.stepCount - s0) / dt);
  console.log(document.getElementById('timings').textContent);
}, 4000);
```

4. Record median of ~5 readings; overlay updates per frame and jitters ±5 ms.

### Native (regression net)

- Perf harness: `cargo nextest run -p rubble3d --profile ci` includes `perf_baseline_tests`; baselines in [crates/rubble3d/tests/perf/baselines.json](crates/rubble3d/tests/perf/baselines.json). NOTE: baselines are intentionally stale (pin the pre-M2(d) shape); do NOT refresh until W5.
- Export env first (Linux/CI): `WGPU_BACKEND=vulkan`, lavapipe ICD per README. On macOS nothing needed.

### Correctness gates (run after every stage, all must pass)

- `cargo nextest run --workspace --profile ci` — full ladder (200+ tests).
- Determinism probe: `cargo nextest run -p rubble3d --test determinism_probe_tests`. The 137-body pyramid fingerprint must stay **bit-identical** across your change unless the change legitimately alters float summation order — if it changes, stop and understand why before proceeding. (Reference fingerprint from W2/W3: `5a97ed437cee8bae`.)
- Web E2E: `npm --prefix web run build:all && npm --prefix web test` (Playwright on SwiftShader; catches web-only breaks — it caught the W1 dispatch-bound incident).
- `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings`.

## Expectations (what "done" looks like at each stage)

| Stage | Web step (137-body Pyramid, M3-class) | Notes |
|---|---|---|
| Today | ~16 ms | ~15 ms is the three awaits |
| Quick win 1 (overlap awaits) | ~8–11 ms | 3 serialized round trips → 1 joint wait |
| W4a (no contact download) | ~8–10 ms | removes await #1+#2 entirely |
| W4a + W4b (no state await) | ~3–6 ms | zero hot-path awaits; step = submit + snapshot poll |
| W5 (single encoder) | ~2–4 ms | plan §4 target; CPU cost = command recording |

If a stage doesn't move the number, re-measure with the tab focused, then profile before writing more code (Chrome tracing: `chrome://tracing` or DevTools Performance panel; look for `mapAsync` resolution gaps).

## Stage QW — quick wins (land first, small diffs)

**QW-0: honest timing buckets.** In `step_with_contacts_async` (gpu/mod.rs ~2296–2326) and `step_with_contacts_timed` (~1770–1805), move the count-read + contact-download out of `solve_ms` into `contact_fetch_ms` (the field exists in `StepTimingsMs`, [crates/rubble-gpu](crates/rubble-gpu)) so the overlay stops mislabeling readbacks as solve time. Update `format_text_overlay` labels if needed. This makes all later measurements trustworthy.

**QW-1: overlap the awaits.** The three awaits are data-independent (count+contacts feed persistence/events; states feed rendering). Restructure the wasm path: enqueue extract + restitution dispatches BEFORE awaiting anything (restitution needs the "had contacts" bool — use `total_upper > 0`, which is already known without a readback, matching the in-shader guards' semantics), then issue all three copies in one encoder and await the three maps jointly (`futures::join!` or sequential awaits after all `map_async` calls are registered — the point is one GPU drain, not three). Expected: ~15 ms → ~5 ms of await time. Keep behavior identical otherwise; determinism fingerprint must not change.

## Stage W4a — delete the hot-path contact download

Three sub-stages, each independently shippable:

**W4a-1: stop the CPU warm-contact round trip.**
- In `World::step_async`/`step` (lib.rs ~1045/~1160): pass `warm_contacts = None` when the previous frame's GPU `prev_contacts` is valid; the pipeline's `swap_contact_buffers` already maintains it. Track validity with a new flag on the pipeline (e.g. `gpu_prev_contacts_valid: bool`), invalidated when: `gpu_dirty` (body set/pose mutated — compaction indices shift), substeps > 1 mid-frame semantics differ, or `World::new`/scene reset.
- `upload_warm_contacts` and the `warm_contacts` parameter stay (tests use them, e.g. explicit-warm-start tests).
- Gate: full ladder + determinism probe unchanged. This changes no math — the uploaded data was bit-identical to what the GPU already had.

**W4a-2: demote events + last_contacts; delete the download.**
- Add `World::set_events_enabled(bool)` (default OFF). Only when ON does the step download contacts and run `contact_persistence.update`. The demo doesn't use events; tests that need `drain_collision_events` flip it on explicitly (grep for `drain_collision_events` in tests and add the opt-in there).
- `World::last_contacts()` becomes lazily populated: a `World::fetch_last_contacts()` (native blocking / wasm async) that downloads on demand. Update the narrowphase-validation tests that call `last_contacts()` (grep `last_contacts(` in `crates/rubble3d/tests/` and `crates/rubble-testkit/`) to call the fetch first.
- With events off and no fetch, `step_with_contacts_async` skips `download_contacts_exact_async` entirely. The frame-end `contact_count.read_async` must STAY (it owns overflow healing — see W1 notes in the production plan) but joins the joint await from QW-1; long-term (W5) it can become fire-and-forget polled next frame.
- `swap_contact_buffers` needs the solved count GPU-side — it currently takes `contact_count: u32` from the CPU. Replace with a GPU-side `copy_buffer_to_buffer` sized by the capacity bound (`total_upper`) or keep using the count from the (now overlapped) count-read — simplest: keep the count-read result, it's no longer on the critical path.
- Gate: full ladder green after test updates; E2E green; browser overlay shows Contacts/fetch ≈ 0.

**W4a-3 (optional, do only if warm-start match shows up in profiles): CAS pair map.**
The current `WARMSTART_MATCH_WGSL` matches new→prev contacts; if it's O(new×prev) it will hurt at 10k+ bodies (fine at demo scale). Port avbd-metal's scheme per production-plan D3: open-addressing map keyed `(body_a,body_b)`, capacity `nextPow2(2*max_pairs)`, linear probe, CAS insert; per-point match within the pair by feature id. This is a perf refactor, not a readback fix — defer unless the native 4k-body perf scene (`perf_baseline_tests`) shows warm-start as a top cost.

## Stage W4b — async snapshot ring for body states

Kills the last await (`body_states.download_async`, the 4.8 ms "Extract" lane).

- New struct in [crates/rubble-gpu/src/buffer.rs](crates/rubble-gpu/src/buffer.rs) (or a new module): `SnapshotRing<T>` with 3 slots, each `{ staging: wgpu::Buffer (MAP_READ|COPY_DST), state: Empty|InFlight|Ready(len), fence/map-callback }`.
  - `enqueue(encoder_or_queue, src_buffer, len)`: pick a non-InFlight slot, record copy, submit, call `map_async` with a callback that flips the slot to Ready. Never blocks.
  - `latest() -> Option<&[T]>`: most recent Ready slot (map, read, keep mapped until slot reuse).
  - `sync() -> &[T]`: blocking drain for tests (native: `device.poll(Wait)`; wasm: await the map future).
- Wire into the step paths: replace the terminal `body_states.download_async` with `ring.enqueue(...)`, and return `ring.latest()` — i.e. **the step returns last frame's states** (≤1 frame lag, industry standard; production-plan D2 records this decision). First frame returns the initial CPU-known states.
- `World` layer: `self.states[orig] = results[slot]` (lib.rs ~1176) now holds slightly-stale data. Everything downstream (wasm `copy_transforms_into`, viewer) reads `World::states` — no changes needed there. BUT tests that step once and assert exact positions will break: add `World::sync()` that drains the ring, and call it from the test harness / testkit scenario runner ([crates/rubble-testkit/src/scenario.rs](crates/rubble-testkit/src/scenario.rs)) so ALL existing oracle/invariant tests see synchronous semantics without individual edits. Grep for direct `world.step(` usage in tests to confirm the harness covers them.
- Restitution caveat: `dispatch_restitution` runs after extract; the snapshot copy must be recorded AFTER restitution's dispatch in the same submission order, or bounces will lag one extra frame in render.
- Demo: no change needed ([web/src/3d/main.ts](web/src/3d/main.ts) `loop_` already just awaits `step()` then reads transforms) — `step()` simply returns faster.
- Gate: full ladder green via harness `sync()`; determinism probe uses `sync()` (fingerprint unchanged); E2E green; browser overlay Extract ≈ 0. Visually check the demo for judder (1-frame lag should be imperceptible at 60fps; if the demo alternates stale/fresh, the ring depth or slot selection is wrong).

## Stage W5 / M2(e) — single-encoder frame + delete readback path

- Collapse remaining per-stage submits into one (or two) command encoders per step: predict/AABB (already one submit), detection chain, sort/warm-start/adjacency/coloring, solve, extract/restitution/snapshot-copy. Most stages already record into single submits (see W1/W2 execution-log entries); this stage merges across stage boundaries where no CPU decision intervenes. The native overflow-retry (grow-and-retry on pair overflow) is the one genuine CPU decision left — keep it as the boundary.
- Add the zero-hot-path-readback assertion: debug-only counter in `GpuContext` incremented by `download*/read*`; `World::step` asserts it stays 0 through the hot path (allow listed: explicit `fetch_last_contacts`, `sync`, overflow-heal count read if still present).
- Delete dead code: the old blocking `step()` variants' readback paths that nothing uses anymore (verify with `cargo build --workspace` + grep for callers before deleting).
- Refresh perf baselines NOW (`baselines.json` — the plan says they pin the pre-rewrite shape until M2 completes; W5 completes it). Commit new baselines and note the delta in the production-plan execution log.
- Re-measure browser per the methodology above; update [docs/production-plan.md](docs/production-plan.md) execution log with W4/W5 entries (same style as W1–W3).
- Gate: two-world determinism green native+web, full ladder, E2E, zero-readback assert active in debug, browser step ≤ 4 ms on the 137-body demo (M3-class hardware).

## Assumptions (verify if anything looks off)

- Chrome `mapAsync` round-trip latency ~3–5 ms is scheduling-bound and cannot be tuned away — the only fix is not awaiting at all. If a single overlapped await still costs >8 ms, investigate Chrome's `queue.onSubmittedWorkDone` timing before blaming the plan.
- 137-body Pyramid demo scene is the reference workload; the persistent-kernel (W3) path covers it. Scenes >256 bodies take the indirect per-color loop — spot-check "Grid Boxes" (larger) after each stage too.
- Compound scenes still take a CPU pair-expansion path with mid-detection awaits (gpu/mod.rs ~2151–2214). Out of scope here; owned by M3 (compound broadphase leaves). No demo scene uses compounds.
- The wasm and native pipelines share all kernels; only the step-orchestration functions differ (`step_with_contacts_async` vs `step_with_contacts_timed`). Every change must be made to BOTH or deliberately scoped with a comment.
- `web/src/wasm/` is generated by wasm-pack (`npm --prefix web run wasm`); never hand-edit.

## After M2 closes → M3

Collision robustness (AABB fattening, speculative margins, compound leaves, stable feature IDs + SAT hysteresis) — this is what flips the pyramid known-failure (the visible bounce/collapse in the demo, which is NOT a perf issue and NOT restitution; see production-plan §1.4 and M3).
