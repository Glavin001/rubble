---
name: eliminating-gpu-readbacks
description: Find and eliminate hot-path GPU-to-CPU readbacks (mapAsync awaits, buffer downloads) that dominate per-step latency, especially on WebGPU/WASM where each await costs 3-5 ms. Use when step time is much larger than GPU compute time, when profiling shows time in "solve" or "extract" that is actually readback latency, or when scaling work to 10k+ bodies.
---

# Eliminating GPU→CPU Readbacks

On WebGPU each `mapAsync` await costs ~3–5 ms of latency regardless of data size, and serialized awaits add up: three per frame put a ~10–15 ms floor under the step no matter how fast the kernels are. This project went 45 ms → ~1 ms on the pyramid demo almost entirely by removing readbacks, not by making kernels faster.

## Step 1: Make timing honest first

Before optimizing, verify the timing buckets attribute readback waits to their own bucket, not to whatever kernel ran before them. A "slow solve" that is actually a `read_async` await sends you optimizing the wrong thing. In this repo the phase buckets live in `step_with_contacts_timed` / `step_with_contacts_async` in `crates/rubble3d/src/gpu/mod.rs`.

## Step 2: Find every hot-path readback

All downloads funnel through `GpuBuffer::{download, download_async, read, read_async}` in `crates/rubble-gpu/src/buffer.rs`, which call `ctx.note_readback()`. Grep call sites reachable from the per-step path. Common offenders and their fixes:

| Readback | Fix pattern |
|---|---|
| Count read (e.g. contact count) to size a later dispatch | Keep the count in a GPU buffer; dispatch an upper bound and have the kernel clamp with `if i >= min(count_buf[0], upper_bound) { return; }` |
| Downloading data only some callers need (e.g. contact events) | Make it opt-in (`set_events_enabled`) with an explicit blocking fetch API (`fetch_last_contacts`) for tests/tools |
| Per-frame state extraction for rendering | Snapshot ring: enqueue copy into a rotating staging buffer, poll non-blocking, return the latest *completed* snapshot (~1 frame lag). See `crates/rubble-gpu/src/snapshot.rs`; keep a blocking `sync()` for tests |
| CPU round-tripping data the GPU already has (re-uploading warm-start contacts) | Trust the GPU-resident buffer; invalidate for one frame when CPU mutations shift indices |

Also overlap what remains: dispatch independent GPU work *before* any unavoidable await so the GPU works while the CPU waits.

## Step 3: Lock it in with an assertion

Add a debug-only zero-readback assertion on the resident path so regressions fail loudly: snapshot `ctx.readback_count()` at step entry and assert it is unchanged at exit (see the `#[cfg(debug_assertions)]` block in `step_with_contacts_async`, `crates/rubble3d/src/gpu/mod.rs` ~line 2330). Any new `download`/`read` on the hot path trips it immediately.

## Constraints to preserve

- **Determinism**: replacing CPU logic with GPU kernels must keep the determinism probe (`determinism_probe_tests.rs`) bit-identical. See the `gpu-determinism-wgsl` skill.
- **Test APIs**: tests that inspected downloaded data need explicit fetch calls, not silent empties — update them in the same change.
- **Buffer overflow**: when dispatch sizes become upper bounds instead of exact counts, prove overflow is impossible (e.g. emission already clamped to capacity) rather than relying on the count you no longer read.
