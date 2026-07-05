# Performance Research Plan

## Goal

Improve rigid body performance for Rust and WASM/WebGPU without degrading physics correctness or splitting the engine into separate platform-specific implementations.

Primary target metric:

- `step_ms_median` from the WebGPU stress benchmark

Primary quality constraint:

- no regression in collision correctness, stacking behavior, or stability relative to the current engine behavior

## Principles

- Optimize one bottleneck at a time.
- Change one ownership boundary at a time: readback, scheduling, extraction, or rendering.
- Do not combine solver scheduling changes with readback removal in the same experiment.
- Treat correctness regressions as blockers even if a benchmark gets faster.
- Prefer shared Rust/WGSL logic over Web-only code paths.

## Benchmarking Strategy

### Primary benchmark

Use the 20k-body WebGPU stress benchmark:

```bash
npm --prefix web run bench:stress
```

Relevant metrics:

- `step_ms_median`: top-line performance target
- `step_ms_p95`: frame-time stability / jitter
- `solve_ms_median`
- `narrowphase_ms_median`
- `extract_ms_median`
- `broadphase_ms_median`
- `upload_ms_median`

Rules for comparison:

- keep scene, browser path, body count, and warmup fixed
- compare median and p95 first, not mean
- record commit SHA with every benchmark capture
- only count a phase optimization as a win if total `step_ms_median` also improves

### Viewer frame timing

For live viewer work, collect end-to-end frame timing separately from physics-only timing:

- world step
- transform copy / readback
- JS matrix update
- render
- total frame

This distinguishes physics bottlenecks from presentation bottlenecks.

## Correctness Gates

### Primary hard gates

Use these first for every experiment:

```bash
npm --prefix web test -- tests/ground-contact.spec.ts --reporter=list
npm --prefix web test -- tests/stability-1step.spec.ts --reporter=list
```

Why these matter:

- `ground-contact.spec.ts` is the best current acceptance test for contact generation and non-tunneling behavior.
- `stability-1step.spec.ts` is a cheap smoke test for basic integration stability.

### Investigative stability tests

Run these as secondary checks:

```bash
npm --prefix web test -- tests/stability-stack.spec.ts --reporter=list
npm --prefix web test -- tests/stability-pyramid.spec.ts --reporter=list
```

Current status:

- `stability-stack.spec.ts` fails in the current checkout/environment even with no tracked code changes.
- `stability-pyramid.spec.ts` is useful for settling behavior, but has also been noisy as a strict blocker.

Because of that, treat `stack` and `pyramid` as investigative tests until their baselines are repaired.

## Rust and Shader Validation

For any GPU pipeline, WGSL, or buffer-layout change:

```bash
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json

cargo check -p rubble3d -p rubble-wasm
cargo test -p rubble3d --test wgsl_validation_tests --test gpu_render_path_tests
```

Why these matter:

- `cargo check` catches API and layout fallout quickly
- `wgsl_validation_tests` catches shader parse / validation issues before browser runs
- `gpu_render_path_tests` guards the explicit GPU-to-render extraction path

## Standard Experiment Loop

For each optimization pass:

1. Capture baseline benchmark numbers.
2. Run hard correctness gates.
3. Make one narrow change.
4. Re-run hard correctness gates immediately.
5. Run Rust/WGSL validation if GPU code changed.
6. Re-run the stress benchmark.
7. Compare median and p95 against baseline.
8. Run investigative stability tests if the change is likely to affect settling or stacking.

If correctness changes, revert first and investigate second.

## What To Measure Explicitly

When profiling synchronization-heavy paths, instrument exact waits instead of inferring them from broad phase buckets.

Measure separately:

- LBVH pair-count readback
- contact-count readback
- contact buffer download
- state / transform download
- queue submissions that force completion

This prevents chasing the wrong sync point.

## Recommended Optimization Order

### 1. Fix the gates and measurement hygiene

- keep `ground-contact` and `stability-1step` as hard blockers
- repair or re-baseline `stability-stack` and `stability-pyramid`
- standardize benchmark capture format with commit SHA and metric snapshot

### 2. Profile readbacks before changing scheduling

- identify the dominant GPU/CPU synchronization points
- do not assume the highest coarse phase bucket is the real blocking sync

### 3. Reduce extraction/readback cost before solver scheduling

Safest likely win:

- keep physics state GPU-resident
- extract only compact render transforms for the viewer hot path
- keep exact CPU-visible sync for diagnostics and tests

### 4. Reintroduce counted / indirect paths only behind verification tooling

Before enabling a GPU-resident count or indirect dispatch path by default, add:

- exact count cross-check mode
- overflow detection
- per-frame mismatch logging
- easy fallback to the existing safe path

### 5. Optimize narrowphase math and memory traffic

If narrowphase remains hot after sync cleanup:

- reduce repeated shape fetches
- specialize common shape-pair paths
- tighten hot buffer layouts
- prune unnecessary work on resting contacts

### 6. Revisit GPU-owned solve scheduling last

GPU-owned solve ranges and indirect primal dispatch are high-risk changes.

Only revisit them after:

- color-group equivalence can be checked
- warmstart behavior can be compared frame-to-frame
- contact count and active-body set mismatches are easy to detect

## Known Lessons From This Research Pass

- Removing synchronization and changing solver scheduling in one pass made regressions hard to localize.
- A benchmark improvement in one phase is not meaningful if total frame time regresses.
- GPU-resident scheduling is plausible, but it needs equivalence tooling first.
- Some stability tests are currently useful for investigation but not reliable as hard gates.

## Minimum Required Checks For Any Performance PR

```bash
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json

cargo check -p rubble3d -p rubble-wasm
cargo test -p rubble3d --test wgsl_validation_tests --test gpu_render_path_tests
npm --prefix web test -- tests/ground-contact.spec.ts --reporter=list
npm --prefix web test -- tests/stability-1step.spec.ts --reporter=list
npm --prefix web run bench:stress
```

Optional but recommended investigative follow-ups:

```bash
npm --prefix web test -- tests/stability-stack.spec.ts --reporter=list
npm --prefix web test -- tests/stability-pyramid.spec.ts --reporter=list
```
