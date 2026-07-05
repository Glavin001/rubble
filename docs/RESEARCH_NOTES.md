What worked best was a layered process: one stable perf benchmark, one small set of fast correctness gates, and immediate rollback of any optimization that changed physics before the numbers were even “good”.

**Performance measurement**

Use [`web/tests/bench-stress-webgpu.spec.ts`](rubble/web/tests/bench-stress-webgpu.spec.ts) via:

```bash
npm --prefix web run bench:stress
```

That was the best primary metric because it already gives the numbers that matter for this project:

- `step_ms_median`: best top-line target metric
- `step_ms_p95`: catches instability/jitter
- phase breakdowns:
  - `solve_ms`
  - `narrowphase_ms`
  - `extract_ms`
  - `broadphase_ms`
  - `upload_ms`

What mattered in practice:

- prioritize `median` and `p95`, not mean
- keep scene, browser path, warmup, and body count fixed
- compare one code change at a time
- record the exact commit SHA with every run
- treat a phase win as real only if total `step_ms_median` drops too

For viewer work, end-to-end frame timing was also useful, but only as a secondary metric. The best split was:

- physics step
- transform copy/readback
- JS/Three matrix update
- render

That told us quickly whether the bottleneck was actually physics or just presentation.

**Correctness / regression gates**

Best primary gate:
- [`web/tests/ground-contact.spec.ts`](rubble/web/tests/ground-contact.spec.ts)

Fast smoke gate:
- [`web/tests/stability-1step.spec.ts`](rubble/web/tests/stability-1step.spec.ts)

Useful but not currently reliable as hard blockers in this environment:
- [`web/tests/stability-stack.spec.ts`](rubble/web/tests/stability-stack.spec.ts)
- [`web/tests/stability-pyramid.spec.ts`](rubble/web/tests/stability-pyramid.spec.ts)

Important finding: on April 10-11, 2026, `stability-stack.spec.ts` failed in this checkout even with no tracked code changes. So for the write-up, I would explicitly say:

- `ground-contact` is the best current blocker test
- `stability-1step` is the best fast smoke test
- `stack` and `pyramid` are good investigative tests, but need baseline repair before they can be treated as hard regression gates

Recommended commands:

```bash
npm --prefix web test -- tests/ground-contact.spec.ts --reporter=list
npm --prefix web test -- tests/stability-1step.spec.ts --reporter=list
npm --prefix web test -- tests/stability-stack.spec.ts --reporter=list
npm --prefix web test -- tests/stability-pyramid.spec.ts --reporter=list
```

**Rust / shader safety checks**

When touching GPU/WGSL paths, these were the most useful cheap checks:

```bash
cargo test -p rubble3d --test wgsl_validation_tests --test gpu_render_path_tests
cargo check -p rubble3d -p rubble-wasm
```

Relevant files:
- [`crates/rubble3d/tests/wgsl_validation_tests.rs`](rubble/crates/rubble3d/tests/wgsl_validation_tests.rs)
- [`crates/rubble3d/tests/gpu_render_path_tests.rs`](rubble/crates/rubble3d/tests/gpu_render_path_tests.rs)

These caught shader/layout mistakes early, before burning minutes in browser benchmarks.

**Best research process**

The safest process was:

1. Establish clean baseline numbers.
2. Run fast correctness first.
3. Make one optimization change.
4. Re-run fast correctness immediately.
5. Only then run `bench:stress`.
6. If correctness changed, revert first and analyze second.
7. If perf improved in one phase but regressed total frame time, reject it.

**What not to do**

The main failed pattern was trying to remove sync and change GPU scheduling at the same time. That made it hard to tell whether regressions came from:
- count ownership
- indirect dispatch
- warmstart changes
- color scheduling changes

For the write-up, I’d recommend a rule:

- never combine readback removal with solver scheduling changes in one experiment

**What I’d write as the formal gate set**

For every optimization PR/experiment:

1. `cargo check -p rubble3d -p rubble-wasm`
2. `cargo test -p rubble3d --test wgsl_validation_tests --test gpu_render_path_tests`
3. `npm --prefix web test -- tests/ground-contact.spec.ts --reporter=list`
4. `npm --prefix web test -- tests/stability-1step.spec.ts --reporter=list`
5. `npm --prefix web run bench:stress`

Then run `stack`/`pyramid` as non-blocking investigative checks until their baselines are repaired.

If you want, I can turn this into a polished “Performance Research Plan” markdown document for the repo.
