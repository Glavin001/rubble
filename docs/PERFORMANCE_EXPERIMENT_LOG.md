# Performance Experiment Log

## Current Baseline

- Current checked-out baseline: `c9d40d7`
- Commit: `physics3d: move color solve scheduling onto GPU`
- Status: kept

## Kept Changes

### 1. Compound substeps and packed web transform path

Status:
- Kept in an earlier commit before the current baseline

Target areas:
- correctness of compound stepping
- web extraction / render-transform handoff
- contact continuity / friction continuity

What changed:
- Fixed compound scenes so `sub_steps` actually execute the full frame instead of only one reduced-`dt` contact step.
- Kept compound scenes on the packed render-transform path instead of falling back to the slower generic step path.
- Preserved tangent continuity on exact warmstart matches.
- Simplified the web-side transform update path.

Why it mattered:
- Fixed a real correctness bug in compound simulation.
- Improved web transform behavior.
- Improved contact/friction continuity.

Observed result:
- User-observed stability and web extraction looked better.
- This was primarily a correctness/stability win, not the major stress-test speed win.

Validation:
- `cargo check -p rubble3d -p rubble-wasm`
- `cargo test -p rubble3d --test wgsl_validation_tests --test gpu_render_path_tests`
- `cargo test -p rubble3d test_compound_substeps_execute_full_frame -- --nocapture`
- `npm --prefix web test -- tests/ground-contact.spec.ts --reporter=list`
- `npm --prefix web test -- tests/stability-1step.spec.ts --reporter=list`
- `npm --prefix web run bench:stress`

### 2. Move color solve scheduling fully onto the GPU

Status:
- Kept
- Commit: `c9d40d7`

Target areas:
- solve
- CPU-side scheduling overhead around solve
- web submission overhead during the colored AVBD pass

What changed:
- Removed per-step CPU readback of sorted body colors.
- Removed CPU rebuild of solve ranges.
- Built solve runs and indirect dispatch args on GPU.
- Drove the primal pass with `dispatch_workgroups_indirect`.

Why it mattered:
- This was the first major solve-side structural win inspired by `webphysics`.
- It removed CPU scheduling overhead from the body-colored AVBD loop.

Benchmark result on the stress scene:
- `step_ms_median`: `156.05 -> 134.10`
- `solve_ms_median`: `71.65 -> 43.95`
- `narrowphase_ms_median`: `45.75 -> 38.40`
- `broadphase_ms_median`: `0.60 -> 0.50`
- `extract_ms_median`: `32.25 -> 48.35`

Interpretation:
- This was a real engine win.
- Solve got much faster.
- Extract became relatively more visible afterward.

Validation:
- `cargo check -p rubble3d -p rubble-wasm`
- `cargo test -p rubble3d --test wgsl_validation_tests --test gpu_render_path_tests`
- `npm --prefix web test -- tests/ground-contact.spec.ts --reporter=list`
- `npm --prefix web test -- tests/stability-1step.spec.ts --reporter=list`
- `npm --prefix web run bench:stress`

## Reverted Experiments

### 1. Extract kernel small-angle angular velocity recovery

Status:
- Reverted

Target areas:
- extract
- velocity finalization correctness

Intent:
- Match the `webphysics` velocity-finalization idea more closely.

What changed:
- Adjusted angular-velocity extraction to use a small-angle-style quaternion delta path.

Result:
- Regressed the stress benchmark.

Observed benchmark:
- `step_ms_median`: about `137.35`
- `solve_ms_median`: about `47.80`
- `narrowphase_ms_median`: about `38.20`
- `extract_ms_median`: about `50.65`

Reason reverted:
- Worse than the current baseline.

### 2. Plane-box manifold reduction to 4 contacts

Status:
- Reverted

Target areas:
- narrowphase
- solve, indirectly through fewer contacts

Intent:
- Reduce solver load by forcing box-plane contact manifolds down to 4 points.

What changed:
- Reduced penetrating box-plane vertices to a smaller manifold.

Result:
- Early samples looked promising, but the full benchmark did not improve enough.

Observed benchmark:
- `step_ms_median`: about `138.85`
- `solve_ms_median`: about `47.70`
- `narrowphase_ms_median`: about `38.75`
- `extract_ms_median`: about `50.85`

Reason reverted:
- Not a meaningful win.

### 3. Local-diagonal / small-angle primal solve fast path

Status:
- Reverted

Target areas:
- solve

Intent:
- Port the `webphysics` local-diagonal primal idea.

What changed:
- Simplified parts of the 6x6 primal solve path around local-diagonal treatment.

Result:
- Narrowphase improved slightly, but solve regressed enough that total step time did not improve.

Observed benchmark:
- `step_ms_median`: about `137.35`
- `solve_ms_median`: about `48.20`
- `narrowphase_ms_median`: about `37.00`
- `extract_ms_median`: about `50.80`

Reason reverted:
- No net improvement.

### 4. Fixed-capacity per-body contact refs replacing the prefix-scan adjacency build

Status:
- Reverted

Target areas:
- solve graph build
- solve
- web submission overhead during graph construction

Intent:
- Remove the prefix-scan adjacency build and replace it with fixed-capacity direct per-body contact references.

What changed:
- Rewrote adjacency construction around fixed-capacity per-body contact slots.

Result:
- Mixed outcome.
- Solve looked lower in many samples, but narrowphase variability increased and total step time got worse.

Observed benchmark:
- `step_ms_median`: about `150.00`
- `solve_ms_median`: about `38.30`
- `narrowphase_ms_median`: about `46.10`
- `extract_ms_median`: about `62.90`

Reason reverted:
- It moved cost around but did not improve the real outcome.

### 5. Friction-row gating in primal/dual

Status:
- Reverted

Target areas:
- solve
- friction row cost inside primal/dual

Intent:
- Copy the `webphysics` policy of skipping tangential rows unless the normal row has real support.

What changed:
- Gated friction row work based on normal support and separation tolerance.

Result:
- Broke basic contact behavior.

Observed failure:
- `web/tests/ground-contact.spec.ts` failed.

Reason reverted:
- Any optimization that breaks basic support contact is invalid.

### 6. Persistent bind-group caching for adjacency/coloring/solve-schedule setup

Status:
- Reverted

Target areas:
- solve graph build
- solve setup
- WebGPU submission overhead

Intent:
- Reduce WebGPU submission overhead by caching solve-setup bind groups instead of recreating them every frame and every coloring sweep.

What changed:
- Added bind-group caches for:
  - adjacency reset/count/init/scatter
  - coloring reset/step
  - solve schedule reset/build

Stress benchmark result:
- `step_ms_median`: `134.10 -> 113.90`
- `narrowphase_ms_median`: `38.40 -> 19.20`
- `solve_ms_median`: `43.95 -> 47.10`
- `extract_ms_median`: `48.35 -> 46.00`
- `frame_step_copy_ms_median`: `6997.9 -> 3988.0`

Why it looked promising:
- Large overall benchmark improvement.

Why it was still reverted:
- It failed stacked stability.
- `web/tests/stability-pyramid.spec.ts` exploded badly:
  - step 200: `maxY=93.36`, `maxAbsX=115.11`
  - step 400: `comY=-16.16`, `maxAbsX=249.22`
  - step 600: `comY=-90.38`, `minY=-352.84`, `maxAbsX=383.34`

Important note:
- The native `demo_pyramid_stays_supported_by_floor_without_exploding_3d` test is currently marked as a known failure and skips by default, so the web pyramid test was the actual guardrail here.

Reason reverted:
- Faster but unstable is not acceptable.

## What We Learned

### What has clearly worked

- GPU-side solve scheduling was a major real win.
- Compound substep correctness and packed web transform handling were good changes.
- `webphysics` is valuable mainly for:
  - GPU-driven solve orchestration
  - persistent state handling
  - friction/contact policy ideas
  - submission-discipline ideas

### What has clearly not transferred cleanly

- Not every `webphysics` solver trick wins in this engine.
- Some optimizations that improve headline benchmark numbers can still destabilize stacked scenes.
- The web pyramid regression is mandatory for keeping solve-path changes.

## Timing Confidence And Interpretation

### Why one code change can move multiple phase timings

- The phases are not independent in practice.
- A narrowphase change can alter contact count and manifold quality, which changes solve cost.
- A solve-scheduling change can change queue submission behavior and synchronization points, which can shift measured phase boundaries.
- An unstable optimization can inject energy, which then inflates later narrowphase and solve work because the scene behavior itself changes.
- Some changes affect CPU overhead around dispatch, which the browser path can expose differently than native.

So yes, it is possible and normal for one code change to move more than one reported timing bucket.

### How much of the timing movement was likely noise

- Single sampled steps are noisy.
- Short runs or “it looked better for a few steps” are not enough to trust.
- The browser stress benchmark uses 200 measured steps after warmup, so median shifts from that run are much more meaningful than one-off step logs.
- I trust large median moves much more than small ones.

Practical confidence levels used during these experiments:

- Very high confidence:
  changes like `solve_ms_median 71.65 -> 43.95` or `step_ms_median 134.10 -> 113.90`
- Moderate confidence:
  changes in the mid single digits where the median moved consistently and scene behavior stayed stable
- Low confidence:
  changes of only a couple of milliseconds, or cases where early samples looked good but full medians did not hold

### Which reversions were mostly noise-vs-signal calls

- `extract kernel small-angle angular velocity recovery`
  low-to-moderate confidence on the exact size, but enough confidence that it was not a win
- `plane-box manifold reduction to 4 contacts`
  moderate confidence that it was not materially better
- `local-diagonal / small-angle primal fast path`
  moderate confidence that it was not a net win
- `fixed-capacity per-body contact refs`
  high confidence that the total result was worse

### Which reversions were correctness blockers, not noise calls

- `friction-row gating in primal/dual`
  high confidence, because ground-contact failed
- `persistent bind-group caching`
  high confidence that the implementation as written was not safe, because the web pyramid regression exploded dramatically

## Could A Reverted Fast Change Still Be Salvaged?

Short answer:
- yes, possibly
- but only when the speedup looked real and the failure looked implementation-specific rather than conceptually wrong

### Most salvageable experiment

- `persistent bind-group caching for adjacency/coloring/solve-schedule setup`

Why:
- The speedup was large enough to be real.
- The intended optimization is conceptually sound.
- It should not, in theory, change physics behavior.

What likely happened:
- The cached binding implementation probably failed to track some resource or state dependency correctly.
- That means the bug was likely in caching invalidation or in what was captured by the cache key, not in the optimization idea itself.

So this one is a good candidate to revisit carefully.

### Possibly salvageable, but less convincing

- `fixed-capacity per-body contact refs replacing the prefix-scan adjacency build`

Why:
- The idea is structurally plausible.
- But the measured outcome was mixed and the tradeoffs were worse even before considering deeper validation.

This could still be worth revisiting later, but it did not show the same quality of upside.

### Less likely to be worth another pass soon

- `friction-row gating in primal/dual`
- `local-diagonal / small-angle primal fast path`
- `plane-box manifold reduction to 4 contacts`
- `extract kernel small-angle angular velocity recovery`

Why:
- Those either broke correctness quickly or did not show enough upside to justify immediate reinvestment.
- They might still be useful later in a more complete reformulation, but they did not look like “almost there” changes.

## Minimum Regression Set For Future Performance Work

Before keeping any new solve or narrowphase optimization, run at least:

- `cargo test -p rubble3d --test wgsl_validation_tests --test gpu_render_path_tests`
- `npm --prefix web test -- tests/ground-contact.spec.ts --reporter=list`
- `npm --prefix web test -- tests/stability-1step.spec.ts --reporter=list`
- `npm --prefix web test -- tests/stability-pyramid.spec.ts --reporter=list`
- `npm --prefix web run bench:stress`

## Next Likely Direction

The next serious optimization pass should target real solve work, not just setup churn:

- reduce per-contact work in primal/dual
- reduce unnecessary contacts entering solve
- improve persistent manifold quality without destabilizing stacks

Any future attempt that improves stress numbers but fails pyramid stability should be rejected immediately.
