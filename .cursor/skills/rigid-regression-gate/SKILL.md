---
name: rigid-regression-gate
description: Prove that a change leaves rigid-only scenes (no vehicles) bit-identical and equally fast compared with main. Use before merging anything that touches shared engine code (World::step, broadphase, shared Slang, wasm host, build scripts), and whenever someone asks "does this slow down rigid bodies?".
---

# Rigid-only regression gate

Rubble's promise: adding soft-body vehicles (or any opt-in feature) costs a rigid-only scene nothing. Claim it only with these four checks, run against `origin/main` in a separate worktree (e.g. `out/main-wt`).

1. **Shaders, proven.** Export WGSL on both builds (`wgsl_export`, and the build's `OUT_DIR` modules) and diff. Every module that isn't vehicle-only (`explicit_*`, `vehicle_coupling`, `*loop_cap_test*`) must be byte-identical, and no existing rigid `.slang` file may change.
2. **Results, proven.** Scene end-state hashes: 11 scenes, a ball fired in at step 100, a hash of every body at 12 checkpoints up to step 300, twice per build (`engine-world/examples/rigid_gate.rs`, which uses only `main`'s API).
   - With sleep off they must be bit-identical.
   - With sleep on, native wgpu isn't reproducible run to run, so compare where `main` diverges from itself with where the branch does.
3. **No extra GPU work.** Count WebGPU calls per step in Chrome (`page-bench.mjs` with `GPU_CALLS=1`): submits, passes, dispatches, indirect dispatches, copies, `writeBuffer`, `mapAsync`. They must be identical with zero vehicles. Back it up by reading the diff: every vehicle hook must return early when there are no vehicles.
4. **Timing, measured.** Interleaved A/B (main, branch, main, branch…; at least 3 rounds), one job at a time under the GPU lock, on a quiet GPU.
   - Native: Ring 110k, 100k Grid, Box Pile 32k, Ragdoll Pile 10k, sleep off and on.
   - Chrome: `demo.html?scene=ring110k` with sleep off and on.
   - Report medians and spread. Sleep-on rows move both ways, because which bodies sleep differs per run.

Any difference in 1–3 blocks the merge until it's explained. Write the result up as "proven" (1–3) versus "measured" (4).

Reference result for PR #105: 335 modules byte-identical; 11 scenes bit-identical with sleep off; identical GPU calls per step; timing +0.0 to +0.5 % with sleep off. Tools and logs are in `~/Development/rubble-artifacts/integration/`.
