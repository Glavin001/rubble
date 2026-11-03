# Blast GPU port: which PhysX vibe-land runs, and the plan to replace it

Recorded 2026-11-02. This is the reference for *what rubble's `blast_gpu` profile must match*,
how that was established, and the plan to finish the port and prove it. Results go in
[`blast-gpu-port.md`](blast-gpu-port.md).

## The target, unambiguously

vibe-land's destruction runtime is **PhysX's native, directly integrated GPU destruction stage
and its CUDA stress solver**, running inside `PxScene::simulate()` on CuMetal (CUDA on Metal) on
this Mac:

| | Path in `Glavin001/PhysX` |
|---|---|
| Stage | `physx/source/gpudestruction/src/` (`PxgDestructionRuntime.cu`, `PxgDestructionMaterial.cuh`, `PxgDestructionTopology.cu`, `PxgDestructionMotionState.cuh`, `PxgDestructionCorrection.cuh`) |
| Solver | `blast/source/sdk/extensions/stressgpu/` (`NvBlastExtStressGpu.cu`, `detail/StressComponentIteration.cuh`, `StressIterationDispatch.inl`, `StressNativePolynomial.cuh`, `StressNativeRigidInverse.cuh`, `StressMotionPair*.cuh`) |

**It is not PhysX's newer `destruction/stress-oracle/` solver** (crates `stress-ref`, `stress-gpu`,
`stress-physx`, `stress-reduce`, `stress-viz`). That one exists only on
`integration/oracle-stress-gpu`, is not on `clean/high-fidelity`, and vibe-land references it
nowhere (`feat/unreal-client`: no Cargo dependency or path in `physx-bridge`, `server`, `shared`
or `destruction/src`). The engine folders above do not use it either.

Naming trap: rubble's own `engine/stress-oracle` (`stress-ref`, `stress-gpu`) descends from that
PhysX oracle workspace. The Blast port lives *inside* rubble's copies of those crates
(`stress-ref/src/blast_bond*`, `stress-gpu/shaders/blast_*.slang`), but it ports the CUDA native
stage and solver above, not the oracle model.

### The exact build vibe-land runs (`feat/unreal-client`, head `344c2d97`, 2026-10-22)

| | Value |
|---|---|
| SDK install | `~/Development/PhysX/out/install/garage-multihull` |
| PhysX commit | **`a08abef50`** (2026-10-07 15:00, "Bearing joints: the fasteners carry the signed normal force"), built clean, feature profile `rigid-demo-experimental`, destruction API v24 |
| Branch line | `clean/high-fidelity` (head `612c63d97`, 2026-10-10) |
| cuda-metal | branch `codex/physx-metal-integration` per vibe-land's README |
| Server launch | `scripts/perf/garage-vehicle-server.sh` (builds `web-fps-server --features native-destruction` into `target/garage-vehicles`, `PHYSX_ROOT=garage-multihull`) |
| Who uses it | the Unreal demo (`scripts/unreal/demo.sh`), single-player `-VibeLocal` (`scripts/native-mac.sh`), the city server, vehicle QA |

Evidence:
- `scripts/unreal/demo.sh`: "play-server.sh links ../PhysX/out/install/macos-cumetal, which
  predates this branch's bridge (destruction scene v18); the garage SDK (v24) is the one the
  native app and this server use."
- `scripts/native-mac.sh`: `PHYSX_ROOT=${PHYSX_ROOT:-$SIBLINGS/PhysX/out/install/garage-multihull}`.
- The newest server binaries (`target/garage-vehicles`, `server-debt`, `merge-verify`,
  2026-10-16/17) all carry `LC_RPATH .../PhysX/out/install/garage-multihull/lib`.
- `garage-multihull/sdk-artifacts.json`: `source_revision a08abef50e…`, `source_dirty false`.

### Runtime settings (`garage-vehicle-server.sh`, `destruction/src/native_runtime.rs`)

| Setting | Value |
|---|---|
| Stress iterations | 16 (`VIBE_CITY_NATIVE_STRESS_ITERATIONS`; 64 for the vehicle fleet / garage) |
| Tolerance | 1e-3 (`VIBE_CITY_NATIVE_STRESS_TOLERANCE`) |
| Force tolerance | 1e-3 (`VIBE_NATIVE_STRESS_FORCE_TOLERANCE`; the code default is 0, the server sets 1e-3) |
| Damage rate, bend gain cap | 2, 3 |
| Warm start | on |
| `BLAST_STRESS_INCREMENTAL_MOTION`, `PX_DESTRUCTION_INCREMENTAL_TOPOLOGY` | 1 (speed: touched components only) |
| `BLAST_STRESS_BALANCED_OPERATOR` | 1 (speed: operator passes split by bonds) |
| `PX_DESTRUCTION_ALLOW_UNCONVERGED` | 1 (unconverged steps are published) |
| Section bending / rotation, crush, Mohr–Coulomb, re-bearing, shear stiffness | off (unset) |

### Other PhysX builds in play (not the target)

| Use | Install | Commit | Note |
|---|---|---|---|
| `play-server.sh` default (bridge default with no `PHYSX_ROOT`) | `out/install/macos-cumetal/release` | `696c341f` (2026-09-25), `claude/perf-integration` | API v18, a separate older line; stale for `feat/unreal-client` |
| Fidelity `runtime` profile (`scripts/fidelity/runtime.env`) | `garage-roof@0e0aba8a0.2` | `0e0aba8a0` (2026-10-06) | verification profile, every new capability off, no force tolerance |
| Fidelity `high` profile (`scripts/fidelity/high.env`) | `garage-clean@612c63d97` | `612c63d97` (2026-10-10), `clean/high-fidelity` head | high-fidelity verification; force tolerance 1e-3 |
| Stress-problem capture | `garage-multihull-stress-problem` | `8bbfaac` (2026-10-07 00:55, dirty) | the build that recorded `vibe-land/target/stress-capture/{m1,c2,c3,f1,f2,rot}` |
| The port so far | none installed | `ec8118341` (2026-10-16), `integration/oracle-stress-gpu` | the commit the port was read from |

How they relate (`git merge-base --is-ancestor`): `8bbfaac`, `0e0aba8a0`, `a08abef50` and
`codex/cumetal-destruction` are all ancestors of `ec8118341`. `612c63d97` and `696c341f` are not.
In the stage and solver folders, `ec8118341` equals `612c63d97` except for fragment wake (9 lines,
on `612c63d97` only; neither vibe-land profile enables it). The rest of
`612c63d97..ec8118341` (341 files) is PhysX's `destruction/stress-oracle` tooling.

## What changed between the target and the commit the port was read from

`a08abef50..ec8118341` in the stage and solver folders (the port read `ec8118341`):

| Commit | Change | Default | For an exact match with `a08abef50` |
|---|---|---|---|
| `a8660b4a8`, `045933819` | Krylov carry | on with rotation only | off |
| `ffc463339` | the residual test also requires `‖r‖ ≤ tol ‖b‖` | **always** | behind a flag, off |
| `3f15dff87` | Mohr–Coulomb joint shear | opt-in | off |
| `069931ff1`, `8eb0163a0`, `a508acf77` | shear stiffness | opt-in | off |
| `aeb0e2faf` and follow-ups | re-bearing | opt-in | off |
| `97e23052d` | corrected pass warm start | opt-in | n/a (no correction pass in rubble) |
| `de4ce93ba` | warp-summed island rows | perf | n/a |

In `a08abef50` but not ported: the balanced operator and incremental motion/topology (speed
only, results unchanged up to rounding).

## Deliberate deviations (forward-only and engine differences)

- **No correction pass / re-simulation** (`PxgDestructionCorrection`): AGENTS.md forbids it.
  Strikes are predicted before the step instead. Break timing therefore differs from PhysX on
  strike ticks; each difference must be explained in the head-to-head.
- **Large components (> 8,192 nodes):** PhysX uses a cooperative kernel with a multilevel
  V-cycle. WebGPU and Metal have no portable grid-wide sync, so rubble uses its many-group
  (`blast_native_wide`) solve. Results must be compared; port the V-cycle as multiple
  dispatches only if they differ beyond rounding.
- **Constraint loads:** not ported (rubble's fracture has no joints on chunks).
- **Fragment wake, resting contact loads:** not in `a08abef50`.
- **Scenes without Blast authoring** (GLB, cuboid houses): rubble derives Blast's inputs from its
  own structures, each bond's strength as both limits (see `blast-gpu-port.md`). vibe-land's
  packs carry their own authoring, so the head-to-head uses packs.

## Status (2026-11-02)

Done, in PR #87 (`worktree-blast-gpu-port`):
- the native solve in Slang (one-group and many-group), rigid modes as PhysX builds them
  (float pairs, spanning forest, cycle-closure rank), rotational and shear stiffness, Krylov
  carry, the stage's material evaluation, crush from Blast forces, load assembly, one bond
  state, zero uploads at rest, a Rust f64/f32 reference, GPU-built events;
- replay of PhysX's recorded solves (`8bbfaac` captures): iterations and convergence match,
  forces within 3.9e-5;
- standing under gravity: house 0 breaks; villa's 2 breaks are the converged model's verdict;
- `blast_gpu` in the fracture lab for packs, GLBs and the cuboid houses.

Open:
- **Chrome fix, uncommitted:** `blast_native` was rejected by Chrome (a barrier under
  non-uniform control flow from the carry count); fixed, all 34 stress shaders compile in
  Chrome, the lab's four stress modes pass in headless Chrome. Also uncommitted: the lab-shot
  test, `stress-gpu/scripts/chrome_wgsl_check.mjs`, `engine-webgpu/fracture/stress-modes-check.mjs`.
- **`main` has moved** (PR #95); merge again.
- **An intermittent lab error** on a fast scene-then-mode switch (an old device's buffer in the
  World renderer); not yet shown to predate this branch.
- **No head-to-head on real scenes**, **no speed comparison**.
- `findings/blast-gpu-port.md` and `reference/blast-gpu-port/provenance.json` still name
  `ec8118341` as the target; re-anchor them to `a08abef50`.

## Plan

1. **Land what is done.** Commit and push the Chrome fix, the lab-shot test and both Chrome
   checks; merge `origin/main`; rerun the native suites (Metal, WGSL at storage limit 10), the
   Chrome WGSL check and the lab check on a fresh `build-web.sh`; confirm the Vercel preview in
   a visible browser pane.
2. **Re-anchor the target.** Point the findings and `provenance.json` at `a08abef50`
   (`garage-multihull`), keeping `ec8118341` as the commit the code was read from.
3. **An `a08abef50` setting in rubble.** `blast_gpu` gets the server's settings above, with
   Krylov carry off and the relative-residual test behind a flag (off). Keep `ec8118341`'s
   behaviour behind flags. Tests from the CUDA definitions: the two convergence rules on one
   system, carry off reproducing the plain recurrence.
4. **Exact forces at `a08abef50`.** Build the stress-problem capture SDK at `a08abef50` with
   vibe-land's `scripts/perf/build-stress-capture-sdk.sh` (installs beside `garage-multihull`;
   the shared install is untouched). Record solves on house-1story, villa-savoye, cannonball,
   meteors and skyline-stable. Replay them in rubble (`stress-gpu/examples/blast_capture.rs`):
   iterations, convergence and per-bond forces within f32 rounding.
5. **Break sets and timing.** Run vibe-land's server-side harness on `garage-multihull` with
   scripted strikes, recording per frame the broken bonds, body count, `cuda.stress` and
   whole-tick time. Run rubble `blast_gpu` on the same packs and strikes. Compare: forces
   within f32 spread; break sets and body counts identical or each difference explained (the
   no-correction-pass deviation chief among them); 0 breaks at rest in both; speed, one engine
   on the GPU at a time under vibe-land's GPU lock (`scripts/perf/gpu-run.sh`), each run under
   10–20 min with a kill timer.
6. **Close differences.** Large components first (the V-cycle question), then anything the
   comparison shows. Record every result and deviation in `blast-gpu-port.md`, with side-by-side
   videos.
7. **Speed.** Only lossless changes, measured against step 5: fewer dispatches in the
   many-group solve (fold the row kernels into the block kernels), the balanced operator, and
   indirect-dispatch validation off once every indirect size is clamped and tested.

Done when every scene's forces and break sets match (or each difference is explained and
accepted), structures stand in both, and rubble's stress stage is no slower than vibe-land
native on this Mac.

## Open questions (need the user)

1. **A measurement harness in vibe-land.** No existing harness writes per-frame broken-bond sets
   for a scripted strike (closest: `physx-bridge/tests/textbook/stage.rs`). Adding one means
   editing vibe-land, which is otherwise off limits. Allowed on a new vibe-land branch in its
   own worktree, or use only existing outputs (captures, solve reports, city-bench reports)?
2. **Building in PhysX.** Step 4 adds an install under `PhysX/out/install/` beside the others.
   Allowed?
3. **Session access.** The rubble session can write only inside its worktree. Steps 4–5 need a
   session started in `~/Development/PhysX` and `~/Development/vibe-land` (or the user runs
   them).
4. **vibe-land's play server** still defaults to the stale `macos-cumetal` (`696c341f`) install.
   Whether to change that is vibe-land's decision.

## Risks

- Chrome-only failures beyond shader compilation (limits on other GPUs); the headless check runs
  this Mac's GPU only.
- The no-correction-pass deviation can make break sets differ on strike ticks by design; the
  acceptance bar for those differences needs the user's agreement.
- Large components may need the V-cycle port if the many-group solve differs beyond rounding.
- GPU PhysX is not bit-reproducible run to run (vibe-land's textbook determinism gap above
  8,192 nodes), so break-set comparisons need a tolerance or repeated runs.
- Headless-Chrome timing can make the lab check flaky (it yields an animation frame per step).
