# Handoff: soft-body vehicles in the rigid World

Written 2026-11-07, for whoever picks up the soft-body vehicle and World work after PR #105 (`feat/soft-rigid-world`). It covers what was built, what we learned (much of it the hard way), how to run and measure things, and where the open problems are. Known bugs are in [known-bugs.md](known-bugs.md). The agent skills that go with this are listed at the end.

## 1. What exists

Rubble has one GPU `World` that runs two very different solvers together:

| | Rigid bodies | Soft-body vehicles |
|---|---|---|
| Solver | AVBD (augmented vertex block descent), GPU | Explicit node–beam (JBeam/BeamNG semantics), GPU |
| Step | World step: 1/60 s (or 1/120 s), a few iterations | 2 kHz substeps inside each World step (33 per 1/60 s) |
| Scale | 100k+ bodies (the 110,441-brick ring) | 1 to a few dozen cars, a few hundred nodes and a few thousand beams each |
| Destruction | Prefractured threshold destructibles, stress-solved, forward-only | Plastic beam deformation (deform), breaking (strength), break and deform groups |
| Source | `engine/slang/` (shared Slang → WGSL/MSL/CUDA) | `engine/slang/kernels/explicit_beams*.slang`, `vehicle_coupling.slang` |

The pieces, by crate:
- `engine-softbody`: the CPU reference of the explicit solver (`explicit_beams.rs` and friends), the JBeam importer (`jbeam.rs`, `jbeam_fixture.rs`), `.truck` (Rigs of Rods) import and export, flexbody and skin visuals, the procedural and authored cars.
- `engine-webgpu`: the explicit solver on the GPU (`ExplicitBeamsGpu`), the build that turns Slang into WGSL, the demo pages (`*.html`), and the recorders.
- `engine-exec`: the shared explicit plan (`explicit_plan.rs`: Params layout, the module table that Metal and CUDA build).
- `engine-world`: `World`, the coupling (`explicit_vehicles.rs`), the scenes (`explicit_scene.rs` crash town, `brick_ring_drive.rs`, `showcase_scene.rs`).
- `engine-world-wasm`: the wasm host (`explicit_world.rs` town, `ring_drive.rs`, `FramePacer`).
- `engine-metal`, `engine-cuda`: native back ends built from the same Slang.

**Read these docs, in order:**
1. `AGENTS.md`, then `CONTRIBUTING.md`.
2. `engine/docs/deformable-vehicles.md`: the plain-language model of a Rubble vehicle (a living doc; start here).
3. `engine/docs/explicit-beams.md`: the solver's specification. It is clean-room: written from public docs, never from Rigs of Rods or BeamNG code.
4. `engine/docs/jbeam-semantics-spec.md`: how JBeam fields map to the solver, with citations to BeamNG's public documentation.
5. `engine/docs/explicit-vehicles-in-world.md`: the coupling design.
6. `engine/docs/vehicle-test-strategy.md` and `engine/docs/verification-plan.md`: how we know it's right, and what isn't proven yet.
7. `engine/docs/brick-ring-drive.md`: the flagship demo, its measurements and its tools.
8. `engine/docs/measurements/explicit-2khz.md`: where the vehicle GPU time goes, and what was optimised.

## 2. How the coupling works (and why it behaves the way it does)

Each World step:
1. The World broadphase finds rigid bodies and chunks near each vehicle. They become **proxies**: a frozen copy of each body's shape, pose, velocity and mass, uploaded for the vehicle frame (`cpl_bounds`, `cpl_query`, `cpl_plan` and the other `cpl_*` passes).
2. The vehicle runs its 33 substeps at 2 kHz. Its nodes collide with the proxies, and **each proxy integrates as a free body during the frame**.
3. The impulses the vehicle gave each proxy go back to the rigid solver and are applied in the **next** rigid step.

Consequences, all measured (`fix/etk-strength` probes, verification-plan §0.1):
- **Momentum closes at the coupling:** CPU box residual 0.004 N·s, GPU World 0.0.
- **An immovable proxy matches the solver's own static wall.** Momentum agrees within about 5 % (0.8 % without raycast wheels), and a 1 m drop gives 697–720 kN peak force on every path. The bit-for-bit version of this test passes only for Rigs of Rods-profile cars (known-bugs F1).
- **Inside one World window, proxies don't touch each other.** The bricks in a wall don't push on each other during the car's 33 substeps, only at the next rigid step. Against a loose 25 kg brick wall the result converged across 60, 120 and 240 Hz World rates, so window length is not the driver at those rates.
- **Only the car's nodes collide with rigid shapes, unless surface contact is on.** Triangle-against-shape contact exists, opt-in. Node-only contact lets a brick corner slip between nodes (known-bugs B1).
- **Raycast wheels ignore rigid bodies** (known-bugs F7).
- **Destruction stays forward-only.** A vehicle's strike on a prefractured house goes into the threshold fracture's ingest (a second substep slot). Bonds break and the house splits before the rigid step of that window. Nothing is re-simulated, as `AGENTS.md` requires.

## 3. Performance: what costs what

M3 Max; Chrome numbers are the page's live loop unless marked.

| Case | Result |
|---|---|
| Ring alone (110,441 bricks), Chrome | 10.6 ms a step; 4.8 ms with sleep on |
| Ring + authored sedan, Chrome | 60 steps/s, 4.2 ms GPU (rigid 1.4, car 2.8); up to 10.7 ms pressed into the wall |
| Ring + ETK, Chrome, after the 2 kHz work | 60 steps/s, 9.5–10.6 ms GPU (car 7.8–8.0) |
| Two ETKs touching (one contact group), Chrome | 26–57 steps/s: the group runs on one workgroup, 21.3 ms a step |
| ETK 2 kHz frame alone, native | 14.3 → 8.5 ms (−40 %) after `perf/explicit-2khz`, bit-identical |
| Rigid-only scenes vs `main` | WGSL byte-identical, same GPU calls per step, timing within noise (§6) |

Where the vehicle time goes:
- **Each vehicle runs on one workgroup,** so one car is latency-bound on one GPU core. Big cars use "wide" passes across workgroups, but those can't run the coupling.
- **Touching cars merge into one contact group,** solved on one workgroup. Splitting a group across workgroups and exchanging contacts would be lossless; it is the next speed-up.
- **The ETK's frame was about two-thirds self-collision** (grid walk and hit tests). The town car's is about three-quarters coupling. Neither is beam forces.
- **Workgroup memory is full:** 29.9–31.5 KB of Metal's 32 KB. Moving node state into workgroup memory is not possible.
- **Each `sync()` guards a real dependency.** There are no spare barriers to remove.

## 4. Lessons, in rough order of how much time they cost

### GPU runaways (three reboots in one day)
- **Metal can't abort a running kernel.** Killing the process doesn't stop it; the GPU stays wedged until reboot.
- **Causes seen:**
  - garbage coupling headers: `cplRebase` read the header before it was written (fix `cd371f29`, which sat unmerged on `feat/vehicle-sleep` for days);
  - nested loops capped at 2^20 each;
  - loops whose bounds came from corrupt state.
- **Prevention now in the code:**
  - every kernel loop takes a per-frame budget (`explicit_loop_cap.slang`, `ebBudget`/`ebSpent`) or a real maximum. Loops containing barriers use uniform-only caps (`ebEndMost`);
  - the host sizes the budget per frame (2^22–2^31), because the ETK's serial self-collision fallback legitimately needs about 6M passes;
  - a loop stop sets a status word, and `World::step` fails with an error;
  - a static census test (`explicit_loop_cap`) fails CI on an uncapped loop.
- **Process rules:**
  - run the CPU reference first, then a tiny GPU run with a low budget, then the real run;
  - one GPU job on the machine at a time, through `~/.rubble/gpu.lock`. Native binaries take it via `hold_gpu_lock()`; wrap anything else with `engine/scripts/with-gpu-lock.py`. Set `RUBBLE_GPU_LOCK=0` in the child to avoid deadlocking on your own wrapper.

### Chrome / WebGPU is stricter than native wgpu
- **Barriers need uniform control flow (Tint).** A branch on a per-lane value before `workgroupBarrier()` fails to compile. Broadcast through workgroup memory with `workgroupUniformLoad`. `uniformize_workgroup_lanes` in `engine-webgpu/src/lib.rs` rewrites the `gsUniform*` names; add new names to its `NAMES`. Check pages with `softbody-check.mjs`.
- **At most 10 storage buffers per shader stage.** The coupling broadphase needed its own bind group layout (`bgl_q`/`pl_q`, bindings 0, 1, 2, 8, 9, 13–16).
- **Compiling a large pipeline synchronously can trip the GPU watchdog** ("device lost"). Warm pipelines with `createComputePipelineAsync` first (`warm_pipelines`).
- **`std::time::Instant` panics on wasm32.** Use `web_time::Instant`.
- **Tint's bounds-checked Metal output can be slow or wrong:**
  - a per-node shape-range lookup compiled to 755 ms a frame, so `set_coupling_node_ranges` (bit 64) is off on wasm. It is lossless;
  - a running min/max loop miscompiled at n = 16, fixed by building the box in its own pass (`memberBox`).
- **New kernel features go in separate functions behind an early return** (Metal fast-math codegen changes otherwise). Run Metal `vehicle_parity` after kernel edits.
- **A hidden or background tab throttles to a few frames a second.** See the `measuring-web-physics-perf` skill.

### Determinism and measurement
- **Native wgpu is not bit-reproducible run to run** with sleep on, and Chrome runs aren't either. The sedan broke 160–185 beams across four identical runs. To A/B a kernel change, run old and new kernels on the same state (`explicit_frame_dump`), interleaved.
- **Interleave A/B runs** (base, new, base, new…), one GPU job at a time, on a quiet GPU. A single back-to-back comparison misleads.
- **Frame-locked recorders show what the physics does, not real-time speed.** Report the page's live steps per second separately.
- **"Impact speed" depends on which frame you sample.** The drive page and the stills tool differ by about 6 km/h.

### Physics judgement
- **"It looks wrong" is a hypothesis.** We called the ETK "too strong" (15 cm of crush at 113 km/h through bricks). Measurement showed the coupling is correct, the importer matches an independent parse of all 2,767 beams, and loose 25 kg bricks peak at about 100 kN, under the ETK front's yield of about 600 kN. The fix would have been wrong. See the `physics-evidence-first` skill.
- **BeamNG and Rigs of Rods are the same family, not the same solver.** JBeam content is tuned to BeamNG's closed solver. Converted to `.truck`, a JBeam car broke at rest and blew up in Rigs of Rods' own solver. Rigs of Rods (unmodified, on Rigs of Rods trucks) is our only black-box ground truth. For BeamNG we have only its public documentation, unless someone records BeamNG itself.
- **Compared with NHTSA small sedans** (rigid wall, 56 km/h; a sanity check only): ETK 0.55 m dynamic crush (band 0.62–0.75), our sedan 0.81 m. The ETK's four front subframe mounts stay elastic. That's in its data.
- **AVBD for vehicles was tried and rejected** (`feat/avbd-vehicles` 42957615):
  - cheap settings sag like jello (55–60 mm at rest);
  - rigid-looking settings cost 4–9× the explicit solver;
  - crashes fail: no rebound, too many breaks, poles pass through.

  The bar was "a human eye can't tell," and it missed.
- **The rigid AVBD solver loses about 5.5 % of momentum** in a multi-body impact (a block into floating bricks, no car). That's the rigid solver, not the coupling, and it's on `main` too.

### Process
- **Never `git stash`:** the stash is shared across worktrees. Work in your own worktree and stage explicit files.
- **`main` is not rustfmt-clean.** Never `cargo fmt -p`; format only the files you edit.
- **Never `pgrep -f`/`pkill -f`.** Background the work and wait on its PID, with a kill timer.
- **Restarts wipe `/tmp` and scratchpads.** Keep videos and frames in `~/Development/rubble-artifacts/<topic>/` or a worktree's gitignored `out/`, and commit scripts.
- **Long-lived feature branches drift:** fixes sit on one branch while others hit the same bug, as with the rebase hang. Merge often; this PR consolidates about 15 branches.

## 5. How to run things

Pages (serve `engine/crates/engine-webgpu` after building the wasm; see `CONTRIBUTING.md`):

| Page | What |
|---|---|
| `brick-ring-drive.html` | Soft car through the 110k ring. `?car=etk\|sedan`, `?n=2`, `?auto=1`, `?drag=0`, `?sleep=0`, `?timing=phases`, `?inspect=auto` (orbit the crashed car with nearby bricks ghosted; `I` to toggle) |
| `one-world.html` | House plus the ETK |
| `explicit-world.html?n=1\|4\|16` | The crash town |
| `crash-orbit.html?snap=a.json[,b.json]` | 360° orbit of a saved post-crash car state |
| `softbody.html?mod=`, `showcase-car.html` | Vehicle labs |
| `demo.html?scene=ring110k` | The rigid baseline |

**BeamNG cars (the ETK) are local only:** the community cache writes them to `engine/crates/engine-webgpu/__mods/`, which is gitignored (see CONTRIBUTING.md). Without it, pages fall back to the sedan or show a "local only" message.

Tools:
- `engine/scripts/with-gpu-lock.py`: GPU lock, process group and kill timer.
- `engine/tools/brick-ring-drive/`: `ab.sh`, `chrome-ab.sh`, `native-wall.sh`, `ring-alone-chrome.mjs`.
- `engine/crates/engine-webgpu/brick-ring-drive-record.mjs <runs> [--inspect]`: frame-locked demo videos.
- `engine/crates/engine-webgpu/crash-orbit.mjs capture|measure|record|stills`: post-crash snapshots, crush measurement (`crash-measure.mjs`) and orbits.
- `engine/tools/explicit-2khz/`: vehicle GPU profiling.
- Probes on `fix/etk-strength`:
  - `engine-softbody/examples/crush_probe.rs`: rigid-wall crush on the CPU reference;
  - `engine-softbody/examples/coupling_probe.rs`: native shape vs coupled rigid body, CPU;
  - `engine-world/examples/coupling_wall_gpu.rs`: the ETK in a small GPU World.
- `engine/crates/engine-world/examples/brick_ring_drive_probe.rs`: native ring drive, which stops at the first NaN or loop-cap error.
- `engine/crates/engine-world/examples/rigid_gate.rs`: the rigid-only regression gate.

Artifacts (not in git): `~/Development/rubble-artifacts/`, under `brick-ring-drive/`, `crash-orbit/`, `post-crash/`, `etk-strength/`, `explicit-2khz/`, `integration/`, `showcase-etk/`.

## 6. The rigid-only guarantee (and how to keep it)

The rule: adding vehicles must not cost a rigid-only scene anything. The gate run for PR #105 found:
- all 335 non-vehicle WGSL modules byte-identical to `main`;
- 11 scenes bit-identical at 12 checkpoints with sleep off;
- identical submits, passes, dispatches, copies, `writeBuffer`s and `mapAsync`s per step;
- interleaved timing within noise.

Re-run it whenever you touch shared code (`World::step`, the broadphase, shared Slang); see the `rigid-regression-gate` skill.

## 7. Decisions the user has made (don't relitigate)

- **The explicit 2 kHz solver is the vehicle solver.** AVBD vehicles are rejected for now.
- **Vehicles must not regress rigid scale or performance.** The vehicle side gets faster; the rigid side is never traded.
- **No lossy changes on our own initiative:**
  - no decimated shapes;
  - no dropped contacts or iterations;
  - no coarser steps;
  - no looser tolerances.

  Limits are explicit, engine-wide settings that fail loudly (`AGENTS.md`).
- **Destruction is forward-only:** no rewind or correction pass in production.
- **Clean-room:** never read or copy BeamNG or Rigs of Rods code. Specs come from public docs, and Rigs of Rods is a black box only.
- **Community mod content is never committed and never executed** (only its data files are read).
- **Physical accuracy and mathematical soundness come first;** visual judgement comes last. If the maths is right and the look is wrong, fix authoring or conversion, not the solver.
- **Vehicle sleep is an explicit engine-wide setting.** Triangle-against-shape surface contact is opt-in.
- **Credit the GPU stress solver as vibe-land's.** Never mention PhysX, NVIDIA or "fork" in PRs or docs.
- **A PR needs the user's go-ahead.**

## 8. Where to go next

1. **Fix the solids-overlap bug** (known-bugs B1) once `diag/ring-phase-through` reports its root cause and reproduction.
2. **Drive feel** (known-bugs B2): the sedan's drag cap and jitter. Decide whether soft cars should ride the smooth rigid raycast-vehicle model from `vehicle.html`.
3. **Work through the verification plan's priority list** (verification-plan §4). Turn the probes into asserting tests.
4. **Split contact groups across workgroups:** two touching cars in real time.
5. **Coupling contact rule for BeamNG-profile cars, and wheels that see rigid bodies:** F1 and F7, both decisions for the user.
6. **Metal and CUDA module tables:** add the speed-up and surface variants (known-bugs P1).
