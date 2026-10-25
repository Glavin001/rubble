# Blast's GPU stress solver in Slang (2026-10-25)

**Goal (user):** rubble replaces PhysX+Blast in vibe-land. That means matching vibe-land's stress solver in accuracy, then in speed, then beating it.

**What was ported:** the model vibe-land destroys structures with. That is Blast's `ExtStressSolver` from the PhysX fork, solved on the GPU by `stressgpu/NvBlastExtStressGpu.cu`. It is reimplemented in Slang as a new capability profile, `blast_gpu`.

**The plan:** `~/.claude/plans/in-rubble-we-have-cheeky-feather.md`.

## What exists now

| Piece | Where | What it does |
|---|---|---|
| f64/f32 reference | `stress-ref/src/blast_bond.rs` | The CUDA host path's math, written from its definitions. It covers centred offsets, length and mass scaling, equalised or native mass weighting, and the compliance column scale. Islands are cut at supports, each with its own tolerance, and the solve is a node-space CGLS (numerator \|B^T rho\|^2, each bond once) with a warm start and a settled-island skip. Stresses use the shared formula (section bending capped at 3, fibres) and the damage law (fatal or 2/s). The CGLS is generic: `device_precision` runs it in f32. |
| Blast's pack inputs | `scene_pack::ImportOptions::blast_authoring` → `Scene::blast` | The pack's node centroids, masses and volumes; its bond centroids, normals and areas; and both limit sets of each material. Bonds map one to one onto the scene's bonds. |
| GPU solve | `stress-gpu/shaders/blast_cgls.slang` | One 1024-thread group per island, using fixed-order group sums (no float atomics). |
| GPU stresses and damage | `blast_material.slang` | The formula with fused operations, as the CUDA code uses them. |
| GPU islands | `blast_topology.slang` | Rebuilt from bond health only after a break, gated on the GPU. It uses a lock-free union-find, scans, scatter, and a per-island bitonic sort, so lists are ascending and deterministic. |
| GPU fracture frame | `blast_fracture.slang`, `GpuFracture::set_blast` | Loads go in as vibe-land feeds them: gravity, contact force over node mass, and the centrifugal load, in the structure frame. Breaks go into the fracture's split. |
| Profile | `Profile::BlastGpu` (`blast_gpu`) | `ThresholdWorld::with_profile`; stress-web; the Fracture lab's "Stress solver · Blast" mode; `fracture_glb STRESS_PROFILE=blast_gpu`. |

Nodes and bonds keep their indices: an island is a list of each. A break therefore rewrites lists and keeps every bond's warm start, which is the CUDA host path's incremental default. A frame reads nothing back.

## Accuracy

These results come from `stress-gpu/tests/blast_host_vs_reference.rs` and `stress-rubble/tests/blast_gpu_world.rs`. They pass on Metal, and on WGSL at 10 storage buffers.

- **Converged solves** (tolerance 1e-5): the GPU is within 4e-5 (equalised) and 8e-5 (native) of the f64 reference on house-1story.
- **Capped solves** (vibe-land's 32 iterations a frame) are compared against the f32 reference.
  - The CUDA host path itself runs this recurrence in f32. At iterations 24-32 on house-1story, a one-ulp nudge to the inputs moves the iterate by up to about 3e-3. Any f32 implementation has that spread, CUDA's included, since its float atomics reorder sums from run to run.
  - The GPU stays within about 1.2× that spread of the f32 reference. It is up to 1.4e-2 from the f64 one: at capped counts, f32 CG drifts from f64 CG.
  - `difference_by_iteration`, an ignored diagnostic, shows the drift growing smoothly. It is 1e-7 at iteration 1 and 4e-6 at 16. Then it jumps where CG loses orthogonality in f32, which is not the signature of a logic error.
- **Stresses and damage** are checked by applying the formula to the device's own impulses: stresses agree within 1e-5, and all 877 of 877 damage verdicts agree. Health after damage is exact.
- **Islands** are identical to the reference's after 693 bonds break.
- **In the world:** a house under gravity tracks the f32 reference frame for frame. The difference is 5e-4 at the first frame and falls to 2e-5 over 20 frames.
- **Determinism:**
  - Reruns are bit-identical, both the standalone solve and the struck-house world.
  - Villa-savoye's outcome (206 bonds, 62 bodies) is the same at 256 and at 1024 threads a group.

**Not done yet:**
- The one-off cross-check against Blast's own solver through the PhysX repo's `blast_ladder_bench` JSON.
- vibe-land's break sets for the same events, which need the P0a baseline run.

## Speed (M3 Max, one engine on the GPU at a time)

| Run | Frame (waited for) | Of which |
|---|---|---|
| villa-savoye (1045 chunks, 2464 bonds), 4 shots, `threshold` | about 2.7 ms | rubble 2.27, ingest 0.41, fracture 0.44 |
| the same, `blast_gpu` | 3.58 ms median, p95 3.81 | rubble 2.05, ingest 0.40, fracture with Blast 1.83 |
| Blast solve alone on villa-savoye (`blast_bench`) | 1.76 ms median, submission and wait included | one island, 32 iterations (Blast stalls on the hull packs, as the 2026-10-16 ladder found, so nothing settles) |

**Change made:** one group per island was latency-bound at 256 threads (4.4 ms a solve). At 1024 threads it is 1.8 ms. The recurrence is the same and the result is still repeatable.

**Discarded measurements:** runs earlier the same day read 23-29 ms a frame while other processes loaded the machine.

**vibe-land's numbers are not comparable yet:**
- Its server-alone figures (cannonball p50 7.6-9.0 ms, meteors about 9 ms; `performance-targets-status-2026-09-25.md`) are whole-city ticks.
- Matching them needs the same scenes here. Towns are multi-structure, and the Blast authoring covers one structure so far.

## Deviations from vibe-land, on purpose

- **No re-simulation.** vibe-land replays a frame after a split (`VIBE_CITY_RESIM_PASSES`); AGENTS.md forbids it. Strikes are predicted instead, as `threshold` predicts them.
- **Gravity loads every piece every frame.** vibe-land skips gravity on sleeping bodies untouched by contact. Here, a settled island's input repeats and the skip makes it free.
- **Islands are numbered by their lowest node, not by first appearance over the bonds.** Only the ids change.

## Next

1. **P0a:** measure vibe-land's stress stage and break sets on the same events. The `vibe-baseline` session brief is in the plan.
2. **Multi-structure Blast authoring (towns),** then 10k+ bodies.
3. **The resident path (P2):** Chebyshev block-Jacobi PCG, rigid-mode projection, native weighting. It converges in fewer iterations, so islands settle and skip.
4. **The V-cycle for large components (P3).**
5. **Guard digests** for `blast_gpu`, and the lab checked in the browser.
6. **The one-off Blast CPU cross-check.**
