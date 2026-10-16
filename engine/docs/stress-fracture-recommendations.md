# Stress-fracture performance: what is done, and what needs your decision

Every change adopted keeps the physics the oracle defines. The lossless changes below
were gated bit for bit: the same end-state digest on all 15 `stress_bench` regimes
(accuracy and town sets, `STRESS_GPU_FIXED_HORIZON=1`), plus the integrated and
coupled-vs-reference tests. The levers after them cost accuracy, change an asset or the
engine's behaviour, or are research. Each says what is measured and what is not.

## Done without changing results (2026-10-16)

| Change | Where | Effect |
|---|---|---|
| Convex hulls built in a deterministic order | stress-ref `hull.rs` | Every coupled run now reproduces bit for bit across processes. Before, the prestress differed per process. |
| Host waste out of every frame: no `Scene` clone, structures shared (`Arc`), faster `pack`, cached engine shapes and step bounds, buffers rewritten in place | stress-gpu, stress-ref, stress-layout | Idle 64-house town 210 → 80 ms (median, interleaved A/B) |
| Quiet clusters left out of the GPU tables (Tier 0 of the plan) | `GpuSolver::quiet_clusters`, `stress_layout::pack_with` | 80 → 25 ms; 16 houses 46–69 → 18–22 ms |
| rubble body read-backs reused, routing bounds without building boxes | stress-rubble, `select_islands` | 25 → 6–8 ms |
| Certified damped steps computed on all cores | `stability::certified_damped_steps` | 64-house setup 55 → 15 s |

A quiet cluster is excluded only while the GPU provably does nothing to it:
- settled, anchored and engine-driven;
- no load on any of its chunks;
- no scripted loads, events or probes in the scene;
- every bond's per-frame static-fatigue decrement under a thousandth of half an f32 ulp of its life.

`STRESS_GPU_QUIET_EXCLUDE=0` turns it off.

## Where the time goes now

- **Idle and resting scenes.**
  - The host is mostly out of the frame. What remains of an idle town frame is the coupling's bookkeeping over the clusters (routing, refresh) and one GPU round trip: 5 ms for 16 houses and 13 ms for 64 (median, three-way rerun).
  - PhysX GPU needs 68 and 176 ms for the same towns.
- **Destruction frames are GPU-bound.**
  - On house-1story rammed, 89 % of the main thread waits on the stress substeps; host work is about 5 %.
  - A substep costs about 44 µs uncontended. It is three dependent dispatches (GPU timestamps: contact islands 26 µs, contact forces 23 µs, sums and bonds 18 µs), about 2,900 per 1/60 s frame.
  - Three trivial dispatches cost 6 µs on this GPU (`stress-gpu/examples/dispatch_floor.rs`), so dispatch count is not the limit; kernel execution is.
- **Kernel trials (2026-10-16).** None was lossless and worth taking:
  - 512-thread island groups: −5 % on two scenes, +2 % on the third, and the results change at the ulp level;
  - a blocked bond-load gather: 0 %;
  - device fences narrowed to group barriers: 0 %.

  The default Metal System Trace records no limiter counters, so the next step there is an Xcode GPU capture with performance limiters on the three kernels. The number of substeps remains the multiplier.

## Levers that change accuracy (measure, then you decide)

0. **Fix the contact-crushing pass-through (a model fault, found 2026-10-16).**
   - With the default methods (`layer_contact` and `contact_crushing` on), a contact face that has crushed through its first layer (`d_u = 2 G_c / f_c`, about 1.3 mm for brick) pushes with zero force while the chunks overlap (stress-ref `pair_layer_contact_with`). Fragments then slide through walls.
   - On `s_house_car`, 30–60 pairs overlap by more than 5 mm from t = 0.3 s, up to 172 mm. With crushing off: one pair, 5.3 mm.
   - Proposed fix: after a crushed layer, the material behind it is a fresh layer with full strength and crushing energy. The energy absorbed per millimetre stays the same, but pushing through meets about half the crushing strength instead of nothing. Mortar has the same fault.
   - It changes oracle results, so it needs your OK. Every gate and golden would then be regenerated. Viewing workaround: `--feature contact_crushing=off`.

1. **Stress step safety 0.9 instead of 0.5.**
   - About 1.6–2× fewer substeps where the bond bound sets the step.
   - Not converged: the house ram drifts to 2.62 m/s against 3.04 (REALTIME_ROADMAP A12 #20).
   - Recommendation: keep 0.5.
2. **Sliver elimination in the assets.**
   - House-1story needs 4.5× fewer substeps once 16 of its 319 chunks are condensed (A3).
   - It changes the model and is an asset (Shardwright) decision. Gate it against the oracle with `fidelity.py` first.
3. **A calibrated critical-speed router.**
   - Below a per-material critical impact speed, resolve impacts quasi-statically instead of in an explicit window. Push-over is right there; punch-through starts at 15–20 m/s on b5.
   - Large savings for debris rain and slow pushes; wrong if the calibration misses a case.
   - The Blast-equivalent session's feature ladder (`docs/stress-fracture-blast-ladder.md`, in progress) measures this trade directly.
4. **Debris release certificates.**
   - Fragments that can no longer break anything leave the stress solve early.
   - Exact only with a rigorous bound; the simple weakest-bond test is not conservative (A12 #19).
5. **Stiffness or mass scaling.** Big cuts in substeps, but it shifts the boundary between impulsive and quasi-static response. Not recommended.

6a. **Keep debris in its impact island until it separates (accuracy up, cost up).**
    - Both coupled pipelines hand debris to the engine when an island ends. b5's punch-through then breaks into 3–32 fragments where the oracle has 106–107, and the ram exits 2–3 % fast.
    - Keeping fragments in the stress solve's own contact until they leave each other's swept bounds would close most of that gap.
    - It lengthens impact windows (cost not measured yet).

6b. **A per-frame static-solve budget, Blast-style (quasi-static use only).**
    - `STRESS_GPU_STATICS_NEWTON=10` takes villa-savoye from 1.9 to 0.65 s a frame.
    - It changes house-2story's rammed outcome (7 broken bonds instead of 34).
    - Our real-time configuration already beats the Blast-equivalent one on cost and accuracy (`docs/stress-fracture-blast-ladder.md`), so this only matters if a quasi-static mode is wanted.

6c. **The ladder's "damping" rung (b5_v20 513 → 350 broken bonds) is not a GPU dashpot fault (checked 2026-10-16).**
    - The GPU world's own dashpot step certificate makes no difference on b5: `STRESS_BENCH_ORACLE_STEP=1` gives identical outcomes at 10, 15 and 20 m/s, since the contact step sets the substep there.
    - The established method set is fully honoured on the GPU.
    - The GPU-vs-reference parity tests (damped materials, 4,000 substeps) hold to 1e-3.
    - The rung's effect is not systematic across speeds: 246 → 370 at 10 m/s, 421 → 424 at 15 m/s, against the oracle's 318 and 367. b5 near its threshold is sensitive to any change.
    - A seed-spread study (the plan's step 7 families) would put a number on that sensitivity.

## Levers that change engine behaviour

6. **Let coupled rigid bodies sleep.**
   - The coupling keeps rubble's free bodies awake so their support reactions reach the stress solve.
   - Freezing a sleeping body's loads into the solve would let rubble sleep debris and cars.
   - Quiet anchored houses already cost nothing.
7. **Swept collision (CCD) in rubble (engine work, changes behaviour).**
   - A fast body makes the coupling sub-step the whole rigid scene: 11 rubble steps per frame on the 40 m/s free-wall ram, the one regime where rubble is behind PhysX GPU (88 against 67 ms median). It also costs 2–4 % on b5 at 30–40 m/s before impact.
   - rubble has no usable CCD to switch on: `ccd_classify` flags fast bodies, but `ccd_sweep` is a plane-only stub ("stretch goal") with no box or hull time of impact.
   - Closing the gap means building swept collision for boxes and hulls (conservative advancement or speculative contacts) in `world_step.slang`, then measuring it against the oracle. That is engine-owner work.

7a. **Merge rubble's grid broadphase fix (8352765, on the Blast branch).**
    - The grid broadphase drops static pairs below the querying collider's index once there are more than 512 colliders, so debris can fall through anchored structures.
    - The fix is correct. It makes b8_sudden over-break (36 bonds against the oracle's 11), because the engine's contact-to-load path then over-breaks where the oracle uses its own penalty contact.
    - This is a decision for the engine owners: take it together with a fix to that path.

7b. **rubble: read the previous step's counts on the GPU (lossless, engine owners).**
    - Each AVBD step's uniforms carry the previous step's contact and manifold counts (`write_host_uniforms`: warm-start hash size, previous contact range), so the host must read the counters back before it can record the next step.
    - If the kernels read those counts from the counters buffer instead, the coupling could record a frame's substeps back to back, with the CPU encoding (about a third of a rubble step) overlapping the GPU.
    - That would recover about 1 ms per substep where rubble substeps for a fast body: the free-wall ram's 11 per frame, and b5 at 30–40 m/s before impact.
    - It is a change to `world_step.slang`, so it belongs to the engine owners, gated bit for bit on the engine's own suites.

## Large engineering steps (lossless; scope needs a go-ahead)

8. **Finish the GPU-resident coupling (stages 2–3 of `docs/stress-fracture-gpu-resident.md`).**
   - Stage 1 and Tier 0 already removed most host work, so stages 2–3 are now worth about 5 % on destruction frames and a few ms on idle ones.
   - Their value is architectural: no CPU mirror, a frame cost that scales with activity, and the base for running inside rubble's AVBD step.
9. **Fewer substeps without losing accuracy (research, running in sessions).**
   - Multirate explicit integration: bonds at power-of-two rates, conservation-tested (WIP).
   - An explicit hot zone with Newmark elsewhere.
   - These attack the real multiplier, the substep count. Each is gated against the oracle before it can be adopted.
10. **Cross-platform determinism of asset bakes.**
    - Bakes are byte-identical across runs and thread counts on this Mac.
    - Shardwright's CI does not check `building_lite` across OS and architecture.
