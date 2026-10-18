# Replacing PhysX GPU + Blast: the quasi-static baseline, status and plan

Written 2026-10-18 on `claude/stress-top3-roi-8a4275` (PR Glavin001/rubble#39). Measured on
an M3 Max, Metal. Unreal Editor sometimes shared the GPU, which moved timings by up to 2×
between runs. Numbers marked "quiet" were taken without it.

## The goal

> The charter (goal, rules, milestones and acceptance criteria) is `stress-baseline-goal.md`.


**Goal.** Make rubble capable of replacing PhysX GPU and its native destruction stage as
vibe-land's destruction physics engine. Integrating it into vibe-land is out of scope; the
work is rubble's.

That means a quasi-static stress solver built into rubble as a first-class GPU feature,
written in Slang:
- **Resident:** one submission and one status read per tick, and no other host communication.
- **Forward-only:** fracture and splits take effect between rubble's substeps, the moment
  bonds fail. No rewind and no correction re-solve.
- **Deterministic** run to run.
- **At least feature, accuracy and speed parity** with the PhysX native stage, measured on
  vibe-land's own town bombard, ported into `stress_bench` as `vibe-town_suite_bombard` and
  `_suite_quick`.

**Targets** (quiet M3 Max), with breakage within vibe-land's variation on the same events:

| Tier | Median | p95 | Worst |
|---|---|---|---|
| Standard | ≤ 11.46 ms | ≤ 37.88 ms | ≤ 48.24 ms |
| Quick | ≤ 11.38 ms | ≤ 15.32 ms | ≤ 35.5 ms |

**Fidelity bar:** at least as faithful as the native stage's correction-limit-1 pass on the
free-wall and wall-row impacts.

Higher-fidelity dynamic impact handling comes after this baseline.

**Milestones: the shortest correct path, in order**

1. **Deterministic baseline.**
   - Fix the race in the first static solve after a split (rebuild and in-place paths).
   - **Done when:** both suite scenes repeat bit for bit and the guard is 24/24.
2. **GPU topology in Slang.**
   - Port the native stage's labelling and candidate bodies (`PxgDestructionTopology.cu`,
     `prepareCandidateBodies`): connected components of broken clusters, and each child's
     mass, centre of mass, inertia, principal frame and velocity.
   - **Done when:** results equal the host's `split_cluster`.
   - Can run in parallel with milestone 3.
3. **GPU bodies in rubble.**
   - A rubble kernel creates bodies from candidate records and re-points chunk colliders to
     their new body.
   - Slots are allocated by scans; the new-body delta rides on the tick's one status read.
   - **Done when:** a split creates its rubble bodies with no host call, deterministically.
4. **GPU split layout.**
   - A split parent's island tables are re-laid out on the GPU (replacing the host in-place
     split): no mirror sync, no rebuild.
   - **Done when:** a split frame does zero host work.
5. **Forward splits inside rubble's step.**
   - The stress stage joins rubble's step graph. After each substep: ingest contacts, run the
     cascade, commit and split. The next substep sees the separated bodies.
   - Every dispatch is sized on the GPU.
   - **Done when:** one submission and one read per tick, and the fidelity bar above is met.
6. **Speed to targets.**
   - Cheaper rubble substep (dispatch count, broadphase).
   - Cascade only on loaded structures.
   - The CCD and iteration policy decided with you. Lossless first; anything lossy only with
     your decision.
   - **Done when:** both tiers meet the targets, with breakage and fidelity reported.

---

## 1. What we are replacing (read from the code)

Sources:
- the side-session notes `blast-stress-solver-research.md` and `physx-blast-techniques-audit.md`;
- vibe-land `server/src/perf_suite.rs`, `physx-bridge/src/native_destruction.cc` and
  `destruction/src/native_runtime.rs`;
- PhysX `physx/source/gpudestruction`.

**The live path.** vibe-land runs PhysX's **native GPU destruction stage**, inside
`simulate()`/`fetchResults()`. The suite asserts `backend_name() == "native"`.

**PhysX rigid bodies:**
- PGS solver with 4 position and 1 velocity iteration;
- one 60 Hz step and **no sweep CCD**. Scene CCD disables the destruction stage, and
  speculative CCD on projectiles is off;
- stabilization on;
- fragment depenetration capped at 2 m/s while vehicles exist.

**Stress solver (Blast's GPU kernels via CuMetal):**
- least-work forces of an isotropic spring network, `sqrt(EA/L)` with floors;
- f32 PCG with a Chebyshev block-Jacobi preconditioner;
- **16 iterations a tick, tolerance 1e-3, warm-started**. Unconverged answers are published:
  `PX_DESTRUCTION_ALLOW_UNCONVERGED=1`, and the town bombard was unconverged on 90 of 90 ticks;
- stress is force over remaining area, with the bending gain capped at 3;
- damage rate 2/s between the elastic and fatal limits.

**Fracture timing:**
- `internalCorrectionLimit = 1`. On a fracture tick the stage applies the verdict and then
  re-solves the whole scene's collide+solve once.
- It also runs a second stress evaluation.
- The last pass's new splits take effect at end-of-tick motion.

**What bounds the tick:**
- the 16-iteration cap, tolerance and force tolerance 1e-3, and the correction limit of 1;
- chunk pairs produce no CPU contact reports;
- delta topology observation;
- verdict sampling every 60 ticks;
- the stage's GPU lifecycle work: candidate bodies, slot allocation, incremental topology.

There are **no caps** on bonds, bodies or contacts per tick.

**Reference numbers.** All are vibe-land perf suite, runtime profile, M3 Max (`scripts/perf/suite-baseline.json`):

| Scene | Median | p95 | Worst | Notes |
|---|---|---|---|---|
| Town bombard, **standard tier** (90 ticks, 5 shots) | **11.46 ms** | **37.88 ms** | **48.24 ms** | 6,707 bonds broken, 1,569 awake bodies peak, 39,852 contacts, 17 correction passes |
| Town bombard, quick tier (60 ticks, 2 shots) | 11.38 | 15.32 | 35.5 | 638 bonds; only the first shot breaks anything. This is the tier I quoted earlier |
| Town idle | 10.2 | 15.3 | | |

---

## 2. What we have, read from our code

**Three coupling paths exist** (`stress_bench.rs` `ladder`; `world_coupling.rs`
`coupled_prepare` / `select_islands`; `world.rs` `step_frame_coupled`):

| `--config` | Fracture timing | Resident? |
|---|---|---|
| `blast`, `qs` (quasi-static, no islands) | rubble resolves every contact against the whole intact body. The stress solve takes those impulses, and fracture applies from the **next frame**. This is effectively `internalCorrectionLimit = 0`: weaker than Blast's 1 | **Yes**, with `STRESS_RESIDENT=1` (opt-in) |
| `tiered` (quasi-static, plus impact islands for impactor hits above 5 m/s) | Routing before the step. Island bodies are simulated by the stress solver at microsecond substeps and break when they break. Forward, no rewind | No |
| `realtime` (impact islands wherever the rigid assumption fails) | As `tiered`, for every such pair | No |

- **The resident path is the limit-0 one.** It runs only under `fast_qs_conditions`, which
  requires `!tiered`.
- **Every bombard comparison I reported earlier used `--config blast` with
  `STRESS_RESIDENT=1`.** That is cheaper and **less faithful** than vibe-land. The comparison
  was not apples to apples, and I wrongly called it matched.

**What "resident" covers today** (`world/resident_frame.rs`):
- rubble steps with `step_submit`;
- its solved contact rows become stress manifolds on the GPU (`coupling.slang`);
- poses, load filter and static loads run on the GPU (`resident.slang`);
- the fracture cascade runs on the GPU, which decides on its own whether each pass runs
  (`cascade.slang`: gate, fixed points, certified repeats);
- **one combined status read per frame**. The CCD bodies' speeds now ride on it (ce75442).

**What still goes through the host:**
- every **split** (see 3c);
- routing, island planning, segment setup and write-back on the impact-island paths;
- new projectiles (`add_impactor` refreshes on the host).

**The stress solver is not part of rubble's step.** It is a separate GPU solver sharing
rubble's device and reading its buffers.

---

## 3. What I tried, and what happened

### 3a. Worked, and kept (lossless unless noted)

| Change | Effect |
|---|---|
| Resident quasi-static frame (phases 2b–5): GPU ingest, poses, filter, static loads, cascade, one read per frame | Vibe Town idle 35 → 4.2 ms (elastic, budget 2 × 32) |
| Repeat gate: unconverged repeats skipped, fixed points skipped, converged repeats certified (your decision (a)); gate carried across rebuilds | Blast-matched idle 55–60 → 9.7 ms; later 5.3 ms median, p95 5.9 (quiet) |
| Metal barrier fix: device-and-threadgroup fences only, no texture/imageblock fence | Static solve 38 → 20 ms per bombard frame; bit-identical |
| Contact-material lookup made a hash map (it had been quadratic) | Split frames 1.3–2.5 s → 130–225 ms |
| Blast-matching switches: `--authored-fatal`, `--pack-masses`, `--method blast_model=on`, `STRESS_QS_UNCONVERGED=commit`, `STRESS_QS_PASSES=2`, `STRESS_QS_BUDGET` | Blast's model and per-tick behaviour as configuration. Off by default |
| Accuracy at Blast's iteration budget (each solver's single frame against its own converged answer) | **Ours is more accurate per iteration.** House-1story at 16 iterations: Pearson 0.93 vs 0.83, median ratio 0.95 vs 0.37. Ours at 10 iterations ≈ Blast at 25 |
| CCD body speeds in the frame's single status read (ce75442) | One fewer per-frame host read (0.2–0.4 ms); resident tests pass |
| `pair_point` with one call site per helper (206af41) | Impact substep −1 to −3 %, bit-identical |
| `blast_model` classified in `methods::TABLE` (a test had been failing) | |
| Diagnostics: `STRESS_RESIDENT_PHASES`, `RUBBLE_GPU_TIMING`, `STRESS_GPU_SPLIT_HASH`, `STRESS_GPU_STATE_DUMP`, `STRESS_GPU_METAL_CONTRACT=off` | |

### 3b. Did not work, or measured and not adopted

| Change | Result |
|---|---|
| **In-place splits** (`split_resident`: only the split parent's slice of the tables is rewritten) | Split frames 130–225 → 37–60 ms. But **the bombard became nondeterministic** and differed from the rebuild path. Now **opt-in** (`STRESS_QS_SPLIT_IN_PLACE=1`). See 3c |
| `impactor_contact` rewrite (one sample and one penetration call site) | Looked like −13 %. In fact the **default Metal build compiled it wrongly**: b6_spall broke 0 of 288 bonds. Built without FMA contraction it is correct but only −3 %. Rejected; documented in `findings/substep-latency.md` |
| rubble look-ahead CCD (`STRESS_RUBBLE_SPECULATIVE_CCD=1`) | Bombard median 17.9 → 9.1 ms, but broken bonds 2,650 → 5,071. Opt-in, waiting on your decision |
| rubble at 4 iterations (PhysX's count) | 20.9 → 15.8 ms, broken bonds about 2×. Not adopted |
| Global speculative contact offset (`STRESS_RUBBLE_SPECULATIVE=1`) | Overflows rubble's candidate capacity (about 2.3 m offset for a meteor) |
| Resident island segments (6a+, `STRESS_RESIDENT_ISLANDS=1`) | Same digest, segment setup 7.9 → 9.7 ms. Off |
| Merging the contact kernels' two-pass loops | No gain (the compiler probably unrolls them back) |

### 3c. The in-place split race (found, not fixed)

**Narrowed down:**
- Off, the 3 s bombard repeats bit for bit. On, identical runs diverge, on Metal and on WGSL alike.
- Per-frame hashes: the tables each split uploads are **identical across runs**. The divergence
  starts in **the first static solve of a child island created by the previous frame's
  in-place split**, from identical inputs.
- One cause found and fixed in the working tree: cut bonds kept their old secant stiffness in
  the static-solve scratch while the incidence padding still pointed at them. Clearing it
  changed the outcome, but did not make it deterministic.
- At least one more source of stale or racy per-bond or per-chunk data remains in the
  in-place layout. It is not found yet.

### 3d. Where a bombard frame goes (quasi-static, no islands, quiet GPU)

| Part | Time per frame |
|---|---|
| rubble: 2 substeps (CCD for the 60–140 m/s projectiles), about 5 ms each | about 10 ms |
| Inside one substep: broadphase | 0.85 ms |
| Inside one substep: narrowphase | 0.55 ms |
| Inside one substep: 10 primal iterations × 11 colour dispatches | about 2.8 ms |
| Stress cascade | about 3 ms |
| Split frames (full sync and rebuild) | 100–400 ms |

Median 14.8–15.0 ms against Blast's 11.46; p95 107–336 ms against 37.88. vibe-land saves the
rigid-body part with a single step, no CCD and 4 iterations, and pays for its correction pass
instead.

---

## 4. Uncertainties

1. **A forward-pass equivalent of Blast's correction pass in quasi-static mode.** Blast
   re-solves once after a fracture. My reading of your design: the separation takes effect at
   the substep it happens in, and the rest of the frame runs on the separated bodies, so
   nothing is re-solved. In quasi-static mode that means a stress evaluation inside rubble's
   step, between substeps, on the frames that carry impacts. Not yet decided:
   - whether rubble's 2–3 CCD substeps give fine enough timing to match limit 1 on the
     benchmark scenes, or whether impact frames need more substeps;
   - what a cascade costs per substep.

   The free-wall ram (oracle: the ram leaves at 35 m/s; Blast-style limit 1: about 10.9 m/s)
   is the measure.
2. **Breakage parity.**
   - Our matched configuration breaks 2,650–3,343 bonds on our bombard, where vibe-land's
     standard tier breaks 6,707.
   - Our scene fires at different houses and on different ticks from vibe-land's.
   - So I don't yet know how close our outcome is. The benchmark has to be vibe-land's exact
     event list first.
3. **CCD and iterations.**
   - vibe-land has no CCD and runs 4 iterations; its thin walls tunnel at 60 m/s. Our CCD is
     more faithful and costs about 5 ms per extra substep.
   - Matching PhysX would be a fidelity loss, which needs your decision. Staying ahead on
     cost while keeping CCD needs rubble's substep cheaper.
4. **The remaining in-place race**, above.
5. **rubble-side work** the plan needs:
   - spawning bodies on the GPU;
   - masking bodies out of a solve;
   - running a stress stage inside rubble's step graph.

   rubble lacks all three. I have not yet estimated their size.
6. **Determinism.** Determinism of the whole coupled frame has only been checked on our
   bombard and the guard set.

---

## 5. Plan

Each step ends with a measurement on the same benchmark (5a). Nothing lossy is adopted
without your decision.

### 5a. Benchmark first: vibe-land's scenes, exactly

- **Port vibe-land's `town-bombard` event list** into `stress_bench` as `vibe-town_standard`
  and `vibe-town_quick`, from `scripts/perf/suite.json` and `perf_suite.rs`:
  - cannonballs at ticks 0, 20 and 40 (house-16, house-23, house-25);
  - meteors at ticks 10 and 30 (house-18, cinema-1);
  - 90 ticks (standard) and 60 ticks (quick).
- Same pack, masses and fatal limits as vibe-land.
- **Report per run:**
  - median / p95 / worst tick;
  - broken bonds, fragments, peak awake bodies;
  - the digest. Two runs must agree.
- **Fidelity checks beside it:** the free-wall ram, two walls, house_ram. Compare against the
  oracle and against a Blast-style limit-1 pass.
- **Done when:** the table compares our numbers with vibe-land's suite baseline row for row.

### 5b. Forward fracture timing in quasi-static mode (replaces Blast's correction pass)

- Inside rubble's frame, on substeps whose engine contacts load a structure:
  1. run the resident cascade on that substep's contact impulses;
  2. commit fractures;
  3. split before the next substep. The rest of the frame runs on the separated bodies.
- Frames without such contacts keep one cascade per frame, as now.
- Only one forward pass ever runs.
- **Done when** the free-wall ram and the two-wall row come out at least as close to the
  oracle as Blast's limit 1, with no rewind and nothing re-solved.

### 5c. Splits on the GPU (phase 6e), so a fracture costs kernels, not host work

1. **Fix or replace the in-place layout.** Find the remaining stale state; the gate is
   determinism plus agreement with the rebuild path.
2. **Labelling on the GPU:** connected components of a split parent on the GPU, children laid
   out in the parent's ranges, island records and incidence written by kernels.
3. **New rubble bodies created on the GPU** in rubble's body and collider buffers, from
   candidate bodies the GPU computes: mass, centre of mass, inertia and point velocity. This
   is what PhysX's stage does: `prepareCandidateBodies`, GPU slot allocation, incremental
   topology.
   - The host learns about it through the frame's single status read, as a delta.
   - Retired bodies' slots are reused.
- **Done when:** a split frame does no host sync and no rebuild, and the bombard p95 is set by
  GPU work.

### 5d. The stress stage inside rubble's step (first-class)

- Make the stress stage part of rubble's step graph (`engine-exec` `rigid_plan`):
  1. contact ingest after the solve;
  2. the cascade;
  3. fracture commit;
  4. GPU splits.
- One command buffer per frame and one status read.
- **Done when:** a vibe-land tick is one submission and one read, with nothing round-tripped.

### 5e. Performance to budget

**Targets (quiet M3 Max):**

| Scene | Median | p95 | Worst |
|---|---|---|---|
| Idle | at most 10.2 ms (now 5.3) | | |
| Bombard, standard tier | ≤ 11.46 ms | ≤ 37.88 ms | ≤ 48.24 ms |
| Bombard, quick tier | ≤ 11.38 ms | ≤ 15.32 ms | ≤ 35.5 ms |

All with breakage and fidelity reported alongside.

**Levers, lossless first:**
1. **rubble's substep.** About 70 small dependent dispatches; the empty colour headroom and
   the per-colour launch dominate. Also the broadphase.
2. **Cascade cost per substep** (5b makes it matter).
3. **CCD policy, your decision:** keep sub-stepping the whole world; sub-step only the fast
   bodies' contacts; or match PhysX (one step, look-ahead CCD).
4. **Iterations, your decision:** 10 vs PhysX's 4.

### 5f. After the baseline

- Dynamic impact handling: the `tiered` and `realtime` impact islands, made resident on the
  same foundation.
- Then their per-substep cost (currently about 55 µs × about 1,190 substeps per impact frame).

## Decisions I need from you

1. **Fracture timing in quasi-static mode.** Is 5b right: stress evaluated and fracture
   applied between rubble's substeps on impact frames, and only one forward pass?
2. **CCD and iterations for the baseline.** Keep rubble's CCD substeps and 10 iterations, and
   make them cheaper? Or match vibe-land's PhysX (no CCD, 4 iterations) with the breakage
   change reported?
3. **The benchmark.** May vibe-land's exact `town-bombard` event list (5a) replace our own
   bombard as the scene of record?
