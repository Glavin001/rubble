# Real-time destruction: review, measurements and roadmap

How the GPU stress solver (`crates/stress-gpu`) gets to real time at city scale, without
giving up the oracle's accuracy:
- **Target:** about 100 buildings of 2–3k chunks each, many impacts, 60 Hz, every complete step ≤ 8 ms, coupled with PhysX.
- **Inputs:** two colleague documents (a research compendium and "Speed-of-Light Analysis v2"), checked against the code and against measurements on this branch.
- **Ground rules:**
  - `stress-ref` is the ground truth, and strictly better methods are adopted there.
  - The GPU approximates only where the oracle measures the cost (`IMPROVEMENTS.md` protocol).
  - Every timing here is an Apple M3 Max proxy. None is an RTX 4090 qualification.

Reproduce the tables with:

```bash
target/release/stress-gpu-steps --levers <packs>                               # A1
cargo run --release -p stress-ref --example rest_utilization -- <packs>        # A2
cargo run --release -p stress-reduce --example condense_report -- <packs>      # A3
STRESS_METHODS=established target/release/stress-gpu-fps --config realtime \
  --gpu-only --town house-1story:100:4                                         # A4
```

Packs are in `packs/` (copied from `blast/blast-stress-demo-rs/assets/scenes`).

## A. Measurements

### A1. What sets the explicit step (`stress-gpu-steps --levers`)

Substeps per 1/60 s frame at safety 0.9:
- **true:** power-iteration estimate plus 3%, with the top mode's damping;
- **certified:** the oracle's `true_step` bound, whose damping is the largest dashpot ratio of any bond.

Gains are ratios of substeps per frame.

| Pack | Chunks | Reference | True | Certified (undamped) | Rot. inertia ×4 | Multirate | Top mode carried by |
|---|---|---|---|---|---|---|---|
| brick-building | 248 | 738 | **179** | 205 (193) | ×1.02 | ×1.00 | uniform 240–480 kg bricks |
| house-1story | 319 | 29,491 | **1,264** | 4,730 (1,124) | ×1.44 | ×3.36 | one 66 kg timber beam (100%) |
| house-2story | 599 | 31,403 | **1,358** | 8,151 (1,292) | ×1.30 | ×3.03 | two 3.2 kg timber stair pieces (98%) |
| parking-garage | 3,177 | 12,473 | **934** | 3,139 (908) | ×1.29 | ×2.71 | three 18 kg concrete stair pieces |
| villa-savoye | 1,045 | 12,741 | **931** | 3,385 (905) | ×1.29 | ×1.90 | two 17.5 kg concrete stair pieces |
| neighbourhood | 8,877 | 31,403 | **1,360** | 17,430 (1,294) | ×1.31 | ×3.76 | 3.2 kg timber stair pieces |
| park-432 | 24,452 | 12,727 | **1,271** | 9,730 (1,205) | ×1.76 | ×2.86 | 18 kg concrete stair pieces |
| petronas | 24,766 | 55,752 | **1,369** | 2.6 M (1,287) | ×1.47 | ×2.89 | one 1.6 t steel parapet piece (97%) |
| skyline | 62,525 | 55,752 | **1,944** | 2.1 M (1,827) | ×2.03 | ×4.34 | 21.8 kg concrete stair pieces |
| minas-tirith | 49,049 | 125,502 | **6,766** | 324 M (6,023) | ×2.09 | ×18.1 | small stones beside massive blocks |

What this shows:

- **One to three light chunks set the step of every mixed building.** On most packs they are stair pieces. Damping is not the limit: switching it off gains 1.02–1.09×.
- **The certified frequency is tight** (undamped column within 2–10% of the estimate).
- **Its damping term is not.** The largest β of any bond, times ω_max, gives 3–6× too many substeps on houses and millions on the steel and stone packs (v2's "worst β × worst ω" pathology). A certified bound on ρ(M⁻¹C) instead of β_max·ω would recover it. **Open, for the oracle.**
- skyline, minas-tirith and petronas now import: bonds whose shared face has no width are left out, 16, 8 and 11 of them.

### A2. Utilization at rest (`rest_utilization`)

Each bond's failure index at the gravity prestress, evaluated without committing:

| Pack | Bonds | > 0.4 | > 1 | Max (where) |
|---|---|---|---|---|
| brick-building | 494 | 0% | 0 | 0.09 |
| house-1story | 877 | 0% | 0 | 0.37 (timber post–wall) |
| comp-wall-bay | 109 | 0% | 0 | 0.01 |
| rig-portal | 39 | 2.6% | 0 | 0.54 |
| house-2story | 1,652 | 1.6% | 7 | 2.22 (mortar under the steel rails) |
| villa-savoye | 2,464 | 15.6% | 37 | 3.92 (concrete stair) |
| parking-garage | 9,194 | 14.3% | 309 | 4.15 (concrete beam–slab) |
| neighbourhood | 23,581 | 12.2% | 311 | 2.98 (concrete slab–core) |
| park-432 | 78,196 | 5.2% | 134 | 1.60 (concrete slab–beam) |

- **Small buildings are well below strength at rest,** so a v2 energy certificate (E_safe, bonds ≤ 0.4) can work there.
- **Every multi-storey reinforced-concrete pack has bonds over strength under its own weight.** These are calibration or authoring issues (the assets were authored for the older Blast model).
- **Effect:**
  - They crack at rest in the truth.
  - They block island-level energy certificates on those buildings until the assets or materials are fixed.
  - A location-aware bound (influence lines) can still certify loads far from the overloaded joints.

### A3. Static condensation on our own partitions (`stress-reduce`)

The network assembled from the oracle's bonds reproduces the oracle's gravity equilibrium to between 2e-14 and 1e-11 (`tests/condensation.rs`). Condensation plus recovery equals the full solve to round-off.

**house-1story** (1,794 free DOFs; full gravity PCG 769 iterations):

| Parts of | DOFs kept | Storage (f32) | Gravity PCG | Explicit step gain |
|---|---|---|---|---|
| 8 | 100% | 0.2 MB | 773 | ×1.00 |
| 32 | 65% | 0.5 MB | 661 | ×1.00 |
| 128 | 25% | 1.2 MB | 519 | ×1.00 |
| 256 | 14% | 3.2 MB | 323 | ×1.00 |

- **Partitioning shrinks the static system** (86% of DOFs gone at 256-chunk parts).
- **It gives no explicit-step gain:** the chunk that sets the step is a boundary chunk. Results by pack:
  - house-1story ×1.00 at every part size;
  - house-2story ×1.04 up to parts of 128, ×1.25 at 256;
  - brick-building ×1.00.

**Sliver elimination.** The study greedily eliminates the retained chunk carrying most of the current top mode, then reports the Guyan step gain:

| Pack | 1 | 2 | 4 | 8 | 16 chunks eliminated | First eliminated |
|---|---|---|---|---|---|---|
| house-1story (319 chunks) | ×1.00 | ×1.33 | ×1.34 | ×1.48 | **×4.51** | 66–67 kg timber beams, then 16 kg timber posts |
| house-2story (599 chunks) | ×1.00 | ×1.03 | ×1.04 | ×1.04 | ×1.66 | 3.2–3.4 kg timber stair pieces |
| brick-building (248 chunks) | ×1.00 | ×1.00 | ×1.00 | ×1.01 | ×1.02 | uniform bricks: nothing to gain |

- **On house-1story, 5% of the chunks are worth ×4.5 in the explicit step.**
- **What the eliminated chunks lose:**
  - They follow their neighbours quasi-statically. Their own vibration, at 15–20 kHz, has no gameplay meaning.
  - Their bonds' forces are recovered exactly in the static limit.
  - Their inertia moves to the retained neighbours (`M~`).
- **Status:** a step and model change, so it must be proven as an oracle switch against explicit (`fidelity.py`) before the GPU uses it. Queued as Phase 1b.

### A6. Cost of an intact-only kernel per substep (`stress-gpu-bench`)

Both kernels run the whole segment in one dispatch (narrow islands), or one round of dispatches per substep (wide), whichever is faster. Packs are timed with rebar stripped, because the intact kernel has no rebar and on an intact bond rebar is only added stiffness.

| Scene | Chunks / bonds | Full model (`GpuSolver`) | Intact kernel (`GpuStress`) | Gain |
|---|---|---|---|---|
| slab 12×6×2 | 144 / 324 | 14.4 µs | 2.4 µs | ×6.0 |
| slab 16×8×2 | 256 / 592 | 22.8 µs | 5.8 µs | ×3.9 |
| brick-building | 248 / 494 | 23.0 µs | 5.8 µs | ×4.0 |
| house-1story | 319 / 877 | 24.9 µs | 9.4 µs | ×2.6 |
| house-2story | 599 / 1,652 | 25.4 µs | 10.0 µs | ×2.5 |
| villa-savoye | 1,045 / 2,464 | 23.5 µs | 7.8 µs | ×3.0 |

- **The full model sits at a ~24 µs floor for every pack size.** That is latency (code size, spills, A2), not work.
- **An intact-only path gains ×2.5–6.**
- **Even the intact kernel is latency-bound on Metal:**
  - one threadgroup is too narrow above ~250 chunks;
  - a dispatch per substep costs ~1 µs each, plus a device-memory exchange.
- **Per awake house-1story on M3:**
  - today: 1,264 substeps × 25 µs = 32 ms;
  - intact kernel: × 9.4 µs ≈ 12 ms;
  - intact kernel plus sliver elimination (÷4.5): **≈ 2.6 ms**.
- **Phase 1 gate** (intact floor ≤ 1 µs): **not met** on Metal with this kernel. The architecture there is therefore:
  - a few awake explicit buildings per frame;
  - everything else asleep or quasi-static behind certificates;
  - slivers eliminated.

### A4. GPU frame times on pack towns (M3 Max, true step, adaptive mode)

Run with `stress-gpu-fps --config realtime --gpu-only --frames 30 --town house-1story:N:RAMS` under `STRESS_METHODS=established`. The GPU world does not run layer contact yet.
- **Idle:** the fresh intact town under gravity only.
- **Destruction:** the first RAMS houses each rammed by a 400 kg box at 12 m/s.
- Frame times are complete `GpuWorld::step_frame` calls, host included.

| Town | Chunks / bonds | Regime | First frame | Median | Worst | Broken bonds | Substeps per frame |
|---|---|---|---|---|---|---|---|
| 16 houses | 5,104 / 14,032 | idle | 15.7 ms | 2.0 ms | 3.7 ms | 0 | 434 |
| 16 houses | | 1 rammed | 42 ms | **306 ms** | **4.5 s** | 733 | 831 |
| 64 houses | 20,416 / 56,128 | idle | 51 ms | 2.2 ms | 3.7 ms | 0 | 434 |
| 64 houses | | 4 rammed | 94 ms | **2.3 s** | **40 s** | 3,249 | 956 |
| 100 houses | 31,900 / 87,700 | idle | 93 ms | 3.4 ms | 5.3 ms | 0 | 434 |

- **Idle is already cheap.** 100 intact houses cost 3.4 ms median, with sleeping and quiet islands, but the startup frame is 93 ms.
- **Destruction is 2–3 orders of magnitude over budget,** and it scales with the number of struck buildings. One rammed house among 16 takes a 4.5 s worst frame. The breakdown is in A5.
- **100 houses, 4 rammed:** 5.8 s median, 94 s worst frame (5,311 bonds broken).
- **parking-garage ×4 is not an idle scene.** Over strength at rest (A2), it collapses under self-weight: 6,328 bonds broken, 1.1 s median, 22 s worst frame. Its rammed run then **crashes** with a 4.87 GB `contact state` buffer, over wgpu's 4 GB limit (debris pair state grows without bound).

### A5. Where destruction frames go

**Standalone GPU world, one rammed house** (`STRESS_GPU_TIMING=1`, 30 frames: 10.3 s in all, 8.7 s of it GPU):

| Kernel | GPU time | Per dispatch |
|---|---|---|
| `contact_forces` (debris pairs, up to 6,808) | 2.7 s | 89 µs |
| `island_frame` (one substep per dispatch inside the contact pipeline) | 2.4 s | 78 µs |
| `impactor_integrate` + `contact_sums` | 0.8 s | |

Host side: 225 splits (+106 resumed), 0.7 s split, 0.3 s readback.

- In the standalone world the stress solver resolves every debris–debris contact at the stress substep, about 830 times per frame. That is the oracle's ground-truth mode, not the real-time architecture.
- In production PhysX owns debris contact, and the stress solver substeps contacts only inside impact islands.

**Engine-coupled GPU solver** (`stress-gpu-coupled-fps --gpu-only`; contacts are the engine's impulses; ram on frames 0–2, then a push):

| Scene | Regime | Stress step | Median | Worst | Substeps per frame |
|---|---|---|---|---|---|
| house-1story | idle | reference | 919 ms | 942 ms | 37,484 |
| house-1story | idle | true | 32.5 ms | 33.0 ms | 1,264 |
| house-1story | idle | true + sleep | 1.6 ms | 1.9 ms | 1 |
| house-1story | destruction | true | 33.2 ms | 35.5 ms | 1,264 |
| house-1story | destruction | true + sleep | 36.1 ms | 40.3 ms | 1,264 |
| wall 16×2×8 | destruction | true | 15.1 ms | 16.8 ms | 452 |
| wall 40×2×20 | destruction | true | 17.3 ms | 18.7 ms | 453 |

- **One awake house costs about 33 ms per frame:** 1,264 true substeps × about 25 µs. That is the per-substep latency floor (A2), whatever the house's work.
- **The levers that close it multiply:**
  - per-substep latency (intact kernel, target ≤ 5 µs);
  - substeps per frame (slivers, ×2–4 by A1);
  - how many buildings are awake (sleeping already brings idle to 1.6 ms).
- **Debris contact must stay with PhysX:** impact islands only (Phase 2).

### A7. A Shardwright building with a real L0/L1/L2 hierarchy

The compendium's example is the 1,182-chunk two-storey building from Shardwright:
- 48 L0 components, 618 L1 fragments, 1,182 L2 chunks;
- 3,657 L2 bonds, 23 of them anchors.

**Import.**
- The `.fracphys` file is exported to `shardwright-stress/1` JSON by a small converter on Shardwright's `frac-io`: fragments of levels 0–2 with their parents, and L2 bonds with area, frame, second moments, joint kind and rebar.
- `stress-ref/src/shardwright.rs` reads it, and `scene_pack::import` dispatches to it.
- Each chunk keeps its ancestry as groups: `member:<L0>`, `l1:<id>`, and its role.
- 80% of the L2 fragments are compounds of 2–16 convex hulls (non-convex chunks), while an oracle chunk holds one hull. Chunks are therefore equivalent-inertia boxes: the asset's centre of mass, principal axes and principal moments per unit mass.
- In production the stress solver needs only mass, inertia and bonds, because PhysX owns the compound shapes and contact. Compound chunks in the oracle (for its standalone world) are a separate task.
- The asset's L2 mass, 115.6 t, matches the reference solve's anchor reactions: 1.136 MN against a weight of 1.134 MN.

**Found while importing:** the oracle's `ConvexHull::new` is not robust to f32-coplanar input. A clean 12-point Shardwright hull re-hulled after a translation alternates between 15 and 14 faces, with volumes differing by 2.6%. Separate task.

**Measurements:**

| | Value |
|---|---|
| True step | 570 substeps per frame (reference: 2,212), set by a 1.4 kg brick wall sliver (100% of the top mode) |
| Utilization at rest | max 0.50; 0.1% of bonds above 0.4; none above 1. Well calibrated: certificates can pass here. |
| Condensation by L0 members (48) | 62% of DOFs kept, gravity PCG 1,816 → 1,491 iterations, 4.3 MB, explicit step ×1.00 |
| Condensation by L1 (618) | 96% kept, ×1.00 |
| Coordinate bisection, parts of 64 / 256 | 57% / 34% kept, step ×1.30 |
| Slivers eliminated: 8 / 16 / 32 | step ×1.34 / ×1.45 / ×1.63 |

- **With a real member hierarchy, member condensation is exact but modest:** it removes 38% of DOFs. Walls and slabs are bonded to their neighbours along whole edges, so most of their chunks are boundary.
- **It does not raise the explicit step.** Its value is precomputed intact statics and location-aware bounds, not dynamics.

### A8. `patch_after_damage` A/B (prerequisite of an exact cold-bond path)

The full entry is in `IMPROVEMENTS.md`.
- 87/91 gated metrics both ways, with no regression.
- Catalogue wall time 178 → 156 s.
- Broken bonds move by under 4% except on b8_frame_sudden (23 → 11), a chaotic collapse whose reference ensemble already admits 17–29.
- Adoption is a model change awaiting review.

### A9. Null results: the mechanism, and when to revisit

A change that shows no gain is not rejected until the reason is known. It may act on
something that is not yet on the longest path, and becomes worth having once other
costs fall. Each null or negative result here, from this branch or others:

| Idea | Result | Mechanism (measured) | Revisit when |
|---|---|---|---|
| Partition / member condensation for the explicit step | step ×1.00 (packs, Shardwright members) | It eliminates 0 of the top-mode carriers (0/2 house-1story, 0/1 Shardwright): those slivers are always boundary chunks | **Already pays after sliver elimination:** house-1story 16 slivers ×4.51; + parts of 32–128 ×5.22; + parts of 256 ×7.65 (12% of DOFs kept). On the Shardwright building the gain stays at ×1.48: the next carriers (column pieces) are boundary too |
| Member condensation for statics | Gravity PCG 1,816 → 1,491 iterations (Shardwright); 769 → 323 (house, parts of 256) | Keeps the boundary graph's diameter, which sets block-Jacobi PCG's count | When the quasi-static tier (T1) and influence bounds are on the path: condensed intact statics as a precomputed GEMV |
| Eliminating one sliver | ×1.00 (house-1story, Shardwright ×1.01) | The top of the spectrum is a cluster: house-1story needs about 16 chunks treated for ×4.5. This reconciles `8c2851e` (mass scaling: "top mode spread") with A1 (one beam carries 100% of the top mode): the next modes are nearly equal | Any per-chunk lever (mass scaling, multirate, elimination) must treat the cluster, not the top chunk |
| Per-island stepping | ×1.00 on single-island packs | One island, one step | It pays on multi-island scenes: neighbourhood ×1.37, skyline ×1.29, skyline-stable ×1.09; more in towns that mix assets (brick-building 179 against a house's 1,264 substeps) |
| Damping off | ×1.02–1.09 | The top mode's own damping ratio is 0.02–0.09 | The *certified* step's damping (β_max · ω) costs ×3–6 on houses and ×1000+ on steel and stone packs (A1); a certified ρ(M⁻¹C) recovers it |
| Rotational inertia ×4 | ×1.0–2.1 | Top modes are partly rotational on slivers | After sliver elimination, re-measure (the remaining top modes may be translational) |
| Intact kernel, whole segment in one threadgroup | slower than a dispatch per substep above ~250 chunks (house-1story 12.4 against 9.4 µs) | One 256-thread group runs every bond: about 3.4 bonds per thread in series | A multi-group intact kernel, or CUDA cooperative launch (one SM per partition) |
| Cold-bond exact variant on the CPU oracle (`perf/oracle-runtime`) | 11% slower | CPU: an extra branch on a bond evaluation that is already about 100 ns | GPU: the cost is latency and code size (A2), and the intact-only path gains ×2.5–6 (A6). Not rejected for the GPU |
| Influence-line wake test (`perf/oracle-runtime`) | not pursued | No catalogue scene runs adaptive, so the oracle has no idle cost to save | GPU T0/T1 tiers (Phase 3) |
| Reduction cadence for free islands (`perf/oracle-runtime`) | < 1.5% of the oracle's time | The oracle has no barriers | GPU: each reduction is a barrier (about 110–130 ns) per substep on Metal |
| `frame_contact_step` (`perf/oracle-runtime`) | 7–20× *more* substeps | Its Gershgorin contact bound is looser than the global lattice bound | Not a lever: in the engine-coupled path the stress step has no contact term outside impact islands |
| Class A frame-level skip of quiet islands | Deferred | Profiled, 100 intact houses idle (60 frames, ~5 ms median): GPU 1.3 ms per frame, of which `island_frame` (quiet islands counting 434 substeps) is 0.65 ms; host motion download (`sync_motion` after every step, ~2 MB) 2.0–2.3 ms; frame tails (settled fatigue, settle countdown) 1.2–1.5 ms; readback 0.3 ms. The skip removes at most the 0.65 ms | After the host work goes: no motion readback without a consumer (the coupled path needs only fractures and events), settled fatigue only on watch-listed bonds. The skip is then the largest idle term |
| Coupled "true + sleep" in destruction | 36 ms median against 33 ms for "true"; sleeping saves nothing while loaded | Traced frame by frame: (1) the push changes the loads every frame, so the house stays active; the earlier 60-frame run's cheap half was the timer settle (`active_time` 0.5 s) firing at frame 30. (2) Adaptive mode dispatched `wide_wake` on every wide substep, but no wide island is ever settled (settled islands are built narrow; the micro tier lists `SETTLED_WIDE` as uncovered): an empty dispatch, about 3 µs per substep | **Fixed:** `wide_wake` only when a wide island is settled. 35.8 → 33.6 ms, bit-identical (quick gate 13/13; micro-tier GPU snapshots unchanged). The remaining +4% (0.9 µs per substep) is the per-chunk load recording that wake decisions need. A settling *certificate* (Phase 3) is what lets a loaded-but-safe building sleep before 0.5 s |

### A10. Why correction passes do not make punch-through (traced)

The same case runs through PhysX CPU and the oracle (`stress-physx`, `ram_into_free_wall`): a 200 kg ram at 40 m/s into a free 528 kg concrete wall. The reference world leaves the ram at 36.0 m/s with 3 fragments; through PhysX with up to 16 corrections at 1/60 s it leaves at 10.93 m/s.

**What is not wrong.** The rollback restores every body's pose and velocity, ram included (`pxb_capture_motion` / `pxb_restore_motion` with an empty scope cover all bodies). Momentum is not "spent", contrary to the earlier README and test comment, now corrected.

**What is wrong**, traced pass by pass (`STRESS_PHYSX_TRACE=1`, test `correction_front_to_split`):
1. **Acceptance granularity.** In the impact frame each trial breaks 70–100 bonds, but a pass accepts only those within one chunk wave-transit of the first: 1–3 bonds. They separate nothing, so PhysX's bodies and contact are unchanged and the redo repeats the same trial. With limits 4 and 16 the wall is still one body after the passes; it first splits at pass 17. With 64 passes the ram reaches 14.7 m/s. Under the native rule of at most one correction, punch-through cannot happen.
2. **The trial's input.** Each trial's fracture comes from PhysX's impulse against the intact rigid wall: a perfectly inelastic whole-wall collision, about 5.8× the real impulse, spread over the wall. Accepting the trial's breaks up to its first split (`front_to_split`, experiment, off) moves the ram only from 10.93 to 12.7–13.4 m/s at limits 1–16. That first split is a push-over-type break, not the plug: the ram still exchanges momentum with about 430 kg. Smaller engine steps give better trials (1/960 s, 4 corrections: 20.5 m/s), because each step's rigid impulse is smaller and closer to the true local load.

3. **Applying the trial's whole fracture** (the design the native pipeline describes: rewind, apply every bond the trial broke, re-step; `front_window = inf`, test `correction_accepts_whole_fracture`):

   | Correction | Ram after | Bodies |
   |---|---|---|
   | 1/60 s, limit 1 | 18.6 m/s | 86 |
   | 1/60 s, limit 2 | 20.8 m/s | 140 |
   | 1/60 s, limit 4 | 23.8 m/s | 179 |
   | 1/960 s, limit 1 | 18.7 m/s | 97 |
   | reference | 36.0 m/s | 3 |

   Momentum is exact in every run. The ram now cuts through, so this is far better than the first-front rule (10.93 m/s). But the wall shatters into 86–179 pieces instead of 3: the trial's fracture was computed from the whole-wall rigid impulse, so applying all of it breaks the wall everywhere that impulse loaded. More corrections break more without converging.

**Conclusion.** A correction can converge only if its trial's input resembles the real load, and a rigid contact gives the whole-body load until the plug is already free. Stress-owned impact islands break this circle: the stress solver resolves the chunk-level contact while bonds fail, and is right on the first pass (35.05 m/s, 6 fragments). The native pipeline, which uses Blast's CGNR solver with PhysX contact impulses and one correction, likely shares this limitation; not measured here.

### A11. Impact islands on the GPU (`GpuWorld::with_engine`), and which step is converged

The GPU world runs the reference's coupled frame on its mirror:
- the reference's own island selection;
- the engine step with island bodies frozen;
- the engine's contacts on driven clusters as GPU load terms;
- island bodies simulated by the GPU's chunk-level contact;
- verify-and-redo from a host snapshot.

Ram (200 kg, 40 m/s) through a free concrete wall, PhysX CPU, established methods (test `stress-physx/tests/engine_world_gpu.rs`):

| Step | Reference + PhysX | GPU + PhysX | GPU frames: impact / median / worst |
|---|---|---|---|
| established bound, safety 0.5 (default) | 35.05 m/s, 6 fragments, 24 bonds | 35.02 m/s, 6, 26 | 902 / 365 / 368 ms |
| **certified true step, safety 0.25 (converged)** | **35.07 m/s, 6, 24** | **35.07 m/s, 6, 24** | **539 / 186 / 188 ms** |
| certified true step, safety 0.5 | 34.84 m/s, 14, 93 | 34.84 m/s, 14, 90 | 381 / 140 / 167 ms |
| `with_true_step` (safety ×7.4 on stress *and* contact) | 33.15 m/s, 55, 240 | 31.64 m/s, 78, 272 | 701 / 60 / 87 ms |

For contrast, PhysX without islands gives 10.9 m/s.

**The GPU reproduces the reference coupled world at every step**, so the remaining differences between rows come from the step.

**Which step is converged** (`reference_refinement`, standalone reference, 0.1 s):
- Established bound at safety 0.5 / 0.25 / 0.125 / 0.0625: 6 fragments and 24–25 bonds at every safety. 39,876 substeps at 0.5.
- Certified true step at safety 0.5: 14 fragments, 93 bonds. **Not converged.**
- Certified true step at 0.25 / 0.125 / 0.0625: 6 fragments, 24 bonds. 23,394 substeps at 0.25, 1.7× fewer than the default.

**Mechanism.** Raising the stress bound leaves the penalty contact's global bound binding in island frames. At safety 0.5 that bound does not resolve a fracturing impact; it had looked converged only because the conservative stress bound sat below it.

**Consequences:**
1. The stress and contact bounds need separate safety factors, as v2 proposed. Here the contact needs about 0.25 and the stress bound 0.9 for stability.
2. **`stress_gpu::stability::with_true_step` is wrong for scenes with contact.** It scales the shared Courant safety, so contacts run 7.4× past their bound and the wall shatters (55–78 fragments instead of 6). The real-time harnesses (`stress-gpu-fps --config realtime`, `stress-gpu-coupled-fps`) use it, so their destruction fidelity is suspect. Use the certified `true_step` method with converged safeties instead.

**Exact cleanups found on the way** (bit-identical elsewhere: quick gate 13/13, micro-tier snapshots unchanged):
- A driven impactor did not count its substeps, so a frame in which the engine drove every impactor never ended.
- The contact pipeline ran whenever impactors existed, even when the engine drove all of them. It now runs only for bodies the stress solver owns: 460 → 365 ms per empty frame.

**Remaining cost.** The frames after the impact are 186 ms, with nothing breaking: about 3,900 substeps on the wall's largest fragment (a wide island at about 48 µs per substep). The levers:
- Sleep the fragments. Adaptive mode with a settling certificate, since a timer waits 0.5 s.
- Per-island steps for the fragments.
- The awake-cluster kernel cost (A6).

### A12. End-to-end peak with PhysX GPU (autoresearch, `examples/realtime_peak.rs`)

**Metric.** `PEAK_MS` is the worst complete frame over three regimes:
- house-1story fresh and idle;
- the 200 kg / 40 m/s wall ram;
- house-1story rammed by a 400 kg box at 12 m/s.

Setup: CuMetal PhysX GPU, established methods, certified true step at safety 0.25, adaptive mode, 0.25 s simulated, first frames included. This is an M3 Max proxy.

**Guard.** Each regime's ram speed, fragments and broken bonds stay within twice the spread of the reference coupled to PhysX CPU. The GPU gate's quick tier must pass 13/13, and the micro tier may not gain a failure (two predate this work). Every row below passes the guard. Log: `target/autoresearch-results.tsv`.

| # | Change | PEAK_MS | Mechanism (measured) |
|---|---|---|---|
| 0 | baseline | 7186 | House ram island frames: 17–19k substeps at the global contact bound |
| 1 | island frames bound only the contacts that can engage | 6039 | The global bound presses every chunk with the scene's stiffest material (rebar steel) |
| 2 | that bound as a block Gershgorin row over each chunk's actual partners (k/mₐ + k/√(mₐm_b)), pair stiffness direction-free | 4987 | The binding chunk was a 24 mm glass pane pressed by six copies of itself |
| 3 | **certified stress step from each bond's own dashpots** | 3050 | Leapfrog with lagged dashpots is stable while (h²/4)K + (h/2)C ≺ M. The certified radius of M⁻¹(K + (2/h)C) is iterated to its fixed point. The oracle's C ≤ βK (worst bond's β at the top frequency) cost ×3.4–6.1 substeps on the mixed packs: house-1story 4,730 → 1,228 per frame at safety 0.9, undamped 1,124 |
| 4 | contact plan rules out pairs by their swept boxes (15-axis test), with a run-time turn check | 2615 | Sphere candidates gave a flat pane 21–25 partners |
| 5 | a SIMD group per contact pair, lanes over its sample points | 2026 | One thread looped twice over about 28 points: about 125 µs per substep, latency-bound. Kernel budget accepted (+7.4 % code) |
| 6 | a resumed halt starts short batches too | 1467 | Bond breaks that leave the cluster connected left long batches launching empty: 14.4k rounds for 4,418 substeps |
| 7 | **fix: the step bounds the GPU's own, enforced contact plan** | 6087 | See below |
| 8 | pair stiffness over reachable directions; plans per quarter frame on ballistic paths; segment redo; two bookkeeping fixes | **2059** | See below |
| 9 | contact step from the contact network's certified radius; the solver stops at every plan it makes and the world re-steps | 1004 | Collatz–Wielandt on the block-norm network B (λmax(A) ≤ ρ(B)): a 200 kg ram on about 230 light candidates is a star, where row sums overstate ρ about 7.5× (test `network_tests`). Plans made within a step (split, re-plan, stride rebuild) stop it before anything runs under them, so there is no segment redo. Re-plan test made relative to the step's span |
| 10 | five dispatches per contact substep instead of seven | 1103 (noise) | sums with impactors, narrow islands with wide chunks (one kernel per threadgroup). Debris frames 540 → 450 ms. PEAK held by the fracture frame's 256 host round trips |
| 11 | **wide islands check their connectivity on the GPU** | 717 | Label propagation with label jumping in threadgroup memory (up to 4096 chunks). The fracture frame had 130 resumed halts (bond broke, cluster whole), each stopping every island for a host round trip: 256 → 80 batches, 1000 → 695 ms |
| 12 | narrow islands too | discarded (749) | Only 2–8 narrow resumed halts per frame here, and the 16 KB label array lowers `island_frame` occupancy. Combined with wide threshold 400: wall 578 vs 678 ms, house fracture 783 vs 775 ms, so PEAK unchanged |
| 13 | impactor candidates on a SIMD group each | **624** | Two serial loops over about 28 slots per candidate thread: fracture 695 → 625 ms, debris 450 → 410 ms |
| 14 | no snapshot per island-frame segment | discarded (629) | Download time unchanged (37 ms, from per-step motion syncs and split downloads) |
| 15 | persistent contact pipeline in one threadgroup (`contact_frames`) | discarded (1932) | Outcomes match the rounds (wall identical, house 32 fragments and 185 bonds), but house debris 395 → 1761 ms. One core runs it all, and the per-island sums by an owner thread serialize device latency (about 100 µs per sum over the 290-chunk house). Even with SIMD-parallel sums, one core bounds it at about 60–90 µs per substep against about 90 now across 40 cores |
| 16 | wide bond groups with the sums when no wide island is free or settled | 565 | 4 dispatches per contact substep for anchored wide islands (house debris 395 → 348 ms) |
| 17 | **narrow islands up to one chunk per thread** (wide threshold 192 → 257), with their own 512-label GPU connectivity check | 524 | The 220-chunk wall runs narrow (3 dispatches instead of 7): wall impact 572 → 473 ms. Micro coverage kept: `RESUMED_ON_DEVICE`, and `resumed_halt_host` for the host fallback |
| 18 | **host readbacks gathered into one round trip** (`Gpu::read_ranges`) | **448** | A download took up to ten blocking reads and a batch three, including the whole scratch buffer. Fracture frame host time about 170 → 77 ms |
| 19 | chunk-level island criterion | discarded | Islands unchanged (32–33). The debris frames hold about 66 genuine chunk-level approaches at 0.5–3 m/s with peak force above the weakest bond's capacity: the collapse is in motion. Releasing debris needs a damage-free certificate with influence bounds (the weakest-bond comparison is not conservative under lever arms) |
| 20 | separate stress safety (GPU-only probe) | parked | At 0.5: house 3.04 m/s, 35 fragments, 196 bonds (reference 3.04, 34, 196); house worst 448 → 302 ms, wall median 125 → 64 ms. PEAK held by the contact-bound wall impact frame. At 0.9: 2.62 m/s, drifting. Adoption needs an oracle convergence study (stress safety separate from contact safety) and a decision on the benchmark's configured safety |
| 21 | rigid-rotation contact plans; adaptive shorter plan horizons | discarded | Rotation: same substeps and batches (the wall impact is bound by the punched plug sweeping past 15–79 hole-edge chunks per horizon, not by budgets), while +32 B per chunk and the rotation in `travel_check` cost about 10%. Horizons: tiny segments, 14 s; needs a cost model |
| 22 | contact-plan fields (64 B) out of `ChunkStatic` into the scratch | about −5% (A/B) | Same-session A/B under other jobs' GPU and CPU load: house worst 605/597 → 564/572 ms, wall impact 564/556 → 498/530 ms |
| 23 | `ChunkStatic` read field by field instead of copied whole | discarded | No difference: the compiler already loads only the fields used. #22 gained from the smaller per-chunk footprint |
| 24 | **benchmark: stress step safety 0.5, contact 0.25** (`sim.stress_courant_safety`, user-approved) | 443 (PEAK now the wall impact) | Study below; house worst about 448 → 298 ms |
| 25 | contact-bound segments weigh shorter plan horizons by measured cost | 395 | Host-only plan previews; a shorter horizon is taken while it lowers (cost per substep)/step + (cost per segment)/horizon. The punched plug sweeps past fewer hole-edge chunks: wall impact 451 → about 390 ms. A version maximizing the step alone collapsed into tiny segments (14 s) |
| 26 | **benchmark: contact step safety 0.35**; the guard's reference held at its converged 0.25 | 302 | GPU study `gpu_contact_safety_convergence`: outcomes the same from 0.125 to 0.5 (wall 6 fragments, 24–26 bonds, 34.99–35.08 m/s; house 3.03–3.06 m/s, 33–35 fragments, 188–195 bonds). 0.35 keeps the wall ram's speed clear of the tolerance. The coupled reference's global contact bound is not converged at 0.5 (A11), so the GPU is checked against the converged truth |
| 27 | **fix: impactors' travel budget enforced on the GPU** (`impactor_travel_check`) | 293 | Impactor candidates assumed travel within 2× speed × horizon, unchecked. The sharper variant (ballistic impactor plans, relative sweeps, direction-aware impactor bounds) failed the guard: wall ram 34.98 vs 35.07 ± 0.085 m/s, with post-punch contact steps about 20% larger, as contact safety 0.5 did |
| 28 | **narrow contact islands' bonds in the sums dispatch** | 250 | Bond groups of up to 256 bonds (as wide islands have); the contact path's island kernel is the bond-free instantiation (`island_frame_at<0>`). The 220-chunk wall had 2 joint laws per thread in one group: wall impact 295 → 240 ms, house worst 268 → 248 ms, outcomes identical. (A first version cleared wide islands' halt flags in the narrow kernel's tail: no splits, caught by the house ram) |
| 29 | one joint-law call site for narrow and wide bond groups | 249 | `sums_impactors_bonds` 65 → 45 KB code, spill 1008 → 832 B |
| 30 | narrow islands up to their connectivity check's 512 chunks | 233 | The 319-chunk house runs narrow (3 dispatches per contact substep): house worst 245 → 231 ms |
| 31 | a pair's separating-axis test spread over its lanes | about −3% (A/B) | Wall 235 → 224–230 ms, house median 139 → 134 ms |

**The certification gap found at #7.**
- Rows 1–6 bounded the contacts of a fresh one-frame plan whose travel budgets nothing enforced. The GPU ran its own four-frame plan, with larger budgets and more pairs.
- Logging every plan made inside a frame (re-plans, host splits) showed real violations. In the house ram fracture frames, splits created pairs that needed 2.66e-6 down to 1.27e-6 s, while those frames ran 3.77e-6 to 3.0e-6 s.
- The outcomes had stayed within the reference's spread, but the step was not certified. Since #7 the step bounds the plan the GPU enforces (`travel_check` halts, then the host re-plans). Every plan made within the frame is checked, and a violated segment is solved again at the step it needs.

**What #8 changed to win the cost back (all exact):**
1. **Directions.** A pair's penalty stiffness follows the line between the centres. Within a plan, that line stays within asin(slack / distance) of the middle of the pair's relative path. Each box turns by at most budget/radius, which the GPU checks for every pair chunk. Each direction cosine is therefore within ψ of its planned value, and the projected areas and half-thicknesses are bounded with it. Coplanar panes split along an edge no longer count as pressed through their 24 mm thickness.
2. **Ballistic budgets.** Budgets bound deviation from x₀ + v₀t + ½gt², not absolute travel. Pairs sweep their relative path, so fragments flying together are not candidates. Wall ram: 1,718 → 472 candidate pairs.
3. **Plans per segment.** Each quarter frame plans and bounds its own contacts, so budgets shrink with the horizon. A violated segment is redone alone.
4. **Pre-existing bugs fixed.**
   - `step_times` kept the times of a state's earlier run. A state restored and run at another step size had stale times: `mirror.time`, probes, and the new plan clock (it looped forever).
   - After a host split, the plan horizon took the substeps left of clusters already split away (2.8× too long).

**Where the peak is now, after #31 (PEAK about 228 ms; 31× the baseline).**
- **Wall impact and house fracture frames, about 225 ms each.**
- **House fracture frame.** About 2,210 substeps at the stress step, about 70% GPU (about 60–74 µs per substep over 3 dispatches: contact forces 20–25 µs, island kernel 23 µs, sums and bond groups 15–17 µs). Host about 30%: 21 host splits with their downloads, rebuilds and readbacks.
- **House debris frames, about 134 ms median.** Wall debris frames 64 ms; idle median 3.2 ms.

Remaining levers:
- **Splits on the GPU.** About 20 ms of split work per fracture frame, plus readbacks.
- **Cold-bond fast path.** Needs the `patch_after_damage` review.
- **Kernel micro-architecture.** `contact_islands` is still 48 KB with 432 B spill; GPU counters would guide it.
- **Per-island multirate steps; certificates for sleeping and debris release.**

**Stress-step safety study (#24).** stress-ref gained `sim.stress_courant_safety` (optional and never serialized, so existing scenes and caches are unchanged): the bond network's fraction of its certified critical step, while contacts keep `courant_safety`. Refinement and spread runs halve both (`SimDesc::scale_safety`).

The reference coupled to PhysX CPU (test `stress_safety_convergence`), house ram after 0.25 s:

| Stress safety | Ram speed | Fragments | Broken bonds | Note |
|---|---|---|---|---|
| 0.125 | 2.63 m/s | 37 | 195 | stress step binds |
| 0.25 / 0.5 / 0.9 | 3.04 m/s | 34 | 196 | contact-bound: identical |
| 0.25, ram +1e-4 | 3.02 m/s | 34 | 197 | |

The wall ram is identical at every safety (contact-bound).

The GPU world (test `gpu_stress_safety_convergence`), house ram:

| Stress safety | Ram speed | Fragments | Broken bonds |
|---|---|---|---|
| 0.125 | 3.06 m/s | 34 | 189 |
| 0.25 | 2.99 m/s | 32 | 185 |
| 0.5 | 3.04 m/s | 35 | 196 |
| 0.5, ram +1e-4 | 3.05 m/s | 36 | 195 |
| 0.9 | 2.62 m/s | 36 | 200 |

From 0.125 to 0.5 the GPU outcomes lie within the reference's own step spread, and 0.5 is closest to the reference. 0.9 lands on the 2.6 m/s branch. The benchmark now runs stress 0.5 and contact 0.25, and the guard's references and spreads were rebuilt for it.

**Where the peak was, after #22 (PEAK about 448 ms uncontended).**
- **House ram, fracture frame (about 444 ms).** About 83% GPU: 4,420 substeps at about 83 µs over 4 dispatches.
- **Wall ram, impact frame (about 444 ms).** About 5,800 substeps. Contact-bound while the punched plug sweeps the hole edge.
- **Debris frames (about 330 ms).** The collapse is genuinely in motion.
- **Idle house.** Median 3.5 ms.

Remaining levers:
1. A separate stress safety (#20): needs an oracle convergence study and a configuration decision.
2. A cost-model plan horizon for fast fragments (#21).
3. Restructuring the island kernel (#15 shows one-core persistence cannot win; dispatches are at their floor without a device-scope fence on macOS 14.4).
4. Moving host splits onto the GPU (20 per fracture frame, about 2.5 ms each).

**Where the peak was, after #13 (PEAK about 620 ms).**
- **Wall ram, frame 0, about 600 ms.** About 5,800 substeps, mostly at contact steps of 2.1–3.8e-6 s from fragments in contact.
- **House ram, fracture frame 1, about 610 ms.** 4,420 substeps at the stress step, 20 host splits.
- **Debris frames about 400 ms.** Per substep (timed): islands with wide chunks 36 µs, contact forces 27, wide bonds 18, sums with impactors 15, wide end 8.
- **Idle house.** First frame 18 ms, median 4.5 ms.

The substep counts are now set by the physics (certified stress and contact steps at safety 0.25). The cost per substep (about 90 µs across 5 dependent dispatches) is the remaining multiplier. The next architectural step is lever (a) below: the contact pipeline of every contact island (contacts, sums, bonds, chunks, rigid motion, connectivity) in one persistent threadgroup over many substeps, barriers instead of dispatches. A2's per-substep floor (about 8 µs) suggests 5–8× on these frames.

**Earlier profile, after #8:**
- **House ram.** Every island frame runs at the stress step (4,420 substeps of 3.77e-6 s) with no redo. Fracture frame 1.05 s; debris frames about 0.54 s (about 120 µs per substep over 7 dispatches).
- **Wall ram, frame 0, 2.4 s.** The impact segment's plans still need 8.4e-7 s once the wall shatters at 46 m/s. Later frames: 0.22 s at the stress step outside islands.
- **Idle house.** First frame 39 ms, median 8.7 ms.

**Next levers, by expected size:**
- **(a) Persistent multi-substep kernel for small contact islands.** All debris (about 30 clusters, about 220 pairs) fits in one threadgroup. Today each substep costs 7 dispatches of about 15 µs latency.
- **(b) Debris islands for the rest of the run.** The island criterion seeds an island whenever the Hertz contact time is below the struck body's wave round trip, which holds for any resting contact on the anchored house. Changing it changes the oracle's model, so it needs the standalone oracle as the truth, not the coupled reference the guard uses.
- **(c) The impact segment's contact bound.** Per-pair Gershgorin over many sample-point partners. A sample-point-level bound would be sharper.

## B. Review of the two documents

### Compendium

| Idea | Verdict | Why |
|---|---|---|
| Regime split by time scales | Principle adopted; rule replaced | The oracle's regime map: `t_c ≤ T_w` misses punch-through, which is a critical impact speed per wall type (10–20 m/s). |
| Guyan condensation | Exact statics; no explicit-step gain by partitions (A3) | Useful for GEMM-shaped intact solves and over-size islands. Sliver elimination is the dynamic variant worth having. |
| Craig–Bampton, mode acceleration, Ritz | Deferred | Only if condensed dynamics pay. |
| Influence-line / modal "definitely safe" bounds | After E_safe | Location-aware; still certifies on buildings with overloaded joints (A2). |
| SMW reanalysis, VDM | Deferred | f32-fragile. The oracle's quasi-static cascades already use direct solves. |
| Sequentially linear cascades | Deferred | A model change in cascade order, to be judged with `fidelity.py`. |
| Energy-balance sudden loss; limit analysis | Screens only, later | |
| P–I / SDOF | **Tier router, never the damage result** | The oracle's speed sweeps are an empirical P–I boundary. |
| HHT-α | Oracle switch; GPU tier after multigrid | κ improves only about 40× at 1/60 s; f32 residual about 6%. |
| VBD / AVBD / XPBD | Rejected for stress | Bond forces are unconverged multipliers. |
| Multirate | Asset or sliver treatment first (A1: ×1.9–4.3) | IMEX interface is research. |
| Corotational bonds | Fidelity track | Catenary arrest. |

### Speed-of-Light v2

| Idea | Status | Verdict |
|---|---|---|
| True step per island | GPU: global, estimated | Per-island n_i; certified damping bound (A1) |
| Persistent on-chip blocks | Narrow islands in device memory | **Intact on-chip kernel first:** the 8.4 µs per-substep floor × about 1,300 substeps ≈ 11 ms is the binding cost on Metal |
| Cold/hot bonds | Not done | Needs `patch_after_damage` and `exact_rate_filter` adopted |
| Neighbour-flag partitions, cooperative launch | Impossible on macOS 14.4 | CUDA phase |
| Certificates A/B/C | Heuristics today | Class A frame skip; E_safe where A2 allows it |
| Smoothed-aggregation multigrid | Not done | Needed for big quasi-static islands and any GPU implicit |
| GPU connectivity | Host mirror | Moved up: host round trips sit on every cascade |
| Scene fixes | Import fixed (A1) | Calibration open (A2) |

### What both miss

- **The GPU is gated against `established` contact while the oracle's truth is `proven`.** The GPU world rejects `layer_contact` scenes.
- **The engine handover.** Through PhysX (one correction), no configuration punches through (`14352f570`). Only stress-owned impact islands reproduce fast-versus-slow loading, and the GPU coupled solver has none yet.
- Host round trips per split.
- The GPU clock ramp after idle.
- Chaotic outcomes need ensemble gates.
- Layer contact's unbounded LIFO cohorts are hostile to the GPU.

## C. Target architecture

| Tier | When | Method |
|---|---|---|
| T0 asleep | Certified: class A unchanged inputs, class B fatigue schedule, class C E_safe or influence bounds | Zero work |
| T1 quasi-static | Slow and sustained loads, settling | GPU PCG (small), multigrid (big), offline f64 prestress |
| T2 structural dynamics | Below the calibrated critical impact speed, support loss, sway | Explicit intact kernel at the true step |
| T3 explicit full law | Hot bonds, damage, punch-through and spall | Full-law island kernel; local windows later |

## D. Roadmap, revised by the measurements

What the measurements (A1–A8) change:

- **Destruction frames in the standalone GPU world are dominated by debris contact at the stress substep (A5).** In production that contact is PhysX's. The real-time path is therefore the engine-coupled solver, and its missing piece is stress-owned impact islands. That piece is also what makes fast loading (punch-through) right (`tier2_velocity`).
- **The cost of one awake building is its true substeps times the per-substep latency** (A5, A6): 1,264 × 25 µs ≈ 32 ms for house-1story on M3. Three measured levers multiply:

  | Lever | Factor | Status |
  |---|---|---|
  | Intact-only kernel | ×2.5–6 | measured with the existing intact kernel |
  | Sliver elimination | ×1.6–4.5 | a step and model change to prove in the oracle |
  | Per-island step | ×1 on single buildings; pays in mixed towns | |

  Together they bring an awake house to about 2–4 ms. So a few awake buildings fit per frame on M3, and everything else must sleep or be quasi-static.
- **Idle is already cheap** (100 intact houses 3.4 ms median, one coupled house 1.6 ms), apart from the startup frame (93 ms).
- **Certificates can pass on well-calibrated buildings:** the small packs and the Shardwright building (max 0.50 at rest). The multi-storey packs are over strength at rest and need their calibration fixed first. Shardwright's own fix is a design check that reinforces bonds to 2× their self-weight load.
- **Member condensation is exact but gives no explicit-step gain** (A3, A7). It moves to later, as an intact-statics and bounds tool.

| Phase | Content | Gate |
|---|---|---|
| 0 | Baselines and blockers | done: A1–A8, import fix, tools |
| 1 | **GPU impact islands in the coupled solver** (port `world_coupling` `select_islands` and stress-owned contacts), with debris contact left to the engine | b5/b7 speed sweeps through the GPU coupled path reproduce push-over against punch-through (`tier2_velocity` cases); destruction frame time measured idle and at peak |
| 2 | **Awake-building cost:** adopt `patch_after_damage` (review); intact island kernel with escalation to the full law; sliver elimination as an oracle switch, then on the GPU; per-island step | `fidelity.py` on the switch; one awake house ≤ 4 ms on M3 |
| 3 | Faithful sleeping: class A frame skip, E_safe and settling certificates (shadow first); the startup frame | Never contradicted in shadow; first frame visible and bounded |
| 4 | Asset calibration: a design check for the packs (as Shardwright's), or their materials | No bond over strength at rest |
| 5 | GPU connectivity and splits (no host mirror inside a frame); contact-state memory bounded (the 4.87 GB crash) | Cascades with no host round trip |
| 6 | Proven oracle methods on the GPU (contact crushing, layer contact with bounded cohorts) | GPU gate against `proven` |
| 7 | Impact router (critical speed / P–I maps); multigrid; HHT; condensed intact statics and influence bounds | Each by its IMPROVEMENTS class |
| 8 | CUDA host at `PxgDestructionRuntime`; 4090 qualification | AGENTS.md idle plus peak complete-step rules |

**Open bugs found here** (separate tasks):
- micro tier: `coupled_impulse` and `coupled_sleeping` fail on the integration base itself (GPU 4 fractures against the reference's 1), before any change here;
- the oracle's convex hull is not robust to f32-coplanar input (A7);
- GPU `contact state` buffer overflow on collapse (A4);
- the certified step's damping term (A1).
