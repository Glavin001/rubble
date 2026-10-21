# Learned impact-outcome stress solver: verified review and research plan

## Context

**The goal.** Replace the explicit impact window, which costs about 50–500 ms per impact frame, with a learned model. The model must be as accurate as the oracle within its own run-to-run spread, and fast enough for real time:
- ≤ 2 ms per impact call;
- every M6 frame target still met;
- the target scale: 10k+ active bodies and many concurrent impacts.

**Why earlier work failed.** A colleague's V2–V39 never produced an accurate, fast replacement. The latest attempt, V39, learned the per-chunk velocity kick at the native ~4.5 µs substep. That made it about 10× slower than the solver it replaced. It also overfractured (100 against 85/87 breaks) and was trained on a single trajectory.

**What this plan builds on.** It keeps the boundary you chose in the draft (`docs/learned-stress-solver/learned-stress-solver.md`): one model call replaces the whole explicit window. It corrects the draft's errors, adds what independent research found, and orders the work so that the expensive steps come after cheap tests that could kill the idea.

**Scope** (your answers):
- compute is the **M3 Max only**;
- the first acceptance is on **rams and walls** (free wall, wall row, procedural walls), then houses. Vibe-town comes last, because it waits on the Shardwright re-export.

## Decisions (2026-10-20, after P0)

Your decisions on P0's five open questions (`engine/stress-oracle/learned-impact/findings/p0-feasibility-2026-10-20.md`):

1. **Acceptance is against the oracle ensemble**, not the single symmetric runs. M5's 35.0/6 and 23.6/133 are restated as their ensembles: free wall 31.98 m/s [30.97, 33.12], 78 fragments [67, 95]; wall row 22.95 [21.81, 24.25], 200 [178, 220].
2. **The skill rule is calibrated by the oracle itself:** a candidate passes a metric when its skill ratio is within the range a held-out oracle member reaches against the other 15, which is 1.0–1.42 in P0. The literal 1.25 bar would fail the oracle's own members.
3. **The 2 ms cost gate is relaxed for now.** Measure and report the cost; accuracy comes first. The P0 operating points (Metal 3.0 ms at 10k chunks, W32 f16, 8 rounds; about 1–1.3 ms for M5-sized islands) are the reference. Revisit the gate after P2.
4. **The energy cap is `KE_out ≤ E_in + W_gravity` with a tolerance of about 5% of E0**, matching the penalty contact's measured ledger residual. The oracle keeps the penalty contact, and `L_energy` uses kinetic-energy changes and per-bond dissipation, not booked contact dissipation.
5. **Routing is extended by the reach of debris launched within the frame**, so a plug punched mid-frame is routed together with the structure it reaches. This sits behind a flag, off by default, measured on the wall row; it closes P0's miss for every path, learned or not.

## What I verified, and what was wrong

### Code facts (audited on main and on 5a2f5e4f)

- **The engine frame is 1/60 s, not 1/120** (`engine-schema/src/params.rs:52`, `scene.rs:827`). The draft's "8.33 ms tick ≈ 1,840 substeps" should read "16.7 ms, about 2,500–14,000 substeps". The free wall's step is 6.7 µs before contact and 1.2 µs once the ram's 236 contact pairs engage, about 5,350 substeps in its impact frame. **The contact law, not the bonds, sets the step during an impact.** The 120 Hz cadence stays as a configurable evaluation target (the handoff's ≤ 5 ms at 120 Hz).
- **The M5 scenes** (`stress-rubble/src/scenes.rs:25-77`):

  | Scene | Chunks | Bonds | Ram | Truth (`findings/rewind-scaling.md:175`) |
  |---|---|---|---|---|
  | Free wall | 220 | 508 | 200 kg steel, 40 m/s, zero gravity | 35.0–35.1 m/s, 6 fragments |
  | Wall row (two walls) | 440 | 1,016 | same ram | 23.6 m/s, 133 fragments |

  The "112 bonds" figure belongs only to V39's 64-chunk fixture.
- **What the GPU punch actually does** (`punch.slang`):
  - it finds the struck chunk, grows a 45° cone plug and sums the capacity of the bonds holding it;
  - a Hertz-time impulse test decides whether those bonds break;
  - it writes `damage = 1` and the DISCONNECTED bit, and the split pass in the same submission splits the island.
  - **It sets no velocities.** Rubble's rigid contact against the freed plug produces the ram's exit speed.

  That is why the free wall comes out at 25.4 m/s against 35.1, and why the wall row's dynamic shattering (133 fragments) is missing.
- **How a model's outputs can reach the GPU path:**
  - write `bond_dyn` damage plus the island status bit, as the punch does;
  - write the chunks' hidden velocities `state[4c+2]` and `state[4c+3]` *before* the split. `split.slang:240-290` then gives each child its momentum-conserving mean velocity.
  - Nothing sets the impactor's velocity today; that needs a new engine-body write.
- **Inputs that already exist:**
  - pre-contact approach speed, the other body's mass, modulus and radius (`api.rs:32-46`, `engine.rs:105`);
  - per-bond `utilization` and `record_peak_utilization` (`statics.rs`, `joint.rs`).
- **Implicit and coarse-step routes are already measured and fail** in this codebase:
  - Newmark at 100 µs gives 28.8 m/s / 163 fragments against 35.1 / 6;
  - AVBD is faithful only up to about 25 µs, and only with the penalty contact inside the solve;
  - a coarse window step loses the wall row at k ≥ 1.5;
  - the modal pre-window passes on accuracy but runs 2–23× slower.

  So "learned corrector over a cheap implicit step" starts from a large error, and its cost floor is still many dependent steps. It stays a fallback, not the main line.
- **The CPU oracle is deterministic for any thread count** (`par.rs`). Running scenes in parallel is the throughput lever: about 0.3 s CPU for a free-wall impact frame, 1.2 s for the wall row, 2–15 s for a house frame.

### Literature corrections (both research sweeps checked against primary sources)

- **arXiv 2601.05667** is about the turbulent breakup of colloidal aggregates, not impacts. Its recipe is still the most transferable published one: a GNN regresses the **peak tensile force per link** over the event, and the real rupture threshold is applied afterwards.
- **MGN-T (2601.23177).** The 1.5 ms is the interval it learns, not its compute time. The full step costs 15 ms for 16k nodes on an RTX 4090; the network alone is 4.1 ms. Per-step whole-scene GNNs do not fit our budget.
- **"DGAR"** could not be found. The projectile-penetration paper (IJIE 2025) is still unread.
- **The RC blast-fragmentation paper is now read in full** (Li, Wang, Chen, Li, Hao, *Eng. Struct.* 308:118009, 2024, CC-BY; your PDF). Its "Fragment Graph Network" (FGN):
  - is **GNS** (Sanchez-Gonzalez 2020) on SPH-like particles: a radius graph (r = 15 mm), 5 message-passing steps, two-layer 64-wide MLPs, ReLU and LayerNorm;
  - takes the charge mass and position as a **global node**, and scaled distance as a per-particle feature;
  - predicts acceleration (forward Euler) and the **non-negative increment** of effective plastic strain, so damage is monotone;
  - steps autoregressively over 35 saved steps of 2 ms, trained one-step with Gaussian input noise σ = 5e-4 (grid-searched), Adam with a cosine schedule from 1e-3 to 1e-6, 100k iterations, batch 1, about 24 h on one TITAN RTX.
  - **Data:** 135 LS-DYNA ALE-FEM-SPH cases (105 train / 15 val / 15 test, plus 14 interpolation and extrapolation cases), about 185k particles each. The grid is narrow: 2–6 kg TNT, 300–500 mm standoff (Z = 0.17–0.40), C30/C50/C80, three slab and rebar layouts.
  - **Results:** about 6% MAPE and 1.6 mm RMSE on fragment-particle positions at the last step, R² ≥ 0.93 even on extrapolation. Fragment-size histograms and Weibull-fitted cumulative mass match closely, as does peak fragment velocity against scaled distance.
  - **Cost:** 5–12 s on a GPU for the whole 2 ms run, about 0.15–0.35 s per step. Not real time.
  - **Limits the authors state:** no conservation of energy or momentum, error that grows with each step, fragments identified only after the fact from separation (Δd > 0.01d), and slabs only.
- **Perera et al.:** the "region-active factorization" does not appear in the paper. The papers do use staged prediction (SIF, then which cracks grow, then the tips).
- **DeepFracture** takes 6–28 s per pattern, and a phase-field DDPM 3.6 s per step. Generative decoders are out for real time. Their lesson is still important: **deterministic regression smears chaotic branches, and thresholding the smear under-fragments.**
- **Calibrating the evaluation:**
  - McGreivy & Hakim (NMI 2024): most ML-for-PDE papers beat weak baselines. Our baselines must be the punch, the QS solve with Hertz peak loads, and the oracle's own spread.
  - A 2026 preprint finds that disagreement in GNN deep ensembles collapses. Use ensemble variance as a fallback trigger only together with physics residuals and input-range checks.

## What each source teaches us: do, don't, and what we do differently

### A. The colleague's failed attempt (V2–V39), lesson by lesson

| Experiment (handoff §) | What happened | Lesson: do / don't | What this plan does differently |
|---|---|---|---|
| V39: graph-state kick at every native substep (§11) | 6,649 calls per 30 ms; 23 s against 2.3 s for the oracle; 100 breaks against 85/87 | **Don't** learn a quantity that is cheap natively, at the native step. A speed-up has to remove substeps; on the GPU the floor is the dependent chain's latency | One call per routed impact island per frame, replacing the window's 450–5,350 substeps. The P0 cost envelope is measured *before* training, so the model's size is set by the budget |
| V39 trained on one 12 m/s trajectory, validated at 16 m/s (§11.3) | Train/val gap of 77×; per-bond geometry features let it memorise bonds | **Don't** treat correlated rows as data; **don't** feed identity-like features | Around 30k events across procedural families; no absolute positions or identity features; family holdout plus a sealed set; 3 seeds |
| V39 graph net and material net J trained separately (§11.1) | Motion and material response could contradict each other; overfracture came from the graph feeding J excess strain | **Do** derive every output from one consistent mechanism | One network predicts velocities, the impactor's Δv and the energy split; an exact projection makes them conserve momentum together, and a consistency term ties them to the energy ledger. Breaks come from one demand-ratio head; the decode is deterministic |
| Wrong reference: stress-ref defaults, not the fidelity configuration (draft audit) | It imitated a fourth configuration that matches neither the GPU nor the M5 truth | **Do** pin the oracle first | P0 step 1 pins the configuration that reproduces 35.0/6 and 23.6/133, and explains the CPU/GPU wall-row gap |
| V25/V26: exact oracle endpoints through the adapter (§7.2) | Even perfect endpoints gave wrong motion (car 8.6 against 13.4 m/s); fragment kinetic energy lives in the rigid frames | **Don't** blame the network before testing the boundary with perfect outputs | P0 step 3 is the perfect-model ceiling on the *new* boundary, with variants. It is a gate: no training if it fails |
| V31–V36: forecasts fixed at frame start, catastrophic in independent runs (§7.3, §8) | 1,258 breaks against 324; largest fragment 5.97% against 66–88%, yet the projectile was only 0.5 mm off after 8 ms | **Don't** accept on short projectile error; **don't** trust teacher-forced loss | Gates use fragment distribution, support survival and exit speed in independent runs. Contact feedback inside the frame is handled by including every contact of the island in the call (pre-contact velocities and timing), not by a stale force forecast |
| V32/V34: corrective branches and naive mixing (§7.3) | Oracle continuations from damaged states also over-fracture; >99% of one loss came from a signed bookkeeping field | **Do** use on-policy data, labelled from *that* state; **don't** let one field dominate the loss; **don't** clip a signed quantity to hide it | DAgger restarts the oracle from the model's decoded reduced state, the state the engine really holds. Loss terms are normalised and their gradients inspected |
| Legacy normalisation erased load amplitude; spin missing (§12.2, V22) | Doubled loads encoded identically; ±20 rad/s spin pairs indistinguishable | **Do** audit observability before adding data | Nondimensional but amplitude-preserving inputs (energy ratio, v/c, d/c), spin on contact edges. P0 adds a "same input, different outcome" check over the recorded events |
| V38 material models A–L (§10) | F passed 2/5; H passed 3/5 alongside a 175.7 m error; freezing history gave zero breaks | **Don't** let pass counts hide catastrophes; **don't** keep tuning a component when the whole is the question | Worst-case reporting next to medians; every gate applies per scene, not as an average |
| Uniform row caps dropped rare fractures (§10) | Kept 17 of 64 new fractures | **Do** preserve rare transitions | The sampler oversamples arrest/penetration boundaries and break events; 20% no-break events are kept on purpose |
| Reused "test" cases (§2.5) | The 16 m/s case picked the checkpoint | **Don't** reuse validation as a test | The sealed confirmation set stays unopened until P4 |
| What worked: V27 provider control, V37 structured operators, exact recording | Exact reproduction when the oracle supplies responses; structured physics generalises within its domain | **Do** keep the controls and the physics structure | Reuse the recorder/replay and comparison tooling; build physics in (edge impulses, real strength threshold, energy cap) |

### B. The draft plan (`learned-stress-solver.md`)

- **Kept:** the one-call boundary; the P0-before-ML ordering; ensemble chaos floor; the momentum projection; the flag next to the punch; the 10-buffer weight packing.
- **Corrected:**
  - the frame is 1/60 s, not 1/120;
  - the contact law sets the impact step (1.2 µs), not the bonds;
  - the punch sets no velocities, so a velocity write path is a new requirement;
  - a separate impactor-Δv head is replaced by the sum of contact impulses.
- **Added:**
  - the GPU cost envelope as an early gate;
  - a gradient-boosted tree baseline;
  - the threshold-against-sampling question;
  - the routed impact island as the unit, not the structure.

### C. This codebase's own measurements

| Finding | Lesson | Use in the plan |
|---|---|---|
| Implicit Newmark at 100 µs: 28.8/163 against 35.1/6; AVBD faithful only ≤ 25 µs; a coarse window step fails at k ≥ 1.5 | **Don't** expect a cheap large-step solver to be a good base for a corrector | Learned-corrector route kept only as a fallback |
| Rubble's rigid contact gives the ram's whole momentum to the intact wall (`rewind-scaling.md`) | Breaks alone, applied after the contact substep, cannot fix exit speed | Outputs include velocities and impactor Δv, applied before the engine steps |
| The punch: one plug, rigid exchange, 25.4 m/s and 19 fragments | **Do** reuse its plumbing and its plug as a feature; **don't** fit parameters to the reference | Punch plug and capacity ratio are input features; the punch is a baseline and the fallback |
| The modal pre-window: exact, but 2–23× slower | Exactness is not enough; cost decides | The cost envelope comes first |
| The CPU oracle is deterministic for any thread count | Run scenes in parallel | About 10k wall events a day on the M3 Max |
| Wall-row outcomes vary across valid configurations (133, 124, 41, 17, 6 fragments) | The wall row is sensitive: a single trajectory is a poor target | Ensembles and spread-relative gates |

### D. The literature

| Source | Do | Don't | How we apply it |
|---|---|---|---|
| Buzzicotti et al. 2026 (arXiv 2601.05667), colloid breakup GNN | Predict the event outcome from one state; regress peak link force, then apply the real threshold | Learn an uncalibrated "break" logit | Per-bond peak demand ratio, threshold τ = 1, calibrated on validation only |
| MGN-T (2601.23177), MeshGraphNets, MS-MGN, BSMS-GNN | Learn at the output interval (≈2,400× the explicit step); give the model enough reach (coarse levels or global tokens) | Run a whole-scene GNN every step (15–55 ms on a 4090); jump cracks with proximity edges | One call per frame; a bi-stride hierarchy built from the bond graph; tokens only if the ablation earns them |
| Dynami-CAL GraphNet; Allen 2022; FIGNet | Equal-and-opposite edge impulses in edge-local frames | Free per-node velocity heads that can create momentum v1: per-node heads plus an exact momentum projection (established, closed form). Dynami-CAL edge impulses are an ablation that must beat it |
| DeepFracture; phase-field DDPM (2607.11971); FIGNet's caveat | Represent multimodality (discrete codes, calibrated probabilities) | Ship generative decoders (seconds per pattern); regress the mean of a chaotic outcome and threshold it | Ensemble soft labels; hash-seeded sampling ablation; judge fragment-count distributions |
| Breaking Good; Müller 2013; min-cut fracture | Structured, topologically consistent break sets | Precompute modes that topology change invalidates | Min-cut on a learned cost field as an ablation; island detection by the existing split |
| GNS noise; pushforward (Brandstetter 2022); PDE-Refiner; DAgger | Train on the model's own states; inject noise | Train only teacher-forced | Pushforward over 2–4 frames, noise on carried state, DAgger from the reduced state |
| Kochkov 2021; Solver-in-the-Loop; Stachenfeld 2022 | A learned correction over a coarse solver can work | Assume it works when the coarse solver is very wrong (ours: 28.8/163) | Kept as the fallback route |
| Perera 2022; ADAPT-GNN | Stage the decision (is it an event, which bonds, then impulses) | Cite the "region-active factorization", which is not in the paper | Staging is a possible ablation of the break head |
| Hurdle/ballistic-limit models; Lambert–Jonas form | Model the perforate/arrest transition explicitly | Read perforation from velocity under erosion | Dense sampling at the transitions; Lambert–Jonas fit as a scalar baseline; perforation labelled geometrically |
| AIFS-CRPS / GenCast; proper scoring rules | Score against the ensemble (CRPS, spread–skill) | Score a chaotic system on single-run MSE | afCRPS on exit speed; gates relative to the chaos floor |
| Li et al. 2024 FGN, RC slab blast fragmentation (read in full) | Standard GNS/MGN recipe works for blast fragmentation from about 105 training cases in a narrow family; carry the charge as global features plus per-node scaled distance; output a non-negative damage increment; tune the input noise σ; score fragment mass with size bins and a Weibull CDF, peak fragment velocity against Z, and separate interpolation and extrapolation test sets | Step autoregressively per saved step with no conservation (their stated weakness: drift); define fragments only post hoc; expect real-time speed from per-step GNNs (0.15–0.35 s per step) | Confirms our established backbone and monotone damage head. Adds a **global load node** (blast charge, impactor summary), the Weibull fragment-size metric, and peak debris velocity against Z. Our sealed set gets **explicit extrapolation splits** (charge, standoff, strength, size) as theirs did. We keep one call per frame plus exact projection to avoid their drift and cost |
| NVIDIA PhysicsNeMo (cloned at `c8f68428`, Apache-2.0; inspected, not run) | Use its MeshGraphNet blocks and BSMS hierarchy as our training base. Its crash example found **one-shot prediction about as accurate as time-conditional, at a fraction of the cost, and autoregressive rollout unstable over long horizons**. Use separate encoders for contact ("world") edges (HybridMGN); `log1p` targets for heavy-tailed quantities; global per-run scalars for conditioning | Take its PyG/`torch_scatter` dependency (no MPS), its CUDA-assuming trainers or checkpoint utilities, or its normalisation-stats code as is (`_compute_edge_stats` has a mean bug). Expect a fracture example: none of its examples predict breakage | Copy and adapt the MLP, edge and node blocks and the BSMS pooling, with attribution, rewritten on `index_add_`. Fix BSMS's `degree()` bug, which breaks on chunks isolated by broken bonds. Add coarse-edge masking (BSMS has none). Add our edge decoder (it has none) and our global heads |
| McGreivy & Hakim (NMI 2024) | Use strong baselines | Compare against weak ones | Punch, QS peak, the gradient-boosted tree, the explicit window, the chaos floor |
| GNN ensemble collapse (2605.22593); conformal prediction (2502.04406) | Gate out-of-distribution inputs with physics residuals plus input ranges | Rely on ensemble variance alone | Residual and range gate with a logged fallback to the punch |
| Subspace Neural Physics; DF-GNN / kernel-fusion work | Fuse kernels; budget per object | Trust FLOP counts on a latency-bound GPU | One fused dispatch per layer in the P0 probe, measured on Metal and WebGPU |
| WebGPU limits (no bf16, 10 buffers per stage) | Plan for fp16/fp32 and packed weights | Assume CUDA-style tensor cores | Weights in one buffer; fp16 measured, adopted only if parity holds |

## The approach

**One call per routed impact island, per engine frame, made before the engine steps.** It goes where the GPU punch is today: after the routing check, on the punch's two-frame look-ahead.

**What the model covers.** An impact island is the union of every bonded cluster that any impactor's swept volume reaches in the frame, plus the impactors. So a ram passing through both walls of the wall row in one frame is one call. Debris that strikes a second structure is a contact edge between two structures in one graph, so both sides see the same exchange.

**Inputs, all invariant or nondimensional, with no absolute positions and no per-bond identity:**
- **Per chunk:**
  - mass and inertia (principal values);
  - support flag;
  - existing damage at its bonds;
  - graph distance and wave-arrival time from the predicted contact, `d/c`, against the frame and contact times;
  - distance to the nearest anchor;
  - load-agnostic pressure-load features, zero for pure impacts. Blast is filled in P5; see that phase.
- **Per bond:**
  - the edge-local frame (Dynami-CAL style);
  - the six stiffnesses, strengths, `G_f·A`, damage, crush and Weibull factor.
- **Physics-prior features per bond:**
  - QS utilization with the impact as a Hertz-peak load (`solve_static_all`, run at converged settings for the feature only: an explicit option);
  - the punch's plug membership and its capacity ratio `m_eff·v/τ`/capacity.
- **Per contact edge:**
  - approach velocity (normal and tangential), the other body's mass, radius, modulus and spin;
  - time of contact within the frame;
  - energy ratio `E_k / Σ G_f·A` over the plug.

## Model specification v1, and how confident each part is

**Confidence, stated plainly:**
- **High** for the parts that are either standard, well-validated practice (MGN encode–process–decode, residual message passing) or physics we impose by construction (conservation, the material's own failure threshold, invariance).
- **Medium** for capacity and reach (the depth, width and level counts below). They are set from the physics and the P0 cost envelope, then confirmed by the matched ablations in P2.
- **Open:** whether **one call per frame** captures multi-stage cascades, such as the wall row's second wall shattering under the plug. The P0 perfect-model ceiling answers that before any training. The "two-pass refinement" ablation below is the planned remedy if it is the gap.

**Every choice below is fixed for v1.** P2 changes one thing at a time, against this baseline.

### Graph

- **Nodes:**
  - chunks;
  - one node per impactor or other colliding body;
  - one **global node** per island holding load summaries: impactor energy ratio, blast charge and Z, frame dt. This is the GraphNet global from Battaglia 2018, as used by FGN;
  - anchored chunks are flagged and treated as infinite mass in the decode.
- **Edges, each with its own encoder:**
  1. intact or damaged bonds;
  2. cracked interfaces: broken bonds whose faces the oracle still treats as unilateral frictional contact (`joint.rs:14-35`);
  3. predicted impactor contacts, from the routing/punch sweep;
  4. contacts between structures inside the island.

  Physical bond, contact and computational edges keep distinct meanings (handoff §5).
- **Hierarchy: BSMS-GNN's bi-stride pooling** (Cao et al., ICML 2023; Apache-2.0 reference code in BSMS-GNN and NVIDIA PhysicsNeMo).
  - It is built once per intact structure, at load time, from the bond graph.
  - When bonds break, a coarse edge is masked once none of the fine bonds beneath it is intact, so no level carries messages across a crack. That is a mask, not a new method.
  - **GPU-implementation alternative, only if the mask proves too costly:** voxel pooling joined by intact bonds. It is admitted only if it matches BSMS's validation metrics.

### Inputs: invariance, built in

- **Vectors are expressed in edge-local frames:**
  - the bond's axes (t1, t2, n) for bond and cracked edges;
  - the contact normal and tangent for contact edges.
- **Node inputs are scalars only:**
  - mass;
  - principal inertia;
  - support flag;
  - gravity's components;
  - the nondimensional distances and arrival times from the input list above.
- **Consequences:** outputs are rotation- and translation-equivariant by construction (Dynami-CAL, MGN practice), and the model is invariant to node permutation.
- **Scales:** every feature is a ratio to a physical scale, not normalised per graph (the handoff's §12.2 amplitude bug):
  - stresses to the strength (f_t, c, f_c);
  - velocities to v0 and to the wave speed c;
  - lengths to the chunk size;
  - energies to `G_f·A`.

### Processor

- **Message-passing block** (MeshGraphNets):
  - `e' = e + MLP([e, h_i, h_j])` and `h' = h + MLP([h, Σ_in e'])`;
  - separate sums per edge type;
  - two-layer MLPs, LayerNorm, SiLU.
- **U-shape:**
  - 3 fine rounds, pool, 2 rounds at each coarse level, unpool with skips, 3 fine rounds;
  - about 12–14 rounds in all.
- **Reach rule:** the levels are chosen so that the top level's graph diameter is ≤ its rounds. That puts the whole island within one call's reach. A P-wave crosses about 50 m in 16.7 ms, so the whole island *is* within the physical light cone.
  - **Witness test:** a load applied at one end of a 28-chunk wall must change the outputs at the other end.
- **Width:** W = 64, provisional. The P0 cost envelope sets the final width and depth, and P2 compares 32/64/128 at matched data.

### Heads (these define the physics contract)

1. **Bond demand: `log r̂_b`.** `r_b` is the oracle's own failure index, `JointState.utilization`, at its maximum over the frame (`record_peak_utilization`).
   - A bond breaks when `r̂_b ≥ τ`, with τ = 1, the material law's own limit.
   - Why: it is physically meaningful, it is the quantity the material law tests, and it matches the colloid GNN's peak-force-then-threshold recipe.
   - A separate **break-probability** head `p_b` uses the same features and is trained on ensemble frequencies. It is used for calibration and for the sampling ablation.
2. **Break time:** 8 bins over the frame plus "none". It orders breaks for the split and feeds the carried state.
3. **Per-node velocity change** (Δv, Δω) for each chunk and each impactor: the standard MGN node decoder.
   - It is predicted in a local frame built from invariant edge features, then rotated back to the world frame, so it rotates with the scene.
   - **Conservation comes from an exact projection in the decode** (below): the classical, closed-form least-squares projection onto the linear and angular momentum constraints.
   - **Ablation:** Dynami-CAL edge impulses (`J_ij = −J_ji` by construction). That is newer (Nat. Commun. 2026) and less reproduced, so it goes in only if it beats the baseline.
3b. **Regime probability `P(perforate)`,** one per impactor, read from the global and impactor nodes. See "Discontinuous outcomes" below.
4. **Energy dissipated in the frame,** one scalar per island, split into fracture work, damping and friction (the oracle's own ledger). See the energy section below.
5. **Carried damage:** Δdamage ≥ 0 for unbroken bonds, made monotone by the output's non-negativity. It matters for repeat hits.

### Decode (deterministic, Slang)

- Break bonds with `r̂ ≥ τ`.
- **Momentum projection.**
  - **Free islands:** project the predicted velocities of chunks plus impactors onto the input linear and angular momentum. This is the minimum mass-weighted change, in closed form.
  - **Anchored islands:** the support reaction is unknown, so only the energy bound applies, and the impactor's velocity is supervised directly.
- Write the chunks' hidden velocities and run the existing split, whose momentum-conserving mean velocity is exact for those chunk velocities.
- Write the impactor's velocity.
- **Energy bound:** project kinetic energy down to at most `E_in + W_gravity − E_fracture,min`.
  - This is a uniform scaling of the relative velocity about the centre-of-mass velocity, so it preserves momentum.
  - `E_fracture,min` is the fracture work the broken bonds must have consumed. It is used only once P0 confirms the ledger supports it.
  - It is counted and reported whenever it binds.
- The impactor–island contact for the frame is either filtered or left to rubble, chosen by the P0 ceiling.
- **Ablation, two-pass refinement:** run the processor a second time with the first pass's breaks masked. That is a learned version of the cascade's passes, at twice the cost.

### Energy lost through penetration (punch-through that slows the impactor)

**Yes, the model predicts it, two ways, and they must agree.**

1. **Implicitly, through velocities.** The impactor's predicted exit velocity, and the fragments' velocities, fix the kinetic energy left after the frame. A ram that punches through but slows from 40 to 35 m/s has `ΔKE = ½·200·(40² − 35²) = 37.5 kJ`. That energy went into fracture work, damping, friction, the kinetic energy of the fragments it pushed, and vibration still stored in the wall.
2. **Explicitly, through the energy head.** One scalar per island, `E_diss`, split into fracture work, damping and friction. It is supervised from the oracle's own energy ledger (`World::mechanical_energy`, `dissipated_energy`, the per-bond `dissipated` in `JointState`).
   - **Consistency check:** `E_in + W_gravity = KE_out + E_elastic,residual + E_diss` within a tolerance. It is a metric, an input to the out-of-distribution residual gate, and a loss term (`L_energy`).
   - **The physical floor.** With softening on, a fully broken bond has dissipated its fracture energy `G_f·A` for its failure mode, so the cut costs at least `Σ_broken G_f·A`. That is the classical crack-band / Griffith budget, and it is what makes a weak wall cost the ram little and a strong one stop it.
     - It becomes the decode's tighter energy bound only if P0 confirms the oracle's ledger obeys it.
     - P0.0 already checks the open "negative overshoot residual". Brittle snapping with softening off releases stored energy instead of dissipating it, so then the bound does not hold.
- **Metric:** the energy partition (impactor KE, fragment KE, dissipated, residual elastic) against the ensemble, alongside exit speed. This catches a model that gets the exit speed right for the wrong reason: say, no fracture work but too much fragment kinetic energy.

### How grounded each part is (engineering, not new architecture)

**The stance.** v1 is assembled from published, reproduced components and textbook physics. The only new things are the dataset and the problem. Where a newer idea might help, it is an **ablation that must beat the established baseline**, never the default.

| Component | Basis | Evidence strength | If it fails |
|---|---|---|---|
| Encode–process–decode GNN with residual MLP message passing | MeshGraphNets (Pfaff et al., ICLR 2021) and GNS; dozens of reproductions; reference code (DeepMind, PhysicsNeMo) | Strong | — |
| Multiscale hierarchy, bi-stride pooling | MS-MGN (2022), BSMS-GNN (ICML 2023, code); HOOD (CVPR 2023) | Strong for reach; masking broken coarse edges is a small extension | Global-token ablation (MGN-T) |
| Edge-local frames, scalar inputs | MGN relative features; Dynami-CAL; classical continuum mechanics (objectivity) | Strong | — |
| Physical-scale nondimensional features | Buckingham-π practice; handoff §12.2's per-graph normalisation failure | Strong | — |
| Momentum by exact projection | Classical constrained least squares; the same projection rubble's split already applies | Exact (mathematics) | — |
| Energy bound and ledger | Conservation of energy; crack-band/Griffith fracture work (Bažant–Oh; the oracle's `G_f·A` softening) | Exact, conditional on P0's ledger check | Cap on KE only |
| Break = regressed peak failure index ≥ 1 | The material law's own criterion; peak-load-then-threshold (Buzzicotti et al. 2026); utilisation checks in structural engineering | Medium-strong (one ML paper, established engineering practice) | Probability head with calibrated threshold |
| Calibrated break probabilities | Proper scoring rules (Gneiting & Raftery 2007); temperature scaling (Guo et al. 2017) | Strong | — |
| Ensemble labels, CRPS, spread-relative evaluation | Weather ML (GenCast, AIFS-CRPS); proper scoring | Strong | — |
| Training on own states: noise, pushforward, DAgger | GNS (2020), MP-PDE (2022), DAgger (Ross et al. 2011) | Strong | — |
| One outcome call per frame | Outcome-level fracture prediction (Breaking Good, DeepFracture, colloid GNN); ballistic residual-velocity surrogates | **Medium**: done before, never at this accuracy with multiple structures | **The P0 perfect-model ceiling decides it before any training.** Fallback: two-pass refinement, or K = 2–4 calls per frame |

**What I would still research before or during P0** (cheap, and it de-risks the plan):
1. **Read the remaining closest paper,** the projectile-penetration GNS (IJIE 2025), if you can get the PDF. The blast-fragmentation paper (Li et al. 2024) is now read: it confirms the GNS/MGN backbone, the non-negative damage increment, the global load node and the fragment metrics we adopted. It solves fragmentation by per-step rollout, not as an outcome-level call.
2. **Start from a reference implementation, not from scratch.** Done: PhysicsNeMo (cloned into `inspiration/`, inspected). Its MGN blocks and BSMS hierarchy are the base; see the Files section for exactly what is copied and fixed.
3. **A pilot data-scaling study** (new P0 step 5). Train vanilla MGN on 250, 1k and 4k free-wall events, and fit metric against data size. That forecasts how many events the gate needs before we spend days generating, and shows early whether the problem is learnable at all.
4. **The baseline ladder** (P0 step 6). If the gradient-boosted tree on physics features nearly meets the bar, the simplest model that passes is the one we ship.

## Objectives: what we optimise and why

### The outcomes that matter

These are what the gameplay and the M5 bar care about, so they define the **metrics**. The losses are differentiable surrogates for them, and checkpoints are **selected by the metrics**.

| Outcome | Metric (always against the oracle ensemble of the same event) | Grounding |
|---|---|---|
| Did it go through? | Regime agreement (arrest / penetrate / rebound, labelled geometrically) on events where ≥ 80% of the ensemble agrees | Hurdle/ballistic-limit literature; V25's "blocked ram" failure |
| Impactor exit velocity | CRPS against the ensemble, and `|v̂ − median| ≤ max(ensemble IQR, 0.05·v0)` | M5's bar; proper scoring rules (Gneiting & Raftery) |
| Which parts separated | Mass-weighted adjusted Rand index between predicted and member partitions, compared with members against each other | Chaos-relative; handles fragment relabelling and V39's identical largest-fragment fraction with very different counts |
| Fragment sizes | Wasserstein distance on log fragment mass, against the ensemble's own pairwise spread; fragment count and largest fraction inside the ensemble range | The wall row's 133 fragments; DeepFracture's under-fragmentation lesson |
| Fragment motion | Mass-weighted RMS of fragment velocity (chunks matched by stable key); debris centroid and spread at +0.1 s and +0.5 s | V31: tiny projectile error can hide structural failure |
| Support survival | Retained anchored mass fraction; collapse yes/no | Handoff §6 error asymmetry |
| Physical validity | KE_out ≤ KE_in + W_gravity; momentum residual (exact by construction); energy-cap activation rate | Conservation is necessary, not sufficient (handoff lesson 6) |
| Calibration (diagnostic) | Reliability of `p_b` (ECE), Brier score, pooled bond precision and recall against consensus | Needed for the threshold and sampling to mean anything |

**The headline acceptance is a skill test:** the model's distance to the ensemble must be at most 1.25× a typical member's distance to the rest of the ensemble, on every primary metric and every scene. That is, the model should be indistinguishable from "one more oracle run". On scenes where the ensemble agrees exactly, that collapses to exact agreement.

### Discontinuous outcomes: penetrate or bounce

**This is the hardest part of the problem, and it is real physics, not only a model artefact.** Near the ballistic limit `v_bl`, a tiny change flips the regime; the oracle does it too (handoff §6, the b5 15–30 m/s regime boundary). The design handles it at four levels:

1. **Smooth learning signals for a discontinuous decision.**
   - The model never regresses "broke / didn't break" alone. Its main bond target is the **continuous peak demand ratio** `r_b`, which varies smoothly with the inputs even where the break decision flips at r = 1.
   - Near the threshold, the error in `r` is small and the gradient is informative. That is why engineering checks utilisation rather than pass/fail, and why the colloid GNN regresses peak force.
   - The ensemble's break frequency `p̄_b` gives a soft target exactly where outcomes are ambiguous.
2. **An explicit regime head**, the hurdle model of the ballistic surrogate literature:
   - one global probability per impactor, `P(perforate)`, trained on the ensemble frequency (log-loss, proper scoring).
   - It is a direct, smooth signal for the bifurcation, and a **consistency check** at inference: if `P(perforate) > 0.9` but the predicted cut does not free the impactor's path, or the reverse, the call is flagged. That counts toward the fallback gate and is reported.
3. **Timing within a frame doesn't change the outcome.** One call outputs the frame-end velocities of chunks *and* the impactor, whether or not anything breaks, so arrest, rebound and penetration are all predicted states. A "late" break cannot leave rubble's rigid contact to stop the ram.
   - Across frames, the contact stays routed, so the next call sees the embedded impactor and continues.
   - The P0 ceiling tests 1/60 against 1/120 frames to measure whether frame length matters.
4. **Measured as a curve, not a point: the ballistic-limit test** (Lambert–Jonas).
   - For each wall family, sweep v0 finely across the transition. Plot exit speed against v0 for the oracle ensemble (median and band), `blast_parity` QS, the punch and the model.
   - **Gates:**
     - the model's `v_bl` is inside the ensemble's `v_bl` spread;
     - its curve is inside the band away from the transition;
     - in the ambiguous band, its perforation frequency matches the ensemble's (calibration), not 0 or 1.

### Proving value over quasi-static: the three-way report

Every evaluation produces one table per scene and family, on identical events and identical metrics:
- **oracle ensemble** (truth and its spread);
- **`blast_parity` QS** (today's default);
- **the GPU punch** (today's best real-time option);
- **the learned model**;
- **the explicit window**, for cost reference.

| Gate | Rule |
|---|---|
| **Superiority** | On impact and blast families, the model's skill score beats both QS and the punch on the primary metrics, per family. A win on average is not enough: per-family wins and losses are listed |
| **Non-inferiority** | On events QS already gets right (low speed, no break, distant blasts), the model is not worse than QS within the ensemble spread. It must not add noise where the cheap solver is already faithful |
| **Cost** | Its frame cost against QS and the punch is reported on the same scenes. The model must be worth its milliseconds: if it ties QS on a family, routing sends that family to QS |
| **Honesty** | Cases where the model is worse than QS are listed with visualisations, never averaged away |

**This report decides adoption (your decision), routing thresholds, and whether a family needs more data.**

### Where to draw the line between numerical and learned: decided by experiment

**One dataset trains three variants**, compared at matched data, parameters and seeds on an **accuracy–cost Pareto frontier.** You pick the point on that frontier.

| Variant | What the model does | Numerical part | Prior expectation |
|---|---|---|---|
| **A. Impact-only** (the v1 default above) | Predicts the whole frame outcome of routed impact and blast islands | `blast_parity` QS handles everything else: gravity, settling, slow loads, progressive collapse | Simple; the model only sees dynamic events |
| **B. Residual over QS** (Kochkov / Solver-in-the-Loop pattern) | Same islands, but the QS frame result (its utilisations, the breaks it would commit, its velocities) is an *input*. The model predicts corrections: extra or vetoed breaks, Δvelocity relative to QS | QS runs first on every island, as today | Most likely the best accuracy per data and per ms. The model learns only the dynamic part QS misses, and stays close to QS where QS is right (non-inferiority comes almost free) |
| **C. Unified** | Replaces QS *and* the dynamic window on every awake structure, every frame | None, except the decode, split and rigid engine | The most uncertain. Learned simulators are known to drift when holding still under gravity, and its cost scales with awake chunks, not with impacts |

**What makes one dataset serve all three:**
- **Every recorded frame stores**, on the same state:
  - the oracle's outcome;
  - **the `blast_parity` QS frame outcome** (breaks, utilisation, velocities);
  - the punch's outcome.

  That provides B's inputs and A's baselines.
- **For C**, the generator also records non-impact frames: idle and settling structures, slow pushes, gravity-driven progressive collapse after support loss (`support_loss`, `frame_column_removal`), and post-impact settling. These are tagged so A and B can ignore them.
- **Shared** recorder, partitions, sealed set and metrics. Only the input set and the routing differ.

**The sweep that answers "how much accuracy for how many milliseconds":**
- **For each variant:**
  - widths 32/64/128;
  - processor depth (8, 12 or 16 rounds);
  - fp16 against fp32;
  - **the routing threshold**: which events go to the model rather than QS, swept from "only fast impacts" to "everything".
- **Each point is measured on:**
  - the three-way skill metrics (impact, blast and quiet families);
  - the ballistic-limit curves;
  - **whole-frame cost on the M6 town tiers** (median, p95, worst) and on the M5 scenes, on a quiet M3 Max, one run at a time.
- **Output:** a Pareto plot plus a table, in `findings/`. It runs in P2 (A against B on walls) and again in P4 (all three, plus the town tiers). C is trained only if A or B passes P2, since it needs the most data.

### Trajectories and fields, not only end states

Metrics follow the **whole event**: impact, separation, secondary contacts and settling, to 0.5–1 s. They are not taken only at the frame of the call.
- **Projectile:** position, velocity and angular velocity against time, compared with the ensemble envelope; the contact impulse on it against time; the final regime.
- **Every chunk,** matched by stable key:
  - the mass-weighted displacement field and velocity field against time;
  - a **motion mask:** IoU of the set of chunks moving faster than a small threshold. This catches "the whole wall starts to drift" and "nothing moved" failures.
  - first-motion time per chunk.
- **Forces and loads:** support reaction impulse against time; per-bond peak demand ratio (calibration plot of predicted against oracle `r`); energy partition against time.
- **Blast:** the same, plus debris velocity distribution, throw distance and breach area.
- **Visual check** of side-by-side playback (oracle, QS, punch, model) on the fracture lab's viewer for every gate scene, recorded as screenshots in the findings.

### How the model gets a clear learning signal (summary)

1. **Dense, smooth, physically meaningful targets** carry most of the gradient: per-bond demand ratio, per-chunk velocity in the kinetic-energy norm, per-island energy split, regime probability. There are thousands per event, so most of the gradient comes from these rather than from rare binary events.
2. **Physics enforced exactly, not learned:** the momentum projection, the energy bound, monotone damage, invariance. The network spends no capacity on these, and cannot violate them.
3. **The consequences of its own mistakes:** pushforward and DAgger relabel the states it actually produces.
4. **Selection by what matters:** checkpoints, λ weights and τ are chosen on the trajectory, regime and three-way skill metrics, never on training loss.

### Loss

Every term is made dimensionless by dividing it by the zero-prediction baseline's value on the training set, so each starts near 1. This addresses the V34 lesson, where one field was 99% of the loss. Gradient norms per term are logged, and the weights λ are tuned only against **validation metrics**.

- **`L_demand`:** Huber on `log r̂ − log r` for bonds that survive in the target.
  - For bonds that broke, `r` saturates after failure, so the loss is a one-sided hinge: penalise only `r̂ < 1`.
  - Censored regression is the physically correct treatment of a saturated label.
- **`L_break`:** log-loss of `p_b` against the ensemble break frequency `p̄_b`, a proper scoring rule.
  - Class imbalance (about 1.5% positive) is handled by per-family **reweighting with a known prior correction**, not focal loss. Focal loss is not a proper score and miscalibrates the probabilities.
  - Focal loss is kept only as an ablation.
  - After training, a validation-only temperature scaling calibrates `p_b`.
- **Consequence weights on both bond terms:** `w_b = 1 + α·[bond on the separating cut of the ensemble's majority partition] + β·[bond on a support load path]`. These are the bonds whose error flips the outcome.
- **`L_time`:** cross-entropy on the break-time bin, for broken bonds only.
- **`L_energy`:** Huber on `E_diss` and its three parts against the oracle's ledger, normalised by `E_in`. It also penalises violations of the energy balance, so the velocities and the energy head describe the same transition (handoff §5: forces, energy, damage and motion must agree).
- **`L_impulse`** (Dynami-CAL ablation only): squared error between `J_e` and the oracle's integrated edge impulse over the frame, normalised by `(m_e·v0)²`. V29 found bond impulse regressions alone lost to persistence, so this term is never the main signal.
- **`L_motion`:** the kinetic-energy norm of the velocity error, `Σ m_i|Δv̂_i − Δv_i|² + ω^T I ω` terms, divided by the impactor's kinetic energy.
  - Energy weighting is the natural physical metric: an error counts in proportion to how much motion it misplaces.
  - It is also computed per fragment under the **target** partition, so it stays differentiable.
- **`L_impactor`:** `(Δv̂ − Δv)²/v0²`, plus afCRPS when ensemble members are available.
- **`L_damage`:** Huber on Δdamage for unbroken bonds.
- **`L_regime`:** log-loss of `P(perforate)` per impactor against the ensemble frequency, or the geometric label when there is no ensemble.

Total: `L = λ_r L_demand + λ_b L_break + λ_t L_time + λ_m L_motion + λ_i L_impactor + λ_e L_energy + λ_d L_damage + λ_g L_regime` (plus `λ_J L_impulse` in the ablation). Starting weights are all 1 after normalisation, with `L_motion`, `L_impactor` and `L_demand` raised to 2. **After that, weights change only on validation metrics, never on the test or the sealed set.**

### Training procedure

- **Labels:**
  - ensembles of 8 runs on 15% of configurations, with the impact point ±1 mm, speed ±0.1% and Weibull strength seeds;
  - soft targets are the per-bond break frequency and the ensemble-mean impulses;
  - single-run events get hard targets with smoothed labels.
- **Curriculum:** single free walls, then anchored walls with openings, then wall rows and repeat hits, then houses.
- **Several frames:**
  - pushforward over 2–4 frames;
  - Gaussian noise on the carried damage and velocities (the GNS recipe, with σ tuned on validation);
  - DAgger rounds: the oracle restarts from the model's **decoded reduced state**, the state the engine really holds, admitted by `branch_dataset.py`'s parent-split rule.
- **Optimiser:** AdamW, a cosine schedule, gradient clipping and 3 seeds. Early stopping and checkpoint choice use the **validation skill metric**, not the loss.

## Phases and go/no-go gates (each phase can kill or redirect the plan)

### P0: reference, ceiling, chaos floor, cost envelope (no learning; about one week)

0. **Port the two foundation fixes the failed attempt found** (both in commit `6622e05a` on `claude/stress-web-demo`; checked absent on main):
   - **`stress-ref/src/polytope.rs` `mass_properties`.** Main divides each signed tetrahedron term by six before summing. On a degenerate contact polytope, an exactly-cancelling volume becomes a tiny positive one, and the centroid explodes: a contact point about 32 km away, 48.7 million m/s, 4.3e18 J. The fix accumulates determinants first and divides once, with no cutoff, and comes with the test `tiny_positive_volumes_are_retained_without_a_cutoff`. Without it, a random-scene generator will eventually produce corrupted labels.
   - **`stress-ref/Cargo.toml`: `serde_json` `float_roundtrip`.** Saved scenes reload bit-exact. Without it, replaying an event can diverge from the run that recorded it.
   - **Then:** run the guard and the stress-ref, stress-gpu and stress-rubble suites. If any digest moves, compare outcomes. Re-recording a digest for the same physics is your decision, as in the PR #37 precedent.
   - **Checked and not needed:**
     - the branch's uncommitted stress-gpu WebGPU buffer-limit workarounds (main's PR #46 already fits every kernel in 10 buffers);
     - its `stress_provider` / `external_stress` / observer hooks, which are research plumbing; we reuse them only where P1 needs them;
     - a co-rotation turn-cache skip, which is performance only and not on the established path.
   - **Optional for P1 data:** its `builders.rs` additions (a ram with yaw angle, a crackling cantilever).
   - **Also checked in P0:** the open "signed overshoot work residual" (`overshoot = frozen − stored − law` can be negative beyond roundoff). Check it on the pinned configuration's energy ledger before using energy as a label or a cap.
1. **Pin the oracle.** Pick the stress-ref configuration that reproduces `rewind-scaling.md`'s 35.0/6 and 23.6/133: methods, softening, `true_step`, safety. Record the scene hashes. Confirm the GPU explicit window agrees with it on the free wall. On the wall row the routed GPU gives 31.65/41 against the CPU's 23.6/133; explain that gap before trusting any GPU labels.
2. **Chaos floor.** Run ensembles of 16 on both M5 scenes and on 5 more walls. Report the spread of ram speed, fragment count, largest-fragment mass and per-bond break frequency. **All later acceptance is measured against this spread.**
3. **Perfect-model ceiling (the handoff control).** Take the oracle's own break set, its chunk velocities at frame end and the ram's velocity. Inject them through the decode above at the routing frame and continue rigid-only. Variants to compare:
   - contact filtered or not;
   - applied at pre-impact or window-end poses;
   - 1/60 against 1/120 frames;
   - reduced against full carried state.
   - **Gate:** the best variant lands within the chaos spread on both M5 scenes and beats the punch. If it fails, change the boundary before any training.
4. **Cost envelope (in parallel).** Write a random-weight Slang GNN of the target shape: gather, then the edge MLP, then a CSR segmented sum, with one fused dispatch per layer, plus a coarse level and weights in one storage buffer. Time it on Metal and on WebGPU (Chrome) for islands of 220, 1k and 10k chunks, at widths 32/64/128, with fp16 and fp32. This sets the width and depth any trained model may have.
   - **Batch scaling.** Every impact island in a frame (20 buildings, say) is packed into one batched graph: CSR concatenation, one dispatch per layer for all of them. Measure wall time for 1, 5, 20 and 50 islands of 220, 1k and 2k chunks, to find the saturation knee. Below it, extra islands cost about the same, because the GPU is latency-bound. Above it, cost grows with total chunks × edges × layers × width².
   - **An explicit, engine-wide capacity** for the batch: chunks, edges, islands, and the 65,535 groups per dispatch. Beyond it the frame records a second batch, or fails with a clear error. It never silently drops or decimates an island.
   - **Also counted:** decode, split and spawn cost per batch, and rubble's extra cost for the new fragments. These grow with destruction, not with the model.
   - **Gate:** some configuration of ≥ 8 rounds reaches ≤ 2 ms at 10k total chunks, whether one island or the sum of many.
5. **Pilot learnability and data scaling** (once the recorder exists; it overlaps P1's start).
   - Train vanilla MGN (the reference implementation, our heads) on 250, 1k and 4k free-wall events.
   - Plot the validation skill metrics against data size.
   - **Gate:** the curve improves, and extrapolates to the bar within the P1 budget. If it is flat, change the features or the boundary before generating more.
6. **Baselines on the P0 metrics:** zero-break, `blast_parity`, the GPU punch, QS with peak loads, the explicit window, and a per-bond **gradient-boosted tree** on the prior features. If the tree already reaches the bar, the graph model has to beat it to earn its cost.

### P1: data (background, one job set at a time, alongside P2)

- **Recorder bin** `learned_impact_outcome_record`. Per event it writes:
  - the pre-frame state, priors and predicted contacts;
  - per-frame outcomes (breaks with times, chunk impulses and velocities, impactor state) for 3 frames;
  - full-run outcome metrics;
  - hashes and a replay recipe.
- **Generator** (reusing `stress-learned/src/generate.rs` and `stress-rubble/src/scenes.rs`): procedural walls, varying independently:
  - size, thickness, chunk resolution, openings, supports;
  - material strength and stiffness, Weibull;
  - ram mass, shape, radius, speed, angle, spin, location (centre, edge, corner, grazing);
  - pre-damage, repeat hits, two rams, rows of walls, debris strikes.

  Sampling is dense near the arrest, penetration and rebound boundaries, and about 20% of events break nothing.
- **Reused from the failed attempt's data tooling.** All of it is on `claude/stress-web-demo`, which the new worktree branches from; main has only the M1 scaffold.
  - **Job, manifest and catalog framework** (`stress-learned/src/generate.rs`, `catalog.rs`, `recorder.rs`, `validate.rs`). `recipes.json` is written *before* generation and must match exactly on resume; `--plan-only`, `--limit`, `--family` and `--jobs N` already exist. Calibration: 80 scenarios in 423 s on 8 workers, all passing the array, provenance and replay audit.
  - **Partition hygiene.** Every recipe carries a `split_group`. Longer clips, cadence variants and corrective branches keep their parent's group. The trainer groups identical physical first frames, so prefixes of one trajectory never straddle train and evaluation. Copy both rules.
  - **`layouts.rs`:** 12 wall layouts, 16–28 × 1–3 × 16–26 chunks, 6 with openings. These become layout families; **whole layouts** are held out.
  - **`scale_layout_job`:** a 10,400-chunk wall ("never truncate to fit a model"). Use it for the P0 cost envelope and a scale check, not for training.
  - **`permutations.rs`:**
    - a `materials` helper that scales strength, toughness and stiffness without touching the steel projectile;
    - location, height, mass, size, approach-yaw, rigid-or-crushable projectile and house-blast recipes;
    - `projectile_spin_jobs`: 52 paired spin experiments, reused as an observability witness test (same input except spin ⇒ the model must tell them apart).
  - **Provenance sidecars:** `scene.json` (full scene and seed), `context.json` (stable chunk and bond keys, recorder revision, generator parameters), `replay.json` (poses, velocities, projectile, energies, event fingerprint), plus `python/learned_impact/audit.py`.
  - **`branch_dataset.py`:** admits corrective branches only from training parents. Reuse it as the admission gate for DAgger data.
  - **The transition-preserving sampler** (`cace5923`): it keeps every damage, crush and new-fracture row, and errors rather than dropping them over budget.
  - **Fail-loud recording:** the fixed-graph recorder rejects newly active chunks or bonds rather than dropping their labels. Keep that behaviour.
- **Record the whole life cycle of every scenario, not only the first hit.**
  - **When a recording stops:** never at a fixed short duration. It runs until every impactor has stopped or left the structure's bounds, **and** no bond has broken for N = 30 frames (0.5 s), **and** the island's kinetic energy is below 1% of the input energy, or it is asleep.
  - **Explicit cap:** 10 s simulated per event. Hitting the cap is logged as `truncated: true` and the event is excluded from outcome metrics, never silently counted as finished.
  - **Phases tagged per frame:** pre-contact, contact or blast, separation, secondary contacts (debris re-hits, the projectile reaching the next wall), and settling. This lets each phase be balanced and audited.
  - **Balanced sampling across the life cycle.** The trainer samples frames by phase with set quotas (for example 35% contact or blast, 20% secondary, 25% separation and settling, 20% quiet), plus every transition frame. It never samples raw frames uniformly, which would be dominated by idle settling, nor only impact frames, which would teach over-fracture. Quotas are reported per epoch.
  - **Multi-wall and multi-stage scenes** (the wall row, rows of 3–8 walls, a ram through a house) are recorded through the last wall *and* the settling. The trajectory metrics already run to 0.5–1 s after the last event, so a model that only gets the first hit right fails.
  - **Audit report per dataset,** recorded in `findings/`: a histogram of event durations; frames per phase; the share of events still breaking in their last 0.5 s (must be 0, where the failed attempt had 52/257); and the share truncated.
- **Multiple simultaneous loads on one structure: a first-class data axis, scaled up by curriculum.** The model is already built for it: any number of contact edges and impactor nodes, plus load features on chunks, in one graph and one call. The data has to teach it. Tiers are added as each earlier tier passes its gate:
  1. one fast impactor (the P2 start);
  2. 2–4 fast impactors with independently drawn mass, speed, shape, angle and location, hitting at different times within and across frames (same face, opposite faces, far apart, overlapping craters);
  3. **mixed regimes on one structure:** resting loads (debris lying on a floor, a parked vehicle, stacked chunks), slow pushes (a vehicle pressing at 1–5 m/s), and fast impacts at once. Resting contacts below the routing speed reach the model as contact edges with their pre-contact velocities, so it learns that they press but rarely break;
  4. **impacts and blasts together**, and repeat hits on already-damaged regions;
  5. **whole buildings** (cuboid houses, then larger assemblies) with 5–20 loads spread over the structure, including debris from one impact striking another part of the same building.
  - **Sampling:** the number of loads, their arrival times and their spatial spread are drawn independently, from Poisson/uniform priors, so the model sees both crowded and sparse cases.
  - **Generalisation tests** (in the sealed set): hold out *load counts* larger than any trained (for example, train on ≤ 4 and test on 6–8) and unseen load combinations, as in FGN's extrapolation splits. Per-load metrics apply (each impactor's exit regime and speed), not only the island's totals.
  - **Superposition sanity check, a diagnostic:** for two hits far apart, beyond the stress-wave travel within the frame, the joint outcome should match the two single-hit outcomes. The oracle confirms where this holds, and the model must match it.
- **Their recorded gaps, which this generator must not repeat:**
  - **Paired single-factor sweeps with correlated factors**, "not a complete factorial". Ours draws balanced random combinations (Latin hypercube over the nondimensional groups) plus targeted transition sampling.
  - **"Seed changes alone do not diversify a deterministic material."** Ensembles must perturb something physical: Weibull strengths (`oracle_concrete(Some(m))`), the impact point, speed. Check that the resulting arrays actually differ.
  - **52 of 257 events still breaking in their last interval; house clips of only 24 ms.** Ours records the whole life cycle (above), so the full-run outcome metrics are final.
  - **1.54% positive bonds.** Report pooled precision and recall per family and per scene, never a mean of per-frame F1.
  - **Four time-bin averages of loads, and QS fields filled from the previous bond force unless `--always-qs` is set.** Our prior is always a real (converged) QS solve with the Hertz peak, recorded as such.
  - **Little independent geometry; nothing near 10k chunks.** P1's procedural walls vary thickness, aspect, openings, supports and resolution, and the house phase adds roof mass and missing members.
  - **Recordings stamp frame-*end* oracle contacts.** Our inputs are only what the routing check knows before the step (predicted contacts and pre-contact velocities), so the recorder captures exactly the model's causal inputs.
- **Recording cost with full life cycles.** Recording to settling costs more per event, typically 1–3 s simulated rather than 30–133 ms. Separated fragments become rigid-only and bodies sleep, so the tail is cheaper per frame than the impact. P0's pilot measures the real cost per event by family and resets the target count. Full life cycles take priority over event count.
- **Budget:** about 10k wall events a day at 12–16 parallel jobs. The target is about 30k events, with ensembles on 15%.
- **Partitions:** by family and by parameter range. A **sealed confirmation set** of unseen families stays closed until P4, and validation is never reused as a test.

### P2: model (PyTorch on MPS)

- Start with a curriculum on single free walls, then wall rows and repeat hits.
- **Variants A (impact-only) and B (residual over QS)** on the same data, as the first matched comparison. Both go forward on the Pareto frontier.
- Matched ablations, 3 seeds each:
  - local GNN against hierarchical against hierarchical plus tokens;
  - with and without the priors;
  - threshold against sampled breaks;
  - edge impulses against a direct velocity head;
  - pushforward/DAgger on and off.
- **Gate, on validation families:** the skill test above passes on every primary metric and every scene. Concretely, relative to the P0 chaos floor:
  - ram exit speed within the spread plus 5% of the initial speed;
  - fragment count and largest-fragment mass inside the ensemble range;
  - bond F1 ≥ 0.9 against the ensemble consensus;
  - < 1% false breaks on no-break events;
  - multi-frame rollouts stay finite and bounded in energy;
  - beats the punch and the gradient-boosted tree on both M5 scenes;
  - the three-way report's superiority and non-inferiority gates pass per family;
  - the ballistic-limit curve gates pass on every wall family;
  - the trajectory and field metrics stay inside the ensemble envelope to 0.5 s.

### P3: GPU (Slang only)

- `stress-gpu/shaders/impact_learned.slang`, built from the P0 kernel, with weights in one buffer (WebGPU's 10-buffer limit).
- The decode in the same submission, through the punch's split path, plus an impactor velocity write.
- Behind the flag `STRESS_IMPACT_PUNCH=learned`, off by default, so the default digests do not change.
- **Out-of-distribution fallback:** an input-range check plus a physics-residual score (for example, energy-cap activation). It falls back to the punch through an explicit config, and every fallback is logged.
- **Parity:** PyTorch against Slang (Metal and WGSL) within fp32 tolerance, with identical break sets on validation. Fixed-order reductions; the guard twice.

### P4: acceptance on the sealed set, then houses

- The P2 gates on unseen families.
- R3's acceptance: both M5 scenes beat the correction pass, every impact frame stays within the M6 worst-frame target, and quick and standard tiers meet M6 with the flag on.
- Then repeat P1–P4 for cuboid houses and the arch.
- Adopting the model into a profile is your decision.

### P5: blast loading (in the final model; after the impact model passes P4)

**The oracle already has blast loading, and it is the same on the CPU and the GPU.** `stress-ref/src/blast.rs` is an empirical pressure pulse on exposed faces, with no gas dynamics:
- Kinney–Graham incident overpressure, duration and impulse, scaled by Hopkinson–Cranz distance Z = R/W^⅓;
- a Friedlander waveform;
- oblique reflection;
- shadowing by other chunks (a 0.2 diffracted fraction);
- UFC 3-340-02 clearing, which shortens around a breach.

`stress-gpu/src/loads.rs` mirrors it exactly (`FaceBlast`, the same exposure and clearing caches). Existing blast scenes: `blast_two_walls`, `masonry_house(HouseEvent::Blast{tnt})`, `pressure_panel(_sized/_weibull)`, and the failed attempt's 16 house-blast recipes (0.25–5 kg TNT, 1–6 m standoff).

**Why it fits the same model without a new architecture.** A blast is another *load* on the same bond graph. The inputs are designed load-agnostic from v1, so blast arrives as new data and features, not a new network:
- **Per-chunk blast features,** computed by the engine's own `FaceBlast` code for the current frame and topology (summed over the chunk's exposed faces, in the chunk-local frame):
  - peak reflected pressure × area ÷ (tensile strength × bond area);
  - the impulse delivered within this frame, ÷ (chunk mass × v_ref);
  - arrival time and positive duration, relative to the frame;
  - clearing time;
  - shadowed fraction.
- **Global features, in a global node as in Li et al. 2024's FGN:** charge TNT mass, scaled distance Z to the island, and the impulse over the whole pulse.
- **Feedback is kept.** Exposure and clearing are recomputed by the engine after each frame's splits, so a breach that vents the pressure changes the next frame's inputs, just as in the oracle.
- **Routing.** A blast routes every island with an exposed face whose peak reflected pressure exceeds a set fraction of its weakest bond's capacity, in the frames its pulse overlaps. Islands below that fraction stay on `blast_parity`. Near-threshold events are in the data, so nothing jumps at the boundary.
- **Duration across frames.** A pulse lasts about 1–20 ms and can arrive mid-frame. The carried state and multi-frame training (pushforward, DAgger) handle it, as with persistent contacts.
- **Energy.** The input energy is the work the pressure does on the moving faces, from the oracle's ledger (`external_work`). The same consistency check and fracture-work floor apply.

**Data (P1 extension):**
- TNT 0.1–30 kg; standoff 0.5–15 m (Z ≈ 0.3–30 m/kg^⅓); charge height and azimuth;
- in front of, beside, behind and inside a structure;
- shielding by a second building, several charges, and blast followed by a ram;
- walls with openings, houses, and the arch.

Dense sampling near the no-damage, breach and collapse transitions, with the same ensembles and family holdouts.

**Gates:** P2–P4's skill test on blast families, plus:
- breach area against the ensemble;
- fragment-size distribution as FGN scores it (mass per size bin, a Weibull fit of the cumulative mass);
- peak and distribution of debris velocity against Z;
- debris throw distance;
- collapse yes/no;
- **explicit interpolation and extrapolation splits** in charge, standoff, strength and size.

**An oracle sanity check, not a training target.** FGN's LS-DYNA model is calibrated against blast tests, and reports peak fragment velocity against Z: about 110–130 m/s at Z = 0.17 for a 120 mm C30 slab, falling to about 25 m/s at 0.40. Our oracle's empirical pulse is least trustworthy at such close range. P5 first runs equivalent slabs through the oracle and reports how far it is from that curve, so we know whether the blast reference itself needs work at small Z before we train on it.

**Edge cases specific to blast:**
- **Contact charges** (Z < 0.3, beyond the Kinney–Graham fit's range): the input-range gate falls back to the explicit window, logged.
- **Confined interior blasts:** the oracle has no gas-pressure build-up, so the model can be no better than the oracle there. Recorded as an oracle limit, not trained around.
- **Blast and impact together:** both load types are present in one graph.

## Progress logging for every long-running job

**Rule: no job runs more than about 30 s without a progress line, and every job can be checked without interrupting it.**

- **Data generation** (Rust bins and the job framework):
  - one line per finished event: `[gen] 412/10000 events (4.1%) | wall_L07_v32 done 3.2 s sim, 2.1 s wall, 186 breaks, phase=settled | 9.8 events/min | ETA 16h40m | failures 0`;
  - a heartbeat every 30 s while a long event runs: event id, simulated time reached, frames, current phase, bonds broken so far.
- **Training:**
  - a line every 30 s or N steps: step, epoch, each normalised loss term, gradient norms per term, learning rate, steps/s, ETA;
  - a validation pass every K minutes, printing the skill metrics and the best checkpoint so far.
- **Evaluation and benchmarks:** per-scene lines as each finishes, with running totals.
- **Where it goes:**
  - everything also goes, line by line and flushed immediately, to `runs/<job>/progress.log`;
  - a small `runs/<job>/status.json` (state, counts, ETA, last update time, last error), rewritten every 30 s, which a check-in can read with one command;
  - stale detection: if `status.json` has not updated for 5 minutes, the job is treated as hung and reported.
- **Failures are loud:** a failed or nonfinite event is logged with its recipe and seed and counted in the status, and generation continues. The run fails if failures exceed a set fraction.
- **Long jobs run in the background** (one perf job at a time), and check-ins read `status.json` and the tail of `progress.log`.

## Edge cases and mitigations

| Risk | Mitigation |
|---|---|
| Chaotic, multimodal outcomes; thresholded marginals under-fragment | Ensemble soft labels; spread-relative gates; hash-seeded sampling ablation; check fragment-count distributions, not MSE |
| The perfect-model handoff itself fails (V26 lesson) | P0 step 3 comes before any training; boundary variants are tested there |
| Double-counted momentum when rubble also resolves the ram against the freed fragments | P0 contact-filter variant; exact momentum projection in the decode; the energy cap is reported whenever it binds |
| Several walls, or debris into a second structure, in one frame | The routed impact island is the unit; contact edges between structures sit in one graph |
| Contact persists across frames (embedded ram, a pushing vehicle) | Carried state, multi-frame data, pushforward and DAgger on the reduced state |
| Contact begins inside the first frame (frame 0 is never resident) | Contact time within the frame is an input; data starts strikes at varied in-frame times |
| Slow contacts near the routing speed (5 m/s) | Train on near-threshold events whose target is the QS outcome, so nothing jumps at the threshold |
| Anchored structures (momentum not conserved) | The support reaction is the impulse on anchored chunks; the decode projects only over free DOFs |
| Generalisation to new geometry, materials or resolution | Nondimensional inputs; family holdout; data that varies resolution; out-of-distribution fallback |
| Many loads on one structure (resting, pushing and impacting at once) | Contact edges per load in one graph; P1's multi-load curriculum tiers 1–5; held-out larger load counts; per-load metrics; the superposition diagnostic |
| Many simultaneous impacts (20+ buildings) | One batched call for all islands; the knee is measured in P0; capacity is explicit, with a second batch or a clear error beyond it; impacts touching one another are merged into one island |
| Large islands (10k+ chunks) | Hierarchy; the P0 cost envelope; crop to the region of influence, measured with and without the crop |
| Sliver chunks in the town (1e-7 s steps) | The learned call needs no substeps, which helps R4. Town acceptance still waits on the re-export |
| A noisy prior (1×16 unconverged CG) | Run the converged QS for the feature only (an explicit option), plus a sensitivity test |
| Determinism and cross-backend drift | Fixed-order sums; hash-seeded sampling; guard digests; Metal against WGSL parity |
| Overfitting or validation leakage (handoff lessons 7–8) | 3 seeds; sealed set; no identity features; validation never used as a test |
| Model drift from the reference | Physics-residual gate; logged fallback; never silent (AGENTS.md) |

## Files

- **Branch:** a new worktree from `claude/stress-web-demo`'s tip. It holds V39's `native_comparison.py`, `interaction_metrics.py` and `stress_provider.rs`, which we reuse; we do not edit its unpushed history.
- **First step:** a findings note, `engine/stress-oracle/learned-impact/findings/independent-review-2026-10-20.md`, recording the corrections above.
- **Rust:**
  - `crates/stress-learned/src/bin/learned_impact_outcome_record.rs` and a handoff-control runner, `learned_impact_outcome_control.rs`;
  - reuse `generate.rs`, `stress-rubble/src/scenes.rs`, `statics.rs` `solve_static_all`, and `world_coupling.rs` `predict_punch` for the plug feature.
- **Python:** `learned-impact/python/learned_impact/impact_outcome.py` (model, trainer, export), `outcome_eval.py`, `gbdt_baseline.py`, and tests in `python/tests/`.
- **Adapted from PhysicsNeMo** into `learned-impact/python/learned_impact/mgn/`. These are copied, so nothing imports `inspiration/` (AGENTS.md). Each file gets an Apache-2.0 header and the source commit `c8f68428`, and the BSMS files also keep BSMS-GNN's upstream notice.
  - `nn/module/gnn_layers/mesh_graph_mlp.py`, `mesh_edge_block.py`, `mesh_node_block.py`: the residual MLP, edge and node blocks, with plain `nn.LayerNorm` and aggregation through `index_add_`. No PyG, `torch_scatter`, TE or nvFuser, so they run on MPS.
  - `models/meshgraphnet/meshgraphnet.py`: the encode–process–decode skeleton, extended with edge-type encoders (the HybridMGN pattern), the global node, an edge decoder and global heads.
  - `nn/module/gnn_layers/bsms.py` plus `datapipes/gnn/bsms.py`: bi-stride pooling and the per-structure hierarchy builder, run at load time. Fixes: deque-based BFS, `degree(num_nodes)`, edge-local fibre features, and coarse-edge masking.
  - **Unit tests:** the copied blocks reproduce the original PhysicsNeMo outputs on CPU for fixed weights. That test is generated once and committed as a fixture, never imported from `inspiration/`. Plus permutation, rotation and batching invariance tests.
  - **Our own instead:** a plain trainer loop and `torch.save` checkpoints, rather than its Hydra/DDP trainers.
- **GPU:**
  - `stress-gpu/shaders/impact_learned.slang`;
  - a cost probe, `stress-gpu/src/bin/stress_gpu_gnn_probe.rs`;
  - the flag in `stress-gpu/src/world.rs` and `world/resident_frame.rs`, next to the punch;
  - shaders regenerated with `scripts/compile_shaders.sh`.

## Verification

- **P0:** a JSON report plus a findings table covering the oracle configuration, the chaos spread, the ceiling variants, baselines and the cost envelope.
- **Python:** `pytest learned-impact/python/tests`.
- **Rust:** `cargo test --release -p stress-ref --test determinism --test dynamics`; `cargo test -p stress-learned --release`.
- **GPU:** `STRESS_GPU_FIXED_HORIZON=1 STRESS_METHODS=established cargo test -p stress-gpu -- --test-threads=1`, plus the new Slang/PyTorch parity test and `tests/punch_gpu.rs`.
- **Scenes:** `stress_bench` on `wall_ram_late2` and `wall_row2_late2` with `STRESS_IMPACT_PUNCH=learned`, compared against `gpu` and the explicit window. One perf job at a time on a quiet M3 Max.
- **Guard:** 32/32 identical with the flag off, and twice-identical with it on.
- **Browser:** the fracture lab's stress mode with the learned option, checked visually on a wall and a cuboid house.
