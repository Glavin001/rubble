# Explicit physical-state experiments, V37

The physical graph is the baseline temporal memory. A learned transition should advance chunk deformation/rotation and velocities together with material history; force, stored energy and motion must describe the same transition. Additional learned temporal memory is optional and must earn its cost. The detailed input/output and ownership contract is in [PHYSICAL_STATE_TRANSITION.md](../PHYSICAL_STATE_TRANSITION.md).

These experiments establish an accurate linear elastic baseline and expose missing recorded state. **They do not repair the failed V36 destruction model, establish nonlinear native accuracy, or meet the whole-world 5 ms target. No checkpoint is promoted.**

## What changed and what the measurements mean

| Experiment | Result | Decision |
| --- | --- | --- |
| Suppress first-frame damage growth where a future oracle mask says it should be unchanged, then run seven oracle continuation frames | House00 final breaks 394→399; house12 276→268. The same model forces remain. | A deliberately oracle-assisted diagnostic. Reject a damage gate alone as the next fix; this is not causal inference. |
| Explicit displacement/velocity, generic multiscale network, energy/momentum projection | Two seeds: 32-step normalized state MSE 77,749→0.253 and 4,303→0.272. Energy remains bounded, wave phase remains inaccurate. | Conservation helps stability but does not establish accuracy. Valid only for free undamped elastic chains. |
| Shared learned linear operators for displacement and velocity | Sequence-based validation selects degree 16, 32 stored coefficients. Unseen 32/64/128-node chains: 32-step normalized MSE 4.61e-15–6.07e-15; max relative energy drift 1.89e-7. | Keep as a physical capability baseline. It is linear system identification with known mechanical structure, not a successful nonlinear neural solver or an equal-optimizer-budget comparison. |
| Same coefficients on heterogeneous scalar graphs | New grids, supports, branches and disconnected systems: max relative state energy error 3.36e-14 after 32 steps. | The graph, mass, stiffness and support inputs transfer within the declared spectral/time domain. |
| Constant external forces, changing independently each interval | Four new graph cases: max relative work-balance defect 9.42e-8 after 32 steps. | Forces and work follow the same propagator. Prescribed forces are not live contacts. |
| Native six-DOF graph geometry and full inertia | Deformations, generalized forces and nodal forces match the oracle's own calculations. Four 24-chunk graphs retain max relative state energy error 4.11e-14 over 32 transitions against a dense modal teacher. | Frozen small-motion linear mechanics validated, including rotations, fixed/pinned supports and rotated anisotropic inertia. No damping, fracture or finite corotation yet. |
| Float32 inference on those native linear graphs | Max relative state energy error 5.66e-9; max work-balance defect 1.79e-5 over 32 transitions. | Promising numerical precision within this narrow domain; measured on CPU, not a GPU throughput result. |
| Exact physical chunk/contact recording | All 13 existing files remain byte-identical over two house frames, 298 phases and 365,379 response rows. New sidecars retain raw f64 state and ordered contact histories. | Keep the lossless recording path. Existing models do not yet consume it; this is not a complete World restart. |
| Independent projectile/contact feedback | Three 1-D elastic impacts: evolving contact yields projectile position RMSE ≤1.84e-8 m against an independently converged force-law reference. Holding each interval's initial contact force yields 0.0025–0.155 m RMSE and 20–38% energy drift. | A causal coupling capability result. Same fitted coefficients in both arms; no friction, fracture or native-engine inference. |
| Equivalent streamed computation on Metal | At 200,046 bonds, resident transition median 303.34→75.25 ms. At 20,784 bonds, 38.55→12.13 ms. | Keep the algebraic optimization, but reject this implementation as a solution to the full-world timing requirement. |
| Native damping capability | Six frozen native fixtures: damping-force parity ≤3.56e-15 absolute; independent RK4 error energy ≤3.02e-18 of initial energy; short-window work defect ≤1.40e-13. | Keep the dense small-graph reference for validation/targets. This does not add damping to the learned polynomial or establish native destruction accuracy. |
| Early response versus frame-end information | Heterogeneous fixture: all 216 mechanical directions needed at 0.5 ms for 5% worst-case state-amplitude error; 30 suffice at 8.33 ms for 1% error in the best per-graph linear factorization. | Preserve detailed transient information. These are measured representation bounds, not permission to truncate modes or a universal learned-network capacity claim. |
| Physical history v2 | Exact bond frames/levers, nominal stiffness/damping and cluster mass-damping rules added. All 13 legacy files still byte-identical across 298 phases and 365,379 response rows. | The old feature vector could alias different runtime damping. New recording closes that gap; the existing trainer does not yet consume it. |

Relative state energy error means the mechanical energy of the displacement/velocity **error**, divided by the teacher state's mechanical energy. It is quadratic, not a percent trajectory accuracy or a fracture score. Work-balance defect uses initial mechanical energy plus accumulated absolute external work as its denominator. Normalized state MSE uses training-only scales. Detailed numbers, source hashes and experiment scopes are in [the result summary](physical-state-v37-2026-10-19.json).

## Why the structured baseline works

For frozen elastic mechanics, `M x'' + K x = f`. The graph supplies `K = Bᵀ diag(k) B`; `B` maps chunk translation/rotation into the six bond deformations. Mass and full inertia supply `M`. After mass normalization, one shared operator pair advances both physical fields:

```
x_next = C(L) x + S(L) v
v_next = -L S(L) x + C(L) v
```

`C` and `S` are fitted Chebyshev polynomials, using training transition pairs. The teacher uses eigenvectors; inference uses sparse graph operations and the fitted coefficients, without eigenvectors or oracle calls. Rigid modes are preserved by construction. Constant-force displacement uses `(I-C)/L`, evaluated without dividing by eigenvalues or inverting a matrix. Its zero-mode constraint is enforced during fitting; incompatible coefficients reject forced inference.

This deliberately encodes known physics instead of asking independent neural output heads to rediscover mutually consistent motion, forces and energy. Persisted state is displacement/rotation and velocity, with no hidden temporal vector. Nonlinear material evolution and live contacts have not been learned by this experiment.

## The performance finding is a limitation, not a speedup

Degree 16 still means sequential graph work, and displacement, velocity and forcing require multiple field evaluations. Its declared dimensionless spectral domain permits only about 0.073–0.122 ms per call on the small stiff concrete fixtures. Covering an illustrative 16.67 ms frame would require 137–230 calls using the conservative spectrum bound. The existing 665-chunk/1,299-bond house00 graph permits about 0.569 ms, or 30 calls. These are supported-domain calculations, not certified universal accuracy bounds or lower bounds on all possible architectures.

The user subsequently specified a **120 Hz test cadence**, leaving headroom alongside likely 60 Hz PhysX: use 8.33 ms of simulated time per tested update with the ≤5 ms entire-world ceiling. Final target GPU remains unspecified; the local M3 Max is provisional. No component timing here establishes that target. The next architecture must demonstrate useful accuracy per unit of total computation, including contacts, event timing, topology, state transfer and integration. A larger physical interval cannot be justified by a small parameter count alone.

The resident float32 Python/PyTorch implementation was measured on local M3 Max Metal with two warmups and five synchronized samples, using repeated intact house graphs. It includes input checks and excludes engine contacts, material evolution, topology, rigid motion and transfers. These are physical chunk/bond counts, **not active rigid-body counts**. At 102,410 chunks/200,046 bonds, a single 0.512 ms physical transition originally cost 303.34 ms. Precomputing the same mass-normalized bond maps and streaming/batching the same polynomial fields reduced this to 75.25 ms, about 4.03× faster. This changes no bond, material parameter, timestep or mathematical propagation operator. CPU/Metal parity passes; repeated f64/f32 native linear audits and all 200 Python tests pass. Float32 max work-balance defect after this change is 1.43e-5 over 32 transitions. This is a useful implementation improvement, still far outside the required cost. It does not measure a fused production kernel or establish a fundamental lower bound on that kernel.

The repo's previous full modal pre-window also matters: [its accuracy passed and its cost failed](../../findings/modal-prewindow.md). Replacing local steps with dense exact modes is not a newly discovered performance solution.

## A specific physical-memory requirement

The oracle's contact layer retains ordered cohorts of material as contact area grows and shrinks. Two layers can have the same displacement, aggregate stiffness, force and energy, but opposite forces after the newest cohort leaves. A new native test demonstrates this. A few current aggregate response values therefore do not encode this contact law's complete state.

`World::contact_history()` exposes those ordered shear/twist cohorts, normal, indentation, crushing and pair-overlap history without changing the world. `RuntimeGraph::capture_with_physical_history()` additionally retains exact raw chunk state, full inertia, mass, support constraints, identities and flags. `learned_impact_phase_record ... --physical-history` writes optional boundary sidecars; the old snapshots and phase arrays retain their existing meaning. In the two-frame house check, the 2 ms boundary holds 189 contacts, 1,283 shear cohorts and 2,006 twist cohorts. No cohorts are pooled or dropped.

The stress model need not own the engine's contact state if the engine computes contact forces during the coupled advance. It does need an interface that receives the changing interactions causally. The sidecars make the oracle history inspectable and available for the eventual coupled training contract; they do not prescribe duplicating the contact engine inside the network.

The new `contact_capability` probe isolates that interface. A projectile starts touching an elastic bar, contact is a unilateral spring, and separation/recontact events use the evolving model state's trajectory. The same fitted elastic coefficients advance either a live contact edge or a force held from the interval's start. Three masses/stiffnesses exercise rebound and, in the heavier case, repeated contact. The live-contact trajectory agrees closely with an independent RK4 force-law reference; a second finer reference checks convergence. A separate two-free-mass test checks the known analytic collision duration and velocity exchange. The near-zero numerical errors are limited by the reference convergence and do not demonstrate real destruction accuracy. Event resolution and bisection require additional computation that has not been optimized or counted as a runtime win.

## Damping and the two timescales

`damped_body_graph` represents the frozen system as `M x'' + C x' + K x = f`, with physical state `x, x'`. Bond dashpots and applicable mass damping form `C`. It keeps full per-chunk inertia and all free DOFs; free-cluster mass damping needing deformed inertia relief is explicitly rejected by this export rather than approximated. A balanced dense first-order exponential supplies small-graph validation/targets only, limited to 256 free DOFs. This uses standard matrix-function machinery, not a newly trained network or a proposed dense production runtime; [Al-Mohy and Higham](https://eprints.maths.manchester.ac.uk/1591/) discuss efficient actions of these functions.

Six native fixtures cover free, fixed, pinned, rotated, heterogeneous bond damping and fitted mass damping. The dense transition agrees with independent NumPy RK4 under refinement and satisfies energy loss against independently integrated dissipated power. Omitting damping yields error energy of 91–98% of initial energy at the first 8.33 ms endpoint for the sampled states. This is a quadratic mechanical error metric, not a trajectory accuracy percentage.

Deleting damping's cross-coupling in the undamped modal basis looks much better at frame endpoints than during early response. In the heterogeneous fixture, its worst-case error energy is about 3.02% of initial energy at 0.5 ms, versus 0.0071% at 8.33 ms. Both numbers come from the energy-normalized operator norm over all initial states of this frozen graph. No such simplification is adopted.

The high-frequency modes can lose almost all their energy by a frame boundary. That is useful evidence for a compact later-response representation, but the earlier stress remains consequential. The singular-value audit retains all modes and measures their information content at several physical times; free rigid modes are explicitly skipped for that energy-norm diagnostic because their displacement has zero elastic energy. A separate analytic damped-oscillator test shows a large early force peak followed by a near-zero endpoint. An endpoint-only loss would not constrain the material damage caused by that peak.

The architecture implication is a detailed fine-graph path for transients and nonlinear events alongside faster long-range communication, with state and work supervised over the interval. The result does not prove that a tiny shared graph network can achieve this at scale. Existing physical coefficients, ordered history and causal feedback remain necessary regardless of the communication mechanism.

## Reproduction and remaining gates

Use the existing Python environment. The scalar fitting CLI chooses capacity using independent 32-step validation sequences:

```sh
python -m learned_impact.wave_operator --force-consistent --out /tmp/new-forced-operator
```

From `engine/stress-oracle`, with the required Rust environment variables set, export each of `free`, `fixed`, `pinned` and `rotated` using:

```sh
cargo run --release -p stress-learned --bin learned_impact_elastic_graph -- \
  --fixture fixed --out /tmp/native-fixed.json
```

Then run `python -m learned_impact.elastic_body_audit --weights /tmp/new-forced-operator/operator.pt --graph /tmp/native-fixed.json --out /tmp/new-native-audit.json`. Outputs must be new paths. `--scene PATH` exports an existing intact, unreinforced native initial graph. Dense modal validation is intended for small graphs; sparse inference does not use that teacher.

204 Python tests and 12 targeted Rust tests pass at this checkpoint. The first new recorder test failed because the chosen constitutive fixture intentionally had zero damping, so doubling it could not distinguish two states. The fixture now initializes a nonzero physical dashpot and verifies the intended alias; the recording implementation needed no change. Both the failure and fix are preserved in Git.

The next gates are an efficient learned transition handling actual damping and nonlinear constitutive consistency, native contact/actor feedback, fracture event timing, coordinate changes at splits, independent long rollouts and complete-engine timing. Full capture/restore/continue parity remains open. New corpus collection must preserve the required state before large-scale retraining; existing frame-start snapshots alone do not contain all of it.
