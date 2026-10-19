# Research for a compact learned dynamic stress solver

Research and design review, 19 October 2026. Scope: replacing dynamic stress evolution inside Rubble while retaining physically credible contact, fracture, fragment motion and external actor response at **10,000+ active rigid bodies**. Graph nodes, chunks, mesh vertices and active rigid bodies are different scale measures.

The literature supports this direction, but does not establish that our present architecture, training set or integration contract can meet it. The strongest design hypothesis is a **small graph transition model with explicit physical history, optional persistent edge memory, multiscale communication and physically structured outputs**, evaluated inside the actual engine. A transformer is one candidate for communication, rather than a prerequisite for temporal memory.

The next decision should come from matched architecture and integration experiments. Producing a much larger corpus before settling the causal input contract, temporal representation and runtime budget risks collecting expensive labels for the wrong problem. This review does not resume training or promote a checkpoint.

## What our experiments already establish

The repository evidence is recorded in [STEERING.md](../STEERING.md), especially the V26–V34 sections, and [COMPARISON.md](../COMPARISON.md). Those results establish several requirements independently of the papers:

- **Correct endpoints are insufficient.** An oracle-assisted endpoint adapter still produced incorrect motion and energy. Forces, fracture timing, integration and coordinate transfer must agree during the interval.
- **A working interface is not a working model.** The native stress-provider oracle control reproduces the reference path exactly, while independent learned trajectories fail badly.
- **Early force errors can create irreversible damage.** Oracle continuations from learned states still fracture heavily. Changing a final break threshold cannot undo damage already induced by incorrect state evolution.
- **One metric can conceal failure.** In the rejected V31 runs, projectile final separation was only about half a millimetre after 8 ms, while the largest surviving structural fragment held 5.97% of mass instead of 83–86%. Actor motion and structural integrity must both pass over a meaningful duration.
- **More force weight did not solve rollout.** V33 improved held-out force RMSE but retained catastrophic fragmentation. V34 admitted verified corrective states, yet naive mixing worsened several metrics.
- **Representation and loss interact.** Over 99% of the frozen model's corrective loss came from a signed bookkeeping residual-rate channel. On those branches, 94.13–99.65% of its absolute integrated value occurred in the first native phase. A smooth interval-rate head is poorly suited to that transient. This is not permission to clamp or erase it.
- **Temporal resolution is not the only bottleneck.** Offline fits of the same 16 force knots approximate the oracle much better than current forecasts. More knots alone are not the first remedy supported by that experiment.
- **Rows are not independent scenarios.** The complete native-phase model used ten house cases, with six training parents. Its millions of correlated bond rows do not supply broad impact, material, support or geometry coverage. The audited 506-record collection is not equivalent to 506 scenarios used by this latest trainer.

The latest complete model forecasts the interval at its start and does not revise that forecast when new contacts appear within the interval. Explicit physical history persists across calls; a persistent learned recurrent state does not. Both facts matter when selecting the next architecture.

## Research most relevant to this problem

These sources cover particle dynamics, deformable meshes, crash mechanics, material memory, fracture and efficient implementation. Results from fixed-topology deformation, prescribed actuators or a few thousand mesh nodes do not establish free-projectile destruction performance at our target scale.

| Research | Useful result and transfer limit |
| --- | --- |
| [Learning to Simulate Complex Physics with Graph Networks, 2020](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html) | The GNS work learns local interactions and rolls predictions forward. Message-passing reach and training noise strongly affect trajectory quality. A short physical history can support dynamics without an explicit temporal transformer. Its materials and particle benchmarks do not validate our fracture/contact coupling. |
| [Learning Mesh-Based Simulation with Graph Networks, 2021](https://arxiv.org/abs/2010.03409) | MeshGraphNets separate structural connectivity from proximity interactions. This supports typed bond/contact edges and relative geometry. Mesh deformation success does not by itself establish brittle failure, irreversible material history or independently moving debris. |
| [MultiScale MeshGraphNets, 2022](https://arxiv.org/abs/2210.00612) and [Bi-Stride Multi-Scale Graph Neural Network, 2023](https://proceedings.mlr.press/v202/cao23a.html) | Coarse communication paths address expensive long-range propagation. Bi-stride construction uses graph topology rather than only Euclidean proximity. For us, a latent hierarchy may accelerate communication while retaining every physical chunk, bond and contact; fracture must invalidate obsolete connectivity. |
| [Piers, 2024](https://doi.org/10.1016/j.cma.2023.116669), [author manuscript](https://ira.lib.polyu.edu.hk/bitstream/10397/116591/1/Chen_Predicting_Dynamic_Responses.pdf) | A particularly useful analogue: recurrent edge updates, physically initialized edge state and supervision of internal forces alongside accelerations. For beam cases, carried edge state represents internal forces and moments. Short increments make changes in interactions easier to learn. Its beam/plate/cloth demonstrations are not a complete fragmentation engine. |
| [ReGUNet, 2025 preprint and subsequent journal version](https://arxiv.org/pdf/2503.17386) | Recurrent edge processing at fine and coarse scales is applied to crashworthiness. This is a direct precedent for combining memory and hierarchy. Its fixed coarse meshes and reported crash benchmarks leave dynamic fragment topology and our runtime budget unresolved. |
| [MeshGraphNet-Transformer, 2026](https://arxiv.org/html/2601.23177v4) | Local graph processing plus pooled global attention supports crash deformation at up to roughly 100,000 mesh nodes. Dynamic inputs carry plastic hardening explicitly. The global tokens are spatial processing states, not persistent temporal memory. |
| [Transolver++, 2025](https://arxiv.org/html/2502.02414) | Adaptive slicing sharpens pooled physical tokens. Useful for testing whether our token bottleneck blurs distinct loads or components. Its results do not establish that a fixed small token count can preserve every stress wave or fracture precursor. |
| [Dynami-CAL GraphNet, 2025 preprint](https://arxiv.org/html/2501.07373) | Contact dynamics are expressed through edge-local exchanges of linear and angular momentum. This motivates a physically structured interface rather than unrelated node accelerations. Its edge memory within processing iterations should not be confused with an across-time GRU; the demonstrations do not cover our material fracture laws. |
| [Learning Memory and Material Dependent Constitutive Laws, 2025](https://arxiv.org/html/2502.05463) | Recurrent neural operators provide reduced internal state for materials with memory. The relevant lesson is that memory can represent unresolved internal variables; a fixed tiny memory is not proved sufficient for arbitrary material histories or changing timesteps. |
| [Thermodynamics-informed local graph networks](https://arxiv.org/html/2405.13093) | Local energy exchange and reversible/irreversible structure can guide learning. The examined degeneracy constraints are trained with penalties; they are not an automatic guarantee of exact discrete-time energy conservation. |
| [Projectile penetration PGN, 2025](https://doi.org/10.1016/j.ijimpeng.2024.105123) | A close task analogue: autoregressive projectile/target response and projectile velocity/acceleration evaluation. The authors relate graph communication radius to physical wave propagation. Evidence reviewed is the publisher abstract and indexed excerpts; its full implementation and conservation behavior were not independently audited. |
| [Fragmentation Graph Networks for blast-loaded reinforced concrete, 2024](https://doi.org/10.1016/j.engstruct.2024.118009) | A close destruction analogue using particle graphs and varied structural/loading cases. It supports looking beyond smooth deformation benchmarks. Only publisher abstract/excerpts were available for this review, so this is a research lead rather than a validated implementation recipe. |
| [DGAR for reinforced-concrete impact, 2025](https://doi.org/10.1016/j.compstruc.2025.107861) | Dynamic proximity graphs and particle erosion address evolving damage. Available publisher excerpts identify mass/momentum issues when eroded particles disappear without redistribution. We should not obtain visually plausible fracture by silently deleting physical mass. Full-text implementation details remain unverified. |
| [ADAPT-GNN, 2022 preprint](https://arxiv.org/html/2208.14364) | Phase-field fracture prediction uses displacement predictions in the subsequent damage prediction. This offers a useful staged-dependency idea. The demonstrated notched-specimen setting is substantially narrower than independent dynamic impact and free debris. |
| [Solver-in-the-Loop, 2020](https://arxiv.org/abs/2007.00016) | Training through a differentiable solver exposes the model to consequences of its own predictions. Our native fracture solver is not currently differentiable, so oracle branching and differentiable rollout losses are distinct techniques. |
| [Message Passing Neural PDE Solvers, 2022](https://arxiv.org/html/2202.03376) | Pushforward training uses model-generated states, and temporal bundling reduces repeated calls. Neither establishes that a precomputed force trajectory remains valid after an unexpected collision. Its training perturbations must not be confused with obtaining fresh oracle labels from a diverged fracture state. |
| [Under-reaching and the CFL condition, 2025](https://arxiv.org/html/2507.08861) | A wave-equation analysis makes the connection between graph reach, timestep and physical domain of dependence explicit. Its regular-grid bound is not a universal theorem for our irregular, fracturing graph. It supplies a useful capability test. |
| [ROBIN, 2025](https://arxiv.org/html/2506.06045) | Hierarchical graph processing plus rolling diffusion is a serious nonlinear mechanics challenger. The method amortizes denoising, so it should not be dismissed as necessarily requiring many full evaluations per physical step. Its window state, graph work and contact responsiveness still need deployment measurement. |
| [NVIDIA PhysicsNeMo MeshGraphNet engineering guide](https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html) | Concrete memory/compute optimizations include splitting the first concatenation-based edge projection and fusing operations. Training checkpointing saves training memory, not automatically inference time. CUDA-specific kernels do not establish equivalent benefits on our eventual deployment backend. |

The linked [crash-def-net implementation](https://github.com/stofe94/crash-def-net), previously inspected at commit `12cdc3b4198f0e5161a407ec2da3fe29c1531c10`, is useful experimental code, not a specification for our solver. In particular, its prescribed-actuator setting, all-pairs proximity construction and lack of persistent temporal state are not features to copy uncritically. Its output labelled velocity is a per-sampled-step displacement in the inspected data path; physical units and timestep must be explicit in our contract.

## First principles that constrain the architecture

### There are three different clocks

Physical simulation time, contact/fracture update times, and neural message-passing iterations are different quantities. Four neural layers do not mean four physical timesteps. A persistent hidden state can store history but cannot automatically transmit a new impact to distant nodes during the current prediction.

For a local graph, a useful diagnostic compares the distance reachable by message passing with the distance a stress disturbance can travel during the predicted interval. In a simplified uniform setting this resembles `L * edge_spacing >= wave_speed * interval`. This is a domain-of-dependence check, not a stability guarantee or a bound applicable unchanged to arbitrary fracture. Hierarchy and attention can bypass hop limits, but must still retain the amplitude, direction and phase information needed by the output. The [under-reaching analysis](https://arxiv.org/html/2507.08861) motivates this test.

There is also no theorem that the network must reproduce the oracle's instruction count. A learned transition can approximate the result of many numerical steps. Future collisions can be predicted from sufficient current state, but the network cannot condition on external changes absent from its state and prescribed forcing, distinguish physically different states encoded identically, or restore information discarded by an inadequate representation.

### Memory has several distinct meanings

The reference already contains substantial memory. Inspect [solver.rs](../../crates/stress-ref/src/solver.rs), [joint.rs](../../crates/stress-ref/src/joint.rs) and [world.rs](../../crates/stress-ref/src/world.rs):

| State | What must remain distinguishable |
| --- | --- |
| Rigid and internal mechanical state | Cluster poses, velocities, angular velocities, inertia, local deformation/rotation and their rates. Identical outer shape need not imply identical internal wave state. |
| Bond and material history | Plastic deformation, damage, crush, hardening, fatigue, rebar slip/work/failure, filtered strain rate and relevant previous stress. Unload/reload behavior depends on history. |
| Contact history | Contact identity, tangential/torsional state, indentation and evolving stick/slip information. A new contact must not inherit another contact's memory. |
| Topology and coordinate ownership | Which pieces belong to each rigid cluster, which interfaces remain connected, and how local state is transferred when fragments split. |
| Learned latent memory | Optional compressed information not represented sufficiently in the explicit state. This needs a defined initialization, transport, reset and restart policy. |
| Within-call scratch state | Intermediate graph features or attention tokens. These are not temporal memory unless deliberately saved and reused. |

With a sufficient Markov state, an across-time neural hidden state is not mathematically necessary. With compressed observations, it can be valuable. The right experiment is explicit state alone versus the same state plus a small recurrent edge state, rather than assuming recurrence is either mandatory or useless.

Piers supplies a concrete alternative to an opaque large memory: make carried edge quantities physically meaningful and initialize them from the actual state. Its force supervision is an empirical training device, not proof of conservation. [Piers manuscript](https://ira.lib.polyu.edu.hk/bitstream/10397/116591/1/Chen_Predicting_Dynamic_Responses.pdf)

For our model, stable chunk/bond IDs should own memory. Breaks retire material interactions; recontact creates or restores contact-specific state according to the physical law. Rotating vector features require coordinate transport at cluster changes. Scalar latent features avoid some transport issues, but cannot be assumed to contain all directional information.

### Contact feedback and material response form a coupled system

The causal sequence is loads and contact → deformation/rate → material response and history → forces/torques → motion → fracture and coordinate transfer → new contact. This is a dependency loop over physical time. Independent heads that each regress an endpoint can disagree about the same event.

An impulse alone determines a momentum change, but does not determine the displacement during the interval or the time of a collision/fracture. A model intended to skip internal steps therefore needs a valid interval transition, sufficiently resolved event timing, or a controlled correction mechanism. Correct total impulse does not justify teleporting to a predicted endpoint.

The initial architecture experiments should retain the verified native application path to isolate prediction quality. A later acceleration experiment must explicitly define which internal operations the learned transition replaces and prove that its coupling remains adequate. Keeping all native steps forever may prevent the intended speedup; skipping them before validating the transition reintroduces the endpoint failure.

### Physical structure should remove avoidable learning tasks

Known masses, inertia, frame transforms, incidence relations and support constraints belong in the deterministic application layer. The model should concentrate capacity on difficult response and evolution.

For pair interactions, represent a consistent wrench or impulse exchange. Equal-and-opposite forces alone do not guarantee angular momentum conservation for noncentral interactions: contact application points and couples matter. External loads and supports require their own momentum/work accounting. [Dynami-CAL](https://arxiv.org/html/2501.07373) provides a useful construction to study; our bonded, rotating fragments require a separate derivation and tests.

Damage irreversibility, valid connectivity and admissible material states can be enforced through a suitable decoder where the law supports them. This should not become arbitrary clipping of bad predictions. Our signed native accounting residual is not physical dissipation and cannot be made nonnegative merely because a thermodynamic loss expects it.

### Spatial communication must respect physical independence

Keep bond, contact and computational shortcut edges distinct. Nearby fragments do not transmit stress without contact. Coarse communication is an inference aid, not a new physical force path.

A hierarchy must update after splits. Token attention must not couple unrelated scenes or independent islands merely because they share a batch. Global load conditioning can be shared where physically justified, while disconnected unrelated systems need isolation. Test this by placing an unrelated object nearby or adding another independent scene and checking that the original prediction remains unchanged.

Relative geometry alone does not guarantee rotational equivariance for an arbitrary MLP. Compare rotation augmentation against a compact local-frame/vector construction, including degenerate frame cases. Use explicit units and timestep, and test consistent unit conversion and coordinate transformations.

Giving the model a timestep input does not establish timestep generalization. In smooth test cases, compare one interval with two half-intervals at the same final physical time, using a declared numerical tolerance. Across fracture, additionally check event ordering and coordinate transfer. Train and evaluate every intended interval regime rather than assuming a model learned at 1 ms works at 16.67 ms.

## Architecture shortlist

The proposed family predicts a **causal stress-state transition**. Start from our stable physical chunk/bond graph with rigid-cluster membership, rather than collapsing each deforming structure to one rigid-body node. The exact state inventory and ownership must be written down before implementation.

The common contract should contain:

1. **Inputs:** current rigid and internal mechanical state; material parameters and explicit history; rest/current bond geometry; current contact observations and retained contact state; supports; prescribed loads; time interval; stable IDs and ownership. No future oracle contact forces, trajectories or topology masks.
2. **Processing:** separate bond/contact encoders; local message passing; one tested long-range communication mechanism; optional small persistent bond memory; local refinement. Known scheduled forcing is valid conditioning, while unknown future contact must arise from the model world's own evolution.
3. **Outputs:** coupled response/history evolution and physically valid fracture events. Derive redundant quantities consistently where possible. Force/torque, impulse, work and motion must describe the same transition; predicting them as unrelated independent answers invites contradictions.
4. **Application:** retain deterministic contact/motion/topology ownership, explicit frame transfer and invariant checks. Define when predictions are refreshed from the actual evolving state. Report every fallback; do not hide it in model accuracy or speed.

| Candidate | Purpose |
| --- | --- |
| Explicit-state local graph baseline | Establish what better state, targets and training accomplish without adding learned recurrence or global communication. It is an important control, even if insufficient at long horizons. |
| Compact recurrent multiscale graph | Primary design hypothesis. Small persistent edge state plus a topology-aware latent hierarchy could preserve material history and communicate waves economically. Every original physical interaction remains represented. |
| Compact recurrent graph with pooled attention | Direct challenger. Replace the hierarchy with a controlled token mechanism, holding state/output contracts and compute budget comparable. Measure localized-load blurring and independent-island leakage. |

An initial engineering point might use width 64 and edge memory 16 or 32, followed by a small capacity sweep. These are test sizes, not an established optimum. Do not combine a hierarchy, transformer, large memory and diffusion into the first candidate: that would obscure which capability helps and its cost.

Keep a larger accurate teacher as an option for later distillation. Distillation cannot repair a student's missing causal inputs or insufficient representational reach. ROBIN is an interesting stronger-model challenger; whether its rolling state is useful for a compact student remains a hypothesis.

The work-rate head needs a separate representation audit. Candidate representations include an explicitly accounted initial transient plus a smooth remainder, or cumulative-work supervision with correct signed totals. These must pass native application tests. Do not assume improving this bookkeeping head will solve the separately established force-feedback error.

A useful feedback experiment is for the graph to predict a nominal response and a compact local correction conditioned on the difference between predicted and actual deformation/rate. This would test whether cheap corrections can respond to live contacts without rerunning the whole graph at every native phase. It is a hypothesis, not a validated approximation; retaining native integration still limits its possible speedup. The final accelerated transition must be tested separately against the complete runtime budget.

## Training objectives and data

### Use losses that constrain the transition

A useful objective should supervise compatible aspects of one evolving system:

- **State and response:** local deformation/rate, relevant history and force/torque, with physical scales and eligibility masks. Report each group's contribution and gradient magnitude.
- **Integrated response:** linear/angular impulse and cumulative physical work over intervals. These are often more directly tied to motion than pointwise force RMSE, but cannot replace event timing or state accuracy.
- **Events:** first disconnection and material mode changes, with irreversible consistency and explicit treatment of intervals with no event. Use timing-aware supervision rather than only a frame-end break bit.
- **Rollout:** short model-driven sequences, increasing only as stability improves. Choose checkpoints using independent physical outcomes as well as component error.
- **Consistency:** momentum/work balances, valid material histories and frame transformations. Apply hard constraints only where the actual physical contract justifies them.

There is no justified universal set of weights yet. Normalize by physical meaning and inspect gradient competition before tuning. The current corrective residual dominating almost the entire loss is evidence that a single undifferentiated normalized MSE is inadequate for these targets.

GNS found that appropriately perturbed training states improve rollout behavior; noise and targets must remain consistent. That does not justify perturbing arbitrary plastic history or relabelling a broken state with the future of an intact one. [GNS paper](https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf)

Use three distinct training mechanisms deliberately: teacher-forced transitions for learning local response; short model-generated sequences for accumulated error; and verified oracle branches from the model's actual training-parent states for corrective labels. In a nondifferentiable native engine, an external actor trajectory metric does not automatically supply a gradient. It can still govern model selection and data collection. With a validated differentiable rollout path, trajectory error can also be a training loss. [Solver-in-the-Loop](https://arxiv.org/abs/2007.00016)

Do not train the model to magically heal physically irreversible mistakes. Sample corrective states before catastrophic divergence as well as later states, retain their true histories, and evaluate whether the oracle continuation can still produce the intended outcome.

### Collect coverage and complete state

The next corpus should vary independent physical factors, using space-filling parameter designs plus deliberately dense coverage around qualitative transitions:

| Family | Useful variation |
| --- | --- |
| Impact | Location, angle, speed, mass, shape, rotational motion, friction and restitution; central, edge, corner and grazing contact. Include bounce, arrest and pass-through outcomes. |
| Structure | Thickness, span, openings, connectivity, joints, supports, material stiffness/strength/density, reinforcement, pre-damage and preload. Avoid impossible parameter combinations. |
| Blast and distributed load | Source location, duration, magnitude, temporal waveform, spatial distribution and support configuration. Distinguish equal impulse with different pulse duration. |
| History | Unload/reload, repeated impacts, reversed loading, oscillation, dwell, fatigue where supported, frictional slip/recontact and sequential failures. |
| Quiet and near-threshold behavior | Stable resting loads, elastic waves without fracture, nearly failing supports, weak impacts and recovery. An all-destruction corpus rewards false positives. |
| Coupled interactions | Multiple impactors, fragment-fragment contact, secondary impacts and collapse after support loss. Include longer post-impact settling. |
| Scale and topology | New layouts, new mesh/chunk resolutions, many independent islands and large active-body counts. Hold out entire families, not random rows. |

Record causal start state, time conventions, material definitions, stable identities, prescribed forcing, native phase states, force/torque and impulses, physical and bookkeeping work separately, contact histories, first-break times, frame transfers, actor trajectories, fragment ownership and complete provenance. Save sufficient restart state or a verified deterministic replay recipe; a visualization snapshot is not automatically a full training restart.

If recurrence is added, training windows need a warmup prefix or a validated state initializer. Randomly starting every window with zero hidden state would create a training/inference mismatch. Keep whole parent trajectories and their corrective branches in one partition. With many experiments against the current test cases, reserve a fresh final holdout before making a final generalization claim.

Generate a small contract-validation corpus first. Expand to large-scale generation only after restart, labels, timing, targets and application agree. A million redundant rows from one trajectory cannot substitute for another load regime.

## How success must be measured

Use a set of pass conditions rather than a single weighted leaderboard score:

| Dimension | Required evidence |
| --- | --- |
| External actor response | Independent position, velocity and orientation trajectories; penetration, rebound or arrest class; residual speed; contact duration and impulse. |
| Structural outcome | Fragment mass distribution, largest connected mass, separated pairs/connectivity, collapse extent, debris motion, false fracture and missed fracture. Compare event timing with declared tolerances. |
| Physical consistency | Linear/angular momentum including external reactions; energy/work accounting; nonnegative physical dissipation where applicable; finite state; no spontaneous healing or mass deletion. |
| Temporal stability | Performance over impact, separation, secondary contact and settling; failure rate and worst cases, not only short average error. |
| Generalization | Entire unseen geometries, parameter ranges, histories and scale regimes; multiple seeds; no holdout-driven tuning. |
| Runtime | Total engine time per simulated second and frame latency at 10,000+ active rigid bodies on target hardware, including graph work, integration, contacts, topology and transfers. |

Oracle similarity and real-world accuracy are separate claims. If the reference material laws, damping or contact parameters are wrong, a perfect imitator is wrong in the same way. First validate reference invariants and convergence; ultimately calibrate representative impact and fracture cases against experimental or otherwise independently validated physical data. The known signed-residual issue deserves investigation, not conversion into an unquestioned physical law.

Quantitative acceptance thresholds should be declared before selecting the final model. They depend on the intended operating range and application tolerance. Neither strict bond-by-bond agreement nor visual plausibility alone is sufficient.

## Performance constraints for a small model

Small parameter count is not a runtime guarantee. Graph width, edge count, memory traffic, number of sequential passes, graph rebuilding and prediction frequency can dominate.

The MGN-T paper reports **4.1 ms model time and 15 ms full-step time** for Pi-Beam on an RTX 4090, while advancing **1.5 ms physical time** per step. Thus its reported full step is about ten times slower than real time, despite a large speedup over FEM. Its mesh-node scale is not an active-rigid-body benchmark. [MGN-T, sections 4.2 and 5.5](https://arxiv.org/html/2601.23177v4)

Our relevant accounting is:

`total_time = contacts + graph_and_features + model_calls + integration + topology + transfers`

`real_time_ratio = elapsed_wall_time / simulated_physical_time`

A ratio below one is necessary for simulation to keep up, with additional headroom needed for rendering and application work. For an illustrative 60 Hz application, the whole frame has 16.67 ms; this is not automatically available to each neural call. Measure tail latency during fragmentation, not only steady averages.

The native V27 diagnostic prefixes attributed about 9.2% of house time and 41.7% of wall time to bond response including application. Those were instrumented concurrent runs, not scale benchmarks. They nevertheless warn that accelerating only local constitutive algebra may leave most cost untouched. Obtain a clean profile at target scale before choosing the replacement boundary. [Local evidence](../STEERING.md)

Design and measure with actual counts of rigid bodies, chunks, active bonds and contacts. Sparse graph processing avoids all-pairs contact construction. A latent hierarchy preserves the physical mesh; it must not silently decimate collision shapes, discard contacts or alter tolerances. Keep feature construction and inference resident on the deployment device where practical. Benchmark the intended backend rather than assuming training-GPU timings transfer.

For typical dense edge MLPs, work grows approximately with `passes * edges * width^2`. Pooled attention also pays for projecting nodes into and out of tokens; its cost is not just the small token-attention matrix. Shared weights reduce parameter storage without removing repeated computation. Persistent memory consumes bandwidth on every update: 100,000 edges with 16 float32 memory values would alone occupy 6.4 MB, before other state and temporary buffers. This is an illustrative count, not a measured Rubble scene.

Start with a reliable precision baseline. Quantization, fewer contact updates or a longer physical step are accuracy tradeoffs to measure explicitly, not silent performance improvements. Exact algebraic refactoring and reduced copies are preferable first steps; floating-point parity still needs verification.

## Experiments that can change the decision

1. **Freeze and audit the physical contract.** Specify state ownership, units, clocks, event ordering, contact feedback and restart. Resolve or separately characterize the reference energy issue. Keep the existing exact oracle-provider control as a regression gate.
2. **Run capability probes before large training.** Use a single joint with unload/reload, a wave-carrying chain/beam, a supported structure, a free impactor and a break-then-recontact case. Measure wave arrival/amplitude/reflection, hysteresis, actor impulse and energy. Include same-observation/different-history pairs to test state sufficiency.
3. **Compare the three architecture candidates under matched conditions.** Same causal fields, targets, partitions, physical application and comparable runtime. Ablate recurrence, communication reach and feedback separately. Use more than one seed before attributing a modest gain to architecture.
4. **Resolve the temporal representation.** Compare fixed forecasts with feedback from actual evolving kinematics, under the same native physics settings. Independently test the initial-transient/cumulative-work representation. Preserve exact event and signed-work accounting.
5. **Prototype runtime at scale early.** Include 10,000+ active bodies and fragmentation/contact bursts. Reject an architecture that cannot fit the full-engine budget before commissioning a huge dataset for it.
6. **Expand coverage and corrective training.** Generate balanced physical regimes and verified branches from training parents. Increase rollout horizons and collect states near the first divergence, not only already-destroyed endpoints.
7. **Compress only after capability is established.** Distill or shrink the strongest stable model, measuring the accuracy/latency frontier on the actual backend. Retain all structural, actor, physical and runtime gates.
8. **Evaluate once on a fresh final holdout.** Use independent worlds, zero oracle resets, explicit fallback accounting, longer trajectories and worst-case failures. Follow with external physical validation for any claim about realism beyond oracle imitation.

The evidence supports confidence in learned acceleration as a research direction. It does not yet support confidence in a particular tiny architecture or in real-time performance for our complete use case. The immediate aim is to identify the smallest model whose state, communication, coupling and runtime are demonstrably sufficient; large-scale data generation should then reinforce that proven contract.
