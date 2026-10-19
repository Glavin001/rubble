# Steering rubble with learned-impact

**Target architecture:** the learned net is a **swap-in for the dynamic stress solve on impact islands** (not D6). QS still carries gravity/resting load. On impact: MGN-T predicts **which bonds fracture** (fail-bin / util / mode) and **what to remember into the next frame** (joint-state / force carry + chunk Δstate). Cluster rigid bodies **split** when bonds cut via the existing layout/spawn path.

Today **Bonds → Learned impact** is HUD-only on the unbonded rubble path. **Bonds → Stress solver** already does cluster rigid + QS/cascade + splits. Coupling = wire learned preds into that break decision.

## Do not use

- **D6 / soft welds** for learned steering — too jello-like; wrong mechanical model for masonry clusters.
- Replacing the whole AVBD *contact* solve with the GNN.

## Recommended path

### 1. Feature parity (required)
Feed the same tensors training saw: `chunk_load_bins`, QS/contact forces, joint/bond state, velocities — from the stress-gpu / stress-ref mirror each tick (not sparse pack zeros). Otherwise fail-bins stay near “never fail.”

### 2. Fracture gate on the stress path
Keep: QS (or stress-gpu QS) for gravity / resting load; cluster rigid bodies; connected-component **split** when bonds cut.

Replace (or gate): the analytic bond-break step with learned **fail-bin / peak-util**. On impact islands / when `rigid_assumption_fails`, run residual MGN-T and break bonds the network marks; then run the existing split/layout/spawn pipeline unchanged.

Optional viz: tint bonds by util before breaks land.

### 3. Fallback
Low-confidence or OOD bonds → existing stress cascade / capacity break (never silent wrong). Do not fall back to D6.

### 4. Demo wiring
Fracture lab **Learned** should eventually share the stress-solver world (clusters + rubble adoption), not `withoutJoints`. HUD-only pack graph is a speed/weight smoke only.

### 5. Engine residency
Same `learned-impact-slang/1` weights in the resident GPU frame with the cascade (M0); fixed-order CSR; bit-identical repeats.

## Acceptance
- A punched/rammed wall or house **splits cluster rigid bodies** where learned fail preds fire, matching stress-mode look (hard pieces, not jelly).
- Holdout punch F1 stays meaningful on teacher-forced eval; coupled demo is qualitative + optional oracle fragment/speed compare.

## Prototype status (2026-10-19)

Incremental **Stress + learned break gate** is in-tree (see `findings/stress-learned-gate-2026-10-19.md`):

- Hook: `ReferenceSolver::break_bonds` ← `GpuWorld::apply_external_breaks` ← stress-web `stress_break_edges`.
- Lab mode **`stress-learned`**: clears analytic `fracture` for that session only (BlastParity profile unchanged); JS runs MGN-T on the lab device from a CPU feature capture, then breaks.
- Gaps: `chunk_load_bins` still zero; `bond_qs_force` = live `bond.force`. Next: bins + resident GPU MGN.


### Causal-input experiment

The Python `physics-preview-v3` architecture and recorder `--rigid-preview` option
separate runtime-computable contact preview inputs from oracle interval-force
labels. `World::learned_rigid_preview` runs the original contact solver on a
frame-start clone with frozen local elastic geometry; it does not run dynamic
bond responses. Oracle recording remains unchanged and is checked byte-for-byte.
This preserves the contact/stress boundary, but the preview is approximate and
expensive. It is **not connected to the engine**, exported to Slang, or validated
as a closed-loop state update. See `DIVERSITY.md` and the matched experiment in
`COMPARISON.md` before using its offline scores as deployment evidence.

### Current state-carry experiments and remaining runtime contract

V15 makes fracture connectivity agree with decoded damage/rebar state. V18 adds
all 18 cached total/elastic force, stored-energy and stress-measure outputs using
the exact response sidecar. Neither is wired into a live World or exported to
the legacy Slang format. Current response accuracy is inadequate even though
CPU/Metal inference agrees; see the exact-response audit in `COMPARISON.md`.

A read-only native snapshot bridge now shares frame-start extraction with the
recorder and passes exact input/prediction parity on the wall-impact integration
(`runs/runtime-start-v20-2026-10-19/`). It preserves stable graph ordering and
exact joint/response observations without future labels. The next runtime work
must define the application phase while preserving the contact/stress boundary. The native Python
branch currently predicts an interval using start state and prescribed forcing;
it is not proof that contact feedback is handled correctly in a coupled frame.
Do not treat oracle interval contact loads as causal inputs or silently replace
the AVBD contact solve with learned world-motion outputs.

Cached bond response is only part of the necessary state. Dynamic geometry,
rigid-cluster state, contact caches, coordinate transfer at splits, accumulated
energy and integration phase still need explicit parity/application checks.
Retirement must use predicted connected components, and cached forces on retired
interfaces are not continuing external forces. Applying damage first and then
calling an old break routine that skips already-disconnected bonds can bypass
the split path. A successful offline endpoint/fragment audit does not validate
that application order or an independent trajectory.

### User-prioritized independent rollout comparison

The next integration deliverable must initialize two identical worlds once and
then advance them independently: oracle dynamic stress on one, the current
learned dynamic stress model on the other. The model world must capture its own
positions, velocities, projectile state, joint history, response caches and
connectivity each interval. It must never reset to oracle states, adopt oracle
breaks/activity masks, or read future oracle contact loads. Keep normal collision,
contact, rigid-body motion and fragment creation; inference replaces dynamic
stress evolution, not the entire physics engine. Any analytic fallback must be
explicitly labeled and counted; a hybrid break gate is not full replacement.

The existing web `stress-learned` mode is a legacy partial prototype: it retains
analytic stress evolution and has zero load bins / approximate QS features. It
cannot serve as evidence for current V18/V22/V23 deployment. The native start
bridge gives verified causal observations; state application remains missing.
Define endpoint versus substep application, stress/rigid-frame transforms,
contact handover, split timing and energy accounting before injecting outputs.
Never overwrite simulated rigid-body endpoints with a learned trajectory merely
to make an independent-looking replay.

Assess the requested run with shared clocks and cameras but independent bodies.
In addition to strict fracture timing, match each bond's first disconnection over
predeclared physical-time tolerances and report unmatched early/late events,
final fragment connectivity and mass distribution, impactor penetration/speed,
collapse extent, cumulative damage, energy drift and total simulation cost.
A missed teacher-forced break may be a delayed break in rollout, but only an
independent run can establish that: oracle-reset evaluation already removes
that bond from later first-failure eligibility. Maintain both evaluations.

### Independent endpoint experiment now available (V25, 2026-10-19)

`World::step_frame_external_endpoint`, `learned_impact_rollout`, and
`learned_impact.rollout` now keep a persistent native World. Only the scene and
checkpoint enter inference; current snapshots feed the next prediction, never
oracle future frames. Both diagnostic runs completed with zero analytical bond
evaluations. Separate post-run comparison requires identical initial worlds,
stable graph keys, matching clocks, finite snapshots and irreversible connectivity.
The additive `/rollout` viewer displays each world's own geometry and activity.

This closes the independent-state execution gap only for an explicitly experimental
endpoint adapter. It preserves native contact substeps, advances rigid bodies from
live loads, and applies predicted local increments over those substeps. Joint state
and cached response commit at the frame boundary; existing split/frame-transfer
routines handle fragmentation. Predicted force is a response cache/input for the
next interval, not a replacement force integrator driving each native substep.
No learned global world endpoint is pasted into rigid bodies. Unsupported coupling,
refinement, multirate and scheduled topology events fail clearly.

Remaining gaps: within-interval failure timing, supervised local increments across
split-frame resets, internal damping/softening work, support reaction accounting,
and momentum/energy consistency under learned hidden updates. This experiment
must not be adopted as the production stress path or used for speed claims.
A new post-run finite check was added after freezing these two jobs; every saved
snapshot of both original runs was independently validated before publishing.

House04, 120 ms, full128 V18: 914 predicted breaks vs 310 oracle, 331 vs 109
fragments, largest fragment 19.2% vs 67.1% of active mass. Same-frame bond F1
3.10%, within ±5 ms 27.94%, any-frame 39.54%. Wall19, 133 ms, wall V22:
255 vs 308 breaks, 34 vs 12 fragments, largest fragment 91.4% vs 38.6%,
any-frame F1 60.75%. Different checkpoints/scenes are separate diagnostics,
not matched architecture comparisons. Results expose accumulation and topology
errors; they do not isolate network error from endpoint integration error.

Next: isolate adapter error with oracle-supplied endpoints from the same live
start state (diagnostic control only, clearly labeled); define physically consistent
split-aware state targets; collect off-oracle states with fresh oracle roll-forward
labels; compare independent trajectories and final fragmentation during validation.
Do not relabel diverged learned states using the original oracle's future trajectory.

### V26 control rejects endpoint-only accuracy attribution

The oracle-assisted endpoint control is complete; do not repeat it or treat V25
trajectory error as network error alone. `learned_impact_endpoint_control` clones
each current adapter world, advances the clone with the reference solver and
supplies exact f64 endpoint local increments/joint/response values to the adapter.
No neural prediction error exists in this control. It remains an oracle-assisted
diagnostic, never learned inference.

House04 first 1 ms: identical 85 breaks and 30 fragments, but the car ends at
8.581 m/s rather than 13.430 m/s; mechanical energy is 172642 J rather than
266032 J. Velocity coordinate RMSE is 1.031 m/s. Wall19 first 16.67 ms: identical
308 breaks and 12 fragments but velocity RMSE 1.682 m/s, position RMSE 0.02798 m and
94,788 J versus 116,844 J. Exact fracture sets cannot compensate for wrong within-frame
motion and coordinate transfer. Raw end-minus-start local targets include resets;
current learned local losses exclude these intervals, so this is a representation
and integration control, not evidence of a model accuracy ceiling.

`World::step_frame_observed` and `learned_impact_trajectory_probe` now capture
native explicit substeps read-only. The ordinary path uses the same frame loop
with a no-op observer; a three-frame parity test compares complete recorded state exactly.
Real house04 first frame captures 150 snapshots (149 steps, 6.711 microseconds each)
and exactly matches the existing oracle endpoint. Its 85 breaks occur across 39
substeps between 0.1275 ms and 0.9799 ms. 665 chunks change cluster frames. At the endpoint,
chunk physical translational KE is 13,706.7 J, while local hidden translational KE
is 204.36 J. Most physical fragment motion is in the rigid state after frame transfer;
replaying local endpoint differences alone cannot restore it. Dense artifacts are
in `runs/dense-trajectory-v26-2026-10-19`, controls in `endpoint-control-v26-*`.

Next contract gate: distinguish stress evolution from native contact/rigid evolution
at their actual shared substep boundary. Record or derive stress-induced impulses
and pre-transfer deformation updates, together with fracture timing and dissipated
work, then require an oracle-provided control to reproduce physical motion through
splits before training that contract. Do not fix the preview by pasting learned or
oracle global trajectories into rigid bodies. Reusing world endpoint supervision as
auxiliary diagnostics is fine; it is not a validated engine update. Do not settle for
learning the cheap constitutive algebra alone if it cannot meet the required speed
improvement. Measure complete runtime costs at 10,000+ active rigid bodies.

### V27: native stress-phase control passes; learned integration remains open

`StressProvider` now supplies ordered bond responses at the native explicit
pre-integration phase. The engine retains force projection, motion integration,
support reactions, fragment frame transfer, split timing and contact handover.
Responses include joint history, total and elastic force, stored energy, stress
measures, dissipated work and the signed native overshoot accounting residual.
Malformed batches fail before stress/history commit with no analytic fallback;
a World error can follow contact updates, so discard that partially advanced World.
This interface currently requires standalone explicit single-rate dynamics without
refinement or scheduled topology events. It is not a trained model or a speedup.

The oracle-provided control passes eight house04 frames (8 ms, 1,167 provider
calls, 1,333,767 analytical responses) and two wall19 frames (33.3 ms, 8,055 calls,
14,852,951 responses). Every captured snapshot, event sequence/time and complete
stress energy ledger matches the normal native path exactly. Initial and end-frame
world poses/motion plus f64 joint/response histories also match the existing V16
recordings; the refactor did not change their physics. Frozen binaries, sources,
hashes, reports, snapshots and independent audits are under
`runs/stress-provider-v27-2026-10-19/signed-residual/`.
The final `guarded/` revision repeats and passes these controls after adding
irreversible-history and fracture-disabled checks. It rejects partial damage
healing as well as connectivity healing. Its three provider tests pass.

The first validator incorrectly required nonnegative overshoot. That field is the
native signed `frozen - stored - law` residual; it must not be clamped. Failed
preflights remain preserved. Physical dissipated/damping work and stored energy
retain nonnegative guards. Subsequent controls reveal negative residuals larger
than roundoff too: house minimum -0.0013214 J, 1.62% of that bond's stored energy.
The read-only energy probe captures all 40 negative house responses, including
32 beyond 1e-12 J on four bonds. All 32 involve increasing damage, unchanged crush
and axial compression. Total negative residual is -0.00574193 J. Direct joint-law
reproduction and the probe's final typed snapshot match exactly. The witnesses in
`energy-witness/` localize a reference constitutive energy issue; they do not fix it
or justify excluding these recordings. Do not silently clamp energy targets or
claim that every negative residual is rounding.

Native serial profiles put bond response **including application** at about 9.2%
of house time and 41.7% of wall time in these controls. These are instrumented,
concurrently executed diagnostic prefixes, not performance benchmarks. They show
why learning only the local constitutive calculation cannot establish the requested
large full-engine speedup, especially for contact-bound scenes. Preserve the full
dynamic solver replacement goal. Next work must define/train the deployable stress
evolution contract, retain causal contact feedback and substep fracture/transfer,
evaluate independent trajectories, and measure the complete runtime at actual scale.
The V18/V22 frame-end checkpoints do not provide this substep contract unchanged.

### V28: native-phase labels and rejected force pilot

`phase_history` records every native stress phase, full incoming/outgoing joint
history and response, work accounting, connectivity and chunk observations. Four
house scenarios (32 frames, 8 ms each) contain 5,444,313 correlated bond-phase rows
over 4,670 phases. Exact replay and history-chain audits pass. This is four
scenarios, not millions of independent examples. Future phase observations remain
labels, never frame-start predictor inputs.

The causal 16-knot force-only pilot selected epoch 57 by validation loss. Test
force RMSE is 3493.96 N versus 3075.52 N holding the previous response; torque
104.16 Nm versus 79.22 Nm. It is rejected. It lacks complete fracture/history/work
outputs and has not run as a deployed provider. The optimistic oracle-fitted
trajectory basis is a representation diagnostic, not a learned accuracy result.
Artifacts: `runs/stress-phase-v28-2026-10-19`; 147 Python tests pass.
The remaining 378 response enrichments are reconciled and verified (360 original
successes plus 18 recovered metadata failures), preserving original artifacts.

### Interaction outcomes govern model acceptance

The user's wall-ram example exposes a categorical interaction failure. At
133.33 ms, the oracle ram travels 1.681 m forward with +12.595 m/s forward speed;
the V25 model-and-adapter run travels 1.527 m backward with -11.501 m/s. Its 60.7%
break F1 does not indicate acceptable destruction. A source-hashed saved-frame
audit is in `runs/interaction-outcome-v29-2026-10-19/wall19-audit.json`.
This view is independent, not teacher forced; V26 proves its endpoint adapter
also introduces errors. Do not attribute the full failure to the neural model.

Accept checkpoints using independent interaction outcomes first: penetration,
rebound or arrest; shared-clock projectile position/velocity, residual energy,
and contact impulse/work. Validate outcome classification against scene geometry
and adequate simulation duration; a signed displacement alone is not penetration.
Require structural response and physically appropriate energy/momentum accounting
alongside projectile motion so weak or ghost walls cannot pass. For support cases,
use load-deflection, reactions, collapse onset and retained load capacity. Exact
bond identities/timing remain diagnostics and auxiliary training targets; permit
crack variation only when interaction and structural response remain credible.

Next: complete the learned response contract through the validated V27 interface,
then evaluate free rollouts with these acceptance requirements. Collect oracle
responses from clones of model-visited states, keeping those states separate from
the original reference trajectory. Train on these recovery states and increase
rollout horizons. The Rust fracture/contact path is not differentiable: adding a
trajectory number to a loss does not supply gradients. Outcome-based validation
and data aggregation are immediately usable; direct rollout-loss optimization
requires a separately validated gradient or optimization method. Hold out physical
scenario families and new seeds, emphasize cases around penetration/arrest
transitions, and retain nonfailure cases. Matching the reference solver is a
surrogate for reality and must eventually be checked against physical benchmarks.

### V29: interaction measurements published; impulse objective remains experimental

`interaction_metrics.py` measures shared-clock projectile position/velocity vector
RMSE with trapezoidal time weighting, signed travel/velocity along initial motion,
final separation, and final direction disagreement. It rejects mismatched actor
identities, clocks, initial states, mass changes and nonfinite trajectories.
Initially stationary and prescribed-motion actors are explicitly qualified. These
measurements do not classify penetration or turn total momentum change into contact
impulse. The independent viewer now shows the failure banner, final actor motion
table, and linked travel/velocity plots while retaining prior views. House04 has no
final direction disagreement but still has 1.255 m final projectile separation;
agreement in direction alone is not acceptance. V25 adapter limitations remain.

A matched 60-epoch, 960-update component experiment adds signed time-integrated
generalized bond-force residual supervision to V28's timewise force loss. The
zero-weight control reproduces V28 metrics exactly. Weight 1 reduces test force
RMSE from 3493.96 to 3152.23 N and bond-force impulse RMSE from 2.7164 to 2.3665 N s.
However, validation force/impulse errors worsen 3.1%/3.8%, and holding the previous
force still wins on test (3075.52 N, 2.1994 N s). No promotion. These are four
familiar house prefixes, oracle participation masks are labels, and these are bond
impulses rather than projectile/contact impulses. No independent rollout or speed
gain is demonstrated. Avoid repeated loss-weight tuning on this test case.

The 128 core and 378 reconciled response recordings now form a verified 506-record,
5200-interval view in `runs/response-reconciled-v28-2026-10-19/all-dataset-v2`.
There are 504 unique observation digests; two duplicate alias pairs remain explicit
in its quality review. Spin recordings remain separate, as does the native-phase
pilot. The first assembly incorrectly required recorder revision metadata to be
identical; the retry explicitly allows the documented v4-to-v5 enrichment transition
while checking all physical input fields and source hashes. Failed attempt retained.

V29 artifacts are under `runs/interaction-outcome-v29-2026-10-19`; 158 Python tests
pass, browser chart scrubbing and the private Tailscale route were verified. Next
work remains a complete causal learned stress evolution interface with independent
physical outcome validation, not a new claim of readiness from this component loss.

### V30: causal boundary kinematics improve the component predictor

`RuntimeGraph::capture_with_kinematics` optionally captures generalized bond
deformation and its actual kinematic rate in f64, before advancing the World.
It uses the configured corotational or linear geometry map; retired slots have
explicit zero placeholders. Existing snapshot capture incurs no extra work.
This is not an oracle stress query and does not read future contacts or targets.

Four original cases were independently replayed: all 36 snapshots match original
bytes, and 37,705 frame-start bond observations match the first native phase
bit-for-bit. Source hashes are checked before attaching inputs. The new encoder
combines scaled deformation/rate with stiffness-times-motion indicators. Input
scales use training observations only. A zero-input control has identical added
MLP capacity and initialization.

Across seeds 0/1/2, 60 epochs and 960 updates per arm, observed inputs improve
mean validation force RMSE from 2278.75 to 2225.45 N (-2.34%); test force RMSE
from 3539.87 to 3128.65 N (-11.62%), and test generalized bond-force impulse
RMSE from 2.7540 to 2.3168 N s (-15.88%). Force and impulse improve in each pair;
torque does not improve in every seed. Test still loses to holding the previous
response (3075.52 N, 2.1994 N s). Retain the inputs as an experimental feature,
not a promoted model or independent-rollout accuracy result.

Native-phase coverage now includes six additional existing house scenarios,
preserving pretrained partitions and original frame duration/native steps.
The combined verified view has ten scenarios, 80 frames (8 ms per scenario),
11,656 native phases, 13,473,259 correlated bond rows and 2,616 break events.
Partitions are six train/two validation/two test scenarios. These are richer labels
for existing cases, not millions of independent examples or new layouts. The
six-run ablation above used only the original four cases. The expanded view has
not yet been used for training. Artifacts: `boundary-kinematics-v30-2026-10-19`
and `phase-expansion-v30-2026-10-19`; combined data lives under the former's
`combined/`. Four native snapshot/phase tests and 159 Python tests pass.

Next priority is completing the interval response contract (joint history,
fracture, total/elastic forces, stored energy and work) through the validated
native phase integration boundary, using these causal inputs and independent
projectile/structural outcomes. The force-only head cannot be plugged in as a
solver. Do not substitute a constitutive-only fit for the dynamic solver
replacement or claim full-engine speed from a component benchmark.


### V31: complete native learned response runs, but model rejected

The expanded ten-case phase corpus now trains a complete 16-knot response head:
21 joint-history values, 18 response values, three work rates and failure/ductility/
mode outputs. Sixty epochs (2,880 updates) select epoch 56 on validation loss.
Loss is normalized residual MSE + 0.25 eligible flag BCE + 0.1 mode CE; training
inputs/scales exclude future oracle data. Oracle-state force RMSE is 1963.28 N
versus hold-previous 2274.96 N on validation and 2128.04 N versus 2442.20 N on test.
These component improvements do not qualify the model for use.

New native-phase trajectory integration applies learned forces, histories and
integrated work inside actual stress phases, while the engine handles contacts,
motion and splitting. There are zero analytical stress evaluations. Forecasts
remain fixed within each frame, so this still lacks feedback from new contacts
inside a frame. Wire format v2 fixes a discovered decoding mismatch: interpolate
signed-log histories before inverse-transforming, matching training. Predictions
from frozen training and current code are bit-identical on audited inputs.

Independent 8 ms runs decisively reject this checkpoint. House04 has 1003 extra
broken bonds and only 5.97% mass in its largest fragment versus oracle 83.04%;
house10 has 1085 extras and 5.97% versus 85.58%. Final projectile separation is
just 0.585/0.516 mm respectively. Short-horizon projectile agreement alone can
therefore hide catastrophic structural failure. Wall19 fails in its second frame:
non-finite energy/work and native local inputs outside finite float32 range. The
failed run is preserved, not truncated into a successful comparison. No speed,
conservation or production accuracy claim. Checkpoint be913c3b remains rejected.

Artifacts and full measurements: runs/complete-trajectory-v31-2026-10-19/
independent-review.json. The two completed rejected runs are available alongside
older views in the private visualizer. 163 Python tests and 12 native phase,
snapshot, provider and trajectory tests pass; these establish plumbing, not model
quality. Initial v1 preflight artifacts are retained as superseded diagnostics.

Next: localize the first independent divergence; replay original scene plus saved
learned forecasts to reconstruct a complete native World, verify replay parity,
then branch to the oracle from that same model-generated state for labels.
Investigate structural failure calibration and force/history/work coupling before
another blind weight sweep. Distribution shift is a hypothesis, not yet a proven
cause. Acceptance must combine external actor motion, structural damage/topology,
energy/work and stable independent evolution over meaningful durations. Eventually
benchmark full runtime at 10,000+ active rigid bodies, not merely graph nodes.


### V32: replayed-state oracle branches isolate force-driven divergence

The prior turn made progress: complete learned native runs exposed severe failures.
This turn adds `learned_impact_oracle_branch`, which reconstructs the complete
World from its original graph and saved forecasts, verifies every prefix snapshot
byte-for-byte, and verifies one additional saved learned step from a cloned state
before producing oracle supervision. It never reconstructs a World from snapshot
fields alone. Branches save exact input snapshot, boundary kinematics, all native
phase labels, endpoint, events, execution report and provenance. A test compares
an oracle branch against an independently cloned native World and rejects a valid
but tampered forecast before producing labels. The initial graph comparison used
an inappropriate serde_json Value equality for f32 statics; six failed attempts
are retained, with no labels admitted. The corrected byte comparison passes.

House04 and house10, each at learned frame boundaries 0/1/2, produce six audited
oracle branches (800,266 native bond rows). All replayed snapshots plus the next
learned step match. Frame-zero oracle labels and endpoints reproduce the original
oracle exactly; all boundary kinematic rows match first native phase inputs.
These are diagnostic test scenarios, not additional training admissions.
After two learned frames, oracle continuation breaks 731/740 more bonds in one
millisecond versus learned 796/783. Earlier errors have already created a state
that the oracle itself predicts will fracture severely. This is evidence against
fixing only the final fracture threshold.

`learned_impact_channel_probe` further isolates first-frame response channels,
followed by seven fully native oracle frames. All-oracle controls match all nine
original snapshots exactly. Total broken bonds at 8 ms, house04/house10:
- Original oracle: 257/170.
- One learned frame then oracle: 365/346.
- Oracle force/elastic/stored/measures/damping channels, learned history: 267/188.
- Learned force channels, oracle history/dissipated/signed work: 375/361.
These deliberate mixtures may be physically inconsistent and are not deployable
models, training labels, accuracy improvements or speed results. Channel coupling
prevents additive causal attribution, but both cases implicate force errors and
the resulting state evolution as a major failure path.

Offline weighted least-squares fits of the first oracle force interval using the
same 16 knots give 89.69/91.40 N force RMSE on house04/10 versus actual V31 forecasts
6050.24/3472.39 N. The train case house00 projection is 277.45 N. These use future
oracle targets and are representational error floors only, not inference. More
knots are not the first bottleneck established by these results.

Artifacts: `runs/oracle-branch-v32-2026-10-19`. No checkpoint was trained or promoted
this turn. Next controlled training experiment should prioritize force accuracy
and its induced motion errors in the complete model, with matched controls and
independent trajectory gates; inspect phase-sensitive force feedback if fixed
forecasts remain unstable. Obtain any corrective training branches only from
parent training scenarios. Do not add these two diagnostic test cases to training.
The full dynamic solver replacement and 10,000+ active-body speed target remain
unmet. Existing visualizer models retain their rejected status.


### V33: force-weighted objective improves offline force error, fails rollouts

The prior turn made progress by isolating force-driven state divergence. This turn
runs a fixed two-arm comparison on the same ten-case corpus: seed 0, 60 epochs,
2,880 updates each, identical V24 initialization, 16 knots. The candidate adds one
normalized total force/torque MSE term to the complete-response loss. This raises
that group's direct output gradient eightfold; other supervised fields remain.
It is not an acceleration, projectile, rollout or energy-consistency loss.
Both arms select using the same original validation objective, fixed before runs.
The zero-weight control reproduces all V31 selected weights bit-for-bit.

Both select epoch 56. Force-weighted validation force RMSE improves 1963.28→1813.24N
(-7.64%) and test 2128.04→2067.55N (-2.84%). Combined validation objective worsens
0.45143→0.48964. Training force error also worsens 2368.09→2433.49N. Test results
are assessment only; no weight sweep or checkpoint selection from test cases.

Six complete independent 8 ms runs assess both arms on train house00 and validation
house12/16. Control extra breaks: 937/1078/1084; candidate: 937/1078/1084. Largest
remaining mass stays 5.97% in every run versus oracle 66.17%/88.14%/81.89%. House12
projectile final error worsens 3.248→4.076mm and velocity RMSE 0.4846→0.5522m/s.
House16 velocity RMSE changes 0.28008→0.27949m/s while final separation worsens.
No analytical fallback in any run. Mechanical energy not exceeding its initial
value does not establish energy consistency. Candidate rejected, no promotion,
no accuracy or runtime-speed claim; default force_weight remains zero.

Separately collected 12 one-frame oracle branches from the V31 model's own states
after 1/2 ms across all six parent training scenarios. These contain 1,304,746
correlated native bond rows and 1,757 phases, not new physical layouts.
They retain all phase inputs/targets, boundary kinematics, joint/work histories,
poses, projectile motion, events and provenance. `branch_dataset.py` audits source
registry membership, original scene identity, learned replay/control identity,
phase history, current kinematics and hashes. It explicitly rejects validation/test
parents. All 12 passed; held-out V32 diagnostic branches remain excluded. The new
corpus was NOT used by either matched training arm and is not yet consumed by the
trainer. Manifest: runs/force-objective-v33-2026-10-19/training-branch-manifest.json.

166 Python tests pass, including force-gradient/mask behavior, altered-parent
rejection and held-out partition exclusion. Artifacts and exact comparisons are
in runs/force-objective-v33-2026-10-19/review.json. All training, collection and
rollout jobs are terminal. Existing viewer models keep their rejected status.

Next priority: causal force feedback inside the evolving interval, coupled with
verified corrective training states. Investigate a cheap local correction around
predicted nominal deformation/rate rather than another blind weighting sweep.
Preserve native phases and contact fidelity, avoid oracle inference fallback,
and compare independent structural/actor/energy outcomes before adoption. The
requested complete dynamic solver replacement and 10,000+ active body realtime
speed requirement remain unmet.


### V34: corrective-state admission works; naive mixing fails on residual-rate loss

The prior turn made progress by rejecting force weighting and collecting verified
corrective training branches. `branch_dataset.load_records` now explicitly admits
such a corpus after fresh registry, parent, input, label, kinematic and hash audits.
Validation/test parents and duplicate parent-frame identities reject. Records retain
parent scenario identity and use only their current snapshot/kinematics as inputs.
Shared record builders preserve the original data path and guard checks.

Matched control/corrective runs use seed 0, 60 epochs, 2,880 updates, identical
initialization, original-only normalization and unchanged complete-response loss.
The candidate replaces 12 of 48 updates per epoch with the twelve corrective
branches; replacement positions rotate and branch shuffling is independent. No
extra training budget or held-out branch admission. Selection remains the original
validation objective. The control selected weights reproduce V31 bit-for-bit.
Control selects epoch 56; corrective selects epoch 7. Validation force RMSE worsens
1963.28→2078.77 N, test 2128.04→2351.68 N; validation objective 0.45143→0.59433.

Six independent 8 ms runs assess train house00 and validation house12/16. Corrective
extra breaks decline 937/1078/1084→885/1024/1038, but largest remaining mass remains
5.97% in every case versus oracle 66.17%/88.14%/81.89%. Projectile final errors worsen
2.048/3.248/2.108→3.615/6.543/2.535 mm; velocity errors also worsen in all three.
All runs complete with zero analytical fallback. Candidate rejected; no promotion.
This does not prove corrective-state training is intrinsically ineffective.

A concrete objective/representation mismatch is exposed: frozen V31's corrective
loss is 7467.26, of which the signed residual-rate regression contributes 7446.79
(over 99%). Corrective force error is 9776.97 N. The signed residual is native
bookkeeping, not physical dissipation; it must remain signed. On all twelve
branches, 94.13–99.65% of its absolute integrated value occurs in the first native
phase. A smooth 16-knot interval-rate output poorly represents that concentrated
initial transient, and this target dominates optimization. The selected corrective
model's corrective objective decreases to 5922.75 but force error worsens to
9954.58 N. Do not silently clamp, delete, or relabel these targets to hide failure.
Next investigate an explicit initial-work transient and/or cumulative-work
supervision, with conserved signed totals and native application tests, alongside
force feedback from the actual evolving state. No additional blind weight sweep.

Native-phase coverage also expands beyond houses: training wall00 at its original
16.67 ms interval has 4029 phases, 6,633,694 response rows and 421 breaks; training
blast01 at its original 1 ms interval has 128 phases, 166,272 rows and one break.
All native phases are retained. Both reproduce original oracle world trajectories, retain audited native histories,
and have independently captured matching boundary kinematics. These are richer labels
for two existing physical scenarios, not newly permuted layouts. They are a
separate verified view under native-diversity/, not used by this matched experiment.
Combining broader horizons requires explicit memory/representation planning; do
not decimate native targets or silently change physics timesteps.

Artifacts: runs/corrective-state-v34-2026-10-19 (review.json,
corrective-field-errors.json, residual-transients.json, native-diversity/review.json).
167 Python tests pass. All training, rollout, collection and audit jobs are terminal.
The complete solver replacement, diverse long-horizon accuracy and 10,000+ active
rigid-body realtime speed requirements remain unmet.

### V36 architecture stage: recurrence and multiscale communication work, native model rejected

Existing work was checkpointed in `6622e05a` before building a new 145,824-parameter
processor with explicit stable-ID edge memory, topology-aware latent hierarchy,
chronological training and exact topology caching. It reuses audited causal
features and the complete native response decoder. See
`findings/recurrent-multiscale-v36-2026-10-19.md` for the full experiment and limits.

Two seeded analytic wave experiments reduce held-out state MSE by 47–51% versus
local communication. Paired outputs preserve momentum, but a separate energy
audit finds large errors; this is a capability result, not physical acceptance.
The actual joint oracle also demonstrates opposite friction forces at identical
current geometry after opposite slip histories. Explicit physical history can
supply this memory; the witness does not prove recurrence mandatory.

Three native arms use the same existing ten-house corpus, seed 0, 60 epochs and
720 updates: local stateless, multiscale stateless and multiscale recurrent.
The loss/temporal decoder remain unchanged to isolate the processor. Recurrent
validation/test force RMSE is 1867.39/2280.32 N versus local 2015.90/2523.33 N.
All nine independent 8 ms native runs complete without oracle reset/fallback,
but all over-fracture severely. The recurrent model leaves 5.97% largest mass
on houses00/12/16 versus oracle 66.17/88.14/81.89%. No promotion.

Metal processor timing is about 4.35 ms for a synthetic 10k-node/19,989-edge graph,
excluding encoders, recurrent cell, heads, contacts, integration and topology.
This is not 10k active bodies or proof of real time. Hierarchy construction costs
about 24 ms in that run; unchanged-topology lookup is 0.032 ms. Cached native
replay preserves nine snapshots and eight predictions byte-for-byte, and saved
model-generated inference inputs reproduce all eight predictions exactly.

187 Python and 13 targeted Rust tests pass. The main next gaps are coupled
force/motion/work outputs, live within-interval feedback, separate material/contact
connectivity, the initial signed-work transient, recurrent corrective prefixes and
complete-engine timing. The current hierarchy follows material connectivity;
closed cracks still carrying contact must receive a separate contact path.
Artifacts: `runs/recurrent-multiscale-v36-2026-10-19/`. All jobs are terminal.

### V37: physical state is primary memory; linear mechanics pass, full solver still open

User clarified the baseline: advance explicit physical graph state rather than
depending on latent temporal memory. See `PHYSICAL_STATE_TRANSITION.md` and
`findings/physical-state-v37-2026-10-19.md` for the contract and measured evidence.
The whole-world target is ≤5 ms across hundreds of structures, hundreds of
thousands of bonds and ≥10,000 active rigid bodies. User subsequently specified
120 Hz to leave GPU headroom alongside likely 60 Hz PhysX: test 8.33 ms physical
updates under the 5 ms ceiling. Final GPU remains unspecified; local M3 Max
measurements are provisional. Do not change production timesteps implicitly.

An oracle-assisted inactive-damage intervention is mixed and insufficient.
Energy projection stabilizes generic explicit-state wave predictions but does
not recover accurate phase. A shared fitted C/S graph propagator succeeds on
linear elastic sequences without latent memory, including variable masses,
supports, prescribed external work and native six-DOF bond mechanics. Degree 16
uses 32 coefficients; float32 native linear state energy error is at most 5.66e-9
over 32 transitions in these probes. These are small linear tests, not nonlinear
native destruction, matched-budget neural training or a speed result.

The cost limitation is explicit: the current spectral domain requires 137–230
calls per illustrative 16.67 ms frame on the stiff small fixtures, and 30 calls
on house00, before contact/fracture work. Do not promote the polynomial baseline
as the final fast architecture. It establishes consistency and representation.

The new optional `--physical-history` phase recorder retains raw f64 chunk state,
full inertia/support identity and ordered contact histories. Native tests show
equal present contact force/energy/stiffness can conceal different future
responses after area change. Two recorded house frames retain all 13 legacy
files byte-for-byte, across 298 phases and 365,379 response rows. These sidecars
are not yet training inputs or complete World restart state. Do not manufacture
missing history from old snapshots; collect it through verified native replay.

The same fitted coefficients now pass a simple 1-D coupled-impact probe, with
live unilateral contact and separation/recontact events. The frozen-current-force
control produces 20–38% energy drift; live contact keeps projectile position RMSE
≤1.84e-8 m against a converged independent force-law reference. A two-mass test
matches analytic collision duration/velocity exchange. This has no friction,
fracture or native destruction coupling and is not a speed result.

Resident Metal at 102,410 chunks/200,046 bonds costs 303.34 ms per 0.512 ms
physical transition in the initial implementation. Equivalent streamed polynomial
fields and cached mass-normalized bond rows reduce it to 75.25 ms (4.03× faster).
All interactions retained; this is algebraic optimization, not a physics change.
CPU/Metal parity and repeated f64/f32 audits pass. Neither timing is a full-world
or 10k active-body benchmark; the implementation still fails the performance goal.
Next architecture choices need more than increasing polynomial degree or adding
latent memory. Actual damping, nonlinear material/contact events, long-range
propagation and a suitable fused runtime remain open.

200 Python and 11 targeted Rust tests pass. No checkpoint promoted; V36 native
destruction remains rejected. Artifacts: `runs/state-consistency-v37-2026-10-19/`.
