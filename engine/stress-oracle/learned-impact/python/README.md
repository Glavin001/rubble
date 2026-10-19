# Learned-impact PyTorch

Trains the MGN-T impact predictor and compares exact checkpoint outputs with recorded
oracle frames. See [COMPARISON.md](../COMPARISON.md) for corpus collection, all-data
training, checkpoint switching, and the local visualizer.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m learned_impact.train --smoke
```

Default smoke uses `H=8`, `P=4`; current training uses `H=48`, `P=16`.

The versioned physical-feature trainer keeps legacy checkpoints compatible:

```sh
python -m learned_impact.train_physics --dataset /path/to/dataset --out /tmp/physics-v2 --epochs 24
python -m learned_impact.coverage --dataset /path/to/dataset --out /tmp/coverage.json
```

This uses physical scenario groups for train/validation/test, saves the split and
source snapshots, and selects the epoch and break threshold on validation only.
`--arch legacy` runs the controlled baseline; `--positive-weight`, `--state-weight`
and `--history-weight` support objective ablations. `--all-data` is a final refit
for visualization, with no held-out claim. V2 is currently PyTorch-only; its outputs
have not been integrated as a replacement physical solver.


Causal-input experiment: `train_physics --arch physics-preview-v3` strictly requires
versioned `--rigid-preview` recordings. Missing preview data and cached features from
another input contract are rejected; oracle force histories are never substituted.
Use `preview_audit` for unchanged-oracle parity and input error/cost, and
`compare_inputs` for matched held-out checkpoint comparisons. The legacy `audit`
command now rejects physics checkpoints instead of applying legacy normalization to
them. See `../COMPARISON.md`: the frozen-deformation preview underperformed and has
not been adopted in the engine.

For matched cadence experiments, `train_physics --split-from /path/to/split.json`
inherits exact scenario membership and rejects missing, conflicting, or leaking
recipe groups. `--schedule-by-step`, `--eval-every N`, and `--evaluation-window N`
allow equal optimizer/validation budgets and common-duration break evaluation.
The latter unions stable bond IDs within each event's adjacent N-interval windows.
It does not turn teacher-forced inputs into a recurrent rollout.

Audit and timing tools (paths below are relative to `learned-impact/`):

```sh
python -m learned_impact.horizon_audit \
  --coarse runs/horizon-2026-10-19/coarse-dataset \
  --fine runs/horizon-2026-10-19/fine-dataset --out /tmp/horizon-audit.json
python -m learned_impact.benchmark_physics \
  --weights runs/horizon-2026-10-19/fine-model --out /tmp/scale.json
python -m learned_impact.mps_parity \
  --weights runs/horizon-2026-10-19/fine-model \
  --dataset runs/horizon-2026-10-19/fine-dataset --out /tmp/mps-parity.json
```

`benchmark_physics` measures synthetic 10k-node forward computation with CPU and,
when available, synchronized Metal paths. `mps_parity` checks real held-out frames
against the CPU checkpoint. Neither measures a coupled physical rollout or Slang
runtime. Preserve the recorded run/source files when interpreting results; source
and sandbox device visibility can change between experiments.

`train_physics --arch physics-preview-state-v4` is an experimental state objective
with the same causal inputs/capacity as preview v3. Motion starts at zero increment;
joint history starts at the previous state. Damage/crush/fatigue and the continuous
rebar-broken surrogate are bounded and irreversible; kappa, kappa_c, rebar work,
and dissipated history cannot decrease. Motion uses physical MSE scaled by four
training-only RMS groups; joint supervision uses encoded error scaled by training
history-increment RMS. Scales are stored as checkpoint buffers. These constraints
do not impose energy balance or convert the packed schema into a full solver state.

Use `compare_state --dataset PATH --weights BASELINE --weights CANDIDATE --out FILE`
to verify matched splits and report held-out fracture, motion, history error,
history constraint violations, and fragmentation-coordinate exclusions together.
Checkpoint selection still uses validation break F1, so state improvements must be
checked independently before considering promotion. V4 is experimental, not an
integrated oracle replacement.

For a controlled state-only follow-up, use `--freeze-backbone-from PREVIEW_V3_DIR`
with the v4 architecture. It copies and freezes the source trunk and break/timing/
utilization/force heads, inherits the source split by default, retains its decision
threshold, and fits only 1,666 motion/history parameters at H=48. Checkpoint selection
then minimizes validation `state_weight * normalized motion MSE + history_weight *
normalized history MSE`. This is an additional fitting stage, not an equal-total-
compute comparison with the source run. Verify retained outputs and state metrics
before using the resulting checkpoint.

## Causal world inputs and duration comparisons

`--arch physics-preview-world-v5` adds strict frame-start replay observations to
V4. `physics-preview-world-control-v5` disables those additional encoders for a
matched comparison. Both require valid scene/replay sidecars and save their hashes.
All projectiles are encoded; feature construction scales with nodes × projectiles.

```sh
python -m learned_impact.train_physics --dataset /path/to/causal-dataset \
  --out /tmp/world-v5 --arch physics-preview-world-v5 --epochs 24 \
  --split-from /path/to/reference/split.json
python -m learned_impact.compare_tails --dataset /path/to/extended-dataset \
  --reference-dataset /path/to/original-dataset \
  --reference-split /path/to/reference/split.json \
  --weights /path/to/short-model --weights /path/to/extended-model \
  --out /tmp/tail-comparison.json
```

Tail comparison verifies explicit recipe identities and partition membership.
It separates original-duration intervals from newly recorded tails at saved model
thresholds. Extra training data/updates are reported, not treated as equal-compute.
`mps_parity --gpu-features` checks raw-state feature construction and inference on
Metal. `benchmark_physics --projectiles 8` measures all eight projectile inputs;
reported inference timing does not include engine contact or state integration.
The visualizer accepts `--tail-dataset` and repeatable `--tail-weights` in addition
to its existing main and preview datasets. Viewer copies must retain exact event
IDs for saved split labels and stamp verified replay provenance.

## Training exposure across recordings

`train_physics --sampling physical-group` gives equal exposure to the physical
trajectory groups recorded in `split.json`, instead of weighting each trajectory
in proportion to its interval count. A shuffled interval queue per group persists
across epochs, ensuring long clips continue into their tails. It retains the same
number of optimizer updates per epoch as the default `--sampling frame` policy.
All intervals remain eligible; some are revisited more often than others. This
changes the empirical training distribution, not the recorded physics.

The default frame policy preserves the previous random ordering exactly, including
optional `--oversample-breaks`. Combining break oversampling with group sampling
is rejected to avoid an ambiguous exposure policy. Validation and test evaluation
always include their full saved partitions without resampling. Training-only
normalization scales are unchanged by the sampling policy, which keeps the
sampling experiment separate from feature/objective scaling.

`history.json` records per-group update counts and cumulative interval coverage;
`sampling.json` records every training interval's final visit count. Compare
policies at matched seeds, optimizer budgets and validation schedules. Equal
exposure also changes family weights when families have different clip lengths;
it is an experiment, not a claim that equal weighting is universally better.

`compare_state` also reports local motion errors above and below the saved
training RMS vector magnitude. Both strata include a zero-increment baseline,
physical per-axis RMSE, and chunk-interval counts. Fragment-coordinate resets stay
excluded, and empty strata report unavailable errors rather than zero. These
strata are diagnostics, not extra model inputs or checkpoint-selection criteria.


Explicit `context.generator.layout_group` now keeps all load recipes of a new
geometry in the same train/validation/test partition. Existing catalogs without
that optional field retain their original grouping. `--split-from` checks both
physical/trajectory and explicit layout identities and rejects leakage. New layout
experiments should save a split plan before inspecting outcomes and pass it via
`--split-from`; they should not repeatedly reshuffle until metrics look better.

Motion evaluation returns `null` errors when no valid local-coordinate targets
remain, along with scored/excluded chunk-interval counts. Viewer displacement
fields use the same mask and render coordinate resets in gray. World replay poses
remain visible and are not changed. `--scale-dataset`/`--scale-weights` expose a
separate physical scale diagnostic without mixing it into the training dataset.


## World motion across fragment splits

`--arch physics-preview-world-motion-v6 --world-motion-weight 0.1` adds an auxiliary
world-motion head. Weight 0 is the matched control. It retains all V5 inputs,
local motion/history outputs, and causal contact-preview requirements.
`load_event(..., world_inputs=True, world_targets=True)` derives versioned labels
from saved replay without re-running or changing the oracle. Only start snapshots
enter the encoders; the future endpoint appears only in `world_motion_target`.
Inference does not require these target tensors.

`world-ballistic-residual/1` means residuals from constant world linear/angular
velocity (no gravity acceleration in the baseline). Rotations use left-composed
principal SO(3) residual vectors, preserving targets across fragment-coordinate
resets. Training-only RMS scales are saved as checkpoint buffers. Evaluation
reports world endpoint residual RMSE and geodesic orientation error, with the
constant-velocity baseline and valid counts. This is not an engine integrator or
an autonomous rollout. The existing local displacement views retain their original
meaning and reset mask. The world displacement and error views display this
head's predicted endpoints with interpolation, resetting to oracle state each
interval; projectile motion remains recorded oracle motion.

## Exact joint-history inputs and matched update budgets

`load_event(path, exact_joint_history=True)` requires the new versioned sidecar;
it never reconstructs missing fields from the legacy clamped pack. Batches expose
`bond_exact_history` (31 finite start features) and `bond_exact_history_end`
(the target on the same live-bond ordering). The raw float64 values remain
available on `event.exact_joint_history`. V5–V7 models ignore these optional keys;
they have not been trained with an exact-history head. See `../SCHEMA.md`.

Use `--sampling physical-group --updates-per-epoch 1032 --epochs 24` for exactly
24,768 training updates regardless of corpus size. Queues persist between epochs,
and the run records interval coverage. Without this option the historical budget
is unchanged. Custom budgets are rejected in shuffled-frame mode; no examples are
silently dropped from eligibility. Checkpoint and threshold selection still use
validation only.

## Material descriptors and experimental gradient projection

`material_inputs.observations(scene, context, manifest, static)` produces 26
source-material features per chunk/bond plus 20 global constitutive settings.
It validates ordered source keys and bond endpoints. Friction, finite/unlimited
shear caps, brittle/ductile behavior, damping, DIF and fatigue laws have explicit
representations; absent optional laws are masked. This data is already present
in saved scenes. Existing models do not consume the new descriptors yet, and
they are not a complete engine-state contract.

`gradient_balance.projected_backward` is an experimental training update rule,
not the default trainer. It removes the part of the shared auxiliary gradient
opposing fracture BCE, while leaving private head gradients intact. A matched
four-epoch deterministic continuation regressed held-out F1 (0.535 ordinary,
0.517 projected), despite slightly higher validation F1. It is retained for
reproducibility and is not promoted. See `../COMPARISON.md` for the experiment.

### Oracle regeneration and physical-scale audit

A numerical source correction requires new recordings, not replacement labels
inside the old corpus. `learned_impact_regenerate SOURCE_EVENT FRESH_OUTPUT`
replays the saved source scene with its original timestep, solver methods, QS
probe and preview settings. It refuses existing outputs and verifies exact scene
and recipe preservation; historical rounding in descriptive recipe metadata is
retained explicitly. Full world replay, forcing descriptors and exact joint
history are recorded by the current recorder. Keep original train/validation/test
membership and source/binary fingerprints outside the event.

```sh
python -m learned_impact.replay_quality \
  --dataset /path/to/new-dataset --split /path/to/inherited-split.json \
  --out /path/to/new-quality-report.json
```

This reports active-body speeds/spins, energy growth, motion residual extremes
and per-event contributions to partition error. It does not filter data or claim
physical correctness. Interpret energy with external work and solver bookkeeping;
finite numbers or bit-identical replay alone do not validate the oracle. The V10
repair and versioned rebuild are documented in `../DIVERSITY.md`.


## Exact constitutive inputs and ordered fracture timing (V10)

`--arch physics-constitutive-native-v10` requires the full saved scene, world
replay, prescribed forcing, exact joint-history sidecar and validated source
chunk/bond mappings. It adds material/history encoders to native V7; their final
layers initialize to zero, preserving V7's initial predictions at the same seed.
The checkpoint has a separate architecture ID. Batch construction filters static
bond material and exact history by the same live bond IDs. Future endpoint labels
never enter these encoders. Missing data is rejected, not synthesized.

The output heads/loss retain V7 semantics for an input-only ablation. In
particular, the old packed endpoint head is not a lossless full-state decoder;
consuming exact history does not establish recurrent state carry or engine parity.

`--fracture-cdf-weight 0.3` enables an experimental ordered timing objective:

```
CDF[k] = sigmoid(break_logit) * cumsum(softmax(time_logits))[k]
truth[k] = 1 if 0 < failure_bin <= k else 0
loss_cdf = mean((CDF - truth)^2)
loss_total = baseline_loss + fracture_cdf_weight * loss_cdf
```

The default weight is **zero**. Every live bond contributes, including intact
intervals and missed breaks. An adjacent-bin confident miss costs less than a
three-bin miss. Class zero means no break during this interval, not permanent
survival. This is the discrete ranked probability score, based on cumulative
probability errors; see [Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
The score alone is proper, but adding it to weighted BCE and other objectives
does not guarantee calibrated model probabilities or improved fracture F1.

Evaluation now reports `fracture_cdf_score` across all bonds and separately on
breaking/intact intervals, independent of the chosen break threshold. Keep F1,
precision, recall, timing, state/history errors and per-geometry results alongside
it. Do not select a candidate merely because its weighted training loss is lower.

The V10 integration smoke fits eight corrected wall intervals and reloads the
checkpoint successfully; it is **not held-out accuracy evidence**. Before
promotion, compare (1) native V7, (2) V10 inputs with unchanged loss, and (3) V10
inputs plus CDF loss, holding corrected recordings, partition membership, seed,
optimizer updates and sampling policy fixed. Compare multiple training seeds
before claiming a small gain. Inspect extreme-force cohorts before fitting global
motion normalizers; do not silently delete them or clip their targets.

A further unresolved objective gap is accumulated rollout error. The present
trainer resets to oracle state each interval. Research on graph simulators shows
that training for robustness to accumulated input error can matter for long
rollouts ([Sanchez-Gonzalez et al., 2020](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html)).
That motivates a future state-carry/rollout experiment here; it does not validate
blind perturbations of damage, material history or fracture topology. Complete
state semantics and an engine adapter must come first.


## Runtime-alive graph and full joint history (V11/V12)

A recorded fracture disconnects a bond, but it does not necessarily retire that
interface from the stress solve. Cracked interfaces inside a still-connected
cluster can continue carrying compression/friction and updating history. In the
first 23 corrected recordings, the old first-failure graph discarded **4,969**
such bond intervals; every one continued changing history. The original
first-fracture model contract remains available for existing checkpoints.

`physics-history-native-v11` and `physics-history-residual-native-v12` instead use
`event_batches(..., bond_policy="runtime_alive")`. The exact frame-start `alive`
flag selects message-passing edges. The separate `connected` flag selects bonds
eligible for a first fracture. Already-disconnected interfaces retain their
material/history/force targets but contribute no new-break BCE, timing loss or
CDF loss. Evaluation uses the same eligibility mask for fracture metrics. The
saved labels are not modified. Retirement after a split still comes from the
recorded oracle in this teacher-forced training path.

Both models add all 21 JointState scalar outputs, finite/unset decisions for the
two ductilities, irreversible rebar rupture and five failure-mode categories.
`decode_joint_state` returns finite float64 values or explicit positive infinity
only for unset ductility. It rejects overflow instead of clipping. Damage/crush/
fatigue stay within [0,1] and cannot decrease; kappa/kappa_c/rebar work/dissipation
cannot decrease. Ductility itself is not assumed monotonic. Legacy endpoint heads
remain auxiliary outputs; the full state comes from the new decoder.

V11's first single-event fit exposed dead gradients through nonnegative history
clamps: damage and dissipation remained at the persistence baseline. V12 uses
ordinary normalized residual MSE before constrained decoding, so a negative raw
increment still receives a gradient toward a positive target. This changes both
the output contract and architecture ID, preserving V11 checkpoint semantics.
Bounds are applied to predictions only, never to recorded labels. The existing
legacy history/force loss remains auxiliary; V12 adds its full-history continuous
loss with `history_weight`, plus categorical losses weighted by `0.1*history_weight`.
These coefficients are experimental, not validated as optimal.

A matched 200-epoch/1,600-update fit on one corrected wall recording reduced
full-head damage encoded RMSE from the unchanged-state baseline **0.1773** to
**0.1000**. V11 stayed at 0.1773. Fracture F1 stayed **0.230** at threshold 0.5
for both: this is a successful history-gradient diagnostic, not a generalization
or fracture-accuracy improvement. A longer single-event fit completed at 2,000 epochs / 16,000 updates (264.5 s):
F1 **0.99838**, all 308 true breaks detected, one false positive and all detected
time bins correct at threshold 0.5. Damage encoded RMSE falls to 0.06916 and
dissipation to 0.002740 (persistence 0.17732 / 0.006302). This demonstrates fitting
capacity on that one training trajectory, not held-out or rollout accuracy.

`full_joint_history` evaluation reports per-field errors against persistence,
finite target counts, ductility-mask/rebar/mode confusion matrices, and the count
of retained cracked interfaces. Scores still reset to oracle state every interval.
No architecture here supplies a complete World/RtBond restart: elastic force,
stored energy, contact handovers, frame transfer and physical split application
remain unresolved. Do not describe these models as deployed state-carry solvers.


The full-head candidate's h48/p16 synthetic 10,400-node / 31,200-edge benchmark
measures 6.61 ms median MPS resident inference, 17.63 ms including GPU feature
construction, and 35.41 ms for CPU features plus transfer plus MPS. Eight timing
samples were taken while collection jobs ran. This is a compute graph, not 10,400
independently moving rigid bodies or a complete engine frame. Contacts, topology
application and rendering are excluded. CPU/GPU checks on the corrected eight-
interval wall trajectory (14,751 runtime bond intervals) found zero differences
in fracture class, ductility mask, rebar rupture or failure-mode decisions.

The corrected-wall comparison completed under
`runs/wall-history-comparison-v12-2026-10-19/`: V10 control versus V12 full-history,
each 24,768 optimizer updates, seed 0, with CDF weight 0.3. Validation selects the
checkpoint and threshold. The same 32 quality-reviewed recordings and exact
20/6/6 training/validation/test loading-recipe split are used in both arms. These
share one source wall geometry; this is not unseen-layout validation.

Test fracture F1 is **0.52525 (V10)** versus **0.51758 (V12)**. Full-history damage
encoded RMSE improves to **0.10000**, versus **0.18547** for unchanged history;
dissipation improves to **0.006168**, versus **0.006799**. Both models' local
position residuals remain worse than a zero-residual baseline. The history change
is useful for state coverage but is not a demonstrated fracture accuracy gain.

## Dynamic projectile inputs (V13)

`physics-projectile-history-native-v13` adds actual frame-start projectile spin,
full inertia, current crush consumption/remaining capacity, depth, driven flag
and constitutive material descriptors to V12. The versioned input contract is
`dynamic-projectile-observations/1`. It requires the rich world replay; missing
historical values fail validation instead of being fabricated. Terminal snapshots
never enter input construction. Negative remaining crush capacity is preserved.
Every projectile contributes, with no nearest-object truncation. Rigid-point
velocity at a chunk center is a kinematic feature, not a detected contact point.

An audit of 90 completed corrected recordings found 1,326 spinning and 825
partly-crushed projectile observations among 1,488 frame starts. The matched
24,768-update wall experiment completed in 381 s: validation F1 **0.53931**, test
F1 **0.51044** (epoch 10, threshold 0.70), compared with V12's **0.51758** test F1.
It remains experimental; adding those real inputs did not improve this held-out
score. Artifacts: `runs/projectile-v13-2026-10-19/`.

## Joint fracture likelihood experiment

`--fracture-objective factorized` preserves the existing weighted BCE plus
`0.15 * conditional timing CE`. Optional `--fracture-objective event-nll
--positive-weight 1` instead minimizes the negative log probability of the full
observed outcome: no break in this interval, or first break in one of K bins.
The probabilities are `P(0)=1-sigmoid(b)` and
`P(k)=sigmoid(b)*softmax(t)[k-1]`. The loss averages over all eligible interfaces,
so positive timing terms carry their actual prevalence within the interval.
No-positive and no-eligible intervals remain valid. The optional CDF score and
continuous-state objectives remain independent, explicitly weighted terms.

This removes class-weight probability distortion from that experimental branch;
it does not guarantee better classification or calibration after fitting. Losses
are still sampled by trajectory/frame, rather than globally by bond count. A
matched experiment under `runs/fracture-likelihood-v14-2026-10-19/` compares the
existing weight-3 V12 control, unweighted factorized loss, and unweighted joint
likelihood. The second comparison isolates timing normalization; all use the
same data, model, optimizer budget and validation-only selection. The completed
results are below; no fracture improvement is demonstrated.

| V12 objective | Test F1 | Brier score | CDF score | Timing accuracy on detected breaks |
|---|---:|---:|---:|---:|
| Weighted factorized control | 0.51758 | 0.02932 | 0.02793 | 0.91819 |
| Unweighted factorized | 0.51577 | 0.02319 | 0.02212 | 0.80420 |
| Unweighted joint event NLL | 0.50947 | 0.02349 | 0.02232 | 0.83772 |

Lower aggregate probability scores mostly reflect better intact-bond scores:
CDF error on actual breaks rises from 0.22435 to 0.43667 / 0.45394. Both positive
and negative cohorts matter. Timing scores condition on detected breaks and
therefore do not compare identical subsets. The original weighted objective
remains the control for the next state-consistency experiment.


## Fracture/state consistency (V15)

`python -m learned_impact.state_consistency --weights RUN --dataset DATA --out REPORT`
audits the complete decoded JointState against `JointState::connected`:
`damage < 1 || (has_rebar && !rebar_broken)`. It reads explicit source rebar
presence and the resolved feature flag, checks oracle flags against this rule,
and verifies first-fracture labels against end history. It never repairs outputs.
On the V12 wall holdout, **852** fracture-head positives retain connected state,
and **503** fracture-head negatives have disconnected state. The state-derived
fracture F1 is **0.37078**, versus **0.51758** for the classifier.

`physics-consistent-history-native-v15` (output contract
`learned-joint-state-head/3`) structures the decoder around this connectivity
rule. A predicted disconnection sets complete material damage and ruptures a
present reinforcing tie. A connected reinforced bond may be fully cracked only
while its tie remains intact. Already-complete damage and broken reinforcement
cannot heal. A separate material-crack classifier handles that reinforced,
connected state. Continuous damage regression is conditioned on partial targets;
complete damage is supervised categorically. Other history fields and legacy
auxiliary heads remain as before. The legacy packed state remains auxiliary;
use the full decoder, not that packed head, for this output contract.

The data loader derives explicit complete-damage start/target booleans from the
existing float64 sidecar before float32 log encoding, preserving the exact
boundary near damage=1. Only the start flag enters prediction. Material loading
adds explicit `bond_has_rebar`; the model requires both causal fields under
`explicit-rebar-complete-damage/1`. No new oracle recording or label change is
needed. The decoder preserves the incomplete boundary despite float32 rounding.

Tests cover independent head disagreements, unreinforced and reinforced states,
previous rupture, retained cracked interfaces, conditional gradients, target
poisoning, near-complete damage and checkpoint roundtrips. This guarantees
connectivity agreement for valid model inputs, not accurate fracture placement,
energy balance, contact handovers or complete engine state application.
The matched wall diagnostic completed under `runs/consistent-history-v15-2026-10-19/`.
V15 selected epoch 14 / threshold 0.60: validation F1 0.54081, test F1 **0.51349**
versus V12's **0.51758**. All **75,709** eligible test interfaces have matching
fracture and decoded-state connectivity (zero contradictions). Damage encoded
RMSE is 0.10523 versus V12's 0.10000. This resolves contradictory decisions but
does not demonstrate improved fracture accuracy. CPU/MPS with GPU feature
construction agree on every fracture class, complete-damage and rebar decision
across 48 held-out frames / 76,564 runtime bond intervals; maximum break-
probability difference is 8.35e-7. Numerical parity is not an engine rollout.
The completed experiment `runs/corrected-impact128-v15-2026-10-19/` compares V12 and
V15 on the entire corrected original 128-scenario cohort, 2,176 intervals,
96 epochs / 99,072 updates per arm, with fixed inherited partitions and
validation-only selection. Its houses still share a source geometry, so their
load holdouts do not establish novel-house generalization. V12/V15 test head F1
is 0.55730/0.53023, while decoded-state fracture F1 is 0.19722/0.53023. V15 has
zero contradictions over 468,534 eligible interfaces, compared with V12's 3,830
head-break/state-connected and 1,943 reverse contradictions. V15's motion and
topology errors are larger; full details are in `../COMPARISON.md`.

A separate endpoint force audit of V12 (`wall-history-comparison-v12-2026-10-19/
force-audit.json`) reports force RMSE 2,826.6 N versus 2,843.9 N for zero and
2,863.4 N for persistence. For the 73,656 interfaces still alive at interval end,
the values are 450.8 / 452.0 / 592.2 N. Moment RMSE there is 16.66 N m versus
15.64 for zero. Retired interfaces preserve their last cached force in the
recorder and dominate the all-bond error; do not interpret those values as
forces continuing after retirement. This audit identifies a force-prediction
weakness and a target-semantics issue to separate in future experiments.

The V15 10,400-node / 31,200-edge synthetic compute benchmark measured 6.95 ms
median MPS resident inference, 17.96 ms with GPU feature construction and
34.46 ms with CPU features plus transfer (eight samples, one CPU thread,
collection/training active). CPU resident inference was 90.23 ms. This does not
measure 10,400 independently active rigid bodies or complete engine frames;
contacts, topology, state application and rendering remain outside the timing.


## Lossless cached bond responses

The recorder now writes `learned-bond-response/1`: total force, elastic force,
stored energy and all five stress measures, with alive/connected flags at every
frame boundary. See `../SCHEMA.md` for field order and units. This closes a
specific training-target gap: contact handovers use elastic force and stored
energy, which cannot be inferred from total force or the old JointState pack.

`load_event(..., bond_response_history=True)` and `event_batches` expose separate
causal start features and future targets. All advertised sidecars are validated
even when not used by a model; old architectures retain their old inputs. Data
fingerprints include response metadata and bytes. No current V12/V15 checkpoint
has been trained on the additional fields yet.

`learned_impact_enrich` writes a fresh event and now verifies existing joint and
response sidecars byte for byte, alongside original arrays, replay fields and
context inputs. The frozen experiment under `runs/response-v16-2026-10-19/` is
completed checks on a fractured wall and a 120-frame house trajectory. All 137 /
1,817 original arrays, 19 / 243 existing joint-history files, replay fields and
context inputs are unchanged. Python additionally verified force/topology
alignment and that changing an added elastic target changes its fingerprint.
Total and elastic force differ in 15,680 / 18,162 wall response rows and
151,320 / 157,179 house response rows: their distinction is material here.
The full original128 cohort is now being enriched by two frozen workers, with
per-event proofs and no combined manifest until all succeed. This is data-contract
version work; there is no V16 network yet.
The currently running corpus/training jobs retain their original frozen code.

## Complete cached-response head (V18)

`physics-response-history-native-v18` extends V15 with a zero-initialized causal
18-field response encoder and a residual endpoint head. Total force, elastic
force, stored energy and all cached stress measures are predicted together.
The six total-force outputs are also the legacy force outputs; there are no
conflicting duplicate force predictions. Initial response prediction is
persistence. Stored energy, shear, normal compression and compressive force are
nonnegative; tension and extreme-fibre compression retain their signed semantics.
The engine's stress measures come from effective undamaged force, so they cannot
simply be reconstructed from the predicted damaged elastic force.

Training fits separate residual scales for active-end interfaces and retired
last caches using only the training partition. The response loss averages the
present cohorts after normalization, so the small population of large cached
fracture forces cannot dominate solely through its physical units. Supervision
uses the raw residual before nonnegative constraints to retain gradients.
`--response-weight` defaults to 0.05 for this architecture; older models do not
consume the extra fields or objective. This is a combined input/output/loss
experiment, not an isolated ablation of one choice.

The matched 32-wall experiment is under `runs/response-model-v18-2026-10-19/`.
All 32 enriched recordings preserve every old array, replay field and joint
history file. The preflight checks all 256 frames for target causality, a single
consistent total-force output, and finite real-data gradients. Training retains
the V15 reference split, seed, 24,768 updates and validation selection. It completed
in 388.1 s, selecting epoch 10 / threshold 0.65. Wall test fracture F1 is 0.52072
versus V15's 0.51349, split-pair F1 is 0.47119 versus 0.45748, and damage encoded
RMSE is 0.10176 versus 0.10523. One source wall geometry and these small changes
do not establish a generalization improvement.

Exact f64 response auditing exposes the larger remaining problem. For 73,656
active-end bond intervals, total-force component RMSE is **453.99 N**, versus
450.63 N for V15, 451.98 N for zero and 592.19 N for persistence. Stored-energy
RMSE is **0.01200 J**, versus 0.008887 J for zero and 0.01197 J for persistence.
Retired cached energy RMSE is 2.53794 J versus 2.53883 J for zero: negligible
improvement. The richer head closes an output-contract gap but is not yet a
reliable stress/energy predictor. The 2,000-epoch / 16,000-update single-trajectory
diagnostic completed in 269.8 s: training fracture F1 0.99838 (308 TP, 1 FP, 0 FN),
but active force RMSE remains 303.23 N versus 348.08 N for zero and 431.54 N for
persistence. Active stored-energy RMSE is 0.01664 J versus 0.01772 J for zero;
retired-cache energy is 1.51943 J versus 1.54328 J for zero. Even this fitting
diagnostic leaves substantial response error. It proves neither held-out accuracy
nor that more recordings alone will resolve the response objective.

CPU/Metal with GPU feature construction agree on all fracture, complete-damage
and rebar decisions over 48 test frames / 76,564 bond intervals. Maximum response
encoding difference is 1.19e-6. A synthetic 10,400-node / 31,200-edge benchmark
measured 7.19 ms resident Metal inference, 18.93 ms including GPU features and
35.85 ms with CPU features plus transfer (eight samples, other work active).
This does not establish real-time complete frames with 10k active rigid bodies.

Metrics separate active-end response errors from retired cached values, with
zero and persistence baselines. `learned_impact.response_audit` scores exact f64
source targets in physical units; ordinary training metrics decode float32
signed-log targets and state that qualification. These predicted fields are not
yet applied to a World, and independent response predictions do not enforce
constitutive force/stress/energy consistency or prove stable coupled dynamics.

The normalization audit found active stored-energy residual RMS **1.03735e-5**
in signed-log units, below the fixed 1e-4 scaling floor. Its squared errors thus
receive about **93 times less weight** than normalization by measured variation
would assign. `--response-scale-mode nonzero-rms` is an optional, controlled
alternative: preserve every positive measured RMS, use unit scale for exactly
constant fields, and reject scales unrepresentable in float32. Existing default
behavior and checkpoint buffers are preserved. A regression test verifies that
a small but nonzero energy target retains a full normalized learning signal.
The matched experiment `runs/response-scale-v19-2026-10-19/` completed. Active
stored-energy RMSE improved 0.012000 → 0.011273 J, still worse than the zero
baseline (0.008887 J). Force RMSE is 453.715 N versus zero 451.978 N. Bond F1
rose 0.52072 → 0.52596, but false breaks rose 1,060 → 1,372 and split-frame
fragment-count MAE worsened 18.69 → 33.54. This mixed result is not promoted;
the existing default stays unchanged. See the exact-f64 audit and comparison.

## Fragment-aware evaluation and optional training loss

`learned_impact.topology_audit` checks the actual connected-component partition
implied by each oracle-start prediction, including fragment mass error, false
merges/splits, separated-pair F1 and a no-new-break baseline. Oracle endpoint
clusters and retired interfaces must agree exactly with the recorded bond
connectivity before a frame is scored. No independent rollout is implied.
See `../COMPARISON.md` for measured wall and corrected128 results.

`train_physics --fragment-cut-weight 0.05` adds an experimental weakest-boundary
term for runtime-alive history architectures. Future endpoint connectivity and
first-fracture labels define target groups only; loading verifies their exact
agreement. The target is added after input preparation and frame deduplication.
The maximum break NLL within each oracle fragment boundary focuses on a missed
bridge; ordinary bond supervision still covers redundant cracks and false
positive breaks. Weight zero is the default and preserves existing training.
This does not optimize full partition likelihood or energy consistency.

Unit tests distinguish redundant cracks from separating cuts, verify sparse
pair counts against an independent quadratic definition on small graphs, and
cover mass weighting, large unsigned cluster IDs, more than 10k small parent
bodies, cut-loss gradients, ties and extreme logits. A 256-frame real-data
preflight verifies identical split/sampling inputs and bitwise-identical model
predictions with and without the new loss targets. The matched wall experiment
completed under `runs/fragment-cut-v17-2026-10-19/`: test bond F1 improved from
0.51349 to 0.52757 and split-pair F1 from 0.45748 to 0.48052, but false-positive
breaks rose from 828 to 1,044 and split-frame fragment-count MAE from 14.15 to
19.92. Paired load-recipe uncertainty includes zero for the separation/mass
improvements. This mixed result is not promoted; weight zero stays the default.


## Read-only native snapshot bridge

`stress-learned::runtime_snapshot::RuntimeGraph` binds stable graph slots and an
initial displacement reference. Its `capture(&World)` shares the recorder's
local-state extraction and replay capture, preserving full current joint and
cached-response state. JSON null means positive-infinite unset ductility only
in fields 4/5; invalid nonfinite values are rejected. New active graph slots,
changed source scenes, and snapshots between frame boundaries fail explicitly.

`learned_impact.runtime_snapshot.NativeGraph` turns **one current snapshot** into
V18 inputs. It needs neither future frames nor fabricated endpoint labels or
contact loads. Cracked live interfaces stay in the graph; retired caches remain
in the source snapshot but are filtered out of model participation. World and
prescribed-load observation functions are shared with the training loader.

To verify a scene against a trained V18 checkpoint, from the oracle directory:

```sh
cargo run --release -p stress-learned --bin learned_impact_snapshot -- \
  --scene /path/to/scene.json --out /path/to/new-parity-directory --frames 8
PYTHONPATH=learned-impact/python learned-impact/python/.venv/bin/python \
  -m learned_impact.runtime_parity --root /path/to/new-parity-directory \
  --weights /path/to/checkpoint --out /path/to/new-report.json
```

The Rust harness captures inputs before stepping each oracle interval and writes
an independent recorder stream. The audit checks every supplied tensor, every
prediction, and exact f64 joint/response snapshots. The wall19, offset-pulse and 120-frame house integrations passed 7,616 input
and 2,584 output comparisons over 136 intervals, including 2,702 cracked-live
and 34,232 retired bond observations. The 1,683 non-QS arrays and all 139 replay
snapshots match the original corrected recordings exactly; the diagnostic
harness deliberately omits the unused scratch QS probe. Results: `runs/runtime-start-v20-2026-10-19/`.
“V20” labels this integration work, not a new architecture or trained model.

This diagnostic bridge serializes JSON and does **not** apply predictions,
replace contact integration, export Slang, or establish runtime speed. Full scene
checks and diagnostic energy capture are deliberate validation costs here, not
a proposed production hot path. The coupled application phase remains open.


## Preserve physical validation candidates

V18 training with `--retain-physical-checkpoints` additionally retains the lowest
validation active-force component RMSE, stored-energy RMSE and world-position
component RMSE in separate `physical-candidates/` checkpoint directories.
Each retains its own validation-calibrated fracture threshold, split, epoch,
metrics and weight hash. Ties retain the earliest checkpoint. The main model
still uses the existing fracture selection; no new composite score or preferred
candidate is imposed. This option requires validation and rejects `--all-data`.
Candidate reports contain validation metrics only; they are not automatically
scored on the test set. Decode-to-SI metrics use float32 training labels; run
`response_audit` separately for exact f64 errors after declaring a candidate.

The 32-wall, one-epoch/two-update integration smoke saved and reloaded all three
candidate checkpoints. This checks the pipeline, not training quality. Unit
tests ensure a force improvement survives even when fracture F1 worsens, other
criteria keep their original weights, and invalid validation data fails before
saving. Source: `checkpoint_candidates.py`.
