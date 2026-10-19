# Recorded oracle versus learned model

## Full-128 response result and physical-unit loss follow-up

The completed full-128 V18 run uses the same split and 99,072 optimizer updates
as the consistent V15 reference. Test fracture F1 improves 0.53023 → 0.55258,
position RMSE 4.198 → 2.934 mm, fragment-count MAE 8.129 → 6.543, and exact
partitions on splitting frames 13/116 → 18/116. Separated-pair F1 worsens
0.54037 → 0.47170 and largest-fragment mass error is essentially unchanged.
Head/state contradictions remain zero over 468,534 eligible bond intervals.
The older inconsistent V12 still scores 0.55730 fracture F1 and 2.644 mm position
RMSE; this is not an across-the-board accuracy improvement.

Active-end exact-f64 force RMSE is 1,099.760 N against 1,174.564 N persistence
and 1,531.838 N zero. Stored-energy RMSE is 0.119787 J against 0.116075 J
persistence and 0.127701 J zero. Retired response caches are reported separately.
Artifacts: `runs/response-impact128-v21-2026-10-19/{experiment-plan,exact-response,topology,state-consistency}.json`.
The additional `impact-responses` viewer dataset exposes all 128 scenarios,
including house/ram replays, using this checkpoint. Existing datasets remain.

V23 changes only V22's response objective to normalized physical-unit residual
MSE, fitted on training data separately for active and retired responses.
Initial predictions match V22 exactly; a signed-log inverse uses a nonzero
correct derivative at zero. No numerical clipping is introduced. In the matched
32-wall experiment, test F1 improves 0.49908 → 0.52058, but active force RMSE
worsens 451.825 → 455.869 N (zero:451.978). Energy improves 0.011766 → 0.011405 J
but remains worse than zero (0.008887 J). Fragment-separated-pair F1 improves
0.37637 → 0.46909 while count MAE worsens 11.923 → 17.846. V23 remains experimental.
Artifacts: `runs/physical-response-v23-2026-10-19/`. A preflight harness contract
failure before any optimizer update is preserved as `failed-preflight-1.*`.

These are repeatedly inspected diagnostic holdouts and oracle-reset intervals,
not fresh generalization tests or autonomous engine rollouts. The next matched
full-128 V22 run isolates dynamic projectile inputs before adding the spin data.

## Data-quality qualification

The world-motion audit found an oracle energy/speed explosion in validation
event `v4_layout_09_c3_s101`, dominating its motion RMSE. No data or metric has
been removed. Existing fracture selection includes that event and remains
provisional pending source repair and regenerated validation. Exact replay
agreement is not proof of physically sound ground truth. See [DIVERSITY.md](DIVERSITY.md)
and `runs/world-quality-v10-2026-10-19/{audit,energy-followup}.json`.

The source failure is now reproduced by a self-contained contact-geometry test
and repaired in polytope mass integration. The same saved failing scene now stays
below its initial mechanical energy and reaches 25.58 m/s maximum speed instead
of 48.7 million m/s. Two comparison scenes also remain bounded. A fresh 290-event
corpus has completed with the corrected oracle and a cohort-specific quality
review; the 18 extreme continued-force cases remain explicitly identified. The corrected-wall
diagnostic below is explicitly separate from the older original-physics results.

## Corrected-wall diagnostic: history and projectile inputs

All three arms use the same 32 physically reviewed corrected recordings,
20/6/6 train/validation/test loading recipes, seed 0, 24,768 updates, h48/p16,
and CDF weight 0.3. Checkpoint and threshold selection use validation only.
These loading recipes share one geometry: this is not unseen-layout evaluation.

| Model | Selected epoch | Threshold | Validation F1 | Test F1 |
|---|---:|---:|---:|---:|
| V10 constitutive control | 14 | 0.60 | 0.54069 | 0.52525 |
| V12 full-history residual | 10 | 0.70 | 0.54138 | 0.51758 |
| V13 dynamic projectile inputs | 10 | 0.70 | 0.53931 | 0.51044 |

Neither new input/output branch improves fracture F1 here. V12 full-head damage
encoded RMSE is 0.10000 versus 0.18547 for persistence, and dissipation 0.006168
versus 0.006799. History prediction improves over unchanged state, while local
displacement residuals remain worse than predicting zero. Do not promote any of
these as a replacement-ready solver. Full engine state application, fracture and
history consistency, autonomous rollouts and 10k-active-body timings remain open.

Artifacts: `runs/wall-history-comparison-v12-2026-10-19/comparison.json` and
`runs/projectile-v13-2026-10-19/comparison.json`. The V12 single-trajectory F1
0.99838 is an overfit diagnostic, not a conflicting generalization result.

The controlled loss experiment completed under
`runs/fracture-likelihood-v14-2026-10-19/`: existing weighted factorized V12,
unweighted factorized V12, and a normalized joint no-break/time-bin likelihood.
It isolates positive weighting from timing normalization with the same data and
budget. Test F1 was 0.51758 / 0.51577 / 0.50947, respectively. Unweighted
objectives improved aggregate probability scores but worsened CDF score on actual
breaks and timing accuracy among detected breaks. They are optional experiments,
not promoted improvements. See `python/README.md` for the full table.

A full-state consistency audit also found 852 V12 fracture positives with
connected decoded state and 503 reverse contradictions. The versioned V15
structured decoder produced zero contradictions on the wall test cohort, with
F1 0.51349 versus V12's 0.51758. CPU/Metal agree on every discrete fracture and
history-boundary decision across 48 test frames. The complete corrected
original128 comparison has also finished (table below). This consistency fix
does not provide a complete World state adapter.

## Corrected original128 comparison: completed

Both arms used all 128 corrected scenarios / 2,176 intervals, fixed inherited
partitions, seed 0, h48/p16, 99,072 identical sampled updates and validation-only
checkpoint/threshold selection. V12 selected epoch 92 / threshold 0.60; V15
selected epoch 24 / threshold 0.65. The house cases still share one geometry.

| Test metric | V12 | V15 |
|---|---:|---:|
| Fracture-head F1 | 0.55730 | 0.53023 |
| Fracture recall | 0.49204 | 0.48610 |
| Decoded-joint-state fracture F1 | 0.19722 | 0.53023 |
| Head-break / state-connected contradictions | 3,830 | 0 |
| Head-intact / state-disconnected contradictions | 1,943 | 0 |
| Damage encoded RMSE | 0.06065 | 0.06355 |
| World endpoint position RMSE (m) | 0.002644 | 0.004198 |
| Separated-pair F1 on 116 splitting intervals | 0.55930 | 0.54037 |
| Largest-fragment mass-fraction MAE on splitting intervals | 0.06262 | 0.07169 |
| Fragment-count MAE on splitting intervals | 5.87 | 8.13 |
| Exact partitions on splitting intervals | 21/116 | 13/116 |

V15 resolves a substantial contradiction between V12's fracture head and its
decoded state, but has worse fracture-head, motion and topology accuracy in this
comparison. V12's higher headline F1 therefore cannot be treated as an engine-ready
state update. Neither is replacement-ready. The experiment completed in 2,739 s
including checks; artifacts and exact sampling/source proofs are under
`runs/corrected-impact128-v15-2026-10-19/`.

## Cached-response experiment (V18): complete, not promoted

The recorder's new elastic-force, stored-energy and stress-measure targets are
now used by a versioned model with 18 causal response inputs and 18 endpoint
outputs. Loss scaling and metrics distinguish active-end interfaces from retired
last caches. A matched 32-wall run retains V15's split, seed and 24,768 updates.
Fracture F1 is 0.52072 versus 0.51349, but active-end force RMSE is 453.99 N versus
450.63 N for V15 and 451.98 N for zero. Stored-energy RMSE is 0.01200 J versus
0.008887 J for zero. These fields are now represented, but not predicted well
enough to support a solver replacement. See `python/README.md` and
`runs/response-model-v18-2026-10-19/{comparison,exact-response-audit}.json`.

The normalization audit found a numerical floor underweighting small nonzero
energy changes by about 93 times. A matched optional normalization experiment is
running under `runs/response-scale-v19-2026-10-19/`; it preserves measured positive
RMS values instead of applying that floor. No physical targets are clipped and
the original behavior remains the default pending evidence.

## Fragment topology audit

`python -m learned_impact.topology_audit --weights MODEL --dataset DATASET
--out REPORT.json` evaluates the fragments formed by predicted breaks from
each recorded frame-start graph. It first verifies that the oracle graph's
connected components reproduce the saved cluster IDs and bond retirements.
Pair counts exclude chunks already in different frame-start clusters; sparse
contingency counts avoid an all-pairs matrix. Parent membership is grouped once,
including the case of 10,400 or more already separate bodies.

On the same held-out wall cohort, only 13 of 48 intervals produce new oracle
splits. Reporting all-frame exact agreement would overemphasize quiet frames:
doing nothing already matches 35/48. The split intervals expose the main error:

| Metric on 13 splitting intervals | V12 | V15 | No new breaks |
|---|---:|---:|---:|
| Separated chunk-pair F1 | 0.4446 | 0.4575 | 0 |
| Largest-fragment mass-fraction MAE | 0.2964 | 0.2946 | 0.3631 |
| Fragment-count MAE | 13.08 | 14.15 | 28.00 |
| Exact fragment partitions | 0 | 0 | 0 |

The remaining errors predominantly leave oracle fragments joined. Better bond
F1 alone is not a sufficient replacement criterion. Mass-fraction errors here
are mass-weighted within each frame and then averaged over splitting frames.
These are oracle-start topology forecasts, not accumulated fragmentation or
predicted motion. Reports are `topology-audit.json` in the V12/V15 wall runs.

The completed V12 arm of the broader corrected128 experiment has held-out bond
F1 **0.55730**, precision 0.64254 and recall 0.49204. Its 440 test intervals include
116 new-split intervals: separated-pair F1 is 0.55930, largest-fragment fraction
MAE 0.06262 versus 0.10512 for no new breaks, and 21/116 exact partitions. Different
cohorts prevent comparing these numbers directly with the wall-only table.
The matching V15 results are in the completed comparison above. See
`history-control-topology.json` and `consistent-history-topology.json` under
`runs/corrected-impact128-v15-2026-10-19/`.

The optional weakest-boundary loss experiment completed under
`runs/fragment-cut-v17-2026-10-19/`. It groups eligible edges by the pair of oracle
endpoint fragments and averages the worst break negative-log-probability per
boundary. This targets an incomplete cut while preserving the existing bond,
timing and state losses. It is a surrogate, not an exact partition likelihood.
The weight is predeclared as 0.05, compared with V15 weight zero on identical
data, initialization, 24,768 updates and validation selection. No default changes
or accuracy promotion follow from the motivation alone. These existing holdouts
are an iterative diagnostic, not a fresh untouched evaluation set.

The candidate selected epoch 10 / threshold 0.70 and completed in 384.8 seconds,
with identical sampling counts and source fingerprints and zero state/fracture
contradictions. Its results are mixed:

| Wall test metric | V15 baseline | Cut weight 0.05 |
|---|---:|---:|
| Bond fracture F1 | 0.51349 | 0.52757 |
| Precision / recall | 0.61667 / 0.43989 | 0.58290 / 0.48184 |
| False-positive breaks | 828 | 1,044 |
| Separated-pair F1, splitting intervals | 0.45748 | 0.48052 |
| Largest-fragment fraction MAE, splitting intervals | 0.29460 | 0.28828 |
| Fragment-count MAE, splitting intervals | 14.15 | 19.92 |
| Damage encoded RMSE | 0.10523 | 0.10093 |

The boundary objective recovers more true breaks but also over-fragments.
Paired resampling of the six loading recipes gives 95% difference intervals
−0.01858…+0.05253 for separated-pair F1 and −0.01518…+0.00284 for largest-fragment
fraction MAE; both include zero. These intervals cover load recipes on one
geometry, not geometry generalization. The option remains experimental and off
by default. Reports: `comparison.json` and `paired-topology-uncertainty.json`.

## Prescribed-forcing augmentation: matched updates, completed

Both native-V7 models use identical training source snapshots, seed 0, 24 epochs
and 24,768 optimizer updates. The control uses the original 128 scenarios; the
augmented model adds 144 forcing scenarios (272 total / 3,328 intervals).
Each selects its checkpoint and threshold on its predeclared validation set.

| Common held-out cohort | Original-data control | With forcing data |
|---|---:|---:|
| All test fracture F1 | 0.5188 | 0.5353 |
| Original impact test F1 | 0.5343 | 0.5301 |
| New pressure test F1 | 0.3767 | 0.5830 |

The aggregate paired 15-group difference interval is −0.0164…+0.0445; the
original-impact interval is −0.0186…+0.0118. Both include zero. Pressure improves
on the one held-out wall layout, which is insufficient for a geometry-level
uncertainty estimate (its one-group bootstrap is degenerate). Both models make
zero false-positive predictions on the non-fracturing point-force cohort; its
fracture F1/recall cannot be assessed. Stronger positive point-force pilot cases
were generated separately and are absent from this comparison.

Artifacts: `runs/forcing-v8-2026-10-19-repaired/{training-plan,comparison}.json`.
These remain oracle-start interval predictions, not autonomous fracture rollouts.

## Larger-network compute screen (untrained)

Native-V7 shapes with hidden widths 48, 96 and 128 were timed on actual Metal
with 10,400 synthetic nodes / 31,200 edges, 16 attention tokens, one projectile
and two prescribed loads. Each uses 12 synchronized measurements after warmup.
These are untrained scratch checkpoints, excluded from the visualizer and all
accuracy claims. CPU training/generation was active during measurement.

| Hidden width | Resident inference | GPU features + inference | CPU features + transfer + inference |
|---|---:|---:|---:|
| 48 | 5.97 ms | 16.35 ms | 28.60 ms |
| 96 | 10.88 ms | 20.74 ms | 32.99 ms |
| 128 | 13.85 ms | 25.20 ms | 41.03 ms |

Larger networks have a measurable runtime cost before engine contacts, topology,
integration and rendering. They may still be useful accuracy experiments, but
none proves 60 FPS for 10k independently moving bodies. Feature construction
remains a substantial part of the current Python/MPS path. Artifacts:
`runs/capacity-screen-v10-2026-10-19/`.

## Auxiliary-gradient projection: completed, not promoted

A training-only diagnostic at the completed native-V7 checkpoint measured the
shared-trunk gradients of every weighted loss term. It sampled the quietest and
most-breaking interval from each of 51 training physical groups, deliberately
probing extremes rather than estimating a representative mean. Auxiliary
gradients had a median norm 11.39 times fracture BCE on quiet samples; the total
gradient opposed fracture BCE in 8/51 of them. This motivated a controlled test,
not a conclusion that auxiliary losses should be removed.

Two continuations start from the exact same checkpoint with unchanged data/loss
weights and saved target scales. Both run 4 epochs / 4,128 identical sampled
updates, one CPU thread, deterministic algorithms, fresh AdamW, and cosine
learning rate 3e-5 → 3e-6. The experimental update projects only the shared
auxiliary component opposing fracture BCE; private head gradients remain intact.
It applies on 1,633 updates. Adam preconditioning means this is not a guarantee
about the actual parameter-step direction.

| Metric | Ordinary continuation | Projected auxiliary gradient |
|---|---:|---:|
| Selected epoch | 4 | 2 |
| Validation F1 | 0.5412 | 0.5425 |
| Selected threshold | 0.55 | 0.60 |
| Test F1 | 0.5352 | 0.5171 |
| Test precision | 0.5648 | 0.5869 |
| Test recall | 0.5086 | 0.4622 |
| World-position RMSE | 4.534 mm | 4.540 mm |
| World-velocity RMSE | 0.3264 m/s | 0.3268 m/s |

Projection improves precision but loses recall, with essentially unchanged motion
error. The paired 15-group F1-difference interval is −0.0302…−0.0088. Damage MAE
on intervals where damage/crush changes also worsens (0.3016 → 0.3188). It is not
adopted by the default trainer or visualizer. The slightly
higher validation F1 did not carry to this held-out set. No inference performance
or closed-loop improvement is implied. Artifacts:
`runs/native-forcing-v7-2026-10-19/loss-gradient-audit.json` and
`runs/loss-projection-v9-2026-10-19/{training-plan,comparison}.json`.

## Exact joint-history recovery and reproducible saved scenes

New recordings preserve all joint-history float64 values, unset ductility masks,
failure mode and alive/connected flags in a versioned sidecar. The existing
22-component float32 pack is retained for checkpoint compatibility. V5–V7
models do not yet consume the new 31-feature encoding; this is a data-contract
repair, not an accuracy result. See [SCHEMA.md](SCHEMA.md).

The saved-scene check exposed a JSON parser issue: tiny float64 changes on reload
could amplify into different impact trajectories. Enabling exact float
round-tripping fixed this. Re-recorded impact and point-force smoke scenarios now
match all 137 original arrays each (1,692,288 and 5,199,304 bytes), and every
original replay frame field exactly. Original context inputs are retained,
including historical rounding of context centers. The added histories contain
4,816 and 28,908 unset ductility values and preserve 3,608 nonzero mode entries
in the impact clip. These were not representable by the legacy state pack.

Verified enrichment of the original 128 scenarios / 2,176 intervals completed
under `runs/joint-history-v9-2026-10-19/` in 3,013 seconds, with no failed events.
Its copied binary is fingerprinted; original recordings are unchanged. The complete
dataset manifest was published only after all events passed exact array/replay
comparison and strict Python sidecar loading. Failed diagnostic outputs are excluded.
This retains the original oracle's physics, including its defective validation
trajectory; it does not supply a full World restart adapter or validate a learned rollout.

Validation: 71 Python tests; 63 stress-ref library tests; 5 stress-learned library,
2 enrichment binary, and 7 oracle recording tests pass. The broader stress-ref
check also exposed an unsigned subtraction in a neighbour-list test's synthetic
motion setup; guarding the unmatched early return fixes the test without changing
runtime physics.

Training now accepts an explicit update count per epoch with physical-group
sampling. Persistent shuffled queues preserve eventual interval coverage, so
dataset-size comparisons can hold optimizer updates fixed. Default sampling and
existing checkpoints are unchanged.

## Matched preview versus native forcing inputs: completed

Both V7 fits completed with identical source snapshots, 128 scenarios / 2,176
intervals, seed 0, physical-group sampling, world-motion weight 0.1 and 24 epochs
/ 24,768 updates. The preview candidate selects epoch 24; the native candidate
selects epoch 23. Selection and thresholds use validation only.

| Metric | V7 with causal preview | V7 without contact-preview inputs |
|---|---:|---:|
| Validation fracture F1 | 0.543 | 0.536 |
| Test fracture F1 | 0.507 | 0.534 |
| Original test scenarios F1 | 0.484 | 0.506 |
| New-layout test F1 | 0.568 | 0.608 |
| Test world-position RMSE | 2.958 mm | 4.595 mm |
| Test world-velocity RMSE | 0.242 m/s | 0.330 m/s |
| Physical 10,400-chunk wall F1 | 0.512 | 0.506 |
| Large-wall world-position RMSE | 6.585 mm | 5.413 mm |

The 15-group paired test-F1 difference interval is −0.0055…+0.0591, including no
improvement. Removing preview helps the wall cohorts but hurts house fracture F1
(0.521 → 0.475) and aggregate motion. It is a viable experimental direction, not
evidence of equivalent accuracy. On the large wall the native model catches 817,
misses 1,104 and predicts 490 extra breaks; preview catches 987, misses 934 and
predicts 946 extra breaks. Native world-position error on that case is below the
constant-velocity baseline (8.520 mm), but this is only one oracle-start interval.

CPU and Metal agree on every held-out break/timing decision (470,072 bonds over
440 intervals); maximum probability difference is 8.35e-7. The trained native
checkpoint measures 5.46 ms resident inference, 14.35 ms including GPU features,
and 21.25 ms with CPU features/transfer on the synthetic 10k-node / 30k-edge graph.
One projectile and two prescribed loads are encoded. These timings exclude engine
contacts, integration, topology changes and rendering. No contact solver is
replaced: this experiment removes the separate contact-preview input requirement.

Artifacts: `runs/native-forcing-v7-2026-10-19/{comparison,preview-scale,native-scale,native-mps-parity,native-benchmark,training-source-verification}.json`.
Both completed checkpoints are added to the longer-house, new-layout and scale
visualizer datasets. The separate 96-epoch native-input fit has also completed on
the same data/architecture, with 99,072 updates; forcing-v8 data was excluded.
Validation selected epoch 48 and threshold 0.50. This is a longer optimization
experiment, not a matched-compute improvement claim.

| Metric | Native 24 epochs | Native 96 epochs |
|---|---:|---:|
| Test fracture F1 | 0.5343 | 0.5450 |
| House test F1 | 0.4753 | 0.5033 |
| New-layout test F1 | 0.6084 | 0.6351 |
| Test world-position RMSE | 4.595 mm | 3.324 mm |
| Test world-velocity RMSE | 0.330 m/s | 0.260 m/s |
| Physical 10,400-chunk wall F1 | 0.5062 | 0.4926 |

The 15-group paired F1 difference interval is −0.0081…+0.0293, including zero.
Large-wall recall declines: the longer fit catches 750, misses 1,171 and predicts
374 extra breaks. Local-displacement RMSE remains worse than a zero-increment
baseline (21.37 versus 20.11 micrometres). CPU/Metal agree on all 470,072 held-out
break and timing decisions; maximum probability difference is 1.13e-6.
The completed checkpoint is available as `native-long` in the visualizer, with
all previous models retained. It remains an oracle-start interval predictor,
without a validated autonomous state rollout.

Artifacts: `runs/native-forcing-v7-2026-10-19/{long-comparison,native-long-scale,native-long-mps-parity}.json`.

## Fracture timing versus location diagnostic

`temporal_metrics.py` accumulates the first predicted and true failure per bond
over each recorded clip, alongside the existing interval-level scores. This is
still prediction on oracle states: a clip union is **not** a learned rollout,
and late predictions are censored because oracle-failed bonds disappear from
subsequent inputs. Bin midpoints estimate times; they are not exact event times.

For the expanded-data V5 checkpoint on its 21 held-out clips, interval F1 is
0.583 and clip-union F1 is 0.624 (3,464 caught, 2,893 missed, 1,281 extra distinct
bonds). Of the caught bonds, 166 first predictions are over one interval early;
95.2% are within one interval. Ignoring timing therefore recovers only part of the
error. Fracture-location recall remains weak, so replacing the objective with only
a wider timing tolerance would conceal errors rather than solve them. The
diagnostic does not choose thresholds or checkpoints.

Artifact: `runs/layout-v4-2026-10-19/combined-temporal.json`. Two regression tests
cover early versus wrong-location predictions, repeat predictions, scene-scoped
bond IDs, quiet clips and invalid/duplicate labels.

## Expanded layout training and forcing inputs, 2026-10-19

The combined V5 fit completed: 128 scenarios, 2,176 intervals, 96 epochs and
99,072 updates. Validation selected epoch 75 and threshold 0.70. The preceding
80-scenario model keeps its own saved threshold (0.55) for comparison; neither
threshold is calibrated on test data. Existing memberships are preserved and all
impacts on one new layout stay in the same partition.

| Held-out cohort | Previous V5 F1 | Expanded-data V5 F1 |
|---|---:|---:|
| Original scenarios | 0.539 | 0.558 |
| New wall layouts | 0.555 | 0.653 |
| Combined | 0.543 | 0.583 |
| Separate 10,400-chunk wall, one interval | 0.187 | 0.479 |

On the large wall, the expanded-data model catches 669 of 1,921 breaks, misses
1,252 and predicts 201 extra breaks (precision 0.769, recall 0.348). This is an
improvement, but still unsuitable as an oracle replacement. No local-motion error
is reported there because every local coordinate target crosses a fragment reset.
The combined test has precision 0.653 and recall 0.526. Its paired group bootstrap
F1-difference interval is +0.0082…+0.0657; the original-cohort interval still
includes zero (−0.0034…+0.0374). Only **two** new layouts are held out for test.
One seed, repeated test inspection, more training data **and** more optimizer
updates do not isolate the cause or establish broad generalization.

The combined held-out local-velocity RMSE improves from 0.01690 to 0.01598 m/s
(zero-increment baseline 0.01850). Changed damage/crush cases have damage MAE
0.226 versus 0.266 previously. Local displacement remains a weakness: 29.64 μm
RMSE versus 20.11 μm for zero increment, despite improving from 30.85 μm. These
241,217 valid local chunk intervals exclude coordinate resets; fracture gains
must not be described as accurate motion integration.

CPU and Metal with GPU feature construction agree on all 470,072 held-out bond
break/timing decisions over 440 intervals; maximum probability difference is
1.44e-6. All 48 new native replays have verified array fingerprints and frame
alignment. The visualizer adds a **New wall layouts · 48 impacts** dataset and
the completed checkpoint in the longer-house and scale datasets; existing views
remain available. Geometry is the recorded oracle trajectory for this V5 model.

Artifacts: `runs/layout-v4-2026-10-19/combined-{comparison,scale-evaluation,mps-parity,viewer-verification}.json`,
`combined-world-v5/`, training logs, fixed split and audit. The comparison reports
motion/history errors separately and preserves both models' cohort/group counts.

The next matched experiment adds the `prescribed-forcing-descriptors/1` input
contract. It encodes point-force application offsets, pressure selection masks and
normals, blast location/mass/time, and all five supported time curves, retaining
every table knot. Descriptors use only the saved scene and frame-start time;
they do not evaluate or copy future contact/pressure forces. Unsupported structural
removal events are rejected explicitly. All 128 scenarios / 2,176 intervals pass
the input audit (36 pressure and 16 blast load instances). The current corpus
contains no point-force examples or nonconstant pressure curves: parser/test
coverage is not learned coverage.

`physics-forcing-preview-v7` and `physics-forcing-native-v7` share architecture,
seed, loss, split, physical-group sampling, and a 24-epoch / 24,768-update budget.
The latter excludes all contact-preview, interval-contact and QS-force inputs.
Missing/poisoned-input tests establish that exclusion. Both retain world-motion
supervision with weight 0.1. Source hashes are frozen and checked before each run.
The matched experiment has completed; its results and limitations are reported above.

A one-epoch training-only smoke checkpoint measured 5.66 ms resident Metal
inference, 15.24 ms with GPU features, and 24.25 ms with CPU features plus transfer
on a synthetic 10,000-node / 30,000-edge graph with one projectile and two loads.
These exclude integration, topology updates, rendering and independent-body
validation; they are not end-to-end real-time results. All 64 Python tests pass,
including future-label poisoning, every curve/selector, source permutation,
table-knot gradients and checkpoint reload. Two viewer interpolation tests pass.

Artifacts: `runs/native-forcing-v7-2026-10-19/` contains the audit, smoke timing,
training plan, immutable source copies and per-model logs. The completed comparison
now permits inspecting both V7 candidates in the visualizer. Remaining input
gaps include projectile spin and remaining crush; rollout and a complete engine
state adapter remain unimplemented.

## Causal-preview cost investigation, 2026-10-19

A new `learned_impact_preview_bench` measures the actual saved scene's preview
without neural inference or oracle stepping. Warm-up and initialization are separate;
profiling runs separately from timed repetitions. It saves every force/torque-bin
float bit and contact flag for exact cross-build comparison, rejects overwrites,
and checks repeat/profile output identity.

```sh
export RUSTUP_TOOLCHAIN=stable WGPU_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run --release --manifest-path engine/stress-oracle/Cargo.toml \
  -p stress-learned --bin learned_impact_preview_bench -- \
  --scene path/to/recorded/scene.json --out /tmp/preview-benchmark.json --repeats 3
```

The output JSON and adjacent `.bins` file must not already exist.

On the physical 10,400-chunk initial wall, one 16.67 ms frame requires **4,062
substeps**. The baseline median preview time is 2.897 s; cloning is about 3 ms.
The profile attributes approximately 25% to solver work despite no bond evaluations:
the solver was rebuilding all chunk rotation/Jacobian caches during rigid-only
substeps. That cache is only consumed by bond response evaluation.

The solver now refreshes it only on ticks with firing bond jobs. Rigid-only steps,
fully disconnected fragments, and multirate ticks without firing bonds can avoid
this unused work; the next tick that evaluates bonds always refreshes current
rotations. No timesteps, contacts, geometry, tolerances or constitutive evaluations
are changed. A mode-transition regression checks the first explicit force against
the constitutive law evaluated directly from current rotations.

The first rigid-only guard measured 1.857 s median versus 2.897 s before. The final
guard generalizes the same rule to all empty bond-job ticks. An alternating-order
benchmark with three process starts per build measures **3.218 s → 1.765 s median**
(**1.82× faster**), with all 2,080,000 output bytes identical and the same 4,062
substeps. Warm-up is excluded; binary/scene/output hashes are saved. The host also
ran CPU training, so this is an observed comparison rather than an isolated speed
guarantee. Library tests (63, including the mode-transition regression) and all
seven multirate energy/momentum tests pass on the final guard. Dynamics (8) and
solve-mode (10) suites passed on the earlier rigid-only guard; all six recorder tests also pass on the final build, including exact oracle-label
and replay preservation with preview instrumentation. Even the improved 1.77 s preview remains far
outside a 16.67 ms real-time budget.

A forcing audit identifies 36 prescribed pressure loads and 16 blast loads among
the existing 80 scenes. V5/V6 world encoders consume projectiles and ground but do
not encode these laws directly. A model that avoids the contact preview must encode
known forcing explicitly from saved scenes, or declare/reject unsupported forcing;
simply zeroing preview loads would create missing-input ambiguity. No such incomplete
model is adopted here.

Artifacts: `runs/preview-profile-2026-10-19/` includes benchmark sources/binaries,
profiles, exact output bytes, a forcing audit and regression logs. The 128-scenario
V5 training run completed separately; see the expanded-layout results above.


## Longer V5 optimization, 2026-10-19

The 96-epoch physical-group run completed and selected epoch 53 using validation
fracture F1. The split, 80 recipes, 1,792 recorded intervals and model architecture
are unchanged. This uses four times the 24-epoch training budget and validation
opportunities; it is not a matched-compute loss ablation.

| Run | Validation F1 | Test F1 | Test precision | Test recall |
|---|---:|---:|---:|---:|
| 24 epochs, physical groups | 0.533 | 0.497 | 0.554 | 0.450 |
| 96 epochs, selected epoch 53 | 0.580 | 0.539 | 0.600 | 0.490 |

The paired 13-group bootstrap interval for the F1 difference is
−0.0216…+0.0956 (2,000 resamples). It still includes no improvement. One seed and
reused layouts do not establish reliable generalization. Local velocity RMSE is
0.01702 m/s versus 0.01859 m/s for zero increment (8.5% lower); displacement is
19.81 μm versus 20.02 μm. For the larger-displacement subset, error is 118.6 μm
versus 144.0 μm for zero increment; quiet-piece error remains worse than baseline.
Changed-damage MAE is 0.255. All these results reset to oracle state each interval.

On the separate physical 10,400-chunk wall, F1 is 0.187, precision 0.897 and recall
0.104: 200 caught, 1,721 missed and 23 extra breaks. The earlier short-data V5
checkpoint scored F1 0.151 on this case. Longer training does not solve unseen
geometry and scale; local motion remains unscorable across all fragment resets.
This case was not used for selection or normalization and is only one interval.

Metal inference with GPU feature construction matches CPU on all 410,844 held-out
bond break/timing decisions across 376 intervals. Maximum probability difference
is 1.20e-6; this validates forward numerical parity, not physical accuracy or rollout.

Artifacts: `runs/long-fit-v5-2026-10-19/{comparison,scale-evaluation,mps-parity}.json`,
checkpoint, source snapshots, plan and logs. The checkpoint is available in the
visualizer under **Longer impact recordings** and **Physical scale test**. It is
not an oracle replacement.

## World-motion supervision through fragmentation, 2026-10-19

The new `world-ballistic-residual/1` target contract preserves supervision when
fragment IDs change. Here the baseline is explicitly **constant world linear and
angular velocity**, not gravity-accelerated motion. Endpoint labels contain:

- position residual from `x0 + dt*v0`;
- principal world-axis rotation vector of `R1 * (Exp(dt*w0)*R0)^T`;
- world linear and angular velocity changes.

The rotation residual composes on the left. Targets are derived in float64 from
saved replay and converted to float32 only after subtraction; nonfinite or invalid
poses are rejected. Validity requires activity at both endpoints, not unchanged
fragment identity. Rotation exp/log tests include zero, near-pi rotations and
world-axis covariance. Inputs still use only snapshot i; endpoint labels are
separate tensors and are not required for inference. Poisoning labels leaves
all model predictions unchanged in tests.

On the physical 10,400-chunk recording, the local contract has 0 valid motion
targets; the new contract has 10,400. Reference decoding reconstructs saved endpoint
positions within 7.02e-9 m, rotation matrix entries within 1.19e-8, velocities within
4.67e-7 m/s and angular velocities within 9.39e-7 rad/s. These are label roundtrip
errors, **not learned prediction accuracy**. No scale-test data fits normalization.

`physics-preview-world-motion-v6` adds a zero-initialized 12-output world head to V5,
while preserving existing local/history semantics. Its scales come only from
training residual RMS; the auxiliary weight is explicit. Initial world prediction
is exactly the constant-velocity baseline. Evaluation reports endpoint residual
RMSE and an exact SO(3) orientation geodesic error against that baseline. This does
not apply predicted topology or world state to the engine and is not a rollout.
The visualizer adds world travel and endpoint-error fields plus an optional
Model endpoint geometry view. That view interpolates from each oracle start to
the learned endpoint, then resets; projectile motion and activity remain recorded.
Endpoint-error colors compare the constant-velocity baseline (left) and model
(right) on one shared scale. Existing views remain available. These controls are
disabled for checkpoints without the world head.

The matched 24-epoch V6 control (world-motion weight 0) and auxiliary (weight 0.1)
pair completed on the same 80-recipe extended dataset, physical-group sampling,
seed, split and 18,624 updates. All nine training source snapshots match exactly.
Checkpoint selection remains validation fracture F1: epoch 22 for control and 18
for auxiliary. The new layout catalog has not entered this fit.

| Held-out test metric | Constant-velocity control | World auxiliary loss |
|---|---:|---:|
| Fracture F1 | 0.459 | 0.462 |
| Fracture precision / recall | 0.452 / 0.466 | 0.664 / 0.354 |
| World position RMSE | 4.108 mm | 1.767 mm |
| World velocity RMSE | 0.2987 m/s | 0.1815 m/s |
| World orientation geodesic RMSE | 0.01451 rad | 0.01398 rad |
| World angular velocity RMSE | 0.6906 rad/s | 0.6691 rad/s |

Position error improves 57.0% and velocity error 39.2% over constant velocity,
including all 250,472 active chunk intervals and fragment resets. Angular motion
improves much less. The paired 13-group bootstrap interval for the fracture F1
difference is −0.0850…+0.0873, including no improvement. Validation fracture F1 **declines from 0.532 to 0.522**; higher
test precision comes with lower recall. This is useful world-motion supervision,
not evidence that this loss weight is best for fracture detection. The control's
fracture score differs from earlier V5 runs; use the matched pair for the loss
comparison rather than treating separate one-seed runs as deterministic controls.

On the separate 10,400-chunk wall, position RMSE only falls from 8.520 to 7.705 mm
and velocity from 0.5395 to 0.5263 m/s. Fracture F1 is 0.144 (150 caught, 1,771 missed,
17 extra). These modest world-motion gains do not solve geometry/scale transfer.
This interval was not used to fit scales, select checkpoints or calibrate thresholds.
Metal and CPU agree on all 410,844 held-out fracture/timing decisions; maximum
decoded world-position difference is 7.46e-8 m and velocity difference 5.25e-6 m/s.

Artifacts: `runs/world-motion-v6-2026-10-19/` includes target reconstruction audit,
matched source hashes, checkpoints, experiment plan, final metrics and scale audit.
The target/visualizer implementation passes 57 Python tests, plus existing JS pose
checks. A separate 160-interval one-epoch small-model run was only a pipeline smoke
check and is not used for any quality claim.


## Physical 10,400-chunk diagnostic and layout diversity, 2026-10-19

**The first large physical recording exposes poor generalization.** This is a
4 m × 0.4 m × 6.4 m concrete wall above a fixed foundation, built at unchanged
0.1 m chunk resolution. It has 10,400 chunks (10,240 unfixed), 28,180 bonds,
and one 16.67 ms oracle interval. It starts as one bonded cluster and ends with
205 clusters; it is not a 10,000-independent-rigid-body workload.

The existing `world-v5-2026-10-19/world` checkpoint, trained on smaller layouts,
catches 175 of 1,921 oracle breaks, misses 1,746, and predicts 219 extra breaks:
**precision 0.444, recall 0.091, F1 0.151**. No chunks, bonds, contacts, timesteps
or solver iterations were dropped. This single unseen case is a scale diagnostic,
not a representative generalization estimate or a rollout validation.

Every chunk changes its fragment coordinate identity in this interval, so there
are **zero comparable local motion targets** under the current local-delta contract.
The evaluator now reports motion RMSE as unavailable, together with scored/excluded
counts, instead of returning a misleading zero through an empty denominator.
The visualizer also masks those pieces in displacement/error views: gray means
coordinate reset and unscored; recorded world trajectories remain available.
This exposes a real target-schema gap for fragmentation, not evidence of perfect
motion prediction.

CPU and Metal, including feature construction from raw state, agree on all 28,180
fracture/timing decisions. Max break-probability difference is 2.98e-7. Numerical
parity does not fix the poor physical predictions. During concurrent work, recorded
oracle time was 90.91 s and causal rigid-preview time was 2.60 s. These are not
isolated speed measurements; they nevertheless show that the current CPU preview
path is far from a 16.67 ms real-time budget. Fast network inference alone is
insufficient. A compatible engine contact path or a separately tested causal
input model is still required.

Artifacts: `runs/layout-v4-2026-10-19/scale-{evaluation,parity}.json`, exact scene,
arrays, replay, generation sources, binary/recipe hashes and scripts. The visualizer
adds **Physical scale test · 10,400 chunks** (`?dataset=scale`).

A separate **48-recipe / 12-layout / 384-interval catalog has completed and passed audit**. It varies
width, height, thickness and physical doors/windows, with the same four impact
recipes for every layout. Solid/opening-stratified holdouts were fixed before
inspecting labels: 8 train, 2 validation, 2 test layouts, all load cases kept
together. New `layout_group` metadata is joined with physical and trajectory
identities by the splitter; inherited splits that leak a layout are rejected.
The audit/collection script verified all arrays, causal/replay sidecars and world
targets, then merged with the 80-recipe extended corpus. The combined dataset has
128 recipes / 2,176 intervals; 1,032 train, 704 validation and 440 test. There are
14,865 new breaks in the layout catalog. All 381,632 active chunk intervals have
world targets; 38,674 local coordinate-reset intervals remain excluded locally.

The frozen older 96-epoch V5 checkpoint scores F1 0.482 on the two new validation
layouts and 0.555 on the two new test layouts, using its original threshold.
All new layouts are unseen by that frozen checkpoint. A new 96-epoch V5 fit on the
combined dataset is running; selection uses validation only. The larger dataset
also increases updates from 74,496 to 99,072, so it is not a same-compute data ablation.
See `recording-audit.json`, `layout-baseline.json`, `training-plan.json` and
`training-split-verification.json` under the layout run. New fitted accuracy is pending.

The 96-epoch V5 run has completed; its selected-checkpoint results are reported above.


## Equal physical-group sampling, 2026-10-19

The longer dataset's two extended training clips contributed 31% of updates under
uniform interval sampling. Their 192 added tail intervals contained 355/12,296
training failures (2.9%) but received 24.7% of updates. This motivated an explicit
training-only sampling experiment, not dropping data or changing physics.

`--sampling physical-group` cycles equally through the 43 saved training groups,
using persistent shuffled interval queues within each group. All 776 training
intervals were visited at least three times over the run; physical-group totals
were 433–434 visits instead of 192–2,880 under uniform interval sampling. Longer
clips remain fully available for validation and evaluation. Family exposure also
changes (house 71.1% → 34.9%, concrete wall 20.6% → 46.5%, masonry wall 8.2% → 18.6%).
Thus this is a test of the overall training distribution, not an isolated causal
claim about quiet tails.

Both runs use V5, seed 0, identical sources/inputs/split and target scales,
24 epochs, 18,624 optimizer updates, and 24 validation checks. Selection uses
validation F1 and threshold only. The inference architecture and runtime cost
are unchanged.

| Policy | Validation F1 | Test F1 | Test precision | Test recall |
|---|---:|---:|---:|---:|
| Uniform intervals, fresh control | 0.531 | 0.474 | 0.678 | 0.365 |
| Equal physical groups | 0.533 | 0.497 | 0.554 | 0.450 |

The paired 13-group bootstrap interval for the F1 difference is −0.0442…+0.1063
(2,000 resamples), including no improvement. The trade-off increases recall and false positives. Test house F1 declines from
0.572 to 0.556; concrete wall changes from 0.418 to 0.420; masonry wall increases
from 0.266 to 0.770 on a small held-out subset. This does not establish a reliable
cross-layout improvement. The default remains uniform intervals; equal-group
sampling is available explicitly for further experiments.

The fresh control has the same saved split and regression-tested sample order as
the earlier extended fit, but CPU optimization is not bitwise reproducible. A
1.21e-8 first-epoch loss difference accumulates and changes the selected epoch
(22 → 13), while final test F1 is almost identical (0.474227 → 0.474257). One seeded
pair is insufficient to characterize optimization uncertainty.

`compare_state` now splits motion RMSE by whether the target vector magnitude is
above the saved training RMS, retaining physical units, no-change baselines and
fragment-reset exclusions. Empty subsets report unavailable errors. This prevents
quiet pieces from masking errors on the moving subset; it is a diagnostic, not a
training input or selection criterion. On 1,962 held-out chunk intervals above
the training displacement RMS, error is 139.2 μm (uniform) versus 135.1 μm
(equal-group), compared with a 144.0 μm zero-increment baseline. For the 210,796
smaller-motion intervals, equal-group error worsens from 14.93 to 17.30 μm
(zero baseline 14.54 μm). Changed-damage MAE improves from 0.299 to 0.260,
but neither the larger-motion gains nor history improvements establish rollout
stability. 37,714 fragment-reset chunk intervals remain excluded from these local
motion metrics.

Artifacts: `runs/sampling-v5-2026-10-19/{experiment-plan,exposure-audit,verification,
comparison}.json`, source snapshots, checkpoints, logs, exact per-interval visit
counts, and 47 passing Python tests. Neither model is a verified oracle replacement. Both sampling checkpoints are
available under the visualizer’s **Longer impact recordings · 120 ms** dataset.
The completed 96-epoch equal-group run is reported above under
`runs/long-fit-v5-2026-10-19/`.


## Frame-start world inputs and longer recordings, 2026-10-19

**The causal model is still not accurate enough to replace the oracle.** V5 adds
world motion, orientation, ground distance and all projectile observations from
snapshot *i* to predict interval *i*. Future snapshots and recorded energies are
not inputs. World/scene identity, timestamps, dimensions and rotations are checked;
input sidecar hashes and source snapshots are retained. An initially zero additive
encoder preserves V4 predictions before fitting. All projectiles are processed,
with O(nodes × projectiles) feature/encoder work; none are truncated.

The matched control disables the added branches. Both runs have 24 epochs,
14,016 optimizer updates, seed 0, identical 584/448/184 train/validation/test
intervals, and identical physical recipe membership. Effective learned capacity
differs. Epoch/threshold selection uses validation only.

| Checkpoint | Test break F1 | Precision | Recall |
|---|---:|---:|---:|
| World-input control | 0.466 | 0.624 | 0.372 |
| World inputs | 0.499 | 0.607 | 0.423 |

The paired physical-group bootstrap interval for the F1 gain is −0.0048…+0.1077
(13 test groups, 2,000 resamples). It includes no improvement; one seed and reused
layouts do not establish generalization. World-input local displacement RMSE is
27.89 μm versus 27.25 μm for zero increment. Motion remains a major weakness.

Six house recipes were extended from 24 to 120 ms without changing timestep or
solver settings. Original array, frame-start input and replay prefixes match exactly.
The 576 added intervals contain 426 additional failures. Replacing those six short
clips gives 80 recipes / 1,792 intervals, with 776/640/376 train/validation/test
intervals and no duplicated training frames. These are longer versions of existing
recipes, not six independent new scenes. The extended fit uses 18,624 updates, so
it changes both training coverage and compute.

| Same held-out intervals | Original V5 | V5 trained with longer clips |
|---|---:|---:|
| Original 184 intervals: F1 | 0.499 | 0.474 |
| Added 192 tail intervals: F1 | 0.483 | 0.507 |
| Tail true / extra / missed breaks | 14 / 16 / 14 | 17 / 22 / 11 |
| Combined 376 intervals: F1 | 0.499 | 0.474 |

Only two test recipes contribute tails, with 28 failures total. The longer fit
regresses overall and is retained as an experiment. Increasing quiet tail coverage
may alter training balance; this is a hypothesis for a controlled sampling
experiment, not a demonstrated explanation. Neither checkpoint is promoted.

Actual Metal measurements at 10,000 synthetic nodes / 30,000 bonds / one projectile:
resident model median 5.17 ms; GPU feature construction plus inference 13.00 ms;
CPU feature construction plus transfer plus inference 18.80 ms. These exclude
contact generation, state updates, topology application and rendering. Eight-projectile
work was also measured without truncation; short timings are too noisy to infer
projectile scaling. No physical 10k-body rollout or Slang integration is established.
CPU versus Metal, including GPU feature construction, agrees on all fracture decisions
across 184 held-out intervals / 207,147 live bond intervals; max probability difference
is 7.16e-7. Numerical parity does not imply oracle accuracy.

Artifacts are in `runs/world-v5-2026-10-19/`: `matched-comparison.json`,
`extended-comparison.json`, `tail-audit.json`, `extended-split-audit.json`,
`mps-gpu-feature-parity.json`, and `scale-benchmark-gpu-features.json`.
The visualizer retains earlier views and checkpoints, adds both matched V5 models
under **Causal input experiment**, and the extended model with six longer motion
replays under **Longer impact recordings · 120 ms**. Both panels still follow
recorded oracle trajectories; model-driven collapse has not been validated.


## Constrained state objective and frozen-head fit, 2026-10-19

**History prediction improved; fracture accuracy and autonomous motion are still
insufficient for replacement.** Two `physics-preview-state-v4` checkpoints are
available under Dataset → Causal input experiment. All earlier checkpoints and
views remain available. `runs/state-v4-2026-10-19/` contains exact sources, training
logs, checkpoints, comparisons, decoder parity, GPU measurements, and a verified
visualizer screenshot.

V4 retains the 108,878-parameter architecture and causal preview inputs. It
initializes motion to zero and joint history to the previous state, instead of
predicting arbitrary absolute history. Damage, crush, fatigue and continuous
rebar-broken surrogates stay in [0,1] and cannot decrease; kappa, kappa_c, rebar work
and dissipated energy cannot decrease. The remaining history fields are signed
residuals in encoded units. This is not a full constitutive update: continuous
rebar-broken values are not discrete solver state, energy balance is not enforced,
and the packed ductility schema remains incomplete.

Motion supervision uses physical MSE normalized by four **training-only** RMS
scales (displacement, rotation, linear velocity, angular velocity). History uses
encoded end-state MSE normalized by training increment RMS per field. Scales are
checkpoint buffers. The break/timing/utilization/force objectives and outer loss
weights remain unchanged. Boundary-gradient and unchanged-state tests caught and
fixed a zero-derivative initialization error before the completed experiments;
the stopped one-epoch run is retained and explicitly marked aborted.

The first candidate trains jointly for the same 24 epochs / 14,016 updates as the
preview-v3 baseline, with identical seed, split, capacity and schedule. It selects
epoch 18 by validation fracture F1. The follow-up loads the original v3 checkpoint,
freezes its trunk and break/timing/utilization/force heads, and fits only the
**1,666 motion/history parameters** for another 24 epochs. It selects epoch 22 by
validation weighted motion/history loss and retains the source break threshold.
This second run uses additional fitting, not equal total training compute.

All test results use the same 184 intervals / 207,147 live bond intervals:

| Model | Break F1 | Damage MAE | Crush MAE | Damage MAE where damage/crush changes |
|---|---:|---:|---:|---:|
| Causal preview v3 | 0.4720 | 0.13460 | 0.025470 | 0.38172 |
| Jointly trained constrained v4 | 0.4639 | 0.03488 | 0.000793 | 0.35477 |
| Frozen fracture + new state | 0.4720 | 0.03353 | 0.000982 | 0.33968 |

The ~75% reduction in pooled damage MAE is partly removal of incorrect quiet-state
drift; improvement on changing-damage intervals is ~11% for the frozen-head model.
Both v4 candidates beat persistence on encoded damage/crush/kappa/kappa_c/dissipated
RMSE. Neither has any audited decreasing irreversible fields or out-of-range
bounded fields. The old model has 93,875 decreasing-damage predictions, among
other violations. Zero changes in fatigue/rebar fields in this particular test
set do not establish coverage of those physical mechanisms.

| Model | Local displacement RMSE (m) | Rotation (rad) | Velocity (m/s) | Angular velocity (rad/s) |
|---|---:|---:|---:|---:|
| Zero increment | 2.7249e-5 | 8.8266e-5 | 0.027092 | 0.142939 |
| Causal preview v3 | 8.4968e-5 | 1.1860e-4 | 0.028177 | 0.141812 |
| Jointly trained constrained v4 | 2.7393e-5 | 8.0715e-5 | 0.026998 | 0.116810 |
| Frozen fracture + new state | 2.7158e-5 | 8.7994e-5 | 0.027079 | 0.142551 |

Joint training improves rotation/angular velocity but narrowly loses to zero
displacement and reduces fracture F1. Frozen-head motion is only 0.05–0.33% below
zero-increment RMSE, and angular velocity is worse than the source model. These
are not strong motion predictors. Motion scoring covers **87,490 chunk intervals**;
**35,302 (28.75%)** are excluded because fragmentation resets local coordinates.
All inputs still use oracle start states/topology, so recurrent drift, new
fragments and autonomous collapse remain untested.

`decoder-parity.json` verifies exact CPU output equality before/after vectorizing
the history decoder on all 184 test intervals for both checkpoints. It also verifies
all frozen fracture/timing/utilization/force outputs exactly equal their source.
Real-data CPU/Metal parity has zero break/timing decision disagreements; maximum
break-probability difference is 5.96e-7. These are PyTorch checks, not Slang parity.

An isolated alternating 24-sample synthetic 10,000-node / 30,000-bond benchmark
measures GPU-resident medians of **4.38 ms** for the original decoder and **4.16 ms**
for the vectorized decoder; p95 is 4.58/4.68 ms, so this is a modest median gain
with overlapping latency variation. Original/vectorized GPU outputs match exactly
on that graph. CPU feature construction + transfer + vectorized GPU inference is
**14.00 ms** median. These exclude contact solving, topology changes, physical state
application, and rendering. Use `decoder-benchmark-isolated.json`; the earlier
vectorized benchmark overlapped another check and is not the timing reference.

All 37 Python tests pass, including boundary gradients, no-label input leakage,
checkpoint scale roundtrips, history audit counts, and frozen-output invariance.
Next work must address missing causal state information and contact coupling as
well as longer/diverse scenarios. The current encoder has local elastic state and
preview loads but no explicit world rigid velocity or projectile state. The replay
already records their start-of-interval values; any future input addition must use
only those starting snapshots and prove that later snapshots cannot leak in.

## Shorter horizon, target audit, and Metal inference, 2026-10-19

**A four-times-shorter prediction interval did not improve fracture F1. GPU
inference is fast enough to justify further work, but replacement accuracy and
the recurrent physical state update remain unresolved.** Artifacts and exact
training/generation sources are in `runs/horizon-2026-10-19/`.

The 32 existing wall recipes were re-recorded at 1/240 s, preserving their
0.133333 s durations: 1,024 intervals versus 256 at 1/60 s. This is additional
temporal coverage of existing recipes, not 32 independent new scenarios. Both
models inherit the original 20/6/6 train/validation/test recipe membership, use
the same causal preview architecture, and receive 15,360 optimizer updates,
24 validation checks, and the same per-update learning-rate schedule. Best
checkpoint and threshold use validation only. New splitting logic unions both
identical first frames and explicit recipe identity to prevent cadence variants
from leaking across partitions.

For comparison, predictions and actual breaks are unioned by stable bond ID over
the same 16.7 ms windows; a repeated positive is counted once per window.

| Prediction interval | Test precision | Test recall | Common-window F1 |
|---|---:|---:|---:|
| 16.7 ms | 0.751 | 0.342 | 0.4703 |
| 4.17 ms, four calls per window | 0.528 | 0.421 | 0.4683 |

The fine-minus-coarse difference is −0.0020, with a paired recipe bootstrap 95%
percentile interval [−0.0245, +0.0204]. This covers only six held-out wall recipes
and one training seed. Fine cadence also receives four oracle state refreshes,
so it is not a closed-loop test. Native fine-interval F1 is 0.4457. Both models'
local displacement and velocity errors remain worse than zero-increment baselines.

Changing frame boundaries also changes stability-bound recomputation/substep
rounding after fracture. The recordings therefore do **not** have identical
oracle labels: held-out final broken-set Jaccard ranges from 0.483 to 0.954, with
final position RMS differences up to 0.118 m. `horizon-audit.json` measures this
instead of assuming unchanged motion. Across all 32 wall recipes, preview force
relative RMS error is 17.13 at coarse cadence and 24.12 at fine cadence; each uses
its own oracle bin averages, so temporal smoothing differs. Neither cadence nor
the frozen-deformation preview has been adopted in the engine.

### What the labels say about the objective

`target-consistency.json` audits all 257 original expanded scenarios. Even using
the **true future oracle peak utilization**, the rule utilization ≥ 1 produces
263,198 false breaks and F1 0.250. It catches all 43,880 failures, but exceeding
strength is not sufficient for immediate fracture under the recorded constitutive
history. A rule using true end damage/crush ≥ 0.999 produces F1 0.991. This is an
oracle-label diagnostic, **not learned accuracy or a deployable future input**.

No recorded increments below −1e-6 were found for damage, crush, kappa, kappa_c,
fatigue, rebar work, rebar broken, or dissipated energy. These observations favor
testing an irreversible history update and directly supervising damage progression,
alongside calibrated break/timing prediction. The existing unconstrained history
head is not sufficient evidence of that property. State losses must also beat
zero-increment baselines and be evaluated on accumulated rollouts; fracture F1
alone must not select a purported replacement integrator. These are the next
experiments, not claims of an already validated new objective.

### Measured GPU opportunity

On this Mac's actual Metal device, with the fine checkpoint and a synthetic
10,000-node / 30,000-bond graph (three warmups, 12 synchronized timed calls):

| Forward path | Median | p95 |
|---|---:|---:|
| CPU, tensors/features resident, four threads | 30.55 ms | 37.78 ms |
| CPU including feature construction | 36.09 ms | 39.02 ms |
| Metal, tensors/features resident | **3.37 ms** | 3.87 ms |
| CPU features + full batch transfer + Metal | **13.90 ms** | 14.43 ms |

These exclude contact solving, changing topology, applying state, and rendering.
The resident path additionally excludes feature construction/transfers. Four fine
calls would consume about 13.5 ms resident or 55.6 ms with the measured full
preparation/transfer path per original 16.7 ms window. This is not an end-to-end
speedup over the oracle or physical validation of a 10k-body simulation.

Separate real-data CPU/Metal parity checks cover 192 held-out intervals and
295,555 bond intervals: zero break or timing-class decision disagreements; maximum
break-probability difference 5.66e-7. See `scale-benchmark-mps.json` and
`mps-parity.json`. This establishes PyTorch Metal forward parity on those inputs,
not Slang parity, recurrent integration, or engine compatibility. The sandbox
masked Metal availability; actual GPU checks were run on the host.

Validation: 19 Rust tests in `stress-learned` pass, including unchanged-oracle
preview recording, complete-clip resume, and cadence/provenance invariants. All
33 Python tests pass, including split leakage and common-window counting checks.

## Causal contact input experiment, 2026-10-19

**The current frozen-deformation contact preview is rejected as a replacement
interface.** It removes future-oracle-force inputs, but the matched model loses
substantial fracture recall and the CPU preview remains expensive. This is a
measured negative result, not a new best model or an engine change.

The existing 80 v3 recipes were re-recorded with `--rigid-preview`: **1,216 intervals**.
Every original static array, frame input/label array, and world-motion replay frame
matches the previous recording exactly. These augment the same 80 scenarios; the
unique expanded corpus remains **257 scenarios**, not 337. Raw and visualizer copies,
exact generator sources/binary digest, per-frame timings, audit, checkpoints,
training source snapshots, and comparison report are under
`runs/causal-preview-2026-10-19/`.

Both models use H=48/P=16, 108,878 parameters, identical physical features/objective,
24 epochs, seed 0, split seed 17, and AdamW/cosine settings. Only load bins/contact
flags differ. There are 69 physical groups, with **584 training / 448 validation /
184 test intervals**; no duplicate training intervals were removed. Best epoch and
threshold use validation only. Test contains 13 physical groups.

| Inputs supplied to the model | Test precision | Test recall | Test F1 | Utilization log-MAE |
|---|---:|---:|---:|---:|
| Recorded oracle interval force history | 0.640 | 0.734 | 0.684 | 0.1574 |
| Frame-start rigid contact preview | 0.640 | 0.374 | 0.472 | 0.1710 |

The F1 difference is −0.212; paired physical-group bootstrap 95% percentile interval
is [−0.283, −0.133] across these scenarios. That interval does not cover training-seed
uncertainty or unseen structural layouts. All models still receive oracle state,
damage, and graph at each interval start. This is **not closed-loop evaluation**.
The split is deliberately conservative about identical initial frames: 13 blast
recipes group into validation, only one blast recipe trains, and two test. Do not
interpret the pooled result as balanced evidence across all load combinations.

The preview's aggregate force RMS error is **5.265×** oracle force RMS; torque error
is **8.420×** oracle torque RMS, computed separately in their own units. Worst force
mismatch is in several wall impacts; the frozen geometry cannot reproduce evolving
elastic contact/fracture. Preview wall time totals 1,542 s versus 2,528 s for oracle
steps across concurrent jobs (**61%**); total generation elapsed 684.5 s with eight
workers. Those are CPU recording timings including clone/contact work and worker
contention, not engine/GPU throughput. Model inference would add further cost.

Local velocity RMSE is 0.02816 m/s with oracle inputs and
0.02818 with preview inputs, versus 0.02709
for a zero-increment baseline. Local displacement is also worse than that baseline
in this restricted test. Selecting checkpoints on break F1 does not establish a
good state integrator. Reweighting losses cannot repair an inaccurate input contract.

Next experiments should test a shorter contact/stress prediction horizon with
causally available loads, then measure recurrent state/damage and fragment topology
errors. Preserve the existing contact solver and oracle quality. Add longer tails,
independent layouts and balanced mass × speed × strength combinations. Require
velocity/energy/history checks alongside fracture metrics before replacement.

The visualizer retains all four prior checkpoints and all old views. Choose
**Dataset → Causal input experiment** to compare the two matched checkpoints.
**Contact inputs** shows max force magnitude across four bin averages; **Contact
input error** shows max vector error across those bins. Both panels share a scale
in newtons. They show inputs, not predicted stress; stress/break views remain
available. The URL `/?dataset=preview` opens the experimental dataset directly.

```sh
python -m learned_impact.train_physics --dataset runs/causal-preview-2026-10-19/dataset \
  --out /tmp/oracle-inputs --arch physics-v2 --epochs 24 --threads 4
python -m learned_impact.train_physics --dataset runs/causal-preview-2026-10-19/dataset \
  --out /tmp/preview-inputs --arch physics-preview-v3 --epochs 24 --threads 4
python -m learned_impact.compare_inputs --dataset runs/causal-preview-2026-10-19/dataset \
  --oracle-weights /tmp/oracle-inputs --preview-weights /tmp/preview-inputs \
  --out /tmp/matched-input-comparison.json
```

## Physics v2 update, 2026-10-19

The visualizer now supports both the original checkpoints and `physics-v2` PyTorch
checkpoints. The new dataset contains 257 events / 2,547 intervals, including 80
new parameter sweeps. See [DIVERSITY.md](DIVERSITY.md) for exact axes, recording
contracts, generation commands, and the measured coverage gaps. The launcher
selects the expanded run once training finishes, retaining the old checkpoint and
all existing views. The badge distinguishes training, validation, test, and unseen
events using saved split membership; it does not infer this from a filename.

### Controlled comparison on the original 177-event corpus

Both architectures used the same seed, H=48/P=16, 24 epochs, AdamW/cosine schedule,
and physical-group split: 656 unique training frames, 276 validation frames and
281 test frames. 118 duplicate training frames were removed; clips sharing the
same physical first frame (including short prefixes and QS variants) stay together.
There are 130 physical groups, with no group crossing a partition. Epoch and
threshold selection use validation only. These are one-seed results, not confidence
intervals or evidence of generalization to new structural layouts.

| Model / decoder | Test precision | Test recall | Test F1 | Utilization log-MAE |
|---|---:|---:|---:|---:|
| Legacy model, original five-way argmax | 0.714 | 0.391 | 0.505 | 0.1555 |
| Same legacy checkpoint, validation-tuned total-break threshold | 0.532 | 0.563 | 0.548 | 0.1555 |
| Physics v2, separate break/timing heads | 0.687 | 0.686 | **0.687** | **0.1246** |

The improvement exceeds changing the decoder alone. On all 80 new scenarios before
training on them, legacy F1 was 0.411 and v2 F1 was 0.594. These are conditional
predictions given recorded oracle inputs, including interval force history.

The expanded-data grouped run holds out 715 test intervals and reaches precision
0.749, recall 0.633, F1 **0.686**, utilization log-MAE **0.09245**. Its test set differs
from the original experiment, so the two 0.687/0.686 scores are not a data ablation.
The separately refitted all-data checkpoint is for inspecting fit in the visualizer;
it must not be reported as held-out accuracy.
Grouping on a shared physical first frame is conservative: distinct delayed loads
can share a quiet initial interval and be kept together. In this split, 13 of the
16 new blast recipes are in the test partition. Inspect the saved membership and
per-family metrics; this is not a uniformly balanced test of every parameter axis.

On the **same 408 new-scenario test intervals**, the original-corpus v2 checkpoint
scores F1 0.572 and the expanded-data held-out checkpoint scores **0.657**; utilization
log-MAE falls from 0.1281 to **0.08785**. Neither model trained on those new test
scenarios. The training partitions differ as well as their size; this is evidence
that the added coverage helps, not an isolated causal estimate for each factor.
The all-data refit scores F1 **0.736**, precision 0.792 and recall 0.687 on all 2,547
recorded intervals, at threshold 0.70 transferred from the expanded validation run.
It trained on 2,262 unique frames after removing 285 exact duplicate training frames.
The transferred threshold does not establish probability calibration for the refit.

### Model and objective changes

`physics.py` is a separate versioned architecture (108,878 parameters versus the
legacy 103,954), so old checkpoints retain their behavior. It consumes support and
contact flags, timestep, previous force, geometry, material strength, stiffness,
fracture energy, rebar, buckling/Weibull fields, and available bond coordinate frames.
Unknown legacy QS convention and missing geometry have explicit markers. Fixed SI
reference scales plus signed-log transforms preserve absolute load magnitude; the
legacy graph-RMS normalization could map doubled loads to identical inputs.

Attention assignments normalize over tokens per node; pooling separately normalizes
token mass. This removes the old 1/N broadcast attenuation, tested by repeating a
five-node input to 10,000 nodes. Complexity remains linear in nodes/edges for a fixed
token count, but runtime/GPU parity and 10k-body physical behavior remain unproven.

A synthetic 10,000-node / 30,000-edge forward pass on this Mac, CPU/four threads,
measured median **29.6 ms** with features precomputed and **36.1 ms** including feature
encoding (12 timed calls after three warmups). This excludes graph construction,
contact/physics, transfers and rendering. It is not a real-time engine benchmark
or a physical accuracy test; results are in `scale-benchmark.json` beside the runs.

The objective is:

```text
BCE(any break; positive weight 3)
+ 0.15 × CE(failure quarter | true break)
+ 0.30 × threshold-weighted Huber(log1p peak utilization)
+ 0.10 × Huber(signed-log, physically scaled chunk delta)
+ 0.05 × [Huber(scaled end joint state) + Huber(scaled end bond force)]
```

Occurrence and time are decoded separately, so uncertainty between failure quarters
does not suppress total break probability. State scales are 0.01 m, 0.01 rad,
1 m/s and 1 rad/s; velocity no longer dominates a mixed-unit MSE. Joint/force history
heads add previously missing supervision. They are not yet an integrated physical
state update, and no claim of nonnegative dissipation or irreversible damage is made.

Loss ablations on the original split: positive weight 1 achieved validation F1
0.687 (test 0.677), versus weight 3 at validation 0.705 (test 0.687). Increasing the
state weight from 0.1 to 1 achieved validation F1 0.708 (test 0.653); it did **not**
fix velocity prediction. The small validation difference is not a robust basis for
claiming an optimum. The expanded run retains the preselected fracture-focused
configuration, and all ablation checkpoints/metrics remain available locally.

**Motion is not solved.** On the original test set, displacement RMSE improved from
1.393 mm to 0.415 mm, but linear-velocity RMSE worsened from 0.0346 to 0.0536 m/s;
the zero-increment baseline is 0.0508 m/s. Expanded-test velocity RMSE is 0.0410 m/s
versus zero's 0.0387 m/s. Better fracture classification is therefore insufficient
to replace the solver. A future motion objective/head must beat the zero and legacy
baselines without losing fracture quality. No engine runtime was switched to v2.

### Reproduce v2 training and audit

From `engine/stress-oracle/learned-impact/python`:

```sh
.venv/bin/python -m learned_impact.experiment collect \
  --dataset ../runs/combined-2026-10-19/dataset \
  --dataset ../runs/permutations-v3-2026-10-19 \
  --out /tmp/stress-expanded-copy
.venv/bin/python -m learned_impact.coverage \
  --dataset /tmp/stress-expanded-copy --out /tmp/stress-coverage.json
.venv/bin/python -m learned_impact.train_physics \
  --dataset /tmp/stress-expanded-copy --out /tmp/stress-v2-heldout --epochs 24
# Refit only after evaluating; transfer a separately validated decision threshold:
.venv/bin/python -m learned_impact.train_physics \
  --dataset /tmp/stress-expanded-copy --out /tmp/stress-v2-all \
  --epochs 24 --all-data --threshold 0.70
```

Run directories contain `run.json`, `split.json`, epoch history, selected
`state_dict.pt`, per-family/pooled metrics, and source snapshots for the new runs.
`runs/physics-v2-2026-10-19/controlled-evaluation.json` includes the original matched
experiment, decoder-only calibration, external 80-scenario evaluation, and state
metrics against zero. `runs/expanded-2026-10-19/coverage.json` audits every event.
Large datasets and checkpoints remain gitignored local artifacts.

## Using the comparison viewer

From the repository root:

```sh
engine/stress-oracle/scripts/view_learned.sh
# Open http://127.0.0.1:8791
```

The viewer replays moving chunks and the ram/car in two synchronized 3D panels.
The left shows oracle peak stress and breaks; the right shows the selected model’s
predictions on the **same oracle motion**. Use **Play impact**, the timeline,
frame stepping, or **First break**. Drag to orbit both cameras and scroll to zoom.

**Stress + breaks** uses one fixed load/strength scale: blue for low load, amber at
1.0 (the strength limit), and red above it. Each chunk takes the maximum utilization
of its attached live bonds. White outlines highlight chunks attached to a newly
broken bond. **Caught / missed / extra breaks** colors the model panel green, pink,
and orange; on chunks touching multiple categories, missed takes priority, then
extra, then caught. The oracle panel retains the stress field. Plain counts and a
sentence explain each interval’s predictions.

The diagnostic views remain available alongside playback: **New bond failures**,
**Peak utilization**, **Displacement increment**, and **Prediction errors**, plus
**Displacement prediction error**. Geometry can be switched to **Undeformed inspection**
without changing frame predictions. Every mode explains its units and aggregation.
Displacement compares local deformation increments, excluding cluster rigid motion;
its colors share one scale across both panels. Hover a piece for numeric bond and
state values; expand numerical diagnostics for precision, recall, F1, utilization
log-MAE, and local displacement RMSE.

Repeat `--weights /path/to/checkpoint-directory` to add a checkpoint. Every checkpoint
must contain `state_dict.pt`; missing weights fail instead of running a random
model. Its SHA-256 and provenance are under “How to read this replay and its limits”.

This is **teacher-forced recorded-frame evaluation**, using the exact PyTorch model.
Each prediction starts with the recorded oracle inputs and live graph. The all-data
checkpoint is evaluated on its training corpus, so its metrics measure fit, not
unseen-scenario generalization. A new failure is scored only while a bond was still
live at the beginning of the frame. Already-broken bonds are excluded using the
previous oracle labels (old files repeat failure class 4 after breakage).

Older training events lack world-space poses. The replay exporter reconstructs the
original source scene, then verifies **every static and per-frame input/label array
byte-for-byte** before attaching motion. The sole exclusion is `bond_qs_force`, an
optional read-only probe whose on/off setting was not always recorded in old metadata;
it does not affect motion. A mismatch rejects the replay. Static-only recordings
remain available and explicitly labeled. The Python server checks the event fingerprint,
frame count, time alignment, and chunk-array shapes before returning a replay.

New oracle recordings save `scene.json`, `context.json` and `replay.json` automatically: initial pose,
every end-of-frame pose, stable chunk indices, activity, cluster IDs, and impactor
shape/pose/velocity. Collection preserves these sidecars and stamps the event hash.
World poses/velocities are visualization data. Static context supplies optional bond
frames and feature conventions to v2; cluster history masks discontinuous local
coordinate resets in its delta loss. Future poses never enter its feature encoder.

Playback interpolates positions and rigid rotations. Stress is the **peak over each
recorded interval**, held for that interval; it is not an instantaneous stress wave.
The model does not drive these trajectories. These short recordings begin just before
contact (e.g. the default house clip is 24 ms); no artificial approach or collapse is
added. Playback is slowed to 4, 8, or 16 seconds. The ground grid is visual context.

The Slang exporter is still available, but its numerical parity with the PyTorch
model is not established. This comparison intentionally uses the checkpoint backend
that was trained. It does not alter `blast_parity` or the live fracture lab.

## Existing stress-viz entry point

```sh
cd engine/stress-oracle
export RUSTUP_TOOLCHAIN=stable WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run --release -p stress-viz --no-default-features -- learned \
  --dataset learned-impact/runs/combined-2026-10-19/dataset \
  --weights learned-impact/runs/combined-2026-10-19/model \
  --geometry learned-impact/runs/combined-2026-10-19/geometry.json
```

The learned command opens a localhost comparison service, with the URL printed in
the terminal. It uses `learned-impact/python/.venv/bin/python`; set
`STRESS_LEARNED_PYTHON` to override the interpreter. Native render/view commands are
unchanged, and the learned command also works in the default-feature build.

## Reproduce collection and training

Run these from `engine/stress-oracle/learned-impact/python`. Use a new run directory
for another experiment; collection/training refuse to overwrite an existing one.

```sh
.venv/bin/python -m learned_impact.experiment collect \
  --discover /tmp --out ../runs/combined-2026-10-19/dataset
.venv/bin/python -m learned_impact.experiment train \
  --dataset ../runs/combined-2026-10-19/dataset \
  --out ../runs/combined-2026-10-19/model --epochs 12 --device auto
```

Collection discovers immediate `/tmp/li-*` dataset/event roots with manifests. You
can instead supply repeated `--dataset DIR` arguments. It validates all arrays,
then copies the unique events into the run folder. Identical complete events are
deduplicated by their array contents and simulation metadata; provenance retains
all original paths. Partially overlapping clips are retained. Future holdout
experiments must split by scenario/load family across every source dataset.

Generate display geometry from the repository root:

```sh
export RUSTUP_TOOLCHAIN=stable WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run --release --manifest-path engine/stress-oracle/Cargo.toml \
  -p stress-learned --bin learned_impact_geometry -- \
  engine/stress-oracle/learned-impact/runs/combined-2026-10-19/dataset \
  engine/stress-oracle/learned-impact/runs/combined-2026-10-19/geometry.json
```

Backfill verified motion for an older combined dataset (from the repository root):

```sh
export RUSTUP_TOOLCHAIN=stable WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
cargo run --release --manifest-path engine/stress-oracle/Cargo.toml \
  -p stress-learned --bin learned_impact_replay -- \
  engine/stress-oracle/learned-impact/runs/combined-2026-10-19/dataset \
  engine/stress-oracle/learned-impact/runs/combined-2026-10-19/replays --jobs 8
```

Add `--event EVENT_ID` for one event. Verified results are cached by fingerprint;
`status.json` lists successes/rejections. A rejected replay makes the command exit
nonzero while preserving verified outputs. The viewer finds `replays/` alongside
`geometry.json`, or native `replay.json` sidecars inside collected events.

To score every event and write a report without starting the server, add
`--report /path/to/report.json` to `view_learned.sh`. Reports include pooled TP/FP/FN,
precision, recall, F1, per-family totals and per-event totals. Frames with no actual
or predicted failures show F1 as “—”, avoiding a misleading perfect-positive score.
The trainer also retains its historical frame-averaged F1 for comparison; prefer
`pooled_fail_f1` for overall performance.

## Historical combined baseline, 2026-10-19

- **173/177 events have verified moving replays**. Four legacy probes remain static-only:
  `li-oracle-tension`, `li-oracle-wall1`, `cantilever_n6`, and `cantilever_n14`.
  Their regenerated input/label arrays differ; motion is not attached.
- 15 source roots → **177 unique events, 1,331 frames**. Includes smoke, mini,
  wall sweeps, prove, diversity v1/v2, and individual oracle probes.
- H48/P16 MGN-T, 103,954 parameters, 12 shuffled epochs / 15,972 updates,
  Adam at 3e-4, seed 0, gradient-norm clipping at 5.0.
- CPU, 4 PyTorch threads; training epochs took 143.5 seconds on this host.
- Mean online loss fell from 0.9582 to 0.6581. Final evaluation loss: 0.6504.
- New-failure pooled F1 **0.4916**, precision **0.8431**, recall **0.3469**.
  TP 8,104; FP 1,508; FN 15,254. These are in-sample results.
- Frame-average F1 is 0.8049, demonstrating why the pooled metric matters.
- The older diversity-v1 `full` checkpoint scores pooled F1 **0.0724** on the same
  combined corpus and live-graph evaluation protocol (precision 0.0406, recall
  0.3338). Its original training used a different graph policy and less data, so
  this is a common-input comparison, not an isolated architecture ablation.
- New-model pooled F1 by family: cantilever **0.9639**, house **0.4603**, wall_b5
  **0.4538**, masonry wall **0.6952**, spall **0.6822**. Families without any true
  or predicted new failures have F1 undefined rather than perfect.
- `runs/combined-2026-10-19/model/` contains the checkpoint, configuration,
  epoch history, evaluation metrics, and development GPU exports.
- `runs/combined-2026-10-19/comparison-metrics.json` contains all-event results.

`runs/` is Git-ignored. The copied corpus and model persist locally independently
of `/tmp`, but need separate backup/sharing; a fresh clone does not contain them.

Validation: Python regression tests cover normalization, removal of historical
breaks, empty-edge frames, deduplication/provenance, missing-checkpoint failures,
exact comparison outputs, and rejection of misaligned replay sidecars. Node tests
check rigid interpolation and timeline endpoints. Rust recorder tests verify initial
and final chunk/projectile poses;
the geometry exporter was run against all 177 collected events.

## Historical baseline audit (before the v2 changes above)

This is a useful prototype, not yet a validated replacement for the dynamic stress
solver. The all-data checkpoint catches 8,104 of 23,358 new breaks and misses 15,254
(65.3%); precision is 84.3%, recall 34.7%, pooled F1 49.2%, on its own training corpus.
Among correctly detected breaks, 6,740/8,104 have the right quarter-frame bin.
The quiet-frame-heavy mean F1 of 80.5% does not summarize fracture performance.

The reproducible read-only audit is `python -m learned_impact.audit --dataset DIR
--weights DIR --out NEW_REPORT.json`. The current report is
`runs/combined-2026-10-19/audit.json`, also available at `/api/audit` in the viewer.
It does not change weights, decoding, thresholds, physics, or the training dataset.

### Measured dataset limitations

- 177 events / 1,331 frames; 952 frames (71.5%) contain no new failure. Only 1.66%
  of live bond-frame labels are positive. Preserve quiet examples, but sample
  deliberately across quiet, near-failure, impact, and propagation regimes.
- 215 frames repeat exactly the inputs consumed by the normalized network.
  No conflicting failure labels were found among those exact repeated inputs.
  Whole-event deduplication does not remove repeated prefixes/overlapping clips.
- 52 cantilever, 69 wall_b5, 30 house events dominate; some families have only one.
  Maximum graph size is 2,016 chunks, and all clips are at most 0.533 s long.
  This does not establish accuracy or throughput at 10,000+ active rigid bodies.
  Test realistic island-size distributions and the largest islands, including
  full contact/feature packing/split costs, not just neural forward latency.
- This all-data fit has no held-out validation/test set. Group future splits by
  source scene, seed, geometry/material/load families and overlapping prefixes;
  never split correlated frames randomly. Retain a final unseen benchmark.
- Four legacy recordings could not be reproduced with the current oracle. Audit
  their label/packing provenance before deciding whether to migrate or quarantine
  them for a new training experiment. Do not silently relabel the existing corpus.

### Input and deployment blockers precede loss tuning

`model.py::encode_inputs` consumes mass/inertia/size/state/load bins/gravity plus
joint history and QS force. It omits recorded support flags, bond area/offsets,
stiffness, strength, fracture energy, rebar, and Weibull values. `dt` is not an
input despite four timestep regimes ranging from 0.1 to 16.67 ms. Stored strengths
range from 0.3 MPa to 5,000 MPa; treating every bond as the same constitutive model
is not an adequate assumption. Use physical parameters rather than relying on
scene or material IDs to memorize behavior. Include relative geometry/bond frames
and support/boundary conditions with a consistent coordinate convention.

`normalize.py` divides loads, QS forces, and mixed joint-state components by
per-graph RMS values; `_scales` are not passed to the model. The audit doubled a
recorded load with RMS 110.086 → 220.173 and observed **zero change in any consumed
normalized input**. With all other state held fixed, this representation cannot
distinguish those load amplitudes. Use training-set-fitted per-feature scales or
physical nondimensional groups (including force/capacity, torque/capacity-length,
and timestep/characteristic time); preserve any necessary physical scale inputs.
Velocity and angular velocity need consistent time scaling, not only length scaling.

`live.rs::FEATURE_GAPS` documents zero live load bins and current force substituted
for the QS probe. Oracle bins are filled after stepping the recorded frame, with
loads transformed by end-frame cluster orientation. Establish exactly when the
contact solver supplies those bins to a replacement stress step, and which frame
basis is used. Recorded force histories can be valid exogenous inputs to a
conditional stress surrogate, but the replay is not evidence of a causal online
predictor when those histories are unavailable or differ after fracture. Resolve
input semantics, true-QS versus previous-force mixing, and PyTorch/Slang numerical
parity before measuring a coupled model against the oracle.

The current network also lacks the joint-state-end and force-end heads specified
for carrying stress history forward. Predict those states or explicitly keep a
compatible analytic state update; oracle reset on every frame hides accumulated
state/history error.

### Objective and decoder findings

The current objective is unweighted 5-class focal CE (gamma 2), a threshold-weighted
Huber loss for log1p utilization, and one MSE over all 12 state components, summed
with coefficients 1/1/1. Focal loss is a reasonable imbalance baseline, not proof
that this combination is best. See the [original focal-loss paper](https://arxiv.org/abs/1708.02002).

Mean terms are 0.01453 failure, 0.03667 utilization, 0.59925 state delta. The state
term is **92.1%** of scalar loss; **98.2% of its squared error** is linear velocity.
This alone does not establish gradient dominance. The audit also measured shared
parameter gradients on one failure frame per positive family: wall and house
examples have much larger delta gradients than failure gradients, while cantilever
and masonry-wall examples differ. Treat these as examples, not a full gradient study.
Split displacement/rotation/velocity/angular-velocity losses, scale physically,
and monitor task gradients; choose weights on validation/rollout outcomes.

The 5-way argmax requires one failure-time bin to beat the no-failure class. It is
not the same decision as summing all four failure probabilities. Descriptive
same-training-set results, with no deployed decoder change:

| Decoder | Precision | Recall | F1 |
|---|---:|---:|---:|
| Current 5-way argmax | 84.3% | 34.7% | 49.2% |
| Sum failure probability >= 0.5 | 71.7% | 43.4% | 54.1% |
| Sum failure probability >= 0.4 | 57.9% | 55.7% | 56.8% |

The higher F1 costs many more false breaks. These are not held-out improvements,
and 0.4 is not a recommended production threshold. False splits and missed fracture
both change physics; choose an operating point only after validation and rollout.

For this use case, test a fracture-focused objective:

1. Binary break/no-break loss (weighted BCE versus alpha-balanced focal as an ablation),
   with class weights estimated from training data and calibration on validation data.
2. Conditional failure-time loss only on true failures; compare a discrete-time hazard
   formulation if survival/timing calibration is the goal. Separate occurrence from
   timing so uncertain timing does not suppress the total break decision.
3. Keep robust log-utilization supervision, including near-capacity examples. Add
   the required joint/force history prediction losses and dimensionally scaled
   motion terms with separately tuned coefficients.
4. Evaluate and then train for multi-frame stability: damage irreversibility,
   nonnegative dissipation and appropriate energy balance under external work,
   fragment connectivity/mass distribution, crack localization, and break timing.
   Connected-component agreement is initially an evaluation metric; it is not
   directly a differentiable loss. Avoid a zero-energy-change penalty for dissipative
   impacts/fracture. No penalty or weight is assumed optimal without an ablation.

Use one-step prediction plus controlled state perturbations, then short coupled
rollouts when the state update/contact interface is ready. Noise-based rollout
robustness and geometry/boundary-aware features are established in
[MeshGraphNets](https://arxiv.org/html/2010.03409v3); this project still needs to
validate those choices for irreversible fracture and its hybrid solver.

### Attention and scale

`PhysicsAttention` uses weights normalized over nodes for both pooling and
broadcast. For identical node embeddings, the broadcast contribution scales as
1/N. Measured per-node attention residual norms at N=8/80/800/10,000 are
13.5582/1.35582/0.135582/0.0108466. This is an architecture scaling concern, not a
benchmark of the full network or a claim that every nonuniform scene follows 1/N.
The [Transolver reference implementation](https://github.com/thuml/Transolver/blob/main/Physics_Attention.py)
normalizes assignments over slices, separately normalizes pooling, and reuses the
per-node slice assignments to broadcast. Test a corrected pooling/broadcast scheme
and graph-size invariance before expanding capacity or claiming 10k-body readiness.

Recommended experiment order: freeze grouped evaluation splits and the current
checkpoint as a baseline; fix feature/normalization/runtime contracts; compare
fracture-focused losses and decoding; validate accumulated state, topology, and
10k-body end-to-end behavior. The present checkpoint is retained for comparison.

## Remote access on this host

The current local server is exposed privately through Tailscale Serve at
`https://glavins-macbook-pro.tail3cda49.ts.net/` (HTTPS 443 → localhost 8791).
The Mac, visualizer process, and Tailscale connection must remain running. Serve
runs in the background; stop only this route with `tailscale serve --https=443 off`.
The page and `/api/index` were verified through that HTTPS address on this host;
connectivity from another device still depends on that device's tailnet access.


## V19 response normalization result

The controlled 32-wall comparison changed only positive-RMS flooring, with
identical source records, split, initialization, sampling visits, and 24,768
updates. Both checkpoints were selected at epoch 10 using validation fracture
F1. Only the active stored-energy cohort scale changed (93× effective squared
weight); this is not a new holdout or an independent geometry test.

| Held-out diagnostic | V18 default floor | V19 measured positive RMS |
|---|---:|---:|
| Bond fracture F1 | 0.52072 | 0.52596 |
| False-positive breaks | 1,060 | 1,372 |
| Separated-pair F1 on split frames | 0.47119 | 0.53892 |
| Fragment-count MAE on split frames | 18.69 | 33.54 |
| Largest-fragment mass-fraction MAE | 0.29011 | 0.27189 |
| World-position component RMSE, m | 0.005701 | 0.005585 |
| Active total-force component RMSE, N | 453.991 | 453.715 |
| Active stored-energy RMSE, J | 0.012000 | 0.011273 |

Exact-f64 zero baselines are 451.978 N and 0.008887 J. Retired cached-force
RMSE worsened 14,289.1 → 14,303.8 N and cached energy 2.53794 → 2.53867 J;
these are last responses, not forces continuing to act on retired interfaces.
The energy scaling defect is real, but fixing it has not produced adequate
physical response accuracy. Fragment oversegmentation also worsened. **No
promotion**; the default remains unchanged. Full report and source/checkpoint
hashes: `runs/response-scale-v19-2026-10-19/exact-response-audit.json`.

Checkpoint selection still optimizes fracture F1 alone. A future comparison
should select using validation-only response/topology criteria alongside F1;
test metrics above must not choose the epoch or loss weights. Neither candidate
has an independent coupled trajectory evaluation or a production World adapter.


The existing Tailscale viewer now also offers **Corrected oracle · experimental
response models** with both V18/V19 checkpoints on the 32-wall cohort. Earlier
datasets and views remain. The added viewer copies preserve event array/history
fingerprints and add replay provenance; training sources are unchanged. Direct
link: https://glavins-macbook-pro.tail3cda49.ts.net/?dataset=responses&event=v3_wall_19_s101&checkpoint=c74295d4&view=breaks&geometry=predicted
This particular wall is a training example, labeled as such. The new models are
experimental, not promoted. The house tab still has its earlier checkpoints.
Browser verification confirmed the dataset, model hash, first-break controls,
side-by-side render and explicit interval-reset limitation. Tailscale API also
returned both new checkpoints. A new dataset deep-link whitelist omission was
found in the browser check and corrected.
# V22 projectile observability and matched response experiment

V22 merges current projectile spin, full inertia, material, driven status and
crush history into V18's consistent damage/connectivity and 18-response model.
The added branch is zero-initialized, preserving all initial V18 outputs. Existing
V13 checkpoint parameter names and computations are preserved by the shared
mixin. Both recorded inference and the targetless native snapshot bridge support
V22. This still predicts an interval endpoint, not a full coupled World restart.

A paired diagnostic used the same wall19 geometry, mass, velocity and material,
changing only ram spin ±20 rad/s about Z. V18 sees identical tensors while the
oracle disagrees on 344 break labels. Its maximum possible F1 for the pair is
0.7772 even with perfect optimization. With 2,000 identical optimizer updates
per arm, unweighted BCE and threshold 0.5, V18 scored F1 0.68694, whereas V22
scored **0.99917 (600 TP, 1 FP, 0 FN)** and resolved all conflicting labels.
This is a two-example **training fit**, not generalization. The saved V22
diagnostic checkpoint passed exact native/recorded parity over 136 intervals:
8,024 input tensors and 2,584 output tensors, including house04's 120 intervals.

The controlled 32-wall run used the same split, sampling visits, seed, h48/p16,
24,768 updates and objectives as V18. It selected epoch10/threshold0.75 by
validation fracture F1. The result is **not promoted**:

| Held-out diagnostic | V18 | V22 |
|---|---:|---:|
| Fracture F1 | 0.520716 | 0.499078 |
| True / false / missed breaks | 1439 / 1060 / 1589 | 1218 / 635 / 1810 |
| Split-frame separated-pair F1 | 0.471189 | 0.376366 |
| Split-frame fragment-count MAE | 18.6923 | 11.9231 |
| Largest-fragment mass-fraction MAE | 0.290110 | 0.310531 |
| World endpoint position RMSE, m | 0.00570145 | 0.00578034 |
| Exact active force component RMSE, N | 453.991 | 451.825 |
| Exact active stored-energy RMSE, J | 0.0119996 | 0.0117662 |

Zero-response baselines are 451.978 N and 0.00888652 J. The tiny force advantage
over zero is insufficient for a useful stress solver; energy is worse than zero.
All decoded connectivity checks pass. Validation-selected physical alternatives
are saved separately and have not been selected using test outcomes. Source
hashes and exact sampling identity pass. Reports/checkpoints are under
`runs/projectile-response-v22-2026-10-19`.

The separately retained minimum-validation-force candidate (epoch12, threshold
0.55) was then selected by that declared criterion before candidate test scoring.
It reaches test force RMSE **449.472 N**, energy RMSE 0.0116135 J, fracture F1
0.511559 and split-pair F1 0.474829. Largest-fragment mass-fraction MAE is
0.288736 and fragment-count MAE 19.6154. This demonstrates a checkpoint-selection
tradeoff, not sufficient solver quality. It is not promoted or substituted for
the main checkpoint. The selection record and complete audit are in
`force-candidate-audit/` beneath the V22 run.

The evidence argues against solving this solely by tuning fracture loss weights.
First make state observable, then train balanced coverage of those states. The
remaining response regression uses normalized encoded residual MSE; test a
physical-unit response objective against zero and persistence, and score topology
and motion alongside fracture. Do not promote a loss from training fit alone.
Full application-phase, energy bookkeeping, contact coupling and long rollout
accuracy remain open. No 10,000-active-body whole-engine speed claim is supported.

The visualizer adds force, force-vector error, stored energy and energy-error
views for response models. Scales are in N/J and fixed across the replay, with
maximum incident active-bond values per piece. Exact f64 oracle endpoints are
used; retired caches are excluded with an explicitly oracle-derived endpoint
mask. Numerical diagnostics compare model, zero and persistence errors. Older
views/models remain available. V22 is an experimental choice, not a replacement.


## Independent destruction playback (2026-10-19)

Open `/rollout?event=house04` on the existing Stress Lab server. The house and
wall runs use independent persistent native worlds; the old `/` recorded-frame
comparison remains available. `scripts/view_learned.sh` detects the audited V25
catalog. The new viewer has shared playback/camera controls, independent fragment
colors, peak utilization and new-break modes, and same-bond break-time tolerances.

Artifacts: `runs/independent-endpoint-v25-2026-10-19/{house04,wall19}/` preserve
plans, frozen inference sources/checkpoints, native start/final snapshots, exact
model commands, reports and comparison bundles. Inference accepts a scene and
checkpoint, never an oracle recording; comparison reads the oracle afterward.

| Run | Oracle / model breaks | Oracle / model fragments | Oracle / model largest mass | Same-frame / any-frame F1 |
| --- | --- | --- | --- | --- |
| House04, V18 full128, 120 ms | 310 / 914 | 109 / 331 | 67.1% / 19.2% | 3.1% / 39.5% |
| Wall19, V22 wall, 133 ms | 308 / 255 | 12 / 34 | 38.6% / 91.4% | 60.7% / 60.7% |

House timing tolerance ±1/2/5 ms yields F1 16.5/21.6/27.9%. Allowing delay
helps but does not eliminate over-fragmentation. Mechanical energy ends at
121.5 kJ model vs 262.2 kJ oracle for the house; 94.8 vs 116.6 kJ for the wall.
This does not establish conservation: the adapter lacks predicted damping and
support work, and applies endpoint increments across contact substeps with
fracture only at interval boundaries. All runs recorded zero analytical bond
evaluations. They are diagnostic, not a production replacement or speed result.
The model and adapter still need separate error attribution; see `STEERING.md`.

## V26: endpoint adapter control and full128 projectile experiment

The independent V25 playback mixes substantial adapter error with model error.
A new control supplies exact oracle outputs from each live start state. Even with
perfect local endpoint/joint/response values, the first house frame slows the car
to 8.58 m/s instead of 13.43 m/s and ends at 172.6 kJ versus 266.0 kJ. The first wall frame
has 2.80 cm position-coordinate RMSE and 1.68 m/s velocity-coordinate RMSE despite
identical predicted/oracle fracture sets. Keep the view as an experiment; these
numbers cannot establish how good the network would be under correct integration.

A read-only dense oracle probe confirms 149 native steps in the first 1 ms house frame,
with 85 breaks spread across 39 steps and665 chunks changing fragment coordinate frames.
Its final physical state exactly matches the existing recorded oracle. Physical
chunk translational KE 13,706.7 J versus local hidden KE 204.36 J demonstrates that most
fragment motion is carried by rigid state at the endpoint. See STEERING for the
required impulse/transfer/timing contract gate before further rollout conclusions.

The matched full128 V22 projectile-response run completed in 1,676.37 s with 99,072 updates,
identical reference sampling visits and source hashes, zero connectivity contradictions
on 468,534 eligible test bond intervals. Selected epoch 24, threshold 0.65, checkpoint
`c698857f`. Test fracture F1 0.53934 versus V18 0.55258; precision 0.57707 versus 0.66346,
recall 0.50625 versus 0.47345. On 116 splitting frames, separated-pair F1 improves from 0.47170 to 
0.55549 and largest-mass-fraction MAE from 0.07170 to 0.06802, but fragment count MAE worsens
from 6.543 to 7.879 and exact partitions from 18 to 12. Active force RMSE from 1,099.76 to 1,095.83 N is a small
improvement; energy RMSE from 0.11979 to 0.11806 J still trails persistence at 0.11608 J. World position
RMSE worsens from 0.002934 to 0.003932 m. Mixed results; not promoted as an overall replacement.
`runs/projectile-impact128-v24-2026-10-19/matched-response-comparison.json` records
both complete summaries. Reused diagnostic holdout, not a fresh generalization test.

## V27: stress-phase integration parity and reference-energy witnesses

The new native stress-provider path reproduces the analytical solver on the actual
house04 and wall19 scenes when supplied analytical responses at each native
substep. House: 8 frames / 1,167 calls / 1,333,767 bond responses. Wall: 2 frames /
8,055 calls / 14,852,951 responses. Captured state, event sequences/times and the
stress energy ledger match exactly at every frame. Both native prefixes also match
the original V16 replay and exact joint/response history. This resolves the tested
application-phase mismatch without changing contact stepping or fragment transfer.
It does **not** convert the existing frame-end models into deployable substep models.

The controls explicitly perform analytical stress work in their provider; zero
engine fallback evaluations does not mean zero oracle work. The old V25 endpoint
rollout viewer remains an experiment with the previously documented adapter error.
No neural checkpoint was trained, promoted or shown to improve accuracy by V27.

Profiles attribute about 9.2% of house runtime and 41.7% of wall runtime to serial
bond response plus application. Contact work dominates the house prefix. These
instrumented controls ran alongside other work and are not timing benchmarks or
10k-active-body tests; replacing just local constitutive evaluations cannot alone
demonstrate a large full-engine speedup.

The new validator initially rejected signed oracle overshoot residuals. Preserving
those values exactly restores parity and exposes a separate reference issue:
40 negative house residuals total -0.00574193 J, with 32 beyond 1e-12 J affecting
four bonds during damage growth under compression. The worst is -0.00132142 J
(1.62% of that bond's stored energy). A read-only probe reproduces each negative
response directly from its saved joint input and deformation, and matches the
control's final state at declared f32/f64 precision. These witnesses are not a
constitutive fix or grounds for a claim of learned physical validity.

Artifacts: `runs/stress-provider-v27-2026-10-19/`, with frozen source/binary hashes,
preserved failed preflights, `signed-residual/audit.json`, and
`energy-witness/{audit.json,house04/witnesses.json}`. Validation on the current
source: 65 stress-ref library tests, four endpoint/observer tests and three provider
tests pass. The final provider also rejects partial healing and changes to fracture
history when fracture is disabled; `guarded/audit.json` confirms the same full real-scene
controls and original-recording parity on that final source.
Training/deployment work must retain the full speed-and-accuracy goal,
rather than treat exact oracle-provider parity as its completion.
