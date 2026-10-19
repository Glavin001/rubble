# Learned-impact event schema (`learned-impact/1`)

On-disk contract for oracle-recorded impact events used to train the residual
MGN-T impact model (`engine/docs/learned-stress-solver-plan.md` §4).
Quasi-static forces are **inputs**; the network learns the dynamic residual.

## Event directory

```
<event_id>/
  manifest.json     # metadata (this version)
  catalog.json      # array name → dtype, shape, relative path
  static/           # per-structure features (written once)
  frames/
    000000/         # zero-padded frame index
      catalog.json
      *.bin
    000001/
    …
```

Arrays are **little-endian** raw binaries (`.bin`), row-major, no header.
The original catalogs use `f32` / `i32` only; the separately versioned exact
joint-history sidecar below uses `f64` / `i32`. Prefer raw arrays over `.npz` in Rust; the Python loader can
export npz for convenience.

## `manifest.json`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | must be `"learned-impact/1"` |
| `event_id` | string | directory basename recommended |
| `structure_type` | string | hold-out key (e.g. `house_1story`) |
| `load_type` | string | hold-out key (e.g. `ram_12ms`) |
| `n_frames` | u32 | |
| `n_chunks` | u32 | active chunks `N` |
| `n_bonds` | u32 | live undirected bonds `E` |
| `K` | u32 | sub-frame time bins; **4** |
| `dt` | f64 | frame length (s) |
| `seed` | u64 | |
| `notes` | string | optional |

## Graph

- **Nodes:** `N` active chunks.
- **Edges:** `E` undirected live bonds, stored **once** with endpoints `(i,j)`.
  Message-passing CSR is built as bidirectional at load (`2E` directed edges).
- Vectors in **bond-local** or chunk-relative frames (no world pose).

## Joint-state pack (`JOINT_STATE_DIM = 22`)

Order matches `stress_ref::joint::JointState` floats (mode is a separate label):

| idx | field |
|---|---|
| 0–1 | `damage`, `crush` |
| 2–3 | `kappa`, `kappa_c` |
| 4–5 | `ductility`, `ductility_c` (∞ / unset → `0`; finite clamped to ≤ 1e3) |
| 6 | `fatigue` |
| 7–12 | `plastic.lin[3]`, `plastic.ang[3]` |
| 13 | `rebar_plastic` |
| 14–15 | `rebar_slip[2]` |
| 16 | `rebar_work` |
| 17 | `rebar_broken` (0/1) |
| 18–19 | `strain_rate`, `governing_stress` |
| 20–21 | `dissipated`, `utilization` |

## Cached bond responses (`learned-bond-response/1`)

New recordings declare `context.bond_response` with path
`bond_response/manifest.json`. This sidecar preserves **18 float64 fields per
bond** at `n_frames + 1` snapshots, with the same ordered `bond_keys` and times
as exact joint history:

| Columns | Quantity | Units |
|---|---|---|
| 0–5 | Total local force x/y/z and moment x/y/z, including damping | N, N m |
| 6–11 | Elastic local force x/y/z and moment x/y/z | N, N m |
| 12 | Cached recoverable elastic energy | J |
| 13–16 | Tension, shear, normal compression, compression measures | Pa |
| 17 | Compressive axial force | N |

The manifest explicitly lists these fields. Each snapshot has `time`,
`values: NNNNNN.f64` (`[E,18]`, little-endian) and `flags: NNNNNN.i32`
(`[E,2]`, ordered alive/connected). All response values must be finite; signed
stress measures and signed zero are preserved. Flags must agree with the exact
joint sidecar. Python also checks that casting total force to float32 matches
the original `bond_force_prev` / `bond_force_end` arrays bit for bit.

**These are cached responses.** A retired bond retains its last force, elastic
force and stored energy. Its values are not forces acting after retirement.
Alive flags must accompany training metrics and visualization. The engine uses
elastic force and stored energy for fracture-to-contact handover; neither can
be reconstructed safely from total force alone. This sidecar does not capture
all World bookkeeping, contact caches, runtime geometry or integration stages,
and does not by itself make a restart or learned solver adapter complete.

`load_event(..., bond_response_history=True)` opts into the raw sidecar.
`ResponseHistory.frame_start(i)` produces 18 signed-log float32 features from
snapshot `i` only. `event_batches` exposes `bond_response_encoded` and the
separate target `bond_response_encoded_end`, plus start/end alive masks. The
raw float64 sidecar remains authoritative. Existing model inputs are unchanged
unless the caller explicitly requests the response fields.

Fresh enrichment preserves existing joint and response sidecar files byte for
byte as well as legacy arrays and replay fields. A modern generator refuses to
resume a dataset without the response contract; use a fresh output directory
or its original frozen binary. Running older collection jobs remain valid for
their declared contracts and do not silently acquire these additional fields.

## Exact joint history (`learned-joint-history/1`)

New oracle recordings additionally declare `context.joint_history` with format
`learned-joint-history/1` and path `joint_history/manifest.json`. This sidecar
preserves all `JointState` fields and bond `alive` / `connected` flags; it does
not replace the 22-component legacy arrays or change existing checkpoints.
Old records without this declaration remain readable by legacy models.

The sidecar manifest names `fields`, `flags`, `bond_keys` (exactly the context's
unique ordered bond keys), and `snapshots`. There are `n_frames + 1` snapshots:
snapshot `i` is the input for interval `i`, and `i+1` is its target. Each entry
contains `time`, `values: NNNNNN.f64`, and `flags: NNNNNN.i32`. Files are raw
little-endian, row-major, with exact byte counts:

- `values`: `[E,21]` float64. Ordered fields: damage, crush, kappa, kappa_c,
  ductility, ductility_c, fatigue, plastic_x/y/z/rx/ry/rz, rebar_plastic,
  rebar_slip_x/y, rebar_work, strain_rate, governing_stress, dissipated, utilization.
- `flags`: `[E,4]` int32. Ordered fields: rebar_broken, mode, alive, connected.
  Flags are 0/1; mode is 0 none, 1 tension, 2 shear, 3 compression, 4 buckling.
  A connected bond must be alive.

Only ductility / ductility_c may contain positive infinity (the unset sentinel).
Finite values are neither clamped nor converted to float32. NaN, negative
infinity, and infinity in other fields are invalid. Rust unpacking preserves
float64 bits, including signed zero. This fixes the legacy pack's inability to
distinguish unset ductility from zero and to retain finite values above 1,000.

Python validates any declared sidecar, including alignment, flags, lengths and
timestamps. `load_event(..., exact_joint_history=True)` opts into access;
`JointHistory.frame_start(i)` returns 31 finite network features: 21 signed-log
values with explicit scales, two unset masks, a rebar flag, five mode indicators,
and two topology flags. It never reads snapshot `i+1` as an input. These encoded
features are not a lossless engine checkpoint; the raw sidecar is the exact data.
Existing V5–V7 models still use the legacy pack and do not consume these features.

`learned_impact_enrich SOURCE_EVENT NEW_OUTPUT` can recover the sidecar only by
re-running the saved scene. It refuses an existing output and accepts a result
only after all original arrays (including preview and QS) match byte-for-byte,
all original replay frame fields match exactly, and original context inputs are
preserved. Older preview context centers may have JSON parser rounding; these
are retained unchanged, while regenerated initial world centers must equal the
original replay exactly. Only `enrichment-verification.json` with `verified: true`
marks a successful enrichment. Failed output directories are diagnostic data,
never training data. Scene parsing enables `serde_json/float_roundtrip` so saved
float64 inputs reproduce the generated scene. Resume with a new recorder rejects
old partial data lacking this sidecar; use its original binary or a fresh output.

This is joint history, not a complete World restart format or a closed-loop
learned solver. Contact state, rigid-body state and topology integration still
need an explicit runtime contract and trajectory validation.

## Static arrays (`static/`)

| name | dtype | shape |
|---|---|---|
| `chunk_mass` | f32 | `[N]` |
| `chunk_inertia` | f32 | `[N,3]` |
| `chunk_half_extents` | f32 | `[N,3]` |
| `chunk_material_id` | i32 | `[N]` |
| `chunk_support` | i32 | `[N]` (0 none, 1 fixed, 2 pinned) |
| `chunk_level` | i32 | `[N]` |
| `chunk_coarse_parent` | i32 | `[N]` (−1 if none) |
| `bond_area` | f32 | `[E]` |
| `bond_width` | f32 | `[E,2]` |
| `bond_offset` | f32 | `[E,3]` |
| `bond_stiffness` | f32 | `[E,6]` |
| `bond_strength` | f32 | `[E,3]` (tension, shear, compression) |
| `bond_fracture_energy` | f32 | `[E,3]` |
| `bond_rebar` | f32 | `[E,5]` (zeros if none) |
| `bond_weibull` | f32 | `[E]` |
| `bond_buckling_length` | f32 | `[E]` (0 if none) |
| `bond_endpoints` | i32 | `[E,2]` |

## Per-frame arrays (`frames/NNNNNN/`)

**Inputs**

| name | dtype | shape |
|---|---|---|
| `chunk_state` | f32 | `[N,12]` (`u,θ,v,ω`) |
| `chunk_load_bins` | f32 | `[N,K,6]` external force∥torque per bin (scripted + contact), local frame; **not** gravity. Mean force over the bin (`impulse / (dt/K)`). |
| `chunk_contact` | i32 | `[N]` (0/1) — chunk saw contact force this frame |
| `chunk_gravity_local` | f32 | `[N,3]` gravity in the cluster/body frame |
| `bond_joint_state` | f32 | `[E,22]` |
| `bond_force_prev` | f32 | `[E,6]` |
| `bond_qs_force` | f32 | `[E,6]` quasi-static generalised force |

**Labels**

| name | dtype | shape |
|---|---|---|
| `bond_fail_class` | i32 | `[E]` (0 = none, 1…K = bin) |
| `bond_peak_util` | f32 | `[E]` |
| `bond_fail_mode` | i32 | `[E]` (0 none, 1 tension, 2 shear, 3 compression, 4 buckling, 5 rebar) |
| `bond_joint_state_end` | f32 | `[E,22]` |
| `bond_force_end` | f32 | `[E,6]` |
| `chunk_state_delta` | f32 | `[N,12]` end − start |

## Constants

```text
SCHEMA_VERSION = "learned-impact/1"
K = 4
CHUNK_STATE_DIM = 12
LOCAL6_DIM = 6
JOINT_STATE_DIM = 22
```

## Optional native boundary kinematics (`native-boundary-kinematics/1`)

A separate observation captured by `RuntimeGraph::capture_with_kinematics` before
native frame stepping. It does not change `learned-native-start/1` snapshots.
JSON fields: `format`, exact frame-boundary `time`, and `values` with one f64 row
per fixed RuntimeGraph bond key. Each row contains generalized displacement
`[t1,t2,normal,rotation_t1,rotation_t2,rotation_normal]` followed by its six rates.
Translation is in metres, rotation in radians, and rates per second. The native
configured linear or corotational geometry map defines those quantities.
Retired bonds have zero placeholders; use snapshot alive flags/runtime bond IDs,
never infer participation from values. Current-state capture is read-only.

`boundary_kinematics.audit` verifies graph identity, byte-identical full snapshots,
source associations and exact first-phase f64 kinematics. Future phase data is
used only for this audit. Training attachment reads the selected boundary file,
checks all recorded hashes, selects runtime-alive IDs and converts to float32.
This optional input has no analytic stress, future contact or fracture labels.
The experimental V30 encoder fits kinematics scales on training inputs only.


### V31 native trajectory wire format (`learned-native-trajectory/2`)

`runtime_trajectory.rs` and `trajectory_response.forecast` define the complete
experimental contract. Every live bond slot appears exactly once, with uniform
normalized-time knots: history[21], response[18], work_rate[3] (J/s), mode,
material failure/rebar failure times, encoded history anchor and two ductility
initialization times. Physical and signed-log history encodings are explicit.
The learned exporter uses signed-log knots: interpolate first, then invert using
the history scales. Do not interpolate decoded physical endpoints; that differs
from the training function. Preserve the exact f64 initial anchor when unchanged.

Native phases query force/response at their start and history at their end.
Work integrates the piecewise-linear rate across the entire phase, including
knot boundaries. Signed overshoot residual stays signed; dissipated and damping
work are nonnegative. Dissipated history accumulates integrated work. Irreversible
history cannot heal, unset ductility initializes at a continuous crossing time,
and complete material/rebar failures latch. Native splitting/connectivity follows
these states; complete material damage alone does not necessarily break rebar.
Malformed batches, clocks, slots and non-finite forecasts reject before application.
The independent driver saves original inputs, each prediction, native snapshots
and reports; it does not reset to oracle states. Snapshot capture rejects values
outside finite local float32 feature range. A failed terminal capture is a failed
run, never a complete trajectory. Numerical constraints do not prove physical
consistency. A fixed interval forecast does not react to new within-frame contacts.


### V32 oracle branches from model-generated Worlds

`learned_impact_oracle_branch SAVED_NATIVE_RUN NEW_OUTPUT PREFIX_FRAMES` requires
original graph bytes, all prefix predictions/snapshots and one further learned
prediction/endpoint for continuation parity. It reconstructs via native replay,
not snapshot restoration. Label output begins only after graph, prefix and cloned
continuation parity. `snapshots/000000.json` is the model-generated branch state;
`000001.json` is its oracle continuation. Their absolute times can be nonzero.
The existing stress-phase format and audit apply. `boundary-kinematics.json`
contains causal observations, `learned-control.json` retains the paired learned
endpoint, and `oracle-events.json` records only events after the branch. The report
identifies the parent run and frame. A branch is supervised counterfactual data,
not an independently learned run; preserve the parent scenario's partition.

The separate channel probe deliberately mixes oracle and predicted outputs for
causal diagnosis and then runs an oracle tail. Its explicit oracle call counts
must never be reported as zero-fallback learned inference. Mixed responses are
not admitted as training targets. Neither diagnostic is published as a successful
learned-model comparison.


### V33 corrective training-branch audits

`branch_dataset.audit_branch` requires an unambiguous `train` parent in the
verified source registry. It checks original scenario hashes, graph/initial-state
identity of the learned parent, completed zero-fallback execution, exact branch
and learned-control bytes, prefix/absolute clock consistency, full phase history,
and first-phase/current-boundary kinematic identity. Audit metadata includes
`source_kind: learned_state_oracle_branch`, the parent event/checkpoint/prefix,
parent input hashes and source oracle audit hash. It deliberately leaves
`original_world_replay_exact` false for the branch: these states were produced by
the learned model, not the original oracle trajectory. Existing oracle dataset
loader guards are unchanged. The separately verified branch manifest is not an
automatic admission to an existing training run; an explicit loader/experiment
must consume it and preserve parent scenario grouping.


### V34 explicit corrective-corpus consumption

`trajectory_response --branch-manifest` uses `branch_dataset.load_records` to
re-audit the full frozen branch provenance and labels. Duplicate parent-event/
prefix pairs reject. Shared `phase_record` and `complete_record` builders produce
the same native phase targets and causal feature contracts as original records,
while retaining `parent_event` and `source_kind`. The fixed experiment requires
one corrective branch per four original training frames, replaces one quarter
of updates, and rotates replacement positions. It preserves original-only scales
and the original validation/test partitions. An absent flag preserves the old
training path. V34 explicitly consumed the frozen V33 training-branch manifest;
its original creation-time description remains preserved for provenance.

Separate wall/blast phase and boundary views retain original intervals, full native
phase observations and source replay identity. Their existing phase schema and
boundary audit apply unchanged. Their pilot manifest contains only inherited
training entries and must be combined explicitly with validation/test data before
any new training experiment; it is not a generalization benchmark by itself.
