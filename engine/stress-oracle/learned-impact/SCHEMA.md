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
`f32` / `i32` only. Prefer this over `.npz` in Rust; the Python loader can
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
