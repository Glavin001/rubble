# Oracle quality profiles: one trusted algorithm, dialled down

Proposal, 2026-10-12. Not implemented yet.

## Idea

stress-ref stays the double-precision ground truth. Alongside it, the same code can run
at a deliberately lower quality, set by named, switchable knobs. Each knob keeps the
algorithm and only coarsens a quality parameter.

The GPU real-time solver must make compromises. Today its error against the full oracle
mixes two things: the approximations we chose, and bugs. Profiles separate them:

- **Level 1, the accuracy cost.** The full oracle against the oracle at a profile. This
  measures what each compromise costs, on the oracle's own trusted code.
- **Level 2, the implementation.** The GPU against the oracle at the matching profile, at
  a tight tolerance. The GPU then targets a reference whose quality we know.

A profile is a named set of knob values, like `Methods::proven()`, serialized with the
scene. The gate and the guard accept `--profile`.

## Knobs

| Knob | Full | Dialled | Notes |
|---|---|---|---|
| `precision` | f64 | f32 | See below |
| `stress_courant_safety`, `courant_safety` | converged (0.25) | 0.35–0.5 | Already exist; the benchmark runs at 0.35/0.5 |
| Static solve tolerance and iterations | oracle defaults | looser | `StaticOptions`; must stay inside a certificate (a converged solve is still a solution) |
| Contact geometry detail | full hulls | reduced vertices | e.g. the recorder's inscribed `hull_points_reduced`; the volume lost is reported |
| Cohort merge | exact stack | merge below f32 energy rounding | Only once proven in the oracle (LAYER_CONTACT_PORT.md §2.4) |
| Later | — | — | Any approximation the GPU needs: per-island multirate, sleeping certificates, etc. |

Every knob:

- is a named, switchable approximation;
- is measured at level 1 on the fidelity suite (`scripts/fidelity.py`, the gate scenes)
  before the GPU may use it;
- keeps the physical invariants (TESTING.md). A dial-down may lose accuracy but must
  never create energy.

## Single precision

The oracle has no float abstraction: `f64` appears 1,164 times in about 19k lines.

**Plan.** Add one type alias (`pub type Real = f64`) and a cargo feature (`single`) that
builds the crate with `Real = f32`. This is a second build, not generics, so it is
mechanical and low risk. The math types (`Vec3`, `Mat3`, `Quat`) follow `Real`.

### What it buys

- A reference for what f32 storage and arithmetic cost, on trusted code.
- Every tolerance written for f64 (`1e-12`, `1e-300`, etc.) fails or misbehaves
  visibly. Those are exactly the places the GPU needs a principled f32 treatment, as it
  already has in places (`world_diff`, `rotation_err`, the statics floor). The f32 oracle
  should adopt the same named techniques, so the GPU and the oracle share them.

### Caveats

- **Not bit-identical to the GPU.** Operation order, FMA contraction, parallel
  reductions and the GPU's compensated tricks differ. Level 2 compares statistically on
  chaotic scenes, and tightly on deterministic micro cases.
- **Conflicts.** It touches every oracle file, so it needs a quiet window in oracle
  development, or it must be landed in slices: first the alias (no behaviour change,
  verified bit-identical), then the feature.

## Order

1. TESTING.md invariant checks (in progress).
2. The layer-contact port (LAYER_CONTACT_PORT.md, in progress).
3. Knob infrastructure and profiles in stress-ref, then `precision = f32`.
4. `stress-gpu-gate --profile` and the two-level gating in the guard.
