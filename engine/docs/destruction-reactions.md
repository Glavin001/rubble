# Destruction reactions

Agreed with the user on 2026-10-25. One strike, blast or collapse can produce every reaction a
real structure shows, and each is decided by the engine and reported as an event:

| # | Reaction | Decided by | Event |
|---|---|---|---|
| 1 | **Crushed**: the chunk's material is comminuted and it stops being a rigid chunk | Compressive overstress of the chunk, integrated to damage | `ChunkCrushed` |
| 2 | **Knocked off**: bonds break under the contact or impact force | Threshold predictor / breaks (exists) | `BondsBroken`, `Split` |
| 3 | **Blast loading**: overpressure on exposed faces | Friedlander face pressures (oracle `loads.rs`), now in `World` | the breaks it causes |
| 4 | **Support loss**: the weight left without a load path breaks bonds, pieces tip | `support_loss.rs` load flow and tipping | `BondsBroken`, `Split` |
| 5 | **Dynamic amplification** of a sudden loss | explicit, opt-in | — |
| 6 | **Debris loads** on the structure below | contact-to-load path (exists) | — |

Rubble decides and reports; the host draws dust. Simulating dust particles is optional.

## Rules every part follows

- **Forward-only** (AGENTS.md): a strike's crushes are predicted with its breaks at the end of the
  frame before it and applied before the engine steps. No rewind, no re-solve.
- **Capability flags.** Each reaction is an opt-in switch (`SolverConfig`, `Methods`,
  `SimFeatures` or the `DestructibleDef`), off by default so every existing golden and digest is
  unchanged. Not part of `blast_parity`.
- **No silent defaults.** A crush-enabled material without crush parameters is an error, never a
  guessed value.
- **No lossy surprises.** A crushed chunk's mass is **conserved by default**: it leaves the
  structure as a few small rigid fragments carrying its mass and momentum, so its weight still
  loads the floor below. Removing the mass entirely (dust) is an explicit `CrushMass::Remove`.
- **Energy.** The impactor pays the crush energy (`dD · crushEnergy · V`) out of its kinetic
  energy, before the step, as `J = M (v − sqrt(v² − 2 dE / M))`, capped at `M v` (it can stop a
  striker, never bounce it).

## 1. Chunk crushing

The model is the one PhysX's Blast stress extension and vibe-land ship
(`extStressCrushStep`; PhysX `blast/blast-stress-solver/docs/chunk-crushing.md`), written fresh
for rubble:

```
sigma = (1/V) sum sym(branch (x) force)      contact and bond tractions, no body forces
p = -tr(sigma)/3 (compression positive)      q = sqrt(1.5 s:s)
crushing needs p > 0
excess = max(q - (cohesion + frictionSlope p), p - capPressure)
D += excess^2 dt / (crushViscosity crushEnergy)       crushed at D >= 1
```

Material parameters are per material (`crush` block), derived from the material's own
compressive strength with cited sources (vibe-land `structures/town-kit/src/materials.mjs`
`CRUSH`: EN 1996-1-1, EN 1992-1-1, Bond 1961 comminution energy, CEB-FIP MC90 rate factor).

`ChunkCrushed { destructible, chunk, mass, volume, pose, linear_velocity (at the chunk),
angular_velocity, peak_pressure, peak_deviator, cause: Impact | Blast | Contact | Load }`.

## Tests and oracle

Each part has unit tests from the law's definition and physical invariants (a structure under
its own weight never crushes; tension never crushes; damage is quadratic in overstress; mass and
momentum are conserved in `Conserve`; the impactor's energy drops by the crush energy), and the
crush-wall and reference-building-crush scenes run end to end with every reaction counted.

## 4. Support loss (`DestructibleDef::support_loss`, threshold model, opt-in)

`engine-world/src/support_loss.rs`, hooked after each frame's breaks in `threshold.rs`. The
pieces those breaks touched carry their weight down their intact bonds to their anchors: a flow,
breadth first from the supports, each chunk handing its weight and what reached it from above to
the neighbours one bond nearer an anchor in proportion to those bonds' compressive capacities
(`compressive_strength * area`). A bond whose load exceeds its capacity breaks with the next
frame (`GpuFracture::break_bonds_sparse`), so a collapse cascades forward a frame at a time.
Tipping: a piece whose centre of mass, along gravity, is outside the hull of its loaded support
bonds lets go of the members standing on those supports (their bonds into it break). The flow
conserves weight (the load reaching the anchors is the piece's weight); it has no bending, no
tension and no redundancy beyond the sharing. An error under the Bonds model. C ABI:
`RbDestructibleDef::support_loss`.

## 5. Dynamic amplification (`DestructibleDef::support_loss_dynamic_factor`, opt-in)

The flow is quasi-static. A suddenly applied load on an undamped elastic system peaks at twice
its static effect (single degree of freedom: `1 + exp(-zeta pi / sqrt(1 - zeta^2))`), and
progressive-collapse guidance applies a dynamic increase factor of up to 2 to the gravity load
over a removed column in linear static analysis (GSA 2003 *Progressive Collapse Analysis and
Design Guidelines*, `2(DL + 0.25 LL)`; DoD UFC 4-023-03 *Design of Buildings to Resist
Progressive Collapse*, load increase factor `Omega_LD`). With factor `k`, a piece whose supports
changed this frame (the piece it was part of before held anchors it no longer holds; a chip
knocked off changes nothing) breaks a bond whose flow load times `k` exceeds its capacity.
Range 1 to 2, anything else an error; `k != 1` needs `support_loss`. Default 1: today's
behaviour. Tipping takes no factor. C ABI: `RbDestructibleDef::support_loss_dynamic_factor`.

`k` multiplies the whole post-loss static demand, as the codes' linear static procedure does.
The stress oracle (stress-ref, explicit mode) amplifies the *change* of load:

| Oracle scene | before | static after | peak | (peak - before) / (after - before) | peak / after |
|---|---|---|---|---|---|
| b4 sudden support loss (zeta 0.02), reaction | 29.4 N | 39.2 N | 48.5 N | 1.94 | 1.24 |
| b8 sudden column removal (elastic), column 0 axial | 20.2 kN | 33.2 kN | 54.5 kN | 2.63 | 1.64 |
| b8 sudden column removal (elastic), column 2 axial | 31.3 kN | 48.2 kN | 74.2 kN | 2.54 | 1.54 |

(b8's static-after is the mean of its last 50 ms.) So `k = 2` on the full demand is
conservative for a member that already carried load before the loss (b4: 1.24x, b8: 1.5-1.6x of
the static-after load) and exact for one that carried none. The factor is not tuned to these.
