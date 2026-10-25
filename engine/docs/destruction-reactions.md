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

### As built (threshold mode, 2026-10-25)

**Switch.** `DestructibleDef::crush: Option<Crush>` (`StrengthModel::Threshold` only; `None`
by default). `Crush { mass: CrushMass, materials, max_per_step }`:

- `CrushMass::Conserve { fragments, budget }` (default 4 fragments, 256 reserved fragment
  bodies) or `CrushMass::Remove`.
- `materials`: `CrushMaterial` by the stress import's chunk material name (a pack import's
  per-node copies such as `concrete#mass12` match their base name), supplying or replacing
  the import's own.
- `max_per_step`: crushes one step reports at most (0: 256).

A material without parameters never crushes (stress-ref `Material::crush: None`). Turning
crushing on where no chunk can crush is an error naming the materials. Exceeding the
fragment budget or the per-step crushes fails the step with an error naming the limit.

**Parameters.**
- stress-ref `crush::CrushParams` on `Material::crush`; `crush::presets` is the cited table
  (vibe-land `CRUSH`: brick masonry, gypsum, concrete, glass, stone masonry, softwood, roof
  tile, slate, with `bond_energy` and `mc90_viscosity`). `presets::for_material` maps the
  calibrated materials (concrete, brick, stone, glass, timber; steel and mortar never crush).
- ScenePack v3 import: `materials[].crush` (camelCase), chosen per chunk by `nodes[].m`.
  - A crush block in a v1/v2 pack is an error, as is a missing required field.
  - Without crush blocks the import is unchanged.
  - `strainRateExponent` gets an import note: rate hardening is not modelled in v1.
- `strength_scale` scales bond strengths, not crush parameters.

**Where it runs.** `stress-gpu` `fracture.slang` (`GpuFracture`), which both `World` and the
lab's `ThresholdWorld` run; `crush.slang` holds the law, mirrored in stress-ref
`crush::crush_step_f32` and compared on the GPU (`tests/crush_parity.rs`). The resident
`GpuWorld`'s own threshold kernels (`threshold.slang`) do not crush.

**The stress estimate.** The threshold mode solves no bond forces, so the virial is built from
equilibrium. A chunk's external forces at their points are its frame's contact loads, as
`fr_loads` defines them:
- the sustained forces and the carried pulses' means, for the frame;
- each new impact at its Hertz peak force, for its Hertz duration (capped at the frame);
- for a predicted strike, the predictor's Hertz peak force `m_eff v / tau` at the contact,
  for `tau`.

They are balanced by reactions through the chunk's intact bonds, shared by bond area at the
bond centroids. Since `sym()` is bilinear, those reactions add `sym(r_A (x) -F)`, where `r_A`
is the bonds' area-weighted centroid relative to the chunk and `F` the net external force. A
chunk with no intact bond has only its inertia as a reaction: the net force is balanced at the
loads' own (|f|-weighted) centroid, so one contact or one strike never crushes a lone chunk.
Body forces (gravity, inertia) are excluded, as the law says.

Consequences:
- Loads through bonds alone (a column under a building's weight) are not estimated: there is
  no bond solve. Cause `Load` is reserved, as is `Blast`.
- Area share counts a block's side bonds too. On the free wall's bottom row the estimate came
  to the same order as `W / 3A`: measured 1.06 W/(3A) at peak, settling included.
  A projected share (`A |n . F|`) would load only the bonds facing the load; not built.

**Forward only.** Everything runs in the fracture's encode after the engine's step and before
the next one; nothing is rewound.
1. `fr_crush_loads` integrates the frame's contact loads, before the frame's bond breaks.
2. The predicted strike (`fr_crush_entry`, `fr_crush_nearest`, `fr_strike_crush`) loads the
   struck body's entry chunk. It runs before `fr_strike_force`, so the struck chunk is
   crushed before the engine's step meets it.
3. A crushed chunk's bonds break with the frame's other breaks (`fr_crush_bonds`). The split
   then leaves it alone on a body.
4. After the split, `fr_crush_apply` parks that body out of the world and writes the event.
   With `Conserve` it also turns `fragments` reserved bodies into boxes:
   - a grid of the chunk's box, halved along its longest cell until there are enough;
   - centred on its centre of mass and shrunk to at most its volume;
   - each with its share of the mass, at the chunk-point velocity `v + w x r`.
   So the fragments carry its mass and momentum exactly. Ranks come from a scan, so event
   order and fragment slots are deterministic.

**Energy.** The strike's damage increment books `dD crushEnergy V` to its shot (`fr_strike_crush`).
Each shot pays its booked total once, as `J = M (v - sqrt(v^2 - 2 dE / M))` capped at `M v`,
against its speed (`fr_crush_bill`). The struck dynamic bodies receive `J` by their share of
the energy, at the contact (`fr_crush_push`), so momentum is kept. Contact-path crushes
charge nobody: debris, sustained pressure and impacts the predictor did not see are already
inside the engine's step.

**v1 limits.**
- A strike crushes only its entry chunk; the neighbours its ball also covers crush from the
  next frame's contact loads, uncharged.
- Strain-rate hardening is omitted.
- The lab (`ThresholdWorld::new_with_crush`, Fracture Lab `Crush`) removes crushed chunks;
  it keeps no fragments.

**A finding, left as it was.** The existing strike prediction picks its entry chunk by
matching a distance recomputed in a second kernel bit for bit (`fr_strike_entry`,
`fr_strike_nearest`). Where the compiler fuses the two differently, no chunk matches and the
strike is dropped. On the crush wall (Metal) no predicted bond broke at all; the contact
broke them a frame later. Crushing picks its entry from stored distances instead. The bond
path was not changed, so its behaviour stays bit for bit.

## Tests and oracle

Crushing's, as built:
- stress-ref `crush::tests`: the law. Uniaxial yield at fc, tension never crushes, damage
  quadratic in overstress, the cap, the f32 mirror, the impulse, the presets.
- `tests/crush_pack.rs`: the v3 import.
- stress-gpu `tests/crush_parity.rs`: the Slang law against the f32 mirror.
- engine-world `tests/crush.rs`:
  - the free wall under its own weight never crushes, and its bottom row's pressure matches
    `W / 3A`;
  - the pack's 4.8 t ball crushes the struck block with cause Impact and pays exactly its
    crush energy;
  - `Conserve` keeps the free wall's mass and the system's momentum in zero g;
  - `Remove` takes the chunk out;
  - crushing that crushes nothing changes no body bit for bit;
  - missing parameters and the wrong model are errors.
- stress-rubble `tests/threshold_crush.rs`: the lab's world.

Each part has unit tests from the law's definition and physical invariants (a structure under
its own weight never crushes; tension never crushes; damage is quadratic in overstress; mass and
momentum are conserved in `Conserve`; the impactor's energy drops by the crush energy), and the
crush-wall and reference-building-crush scenes run end to end with every reaction counted.
