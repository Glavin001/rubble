# Sleep work: decisions pending for you

This file tracks choices that change physics behaviour or trade memory or
cost, so they are not mine to make alone. Each entry gives what was measured,
the options, and my recommendation. The evidence is in
`sleep-investigation-2026-10-16.md`.

All flags below are `params.flags` bits on the fracture scenes. Turn one on
from the headless runner with `params={"flags":N}`, and compare runs with
`sleepdiag`.

## 1. Light mortar slivers in bonded scenes (mass ratio)

**What was measured**
- In the bonded suburb at level 1, about 1,500 bonded pieces keep moving after
  the shot, so about 2,750 bodies stay awake from 12 s on.
- The movers are 13–50 g mortar slivers bonded between 1–3 kg bricks. One
  13 g sliver carries about 800 N through five bonds, with anchors about 3 cm
  from its centre. That is about 24 N·m on I ≈ 10⁻⁶ kg·m², so any imbalance
  within one solver iteration flips it.
- A mass ratio of about 1:200 is past what iterative solvers handle. Jolt and
  PhysX documentation advise ratios under about 10–100.
- Already done: bond geometric stiffness, the effective-mass cap on bond rows,
  and the rotation limit. The divergence is gone, but the slivers still
  jitter.

**Options** (all used by mature engines, and all of them change sliver physics)
- **A. Inertia armature (MuJoCo `armature`).** Add a minimum rotational
  inertia to light bodies. Slivers would turn more sluggishly than real mortar.
- **B. Mass scaling in bond/contact rows (PhysX `invMassScale`).** The solver
  treats the light body as heavier in those rows. Bonds then load slightly
  incorrectly.
- **C. Merge slivers into a neighbour in the asset** (Shardwright export). This
  is lossless at run time and changes the fracture geometry.
- **D. Leave it to the stress solver**, which replaces these bonds.

**My recommendation:** C, or D if the stress solver lands soon. A and B both
alter physics everywhere.

## 2. Rolling resistance as a default (flag 2097152)

**What was measured**
- Without it, a rolling body never stops on rigid ground. In the grid, the
  200 t ball keeps rolling at 11 m/s.
- With μr = 0.02 it comes to rest in about two minutes.
- In the 7×7 grid it left more pieces jammed: 2,059–9,145 awake at 30 s with
  it, against 158–3,000 without. That is three runs; runs vary.

**Options**
- Keep it off (the current default).
- Turn it on for the fracture scenes.
- Make μr a per-material property, as Box2D's `rollingResistance` does.

**My recommendation:** keep it off by default, and add a per-material
coefficient when materials need it.

## 3. Bond-solver energy at breakage

Bonded pieces still left the bonded scenes airborne (about 400–500 at 30 s
in the suburb, most of them hanging from bonds rather than flying). Breaking
bonds can also release pieces fast. This overlaps the stress-solver work.

**Decision:** keep investing in the legacy bonds, or wait for the stress
solver?
