# Sleep work: decisions pending for you

This file tracks choices that change physics behaviour or trade memory or
cost, so they are not mine to make alone. Each entry gives what was measured,
the options, and my recommendation. The evidence is in
`sleep-investigation-2026-10-16.md`.

All flags below are `params.flags` bits on the fracture scenes. Turn one on
from the headless runner with `params={"flags":N}`, and compare runs with
`sleepdiag`.

## 1. Struck bonded houses never sleep (legacy bond solver)

**What was measured** (bonded suburb, level 1, defaults 6222017, at 30 s)
- About 2,180 awake, flat from 15 s on. Every one is in one of the 5 struck
  houses, and each of those houses is almost entirely awake (529–563 of 567
  bodies). The other 43 houses are asleep.
- By fragment kind (awake / moving): pine 0.5–5 kg 1,412 / 1,278; pine
  60–500 g 176 / 151; brick >50 kg 161 / 64; pine <60 g 136 / 108. The light
  pieces are no more likely to be awake than the heavy ones.
- 335 awake bodies have no contact: they hang or spin on bonds (up to
  90 rad/s).
- **The same suburb without bonds is all asleep by 30 s** except the ball.
- The asset checks out: at level 1 every fragment's mass, volume, centre of
  mass and inertia agree with its collision hulls (volume within 18%, centre
  within 8.5 cm, inertia within 0.93–1.32×).

**Correction.** An earlier version of this entry blamed 13–50 g "mortar
slivers" and a 1:200 mass ratio. The light pieces are pine offcuts from cell
agglomeration (34 under 60 g per house, the lightest 8 g), and the breakdown
above shows they are not the cause. Merging them in Shardwright would not fix
this.

**Tried in the bond solver:** bond geometric stiffness (fixed the divergence,
on by default), the effective-mass cap, the rotation limit, and inertia
armature (flag 8388608, since removed; worse: 3,275 awake). Post-stabilization off: 2,178.

**Options**
- **A. Fix the legacy bond solver** for damaged, partly broken houses
  (bond rows on hanging pieces, breakage energy).
- **B. Leave it to the stress solver**, which replaces these bonds.

**My recommendation:** B. This is the same question as item 3.

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

**Update with the final sleep test** (7×7 grid):
- With rolling resistance (8319169): every body except the ball is asleep by
  35 s, and the ball slows to 7.2 m/s by 40 s, so it stops about 1.5 minutes
  later.
- Without it (the default): every body except the ball is asleep by 25 s, and
  the ball keeps rolling forever.

**My recommendation:** if you want truly everything asleep, turn it on and
accept about 10 s slower settling. Otherwise keep it off.

## 3. Bond-solver energy at breakage

Bonded pieces still left the bonded scenes airborne (about 400–500 at 30 s
in the suburb, most of them hanging from bonds rather than flying). Breaking
bonds can also release pieces fast. This overlaps the stress-solver work.

**Decision:** keep investing in the legacy bonds, or wait for the stress
solver?

## 4. Push-out speed limit value (flag 4194304, `PUSH_MAX_SPEED`)

**What was measured**
- The limit caps how fast overlap is pushed out. Box2D's
  `contactPushMaxVelocity` defaults to 3 m/s; PhysX's
  `maxDepenetrationVelocity` is unbounded by default and usually set to
  1–10 m/s.
- Jammed compound pieces (which cannot separate) squirm at the cap speed, so a
  lower cap helps them sleep. The cost is slower recovery from genuine deep
  overlaps: 5 cm takes 0.5 s at 0.1 m/s.

**7×7 grid at 30 s**

| Cap | Awake | Moving | Jammed | Resetting |
|---|---|---|---|---|
| 0.5 m/s | ~3,520 | 0 | 1 | 2 |
| 0.1 m/s | 2,540–5,160 | 1–4 | 0–4 | 0–20 |

At 0.1 m/s the grid plateaus at about 4,750 awake to 45 s, held by 5–7 slow
movers.

**Decision:** the cap value. It is 0.1 m/s now.
