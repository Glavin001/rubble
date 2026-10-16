# Why rubble does not fall asleep (2026-10-16)

## The question

After the ball goes through the 7×7 veneer-house grid, rubble that looks
settled stays awake for 30 s and more. Nothing adds energy after the impact,
so every body that is still moving long after it should be treated as a bug.

## Tools

The tools are `fracture/sleep-diag.mjs`, driven by `headless-page.mjs`. In the
examples below, `[…]` marks optional arguments, and the `node` commands run from
`engine/crates/engine-webgpu/fracture`.

- **Where each awake body stands** (`sleepdiag`). It sorts every awake body by
  cause:

  | Cause | Meaning |
  |---|---|
  | `flying` | No contacts |
  | `opposed` | Two contacts with the same partner push in opposite directions |
  | `deep` | Overlaps by more than 1 cm |
  | `groundSlide` | Touches only static bodies |
  | `moving` | Moving among dynamic neighbours |
  | `held` | Slow for at least `sleepTime`, but kept awake by the island gate |
  | `settling` | Slow, with its timer still running |

  It also reports energy, overlap depth, and how many contact keys survive
  from one tick to the next.

  ```
  node headless.mjs grid sleepdiag=3600 [every=300]
  ```

- **One tick at a time** (`sleeptrace`). It records the pose, velocity and
  every contact (normal, overlap, λ, penalties, stick flag) of the fastest
  and spinniest bodies, or of the bodies you list.

  ```
  node headless.mjs grid sleeptrace=1200 ticks=20 [pick=3 | bodies=a,b]
  ```

- **What starts a burst** (`sleepwatch`). It reports the bodies that start
  moving first whenever a quiet scene starts moving again.

  ```
  node headless.mjs … sleepwatch=600 to=2400
  ```

- **Scene options:**
  - `awake=1` starts every body awake. Scenes start asleep, so without it an
    unhit house proves nothing.
  - `params={"flags":N}` overrides the solver flags. It is for measurement
    only.
  - The fast reproduction is one level-2 house with no shot, started awake:

    ```
    node headless.mjs grid columns=1 rows=1 shot=none awake=1 sleepdiag=2400
    ```

    Unbonded, it collapses under its own weight, then has to come to rest as
    a pile of 3,157 pieces. Each run takes about a minute.

## Root causes, with evidence

### 1. Pushing overlaps out became velocity, and loaded contacts sank by |λ|/k

**The mechanism.**
- AVBD's warm start decays every multiplier as λ ← αγλ (α = 0.95,
  γ = 0.999). It corrects only (1 − α) of the error left at the start of a
  step.
- At steady state, a contact under load therefore rests at an overlap of
  c₀ ≈ (1 − αγ)λ / ((1 − α)k) ≈ |λ|/k.
- The penalty k settles where its growth (iterations · β · |C|) balances its
  decay by γ. So the overlap grows like √|λ|:
  - about 3 mm under 50 N;
  - about 1 cm under 500 N.

**Measured.**
- **Creepers.** Bodies 3048 and 3155 crept at 1.5–2 cm/s, holding a constant
  1–3 mm overlap. A small, light compound (941, 39 g) had sunk 1 cm into its
  neighbour (943, 49 g) under 550 N of contact force.
- **Interlocking.** Once a piece sinks by about its own thickness, the convex
  children of two compounds interlock. The trace shows about 15 manifolds
  between one pair, with normals in every direction, some exactly opposed.
- **Two-tick oscillation.** The angular velocity of 941 flipped sign every
  tick: ω ≈ (11, ·, −12) → (−12, ·, 13) → (11, ·, −14) rad/s, with no
  physical motion behind it.
- **A brick inside a compound.** In the grid, a 2.9 kg brick (76076) sat
  4.5 cm deep inside a 175 kg six-hull compound (75907). Two of the compound's
  children pushed it in opposite directions with λ up to 4,000. Its spin
  swung 73 → 0.7 → 26 → 0.15 → 55 rad/s from tick to tick.
- **Push-out became velocity.** Velocity is (x − x⁻)/h, so the push-out of
  every overlap is counted as speed. For example, 5% of a 3 mm overlap per
  tick is about 0.02 m/s, close to the 0.03 m/s sleep threshold.

**The fix: post-stabilization (flag 4096).** It follows the AVBD authors' 2D
reference, `avbd-demo2d` `Solver::step` with `postStabilize = true`:

1. The warm start keeps λ whole, and only the penalty decays by γ.
2. The iterations run with α = 1. Error present at the start of the step is
   not corrected; contacts only stop it growing.
3. The velocity is taken after those iterations.
4. One extra primal pass with α = 0, and no dual update, removes the whole
   position error. It moves the bodies but adds no velocity.

This is the same idea as Box2D's relax pass and Bullet's split impulse. Our 3D
port of the reference had left it out.

**Result on the one-house pile.**
- Energy is 0.0 J from 4 s on.
- No `opposed` or `deep` bodies remain from 6 s on.
- The deepest overlap falls from 9 cm to 1.5 cm.

### 2. Static friction lost its anchors every tick

**The mechanism.**
- A contact keeps its friction anchors only while it "sticks". Our test
  required the tangential residual at the end of the step to be under 10 µm.
- With λ decayed by αγ, a truly static contact settles at a residual of
  (1 − αγ)·λT/kT. For body 3155, that is 0.05 × 56 N / 5160 ≈ 0.5 mm, fifty
  times over the tolerance.
- So the anchors were dropped every tick, and static friction degraded into a
  slow slide.
- The predicted creep, (1 − αγ)·λ / (iterations·kT·h) ≈ 0.016 m/s, matches the
  measured 0.015–0.02 m/s.

**The fix: Coulomb stick (flag 128), from `avbd-demo2d`.** A contact sticks
while its unclamped friction is inside the cone and its anchor offset is under
1 cm.

Post-stabilization removes the λ decay that caused this, so with flag 4096 the
original test works as well.

### 3. Light pieces slid too far

**The mechanism.** Sliding friction was applied too weakly to light pieces:
- The duals held the right force, μ|λN|.
- But each primal step was damped by the stick penalty kT. kT is left over
  from an earlier sticking phase and decays by only 0.1% per tick, while
  m/h² for a 34 g piece is just 490.

**Measured.** Body 143075 (34 g) slowed at 0.97 m/s². Pieces of 2.9 kg and
9.9 kg slowed at 5.0–5.1 m/s², against μg = 4.9 m/s².

**The fix: sliding Hessian (flag 1024).** A sliding row uses the secant
stiffness μ|fN|/|C_T|, IPC's lagged friction Hessian. The force itself is
unchanged.

### 4. The sleep test held small pieces to a far stricter standard

**The mechanism.** The angular threshold, 0.05 rad/s, ignores body size:
- For a 3 cm piece it means 1.5 mm/s of surface motion. That is 20 times
  stricter than the 3 cm/s linear threshold.
- One-tick friction flickers of 0.1–0.4 rad/s on lightly loaded corners
  therefore kept resetting the sleep timers.

**What mature engines do.** They all scale rotation by body size:
- Box2D v3 uses |v| + |ω|·r_max;
- Jolt uses point velocities;
- PhysX uses mass-normalised kinetic energy.

**The fix: point-speed test (flag 8192).** A body is slow when
|v| + |ω|·r_max < sleepLin, where r_max is its farthest collider point from
the centre of mass.

### 5. Impacts create deep overlaps

**The mechanism.**
- Contacts are detected at the start of the step with no margin (flags 65).
  A pair closing at v reaches up to v·h deep before it gets a contact: 17 cm
  at 20 m/s.
- The 2 mm speculative margin (flag 32) is far smaller than that, and
  measured no better.
- **The fix: look-ahead contacts (flag 16384).** They are sized to how fast
  each pair is closing, as PhysX speculative CCD and Box2D/Jolt speculative
  distance do:
  - each awake collider's bounds are swept by
    (|v| + |ω|·r)·h + |g|·h², but only when that motion exceeds the 2 mm
    contact depth;
  - the narrowphase and hull SAT keep a pair up to its relative reach;
  - a separated point is kept only if the relative normal velocity at that
    point, (v_A + ω_A×r_A − v_B − ω_B×r_B)·n, closes the gap within the step;
  - only a real gap (C₀ beyond the 2 mm margin) is kept whole as clearance.

  Each of those choices replaced a version that failed on the grid:
  - **gap-mode contacts** (rest at touch) made bricks rock on their hovering
    corners;
  - **reach for every awake body** gave resting bodies speculative contacts;
  - **an isotropic spin bound** made struck bricks spinning at 240 rad/s
    contact everything within 20 cm, overflowing the 64-colour palette.

### 6. The island gate

The island gate keeps a connected pile awake while any one of its bodies
moves. Box2D, PhysX and Jolt behave the same way, so this is correct. In the
grid it multiplied a few thousand movers into more than 100,000 awake bodies.
The fix is to remove the movers, not to change the gate.

## Experiments

The flags are added to the scene's 65. Flags 256 and 2048 are rejected.

| Flag | Change |
|---|---|
| 128 | Coulomb stick |
| 256 | Tangential penalty floored at the mass stiffness (**rejected**) |
| 1024 | Sliding Hessian |
| 2048 | Penalty floored by load (**rejected**) |
| 4096 | Post-stabilization |
| 8192 | Point-speed sleep test |
| 16384 | Look-ahead contacts sized to closing speed |

**Defaults.** The fracture scenes now default to 4096+128+8192:
- the veneer houses use 12481;
- the cuboid houses use 12513.

Look-ahead (16384) is off by default.

### One house, unhit, started awake

The pile has 3,157 bodies. Values are at 20 s.

| Flags | Awake | Moving | Opposed | Kinetic energy | Notes |
|---|---|---|---|---|---|
| 65 (baseline) | 2,467 | 62 | 6 | 0.1–7 J bursts | Creeps forever |
| +128 | 2,376 | 25 | 10 | 0.1 J | |
| +128+256 | 2,032 | 3 | 6 | 0 | 256 keeps the house standing (75 kJ of potential energy) |
| +2048 | | | | **2.4 MJ** | Explodes: λ/slop makes the dual step too large |
| +4096 | 1,788 | 0–245 | 0 | 0 from 4 s | Bodies fall asleep, then get woken again in waves |
| +4096+128+256+1024 | 2,343 | 4 | 0 | 150 J bursts | 256 is unstable with 4096 |
| +4096+128 | 2,731 | 16 | 0 | 0 | |
| +4096+1024 | 2,494 | 3 | 0 | 0 | |
| +8192 | 2,117 | 50 | 12 | 0.3 J | |
| **+4096+8192** | **0 (all asleep from 18 s)** | 0 | 0 | 0 | |
| +4096+1024+8192 | 1,720 | 0 | 0 | 0 | |

### 7×7 grid, ball shot, 30 s

The grid has 154,759 bodies.

| Flags | Moving at 20 s | Opposed at 20 s | Moving at 30 s | Awake at 30 s |
|---|---|---|---|---|
| 65 | 5,047 | 858 | 5,674 | 135,734 |
| +4096+8192 | 1,147 | 136 | 688 | 116,378 |
| +4096+1024+8192 | 641 | 90 | 2,417 | 114,522 |
| **+4096+128+8192** | **294** | 119 | 350 | 97,949 |
| +4096+128+8192+16384 | 256 | 92 | 285 | 124,114 |

Look-ahead cuts impact interpenetration about threefold:

| Contacts, 20–30 s | Without look-ahead | With look-ahead |
|---|---|---|
| More than 1 cm deep | about 3,000 | about 900 |
| More than 5 cm deep | 6–9 | 1–2 |

It does not get the grid to sleep. From about 15 s, **the grid is one
connected island of about 124,000 bodies**: debris piles bridge the houses, as
the awake-island statistics in `sleepdiag` show. Within it, the gate
(8 rounds, so each blocker holds its 8-hop neighbourhood) keeps everything
near a non-candidate awake. At 30 s there are about 5,500 non-candidates:
- about 285 moving bodies, mostly small pieces tumbling at 4–16 rad/s;
- about 55 jammed bodies;
- about 4,000–5,000 bodies whose timers were reset in the last second.

On the one-house pile, look-ahead with the defaults has 0–3 movers from
12 s, and its deepest overlap is 4–5 mm instead of 30 mm.

With 4096+128+8192:
- the movers fall steadily, to 171 at 25 s;
- the peak awake count drops from 136k to 113k.

About 93,000 slow bodies are still held awake at 30 s by islands that contain a
jammed or moving piece:
- about 120 pieces stay `opposed` (jammed during the impact);
- about 3,000 contacts stay more than 1 cm deep (deepest about 7 cm).

Those overlaps form before post-stabilization can act, in the tick where fast
pieces pass into each other (cause 5).

The ball is still rolling at 11.5 m/s at 30 s. A rigid ball on rigid ground
without rolling resistance never stops, but it touches only the ground.

The houses have no bonds, so a house that debris wakes collapses under its own
weight. In the grid, about 37 of the 49 houses wake within 7.5 s of the shot,
and each collapse then takes as long as the one-house pile.

## Cost

Measured on the grid, mean ms per tick over ticks 420–540 (3.5–4.5 s after
the shot), two runs each:

| Flags | ms per tick | Awake | µs per awake body |
|---|---|---|---|
| 65 | 27.2, 26.5 | 120.8k, 117.8k | 0.225 |
| +4096+128+8192 | 25.4, 25.8 | 102.7k, 105.8k | 0.247 |

- **The extra pass costs about 10% per awake body.**
- **The ticks were still faster overall**, because fewer bodies are awake by
  then.

## Still open

1. **Whether to enable look-ahead by default.** It is more accurate at impact
   but does not change the grid's sleep outcome yet.
2. **The last movers in a giant island.** Trace the ~285 tumbling small pieces
   and the timer resets (wake waves) in the 124k-body island.
3. **A slider check for flag 1024.** Confirm on a scene with light debris
   sliding on the ground before adopting it.
4. **Bonds.** Unbonded houses collapse completely once woken. Bonds, or the
   stress solver, change what "after the shot" means for the houses next
   door.
