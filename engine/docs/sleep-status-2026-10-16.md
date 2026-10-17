# Sleep work: status (2026-10-16)

**Where it lives**
- Branch `feat/convex-hulls`, worktree
  the `feat/convex-hulls` worktree.
- Latest commit: `610585c`. Nothing is pushed.
- Detail and evidence: `sleep-investigation-2026-10-16.md`.
- Your pending decisions: `sleep-decisions-pending.md`.

## Where it stands

The awake bodies at 30 s, for the default flags of each stage:

| Scene | Before (flags 65) | Now (defaults 6222017) |
|---|---|---|
| One unhit house, started awake (3,157 bodies) | 2,467, never sleeps | **0 from about 8 s** |
| Bonded house started awake (level 1, 567 bodies) | 561, spins to 240 rad/s | **0 from about 8 s** |
| 7×7 grid with the ball (154,759 bodies) | 135,734 | **1: only the rolling ball, from 25 s** (deterministic and normal runs) |
| Bonded suburb, level 1 (27,233 bodies) | 14,391, growing | **about 2,190**, flat (legacy bond solver) |

The remaining awake bodies:
- **7×7 grid:** 2 islands of about 2,400 bodies, each held by 5–7 slow movers
  (bricks still moving at about 5 cm/s on a few contacts). The 200 t ball keeps
  rolling at about 11 m/s; it touches only the ground.
- **Bonded suburb:** the awake bodies are almost the whole of the 5 struck
  houses (529–563 of 567 bodies each); no other house has an awake body. Most
  are ordinary 0.5–5 kg pine members (1,412 of 2,182); pieces under 60 g are
  only 136, in proportion to their share of the asset. 335 have no contact at
  all: they hang or spin on bonds. The same suburb **without bonds** is all
  asleep by 30 s except the ball, so the cause is the legacy bond solver, not
  the asset or the sleep logic. (An earlier version of this note blamed
  "mortar slivers"; the light pieces are pine offcuts, and they are not the
  cause.)

## Changes

Each change is a `params.flags` bit. The fracture-scene defaults are 6222017
for the veneer houses and 6222049 for the cuboid houses. A flag marked off is
opt-in.

### On by default

| Flag | Change | Source | Worked? | Evidence |
|---|---|---|---|---|
| 4096 | Post-stabilization, contacts only | avbd-demo2d | Yes, the biggest single win | One house: KE 0 from 4 s; deep overlaps 9 cm → 1.5 cm. Bonds had to be excluded (they exploded). |
| 128 | Coulomb stick | avbd-demo2d | Yes, on the grid | Movers at 20 s: 1,147 → 294 |
| 8192 | Point-speed sleep test | Box2D v3 | Yes | With 4096, the one house sleeps |
| 16384 | Look-ahead contacts sized to closing speed, linear and spin, with a relative-sweep candidate test | PhysX speculative CCD, Box2D, Jolt | Yes | Grid deep (>1 cm) contacts 3,000 → 900. Needed four fixes: rest semantics, a per-point closing-velocity filter, a relative sweep, and a spin bound. |
| 32768 | Displacement sleep test: a body counts as moving only if both its low-pass pose and its velocity integral Σv·h moved more than sleepLin·sleepTime this window | Jolt sleep spheres; Box2D, Jolt and PhysX velocity tests | Yes, the final key | Ignores velocity blips that post-stabilization undoes, and push-out that is not velocity. The grid went from about 4,750 to 1 awake (the ball). |
| 131072 | Deferred wake | Box2D (woken islands keep their contacts) | Yes | Stops the wake cascade (1,522 → 0 wakes on one pile) |
| 262144 | Resting-pair carry: sleeping pairs keep their λ | Box2D; Rust host `ENGINE_REST=carry` | Yes | One house asleep by 10 s; no cold-start sag on waking |
| 524288 | Bond geometric stiffness | AVBD paper Sec. 3.5, avbd-demo3d | Yes | Bond iteration stopped diverging; the bonded house sleeps by 8 s |
| 1048576 | Rotation limit, 0.25π per step | Box2D, Jolt | Partly | Bounds spin and the look-ahead pad; light pieces no longer reach 240 rad/s |
| 4194304 | Push-out limit, 0.1 m/s | Box2D, PhysX | Yes | Jammed pieces at 30 s: about 40 → 0–4 |

### Off by default

| Flag | Change | Source | Worked? | Evidence |
|---|---|---|---|---|
| 1024 | Sliding-friction secant Hessian | IPC | Unproven | Mixed on the grid |
| 65536 | Stabilization damping | PhysX | No | No measurable gain |
| 2097152 | Rolling resistance, μr 0.02, stiffness bounded by inertia | Box2D v3, Bullet | Stops the ball in about 2 minutes, but worse elsewhere | Grid at 30 s: 2,059–9,145 awake vs 158–3,000 without |
| 256 | Tangential penalty mass floor | — | No, rejected | Unstable with 4096 |
| 2048 | Load-based penalty floor | — | No, rejected | Exploded (2.4 MJ) |

| 8388608 | Rotational armature, I ≥ 0.1·m·r_max² | MuJoCo | No, removed | Bonded suburb: 3,275 awake vs 2,191. Removed: it read colliders inside the primal solve. |

### Host-side

- **Bond effective-mass cap** in `scaleBondSprings`: no measurable change on
  its own.

## Performance

- **Post-stabilization (4096): measured.** About +10% GPU time per awake
  body, from one extra primal pass (0.247 vs 0.225 µs per awake body per tick
  at 3.5–4.5 s on the grid). Ticks were still faster overall, because fewer
  bodies were awake: 25.4–25.8 ms vs 26.5–27.2 ms.
- **Look-ahead: roughly measured.** The early isotropic version cost about
  +15–35% per tick at impact. That measurement was noisy, taken while the lab
  was also using the GPU. Since then, the relative-sweep test cut candidates
  from about 34k to 6–9k on the small bonded scene (about 2× the count without
  look-ahead).
- **Everything else: not measured.** The displacement test, rolling
  resistance and the stabilization damping add a loop over each awake body's
  colliders.
- **Re-measure needed.** All the per-tick figures should be re-measured on a
  quiet GPU with `node headless.mjs grid warm=420 measure=120 profile=1`,
  comparing `params={"flags":65}` with the defaults.

## Tools added

All are in `sleep-diag.mjs`, driven by `headless-page.mjs`.

| Mode | What it does |
|---|---|
| `sleepdiag` | Classifies every awake body by cause, with island, drift and deep-contact statistics |
| `sleeptrace` (with `pick`, `bodies` or `settling=1`) | Per-tick trace of chosen bodies, including bond state and displacement |
| `sleepwatch` | Reports the bodies that start each burst of motion |
| `wakewatch` | Reports who woke whom |
| `stepdebug` | Per-tick counters, with NaN checks on failure |

Other options:
- `awake=1` starts every body awake;
- `level=` sets the fracture level;
- `params=` overrides the solver flags, including in the bond preload.

## Open, and what I would try next

1. **The grid's last slow movers.** Trace them (`sleeptrace=3600 pick=3`);
   my last attempt produced no output.
2. **Bonded houses stay awake.** The legacy bond solver keeps struck bonded
   houses moving (pending-decisions items 1 and 3).
3. **Rolling resistance as a default, and the push-out cap value.** Pending
   decisions items 2 and 4.
4. **Run variance.** The grid varies a lot between runs, by up to 3× in awake
   count at 30 s, so decisions should rest on several runs per flag set.
