# Character controllers and scene queries

A Rubble character is a kinematic mover solved on the GPU inside the physics
step, in the style of Jolt's `CharacterVirtual`
(`slang/kernels/character_step.slang`, pool at binding 32,
`slang/physics/character_layout.slang` ↔ `engine-schema/src/character.rs`).
It implements the character audit's Phases 0–3. Phase 4 (a dynamic-body NPC
mode) is not implemented.

## Using it (native wgpu host)

```rust
use engine_schema::character::{CharacterDesc, CharacterInput};
let h = sim.add_character(&CharacterDesc::default(), [0.0, 1.0, 0.0]);
sim.set_character_input(h, CharacterInput { velocity: [3.0, 0.0, 0.0], jump_seq: 0 });
sim.step()?;
let s = sim.character_state(h);           // position, velocity, ground state and body…
for e in sim.poll_motion_events() { … }   // Landed, SupportLost, Crushed, projectile hits
```

Jumping: bump `jump_seq`; the character jumps once per new value while on the
ground. The browser host has the same calls (`world-host.mjs`:
`addCharacter`, `setCharacterInput`, `readMotion`), read back asynchronously.

`CharacterDesc` (defaults): radius 0.3 m, half height 0.6 m, skin 0.02 m,
max slope 45°, step up and step down 0.35 m, jump 5 m/s, air control 4/s,
mass 80 kg, push ratio 1, max recovery 0.05 m per step, gravity scale 1,
query mask 0 (all), push weight on.

## The step

`char_move` runs one workgroup per character after the broadphase (it walks
the same grid), before the narrowphase:

1. **Velocity.** On the ground: the input along the ground plane plus the
   ground's velocity at the contact point (platforms carry the character). In
   the air: gravity and air control. A jump adds the ground's upward speed.
2. **Gather** every collider near the capsule swept along its motion
   (`sceneWalk`: uniform, split or hierarchical grid, rest tables, plane,
   large and fast lists). Dynamic bodies lighter than `mass × push_ratio` do
   not block; the character's inner capsule pushes them in the solve.
3. **Planes.** One contact plane per collider at the start pose, moving with
   its body's step motion.
4. **Solve** the displacement against the blocking planes (slide, crease,
   stop; cyclic projection), keeping `skin` from each. Overlap is removed at
   most `max_recovery` per step, so a character pinned by debris works free
   without being launched.
5. **Verify** with a capsule cast along the solved move (obstacles moving
   with their bodies). A cast hit cuts the move.
6. **Step up** when walking into a blocker: up, forward, down; kept only if it
   lands on walkable ground (the face under the contact, not the contact
   normal, decides, so a stair nosing does not read as steep).
7. **Stick to the floor** when walking down steps or slopes, up to
   `step_down`.
8. **Ground probe**: on ground (walkable), on steep ground, or in the air.
   Events: `Landed`, `SupportLost`, `Crushed` (a dynamic body pressed into a
   grounded character).
9. The inner kinematic capsule (a body and collider of its own) gets the
   velocity that takes it to the new pose, so dynamic bodies collide with the
   character in the same solve.

`char_post` (one lane, after the solve) reads the ground's velocity for the
next step and puts the character's weight on a dynamic support as a velocity
change on that body, then wakes it.

Scene queries (`set_queries`, `query_results`) and projectiles
(`spawn_projectile`, [ccd.md](ccd.md) row 5) use the same grid walk.

## Scene queries

| Query | `SceneQuery` |
|---|---|
| Ray | `radius = 0` |
| Sphere cast | `radius > 0`, `half_height = 0` |
| Upright capsule cast | `radius > 0`, `half_height > 0` |

Rays test hulls exactly (slab test over the hull planes); shape casts use a GJK
cast, with a second cast from behind when the shape starts inside a collider.
Filters: an excluded body and a category mask. `full_scan` walks every collider
instead of the grid: it is the oracle the grid walk is tested against.

## Limits

| Limit | Value | Over it |
|---|---|---|
| Characters, queries, projectiles | 64, 256, 256 (`CharacterCapacity`) | host assert |
| Candidates per character | 192 | `OVERFLOW_CHAR_CANDIDATES` (16384) |
| Grid cells per query walk | 4096 | the walk falls back to a full scan |

## Deviations from the audit's plan

- The character's weight on a dynamic support is a velocity change in
  `char_post`, not a two-body constraint row in the solve.
- Step-up and stick-to-floor casts see obstacles where the step leaves them,
  not moving during the cast (the main move and verify casts do move them).
- `compact()` renumbers characters, projectiles and queued impulses; a
  character's inner capsule must be kept.

## Tests (`engine-webgpu/tests/character.rs`)

| Scenario | Result |
|---|---|
| Walk on flat ground at 3 m/s | 6.00 m in 2 s, never leaves the ground |
| Five 0.3 m stairs | climbs them with no frames in the air or on steep ground |
| 30° slope, no input | holds (drift 0) |
| 55° slope | slides |
| Descending elevator | no hover gap, grounded every frame |
| Moving platform | carried; jumps 1.21 m (ballistic 1.25 m) and gets `Landed` |
| Light crate, heavy crate, plank | pushes the light one, blocked by the heavy one, tips the plank (800 N) |
| CCD rocket at the capsule | stops at the character |
| `compact()` | inner capsule and ground renumbered; keeps standing |
| Block pressed into the head | `Crushed`; a block resting on the capsule is not |

The browser demo is `crates/engine-webgpu/motion.html?scene=character`.
