# Demo scenarios: what viewers rewatch, and how our takes score

The source is a review of 51 popular YouTube destruction clips, collected 2026-10-23:
- 27 vehicle clips: BeamNG, Wreckfest, FlatOut, MotorStorm;
- 24 structural clips: Red Faction: Guerrilla, The Finals, Teardown, Instruments of
  Destruction, Battlefield 4.

They were picked by replay-heatmap peaks and view counts. This page is the brief for
every demo take (`engine/crates/engine-webgpu/takes/`); the full plan was approved in
session.

## What viewers rewatch

Replay peaks sit on three beats:

1. **Anticipation.** You can see it coming: the load, the lean, the sag, the vehicle's
   approach.
2. **Big pieces first.** A storey, a span or a slab moves as one recognizable mass before
   it breaks up. Instant dust and confetti lose the moment.
3. **A secondary impact.** The falling mass hits the ground, a car or another building,
   and breaks again. Most of the rewatching happens here.

Patterns, strongest first:

| Id | Pattern | Strongest clips |
|---|---|---|
| S1 | A coherent mass leans, then falls (topple) | BF4 skyscraper (2.0M views); The Finals street topple; Teardown high-rise |
| S2 | A bridge or span fails at a joint and drops, with vehicles on it | 8 of 51 clips across both collections; RFG spans, The Finals skybridges, BeamNG bridge pileup |
| S3 | A huge impactor (an aircraft) cuts a building; the upper block tilts away | Teardown aircraft ×3 |
| S4 | A heavy ball or sphere pulps a structure | Teardown red sphere; Instruments of Destruction ball (753k) |
| S5 | Progressive collapse climbs; a core survives | Teardown concrete frame |
| S6 | "Still hanging on": a slab or skybridge sags before it lets go | The Finals official update demo (1.0M) |
| S7 | Debris density and dust that hides the transition | almost every structural clip |
| V1 | Parts come off: doors, wheels, panels | BeamNG, MotorStorm, FlatOut |
| V2 | Visible crumple | BeamNG front crush, wall fold |
| V3 | Pileup and chain reaction | every vehicle compilation |
| V4 | Heavy versus light | MotorStorm cab crush, Wreckfest RV, BeamNG bus |
| V5 | Vehicle into a structure | MotorStorm nose-first wall; BeamNG bridge |

**Where Rubble can beat them:** vehicles and structures in one solver (no reference game
breaks both), the stress X-ray (why it fell), scale (10k+ bodies), and live, repeatable
runs in a browser.

## Our takes (2026-10-24), scored on those beats

19 lab takes and 2 destruction-page takes, recorded in headless Chrome with `record-takes.mjs
--page lab --check --repeat` (and `--page destruction`). Every stress-mode take passes the
checks and repeats frame for frame.

| Take | Patterns | Works | Gaps |
|---|---|---|---|
| `wy_frame10_ball` | S4 S5 S7 | 50 t ball into the corner columns: the collapse climbs, the core and upper storeys stand (483 bonds) | — |
| `wy_frame10_xray` | S4 + X-ray | The bond graph drops out as bonds fail; one colour per body | — |
| `wy_round_tower_sphere` | S4 | 15 m sphere: the floors drop, the core stands (645) | — |
| `bo_viaduct_pier` | S2 S1 | Both drop-in spans hinge down whole and break on the road (459) | Stress solver only (threshold breaks a few bonds at the pier) |
| `bo_viaduct_traffic` | S2 V3 V4 | The spans carry 5 cars and a bus down onto a queue of 2 cars and a bus (1,237) | The traffic's weight breaks the spans up sooner |
| `topple_detonate` | S1 | The charges fire; it leans out as one mass for about 4 s and lands across the street, in Chrome as natively (310) | It lands as one long piece, not storey-sized slabs. No cars on the street yet |
| `sb_skybridge` | S2 S6 | One charge: the walkway hangs from its west ledge, tears out and lands on the plaza in big sections; the towers stand (150) | Threshold needs both charges |
| `rr_storefront_car_bus` | V5 V1 V2 | A bonded car through the shop window crumples inside; the bus drives into the next shop with the fascia on its roof (442) | — |
| `rr_apartment_car_bus` | V5 | Both hit the apartment block's ground floor | Neither gets through: the brick infill is two 1.15 t panels a bay |
| `rr_storefront_car` | V5 | Ball stand-in: a local break | Superseded by `rr_storefront_car_bus` |
| `rr_drive_storefront` | V5 V4 | Driven, not launched: rubble's raycast sedan and six-wheel truck, in the world API (`World`, the `Threshold` model), accelerate on their own wheels from 30 m out. The sedan breaks the window at about 50 km/h and drives into the shop; the truck's front wheels meet the brick stallriser under the fourth shop's window and it breaks through slewed about 40° (136) | The vehicles are rigid: no crumple yet (soft bodies, plan Phase D). A raycast wheel cannot climb the stallriser |
| `bo_drive_viaduct` | S2 V3 V5 | Driven traffic on the viaduct: four sedans, two six-wheel trucks and two on the road below, cruising. The hinge pier's charge cuts the drop-in spans loose; they swing down about their seats with four vehicles on them, the traffic behind drives into the gap, and a span lands on the lower road's truck (292) | Under the threshold model the ball alone cannot drop the spans (a span bonded at one end hangs there): the charge cuts what the stress solver would break |
| `sc_ram_raid`, `sc_bridge_out` (destruction page) | as above | The same two scenarios on the destruction page (`destruction/`: engine-world-wasm steps the world, the lab's renderer draws it, chunks in their materials' colours); the same breaks, frame for frame | Flat shading, no shadows or dust yet (the lab's stress renderer has them) |
| `ft_apartment_airliner` | S3 | A bonded 35 t jet at 70 m/s punches through five storeys of the middle bays; engines and nose fly off; it falls back out (1,148) | — |
| `ft_glass_tower_airliner` | S3 | The nose breaks up in the curtain wall and floors (710) | It stops in the first bays and slides down the facade; no upper block fails (threshold has no stress solve) |
| `vh_car_barrier_crash`, `vh_bus_barrier_crash` | V1 V2 | Into a rigid barrier: the front comes apart, wheels come off, the cabin holds (45, 39) | Wheels do not roll: the vehicles slide on them |
| `lab_house_*` | S4 S7 | The veneer house in threshold and stress modes | — |
| `lab_cuboid_grid_ball` | S4 S7 | Scale: about 11k awake bodies | Rubble's own mode does not repeat exactly at this scale |

**Look:** true material colours (brick, tinted glass, concrete, a red body shell), sun
shadows, a filmic tone curve, and dust and chips from real impacts only. The dust repeats
exactly; vehicles, glass and metal shed chips, not dust.

## Known engine limits the demos hit

- **Strike loads in blast_parity.** A strike loads the whole anchored structure, so heavy
  hits break bonds in every storey instead of along their path (the apartment block, the
  glass tower).
- **Static friction at the threshold profile's 4 rigid iterations.** Heavy bodies creep
  where they should hold: the topple tower's base, the landed tower sliding. The topple
  scene works around it with a kerb.
- **No stress solve in threshold.** Loads only reach chunks in contact, so nothing fails
  across a cut (the glass tower's upper block) and a landing tower does not break into
  storeys.
- **Rubble's own mode repeats on a house but not at the 7 × 7 grid's scale.**

## Next, in order of impact

1. **Parked cars under the topple tower and on the skybridge plaza** (S1 + V4): the
   vehicles module is ready.
2. **The airliner through the glass tower** (S3): a lighter curtain wall, or the stress
   profile once its strikes stay local, so the upper block tilts away.
3. **Driven vehicles: done for the storefront and the viaduct** (`rr_drive_storefront`,
   `bo_drive_viaduct`). Threshold fracture runs in `World` (D1b); a chassis's boxes strike
   exactly; wheels load what they stand on (W5); charges cut bonds between steps. One host,
   `rubble::scenario`, runs them for the lab (stress-web), the destruction page
   (engine-world-wasm) and the native tests. Next:
   - drives for the apartment block;
   - a vehicle you steer from the page;
   - the destruction page's look: the lab's shadows and dust, and Safari (an Asyncify build);
   - soft-body hero cars that crumple (plan Phase D).
4. **A reel per pattern**, for sharing: the strongest takes cut together at the beats
   viewers replay.
