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

| Take | Patterns | Works | Gaps |
|---|---|---|---|
| `wy_frame10_ball` | S4 S5 | The collapse climbs and the core stands | The subject is small in frame; pastel debris |
| `wy_round_tower_sphere` | S4 | Floors drop, the core stands | As above |
| `bo_viaduct_pier` | S2 S1 | Spans hinge down whole, then break on the road | No vehicles on it (S2 without V3/V4); the camera is far; stress solver only |
| `topple_detonate` | S1 | The charges fire; it leans out as one mass for about 4 s and lands across the street, in Chrome as natively; it repeats exactly | It lands as one long piece, not storey-sized slabs. No cars on the street |
| `rr_storefront_car` | V5 | A local break; the shop stands | A ball stands in for the car; the façade renders as a blank pale wall |
| `ft_glass_tower_jet` | S3 | The cut and the collapse | A ball stands in for the aircraft; no tilt; a parked ball floats over the rubble at the end |
| `lab_house_*`, `lab_cuboid_grid_ball` | S4 S7 | Scale | Rubble's own mode does not repeat exactly at scale |

**Look, across every take:** the frames are pale and low contrast. In the Materials view,
free debris is tinted with its body's colour (pastel confetti). Brick and glass do not read
as brick and glass. Subjects fill about a third of the frame.

## Next, in order of impact

1. **The look:** true material colours for debris in takes, contrast and lighting,
   framing. Fix the floating parked ball. Add automated checks on recorded takes.
2. **Vehicles and aircraft as bonded bodies that crumple** (V1 V2 V5):
   - a launched, bonded free body: a car, a bus, an airliner;
   - the airliner thrown into the glass tower like a paper aeroplane: it crumples
     through it (S3);
   - cars driven through ground floors (V5);
   - stalled traffic on the viaduct that rides the spans down (S2 + V3 + V4).
3. **Topple that topples in Chrome** (S1) with a margin, not on a knife edge, then cars
   on the street (V4).
4. **A skybridge between two towers** (S2 + S6): it sags at its joints, hangs, then drops.
