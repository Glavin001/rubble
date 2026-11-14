# Hood, boot and door hinges: why they tore, and the fix (2026-11-14)

The user's report, on the lab's mountain (`vehicle-lab.html?mode=mountain&car=gen&steepness=55&roughness=0.25&launch=12&surface=0&drag=0`):
the hood and the boot lid "push in, break the hinge, and fall into the car"; a
hinge should hold the panel while the panel crumples.

## The measurement

`engine-world/examples/hinge_breaks.rs` runs the page's mountain on the CPU reference
(`vehicle_lab::cpu`, the same scene and car as the page; `CpuLab::step_observed` hands it
the car after every 2 kHz step) and reports, for every break group, the step it broke
at, the beam that set it off (the one past its strength), that beam's force, strength,
stretch and length, the group's other beams, and the panel's own beams then (yielded,
broken, largest plastic strain); then when each part left the main body and the share
of the car's mass still on it.

```bash
cargo run --release -p engine-world --example hinge_breaks -- \
  [--car DIR] [--secs 30] [--seed 1] [--steepness 55] [--roughness 0.25] [--launch 12] \
  [--surface 0] [--drag 0] [--json FILE]
```

## The cause (the export as it was, `sedan_new.prev-2026-11-14`)

- **No hinge beam could yield.** Every beam of every hinge group (hood, boot, the four
  doors' upper and lower) was NORMAL with `beamDeform` FLT_MAX, k up to 1.5e6 N/m and
  `beamStrength` 40 kN (boot 25 kN). The first beam to reach its strength broke at
  17-29 mm of stretch and took its whole group with it.
- **The panels were barely bent when it happened.** At the hinge's break the doors'
  largest plastic strain was 2-6 % (9-20 of 108-110 beams yielded), the boot lid's 3 %
  (3 of 125). (The hood's hinges, at 12.5 s, went in a 50-group impact with the hood
  already crushed: 146 of 187 beams yielded.)
- **The beam that broke first was usually a long one to a far body node.** 8 of the
  12 hinge triggers in the seed-1 run were 0.5-0.77 m long (a fixed spring per beam:
  the same pull is a larger share of a long beam's strength); the boot's left hinge
  broke on `tk_i02-sh_e1_2dl`, 0.77 m to the rear rail, a crush member whose length
  changes when the rear folds whatever the lid does.
- **The hinge went before the latch.** The boot lid's latch (45 kN, one coupler) was
  intact when both its hinges had torn (seed 1): the lid hung on its latch.

## The fix (authoring only: `gen2/build.rs` `Role::Hinge`, `gen2/rig.rs`)

From the census (`beamng-authoring-census.md` §3: door hinges k 0.5-3e6 N/m, deform
10-77 kN, strength 30-98 kN a beam; bonnet 35-155 kN; boot 10-35 kN):

- **Every hinge beam yields before it breaks:** `beamDeform` = 0.4 × `beamStrength`
  (`HINGE_YIELD_SHARE`: doors and hood 16 kN, boot 10 kN), with the body's
  `deformLimitExpansion` 1.2 (a bounded bend, then elastic about it until its strength).
  Strengths and springs unchanged (in the census bands).
- **Anchored near the hinge, on the structure:** the body nodes are the shell's within
  `HINGE_REACH` 0.6 m of the hinge's marker, off the crush members (rails, crush cans,
  bumper beam, aprons, rear rails), and at least `HINGE_MIN_LEN` 0.12 m from the hinge's
  own nodes (the solver keeps a JBeam rest length at 0.1 m or more, `MIN_REST`, so a
  shorter beam cannot yield in compression and breaks on its strength instead).
  Groups: 8-9 beams, the rear doors' upper 5 (census 4-14).
- `tests/sedan_hinges.rs`: every hinge beam's deform is finite, in 10-77 kN, below its
  strength, with a bounded bend; strengths in the census bands; anchors within reach
  and off crush members; every hinge beam, as spawned, stretched 40 mm and squeezed
  40 mm (at most half its length) quasi-statically by the solver's own beam law yields
  and does not break (before: all 204 pulls broke at 26.7 mm without yielding).

Regenerated with `cargo run --release -p engine-softbody --example vehicle_gen_rig --
sedan --out ~/Development/rubble-artifacts/sedan-live/mods/sedan_new` (the generator
on `feat/sedan-body`, merged into `integration/lab-skin` at 040b55a1; it reproduced the
old export byte for byte before the change). The previous export is kept beside it as
`sedan_new.prev-2026-11-14`.

## Before and after

The mountain on the CPU reference, 30 s, steepness 55, roughness 0.25, launch 12 m/s,
surface 0, drag 0, rockslide off, seeds 1-4 (a tumble is chaotic: one run is one
sample). Columns: hinge groups broken (all, hood, boot, doors), latch groups broken,
the first hinge break (s), closures and fenders that left the main body, the attached
share of the car's mass at 30 s, beams broken at 30 s.

| car | seed | hinges | hood | boot | doors | latches | first (s) | left the body | attached | broken |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 1 | 12/12 | 2/2 | 2/2 | 8/8 | 4/7 | 5.2 | door_FL, fender_L, fender_R | 94.1 % | 347 |
| before | 2 | 11/12 | 1/2 | 2/2 | 8/8 | 2/7 | 3.6 | door_FR, door_RL | 94.9 % | 249 |
| before | 3 | 12/12 | 2/2 | 2/2 | 8/8 | 5/7 | 3.5 | door_FL, hood, trunk, fender_L, fender_R | 88.5 % | 443 |
| before | 4 | 9/12 | 2/2 | 2/2 | 5/8 | 1/7 | 3.7 | - | 97.2 % | 231 |
| after | 1 | 0/12 | 0/2 | 0/2 | 0/8 | 1/7 | - | - | 98.5 % | 109 |
| after | 2 | 1/12 | 0/2 | 1/2 | 0/8 | 1/7 | 16.3 | - | 98.2 % | 143 |
| after | 3 | 3/12 | 1/2 | 1/2 | 1/8 | 3/7 | 9.8 | fender_L, fender_R | 95.8 % | 246 |
| after | 4 | 2/12 | 0/2 | 0/2 | 2/8 | 1/7 | 15.7 | fender_L | 93.9 % | 243 |

Hinge groups broken: 44 of 48 before, 6 of 48 after; no hood, boot lid or door left
the body after. In seed 3 the hood's right hinge broke at 23.4 s after both hood
latches (21.8 s): the latch first, as the census has it. A remaining boot hinge break
(seed 3, 9.8 s) came with its latch in the same step.

An intermediate step, measured and not kept: a hinge bend of 25 % (`deformLimitExpansion`
0.25) broke 19 of 48 groups (short beams used up their bend and tore at 27-34 mm).

Other numbers the regeneration moved (the generator's step-bound pass re-softens springs
whose rows changed with the new anchors): the fenders', rear bumper's, body's and glass's
JBeam files differ in a few springs (fender ties about 6 % softer); the lint's ω·h 1.849
-> 1.864 (bound 1.92); `gen3_rig` `rest_fidelity_at_spawn_and_parked` parked 0.809 ->
0.813 mm (failing before and after); `sedan_jello` kerb ripple body 0.19 -> 0.21 mm, door
0.26 -> 0.29, hood 0.33 -> 0.37 (failing before and after: its ETK bands); first modes
doors 20.4 -> 20.0 Hz, hood 22.7 -> 22.4, boot 31.3 -> 30.0. `sedan_energy`,
`sedan_parts_collide`, `vehicle_lint` (tests) pass before and after; the lint's
`skin_coverage` hard failure is the same before and after.

Not measured: the GPU World's mountain (the page itself) before and after; the
flat-wall crash numbers.
