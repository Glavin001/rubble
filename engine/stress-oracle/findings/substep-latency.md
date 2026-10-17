# Where a stress substep's latency goes, and what is lossless to take out

Written 2026-10-17 (branch `claude/stress-top3-roi-8a4275`, M3 Max, Metal, precise math).
It covers tracker items 2 (intact-only kernel) and 1 (low-latency substep replay). Every
timing below comes from interleaved A/B runs. Another app (Unreal Editor) and other
sessions' benchmarks shared the GPU at times, and absolute numbers moved by up to 2× between
runs. Only the ratios within one interleaved pair are comparable.

## The two paths a substep takes

- **Contact pipeline** (impact islands: the rammed house, the struck wall, debris in
  contact). One round of dependent dispatches per substep. Before this work there were three:
  `contact_forces` (pairs, impactor candidates, travel checks), `sums_impactors_bonds`
  (contact sums, impactor integration, the joint law of the narrow contact islands' bonds,
  one bond per thread) and `contact_islands` (one threadgroup per island: chunk updates,
  rigid frame, drift removal). house_ram spends 35,709 of its 44,302 substeps here.
- **Segment mode** (narrow islands outside the pipeline, for example a free wall's or an awake
  house's own vibration). `island_frame` runs every substep of a segment in one dispatch,
  one 256-thread group per island. Each thread evaluates 2–4 joint laws in series.

## Measurements

### What the dispatches cost

`examples/dispatch_floor.rs`: a round of three dependent trivial dispatches costs 7.2–8.3 µs
(2.4–2.8 µs each). A production substep costs about 45–50 µs, so launches are about a sixth
of it. The rest is the kernels' own dependent chains.

A Metal indirect command buffer would replay the encoding, which is host work (5–6 ms per
house_ram frame, measured by the new `profile.encode`). It does not remove the GPU's
per-dispatch cost. A persistent kernel would need a grid-wide barrier, and Metal does not
guarantee one: threadgroups have no forward-progress guarantee, so spinning on device
atomics can deadlock. Neither option reaches "2–5 µs per substep".

### Ablations of the joint law (lossy, for measurement only, never adopted)

Each ablation removed one part of the bond update in `world.slang` / `joint.slang`.

| Variant | Segment mode, house-1story intact (µs/substep) | house_ram contact path |
|---|---|---|
| Production law | 36.0 | 44.3 µs/substep (188 bonds broken) |
| No contact-offset tracking | 36.9 | 43.6 |
| No static fatigue | 36.1 | 43.4 |
| Both of the above | 35.0 | 43.0 |
| No damage evolution (kappa still tracked) | 34.6 | 39.1 (nothing breaks, so a different scene composition) |
| Bare spring-dashpot law (as `stress_step.slang`) | 15.5 | 35.7 (nothing breaks) |
| Bare law plus `BondDyn` read and write | 18.1 | 36.5 |

Per-kernel GPU timestamps (`STRESS_GPU_TIMING=1`) on house_ram, where both rows break no
bonds and so run the same composition:

| Variant | `sums_impactors_bonds` (holds the bonds) |
|---|---|
| Full law without damage evolution | 16.0 µs |
| Bare spring-dashpot law | 16.5 µs |

- **In the contact pipeline the joint law is off the critical path.** The dispatch that
  carries the bonds is set by the impactor integration's tree sum and rigid update.
  Replacing the law with a bare spring does not shorten it.
- **In segment mode the law dominates, but almost all of it is required physics.** The
  only part an intact bond can skip without changing results is damage evolution, about
  4 % of the law. The rest are state every substep must update exactly:
  - stress measures and strength indices (utilization and kappa);
  - the strain-rate filter and dynamic increase factor;
  - fatigue life;
  - contact-interface friction offsets, since `patch_after_damage` is off in the established methods;
  - rebar plasticity and the energy accounting.

  The ×2.5–6 of REALTIME_ROADMAP A6 compared the full law with a kernel that computes
  none of these.

### The island kernel

- Without its chunk updates, `contact_islands` falls from about 26 to 17 µs (timestamped).
  The kernel's skeleton dominates: island load, barriers, reductions and the island
  write-back.
- Segment mode with the bonds spread over a larger group (bit-identical, see below):
  - 512 threads: house-1story 24.7k → 25.8k substeps/s (+4.5 %), brick-building +2.7 %;
  - 1024 threads: 24.5k → 21.0k (−14 %). A 1024-thread group of this kernel forces
    register spills.

## What was adopted (bit-identical)

1. **The contact pipeline in two dispatches per substep** (`forces_and_bonds`,
   `islands_sums_impactors`, with `travel_start` once per batch). `STRESS_GPU_FUSE=0`
   restores three.
   - The narrow contact islands' bonds run with the contact forces: both read only the
     last substep's state.
   - Each island sums its own chunks' contact segments. The segments are in chunk order, so
     an island's segments are contiguous.
   - The impactors integrate in the same dispatch as the islands, in other threadgroups.
     Neither reads the other's output.
   - **Batch control.** The joint law must not run in a substep that a travel check
     discards, so the travel checks move to the end of the previous substep (island
     kernel, impactor integration) and record the substep they stop (`range.x` of the
     sentinel island). Each kernel compares it with its own substep, which reproduces
     the three-dispatch order exactly:
     - the crush passes and contact forces still run in the stopped substep, as they do
       when STOP_NOW lands inside `contact_forces`;
     - the joint law, sums, islands and impactors do not;
     - a stop after a split's substep is moot, because the old order never checks it.
2. **Segment-mode bonds over 512 threads** (`island_frame_bonds_wide`). Chunk loops and
   every reduction stay on the first 256 threads with the same partition. Bonds are
   independent: no sum runs over bonds, and the halt and coverage flags are ORs. So the
   results are the same bit for bit. `STRESS_GPU_ISLAND_BONDS=256` restores the old kernel.

**Gate.**
- `stress_bench` digests, unmodified HEAD (655ed12) against this branch: realtime 4/4,
  accuracy 12/12, towns 3/3 and packs 14/14 identical. That includes wall_ram, b5_v10,
  b5_v30 and b5_v40, which re-plan mid-batch.
- `cargo test -p stress-gpu`: 23 pass, 6 ignored. `cargo test -p stress-rubble --features gpu`:
  all pass, including the handover-energy, no-rewind and segment-snapshot tests.

## Found on the way (not changed)

**A latent race in the three-dispatch order.**
- When a travel check sets STOP_NOW inside `contact_forces`, the pair and candidate
  threads that already started in that dispatch have advanced their contact history
  (`contact_state`, ledgers) for a substep the host then re-plans and runs again.
- Crush passes before it do the same to the impactor's crush state.
- In practice the pairs, in the lower threadgroups, always win the race, so the double
  step is deterministic and every digest repeats. The two-dispatch path keeps the same
  behaviour on purpose.
- Running none of the stopped substep's contact work would be more correct, but it
  changes results on re-planning scenes. That is a fidelity decision (tracker item 31's
  family).
