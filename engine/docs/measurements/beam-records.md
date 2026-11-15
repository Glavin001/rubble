# The beam phase on records sorted by kind (2026-11-14)

The beams stream of the vehicle performance rebuild (`vehicle-perf-first-principles.md` §3.4,
"compact beam records sorted by kind"). Branch `perf/beam-records`. Lossless: the same
arithmetic, operation for operation; only which lane evaluates a beam, and when, change.

## What changed

`explicit_plan::W_BEAM_ORDER` (Params word 96): a gStatic region, per vehicle a header
(entries, hot entries, its first incidence position), then its entries, 2 records each:

| Class | Beams | Entry | Kernel |
|---|---|---|---|
| hot | JBeam plain, support, bounded, rope without an extension record | (local beam, kind \| type \| group as i16, ends as two u16, its two incidence slots as two u16), (k, c, damping filter, long bound) | `jbeamHot` |
| aero triangle | an unbroken (18, 19) pair | (local beam, kind \| third node, x0 \| x1, first record's slots), (drag, lift, sin stall, second record's slots) | `atriHot` |
| pair | torsion, pressure triangle, LBEAM (and a broken aero pair) | local beam | `beamUpdate` on the second record, then the first, one lane |
| single | everything else | local beam | `beamUpdate` |

Hot entries first, then the rest, each block sorted by kind. `beamPhaseSorted` runs the
hot block (each beam: its entry, then its state and both ends: two dependent load levels;
the limit and plastic fields only where the yield or bounded rule reads them), then the
rest. `RUBBLE_BEAM_ORDER=0`, `ExplicitBeamsGpu::set_beam_order(false)` or word 96 = 0
runs the old file-order loop, which is untouched.

Why it is bit for bit: a beam writes only its own state, its own incidence entries (a
two-record element: its own two records' entries, on one lane) and atomic minima of the
group slots; a node sums its incidence entries in its own CSR order. The CPU reference
is unchanged.

## The generated sedan (`vehicle_lab --measure beam-census`)

548 nodes, 4,251 beams: plain 2,965, support 334, bounded 15 (hot: 3,314); extension
plain 199, anisotropic 33, aero triangles 335 pairs, couplers 7, torsion 12 pairs, RoR
plain 4. In file order 73 of 133 SIMD groups of 32 beams mix code paths (1.62 paths each).
Sorted: 3,904 entries (3,314 hot, 335 aero pairs, 255 other), 8 a lane at 512 lanes
(was 9). Nodes: 8,502 incidence entries, 15.5 a node (median 15, p90 24, max 32).

## Bytes a beam (common path, read + written)

| | Before (file order, `jbeamUpdate`) | After (`jbeamHot`) |
|---|---|---|
| static | 48 (3 records) + 8 slot indices | 32 (entry) |
| state | 32 | 32 |
| ends | 64 | 64 |
| incidence writes | 32 | 32 |
| total | 184 | 160 |
| dependent load levels before the arithmetic | 4–5 (state → record → ends, slot indices) | 2 (entry → state, ends) |

Aero triangle pair: 376 B over two lanes (the second record loads its extension records
only to return) → 224 B on one lane (entry 32, two states 32, six node records 96,
four writes 64).

Nodes: unchanged. The node records are already the hot integration state (3 × 16 B:
position and mass, velocity and friction, force and self group). The incidence entries
are 16 B (`.w` carries centred damping's slope); 12 B entries would need scalar arrays
(three loads an entry) and save no dependent level.

## Measured

**Directional only.** Every Mac run below started with the GPU at 82–100 % use from other
jobs on the shared machine, so absolute numbers are inflated and small differences are
within noise. Native, `vehicle_lab --ab T --ab-switch beams` (20 frames of 34 steps each
way from one snapshot, ms a frame, sorted vs file order):

| State | Sorted | File order |
|---|---|---|
| Parked, t 2.5 (hot path only, before the aero pairs) | 6.15 | 6.30 |
| Parked, t 2.5 | 6.02 | 6.42 |
| Mountain, t 9 | 12.45 | 11.04 |
| Mountain, t 20 | 12.01 | 12.41 |
| Mountain, t 35 | 11.61 | 11.99 |

The beam phase's own cost, by repetition (a temporary build running the sorted beam
phase 5× a step, time only, before the aero pairs): parked 5.95 → 8.53 ms a frame, so
≈0.65 ms a frame (≈19 µs a step) for the sorted phase, against the audit's one-core
floor of ≈0.25 ms. Node-phase repetition does not measure (integrating 5× a step blows
the state up). Fleet (32 cars apart): not measured (the runs did not finish within the
GPU lock's limit). All of this needs repeating on an idle machine.

## What did not pay

- Software pipelining of the hot entries (entry k + 2·lanes and beam k + lanes's state
  and ends in flight while beam k computes): parked 5.92 vs 5.90 ms a frame, no gain
  (`engine/out/explicit_beams.pipelined.slang` in the worktree). The compiler already
  overlaps what it can; a lane's 7–8 hot beams are not the long pole.
- Ablation-kernel phase shares (`--ab-ablate`): the ablation variant's three frame
  modules did not compile within the GPU lock's 10 minutes on the shared machine; phase
  costs were taken by repetition instead (a temporary build running the beam phase 5×,
  time only).

## Barriers in the beam phases

Each step has two syncs around the beams: after the node phase (the beams read the
integrated ends) and after the beam phase (the nodes read the incidence entries). Explicit
integration needs one per step (force from positions, then positions from force). The
second could go if each node lane computed its own beams' forces after the first barrier
(every beam twice, the beam state double-buffered by parity, and a group break seen by its
nodes one step later unless masters are found in a separate pass): about 34 syncs a frame
(≈0.05–0.1 ms) for twice the beam arithmetic and a change to when group breaks take
effect, which is not bit for bit. Not done; it needs the owner's decision.
