# The threshold profile on the GPU alone

The simple threshold mode is the cheapest stress model there is: a bond breaks when the
contact load on one of its chunks beats its tensile capacity. Profiling the lab's
threshold mode (2026-10-22) showed the model costs nothing, and the integration around it
costs everything. The resident `GpuWorld` used for it is built around the host's mirror:

- the host follows every split, about 5 ms on fracturedTower;
- every chip rebuilds its whole parent: one 256-thread workgroup runs connected components
  over all 3,711 chunks (2.5 ms), and another re-lays out all its chunks and bonds
  (4.1 ms);
- every split retires the parent's colliders and appends new ones, so the collider count
  doubles until a compaction;
- two blocking GPU reads per frame;
- the renderer computes every chunk's transform on the CPU.

`stress-rubble::threshold::ThresholdWorld` runs the same profile with none of that.
`stress-web` uses it for the lab's threshold mode by default
(`stress_set_threshold_gpu(false)` keeps the `GpuWorld`).

## Design

**Fixed slots.** At set-up the engine gets a body slot for every body a split can make
(one per chunk at most), a parked sphere per shot, and every chunk's colliders. Nothing
is created or retired afterwards, so the engine's counts never change and there is no
compaction. A split rewrites body records and re-parents colliders in place.

**One submission a frame, nothing read back.** rubble's step, then the contact ingest
(`EngineIngest`), then `GpuFracture` (`stress-gpu/shaders/fracture.slang`):

1. `fr_loads`: each chunk's engine load (`impact.slang` `chunk_engine_load`, the same
   function the resident frame uses);
2. `fr_breaks`: each intact bond against its capacity (strength × area × Weibull factor);
3. `fr_gate`: sizes every split kernel's indirect dispatch. All are zero when no bond
   broke, so a quiet frame costs nothing more.

**The split, at any scale.**

- *Pieces:* connected components of every dirty body's chunks over its intact bonds,
  found by lock-free union-find with each root hooked under the smaller. A piece's root
  is its lowest chunk, whatever order the threads ran in.
- *Minimum edit:* each dirty body's largest piece keeps the body: its slot, its colliders'
  slots and its identity. This is blast-stress-solver's `plan_split_migration` reduced to
  fracture, where every child has one parent. The k-th other piece by root takes the k-th
  free slot, found by two scans.
- *Mass properties:* each piece's body is `cook_with`'s (mass, centre, principal axes from
  its chunks' parts), from 64-bit fixed-point sums. Integer sums do not depend on the
  order of the atomics. The body is posed in its parent's frame and moves with its
  parent's rigid motion.
- *Colliders:* re-parented in place, and the ingest's collider table follows. A
  re-parented body's `generation` is the engine's next step, so no contact cached before
  the split warm-starts it.

**The view.** `fr_view_*` write the renderer's object transforms and colours, the shots
and the bond lines on the GPU. `Renderer::draw_resident` draws them. The *Bodies* colour
is a body slot's, stable because the largest piece keeps its slot.

**Carried impact pulses** live in a two-half pool, not in per-chunk slots. Each frame a
chunk reads its carried pulses from one half and writes the ones that last into the
other, claiming its run with one atomic add. Where the run lands changes no result,
because each chunk reads only its own pulses, in order. The pool holds two pulses per
engine contact.

**Capacity.** rubble's contacts are sized for a whole suburb in a heap: 12 per chunk, at
most 2^19 and what the device's storage binding holds. 2^19 of rubble's 256-byte records is
WebGPU's default 128 MiB binding, so the world fits any WebGPU device. More is an error the
world reports (rubble's overflow bits), never a silent drop. The 96 houses at bond strength
×1/8 with 16 shots (58k bodies, 115k bonds broken) stay within it.

**Strikes.** A shot that will reach a body within the next frame, closing faster than
`tiered_impact_speed` (5 m/s), breaks every intact bond of that body whose share of the hit
beats its capacity, before the engine steps: the threshold profile's `threshold_predict`.
The force is the impulse of an inelastic hit over the Hertz contact time, spread as a point
load on a half-space: `F min(1, A / (2 pi d^2))` at distance d from the contact. The
contact is where the shot first meets a chunk's box; the struck chunk is the nearest to its
path. `fr_strike_*` run per body, chunk and bond (atomics, not a group per island), so the
cost does not depend on body size. `STRESS_THRESHOLD_STRIKE=0` turns strikes off (a
measurement), and `STRESS_THRESHOLD_STRIKES=1` in the harness prints each strike.

**Counters.** A ring of mapped buffers, read without waiting (`ThresholdWorld::poll`):
broken bonds, bodies, error bits, and rubble's own step counters. A fracture error (no
free slot, a sum out of range, a union that did not settle, a full pulse pool) or an
engine overflow is reported as an error, and the lab pauses on it.

**Pacing.** The lab queues at most 3 physics frames ahead of the last one the GPU
finished. A slow stretch then slows the simulation instead of backing up the queue.

**Shots.** There is no limit. Past the 16 parked spheres, each shot reuses the oldest.

## Tests and measurements

- `stress-gpu/tests/fracture.rs`: the split against a host reference
  (`fracture::reference_split`) on free and anchored grids, and a 24k-chunk grid
  shattered over three rounds. Every chunk's body and slot match exactly; mass, centre of
  mass, velocity transfer and collider poses match to f32.
- `bench_split` (ignored): the split alone on 100k chunks.

| Split on 100k chunks | Wall time, submit to done |
|---|---|
| Quiet frame (the measurement floor) | 1.7 ms |
| One chip | 2.0 ms |
| 40% of bonds in one frame (549 bodies) | 2.5 ms |
| Every bond in one frame (99,999 bodies) | 2.3 ms |

`stress-rubble --example fracture_glb` with `STRESS_THRESHOLD_GPU=1` (frames timed to the
GPU's end, M3 Max):

| Scene | Rate, shots | Median | p95 | Worst | `GpuWorld` |
|---|---|---|---|---|---|
| fracturedTower | 60 Hz, 4 | 3.6 ms | 6.5 | 8.1 | 3.2, 15.5, 32.6 |
| fracturedTower | 120 Hz, 4 × 16 mass | 2.3 | 4.9 | 6.9 | stops (`STRESS_RESIDENT_PULSES`) |
| 49 cuboid houses (39,648 chunks) | 60 Hz, 4 | 7.0 | 10.2 | 14.8 | |

In Chrome, fracturedTower at 120 Hz with four ×16 balls holds 0.97× real time through the
hits and 1.0× after. The `GpuWorld` dropped to 0.53× and then stopped.

## Open

- **Strikes from debris.** Shots strike (above). Fast free debris as projectiles (the
  `GpuWorld`'s fast list) is not ported. At scale it needs a spatial grid: the fast list
  is O(islands × fast bodies) and truncated at `MAX_FAST`.
- On the 96 houses, the `GpuWorld`'s own `threshold_predict` never fired for the lab's
  ball (one shot: 366 bonds broken with strikes on or off). That is why shots barely
  dented the houses. This world breaks 1,250-1,340 bonds for the same shot.
- **Fixed:** determinism and settling at scale. rubble sized its warm-start hash from a
  read-back contact count, which this world never read back, so most contacts
  cold-started. `WgpuSim::set_plan_hints` now gives it a fixed plan: the hash sized for the
  contact capacity, every colour its own dispatch.
  - The 49 houses now settle as the `GpuWorld` does: 64 bonds broken against 62, and
    bit-identical run to run.
  - 96 houses: median 4.7 ms per frame at 60 Hz (was 19.5), and 5.1 ms at 120 Hz with
    p95 5.6 ms (budget 8.3).
- **Host involvement left:** a shot's launch (one body write), and the error check, which
  the host reads a frame or more late.
