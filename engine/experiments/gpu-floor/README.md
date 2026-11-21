# gpu-floor

An experiment, not engine code: nothing depends on this crate. It measures what the lab's
generated sedan (548 nodes, 3,581 beams, 528 collision triangles) costs on the GPU once the
production kernels' generality is gone, with hand-written Metal kernels checked against
the CPU reference (`engine-softbody`). Results and conclusions:
`engine/docs/measurements/vehicle-perf-first-principles.md` §8.42 and §8.43. How it compares with
production's kernels and what to port: `engine/docs/measurements/gpu-floor-vs-production.md`.

Three mechanisms, one kernel path each:

| mechanism | kernel | driver |
|---|---|---|
| one body and a fixed world | `kernels/floor2.metal` (`floor.metal` is the first layout) | `floor --v2` |
| body against body, two workgroups a pair in lockstep | `floor2.metal` + `pair.metal` | `pair --scene headon\|side\|rear\|apart` |
| body against rigid bodies (the reference's proxy set) | `floor2.metal`, `RIGID` blocks | `rigid --scene bricks\|rocks\|post` |
| surface contact (triangles, beams and band nodes against the world's boxes and spheres) | `floor2.metal`, `SURF` blocks; the queries are `slang/physics/feature_query.slang` converted as text (`floor.rs` `fq_msl`) | `rigid --surface`, `floor --v2 --pole 0.3 --surface` |

## Checking without the GPU

`--emu` compiles the Metal source as C++ (`emu/shim.h`: a lane is a coroutine, a barrier
the point where every lane has finished its phase) and compares every state word with the
reference:

```bash
cargo build --profile fast -p gpu-floor
./target/fast/floor --v2 --emu --frames 20 --wall 3 --speed 16      # a frame at a time
./target/fast/floor --v2 --emu --frames 3 --by-step                  # a substep at a time
./target/fast/pair --scene headon --emu --frames 12 --reverse        # lanes and groups in the other order
./target/fast/rigid --scene bricks --emu --frames 12
EMU_TRACE=1 ./target/fast/floor --v2 --emu --frames 1                # the kernel's own counts
./target/fast/rigid --scene post --surface --emu --frames 24        # surface contact
./target/fast/floor --v2 --emu --pole 0.3 --speed 11 --surface --frames 14   # a fixed pole
./surf-emu.sh                                                        # surface contact's scenes, both lane orders
./target/fast/rigid --scene post --surface --emu --reach --gap 0.8 --frames 10   # a frame by the kernel its plan gives it
./reach-emu.sh                                                       # the kernel choice: every scene, the hand-over at many substeps
./target/fast/fqcheck 300000                                         # the converted queries against the engine's
./target/fast/surfdemo                                               # what surface contact changes (CPU reference)
```

One emulator run at a time: they share `out/gpu-floor/emu*`.

`--reach` (on `rigid`, and on `floor --v2 --emu`) runs a frame by the kernel the body's
plan gives it (`floor.rs` `reach_plan`): the contact-free kernel (`REACH 1`: no code for
the proxies' shapes or surface contact, a guard of one compare a node a substep) while
everything is out of reach, and the whole kernel otherwise. A guard that finds a node
within the margin ends the contact-free dispatch at that substep's end and the whole
kernel (`REACH 2`) takes the frame up from there, in the same command buffer; nothing is
done twice. `--reach-eager` hands frames over part way on purpose; `--gap M` sets the
scene's distance to its shapes.

## On the GPU

Only under the lock, bounded:

```bash
python3 scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 ./target/fast/floor --v2 --frames 60 --phases
python3 scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 ./target/fast/pair --scene headon --gpu --frames 12
python3 scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 ./target/fast/rigid --scene rocks --gpu --frames 14
```

`--profile F` (all three drivers, with `--gpu` on `pair` and `rigid`) says where frame
F + 1's time goes: a substep a dispatch from its own saved state, cut short after one more
phase group at a time. Judge a change on the whole frame, two builds in one batch: the
profiler's sums have favoured changes that made the frame slower.

A dispatch outside the lock is refused (`Gpu::run_n`). Strict (the default) is fast math
off plus `#pragma clang fp contract(off)` and is bit for bit with the reference; `--fast
--native` is the fast variant, judged by the closeness report the drivers print.
`--cars N` / `--pairs N` put N copies in one dispatch (the staircase). `rigid --gpu
--reach` is the kernel choice on the GPU (`--list N`: the run's frames timed one by one
with what ran; `--reach-mix --cars 2`: two bodies with different kernels in one dispatch
against each alone; comparisons `--reach-cont`, `--reach-direct`, `--reach-nocont`,
`--reach-host`, `--def REACH_GUARD=0`, `--def REACH_NORIG=1`). `pair --gpu` also
prints the contact pass by stage on whole frames. `--def NAME=V` on any driver switches a
kernel option for an A/B from one binary: `pair.metal`'s `XT_OVERLAP`, `XB_ATOM`,
`XF_MASK`, `XD_MASK`, `XG_SPLIT`, `XP_LANE`; `floor2.metal`'s `LANE_LOCAL`, `SURF_BOX_ATOM`,
and the measurements `SURF_CUT`, `SURF_SUB`, `LIVE_DUMMY`, `COLD_OPAQUE`. `GPU_FLOOR_DUMP=file`
writes the assembled Metal source (for `xcrun metal -S -emit-llvm`); `GPU_FLOOR_INLINE=1`
marks every helper always-inline. `microbench 1 only=dead` (and `only=unchanging`,
`only=which:`, `only=stack:`, `only=live`) are the rows on code that never runs. `rigid --direct`
tests a box a lane a pair (the whole query) instead of staged; `fqcheck --gpu --latency`
is one query's latency by element, shape and outcome. `microbench`,
`mathprobe` and `xsync [--data]` measure the GPU's primitives, its arithmetic and its
memory across cores.

The car is read from `CAR` or `~/Development/rubble-artifacts/sedan-live/mods/sedan_new`.
