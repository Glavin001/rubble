# gpu-floor

An experiment, not engine code: nothing depends on this crate. It measures what the lab's
generated sedan (548 nodes, 3,581 beams, 528 collision triangles) costs on the GPU once the
production kernels' generality is gone, with hand-written Metal kernels checked against
the CPU reference (`engine-softbody`). Results and conclusions:
`engine/docs/measurements/vehicle-perf-first-principles.md` §8.39.

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
./target/fast/fqcheck 300000                                         # the converted queries against the engine's
./target/fast/surfdemo                                               # what surface contact changes (CPU reference)
```

One emulator run at a time: they share `out/gpu-floor/emu*`.

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
`--cars N` / `--pairs N` put N copies in one dispatch (the staircase). `pair --gpu` also
prints the contact pass by stage on whole frames, and `pair --def NAME=V` switches one of
`pair.metal`'s options (`XT_OVERLAP`, `XB_ATOM`, `XF_MASK`, `XD_MASK`, `XG_SPLIT`,
`XP_LANE`) for an A/B from one binary. `rigid --direct`
tests a box a lane a pair (the whole query) instead of staged; `fqcheck --gpu --latency`
is one query's latency by element, shape and outcome. `microbench`,
`mathprobe` and `xsync [--data]` measure the GPU's primitives, its arithmetic and its
memory across cores.

The car is read from `CAR` or `~/Development/rubble-artifacts/sedan-live/mods/sedan_new`.
