# gpu-floor

An experiment, not engine code: nothing depends on this crate. It measures what the lab's
generated sedan (548 nodes, 3,581 beams, 528 collision triangles) costs on the GPU once the
production kernels' generality is gone, with hand-written Metal kernels checked against
the CPU reference (`engine-softbody`). Results and conclusions:
`engine/docs/measurements/vehicle-perf-first-principles.md` §8.38.

Three mechanisms, one kernel path each:

| mechanism | kernel | driver |
|---|---|---|
| one body and a fixed world | `kernels/floor2.metal` (`floor.metal` is the first layout) | `floor --v2` |
| body against body, two workgroups a pair in lockstep | `floor2.metal` + `pair.metal` | `pair --scene headon\|side\|rear\|apart` |
| body against rigid bodies (the reference's proxy set) | `floor2.metal`, `RIGID` blocks | `rigid --scene bricks\|rocks` |

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
```

One emulator run at a time: they share `out/gpu-floor/emu*`.

## On the GPU

Only under the lock, bounded:

```bash
python3 scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 ./target/fast/floor --v2 --frames 60 --phases
python3 scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 ./target/fast/pair --scene headon --gpu --frames 12
python3 scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 ./target/fast/rigid --scene rocks --gpu --frames 14
```

A dispatch outside the lock is refused (`Gpu::run_n`). Strict (the default) is fast math
off plus `#pragma clang fp contract(off)` and is bit for bit with the reference; `--fast
--native` is the fast variant, judged by the closeness report the drivers print.
`--cars N` / `--pairs N` put N copies in one dispatch (the staircase). `microbench`,
`mathprobe` and `xsync [--data]` measure the GPU's primitives, its arithmetic and its
memory across cores.

The car is read from `CAR` or `~/Development/rubble-artifacts/sedan-live/mods/sedan_new`.
