# The web World on Dawn (Chrome's WebGPU) on Modal

`engine/tools/dawn/run.sh` runs the vehicle lab's physics, the same World wasm and JS the
pages ship, in Node with Dawn on a Modal T4. Dawn (with its WGSL compiler, Tint) is
Chrome's WebGPU implementation. The harness is the lab without its page:

1. Build the lab as the page does (`createVehicleLab`, the generated sedan read from disk).
2. Step it to a moment.
3. Time frames, by wall clock and by the World's own GPU timestamps.
4. Run the explicit vehicles' A/B of a perf flag (`world.explicitAb`: the same frames with
   the flag on and off from one state, every state word compared).

```bash
engine/tools/dawn/run.sh -- --cars 2 --scene mountain --until 0.5 --ab 1 \
  --frames 10 --time-frames 30 --max-compile-s 60
```

## Why Dawn in Node, not Chrome

Headless Chrome gets no WebGPU adapter on Modal. The sandbox (gVisor) exposes
`/dev/nvidia*` but no `/dev/dri`, and Chrome's GPU process logs
`drmGetDevices2() has not found any devices`. Five flag sets failed the same way:
ANGLE/Vulkan, in-process GPU, dev-shm, GL disabled, Graphite.

Dawn's Node bindings (the npm `webgpu` package) enumerate Vulkan devices directly. They
report `nvidia / turing / tesla-t4` with `subgroups` and `timestamp-query`.
NVIDIA's Vulkan library needs `libx11-6 libxext6 libegl1` in the image.

## How a run works

- **The build happens on Modal:** `remote_build.sh --web` (the release web build's kernels
  and World parts, kernels embedded, as the pages ship), plus the generated sedan from the
  tree's code.
  - The bundle is stored by content.
  - A web build whose sources and settings are unchanged is reused without a build call.
    Rebuilding took 330–360 s even with nothing changed.
- **The run happens on a T4** (`dawn_run.py`), Node 22 (Node 24 crashed in V8 on load).
  Output streams live, including each kernel's compile as the lab reports it.
- **Kernels compile once per driver.** The NVIDIA driver's shader cache lives on a Volume
  (`rubble-dawn-caches`) and is committed every minute, so preemption or a time limit
  keeps finished compiles.
- **Harness options:**
  - `--max-compile-s S` stops a run as soon as any kernel compiles longer than S, naming
    the kernel and the driver. A timing run then costs cents, not a cache fill.
  - `--wait-cap S` bounds the wait for deferred kernels.
  - `--gpu-timing step|phases` sets the World's GPU timestamps (default `step`).
  - `--dawn-features a,b` passes Dawn toggles.
- **Limits:** `DAWN_CAP` (default 900 s) is enforced by a timer. The container's own limit
  is 1,800 s.
- **Local packing:** `--web DIR --car DIR` packs an existing local build instead of
  building on Modal.

## Measured (2026-11-24, T4, main at 8e4377ac, the generated sedan)

**Correctness.** `explicitAb` with `PERF_HIT_FB` (mask 1), 10 or 20 frames of 34 steps:
bit for bit on both drivers seen, 610.57.04 and 620.70.06, in every run.

**Compiles, through Dawn on NVIDIA's Vulkan driver.** Cold means a driver's empty cache;
warm means from the Volume:

| Kernel | Cold | Warm |
|---|---|---|
| `explicit_beams_split_group_wide` (deferred) | 766 s, 942 s | 0.6 s |
| `explicit_beams_split_member_wide` (deferred) | 483 s, 548 s | 1.2 s |
| `explicit_beams_contact_wide_nosurf` | 178–205 s | 0.85 s |
| `explicit_beams_wide_nosurf` (deferred) | 152–178 s | 0.74 s |
| `explicit_beams_solo_wide_nosurf` | 148–172 s | 0.84 s |
| `explicit_beams_split_grouplevels_wide` (deferred) | 106–121 s | — |
| `explicit_beams_selfsplit_list_wide`, `selfsplit_member_vv_wide` (deferred) | 84–114 s | — |
| The lab, before the deferred kernels | 346–433 s | 6.3–10 s |

- **Cold, every kernel the lab uses took about 25–30 minutes, per driver version.** Modal
  mixes at least two, and one host compiled a kernel 80× slower than another (`wide_hits`:
  33 s against 0.4 s).
- **The same applies to Chrome on Linux with NVIDIA,** which also compiles WebGPU through
  this driver. A first visit there could wait over 15 minutes for the split kernels. CI's
  first-load compile check (`compile_cost.py`, capped at 10 s) doesn't predict this driver.

**Frame time.** 2 cars, mountain, t ≈ 0.5 s, 30 frames:
- **The GPU's own time a frame** (the World's timestamps): **81 ms**, of which
  `explicit_frame` is 79 ms.
- **Wall time a frame:** 415–1,100 ms between runs.
  - The roughly 850 ms more than the GPU is Node plus Dawn: submission, readback
    callbacks, and the step loop polling each macrotask. It isn't what Chrome adds, so use
    the GPU time.
  - Process CPU time is not a host cost here: the loop spins while it waits.
  - Skipping Dawn's validation (`skip_validation`) did not lower wall time.

**Not covered.**
- A like-for-like comparison with CUDA: the same scene moment, from a portable fixture.
- Why Node plus Dawn adds hundreds of milliseconds a frame (Dawn's Node event pumping is
  in its native code).
- Rows other than two cars on the mountain.
