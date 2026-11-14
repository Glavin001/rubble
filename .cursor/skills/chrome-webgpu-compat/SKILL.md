---
name: chrome-webgpu-compat
description: Make compute shaders and wasm hosts that run on native wgpu also work in Chrome/Safari WebGPU. Use when a page shows "device lost", a WGSL uniformity or binding-limit error, a wasm panic, a pathological shader compile time, or results that differ from native; and before shipping any new kernel or bind group to the web demos.
---

# Chrome / WebGPU compatibility

Native wgpu accepts things Chrome's Tint compiler and WebGPU's limits reject. Every item below cost real debugging time on the vehicle work.

| Symptom | Cause | Fix |
|---|---|---|
| Uniformity error near `workgroupBarrier()` | Branching on a per-lane value before a barrier | Lane 0 loads, barrier, then everyone reads a `workgroupUniformLoad` copy. `uniformize_workgroup_lanes` (`engine-webgpu/src/lib.rs`) rewrites `gsUniform*` names; add new names to `NAMES` |
| "Too many storage buffers" | More than 10 storage buffers per stage | Split the bind group layout for the passes that need extra bindings (as `bgl_q`/`pl_q` in `engine-world/src/explicit_vehicles.rs`) |
| "Device lost" on page load | A huge pipeline compiled synchronously trips the GPU watchdog | Warm pipelines with `createComputePipelineAsync` before building the scene (`warm_pipelines`) |
| wasm panic in `World::step` | `std::time::Instant` on wasm32 | Use `web_time::Instant` |
| A frame takes hundreds of ms | Tint's bounds-checked Metal output compiles some loops pathologically (a per-node shape-range lookup ran at 755 ms) | Restructure, or disable that lossless optimisation on wasm (`set_coupling_node_ranges`, bit 64) |
| Wrong values only in Chrome, at some sizes | Apple back-end miscompile, e.g. a running min/max loop at n = 16 | Split into its own pass (as `memberBox` did) and test several sizes |
| Results differ on Metal after a kernel edit | Metal fast-math codegen shifts | Put new feature paths in separate functions behind an early return; run Metal `vehicle_parity` |

## Checks before shipping a page
- **Build the wasm,** serve the page and open it in headless Chrome with the SwiftShader or real-GPU flags from CONTRIBUTING.md. Read console errors, and treat any validation error as a failure.
- **For soft-body kernels,** run `softbody-check.mjs` (compiles every shipped module in Chrome).
- **Keep the tab visible when timing it** (see `measuring-web-physics-perf`).
- **Mind the limits:** the vehicle kernels use about 30 of 32 KB of workgroup memory. Phones (Safari on iOS) have the same or tighter limits and are untested.
