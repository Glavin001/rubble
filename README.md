# rubble

`rubble` is a Rust rigid-body physics workspace targeting GPU-first execution paths and WebGPU/Vulkan portability.

## Workspace crates

- `rubble-math`: POD GPU-friendly data layouts (`RigidBodyState3D`, `Contact3D`, `Aabb3D`, handles/events).
- `rubble-gpu`: lightweight `wgpu` compute context + typed GPU buffers with growth and readback.
- `rubble-primitives`: baseline primitives (`radix_sort_u64_keyval`, prefix scan, stream compaction).
- `rubble-shapes3d`: 3D shape descriptions + AABB generation + convex-hull validation.
- `rubble-broadphase3d`: pair generation from AABB overlap.
- `rubble-narrowphase3d`: contact generation (current implementation includes sphere-sphere).
- `rubble-solver3d`: iterative position solver + integration.
- `rubble3d`: end-to-end 3D world orchestration, handles, events, raycast/overlap queries, and invariants-focused tests.
- `rubble-shapes2d`/`rubble-broadphase2d`/`rubble-narrowphase2d`/`rubble-solver2d`/`rubble2d`: 2D pipeline baseline and facade world.

## Running tests with CPU Vulkan (lavapipe)

```bash
sudo apt-get update
sudo apt-get install -y mesa-vulkan-drivers vulkan-tools
mkdir -p /tmp/xdg
XDG_RUNTIME_DIR=/tmp/xdg WGPU_BACKEND=vulkan cargo test --workspace
```

This repository treats unavailable GPU backends as a test failure condition.

## Current known limitations (documented by design)

- This implementation currently includes verified end-to-end 3D and baseline 2D pipelines with events and invariants tests, but does not yet implement the complete AVBD feature matrix from the full long-form specification (e.g. full convex-convex SAT manifold clipping and graph coloring AVBD kernels).
- The solver is an iterative positional baseline tuned for stability/invariant tests, not yet a full SIGGRAPH 2025 AVBD implementation.
- Determinism remains best-effort and can vary across drivers/hardware due to GPU backend behavior.

## CI

GitHub Actions (`.github/workflows/ci.yml`) installs lavapipe, runs fmt/clippy/tests, and parses test output into a machine-generated summary artifact.
