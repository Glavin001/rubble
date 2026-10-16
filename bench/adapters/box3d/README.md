# Box3D adapter

[Box3D](https://github.com/erincatto/box3d), Erin Catto's 3D engine (C17, SIMD,
soft-step solver), on the CPU. Sources are pinned in `../../third_party/fetch.sh`
(currently `efca0c18`, 2026-10-15). `build.rs` builds them with Box3D's own CMake in
Release with `BOX3D_VALIDATE=OFF` (it defaults to ON: heavy validation), `BOX3D_PROFILE`,
samples, tests and benchmarks all off. A C shim (`src/box3d_shim.c`) drives the world.

```sh
bench/third_party/fetch.sh
cd bench && cargo build --release -p bench-box3d
./target/release/bench-box3d --scene scenes/out/brick_pyramid_12.scene.json --out r.json --threads 12
```

## Settings (recorded in each result's `engine.config`)

- **Threads:** Box3D's internal scheduler with `workerCount = --threads`, as its own
  `benchmark/main.c` does. Its docs advise performance cores only, which is what `run.py`'s
  `box3d-mt` uses (12 on an M3 Max).
- **Iterations:** `--iterations` is the substep count, default 4 (Box3D's benchmark value).
  The relax and restitution passes stay at their defaults.
- **Exact mass properties:** every shape has `updateBodyMass = false`, then
  `b3Body_SetMassData(mass, centre 0, diagonal principal inertia)`.
- **Contacts:** friction per body, combined by Box3D's default sqrt(a·b), the same rule as
  three-avbd and rubble. Restitution 0. Soft contact tuning stays at the defaults
  (`contactHertz` 30, damping ratio 10, `contactSpeed` 3 m/s); see `--set` keys in
  `src/main.rs`.
- **Things turned off:** no damping, `allowFastRotation` (no per-substep rotation clamp),
  maximum linear speed 10⁴ m/s, CCD off, sleeping off unless `--sleep on`, contact
  recycling on (Box3D's default; `--set recycling=off`).
- **Shapes:** boxes become `b3MakeBoxHull` hulls. Convex compounds get one
  `b3CreateHull(points, 128)` shape per hull, and the result reports any vertex-count
  change. The ground plane becomes a static box (4× the scene extent, at least 200 m wide,
  2 m thick) with its top face on the plane, because Box3D has no plane shape.
- **Joints:** ball joints become `b3SphericalJoint`s with `collideConnected` from the scene
  (false). Format v3 D6 joints (hinges, cones, limits, motors) are refused with a clear
  error. Box3D has revolute/prismatic/motor joints, but they are not mapped yet.
