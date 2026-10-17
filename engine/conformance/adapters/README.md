# Oracle adapters

Thin translators from `scene_v1` into external engines. They do **not** reimplement physics.

Rust API: `engine-oracles` crate. JSON sidecars in this folder declare revision pins and unsupported features.

| Adapter | Oracle | Role |
|---------|--------|------|
| `three_avbd.json` | `inspiration/pinned/three-avbd` @ `2bb1a318` | Primary 3D GPU AVBD numerical / seeded |
| `avbd_demo3d.json` | `inspiration/pinned/avbd-demo3d` @ `7701bd42` | Official demo parameter & residual tie-break (Z-up → Y-up) |
| `newton_vbd.json` | `inspiration/pinned/newton` @ `fb92f6d8` | Joint semantics (record compliant-ALM vs legacy) |
| `jolt.json` | `inspiration/pinned/JoltPhysics` @ `v5.3.0` | Independent rigid-body / sleep / LinearCast CCD |

Demo scenes that mirror three-avbd's menu live in `crates/engine-scenes` (`demo_viewer --list`).

Unsupported features must be declared; stiffness numbers are not compared across formulations.

Agreement levels: **numerical** (seeded step), **physical** (envelopes), **cross-backend** (Metal first).
