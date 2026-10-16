# Joints

One GPU joint model for every backend (Slang → MSL on Metal, Slang → WGSL on wgpu and in
the browser). Each joint is a PhysX-style **D6**: six degrees of freedom in joint frame A.
The DOFs are X, Y, Z, then twist about frame X, then swing1 (about Y) and swing2 (about Z).
Each DOF is LOCKED, LIMITED or FREE. The familiar joint types are presets of that descriptor.

| Preset (`JointDesc::…`) | Linear | Angular | Notes |
|---|---|---|---|
| `ball` | locked | free | three-avbd `Joint` with `stiffnessAng = 0` |
| `fixed` | locked | locked | three-avbd rigid joint, 6 rows |
| `hinge` | locked | swings locked, twist free | add `with_twist_limit`, `with_drive(JointDrive::motor/servo)` |
| `slider` | X free, Y/Z locked | locked | `with_linear_limit(Dof::X, lo, hi)`, linear drive |
| `cone` | locked | swings in an elliptical cone, twist free | `with_twist_limit` for ragdoll shoulders and hips |
| `distance` | one radial row, \|pA − pB\| = L | free | rigid rod |
| `max_distance` | radial row, \|pA − pB\| ≤ L | free | tether, one-sided |
| `spring` | soft radial row with rest length | free | three-avbd `T_SPRING` |
| `d6` | any | any | `with_motion(dof, Motion::…)` |

Modifiers available on every preset:

- `with_break(force, torque)`: breaks the joint (PhysX `setBreakForce`).
- `with_yield(torque, bend)`: plastic welds, three-avbd yield / breakBend.
- `with_stiffness(lin, ang)`: soft joints and the mouse-drag spring.
- `with_soft_limits(k, d)`: soft limits.
- `with_collision(true)`: lets the two jointed bodies collide.

## Files

- `crates/engine-schema/src/joint.rs` holds the 288 B GPU record `Joint`, the `JointDesc`
  presets, and the motion, flag and drive encodings. Its size is asserted in `abi_sizes`.
- `slang/physics/joints.slang` holds the row math, shared by every kernel and backend.
- `slang/kernels/world_step.slang` holds the joint passes:
  - `joint_hash_insert` skips broken and `collide` joints.
  - `warmstart_joints` stores C0 and decays λ and k. It also wakes a sleeper that is jointed to an awake body.
  - `csr_degree_joints` and `csr_scatter_joints` skip broken joints.
  - The joint branch of `primalAccumulate`.
  - The joint block of `solve_dual`.
  - `sleep_island_gate` keeps joint islands together and keeps bodies with running drives awake.
- Hosts:
  - `MetalWorld` (Metal shared memory) has `add_joint_desc`, `remove_joint`,
    `drain_broken_joints`, `set_joint_drive`, `joints` and `snapshot`.
  - `WgpuSim` has the same API, plus `load_snapshot` and `download_joints`.
  - `SceneBuilder::add_joint` puts a joint at rest in the bodies' current orientations
    (three-avbd `rest: 'current'`). `world_frames` builds frames from a world anchor and axis.

## Rows

There are seven scalar row slots per joint:

| Slot | Row |
|---|---|
| 0–2 | linear: world axes when all are locked, frame-A axes otherwise, or one radial row |
| 3 | twist |
| 4 | swing1, or the cone row |
| 5 | swing2 |
| 6 | drive |

AVBD re-evaluates C(x) every iteration (joints are not Taylor-linearised like contacts).
Each row has a mode:

| Mode | Force | Dual update |
|---|---|---|
| EQ (hard) | `f = k(C − αC0) + λ` | `λ ← f`, `k ← min(k + β\|C\|, kMax)` |
| EQ (soft) | `f = k_soft · C` | no λ |
| LIMIT | one-sided, on the nearer bound: `f = min(k·Ĉ + λ, 0)` | stabilises only an existing violation (`Ĉ = C − α·min(C0, 0)`). A slack limit's penalty restarts at kStart, so a new touch is soft, like a new contact. |
| MOTOR (hard) | `C = (v − v0) − v_target·dt`, `f` clamped to ±F | clamped multiplier (three-avbd 2D motor). Saturated rows drop their Hessian, so the body accelerates at F/I. |
| SERVO (soft) | `f = k(v − target) + d((v − v0)/dt − v_target)`, clamped to ±F | none |

Angular hard rows are scaled by the torque arm (three-avbd `|size_a + size_b|²`). Penalties
then share the linear rows' range, and soft rows use the user's physical stiffness.

Math notes:

- **Linear frame-A axes.** The exact gradient puts body A's lever arm at anchor B
  (`(pB − xA) × axis`), because rotating A also swings the axis (PhysX ConstraintHelper).
- **All three angular DOFs locked.** Uses the three-avbd small-angle quaternion error,
  `2·vec(cA·conj(cB))`, which is the weld and slider path.
- **Twist.** `2·atan2(q.x, q.w)` of `q = conj(cA)·cB` (PhysX computePhi). The gradient is exact:
  `dφ = g·R_Aᵀ(δθB − δθA)` with `g = (qw²+qx², qw·qz+qx·qy, qx·qz−qw·qy)/(qw²+qx²)`.
  - Using bX as the axis, which is exact only at zero swing, blew up ragdolls with free swings.
  - The row drops out past about 150° of swing, where twist is undefined.
- **Swings.** `atan2` of frame-B X projected into frame A, with the exact gradient.
  - Locked swings are the hinge's two alignment rows.
  - Limited swings form either one elliptical cone row or two pyramid rows.
- **Angle limits.** The side is picked on the circle, measured from the range midpoint. A
  joint forced past ±π then still pushes back toward its near bound. Before this fix, the
  ragdoll knees flipped sign.
- **Breaking.**
  - The condition is `|λ_lin| > breakForce` or `|λ_ang|·arm > breakTorque`.
  - Breaking sets `JOINT_FLAG_BROKEN`. From the next step the joint leaves the CSR and the
    joint hash, so the pair collides again.
  - Hosts find breaks by scanning flags (`drain_broken_joints`), which also frees the slots for reuse.
- **Plastic yield.** Only for all-angular-locked joints. Past `yieldTorque`, the angular force
  is clamped and frame B is re-based on the current pose. The joint breaks once the bend from
  `rest_start` exceeds `breakBend`.

**Ball fast path.** A pure ball joint loads only the fields its three rows need, through
`JointBall`, in the primal, dual and warm start. On chain mail 1.6k this is 1.13 ms against
1.40 ms for the generic path, in interleaved A/B runs.

**WebGPU limit.** Chrome allows 10 storage buffers per stage, and the primal kernels already
use 9–10. Joints therefore stay in one buffer (binding 5) and never get a second buffer.

## Tests

```sh
cargo test -p engine-schema                     # ABI size, motion bits, rest_from_bodies
cargo test -p engine-metal --release --test joints -- --test-threads=1
cargo test -p engine-metal --release --test joints_wgpu_parity -- --nocapture
cargo test -p engine-webgpu --release --lib -- --test-threads=1   # CSR / primal fixtures with joints
```

`tests/joints.rs` has 18 behaviour tests:

- A hinge frees only its axis, and its twist limit stops the swing.
- A motor reaches its speed and respects its force limit.
- A slider stays on its axis and stops at its limits; a servo tracks its target.
- A weld holds a cantilever and resists twist.
- A cone holds a pendulum; a distance rod and a tether keep their lengths.
- A spring settles at Hooke's extension; a soft joint sags by its stiffness.
- Joints break at their threshold and the slot is reused.
- Plastic welds yield, and break past their bend.
- A running motor never sleeps, while a still chain sleeps as one island.
- The `collide` flag controls contact.
- A twist limit with free swings stays stable.

`joints_wgpu_parity.rs` steps a "zoo" of every preset on Metal and on wgpu from one Rust
snapshot. Trajectories must agree within 1 mm until the first limit event, and break states
must match.

## Demos

The scenes are in `crates/engine-scenes/src/joints.rs`:

- `Joints Showcase`: 12 stations, one per feature.
- `Ragdolls (40)`.
- `Ragdoll Pile (10k)`: 11k bodies and 10k joints.
- `Hinge Chains (10k)`.
- `Motor Field (10k)`: 2.5k motors.

```sh
cargo run -p engine-scenes --release --bin scene_export     # → engine-webgpu/generated/scenes/*.bin
cargo run -p engine-webgpu --release --bin wgsl_export
cd crates/engine-webgpu && python3 -m http.server 8765
# Chrome: http://127.0.0.1:8765/demo.html?scene=joints  (ragdolls, ragdolls10k, hingechains10k, motorfield10k)
node crates/engine-webgpu/tools/record-joints.mjs         # → test-results/joints/*.mp4 (headless Chrome WebGPU)
cargo run -p engine-scenes --release --example joint_settle -- "Ragdolls (40)" 1200   # KE / sleep probe
```

The browser page uploads the Rust-built records as-is; JS only patches fields for interaction:

- Alt-drag grabs a body with a soft drag joint.
- `F` fires a ball.
- `M` toggles the motors.
- Servos sweep on their own.

## Benchmarks

`bench/tools/joint-export` writes the D6 scenes in bench format v3, which adds 32-word joint
records with frames, motions, limits and a drive. PhysX runs them as `PxD6Joint`, PhysX's
GPU-prepared joint type. three-avbd has no hinge, cone, limit or motor rows, so it runs only
the v1 ball-joint scenes.

```sh
python3 bench/run.py --engines rubble-metal,rubble-wgpu,three-avbd,physx-gpu --scenes chain_mail_1.6k,jointed_drop_6k,joints
```

## Not yet

- Gear and rack-and-pinion joints. PhysX keeps these CPU-driven because they accumulate angle on the host.
- SLERP angular drives.
- Per-axis Y/Z linear limits in the bench format.
- `CMD_SPAWN_JOINT` / `CMD_DESTROY_JOINT` GPU commands. Hosts write joint records directly.
