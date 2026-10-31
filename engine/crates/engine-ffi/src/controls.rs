//! The calls the demo pages make that the core ABI did not cover yet: world
//! and shape edits, live joint drives, vehicle damage and telemetry, soft
//! bodies from descriptors, cuboid houses and debug drawing.
//!
//! Descriptors and settings cross as JSON, in the engine's own units and
//! design frame (metres, Y up, as `vehicle_presets.json` and the soft
//! descriptors are written); only the spawn transform and velocity use the
//! world's coordinates.

use super::*;
use std::ffi::c_void;

/// Copy `text` into `buffer` (NUL-terminated) when it fits. `*out_length`
/// always receives the length without the NUL, so a host can size a buffer
/// with a first call that passes `capacity` 0.
unsafe fn write_text(
    text: &str,
    buffer: *mut c_char,
    capacity: usize,
    out_length: *mut usize,
) -> anyhow::Result<()> {
    write(out_length, text.len())?;
    if !buffer.is_null() && capacity > text.len() {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buffer.cast::<u8>(), text.len());
        *buffer.add(text.len()) = 0;
    }
    Ok(())
}

unsafe fn text<'a>(
    p: *const c_char,
    what: &'static str,
) -> anyhow::Result<std::borrow::Cow<'a, str>> {
    if p.is_null() {
        return Err(invalid(what));
    }
    Ok(CStr::from_ptr(p).to_string_lossy())
}

// ----- world, bodies, shapes, joints -----------------------------------------

/// Change gravity (the world's coordinates and units).
#[no_mangle]
pub unsafe extern "C" fn rbWorld_SetGravity(world: *mut RbWorld, gravity: RbVec3) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let g = w.coords.linear_in(gravity.a());
        w.world.set_gravity(g)
    })
}

/// A body's velocities now. **Blocks** on a GPU readback: for tools and
/// occasional queries, not per-frame use (moves carry poses every frame).
#[no_mangle]
pub unsafe extern "C" fn rbBody_GetVelocity(
    world: *mut RbWorld,
    body: RbBodyId,
    out_linear: *mut RbVec3,
    out_angular: *mut RbVec3,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let s = w.world.read_body_state(BodyId::from_bits(body))?;
        write(
            out_linear,
            RbVec3::of(w.coords.linear_out(s.linear_velocity)),
        )?;
        write(
            out_angular,
            RbVec3::of(w.coords.angular_velocity_out(s.angular_velocity)),
        )
    })
}

/// Move a kinematic body (`RB_KINEMATIC_BODY`) to `transform` over the next
/// step: the velocity that gets it there is computed on the GPU from where it
/// is when the step starts, so animation-driven bodies track without drift.
/// It pushes and carries dynamic bodies on the way.
#[no_mangle]
pub unsafe extern "C" fn rbBody_SetTargetTransform(
    world: *mut RbWorld,
    body: RbBodyId,
    transform: RbTransform,
) -> RbResult {
    body_command(world, body, |w, id| {
        let t = w.transform_in(transform);
        w.world.body_set_target_transform(id, t)
    })
}

/// Box3D damping (1/s): velocities shrink by `1 / (1 + h c)` each step.
#[no_mangle]
pub unsafe extern "C" fn rbBody_SetDamping(
    world: *mut RbWorld,
    body: RbBodyId,
    linear: f32,
    angular: f32,
) -> RbResult {
    body_command(world, body, |w, id| {
        w.world.body_set_damping(id, linear, angular)
    })
}

/// Multiply gravity for this body (it wakes).
#[no_mangle]
pub unsafe extern "C" fn rbBody_SetGravityScale(
    world: *mut RbWorld,
    body: RbBodyId,
    scale: f32,
) -> RbResult {
    body_command(world, body, |w, id| {
        w.world.body_set_gravity_scale(id, scale)
    })
}

/// `RB_LOCK_*` bits (host axes).
#[no_mangle]
pub unsafe extern "C" fn rbBody_SetMotionLocks(
    world: *mut RbWorld,
    body: RbBodyId,
    locks: u32,
) -> RbResult {
    body_command(world, body, |w, id| {
        let bits = w.coords.locks_in(locks);
        w.world
            .body_set_motion_locks(id, MotionLocks::from_bits(bits))
    })
}

/// Mass, centre of mass (body frame) and rotational inertia about the centre
/// (body frame, rows), Box3D `b3MassData`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RbMassData {
    pub mass: f32,
    pub center: RbVec3,
    pub rotational_inertia: [RbVec3; 3],
}

/// The body's mass data (zero for static and kinematic bodies).
#[no_mangle]
pub unsafe extern "C" fn rbBody_GetMassData(
    world: *mut RbWorld,
    body: RbBodyId,
    out: *mut RbMassData,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let m = w.world.body_mass_data(BodyId::from_bits(body))?;
        let c = w.coords;
        let out_data = m.map_or(RbMassData::default(), |m| {
            let t = c.inertia_out(m.rotational_inertia);
            RbMassData {
                mass: m.mass,
                center: RbVec3::of(c.position_out(m.center)),
                rotational_inertia: t.map(RbVec3::of),
            }
        });
        write(out, out_data)
    })
}

/// Override a dynamic body's mass data (Box3D `b3Body_SetMassData`).
#[no_mangle]
pub unsafe extern "C" fn rbBody_SetMassData(
    world: *mut RbWorld,
    body: RbBodyId,
    mass: RbMassData,
) -> RbResult {
    body_command(world, body, |w, id| {
        let c = w.coords;
        let data = MassData {
            mass: mass.mass,
            center: c.position_in(mass.center.a()),
            rotational_inertia: c.inertia_in(mass.rotational_inertia.map(|r| r.a())),
        };
        w.world.body_set_mass_data(id, data)
    })
}

#[no_mangle]
pub unsafe extern "C" fn rbBody_GetType(
    world: *mut RbWorld,
    body: RbBodyId,
    out: *mut RbBodyType,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let t = match w.world.body_type(BodyId::from_bits(body))? {
            BodyType::Static => RbBodyType::RB_STATIC_BODY,
            BodyType::Kinematic => RbBodyType::RB_KINEMATIC_BODY,
            BodyType::Dynamic => RbBodyType::RB_DYNAMIC_BODY,
        };
        write(out, t)
    })
}

#[no_mangle]
pub unsafe extern "C" fn rbBody_GetUserData(
    world: *mut RbWorld,
    body: RbBodyId,
    out: *mut u64,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        write(out, w.world.body_user_data(BodyId::from_bits(body))?)
    })
}

/// GPU time of the last step that ran (ms; 0 where timestamps are off).
#[no_mangle]
pub unsafe extern "C" fn rbWorld_GetProfileMs(world: *mut RbWorld, out: *mut f64) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        write(out, w.world.profile_ms())
    })
}

/// Awake as of the last polled frame.
#[no_mangle]
pub unsafe extern "C" fn rbBody_IsAwake(
    world: *mut RbWorld,
    body: RbBodyId,
    out: *mut bool,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        write(out, w.world.body_is_awake(BodyId::from_bits(body))?)
    })
}

/// Change a joint's drive in place: the motor speed and the spring or servo
/// target. Revolute and spherical joints take rad/s and rad; prismatic and
/// distance joints take speed and length.
#[no_mangle]
pub unsafe extern "C" fn rbJoint_SetDrive(
    world: *mut RbWorld,
    joint: RbJointId,
    velocity: f32,
    target: f32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let (v, t) = if w.linear_joints.contains(&joint) {
            (w.coords.length_in(velocity), w.coords.length_in(target))
        } else {
            (
                w.coords.joint_angle_in(velocity),
                w.coords.joint_angle_in(target),
            )
        };
        w.world.joint_set_drive(JointId::from_bits(joint), v, t)
    })
}

#[no_mangle]
pub unsafe extern "C" fn rbShape_SetFriction(
    world: *mut RbWorld,
    shape: RbShapeId,
    friction: f32,
    restitution: f32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let id = ShapeId::from_bits(shape);
        let mut m = w.world.shape_surface_material(id)?;
        m.friction = friction;
        m.restitution = restitution;
        w.world.shape_set_surface_material(id, m)
    })
}

#[no_mangle]
pub unsafe extern "C" fn rbShape_SetFilter(
    world: *mut RbWorld,
    shape: RbShapeId,
    category_bits: u16,
    mask_bits: u16,
    group_index: i16,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let id = ShapeId::from_bits(shape);
        let mut f = w.world.shape_filter(id)?;
        f.categories = category_bits;
        f.mask = mask_bits;
        f.group = group_index;
        w.world.shape_set_filter(id, f)
    })
}

// ----- vehicles ----------------------------------------------------------------

/// A preset's descriptor as JSON (to edit, then `rbCreateVehicleFromJson`).
#[no_mangle]
pub unsafe extern "C" fn rbGetVehiclePresetJson(
    name: *const c_char,
    buffer: *mut c_char,
    capacity: usize,
    out_length: *mut usize,
) -> RbResult {
    guard_args(|| {
        let name = text(name, "null preset name")?;
        let desc = engine_vehicle::presets::by_name(&name)
            .ok_or_else(|| anyhow::anyhow!("no vehicle preset {name:?}"))?;
        write_text(&serde_json::to_string(&desc)?, buffer, capacity, out_length)
    })
}

/// A vehicle from a `VehicleDesc` as JSON (the presets' format).
#[no_mangle]
pub unsafe extern "C" fn rbCreateVehicleFromJson(
    world: *mut RbWorld,
    json: *const c_char,
    transform: RbTransform,
    out: *mut RbVehicleId,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let desc: VehicleDesc = serde_json::from_str(&text(json, "null vehicle json")?)?;
        let t = w.transform_in(transform);
        let h = w.world.add_vehicle(&desc, t.position, t.rotation)?;
        write(out, h.0)
    })
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbVehicleDamageType {
    /// The wheel falls off as a free body.
    RB_DETACH_WHEEL = 0,
    /// The wheel stays on but stops gripping, driving and steering.
    RB_DISABLE_WHEEL = 1,
    /// The part becomes a free body.
    RB_DETACH_PART = 2,
    /// The part stays on but its function is lost.
    RB_DISABLE_PART = 3,
}

/// Sever or switch off a wheel or part (by its index in the descriptor).
#[no_mangle]
pub unsafe extern "C" fn rbVehicle_Damage(
    world: *mut RbWorld,
    vehicle: RbVehicleId,
    type_: RbVehicleDamageType,
    index: u32,
) -> RbResult {
    guard_args(|| {
        let d = match type_ {
            RbVehicleDamageType::RB_DETACH_WHEEL => VehicleDamage::DetachWheel(index),
            RbVehicleDamageType::RB_DISABLE_WHEEL => VehicleDamage::DisableWheel(index),
            RbVehicleDamageType::RB_DETACH_PART => VehicleDamage::DetachPart(index),
            RbVehicleDamageType::RB_DISABLE_PART => VehicleDamage::DisablePart(index),
        };
        world_mut(world)?
            .world
            .vehicle_damage(VehicleHandle(vehicle), d)
    })
}

/// Functional overrides, ANDed with what the intact parts allow (bit i of a
/// mask: wheel i).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbFunctionalState {
    pub wheel_mask: u32,
    pub drive_mask: u32,
    pub steer_mask: u32,
    pub brake_mask: u32,
    pub engine_connected: bool,
}

#[no_mangle]
pub extern "C" fn rbDefaultFunctionalState() -> RbFunctionalState {
    let s = FunctionalState::default();
    RbFunctionalState {
        wheel_mask: s.wheel_mask,
        drive_mask: s.drive_mask,
        steer_mask: s.steer_mask,
        brake_mask: s.brake_mask,
        engine_connected: s.engine_connected,
    }
}

#[no_mangle]
pub unsafe extern "C" fn rbVehicle_SetFunctionalState(
    world: *mut RbWorld,
    vehicle: RbVehicleId,
    state: RbFunctionalState,
) -> RbResult {
    guard_args(|| {
        world_mut(world)?.world.vehicle_set_functional_state(
            VehicleHandle(vehicle),
            FunctionalState {
                wheel_mask: state.wheel_mask,
                drive_mask: state.drive_mask,
                steer_mask: state.steer_mask,
                brake_mask: state.brake_mask,
                engine_connected: state.engine_connected,
            },
        )
    })
}

/// One breakable mount's load against its limits.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RbMountLoad {
    pub index: u32,
    /// A wheel's mount (else a part's).
    pub wheel: bool,
    pub force: f32,
    pub torque: f32,
    pub break_force: f32,
    pub break_torque: f32,
    /// Largest max(force / break_force, torque / break_torque) since spawn.
    pub peak_ratio: f32,
}

/// Mount loads (**blocks** on a readback; for damage HUDs and tuning).
/// Writes up to `capacity`; `*out_count` gets the total.
#[no_mangle]
pub unsafe extern "C" fn rbVehicle_GetMountLoads(
    world: *mut RbWorld,
    vehicle: RbVehicleId,
    out: *mut RbMountLoad,
    capacity: u32,
    out_count: *mut u32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let loads = w.world.vehicle_mount_loads(VehicleHandle(vehicle))?;
        let c = w.coords;
        write(out_count, loads.len() as u32)?;
        if capacity > 0 && out.is_null() {
            return Err(invalid("null mount load array"));
        }
        for (k, l) in loads.iter().take(capacity as usize).enumerate() {
            *out.add(k) = RbMountLoad {
                index: l.index,
                wheel: l.wheel,
                force: c.force_out(l.force),
                torque: c.torque_out(l.torque),
                break_force: c.force_out(l.break_force),
                break_torque: c.torque_out(l.break_torque),
                peak_ratio: l.peak_ratio,
            };
        }
        Ok(())
    })
}

// ----- soft bodies -----------------------------------------------------------------

/// A built-in soft vehicle's descriptor as JSON: `"procedural_car"` (the
/// softbody page's hatchback; `strength` scales its beams) or `"dune_buggy"`
/// (soft frame with bolted rigid parts; `strength` scales its mounts). 0 or
/// less: the default.
#[no_mangle]
pub unsafe extern "C" fn rbGetSoftVehiclePresetJson(
    name: *const c_char,
    strength: f32,
    buffer: *mut c_char,
    capacity: usize,
    out_length: *mut usize,
) -> RbResult {
    guard_args(|| {
        let name = text(name, "null preset name")?;
        let desc = match name.as_ref() {
            "procedural_car" => {
                let d = engine_softbody::procedural::CarParams::default();
                let p = engine_softbody::procedural::CarParams {
                    strength: if strength > 0.0 { strength } else { d.strength },
                    ..d
                };
                engine_softbody::procedural::car_desc(&p).0
            }
            "dune_buggy" => {
                engine_softbody::buggy::dune_buggy_soft_with(&engine_softbody::buggy::BuggyBeams {
                    mount_scale: if strength > 0.0 { strength } else { 1.0 },
                    ..Default::default()
                })
            }
            other => anyhow::bail!("no soft vehicle preset {other:?}"),
        };
        write_text(&serde_json::to_string(&desc)?, buffer, capacity, out_length)
    })
}

unsafe fn spawn_soft(
    w: &mut RbWorld,
    desc: &SoftVehicleDesc,
    options: *const c_char,
    transform: RbTransform,
    velocity: RbVec3,
) -> anyhow::Result<SoftHandle> {
    let mut opts: SpawnOptions = if options.is_null() {
        SpawnOptions::default()
    } else {
        serde_json::from_str(&text(options, "options")?)?
    };
    opts.velocity = w.coords.linear_in(velocity.a());
    let t = w.transform_in(transform);
    w.world
        .add_soft_vehicle(desc, t.position, t.rotation, &opts)
}

/// A soft vehicle from a `SoftVehicleDesc` as JSON. `options` is a
/// `SpawnOptions` JSON (`rigid_core`, node radius, suspension) or null for
/// the defaults; `velocity` is the initial velocity of every node.
#[no_mangle]
pub unsafe extern "C" fn rbCreateSoftVehicle(
    world: *mut RbWorld,
    desc_json: *const c_char,
    options_json: *const c_char,
    transform: RbTransform,
    velocity: RbVec3,
    out: *mut RbSoftId,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let desc: SoftVehicleDesc = serde_json::from_str(&text(desc_json, "null soft json")?)?;
        let h = spawn_soft(w, &desc, options_json, transform, velocity)?;
        write(out, h.0)
    })
}

/// A soft vehicle from BeamNG JBeam files (`names[i]` and `contents[i]`,
/// `count` of them) and an optional part configuration (`.pc` text or null).
#[no_mangle]
pub unsafe extern "C" fn rbCreateSoftVehicleJBeam(
    world: *mut RbWorld,
    names: *const *const c_char,
    contents: *const *const c_char,
    count: u32,
    config: *const c_char,
    options_json: *const c_char,
    transform: RbTransform,
    velocity: RbVec3,
    out: *mut RbSoftId,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        if count == 0 || names.is_null() || contents.is_null() {
            return Err(invalid("no jbeam files"));
        }
        let mut files = Vec::with_capacity(count as usize);
        for k in 0..count as usize {
            files.push((
                text(*names.add(k), "null jbeam name")?.into_owned(),
                text(*contents.add(k), "null jbeam contents")?.into_owned(),
            ));
        }
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(n, c)| (n.as_str(), c.as_str()))
            .collect();
        let pc = if config.is_null() {
            None
        } else {
            Some(text(config, "config")?.into_owned())
        };
        let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref())
            .map_err(|e| anyhow::anyhow!("jbeam: {e}"))?;
        let h = spawn_soft(w, &desc, options_json, transform, velocity)?;
        write(out, h.0)
    })
}

/// The demos' soft-solver settings for a step rate (`SoftSettings` JSON):
/// edit `rigid_mode`, `threshold_scale`, `sweeps` and the rest, then
/// `rbWorld_SetSoftSettingsJson`.
#[no_mangle]
pub unsafe extern "C" fn rbGetCalibratedSoftSettingsJson(
    hz: f32,
    buffer: *mut c_char,
    capacity: usize,
    out_length: *mut usize,
) -> RbResult {
    guard_args(|| {
        let s = SoftSettings::calibrated_for(hz);
        write_text(&serde_json::to_string(&s)?, buffer, capacity, out_length)
    })
}

#[no_mangle]
pub unsafe extern "C" fn rbWorld_SetSoftSettingsJson(
    world: *mut RbWorld,
    json: *const c_char,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let s: SoftSettings = serde_json::from_str(&text(json, "null settings json")?)?;
        w.world.set_soft_settings(s)
    })
}

/// Let soft bodies' skins collide with this rigid shape (walls, poles).
#[no_mangle]
pub unsafe extern "C" fn rbSoft_AddObstacle(world: *mut RbWorld, shape: RbShapeId) -> RbResult {
    guard_args(|| {
        world_mut(world)?
            .world
            .soft_add_obstacle(ShapeId::from_bits(shape))
    })
}

/// One beam of a soft body (`rbSoft_GetBeams`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RbSoftBeam {
    /// Node ordinals (index `RbSoftOutput::nodes`).
    pub node_a: u32,
    pub node_b: u32,
    /// Rest length now (plastic deformation moves it) and at spawn.
    pub rest_length: f32,
    pub spawn_length: f32,
    /// Signed axial force (+ = compression).
    pub force: f32,
    pub broken: bool,
    pub yielded: bool,
    pub visible: bool,
}

/// Every beam (**blocks** on a readback; for beam views and damage maps).
/// Writes up to `capacity`; `*out_count` gets the total.
#[no_mangle]
pub unsafe extern "C" fn rbSoft_GetBeams(
    world: *mut RbWorld,
    soft: RbSoftId,
    out: *mut RbSoftBeam,
    capacity: u32,
    out_count: *mut u32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let beams = w.world.soft_beams(SoftHandle(soft))?;
        let c = w.coords;
        write(out_count, beams.len() as u32)?;
        if capacity > 0 && out.is_null() {
            return Err(invalid("null beam array"));
        }
        for (k, b) in beams.iter().take(capacity as usize).enumerate() {
            *out.add(k) = RbSoftBeam {
                node_a: b.node_a,
                node_b: b.node_b,
                rest_length: c.length_out(b.rest_length),
                spawn_length: c.length_out(b.spawn_length),
                force: c.force_out(b.force),
                broken: b.broken,
                yielded: b.yielded,
                visible: b.visible,
            };
        }
        Ok(())
    })
}

/// A soft body's damage so far.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RbSoftDamage {
    /// Beams that have yielded at least once, broken, and in total.
    pub bent: u32,
    pub broken: u32,
    pub beams: u32,
    /// Energy the deformation absorbed.
    pub plastic_work: f32,
    /// Largest permanent beam strain, |L/L0 - 1|.
    pub max_beam_strain: f32,
    /// Riding its rigid core right now (rigid mode).
    pub rigid: bool,
}

/// Damage (**blocks** on a readback; for HUDs and scoring).
#[no_mangle]
pub unsafe extern "C" fn rbSoft_GetDamage(
    world: *mut RbWorld,
    soft: RbSoftId,
    out: *mut RbSoftDamage,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let h = SoftHandle(soft);
        let rigid = w.world.soft_is_rigid(h)?;
        let d = w.world.soft_damage(h);
        write(
            out,
            RbSoftDamage {
                bent: d.bent,
                broken: d.broken,
                beams: d.beams,
                plastic_work: w.coords.energy_out(d.plastic_work),
                max_beam_strain: d.max_beam_strain,
                rigid,
            },
        )
    })
}

// ----- destructibles ------------------------------------------------------------

/// The fracture lab's procedural cuboid house (box chunks, default options).
#[no_mangle]
pub unsafe extern "C" fn rbCreateDestructibleAssetCuboidHouse(
    world: *mut RbWorld,
    out: *mut RbDestructibleAssetId,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let desc = DestructibleAssetDesc::cuboid_house(&Default::default())?;
        let id = w.world.create_destructible_asset(&desc)?;
        write(out, id.0)
    })
}

/// The fracture lab's bond preloading: settle one copy of the asset on a
/// ground plane at its frame's y = 0 with breaking off (blocks, up to 600
/// steps); instances placed afterwards start asleep at the settled poses.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_PreloadDestructibleAsset(
    world: *mut RbWorld,
    asset: RbDestructibleAssetId,
    out_settled: *mut bool,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let settled = w
            .world
            .preload_destructible_asset(DestructibleAssetId(asset))?;
        write(out_settled, settled)
    })
}

/// Switch the world to the solver settings the asset was tuned with (the
/// fracture lab's: 120 Hz, 8 iterations, its contact and sleep flags; gravity
/// included). `*out_time_step` receives the new fixed step to step at.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_UseDestructibleAssetSolver(
    world: *mut RbWorld,
    asset: RbDestructibleAssetId,
    out_time_step: *mut f32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let solver = w
            .world
            .destructible_asset(DestructibleAssetId(asset))?
            .solver
            .ok_or_else(|| anyhow::anyhow!("the asset carries no solver settings"))?;
        w.world.set_solver(solver);
        write(out_time_step, solver.dt)
    })
}

/// Switch the world to the solver settings the asset's source was tuned with for
/// `model`: `rbWorld_UseDestructibleAssetSolver`'s for the bonds model; for the
/// threshold model the fracture lab's threshold mode's (never asleep, 4
/// iterations, 60 Hz; gravity included). `*out_time_step` receives the step.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_UseDestructibleAssetSolverFor(
    world: *mut RbWorld,
    asset: RbDestructibleAssetId,
    model: RbStrengthModel,
    out_time_step: *mut f32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let solver = w
            .world
            .destructible_asset(DestructibleAssetId(asset))?
            .solver_for(strength_model(model))
            .ok_or_else(|| anyhow::anyhow!("the asset carries no solver settings for {model:?}"))?;
        w.world.set_solver(solver);
        write(out_time_step, solver.dt)
    })
}

/// The ground the asset's stress scene stands on (the fracture lab's ground
/// plane for the threshold model, which the host adds itself): a point on it in
/// the world's coordinates for an instance placed at the identity.
/// `*out_has_ground` is false when the scene has none.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_GetDestructibleAssetGround(
    world: *mut RbWorld,
    asset: RbDestructibleAssetId,
    out_has_ground: *mut bool,
    out_point: *mut RbVec3,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let height = w
            .world
            .destructible_asset(DestructibleAssetId(asset))?
            .stress
            .as_ref()
            .and_then(|s| s.ground_height());
        write(out_has_ground, height.is_some())?;
        write(
            out_point,
            RbVec3::of(w.coords.position_out([0.0, height.unwrap_or(0.0), 0.0])),
        )
    })
}

/// A projectile's material for threshold destructibles (`rbBody_SetProjectile`):
/// Young's modulus (Pa) and Poisson's ratio, in SI units in every coordinate
/// convention. The default is the fracture lab's steel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbProjectile {
    pub youngs_modulus: f32,
    pub poisson_ratio: f32,
}

#[no_mangle]
pub extern "C" fn rbDefaultProjectile() -> RbProjectile {
    let p = Projectile::default();
    RbProjectile {
        youngs_modulus: p.youngs_modulus,
        poisson_ratio: p.poisson_ratio,
    }
}

/// Mark a dynamic body with one sphere or box shape as a projectile for
/// threshold destructibles (null clears it): its contacts load chunks as the
/// lab's shots do and strike prediction checks it every step. At most 64.
#[no_mangle]
pub unsafe extern "C" fn rbBody_SetProjectile(
    world: *mut RbWorld,
    body: RbBodyId,
    projectile: *const RbProjectile,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let p = (!projectile.is_null()).then(|| {
            let p = *projectile;
            Projectile {
                youngs_modulus: p.youngs_modulus,
                poisson_ratio: p.poisson_ratio,
            }
        });
        w.world.body_set_projectile(BodyId::from_bits(body), p)
    })
}

/// An infinite ground plane on a static body: the half-space below
/// `normal · x = offset` in the body frame. At most 8 per world.
#[no_mangle]
pub unsafe extern "C" fn rbCreatePlaneShape(
    world: *mut RbWorld,
    body: RbBodyId,
    def: *const RbShapeDef,
    normal: RbVec3,
    offset: f32,
    out: *mut RbShapeId,
) -> RbResult {
    create_shape(world, body, def, out, |w| {
        Ok(Geometry::Plane {
            normal: w.coords.direction_in(normal.a()),
            offset: w.coords.length_in(offset),
        })
    })
}

// ----- queries and explosions -------------------------------------------------------

/// Shapes whose bounds overlap the box `min..max` at the next step's final
/// poses (Box3D `b3World_OverlapAABB`; sensors are not reported), filtered as
/// Box3D's query filter: a shape is reported when its categories meet
/// `mask_bits` and its mask meets `category_bits`. The result arrives in that
/// step's frame (`RbFrame::overlaps`, matched by `*out_query`).
#[no_mangle]
pub unsafe extern "C" fn rbWorld_OverlapAABB(
    world: *mut RbWorld,
    min: RbVec3,
    max: RbVec3,
    category_bits: u16,
    mask_bits: u16,
    out_query: *mut u64,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let (lo, hi) = w.coords.box_in(min.a(), max.a());
        let filter = QueryFilter {
            categories: category_bits,
            mask: mask_bits,
        };
        let id = w.world.overlap_aabb(lo, hi, filter)?;
        write(out_query, id.0)
    })
}

/// A radial impulse (Box3D `b3ExplosionDef`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbExplosionDef {
    pub position: RbVec3,
    /// Full strength within this distance of a shape's surface.
    pub radius: f32,
    /// Then fading linearly to zero over this distance.
    pub falloff: f32,
    /// Impulse per unit of a shape's cross-section.
    pub impulse_per_area: f32,
    pub mask_bits: u16,
}

/// Defaults in `coordinates`' units.
#[no_mangle]
pub extern "C" fn rbDefaultExplosionDef(coordinates: RbCoordinates) -> RbExplosionDef {
    let c = Coords::new(coordinates);
    let d = ExplosionDef::default();
    RbExplosionDef {
        position: RbVec3::default(),
        radius: c.length_out(d.radius),
        falloff: c.length_out(d.falloff),
        impulse_per_area: c.impulse_per_area_out(d.impulse_per_area),
        mask_bits: d.mask,
    }
}

/// A radial impulse at the start of the next step (Box3D `b3World_Explode`);
/// bodies it reaches wake. Up to 16 per step.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_Explode(
    world: *mut RbWorld,
    def: *const RbExplosionDef,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let d = read(def, "null explosion def")?;
        let c = w.coords;
        w.world.explode(&ExplosionDef {
            position: c.position_in(d.position.a()),
            radius: c.length_in(d.radius),
            falloff: c.length_in(d.falloff),
            impulse_per_area: c.impulse_per_area_in(d.impulse_per_area),
            mask: d.mask_bits,
        })
    })
}

/// A charge (`rbWorld_Detonate`; engine/docs/destruction-reactions.md section 3).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbBlastDef {
    pub position: RbVec3,
    /// TNT-equivalent mass (kg), > 0.
    pub tnt_kg: f32,
    /// Also push bodies with `rbWorld_Explode`'s radial impulse from the same point
    /// (`radius`, `falloff`, `impulse_per_area`, `mask_bits` as `RbExplosionDef`'s). It
    /// reaches destructible pieces too unless their category is left out of the mask.
    pub radial: bool,
    pub radius: f32,
    pub falloff: f32,
    pub impulse_per_area: f32,
    pub mask_bits: u16,
    pub internal_value: u32,
}

/// Defaults in `coordinates`' units: 1 kg of TNT at the origin, no radial impulse.
#[no_mangle]
pub extern "C" fn rbDefaultBlastDef(coordinates: RbCoordinates) -> RbBlastDef {
    let e = rbDefaultExplosionDef(coordinates);
    let d = BlastDef::default();
    RbBlastDef {
        position: RbVec3::default(),
        tnt_kg: d.tnt_kg,
        radial: false,
        radius: e.radius,
        falloff: e.falloff,
        impulse_per_area: e.impulse_per_area,
        mask_bits: e.mask_bits,
        internal_value: COOKIE,
    }
}

/// Detonate a charge at the start of the next step: every destructible placed with
/// `blast_loading` takes its face pressures over the steps its pulse lasts, breaking the
/// bonds it overloads (`RB_BREAK_BLAST`) and, with crushing, crushing the chunks it
/// overstresses (`RB_CRUSH_BLAST`). Forward only. Errors: a non-positive or non-finite
/// charge, a loaded face nearer than the fits cover, more than 16 blasts at once.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_Detonate(world: *mut RbWorld, def: *const RbBlastDef) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let d = read(def, "null blast def")?;
        check_cookie(d.internal_value, "blast def not from rbDefaultBlastDef")?;
        let c = w.coords;
        w.world.detonate(&BlastDef {
            position: c.position_in(d.position.a()),
            tnt_kg: d.tnt_kg,
            radial: d.radial.then(|| RadialImpulse {
                radius: c.length_in(d.radius),
                falloff: c.length_in(d.falloff),
                impulse_per_area: c.impulse_per_area_in(d.impulse_per_area),
                mask: d.mask_bits,
            }),
        })
    })
}

// ----- debug draw ------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RbColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// Callbacks for `rbWorld_DebugDraw` (null entries are skipped). Everything
/// is in the world's coordinates; `context` is passed back unchanged.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RbDebugDraw {
    pub context: *mut c_void,
    pub draw_box: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            transform: RbTransform,
            half: RbVec3,
            color: RbColor,
        ),
    >,
    pub draw_sphere: Option<
        unsafe extern "C" fn(context: *mut c_void, center: RbVec3, radius: f32, color: RbColor),
    >,
    pub draw_capsule: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            p0: RbVec3,
            p1: RbVec3,
            radius: f32,
            color: RbColor,
        ),
    >,
    /// A convex hull: `vertices` in its own frame placed by `transform`,
    /// `indices` a triangle list in the world's winding.
    pub draw_mesh: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            transform: RbTransform,
            vertices: *const RbVec3,
            vertex_count: u32,
            indices: *const u32,
            index_count: u32,
            color: RbColor,
        ),
    >,
    pub draw_transform: Option<unsafe extern "C" fn(context: *mut c_void, transform: RbTransform)>,
    pub draw_shapes: bool,
    pub draw_body_transforms: bool,
}

struct CDraw<'a> {
    draw: &'a RbDebugDraw,
    coords: Coords,
}

fn color(c: DebugColor) -> RbColor {
    RbColor {
        r: c[0],
        g: c[1],
        b: c[2],
        a: c[3],
    }
}

impl CDraw<'_> {
    fn t(&self, t: Transform) -> RbTransform {
        RbTransform {
            p: RbVec3::of(self.coords.position_out(t.position)),
            q: RbQuat::of(self.coords.rotation_out(t.rotation)),
        }
    }
}

impl DebugDraw for CDraw<'_> {
    fn draw_box(&mut self, transform: Transform, half: [f32; 3], c: DebugColor) {
        if let Some(f) = self.draw.draw_box {
            // Body-frame axes map like world ones, so the extents permute.
            let h = self.coords.extents_out(half);
            unsafe {
                f(
                    self.draw.context,
                    self.t(transform),
                    RbVec3::of(h),
                    color(c),
                )
            }
        }
    }
    fn draw_sphere(&mut self, center: [f32; 3], radius: f32, c: DebugColor) {
        if let Some(f) = self.draw.draw_sphere {
            let p = RbVec3::of(self.coords.position_out(center));
            unsafe {
                f(
                    self.draw.context,
                    p,
                    self.coords.length_out(radius),
                    color(c),
                )
            }
        }
    }
    fn draw_capsule(&mut self, p0: [f32; 3], p1: [f32; 3], radius: f32, c: DebugColor) {
        if let Some(f) = self.draw.draw_capsule {
            let (a, b) = (
                RbVec3::of(self.coords.position_out(p0)),
                RbVec3::of(self.coords.position_out(p1)),
            );
            unsafe {
                f(
                    self.draw.context,
                    a,
                    b,
                    self.coords.length_out(radius),
                    color(c),
                )
            }
        }
    }
    fn draw_mesh(&mut self, transform: Transform, mesh: &ShapeMesh, c: DebugColor) {
        if let Some(f) = self.draw.draw_mesh {
            let v: Vec<RbVec3> = mesh
                .vertices
                .iter()
                .map(|p| RbVec3::of(self.coords.position_out(*p)))
                .collect();
            let i: Vec<u32> = mesh
                .triangles
                .iter()
                .flat_map(|t| self.coords.triangle_out(*t))
                .collect();
            unsafe {
                f(
                    self.draw.context,
                    self.t(transform),
                    v.as_ptr(),
                    v.len() as u32,
                    i.as_ptr(),
                    i.len() as u32,
                    color(c),
                )
            }
        }
    }
    fn draw_transform(&mut self, transform: Transform) {
        if let Some(f) = self.draw.draw_transform {
            unsafe { f(self.draw.context, self.t(transform)) }
        }
    }
}

/// Draw every shape at its body's last polled pose, coloured by state
/// (static, awake, asleep) unless a material colour is set.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_DebugDraw(
    world: *mut RbWorld,
    draw: *const RbDebugDraw,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let d = read(draw, "null debug draw")?;
        let mut c = CDraw {
            draw: d,
            coords: w.coords,
        };
        w.world.draw(
            &mut c,
            DebugDrawOptions {
                shapes: d.draw_shapes,
                body_transforms: d.draw_body_transforms,
                ..DebugDrawOptions::default()
            },
        );
        Ok(())
    })
}
