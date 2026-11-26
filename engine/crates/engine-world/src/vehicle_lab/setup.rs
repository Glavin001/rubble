//! The lab's scene fixtures, car placement and World settings (split from `mod.rs`).

use super::*;

/// The launch direction that carries a projectile of `speed` (m/s), falling under gravity,
/// from the origin through `to` (the low trajectory), and whether the speed reaches it
/// (if not: straight at it).
pub fn ballistic(to: DVec3, speed: f32) -> (Vec3, bool) {
    let g = f64::from(-GRAVITY);
    let (dx, dy) = ((to.x * to.x + to.z * to.z).sqrt(), to.y);
    let v2 = f64::from(speed) * f64::from(speed);
    let disc = v2 * v2 - g * (g * dx * dx + 2.0 * dy * v2);
    if dx < 1e-6 || disc < 0.0 || speed <= 0.0 {
        return (to.as_vec3().normalize_or_zero(), dx < 1e-6 && speed > 0.0);
    }
    let tan = (v2 - disc.sqrt()) / (g * dx);
    let h = DVec3::new(to.x / dx, 0.0, to.z / dx);
    ((h + DVec3::Y * tan).normalize().as_vec3(), true)
}

/// The flat arena's fixtures, in the car's own solver (its raycast wheels see them, and
/// its nodes get the solver's static contact): a ramp to the car's left and a kerb to its
/// right, 10 m ahead of its start.
pub fn arena_fixtures() -> Vec<(Vec3, Vec3, Quat)> {
    let (len, rise) = (6.0f32, 1.0f32);
    let tilt = Quat::from_rotation_x(-(rise / len).atan());
    let half = Vec3::new(1.6, 0.2, 0.5 * (len * len + rise * rise).sqrt());
    let centre = Vec3::new(14.0, GROUND + 0.5 * rise, 12.0 + 0.5 * len) - tilt * Vec3::Y * 0.2;
    vec![
        (centre, half, tilt),
        (
            Vec3::new(-14.0, GROUND + 0.06, 12.0),
            Vec3::new(3.0, 0.06, 0.2),
            Quat::IDENTITY,
        ),
    ]
}

/// The scene's fixtures (in the cars' solver and as World fixture bodies): the flat
/// arena's ramp and kerb (far overhead on the mountain), or the derby's walls and ramps.
pub fn scene_fixtures(opts: &LabOptions) -> Vec<(Vec3, Vec3, Quat)> {
    if opts.scene == Scene::Derby {
        derby::arena_fixtures(opts.derby.obstacles)
    } else {
        arena_fixtures()
    }
}

/// Where and how the car starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Start {
    /// Speed along its heading (m/s) and to its left (m/s: a sled's sideways launch).
    pub speed: f32,
    pub lateral: f32,
    /// Its heading, turned about +y from +z (rad).
    pub yaw: f32,
    /// Its centre (x, z) and how far above its standing height it starts (m: on a sled
    /// its wheels do not reach the ground until it has dropped).
    pub at: [f32; 2],
    pub lift: f32,
}

impl Default for Start {
    fn default() -> Self {
        Self {
            speed: 0.0,
            lateral: 0.0,
            yaw: 0.0,
            at: [0.0, 0.0],
            lift: 0.0,
        }
    }
}

/// The lab's car from `template` for `opts`, started as `start` (none: the scene's own
/// start: standing on the flat arena, or just off the mountain's lip at its launch
/// speed): placed, its ground and fixtures set, surface contact as asked. Every lab car
/// of one template has the same layout (the same static shapes: the arena's fixtures,
/// far overhead on the mountain), so a restart is `World::reset_explicit_vehicles`.
pub fn lab_car(template: &ExplicitBeams, opts: &LabOptions, start: Option<Start>) -> ExplicitBeams {
    let mountain = opts.scene == Scene::Mountain;
    let ground = if mountain { -1.0e5 } else { GROUND };
    // Facing +z, standing on y = 0, centred on its node box.
    let mut car = place_vehicle(template, Vec3::ZERO, 0, 0.0, 0.0);
    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
    for n in &car.nodes {
        lo = lo.min(n.pos);
        hi = hi.max(n.pos);
    }
    let mid = 0.5 * (lo + hi);
    let mountain_start = Start {
        speed: opts.mountain.launch,
        at: [0.0, 0.5 * (hi.z - lo.z) + 0.3],
        lift: 0.3,
        ..Start::default()
    };
    let start = start.unwrap_or(if mountain {
        // Just off the lip (its rear clear of it), at the launch speed.
        Start {
            speed: opts.mountain.launch,
            at: [0.0, 0.5 * (hi.z - lo.z) + 0.3],
            lift: 0.3,
            ..Start::default()
        }
    } else {
        Start::default()
    });
    let turn = Quat::from_rotation_y(start.yaw);
    let (fwd, left) = (turn * Vec3::Z, turn * Vec3::X);
    let vel = fwd * start.speed + left * start.lateral;
    let base = if mountain { 0.0 } else { GROUND };
    for n in &mut car.nodes {
        let mut p = n.pos - Vec3::new(mid.x, 0.0, mid.z);
        if start.yaw != 0.0 {
            p = turn * p;
        }
        n.pos = p + Vec3::new(start.at[0], base + start.lift, start.at[1]);
        n.vel = vel;
    }
    car.origin = Vec3::ZERO;
    if mountain && opts.mountain.start > 0.0 && start == mountain_start {
        // A run that starts down the mountain: the car 2 m over the fall line there, its
        // nodes about its own origin (as the solver's rebase leaves them).
        car.origin = (opts.mountain.line(f64::from(opts.mountain.start))
            + DVec3::new(0.0, 2.0, 0.0))
        .as_vec3();
    }
    car.ground = ground;
    if start.speed != 0.0 && start.lateral == 0.0 {
        if let Some(d) = car.drive.as_mut() {
            d.roll(start.speed);
        }
    }
    for (centre, half, rot) in scene_fixtures(opts) {
        let up = if mountain {
            Vec3::new(0.0, 1.0e4, 0.0)
        } else {
            Vec3::ZERO
        };
        car.shapes
            .push(StaticShape::oriented_box(centre + up, half, rot));
    }
    car.surface_contact = opts.surface_contact;
    car.own_travel_reach = opts.own_travel_reach;
    // (Edge contact runs on two-body dissipative contact: the BeamNG profile's.)
    car.edge_contact = opts.edge_contact && car.two_body_contact && car.dissipative_contact;
    car.vehicle_contact_range = opts.vehicle_contact_range;
    car
}

/// A node-beam structure (`LabOptions::structures`: a `.truck` text of nodes, beams and
/// `fixes`, no wheels) placed at `start`: its node box centred on `start.at` in x and z,
/// turned by `start.yaw`, its lowest node `start.lift` over the ground, moving at
/// `start.speed` along its heading (its anchored nodes still). Nothing about its physics
/// is changed: the solver steps it as any vehicle without a drive.
pub fn lab_structure(
    text: &str,
    settings: engine_softbody::explicit_beams::ExplicitBeamsSettings,
    opts: &LabOptions,
    start: Start,
) -> anyhow::Result<ExplicitBeams> {
    let mut s = ExplicitBeams::from_truck(text, Vec3::ZERO, Vec3::ZERO, settings)
        .map_err(|e| anyhow::anyhow!("vehicle lab: a structure's .truck: {e}"))?;
    anyhow::ensure!(!s.nodes.is_empty(), "vehicle lab: a structure with no nodes");
    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
    for n in &s.nodes {
        lo = lo.min(n.pos);
        hi = hi.max(n.pos);
    }
    let mid = 0.5 * (lo + hi);
    let turn = Quat::from_rotation_y(start.yaw);
    let vel = turn * Vec3::Z * start.speed;
    let base = if opts.scene == Scene::Mountain { 0.0 } else { GROUND };
    for n in &mut s.nodes {
        let p = turn * (n.pos - Vec3::new(mid.x, lo.y, mid.z));
        n.pos = p + Vec3::new(start.at[0], base + start.lift, start.at[1]);
        n.vel = if n.has(engine_softbody::explicit_beams::NODE_IMMOVABLE) { Vec3::ZERO } else { vel };
    }
    s.origin = Vec3::ZERO;
    s.surface_contact = opts.surface_contact;
    s.own_travel_reach = opts.own_travel_reach;
    s.edge_contact = opts.edge_contact && s.two_body_contact && s.dissipative_contact;
    s.vehicle_contact_range = opts.vehicle_contact_range;
    Ok(s)
}

/// The lab's World settings: the brick ring drive's solver (1/60 s steps of 4 iterations,
/// persistent contacts) for a few thousand bodies.
pub fn lab_world_def(opts: &LabOptions) -> WorldDef {
    let mut def = crate::brick_ring_drive::ring_world_def(4096);
    def.gravity = [0.0, GRAVITY, 0.0];
    // The mountain run goes down for kilometres.
    def.kill_bounds = Some(KillBounds::below(-1.0e6));
    // Eight steps' results may be in flight (3 by default): a page whose frames run
    // late, or whose readbacks arrive late, can still take every step real time owes.
    def.readback_frames = 8;
    let _ = opts;
    def.contacts = ContactStorage::Persistent;
    if opts.sleep {
        def.solver.sleep_time = 0.5;
    }
    def
}
