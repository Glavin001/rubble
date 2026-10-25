//! Two-way coupling of rigid bodies with the terrain MPM (T3, explicit and
//! staggered: `WgpuSim::step_with_terrain`). Checked against what the coupling
//! must do: momentum the sand loses at a body's surface is the body's (the pair
//! conserves it), and a heavy body set on sand is carried by the sand, not the
//! rigid bed under it.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_terrain::layout::MpmParticle;
use engine_terrain::{substeps_for, RegionPlan, SandMaterial, TerrainConfig};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Quat, Vec3};

const FRAME: f32 = 1.0 / 60.0;

fn sim(gravity: f32) -> WgpuSim {
    let mut s = WgpuSim::new(WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu adapter");
    let mut p = AvbdParams {
        dt: FRAME,
        iterations: 10,
        ..AvbdParams::default()
    };
    p.gravity_y = gravity;
    p.sleep_time = 1.0e9;
    s.set_params(p);
    s.set_material(
        0,
        Material {
            friction: 0.5,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    s
}

fn config() -> TerrainConfig {
    TerrainConfig {
        max_particles: 1 << 18,
        max_blocks: 1 << 13,
        ..TerrainConfig::default()
    }
}

/// A dynamic box of half extents `h` and `mass` at `pos`, moving at `v`.
fn crate_box(s: &mut WgpuSim, pos: Vec3, h: Vec3, mass: f32, v: Vec3) -> u32 {
    let inertia = Vec3::new(
        h.y * h.y + h.z * h.z,
        h.x * h.x + h.z * h.z,
        h.x * h.x + h.y * h.y,
    ) * (mass / 3.0);
    let mut state = BodyState::dynamic(pos, Quat::IDENTITY, mass, inertia);
    state.linear_vel = v.to_array();
    let body = s.add_body(state);
    s.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 1.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [h.x, h.y, h.z, h.length()],
    });
    body
}

fn sand_momentum(ps: &[MpmParticle]) -> DVec3 {
    ps.iter()
        .map(|p| Vec3::from_slice(&p.vj[..3]).as_dvec3() * f64::from(p.xm[3]))
        .sum()
}

#[test]
fn a_body_and_the_sand_it_strikes_conserve_momentum() {
    let mut s = sim(0.0);
    let cfg = config();
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    // A 0.4 m sand cube at rest, and a 0.1 m-half box closing on it at 2 m/s.
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-1.0, -0.6, -0.6),
        Vec3::new(0.8, 0.6, 0.6),
        0.05,
        0,
        0,
    )
    .unwrap();
    let h = plan.dx / 2.0;
    let mut ps = Vec::new();
    let n = (0.4 / h) as i32;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let x = Vec3::new(0.0, -0.2, -0.2)
                    + (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * h;
                ps.push(MpmParticle::at_rest(x.to_array(), h * h * h, sand.density));
            }
        }
    }
    s.terrain_load(&plan, &ps, None, sand.friction_angle.tan())
        .unwrap();
    let mass = 5.0;
    let b = crate_box(
        &mut s,
        Vec3::new(-0.3, 0.0, 0.0),
        Vec3::splat(0.1),
        mass,
        Vec3::new(2.0, 0.0, 0.0),
    );
    let body_p = |s: &WgpuSim| Vec3::from(s.read_body(b).linear_vel).as_dvec3() * f64::from(mass);
    let before = sand_momentum(&ps) + body_p(&s);
    let (n, dt) = substeps_for(&cfg, FRAME, plan.dx, sand.p_wave_speed(), 5.0).unwrap();
    for _ in 0..12 {
        s.step_with_terrain(n, dt).unwrap();
    }
    let sand_after = sand_momentum(&s.terrain_particles());
    let after = sand_after + body_p(&s);
    eprintln!(
        "momentum {before:?} -> {after:?} (sand {sand_after:?}, body {:?})",
        body_p(&s)
    );
    assert!(
        sand_after.x > 0.5,
        "the sand took no momentum from the box ({sand_after:?})"
    );
    // The reaction of a frame reaches the body in the next rigid step: the sum
    // is exact once the last frame's impulse is in (one frame's worth of lag
    // bounds the difference).
    assert!(
        (after - before).length() < 0.02 * before.length(),
        "momentum {before:?} before, {after:?} after"
    );
}

#[test]
fn a_heavy_box_rests_on_sand_not_on_the_bed() {
    let mut s = sim(-9.81);
    let cfg = config();
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    // A flat field at y = 0; the region's bed is 0.275 m below it, so the
    // activated layer is that thick.
    let field = HeightField::new(33, 33, 0.0625, 0.0625, vec![0.0; 33 * 33]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-0.9, -0.35, -0.9),
        Vec3::new(0.9, 0.6, 0.9),
        0.05,
        0,
        0,
    )
    .unwrap();
    s.terrain_activate(
        &plan,
        bed,
        &field,
        Vec3::ZERO,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    // 0.3 m cube of 54 kg (2000 kg/m³) set on the surface.
    let half = 0.15;
    let b = crate_box(
        &mut s,
        Vec3::new(0.0, half + 0.005, 0.0),
        Vec3::splat(half),
        54.0,
        Vec3::ZERO,
    );
    let (n, dt) = substeps_for(&cfg, FRAME, plan.dx, sand.p_wave_speed(), 5.0).unwrap();
    let (mut lowest, mut highest) = (f32::MAX, f32::MIN);
    for f in 0..120 {
        s.step_with_terrain(n, dt).unwrap();
        let y = s.read_body(b).position[1];
        lowest = lowest.min(y);
        highest = highest.max(y);
        if f % 20 == 19 {
            eprintln!(
                "t {:.2} box y {y:.4} vy {:.4}",
                (f + 1) as f32 * FRAME,
                s.read_body(b).linear_vel[1]
            );
        }
    }
    let state = s.read_body(b);
    let sink = half - state.position[1];
    eprintln!(
        "box settled {sink:.4} m into the sand (lowest {:.4} m), vy {:.4}",
        half - lowest,
        state.linear_vel[1]
    );
    // Set down, it must not be thrown up (an explicit coupling holding the box
    // at its frame-start velocity once launched it 0.45 m).
    assert!(
        highest < half + 0.005 + 0.01,
        "the box was thrown up to {highest} m"
    );
    // The bed is 0.275 m down: resting on it would sink the box that far.
    assert!(
        sink < 0.1,
        "the box sank {sink} m (the bed is 0.275 m down)"
    );
    assert!(
        state.linear_vel[1].abs() < 0.05,
        "the box is still moving at {} m/s",
        state.linear_vel[1]
    );
}

/// A light box on sand far heavier than it: the contact exchange is the
/// inelastic one (nodes and body share an impulse in proportion to their
/// masses), so it stays at rest whatever the mass ratio. Holding the nodes to
/// the body and then pushing the body with their whole reaction once threw such
/// a box up at 136 m/s.
#[test]
fn a_light_box_rests_on_sand_whatever_the_mass_ratio() {
    let mut s = sim(-9.81);
    let cfg = config();
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    let field = HeightField::new(33, 33, 0.0625, 0.0625, vec![0.0; 33 * 33]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-0.5, -0.3, -0.5),
        Vec3::new(0.5, 0.5, 0.5),
        0.025,
        0,
        0,
    )
    .unwrap();
    s.terrain_activate(
        &plan,
        bed,
        &field,
        Vec3::ZERO,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    // 0.16 m cube of 1 kg (244 kg/m³): the sand nodes it touches outweigh it.
    let half = 0.08;
    let b = crate_box(
        &mut s,
        Vec3::new(0.0, half + 0.05, 0.0),
        Vec3::splat(half),
        1.0,
        Vec3::ZERO,
    );
    let (n, dt) = substeps_for(&cfg, FRAME, plan.dx, sand.p_wave_speed(), 5.0).unwrap();
    let mut fastest = 0.0f32;
    for _ in 0..60 {
        s.step_with_terrain(n, dt).unwrap();
        fastest = fastest.max(Vec3::from(s.read_body(b).linear_vel).length());
    }
    let state = s.read_body(b);
    let sink = half - state.position[1];
    // Dropped 5 cm: never faster than free fall from there (1 m/s).
    assert!(fastest < 1.1, "the box reached {fastest} m/s");
    assert!(
        (-0.005..0.03).contains(&sink),
        "box bottom {sink} m into the sand"
    );
    assert!(
        Vec3::from(state.linear_vel).length() < 0.02,
        "still moving: {:?}",
        state.linear_vel
    );
}
