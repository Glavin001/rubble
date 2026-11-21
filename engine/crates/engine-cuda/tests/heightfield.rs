//! Height fields on CUDA: engine-webgpu `heightfield.rs`'s two reference checks, with its
//! world, shapes and tolerances. A flat field rests bodies as a plane does (within 2 mm,
//! settled); a sphere rolls down a 15° field as down a 15° plane (2 %) and as
//! 5/7·g·sin θ (6 %). `ENGINE_CUDA_DEVICE=1` runs them; Modal sets it.
use engine_cuda::CudaSim;
use engine_schema::convex::{ConvexHull, HullArena};
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use glam::{Quat, Vec3};

fn gpu() -> bool {
    if std::env::var("ENGINE_CUDA_DEVICE").as_deref() != Ok("1") {
        eprintln!("skipped: set ENGINE_CUDA_DEVICE=1 on an NVIDIA host");
        return false;
    }
    assert!(
        CudaSim::available(),
        "ENGINE_CUDA_DEVICE=1 but no device code or no NVIDIA device"
    );
    true
}

fn world(friction: f32) -> CudaSim {
    let mut sim = CudaSim::new(WorldCapacity {
        max_bodies: 1024,
        max_colliders: 1024,
        max_contacts: 32_768,
        max_joints: 16,
        max_candidates: 65_536,
        max_colors: 32,
        max_commands: 64,
        // The arena's whole size on CUDA (no growth): room for the fields below.
        max_hull_vertices: 1 << 14,
    })
    .expect("CudaSim");
    let mut p = AvbdParams {
        dt: 1.0 / 120.0,
        iterations: 10,
        ..AvbdParams::default()
    };
    p.gravity_y = -9.81;
    p.sleep_time = 1.0e9;
    sim.set_params(p);
    sim.set_materials(
        &[Material {
            friction,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        }; 64],
    )
    .expect("materials");
    sim
}

fn plane(sim: &mut CudaSim, normal: Vec3) {
    let ground = sim.add_bodies(&[BodyState::static_plane_support()]).start;
    let n = normal.normalize();
    sim.add_colliders(&[Collider {
        body: ground,
        shape: ShapeType::Plane as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [n.x, n.y, n.z, 0.0],
    }]);
}

fn dynamic(
    sim: &mut CudaSim,
    pos: Vec3,
    rot: Quat,
    shape: ShapeType,
    params: [f32; 4],
    inertia: Vec3,
) -> u32 {
    let body = sim
        .add_bodies(&[BodyState::dynamic(pos, rot, 1.0, inertia)])
        .start;
    sim.add_colliders(&[Collider {
        body,
        shape: shape as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 1.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params,
    }]);
    body
}

fn sphere(sim: &mut CudaSim, pos: Vec3, r: f32) -> u32 {
    dynamic(
        sim,
        pos,
        Quat::IDENTITY,
        ShapeType::Sphere,
        [r, 0.0, 0.0, r],
        Vec3::splat(0.4 * r * r),
    )
}

fn cuboid(sim: &mut CudaSim, pos: Vec3, h: Vec3) -> u32 {
    let i = Vec3::new(
        h.y * h.y + h.z * h.z,
        h.x * h.x + h.z * h.z,
        h.x * h.x + h.y * h.y,
    ) / 3.0;
    dynamic(
        sim,
        pos,
        Quat::IDENTITY,
        ShapeType::Box,
        [h.x, h.y, h.z, h.length()],
        i,
    )
}

fn capsule(sim: &mut CudaSim, pos: Vec3, rot: Quat, r: f32, half: f32) -> u32 {
    let len = 2.0 * (half + r);
    let i = Vec3::new(
        (3.0 * r * r + len * len) / 12.0,
        0.5 * r * r,
        (3.0 * r * r + len * len) / 12.0,
    );
    dynamic(
        sim,
        pos,
        rot,
        ShapeType::Capsule,
        [r, half, 0.0, r + half],
        i,
    )
}

fn run(sim: &mut CudaSim, seconds: f32) {
    for step in 0..(seconds * 120.0).round() as u32 {
        sim.step().unwrap_or_else(|e| panic!("step {step}: {e:?}"));
    }
}

/// engine-webgpu's five shapes (sphere, box, a convex-hull box, two capsules lying
/// down), dropped for 2 s.
fn drop_on(ground: impl Fn(&mut CudaSim, &mut HullArena)) -> Vec<BodyState> {
    let mut sim = world(0.8);
    let mut arena = HullArena::new();
    ground(&mut sim, &mut arena);
    let lie = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let at = |k: usize| Vec3::new(-4.0 + 2.0 * k as f32, 1.5, 0.3 * k as f32 - 0.6);
    let ids = [
        sphere(&mut sim, at(0), 0.4),
        cuboid(&mut sim, at(1), Vec3::new(0.5, 0.3, 0.4)),
        sim.add_convex_body(
            &mut arena,
            &[ConvexHull::cuboid(Vec3::splat(0.35))],
            at(2),
            Quat::IDENTITY,
            1.0,
        )
        .expect("hull"),
        capsule(&mut sim, at(3), lie, 0.25, 0.5),
        capsule(&mut sim, at(4), lie, 0.2, 0.6),
    ];
    run(&mut sim, 2.0);
    let all = sim.download_bodies();
    ids.iter().map(|&b| all[b as usize]).collect()
}

#[test]
fn cuda_flat_heightfield_rests_bodies_as_the_plane_does() {
    if !gpu() {
        return;
    }
    let on_plane = drop_on(|sim, _| plane(sim, Vec3::Y));
    let field = HeightField::new(41, 41, 0.5, 0.5, vec![0.0; 41 * 41]).unwrap();
    let on_field = drop_on(|sim, arena| {
        sim.add_height_field(arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
            .unwrap();
    });
    for (k, (a, b)) in on_plane.iter().zip(&on_field).enumerate() {
        eprintln!(
            "shape {k}: field y {:.5} q {:?} v {:?}; plane y {:.5} q {:?}",
            b.position[1], b.orientation, b.linear_vel, a.position[1], a.orientation
        );
    }
    for (k, (a, b)) in on_plane.iter().zip(&on_field).enumerate() {
        let (pa, pb) = (Vec3::from_array(a.position), Vec3::from_array(b.position));
        assert!(pb.is_finite(), "shape {k} not finite");
        assert!(
            (pa.y - pb.y).abs() < 2.0e-3,
            "shape {k}: rests at {} on the heightfield, {} on the plane",
            pb.y,
            pa.y
        );
        let v = Vec3::from_array(b.linear_vel).length();
        assert!(
            v < 0.05,
            "shape {k} still moving at {v} m/s on the heightfield"
        );
    }
}

fn rolled(ground: impl Fn(&mut CudaSim, &mut HullArena), t: f32) -> f32 {
    let theta = 15f32.to_radians();
    let mut sim = world(1.0);
    let mut arena = HullArena::new();
    ground(&mut sim, &mut arena);
    let n = Vec3::new(theta.sin(), theta.cos(), 0.0);
    let r = 0.3;
    let start = Vec3::new(-3.0, 3.0 * theta.tan(), 0.0) + n * r;
    let b = sphere(&mut sim, start, r);
    run(&mut sim, t);
    let end = Vec3::from_array(sim.download_bodies()[b as usize].position);
    (end - start).dot(Vec3::new(theta.cos(), -theta.sin(), 0.0))
}

#[test]
fn cuda_sphere_rolls_down_a_tilted_heightfield_at_five_sevenths_g_sin_theta() {
    if !gpu() {
        return;
    }
    let theta = 15f32.to_radians();
    let t = 1.0;
    let analytic = 0.5 * (5.0 / 7.0) * 9.81 * theta.sin() * t * t;
    let field = HeightField::from_fn(81, 21, 0.25, 0.25, |x, _| -x * theta.tan()).unwrap();
    let on_field = rolled(
        |sim, arena| {
            sim.add_height_field(arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
                .unwrap();
        },
        t,
    );
    let on_plane = rolled(
        |sim, _| plane(sim, Vec3::new(theta.sin(), theta.cos(), 0.0)),
        t,
    );
    eprintln!("rolled {on_field:.4} m on the field, {on_plane:.4} m on the plane, analytic {analytic:.4} m");
    assert!(
        (on_field - on_plane).abs() < 0.02 * on_plane,
        "heightfield {on_field} m vs plane {on_plane} m"
    );
    assert!(
        (on_field - analytic).abs() < 0.06 * analytic,
        "heightfield {on_field} m vs analytic {analytic} m"
    );
}
