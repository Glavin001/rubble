//! Heightfield terrain collider (deformable terrain T1). Checks come from
//! geometry and mechanics, with the existing plane as the reference where one
//! applies: a flat heightfield must hold bodies exactly as the plane does, a
//! sphere rolling down a tilted heightfield must match the analytic
//! acceleration (5/7)·g·sin θ, and bodies on uneven terrain must rest on it.

use engine_schema::convex::{ConvexHull, HullArena};
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 1024,
        max_colliders: 1024,
        max_contacts: 32_768,
        max_joints: 16,
        max_candidates: 65_536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn world(friction: f32) -> WgpuSim {
    world_flags(friction, 0)
}

fn world_flags(friction: f32, flags: u32) -> WgpuSim {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu adapter");
    let mut p = AvbdParams {
        dt: 1.0 / 120.0,
        iterations: 10,
        flags,
        ..AvbdParams::default()
    };
    p.gravity_y = -9.81;
    // Bodies here must keep responding (the rolling and edit checks).
    p.sleep_time = 1.0e9;
    sim.set_params(p);
    sim.set_material(
        0,
        Material {
            friction,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    sim
}

fn plane(sim: &mut WgpuSim, normal: Vec3) {
    let ground = sim.add_body(BodyState::static_plane_support());
    let n = normal.normalize();
    sim.add_collider(Collider {
        body: ground,
        shape: ShapeType::Plane as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [n.x, n.y, n.z, 0.0],
    });
}

fn dynamic(
    sim: &mut WgpuSim,
    pos: Vec3,
    rot: Quat,
    shape: ShapeType,
    params: [f32; 4],
    mass: f32,
    inertia: Vec3,
) -> u32 {
    launched(sim, pos, rot, shape, params, mass, inertia, Vec3::ZERO, 0)
}

#[allow(clippy::too_many_arguments)]
fn launched(
    sim: &mut WgpuSim,
    pos: Vec3,
    rot: Quat,
    shape: ShapeType,
    params: [f32; 4],
    mass: f32,
    inertia: Vec3,
    v: Vec3,
    flags: u32,
) -> u32 {
    let mut state = BodyState::dynamic(pos, rot, mass, inertia);
    state.linear_vel = v.to_array();
    state.flags |= flags;
    let body = sim.add_body(state);
    sim.add_collider(Collider {
        body,
        shape: shape as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 1.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params,
    });
    body
}

fn sphere(sim: &mut WgpuSim, pos: Vec3, r: f32) -> u32 {
    let m = 1.0;
    let i = 0.4 * m * r * r;
    dynamic(
        sim,
        pos,
        Quat::IDENTITY,
        ShapeType::Sphere,
        [r, 0.0, 0.0, r],
        m,
        Vec3::splat(i),
    )
}

fn cuboid(sim: &mut WgpuSim, pos: Vec3, rot: Quat, h: Vec3) -> u32 {
    let m = 1.0;
    let i = Vec3::new(
        h.y * h.y + h.z * h.z,
        h.x * h.x + h.z * h.z,
        h.x * h.x + h.y * h.y,
    ) * (m / 3.0);
    dynamic(
        sim,
        pos,
        rot,
        ShapeType::Box,
        [h.x, h.y, h.z, h.length()],
        m,
        i,
    )
}

fn capsule(sim: &mut WgpuSim, pos: Vec3, rot: Quat, r: f32, half: f32) -> u32 {
    // Solid cylinder of the capsule's length about its axis (local y).
    let m = 1.0;
    let len = 2.0 * (half + r);
    let i = Vec3::new(
        m * (3.0 * r * r + len * len) / 12.0,
        0.5 * m * r * r,
        m * (3.0 * r * r + len * len) / 12.0,
    );
    dynamic(
        sim,
        pos,
        rot,
        ShapeType::Capsule,
        [r, half, 0.0, r + half],
        m,
        i,
    )
}

/// Lowest point of each shape the tests use, at its pose.
fn lowest_point(shape: &Shape, b: &BodyState) -> Vec3 {
    let p = Vec3::from_array(b.position);
    let q = Quat::from_array(b.orientation);
    match *shape {
        Shape::Sphere(r) => p - Vec3::Y * r,
        Shape::Capsule(r, half) => {
            let a = q * Vec3::Y * half;
            let e = if a.y < 0.0 { p + a } else { p - a };
            e - Vec3::Y * r
        }
        Shape::Box(h) => {
            let mut low = Vec3::splat(f32::INFINITY);
            for k in 0..8 {
                let l = Vec3::new(
                    if k & 1 != 0 { h.x } else { -h.x },
                    if k & 2 != 0 { h.y } else { -h.y },
                    if k & 4 != 0 { h.z } else { -h.z },
                );
                let w = p + q * l;
                if w.y < low.y {
                    low = w;
                }
            }
            low
        }
    }
}

/// Closest point on triangle abc to p (Ericson 5.1.5).
fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// Signed distance from p to the terrain surface (negative below it), over
/// the triangles within `reach` of p.
fn terrain_distance(field: &HeightField, p: Vec3, reach: f32) -> f32 {
    let (nx, nz) = field.samples();
    let (dx, dz) = field.cell_size();
    let x0 = field.vertex(0, 0);
    let i0 = (((p.x - reach - x0.x) / dx).floor().max(0.0) as u32).min(nx - 2);
    let i1 = (((p.x + reach - x0.x) / dx).ceil().max(0.0) as u32).min(nx - 2);
    let j0 = (((p.z - reach - x0.z) / dz).floor().max(0.0) as u32).min(nz - 2);
    let j1 = (((p.z + reach - x0.z) / dz).ceil().max(0.0) as u32).min(nz - 2);
    let mut best = f32::INFINITY;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let (v00, v10, v01, v11) = (
                field.vertex(i, j),
                field.vertex(i + 1, j),
                field.vertex(i, j + 1),
                field.vertex(i + 1, j + 1),
            );
            for q in [
                closest_on_triangle(p, v00, v10, v11),
                closest_on_triangle(p, v00, v11, v01),
            ] {
                best = best.min((p - q).length());
            }
        }
    }
    match field.surface(p.x, p.z) {
        Some((h, _)) if p.y < h => -best,
        _ => best,
    }
}

/// Deepest penetration of a shape into the terrain (positive = inside).
fn penetration(field: &HeightField, shape: &Shape, b: &BodyState) -> f32 {
    let p = Vec3::from_array(b.position);
    let q = Quat::from_array(b.orientation);
    match *shape {
        Shape::Sphere(r) => r - terrain_distance(field, p, r + 0.5),
        Shape::Capsule(r, half) => {
            let a = q * Vec3::Y * half;
            (0..=64)
                .map(|k| r - terrain_distance(field, p - a + a * (2.0 * k as f32 / 64.0), r + 0.5))
                .fold(f32::NEG_INFINITY, f32::max)
        }
        Shape::Box(_) => {
            let low = lowest_point(shape, b);
            field
                .surface(low.x, low.z)
                .map_or(f32::NEG_INFINITY, |(h, _)| h - low.y)
        }
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Sphere(f32),
    Capsule(f32, f32),
    Box(Vec3),
}

fn spawn(
    sim: &mut WgpuSim,
    arena: &mut HullArena,
    shape: Shape,
    hull: bool,
    pos: Vec3,
    rot: Quat,
) -> u32 {
    match shape {
        Shape::Sphere(r) => sphere(sim, pos, r),
        Shape::Capsule(r, h) => capsule(sim, pos, rot, r, h),
        Shape::Box(h) if hull => {
            sim.add_convex_body(arena, &[ConvexHull::cuboid(h)], pos, rot, 1.0)
        }
        Shape::Box(h) => cuboid(sim, pos, rot, h),
    }
}

fn run(sim: &mut WgpuSim, seconds: f32) {
    for step in 0..(seconds * 120.0).round() as u32 {
        sim.step().unwrap_or_else(|e| panic!("step {step}: {e:?}"));
    }
}

const SHAPES: [(Shape, bool); 5] = [
    (Shape::Sphere(0.4), false),
    (Shape::Box(Vec3::new(0.5, 0.3, 0.4)), false),
    (Shape::Box(Vec3::new(0.35, 0.35, 0.35)), true),
    (Shape::Capsule(0.25, 0.5), false),
    (Shape::Capsule(0.2, 0.6), false),
];

fn drop_on(ground: impl Fn(&mut WgpuSim, &mut HullArena)) -> Vec<BodyState> {
    let mut sim = world(0.8);
    let mut arena = HullArena::new();
    ground(&mut sim, &mut arena);
    let mut ids = Vec::new();
    for (k, &(shape, hull)) in SHAPES.iter().enumerate() {
        // Capsules lie down (axis along x) so both caps and the side touch.
        let rot = match shape {
            Shape::Capsule(..) => Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            _ => Quat::IDENTITY,
        };
        let pos = Vec3::new(-4.0 + 2.0 * k as f32, 1.5, 0.3 * k as f32 - 0.6);
        ids.push(spawn(&mut sim, &mut arena, shape, hull, pos, rot));
    }
    run(&mut sim, 2.0);
    let all = sim.download_bodies();
    ids.iter().map(|&b| all[b as usize]).collect()
}

#[test]
fn flat_heightfield_rests_bodies_as_the_plane_does() {
    let on_plane = drop_on(|sim, _| plane(sim, Vec3::Y));
    let field = HeightField::new(41, 41, 0.5, 0.5, vec![0.0; 41 * 41]).unwrap();
    let on_field = drop_on(|sim, arena| {
        sim.add_height_field(arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
            .unwrap();
    });
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

/// Distance travelled down a 15° slope from rest in `t` seconds by a rolling
/// sphere on `ground` (a slope through the origin whose downhill is +x).
fn rolled(ground: impl Fn(&mut WgpuSim, &mut HullArena), t: f32) -> f32 {
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
    let downhill = Vec3::new(theta.cos(), -theta.sin(), 0.0);
    (end - start).dot(downhill)
}

#[test]
fn sphere_rolls_down_a_tilted_heightfield_at_five_sevenths_g_sin_theta() {
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
    assert!(
        (on_field - on_plane).abs() < 0.02 * on_plane,
        "heightfield {on_field} m vs plane {on_plane} m"
    );
    assert!(
        (on_field - analytic).abs() < 0.06 * analytic,
        "heightfield {on_field} m vs analytic {analytic} m"
    );
}

#[test]
fn bodies_rest_on_uneven_terrain_without_sinking_in() {
    // Slopes stay below 25°, under the friction angle (atan 0.8 ≈ 39°): every
    // body must come to rest on the field.
    let field = HeightField::from_fn(97, 97, 0.25, 0.25, |x, z| {
        0.4 * (0.5 * x).sin() * (0.4 * z).cos()
            + 0.06 * (1.9 * x + 0.4).sin()
            + 0.05 * (2.3 * z).cos()
    })
    .unwrap();
    let steepest = (0..97u32)
        .flat_map(|j| (0..97u32).map(move |i| (i, j)))
        .filter_map(|(i, j)| {
            field.surface(field.vertex(i, j).x + 0.05, field.vertex(i, j).z + 0.02)
        })
        .map(|(_, n)| n.y.acos().to_degrees())
        .fold(0.0f32, f32::max);
    assert!(steepest < 25.0, "test terrain too steep: {steepest}°");
    let mut sim = world(0.8);
    let mut arena = HullArena::new();
    sim.add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let mut placed = Vec::new();
    for k in 0..60u32 {
        let (shape, hull) = SHAPES[k as usize % SHAPES.len()];
        let x = -9.0 + 2.0 * (k % 10) as f32;
        let z = -6.0 + 2.0 * (k / 10) as f32;
        let rot = Quat::from_rotation_y(0.7 * k as f32) * Quat::from_rotation_x(0.4 * k as f32);
        let pos = Vec3::new(x, 2.5, z);
        placed.push((spawn(&mut sim, &mut arena, shape, hull, pos, rot), shape));
    }
    // Penetration is checked throughout, not just at the end: rolling shapes
    // (spheres, capsules) keep moving downhill. Landing from 2.5 m sinks a body
    // up to 5.3 cm into the plane as well (the solver's impact transient, same
    // shapes and drop): the heightfield must stay within that during landing,
    // and close to resting depth afterwards.
    let mut landing = vec![0.0f32; placed.len()];
    let mut after = vec![0.0f32; placed.len()];
    for tick in 0..16 {
        run(&mut sim, 0.25);
        let all = sim.download_bodies();
        for (k, &(b, shape)) in placed.iter().enumerate() {
            let s = all[b as usize];
            assert!(
                Vec3::from_array(s.position).is_finite(),
                "body {b} not finite"
            );
            if field.surface(s.position[0], s.position[2]).is_none() {
                continue;
            }
            let d = penetration(&field, &shape, &s);
            if tick < 6 {
                landing[k] = landing[k].max(d)
            } else {
                after[k] = after[k].max(d)
            }
        }
    }
    let all = sim.download_bodies();
    for (k, &(b, shape)) in placed.iter().enumerate() {
        let s = all[b as usize];
        assert!(
            landing[k] < 0.063,
            "body {b} sank {} m into the terrain on landing",
            landing[k]
        );
        assert!(
            after[k] < 0.035,
            "body {b} sank {} m into the terrain after landing",
            after[k]
        );
        // Boxes and hulls do not roll: on slopes under the friction angle they rest.
        if let Shape::Box(_) = shape {
            let v = Vec3::from_array(s.linear_vel).length();
            assert!(v < 0.05, "body {b} still sliding at {v} m/s");
            assert!(after[k] < 0.02, "body {b} rests {} m deep", after[k]);
            assert!(
                penetration(&field, &shape, &s) > -0.05,
                "body {b} hovers above the terrain"
            );
        }
    }
}

/// A capsule rolling downhill keeps its contacts at the true contact point
/// (contact persistence does not pin its anchors, as for spheres): it rolls
/// down a tilted heightfield exactly as down the tilted plane, and sinks into
/// neither. (With pinned anchors it fell through the plane within 1.5 s.)
#[test]
fn rolling_capsule_stays_on_the_slope() {
    let theta = 15f32.to_radians();
    let field = HeightField::from_fn(81, 41, 0.25, 0.25, |x, _| -x * theta.tan()).unwrap();
    let mut ends = Vec::new();
    for use_plane in [false, true] {
        let mut sim = world(0.8);
        let mut arena = HullArena::new();
        if use_plane {
            plane(&mut sim, Vec3::new(theta.sin(), theta.cos(), 0.0));
        } else {
            sim.add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
                .unwrap();
        }
        let shape = Shape::Capsule(0.25, 0.5);
        // Axis across the slope (z): it rolls down along +x.
        let start = Vec3::new(-3.0, 1.6, 0.4);
        let b = capsule(
            &mut sim,
            start,
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            0.25,
            0.5,
        );
        let mut deepest = 0.0f32;
        for _ in 0..40 {
            run(&mut sim, 0.05);
            deepest = deepest.max(penetration(
                &field,
                &shape,
                &sim.download_bodies()[b as usize],
            ));
        }
        assert!(
            deepest < 0.04,
            "plane {use_plane}: the rolling capsule sank {deepest} m"
        );
        ends.push(sim.download_bodies()[b as usize].position[0] - start.x);
    }
    let (on_field, on_plane) = (ends[0], ends[1]);
    assert!(
        on_plane > 1.0,
        "the capsule did not roll down the plane ({on_plane} m)"
    );
    assert!(
        (on_field - on_plane).abs() < 0.02 * on_plane,
        "heightfield {on_field} m vs plane {on_plane} m"
    );
}

#[test]
fn raising_the_terrain_lifts_a_resting_box() {
    let mut field = HeightField::new(41, 41, 0.5, 0.5, vec![0.0; 41 * 41]).unwrap();
    let mut sim = world(0.8);
    let mut arena = HullArena::new();
    let (_, collider) = sim
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let half = Vec3::new(0.4, 0.25, 0.4);
    let b = cuboid(&mut sim, Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, half);
    run(&mut sim, 1.5);
    let rest = sim.download_bodies()[b as usize].position[1];
    assert!((rest - half.y).abs() < 0.02, "rests at {rest}");
    // A 0.6 m plateau under the box (samples 16..25 in both directions).
    field.set_region(16, 16, 9, 9, &[0.6; 81]).unwrap();
    sim.update_height_field(&mut arena, collider, &field, 16..25)
        .unwrap();
    // The box starts 0.6 m inside the plateau; the solver pushes it out at its
    // usual depenetration rate, then it settles.
    run(&mut sim, 4.0);
    let lifted = sim.download_bodies()[b as usize].position[1];
    assert!(
        (lifted - (0.6 + half.y)).abs() < 0.03,
        "after the edit it rests at {lifted}"
    );
}

/// Fast bodies (150 m/s, 1.25 m per step at 120 Hz) shot into the terrain at
/// 45°: with continuous collision they stop on the surface; without it (the
/// control) the same shots go through, so the check is sensitive.
#[test]
fn fast_bodies_do_not_tunnel_through_the_terrain_with_ccd() {
    use engine_schema::{BODY_CCD, FLAG_SPEC_VELOCITY};
    let field = HeightField::from_fn(129, 129, 0.25, 0.25, |x, z| {
        0.3 * (0.4 * x).sin() + 0.2 * (0.3 * z).cos()
    })
    .unwrap();
    let mut through = [0u32; 2];
    for (k, ccd) in [true, false].into_iter().enumerate() {
        let mut sim = world_flags(0.8, FLAG_SPEC_VELOCITY);
        let mut arena = HullArena::new();
        sim.add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
            .unwrap();
        let flags = if ccd { BODY_CCD } else { 0 };
        let v = Vec3::new(106.0, -106.0, 0.0);
        let mut placed = Vec::new();
        for lane in 0..4 {
            let at = |s: f32| Vec3::new(-6.0 + 3.0 * lane as f32, 6.0, -6.0 + 4.0 * s);
            let i = 0.4 * 0.1 * 0.1;
            placed.push((
                launched(
                    &mut sim,
                    at(0.0),
                    Quat::IDENTITY,
                    ShapeType::Sphere,
                    [0.1, 0.0, 0.0, 0.1],
                    1.0,
                    Vec3::splat(i),
                    v,
                    flags,
                ),
                Shape::Sphere(0.1),
            ));
            let h = Vec3::splat(0.08);
            placed.push((
                launched(
                    &mut sim,
                    at(1.0),
                    Quat::IDENTITY,
                    ShapeType::Box,
                    [h.x, h.y, h.z, h.length()],
                    1.0,
                    Vec3::splat(2.0 * 0.08 * 0.08 / 3.0),
                    v,
                    flags,
                ),
                Shape::Box(h),
            ));
            placed.push((
                launched(
                    &mut sim,
                    at(2.0),
                    Quat::IDENTITY,
                    ShapeType::Capsule,
                    [0.06, 0.1, 0.0, 0.16],
                    1.0,
                    Vec3::new(0.005, 0.002, 0.005),
                    v,
                    flags,
                ),
                Shape::Capsule(0.06, 0.1),
            ));
        }
        run(&mut sim, 0.5);
        let all = sim.download_bodies();
        for &(b, shape) in &placed {
            let s = all[b as usize];
            let p = Vec3::from_array(s.position);
            let below = match field.surface(p.x, p.z) {
                Some((h, _)) => p.y < h,
                None => p.y < -1.0, // off the field: it went through on the way
            };
            if below {
                through[k] += 1;
            } else if ccd {
                assert!(
                    penetration(&field, &shape, &s) < 0.05,
                    "body {b} sank into the terrain: {:?}",
                    s.position
                );
            }
        }
    }
    assert_eq!(
        through[0], 0,
        "{} CCD bodies went through the terrain",
        through[0]
    );
    assert!(
        through[1] > 0,
        "the control (no CCD) never tunnelled: the shots are too slow to test anything"
    );
}

/// A pile of small capsules (r = 0.12 m, half the cell size) dropped on hills at
/// 60 Hz rolls and tumbles without any sinking through the surface. Before the
/// solver took capsules' lever arms at the step-start rotation (as for spheres),
/// more than half of them rolled into the ground (0.75 m), on a plane as well.
///
/// Bounds come from the capsule, not from a tuned run: never as deep as its
/// radius (that is falling through), and settled within half of it. For scale,
/// the same drop on a `Plane` without look-ahead lands 0.127 m deep and rests at
/// 0.10 m (AVBD contact sag at 60 Hz); this pile lands 0.08 m deep and rests
/// within 0.035 m.
#[test]
fn small_capsules_roll_over_hills_without_sinking() {
    use engine_schema::FLAG_SPEC_VELOCITY;
    let hills = |x: f32, z: f32| {
        1.4 * (0.22 * x).sin() * (0.18 * z).cos()
            + 0.5 * (0.55 * x + 1.0).sin() * (0.43 * z).sin()
            + 0.9 * (-((x - 6.0) * (x - 6.0)) / 8.0).exp() * (0.15 * z).cos()
    };
    let field = HeightField::from_fn(161, 161, 0.25, 0.25, hills).unwrap();
    let mut sim = world_flags(0.8, FLAG_SPEC_VELOCITY);
    let mut p = AvbdParams {
        dt: 1.0 / 60.0,
        iterations: 10,
        flags: FLAG_SPEC_VELOCITY,
        ..AvbdParams::default()
    };
    p.gravity_y = -9.81;
    sim.set_params(p);
    let mut arena = HullArena::new();
    sim.add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let mut caps = Vec::new();
    for k in 0..200u32 {
        let x = -7.0 + 1.1 * (k % 14) as f32;
        let z = -7.0 + 1.1 * ((k / 14) % 14) as f32;
        let y = 4.0 + (k / 196) as f32 + hills(x, z);
        let rot = Quat::from_rotation_y(0.7 * k as f32) * Quat::from_rotation_x(0.9 * k as f32);
        let hh = 0.18 + 0.1 * ((k * 37) % 10) as f32 / 10.0;
        caps.push((
            capsule(&mut sim, Vec3::new(x, y, z), rot, 0.12, hh),
            Shape::Capsule(0.12, hh),
        ));
    }
    let r = 0.12;
    let (mut deepest, mut settled) = (0.0f32, 0.0f32);
    for check in 0..30 {
        for _ in 0..10 {
            sim.step().unwrap();
        }
        let all = sim.download_bodies();
        for &(b, shape) in &caps {
            let s = all[b as usize];
            if field.surface(s.position[0], s.position[2]).is_some() {
                let d = penetration(&field, &shape, &s);
                deepest = deepest.max(d);
                // The last second (the pile has landed by check 5, 0.8 s).
                if check >= 24 {
                    settled = settled.max(d);
                }
            }
        }
    }
    assert!(deepest < r, "a capsule sank {deepest} m into the terrain");
    assert!(
        settled < 0.5 * r,
        "a settled capsule rests {settled} m deep in the terrain"
    );
}

/// Where a sphere (or upright capsule) of radius `r` swept from `o` along unit
/// `d` first touches the terrain, by conservative advancement against the
/// exact triangle distance (steps never longer than the distance's reach).
fn swept_contact(
    field: &HeightField,
    o: Vec3,
    d: Vec3,
    r: f32,
    half: f32,
    max: f32,
) -> Option<f32> {
    let reach = r + 1.0;
    let mut t = 0.0f32;
    while t <= max {
        let c = o + d * t;
        let gap = (0..=16)
            .map(|k| {
                terrain_distance(
                    field,
                    c + Vec3::Y * half * (2.0 * k as f32 / 16.0 - 1.0),
                    reach,
                )
            })
            .fold(f32::INFINITY, f32::min)
            - r;
        if gap <= 1.0e-5 {
            return Some(t);
        }
        t += gap.min(reach - r).max(1.0e-5);
    }
    None
}

/// Sphere and capsule casts against the terrain meet it where the swept shape
/// first touches it: close to the ground (the tolerance once grew with the
/// query length, so anything within 1 m reported a hit at its start) and from
/// far off the field's footprint (the march once gave up there).
#[test]
fn round_casts_meet_the_terrain_where_the_swept_shape_touches_it() {
    use engine_schema::character::SceneQuery;
    let field = HeightField::from_fn(41, 41, 0.25, 0.25, |x, z| {
        2.0 * (-(x * x + z * z) / 4.0).exp() + 0.1 * (1.3 * x).sin()
    })
    .unwrap();
    let mut sim = world(0.8);
    let mut arena = HullArena::new();
    sim.add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let ground = |x: f32, z: f32| field.surface(x, z).unwrap().0;
    // (origin, direction, radius, capsule half height)
    let casts = [
        (
            Vec3::new(0.3, ground(0.3, -0.2) + 0.5, -0.2),
            Vec3::NEG_Y,
            0.3,
            0.0,
        ),
        (
            Vec3::new(-3.1, ground(-3.1, 2.2) + 0.9, 2.2),
            Vec3::NEG_Y,
            0.25,
            0.0,
        ),
        (Vec3::new(-50.0, 1.0, 0.4), Vec3::X, 0.25, 0.0),
        (Vec3::new(1.2, 40.0, -0.7), Vec3::NEG_Y, 0.2, 0.0),
        (
            Vec3::new(0.8, ground(0.8, 0.5) + 0.2 + 0.3 + 0.6, 0.5),
            Vec3::NEG_Y,
            0.2,
            0.3,
        ),
    ];
    let queries: Vec<SceneQuery> = casts
        .iter()
        .map(|&(o, d, r, half)| SceneQuery {
            radius: r,
            half_height: half,
            ..SceneQuery::ray(o.to_array(), d.to_array(), 100.0)
        })
        .collect();
    sim.set_queries(&queries);
    sim.step().unwrap();
    for (k, (hit, &(o, d, r, half))) in sim.query_results().iter().zip(&casts).enumerate() {
        let want = swept_contact(&field, o, d, r, half, 100.0).expect("the reference hits");
        let got = hit
            .unwrap_or_else(|| panic!("cast {k} missed (reference {want} m)"))
            .distance;
        assert!(
            (got - want).abs() < 2.0e-3,
            "cast {k}: hit at {got} m, the swept shape touches at {want} m"
        );
    }
}

/// A shot through two ridges loses the energy for the ground it crosses in
/// each (cost per metre × the path under the surface), and a shot into the
/// ground that never comes back out over the field stays in it. The
/// thickness once came from a cast back from beyond the whole collider, which
/// on terrain found the far side of the last ridge: the first one charged for
/// both and the gap between them, and stopped the shot.
#[test]
fn shots_through_ridges_pay_for_the_ground_they_cross() {
    use engine_schema::character::{ProjectileDesc, NO_INDEX};
    // Ridges h = 2 - |x ∓ 3| on |x ∓ 3| < 2: at y = 1 the path runs 2 m under
    // each. Their kinks lie on sample columns, so the field is exact.
    let field = HeightField::from_fn(81, 21, 0.25, 0.25, |x, _| {
        (2.0 - (x + 3.0).abs()).max(0.0) + (2.0 - (x - 3.0).abs()).max(0.0)
    })
    .unwrap();
    let mut sim = world(0.8);
    let mut arena = HullArena::new();
    sim.add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let (m, v0, cost) = (0.01f32, 900.0f32, 800.0f32);
    let shot = |position: [f32; 3], velocity: [f32; 3]| ProjectileDesc {
        position,
        velocity,
        radius: 0.0,
        mass: m,
        penetration_cost: cost,
        drag: 0.0,
        max_hits_per_step: 8,
        exclude_body: NO_INDEX,
        mask: 0,
        impulses: false,
    };
    sim.spawn_projectile(&shot([-8.0, 1.0, 0.1], [v0, 0.0, 0.0]));
    // Down into the flat ground between the ridges at 20°.
    let dive = Vec3::new(20f32.to_radians().cos(), -20f32.to_radians().sin(), 0.0) * v0;
    sim.spawn_projectile(&shot([-0.9, 0.5, -1.0], dive.to_array()));
    for _ in 0..3 {
        sim.step().unwrap();
    }
    let into = sim.projectile_states()[1];
    let left = Vec3::from_array(into.velocity).length();
    assert!(
        !into.alive || left < 1.0e-3,
        "the shot into the ground is still flying at {left} m/s from {:?}",
        into.position
    );
    let p = sim.projectile_states()[0];
    assert!(
        p.position[0] > 5.0,
        "the shot stopped at x = {}",
        p.position[0]
    );
    let speed = Vec3::from_array(p.velocity).x;
    let want = (v0 * v0 - 2.0 * cost * 4.0 / m).sqrt();
    assert!(
        (speed - want).abs() < 0.01 * want,
        "leaves the ridges at {speed} m/s; crossing 4 m of ground leaves {want} m/s"
    );
}

/// With contact reuse (FLAG_REUSE_CONTACTS) a body that has not moved keeps
/// last step's manifolds. A height edit must still reach it: a resting box
/// once hovered over the pit dug under it.
#[test]
fn digging_under_a_box_drops_it_with_contact_reuse() {
    use engine_schema::FLAG_REUSE_CONTACTS;
    let mut field = HeightField::new(41, 41, 0.5, 0.5, vec![0.0; 41 * 41]).unwrap();
    let mut sim = world_flags(0.8, FLAG_REUSE_CONTACTS);
    let mut arena = HullArena::new();
    let (_, collider) = sim
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let half = Vec3::new(0.4, 0.25, 0.4);
    let b = cuboid(&mut sim, Vec3::new(0.0, 0.3, 0.0), Quat::IDENTITY, half);
    run(&mut sim, 1.5);
    let rest = sim.download_bodies()[b as usize].position[1];
    assert!((rest - half.y).abs() < 0.02, "rests at {rest}");
    // A 0.5 m pit under the box: samples 16..25 (flat floor x, z in [-2, 2]).
    field.set_region(16, 16, 9, 9, &[-0.5; 81]).unwrap();
    sim.update_height_field(&mut arena, collider, &field, 16..25)
        .unwrap();
    run(&mut sim, 1.5);
    let fallen = sim.download_bodies()[b as usize].position[1];
    assert!(
        (fallen - (-0.5 + half.y)).abs() < 0.02,
        "after the dig the box is at {fallen} (rested at {rest})"
    );
}

/// A capsule lying straight down a slope cannot roll; static friction holds
/// it (tan 20° = 0.36 < μ = 0.8), on the plane and on the terrain alike. The
/// persistence pass no longer pins a capsule's anchors while friction holds
/// (a rolling capsule sank through the ground that way), so this checks the
/// unpinned anchors still hold one that is not rolling.
#[test]
fn a_capsule_down_the_slope_holds_by_static_friction() {
    let theta = 20f32.to_radians();
    let field = HeightField::from_fn(81, 41, 0.25, 0.25, |x, _| -x * theta.tan()).unwrap();
    for use_plane in [false, true] {
        let mut sim = world(0.8);
        let mut arena = HullArena::new();
        if use_plane {
            plane(&mut sim, Vec3::new(theta.sin(), theta.cos(), 0.0));
        } else {
            sim.add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
                .unwrap();
        }
        // Axis along the fall line (local y onto the slope's downhill).
        let (r, half) = (0.25, 0.5);
        let down = Vec3::new(theta.cos(), -theta.sin(), 0.0);
        let up = Vec3::new(theta.sin(), theta.cos(), 0.0);
        let start = up * (r + 0.005);
        let b = capsule(
            &mut sim,
            start,
            Quat::from_rotation_arc(Vec3::Y, down),
            r,
            half,
        );
        run(&mut sim, 0.5);
        let settled = Vec3::from_array(sim.download_bodies()[b as usize].position);
        run(&mut sim, 2.0);
        let end = Vec3::from_array(sim.download_bodies()[b as usize].position);
        let creep = (end - settled).dot(down);
        assert!(
            creep.abs() < 2.0e-3,
            "plane {use_plane}: crept {creep} m down the slope in 2 s"
        );
    }
}

/// A capsule dropped beside a spike (one raised sample) rests against its tip
/// without sinking onto it: the tip is the closest terrain point to the axis.
/// (Its vertex contact was once never found, the test for it also asking
/// triangles without that corner; the flank and ridge-edge contacts held the
/// capsule as well, so this guards the physics rather than that predicate.)
#[test]
fn a_capsule_beside_a_spike_rests_on_its_tip() {
    let mut heights = vec![0.0; 41 * 41];
    heights[20 * 41 + 20] = 0.6;
    let field = HeightField::new(41, 41, 0.25, 0.25, heights).unwrap();
    let (r, half) = (0.15, 0.5);
    let shape = Shape::Capsule(r, half);
    let mut sim = world(0.8);
    let mut arena = HullArena::new();
    sim.add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    // Axis along x, 8 cm beside the tip, starting just above it.
    let b = capsule(
        &mut sim,
        Vec3::new(0.05, 0.6 + r, 0.08),
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        r,
        half,
    );
    let mut deepest = 0.0f32;
    for _ in 0..40 {
        run(&mut sim, 0.025);
        deepest = deepest.max(penetration(
            &field,
            &shape,
            &sim.download_bodies()[b as usize],
        ));
    }
    assert!(
        deepest < 0.3 * r,
        "the capsule sank {deepest} m onto the spike"
    );
}
