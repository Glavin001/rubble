//! In-step scene queries and projectiles (character audit Phase 1, CCD audit
//! row 5): the grid walk against its full-scan oracle on a hierarchical
//! (convex) scene and a uniform one, exact hull hits against a CPU reference,
//! and bullet-class projectiles through walls.

use engine_schema::character::*;
use engine_schema::convex::{cook_body, ConvexHull, HullArena};
use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity};
use engine_webgpu::{MotionEvent, WgpuSim};
use glam::{Quat, Vec3};

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 8192,
        max_colliders: 16384,
        max_contacts: 262_144,
        max_joints: 64,
        max_candidates: 524_288,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 65_536,
    }
}

struct Rng(u32);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
}

fn static_box(sim: &mut WgpuSim, pos: Vec3, half: [f32; 3]) -> u32 {
    let b = sim.add_body(BodyState {
        position: pos.to_array(),
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos.to_array(),
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    });
    sim.add_collider(Collider {
        body: b,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half[0], half[1], half[2], Vec3::from_array(half).length()],
    })
}

/// Hull bodies as cooked (body-frame hull, local position), for the CPU reference.
struct HullBody {
    body: u32,
    collider: u32,
    hull: ConvexHull,
    local: Vec3,
}

/// A convex pile (hierarchical grid) with boxes, spheres and capsules mixed in.
fn convex_scene() -> (WgpuSim, Vec<HullBody>) {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    let mut p = AvbdParams::default();
    p.dt = 1.0 / 60.0;
    sim.set_params(p);
    static_box(&mut sim, Vec3::new(0.0, -0.5, 0.0), [40.0, 0.5, 40.0]);
    let mut arena = HullArena::new();
    let mut rng = Rng(7);
    let mut out = Vec::new();
    for i in 0..400u32 {
        let pts: Vec<Vec3> = (0..10).map(|_| Vec3::new(rng.range(-0.3, 0.3), rng.range(-0.25, 0.25), rng.range(-0.3, 0.3))).collect();
        let hull = ConvexHull::from_points(&pts).unwrap();
        let pos = Vec3::new((i % 20) as f32 * 0.9 - 9.0, 0.5 + (i / 200) as f32 * 0.8, ((i / 20) % 10) as f32 * 0.9 - 4.5);
        let rot = Quat::from_rotation_y(i as f32 * 0.37) * Quat::from_rotation_x(i as f32 * 0.21);
        let cooked = cook_body(std::slice::from_ref(&hull), 500.0);
        let body = sim.add_convex_body(&mut arena, std::slice::from_ref(&hull), pos, rot, 500.0);
        let collider = sim.collider_count() - 1;
        out.push(HullBody { body, collider, hull: cooked.children[0].hull.clone(), local: cooked.children[0].local_pos });
    }
    for k in 0..40u32 {
        static_box(&mut sim, Vec3::new(k as f32 * 0.5 - 10.0, 0.3, 6.0), [0.2, 0.3, 0.2]);
    }
    (sim, out)
}

/// CPU ray vs a cooked hull at its body's pose (slab test over face planes).
fn cpu_ray_hull(h: &HullBody, b: &BodyState, o: Vec3, d: Vec3) -> Option<f32> {
    let q = Quat::from_array(b.orientation);
    let lo = q.inverse() * (o - Vec3::from_array(b.position)) - h.local;
    let ld = q.inverse() * d;
    let (mut tn, mut tf) = (f32::NEG_INFINITY, f32::INFINITY);
    for f in &h.hull.faces {
        let denom = f.normal.dot(ld);
        let num = f.d - f.normal.dot(lo);
        if denom.abs() < 1e-12 {
            if num < 0.0 {
                return None;
            }
            continue;
        }
        let t = num / denom;
        if denom < 0.0 {
            tn = tn.max(t);
        } else {
            tf = tf.min(t);
        }
        if tn > tf {
            return None;
        }
    }
    (tf >= 0.0 && tn > f32::NEG_INFINITY).then_some(tn)
}

fn queries(rng: &mut Rng, n: usize) -> Vec<SceneQuery> {
    (0..n)
        .map(|k| {
            let o = Vec3::new(rng.range(-11.0, 11.0), rng.range(0.2, 4.0), rng.range(-7.0, 7.0));
            let d = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 0.3), rng.range(-1.0, 1.0)).normalize();
            let mut q = SceneQuery::ray(o.to_array(), d.to_array(), rng.range(2.0, 12.0));
            match k % 4 {
                1 => q.radius = 0.15,
                2 => {
                    q.radius = 0.2;
                    q.half_height = 0.4;
                }
                _ => {}
            }
            q
        })
        .collect()
}

fn grid_walk_matches_full_scan(mut sim: WgpuSim, label: &str) -> (Vec<SceneQuery>, Vec<Option<QueryHit>>) {
    for _ in 0..30 {
        sim.step().expect("step");
    }
    let mut rng = Rng(99);
    let qs = queries(&mut rng, 120);
    let mut both = qs.clone();
    both.extend(qs.iter().map(|q| SceneQuery { full_scan: true, ..*q }));
    sim.set_queries(&both);
    sim.step().expect("step");
    let res = sim.query_results();
    let (grid, full) = res.split_at(qs.len());
    let mut hits = 0;
    for (k, (g, f)) in grid.iter().zip(full).enumerate() {
        match (g, f) {
            (None, None) => {}
            (Some(g), Some(f)) => {
                hits += 1;
                assert_eq!(g.collider, f.collider, "{label} query {k}: grid hit collider {} vs full scan {}", g.collider, f.collider);
                assert!((g.distance - f.distance).abs() < 1e-5, "{label} query {k}: {} vs {}", g.distance, f.distance);
            }
            _ => panic!("{label} query {k}: grid {g:?} vs full scan {f:?}"),
        }
    }
    println!("{label}: {hits} of {} queries hit; grid walk = full scan", qs.len());
    assert!(hits > 30, "{label}: the queries should mostly hit something ({hits})");
    (qs, grid.to_vec())
}

#[test]
fn hierarchical_grid_walk_matches_full_scan_and_hulls_are_exact() {
    let (sim, hulls) = convex_scene();
    // Capture the poses the queried step starts from: queries run at x⁻.
    let mut sim = sim;
    for _ in 0..30 {
        sim.step().expect("step");
    }
    let bodies = sim.download_bodies();
    let mut rng = Rng(99);
    let qs = queries(&mut rng, 120);
    let mut both = qs.clone();
    both.extend(qs.iter().map(|q| SceneQuery { full_scan: true, ..*q }));
    sim.set_queries(&both);
    sim.step().expect("step");
    let res = sim.query_results();
    let (grid, full) = res.split_at(qs.len());
    let mut hull_hits = 0;
    let mut hits = 0;
    for (k, (g, f)) in grid.iter().zip(full).enumerate() {
        match (g, f) {
            (None, None) => {}
            (Some(g), Some(f)) => {
                hits += 1;
                assert_eq!(g.collider, f.collider, "query {k}: grid {} vs full scan {}", g.collider, f.collider);
                assert!((g.distance - f.distance).abs() < 1e-5, "query {k}");
                // Rays on hulls: the CPU slab test over the cooked faces.
                let q = &qs[k];
                if q.radius == 0.0 {
                    if let Some(h) = hulls.iter().find(|h| h.collider == g.collider) {
                        let t = cpu_ray_hull(h, &bodies[h.body as usize], Vec3::from_array(q.origin), Vec3::from_array(q.dir).normalize())
                            .expect("CPU ray misses a hull the GPU hit");
                        assert!((t - g.distance).abs() < 1e-3, "query {k}: GPU hull hit {} vs CPU {t}", g.distance);
                        hull_hits += 1;
                    }
                }
            }
            _ => panic!("query {k}: grid {g:?} vs full scan {f:?}"),
        }
    }
    println!("hierarchical: {hits} hits, {hull_hits} exact hull ray hits checked on the CPU");
    assert!(hull_hits > 10, "too few hull ray hits to check ({hull_hits})");
}

#[test]
fn uniform_grid_walk_matches_full_scan() {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    sim.set_params(AvbdParams::default());
    static_box(&mut sim, Vec3::new(0.0, -0.5, 0.0), [40.0, 0.5, 40.0]);
    for i in 0..900u32 {
        let p = Vec3::new((i % 30) as f32 * 0.7 - 10.5, 0.3 + (i / 300) as f32 * 0.9, ((i / 30) % 10) as f32 * 1.2 - 6.0);
        static_box(&mut sim, p, [0.2, 0.2, 0.3]);
    }
    grid_walk_matches_full_scan(sim, "uniform");
}

/// Walls of `thickness` at x = 10, 12, 14, 16, 18 (static boxes), and a bullet
/// fired along +x from x = 0.
fn bullet_through_walls(thickness: f32, cost: f32) -> (WgpuSim, u32, Vec<u32>) {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    let mut p = AvbdParams::default();
    p.dt = 1.0 / 60.0;
    sim.set_params(p);
    static_box(&mut sim, Vec3::new(0.0, -0.5, 0.0), [60.0, 0.5, 60.0]);
    let walls: Vec<u32> = (0..5).map(|k| static_box(&mut sim, Vec3::new(10.0 + 2.0 * k as f32 + 0.5 * thickness, 1.5, 0.0), [0.5 * thickness, 1.5, 1.5])).collect();
    let b = sim.spawn_projectile(&ProjectileDesc {
        position: [0.0, 1.5, 0.0],
        velocity: [900.0, 0.0, 0.0],
        radius: 0.0,
        mass: 0.01,
        penetration_cost: cost,
        drag: 0.0,
        max_hits_per_step: 8,
        exclude_body: NO_INDEX,
        mask: 0,
        impulses: true,
    });
    (sim, b, walls)
}

#[test]
fn a_bullet_stops_at_a_thin_wall_or_passes_five_in_order() {
    // 0.01 kg at 900 m/s: 4,050 J. A 0.05 m wall costing 1e6 J/m (50 kJ) stops it.
    let (mut sim, _, walls) = bullet_through_walls(0.05, 1.0e6);
    for _ in 0..4 {
        sim.step().expect("step");
    }
    let s = sim.projectile_states()[0];
    println!("stopped: {s:?}");
    assert!(!s.alive, "the bullet stopped");
    assert!((s.position[0] - 10.0).abs() < 1e-3, "at the wall's near face: {:?}", s.position);
    let events = sim.poll_motion_events();
    let hits: Vec<_> = events.iter().filter_map(|e| match e { MotionEvent::ProjectileHit { collider, .. } => Some(*collider), _ => None }).collect();
    assert_eq!(hits, vec![walls[0]], "one hit, the first wall: {events:?}");

    // 5,000 J/m: each 0.05 m wall costs 250 J, so it passes all five in order,
    // slower after each.
    let (mut sim, _, walls) = bullet_through_walls(0.05, 5000.0);
    for _ in 0..4 {
        sim.step().expect("step");
    }
    let s = sim.projectile_states()[0];
    let events = sim.poll_motion_events();
    let hits: Vec<(u32, f32)> = events
        .iter()
        .filter_map(|e| match e { MotionEvent::ProjectileHit { collider, thickness, .. } => Some((*collider, *thickness)), _ => None })
        .collect();
    println!("through: {s:?} hits {hits:?}");
    assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), walls, "every wall, in order");
    for (_, t) in &hits {
        assert!((t - 0.05).abs() < 1e-3, "thickness along the path {t}");
    }
    assert!(s.alive && s.position[0] > 18.05, "carried on past the last wall: {:?}", s.position);
    let e_left = 0.5 * 0.01 * s.velocity[0] * s.velocity[0];
    assert!((e_left - (4050.0 - 5.0 * 250.0)).abs() < 20.0, "energy left {e_left}");
}

#[test]
fn bullet_hits_push_dynamic_bodies_and_repeat_bit_for_bit() {
    let run = || {
        let mut sim = WgpuSim::new(capacity()).expect("wgpu");
        let mut p = AvbdParams::default();
        p.gravity_y = 0.0;
        sim.set_params(p);
        let mass = 2.0f32;
        let half = 0.25f32;
        let i = 3.0 / (mass * 2.0 * half * half);
        let mut targets = Vec::new();
        for k in 0..3 {
            let pos = Vec3::new(5.0 + 2.0 * k as f32, 1.0, 0.0);
            let b = sim.add_body(BodyState::dynamic(pos, Quat::IDENTITY, mass, Vec3::splat(1.0 / i)));
            sim.add_collider(Collider {
                body: b,
                shape: ShapeType::Box as u32,
                material: 0,
                generation: 0,
                local_pos: [0.0; 3],
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [half, half, half, half * 3f32.sqrt()],
            });
            targets.push(b);
        }
        for k in 0..6 {
            sim.spawn_projectile(&ProjectileDesc {
                position: [0.0, 1.0 + 0.02 * k as f32, 0.01 * k as f32],
                velocity: [600.0, 0.0, 0.0],
                radius: 0.0,
                mass: 0.01,
                penetration_cost: 2000.0,
                drag: 0.0,
                max_hits_per_step: 8,
                exclude_body: NO_INDEX,
                mask: 0,
                impulses: true,
            });
        }
        for _ in 0..3 {
            sim.step().expect("step");
        }
        targets.iter().map(|&b| sim.read_body(b)).collect::<Vec<_>>()
    };
    let a = run();
    let b = run();
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(bytemuck::bytes_of(x), bytemuck::bytes_of(y), "the replay differs");
    }
    // 0.01 kg at 600 m/s (1,800 J); 0.5 m of box costs 1,000 J: each bullet
    // crosses the first target (600 -> 400 m/s, 2 N·s) and stops in the second
    // (4 N·s). Six bullets on 2 kg boxes: 6 and 12 m/s.
    println!("targets {:?} {:?} {:?}", a[0].linear_vel, a[1].linear_vel, a[2].linear_vel);
    assert!((a[0].linear_vel[0] - 6.0).abs() < 0.05, "first target: {:?}", a[0].linear_vel);
    assert!((a[1].linear_vel[0] - 12.0).abs() < 0.1, "second target: {:?}", a[1].linear_vel);
    assert!(a[2].linear_vel[0].abs() < 1e-3, "nothing reached the third: {:?}", a[2].linear_vel);
}
