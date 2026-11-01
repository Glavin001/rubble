//! The awake-indexed broadphase (`EngineFeatures::awake_broadphase`) finds exactly
//! the full broadphase's candidate pairs, on the same GPU state, step after step.
//!
//! A hierarchical-grid (hull) world: a static ground and static blocks of several
//! sizes, piles of small hull cubes that settle and fall asleep, a large dynamic block
//! (a coarse mover over fine sleepers: the finer-cell scan), a moving and a stopped
//! kinematic, small bodies thrown into the piles (wakes, new sleepers, resting-table
//! rebuilds) and a fast CCD bullet (the fast list). Every few steps the broadphase runs on the same state both ways
//! (`debug_broadphase_pairs`): the pair lists and the collider bounds must be equal.

use engine_schema::convex::{convex_collider_params, ConvexHull, HullArena};
use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity};
use engine_webgpu::WgpuSim;
use glam::Vec3;

const DYNAMIC: u32 = 1 | 8;
const KINEMATIC: u32 = 2 | 8;

struct Scene {
    sim: WgpuSim,
    arena: HullArena,
}

impl Scene {
    fn block(&mut self, pos: Vec3, half: Vec3, inv_mass: f32, flags: u32, v: Vec3) -> u32 {
        let hull = ConvexHull::cuboid(half);
        let radius = hull.bounding_radius();
        let base = self.arena.insert(&hull).expect("hull");
        let inertia = |a: f32, b: f32| if inv_mass > 0.0 { 3.0 * inv_mass / (a * a + b * b) } else { 0.0 };
        let p = pos.to_array();
        let b = self.sim.add_body(BodyState {
            position: p,
            inv_mass,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [
                inertia(half.y, half.z),
                inertia(half.x, half.z),
                inertia(half.x, half.y),
            ],
            flags,
            linear_vel: v.to_array(),
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: p,
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        });
        self.sim.add_collider(Collider {
            body: b,
            shape: ShapeType::ConvexHull as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: convex_collider_params(base, 1.0, radius),
        });
        b
    }
}

fn sorted(mut v: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    v.sort_unstable();
    v
}

#[test]
fn awake_broadphase_finds_the_full_broadphase_pairs() {
    let sim = WgpuSim::new(WorldCapacity {
        max_bodies: 4096,
        max_colliders: 4096,
        max_contacts: 131_072,
        max_joints: 64,
        max_candidates: 262_144,
        max_colors: 32,
        max_commands: 256,
        max_hull_vertices: 65_536,
    })
    .expect("wgpu");
    let mut s = Scene {
        sim,
        arena: HullArena::new(),
    };
    let mut p = AvbdParams::default();
    p.dt = 1.0 / 60.0;
    s.sim.set_params(p);
    let zero = Vec3::ZERO;
    // Static ground and static blocks of three sizes (three grid levels).
    s.block(Vec3::new(0.0, -0.5, 0.0), Vec3::new(30.0, 0.5, 30.0), 0.0, 0, zero);
    for k in 0..6 {
        let h = [0.3, 0.8, 1.6][k % 3];
        s.block(Vec3::new(-12.0 + 4.0 * k as f32, h, 9.0), Vec3::splat(h), 0.0, 0, zero);
    }
    // Four piles of 0.2 m cubes, stacked exactly: they settle and sleep.
    let piles = [(-8.0, -6.0), (-2.0, -6.0), (4.0, -6.0), (10.0, -6.0)];
    for &(x, z) in &piles {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..4 {
                    let at = Vec3::new(x + 0.42 * i as f32, 0.2 + 0.4 * k as f32, z + 0.42 * j as f32);
                    s.block(at, Vec3::splat(0.2), 1.0, DYNAMIC, zero);
                }
            }
        }
    }
    // A stopped kinematic beside the second pile, a moving one heading for the third.
    s.block(Vec3::new(-2.0, 0.5, -3.0), Vec3::splat(0.5), 0.0, KINEMATIC, zero);
    s.block(Vec3::new(4.5, 0.6, 2.0), Vec3::new(0.6, 0.4, 0.6), 0.0, KINEMATIC, Vec3::new(0.0, 0.0, -1.5));
    s.sim.set_hulls(&mut s.arena);
    let mut compared = 0;
    let mut most_asleep = 0;
    let mut thrown = Vec::new();
    for step in 0..480u32 {
        // Once the piles sleep: a coarse block slides over the first pile, then small
        // cubes are thrown into the others (wakes, new sleepers).
        if step == 200 {
            s.block(Vec3::new(-11.0, 2.5, -5.4), Vec3::new(1.4, 0.3, 1.4), 0.05, DYNAMIC, Vec3::new(3.0, 0.0, 0.0));
            s.sim.set_hulls(&mut s.arena);
        }
        if step == 300 {
            // A fast CCD bullet through the last pile (the fast list walks both tables).
            s.block(Vec3::new(4.0, 0.6, -5.6), Vec3::splat(0.1), 2.0, DYNAMIC | 256, Vec3::new(45.0, 0.0, 0.0));
            s.sim.set_hulls(&mut s.arena);
        }
        if step >= 240 && step % 40 == 0 && thrown.len() < 4 {
            let (x, z) = piles[1 + thrown.len() % 3];
            let b = s.block(Vec3::new(x + 0.4, 3.0, z + 0.4), Vec3::splat(0.15), 1.0, DYNAMIC, Vec3::new(0.0, -6.0, 0.0));
            thrown.push(b);
            s.sim.set_hulls(&mut s.arena);
        }
        if step % 6 == 3 {
            // Collider bounds (binding 9 past the body slots), without the spare word
            // the grids keep their chain links in.
            let bounds = |sim: &WgpuSim| -> Vec<u32> {
                let n = sim.collider_count() as usize;
                let at = u64::from(sim.capacity().max_bodies) * 8;
                let mut w = sim.read_words(9, at, n * 8);
                for box_ in w.chunks_mut(8) {
                    box_[7] = 0;
                }
                w
            };
            s.sim.set_awake_broadphase(true);
            let awake = sorted(s.sim.debug_broadphase_pairs());
            let awake_bounds = bounds(&s.sim);
            s.sim.set_awake_broadphase(false);
            let full = sorted(s.sim.debug_broadphase_pairs());
            let full_bounds = bounds(&s.sim);
            s.sim.set_awake_broadphase(true);
            assert!(awake_bounds == full_bounds, "step {step}: collider bounds differ");
            if awake != full {
                let bodies = s.sim.download_bodies();
                let show = |v: &Vec<(u32, u32)>, w: &Vec<(u32, u32)>| -> Vec<String> {
                    v.iter()
                        .filter(|x| !w.contains(x))
                        .map(|&(a, b)| {
                            let (ca, cb) = (s.sim.collider(a), s.sim.collider(b));
                            let f = |c: &Collider| bodies[c.body as usize].flags;
                            format!("({a} b{} f{}, {b} b{} f{})", ca.body, f(&ca), cb.body, f(&cb))
                        })
                        .collect()
                };
                panic!(
                    "step {step}: awake-only extra {:?}; full-only extra {:?}",
                    show(&awake, &full),
                    show(&full, &awake)
                );
            }
            assert_eq!(awake, full, "step {step}: pairs differ");
            compared += usize::from(!full.is_empty());
        }
        let c = s.sim.step().expect("step");
        let asleep = s.sim.download_bodies().iter().filter(|b| b.is_dynamic() && b.is_sleeping()).count();
        most_asleep = most_asleep.max(asleep);
        let _ = c;
    }
    // The comparison covered steps with pairs, and the piles really slept (so the
    // resting table held sleepers while movers queried it).
    assert!(compared > 40, "compared {compared} steps with pairs");
    assert!(most_asleep >= 100, "at most {most_asleep} bodies asleep");
}
