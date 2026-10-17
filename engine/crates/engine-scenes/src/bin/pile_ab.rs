//! A/B: does random orientation cause Box Pile energy vs aligned equal cubes?
use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::{AvbdParams, WorldCapacity, FLAG_REUSE_CONTACTS};
use glam::{Quat, Vec3};

fn main() {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "random".into());
    let steps: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(180);
    let n: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let h = n;
    let mut p = AvbdParams::default();
    p.iterations = 10;
    let need = (n * n * h + 8).next_power_of_two().max(4096);
    let mut world = MetalWorld::new(WorldCapacity {
        max_bodies: need,
        max_colliders: need,
        max_contacts: (need * 16).max(65_536),
        max_joints: 64,
        max_candidates: (need * 32).max(65_536),
        max_colors: 64,
        max_commands: 4096,
        max_hull_vertices: 64,
    })
    .expect("mw");
    p.flags |= FLAG_REUSE_CONTACTS;
    world.set_params(p);
    let mut seed = 12345u32;
    let mut rand = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as f32) / (u32::MAX as f32)
    };
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(0.5);
        for z in 0..h {
            for x in 0..n {
                for y in 0..n {
                    let half = if mode == "aligned" {
                        Vec3::splat(0.4)
                    } else {
                        Vec3::new(
                            0.25 + rand() * 0.35,
                            0.25 + rand() * 0.35,
                            0.25 + rand() * 0.35,
                        )
                    };
                    let pos = Vec3::new(
                        (x as f32 - n as f32 / 2.0) * 1.6,
                        2.0 + z as f32 * 1.6,
                        (y as f32 - n as f32 / 2.0) * 1.6,
                    );
                    let mass = 8.0 * half.x * half.y * half.z;
                    let id = b.add_box(pos, half, mass);
                    if mode == "random" {
                        let q = Quat::from_xyzw(
                            rand() - 0.5,
                            rand() - 0.5,
                            rand() - 0.5,
                            rand() - 0.5,
                        )
                        .normalize();
                        let mut body = b.world_mut().body(id);
                        body.orientation = q.to_array();
                        b.world_mut().set_body(id, body);
                    }
                }
            }
        }
    }
    for _ in 0..steps {
        world.step().expect("step");
    }
    let bodies = world.download_bodies();
    let mut ke = 0.0f32;
    let mut flying = 0u32;
    let mut awake = 0u32;
    let mut n_dyn = 0u32;
    for b in &bodies {
        if !b.is_dynamic() {
            continue;
        }
        n_dyn += 1;
        if (b.flags & 4) == 0 {
            awake += 1;
        }
        let v = glam::Vec3::from_array(b.linear_vel).length();
        if v > 1.0 {
            flying += 1;
        }
        let m = if b.inv_mass > 1e-12 {
            1.0 / b.inv_mass
        } else {
            0.0
        };
        ke += 0.5 * m * v * v;
    }
    let c = world.last_counters();
    println!(
        "mode={mode} n={n} steps={steps} dyn={n_dyn} awake={awake} contacts={} flying={flying} ke={ke:.1}",
        c.contact_count
    );
}
