//! Diagnostic: KE + contact warm-start survival for a 10-box stack.
use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::{AvbdParams, WorldCapacity};
use glam::Vec3;
use std::collections::HashSet;

fn ke(world: &MetalWorld) -> f32 {
    world
        .download_bodies()
        .iter()
        .filter(|b| b.is_dynamic())
        .map(|b| {
            let v = glam::Vec3::from_array(b.linear_vel);
            let w = glam::Vec3::from_array(b.angular_vel);
            // mass≈1, rough I≈0.166 for unit box
            0.5 * v.length_squared() + 0.5 * 0.166 * w.length_squared()
        })
        .sum()
}

fn contact_keys(world: &MetalWorld) -> HashSet<(u32, u32, u32)> {
    let mut set = HashSet::new();
    for c in world.download_contacts() {
        if c.alive == 0 {
            continue;
        }
        let a = c.collider_a.min(c.collider_b);
        let b = c.collider_a.max(c.collider_b);
        set.insert((a, b, c.feature_id));
    }
    set
}

fn main() {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    let mut p = AvbdParams::default();
    p.iterations = 20;
    world.set_params(p);
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(0.0);
        for i in 0..10u32 {
            b.add_box(Vec3::new(0.0, 0.5 + i as f32, 0.0), Vec3::splat(0.5), 1.0);
        }
    }
    let mut prev = HashSet::new();
    for s in 0..400 {
        world.step().unwrap();
        let keys = contact_keys(&world);
        let surv = if keys.is_empty() || prev.is_empty() {
            0.0
        } else {
            keys.intersection(&prev).count() as f32 / keys.len() as f32
        };
        if s % 20 == 0 || s < 5 {
            let c = world.last_counters();
            let sample: Vec<_> = keys.iter().take(3).collect();
            println!(
                "step={s} ke={:.4} contacts={} surv={:.3} awake={} sample={sample:?}",
                ke(&world),
                c.contact_count,
                surv,
                c.awake_count
            );
        }
        prev = keys;
    }
}
