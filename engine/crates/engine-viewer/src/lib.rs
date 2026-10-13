//! Instanced draw over the physics pose buffer.
//! The viewer holds a reference to the Metal body buffer; it never downloads poses each frame.

use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::WorldCapacity;
use glam::Vec3;

pub struct ViewerApp {
    pub world: MetalWorld,
}

impl ViewerApp {
    pub fn demo_pile() -> anyhow::Result<Self> {
        let mut world = MetalWorld::new(WorldCapacity::small())?;
        {
            let mut b = SceneBuilder::new(&mut world);
            b.add_ground_plane(0.0);
            b.pile_spheres(512, 0.35);
        }
        Ok(Self { world })
    }

    pub fn frame(&mut self) -> anyhow::Result<()> {
        self.world.step()?;
        // Draw would encode a render command buffer on a second queue reading poses_buffer().
        let _poses = self.world.poses_buffer();
        Ok(())
    }

    pub fn run_headless(&mut self, frames: u32) -> anyhow::Result<()> {
        for i in 0..frames {
            self.frame()?;
            if i % 30 == 0 {
                let t = self.world.last_timings();
                println!(
                    "viewer frame={i} gpu={:.3}ms cpu_submit={:.3}ms bodies={}",
                    t.total_gpu,
                    t.cpu_submit,
                    self.world.body_count()
                );
            }
        }
        Ok(())
    }
}

pub fn run_destruction_demo() -> anyhow::Result<()> {
    let mut world = MetalWorld::new(WorldCapacity::default())?;
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(0.0);
        for y in 0..12 {
            for x in 0..12 {
                for z in 0..4 {
                    b.add_box(
                        Vec3::new(x as f32 * 0.5 - 3.0, 0.5 + y as f32 * 0.5, z as f32 * 0.5),
                        Vec3::splat(0.22),
                        1.0,
                    );
                }
            }
        }
    }
    println!(
        "destruction demo chip={} bodies={} (Metal buffer-shared draw)",
        world.chip,
        world.body_count()
    );
    for i in 0..180 {
        world.step()?;
        if i % 60 == 0 {
            let t = world.last_timings();
            println!("frame={i} gpu={:.3}ms contacts={}", t.total_gpu, world.last_counters().contact_count);
        }
    }
    Ok(())
}
