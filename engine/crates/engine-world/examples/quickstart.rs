//! The README's quick start: a box, a destructible arch and a ray, stepped
//! without stalling the GPU.
//!
//! `cargo run -p engine-world --release --example quickstart`

use rubble::*;

fn main() -> anyhow::Result<()> {
    let mut world = World::new(GpuContext::Default, &WorldDef::default())?;

    // Ground: a static body (the BodyDef default) with a box shape.
    let ground = world.create_body(&BodyDef {
        position: [0.0, -0.5, 0.0],
        ..BodyDef::default()
    })?;
    world.create_box_shape(ground, &ShapeDef::default(), [40.0, 0.5, 40.0])?;

    // A dynamic crate.
    let crate_ = world.create_body(&BodyDef::dynamic([0.0, 4.0, 0.0]))?;
    world.create_box_shape(crate_, &ShapeDef::default(), [0.5; 3])?;
    world.body_apply_linear_impulse_to_center(crate_, [2.0, 0.0, 0.0], true)?;

    // A destructible from a Shardwright fracture asset (Git LFS).
    let glb = std::fs::read("crates/engine-webgpu/fracture/assets/masonry-arch.glb")?;
    let asset = world.create_destructible_asset(&DestructibleAssetDesc::from_glb(&glb, 1)?)?;
    let mut arch = DestructibleDef::new(asset);
    arch.transform = Transform::new([6.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
    world.create_destructible(&arch)?;

    let ray = world.cast_rays(&[RayInput {
        origin: [0.0, 10.0, 0.0],
        translation: [0.0, -20.0, 0.0],
        ignore_body: None,
    }])?;

    for _ in 0..600 {
        world.step()?; // one fixed step; never waits for the GPU
        for frame in world.poll()? {
            // Finished steps, oldest first.
            for m in &frame.body_events.moves {
                // Interpolate m.previous -> m.transform in your renderer.
                let _ = (m.body, m.transform, m.previous, m.fell_asleep);
            }
            for e in &frame.destruction_events {
                if let DestructionEvent::Split { children, .. } = e {
                    println!("step {}: split into {} pieces", frame.step, children.len());
                }
            }
            for r in frame.ray_results.iter().filter(|r| r.batch == ray) {
                println!("ray hit: {:?}", r.hits[0]);
            }
        }
    }
    world.poll_blocking()?;
    println!("crate at {:?}", world.body_transform(crate_)?);
    Ok(())
}
