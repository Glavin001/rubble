//! Destructibles in the world API: descriptor-first assets from every source,
//! the Bonds strength model, the render contract, and Blast-style split
//! events.

use rubble::*;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 16_384,
        max_colliders: 32_768,
        max_contacts: 262_144,
        max_joints: 32_768,
        max_candidates: 524_288,
        max_colors: 64,
        max_commands: 1024,
        max_hull_vertices: 1 << 20,
    }
}

fn world() -> World {
    let def = WorldDef {
        capacity: capacity(),
        features: EngineFeatures {
            grid_levels: Some(true),
            ..EngineFeatures::default()
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -1.0, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [200.0, 1.0, 200.0])
        .unwrap();
    w
}

fn house() -> DestructibleAssetDesc {
    DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap()
}

fn run(w: &mut World, steps: u32) -> Vec<Frame> {
    let mut frames = Vec::new();
    for _ in 0..steps {
        w.step().unwrap();
        frames.extend(w.poll().unwrap());
    }
    frames.extend(w.poll_blocking().unwrap());
    frames
}

fn shoot(w: &mut World, from: [f32; 3], velocity: [f32; 3], radius: f32, mass: f32) -> BodyId {
    let b = w
        .create_body(&BodyDef {
            linear_velocity: velocity,
            ..BodyDef::dynamic(from)
        })
        .unwrap();
    let volume = 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3);
    let s = ShapeDef {
        density: mass / volume,
        ..ShapeDef::default()
    };
    w.create_sphere_shape(b, &s, [0.0; 3], radius).unwrap();
    b
}

#[test]
fn every_source_gives_a_valid_descriptor() {
    let h = house();
    assert!(
        h.chunks.len() > 10 && h.bonds.len() > 10,
        "{} chunks",
        h.chunks.len()
    );
    assert!(
        h.bonds.iter().any(|b| b.b.is_none()),
        "a house is anchored to the ground"
    );
    let glb = include_bytes!("../../engine-webgpu/fracture/assets/masonry-arch.glb");
    let arch = DestructibleAssetDesc::from_glb(glb, 1).unwrap();
    assert!(!arch.chunks.is_empty());
    assert!(
        arch.chunks
            .iter()
            .all(|c| c.fragment.is_some() && c.node.is_some()),
        "every GLB chunk maps to its fragment and glTF node"
    );
    assert!(arch
        .chunks
        .iter()
        .flat_map(|c| &c.shapes)
        .any(|g| matches!(g, Geometry::Hull { .. })));
    let pack = include_bytes!("../../../stress-oracle/packs/house-1story.json");
    let p = DestructibleAssetDesc::from_pack(pack).unwrap();
    assert!(!p.chunks.is_empty() && p.name.is_some());
    // Invalid descriptors are refused, never repaired.
    let mut bad = h.clone();
    bad.bonds[0].a = 99_999;
    assert!(bad.validate().is_err());
}

#[test]
fn a_house_stands_and_obeys_the_render_contract() {
    let mut w = world();
    let asset = w.create_destructible_asset(&house()).unwrap();
    let at = Transform::new([5.0, 0.0, -3.0], [0.0, 0.38268343, 0.0, 0.9238795]);
    let d = w
        .create_destructible(&DestructibleDef {
            transform: at,
            ..DestructibleDef::new(asset)
        })
        .unwrap();
    let stats = w.destructible_stats(d).unwrap();
    assert_eq!(stats.bonds_intact, stats.bonds);
    let frames = run(&mut w, 120);
    assert!(
        frames.iter().all(|f| f.destruction_events.is_empty()),
        "a house holds under its own weight"
    );
    assert_eq!(w.destructible_stats(d).unwrap(), stats, "no actor split");
    // Render contract: every chunk body's origin is the instance frame, and a
    // chunk's transform relative to its body is fixed (identity in Bonds).
    for chunk in [0, stats.chunks / 2, stats.chunks - 1] {
        let body = w.destructible_chunk_body(d, chunk).unwrap();
        assert_eq!(
            w.destructible_chunk_local(d, chunk).unwrap(),
            Transform::IDENTITY
        );
        let t = w.read_body_state(body).unwrap().transform;
        let dp = glam::Vec3::from(t.position) - glam::Vec3::from(at.position);
        assert!(
            dp.length() < 0.02,
            "chunk {chunk} body origin drifted: {t:?}"
        );
        let dq = glam::Quat::from_array(t.rotation).dot(glam::Quat::from_array(at.rotation));
        assert!(dq.abs() > 0.9999, "chunk {chunk} body rotation: {t:?}");
    }
}

#[test]
fn a_shot_breaks_bonds_and_splits_actors() {
    let mut w = world();
    let asset = w.create_destructible_asset(&house()).unwrap();
    let d = w.create_destructible(&DestructibleDef::new(asset)).unwrap();
    let before = w.destructible_stats(d).unwrap();
    // A heavy ball through the house at speed.
    shoot(&mut w, [-15.0, 1.5, 0.0], [60.0, 0.0, 0.0], 0.6, 2000.0);
    let frames = run(&mut w, 120);
    let mut broken = 0;
    let mut splits = 0;
    for e in frames.iter().flat_map(|f| f.destruction_events.iter()) {
        match e {
            DestructionEvent::BondsBroken {
                destructible,
                bonds,
            } => {
                assert_eq!(*destructible, d);
                broken += bonds.len();
            }
            DestructionEvent::Split {
                destructible,
                parent,
                children,
            } => {
                assert_eq!(*destructible, d);
                splits += 1;
                assert!(children.len() >= 2);
                let mut all: Vec<u32> = children.iter().flat_map(|c| c.chunks.clone()).collect();
                all.sort_unstable();
                assert_eq!(all, parent.chunks, "children partition the parent");
                let mass: f32 = children.iter().map(|c| c.mass).sum();
                assert!((mass - parent.mass).abs() <= 1e-3 * parent.mass.max(1.0));
                assert!(children.iter().all(|c| c.bodies.len() == c.chunks.len()));
            }
            _ => {}
        }
    }
    assert!(broken > 0, "the shot broke bonds");
    assert!(splits > 0, "pieces came loose");
    let after = w.destructible_stats(d).unwrap();
    assert_eq!(
        after.bonds_intact as usize,
        before.bonds_intact as usize - broken
    );
    assert!(after.actors > 1);
    // Actor queries agree with the split.
    let one = w.destructible_actor_of(d, 0).unwrap();
    assert!(one.chunks.contains(&0));
}

#[test]
fn threshold_model_is_refused_until_it_lands() {
    let mut w = world();
    let asset = w.create_destructible_asset(&house()).unwrap();
    let def = DestructibleDef {
        model: StrengthModel::Threshold,
        ..DestructibleDef::new(asset)
    };
    let e = w.create_destructible(&def).unwrap_err();
    assert!(e.downcast_ref::<Unsupported>().is_some(), "{e}");
}

#[test]
fn destroying_a_destructible_removes_its_bodies() {
    let mut w = world();
    let asset = w.create_destructible_asset(&house()).unwrap();
    let before = w.body_count();
    let d = w.create_destructible(&DestructibleDef::new(asset)).unwrap();
    assert!(w.body_count() > before);
    let chunk0 = w.destructible_chunk_body(d, 0).unwrap();
    w.destroy_destructible(d).unwrap();
    assert!(!w.destructible_is_valid(d));
    assert!(!w.body_is_valid(chunk0));
    assert_eq!(w.body_count(), before);
}
