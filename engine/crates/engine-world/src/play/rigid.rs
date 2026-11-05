//! The rigid benchmark scenes (wall, ring, gables, pyramid, pile, columns, the large tiers)
//! placed in a [`World`]: engine-scenes' generators and scene table (`engine_scenes::web`,
//! shared with the native viewers), their body records, and the records made into bodies.
//! The web demo (`demo.html`) and any host place them the same way.

use crate::*;
pub use engine_scenes::web::{
    bodies, records, scenes, scenes_json, upload_order, Body, Capacity, Piece, Shape, WebScene,
    BODY_RECORD, BUILT,
};

/// Make bodies from [`BODY_RECORD`]-float records (one shape and the mass data each), in
/// record order: type (0 static, 1 kinematic, 2 dynamic), bullet (0 / 1), position xyz,
/// rotation xyzw, linear and angular velocity, mass, principal inertia xyz (centre at the
/// origin, axes aligned; ignored unless dynamic), shape (0 sphere: radius; 2 box: half
/// extents), its three numbers, then the shape's density. `shape` is the `ShapeDef`
/// (material, filter, ...) every shape takes; the records carry the mass. Returns the
/// bodies.
pub fn create_bodies(
    world: &mut World,
    records: &[f32],
    shape: &ShapeDef,
) -> anyhow::Result<Vec<BodyId>> {
    anyhow::ensure!(
        records.len() % BODY_RECORD == 0,
        "createBodies: {} floats is not whole records of {BODY_RECORD}",
        records.len()
    );
    let rows: Vec<&[f32]> = records.chunks(BODY_RECORD).collect();
    let mut defs = Vec::with_capacity(rows.len());
    for r in &rows {
        let body_type = match r[0] as u32 {
            0 => BodyType::Static,
            1 => BodyType::Kinematic,
            2 => BodyType::Dynamic,
            t => anyhow::bail!("createBodies: body type {t}"),
        };
        defs.push(BodyDef {
            body_type,
            is_bullet: r[1] != 0.0,
            position: [r[2], r[3], r[4]],
            rotation: [r[5], r[6], r[7], r[8]],
            linear_velocity: [r[9], r[10], r[11]],
            angular_velocity: [r[12], r[13], r[14]],
            ..BodyDef::default()
        });
    }
    let ids = world.create_bodies(&defs)?;
    let shape = ShapeDef {
        update_body_mass: false,
        ..shape.clone()
    };
    for (b, r) in ids.iter().zip(&rows) {
        let geometry = match r[19] as u32 {
            0 => Geometry::Sphere {
                center: [0.0; 3],
                radius: r[20],
            },
            2 => Geometry::Box {
                half: [r[20], r[21], r[22]],
                transform: Transform::IDENTITY,
            },
            s => anyhow::bail!("createBodies: shape {s}"),
        };
        let def = ShapeDef {
            density: r[23],
            ..shape.clone()
        };
        world.create_shape(*b, &def, geometry)?;
        if r[0] as u32 == 2 {
            let mass = MassData::principal(r[15], [0.0; 3], [r[16], r[17], r[18]]);
            world.body_set_mass_data(*b, mass)?;
        }
    }
    Ok(ids)
}

/// A built scene placed in a world.
#[derive(Clone, Debug)]
pub struct Placed {
    /// The bodies in upload (slot) order.
    pub bodies: Vec<BodyId>,
    /// `gpu_to_scene[slot]`: the scene body each slot holds (identity unless spatial).
    pub gpu_to_scene: Vec<u32>,
    /// Each scene body's starting position (scene order).
    pub initial: Vec<[f32; 3]>,
}

/// Place built scene `name` ([`BUILT`]) in `world`: its ground slab (the scene table's
/// extent) and pieces, `levels` for the pyramid, `ccd` for its balls, uploaded in Morton
/// order when `spatial`. `shape` is every shape's `ShapeDef` (its material: the scene's
/// friction).
pub fn place(
    world: &mut World,
    name: &str,
    levels: u32,
    ccd: bool,
    spatial: bool,
    shape: &ShapeDef,
) -> anyhow::Result<Placed> {
    let table = scenes();
    let (_, scene) = table
        .iter()
        .find(|(n, _)| *n == name)
        .ok_or_else(|| anyhow::anyhow!("unknown scene {name}"))?;
    anyhow::ensure!(
        scene.snapshot.is_none(),
        "{name} is a snapshot scene (import its RUBSNAP1 records)"
    );
    let bodies = bodies(name, levels, scene.ground, ccd)?;
    if let Some(cap) = scene.cap {
        anyhow::ensure!(
            bodies.len() <= cap.max_bodies as usize,
            "{} bodies exceed capacity {}",
            bodies.len(),
            cap.max_bodies
        );
    }
    let gpu_to_scene = upload_order(&bodies, spatial)?;
    let ids = create_bodies(world, &records(&bodies, &gpu_to_scene), shape)?;
    Ok(Placed {
        bodies: ids,
        gpu_to_scene,
        initial: bodies.iter().map(|b| b.p.map(|x| x as f32)).collect(),
    })
}
