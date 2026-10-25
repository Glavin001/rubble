//! Support loss in the world API (`DestructibleDef::support_loss`, threshold model): a slab on
//! a row of columns, and a block on three columns, built as scene packs with the bond
//! strengths and masses set exactly. One column is shot away (its bonds are weak in tension
//! and shear, so the strike breaks only them); what the rest does is support loss's call.
#![cfg(feature = "stress")]

use rubble::*;
use serde_json::{json, Value};

const G: f64 = 9.81;
/// Column section (m): 0.4 x 0.4, so every bond's area is 0.16 m².
const COLUMN_HALF: f64 = 0.2;
const AREA: f64 = 4.0 * COLUMN_HALF * COLUMN_HALF;

/// A pack: `columns` at the (x, z) given on footings (mass 0: supports), carrying one `top`
/// chunk of `top_mass` kg and half extents `top_half`. Column bonds hold `compression` Pa;
/// the columns listed in `weak` are bonded with negligible tension and shear (the shot breaks
/// them), the rest with plenty. Returns the pack and its chunk indices.
struct Frame {
    pack: Vec<u8>,
    /// Chunk indices: the footings, the columns (both in `columns` order) and the top.
    footings: Vec<u32>,
    columns: Vec<u32>,
    top: u32,
}

fn frame(
    columns: &[[f64; 2]],
    weak: &[usize],
    top_mass: f64,
    top_half: [f64; 3],
    compression: f64,
) -> Frame {
    let (footing_h, column_h) = (0.25, 1.0);
    let mut nodes = Vec::new();
    let mut sizes = Vec::new();
    let mut colliders = Vec::new();
    let mut materials = Vec::new();
    let mut node = |centre: [f64; 3], half: [f64; 3], mass: f64| {
        nodes.push(
            json!({ "centroid": { "x": centre[0], "y": centre[1], "z": centre[2] }, "mass": mass }),
        );
        sizes.push(json!({ "x": 2.0 * half[0], "y": 2.0 * half[1], "z": 2.0 * half[2] }));
        colliders.push(json!({ "kind": "cuboid", "halfExtents": { "x": half[0], "y": half[1], "z": half[2] } }));
        materials.push("concrete");
        nodes.len() as u32 - 1
    };
    let footings: Vec<u32> = columns
        .iter()
        .map(|&[x, z]| node([x, footing_h, z], [0.3, footing_h, 0.3], 0.0))
        .collect();
    let column_y = 2.0 * footing_h + column_h;
    let ids: Vec<u32> = columns
        .iter()
        .map(|&[x, z]| {
            node(
                [x, column_y, z],
                [COLUMN_HALF, column_h, COLUMN_HALF],
                400.0,
            )
        })
        .collect();
    let deck = 2.0 * (footing_h + column_h);
    let top = node([0.0, deck + top_half[1], 0.0], top_half, top_mass);
    let mut bonds = Vec::new();
    for (k, &[x, z]) in columns.iter().enumerate() {
        let m = if weak.contains(&k) { 1 } else { 0 };
        for (a, b, y) in [(footings[k], ids[k], 2.0 * footing_h), (ids[k], top, deck)] {
            bonds.push(json!({
                "node0": a, "node1": b, "m": m, "area": AREA,
                "centroid": { "x": x, "y": y, "z": z }, "normal": { "x": 0, "y": 1, "z": 0 },
            }));
        }
    }
    let material = |name: &str, tension: f64| {
        json!({
            "name": name, "elasticModulus": 30e9,
            "compressionElastic": compression, "compressionFatal": compression,
            "tensionElastic": tension, "tensionFatal": tension,
            "shearElastic": tension, "shearFatal": tension,
        })
    };
    let pack: Value = json!({
        "version": 2, "key": "support_frame", "title": "Support-loss frame",
        "defaults": { "solver": { "gravity": -G, "materialScale": 1,
            "materials": [material("concrete-member", 1e10), material("concrete-shot", 1e3)] } },
        "scenario": {
            "nodes": nodes, "nodeSizes": sizes, "nodeColliders": colliders,
            "nodeMaterials": materials, "bonds": bonds,
        },
    });
    Frame {
        pack: serde_json::to_vec(&pack).unwrap(),
        footings,
        columns: ids,
        top,
    }
}

/// The capacity the threshold tests give a scene (every split's piece a body).
fn capacity(asset: &DestructibleAssetDesc) -> WorldCapacity {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c
        .max_bodies
        .max(((chunks + 16 + 4) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c
}

struct Run {
    world: World,
    id: DestructibleId,
    asset: DestructibleAssetDesc,
    broken: usize,
}

/// Place the frame (support loss on or off, `factor` its dynamic factor), shoot the columns
/// `shoot` (indices into the frame's columns) from the side, step `frames`.
fn run(f: &Frame, support_loss: bool, factor: f32, shoot: &[usize], frames: u32) -> Run {
    let asset = DestructibleAssetDesc::from_pack(&f.pack).unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let offset = stress.placement().map(|x| -x as f32);
    let def = WorldDef {
        capacity: capacity(&asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let ground = w.create_body(&BodyDef::default()).unwrap();
    let h = stress.ground_height().unwrap_or(0.0) + offset[1];
    w.create_shape(
        ground,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: h,
        },
    )
    .unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let id = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            support_loss,
            support_loss_dynamic_factor: factor,
            ..DestructibleDef::new(a)
        })
        .unwrap();
    // Each shot from -z through the middle of its column, in the asset frame.
    for &column in shoot {
        let (r, m) = (0.3f32, 400.0f32);
        let p = asset.chunks[f.columns[column] as usize].pose.position;
        let at = [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2] - 2.5];
        let b = w
            .create_body(&BodyDef {
                linear_velocity: [0.0, 0.0, 40.0],
                is_bullet: true,
                ..BodyDef::dynamic(at)
            })
            .unwrap();
        w.create_sphere_shape(
            b,
            &ShapeDef {
                update_body_mass: false,
                ..ShapeDef::default()
            },
            [0.0; 3],
            r,
        )
        .unwrap();
        let i = 0.4 * m * r * r;
        w.body_set_mass_data(
            b,
            MassData {
                mass: m,
                center: [0.0; 3],
                rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
            },
        )
        .unwrap();
        w.body_set_projectile(b, Some(Projectile::default()))
            .unwrap();
    }
    let mut broken = 0;
    for n in 0..frames {
        if std::env::var("SL_DEBUG").is_ok() && n % 10 == 0 {
            let body = w.destructible_chunk_body(id, f.top).unwrap();
            let st = w.read_body_state(body).unwrap();
            let pose = asset.chunks[f.top as usize].pose.position;
            let p = glam::Quat::from_array(st.transform.rotation) * glam::Vec3::from(pose)
                + glam::Vec3::from(st.transform.position);
            let ty = w.body_type(body).ok();
            eprintln!(
                "frame {n}: top {p} body {body:?} {ty:?} v {:?}",
                st.linear_velocity
            );
        }
        w.step().unwrap();
        for fr in w.poll_blocking().unwrap() {
            for e in &fr.destruction_events {
                if let DestructionEvent::BondsBroken { bonds, .. } = e {
                    broken += bonds.len();
                }
            }
        }
    }
    Run {
        world: w,
        id,
        asset,
        broken,
    }
}

impl Run {
    /// Whether `chunk` still hangs on any of `footings` (one actor).
    fn on_footings(&self, chunk: u32, footings: &[u32]) -> bool {
        let actor = self.world.destructible_actor_of(self.id, chunk).unwrap();
        footings.iter().any(|f| actor.chunks.contains(f))
    }

    /// `chunk`'s centre of mass now (world frame), and its body's rotation.
    fn centre(&self, chunk: u32) -> (glam::Vec3, glam::Quat) {
        let body = self.world.destructible_chunk_body(self.id, chunk).unwrap();
        let st = self.world.read_body_state(body).unwrap();
        // Chunk world transform = body transform * the chunk's pose in the asset frame.
        let pose = self.asset.chunks[chunk as usize].pose.position;
        let q = glam::Quat::from_array(st.transform.rotation);
        (
            q * glam::Vec3::from(pose) + glam::Vec3::from(st.transform.position),
            q,
        )
    }
}

/// A 7.2 m slab of 100 t on four columns in a row (x = -3, -1, 1, 3); the one at x = -3 is
/// shot away. Intact, each column carries a quarter of the slab; after, a third.
fn slab_row(utilisation_after: f64) -> Frame {
    let weight = 1.0e5 * G;
    let compression = weight / 3.0 / utilisation_after / AREA;
    let columns = [[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]];
    frame(&columns, &[0], 1.0e5, [3.6, 0.2, 0.6], compression)
}

#[test]
fn a_slab_whose_column_is_shot_away_collapses_onto_the_rest() {
    // The remaining three columns' bonds at 120% of capacity once the fourth is gone (each
    // held 90% before: it stands until then).
    let f = slab_row(1.2);
    let start = run(&f, true, 1.0, &[], 1).centre(f.top).0;
    let off = run(&f, false, 1.0, &[0], 90);
    assert!(off.broken > 0, "the shot breaks the shot column's bonds");
    assert!(
        !off.on_footings(f.columns[0], &f.footings[..1]),
        "the shot column is off its footing"
    );
    assert!(
        off.on_footings(f.top, &f.footings[1..]),
        "without support loss the slab stands on three columns"
    );
    assert!((off.centre(f.top).0 - start).length() < 0.05);
    let on = run(&f, true, 1.0, &[0], 90);
    eprintln!(
        "support loss: {} bonds broken (without: {})",
        on.broken, off.broken
    );
    assert!(
        !on.on_footings(f.top, &f.footings),
        "the slab's weight breaks the columns left"
    );
    let drop = start.y - on.centre(f.top).0.y;
    assert!(drop > 0.3, "the slab comes down ({drop} m)");
}

#[test]
fn a_structure_intact_under_its_own_weight_never_breaks() {
    // Every column at 75% under the slab, the dynamic factor at its maximum: nothing struck,
    // nothing breaks, nothing moves.
    let f = slab_row(1.0);
    let start = run(&f, true, 2.0, &[], 1).centre(f.top).0;
    let r = run(&f, true, 2.0, &[], 120);
    assert_eq!(r.broken, 0);
    assert!(r.on_footings(f.top, &f.footings));
    assert!((r.centre(f.top).0 - start).length() < 0.01);
    // Struck, with every column left at 75% (the shot column's own bonds aside): the pass runs
    // on the piece and breaks nothing more.
    let f = slab_row(0.75);
    let r = run(&f, true, 1.0, &[0], 90);
    assert!(
        r.broken > 0 && r.on_footings(f.top, &f.footings[1..]),
        "{} broken",
        r.broken
    );
    assert!((r.centre(f.top).0 - start).length() < 0.05);
}

#[test]
fn a_sudden_loss_with_the_dynamic_factor_fails_a_member_the_static_flow_holds() {
    // After the loss each column left carries 60% of its capacity under the static flow.
    let f = slab_row(0.6);
    let quasi = run(&f, true, 1.0, &[0], 90);
    assert!(
        quasi.on_footings(f.top, &f.footings[1..]),
        "factor 1: 60% holds"
    );
    let dynamic = run(&f, true, 2.0, &[0], 90);
    assert!(
        !dynamic.on_footings(f.top, &f.footings),
        "factor 2: 120% of capacity breaks the columns"
    );
    eprintln!(
        "factor 1: {} bonds broken; factor 2: {}",
        quasi.broken, dynamic.broken
    );
    assert!(dynamic.broken > quasi.broken);
}

#[test]
fn a_block_whose_centre_leaves_its_footprint_tips() {
    // A 4 m block of 40 t on two columns at x = -1 (z = +-1) and one at x = 1.6, shot away:
    // its centre (x = 0) is then off the line the two make. The columns are strong: weight
    // alone breaks nothing, only the geometry does.
    let columns = [[-1.0, -1.0], [-1.0, 1.0], [1.6, 0.0]];
    let f = frame(&columns, &[2], 4.0e4, [2.0, 2.0, 1.4], 1.0e9);
    let (start, _) = run(&f, true, 1.0, &[], 1).centre(f.top);
    let off = run(&f, false, 1.0, &[2], 90);
    assert!(
        off.on_footings(f.top, &f.footings[..2]),
        "without support loss it hangs on two welds"
    );
    let on = run(&f, true, 1.0, &[2], 150);
    assert!(
        !on.on_footings(f.top, &f.footings),
        "it lets go of its last columns"
    );
    // The columns it let go of stay on their footings.
    for k in 0..2 {
        assert!(
            on.on_footings(f.columns[k], &f.footings[k..=k]),
            "column {k} stands"
        );
    }
    // Free of its welds it moves under gravity. (Where it ends up is the rigid engine's: see
    // `probe_a_dropped_block_lands`.)
    let (now, _) = on.centre(f.top);
    let body = on.world.destructible_chunk_body(on.id, f.top).unwrap();
    assert_eq!(on.world.body_type(body).unwrap(), BodyType::Dynamic);
    eprintln!("block: centre {start} -> {now}");
    assert!(
        now.y < start.y - 0.2,
        "it comes off its columns (centre {now})"
    );
}

#[test]
fn the_dynamic_factor_is_validated() {
    let f = slab_row(1.0);
    let asset = DestructibleAssetDesc::from_pack(&f.pack).unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let def = WorldDef {
        capacity: capacity(&asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let place = |w: &mut World, support_loss: bool, k: f32| {
        w.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            support_loss,
            support_loss_dynamic_factor: k,
            ..DestructibleDef::new(a)
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
    };
    for k in [0.5, 2.5, f32::NAN, f32::INFINITY] {
        let e = place(&mut w, true, k).unwrap_err();
        assert!(e.contains("outside 1 to 2"), "{k}: {e}");
    }
    let e = place(&mut w, false, 2.0).unwrap_err();
    assert!(e.contains("needs support_loss"), "{e}");
    assert_eq!(DestructibleDef::new(a).support_loss_dynamic_factor, 1.0);
    assert!(!DestructibleDef::new(a).support_loss);
    let e = w
        .create_destructible(&DestructibleDef {
            support_loss: true,
            ..DestructibleDef::new(a)
        })
        .unwrap_err()
        .to_string();
    assert!(e.contains("Threshold strength model"), "{e}");
}

/// Probe (ignored), a finding outside support loss: with support loss off, all three of the
/// block's columns shot away, the free block falls through the world's ground plane (and
/// through static pieces) instead of landing. Pieces of the cuboid house land on the same
/// plane. `SL_DEBUG=1` prints the block's centre every 10 frames.
#[test]
#[ignore]
fn probe_a_dropped_block_lands() {
    let columns = [[-1.0, -1.0], [-1.0, 1.0], [1.6, 0.0]];
    let f = frame(&columns, &[0, 1, 2], 4.0e4, [2.0, 2.0, 1.4], 1.0e9);
    let r = run(&f, false, 1.0, &[0, 1, 2], 150);
    let (now, _) = r.centre(f.top);
    eprintln!(
        "{} bonds broken; the block's centre at {now} (it lands at y = 2)",
        r.broken
    );
    assert!(now.y > 1.5, "the block fell through the ground: {now}");
}
