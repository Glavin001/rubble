//! Chunk crushing in the threshold mode (`DestructibleDef::crush`,
//! engine/docs/destruction-reactions.md section 1), from the law's definition and
//! physical invariants: a structure under its own weight never crushes (and its stress
//! estimate is its weight over its section); a hard projectile crushes the chunks it
//! strikes and pays their crush energy out of its kinetic energy; `Conserve` keeps the
//! chunk's mass and the system's momentum; `Remove` takes the mass out; with nothing to
//! crush the switch changes nothing; a crush switch with nothing crushable is an error.
#![cfg(feature = "stress")]

use glam::Vec3;
use rubble::*;

/// The punch-through wall (ScenePack v3, blast-stress-solver `crush-wall.json`): 12 x 8 x 2
/// blocks of 0.4 m on a footing of zero-mass anchors; `block-wall` crushes
/// (capPressure 20 MPa, cohesion 4.8 MPa, slope 1.2, 4 MJ/m^3, 2e5 Pa s).
fn crush_wall_json() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../engine-webgpu/fracture/assets/crush-wall.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The wall without its footing, lowered onto the ground: one free cluster.
fn free_wall_json() -> serde_json::Value {
    let mut json = crush_wall_json();
    let sc = &mut json["scenario"];
    let nodes = sc["nodes"].as_array().unwrap().clone();
    let keep: Vec<usize> = (0..nodes.len())
        .filter(|&i| nodes[i]["mass"].as_f64().unwrap() > 0.0)
        .collect();
    let mut new_index = vec![usize::MAX; nodes.len()];
    for (n, &i) in keep.iter().enumerate() {
        new_index[i] = n;
    }
    let drop = 0.4;
    for key in ["nodes", "nodeTypes", "nodeSizes", "nodeColliders"] {
        if let Some(a) = sc[key].as_array() {
            let a: Vec<serde_json::Value> = keep.iter().map(|&i| a[i].clone()).collect();
            sc[key] = a.into();
        }
    }
    for n in sc["nodes"].as_array_mut().unwrap() {
        let y = n["centroid"]["y"].as_f64().unwrap();
        n["centroid"]["y"] = (y - drop).into();
    }
    let bonds: Vec<serde_json::Value> = sc["bonds"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|b| {
            let (a, c) = (
                new_index[b["node0"].as_u64()? as usize],
                new_index[b["node1"].as_u64()? as usize],
            );
            if a == usize::MAX || c == usize::MAX {
                return None;
            }
            let mut b = b.clone();
            b["node0"] = a.into();
            b["node1"] = c.into();
            let y = b["centroid"]["y"].as_f64().unwrap();
            b["centroid"]["y"] = (y - drop).into();
            Some(b)
        })
        .collect();
    sc["bonds"] = bonds.into();
    json
}

fn asset_of(json: &serde_json::Value) -> DestructibleAssetDesc {
    DestructibleAssetDesc::from_pack(&serde_json::to_vec(json).unwrap()).unwrap()
}

/// The capacity the lab gives the scene, with room for `extra` bodies (the fragments).
fn capacity(asset: &DestructibleAssetDesc, extra: u32) -> WorldCapacity {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c
        .max_bodies
        .max(((chunks + 16 + 4) as u32 + extra).next_power_of_two());
    c.max_colliders = c.max_colliders.max(c.max_bodies + chunks as u32 + 64);
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c
}

/// A world with the asset's threshold solver; `gravity` and a ground plane at the stress
/// scene's grade optional.
fn world_for(asset: &DestructibleAssetDesc, extra: u32, gravity: bool, ground: bool) -> World {
    let solver = asset
        .solver_for(StrengthModel::Threshold)
        .expect("threshold solver");
    let def = WorldDef {
        capacity: capacity(asset, extra),
        gravity: [0.0, if gravity { solver.gravity_y } else { 0.0 }, 0.0],
        solver: engine_schema::AvbdParams {
            gravity_y: if gravity { solver.gravity_y } else { 0.0 },
            ..solver
        },
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    if let (true, Some(h)) = (ground, stress.ground_height()) {
        let placement = stress.placement();
        let g = w.create_body(&BodyDef::default()).unwrap();
        w.create_shape(
            g,
            &ShapeDef::default(),
            Geometry::Plane {
                normal: [0.0, 1.0, 0.0],
                offset: h - placement[1] as f32,
            },
        )
        .unwrap();
    }
    w
}

/// Place the asset (its stress frame at the world's origin, as the lab places it).
fn place(
    w: &mut World,
    asset: &DestructibleAssetDesc,
    crush: Option<Crush>,
) -> anyhow::Result<DestructibleId> {
    let offset = asset
        .stress
        .as_ref()
        .unwrap()
        .placement()
        .map(|x| -x as f32);
    let a = w.create_destructible_asset(asset)?;
    w.create_destructible(&DestructibleDef {
        model: StrengthModel::Threshold,
        transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
        asleep: false,
        crush,
        ..DestructibleDef::new(a)
    })
}

/// A ball of the pack's projectile radius (0.45 m) at the block in the wall's fourth
/// row, 0.2 m right of its middle, from 4 m in front, at `speed` (m/s) along -x (the wall
/// is two blocks thick along x, twelve wide along z).
fn shoot(w: &mut World, speed: f32, mass: f32) -> BodyId {
    let r = 0.45f32;
    let ball = w
        .create_body(&BodyDef {
            linear_velocity: [-speed, 0.0, 0.0],
            ..BodyDef::dynamic([4.0, 1.8, 0.2])
        })
        .unwrap();
    w.create_sphere_shape(
        ball,
        &ShapeDef {
            update_body_mass: false,
            ..ShapeDef::default()
        },
        [0.0; 3],
        r,
    )
    .unwrap();
    let i = 0.4 * mass * r * r;
    w.body_set_mass_data(
        ball,
        MassData {
            mass,
            center: [0.0; 3],
            rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
        },
    )
    .unwrap();
    w.body_set_projectile(ball, Some(Projectile::default()))
        .unwrap();
    ball
}

/// Per chunk (global order): volume (m^3), density, crush energy (J/m^3, 0: none).
fn chunk_data(asset: &DestructibleAssetDesc) -> Vec<(f64, f64, f64)> {
    let scene = asset
        .stress
        .as_ref()
        .unwrap()
        .threshold_scene(1.0, 1.0 / 60.0)
        .unwrap();
    let world = stress_ref::world::World::new(&scene);
    world
        .solver
        .structures
        .iter()
        .flat_map(|s| &s.chunks)
        .map(|c| {
            let m = scene.material(&c.material);
            (
                c.volume(),
                m.density,
                m.crush.map_or(0.0, |p| p.crush_energy),
            )
        })
        .collect()
}

fn crushes(frame: &Frame) -> Vec<DestructionEvent> {
    frame
        .destruction_events
        .iter()
        .filter(|e| matches!(e, DestructionEvent::ChunkCrushed { .. }))
        .cloned()
        .collect()
}

/// Every live dynamic body's linear momentum and kinetic energy (blocking reads).
fn momentum(w: &World, bodies: &[BodyId]) -> (Vec3, f64) {
    let mut p = Vec3::ZERO;
    let mut ke = 0.0;
    for &b in bodies.iter().filter(|b| w.body_is_valid(**b)) {
        let Some(m) = w.body_mass_data(b).unwrap() else {
            continue;
        };
        let v = Vec3::from(w.read_body_state(b).unwrap().linear_velocity);
        p += v * m.mass;
        ke += 0.5 * m.mass as f64 * v.length_squared() as f64;
    }
    (p, ke)
}

/// Every body a destructible may be on now (its chunks', its fragments').
fn live_bodies(w: &World, id: DestructibleId, extra: &[BodyId]) -> Vec<BodyId> {
    let stats = w.destructible_stats(id).unwrap();
    let mut out: Vec<BodyId> = (0..stats.chunks)
        .filter_map(|c| w.destructible_chunk_body(id, c).ok())
        .filter(|b| w.body_is_valid(*b))
        .collect();
    out.extend_from_slice(extra);
    out.sort_by_key(|b| b.index());
    out.dedup();
    out
}

#[test]
fn a_wall_under_its_own_weight_never_crushes() {
    let asset = asset_of(&free_wall_json());
    let mut w = world_for(&asset, 0, true, true);
    let id = place(
        &mut w,
        &asset,
        Some(Crush {
            mass: CrushMass::Remove,
            ..Crush::default()
        }),
    )
    .unwrap();
    for _ in 0..180 {
        w.step().unwrap();
        for f in w.poll().unwrap() {
            assert!(
                crushes(&f).is_empty(),
                "frame {}: a chunk crushed under the wall's own weight",
                f.step
            );
        }
    }
    for f in w.poll_blocking().unwrap() {
        assert!(crushes(&f).is_empty());
    }
    let state = w.destructible_crush_state(id).unwrap();
    assert!(
        state.iter().all(|s| s.0 == 0.0 && !s.3),
        "no damage under its own weight"
    );
    // The stress estimate of a bottom block: the wall's weight over the bottom section is
    // p = W / (3 A) (uniaxial: q = W / A; 12 x 2 blocks of 0.4 m). The estimate shares the
    // ground's reaction among all of a block's bonds by area (side bonds at mid height
    // too, a shorter lever), and the peak includes the settling: the same order.
    let mass: f64 = chunk_data(&asset).iter().map(|c| c.0 * c.1).sum();
    let weight = mass * 9.81;
    let area = 24.0 * 0.16;
    let expected = weight / (3.0 * area);
    let peak = state.iter().map(|s| s.1).fold(0.0f32, f32::max) as f64;
    eprintln!("bottom-row pressure: peak {peak:.0} Pa, W/(3A) {expected:.0} Pa");
    assert!(
        peak > 0.4 * expected && peak < 1.5 * expected,
        "a resting wall's bottom blocks are in compression: {peak} Pa against W/(3A) {expected} Pa"
    );
    // Far below the material's yield (cohesion 4.8 MPa).
    assert!(peak < 1e5, "{peak}");
}

#[test]
fn a_hard_projectile_crushes_the_struck_chunk_and_pays_for_it() {
    let json = crush_wall_json();
    let asset = asset_of(&json);
    let mut w = world_for(&asset, 256, true, true);
    let fragments = 4;
    let id = place(
        &mut w,
        &asset,
        Some(Crush {
            mass: CrushMass::Conserve {
                fragments,
                budget: 256,
            },
            ..Crush::default()
        }),
    )
    .unwrap();
    let ball = shoot(&mut w, 35.0, 4800.0);
    let data = chunk_data(&asset);
    let mut seen = Vec::new();
    let mut before = w.destructible_crush_state(id).unwrap();
    let mut ke_before = momentum(&w, &[ball]).1;
    for _ in 0..30 {
        let f = w.step_sync().unwrap();
        let now = w.destructible_crush_state(id).unwrap();
        let ke = momentum(&w, &[ball]).1;
        let events = crushes(&f);
        if !events.is_empty() && seen.is_empty() {
            // The frame of the strike, predicted before the ball touches: it paid each
            // struck chunk's damage times its crush energy and volume out of its kinetic
            // energy (its free flight in the frame changed that by m g vy dt only).
            let booked: f64 = (0..now.len())
                .map(|k| (now[k].0 - before[k].0) as f64 * data[k].2 * data[k].0)
                .sum();
            let paid = ke_before - ke;
            eprintln!(
                "strike: {} crushed, booked {booked:.0} J, the ball paid {paid:.0} J",
                events.len()
            );
            assert!(booked > 0.0);
            assert!(
                (paid - booked).abs() < 1e-3 * booked + 4800.0 * 9.81 * 1.0 / 60.0,
                "paid {paid} J, booked {booked} J"
            );
            assert!(
                events.iter().all(|e| matches!(
                    e,
                    DestructionEvent::ChunkCrushed {
                        cause: CrushCause::Impact,
                        ..
                    }
                )),
                "the strike's crush is predicted"
            );
        }
        for e in &events {
            let DestructionEvent::ChunkCrushed {
                destructible,
                chunk,
                mass,
                volume,
                pose,
                linear_velocity,
                peak_pressure,
                cause,
                fragments: frs,
                ..
            } = e
            else {
                unreachable!()
            };
            assert_eq!(*destructible, id);
            let (vol, rho, _) = data[*chunk as usize];
            assert!((*volume as f64 - vol).abs() < 1e-6, "{volume} vs {vol}");
            let m = vol * rho;
            assert!((*mass as f64 - m).abs() < 1e-3 * m, "{mass} vs {m}");
            assert!(*peak_pressure > 0.0, "{peak_pressure}");
            assert!(w.destructible_crush_state(id).unwrap()[*chunk as usize].3);
            assert!(
                !w.body_is_valid(w.destructible_chunk_body(id, *chunk).unwrap()),
                "the crushed chunk's body is gone"
            );
            // Conserve: its mass in `fragments` live bodies, at its velocity, inside its box.
            assert_eq!(frs.len(), fragments as usize);
            let total: f32 = frs
                .iter()
                .map(|b| w.body_mass_data(*b).unwrap().unwrap().mass)
                .sum();
            assert!(
                (total - mass).abs() < 1e-3 * mass,
                "fragments {total} kg of {mass} kg"
            );
            for b in frs {
                let s = w.read_body_state(*b).unwrap();
                if *cause == CrushCause::Impact {
                    // The wall is anchored: the struck chunk was at rest, and so are they.
                    assert!(Vec3::from(*linear_velocity).length() < 1e-3);
                    assert!(Vec3::from(s.linear_velocity).length() < 1e-3);
                }
                let d = Vec3::from(s.transform.position) - Vec3::from(pose.position);
                assert!(
                    d.length() < 0.35,
                    "a fragment {d:?} from its chunk's centre"
                );
            }
            seen.push(*chunk);
        }
        before = now;
        ke_before = ke;
    }
    assert!(
        !seen.is_empty(),
        "the 4.8 t ball at 35 m/s crushes the block it strikes"
    );
    // It slowed: the wall took energy out of it.
    let v = Vec3::from(w.read_body_state(ball).unwrap().linear_velocity);
    eprintln!("{} chunks crushed; the ball leaves at {v:?}", seen.len());
}

#[test]
fn conserve_keeps_mass_and_momentum_of_a_free_wall() {
    // No gravity, no ground: the free wall and the ball conserve momentum until they meet,
    // and the crush (the ball's impulse to the struck wall, the split, the fragments at the
    // chunk's velocity) keeps it.
    let asset = asset_of(&free_wall_json());
    let mut w = world_for(&asset, 256, false, false);
    let id = place(
        &mut w,
        &asset,
        Some(Crush {
            mass: CrushMass::Conserve {
                fragments: 8,
                budget: 256,
            },
            ..Crush::default()
        }),
    )
    .unwrap();
    let ball = shoot(&mut w, 35.0, 4800.0);
    w.step_sync().unwrap();
    let mut fragments: Vec<BodyId> = Vec::new();
    let mut prev = momentum(&w, &live_bodies(&w, id, &[ball])).0;
    let mut crushed = 0;
    let wall_mass: f32 = {
        let stats = w.destructible_stats(id).unwrap();
        let mut bodies: Vec<BodyId> = (0..stats.chunks)
            .map(|c| w.destructible_chunk_body(id, c).unwrap())
            .collect();
        bodies.dedup();
        bodies
            .iter()
            .filter_map(|b| w.body_mass_data(*b).unwrap())
            .map(|m| m.mass)
            .sum()
    };
    for _ in 0..12 {
        let f = w.step_sync().unwrap();
        let events = crushes(&f);
        for e in &events {
            if let DestructionEvent::ChunkCrushed {
                mass,
                linear_velocity,
                fragments: frs,
                ..
            } = e
            {
                // The fragments carry the chunk's momentum exactly.
                let p: Vec3 = frs
                    .iter()
                    .map(|b| {
                        Vec3::from(w.read_body_state(*b).unwrap().linear_velocity)
                            * w.body_mass_data(*b).unwrap().unwrap().mass
                    })
                    .sum();
                let expected = Vec3::from(*linear_velocity) * *mass;
                assert!(
                    (p - expected).length() <= 1e-3 * expected.length() + 1e-3,
                    "fragments {p:?}, chunk {expected:?}"
                );
                fragments.extend_from_slice(frs);
                crushed += 1;
            }
        }
        let bodies = live_bodies(&w, id, &[&fragments[..], &[ball]].concat());
        let now = momentum(&w, &bodies).0;
        if !events.is_empty() {
            eprintln!("crush frame: momentum {prev:?} -> {now:?}");
            assert!(
                (now - prev).length() <= 1e-3 * prev.length(),
                "momentum {prev:?} -> {now:?} over the crush"
            );
            // The wall's mass is all still there (pieces and fragments).
            let total: f32 = bodies
                .iter()
                .filter(|b| **b != ball)
                .filter_map(|b| w.body_mass_data(*b).unwrap())
                .map(|m| m.mass)
                .sum();
            assert!(
                (total - wall_mass).abs() < 1e-3 * wall_mass,
                "{total} kg of {wall_mass} kg"
            );
            break;
        }
        prev = now;
    }
    assert!(crushed > 0, "the strike crushes the struck block");
}

#[test]
fn remove_takes_the_chunk_out() {
    let asset = asset_of(&crush_wall_json());
    let mut w = world_for(&asset, 0, true, true);
    let id = place(
        &mut w,
        &asset,
        Some(Crush {
            mass: CrushMass::Remove,
            ..Crush::default()
        }),
    )
    .unwrap();
    shoot(&mut w, 35.0, 4800.0);
    let mut removed = Vec::new();
    for _ in 0..20 {
        let f = w.step_sync().unwrap();
        for e in crushes(&f) {
            if let DestructionEvent::ChunkCrushed {
                chunk,
                fragments,
                mass,
                ..
            } = e
            {
                assert!(fragments.is_empty(), "Remove keeps no fragments");
                assert!(mass > 0.0);
                assert!(!w.body_is_valid(w.destructible_chunk_body(id, chunk).unwrap()));
                removed.push(chunk);
            }
        }
    }
    assert!(!removed.is_empty());
    // The chunk is no part of any live piece.
    for &c in &removed {
        let actor = w.destructible_actor_of(id, c).unwrap();
        assert!(actor.bodies.iter().all(|b| !w.body_is_valid(*b)));
    }
    assert_eq!(
        w.destructible_stats(id).unwrap().crushed,
        removed.len() as u32
    );
    // Removing the destructible retires its parked chunks' shapes too: the world steps on.
    w.destroy_destructible(id).unwrap();
    for _ in 0..3 {
        w.step_sync().unwrap();
    }
}

/// Every body's pose and velocity after `frames` of the wall struck by `speed`, as bits.
fn digest(crush: Option<Crush>, frames: u32) -> Vec<u32> {
    let asset = asset_of(&crush_wall_json());
    let mut w = world_for(&asset, 0, true, true);
    let id = place(&mut w, &asset, crush).unwrap();
    let ball = shoot(&mut w, 12.0, 400.0);
    for _ in 0..frames {
        w.step_sync().unwrap();
    }
    let mut out = Vec::new();
    for b in live_bodies(&w, id, &[ball]) {
        let s = w.read_body_state(b).unwrap();
        out.extend(
            s.transform
                .position
                .iter()
                .chain(&s.transform.rotation)
                .chain(&s.linear_velocity)
                .chain(&s.angular_velocity)
                .map(|x| x.to_bits()),
        );
    }
    out
}

#[test]
fn crushing_that_never_crushes_changes_nothing() {
    // A ball that breaks bonds but crushes nothing (an unyielding material): every body's
    // state bit for bit as with crushing off.
    let strong = CrushMaterial {
        cap_pressure: 1e15,
        cohesion: 1e15,
        friction_slope: 0.0,
        crush_energy: 1e9,
        crush_viscosity: 1e9,
    };
    let crush = Crush {
        mass: CrushMass::Remove,
        materials: vec![("concrete".into(), strong)],
        max_per_step: 0,
    };
    let off = digest(None, 40);
    let on = digest(Some(crush), 40);
    assert_eq!(off.len(), on.len());
    assert!(
        off == on,
        "crushing that crushes nothing changed the simulation"
    );
}

#[test]
fn crush_needs_parameters_and_the_threshold_model() {
    // The cuboid house's materials have no crush parameters.
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
    let mut w = world_for(&asset, 256, true, true);
    let e = place(&mut w, &asset, Some(Crush::default()))
        .unwrap_err()
        .to_string();
    assert!(e.contains("no crush parameters"), "{e}");
    // An unknown material name is an error too, and so is a bad parameter.
    let e = place(
        &mut w,
        &asset,
        Some(Crush {
            materials: vec![(
                "unobtainium".into(),
                CrushMaterial {
                    cap_pressure: 1.0,
                    cohesion: 0.0,
                    friction_slope: 0.0,
                    crush_energy: 1.0,
                    crush_viscosity: 1.0,
                },
            )],
            ..Crush::default()
        }),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("unobtainium"), "{e}");
    let a = w.create_destructible_asset(&asset).unwrap();
    let e = w
        .create_destructible(&DestructibleDef {
            crush: Some(Crush::default()),
            ..DestructibleDef::new(a)
        })
        .unwrap_err()
        .to_string();
    assert!(e.contains("StrengthModel::Threshold"), "{e}");
}
