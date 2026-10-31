//! Strike prediction in the threshold mode (forward only, AGENTS.md): a shot that will reach
//! a destructible within the next frame breaks the bonds its strike breaks at the end of the
//! frame before it, so the engine's step meets the pieces. Measured on the punch-through wall
//! (`crush-wall.json`, crushing off) by the shot's own velocity: its first contact is the
//! first frame it slows; the predicted breaks are reported in an earlier frame.
#![cfg(feature = "stress")]

use rubble::*;

/// The punch-through wall: 12 x 8 x 2 blocks of 0.4 m on a footing of zero-mass anchors.
fn crush_wall() -> DestructibleAssetDesc {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../engine-webgpu/fracture/packs/crush-wall.json"
    );
    DestructibleAssetDesc::from_pack(&std::fs::read(path).unwrap()).unwrap()
}

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

/// What a shot did, per frame (`step_sync`'s frame index).
#[derive(Debug)]
struct Shot {
    /// The first frame reporting broken bonds.
    first_break: Option<usize>,
    /// The first frame after which the shot is slower than it was launched (its contact).
    first_contact: Option<usize>,
    /// Bonds broken in all.
    broken: usize,
}

/// The pack's projectile (0.45 m, 4.8 t, 35 m/s) along -x at the wall's middle block of its
/// fourth row, its first contact `gap_frames` frames' travel away. No gravity: the shot's
/// speed is exactly its launch speed until it touches.
fn shoot(gap_frames: f32, frames: usize) -> Shot {
    shoot_with(Aim::Gap(gap_frames), false, frames)
}

enum Aim {
    /// Along -x at the middle block of the fourth row, first contact this many frames away.
    Gap(f32),
    /// From here at this velocity (world frame).
    From([f32; 3], [f32; 3]),
}

fn shoot_with(aim: Aim, gravity: bool, frames: usize) -> Shot {
    let asset = crush_wall();
    let base = asset.solver_for(StrengthModel::Threshold).unwrap();
    let g = if gravity { base.gravity_y } else { 0.0 };
    let solver = engine_schema::AvbdParams {
        gravity_y: g,
        ..base
    };
    let dt = solver.dt;
    let stress = asset.stress.as_ref().unwrap();
    let offset = stress.placement().map(|x| -x as f32);
    let def = WorldDef {
        capacity: capacity(&asset),
        gravity: [0.0, g, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    w.create_destructible(&DestructibleDef {
        model: StrengthModel::Threshold,
        transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
        asleep: false,
        ..DestructibleDef::new(a)
    })
    .unwrap();
    // The wall's front face (+x) and the struck block: the one nearest (x, 1.8, 0.2) in the
    // world (0.4 m blocks, half 0.2).
    let world = |p: [f32; 3]| [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]];
    let face = asset
        .chunks
        .iter()
        .map(|c| world(c.pose.position)[0] + 0.2)
        .fold(f32::MIN, f32::max);
    let target = asset
        .chunks
        .iter()
        .map(|c| world(c.pose.position))
        .min_by(|a, b| {
            let d = |p: &[f32; 3]| (p[1] - 1.8).powi(2) + (p[2] - 0.2).powi(2) - p[0];
            d(a).total_cmp(&d(b))
        })
        .unwrap();
    let (r, m, speed) = (0.45f32, 4800.0f32, 35.0f32);
    let (at, v0) = match aim {
        Aim::Gap(gap) => (
            [face + r + gap * speed * dt, target[1], target[2] + 0.1],
            [-speed, 0.0, 0.0],
        ),
        Aim::From(at, v) => (at, v),
    };
    let speed = (v0[0] * v0[0] + v0[2] * v0[2]).sqrt();
    let ball = w
        .create_body(&BodyDef {
            linear_velocity: v0,
            ..BodyDef::dynamic(at)
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
    let i = 0.4 * m * r * r;
    w.body_set_mass_data(
        ball,
        MassData {
            mass: m,
            center: [0.0; 3],
            rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
        },
    )
    .unwrap();
    w.body_set_projectile(ball, Some(Projectile::default()))
        .unwrap();
    let mut shot = Shot {
        first_break: None,
        first_contact: None,
        broken: 0,
    };
    for n in 0..frames {
        let f = w.step_sync().unwrap();
        for e in &f.destruction_events {
            if let DestructionEvent::BondsBroken { bonds, .. } = e {
                shot.broken += bonds.len();
                shot.first_break.get_or_insert(n);
            }
        }
        let v = w.read_body_state(ball).unwrap().linear_velocity;
        // Gravity changes only its vertical speed.
        if ((v[0] - v0[0]).powi(2) + (v[2] - v0[2]).powi(2)).sqrt() > 1e-3 * speed {
            shot.first_contact.get_or_insert(n);
        }
    }
    eprintln!("{shot:?}");
    shot
}

fn assert_predicted(gap_frames: f32) {
    let shot = shoot(gap_frames, (gap_frames as usize) + 8);
    let contact = shot.first_contact.expect("the shot reaches the wall");
    let broken = shot
        .first_break
        .expect("the 4.8 t shot at 35 m/s breaks bonds of the wall");
    assert!(
        broken < contact,
        "the strike's breaks are predicted the frame before its contact: \
         breaks in frame {broken}, contact in frame {contact} ({shot:?})"
    );
}

#[test]
fn a_strike_on_the_crush_wall_breaks_bonds_the_frame_before_contact() {
    assert_predicted(3.4);
}

#[test]
fn a_shot_exactly_a_whole_number_of_frames_away_is_predicted() {
    // The first prediction runs after the first step: from two frames' travel, the strike
    // is exactly one frame away when it is checked; from three, two (out of reach) and then
    // exactly one.
    assert_predicted(2.0);
    assert_predicted(3.0);
}

#[test]
fn the_packs_shots_on_the_crush_wall_are_predicted() {
    // The pack's shots (`crush-wall.json` `shots`), as the reactions harness fires them:
    // from (-12, 3.12, -2.25), gravity on.
    for scale in [1.0f32, 1.4] {
        let v = [35.85099 * scale, -4.48137 * scale, 6.72206 * scale];
        let shot = shoot_with(Aim::From([-12.0, 3.12, -2.25], v), true, 40);
        let contact = shot.first_contact.expect("the shot reaches the wall");
        let broken = shot.first_break.expect("the shot breaks bonds of the wall");
        assert!(
            broken < contact,
            "{scale}x: breaks in frame {broken}, contact in frame {contact}"
        );
    }
}

#[test]
fn every_aim_on_the_crush_wall_is_predicted() {
    // Shots at points across the wall's front, off the blocks' centres by uneven amounts, and
    // at angles: wherever the strike lands, the struck chunk is found and the strike
    // predicted. (Picked by matching a recomputed distance bit for bit, a strike whose two
    // computations rounded apart found no chunk and was dropped: the contact broke the
    // bonds a frame late.)
    let mut missed = Vec::new();
    for (i, &(y, z)) in [
        (0.9f32, -1.37f32),
        (1.3, 0.61),
        (1.8, 0.2),
        (2.23, -0.55),
        (2.7, 1.49),
        (3.05, -2.03),
    ]
    .iter()
    .enumerate()
    {
        for (j, &(vy, vz)) in [(0.0f32, 0.0f32), (-4.5, 6.7), (3.1, -2.9)]
            .iter()
            .enumerate()
        {
            // 1.5 frames' travel from the front (x = 0.4) along the path: the first
            // prediction (after the first step) finds it half a frame away, and breaks its
            // bonds in frame 0, before the shot first slows.
            let v = [-35.0f32, vy, vz];
            let back = 1.5 / 60.0 + 0.45 / 35.0;
            let at = [0.4 - v[0] * back, y - vy * back, z - vz * back];
            let shot = shoot_with(Aim::From(at, v), false, 6);
            match (shot.first_break, shot.first_contact) {
                (Some(0), Some(c)) if c > 0 => {}
                other => missed.push((i, j, other)),
            }
        }
    }
    assert!(
        missed.is_empty(),
        "strikes not predicted (aim, angle, (break, contact)): {missed:?}"
    );
}
