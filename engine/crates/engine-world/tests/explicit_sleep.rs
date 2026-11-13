//! Vehicle sleep in the World (`engine_softbody::explicit_sleep`, on by default in
//! `ExplicitVehiclesDef`): a parked sedan falls asleep, the frame reports it, a sleeping
//! vehicle does no work (its state stands, and with nothing else awake the World skips the
//! step), and a box dropped onto it wakes it through the coupling plan (a moving rigid body
//! among its proxy candidates) before the box reaches it; at waking the sedan is within the
//! frozen jitter of a twin world's sedan that never slept, and the strike that follows
//! differs from the twin's no more than a twin's whose sedan was moved by that much.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use rubble::explicit_scene::place_vehicle;
use rubble::*;

fn sedan() -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    ExplicitBeams::from_truck(
        &text,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )
    .unwrap()
}

/// A world with a ground plane and a parked sedan, vehicle sleep as `sleep`, the sedan
/// moved by `nudge` (m) along x at spawn.
fn parked(sleep: bool, nudge: f32) -> anyhow::Result<Option<(World, f32, Vec3)>> {
    let mut solver = AvbdParams::default();
    solver.dt = 1.0 / 120.0;
    let Ok(mut w) = World::new(
        GpuContext::Default,
        &WorldDef {
            solver,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        },
    ) else {
        return Ok(None);
    };
    let g = w.create_body(&BodyDef::default())?;
    w.create_shape(
        g,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
        },
    )?;
    let mut car = place_vehicle(&sedan(), Vec3::ZERO, 0, 0.0, 0.0);
    for n in &mut car.nodes {
        n.pos.x += nudge;
    }
    let top = car.nodes.iter().map(|n| n.pos.y).fold(f32::MIN, f32::max);
    let mid = car.nodes.iter().map(|n| n.pos).sum::<Vec3>() / car.nodes.len() as f32;
    let def = ExplicitVehiclesDef {
        sleep: sleep.then(Default::default),
        ..ExplicitVehiclesDef::default()
    };
    w.add_explicit_vehicles(&[car], &def)?;
    Ok(Some((w, top, mid)))
}

fn drop_box(w: &mut World, at: [f32; 3]) -> anyhow::Result<BodyId> {
    let b = w.create_body(&BodyDef::dynamic(at))?;
    w.create_box_shape(
        b,
        &ShapeDef {
            density: 50.0 / 0.125,
            ..ShapeDef::default()
        },
        [0.25; 3],
    )?;
    Ok(b)
}

/// Step every world once; the first one's frames.
fn step_all(worlds: &mut [&mut World]) -> anyhow::Result<Vec<Frame>> {
    for w in worlds.iter_mut() {
        w.step()?;
    }
    let mut first = Vec::new();
    for (k, w) in worlds.iter_mut().enumerate() {
        let f = w.poll_blocking()?;
        if k == 0 {
            first = f;
        }
    }
    Ok(first)
}

fn sedan_gap(a: &World, b: &World) -> f32 {
    let (p, q) = (
        a.explicit_vehicles().unwrap().positions(),
        b.explicit_vehicles().unwrap().positions(),
    );
    p[0].iter()
        .zip(&q[0])
        .map(|(x, y)| (*x - *y).abs().max_element())
        .fold(0.0, f32::max)
}

fn box_gap(a: &World, ab: BodyId, b: &World, bb: BodyId) -> anyhow::Result<f32> {
    let (p, q) = (
        a.read_body_state(ab)?.transform.position,
        b.read_body_state(bb)?.transform.position,
    );
    Ok((Vec3::from(p) - Vec3::from(q)).abs().max_element())
}

#[test]
fn a_parked_vehicle_sleeps_and_a_falling_box_wakes_it() -> anyhow::Result<()> {
    let (Some((mut w, top, mid)), Some((mut twin, _, _))) =
        (parked(true, 0.0)?, parked(false, 0.0)?)
    else {
        eprintln!("no GPU: skipped");
        return Ok(());
    };
    // A third world, never asleep, its sedan moved 0.02 mm: how far the strike below
    // carries a difference that small.
    let (mut nudged, _, _) = parked(false, 2.0e-5)?.unwrap();
    // Asleep within 6 s, reported by a frame (the twins step alongside, never asleep).
    let mut slept = None;
    for step in 0..720 {
        let frames = step_all(&mut [&mut w, &mut twin, &mut nudged])?;
        if frames.iter().any(|f| {
            f.explicit_sleep.contains(&ExplicitSleepEvent {
                vehicle: 0,
                asleep: true,
            })
        }) {
            slept = Some(step);
            break;
        }
    }
    let slept = slept.expect("the parked sedan never fell asleep in 6 s");
    eprintln!("asleep at step {slept} ({:.2} s)", slept as f32 / 120.0);
    assert!(w.explicit_sleep_states()[0].1);
    assert!(twin.explicit_sleep_states().is_empty());
    // Asleep and nothing else awake: the World skips its steps; the vehicle's state stands.
    let before = w.explicit_vehicles().unwrap().state_words();
    let mut skipped = 0;
    for _ in 0..60 {
        skipped += step_all(&mut [&mut w, &mut twin, &mut nudged])?
            .iter()
            .filter(|f| !f.ran)
            .count();
    }
    eprintln!("{skipped} of 60 steps skipped while asleep");
    assert!(skipped >= 55, "only {skipped} of 60 steps skipped");
    for ((name, a), (_, b)) in before
        .iter()
        .zip(w.explicit_vehicles().unwrap().state_words())
    {
        assert!(*a == b, "a sleeping vehicle's {name} changed");
    }
    // A 50 kg box dropped from 3 m above the roof, in every world: it wakes the sedan
    // before it reaches it.
    let drop = [mid.x, top + 3.0, mid.z];
    let b = drop_box(&mut w, drop)?;
    let tb = drop_box(&mut twin, drop)?;
    let nb = drop_box(&mut nudged, drop)?;
    let mut woke = None;
    let mut box_y = drop[1];
    for step in 0..240 {
        let frames = step_all(&mut [&mut w, &mut twin, &mut nudged])?;
        if frames.iter().any(|f| {
            f.explicit_sleep.contains(&ExplicitSleepEvent {
                vehicle: 0,
                asleep: false,
            })
        }) {
            woke = Some(step);
            box_y = w.read_body_state(b)?.transform.position[1];
            break;
        }
    }
    let woke = woke.expect("the falling box never woke the sedan");
    eprintln!(
        "woke {woke} steps after the drop, the box {:.2} m above the roof's top node",
        box_y - top
    );
    assert!(box_y > top, "woke after the box reached the roof");
    let at_wake = sedan_gap(&w, &twin);
    eprintln!(
        "woken sedan against the twin that never slept, at waking: {:.3} mm",
        at_wake * 1e3
    );
    assert!(at_wake < 1e-3, "{at_wake} m");
    // The strike (1 s): the box between the roof's nodes amplifies any difference; the
    // woken world's departs from the twin's no more than the nudged twin's does.
    let (mut worst, mut nudged_worst) = (0.0f32, 0.0f32);
    for _ in 0..120 {
        step_all(&mut [&mut w, &mut twin, &mut nudged])?;
        worst = worst.max(box_gap(&w, b, &twin, tb)?);
        nudged_worst = nudged_worst.max(box_gap(&nudged, nb, &twin, tb)?);
    }
    eprintln!(
        "over the strike: the box within {:.2} mm of the twin's (the nudged twin's within {:.2} mm); the sedan {:.2} mm (nudged {:.2} mm)",
        worst * 1e3,
        nudged_worst * 1e3,
        sedan_gap(&w, &twin) * 1e3,
        sedan_gap(&nudged, &twin) * 1e3
    );
    assert!(
        worst <= 2.0 * nudged_worst + 5e-3,
        "the box's path differs by {worst} m (a 0.02 mm nudge: {nudged_worst} m)"
    );
    Ok(())
}
