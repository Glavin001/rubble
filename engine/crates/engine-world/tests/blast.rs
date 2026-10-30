//! Blast loading of threshold destructibles (`World::detonate`,
//! `DestructibleDef::blast_loading`): the stress oracle's face pressures break bonds in
//! `World` as its showcase does (a facing wall breaches, a wall in its shadow survives,
//! one far away survives), the chunk loads are the Kinney-Graham reflected pressures and
//! impulses on their faces, the pieces carry the blast's momentum, and a destructible
//! without the flag is unchanged.
#![cfg(feature = "stress")]

use rubble::*;
use stress_ref::blast::{incident_impulse, incident_overpressure, oblique_overpressure};
use stress_ref::builders::{auto_bonds, body, grid, grid_index, new_scene, oracle_concrete};
use stress_ref::math::Vec3 as SVec3;
use stress_ref::scene::{Scene, Support};

/// Walls of 0.1 m concrete cubes, 2 m wide and 2.1 m tall, standing on two fixed courses
/// (stress-ref `blast_two_walls`): one per `(name, y)`, `layers` thick, at stress-frame `y`.
fn walls(list: &[(&str, f64)], layers: usize, dt: f64) -> Scene {
    let mut s = new_scene("blast_walls", "walls for blast loading");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    for &(name, y0) in list {
        let counts = [20, layers, 21];
        let mut chunks = grid(
            SVec3::new(-1.0, y0, -0.1),
            counts,
            SVec3::splat(0.1),
            "concrete",
        );
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                chunks[grid_index(counts, i, j, 0)].support = Support::Fixed;
            }
        }
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        s.bodies.push(body(name, chunks, bonds));
    }
    s.sim.frame_dt = dt;
    s
}

/// The stress frame (Z up) in rubble's (Y up): (x, z, -y).
fn y_up(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[2] as f32, -p[1] as f32]
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

struct Placed {
    world: World,
    id: DestructibleId,
    /// Per chunk: its body at placement.
    initial: Vec<BodyId>,
    /// Chunks per wall, in scene order.
    per_wall: usize,
}

fn place(scene: &Scene, blast_loading: bool, strength_scale: f32) -> Placed {
    let asset = DestructibleAssetDesc::from_stress_scene(scene).unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let def = WorldDef {
        capacity: capacity(&asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut world = World::new(GpuContext::Default, &def).unwrap();
    let a = world.create_destructible_asset(&asset).unwrap();
    let id = world
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            asleep: false,
            blast_loading,
            strength_scale,
            ..DestructibleDef::new(a)
        })
        .unwrap();
    let chunks = world.destructible_stats(id).unwrap().chunks;
    let initial = (0..chunks)
        .map(|c| world.destructible_chunk_body(id, c).unwrap())
        .collect();
    Placed {
        world,
        id,
        initial,
        per_wall: scene.bodies[0].chunks.len(),
    }
}

impl Placed {
    /// Chunks of wall `w` no longer on the body they started on.
    fn detached(&self, w: usize) -> usize {
        (w * self.per_wall..(w + 1) * self.per_wall)
            .filter(|&c| {
                self.world
                    .destructible_chunk_body(self.id, c as u32)
                    .unwrap()
                    != self.initial[c]
            })
            .count()
    }

    /// Step `n` times; the frames' bond breaks and splits.
    fn run(&mut self, n: u32) -> (usize, usize) {
        let (mut bonds, mut splits) = (0, 0);
        for _ in 0..n {
            self.world.step().unwrap();
        }
        for f in self.world.poll_blocking().unwrap() {
            for e in &f.destruction_events {
                match e {
                    DestructionEvent::BondsBroken { bonds: b, .. } => bonds += b.len(),
                    DestructionEvent::Split { .. } => splits += 1,
                    _ => {}
                }
            }
        }
        (bonds, splits)
    }
}

const CHARGE: [f64; 3] = [0.0, -2.0, 1.0];

/// The oracle's showcase (`blast_breaches_front_wall_and_shadowed_wall_survives`) in
/// `World`: the wall facing the charge breaches; the wall behind it, in its shadow,
/// survives; alone, the same wall breaches (the shadowing is not vacuous). The threshold
/// model breaks a bond on its chunks' own loads (no stress solve spreads them), so it
/// takes 60 kg where the oracle's solve takes 30.
#[test]
fn blast_breaches_front_wall_and_shadowed_wall_survives() {
    let tnt = 60.0;
    let both = walls(&[("front", 0.0), ("back", 2.0)], 2, 1e-3);
    let mut p = place(&both, true, 1.0);
    p.world
        .detonate(&BlastDef {
            position: y_up(CHARGE),
            tnt_kg: tnt,
            ..BlastDef::default()
        })
        .unwrap();
    let (bonds, splits) = p.run(30);
    let (front, back) = (p.detached(0), p.detached(1));
    eprintln!("both: {bonds} bonds, {splits} splits; detached front {front}, back {back}");
    assert!(bonds > 0 && splits > 0, "no breaks reported");
    assert!(front > 0, "the facing wall did not breach");
    assert_eq!(back, 0, "the shadowed wall breached");

    let alone = walls(&[("back", 2.0)], 2, 1e-3);
    let mut p = place(&alone, true, 1.0);
    p.world
        .detonate(&BlastDef {
            position: y_up(CHARGE),
            tnt_kg: tnt,
            ..BlastDef::default()
        })
        .unwrap();
    p.run(30);
    let back = p.detached(0);
    eprintln!("alone: detached back {back}");
    assert!(
        back > 0,
        "the back wall alone did not breach: the shadowing test is vacuous"
    );
}

/// Pressure falls with distance: the charge that breaches a wall 2 m away loads the same
/// wall 25 m away (its chunks feel it) without breaking a bond.
#[test]
fn a_far_wall_survives_what_breaches_a_near_one() {
    let scene = walls(&[("wall", 0.0)], 1, 1.0 / 60.0);
    let near = [0.0, -2.0, 1.0];
    let far = [0.0, -25.0, 1.0];
    let mut outcome = Vec::new();
    for at in [near, far] {
        let mut p = place(&scene, true, 1.0);
        p.world
            .detonate(&BlastDef {
                position: y_up(at),
                tnt_kg: 10.0,
                ..BlastDef::default()
            })
            .unwrap();
        let mut loaded = 0;
        let mut peak: f32 = 0.0;
        let mut bonds = 0;
        for _ in 0..6 {
            bonds += p.run(1).0;
            let loads = p.world.destructible_blast_loads(p.id).unwrap();
            loaded = loaded.max(loads.len());
            peak = loads.iter().map(|l| l.peak_pressure).fold(peak, f32::max);
        }
        eprintln!("charge at {at:?}: {loaded} chunks loaded, peak {peak} Pa, {bonds} bonds broken, {} detached", p.detached(0));
        outcome.push((loaded, peak, bonds, p.detached(0)));
    }
    let (n, f) = (outcome[0], outcome[1]);
    assert!(n.2 > 0 && n.3 > 0, "the near charge broke nothing");
    assert!(f.0 > 0, "the far charge loaded nothing");
    assert!(
        f.1 < 0.1 * n.1,
        "the far peak {} is not far below the near {}",
        f.1,
        n.1
    );
    assert_eq!((f.2, f.3), (0, 0), "the far wall broke");
}

/// The World path's chunk loads are the oracle's face pressures: a wall too strong to
/// break, clearing off, its chunk facing the charge most squarely takes the Kinney-Graham
/// reflected (oblique) peak on its face's area, and over the pulse the impulse the fitted
/// Friedlander pulse carries (the incident impulse times the reflection factor) times that
/// area, pushed away from the charge; a chunk farther off takes less.
#[test]
fn chunk_loads_are_kinney_graham_face_pressures() {
    let mut scene = walls(&[("wall", 0.0)], 1, 1e-3);
    scene.sim.features.blast_clearing = false;
    let mut p = place(&scene, true, 1e6);
    let tnt = 5.0f64;
    let charge = [0.0, -3.0, 1.0];
    p.world
        .detonate(&BlastDef {
            position: y_up(charge),
            tnt_kg: tnt as f32,
            ..BlastDef::default()
        })
        .unwrap();
    let counts = [20, 1, 21];
    // The chunk at x = 0.05, z = 0.95 and its front face (stress frame y = 0).
    let (i, k) = (10, 10);
    let c = grid_index(counts, i, 0, k) as u32;
    let face = [
        -1.0 + (i as f64 + 0.5) * 0.1,
        0.0,
        -0.1 + (k as f64 + 0.5) * 0.1,
    ];
    let d = [
        face[0] - charge[0],
        face[1] - charge[1],
        face[2] - charge[2],
    ];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let cos = d[1] / r;
    let z = r / tnt.cbrt();
    let p_so = incident_overpressure(z);
    let peak = oblique_overpressure(p_so, cos);
    let area = 0.01;
    let want_force = peak * area;
    let want_impulse = incident_impulse(z, tnt) * peak / p_so * area;
    let corner = grid_index(counts, 0, 0, 20) as u32;
    let (mut f_max, mut impulse, mut corner_impulse) = (0.0f64, [0.0f64; 3], 0.0f64);
    let mut bonds = 0;
    for _ in 0..40 {
        bonds += p.run(1).0;
        for l in p.world.destructible_blast_loads(p.id).unwrap() {
            if l.chunk == c {
                f_max =
                    f_max.max((l.force.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt());
                for (sum, x) in impulse.iter_mut().zip(l.impulse) {
                    *sum += x as f64;
                }
            }
            if l.chunk == corner {
                corner_impulse += l
                    .impulse
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
            }
        }
    }
    assert_eq!(bonds, 0, "the strong wall broke");
    eprintln!("force {f_max} want {want_force}; impulse {impulse:?} want {want_impulse}; corner {corner_impulse}");
    assert!(
        (f_max - want_force).abs() < 1e-3 * want_force,
        "peak force {f_max} vs {want_force}"
    );
    let j = (impulse.iter().map(|x| x * x).sum::<f64>()).sqrt();
    assert!(
        (j - want_impulse).abs() < 1e-3 * want_impulse,
        "impulse {j} vs {want_impulse}"
    );
    // Pushed away from the charge: stress +y is rubble -z.
    assert!(impulse[2] < -0.99 * j, "{impulse:?}");
    assert!(
        corner_impulse > 0.0 && corner_impulse < 0.8 * j,
        "corner {corner_impulse} vs {j}"
    );
}

/// The pieces a blast knocks off carry its momentum: one layer thick, nothing behind
/// them, the momentum of the detached chunks a step after the blast is the impulse their
/// faces took (gravity's share over the step aside).
#[test]
fn knocked_off_pieces_carry_the_blast_impulse() {
    let scene = walls(&[("wall", 0.0)], 1, 1.0 / 60.0);
    let mut p = place(&scene, true, 1.0);
    p.world
        .detonate(&BlastDef {
            position: y_up([0.0, -1.5, 1.0]),
            tnt_kg: 2.0,
            ..BlastDef::default()
        })
        .unwrap();
    p.world.step().unwrap();
    p.world.poll_blocking().unwrap();
    let loads = p.world.destructible_blast_loads(p.id).unwrap();
    let moved: Vec<u32> = (0..p.per_wall as u32)
        .filter(|&c| p.world.destructible_chunk_body(p.id, c).unwrap() != p.initial[c as usize])
        .collect();
    assert!(!moved.is_empty(), "nothing broke");
    let j: f64 = loads
        .iter()
        .filter(|l| moved.contains(&l.chunk))
        .map(|l| l.impulse[2] as f64)
        .sum();
    // Momentum along rubble z of every piece body (each once).
    let mut bodies: Vec<BodyId> = moved
        .iter()
        .map(|&c| p.world.destructible_chunk_body(p.id, c).unwrap())
        .collect();
    bodies.sort_by_key(|b| b.to_bits());
    bodies.dedup();
    let mut momentum = 0.0;
    for b in &bodies {
        let m = p.world.body_mass_data(*b).unwrap().map_or(0.0, |m| m.mass) as f64;
        let v = p.world.read_body_state(*b).unwrap().linear_velocity;
        momentum += m * v[2] as f64;
    }
    eprintln!(
        "{} pieces of {} chunks: momentum {momentum}, impulse {j}",
        bodies.len(),
        moved.len()
    );
    assert!(j < 0.0, "the impulse does not push away from the charge");
    assert!(
        (momentum - j).abs() < 0.15 * j.abs(),
        "momentum {momentum} vs impulse {j}"
    );
}

/// Off by default: a destructible placed without `blast_loading` is unchanged by a blast
/// (no loads, no breaks, the same poses as a world without one).
#[test]
fn blast_loading_off_changes_nothing() {
    let scene = walls(&[("wall", 0.0)], 1, 1.0 / 60.0);
    let mut poses = Vec::new();
    for blast in [false, true] {
        let mut p = place(&scene, false, 1.0);
        if blast {
            p.world
                .detonate(&BlastDef {
                    position: y_up([0.0, -1.5, 1.0]),
                    tnt_kg: 2.0,
                    ..BlastDef::default()
                })
                .unwrap();
        }
        let (bonds, splits) = p.run(5);
        assert_eq!((bonds, splits), (0, 0));
        let stats = p.world.destructible_stats(p.id).unwrap();
        assert_eq!(stats.bonds_intact, stats.bonds);
        assert!(p.world.destructible_blast_loads(p.id).is_err());
        let all: Vec<_> = p.world.read_transforms().unwrap();
        poses.push(all);
    }
    assert_eq!(poses[0], poses[1]);
}

/// Out-of-range inputs are errors, never clamped: no TNT, a non-finite position, a charge
/// nearer a face than the Kinney-Graham fits reach, blast loading on the bonds model.
#[test]
fn out_of_range_blasts_are_errors() {
    let scene = walls(&[("wall", 0.0)], 1, 1.0 / 60.0);
    let mut p = place(&scene, true, 1.0);
    let ok = BlastDef {
        position: y_up([0.0, -1.5, 1.0]),
        tnt_kg: 1.0,
        ..BlastDef::default()
    };
    assert!(p.world.detonate(&BlastDef { tnt_kg: 0.0, ..ok }).is_err());
    assert!(p
        .world
        .detonate(&BlastDef {
            tnt_kg: f32::NAN,
            ..ok
        })
        .is_err());
    assert!(p
        .world
        .detonate(&BlastDef {
            position: [f32::INFINITY, 0.0, 0.0],
            ..ok
        })
        .is_err());
    // 1 cm from the face of the chunk at x = 0.05, z = 0.95: Z = 0.01 m/kg^(1/3).
    let err = p
        .world
        .detonate(&BlastDef {
            position: y_up([0.05, -0.01, 0.95]),
            ..ok
        })
        .unwrap_err();
    assert!(err.to_string().contains("Kinney-Graham"), "{err}");
    // The failed charge left nothing behind: the next is the only one.
    p.world.detonate(&ok).unwrap();
    p.run(3);

    let asset = DestructibleAssetDesc::from_stress_scene(&scene).unwrap();
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let err = w
        .create_destructible(&DestructibleDef {
            blast_loading: true,
            ..DestructibleDef::new(a)
        })
        .unwrap_err();
    assert!(err.to_string().contains("Threshold"), "{err}");
}
