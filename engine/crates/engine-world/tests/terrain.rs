//! Deformable terrain through the World API: an impact opens a region, the
//! pipelined steps run its MPM substeps with the rigid step, the settled region
//! bakes back into the heightfield on its own, and the host's copy of the field
//! follows the GPU's. A box dropped on an open region rests on the sand.

use rubble::*;

const G: f32 = 9.81;

fn world() -> World {
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 64,
            max_colliders: 128,
            max_contacts: 4096,
            max_joints: 16,
            max_candidates: 8192,
            max_colors: 32,
            max_commands: 256,
            max_hull_vertices: 64,
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    w.set_gravity([0.0, -G, 0.0]).unwrap();
    w
}

fn terrain_def() -> TerrainDef {
    TerrainDef {
        config: TerrainConfig {
            max_particles: 1 << 20,
            max_blocks: 1 << 14,
            max_parcels: 1 << 18,
            max_substeps: 1 << 13,
            cells_per_radius: 8.0,
            min_cell_size: 0.002,
            ..TerrainConfig::default()
        },
        region_margin: 0.6,
        settle_speed: 0.05,
        settle_frames: 3,
        ..TerrainDef::default()
    }
}

fn flat_ground(w: &mut World, n: u32, spacing: f32) -> (ShapeId, HeightField) {
    let g = w.create_body(&BodyDef::default()).unwrap();
    let field = HeightField::new(n, n, spacing, spacing, vec![0.0; (n * n) as usize]).unwrap();
    let s = w
        .create_height_field_shape(g, &ShapeDef::default(), &field, Transform::IDENTITY)
        .unwrap();
    (s, field)
}

fn mirror(w: &World, ground: ShapeId) -> Vec<f32> {
    let Geometry::HeightField { field, .. } = w.shape_geometry(ground).unwrap() else {
        panic!("not a heightfield");
    };
    field.heights().to_vec()
}

#[test]
fn an_impact_digs_a_crater_that_bakes_into_the_heightfield() {
    let mut w = world();
    let def = terrain_def();
    w.enable_terrain(&def).unwrap();
    let impact = ImpactSource {
        impactor_radius: 0.005,
        impactor_density: 2700.0,
        velocity: glam::DVec3::new(0.0, -2000.0, 0.0),
        target: CraterTarget::sand(&def.material).unwrap(),
    };
    // The crater's predicted radius is 0.147 m: samples at an eighth of it.
    let spacing = 0.147 / 8.0;
    let (ground, field) = flat_ground(&mut w, 97, spacing);
    let crater = w.terrain_impact(ground, &impact, [0.0, 0.0, 0.0]).unwrap();
    let (_, plan) = w.terrain_region().expect("an open region");
    let mut last = None;
    let mut frames = 0;
    while w.terrain_region().is_some() {
        assert!(frames < 240, "the region never settled");
        w.step().unwrap();
        for f in w.poll().unwrap() {
            if let Some(r) = f.terrain {
                last = Some(r);
            }
        }
        frames += 1;
    }
    // One more step copies the baked heights back; then wait for its frame.
    w.step().unwrap();
    w.poll_blocking().unwrap();
    let rep = last.expect("frames report the region");
    assert_eq!(rep.airborne, 0, "a region bakes with nothing in flight");

    let after = w.read_height_field(ground).unwrap();
    assert_eq!(
        mirror(&w, ground),
        after,
        "the host's field follows the bake"
    );
    // A crater was dug and baked: the centre below the surface. (Its size
    // against π-scaling is the crater validation's job, at finer cells: at 8
    // cells per radius it comes out smaller.)
    let (nx, _) = field.samples();
    let centre = after[(nx / 2 * nx + nx / 2) as usize];
    assert!(
        centre < -0.25 * crater.depth as f32,
        "centre at {centre} m; transient depth {} m",
        crater.depth
    );
    let quantum = f64::from(plan.dx / def.config.particles_per_cell_axis as f32).powi(3) / 64.0;
    let field_change: f64 =
        after.iter().map(|&h| f64::from(h)).sum::<f64>() * f64::from(spacing * spacing);
    let escaped = f64::from(rep.escaped) * quantum;
    assert!(
        (field_change + escaped).abs() < 1e-3 * escaped.max(1e-4) + 4.0 * quantum * 97.0 * 97.0,
        "field change {field_change} m³ against {escaped} m³ escaped"
    );
    eprintln!(
        "{frames} frames to settle; centre {centre:.4} m (transient depth {:.4} m), \
         {} parcels, {escaped:.2e} m³ escaped",
        crater.depth, rep.parcels
    );
}

#[test]
fn a_box_dropped_on_an_open_region_rests_on_the_sand() {
    let mut w = world();
    let def = terrain_def();
    w.enable_terrain(&def).unwrap();
    let (ground, _) = flat_ground(&mut w, 65, 0.02);
    w.terrain_open(ground, [-0.3, -0.2, -0.3], [0.3, 0.4, 0.3], 0.02)
        .unwrap();
    let half = 0.08;
    let b = w.create_body(&BodyDef::dynamic([0.0, 0.25, 0.0])).unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [half; 3])
        .unwrap();
    let mut ys = Vec::new();
    let trace = std::env::var("TERRAIN_TRACE").is_ok();
    for k in 0..90 {
        w.step().unwrap();
        for f in w.poll().unwrap() {
            if trace {
                eprintln!("step {} terrain {:?}", f.step, f.terrain);
            }
        }
        ys.push(
            w.body_transform(b)
                .unwrap()
                .map_or(f32::NAN, |t| t.position[1]),
        );
        if trace {
            eprintln!(
                "frame {k}: box y {:.4}, region {}",
                ys[k],
                w.terrain_region().is_some()
            );
        }
    }
    w.poll_blocking().unwrap();
    let y = w.body_transform(b).unwrap().unwrap().position[1];
    // On sand: the bottom within a few cells of the surface, not through it.
    assert!(
        (y - half).abs() < 0.05,
        "box bottom at {} m (positions {:?})",
        y - half,
        &ys[ys.len() - 10..]
    );
    let late = &ys[ys.len() - 20..];
    let drift = late.iter().cloned().fold(f32::MIN, f32::max)
        - late.iter().cloned().fold(f32::MAX, f32::min);
    assert!(drift < 2e-3, "still moving: {late:?}");
}

/// A heavy ball dropped on a watched field: the look-ahead opens a region
/// before it arrives (none missed), and the ball buries itself as deep as the
/// scaling of spheres dropped into dry sand says (Uehara et al. 2003:
/// d = (0.14/μ)·(ρ_b/ρ_g)^½·D^⅔·H^⅓, H the drop height plus d), within the
/// spread between their packings.
#[test]
fn a_dropped_ball_opens_a_region_before_it_lands_and_penetrates_as_uehara_says() {
    let mut w = world();
    let mut def = terrain_def();
    def.strike.cells_per_body = 6.0;
    def.strike.min_energy = 10.0;
    w.enable_terrain(&def).unwrap();
    let (ground, _) = flat_ground(&mut w, 65, 0.04);
    w.terrain_watch_shape(ground, true).unwrap();
    let (r, rho, drop) = (0.1f32, 7800.0f32, 1.0f32);
    let b = w
        .create_body(&BodyDef::dynamic([0.0, r + drop, 0.0]))
        .unwrap();
    w.create_sphere_shape(
        b,
        &ShapeDef {
            density: rho,
            ..ShapeDef::default()
        },
        [0.0; 3],
        r,
    )
    .unwrap();
    let mut opened_above = None;
    let mut deepest = 0.0f32;
    for _ in 0..150 {
        w.step().unwrap();
        w.poll().unwrap();
        let y = w
            .body_transform(b)
            .unwrap()
            .map_or(f32::NAN, |t| t.position[1]);
        if w.terrain_region().is_some() {
            opened_above.get_or_insert(y - r);
            deepest = deepest.max(r - y);
        } else if opened_above.is_some() {
            break;
        }
    }
    w.poll_blocking().unwrap();
    if std::env::var("TERRAIN_TRACE").is_ok() {
        let h = w.read_height_field(ground).unwrap();
        let row: Vec<String> = (24..41).map(|i| format!("{:.3}", h[32 * 65 + i])).collect();
        let yb = w.body_transform(b).unwrap().unwrap().position[1];
        eprintln!("baked row through the ball (x -0.32..0.32): {row:?}; ball centre {yb:.4}");
    }
    // After the bake: the ball on the baked ground (the heightfield cannot
    // hold it buried), not thrown off it.
    let mut after = Vec::new();
    for _ in 0..60 {
        w.step().unwrap();
        w.poll().unwrap();
        after.push(w.body_transform(b).unwrap().map_or(f32::NAN, |t| t.position[1]) - r);
    }
    eprintln!("ball bottom after the bake: {:?}", &after[..8]);
    eprintln!("... settles at {:.4} m", after[after.len() - 1]);
    assert_eq!(w.terrain_missed_strikes(), 0);
    let above = opened_above.expect("a region opened for the strike");
    assert!(
        above > 0.0,
        "the region opened with the ball {above} m above the sand"
    );
    // Uehara's law for this ball and sand.
    let mu = f64::from(def.material.friction_angle).tan();
    let ratio = f64::from(rho / def.material.density);
    let mut d = 0.0f64;
    for _ in 0..16 {
        d = 0.14 / mu
            * ratio.sqrt()
            * (2.0 * f64::from(r)).powf(2.0 / 3.0)
            * (f64::from(drop) + d).cbrt();
    }
    let got = f64::from(deepest);
    eprintln!("region opened {above:.3} m above; penetration {got:.3} m, Uehara {d:.3} m");
    assert!(
        (0.5 * d..=2.0 * d).contains(&got),
        "penetration {got} m against Uehara's {d} m"
    );
}
