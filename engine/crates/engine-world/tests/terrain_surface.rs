//! The live terrain surface through the World API (`TerrainDef::surface`,
//! `engine/docs/deformable-terrain.md`, "Live surface in World"): each frame
//! carries every open area's ground rebuilt from its sand and the debris in
//! flight; a patch per area slot keeps that ground under bodies; the last
//! surface is what the area bakes to, the bake comes as a frame event with
//! the host's copy of the field already refreshed, and the patches sink.
//! Checked against the surface's definition (the bake's splat of the same
//! quanta) as `engine-webgpu`'s `terrain_surface` checks the bare pass.

use rubble::*;

const G: f32 = 9.81;
/// Samples of each patch along x and z.
const PN: u32 = 64;

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

fn terrain_def(max_debris: u32) -> TerrainDef {
    TerrainDef {
        config: TerrainConfig {
            max_particles: 1 << 20,
            max_blocks: 1 << 14,
            max_parcels: 1 << 18,
            max_substeps: 1 << 13,
            cells_per_radius: 8.0,
            min_cell_size: 0.002,
            max_regions: 4,
            ..TerrainConfig::default()
        },
        region_margin: 0.6,
        settle_speed: 0.05,
        settle_frames: 3,
        surface: Some(TerrainSurfaceDef {
            patch_samples: PN,
            max_debris,
            ..TerrainSurfaceDef::default()
        }),
        ..TerrainDef::default()
    }
}

/// The crater validation's 5 mm aluminium sphere at 2 km/s into dry sand
/// (transient radius 0.147 m), on a flat field with samples an eighth of it
/// apart.
fn impact(def: &TerrainDef) -> ImpactSource {
    ImpactSource {
        impactor_radius: 0.005,
        impactor_density: 2700.0,
        velocity: glam::DVec3::new(0.0, -2000.0, 0.0),
        target: CraterTarget::sand(&def.material).unwrap(),
    }
}

const N: u32 = 97;

fn flat_ground(w: &mut World) -> ShapeId {
    let spacing = 0.147 / 8.0;
    let g = w.create_body(&BodyDef::default()).unwrap();
    let field = HeightField::new(N, N, spacing, spacing, vec![0.0; (N * N) as usize]).unwrap();
    w.create_height_field_shape(g, &ShapeDef::default(), &field, Transform::IDENTITY)
        .unwrap()
}

fn mirror(w: &World, ground: ShapeId) -> Vec<f32> {
    let Geometry::HeightField { field, .. } = w.shape_geometry(ground).unwrap() else {
        panic!("not a heightfield");
    };
    field.heights().to_vec()
}

/// The largest |live − field| over window `w`.
fn window_error(w: [u32; 4], live: &[f32], field: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for tj in 0..w[3] {
        for ti in 0..w[2] {
            let f = field[((w[1] + tj) * N + w[0] + ti) as usize];
            worst = worst.max((live[(tj * w[2] + ti) as usize] - f).abs());
        }
    }
    worst
}

/// Patch `p` (its sample (a, b) is the field's (i0 − 1 + a, j0 − 1 + b))
/// against window `w`'s live heights inside it and `field` elsewhere.
fn patch_error(p: &[f32], w: [u32; 4], live: &[f32], field: &[f32]) -> f32 {
    let inside = |i: u32, j: u32| i >= w[0] && j >= w[1] && i < w[0] + w[2] && j < w[1] + w[3];
    let mut worst = 0.0f32;
    for b in 0..PN {
        for a in 0..PN {
            let i = (w[0] as i32 - 1 + a as i32).clamp(0, N as i32 - 1) as u32;
            let j = (w[1] as i32 - 1 + b as i32).clamp(0, N as i32 - 1) as u32;
            let want = if inside(i, j) {
                live[((j - w[1]) * w[2] + i - w[0]) as usize]
            } else {
                field[(j * N + i) as usize]
            };
            worst = worst.max((p[(b * PN + a) as usize] - want).abs());
        }
    }
    worst
}

#[test]
fn frames_carry_the_live_surface_until_it_bakes_and_the_patch_follows_it() {
    let mut w = world();
    let def = terrain_def(1 << 16);
    w.enable_terrain(&def).unwrap();
    let ground = flat_ground(&mut w);
    let flat = mirror(&w, ground);
    w.terrain_impact(ground, &impact(&def), [0.0, 0.0, 0.0])
        .unwrap();
    let dx = w.terrain_region().unwrap().1.dx;
    let patches = w.terrain_surface_patches();
    assert_eq!(patches.len(), 4, "one patch per max_regions slot");

    // Early on: the surface is shown, and the excavation flow is debris.
    let mut most_debris = 0;
    let mut last: Option<TerrainSurfaceView> = None;
    let mut last_debris = Vec::new();
    let mut bakes = Vec::new();
    let mut checked_patch = false;
    for k in 0..400 {
        w.step().unwrap();
        let frames = if k == 2 {
            w.poll_blocking().unwrap()
        } else {
            w.poll().unwrap()
        };
        for f in frames {
            bakes.extend(f.terrain_bakes.iter().copied());
            if let Some(v) = f.terrain_surface.first() {
                assert_eq!(v.shape, ground);
                assert_eq!(v.heights.len(), (v.window[2] * v.window[3]) as usize);
                last = Some(v.clone());
                most_debris = most_debris.max(f.terrain_debris.len());
                last_debris = f.terrain_debris.clone();
            }
        }
        if k == 2 {
            assert!(!w.terrain_areas().is_empty(), "still open");
            // The newest frame's surface is the one the patch holds now.
            let v = last.clone().expect("a surface by frame 2");
            let p = w.read_height_field(patches[0]).unwrap();
            let e = patch_error(&p, v.window, &v.heights, &flat);
            eprintln!("patch 0 vs the live surface at frame 2: {e:.2e} m");
            assert_eq!(e, 0.0);
            checked_patch = true;
        }
        if !bakes.is_empty() {
            break;
        }
    }
    assert!(checked_patch);
    assert_eq!(bakes.len(), 1, "one area baked");
    assert_eq!(bakes[0].shape, ground);
    let v = last.expect("frames carried the surface");
    // The bake's frame: the host's copy of the field already holds the bake.
    let baked = mirror(&w, ground);
    assert_eq!(
        baked,
        w.read_height_field(ground).unwrap(),
        "the copy follows the bake"
    );
    let e = window_error(v.window, &v.heights, &baked);
    let deepest = v.heights.iter().copied().fold(0.0f32, f32::min);
    eprintln!(
        "last live surface vs the bake {e:.2e} m (layer {dx:.4} m); crater {deepest:.4} m deep; \
         debris at most {most_debris}, {} in the last surface frame",
        last_debris.len()
    );
    assert!(deepest < -0.03, "a crater in the live surface");
    // The same splat of the same quanta, less the residual motion when the
    // last frame's surface was built: under a tenth of a layer.
    assert!(e < 0.1 * dx, "{e}");
    assert!(most_debris > 0, "the excavation flow listed as debris");
    assert!(last_debris.is_empty(), "nothing flies once it has settled");
    // Nothing open: the patches sink, and frames carry no surface.
    w.step().unwrap();
    for f in w.poll_blocking().unwrap() {
        assert!(f.terrain_surface.is_empty() && f.terrain_debris.is_empty());
    }
    for p in patches {
        assert!(
            w.read_height_field(p).unwrap().iter().all(|&h| h == -1.0e4),
            "patch sunk"
        );
        assert!(
            w.set_height_field_region(p, 0, 0, 1, 1, &[0.0]).is_err(),
            "patches are the GPU's"
        );
    }
}

#[test]
fn a_frame_with_more_debris_than_the_limit_fails_by_name() {
    let mut w = world();
    let def = terrain_def(8);
    w.enable_terrain(&def).unwrap();
    let ground = flat_ground(&mut w);
    w.terrain_impact(ground, &impact(&def), [0.0, 0.0, 0.0])
        .unwrap();
    let mut err = None;
    for _ in 0..10 {
        w.step().unwrap();
        if let Err(e) = w.poll_blocking() {
            err = Some(e.to_string());
            break;
        }
    }
    let e = err.expect("an excavation flies more than 8 pieces");
    eprintln!("{e}");
    assert!(e.contains("surface debris"), "{e}");
}

/// A car drives through an open area on its patch, through `World` alone
/// (as `engine-webgpu`'s `terrain_wheels` drives one on a bare `WgpuSim`):
/// its wheels ride the live sand, not the lowered bed; the sand carries
/// their load; it leaves a rut whose volume is all still there after the
/// bake.
#[test]
fn a_car_crosses_an_open_area_on_its_patch_and_leaves_a_rut() {
    const DX: f32 = 0.15;
    const NX: u32 = 55;
    const NZ: u32 = 121;
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 4096,
            max_colliders: 8192,
            max_contacts: 32_768,
            max_joints: 64,
            max_candidates: 65_536,
            max_colors: 32,
            max_commands: 1024,
            max_hull_vertices: 1024,
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    w.set_gravity([0.0, -G, 0.0]).unwrap();
    let tdef = TerrainDef {
        config: TerrainConfig {
            max_particles: 1 << 18,
            max_blocks: 1 << 12,
            max_wheel_loads: 64,
            max_regions: 1,
            ..TerrainConfig::default()
        }
        .with_profile(TerrainProfile::Game),
        // Held open for the test: it bakes when the car is through.
        settle_frames: 1 << 20,
        surface: Some(TerrainSurfaceDef {
            patch_samples: 48,
            max_debris: 0,
            ..TerrainSurfaceDef::default()
        }),
        ..TerrainDef::default()
    };
    w.enable_terrain(&tdef).unwrap();
    let g = w.create_body(&BodyDef::default()).unwrap();
    let field = HeightField::new(NX, NZ, DX, DX, vec![0.0; (NX * NZ) as usize]).unwrap();
    let ground = w
        .create_height_field_shape(g, &ShapeDef::default(), &field, Transform::IDENTITY)
        .unwrap();
    w.terrain_open(ground, [-2.0, -1.2, -3.0], [2.0, 0.6, 3.0], DX)
        .unwrap();
    let bed = -1.2;
    let car = w
        .add_vehicle(
            &engine_vehicle::presets::sedan(),
            [0.0, 0.8, -5.5],
            [0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
    let mut most_wheels = 0;
    let mut lowest_wheel = f32::INFINITY;
    let mut tracks: Vec<[f32; 2]> = Vec::new();
    let mut crossed = false;
    for k in 0..1200 {
        if k == 60 {
            w.set_vehicle_input(
                car,
                VehicleInput {
                    forward: 0.3,
                    ..VehicleInput::default()
                },
            );
        }
        w.step().unwrap();
        for f in w.poll().unwrap() {
            if let Some(r) = f.terrain {
                most_wheels = most_wheels.max(r.wheels);
            }
            for v in &f.vehicles {
                for wh in &v.state.wheels {
                    let [x, y, z] = wh.contact_point;
                    if wh.in_contact && z.abs() < 2.4 && x.abs() < 1.8 {
                        tracks.push([x, z]);
                        lowest_wheel = lowest_wheel.min(y);
                    }
                }
                crossed |= v.state.wheels.iter().all(|wh| wh.contact_point[2] > 3.6);
            }
        }
        if crossed {
            break;
        }
    }
    assert!(crossed, "the car crossed the area");
    eprintln!(
        "wheels on the sand at once: {most_wheels}; lowest contact {lowest_wheel:.4} m (bed {bed} m); \
         {} track samples",
        tracks.len()
    );
    assert_eq!(most_wheels, 4, "all four wheels pressed the sand");
    assert!(
        lowest_wheel > -0.3,
        "the wheels rode the sand, not the bed at {bed} m"
    );
    w.terrain_close().unwrap();
    w.step().unwrap();
    w.poll_blocking().unwrap();
    let after = mirror(&w, ground);
    let sample = |x: f32, z: f32| {
        let i = (x / DX + (NX - 1) as f32 / 2.0).round() as u32;
        let j = (z / DX + (NZ - 1) as f32 / 2.0).round() as u32;
        (j * NX + i) as usize
    };
    let rut: Vec<f32> = tracks.iter().map(|t| after[sample(t[0], t[1])]).collect();
    let mean_rut = rut.iter().sum::<f32>() / rut.len() as f32;
    let mut clear = Vec::new();
    for j in 0..NZ {
        for i in 0..NX {
            let (x, z) = (
                (i as f32 - (NX - 1) as f32 / 2.0) * DX,
                (j as f32 - (NZ - 1) as f32 / 2.0) * DX,
            );
            if x.abs() < 1.8 && z.abs() < 2.4 && tracks.iter().all(|t| (t[0] - x).abs() > 0.45) {
                clear.push(after[(j * NX + i) as usize]);
            }
        }
    }
    let mean_clear = clear.iter().sum::<f32>() / clear.len() as f32;
    let volume: f64 = after.iter().map(|&y| f64::from(y)).sum::<f64>() * f64::from(DX * DX);
    eprintln!("rut under the tracks {mean_rut:.4} m, clear of them {mean_clear:.4} m; volume change {volume:.2e} m³");
    assert!(
        mean_rut < mean_clear - 0.002,
        "a rut: {mean_rut} vs {mean_clear}"
    );
    assert!(
        volume.abs() < 1.0e-4 * (4.0 * 6.0 * 1.2),
        "the rut's volume stays: {volume}"
    );
}
