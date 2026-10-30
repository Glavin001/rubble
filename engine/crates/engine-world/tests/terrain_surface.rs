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
