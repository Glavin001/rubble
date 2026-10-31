//! The render-only live surface over several open areas at once
//! (`engine_terrain::surface`, `engine/docs/deformable-terrain.md`,
//! "Rendering and showcase"): two impacts far apart open two areas, and one
//! surface pass rebuilds both windows and publishes each into its own patch.
//! Checked from the surface's definition: right after the activation each
//! window's surface is the ground it was activated from; once the material
//! settles, each window's surface is the heightfield that area then bakes to
//! (the same splat of the same quanta); each patch holds its window's
//! surface and a ring of the field round it; the ground between the areas
//! stays the flat field; a patch whose area bakes sinks while the other's
//! stays.
//!
//! The event is the crater validation's 5 mm aluminium sphere at 2 km/s into
//! dry sand (transient radius 0.147 m) at 8 cells per radius, as in
//! `terrain_multi`.

use engine_schema::convex::{convex_params_base, HullArena};
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, Material, WorldCapacity};
use engine_terrain::plan::source_peak_speed;
use engine_terrain::surface::{
    SurfacePatch, SurfacePlan, DEFAULT_SURFACE_CELLS, SURFACE_BINDING, SURFACE_VIEW_OPEN,
    SURFACE_VIEW_WORD, SURFACE_VIEW_WORDS,
};
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    substeps_for, CraterPrediction, CraterTarget, ImpactSource, SandMaterial, TerrainConfig, ZModel,
};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Quat, Vec3};

const FRAME: f32 = 1.0 / 60.0;
const G: f32 = -9.81;
const NX: u32 = 161;
const NZ: u32 = 65;
/// Samples of each patch along x and z.
const PN: u32 = 64;

fn sim() -> WgpuSim {
    let mut s = WgpuSim::new(WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu adapter");
    let mut p = AvbdParams {
        dt: FRAME,
        iterations: 10,
        ..AvbdParams::default()
    };
    p.gravity_y = G;
    p.sleep_time = 1.0e9;
    s.set_params(p);
    s.set_material(
        0,
        Material {
            friction: 0.5,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    s
}

struct Scene {
    s: WgpuSim,
    cfg: TerrainConfig,
    sand: SandMaterial,
    field: HeightField,
    bed: u32,
    patches: Vec<u32>,
    surface: SurfacePlan,
    crater: CraterPrediction,
    model: ZModel,
    speed: f32,
    dx: f32,
    airborne: u32,
}

impl Scene {
    fn new() -> Self {
        let mut s = sim();
        let cfg = TerrainConfig {
            max_particles: 1 << 20,
            max_blocks: 1 << 14,
            max_parcels: 1 << 18,
            max_substeps: 1 << 12,
            cells_per_radius: 8.0,
            min_cell_size: 0.002,
            ..TerrainConfig::default()
        };
        s.terrain_enable(&cfg).unwrap();
        let sand = SandMaterial::default();
        s.terrain_set_material(Some(&sand)).unwrap();
        let src = ImpactSource {
            impactor_radius: 0.005,
            impactor_density: 2700.0,
            velocity: DVec3::new(0.0, -2000.0, 0.0),
            target: CraterTarget::sand(&sand).unwrap(),
        };
        let crater = src.crater(9.81).unwrap();
        let model = src
            .field(9.81, point_source_z(src.target.ejecta.mu))
            .unwrap();
        let spacing = crater.radius as f32 / 8.0;
        let field =
            HeightField::new(NX, NZ, spacing, spacing, vec![0.0; (NX * NZ) as usize]).unwrap();
        let mut arena = HullArena::new();
        let (_, bed) = s
            .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
            .unwrap();
        // Two patches, out of reach until their areas open.
        let sunk =
            HeightField::new(PN, PN, spacing, spacing, vec![-1.0e4; (PN * PN) as usize]).unwrap();
        let patches: Vec<u32> = (0..2)
            .map(|_| {
                s.add_height_field(&mut arena, &sunk, Vec3::ZERO, Quat::IDENTITY, 0)
                    .unwrap()
                    .1
            })
            .collect();
        let surface = SurfacePlan::new(&cfg, DEFAULT_SURFACE_CELLS)
            .unwrap()
            .with_patches(
                patches
                    .iter()
                    .map(|&collider| SurfacePatch {
                        collider,
                        nx: PN,
                        nz: PN,
                    })
                    .collect(),
            )
            .unwrap();
        s.terrain_surface_enable(&surface).unwrap();
        Self {
            s,
            cfg,
            sand,
            field,
            bed,
            patches,
            surface,
            crater,
            model,
            speed: 0.0,
            dx: 0.0,
            airborne: 0,
        }
    }

    /// The impact's area at `at`, opened (its source not yet applied).
    fn open(&mut self, at: Vec3) -> u32 {
        let dx = self
            .s
            .terrain_plan_ref()
            .unwrap()
            .event_cell(self.crater.asked_cell(&self.cfg))
            .unwrap();
        let plan = self
            .crater
            .region_with_cell(&self.cfg, at, 0.6, dx)
            .unwrap();
        let opened = self
            .s
            .terrain_open_area(
                &plan,
                self.bed,
                &self.field,
                Vec3::ZERO,
                self.sand.density,
                self.sand.friction_angle.tan(),
            )
            .unwrap();
        assert!(
            opened.activated && opened.absorbed.is_empty(),
            "far apart: an area of its own"
        );
        self.dx = dx;
        opened.area
    }

    /// The impact's source in area `area` at `at`.
    fn strike(&mut self, area: u32, at: Vec3) {
        self.s
            .terrain_apply_source_to(area, &self.model, at)
            .unwrap();
        self.speed = self.speed.max(source_peak_speed(&self.model));
    }

    fn frames(&mut self, n: u32) {
        for _ in 0..n {
            let (k, dt) = substeps_for(
                &self.cfg,
                FRAME,
                self.dx,
                self.sand.p_wave_speed(),
                1.5 * self.speed + 9.81 * FRAME + 20.0,
            )
            .unwrap();
            let r = self.s.terrain_step(k, dt, G).unwrap();
            self.speed = r.max_speed;
            self.airborne = r.airborne;
        }
    }

    fn land(&mut self, cap: u32) {
        for _ in 0..cap {
            if self.airborne == 0 {
                return;
            }
            self.frames(1);
        }
        panic!("parcels still in flight after {cap} frames");
    }

    fn records(&self, collider: u32, samples: u32) -> Vec<f32> {
        let base = convex_params_base(self.s.collider(collider).params);
        self.s
            .read_words(18, u64::from(base + 2) * 4, samples as usize)
            .into_iter()
            .map(f32::from_bits)
            .collect()
    }

    fn heights(&self) -> Vec<f32> {
        self.records(self.bed, NX * NZ)
    }

    fn patch(&self, k: usize) -> Vec<f32> {
        self.records(self.patches[k], PN * PN)
    }

    /// The surface pass now; then each slot's view: its window and heights.
    fn surface(&mut self) -> Vec<Option<([u32; 4], Vec<f32>)>> {
        let surface = self.surface.clone();
        self.s.terrain_surface(&surface).unwrap();
        let table = self.s.read_words(
            SURFACE_BINDING,
            u64::from(SURFACE_VIEW_WORD),
            (SURFACE_VIEW_WORDS * 2) as usize,
        );
        table
            .chunks(SURFACE_VIEW_WORDS as usize)
            .map(|v| {
                (v[0] & SURFACE_VIEW_OPEN != 0).then(|| {
                    let w = [v[6], v[7], v[8], v[9]];
                    let h = self
                        .s
                        .read_words(SURFACE_BINDING, u64::from(v[2]), (w[2] * w[3]) as usize)
                        .into_iter()
                        .map(f32::from_bits)
                        .collect();
                    (w, h)
                })
            })
            .collect()
    }
}

/// The largest |live − field| over window `w`.
fn window_error(w: [u32; 4], live: &[f32], field: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for tj in 0..w[3] {
        for ti in 0..w[2] {
            let f = field[((w[1] + tj) * NX + w[0] + ti) as usize];
            worst = worst.max((live[(tj * w[2] + ti) as usize] - f).abs());
        }
    }
    worst
}

fn inside(w: [u32; 4], i: u32, j: u32) -> bool {
    i >= w[0] && j >= w[1] && i < w[0] + w[2] && j < w[1] + w[3]
}

/// Patch `p` (its sample (a, b) is the field's (i0 − 1 + a, j0 − 1 + b))
/// against the live surface: window `w`'s heights inside it, `field`
/// elsewhere. The largest difference.
fn patch_error(p: &[f32], w: [u32; 4], live: &[f32], field: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for b in 0..PN {
        for a in 0..PN {
            let (i, j) = (w[0] as i32 - 1 + a as i32, w[1] as i32 - 1 + b as i32);
            let (i, j) = (
                i.clamp(0, NX as i32 - 1) as u32,
                j.clamp(0, NZ as i32 - 1) as u32,
            );
            let want = if inside(w, i, j) {
                live[((j - w[1]) * w[2] + i - w[0]) as usize]
            } else {
                field[(j * NX + i) as usize]
            };
            worst = worst.max((p[(b * PN + a) as usize] - want).abs());
        }
    }
    worst
}

#[test]
fn two_far_apart_areas_each_get_their_own_live_surface_and_patch() {
    let (p1, p2) = (Vec3::new(-0.9, 0.0, 0.0), Vec3::new(0.9, 0.0, 0.0));
    let mut sc = Scene::new();
    let flat = sc.heights();
    let a = sc.open(p1);
    let b = sc.open(p2);
    assert_eq!((a, b), (0, 1));

    // Right after the activation: each window's surface is the flat ground
    // it came from (to the activation's whole quanta and the fixed point).
    let views = sc.surface();
    let (wa, la) = views[0].clone().expect("area 0 shown");
    let (wb, lb) = views[1].clone().expect("area 1 shown");
    assert!(wa[0] + wa[2] < wb[0], "separate windows: {wa:?} {wb:?}");
    let e0 = window_error(wa, &la, &flat).max(window_error(wb, &lb, &flat));
    eprintln!(
        "activation: surface vs the flat ground {e0:.2e} m (layer {:.4} m)",
        sc.dx
    );
    assert!(e0 < 1.0e-3 * sc.crater.radius as f32, "{e0}");
    sc.strike(a, p1);
    sc.strike(b, p2);

    // Settled: each window's surface is what its area bakes to.
    sc.frames(40);
    sc.land(240);
    let views = sc.surface();
    let (wa2, la) = views[0].clone().unwrap();
    let (wb2, lb) = views[1].clone().unwrap();
    assert_eq!((wa2, wb2), (wa, wb), "the windows stay");
    let field = sc.heights();
    // The craters are there: each surface dug well below the flat ground.
    let deepest = |l: &[f32]| l.iter().copied().fold(0.0f32, f32::min);
    let r = sc.crater.radius as f32;
    eprintln!(
        "live crater depths {:.4} m and {:.4} m (radius {r:.3} m)",
        -deepest(&la),
        -deepest(&lb)
    );
    assert!(deepest(&la) < -0.2 * r && deepest(&lb) < -0.2 * r);
    // The two areas are mirror images: the same depth to a tenth.
    assert!((deepest(&la) - deepest(&lb)).abs() < 0.1 * -deepest(&la));
    // Each patch: its window's surface and the field round it.
    let (pa, pb) = (sc.patch(0), sc.patch(1));
    let (ea, eb) = (
        patch_error(&pa, wa, &la, &field),
        patch_error(&pb, wb, &lb, &field),
    );
    eprintln!("patches vs the live surface: {ea:.2e} m, {eb:.2e} m");
    assert!(ea == 0.0 && eb == 0.0, "{ea} {eb}");
    // Between the areas the ground is the flat field, in the heightfield and
    // in both patches' rings.
    let gap = (wa[0] + wa[2] + 1)..wb[0].saturating_sub(1);
    assert!(!gap.is_empty(), "a gap between {wa:?} and {wb:?}");
    for j in 0..NZ {
        for i in gap.clone() {
            let k = (j * NX + i) as usize;
            assert_eq!(field[k], flat[k], "field ({i}, {j}) between the areas");
        }
    }

    // Area 0 bakes: its surface is its baked ground; patch 0 sinks, patch 1
    // keeps area 1's surface.
    sc.s.terrain_bake_area(a).unwrap();
    let baked = sc.heights();
    let ea = window_error(wa, &la, &baked);
    eprintln!("area 0: live surface vs its bake {ea:.2e} m");
    let views = sc.surface();
    assert!(views[0].is_none(), "area 0 no longer shown");
    let (_, lb2) = views[1].clone().unwrap();
    assert!(sc.patch(0).iter().all(|&h| h == -1.0e4), "patch 0 sunk");
    assert_eq!(patch_error(&sc.patch(1), wb, &lb2, &baked), 0.0);
    // The other area's surface did not change with the first's bake.
    assert_eq!(lb2, lb);
    sc.s.terrain_bake_area(b).unwrap();
    let baked = sc.heights();
    let eb = window_error(wb, &lb, &baked);
    eprintln!("area 1: live surface vs its bake {eb:.2e} m");
    // The bake's splat of the same quanta: the difference is the residual
    // motion (sand still sliding when the frame's surface was built) and the
    // fixed point, under a tenth of a layer (the single-area switch error,
    // "Rendering and showcase").
    assert!(ea < 0.1 * sc.dx && eb < 0.1 * sc.dx, "{ea} {eb}");
    // Nothing open: both patches sink.
    let views = sc.surface();
    assert!(views.iter().all(Option::is_none));
    assert!(sc.patch(1).iter().all(|&h| h == -1.0e4), "patch 1 sunk");
}
