//! Area clocks and block sleep (`engine/docs/deformable-terrain.md`, "Area
//! clocks and block sleep"): two opt-in ways to spend substeps only where the
//! CFL bound needs them. Each run here is compared with the same scene on the
//! shared clock without sleep, on the same device (native wgpu is not bit
//! reproducible across runs, so the comparison is old path against new path
//! on the same scene, with a repeat of the old path for the noise floor).
//!
//! Checked from physics: the volume ledger is exact in quanta; the craters
//! keep their cavity volume, depth and rim within the stated bands; a strike
//! wakes sleeping sand.
//!
//! The event is the crater validation's 5 mm aluminium sphere at 2 km/s into
//! dry sand (transient radius 0.147 m) at 8 cells per radius.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, Material, WorldCapacity};
use engine_terrain::layout::{MPM_MAX_AREAS, MPM_PARTICLE_ASLEEP, MPM_PARTICLE_DEAD};
use engine_terrain::plan::source_peak_speed;
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
const AREAS: usize = MPM_MAX_AREAS as usize;

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

fn config(area_clocks: bool, sleep_speed: f32) -> TerrainConfig {
    TerrainConfig {
        max_regions: 4,
        max_particles: 1 << 20,
        max_blocks: 1 << 14,
        max_parcels: 1 << 18,
        max_substeps: 1 << 12,
        cells_per_radius: 8.0,
        min_cell_size: 0.002,
        area_clocks,
        sleep_speed,
        ..TerrainConfig::default()
    }
}

/// What a run did, per frame: the frame's substeps, each area's, the blocks
/// of its last substep, the particles asleep.
#[derive(Default)]
struct Work {
    substeps: u64,
    area_substeps: u64,
    blocks: u64,
    frames: u64,
}

struct Scene {
    s: WgpuSim,
    cfg: TerrainConfig,
    sand: SandMaterial,
    field: HeightField,
    bed: u32,
    crater: CraterPrediction,
    model: ZModel,
    /// Each area slot's speed bound basis (m/s).
    speed: [f32; AREAS],
    dx: f32,
    escaped: u32,
    airborne: u32,
    work: Work,
}

impl Scene {
    fn new(area_clocks: bool, sleep_speed: f32) -> Self {
        let mut s = sim();
        let cfg = config(area_clocks, sleep_speed);
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
        Self {
            s,
            cfg,
            sand,
            field,
            bed,
            crater,
            model,
            speed: [0.0; AREAS],
            dx: 0.0,
            escaped: 0,
            airborne: 0,
            work: Work::default(),
        }
    }

    fn impact(&mut self, at: Vec3) -> u32 {
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
        let o = self
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
        self.s
            .terrain_apply_source_to(o.area, &self.model, at)
            .unwrap();
        let peak = source_peak_speed(&self.model);
        let k = o.area as usize;
        if o.activated && o.absorbed.is_empty() {
            self.speed[k] = self.speed[k].max(peak);
        } else {
            let merged = o
                .absorbed
                .iter()
                .map(|&a| self.speed[a as usize])
                .fold(self.speed[k], f32::max);
            self.speed[k] = merged + peak;
        }
        self.dx = dx;
        o.area
    }

    /// `n` frames: each open area's CFL count from its own speed bound (the
    /// World's schedule, per area); the frame takes the finest, and on the
    /// area clocks each area its own.
    fn frames(&mut self, n: u32) {
        for _ in 0..n {
            let open: Vec<u32> = self.s.terrain_areas().iter().map(|a| a.slot).collect();
            let mut each = [0u32; AREAS];
            let mut frame = 1;
            for &k in &open {
                let (n, _) = substeps_for(
                    &self.cfg,
                    FRAME,
                    self.dx,
                    self.sand.p_wave_speed(),
                    1.5 * self.speed[k as usize] + 9.81 * FRAME + 20.0,
                )
                .unwrap();
                each[k as usize] = n;
                frame = frame.max(n);
            }
            self.s.terrain_set_area_substeps(each).unwrap();
            let r = self.s.terrain_step(frame, FRAME / frame as f32, G).unwrap();
            for &k in &open {
                self.speed[k as usize] = r.area_speed[k as usize];
            }
            self.escaped = r.escaped;
            self.airborne = r.airborne;
            self.work.frames += 1;
            self.work.substeps += u64::from(frame);
            self.work.area_substeps += open
                .iter()
                .map(|&k| {
                    if self.cfg.area_clocks {
                        u64::from(each[k as usize])
                    } else {
                        u64::from(frame)
                    }
                })
                .sum::<u64>();
            self.work.blocks += u64::from(r.blocks);
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

    fn heights(&self) -> Vec<f32> {
        let base = engine_schema::convex::convex_params_base(self.s.collider(self.bed).params);
        self.s
            .read_words(18, u64::from(base + 2) * 4, (NX * NZ) as usize)
            .into_iter()
            .map(f32::from_bits)
            .collect()
    }

    fn quantum(&self) -> (f64, f64) {
        let dx = f64::from(self.dx);
        let s = dx / f64::from(self.cfg.particles_per_cell_axis);
        let (a, b) = self.field.cell_size();
        (s * s * s / 64.0, f64::from(a) * f64::from(b))
    }

    fn ledger(&self, heights: &[f32]) -> i64 {
        let (q, area) = self.quantum();
        let field: i64 = heights
            .iter()
            .map(|&h| (f64::from(h) * area / q).round() as i64)
            .sum();
        let live: i64 = self
            .s
            .terrain_particles()
            .iter()
            .filter(|p| p.f[2][3].to_bits() & MPM_PARTICLE_DEAD == 0)
            .map(|p| i64::from(p.f[1][3].to_bits()))
            .sum();
        field + live + i64::from(self.escaped)
    }

    /// Live particles, and how many of them are asleep.
    fn asleep(&self) -> (usize, usize) {
        let ps = self.s.terrain_particles();
        let live: Vec<_> = ps
            .iter()
            .filter(|p| p.f[2][3].to_bits() & MPM_PARTICLE_DEAD == 0)
            .collect();
        let asleep = live
            .iter()
            .filter(|p| p.f[2][3].to_bits() & MPM_PARTICLE_ASLEEP != 0)
            .count();
        (live.len(), asleep)
    }

    fn x(&self, i: u32) -> f32 {
        (i as f32 - 0.5 * (NX - 1) as f32) * self.field.cell_size().0
    }

    fn z(&self, j: u32) -> f32 {
        (j as f32 - 0.5 * (NZ - 1) as f32) * self.field.cell_size().1
    }

    /// The crater at `at`: cavity below the flat surface (m³), deepest
    /// point (m), highest rim (m) and the rim's distance from `at` (m).
    fn crater(&self, h: &[f32], at: Vec3) -> (f64, f32, f32, f32) {
        let (_, area) = self.quantum();
        let reach = 3.0 * self.crater.radius as f32;
        let (mut cavity, mut deep, mut rim, mut rim_r) = (0.0, 0.0f32, 0.0f32, 0.0f32);
        for j in 0..NZ {
            for i in 0..NX {
                let r = (self.x(i) - at.x).hypot(self.z(j) - at.z);
                if r > reach {
                    continue;
                }
                let v = h[(j * NX + i) as usize];
                cavity += f64::from((-v).max(0.0)) * area;
                deep = deep.min(v);
                if v > rim {
                    rim = v;
                    rim_r = r;
                }
            }
        }
        (cavity, deep, rim, rim_r)
    }
}

/// The largest height difference of two heightfields near `at`, in quanta.
fn worst(sc: &Scene, a: &[f32], b: &[f32], at: Vec3) -> f32 {
    let (q, area) = sc.quantum();
    let reach = 3.0 * sc.crater.radius as f32;
    let mut w = 0.0f32;
    for j in 0..NZ {
        for i in 0..NX {
            if (sc.x(i) - at.x).hypot(sc.z(j) - at.z) > reach {
                continue;
            }
            let k = (j * NX + i) as usize;
            w = w.max((a[k] - b[k]).abs());
        }
    }
    w / (q / area) as f32
}

const P1: Vec3 = Vec3::new(-0.9, 0.0, 0.0);
const P2: Vec3 = Vec3::new(0.9, 0.0, 0.0);

/// A first impact settles a while; a second, far off, lands on fresh ground
/// while the first still creeps; both settle, land and bake.
fn two_impacts(area_clocks: bool, sleep: f32) -> (Scene, Vec<f32>) {
    let mut sc = Scene::new(area_clocks, sleep);
    sc.impact(P1);
    sc.frames(20);
    sc.impact(P2);
    sc.frames(40);
    sc.land(240);
    sc.s.terrain_bake().unwrap();
    let h = sc.heights();
    (sc, h)
}

fn report(name: &str, sc: &Scene, h: &[f32], base: &Scene, hb: &[f32], noise: f32) {
    let w = sc.work.frames.max(1) as f64;
    eprintln!(
        "{name}: frames {}, substeps/frame {:.1}, area-substeps/frame {:.1} (shared {:.1}), \
         blocks (last substep) {:.0}/frame (shared {:.0})",
        sc.work.frames,
        sc.work.substeps as f64 / w,
        sc.work.area_substeps as f64 / w,
        base.work.area_substeps as f64 / base.work.frames.max(1) as f64,
        sc.work.blocks as f64 / w,
        base.work.blocks as f64 / base.work.frames.max(1) as f64,
    );
    for (k, at) in [P1, P2].into_iter().enumerate() {
        let (v, d, rim, rr) = sc.crater(h, at);
        let (vb, db, rimb, rrb) = base.crater(hb, at);
        eprintln!(
            "  crater {}: cavity {v:.4e} m³ (shared {vb:.4e}, {:+.2}%), deepest {d:.4} m ({db:.4}), \
             rim {rim:.4} m at {rr:.3} m ({rimb:.4} at {rrb:.3}); worst height difference {:.1} \
             quanta (repeat of the shared run: {noise:.1})",
            k + 1,
            100.0 * (v - vb) / vb,
            worst(sc, h, hb, at)
        );
    }
}

#[test]
fn area_clocks_step_a_quiet_area_on_its_own_substeps() {
    let (base, hb) = two_impacts(false, 0.0);
    let (again, ha) = two_impacts(false, 0.0);
    let (sc, h) = two_impacts(true, 0.0);
    assert_eq!(base.ledger(&hb), 0, "the shared clock's ledger");
    assert_eq!(sc.ledger(&h), 0, "the area clocks' ledger");
    let noise = worst(&base, &hb, &ha, P1).max(worst(&base, &hb, &ha, P2));
    report("area clocks", &sc, &h, &base, &hb, noise);
    assert!(
        sc.work.area_substeps < base.work.area_substeps,
        "the quiet area took fewer substeps"
    );
    for at in [P1, P2] {
        let (v, d, rim, _) = sc.crater(&h, at);
        let (vb, db, rimb, _) = base.crater(&hb, at);
        assert!(((v - vb) / vb).abs() < 0.02, "cavity {v} against {vb}");
        assert!((d - db).abs() < 0.1 * db.abs(), "deepest {d} against {db}");
        assert!((rim - rimb).abs() < 0.25 * rimb, "rim {rim} against {rimb}");
    }
}

#[test]
fn block_sleep_freezes_settled_sand_and_a_strike_wakes_it() {
    let sleep = 0.01;
    let (base, hb) = two_impacts(false, 0.0);
    let (again, ha) = two_impacts(false, 0.0);
    let noise = worst(&base, &hb, &ha, P1).max(worst(&base, &hb, &ha, P2));
    drop(again);

    // The same scene with sleep: the first crater's sand falls asleep while
    // the second strike runs, and the second strike's own ground is woken
    // by its source.
    let mut sc = Scene::new(false, sleep);
    sc.impact(P1);
    sc.frames(20);
    let (live1, asleep1) = sc.asleep();
    sc.impact(P2);
    sc.frames(1);
    let (live2, asleep2) = sc.asleep();
    sc.frames(39);
    let (live3, asleep3) = sc.asleep();
    sc.land(240);
    sc.s.terrain_bake().unwrap();
    let h = sc.heights();
    eprintln!(
        "asleep: {asleep1}/{live1} after the first impact's 20 frames, {asleep2}/{live2} a frame \
         after the second, {asleep3}/{live3} after 40 more"
    );
    assert_eq!(sc.ledger(&h), 0, "the sleeping run's ledger");
    report("block sleep", &sc, &h, &base, &hb, noise);
    assert!(asleep3 > 0, "settled sand fell asleep");
    for at in [P1, P2] {
        let (v, d, rim, _) = sc.crater(&h, at);
        let (vb, db, rimb, _) = base.crater(&hb, at);
        assert!(((v - vb) / vb).abs() < 0.02, "cavity {v} against {vb}");
        assert!((d - db).abs() < 0.1 * db.abs(), "deepest {d} against {db}");
        assert!((rim - rimb).abs() < 0.25 * rimb, "rim {rim} against {rimb}");
    }
}
