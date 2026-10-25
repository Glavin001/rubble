//! The terrain plan on native Metal (`MetalSim`, Slang → MSL) against the
//! native wgpu host (`WgpuSim`, Slang → SPIR-V → WGSL → naga → MSL) for the
//! same commands: heightfield contacts in the rigid step, host-loaded and
//! activated sand, the Z-model source, uncoupled and coupled substeps, ejecta
//! and the bake. Run one process at a time (they share the GPU):
//!
//! `cargo test --release -p engine-metal --test terrain_parity -- --test-threads=1 --nocapture`

use engine_metal::{MetalSim, TerrainCaps};
use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_terrain::layout::{MpmParcel, MpmParticle};
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    substeps_for, CraterTarget, ImpactSource, RegionPlan, SandMaterial, TerrainConfig,
    TerrainStepReport,
};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Quat, Vec3};

const FRAME: f32 = 1.0 / 60.0;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn params() -> AvbdParams {
    let mut p = AvbdParams {
        dt: FRAME,
        iterations: 10,
        ..AvbdParams::default()
    };
    p.gravity_y = -9.81;
    p.sleep_time = 1.0e9;
    p
}

const MAT: Material = Material {
    friction: 0.5,
    restitution: 0.0,
    contact_offset: 0.01,
    _pad: 0.0,
};

/// The portable path: the plan's own kernels, whatever ENGINE_TERRAIN_CAPS says.
fn metal() -> MetalSim {
    let mut s = MetalSim::new(capacity()).expect("metal");
    s.set_params(params());
    s.set_materials(&[MAT; 64]);
    s.set_terrain_caps(TerrainCaps::default());
    s
}

/// Every native fast path on.
fn metal_fast() -> MetalSim {
    let mut s = metal();
    s.set_terrain_caps(TerrainCaps::all());
    s
}

fn wgpu() -> WgpuSim {
    let mut s = WgpuSim::new(capacity()).expect("wgpu adapter");
    s.set_params(params());
    s.set_material(0, MAT);
    s
}

/// The same calls on both hosts.
trait Host {
    fn terrain_enable(&mut self, c: &TerrainConfig);
    fn set_sand(&mut self, m: &SandMaterial);
    fn add_field(&mut self, arena: &mut HullArena, f: &HeightField, at: Vec3) -> u32;
    fn add_body(&mut self, b: BodyState, c: Collider) -> u32;
    fn load(&mut self, plan: &RegionPlan, ps: &[MpmParticle]);
    fn activate(&mut self, plan: &RegionPlan, bed: u32, f: &HeightField, sand: &SandMaterial);
    fn source(&mut self, m: &engine_terrain::ZModel);
    fn step(&mut self, n: u32, dt: f32) -> TerrainStepReport;
    fn coupled(&mut self, n: u32, dt: f32) -> TerrainStepReport;
    fn rigid_step(&mut self);
    fn bake(&mut self);
    fn particles(&mut self) -> Vec<MpmParticle>;
    fn parcels(&mut self) -> Vec<MpmParcel>;
    fn bodies(&mut self) -> Vec<BodyState>;
    fn heights(&mut self, bed: u32, f: &HeightField) -> Vec<f32>;
}

impl Host for MetalSim {
    fn terrain_enable(&mut self, c: &TerrainConfig) {
        MetalSim::terrain_enable(self, c).unwrap()
    }
    fn set_sand(&mut self, m: &SandMaterial) {
        self.terrain_set_material(Some(m)).unwrap()
    }
    fn add_field(&mut self, arena: &mut HullArena, f: &HeightField, at: Vec3) -> u32 {
        self.add_height_field(arena, f, at, Quat::IDENTITY, 0)
            .unwrap()
            .1
    }
    fn add_body(&mut self, b: BodyState, mut c: Collider) -> u32 {
        c.body = self.add_bodies(&[b]).start;
        self.add_colliders(&[c]).start
    }
    fn load(&mut self, plan: &RegionPlan, ps: &[MpmParticle]) {
        self.terrain_load(plan, ps, None, 0.5).unwrap()
    }
    fn activate(&mut self, plan: &RegionPlan, bed: u32, f: &HeightField, sand: &SandMaterial) {
        self.terrain_activate(
            plan,
            bed,
            f,
            Vec3::ZERO,
            sand.density,
            sand.friction_angle.tan(),
        )
        .unwrap()
    }
    fn source(&mut self, m: &engine_terrain::ZModel) {
        self.terrain_apply_source(m, Vec3::ZERO).unwrap()
    }
    fn step(&mut self, n: u32, dt: f32) -> TerrainStepReport {
        self.terrain_step(n, dt, -9.81).unwrap()
    }
    fn coupled(&mut self, n: u32, dt: f32) -> TerrainStepReport {
        self.step_with_terrain(n, dt).unwrap().1
    }
    fn rigid_step(&mut self) {
        self.step().unwrap();
    }
    fn bake(&mut self) {
        self.terrain_bake().unwrap()
    }
    fn particles(&mut self) -> Vec<MpmParticle> {
        self.terrain_particles()
    }
    fn parcels(&mut self) -> Vec<MpmParcel> {
        self.terrain_parcels()
    }
    fn bodies(&mut self) -> Vec<BodyState> {
        self.download_bodies()
    }
    fn heights(&mut self, bed: u32, f: &HeightField) -> Vec<f32> {
        self.height_field_heights(bed, f)
    }
}

impl Host for WgpuSim {
    fn terrain_enable(&mut self, c: &TerrainConfig) {
        WgpuSim::terrain_enable(self, c).unwrap()
    }
    fn set_sand(&mut self, m: &SandMaterial) {
        self.terrain_set_material(Some(m)).unwrap()
    }
    fn add_field(&mut self, arena: &mut HullArena, f: &HeightField, at: Vec3) -> u32 {
        self.add_height_field(arena, f, at, Quat::IDENTITY, 0)
            .unwrap()
            .1
    }
    fn add_body(&mut self, b: BodyState, mut c: Collider) -> u32 {
        c.body = WgpuSim::add_body(self, b);
        self.add_collider(c)
    }
    fn load(&mut self, plan: &RegionPlan, ps: &[MpmParticle]) {
        self.terrain_load(plan, ps, None, 0.5).unwrap()
    }
    fn activate(&mut self, plan: &RegionPlan, bed: u32, f: &HeightField, sand: &SandMaterial) {
        self.terrain_activate(
            plan,
            bed,
            f,
            Vec3::ZERO,
            sand.density,
            sand.friction_angle.tan(),
        )
        .unwrap()
    }
    fn source(&mut self, m: &engine_terrain::ZModel) {
        self.terrain_apply_source(m, Vec3::ZERO).unwrap()
    }
    fn step(&mut self, n: u32, dt: f32) -> TerrainStepReport {
        self.terrain_step(n, dt, -9.81).unwrap()
    }
    fn coupled(&mut self, n: u32, dt: f32) -> TerrainStepReport {
        self.step_with_terrain(n, dt).unwrap().1
    }
    fn rigid_step(&mut self) {
        self.step().unwrap();
    }
    fn bake(&mut self) {
        self.terrain_bake().unwrap()
    }
    fn particles(&mut self) -> Vec<MpmParticle> {
        self.terrain_particles()
    }
    fn parcels(&mut self) -> Vec<MpmParcel> {
        self.terrain_parcels()
    }
    fn bodies(&mut self) -> Vec<BodyState> {
        self.download_bodies()
    }
    fn heights(&mut self, bed: u32, f: &HeightField) -> Vec<f32> {
        let base = engine_schema::convex::convex_params_base(self.collider(bed).params);
        let (nx, nz) = f.samples();
        self.read_words(18, u64::from(base + 2) * 4, (nx * nz) as usize)
            .into_iter()
            .map(f32::from_bits)
            .collect()
    }
}

/// How far two particle sets are apart.
#[derive(Debug)]
struct Diff {
    /// Particles bit-identical in every field.
    identical: usize,
    count: usize,
    /// Largest position and velocity difference (m, m/s).
    dx: f32,
    dv: f32,
}

fn diff(a: &[MpmParticle], b: &[MpmParticle]) -> Diff {
    assert_eq!(a.len(), b.len(), "particle counts differ");
    let mut d = Diff {
        identical: 0,
        count: a.len(),
        dx: 0.0,
        dv: 0.0,
    };
    for (p, q) in a.iter().zip(b) {
        if bytemuck::bytes_of(p) == bytemuck::bytes_of(q) {
            d.identical += 1;
        }
        for k in 0..3 {
            d.dx = d.dx.max((p.xm[k] - q.xm[k]).abs());
            d.dv = d.dv.max((p.vj[k] - q.vj[k]).abs());
        }
    }
    d
}

/// Frame reports, particles and the speed of their fastest particle.
struct Run {
    reports: Vec<TerrainStepReport>,
    particles: Vec<MpmParticle>,
    parcels: Vec<MpmParcel>,
    heights: Vec<f32>,
    bodies: Vec<BodyState>,
}

fn sand_cube(h: &mut dyn Host) -> Run {
    let cfg = TerrainConfig {
        max_particles: 1 << 16,
        max_blocks: 1 << 12,
        ..TerrainConfig::default()
    };
    h.terrain_enable(&cfg);
    let sand = SandMaterial::default();
    h.set_sand(&sand);
    let dx = 0.05;
    let plan = RegionPlan::spanning(&cfg, Vec3::splat(-0.6), Vec3::splat(0.6), dx, 0, 0).unwrap();
    let mut ps = Vec::new();
    let sp = dx / 2.0;
    for i in 0..12 {
        for j in 0..12 {
            for k in 0..12 {
                let x = Vec3::new(i as f32, j as f32, k as f32) * sp - Vec3::splat(0.15);
                let mut p = MpmParticle::at_rest(x.to_array(), sp * sp * sp, 1600.0);
                p.vj = [0.3 * x.y, -0.5, 0.2 * x.x, 0.0];
                ps.push(p);
            }
        }
    }
    h.load(&plan, &ps);
    let (n, dt) = substeps_for(&cfg, FRAME, dx, sand.p_wave_speed(), 5.0).unwrap();
    let reports = (0..6).map(|_| h.step(n, dt)).collect();
    Run {
        reports,
        particles: h.particles(),
        parcels: Vec::new(),
        heights: Vec::new(),
        bodies: Vec::new(),
    }
}

/// An impact into a flat sand field: activation, the Z-model source, then
/// `frames` frames (coupled with a falling box when `coupled`), then a bake.
fn impact(h: &mut dyn Host, frames: usize, coupled: bool) -> Run {
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        max_blocks: 1 << 13,
        max_parcels: 1 << 16,
        max_substeps: 1 << 12,
        cells_per_radius: 8.0,
        min_cell_size: 0.002,
        ..TerrainConfig::default()
    };
    h.terrain_enable(&cfg);
    let sand = SandMaterial::default();
    h.set_sand(&sand);
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
    let n = 65u32;
    let field = HeightField::new(n, n, spacing, spacing, vec![0.0; (n * n) as usize]).unwrap();
    let mut arena = HullArena::new();
    let bed = h.add_field(&mut arena, &field, Vec3::ZERO);
    let plan = crater.region(&cfg, Vec3::ZERO, 0.6, 0, 0).unwrap();
    h.activate(&plan, bed, &field, &sand);
    h.source(&model);
    if coupled {
        let hb = Vec3::splat(0.02);
        let inertia = Vec3::splat(2.0 * hb.x * hb.x) * (0.05 / 3.0);
        let mut state =
            BodyState::dynamic(Vec3::new(0.06, 0.08, 0.0), Quat::IDENTITY, 0.05, inertia);
        state.linear_vel = [0.0, -1.0, 0.0];
        h.add_body(
            state,
            Collider {
                body: 0,
                shape: ShapeType::Box as u32,
                material: 0,
                generation: 0,
                local_pos: [0.0; 3],
                density: 1.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [hb.x, hb.y, hb.z, hb.length()],
            },
        );
    }
    let mut speed = engine_terrain::plan::source_peak_speed(&model);
    let mut reports = Vec::new();
    for _ in 0..frames {
        let (k, dt) = substeps_for(
            &cfg,
            FRAME,
            plan.dx,
            sand.p_wave_speed(),
            1.5 * speed + 20.0,
        )
        .unwrap();
        let r = if coupled {
            h.coupled(k, dt)
        } else {
            h.step(k, dt)
        };
        speed = r.max_speed;
        reports.push(r);
    }
    let particles = h.particles();
    let parcels = h.parcels();
    let bodies = h.bodies();
    h.bake();
    Run {
        reports,
        particles,
        parcels,
        heights: h.heights(bed, &field),
        bodies,
    }
}

fn report_line(name: &str, m: &Run, w: &Run) -> Diff {
    let d = diff(&m.particles, &w.particles);
    let ident_reports = m
        .reports
        .iter()
        .zip(&w.reports)
        .filter(|(a, b)| a == b)
        .count();
    let dh = m
        .heights
        .iter()
        .zip(&w.heights)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let same_heights = m
        .heights
        .iter()
        .zip(&w.heights)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    println!(
        "{name}: {}/{} particles bit-identical, max |dx| {:.3e} m, max |dv| {:.3e} m/s; \
         {ident_reports}/{} frame reports equal; heights {same_heights}/{} identical, max |dh| \
         {dh:.3e} m; parcels {} vs {}",
        d.identical,
        d.count,
        d.dx,
        d.dv,
        m.reports.len(),
        m.heights.len(),
        m.parcels.len(),
        w.parcels.len()
    );
    for (i, (a, b)) in m.reports.iter().zip(&w.reports).enumerate() {
        if a != b {
            println!("  frame {i}: metal {a:?}\n           wgpu  {b:?}");
        }
    }
    d
}

/// The native fast paths (`TerrainCaps::all`) against the plan's portable
/// kernels on Metal: deterministic run to run, and within the rounding the
/// two codegens already differ by.
#[test]
fn the_native_fast_paths_match_the_portable_kernels() {
    let f = impact(&mut metal_fast(), 4, false);
    let g = impact(&mut metal_fast(), 4, false);
    assert_eq!(f.reports, g.reports, "two fast-path runs differ");
    let d = diff(&f.particles, &g.particles);
    assert_eq!(d.identical, d.count, "two fast-path runs differ: {d:?}");
    let p = impact(&mut metal(), 4, false);
    let d = report_line("fast paths vs portable (Metal)", &f, &p);
    assert!(d.dx < 1e-4 && d.dv < 1e-2, "{d:?}");
    for (a, b) in f.reports.iter().zip(&p.reports) {
        assert_eq!(a.particles, b.particles);
        assert!((a.max_speed - b.max_speed).abs() <= 0.05 * b.max_speed.max(1.0));
    }
}

#[test]
fn metal_terrain_runs_are_bit_identical() {
    let a = impact(&mut metal(), 4, false);
    let b = impact(&mut metal(), 4, false);
    assert_eq!(a.reports, b.reports);
    let d = diff(&a.particles, &b.particles);
    assert_eq!(d.identical, d.count, "two Metal runs differ: {d:?}");
    assert!(
        a.heights
            .iter()
            .zip(&b.heights)
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "two Metal bakes differ"
    );
}

#[test]
fn a_loaded_sand_cube_matches_wgpu() {
    let m = sand_cube(&mut metal());
    let w = sand_cube(&mut wgpu());
    let d = report_line("sand cube", &m, &w);
    assert!(d.dx < 1e-4 && d.dv < 1e-2, "{d:?}");
    for (a, b) in m.reports.iter().zip(&w.reports) {
        assert_eq!((a.blocks, a.particles), (b.blocks, b.particles));
        assert!((a.max_speed - b.max_speed).abs() <= 1e-4 * b.max_speed.max(1.0));
    }
}

#[test]
fn an_impact_matches_wgpu() {
    let m = impact(&mut metal(), 4, false);
    let w = impact(&mut wgpu(), 4, false);
    let d = report_line("impact, uncoupled", &m, &w);
    assert_eq!(m.particles.len(), w.particles.len());
    // A km/s source over a few frames: the chaotic flow amplifies rounding,
    // so compare the region's state by its ledger and bulk quantities.
    for (a, b) in m.reports.iter().zip(&w.reports) {
        assert_eq!(a.particles, b.particles);
        assert!(
            (a.max_speed - b.max_speed).abs() <= 0.05 * b.max_speed.max(1.0),
            "max speed {} vs {}",
            a.max_speed,
            b.max_speed
        );
    }
    let mean_y = |r: &Run| {
        r.particles.iter().map(|p| f64::from(p.xm[1])).sum::<f64>() / r.particles.len() as f64
    };
    assert!(
        (mean_y(&m) - mean_y(&w)).abs() < 1e-4,
        "mean height {} vs {}",
        mean_y(&m),
        mean_y(&w)
    );
    let vol = |r: &Run| r.heights.iter().map(|&h| f64::from(h)).sum::<f64>();
    println!("baked volume sums {:.6} vs {:.6}; {d:?}", vol(&m), vol(&w));
}

#[test]
fn a_coupled_impact_matches_wgpu() {
    let m = impact(&mut metal(), 6, true);
    let w = impact(&mut wgpu(), 6, true);
    report_line("impact, coupled with a box", &m, &w);
    let (a, b) = (m.bodies.last().unwrap(), w.bodies.last().unwrap());
    println!("box at {:?} (metal) vs {:?} (wgpu)", a.position, b.position);
    for k in 0..3 {
        assert!(
            (a.position[k] - b.position[k]).abs() < 2e-3,
            "box {:?} vs {:?}",
            a.position,
            b.position
        );
    }
}

/// Spheres and boxes dropped on a bumpy heightfield: the rigid step's
/// heightfield contacts (the convex stages) on Metal against wgpu.
#[test]
fn bodies_on_a_heightfield_match_wgpu() {
    fn run(h: &mut dyn Host) -> Vec<BodyState> {
        let n = 33u32;
        let s = 0.25f32;
        let heights: Vec<f32> = (0..n * n)
            .map(|k| {
                let (i, j) = ((k % n) as f32, (k / n) as f32);
                0.3 * (0.4 * i).sin() * (0.3 * j).cos()
            })
            .collect();
        let field = HeightField::new(n, n, s, s, heights).unwrap();
        let mut arena = HullArena::new();
        h.add_field(&mut arena, &field, Vec3::ZERO);
        for k in 0..6 {
            let x = -1.5 + 0.6 * k as f32;
            let sphere = k % 2 == 0;
            let r = 0.2f32;
            let inertia = Vec3::splat(if sphere {
                0.4 * r * r
            } else {
                2.0 * r * r / 3.0
            });
            let b = BodyState::dynamic(
                Vec3::new(x, 1.0 + 0.1 * k as f32, 0.3 * x),
                Quat::IDENTITY,
                1.0,
                inertia,
            );
            h.add_body(
                b,
                Collider {
                    body: 0,
                    shape: if sphere {
                        ShapeType::Sphere
                    } else {
                        ShapeType::Box
                    } as u32,
                    material: 0,
                    generation: 0,
                    local_pos: [0.0; 3],
                    density: 1.0,
                    local_rot: [0.0, 0.0, 0.0, 1.0],
                    params: if sphere {
                        [r, 0.0, 0.0, r]
                    } else {
                        [r, r, r, r * 3f32.sqrt()]
                    },
                },
            );
        }
        for _ in 0..120 {
            h.rigid_step();
        }
        h.bodies()
    }
    let m = run(&mut metal());
    let w = run(&mut wgpu());
    let mut worst = 0.0f32;
    for (a, b) in m.iter().zip(&w).skip(1) {
        for k in 0..3 {
            worst = worst.max((a.position[k] - b.position[k]).abs());
        }
        println!("metal {:?} wgpu {:?}", a.position, b.position);
        // Resting on the field, not through it.
        assert!(a.position[1] > -0.5, "body fell through: {:?}", a.position);
        assert!(a.position.iter().all(|x| x.is_finite()));
    }
    println!("bodies on a heightfield: worst position difference {worst:.3e} m");
    assert!(worst < 5e-3, "metal vs wgpu diverged by {worst}");
}

/// Where the two codegens part: activation and activate-then-bake are
/// integer work and must agree bit for bit; the source and one substep are
/// float arithmetic, printed.
#[test]
fn activation_and_bake_match_wgpu_bit_for_bit() {
    struct Stages {
        lowered: Vec<f32>,
        activated: Vec<MpmParticle>,
        sourced: Vec<MpmParticle>,
        stepped: Vec<MpmParticle>,
        baked: Vec<f32>,
        rest_baked: Vec<f32>,
    }
    fn run(h: &mut dyn Host, step: bool) -> Stages {
        let cfg = TerrainConfig {
            max_particles: 1 << 18,
            max_blocks: 1 << 13,
            max_parcels: 1 << 16,
            max_substeps: 1 << 12,
            cells_per_radius: 8.0,
            min_cell_size: 0.002,
            ..TerrainConfig::default()
        };
        h.terrain_enable(&cfg);
        let sand = SandMaterial::default();
        h.set_sand(&sand);
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
        let n = 65u32;
        let field = HeightField::new(n, n, spacing, spacing, vec![0.0; (n * n) as usize]).unwrap();
        let mut arena = HullArena::new();
        let bed = h.add_field(&mut arena, &field, Vec3::ZERO);
        let plan = crater.region(&cfg, Vec3::ZERO, 0.6, 0, 0).unwrap();
        h.activate(&plan, bed, &field, &sand);
        let lowered = h.heights(bed, &field);
        let activated = h.particles();
        if !step {
            h.bake();
            let rest_baked = h.heights(bed, &field);
            return Stages {
                lowered,
                activated,
                sourced: Vec::new(),
                stepped: Vec::new(),
                baked: Vec::new(),
                rest_baked,
            };
        }
        h.source(&model);
        let sourced = h.particles();
        let speed = 1.5 * engine_terrain::plan::source_peak_speed(&model) + 20.0;
        let (_, dt) = substeps_for(&cfg, FRAME, plan.dx, sand.p_wave_speed(), speed).unwrap();
        h.step(1, dt);
        let stepped = h.particles();
        h.bake();
        Stages {
            lowered,
            activated,
            sourced,
            stepped,
            baked: h.heights(bed, &field),
            rest_baked: Vec::new(),
        }
    }
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let (m, w) = (run(&mut metal(), false), run(&mut wgpu(), false));
    assert_eq!(
        bits(&m.lowered),
        bits(&w.lowered),
        "activation lowered the field differently"
    );
    let d = diff(&m.activated, &w.activated);
    assert_eq!(d.identical, d.count, "activated particles differ: {d:?}");
    assert_eq!(
        bits(&m.rest_baked),
        bits(&w.rest_baked),
        "activate-then-bake differs"
    );
    let (m, w) = (run(&mut metal(), true), run(&mut wgpu(), true));
    println!("after the source: {:?}", diff(&m.sourced, &w.sourced));
    println!("after one substep: {:?}", diff(&m.stepped, &w.stepped));
    let same = m
        .baked
        .iter()
        .zip(&w.baked)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    println!(
        "baked after one substep: {same}/{} samples identical",
        m.baked.len()
    );
}
