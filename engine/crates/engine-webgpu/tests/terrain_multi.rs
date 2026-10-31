//! Several impacts on one terrain (`engine/docs/deformable-terrain.md`,
//! "Multiple impacts"): impacts far apart open separate areas of one sparse
//! domain, each stepping as it would alone; an impact near an open area grows
//! it (into a crater still forming, on its rim, partly outside it) and its
//! Z-model field adds to the material's motion there; baked ground
//! reactivates. Checked from physics: the volume ledger is exact in quanta,
//! a second source adds exactly its field's momentum, far areas match a
//! single impact, and an overlapping second hit digs at least the larger
//! single crater.
//!
//! The event is the crater validation's 5 mm aluminium sphere at 2 km/s into
//! dry sand (transient radius 0.147 m) at 8 cells per radius.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, Material, WorldCapacity};
use engine_terrain::layout::MpmParticle;
use engine_terrain::plan::source_peak_speed;
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    substeps_for, CraterPrediction, CraterTarget, ImpactSource, SandMaterial, TerrainConfig, ZModel,
};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Quat, Vec3};

const FRAME: f32 = 1.0 / 60.0;
const G: f32 = -9.81;
/// Field samples along x and z.
const NX: u32 = 161;
const NZ: u32 = 65;

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

fn config() -> TerrainConfig {
    TerrainConfig {
        max_particles: 1 << 20,
        max_blocks: 1 << 14,
        max_parcels: 1 << 18,
        max_substeps: 1 << 12,
        cells_per_radius: 8.0,
        min_cell_size: 0.002,
        ..TerrainConfig::default()
    }
}

/// The terrain, its flat sand field (samples an eighth of the crater's radius
/// apart, NX × NZ about the origin) and the event.
struct Scene {
    s: WgpuSim,
    cfg: TerrainConfig,
    sand: SandMaterial,
    field: HeightField,
    bed: u32,
    crater: CraterPrediction,
    model: ZModel,
    /// Speed bound basis for the next frame (m/s).
    speed: f32,
    /// The grid's cell (m), from the first impact.
    dx: f32,
    escaped: u32,
    airborne: u32,
}

impl Scene {
    fn new() -> Self {
        let mut s = sim();
        let cfg = config();
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
            speed: 0.0,
            dx: 0.0,
            escaped: 0,
            airborne: 0,
        }
    }

    fn radius(&self) -> f32 {
        self.crater.radius as f32
    }

    /// An impact at `at` (on the surface): its region on the open grid, then
    /// its source in the area that covers it. Returns that area.
    fn impact(&mut self, at: Vec3) -> engine_terrain::plan::Opened {
        self.impact_in(at, 0.6)
    }

    /// [`Self::impact`] in a region `margin` transient radii past the
    /// crater's (a tight one throws ejecta past its box as parcels).
    fn impact_in(&mut self, at: Vec3, margin: f64) -> engine_terrain::plan::Opened {
        let dx = self
            .s
            .terrain_plan_ref()
            .unwrap()
            .event_cell(self.crater.asked_cell(&self.cfg))
            .unwrap();
        let plan = self
            .crater
            .region_with_cell(&self.cfg, at, margin, dx)
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
        self.s
            .terrain_apply_source_to(opened.area, &self.model, at)
            .unwrap();
        // The World's clocks: a source on moving material adds its peak to
        // that area's bound; a new area has its own, and the frame takes the
        // fastest.
        let peak = source_peak_speed(&self.model);
        if opened.activated && opened.absorbed.is_empty() {
            self.speed = self.speed.max(peak);
        } else {
            self.speed += peak;
        }
        self.dx = dx;
        opened
    }

    fn dx(&self) -> f32 {
        self.dx
    }

    /// `n` uncoupled frames, each at the CFL bound for 1.5 × the last
    /// reported speed plus headroom (the World's schedule).
    fn frames(&mut self, n: u32) {
        let dx = self.dx();
        for _ in 0..n {
            let (k, dt) = substeps_for(
                &self.cfg,
                FRAME,
                dx,
                self.sand.p_wave_speed(),
                1.5 * self.speed + 9.81 * FRAME + 20.0,
            )
            .unwrap();
            let r = self.s.terrain_step(k, dt, G).unwrap();
            self.speed = r.max_speed;
            self.escaped = r.escaped;
            self.airborne = r.airborne;
        }
    }

    /// Frames until nothing is in flight (at most `cap`).
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

    /// The volume quantum (m³) and a sample's area (m²).
    fn quantum(&self) -> (f64, f64) {
        let dx = f64::from(self.dx());
        let s = dx / f64::from(self.cfg.particles_per_cell_axis);
        let (a, b) = self.field.cell_size();
        (s * s * s / 64.0, f64::from(a) * f64::from(b))
    }

    /// The ledger in whole quanta: the field's change (from flat), the live
    /// particles' and the escaped. Exactly zero when nothing is lost.
    fn ledger(&self, heights: &[f32], q: f64, area: f64) -> i64 {
        let field: i64 = heights
            .iter()
            .map(|&h| (f64::from(h) * area / q).round() as i64)
            .sum();
        let live: i64 = self
            .s
            .terrain_particles()
            .iter()
            .filter(|p| p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_DEAD == 0)
            .map(|p| i64::from(p.f[1][3].to_bits()))
            .sum();
        field + live + i64::from(self.escaped)
    }

    fn x(&self, i: u32) -> f32 {
        (i as f32 - 0.5 * (NX - 1) as f32) * self.field.cell_size().0
    }

    fn z(&self, j: u32) -> f32 {
        (j as f32 - 0.5 * (NZ - 1) as f32) * self.field.cell_size().1
    }
}

/// The crater around `at` (within `reach` horizontally): its cavity volume
/// below the original flat surface (m³), its deepest point (m) and the
/// largest height change between two heightfields there.
fn crater_stats(sc: &Scene, h: &[f32], at: Vec3, reach: f32) -> (f64, f32) {
    let (_, area) = sc.quantum();
    let mut cavity = 0.0;
    let mut deepest = 0.0f32;
    for j in 0..NZ {
        for i in 0..NX {
            let (x, z) = (sc.x(i), sc.z(j));
            if (x - at.x).hypot(z - at.z) > reach {
                continue;
            }
            let v = h[(j * NX + i) as usize];
            cavity += f64::from((-v).max(0.0)) * area;
            deepest = deepest.min(v);
        }
    }
    (cavity, deepest)
}

fn max_diff(sc: &Scene, a: &[f32], b: &[f32], at: Vec3, reach: f32) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut differ = 0;
    for j in 0..NZ {
        for i in 0..NX {
            if (sc.x(i) - at.x).hypot(sc.z(j) - at.z) > reach {
                continue;
            }
            let k = (j * NX + i) as usize;
            if a[k].to_bits() != b[k].to_bits() {
                differ += 1;
            }
            worst = worst.max((a[k] - b[k]).abs());
        }
    }
    (worst, differ)
}

const SETTLE_FRAMES: u32 = 40;

/// A single impact at `at`: frames, landing, bake. Returns the heights.
fn single(at: Vec3) -> (Scene, Vec<f32>) {
    let mut sc = Scene::new();
    sc.impact(at);
    sc.frames(SETTLE_FRAMES);
    sc.land(240);
    sc.s.terrain_bake().unwrap();
    let h = sc.heights();
    (sc, h)
}

#[test]
fn far_apart_impacts_each_dig_the_single_impacts_crater() {
    let r = 0.147;
    let (p1, p2) = (Vec3::new(-0.9, 0.0, 0.0), Vec3::new(0.9, 0.0, 0.0));
    let (sc1, h1) = single(p1);
    let (_, h2) = single(p2);
    let (_, h1b) = single(p1);

    let mut sc = Scene::new();
    let a = sc.impact(p1);
    let b = sc.impact(p2);
    assert_ne!(a.area, b.area, "far apart: two areas");
    assert!(b.absorbed.is_empty() && b.activated);
    assert_eq!(sc.s.terrain_areas().len(), 2);
    sc.frames(SETTLE_FRAMES);
    sc.land(240);
    // Each area bakes on its own: the first leaves the second's material moving.
    let live_before = sc.s.terrain_particles().len();
    sc.s.terrain_bake_area(a.area).unwrap();
    assert_eq!(sc.s.terrain_areas().len(), 1);
    assert_eq!(
        sc.s.terrain_particles().len(),
        live_before,
        "the pool keeps its slots while an area is open"
    );
    sc.s.terrain_bake_area(b.area).unwrap();
    assert!(sc.s.terrain_areas().is_empty());
    let h = sc.heights();

    let (q, area) = sc.quantum();
    assert_eq!(sc.ledger(&h, q, area), 0, "the two-impact ledger");
    assert_eq!(sc1.ledger(&h1, q, area), 0, "the single impact's ledger");

    let reach = 3.0 * r;
    let repeat = max_diff(&sc1, &h1, &h1b, p1, reach);
    let (d1, n1) = max_diff(&sc, &h, &h1, p1, reach);
    let (d2, n2) = max_diff(&sc, &h, &h2, p2, reach);
    let (v1, deep1) = crater_stats(&sc, &h1, p1, reach);
    let (v, deep) = crater_stats(&sc, &h, p1, reach);
    let (v2s, _) = crater_stats(&sc, &h2, p2, reach);
    let (v2, _) = crater_stats(&sc, &h, p2, reach);
    eprintln!(
        "single repeated: worst {:.3e} m ({} samples differ); two impacts against single: \
         at p1 worst {d1:.3e} m ({n1} differ), at p2 worst {d2:.3e} m ({n2} differ); cavity p1 \
         {v:.4e} vs {v1:.4e} m³, p2 {v2:.4e} vs {v2s:.4e} m³; deepest {deep:.4} vs {deep1:.4} m",
        repeat.0, repeat.1
    );
    // Each area steps in its own frame and bakes with dithers counted from its
    // own first slot. The areas share one pool of parcels, whose slots (and so
    // their deposits' dithers, and the slots of those landing back in an area)
    // interleave in the atomics' order: an area differs from its single impact
    // only by a few whole quanta where ejecta land (which area, and whether at
    // all, follows that order; so far one of the two has come out
    // bit-identical in every run, which is not asserted).
    let quantum = (q / area) as f32;
    assert!(
        d1 <= 8.0 * quantum && d2 <= 8.0 * quantum,
        "areas against single impacts: {d1} m, {d2} m; 8 quanta are {} m",
        8.0 * quantum
    );
    assert!(((v - v1) / v1).abs() < 1e-3 && ((v2 - v2s) / v2s).abs() < 1e-3);
}

/// Total momentum (f64) of the live particles.
fn momentum(ps: &[MpmParticle]) -> DVec3 {
    ps.iter()
        .filter(|p| p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_DEAD == 0)
        .map(|p| Vec3::from_slice(&p.vj[..3]).as_dvec3() * f64::from(p.xm[3]))
        .sum()
}

/// The momentum a source at `point` adds to `ps` by its definition: each
/// particle below the point within the field's extent takes its velocity.
fn field_momentum(model: &ZModel, point: Vec3, ps: &[MpmParticle]) -> DVec3 {
    ps.iter()
        .filter(|p| p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_DEAD == 0)
        .map(|p| {
            let d = (Vec3::from_slice(&p.xm[..3]) - point).as_dvec3();
            if d.y > 0.0 {
                return DVec3::ZERO;
            }
            model.velocity(d) * f64::from(p.xm[3])
        })
        .sum()
}

#[test]
fn a_second_source_adds_exactly_its_fields_momentum() {
    let mut sc = Scene::new();
    let r = sc.radius();
    let p1 = Vec3::ZERO;
    sc.impact(p1);
    let before = sc.s.terrain_particles();
    // A second, overlapping source on the same material before it moves.
    let p2 = Vec3::new(0.5 * r, 0.0, 0.2 * r);
    let o = sc.impact(p2);
    assert_eq!(sc.s.terrain_areas().len(), 1, "one area holds both");
    assert!(!o.activated || o.absorbed.len() == 1);
    let after = sc.s.terrain_particles();
    let added = momentum(&after) - momentum(&before);
    let expect = field_momentum(&sc.model, p2, &before);
    let err = (added - expect).length() / expect.length();
    eprintln!("still: added {added:?}, the field's {expect:?} (relative {err:.2e})");
    assert!(err < 1e-4);
    // Superposition: the two sources' sum, from rest.
    let total = momentum(&after);
    let both = field_momentum(&sc.model, p1, &before) + expect;
    assert!((total - both).length() / both.length() < 1e-4);

    // On moving material, the source sits on the material's surface under its
    // point (its top particle there plus half a lattice spacing).
    sc.frames(6);
    let moving = sc.s.terrain_particles();
    let s = sc.dx() / sc.cfg.particles_per_cell_axis as f32;
    let p3 = Vec3::new(-0.4 * r, 0.0, 0.0);
    let top = moving
        .iter()
        .filter(|p| p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_DEAD == 0)
        .filter(|p| (p.xm[0] - p3.x).abs() <= s && (p.xm[2] - p3.z).abs() <= s)
        .map(|p| p.xm[1] + 0.5 * s)
        .fold(f32::MIN, f32::max);
    sc.impact(p3);
    let kicked = sc.s.terrain_particles();
    let added = momentum(&kicked) - momentum(&moving);
    let surface = Vec3::new(p3.x, top, p3.z);
    let expect = field_momentum(&sc.model, surface, &moving);
    let at_point = field_momentum(&sc.model, p3, &moving);
    let err = (added - expect).length() / expect.length();
    eprintln!(
        "moving: surface {top:.4} m; added {added:?}, the field's there {expect:?} (relative \
         {err:.2e}); at the point's own height it would be {at_point:?}"
    );
    assert!(err < 1e-4);
}

/// The second hit of a pair, `frames` after the first at the origin, at
/// `at`; both settle and bake. Returns the scene and heights.
fn pair(at: Vec3, frames: u32) -> (Scene, Vec<f32>, engine_terrain::plan::Opened) {
    let mut sc = Scene::new();
    sc.impact(Vec3::ZERO);
    sc.frames(frames);
    let o = sc.impact(at);
    sc.frames(SETTLE_FRAMES);
    sc.land(240);
    sc.s.terrain_bake().unwrap();
    let h = sc.heights();
    (sc, h, o)
}

fn check_overlap(name: &str, at: Vec3, frames: u32) -> engine_terrain::plan::Opened {
    let (single_sc, h1) = single(Vec3::ZERO);
    let (sc, h, o) = pair(at, frames);
    assert!(
        !o.activated || o.absorbed.len() == 1,
        "{name}: the open area grew or covered it"
    );
    let (q, area) = sc.quantum();
    let ledger = sc.ledger(&h, q, area);
    let r = sc.radius();
    let (v1, d1) = crater_stats(&single_sc, &h1, Vec3::ZERO, 4.0 * r);
    let (v, d) = crater_stats(&sc, &h, Vec3::ZERO, 4.0 * r + at.length());
    eprintln!(
        "{name}: cavity {v:.4e} m³ against the single crater's {v1:.4e} ({:.2}×), deepest \
         {d:.4} m against {d1:.4}; ledger {ledger} quanta; areas absorbed {:?}",
        v / v1,
        o.absorbed
    );
    assert_eq!(ledger, 0, "{name}: volume is kept exactly");
    assert!(
        v >= v1,
        "{name}: the combined crater is at least the single one"
    );
    o
}

#[test]
fn a_hit_on_the_rim_of_a_forming_crater_grows_its_area_and_digs_deeper() {
    // The rim of the forming crater: one transient radius out, 10 frames in.
    let r = 0.147;
    let o = check_overlap("rim", Vec3::new(r, 0.0, 0.0), 10);
    assert_eq!(o.absorbed.len(), 1);
}

#[test]
fn a_hit_inside_a_still_moving_crater_adds_to_its_flow() {
    let r = 0.147;
    check_overlap("inside", Vec3::new(0.2 * r, 0.0, -0.1 * r), 10);
}

#[test]
fn a_hit_partly_outside_the_open_area_grows_it_to_cover_both() {
    // Its box reaches past the open one's (1.6 R from the centre) by half.
    let r = 0.147;
    let o = check_overlap("partly outside", Vec3::new(2.4 * r, 0.0, 0.8 * r), 4);
    assert_eq!(o.absorbed.len(), 1, "it grew the open area");
}

#[test]
fn simultaneous_far_hits_and_a_hit_on_baked_ground() {
    let mut sc = Scene::new();
    let (p1, p2) = (Vec3::new(-0.9, 0.0, 0.0), Vec3::new(0.9, 0.0, 0.0));
    // Two hits in the same frame.
    let a = sc.impact(p1);
    let b = sc.impact(p2);
    assert!(a.area != b.area && b.absorbed.is_empty());
    sc.frames(SETTLE_FRAMES);
    sc.land(240);
    sc.s.terrain_bake().unwrap();
    let h1 = sc.heights();
    let (q, area) = sc.quantum();
    assert_eq!(sc.ledger(&h1, q, area), 0);
    let (v1, d1) = crater_stats(&sc, &h1, p1, 0.6);
    // Again on the baked crater at p1: the baked ground is the same material.
    sc.speed = 0.0;
    let c = sc.impact(p1);
    assert!(c.activated && c.absorbed.is_empty());
    sc.frames(SETTLE_FRAMES);
    sc.land(240);
    sc.s.terrain_bake().unwrap();
    let h2 = sc.heights();
    assert_eq!(sc.ledger(&h2, q, area), 0);
    let (v2, d2) = crater_stats(&sc, &h2, p1, 0.6);
    eprintln!("baked ground hit again: cavity {v1:.4e} → {v2:.4e} m³, deepest {d1:.4} → {d2:.4} m");
    assert!(v2 > v1 && d2 < d1);
}

/// The domain's escaped quanta and parcels still in flight, from its counters
/// (read after a bake, which no step report follows).
fn counters(sc: &Scene) -> (u32, u32) {
    use engine_terrain::layout::{MPM_COUNTERS_BINDING, MPM_C_AIRBORNE, MPM_C_ESCAPED};
    let c = sc.s.read_words(MPM_COUNTERS_BINDING, 0, MPM_C_AIRBORNE + 1);
    (c[MPM_C_ESCAPED], c[MPM_C_AIRBORNE])
}

/// Parcels still in flight and the quanta they carry.
fn in_flight(sc: &Scene) -> (usize, u64) {
    let ps: Vec<_> =
        sc.s.terrain_parcels()
            .into_iter()
            .filter(|p| p.vm[3] > 0.0)
            .collect();
    (
        ps.len(),
        ps.iter().map(|p| u64::from(p.xq[3].to_bits())).sum(),
    )
}

/// The material of the area around `at` (within 0.6 m in x) thrown up at
/// 4 m/s and out from `at` at 1 m/s, as a blast would (this impact's own
/// ejecta stay inside its box): it leaves the area's box and flies as parcels
/// for most of a second.
fn throw(sc: &mut Scene, at: Vec3) {
    let mut ps = sc.s.terrain_particles();
    let mut n = 0;
    for p in &mut ps {
        let x = Vec3::from_slice(&p.xm[..3]);
        if (x.x - at.x).abs() < 0.6 {
            let out = Vec3::new(x.x - at.x, 0.0, x.z - at.z).normalize_or_zero() * 1.0;
            p.vj = [out.x, 4.0, out.z, p.vj[3]];
            n += 1;
        }
    }
    assert!(n > 0, "material to throw at {at}");
    sc.s.terrain_overwrite_particles(&ps).unwrap();
    sc.speed = sc.speed.max(6.0);
}

/// Frames after an impact until ejecta are in flight (at most `cap`).
fn until_airborne(sc: &mut Scene, cap: u32) {
    for _ in 0..cap {
        sc.frames(1);
        if sc.airborne > 0 {
            return;
        }
    }
    panic!("no parcels in flight after {cap} frames");
}

#[test]
fn a_forced_close_lands_the_parcels_still_in_flight() {
    let mut sc = Scene::new();
    sc.impact(Vec3::new(-0.9, 0.0, 0.0));
    throw(&mut sc, Vec3::new(-0.9, 0.0, 0.0));
    until_airborne(&mut sc, 30);
    sc.frames(2);
    let (n, quanta) = in_flight(&sc);
    assert!(n > 0, "parcels in flight at the close");
    // Before the close: the field, the live and the escaped, with what flies.
    let (q, area) = sc.quantum();
    let before = sc.heights();
    assert_eq!(
        sc.ledger(&before, q, area) + quanta as i64,
        0,
        "ledger in flight"
    );
    // Closed by force mid-flight: everything bakes, and what flies lands.
    sc.s.terrain_bake().unwrap();
    assert!(sc.s.terrain_areas().is_empty());
    let (escaped, airborne) = counters(&sc);
    assert_eq!(airborne, 0, "nothing left in flight");
    assert_eq!(in_flight(&sc).0, 0);
    sc.escaped = escaped;
    let h = sc.heights();
    assert_eq!(sc.ledger(&h, q, area), 0, "the forced close's ledger");
    eprintln!(
        "forced close: {n} parcels ({quanta} quanta) in flight landed; escaped {escaped} quanta"
    );

    // A fresh domain afterwards: nothing of the old one flies into it (the
    // parcels' source slots died with the old pool).
    let old_escaped = escaped;
    sc.speed = 0.0;
    sc.impact(Vec3::new(0.9, 0.0, 0.0));
    sc.frames(SETTLE_FRAMES);
    sc.land(240);
    sc.s.terrain_bake().unwrap();
    let (escaped, airborne) = counters(&sc);
    assert_eq!(airborne, 0);
    sc.escaped = old_escaped + escaped;
    let h2 = sc.heights();
    assert_eq!(sc.ledger(&h2, q, area), 0, "the next domain's ledger");
}

#[test]
fn closing_one_area_by_force_lets_its_parcels_fly_on() {
    let mut sc = Scene::new();
    let (p1, p2) = (Vec3::new(-0.9, 0.0, 0.0), Vec3::new(0.9, 0.0, 0.0));
    let a = sc.impact(p1);
    let b = sc.impact(p2);
    throw(&mut sc, p1);
    assert_ne!(a.area, b.area);
    until_airborne(&mut sc, 30);
    sc.frames(2);
    let (n, _) = in_flight(&sc);
    assert!(n > 0);
    // The first area closes mid-flight; the second steps on and flies both
    // areas' parcels until they land (or come back into it).
    sc.s.terrain_bake_area(a.area).unwrap();
    sc.frames(1);
    sc.land(240);
    sc.frames(SETTLE_FRAMES);
    sc.land(240);
    sc.s.terrain_bake_area(b.area).unwrap();
    let (escaped, airborne) = counters(&sc);
    assert_eq!(airborne, 0);
    sc.escaped = escaped;
    let (q, area) = sc.quantum();
    let h = sc.heights();
    assert_eq!(sc.ledger(&h, q, area), 0, "one area closed by force");
    eprintln!("one area closed with {n} parcels in flight: ledger exact");
}
