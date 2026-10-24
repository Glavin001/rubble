//! Axisymmetric granular column collapse (`engine/docs/deformable-terrain.md`,
//! "Validation targets: Granular") against published laboratory scaling.
//!
//! A cylinder of dry sand (radius r0, height h0, aspect ratio a = h0/r0) is
//! released at rest on a flat bed and spreads until it stops. Laboratory
//! experiments find that the deposit's runout (r∞ − r0)/r0 and final height
//! h∞/r0 depend on a alone, independent of the column's size and, to first
//! order, of the grains (dimensional analysis leaves a as the only parameter
//! once the column is many grains wide).
//!
//! # Published relations (cohesionless grains, axisymmetric)
//!
//! Runout (r∞ − r0)/r0:
//! - Lube, Huppert, Sparks & Hallworth (2004), J. Fluid Mech. 508, 175–199,
//!   abstract: r∞ = r0(1 + 1.24a) for a < 1.7, r∞ = r0(1 + 1.6a^½) for a > 1.7.
//!   The boundary a = 1.7 is the aspect ratio of the central undisturbed cone
//!   (≈59°). Emplacement time t∞ = 3(h0/g)^½ (same abstract).
//! - Lajeunesse, Mangeney-Castelnau & Vilotte (2004), Phys. Fluids 16, 2371:
//!   linear in a below a ≈ 3, ∝ a^½ above (regime boundary a = 3, as restated
//!   by Sharma et al. 2024, §IV.2; their prefactors were not accessible here and
//!   are not used).
//! - Warnett, Denissenko, Thomas, Kiraci & Williams (2014), Granular Matter 16,
//!   115–124, §3.2 and Table 2: for a ≳ 1.7, r* = P2·a^0.66 with P2 = 1.33–1.34
//!   for r0/d ≥ 50 (P2 saturates with system size; a continuum is the large
//!   r0/d limit). Their linear-regime fit r* = P1·a + Q has an intercept Q not
//!   given in the text, so it is not used.
//! - Sharma et al. (2024), "The effects of interparticle cohesion on the
//!   collapse of granular columns", arXiv:2402.07285, §IV.2 and Table II, Co = 0
//!   (cohesionless): ΔR∞/R0 = (1.34 ± 0.08)·a for a ≲ 3 and (2.05 ± 0.18)·a^½
//!   for a ≳ 3.
//!
//! Final height h∞/r0:
//! - Below a critical aspect ratio the summit is never reached by the
//!   avalanche and the deposit is a truncated cone with h∞ = h0, i.e.
//!   h∞/r0 = a. The critical value is 0.74 (Lajeunesse et al. 2004 and Lube et
//!   al. 2004, as cited by Warnett et al. 2014 §3.1), 0.71 (Sharma et al. 2024
//!   §IV.2) and 0.90 (Warnett et al. 2014 §3.1).
//! - Above it: Lube et al. (2004) h∞/r0 ≈ 0.88·a^(1/6) (the prefactor as restated
//!   by secondary sources; the exponent ≈ 0.17 is confirmed by Warnett et al.
//!   2014 §3.3); Lajeunesse et al. (2004) a plateau of 0.74 (Sharma et al. 2024
//!   §IV.2); Warnett et al. (2014) Eq. (10) h∞/r0 = 0.90·a^0.09; Sharma et al.
//!   (2024) a plateau of 0.66 ± 0.03.
//!
//! # Scatter, and the band asserted
//!
//! The laboratories disagree with each other by more than their measurement
//! errors (Warnett et al. 2014 report r∞ to ±1.6–2.3 mm and h∞ to ±0.5 mm on
//! r0 = 20–75 mm columns, yet their prefactors and exponents differ from Lube's
//! and Lajeunesse's; they attribute this to grain material and r0/d). The
//! published spread is therefore taken as the envelope of every fit above that
//! is valid at the case's a (using each author's own regime boundary and ±1σ
//! where given). To it is added a discretisation allowance: the deposit edge is
//! resolved to one ring of width dx and its thin toe to about one more cell, so
//! ±2·dx on r∞; the summit to ±1·dx on h∞. With r0 = 16·dx that is ±0.125 on
//! the runout and ±0.0625 on h∞/r0.
//!
//! # Setup
//!
//! SandMaterial::default() (dry quartz sand: ρ = 1600 kg/m³, E = 50 MPa,
//! ν = 0.3, φ = 30°, no cohesion), bed Coulomb friction tan φ. r0 = 0.8 m and
//! dx = r0/16 = 0.05 m (2³ particles per cell). The result should not depend on
//! the column's size: the only other dimensionless group, ρ·g·h0/E (elastic
//! strain under the column's weight), is below 1e-3 for every case, as it is in
//! the laboratory, so the material is effectively rigid-plastic.
//!
//! Measured quantities:
//! - r∞, the deposit edge: the outermost ring (width dx, contiguous from the
//!   axis) whose azimuthally averaged thickness, at the initial packing, is at
//!   least a quarter of a particle layer (h/4 = dx/8 = 0.008·r0). That is the
//!   laboratory's definition (the edge of the continuous deposit, measured to a
//!   few grain diameters; Lube's and Warnett's grains are 0.3–1.5 mm on r0 of
//!   20–150 mm, so a few d is 0.01–0.1·r0). A mass quantile (e.g. the radius
//!   holding 99% of the particles) is biased inward on a conical deposit, whose
//!   outer 1% of mass lies beyond 0.94·r∞; it is printed for comparison only.
//! - h∞, the top of the deposit on the axis: the highest particle within one
//!   cell of the axis, plus half a particle spacing (particles are the centres
//!   of h-sized cubes), above the bed.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::layout::MpmParticle;
use engine_terrain::{substeps_for, RegionPlan, SandMaterial, TerrainConfig, BLOCK_CELLS};
use engine_webgpu::WgpuSim;
use glam::{IVec3, Quat, Vec3};

const G: f32 = 9.81;
/// Initial column radius (m).
const R0: f32 = 0.8;
/// Cells across r0: the region's `cells_per_radius`.
const CELLS_PER_R0: f32 = 16.0;
/// The region (and the bed) reach this many r0 from the axis: past every
/// published runout of the cases (at most 5.4·r0 with the allowance at a = 3).
const REACH_R0: f32 = 6.0;
const FRAME: f32 = 1.0 / 60.0;

fn sim() -> WgpuSim {
    WgpuSim::new(WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu adapter")
}

/// The fits of the module docs valid at aspect ratio `a`, as (source, runout
/// (r∞ − r0)/r0 low, high).
fn published_runout(a: f32) -> Vec<(&'static str, f32, f32)> {
    let mut out = Vec::new();
    let lube = if a < 1.7 { 1.24 * a } else { 1.6 * a.sqrt() };
    out.push(("Lube et al. 2004", lube, lube));
    if a >= 1.7 {
        out.push((
            "Warnett et al. 2014",
            1.33 * a.powf(0.66),
            1.34 * a.powf(0.66),
        ));
    }
    if a <= 3.0 {
        out.push(("Sharma et al. 2024 (a<=3)", 1.26 * a, 1.42 * a));
    }
    if a >= 3.0 {
        out.push((
            "Sharma et al. 2024 (a>=3)",
            1.87 * a.sqrt(),
            2.23 * a.sqrt(),
        ));
    }
    out
}

/// The fits of the module docs valid at `a`, as (source, h∞/r0 low, high).
fn published_height(a: f32) -> Vec<(&'static str, f32, f32)> {
    // Below every author's critical aspect ratio (0.71 is the smallest) the
    // summit survives: h∞ = h0. Between 0.71 and 0.90 the authors disagree on
    // the regime; the cases avoid that interval.
    if a < 0.71 {
        return vec![("all (summit untouched)", a, a)];
    }
    assert!(
        a > 0.90,
        "a = {a} lies between the published critical ratios"
    );
    let lube = 0.88 * a.powf(1.0 / 6.0);
    let warnett = 0.90 * a.powf(0.09);
    vec![
        ("Lube et al. 2004", lube, lube),
        ("Lajeunesse et al. 2004", 0.74, 0.74),
        ("Warnett et al. 2014", warnett, warnett),
        ("Sharma et al. 2024", 0.63, 0.69),
    ]
}

fn envelope(fits: &[(&'static str, f32, f32)]) -> (f32, f32) {
    fits.iter().fold((f32::MAX, f32::MIN), |(lo, hi), f| {
        (lo.min(f.1), hi.max(f.2))
    })
}

struct Deposit {
    /// Deposit edge (m from the axis).
    edge: f32,
    /// Edge per quadrant (+x+z, −x+z, −x−z, +x−z), for axisymmetry.
    quadrant_edge: [f32; 4],
    /// Radius holding 99% and 99.9% of the particles, and the farthest one.
    r99: f32,
    r999: f32,
    r_max: f32,
    /// Top of the deposit on the axis, above the bed (m).
    height: f32,
    /// Σ V0·det F / Σ V0.
    volume_ratio: f64,
}

/// Measure the deposit (module docs, "Measured quantities"). `h` is the
/// particle spacing, `dx` the cell size, the bed at y = 0.
fn measure(ps: &[MpmParticle], h: f32, dx: f32) -> Deposit {
    let rings = (REACH_R0 * R0 / dx).ceil() as usize + 2;
    let mut vol = vec![0.0f64; rings];
    let mut qvol = vec![[0.0f64; 4]; rings];
    let mut radii = Vec::with_capacity(ps.len());
    let mut height = 0.0f32;
    let (mut v0, mut v) = (0.0f64, 0.0f64);
    for p in ps {
        let (x, y, z) = (p.xm[0], p.xm[1], p.xm[2]);
        let r = (x * x + z * z).sqrt();
        radii.push(r);
        let k = ((r / dx) as usize).min(rings - 1);
        let rest = f64::from(p.f[0][3]);
        vol[k] += rest;
        let q = match (x >= 0.0, z >= 0.0) {
            (true, true) => 0,
            (false, true) => 1,
            (false, false) => 2,
            (true, false) => 3,
        };
        qvol[k][q] += rest;
        if r < dx {
            height = height.max(y + h / 2.0);
        }
        let f = glam::Mat3::from_cols_array(&[
            p.f[0][0], p.f[1][0], p.f[2][0], p.f[0][1], p.f[1][1], p.f[2][1], p.f[0][2], p.f[1][2],
            p.f[2][2],
        ]);
        v0 += rest;
        v += rest * f64::from(f.determinant());
    }
    // Thickness of ring k: its volume over its area π((k+1)² − k²)dx².
    let threshold = f64::from(h / 4.0);
    let area = |k: usize, share: f64| {
        std::f64::consts::PI * f64::from(dx * dx) * ((2 * k + 1) as f64) * share
    };
    let edge_of = |vols: &dyn Fn(usize) -> f64, share: f64| {
        let mut k = 0;
        while k < rings && vols(k) / area(k, share) >= threshold {
            k += 1;
        }
        k as f32 * dx
    };
    let edge = edge_of(&|k| vol[k], 1.0);
    let mut quadrant_edge = [0.0; 4];
    for (q, e) in quadrant_edge.iter_mut().enumerate() {
        *e = edge_of(&|k| qvol[k][q], 0.25);
    }
    radii.sort_by(|a, b| a.total_cmp(b));
    let at = |f: f64| radii[((radii.len() as f64 * f) as usize).min(radii.len() - 1)];
    Deposit {
        edge,
        quadrant_edge,
        r99: at(0.99),
        r999: at(0.999),
        r_max: *radii.last().unwrap(),
        height,
        volume_ratio: v / v0,
    }
}

struct Outcome {
    r0: f32,
    dx: f32,
    particles: usize,
    deposit: Deposit,
    stopped_at: f32,
    max_speed: f32,
    wall: f32,
}

/// Collapse a column of aspect ratio `a` and measure its deposit once it has
/// stopped.
fn collapse(a: f32) -> Outcome {
    let mut s = sim();
    let sand = SandMaterial::default();
    let mut cfg = TerrainConfig {
        cells_per_radius: CELLS_PER_R0,
        ..TerrainConfig::default()
    };
    // dx = r0 / cells_per_radius, as a region planned around the column would
    // have. `around` boxes a cube of the padded radius; this collapse needs a
    // flat slab REACH_R0·r0 wide and only the column's height tall, so the box
    // is cut down to that (same dx, same block lattice).
    let dx = RegionPlan::around(&cfg, Vec3::ZERO, R0, 0.0, 0, 0)
        .unwrap()
        .dx;
    let block = dx * BLOCK_CELLS as f32;
    let h0 = a * R0;
    let half = (REACH_R0 * R0 / block).ceil() as i32;
    let below = 2; // blocks under the bed
    let above = (h0 / block).ceil() as i32 + 2;
    let plan = RegionPlan {
        dx,
        origin: Vec3::new(-half as f32, -below as f32, -half as f32) * block,
        blocks: IVec3::new(2 * half, below + above, 2 * half),
    };
    // Every block the material can touch (the deposit and the column) fits;
    // the box itself is only addressed through the hash.
    cfg.max_blocks = 1 << 14;
    s.terrain_enable(&cfg).unwrap();
    s.terrain_set_material(Some(&sand)).unwrap();

    // A flat bed at y = 0 under the whole region.
    let spacing = 0.15;
    let n = ((2.0 * REACH_R0 * R0 + 2.0) / spacing).ceil() as u32 + 1;
    let field = HeightField::new(n, n, spacing, spacing, vec![0.0; (n * n) as usize]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();

    // The column: particles at spacing h = dx/2 whose columns lie inside r0.
    let h = dx / 2.0;
    let nr = (R0 / h).ceil() as i32;
    let layers = (h0 / h).round() as i32;
    let mut ps = Vec::new();
    let mut footprint = 0usize;
    for k in -nr..nr {
        for i in -nr..nr {
            let (x, z) = ((i as f32 + 0.5) * h, (k as f32 + 0.5) * h);
            if x * x + z * z > R0 * R0 {
                continue;
            }
            footprint += 1;
            for j in 0..layers {
                let y = (j as f32 + 0.5) * h;
                ps.push(MpmParticle::at_rest([x, y, z], h * h * h, sand.density));
            }
        }
    }
    // The column as discretised: r0 from its footprint area, h0 from its layers.
    let r0 = (footprint as f32 * h * h / std::f32::consts::PI).sqrt();
    s.terrain_load(&plan, &ps, Some(bed), sand.friction_angle.tan())
        .unwrap();

    // The flow cannot outrun free fall from the column's top; allow 1.5×.
    let flow_bound = 1.5 * (2.0 * G * h0).sqrt();
    let (substeps, dt) = substeps_for(&cfg, FRAME, dx, sand.p_wave_speed(), flow_bound).unwrap();
    let t_inf = 3.0 * (h0 / G).sqrt();
    // At rest: every particle slower than 1% of the column's free-fall speed
    // scale √(g·h0), and the deposit's edge and summit unchanged (to a tenth
    // of a cell) over the last 0.5 s.
    let rest_speed = 0.01 * (G * h0).sqrt();
    let cap = 6.0 * t_inf + 1.0;
    let check_every = 15; // frames: 0.25 s
    let t0 = std::time::Instant::now();
    let mut history: Vec<(f32, f32)> = Vec::new();
    let mut frame = 0u32;
    let (mut t, mut speed);
    loop {
        speed = s.terrain_step(substeps, dt, -G).unwrap().max_speed;
        frame += 1;
        t = frame as f32 * FRAME;
        if frame % check_every != 0 {
            continue;
        }
        let d = measure(&s.terrain_particles(), h, dx);
        history.push((d.edge, d.height));
        eprintln!(
            "  a = {a}: t = {t:.2} s, max speed {speed:.3} m/s, edge {:.3} m, summit {:.3} m",
            d.edge, d.height
        );
        let settled = history.len() >= 3 && {
            let w = &history[history.len() - 3..];
            w.iter()
                .all(|e| (e.0 - w[2].0).abs() <= 0.1 * dx && (e.1 - w[2].1).abs() <= 0.1 * dx)
        };
        if (t > t_inf && speed < rest_speed && settled) || t >= cap {
            break;
        }
    }
    let wall = t0.elapsed().as_secs_f32();
    assert!(
        speed < rest_speed,
        "a = {a}: still moving at {speed} m/s after {t} s (rest is < {rest_speed} m/s)"
    );
    let out = s.terrain_particles();
    Outcome {
        r0,
        dx,
        particles: out.len(),
        deposit: measure(&out, h, dx),
        stopped_at: t,
        max_speed: speed,
        wall,
    }
}

/// Runout and final height of collapsed sand columns at aspect ratios 0.5 and
/// 1 (Lube's linear regime) and 3 (Lube's a^½ regime), against the envelope of
/// the published laboratory fits plus a discretisation allowance (module docs).
///
/// Ignored: the solver disagrees with the published runout (measured
/// 2026-10-24 at d535321d, r0 = 16·dx, dx = 0.05 m; 187 s for the three cases).
///
/// | a   | runout measured | band (fits)              | h∞/r0 measured | band (fits)          |
/// |-----|-----------------|--------------------------|----------------|----------------------|
/// | 0.5 | 0.373           | 0.495–0.835 (0.62–0.71)  | 0.499          | 0.438–0.562 (0.50)   |
/// | 1   | 0.747           | 1.115–1.545 (1.24–1.42)  | 0.744          | 0.567–0.962 (0.63–0.90) |
/// | 3   | 2.244           | 2.621–4.385 (2.75–4.26)  | 0.746          | 0.567–1.119 (0.63–1.06) |
///
/// The final heights agree (and match Lajeunesse's plateau of 0.74 above the
/// critical ratio). The runouts are short: about 0.75·a below a = 1.7 against
/// Lube's 1.24·a (60% of it), and 1.30·a^½ at a = 3 against 1.6·a^½ (81%).
/// The deposit edge is not a measurement artefact: the farthest particle lies
/// within a cell of it (r_max 1.075, 1.380, 2.619 m against edges 1.10, 1.40,
/// 2.60 m). Refining a = 1 to r0 = 24·dx moved its runout only from 0.747 to
/// 0.792. The material and solver are not changed to fit: the gap is left for
/// the solver's owners to explain. At a = 0.5 the deposit is what a
/// quasi-static pile would leave: a truncated cone of the column's volume with
/// 30° flanks runs out 0.40·r0, against 0.373 measured; laboratory deposits
/// spread further, to flanks well below the angle of repose.
#[test]
#[ignore = "runout ~60-80% of the published fits; see the doc comment"]
fn granular_column_collapse_follows_published_scaling() {
    let runout_allowance = 2.0 / CELLS_PER_R0;
    let height_allowance = 1.0 / CELLS_PER_R0;
    let mut failures = Vec::new();
    for a in [0.5f32, 1.0, 3.0] {
        let o = collapse(a);
        let d = &o.deposit;
        let runout = (d.edge - o.r0) / o.r0;
        let height = d.height / o.r0;
        let rf = published_runout(a);
        let hf = published_height(a);
        let (rlo, rhi) = envelope(&rf);
        let (hlo, hhi) = envelope(&hf);
        let (rlo, rhi) = (rlo - runout_allowance, rhi + runout_allowance);
        let (hlo, hhi) = (hlo - height_allowance, hhi + height_allowance);
        eprintln!(
            "a = {a}: r0 = {:.4} m ({} cells), {} particles, stopped at {:.2} s \
             (Lube t∞ = {:.2} s), max speed {:.4} m/s, {:.1} s wall",
            o.r0,
            o.r0 / o.dx,
            o.particles,
            o.stopped_at,
            3.0 * (a * R0 / G).sqrt(),
            o.max_speed,
            o.wall
        );
        eprintln!("  runout (r∞−r0)/r0 = {runout:.3}  [band {rlo:.3}, {rhi:.3}]  fits: {rf:?}");
        eprintln!("  height h∞/r0 = {height:.3}  [band {hlo:.3}, {hhi:.3}]  fits: {hf:?}");
        eprintln!(
            "  edge {:.3} m, quadrants {:?}, r99 {:.3} m, r99.9 {:.3} m, r_max {:.3} m, \
             volume ratio {:.4}",
            d.edge, d.quadrant_edge, d.r99, d.r999, d.r_max, d.volume_ratio
        );
        if !(rlo..=rhi).contains(&runout) {
            failures.push(format!(
                "a = {a}: runout {runout:.3} outside [{rlo:.3}, {rhi:.3}]"
            ));
        }
        if !(hlo..=hhi).contains(&height) {
            failures.push(format!(
                "a = {a}: final height {height:.3} r0 outside [{hlo:.3}, {hhi:.3}]"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
