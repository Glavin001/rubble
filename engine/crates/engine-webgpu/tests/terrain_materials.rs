//! T5 materials in the MPM pipeline (`engine/docs/deformable-terrain.md`, "T5
//! materials"), against laws from soil and fracture mechanics:
//!
//! - **Cohesive soil.** An unsupported vertical cut in a Mohr–Coulomb soil of
//!   cohesion c, friction angle φ and unit weight γ stands up to the critical
//!   height H_c = (4c/γ)·tan(45° + φ/2) (Terzaghi 1943, from Rankine's active
//!   state; the upper bound of limit analysis is 3.83c/γ·tan(45° + φ/2)). A
//!   square tower of soil (1.2 times as wide as tall, so the failure wedges of
//!   opposite faces do not meet) on a bed of friction tan φ, with gravity
//!   ramped on over a second (so no dynamic overshoot), stands at 0.7·H_c and
//!   collapses at 1.5·H_c. A sweep (2026-10-24, dx = H_c/16) put the solver's
//!   critical height between 1.0·H_c (stands, the top 2 cm down) and 1.1·H_c
//!   (collapses).
//! - **Damage-softening rock.** A bar pulled slowly by rigid grips breaks
//!   beside them. Each crack opens across two cells (the crack band,
//!   `CRACK_BAND_CELLS`) at either cell size, and the energy the damage
//!   dissipates per crack area is the fracture energy G_f (Bažant & Oh 1983:
//!   the softening is scaled by the band so it does not depend on the cell
//!   size). Measured from each particle's damage through the softening law
//!   (the energy a particle of that damage has dissipated under uniaxial
//!   tension).
//!
//! Dynamic loading was tried first and is not a G_f test: a bar stretched by
//! an initial velocity field breaks into one, two or more pieces depending on
//! the rate and the cell size (dynamic fragmentation), with damage spread over
//! the overstressed zone before it localises.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::layout::MpmParticle;
use engine_terrain::material::CRACK_BAND_CELLS;
use engine_terrain::TerrainError;
use engine_terrain::{
    substeps_for, RegionPlan, RockMaterial, SandMaterial, TerrainConfig, YieldSurface,
};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

const G: f32 = 9.81;
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

/// A stiff soil (E = 50 MPa, ν = 0.3, ρ = 1800 kg/m³, φ = 30°) whose vertical
/// cuts stand to `critical_height` m: c = H_c·γ / (4·tan(45° + φ/2)).
fn soil(critical_height: f32) -> SandMaterial {
    let phi = 30f32.to_radians();
    let density = 1800.0;
    let gamma = density * G;
    SandMaterial {
        density,
        youngs_modulus: 50.0e6,
        poisson_ratio: 0.3,
        friction_angle: phi,
        cohesion: critical_height * gamma / (4.0 * (std::f32::consts::FRAC_PI_4 + phi / 2.0).tan()),
        yield_surface: YieldSurface::MohrCoulomb,
    }
}

struct Cut {
    /// Tower height and the top of the material at the end (m above the bed).
    height: f32,
    top: f32,
    /// Share of the particles that moved more than one cell.
    moved: f32,
    /// Largest particle speed over the last frame (m/s).
    speed: f32,
    particles: usize,
}

/// A square tower of `soil`, `height` m tall and 1.2·height wide, on a flat bed
/// of friction tan φ, with gravity ramped from 0 to g over `ramp` s and then
/// held to `end` s. dx = H_c/16.
fn tower(soil: &SandMaterial, critical_height: f32, height: f32) -> Cut {
    let (ramp, end) = (1.0f32, 4.0f32);
    let mut s = sim();
    let dx = critical_height / 16.0;
    let h = dx / 2.0;
    let layers = (height / h).round() as i32;
    let height = layers as f32 * h;
    let across = (1.2 * height / h).round() as i32;
    let width = across as f32 * h;
    let reach = width / 2.0 + 2.0 * height;
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        max_blocks: 1 << 14,
        max_parcels: 0,
        ..TerrainConfig::default()
    };
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-reach, -8.0 * dx, -reach),
        Vec3::new(reach, height + 8.0 * dx, reach),
        dx,
        0,
        0,
    )
    .unwrap();
    s.terrain_enable(&cfg).unwrap();
    s.terrain_set_material(Some(soil)).unwrap();
    let spacing = 0.5;
    let n = ((2.0 * reach + 4.0) / spacing).ceil() as u32 + 1;
    let field = HeightField::new(n, n, spacing, spacing, vec![0.0; (n * n) as usize]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let mut ps = Vec::new();
    for k in 0..across {
        for j in 0..layers {
            for i in 0..across {
                let x = Vec3::new(-width / 2.0, 0.0, -width / 2.0)
                    + (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * h;
                ps.push(MpmParticle::at_rest(x.to_array(), h * h * h, soil.density));
            }
        }
    }
    let start: Vec<Vec3> = ps.iter().map(|p| Vec3::from_slice(&p.xm[..3])).collect();
    s.terrain_load(&plan, &ps, Some(bed), soil.friction_angle.tan())
        .unwrap();
    let flow_bound = 1.5 * (2.0 * G * height).sqrt();
    let (substeps, dt) = substeps_for(&cfg, FRAME, dx, soil.p_wave_speed(), flow_bound).unwrap();
    let frames = (end / FRAME).round() as u32;
    let mut speed = 0.0;
    for f in 0..frames {
        let t = (f as f32 + 0.5) * FRAME;
        let g = -G * (t / ramp).min(1.0);
        speed = s.terrain_step(substeps, dt, g).unwrap().max_speed;
    }
    let out = s.terrain_particles();
    let top = out.iter().map(|p| p.xm[1]).fold(0.0f32, f32::max) + h / 2.0;
    let moved = out
        .iter()
        .zip(&start)
        .filter(|(p, x0)| (Vec3::from_slice(&p.xm[..3]) - **x0).length() > dx)
        .count();
    Cut {
        height,
        top,
        moved: moved as f32 / out.len() as f32,
        speed,
        particles: out.len(),
    }
}

/// Terzaghi's critical height of a vertical cut: a tower of cohesive soil
/// stands at 0.7·H_c and collapses at 1.5·H_c.
#[test]
fn a_cohesive_cut_stands_below_its_critical_height_and_fails_above() {
    let hc = 4.0;
    let soil = soil(hc);
    eprintln!(
        "soil: c = {:.0} Pa, φ = 30°, γ = {:.0} N/m³, H_c = {hc} m",
        soil.cohesion,
        soil.density * G
    );
    let t0 = std::time::Instant::now();
    let low = tower(&soil, hc, 0.7 * hc);
    eprintln!(
        "  0.7·H_c: {} particles, height {:.3} m, top {:.3} m, {:.2}% moved > dx, \
         max speed {:.4} m/s ({:.1} s wall)",
        low.particles,
        low.height,
        low.top,
        100.0 * low.moved,
        low.speed,
        t0.elapsed().as_secs_f32()
    );
    let t0 = std::time::Instant::now();
    let high = tower(&soil, hc, 1.5 * hc);
    eprintln!(
        "  1.5·H_c: {} particles, height {:.3} m, top {:.3} m, {:.2}% moved > dx, \
         max speed {:.4} m/s ({:.1} s wall)",
        high.particles,
        high.height,
        high.top,
        100.0 * high.moved,
        high.speed,
        t0.elapsed().as_secs_f32()
    );
    let dx = hc / 16.0;
    assert!(
        low.moved < 0.01 && low.top > low.height - 0.5 * dx,
        "the cut at 0.7·H_c failed: top {} of {} m, {:.2}% moved",
        low.top,
        low.height,
        100.0 * low.moved
    );
    assert!(
        high.moved > 0.1 && high.top < 0.9 * high.height,
        "the cut at 1.5·H_c stood: top {} of {} m, {:.2}% moved",
        high.top,
        high.height,
        100.0 * high.moved
    );
}

/// The rock of the bar test: E = 10 GPa, ν = 0.25, ρ = 2500 kg/m³,
/// f_t = 2 MPa, G_f = 50 J/m² (l_ch = E·G_f/f_t² = 0.125 m, the largest cell
/// its two-cell crack band allows).
fn rock() -> RockMaterial {
    RockMaterial {
        density: 2500.0,
        youngs_modulus: 10.0e9,
        poisson_ratio: 0.25,
        tensile_strength: 2.0e6,
        fracture_energy: 50.0,
    }
}

/// Energy (J/m³ of rest volume) the exponential softening law has dissipated
/// at damage `d` along uniaxial loading: with κ the largest effective
/// stress reached (1 − d = (f_t/κ)·exp(−(κ − f_t)/σ_s)) and X = exp(−(κ − f_t)/σ_s),
/// D = [f_t²/2 + f_t·σ_s·(1 − X) − f_t·κ·X/2] / E, which tends to G_f/dx.
fn dissipated(m: &RockMaterial, dx: f32, d: f32) -> f64 {
    let (ft, e) = (f64::from(m.tensile_strength), f64::from(m.youngs_modulus));
    let ss = f64::from(m.softening_stress(dx).unwrap());
    let full = f64::from(m.fracture_energy) / f64::from(CRACK_BAND_CELLS * dx);
    let d = f64::from(d);
    if d <= 0.0 {
        return 0.0;
    }
    if d >= 1.0 - 1e-7 {
        return full;
    }
    // 1 − d is decreasing in κ: bisect on [f_t, f_t + 200·σ_s].
    let resid = |k: f64| (ft / k) * (-(k - ft) / ss).exp() - (1.0 - d);
    let (mut lo, mut hi) = (ft, ft + 200.0 * ss);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if resid(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let k = 0.5 * (lo + hi);
    let x = (-(k - ft) / ss).exp();
    (ft * ft / 2.0 + ft * ss * (1.0 - x) - ft * k * x / 2.0) / e
}

struct Bar {
    /// Cracks: runs of particle layers across the bar whose mean damage
    /// passes 0.5.
    cracks: usize,
    /// Dissipated energy over the crack area (cracks × section, J/m²).
    energy_per_area: f64,
    /// Width of each crack: the layers' volume with d > 0.5 over the crack
    /// area, in cells.
    band_cells: f64,
    /// Mean damage of each layer past 1% (layer index from the centre).
    profile: Vec<(i64, f64)>,
    /// Particles with any damage, and fully broken (d > 0.99).
    damaged: usize,
    broken: usize,
    particles: usize,
}

/// A bar of `rock` along x, 1.2 m long with a 0.24 × 0.24 m section, in cells
/// of `dx` (2³ particles each), pulled slowly by grips: the last four cells at
/// each end are 10⁶ times denser, so they keep their velocity ±v (a
/// displacement control). The gauge starts with the matching linear velocity,
/// so it stretches from rest without a shock, and v brings it to the tensile
/// strain f_t/E in ten P-wave transits (quasi-static). No gravity. Runs to
/// twice that time; the rigid grips restrain the gauge's Poisson contraction,
/// so it breaks beside them.
fn bar(rock: &RockMaterial, dx: f32) -> Bar {
    const GRIP_CELLS: f32 = 4.0;
    const GRIP_DENSITY: f32 = 1.0e6;
    let mut s = sim();
    let (length, side) = (1.2f32, 0.24f32);
    let h = dx / 2.0;
    let (nl, ns) = ((length / h).round() as i32, (side / h).round() as i32);
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        max_blocks: 1 << 14,
        max_parcels: 0,
        max_substeps: 4096,
        ..TerrainConfig::default()
    };
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-length, -side, -side),
        Vec3::new(length, 2.0 * side, 2.0 * side),
        dx,
        0,
        0,
    )
    .unwrap();
    s.terrain_enable(&cfg).unwrap();
    // `None` (no stress) is still accepted as it always was; rock replaces it.
    s.terrain_set_material(None).unwrap();
    s.terrain_set_material(rock).unwrap();
    let half_gauge = length / 2.0 - GRIP_CELLS * dx;
    let wave = rock.p_wave_speed();
    let rise = 10.0 * 2.0 * half_gauge / wave;
    let v = rock.tensile_strength / rock.youngs_modulus * half_gauge / rise;
    let rate = v / half_gauge;
    let mut ps = Vec::new();
    for k in 0..ns {
        for j in 0..ns {
            for i in 0..nl {
                let x = Vec3::new(-length / 2.0, 0.0, 0.0)
                    + (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * h;
                let grip = x.x.abs() > half_gauge;
                let density = if grip { GRIP_DENSITY } else { 1.0 } * rock.density;
                let mut p = MpmParticle::at_rest(x.to_array(), h * h * h, density);
                if grip {
                    p.vj[0] = v * x.x.signum();
                } else {
                    p.vj[0] = rate * x.x;
                    p.c[0][0] = rate;
                }
                ps.push(p);
            }
        }
    }
    s.terrain_load(&plan, &ps, None, 0.0).unwrap();
    let dt = cfg.courant * dx / (wave + 4.0 * v) * 0.999;
    let steps = (2.0 * rise / dt).ceil() as u32;
    let mut done = 0;
    while done < steps {
        let n = (steps - done).min(512);
        s.terrain_step(n, dt, 0.0).unwrap();
        done += n;
    }
    let out = s.terrain_particles();
    // Mean damage per layer of particles across the bar (they move by well
    // under a layer).
    let mut layers = vec![(0.0f64, 0usize); nl as usize];
    for p in &out {
        let k = (((p.xm[0] + length / 2.0) / h).floor().max(0.0) as usize).min(layers.len() - 1);
        layers[k].0 += f64::from(p.damage());
        layers[k].1 += 1;
    }
    let mean: Vec<f64> = layers.iter().map(|(d, n)| d / (*n).max(1) as f64).collect();
    let cracks = (0..mean.len())
        .filter(|&k| mean[k] > 0.5 && (k == 0 || mean[k - 1] <= 0.5))
        .count();
    let profile = mean
        .iter()
        .enumerate()
        .filter(|(_, d)| **d > 0.01)
        .map(|(k, d)| (k as i64 - i64::from(nl / 2), *d))
        .collect();
    let mut energy = 0.0f64;
    let (mut damaged, mut broken, mut band_volume) = (0, 0, 0.0f64);
    for p in &out {
        let d = p.damage();
        let vol = f64::from(p.f[0][3]);
        energy += vol * dissipated(rock, dx, d);
        damaged += usize::from(d > 0.0);
        broken += usize::from(d > 0.99);
        if d > 0.5 {
            band_volume += vol;
        }
    }
    let area = f64::from(side * side) * cracks.max(1) as f64;
    Bar {
        cracks,
        energy_per_area: energy / area,
        band_cells: band_volume / area / f64::from(dx),
        profile,
        damaged,
        broken,
        particles: out.len(),
    }
}

/// The crack band: in a slowly pulled bar each crack opens across two cells
/// at either cell size, and the energy the damage dissipates per crack area is
/// the fracture energy G_f.
#[test]
fn a_rock_bar_breaks_with_its_fracture_energy_at_two_cell_sizes() {
    let rock = rock();
    let gf = f64::from(rock.fracture_energy);
    let mut per_area = Vec::new();
    for dx in [0.04f32, 0.02] {
        let t0 = std::time::Instant::now();
        let b = bar(&rock, dx);
        eprintln!(
            "  dx = {dx} m: {} particles, {} damaged, {} broken; {} crack(s), {:.2} cells wide; \
             dissipated {:.2} J/m² of crack (G_f {gf}) ({:.1} s wall)",
            b.particles,
            b.damaged,
            b.broken,
            b.cracks,
            b.band_cells,
            b.energy_per_area,
            t0.elapsed().as_secs_f32()
        );
        let line: Vec<String> = b
            .profile
            .iter()
            .map(|(k, d)| format!("{k}:{d:.2}"))
            .collect();
        eprintln!(
            "    mean damage by layer from the centre: {}",
            line.join(" ")
        );
        assert!(b.cracks > 0, "dx = {dx}: the bar did not break");
        assert!(
            (b.band_cells - f64::from(CRACK_BAND_CELLS)).abs() < 0.5,
            "dx = {dx}: cracks {} cells wide, the crack band is {CRACK_BAND_CELLS}",
            b.band_cells
        );
        per_area.push(b.energy_per_area);
    }
    for (dx, g) in [0.04, 0.02].iter().zip(&per_area) {
        assert!(
            (g / gf - 1.0).abs() < 0.15,
            "dx = {dx}: {g:.2} J/m² dissipated per crack area, G_f = {gf}"
        );
    }
    assert!(
        (per_area[0] / per_area[1] - 1.0).abs() < 0.05,
        "energy per crack area depends on the cell size: {per_area:?}"
    );
}

/// A region whose cells are too coarse for the rock's crack band fails its
/// step by name instead of weakening the rock.
#[test]
fn rock_in_cells_past_its_crack_band_fails_the_step() {
    let rock = RockMaterial::default();
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 12,
        max_blocks: 1 << 10,
        ..TerrainConfig::default()
    };
    s.terrain_enable(&cfg).unwrap();
    s.terrain_set_material(rock).unwrap();
    let dx = 2.0 * rock.max_cell_size();
    let plan = RegionPlan::spanning(&cfg, Vec3::splat(-1.0), Vec3::splat(1.0), dx, 0, 0).unwrap();
    let ps = [MpmParticle::at_rest([0.0; 3], 1e-3, rock.density)];
    s.terrain_load(&plan, &ps, None, 0.0).unwrap();
    let err = s.terrain_step(1, 1e-6, 0.0).unwrap_err();
    assert!(
        matches!(
            err.downcast_ref::<TerrainError>(),
            Some(TerrainError::CrackBand { .. })
        ),
        "{err}"
    );
    // Below the limit the same region steps.
    let fine = RegionPlan::spanning(&cfg, Vec3::splat(-1.0), Vec3::splat(1.0), 0.1, 0, 0).unwrap();
    s.terrain_load(&fine, &ps, None, 0.0).unwrap();
    s.terrain_step(1, 1e-6, 0.0).unwrap();
}
