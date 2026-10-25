//! The absorbing ground around an activated region (`RegionBoundary::Absorbing`,
//! `slang/kernels/terrain_mpm.slang` `mpmAbsorbNode`): Lysmer–Kuhlemeyer
//! dashpots and viscous-spring springs on the region's bed and side walls,
//! carrying the geostatic traction the material starts with. Checked against
//! physics only:
//! - material at rest stays at rest and bakes back to the terrain it came
//!   from (an elastic layer to micrometres, sand to its own creep);
//! - a plane P-wave sent down at the bed leaves through it (Lysmer &
//!   Kuhlemeyer's dashpot is exact at normal incidence), where the rigid bed
//!   sends it back;
//! - a region the size of the browser demo's settles after a km/s impact,
//!   where between rigid walls it keeps ringing.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::layout::MpmParticle;
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    substeps_for, CraterTarget, ImpactSource, InitialStress, RegionBoundary, RegionPlan,
    SandMaterial, TerrainConfig,
};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Quat, Vec3};

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

/// `config` with the ground absorbing or rigid, and the material's start.
fn ground(config: TerrainConfig, absorbing: bool, initial_stress: InitialStress) -> TerrainConfig {
    TerrainConfig {
        boundary: if absorbing {
            RegionBoundary::Absorbing
        } else {
            RegionBoundary::Rigid
        },
        initial_stress,
        ..config
    }
}

fn live(p: &MpmParticle) -> bool {
    p.f[2][3].to_bits() & 1 == 0
}

fn gpu_heights(s: &WgpuSim, collider: u32, field: &HeightField) -> Vec<f32> {
    let base = engine_schema::convex::convex_params_base(s.collider(collider).params);
    let (nx, nz) = field.samples();
    let words = s.read_words(18, u64::from(base + 2) * 4, (nx * nz) as usize);
    words.into_iter().map(f32::from_bits).collect()
}

/// A flat field at y = 0 of `n`² samples `spacing` apart, as collider.
fn flat(s: &mut WgpuSim, n: u32, spacing: f32) -> (HeightField, u32) {
    let field = HeightField::new(n, n, spacing, spacing, vec![0.0; (n * n) as usize]).unwrap();
    let mut arena = HullArena::new();
    let (_, collider) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    (field, collider)
}

/// An elastic material of sand's density and stiffness (a rock that does not
/// crack at these stresses: its tensile strength is 1 MPa).
fn elastic(sand: &SandMaterial) -> engine_terrain::RockMaterial {
    engine_terrain::RockMaterial {
        density: sand.density,
        youngs_modulus: sand.youngs_modulus,
        poisson_ratio: sand.poisson_ratio,
        tensile_strength: 1e6,
        fracture_energy: 1e4,
    }
}

/// What a region at rest does in a second.
struct Rest {
    /// Fastest particle over the second (m/s).
    top: f32,
    /// Fastest particle at its end (m/s).
    last: f32,
    /// Largest displacement of a particle (m).
    moved: f32,
}

/// Material activated with no source (1.2 m deep at most) and left under
/// gravity for a second, then baked: the bake must give back the terrain it
/// came from, and nothing may leave the region. `mound`: a sand mound 0.3 m
/// high on the field, whose samples are not the region's cells (12.5 cm
/// against 10 cm, so the walls stand off the node lattice); otherwise flat.
fn at_rest(cfg: TerrainConfig, sand: bool, mound: bool) -> Rest {
    let mut s = sim();
    s.terrain_enable(&cfg).unwrap();
    let material = SandMaterial::default();
    if sand {
        s.terrain_set_material(Some(&material)).unwrap();
    } else {
        s.terrain_set_material(&elastic(&material)).unwrap();
    }
    let peak = if mound { 0.3 } else { 0.0 };
    let field = HeightField::from_fn(33, 33, 0.125, 0.125, |x, z| {
        peak * (-(x * x + z * z) / 0.25).exp()
    })
    .unwrap();
    let mut arena = HullArena::new();
    let origin = Vec3::new(0.0, -0.5, 0.0);
    let (_, collider) = s
        .add_height_field(&mut arena, &field, origin, Quat::IDENTITY, 0)
        .unwrap();
    let before = gpu_heights(&s, collider, &field);
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-1.0, -1.6, -1.0),
        Vec3::new(1.0, 0.0, 1.0),
        0.1,
        0,
        0,
    )
    .unwrap();
    s.terrain_activate(&plan, collider, &field, origin, material.density, 0.5)
        .unwrap();
    let start = s.terrain_particles();
    let (k, dt) = substeps_for(&cfg, FRAME, plan.dx, material.p_wave_speed(), 1.0).unwrap();
    let mut top = 0.0f32;
    for _ in 0..60 {
        let rep = s.terrain_step(k, dt, -G).unwrap();
        top = top.max(rep.max_speed);
        assert_eq!(rep.parcels, 0, "material left the region at rest");
    }
    let end = s.terrain_particles();
    let moved = start
        .iter()
        .zip(&end)
        .map(|(a, b)| (Vec3::from_slice(&a.xm[..3]) - Vec3::from_slice(&b.xm[..3])).length())
        .fold(0.0f32, f32::max);
    let last = end
        .iter()
        .map(|p| Vec3::from_slice(&p.vj[..3]).length())
        .fold(0.0f32, f32::max);
    eprintln!(
        "at rest ({:?}, {:?}, {}, {}): top speed {top:.2e} m/s, last {last:.2e} m/s, largest move {moved:.2e} m",
        cfg.boundary,
        cfg.initial_stress,
        if sand { "sand" } else { "elastic" },
        if mound { "mound" } else { "flat" }
    );
    s.terrain_bake().unwrap();
    let after = gpu_heights(&s, collider, &field);
    for (k, (a, b)) in before.iter().zip(&after).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "sample {k}: {a} before, {b} after the bake"
        );
    }
    Rest { top, last, moved }
}

fn rest_config(absorbing: bool, initial_stress: InitialStress) -> TerrainConfig {
    ground(
        TerrainConfig {
            max_particles: 1 << 18,
            max_blocks: 1 << 12,
            ..TerrainConfig::default()
        },
        absorbing,
        initial_stress,
    )
}

/// The absorbing ground carries the geostatic state the material starts in,
/// so an elastic layer is in equilibrium on it: it moves by a few
/// micrometres, against the 0.2 mm (and up to g·H/c_p ≈ 6 cm/s) an unstressed
/// layer would sag between rigid walls.
#[test]
fn an_elastic_layer_at_rest_stays_put_on_the_absorbing_ground() {
    let r = at_rest(rest_config(true, InitialStress::Geostatic), false, false);
    assert!(r.top < 1e-3, "the layer at rest reached {} m/s", r.top);
    assert!(
        r.last < 1e-4,
        "still moving at {} m/s after a second",
        r.last
    );
    assert!(r.moved < 2e-5, "a particle moved {} m", r.moved);
}

/// Sand on the absorbing ground, a mound with walls off the lattice: what
/// is left is the sand's own slow plastic creep near its free surface, where
/// it has almost no confinement (rigid walls with an unstressed start: 9 cm/s
/// at first, 2 mm/s and 1.8 mm after a second).
#[test]
fn sand_at_rest_stays_put_on_the_absorbing_ground() {
    let r = at_rest(rest_config(true, InitialStress::Geostatic), true, true);
    assert!(r.top < 0.03, "the sand at rest reached {} m/s", r.top);
    assert!(
        r.last < 1e-3,
        "still moving at {} m/s after a second",
        r.last
    );
    assert!(r.moved < 1e-3, "a particle moved {} m", r.moved);
}

/// The wave's energy in the live particles within `half` of the region's
/// axis: kinetic, plus the elastic energy of the strain since `rest` (the same
/// particles before the pulse), μ·e:e + ½λ·(tr e)² per unit volume.
fn central_energy(ps: &[MpmParticle], rest: &[MpmParticle], half: f32, lam: f32, mu: f32) -> f64 {
    let f = |p: &MpmParticle| {
        glam::DMat3::from_cols_array(&[
            f64::from(p.f[0][0]),
            f64::from(p.f[1][0]),
            f64::from(p.f[2][0]),
            f64::from(p.f[0][1]),
            f64::from(p.f[1][1]),
            f64::from(p.f[2][1]),
            f64::from(p.f[0][2]),
            f64::from(p.f[1][2]),
            f64::from(p.f[2][2]),
        ])
    };
    ps.iter()
        .zip(rest)
        .filter(|(p, _)| live(p) && p.xm[0].abs() < half && p.xm[2].abs() < half)
        .map(|(p, r)| {
            let v = Vec3::from_slice(&p.vj[..3]).as_dvec3();
            let d = f(p) - f(r);
            let e = (d + d.transpose()) * 0.5;
            let ee: f64 = (0..3)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .map(|(i, j)| e.col(j)[i].powi(2))
                .sum();
            let tr = e.col(0)[0] + e.col(1)[1] + e.col(2)[2];
            let w = f64::from(mu) * ee + 0.5 * f64::from(lam) * tr * tr;
            0.5 * f64::from(p.xm[3]) * v.length_squared() + w * f64::from(p.f[0][3])
        })
        .sum()
}

/// The plane P-waves sent down at the bed, of an amplitude (m/s).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Pulse {
    /// One sine period 8 cells long (no net displacement), in an elastic
    /// material of sand's density and stiffness (a rock that never cracks
    /// here: its tensile strength is 1 MPa).
    Wavelet(f32),
    /// A compressive cos² pulse 6 cells long, in that elastic material. It
    /// displaces the layer by its integral, ∫v dt.
    Compressive(f32),
    /// The compressive pulse in the sand itself.
    SandCompressive(f32),
}

/// A plane P-wave moving down a layer 1.4 m deep (geostatic, under gravity),
/// measured on the region's axis 8 cells after the whole pulse has reached
/// the bed: the wave energy left in the layer is what the bed sent back.
/// The region is 5 m wide so that what its side walls do to the wave (they
/// are not at normal incidence) has not reached the axis by then. Lysmer &
/// Kuhlemeyer's dashpot is exact for plane waves at normal incidence; the rigid
/// bed sends all of it back, less the transfers' own dissipation.
fn p_wave_reflection(absorbing: bool, pulse: Pulse) -> f64 {
    let mut s = sim();
    let cfg = ground(
        TerrainConfig {
            max_particles: 1 << 21,
            max_blocks: 1 << 13,
            max_columns: 1 << 14,
            ..TerrainConfig::default()
        },
        absorbing,
        // Both start geostatic, so they differ only at the bed.
        InitialStress::Geostatic,
    );
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    if !matches!(pulse, Pulse::SandCompressive(_)) {
        s.terrain_set_material(&elastic(&sand)).unwrap();
    }
    let dx = 1.0 / 16.0;
    let (field, collider) = flat(&mut s, 89, dx);
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-2.5, -1.5, -2.5),
        Vec3::new(2.5, 0.5, 2.5),
        dx,
        0,
        0,
    )
    .unwrap();
    s.terrain_activate(&plan, collider, &field, Vec3::ZERO, sand.density, 0.5)
        .unwrap();
    let c = sand.p_wave_speed();
    let (_, dt) = substeps_for(&cfg, FRAME, dx, c, 2.0).unwrap();
    // One substep first: the geostatic start sets F.
    s.terrain_step(1, dt, -G).unwrap();
    let bed = plan.origin.y + 1.5 * dx;
    let (len, amp) = match pulse {
        Pulse::Wavelet(a) => (8.0 * dx, a),
        Pulse::Compressive(a) | Pulse::SandCompressive(a) => (6.0 * dx, a),
    };
    let y0 = -0.5 * len - dx;
    let rest = s.terrain_particles();
    let mut ps = rest.clone();
    let pi = std::f32::consts::PI;
    for p in ps.iter_mut() {
        let u = (p.xm[1] - y0) / len;
        if u.abs() < 0.5 {
            // Moving down, u_y = f(y + c·t): v_y = c·f' and ε_yy = f' = v_y/c;
            // and APIC's affine velocity dv_y/dy.
            let (v, dvdy) = match pulse {
                Pulse::Wavelet(_) => (
                    amp * (2.0 * pi * u).sin(),
                    amp * 2.0 * pi * (2.0 * pi * u).cos() / len,
                ),
                Pulse::Compressive(_) | Pulse::SandCompressive(_) => (
                    -amp * (pi * u).cos().powi(2),
                    amp * pi * (2.0 * pi * u).sin() / len,
                ),
            };
            p.vj[1] += v;
            p.c[1][1] += dvdy;
            for row in p.f.iter_mut() {
                row[1] *= 1.0 + v / c;
            }
        }
    }
    s.terrain_write_particles(0, &ps).unwrap();
    let half = 0.25;
    let (lam, mu) = (sand.lambda(), sand.mu());
    let start = central_energy(&ps, &rest, half, lam, mu);
    // The wave as it arrives, its front a cell above the bed (the transfers
    // dissipate some on the way down) ...
    let t0 = (y0 - 0.5 * len - bed - dx) / c;
    let arrive = (t0 / dt).floor() as u32;
    s.terrain_step(arrive, dt, -G).unwrap();
    let e0 = central_energy(&s.terrain_particles(), &rest, half, lam, mu);
    // ... and 8 cells after its tail has reached the bed.
    let t1 = (y0 + 0.5 * len - bed + 8.0 * dx) / c;
    let steps = (t1 / dt).ceil() as u32 - arrive;
    s.terrain_step(steps, dt, -G).unwrap();
    let e1 = central_energy(&s.terrain_particles(), &rest, half, lam, mu);
    let r = e1 / e0;
    eprintln!(
        "{} bed, {pulse:?}: wave energy on the axis {start:.4e} J, arriving {e0:.4e} J, {e1:.4e} J {:.2} ms later: {r:.4}",
        if absorbing { "absorbing" } else { "rigid" },
        1e3 * steps as f32 * dt,
    );
    r
}

#[test]
fn a_plane_p_wave_leaves_through_the_absorbing_bed() {
    // Neither bed pulls (sand does not stick to the ground), so a wave whose
    // tension would exceed the overburden's 22 kPa at the bed (6.7 cm/s) is
    // not sent back as tension by either. The rigid bed's wave is the
    // compressive pulse at 1 m/s: the geostatic layer on a rigid bed is not
    // quite at rest (its floor's load is not split between the node layers as
    // the stress transmits it), and that motion is about 2 cm/s.
    let rigid = p_wave_reflection(false, Pulse::Compressive(1.0));
    let absorbing = p_wave_reflection(true, Pulse::Wavelet(0.02));
    assert!(
        rigid > 0.8,
        "the rigid bed sent back only {rigid} of the pulse"
    );
    assert!(
        absorbing < 0.015,
        "the absorbing bed sent back {absorbing} of the wave's energy"
    );
    // The compressive pulse in the absorbing bed: the dashpots take it, then
    // the springs push the layer back up by the pulse's displacement (which a
    // plane wave carries away for good, the ground beyond a finite region does
    // not). And in the sand itself, 330 kPa on the overburden.
    let elastic = p_wave_reflection(true, Pulse::Compressive(1.0));
    let sand_rigid = p_wave_reflection(false, Pulse::SandCompressive(1.0));
    let sand = p_wave_reflection(true, Pulse::SandCompressive(1.0));
    assert!(
        elastic < 0.04,
        "the absorbing bed sent back {elastic} of the pulse"
    );
    assert!(
        sand_rigid > 0.6,
        "the rigid bed sent back only {sand_rigid} of the pulse in sand"
    );
    assert!(
        sand < 0.04,
        "the absorbing bed sent back {sand} of the pulse in sand"
    );
}

/// The browser demo's event (`impact.html`: an 8 m stony impactor at 20 km/s
/// into dry sand, a 151 m transient radius), at 8 cells per radius instead of
/// 16 and margin 0.6, a 300 m deep box: the fraction of the material still
/// faster than 1 m/s ten seconds after the strike.
fn demo_after_impact(absorbing: bool) -> (f64, f64) {
    let mut s = sim();
    let cfg = ground(
        TerrainConfig {
            max_particles: 1 << 18,
            max_blocks: 4096,
            max_parcels: 1 << 16,
            max_columns: 1 << 13,
            max_field_samples: 1 << 16,
            cells_per_radius: 8.0,
            ..TerrainConfig::default()
        },
        absorbing,
        // Against the rigid ground as it was (and as the demo ran).
        if absorbing {
            InitialStress::Geostatic
        } else {
            InitialStress::Unstressed
        },
    );
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    let src = ImpactSource {
        impactor_radius: 8.0,
        impactor_density: 3000.0,
        velocity: DVec3::new(0.0, -20000.0, 0.0),
        target: CraterTarget::sand(&sand).unwrap(),
    };
    let g = f64::from(G);
    let crater = src.crater(g).unwrap();
    let model = src.field(g, point_source_z(src.target.ejecta.mu)).unwrap();
    let plan = crater.region(&cfg, Vec3::ZERO, 0.6, 0, 0).unwrap();
    let n = 2 * (3.0 * crater.radius as f32 / plan.dx).ceil() as u32 + 1;
    let (field, collider) = flat(&mut s, n, plan.dx);
    s.terrain_activate(
        &plan,
        collider,
        &field,
        Vec3::ZERO,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    s.terrain_apply_source(&model, Vec3::ZERO).unwrap();
    let mut speed = engine_terrain::plan::source_peak_speed(&model);
    let mut t = 0.0;
    while t < 10.0 {
        let (k, dt) = substeps_for(
            &cfg,
            FRAME,
            plan.dx,
            sand.p_wave_speed(),
            1.5 * speed + 20.0,
        )
        .unwrap();
        speed = s.terrain_step(k, dt, -G).unwrap().max_speed;
        t += FRAME;
    }
    let ps: Vec<f32> = s
        .terrain_particles()
        .iter()
        .filter(|p| live(p))
        .map(|p| Vec3::from_slice(&p.vj[..3]).length())
        .collect();
    let frac = |v: f32| ps.iter().filter(|&&s| s > v).count() as f64 / ps.len() as f64;
    let (fast, moving) = (frac(1.0), frac(0.05));
    eprintln!(
        "{} ground, 10 s after the strike: {fast:.4} of {} particles faster than 1 m/s, {moving:.4} \
         faster than 5 cm/s, top {speed:.2} m/s",
        if absorbing { "absorbing" } else { "rigid" },
        ps.len()
    );
    (fast, moving)
}

#[test]
fn a_demo_scale_region_settles_on_the_absorbing_ground() {
    let (rigid, _) = demo_after_impact(false);
    let (absorbing, moving) = demo_after_impact(true);
    assert!(
        rigid > 0.3,
        "between rigid walls only {rigid} still moves at 1 m/s"
    );
    assert!(
        absorbing < 1e-3,
        "{absorbing} of the region is still faster than 1 m/s"
    );
    assert!(
        moving < 0.6,
        "{moving} of the region is still faster than 5 cm/s"
    );
}
