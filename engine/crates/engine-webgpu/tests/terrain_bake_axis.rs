//! The browser demo's crater (`impact.html`: an 8 m stony impactor at 20 km/s
//! into dry sand, a 151 m transient radius, on the absorbing ground with a
//! geostatic start), settled and baked, along the two axes through the impact
//! point.
//!
//! The event is mirror-symmetric about those axes, and no particle crosses a
//! mirror plane. Away from the planes, the flow mixes particles between their
//! rest lattice sheets; at a plane it cannot. So after the crater forms, the
//! particles beside each plane stand a little apart across it: a sub-cell
//! crack, too thin for P2G's quadratic B-spline (three cells wide) to see,
//! which nothing in the simulation then closes. A bake that resolves finer
//! than the grid turns it into a trench a sample wide: the bilinear splat at
//! one sample put the axis lines 4.8 m below their neighbours at the demo's
//! 16 cells per radius (5.5 m at 8), while P2G's own kernel shows no trench.
//! The bake now splats material the MPM moved with that kernel
//! (`terrain_mpm.slang`, `mpmHomeShare`).

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::layout::MpmParticle;
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    substeps_for, CraterTarget, ImpactSource, InitialStress, RegionBoundary, SandMaterial,
    TerrainConfig,
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

fn live(p: &MpmParticle) -> bool {
    p.f[2][3].to_bits() & 1 == 0
}

fn gpu_heights(s: &WgpuSim, collider: u32, field: &HeightField) -> Vec<f32> {
    let base = engine_schema::convex::convex_params_base(s.collider(collider).params);
    let (nx, nz) = field.samples();
    let words = s.read_words(18, u64::from(base + 2) * 4, (nx * nz) as usize);
    words.into_iter().map(f32::from_bits).collect()
}

/// The demo's crater at `cells` per radius, settled (the demo's tolerance,
/// 0.01·√(gR)) and baked.
struct Crater {
    field: HeightField,
    /// Heights after activation (the window at the region's bed).
    lowered: Vec<f32>,
    baked: Vec<f32>,
    /// The live particles just before the bake.
    particles: Vec<MpmParticle>,
    /// The transient radius in samples.
    radius: i64,
}

fn demo_crater(cells: f32) -> Crater {
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 3 << 19,
        max_blocks: 8192,
        max_parcels: 1 << 18,
        max_columns: 1 << 17,
        max_field_samples: 1 << 19,
        max_substeps: 1024,
        cells_per_radius: cells,
        boundary: RegionBoundary::Absorbing,
        initial_stress: InitialStress::Geostatic,
        ..TerrainConfig::default()
    };
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
    let plan = crater.region(&cfg, Vec3::ZERO, 1.0, 0, 0).unwrap();
    let n = 2 * (3.0 * crater.radius as f32 / plan.dx).ceil() as u32 + 1;
    let field = HeightField::new(n, n, plan.dx, plan.dx, vec![0.0; (n * n) as usize]).unwrap();
    let mut arena = HullArena::new();
    let (_, collider) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    s.terrain_activate(
        &plan,
        collider,
        &field,
        Vec3::ZERO,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    let lowered = gpu_heights(&s, collider, &field);
    s.terrain_apply_source(&model, Vec3::ZERO).unwrap();
    let settle = 0.01 * (g * crater.radius).sqrt() as f32;
    let mut speed = engine_terrain::plan::source_peak_speed(&model);
    let (mut t, mut airborne) = (0.0f32, 1u32);
    while t < 40.0 && !(t > 2.0 && speed < settle && airborne == 0) {
        let (k, dt) = substeps_for(
            &cfg,
            FRAME,
            plan.dx,
            sand.p_wave_speed(),
            1.5 * speed + 20.0,
        )
        .unwrap();
        let rep = s.terrain_step(k, dt, -G).unwrap();
        speed = rep.max_speed;
        airborne = rep.airborne;
        t += FRAME;
    }
    assert!(speed < settle, "not settled after {t} s ({speed} m/s)");
    eprintln!(
        "{cells} cells/R: settled to {speed:.3} m/s (tolerance {settle:.3}) at t = {t:.2} s, dx {:.2} m",
        plan.dx
    );
    let particles = s.terrain_particles().into_iter().filter(live).collect();
    s.terrain_bake().unwrap();
    Crater {
        baked: gpu_heights(&s, collider, &field),
        field,
        lowered,
        particles,
        radius: (crater.radius as f32 / plan.dx).round() as i64,
    }
}

/// Mean over the four arms from `r0` to `r1` samples out of (the line `k`
/// samples off an axis − the mean of the lines beside it): k = 0 is the axes
/// themselves; k > 0 averages the lines on both sides.
fn line_excess(h: &dyn Fn(i64, i64) -> f64, c: i64, k: i64, r0: i64, r1: i64) -> f64 {
    let (mut sum, mut n) = (0.0, 0);
    for t in r0..=r1 {
        for side in if k == 0 { vec![0] } else { vec![-k, k] } {
            for (sx, sz) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                // Along the arm by t, across it by `side`.
                let (i, j) = (c + sx * t + sz.abs() * side, c + sz * t + sx.abs() * side);
                let (a, b) = if sx != 0 {
                    ((i, j - 1), (i, j + 1))
                } else {
                    ((i - 1, j), (i + 1, j))
                };
                sum += h(i, j) - 0.5 * (h(a.0, a.1) + h(b.0, b.1));
                n += 1;
            }
        }
    }
    sum / f64::from(n)
}

/// The axes of a baked crater stand no lower than the lines parallel to them
/// (2 to 4 samples off) do against their own neighbours, from 2 samples out to
/// the transient radius. Each line's excess over its neighbours is the
/// crater's own curvature across it plus the bake's noise; the axes must not
/// add a trench to that. The bilinear bake left them 5.5 m low here.
#[test]
fn a_crater_bakes_without_a_trench_along_its_axes() {
    let c = demo_crater(8.0);
    let n = c.field.samples().0 as i64;
    let at = |i: i64, j: i64| (j * n + i) as usize;
    let h = |i: i64, j: i64| f64::from(c.baked[at(i, j)]);
    let mid = n / 2;
    let axis = line_excess(&h, mid, 0, 2, c.radius);
    let others: Vec<f64> = (2..=4)
        .map(|k| line_excess(&h, mid, k, 2, c.radius))
        .collect();
    let worst = others.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    eprintln!(
        "baked: the axes stand {axis:.3} m over their neighbours; lines 2, 3, 4 samples off: {:?}",
        others
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    assert!(
        axis >= -worst.max(1.0),
        "the axes stand {axis} m below their neighbours (lines off them: {others:?})"
    );
}

/// Diagnostic at the demo's own resolution: the axes against their
/// neighbours in the particles just before the bake (the highest particle
/// over each sample; and the rest volume binned by nearest sample), the
/// bilinear bake the GPU did before, P2G's B-spline alone, and the GPU bake.
#[test]
#[ignore = "diagnostic (about a minute): cargo test --release ... -- --ignored --nocapture"]
fn the_demo_crater_along_its_axes() {
    let cells: f32 = std::env::var("AXIS_CPR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16.0);
    let c = demo_crater(cells);
    let n = c.field.samples().0 as i64;
    let nn = n as usize;
    let (dx, _) = c.field.cell_size();
    let x0 = c.field.vertex(0, 0);
    let area = f64::from(dx * dx);
    let at = |i: i64, j: i64| (j * n + i) as usize;
    let mut top = vec![f64::MIN; nn * nn];
    let mut rest = vec![0.0f64; nn * nn];
    let tent = |t: f64| -> [(i64, f64); 3] {
        let b = t.floor();
        [
            (b as i64, 1.0 - (t - b)),
            (b as i64 + 1, t - b),
            (b as i64 + 2, 0.0),
        ]
    };
    let bspl = |t: f64| -> [(i64, f64); 3] {
        let b = (t - 0.5).floor();
        let f = t - b;
        [
            (b as i64, 0.5 * (1.5 - f).powi(2)),
            (b as i64 + 1, 0.75 - (f - 1.0).powi(2)),
            (b as i64 + 2, 0.5 * (f - 0.5).powi(2)),
        ]
    };
    let mut bilinear = vec![0.0f64; nn * nn];
    let mut bspline = vec![0.0f64; nn * nn];
    let fades = [1.0f64, 0.5, 0.25];
    let mut hyb_home = vec![vec![0.0f64; nn * nn]; 3];
    let mut hyb_partial = vec![0.0f64; nn * nn];
    for p in &c.particles {
        let v0 = f64::from(p.f[0][3]);
        let ax = |x: f32, h: f32, o: f32| {
            let s0 = f64::from(h - o) / f64::from(dx);
            (
                f64::from(x - o) / f64::from(dx),
                s0 - (s0 + 0.5).floor(),
                f64::from(x - h) / f64::from(dx),
            )
        };
        let (ux, ox, mx) = ax(p.xm[0], p.c[1][3], x0.x);
        let (uz, oz, mz) = ax(p.xm[2], p.c[2][3], x0.z);
        let w = if p.f[2][3].to_bits() & 2 != 0 {
            // The bilinear bake's default fade: one sample spacing.
            (1.0 - mx.hypot(mz)).max(0.0)
        } else {
            0.0
        };
        let mut add = |arr: &mut Vec<f64>, kx: [(i64, f64); 3], kz: [(i64, f64); 3]| {
            for (i, wx) in kx {
                for (j, wz) in kz {
                    if i >= 0 && j >= 0 && i < n && j < n {
                        arr[at(i, j)] += wx * wz * v0;
                    }
                }
            }
        };
        // The bilinear bake before: the tent at u − w·o.
        add(&mut bilinear, tent(ux - w * ox), tent(uz - w * oz));
        add(&mut bspline, bspl(ux), bspl(uz));
        let home = p.f[2][3].to_bits() & 2 != 0;
        for (k, fade) in fades.iter().enumerate() {
            let wk = if home {
                (1.0 - mx.hypot(mz) / fade).max(0.0)
            } else {
                0.0
            };
            let (vh, vb) = (v0 * wk, v0 * (1.0 - wk));
            for (i, wx) in tent(ux - ox) {
                for (j, wz) in tent(uz - oz) {
                    if i >= 0 && j >= 0 && i < n && j < n {
                        hyb_home[k][at(i, j)] += wx * wz * vh;
                    }
                }
            }
            for (i, wx) in bspl(ux) {
                for (j, wz) in bspl(uz) {
                    if i >= 0 && j >= 0 && i < n && j < n {
                        hyb_home[k][at(i, j)] += wx * wz * vb;
                    }
                }
            }
        }
        for (i, wx) in tent(ux - w * ox) {
            for (j, wz) in tent(uz - w * oz) {
                if i >= 0 && j >= 0 && i < n && j < n {
                    hyb_partial[at(i, j)] += wx * wz * v0 * w;
                }
            }
        }
        for (i, wx) in bspl(ux) {
            for (j, wz) in bspl(uz) {
                if i >= 0 && j >= 0 && i < n && j < n {
                    hyb_partial[at(i, j)] += wx * wz * v0 * (1.0 - w);
                }
            }
        }
        let (i, j) = (ux.round() as i64, uz.round() as i64);
        if i >= 0 && j >= 0 && i < n && j < n {
            top[at(i, j)] = top[at(i, j)].max(f64::from(p.xm[1]));
            rest[at(i, j)] += v0;
        }
    }
    let low = |v: &Vec<f64>, i: i64, j: i64| f64::from(c.lowered[at(i, j)]) + v[at(i, j)] / area;
    let measures: Vec<(&str, Box<dyn Fn(i64, i64) -> f64 + '_>)> = vec![
        ("particle top", Box::new(|i, j| top[at(i, j)])),
        (
            "rest volume, nearest sample",
            Box::new(|i, j| low(&rest, i, j)),
        ),
        (
            "bilinear bake (before)",
            Box::new(|i, j| low(&bilinear, i, j)),
        ),
        ("P2G B-spline alone", Box::new(|i, j| low(&bspline, i, j))),
        ("GPU bake", Box::new(|i, j| f64::from(c.baked[at(i, j)]))),
        (
            "hybrid at home+d, fade 1",
            Box::new(|i, j| low(&hyb_home[0], i, j)),
        ),
        (
            "hybrid at home+d, fade 0.5",
            Box::new(|i, j| low(&hyb_home[1], i, j)),
        ),
        (
            "hybrid at home+d, fade 0.25",
            Box::new(|i, j| low(&hyb_home[2], i, j)),
        ),
        (
            "hybrid at u-w·o, fade 1",
            Box::new(|i, j| low(&hyb_partial, i, j)),
        ),
    ];
    let mid = n / 2;
    for (name, h) in &measures {
        let axis = line_excess(h.as_ref(), mid, 0, 2, c.radius);
        let others: Vec<String> = (2..=4)
            .map(|k| format!("{:.2}", line_excess(h.as_ref(), mid, k, 2, c.radius)))
            .collect();
        let prof: Vec<String> = (2..=c.radius)
            .step_by(3)
            .map(|t| {
                let (i, j) = (mid - t, mid);
                format!("{:.1}/{:.1}", h(i, j), 0.5 * (h(i, j - 1) + h(i, j + 1)))
            })
            .collect();
        eprintln!(
            "{name}: axes {axis:.2} m over their neighbours, lines 2-4 off {}; −x arm (axis/beside): {}",
            others.join(" "),
            prof.join(" ")
        );
    }
}
