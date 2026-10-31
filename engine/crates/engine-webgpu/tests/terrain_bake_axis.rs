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

fn demo_crater(cells: f32, margin: f64) -> Crater {
    demo_crater_grown(cells, margin, None)
}

/// [`demo_crater`], with the settled area grown before the bake by a
/// sourceless box reaching `grow` transient radii past its +x side (the
/// other sides stay where they are).
fn demo_crater_grown(cells: f32, margin: f64, grow: Option<f32>) -> Crater {
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
    let plan = crater.region(&cfg, Vec3::ZERO, margin, 0, 0).unwrap();
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
    if let Some(grow) = grow {
        let block = plan.dx * 4.0;
        let (o, e) = (plan.origin, plan.end());
        let r = crater.radius as f32;
        let ext = engine_terrain::RegionPlan::spanning(
            &cfg,
            Vec3::new(e.x - 2.0 * block, o.y + block, o.z + block),
            Vec3::new(e.x + grow * r, e.y - block, e.z - block),
            plan.dx,
            0,
            0,
        )
        .unwrap();
        let opened = s
            .terrain_open_area(
                &ext,
                collider,
                &field,
                Vec3::ZERO,
                sand.density,
                sand.friction_angle.tan(),
            )
            .unwrap();
        assert!(
            opened.activated && opened.absorbed.len() == 1,
            "the area grew"
        );
    }
    let particles = s.terrain_particles().into_iter().filter(live).collect();
    // The absorbing ground's node displacements (binding 49, after the
    // coupling slots): the largest horizontal one.
    let records = engine_terrain::plan::TerrainPlan::boundary_nodes([
        plan.blocks.x as u32 * 4,
        plan.blocks.y as u32 * 4,
        plan.blocks.z as u32 * 4,
    ]) as usize;
    let words = s.read_words(49, 64 * u64::from(cfg.max_coupled), 4 * records);
    let wall = words
        .chunks(4)
        .map(|w| f32::from_bits(w[0]).abs().max(f32::from_bits(w[2]).abs()))
        .fold(0.0f32, f32::max);
    eprintln!("largest horizontal ground displacement at the walls: {wall:.3} m");
    if std::env::var("WALL_TRACE").is_ok() {
        // The low-x side's face points on the region's middle z row.
        let d = [
            plan.blocks.x * 4 + 1,
            plan.blocks.y * 4 + 1,
            plan.blocks.z * 4 + 1,
        ];
        let (nx, ny, nz) = (d[0] as usize, d[1] as usize, d[2] as usize);
        let gz = nz / 2;
        let u: Vec<String> = (0..ny)
            .map(|gy| {
                format!(
                    "{:+.2}",
                    f32::from_bits(words[4 * (nx * nz + gy * nz + gz)])
                )
            })
            .collect();
        eprintln!("  low-x side, u_x by node row: {}", u.join(" "));
    }
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
    let c = demo_crater(8.0, 1.0);
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
    let c = demo_crater(cells, 1.0);
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
    // Mean displacement of each home column's particles (rest-volume
    // weighted), for the share at home measured against the column.
    let lattice = f64::from(dx) / 2.0;
    let mut col = vec![(0.0f64, 0.0f64, 0.0f64); nn * nn];
    for p in &c.particles {
        if p.f[2][3].to_bits() & 2 == 0 {
            continue;
        }
        let hi = ((p.c[1][3] - x0.x) / dx).round() as i64;
        let hj = ((p.c[2][3] - x0.z) / dx).round() as i64;
        let v0 = f64::from(p.f[0][3]);
        let e = &mut col[at(hi, hj)];
        e.0 += v0 * f64::from(p.xm[0] - p.c[1][3]);
        e.1 += v0 * f64::from(p.xm[2] - p.c[2][3]);
        e.2 += v0;
    }
    // Variants: (relative to the column?, fade in lattice spacings).
    let variants = [(false, 0.5f64), (true, 0.5), (true, 0.25), (true, 0.125)];
    let mut hyb = vec![vec![0.0f64; nn * nn]; variants.len()];
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
        let home = p.f[2][3].to_bits() & 2 != 0;
        let w = if home {
            (1.0 - mx.hypot(mz)).max(0.0)
        } else {
            0.0
        };
        let mut add = |arr: &mut Vec<f64>, kx: [(i64, f64); 3], kz: [(i64, f64); 3], q: f64| {
            for (i, wx) in kx {
                for (j, wz) in kz {
                    if i >= 0 && j >= 0 && i < n && j < n {
                        arr[at(i, j)] += wx * wz * q;
                    }
                }
            }
        };
        // The bilinear bake before: the tent at u − w·o, w over one sample.
        add(&mut bilinear, tent(ux - w * ox), tent(uz - w * oz), v0);
        add(&mut bspline, bspl(ux), bspl(uz), v0);
        for (k, &(relative, fade)) in variants.iter().enumerate() {
            let wk = if !home {
                0.0
            } else {
                let (mut rx, mut rz) = (mx * f64::from(dx), mz * f64::from(dx));
                if relative {
                    let hi = ((p.c[1][3] - x0.x) / dx).round() as i64;
                    let hj = ((p.c[2][3] - x0.z) / dx).round() as i64;
                    let e = col[at(hi, hj)];
                    rx -= e.0 / e.2;
                    rz -= e.1 / e.2;
                }
                (1.0 - rx.hypot(rz) / (fade * lattice)).max(0.0)
            };
            add(&mut hyb[k], tent(ux - ox), tent(uz - oz), v0 * wk);
            add(&mut hyb[k], bspl(ux), bspl(uz), v0 * (1.0 - wk));
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
            "hybrid, absolute displacement, fade 0.5 (this bake)",
            Box::new(|i, j| low(&hyb[0], i, j)),
        ),
        (
            "hybrid, against the column mean, fade 0.5",
            Box::new(|i, j| low(&hyb[1], i, j)),
        ),
        (
            "hybrid, against the column, fade 0.25",
            Box::new(|i, j| low(&hyb[2], i, j)),
        ),
        (
            "hybrid, against the column, fade 0.125",
            Box::new(|i, j| low(&hyb[3], i, j)),
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
        // The largest local bump (a sample against the mean of its four
        // neighbours) from 1.6 to 2.6 radii out: the activation window's edge.
        let mut bump = 0.0f64;
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let r = (((i - mid).pow(2) + (j - mid).pow(2)) as f64).sqrt() / c.radius as f64;
                if !(1.6..=2.6).contains(&r) {
                    continue;
                }
                let v = h(i, j) - 0.25 * (h(i - 1, j) + h(i + 1, j) + h(i, j - 1) + h(i, j + 1));
                if v.abs() > bump.abs() {
                    bump = v;
                }
            }
        }
        let edge: Vec<String> = (mid - 35..=mid - 26)
            .map(|i| format!("{:.0}", h(i, mid + 5)))
            .collect();
        eprintln!(
            "{name}: row 5 off the axis, samples {}..{}: {}",
            -35,
            -26,
            edge.join(" ")
        );
        eprintln!(
            "{name}: axes {axis:.2} m over their neighbours, lines 2-4 off {}; window-edge bump {bump:.1} m; −x arm (axis/beside): {}",
            others.join(" "),
            prof.join(" ")
        );
    }
}

/// The edge of the activation window, after the demo's event (margin 1.5 R as
/// the demo, at 8 cells per radius): no sample of the window's rim may stand
/// out of the ground beside it. The absorbing walls give outward under the
/// crater's residual push (metres, at E = 50 MPa and R ≈ 0.4 km), and the
/// unactivated ground beyond is a heightfield, so a bake in the region's own
/// frame put all of that volume on the window's last samples: a ridge a
/// sample wide.
#[test]
fn the_window_edge_bakes_without_a_ridge() {
    // RIDGE_CPR: cells per radius (the demo runs 12).
    let cells = std::env::var("RIDGE_CPR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8.0);
    let c = demo_crater(cells, 1.5);
    let n = c.field.samples().0 as i64;
    let at = |i: i64, j: i64| (j * n + i) as usize;
    let inside = |i: i64, j: i64| c.lowered[at(i, j)] < -1.0;
    let h = |i: i64, j: i64| f64::from(c.baked[at(i, j)]);
    if std::env::var("RIDGE_TRACE").is_ok() {
        // On the −x arm's axis row: the particles' rest volume binned by the
        // nearest sample (as height), their mean x displacement from home,
        // and the highest particle, near the window's edge.
        let (dx, _) = c.field.cell_size();
        let x0 = c.field.vertex(0, 0);
        let mid = n / 2;
        let edge = (0..n).find(|&i| inside(i, mid)).unwrap();
        let mut bins = vec![(0.0f64, 0.0f64, f32::MIN, 0u32, 0.0f64); n as usize];
        for p in &c.particles {
            let i = ((p.xm[0] - x0.x) / dx).round() as i64;
            let j = ((p.xm[2] - x0.z) / dx).round() as i64;
            if j != mid || !(0..n).contains(&i) {
                continue;
            }
            let b = &mut bins[i as usize];
            b.0 += f64::from(p.f[0][3]);
            b.1 += f64::from(p.f[0][3]) * f64::from(p.xm[0] - p.c[1][3]);
            b.2 = b.2.max(p.xm[1]);
            b.3 += 1;
            if p.xm[1] < 0.0 {
                b.4 += f64::from(p.f[0][3]);
            }
        }
        // The edge column's particles by depth: mean x displacement.
        let mut prof = std::collections::BTreeMap::<i32, (f64, f64)>::new();
        for p in &c.particles {
            let i = ((p.c[1][3] - x0.x) / dx).round() as i64;
            let j = ((p.c[2][3] - x0.z) / dx).round() as i64;
            if i == edge && j == mid {
                let e = prof.entry((p.xm[1] / dx).floor() as i32).or_default();
                e.0 += f64::from(p.xm[0] - p.c[1][3]);
                e.1 += 1.0;
            }
        }
        let s: Vec<String> = prof
            .iter()
            .map(|(k, v)| format!("{k}:{:+.2}", v.0 / v.1))
            .collect();
        eprintln!(
            "  edge column (home), mean dx by depth cell: {}",
            s.join(" ")
        );
        for i in edge - 2..edge + 4 {
            let b = bins[i as usize];
            eprintln!(
                "  sample {:+}: lowered {:.1}, baked {:.1}; nearest-bin height {:.1} m (from {:.1}), mean dx {:+.2} m, top particle {:.1}, {} particles",
                i - mid,
                c.lowered[at(i, mid)],
                h(i, mid),
                b.0 / f64::from(dx * dx),
                f64::from(c.lowered[at(i, mid)]),
                b.1 / b.0.max(1e-9),
                b.2,
                b.3
            );
        }
    }
    // Window samples with an unactivated neighbour across one edge: how far
    // they stand over both that neighbour and the window sample behind them
    // (a ridge; a step down to the ground beside is not one).
    let (mut worst, mut at_worst) = (0.0f64, (0, 0));
    for j in 1..n - 1 {
        for i in 1..n - 1 {
            if !inside(i, j) {
                continue;
            }
            for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                if inside(i + di, j + dj) || !inside(i - di, j - dj) {
                    continue;
                }
                let v = h(i, j) - h(i + di, j + dj).max(h(i - di, j - dj));
                if v > worst {
                    worst = v;
                    at_worst = (i - n / 2, j - n / 2);
                }
            }
        }
    }
    let mid = n / 2;
    let row: Vec<String> = (0..n)
        .filter(|&i| inside(i, mid) && !inside(i - 1, mid) || !inside(i, mid) && inside(i + 1, mid))
        .flat_map(|i| (i - 2..=i + 2).map(move |k| k))
        .map(|k| format!("{:.1}", h(k, mid)))
        .collect();
    eprintln!(
        "window edge: largest ridge {worst:.2} m at sample {at_worst:?} from the centre; \
         the −x and +x edges on the axis (2 samples each side): {}",
        row.join(" ")
    );
    // Around the worst sample (rows of 7, from the edge inward).
    let (wi, wj) = (at_worst.0 + mid, at_worst.1 + mid);
    for j in wj - 2..=wj + 2 {
        let r: Vec<String> = (wi - 2..=wi + 4)
            .map(|i| format!("{:5.1}", h(i, j)))
            .collect();
        eprintln!("  row {:+}: {}", j - mid, r.join(" "));
    }
    // The bake's own noise: the same measure 2 to 4 samples in from the edge
    // (the first unactivated sample 3 to 5 steps away).
    let inb = |i: i64, j: i64| (0..n).contains(&i) && (0..n).contains(&j) && inside(i, j);
    let mut noise = 0.0f64;
    for j in 1..n - 1 {
        for i in 1..n - 1 {
            for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let out = (1..=6).find(|&d| !inb(i + d * di, j + d * dj));
                if inb(i, j) && matches!(out, Some(3..=5)) {
                    noise = noise.max(h(i, j) - h(i + di, j + dj).max(h(i - di, j - dj)));
                }
            }
        }
    }
    eprintln!("the same 2 to 4 samples in (the bake's noise): {noise:.2} m");
    assert!(
        worst < noise + 1.0,
        "a ridge {worst} m high stands at the window's edge (noise inside {noise} m)"
    );
}

/// An area grown over the ground beyond its absorbing walls (the user's bug,
/// 2026-10-29: `showcase.html?scene=bts`). The walls gave outward, and the
/// material against them stood over the samples just outside the window; the
/// grown area then activated those samples' full columns, so the pushed-in
/// volume counted twice and baked as a ridge a sample wide along the old
/// window's ring. Growth now takes the give of the walls that stop being
/// walls out of the old material first (`mpm_grow_shift`): the ring bakes as
/// the ungrown area's bake leaves it, the ground beyond at its own height.
#[test]
fn growing_an_area_over_its_walls_leaves_no_ridge_on_the_ring() {
    let base = demo_crater(8.0, 1.5);
    let grown = demo_crater_grown(8.0, 1.5, Some(1.0));
    let n = base.field.samples().0 as i64;
    let at = |i: i64, j: i64| (j * n + i) as usize;
    let inside = |i: i64, j: i64| base.lowered[at(i, j)] < -1.0;
    // The ring beyond the old window's +x edge, on every row of the window.
    let mut diffs = Vec::new();
    for j in 0..n {
        let Some(edge) = (0..n - 1).rev().find(|&i| inside(i, j)) else {
            continue;
        };
        let ring = edge + 1;
        let g = |i: i64| f64::from(grown.baked[at(i, j)]);
        diffs.push((
            j - n / 2,
            f64::from(grown.baked[at(ring, j)]) - f64::from(base.baked[at(ring, j)]),
            // The ring over the mean of the samples beside it on its row.
            g(ring) - 0.5 * (g(ring - 1) + g(ring + 1)),
        ));
    }
    let ridge = diffs.iter().map(|d| d.2).sum::<f64>() / diffs.len() as f64;
    // The bake's own row-to-row noise: the same measure (a sample over the
    // mean of its row neighbours, averaged over the rows) on the columns 2 to
    // 4 samples either side of the ring.
    let mut noise = 0.0f64;
    for off in [-4i64, -3, -2, 2, 3, 4] {
        let mut sum = 0.0;
        let mut rows = 0.0;
        for j in 0..n {
            let Some(edge) = (0..n - 1).rev().find(|&i| inside(i, j)) else {
                continue;
            };
            let i = edge + 1 + off;
            let g = |i: i64| f64::from(grown.baked[at(i, j)]);
            sum += g(i) - 0.5 * (g(i - 1) + g(i + 1));
            rows += 1.0;
        }
        noise = noise.max((sum / rows).abs());
    }
    // GROW_TRACE: the particles' own volume over each sample (rest volume
    // binned by the nearest sample, as height above the bed) against the
    // bakes, on the middle row about the old +x edge.
    if std::env::var("GROW_TRACE").is_ok() {
        let (sx, _) = base.field.cell_size();
        let x0 = base.field.vertex(0, 0);
        let mut bins = vec![0.0f64; (n * n) as usize];
        for q in &grown.particles {
            let i = ((q.xm[0] - x0.x) / sx).round() as i64;
            let j = ((q.xm[2] - x0.z) / sx).round() as i64;
            if (0..n).contains(&i) && (0..n).contains(&j) {
                bins[at(i, j)] += f64::from(q.f[0][3]) / f64::from(sx * sx);
            }
        }
        let j = n / 2;
        let edge = (0..n - 1).rev().find(|&i| inside(i, j)).unwrap();
        let bed = f64::from(grown.lowered[at(edge, j)]);
        for i in edge - 2..edge + 4 {
            eprintln!(
                "  sample {:+}: particles to {:.1} m, grown bake {:.1}, ungrown bake {:.1}",
                i - edge,
                bins[at(i, j)] + bed,
                grown.baked[at(i, j)],
                base.baked[at(i, j)]
            );
        }
    }
    let worst = diffs.iter().map(|d| d.1).fold(f64::MIN, f64::max);
    let mean = diffs.iter().map(|d| d.1).sum::<f64>() / diffs.len() as f64;
    let (dx, _) = base.field.cell_size();
    eprintln!(
        "ring past the old +x edge ({} rows, {dx:.1} m samples): grown minus ungrown bake mean \
         {mean:+.2} m, highest {worst:+.2} m; over its neighbours on the row {ridge:+.2} m \
         (the bake's noise on columns 2-4 either side: {noise:.2} m)",
        diffs.len()
    );
    // Counted twice (before 9282acfc), the pushed-in share of the 190 m
    // columns stood the ring 27.0 m over its row neighbours; with the give
    // taken out of the old material alone, the bake's two kernels still met
    // there (4.8 m: moved edge columns on P2G's B-spline beside new columns at
    // rest on the tent). With the ground's give carried on into the new
    // columns, both sides are moved material and the ring is as smooth as the
    // bake anywhere else. The grown bake stands higher than the ungrown one
    // over the ring (`mean`): the ground the walls compressed comes back as a
    // rise across the new columns, not folded onto the old edge.
    assert!(
        ridge.abs() <= noise + 0.25,
        "the ring stands {ridge:+.2} m over its row neighbours; the bake's noise is {noise:.2} m"
    );
    assert!(
        worst < 6.0,
        "the ring stands up to {worst:+.2} m over the ungrown bake"
    );
}
