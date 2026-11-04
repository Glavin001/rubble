//! The GPU port of PhysX's resident (native) Blast solve (`shaders/blast_native.slang`)
//! against its reference (`stress_ref::blast_bond`, `SolverKind::Native`): converged, the f64
//! forces; capped and warm-started frame after frame, the f32 reference within its own one-ulp
//! spread, iteration counts alike; free pieces, zero loads and the settled certificate as the
//! reference has them; bit-identical reruns.

use stress_gpu::blast::GpuBlastSolver;
use stress_gpu::gpu::Gpu;
use stress_ref::blast_bond::*;
use stress_ref::math::Vec3;

fn gpu() -> Option<Gpu> {
    Gpu::new().map_err(|_| eprintln!("no GPU; skipping")).ok()
}

fn house(settings: BlastSettings) -> (BlastGraph, Vec<BlastMaterial>, Vec3) {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/house-1story.json");
    let options = stress_ref::scene_pack::ImportOptions {
        authored_strengths: true,
        authored_fatal: true,
        pack_masses: true,
        pack_modulus: true,
        blast_authoring: true,
    };
    let scene = stress_ref::scene_pack::import_with(&path, "house", "house", options)
        .expect("house-1story");
    let (graph, materials) = BlastAuthoring::graph(&scene, settings).unwrap();
    (graph, materials, Vec3::from_array(scene.gravity))
}

fn pack(name: &str, settings: BlastSettings) -> (BlastGraph, Vec<BlastMaterial>, Vec3) {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../packs/{name}.json"));
    let options = stress_ref::capability::Profile::BlastGpu.import_options();
    let scene = stress_ref::scene_pack::import_with(&path, name, name, options).expect("pack");
    let (graph, materials) = BlastAuthoring::graph(&scene, settings).unwrap();
    (graph, materials, Vec3::from_array(scene.gravity))
}

#[test]
fn the_wide_solve_of_a_multi_block_island_tracks_the_reference() {
    let Some(gpu) = gpu() else { return };
    // villa-savoye's one island of 1045 nodes and 2464 bonds spans two node blocks and three
    // bond blocks of the many-group solve (blast_native_wide.slang): against the f32 reference
    // within its one-ulp spread, and against the one-group solve of the same island.
    let (graph, materials, g) = pack("villa-savoye", BlastSettings::native());
    let loads: Vec<_> = (0..12)
        .map(|f| uniform(graph.node_count(), g * if f < 6 { 1.0 } else { 1.3 }))
        .collect();
    let spread = rounding_spread(&graph, &loads);
    let bound = (10.0 * spread).max(1e-4);
    let mut reference = single(&graph);
    let mut wide = GpuBlastSolver::with_wide_min(&gpu, graph.clone(), materials.clone(), 0);
    let mut group = GpuBlastSolver::with_wide_min(&gpu, graph.clone(), materials, u32::MAX);
    let (mut worst, mut worst_group) = (0.0f64, 0.0f64);
    for (frame, accel) in loads.iter().enumerate() {
        let r = reference.solve(accel, false);
        let w = wide.solve(accel, false);
        group.solve(accel, false);
        for (k, (r, w)) in r.iter().zip(&w).enumerate() {
            assert_eq!(
                (r.skipped, r.converged),
                (w.skipped, w.converged),
                "frame {frame} island {k}: {r:?} vs {w:?}"
            );
        }
        let e = difference(&wide.impulses(), &reference_impulses(&reference));
        worst = worst.max(e);
        worst_group = worst_group.max(difference(&wide.impulses(), &group.impulses()));
        assert!(e < bound, "frame {frame}: {e:.3e} (bound {bound:.3e})");
    }
    println!("wide: worst {worst:.3e} from the f32 reference, {worst_group:.3e} from the one-group solve (one-ulp spread {spread:.3e})");
    assert!(worst_group < bound, "{worst_group:.3e}");
}

fn uniform(n: usize, lin: Vec3) -> Vec<AngLin> {
    vec![
        AngLin {
            ang: Vec3::ZERO,
            lin
        };
        n
    ]
}

fn difference(a: &[AngLin], b: &[AngLin]) -> f64 {
    let scale = b
        .iter()
        .map(|j| j.lin.norm().max(j.ang.norm()))
        .fold(0.0, f64::max)
        .max(1e-30);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.lin - y.lin).norm().max((x.ang - y.ang).norm()))
        .fold(0.0, f64::max)
        / scale
}

fn reference_impulses(r: &BlastSolver) -> Vec<AngLin> {
    (0..r.graph.bond_count()).map(|j| r.impulse(j)).collect()
}

fn single(graph: &BlastGraph) -> BlastSolver {
    let mut s = BlastSolver::new(graph.clone());
    s.device_precision = true;
    s
}

/// The f32 reference against itself with every load one ulp off (two patterns), frame by frame.
fn rounding_spread(graph: &BlastGraph, loads: &[Vec<AngLin>]) -> f64 {
    let eps = f32::EPSILON as f64;
    let mut worst = 0.0f64;
    for pattern in 0..2usize {
        let (mut a, mut b) = (single(graph), single(graph));
        for accel in loads {
            let nudged: Vec<AngLin> = accel
                .iter()
                .enumerate()
                .map(|(i, v)| AngLin {
                    ang: v.ang,
                    lin: v.lin * (1.0 + eps * if (i + pattern) % 2 == 0 { 1.0 } else { -1.0 }),
                })
                .collect();
            a.solve(accel, false);
            b.solve(&nudged, false);
            worst = worst.max(difference(&reference_impulses(&b), &reference_impulses(&a)));
        }
    }
    worst
}

#[test]
fn converged_native_solves_are_the_f64_answer() {
    let Some(gpu) = gpu() else { return };
    let (graph, materials, g) = house(BlastSettings {
        max_iterations: 2000,
        tolerance: 1e-4,
        ..BlastSettings::native()
    });
    let accel = uniform(graph.node_count(), g);
    let mut reference = BlastSolver::new(graph.clone());
    let r = reference.solve(&accel, false);
    let mut device = GpuBlastSolver::new(&gpu, graph, materials);
    let d = device.solve(&accel, false);
    let e = difference(&device.impulses(), &reference_impulses(&reference));
    let mut f32_reference = single(&reference.graph);
    let s = f32_reference.solve(&accel, false);
    println!(
        "converged: {e:.3e}; iterations {} vs {}; f32 reference {s:?}",
        r[0].iterations, d[0].iterations
    );
    // The f64 solve converges. In f32 the residual test has a floor on this house far above the
    // tolerance (it reads |B^T r| from a difference of nearly equal terms) while the forces are
    // within 1e-5 of f64: the f32 reference stalls the same way (stressgpu/detail/README.md: "on
    // stiff structures it can read 1e3x over tolerance with forces within 1e-3", the reason for
    // the force tolerance). So the device is judged by its forces.
    assert!(r[0].converged, "{r:?}");
    assert!(e < 1e-4, "{e:.3e}");
    assert_eq!(
        d[0].converged, s[0].converged,
        "the device and the f32 reference agree on convergence: {d:?} vs {s:?}"
    );
}

#[test]
fn capped_native_frames_track_the_f32_reference() {
    let Some(gpu) = gpu() else { return };
    // vibe-land's Mac budget: 16 iterations a frame, warm-started, the settled certificate on.
    capped_frames(&gpu, BlastSettings::native(), u32::MAX, |_| {});
}

#[test]
fn the_polynomial_first_tracks_the_f32_reference_in_one_group_and_many() {
    let Some(gpu) = gpu() else { return };
    // BLAST_STRESS_FIRST_PRECONDITIONER=polynomial: no steepest-descent first step; and
    // `warm`, which differs from it only on a cold solve.
    for first in [FirstPreconditioner::Polynomial, FirstPreconditioner::Warm] {
        let settings = BlastSettings { first_preconditioner: first, ..BlastSettings::native() };
        capped_frames(&gpu, settings, u32::MAX, |_| {});
        capped_frames(&gpu, settings, 0, |_| {});
    }
}

/// Capped warm frames on the house, the device (one group per island, or every island over
/// `wide_min` nodes on many) against the f32 reference.
fn capped_frames(gpu: &Gpu, settings: BlastSettings, wide_min: u32, edit: fn(&mut BlastGraph)) {
    let (mut graph, materials, g) = house(settings);
    edit(&mut graph);
    let loads: Vec<_> = (0..24)
        .map(|f| uniform(graph.node_count(), g * if f < 12 { 1.0 } else { 1.2 }))
        .collect();
    let spread = rounding_spread(&graph, &loads);
    let bound = (10.0 * spread).max(1e-4);
    let mut reference = single(&graph);
    let mut device = GpuBlastSolver::with_wide_min(gpu, graph, materials, wide_min);
    let mut worst = 0.0f64;
    for (frame, accel) in loads.iter().enumerate() {
        let r = reference.solve(accel, false);
        let d = device.solve(accel, false);
        for (k, (r, d)) in r.iter().zip(&d).enumerate() {
            // At f32's residual floor the verdict is rounding (stressgpu/detail/README.md): a
            // solve that stopped short within 2x of its target may be judged converged by the
            // other order of sums (WGSL under the driver's fast math meets it). The forces,
            // below, must agree either way.
            let floor = r.converged != d.converged && r.residual2.max(d.residual2) <= 2.0 * r.target2;
            // Converged a frame apart there, one side takes its settled certificate a frame
            // before the other, which verifies the same output in 0 iterations.
            let certified = (r.skipped && d.converged && d.iterations == 0) || (d.skipped && r.converged && r.iterations == 0);
            assert!(
                (r.skipped == d.skipped || certified) && (r.converged == d.converged || floor),
                "frame {frame} island {k}: {r:?} vs {d:?}"
            );
            // A warm frame that converges at the edge of the tolerance may stop a few iterations
            // either side, or many when one sum order lands just under it (the forces, below,
            // agree either way).
            let edge = floor
                || (r.converged && d.converged && r.residual2.max(d.residual2) >= 0.5 * r.target2);
            assert!(
                edge || r.iterations.abs_diff(d.iterations) <= 2 + r.iterations / 5,
                "frame {frame} island {k}: {r:?} vs {d:?}"
            );
        }
        let e = difference(&device.impulses(), &reference_impulses(&reference));
        worst = worst.max(e);
        assert!(e < bound, "frame {frame}: {e:.3e} (bound {bound:.3e})");
    }
    println!(
        "capped frames ({:?}, wide_min {wide_min}): worst {worst:.3e} from the f32 reference (one-ulp spread {spread:.3e})",
        settings.first_preconditioner
    );
}

#[test]
fn a_free_pair_and_a_zero_load_match_the_reference() {
    let Some(gpu) = gpu() else { return };
    let node = |x: f64, m: f64| BlastNode {
        position: Vec3::new(x, 0.0, 0.0),
        mass: m,
        inertia: 0.0,
        volume: 1.0,
    };
    let nodes = vec![node(0.0, 3.0), node(1.0, 5.0), node(2.0, 2.0)];
    let bond = |a: usize, b: usize| BlastBond {
        nodes: [a, b],
        centroid: (nodes[a].position + nodes[b].position) * 0.5,
        normal: Vec3::X,
        area: 0.5,
        modulus: 0.0,
        material: 0,
    };
    let bonds = vec![bond(0, 1), bond(1, 2)];
    let graph = BlastGraph::new(
        nodes.clone(),
        bonds,
        BlastSettings {
            max_iterations: 200,
            tolerance: 1e-6,
            ..BlastSettings::native()
        },
    )
    .unwrap();
    // Pulled apart at both ends: tension through both bonds, nothing into the rigid modes.
    let f = 60.0;
    let accel = vec![
        AngLin {
            ang: Vec3::ZERO,
            lin: Vec3::new(-f / 3.0, 0.0, 0.0),
        },
        AngLin::ZERO,
        AngLin {
            ang: Vec3::ZERO,
            lin: Vec3::new(f / 2.0, 0.0, 0.0),
        },
    ];
    let mut reference = BlastSolver::new(graph.clone());
    reference.solve(&accel, false);
    let mut device = GpuBlastSolver::new(&gpu, graph, vec![BlastMaterial::CONCRETE]);
    let d = device.solve(&accel, false);
    assert!(d[0].converged, "{d:?}");
    let e = difference(&device.impulses(), &reference_impulses(&reference));
    assert!(e < 1e-4, "{e:.3e}");
    assert!(
        device
            .impulses()
            .iter()
            .all(|j| (j.lin.x.abs() - f).abs() < 1e-3 * f),
        "{:?}",
        device.impulses()
    );
    // A zero load: zero forces at once.
    let z = device.solve(&uniform(3, Vec3::ZERO), false);
    assert!(z[0].converged && z[0].iterations == 0, "{z:?}");
    assert!(device.impulses().iter().all(|j| j.norm2() == 0.0));
}

#[test]
fn the_settled_certificate_skips_as_the_reference_does() {
    let Some(gpu) = gpu() else { return };
    let (graph, materials, g) = house(BlastSettings {
        max_iterations: 200,
        ..BlastSettings::native()
    });
    let accel = uniform(graph.node_count(), g);
    let mut reference = single(&graph);
    let mut device = GpuBlastSolver::new(&gpu, graph, materials);
    for frame in 0..6 {
        let r = reference.solve(&accel, false);
        let d = device.solve(&accel, false);
        assert_eq!(
            (r[0].skipped, r[0].converged, r[0].iterations == 0),
            (d[0].skipped, d[0].converged, d[0].iterations == 0),
            "frame {frame}: {r:?} vs {d:?}"
        );
    }
    assert!(device.reports()[0].skipped, "settled by frame 6");
}

#[test]
fn a_native_rerun_is_bit_identical() {
    let Some(gpu) = gpu() else { return };
    let (graph, materials, g) = house(BlastSettings::native());
    let accel = uniform(graph.node_count(), g);
    let run = || {
        let mut s = GpuBlastSolver::new(&gpu, graph.clone(), materials.clone());
        for _ in 0..3 {
            s.solve(&accel, false);
        }
        s.solution()
    };
    let (a, b) = (run(), run());
    let bits = |v: &[AngLin]| {
        v.iter()
            .flat_map(|x| x.ang.to_array().into_iter().chain(x.lin.to_array()))
            .map(f64::to_bits)
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&a), bits(&b));
}

/// The house with per-bond rotational stiffness (each bond's square patch, as the stage takes
/// it without section data); with `shear`, also half the shear stiffness of the normal one.
fn stiffen(graph: &mut BlastGraph, shear: bool) {
    graph.set_rotational_stiffness(None).unwrap();
    if shear {
        graph.set_shear_stiffness(&vec![0.5; graph.bond_count()]).unwrap();
    }
}

#[test]
fn per_bond_stiffness_matches_the_reference() {
    let Some(gpu) = gpu() else { return };
    // Rotational stiffness (s A on the angular rows: the Schur block with W), and with the shear
    // stiffness (s Al on the linear rows: the whole 6x6 block and its inverse): converged forces
    // against f64; capped warm frames against f32, in one group and in many.
    for shear in [false, true] {
        let (mut graph, materials, g) = house(BlastSettings {
            max_iterations: 4000,
            tolerance: 1e-4,
            ..BlastSettings::native()
        });
        stiffen(&mut graph, shear);
        let accel = uniform(graph.node_count(), g);
        let mut reference = BlastSolver::new(graph.clone());
        let r = reference.solve(&accel, false);
        let mut device = GpuBlastSolver::new(&gpu, graph.clone(), materials.clone());
        device.solve(&accel, false);
        let e = difference(&device.impulses(), &reference_impulses(&reference));
        println!("stiffness (shear {shear}): converged {e:.3e} from f64 ({} iterations)", r[0].iterations);
        assert!(r[0].converged, "{r:?}");
        assert!(e < 1e-4, "shear {shear}: {e:.3e}");
        // The stiffness changes the answer: not the uniform solve's forces.
        let (plain, _, _) = house(BlastSettings { max_iterations: 4000, tolerance: 1e-4, ..BlastSettings::native() });
        let mut uniform_solve = BlastSolver::new(plain);
        uniform_solve.solve(&accel, false);
        assert!(difference(&reference_impulses(&uniform_solve), &reference_impulses(&reference)) > 1e-2);
    }
    capped_frames(&gpu, BlastSettings::native(), u32::MAX, |g| stiffen(g, false));
    capped_frames(&gpu, BlastSettings::native(), 0, |g| stiffen(g, false));
    capped_frames(&gpu, BlastSettings::native(), u32::MAX, |g| stiffen(g, true));
    capped_frames(&gpu, BlastSettings::native(), 0, |g| stiffen(g, true));
}

#[test]
fn krylov_carry_tracks_the_f32_reference_in_one_group_and_many() {
    let Some(gpu) = gpu() else { return };
    // BLAST_STRESS_CARRY_KRYLOV=1 without the rotational stiffness (with it, the default, it runs
    // in per_bond_stiffness_matches_the_reference): capped warm frames continue each
    // unconverged solve's recurrence, on the device as in the reference.
    let settings = BlastSettings { carry_krylov: Some(true), ..BlastSettings::native() };
    capped_frames(&gpu, settings, u32::MAX, |_| {});
    capped_frames(&gpu, settings, 0, |_| {});
}

#[test]
fn section_bending_bearing_and_friction_match_the_reference_formula() {
    let Some(gpu) = gpu() else { return };
    // The stage's material evaluation with its opt-in features: section bending (half the bonds
    // with a section of their own, the rest the square patch, the moment moved to the
    // centroid), bearing joints on some, Mohr-Coulomb friction, and limits low enough that
    // gravity overloads bonds; against the reference's formula and law on the device's impulses.
    let (mut graph, mut materials, g) = house(BlastSettings { section_bending: true, ..BlastSettings::native() });
    for m in &mut materials {
        for v in [&mut m.compression_elastic, &mut m.compression_fatal, &mut m.tension_elastic, &mut m.tension_fatal, &mut m.shear_elastic, &mut m.shear_fatal] {
            *v *= 1e-3;
        }
        m.shear_friction = 0.4;
        m.shear_capacity_limit = 2.0 * m.shear_fatal;
    }
    graph.sections = (0..graph.bond_count())
        .map(|j| {
            let n = graph.normal[j];
            let axis = n.cross(if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y }).normalized();
            let side = graph.bonds[j].area.sqrt();
            let modulus = side * side * side / 6.0;
            match j % 4 {
                0 => BondSection { axis, bend_modulus0: modulus, bend_modulus1: 2.0 * modulus, twist_modulus: 1.5 * modulus, ..Default::default() },
                1 => BondSection { axis, bearing_depth0: 0.5 * side, bearing_depth1: 0.25 * side, ..Default::default() },
                _ => BondSection { axis, ..Default::default() },
            }
        })
        .collect();
    let mut device = GpuBlastSolver::new(&gpu, graph.clone(), materials.clone());
    let accel = uniform(graph.node_count(), g);
    for _ in 0..4 {
        device.solve(&accel, false);
    }
    let dt = 1.0 / 60.0;
    let results = device.bond_results(dt, false);
    let healths = device.health();
    let (mut commands, mut agree) = (0, 0);
    for (j, d) in results.iter().enumerate() {
        let health = healths[j];
        if !(health > 0.0) {
            continue;
        }
        let r = graph.bond_stress_of(j, d.impulse, health);
        let scale = r.compression.max(r.tension).max(r.shear).max(r.normal.abs()).max(1e-30);
        for (a, b, what) in [
            (d.stress.normal, r.normal, "normal"),
            (d.stress.shear, r.shear, "shear"),
            (d.stress.bend, r.bend, "bend"),
            (d.stress.compression, r.compression, "compression"),
            (d.stress.tension, r.tension, "tension"),
        ] {
            assert!((a - b).abs() <= 1e-5 * scale, "bond {j} {what}: {a} vs {b}");
        }
        let material = &materials[graph.bonds[j].material];
        let rd = bond_damage(r, health, graph.bonds[j].area, material, dt, graph.settings.damage_rate);
        commands += rd.command as usize;
        agree += (rd.command == d.damage.command) as usize;
    }
    println!("section bending: {agree} of the verdicts agree; {commands} commands");
    assert!(commands > 0, "the limits are low enough to command damage");
    let live = healths.iter().filter(|h| **h > 0.0).count();
    assert!(agree + 2 >= live, "{agree} of {live}");
}

#[test]
fn the_wide_solve_stops_on_the_force_test_as_the_one_group_solve_does() {
    let Some(gpu) = gpu() else { return };
    // vibe-land's force tolerance (1e-3): in f32 the house's residual never reaches its target,
    // so the force test is what stops a solve, and its screen measures the step against the
    // forces at the start plus the steps since. The many-group solve must start from the
    // forces as the one-group solve does: it once started from 0 and never stopped.
    let settings = BlastSettings { force_tolerance: 1e-3, ..BlastSettings::native() };
    let (graph, materials, g) = house(settings);
    let accel = uniform(graph.node_count(), g);
    let mut wide = GpuBlastSolver::with_wide_min(&gpu, graph.clone(), materials.clone(), 0);
    let mut group = GpuBlastSolver::with_wide_min(&gpu, graph, materials, u32::MAX);
    let (mut stopped_wide, mut stopped_group) = (0, 0);
    for frame in 0..60 {
        let w = wide.solve(&accel, false);
        let o = group.solve(&accel, false);
        stopped_wide += (w[0].converged && w[0].iterations < 16) as usize;
        stopped_group += (o[0].converged && o[0].iterations < 16) as usize;
        if frame >= 20 {
            assert_eq!((w[0].converged, w[0].skipped), (o[0].converged, o[0].skipped), "frame {frame}: {w:?} vs {o:?}");
            assert!(w[0].iterations.abs_diff(o[0].iterations) <= 1, "frame {frame}: {w:?} vs {o:?}");
        }
    }
    println!("frames stopped early by the force test: many groups {stopped_wide}, one group {stopped_group}");
    assert!(stopped_group > 30 && stopped_wide > 30);
}
