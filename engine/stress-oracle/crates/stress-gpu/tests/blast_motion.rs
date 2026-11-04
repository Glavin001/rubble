//! The native solve's rigid-mode frames on the GPU (`shaders/blast_motion.slang`, PhysX's
//! `StressMotionPair*.cuh` construction) against their reference (`stress_ref::blast_bond::
//! rigid_frames`, the binary64 construction): anchored islands, free islands with all three
//! rotations, and cycles that close on real moments (one rotation, or none). The float-pair
//! and expansion arithmetic it rests on is checked on each shader path first.

use stress_gpu::blast::{GpuBlastSolver, Params};
use stress_gpu::gpu::{Binding, Gpu, ShaderPath};
use stress_ref::blast_bond::*;
use stress_ref::math::Vec3;

fn gpu() -> Option<Gpu> {
    Gpu::new().map_err(|_| eprintln!("no GPU; skipping")).ok()
}

fn pack(name: &str) -> (BlastGraph, Vec<BlastMaterial>) {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../packs/{name}.json"));
    let options = stress_ref::capability::Profile::BlastGpu.import_options();
    let scene = stress_ref::scene_pack::import_with(&path, name, name, options).expect("pack");
    BlastAuthoring::graph(&scene, BlastSettings::native()).unwrap()
}

/// The shader paths this device can run the kernels in.
fn paths(gpu: &Gpu) -> Vec<(ShaderPath, Gpu)> {
    [
        ShaderPath::Wgsl,
        ShaderPath::MetalLib,
        ShaderPath::MetalSource,
    ]
    .into_iter()
    .filter_map(|p| {
        let mut g = gpu.clone();
        g.set_shader_path(p).then_some((p, g))
    })
    .collect()
}

#[test]
fn two_sum_and_two_product_are_exact_on_every_shader_path() {
    let Some(gpu) = gpu() else { return };
    let cases: [(f32, f32, f32, f32); 5] = [
        (1.0, 1.0e-8, 1.0 + f32::EPSILON, 1.0 - f32::EPSILON),
        (1.0e8, -1.0 + 1.0e-7, 0.1, 0.3),
        (3.0, 9.313226e-10, 100001.0, 99999.0),
        (-2.5e-3, 7.1e3, 1.0 / 3.0, 3.0),
        (0.1, 0.2, 1.000_000_1, 1.000_000_1),
    ];
    let mut wrong = Vec::new();
    for (path, gpu) in paths(&gpu) {
        let rows = cases.len();
        let mut words = vec![[0.0f32; 4]; 2 * rows];
        for (i, c) in cases.iter().enumerate() {
            words[2 * i] = [c.0, c.1, c.2, c.3];
        }
        use Binding::{Storage as R, StorageRw as W, Uniform as U};
        let k = gpu.compute(
            &stress_gpu::shaders::BLAST_MOTION,
            "blast_motion_selftest",
            &[U, W, W, R, R, R, W],
        );
        let params = gpu.uniform(
            "params",
            &Params {
                node_count: rows as u32,
                ..Default::default()
            },
        );
        let dummies: Vec<wgpu::Buffer> =
            (0..5).map(|_| gpu.storage("dummy", &[0u32; 64])).collect();
        let motion = gpu.storage("motion", &words);
        let bind = gpu.bind(
            &k.group,
            &[
                &params,
                &dummies[0],
                &dummies[1],
                &dummies[2],
                &dummies[3],
                &dummies[4],
                &motion,
            ],
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        let out: Vec<[f32; 4]> = gpu.read_range(&motion, 0, 2 * rows);
        for (i, c) in cases.iter().enumerate() {
            let [s, e, p, pe] = out[2 * i + 1];
            let sum = c.0 as f64 + c.1 as f64;
            let product = c.2 as f64 * c.3 as f64;
            if s != c.0 + c.1 || s as f64 + e as f64 != sum {
                wrong.push(format!("{path:?} case {i}: two-sum {s} + {e} for {sum}"));
            }
            if p != c.2 * c.3 || p as f64 + pe as f64 != product {
                wrong.push(format!(
                    "{path:?} case {i}: two-product {p} + {pe} for {product}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Solve once with no load (the islands and their frames are built before it).
fn build(solver: &mut GpuBlastSolver) {
    let accel = vec![AngLin::ZERO; solver.graph.node_count()];
    solver.solve(&accel, false);
}

fn compare(gpu: &[RigidFrame], reference: &[RigidFrame], what: &str) {
    assert_eq!(gpu.len(), reference.len(), "{what}: island count");
    for (k, (g, r)) in gpu.iter().zip(reference).enumerate() {
        assert_eq!(
            (g.anchored, g.rotations, g.error),
            (r.anchored, r.rotations, r.error),
            "{what}: island {k}"
        );
        if g.anchored {
            continue;
        }
        let close = |a: f64, b: f64, scale: f64| (a - b).abs() <= 1e-5 * scale.max(1.0);
        if g.rotations == 1 {
            for c in 0..3 {
                assert!(
                    close(g.axis[c], r.axis[c], 1.0),
                    "{what}: island {k} axis {:?} vs {:?}",
                    g.axis,
                    r.axis
                );
            }
        }
        let d = g.rotations + 3;
        for i in 0..d {
            assert!(
                close(g.scale[i], r.scale[i], r.scale[i].abs()),
                "{what}: island {k} scale {i}: {} vs {}",
                g.scale[i],
                r.scale[i]
            );
        }
        for t in 0..d * (d + 1) / 2 {
            assert!(
                close(g.factor[t], r.factor[t], 1.0),
                "{what}: island {k} factor {t}: {} vs {}",
                g.factor[t],
                r.factor[t]
            );
        }
        assert_eq!(g.relative.len(), r.relative.len());
        let extent = r
            .relative
            .iter()
            .flat_map(|(_, q)| q.map(f64::abs))
            .fold(0.0, f64::max);
        for ((ng, qg), (nr, qr)) in g.relative.iter().zip(&r.relative) {
            assert_eq!(ng, nr);
            for c in 0..3 {
                assert!(
                    close(qg[c], qr[c], extent),
                    "{what}: island {k} node {ng}: {qg:?} vs {qr:?}"
                );
            }
        }
    }
}

fn reference(solver: &GpuBlastSolver) -> Vec<RigidFrame> {
    rigid_frames(&solver.graph, &solver.islands(), &solver.health())
}

/// Every bond to a support broken: the structure stands free.
fn free(solver: &mut GpuBlastSolver) {
    let g = solver.graph.clone();
    for (j, b) in g.bonds.iter().enumerate() {
        if b.nodes.iter().any(|&v| !(g.weight[v].lin > 0.0)) {
            solver.set_health(j, 0.0);
        }
    }
}

#[test]
fn frames_match_the_reference_anchored_and_free() {
    let Some(gpu) = gpu() else { return };
    for ((path, gpu), name) in paths(&gpu)
        .into_iter()
        .flat_map(|p| ["house-1story", "villa-savoye"].map(|n| (p.clone(), n)))
    {
        let name = &format!("{name} ({path:?})");
        let (graph, materials) = pack(name.split(' ').next().unwrap());
        let mut solver = GpuBlastSolver::new(&gpu, graph, materials);
        build(&mut solver);
        let frames = solver.rigid_frames();
        assert!(
            frames.iter().all(|f| f.anchored),
            "{name}: standing on its supports"
        );
        compare(&frames, &reference(&solver), name);
        free(&mut solver);
        build(&mut solver);
        let frames = solver.rigid_frames();
        assert!(
            frames.iter().any(|f| !f.anchored && f.rotations == 3),
            "{name}: free pieces"
        );
        compare(&frames, &reference(&solver), &format!("{name} free"));
    }
}

#[test]
fn cycles_closing_on_real_moments_leave_one_rotation_or_none() {
    let Some(gpu) = gpu() else { return };
    let (mut graph, materials) = pack("house-1story");
    // Offsets on a dyadic grid, consistent with grid positions: every cycle closes exactly.
    // Then bonds bent off their midpoints by a dyadic step: every cycle through one closes on
    // the same moment (one rotation about it); through two non-collinear ones, none. (Offsets
    // from real positions leave each cycle its own rounding residue, so two closures through
    // one bent bond are never exactly collinear: the target then keeps no rotation.)
    let grid = |v: Vec3| {
        let snap = |x: f64| (x * 1024.0).round() / 1024.0;
        Vec3::new(snap(v.x), snap(v.y), snap(v.z))
    };
    let scale = 1.0 / graph.length_scale;
    for j in 0..graph.bond_count() {
        let [a, b] = graph.bonds[j].nodes;
        let half =
            (grid(graph.nodes[b].position * scale) - grid(graph.nodes[a].position * scale)) * 0.5;
        graph.offset0[j] = half;
        graph.offset1[j] = half * -1.0;
    }
    let inner: Vec<usize> = (0..graph.bond_count())
        .filter(|&j| {
            graph.bonds[j]
                .nodes
                .iter()
                .all(|&v| graph.weight[v].lin > 0.0)
        })
        .collect();
    let (a, b) = (inner[inner.len() / 3], inner[2 * inner.len() / 3]);
    let consistent = graph.clone();
    graph.offset0[a] += Vec3::new(0.0, 0.0, 0.0625);
    let one = graph.clone();
    graph.offset0[b] += Vec3::new(0.0625, 0.0, 0.0);
    for ((path, gpu), (what, g, rotations)) in paths(&gpu).into_iter().flat_map(|p| {
        [
            ("no closure", consistent.clone(), 3),
            ("one closure", one.clone(), 1),
            ("two closures", graph.clone(), 0),
        ]
        .map(|c| (p.clone(), c))
    }) {
        let what = &format!("{what} ({path:?})");
        let mut solver = GpuBlastSolver::new(&gpu, g, materials.clone());
        free(&mut solver);
        build(&mut solver);
        let frames = solver.rigid_frames();
        let main = frames.iter().max_by_key(|f| f.relative.len()).unwrap();
        assert_eq!(
            (main.anchored, main.rotations, main.error),
            (false, rotations, 0),
            "{what}"
        );
        compare(&frames, &reference(&solver), what);
    }
}
