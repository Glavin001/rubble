//! The event timelines of the GPU world and the reference side by side (cracks, creaks,
//! breaks, splits), with the clusters' activity at the end.
//!
//!   stress-gpu-events catalogue-name [--set path=value ...] [--duration S] [--lockstep]
//!
//! `--lockstep`: step both frame by frame and report each frame where the clusters'
//! activity or count, or the substeps, differ (adaptive-mode decisions).

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::solver::SolverEvent;
use stress_ref::world::World;

fn line(e: &SolverEvent) -> String {
    match e {
        SolverEvent::Cracked { time, bond, mode, .. } => format!("{time:10.5} cracked b{bond} {mode:?}"),
        SolverEvent::Creaked { time, bond, .. } => format!("{time:10.5} creaked b{bond}"),
        SolverEvent::Broken { time, bond, mode, .. } => format!("{time:10.5} broken  b{bond} {mode:?}"),
        other => format!("{other:?}").chars().take(60).collect(),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args.next().expect("catalogue name");
    let mut scene = stress_ref::builders::catalog()
        .into_iter()
        .chain(stress_ref::showcases::catalog())
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no catalogue scene {name}"));
    let mut lockstep = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--lockstep" => lockstep = true,
            "--set" => {
                let o = args.next().expect("--set path=value");
                let (path, value) = o.split_once('=').expect("--set path=value");
                scene = scene.with_override(path, value).unwrap_or_else(|e| panic!("--set {o}: {e}"));
            }
            "--duration" => scene.sim.duration = args.next().and_then(|v| v.parse().ok()).expect("--duration S"),
            other => panic!("unknown argument {other}"),
        }
    }
    let gpu = Gpu::new().expect("GPU");
    let mut ours = GpuWorld::new(&gpu, &scene).expect("GPU world");
    let mut reference = World::new(&scene);
    if lockstep {
        let frames = (scene.sim.duration / scene.sim.frame_dt).round() as u64;
        let mut reported = 0;
        let describe = |cl: &[stress_ref::solver::Cluster]| cl.iter().map(|c| format!("{:?}:{:.4}", c.activity, c.active_timer)).collect::<Vec<_>>().join(" ");
        let ke = |m: &stress_ref::solver::ReferenceSolver| -> f64 {
            m.clusters
                .iter()
                .flat_map(|cl| cl.chunks.iter().map(move |&c| (cl.structure, c)))
                .map(|(s, c)| {
                    let (ch, st) = (&m.structures[s].chunks[c], &m.chunks[s][c]);
                    0.5 * ch.mass * st.v.norm2() + 0.5 * st.w.dot(ch.inertia * st.w)
                })
                .sum()
        };
        for f in 0..frames {
            reference.step_frame();
            ours.step_frame(&gpu).expect("frame");
            let (ra, ga) = (describe(&reference.solver.clusters), describe(&ours.solver.mirror.clusters));
            let (rs, gs) = (reference.solver.substeps, ours.solver.mirror.substeps);
            if ra != ga || rs != gs {
                println!("frame {f} t={:.4}: substeps {rs} | {gs}; ke {:.3e} | {:.3e}", reference.solver.time, ke(&reference.solver), ke(&ours.solver.mirror));
                println!("  ref {ra}");
                println!("  gpu {ga}");
                reported += 1;
                if reported >= 8 {
                    break;
                }
            }
        }
        return;
    }
    ours.run(&gpu).expect("run");
    reference.run();
    let (a, b) = (&reference.solver.events, &ours.solver.mirror.events);
    println!("{:<44} | {}", "reference", "gpu");
    for i in 0..a.len().max(b.len()) {
        let l = a.get(i).map(line).unwrap_or_default();
        let r = b.get(i).map(line).unwrap_or_default();
        println!("{l:<44} | {r}");
    }
    let activity = |cl: &[stress_ref::solver::Cluster]| cl.iter().map(|c| format!("{:?}", c.activity)).collect::<Vec<_>>().join(",");
    println!("activity: reference {} | gpu {}", activity(&reference.solver.clusters), activity(&ours.solver.mirror.clusters));
}
