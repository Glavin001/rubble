//! The reference's own spread of a scene's outcome (broken bonds, fragments): the
//! reference run with its gravity and impactor and body velocities scaled by 1 + eps
//! for a few eps, and at half its substep; with the GPU's outcome beside them.
//!
//!   stress-gpu-spread catalogue-name [--set path=value ...] [--duration S]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::observation::Observation;
use stress_ref::scene::Scene;
use stress_ref::world::World;

fn perturbed(scene: &Scene, eps: f64) -> Scene {
    let mut s = scene.clone();
    let k = 1.0 + eps;
    s.gravity = s.gravity.map(|g| g * k);
    for imp in &mut s.impactors {
        imp.velocity = imp.velocity.map(|v| v * k);
    }
    for b in &mut s.bodies {
        b.linear_velocity = b.linear_velocity.map(|v| v * k);
    }
    s
}

fn outcome(o: &Observation) -> String {
    let v = |k: &str| o.values.get(k).copied().unwrap_or(0.0);
    format!("broken {:4.0}  fragments {:4.0}  dissipation {:10.3}", v("broken_bonds"), v("fragments"), v("bond_dissipation"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args.next().expect("catalogue name");
    let mut scene = stress_ref::builders::catalog()
        .into_iter()
        .chain(stress_ref::showcases::catalog())
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no catalogue scene {name}"));
    while let Some(a) = args.next() {
        match a.as_str() {
            "--set" => {
                let o = args.next().expect("--set path=value");
                let (path, value) = o.split_once('=').expect("--set path=value");
                scene = scene.with_override(path, value).unwrap_or_else(|e| panic!("--set {o}: {e}"));
            }
            "--duration" => scene.sim.duration = args.next().and_then(|v| v.parse().ok()).expect("--duration S"),
            other => panic!("unknown argument {other}"),
        }
    }
    for eps in [0.0, 1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 2e-4, -2e-4] {
        println!("reference eps {eps:+.0e}: {}", outcome(&World::new(&perturbed(&scene, eps)).run()));
    }
    let mut half = scene.clone();
    half.sim.scale_safety(0.5);
    println!("reference half step:  {}", outcome(&World::new(&half).run()));
    let gpu = Gpu::new().expect("GPU");
    let o = GpuWorld::new(&gpu, &scene).and_then(|mut w| w.run(&gpu)).expect("GPU world");
    println!("gpu:                  {}", outcome(&o));
}
