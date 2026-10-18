//! The scenes rubble's stress pipeline is benchmarked and viewed on (`stress_bench`,
//! `destruction-view`), in the real-time configuration both run them in.

use std::path::{Path, PathBuf};

use stress_ref::scene::{Methods, Scene, SolveMode};

/// A12's real-time configuration: adaptive, established methods, the certified true
/// step, contact safety 0.35 and stress safety 0.5; `seconds` of simulated time.
pub fn realtime(mut s: Scene, seconds: f64) -> Scene {
    s.sim.methods = Methods::established();
    s.sim.methods.true_step = true;
    s.sim.courant_safety = 0.35;
    s.sim.stress_courant_safety = Some(0.5);
    s.sim.solve_mode = SolveMode::Adaptive;
    s.sim.duration = seconds;
    s
}

/// The oracle's packs (`engine/stress-oracle/packs`).
pub fn packs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs")
}

/// A free concrete wall (10 x 2 x 11 chunks of 0.1 m, no gravity) hit by a 200 kg steel
/// ram at 40 m/s.
pub fn wall_ram() -> Scene {
    use stress_ref::builders::*;
    use stress_ref::math::Vec3;
    use stress_ref::scene::*;
    let mut s = new_scene("wall_ram", "free concrete wall hit by a steel ram");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let chunks = grid(Vec3::new(-0.5, 0.0, -0.1), [10, 2, 11], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.1, 0.1, 0.1] },
        mass: 200.0,
        position: [0.0, -0.15, 0.5],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 40.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s
}

/// `wall_ram` with the ram 1.1 m further back: it strikes in the second frame (the first a
/// resident frame can take, `GpuWorld::fast_qs_conditions`).
pub fn wall_ram_late() -> Scene {
    let mut s = wall_ram();
    s.name = "wall_ram_late".into();
    s.impactors[0].position[1] -= 1.1;
    s
}

/// `n` free walls of `wall_ram`'s, 0.1 m apart along the ram's path (the wall row of
/// findings/rewind-scaling.md, `wall_row(n, 40, 0.1)`), with `wall_ram_late`'s ram: it
/// strikes in the second frame.
pub fn wall_row_late(n: usize) -> Scene {
    use stress_ref::builders::*;
    use stress_ref::math::Vec3;
    let mut s = wall_ram_late();
    s.name = format!("wall_row{n}_late");
    s.bodies.clear();
    for k in 0..n {
        let y = k as f64 * (0.2 + 0.1);
        let chunks = grid(Vec3::new(-0.5, y, -0.1), [10, 2, 11], Vec3::splat(0.1), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        s.bodies.push(body(&format!("wall{k}"), chunks, bonds));
    }
    s
}

/// The house-1story pack.
pub fn house() -> Scene {
    stress_ref::scene_pack::import(&packs_dir().join("house-1story.json"), "house-1story", "structure").expect("house-1story pack")
}
