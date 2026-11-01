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

/// `scene` with its impactors `frames` frames (of 1/60 s) further back along their
/// velocity: they strike that many frames later (a GPU check at the end of a resident frame
/// then sees the strike coming).
pub fn later(mut s: Scene, frames: usize) -> Scene {
    s.name = format!("{}{}", s.name, frames);
    for imp in &mut s.impactors {
        for a in 0..3 {
            imp.position[a] -= imp.velocity[a] * frames as f64 / 60.0;
        }
    }
    s
}

/// The house-1story pack.
pub fn house() -> Scene {
    stress_ref::scene_pack::import(&packs_dir().join("house-1story.json"), "house-1story", "structure").expect("house-1story pack")
}

/// A concrete wall for shots (the fracture world's models: `stress_bench` `wall_shot`,
/// tests/fracture_models.rs): 1 m wide (10 chunks of 0.1 m along x), `layers` chunks thick
/// along +y from y = 0, 1.1 m tall (11), under gravity; its bottom row fixed when
/// `anchored`. No impactor: shots are launched ([`shot_at`]).
pub fn shot_wall(layers: usize, anchored: bool) -> Scene {
    shot_row(1, layers, anchored)
}

/// `n` walls of [`shot_wall`]'s, 0.3 m apart along +y.
pub fn shot_row(n: usize, layers: usize, anchored: bool) -> Scene {
    use stress_ref::builders::*;
    use stress_ref::math::Vec3;
    use stress_ref::scene::Support;
    let mut s = new_scene(&format!("shot_row{n}x{layers}"), "concrete walls for launched shots");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("projectile".into(), elastic_steel());
    let counts = [10, layers, 11];
    for k in 0..n {
        let y = k as f64 * (0.1 * layers as f64 + 0.3);
        let mut chunks = grid(Vec3::new(-0.5, y, 0.0), counts, Vec3::splat(0.1), "concrete");
        if anchored {
            for i in 0..counts[0] {
                for j in 0..counts[1] {
                    chunks[grid_index(counts, i, j, 0)].support = Support::Fixed;
                }
            }
        }
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        s.bodies.push(body(&format!("wall{k}"), chunks, bonds));
    }
    s
}

/// A concrete cantilever plate under gravity: `len` chunks of 0.1 m along +x (the first
/// column fixed), 3 wide along y, one thick.
pub fn cantilever_plate(len: usize) -> Scene {
    use stress_ref::builders::*;
    use stress_ref::math::Vec3;
    use stress_ref::scene::Support;
    let mut s = new_scene(&format!("cantilever_plate{len}"), "concrete cantilever plate under gravity");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("projectile".into(), elastic_steel());
    let counts = [len, 3, 1];
    let mut chunks = grid(Vec3::new(0.0, -0.15, 1.0), counts, Vec3::splat(0.1), "concrete");
    for j in 0..3 {
        chunks[grid_index(counts, 0, j, 0)].support = Support::Fixed;
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("plate", chunks, bonds));
    s
}

/// A steel shot of `radius` and `mass` at `position`, moving at `velocity` (the scene's
/// "projectile" material).
pub fn shot_at(radius: f64, mass: f64, position: [f64; 3], velocity: [f64; 3]) -> stress_ref::scene::ImpactorDesc {
    stress_ref::scene::ImpactorDesc {
        name: "shot".into(),
        shape: stress_ref::scene::ImpactorShape::Sphere { radius },
        mass,
        position,
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity,
        angular_velocity: [0.0; 3],
        material: "projectile".into(),
        crush: None,
    }
}
