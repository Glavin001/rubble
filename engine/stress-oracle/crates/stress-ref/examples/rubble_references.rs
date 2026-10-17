//! Reference runs that gate rubble's GPU stress solver (`engine/stress-oracle/reference/`).
//!
//! Each case writes `<out>/<methods>/<case>/scene.json` (with derived bond data, the exact
//! input) and `observation.json` (the oracle's run), plus one `provenance.json` per method
//! set. The method set is pinned per run: `established` is what stress-gpu implements,
//! `proven` is the oracle's default and the stricter truth.
//!
//! ```text
//! cargo run --release -p stress-ref --example rubble_references -- [--methods established|proven]
//!     [--out reference] [--only PREFIX] [--safety-divisor N] [--veneer]
//! ```
//! `--safety-divisor N` divides the stress and contact Courant safeties by N (the refined
//! runs of the timing and refinement gates); the case names gain `_refN`.

use std::path::{Path, PathBuf};

use stress_ref::builders::*;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::world::World;

/// A 200 kg ram at 40 m/s into a free 528 kg concrete wall (REALTIME_ROADMAP A10/A11:
/// the reference leaves the ram near 35-36 m/s; a rigid stop gives 11 m/s).
fn ram_into_free_wall() -> Scene {
    let mut s = new_scene("ram_into_free_wall", "200 kg steel ram at 40 m/s into a free concrete wall");
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
    s.sim.duration = 0.5;
    s
}

/// A 400 kg box rammed at 12 m/s into the middle of a pack's -x face (stress-gpu's
/// `scenes::rammed`, the A12 house regime).
fn rammed(mut scene: Scene, name: &str) -> Scene {
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for c in scene.bodies.iter().flat_map(|b| b.chunks.iter()) {
        let p = Vec3::from_array(c.center);
        lo = lo.component_min(p);
        hi = hi.component_max(p);
    }
    let mid = (lo + hi) * 0.5;
    let material = scene.bodies[0].chunks[0].material.clone();
    scene.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.15, 0.3, 0.3] },
        mass: 400.0,
        position: [lo.x - 0.35, mid.y, mid.z],
        orientation: [0.0, 0.0, 0.0, 1.0],
        velocity: [12.0, 0.0, 0.0],
        angular_velocity: [0.0; 3],
        material,
        crush: None,
    });
    scene.name = name.into();
    scene
}

/// The lab's wrecking ball (200 t, radius 1.6 m, 40 m/s) into the middle of a structure's
/// -x face, as `fracture/geometry.mjs` `buildGrid` shoots it.
fn wrecking_ball(mut scene: Scene, name: &str) -> Scene {
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for c in scene.bodies.iter().flat_map(|b| b.chunks.iter()) {
        let p = Vec3::from_array(c.center);
        lo = lo.component_min(p);
        hi = hi.component_max(p);
    }
    let mid = (lo + hi) * 0.5;
    scene.materials.entry("steel".into()).or_insert_with(elastic_steel);
    scene.impactors.push(ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 1.6 },
        mass: 200_000.0,
        position: [lo.x - 1.8, mid.y, mid.z],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [40.0, 0.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    scene.sim.duration = 0.3;
    scene.name = name.into();
    scene
}

fn cases() -> Vec<Scene> {
    let named = |n: &str| {
        builders_and_showcases().into_iter().find(|s| s.name == n).unwrap_or_else(|| panic!("no scene {n}"))
    };
    let packs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs");
    let house = stress_ref::scene_pack::import(&packs.join("house-1story.json"), "house-1story", "structure").expect("house-1story pack");
    let mut out = vec![
        named("b2_cantilever"),
        named("b4_support_loss_sudden"),
        named("b4_support_loss_gradual"),
        named("b8_frame_sudden"),
        named("b8_frame_gradual"),
    ];
    for v in [10.0, 15.0, 20.0, 30.0, 40.0] {
        out.push(wall_impact(v, &format!("b5_wall_impact_v{v:02}")));
    }
    out.push(ram_into_free_wall());
    out.push(named("s_arch_keystone_removed"));
    out.push(named("s_floor_drop"));
    out.push(named("s_house_car"));
    out.push(rammed(house, "house_1story_ram"));
    // The lab's own asset (`fracture/assets/veneer-house.glb`, exported by
    // `engine-fracture`'s `fracture_stress_export` at level 0): explicit at the true step
    // these take hours of CPU, so only with `--veneer`.
    if !std::env::args().any(|a| a == "--veneer") {
        return out;
    }
    let veneer = stress_ref::scene_pack::import(&packs.join("veneer-house-l0.json"), "veneer_house_l0", "structure").expect("veneer-house-l0 pack");
    let mut rest = veneer.clone();
    rest.name = "veneer_house_l0_rest".into();
    rest.sim.duration = 1.0;
    out.push(rest);
    out.push(wrecking_ball(veneer, "veneer_house_l0_ball"));
    out
}

fn builders_and_showcases() -> Vec<Scene> {
    stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()).collect()
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let methods_name = flag(&args, "--methods").unwrap_or_else(|| "established".into());
    let methods = match methods_name.as_str() {
        "established" => Methods::established(),
        "proven" => Methods::proven(),
        other => panic!("--methods established|proven, not {other}"),
    };
    let divisor: f64 = flag(&args, "--safety-divisor").map_or(1.0, |v| v.parse().expect("--safety-divisor N"));
    let only = flag(&args, "--only");
    let root = PathBuf::from(flag(&args, "--out").unwrap_or_else(|| "reference".into())).join(&methods_name);
    std::fs::create_dir_all(&root).expect("create output directory");
    let mut summary = Vec::new();
    for mut scene in cases() {
        if divisor != 1.0 {
            scene.sim.scale_safety(1.0 / divisor);
            scene.name = format!("{}_ref{divisor}", scene.name);
        }
        if only.as_ref().is_some_and(|o| !scene.name.starts_with(o.as_str())) {
            continue;
        }
        scene.sim.methods = methods;
        scene.validate().unwrap_or_else(|e| panic!("{}: {e}", scene.name));
        let dir = root.join(&scene.name);
        std::fs::create_dir_all(&dir).expect("case directory");
        let derived = scene.with_derived();
        std::fs::write(dir.join("scene.json"), serde_json::to_string(&derived).unwrap() + "\n").expect("write scene");
        let start = std::time::Instant::now();
        let mut obs = World::new(&derived).run();
        let seconds = start.elapsed().as_secs_f64();
        obs.values.insert("wall_seconds".into(), seconds);
        std::fs::write(dir.join("observation.json"), obs.to_json_pretty() + "\n").expect("write observation");
        println!("{}: {seconds:.1} s", scene.name);
        summary.push(serde_json::json!({"case": scene.name, "wall_seconds": seconds}));
    }
    let provenance = serde_json::json!({
        "generator": "engine/stress-oracle/crates/stress-ref/examples/rubble_references.rs",
        "oracle_source": "PhysX integration/oracle-stress-gpu ec8118341, copied into rubble 2026-10-16",
        "methods": methods_name,
        "safety_divisor": divisor,
        "cases": summary,
    });
    let name = if divisor == 1.0 { "provenance.json".to_string() } else { format!("provenance_ref{divisor}.json") };
    std::fs::write(root.join(name), serde_json::to_string_pretty(&provenance).unwrap() + "\n").expect("write provenance");
}
