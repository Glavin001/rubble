//! Write Slang→WGSL sources next to this crate for the Chrome load harness, plus
//! the vehicle presets and packer golden data the browser demo and node tests use.
//!
//! ```bash
//! cargo run -p engine-webgpu --release --bin wgsl_export
//! cd engine/crates/engine-webgpu && python3 -m http.server 8765
//! # Chrome → http://127.0.0.1:8765/chrome_wgsl_load.html
//! # Chrome → http://127.0.0.1:8765/vehicle.html
//! ```

use engine_vehicle::{pack, presets, VehicleSlots};
use engine_webgpu::WebGpuWorld;
use std::path::PathBuf;

fn words<T: bytemuck::Pod>(v: &[T]) -> Vec<u32> {
    bytemuck::cast_slice(v).to_vec()
}

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("generated");
    let out = root.join("wgsl");
    std::fs::create_dir_all(&out)?;
    let report = WebGpuWorld::compile_shaders_report();
    let mut names = Vec::new();
    let mut bad = 0u32;
    for (name, r) in report {
        match r {
            Ok(wgsl) => {
                std::fs::write(out.join(format!("{name}.wgsl")), wgsl)?;
                names.push(name);
            }
            Err(e) => {
                eprintln!("FAIL {name}: {e}");
                bad += 1;
            }
        }
    }
    names.sort();
    let entries = names
        .iter()
        .map(|n| format!("    \"{n}\""))
        .collect::<Vec<_>>()
        .join(",\n");
    let manifest = format!(
        "{{\n  \"note\": \"Fair bar vs three-avbd: load in Chrome WebGPU\",\n  \"count\": {},\n  \"entries\": [\n{entries}\n  ]\n}}\n",
        names.len()
    );
    std::fs::write(out.join("manifest.json"), manifest)?;
    println!("Wrote {} WGSL modules to {}", names.len(), out.display());

    // Vehicle presets (fully expanded descriptors) for vehicle.html.
    let all = presets::all();
    std::fs::write(root.join("vehicle_presets.json"), serde_json::to_string_pretty(&all)?)?;
    // Golden packer output at odd slot bases: vehicle.mjs must match bit for bit.
    let slots = VehicleSlots {
        vehicle: 1,
        wheel_base: 3,
        curve_point_base: 5,
        diff_base: 2,
        anti_roll_base: 1,
        part_base: 4,
        body_base: 10,
        collider_base: 20,
        aabb_block: 1000,
    };
    let rot = glam::Quat::from_rotation_y(0.3);
    let mut golden = Vec::new();
    for d in &all {
        let p = pack::pack(d, glam::Vec3::new(1.0, 2.0, 3.0), rot, slots);
        let r = pack::requirements(d);
        let req = serde_json::json!({"wheelSlots": r.wheel_slots, "curvePoints": r.curve_points, "diffs": r.diffs,
            "antiRolls": r.anti_rolls, "parts": r.parts, "bodies": r.bodies, "colliders": r.colliders});
        golden.push(serde_json::json!({
            "name": d.name,
            "position": [1.0, 2.0, 3.0],
            "rotation": rot.to_array(),
            "slots": {
                "vehicle": slots.vehicle, "wheelBase": slots.wheel_base, "curvePointBase": slots.curve_point_base,
                "diffBase": slots.diff_base, "antiRollBase": slots.anti_roll_base, "partBase": slots.part_base,
                "bodyBase": slots.body_base, "colliderBase": slots.collider_base, "aabbBlock": slots.aabb_block,
            },
            "requirements": req,
            "bodies": words(&p.bodies),
            "colliders": words(&p.colliders),
            "vehicle": p.vehicle,
            "wheels": p.wheels,
            "curvePoints": p.curve_points.iter().flat_map(|c| [c[0].to_bits(), c[1].to_bits()]).collect::<Vec<_>>(),
            "diffs": p.diffs,
            "antiRolls": p.anti_rolls,
            "parts": p.parts,
            "centerOfMass": d.center_of_mass(),
        }));
    }
    std::fs::write(root.join("vehicle_golden.json"), serde_json::to_string(&golden)?)?;
    println!("Wrote {} vehicle presets and packer golden data", all.len());
    if bad > 0 {
        anyhow::bail!("{bad} translations failed");
    }
    Ok(())
}
