//! The rig data of a Rust-generated design in memory, and the renames of the gen2 export (split from `vg_input.rs`).

use super::*;

/// gen2's export names (vehicle `gen2`, parts `gen2_*`) as the design's.
pub(super) fn rename(s: &str, name: &str) -> String {
    s.replace("gen2_", &format!("{name}_"))
        .replace("\"gen2\"", &format!("\"{name}\""))
        .replace("Rubble gen2 sedan", &format!("Rubble {name} (vehicle-gen)"))
}

/// `gen2/gen2_body.jbeam` as `<name>/<name>_body.jbeam`.
pub(super) fn rename_path(p: &str, name: &str) -> String {
    p.replace("gen2", name)
}

/// The rig data of a Rust-generated design (what `rigexport.py` writes for a
/// Python one), in memory.
pub(super) fn rig_json_of(
    name: &str,
    d: &dyn engine_vehiclegen::Design,
    cb: &engine_vehiclegen::carbody::CarBody,
    markers: &[engine_vehiclegen::export::Marker],
    spec: &engine_vehiclegen::RigSpec,
) -> RigJson {
    const KEYLINES: [&str; 10] = [
        "floor",
        "sill",
        "door_bottom",
        "widest",
        "waist",
        "shoulder",
        "belt",
        "roof_rail",
        "glass_edge",
        "centre",
    ];
    let (dz, dv, dxy) = (0.02f64, 1.0 / 16.0, 0.02f64);
    let tube = &cb.l.tube;
    let z0 = spec.z_rear - 0.06;
    let nz = (((spec.z_front + 0.06) - z0) / dz + 0.5).floor() as usize + 1;
    let zs: Vec<f64> = (0..nz).map(|i| z0 + dz * i as f64).collect();
    let f = |x: f64| x as f32;
    let mut lines = BTreeMap::new();
    for (k, n) in KEYLINES.iter().enumerate().take(tube.nk) {
        lines.insert(
            n.to_string(),
            zs.iter()
                .map(|&z| {
                    let p = tube.key(k, z);
                    [f(p.x), f(p.y)]
                })
                .collect(),
        );
    }
    let nv = ((tube.nk - 1) as f64 / dv).round() as usize + 1;
    let xy: Vec<Vec<[f32; 2]>> = zs
        .iter()
        .map(|&z| {
            (0..nv)
                .map(|j| {
                    let p = tube.eval(z, dv * j as f64);
                    [f(p.x), f(p.y)]
                })
                .collect()
        })
        .collect();
    let (nx, ny) = (61usize, 111usize);
    let tab = |g: &dyn Fn(f64, f64) -> f64| -> Vec<Vec<f32>> {
        (0..nx)
            .map(|i| {
                (0..ny)
                    .map(|j| f(g(dxy * i as f64, dxy * j as f64)))
                    .collect()
            })
            .collect()
    };
    let skins = cb
        .skins
        .iter()
        .map(|(n, m)| {
            let loops = engine_vehiclegen::panels::boundary_loops(m)
                .into_iter()
                .filter(|l| l.len() >= 3)
                .map(|l| {
                    l.iter()
                        .map(|&i| {
                            let p = m.v[i as usize];
                            // 0.1 mm, as rigexport.py rounds (shut lines'
                            // points coincide across parts).
                            let r = |x: f64| ((x * 1e4).round() * 1e-4) as f32;
                            [r(p.x), r(p.y), r(p.z)]
                        })
                        .collect()
                })
                .collect();
            (n.clone(), loops)
        })
        .collect();
    let mk = markers
        .iter()
        .map(|m| {
            (
                m.name.clone(),
                MarkerJson {
                    pos: [f(m.pos.x), f(m.pos.y), f(m.pos.z)],
                    axis: [f(m.xaxis.x), f(m.xaxis.y), f(m.xaxis.z)],
                },
            )
        })
        .collect();
    RigJson {
        format: 1,
        design: name.to_string(),
        base: None,
        title: d.name(),
        dims: RigDims {
            wheelbase: f(spec.wheelbase),
            track: f(spec.track),
            axle_f: f(spec.axle_f),
            axle_r: f(spec.axle_r),
            tyre_d: f(spec.tyre_d),
            tyre_w: f(spec.tyre_w),
            rim_d: f(spec.rim_d),
            wheel_y: f(spec.wheel_y),
            z_front: f(spec.z_front),
            z_rear: f(spec.z_rear),
            doors: spec.doors.clone(),
            rear_closure: spec.rear_closure.clone(),
            rear_wheel: spec.rear_wheel.map(|r| RearWheel {
                wheel_y: f(r.wheel_y),
                tyre_d: f(r.tyre_d),
                tyre_w: f(r.tyre_w),
                rim_d: f(r.rim_d),
            }),
            pose: spec.pose,
        },
        arches: spec
            .arches
            .iter()
            .map(|a| Arch {
                center: [f(a.0), f(a.1)],
                r: f(a.2),
            })
            .collect(),
        keys: Keys {
            z0: f(z0),
            dz: f(dz),
            n: nz,
            lines,
        },
        zlines: d
            .zlines()
            .iter()
            .map(|(n, z)| (n.to_string(), f(*z)))
            .collect(),
        vlines: d.vlines().iter().map(|&v| f(v)).collect(),
        sections: Sections {
            z0: f(z0),
            dz: f(dz),
            nz,
            dv: f(dv),
            nv,
            xy,
        },
        ends: Ends {
            x0: 0.0,
            y0: 0.0,
            d: f(dxy),
            nx,
            ny,
            front_z: tab(&|x, y| (spec.front_z)(x, y)),
            rear_z: tab(&|x, y| (spec.rear_z)(x, y)),
        },
        skins,
        markers: mk,
        physics: serde_json::to_value(&spec.physics).unwrap(),
    }
}
