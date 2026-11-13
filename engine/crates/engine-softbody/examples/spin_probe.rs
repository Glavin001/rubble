//! A small free part spinning in the air on the CPU reference, with the
//! energy ledger: three nodes in a triangle (a hub cap that has left its
//! wheel), no contact. A free body's kinetic + gravitational + elastic energy
//! can only fall (its beams' damping); a rise is energy the step created.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example spin_probe -- \
//!     [--kg 0.1] [--side 0.095] [--k 68000] [--c 20] [--rim 20] [--centred 1] [--objective 1] [--secs 2]
//! ```
//! `--rim`: the nodes' speed about the centre (m/s). Prints the rotation per
//! step, the beams' ω·h and damping share, and the energy every 0.25 s.
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, STEP};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::Vec3;

fn num(k: &str, d: f32) -> f32 {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() -> Result<(), String> {
    let (kg, side, k, c) = (
        num("--kg", 0.1),
        num("--side", 0.095),
        num("--k", 68000.0),
        num("--c", 20.0),
    );
    let (rim, secs) = (num("--rim", 20.0), num("--secs", 2.0));
    let centred = num("--centred", 1.0) != 0.0;
    let r = side / 3f32.sqrt();
    let p: Vec<[f32; 3]> = (0..3)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 3.0;
            [r * a.cos(), r * a.sin(), 0.0]
        })
        .collect();
    let text = format!(
        r#"{{"probe":{{"information":{{"name":"probe"}},"slotType":"main",
        "nodes":[["id","posX","posY","posZ"],{{"nodeWeight":{kg},"collision":false,"selfCollision":false}},
          ["a",{},{},{}],["b",{},{},{}],["c",{},{},{}]],
        "beams":[["id1:","id2:"],{{"beamSpring":{k},"beamDamp":{c},"beamDeform":"FLT_MAX","beamStrength":"FLT_MAX"}},
          ["a","b"],["b","c"],["c","a"]]}}}}"#,
        p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], p[2][0], p[2][1], p[2][2]
    );
    let desc = engine_softbody::jbeam::load_jbeam(&[("probe.jbeam", &text)], None)?;
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
        &desc,
        JbeamWheels::Pressure,
        Vec3::Y * 200.0,
        Vec3::ZERO,
        settings,
    )?;
    // The JBeam (BeamNG) profile turns centred damping on; `--centred 0` turns it off.
    s.settings.centred_damping = centred;
    // `--objective 0`: the end-direction form (before spec §39.6), which gains energy.
    s.settings.centred_objective = num("--objective", 1.0) != 0.0;
    let centre = s.nodes.iter().map(|n| n.pos).sum::<Vec3>() / 3.0;
    // The triangle's normal, and a spin about it.
    let nrm = (s.nodes[1].pos - s.nodes[0].pos)
        .cross(s.nodes[2].pos - s.nodes[0].pos)
        .normalize();
    let radius = (s.nodes[0].pos - centre).length();
    for n in &mut s.nodes {
        n.vel = nrm.cross(n.pos - centre) * (rim / radius);
    }
    let m = s.nodes[0].mass;
    println!(
        "{m:.3} kg nodes, side {side} m, k {k}, c {c}, rim {rim} m/s, centred damping {}: rotation {:.3} rad a step; beams' omega h {:.2}; damping h c (2/m) {:.2}",
        centred as u8,
        rim / radius * STEP,
        STEP * (3.0 * k / m).sqrt(),
        STEP * c * 2.0 / m,
    );
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.enable_ledger();
    let e0 = w.ledger.as_ref().unwrap().mechanical();
    let pe0 = w.ledger.as_ref().unwrap().gravity_pe;
    let steps = (secs / STEP) as usize;
    for kstep in 0..steps {
        w.step();
        if kstep % 500 == 499 || kstep + 1 == steps {
            let l = w.ledger.as_ref().unwrap();
            let v = w.vehicles[0]
                .nodes
                .iter()
                .map(|n| n.vel.length())
                .fold(0.0f32, f32::max);
            println!(
                "  t {:.2} s: energy less the fall's {:+.3} J (start {:.3}); elastic {:.3} J; fastest node {:.1} m/s",
                (kstep + 1) as f32 * STEP,
                l.mechanical() - e0,
                e0 - pe0,
                l.elastic,
                v
            );
            if !l.mechanical().is_finite() {
                break;
            }
        }
    }
    Ok(())
}
