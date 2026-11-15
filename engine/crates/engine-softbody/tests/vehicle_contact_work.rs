//! Vehicle contact never does positive work over a step, measured with the
//! step's **final** velocities (jbeam-semantics-spec.md §37.1's work guard).
//!
//! **The bug (2026-11-15).** The guard sized each staged two-body hit so its
//! work at the pair's mean velocity over the next step is at most zero, with
//! the end velocity predicted as v + h·F/m. The next step's centred damping
//! correction (`ExplicitBeamsSettings::centred_damping`, spec §39) then changes
//! the end velocity by h·c/m, which the guard never saw. A stuck contact under
//! a tangential load did positive work: the showcase T-bone at 80 km/h, striker
//! node 947 on the struck door's triangle (151, 171, 173) at 2.5 kN, +0.5–1 mJ
//! a step for 0.1 s (`showcase_car::contacts_do_no_positive_work`). With the
//! centred damping off, the end velocities are exactly v + h·F/m.
//!
//! **The scene.** A damped tetrahedron (four 20 kg nodes, the BeamNG profile:
//! two-body dissipative contact, centred damping) set point-down, vibrating,
//! on a fixed plate inclined 19°: friction holds it against the slope while
//! its beams ring.
//!
//! **The fix** (`ExplicitBeamsWorld::guard_final`, `final_guard`): after every
//! vehicle's node update and correction, each staged hit is checked on the
//! final velocities and scaled to zero work. Off by default until the kernels
//! carry it; this test turns it on (`FINAL_GUARD=0` runs it off: it fails,
//! +1.0e-5 J past the bound in a step).
//!
//! **The bound.** The contact forces reach the nodes as fixed-point sums
//! (`CONTACT_FIXED` steps a newton per component, each hit's share rounded on
//! its own), while the ledger reads them unrounded: per node, per hit, at most
//! √3/(2·CONTACT_FIXED) N of force the ledger misattributes, times the node's
//! mean speed over the step and h; and the velocities' single precision (a
//! few ε of |F|·(|v₀| + |v′| + |v′ − v₀|)·h: the velocities and the step's
//! change of them each rounded once more). Nothing else is allowed.
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, CONTACT_FIXED, STEP,
};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::Vec3;

const SLOPE: f32 = 0.35;

/// A body of `points` (JBeam frame: z up) with every pair of nodes joined,
/// `tris` its collision triangles.
fn body(
    name: &str,
    points: &[[f32; 3]],
    tris: &[[usize; 3]],
    kg: f32,
    fixed: bool,
) -> ExplicitBeams {
    let nodes: Vec<String> = points
        .iter()
        .enumerate()
        .map(|(i, p)| format!(r#"["n{i}",{},{},{}]"#, p[0], p[1], p[2]))
        .collect();
    let mut rows = Vec::new();
    for a in 0..points.len() {
        for b in a + 1..points.len() {
            rows.push(format!(r#"["n{a}","n{b}"]"#));
        }
    }
    let t: Vec<String> = tris
        .iter()
        .map(|t| format!(r#"["n{}","n{}","n{}"]"#, t[0], t[1], t[2]))
        .collect();
    let text = format!(
        r#"{{"{name}":{{"information":{{"name":"{name}"}},"slotType":"main",
        "nodes":[["id","posX","posY","posZ"],{{"nodeWeight":{kg},"fixed":{fixed},"collision":true,"selfCollision":false,"frictionCoef":1.0}},{}],
        "beams":[["id1:","id2:"],{{"beamSpring":200000,"beamDamp":200,"beamDeform":"FLT_MAX","beamStrength":"FLT_MAX"}},{}],
        "triangles":[["id1:","id2:","id3:"],{}]}}}}"#,
        nodes.join(","),
        rows.join(","),
        t.join(",")
    );
    let desc = engine_softbody::jbeam::load_jbeam(&[(&format!("{name}.jbeam"), &text)], None)
        .expect("the probe's jbeam");
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let (s, _) = ExplicitBeams::from_desc_jbeam_with(
        &desc,
        JbeamWheels::Pressure,
        Vec3::Y * 30.0,
        Vec3::ZERO,
        settings,
    )
    .expect("spawn");
    s
}

#[test]
fn a_stuck_vibrating_block_s_contact_never_does_positive_work_in_a_step() {
    let plate = body(
        "plate",
        &[
            [-0.6, -0.5, 0.02],
            [0.6, -0.5, 0.02],
            [0.0, 0.7, 0.02],
            [-0.6, -0.5, 0.3 - 0.5 * SLOPE],
            [0.6, -0.5, 0.3 - 0.5 * SLOPE],
            [0.0, 0.7, 0.3 + 0.7 * SLOPE],
        ],
        &[[3, 5, 4]],
        10.0,
        true,
    );
    let mut block = body(
        "block",
        &[
            [0.0, 0.0, 0.33],
            [-0.2, -0.15, 0.6],
            [0.2, -0.15, 0.6],
            [0.0, 0.2, 0.6],
        ],
        &[],
        20.0,
        false,
    );
    assert!(block.settings.centred_damping && block.two_body_contact && block.dissipative_contact);
    // Its point 3 cm over the plate's face (the face is at y 30.3 under it), ringing.
    let lift = 30.33 - block.nodes[0].pos.y;
    for (n, v) in block.nodes.iter_mut().zip([
        Vec3::ZERO,
        Vec3::new(0.4, 0.1, 0.0),
        Vec3::new(-0.4, 0.0, 0.2),
        Vec3::new(0.0, -0.3, -0.3),
    ]) {
        n.pos.y += lift;
        n.vel = v;
    }
    let mut w = ExplicitBeamsWorld::new(vec![plate, block]);
    // The final guard (off by default until the GPU kernels carry it).
    w.final_guard = std::env::var("FINAL_GUARD").map_or(true, |v| v != "0");
    w.enable_ledger();
    let rounding = 3f64.sqrt() / (2.0 * CONTACT_FIXED as f64);
    let (mut touching, mut worst) = (0, (f64::MIN, 0usize, 0.0f64));
    let ledger = |w: &ExplicitBeamsWorld| {
        let l = w.ledger.as_ref().unwrap();
        l.w_contact_normal + l.w_contact_friction
    };
    for k in 1..=(2.0 / STEP).round() as usize {
        // The forces this step applies are the ones the last contact pass staged.
        let at: Vec<Vec<(Vec3, Vec3)>> = {
            let l = w.ledger.as_ref().unwrap();
            w.vehicles
                .iter()
                .enumerate()
                .map(|(vi, v)| {
                    (0..v.nodes.len())
                        .map(|i| (v.nodes[i].vel, l.vehicle_contact(vi, i)))
                        .collect()
                })
                .collect()
        };
        let w0 = ledger(&w);
        w.step();
        // The ledger's vehicle-contact work over the step (at the step's mean
        // velocities, the forces as applied).
        let work = ledger(&w) - w0;
        let mut bound = 0.0f64;
        for (vi, v) in w.vehicles.iter().enumerate() {
            for (i, n) in v.nodes.iter().enumerate() {
                let (v0, f) = at[vi][i];
                if f != Vec3::ZERO {
                    let mean = ((v0 + n.vel).as_dvec3() * 0.5).length();
                    bound += rounding * mean * STEP as f64
                        + 4.0
                            * f64::from(f32::EPSILON)
                            * f64::from(f.length())
                            * f64::from(v0.length() + n.vel.length() + (n.vel - v0).length())
                            * STEP as f64;
                }
            }
        }
        if bound > 0.0 {
            touching += 1;
        }
        if work - bound > worst.0 {
            worst = (work - bound, k, work);
        }
    }
    eprintln!(
        "{touching} steps in contact; the most work beyond the rounding: {:+.3e} J (step {}, work {:+.3e} J)",
        worst.0, worst.1, worst.2
    );
    assert!(
        touching > 1000,
        "the block rests on the plate ({touching} steps in contact)"
    );
    assert!(
        worst.0 <= 0.0,
        "the contact did {:+.3e} J in step {} ({:+.3e} J past the rounding)",
        worst.2,
        worst.1,
        worst.0
    );
}
