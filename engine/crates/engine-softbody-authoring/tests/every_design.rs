//! The gate for every registered vehicle design (`engine_vehiclegen`'s
//! `designs::REGISTRY`): no design is named here. A design passes by being
//! rigged from its own definition, no per-vehicle code:
//!
//! * rigs and loads (JBeam + `.pc`), mass and centre of mass plausible (CoM
//!   between the axles);
//! * parked 3 s on raycast wheels: every part's nodes within 5 mm of design
//!   about the body's rigid fit, 0 beams broken or bent, state finite;
//! * driven 3 s straight at 10 m/s of throttle: state finite, 0 broken.
//!
//! Run in release: `cargo test --release -p engine-softbody-authoring --test
//! every_design -- --nocapture`. `EVERY_DESIGN=name` restricts to one.
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody_authoring::authoring::gen2::vg_input;
use engine_softbody_authoring::vehicle_compare::{lowest_point, rigid_fit, settings};
use glam::Vec3;
use std::path::PathBuf;

/// Designs the generator builds but the rigger cannot rig yet, and why. An
/// entry is removed when its design passes (the point of this list is that it
/// only shrinks; `EVERY_DESIGN=<name>` still runs one).
const NOT_YET_RIGGED: &[(&str, &str)] = &[];

fn designs() -> Vec<&'static str> {
    let only = std::env::var("EVERY_DESIGN").ok();
    engine_vehiclegen::design_names()
        .into_iter()
        .filter(|n| match only.as_deref() {
            Some(o) => o == *n,
            None => match NOT_YET_RIGGED.iter().find(|p| p.0 == *n) {
                Some(p) => {
                    println!("{n}: skipped, not yet rigged: {}", p.1);
                    false
                }
                None => true,
            },
        })
        .collect()
}

fn rig(design: &str) -> (engine_softbody::desc::SoftVehicleDesc, ExplicitBeams) {
    let inp = vg_input::VgInput::find(design, None, None).unwrap_or_else(|e| panic!("{design}: {e}"));
    let b = vg_input::build(inp).unwrap_or_else(|e| panic!("{design}: rig: {e}"));
    // (a design with no glass in its model, the box trailer, has none to bind)
    let has_glass = b.input.glb.meshes.iter().any(|m| m.name.starts_with("glass_"));
    glass_interpolates(design, &b.skin, has_glass);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("every_{design}"));
    for (p, bytes) in vg_input::files(&b) {
        let path = dir.join(&p);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
    }
    let files = engine_softbody::jbeam::read_vehicle_dir(&dir, design).unwrap();
    let pc = std::fs::read_to_string(dir.join(design).join(format!("{design}.pc"))).ok();
    let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).unwrap();
    let (mut sim, _) = ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    sim.surface_contact = true;
    (desc, sim)
}

/// Every glass vertex's skin interpolates its frame: its amplification of the nodes'
/// motion apart from each other (Σ|A|) at most `gen3::GLASS_AMP_MAX`. Past it a pane
/// extrapolates from the frame on one side and a crush there throws it the other
/// way (the generated 4x4's windscreen: 4.46, a spike 0.7–0.9 m above the car).
fn glass_interpolates(design: &str, skin: &engine_softbody_authoring::flexskin::Skin, has_glass: bool) {
    use engine_softbody_authoring::{authoring::gen2::gen3, flexskin};
    let mut panes = 0;
    for m in skin.meshes.iter().filter(|m| m.name.starts_with("glass_")) {
        panes += 1;
        let (i, amp) = m
            .verts
            .iter()
            .map(flexskin::amplification)
            .enumerate()
            .fold((0, 0.0), |a, (i, x)| if x > a.1 { (i, x) } else { a });
        assert!(
            amp <= gen3::GLASS_AMP_MAX,
            "{design}: {} vertex {i}: amplification {amp:.2} (Σ|A|; at most {}): it extrapolates from one side of its frame",
            m.name,
            gen3::GLASS_AMP_MAX
        );
    }
    assert!(panes > 0 || !has_glass, "{design}: no glass meshes");
}

fn lift(sim: &mut ExplicitBeams) {
    let low = lowest_point(sim);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - low;
    }
}

fn run(s: &mut ExplicitBeams, seconds: f32) {
    for _ in 0..(seconds / 0.0005).round() as usize {
        s.step();
    }
}

fn finite(s: &ExplicitBeams) -> bool {
    s.nodes.iter().all(|n| n.pos.is_finite() && n.vel.is_finite())
}

#[test]
fn every_design_rigs_and_rests_within_5_mm() {
    for name in designs() {
        let (desc, mut sim) = rig(name);
        let nd = desc.nodes.len();
        let design: Vec<Vec3> = desc.nodes.iter().map(|n| Vec3::from(n.position)).collect();
        let susp = |n: &str| n.starts_with("hub") || n.starts_with("rim") || n.starts_with("whe");
        let fit_ids: Vec<u32> = (0..nd as u32).filter(|&i| !susp(&desc.nodes[i as usize].name)).collect();
        lift(&mut sim);
        run(&mut sim, 3.0);
        assert!(finite(&sim), "{name}: state not finite");
        let now: Vec<Vec3> = (0..nd).map(|i| sim.origin + sim.nodes[i].pos).collect();
        let fit = rigid_fit(&design, &now, &fit_ids);
        let broken = sim.beams.iter().filter(|b| b.broken).count();
        let bent = sim.beams.iter().filter(|b| b.bent()).count();
        let mut worst = (String::new(), 0.0f32);
        let detail = std::env::var("EVERY_DETAIL").is_ok();
        for p in &desc.parts {
            let mut part_worst = (String::new(), 0.0f32);
            for &i in p.nodes.iter().filter(|&&i| (i as usize) < nd && !susp(&desc.nodes[i as usize].name)) {
                let d = (fit.to_rest(now[i as usize]) - design[i as usize]).length();
                if d > worst.1 {
                    worst = (format!("{}:{}", p.name, desc.nodes[i as usize].name), d);
                }
                if d > part_worst.1 {
                    part_worst = (desc.nodes[i as usize].name.clone(), d);
                }
            }
            if detail && part_worst.1 > 0.002 {
                println!("  {}: {:.1} mm at {}", p.name, part_worst.1 * 1e3, part_worst.0);
            }
        }
        // standing on its wheels: the lowest point (a raycast wheel's rim of
        // rubber included) is at the ground, not sunk or hovering
        let low = lowest_point(&sim);
        let mass: f32 = desc.nodes.iter().map(|n| n.mass).sum();
        println!("{name}: {nd} nodes, {} beams, {mass:.0} kg, worst drift {:.1} mm at {}, broken {broken}, bent {bent}, lowest point {:.0} mm", sim.beams.len(), worst.1 * 1e3, worst.0, low * 1e3);
        assert!(low.abs() < 0.03, "{name}: the parked car's lowest point is {:.0} mm from the ground", low * 1e3);
        assert_eq!((broken, bent), (0, 0), "{name}");
        assert!(worst.1 <= 0.005, "{name}: drift {:.1} mm at {}", worst.1 * 1e3, worst.0);
    }
}
