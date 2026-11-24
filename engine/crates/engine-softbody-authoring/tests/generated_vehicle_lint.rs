//! The vehicle lint (`authoring::lint`) on every generated design
//! (`engine_vehiclegen::DESIGNS`), rigged in memory as the site builds it: each hard rule
//! is a test of its own, so a design or rigger change that breaks one is caught at build
//! time, not found in a crash study. The lint's rules are proven on small synthetic cars
//! in `tests/host/vehicle_lint.rs` (a sound one passes, each broken variant fails its rule).
//! Failing rules are on `engine/tests-expected-failures.txt` with their measured values.
//!
//! Run: `cargo test --release -p engine-softbody-authoring --test generated_vehicle_lint -- --nocapture`

use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody_authoring::authoring::gen2::rig::WHEEL_KG;
use engine_softbody_authoring::authoring::gen2::vg_input::VgInput;
use engine_softbody_authoring::authoring::lint::{self, Expect, Finding, Level};
use engine_softbody_authoring::authoring::rig;
use engine_softbody_authoring::authoring::skin::ShowcaseVisual;
use engine_softbody_authoring::vehicle_compare::settings;
use glam::Vec3;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Each design's findings (rigged, loaded and linted once per test binary).
fn findings() -> &'static HashMap<&'static str, Vec<Finding>> {
    static F: OnceLock<HashMap<&'static str, Vec<Finding>>> = OnceLock::new();
    F.get_or_init(|| {
        let mut out = HashMap::new();
        for &design in engine_vehiclegen::DESIGNS {
            let built = rig::rig_vehicle(design, None, None).expect("rig");
            let files = rig::mod_files(&built);
            let refs: Vec<(String, String)> = files
                .iter()
                .filter(|(p, _)| p.ends_with(".jbeam"))
                .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
                .collect();
            let pc = files
                .iter()
                .find(|(p, _)| p.ends_with(".pc"))
                .map(|(_, b)| String::from_utf8_lossy(b).into_owned());
            let r: Vec<(&str, &str)> = refs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let desc: SoftVehicleDesc = engine_softbody::jbeam::load_jbeam(&r, pc.as_deref()).expect("load");
            let vis = visual(&built, &refs, pc.as_deref(), &desc);
            let mut ex = Expect::default();
            if let Ok(inp) = VgInput::find(design, None, None) {
                ex.mass = Some(inp.physics.kerb - 4.0 * WHEEL_KG);
                let af = inp.rig.dims.axle_f;
                ex.markers = inp
                    .rig
                    .markers
                    .iter()
                    .map(|(n, m)| (n.clone(), Vec3::new(m.pos[0], m.pos[1], m.pos[2] - af)))
                    .collect();
            }
            let f = lint::lint(&desc, vis.as_ref(), &ex);
            lint::print(design, &f);
            out.insert(design, f);
        }
        out
    })
}

/// The design's bound render skin (for the coverage rule), as the lab draws it.
fn visual(
    built: &rig::RiggedVehicle,
    refs: &[(String, String)],
    pc: Option<&str>,
    desc: &SoftVehicleDesc,
) -> Option<ShowcaseVisual> {
    let r: Vec<(&str, &str)> = refs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let flex = engine_softbody::jbeam::assembled_rows(&r, pc, "flexbodies").ok()?;
    let (sim, _) = ExplicitBeams::from_desc_jbeam_raycast(desc, Vec3::ZERO, Vec3::ZERO, settings()).ok()?;
    let g = &built.input.glb;
    let meshes: Vec<_> = g.meshes.iter().map(|m| engine_softbody_authoring::flexskin::glb_mesh(g, m)).collect();
    let paint = engine_softbody_authoring::glb::paint_of(g).unwrap_or([0.5, 0.05, 0.04]);
    engine_softbody_authoring::flexmesh::jbeam_visual_skinned(
        &flex,
        &[],
        desc,
        &sim,
        &meshes,
        &Default::default(),
        paint,
        Some(&built.skin),
    )
    .ok()
    .map(|v| v.0)
}

/// Every design passes `rule` (no Fail; a rule a design does not have is not checked).
fn passes(rule: &str) {
    let mut bad = Vec::new();
    for (design, f) in findings() {
        for x in f.iter().filter(|x| x.rule == rule && x.level == Level::Fail) {
            bad.push(format!("{design} {} {}: {} (band {}); {}", x.rule, x.part, x.value, x.band, x.offenders.join("; ")));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// Every hard rule the lint reports is one of the tests below (a new hard rule gets its
/// test, so it cannot be skipped silently).
#[test]
fn every_hard_rule_has_a_test() {
    const TESTED: &[&str] = &[
        "free_mechanism",
        "stability",
        "mass",
        "hardware",
        "skin_coverage",
        "rest_penetration",
        "skin_winding",
        "skin_facing",
        "skin_self_crossing",
        "skin_holes",
        "bound_wall",
        "tether",
    ];
    let mut missing: Vec<&str> = findings()
        .values()
        .flatten()
        .filter(|x| x.hard && !TESTED.contains(&x.rule))
        .map(|x| x.rule)
        .collect();
    missing.sort();
    missing.dedup();
    assert!(missing.is_empty(), "hard rules without a test here: {missing:?}");
}

#[test]
fn generated_designs_have_no_free_mechanism() {
    passes("free_mechanism");
}
#[test]
fn generated_designs_are_stable_at_the_step() {
    passes("stability");
}
#[test]
fn generated_designs_have_their_mass() {
    passes("mass");
}
#[test]
fn generated_designs_have_their_hardware() {
    passes("hardware");
}
#[test]
fn generated_designs_collision_skin_covers_the_render_skin() {
    passes("skin_coverage");
}
#[test]
fn generated_designs_start_without_penetration() {
    passes("rest_penetration");
}
#[test]
fn generated_designs_collision_skin_winding_is_consistent() {
    passes("skin_winding");
}
#[test]
fn generated_designs_collision_skin_faces_out() {
    passes("skin_facing");
}
#[test]
fn generated_designs_collision_skin_does_not_cross_itself() {
    passes("skin_self_crossing");
}
#[test]
fn generated_designs_collision_skin_has_no_holes() {
    passes("skin_holes");
}
#[test]
fn generated_designs_have_no_bound_walls() {
    passes("bound_wall");
}
#[test]
fn generated_designs_have_no_tethers() {
    passes("tether");
}
