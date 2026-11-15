//! The coupling's arithmetic has one code shape (`explicit_beams_rigid.slang`
//! `cplWindowAdd`, `cplTorqueTerm`, `cplIntegrateStep`). The unfolded step, the folded
//! step (`cplStepFolded`) and a contact group's members compute the same window sums,
//! torque terms and body integration, and the tests compare those paths bit for bit
//! (engine-world `explicit_vehicles`: the candidate list on and off, a vehicle's contact
//! margin). Written out once per path, a device's shader compiler may contract each copy
//! differently (a fused multiply-add in one, a rounded product and a sum in another):
//! on an NVIDIA T4 through Vulkan three of those comparisons differed in the coupling's
//! records (window sums) while passing on Metal. This test reads the shipped kernel
//! sources (not the frozen `_ref` / `_baseline` kernels) and requires each of those
//! expressions to appear exactly once.

use std::path::Path;

fn shipped_sources() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../slang/kernels");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).expect("kernels") {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".slang") || name.contains("_ref") || name.contains("baseline") {
            continue;
        }
        out.push((name, std::fs::read_to_string(&p).unwrap()));
    }
    out
}

#[test]
fn the_coupling_arithmetic_has_one_code_shape() {
    let sources = shipped_sources();
    let shapes = [
        ("the window impulse", "imp.xyz + F * STEP"),
        ("the window moment", "mom.xyz + M * STEP"),
        ("the torque term", "cross(ref - c"),
        ("the body's velocity", "force * (c0.w * STEP)"),
        ("the body's angular velocity", "cplRot(q, c4.xyz * lt) * STEP"),
        ("the body's position", "c0.xyz + vel * STEP"),
        ("the body's rotation", "q + 0.5 * STEP * dq"),
    ];
    let mut bad = Vec::new();
    for (what, text) in shapes {
        let at: Vec<String> = sources
            .iter()
            .flat_map(|(name, src)| {
                src.lines()
                    .enumerate()
                    .filter(|(_, l)| l.contains(text) && !l.trim_start().starts_with("//"))
                    .map(move |(i, _)| format!("{name}:{}", i + 1))
            })
            .collect();
        if at.len() != 1 {
            bad.push(format!("{what} (`{text}`) written {} times: {at:?}", at.len()));
        }
    }
    assert!(bad.is_empty(), "the coupling's arithmetic in more than one code shape:\n{}", bad.join("\n"));
}
