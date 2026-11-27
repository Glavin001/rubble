//! The generated sedan's front structure against its own stated sections
//! (`engine/docs/sedan-front-audit.md`). Diagnostics, `#[ignore]`d (each rigs the sedan on the
//! CPU; the push runs 2 kHz steps); run with `--release -- --ignored --nocapture`.

use engine_softbody::crash_reference::Template;
use engine_softbody::explicit_beams::ExplicitBeams;
use glam::Vec3;
use std::fmt::Write as _;

fn sedan() -> (Template, ExplicitBeams, Vec<String>) {
    let t = engine_softbody_authoring::crash_templates::template("sedan").expect("sedan");
    let (s, _) = ExplicitBeams::from_desc_jbeam_raycast(&t.desc, Vec3::ZERO, Vec3::ZERO, t.settings).expect("spawn");
    let names = t.desc.nodes.iter().map(|n| n.name.clone()).collect();
    (t, s, names)
}

fn out_dir() -> std::path::PathBuf {
    let d = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/sections");
    std::fs::create_dir_all(&d).expect("out dir");
    d
}

#[test]
#[ignore = "diagnostic: dumps the rigged sedan's front nodes and beams"]
fn sedan_front_dump() {
    let (t, s, names) = sedan();
    let mut o = String::new();
    let zmax = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    writeln!(o, "nodes {} beams {} front z {zmax}", s.nodes.len(), s.beams.len()).ok();
    for (i, n) in s.nodes.iter().enumerate() {
        if n.pos.z > zmax - 1.6 {
            writeln!(o, "N {i} {} {:.3} {:.3} {:.3} m {:.2}", names.get(i).map_or("?", |s| s.as_str()), n.pos.x, n.pos.y, n.pos.z, n.mass).ok();
        }
    }
    for (j, b) in s.beams.iter().enumerate() {
        let (pa, pb) = (s.nodes[b.a as usize].pos, s.nodes[b.b as usize].pos);
        if pa.z.max(pb.z) > zmax - 1.6 {
            let nm = |i: u32| names.get(i as usize).map_or("?", |s| s.as_str());
            writeln!(o, "B {j} {} {} {:?} k {:.3e} tc {:.3e} tt {:.3e} S {:.3e} rest {:.3} short {:.3} long {:.3} lim {:.3e} grp {}", nm(b.a), nm(b.b), b.kind, b.k, b.tc, b.tt, b.strength, b.rest, b.short_bound, b.long_bound, b.limit_k, b.group).ok();
        }
    }
    writeln!(o, "zones {:?}", t.zones.parts).ok();
    std::fs::write(out_dir().join("sedan_front_dump.txt"), o).expect("write");
}
