//! Scratch probe (not committed): the door's nodes.
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use std::collections::HashSet;

#[test]
#[ignore]
fn door_nodes() {
    let part_name = std::env::var("PART").unwrap_or("door_f_l".into());
    let b = build(&style("sedan1").unwrap(), &BuildParams { resolution: Resolution::Medium, ..Default::default() }).unwrap();
    let sk = &b.skeleton;
    let q = sk.parts.iter().position(|p| *p == part_name).unwrap() as u16;
    let shell: HashSet<u32> = b.shell.nodes.iter().copied().collect();
    let mut ends = vec![0usize; sk.nodes.len()];
    for bm in &sk.beams { if !bm.damper { ends[bm.a as usize]+=1; ends[bm.b as usize]+=1; } }
    let mut tot = 0.0;
    for (i, n) in sk.nodes.iter().enumerate() {
        if n.part != q { continue; }
        tot += n.mass;
        let groups: Vec<&String> = b.groups.iter().filter(|(_, g)| g.contains(&(i as u32))).map(|(k, _)| k).collect();
        eprintln!("{i:5} ({:6.3},{:6.3},{:6.3}) shell={} m={:.3} ends={} seg={:?} {:?}", n.p.x, n.p.y, n.p.z, shell.contains(&(i as u32)), n.mass, ends[i], b.shell.seg.get(&(i as u32)), groups);
    }
    eprintln!("total {tot:.2} kg");
    for r in &b.shell.rings { if r.z > 1.3 && r.z < 3.4 { eprintln!("ring z {:.3} n {} minor {}", r.z, r.half.len(), r.minor); } }
}
