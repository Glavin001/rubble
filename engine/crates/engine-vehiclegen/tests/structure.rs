//! The body-in-white (`structure`): every member a closed solid inside the
//! skin, the shell's 131 nodes on their members with the study's masses
//! (`engine/docs/beamng-structure-study.md` §10.2, §10.3), the triangle
//! budget, and parametric builds.

use engine_vehiclegen::designs::sedan::{Sedan, SedanParams};
use engine_vehiclegen::export::split_triangles;
use engine_vehiclegen::mesh::Mesh;
use engine_vehiclegen::solid::{check, pair_crossings};
use engine_vehiclegen::structure::{Skin, Structure, GROUPS};
use engine_vehiclegen::{build, design_from_json, BuildOpts};
use glam::DVec3;

fn sedan(p: &str) -> Sedan {
    let p: SedanParams = serde_json::from_str(if p.is_empty() { "{}" } else { p }).unwrap();
    Sedan::new(p)
}

fn tri_dist(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    // closest point on a triangle (Ericson)
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (a + ab * (d1 / (d1 - d3)) - p).length();
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (a + ac * (d2 / (d2 - d6)) - p).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))) - p).length();
    }
    let den = 1.0 / (va + vb + vc);
    (a + ab * (vb * den) + ac * (vc * den) - p).length()
}

fn mesh_dist(m: &Mesh, p: DVec3) -> f64 {
    split_triangles(m).iter().map(|t| tri_dist(p, m.v[t[0] as usize], m.v[t[1] as usize], m.v[t[2] as usize])).fold(f64::INFINITY, f64::min)
}

fn with_skin<T>(s: &Sedan, f: impl FnOnce(&Skin) -> T) -> T {
    use engine_vehiclegen::Design;
    let tube = s.tube();
    let (a, b) = (s.clone(), s.clone());
    let (fz, rz) = (move |x: f64, y: f64| a.front_z(x, y), move |x: f64, y: f64| b.rear_z(x, y));
    f(&Skin { tube: &tube, front_z: &fz, rear_z: &rz })
}

/// The study's member types (§10.2).
const KINDS: [&str; 22] = [
    "crash_box",
    "front_rail",
    "radiator_support",
    "upper_rail",
    "strut_tower",
    "firewall",
    "cowl",
    "floor_pan",
    "sill",
    "seat_crossmember",
    "a_pillar",
    "b_pillar",
    "c_pillar",
    "roof_rail",
    "roof_bow",
    "roof_panel",
    "rear_shelf",
    "rear_rail",
    "rear_tower",
    "boot_floor",
    "tail_panel",
    "quarter",
];

fn check_structure(st: &Structure, what: &str) {
    for k in KINDS {
        assert!(st.members.iter().any(|m| m.kind == k), "{what}: no member of type {k}");
    }
    for m in &st.members {
        assert!(KINDS.contains(&m.kind), "{}", m.kind);
        if m.on_shell {
            assert!(m.mesh.v.is_empty());
            continue;
        }
        let r = check(&m.mesh);
        assert!(r.closed(), "{what}: {}: {} open edges, {} non-manifold", m.name, r.boundary_edges, r.nonmanifold_edges);
        assert_eq!(r.components, 1, "{what}: {}", m.name);
        assert!(r.min_component_volume > 1e-6, "{what}: {}: volume {}", m.name, r.min_component_volume);
        assert!(m.mesh.v.iter().all(|v| v.is_finite()));
        assert!(m.steel_kg() > 0.05 && m.steel_kg() < 40.0, "{what}: {}: {} kg", m.name, m.steel_kg());
    }
    // the shell lattice: 131 nodes in the study's 24 groups, names unique
    assert_eq!(st.nodes.len(), 131, "{what}");
    for (g, n, kg) in GROUPS {
        let of: Vec<_> = st.nodes.iter().filter(|x| x.group == g).collect();
        assert_eq!(of.len(), n, "{what}: group {g}");
        assert!(of.iter().all(|x| x.mass == kg));
    }
    let mut names: Vec<&str> = st.nodes.iter().map(|n| n.name.as_str()).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), 131);
    assert!((st.node_mass() - 487.1).abs() < 0.11);
    // symmetric: every node has its mirror
    for n in &st.nodes {
        let m = DVec3::new(-n.pos.x, n.pos.y, n.pos.z);
        assert!(st.nodes.iter().any(|o| o.group == n.group && (o.pos - m).length() < 1e-9), "{what}: {} has no mirror", n.name);
    }
}

#[test]
fn every_member_is_a_closed_solid_and_the_shell_has_its_131_nodes() {
    let s = sedan("");
    let st = s.unibody();
    // the modelled steel: sections and pressed panels (the outer skin's own
    // solids and the fixed trim make up the rest of the shell's 487 kg)
    let steel = st.steel_mass();
    println!("members {}, steel {:.1} kg of {:.1} kg shell nodes", st.members.len(), steel, st.node_mass());
    for m in &st.members {
        println!("  {:<24} {:<16} {:>6.2} kg  squash {:>6.0} kN  crush {:>5.1} kN  {} tris", m.name, m.kind, m.steel_kg(), m.squash_n() / 1e3, m.crush_n() / 1e3, m.mesh.ntris());
    }
    check_structure(&st, "reference");
    assert!((120.0..350.0).contains(&steel), "{steel}");
    // a rail's mean crush force is tens of kN, and the crash box folds first
    let rail = st.member("front_rail_l").unwrap();
    assert!((40e3..90e3).contains(&rail.crush_n()));
    let cb = st.member("crash_box_l").unwrap();
    assert!(cb.crush_n() < rail.crush_n(), "the crash box folds before the rail");
}

#[test]
fn nodes_sit_on_their_members() {
    // study §10.3 / lint rule 2: within 2 cm of the member's surface (80 %
    // of nodes), none over 5 cm
    let s = sedan("");
    let st = s.unibody();
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
    let shell = b.parts.get("body_shell").unwrap();
    let mut worst: (f64, &str) = (0.0, "");
    let mut near = 0;
    for n in &st.nodes {
        let m = &st.members[n.member];
        let dist = mesh_dist(if m.on_shell { shell } else { &m.mesh }, n.pos);
        if dist > worst.0 {
            worst = (dist, &n.name);
        }
        if dist <= 0.02 {
            near += 1;
        } else {
            println!("{} is {:.1} mm off {}", n.name, dist * 1e3, m.name);
        }
    }
    println!("{near} of 131 within 2 cm; worst {} at {:.1} mm", worst.1, worst.0 * 1e3);
    assert!(near * 10 >= 131 * 8, "{near}");
    assert!(worst.0 <= 0.05, "{} is {:.1} mm off its member", worst.1, worst.0 * 1e3);
}

#[test]
fn the_structure_lies_inside_the_skin() {
    let s = sedan("");
    let st = s.unibody();
    with_skin(&s, |skin| {
        let mut bad = 0;
        for m in &st.members {
            if m.on_shell {
                continue;
            }
            let c = m.mesh.v.iter().map(|p| skin.clearance(*p)).fold(f64::INFINITY, f64::min);
            println!("{:<24} clearance {:.1} mm", m.name, c * 1e3);
            if c < 0.008 {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "members within 8 mm of the skin or outside it");
        for n in &st.nodes {
            assert!(skin.clearance(n.pos) > 0.0, "{} is outside the skin", n.name);
        }
    });
}

#[test]
fn the_structure_cuts_through_no_other_part() {
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
    assert_eq!(b.parts.0.iter().filter(|p| p.0.starts_with("structure_")).count(), 4);
    let st = sedan("").unibody();
    let mut total = 0;
    for (i, m) in st.members.iter().enumerate() {
        if m.on_shell {
            continue;
        }
        // against every other part, and every later member
        for (on, om) in &b.parts.0 {
            if on.starts_with("structure_") {
                continue;
            }
            let hits = pair_crossings(&m.mesh, om);
            let n = hits.len();
            if n > 0 {
                let (lo, hi) = hits.iter().fold((DVec3::splat(9.0), DVec3::splat(-9.0)), |a, h| (a.0.min(*h), a.1.max(*h)));
                println!("{} / {on}: {n}  ({:.3} {:.3} {:.3})..({:.3} {:.3} {:.3})", m.name, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
            }
            total += n;
        }
        for o in &st.members[i + 1..] {
            if o.on_shell {
                continue;
            }
            let hits = pair_crossings(&m.mesh, &o.mesh);
            let n = hits.len();
            if n > 0 {
                let (lo, hi) = hits.iter().fold((DVec3::splat(9.0), DVec3::splat(-9.0)), |a, h| (a.0.min(*h), a.1.max(*h)));
                println!("{} / {}: {n}  ({:.3} {:.3} {:.3})..({:.3} {:.3} {:.3})", m.name, o.name, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
            }
            total += n;
        }
    }
    assert_eq!(total, 0);
}

#[test]
fn parametric_builds_keep_the_structure_inside() {
    for p in [
        r#"{"wheelbase":2.55,"front_overhang":0.80,"rear_overhang":0.85,"width":1.78,"height":1.42,"track":1.53,"tyre_d":0.64,"tyre_w":0.205,"rim_in":16,"rim_w_in":6.5}"#,
        r#"{"ride":0.15,"height":1.52,"tyre_d":0.78,"tyre_w":0.265,"rim_in":17,"rim_w_in":8,"track":1.66,"width":1.90,"rim_style":"steel"}"#,
        r#"{"wheelbase":3.10,"front_overhang":0.95,"rear_overhang":1.15,"width":1.92,"height":1.48,"track":1.64,"tyre_d":0.70,"tyre_w":0.255,"rim_in":19,"rim_w_in":8.5}"#,
        r#"{"wheelbase":2.80,"width":2.00,"height":1.30,"ride":-0.02,"track":1.76,"tyre_d":0.68,"tyre_w":0.295,"rim_in":20,"rim_w_in":10}"#,
    ] {
        let s = sedan(p);
        let st = s.unibody();
        check_structure(&st, p);
        let mut bad = Vec::new();
        with_skin(&s, |skin| {
            for m in st.members.iter().filter(|m| !m.on_shell) {
                let c = m.mesh.v.iter().map(|q| skin.clearance(*q)).fold(f64::INFINITY, f64::min);
                if c < 0.004 {
                    bad.push(format!("{}: {:.1} mm inside the skin", m.name, c * 1e3));
                }
            }
            for n in &st.nodes {
                if skin.clearance(n.pos) <= 0.0 {
                    bad.push(format!("{} is outside the skin", n.name));
                }
            }
        });
        // members do not cut through each other
        for (i, m) in st.members.iter().enumerate() {
            for o in st.members[i + 1..].iter().filter(|o| !o.on_shell && !m.on_shell) {
                let n = pair_crossings(&m.mesh, &o.mesh).len();
                if n > 0 {
                    bad.push(format!("{} / {}: {n}", m.name, o.name));
                }
            }
        }
        assert!(bad.is_empty(), "{p}:\n{}", bad.join("\n"));
    }
}
