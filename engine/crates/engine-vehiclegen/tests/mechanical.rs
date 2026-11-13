//! The mechanical and interior parts (`mechanical`): node counts, masses
//! and mount values of the study's §10.5, closed solids that cut through
//! nothing, every mount and link landing on a real node, each hub held by
//! exactly six members, and the wheel's failure ladder in order.

use std::collections::{BTreeMap, HashMap, HashSet};

use engine_vehiclegen::designs::sedan::{Sedan, SedanParams};
use engine_vehiclegen::mechanical::{BeamKind, MechPart};
use engine_vehiclegen::solid::{check, pair_crossings};
use engine_vehiclegen::{build, design_from_json, BuildOpts, Design};
use glam::DVec3;

fn sedan(p: &str) -> Sedan {
    let p: SedanParams = serde_json::from_str(if p.is_empty() { "{}" } else { p }).unwrap();
    Sedan::new(p)
}

fn parts(s: &Sedan) -> (engine_vehiclegen::structure::Structure, Vec<MechPart>) {
    let st = s.unibody();
    let m = s.mechanical(&st);
    (st, m)
}

fn part<'a>(m: &'a [MechPart], name: &str) -> &'a MechPart {
    m.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no part {name}"))
}

#[test]
fn node_counts_and_masses_follow_the_study() {
    let (_, m) = parts(&sedan(""));
    // (part, nodes, kg): §10.5
    for (name, n, kg) in [
        ("impact_beam_front", 10, 8.0),
        ("impact_beam_rear", 10, 6.0),
        ("subframe_front", 7, 46.5),
        ("subframe_rear", 6, 42.0),
        ("hub_FL", 4, 17.0),
        ("hub_RR", 4, 24.0),
        ("steering_rack", 2, 10.0),
        ("wheel_FL", 5, 19.0),
        ("hub_cap_RL", 3, 0.6),
        ("engine", 10, 170.0),
        ("transmission", 2, 45.0),
        ("driveshaft", 2, 1.0),
        ("differential", 3, 42.0),
        ("exhaust", 13, 18.2),
        ("fuel_tank", 8, 40.0),
        ("radiator", 4, 5.6),
        ("seat_FL", 7, 18.9),
        ("seat_rear", 8, 18.4),
        ("dash", 7, 45.0),
        ("console", 4, 5.0),
    ] {
        let p = part(&m, name);
        assert_eq!(p.nodes.len(), n, "{name}");
        assert!((p.mass() - kg).abs() < 1e-6, "{name}: {} kg", p.mass());
    }
    // arms, struts, tie rods and half-shafts are beams between other parts'
    // nodes: no nodes of their own
    for p in m.iter().filter(|p| ["arm_", "strut_", "toe_link_", "halfshaft_"].iter().any(|k| p.name.starts_with(k))) {
        assert!(p.nodes.is_empty() && !p.links.is_empty() && p.binds.len() >= 3, "{}", p.name);
    }
    // names are unique
    let mut names = HashSet::new();
    for n in m.iter().flat_map(|p| &p.nodes) {
        assert!(names.insert(n.name.clone()), "{} twice", n.name);
    }
    // mass by class (§10.1 lists the bands; the report goes to the docs)
    let mut by: BTreeMap<&str, f64> = BTreeMap::new();
    for p in &m {
        *by.entry(p.class).or_default() += p.mass();
    }
    println!("{by:?}");
    assert!((150.0..=190.0).contains(&by["engine"]));
    assert!((4.0 * 18.0..=4.0 * 20.0 + 3.0).contains(&by["wheel"]));
}

#[test]
fn mounts_and_links_land_on_real_nodes_in_named_groups() {
    let (st, m) = parts(&sedan(""));
    let mut pos: HashMap<String, DVec3> = st.nodes.iter().map(|n| (n.name.clone(), n.pos)).collect();
    pos.extend(m.iter().flat_map(|p| &p.nodes).map(|n| (n.name.clone(), n.pos)));
    let mut groups = HashSet::new();
    for p in &m {
        for i in &p.internal {
            assert!(!i.pairs.is_empty());
            for &(a, b) in &i.pairs {
                assert!(a < b && b < p.nodes.len(), "{}", p.name);
            }
            // nothing inside a part is weaker than a real member would be
            assert!(i.v.deform > 0.0 && i.v.strength > i.v.deform);
        }
        for mt in &p.mounts {
            assert!(groups.insert(mt.group.clone()), "group {} twice", mt.group);
            for n in mt.from.iter().chain(&mt.to) {
                assert!(pos.contains_key(n), "{}: mount {} names {n}", p.name, mt.group);
            }
            assert!(mt.from.iter().all(|n| p.nodes.iter().any(|x| &x.name == n)), "{}", mt.group);
            assert!(mt.beams <= mt.from.len() * mt.to.len(), "{}: {} beams from {} × {}", mt.group, mt.beams, mt.from.len(), mt.to.len());
            assert!(mt.v.deform > 0.0 && mt.v.strength >= mt.v.deform, "{}", mt.group);
            if mt.kind == BeamKind::Support {
                assert!(mt.precomp <= 1.0);
            }
            // a mount beam is short: its nearest pairs are neighbours
            let mut d: Vec<f64> = mt.from.iter().flat_map(|a| mt.to.iter().map(|b| (pos[a] - pos[b]).length())).collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(d[mt.beams - 1] < 0.95, "{}: its {}th beam is {:.2} m", mt.group, mt.beams, d[mt.beams - 1]);
        }
        for l in &p.links {
            assert!(pos.contains_key(&l.a) && pos.contains_key(&l.b), "{}: {} - {}", l.name, l.a, l.b);
            assert!(l.v.strength.is_finite(), "{} cannot break", l.name);
        }
        for b in &p.binds {
            assert!(pos.contains_key(b), "{} binds to {b}", p.name);
        }
        // every part that owns nodes hangs on a mount with a finite break
        // force, or on breakable links (the engine side of the gearbox and
        // the rack on its subframe are bolted solid: the study's table)
        if !p.nodes.is_empty() {
            let breakable = p.mounts.iter().any(|x| x.v.strength.is_finite());
            let held_by_links = m.iter().flat_map(|q| &q.links).any(|l| p.nodes.iter().any(|n| n.name == l.a || n.name == l.b));
            assert!(breakable || held_by_links || p.name == "steering_rack", "{} has no breakable mount", p.name);
        }
    }
}

#[test]
fn each_hub_is_held_by_six_members_and_the_ladder_is_ordered() {
    let (_, m) = parts(&sedan(""));
    let links: Vec<_> = m.iter().flat_map(|p| &p.links).collect();
    for c in ["FL", "FR", "RL", "RR"] {
        let hub = part(&m, &format!("hub_{c}"));
        let held: Vec<_> = links.iter().filter(|l| hub.nodes.iter().any(|n| n.name == l.a || n.name == l.b)).collect();
        assert_eq!(held.len(), 6, "hub_{c}: {:?}", held.iter().map(|l| &l.name).collect::<Vec<_>>());
        // one wheel group of 8 beams
        let w = part(&m, &format!("wheel_{c}"));
        assert_eq!((w.mounts.len(), w.mounts[0].beams), (1, 8));
    }
    // toe link bends < arm bends < wheel leaves < strut breaks < toe link
    // breaks < arm breaks < front subframe leaves (§4.4, §10.5)
    let link = |n: &str| links.iter().find(|l| l.name == n).unwrap().v;
    let (toe, arm, strut) = (link("toe_link_fl"), link("arm_lower_fl_f"), link("strut_fl"));
    let wheel = part(&m, "wheel_FL").mounts[0].v.strength;
    let sub = part(&m, "subframe_front").mounts[0].v.strength;
    let ladder = [toe.deform, arm.deform, wheel, strut.strength, toe.strength, arm.strength, sub];
    assert!(ladder.windows(2).all(|w| w[0] < w[1]), "{ladder:?}");
}

#[test]
fn parts_are_closed_solids_and_cut_through_nothing() {
    let s = sedan("");
    let (_, m) = parts(&s);
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
    let mut total = 0;
    let mut tris = 0;
    for (i, p) in m.iter().enumerate() {
        if p.mesh.v.is_empty() {
            // (a wheel's solids are the rim and the tyre)
            assert!(p.class == "wheel" && p.meshes.len() == 2, "{} has no solid", p.name);
            continue;
        }
        let r = check(&p.mesh);
        assert!(r.closed(), "{}: {} open edges, {} non-manifold", p.name, r.boundary_edges, r.nonmanifold_edges);
        assert!(r.min_component_volume > 1e-7, "{}: a piece's volume is {}", p.name, r.min_component_volume);
        // above the road
        let low = p.mesh.v.iter().map(|v| v.y).fold(f64::INFINITY, f64::min);
        assert!(low > 0.12 || p.class == "wheel", "{} hangs {:.3} m above the road", p.name, low);
        assert!(b.parts.get(&p.name).is_some(), "{} is not a render part", p.name);
        if p.class != "interior" {
            tris += p.mesh.ntris();
        }
        let own: Vec<&String> = p.meshes.iter().collect();
        for (on, om) in &b.parts.0 {
            if on == &p.name || own.contains(&on) || m[..i].iter().any(|q| &q.name == on) {
                continue;
            }
            let hits = pair_crossings(&p.mesh, om);
            if !hits.is_empty() {
                let (lo, hi) = hits.iter().fold((DVec3::splat(9.0), DVec3::splat(-9.0)), |a, h| (a.0.min(*h), a.1.max(*h)));
                println!("{} / {on}: {}  ({:.3} {:.3} {:.3})..({:.3} {:.3} {:.3})", p.name, hits.len(), lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
            }
            total += hits.len();
        }
    }
    println!("mechanical triangles {tris}");
    assert_eq!(total, 0);
    assert!(tris <= 60_000, "{tris}");
}

#[test]
fn parametric_builds_keep_their_parts() {
    for p in [
        r#"{"wheelbase":2.55,"front_overhang":0.80,"rear_overhang":0.85,"width":1.78,"height":1.42,"track":1.53,"tyre_d":0.64,"tyre_w":0.205,"rim_in":16,"rim_w_in":6.5}"#,
        r#"{"ride":0.15,"height":1.52,"tyre_d":0.78,"tyre_w":0.265,"rim_in":17,"rim_w_in":8,"track":1.66,"width":1.90,"rim_style":"steel"}"#,
        r#"{"wheelbase":3.10,"front_overhang":0.95,"rear_overhang":1.15,"width":1.92,"height":1.48,"track":1.64,"tyre_d":0.70,"tyre_w":0.255,"rim_in":19,"rim_w_in":8.5}"#,
        r#"{"wheelbase":2.80,"width":2.00,"height":1.30,"ride":-0.02,"track":1.76,"tyre_d":0.68,"tyre_w":0.295,"rim_in":20,"rim_w_in":10}"#,
    ] {
        let (_, m) = parts(&sedan(p));
        assert_eq!(m.len(), parts(&sedan("")).1.len());
        for q in m.iter().filter(|q| !q.mesh.v.is_empty()) {
            let r = check(&q.mesh);
            assert!(r.closed() && r.min_component_volume > 1e-7, "{p}: {}", q.name);
            assert!(q.mesh.v.iter().all(|v| v.is_finite()));
        }
    }
}
