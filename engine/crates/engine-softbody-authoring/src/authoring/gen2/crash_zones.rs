//! `crash.json`: the crash reference suite's zones and gauges for a rigged design
//! (`engine_softbody::crash_reference::ZoneMap`), from the design's own definitions.
//!
//! **Zones.** Every node takes the zone of what it is part of, by one rule: a shell node
//! its member type's ([`engine_vehiclegen::crash::member_zone`]); a mechanical or interior
//! part's node its part's ([`engine_vehiclegen::mechanical::MechPart::zone`]); a closure's
//! node its panel's ([`part_zone`]); any other body node (a shut line's edge node, a latch
//! striker) the zone of the nearest shell node. The nodes' names are then written as the
//! fewest prefixes (cut after an `_` or before a digit) that give every node its own zone
//! by the longest match; a name no prefix covers exactly is listed whole.
//!
//! **Gauges.** Node pairs the generator picks from the parts' geometry (never by index):
//! wheelbase each side, the front door opening at the belt, the cabin's width and the roof's
//! height at the B-pillars, the front overhang. The car's own frame: +Z forward, +Y up,
//! +X left (the generator's `_l` nodes).

use super::build::Part;
use super::rig::Rig;
use super::vg_input::VgInput;
use engine_softbody::crash_reference::{Gauge, Section, ZoneMap};
use engine_vehiclegen::crash::Zone;
use glam::Vec3;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// The file's name in the mod folder (`<design>/crash.json`).
pub const FILE: &str = "crash.json";

/// A body panel's zone; the body itself is its members' ([`Part::Body`]: None).
pub fn part_zone(p: Part) -> Option<Zone> {
    Some(match p {
        Part::Body => return None,
        // (the definition parts' own nodes say theirs; the rest of this part is carried)
        Part::Engine => Zone::Other,
        Part::DoorFL | Part::DoorFR | Part::DoorRL | Part::DoorRR => Zone::Door,
        Part::Hood | Part::FenderL | Part::FenderR => Zone::FrontUpper,
        Part::Trunk => Zone::Rear,
        Part::BumperF => Zone::FrontLower,
        Part::BumperR => Zone::Rear,
    })
}

/// The crash reference's name for a zone.
pub fn section(z: Zone) -> Section {
    match z {
        Zone::FrontLower => Section::FrontLower,
        Zone::FrontUpper => Section::FrontUpper,
        Zone::Cabin => Section::Cabin,
        Zone::Door => Section::Door,
        Zone::Rear => Section::Rear,
        Zone::Wheel => Section::Wheel,
        Zone::Other => Section::Other,
    }
}

/// Every node's zone, in the rig's node order. Err: a design with no defined structure
/// (a Python build), or a member type with no zone.
pub fn node_zones(inp: &VgInput, r: &Rig) -> Result<Vec<Zone>, String> {
    let st = inp.structure.as_ref().ok_or("no body-in-white definition")?;
    let shell: HashMap<&str, Zone> = st
        .nodes
        .iter()
        .map(|n| {
            let m = &st.members[n.member];
            m.zone().map(|z| (n.name.as_str(), z)).ok_or(format!("member type {} has no crash zone", m.kind))
        })
        .collect::<Result<_, _>>()?;
    let parts: HashMap<&str, Zone> = inp.mechanical.iter().map(|p| (p.name.as_str(), p.zone)).collect();
    let nodes = &r.car.nodes;
    let own: Vec<Option<Zone>> = nodes
        .iter()
        .map(|nd| {
            if let Some(&z) = shell.get(nd.name.as_str()) {
                return Some(z);
            }
            if let Some(z) = nd.sub.as_deref().and_then(|s| parts.get(s)) {
                return Some(*z);
            }
            part_zone(nd.part)
        })
        .collect();
    let shell_ids: Vec<usize> = (0..nodes.len()).filter(|&i| shell.contains_key(nodes[i].name.as_str())).collect();
    if shell_ids.is_empty() {
        return Err("no shell nodes in the rig".into());
    }
    Ok((0..nodes.len())
        .map(|i| {
            own[i].unwrap_or_else(|| {
                let g = r.g(i);
                let j = *shell_ids.iter().min_by(|&&a, &&b| r.g(a).distance(g).total_cmp(&r.g(b).distance(g))).unwrap();
                own[j].unwrap()
            })
        })
        .collect())
}

/// The zone the longest matching prefix gives `name` (the crash reference's rule).
pub fn lookup(parts: &[(String, Section)], name: &str) -> Option<Section> {
    parts.iter().filter(|(p, _)| name.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map(|x| x.1)
}

/// The fewest prefixes (cut after an `_` or before a digit, or the whole name) whose
/// longest match gives every name its zone: shortest first, each written where the
/// majority under it differs from what a shorter one gives; any name still wrong is
/// then listed whole.
pub fn prefixes(named: &[(String, Zone)]) -> Vec<(String, Zone)> {
    let cuts = |n: &str| -> Vec<String> {
        let b = n.as_bytes();
        let mut v: Vec<String> = (1..b.len())
            .filter(|&i| b[i - 1] == b'_' || (b[i - 1].is_ascii_alphabetic() && b[i].is_ascii_digit()))
            .map(|i| n[..i].to_string())
            .collect();
        v.push(n.to_string());
        v
    };
    let mut cand: Vec<String> = named.iter().flat_map(|(n, _)| cuts(n)).collect::<BTreeSet<_>>().into_iter().collect();
    cand.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    let resolve = |out: &BTreeMap<String, Zone>, n: &str| -> Option<Zone> {
        out.iter().filter(|(p, _)| n.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map(|x| *x.1)
    };
    let mut out: BTreeMap<String, Zone> = BTreeMap::new();
    for p in &cand {
        let mut count: BTreeMap<Zone, usize> = BTreeMap::new();
        for (n, z) in named {
            if n.starts_with(p.as_str()) {
                *count.entry(*z).or_default() += 1;
            }
        }
        // (the most nodes; a tie to the first zone)
        let most = count.iter().map(|x| *x.1).max().unwrap_or(0);
        let Some(maj) = count.iter().find(|x| *x.1 == most).map(|x| *x.0) else { continue };
        if resolve(&out, p) != Some(maj) {
            out.insert(p.clone(), maj);
        }
    }
    loop {
        let wrong: Vec<(String, Zone)> = named.iter().filter(|(n, z)| resolve(&out, n) != Some(*z)).cloned().collect();
        if wrong.is_empty() {
            break;
        }
        out.extend(wrong);
    }
    out.into_iter().collect()
}

/// The standard gauges, by node names picked from the parts' geometry. Empty when the
/// car has no structure or no four wheels.
pub fn gauges(inp: &VgInput, r: &Rig) -> Vec<Gauge> {
    let Some(st) = inp.structure.as_ref() else { return Vec::new() };
    let nodes = &r.car.nodes;
    let id: HashMap<&str, usize> = nodes.iter().enumerate().map(|(i, n)| (n.name.as_str(), i)).collect();
    let at = |n: &str| id.get(n).map(|&i| r.g(i));
    // shell nodes on members of a type, on one side (+1 left, -1 right)
    let on = |kind: &str, side: f32| -> Vec<&str> {
        st.nodes
            .iter()
            .filter(|n| st.members[n.member].kind == kind)
            .map(|n| n.name.as_str())
            .filter(|n| at(n).is_some_and(|p| p.x * side > 0.0))
            .collect()
    };
    let nearest = |of: &[&str], g: Vec3| -> Option<String> {
        of.iter().filter_map(|n| at(n).map(|p| (p.distance(g), *n))).min_by(|a, b| a.0.total_cmp(&b.0)).map(|x| x.1.to_string())
    };
    let gauge = |name: &str, a: Option<String>, b: Option<String>| -> Option<Gauge> { Some(Gauge { name: name.into(), a: a?, b: b? }) };
    // Each wheel's axle node nearest the car's centre plane (its inner axle node, on the hub).
    let wheels: Vec<(Vec3, String)> = inp
        .mechanical
        .iter()
        .filter(|p| p.class == "wheel" && p.name.starts_with("wheel_"))
        .filter_map(|p| {
            let pts: Vec<(Vec3, &str)> = p.nodes.iter().filter_map(|n| at(&n.name).map(|g| (g, n.name.as_str()))).collect();
            let c = pts.iter().map(|x| x.0).sum::<Vec3>() / pts.len().max(1) as f32;
            let inner = pts.iter().min_by(|a, b| a.0.x.abs().total_cmp(&b.0.x.abs()))?;
            Some((c, inner.1.to_string()))
        })
        .collect();
    if wheels.len() != 4 {
        return Vec::new();
    }
    let axle_z_front = wheels.iter().map(|w| w.0.z).fold(f32::MIN, f32::max);
    let wheel = |side: f32, front: bool| -> Option<String> {
        wheels
            .iter()
            .filter(|w| w.0.x * side > 0.0)
            .max_by(|a, b| if front { a.0.z.total_cmp(&b.0.z) } else { b.0.z.total_cmp(&a.0.z) })
            .map(|w| w.1.clone())
    };
    // The B-pillar's node at the belt: its member's lowest (the top is on the roof rail).
    let b_belt = |side: f32| -> Option<String> {
        on("b_pillar", side).into_iter().min_by(|a, b| at(a).unwrap().y.total_cmp(&at(b).unwrap().y)).map(str::to_string)
    };
    let body: Vec<&str> = nodes.iter().filter(|n| n.part == Part::Body && !n.contact).map(|n| n.name.as_str()).collect();
    // The front door opening at the belt: from the B-pillar's belt node forward to the body
    // node nearest the front door's leading edge at that height.
    let door = |side: f32| -> Option<Gauge> {
        let b = b_belt(side)?;
        let pb = at(&b)?;
        let dp = if side > 0.0 { Part::DoorFL } else { Part::DoorFR };
        let z_front = nodes.iter().enumerate().filter(|(_, n)| n.part == dp).map(|(i, _)| r.g(i).z).fold(f32::MIN, f32::max);
        if z_front == f32::MIN {
            return None;
        }
        let others: Vec<&str> = body.iter().copied().filter(|n| *n != b).collect();
        gauge(if side > 0.0 { "door_left" } else { "door_right" }, nearest(&others, Vec3::new(pb.x, pb.y, z_front)), Some(b))
    };
    let roof = |side: f32| -> Option<Gauge> {
        let pb = at(&b_belt(side)?)?;
        let rail = nearest(&on("roof_rail", side), pb)?;
        let sill = nearest(&on("sill", side), at(&rail)?)?;
        gauge("roof_height", Some(sill), Some(rail))
    };
    // The front overhang: the front fascia's centre (its most forward node on the centre
    // plane) to the shell node nearest the front axle's line (across the car) at that
    // height; of a mirrored pair, the left one.
    let overhang = || -> Option<Gauge> {
        let fascia: Vec<(Vec3, &str)> = nodes.iter().enumerate().filter(|(_, n)| n.part == Part::BumperF).map(|(i, n)| (r.g(i), n.name.as_str())).collect();
        let xmin = fascia.iter().map(|x| x.0.x.abs()).fold(f32::MAX, f32::min);
        let (pc, c) = fascia.iter().filter(|x| x.0.x.abs() <= xmin + 0.01).max_by(|a, b| a.0.z.total_cmp(&b.0.z))?;
        let off = |p: Vec3| (p.y - pc.y).hypot(p.z - axle_z_front);
        let axle = st
            .nodes
            .iter()
            .filter_map(|n| at(&n.name).map(|p| (off(p), -p.x, n.name.as_str())))
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)))?;
        gauge("front_overhang", Some(c.to_string()), Some(axle.2.to_string()))
    };
    [
        gauge("wheelbase_left", wheel(1.0, true), wheel(1.0, false)),
        gauge("wheelbase_right", wheel(-1.0, true), wheel(-1.0, false)),
        door(1.0),
        door(-1.0),
        gauge("cabin_width", b_belt(1.0), b_belt(-1.0)),
        roof(1.0),
        overhang(),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// The design's `crash.json` (None: a design without a defined structure).
pub fn zone_map(inp: &VgInput, r: &Rig) -> Option<ZoneMap> {
    let zones = node_zones(inp, r).ok()?;
    let named: Vec<(String, Zone)> = r.car.nodes.iter().zip(&zones).map(|(n, z)| (n.name.clone(), *z)).collect();
    let parts: Vec<(String, Section)> = prefixes(&named).into_iter().map(|(p, z)| (p, section(z))).collect();
    Some(ZoneMap { parts, gauges: gauges(inp, r) })
}

/// `<design>/crash.json`, for the mod folder's files.
pub fn file(inp: &VgInput, r: &Rig) -> Option<(String, Vec<u8>)> {
    let m = zone_map(inp, r)?;
    Some((format!("{}/{FILE}", inp.design()), serde_json::to_vec_pretty(&m).ok()?))
}
