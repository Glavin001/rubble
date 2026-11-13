//! The generated car as BeamNG content: one JBeam file per part, a `.pc`
//! configuration, and (from [`super::mesh`]) a Collada render mesh with its
//! `materials.json`, whose meshes bind to their part's node group as
//! flexbodies. The same loader that reads community cars reads these
//! (`jbeam::load_jbeam`, the page's `?mod=`), so the generated car is plain,
//! editable JBeam.

use super::build::*;
use serde_json::{json, Map, Value};

/// The vehicle's folder name under `vehicles/`.
pub const NAME: &str = "gen2";
/// The engine-to-dash stops: engage with this much length left (m), at this
/// limit spring (N/m) and damping (N·s/m).
pub const STOP_LEFT: f32 = 0.08;
pub const STOP_SPRING: f32 = 2.0e6;
pub const STOP_DAMP: f32 = 4000.0;
/// Each rear wheel's parking brake torque (N·m): FMVSS 135 S7.12 asks the
/// parking brake to hold the car on a 20 % grade. At the Accord's 1,950 kg
/// GVWR on a 0.33 m tyre that is 1,950 × 9.81 × sin(atan 0.2) × 0.33 / 2 =
/// 619 N·m a wheel; twice that for margin (a cable brake's design factor).
pub const PARKING_TORQUE: f32 = 1240.0;

fn num(x: f32) -> Value {
    if !x.is_finite() {
        return json!("FLT_MAX");
    }
    // Four significant digits.
    let mag = x.abs().max(1e-9).log10().floor() as i32;
    let q = 10f64.powi(3 - mag);
    json!(((x as f64) * q).round() / q)
}

fn pos(x: f32) -> Value {
    json!(((x as f64) * 1e4).round() / 1e4)
}

/// A flexbody row: mesh name, its node groups, and its deform group (glass,
/// lamps) if any.
#[derive(Clone, Debug)]
pub struct Flex {
    pub mesh: String,
    pub groups: Vec<String>,
    pub deform_group: Option<String>,
    /// A weighted skin's sidecar file (`"rubbleSkin"`, a Rubble extension:
    /// [`crate::flexskin`]) and the mesh's placement (`"pos"`, JBeam frame).
    pub skin: Option<String>,
    pub pos: Option<[f32; 3]>,
    /// The definition part whose JBeam part carries the row
    /// ([`super::build::Bind::owner`]); None: its [`Part`]'s.
    pub owner: Option<String>,
}

/// A definition part's JBeam part name.
pub fn sub_name(sub: &str) -> String {
    format!("{NAME}_{sub}")
}

/// The definition parts, in node order, then those that are beams only (an
/// arm's legs, a strut, a tie rod, a shaft) in beam order.
pub fn subs(car: &Car) -> Vec<(Part, String)> {
    let mut out: Vec<(Part, String)> = Vec::new();
    let named = car
        .nodes
        .iter()
        .map(|n| (n.part, &n.sub))
        .chain(car.beams.iter().map(|b| (b.part, &b.sub)));
    for (part, sub) in named {
        if let Some(s) = sub {
            if !out.iter().any(|o| &o.1 == s) {
                out.push((part, s.clone()));
            }
        }
    }
    out
}

fn rows(header: Value, rows: Vec<Value>) -> String {
    let mut s = String::from("[\n      ");
    s += &header.to_string();
    for r in rows {
        s += ",\n      ";
        s += &r.to_string();
    }
    s += "\n    ]";
    s
}

/// JBeam files (path under the vehicles folder, text) and the `.pc`.
pub fn jbeam_files(
    car: &Car,
    flex: &[(Part, Flex)],
    deform_groups: &[(String, Vec<usize>, f32)],
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    // Main part: slots and the reference nodes.
    let subs = subs(car);
    let slots: Vec<Value> = Part::ALL
        .iter()
        .map(|p| p.name().to_string())
        .chain(subs.iter().map(|s| sub_name(&s.1)))
        .map(|p| json!([p, p, p]))
        .collect();
    let mut main = String::new();
    main += "{\n\"gen2\": {\n";
    main += "    \"information\": {\"authors\": \"Rubble (generated: engine_softbody::authoring::gen2)\", \"name\": \"Rubble gen2 sedan\", \"value\": 0},\n";
    main += "    \"slotType\": \"main\",\n";
    main += &format!(
        "    \"slots\": {},\n",
        rows(json!(["type", "default", "description"]), slots)
    );
    let n = |i: usize| car.nodes[i].name.clone();
    main += &format!(
        "    \"refNodes\": {}\n",
        rows(
            json!(["ref:", "back:", "left:", "up:"]),
            vec![json!([
                n(car.refs[0]),
                n(car.refs[1]),
                n(car.refs[2]),
                n(car.refs[3])
            ])]
        )
    );
    main += "}\n}\n";
    out.push((format!("{NAME}/{NAME}.jbeam"), main));

    let units: Vec<(Part, Option<&str>)> = Part::ALL
        .iter()
        .map(|&p| (p, None))
        .chain(subs.iter().map(|s| (s.0, Some(s.1.as_str()))))
        .collect();
    for (part, sub) in units {
        let pname = sub.map_or(part.name().to_string(), sub_name);
        let pname = pname.as_str();
        let mut sec: Vec<String> = Vec::new();
        sec.push(
            "    \"information\": {\"authors\": \"Rubble (generated)\", \"name\": \"".to_string()
                + pname
                + "\"}",
        );
        sec.push(format!("    \"slotType\": \"{}\"", pname));
        // Nodes.
        let mut nrows = Vec::new();
        for (i, nd) in car.nodes.iter().enumerate() {
            if nd.part != part || nd.sub.as_deref() != sub {
                continue;
            }
            let _ = i;
            let mut o = Map::new();
            o.insert("nodeWeight".into(), num(nd.mass));
            o.insert("selfCollision".into(), json!(nd.self_collision));
            o.insert("collision".into(), json!(true));
            o.insert("frictionCoef".into(), json!(0.7));
            o.insert(
                "group".into(),
                if nd.no_group {
                    json!("")
                } else {
                    json!(pname)
                },
            );
            for (k, v) in &nd.opts {
                o.insert(k.clone(), v.clone());
            }
            nrows.push(json!([
                nd.name,
                pos(nd.p.x),
                pos(nd.p.y),
                pos(nd.p.z),
                Value::Object(o)
            ]));
        }
        sec.push(format!(
            "    \"nodes\": {}",
            rows(json!(["id", "posX", "posY", "posZ"]), nrows)
        ));
        // Beams.
        let mut brows = Vec::new();
        let mut list: Vec<usize> = (0..car.beams.len())
            .filter(|&j| car.beams[j].part == part && car.beams[j].sub.as_deref() == sub)
            .collect();
        list.sort_by_key(|&j| (car.beams[j].role, car.beams[j].break_group.clone()));
        for j in list {
            let b = &car.beams[j];
            let sp = b.role.spec(b.part);
            let mut o = Map::new();
            if let Some(f) = car.fixed.get(&j) {
                // A defined beam: its own values and kind.
                use super::build::FixedKind;
                o.insert(
                    "beamType".into(),
                    json!(match f.kind {
                        FixedKind::Normal => "|NORMAL",
                        FixedKind::Support => "|SUPPORT",
                        FixedKind::Bounded { .. } => "|BOUNDED",
                    }),
                );
                o.insert("beamSpring".into(), num(b.k));
                o.insert("beamDamp".into(), num(b.c));
                o.insert("beamDeform".into(), num(b.deform));
                o.insert("beamStrength".into(), num(b.strength));
                match f.kind {
                    FixedKind::Normal => {}
                    FixedKind::Support => {
                        o.insert("beamLongBound".into(), json!(100.0));
                    }
                    FixedKind::Bounded {
                        travel,
                        limit_k,
                        limit_c,
                    } => {
                        o.insert("beamShortBound".into(), num(travel));
                        o.insert("beamLongBound".into(), num(travel));
                        o.insert("beamLimitSpring".into(), num(limit_k));
                        o.insert("beamLimitDamp".into(), num(limit_c));
                        o.insert("boundZone".into(), json!(0.02));
                    }
                }
                if f.limit_expansion > 0.0 {
                    o.insert("deformLimitExpansion".into(), num(f.limit_expansion));
                }
                if let Some(pc) = car.precomp.get(&j) {
                    o.insert(
                        "beamPrecompression".into(),
                        json!((*pc as f64 * 1e6).round() / 1e6),
                    );
                }
                o.insert(
                    "breakGroup".into(),
                    json!(b.break_group.clone().unwrap_or_default()),
                );
                if let Some((g, ratio)) = &f.deform_group {
                    o.insert("deformGroup".into(), json!(g));
                    o.insert("deformationTriggerRatio".into(), json!(ratio));
                }
                brows.push(json!([n(b.a), n(b.b), Value::Object(o)]));
                continue;
            }
            o.insert(
                "beamType".into(),
                json!(if sp.support { "|SUPPORT" } else { "|NORMAL" }),
            );
            if b.role == Role::Stop {
                // A gapped stop (spec §5): free until it has shortened to
                // STOP_LEFT of its length, then the limit spring.
                let l = car.nodes[b.a].p.distance(car.nodes[b.b].p);
                let mut s = Map::new();
                s.insert("beamType".into(), json!("|BOUNDED"));
                s.insert("beamSpring".into(), json!(0));
                s.insert("beamDamp".into(), json!(0));
                let ks = env_f("GEN3_STOP_SPRING", STOP_SPRING);
                s.insert("beamLimitSpring".into(), json!(ks));
                s.insert(
                    "beamLimitDamp".into(),
                    json!(STOP_DAMP * (ks / STOP_SPRING).sqrt()),
                );
                // `GEN3_STOP_CLOSE`: engage after this much closure instead.
                let close = std::env::var("GEN3_STOP_CLOSE")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .or_else(|| car.stop_close.get(&j).copied())
                    .unwrap_or(l - STOP_LEFT);
                s.insert("beamShortBound".into(), num((close / l).max(0.05)));
                s.insert("beamLongBound".into(), json!(100.0));
                s.insert("boundZone".into(), json!(0.02));
                s.insert("beamStrength".into(), json!("FLT_MAX"));
                s.insert("breakGroup".into(), json!(""));
                brows.push(json!([n(b.a), n(b.b), Value::Object(s)]));
                continue;
            }
            if b.role == Role::Clip {
                // ANISOTROPIC (spec §20): soft while compressed, the bolt in
                // tension.
                o.insert("beamType".into(), json!("|ANISOTROPIC"));
                o.insert("beamSpring".into(), num(b.k * CLIP_COMPRESSION));
                o.insert("beamDamp".into(), num(b.c * CLIP_COMPRESSION.sqrt()));
                o.insert("springExpansion".into(), num(b.k));
                o.insert("dampExpansion".into(), num(b.c));
            } else {
                o.insert("beamSpring".into(), num(b.k));
                o.insert("beamDamp".into(), num(b.c));
            }
            o.insert("beamDeform".into(), num(b.deform));
            o.insert("beamStrength".into(), num(b.strength));
            if b.deform_limit > 0.0 {
                o.insert("deformLimit".into(), json!(b.deform_limit));
            }
            if sp.limit_expansion {
                o.insert("deformLimitExpansion".into(), json!(1.2));
            }
            if sp.support {
                o.insert("beamLongBound".into(), json!(100.0));
            }
            if let Some(pc) = car.precomp.get(&j) {
                o.insert("beamPrecompression".into(), json!((*pc as f64 * 1e6).round() / 1e6));
            }
            o.insert(
                "breakGroup".into(),
                json!(b.break_group.clone().unwrap_or_default()),
            );
            let dg: Vec<&(String, Vec<usize>, f32)> = deform_groups
                .iter()
                .filter(|(_, ns, _)| ns.contains(&b.a) && ns.contains(&b.b))
                .collect();
            if let Some((g, _, ratio)) = dg.first() {
                o.insert("deformGroup".into(), json!(g));
                o.insert("deformationTriggerRatio".into(), json!(ratio));
            }
            brows.push(json!([n(b.a), n(b.b), Value::Object(o)]));
        }
        sec.push(format!(
            "    \"beams\": {}",
            rows(json!(["id1:", "id2:"]), brows)
        ));
        // Collision triangles.
        // A triangle with a break group sits between modifier rows setting
        // it and clearing it (triangles without one first, unchanged).
        let mut trows: Vec<Value> = Vec::new();
        let mut cur: Option<&String> = None;
        let mut own: Vec<&super::build::Tri> = car
            .tris
            .iter()
            .filter(|t| t.part == part && sub.is_none())
            .collect();
        own.sort_by_key(|t| t.group.is_some());
        for t in own {
            if t.group.as_ref() != cur {
                cur = t.group.as_ref();
                trows.push(json!({"breakGroup": cur.cloned().unwrap_or_default()}));
            }
            trows.push(json!([n(t.n[0]), n(t.n[1]), n(t.n[2])]));
        }
        if cur.is_some() {
            trows.push(json!({"breakGroup": ""}));
        }
        let trows: Vec<Value> = match car.tri_aero {
            Some((d, l)) if !trows.is_empty() => {
                let mut v = vec![json!({"dragCoef": num(d), "liftCoef": num(l)})];
                v.extend(trows);
                v
            }
            _ => trows,
        };
        if !trows.is_empty() {
            sec.push(format!(
                "    \"triangles\": {}",
                rows(json!(["id1:", "id2:", "id3:"]), trows)
            ));
        }
        // Wheels (raycast on the body's hub nodes).
        // Torsion bars (the shell's floor rows).
        if part == Part::Body && sub.is_none() && !car.torsion.is_empty() {
            let trows: Vec<Value> = car
                .torsion
                .iter()
                .map(|t| {
                    json!([
                        n(t.n[0]),
                        n(t.n[1]),
                        n(t.n[2]),
                        n(t.n[3]),
                        {"spring": num(t.k), "damp": num(t.c), "deform": num(t.deform), "strength": "FLT_MAX"}
                    ])
                })
                .collect();
            sec.push(format!(
                "    \"torsionbars\": {}",
                rows(json!(["id1:", "id2:", "id3:", "id4:"]), trows)
            ));
        }
        if part == Part::Body && sub.is_none() && !car.wheels.is_empty() {
            // A 215/60 R16 on a 16" rim, for JBeam pressure wheels (spec §25)
            // as well as raycast wheels (which read radius and width): Rubble's
            // own values (the PART2_CAR fixture's, at a 20 kg wheel: 16 rays,
            // 32 hub and 32 tread nodes).
            let mut w = vec![json!({
                "radius": 0.33, "tireWidth": 0.215, "hubRadius": 0.2, "hubWidth": 0.2, "numRays": 16,
                "hubNodeWeight": 0.35, "nodeWeight": 0.3, "pressurePSI": 32, "frictionCoef": 1.0, "hasTire": true,
                "brakeTorque": 2000,
                "hubBeamSpring": 1000000, "hubBeamDamp": 60, "hubSideBeamSpring": 1000000, "hubSideBeamDamp": 60,
                "hubTreadBeamSpring": 600000, "hubTreadBeamDamp": 40, "hubPeripheryBeamSpring": 1000000, "hubPeripheryBeamDamp": 60,
                "wheelSideBeamSpring": 30000, "wheelSideBeamDamp": 120, "wheelSideBeamSpringExpansion": 300000,
                "wheelSideBeamDampExpansion": 120, "wheelSideTransitionZone": 0.05,
                "wheelTreadBeamSpring": 250000, "wheelTreadBeamDamp": 30, "wheelPeripheryBeamSpring": 250000,
                "wheelPeripheryBeamDamp": 30, "enableTireLbeams": true, "wheelLbeamSpring": 6000, "wheelLbeamDamp": 5,
                "enableTreadReinfBeams": true, "wheelTreadReinfBeamSpring": 120000, "wheelTreadReinfBeamDamp": 20
            })];
            for wh in &car.wheels {
                // The parking brake acts on the rear wheels (spec §25.1).
                let rear = wh.name.starts_with('R');
                w.push(json!({"parkingTorque": if rear { PARKING_TORQUE } else { 0.0 }}));
                w.push(json!([
                    wh.name,
                    format!("wheel_{}", wh.name),
                    format!("tire_{}", wh.name),
                    n(wh.n1),
                    n(wh.n2),
                    9999,
                    n(wh.arm),
                    wh.dir
                ]));
            }
            sec.push(format!(
                "    \"pressureWheels\": {}",
                rows(
                    json!([
                        "name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:",
                        "wheelDir"
                    ]),
                    w
                )
            ));
        }
        // Flexbodies.
        let frows: Vec<Value> = flex
            .iter()
            .filter(|(p, f)| match (&f.owner, sub) {
                (Some(o), Some(s)) => o == s,
                (None, None) => *p == part,
                _ => false,
            })
            .map(|(_, f)| {
                let mut o = Map::new();
                if let Some(g) = &f.deform_group {
                    o.insert("deformGroup".into(), json!(g));
                }
                if let Some(k) = &f.skin {
                    o.insert("rubbleSkin".into(), json!(k));
                }
                if let Some(p) = f.pos {
                    o.insert("pos".into(), json!({"x": p[0], "y": p[1], "z": p[2]}));
                }
                json!([f.mesh, f.groups, [], Value::Object(o)])
            })
            .collect();
        if !frows.is_empty() {
            sec.push(format!(
                "    \"flexbodies\": {}",
                rows(json!(["mesh", "[group]:", "nonFlexMaterials"]), frows)
            ));
        }
        let text = format!("{{\n\"{}\": {{\n{}\n}}\n}}\n", pname, sec.join(",\n"));
        out.push((format!("{NAME}/{}.jbeam", pname), text));
    }
    // The configuration.
    let mut parts = Map::new();
    for p in Part::ALL {
        parts.insert(p.name().into(), json!(p.name()));
    }
    for s in &subs {
        parts.insert(sub_name(&s.1), json!(sub_name(&s.1)));
    }
    let pc = json!({"format": 2, "mainPartName": NAME, "model": NAME, "parts": parts, "vars": {}});
    out.push((
        format!("{NAME}/{NAME}.pc"),
        serde_json::to_string_pretty(&pc).unwrap(),
    ));
    out
}
