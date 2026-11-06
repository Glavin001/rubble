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
pub fn jbeam_files(car: &Car, flex: &[(Part, Flex)], deform_groups: &[(String, Vec<usize>, f32)]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    // Main part: slots and the reference nodes.
    let slots: Vec<Value> = Part::ALL.iter().map(|p| json!([p.name(), p.name(), p.name()])).collect();
    let mut main = String::new();
    main += "{\n\"gen2\": {\n";
    main += "    \"information\": {\"authors\": \"Rubble (generated: engine_softbody::authoring::gen2)\", \"name\": \"Rubble gen2 sedan\", \"value\": 0},\n";
    main += "    \"slotType\": \"main\",\n";
    main += &format!("    \"slots\": {},\n", rows(json!(["type", "default", "description"]), slots));
    let n = |i: usize| car.nodes[i].name.clone();
    main += &format!(
        "    \"refNodes\": {}\n",
        rows(json!(["ref:", "back:", "left:", "up:"]), vec![json!([n(car.refs[0]), n(car.refs[1]), n(car.refs[2]), n(car.refs[3])])])
    );
    main += "}\n}\n";
    out.push((format!("{NAME}/{NAME}.jbeam"), main));

    for &part in &Part::ALL {
        let mut sec: Vec<String> = Vec::new();
        sec.push("    \"information\": {\"authors\": \"Rubble (generated)\", \"name\": \"".to_string() + part.name() + "\"}");
        sec.push(format!("    \"slotType\": \"{}\"", part.name()));
        // Nodes.
        let mut nrows = Vec::new();
        for (i, nd) in car.nodes.iter().enumerate() {
            if nd.part != part {
                continue;
            }
            let _ = i;
            let mut o = Map::new();
            o.insert("nodeWeight".into(), num(nd.mass));
            o.insert("selfCollision".into(), json!(nd.self_collision));
            o.insert("collision".into(), json!(true));
            o.insert("frictionCoef".into(), json!(0.7));
            o.insert("group".into(), if nd.no_group { json!("") } else { json!(part.name()) });
            for (k, v) in &nd.opts {
                o.insert(k.clone(), v.clone());
            }
            nrows.push(json!([nd.name, pos(nd.p.x), pos(nd.p.y), pos(nd.p.z), Value::Object(o)]));
        }
        sec.push(format!("    \"nodes\": {}", rows(json!(["id", "posX", "posY", "posZ"]), nrows)));
        // Beams.
        let mut brows = Vec::new();
        let mut list: Vec<usize> = (0..car.beams.len()).filter(|&j| car.beams[j].part == part).collect();
        list.sort_by_key(|&j| (car.beams[j].role, car.beams[j].break_group.clone()));
        for j in list {
            let b = &car.beams[j];
            let sp = b.role.spec(b.part);
            let mut o = Map::new();
            o.insert("beamType".into(), json!(if sp.support { "|SUPPORT" } else { "|NORMAL" }));
            o.insert("beamSpring".into(), num(b.k));
            o.insert("beamDamp".into(), num(b.c));
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
            o.insert("breakGroup".into(), json!(b.break_group.clone().unwrap_or_default()));
            let dg: Vec<&(String, Vec<usize>, f32)> =
                deform_groups.iter().filter(|(_, ns, _)| ns.contains(&b.a) && ns.contains(&b.b)).collect();
            if let Some((g, _, ratio)) = dg.first() {
                o.insert("deformGroup".into(), json!(g));
                o.insert("deformationTriggerRatio".into(), json!(ratio));
            }
            brows.push(json!([n(b.a), n(b.b), Value::Object(o)]));
        }
        sec.push(format!("    \"beams\": {}", rows(json!(["id1:", "id2:"]), brows)));
        // Collision triangles.
        let trows: Vec<Value> = car.tris.iter().filter(|t| t.part == part).map(|t| json!([n(t.n[0]), n(t.n[1]), n(t.n[2])])).collect();
        if !trows.is_empty() {
            sec.push(format!("    \"triangles\": {}", rows(json!(["id1:", "id2:", "id3:"]), trows)));
        }
        // Wheels (raycast on the body's hub nodes).
        if part == Part::Body && !car.wheels.is_empty() {
            let mut w = vec![json!({"radius": 0.33, "tireWidth": 0.215, "hubRadius": 0.2, "hubWidth": 0.2, "numRays": 12, "brakeTorque": 2000})];
            for wh in &car.wheels {
                w.push(json!([wh.name, format!("wheel_{}", wh.name), format!("tire_{}", wh.name), n(wh.n1), n(wh.n2), 9999, n(wh.arm), wh.dir]));
            }
            sec.push(format!(
                "    \"pressureWheels\": {}",
                rows(json!(["name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:", "wheelDir"]), w)
            ));
        }
        // Flexbodies.
        let frows: Vec<Value> = flex
            .iter()
            .filter(|(p, _)| *p == part)
            .map(|(_, f)| {
                let mut o = Map::new();
                if let Some(g) = &f.deform_group {
                    o.insert("deformGroup".into(), json!(g));
                }
                json!([f.mesh, f.groups, [], Value::Object(o)])
            })
            .collect();
        if !frows.is_empty() {
            sec.push(format!("    \"flexbodies\": {}", rows(json!(["mesh", "[group]:", "nonFlexMaterials"]), frows)));
        }
        let text = format!("{{\n\"{}\": {{\n{}\n}}\n}}\n", part.name(), sec.join(",\n"));
        out.push((format!("{NAME}/{}.jbeam", part.name()), text));
    }
    // The configuration.
    let mut parts = Map::new();
    for p in Part::ALL {
        parts.insert(p.name().into(), json!(p.name()));
    }
    let pc = json!({"format": 2, "mainPartName": NAME, "model": NAME, "parts": parts, "vars": {}});
    out.push((format!("{NAME}/sedan.pc"), serde_json::to_string_pretty(&pc).unwrap()));
    out
}
