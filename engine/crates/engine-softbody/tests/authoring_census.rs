//! How real BeamNG community cars are authored, measured beside sedan1, so the
//! generator can take its rules from them (`engine/docs/beamng-authoring-census.md`).
//! A measurement: every test here is `#[ignore]` and opt-in.
//!
//! `RUBBLE_CENSUS_CARS` lists the cars, `;`-separated, each
//! `label=<vehicles dir>,<vehicle>,<config .pc or empty>`, e.g.
//!
//! ```bash
//! RUBBLE_CENSUS_CARS="etkk=$HOME/Downloads/jbeams/etkk_lucasbe/vehicles,etkk,kr6.pc" \
//!   cargo test --release -p engine-softbody --test authoring_census -- --ignored --nocapture
//! ```
//!
//! Without it only sedan1 is measured. The content is read as data (its
//! `.jbeam` rows and its `.dae` geometry); nothing in the repository depends on
//! it or on what these tests print.
//!
//! Per car the rows are put into one model ([`Car`]): nodes (position in
//! Rubble's frame, mass, part, contacter), beams (type, k, c, deform,
//! strength, break groups, deform limits, the part that defines them),
//! collision triangles, and the visual meshes with the nodes they bind to.
//! sedan1's come from its `.truck` as the solver loads it, its skeleton's
//! parts and its visual. One analysis then runs on every car.

use engine_softbody::authoring::skeleton::G_LIMIT;
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, ExplicitType, NODE_CONTACTER, STEP,
};
use glam::{Mat4, Quat, Vec3};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

// --- The model --------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Node {
    pos: Vec3,
    mass: f64,
    part: usize,
    /// Collides with the world and other vehicles (JBeam `collision`).
    contacter: bool,
    self_collision: bool,
    /// Coupler (latch) node: `couplerStrength` (N), `couplerTag`, `tag`.
    coupler_strength: Option<f64>,
    coupler_tag: bool,
}

#[derive(Clone, Debug, Default)]
struct Beam {
    a: usize,
    b: usize,
    /// The part whose rows define it.
    part: usize,
    ty: String,
    k: f64,
    c: f64,
    deform: f64,
    strength: f64,
    l: f64,
    break_groups: Vec<String>,
    deform_limit: bool,
    deform_limit_expansion: bool,
    /// `deformLimitExpansion` when set (fraction of the length).
    dle: Option<f64>,
    deform_limit_stress: bool,
    precompression: bool,
    deform_group: bool,
    /// Breaking does not set off its group (`breakGroupType` 1).
    passive: bool,
}

#[derive(Clone, Debug, Default)]
struct Tri {
    n: [usize; 3],
    part: usize,
    collidable: bool,
}

/// One visual mesh and the node pool it binds to.
#[derive(Clone, Debug, Default)]
struct Flex {
    part: usize,
    mesh: String,
    groups: Vec<String>,
    pool: Vec<usize>,
    pos: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
}

#[derive(Default)]
struct Car {
    label: String,
    parts: Vec<(String, String)>,
    nodes: Vec<Node>,
    beams: Vec<Beam>,
    tris: Vec<Tri>,
    flex: Vec<Flex>,
    /// Torsion bars and LBEAMs.
    torsionbars: usize,
    lbeams: usize,
    /// JBeam `couplerNodes` rows (spawn-coupled latches) and node couplers.
    coupler_pairs: Vec<(usize, usize, f64)>,
    features: BTreeMap<String, usize>,
}

fn num(v: Option<&Value>) -> Option<f64> {
    let big = |x: f64| if x >= 1e30 { f64::INFINITY } else { x };
    match v? {
        Value::Number(n) => n.as_f64().map(big),
        Value::String(s) if s == "FLT_MAX" => Some(f64::INFINITY),
        Value::String(s) => s.parse().ok().map(big),
        _ => None,
    }
}

fn idstr(f: &Map<String, Value>, k: &str) -> Option<String> {
    let v = f.get(k)?;
    v.as_str()
        .map(str::to_string)
        .or_else(|| v.as_f64().map(|x| x.to_string()))
}

fn strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

fn vec3_of(v: Option<&Value>, d: Vec3) -> Vec3 {
    match v {
        Some(Value::Object(m)) => Vec3::new(
            num(m.get("x")).unwrap_or(d.x as f64) as f32,
            num(m.get("y")).unwrap_or(d.y as f64) as f32,
            num(m.get("z")).unwrap_or(d.z as f64) as f32,
        ),
        Some(Value::Array(a)) if a.len() == 3 => Vec3::new(
            num(a.first()).unwrap_or(d.x as f64) as f32,
            num(a.get(1)).unwrap_or(d.y as f64) as f32,
            num(a.get(2)).unwrap_or(d.z as f64) as f32,
        ),
        _ => d,
    }
}

fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(d) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }
}

/// A JBeam car's assembled configuration as a [`Car`].
fn load_jbeam_car(
    label: &str,
    dir: &std::path::Path,
    name: &str,
    pc: Option<&str>,
) -> Result<Car, String> {
    use engine_softbody::jbeam::{
        assembled_parts, assembled_rows, feature_census, load_jbeam, read_vehicle_dir,
    };
    let texts = read_vehicle_dir(dir, name)?;
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pc = match pc {
        Some(p) if !p.is_empty() => {
            Some(std::fs::read_to_string(dir.join(name).join(p)).map_err(|e| e.to_string())?)
        }
        _ => None,
    };
    let pc = pc.as_deref();
    let desc = load_jbeam(&files, pc)?;
    let parts = assembled_parts(&files, pc)?;
    let part_ix: HashMap<String, usize> = parts
        .iter()
        .enumerate()
        .map(|(i, (n, _))| (n.clone(), i))
        .collect();
    let mut car = Car {
        label: label.into(),
        parts: parts.clone(),
        features: feature_census(&files, pc)?,
        ..Default::default()
    };
    let ix = desc.node_index();
    car.nodes = desc
        .nodes
        .iter()
        .map(|n| Node {
            pos: Vec3::from_array(n.position),
            mass: n.mass as f64,
            part: usize::MAX,
            contacter: n.collision,
            self_collision: n.self_collision,
            ..Default::default()
        })
        .collect();
    for (p, f) in assembled_rows(&files, pc, "nodes")? {
        let Some(i) = idstr(&f, "id").and_then(|n| ix.get(&n).copied()) else {
            continue;
        };
        let nd = &mut car.nodes[i as usize];
        nd.part = part_ix[&p];
        nd.coupler_strength = num(f.get("couplerStrength"));
        nd.coupler_tag = f
            .get("couplerTag")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
            || f.get("tag")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty());
    }
    for (p, f) in assembled_rows(&files, pc, "beams")? {
        let (Some(a), Some(b)) = (
            idstr(&f, "id1").and_then(|n| ix.get(&n).copied()),
            idstr(&f, "id2").and_then(|n| ix.get(&n).copied()),
        ) else {
            continue;
        };
        let ty = f
            .get("beamType")
            .and_then(Value::as_str)
            .unwrap_or("|NORMAL")
            .trim_start_matches('|')
            .to_ascii_uppercase();
        let ty = if ty.is_empty() {
            "NORMAL".to_string()
        } else {
            ty
        };
        if ty == "LBEAM" {
            car.lbeams += 1;
            continue;
        }
        if ty == "BROKEN" {
            continue;
        }
        let (pa, pb) = (car.nodes[a as usize].pos, car.nodes[b as usize].pos);
        let set = |k: &str| num(f.get(k)).is_some_and(|x| x.is_finite());
        car.beams.push(Beam {
            a: a as usize,
            b: b as usize,
            part: part_ix[&p],
            ty,
            k: num(f.get("beamSpring")).unwrap_or(4_300_000.0),
            c: num(f.get("beamDamp")).unwrap_or(580.0),
            deform: num(f.get("beamDeform")).unwrap_or(220_000.0),
            strength: num(f.get("beamStrength")).unwrap_or(f64::INFINITY),
            l: pa.distance(pb) as f64,
            break_groups: strings(f.get("breakGroup")),
            deform_limit: set("deformLimit"),
            deform_limit_expansion: set("deformLimitExpansion"),
            dle: num(f.get("deformLimitExpansion")).filter(|x| x.is_finite()),
            deform_limit_stress: set("deformLimitStress"),
            precompression: num(f.get("beamPrecompression"))
                .is_some_and(|x| (x - 1.0).abs() > 1e-6)
                || f.get("precompressionRange").is_some(),
            deform_group: !strings(f.get("deformGroup")).is_empty(),
            passive: num(f.get("breakGroupType")).is_some_and(|t| t as i32 == 1),
        });
    }
    for (p, f) in assembled_rows(&files, pc, "triangles")? {
        let n: Vec<usize> = ["id1", "id2", "id3"]
            .iter()
            .filter_map(|k| {
                idstr(&f, k)
                    .and_then(|n| ix.get(&n).copied())
                    .map(|x| x as usize)
            })
            .collect();
        if n.len() != 3 {
            continue;
        }
        let ty = f
            .get("triangleType")
            .and_then(Value::as_str)
            .unwrap_or("NORMALTYPE");
        car.tris.push(Tri {
            n: [n[0], n[1], n[2]],
            part: part_ix[&p],
            collidable: !ty.contains("NONCOLLIDABLE"),
        });
    }
    for (_, f) in assembled_rows(&files, pc, "torsionbars")? {
        if f.contains_key("id1") {
            car.torsionbars += 1;
        }
    }
    for c in &desc.couplers {
        car.coupler_pairs
            .push((c.a as usize, c.b as usize, c.strength as f64));
    }
    // Visual meshes: every Collada file of the vehicle, the flexbody rows'
    // groups as node pools (transformed as the visual layer does).
    let mut meshes = HashMap::new();
    let mut paths = Vec::new();
    walk(&dir.join(name), &mut paths);
    for p in paths
        .iter()
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("dae")))
    {
        let src =
            String::from_utf8_lossy(&std::fs::read(p).map_err(|e| e.to_string())?).into_owned();
        match engine_softbody::flexmesh::read_collada(&src) {
            Ok(ms) => {
                for m in ms {
                    meshes.entry(m.name.clone()).or_insert_with(|| m.clone());
                    meshes.entry(m.geometry.clone()).or_insert(m);
                }
            }
            Err(e) => eprintln!("({label}) {}: {e}", p.display()),
        }
    }
    let to_r = |p: Vec3| Vec3::from_array(engine_softbody::jbeam::to_rubble(p.to_array()));
    for (p, f) in assembled_rows(&files, pc, "flexbodies")? {
        let Some(mesh) = f.get("mesh").and_then(Value::as_str) else {
            continue;
        };
        let Some(m) = meshes.get(mesh) else { continue };
        let groups = strings(f.get("[group]").or_else(|| f.get("group")));
        let mut pool: Vec<usize> = groups
            .iter()
            .filter_map(|g| {
                desc.node_group_nodes
                    .iter()
                    .find(|(n, _)| n == g)
                    .map(|x| &x.1)
            })
            .flat_map(|v| v.iter().map(|&i| i as usize))
            .collect();
        pool.sort_unstable();
        pool.dedup();
        if pool.len() < 3 {
            continue;
        }
        let t = vec3_of(f.get("pos"), Vec3::ZERO);
        let r = vec3_of(f.get("rot"), Vec3::ZERO);
        let s = vec3_of(f.get("scale"), Vec3::ONE);
        let xf = Mat4::from_scale_rotation_translation(
            s,
            Quat::from_euler(
                glam::EulerRot::XYZ,
                r.x.to_radians(),
                r.y.to_radians(),
                r.z.to_radians(),
            ),
            t,
        );
        car.flex.push(Flex {
            part: part_ix[&p],
            mesh: mesh.into(),
            groups,
            pool,
            pos: m
                .pos
                .iter()
                .map(|&q| to_r(xf.transform_point3(q)))
                .collect(),
            tris: m.tris.clone(),
        });
    }
    Ok(car)
}

/// sedan1 (`RES`, default Medium) as a [`Car`]: its `.truck` as the solver
/// loads it, nodes by skeleton part, hardware groups as named break groups,
/// and its visual's pieces with the nodes each binds to.
fn load_sedan1() -> Car {
    let res = std::env::var("RES")
        .ok()
        .and_then(|r| Resolution::parse(&r))
        .unwrap_or(Resolution::Medium);
    let b = build(
        &style("sedan1").unwrap(),
        &BuildParams {
            resolution: res,
            ..Default::default()
        },
    )
    .unwrap();
    let sk = &b.skeleton;
    let s = ExplicitBeams::from_truck(
        &b.truck,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )
    .unwrap();
    let mut car = Car {
        label: "sedan1".into(),
        parts: sk.parts.iter().map(|p| (p.clone(), p.clone())).collect(),
        ..Default::default()
    };
    let n = sk.nodes.len();
    car.nodes = (0..s.nodes.len())
        .map(|i| Node {
            pos: s.nodes[i].pos,
            mass: s.nodes[i].mass as f64,
            part: if i < n {
                sk.nodes[i].part as usize
            } else {
                usize::MAX
            },
            contacter: s.nodes[i].has(NODE_CONTACTER),
            ..Default::default()
        })
        .collect();
    // Hardware: the solver's detacher groups by their names.
    let mut group_name: HashMap<i32, String> = HashMap::new();
    for h in &b.hardware {
        group_name.insert(h.group as i32, format!("{} {:?}", h.name, h.kind));
    }
    for bm in &s.beams {
        let (a, c) = (bm.a as usize, bm.b as usize);
        let ty = match (bm.kind, bm.ty) {
            (ExplicitKind::Support, _) => "SUPPORT",
            (ExplicitKind::Rope, _) => "ROPE",
            (ExplicitKind::LimitedSpring, _) | (ExplicitKind::Shock2, _) => "BOUNDED",
            (_, ExplicitType::Deformable) if bm.k <= 0.0 => "DASHPOT",
            (_, ExplicitType::Deformable) => "NORMAL",
            _ if bm.k <= 0.0 => "DASHPOT",
            _ => "NORMAL",
        };
        let part = if a < n {
            sk.nodes[a].part as usize
        } else {
            usize::MAX
        };
        car.beams.push(Beam {
            a,
            b: c,
            part,
            ty: ty.into(),
            k: bm.k as f64,
            c: bm.c as f64,
            deform: if bm.ty == ExplicitType::Deformable && bm.tc < 1e30 {
                bm.tc as f64
            } else {
                f64::INFINITY
            },
            strength: if bm.strength < 1e30 {
                bm.strength as f64
            } else {
                f64::INFINITY
            },
            l: bm.rest as f64,
            break_groups: group_name.get(&bm.group).cloned().into_iter().collect(),
            ..Default::default()
        });
    }
    for t in &s.triangles {
        let [x, y, z] = t.nodes;
        car.tris.push(Tri {
            n: [x as usize, y as usize, z as usize],
            part: sk
                .nodes
                .get(x as usize)
                .map_or(usize::MAX, |n| n.part as usize),
            collidable: true,
        });
    }
    if let Ok(v) = engine_softbody::authoring::skin::showcase_visual(&b) {
        for (name, first, count) in &v.pieces {
            let r = *first as usize..(*first + *count) as usize;
            let mut pool: Vec<usize> = v.bind_nodes[r.clone()]
                .iter()
                .flat_map(|t| t.iter().map(|&i| i as usize))
                .collect();
            pool.sort_unstable();
            pool.dedup();
            if pool.is_empty() {
                continue;
            }
            // The piece's part: the most common part of its nodes.
            let mut cnt: HashMap<usize, usize> = HashMap::new();
            for &i in &pool {
                *cnt.entry(car.nodes[i].part).or_default() += 1;
            }
            let part = cnt
                .into_iter()
                .max_by_key(|x| (x.1, std::cmp::Reverse(x.0)))
                .map_or(usize::MAX, |x| x.0);
            let local: HashMap<u32, u32> = r
                .clone()
                .enumerate()
                .map(|(k, g)| (g as u32, k as u32))
                .collect();
            let tris: Vec<[u32; 3]> = v
                .indices
                .chunks(3)
                .filter_map(|t| Some([*local.get(&t[0])?, *local.get(&t[1])?, *local.get(&t[2])?]))
                .collect();
            car.flex.push(Flex {
                part,
                mesh: name.clone(),
                groups: vec![sk.parts.get(part).cloned().unwrap_or_default()],
                pool,
                pos: v.positions[r]
                    .iter()
                    .map(|p| Vec3::from_array(*p))
                    .collect(),
                tris,
            });
        }
    }
    car
}

/// The cars `RUBBLE_CENSUS_CARS` names (absent ones reported and skipped).
fn jbeam_cars() -> Vec<Car> {
    let Ok(list) = std::env::var("RUBBLE_CENSUS_CARS") else {
        eprintln!("skip: RUBBLE_CENSUS_CARS not set (only sedan1 is measured)");
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in list.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let Some((label, rest)) = item.split_once('=') else {
            continue;
        };
        let f: Vec<&str> = rest.split(',').collect();
        let (dir, name, pc) = (
            f.first().copied().unwrap_or(""),
            f.get(1).copied().unwrap_or(""),
            f.get(2).copied(),
        );
        let dir = std::path::Path::new(dir);
        if !dir.join(name).is_dir() {
            eprintln!("skip {label}: {} absent", dir.join(name).display());
            continue;
        }
        match load_jbeam_car(label, dir, name, pc) {
            Ok(c) => out.push(c),
            Err(e) => eprintln!("skip {label}: {e}"),
        }
    }
    out
}

fn all_cars() -> Vec<Car> {
    let mut v = jbeam_cars();
    if std::env::var("CENSUS_NO_SEDAN1").is_err() {
        v.push(load_sedan1());
    }
    v
}

/// The parts of each car: name, slot, nodes, mass, beams, triangles,
/// flexbodies. For mapping parts to categories.
#[test]
#[ignore]
fn census_parts() {
    for car in all_cars() {
        let mut nodes = vec![0usize; car.parts.len() + 1];
        let mut mass = vec![0.0f64; car.parts.len() + 1];
        let mut beams = vec![0usize; car.parts.len() + 1];
        let mut tris = vec![0usize; car.parts.len() + 1];
        let mut flex = vec![0usize; car.parts.len() + 1];
        let k = |p: usize| p.min(car.parts.len());
        for nd in &car.nodes {
            nodes[k(nd.part)] += 1;
            mass[k(nd.part)] += nd.mass;
        }
        for b in &car.beams {
            beams[k(b.part)] += 1;
        }
        for t in &car.tris {
            tris[k(t.part)] += 1;
        }
        for f in &car.flex {
            flex[k(f.part)] += 1;
        }
        let total: f64 = car.nodes.iter().map(|n| n.mass).sum();
        eprintln!(
            "== {}: {} nodes, {:.1} kg, {} beams, {} triangles, {} meshes",
            car.label,
            car.nodes.len(),
            total,
            car.beams.len(),
            car.tris.len(),
            car.flex.len()
        );
        for (i, (p, slot)) in car.parts.iter().enumerate() {
            if nodes[i] + beams[i] + tris[i] + flex[i] == 0 {
                continue;
            }
            eprintln!(
                "  {:<40} {:<32} -> {:<10} nodes {:4} mass {:7.1} beams {:5} tris {:4} flex {:3}",
                p,
                slot,
                category(p, slot),
                nodes[i],
                mass[i],
                beams[i],
                tris[i],
                flex[i]
            );
        }
    }
}

/// A part's category from its name and slot.
fn category(part: &str, slot: &str) -> &'static str {
    let p = part.to_ascii_lowercase();
    let s = slot.to_ascii_lowercase();
    let has = |k: &str| p.contains(k) || s.contains(k);
    if has("light") || has("signal") || has("lamp") {
        "lamp"
    } else if has("glass") || has("windshield") || has("windscreen") || has("window") {
        "glass"
    } else if has("door") {
        "door"
    } else if has("hood") || has("bonnet") {
        "bonnet"
    } else if has("trunk") || has("tailgate") || has("boot") || has("liftgate") || has("decklid") {
        "boot"
    } else if has("fender") || (has("wing") && !has("wingmirror") && !has("wingflare")) {
        "fender"
    } else if has("bumper")
        && (p.ends_with("_f")
            || p.contains("bumper_f")
            || p.contains("_f_")
            || s.contains("bumper_f"))
    {
        "bumper_f"
    } else if has("bumper") {
        "bumper_r"
    } else if has("subframe")
        || has("crossmember")
        || has("radsupport")
        || has("frame")
        || has("rail")
    {
        "rails"
    } else if s.ends_with("_body")
        || s == "body"
        || p == "body"
        || p.ends_with("_body")
        || s.contains("_body_")
        || p.contains("body_")
    {
        "body"
    } else if has("engine")
        || has("transmission")
        || has("gearbox")
        || has("powertrain")
        || has("radiator")
        || has("battery")
        || has("fueltank")
        || has("exhaust")
        || has("differential")
        || has("driveshaft")
    {
        "powertrain"
    } else if has("suspension")
        || has("strut")
        || has("coilover")
        || has("spring")
        || has("shock")
        || has("swaybar")
        || has("hub")
        || has("steer")
        || has("brake")
        || has("wheel")
        || has("tire")
        || has("axle")
        || has("arm")
    {
        "chassis"
    } else if has("seat") || has("interior") || has("dash") {
        "interior"
    } else {
        "other"
    }
}

// --- The analysis -------------------------------------------------------------

/// The categories reported, in order.
const CATS: [&str; 9] = [
    "body", "door", "bonnet", "boot", "fender", "bumper_f", "bumper_r", "rails", "glass",
];

/// A part's instance within its category: its side (`fl`, `r`, ...) or its last
/// name token, so a door and its door panel are one door.
fn instance(part: &str) -> String {
    let p = part.to_ascii_lowercase();
    let t: Vec<&str> = p.split('_').collect();
    let n = t.len();
    let fr = |x: &str| x == "f" || x == "r";
    let lr = |x: &str| x == "l" || x == "r";
    if n >= 2 && fr(t[n - 2]) && lr(t[n - 1]) && !t[n - 2].is_empty() {
        return format!("{}{}", t[n - 2], t[n - 1]);
    }
    for x in t.iter().rev() {
        if ["fl", "fr", "rl", "rr", "l", "r"].contains(x) {
            return x.to_string();
        }
    }
    t.last().copied().unwrap_or("").to_string()
}

fn q(v: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// A spring beam: one the step's bound sees with stiffness.
fn spring(b: &Beam) -> bool {
    b.k > 0.0 && b.ty != "DASHPOT"
}

/// Yielding beams: NORMAL, SUPPORT, ANISOTROPIC (their yield and break).
fn structural(b: &Beam) -> bool {
    matches!(
        b.ty.as_str(),
        "NORMAL" | "SUPPORT" | "ANISOTROPIC" | "PRESSURED"
    ) && b.k > 0.0
        && b.l > 1e-3
}

struct Ctx<'a> {
    car: &'a Car,
    cat: Vec<&'static str>,
    /// Per node: its category and instance.
    node_cat: Vec<&'static str>,
    node_inst: Vec<String>,
    /// Per node: the Gershgorin row over the step's bound, spring ends.
    row: Vec<f64>,
    ends: Vec<usize>,
}

impl<'a> Ctx<'a> {
    fn new(car: &'a Car) -> Self {
        let cat: Vec<&'static str> = car.parts.iter().map(|(p, s)| category(p, s)).collect();
        let node_cat = car
            .nodes
            .iter()
            .map(|n| cat.get(n.part).copied().unwrap_or("none"))
            .collect();
        let node_inst = car
            .nodes
            .iter()
            .map(|n| {
                car.parts
                    .get(n.part)
                    .map_or(String::new(), |(p, _)| instance(p))
            })
            .collect();
        let h = STEP as f64;
        let limit = G_LIMIT as f64 * h * h;
        let mut row = vec![0.0; car.nodes.len()];
        let mut ends = vec![0usize; car.nodes.len()];
        for b in &car.beams {
            let (ma, mb) = (car.nodes[b.a].mass.max(1e-6), car.nodes[b.b].mass.max(1e-6));
            let qq = h * h * b.k.max(0.0) + 2.0 * h * b.c.max(0.0);
            let w = 1.0 / (ma * mb).sqrt();
            row[b.a] += qq * (1.0 / ma + w) / limit;
            row[b.b] += qq * (1.0 / mb + w) / limit;
            if spring(b) {
                ends[b.a] += 1;
                ends[b.b] += 1;
            }
        }
        Self {
            car,
            cat,
            node_cat,
            node_inst,
            row,
            ends,
        }
    }
    fn beam_cat(&self, b: &Beam) -> &'static str {
        self.cat.get(b.part).copied().unwrap_or("none")
    }
}

fn zeta(car: &Car, b: &Beam) -> f64 {
    let (ma, mb) = (car.nodes[b.a].mass, car.nodes[b.b].mass);
    let mu = ma * mb / (ma + mb);
    b.c / (2.0 * (b.k * mu).sqrt())
}

/// 1. Per part category: nodes, spacing, mass, spring ends, beams.
fn report_parts(x: &Ctx) {
    let car = x.car;
    for cat in CATS {
        let nodes: Vec<usize> = (0..car.nodes.len())
            .filter(|&i| x.node_cat[i] == cat)
            .collect();
        if nodes.is_empty() {
            continue;
        }
        let insts: HashSet<&str> = nodes.iter().map(|&i| x.node_inst[i].as_str()).collect();
        let ni = insts.len().max(1) as f64;
        let mass: Vec<f64> = nodes.iter().map(|&i| car.nodes[i].mass).collect();
        let total: f64 = mass.iter().sum();
        // Spacing: nearest other node of the same instance.
        let spacing: Vec<f64> = nodes
            .iter()
            .map(|&i| {
                nodes
                    .iter()
                    .filter(|&&j| j != i && x.node_inst[j] == x.node_inst[i])
                    .map(|&j| car.nodes[i].pos.distance(car.nodes[j].pos) as f64)
                    .filter(|d| *d > 1e-3)
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let ends: Vec<f64> = nodes.iter().map(|&i| x.ends[i] as f64).collect();
        let at_bound =
            nodes.iter().filter(|&&i| x.row[i] > 0.95).count() as f64 / nodes.len() as f64;
        let rows: Vec<f64> = nodes.iter().map(|&i| x.row[i]).collect();
        let own: Vec<&Beam> = car
            .beams
            .iter()
            .filter(|b| x.beam_cat(b) == cat && structural(b))
            .collect();
        let l: Vec<f64> = own.iter().map(|b| b.l).collect();
        let k: Vec<f64> = own.iter().map(|b| b.k).collect();
        let kl: Vec<f64> = own.iter().map(|b| b.k * b.l).collect();
        let d: Vec<f64> = own.iter().map(|b| b.deform).collect();
        let ey: Vec<f64> = own.iter().map(|b| b.deform / (b.k * b.l)).collect();
        let st: Vec<f64> = own.iter().map(|b| b.strength).collect();
        let inf = own
            .iter()
            .filter(|b| !b.strength.is_finite() || b.strength >= 1e30)
            .count() as f64
            / own.len().max(1) as f64;
        let eb: Vec<f64> = own.iter().map(|b| b.strength / (b.k * b.l)).collect();
        let sd: Vec<f64> = own.iter().map(|b| b.strength / b.deform).collect();
        let c: Vec<f64> = own.iter().map(|b| b.c).collect();
        let z: Vec<f64> = own.iter().map(|b| zeta(car, b)).collect();
        let e50 = q(&ey, 0.5);
        let dle: Vec<f64> = own.iter().filter_map(|b| b.dle).collect();
        eprintln!(
            "[part] {:<8} {:<9} inst {:2} nodes/inst {:5.1} kg/inst {:6.1} | m p10/50/90 {:.2}/{:.2}/{:.2} kg | spacing p50 {:.3} m | ends p50 {:.0} | bound {:3.0} % row p50/p90/max {:.2}/{:.2}/{:.2} | deformLimitExpansion on {:.0} % p50 {:.3}",
            car.label, cat, ni, nodes.len() as f64 / ni, total / ni, q(&mass, 0.1), q(&mass, 0.5), q(&mass, 0.9), q(&spacing, 0.5), q(&ends, 0.5), at_bound * 100.0, q(&rows, 0.5), q(&rows, 0.9), q(&rows, 1.0),
            dle.len() as f64 / own.len().max(1) as f64 * 100.0, q(&dle, 0.5)
        );
        eprintln!(
            "[beam] {:<8} {:<9} beams {:5} | L p50 {:.3} | k p50 {:.3e} | k·L p50 {:.2e} | deform p50 {:.3e} | ε_y p10/50/90 {:.2}/{:.2}/{:.2} % sb20 {:.0} % | strength p50 {:.3e} inf {:.0} % | ε_b p50 {:.2} % | str/def p50 {:.2} | c p50 {:.0} ζ p50 {:.3}",
            car.label, cat, own.len(), q(&l, 0.5), q(&k, 0.5), q(&kl, 0.5), q(&d, 0.5), q(&ey, 0.1) * 100.0, e50 * 100.0, q(&ey, 0.9) * 100.0, e50 / (e50 + 0.2) * 100.0, q(&st, 0.5), inf * 100.0, q(&eb, 0.5) * 100.0, q(&sd, 0.5), q(&c, 0.5), q(&z, 0.5)
        );
    }
}

/// 2. Beam types and options: car-wide and per category.
fn report_types(x: &Ctx) {
    let car = x.car;
    let mut by: BTreeMap<(&str, String), usize> = BTreeMap::new();
    let mut tot: BTreeMap<&str, usize> = BTreeMap::new();
    let mut opt: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for b in &car.beams {
        let c = x.beam_cat(b);
        let c = if CATS.contains(&c) { c } else { "rest" };
        for cc in [c, "all"] {
            *by.entry((cc, b.ty.clone())).or_default() += 1;
            *tot.entry(cc).or_default() += 1;
            for (f, name) in [
                (b.deform_limit, "deformLimit"),
                (b.deform_limit_expansion, "deformLimitExpansion"),
                (b.deform_limit_stress, "deformLimitStress"),
                (b.precompression, "precompression"),
                (b.deform_group, "deformGroup"),
                (!b.break_groups.is_empty(), "breakGroup"),
                (b.passive, "breakGroupType1"),
                (
                    !b.strength.is_finite() || b.strength >= 1e30,
                    "strength FLT_MAX",
                ),
            ] {
                if f {
                    *opt.entry((cc, name)).or_default() += 1;
                }
            }
        }
    }
    for (cat, n) in &tot {
        let types: Vec<String> = by
            .iter()
            .filter(|((c, _), _)| c == cat)
            .map(|((_, t), k)| format!("{t} {:.1} %", *k as f64 / *n as f64 * 100.0))
            .collect();
        let opts: Vec<String> = opt
            .iter()
            .filter(|((c, _), _)| c == cat)
            .map(|((_, o), k)| format!("{o} {:.1} %", *k as f64 / *n as f64 * 100.0))
            .collect();
        eprintln!(
            "[types] {:<8} {:<9} {n:5} beams: {} | {}",
            car.label,
            cat,
            types.join(", "),
            opts.join(", ")
        );
    }
    let node_couplers = car
        .nodes
        .iter()
        .filter(|n| n.coupler_strength.is_some())
        .count();
    let tags = car.nodes.iter().filter(|n| n.coupler_tag).count();
    eprintln!(
        "[types] {:<8} LBEAM {} torsionbars {} couplerNodes pairs {} (strength p50 {:.0} N) node couplerStrength {} (p50 {:.0} N) couplerTag/tag nodes {}",
        car.label,
        car.lbeams,
        car.torsionbars,
        car.coupler_pairs.len(),
        q(&car.coupler_pairs.iter().map(|c| c.2).collect::<Vec<_>>(), 0.5),
        node_couplers,
        q(&car.nodes.iter().filter_map(|n| n.coupler_strength).collect::<Vec<_>>(), 0.5),
        tags
    );
    let sections: Vec<String> = car
        .features
        .iter()
        .filter(|(k, _)| k.starts_with("section "))
        .map(|(k, n)| format!("{}:{n}", &k[8..]))
        .collect();
    if !sections.is_empty() {
        eprintln!("[types] {:<8} sections {}", car.label, sections.join(" "));
    }
}

/// 3. How each closure is held: the beams from it to the rest, by break group.
fn report_holding(x: &Ctx) {
    let car = x.car;
    let mut group_size: HashMap<&str, usize> = HashMap::new();
    for b in &car.beams {
        for g in &b.break_groups {
            *group_size.entry(g.as_str()).or_default() += 1;
        }
    }
    for cat in ["door", "bonnet", "boot"] {
        let mut insts: Vec<&str> = (0..car.nodes.len())
            .filter(|&i| x.node_cat[i] == cat)
            .map(|i| x.node_inst[i].as_str())
            .collect();
        insts.sort_unstable();
        insts.dedup();
        for inst in insts {
            let mine = |i: usize| x.node_cat[i] == cat && x.node_inst[i] == inst;
            let n = (0..car.nodes.len()).filter(|&i| mine(i)).count();
            let mut by: BTreeMap<String, Vec<&Beam>> = BTreeMap::new();
            for b in &car.beams {
                if mine(b.a) == mine(b.b) {
                    continue;
                }
                // (Its own glass is not what holds it.)
                let other = if mine(b.a) { b.b } else { b.a };
                if x.node_cat[other] == "glass" || x.node_cat[other] == "lamp" {
                    continue;
                }
                let g = b
                    .break_groups
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "(no group)".into());
                by.entry(g).or_default().push(b);
            }
            let couplers: Vec<f64> = car
                .coupler_pairs
                .iter()
                .filter(|c| mine(c.0) != mine(c.1))
                .map(|c| c.2)
                .collect();
            let node_c: Vec<f64> = (0..car.nodes.len())
                .filter(|&i| mine(i))
                .filter_map(|i| car.nodes[i].coupler_strength)
                .collect();
            eprintln!(
                "[hold] {:<8} {:<7} {:<6} nodes {:3} | couplerNodes pairs {} strength {:?} | own nodes with couplerStrength {} {:?}",
                car.label, cat, inst, n, couplers.len(), couplers.iter().map(|s| *s as i64).collect::<Vec<_>>(), node_c.len(), node_c.iter().map(|s| *s as i64).collect::<Vec<_>>()
            );
            for (g, bs) in &by {
                let mut types: BTreeMap<&str, usize> = BTreeMap::new();
                for b in bs {
                    *types.entry(b.ty.as_str()).or_default() += 1;
                }
                let st: Vec<f64> = bs.iter().map(|b| b.strength).collect();
                let fin: f64 = st.iter().filter(|s| s.is_finite() && **s < 1e30).sum();
                eprintln!(
                    "[hold]      group {:<34} beams {:3} (group size {:3}) {:?} | k p50 {:.2e} | deform p50 {:.2e} | strength p50 {:.2e} sum(finite) {:.2e} | L p50 {:.3}",
                    g,
                    bs.len(),
                    group_size.get(g.as_str()).copied().unwrap_or(0),
                    types,
                    q(&bs.iter().map(|b| b.k).collect::<Vec<_>>(), 0.5),
                    q(&bs.iter().map(|b| b.deform).collect::<Vec<_>>(), 0.5),
                    q(&st, 0.5),
                    fin,
                    q(&bs.iter().map(|b| b.l).collect::<Vec<_>>(), 0.5)
                );
            }
        }
    }
}

fn tri_area(car: &Car, t: &Tri) -> f64 {
    let p = |k: usize| car.nodes[t.n[k]].pos;
    0.5 * (p(1) - p(0)).cross(p(2) - p(0)).length() as f64
}

/// 4. Collision triangles and contacters per category.
fn report_tris(x: &Ctx) {
    let car = x.car;
    let mut on_tri = vec![false; car.nodes.len()];
    for t in car.tris.iter().filter(|t| t.collidable) {
        for &i in &t.n {
            on_tri[i] = true;
        }
    }
    let contacters = car.nodes.iter().filter(|n| n.contacter).count();
    eprintln!(
        "[tris] {:<8} all: {} triangles ({} collidable), {:.2} m², nodes on a triangle {:.0} %, contacters {:.0} % of nodes, self-collision {:.0} %",
        car.label,
        car.tris.len(),
        car.tris.iter().filter(|t| t.collidable).count(),
        car.tris.iter().map(|t| tri_area(car, t)).sum::<f64>(),
        on_tri.iter().filter(|&&b| b).count() as f64 / car.nodes.len() as f64 * 100.0,
        contacters as f64 / car.nodes.len() as f64 * 100.0,
        car.nodes.iter().filter(|n| n.self_collision).count() as f64 / car.nodes.len() as f64 * 100.0
    );
    for cat in CATS
        .iter()
        .chain(["powertrain", "chassis", "interior"].iter())
    {
        let nodes: Vec<usize> = (0..car.nodes.len())
            .filter(|&i| x.node_cat[i] == *cat)
            .collect();
        if nodes.is_empty() {
            continue;
        }
        // A triangle is the category's when its first node is.
        let ts: Vec<&Tri> = car
            .tris
            .iter()
            .filter(|t| x.node_cat[t.n[0]] == *cat)
            .collect();
        let area: Vec<f64> = ts.iter().map(|t| tri_area(car, t)).collect();
        let edge: Vec<f64> = ts
            .iter()
            .flat_map(|t| (0..3).map(move |k| (t.n[k], t.n[(k + 1) % 3])))
            .map(|(a, b)| car.nodes[a].pos.distance(car.nodes[b].pos) as f64)
            .collect();
        let on = nodes.iter().filter(|&&i| on_tri[i]).count() as f64 / nodes.len() as f64;
        let cont =
            nodes.iter().filter(|&&i| car.nodes[i].contacter).count() as f64 / nodes.len() as f64;
        let cont_off = nodes
            .iter()
            .filter(|&&i| car.nodes[i].contacter && !on_tri[i])
            .count() as f64
            / nodes.len() as f64;
        eprintln!(
            "[tris] {:<8} {:<10} tris {:4} area {:.2} m² | tri area p50 {:.4} m² edge p50 {:.3} m | nodes on a tri {:3.0} % | contacters {:3.0} % (of which off every tri {:3.0} %)",
            car.label, cat, ts.len(), area.iter().sum::<f64>(), q(&area, 0.5), q(&edge, 0.5), on * 100.0, cont * 100.0, cont_off * 100.0
        );
    }
}

/// A uniform grid over points, for nearest-within-radius queries.
struct Grid {
    cell: f32,
    map: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl Grid {
    fn new(pts: &[Vec3], cell: f32) -> Self {
        let mut map: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, p) in pts.iter().enumerate() {
            map.entry(Self::key(*p, cell)).or_default().push(i as u32);
        }
        Self { cell, map }
    }
    fn key(p: Vec3, cell: f32) -> (i32, i32, i32) {
        (
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        )
    }
    /// Distance to the nearest point within one cell (∞ past it).
    fn near(&self, pts: &[Vec3], p: Vec3) -> f32 {
        let k = Self::key(p, self.cell);
        let mut best = f32::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.map.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        for &i in v {
                            best = best.min(pts[i as usize].distance(p));
                        }
                    }
                }
            }
        }
        if best <= self.cell {
            best
        } else {
            f32::INFINITY
        }
    }
}

/// A mesh welded at 1 mm: its positions, triangles, and per vertex whether it
/// is on an open edge (a panel's edge or shut line: an edge one triangle
/// uses) and on a crease (two triangles meeting sharper than 30°).
fn weld(pos: &[Vec3], tris: &[[u32; 3]]) -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<bool>, Vec<bool>) {
    let mut at: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let mut out: Vec<Vec3> = Vec::new();
    let map: Vec<u32> = pos
        .iter()
        .map(|p| {
            let k = (
                (p.x * 1000.0).round() as i32,
                (p.y * 1000.0).round() as i32,
                (p.z * 1000.0).round() as i32,
            );
            *at.entry(k).or_insert_with(|| {
                out.push(*p);
                out.len() as u32 - 1
            })
        })
        .collect();
    let t: Vec<[u32; 3]> = tris
        .iter()
        .map(|t| [map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]])
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .collect();
    let mut edges: HashMap<(u32, u32), Vec<Vec3>> = HashMap::new();
    for tr in &t {
        let n = (out[tr[1] as usize] - out[tr[0] as usize])
            .cross(out[tr[2] as usize] - out[tr[0] as usize])
            .normalize_or_zero();
        for k in 0..3 {
            let (a, b) = (tr[k], tr[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(n);
        }
    }
    let mut open = vec![false; out.len()];
    let mut crease = vec![false; out.len()];
    let cos30 = 30f32.to_radians().cos();
    for ((a, b), ns) in &edges {
        if ns.len() == 1 {
            open[*a as usize] = true;
            open[*b as usize] = true;
        } else if ns.len() == 2 && ns[0].dot(ns[1]).abs() < cos30 {
            crease[*a as usize] = true;
            crease[*b as usize] = true;
        }
    }
    (out, t, open, crease)
}

fn nearest(pts: &[Vec3], p: Vec3) -> f64 {
    pts.iter()
        .map(|q| q.distance(p))
        .fold(f32::INFINITY, f32::min) as f64
}

/// 5. The visual meshes against their nodes. Per mesh (≥ 100 vertices, the
/// panels, not their bolts): each vertex's distance to the nearest node of
/// the groups it binds to; the panel's nodes (pool nodes within 10 cm of the
/// mesh) and their distance to the surface, to the nearest open edge (panel
/// edge, shut line) and to the nearest crease, against the same distances for
/// points spread evenly over the surface (what a node placed at random on the
/// panel would give).
fn report_flex(x: &Ctx) {
    let car = x.car;
    #[derive(Default)]
    struct Acc {
        meshes: usize,
        dist: Vec<f64>,
        area: f64,
        panel_nodes: usize,
        pool_nodes: usize,
        to_surf: Vec<f64>,
        to_open: Vec<f64>,
        to_crease: Vec<f64>,
        base_open: Vec<f64>,
        base_crease: Vec<f64>,
        groups: Vec<usize>,
        density: Vec<f64>,
        examples: Vec<String>,
    }
    let mut by: BTreeMap<&str, Acc> = BTreeMap::new();
    for f in &car.flex {
        let cat = x.cat.get(f.part).copied().unwrap_or("none");
        if !CATS.contains(&cat) || cat == "glass" || f.pos.len() < 100 {
            continue;
        }
        let (pos, tris, open, crease) = weld(&f.pos, &f.tris);
        if tris.is_empty() {
            continue;
        }
        let pool: Vec<Vec3> = f.pool.iter().map(|&i| car.nodes[i].pos).collect();
        let a = by.entry(cat).or_default();
        a.meshes += 1;
        a.groups.push(f.groups.len());
        for p in &pos {
            a.dist.push(nearest(&pool, *p));
        }
        let opts: Vec<Vec3> = pos
            .iter()
            .zip(&open)
            .filter(|(_, f)| **f)
            .map(|(p, _)| *p)
            .collect();
        let cpts: Vec<Vec3> = pos
            .iter()
            .zip(&crease)
            .filter(|(_, f)| **f)
            .map(|(p, _)| *p)
            .collect();
        let g = Grid::new(&pos, 0.10);
        let panel: Vec<Vec3> = pool
            .iter()
            .copied()
            .filter(|p| g.near(&pos, *p).is_finite())
            .collect();
        for p in &panel {
            a.to_surf.push(g.near(&pos, *p) as f64);
            a.to_open.push(nearest(&opts, *p));
            a.to_crease.push(nearest(&cpts, *p));
        }
        a.panel_nodes += panel.len();
        a.pool_nodes += pool.len();
        // Area and an even sample of the surface (triangle centroids, every
        // k-th by area so about 400 samples per mesh).
        let areas: Vec<f64> = tris
            .iter()
            .map(|t| {
                0.5 * (pos[t[1] as usize] - pos[t[0] as usize])
                    .cross(pos[t[2] as usize] - pos[t[0] as usize])
                    .length() as f64
            })
            .collect();
        let area: f64 = areas.iter().sum();
        a.area += area;
        let step = area / 400.0;
        let mut acc = 0.0;
        for (t, ar) in tris.iter().zip(&areas) {
            acc += ar;
            while acc >= step && step > 0.0 {
                acc -= step;
                let c = (pos[t[0] as usize] + pos[t[1] as usize] + pos[t[2] as usize]) / 3.0;
                a.base_open.push(nearest(&opts, c));
                a.base_crease.push(nearest(&cpts, c));
            }
        }
        if area > 0.05 {
            a.density.push(panel.len() as f64 / area);
        }
        if a.examples.len() < 4 {
            a.examples.push(format!(
                "{} ← {:?} ({} nodes, {} within 10 cm, {:.2} m²)",
                f.mesh,
                f.groups,
                f.pool.len(),
                panel.len(),
                area
            ));
        }
    }
    for (cat, a) in &by {
        eprintln!(
            "[flex] {:<8} {:<9} meshes {:3} area {:5.2} m² | vertex→node p50/p90 {:.3}/{:.3} m | panel nodes {:3.0} % of pool, {:.1} per m² | node→surface p50 {:.3} | node→open edge p50 {:.3} (even {:.3}) | node→crease p50 {:.3} (even {:.3}) | groups/mesh {:.0}",
            x.car.label,
            cat,
            a.meshes,
            a.area,
            q(&a.dist, 0.5),
            q(&a.dist, 0.9),
            a.panel_nodes as f64 / a.pool_nodes.max(1) as f64 * 100.0,
            q(&a.density, 0.5),
            q(&a.to_surf, 0.5),
            q(&a.to_open, 0.5),
            q(&a.base_open, 0.5),
            q(&a.to_crease, 0.5),
            q(&a.base_crease, 0.5),
            q(&a.groups.iter().map(|&g| g as f64).collect::<Vec<_>>(), 0.5),
        );
        for e in &a.examples {
            eprintln!("[flex]      {e}");
        }
    }
}

/// 6. Where the mass is.
fn report_mass(x: &Ctx) {
    let car = x.car;
    let total: f64 = car.nodes.iter().map(|n| n.mass).sum();
    let mut by: BTreeMap<&str, f64> = BTreeMap::new();
    for (i, n) in car.nodes.iter().enumerate() {
        *by.entry(x.node_cat[i]).or_default() += n.mass;
    }
    let heavy: f64 = car
        .nodes
        .iter()
        .filter(|n| n.mass >= 10.0)
        .map(|n| n.mass)
        .sum();
    let heavy_n = car.nodes.iter().filter(|n| n.mass >= 10.0).count();
    let shell_cats = [
        "body", "door", "bonnet", "boot", "fender", "bumper_f", "bumper_r", "rails", "glass",
        "lamp", "other",
    ];
    let shell: f64 = by
        .iter()
        .filter(|(c, _)| shell_cats.contains(c))
        .map(|(_, m)| *m)
        .sum();
    let all: Vec<f64> = car.nodes.iter().map(|n| n.mass).collect();
    eprintln!(
        "[mass] {:<8} total {:.0} kg, {} nodes, node mass p10/50/90 {:.2}/{:.2}/{:.2} kg | nodes ≥ 10 kg: {} with {:.0} kg ({:.0} %) | body and panels {:.0} kg ({:.0} %), the rest (powertrain, chassis, interior) {:.0} kg",
        car.label,
        total,
        car.nodes.len(),
        q(&all, 0.1),
        q(&all, 0.5),
        q(&all, 0.9),
        heavy_n,
        heavy,
        heavy / total * 100.0,
        shell,
        shell / total * 100.0,
        total - shell
    );
    let list: Vec<String> = by
        .iter()
        .map(|(c, m)| format!("{c} {m:.0} ({:.0} %)", m / total * 100.0))
        .collect();
    eprintln!("[mass] {:<8} by category: {}", car.label, list.join(", "));
}

/// Everything, per car (`--nocapture`; grep the tags: `[part]`, `[beam]`,
/// `[types]`, `[hold]`, `[tris]`, `[flex]`, `[mass]`).
#[test]
#[ignore]
fn authoring_census() {
    for car in all_cars() {
        let x = Ctx::new(&car);
        report_parts(&x);
        report_types(&x);
        report_holding(&x);
        report_tris(&x);
        report_flex(&x);
        report_mass(&x);
    }
}

/// The raw model of each `RUBBLE_CENSUS_CARS` car as one JSON file, for
/// `engine/tools/beamng-structure-study/analyze.py`
/// (`engine/docs/beamng-structure-study.md`). Written to `RUBBLE_STRUCTURE_OUT`
/// (a directory outside the repository: the files hold the cars' node tables
/// and meshes, which are never committed). Skipped when either variable is
/// absent.
///
/// Positions are in the JBeam frame (+X left, +Y rear, +Z up), metres. Per car:
/// the assembled parts and their slot types, every assembled row of the
/// physics and visual sections with the part that defines it, each node's
/// final position and mass as the loader builds it, the node groups, and each
/// flexbody's mesh (vertices and triangles, its row's transform applied).
#[test]
#[ignore]
fn structure_dump() {
    use engine_softbody::jbeam::{assembled_parts, assembled_rows, load_jbeam, read_vehicle_dir};
    let (Ok(list), Ok(out)) = (
        std::env::var("RUBBLE_CENSUS_CARS"),
        std::env::var("RUBBLE_STRUCTURE_OUT"),
    ) else {
        eprintln!("skip: RUBBLE_CENSUS_CARS and RUBBLE_STRUCTURE_OUT not both set");
        return;
    };
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    const SECTIONS: [&str; 16] = [
        "nodes",
        "beams",
        "triangles",
        "flexbodies",
        "props",
        "torsionbars",
        "slots",
        "slots2",
        "pressureWheels",
        "hubWheels",
        "hydros",
        "rails",
        "slidenodes",
        "powertrain",
        "glowMap",
        "quads",
    ];
    for item in list.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let Some((label, rest)) = item.split_once('=') else {
            continue;
        };
        let f: Vec<&str> = rest.split(',').collect();
        let (dir, name, pc) = (
            std::path::Path::new(f.first().copied().unwrap_or("")),
            f.get(1).copied().unwrap_or(""),
            f.get(2).copied().filter(|p| !p.is_empty()),
        );
        if !dir.join(name).is_dir() {
            eprintln!("skip {label}: {} absent", dir.join(name).display());
            continue;
        }
        let run = || -> Result<Value, String> {
            let texts = read_vehicle_dir(dir, name)?;
            let files: Vec<(&str, &str)> = texts
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            let pc_file = pc;
            let pc = match pc {
                Some(p) => Some(
                    std::fs::read_to_string(dir.join(name).join(p)).map_err(|e| e.to_string())?,
                ),
                None => None,
            };
            let pc = pc.as_deref();
            let desc = load_jbeam(&files, pc)?;
            let mut root = Map::new();
            root.insert("label".into(), label.into());
            root.insert(
                "parts".into(),
                assembled_parts(&files, pc)?
                    .into_iter()
                    .map(|(n, s)| serde_json::json!([n, s]))
                    .collect(),
            );
            let mut rows = Map::new();
            for s in SECTIONS {
                let v: Vec<Value> = assembled_rows(&files, pc, s)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(p, f)| serde_json::json!({ "part": p, "row": f }))
                    .collect();
                rows.insert(s.into(), v.into());
            }
            root.insert("rows".into(), rows.into());
            root.insert(
                "nodes".into(),
                desc.nodes
                    .iter()
                    .map(|n| {
                        let p = n.position;
                        serde_json::json!({
                            "id": n.name, "pos": [p[0], -p[2], p[1]], "mass": n.mass,
                            "collision": n.collision, "selfCollision": n.self_collision,
                        })
                    })
                    .collect(),
            );
            root.insert(
                "groups".into(),
                desc.node_group_nodes
                    .iter()
                    .map(|(g, v)| serde_json::json!([g, v]))
                    .collect(),
            );
            // The flexbody meshes, as `load_jbeam_car` reads them.
            let car = load_jbeam_car(label, dir, name, pc_file)?;
            let to_j = |p: Vec3| [p.x, -p.z, p.y];
            root.insert(
                "flex".into(),
                car.flex
                    .iter()
                    .map(|fx| {
                        serde_json::json!({
                            "part": car.parts[fx.part].0, "mesh": fx.mesh, "groups": fx.groups,
                            "pool": fx.pool,
                            "pos": fx.pos.iter().map(|&p| to_j(p)).collect::<Vec<_>>(),
                            "tris": fx.tris,
                        })
                    })
                    .collect(),
            );
            Ok(root.into())
        };
        match run() {
            Ok(v) => {
                let p = out.join(format!("{label}.json"));
                std::fs::write(&p, serde_json::to_vec(&v).unwrap()).unwrap();
                eprintln!("wrote {}", p.display());
            }
            Err(e) => eprintln!("skip {label}: {e}"),
        }
    }
}
