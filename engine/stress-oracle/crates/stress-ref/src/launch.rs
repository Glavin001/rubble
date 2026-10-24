//! Launched bonded bodies: free islands of a structure (chunks bonded to each other but,
//! through no path of bonds, to a support) that a host sets moving between two frames,
//! never by re-running one. A launched car, bus or aircraft is such an island: it is the
//! structure's own chunks and bonds, so it collides with the rest of the structure and with
//! itself, and its bonds break as any bond does.
//!
//! A pack carries its launches as `defaults.launches` ([`write`], read with
//! [`launches_of`]), in the packs' frame (Y-up, m/s):
//!
//! ```json
//! {"name": "car", "nodes": [node, ...], "velocity": {"x": 0, "y": 0, "z": 15},
//!  "angularVelocity": {"x": 0, "y": 0, "z": 0}, "delay": 0,
//!  "strikes": [{"centre": {"x": 0, "y": 0.6, "z": 2.0}, "radius": 0.9}]}
//! ```
//!
//! * `nodes` are the island's node indices (every import keeps the pack's node order);
//!   `group` (a `nodeGroups` name) may stand in for them.
//! * `velocity` is the island's centre-of-mass velocity and `angularVelocity` its spin:
//!   each body the island is made of when it is released gets `v + w x (c - c_island)`
//!   (its centre `c`), so a launch is a rigid motion of the whole island.
//! * `delay` is in seconds after the launch is asked for. Until then the island is a free
//!   body like any other (a car stands on its wheels, an aircraft placed in the air falls):
//!   launch an airborne island at once.
//! * `strikes` are spheres fixed in the island (centre in the structure frame, as built),
//!   its leading parts (a car's nose, an aircraft's nose and wings), which the threshold
//!   profile's strike prediction checks as it checks a shot: a strike they will make within
//!   the next frame breaks the struck body's bonds before the engine steps
//!   (`stress-gpu` `fracture.slang` `fr_strike_*`), so a fast island goes through a shop
//!   window as a ball does instead of stopping dead on it. Optional: without them the
//!   island's contacts break bonds after the step only.

use serde_json::{json, Value};

use crate::scene::{BondDesc, ChunkDesc, Support};

/// A sphere fixed in a launched island for strike prediction (Z-up, the structure frame
/// as built, m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Strike {
    pub centre: [f64; 3],
    pub radius: f64,
}

/// A launch (Z-up, the stress scene's frame): the island's chunks, its centre-of-mass
/// velocity and spin (m/s, rad/s), the delay after it is asked for (s) and its strike
/// spheres.
#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub name: String,
    /// Chunk (node) indices, ascending.
    pub chunks: Vec<usize>,
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub delay: f64,
    pub strikes: Vec<Strike>,
}

/// Z-up to the packs' Y-up (`scene_pack`'s map `(x, y, z) -> (x, -z, y)` inverted).
fn y_up(v: [f64; 3]) -> [f64; 3] {
    [v[0], v[2], -v[1]]
}

fn z_up(v: [f64; 3]) -> [f64; 3] {
    [v[0], -v[2], v[1]]
}

fn xyz(v: [f64; 3]) -> Value {
    let p = y_up(v);
    json!({"x": p[0], "y": p[1], "z": p[2]})
}

fn read(v: &Value) -> [f64; 3] {
    z_up([v["x"].as_f64().unwrap_or(0.0), v["y"].as_f64().unwrap_or(0.0), v["z"].as_f64().unwrap_or(0.0)])
}

/// Write `launches` into a pack as `defaults.launches` (none: the key is left out).
pub fn write(pack: &mut Value, launches: &[Launch]) {
    if launches.is_empty() {
        return;
    }
    pack["defaults"]["launches"] = launches
        .iter()
        .map(|l| {
            json!({
                "name": l.name,
                "nodes": l.chunks,
                "velocity": xyz(l.velocity),
                "angularVelocity": xyz(l.angular_velocity),
                "delay": l.delay,
                "strikes": l.strikes.iter().map(|s| json!({"centre": xyz(s.centre), "radius": s.radius})).collect::<Vec<_>>(),
            })
        })
        .collect();
}

/// The launches a pack carries (`defaults.launches`), in the stress scene's frame (Z-up),
/// in the pack's order; empty for a pack without any. A launch given by `group` gets the
/// nodes of that `nodeGroups` name. An error names a launch without a name, without nodes
/// or with a node the pack does not have.
pub fn launches_of(pack: &Value) -> Result<Vec<Launch>, String> {
    let Some(list) = pack["defaults"]["launches"].as_array() else { return Ok(Vec::new()) };
    let nodes = pack["scenario"]["nodes"].as_array().map_or(0, |n| n.len());
    let groups: Vec<&str> = pack["scenario"]["nodeGroups"].as_array().map(|g| g.iter().map(|g| g.as_str().unwrap_or("")).collect()).unwrap_or_default();
    list.iter()
        .map(|l| {
            let name = l["name"].as_str().ok_or("a launch without a name")?.to_string();
            let mut chunks: Vec<usize> = match (l["nodes"].as_array(), l["group"].as_str()) {
                (Some(n), _) => n.iter().map(|n| n.as_u64().map(|n| n as usize).ok_or_else(|| format!("launch {name}: a node that is not an index"))).collect::<Result<_, _>>()?,
                (None, Some(g)) => (0..groups.len()).filter(|&i| groups[i] == g).collect(),
                (None, None) => return Err(format!("launch {name}: no nodes and no group")),
            };
            chunks.sort_unstable();
            chunks.dedup();
            if chunks.is_empty() {
                return Err(format!("launch {name}: no nodes"));
            }
            if let Some(&n) = chunks.iter().find(|&&n| n >= nodes) {
                return Err(format!("launch {name}: node {n} of a pack of {nodes}"));
            }
            let strikes = l["strikes"].as_array().map(|s| s.iter().map(|s| Strike { centre: read(&s["centre"]), radius: s["radius"].as_f64().unwrap_or(0.0) }).collect()).unwrap_or_default();
            Ok(Launch {
                name,
                chunks,
                velocity: read(&l["velocity"]),
                angular_velocity: read(&l["angularVelocity"]),
                delay: l["delay"].as_f64().unwrap_or(0.0),
                strikes,
            })
        })
        .collect()
}

/// The free islands of a structure: connected components over `bonds` with no support
/// among their chunks, each ascending, in order of their lowest chunk.
pub fn free_islands(chunks: &[ChunkDesc], bonds: &[BondDesc]) -> Vec<Vec<usize>> {
    let n = chunks.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for b in bonds {
        let (ra, rb) = (find(&mut parent, b.a), find(&mut parent, b.b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    let mut islands: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for i in 0..n {
        let r = find(&mut parent, i);
        islands.entry(r).or_default().push(i);
    }
    islands.into_values().filter(|c| c.iter().all(|&k| chunks[k].support == Support::None)).collect()
}

/// The capability's check (`tests/launch.rs` here and in stress-rubble): a concrete wall,
/// 4 m wide, 2 m tall and 0.3 m thick (4 x 4 blocks of 1 x 0.5 m on a strip footing), and a
/// free bonded concrete ram (a T of five 0.5 m cubes: three in a row along +y, two wings on the rear one; 1.5 t) 1.25 m up and
/// 4 m in front of it. Its launch, "ram", sends it at the wall at 30 m/s (with the lift
/// that keeps it level at the wall under gravity), its strike sphere on its front cube.
/// Returns the pack.
pub fn ram_and_wall() -> Value {
    use crate::builders::{auto_bonds, chunk};
    use crate::math::Vec3;
    let mut chunks = Vec::new();
    let tag = |mut c: ChunkDesc, kind: &str| {
        c.groups = vec![kind.into(), String::new()];
        c
    };
    let mut footing = tag(chunk(Vec3::new(0.0, 0.0, 0.25), Vec3::new(2.0, 0.15, 0.25), "footing-anchor"), "footing");
    footing.support = Support::Fixed;
    chunks.push(footing);
    for row in 0..4 {
        for col in 0..4 {
            chunks.push(tag(chunk(Vec3::new(-1.5 + col as f64, 0.0, 0.75 + 0.5 * row as f64), Vec3::new(0.5, 0.15, 0.25), "reinforced-concrete"), "wall"));
        }
    }
    let first = chunks.len();
    let (speed, z, standoff) = (30.0, 1.25, 4.0);
    for k in 0..3 {
        chunks.push(tag(chunk(Vec3::new(0.0, -0.15 - standoff - 0.25 - 0.5 * k as f64, z), Vec3::splat(0.25), "reinforced-concrete"), "ram"));
    }
    // Its wings, on the rear cube's sides (a T): they hang on their bonds in shear when it stops.
    for x in [-0.5, 0.5] {
        chunks.push(tag(chunk(Vec3::new(x, -0.15 - standoff - 0.25 - 1.0, z), Vec3::splat(0.25), "reinforced-concrete"), "ram"));
    }
    let bonds: Vec<BondDesc> = auto_bonds(&chunks, |_, _| "reinforced-concrete".into());
    let s = crate::wrecking::Structure {
        key: "launch_ram_wall",
        title: "Launch check: a concrete ram at a wall",
        chunks,
        bonds,
        aim: crate::wrecking::Aim { target: [0.0, -0.15, z], direction: [0.0, 1.0, 0.0] },
        ball: crate::wrecking::BALLS[0],
        charges: Vec::new(),
    };
    let mut pack = crate::wrecking::to_pack(&s);
    // The lift that brings it to the wall at its height: half the fall over the flight.
    let flight = standoff / speed;
    let ram = Launch {
        name: "ram".into(),
        chunks: (first..first + 5).collect(),
        velocity: [0.0, speed, 0.5 * 9.81 * flight],
        angular_velocity: [0.0; 3],
        delay: 0.0,
        strikes: vec![Strike { centre: s.chunks[first].center, radius: 0.25 }],
    };
    write(&mut pack, &[ram]);
    pack
}

/// A launch's chunks are one free island of the structure (all of it): an error otherwise,
/// naming the launch.
pub fn check_island(launch: &Launch, chunks: &[ChunkDesc], bonds: &[BondDesc]) -> Result<(), String> {
    if free_islands(chunks, bonds).iter().any(|i| *i == launch.chunks) {
        Ok(())
    } else {
        Err(format!("launch {}: its {} chunks are not one whole free island of the structure", launch.name, launch.chunks.len()))
    }
}
