//! Stage-2 recipes (the plan's houses and arch, multi-load tiers 3-5): structures that stand
//! under gravity on a ground, loaded by several bodies at once.
//!
//! Families:
//! - `cuboid_house`: a parametric one-chunk-thick shell (four walls and a roof, a door) of
//!   brick or concrete, footing fixed, on a ground;
//! - `masonry_house`: the showcase brick house (running bond, lintels, roof load);
//! - `arch`: the showcase masonry arch.
//!
//! Load tiers (`plan` P1 "Multiple simultaneous loads"):
//! - 3, mixed regimes: a resting load (a heavy body set down on the top, at rest), a slow
//!   push (a heavy body pressing at 1-5 m/s) and one or two fast impacts, at once;
//! - 4, impacts and a blast together, and a repeat hit (a second impactor on the first's
//!   spot a little later);
//! - 5, a whole building with 5-12 loads of every kind spread over it.
//!
//! Debris re-hits arise from the loads themselves (the recorder tags them `secondary`).
//! Every draw is independent; nothing here changes the oracle's accuracy settings.

use serde_json::{json, Value};
use stress_ref::builders;
use stress_ref::material::Material;
use stress_ref::math::Vec3;
use stress_ref::scene::{GroundDesc, ImpactorDesc, ImpactorShape, LoadDesc, Scene, Support};
use stress_ref::showcases::{self, HouseEvent};

use crate::outcome::OutcomeRecipe;
use crate::procedural::{bounding_radius, latin_hypercube, log_uniform, scale_materials, Rng};

pub const STAGE2_FAMILIES: [&str; 3] = ["cuboid_house", "masonry_house", "arch"];

/// A cuboid shell: `n` chunks of `cell` metres along x, y and z, one chunk thick, a roof,
/// a door in the front wall (y = 0), the footing row fixed, standing on z = 0.
pub fn cuboid_house(n: [usize; 3], cell: f64, material: &str) -> stress_ref::scene::BodyDesc {
    let [nx, ny, nz] = n;
    let door = (nx / 2 - 1, nx / 2 + 1, (nz * 2 / 3).max(2));
    let mut chunks = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let shell = x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == nz - 1;
                let in_door = y == 0 && x >= door.0 && x < door.1 && z >= 1 && z < door.2;
                if !shell || in_door {
                    continue;
                }
                let mut c = builders::chunk(Vec3::new((x as f64 + 0.5) * cell, (y as f64 + 0.5) * cell, (z as f64 + 0.5) * cell), Vec3::splat(0.5 * cell), material);
                if z == 0 {
                    c.support = Support::Fixed;
                    c.groups.push("footing".into());
                }
                chunks.push(c);
            }
        }
    }
    let bonds = builders::auto_bonds(&chunks, |_, _| material.to_string());
    builders::body("house", chunks, bonds)
}

/// The kinds of load a stage-2 recipe places.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadKind {
    Resting,
    Push,
    Impact,
    Repeat,
}

impl LoadKind {
    fn name(self) -> &'static str {
        match self {
            LoadKind::Resting => "resting",
            LoadKind::Push => "push",
            LoadKind::Impact => "impact",
            LoadKind::Repeat => "repeat",
        }
    }
}

/// The structure's axis-aligned box (min, max) from its chunks.
fn bounds(scene: &Scene) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
    for b in &scene.bodies {
        for c in &b.chunks {
            // The chunk's own box (rotated chunks: its bounding cube, by the rotation's support).
            let (x, h) = (Vec3::from_array(c.center), Vec3::from_array(c.half_extents));
            let r = stress_ref::math::Quat::from_wxyz(c.orientation).to_mat3();
            let ext = Vec3::new(
                (0..3).map(|k| r.m[0][k].abs() * h[k]).sum(),
                (0..3).map(|k| r.m[1][k].abs() * h[k]).sum(),
                (0..3).map(|k| r.m[2][k].abs() * h[k]).sum(),
            );
            lo = lo.component_min(x - ext);
            hi = hi.component_max(x + ext);
        }
    }
    (lo, hi)
}

fn clear(d: &ImpactorDesc, placed: &[ImpactorDesc], lo: Vec3, hi: Vec3, ground: f64) -> bool {
    let (c, r) = (Vec3::from_array(d.position), bounding_radius(d) + 0.002);
    if c.z - r < ground {
        return false;
    }
    if (c.component_max(lo).component_min(hi) - c).norm() < r {
        return false;
    }
    placed.iter().all(|o| (Vec3::from_array(o.position) - c).norm() >= r + bounding_radius(o))
}

/// One load placed on the structure's box: a side face (or the top, for a resting load) at
/// fractions `(u, w)` across it.
fn place(kind: LoadKind, lo: Vec3, hi: Vec3, rng: &mut Rng, k: usize, repeat_of: Option<(Vec3, Vec3, Vec3)>) -> (ImpactorDesc, Value, Vec3, Vec3, Vec3) {
    let size = rng.range(0.1, 0.35);
    let sphere = rng.unit() < 0.4;
    let shape = if sphere { ImpactorShape::Sphere { radius: size } } else { ImpactorShape::Box { half_extents: [size, size, size] } };
    let reach = if sphere { size } else { size * 3f64.sqrt() };
    let (mass, speed) = match kind {
        LoadKind::Resting => (log_uniform(rng.unit(), 200.0, 3000.0), 0.0),
        LoadKind::Push => (log_uniform(rng.unit(), 800.0, 3000.0), rng.range(1.0, 5.0)),
        LoadKind::Impact | LoadKind::Repeat => (log_uniform(rng.unit(), 50.0, 2000.0), log_uniform(rng.unit(), 3.0, 40.0)),
    };
    // (point on the face, direction of travel, inward face normal, face)
    let (point, dir, n_in, face) = if let (LoadKind::Repeat, Some((at, d, n))) = (kind, repeat_of) {
        // The first impact's spot, a few centimetres off, along its direction.
        (at + Vec3::new(rng.range(-0.05, 0.05), 0.0, rng.range(-0.05, 0.05)), d, n, "repeat")
    } else if kind == LoadKind::Resting {
        let p = Vec3::new(lo.x + rng.range(0.1, 0.9) * (hi.x - lo.x), lo.y + rng.range(0.1, 0.9) * (hi.y - lo.y), hi.z);
        (p, Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, -1.0), "top")
    } else {
        // One of the four side faces, inward.
        let f = rng.int(0, 3);
        let (u, w) = (rng.range(0.1, 0.9), rng.range(0.1, 0.8));
        let z = lo.z + w * (hi.z - lo.z);
        let (p, n) = match f {
            0 => (Vec3::new(lo.x + u * (hi.x - lo.x), lo.y, z), Vec3::new(0.0, 1.0, 0.0)),
            1 => (Vec3::new(lo.x + u * (hi.x - lo.x), hi.y, z), Vec3::new(0.0, -1.0, 0.0)),
            2 => (Vec3::new(lo.x, lo.y + u * (hi.y - lo.y), z), Vec3::new(1.0, 0.0, 0.0)),
            _ => (Vec3::new(hi.x, lo.y + u * (hi.y - lo.y), z), Vec3::new(-1.0, 0.0, 0.0)),
        };
        // Up to 30 degrees off the normal, in the horizontal plane.
        let a = rng.range(-0.5, 0.5);
        let t = Vec3::new(-n.y, n.x, 0.0);
        (p, (n * a.cos() + t * a.sin()).normalized(), n, ["front", "back", "left", "right"][f])
    };
    let arrival = match kind {
        LoadKind::Resting | LoadKind::Push => 0.0,
        LoadKind::Impact => rng.range(0.002, 0.1),
        LoadKind::Repeat => rng.range(0.1, 0.3),
    };
    // Off the face along its outward normal by the body's reach, then back along its path.
    let start = point - n_in * (reach + 0.003) - dir * (speed * arrival);
    let desc = ImpactorDesc {
        name: format!("{}{k}", kind.name()),
        shape,
        mass,
        position: start.to_array(),
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: (dir * speed).to_array(),
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    };
    let params = json!({"kind": kind.name(), "face": face, "mass": mass, "speed": speed, "size": size, "shape": if sphere { "sphere" } else { "box" }, "arrival": arrival});
    (desc, params, point, dir, n_in)
}

/// `n` stage-2 recipes of `family` at load `tier` (3, 4 or 5).
#[allow(clippy::too_many_lines)]
pub fn stage2_recipes(family: &str, tier: u32, n: usize, seed: u64) -> Result<Vec<OutcomeRecipe>, String> {
    if !STAGE2_FAMILIES.contains(&family) || !(3..=5).contains(&tier) {
        return Err(format!("unknown stage-2 family/tier {family}/{tier}"));
    }
    let mut rng = Rng::new(seed ^ ((tier as u64) << 40) ^ 0x5742);
    let lhs = latin_hypercube(n, 6, &mut rng);
    let mut out = Vec::with_capacity(n);
    for (i, u) in lhs.iter().enumerate() {
        let (strength, toughness, stiffness) = (log_uniform(u[0], 0.5, 2.0), log_uniform(u[1], 0.5, 2.0), log_uniform(u[2], 0.7, 1.5));
        let (mut scene, layout, ground_z) = match family {
            "cuboid_house" => {
                let pick = |v: f64, a: usize, b: usize| a + ((v * (b - a + 1) as f64) as usize).min(b - a);
                let dims = [pick(u[3], 6, 12), pick(u[4], 6, 12), pick(u[5], 6, 10)];
                let cell = if rng.unit() < 0.5 { 0.2 } else { 0.25 };
                let material = if rng.unit() < 0.5 { "brick" } else { "concrete" };
                let mut s = builders::new_scene("stage2", "P1 stage-2 cuboid house under several loads");
                s.materials.insert("brick".into(), Material::brick());
                s.materials.insert("concrete".into(), builders::oracle_concrete(None));
                s.bodies.push(cuboid_house(dims, cell, material));
                s.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
                (s, json!({"dims": dims, "cell": cell, "material": material, "key": format!("{}x{}x{}_c{:.0}mm_{material}", dims[0], dims[1], dims[2], cell * 1000.0)}), 0.0)
            }
            "masonry_house" => {
                let mut s = showcases::masonry_house(HouseEvent::None, "stage2");
                s.ground = Some(GroundDesc { height: -0.15, friction: 0.6, material: "brick".into() });
                (s, json!({"key": "masonry_house"}), -0.15)
            }
            _ => {
                let mut s = showcases::arch(None, "stage2");
                s.ground = Some(GroundDesc { height: -0.15, friction: 0.6, material: "brick".into() });
                (s, json!({"key": "arch"}), -0.15)
            }
        };
        scene.materials.insert("steel".into(), builders::elastic_steel());
        scale_materials(&mut scene, strength, toughness, stiffness);
        let (lo, hi) = bounds(&scene);
        let kinds: Vec<LoadKind> = match tier {
            3 => {
                let mut k = vec![LoadKind::Resting, LoadKind::Push, LoadKind::Impact];
                if rng.unit() < 0.5 {
                    k.push(LoadKind::Impact);
                }
                k
            }
            4 => vec![LoadKind::Impact, LoadKind::Repeat],
            _ => {
                let m = rng.int(5, 12);
                (0..m).map(|j| match j % 4 { 0 => LoadKind::Impact, 1 => LoadKind::Resting, 2 => LoadKind::Push, _ => LoadKind::Impact }).collect()
            }
        };
        let mut placed: Vec<ImpactorDesc> = Vec::new();
        let mut loads: Vec<Value> = Vec::new();
        let mut first_hit: Option<(Vec3, Vec3, Vec3)> = None;
        let mut redraws = 0;
        for (k, &kind) in kinds.iter().enumerate() {
            let mut attempt = 0;
            loop {
                let (d, p, at, dir, n_in) = place(kind, lo, hi, &mut rng, k, first_hit);
                if clear(&d, &placed, lo, hi, ground_z) {
                    if kind == LoadKind::Impact && first_hit.is_none() {
                        first_hit = Some((at, dir, n_in));
                    }
                    placed.push(d);
                    loads.push(p);
                    break;
                }
                attempt += 1;
                redraws += 1;
                if attempt > 200 {
                    return Err(format!("stage-2 recipe {i}: no clear start for load {k} ({kind:?})"));
                }
            }
        }
        scene.impactors = placed;
        let blast = (tier == 4 || (tier == 5 && rng.unit() < 0.5)).then(|| {
            let (tnt, standoff) = (log_uniform(rng.unit(), 0.1, 2.0), rng.range(2.0, 6.0));
            let at = [0.5 * (lo.x + hi.x), lo.y - standoff, lo.z + 0.4 * (hi.z - lo.z)];
            let time = rng.range(0.0, 0.1);
            scene.loads.push(LoadDesc::Blast { position: at, tnt_mass: tnt, time });
            json!({"tnt": tnt, "standoff": standoff, "position": at, "time": time})
        });
        let event_id = format!("p1_{family}_t{tier}_{i:05}_s{seed}");
        scene.name = event_id.clone();
        let layout_key = layout["key"].as_str().unwrap_or(family).to_string();
        let params = json!({"structure": layout, "material": {"strength_scale": strength, "toughness_scale": toughness, "stiffness_scale": stiffness},
            "loads": loads, "blast": blast, "seed": seed, "index": i, "catalog": "p1-stage2", "redraws": redraws});
        out.push(OutcomeRecipe { event_id: event_id.clone(), family: family.into(), tier, split_group: event_id, layout_group: format!("{family}:{layout_key}"), parameters: params, scene });
    }
    Ok(out)
}
