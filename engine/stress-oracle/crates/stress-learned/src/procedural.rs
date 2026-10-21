//! Procedural wall recipes for the impact-outcome dataset (plan P1, tiers 1 and 2).
//!
//! Every factor is drawn independently: a Latin hypercube over the factors of each
//! family (not paired single-factor sweeps), so no two factors are correlated by design.
//! The impact speed is not drawn directly: the punch's force-to-capacity ratio at the
//! struck point is drawn log-uniform around 1, and the speed that gives it is solved for,
//! so samples crowd the arrest/penetration transition instead of the easy extremes.
//!
//! Solver accuracy is never touched here: the oracle configuration is applied by the
//! recorder (`OracleConfig`); recipes only describe physical scenes.

use serde_json::json;
use stress_ref::builders;
use stress_ref::math::Vec3;
use stress_ref::scene::{GroundDesc, ImpactorDesc, ImpactorShape, Scene, Support};
use stress_ref::world::World;

use crate::outcome::OutcomeRecipe;

/// splitmix64: a small deterministic generator (recipes must not depend on a crate's RNG
/// version).
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e3779b97f4a7c15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    pub fn int(&mut self, lo: usize, hi_inclusive: usize) -> usize {
        lo + (self.next_u64() % (hi_inclusive - lo + 1) as u64) as usize
    }
}

/// `n` points of a Latin hypercube in `[0, 1)^dims`: each dimension's `n` strata are each
/// used once, in an independent random order per dimension.
pub fn latin_hypercube(n: usize, dims: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(dims);
    for _ in 0..dims {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        cols.push(perm.iter().map(|&k| (k as f64 + rng.unit()) / n as f64).collect());
    }
    (0..n).map(|i| cols.iter().map(|c| c[i]).collect()).collect()
}

fn log_uniform(u: f64, lo: f64, hi: f64) -> f64 {
    (lo.ln() + u * (hi.ln() - lo.ln())).exp()
}

/// The ranges of a catalog's walls and impactors.
#[derive(Clone, Debug)]
pub struct Ranges {
    pub nx: (usize, usize),
    pub ny: (usize, usize),
    pub nz: (usize, usize),
    pub chunk_sizes: Vec<f64>,
    pub opening_probability: f64,
    pub mass: (f64, f64),
    pub impactor_size: (f64, f64),
    /// Punch force-to-capacity ratio the speed is solved for (log-uniform).
    pub punch_ratio: (f64, f64),
    pub speed: (f64, f64),
    pub max_angle_degrees: f64,
    pub spin_probability: f64,
    pub max_spin: f64,
    pub weibull_probability: f64,
    /// Share of first impactors drawn from the sub-threshold band `low_speed` (log-uniform),
    /// for the plan's ~20% of events that break nothing (the punch-ratio draw alone gave 1-15%).
    pub low_speed_share: f64,
    pub low_speed: (f64, f64),
}

impl Ranges {
    /// Small walls for pilots: cheap enough to measure cost per family quickly.
    pub fn pilot() -> Ranges {
        Ranges {
            nx: (6, 14),
            ny: (1, 2),
            nz: (6, 12),
            chunk_sizes: vec![0.1, 0.15],
            opening_probability: 0.25,
            mass: (50.0, 1500.0),
            impactor_size: (0.05, 0.2),
            // Centred on the punch's threshold; the oracle breaks bonds well below it (pilots
            // v1-v3: at 1-2 m/s 88% of hits broke something), hence `low_speed_share`.
            punch_ratio: (0.03, 20.0),
            speed: (0.5, 60.0),
            max_angle_degrees: 35.0,
            spin_probability: 0.3,
            max_spin: 20.0,
            weibull_probability: 0.3,
            low_speed_share: 0.2,
            low_speed: (0.1, 1.0),
        }
    }

    /// The bulk ranges (plan P1: size, thickness, resolution, openings, supports).
    pub fn bulk() -> Ranges {
        Ranges { nx: (6, 28), ny: (1, 3), nz: (6, 26), chunk_sizes: vec![0.1, 0.15, 0.2], mass: (50.0, 4000.0), impactor_size: (0.05, 0.3), ..Ranges::pilot() }
    }
}

/// Scale the structural material's strength, toughness and stiffness; the steel of the
/// impactors stays as it is (`permutations.rs`'s helper, same rule).
fn scale_materials(scene: &mut Scene, strength: f64, toughness: f64, stiffness: f64) {
    for (name, m) in &mut scene.materials {
        if name == "steel" {
            continue;
        }
        m.tensile_strength *= strength;
        m.compressive_strength *= strength;
        m.cohesion *= strength;
        if let Some(cap) = &mut m.shear_cap {
            *cap *= strength;
        }
        m.fracture_energy.tension *= toughness;
        m.fracture_energy.shear *= toughness;
        m.fracture_energy.compression *= toughness;
        m.youngs_modulus *= stiffness;
    }
}

/// A wall's layout: chunk counts and size, an opening (grid ranges `[x0, x1, z0, z1)`,
/// piers, foundation and lintel kept) and its supports.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub counts: [usize; 3],
    pub chunk: f64,
    pub opening: Option<[usize; 4]>,
    /// "free", "base" (bottom row fixed) or "base_sides" (bottom row and both ends).
    pub supports: &'static str,
}

impl Layout {
    pub fn key(&self) -> String {
        let [nx, ny, nz] = self.counts;
        let o = self.opening.map_or("solid".to_string(), |[a, b, c, d]| format!("o{a}-{b}-{c}-{d}"));
        format!("{nx}x{ny}x{nz}_c{:.0}mm_{o}_{}", self.chunk * 1000.0, self.supports)
    }

    pub fn size(&self) -> Vec3 {
        Vec3::new(self.counts[0] as f64, self.counts[1] as f64, self.counts[2] as f64) * self.chunk
    }

    fn sample(r: &Ranges, supports: &'static str, u: &[f64], rng: &mut Rng) -> Layout {
        let pick = |u: f64, (lo, hi): (usize, usize)| lo + ((u * (hi - lo + 1) as f64) as usize).min(hi - lo);
        let counts = [pick(u[0], r.nx), pick(u[1], r.ny), pick(u[2], r.nz)];
        let chunk = r.chunk_sizes[((u[3] * r.chunk_sizes.len() as f64) as usize).min(r.chunk_sizes.len() - 1)];
        let [nx, _, nz] = counts;
        let opening = (rng.unit() < r.opening_probability && nx >= 6 && nz >= 6).then(|| {
            let w = rng.int(1, (nx - 4) / 2);
            let x0 = rng.int(2, nx - 2 - w);
            let h = rng.int(1, (nz - 4) / 2);
            let z0 = rng.int(1, nz - 2 - h);
            [x0, x0 + w, z0, z0 + h]
        });
        Layout { counts, chunk, opening, supports }
    }
}

/// The wall's chunks (front face at y = 0, the wall extending to +y; x centred; the bottom
/// at z = 0), bonded face to face.
pub fn wall_body(layout: &Layout, name: &str, y0: f64) -> stress_ref::scene::BodyDesc {
    let [nx, ny, nz] = layout.counts;
    let cs = layout.chunk;
    let width = nx as f64 * cs;
    let mut chunks = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if layout.opening.is_some_and(|[x0, x1, z0, z1]| x >= x0 && x < x1 && z >= z0 && z < z1) {
                    continue;
                }
                let mut c = builders::chunk(Vec3::new(-width * 0.5 + (x as f64 + 0.5) * cs, y0 + (y as f64 + 0.5) * cs, (z as f64 + 0.5) * cs), Vec3::splat(cs * 0.5), "concrete");
                let fixed = match layout.supports {
                    "base" => z == 0,
                    "base_sides" => z == 0 || x == 0 || x == nx - 1,
                    _ => false,
                };
                if fixed {
                    c.support = Support::Fixed;
                    c.groups.push("support".into());
                }
                chunks.push(c);
            }
        }
    }
    let bonds = builders::auto_bonds(&chunks, |_, _| "concrete".into());
    builders::body(name, chunks, bonds)
}

/// One impactor's draw.
#[derive(Clone, Debug)]
pub struct ImpactorDraw {
    pub mass: f64,
    pub sphere: bool,
    pub size: f64,
    /// Impact point on the face as fractions of width and height (may be past an edge: a
    /// grazing hit), and which face (false: front, y = 0; true: back).
    pub x_fraction: f64,
    pub z_fraction: f64,
    pub back: bool,
    pub yaw_degrees: f64,
    pub pitch_degrees: f64,
    pub spin: [f64; 3],
    pub punch_ratio: f64,
    /// Time after the start at which it reaches the face (s).
    pub arrival: f64,
    /// Drawn from the sub-threshold speed band instead of the punch-ratio draw.
    pub low: bool,
}

impl ImpactorDraw {
    fn location_tag(&self) -> &'static str {
        let (x, z) = ((self.x_fraction - 0.5).abs(), (self.z_fraction - 0.5).abs());
        if x > 0.5 || z > 0.5 {
            "grazing"
        } else if x > 0.35 && z > 0.35 {
            "corner"
        } else if x > 0.35 || z > 0.35 {
            "edge"
        } else {
            "centre"
        }
    }
}

/// The impactor's description at a speed: placed so it reaches its point on the face at
/// `arrival`, moving along its yaw and pitch.
fn place(d: &ImpactorDraw, layout: &Layout, y_back: f64, speed: f64, k: usize) -> ImpactorDesc {
    let size = layout.size();
    let (yaw, pitch) = (d.yaw_degrees.to_radians(), d.pitch_degrees.to_radians());
    let sign = if d.back { -1.0 } else { 1.0 };
    let dir = Vec3::new(yaw.sin() * pitch.cos(), sign * yaw.cos() * pitch.cos(), pitch.sin());
    let hit = Vec3::new((d.x_fraction - 0.5) * size.x, if d.back { y_back } else { 0.0 }, d.z_fraction * size.z);
    let (shape, reach) = if d.sphere {
        (ImpactorShape::Sphere { radius: d.size }, d.size)
    } else {
        let h = [d.size, d.size, d.size];
        // Support distance of the axis-aligned box along the travel direction.
        (ImpactorShape::Box { half_extents: h }, d.size * (dir.x.abs() + dir.y.abs() + dir.z.abs()))
    };
    let start = hit - dir * (reach + 0.002 + speed * d.arrival);
    ImpactorDesc {
        name: format!("ram{k}"),
        shape,
        mass: d.mass,
        position: start.to_array(),
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: (dir * speed).to_array(),
        angular_velocity: d.spin,
        material: "steel".into(),
        crush: None,
    }
}

fn bounding_radius(d: &ImpactorDesc) -> f64 {
    match d.shape {
        ImpactorShape::Sphere { radius } => radius,
        ImpactorShape::Box { half_extents } => Vec3::from_array(half_extents).norm(),
    }
}

/// Whether `d` starts clear of the ground (if any), the wall's box and the impactors
/// already placed, by its bounding sphere with 2 mm to spare.
fn starts_clear(d: &ImpactorDesc, placed: &[ImpactorDesc], layout: &Layout, ground: bool) -> bool {
    let (c, r) = (Vec3::from_array(d.position), bounding_radius(d) + 0.002);
    if ground && c.z - r < 0.0 {
        return false;
    }
    let size = layout.size();
    let (lo, hi) = (Vec3::new(-0.5 * size.x, 0.0, 0.0), Vec3::new(0.5 * size.x, size.y, size.z));
    if (c.component_max(lo).component_min(hi) - c).norm() < r {
        return false;
    }
    placed.iter().all(|o| (Vec3::from_array(o.position) - c).norm() >= r + bounding_radius(o))
}

/// The speed at which the punch's force-to-capacity ratio on this impactor's predicted
/// contact is `d.punch_ratio`, from the punch's own law at a reference speed (force grows
/// as `v^1.2`: the impulse as `v`, the Hertz time as `v^-0.2`). `None` when the punch sees
/// no plug there (a grazing miss, or the cone takes the whole wall).
fn solve_speed(scene: &Scene, layout: &Layout, y_back: f64, d: &ImpactorDraw, r: &Ranges) -> Option<f64> {
    let v_ref = 20.0;
    let mut s = scene.clone();
    s.impactors = vec![place(d, layout, y_back, v_ref, 0)];
    // Only the geometry of the first contact matters: put it at the face now.
    s.impactors[0].position = (Vec3::from_array(s.impactors[0].position) + Vec3::from_array(s.impactors[0].velocity) * d.arrival).to_array();
    let w = World::new(&s);
    let c = w.predict_contacts(1.0 / 60.0, 0.0).into_iter().find(|c| c.plug.is_some())?;
    let p = c.plug.as_ref()?;
    let ratio = p.force / p.capacity;
    let speed = v_ref * (d.punch_ratio / ratio).powf(1.0 / 1.2);
    Some(speed.clamp(r.speed.0, r.speed.1))
}

/// A family of wall recipes (`p1-walls`): `family` is "free_wall" (no supports, no gravity:
/// the M5 free wall's family) or "anchored_wall" (base or base-and-sides supports, gravity
/// and a ground); `tier` 1 has one impactor, 2 has two to four.
pub fn wall_recipes(family: &str, tier: u32, n: usize, seed: u64, r: &Ranges) -> Result<Vec<OutcomeRecipe>, String> {
    if !matches!(family, "free_wall" | "anchored_wall") || !(1..=2).contains(&tier) {
        return Err(format!("unknown family/tier {family}/{tier}"));
    }
    let mut rng = Rng::new(seed ^ (tier as u64) << 32 ^ if family == "free_wall" { 0x11 } else { 0x22 });
    // Dimensions: nx, ny, nz, chunk, supports, strength, toughness, stiffness, mass, size,
    // x, z, angle, punch ratio, arrival.
    let lhs = latin_hypercube(n, 15, &mut rng);
    let mut out = Vec::with_capacity(n);
    for (i, u) in lhs.iter().enumerate() {
        let supports = if family == "free_wall" { "free" } else if u[4] < 0.7 { "base" } else { "base_sides" };
        let layout = Layout::sample(r, supports, u, &mut rng);
        let (strength, toughness, stiffness) = (log_uniform(u[5], 0.5, 2.0), log_uniform(u[6], 0.5, 2.0), log_uniform(u[7], 0.7, 1.5));
        let weibull = (rng.unit() < r.weibull_probability).then(|| rng.range(6.0, 12.0));
        let mut scene = builders::new_scene(&format!("p1_{family}_t{tier}"), "P1 procedural wall impact (learned impact-outcome dataset)");
        scene.materials.insert("concrete".into(), builders::oracle_concrete(weibull));
        scene.materials.insert("steel".into(), builders::elastic_steel());
        scale_materials(&mut scene, strength, toughness, stiffness);
        if family == "free_wall" {
            scene.gravity = [0.0; 3];
        } else {
            scene.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
        }
        scene.bodies.push(wall_body(&layout, "wall", 0.0));
        let y_back = layout.size().y;
        let loads = if tier == 1 { 1 } else { rng.int(2, 4) };
        let mut draws = Vec::new();
        let mut impactors = Vec::new();
        let mut speeds = Vec::new();
        let mut redraws = 0usize;
        for k in 0..loads {
            // A draw that starts inside the ground, the wall or another impactor is drawn
            // again (independently); the count is recorded.
            let mut attempt = 0usize;
            loop {
                // The first impactor takes the hypercube's draws; the others independent ones.
                let mut v: Vec<f64> = if k == 0 && attempt == 0 { u[8..].to_vec() } else { (0..7).map(|_| rng.unit()).collect() };
                if k == 0 {
                    // A redraw keeps the hypercube's speed draw: the band shares stay exact.
                    v[5] = u[13];
                }
                let angle = (v[4] * 2.0 - 1.0) * r.max_angle_degrees;
                let along_x = rng.unit() < 0.5;
                let spin = if rng.unit() < r.spin_probability {
                    let a = rng.int(0, 2);
                    let mut s = [0.0; 3];
                    s[a] = rng.range(-r.max_spin, r.max_spin);
                    s
                } else {
                    [0.0; 3]
                };
                let size = log_uniform(v[1], r.impactor_size.0, r.impactor_size.1).min(0.45 * layout.size().x.min(layout.size().z));
                let d = ImpactorDraw {
                    mass: log_uniform(v[0], r.mass.0, r.mass.1),
                    sphere: rng.unit() < 0.4,
                    size,
                    // 10% of hits graze an edge (centre up to 0.1 past it).
                    x_fraction: if rng.unit() < 0.1 { if rng.unit() < 0.5 { -0.05 } else { 1.05 } } else { 0.05 + 0.9 * v[2] },
                    z_fraction: 0.05 + 0.9 * v[3],
                    back: k > 0 && rng.unit() < 0.3,
                    yaw_degrees: if along_x { angle } else { 0.0 },
                    pitch_degrees: if along_x { 0.0 } else { angle },
                    spin,
                    punch_ratio: log_uniform(if k == 0 { ((v[5] - r.low_speed_share) / (1.0 - r.low_speed_share)).max(0.0) } else { v[5] }, r.punch_ratio.0, r.punch_ratio.1),
                    arrival: if k == 0 { 0.002 + v[6] * 0.03 } else { rng.range(0.002, 0.08) },
                    low: false,
                };
                // The hypercube's ratio dimension decides the band, so the share is exact.
                let low = k == 0 && v[5] < r.low_speed_share;
                let speed = if low {
                    log_uniform(v[5] / r.low_speed_share, r.low_speed.0, r.low_speed.1)
                } else {
                    match solve_speed(&scene, &layout, y_back, &d, r) {
                        Some(v) => v,
                        None => log_uniform(v[5], r.speed.0, r.speed.1),
                    }
                };
                let d = ImpactorDraw { low, ..d };
                let desc = place(&d, &layout, y_back, speed, k);
                if !starts_clear(&desc, &impactors, &layout, scene.ground.is_some()) {
                    attempt += 1;
                    redraws += 1;
                    if attempt > 200 {
                        return Err(format!("recipe {i}: no clear start for impactor {k} in 200 draws"));
                    }
                    continue;
                }
                impactors.push(desc);
                speeds.push(speed);
                draws.push(d);
                break;
            }
        }
        scene.impactors = impactors;
        let event_id = format!("p1_{family}_t{tier}_{i:05}_s{seed}");
        scene.name = event_id.clone();
        let params = json!({
            "layout": {"counts": layout.counts, "chunk": layout.chunk, "opening": layout.opening, "supports": layout.supports, "key": layout.key()},
            "material": {"strength_scale": strength, "toughness_scale": toughness, "stiffness_scale": stiffness, "weibull_modulus": weibull},
            "impactors": draws.iter().zip(&speeds).map(|(d, v)| json!({
                "mass": d.mass, "shape": if d.sphere { "sphere" } else { "box" }, "size": d.size, "speed": v,
                "x_fraction": d.x_fraction, "z_fraction": d.z_fraction, "face": if d.back { "back" } else { "front" },
                "location": d.location_tag(), "yaw_degrees": d.yaw_degrees, "pitch_degrees": d.pitch_degrees,
                "spin": d.spin, "target_punch_ratio": d.punch_ratio, "arrival": d.arrival,
                "speed_band": if d.low { "sub_threshold" } else { "punch_ratio" },
            })).collect::<Vec<_>>(),
            "seed": seed, "index": i, "catalog": "p1-walls", "redraws": redraws,
        });
        out.push(OutcomeRecipe { event_id: event_id.clone(), family: family.into(), tier, split_group: event_id, layout_group: layout.key(), parameters: params, scene });
    }
    Ok(out)
}

/// The M5 walls: `wall_row(n, 40, 0.1)` of findings/rewind-scaling.md
/// (`stress-rubble/tests/rewind_scenes`): `n` free concrete walls of 10 x 2 x 11 chunks of
/// 0.1 m, 0.1 m apart along the path of a 200 kg steel ram at 40 m/s, no gravity. `n = 1`
/// is the free wall, `n = 2` the wall row. Rebuilt here (stress-learned does not depend on
/// the engine crates); `tests/outcome.rs` checks the scene hashes against P0's.
pub fn m5_wall_row_n(n: usize) -> Scene {
    use stress_ref::builders::*;
    let mut s = new_scene(&format!("wall_row_{n}_v40"), "a steel ram through a row of free concrete walls");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    for k in 0..n {
        let y = k as f64 * (0.2 + 0.1);
        let chunks = grid(Vec3::new(-0.5, y, -0.1), [10, 2, 11], Vec3::splat(0.1), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        s.bodies.push(body(&format!("wall{k}"), chunks, bonds));
    }
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.1, 0.1, 0.1] },
        mass: 200.0,
        position: [0.0, -0.15, 0.5],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 40.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s
}

pub fn m5_free_wall() -> Scene {
    m5_wall_row_n(1)
}

pub fn m5_wall_row() -> Scene {
    m5_wall_row_n(2)
}

/// The M5 reference events (family `m5_reference`, held out of training by family).
pub fn m5_recipes() -> Vec<OutcomeRecipe> {
    [("m5_free_wall", m5_free_wall()), ("m5_wall_row", m5_wall_row())]
        .into_iter()
        .map(|(id, scene)| OutcomeRecipe {
            event_id: id.into(),
            family: "m5_reference".into(),
            tier: 1,
            split_group: id.into(),
            layout_group: id.into(),
            parameters: json!({"source": "stress_rubble::scenes (M5 truths: findings/rewind-scaling.md)"}),
            scene,
        })
        .collect()
}

fn fnv_str(text: &str, seed: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ seed;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// How a catalog is partitioned and which configurations get ensembles (the user's decision
/// after P0: acceptance is against oracle ensembles).
#[derive(Clone, Debug)]
pub struct EnsemblePlan {
    /// Train / validation / test shares, by layout group (whole layouts held out).
    pub fractions: [f64; 3],
    /// Families sent to the sealed set whole.
    pub sealed: Vec<String>,
    /// Share of training configurations recorded as ensembles; every validation, test and
    /// sealed configuration is.
    pub train_share: f64,
    pub members: usize,
    /// Perturbation of a member: impact point across the approach (uniform per axis, m) and
    /// speed (relative, uniform); each member also gets its own Weibull seed.
    pub offset: f64,
    pub speed: f64,
    pub seed: u64,
}

impl Default for EnsemblePlan {
    fn default() -> Self {
        EnsemblePlan { fractions: [0.8, 0.1, 0.1], sealed: vec!["m5_reference".into()], train_share: 0.15, members: 8, offset: 1e-3, speed: 1e-3, seed: 0 }
    }
}

/// The partition of a layout group: a seeded hash, so it never depends on recipe order.
pub fn split_of(layout_group: &str, family: &str, plan: &EnsemblePlan) -> &'static str {
    if plan.sealed.iter().any(|f| f == family) {
        return "sealed";
    }
    let u = (fnv_str(layout_group, plan.seed) >> 11) as f64 / (1u64 << 53) as f64;
    let t = plan.fractions.iter().sum::<f64>();
    if u < plan.fractions[0] / t {
        "train"
    } else if u < (plan.fractions[0] + plan.fractions[1]) / t {
        "val"
    } else {
        "test"
    }
}

/// Assign every recipe its split and expand the configurations chosen for ensembles into
/// `members` perturbed members (event `<id>_e<k>`, the parent's split group, an
/// `ensemble` record in `parameters`). Members replace the unperturbed recipe: P0 found the
/// unperturbed runs of aligned hits are knife-edge cases outside their own ensemble.
pub fn with_splits_and_ensembles(recipes: Vec<OutcomeRecipe>, plan: &EnsemblePlan) -> Vec<OutcomeRecipe> {
    let mut out = Vec::new();
    for mut r in recipes {
        let split = split_of(&r.layout_group, &r.family, plan);
        r.parameters["split"] = json!(split);
        let u = (fnv_str(&r.split_group, plan.seed ^ 0xe5e5) >> 11) as f64 / (1u64 << 53) as f64;
        let ensemble = plan.members > 1 && (split != "train" || u < plan.train_share);
        if !ensemble {
            r.parameters["ensemble"] = json!({"id": null, "member": null, "size": 1});
            out.push(r);
            continue;
        }
        let mut rng = Rng::new(fnv_str(&r.event_id, plan.seed ^ 0x0e0e));
        for k in 0..plan.members {
            let mut m = r.clone();
            m.event_id = format!("{}_e{k}", r.event_id);
            m.scene.name = m.event_id.clone();
            let weibull_seed = rng.next_u64();
            m.scene.sim.seed = weibull_seed;
            let mut perturbation = Vec::new();
            for imp in &mut m.scene.impactors {
                let v = Vec3::from_array(imp.velocity);
                let d = v.normalized();
                let a = if d.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
                let t1 = (a - d * a.dot(d)).normalized();
                let t2 = d.cross(t1);
                let off = t1 * rng.range(-plan.offset, plan.offset) + t2 * rng.range(-plan.offset, plan.offset);
                let scale = 1.0 + rng.range(-plan.speed, plan.speed);
                imp.position = (Vec3::from_array(imp.position) + off).to_array();
                imp.velocity = (v * scale).to_array();
                perturbation.push(json!({"offset_m": off.to_array(), "speed_scale": scale}));
            }
            m.parameters["ensemble"] = json!({"id": r.event_id, "member": k, "size": plan.members,
                "perturbation": {"impactors": perturbation, "weibull_seed": weibull_seed}});
            out.push(m);
        }
    }
    out
}
