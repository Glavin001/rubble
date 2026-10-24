//! Bonded vehicles: a car, a bus and an airliner built as free bonded islands
//! (`launch`), each part a chunk or a few, bonded where its faces touch with joints that
//! tear, and scenes that place them beside the Wrecking Yard's structures and launch them.
//!
//! * [`car`]: a 1.4 t saloon, 4.5 x 1.8 x 1.45 m: chassis (floor, sills, subframes and
//!   rails), engine bay (engine, firewall, bonnet, bolted wings), cabin (A-, B- and
//!   C-pillars, roof), four doors, boot, bumpers, four wheels and its glass. 50 chunks.
//! * [`bus`]: a 10.4 t single-deck bus, 12 x 2.5 x 3.2 m: underframes, axles, floor, side
//!   panels between window pillars, glass, roof, front and rear, four wheels. 87 chunks.
//! * [`airliner`]: a 35 t regional jet (Embraer E175 class), 30 m long, 26 m span: an
//!   octagonal fuselage barrel in upper and lower halves, nose and tail cones, wings in
//!   four pieces a side, under-wing engines on pylons, fin and tailplane. 36 chunks.
//!
//! Every part is a convex prism (its profile in one plane, extruded across it) or a loft
//! (`wrecking::hull_chunk`), so the vehicles can be placed at any heading. Bonds join every
//! pair of parts whose faces touch over an area (`wrecking::face_bonds`), of the joint
//! material [`joint`] picks: `glass` where a pane meets its frame; `riveted` in an
//! airframe; `bolted` where a door, wheel, bonnet, boot lid, bumper, bolted wing or engine
//! is attached; `spot-weld` elsewhere in a road vehicle's welded shell. The materials are
//! the calibrated ones (`wrecking::calibrated`, `material.rs`): metal shells smeared over
//! the chunks. Each part's mass is the vehicle's own (its curb mass shared out by part,
//! [`Vehicle::masses`]), written as the pack's node masses, which both lab profiles import.
//!
//! Frames: a vehicle is built in its own (Z-up, x forward, y to its left), a road vehicle's
//! wheels on z = 0, an aircraft on its fuselage axis; [`Placement`] turns it about z and
//! moves it into a scene.

use serde_json::Value;

use crate::launch::{Launch, Strike};
use crate::math::Vec3;
use crate::scene::{BondDesc, ChunkDesc, Support};
use crate::wrecking::{block, face_bonds, footing, hull_chunk, mass, to_pack, Structure};

/// The smeared body shell of a road vehicle (`Material::sheet_steel`).
pub const STEEL: &str = "sheet-steel";
/// A road wheel: rim and tyre, the shell's smeared material (its own name for its colour).
pub const WHEEL: &str = "wheel";
pub const GLASS: &str = "glass";
/// The smeared airframe (`Material::aluminium`).
pub const ALUMINIUM: &str = "aluminium";

/// Part kinds attached by bolts or hinges (`Material::bolted`).
pub const BOLTED: &[&str] = &["door", "wheel", "bonnet", "boot-lid", "bumper", "wing", "engine"];

/// The joint material between two parts of a vehicle (the module's notes).
pub fn joint(a: &ChunkDesc, b: &ChunkDesc) -> String {
    let kind = |c: &ChunkDesc| c.groups.first().cloned().unwrap_or_default();
    if a.material == GLASS || b.material == GLASS {
        "glass".into()
    } else if a.material == ALUMINIUM || b.material == ALUMINIUM {
        "riveted".into()
    } else if BOLTED.contains(&kind(a).as_str()) || BOLTED.contains(&kind(b).as_str()) {
        "bolted".into()
    } else {
        "spot-weld".into()
    }
}

/// A bonded vehicle in its own frame: its parts (`groups`: the part's kind, then the
/// vehicle's name), each part's mass (kg) and the spheres its strikes are predicted with
/// (`launch::Strike`: its nose, an aircraft's wings).
#[derive(Clone, Debug)]
pub struct Vehicle {
    pub name: String,
    pub chunks: Vec<ChunkDesc>,
    pub masses: Vec<f64>,
    pub strikes: Vec<Strike>,
}

impl Vehicle {
    fn new(name: &str) -> Vehicle {
        Vehicle { name: name.into(), chunks: Vec::new(), masses: Vec::new(), strikes: Vec::new() }
    }

    fn hull(&mut self, kind: &str, material: &str, points: &[Vec3], mass: f64) {
        self.chunks.push(hull_chunk(points, material, kind, self.name.clone()));
        self.masses.push(mass);
    }

    /// A prism: a convex profile in the xz plane (counter-clockwise or not), across `y`.
    fn side(&mut self, kind: &str, material: &str, profile: &[(f64, f64)], y: (f64, f64), mass: f64) {
        let points: Vec<Vec3> = [y.0, y.1].iter().flat_map(|&y| profile.iter().map(move |&(x, z)| Vec3::new(x, y, z))).collect();
        self.hull(kind, material, &points, mass);
    }

    /// [`Self::side`] on both sides: across `y` (y > 0, the left) and its mirror.
    fn sides(&mut self, kind: &str, material: &str, profile: &[(f64, f64)], y: (f64, f64), mass: f64) {
        self.side(kind, material, profile, y, mass);
        self.side(kind, material, profile, (-y.1, -y.0), mass);
    }

    /// A box spanning `x`, `y` and `z`.
    fn cuboid(&mut self, kind: &str, material: &str, x: (f64, f64), y: (f64, f64), z: (f64, f64), mass: f64) {
        self.side(kind, material, &rect(x, z), y, mass);
    }

    /// A loft along x: profile `a` (yz) at `xa` to profile `b` at `xb`.
    fn loft(&mut self, kind: &str, material: &str, xa: f64, a: &[(f64, f64)], xb: f64, b: &[(f64, f64)], mass: f64) {
        let points: Vec<Vec3> = a.iter().map(|&(y, z)| Vec3::new(xa, y, z)).chain(b.iter().map(|&(y, z)| Vec3::new(xb, y, z))).collect();
        self.hull(kind, material, &points, mass);
    }

    /// Total mass (kg).
    pub fn mass(&self) -> f64 {
        self.masses.iter().sum()
    }

    /// Its parts placed: turned `placement.heading` about z, then moved to `placement.at`;
    /// its strike spheres with them.
    pub fn placed(&self, placement: Placement) -> (Vec<ChunkDesc>, Vec<Strike>) {
        let (s, c) = placement.heading.sin_cos();
        let at = Vec3::from_array(placement.at);
        let turn = |p: Vec3| Vec3::new(c * p.x - s * p.y, s * p.x + c * p.y, p.z) + at;
        let chunks = self
            .chunks
            .iter()
            .map(|ch| {
                let centre = Vec3::from_array(ch.center);
                let points: Vec<Vec3> = ch.hull.as_ref().expect("vehicle parts are hulls").iter().map(|p| turn(centre + Vec3::from_array(*p))).collect();
                hull_chunk(&points, &ch.material, &ch.groups[0], ch.groups[1].clone())
            })
            .collect();
        let strikes = self.strikes.iter().map(|st| Strike { centre: turn(Vec3::from_array(st.centre)).to_array(), radius: st.radius }).collect();
        (chunks, strikes)
    }
}

/// Where a vehicle goes in a scene: turned `heading` (rad, about z) from its own frame's
/// +x, its own origin at `at` (Z-up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub at: [f64; 3],
    pub heading: f64,
}

/// A rectangle `x` by `z` (a profile).
fn rect(x: (f64, f64), z: (f64, f64)) -> Vec<(f64, f64)> {
    vec![(x.0, z.0), (x.1, z.0), (x.1, z.1), (x.0, z.1)]
}

/// A regular `n`-gon of circumradius `r` about `(u, v)`, its first vertex at `phase` rad.
fn ngon(u: f64, v: f64, r: f64, n: usize, phase: f64) -> Vec<(f64, f64)> {
    (0..n).map(|k| {
        let a = phase + std::f64::consts::TAU * k as f64 / n as f64;
        (u + r * a.cos(), v + r * a.sin())
    })
    .collect()
}

/// A road wheel's profile (xz): a 12-gon of radius `r` about `x`, flat on the ground.
fn wheel(x: f64, r: f64) -> Vec<(f64, f64)> {
    let phase = std::f64::consts::PI / 12.0;
    ngon(x, r * phase.cos(), r, 12, phase)
}

/// The car's dimensions (m).
pub struct CarLayout;

impl CarLayout {
    pub const LENGTH: f64 = 4.5;
    pub const WIDTH: f64 = 1.8;
    pub const HEIGHT: f64 = 1.45;
    /// Curb mass (kg).
    pub const MASS: f64 = 1400.0;
    pub const WHEELBASE: f64 = 2.7;
    pub const WHEEL: f64 = 0.32;
    /// Half the width between the wheels (the chassis), and inside the doors (the cabin).
    pub const CHASSIS: f64 = 0.68;
    pub const CABIN: f64 = 0.75;
}

/// A 1.4 t saloon car (the module's notes), its nose along +x, its wheels on z = 0, 50
/// parts. Its strike sphere is its front: 0.75 m round (1.5, 0, 0.6), reaching its bumper.
pub fn car(name: &str) -> Vehicle {
    type L = CarLayout;
    let mut v = Vehicle::new(name);
    let (w, cab, ch) = (0.5 * L::WIDTH, L::CABIN, L::CHASSIS);
    let ax = 0.5 * L::WHEELBASE;
    // Chassis: bumpers, rails, subframes, floor and sills.
    for s in [1.0, -1.0] {
        let x = |a: f64, b: f64| if s > 0.0 { (a, b) } else { (-b, -a) };
        v.cuboid("bumper", STEEL, x(2.05, 2.25), (-0.85, 0.85), (0.25, 0.55), 10.0);
        v.cuboid("rails", STEEL, x(1.67, 2.05), (-ch, ch), (0.20, 0.55), 30.0);
        v.cuboid("subframe", STEEL, x(1.0, 1.67), (-ch, ch), (0.15, 0.45), 60.0);
    }
    let floor = v.chunks.len();
    v.cuboid("floor", STEEL, (0.0, 1.0), (-ch, ch), (0.15, 0.40), 0.0);
    v.cuboid("floor", STEEL, (-1.0, 0.0), (-ch, ch), (0.15, 0.40), 0.0);
    v.sides("sill", STEEL, &rect((-1.03, 1.03), (0.15, 0.40)), (ch, w), 40.0);
    // Engine bay: engine on the subframe, firewall, A-posts, bolted wings, bonnet.
    v.cuboid("engine", STEEL, (1.15, 1.62), (-0.40, 0.40), (0.45, 0.85), 160.0);
    v.cuboid("firewall", STEEL, (1.0, 1.15), (-cab, cab), (0.45, 0.95), 25.0);
    v.sides("pillar", STEEL, &rect((1.0, 1.15), (0.40, 0.95)), (cab, w), 12.0);
    v.sides("wing", STEEL, &rect((1.15, 2.05), (0.64, 0.95)), (ch, w), 6.0);
    v.sides("wing", STEEL, &rect((1.67, 2.05), (0.25, 0.64)), (ch, w), 4.0);
    v.cuboid("bonnet", STEEL, (1.15, 2.05), (-ch, ch), (0.90, 0.95), 14.0);
    // Cabin: B-pillars, C-posts, rear bulkhead, the doors (on their pillars, clear of the
    // sills), pillars and glass above the waist, the roof.
    v.sides("pillar", STEEL, &rect((-0.05, 0.05), (0.40, 1.40)), (cab, w), 10.0);
    v.sides("pillar", STEEL, &rect((-1.15, -1.0), (0.40, 0.95)), (cab, w), 10.0);
    v.cuboid("bulkhead", STEEL, (-1.15, -1.0), (-cab, cab), (0.45, 0.95), 20.0);
    v.sides("door", STEEL, &rect((0.05, 1.0), (0.42, 0.95)), (cab, w), 25.0);
    v.sides("door", STEEL, &rect((-1.0, -0.05), (0.42, 0.95)), (cab, w), 22.0);
    v.side("glass", GLASS, &[(1.15, 0.95), (1.12, 0.95), (0.52, 1.40), (0.55, 1.40)], (-cab, cab), 14.0);
    v.sides("pillar", STEEL, &[(1.15, 0.95), (1.03, 0.95), (0.43, 1.40), (0.55, 1.40)], (cab, w), 6.0);
    v.sides("glass", GLASS, &[(0.05, 0.95), (1.03, 0.95), (0.43, 1.40), (0.05, 1.40)], (0.86, 0.88), 3.5);
    v.sides("glass", GLASS, &[(-0.05, 0.95), (-0.05, 1.40), (-0.75, 1.40), (-1.25, 0.95)], (0.86, 0.88), 3.5);
    v.sides("pillar", STEEL, &[(-0.75, 1.40), (-0.90, 1.40), (-1.40, 0.95), (-1.25, 0.95)], (cab, w), 6.0);
    v.side("glass", GLASS, &[(-0.75, 1.40), (-0.78, 1.40), (-1.28, 0.95), (-1.25, 0.95)], (-cab, cab), 9.0);
    v.cuboid("roof", STEEL, (-0.175, 0.55), (-w, w), (1.40, 1.45), 10.0);
    v.cuboid("roof", STEEL, (-0.90, -0.175), (-w, w), (1.40, 1.45), 10.0);
    // Boot: rear quarters (welded), parcel shelf, boot lid.
    v.sides("quarter", STEEL, &rect((-2.05, -1.15), (0.64, 0.95)), (ch, w), 8.0);
    v.sides("quarter", STEEL, &rect((-2.05, -1.67), (0.25, 0.64)), (ch, w), 5.0);
    v.cuboid("shelf", STEEL, (-1.30, -1.15), (-ch, ch), (0.90, 0.95), 6.0);
    v.cuboid("boot-lid", STEEL, (-2.05, -1.30), (-ch, ch), (0.90, 0.95), 12.0);
    // Wheels on the subframes.
    for x in [ax, -ax] {
        v.sides("wheel", WHEEL, &wheel(x, L::WHEEL), (ch, w), 20.0);
    }
    // The floor takes the rest of the curb mass (its seats, trim, fuel and fittings).
    let rest = L::MASS - v.mass();
    v.masses[floor] = 0.5 * rest;
    v.masses[floor + 1] = 0.5 * rest;
    v.strikes.push(Strike { centre: [1.5, 0.0, 0.6], radius: 0.75 });
    v
}

/// The bus's dimensions (m).
pub struct BusLayout;

impl BusLayout {
    pub const LENGTH: f64 = 12.0;
    pub const WIDTH: f64 = 2.5;
    pub const HEIGHT: f64 = 3.2;
    pub const WHEEL: f64 = 0.5;
    /// The axles' positions along it.
    pub const AXLES: [f64; 2] = [3.8, -2.6];
    /// The window bays' edges along it, rear to front (window pillars on the inner ones).
    pub const BAYS: [f64; 8] = [-6.0, -4.2, -2.4, -0.6, 1.2, 3.0, 4.8, 6.0];
    /// The side walls' inner face (half width).
    pub const INNER: f64 = 1.10;
}

/// A 10.4 t single-deck bus (the module's notes), its front along +x, its wheels on
/// z = 0, 87 parts. Its strike sphere is its front: 1.25 m round (4.95, 0, 1.6).
pub fn bus(name: &str) -> Vehicle {
    type L = BusLayout;
    let mut v = Vehicle::new(name);
    let (w, inner) = (0.5 * L::WIDTH, L::INNER);
    let (floor0, floor1, sill, head, eaves, top) = (1.0, 1.25, 1.75, 2.65, 3.05, 3.2);
    // Running gear: axles, wheels (bolted on), underframes (the engine in the rear one),
    // bumpers.
    for &x in &L::AXLES {
        v.cuboid("axle", STEEL, (x - 0.55, x + 0.55), (-0.94, 0.94), (0.05, floor0), 500.0);
        v.sides("wheel", WHEEL, &wheel(x, L::WHEEL), (0.94, 1.24), 120.0);
    }
    v.cuboid("underframe", STEEL, (L::AXLES[0] + 0.55, 6.0), (-w, w), (0.35, floor0), 500.0);
    v.cuboid("underframe", STEEL, (-6.0, L::AXLES[1] - 0.55), (-w, w), (0.35, floor0), 1800.0);
    v.cuboid("bumper", STEEL, (6.0, 6.2), (-w, w), (0.35, 0.9), 60.0);
    v.cuboid("bumper", STEEL, (-6.2, -6.0), (-w, w), (0.35, 0.9), 60.0);
    // Floor and roof by bay; side walls: panels under and over the windows, pillars
    // between them (and at the corners), glass.
    let bays = L::BAYS;
    for k in 0..bays.len() - 1 {
        let (a, b) = (bays[k], bays[k + 1]);
        v.cuboid("floor", STEEL, (a, b), (-w, w), (floor0, floor1), 400.0);
        v.cuboid("roof", STEEL, (a, b), (-w, w), (eaves, top), 100.0);
        v.sides("panel", STEEL, &rect((a, b), (floor1, sill)), (inner, w), 80.0);
        v.sides("panel", STEEL, &rect((a, b), (head, eaves)), (inner, w), 50.0);
        // The window between this bay's pillars (the corners' posts at the ends).
        let (g0, g1) = (if k == 0 { a + 0.15 } else { a + 0.08 }, if k == bays.len() - 2 { b - 0.15 } else { b - 0.08 });
        v.sides("glass", GLASS, &rect((g0, g1), (sill, head)), (1.165, 1.185), 30.0);
    }
    for (k, &e) in bays.iter().enumerate() {
        let x = if k == 0 { (e, e + 0.15) } else if k == bays.len() - 1 { (e - 0.15, e) } else { (e - 0.08, e + 0.08) };
        v.sides("pillar", STEEL, &rect(x, (sill, head)), (inner, w), 25.0);
    }
    // Front: the panel under the windscreen, the windscreen, the destination board; rear
    // wall in two.
    v.cuboid("front", STEEL, (5.85, 6.0), (-inner, inner), (floor1, sill), 80.0);
    v.side("glass", GLASS, &rect((5.95, 5.98), (sill, 2.85)), (-inner, inner), 60.0);
    v.cuboid("front", STEEL, (5.85, 6.0), (-inner, inner), (2.85, eaves), 40.0);
    v.cuboid("rear", STEEL, (-6.0, -5.85), (-inner, inner), (floor1, 2.15), 100.0);
    v.cuboid("rear", STEEL, (-6.0, -5.85), (-inner, inner), (2.15, eaves), 100.0);
    v.strikes.push(Strike { centre: [4.95, 0.0, 1.6], radius: 1.25 });
    v
}

/// The airliner's dimensions (m).
pub struct AirlinerLayout;

impl AirlinerLayout {
    pub const LENGTH: f64 = 30.0;
    pub const SPAN: f64 = 26.0;
    /// The fuselage's octagon: circumradius (3 m across its flats).
    pub const RADIUS: f64 = 1.5 / 0.923_879_532_511_286_7;
    /// The barrel's segments along x (the nose and tail cones beyond).
    pub const BARREL: [f64; 9] = [-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0];
    /// The wings' pieces along y (from the fuselage's side to the tip).
    pub const WING: [f64; 5] = [1.5, 4.4, 7.3, 10.2, 13.0];
    /// The engines' span station.
    pub const ENGINE: f64 = 5.8;
}

/// A 35 t regional jet (the module's notes), its nose along +x, its fuselage axis on z = 0,
/// 36 parts. Its strike spheres: its nose, and each wing at its root and its engine.
pub fn airliner(name: &str) -> Vehicle {
    type L = AirlinerLayout;
    let mut v = Vehicle::new(name);
    let oct = |r: f64, dz: f64| -> Vec<(f64, f64)> { ngon(0.0, dz, r, 8, std::f64::consts::PI / 8.0) };
    let r = L::RADIUS;
    let flat = r * (std::f64::consts::PI / 8.0).cos();
    let half = |upper: bool| -> Vec<(f64, f64)> {
        let s = if upper { 1.0 } else { -1.0 };
        let mut p = vec![(flat, 0.0)];
        p.extend(oct(r, 0.0).into_iter().filter(|q| q.1 * s > 0.0));
        p.push((-flat, 0.0));
        p
    };
    // The barrel in upper and lower halves (the floor beams at its axis); 18 t with the
    // cabin's payload.
    for k in 0..L::BARREL.len() - 1 {
        let (a, b) = (L::BARREL[k], L::BARREL[k + 1]);
        for upper in [true, false] {
            let p = half(upper);
            v.loft("fuselage", ALUMINIUM, a, &p, b, &p, 1125.0);
        }
    }
    // Nose and tail cones, the tail swept up.
    v.loft("nose", ALUMINIUM, 12.0, &oct(r, 0.0), 14.0, &oct(0.8, -0.2), 700.0);
    v.loft("nose", ALUMINIUM, 14.0, &oct(0.8, -0.2), 15.0, &oct(0.25, -0.3), 250.0);
    v.loft("tail", ALUMINIUM, -12.0, &oct(r, 0.0), -14.0, &oct(0.7, 0.6), 600.0);
    v.loft("tail", ALUMINIUM, -14.0, &oct(0.7, 0.6), -15.0, &oct(0.3, 0.9), 200.0);
    // Wings: low, on the lower barrel's sides; chord 4 m at the root to 1.5 m at the tip,
    // swept, 0.5 m deep at the root to 0.2 m; the fuel in the inner pieces.
    let span = L::WING;
    let (y0, y1) = (span[0], span[span.len() - 1]);
    let at = |y: f64| (y - y0) / (y1 - y0);
    let le = |y: f64| 2.0 - 3.5 * at(y);
    let te = |y: f64| -2.0 - 1.0 * at(y);
    let depth = |y: f64| 0.5 - 0.3 * at(y);
    let section = |y: f64| -> Vec<(f64, f64)> { vec![(te(y), -depth(y)), (le(y), -depth(y)), (le(y), 0.0), (te(y), 0.0)] };
    let wing_mass = [2200.0, 1600.0, 1000.0, 500.0];
    for s in [1.0, -1.0] {
        for k in 0..span.len() - 1 {
            let (ya, yb) = (span[k], span[k + 1]);
            let points: Vec<Vec3> = [(ya, section(ya)), (yb, section(yb))].iter().flat_map(|(y, p)| p.iter().map(move |&(x, z)| Vec3::new(x, s * y, z))).collect();
            v.hull("wing", ALUMINIUM, &points, wing_mass[k]);
        }
        // The engine on its pylon under the second piece, its top on the wing's underside.
        let (e, py) = (L::ENGINE, 0.25);
        let under = |y: f64| -depth(y);
        let (px0, px1) = (le(e) - 1.6, le(e) - 0.2);
        let low = under(e) - 0.6;
        let pylon: Vec<Vec3> = [e - py, e + py].iter().flat_map(|&y| [Vec3::new(px0, s * y, under(y)), Vec3::new(px1, s * y, under(y)), Vec3::new(px0, s * y, low), Vec3::new(px1, s * y, low)]).collect();
        v.hull("pylon", ALUMINIUM, &pylon, 200.0);
        let rn = 0.75;
        let cz = low - rn * (std::f64::consts::PI / 8.0).cos();
        let nacelle: Vec<(f64, f64)> = oct(rn, cz).into_iter().map(|(y, z)| (s * e + y, z)).collect();
        v.loft("engine", ALUMINIUM, le(e) + 1.0, &nacelle, le(e) - 2.2, &nacelle, 1500.0);
    }
    // Fin on the last barrel segment's top, in two; tailplane on its upper sides.
    let base = r * (std::f64::consts::PI / 8.0).cos();
    let fin = |z0: f64, z1: f64, x0: (f64, f64), x1: (f64, f64)| -> Vec<Vec3> { [-0.15, 0.15].iter().flat_map(|&y| [Vec3::new(x0.0, y, z0), Vec3::new(x0.1, y, z0), Vec3::new(x1.0, y, z1), Vec3::new(x1.1, y, z1)]).collect() };
    v.hull("fin", ALUMINIUM, &fin(base, base + 2.0, (-12.0, -9.2), (-13.25, -10.85)), 300.0);
    v.hull("fin", ALUMINIUM, &fin(base + 2.0, base + 4.0, (-13.25, -10.85), (-14.5, -13.0)), 150.0);
    for s in [1.0, -1.0] {
        let stab: Vec<Vec3> = [(base, (-12.0, -9.5)), (6.0, (-13.8, -12.8))].iter().flat_map(|&(y, (x0, x1))| [Vec3::new(x0, s * y, 0.05), Vec3::new(x1, s * y, 0.05), Vec3::new(x0, s * y, 0.30), Vec3::new(x1, s * y, 0.30)]).collect();
        v.hull("tailplane", ALUMINIUM, &stab, 250.0);
    }
    v.strikes.push(Strike { centre: [13.4, 0.0, -0.1], radius: 1.6 });
    for s in [1.0, -1.0] {
        v.strikes.push(Strike { centre: [0.8, s * 3.0, -0.25], radius: 1.4 });
        v.strikes.push(Strike { centre: [le(L::ENGINE) - 0.6, s * L::ENGINE, -1.4], radius: 1.0 });
    }
    v
}

/// A scene: a structure with vehicles in it (its chunks the structure's, then any supports
/// added for them, then each vehicle's), each chunk's mass, and the launches.
pub struct Scene {
    pub structure: Structure,
    /// Per chunk (kg; 0 for a support): the structure's at its calibrated density, the
    /// vehicles' their own.
    pub masses: Vec<f64>,
    pub launches: Vec<Launch>,
    /// Each vehicle's name and its chunks (ascending).
    pub vehicles: Vec<(String, Vec<usize>)>,
    /// Raycast vehicles driven at the structure (`drive`): not chunks of it.
    pub drives: Vec<crate::drive::Drive>,
}

impl Scene {
    /// Its scene pack: the structure's (`wrecking::to_pack`) with the vehicles' masses as
    /// the node masses and the launches (`launch::write`).
    pub fn to_pack(&self) -> Value {
        let mut pack = to_pack(&self.structure);
        for (node, &m) in pack["scenario"]["nodes"].as_array_mut().expect("nodes").iter_mut().zip(&self.masses) {
            node["mass"] = m.into();
        }
        crate::launch::write(&mut pack, &self.launches);
        crate::drive::write(&mut pack, &self.drives);
        pack
    }
}

/// How a placed vehicle is launched (Z-up, m/s, rad/s, s), or None: it stands where it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub delay: f64,
}

/// `base` with `supports` added (static, bonded to nothing) and `vehicles` placed in it,
/// each bonded within itself only, and launched where it has a [`Motion`] (its launch
/// named as the vehicle is).
pub fn compose(key: &'static str, title: &'static str, base: Structure, supports: Vec<ChunkDesc>, vehicles: &[(Vehicle, Placement, Option<Motion>)]) -> Scene {
    let mut s = base;
    s.key = key;
    s.title = title;
    let mut masses: Vec<f64> = s.chunks.iter().map(mass).collect();
    for c in supports {
        debug_assert_eq!(c.support, Support::Fixed);
        s.chunks.push(c);
        masses.push(0.0);
    }
    let mut launches = Vec::new();
    let mut placed = Vec::new();
    for (v, at, motion) in vehicles {
        let (chunks, strikes) = v.placed(*at);
        let first = s.chunks.len();
        let bonds: Vec<BondDesc> = face_bonds(&chunks, joint).into_iter().map(|mut b| {
            b.a += first;
            b.b += first;
            b
        })
        .collect();
        s.chunks.extend(chunks);
        s.bonds.extend(bonds);
        masses.extend(&v.masses);
        let ids: Vec<usize> = (first..s.chunks.len()).collect();
        if let Some(m) = motion {
            launches.push(Launch { name: v.name.clone(), chunks: ids.clone(), velocity: m.velocity, angular_velocity: m.angular_velocity, delay: m.delay, strikes });
        }
        placed.push((v.name.clone(), ids));
    }
    Scene { structure: s, masses, launches, vehicles: placed, drives: Vec::new() }
}

/// A launch along `heading` at `speed` (m/s), with `lift` (m/s) up.
pub fn along(heading: f64, speed: f64, lift: f64, delay: f64) -> Motion {
    Motion { velocity: [speed * heading.cos(), speed * heading.sin(), lift], angular_velocity: [0.0; 3], delay }
}

/// The sliding speed a road vehicle launched on its wheels needs to arrive at `speed`
/// after `distance` m: its wheels are rigid parts of it (they do not roll), so it slides
/// on them, braked by the ground's friction (rubble's ground, 0.6: a locked-wheel stop on
/// dry asphalt is 0.6-0.8 g).
pub fn sliding_launch(speed: f64, distance: f64) -> f64 {
    (speed * speed + 2.0 * GROUND_FRICTION * 9.81 * distance).sqrt()
}

/// rubble's ground friction (`scene_pack`'s `GroundDesc`, the same in the engine).
pub const GROUND_FRICTION: f64 = 0.6;

/// A rigid crash barrier: a concrete block (a support) `size` m, its face toward -x at
/// `x`, on the ground, centred on y = 0.
fn barrier(x: f64, size: [f64; 3]) -> ChunkDesc {
    footing(block([x, -0.5 * size[1], 0.0], [x + size[0], 0.5 * size[1], size[2]], "footing-anchor", "barrier", "barrier".into()))
}

/// The structure of a crash test: no building, the barrier only.
fn crash_site(key: &'static str, title: &'static str) -> Structure {
    Structure {
        key,
        title,
        chunks: Vec::new(),
        bonds: Vec::new(),
        aim: crate::wrecking::Aim { target: [0.0, 0.0, 1.0], direction: [1.0, 0.0, 0.0] },
        ball: crate::wrecking::BALLS[0],
        charges: Vec::new(),
    }
}

/// Clear road between a vehicle's nose and what it hits when it is launched (m).
pub const RUN_UP: f64 = 12.0;

/// The speed vehicles hit at in the crash tests: 56 km/h, Euro NCAP's (m/s).
pub const CRASH_SPEED: f64 = 15.6;

/// A street (a support, bonded to nothing) over `x` and `y` (Z-up), `top` m high: the
/// road a vehicle drives on up to a building whose ground floor stands on footings above
/// grade. One slab, so a sliding wheel meets no seam.
fn street(x: (f64, f64), y: (f64, f64), top: f64) -> Vec<ChunkDesc> {
    vec![footing(block([x.0, y.0, 0.0], [x.1, y.1, top], "footing-anchor", "street", "street".into()))]
}

/// The speed the cars and buses hit buildings at: 54 km/h (m/s).
pub const RAM_SPEED: f64 = 15.0;

/// Road vehicles driven at a building's street front (+y, the facade at `face`), on a
/// street level with its footings' tops (`top`), each nose `RUN_UP` m out: (vehicle, the x
/// it is aimed at, its nose's distance from its origin).
fn ram_raid(key: &'static str, title: &'static str, base: Structure, face: f64, top: f64, street_x: (f64, f64), near: f64, vehicles: Vec<(Vehicle, f64, f64)>) -> Scene {
    let heading = std::f64::consts::FRAC_PI_2;
    let speed = sliding_launch(RAM_SPEED, RUN_UP);
    let placed: Vec<(Vehicle, Placement, Option<Motion>)> = vehicles.into_iter().map(|(v, x, nose)| (v, Placement { at: [x, face - RUN_UP - nose, top], heading }, Some(along(heading, speed, 0.0, 0.0)))).collect();
    compose(key, title, base, street(street_x, (near - 24.0, near), top), &placed)
}

/// The airliner's speed into the Fly-through buildings: 70 m/s (252 km/h, about an E175's
/// approach speed).
pub const AIRLINER_SPEED: f64 = 70.0;

/// How far the airliner's nose starts from the facade (m): 0.57 s of flight.
pub const AIRLINER_STANDOFF: f64 = 40.0;

/// The airliner thrown at a structure along its line of attack (`Structure::aim`, +y), its
/// fuselage axis at the aim's height, its nose `AIRLINER_STANDOFF` m out: launched at
/// `AIRLINER_SPEED` with the lift that brings it to the facade level under gravity (a
/// thrown paper aeroplane: nothing holds it up), released at once.
fn fly_through(key: &'static str, title: &'static str, base: Structure, offset: f64) -> Scene {
    let mut aim = base.aim;
    aim.target[0] += offset;
    let heading = aim.direction[1].atan2(aim.direction[0]);
    let nose = 0.5 * AirlinerLayout::LENGTH;
    let back = AIRLINER_STANDOFF + nose;
    let at = [aim.target[0] - back * heading.cos(), aim.target[1] - back * heading.sin(), aim.target[2]];
    let flight = AIRLINER_STANDOFF / AIRLINER_SPEED;
    let lift = 0.5 * 9.81 * flight;
    compose(key, title, base, Vec::new(), &[(airliner("airliner"), Placement { at, heading }, Some(along(heading, AIRLINER_SPEED, lift, 0.0)))])
}

/// The scenes with vehicles: (key, title), in menu order.
pub const SCENES: &[(&str, &str)] = &[
    ("vh_car_barrier", "Car into a crash barrier"),
    ("vh_bus_barrier", "Bus into a crash barrier"),
    ("rr_vehicles_storefront", "Car and bus through the storefront row"),
    ("rr_vehicles_apartment", "Car and bus through the apartment block"),
    ("ft_airliner_tower", "Airliner into the glass tower"),
    ("ft_airliner_apartment", "Airliner into the apartment block"),
    ("bo_viaduct_traffic", "Highway viaduct with stalled traffic"),
    ("rr_drive_storefront", "Sedan and truck driven through the storefront row"),
    ("bo_drive_viaduct", "Highway viaduct with traffic driving across"),
];

/// How far out a driven vehicle starts (m from its nose to the facade): room for its
/// wheels to bring it up to speed (the sedan reaches about 50 km/h at the facade).
pub const DRIVE_RUN_UP: f64 = 30.0;

/// Traffic driving on the viaduct (S2 + V3 + V5, the Bridge Out beat with real vehicles):
/// rubble's raycast presets cruising both ways on the deck (lanes 2 m either side of its
/// centre line), sedans and a six-wheel truck each way on the drop-in spans and the end
/// spans, and a sedan and a truck on the lower road crossing under span 2 (along it). The
/// hinge pier's charge (`hinge_charge`) drops the drop-in spans under them.
fn viaduct_drives() -> Vec<crate::drive::Drive> {
    use crate::viaduct::ViaductLayout as L;
    let (deck, road) = (L::DECK_TOP, L::ROAD_BASE + L::ROAD_DEPTH);
    let drive = |name: &str, vehicle: &str, at: [f64; 3], forward: [f64; 3], speed: f64| crate::drive::Drive {
        name: name.into(),
        vehicle: vehicle.into(),
        at,
        forward,
        throttle: 0.6,
        delay: 0.0,
        speed: Some(speed),
    };
    let (east, west, north, south) = ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]);
    // Slow traffic (3-3.5 m/s): nothing reaches an abutment's end in 9 s.
    vec![
        // Eastbound (+x, lane y = -2): span 1, then span 2 (the drop-in west of the hinge).
        drive("east-1", "sedan", [-28.0, -2.0, deck], east, 3.5),
        drive("east-truck", "six_wheel_truck", [-16.0, -2.0, deck], east, 3.0),
        drive("east-2", "sedan", [-8.0, -2.0, deck], east, 3.5),
        // Westbound (-x, lane y = +2): span 4, then span 3.
        drive("west-1", "sedan", [28.0, 2.0, deck], west, 3.5),
        drive("west-truck", "six_wheel_truck", [16.0, 2.0, deck], west, 3.0),
        drive("west-2", "sedan", [8.0, 2.0, deck], west, 3.5),
        // The lower road under span 2 (x -11.5..-2.5, along y): under the deck by 3 s.
        drive("road-1", "sedan", [-9.0, -12.0, road], north, 3.0),
        drive("road-truck", "six_wheel_truck", [-5.0, 11.0, road], south, 2.5),
    ]
}

/// The Bridge Out charge for a world API host (the threshold model): the hinge pier
/// blown (every bond of its chunks) and the drop-in spans cut from their seats (every bond
/// of spans 2 and 3 to anything else). Under `blast_parity` the ball does this: the pier
/// goes and the spans' own weight breaks their links over the seat piers. The threshold
/// model breaks a bond only under a contact's load, so a span bonded at one end hangs
/// there rigidly, however it is loaded; the charge cuts what the stress solver would
/// break. The spans then rest on their seats' bearings and swing down about them.
fn hinge_charge(s: &crate::wrecking::Structure) -> crate::wrecking::Charge {
    use crate::viaduct::ViaductLayout as L;
    let group = |k: usize| s.chunks[k].groups.get(1).map(String::as_str).unwrap_or("");
    let pier = format!("pier-{}", L::HINGE_PIER);
    let drop_in = |g: &str| g == "span-2" || g == "span-3";
    let bonds = s
        .bonds
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            let (ga, gb) = (group(b.a), group(b.b));
            ga == pier || gb == pier || (drop_in(ga) || drop_in(gb)) && ga != gb
        })
        .map(|(i, _)| i)
        .collect();
    crate::wrecking::Charge { name: "hinge pier".into(), delay: 0.0, bonds, debris: Vec::new(), kick: [0.0; 3] }
}

/// Stalled traffic on the viaduct (S2 + V3 + V4), none launched: cars and a bus parked on
/// the two drop-in spans either side of the hinge pier (in lanes 2 m either side of the
/// deck's centre line, on the deck's top), and cars and a bus on the lower road crossing
/// under span 2 (heading along it, on its slabs). The ball takes the hinge pier out and
/// the spans carry them down onto the road.
fn traffic() -> Vec<(Vehicle, Placement, Option<Motion>)> {
    use crate::viaduct::ViaductLayout as L;
    use std::f64::consts::{FRAC_PI_2, PI};
    let deck = L::DECK_TOP;
    let road = L::ROAD_BASE + L::ROAD_DEPTH;
    let at = |x: f64, y: f64, z: f64, heading: f64| Placement { at: [x, y, z], heading };
    vec![
        // Span 2 (west of the hinge, x -15.5..-1): three cars.
        (car("car-1"), at(-12.0, -2.0, deck, 0.0), None),
        (car("car-2"), at(-5.5, -2.0, deck, 0.0), None),
        (car("car-3"), at(-9.0, 2.0, deck, PI), None),
        // Span 3 (east of it, x 1..15.5): a bus and two cars.
        (bus("bus-1"), at(8.0, -2.0, deck, 0.0), None),
        (car("car-4"), at(4.5, 2.0, deck, PI), None),
        (car("car-5"), at(11.5, 2.0, deck, PI), None),
        // The lower road under span 2 (x -11.5..-2.5, along y): two cars and a bus queued.
        (car("car-6"), at(-9.0, -6.0, road, FRAC_PI_2), None),
        (car("car-7"), at(-5.0, 1.0, road, -FRAC_PI_2), None),
        (bus("bus-2"), at(-9.0, 4.5, road, FRAC_PI_2), None),
    ]
}

/// A scene with vehicles by key.
pub fn scene(key: &str) -> Option<Scene> {
    let &(key, title) = SCENES.iter().find(|s| s.0 == key)?;
    let speed = |run: f64| sliding_launch(CRASH_SPEED, run);
    match key {
        "vh_car_barrier" => {
            let nose = 0.5 * CarLayout::LENGTH;
            Some(compose(key, title, crash_site(key, title), vec![barrier(0.0, [1.0, 3.0, 1.5])], &[(car("car"), Placement { at: [-RUN_UP - nose, 0.0, 0.0], heading: 0.0 }, Some(along(0.0, speed(RUN_UP), 0.0, 0.0)))]))
        }
        "vh_bus_barrier" => {
            let nose = 0.5 * BusLayout::LENGTH + 0.2;
            Some(compose(key, title, crash_site(key, title), vec![barrier(0.0, [1.5, 4.0, 3.0])], &[(bus("bus"), Placement { at: [-RUN_UP - nose, 0.0, 0.0], heading: 0.0 }, Some(along(0.0, speed(RUN_UP), 0.0, 0.0)))]))
        }
        // V5: the car at the middle shop's window, the bus at the fourth shop's, from a
        // street level with the footings (the shop fronts stand on 0.5 m strip footings).
        "rr_vehicles_storefront" => {
            use crate::buildings::StorefrontLayout as L;
            let window = |i: usize| {
                let (_, w) = L::shopfront(i);
                0.5 * (w.0 + w.1)
            };
            let x = L::half_length();
            Some(ram_raid(key, title, crate::buildings::storefront_row(), -0.5 * L::DEPTH, L::FOOTING_DEPTH, (-x, x), -0.5 * L::DEPTH, vec![(car("car"), window(2), 0.5 * CarLayout::LENGTH), (bus("bus"), window(4), 0.5 * BusLayout::LENGTH + 0.2)]))
        }
        // V5: the car into front bay 0's ground floor (its brick sill and window), the bus
        // into bay 3's (sill, jambs, window, the first floor's beam over them). (At the
        // glazed entrance the car stops dead on the pane: a pane's bond at its sill holds
        // 1.6 MN.) The street stops short of the pad footings (1 m) on the front grid line.
        "rr_vehicles_apartment" => {
            use crate::buildings::ApartmentLayout as L;
            let bay = |i: usize| 0.5 * (L::line_x(i) + L::line_x(i + 1));
            let x = L::line_x(L::BAYS_X) + 0.5 * L::FOOTING;
            let face = L::line_y(0) - 0.5 * L::INFILL;
            Some(ram_raid(key, title, crate::buildings::apartment_block(), face, L::FOOTING_DEPTH, (-x, x), L::line_y(0) - 0.5 * L::FOOTING, vec![(car("car"), bay(0), 0.5 * CarLayout::LENGTH), (bus("bus"), bay(3), 0.5 * BusLayout::LENGTH + 0.2)]))
        }
        // S3: the airliner into the glass tower's sixth storey and the apartment block's
        // fourth, along their lines of attack.
        "ft_airliner_tower" => Some(fly_through(key, title, crate::buildings::glass_tower(), 0.0)),
        "ft_airliner_apartment" => Some(fly_through(key, title, crate::buildings::apartment_block(), 0.0)),
        "bo_viaduct_traffic" => Some(compose(key, title, crate::viaduct::viaduct(), Vec::new(), &traffic())),
        // Bridge Out driven: the viaduct with its traffic as raycast vehicles (a world API
        // host creates them, `drive`), the ball the lab's.
        "bo_drive_viaduct" => {
            let mut viaduct = crate::viaduct::viaduct();
            viaduct.charges = vec![hinge_charge(&viaduct)];
            let mut s = compose(key, title, viaduct, Vec::new(), &[]);
            s.drives = viaduct_drives();
            Some(s)
        }
        // V5 driven: the sedan (rubble's raycast preset, its wheels driving it) at the
        // middle shop's window from 30 m out, the six-wheel truck at the fourth shop's a
        // second later, both at full throttle up the street (+y) and level with the
        // footings' tops. Not chunks: a world API host creates them (`drive`).
        "rr_drive_storefront" => {
            use crate::buildings::StorefrontLayout as L;
            let window = |i: usize| {
                let (_, w) = L::shopfront(i);
                0.5 * (w.0 + w.1)
            };
            let x = L::half_length();
            let face = -0.5 * L::DEPTH;
            let mut s = compose(key, title, crate::buildings::storefront_row(), street((-x, x), (face - 48.0, face), L::FOOTING_DEPTH), &[]);
            // (name, preset, its nose ahead of its centre of mass, its delay; on the street.)
            let drive = |name: &str, vehicle: &str, x: f64, nose: f64, delay: f64| crate::drive::Drive {
                name: name.into(),
                vehicle: vehicle.into(),
                at: [x, face - DRIVE_RUN_UP - nose, L::FOOTING_DEPTH],
                forward: [0.0, 1.0, 0.0],
                throttle: 1.0,
                delay,
                speed: None,
            };
            s.drives = vec![drive("sedan", "sedan", window(2), 2.2, 0.0), drive("truck", "six_wheel_truck", window(4), 3.4, 1.0)];
            Some(s)
        }
        _ => None,
    }
}

/// A scene's pack by key.
pub fn pack(key: &str) -> Option<Value> {
    scene(key).map(|s| s.to_pack())
}
