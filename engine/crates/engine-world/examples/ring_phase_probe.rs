//! Diagnosis (branch diag/ring-phase-through): why the brick ring drive's soft sedan and
//! its bricks occupy the same space, why it drives slowly, and how much it jitters.
//! Measurement only: nothing here changes a solver default.
//!
//! The car is the page's `?car=sedan`: the authored showcase sedan (style `sedan`,
//! resolution low) on raycast wheels, with the page's settings (fast inverse square root,
//! drag and turbulence unless PP_DRAG=0).
//!
//! Scenes (PP_SCENE):
//! * `cpu-drive` (CPU reference, no GPU): flat ground, full throttle (or PP_KMH cruise) for
//!   PP_SECS; speed against time, the drag force against ½ρC_dAv², and the jitter at
//!   steady speed (per 60 Hz frame: the centroid's vertical position about its 0.5 s mean,
//!   its acceleration, and the node velocities about the centroid's);
//! * `cpu-post` (CPU reference): a brick (1 × 0.5 × 0.5 m) as a static shape, turned 45°
//!   about y so a vertical edge meets the car's nose between nodes, at PP_Y (m above the
//!   ground, default 0.55), the car on cruise at PP_KMH then pushing on full throttle;
//! * `wall` (GPU World, under the lock): a short wall of free 25 kg bricks (PP_COLS ×
//!   PP_ROWS × PP_DEEP, running bond) PP_RUNUP m ahead of the car; cruise at PP_KMH, then
//!   (from the first coupling impulse) full throttle, as a player holding W;
//! * `loose` (GPU World): PP_COLS free bricks lying on the road ahead, the car on cruise
//!   over them (the raycast wheels' rays do not see rigid bodies);
//! * `flat` (GPU World): no bricks; the jitter numbers as `cpu-drive`'s, on the GPU;
//! * `rigid` (GPU World, the control): no car; a 1,300 kg rigid box of the car's size,
//!   its velocity set to PP_KMH every step, pushed into the same wall.
//!
//! PP_SURFACE=1 turns on the opt-in surface contact (triangles against collider shapes).
//!
//! Penetration measures (per step, max over the step's state):
//! * `node`: car nodes inside a brick (depth, m);
//! * `skin`: points sampled on the car's collision triangles (7 × 7 grid) inside a brick;
//! * `hull`: points sampled on the bricks (corners, edges, faces) inside the car's closed
//!   collision surface (generalised winding number > 0.5), depth = distance to it;
//! * `wheel`: brick points inside the raycast wheels' cylinders (hub midpoint, radius,
//!   width).
//!
//! GPU runs: `engine/scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0
//! target/release/examples/ring_phase_probe`.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticShape, DRAG};
use engine_softbody::explicit_drive::DriveInput;
use glam::{Quat, Vec3};
use rubble::brick_ring_drive::{ring_coupling, ring_world_def};
use rubble::explicit_scene::place_vehicle;
use rubble::*;

fn env_f(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

const GROUND: f32 = 0.5;
const HALF: Vec3 = Vec3::new(0.5, 0.25, 0.25);

#[derive(Clone, Copy)]
struct Obb {
    c: Vec3,
    q: Quat,
    h: Vec3,
}

impl Obb {
    /// Depth of `p` inside (positive), or minus a lower bound of its distance outside.
    fn depth(&self, p: Vec3) -> f32 {
        let l = self.q.inverse() * (p - self.c);
        (self.h - l.abs()).min_element()
    }
    fn points(&self, n: usize) -> Vec<Vec3> {
        // A grid of n × n × n on the surface (every point with some |coordinate| = 1).
        let mut out = Vec::new();
        let g = |i: usize| -1.0 + 2.0 * i as f32 / (n - 1) as f32;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let v = Vec3::new(g(i), g(j), g(k));
                    if v.abs().max_element() >= 0.999 {
                        out.push(self.c + self.q * (v * self.h));
                    }
                }
            }
        }
        out
    }
    fn bound(&self) -> (Vec3, Vec3) {
        let m = glam::Mat3::from_quat(self.q);
        let e = m.x_axis.abs() * self.h.x + m.y_axis.abs() * self.h.y + m.z_axis.abs() * self.h.z;
        (self.c - e, self.c + e)
    }
}

fn closest_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    // Ericson, Real-Time Collision Detection 5.1.5.
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let den = 1.0 / (va + vb + vc);
    a + ab * (vb * den) + ac * (vc * den)
}

/// Generalised winding number of the triangle soup about `p` (Van Oosterom–Strackee).
fn winding(p: Vec3, tris: &[[Vec3; 3]]) -> f32 {
    let mut s = 0.0f64;
    for t in tris {
        let (a, b, c) = (
            (t[0] - p).as_dvec3(),
            (t[1] - p).as_dvec3(),
            (t[2] - p).as_dvec3(),
        );
        let (la, lb, lc) = (a.length(), b.length(), c.length());
        let num = a.dot(b.cross(c));
        let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
        s += 2.0 * num.atan2(den);
    }
    (s / (4.0 * std::f64::consts::PI)) as f32
}

/// The car's surface: its collision triangles at `pos`, and their closed-ness.
struct CarSurface {
    tris: Vec<[u32; 3]>,
    wheels: Vec<(u32, u32, f32, f32)>,
}

impl CarSurface {
    fn of(s: &ExplicitBeams) -> Self {
        let tris: Vec<[u32; 3]> = s.triangles.iter().map(|t| t.nodes).collect();
        let mut edges: std::collections::HashMap<(u32, u32), u32> = Default::default();
        for t in &tris {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let open = edges.values().filter(|&&c| c == 1).count();
        let over = edges.values().filter(|&&c| c > 2).count();
        println!(
            "car surface: {} triangles, {} edges, {open} boundary edges, {over} edges in > 2 triangles",
            tris.len(),
            edges.len()
        );
        let wheels = s
            .drive
            .as_ref()
            .map(|d| {
                d.config
                    .wheels
                    .iter()
                    .map(|w| (w.node1, w.node2, w.radius, w.width))
                    .collect()
            })
            .unwrap_or_default();
        Self { tris, wheels }
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct Pen {
    node: f32,
    skin: f32,
    hull: f32,
    wheel: f32,
    /// Bricks with a hull point deeper than 5 cm.
    hull_bricks: u32,
}

impl Pen {
    fn max(&mut self, o: Pen) {
        self.node = self.node.max(o.node);
        self.skin = self.skin.max(o.skin);
        self.hull = self.hull.max(o.hull);
        self.wheel = self.wheel.max(o.wheel);
        self.hull_bricks = self.hull_bricks.max(o.hull_bricks);
    }
}

fn measure(surf: &CarSurface, pos: &[Vec3], bricks: &[Obb]) -> Pen {
    let (lo, hi) = pos
        .iter()
        .fold((Vec3::MAX, Vec3::MIN), |(a, b), p| (a.min(*p), b.max(*p)));
    let near: Vec<Obb> = bricks
        .iter()
        .copied()
        .filter(|b| {
            let (bl, bh) = b.bound();
            bl.cmple(hi + 0.3).all() && bh.cmpge(lo - 0.3).all()
        })
        .collect();
    let mut m = Pen::default();
    if near.is_empty() {
        return m;
    }
    for p in pos {
        for b in &near {
            m.node = m.node.max(b.depth(*p));
        }
    }
    let tris: Vec<[Vec3; 3]> = surf
        .tris
        .iter()
        .map(|t| [pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]])
        .collect();
    const G: usize = 7;
    for t in &tris {
        for i in 0..G {
            for j in 0..G - i {
                let (u, v) = (i as f32 / (G - 1) as f32, j as f32 / (G - 1) as f32);
                let p = t[0] + (t[1] - t[0]) * u + (t[2] - t[0]) * v;
                for b in &near {
                    m.skin = m.skin.max(b.depth(p));
                }
            }
        }
    }
    for b in &near {
        let mut deepest = 0.0f32;
        for p in b.points(5) {
            if p.cmplt(lo).any() || p.cmpgt(hi).any() {
                continue;
            }
            if winding(p, &tris) > 0.5 {
                let d = tris
                    .iter()
                    .map(|t| closest_on_tri(p, t[0], t[1], t[2]).distance(p))
                    .fold(f32::MAX, f32::min);
                deepest = deepest.max(d);
            }
        }
        m.hull = m.hull.max(deepest);
        if deepest > 0.05 {
            m.hull_bricks += 1;
        }
        for &(n1, n2, r, wdt) in &surf.wheels {
            let (a, c) = (pos[n1 as usize], pos[n2 as usize]);
            let centre = 0.5 * (a + c);
            let axle = (c - a).try_normalize().unwrap_or(Vec3::X);
            for p in b.points(5) {
                let d = p - centre;
                let along = d.dot(axle);
                let radial = (d - axle * along).length();
                let depth = (r - radial).min(0.5 * wdt - along.abs());
                m.wheel = m.wheel.max(depth);
            }
        }
    }
    m
}

fn sedan(drag: bool) -> anyhow::Result<ExplicitBeams> {
    use engine_softbody::authoring::{build, style, BuildParams, Resolution};
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag,
        turbulence: drag,
        ..Default::default()
    };
    let def = style("sedan").ok_or_else(|| anyhow::anyhow!("no sedan style"))?;
    let b = build(
        &def,
        &BuildParams {
            resolution: Resolution::Low,
            ..BuildParams::default()
        },
    )
    .map_err(anyhow::Error::msg)?;
    let (mut s, notes) =
        ExplicitBeams::from_truck_raycast(&b.truck, None, Vec3::ZERO, 0.0, Vec3::ZERO, settings)
            .map_err(anyhow::Error::msg)?;
    s.surface_contact = std::env::var("PP_SURFACE").as_deref() == Ok("1");
    for key in [
        "rubble_beamng_profile",
        "rubble_drag_coef",
        "rubble_aero_tri",
    ] {
        let line = b
            .truck
            .lines()
            .position(|l| l.trim_start().starts_with(key));
        let rows = line.map(|i| {
            b.truck
                .lines()
                .skip(i + 1)
                .take_while(|l| !l.trim().is_empty())
                .filter(|l| !l.trim_start().starts_with(';'))
                .count()
        });
        println!(
            "truck: {key}: {}",
            match line {
                Some(i) => format!(
                    "line {i}: {:?}, {} rows after",
                    b.truck.lines().nth(i).unwrap_or(""),
                    rows.unwrap_or(0)
                ),
                None => "absent".into(),
            }
        );
    }
    println!(
        "settings after load: per-node drag {}, turbulence {}",
        s.settings.drag, s.settings.turbulence
    );
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let (lo, hi) = s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    });
    println!(
        "showcase sedan (low): {} nodes, {} beams, {} triangles, {m:.0} kg, extent {:.2?}; surface contact {}; collision range {:.3} m; {notes:?}",
        s.nodes.len(),
        s.beams.len(),
        s.triangles.len(),
        hi - lo,
        s.surface_contact,
        s.collision_range,
    );
    if let Some(d) = &s.drive {
        let c = &d.config;
        println!(
            "drive: {} wheels, max torque {:.0} N m per driven wheel, top speed {:.1} km/h, radius {:.3} m, driven {}",
            c.wheels.len(),
            c.max_torque,
            c.top_speed * 3.6,
            c.wheels[0].radius,
            c.wheels.iter().filter(|w| w.driven).count()
        );
    }
    Ok(s)
}

/// Mean node velocity and centroid.
fn centroid(pos: &[Vec3]) -> Vec3 {
    pos.iter().copied().sum::<Vec3>() / pos.len() as f32
}

/// Cruise: throttle to hold `target` m/s (brake above it).
fn cruise(v: f32, target: f32) -> DriveInput {
    let e = target - v;
    DriveInput {
        forward: (0.6 * e + 0.05).clamp(0.0, 1.0),
        brake: if e < -0.3 { (-0.5 * e).min(1.0) } else { 0.0 },
        ..Default::default()
    }
}

/// Jitter at steady speed from per-frame samples (centroid, mean velocity, node velocity
/// spread): reports the vertical position about its 0.5 s moving mean, the centroid's
/// acceleration (second differences at 60 Hz) and the node velocity spread.
fn jitter(label: &str, c: &[Vec3], spread: &[f32], hz: f32, steps: &[u32]) {
    if c.len() < 61 {
        return;
    }
    let w = (0.25 * hz) as usize;
    let mut dev = Vec::new();
    for i in w..c.len() - w {
        let mean: f32 = c[i - w..=i + w].iter().map(|p| p.y).sum::<f32>() / (2 * w + 1) as f32;
        dev.push(c[i].y - mean);
    }
    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    let mut ay = Vec::new();
    let mut ax = Vec::new();
    for i in 1..c.len() - 1 {
        let a = (c[i + 1] - 2.0 * c[i] + c[i - 1]) * hz * hz;
        ay.push(a.y);
        ax.push(a.z);
    }
    let pk = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if steps.len() == c.len() {
        // The same longitudinal acceleration with each frame's own simulated time (its
        // explicit steps × 0.5 ms), not 1/60 s: what is left is physical.
        let mut a2 = Vec::new();
        for i in 1..c.len() - 1 {
            let (d0, d1) = (steps[i] as f32 * 0.0005, steps[i + 1] as f32 * 0.0005);
            let (v0, v1) = ((c[i].z - c[i - 1].z) / d0, (c[i + 1].z - c[i].z) / d1);
            a2.push((v1 - v0) / (0.5 * (d0 + d1)));
        }
        let mut hist = std::collections::BTreeMap::new();
        for s in steps {
            *hist.entry(*s).or_insert(0u32) += 1;
        }
        println!(
            "{label}: explicit steps per 1/60 s frame {hist:?}; longitudinal accel on each frame's own time rms {:.3} m/s² (peak {:.3})",
            rms(&a2),
            pk(&a2)
        );
    }
    println!(
        "{label}: vertical about 0.5 s mean rms {:.2} mm (peak {:.2}); centroid accel vertical rms {:.3} m/s² (peak {:.3}), longitudinal rms {:.3} (peak {:.3}); node speed about the centroid's rms {:.3} m/s",
        1e3 * rms(&dev),
        1e3 * pk(&dev),
        rms(&ay),
        pk(&ay),
        rms(&ax),
        pk(&ax),
        spread.iter().sum::<f32>() / spread.len() as f32
    );
}

fn spread(vel: &[Vec3]) -> f32 {
    let v = vel.iter().copied().sum::<Vec3>() / vel.len() as f32;
    (vel.iter().map(|x| (*x - v).length_squared()).sum::<f32>() / vel.len() as f32).sqrt()
}

fn cpu_drive() -> anyhow::Result<()> {
    let drag = std::env::var("PP_DRAG").as_deref() != Ok("0");
    let t = sedan(drag)?;
    let secs = env_f("PP_SECS", 30.0);
    let kmh = env_f("PP_KMH", 0.0);
    let pe = env_f("PP_EVERY", 60.0) as u32;
    let coast = std::env::var("PP_COAST").as_deref() == Ok("1");
    let v0 = if coast { kmh / 3.6 } else { 0.0 };
    let mut sim = place_vehicle(&t, Vec3::new(0.0, GROUND, -80.0), 0, v0, GROUND);
    if let Some(d) = sim.drive.as_mut() {
        d.roll(v0);
    }
    let n = sim.nodes.len() as f32;
    let m: f32 = sim.nodes.iter().map(|n| n.mass).sum();
    let (lo, hi) = sim.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    });
    // Frontal area: the car's width × its height above the lowest node, × 0.85 (a car's
    // frontal area is about 80–85 % of its bounding rectangle).
    let area = 0.85 * (hi.x - lo.x) * (hi.y - lo.y);
    println!(
        "drag model: per node F = −{DRAG}·|v|·v, {n} nodes: {:.1}·v² N in total; physical ½ρC_dA v² (ρ 1.2, C_d 0.30, A {area:.2} m²) = {:.3}·v² N; ratio {:.0}×",
        n * DRAG,
        0.5 * 1.2 * 0.30 * area,
        n * DRAG / (0.5 * 1.2 * 0.30 * area)
    );
    for kmh in [20.0f32, 40.0, 100.0] {
        let v = kmh / 3.6;
        println!(
            "  at {kmh} km/h: model {:.0} N ({:.2} g on {m:.0} kg), physical {:.0} N",
            n * DRAG * v * v,
            n * DRAG * v * v / (m * 9.807),
            0.5 * 1.2 * 0.30 * area * v * v
        );
    }
    if let Some(d) = &sim.drive {
        let c = &d.config;
        let nd = c.wheels.iter().filter(|w| w.driven).count() as f32;
        let f = c.max_torque * nd / c.wheels[0].radius;
        println!(
            "drive force at full throttle below {:.0} km/h: {f:.0} N; equilibrium with the drag model alone at {:.1} km/h (fade ignored)",
            c.top_speed * 3.6 / 3.0,
            (f / (n * DRAG)).sqrt() * 3.6
        );
    }
    let mut cs = Vec::new();
    let mut spreads = Vec::new();
    let mut steps = Vec::new();
    let mut reached = [None::<f32>; 4];
    let marks = [20.0f32, 30.0, 40.0, 60.0];
    let mut prev_v = 0.0f32;
    for f in 1..=(secs * 60.0) as u32 {
        let v = sim.nodes.iter().map(|n| n.vel.z).sum::<f32>() / n;
        sim.set_drive_input(if coast {
            DriveInput::default()
        } else if kmh > 0.0 {
            cruise(v, kmh / 3.6)
        } else {
            DriveInput {
                forward: 1.0,
                ..Default::default()
            }
        });
        let st = sim.frame();
        let pos: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos + sim.origin).collect();
        let vel: Vec<Vec3> = sim.nodes.iter().map(|n| n.vel).collect();
        let v = vel.iter().map(|x| x.z).sum::<f32>() / n;
        for (k, mk) in marks.iter().enumerate() {
            if reached[k].is_none() && v * 3.6 >= *mk {
                reached[k] = Some(f as f32 / 60.0);
            }
        }
        if f % pe == 0 {
            let tot_drag: Vec3 = vel.iter().map(|x| -DRAG * x.length() * *x).sum::<Vec3>();
            println!(
                "t {:5.1} s: {:5.1} km/h, accel {:+.3} m/s², drag on nodes (from their velocities) {:.0} N",
                f as f32 / 60.0,
                v * 3.6,
                (v - prev_v) * 60.0 / pe as f32,
                if drag { tot_drag.z } else { 0.0 }
            );
            prev_v = v;
        }
        // Jitter over the last 10 s (steady speed).
        if f as f32 > (secs - 10.0) * 60.0 {
            cs.push(centroid(&pos));
            spreads.push(spread(&vel));
            steps.push(st);
        }
    }
    println!("time to {:?} km/h: {:?} s", marks, reached);
    jitter(
        &format!("CPU flat, drag {drag}, last 10 s"),
        &cs,
        &spreads,
        60.0,
        &steps,
    );
    Ok(())
}

fn cpu_post() -> anyhow::Result<()> {
    let t = sedan(std::env::var("PP_DRAG").as_deref() != Ok("0"))?;
    let kmh = env_f("PP_KMH", 5.0);
    let y = env_f("PP_Y", 0.55);
    let yaw = env_f("PP_YAW", 45.0).to_radians();
    let secs = env_f("PP_SECS", 6.0);
    let surf = CarSurface::of(&t);
    let mut sim = place_vehicle(&t, Vec3::new(0.0, GROUND, -0.6), 0, kmh / 3.6, GROUND);
    if let Some(d) = sim.drive.as_mut() {
        d.roll(kmh / 3.6);
    }
    let post = Obb {
        c: Vec3::new(env_f("PP_X", 0.0), GROUND + y, 0.5),
        q: Quat::from_rotation_y(yaw),
        h: HALF,
    };
    sim.shapes
        .push(StaticShape::oriented_box(post.c, post.h, post.q));
    let mut pushing = false;
    let mut worst = Pen::default();
    for f in 1..=(secs * 60.0) as u32 {
        let v = sim.nodes.iter().map(|n| n.vel.z).sum::<f32>() / sim.nodes.len() as f32;
        let pos: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos + sim.origin).collect();
        let p = measure(&surf, &pos, &[post]);
        if p.node > -0.05 || p.skin > 0.0 {
            pushing = true;
        }
        sim.set_drive_input(if pushing {
            DriveInput {
                forward: 1.0,
                ..Default::default()
            }
        } else {
            cruise(v, kmh / 3.6)
        });
        sim.frame();
        let pos: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos + sim.origin).collect();
        let p = measure(&surf, &pos, &[post]);
        worst.max(p);
        if f % 15 == 0 {
            let broken = sim.beams.iter().filter(|b| b.broken).count();
            println!(
                "t {:.2}: v {:.2} km/h{}; node {:.3}, skin {:.3}, hull {:.3} m; broken {broken}; surface hits {}",
                f as f32 / 60.0,
                v * 3.6,
                if pushing { " (full throttle)" } else { "" },
                p.node,
                p.skin,
                p.hull,
                sim.surface_last
            );
        }
    }
    println!("worst over the run: {worst:.3?}");
    Ok(())
}

fn gpu(scene: &str) -> anyhow::Result<()> {
    let kmh = env_f("PP_KMH", 5.0);
    let secs = env_f("PP_SECS", 12.0);
    let runup = env_f("PP_RUNUP", 2.0);
    // PP_HZ: World steps per second (the coupling window is one step).
    let hz = env_f("PP_HZ", 60.0);
    let mut def = ring_world_def(4096);
    def.solver.dt = 1.0 / hz;
    if env_f("PP_SLEEP", 1.0) > 0.0 {
        def.solver.sleep_time = 0.5;
    }
    let mut w = World::new(GpuContext::Default, &def)?;
    let g = w.create_body(&BodyDef::default())?;
    w.create_box_shape(
        g,
        &ShapeDef {
            material: SurfaceMaterial {
                friction: 0.5,
                ..Default::default()
            },
            density: 0.0,
            filter: engine_schema::convex::CollisionFilter {
                categories: 0x8000,
                ..Default::default()
            },
            ..ShapeDef::default()
        },
        [200.0, 0.5, 200.0],
    )?;
    let face = 0.0f32;
    let mut defs = Vec::new();
    match scene {
        "wall" | "rigid" => {
            let (cols, rows, deep) = (
                env_f("PP_COLS", 5.0) as i32,
                env_f("PP_ROWS", 4.0) as i32,
                env_f("PP_DEEP", 2.0) as i32,
            );
            for d in 0..deep {
                for r in 0..rows {
                    for c in 0..cols {
                        let x = (c as f32 - 0.5 * (cols - 1) as f32) * 1.02
                            + if r % 2 == 1 { 0.51 } else { 0.0 };
                        defs.push(BodyDef::dynamic([
                            x,
                            GROUND + 0.25 + 0.52 * r as f32,
                            face + 0.25 + 0.52 * d as f32,
                        ]));
                    }
                }
            }
        }
        "loose" => {
            // Bricks lying across the road (long axis along x), spaced along it, in the
            // wheels' tracks.
            let cols = env_f("PP_COLS", 4.0) as i32;
            for c in 0..cols {
                defs.push(BodyDef::dynamic([
                    if c % 2 == 0 { -0.75 } else { 0.75 },
                    GROUND + 0.25,
                    face + 1.5 * c as f32,
                ]));
            }
        }
        _ => {}
    }
    // PP_STATIC=1: the bricks are static bodies (a control: no motion, no window lag).
    if std::env::var("PP_STATIC").as_deref() == Ok("1") {
        for d in &mut defs {
            *d = BodyDef {
                position: d.position,
                ..BodyDef::default()
            };
        }
    }
    let bricks = w.create_bodies(&defs)?;
    let shape = ShapeDef {
        material: SurfaceMaterial {
            friction: 0.5,
            ..Default::default()
        },
        density: if std::env::var("PP_STATIC").as_deref() == Ok("1") {
            0.0
        } else {
            env_f("PP_DENSITY", 100.0)
        },
        ..ShapeDef::default()
    };
    for id in &bricks {
        w.create_box_shape(*id, &shape, HALF.to_array())?;
    }
    w.enable_sleep_archive(None)?;
    let read_bricks = |w: &World| -> anyhow::Result<Vec<Obb>> {
        bricks
            .iter()
            .map(|b| {
                let s = w.read_body_state(*b)?;
                Ok(Obb {
                    c: Vec3::from(s.transform.position),
                    q: Quat::from_array(s.transform.rotation),
                    h: HALF,
                })
            })
            .collect()
    };
    if scene == "rigid" {
        // The control: a car-sized rigid box (1.8 × 1.4 × 4.6 m, 1,300 kg) at the car's
        // height, its velocity held at PP_KMH, into the same wall.
        let h = Vec3::new(0.9, 0.7, 2.3);
        let block = w.create_body(&BodyDef {
            linear_velocity: [0.0, 0.0, kmh / 3.6],
            ..BodyDef::dynamic([0.0, GROUND + 0.2 + h.y, face - runup - h.z])
        })?;
        w.create_box_shape(
            block,
            &ShapeDef {
                density: 1300.0 / (8.0 * h.x * h.y * h.z),
                ..ShapeDef::default()
            },
            h.to_array(),
        )?;
        let mut worst = 0.0f32;
        for k in 1..=(secs * hz) as u32 {
            w.body_set_linear_velocity(block, [0.0, 0.0, kmh / 3.6])?;
            w.step()?;
            w.poll_blocking()?;
            let s = w.read_body_state(block)?;
            let bo = Obb {
                c: Vec3::from(s.transform.position),
                q: Quat::from_array(s.transform.rotation),
                h,
            };
            let mut d = 0.0f32;
            for b in read_bricks(&w)? {
                for p in b.points(5) {
                    d = d.max(bo.depth(p));
                }
                for p in bo.points(9) {
                    d = d.max(b.depth(p));
                }
            }
            worst = worst.max(d);
            if k % 30 == 0 {
                println!(
                    "t {:.2}: block front z {:.2}, v {:.2} m/s; deepest box point inside another box {:.3} m",
                    k as f32 / hz,
                    bo.c.z + h.z,
                    Vec3::from(s.linear_velocity).z,
                    d
                );
            }
        }
        println!("rigid control worst overlap depth: {worst:.3} m");
        return Ok(());
    }
    let drag = std::env::var("PP_DRAG").as_deref() != Ok("0");
    let t = sedan(drag)?;
    let surf = CarSurface::of(&t);
    let mut car = place_vehicle(
        &t,
        Vec3::new(0.0, GROUND, face - runup),
        0,
        kmh / 3.6,
        GROUND,
    );
    if let Some(d) = car.drive.as_mut() {
        d.roll(kmh / 3.6);
    }
    w.add_explicit_vehicles(std::slice::from_ref(&car), &ring_coupling(1024))?;
    let every = env_f("PP_EVERY", 15.0) as u32;
    let full = std::env::var("PP_PUSH").as_deref() != Ok("0") && scene == "wall";
    let mut pushing = false;
    let mut worst = Pen::default();
    let mut worst_push = Pen::default();
    let mut cs = Vec::new();
    let mut spreads = Vec::new();
    for k in 1..=(secs * hz) as u32 {
        let v = {
            let x = w.explicit_vehicles().unwrap();
            let vel = &x.velocities()[0];
            vel.iter().map(|v| v.z).sum::<f32>() / vel.len() as f32
        };
        let input = if pushing {
            DriveInput {
                forward: 1.0,
                ..Default::default()
            }
        } else {
            cruise(v, kmh / 3.6)
        };
        w.explicit_vehicles_mut().unwrap().set_drive_input(0, input);
        w.step()?;
        w.poll_blocking()?;
        let x = w.explicit_vehicles().unwrap();
        if let Err(e) = x.kernel_status() {
            println!("step {k}: kernel status {e}; stopping");
            break;
        }
        let pos = x.positions()[0].clone();
        let vel = x.velocities()[0].clone();
        if pos.iter().any(|p| !p.is_finite()) {
            println!("step {k}: non-finite; stopping");
            break;
        }
        let imp = w.explicit_window_impulses();
        let j: Vec3 = imp[0].iter().map(|s| Vec3::from(s.1)).sum();
        if full && j.length() > 1.0 {
            pushing = true;
        }
        let p = if bricks.is_empty() {
            Pen::default()
        } else {
            measure(&surf, &pos, &read_bricks(&w)?)
        };
        worst.max(p);
        if pushing {
            worst_push.max(p);
        }
        if scene == "flat" && k as f32 > (secs - 10.0) * hz {
            cs.push(centroid(&pos));
            spreads.push(spread(&vel));
        }
        if k % every == 0 {
            let broken = x.beams()[0].iter().filter(|b| b.broken).count();
            let nose = pos.iter().map(|p| p.z).fold(f32::MIN, f32::max);
            println!(
                "t {:.2}: v {:.2} km/h{}; nose z {:.2}; window impulse {:.0} N s on {} shapes; node {:.3}, skin {:.3}, hull {:.3} ({} bricks > 5 cm), wheel {:.3} m; broken {broken}",
                k as f32 / hz,
                v * 3.6,
                if pushing { " (full throttle)" } else { "" },
                nose,
                j.length(),
                imp[0].len(),
                p.node,
                p.skin,
                p.hull,
                p.hull_bricks,
                p.wheel,
            );
        }
    }
    if scene == "flat" {
        jitter("GPU World flat, last 10 s", &cs, &spreads, hz, &[]);
    }
    println!("worst over the run: {worst:.3?}");
    println!("worst while pushing: {worst_push:.3?}");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let scene = std::env::var("PP_SCENE").unwrap_or_else(|_| "cpu-drive".into());
    match scene.as_str() {
        "cpu-drive" => cpu_drive(),
        "cpu-post" => cpu_post(),
        s => gpu(s),
    }
}
