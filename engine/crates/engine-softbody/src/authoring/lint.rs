//! `vehicle_lint`: a node–beam vehicle checked against what we know a good
//! one looks like (`engine/docs/generated-vehicles.md`, the BeamNG authoring
//! census `beamng-authoring-census.md`). It reads any JBeam car as the loader
//! does (a [`SoftVehicleDesc`]) and, when it has one, its bound render mesh
//! ([`ShowcaseVisual`]), so it runs on generated cars and community cars
//! alike.
//!
//! **Hard rules** (a generated car must pass every one):
//! * `skin_coverage`: the outer render skin (paint and glass facing out;
//!   not the wheels or engine) lies within [`COVER_FAIL`] of a collision
//!   triangle at the 99th percentile: no hole in the collision skin a
//!   254 mm FMVSS 214 pole fits (the ETK: p99 0.107 m, max 0.173 m);
//! * `free_mechanism`: every node is held in all three directions by
//!   bilateral beams (no node with a zero-stiffness direction), and the beam
//!   graph is one piece;
//! * `stability`: the springs' highest mode ω·h within the explicit step's
//!   bound ([`OMEGA_H_PASS`], centred damping; [`OMEGA_H_FAIL`] the
//!   undamped bound);
//! * `mass`: total within ±10 % of the design's ([`Expect::mass`]);
//! * `hardware`: every hinge, latch and bolt marker has its hardware (a
//!   break-group beam or a coupler within [`HARDWARE_REACH`]), every wheel
//!   centre its hub node.
//!
//! **Bands** (pass / warn, from the census): node mass, k·L, yield strain and
//! damping per part; nodes per part; spring ends per node; the SUPPORT share;
//! hinge, latch and seal patterns per door; body shell mass share; collision
//! triangles; and on the render mesh, node distance to the skin, nodes per m²
//! of panel and how far each vertex reaches to its nodes.

use crate::authoring::skin::{mat, ShowcaseVisual};
use crate::desc::{BeamKind, SoftVehicleDesc};
use glam::{DMat3, DVec3, Vec3};
use std::collections::{BTreeMap, HashMap, HashSet};

/// A collision skin hole: an outer vertex this far from every collision
/// triangle fails (half the FMVSS 214 pole's 254 mm); past [`COVER_WARN`] warns.
pub const COVER_FAIL: f32 = 0.127;
pub const COVER_WARN: f32 = 0.05;
/// The explicit step's bound on ω_max·h: centred damping's (pass) and the
/// undamped symplectic step's (fail).
pub const OMEGA_H_PASS: f64 = 1.92;
pub const OMEGA_H_FAIL: f64 = 2.0;
/// A node's weakest direction under this many Hz (its stiffness there over
/// its mass) is a free mechanism.
pub const MECHANISM_HZ: f64 = 1.0;
/// Hardware within this of its marker (m).
pub const HARDWARE_REACH: f32 = 0.15;
/// The explicit step (s).
pub const STEP: f64 = 0.0005;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Pass,
    Warn,
    Fail,
    /// Not applicable (no data, or no band).
    Na,
}

impl Level {
    pub fn tag(self) -> &'static str {
        match self {
            Level::Pass => "pass",
            Level::Warn => "warn",
            Level::Fail => "FAIL",
            Level::Na => "-",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Finding {
    pub rule: &'static str,
    pub hard: bool,
    /// The part (or `car`).
    pub part: String,
    pub level: Level,
    pub value: String,
    pub band: String,
    /// Named offenders (nodes, pieces, markers), at most a few.
    pub offenders: Vec<String>,
}

/// What the design says the car should be.
#[derive(Clone, Debug, Default)]
pub struct Expect {
    /// The nodes' total mass (kg): the kerb less what the loader adds (the
    /// raycast wheels' mass). None: not checked.
    pub mass: Option<f32>,
    /// The art spec's markers in the solver's frame (name, position).
    pub markers: Vec<(String, Vec3)>,
}

/// Part categories the census measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Cat {
    Body,
    Door,
    Bonnet,
    Boot,
    Fender,
    Bumper,
    Other,
}

impl Cat {
    pub fn of(part: &str) -> Cat {
        let p = part.to_ascii_lowercase();
        if p.contains("door") {
            Cat::Door
        } else if p.contains("hood") || p.contains("bonnet") {
            Cat::Bonnet
        } else if p.contains("trunk")
            || p.contains("tailgate")
            || p.contains("boot")
            || p.contains("decklid")
        {
            Cat::Boot
        } else if p.contains("fender") {
            Cat::Fender
        } else if p.contains("bumper") {
            Cat::Bumper
        } else if p.ends_with("body") || p.contains("_body") || p == "body" {
            Cat::Body
        } else {
            Cat::Other
        }
    }
}

/// A band (lo, hi) per category for one measure, from the census (§1.2 of
/// `generated-vehicles.md`).
fn band(measure: &str, c: Cat) -> Option<(f64, f64)> {
    use Cat::*;
    Some(match (measure, c) {
        ("node_kg", Body) => (2.4, 4.0),
        ("node_kg", Door) => (1.2, 1.9),
        ("node_kg", Bonnet) => (0.55, 1.5),
        ("node_kg", Boot) => (0.6, 1.6),
        ("node_kg", Fender) => (0.4, 0.8),
        ("node_kg", Bumper) => (0.3, 0.9),
        ("kl", Body) => (0.8e6, 1.1e6),
        ("kl", Door) => (0.4e6, 1.0e6),
        ("kl", Bonnet) => (2.2e5, 4.7e5),
        ("kl", Boot) => (2.2e5, 5.0e5),
        ("kl", Fender) => (1.5e5, 4.3e5),
        ("kl", Bumper) => (0.4e5, 1.3e5),
        ("eps", Body) => (0.010, 0.013),
        ("eps", Door) => (0.013, 0.016),
        ("eps", Bonnet) => (0.015, 0.055),
        ("eps", Boot) => (0.02, 0.05),
        ("eps", Bumper) => (0.03, 0.21),
        ("zeta", Body) => (0.03, 0.04),
        ("zeta", Bumper) => (0.07, 0.33),
        ("nodes", Body) => (96.0, 138.0),
        ("nodes", Door) => (9.0, 13.0),
        ("nodes", Bonnet) => (14.0, 30.0),
        ("nodes", Boot) => (12.0, 22.0),
        ("nodes", Fender) => (10.0, 13.0),
        ("nodes", Bumper) => (19.0, 38.0),
        ("ends", Body) => (15.0, 18.0),
        ("node_dist", Body | Door | Bonnet | Boot | Fender | Bumper) => (0.01, 0.04),
        ("per_m2", Body | Door | Bonnet | Boot | Fender | Bumper) => (4.5, 11.0),
        ("reach", Body | Door | Bonnet | Boot | Fender | Bumper) => (0.08, 0.20),
        _ => return None,
    })
}

fn q(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn fmt(x: f64) -> String {
    if !x.is_finite() {
        return "-".into();
    }
    let a = x.abs();
    if a >= 1e4 {
        format!("{x:.2e}")
    } else if a >= 10.0 {
        format!("{x:.0}")
    } else if a >= 0.1 {
        format!("{x:.2}")
    } else {
        format!("{x:.3}")
    }
}

fn banded(rule: &'static str, part: &str, x: f64, b: Option<(f64, f64)>) -> Finding {
    let (level, band) = match b {
        Some((lo, hi)) if x.is_finite() => (
            if x >= lo * 0.999 && x <= hi * 1.001 {
                Level::Pass
            } else {
                Level::Warn
            },
            format!("{}–{}", fmt(lo), fmt(hi)),
        ),
        _ => (Level::Na, "-".into()),
    };
    Finding {
        rule,
        hard: false,
        part: part.into(),
        level,
        value: fmt(x),
        band,
        offenders: vec![],
    }
}

/// Bilateral beams (they hold both ways): not SUPPORT (compression only) or
/// ropes (tension only).
fn bilateral(k: BeamKind) -> bool {
    !matches!(k, BeamKind::Support | BeamKind::Rope)
}

/// Each node's part: the first part listing it (names), and per node its
/// part index (usize::MAX: none).
fn node_parts(d: &SoftVehicleDesc) -> Vec<usize> {
    let mut out = vec![usize::MAX; d.nodes.len()];
    for (pi, p) in d.parts.iter().enumerate() {
        for &n in &p.nodes {
            if let Some(o) = out.get_mut(n as usize) {
                if *o == usize::MAX {
                    *o = pi;
                }
            }
        }
    }
    out
}

/// ω_max·h of the springs (power iteration on M⁻¹K, damping not counted).
pub fn omega_h(d: &SoftVehicleDesc) -> f64 {
    let n = d.nodes.len();
    let p: Vec<DVec3> = d
        .nodes
        .iter()
        .map(|x| Vec3::from_array(x.position).as_dvec3())
        .collect();
    let m: Vec<f64> = d.nodes.iter().map(|x| (x.mass as f64).max(1e-3)).collect();
    let beams: Vec<(usize, usize, f64, DVec3)> = d
        .beams
        .iter()
        .filter_map(|b| {
            let (a, c) = (b.a as usize, b.b as usize);
            let dd = p[c] - p[a];
            let l = dd.length();
            let k = (b.spring.max(b.spring_expansion)) as f64;
            (a < n && c < n && l > 1e-6 && k > 0.0).then(|| (a, c, k, dd / l))
        })
        .collect();
    let mut x: Vec<DVec3> = (0..n)
        .map(|i| {
            let h = (i as f64 * 12.9898).sin() * 43758.5453;
            DVec3::new(h.fract(), (h * 1.7).fract(), (h * 2.3).fract()) - 0.5
        })
        .collect();
    let mut lam = 0.0;
    for _ in 0..300 {
        let mut y = vec![DVec3::ZERO; n];
        for &(a, c, k, u) in &beams {
            let f = u * (k * u.dot(x[a] - x[c]));
            y[a] += f;
            y[c] -= f;
        }
        for i in 0..n {
            y[i] /= m[i];
        }
        let norm = y.iter().map(|v| v.length_squared()).sum::<f64>().sqrt();
        let xn = x.iter().map(|v| v.length_squared()).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        lam = norm / xn.max(1e-30);
        for i in 0..n {
            x[i] = y[i] / norm;
        }
    }
    lam.sqrt() * STEP
}

/// Nodes with a direction no bilateral beam holds (its stiffness there,
/// over the node's mass, under [`MECHANISM_HZ`]); and the beam graph's
/// pieces (beams and couplers), largest first.
pub fn mechanisms(d: &SoftVehicleDesc) -> (Vec<(usize, f64)>, Vec<Vec<usize>>) {
    let n = d.nodes.len();
    let mut t = vec![DMat3::ZERO; n];
    let pos = |i: usize| Vec3::from_array(d.nodes[i].position).as_dvec3();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in &d.beams {
        let (a, c) = (b.a as usize, b.b as usize);
        if a >= n || c >= n {
            continue;
        }
        adj[a].push(c);
        adj[c].push(a);
        if !bilateral(b.kind) {
            continue;
        }
        let dd = pos(c) - pos(a);
        let l = dd.length();
        if l < 1e-6 {
            continue;
        }
        let u = dd / l;
        let k = b.spring.max(b.spring_expansion) as f64;
        let o = DMat3::from_cols(u * u.x, u * u.y, u * u.z) * k;
        t[a] += o;
        t[c] += o;
    }
    for c in &d.couplers {
        let (a, b) = (c.a as usize, c.b as usize);
        if a < n && b < n {
            adj[a].push(b);
            adj[b].push(a);
        }
    }
    let wmin = (2.0 * std::f64::consts::PI * MECHANISM_HZ).powi(2);
    let mut free = Vec::new();
    for i in 0..n {
        let m = (d.nodes[i].mass as f64).max(1e-3);
        let e = sym_eig_min(t[i]);
        if e / m < wmin {
            free.push((i, (e.max(0.0) / m).sqrt() / (2.0 * std::f64::consts::PI)));
        }
    }
    let mut seen = vec![false; n];
    let mut comps = Vec::new();
    for s in 0..n {
        if seen[s] {
            continue;
        }
        let mut stack = vec![s];
        let mut comp = Vec::new();
        seen[s] = true;
        while let Some(i) = stack.pop() {
            comp.push(i);
            for &j in &adj[i] {
                if !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        comps.push(comp);
    }
    comps.sort_by_key(|c| std::cmp::Reverse(c.len()));
    (free, comps)
}

/// The smallest eigenvalue of a symmetric 3×3 (Jacobi sweeps).
fn sym_eig_min(m: DMat3) -> f64 {
    let mut a = [
        [m.x_axis.x, m.y_axis.x, m.z_axis.x],
        [m.x_axis.y, m.y_axis.y, m.z_axis.y],
        [m.x_axis.z, m.y_axis.z, m.z_axis.z],
    ];
    for _ in 0..30 {
        for (p, qq) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][qq].abs() < 1e-30 {
                continue;
            }
            let th = 0.5 * (2.0 * a[p][qq]).atan2(a[qq][qq] - a[p][p]);
            let (c, s) = (th.cos(), th.sin());
            let mut r = [[0.0; 3]; 3];
            for (i, row) in r.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            r[p][p] = c;
            r[qq][qq] = c;
            r[p][qq] = s;
            r[qq][p] = -s;
            // a = rᵀ a r
            let mut tmp = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    tmp[i][j] = (0..3).map(|k| a[i][k] * r[k][j]).sum();
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    a[i][j] = (0..3).map(|k| r[k][i] * tmp[k][j]).sum();
                }
            }
        }
    }
    a[0][0].min(a[1][1]).min(a[2][2])
}

/// Point to triangle distance.
fn tri_dist(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // Ericson, Real-Time Collision Detection §5.1.5 (closest point).
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return p.distance(a);
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return p.distance(b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return p.distance(a + ab * v);
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return p.distance(c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return p.distance(a + ac * w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return p.distance(b + (c - b) * w);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    p.distance(a + ab * v + ac * w)
}

/// Lint a car. `vis` positions and `node_rest` share a frame; the desc's
/// positions may be in another (only shapes are read from them).
pub fn lint(d: &SoftVehicleDesc, vis: Option<&ShowcaseVisual>, ex: &Expect) -> Vec<Finding> {
    let mut out = Vec::new();
    let np = node_parts(d);
    let pname = |i: usize| -> String {
        np.get(i)
            .and_then(|&p| d.parts.get(p))
            .map_or("-".into(), |p| p.name.clone())
    };
    let nname = |i: usize| d.nodes.get(i).map_or(format!("#{i}"), |n| n.name.clone());

    // --- Hard: free mechanisms -------------------------------------------
    let (free, comps) = mechanisms(d);
    let mut off: Vec<String> = free
        .iter()
        .take(8)
        .map(|(i, hz)| format!("{} ({}, {:.2} Hz)", nname(*i), pname(*i), hz))
        .collect();
    if comps.len() > 1 {
        off.extend(comps.iter().skip(1).take(4).map(|c| {
            format!(
                "loose piece of {} nodes ({})",
                c.len(),
                c.iter()
                    .take(3)
                    .map(|&i| nname(i))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }));
    }
    out.push(Finding {
        rule: "free_mechanism",
        hard: true,
        part: "car".into(),
        level: if free.is_empty() && comps.len() <= 1 {
            Level::Pass
        } else {
            Level::Fail
        },
        value: format!("{} free nodes, {} pieces", free.len(), comps.len()),
        band: format!("0 nodes under {MECHANISM_HZ} Hz, 1 piece"),
        offenders: off,
    });

    // --- Hard: stability ---------------------------------------------------
    let wh = omega_h(d);
    out.push(Finding {
        rule: "stability",
        hard: true,
        part: "car".into(),
        level: if wh <= OMEGA_H_PASS + 1e-3 {
            Level::Pass
        } else if wh <= OMEGA_H_FAIL {
            Level::Warn
        } else {
            Level::Fail
        },
        value: format!("ω·h {wh:.3}"),
        band: format!("≤ {OMEGA_H_PASS} (fail > {OMEGA_H_FAIL})"),
        offenders: vec![],
    });

    // --- Hard: mass ---------------------------------------------------------
    let total: f64 = d.nodes.iter().map(|n| n.mass as f64).sum();
    out.push(match ex.mass {
        Some(m) => {
            let r = total / m as f64 - 1.0;
            Finding {
                rule: "mass",
                hard: true,
                part: "car".into(),
                level: if r.abs() <= 0.03 {
                    Level::Pass
                } else if r.abs() <= 0.10 {
                    Level::Warn
                } else {
                    Level::Fail
                },
                value: format!("{total:.1} kg ({:+.1} %)", 100.0 * r),
                band: format!("{m:.0} kg ± 10 %"),
                offenders: vec![],
            }
        }
        None => Finding {
            rule: "mass",
            hard: true,
            part: "car".into(),
            level: Level::Na,
            value: format!("{total:.1} kg"),
            band: "no design mass".into(),
            offenders: vec![],
        },
    });

    // --- Hard: hardware at the markers -----------------------------------
    {
        let pos = |i: u32| Vec3::from_array(d.nodes[i as usize].position);
        let hw: Vec<(Vec3, Vec3)> = d
            .beams
            .iter()
            .filter(|b| b.break_group != 0)
            .map(|b| (pos(b.a), pos(b.b)))
            .collect();
        let cp: Vec<Vec3> = d
            .couplers
            .iter()
            .flat_map(|c| [pos(c.a), pos(c.b)])
            .collect();
        let mut missing = Vec::new();
        let mut checked = 0;
        for (name, at) in &ex.markers {
            let near = |p: Vec3| p.distance(*at) <= HARDWARE_REACH;
            let ok = if name.starts_with("latch_") {
                cp.iter().any(|&p| near(p))
            } else if name.starts_with("hinge_") || name.starts_with("bolt_") {
                hw.iter()
                    .any(|&(a, b)| near(a) || near(b) || near(0.5 * (a + b)))
            } else if name.starts_with("wheel_center_") {
                // (a wheel's axle there, or a hub node)
                d.wheels
                    .iter()
                    .any(|w| near(0.5 * (pos(w.node1) + pos(w.node2))))
                    || d.nodes.iter().any(|n| {
                        n.name.starts_with("hub") && near(Vec3::from_array(n.position))
                    })
            } else {
                continue;
            };
            checked += 1;
            if !ok {
                missing.push(name.clone());
            }
        }
        out.push(Finding {
            rule: "hardware",
            hard: true,
            part: "car".into(),
            level: if ex.markers.is_empty() {
                Level::Na
            } else if missing.is_empty() {
                Level::Pass
            } else {
                Level::Fail
            },
            value: format!("{}/{} markers held", checked - missing.len(), checked),
            band: format!("all, within {HARDWARE_REACH} m"),
            offenders: missing,
        });
    }

    // --- Hard: collision skin coverage -------------------------------------
    if let Some(v) = vis {
        let nr: Vec<Vec3> = v.node_rest.iter().map(|p| Vec3::from_array(*p)).collect();
        let tris: Vec<[Vec3; 3]> = d
            .triangles
            .iter()
            .filter(|t| t.collision && t.nodes.iter().all(|&i| (i as usize) < nr.len()))
            .map(|t| t.nodes.map(|i| nr[i as usize]))
            .collect();
        let cell = 0.1f32;
        let key = |p: Vec3| {
            (
                (p.x / cell).floor() as i32,
                (p.y / cell).floor() as i32,
                (p.z / cell).floor() as i32,
            )
        };
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (k, t) in tris.iter().enumerate() {
            let lo = t[0].min(t[1]).min(t[2]);
            let hi = t[0].max(t[1]).max(t[2]);
            let (a, b) = (key(lo), key(hi));
            for x in a.0..=b.0 {
                for y in a.1..=b.1 {
                    for z in a.2..=b.2 {
                        grid.entry((x, y, z)).or_default().push(k);
                    }
                }
            }
        }
        // The car's axis (rest frame): the outer skin faces away from it.
        let (lo, hi) = nr.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(a, b), p| (a.min(*p), b.max(*p)),
        );
        let c = 0.5 * (lo + hi);
        // Rubble's frame is +Y up: the car's length is the longer of x and z.
        let e = hi - lo;
        let len_axis = if e.x > e.z { 0 } else { 2 };
        // Outward from the car's box-scaled centre, the end faces included
        // (zeroing the length axis left the nose and tail out of the check:
        // the generated sedan's front face had no collision triangle within
        // 0.6 m over a fifth of its area and passed).
        let half = (e * 0.5).max(Vec3::splat(1e-3));
        let outer = |p: Vec3, n: Vec3| n.dot((p - c) / (half * half)) > 0.0;
        let _ = len_axis;
        let piece_of = |i: u32| {
            v.pieces
                .iter()
                .find(|q| (q.1..q.1 + q.2).contains(&i))
                .map_or("?", |q| q.0.as_str())
        };
        let mut worst: BTreeMap<String, (f32, usize, Vec3)> = BTreeMap::new();
        let mut ds = Vec::new();
        for i in 0..v.positions.len() {
            let m = v.material.get(i).copied().unwrap_or(255);
            if !matches!(m, mat::PAINT | mat::GLASS | mat::SCREEN | mat::PRIVACY) {
                continue;
            }
            // The wheels' and the engine's pieces are not the body's skin.
            let pc = piece_of(i as u32);
            let bare = pc.rsplit(':').next().unwrap_or(pc);
            if [
                "rim",
                "tyre",
                "brake",
                "engine",
                "wheel",
                "seat",
                "dash",
                "steering",
                "console",
                "headliner",
                "door_trim",
                "interior",
                "mirror",
            ]
            .iter()
            .any(|w| bare.starts_with(w))
            {
                continue;
            }
            let p = Vec3::from_array(v.positions[i]);
            let n = Vec3::from_array(v.normals[i]);
            if !outer(p, n) {
                continue;
            }
            let (x, y, z) = key(p);
            let mut best = f32::MAX;
            for dx in -2..=2 {
                for dy in -2..=2 {
                    for dz in -2..=2 {
                        if let Some(l) = grid.get(&(x + dx, y + dy, z + dz)) {
                            for &k in l {
                                let t = tris[k];
                                best = best.min(tri_dist(p, t[0], t[1], t[2]));
                            }
                        }
                    }
                }
            }
            let best = best.min(0.25);
            ds.push(best as f64);
            if best > COVER_FAIL && std::env::var("LINT_HOLES").is_ok() {
                println!("hole {pc} {:.3} {:.3} {:.3} {best:.2}", p.x, p.y, p.z);
            }
            if best > COVER_FAIL {
                let e = worst
                    .entry(piece_of(i as u32).to_string())
                    .or_insert((0.0, 0, Vec3::ZERO));
                e.0 = e.0.max(best);
                e.1 += 1;
                e.2 += p;
            }
        }
        let n_out = ds.len();
        let mx = ds.iter().cloned().fold(0.0, f64::max);
        let p99 = q(&mut ds, 0.99);
        let mut off: Vec<(String, (f32, usize, Vec3))> = worst.into_iter().collect();
        off.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        out.push(Finding {
            rule: "skin_coverage",
            hard: true,
            part: "car".into(),
            level: if tris.is_empty() || p99 as f32 > COVER_FAIL {
                Level::Fail
            } else if mx as f32 > COVER_FAIL || p99 as f32 > COVER_WARN {
                Level::Warn
            } else {
                Level::Pass
            },
            value: format!(
                "{} collision tris; outer vertex to skin p99 {:.3} m, max {:.3} m ({} vertices)",
                tris.len(),
                p99,
                mx,
                n_out
            ),
            band: format!("p99 ≤ {COVER_FAIL} m (warn: p99 > {COVER_WARN} or max > {COVER_FAIL})"),
            offenders: off
                .iter()
                .take(8)
                .map(|(k, (m, n, c))| {
                    let c = *c / *n as f32;
                    format!(
                        "{k}: {n} vertices, to {m:.2} m, about ({:.2}, {:.2}, {:.2})",
                        c.x, c.y, c.z
                    )
                })
                .collect(),
        });
    } else {
        out.push(Finding {
            rule: "skin_coverage",
            hard: true,
            part: "car".into(),
            level: Level::Na,
            value: "no render mesh".into(),
            band: "-".into(),
            offenders: vec![],
        });
    }

    // --- Bands per part ------------------------------------------------------
    let pos = |i: usize| Vec3::from_array(d.nodes[i].position).as_dvec3();
    let mass = |i: usize| d.nodes[i].mass as f64;
    let mut ends = vec![0usize; d.nodes.len()];
    for b in &d.beams {
        if b.kind == BeamKind::Normal || b.kind == BeamKind::Anisotropic {
            ends[b.a as usize] += 1;
            ends[b.b as usize] += 1;
        }
    }
    let contact_only = |i: usize| d.nodes[i].name.starts_with('x') && d.nodes[i].mass < 0.2;
    for (pi, p) in d.parts.iter().enumerate() {
        let c = Cat::of(&p.name);
        if c == Cat::Other {
            continue;
        }
        let own: Vec<usize> = p
            .nodes
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| i < d.nodes.len() && np[i] == pi)
            .collect();
        let part_kg: f64 = own.iter().map(|&i| mass(i)).sum();
        let structural: Vec<usize> = own.iter().copied().filter(|&i| !contact_only(i)).collect();
        let mut nk: Vec<f64> = structural.iter().map(|&i| mass(i)).collect();
        out.push(banded(
            "nodes",
            &p.name,
            structural.len() as f64,
            band("nodes", c),
        ));
        out.push(banded(
            "node_kg_p50",
            &p.name,
            q(&mut nk, 0.5),
            band("node_kg", c),
        ));
        let mut en: Vec<f64> = structural.iter().map(|&i| ends[i] as f64).collect();
        out.push(banded(
            "spring_ends_p50",
            &p.name,
            q(&mut en, 0.5),
            band("ends", c),
        ));
        let (mut kl, mut eps, mut zeta) = (Vec::new(), Vec::new(), Vec::new());
        for b in &d.beams {
            let (a, bb) = (b.a as usize, b.b as usize);
            if a >= np.len() || bb >= np.len() || np[a] != pi || np[bb] != pi {
                continue;
            }
            if !matches!(b.kind, BeamKind::Normal | BeamKind::Anisotropic) || b.spring <= 0.0 {
                continue;
            }
            let l = (pos(a) - pos(bb)).length();
            let k = b.spring as f64;
            kl.push(k * l);
            if b.deform.is_finite() && l > 0.0 {
                eps.push(b.deform as f64 / (k * l));
            }
            let me = mass(a) * mass(bb) / (mass(a) + mass(bb)).max(1e-9);
            if me > 0.0 {
                zeta.push(b.damp as f64 / (2.0 * (k * me).sqrt()));
            }
        }
        out.push(banded("kL_p50", &p.name, q(&mut kl, 0.5), band("kl", c)));
        out.push(banded(
            "yield_strain_p50",
            &p.name,
            q(&mut eps, 0.5),
            band("eps", c),
        ));
        out.push(banded(
            "zeta_p50",
            &p.name,
            q(&mut zeta, 0.5),
            band("zeta", c),
        ));
        if c == Cat::Body {
            out.push(banded(
                "body_mass_share",
                &p.name,
                part_kg / total.max(1e-9),
                Some((0.23, 0.38)),
            ));
        }
        if c == Cat::Door {
            // Hinges: break groups from the door to another part.
            let mut groups: BTreeMap<u32, usize> = BTreeMap::new();
            let mut seals = 0;
            for b in &d.beams {
                let (a, bb) = (b.a as usize, b.b as usize);
                let (ia, ib) = (np.get(a) == Some(&pi), np.get(bb) == Some(&pi));
                if ia == ib {
                    continue;
                }
                if b.break_group != 0 {
                    *groups.entry(b.break_group).or_default() += 1;
                }
                if b.kind == BeamKind::Support {
                    seals += 1;
                }
            }
            let name_of = |g: u32| {
                d.break_groups
                    .get(g as usize - 1)
                    .cloned()
                    .unwrap_or_default()
            };
            let hinge: Vec<usize> = groups
                .iter()
                .filter(|(g, _)| name_of(**g).contains("hinge"))
                .map(|(_, n)| *n)
                .collect();
            let ok = hinge.len() >= 2 && hinge.iter().all(|&n| (4..=14).contains(&n));
            out.push(Finding {
                rule: "hinges",
                hard: false,
                part: p.name.clone(),
                level: if ok { Level::Pass } else { Level::Warn },
                value: format!("{} groups of {:?} beams", hinge.len(), hinge),
                band: "≥ 2 groups of 4–14".into(),
                offenders: vec![],
            });
            let latch: Vec<f32> = d
                .couplers
                .iter()
                .filter(|cc| {
                    np.get(cc.a as usize) == Some(&pi) || np.get(cc.b as usize) == Some(&pi)
                })
                .map(|cc| cc.strength)
                .collect();
            out.push(Finding {
                rule: "latch",
                hard: false,
                part: p.name.clone(),
                level: if !latch.is_empty() && latch.iter().all(|&s| (35e3..=85e3).contains(&s)) {
                    Level::Pass
                } else {
                    Level::Warn
                },
                value: format!("{} couplers {:?} N", latch.len(), latch),
                band: "≥ 1 at 35–85 kN".into(),
                offenders: vec![],
            });
            out.push(banded(
                "seals_support",
                &p.name,
                seals as f64,
                Some((4.0, 30.0)),
            ));
        }
    }

    // --- Car-wide bands ------------------------------------------------------
    let springs: Vec<_> = d.beams.iter().collect();
    let sup = springs
        .iter()
        .filter(|b| b.kind == BeamKind::Support)
        .count();
    out.push(banded(
        "support_share",
        "car",
        sup as f64 / springs.len().max(1) as f64,
        Some((0.10, 0.17)),
    ));
    let ct: Vec<_> = d.triangles.iter().filter(|t| t.collision).collect();
    out.push(banded(
        "collision_tris",
        "car",
        ct.len() as f64,
        Some((519.0, 611.0)),
    ));
    let mut te: Vec<f64> = ct
        .iter()
        .flat_map(|t| {
            let [a, b, c] = t.nodes.map(|i| pos(i as usize));
            [(a - b).length(), (b - c).length(), (c - a).length()]
        })
        .collect();
    out.push(banded(
        "tri_edge_p50",
        "car",
        q(&mut te, 0.5),
        Some((0.36, 0.54)),
    ));
    // Sliver collision triangles (height under 1/10 of the longest edge):
    // a surface contact's normal there is ill-defined.
    let slivers: Vec<String> = ct
        .iter()
        .filter(|t| {
            let [a, b, c] = t.nodes.map(|i| pos(i as usize));
            let l = (a - b).length().max((b - c).length()).max((c - a).length());
            let h = (b - a).cross(c - a).length() / l.max(1e-9);
            h < 0.1 * l
        })
        .take(1000)
        .map(|t| t.nodes.map(|i| nname(i as usize)).join("-"))
        .collect();
    out.push(Finding {
        rule: "sliver_tris",
        hard: false,
        part: "car".into(),
        level: if slivers.is_empty() {
            Level::Pass
        } else {
            Level::Warn
        },
        value: format!("{} of {}", slivers.len(), ct.len()),
        band: "0 (height < 0.1 × longest edge)".into(),
        offenders: slivers.into_iter().take(6).collect(),
    });
    let mut dc: Vec<f64> = ct.iter().map(|t| t.aero[0] as f64).collect();
    let mut lc: Vec<f64> = ct.iter().map(|t| t.aero[1] as f64).collect();
    out.push(Finding {
        rule: "tri_aero",
        hard: false,
        part: "car".into(),
        level: Level::Na,
        value: format!(
            "dragCoef p50 {}, liftCoef p50 {}",
            fmt(q(&mut dc, 0.5)),
            fmt(q(&mut lc, 0.5))
        ),
        band: "JBeam default 100 / 100".into(),
        offenders: vec![],
    });

    // --- Render mesh bands -----------------------------------------------------
    if let Some(v) = vis {
        let nr: Vec<Vec3> = v.node_rest.iter().map(|p| Vec3::from_array(*p)).collect();
        // Each piece's part: the part most of its weights' nodes belong to.
        let weights = |i: usize| -> Vec<u32> {
            if v.wt_start.len() == v.positions.len() + 1 {
                let (a, b) = (v.wt_start[i] as usize, v.wt_start[i + 1] as usize);
                v.wt_node[a..b].to_vec()
            } else {
                v.bind_nodes.get(i).map_or(vec![], |b| b.to_vec())
            }
        };
        let mut part_area: HashMap<usize, f64> = HashMap::new();
        let mut part_verts: HashMap<usize, Vec<Vec3>> = HashMap::new();
        let mut reach: HashMap<usize, Vec<f64>> = HashMap::new();
        for (_, first, count) in &v.pieces {
            let (f, n) = (*first as usize, *count as usize);
            let mut votes: HashMap<usize, usize> = HashMap::new();
            for i in f..f + n {
                for w in weights(i) {
                    if let Some(&pp) = np.get(w as usize) {
                        *votes.entry(pp).or_default() += 1;
                    }
                }
            }
            let Some((&pp, _)) = votes.iter().max_by_key(|x| (*x.1, std::cmp::Reverse(*x.0)))
            else {
                continue;
            };
            if pp == usize::MAX || Cat::of(&d.parts[pp].name) == Cat::Other {
                continue;
            }
            // Outer painted surface only, for area and distances.
            let painted = (f..f + n).any(|i| v.material.get(i) == Some(&mat::PAINT));
            if !painted {
                continue;
            }
            let verts: Vec<Vec3> = (f..f + n)
                .map(|i| Vec3::from_array(v.positions[i]))
                .collect();
            part_verts.entry(pp).or_default().extend(&verts);
            for i in f..f + n {
                let p = Vec3::from_array(v.positions[i]);
                let ws = weights(i);
                if ws.is_empty() {
                    continue;
                }
                let mut ds: Vec<f64> = ws
                    .iter()
                    .filter_map(|&w| nr.get(w as usize).map(|q| q.distance(p) as f64))
                    .collect();
                reach.entry(pp).or_default().push(q(&mut ds, 0.5));
            }
            let _ = &mut part_area;
        }
        // Area: the pieces' triangles (indices are per vertex buffer).
        for t in v.indices.chunks_exact(3) {
            let i = t[0] as usize;
            if v.material.get(i) != Some(&mat::PAINT) {
                continue;
            }
            let ws = weights(i);
            let Some(&w) = ws.first() else { continue };
            let Some(&pp) = np.get(w as usize) else {
                continue;
            };
            let [a, b, c] = [t[0], t[1], t[2]].map(|k| Vec3::from_array(v.positions[k as usize]));
            *part_area.entry(pp).or_default() += 0.5 * (b - a).cross(c - a).length() as f64;
        }
        for (pi, p) in d.parts.iter().enumerate() {
            let c = Cat::of(&p.name);
            if c == Cat::Other {
                continue;
            }
            let structural: Vec<usize> = p
                .nodes
                .iter()
                .map(|&i| i as usize)
                .filter(|&i| i < nr.len() && np[i] == pi && !contact_only(i))
                .collect();
            // Painted area counts both faces of a closure's sheet (outer and
            // inner panel): halve it.
            let area = part_area.get(&pi).copied().unwrap_or(0.0) * 0.5;
            if area > 0.0 {
                out.push(banded(
                    "nodes_per_m2",
                    &p.name,
                    structural.len() as f64 / area,
                    band("per_m2", c),
                ));
            }
            if let Some(vs) = part_verts.get(&pi) {
                // A node's distance to its part's skin (nearest vertex).
                let mut dd: Vec<f64> = structural
                    .iter()
                    .map(|&i| {
                        vs.iter()
                            .map(|q| q.distance(nr[i]))
                            .fold(f32::MAX, f32::min) as f64
                    })
                    .collect();
                out.push(banded(
                    "node_to_skin_p50",
                    &p.name,
                    q(&mut dd, 0.5),
                    band("node_dist", c),
                ));
            }
            if let Some(r) = reach.get_mut(&pi) {
                out.push(banded(
                    "vertex_reach_p50",
                    &p.name,
                    q(r, 0.5),
                    band("reach", c),
                ));
            }
        }
    }
    let _ = HashSet::<u8>::new();
    out
}

/// Print findings as a table; returns the number of hard failures.
pub fn print(name: &str, f: &[Finding]) -> usize {
    println!("\n## {name}\n");
    println!("| rule | part | level | value | band | offenders |");
    println!("|---|---|---|---|---|---|");
    let mut hard = 0;
    for x in f {
        if x.hard && x.level == Level::Fail {
            hard += 1;
        }
        println!(
            "| {}{} | {} | {} | {} | {} | {} |",
            x.rule,
            if x.hard { " (hard)" } else { "" },
            x.part,
            x.level.tag(),
            x.value,
            x.band,
            x.offenders.join("; ")
        );
    }
    let warns = f.iter().filter(|x| x.level == Level::Warn).count();
    let pass = f.iter().filter(|x| x.level == Level::Pass).count();
    println!("\n{name}: {hard} hard failures, {warns} warnings, {pass} passes");
    hard
}
