//! Why the generated sedan looks like jelly beside the ETK (a measurement,
//! CPU reference only). Per car and per part category (body, doors, hood,
//! trunk, fenders, bumpers):
//!
//! 1. static authoring numbers (nodes, node mass, k·L, spring ends, SUPPORT share,
//!    per-node rate √(Σk/m));
//! 2. the explicit bound ω·h (power iteration on the loaded springs) and, for the
//!    generated sedan, the stability pass's softening (built k against the rule's k);
//! 3. lowest linear modes: each closure on the car (its neighbours held), the body
//!    shell free-free, and the whole car free-free (bilateral springs and couplers;
//!    SUPPORT, ropes, shocks and torsion bars left out);
//! 4. dynamics from the crash spawn: settle after spawn, a 0.2 g lateral and a
//!    0.2 g vertical 100 ms half-sine body load, and a 5 cm bump at 30 km/h: per
//!    category ripple (crash_invariants' measure: RMS of √2·σ per 0.1 s window of
//!    each node about its part's rigid fit, 1 kHz), the same about the body's fit,
//!    internal kinetic energy and its decay;
//! 5. the visible mesh (the car's visual, flexskin for the generated sedan) about
//!    the body's fit, the same windows.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example sedan_stiffness -- [sedan] [etkk] [sedan1] [--no-dyn] [--no-modes]
//! ```
//! `etkk` is community content read from `__mods/etkk` (local only, never committed).
use engine_softbody::authoring::skin::{deform_beams, unit_states, BeamState};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitKind, JbeamWheels, StaticBox, GRAVITY, NODE_TYRE, STEP,
};
use engine_softbody::vehicle_compare::{rigid_fit, settings, Car, CarSpec, Source};
use glam::{DVec3, Quat, Vec3};
use std::collections::BTreeMap;

static NAMES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

const CATS: [&str; 6] = ["body", "door", "hood", "trunk", "fender", "bumper"];

fn cat_of(part: &str) -> &'static str {
    let p = part.to_lowercase();
    if p.contains("door") {
        "door"
    } else if p.contains("hood") || p.contains("bonnet") {
        "hood"
    } else if p.contains("trunk") || p.contains("boot") || p.contains("decklid") || p.contains("tailgate") {
        "trunk"
    } else if p.contains("fender") || p.contains("wing") {
        "fender"
    } else if p.contains("bumper") {
        "bumper"
    } else if p.ends_with("body") || p == "body" || p.contains("_body") || p.contains("shell") {
        "body"
    } else {
        "other"
    }
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// A linear spring for the modal model: nodes, unit direction, k, isotropic.
#[derive(Clone, Copy)]
struct Spring {
    a: usize,
    b: usize,
    d: DVec3,
    k: f64,
    iso: bool,
}

fn bilateral(kind: ExplicitKind) -> Option<bool> {
    match kind {
        ExplicitKind::Plain
        | ExplicitKind::LimitedSpring
        | ExplicitKind::JbeamPlain
        | ExplicitKind::JbeamBounded
        | ExplicitKind::JbeamAniso => Some(false),
        ExplicitKind::JbeamCoupler => Some(true),
        ExplicitKind::Support | ExplicitKind::JbeamSupport
            if std::env::var("MODES_SUPPORT").is_ok_and(|v| v == "1") =>
        {
            Some(false)
        }
        _ => None,
    }
}

fn springs_of(s: &ExplicitBeams, k_override: Option<&[f32]>) -> Vec<Spring> {
    let mut out = Vec::new();
    for (j, b) in s.beams.iter().enumerate() {
        if b.broken {
            continue;
        }
        let Some(iso) = bilateral(b.kind) else { continue };
        let k = k_override.map(|o| o[j]).unwrap_or(b.k) as f64;
        if !(k > 0.0) || !k.is_finite() {
            continue;
        }
        let (a, c) = (b.a as usize, b.b as usize);
        let d = (s.nodes[c].pos - s.nodes[a].pos).as_dvec3();
        let d = if d.length() > 1e-6 { d.normalize() } else { DVec3::X };
        out.push(Spring { a, b: c, d, k, iso });
    }
    out
}

/// Lowest eigen-frequencies (Hz) of the free nodes `free` (indices into
/// `mass`), springs to nodes outside `free` held at the far end (`held`) or
/// dropped (`!held`). Returns (freq, per mode the share of kinetic energy per
/// label of the free nodes).
fn modes(
    mass: &[f64],
    springs: &[Spring],
    free: &[usize],
    held: bool,
    count: usize,
    label: &[usize],
    nlabels: usize,
) -> Vec<(f64, Vec<f64>)> {
    let nf = free.len();
    let n = 3 * nf;
    if n == 0 {
        return vec![];
    }
    let mut at = vec![usize::MAX; mass.len()];
    for (k, &i) in free.iter().enumerate() {
        at[i] = k;
    }
    let mut kmat = vec![0.0f64; n * n];
    for sp in springs {
        let (ia, ib) = (at[sp.a], at[sp.b]);
        if ia == usize::MAX && ib == usize::MAX {
            continue;
        }
        if (ia == usize::MAX || ib == usize::MAX) && !held {
            continue;
        }
        let dd = sp.d.to_array();
        let blk = |p: usize, q: usize| -> f64 {
            if sp.iso {
                if p == q {
                    sp.k
                } else {
                    0.0
                }
            } else {
                sp.k * dd[p] * dd[q]
            }
        };
        for p in 0..3 {
            for q in 0..3 {
                let v = blk(p, q);
                if ia != usize::MAX {
                    kmat[(3 * ia + p) * n + 3 * ia + q] += v;
                }
                if ib != usize::MAX {
                    kmat[(3 * ib + p) * n + 3 * ib + q] += v;
                }
                if ia != usize::MAX && ib != usize::MAX {
                    kmat[(3 * ia + p) * n + 3 * ib + q] -= v;
                    kmat[(3 * ib + p) * n + 3 * ia + q] -= v;
                }
            }
        }
    }
    let ms: Vec<f64> = (0..n).map(|r| mass[free[r / 3]].max(1e-3).sqrt()).collect();
    for r in 0..n {
        for c in 0..n {
            kmat[r * n + c] /= ms[r] * ms[c];
        }
    }
    // Shift so rigid modes are invertible: σ = (2π·0.3 Hz)².
    let sigma = (std::f64::consts::TAU * 0.3).powi(2);
    let mut l = kmat.clone();
    for r in 0..n {
        l[r * n + r] += sigma;
    }
    // Cholesky (lower, in place).
    for j in 0..n {
        let mut d = l[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        let d = d.max(1e-12).sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut v = l[i * n + j];
            for k in 0..j {
                v -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = v / d;
        }
    }
    let solve = |b: &mut [f64]| {
        for i in 0..n {
            let mut v = b[i];
            for k in 0..i {
                v -= l[i * n + k] * b[k];
            }
            b[i] = v / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut v = b[i];
            for k in i + 1..n {
                v -= l[k * n + i] * b[k];
            }
            b[i] = v / l[i * n + i];
        }
    };
    let p = (count + 8).min(n);
    let mut x: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            (0..n)
                .map(|i| (((i * 7919 + j * 104729) % 1000) as f64 / 1000.0) - 0.5)
                .collect()
        })
        .collect();
    let ortho = |x: &mut Vec<Vec<f64>>| {
        for j in 0..x.len() {
            for k in 0..j {
                let d: f64 = (0..n).map(|i| x[j][i] * x[k][i]).sum();
                for i in 0..n {
                    x[j][i] -= d * x[k][i];
                }
            }
            let nn: f64 = x[j].iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
            for v in x[j].iter_mut() {
                *v /= nn;
            }
        }
    };
    ortho(&mut x);
    let iters = if n > 1500 { 60 } else { 120 };
    for _ in 0..iters {
        for v in x.iter_mut() {
            solve(v);
        }
        ortho(&mut x);
    }
    // Rayleigh–Ritz on the subspace with S = kmat.
    let sx: Vec<Vec<f64>> = x
        .iter()
        .map(|v| {
            (0..n)
                .map(|r| (0..n).map(|c| kmat[r * n + c] * v[c]).sum())
                .collect()
        })
        .collect();
    let mut h = vec![0.0f64; p * p];
    for a in 0..p {
        for b in 0..p {
            h[a * p + b] = (0..n).map(|i| x[a][i] * sx[b][i]).sum();
        }
    }
    let (ev, vecs) = jacobi(&mut h, p);
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_by(|&a, &b| ev[a].total_cmp(&ev[b]));
    idx.into_iter()
        .take(count)
        .map(|m| {
            let f = ev[m].max(0.0).sqrt() / std::f64::consts::TAU;
            // Mode shape (mass-weighted coords) → kinetic share per label.
            let mut sh = vec![0.0f64; nlabels];
            let mut per_node = vec![0.0f64; nf];
            for i in 0..n {
                let y: f64 = (0..p).map(|a| vecs[a * p + m] * x[a][i]).sum();
                let lb = label[free[i / 3]];
                if lb < nlabels {
                    sh[lb] += y * y;
                }
                per_node[i / 3] += y * y;
            }
            let t: f64 = sh.iter().sum::<f64>().max(1e-300);
            // Participation ratio: the share of the free nodes that move (1: all alike).
            let s2: f64 = per_node.iter().sum();
            let s4: f64 = per_node.iter().map(|v| v * v).sum();
            let pr = s2 * s2 / (nf as f64 * s4.max(1e-300));
            // MODE_NODES=1: the nodes that carry each mode (share of its kinetic energy).
            if f >= 1.0 && std::env::var("MODE_NODES").is_ok() {
                let names = NAMES.lock().unwrap();
                let mut v: Vec<(f64, usize)> = per_node.iter().enumerate().map(|(k, &e)| (e / s2.max(1e-300), free[k])).collect();
                v.sort_by(|a, b| b.0.total_cmp(&a.0));
                eprintln!(
                    "  mode {f:.1} Hz (of {nf} nodes): {}",
                    v.iter().take(5).map(|&(e, i)| format!("{} {:.2}", names.get(i).map_or("?", |s| s.as_str()), e)).collect::<Vec<_>>().join(", ")
                );
            }
            let mut out: Vec<f64> = sh.into_iter().map(|v| v / t).collect();
            out.push(pr);
            (f, out)
        })
        .collect()
}

fn jacobi(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[p * n + q] * a[p * n + q])
            .sum();
        if off < 1e-20 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let th = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
                let t = if th == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// Highest ω·h of the loaded springs (power iteration, k only).
fn omega_h(s: &ExplicitBeams, k_override: Option<&[f32]>) -> f64 {
    let sp = springs_of(s, k_override);
    let n = s.nodes.len();
    let m: Vec<f64> = s.nodes.iter().map(|q| q.mass as f64).collect();
    let mut x: Vec<DVec3> = (0..n)
        .map(|i| DVec3::new((i % 7) as f64 - 3.0, (i % 5) as f64 - 2.0, (i % 3) as f64 - 1.5))
        .collect();
    let mut lam = 0.0;
    for _ in 0..600 {
        let mut y = vec![DVec3::ZERO; n];
        for q in &sp {
            let f = if q.iso {
                (x[q.a] - x[q.b]) * q.k
            } else {
                q.d * (q.k * q.d.dot(x[q.a] - x[q.b]))
            };
            y[q.a] += f;
            y[q.b] -= f;
        }
        for i in 0..n {
            y[i] /= m[i];
        }
        let ny = y.iter().map(|v| v.length_squared()).sum::<f64>().sqrt();
        let nx = x.iter().map(|v| v.length_squared()).sum::<f64>().sqrt();
        lam = ny / nx.max(1e-300);
        x = y.into_iter().map(|v| v / ny.max(1e-300)).collect();
    }
    lam.sqrt() * STEP as f64
}


/// The crash_invariants panel strike: the part cut out of the car at rest, its
/// neighbours immovable, every beam touching it (SUPPORT, torsion, couplers
/// included; JBeam extension beams skipped), struck at 0.1 m/s along its
/// thinnest axis: (f Hz, ζ, decay to 1 % ms, first 0.2 s amplitude mm).
fn strike(s: &ExplicitBeams, ids: &[u32]) -> (f64, f64, f64, f64) {
    use engine_softbody::explicit_beams::{ExplicitBeamsSettings, Node, NODE_IMMOVABLE, NODE_NO_GROUND};
    let inside: std::collections::HashSet<usize> = ids.iter().map(|&i| i as usize).collect();
    let mut map: std::collections::HashMap<usize, u32> = Default::default();
    let mut nodes: Vec<Node> = Vec::new();
    let g = GRAVITY;
    let mut add = |i: usize, fixed: bool, nodes: &mut Vec<Node>| -> u32 {
        *map.entry(i).or_insert_with(|| {
            let mut q = s.nodes[i];
            q.vel = Vec3::ZERO;
            q.flags = NODE_NO_GROUND | if fixed { NODE_IMMOVABLE } else { 0 };
            q.force = Vec3::new(0.0, q.mass * g, 0.0);
            nodes.push(q);
            nodes.len() as u32 - 1
        })
    };
    for &i in ids {
        add(i as usize, false, &mut nodes);
    }
    let mut beams = Vec::new();
    for (j, bm) in s.beams.iter().enumerate() {
        let (a, c) = (bm.a as usize, bm.b as usize);
        if !(inside.contains(&a) || inside.contains(&c)) {
            continue;
        }
        if !s.ext_of.is_empty() && s.ext_of[j] != 0 {
            continue;
        }
        if matches!(bm.kind, ExplicitKind::JbeamTorsion | ExplicitKind::JbeamTorsionPair) {
            continue;
        }
        let mut nb = *bm;
        nb.a = add(a, !inside.contains(&a), &mut nodes);
        nb.b = add(c, !inside.contains(&c), &mut nodes);
        nb.shock = 0;
        beams.push(nb);
    }
    let mut iso = ExplicitBeams::empty(ExplicitBeamsSettings {
        fast_inv_sqrt: s.settings.fast_inv_sqrt,
        drag: false,
        turbulence: false,
        ..s.settings
    });
    iso.ground = -1.0e3;
    iso.nodes = nodes;
    iso.beams = beams;
    iso.finish();
    let m_n = ids.len();
    let p: Vec<Vec3> = ids.iter().map(|&i| s.nodes[i as usize].pos).collect();
    let (lo, hi) = p.iter().fold((Vec3::MAX, Vec3::MIN), |(l, h), q| (l.min(*q), h.max(*q)));
    let ext = hi - lo;
    let axis = if ext.x <= ext.y && ext.x <= ext.z { 0 } else if ext.y <= ext.z { 1 } else { 2 };
    let dir = Vec3::AXES[axis];
    for k in 0..m_n {
        iso.nodes[k].vel = dir * 0.1;
    }
    let m: f64 = (0..m_n).map(|k| iso.nodes[k].mass as f64).sum();
    let mut y = Vec::new();
    for _ in 0..4000 {
        iso.step();
        y.push((0..m_n).map(|k| iso.nodes[k].mass as f64 * iso.nodes[k].pos.dot(dir) as f64).sum::<f64>() / m);
    }
    let mean = y[2000..].iter().sum::<f64>() / 2000.0;
    let y: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let x: Vec<f64> = (1..y.len()).filter(|&i| y[i - 1] < 0.0 && y[i] >= 0.0).map(|i| (i - 1) as f64 + y[i - 1] / (y[i - 1] - y[i])).collect();
    let f = if x.len() > 2 { (x.len() - 1) as f64 / ((x[x.len() - 1] - x[0]) * STEP as f64) } else { f64::NAN };
    let pk: Vec<(f64, f64)> = (1..y.len() - 1)
        .filter(|&i| y[i] > y[i - 1] && y[i] >= y[i + 1] && y[i] > 1e-7)
        .map(|i| (i as f64 * STEP as f64, y[i].ln()))
        .collect();
    let n = pk.len() as f64;
    let sigma = if pk.len() > 2 {
        let (mx, my) = pk.iter().fold((0.0, 0.0), |a, q| (a.0 + q.0 / n, a.1 + q.1 / n));
        -pk.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum::<f64>() / pk.iter().map(|q| (q.0 - mx).powi(2)).sum::<f64>()
    } else {
        f64::NAN
    };
    let zeta = sigma / (std::f64::consts::TAU * f);
    let amp0 = y.iter().take(400).map(|v| v.abs()).fold(0.0, f64::max);
    (f, zeta, 1e3 * (100f64).ln() / sigma, amp0 * 1e3)
}


/// The generated sedan's stability pass: each beam's built k against the
/// rule's k (`Role::spec`, the census k·L over max(L, SPRING_LENGTH)), the bound
/// with the rule's k, and the closures' modes both ways (from the rig in memory).
fn rig_report(design: &str) {
    use engine_softbody::authoring::gen2::build::{stiff_mult, Part, Role, H, SPRING_LENGTH};
    use engine_softbody::authoring::gen2::rig::hub_extra;
    use engine_softbody::authoring::rig;
    let built = rig::rig_vehicle(design, None, None).unwrap();
    let c = &built.rig.car;
    let hub = hub_extra(c);
    let n = c.nodes.len();
    let mass: Vec<f64> = (0..n).map(|i| (c.nodes[i].mass + hub.get(&i).copied().unwrap_or(0.0)) as f64).collect();
    let cat = |p: Part| -> usize {
        match p {
            Part::Body => 0,
            Part::DoorFL | Part::DoorFR | Part::DoorRL | Part::DoorRR => 1,
            Part::Hood => 2,
            Part::Trunk => 3,
            Part::FenderL | Part::FenderR => 4,
            Part::BumperF | Part::BumperR => 5,
            Part::Engine => 6,
        }
    };
    let mult = stiff_mult();
    let nominal: Vec<f32> = c
        .beams
        .iter()
        .map(|b| {
            let sp = b.role.spec(b.part);
            let l = c.p(b.a).distance(c.p(b.b)).max(0.02);
            mult * if sp.k_fixed > 0.0 { sp.k_fixed } else { sp.kl / l.max(SPRING_LENGTH) }
        })
        .collect();
    let live = |j: usize| {
        let b = &c.beams[j];
        !b.role.spec(b.part).support && !matches!(b.role, Role::Stop) && nominal[j] > 1.0
    };
    println!("\n## {design}: the stability pass (built k against the rule's k)\n");
    println!("rig ω·h {:.3} (centred damping {}; the bound {:.2})", c.omega_h, c.centred_damping, if c.centred_damping { 1.92 } else { 1.85 });
    println!("\n| cat | spring beams | softened < 0.9 | < 0.5 | ratio p10 / p50 / p90 | Σk built / Σk rule | k·L built p50 / rule p50 (N) | most softened roles (count < 0.9) |");
    println!("|---|---|---|---|---|---|---|---|");
    for (ci, cname) in CATS.iter().chain(["engine"].iter()).enumerate() {
        let js: Vec<usize> = (0..c.beams.len()).filter(|&j| live(j) && cat(c.beams[j].part) == ci).collect();
        if js.is_empty() {
            continue;
        }
        let mut r: Vec<f64> = js.iter().map(|&j| (c.beams[j].k / nominal[j]) as f64).collect();
        let s09 = r.iter().filter(|&&x| x < 0.9).count();
        let s05 = r.iter().filter(|&&x| x < 0.5).count();
        let sb: f64 = js.iter().map(|&j| c.beams[j].k as f64).sum();
        let sn: f64 = js.iter().map(|&j| nominal[j] as f64).sum();
        let len = |j: usize| c.p(c.beams[j].a).distance(c.p(c.beams[j].b)) as f64;
        let mut klb: Vec<f64> = js.iter().map(|&j| c.beams[j].k as f64 * len(j)).collect();
        let mut kln: Vec<f64> = js.iter().map(|&j| nominal[j] as f64 * len(j)).collect();
        let mut by: BTreeMap<String, usize> = BTreeMap::new();
        for &j in &js {
            if c.beams[j].k < 0.9 * nominal[j] {
                *by.entry(format!("{:?}", c.beams[j].role)).or_default() += 1;
            }
        }
        let mut by: Vec<(String, usize)> = by.into_iter().collect();
        by.sort_by(|a, b| b.1.cmp(&a.1));
        println!(
            "| {cname} | {} | {} ({:.0} %) | {} | {:.2} / {:.2} / {:.2} | {:.2} | {:.2e} / {:.2e} | {} |",
            js.len(),
            s09,
            100.0 * s09 as f64 / js.len() as f64,
            s05,
            pct(&mut r, 0.1),
            pct(&mut r, 0.5),
            pct(&mut r, 0.9),
            sb / sn,
            pct(&mut klb, 0.5),
            pct(&mut kln, 0.5),
            by.iter().take(4).map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", ")
        );
    }
    // Softening against node mass: beams whose lighter node is under 0.3 kg vs the rest.
    for (lo, hi) in [(0.0, 0.3), (0.3, 1.0), (1.0, 1e9)] {
        let js: Vec<usize> = (0..c.beams.len())
            .filter(|&j| live(j) && {
                let m = mass[c.beams[j].a].min(mass[c.beams[j].b]);
                m >= lo && m < hi
            })
            .collect();
        let sb: f64 = js.iter().map(|&j| c.beams[j].k as f64).sum();
        let sn: f64 = js.iter().map(|&j| nominal[j] as f64).sum();
        let s09 = js.iter().filter(|&&j| c.beams[j].k < 0.9 * nominal[j]).count();
        println!(
            "lighter node {lo}–{hi} kg: {} beams, {} softened < 0.9 ({:.0} %), Σk built / rule {:.2}",
            js.len(),
            s09,
            100.0 * s09 as f64 / js.len().max(1) as f64,
            sb / sn.max(1e-9)
        );
    }
    // Springs both ways, couplers as the pass models them.
    let mk = |use_nominal: bool| -> Vec<Spring> {
        let mut v: Vec<Spring> = (0..c.beams.len())
            .filter(|&j| live(j))
            .map(|j| {
                let b = &c.beams[j];
                let d = (c.p(b.b) - c.p(b.a)).as_dvec3();
                let d = if d.length() > 1e-6 { d.normalize() } else { DVec3::X };
                Spring { a: b.a, b: b.b, d, k: if use_nominal { nominal[j] } else { b.k } as f64, iso: false }
            })
            .collect();
        for i in 0..n {
            let tag = c.nodes[i].opts.iter().find(|(o, _)| o == "couplerTag").and_then(|(_, v)| v.as_str().map(str::to_string));
            if let Some(t) = tag {
                if let Some(o) = (0..n).find(|&o| c.nodes[o].opts.iter().any(|(k, v)| k == "tag" && v.as_str() == Some(&t))) {
                    let mu = mass[i] * mass[o] / (mass[i] + mass[o]);
                    v.push(Spring { a: i, b: o, d: DVec3::X, k: 0.96 * mu / (H as f64 * H as f64), iso: true });
                }
            }
        }
        v
    };
    let (sb, sn) = (mk(false), mk(true));
    let wh = |sp: &[Spring]| {
        let mut x: Vec<DVec3> = (0..n).map(|i| DVec3::new((i % 7) as f64 - 3.0, (i % 5) as f64 - 2.0, (i % 3) as f64 - 1.5)).collect();
        let mut lam = 0.0;
        for _ in 0..800 {
            let mut y = vec![DVec3::ZERO; n];
            for q in sp {
                let f = if q.iso { (x[q.a] - x[q.b]) * q.k } else { q.d * (q.k * q.d.dot(x[q.a] - x[q.b])) };
                y[q.a] += f;
                y[q.b] -= f;
            }
            for i in 0..n {
                y[i] /= mass[i];
            }
            let ny = y.iter().map(|v| v.length_squared()).sum::<f64>().sqrt();
            let nx = x.iter().map(|v| v.length_squared()).sum::<f64>().sqrt();
            lam = ny / nx.max(1e-300);
            x = y.into_iter().map(|v| v / ny.max(1e-300)).collect();
        }
        lam.sqrt() * H as f64
    };
    println!("\nω·h with the built springs {:.3}; with the rule's springs (no stability pass) {:.3}", wh(&sb), wh(&sn));
    let label: Vec<usize> = (0..n).map(|i| cat(c.nodes[i].part).min(6)).collect();
    println!("\n| part | nodes | kg | first three modes ≥ 1 Hz, built k | the same, rule k (no softening) |");
    println!("|---|---|---|---|---|");
    for part in Part::ALL {
        if part == Part::Engine {
            continue;
        }
        let free: Vec<usize> = (0..n).filter(|&i| c.nodes[i].part == part).collect();
        let kg: f64 = free.iter().map(|&i| mass[i]).sum();
        let held = part != Part::Body;
        let cnt = if held { 10 } else { 24 };
        let f = |sp: &[Spring]| -> String {
            let m = modes(&mass, sp, &free, held, cnt, &label, 7);
            let el: Vec<String> = m
                .iter()
                .filter(|x| x.0 >= 1.0)
                .take(3)
                .map(|x| format!("{:.1} (PR {:.2})", x.0, x.1.last().unwrap()))
                .collect();
            let g = if held { String::new() } else {
                m.iter().find(|x| x.0 >= 1.0 && *x.1.last().unwrap() >= 0.2).map(|x| format!("; first global {:.1}", x.0)).unwrap_or_default()
            };
            format!("{}{g}", el.join(", "))
        };
        println!("| {:?}{} | {} | {:.1} | {} | {} |", part, if held { "" } else { " (free-free)" }, free.len(), kg, f(&sb), f(&sn));
    }
}

// --- Spawn (the crash path, as vehicle_compare's, visual indices match) ----------

fn spawn(car: &Car) -> ExplicitBeams {
    let (mut s, angle) = match &car.source {
        Source::Authored(b) => {
            let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
            let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            let s = ExplicitBeams::from_truck(
                &b.truck,
                Vec3::new(0.0, probe.spawn_clearance() - low, 0.0),
                Vec3::ZERO,
                settings(),
            )
            .unwrap();
            (s, 0.0)
        }
        Source::Jbeam { desc, .. } if std::env::var("RAYCAST").is_ok_and(|v| v == "1") => {
            let (s, _) = ExplicitBeams::from_desc_jbeam_raycast(desc, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
            let f = engine_softbody::pack::forward_axis(desc);
            (s, -f.x.atan2(f.z))
        }
        Source::Jbeam { desc, .. } => {
            let (s, _) = ExplicitBeams::from_desc_jbeam_with(
                desc,
                JbeamWheels::Pressure,
                Vec3::ZERO,
                Vec3::ZERO,
                settings(),
            )
            .unwrap();
            let f = engine_softbody::pack::forward_axis(desc);
            (s, -f.x.atan2(f.z))
        }
    };
    s.surface_contact = car.template.surface_contact;
    // place(): turn, centre on the skin box, lowest node 2 cm up.
    let o = s.origin;
    for n in &mut s.nodes {
        n.pos += o;
    }
    s.origin = Vec3::ZERO;
    let q = Quat::from_rotation_y(angle);
    for n in &mut s.nodes {
        n.pos = q * n.pos;
        n.vel = q * n.vel;
    }
    let skin = &car.skin;
    let (lo, hi) = skin.iter().map(|&i| s.nodes[i as usize].pos).fold(
        (Vec3::MAX, Vec3::MIN),
        |(a, b), p| (a.min(p), b.max(p)),
    );
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mid = 0.5 * (lo + hi);
    for n in &mut s.nodes {
        n.pos.x -= mid.x;
        n.pos.z -= mid.z;
        n.pos.y -= low - 0.02;
    }
    s
}

// --- Dynamics measurement --------------------------------------------------------

struct Sets {
    /// Per category: (part name, nodes) of its parts.
    cats: Vec<Vec<(String, Vec<u32>)>>,
    body: Vec<u32>,
    nontyre: Vec<u32>,
    /// Visual vertices (subsampled) per category.
    vcats: Vec<Vec<usize>>,
}

#[derive(Default, Clone)]
struct Series {
    /// Per category per 0.1 s window: ripple (own part fit), about the body fit.
    own: Vec<Vec<f64>>,
    body: Vec<Vec<f64>>,
    vis: Vec<Vec<f64>>,
    /// Internal kinetic energy (J) every 10 ms, and the time.
    ke: Vec<f64>,
}

fn internal_ke(s: &ExplicitBeams, ids: &[u32]) -> f64 {
    let mut m = 0.0;
    let mut c = DVec3::ZERO;
    let mut p = DVec3::ZERO;
    for &i in ids {
        let n = &s.nodes[i as usize];
        let mi = n.mass as f64;
        m += mi;
        c += n.pos.as_dvec3() * mi;
        p += n.vel.as_dvec3() * mi;
    }
    c /= m;
    let v = p / m;
    let mut l = DVec3::ZERO;
    let mut inertia = [[0.0f64; 3]; 3];
    for &i in ids {
        let n = &s.nodes[i as usize];
        let mi = n.mass as f64;
        let r = n.pos.as_dvec3() - c;
        l += r.cross(n.vel.as_dvec3() - v) * mi;
        let rr = r.length_squared();
        let ra = r.to_array();
        for a in 0..3 {
            for b in 0..3 {
                inertia[a][b] += mi * (if a == b { rr } else { 0.0 } - ra[a] * ra[b]);
            }
        }
    }
    let im = glam::DMat3::from_cols_array_2d(&inertia).transpose();
    let w = if im.determinant().abs() > 1e-12 { im.inverse() * l } else { DVec3::ZERO };
    let mut e = 0.0;
    for &i in ids {
        let n = &s.nodes[i as usize];
        let r = n.pos.as_dvec3() - c;
        let u = n.vel.as_dvec3() - v - w.cross(r);
        e += 0.5 * n.mass as f64 * u.length_squared();
    }
    e
}

/// Window accumulator: per node its samples (in some frame).
struct Win(Vec<Vec<Vec3>>);
impl Win {
    fn new(n: usize) -> Win {
        Win(vec![Vec::new(); n])
    }
    fn amp(&mut self) -> f64 {
        if self.0.is_empty() || self.0[0].is_empty() {
            return 0.0;
        }
        let a2: f64 = self
            .0
            .iter()
            .map(|v| {
                let m = v.iter().map(|p| p.as_dvec3()).sum::<DVec3>() / v.len() as f64;
                2.0 * v.iter().map(|d| (d.as_dvec3() - m).length_squared()).sum::<f64>() / v.len() as f64
            })
            .sum::<f64>()
            / self.0.len() as f64;
        for v in self.0.iter_mut() {
            v.clear();
        }
        a2.sqrt()
    }
}

/// Run `seconds` from `s`, applying `accel(t)` (m/s², every node) before each
/// step; measure.
fn run(
    car: &Car,
    sets: &Sets,
    s: &mut ExplicitBeams,
    seconds: f32,
    accel: &dyn Fn(f32) -> Vec3,
    vis: bool,
) -> Series {
    let nc = CATS.len();
    let rest: Vec<Vec3> = car.template.nodes.iter().map(|n| n.pos).collect();
    let mut out = Series {
        own: vec![Vec::new(); nc],
        body: vec![Vec::new(); nc],
        vis: vec![Vec::new(); nc],
        ke: Vec::new(),
    };
    let mut w_own: Vec<Vec<Win>> = sets
        .cats
        .iter()
        .map(|parts| parts.iter().map(|(_, ids)| Win::new(ids.len())).collect())
        .collect();
    let mut w_body: Vec<Vec<Win>> = sets
        .cats
        .iter()
        .map(|parts| parts.iter().map(|(_, ids)| Win::new(ids.len())).collect())
        .collect();
    let mut w_vis: Vec<Win> = sets.vcats.iter().map(|v| Win::new(v.len())).collect();
    let rest_beams: Vec<BeamState> = car
        .template
        .beams
        .iter()
        .map(|b| BeamState { broken: b.broken, rest: b.rest })
        .collect();
    let v = car.visual.as_ref().filter(|_| vis);
    // The visual at the template (its rest frame for the body fit).
    let vrest: Option<Vec<Vec3>> = v.map(|v| {
        let u = unit_states(v, &rest_beams);
        deform_beams(v, &rest, Some(&u), Some(&rest_beams)).0
    });
    let _ = vrest;
    let steps = (seconds / STEP).round() as usize;
    for k in 0..steps {
        let t = k as f32 * STEP;
        let a = accel(t);
        if a != Vec3::ZERO {
            for n in s.nodes.iter_mut() {
                n.force += a * n.mass;
            }
        }
        s.step();
        if k % 2 == 1 {
            let now: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
            let fb = rigid_fit(&rest, &now, &sets.body);
            for (c, parts) in sets.cats.iter().enumerate() {
                for (pi, (_, ids)) in parts.iter().enumerate() {
                    let fo = rigid_fit(&rest, &now, ids);
                    for (j, &i) in ids.iter().enumerate() {
                        let p = now[i as usize];
                        w_own[c][pi].0[j].push(fo.to_rest(p));
                        w_body[c][pi].0[j].push(fb.to_rest(p));
                    }
                }
            }
            if let Some(v) = v {
                if k % 10 == 9 {
                    let beams: Vec<BeamState> = s
                        .beams
                        .iter()
                        .map(|b| BeamState { broken: b.broken, rest: b.rest })
                        .collect();
                    let u = unit_states(v, &beams);
                    let (pv, _) = deform_beams(v, &now, Some(&u), Some(&beams));
                    for (c, verts) in sets.vcats.iter().enumerate() {
                        for (j, &i) in verts.iter().enumerate() {
                            if pv[i].is_finite() {
                                w_vis[c].0[j].push(fb.to_rest(pv[i]));
                            }
                        }
                    }
                }
            }
        }
        if k % 20 == 19 {
            out.ke.push(internal_ke(s, &sets.nontyre));
        }
        if k % 200 == 199 {
            for c in 0..nc {
                let n: usize = w_own[c].len();
                if n == 0 {
                    out.own[c].push(f64::NAN);
                    out.body[c].push(f64::NAN);
                } else {
                    // RMS over the category's parts' nodes.
                    let mut so = 0.0;
                    let mut sb = 0.0;
                    let mut cnt = 0.0;
                    for (pi, wo) in w_own[c].iter_mut().enumerate() {
                        let m = wo.0.len() as f64;
                        let ao = wo.amp();
                        let ab = w_body[c][pi].amp();
                        so += ao * ao * m;
                        sb += ab * ab * m;
                        cnt += m;
                    }
                    out.own[c].push((so / cnt).sqrt());
                    out.body[c].push((sb / cnt).sqrt());
                }
                if v.is_some() && !w_vis[c].0.is_empty() {
                    // Vertices whose samples were NaN have short series: amp() handles any length.
                    w_vis[c].0.retain(|_| true);
                    let a = {
                        let wv = &mut w_vis[c];
                        let good: Vec<usize> = (0..wv.0.len()).filter(|&j| wv.0[j].len() >= 2).collect();
                        if good.is_empty() {
                            for x in wv.0.iter_mut() {
                                x.clear();
                            }
                            f64::NAN
                        } else {
                            let a2: f64 = good
                                .iter()
                                .map(|&j| {
                                    let vv = &wv.0[j];
                                    let m = vv.iter().map(|p| p.as_dvec3()).sum::<DVec3>() / vv.len() as f64;
                                    2.0 * vv.iter().map(|d| (d.as_dvec3() - m).length_squared()).sum::<f64>()
                                        / vv.len() as f64
                                })
                                .sum::<f64>()
                                / good.len() as f64;
                            for x in wv.0.iter_mut() {
                                x.clear();
                            }
                            a2.sqrt()
                        }
                    };
                    out.vis[c].push(a);
                } else {
                    out.vis[c].push(f64::NAN);
                }
            }
        }
    }
    out
}

/// Time (s, from `from`) after which a per-window series stays below `thr`.
fn settle_time(series: &[f64], from: usize, thr: f64) -> f64 {
    let mut last = None;
    for (i, &a) in series.iter().enumerate().skip(from) {
        if a.is_finite() && a > thr {
            last = Some(i);
        }
    }
    match last {
        None => 0.0,
        Some(i) => (i + 1 - from) as f64 * 0.1,
    }
}

fn mm(x: f64) -> String {
    if x.is_finite() {
        format!("{:.2}", x * 1e3)
    } else {
        "-".into()
    }
}

fn report_series(label: &str, s: &Series, from: usize) {
    println!("\n#### {label}\n");
    println!("| cat | ripple own-fit peak / +0.3 s / +1 s / end (mm) | settle < 0.1 mm (s) | about body peak / +0.3 / +1 / end (mm) | settle < 0.2 mm (s) | visible mesh about body peak / +0.3 / +1 / end (mm) | vis settle < 0.2 mm (s) |");
    println!("|---|---|---|---|---|---|---|");
    for (c, name) in CATS.iter().enumerate() {
        let o = &s.own[c];
        if o.is_empty() || o.iter().all(|x| x.is_nan()) {
            continue;
        }
        let pk = |v: &[f64]| v.iter().skip(from).copied().filter(|x| x.is_finite()).fold(0.0, f64::max);
        let at = |v: &[f64], dt: usize| v.get(from + dt).copied().unwrap_or(f64::NAN);
        let end = |v: &[f64]| v.last().copied().unwrap_or(f64::NAN);
        let b = &s.body[c];
        let vi = &s.vis[c];
        println!(
            "| {name} | {} / {} / {} / {} | {:.1} | {} / {} / {} / {} | {:.1} | {} / {} / {} / {} | {} |",
            mm(pk(o)), mm(at(o, 3)), mm(at(o, 10)), mm(end(o)),
            settle_time(o, from, 1e-4),
            mm(pk(b)), mm(at(b, 3)), mm(at(b, 10)), mm(end(b)),
            settle_time(b, from, 2e-4),
            mm(pk(vi)), mm(at(vi, 3)), mm(at(vi, 10)), mm(end(vi)),
            if vi.iter().any(|x| x.is_finite()) { format!("{:.1}", settle_time(vi, from, 2e-4)) } else { "-".into() },
        );
    }
    // Internal KE: peak after `from`, time to 1 % and 10 % of it.
    let k0 = from * 10;
    let ke = &s.ke;
    if ke.len() > k0 {
        let (ip, pk) = ke.iter().enumerate().skip(k0).fold((k0, 0.0f64), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        let t_to = |f: f64| {
            let mut last = ip;
            for (i, &v) in ke.iter().enumerate().skip(ip) {
                if v > f * pk {
                    last = i;
                }
            }
            (last + 1 - ip) as f64 * 0.01
        };
        println!(
            "\ninternal KE (car minus its rigid motion): peak {:.2} J at +{:.2} s; to 10 % in {:.2} s, to 1 % in {:.2} s; end {:.3} J",
            pk,
            (ip - k0) as f64 * 0.01,
            t_to(0.1),
            t_to(0.01),
            ke.last().unwrap()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let no_dyn = args.iter().any(|a| a == "--no-dyn");
    let only_rest = args.iter().any(|a| a == "--rest-only");
    let no_modes = args.iter().any(|a| a == "--no-modes");
    if args.iter().any(|a| a == "--rig") {
        rig_report("sedan");
        return;
    }
    let mut names: Vec<String> = args.iter().filter(|a| !a.starts_with("--")).cloned().collect();
    if names.is_empty() {
        names = vec!["sedan".into(), "etkk".into(), "sedan1".into()];
    }
    for name in names {
        let spec: CarSpec = match name.as_str() {
            "sedan1" => serde_json::from_value(serde_json::json!({"name": "sedan1", "style": "sedan1"})).unwrap(),
            "etkk" => serde_json::from_value(serde_json::json!({"name": "etkk", "mod": "etkk", "vehicle": "etkk", "pc": "kr6.pc"})).unwrap(),
            // `label=vehicle,config.pc`: any JBeam vehicle under `<mods>/<label>/`
            // (`RUBBLE_COMPARE_MODS` sets the mods folder), e.g. `ccf=ccf,f4_sport_M.pc`.
            n if n.contains('=') => {
                let (label, rest) = n.split_once('=').unwrap();
                let (vehicle, pc) = rest.split_once(',').unwrap_or((rest, ""));
                serde_json::from_value(serde_json::json!({"name": label, "mod": label, "vehicle": vehicle, "pc": pc})).unwrap()
            }
            n => serde_json::from_value(serde_json::json!({"name": n, "mod": n, "vehicle": n, "pc": format!("{n}.pc")})).unwrap(),
        };
        let t0 = std::time::Instant::now();
        let car = match Car::load(&spec) {
            Ok(c) => c,
            Err(e) => {
                println!("## {name}: load failed: {e}");
                continue;
            }
        };
        eprintln!("{name}: loaded in {:.1} s", t0.elapsed().as_secs_f32());
        let s = &car.template;
        let nn = s.nodes.len();
        *NAMES.lock().unwrap() = (0..nn).map(|i| car.node_name(i).to_string()).collect();
        let tyre = |i: usize| s.nodes[i].flags & NODE_TYRE != 0;
        // Category per node (first part wins).
        let mut cat = vec![usize::MAX; nn];
        let mut cats: Vec<Vec<(String, Vec<u32>)>> = vec![Vec::new(); CATS.len()];
        for (pname, ids) in &car.parts {
            let c = cat_of(pname);
            let Some(ci) = CATS.iter().position(|x| *x == c) else { continue };
            // Wheel hubs and rims riding the body part (the generated sedan's) are running gear, not shell.
            let ids: Vec<u32> = ids
                .iter()
                .copied()
                .filter(|&i| cat[i as usize] == usize::MAX)
                .filter(|&i| {
                    let nm = car.node_name(i as usize);
                    !(ci == 0 && (nm.starts_with("rim") || nm.starts_with("hub")))
                })
                .collect();
            if ids.len() < 3 {
                continue;
            }
            for &i in &ids {
                cat[i as usize] = ci;
            }
            cats[ci].push((pname.clone(), ids));
        }
        println!("\n## {name}\n");
        println!(
            "{} nodes ({} tyre), {} beams, {:.0} kg; parts: {}",
            nn,
            (0..nn).filter(|&i| tyre(i)).count(),
            s.beams.len(),
            car.mass(),
            car.parts.iter().map(|(p, v)| format!("{p}[{}]→{}", v.len(), cat_of(p))).collect::<Vec<_>>().join(", ")
        );
        // 1. Static numbers per category.
        let support = |k: ExplicitKind| matches!(k, ExplicitKind::Support | ExplicitKind::JbeamSupport);
        let mut ends = vec![0usize; nn];
        let mut sumk = vec![0.0f64; nn];
        for b in &s.beams {
            if b.k > 0.0 && b.kind != ExplicitKind::Trigger && !b.broken {
                ends[b.a as usize] += 1;
                ends[b.b as usize] += 1;
                if bilateral(b.kind).is_some() {
                    sumk[b.a as usize] += b.k as f64;
                    sumk[b.b as usize] += b.k as f64;
                }
            }
        }
        println!("\n### Authoring numbers\n");
        println!("| cat | parts | nodes (per part) | kg | node kg p50 | k·L p50 (N) | k p50 (N/m) | L p50 (m) | spring ends p50 | SUPPORT share | ζ p50 | node rate √(Σk/m) p50 (Hz) |");
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
        for (ci, cname) in CATS.iter().enumerate() {
            let ids: Vec<usize> = (0..nn).filter(|&i| cat[i] == ci).collect();
            if ids.is_empty() {
                continue;
            }
            let kg: f64 = ids.iter().map(|&i| s.nodes[i].mass as f64).sum();
            let mut nm: Vec<f64> = ids.iter().map(|&i| s.nodes[i].mass as f64).collect();
            let mut kl = Vec::new();
            let mut kk = Vec::new();
            let mut ll = Vec::new();
            let mut zeta = Vec::new();
            let (mut sup, mut all) = (0usize, 0usize);
            for b in &s.beams {
                let (a, c) = (b.a as usize, b.b as usize);
                if cat[a] != ci && cat[c] != ci {
                    continue;
                }
                if b.kind == ExplicitKind::Trigger || b.k <= 0.0 {
                    continue;
                }
                all += 1;
                if support(b.kind) {
                    sup += 1;
                    continue;
                }
                if cat[a] == ci && bilateral(b.kind) == Some(false) {
                    kl.push((b.k * b.rest0) as f64);
                    kk.push(b.k as f64);
                    ll.push(b.rest0 as f64);
                    let (ma, mb) = (s.nodes[a].mass as f64, s.nodes[c].mass as f64);
                    let mu = ma * mb / (ma + mb);
                    zeta.push(b.c as f64 / (2.0 * (b.k as f64 * mu).sqrt()));
                }
            }
            let mut se: Vec<f64> = ids.iter().map(|&i| ends[i] as f64).collect();
            let mut rate: Vec<f64> = ids
                .iter()
                .map(|&i| (sumk[i] / s.nodes[i].mass as f64).sqrt() / std::f64::consts::TAU)
                .collect();
            println!(
                "| {cname} | {} | {} ({}) | {:.1} | {:.2} | {:.2e} | {:.2e} | {:.3} | {:.0} | {:.1} % | {:.3} | {:.0} |",
                cats[ci].len(),
                ids.len(),
                cats[ci].iter().map(|(_, v)| v.len().to_string()).collect::<Vec<_>>().join("/"),
                kg,
                pct(&mut nm, 0.5),
                pct(&mut kl, 0.5),
                pct(&mut kk, 0.5),
                pct(&mut ll, 0.5),
                pct(&mut se, 0.5),
                100.0 * sup as f64 / all.max(1) as f64,
                pct(&mut zeta, 0.5),
                pct(&mut rate, 0.5),
            );
        }
        let support_all = s.beams.iter().filter(|b| support(b.kind)).count();
        println!(
            "\nwhole car: SUPPORT {} of {} beams ({:.1} %); ω·h (springs only, power iteration) {:.3}",
            support_all,
            s.beams.len(),
            100.0 * support_all as f64 / s.beams.len() as f64,
            omega_h(s, None)
        );
        // 3. Modes.
        if !no_modes {
            let mass: Vec<f64> = s.nodes.iter().map(|n| n.mass as f64).collect();
            let sp = springs_of(s, None);
            let mut label = vec![CATS.len(); nn];
            for i in 0..nn {
                if cat[i] < CATS.len() {
                    label[i] = cat[i];
                }
            }
            println!("\n### Lowest linear modes (Hz; SUPPORT, ropes, shocks, torsion bars left out)\n");
            println!("| part | nodes | kg | held at neighbours: modes < 1 Hz | first three ≥ 1 Hz | strike (all beams): f Hz, ζ, to 1 % ms, amp mm |");
            println!("|---|---|---|---|---|---|");
            for (ci, parts) in cats.iter().enumerate() {
                if ci == 0 {
                    continue;
                }
                for (pname, ids) in parts {
                    let free: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
                    let kg: f64 = free.iter().map(|&i| mass[i]).sum();
                    let m = modes(&mass, &sp, &free, true, 12, &label, CATS.len() + 1);
                    let zero = m.iter().filter(|x| x.0 < 1.0).count();
                    let el: Vec<String> = m.iter().filter(|x| x.0 >= 1.0).take(3).map(|x| format!("{:.1} (PR {:.2})", x.0, x.1.last().unwrap())).collect();
                    let (sf, sz, sd, sa) = strike(s, ids);
                    println!("| {pname} | {} | {:.1} | {} | {} | {sf:.1}, {sz:.3}, {sd:.0}, {sa:.2} |", free.len(), kg, zero, el.join(", "));
                }
            }
            // Body shell free-free.
            for (pname, ids) in &cats[0] {
                let free: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
                let kg: f64 = free.iter().map(|&i| mass[i]).sum();
                let m = modes(&mass, &sp, &free, false, 24, &label, CATS.len() + 1);
                let zero = m.iter().filter(|x| x.0 < 1.0).count();
                let el: Vec<String> = m.iter().filter(|x| x.0 >= 1.0).take(5).map(|x| format!("{:.1} (PR {:.2})", x.0, x.1.last().unwrap())).collect();
                // The first mode that moves at least a fifth of the shell (a global mode).
                let g = m.iter().find(|x| x.0 >= 1.0 && *x.1.last().unwrap() >= 0.2).map(|x| format!("{:.1}", x.0)).unwrap_or("-".into());
                println!("| {pname}: first global mode (PR ≥ 0.2) | | | | {g} |");
                println!("| {pname} (free-free, its own beams) | {} | {:.1} | {} (6 rigid) | {} |", free.len(), kg, zero, el.join(", "));
            }
            // Whole car free-free (non-tyre nodes).
            let free: Vec<usize> = (0..nn).filter(|&i| !tyre(i)).collect();
            let kg: f64 = free.iter().map(|&i| mass[i]).sum();
            let m = modes(&mass, &sp, &free, false, 60, &label, CATS.len() + 1);
            let zero = m.iter().filter(|x| x.0 < 1.0).count();
            let el: Vec<String> = m
                .iter()
                .filter(|x| x.0 >= 1.0 && *x.1.last().unwrap() >= 0.15)
                .take(6)
                .map(|x| {
                    let sh = &x.1[..x.1.len() - 1];
                    let (bi, bv) = sh.iter().enumerate().fold((0, 0.0), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
                    let nm = if bi < CATS.len() { CATS[bi] } else { "other" };
                    format!("{:.1} ({nm} {:.0} %, PR {:.2})", x.0, bv * 100.0, x.1.last().unwrap())
                })
                .collect();
            let local = m.iter().filter(|x| x.0 >= 1.0 && *x.1.last().unwrap() < 0.15).count();
            println!("| whole car (free-free, no tyres): global modes (PR ≥ 0.15) | {} | {:.1} | {} (local modes below them: {local}) | {} |", free.len(), kg, zero, el.join(", "));
        }
        if no_dyn {
            continue;
        }
        // 4–5. Dynamics.
        let body: Vec<u32> = cats[0].iter().flat_map(|(_, v)| v.iter().copied()).collect();
        let nontyre: Vec<u32> = (0..nn as u32).filter(|&i| !tyre(i as usize)).collect();
        let mut vcats: Vec<Vec<usize>> = vec![Vec::new(); CATS.len()];
        if let Some(v) = &car.visual {
            for (pname, first, count) in &v.pieces {
                let c = cat_of(pname.split(':').next().unwrap_or(pname));
                let c2 = if c == "other" { cat_of(pname) } else { c };
                let Some(ci) = CATS.iter().position(|x| *x == c2) else { continue };
                let lname = pname.to_lowercase();
                if lname.contains("glass") || lname.contains("window") || lname.contains("lamp") || lname.contains("light") {
                    continue;
                }
                for i in (*first..first + count).step_by(7) {
                    if v.wheel.get(i as usize).is_some_and(|&w| w != u32::MAX) {
                        continue;
                    }
                    vcats[ci].push(i as usize);
                }
            }
            eprintln!(
                "visual: {} vertices; sampled per category {:?}",
                v.positions.len(),
                vcats.iter().map(|x| x.len()).collect::<Vec<_>>()
            );
        }
        let sets = Sets { cats: cats.clone(), body, nontyre, vcats };
        let zero = |_: f32| Vec3::ZERO;
        // (a) Spawn and settle 4 s.
        let mut sp = spawn(&car);
        let t1 = std::time::Instant::now();
        let vis_on = !std::env::var("RAYCAST").is_ok_and(|v| v == "1");
        let r = run(&car, &sets, &mut sp, 4.0, &zero, vis_on);
        eprintln!("{name}: spawn run {:.1} s", t1.elapsed().as_secs_f32());
        report_series(&format!("{name}: spawn (2 cm drop on its own wheels), settle 4 s"), &r, 3);
        // showcase_rest_wobble-style: RMS / max of each node about its own mean over 3–4 s (world frame).
        {
            let mut snap = sp.clone();
            let n = snap.nodes.len();
            let mut sum = vec![Vec3::ZERO; n];
            let mut samples = Vec::new();
            for k in 0..2000 {
                snap.step();
                if k % 20 == 0 {
                    let p: Vec<Vec3> = snap.nodes.iter().map(|x| x.pos).collect();
                    for i in 0..n {
                        sum[i] += p[i];
                    }
                    samples.push(p);
                }
            }
            let f = samples.len() as f32;
            let mean: Vec<Vec3> = sum.iter().map(|v| *v / f).collect();
            let (mut sq, mut dmax) = (0.0f64, 0.0f32);
            for p in &samples {
                for &i in &sets.nontyre {
                    let d = p[i as usize].distance(mean[i as usize]);
                    sq += (d * d) as f64;
                    dmax = dmax.max(d);
                }
            }
            println!(
                "\nrest wobble 4–5 s (each non-tyre node about its own mean, world frame): RMS {:.3} mm, max {:.2} mm",
                (sq / (sets.nontyre.len() as f64 * f as f64)).sqrt() * 1e3,
                dmax * 1e3
            );
        }

        // Where the rest wobble lives: each node's amplitude (√2·σ about the body's fit, 4–5 s).
        {
            let mut snap = sp.clone();
            let rest: Vec<Vec3> = car.template.nodes.iter().map(|n| n.pos).collect();
            let mut w = Win::new(nn);
            for k in 0..2000 {
                snap.step();
                if k % 4 == 0 {
                    let now: Vec<Vec3> = snap.nodes.iter().map(|n| n.pos).collect();
                    let fb = rigid_fit(&rest, &now, &sets.body);
                    for i in 0..now.len() {
                        w.0[i].push(fb.to_rest(now[i]));
                    }
                }
            }
            let amp: Vec<f64> = w
                .0
                .iter()
                .map(|v| {
                    let m = v.iter().map(|p| p.as_dvec3()).sum::<DVec3>() / v.len() as f64;
                    (2.0 * v.iter().map(|d| (d.as_dvec3() - m).length_squared()).sum::<f64>() / v.len() as f64).sqrt()
                })
                .collect();
            for (lo, hi) in [(0.0f32, 0.3f32), (0.3, 1.0), (1.0, 3.0), (3.0, 1e9)] {
                let mut v: Vec<f64> = sets.nontyre.iter().map(|&i| i as usize).filter(|&i| s.nodes[i].mass >= lo && s.nodes[i].mass < hi).map(|i| amp[i]).collect();
                let n = v.len();
                if n == 0 { continue; }
                println!("rest 4–5 s, nodes {lo}–{hi} kg: {n}, amplitude about body p50 {:.2} / p90 {:.2} / max {:.2} mm", pct(&mut v, 0.5) * 1e3, pct(&mut v, 0.9) * 1e3, pct(&mut v, 1.0) * 1e3);
            }
            let mut idx: Vec<usize> = sets.nontyre.iter().map(|&i| i as usize).collect();
            idx.sort_by(|&a, &b| amp[b].total_cmp(&amp[a]));
            let part_of = |i: usize| car.parts.iter().find(|(_, v)| v.contains(&(i as u32))).map(|(p, _)| p.clone()).unwrap_or_default();
            println!("top nodes: {}", idx.iter().take(10).map(|&i| format!("{} {} {:.2} kg {:.1} mm at ({:.2},{:.2},{:.2})", car.node_name(i), part_of(i), s.nodes[i].mass, amp[i] * 1e3, rest[i].x, rest[i].y, rest[i].z)).collect::<Vec<_>>().join("; "));
        }
        if only_rest {
            continue;
        }
        // Settled state at 4 s for the load cases.
        let base = sp;
        let g = -GRAVITY;
        for (label, dir) in [("0.2 g lateral half-sine, 100 ms", Vec3::X), ("0.2 g vertical (down) half-sine, 100 ms", Vec3::NEG_Y)] {
            let mut s2 = base.clone();
            let pulse = move |t: f32| {
                if t < 0.1 {
                    dir * (0.2 * g * (std::f32::consts::PI * t / 0.1).sin())
                } else {
                    Vec3::ZERO
                }
            };
            let r = run(&car, &sets, &mut s2, 2.5, &pulse, vis_on);
            report_series(&format!("{name}: {label} (from settled), 2.5 s"), &r, 0);
        }
        // 5 cm bump at 30 km/h (both axles cross it).
        {
            let mut s2 = base.clone();
            let v = Vec3::Z * (30.0 / 3.6);
            for n in s2.nodes.iter_mut() {
                n.vel += v;
            }
            s2.roll_tyres(v);
            if let Some(d) = s2.drive.as_mut() {
                d.roll(v.z);
            }
            let front = s2.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
            // Bump 0.5 m long, 5 cm high, 0.6 m ahead of the front bumper.
            let z0 = front + 0.6;
            s2.boxes.push(StaticBox::centered(Vec3::new(0.0, -0.45, z0 + 0.25), Vec3::new(4.0, 0.5, 0.25)));
            let r = run(&car, &sets, &mut s2, 3.0, &zero, vis_on);
            report_series(&format!("{name}: 5 cm × 0.5 m bump at 30 km/h (coasting), 3 s"), &r, 0);
            // Beams the bump bent (rest length changed by more than 0.1 mm) or broke, by member pair.
            let mut bent: BTreeMap<String, (usize, f32)> = BTreeMap::new();
            let (mut nb, mut nbr) = (0, 0);
            let member = |n: &str| n.trim_end_matches(['L', 'R']).trim_end_matches(|c: char| c.is_ascii_digit()).trim_end_matches('_').to_string();
            for b in &s2.beams {
                if b.broken {
                    nbr += 1;
                }
                let d = (b.rest - b.rest0).abs();
                if d > 1e-4 {
                    nb += 1;
                    let (ma, mb) = (member(&car.node_name(b.a as usize)), member(&car.node_name(b.b as usize)));
                    let e = bent.entry(if ma <= mb { format!("{ma}-{mb}") } else { format!("{mb}-{ma}") }).or_default();
                    e.0 += 1;
                    e.1 = e.1.max(d);
                }
            }
            let mut v: Vec<(&String, &(usize, f32))> = bent.iter().collect();
            v.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
            println!(
                "\nbump: {nb} beams bent, {nbr} broken; worst: {}",
                v.iter().take(14).map(|(m, (n, d))| format!("{m} ×{n} {:.1} mm", d * 1e3)).collect::<Vec<_>>().join(", ")
            );
        }
        eprintln!("{name}: done in {:.0} s", t0.elapsed().as_secs_f32());
    }
}
