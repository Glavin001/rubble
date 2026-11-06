//! gen3 at rest, in memory (no files): rigs the render model, loads the JBeam
//! through the loader, parks it on raycast (and pressure) wheels, and prints
//! each part's drift from design about the body's rigid fit, with broken and
//! bent beams and kinetic energy.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example gen3_rest -- [seconds] [raycast|pressure|both]
//! ```
use engine_softbody::authoring::gen2::{export, model, rig};
use engine_softbody::explicit_beams::{ExplicitBeams, JbeamWheels};
use engine_softbody::vehicle_compare::{rigid_fit, settings};
use glam::Vec3;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let secs: f32 = a.first().and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let which = a.get(1).cloned().unwrap_or_else(|| "raycast".into());
    let t0 = std::time::Instant::now();
    let m = model::build();
    let r = rig::rig(&m);
    eprintln!(
        "rigged in {:.1} s; omega_h {:.3}",
        t0.elapsed().as_secs_f32(),
        r.car.omega_h
    );
    if std::env::var("GEN3_SOFT").is_ok() {
        use engine_softbody::authoring::gen2::build::{Part, SPRING_LENGTH};
        for part in Part::ALL {
            let mut v: Vec<(f32, String)> = r
                .car
                .beams
                .iter()
                .filter(|b| b.part == part && b.k > 1.0)
                .map(|b| {
                    let sp = b.role.spec(b.part);
                    let l = r.car.p(b.a).distance(r.car.p(b.b)).max(0.02);
                    let nom = if sp.k_fixed > 0.0 {
                        sp.k_fixed
                    } else {
                        sp.kl / l.max(SPRING_LENGTH)
                    };
                    (
                        b.k / nom,
                        format!(
                            "{}-{} {:?}",
                            r.car.nodes[b.a].name, r.car.nodes[b.b].name, b.role
                        ),
                    )
                })
                .collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            let n = v.len();
            let soft = v.iter().filter(|x| x.0 < 0.9).count();
            v.retain(|x| !x.1.contains("Tie") && !x.1.contains("Backing"));
            let n = v.len();
            println!(
                "{:?}: {n} beams, {soft} softened below 0.9, p10 {:.2} p50 {:.2}; softest {:?}",
                part,
                v[n / 10].0,
                v[n / 2].0,
                &v[..8.min(n)]
            );
        }
    }
    if let Ok(nm) = std::env::var("GEN3_NODE") {
        let i = r.car.id(&nm);
        println!("{nm}: {:.2} kg at {:?}", r.car.nodes[i].mass, r.car.p(i));
        for b in r.car.beams.iter().filter(|b| b.a == i || b.b == i) {
            let o = if b.a == i { b.b } else { b.a };
            println!(
                "  - {} {:?} k {:.3e} L {:.3}",
                r.car.nodes[o].name,
                b.role,
                b.k,
                r.car.p(i).distance(r.car.p(o))
            );
        }
    }
    if std::env::var("GEN3_MODES").is_ok() {
        use engine_softbody::authoring::gen2::build::Part;
        for part in Part::ALL {
            let ids: Vec<usize> = (0..r.car.nodes.len())
                .filter(|&i| r.car.nodes[i].part == part)
                .collect();
            let n = ids.len() * 3;
            if n > 600 {
                continue;
            }
            let at: std::collections::HashMap<usize, usize> =
                ids.iter().enumerate().map(|(k, &i)| (i, k)).collect();
            let mut k = vec![0.0f64; n * n];
            for b in &r.car.beams {
                let (Some(&ia), Some(&ib)) = (at.get(&b.a), at.get(&b.b)) else {
                    continue;
                };
                if b.role.spec(b.part).support || b.k <= 1.0 {
                    continue;
                }
                let d = (r.car.p(b.b) - r.car.p(b.a)).normalize().as_dvec3();
                let dd = [d.x, d.y, d.z];
                for p in 0..3 {
                    for q in 0..3 {
                        let v = b.k as f64 * dd[p] * dd[q];
                        k[(3 * ia + p) * n + 3 * ia + q] += v;
                        k[(3 * ib + p) * n + 3 * ib + q] += v;
                        k[(3 * ia + p) * n + 3 * ib + q] -= v;
                        k[(3 * ib + p) * n + 3 * ia + q] -= v;
                    }
                }
            }
            let m: Vec<f64> = ids
                .iter()
                .map(|&i| r.car.nodes[i].mass.max(0.05) as f64)
                .collect();
            for a in 0..n {
                for b in 0..n {
                    k[a * n + b] /= (m[a / 3] * m[b / 3]).sqrt();
                }
            }
            // Jacobi eigenvalues.
            let mut v = vec![0.0f64; n * n];
            for a in 0..n {
                v[a * n + a] = 1.0;
            }
            for _sweep in 0..30 {
                let mut off = 0.0;
                for p in 0..n {
                    for q in p + 1..n {
                        off += k[p * n + q] * k[p * n + q];
                    }
                }
                if off < 1e-12 {
                    break;
                }
                for p in 0..n {
                    for q in p + 1..n {
                        let apq = k[p * n + q];
                        if apq.abs() < 1e-14 {
                            continue;
                        }
                        let th = (k[q * n + q] - k[p * n + p]) / (2.0 * apq);
                        let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
                        let t = if th == 0.0 { 1.0 } else { t };
                        let c = 1.0 / (t * t + 1.0).sqrt();
                        let sn = t * c;
                        for r2 in 0..n {
                            let (a, b) = (k[r2 * n + p], k[r2 * n + q]);
                            k[r2 * n + p] = c * a - sn * b;
                            k[r2 * n + q] = sn * a + c * b;
                        }
                        for c2 in 0..n {
                            let (a, b) = (k[p * n + c2], k[q * n + c2]);
                            k[p * n + c2] = c * a - sn * b;
                            k[q * n + c2] = sn * a + c * b;
                        }
                        for r2 in 0..n {
                            let (a, b) = (v[r2 * n + p], v[r2 * n + q]);
                            v[r2 * n + p] = c * a - sn * b;
                            v[r2 * n + q] = sn * a + c * b;
                        }
                    }
                }
            }
            let mut ev: Vec<(f64, usize)> = (0..n).map(|a| (k[a * n + a], a)).collect();
            ev.sort_by(|a, b| a.0.total_cmp(&b.0));
            let desc: Vec<String> = ev
                .iter()
                .skip(6)
                .take(4)
                .map(|&(l, col)| {
                    let mut amp: Vec<(f64, usize)> = (0..ids.len())
                        .map(|q| {
                            (
                                (0..3)
                                    .map(|c| v[(3 * q + c) * n + col].powi(2))
                                    .sum::<f64>(),
                                q,
                            )
                        })
                        .collect();
                    amp.sort_by(|a, b| b.0.total_cmp(&a.0));
                    let top: Vec<String> = amp
                        .iter()
                        .take(3)
                        .map(|&(_, q)| r.car.nodes[ids[q]].name.clone())
                        .collect();
                    format!(
                        "{:.1} Hz [{}]",
                        l.max(0.0).sqrt() / std::f64::consts::TAU,
                        top.join(" ")
                    )
                })
                .collect();
            let zeros = ev.iter().filter(|e| e.0 < 1.0).count();
            println!(
                "{:?}: {} nodes, near-zero {} (rigid 6), lowest internal: {}",
                part,
                ids.len(),
                zeros,
                desc.join("; ")
            );
        }
    }
    let files = export::jbeam_files(&r.car, &[], &[]);
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pc = files
        .iter()
        .find(|f| f.0.ends_with(".pc"))
        .map(|f| f.1.clone());
    let jb: Vec<(&str, &str)> = refs
        .iter()
        .copied()
        .filter(|f| f.0.ends_with(".jbeam"))
        .collect();
    let desc = engine_softbody::jbeam::load_jbeam(&jb, pc.as_deref())?;
    let design: Vec<Vec3> = desc.nodes.iter().map(|n| Vec3::from(n.position)).collect();
    let nd = design.len();
    let susp = |name: &str| name.starts_with("hub") || name.starts_with("rim");
    let fit_ids: Vec<u32> = (0..nd as u32)
        .filter(|&i| !susp(&desc.nodes[i as usize].name))
        .collect();
    let modes: Vec<(&str, JbeamWheels)> = match which.as_str() {
        "pressure" => vec![("pressure", JbeamWheels::Pressure)],
        "both" => vec![
            ("raycast", JbeamWheels::Raycast),
            ("pressure", JbeamWheels::Pressure),
        ],
        _ => vec![("raycast", JbeamWheels::Raycast)],
    };
    for (label, wheels) in modes {
        let (mut s, _) =
            ExplicitBeams::from_desc_jbeam_with(&desc, wheels, Vec3::ZERO, Vec3::ZERO, settings())?;
        s.surface_contact = std::env::var("SURFACE_CONTACT").map_or(true, |v| v != "0");
        let low = s
            .nodes
            .iter()
            .map(|n| s.origin.y + n.pos.y)
            .fold(f32::MAX, f32::min);
        for n in &mut s.nodes {
            n.pos.y += 0.02 - low;
        }
        let mut t = 0.0f32;
        let trace = std::env::var("GEN3_TRACE").is_ok();
        let mut k = 0;
        while t < secs - 1e-6 {
            s.step();
            t += 0.0005;
            k += 1;
            if k % 1000 == 0 && std::env::var("GEN3_KE").is_ok() {
                let (mut ke, mut m, mut p) = (0.0f32, 0.0f32, Vec3::ZERO);
                for n in &s.nodes {
                    ke += 0.5 * n.mass * n.vel.length_squared();
                    m += n.mass;
                    p += n.mass * n.vel;
                }
                let v = p / m;
                println!(
                    "  t {t:.1} s: KE {ke:.4} J, of it the mean motion {:.4} J (v {:.4} {:.4} {:.4})",
                    0.5 * m * v.length_squared(),
                    v.x,
                    v.y,
                    v.z
                );
            }
            if trace && (k <= 60 || k % 2000 == 0) {
                let (i, v) = s
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (i, n.vel.length()))
                    .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
                let nm = desc
                    .nodes
                    .get(i)
                    .map_or(format!("#{i}"), |n| n.name.clone());
                println!("  step {k}: fastest {nm} {v:.3} m/s");
            }
        }
        let now: Vec<Vec3> = (0..nd).map(|i| s.origin + s.nodes[i].pos).collect();
        let fit = rigid_fit(&design, &now, &fit_ids);
        let d: Vec<Vec3> = (0..nd).map(|i| fit.to_rest(now[i]) - design[i]).collect();
        let broken = s.beams.iter().filter(|b| b.broken).count();
        let bent = s.beams.iter().filter(|b| b.bent()).count();
        let ke: f32 = s
            .nodes
            .iter()
            .map(|n| 0.5 * n.mass * n.vel.length_squared())
            .sum();
        println!("[{label}] parked {secs} s: broken {broken}, bent {bent}, KE {ke:.3} J");
        // `GEN3_KE`: where the kinetic energy is (the fastest nodes, the parts).
        if std::env::var("GEN3_KE").is_ok() {
            let nm = |i: usize| {
                desc.nodes
                    .get(i)
                    .map_or(format!("#{i}"), |n| n.name.clone())
            };
            let mut v: Vec<(f32, usize)> = s
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (0.5 * n.mass * n.vel.length_squared(), i))
                .collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(e, i) in v.iter().take(12) {
                println!(
                    "    ke {} {:.4} J ({:.3} m/s, {:.2} kg)",
                    nm(i),
                    e,
                    s.nodes[i].vel.length(),
                    s.nodes[i].mass
                );
            }
            for p in desc.parts.iter().filter(|p| !p.nodes.is_empty()) {
                let e: f32 = p
                    .nodes
                    .iter()
                    .filter(|&&i| (i as usize) < s.nodes.len())
                    .map(|&i| {
                        let n = &s.nodes[i as usize];
                        0.5 * n.mass * n.vel.length_squared()
                    })
                    .sum();
                println!("    ke part {} {:.4} J", p.name, e);
            }
        }
        if std::env::var("GEN3_DUMP").is_ok() {
            for b in s.beams.iter().filter(|b| b.broken || b.bent()).take(30) {
                let nm = |i: u32| {
                    desc.nodes
                        .get(i as usize)
                        .map_or(format!("#{i}"), |n| n.name.clone())
                };
                println!(
                    "    {} {}-{} rest0 {:.3} rest {:.3}",
                    if b.broken { "broken" } else { "bent" },
                    nm(b.a),
                    nm(b.b),
                    b.rest0,
                    b.rest
                );
            }
        }
        if let Ok(pfx) = std::env::var("GEN3_STRAIN") {
            let nm = |i: u32| {
                desc.nodes
                    .get(i as usize)
                    .map_or(format!("#{i}"), |n| n.name.clone())
            };
            let mut v: Vec<(f32, String)> = s
                .beams
                .iter()
                .filter(|b| b.k > 0.0 && (nm(b.a).starts_with(&pfx) || nm(b.b).starts_with(&pfx)))
                .map(|b| {
                    let l = (s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos).length();
                    (
                        l - b.rest,
                        format!(
                            "{}-{} dL {:+.2} mm k {:.2e}",
                            nm(b.a),
                            nm(b.b),
                            (l - b.rest) * 1e3,
                            b.k
                        ),
                    )
                })
                .collect();
            v.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
            for x in v.iter().take(12) {
                println!("    strain {}", x.1);
            }
        }
        let mut rows: Vec<(f32, String)> = desc
            .parts
            .iter()
            .filter(|p| !p.nodes.is_empty())
            .map(|p| {
                let mut v: Vec<(f32, usize)> = p
                    .nodes
                    .iter()
                    .filter(|&&i| (i as usize) < nd && !susp(&desc.nodes[i as usize].name))
                    .map(|&i| (d[i as usize].length(), i as usize))
                    .collect();
                v.sort_by(|x, y| y.0.total_cmp(&x.0));
                let worst: Vec<String> = v
                    .iter()
                    .take(3)
                    .map(|&(mm, i)| {
                        format!(
                            "{} {:.1} (dz {:+.1})",
                            desc.nodes[i].name,
                            mm * 1e3,
                            d[i].z * 1e3
                        )
                    })
                    .collect();
                let mx = v.first().map_or(0.0, |x| x.0);
                // Its own warp: about its own nodes' rigid fit.
                let own: Vec<u32> = p
                    .nodes
                    .iter()
                    .copied()
                    .filter(|&i| (i as usize) < nd && !susp(&desc.nodes[i as usize].name))
                    .collect();
                let warp = if own.len() >= 3 {
                    let f = rigid_fit(&design, &now, &own);
                    own.iter()
                        .map(|&i| (f.to_rest(now[i as usize]) - design[i as usize]).length())
                        .fold(0.0, f32::max)
                } else {
                    0.0
                };
                (
                    mx,
                    format!(
                        "  {:<16} max {:6.1} mm  warp {:5.1} mm  worst: {}",
                        p.name,
                        mx * 1e3,
                        warp * 1e3,
                        worst.join(", ")
                    ),
                )
            })
            .collect();
        rows.sort_by(|x, y| y.0.total_cmp(&x.0));
        if let Ok(lim) = std::env::var("GEN3_DUMP") {
            let lim: f32 = lim.parse().unwrap_or(5.0) * 1e-3;
            let mut ends = vec![0usize; nd];
            for b in &desc.beams {
                if (b.a as usize) < nd && (b.b as usize) < nd {
                    ends[b.a as usize] += 1;
                    ends[b.b as usize] += 1;
                }
            }
            let mut v: Vec<usize> = (0..nd)
                .filter(|&i| d[i].length() > lim && !susp(&desc.nodes[i].name))
                .collect();
            v.sort_by(|&x, &y| d[y].length().total_cmp(&d[x].length()));
            for i in v.iter().take(40) {
                println!(
                    "    {} {:.1} mm d ({:+.1},{:+.1},{:+.1}) ends {} at {:?}",
                    desc.nodes[*i].name,
                    d[*i].length() * 1e3,
                    d[*i].x * 1e3,
                    d[*i].y * 1e3,
                    d[*i].z * 1e3,
                    ends[*i],
                    design[*i]
                );
            }
        }
        for (_, r) in rows {
            println!("{r}");
        }
    }
    Ok(())
}
