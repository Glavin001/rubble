//! A rigid-wall frontal for one car on the CPU reference, measured as a crash lab
//! would: the car settles 1 s on its raycast wheels, then every node gets `kmh=`
//! (default 56) toward an immovable wall 0.2 m ahead of its nose; drag off.
//!
//! Reports, from the rear third of the body (its nodes' mass-weighted mean, a
//! rear-crossmember accelerometer stand-in): time of first contact, dynamic crush
//! (its travel from contact to zero velocity) and the time of it, the peak
//! deceleration (5 ms box-filtered), the rebound speed; and at the end (`t=`,
//! default 1.0 s after contact) the residual crush: the nose nodes' (within 10 cm
//! of the rest nose) travel back in the rear third's rigid frame (Horn fit), mean
//! and max, and the length lost; beams broken and bent.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example crush_probe -- jbeam <mods-dir> <name> <config.pc> [kmh=56] [t=1.0]
//! cargo run --release -p engine-softbody --example crush_probe -- sedan [res=low] [kmh=56]
//! ```
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox, STEP};
use glam::{DMat3, DQuat, DVec3, Vec3};

/// The rotation taking points `a` onto `b` best (Horn's quaternion; both centred).
fn horn(a: &[DVec3], b: &[DVec3]) -> DQuat {
    let mut s = DMat3::ZERO;
    for (p, q) in a.iter().zip(b) {
        s += DMat3::from_cols(*p * q.x, *p * q.y, *p * q.z);
    }
    // s[i][j] = Σ a_i b_j (col j, row i).
    let m = |i: usize, j: usize| s.col(j)[i];
    let (sxx, sxy, sxz) = (m(0, 0), m(0, 1), m(0, 2));
    let (syx, syy, syz) = (m(1, 0), m(1, 1), m(1, 2));
    let (szx, szy, szz) = (m(2, 0), m(2, 1), m(2, 2));
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    // Largest eigenvector by power iteration on N + c·I (c makes it positive).
    let c: f64 = n.iter().flatten().map(|x| x.abs()).sum::<f64>() + 1.0;
    let mut v = [1.0, 0.0, 0.0, 0.0];
    for _ in 0..500 {
        let mut w = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                w[i] += (n[i][j] + if i == j { c } else { 0.0 }) * v[j];
            }
        }
        let l = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        v = w.map(|x| x / l);
    }
    DQuat::from_xyzw(v[1], v[2], v[3], v[0]).normalize()
}

/// Elastic energy ½k(ℓ−L)² by beam at the moment of maximum crush (supports only
/// when compressed), grouped by kind and spawn yield threshold, and the top beams.
fn elastic_report(
    s: &ExplicitBeams,
    spawn: &[engine_softbody::explicit_beams::Beam],
    names: &[String],
    ke0: f64,
) {
    use engine_softbody::explicit_beams::ExplicitKind as K;
    let nm = |i: u32| {
        names
            .get(i as usize)
            .cloned()
            .unwrap_or_else(|| format!("#{i}"))
    };
    let mut rows: Vec<(f64, usize)> = Vec::new();
    let mut by: std::collections::BTreeMap<String, f64> = Default::default();
    for (j, b) in s.beams.iter().enumerate() {
        if b.broken || b.a == b.b || !b.k.is_finite() {
            continue;
        }
        let len = s.nodes[b.a as usize]
            .pos
            .distance(s.nodes[b.b as usize].pos);
        let e = match b.kind {
            K::JbeamSupport => (len - b.rest).min(0.0),
            K::JbeamPlain | K::JbeamBounded | K::JbeamAniso | K::Plain => len - b.rest,
            _ => continue,
        };
        let u = 0.5 * f64::from(b.k) * f64::from(e) * f64::from(e);
        rows.push((u, j));
        let t0 = spawn[j].tc;
        let class: String = if !t0.is_finite() {
            "inf".into()
        } else if t0 >= 2e5 {
            ">=200k".into()
        } else if t0 >= 5e4 {
            "50k-200k".into()
        } else if t0 >= 2e4 {
            "20k-50k".into()
        } else {
            "<20k".into()
        };
        *by.entry(format!("{:?} deform {class}", b.kind))
            .or_default() += u;
    }
    let total: f64 = rows.iter().map(|r| r.0).sum();
    rows.sort_by(|x, y| y.0.total_cmp(&x.0));
    println!(
        "  at max crush: elastic {:.1} kJ of the {:.1} kJ impact KE",
        total / 1e3,
        ke0 / 1e3
    );
    for (k, u) in &by {
        if *u > 0.01 * total {
            println!("    {k}: {:.1} kJ", u / 1e3);
        }
    }
    for &(u, j) in rows.iter().take(12) {
        let b = s.beams[j];
        let len = s.nodes[b.a as usize]
            .pos
            .distance(s.nodes[b.b as usize].pos);
        println!(
            "    {:.0} J {:?} {}-{} k {} deform {} (now {}) strength {} rest {:.3}->{:.3} len {:.3} force {:.0}",
            u, b.kind, nm(b.a), nm(b.b), b.k, spawn[j].tc, b.tc, b.strength, spawn[j].rest, b.rest, len, b.force
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| a.iter().find_map(|x| x.strip_prefix(k)).map(str::to_string);
    let f = |k: &str, d: f32| opt(k).map_or(d, |s| s.parse().unwrap());
    let (kmh, t_end) = (f("kmh=", 56.0), f("t=", 1.0));
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let mut names_of: Vec<String> = Vec::new();
    let mut s = match a[0].as_str() {
        "jbeam" => {
            let dir = std::path::Path::new(&a[1]);
            let texts = engine_softbody::jbeam::read_vehicle_dir(dir, &a[2]).unwrap();
            let files: Vec<(&str, &str)> = texts
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            let pcs = std::fs::read_to_string(dir.join(&a[2]).join(&a[3])).unwrap();
            let desc = engine_softbody::jbeam::load_jbeam(&files, Some(&pcs)).unwrap();
            if let Some(path) = opt("census=") {
                // The desc's beams as the solver receives them (TSV), and its parts.
                use std::fmt::Write;
                let nm = |i: u32| desc.nodes[i as usize].name.clone();
                let mut t = String::from("# parts");
                for p in &desc.parts {
                    let _ = write!(t, " {}", p.name);
                }
                t.push('\n');
                for b in &desc.beams {
                    let jb = b.jbeam.unwrap_or_default();
                    let _ = writeln!(
                        t,
                        "{}\t{}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        nm(b.a),
                        nm(b.b),
                        b.kind,
                        b.spring,
                        b.damp,
                        jb.deform,
                        b.strength,
                        b.precompression,
                        jb.deform_limit,
                        jb.deform_limit_expansion
                    );
                }
                std::fs::write(&path, t).unwrap();
                println!("census: {} beams to {path}", desc.beams.len());
                return;
            }
            let (mut s, _) =
                ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings)
                    .unwrap();
            names_of = desc.nodes.iter().map(|q| q.name.clone()).collect();
            let fw = engine_softbody::pack::forward_axis(&desc);
            let turn = glam::Quat::from_rotation_y(-fw.x.atan2(fw.z));
            for n in &mut s.nodes {
                n.pos = turn * n.pos;
            }
            s
        }
        "sedan" => {
            use engine_softbody::authoring::{build, style, BuildParams, Resolution};
            let res = match opt("res=").as_deref() {
                Some("medium") => Resolution::Medium,
                _ => Resolution::Low,
            };
            let b = build(
                &style(&opt("style=").unwrap_or_else(|| "sedan".into())).unwrap(),
                &BuildParams {
                    resolution: res,
                    ..BuildParams::default()
                },
            )
            .unwrap();
            ExplicitBeams::from_truck_raycast(&b.truck, None, Vec3::ZERO, 0.0, Vec3::ZERO, settings)
                .unwrap()
                .0
        }
        other => panic!("unknown car kind {other}"),
    };
    let low = s
        .nodes
        .iter()
        .map(|n| n.pos.y)
        .fold(f32::INFINITY, f32::min);
    for n in &mut s.nodes {
        n.pos.y -= low - 0.02;
    }
    // Measurements only (never a default): `deformscale=` multiplies every
    // deformable plain beam's yield threshold (as a part-level `scalebeamDeform`
    // would); `nohardening=1` caps each threshold at its spawn value.
    if let Some(k) = opt("deformscale=") {
        let k: f32 = k.parse().unwrap();
        use engine_softbody::explicit_beams::{ExplicitKind as K, ExplicitType as T};
        for b in s.beams.iter_mut().filter(|b| {
            b.ty == T::Deformable && matches!(b.kind, K::JbeamPlain | K::JbeamSupport | K::Plain)
        }) {
            b.tc *= k;
            b.tt *= k;
            b.gate = b.tc.min(b.tt).min(b.strength);
        }
    }
    if opt("nohardening=").is_some() {
        for b in s.beams.iter_mut() {
            b.limit_k = b.tc.min(b.tt);
        }
    }
    if opt("expansionratio=").is_some() {
        // The other reading of `deformLimitExpansion` dₑ (spec §3): a cap on L/L₀
        // (L ≤ dₑ·L₀) instead of on the plastic change (L ≤ (1 + dₑ)·L₀).
        use engine_softbody::explicit_beams::ExplicitKind as K;
        for b in s
            .beams
            .iter_mut()
            .filter(|b| b.kind == K::JbeamPlain && b.long_bound.is_finite())
        {
            b.long_bound -= b.rest;
        }
    }
    s.run(1.0);
    let n = s.nodes.len();
    let spawn_beams = s.beams.clone();
    let names: Vec<String> = names_of.clone();
    let mass: f32 = s.nodes.iter().map(|q| q.mass).sum();
    let (zlo, zhi) = s.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), q| {
        (l.min(q.pos.z), h.max(q.pos.z))
    });
    let len0 = zhi - zlo;
    let rear: Vec<usize> = (0..n)
        .filter(|&i| s.nodes[i].pos.z < zlo + 0.33 * len0)
        .collect();
    let nose: Vec<usize> = (0..n).filter(|&i| s.nodes[i].pos.z > zhi - 0.10).collect();
    let rest: Vec<DVec3> = s.nodes.iter().map(|q| q.pos.as_dvec3()).collect();
    let rear_mass: f64 = rear.iter().map(|&i| f64::from(s.nodes[i].mass)).sum();
    let rear_com = |s: &ExplicitBeams| -> (f64, f64) {
        let (mut z, mut v) = (0.0, 0.0);
        for &i in &rear {
            let m = f64::from(s.nodes[i].mass);
            z += m * f64::from(s.nodes[i].pos.z);
            v += m * f64::from(s.nodes[i].vel.z);
        }
        (z / rear_mass, v / rear_mass)
    };
    println!(
        "{}: {n} nodes, {} beams, {mass:.0} kg, length {len0:.3} m; rear third {} nodes {rear_mass:.0} kg; nose {} nodes",
        a[0],
        s.beams.len(),
        rear.len(),
        nose.len()
    );
    let v0 = kmh / 3.6;
    for q in &mut s.nodes {
        q.vel.z += v0;
    }
    s.roll_tyres(Vec3::new(0.0, 0.0, v0));
    if let Some(d) = s.drive.as_mut() {
        d.roll(v0);
    }
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.5, zhi + 0.2 + 1.0),
        Vec3::new(5.0, 3.0, 1.0),
    ));
    let steps = ((0.2 / v0 + t_end) / STEP).round() as usize;
    let mut hist: Vec<(f64, f64)> = Vec::with_capacity(steps);
    let mut contact: Option<usize> = None;
    let mut at_max = false;
    for k in 0..steps {
        s.step();
        hist.push(rear_com(&s));
        if contact.is_none() && s.nodes.iter().any(|q| q.pos.z > zhi + 0.2 - 0.03) {
            contact = Some(k);
        }
        if contact.is_some() && !at_max && hist[k].1 <= 0.0 {
            at_max = true;
            elastic_report(&s, &spawn_beams, &names, f64::from(0.5 * mass * v0 * v0));
        }
    }
    let c = contact.expect("never reached the wall");
    // Dynamic crush: the rear third's travel from contact to its first zero velocity.
    let tm = (c..hist.len())
        .find(|&k| hist[k].1 <= 0.0)
        .unwrap_or(hist.len() - 1);
    let d_dyn = hist[tm].0 - hist[c].0;
    // Peak deceleration over 5 ms windows.
    let w = (0.005 / STEP).round() as usize;
    let peak = (c + w..hist.len())
        .map(|k| (hist[k - w].1 - hist[k].1) / (w as f64 * f64::from(STEP)))
        .fold(0.0, f64::max)
        / 9.80665;
    let rebound = hist.last().unwrap().1;
    // Residual crush in the rear third's rigid frame.
    let ca: DVec3 = rear.iter().map(|&i| rest[i]).sum::<DVec3>() / rear.len() as f64;
    let now: Vec<DVec3> = s.nodes.iter().map(|q| q.pos.as_dvec3()).collect();
    let cb: DVec3 = rear.iter().map(|&i| now[i]).sum::<DVec3>() / rear.len() as f64;
    let pa: Vec<DVec3> = rear.iter().map(|&i| now[i] - cb).collect();
    let pb: Vec<DVec3> = rear.iter().map(|&i| rest[i] - ca).collect();
    let r = horn(&pa, &pb);
    let back = |i: usize| ca + r * (now[i] - cb);
    let rms = (rear
        .iter()
        .map(|&i| (back(i) - rest[i]).length_squared())
        .sum::<f64>()
        / rear.len() as f64)
        .sqrt();
    let crush: Vec<f64> = nose
        .iter()
        .map(|&i| rest[i].z - back(i).z)
        .filter(|d| d.abs() < 1.5)
        .collect();
    let mean = crush.iter().sum::<f64>() / crush.len().max(1) as f64;
    let max = crush.iter().copied().fold(f64::MIN, f64::max);
    let zhi1 = (0..n)
        .map(|i| back(i).z)
        .filter(|z| z.is_finite())
        .fold(f64::MIN, f64::max);
    let (broken, bent) = s.damage();
    println!(
        "{kmh} km/h: contact at {:.1} ms; dynamic crush {:.3} m at {:.0} ms after contact; peak decel (5 ms) {peak:.1} g; \
         rear third at the end {rebound:+.2} m/s",
        c as f32 * STEP * 1e3,
        d_dyn,
        (tm - c) as f32 * STEP * 1e3
    );
    println!(
        "  residual ({t_end} s): nose crush mean {mean:.3} m, max {max:.3} m ({} of {} nose nodes); nose moved back {:.3} m; \
         rear fit RMS {:.1} mm; broken {broken} / {}, bent {bent}",
        crush.len(),
        nose.len(),
        f64::from(zhi) - zhi1,
        rms * 1e3,
        s.beams.len()
    );
    // Plastic state: rest-length change against the settled spawn, threshold growth.
    let mut strain: Vec<f32> = Vec::new();
    let mut grow: Vec<f32> = Vec::new();
    let (mut front_n, mut front_y) = (0, 0);
    for (b, b0) in s.beams.iter().zip(&spawn_beams) {
        if b.broken
            || b0.rest <= 0.0
            || b.ty != engine_softbody::explicit_beams::ExplicitType::Deformable
        {
            continue;
        }
        let e = (b.rest - b0.rest) / b0.rest;
        let front = rest[b.a as usize].z > f64::from(zhi) - 1.2
            || rest[b.b as usize].z > f64::from(zhi) - 1.2;
        if front {
            front_n += 1;
        }
        if e.abs() > 2e-3 {
            strain.push(e);
            if front {
                front_y += 1;
            }
            if b0.tc.is_finite() && b0.tc > 0.0 {
                grow.push((b.tc / b0.tc).max(b.tt / b0.tt));
            }
        }
    }
    strain.sort_by(f32::total_cmp);
    grow.sort_by(f32::total_cmp);
    let q = |v: &[f32], p: f32| {
        v.get(((v.len() as f32 - 1.0) * p).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    println!(
        "  plastic: {} deformable beams with |rest change| > 0.2 % ({} of {front_n} in the front 1.2 m); strain p1 {:.3} p50 {:.3} p99 {:.3}; \
         threshold growth of those p50 {:.2} p90 {:.2} max {:.2}",
        strain.len(), front_y, q(&strain, 0.01), q(&strain, 0.5), q(&strain, 0.99),
        q(&grow, 0.5), q(&grow, 0.9), grow.last().copied().unwrap_or(1.0)
    );
}
