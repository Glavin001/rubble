//! Leaf helpers of [`super::build`] (split from `body2.rs`): roles, mount pairs, the 6x6 solve and the mount preload.

use super::*;

/// Whether a node hung on spokes to `ids` lies on the one plane they (nearly)
/// span, with a component of gravity across it: the node swings out of that
/// plane under its own weight. The plane is the least-squares one (the
/// covariance C's weakest axis); "nearly" is 27 det(C) / tr(C)³ under
/// [`MIN_SPREAD`] (1: isotropic, 0: planar), and "on" is within [`ON_PLANE`].
pub(super) fn hangs_in_a_plane(r: &Rig, ids: &[usize], hung: Vec3) -> bool {
    let g: Vec<Vec3> = ids.iter().map(|&i| r.g(i)).collect();
    let c = g.iter().fold(Vec3::ZERO, |s, &p| s + p) / g.len() as f32;
    let m = g.iter().fold(glam::Mat3::ZERO, |m, &p| {
        let d = p - c;
        m + glam::Mat3::from_cols(d * d.x, d * d.y, d * d.z)
    }) * (1.0 / g.len() as f32);
    let tr = m.x_axis.x + m.y_axis.y + m.z_axis.z;
    if tr <= 0.0 || 27.0 * m.determinant() / (tr * tr * tr) >= MIN_SPREAD {
        return false;
    }
    // the weakest axis: the power iteration of tr·I - C
    let a = glam::Mat3::from_diagonal(Vec3::splat(tr)) - m;
    let mut n = Vec3::new(0.3, 0.5, 0.8).normalize();
    for _ in 0..60 {
        n = (a * n).normalize_or(Vec3::Y);
    }
    // (up is glTF's y)
    n.y.abs() > 0.25 && n.dot(hung - c).abs() < ON_PLANE
}

/// A pane's ring is nearly planar below this ([`hangs_in_a_plane`]).
pub(super) const MIN_SPREAD: f32 = 0.01;
/// A pane's centre is on its ring's plane within this (m).
pub(super) const ON_PLANE: f32 = 0.10;

pub(super) fn v3(p: DVec3) -> Vec3 {
    p.as_vec3()
}

/// The member tag the closures' hardware looks for on a shell node.
pub(super) fn node_role(group: &str, name: &str) -> Role {
    let tail = name.rsplit('_').next().unwrap_or("");
    match group {
        "A1" | "A2" | "A4" | "A5" => Role::Rail,
        "A3" => Role::Cross,
        "A6" | "A7" | "A8" => Role::Apron,
        "B1" => Role::Firewall,
        // the cowl's outer two are the A-pillars' bases
        "B2" if tail.starts_with('o') => Role::PillarA,
        "B2" => Role::Firewall,
        // the floor rows' outer lane is the sill
        "C1" if tail.len() >= 2 && tail.as_bytes()[1] == b'o' => Role::Sill,
        "C1" | "E3" => Role::Floor,
        "C2" => Role::Sill,
        "C3" => Role::Tunnel,
        "D1" => Role::PillarA,
        "D2" => Role::PillarB,
        "D3" => Role::PillarC,
        "D4" => Role::RoofBow,
        "E1" | "E2" => Role::RearRail,
        "E5" => Role::Sheet,
        _ => Role::Cross,
    }
}

pub(super) fn beam_role(z: Zone) -> Role {
    match z {
        Zone::CrashBox => Role::CrushCan,
        Zone::FrontRail => Role::Rail,
        Zone::Tower => Role::Apron,
        Zone::Cabin => Role::Floor,
        Zone::Primary => Role::Sill,
        Zone::Pillar => Role::PillarB,
        Zone::Bow => Role::RoofBow,
        Zone::RearRail => Role::RearRail,
        Zone::Tail => Role::Cross,
    }
}

pub(super) fn fixed_of(v: &Vals, kind: FixedKind) -> Fixed {
    Fixed {
        k: v.k.map(|k| k as f32),
        c: v.c as f32,
        deform: v.deform as f32,
        kind,
        limit_expansion: 0.0,
        deform_group: None,
    }
}

/// A defined beam of `sub` (a second request for a pair keeps the first).
#[allow(clippy::too_many_arguments)]
pub(super) fn put(
    r: &mut Rig,
    a: usize,
    b: usize,
    role: Role,
    sub: &str,
    f: Fixed,
    strength: f32,
    group: Option<&str>,
) -> Option<usize> {
    if a == b || r.car.has_beam(a, b) {
        return None;
    }
    let j = r.car.beam(a, b, role, Part::Engine);
    let bm = &mut r.car.beams[j];
    bm.sub = Some(sub.to_string());
    bm.strength = strength;
    if strength.is_finite() {
        bm.break_group = group.map(str::to_string);
    }
    r.car.fixed.insert(j, f);
    Some(j)
}

/// `n` pairs between `from` and `to`: each `from` node's nearest `to` node,
/// then each one's second nearest, and so on (so every `from` node is held
/// before any is held twice).
pub(super) fn mount_pairs(r: &Rig, from: &[usize], to: &[usize], n: usize) -> Vec<(usize, usize)> {
    let order: Vec<Vec<usize>> = from
        .iter()
        .map(|&f| {
            let g = r.g(f);
            let mut v: Vec<(f32, usize)> = to
                .iter()
                .filter(|&&t| t != f)
                .map(|&t| (r.g(t).distance(g), t))
                .collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v.into_iter().map(|x| x.1).collect()
        })
        .collect();
    let mut out = Vec::new();
    for round in 0..to.len() {
        for (fi, &f) in from.iter().enumerate() {
            if out.len() == n {
                return out;
            }
            if let Some(&t) = order[fi].get(round) {
                out.push((f, t));
            }
        }
    }
    out
}

/// Solve the 6 × 6 system `a x = b` (Gaussian elimination, partial pivot).
pub(super) fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for i in 0..6 {
        let p = (i..6).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs()))?;
        if a[p][i].abs() < 1e-9 {
            return None;
        }
        a.swap(i, p);
        b.swap(i, p);
        for k in i + 1..6 {
            let f = a[k][i] / a[i][i];
            for c in i..6 {
                a[k][c] -= f * a[i][c];
            }
            b[k] -= f * b[i];
        }
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let s: f64 = (i + 1..6).map(|c| a[i][c] * x[c]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

/// Precompress `beams` (each with its spring, from a node of `body` to a
/// node outside it) so they carry `body`'s weight at the design position:
/// the least strain energy that balances its weight and leaves no moment.
pub(super) fn preload(r: &mut Rig, body: &[usize], beams: &[(usize, f64)], mass_of: &dyn Fn(usize) -> f64) {
    if beams.len() < 6 {
        return;
    }
    let m: f64 = body.iter().map(|&i| mass_of(i)).sum();
    let com = body
        .iter()
        .fold(DVec3::ZERO, |s, &i| s + r.car.p(i).as_dvec3() * mass_of(i))
        / m;
    // JBeam frame: z up. Per beam its force on the body per metre of rest
    // length added: k along the beam, from the outside end to the body's.
    let mut rows: Vec<([f64; 6], f64)> = Vec::new();
    for &(j, k) in beams {
        let b = &r.car.beams[j];
        let (inn, out) = if body.contains(&b.a) { (b.a, b.b) } else { (b.b, b.a) };
        let (pi, po) = (r.car.p(inn).as_dvec3(), r.car.p(out).as_dvec3());
        let u = (pi - po).normalize_or_zero();
        let t = (pi - com).cross(u);
        rows.push(([u.x * k, u.y * k, u.z * k, t.x * k, t.y * k, t.z * k], k));
    }
    let mut a = [[0.0; 6]; 6];
    for (row, k) in &rows {
        for x in 0..6 {
            for y in 0..6 {
                a[x][y] += row[x] * row[y] / k;
            }
        }
    }
    let rhs = [0.0, 0.0, m * 9.81, 0.0, 0.0, 0.0];
    let Some(lam) = solve6(a, rhs) else {
        return;
    };
    for (&(j, _), (row, k)) in beams.iter().zip(&rows) {
        let delta: f64 = (0..6).map(|x| row[x] * lam[x]).sum::<f64>() / k;
        let b = &r.car.beams[j];
        let l = r.car.p(b.a).distance(r.car.p(b.b)) as f64;
        let pc = ((l + delta) / l).clamp(0.8, 1.25);
        let old = r.car.precomp.get(&j).copied().unwrap_or(1.0);
        r.car.precomp.insert(j, old * pc as f32);
        // (a rubber mount's free travel is about its loaded position)
        if let Some(Fixed {
            kind: FixedKind::Bounded { travel, .. },
            ..
        }) = r.car.fixed.get_mut(&j)
        {
            *travel = travel.max((pc as f32 - 1.0).abs() + MOUNT_TRAVEL);
        }
    }
}

/// Engine stops (each block corner against the structure around it) and
/// exhaust rests (each chain node on the floor above it), split from `build`.
pub(super) fn stops_and_rests(r: &mut Rig, d: &Defs, ids: &HashMap<String, usize>) {
    let id = |ids: &HashMap<String, usize>, n: &str| -> usize {
        *ids.get(n).unwrap_or_else(|| panic!("no defined node {n}"))
    };
    // ---- engine stops: each block corner against the structure around it
    if let Some(eng) = d.parts.iter().find(|p| p.name == "engine") {
        let around: Vec<usize> = (0..r.car.nodes.len())
            .filter(|&i| {
                let n = &r.car.nodes[i];
                (n.part == Part::Body
                    && matches!(
                        r.member[i],
                        Some(Role::Firewall)
                            | Some(Role::Rail)
                            | Some(Role::Apron)
                            | Some(Role::Cross)
                    ))
                    || n.sub.as_deref() == Some("subframe_front")
            })
            .filter(|&i| r.g(i).z > v3(eng.nodes[0].pos).z - 1.2)
            .collect();
        let (k, c, deform, pc) = ENGINE_STOP;
        for n in eng.nodes.iter().take(8) {
            let e = id(ids, &n.name);
            for (a, b) in mount_pairs(r, &[e], &around, ENGINE_STOPS_EACH) {
                let f = Fixed {
                    k: Some(k),
                    c,
                    deform,
                    kind: FixedKind::Support,
                    limit_expansion: 0.0,
                    deform_group: None,
                };
                if let Some(j) = put(r, a, b, Role::Bearing, "engine", f, f32::INFINITY, None) {
                    r.car.precomp.insert(j, pc);
                }
            }
        }
    }
    // ---- exhaust rests: each chain node on the floor above it
    if let Some(ex) = d.parts.iter().find(|p| p.name == "exhaust") {
        let floor: Vec<usize> = (0..r.car.nodes.len())
            .filter(|&i| {
                r.car.nodes[i].part == Part::Body
                    && matches!(
                        r.member[i],
                        Some(Role::Floor)
                            | Some(Role::Tunnel)
                            | Some(Role::Sill)
                            | Some(Role::RearRail)
                            | Some(Role::Firewall)
                    )
            })
            .collect();
        let (k, c, deform, pc) = EXHAUST_REST;
        for n in &ex.nodes {
            let e = id(ids, &n.name);
            for (a, b) in mount_pairs(r, &[e], &floor, 2) {
                let f = Fixed {
                    k: Some(k),
                    c,
                    deform,
                    kind: FixedKind::Support,
                    limit_expansion: 0.0,
                    deform_group: None,
                };
                if let Some(j) = put(r, a, b, Role::Bearing, "exhaust", f, f32::INFINITY, None) {
                    r.car.precomp.insert(j, pc);
                }
            }
        }
    }
}
