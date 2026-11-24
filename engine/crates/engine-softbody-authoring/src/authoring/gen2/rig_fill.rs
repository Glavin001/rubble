//! The rigger's collision-skin repairs: holes closed, seam bridges that fold or cross
//! refused, body-shell triangles through other parts dropped (known-bugs S4; the lint's
//! collision-skin rules).

use super::*;

/// Whether segment p→q passes through triangle (a, b, c) (Möller–Trumbore, ends included).
pub(super) fn seg_tri_cross(p: Vec3, q: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let d = q - p;
    let (e1, e2) = (b - a, c - a);
    let h = d.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-12 {
        return false;
    }
    let f = 1.0 / det;
    let s = p - a;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let qv = s.cross(e1);
    let v = f * d.dot(qv);
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = f * e2.dot(qv);
    (0.0..=1.0).contains(&t)
}

/// The longest hole (its rim's length, m) [`fill_holes`] closes. **Choice**: past the
/// corners over the lamps (2.2 m round, between the hood's side edge and the fender's top),
/// short of the doors' and the trunk lid's outlines (3.0–4.5 m) and the underside's
/// opening; `RIG_FILL_HOLES_M` sets it, 0 none. On (2.5 m) since the seam bridges no
/// longer lay doubled layers (2026-11-24): a skin with no holes under that is the lint's
/// skin_holes rule (measured before: node-only contact's head-on overlap 191 → 166 L).
pub const FILL_HOLES_M: f32 = 2.5;

/// Closes the collision skin's small holes (known-bugs S4: the corners over the lamps let a
/// partner's nodes into the engine bay). A hole is a loop of open edges (on one triangle
/// each), followed the way its triangle runs, so loops that share a node come apart. Each
/// loop no longer than [`FILL_HOLES_M`] is filled by ear clipping in its best-fit plane,
/// its triangles running against the rim's (facing as the skin round it does), of the part
/// of their first node, inner as the rim's triangle. A flat part's single face (the
/// radiator, the seats' flat panels) is such a loop: it gets its other side.
pub(super) fn fill_holes(r: &mut Rig) {
    let max = std::env::var("RIG_FILL_HOLES_M").ok().and_then(|v| v.parse().ok()).unwrap_or(FILL_HOLES_M);
    if max <= 0.0 {
        return;
    }
    let mut add: Vec<Tri> = Vec::new();
    for (lp, k0) in hole_loops(r) {
        cap_hole(r, &lp, k0, max, &mut add);
    }
    if std::env::var("RIG_HULL_DEBUG").is_ok() {
        eprintln!("fill_holes: {} triangles close the skin's holes up to {max} m round", add.len());
    }
    r.car.tris.extend(add);
}

/// The skin's holes as loops (nodes in the rim's running order, a rim triangle's index):
/// the open edges walked the way their triangles run them, the rest as plain loops, a loop
/// through one node twice split there.
fn hole_loops(r: &Rig) -> Vec<(Vec<usize>, usize)> {
    let mut uses: HashMap<(usize, usize), usize> = HashMap::new();
    for t in &r.car.tris {
        for e in 0..3 {
            let (a, b) = (t.n[e], t.n[(e + 1) % 3]);
            *uses.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    // The open edges as their triangle runs them (a → b), with the triangle's index.
    let mut out_of: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
    for (k, t) in r.car.tris.iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (t.n[e], t.n[(e + 1) % 3]);
            if uses[&(a.min(b), a.max(b))] == 1 {
                out_of.entry(a).or_default().push((b, k));
            }
        }
    }
    let mut used: std::collections::HashSet<(usize, usize)> = Default::default();
    // Loops, each as (nodes in the rim's running order, a rim triangle's index).
    let mut loops: Vec<(Vec<usize>, usize)> = Vec::new();
    let starts: Vec<(usize, usize, usize)> =
        out_of.iter().flat_map(|(&a, v)| v.iter().map(move |&(b, k)| (a, b, k))).collect();
    let mut walked: std::collections::HashSet<(usize, usize)> = Default::default();
    for &(a0, b0, k0) in &starts {
        if used.contains(&(a0, b0)) {
            continue;
        }
        let mut lp = vec![a0];
        let mut mine = vec![(a0, b0)];
        used.insert((a0, b0));
        let mut cur = b0;
        let mut closed = false;
        for _ in 0..64 {
            if cur == a0 {
                closed = true;
                break;
            }
            lp.push(cur);
            let Some(&(next, _)) = out_of.get(&cur).and_then(|v| v.iter().find(|&&(n, _)| !used.contains(&(cur, n)))) else {
                break;
            };
            used.insert((cur, next));
            mine.push((cur, next));
            cur = next;
        }
        if closed && lp.len() >= 3 {
            walked.extend(mine.iter().map(|&(a, b)| (a.min(b), a.max(b))));
            loops.push((lp, k0));
        }
    }
    // Where neighbouring triangles run a shared edge the same way (a doubled layer at a
    // seam) the walk breaks: the rest as plain loops (components of the open edges whose
    // every node has two), in the running order most of their rim triangles give.
    {
        let mut nb: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        let mut dir: HashMap<(usize, usize), usize> = HashMap::new();
        for &(a, b, k) in &starts {
            let key = (a.min(b), a.max(b));
            if walked.contains(&key) {
                continue;
            }
            nb.entry(a).or_default().push(b);
            nb.entry(b).or_default().push(a);
            dir.insert((a, b), k);
        }
        let mut seen: std::collections::HashSet<usize> = Default::default();
        for (&s0, _) in nb.iter() {
            if seen.contains(&s0) {
                continue;
            }
            let mut comp = vec![s0];
            let mut stack = vec![s0];
            seen.insert(s0);
            while let Some(x) = stack.pop() {
                for &y in &nb[&x] {
                    if seen.insert(y) {
                        comp.push(y);
                        stack.push(y);
                    }
                }
            }
            if comp.len() < 3 || comp.iter().any(|x| nb[x].len() != 2) {
                continue;
            }
            let mut lp = vec![s0];
            let (mut prev, mut cur) = (s0, nb[&s0][0]);
            while cur != s0 && lp.len() <= comp.len() {
                lp.push(cur);
                let next = if nb[&cur][0] != prev { nb[&cur][0] } else { nb[&cur][1] };
                prev = cur;
                cur = next;
            }
            if lp.len() != comp.len() {
                continue;
            }
            let n = lp.len();
            let along = (0..n).filter(|&k| dir.contains_key(&(lp[k], lp[(k + 1) % n]))).count();
            if 2 * along < n {
                lp.reverse();
            }
            let k0 = (0..n)
                .find_map(|k| dir.get(&(lp[k], lp[(k + 1) % n])).copied())
                .or_else(|| (0..n).find_map(|k| dir.get(&(lp[(k + 1) % n], lp[k])).copied()))
                .unwrap_or(0);
            loops.push((lp, k0));
        }
    }
    // A loop through one node twice is two holes touching there: split at it.
    let mut simple: Vec<(Vec<usize>, usize)> = Vec::new();
    let mut work = loops;
    while let Some((lp, k0)) = work.pop() {
        let dup = (0..lp.len()).find_map(|i| (i + 1..lp.len()).find(|&j| lp[j] == lp[i]).map(|j| (i, j)));
        match dup {
            Some((i, j)) => {
                let a: Vec<usize> = lp[i..j].to_vec();
                let b: Vec<usize> = lp[j..].iter().chain(lp[..i].iter()).copied().collect();
                for x in [a, b] {
                    if x.len() >= 3 {
                        work.push((x, k0));
                    }
                }
            }
            None => simple.push((lp, k0)),
        }
    }
    simple
}

/// One hole (loop `lp`, rim triangle `k0`) no longer than `max` round, ear-clipped into `add`.
fn cap_hole(r: &Rig, lp: &[usize], k0: usize, max: f32, add: &mut Vec<Tri>) {
    let pts: Vec<Vec3> = lp.iter().map(|&i| r.g(i)).collect();
    let len: f32 = (0..pts.len()).map(|k| pts[k].distance(pts[(k + 1) % pts.len()])).sum();
    if len > max {
        return;
    }
    // The cap runs against the rim: each loop edge walked the other way from the
    // triangle already on it (by majority of the rim's edges), ear-clipped in its plane
    // (Newell's normal of the cap's ring). That is what keeps the skin's winding
    // consistent, a flat part's back included.
    let along = (0..lp.len())
        .filter(|&k| {
            let (u, v) = (lp[k], lp[(k + 1) % lp.len()]);
            r.car.tris.iter().any(|t| (0..3).any(|e| t.n[e] == u && t.n[(e + 1) % 3] == v))
        })
        .count();
    let mut ring: Vec<usize> = if 2 * along >= lp.len() {
        (0..lp.len()).rev().collect()
    } else {
        (0..lp.len()).collect()
    };
    let mut nrm = Vec3::ZERO;
    for k in 0..ring.len() {
        let (p, q) = (pts[ring[k]], pts[ring[(k + 1) % ring.len()]]);
        nrm += Vec3::new((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y));
    }
    let nrm = nrm.normalize_or_zero();
    if nrm == Vec3::ZERO {
        return;
    }
    let (part, inner) = (r.car.nodes[lp[0]].part, r.car.tris[k0].inner);
    let sub = r.car.tris[k0].sub.clone();
    // Which way the cap faces: as the skin round the hole does (its rim triangles'
    // normals), unless the rim's triangles lie inside the loop (a flat part's single
    // face: the cap is its back, facing the other way).
    let rim_tris: Vec<usize> = (0..lp.len())
        .filter_map(|k| {
            let (u, v) = (lp[k], lp[(k + 1) % lp.len()]);
            r.car.tris.iter().position(|t| (0..3).any(|e| {
                let (a, b) = (t.n[e], t.n[(e + 1) % 3]);
                (a == u && b == v) || (a == v && b == u)
            }))
        })
        .collect();
    // A flat part's single face: every rim triangle made of the loop's own nodes.
    let backs = if rim_tris.iter().all(|&k| r.car.tris[k].n.iter().all(|i| lp.contains(i))) {
        rim_tris.len()
    } else {
        0
    };
    let skin_now: Vec<[Vec3; 3]> = r.car.tris.iter().map(|t| t.n.map(|i| r.g(i))).collect();
    let crosses_skin = |n: [usize; 3]| {
        let q = n.map(|i| r.g(i));
        skin_now.iter().zip(&r.car.tris).any(|(x, xt)| {
            (0..3).any(|e| {
                let (u, v) = (n[e], n[(e + 1) % 3]);
                let (xu, xv) = (xt.n[e], xt.n[(e + 1) % 3]);
                (!xt.n.contains(&u) && !xt.n.contains(&v) && seg_tri_cross(r.g(u), r.g(v), x[0], x[1], x[2]))
                    || (!n.contains(&xu) && !n.contains(&xv) && seg_tri_cross(r.g(xu), r.g(xv), q[0], q[1], q[2]))
            })
        })
    };
    // (A slit closes with a thin triangle: slivers are taken, crossings are not.)
    let mut push = |add: &mut Vec<Tri>, n: [usize; 3]| {
        let [a, b, c] = n.map(|i| r.g(i));
        if (b - a).cross(c - a).length() < 1e-7 || crosses_skin(n) {
            return;
        }
        add.push(Tri { n, part, group: None, sub: sub.clone(), inner });
    };
    if std::env::var("RIG_HULL_DEBUG").is_ok() {
        eprintln!(
            "fill_holes: loop {:.2} m, {} nodes {:?}{}",
            len,
            lp.len(),
            lp.iter().map(|&i| r.car.nodes[i].name.as_str()).collect::<Vec<_>>(),
            if 2 * backs > rim_tris.len() { " (a flat part's back)" } else { "" }
        );
    }
    let mut guard = 0;
    while ring.len() > 3 && guard < 256 {
        guard += 1;
        let m = ring.len();
        let ear = (0..m).find(|&k| {
            let (ia, ib, ic) = (ring[(k + m - 1) % m], ring[k], ring[(k + 1) % m]);
            let (pa, pb, pc) = (pts[ia], pts[ib], pts[ic]);
            if (pb - pa).cross(pc - pb).dot(nrm) <= 0.0 {
                return false;
            }
            !ring.iter().any(|&j| {
                if j == ia || j == ib || j == ic {
                    return false;
                }
                let p = pts[j];
                let s1 = (pb - pa).cross(p - pa).dot(nrm);
                let s2 = (pc - pb).cross(p - pb).dot(nrm);
                let s3 = (pa - pc).cross(p - pc).dot(nrm);
                s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0
            })
        });
        let Some(k) = ear else {
            if std::env::var("RIG_HULL_DEBUG").is_ok() {
                eprintln!("fill_holes:   no ear left with {} nodes", ring.len());
            }
            break;
        };
        let (ia, ib, ic) = (ring[(k + m - 1) % m], ring[k], ring[(k + 1) % m]);
        push(add, [lp[ia], lp[ib], lp[ic]]);
        ring.remove(k);
    }
    if ring.len() == 3 {
        push(add, [lp[ring[0]], lp[ring[1]], lp[ring[2]]]);
    }
}

/// Parts through each other: where a body-shell triangle and another part's cross (one's
/// edge through the other), the body's goes; the part that moves (a door at the C-pillar)
/// keeps its skin (the lint's skin_self_crossing). `RIG_HULL_KEEP_CROSSING=1`: as before.
pub(super) fn drop_body_crossing(r: &Rig, tris: &mut Vec<Tri>) {
    if std::env::var("RIG_HULL_KEEP_CROSSING").is_ok_and(|v| v == "1") {
        return;
    }
    let crosses = |x: &Tri, y: &Tri| {
        (0..3).any(|e| {
            let (u, v) = (x.n[e], x.n[(e + 1) % 3]);
            !y.n.contains(&u) && !y.n.contains(&v) && seg_tri_cross(r.g(u), r.g(v), r.g(y.n[0]), r.g(y.n[1]), r.g(y.n[2]))
        })
    };
    let drop: std::collections::HashSet<usize> = (0..tris.len())
        .filter(|&i| tris[i].part == Part::Body)
        .filter(|&i| tris.iter().any(|y| y.part != Part::Body && (crosses(&tris[i], y) || crosses(y, &tris[i]))))
        .collect();
    if !drop.is_empty() {
        let mut k = 0;
        tris.retain(|_| {
            k += 1;
            !drop.contains(&(k - 1))
        });
    }
}

/// A seam bridge `t` (flipped: it will face the other way) that folds or crosses (unless
/// `RIG_BRIDGE_OVERLAP=1`, as before): it walks an edge it shares with a triangle the same
/// way (it lies on top of it: a doubled layer, the lint's skin_winding and skin_facing), or
/// passes through a triangle of the skin, or has one of the skin's edges through it
/// (skin_self_crossing).
pub(super) fn bridge_folds_or_crosses(r: &Rig, t: [usize; 3], flipped: bool, tris: &[Tri]) -> bool {
    if std::env::var("RIG_BRIDGE_OVERLAP").is_ok_and(|v| v == "1") {
        return false;
    }
    let o = if flipped { [t[0], t[2], t[1]] } else { t };
    for e in 0..3 {
        let (u, v) = (o[e], o[(e + 1) % 3]);
        if tris.iter().any(|x| (0..3).any(|k| x.n[k] == u && x.n[(k + 1) % 3] == v)) {
            return true;
        }
    }
    let [qa, qb, qc] = t.map(|i| r.g(i));
    tris.iter().any(|x| {
        let [xa, xb, xc] = x.n.map(|i| r.g(i));
        (0..3).any(|e| {
            let (u, v) = (t[e], t[(e + 1) % 3]);
            let (xu, xv) = (x.n[e], x.n[(e + 1) % 3]);
            (!x.n.contains(&u) && !x.n.contains(&v) && seg_tri_cross(r.g(u), r.g(v), xa, xb, xc))
                || (!t.contains(&xu) && !t.contains(&xv) && seg_tri_cross(r.g(xu), r.g(xv), qa, qb, qc))
        })
    })
}

/// Triangle `n` crosses one of `placed` (an edge of either through the other), at `pos`.
pub(super) fn tri_crosses(pos: &[Vec3], n: [usize; 3], placed: &[[usize; 3]]) -> bool {
    placed.iter().any(|x| {
        (0..3).any(|e| {
            let (u, v) = (n[e], n[(e + 1) % 3]);
            let (xu, xv) = (x[e], x[(e + 1) % 3]);
            (!x.contains(&u) && !x.contains(&v) && seg_tri_cross(pos[u], pos[v], pos[x[0]], pos[x[1]], pos[x[2]]))
                || (!n.contains(&xu) && !n.contains(&xv) && seg_tri_cross(pos[xu], pos[xv], pos[n[0]], pos[n[1]], pos[n[2]]))
        })
    })
}

/// Each part's nodes at its skin (skin_hull's): structural, within `RIG_HULL_POOL` (3 cm) of
/// the surface.
pub(super) fn hull_pool(r: &Rig, inp: &dyn RigInput) -> HashMap<Part, Vec<usize>> {
    let mut pool: HashMap<Part, Vec<usize>> = HashMap::new();
    for i in 0..r.car.nodes.len() {
        let n = &r.car.nodes[i];
        if n.contact
            || n.no_group
            || n.sub.is_some()
            || n.name.starts_with("hub")
            || n.name.starts_with("rim")
        {
            continue;
        }
        let within = std::env::var("RIG_HULL_POOL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.03f32);
        if inp.sdf(r.g(i)).abs() < within {
            if std::env::var("RIG_HULL_DEBUG").is_ok() {
                eprintln!(
                    "pool {:?} {} sdf {:.3} skin {}",
                    n.part,
                    n.name,
                    inp.sdf(r.g(i)),
                    r.skin[i]
                );
            }
            pool.entry(n.part).or_default().push(i);
        }
    }
    pool
}

/// The underside closed (on; `RIG_BELLY=0`: as before): the downward faces of the
/// convex hull of the low skin nodes (on an outer collision triangle, below `RIG_BELLY_Y`,
/// default 0.4 m over the lowest), fanned into triangles facing down and out, as outer
/// skin; none that crosses the skin, doubles one of its triangles or lies on top of it.
/// The engine bay and the floor were open from below: a partner's nodes slipped under the
/// bumper's bottom edge into the car.
pub(super) fn belly_cap(r: &mut Rig) {
    if std::env::var("RIG_BELLY").is_ok_and(|v| v == "0") {
        return;
    }
    let mut skin: std::collections::BTreeSet<usize> = Default::default();
    for t in &r.car.tris {
        if !t.inner {
            skin.extend(t.n);
        }
    }
    let low = skin.iter().map(|&i| r.g(i).y).fold(f32::MAX, f32::min);
    let cut = low + std::env::var("RIG_BELLY_Y").ok().and_then(|v| v.parse().ok()).unwrap_or(0.4f32);
    let ids: Vec<usize> = skin.into_iter().filter(|&i| r.g(i).y < cut).collect();
    let pts: Vec<Vec3> = ids.iter().map(|&i| r.g(i)).collect();
    if pts.len() < 4 {
        return;
    }
    let cen = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
    let have: std::collections::HashSet<[usize; 3]> = r
        .car
        .tris
        .iter()
        .map(|t| {
            let mut k = t.n;
            k.sort_unstable();
            k
        })
        .collect();
    let skin_now: Vec<[usize; 3]> = r.car.tris.iter().map(|t| t.n).collect();
    let pos: Vec<Vec3> = (0..r.car.nodes.len()).map(|i| r.g(i)).collect();
    let mut add: Vec<Tri> = Vec::new();
    for face in super::hull_faces(&pts) {
        for k in 1..face.len().saturating_sub(1) {
            let mut n = [ids[face[0]], ids[face[k]], ids[face[k + 1]]];
            let [a, b, c] = n.map(|i| pos[i]);
            let mut nn = (b - a).cross(c - a);
            if nn.length() < 1e-6 {
                continue;
            }
            if nn.dot((a + b + c) / 3.0 - cen) < 0.0 {
                n = [n[0], n[2], n[1]];
                nn = -nn;
            }
            if nn.normalize().y > -0.6 {
                continue;
            }
            let mut key = n;
            key.sort_unstable();
            if have.contains(&key) || tri_crosses(&pos, n, &skin_now) {
                continue;
            }
            // Not on top of the skin already there (a doubled layer): its centre clear of
            // every skin triangle by 3 cm.
            let mid = (a + b + c) / 3.0;
            let near = r.car.tris.iter().any(|t| {
                let [x, y, z] = t.n.map(|i| pos[i]);
                crate::authoring::lint::tri_dist_pub(mid, x, y, z) < 0.03
            });
            if near {
                continue;
            }
            add.push(Tri { n, part: Part::Body, group: None, sub: None, inner: false });
        }
    }
    if std::env::var("RIG_HULL_DEBUG").is_ok() {
        eprintln!("belly_cap: {} triangles under the car ({} low skin nodes)", add.len(), ids.len());
    }
    r.car.tris.extend(add);
}
