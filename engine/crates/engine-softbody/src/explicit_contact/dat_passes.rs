//! Continuous contact's velocity and position passes over the listed pairs (`dat`'s
//! [`super::Rec`]): the speculative contact, its rows' form, the split correction, and the
//! penalty contact's forces.

use super::{Rec, STEP};
use glam::Vec3;

/// The speculative inelastic contact: Jacobi with mass splitting over the pairs, `iters`
/// rounds, on velocities. Each pair's closing speed along its normal is cut to what
/// brings its gap to its paint thickness by the step's end (to zero closing speed when it
/// is already closer): an inelastic impulse by the nodes' inverse masses, so momentum
/// passes to the struck side. It removes approach and never adds separation. Each change
/// of velocity moves the node by h times it (as the node update would have).
pub(super) fn speculative(recs: &[&Rec], v: &mut [Vec<Vec3>], im: &[Vec<f32>], iters: u32) {
    if std::env::var("DAT_RESPONSE").is_ok_and(|x| x == "rows") {
        return speculative_rows(recs, v, im);
    }
    let h = STEP;
    // Mass splitting (Tonge et al. 2012): a node's inverse mass is shared by its active pairs
    // in each pair's effective mass, and each pair's impulse is applied in full, equal and
    // opposite on its two sides: momentum is exact.
    let mut count: Vec<Vec<u32>> = v.iter().map(|x| vec![0; x.len()]).collect();
    for _ in 0..iters {
        let act: Vec<(&Rec, f32)> = recs
            .iter()
            .filter_map(|&r| {
                let rel: f32 = (0..r.len).map(|j| r.sign(j) * r.w[j] * v[r.k[j].0][r.k[j].1 as usize].dot(r.n)).sum();
                let c = rel + (r.g - r.d).max(0.0) / h;
                (c < 0.0).then_some((r, c))
            })
            .collect();
        if act.is_empty() {
            break;
        }
        for &(r, _) in &act {
            for j in 0..r.len {
                let (q, i) = r.k[j];
                if r.w[j] > 0.0 && im[q][i as usize] > 0.0 {
                    count[q][i as usize] += 1;
                }
            }
        }
        let mut sums = super::super::ccd::FixedSums::new(v);
        for &(r, c) in &act {
            let wsum: f32 = (0..r.len).map(|j| r.w[j] * r.w[j] * im[r.k[j].0][r.k[j].1 as usize] * count[r.k[j].0][r.k[j].1 as usize] as f32).sum();
            if wsum <= 0.0 {
                continue;
            }
            let l = -c / wsum;
            for j in 0..r.len {
                let (q, i) = r.k[j];
                let m = im[q][i as usize];
                if m > 0.0 && r.w[j] > 0.0 {
                    sums.add_v(q, i, r.n * (r.sign(j) * r.w[j] * m * l));
                }
            }
        }
        sums.apply(v, None);
        for c in count.iter_mut() {
            c.iter_mut().for_each(|c| *c = 0);
        }
    }
}

/// [`speculative`] as rows with accumulated impulses (`DAT_RESPONSE=rows`; design C): each
/// pair's λ ≥ 0 moves to where its constraint (closing speed no more than brings its gap to its
/// thickness by the step's end) would hold with its nodes shared among all the pairs, projected
/// Jacobi to a fixed point (at most `ccd_rows::ROW_ITERS`), so the velocities end on the
/// mass-weighted projection, not on whatever the rounds reached.
pub(super) fn speculative_rows(recs: &[&Rec], v: &mut [Vec<Vec3>], im: &[Vec<f32>]) {
    let h = STEP;
    let iters = std::env::var("DAT_ROW_ITERS").ok().and_then(|x| x.parse().ok()).unwrap_or(super::super::ccd_rows::ROW_ITERS);
    let mut count: Vec<Vec<u32>> = v.iter().map(|x| vec![0; x.len()]).collect();
    for r in recs {
        for j in 0..r.len {
            let (q, i) = r.k[j];
            if r.w[j] > 0.0 && im[q][i as usize] > 0.0 {
                count[q][i as usize] += 1;
            }
        }
    }
    let d: Vec<f32> = recs.iter().map(|r| (0..r.len).map(|j| r.w[j] * r.w[j] * im[r.k[j].0][r.k[j].1 as usize] * count[r.k[j].0][r.k[j].1 as usize] as f32).sum()).collect();
    let mut lam = vec![0.0f32; recs.len()];
    let mut sums = super::super::ccd::FixedSums::new(v);
    for _ in 0..iters {
        let mut changed = false;
        for (q, r) in recs.iter().enumerate() {
            if d[q] <= 0.0 {
                continue;
            }
            let rel: f32 = (0..r.len).map(|j| r.sign(j) * r.w[j] * v[r.k[j].0][r.k[j].1 as usize].dot(r.n)).sum();
            let c = rel + (r.g - r.d).max(0.0) / h;
            let l = (lam[q] - c / d[q]).max(0.0);
            let dl = l - lam[q];
            if dl == 0.0 {
                continue;
            }
            changed = true;
            lam[q] = l;
            for j in 0..r.len {
                let (k, i) = r.k[j];
                let m = im[k][i as usize];
                if m > 0.0 && r.w[j] > 0.0 {
                    sums.add_v(k, i, r.n * (r.sign(j) * r.w[j] * m * dl));
                }
            }
        }
        sums.apply(v, None);
        if !changed {
            break;
        }
    }
}

/// The split correction (Box2D's and Bullet's split impulse): pairs whose paint overlaps
/// at the step's end (gap below their thickness) are moved apart by `beta` of the overlap,
/// positions only, Jacobi with mass splitting, `iters` rounds. No velocity changes, so no
/// kinetic energy is added; the truncation that follows keeps the node mesh uncrossed.
pub(super) fn separate(recs: &[&Rec], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], im: &[Vec<f32>], iters: u32, beta: f32) {
    let mut acc: Vec<Vec<(Vec3, u32)>> = x1.iter().map(|x| vec![(Vec3::ZERO, 0); x.len()]).collect();
    for _ in 0..iters {
        let mut any = false;
        for &r in recs {
            let dx = |j: usize| x1[r.k[j].0][r.k[j].1 as usize] - x0[r.k[j].0][r.k[j].1 as usize];
            let gap = r.g + (0..r.len).map(|j| r.sign(j) * r.w[j] * dx(j).dot(r.n)).sum::<f32>();
            let c = gap - r.d;
            if c >= 0.0 {
                continue;
            }
            let wsum: f32 = (0..r.len).map(|j| r.w[j] * r.w[j] * im[r.k[j].0][r.k[j].1 as usize]).sum();
            if wsum <= 0.0 {
                continue;
            }
            let l = -c * beta / wsum;
            for j in 0..r.len {
                let (q, i) = r.k[j];
                let m = im[q][i as usize];
                if m > 0.0 && r.w[j] > 0.0 {
                    let e = &mut acc[q][i as usize];
                    e.0 += r.n * (r.sign(j) * r.w[j] * m * l);
                    e.1 += 1;
                    any = true;
                }
            }
        }
        if !any {
            break;
        }
        for (x, a) in x1.iter_mut().zip(acc.iter_mut()) {
            for (p, e) in x.iter_mut().zip(a.iter_mut()) {
                if e.1 > 0 {
                    *p += e.0 / e.1 as f32;
                    *e = (Vec3::ZERO, 0);
                }
            }
        }
    }
}

/// Subcycled penalty contact (crash codes' contact subcycling; Belytschko's mixed-time
/// integration). The contact nodes C (the movable nodes of the listed pairs) move over the step
/// again from their start `x0` and their velocity before the node update, in `subs` sub-steps
/// of h/subs (symplectic Euler: kick then drift). Their force each sub-step is the step's force
/// (`pre`, everything the beam pass summed), plus the change since the step's start of their
/// own linear beams' forces (k(ℓ − L) + c·ℓ̇ along the beam, with a neighbour outside C on its
/// straight path from `x0` to `x1`), plus the pairs' springs: k = `scale`·m/h_s² (m the pair's
/// lightest movable node; the soft constraint at the sub-step), ½k(d − dist)² on the pair's
/// exact distance, so its force is the closest points' signed weights along the unit gap
/// (equal and opposite). A neighbour outside C takes back the beams' change averaged over the
/// sub-steps, so each beam's two ends stay equal and opposite: momentum kept. A pair's
/// thickness is at least `shell` (crash codes' shell thickness). `x1` and `vel` are updated in
/// place; returns, per vehicle, the nodes moved.
#[allow(clippy::too_many_arguments)]
pub(super) fn penalty_substeps(recs: &[Rec], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], scale: f32, subs: u32, shell: f32, vehicles: &[crate::explicit_beams::ExplicitBeams], pre: &[Vec<(Vec3, Vec3)>]) -> Vec<Vec<u32>> {
    use crate::explicit_beams::ExplicitKind;
    let subs = subs.max(1);
    let (h, hs) = (STEP, STEP / subs as f32);
    let mut moved: Vec<Vec<u32>> = x1.iter().map(|_| Vec::new()).collect();
    let mut on: Vec<Vec<bool>> = x1.iter().map(|x| vec![false; x.len()]).collect();
    let mut springs: Vec<(&Rec, f32)> = Vec::new();
    for r in recs.iter().filter(|r| r.len == 4) {
        let ml = (0..r.len).filter(|&j| r.w[j] > 0.0 && im[r.k[j].0][r.k[j].1 as usize] > 0.0).map(|j| 1.0 / im[r.k[j].0][r.k[j].1 as usize]).fold(f32::MAX, f32::min);
        if ml == f32::MAX {
            continue;
        }
        springs.push((r, scale * ml / (hs * hs)));
        for j in 0..r.len {
            let (v, i) = r.k[j];
            if r.w[j] > 0.0 && im[v][i as usize] > 0.0 && !on[v][i as usize] {
                on[v][i as usize] = true;
                moved[v].push(i);
            }
        }
    }
    if springs.is_empty() || pre.is_empty() {
        return moved;
    }
    // The linear beams at a contact node (each once): (vehicle, beam).
    let mut beams: Vec<(usize, u32)> = Vec::new();
    for (v, m) in moved.iter().enumerate() {
        let veh = &vehicles[v];
        for &i in m {
            for &j in veh.incidence.get(i as usize).map_or(&[][..], |x| x.as_slice()) {
                let b = &veh.beams[j as usize];
                if b.broken || !matches!(b.kind, ExplicitKind::Plain | ExplicitKind::JbeamPlain) {
                    continue;
                }
                let other = if b.a == i { b.b } else { b.a };
                // (A beam between two contact nodes: once, from its lower end.)
                if on[v][other as usize] && other < i {
                    continue;
                }
                beams.push((v, j));
            }
        }
    }
    let path = |v: usize, i: usize, t: f32, x1: &[Vec<Vec3>]| x0[v][i] + (x1[v][i] - x0[v][i]) * t;
    // A beam's force on its end a (b's is the opposite) at positions and velocities given.
    let force = |b: &crate::explicit_beams::Beam, xa: Vec3, xb: Vec3, va: Vec3, vb: Vec3| -> Vec3 {
        let d = xb - xa;
        let l = d.length();
        if l <= 0.0 {
            return Vec3::ZERO;
        }
        let u = d / l;
        u * (b.k * (l - b.rest) + b.c * (vb - va).dot(u))
    };
    // Each beam's force at the step's start (positions x0, velocities before the update).
    let x1s: Vec<Vec<Vec3>> = x1.to_vec();
    let start: Vec<Vec3> = beams.iter().map(|&(v, j)| {
        let b = &vehicles[v].beams[j as usize];
        force(b, x0[v][b.a as usize], x0[v][b.b as usize], pre[v][b.a as usize].1, pre[v][b.b as usize].1)
    }).collect();
    let mut avg: Vec<Vec3> = vec![Vec3::ZERO; beams.len()];
    for (v, m) in moved.iter().enumerate() {
        for &i in m {
            x1[v][i as usize] = x0[v][i as usize];
            vel[v][i as usize] = pre[v][i as usize].1;
        }
    }
    let mut f: Vec<Vec<Vec3>> = x1.iter().map(|x| vec![Vec3::ZERO; x.len()]).collect();
    for s in 0..subs {
        let t = s as f32 / subs as f32;
        for (v, m) in moved.iter().enumerate() {
            for &i in m {
                f[v][i as usize] = pre[v][i as usize].0;
            }
        }
        for (q, &(v, j)) in beams.iter().enumerate() {
            let b = &vehicles[v].beams[j as usize];
            let (a, c) = (b.a as usize, b.b as usize);
            let pos = |i: usize| if on[v][i] { x1[v][i] } else { path(v, i, t, &x1s) };
            let vv = |i: usize| if on[v][i] { vel[v][i] } else { (x1s[v][i] - x0[v][i]) / h };
            let df = force(b, pos(a), pos(c), vv(a), vv(c)) - start[q];
            avg[q] += df / subs as f32;
            if on[v][a] {
                f[v][a] += df;
            }
            if on[v][c] {
                f[v][c] -= df;
            }
        }
        for &(r, k) in &springs {
            let p = r.k.map(|(v, i)| x1[v][i as usize]);
            let (w, gv) = super::super::ccd::closest(r.nn, p);
            let dist = gv.length();
            let d = r.d.max(shell);
            if dist >= d || dist <= 0.0 {
                continue;
            }
            let n = gv / dist;
            let push = k * (d - dist);
            for j in 0..4 {
                let (v, i) = r.k[j];
                if on[v][i as usize] {
                    f[v][i as usize] += n * (w[j] * push);
                }
            }
        }
        for (v, m) in moved.iter().enumerate() {
            for &i in m {
                let i = i as usize;
                vel[v][i] += f[v][i] * (hs * im[v][i]);
                x1[v][i] += vel[v][i] * hs;
            }
        }
    }
    // The neighbours outside C take back the beams' change, averaged over the step.
    for (q, &(v, j)) in beams.iter().enumerate() {
        let b = &vehicles[v].beams[j as usize];
        for (end, sign) in [(b.a as usize, 1.0f32), (b.b as usize, -1.0)] {
            if !on[v][end] && im[v][end] > 0.0 {
                let dv = avg[q] * (sign * h * im[v][end]);
                vel[v][end] += dv;
                x1[v][end] += dv * h;
                if !moved[v].contains(&(end as u32)) {
                    moved[v].push(end as u32);
                }
            }
        }
    }
    moved
}
