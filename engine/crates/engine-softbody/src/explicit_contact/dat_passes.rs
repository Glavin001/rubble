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

/// Subcycled penalty contact (multiple-time-step splitting: RESPA, Tuckerman, Berne and
/// Martyna 1992; crash codes' contact subcycling). The node update has kicked every node by the
/// step's slow forces (beams, gravity, ...). Here the nodes of the listed pairs move over the
/// step again from their start `x0`, in `subs` sub-steps of h/subs, each a kick by the pairs'
/// springs at the current positions then a drift: symplectic Euler at the sub-step, so the
/// springs are integrated stably at stiffness k = `scale`·m/h_s² (m the pair's lightest movable
/// node; explicit crash codes' soft constraint) and their energy comes back. A pair's gap is
/// linearised from its start record (as [`separate`]); the spring k·(d − gap) acts along its
/// normal, shared over its nodes by its weights, equal and opposite: momentum kept. `x1` and
/// `vel` are updated in place; returns, per vehicle, the nodes moved. A pair's thickness is at
/// least `shell`, the sheets' own (crash codes' shell thickness), so the springs act before
/// the collision meshes touch.
pub(super) fn penalty_substeps(recs: &[Rec], x0: &[Vec<Vec3>], x1: &mut [Vec<Vec3>], vel: &mut [Vec<Vec3>], im: &[Vec<f32>], scale: f32, subs: u32, shell: f32) -> Vec<Vec<u32>> {
    let subs = subs.max(1);
    let hs = STEP / subs as f32;
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
    if springs.is_empty() {
        return moved;
    }
    for (v, m) in moved.iter().enumerate() {
        for &i in m {
            x1[v][i as usize] = x0[v][i as usize];
        }
    }
    let mut f: Vec<Vec<Vec3>> = x1.iter().map(|x| vec![Vec3::ZERO; x.len()]).collect();
    for _ in 0..subs {
        for &(r, k) in &springs {
            // The exact distance of the pair's primitives now (a function of the positions
            // alone, so ½k(d − dist)² is a potential and its force, −k(d − dist)·∂dist/∂x =
            // the closest points' signed weights along the unit gap, conserves energy).
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
                f[v][i as usize] += n * (w[j] * push);
            }
        }
        for (v, m) in moved.iter().enumerate() {
            for &i in m {
                let i = i as usize;
                vel[v][i] += f[v][i] * (hs * im[v][i]);
                f[v][i] = Vec3::ZERO;
                x1[v][i] += vel[v][i] * hs;
            }
        }
    }
    moved
}
