//! The resident (native) solve of PhysX's GPU destruction stage, one island at a time
//! (`stressgpu/detail/StressComponentIteration.cuh` `componentStressSolve`, without rotational
//! or shear stiffness): projected preconditioned CG in node space on `L = B B^T`.
//!
//! Per solve, for each island (Blast's component):
//! - `r = b - B x0` (warm) or `b`; `delta^2 = tol^2 |b|^2`;
//! - a zero load retires at once with zero forces (`retireHomogeneousTreeComponent`);
//! - each iteration, in the device's order:
//!   1. `r` projected off the island's rigid modes (a free island; `prepareNativeResidualComponent`);
//!   2. `num = |B^T r|^2`; the relative residual `|r|^2 <= delta^2` must also hold (`hold`);
//!   3. passing, after an update or on a warm start: the true residual rebuilt from the
//!      accumulated solution, the recurrence restarted, and both tests taken again (reliable
//!      residual);
//!   4. stagnation: no 1% gain on the best `num` in 512 iterations stops, unconverged;
//!   5. `num <= delta^2` and no hold: converged;
//!   6. the force test (`force_tolerance`): the last step moved the forces by at most that
//!      fraction of them;
//!   7. the direction source `g`: `r` on iteration 0 (projected steepest descent), else the
//!      two-step Chebyshev block-Jacobi polynomial `P = (a+b-ab)M - ab M E M` (`M` the inverse
//!      of each node's 6x6 block of `L`, `E` the rest of `L`); projected, scaled by its
//!      largest component; `gamma = r.g` must be positive;
//!   8. `beta = gamma / gamma_prev` from iteration 2 on (iteration 1 restarts: the
//!      preconditioner changed); `p = g + beta p`;
//!   9. `q = L p` explicitly, `den = |B^T p|^2`; a non-positive or non-finite `den` retires;
//!   10. `alpha = gamma / den`, `u += alpha p`, `r -= alpha q`;
//! - then `x = x0 + B^T u`.
//!
//! The device builds the rigid modes in float-pair arithmetic from an exact spanning-forest
//! frame (`StressMotionPair*.cuh`); here they are the island's node positions relative to its
//! weighted centre, solved exactly in `T`.

use super::*;

/// The polynomial's weights: the reciprocals of the degree-2 Chebyshev roots on [0.1, 2.01]
/// (`StressNativePolynomial.cuh`).
pub(crate) const LOW_WEIGHT: f64 = 0.5779388123770052;
pub(crate) const HIGH_WEIGHT: f64 = 2.6335678180143502;
/// Iterations without a 1% gain on the best residual before a solve stops (`kStagnation`).
pub(crate) const STAGNATION: u32 = 512;

type Block<T> = [[T; 6]; 6];

fn flat<T: Real>(p: Pair<T>) -> [T; 6] {
    [p.a[0], p.a[1], p.a[2], p.l[0], p.l[1], p.l[2]]
}
fn pair<T: Real>(v: [T; 6]) -> Pair<T> {
    Pair {
        a: [v[0], v[1], v[2]],
        l: [v[3], v[4], v[5]],
    }
}

/// Solve `a x = b` (n <= 6) by Gaussian elimination with partial pivoting; `None` when singular.
fn solve_dense<T: Real>(mut a: Block<T>, mut b: [T; 6], n: usize) -> Option<[T; 6]> {
    let zero = T::default();
    let abs = |x: T| if x < zero { -x } else { x };
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| {
            abs(a[i][c])
                .partial_cmp(&abs(a[j][c]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if !(abs(a[p][c]) > zero) {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] = a[r][k] - f * a[c][k];
            }
            b[r] = b[r] - f * b[c];
        }
    }
    let mut x = [zero; 6];
    for r in (0..n).rev() {
        let mut s = b[r];
        for k in r + 1..n {
            s = s - a[r][k] * x[k];
        }
        x[r] = s / a[r][r];
    }
    Some(x)
}

fn invert<T: Real>(a: &Block<T>) -> Option<Block<T>> {
    let zero = T::default();
    let mut inv = [[zero; 6]; 6];
    for c in 0..6 {
        let mut e = [zero; 6];
        e[c] = T::of(1.0);
        let col = solve_dense(*a, e, 6)?;
        for r in 0..6 {
            inv[r][c] = col[r];
        }
    }
    Some(inv)
}

fn mul<T: Real>(m: &Block<T>, v: [T; 6]) -> [T; 6] {
    let mut out = [T::default(); 6];
    for r in 0..6 {
        let mut s = T::default();
        for c in 0..6 {
            s = s + m[r][c] * v[c];
        }
        out[r] = s;
    }
    out
}

/// One island's operator in `T`: its nodes, its live bonds, and the graph's fixed data.
struct Island<'a, T> {
    nodes: Vec<usize>,
    bonds: Vec<usize>,
    /// Each island node's incident live bonds (bond, is second).
    incident: Vec<Vec<(usize, bool)>>,
    slot: &'a [usize],
    w: &'a [(T, T)],
    s: &'a [T],
    o0: &'a [[T; 3]],
    o1: &'a [[T; 3]],
    ends: &'a [[usize; 2]],
}

impl<T: Real> Island<'_, T> {
    /// `t_j = S C^T D v` for bond `j` (v by island node slot; nodes outside the island are 0).
    fn term(&self, j: usize, v: &[Pair<T>]) -> Pair<T> {
        let [n0, n1] = self.ends[j];
        let at = |n: usize| -> Pair<T> {
            let k = self.slot[n];
            if k == usize::MAX {
                Pair::default()
            } else {
                Pair {
                    a: scale(v[k].a, self.w[n].0),
                    l: scale(v[k].l, self.w[n].1),
                }
            }
        };
        let (p0, p1) = (at(n0), at(n1));
        let s = self.s[j];
        Pair {
            a: scale(sub(p0.a, p1.a), s),
            l: scale(
                add(
                    sub(p0.l, p1.l),
                    sub(cross(self.o0[j], p0.a), cross(self.o1[j], p1.a)),
                ),
                s,
            ),
        }
    }

    /// Bond `j`'s share (bond vector `y`, carrying `S` once more) at an endpoint, before `D`.
    fn share(&self, j: usize, second: bool, y: Pair<T>) -> Pair<T> {
        let (ya, yl) = (scale(y.a, self.s[j]), scale(y.l, self.s[j]));
        if second {
            Pair {
                a: sub(cross(self.o1[j], yl), ya),
                l: scale(yl, -T::of(1.0)),
            }
        } else {
            Pair {
                a: sub(ya, cross(self.o0[j], yl)),
                l: yl,
            }
        }
    }

    /// `L v` and `|B^T v|^2`.
    fn apply(&self, v: &[Pair<T>]) -> (Vec<Pair<T>>, T) {
        let mut t = vec![Pair::<T>::default(); self.bonds.len()];
        let mut norm = T::default();
        for (k, &j) in self.bonds.iter().enumerate() {
            t[k] = self.term(j, v);
            norm = norm + t[k].norm2();
        }
        let mut out = vec![Pair::<T>::default(); self.nodes.len()];
        let index: std::collections::HashMap<usize, usize> = self
            .bonds
            .iter()
            .enumerate()
            .map(|(k, &j)| (j, k))
            .collect();
        for (i, &n) in self.nodes.iter().enumerate() {
            let mut acc = Pair::<T>::default();
            for &(j, second) in &self.incident[i] {
                let sh = self.share(j, second, t[index[&j]]);
                acc = Pair {
                    a: add(acc.a, sh.a),
                    l: add(acc.l, sh.l),
                };
            }
            out[i] = Pair {
                a: scale(acc.a, self.w[n].0),
                l: scale(acc.l, self.w[n].1),
            };
        }
        (out, norm)
    }

    /// Node `i`'s 6x6 block of `L` (its own bonds' terms, the other ends held at zero).
    fn block(&self, i: usize) -> Block<T> {
        let n = self.nodes[i];
        let mut b = [[T::default(); 6]; 6];
        for c in 0..6 {
            let mut e = [T::default(); 6];
            e[c] = T::of(1.0);
            let mut v = vec![Pair::<T>::default(); self.nodes.len()];
            v[i] = pair(e);
            let mut acc = Pair::<T>::default();
            for &(j, second) in &self.incident[i] {
                let sh = self.share(j, second, self.term(j, &v));
                acc = Pair {
                    a: add(acc.a, sh.a),
                    l: add(acc.l, sh.l),
                };
            }
            let col = flat(Pair {
                a: scale(acc.a, self.w[n].0),
                l: scale(acc.l, self.w[n].1),
            });
            for r in 0..6 {
                b[r][c] = col[r];
            }
        }
        b
    }
}

/// The rigid modes of a free island, in node-space coordinates (`b_rot,k = (e_k/d_a,
/// (q x e_k)/d_l)`, `b_trans,j = (0, e_j/d_l)`, `q` the node's position about the weighted
/// centre), and their Gram matrix: `project` removes their span (`StressMotionPairModes.cuh`).
struct Modes<T> {
    basis: Vec<Vec<Pair<T>>>,
    gram: Block<T>,
}

impl<T: Real> Modes<T> {
    fn new(isl: &Island<T>, position: &[[T; 3]]) -> Modes<T> {
        let zero = T::default();
        let one = T::of(1.0);
        let (mut wsum, mut c) = (zero, [zero; 3]);
        for (i, &n) in isl.nodes.iter().enumerate() {
            let w = one / (isl.w[n].1 * isl.w[n].1);
            wsum = wsum + w;
            c = add(c, scale(position[i], w));
        }
        let centre = scale(c, one / wsum);
        let mut basis = Vec::with_capacity(6);
        for axis in 0..3 {
            let mut e = [zero; 3];
            e[axis] = one;
            basis.push(
                isl.nodes
                    .iter()
                    .enumerate()
                    .map(|(i, &n)| Pair {
                        a: scale(e, one / isl.w[n].0),
                        l: scale(cross(sub(position[i], centre), e), one / isl.w[n].1),
                    })
                    .collect::<Vec<_>>(),
            );
        }
        for axis in 0..3 {
            let mut e = [zero; 3];
            e[axis] = one;
            basis.push(
                isl.nodes
                    .iter()
                    .map(|&n| Pair {
                        a: [zero; 3],
                        l: scale(e, one / isl.w[n].1),
                    })
                    .collect(),
            );
        }
        let mut gram = [[zero; 6]; 6];
        for a in 0..6 {
            for b in 0..6 {
                let mut s = zero;
                for i in 0..isl.nodes.len() {
                    s = s + dot(basis[a][i].a, basis[b][i].a) + dot(basis[a][i].l, basis[b][i].l);
                }
                gram[a][b] = s;
            }
        }
        Modes { basis, gram }
    }

    fn project(&self, v: &mut [Pair<T>]) {
        let mut rhs = [T::default(); 6];
        for (a, b) in self.basis.iter().enumerate() {
            let mut s = T::default();
            for i in 0..v.len() {
                s = s + dot(b[i].a, v[i].a) + dot(b[i].l, v[i].l);
            }
            rhs[a] = s;
        }
        let Some(c) = solve_dense(self.gram, rhs, 6) else {
            return;
        };
        for (a, b) in self.basis.iter().enumerate() {
            for i in 0..v.len() {
                v[i] = v[i].plus(b[i], -c[a]);
            }
        }
    }
}

fn dot_all<T: Real>(u: &[Pair<T>], v: &[Pair<T>]) -> T {
    u.iter()
        .zip(v)
        .fold(T::default(), |s, (a, b)| s + dot(a.a, b.a) + dot(a.l, b.l))
}

fn largest<T: Real>(v: &[Pair<T>]) -> T {
    let zero = T::default();
    let abs = |x: T| if x < zero { -x } else { x };
    v.iter()
        .flat_map(|p| flat(*p))
        .fold(zero, |m, x| if abs(x) > m { abs(x) } else { m })
}

/// The resident solve of every island that enters (`live` bonds couple; a skipped island's are
/// not live), in `T`. Fills each report's iterations, convergence and norms.
pub(super) fn native_pcg<T: Real>(
    g: &BlastGraph,
    islands: &Islands,
    accel: &[AngLin],
    live: &[bool],
    warm: bool,
    x_out: &mut [AngLin],
    reports: &mut [IslandReport],
) {
    let (n, m) = (g.node_count(), g.bond_count());
    let zero = T::default();
    let one = T::of(1.0);
    let w: Vec<(T, T)> = g
        .weight
        .iter()
        .map(|w| (T::of(w.ang), T::of(w.lin)))
        .collect();
    let s: Vec<T> = g.col_scale.iter().map(|&v| T::of(v)).collect();
    let o0: Vec<[T; 3]> = g.offset0.iter().map(|&o| of3(o)).collect();
    let o1: Vec<[T; 3]> = g.offset1.iter().map(|&o| of3(o)).collect();
    let ends: Vec<[usize; 2]> = g.bonds.iter().map(|b| b.nodes).collect();
    let recip_length = T::of(1.0 / g.length_scale);
    let force_tolerance = T::of(g.settings.force_tolerance);
    let (a_w, b_w) = (T::of(LOW_WEIGHT), T::of(HIGH_WEIGHT));
    let (diagonal, coupling) = (a_w + b_w - a_w * b_w, a_w * b_w);
    let max_iterations = g.settings.max_iterations;

    let mut members: Vec<(Vec<usize>, Vec<usize>)> = vec![(Vec::new(), Vec::new()); islands.count];
    for i in 0..n {
        if islands.node[i] != NO_ISLAND {
            members[islands.node[i] as usize].0.push(i);
        }
    }
    for j in 0..m {
        if islands.bond[j] != NO_ISLAND && live[j] {
            members[islands.bond[j] as usize].1.push(j);
        }
    }
    let mut slot = vec![usize::MAX; n];
    for (k, (nodes, bonds)) in members.into_iter().enumerate() {
        if reports[k].skipped || bonds.is_empty() {
            continue;
        }
        for (i, &v) in nodes.iter().enumerate() {
            slot[v] = i;
        }
        let mut incident = vec![Vec::new(); nodes.len()];
        let mut anchored = false;
        for &j in &bonds {
            for (side, &v) in ends[j].iter().enumerate() {
                if slot[v] == usize::MAX {
                    anchored = true;
                } else {
                    incident[slot[v]].push((j, side == 1));
                }
            }
        }
        let island_nodes = nodes.clone();
        'solve: {
            let isl = Island {
                nodes,
                bonds,
                incident,
                slot: &slot,
                w: &w,
                s: &s,
                o0: &o0,
                o1: &o1,
                ends: &ends,
            };
            let count = isl.nodes.len();

            // b = -D^-1 a (linear over L), the warm residual, the island's target.
            let b: Vec<Pair<T>> = isl
                .nodes
                .iter()
                .map(|&v| {
                    let ia = if w[v].0 > zero { one / w[v].0 } else { one };
                    let il = if w[v].1 > zero {
                        recip_length / w[v].1
                    } else {
                        recip_length
                    };
                    Pair {
                        a: scale(of3(accel[v].ang), -ia),
                        l: scale(of3(accel[v].lin), -il),
                    }
                })
                .collect();
            let tol = T::of(g.settings.tolerance);
            let delta2 = b.iter().fold(zero, |a, p| a + p.norm2()) * tol * tol;
            let x0: Vec<Pair<T>> = isl
                .bonds
                .iter()
                .map(|&j| Pair {
                    a: of3(x_out[j].ang),
                    l: of3(x_out[j].lin),
                })
                .collect();
            // The true residual b - D C S (x0 + B^T u).
            let residual = |u: Option<&[Pair<T>]>| -> Vec<Pair<T>> {
                let mut out = b.clone();
                for (k, &j) in isl.bonds.iter().enumerate() {
                    let mut x = x0[k];
                    if let Some(u) = u {
                        let t = isl.term(j, u);
                        x = x.plus(t, one);
                    }
                    for (side, &v) in ends[j].iter().enumerate() {
                        let i = isl.slot[v];
                        if i == usize::MAX {
                            continue;
                        }
                        let sh = isl.share(j, side == 1, x);
                        out[i] = Pair {
                            a: sub(out[i].a, scale(sh.a, w[v].0)),
                            l: sub(out[i].l, scale(sh.l, w[v].1)),
                        };
                    }
                }
                out
            };
            let report = &mut reports[k];
            report.target2 = delta2.to();

            // A zero load: zero forces at once (`retireHomogeneousTreeComponent`).
            if !(delta2 > zero) {
                for &j in &isl.bonds {
                    x_out[j] = AngLin::ZERO;
                }
                report.converged = true;
                break 'solve;
            }

            let mut r = if warm { residual(None) } else { b.clone() };
            let mut inverse = Vec::with_capacity(count);
            let mut failed = false;
            for i in 0..count {
                match invert(&isl.block(i)) {
                    Some(m) => inverse.push(m),
                    None => {
                        failed = true;
                        inverse.push([[zero; 6]; 6]);
                    }
                }
            }
            // A free island's rigid modes, about the positions the offsets chain to (midpoint
            // offsets: the node centres over the length scale).
            let modes = (!anchored).then(|| {
                let position: Vec<[T; 3]> = isl
                    .nodes
                    .iter()
                    .map(|&v| scale(of3(g.nodes[v].position), recip_length))
                    .collect();
                Modes::new(&isl, &position)
            });
            let project = |v: &mut Vec<Pair<T>>| {
                if let Some(m) = &modes {
                    m.project(v);
                }
            };
            let blocks: Vec<Block<T>> = (0..count).map(|i| isl.block(i)).collect();
            let precondition = |v: &[Pair<T>]| -> Vec<Pair<T>> {
                let local: Vec<Pair<T>> = (0..count)
                    .map(|i| pair(mul(&inverse[i], flat(v[i]))))
                    .collect();
                let (lv, _) = isl.apply(&local);
                (0..count)
                    .map(|i| {
                        let self_part = mul(&blocks[i], flat(local[i]));
                        let mut off = flat(lv[i]);
                        for c in 0..6 {
                            off[c] = off[c] - self_part[c];
                        }
                        let mo = mul(&inverse[i], off);
                        let l = flat(local[i]);
                        let mut out = [zero; 6];
                        for c in 0..6 {
                            out[c] = diagonal * l[c] - coupling * mo[c];
                        }
                        pair(out)
                    })
                    .collect()
            };

            let mut u = vec![Pair::<T>::default(); count];
            let mut p = vec![Pair::<T>::default(); count];
            let mut previous = zero;
            let (mut best, mut best_iteration) = (T::of(f64::INFINITY), 0u32);
            let mut converged = false;
            let mut iteration = 0u32;
            // The force test's measures: the starting forces and the steps travelled since.
            let force_start = if force_tolerance > zero {
                x0.iter().fold(zero, |a, x| a + x.norm2())
            } else {
                zero
            };
            let force_start = T::of(force_start.to().sqrt());
            let (mut travel, mut step) = (zero, T::of(f64::INFINITY));
            while !failed && iteration < max_iterations {
                project(&mut r);
                let norm = |r: &Vec<Pair<T>>| {
                    let num = isl
                        .bonds
                        .iter()
                        .fold(zero, |a, &j| a + isl.term(j, r).norm2());
                    let hold =
                        num <= delta2 && !(r.iter().fold(zero, |a, p| a + p.norm2()) <= delta2);
                    (num, hold)
                };
                let (mut num, mut hold) = norm(&r);
                if (iteration > 0 || warm) && num <= delta2 && !hold {
                    r = residual(Some(&u));
                    previous = zero;
                    project(&mut r);
                    (num, hold) = norm(&r);
                }
                report.residual2 = num.to();
                if num < best * T::of(0.99) {
                    best = num;
                    best_iteration = iteration;
                } else if iteration - best_iteration >= STAGNATION {
                    break;
                }
                if num <= delta2 && !hold {
                    converged = true;
                    break;
                }
                if force_tolerance > zero
                    && iteration > 0
                    && step <= force_tolerance * (force_start + travel)
                {
                    let exact: T = isl.bonds.iter().enumerate().fold(zero, |a, (k, &j)| {
                        a + x0[k].plus(isl.term(j, &u), one).norm2()
                    });
                    if step <= force_tolerance * T::of(exact.to().sqrt()) {
                        converged = true;
                        break;
                    }
                }
                let mut gdir = if iteration == 0 {
                    r.clone()
                } else {
                    precondition(&r)
                };
                project(&mut gdir);
                let mag = largest(&gdir);
                if !(mag > zero) || !mag.finite() {
                    failed = true;
                    break;
                }
                let inv = one / mag;
                for v in gdir.iter_mut() {
                    *v = Pair {
                        a: scale(v.a, inv),
                        l: scale(v.l, inv),
                    };
                }
                let gamma = dot_all(&r, &gdir);
                if !(gamma > zero) || !gamma.finite() {
                    failed = true;
                    break;
                }
                let beta = if iteration > 1 && previous > zero {
                    gamma / previous
                } else {
                    zero
                };
                for i in 0..count {
                    p[i] = if beta == zero {
                        gdir[i]
                    } else {
                        gdir[i].plus(p[i], beta)
                    };
                }
                let (q, den) = isl.apply(&p);
                if force_tolerance > zero {
                    step = if den > zero && gamma > zero {
                        gamma / T::of(den.to().sqrt())
                    } else {
                        T::of(f64::INFINITY)
                    };
                    if step.finite() {
                        travel = travel + step;
                    }
                }
                previous = gamma;
                iteration += 1;
                if !(den > zero) || !den.finite() {
                    break;
                }
                let alpha = gamma / den;
                for i in 0..count {
                    u[i] = u[i].plus(p[i], alpha);
                    r[i] = r[i].plus(q[i], -alpha);
                }
            }
            report.iterations = iteration;
            report.converged = converged && !failed;

            for (k2, &j) in isl.bonds.iter().enumerate() {
                let x = x0[k2].plus(isl.term(j, &u), one);
                x_out[j] = AngLin {
                    ang: to3(x.a),
                    lin: to3(x.l),
                };
            }
        }
        for v in island_nodes {
            slot[v] = usize::MAX;
        }
    }
    // Broken bonds carry nothing.
    for j in 0..m {
        if islands.bond[j] != NO_ISLAND && !live[j] && !reports[islands.bond[j] as usize].skipped {
            x_out[j] = AngLin::ZERO;
        }
    }
}
