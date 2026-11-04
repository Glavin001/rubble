//! A free island's rigid modes as PhysX's native stage builds them
//! (`stressgpu/detail/StressMotionForest.cuh` and `StressMotionModes.cuh`, the binary64
//! construction; `StressMotionPair*.cuh` is the same construction in float expansions and
//! float pairs, which CuMetal runs and `blast_motion.slang` ports).
//!
//! The modes are not taken from the authored node positions. They come from the live bond
//! equations themselves:
//! - positions: exact sums of the bonds' offset differences `o0 - o1` along a spanning tree of
//!   the island, from its root (its lowest node) at 0;
//! - closures: each live bond off the tree closes a cycle, `p0 + (o0 - o1) - p1`. One larger than
//!   2^-17 of the cycle's own coordinate scale is a real moment, not the rounding of the
//!   offsets; the lowest such bond is the island's closure;
//! - rotations: 3 without a closure; with one, 1 about its axis, or 0 when another significant
//!   closure is not collinear with it (exact products);
//! - anchored (dimension 0): a live bond to a support;
//! - the frame: the centre `sum p/d_l^2 / sum 1/d_l^2`, each node's offset `q` from it, and the
//!   Gram matrix of the basis `(axis_k / d_a, (q x axis_k) / d_l)`, `(0, e_j / d_l)`, factored
//!   after scaling by its diagonal (`buildMotionFactor`).
//!
//! The target's tree is whichever bond won each union of a lock-free union-find, a race; the
//! exact sums make the positions the same on any tree up to the rounding residue of the
//! offsets. Here the tree is deterministic: the islands are walked in levels from the root, and
//! a node's tree bond is its lowest-index live bond to a node of the level before.
//!
//! Exact arithmetic is Shewchuk's (Adaptive Precision Floating-Point Arithmetic, 1997):
//! nonoverlapping f64 expansions with zero elimination.

/// An exact sum: nonoverlapping terms, increasing magnitude, no zeros.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Exact(Vec<f64>);

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

impl Exact {
    pub(crate) fn of(x: f64) -> Exact {
        Exact(if x == 0.0 { Vec::new() } else { vec![x] })
    }
    /// Shewchuk's grow-expansion of every term of `o` into this one.
    pub(crate) fn add(&self, o: &Exact) -> Exact {
        let mut e = self.0.clone();
        for &b in &o.0 {
            let mut q = b;
            let mut out = Vec::with_capacity(e.len() + 1);
            for &t in &e {
                let (s, h) = two_sum(q, t);
                if h != 0.0 {
                    out.push(h);
                }
                q = s;
            }
            if q != 0.0 {
                out.push(q);
            }
            e = out;
        }
        Exact(e)
    }
    pub(crate) fn neg(&self) -> Exact {
        Exact(self.0.iter().map(|x| -x).collect())
    }
    pub(crate) fn sub(&self, o: &Exact) -> Exact {
        self.add(&o.neg())
    }
    /// The exact product (every term pair's two-product, grown in).
    pub(crate) fn mul(&self, o: &Exact) -> Exact {
        let mut out = Exact::default();
        for &a in &self.0 {
            for &b in &o.0 {
                let p = a * b;
                let e = a.mul_add(b, -p);
                out = out.add(&Exact(if e == 0.0 { vec![p] } else { vec![e, p] }));
            }
        }
        out
    }
    pub(crate) fn scaled(&self, power_of_two: f64) -> Exact {
        Exact(self.0.iter().map(|x| x * power_of_two).collect())
    }
    pub(crate) fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
    /// The sign of the value: its largest term's.
    pub(crate) fn sign(&self) -> f64 {
        self.0.last().map_or(0.0, |x| x.signum())
    }
    pub(crate) fn magnitude(&self) -> Exact {
        if self.sign() < 0.0 {
            self.neg()
        } else {
            self.clone()
        }
    }
    /// `self > o`, exactly.
    pub(crate) fn greater(&self, o: &Exact) -> bool {
        self.sub(o).sign() > 0.0
    }
    /// The value rounded (smallest terms first).
    pub(crate) fn value(&self) -> f64 {
        self.0.iter().sum()
    }
    pub(crate) fn finite(&self) -> bool {
        self.0.iter().all(|x| x.is_finite())
    }
}

pub(crate) type Exact3 = [Exact; 3];

fn add3(a: &Exact3, b: &Exact3) -> Exact3 {
    [a[0].add(&b[0]), a[1].add(&b[1]), a[2].add(&b[2])]
}
fn neg3(a: &Exact3) -> Exact3 {
    [a[0].neg(), a[1].neg(), a[2].neg()]
}
fn max_magnitude(values: impl IntoIterator<Item = Exact>) -> Exact {
    values.into_iter().fold(Exact::default(), |m, v| {
        let v = v.magnitude();
        if v.greater(&m) {
            v
        } else {
            m
        }
    })
}

/// `motionOffset`: a bond's `o0 - o1`, exactly, from the device's float offsets.
fn offset(o0: [f32; 3], o1: [f32; 3]) -> Exact3 {
    std::array::from_fn(|c| Exact::of(o0[c] as f64).sub(&Exact::of(o1[c] as f64)))
}

/// Construction failures (`Status::error` bits as the target sets them).
pub(crate) const ERROR_TOPOLOGY: u32 = 2;
pub(crate) const ERROR_FRAME: u32 = 4;
pub(crate) const ERROR_INEXACT: u32 = 16;

/// `triangle(row, col)`, row >= col: the packed lower triangle.
pub(crate) fn triangle(row: usize, col: usize) -> usize {
    row * (row + 1) / 2 + col
}

/// A free island's frame (`MotionComponent`); `relative` by island node slot.
#[derive(Clone, Debug, Default)]
pub(crate) struct Frame {
    pub anchored: bool,
    pub rotations: usize,
    pub axis: [f64; 3],
    pub relative: Vec<[f64; 3]>,
    pub scale: [f64; 6],
    pub factor: [f64; 21],
    pub error: u32,
}

impl Frame {
    /// `motionDimension`: 0 when anchored, else the rotations and three translations.
    pub(crate) fn dimension(&self) -> usize {
        if self.anchored {
            0
        } else {
            self.rotations + 3
        }
    }
    /// The basis vector of rotation `k`.
    pub(crate) fn axis(&self, k: usize) -> [f64; 3] {
        if self.rotations == 1 {
            self.axis
        } else {
            let mut e = [0.0; 3];
            e[k] = 1.0;
            e
        }
    }
}

/// The island: its nodes (ascending; the root first), each node's island slot (`usize::MAX`
/// outside it: a support), its live bonds (those to supports included), and the graph's data.
pub(crate) struct Input<'a> {
    pub nodes: &'a [usize],
    pub slot: &'a [usize],
    pub bonds: &'a [usize],
    pub ends: &'a [[usize; 2]],
    pub o0: &'a [[f32; 3]],
    pub o1: &'a [[f32; 3]],
    /// Per node `(d_a, d_l)`.
    pub w: &'a [(f64, f64)],
}

/// The deterministic spanning tree: per island node, its tree bond (`None` for the root), by
/// levels from the root, each node taking its lowest-index live bond to the level before.
pub(crate) fn tree(input: &Input) -> Vec<Option<usize>> {
    let count = input.nodes.len();
    let mut level = vec![u32::MAX; count];
    let mut parent = vec![None; count];
    if count == 0 {
        return parent;
    }
    let mut incident = vec![Vec::new(); count];
    for &j in input.bonds {
        let [a, b] = input.ends[j];
        let (sa, sb) = (input.slot[a], input.slot[b]);
        if sa != usize::MAX && sb != usize::MAX {
            incident[sa].push((j, sb));
            incident[sb].push((j, sa));
        }
    }
    level[0] = 0;
    for l in 0.. {
        let mut changed = false;
        for i in 0..count {
            if level[i] != u32::MAX {
                continue;
            }
            let best = incident[i]
                .iter()
                .filter(|&&(_, o)| level[o] == l)
                .map(|&(j, _)| j)
                .min();
            if let Some(j) = best {
                level[i] = l + 1;
                parent[i] = Some(j);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    parent
}

/// Build the frame (`StressMotionForest.cuh` then `buildMotionFactor`).
pub(crate) fn frame(input: &Input) -> Frame {
    let count = input.nodes.len();
    let mut f = Frame::default();
    for &j in input.bonds {
        let [a, b] = input.ends[j];
        if (input.slot[a] == usize::MAX) != (input.slot[b] == usize::MAX) {
            f.anchored = true;
        }
    }
    if f.anchored || count == 0 {
        return f;
    }
    // Positions: the root at 0; each node its parent's plus the tree bond's offset difference.
    let parent = tree(input);
    let mut position: Vec<Option<Exact3>> = vec![None; count];
    position[0] = Some(Default::default());
    let mut error = 0;
    let mut pending: Vec<usize> = (1..count).collect();
    while !pending.is_empty() {
        let before = pending.len();
        pending.retain(|&i| {
            let Some(j) = parent[i] else {
                return true;
            };
            let [a, b] = input.ends[j];
            let (sa, sb) = (input.slot[a], input.slot[b]);
            let (other, d) = if sb == i {
                (sa, offset(input.o0[j], input.o1[j]))
            } else {
                (sb, neg3(&offset(input.o0[j], input.o1[j])))
            };
            match &position[other] {
                Some(p) => {
                    position[i] = Some(add3(p, &d));
                    false
                }
                None => true,
            }
        });
        if pending.len() == before {
            // Unreached: the island is not connected over its live bonds.
            f.error |= ERROR_TOPOLOGY;
            return f;
        }
    }
    let position: Vec<Exact3> = position.into_iter().map(Option::unwrap).collect();
    if position.iter().any(|p| p.iter().any(|c| !c.finite())) {
        error |= ERROR_INEXACT;
    }
    // Closures of the live bonds inside the island.
    let closure = |j: usize| -> Exact3 {
        let [a, b] = input.ends[j];
        add3(
            &add3(&position[input.slot[a]], &offset(input.o0[j], input.o1[j])),
            &neg3(&position[input.slot[b]]),
        )
    };
    let significant = |j: usize, c: &Exact3| -> bool {
        let [a, b] = input.ends[j];
        let (p, q) = (&position[input.slot[a]], &position[input.slot[b]]);
        let offsets = input.o0[j]
            .iter()
            .chain(&input.o1[j])
            .fold(0.0f32, |m, x| m.max(x.abs()));
        let scale = max_magnitude(
            p.iter()
                .chain(q.iter())
                .cloned()
                .chain([Exact::of(offsets as f64)]),
        );
        let magnitude = max_magnitude(c.iter().cloned());
        magnitude.greater(&scale.scaled(f64::powi(2.0, -17)))
    };
    let inside = |j: usize| {
        let [a, b] = input.ends[j];
        input.slot[a] != usize::MAX && input.slot[b] != usize::MAX
    };
    let mut seed = None;
    for &j in input.bonds {
        if !inside(j) {
            continue;
        }
        let c = closure(j);
        if c.iter().any(|v| !v.is_zero()) && significant(j, &c) {
            seed = Some(seed.map_or(j, |s: usize| s.min(j)));
        }
    }
    f.rotations = 3;
    if let Some(s) = seed {
        f.rotations = 1;
        let c = closure(s);
        let axis = [c[0].value(), c[1].value(), c[2].value()];
        let maximum = axis.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        if !(maximum > 0.0) || !maximum.is_finite() {
            f.error = error | ERROR_FRAME;
            return f;
        }
        let axis = axis.map(|x| x / maximum);
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        f.axis = axis.map(|x| x / norm);
        for &j in input.bonds {
            if !inside(j) {
                continue;
            }
            let v = closure(j);
            if v.iter().all(Exact::is_zero) || !significant(j, &v) {
                continue;
            }
            let collinear = (0..3).all(|k| {
                let (y, z) = ((k + 1) % 3, (k + 2) % 3);
                c[y].mul(&v[z]).sub(&c[z].mul(&v[y])).is_zero()
            });
            if !collinear {
                f.rotations = 0;
            }
        }
    }
    // The frame (binary64, as buildMotionFactor's double version).
    let p: Vec<[f64; 3]> = position
        .iter()
        .map(|p| [p[0].value(), p[1].value(), p[2].value()])
        .collect();
    let (mut wsum, mut c) = (0.0, [0.0; 3]);
    for (i, &n) in input.nodes.iter().enumerate() {
        let w = 1.0 / (input.w[n].1 * input.w[n].1);
        wsum += w;
        for k in 0..3 {
            c[k] += p[i][k] * w;
        }
    }
    if !(wsum > 0.0) || !wsum.is_finite() {
        f.error = error | ERROR_FRAME;
        return f;
    }
    let centre = c.map(|x| x / wsum);
    let dimension = f.rotations + 3;
    let axial = if f.rotations == 1 {
        f.axis.iter().map(|x| x * x).sum::<f64>()
    } else {
        1.0
    };
    let mut gram = [0.0; 21];
    f.relative = Vec::with_capacity(count);
    for (i, &n) in input.nodes.iter().enumerate() {
        let (da, dl) = input.w[n];
        let (w, angular) = (1.0 / (dl * dl), 1.0 / (da * da));
        let q = [
            p[i][0] - centre[0],
            p[i][1] - centre[1],
            p[i][2] - centre[2],
        ];
        f.relative.push(q);
        let moment: Vec<[f64; 3]> = (0..f.rotations).map(|k| cross(q, f.axis(k))).collect();
        for row in 0..dimension {
            for col in 0..=row {
                let value = if row < f.rotations {
                    (if row == col { angular * axial } else { 0.0 })
                        + w * dot(moment[row], moment[col])
                } else if col < f.rotations {
                    w * moment[col][row - f.rotations]
                } else if row == col {
                    w
                } else {
                    0.0
                };
                gram[triangle(row, col)] += value;
            }
        }
    }
    for i in 0..dimension {
        let d = gram[triangle(i, i)];
        if !(d > 0.0) || !d.is_finite() {
            f.error = error | ERROR_FRAME;
            return f;
        }
        f.scale[i] = 1.0 / d.sqrt();
    }
    for row in 0..dimension {
        for col in 0..=row {
            let mut value = gram[triangle(row, col)] * f.scale[row] * f.scale[col];
            for k in 0..col {
                value -= f.factor[triangle(row, k)] * f.factor[triangle(col, k)];
            }
            if row == col {
                if !(value > 0.0) || !value.is_finite() {
                    f.error = error | ERROR_FRAME;
                    return f;
                }
                value = value.sqrt();
            } else {
                value /= f.factor[triangle(col, col)];
            }
            f.factor[triangle(row, col)] = value;
        }
    }
    f.error = error;
    f
}

fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}
fn dot(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sums_keep_what_rounding_loses() {
        let a = Exact::of(1.0).add(&Exact::of(1e-30)).add(&Exact::of(-1.0));
        assert_eq!(a.value(), 1e-30);
        assert!(Exact::of(1e-30).greater(&Exact::default()));
        let p = Exact::of(1.0 + f64::EPSILON).mul(&Exact::of(1.0 - f64::EPSILON));
        // (1+e)(1-e) = 1 - e^2: not representable in one f64, exact here.
        assert!(p
            .sub(&Exact::of(1.0))
            .add(&Exact::of(f64::EPSILON * f64::EPSILON))
            .is_zero());
    }

    fn chain(
        points: &[[f32; 3]],
        extra: &[(usize, usize)],
    ) -> (Vec<[usize; 2]>, Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let mut ends = Vec::new();
        let mut o0 = Vec::new();
        let mut o1 = Vec::new();
        let edges: Vec<(usize, usize)> = (1..points.len())
            .map(|i| (i - 1, i))
            .chain(extra.iter().copied())
            .collect();
        for (a, b) in edges {
            let d: [f32; 3] = std::array::from_fn(|k| (points[b][k] - points[a][k]) * 0.5);
            ends.push([a, b]);
            o0.push(d);
            o1.push(d.map(|x| -x));
        }
        (ends, o0, o1)
    }

    fn build(
        points: &[[f32; 3]],
        extra: &[(usize, usize)],
        o0_edit: impl Fn(&mut Vec<[f32; 3]>),
    ) -> Frame {
        let (ends, mut o0, o1) = chain(points, extra);
        o0_edit(&mut o0);
        let nodes: Vec<usize> = (0..points.len()).collect();
        let bonds: Vec<usize> = (0..ends.len()).collect();
        let w = vec![(1.0, 1.0); points.len()];
        frame(&Input {
            nodes: &nodes,
            slot: &nodes,
            bonds: &bonds,
            ends: &ends,
            o0: &o0,
            o1: &o1,
            w: &w,
        })
    }

    #[test]
    fn a_consistent_ring_keeps_all_three_rotations() {
        let ring = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.3],
        ];
        let f = build(&ring, &[(3, 0)], |_| {});
        assert_eq!((f.anchored, f.rotations, f.error), (false, 3, 0));
        // The positions chain back to the authored ones (centred).
        let c: Vec<f64> = (0..3)
            .map(|k| ring.iter().map(|p| p[k] as f64).sum::<f64>() / 4.0)
            .collect();
        for (q, p) in f.relative.iter().zip(&ring) {
            for k in 0..3 {
                assert!((q[k] - (p[k] as f64 - c[k])).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn a_real_closure_leaves_one_rotation_and_two_leave_none() {
        let ring = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // One offset off by 0.1 along z: that cycle's closure is a moment about z... its axis.
        let one = build(&ring, &[(3, 0)], |o0| o0[3][2] += 0.1);
        assert_eq!((one.rotations, one.error), (1, 0));
        assert!((one.axis[2].abs() - 1.0).abs() < 1e-12);
        let two = build(&ring, &[(3, 0), (0, 2)], |o0| {
            o0[3][2] += 0.1;
            o0[4][0] += 0.1;
        });
        assert_eq!(two.rotations, 0);
        // Rounding-sized closures are no moment.
        let rounding = build(&ring, &[(3, 0)], |o0| o0[3][2] += 1e-9);
        assert_eq!(rounding.rotations, 3);
    }

    #[test]
    fn a_bond_to_a_support_anchors() {
        let ends = vec![[0, 1], [1, 2]];
        let o = vec![[0.5f32, 0.0, 0.0]; 2];
        let o1 = vec![[-0.5f32, 0.0, 0.0]; 2];
        let nodes = [0usize, 1];
        let slot = [0, 1, usize::MAX];
        let w = vec![(1.0, 1.0); 3];
        let f = frame(&Input {
            nodes: &nodes,
            slot: &slot,
            bonds: &[0, 1],
            ends: &ends,
            o0: &o,
            o1: &o1,
            w: &w,
        });
        assert!(f.anchored);
        assert_eq!(f.dimension(), 0);
    }
}
