//! Support loss for threshold destructibles (`DestructibleDef::support_loss`): the threshold
//! model breaks bonds only by contact and impact loads, with no static solve, so a structure
//! whose supports are shot away stands as long as one tie to its anchors holds. This pass
//! gives it its weight: after a frame's breaks, each anchored piece's weight flows down its
//! intact bonds to its supports, and a bond carrying more than its compressive capacity
//! breaks. The breaks feed the next frame's, so a collapse cascades forward, a frame at a time.
//!
//! Tipping: the support bonds that carry load make the piece's footprint. A piece whose centre
//! of mass, along gravity, lies outside that footprint cannot stand. The members on those
//! supports (its last columns and walls) stay standing on them, and the piece above lets go of
//! them: their bonds into it break, so it rests on their tops by contact alone and rolls over
//! them. (Breaking the supports instead drops the piece, members and all, flat; a weld left as a
//! hinge would hold it hanging.)
//!
//! The load path is a flow, not a solve: a chunk hands its weight, and what reached it from
//! above, to the neighbours one bond nearer an anchor (breadth first from the supports), in
//! proportion to those bonds' capacities. It holds a column's or a wall's load to its footing
//! and shares a floor's over its columns; it has no bending, no tension and no redundancy
//! beyond that sharing.
//!
//! Dynamic amplification (`DestructibleDef::support_loss_dynamic_factor`, opt-in): the flow is
//! quasi-static, so a piece that suddenly loses a support loads what is left with its static
//! weight. A load suddenly applied to an elastic system overshoots its static effect: the
//! single-degree-of-freedom step response peaks at `1 + exp(-zeta pi / sqrt(1 - zeta^2))`
//! times the step (2 undamped, zeta the damping ratio). Two conventions apply that:
//!
//! - on the whole post-loss demand: progressive-collapse practice's linear static procedure
//!   multiplies the gravity load of the bays over a removed column by a dynamic increase factor
//!   of up to 2 (GSA 2003 *Progressive Collapse Analysis and Design Guidelines*,
//!   `2(DL + 0.25 LL)`; DoD UFC 4-023-03 *Design of Buildings to Resist Progressive Collapse*,
//!   load increase factor `Omega_LD`). A design check, conservative for a member that already
//!   carried load;
//! - on the change of load: the step response itself. A member that carried `before` and
//!   carries `after` once the support is gone peaks at `before + k (after - before)`. The
//!   stress oracle's explicit runs follow it (`docs/destruction-reactions.md` section 5: b4,
//!   increment ratio 1.944 against 1.939 analytic at zeta 0.02).
//!
//! We use the second: it is the physics of the event we simulate (the pass predicts what
//! breaks, it does not size members), and it is exact both for a member that carried nothing
//! (`k after`) and for one whose load does not change (no amplification). With the factor `k`,
//! a piece whose supports changed this frame breaks a bond whose demand
//! `before + k max(after - before, 0)` exceeds its capacity, `before` the bond's flow load the
//! frame before (0 if it carried none) and `after` its flow load now. Only an increase is
//! amplified: a bond whose load drops peaks at `before`, which it already held. `k` = 1 (the
//! default) is the quasi-static flow. Tipping is geometric and takes no factor.

use std::collections::VecDeque;

/// One piece's chunks and the bonds among them, as the pass reads them.
pub struct Piece<'a> {
    /// Chunk masses (kg) and whether each is a support (anchored).
    pub mass: &'a [f32],
    pub support: &'a [bool],
    /// Chunk centres of mass (any frame, gravity's).
    pub centre: &'a [[f32; 3]],
    /// Bonds: chunk a, chunk b (indices into the chunks above), compressive capacity (N).
    pub bonds: &'a [(u32, u32, f32)],
}

/// A sudden support loss's dynamic amplification: the factor `k` (1 to 2) and, per bond of
/// the piece (`Piece::bonds`' order), its flow load (N) the frame before the loss.
pub struct Sudden<'a> {
    pub factor: f32,
    pub before: &'a [f32],
}

/// What the pass decided for a piece.
pub struct Failing {
    /// The bonds (indices into `Piece::bonds`) that break.
    pub bonds: Vec<u32>,
    /// Every bond's static flow load now (N): the next frame's `Sudden::before`.
    pub load: Vec<f64>,
}

/// What the pass breaks in a piece under gravity `g` (m/s², its vector): the bonds (indices
/// into `Piece::bonds`) its weight overloads; or, when its centre of mass is outside the
/// footprint of its loaded support bonds, the bonds from the members on those supports into
/// the rest of the piece (it topples over them). A piece with no support carries nothing: it
/// falls. With `sudden` (its supports changed this frame) a bond's demand is its load before
/// plus `k` times its increase (the step response); without, its static flow load.
pub fn failing(piece: &Piece, g: [f32; 3], sudden: Option<&Sudden>) -> Failing {
    let magnitude = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let carried = flow(piece, magnitude);
    let over = (0..piece.bonds.len() as u32)
        .filter(|&i| demand(&carried, i as usize, sudden) > piece.bonds[i as usize].2 as f64)
        .collect();
    let out = |bonds| Failing {
        bonds,
        load: carried.clone(),
    };
    // Loaded support bonds, at the middle of their chunks' centres.
    let feet: Vec<u32> = (0..piece.bonds.len() as u32)
        .filter(|&i| {
            let (a, b, _) = piece.bonds[i as usize];
            carried[i as usize] > 0.0 && (piece.support[a as usize] != piece.support[b as usize])
        })
        .collect();
    if feet.is_empty() || magnitude <= 0.0 {
        return out(over);
    }
    let down = g.map(|x| x / magnitude);
    // A plane across gravity: two axes perpendicular to it.
    let any = if down[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalize(cross(down, any));
    let v = cross(down, u);
    let flat = |p: [f32; 3]| [dot(p, u) as f64, dot(p, v) as f64];
    let points: Vec<[f64; 2]> = feet
        .iter()
        .map(|&i| {
            let (a, b, _) = piece.bonds[i as usize];
            let (pa, pb) = (piece.centre[a as usize], piece.centre[b as usize]);
            flat([
                0.5 * (pa[0] + pb[0]),
                0.5 * (pa[1] + pb[1]),
                0.5 * (pa[2] + pb[2]),
            ])
        })
        .collect();
    let total: f64 = piece.mass.iter().map(|&m| m as f64).sum();
    if total <= 0.0 {
        return out(over);
    }
    let com = (0..piece.mass.len()).fold([0.0f64; 2], |acc, c| {
        let p = flat(piece.centre[c]);
        let m = piece.mass[c] as f64 / total;
        [acc[0] + p[0] * m, acc[1] + p[1] * m]
    });
    if inside(&hull(points), com) {
        return out(over);
    }
    // The members standing on the feet, and their bonds to anything but supports and each other.
    let mut member = vec![false; piece.mass.len()];
    for &i in &feet {
        let (a, b, _) = piece.bonds[i as usize];
        member[if piece.support[a as usize] { b } else { a } as usize] = true;
    }
    out((0..piece.bonds.len() as u32)
        .filter(|&i| {
            let (a, b, _) = piece.bonds[i as usize];
            let (a, b) = (a as usize, b as usize);
            (member[a] != member[b]) && !piece.support[a] && !piece.support[b]
        })
        .collect())
}

/// Bond `i`'s demand (N) from its static flow load now (`carried`): that load, or after a
/// sudden loss its load before plus `k` times the increase.
fn demand(carried: &[f64], i: usize, sudden: Option<&Sudden>) -> f64 {
    let after = carried[i];
    match sudden {
        None => after,
        Some(s) => {
            let before = s.before.get(i).copied().unwrap_or(0.0) as f64;
            after + (s.factor as f64 - 1.0) * (after - before).max(0.0)
        }
    }
}

/// The bonds (indices into `Piece::bonds`) the piece's weight overloads under gravity `g`
/// (m/s², its magnitude): compression only, no tipping.
#[cfg(test)]
fn overloaded(piece: &Piece, g: f32) -> Vec<u32> {
    let carried = flow(piece, g);
    (0..piece.bonds.len() as u32)
        .filter(|&i| carried[i as usize] > piece.bonds[i as usize].2 as f64)
        .collect()
}

/// The load flow: every bond's static load (N).
fn flow(piece: &Piece, g: f32) -> Vec<f64> {
    let n = piece.mass.len();
    let mut adjacent: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, &(a, b, _)) in piece.bonds.iter().enumerate() {
        adjacent[a as usize].push(i as u32);
        adjacent[b as usize].push(i as u32);
    }
    // Bonds from the supports, breadth first.
    let mut depth = vec![u32::MAX; n];
    let mut queue = VecDeque::new();
    for c in 0..n {
        if piece.support[c] {
            depth[c] = 0;
            queue.push_back(c);
        }
    }
    let mut order = Vec::with_capacity(n);
    while let Some(c) = queue.pop_front() {
        order.push(c);
        for &i in &adjacent[c] {
            let (a, b, _) = piece.bonds[i as usize];
            let o = if a as usize == c { b } else { a } as usize;
            if depth[o] == u32::MAX {
                depth[o] = depth[c] + 1;
                queue.push_back(o);
            }
        }
    }
    // Deepest first: each chunk passes its load on before the chunks it rests on do theirs.
    let mut load: Vec<f64> = piece.mass.iter().map(|&m| (m * g) as f64).collect();
    let mut carried = vec![0.0f64; piece.bonds.len()];
    for &c in order.iter().rev() {
        if depth[c] == 0 {
            continue;
        }
        let down: Vec<u32> = adjacent[c]
            .iter()
            .copied()
            .filter(|&i| {
                let (a, b, _) = piece.bonds[i as usize];
                let o = if a as usize == c { b } else { a } as usize;
                depth[o] + 1 == depth[c]
            })
            .collect();
        let total: f64 = down
            .iter()
            .map(|&i| piece.bonds[i as usize].2.max(0.0) as f64)
            .sum();
        for &i in &down {
            let (a, b, cap) = piece.bonds[i as usize];
            let share = if total > 0.0 {
                cap.max(0.0) as f64 / total
            } else {
                1.0 / down.len() as f64
            };
            let f = load[c] * share;
            carried[i as usize] += f;
            let o = if a as usize == c { b } else { a } as usize;
            load[o] += f;
        }
    }
    carried
}

/// Which of the pieces a frame's breaks touched lost supports in them: a piece lost support
/// when the piece it was part of before the frame (it and the pieces the frame's broken bonds
/// join it to: those bonds were intact, so their ends were one piece) held support chunks it
/// no longer holds. A chip knocked off a wall changes nothing; a column shot away does.
///
/// `bodies`: the touched pieces' bodies, sorted; `chunks_of`: each one's chunks; `ends`: a
/// broken bond's two chunks' bodies now; `is_support`: whether a chunk is anchored.
pub fn lost_support(
    bodies: &[u32],
    chunks_of: &[Vec<u32>],
    broken: &[u32],
    ends: impl Fn(u32) -> (u32, u32),
    is_support: impl Fn(u32) -> bool,
) -> Vec<bool> {
    let mut parent: Vec<usize> = (0..bodies.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for &b in broken {
        let (x, y) = ends(b);
        if let (Ok(x), Ok(y)) = (bodies.binary_search(&x), bodies.binary_search(&y)) {
            let (x, y) = (root(&mut parent, x), root(&mut parent, y));
            parent[x] = y;
        }
    }
    let supports: Vec<usize> = chunks_of
        .iter()
        .map(|c| c.iter().filter(|&&c| is_support(c)).count())
        .collect();
    let mut before = vec![0usize; bodies.len()];
    for i in 0..bodies.len() {
        let r = root(&mut parent, i);
        before[r] += supports[i];
    }
    (0..bodies.len())
        .map(|i| before[root(&mut parent, i)] > supports[i])
        .collect()
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let n = dot(a, a).sqrt().max(1e-12);
    a.map(|x| x / n)
}

fn turn(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Convex hull, counter-clockwise (Andrew's monotone chain); collinear points dropped.
fn hull(mut p: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let mut h: Vec<[f64; 2]> = Vec::with_capacity(2 * p.len());
    for pass in 0..2 {
        let start = h.len();
        let points: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 {
            Box::new(p.iter())
        } else {
            Box::new(p.iter().rev())
        };
        for &q in points {
            while h.len() >= start + 2 && turn(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
                h.pop();
            }
            h.push(q);
        }
        h.pop();
    }
    h
}

/// Whether `q` is on or inside a counter-clockwise hull (a point or a segment: on it, within
/// a millimetre).
fn inside(h: &[[f64; 2]], q: [f64; 2]) -> bool {
    const ON: f64 = 1e-3;
    match h.len() {
        0 => false,
        1 => (h[0][0] - q[0]).hypot(h[0][1] - q[1]) <= ON,
        2 => {
            let (a, b) = (h[0], h[1]);
            let len = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-12);
            let t = ((q[0] - a[0]) * (b[0] - a[0]) + (q[1] - a[1]) * (b[1] - a[1])) / (len * len);
            (-ON..=1.0 + ON).contains(&t) && turn(a, b, q).abs() / len <= ON
        }
        n => (0..n).all(|i| turn(h[i], h[(i + 1) % n], q) >= -ON * 1e-3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stack of `n` chunks of `m` kg on a support, bonds of capacity `cap` N.
    fn stack(n: usize, m: f32, cap: f32) -> (Vec<f32>, Vec<bool>, Vec<(u32, u32, f32)>) {
        let mass = std::iter::once(0.0)
            .chain(std::iter::repeat(m).take(n))
            .collect();
        let support = (0..=n).map(|c| c == 0).collect();
        let bonds = (0..n as u32).map(|c| (c, c + 1, cap)).collect();
        (mass, support, bonds)
    }

    #[test]
    fn a_column_fails_at_the_bond_its_weight_exceeds() {
        // 10 chunks of 100 kg: the bond under chunk k carries (10 - k) * 981 N.
        let (mass, support, bonds) = stack(10, 100.0, 5000.0);
        let centre = vec![[0.0; 3]; mass.len()];
        let piece = Piece {
            mass: &mass,
            support: &support,
            centre: &centre,
            bonds: &bonds,
        };
        // Bonds 0..4 carry 9810, 8829, 7848, 6867, 5886 N: over 5000; bond 5 carries 4905.
        assert_eq!(overloaded(&piece, 9.81), vec![0, 1, 2, 3, 4]);
        let strong = stack(10, 100.0, 1.0e4);
        let piece = Piece {
            mass: &strong.0,
            support: &strong.1,
            centre: &centre,
            bonds: &strong.2,
        };
        assert!(overloaded(&piece, 9.81).is_empty());
    }

    #[test]
    fn a_slab_shares_its_weight_over_its_columns_by_capacity() {
        // A slab (chunk 3) on two columns (1, 2) on supports (0, 4). Column 1's bonds are
        // twice as strong as column 2's, so they carry two thirds of the slab's 3000 kg.
        let mass = [0.0, 0.0, 0.0, 3000.0, 0.0];
        let support = [true, false, false, false, true];
        let bonds = [(0, 1, 2.0e4), (1, 3, 2.0e4), (4, 2, 1.0e4), (2, 3, 1.0e4)];
        let centre = [[0.0; 3]; 5];
        let piece = Piece {
            mass: &mass,
            support: &support,
            centre: &centre,
            bonds: &bonds,
        };
        assert!(
            overloaded(&piece, 10.0).is_empty(),
            "20 kN on the strong column, 10 kN on the weak"
        );
        // One column gone: the other takes all 30 kN.
        let bonds = [(0, 1, 2.0e4), (1, 3, 2.0e4)];
        let piece = Piece {
            mass: &mass,
            support: &support,
            centre: &centre,
            bonds: &bonds,
        };
        assert_eq!(overloaded(&piece, 10.0), vec![0, 1]);
    }

    #[test]
    fn an_unsupported_piece_carries_nothing() {
        let (mass, _, bonds) = stack(4, 1.0e6, 1.0);
        let support = vec![false; mass.len()];
        let centre = vec![[0.0; 3]; mass.len()];
        let piece = Piece {
            mass: &mass,
            support: &support,
            centre: &centre,
            bonds: &bonds,
        };
        assert!(overloaded(&piece, 9.81).is_empty());
    }

    /// A 4 x 4 m slab (chunk 0, 10 t) at height 3 on columns (chunks 1..) over footings (the
    /// supports after them), at the corners given; Y up.
    fn table(corners: &[[f32; 2]]) -> (Vec<f32>, Vec<bool>, Vec<[f32; 3]>, Vec<(u32, u32, f32)>) {
        let n = corners.len() as u32;
        let mut mass = vec![1.0e4];
        let mut centre = vec![[2.0, 3.0, 2.0]];
        let mut support = vec![false];
        let mut bonds = Vec::new();
        for (k, c) in corners.iter().enumerate() {
            mass.push(100.0);
            centre.push([c[0], 1.5, c[1]]);
            support.push(false);
            bonds.push((0, 1 + k as u32, 1.0e9));
        }
        for (k, c) in corners.iter().enumerate() {
            mass.push(0.0);
            centre.push([c[0], 0.0, c[1]]);
            support.push(true);
            bonds.push((1 + k as u32, 1 + n + k as u32, 1.0e9));
        }
        (mass, support, centre, bonds)
    }

    #[test]
    fn a_table_stands_on_its_legs_and_tips_off_one_side() {
        let g = [0.0, -9.81, 0.0];
        let (m, s, c, b) = table(&[[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]]);
        let piece = Piece {
            mass: &m,
            support: &s,
            centre: &c,
            bonds: &b,
        };
        assert!(
            failing(&piece, g, None).bonds.is_empty(),
            "four legs round its centre"
        );
        // Two legs on one side: its centre (x = 2) is off the line x = 4 they make.
        let (m, s, c, b) = table(&[[4.0, 0.0], [4.0, 4.0]]);
        let piece = Piece {
            mass: &m,
            support: &s,
            centre: &c,
            bonds: &b,
        };
        assert_eq!(
            failing(&piece, g, None).bonds,
            vec![0, 1],
            "the slab lets go of both legs, which stay on their feet: it topples"
        );
        // Three legs round the centre still hold it.
        let (m, s, c, b) = table(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]]);
        let piece = Piece {
            mass: &m,
            support: &s,
            centre: &c,
            bonds: &b,
        };
        assert!(
            failing(&piece, g, None).bonds.is_empty(),
            "the centre (2, 2) is on the triangle's edge"
        );
        let (m, s, c, b) = table(&[[1.0, 0.0], [4.0, 0.0], [4.0, 4.0]]);
        let piece = Piece {
            mass: &m,
            support: &s,
            centre: &c,
            bonds: &b,
        };
        assert_eq!(
            failing(&piece, g, None).bonds,
            vec![0, 1, 2],
            "(2, 2) is outside (1,0)-(4,0)-(4,4)"
        );
    }

    /// A wall `w` chunks wide and `h` high on a row of footings, every chunk bonded to its
    /// neighbours (unequal capacities, so the shares are uneven), Y up.
    fn wall(w: usize, h: usize) -> (Vec<f32>, Vec<bool>, Vec<[f32; 3]>, Vec<(u32, u32, f32)>) {
        let index = |x: usize, y: usize| (y * w + x) as u32;
        let (mut mass, mut support, mut centre, mut bonds) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for y in 0..=h {
            for x in 0..w {
                mass.push(if y == 0 {
                    0.0
                } else {
                    50.0 + 10.0 * ((x * 7 + y * 3) % 5) as f32
                });
                support.push(y == 0);
                centre.push([x as f32, y as f32, 0.0]);
                if x > 0 {
                    bonds.push((index(x - 1, y), index(x, y), 1.0e6 + 1.0e5 * x as f32));
                }
                if y > 0 {
                    bonds.push((
                        index(x, y - 1),
                        index(x, y),
                        2.0e6 + 3.0e5 * ((x + y) % 4) as f32,
                    ));
                }
            }
        }
        (mass, support, centre, bonds)
    }

    #[test]
    fn the_load_reaching_the_anchors_is_the_piece_weight() {
        let g = 9.81;
        for (w, h) in [(1, 6), (5, 4), (9, 3)] {
            let (mass, support, centre, bonds) = wall(w, h);
            let piece = Piece {
                mass: &mass,
                support: &support,
                centre: &centre,
                bonds: &bonds,
            };
            let carried = flow(&piece, g);
            let into_anchors: f64 = bonds
                .iter()
                .zip(&carried)
                .filter(|((a, b, _), _)| support[*a as usize] != support[*b as usize])
                .map(|(_, f)| f)
                .sum();
            let weight: f64 = mass
                .iter()
                .zip(&support)
                .filter(|(_, s)| !**s)
                .map(|(m, _)| *m as f64 * g as f64)
                .sum();
            assert!(
                (into_anchors - weight).abs() <= 1e-6 * weight,
                "{w}x{h}: {into_anchors} N reach the anchors of {weight} N"
            );
            // Every load is a share of that weight: nothing is negative or created.
            assert!(carried
                .iter()
                .all(|&f| (0.0..=weight * (1.0 + 1e-6)).contains(&f)));
        }
    }

    #[test]
    fn a_dynamic_factor_amplifies_the_increase_of_load_and_only_that() {
        // A slab (chunk 1, 1000 kg) on one column bond of capacity 12 kN under 10 m/s²:
        // 10 kN static now, 83% utilised.
        let mass = [0.0, 1000.0];
        let support = [true, false];
        let centre = [[0.0; 3]; 2];
        let bonds = [(0, 1, 1.2e4)];
        let piece = Piece {
            mass: &mass,
            support: &support,
            centre: &centre,
            bonds: &bonds,
        };
        let g = [0.0, -10.0, 0.0];
        let sudden = |factor, before: &'static [f32]| Sudden { factor, before };
        let breaks = |s: Option<&Sudden>| failing(&piece, g, s).bonds;
        assert!(breaks(None).is_empty());
        // It carried nothing before (the slab was on another support): demand k * 10 kN,
        // over 12 kN once k > 1.2.
        assert!(breaks(Some(&sudden(1.19, &[0.0]))).is_empty());
        assert_eq!(breaks(Some(&sudden(1.21, &[0.0]))), vec![0]);
        assert_eq!(breaks(Some(&sudden(2.0, &[0.0]))), vec![0]);
        // It carried 7.5 kN before (a share of the slab): demand 7.5 + k * 2.5 kN, over
        // 12 kN once k > 1.8. The whole-demand rule (k * 10 kN) would break it at 1.21.
        assert!(breaks(Some(&sudden(1.79, &[7.5e3]))).is_empty());
        assert_eq!(breaks(Some(&sudden(1.81, &[7.5e3]))), vec![0]);
        // Its load did not change, or dropped: no amplification at any k.
        assert!(breaks(Some(&sudden(2.0, &[1.0e4]))).is_empty());
        assert!(breaks(Some(&sudden(2.0, &[1.1e4]))).is_empty());
        // The loads reported are the static flow's (the next frame's `before`).
        assert_eq!(
            failing(&piece, g, Some(&sudden(2.0, &[0.0]))).load,
            vec![1.0e4]
        );
        assert_eq!(demand(&[1.0e4], 0, Some(&sudden(2.0, &[7.5e3]))), 1.25e4);
    }

    #[test]
    fn a_piece_lost_support_when_its_old_piece_held_anchors_it_no_longer_does() {
        // Chunks 0, 1 anchors; a frame broke bond 0 (chunks 2-3: a chip, 3, off piece 10) and
        // bond 1 (chunks 4-5: column 5 with anchor 1 off piece 10). After: piece 10 holds 0, 2,
        // 4; piece 11 the chip; piece 12 the column and anchor 1.
        let body = [10u32, 12, 10, 11, 10, 12];
        let bonds = [(2u32, 3u32), (4, 5)];
        let bodies = [10u32, 11, 12];
        let chunks_of = vec![vec![0, 2, 4], vec![3], vec![1, 5]];
        let ends = |b: u32| {
            let (a, o) = bonds[b as usize];
            (body[a as usize], body[o as usize])
        };
        let lost = lost_support(&bodies, &chunks_of, &[0, 1], ends, |c| c < 2);
        assert_eq!(
            lost,
            vec![true, true, true],
            "10 lost anchor 1; 12 lost anchor 0; 11 lost both"
        );
        // The chip alone: piece 10 keeps both anchors (11 had none).
        let body = [10u32, 10, 10, 11, 10, 10];
        let ends = |b: u32| {
            let (a, o) = bonds[b as usize];
            (body[a as usize], body[o as usize])
        };
        let chunks_of = vec![vec![0, 1, 2, 4, 5], vec![3]];
        let lost = lost_support(&[10, 11], &chunks_of, &[0], ends, |c| c < 2);
        assert_eq!(lost, vec![false, true]);
    }
}
