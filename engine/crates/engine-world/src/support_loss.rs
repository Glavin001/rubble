//! Support loss for threshold destructibles (`DestructibleDef::support_loss`): the threshold
//! model breaks bonds only by contact and impact loads, with no static solve, so a structure
//! whose supports are shot away stands as long as one tie to its anchors holds. This pass
//! gives it its weight: after a frame's breaks, each anchored piece's weight flows down its
//! intact bonds to its supports, and a bond carrying more than its compressive capacity
//! breaks. The breaks feed the next frame's, so a collapse cascades forward, a frame at a time.
//!
//! The load path is a flow, not a solve: a chunk hands its weight, and what reached it from
//! above, to the neighbours one bond nearer an anchor (breadth first from the supports), in
//! proportion to those bonds' capacities. It holds a column's or a wall's load to its footing
//! and shares a floor's over its columns; it has no bending, no tension and no redundancy
//! beyond that sharing.

use std::collections::VecDeque;

/// One piece's chunks and the bonds among them, as the pass reads them.
pub struct Piece<'a> {
    /// Chunk masses (kg) and whether each is a support (anchored).
    pub mass: &'a [f32],
    pub support: &'a [bool],
    /// Bonds: chunk a, chunk b (indices into the chunks above), compressive capacity (N).
    pub bonds: &'a [(u32, u32, f32)],
}

/// The bonds (indices into `Piece::bonds`) the piece's weight overloads under gravity `g`
/// (m/s²). A piece with no support carries nothing: it falls.
pub fn overloaded(piece: &Piece, g: f32) -> Vec<u32> {
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
        let total: f64 = down.iter().map(|&i| piece.bonds[i as usize].2.max(0.0) as f64).sum();
        for &i in &down {
            let (a, b, cap) = piece.bonds[i as usize];
            let share = if total > 0.0 { cap.max(0.0) as f64 / total } else { 1.0 / down.len() as f64 };
            let f = load[c] * share;
            carried[i as usize] += f;
            let o = if a as usize == c { b } else { a } as usize;
            load[o] += f;
        }
    }
    (0..piece.bonds.len() as u32)
        .filter(|&i| carried[i as usize] > piece.bonds[i as usize].2 as f64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stack of `n` chunks of `m` kg on a support, bonds of capacity `cap` N.
    fn stack(n: usize, m: f32, cap: f32) -> (Vec<f32>, Vec<bool>, Vec<(u32, u32, f32)>) {
        let mass = std::iter::once(0.0).chain(std::iter::repeat(m).take(n)).collect();
        let support = (0..=n).map(|c| c == 0).collect();
        let bonds = (0..n as u32).map(|c| (c, c + 1, cap)).collect();
        (mass, support, bonds)
    }

    #[test]
    fn a_column_fails_at_the_bond_its_weight_exceeds() {
        // 10 chunks of 100 kg: the bond under chunk k carries (10 - k) * 981 N.
        let (mass, support, bonds) = stack(10, 100.0, 5000.0);
        let piece = Piece { mass: &mass, support: &support, bonds: &bonds };
        // Bonds 0..4 carry 9810, 8829, 7848, 6867, 5886 N: over 5000; bond 5 carries 4905.
        assert_eq!(overloaded(&piece, 9.81), vec![0, 1, 2, 3, 4]);
        let strong = stack(10, 100.0, 1.0e4);
        let piece = Piece { mass: &strong.0, support: &strong.1, bonds: &strong.2 };
        assert!(overloaded(&piece, 9.81).is_empty());
    }

    #[test]
    fn a_slab_shares_its_weight_over_its_columns_by_capacity() {
        // A slab (chunk 3) on two columns (1, 2) on supports (0, 4). Column 1's bonds are
        // twice as strong as column 2's, so they carry two thirds of the slab's 3000 kg.
        let mass = [0.0, 0.0, 0.0, 3000.0, 0.0];
        let support = [true, false, false, false, true];
        let bonds = [(0, 1, 2.0e4), (1, 3, 2.0e4), (4, 2, 1.0e4), (2, 3, 1.0e4)];
        let piece = Piece { mass: &mass, support: &support, bonds: &bonds };
        assert!(overloaded(&piece, 10.0).is_empty(), "20 kN on the strong column, 10 kN on the weak");
        // One column gone: the other takes all 30 kN.
        let bonds = [(0, 1, 2.0e4), (1, 3, 2.0e4)];
        let piece = Piece { mass: &mass, support: &support, bonds: &bonds };
        assert_eq!(overloaded(&piece, 10.0), vec![0, 1]);
    }

    #[test]
    fn an_unsupported_piece_carries_nothing() {
        let (mass, _, bonds) = stack(4, 1.0e6, 1.0);
        let support = vec![false; mass.len()];
        let piece = Piece { mass: &mass, support: &support, bonds: &bonds };
        assert!(overloaded(&piece, 9.81).is_empty());
    }
}
