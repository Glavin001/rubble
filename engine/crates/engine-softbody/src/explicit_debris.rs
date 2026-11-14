//! Pieces torn off an explicit vehicle: which nodes are still joined to its main body.
//!
//! "Joined" is by live (unbroken) beams only: the pieces are the connected components of
//! the beams that have not broken, and the main body is the heaviest of them. Any host
//! can run this on a beam readback (the GPU's broken flags, or the CPU's `Beam::broken`):
//! it is a union-find over the beams, O(beams) a call.
//!
//! [`piece_labels`] turns the pieces into the GPU's reused pair list's labels
//! (`ExplicitBeamsGpu::set_piece_labels`), a hint that lets a piece in free fall stop
//! invalidating the main body's list (lossless whatever the labels).

use crate::explicit_beams::ExplicitBeams;

/// The connected components of a vehicle's unbroken beams.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pieces {
    /// Per node its piece (an index into `mass`).
    pub piece: Vec<u32>,
    /// Per piece its mass (kg) and node count.
    pub mass: Vec<f32>,
    pub nodes: Vec<u32>,
    /// The heaviest piece (the main body; the lowest index on a tie).
    pub main: u32,
}

impl Pieces {
    /// The pieces of `s` with its beams' broken flags `broken` (one per beam; a missing
    /// flag reads unbroken).
    pub fn of(s: &ExplicitBeams, broken: &[bool]) -> Self {
        let n = s.nodes.len();
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        for (j, b) in s.beams.iter().enumerate() {
            if broken.get(j).copied().unwrap_or(false) {
                continue;
            }
            let (a, c) = (b.a as usize, b.b as usize);
            if a >= n || c >= n {
                continue;
            }
            let (ra, rc) = (find(&mut parent, a as u32), find(&mut parent, c as u32));
            if ra != rc {
                // The lower root wins: labels independent of the beams' order.
                let (lo, hi) = if ra < rc { (ra, rc) } else { (rc, ra) };
                parent[hi as usize] = lo;
            }
        }
        // Pieces numbered by their lowest node.
        let mut id = vec![u32::MAX; n];
        let (mut piece, mut mass, mut nodes) = (vec![0u32; n], Vec::new(), Vec::new());
        for i in 0..n {
            let r = find(&mut parent, i as u32) as usize;
            if id[r] == u32::MAX {
                id[r] = mass.len() as u32;
                mass.push(0.0f32);
                nodes.push(0u32);
            }
            let k = id[r];
            piece[i] = k;
            mass[k as usize] += s.nodes[i].mass;
            nodes[k as usize] += 1;
        }
        let main = (0..mass.len()).fold(0usize, |b, k| if mass[k] > mass[b] { k } else { b }) as u32;
        Self { piece, mass, nodes, main }
    }

    /// Whether node `i` is in the main body.
    pub fn attached(&self, i: usize) -> bool {
        self.piece.get(i) == Some(&self.main)
    }
}

/// The GPU pair list's piece labels for `s` (one per node): 0 the main body; 1..=6 the
/// pieces with the most live collision triangles of their own (all three nodes in the
/// piece; a triangle is dead once one of its break beams broke); 7 (`PIECE_LOOSE`) every
/// other node off the main body (its triangles are then tested every step).
pub fn piece_labels(s: &ExplicitBeams, pieces: &Pieces, broken: &[bool]) -> Vec<u8> {
    const LOOSE: u8 = 7;
    let mut own = vec![0u32; pieces.mass.len()];
    for (t, tri) in s.triangles.iter().enumerate() {
        let dead = s
            .tri_breaks
            .get(t)
            .is_some_and(|k| k.iter().any(|&b| b != u32::MAX && broken.get(b as usize).copied().unwrap_or(false)));
        let p = pieces.piece[tri.nodes[0] as usize];
        if !dead && tri.nodes.iter().all(|&v| pieces.piece[v as usize] == p) {
            own[p as usize] += 1;
        }
    }
    let mut order: Vec<u32> = (0..pieces.mass.len() as u32)
        .filter(|&k| k != pieces.main && own[k as usize] > 0)
        .collect();
    order.sort_by(|a, b| own[*b as usize].cmp(&own[*a as usize]).then(a.cmp(b)));
    let mut label = vec![LOOSE; pieces.mass.len()];
    label[pieces.main as usize] = 0;
    for (k, &p) in order.iter().take(6).enumerate() {
        label[p as usize] = k as u8 + 1;
    }
    pieces.piece.iter().map(|&p| label[p as usize]).collect()
}
