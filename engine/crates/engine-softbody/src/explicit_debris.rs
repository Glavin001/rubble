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

/// Time to live for small debris (an explicit, engine-wide setting; off unless a host
/// sets it). A piece that has been off the vehicle's main body (no live beam joins
/// them) for `seconds`, of at most `max_mass_kg` and `max_nodes`, is retired: its nodes
/// are no longer simulated and cost nothing but their own beams' updates
/// (`ExplicitBeams::retired`, `ExplicitBeamsGpu::set_retired`). The rule:
/// - never the main body (the heaviest piece), anything heavier or larger than the
///   thresholds, nor a piece still joined to the main body by a live beam;
/// - a piece at rest on the ground beside the car is retired too once its time is up: it
///   is debris all the same;
/// - never the piece holding node 0 (the vehicle's frame origin follows node 0);
/// - a piece's time counts from the reading at which its nodes were first seen off the
///   main body (its last node to leave), at the host's beam readings (the lab: 10 a
///   second).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebrisTtl {
    pub max_mass_kg: f32,
    pub max_nodes: u32,
    pub seconds: f32,
}

impl DebrisTtl {
    /// Whether a vehicle can retire debris: not with centred damping (its correction
    /// reads every node) nor with slide nodes (they move nodes along their rails).
    pub fn check(s: &ExplicitBeams) -> Result<(), String> {
        if s.settings.centred_damping {
            return Err("debris time to live needs centred damping off (its correction reads every node)".into());
        }
        if !s.slides.is_empty() {
            return Err("debris time to live needs a vehicle without slide nodes".into());
        }
        Ok(())
    }
}

/// A piece retired at a reading.
#[derive(Clone, Debug, PartialEq)]
pub struct Retired {
    /// The reading's time (s), the piece's nodes and mass (kg), how long it had been off
    /// the main body (s).
    pub t: f64,
    pub nodes: u32,
    pub mass: f32,
    pub off_for: f64,
}

/// The host's side of [`DebrisTtl`]: which nodes are off the main body since when, and
/// which are retired. Feed it every beam reading ([`Self::update`]); when it reports a
/// change, hand [`Self::retired`] to the solver.
#[derive(Clone, Debug, Default)]
pub struct DebrisTracker {
    pub ttl: Option<DebrisTtl>,
    /// Per node: the reading at which it was first seen off the main body.
    off_since: Vec<Option<f64>>,
    /// Per node: retired.
    pub retired: Vec<bool>,
    /// Retired so far: pieces, nodes, mass (kg).
    pub pieces: u32,
    pub nodes: u32,
    pub mass: f32,
    /// Pieces retired since the last [`Self::take_events`].
    events: Vec<Retired>,
}

impl DebrisTracker {
    pub fn new(ttl: Option<DebrisTtl>, nodes: usize) -> Self {
        Self { ttl, off_since: vec![None; nodes], retired: vec![false; nodes], ..Default::default() }
    }

    /// One reading at time `t` of `s`'s pieces: true when [`Self::retired`] changed.
    pub fn update(&mut self, s: &ExplicitBeams, pieces: &Pieces, t: f64) -> bool {
        let n = s.nodes.len().min(pieces.piece.len());
        self.off_since.resize(n, None);
        self.retired.resize(n, false);
        for i in 0..n {
            if pieces.attached(i) {
                self.off_since[i] = None;
            } else if self.off_since[i].is_none() {
                self.off_since[i] = Some(t);
            }
        }
        let Some(ttl) = self.ttl else { return false };
        // Per piece: its latest leave time, whether it holds node 0 or a retired node.
        let k = pieces.mass.len();
        let (mut since, mut origin, mut done) = (vec![f64::MIN; k], vec![false; k], vec![false; k]);
        for i in 0..n {
            let p = pieces.piece[i] as usize;
            since[p] = since[p].max(self.off_since[i].unwrap_or(f64::MAX));
            origin[p] |= i == 0;
            done[p] |= self.retired[i];
        }
        let mut changed = false;
        for p in 0..k {
            if p == pieces.main as usize || done[p] || origin[p] {
                continue;
            }
            let small = pieces.mass[p] <= ttl.max_mass_kg && pieces.nodes[p] <= ttl.max_nodes;
            if !small || since[p] == f64::MAX || t - since[p] < f64::from(ttl.seconds) {
                continue;
            }
            for i in 0..n {
                if pieces.piece[i] as usize == p {
                    self.retired[i] = true;
                }
            }
            self.pieces += 1;
            self.nodes += pieces.nodes[p];
            self.mass += pieces.mass[p];
            self.events.push(Retired { t, nodes: pieces.nodes[p], mass: pieces.mass[p], off_for: t - since[p] });
            changed = true;
        }
        changed
    }

    /// The pieces retired since the last call.
    pub fn take_events(&mut self) -> Vec<Retired> {
        std::mem::take(&mut self.events)
    }
}
