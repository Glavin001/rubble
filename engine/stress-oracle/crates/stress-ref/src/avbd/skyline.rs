//! Profile (skyline) Cholesky for the exterior's Newmark operator, with a reverse
//! Cuthill-McKee ordering of the bodies. Part of the multi-time-step prototype
//! (`mts.rs`); the bonded walls and houses are banded graphs, for which a profile
//! factorisation is close to optimal and needs no fill-in bookkeeping.

/// Reverse Cuthill-McKee order of an undirected graph (`adj`), from a pseudo-peripheral
/// start in every connected component.
pub fn rcm(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let bfs_last = |start: usize, seen_mask: &[bool]| -> (usize, usize) {
        // (last node, depth) of a BFS from `start` over unseen nodes.
        let mut level = vec![usize::MAX; n];
        let mut queue = std::collections::VecDeque::new();
        level[start] = 0;
        queue.push_back(start);
        let mut last = start;
        while let Some(i) = queue.pop_front() {
            last = i;
            for &j in &adj[i] {
                if !seen_mask[j] && level[j] == usize::MAX {
                    level[j] = level[i] + 1;
                    queue.push_back(j);
                }
            }
        }
        (last, level[last])
    };
    for s0 in 0..n {
        if seen[s0] {
            continue;
        }
        // Pseudo-peripheral node: repeat BFS from the farthest node while depth grows.
        let (mut start, mut depth) = bfs_last(s0, &seen);
        loop {
            let (next, d) = bfs_last(start, &seen);
            if d <= depth {
                break;
            }
            start = next;
            depth = d;
        }
        let mut queue = std::collections::VecDeque::new();
        seen[start] = true;
        queue.push_back(start);
        while let Some(i) = queue.pop_front() {
            order.push(i);
            let mut nb: Vec<usize> = adj[i].iter().copied().filter(|&j| !seen[j]).collect();
            nb.sort_by_key(|&j| adj[j].len());
            nb.dedup();
            for j in nb {
                if !seen[j] {
                    seen[j] = true;
                    queue.push_back(j);
                }
            }
        }
    }
    order.reverse();
    order
}

/// A symmetric matrix in profile storage: row `i` holds columns `first[i]..=i`.
#[derive(Clone, Debug)]
pub struct Skyline {
    pub n: usize,
    first: Vec<usize>,
    start: Vec<usize>,
    val: Vec<f64>,
    factored: bool,
}

impl Skyline {
    /// An empty matrix with the profile of the lower-triangle pattern `(i, j)`, `j <= i`.
    pub fn with_pattern(n: usize, pattern: impl Iterator<Item = (usize, usize)>) -> Skyline {
        let mut first: Vec<usize> = (0..n).collect();
        for (i, j) in pattern {
            let (i, j) = if i >= j { (i, j) } else { (j, i) };
            first[i] = first[i].min(j);
        }
        let mut start = Vec::with_capacity(n + 1);
        let mut s = 0;
        for i in 0..n {
            start.push(s);
            s += i - first[i] + 1;
        }
        start.push(s);
        Skyline { n, first, start, val: vec![0.0; s], factored: false }
    }

    /// Zero every entry (keeping the profile).
    pub fn clear(&mut self) {
        self.val.iter_mut().for_each(|v| *v = 0.0);
        self.factored = false;
    }

    /// Floating-point operations of the factorisation (multiply-adds).
    pub fn factor_flops(&self) -> f64 {
        let mut f = 0.0;
        for i in 0..self.n {
            for j in self.first[i]..=i {
                f += (j - self.first[i].max(self.first[j])) as f64;
            }
        }
        f
    }

    /// Add `v` to entry `(i, j)` (either triangle; stored once).
    #[inline]
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        let (i, j) = if i >= j { (i, j) } else { (j, i) };
        debug_assert!(j >= self.first[i], "outside the profile");
        let idx = self.start[i] + (j - self.first[i]);
        self.val[idx] += v;
    }

    /// `y = A x` (before factorisation; the stored lower triangle mirrored).
    pub fn mul(&self, x: &[f64]) -> Vec<f64> {
        debug_assert!(!self.factored);
        let mut y = vec![0.0; self.n];
        for i in 0..self.n {
            let fi = self.first[i];
            let row = &self.val[self.start[i]..self.start[i] + (i - fi)];
            let mut s = self.at(i, i) * x[i];
            for (k, a) in row.iter().enumerate() {
                s += a * x[fi + k];
                y[fi + k] += a * x[i];
            }
            y[i] += s;
        }
        y
    }

    /// The diagonal (before factorisation).
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.at(i, i)).collect()
    }

    #[inline]
    fn at(&self, i: usize, j: usize) -> f64 {
        self.val[self.start[i] + (j - self.first[i])]
    }

    /// In-place Cholesky `A = L L^T` (row-oriented). Returns false (and leaves the matrix
    /// unusable) if a pivot is not positive.
    pub fn factor(&mut self) -> bool {
        for i in 0..self.n {
            let fi = self.first[i];
            for j in fi..=i {
                let fj = self.first[j];
                let k0 = fi.max(fj);
                let ri = self.start[i] + (k0 - fi);
                let rj = self.start[j] + (k0 - fj);
                let len = j - k0;
                let mut s = 0.0;
                {
                    let a = &self.val[ri..ri + len];
                    let b = &self.val[rj..rj + len];
                    for (x, y) in a.iter().zip(b) {
                        s += x * y;
                    }
                }
                let idx = self.start[i] + (j - fi);
                let v = self.val[idx] - s;
                if j == i {
                    if v <= 0.0 || !v.is_finite() {
                        return false;
                    }
                    self.val[idx] = v.sqrt();
                } else {
                    self.val[idx] = v / self.at(j, j);
                }
            }
        }
        self.factored = true;
        true
    }

    /// Solve `L L^T x = b` in place.
    pub fn solve(&self, b: &mut [f64]) {
        debug_assert!(self.factored);
        let n = self.n;
        for i in 0..n {
            let fi = self.first[i];
            let row = &self.val[self.start[i]..self.start[i] + (i - fi)];
            let mut s = b[i];
            for (k, l) in row.iter().enumerate() {
                s -= l * b[fi + k];
            }
            b[i] = s / self.at(i, i);
        }
        for i in (0..n).rev() {
            let fi = self.first[i];
            b[i] /= self.at(i, i);
            let xi = b[i];
            let row = &self.val[self.start[i]..self.start[i] + (i - fi)];
            for (k, l) in row.iter().enumerate() {
                b[fi + k] -= l * xi;
            }
        }
    }

    /// Multiply-adds of one solve.
    pub fn solve_flops(&self) -> f64 {
        2.0 * self.val.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skyline_solves_a_banded_spd_system() {
        // 1D Laplacian plus identity, permuted by RCM.
        let n = 50;
        let mut adj = vec![Vec::new(); n];
        for i in 0..n - 1 {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        let order = rcm(&adj);
        let mut pos = vec![0; n];
        for (p, &i) in order.iter().enumerate() {
            pos[i] = p;
        }
        let mut pattern = Vec::new();
        for i in 0..n - 1 {
            pattern.push((pos[i], pos[i + 1]));
        }
        let mut a = Skyline::with_pattern(n, pattern.into_iter());
        for i in 0..n {
            a.add(pos[i], pos[i], 3.0);
            if i + 1 < n {
                a.add(pos[i], pos[i + 1], -1.0);
            }
        }
        let x_true: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let mut b = vec![0.0; n];
        for i in 0..n {
            b[pos[i]] = 3.0 * x_true[i]
                - if i > 0 { x_true[i - 1] } else { 0.0 }
                - if i + 1 < n { x_true[i + 1] } else { 0.0 };
        }
        assert!(a.factor());
        a.solve(&mut b);
        for i in 0..n {
            assert!((b[pos[i]] - x_true[i]).abs() < 1e-12);
        }
    }
}
