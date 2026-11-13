//! Editable 1-D curves: control points `(x, y)` joined by a monotone cubic
//! (Fritsch–Carlson), so a profile drawn through a handful of points is smooth and
//! never overshoots between them (a roof never bulges above its highest point).

use serde::{Deserialize, Serialize};

/// A smooth curve through control points `(x, y)`, `x` increasing. Constant
/// beyond the first and last point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Curve(pub Vec<[f32; 2]>);

impl Curve {
    pub fn new(points: &[(f32, f32)]) -> Self {
        let mut v: Vec<[f32; 2]> = points.iter().map(|&(x, y)| [x, y]).collect();
        v.sort_by(|a, b| a[0].total_cmp(&b[0]));
        Self(v)
    }

    /// A constant.
    pub fn flat(y: f32) -> Self {
        Self(vec![[0.0, y]])
    }

    /// Every control point's y scaled by `s` (a quick proportional tweak).
    pub fn scaled(&self, s: f32) -> Self {
        Self(self.0.iter().map(|p| [p[0], p[1] * s]).collect())
    }

    /// Every control point's x scaled by `s` (stretch along the car).
    pub fn stretched(&self, s: f32) -> Self {
        Self(self.0.iter().map(|p| [p[0] * s, p[1]]).collect())
    }

    /// The control points with `dy` added to every y.
    pub fn offset(&self, dy: f32) -> Self {
        Self(self.0.iter().map(|p| [p[0], p[1] + dy]).collect())
    }

    /// Slopes at the control points (Fritsch–Carlson monotone tangents).
    fn slopes(&self) -> Vec<f32> {
        let p = &self.0;
        let n = p.len();
        if n < 2 {
            return vec![0.0; n];
        }
        let d: Vec<f32> = p
            .windows(2)
            .map(|w| (w[1][1] - w[0][1]) / (w[1][0] - w[0][0]).max(1e-6))
            .collect();
        let mut m = vec![0.0f32; n];
        m[0] = d[0];
        m[n - 1] = d[n - 2];
        for i in 1..n - 1 {
            m[i] = if d[i - 1] * d[i] <= 0.0 {
                0.0
            } else {
                0.5 * (d[i - 1] + d[i])
            };
        }
        for i in 0..n - 1 {
            if d[i] == 0.0 {
                m[i] = 0.0;
                m[i + 1] = 0.0;
                continue;
            }
            let a = m[i] / d[i];
            let b = m[i + 1] / d[i];
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[i] = t * a * d[i];
                m[i + 1] = t * b * d[i];
            }
        }
        m
    }

    pub fn eval(&self, x: f32) -> f32 {
        let p = &self.0;
        match p.len() {
            0 => return 0.0,
            1 => return p[0][1],
            _ => {}
        }
        if x <= p[0][0] {
            return p[0][1];
        }
        if x >= p[p.len() - 1][0] {
            return p[p.len() - 1][1];
        }
        let m = self.slopes();
        let i = p
            .windows(2)
            .position(|w| x <= w[1][0])
            .unwrap_or(p.len() - 2);
        let (x0, y0, x1, y1) = (p[i][0], p[i][1], p[i + 1][0], p[i + 1][1]);
        let h = x1 - x0;
        let t = (x - x0) / h;
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * y0
            + (t3 - 2.0 * t2 + t) * h * m[i]
            + (-2.0 * t3 + 3.0 * t2) * y1
            + (t3 - t2) * h * m[i + 1]
    }
}

/// A closed polygon in 2-D (a panel or window outline in side or front view).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outline(pub Vec<[f32; 2]>);

impl Outline {
    pub fn new(points: &[(f32, f32)]) -> Self {
        Self(points.iter().map(|&(x, y)| [x, y]).collect())
    }

    /// An axis-aligned rectangle.
    pub fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self::new(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        let p = &self.0;
        let mut inside = false;
        let mut j = p.len().wrapping_sub(1);
        for i in 0..p.len() {
            let (xi, yi, xj, yj) = (p[i][0], p[i][1], p[j][0], p[j][1]);
            if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Distance from (x, y) to the outline's edges.
    pub fn edge_distance(&self, x: f32, y: f32) -> f32 {
        let p = &self.0;
        let mut best = f32::MAX;
        for i in 0..p.len() {
            let a = p[i];
            let b = p[(i + 1) % p.len()];
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let l2 = dx * dx + dy * dy;
            let t = if l2 > 0.0 {
                (((x - a[0]) * dx + (y - a[1]) * dy) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (ex, ey) = (a[0] + t * dx - x, a[1] + t * dy - y);
            best = best.min((ex * ex + ey * ey).sqrt());
        }
        best
    }

    /// The nearest point on the outline's edges to (x, y), and its distance.
    pub fn nearest(&self, x: f32, y: f32) -> ([f32; 2], f32) {
        let p = &self.0;
        let mut best = ([x, y], f32::MAX);
        for i in 0..p.len() {
            let a = p[i];
            let b = p[(i + 1) % p.len()];
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let l2 = dx * dx + dy * dy;
            let t = if l2 > 0.0 {
                (((x - a[0]) * dx + (y - a[1]) * dy) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = [a[0] + t * dx, a[1] + t * dy];
            let d = ((q[0] - x).powi(2) + (q[1] - y).powi(2)).sqrt();
            if d < best.1 {
                best = (q, d);
            }
        }
        best
    }

    /// Signed distance: negative inside.
    pub fn signed_distance(&self, x: f32, y: f32) -> f32 {
        let d = self.edge_distance(x, y);
        if self.contains(x, y) {
            -d
        } else {
            d
        }
    }

    pub fn shifted(&self, dx: f32, dy: f32) -> Self {
        Self(self.0.iter().map(|p| [p[0] + dx, p[1] + dy]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_passes_through_points_without_overshoot() {
        let c = Curve::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 2.0), (3.0, 0.5)]);
        for &(x, y) in &[(0.0, 1.0), (1.0, 2.0), (2.0, 2.0), (3.0, 0.5)] {
            assert!((c.eval(x) - y).abs() < 1e-5);
        }
        for i in 0..=300 {
            let y = c.eval(i as f32 / 100.0);
            assert!(y <= 2.0 + 1e-5 && y >= 0.5 - 1e-5, "{y}");
        }
    }

    #[test]
    fn outline_contains_and_distance() {
        let o = Outline::rect(0.0, 0.0, 2.0, 1.0);
        assert!(o.contains(1.0, 0.5));
        assert!(!o.contains(2.5, 0.5));
        assert!((o.signed_distance(1.0, 0.5) + 0.5).abs() < 1e-6);
    }
}
