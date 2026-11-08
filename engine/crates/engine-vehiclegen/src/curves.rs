//! 1D interpolation used by every design table: monotone cubic (PCHIP) so a
//! profile never overshoots between stations, plus small smoothing helpers.

use glam::DVec3;

/// Fritsch–Carlson monotone cubic through (x, y); clamps outside the range.
#[derive(Clone, Debug)]
pub struct Pchip {
    x: Vec<f64>,
    y: Vec<f64>,
    m: Vec<f64>,
}

/// numpy.sign: -1, 0 or 1.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn end_slope(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if sign(m) != sign(d0) {
        return 0.0;
    }
    if sign(d0) != sign(d1) && m.abs() > (3.0 * d0).abs() {
        return 3.0 * d0;
    }
    m
}

impl Pchip {
    pub fn new(pairs: &[(f64, f64)]) -> Pchip {
        let mut p: Vec<(f64, f64)> = pairs.to_vec();
        p.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (mut x, mut y): (Vec<f64>, Vec<f64>) = p.into_iter().unzip();
        if x.len() == 1 {
            x = vec![x[0] - 1.0, x[0] + 1.0];
            y = vec![y[0], y[0]];
        }
        let n = x.len();
        let h: Vec<f64> = (0..n - 1).map(|k| x[k + 1] - x[k]).collect();
        let d: Vec<f64> = (0..n - 1).map(|k| (y[k + 1] - y[k]) / h[k]).collect();
        let mut m = vec![0.0; n];
        if n == 2 {
            m[0] = d[0];
            m[1] = d[0];
        } else {
            for k in 1..n - 1 {
                if d[k - 1] * d[k] <= 0.0 {
                    m[k] = 0.0;
                } else {
                    let w1 = 2.0 * h[k] + h[k - 1];
                    let w2 = h[k] + 2.0 * h[k - 1];
                    m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k]);
                }
            }
            m[0] = end_slope(h[0], h[1], d[0], d[1]);
            m[n - 1] = end_slope(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
        }
        Pchip { x, y, m }
    }

    pub fn constant(v: f64) -> Pchip {
        Pchip::new(&[(-10.0, v), (10.0, v)])
    }

    pub fn eval(&self, q: f64) -> f64 {
        let (x, y, m) = (&self.x, &self.y, &self.m);
        let n = x.len();
        let qc = q.clamp(x[0], x[n - 1]);
        // searchsorted (left) - 1, clipped to [0, n-2]
        let ss = x.partition_point(|&xi| xi < qc);
        let k = (ss as isize - 1).clamp(0, n as isize - 2) as usize;
        let h = x[k + 1] - x[k];
        let t = (qc - x[k]) / h;
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * y[k]
            + (t3 - 2.0 * t2 + t) * h * m[k]
            + (-2.0 * t3 + 3.0 * t2) * y[k + 1]
            + (t3 - t2) * h * m[k + 1]
    }

    /// The table's stations (for remapping a design's tables).
    pub fn points(&self) -> Vec<(f64, f64)> {
        self.x.iter().cloned().zip(self.y.iter().cloned()).collect()
    }
}

pub fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smooth compact bump: 1 at c, 0 beyond |x-c| >= w.
pub fn bump(x: f64, c: f64, w: f64) -> f64 {
    let t = (1.0 - ((x - c) / w).powi(2)).clamp(0.0, 1.0);
    t * t
}

/// numpy.interp: piecewise-linear, clamped at the ends.
pub fn interp(q: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if q <= xs[0] {
        return ys[0];
    }
    if q >= xs[n - 1] {
        return ys[n - 1];
    }
    let k = xs.partition_point(|&x| x <= q).saturating_sub(1).min(n - 2);
    let (x0, x1) = (xs[k], xs[k + 1]);
    if x1 == x0 {
        return ys[k + 1];
    }
    ys[k] + (ys[k + 1] - ys[k]) * (q - x0) / (x1 - x0)
}

/// numpy.linspace(a, b, n) (endpoint included).
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

/// n+1 points evenly spaced by arc length along a 3D polyline.
pub fn resample_polyline3(p: &[DVec3], n: usize) -> Vec<DVec3> {
    let mut s = vec![0.0];
    for w in p.windows(2) {
        s.push(s.last().unwrap() + (w[1] - w[0]).length());
    }
    let total = *s.last().unwrap();
    linspace(0.0, total, n + 1)
        .into_iter()
        .map(|t| {
            let xs: Vec<f64> = p.iter().map(|q| q.x).collect();
            let ys: Vec<f64> = p.iter().map(|q| q.y).collect();
            let zs: Vec<f64> = p.iter().map(|q| q.z).collect();
            DVec3::new(interp(t, &s, &xs), interp(t, &s, &ys), interp(t, &s, &zs))
        })
        .collect()
}

/// n+1 points evenly spaced by arc length along a 2D polyline.
pub fn resample_polyline2(p: &[glam::DVec2], n: usize) -> Vec<glam::DVec2> {
    let mut s = vec![0.0];
    for w in p.windows(2) {
        s.push(s.last().unwrap() + (w[1] - w[0]).length());
    }
    let total = *s.last().unwrap();
    let xs: Vec<f64> = p.iter().map(|q| q.x).collect();
    let ys: Vec<f64> = p.iter().map(|q| q.y).collect();
    linspace(0.0, total, n + 1)
        .into_iter()
        .map(|t| glam::DVec2::new(interp(t, &s, &xs), interp(t, &s, &ys)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pchip_is_monotone_and_hits_stations() {
        let p = Pchip::new(&[(0.0, 0.0), (1.0, 1.0), (2.0, 1.2), (3.0, 3.0)]);
        for &(x, y) in &[(0.0, 0.0), (1.0, 1.0), (2.0, 1.2), (3.0, 3.0)] {
            assert!((p.eval(x) - y).abs() < 1e-12);
        }
        let mut last = -1.0;
        for i in 0..=300 {
            let v = p.eval(i as f64 / 100.0);
            assert!(v >= last - 1e-12);
            last = v;
        }
        assert_eq!(p.eval(-5.0), 0.0);
        assert_eq!(p.eval(9.0), 3.0);
    }
}
