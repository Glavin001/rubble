//! Proportions: map a design's reference coordinates onto the requested
//! wheelbase, overhangs, width, ride height and roof height.
//!
//! - z (along the car) is piecewise linear with slope 1 through a zone round
//!   each axle, so wheel arches stay round; the cabin and the overhangs
//!   stretch.
//! - x scales uniformly with the body width.
//! - y shifts with the ride height and the wheel centre (so an arch stays
//!   centred on its wheel) and scales smoothly above the shoulder (the roof).
//!
//! With the reference proportions every map is exactly the identity, so a
//! default build matches the Python generator.

use glam::DVec3;

#[derive(Clone, Debug)]
pub struct Frame {
    ident: bool,
    /// knots: reference z -> new z (increasing)
    zr: Vec<f64>,
    zn: Vec<f64>,
    pub sx: f64,
    /// vertical shift (ride height + wheel centre change)
    pub dy: f64,
    /// roof scale above y0 (C1 blend over [y0, y0 + w])
    pub kr: f64,
    pub y0: f64,
    pub w: f64,
}

/// The reference car's proportions a design was drawn at.
#[derive(Clone, Copy, Debug)]
pub struct RefDims {
    pub axle_f: f64,
    pub axle_r: f64,
    pub tip: f64,
    pub tail: f64,
    /// half-length of the undistorted zone round each axle
    pub arch_zone: f64,
}

/// Target proportions.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub wheelbase: f64,
    pub front_overhang: f64,
    pub rear_overhang: f64,
    pub sx: f64,
    pub dy: f64,
    pub kr: f64,
    pub y0: f64,
}

fn pl(q: f64, xs: &[f64], ys: &[f64]) -> f64 {
    // piecewise linear with linear extrapolation past both ends
    let n = xs.len();
    let k = if q <= xs[0] {
        0
    } else if q >= xs[n - 1] {
        n - 2
    } else {
        xs.partition_point(|&x| x <= q).saturating_sub(1).min(n - 2)
    };
    ys[k] + (ys[k + 1] - ys[k]) * (q - xs[k]) / (xs[k + 1] - xs[k])
}

impl Frame {
    pub fn identity() -> Frame {
        Frame { ident: true, zr: vec![0.0, 1.0], zn: vec![0.0, 1.0], sx: 1.0, dy: 0.0, kr: 1.0, y0: 0.95, w: 0.10 }
    }

    pub fn new(r: RefDims, t: Target) -> Frame {
        let a = r.arch_zone;
        let (hf, hr) = (t.wheelbase / 2.0, t.wheelbase / 2.0);
        let zr = vec![r.tail, r.axle_r - a, r.axle_r + a, r.axle_f - a, r.axle_f + a, r.tip];
        let zn = vec![-hr - t.rear_overhang, -hr - a, -hr + a, hf - a, hf + a, hf + t.front_overhang];
        let ident = zr.iter().zip(&zn).all(|(a, b)| (a - b).abs() < 1e-12) && t.sx == 1.0 && t.dy == 0.0 && t.kr == 1.0;
        for w in zn.windows(2) {
            assert!(w[1] > w[0] + 1e-3, "proportions out of range: z knots {zn:?}");
        }
        Frame { ident, zr, zn, sx: t.sx, dy: t.dy, kr: t.kr, y0: t.y0, w: 0.10 }
    }

    pub fn is_identity(&self) -> bool {
        self.ident
    }

    #[inline]
    pub fn z(&self, z: f64) -> f64 {
        if self.ident {
            z
        } else {
            pl(z, &self.zr, &self.zn)
        }
    }

    #[inline]
    pub fn zi(&self, z: f64) -> f64 {
        if self.ident {
            z
        } else {
            pl(z, &self.zn, &self.zr)
        }
    }

    #[inline]
    pub fn x(&self, x: f64) -> f64 {
        if self.ident {
            x
        } else {
            x * self.sx
        }
    }

    #[inline]
    pub fn xi(&self, x: f64) -> f64 {
        if self.ident {
            x
        } else {
            x / self.sx
        }
    }

    /// the roof blend: 0 below y0, C1 ramp to slope 1 above y0 + w
    fn g(&self, y: f64) -> f64 {
        let (y0, w) = (self.y0, self.w);
        if y <= y0 {
            0.0
        } else if y < y0 + w {
            (y - y0) * (y - y0) / (2.0 * w)
        } else {
            y - y0 - w / 2.0
        }
    }

    #[inline]
    pub fn y(&self, y: f64) -> f64 {
        if self.ident {
            y
        } else {
            y + (self.kr - 1.0) * self.g(y) + self.dy
        }
    }

    pub fn yi(&self, y: f64) -> f64 {
        if self.ident {
            return y;
        }
        // monotone: bisection
        let (mut lo, mut hi) = (y - self.dy - 2.0, y - self.dy + 2.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.y(mid) < y {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// A reference point in the new proportions.
    #[inline]
    pub fn p(&self, x: f64, y: f64, z: f64) -> DVec3 {
        DVec3::new(self.x(x), self.y(y), self.z(z))
    }

    #[inline]
    pub fn pv(&self, v: DVec3) -> DVec3 {
        self.p(v.x, v.y, v.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_invert_and_keep_axle_zones() {
        let r = RefDims { axle_f: 1.415, axle_r: -1.415, tip: 2.385, tail: -2.495, arch_zone: 0.55 };
        let f = Frame::new(r, Target { wheelbase: 2.6, front_overhang: 0.85, rear_overhang: 0.9, sx: 0.95, dy: 0.03, kr: 0.9, y0: 0.95 });
        for z in [-2.6, -1.9, -1.2, 0.0, 1.0, 1.5, 2.4] {
            assert!((f.zi(f.z(z)) - z).abs() < 1e-12);
        }
        for y in [0.2, 0.9, 0.97, 1.3] {
            assert!((f.yi(f.y(y)) - y).abs() < 1e-9);
        }
        // slope 1 round each axle: the arch keeps its shape
        assert!((f.z(1.415 + 0.3) - f.z(1.415) - 0.3).abs() < 1e-12);
        assert!((f.z(1.415) - 1.3).abs() < 1e-12);
        assert!((f.z(2.385) - (1.3 + 0.85)).abs() < 1e-12);
        let id = Frame::new(r, Target { wheelbase: 2.83, front_overhang: 0.97, rear_overhang: 1.08, sx: 1.0, dy: 0.0, kr: 1.0, y0: 0.95 });
        assert!(id.is_identity());
    }
}
