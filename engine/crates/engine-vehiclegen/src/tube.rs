//! The body's outer skin as one smooth parametric surface S(z, v).
//!
//! Every cross-section (constant z) is a G1 chain of cubic Béziers through
//! key points K0..K(n-1) on the car's left half: K0 on the centre line under
//! the floor, then out and up the side, over the shoulder and the greenhouse,
//! to K(n-1) on the centre line of the roof (or bonnet, or boot lid). Each key
//! point's (x, y) is a monotone spline of z. v in [0, n-1] runs along the
//! left half, v in [n-1, 2(n-1)] mirrors it down the right side.

use std::rc::Rc;

use glam::{DVec2, DVec3};

use crate::curves::Pchip;

pub type Fz = Rc<dyn Fn(f64) -> f64>;
/// A displacement f(P, N, z, v) -> offset added to the surface point.
pub type Displace = Rc<dyn Fn(DVec3, DVec3, f64, f64) -> DVec3>;

/// One key point: x(z) and y(z).
#[derive(Clone)]
pub struct Key {
    pub x: Fz,
    pub y: Fz,
}

impl Key {
    pub fn tables(x: &[(f64, f64)], y: &[(f64, f64)]) -> Key {
        let (px, py) = (Pchip::new(x), Pchip::new(y));
        Key { x: Rc::new(move |z| px.eval(z)), y: Rc::new(move |z| py.eval(z)) }
    }
    pub fn xconst(x: f64, y: &[(f64, f64)]) -> Key {
        let py = Pchip::new(y);
        Key { x: Rc::new(move |_| x), y: Rc::new(move |z| py.eval(z)) }
    }
    pub fn consts(x: f64, y: f64) -> Key {
        Key { x: Rc::new(move |_| x), y: Rc::new(move |_| y) }
    }
}

/// A Bézier handle factor: a constant or a function of z.
#[derive(Clone)]
pub enum Handle {
    C(f64),
    F(Fz),
}

impl Handle {
    #[inline]
    fn at(&self, z: f64) -> f64 {
        match self {
            Handle::C(c) => *c,
            Handle::F(f) => f(z),
        }
    }
}

#[derive(Clone)]
pub struct Tube {
    pub keys: Vec<Key>,
    pub nk: usize,
    pub vm: f64,
    pub h: Vec<(Handle, Handle)>,
    /// key index -> tangent angle of z (radians, direction of increasing v)
    pub tan: Vec<(usize, Fz)>,
    pub disp: Vec<Displace>,
    gt: Option<GirthTable>,
}

#[derive(Clone)]
struct GirthTable {
    z0: f64,
    dz: f64,
    nz: usize,
    dv: f64,
    nv: usize,
    s: Vec<f64>,
}

impl Tube {
    pub fn new(keys: Vec<Key>, handles: Vec<(Handle, Handle)>, disp: Vec<Displace>) -> Tube {
        let nk = keys.len();
        assert_eq!(handles.len(), nk);
        Tube { keys, nk, vm: 2.0 * (nk as f64 - 1.0), h: handles, tan: vec![], disp, gt: None }
    }

    /// Left-frame section point, derivative wrt v, and mirror flag.
    fn section(&self, z: f64, v: f64) -> (DVec2, DVec2, bool) {
        let nk = self.nk;
        let v = v.rem_euclid(self.vm);
        let mir = v > (nk - 1) as f64;
        let vm = if mir { self.vm - v } else { v };
        let k = (vm.floor() as isize).clamp(0, nk as isize - 2) as usize;
        let t = vm - k as f64;
        // key points needed: k-1 .. k+2 (with mirrored ends)
        let kp = |i: isize| -> DVec2 {
            if i < 0 {
                let p = self.key(1, z);
                DVec2::new(-p.x, p.y)
            } else if i as usize >= nk {
                let p = self.key(nk - 2, z);
                DVec2::new(-p.x, p.y)
            } else {
                self.key(i as usize, z)
            }
        };
        let tangent = |i: usize| -> DVec2 {
            for (kk, ang) in &self.tan {
                if *kk == i {
                    let a = ang(z);
                    return DVec2::new(a.cos(), a.sin());
                }
            }
            let d = kp(i as isize + 1) - kp(i as isize - 1);
            d / d.length().max(1e-12)
        };
        let a = kp(k as isize);
        let b = kp(k as isize + 1);
        let ta = tangent(k);
        let tb = tangent(k + 1);
        let l = (b - a).length();
        let hout = self.h[k].1.at(z);
        let hin = self.h[k + 1].0.at(z);
        let p1 = a + ta * hout * l;
        let p2 = b - tb * hin * l;
        // arc-length parametrisation within the segment (17 samples)
        let bez = |s: f64| {
            let u = 1.0 - s;
            a * (u * u * u) + p1 * (3.0 * u * u * s) + p2 * (3.0 * u * s * s) + b * (s * s * s)
        };
        let mut cum = [0.0f64; 17];
        let mut prev = bez(0.0);
        for (j, c) in cum.iter_mut().enumerate().skip(1) {
            let p = bez(j as f64 / 16.0);
            *c = (p - prev).length();
            prev = p;
        }
        for j in 1..17 {
            cum[j] += cum[j - 1];
        }
        let tot = cum[16].max(1e-12);
        for c in &mut cum {
            *c /= tot;
        }
        let mut j = 0usize;
        for c in cum.iter() {
            if *c < t {
                j += 1;
            }
        }
        let j = (j as isize - 1).clamp(0, 15) as usize;
        let (c0, c1) = (cum[j], cum[j + 1]);
        let tt = (j as f64 + ((t - c0) / (c1 - c0).max(1e-12)).clamp(0.0, 1.0)) / 16.0;
        let u = 1.0 - tt;
        let pt = bez(tt);
        let d = (p1 - a) * (3.0 * u * u) + (p2 - p1) * (6.0 * u * tt) + (b - p2) * (3.0 * tt * tt);
        (pt, d, mir)
    }

    #[inline]
    pub fn key(&self, k: usize, z: f64) -> DVec2 {
        let key = &self.keys[k];
        DVec2::new((key.x)(z), (key.y)(z))
    }

    /// Surface point and (undisplaced) outward normal.
    pub fn eval_n(&self, z: f64, v: f64) -> (DVec3, DVec3) {
        let (pt, d, mir) = self.section(z, v);
        let mut n2 = DVec2::new(d.y, -d.x);
        n2 /= n2.length().max(1e-12);
        let x = if mir { -pt.x } else { pt.x };
        let nx = if mir { -n2.x } else { n2.x };
        let mut p = DVec3::new(x, pt.y, z);
        let n = DVec3::new(nx, n2.y, 0.0);
        for f in &self.disp {
            p += f(p, n, z, v);
        }
        (p, n)
    }

    #[inline]
    pub fn eval(&self, z: f64, v: f64) -> DVec3 {
        self.eval_n(z, v).0
    }

    /// Solve y(z, v) = y for v in [v0, v1] (y must be monotone there).
    pub fn v_at_height(&self, z: f64, y: f64, v0: f64, v1: f64) -> f64 {
        let (mut lo, mut hi) = (v0, v1);
        let ylo = self.eval(z, lo).y;
        let inc = self.eval(z, hi).y > ylo;
        for _ in 0..48 {
            let mid = 0.5 * (lo + hi);
            let ym = self.eval(z, mid).y;
            if (ym < y) == inc {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    pub fn v_at_x(&self, z: f64, x: f64, v0: f64, v1: f64) -> f64 {
        let (mut lo, mut hi) = (v0, v1);
        let inc = self.eval(z, hi).x > self.eval(z, lo).x;
        for _ in 0..48 {
            let mid = 0.5 * (lo + hi);
            let xm = self.eval(z, mid).x;
            if (xm < x) == inc {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Cumulative girth length s(z, v) table for the UV chart.
    pub fn girth_table(&mut self, z0: f64, z1: f64) {
        let (dz, dv) = (0.01, 0.01);
        // np.arange(z0 - dz, z1 + 2 dz, dz)
        let zs0 = z0 - dz;
        let nz = (((z1 + 2.0 * dz) - zs0) / dz).ceil() as usize;
        // np.arange(0, VM + dv/2, dv)
        let nv = ((self.vm + dv / 2.0) / dv).ceil() as usize;
        let mut s = vec![0.0; nz * nv];
        for i in 0..nz {
            let z = zs0 + i as f64 * dz;
            let mut prev = self.eval(z, 0.0);
            for j in 1..nv {
                let p = self.eval(z, j as f64 * dv);
                let seg = ((p.x - prev.x).powi(2) + (p.y - prev.y).powi(2)).sqrt();
                s[i * nv + j] = s[i * nv + j - 1] + seg;
                prev = p;
            }
        }
        self.gt = Some(GirthTable { z0: zs0, dz, nz, dv, nv, s });
    }

    pub fn has_girth(&self) -> bool {
        self.gt.is_some()
    }

    pub fn girth(&self, z: f64, v: f64) -> f64 {
        let g = self.gt.as_ref().expect("girth_table first");
        let zmax = g.z0 + (g.nz - 1) as f64 * g.dz;
        let z = z.clamp(g.z0, zmax);
        let v = v.rem_euclid(self.vm + 1e-9);
        let fi = (z - g.z0) / g.dz;
        let fj = v / g.dv;
        let i = (fi.floor() as isize).clamp(0, g.nz as isize - 2) as usize;
        let j = (fj.floor() as isize).clamp(0, g.nv as isize - 2) as usize;
        let (a, b) = (fi - i as f64, fj - j as f64);
        let s = |i: usize, j: usize| g.s[i * g.nv + j];
        (1.0 - a) * (1.0 - b) * s(i, j) + a * (1.0 - b) * s(i + 1, j) + (1.0 - a) * b * s(i, j + 1) + a * b * s(i + 1, j + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_tube() -> Tube {
        // a rounded section: floor, side, roof
        let keys = vec![
            Key::consts(0.0, 0.2),
            Key::consts(0.8, 0.25),
            Key::consts(0.9, 0.6),
            Key::consts(0.7, 1.2),
            Key::consts(0.0, 1.3),
        ];
        let h = vec![(Handle::C(1.0 / 3.0), Handle::C(1.0 / 3.0)); 5];
        Tube::new(keys, h, vec![])
    }

    #[test]
    fn hits_keys_and_mirrors() {
        let t = box_tube();
        for (k, want) in [(0.0, (0.0, 0.2)), (2.0, (0.9, 0.6)), (4.0, (0.0, 1.3))] {
            let p = t.eval(0.5, k);
            assert!((p.x - want.0).abs() < 1e-9 && (p.y - want.1).abs() < 1e-9, "{k} {p}");
        }
        let l = t.eval(0.5, 2.5);
        let r = t.eval(0.5, t.vm - 2.5);
        assert!((l.x + r.x).abs() < 1e-12 && (l.y - r.y).abs() < 1e-12);
        let (_, n) = t.eval_n(0.5, 2.0);
        assert!(n.x > 0.9, "side normal points out: {n}");
    }
}
