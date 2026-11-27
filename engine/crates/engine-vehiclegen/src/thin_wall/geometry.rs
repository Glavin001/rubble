//! Exact plane-region integrals of wall pieces (polygons): area, first and second
//! moments by Green's theorem, and the clip by a half-plane the plastic moduli need.

/// A point in the section's plane (m): x across, y up.
pub type P2 = [f64; 2];

/// ∫dA, ∫x dA, ∫y dA, ∫x² dA, ∫y² dA, ∫xy dA of a simple polygon (either winding; the
/// result is that of the region, positive area).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moments {
    pub a: f64,
    pub sx: f64,
    pub sy: f64,
    pub ixx_o: f64,
    pub iyy_o: f64,
    pub ixy_o: f64,
}

impl Moments {
    pub fn of(poly: &[P2]) -> Self {
        let n = poly.len();
        if n < 3 {
            return Self::default();
        }
        let mut m = Self::default();
        for i in 0..n {
            let [x0, y0] = poly[i];
            let [x1, y1] = poly[(i + 1) % n];
            let c = x0 * y1 - x1 * y0;
            m.a += c;
            m.sx += (x0 + x1) * c;
            m.sy += (y0 + y1) * c;
            // ∫y² dA and ∫x² dA (y² terms give the moment about the x axis)
            m.ixx_o += (y0 * y0 + y0 * y1 + y1 * y1) * c;
            m.iyy_o += (x0 * x0 + x0 * x1 + x1 * x1) * c;
            m.ixy_o += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * c;
        }
        let s = if m.a < 0.0 { -1.0 } else { 1.0 };
        Self {
            a: s * m.a / 2.0,
            // ∫x dA = Σ(x0 + x1)c / 6, ∫y dA likewise
            sx: s * m.sx / 6.0,
            sy: s * m.sy / 6.0,
            ixx_o: s * m.ixx_o / 12.0,
            iyy_o: s * m.iyy_o / 12.0,
            ixy_o: s * m.ixy_o / 24.0,
        }
    }

    pub fn add(self, o: Self) -> Self {
        Self { a: self.a + o.a, sx: self.sx + o.sx, sy: self.sy + o.sy, ixx_o: self.ixx_o + o.ixx_o, iyy_o: self.iyy_o + o.iyy_o, ixy_o: self.ixy_o + o.ixy_o }
    }
}

/// The part of `poly` where `n · p ≥ s` (Sutherland–Hodgman against one half-plane: exact
/// for any simple polygon's area and moments, degenerate edges included).
pub fn clip(poly: &[P2], n: P2, s: f64) -> Vec<P2> {
    let d = |p: P2| n[0] * p[0] + n[1] * p[1] - s;
    let mut out = Vec::with_capacity(poly.len() + 2);
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let (dp, dq) = (d(p), d(q));
        if dp >= 0.0 {
            out.push(p);
        }
        if (dp >= 0.0) != (dq >= 0.0) {
            let t = dp / (dp - dq);
            out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
        }
    }
    out
}

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn unit(a: P2) -> P2 {
    let l = (a[0] * a[0] + a[1] * a[1]).sqrt();
    [a[0] / l, a[1] / l]
}

/// The quadrilateral pieces of a wall of thickness `t` on the midline `pts` (`closed`:
/// the last point joins the first). Neighbouring pieces meet on their mitre line, so a
/// corner is solid and counted once; an open end is cut square. The pieces of a closed
/// rectangular midline (b − t) × (h − t) are exactly a b × h tube of wall t.
///
/// Errors: fewer than two points, a zero-length segment, a reversal (a turn of 180°), or a
/// turn so sharp that its mitre is longer than 10 wall thicknesses (the pieces would
/// overlap their neighbours').
pub fn wall_pieces(pts: &[P2], t: f64, closed: bool) -> Result<Vec<[P2; 4]>, String> {
    let n = pts.len();
    if n < 2 || (closed && n < 3) {
        return Err("a wall needs two points (three when closed)".into());
    }
    if !(t > 0.0) {
        return Err(format!("wall thickness {t} must be positive"));
    }
    let segs = if closed { n } else { n - 1 };
    let mut dir = Vec::with_capacity(segs);
    for i in 0..segs {
        let d = sub(pts[(i + 1) % n], pts[i]);
        if d[0].hypot(d[1]) <= 1e-12 {
            return Err(format!("wall segment {i} has zero length"));
        }
        dir.push(unit(d));
    }
    let normal = |d: P2| [-d[1], d[0]];
    // The offset of the midline at vertex v (towards the +normal side by t/2): a mitre
    // between the segments that meet there, square at an open end.
    let offset = |v: usize| -> Result<P2, String> {
        let (before, after) = if closed {
            (Some(dir[(v + segs - 1) % segs]), Some(dir[v % segs]))
        } else {
            (v.checked_sub(1).map(|i| dir[i]), (v < segs).then(|| dir[v]))
        };
        Ok(match (before, after) {
            (Some(a), Some(b)) => {
                let (na, nb) = (normal(a), normal(b));
                let m = [na[0] + nb[0], na[1] + nb[1]];
                let ml = m[0].hypot(m[1]);
                if ml < 1e-9 {
                    return Err(format!("wall turns back on itself at point {v}"));
                }
                let mu = [m[0] / ml, m[1] / ml];
                let cos_half = mu[0] * nb[0] + mu[1] * nb[1];
                let len = 0.5 * t / cos_half;
                if len > 5.0 * t {
                    return Err(format!("wall turns too sharply at point {v} (mitre {len:.4} m)"));
                }
                [mu[0] * len, mu[1] * len]
            }
            (Some(a), None) | (None, Some(a)) => {
                let na = normal(a);
                [na[0] * 0.5 * t, na[1] * 0.5 * t]
            }
            (None, None) => unreachable!(),
        })
    };
    let mut out = Vec::with_capacity(segs);
    for i in 0..segs {
        let (v0, v1) = (i, (i + 1) % n);
        let (o0, o1) = (offset(v0)?, offset(v1)?);
        let (p0, p1) = (pts[v0], pts[v1]);
        out.push([
            [p0[0] - o0[0], p0[1] - o0[1]],
            [p1[0] - o1[0], p1[1] - o1[1]],
            [p1[0] + o1[0], p1[1] + o1[1]],
            [p0[0] + o0[0], p0[1] + o0[1]],
        ]);
    }
    Ok(out)
}
