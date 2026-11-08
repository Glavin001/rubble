//! UV0 atlas for the painted panels: every island keeps its metric scale
//! (even texel density) and is shelf-packed into [0, 1]^2 without overlap.
//! Non-panel parts get a box projection scaled into [0, 1] per part.

use glam::DVec2;

use crate::mesh::{cube_uv, Mesh};

const PANEL_PREFIX: [&str; 9] = ["body_shell", "hood", "trunk_lid", "door_", "fender_", "bumper_", "tailgate", "bed_", "cab_"];

pub fn is_panel(name: &str) -> bool {
    PANEL_PREFIX.iter().any(|p| name.starts_with(p))
}

/// Named parts in insertion order (the order decides the atlas layout).
#[derive(Clone, Default)]
pub struct Parts(pub Vec<(String, Mesh)>);

impl Parts {
    pub fn get(&self, k: &str) -> Option<&Mesh> {
        self.0.iter().find(|(n, _)| n == k).map(|(_, m)| m)
    }
    pub fn get_mut(&mut self, k: &str) -> Option<&mut Mesh> {
        self.0.iter_mut().find(|(n, _)| n == k).map(|(_, m)| m)
    }
    /// Insert or replace (a replaced part keeps its place, like a dict).
    pub fn insert(&mut self, k: &str, m: Mesh) {
        match self.get_mut(k) {
            Some(x) => *x = m,
            None => self.0.push((k.to_string(), m)),
        }
    }
    /// Add geometry to a part, creating it if needed.
    pub fn extend(&mut self, k: &str, m: &Mesh) {
        match self.get_mut(k) {
            Some(x) => {
                x.add(m);
            }
            None => self.0.push((k.to_string(), m.clone())),
        }
    }
    pub fn names(&self) -> Vec<String> {
        self.0.iter().map(|(n, _)| n.clone()).collect()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

struct Island {
    part: usize,
    qs: Vec<usize>,
    ts: Vec<usize>,
    lo: DVec2,
    size: DVec2,
    off: DVec2,
}

/// Pack the panel islands; returns the atlas size in metres.
pub fn pack(parts: &mut Parts, margin: f64) -> f64 {
    let mut isl: Vec<Island> = Vec::new();
    for (pi, (name, m)) in parts.0.iter_mut().enumerate() {
        if !is_panel(name) {
            continue;
        }
        // anything without an island gets a box projection
        let noq: Vec<bool> = m.qi.iter().map(|&i| i < 0).collect();
        let nt: Vec<bool> = m.ti.iter().map(|&i| i < 0).collect();
        if noq.iter().any(|&b| b) || nt.iter().any(|&b| b) {
            let mut tmp = m.clone();
            cube_uv(&mut tmp, 50000, 1.0);
            for (i, &b) in noq.iter().enumerate() {
                if b {
                    m.quv[i] = tmp.quv[i];
                    m.qi[i] = tmp.qi[i];
                }
            }
            for (i, &b) in nt.iter().enumerate() {
                if b {
                    m.tuv[i] = tmp.tuv[i];
                    m.ti[i] = tmp.ti[i];
                }
            }
        }
        let mut ids: Vec<i32> = m.qi.iter().chain(m.ti.iter()).cloned().collect();
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            let qs: Vec<usize> = (0..m.qi.len()).filter(|&k| m.qi[k] == id).collect();
            let ts: Vec<usize> = (0..m.ti.len()).filter(|&k| m.ti[k] == id).collect();
            let mut lo = DVec2::splat(f64::INFINITY);
            let mut hi = DVec2::splat(f64::NEG_INFINITY);
            for &k in &qs {
                for u in m.quv[k] {
                    lo = lo.min(u);
                    hi = hi.max(u);
                }
            }
            for &k in &ts {
                for u in m.tuv[k] {
                    lo = lo.min(u);
                    hi = hi.max(u);
                }
            }
            isl.push(Island { part: pi, qs, ts, lo, size: hi - lo, off: DVec2::ZERO });
        }
    }
    if isl.is_empty() {
        return 1.0;
    }
    let area: f64 = isl.iter().map(|i| (i.size.x + margin) * (i.size.y + margin)).sum();
    let widest = isl.iter().map(|i| i.size.x).fold(f64::NEG_INFINITY, f64::max);
    let w = (area * 1.25).sqrt().max(widest + 2.0 * margin);
    // stable sort by height, tallest first
    isl.sort_by(|a, b| b.size.y.partial_cmp(&a.size.y).unwrap());
    let (mut x, mut y) = (margin, margin);
    let mut row_h: f64 = 0.0;
    for i in &mut isl {
        let (iw, ih) = (i.size.x, i.size.y);
        if x + iw + margin > w {
            x = margin;
            y += row_h + margin;
            row_h = 0.0;
        }
        i.off = DVec2::new(x, y);
        x += iw + margin;
        row_h = row_h.max(ih);
    }
    let h = y + row_h + margin;
    let s = w.max(h);
    for i in &isl {
        let m = &mut parts.0[i.part].1;
        let sh = i.off - i.lo;
        for &k in &i.qs {
            for u in &mut m.quv[k] {
                *u = (*u + sh) / s;
            }
        }
        for &k in &i.ts {
            for u in &mut m.tuv[k] {
                *u = (*u + sh) / s;
            }
        }
    }
    s
}

/// Non-panel parts: box projection scaled into [0, 1] per part.
pub fn normalise_other(parts: &mut Parts) {
    for (name, m) in parts.0.iter_mut() {
        if is_panel(name) {
            continue;
        }
        cube_uv(m, 0, 1.0);
        let mut lo = DVec2::splat(f64::INFINITY);
        let mut hi = DVec2::splat(f64::NEG_INFINITY);
        for u in m.quv.iter().flatten().chain(m.tuv.iter().flatten()) {
            lo = lo.min(*u);
            hi = hi.max(*u);
        }
        if !lo.is_finite() {
            continue;
        }
        let s = (hi - lo).max_element().max(1e-6);
        for u in m.quv.iter_mut().flatten().chain(m.tuv.iter_mut().flatten()) {
            *u = (*u - lo) / s;
        }
    }
}
