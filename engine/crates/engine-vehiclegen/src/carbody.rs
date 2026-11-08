//! The body as a block layout: z-lines (stations) x v-lines (girth lines)
//! on the tube, wheel-arch openings, and end caps (nose, tail panel).
//!
//! Node ids: T(i, j) for z-line i and ring line j; C(cap, r, c) for cap
//! nodes.

use crate::fx::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use std::rc::Rc;

use glam::DVec2;

use crate::layout::{Chart, Curve, FaceFn, Layout, NodeId, TUBE};
use crate::mesh::{Mat, Mesh};
use crate::tube::Tube;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    L,
    R,
}

impl Side {
    pub fn tag(self) -> &'static str {
        match self {
            Side::L => "L",
            Side::R => "R",
        }
    }
    pub fn sign(self) -> f64 {
        match self {
            Side::L => 1.0,
            Side::R => -1.0,
        }
    }
}

/// A cap boundary entry: an existing node or a new cap point.
#[derive(Clone, Copy, Debug)]
pub enum CapPt {
    N(NodeId),
    P(f64, f64),
}

pub type PartFn<'a> = &'a dyn Fn(usize, usize) -> Option<String>;

pub struct CarBody {
    pub l: Layout,
    pub zn: Vec<String>,
    pub zi: FxHashMap<String, usize>,
    pub z0: Vec<f64>,
    pub vl: Vec<f64>,
    pub nl: usize,
    pub ring: Vec<f64>,
    pub jn: usize,
    zover: FxHashMap<(usize, usize), f64>,
    vover: FxHashMap<(usize, usize), f64>,
    pub holes: FxHashSet<(usize, usize)>,
    pub arcs: Vec<(NodeId, NodeId, Curve)>,
    caps: Vec<String>,
    /// the welded raw skins of every part (lamp housings follow them)
    pub skins: BTreeMap<String, Mesh>,
    /// design scratch values passed from caps to parts (grille columns ...)
    pub extra: FxHashMap<String, Vec<f64>>,
}

impl CarBody {
    pub fn new(tube: Tube, zlines: &[(&str, f64)], vlines: &[f64], h: f64) -> CarBody {
        let vm = tube.vm;
        let zn: Vec<String> = zlines.iter().map(|z| z.0.to_string()).collect();
        let zi = zn.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let z0 = zlines.iter().map(|z| z.1).collect();
        let vl = vlines.to_vec();
        let nl = vl.len();
        let mut ring = vl.clone();
        for v in vl[1..nl - 1].iter().rev() {
            ring.push(vm - v);
        }
        let jn = ring.len();
        CarBody {
            l: Layout::new(tube, h),
            zn,
            zi,
            z0,
            vl,
            nl,
            ring,
            jn,
            zover: FxHashMap::default(),
            vover: FxHashMap::default(),
            holes: FxHashSet::default(),
            arcs: vec![],
            caps: vec![],
            skins: BTreeMap::new(),
            extra: FxHashMap::default(),
        }
    }

    pub fn tube(&self) -> &Tube {
        &self.l.tube
    }

    pub fn zi(&self, name: &str) -> usize {
        self.zi[name]
    }

    /// ring index of left-half v-line index jl on a side.
    pub fn j(&self, jl: usize, side: Side) -> usize {
        match side {
            Side::L => jl,
            Side::R => (self.jn - jl) % self.jn,
        }
    }

    /// (left index, side) for ring index j.
    pub fn jl_of(&self, j: usize) -> (usize, Side) {
        if j < self.nl {
            (j, Side::L)
        } else {
            (self.jn - j, Side::R)
        }
    }

    /// Ring line j in [0, J]; J is the floor centre line again (v = VM), a
    /// distinct node at the same place so the wrap edge runs J-1 -> J.
    pub fn nid(&self, i: usize, j: usize) -> NodeId {
        NodeId::T(i as u16, j as u16)
    }

    pub fn set_z(&mut self, i: usize, jl: usize, z: f64) {
        self.zover.insert((i, jl), z);
    }

    pub fn set_zv(&mut self, i: usize, jl: usize, z: f64, v: f64) {
        self.zover.insert((i, jl), z);
        self.vover.insert((i, jl), v);
    }

    pub fn param(&self, i: usize, j: usize) -> (f64, f64) {
        if j == self.jn {
            let (z, v) = self.param(i, 0);
            return (z, self.tube().vm - v);
        }
        let (jl, side) = self.jl_of(j);
        let z = *self.zover.get(&(i, jl)).unwrap_or(&self.z0[i]);
        let mut v = *self.vover.get(&(i, jl)).unwrap_or(&self.vl[jl]);
        if side == Side::R {
            v = self.tube().vm - v;
        }
        (z, v)
    }

    /// Make z-line i lie on the end surface z = fz(x, y).
    pub fn end_fit(&mut self, i: usize, fz: &dyn Fn(f64, f64) -> f64) {
        for jl in 0..self.nl {
            let mut z = *self.zover.get(&(i, jl)).unwrap_or(&self.z0[i]);
            let v = *self.vover.get(&(i, jl)).unwrap_or(&self.vl[jl]);
            for _ in 0..8 {
                let p = self.tube().eval(z, v);
                z = fz(p.x.abs(), p.y);
            }
            self.zover.insert((i, jl), z);
        }
    }

    /// Wheel opening. cols: (front, top, rear) z-line indices; rows: left
    /// v-line indices from the lowest to the arch-top line; thetas: angles
    /// (deg, 0 = forward, 90 = up) of the boundary nodes in order.
    pub fn arch(&mut self, center: (f64, f64), r: f64, cols: (usize, usize, usize), rows: &[usize], thetas: &[f64], vrange: (f64, f64)) {
        let (zc, yc) = center;
        let (i_f, i_t, i_r) = cols;
        let mut nodes: Vec<(usize, usize)> = rows.iter().map(|&rw| (i_f, rw)).collect();
        nodes.push((i_t, *rows.last().unwrap()));
        nodes.extend(rows.iter().rev().map(|&rw| (i_r, rw)));
        assert_eq!(nodes.len(), thetas.len());
        for (&(i, jl), &th) in nodes.iter().zip(thetas) {
            let a = th.to_radians();
            let z = zc + r * a.cos();
            let y = yc + r * a.sin();
            let v = self.tube().v_at_height(z, y, vrange.0, vrange.1);
            self.set_zv(i, jl, z, v);
        }
        for k in 0..nodes.len() - 1 {
            let ((i0, j0), (i1, j1)) = (nodes[k], nodes[k + 1]);
            let (t0, t1) = (thetas[k].to_radians(), thetas[k + 1].to_radians());
            for side in [Side::L, Side::R] {
                let a_ = self.nid(i0, self.j(j0, side));
                let b_ = self.nid(i1, self.j(j1, side));
                let tube = self.tube().clone();
                let vm = tube.vm;
                let curve: Curve = Rc::new(move |t: f64| {
                    let a = t0 + (t1 - t0) * t;
                    let z = zc + r * a.cos();
                    let y = yc + r * a.sin();
                    let mut v = tube.v_at_height(z, y, vrange.0, vrange.1);
                    if side == Side::R {
                        v = vm - v;
                    }
                    DVec2::new(z, v)
                });
                self.arcs.push((a_, b_, curve));
            }
        }
        // hole blocks
        assert!(i_t == i_f + 1 && i_r == i_t + 1);
        for w in rows.windows(2) {
            let (r0, r1) = (w[0], w[1]);
            for i in [i_f, i_t] {
                self.holes.insert((i, r0));
                self.holes.insert((i, (self.jn - r1) % self.jn));
            }
        }
    }

    pub fn build_nodes(&mut self) {
        for i in 0..self.zn.len() {
            for j in 0..=self.jn {
                let (z, v) = self.param(i, j);
                let id = self.nid(i, j);
                self.l.node(id, TUBE, DVec2::new(z, v));
            }
        }
        let arcs = self.arcs.clone();
        for (a, b, c) in arcs {
            self.l.edge(a, b, Some(TUBE), Some(c));
        }
    }

    /// assign(z-line from, z-line to, jl_lo, jl_hi, side) -> (part, mat).
    /// Blocks between ring lines j and j+1; on the right side jl decreases.
    pub fn tube_blocks(&mut self, assign: &dyn Fn(&str, &str, usize, usize, Side) -> Option<(String, Mat)>) {
        for i in 0..self.zn.len() - 1 {
            for j in 0..self.jn {
                let j2 = j + 1;
                if self.holes.contains(&(i, j)) {
                    continue;
                }
                let (jl0, _) = self.jl_of(j);
                let (jl1, _) = self.jl_of(j2 % self.jn);
                let side = if j < self.nl - 1 { Side::L } else { Side::R };
                let (mut lo, mut hi) = if side == Side::L { (jl0, jl1) } else { (jl1, jl0) };
                if j2 == self.jn {
                    lo = 0;
                    hi = 1;
                }
                let Some((part, mat)) = assign(&self.zn[i], &self.zn[i + 1], lo, hi, side) else {
                    continue;
                };
                let c = [self.nid(i, j), self.nid(i + 1, j), self.nid(i + 1, j2), self.nid(i, j2)];
                self.l.block(c, TUBE, &part, mat);
            }
        }
    }

    /// A structured cap. left/right: lists bottom->top; bottom/top: lists
    /// left->right; entries are nodes or new cap points. Corner entries must
    /// agree. holes: blocks left open; overrides: (r, c) -> (x, y) for cap
    /// nodes; arcs: cap-chart edge curves between (r, c) nodes.
    #[allow(clippy::too_many_arguments)]
    pub fn cap(
        &mut self,
        name: &str,
        fz: FaceFn,
        facing: f64,
        left: &[CapPt],
        right: &[CapPt],
        bottom: &[CapPt],
        top: &[CapPt],
        part: PartFn,
        holes: &FxHashSet<(usize, usize)>,
        overrides: &FxHashMap<(usize, usize), DVec2>,
        arcs: Vec<((usize, usize), (usize, usize), Curve)>,
    ) -> (FxHashMap<(usize, usize), NodeId>, Vec<Vec<DVec2>>) {
        let ch = self.l.add_chart(name, Chart::Graph { fz, facing });
        let ci = match self.caps.iter().position(|c| c == name) {
            Some(i) => i,
            None => {
                self.caps.push(name.into());
                self.caps.len() - 1
            }
        };
        let rr = left.len() - 1;
        let cc = bottom.len() - 1;
        assert!(right.len() == rr + 1 && top.len() == cc + 1);
        let mut ids: FxHashMap<(usize, usize), NodeId> = FxHashMap::default();
        let mut xy = vec![vec![DVec2::ZERO; cc + 1]; rr + 1];
        let setb = |r: usize, c: usize, e: CapPt, ids: &mut FxHashMap<(usize, usize), NodeId>, xy: &mut Vec<Vec<DVec2>>| match e {
            CapPt::N(id) => {
                let p = self.l.node_pos(id);
                xy[r][c] = DVec2::new(p.x, p.y);
                ids.insert((r, c), id);
            }
            CapPt::P(x, y) => xy[r][c] = DVec2::new(x, y),
        };
        for r in 0..=rr {
            setb(r, 0, left[r], &mut ids, &mut xy);
            setb(r, cc, right[r], &mut ids, &mut xy);
        }
        for c in 0..=cc {
            setb(0, c, bottom[c], &mut ids, &mut xy);
            setb(rr, c, top[c], &mut ids, &mut xy);
        }
        // interior by TFI over node indices
        for r in 1..rr {
            for c in 1..cc {
                let s = c as f64 / cc as f64;
                let t = r as f64 / rr as f64;
                xy[r][c] = xy[0][c] * (1.0 - t) + xy[rr][c] * t + xy[r][0] * (1.0 - s) + xy[r][cc] * s
                    - (xy[0][0] * ((1.0 - s) * (1.0 - t)) + xy[0][cc] * (s * (1.0 - t)) + xy[rr][0] * ((1.0 - s) * t) + xy[rr][cc] * (s * t));
            }
        }
        for (k, p) in overrides {
            xy[k.0][k.1] = *p;
        }
        for r in 0..=rr {
            for c in 0..=cc {
                if let std::collections::hash_map::Entry::Vacant(e) = ids.entry((r, c)) {
                    let nid = NodeId::C(ci as u16, r as u16, c as u16);
                    self.l.node(nid, ch, xy[r][c]);
                    e.insert(nid);
                }
            }
        }
        for (ka, kb, cur) in arcs {
            self.l.edge(ids[&ka], ids[&kb], Some(ch), Some(cur));
        }
        for r in 0..rr {
            for c in 0..cc {
                if holes.contains(&(r, c)) {
                    continue;
                }
                let Some(pt) = part(r, c) else { continue };
                let corners = [ids[&(r, c)], ids[&(r, c + 1)], ids[&(r + 1, c + 1)], ids[&(r + 1, c)]];
                self.l.block(corners, ch, &pt, Mat::Paint);
            }
        }
        (ids, xy)
    }
}

/// A circular arc in a cap chart, angles in degrees.
pub fn circle_arc(xc: f64, yc: f64, r: f64, a0: f64, a1: f64) -> Curve {
    let (a0, a1) = (a0.to_radians(), a1.to_radians());
    Rc::new(move |t| {
        let a = a0 + (a1 - a0) * t;
        DVec2::new(xc + r * a.cos(), yc + r * a.sin())
    })
}
