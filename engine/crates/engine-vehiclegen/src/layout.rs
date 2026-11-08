//! Block-structured quad meshing over charts.
//!
//! A layout is a set of corner nodes, edges between them and four-sided
//! blocks. Each block lives in a chart (a 2D parameter space mapped to 3D).
//! Opposite edges of a block share a chord, so they get the same number of
//! segments; counts are solved over the whole layout (union-find), which makes
//! every shared edge sample identically and lets neighbouring blocks weld
//! exactly. Edges are sampled evenly by 3D arc length; block interiors come
//! from transfinite (Coons) interpolation in the block's chart.

use std::collections::HashMap;
use std::rc::Rc;

use glam::{DVec2, DVec3};

use crate::curves::interp;
use crate::mesh::{grid, Grid2, Mat, Mesh};
use crate::tube::Tube;

/// z = f(x, y): the front and rear end faces.
pub type FaceFn = Rc<dyn Fn(f64, f64) -> f64>;
/// An edge curve: t in [0, 1] -> chart parameters.
pub type Curve = Rc<dyn Fn(f64) -> DVec2>;

#[derive(Clone)]
pub enum Chart {
    /// (z, v) on the body tube
    Tube,
    /// (x, y) on z = fz(x, y); facing +1 front, -1 rear
    Graph { fz: FaceFn, facing: f64 },
}

/// ('t', i, j): z-line i, ring line j. ('c', cap, r, c): cap node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NodeId {
    T(u16, u16),
    C(u16, u16, u16),
}

type EKey = (NodeId, NodeId);

#[derive(Clone)]
struct EdgeDef {
    chart: Option<usize>,
    curve: Option<Curve>,
    min: usize,
}

#[derive(Clone)]
pub struct Block {
    pub c: [NodeId; 4],
    pub chart: usize,
    pub part: String,
    pub mat: Mat,
}

pub struct Layout {
    pub h: f64,
    pub h_max: f64,
    pub tube: Tube,
    pub charts: Vec<(String, Chart)>,
    pub nodes: HashMap<NodeId, (usize, DVec2)>,
    edges: HashMap<EKey, EdgeDef>,
    pub blocks: Vec<Block>,
    counts: HashMap<EKey, usize>,
    cache: HashMap<(EKey, usize), (usize, Vec<DVec2>, Vec<DVec3>)>,
}

pub const TUBE: usize = 0;

impl Layout {
    pub fn new(tube: Tube, h: f64) -> Layout {
        Layout {
            h,
            h_max: 0.029,
            tube,
            charts: vec![("tube".into(), Chart::Tube)],
            nodes: HashMap::new(),
            edges: HashMap::new(),
            blocks: vec![],
            counts: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    pub fn chart_id(&self, name: &str) -> Option<usize> {
        self.charts.iter().position(|(n, _)| n == name)
    }

    pub fn add_chart(&mut self, name: &str, ch: Chart) -> usize {
        if let Some(i) = self.chart_id(name) {
            return i;
        }
        self.charts.push((name.into(), ch));
        self.charts.len() - 1
    }

    // -- chart maps -------------------------------------------------------------
    pub fn map(&self, ch: usize, p: DVec2) -> DVec3 {
        match &self.charts[ch].1 {
            Chart::Tube => self.tube.eval(p.x, p.y),
            Chart::Graph { fz, .. } => DVec3::new(p.x, p.y, fz(p.x, p.y)),
        }
    }

    fn inv(&self, ch: usize, p: DVec3) -> DVec2 {
        match &self.charts[ch].1 {
            Chart::Tube => panic!("tube chart has no inverse; define the edge in the tube chart"),
            Chart::Graph { .. } => DVec2::new(p.x, p.y),
        }
    }

    fn normal(&self, ch: usize, p: DVec2) -> DVec3 {
        match &self.charts[ch].1 {
            Chart::Tube => self.tube.eval_n(p.x, p.y).1,
            Chart::Graph { facing, .. } => DVec3::new(0.0, 0.0, *facing),
        }
    }

    fn uv(&self, ch: usize, p: DVec2) -> DVec2 {
        match &self.charts[ch].1 {
            Chart::Tube => DVec2::new(p.x, self.tube.girth(p.x, p.y)),
            Chart::Graph { facing, .. } => DVec2::new(p.x * facing, p.y),
        }
    }

    // -- nodes and edges ----------------------------------------------------------
    pub fn node(&mut self, id: NodeId, chart: usize, p: DVec2) {
        self.nodes.insert(id, (chart, p));
    }

    pub fn node_pos(&self, id: NodeId) -> DVec3 {
        let (ch, p) = self.nodes[&id];
        self.map(ch, p)
    }

    fn node_param(&self, id: NodeId, chart: usize) -> DVec2 {
        let (ch, p) = self.nodes[&id];
        if ch == chart {
            return p;
        }
        self.inv(chart, self.map(ch, p))
    }

    /// curve(t) -> params in `chart`, from node a to node b.
    pub fn edge(&mut self, a: NodeId, b: NodeId, chart: Option<usize>, curve: Option<Curve>) {
        self.edges.remove(&(a, b));
        self.edges.remove(&(b, a));
        self.edges.insert((a, b), EdgeDef { chart, curve, min: 1 });
    }

    /// corners (a, b, c, d): bottom a->b, right b->c, top d->c, left a->d.
    pub fn block(&mut self, c: [NodeId; 4], chart: usize, part: &str, mat: Mat) {
        self.blocks.push(Block { c, chart, part: part.into(), mat });
    }

    fn edef(&mut self, a: NodeId, b: NodeId) -> (EKey, bool) {
        if self.edges.contains_key(&(a, b)) {
            return ((a, b), false);
        }
        if self.edges.contains_key(&(b, a)) {
            return ((b, a), true);
        }
        self.edges.insert((a, b), EdgeDef { chart: None, curve: None, min: 1 });
        ((a, b), false)
    }

    fn ek(&self, a: NodeId, b: NodeId) -> EKey {
        if self.edges.contains_key(&(a, b)) {
            (a, b)
        } else {
            (b, a)
        }
    }

    fn echart(&self, k: EKey, chart_hint: usize) -> usize {
        let d = &self.edges[&k];
        if let Some(c) = d.chart {
            return c;
        }
        let ca = self.nodes[&k.0].0;
        let cb = self.nodes[&k.1].0;
        if ca == cb {
            ca
        } else {
            chart_hint
        }
    }

    fn dense(&self, k: EKey, chart_hint: usize, n: usize) -> (usize, Vec<DVec2>, Vec<DVec3>) {
        let ch = self.echart(k, chart_hint);
        let d = &self.edges[&k];
        let ts = crate::curves::linspace(0.0, 1.0, n + 1);
        let p: Vec<DVec2> = match &d.curve {
            Some(c) => ts.iter().map(|&t| c(t)).collect(),
            None => {
                let pa = self.node_param(k.0, ch);
                let pb = self.node_param(k.1, ch);
                ts.iter().map(|&t| pa + (pb - pa) * t).collect()
            }
        };
        let mut pp: Vec<DVec3> = p.iter().map(|q| self.map(ch, *q)).collect();
        // snap ends to the nodes' exact positions
        pp[0] = self.node_pos(k.0);
        pp[n] = self.node_pos(k.1);
        (ch, p, pp)
    }

    pub fn sample(&mut self, a: NodeId, b: NodeId, n: usize, chart_hint: usize) -> (usize, Vec<DVec2>, Vec<DVec3>) {
        let (k, rev) = self.edef(a, b);
        let key = (k, n);
        if !self.cache.contains_key(&key) {
            let (ch, p, pd) = self.dense(k, chart_hint, 160);
            let mut s = vec![0.0];
            for w in pd.windows(2) {
                s.push(s.last().unwrap() + (w[1] - w[0]).length());
            }
            let idxs: Vec<f64> = (0..pd.len()).map(|i| i as f64).collect();
            let total = *s.last().unwrap();
            let mut pp = Vec::with_capacity(n + 1);
            for t in crate::curves::linspace(0.0, total, n + 1) {
                let idx = interp(t, &s, &idxs);
                let i0 = (idx.floor() as isize).clamp(0, pd.len() as isize - 2) as usize;
                let f = idx - i0 as f64;
                pp.push(p[i0] * (1.0 - f) + p[i0 + 1] * f);
            }
            let mut pq: Vec<DVec3> = pp.iter().map(|q| self.map(ch, *q)).collect();
            pq[0] = pd[0];
            pq[n] = pd[pd.len() - 1];
            self.cache.insert(key, (ch, pp, pq));
        }
        let (ch, pp, pq) = self.cache[&key].clone();
        if rev {
            (ch, pp.into_iter().rev().collect(), pq.into_iter().rev().collect())
        } else {
            (ch, pp, pq)
        }
    }

    fn length(&mut self, a: NodeId, b: NodeId, chart_hint: usize) -> f64 {
        let (k, _) = self.edef(a, b);
        let (_, _, p) = self.dense(k, chart_hint, 96);
        p.windows(2).map(|w| (w[1] - w[0]).length()).sum()
    }

    // -- counts -------------------------------------------------------------------
    pub fn solve(&mut self) {
        let mut parent: HashMap<EKey, EKey> = HashMap::new();
        fn find(parent: &mut HashMap<EKey, EKey>, x: EKey) -> EKey {
            parent.entry(x).or_insert(x);
            let mut x = x;
            while parent[&x] != x {
                let gp = parent[&parent[&x]];
                parent.insert(x, gp);
                x = gp;
            }
            x
        }
        let blocks = self.blocks.clone();
        for bl in &blocks {
            let [a, b, c, d] = bl.c;
            for (x, y) in [(a, b), (b, c), (d, c), (a, d)] {
                self.edef(x, y);
            }
            let (e1, e2) = (self.ek(a, b), self.ek(d, c));
            let r1 = find(&mut parent, e1);
            let r2 = find(&mut parent, e2);
            parent.insert(r1, r2);
            let (e3, e4) = (self.ek(b, c), self.ek(a, d));
            let r3 = find(&mut parent, e3);
            let r4 = find(&mut parent, e4);
            parent.insert(r3, r4);
        }
        // a chord's count: its median edge at the target size, unless that
        // would stretch its longest edge past h_max
        let mut lens: HashMap<EKey, Vec<f64>> = HashMap::new();
        let mut mins: HashMap<EKey, usize> = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for bl in &blocks {
            let [a, b, c, d] = bl.c;
            for (x, y) in [(a, b), (b, c), (d, c), (a, d)] {
                let k = self.ek(x, y);
                if !seen.insert(k) {
                    continue;
                }
                let r = find(&mut parent, k);
                let l = self.length(x, y, bl.chart);
                lens.entry(r).or_default().push(l);
                let mn = self.edges[&k].min;
                let e = mins.entry(r).or_insert(1);
                *e = (*e).max(mn);
            }
        }
        let mut need: HashMap<EKey, usize> = HashMap::new();
        for (r, ls) in &lens {
            let mut s = ls.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = if s.len() % 2 == 1 { s[s.len() / 2] } else { 0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2]) };
            let mx = *s.last().unwrap();
            let n = ((med / self.h - 0.15).ceil() as i64).max((mx / self.h_max - 0.05).ceil() as i64).max(mins[r] as i64).max(1);
            need.insert(*r, n as usize);
        }
        let keys: Vec<EKey> = parent.keys().cloned().collect();
        self.counts.clear();
        for k in keys {
            let r = find(&mut parent, k);
            self.counts.insert(k, need[&r]);
        }
    }

    pub fn count(&self, a: NodeId, b: NodeId) -> usize {
        self.counts[&self.ek(a, b)]
    }

    // -- meshing ------------------------------------------------------------------
    pub fn mesh_block(&mut self, bl: &Block) -> (Mesh, Grid2<DVec2>, Grid2<DVec3>) {
        self.mesh_block_with(bl, true)
    }

    /// `uv` false skips the girth-based UVs (zero UVs; no girth table needed).
    pub fn mesh_block_with(&mut self, bl: &Block, uv: bool) -> (Mesh, Grid2<DVec2>, Grid2<DVec3>) {
        let [a, b, c, d] = bl.c;
        let ch = bl.chart;
        let n = self.count(a, b);
        let m = self.count(a, d);
        let mut sides: Vec<(Vec<DVec2>, Vec<DVec3>)> = Vec::new();
        for (x, y, cnt) in [(a, b, n), (d, c, n), (a, d, m), (b, c, m)] {
            let (ech, mut pp, p3) = self.sample(x, y, cnt, ch);
            if ech != ch {
                pp = p3.iter().map(|q| self.inv(ch, *q)).collect();
            }
            sides.push((pp, p3));
        }
        let (bb, b3) = (&sides[0].0, &sides[0].1);
        let (tt, t3) = (&sides[1].0, &sides[1].1);
        let (lf, l3) = (&sides[2].0, &sides[2].1);
        let (rr, r3) = (&sides[3].0, &sides[3].1);
        let (c00, c10, c01, c11) = (bb[0], bb[n], tt[0], tt[n]);
        let pp = Grid2::from_fn(n + 1, m + 1, |i, j| {
            let s = i as f64 / n as f64;
            let t = j as f64 / m as f64;
            bb[i] * (1.0 - t) + tt[i] * t + lf[j] * (1.0 - s) + rr[j] * s
                - (c00 * ((1.0 - s) * (1.0 - t)) + c10 * (s * (1.0 - t)) + c01 * ((1.0 - s) * t) + c11 * (s * t))
        });
        let p3 = Grid2::from_fn(n + 1, m + 1, |i, j| {
            // the Python assigns bottom, top, then left and right last
            if i == 0 {
                l3[j]
            } else if i == n {
                r3[j]
            } else if j == 0 {
                b3[i]
            } else if j == m {
                t3[i]
            } else {
                self.map(ch, pp.at(i, j))
            }
        });
        let uv = Grid2::from_fn(n + 1, m + 1, |i, j| if uv { self.uv(ch, pp.at(i, j)) } else { DVec2::ZERO });
        let mut g = grid(&p3, Some(&uv), bl.mat, false, false, -1);
        // orient: compare the grid normal with the chart's outward normal
        let (ci, cj) = (n / 2, m / 2);
        let du = p3.at((ci + 1).min(n), cj) - p3.at(ci.saturating_sub(1), cj);
        let dv = p3.at(ci, (cj + 1).min(m)) - p3.at(ci, cj.saturating_sub(1));
        let nn = du.cross(dv);
        let rf = self.normal(ch, pp.at(ci, cj));
        if nn.dot(rf) < 0.0 {
            g.flip();
        }
        (g, pp, p3)
    }
}
