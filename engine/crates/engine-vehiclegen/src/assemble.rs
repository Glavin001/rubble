//! Design -> finished part meshes (the Python `assemble.body_parts` plus
//! `build.py`'s merging and UV steps).

use crate::fx::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;

use crate::carbody::{CarBody, Side};
use crate::export::Marker;
use crate::mesh::{key3, reset_islands, Mat, Mesh};
use crate::panels::{boundary_loops, glass_from, panel, Key};
use crate::tube::Tube;
use crate::uvpack::{normalise_other, pack, Parts};

/// A vehicle design: the surface, its layout and its parts.
pub trait Design {
    fn name(&self) -> String;
    /// (linear RGB, metallic, roughness)
    fn paint(&self) -> ([f64; 3], f64, f64);
    fn tube(&self) -> Tube;
    fn zlines(&self) -> Vec<(&'static str, f64)>;
    fn vlines(&self) -> Vec<f64>;
    fn assign(&self, zf: &str, zt: &str, lo: usize, hi: usize, side: Side) -> Option<(String, Mat)>;
    fn build_layout(&self, cb: &mut CarBody);
    fn build_caps(&self, cb: &mut CarBody);
    fn lens_tint(&self, _name: &str) -> [f64; 4] {
        [1.0, 1.0, 1.0, 1.0]
    }
    /// (new parts, extras merged into existing parts, markers)
    fn parts(&self, cb: &CarBody) -> (Parts, Parts, Vec<Marker>);
    fn post(&self, _parts: &mut Parts) {}
    /// What the physics rigger needs besides the meshes (None: not rigged).
    fn rig_spec(&self) -> Option<RigSpec> {
        None
    }
}

/// A design's running gear, ends and physics for the crash rigger
/// (`engine-softbody` `authoring::gen2::vg_input`; the Python generator
/// writes the same data as `<design>.rig.json`). glTF metres.
#[derive(Clone)]
pub struct RigSpec {
    pub wheelbase: f64,
    pub track: f64,
    pub axle_f: f64,
    pub axle_r: f64,
    pub tyre_d: f64,
    pub tyre_w: f64,
    pub rim_d: f64,
    pub wheel_y: f64,
    pub z_front: f64,
    pub z_rear: f64,
    pub doors: Vec<String>,
    /// "trunk_lid" or "tailgate"
    pub rear_closure: String,
    /// Wheel openings: (centre z, centre y, radius).
    pub arches: Vec<(f64, f64, f64)>,
    /// The nose and tail faces: z at (x, y).
    pub front_z: std::rc::Rc<dyn Fn(f64, f64) -> f64>,
    pub rear_z: std::rc::Rc<dyn Fn(f64, f64) -> f64>,
    pub physics: crate::physics::Physics,
}

fn closure_inner(name: &str) -> Option<f64> {
    match name {
        "hood" | "trunk_lid" => Some(0.030),
        _ if name.starts_with("door_") => Some(0.035),
        _ => None,
    }
}

fn boundary_keys(sk: &Mesh) -> FxHashSet<Key> {
    let mut ks = FxHashSet::default();
    for lp in boundary_loops(sk) {
        for i in lp {
            ks.insert(key3(sk.v[i as usize], 1e-6));
        }
    }
    ks
}

/// The body's panels, glass and lenses, and the CarBody (layout, skins).
pub fn body_parts(d: &dyn Design, h: f64, uv: bool) -> (Parts, CarBody) {
    let tube = d.tube();
    let zl = d.zlines();
    let mut cb = CarBody::new(tube, &zl, &d.vlines(), h);
    d.build_layout(&mut cb);
    cb.build_nodes();
    cb.tube_blocks(&|zf, zt, lo, hi, side| d.assign(zf, zt, lo, hi, side));
    d.build_caps(&mut cb);
    cb.l.solve();
    let zmin = zl.iter().map(|z| z.1).fold(f64::INFINITY, f64::min) - 0.1;
    let zmax = zl.iter().map(|z| z.1).fold(f64::NEG_INFINITY, f64::max) + 0.1;
    if uv {
        cb.l.tube.girth_table(zmin, zmax);
    }
    // blocks grouped by part, in order of first appearance; chart names
    // decide the UV island numbering (sorted, as in the Python)
    let mut order: Vec<String> = Vec::new();
    let mut groups: FxHashMap<String, Vec<(Mesh, String)>> = FxHashMap::default();
    let blocks = cb.l.blocks.clone();
    for bl in &blocks {
        let (mut g, _, _) = if uv {
            cb.l.mesh_block(bl)
        } else {
            mesh_block_no_uv(&mut cb, bl)
        };
        g.set_islands(-1);
        let chart = cb.l.charts[bl.chart].0.clone();
        if !groups.contains_key(&bl.part) {
            order.push(bl.part.clone());
        }
        groups.entry(bl.part.clone()).or_default().push((g, chart));
    }
    let skin_of = |blocks: &[(Mesh, String)]| {
        let mut sk = Mesh::new();
        for (g, _) in blocks {
            sk.add(g);
        }
        sk.weld(1e-7);
        sk
    };
    let is_glassy = |n: &str| n.starts_with("glass") || n.ends_with("_lens");
    // boundary vertices shared by two panes (or two panels): shut lines
    let mut gcount: FxHashMap<Key, usize> = FxHashMap::default();
    let mut pcount: FxHashMap<Key, usize> = FxHashMap::default();
    let mut skins: BTreeMap<String, Mesh> = BTreeMap::new();
    for name in &order {
        let sk = skin_of(&groups[name]);
        let target = if is_glassy(name) { &mut gcount } else { &mut pcount };
        for k in boundary_keys(&sk) {
            *target.entry(k).or_insert(0) += 1;
        }
        skins.insert(name.clone(), sk);
    }
    let shared_glass: FxHashSet<Key> = gcount.into_iter().filter(|(_, c)| *c > 1).map(|(k, _)| k).collect();
    let shut: FxHashSet<Key> = pcount.into_iter().filter(|(_, c)| *c > 1).map(|(k, _)| k).collect();
    let mut parts = Parts::default();
    for name in &order {
        let blocks = &groups[name];
        let mut charts: Vec<&String> = blocks.iter().map(|(_, c)| c).collect();
        charts.sort();
        charts.dedup();
        let bm: Vec<(Mesh, i32)> = blocks
            .iter()
            .map(|(g, c)| (g.clone(), charts.iter().position(|x| *x == c).unwrap() as i32))
            .collect();
        let m = if name.starts_with("glass") {
            glass_from(&bm, 0.012, 0.006, Some(&shared_glass))
        } else if name.ends_with("_lens") {
            let mut lens = glass_from(&bm, 0.013, 0.003, None);
            lens.set_mat(Mat::LampLens);
            lens.tint(d.lens_tint(name));
            lens
        } else {
            panel(&bm, 0.0046, true, closure_inner(name), 0, Some(&shut))
        };
        parts.insert(name, m);
    }
    cb.skins = skins;
    (parts, cb)
}

/// Block meshing without the girth table (no panel UVs: the live preview).
fn mesh_block_no_uv(cb: &mut CarBody, bl: &crate::layout::Block) -> (Mesh, crate::mesh::Grid2<glam::DVec2>, crate::mesh::Grid2<glam::DVec3>) {
    cb.l.mesh_block_with(bl, false)
}

/// Everything a build produces.
pub struct Built {
    pub name: String,
    pub parts: Parts,
    pub markers: Vec<Marker>,
    pub paint: [f64; 3],
    pub metallic: f64,
    pub roughness: f64,
    /// panel UV atlas size in metres (0 without UVs)
    pub atlas: f64,
}

#[derive(Clone, Copy)]
pub struct BuildOpts {
    /// panel UVs (girth table + atlas); off for the live preview
    pub uv: bool,
    /// target edge length of the body mesh (metres)
    pub h: f64,
}

impl Default for BuildOpts {
    fn default() -> Self {
        BuildOpts { uv: true, h: 0.02 }
    }
}

pub fn build(d: &dyn Design, o: BuildOpts) -> Built {
    build_with_body(d, o).0
}

/// [`build`], also returning the CarBody (the tube, the layout and the
/// welded skins the physics rigger reads).
pub fn build_with_body(d: &dyn Design, o: BuildOpts) -> (Built, CarBody) {
    reset_islands();
    let (mut parts, cb) = body_parts(d, o.h, o.uv);
    let (new, extras, markers) = d.parts(&cb);
    for (k, m) in &extras.0 {
        parts.extend(k, m);
    }
    for (k, m) in new.0 {
        parts.insert(&k, m);
    }
    d.post(&mut parts);
    let atlas = if o.uv { pack(&mut parts, 0.012) } else { 0.0 };
    if o.uv {
        normalise_other(&mut parts);
    }
    let (paint, metallic, roughness) = d.paint();
    (Built { name: d.name(), parts, markers, paint, metallic, roughness, atlas }, cb)
}

/// Part names in a stable order with their triangle counts (for tests and
/// comparisons with the Python generator).
pub fn tri_counts(b: &Built) -> Vec<(String, usize)> {
    b.parts.0.iter().map(|(n, m)| (n.clone(), m.ntris())).collect()
}

