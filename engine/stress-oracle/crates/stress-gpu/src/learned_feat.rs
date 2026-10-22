//! The learned impact model's input features on the GPU (`shaders/learned_feat.slang`): the
//! calls schema's arrays for every call of a batch, as P1's Rust call builder
//! (stress-learned `outcome_calls.rs`) defines them, from the frame-start state
//! `learned.rs` gathers and static per-key tables uploaded once (`Statics`).
//!
//! The arrays' columns are placed through a map built from the schema's column lists
//! (`ColumnMap::new`, from `schema_columns()` or the caller's lists), so a schema change
//! (a column added, moved or removed) moves them without touching the kernels; a column the
//! kernels do not compute is an error, never a silent zero.

use crate::learned_timing::StageTimer;
use std::collections::BTreeMap;

use stress_ref::world::World;

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

pub const SC_F4: usize = 6;
pub const SB_F4: usize = 9;
pub const CON_F4: usize = 6;
pub const WORK_F: usize = 24;
/// Cracked-face candidates per chunk node (`feat_cracked_find`).
pub const CRACKED_PER_NODE: usize = 8;
pub const SCRATCH: usize = 131072;
pub const COLS_MAX: usize = 64;
pub const SECTIONS: usize = 7;
const NONE: u32 = u32::MAX;

/// Sections of the feature buffer, in the kernels' order: each the arrays it concatenates.
pub const SECTION_ARRAYS: [&[&str]; SECTIONS] = [
    &["chunk_feat", "chunk_qs_feat"],
    &["impactor_feat", "impactor_qs_feat"],
    &["global_feat"],
    &["bond_feat", "bond_qs_feat"],
    &["cracked_feat"],
    &["impactor_contact_feat"],
    &["structure_contact_feat"],
];

fn v(name: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{name}_{i}")).collect()
}

fn rot_cols() -> Vec<String> {
    let mut r = v("rot_src", 6);
    r.extend(v("rot_dst", 6));
    r
}

fn cat(parts: Vec<Vec<String>>) -> Vec<String> {
    parts.into_iter().flatten().collect()
}

fn one(s: &str) -> Vec<String> {
    vec![s.to_string()]
}

/// The columns the kernels compute (calls schema `/3`), per array, in their order. A test
/// (stress-rubble `learned_features.rs`) checks them against stress-learned's
/// `feature_columns()`, the builder's own lists.
pub fn schema_columns() -> BTreeMap<String, Vec<String>> {
    let mut m = BTreeMap::new();
    m.insert("chunk_feat".into(), cat(vec![one("log_mass"), v("inertia", 3), v("half_extent", 3), one("support_fixed"), one("support_pinned"), one("bond_damage_mean"), one("bond_damage_max"), one("n_live_bonds"), one("n_broken_bonds"), one("hops_to_contact"), one("wave_arrival"), one("hops_to_anchor"), v("gravity_body", 3), v("v_rel_body", 3), v("w_body", 3), one("punch_in_plug"), v("blast", 6), one("in_contact_any")]));
    m.insert("impactor_feat".into(), cat(vec![one("log_mass_island"), one("log_mass_chunk"), v("inertia", 3), one("log_radius"), one("log_modulus"), one("crushable"), v("gravity_body", 3), v("v_rel_body", 3), one("speed_over_c"), one("in_contact_any")]));
    m.insert("global_feat".into(), ["log_energy_ratio", "log_energy_ratio_plug", "log_dt_c_over_L", "dt_v_over_L", "n_impactors", "log_n_chunks", "gravity_mag", "anchored", "blast_log_tnt", "blast_z", "blast_impulse", "v_over_c"].iter().map(|s| s.to_string()).collect());
    m.insert("bond_feat".into(), cat(vec![one("log_area"), v("offset", 3), v("log_stiffness", 6), v("log_strength", 3), v("log_ga", 3), one("weibull"), one("damage"), one("crush"), one("fail_margin"), one("log1p_utilization"), one("log1p_peak_util_prev"), one("rebar"), one("log_qs_hertz_util"), one("punch_plug_bond"), one("log_punch_capacity_ratio"), v("v_rel_edge", 3), rot_cols()]));
    m.insert("cracked_feat".into(), cat(vec![one("log_area"), v("offset", 3), one("gap"), one("friction"), v("v_rel_edge", 3), one("frames_since_broken"), rot_cols()]));
    m.insert("impactor_contact_feat".into(), cat(vec![one("v_n"), v("v_t", 2), one("v_n_over_c"), one("log_mass_ratio"), one("log_radius"), one("log_modulus"), v("spin", 3), one("t_contact"), one("log_energy_ratio"), one("log_hertz_force"), one("hertz_duration"), one("is_self_impactor"), one("overlap"), one("friction"), one("restitution"), rot_cols()]));
    m.insert("structure_contact_feat".into(), cat(vec![one("v_n"), v("v_t", 2), one("v_n_over_c"), one("log_mass_ratio"), one("t_contact"), one("overlap"), one("friction"), one("log_area"), one("same_structure"), rot_cols()]));
    m.insert("chunk_qs_feat".into(), cat(vec![v("qs_dv_body", 3), v("qs_dw_body", 3)]));
    m.insert("impactor_qs_feat".into(), cat(vec![v("qs_dv_body", 3), v("qs_dw_body", 3)]));
    m.insert("bond_qs_feat".into(), ["log_qs_frame_peak_util", "qs_frame_broke", "qs_frame_break_time"].iter().map(|s| s.to_string()).collect());
    m
}

/// Where each section's columns go (`cols` in the kernels): per section its stride and, for
/// each column the kernels compute, its index in the section (NONE: not in the schema).
#[derive(Clone, Debug)]
pub struct ColumnMap {
    pub words: Vec<u32>,
    pub strides: [u32; SECTIONS],
    /// Per section, the target arrays and their column counts (a section's row holds them
    /// concatenated in this order).
    pub arrays: [Vec<(String, usize)>; SECTIONS],
}

impl ColumnMap {
    /// The map from the kernels' columns to `target`'s (the schema the network was trained
    /// on). Every target column must be one the kernels compute.
    pub fn new(target: &BTreeMap<String, Vec<String>>) -> Result<ColumnMap, String> {
        let kernel = schema_columns();
        let mut words = vec![NONE; SECTIONS * (COLS_MAX + 1)];
        let mut strides = [0u32; SECTIONS];
        let mut arrays: [Vec<(String, usize)>; SECTIONS] = Default::default();
        for (s, names) in SECTION_ARRAYS.iter().enumerate() {
            let mut have: Vec<String> = Vec::new();
            let mut want: Vec<String> = Vec::new();
            for a in names.iter() {
                let k = kernel.get(*a).ok_or_else(|| format!("no kernel columns for {a}"))?;
                // Columns are named per array (two arrays of one section may share a name).
                have.extend(k.iter().map(|c| format!("{a}.{c}")));
                let t = target.get(*a).ok_or_else(|| format!("the schema has no {a}"))?;
                want.extend(t.iter().map(|c| format!("{a}.{c}")));
                arrays[s].push((a.to_string(), t.len()));
            }
            if have.len() > COLS_MAX {
                return Err(format!("section {s}: {} columns, the map holds {COLS_MAX}", have.len()));
            }
            for w in &want {
                if !have.contains(w) {
                    return Err(format!("feature column {w} is not computed on the GPU (learned_feat.slang)"));
                }
            }
            strides[s] = want.len() as u32;
            words[s * (COLS_MAX + 1)] = want.len() as u32;
            for (k, h) in have.iter().enumerate() {
                words[s * (COLS_MAX + 1) + 1 + k] = want.iter().position(|w| w == h).map_or(NONE, |i| i as u32);
            }
        }
        Ok(ColumnMap { words, strides, arrays })
    }
}

/// The static per-key tables (`learned_feat.slang`'s `sc`, `sb`, `sinc`, `slook`), from the
/// world's pristine structures and its scene, as the builder's `Statics` takes them.
pub struct Statics {
    pub sc: Vec<[f32; 4]>,
    pub sb: Vec<[f32; 4]>,
    pub sinc: Vec<u32>,
    pub slook: Vec<[u32; 4]>,
    /// Stable chunk rows and bond rows.
    pub chunks: usize,
    pub rows: usize,
    /// Per stable chunk row: (structure, chunk); per bond row: (structure, source).
    pub chunk_key: Vec<(u32, u32)>,
    pub bond_key: Vec<(u32, u32)>,
}

pub fn statics(w: &World) -> Result<Statics, String> {
    let scale = w.scene.sim.stiffness_scale;
    let f = |x: f64| x as f32;
    let mut chunk0 = Vec::new();
    let mut n = 0usize;
    for st in &w.solver.structures {
        chunk0.push(n);
        n += st.chunks.len();
    }
    let mut sc = Vec::with_capacity(n * SC_F4);
    let mut chunk_key = Vec::with_capacity(n);
    for (s, st) in w.solver.structures.iter().enumerate() {
        let body = w.scene.bodies.get(s).ok_or("a structure without its scene body")?;
        for (c, ch) in st.chunks.iter().enumerate() {
            let desc = body.chunks.get(c).ok_or("a chunk without its scene desc")?;
            let mat = w.scene.material(&desc.material);
            let plane = mat.youngs_modulus * scale / (1.0 - mat.poisson_ratio * mat.poisson_ratio);
            let fixed = desc.support == stress_ref::scene::Support::Fixed;
            let pinned = desc.support == stress_ref::scene::Support::Pinned;
            let h = ch.half_extents;
            let q = desc.orientation; // w, x, y, z
            sc.push([f(ch.mass), f(mat.youngs_modulus * scale), f(mat.density), f(mat.tensile_strength)]);
            sc.push([f(mat.fracture_energy.tension), f(mat.friction), f32::from(u8::from(fixed)), f32::from(u8::from(pinned))]);
            sc.push([f(h.x), f(h.y), f(h.z), f(h.norm())]);
            sc.push([f(ch.inertia.col(0).x), f(ch.inertia.col(1).y), f(ch.inertia.col(2).z), f(plane)]);
            sc.push([f(ch.center.x), f(ch.center.y), f(ch.center.z), 0.0]);
            sc.push([f(q[1]), f(q[2]), f(q[3]), f(q[0])]);
            chunk_key.push((s as u32, c as u32));
        }
    }
    let mut sb = Vec::new();
    let mut bond_key = Vec::new();
    let mut ends: Vec<(u32, u32)> = Vec::new();
    for (s, bonds) in w.solver.bonds.iter().enumerate() {
        for b in bonds {
            let g = &b.geometry;
            let (a, bb) = ((chunk0[s] + g.a) as u32, (chunk0[s] + g.b) as u32);
            let k = &b.stiffness;
            let t = &b.strength;
            let inf = |x: f64| if x.is_finite() { x as f32 } else { -1.0 };
            let rupture = b.rebar.as_ref().map_or(0.0, |r| r.rupture_work);
            sb.push([f(g.centroid.x), f(g.centroid.y), f(g.centroid.z), f(g.area)]);
            sb.push([f(g.normal.x), f(g.normal.y), f(g.normal.z), f(g.width[0])]);
            sb.push([f(g.t1.x), f(g.t1.y), f(g.t1.z), f(g.width[1])]);
            sb.push([f(g.length), f(k.kn), f(k.ks), f(k.kb_t1)]);
            sb.push([f(k.kb_t2), f(k.kt), f(b.weibull), f32::from(u8::from(b.rebar.is_some()))]);
            sb.push([inf(t.tensile), inf(t.compressive), inf(t.cohesion), inf(t.friction)]);
            sb.push([inf(t.shear_cap), inf(t.g_tension), inf(t.g_shear), inf(t.g_compression)]);
            sb.push([f32::from_bits(a), f32::from_bits(bb), f32::from_bits(s as u32), f32::from_bits(b.source as u32)]);
            sb.push([f32::from(u8::from(t.kind == stress_ref::material::BondKind::Ductile)), f32::from(u8::from(t.softening)), f(rupture), 0.0]);
            bond_key.push((s as u32, b.source as u32));
            ends.push((a, bb));
        }
    }
    let rows = ends.len();
    // incidence: per chunk row, its bond rows (in row order)
    let mut inc: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (r, &(a, b)) in ends.iter().enumerate() {
        inc[a as usize].push(r as u32);
        inc[b as usize].push(r as u32);
    }
    let structures = w.solver.structures.len();
    let mut sinc = vec![structures as u32, n as u32];
    sinc.extend(chunk0.iter().map(|&c| c as u32));
    let mut off = 0u32;
    for l in &inc {
        sinc.push(off);
        off += l.len() as u32;
    }
    sinc.push(off);
    for l in &inc {
        sinc.extend(l);
    }
    let mut slook: Vec<[u32; 4]> = ends.iter().enumerate().map(|(r, &(a, b))| [a.min(b), a.max(b), r as u32, 0]).collect();
    slook.sort_unstable();
    for p in slook.windows(2) {
        if p[0][0] == p[1][0] && p[0][1] == p[1][1] {
            return Err(format!("two bonds join stable chunks {} and {}: the feature lookup keys bonds by their chunks", p[0][0], p[0][1]));
        }
    }
    Ok(Statics { sc, sb, sinc, slook, chunks: n, rows, chunk_key, bond_key })
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FeatParams {
    impactor_count: u32,
    frame: u32,
    layout_bonds: u32,
    rows: u32,
    dt: f32,
    speed: f32,
    restitution: f32,
    contact_friction: f32,
    gravity: [f32; 4],
    bases: [u32; 4],
    bases2: [u32; 4],
    caps: [u32; 4],
    count_only: u32,
    max_nodes: u32,
    layout_cap: u32,
    max_bonds_cap: u32,
    prev_frame: u32,
    engine_base: u32,
    chunk_count: u32,
    pad1: u32,
}

const BINDINGS: [Binding; 28] = [
    Binding::Uniform,   // 0 params
    Binding::Storage,   // 1 ctl
    Binding::StorageRw, // 2 calls
    Binding::StorageRw, // 3 nodes
    Binding::Storage,   // 4 bonds
    Binding::Storage,   // 5 cisl
    Binding::Storage,   // 6 cislf
    Binding::Storage,   // 7 sc
    Binding::Storage,   // 8 sb
    Binding::Storage,   // 9 sinc
    Binding::Storage,   // 10 slook
    Binding::StorageRw, // 11 bmap
    Binding::StorageRw, // 12 key_node
    Binding::Storage,   // 13 chunk statics
    Binding::Storage,   // 14 bond statics
    Binding::Storage,   // 15 bond records
    Binding::Storage,   // 16 impactors
    Binding::StorageRw, // 17 contacts
    Binding::StorageRw, // 18 work
    Binding::StorageRw, // 19 feat
    Binding::StorageRw, // 20 edges
    Binding::Storage,   // 21 cols
    Binding::StorageRw, // 22 bedge
    Binding::StorageRw, // 23 fctl
    Binding::Storage,   // 24 islands
    Binding::Storage,   // 25 ctab
    Binding::Storage,   // 26 engine bodies
    Binding::StorageRw, // 27 chunk state
];

struct Kernels {
    map: Kernel,
    keys: Kernel,
    brow: Kernel,
    contacts: Kernel,
    scales: Kernel,
    summ: Kernel,
    graph: Kernel,
    ic: Kernel,
    cracked_find: Kernel,
    cracked: Kernel,
    sc: Kernel,
    prefix: Kernel,
    nodes: Kernel,
    bonds: Kernel,
    deform: Kernel,
}

/// The feature buffer's layout (float offsets) and the edge capacities (undirected).
#[derive(Clone, Copy, Debug)]
pub struct FeatLayout {
    pub chunk: u32,
    pub imp: u32,
    pub glob: u32,
    pub bond: u32,
    pub cracked: u32,
    pub ic: u32,
    pub sc: u32,
    pub raw: u32,
    pub total: u32,
    pub cap_cracked: u32,
    pub cap_ic: u32,
    pub cap_sc: u32,
}

/// The scene's contact parameters the features take.
#[derive(Clone, Copy, Debug)]
pub struct FeatScene {
    pub restitution: f32,
    /// The scene's pair friction override, if any.
    pub contact_friction: Option<f32>,
}

pub struct Features {
    k: Kernels,
    pub map: ColumnMap,
    pub layout: FeatLayout,
    pub statics: Statics,
    pub scene: FeatScene,
    pub sc: wgpu::Buffer,
    pub sb: wgpu::Buffer,
    pub sinc: wgpu::Buffer,
    pub slook: wgpu::Buffer,
    pub bmap: wgpu::Buffer,
    pub key_node: wgpu::Buffer,
    pub contacts: wgpu::Buffer,
    pub work: wgpu::Buffer,
    pub feat: wgpu::Buffer,
    pub edges: wgpu::Buffer,
    pub cols: wgpu::Buffer,
    pub bedge: wgpu::Buffer,
    pub fctl: wgpu::Buffer,
    layout_cap: u32,
    max_nodes: u32,
    max_bonds: u32,
}

fn rw(gpu: &Gpu, label: &str, bytes: u64) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

/// Words of `fctl`: totals, then per call (counts, bases) of the three edge sets.
pub const FCTL_WORDS: usize = 8 + 6 * 64;

/// The buffers of the batch the features read (`learned.rs` owns them).
pub struct BatchBuffers<'a> {
    pub ctl: &'a wgpu::Buffer,
    pub calls: &'a wgpu::Buffer,
    pub nodes: &'a wgpu::Buffer,
    pub bonds: &'a wgpu::Buffer,
    pub cisl: &'a wgpu::Buffer,
    pub cislf: &'a wgpu::Buffer,
    pub imps: &'a wgpu::Buffer,
    pub chunks: &'a wgpu::Buffer,
    pub bond_static: &'a wgpu::Buffer,
    pub bond_dyn: &'a wgpu::Buffer,
    pub islands: &'a wgpu::Buffer,
    pub ctab: &'a wgpu::Buffer,
    pub bodies: &'a wgpu::Buffer,
    pub state: &'a wgpu::Buffer,
    pub engine_base: u32,
    pub chunk_count: u32,
    /// The frame of the batch `feat_deform` applies.
    pub prev_frame: u32,
    pub impactor_count: u32,
    pub frame: u32,
    pub dt: f64,
    pub speed: f64,
    pub gravity: [f32; 3],
}

impl Features {
    /// `max_nodes`, `max_bonds`, `max_cisl`, `max_calls`: the batch's capacities; the edge
    /// sets' follow from them (cracked faces up to the bonds, impactor contacts up to 32 per
    /// call island, structure contacts up to 4 per node).
    pub fn new(gpu: &Gpu, statics: Statics, scene: FeatScene, map: ColumnMap, max_nodes: u32, max_bonds: u32, max_cisl: u32, max_calls: u32, layout_cap: u32) -> Features {
        let k = |e: &str| gpu.compute(&shaders::LEARNED_FEAT, e, &BINDINGS);
        let kernels = Kernels {
            map: k("feat_map"),
            keys: k("feat_keys"),
            brow: k("feat_brow"),
            contacts: k("feat_contacts"),
            scales: k("feat_scales"),
            summ: k("feat_summ"),
            graph: k("feat_graph"),
            ic: k("feat_ic"),
            cracked_find: k("feat_cracked_find"),
            cracked: k("feat_cracked"),
            sc: k("feat_sc"),
            prefix: k("feat_prefix"),
            nodes: k("feat_nodes"),
            bonds: k("feat_bonds"),
            deform: k("feat_deform"),
        };
        let st = &map.strides;
        let (cap_cracked, cap_ic, cap_sc) = (max_bonds, 32 * max_cisl, 4 * max_nodes);
        let imp_rows = 32 * max_calls;
        let mut at = 0u32;
        let mut take = |n: u32| {
            let a = at;
            at += n;
            a
        };
        let chunk = take(max_nodes * st[0]);
        let imp = take(imp_rows * st[1]);
        let glob = take(max_calls * st[2]);
        let bond = take(2 * max_bonds * st[3]);
        let cracked = take(2 * cap_cracked * st[4]);
        let ic = take(2 * cap_ic * st[5]);
        let sc_b = take(2 * cap_sc * st[6]);
        let raw = take(7 * (max_nodes + imp_rows));
        let layout = FeatLayout { chunk, imp, glob, bond, cracked, ic, sc: sc_b, raw, total: at, cap_cracked, cap_ic, cap_sc };
        let edge_words = 2 * 2 * (cap_cracked + cap_ic + cap_sc) + 2 * (cap_cracked + cap_ic + cap_sc);
        let rows = statics.rows.max(1);
        let mut bmap_init = vec![NONE; layout_cap as usize + 2 * rows];
        bmap_init.truncate(layout_cap as usize + 2 * rows);
        let one = [[0f32; 4]];
        Features {
            k: kernels,
            sc: gpu.storage("learned static chunks", if statics.sc.is_empty() { &one[..] } else { &statics.sc }),
            sb: gpu.storage("learned static bonds", if statics.sb.is_empty() { &one[..] } else { &statics.sb }),
            sinc: gpu.storage("learned static incidence", &statics.sinc),
            slook: gpu.storage("learned bond lookup", if statics.slook.is_empty() { &[[0u32; 4]][..] } else { &statics.slook }),
            bmap: gpu.storage("learned bond map", &bmap_init),
            key_node: gpu.storage("learned key map", &vec![[0u32; 2]; statics.chunks.max(1)]),
            contacts: rw(gpu, "learned contacts", 16 * (CON_F4 * 32 * max_cisl as usize) as u64),
            work: rw(gpu, "learned feature work", 4 * (WORK_F * max_nodes as usize + 3 * SCRATCH) as u64),
            feat: rw(gpu, "learned features", 4 * layout.total as u64),
            edges: rw(gpu, "learned edges", 4 * edge_words as u64),
            cols: gpu.storage("learned column map", &map.words),
            bedge: rw(gpu, "learned bond edges", 16 * (max_bonds as u64 + (CRACKED_PER_NODE as u64) * max_nodes as u64)),
            fctl: rw(gpu, "learned feature control", 4 * FCTL_WORDS as u64),
            map,
            layout,
            statics,
            scene,
            layout_cap,
            max_nodes,
            max_bonds,
        }
    }

    fn params(&self, gpu: &Gpu, b: &BatchBuffers, count_only: bool) -> wgpu::Buffer {
        let l = &self.layout;
        let g = b.gravity;
        gpu.uniform(
            "learned feature params",
            &FeatParams {
                impactor_count: b.impactor_count,
                frame: b.frame,
                layout_bonds: self.layout_cap,
                rows: self.statics.rows as u32,
                dt: b.dt as f32,
                speed: b.speed as f32,
                restitution: self.scene.restitution,
                contact_friction: self.scene.contact_friction.unwrap_or(-1.0),
                gravity: [g[0], g[1], g[2], (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()],
                bases: [l.chunk, l.imp, l.glob, l.bond],
                bases2: [l.cracked, l.ic, l.sc, l.raw],
                caps: [l.cap_cracked, l.cap_ic, l.cap_sc, 64],
                count_only: count_only as u32,
                max_nodes: self.max_nodes,
                layout_cap: self.layout_cap,
                max_bonds_cap: self.max_bonds,
                prev_frame: b.prev_frame,
                engine_base: b.engine_base,
                chunk_count: b.chunk_count,
                pad1: 0,
            },
        )
    }

    fn bufs<'a>(&'a self, params: &'a wgpu::Buffer, b: &'a BatchBuffers) -> [&'a wgpu::Buffer; 28] {
        [params, b.ctl, b.calls, b.nodes, b.bonds, b.cisl, b.cislf, &self.sc, &self.sb, &self.sinc, &self.slook, &self.bmap, &self.key_node, b.chunks, b.bond_static, b.bond_dyn, b.imps, &self.contacts, &self.work, &self.feat, &self.edges, &self.cols, &self.bedge, &self.fctl, b.islands, b.ctab, b.bodies, b.state]
    }

    /// Before `learned_finish` (positions still absolute): the bond map, the key map, the
    /// call bonds' rows and the predicted contacts. `calls_args` / `nodes`: indirect
    /// arguments over the calls (one group each) and the batch's capacities for the
    /// thread-per-item kernels (which exit past the batch's counts).
    pub fn encode_pre(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, b: &BatchBuffers, calls_args: (&wgpu::Buffer, u64), max_bonds: u32, timer: Option<&StageTimer>) {
        let params = self.params(gpu, b, false);
        let bufs = self.bufs(&params, b);
        let run = |enc: &mut wgpu::CommandEncoder, k: &Kernel, groups: Option<u32>| {
            let group = gpu.bind(&k.group, &bufs);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: timer.and_then(|t| t.writes("features")) });
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &group, &[]);
            match groups {
                Some(n) => pass.dispatch_workgroups(n.max(1), 1, 1),
                None => pass.dispatch_workgroups_indirect(calls_args.0, calls_args.1),
            }
        };
        run(enc, &self.k.map, Some(self.layout_cap.div_ceil(64)));
        run(enc, &self.k.keys, Some(self.max_nodes.div_ceil(64)));
        run(enc, &self.k.brow, Some(max_bonds.div_ceil(64)));
        run(enc, &self.k.contacts, None);
    }

    /// After `learned_finish`: scales, summaries, graph distances, the edge sets (counted,
    /// placed, written), node and bond features.
    pub fn encode_post(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, b: &BatchBuffers, calls_args: (&wgpu::Buffer, u64), max_bonds: u32, timer: Option<&StageTimer>) {
        let params = self.params(gpu, b, false);
        let count = self.params(gpu, b, true);
        let bufs = self.bufs(&params, b);
        let bufs_count = self.bufs(&count, b);
        let run = |enc: &mut wgpu::CommandEncoder, k: &Kernel, groups: Option<u32>, counting: bool| {
            let group = gpu.bind(&k.group, if counting { &bufs_count } else { &bufs });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: timer.and_then(|t| t.writes("features")) });
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &group, &[]);
            match groups {
                Some(n) => pass.dispatch_workgroups(n.max(1), 1, 1),
                None => pass.dispatch_workgroups_indirect(calls_args.0, calls_args.1),
            }
        };
        run(enc, &self.k.scales, Some(1), false);
        run(enc, &self.k.summ, Some(self.max_nodes.div_ceil(64)), false);
        run(enc, &self.k.graph, None, false);
        run(enc, &self.k.cracked_find, None, false);
        for k in [&self.k.ic, &self.k.cracked, &self.k.sc] {
            run(enc, k, None, true);
        }
        run(enc, &self.k.prefix, Some(1), false);
        for k in [&self.k.ic, &self.k.cracked, &self.k.sc] {
            run(enc, k, None, false);
        }
        run(enc, &self.k.nodes, None, false);
        run(enc, &self.k.bonds, Some(max_bonds.div_ceil(64)), false);
    }

    /// After the split pass of a frame whose batch was applied: each chunk of a call that
    /// stood keeps its decoded departure from its body's fit as its hidden state
    /// (`STRESS_DECODED_DEFORMATION=1`, P0's kept-deformation boundary).
    pub fn encode_deform(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, b: &BatchBuffers, timer: Option<&StageTimer>) {
        let params = self.params(gpu, b, false);
        let bufs = self.bufs(&params, b);
        let group = gpu.bind(&self.k.deform.group, &bufs);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: timer.and_then(|t| t.writes("deform")) });
        pass.set_pipeline(&self.k.deform.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(b.chunk_count.div_ceil(64).max(1), 1, 1);
    }

    /// The feature control words' range (edge totals, overflow bits) for the frame's read.
    pub fn range(&self) -> (&wgpu::Buffer, u64, u64) {
        (&self.fctl, 0, 32)
    }
}
