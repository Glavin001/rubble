//! The call builder: P1's `learned-impact-frames/1` frame records → P2's per-call
//! `learned-impact-outcome/3` records (P2's `OUTCOME_CALLS_SCHEMA.md` and `schema.py` are
//! normative). The single source of every model feature: P3 ports this code to the engine.
//!
//! One call per frame whose routing put chunks in an island: every routed chunk and
//! impactor (island 0), as P0's closed-loop harness calls the provider. Inputs come only
//! from the frame-start state and the frame's causal inputs (predicted contacts, the punch,
//! the QS prior, near pairs); targets and baselines from the recorded outcome.
//!
//! Two sources share one feature computation:
//! - a recorded event (`build_event`), with targets;
//! - a live `World` (`live_call`), for the closed loop: its inputs are captured by the
//!   recorder's own `capture_inputs`, so a live call equals the recorded call of the same
//!   state (tested). No targets.
//!
//! Stable keys come from `stress_ref::keys` only.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use stress_ref::keys;
use stress_ref::math::{Mat3, Quat, Vec3};
use stress_ref::scene::{ImpactorShape, Scene, Support};
use stress_ref::world::{tiered_impact_speed, World};

use crate::catalog::{self, Catalog};
use crate::outcome::{capture_inputs, capture_state, capture_static, Arrays, Index, PHASES};

pub const CALLS_SCHEMA: &str = "learned-impact-outcome/4";
/// `/4` plus the island's complete oracle state at the frame's start and end
/// (`full_state_calls`), for events recorded with the full state (recorder revision 9 on).
pub const CALLS_SCHEMA_FULL: &str = "learned-impact-outcome/5";
/// Bumped whenever the features or targets this builder writes change.
pub const BUILDER_REVISION: u32 = 7;
pub const N_TIME_BINS: usize = 8;
const G0: f64 = 9.81;
const TINY: f64 = 1e-6;
/// A cracked face or a structure contact is an edge while its gap is within this many chunk
/// sizes `L`.
const CONTACT_MARGIN: f64 = 0.25;

fn v(name: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{name}_{i}")).collect()
}

fn rot_cols() -> Vec<String> {
    let mut r = v("rot_src", 6);
    r.extend(v("rot_dst", 6));
    r
}

macro_rules! cols {
    ($($e:expr),* $(,)?) => {{ let mut out: Vec<String> = Vec::new(); $( out.extend(Into::<Cols>::into($e).0); )* out }};
}

struct Cols(Vec<String>);
impl From<&str> for Cols {
    fn from(s: &str) -> Cols {
        Cols(vec![s.to_string()])
    }
}
impl From<Vec<String>> for Cols {
    fn from(v: Vec<String>) -> Cols {
        Cols(v)
    }
}

/// The feature columns of every feature array, in P2's `schema.py` order.
pub fn feature_columns() -> BTreeMap<&'static str, Vec<String>> {
    let mut m = BTreeMap::new();
    m.insert("chunk_feat", cols!["log_mass", v("inertia", 3), v("half_extent", 3), "support_fixed", "support_pinned", "bond_damage_mean", "bond_damage_max", "n_live_bonds", "n_broken_bonds", "hops_to_contact", "wave_arrival", "hops_to_anchor", v("gravity_body", 3), v("v_rel_body", 3), v("w_body", 3), "punch_in_plug", v("blast", 6), "in_contact_any", v("def_u_body", 3), v("def_rot_body", 3), v("def_v_body", 3), v("def_w_body", 3)]);
    m.insert("impactor_feat", cols!["log_mass_island", "log_mass_chunk", v("inertia", 3), "log_radius", "log_modulus", "crushable", v("gravity_body", 3), v("v_rel_body", 3), "speed_over_c", "in_contact_any"]);
    m.insert("global_feat", cols!["log_energy_ratio", "log_energy_ratio_plug", "log_dt_c_over_L", "dt_v_over_L", "n_impactors", "log_n_chunks", "gravity_mag", "anchored", "blast_log_tnt", "blast_z", "blast_impulse", "v_over_c"]);
    m.insert("bond_feat", cols!["log_area", v("offset", 3), v("log_stiffness", 6), v("log_strength", 3), v("log_ga", 3), "weibull", "damage", "crush", "fail_margin", "log1p_utilization", "log1p_peak_util_prev", "rebar", "log_qs_hertz_util", "punch_plug_bond", "log_punch_capacity_ratio", v("v_rel_edge", 3), rot_cols()]);
    m.insert("cracked_feat", cols!["log_area", v("offset", 3), "gap", "friction", v("v_rel_edge", 3), "frames_since_broken", rot_cols()]);
    m.insert("impactor_contact_feat", cols!["v_n", v("v_t", 2), "v_n_over_c", "log_mass_ratio", "log_radius", "log_modulus", v("spin", 3), "t_contact", "log_energy_ratio", "log_hertz_force", "hertz_duration", "is_self_impactor", "overlap", "friction", "restitution", rot_cols()]);
    m.insert("structure_contact_feat", cols!["v_n", v("v_t", 2), "v_n_over_c", "log_mass_ratio", "t_contact", "overlap", "friction", "log_area", "same_structure", rot_cols()]);
    m.insert("chunk_qs_feat", cols![v("qs_dv_body", 3), v("qs_dw_body", 3)]);
    m.insert("impactor_qs_feat", cols![v("qs_dv_body", 3), v("qs_dw_body", 3)]);
    m.insert("bond_qs_feat", cols!["log_qs_frame_peak_util", "qs_frame_broke", "qs_frame_break_time"]);
    m
}

// ------------------------------------------------------------------ the record

/// A frame record, from disk or from memory (the live entry's one-frame record).
pub struct Record {
    pub name: String,
    pub scene: Scene,
    pub ctx: Value,
    pub outcome: Option<Value>,
    pub frames: Vec<Value>,
    f: HashMap<String, (Vec<usize>, Vec<f32>)>,
    i: HashMap<String, (Vec<usize>, Vec<i32>)>,
    /// A thinned training event (`context.storage.kept_frames`): frame and boundary index
    /// to stored row.
    frame_row: Option<HashMap<usize, usize>>,
    state_row: Option<HashMap<usize, usize>>,
}

impl Record {
    pub fn load(dir: &Path) -> Result<Record, String> {
        let read = |n: &str| -> Result<Value, String> { serde_json::from_slice(&fs::read(dir.join(n)).map_err(|e| format!("{}/{n}: {e}", dir.display()))?).map_err(|e| format!("{n}: {e}")) };
        let ctx = read("context.json")?;
        let schema = ctx["schema"].as_str().unwrap_or("");
        let rev = ctx["recorder_revision"].as_u64().unwrap_or(0);
        if schema != crate::outcome::OUTCOME_SCHEMA && !(schema == "learned-impact-outcome/1" && rev == 1) {
            return Err(format!("{}: schema {schema:?}", dir.display()));
        }
        let scene: Scene = serde_json::from_value(read("scene.json")?).map_err(|e| format!("scene.json: {e}"))?;
        let cat = Catalog::load(&dir.join("catalog.json"))?;
        let (mut f, mut i) = (HashMap::new(), HashMap::new());
        for (name, e) in &cat.arrays {
            match e.dtype.as_str() {
                "f32" => {
                    f.insert(name.clone(), (e.shape.clone(), catalog::read_f32(dir, e)?));
                }
                "i32" => {
                    i.insert(name.clone(), (e.shape.clone(), catalog::read_i32(dir, e)?));
                }
                // Exact f64 copies of state, break times and impactor impulses: labels and
                // restart data, not features.
                "f64" => {}
                d => return Err(format!("{name}: dtype {d}")),
            }
        }
        let frames = read("frames.json")?.as_array().cloned().unwrap_or_default();
        let outcome = read("outcome.json").ok();
        let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let index = |k: &str| ctx["storage"][k].as_array().map(|v| v.iter().enumerate().map(|(r, x)| (x.as_u64().unwrap_or(0) as usize, r)).collect::<HashMap<_, _>>());
        let (frame_row, state_row) = (index("kept_frames"), index("state_rows"));
        Ok(Record { name, scene, ctx, outcome, frames, f, i, frame_row, state_row })
    }

    fn from_arrays(name: String, scene: Scene, ctx: Value, a: Arrays) -> Record {
        let mut f = HashMap::new();
        for (k, (tail, data, rows)) in a.f32s {
            f.insert(k, (std::iter::once(rows).chain(tail).collect(), data));
        }
        let mut i = HashMap::new();
        for (k, (tail, data, rows)) in a.i32s {
            i.insert(k, (std::iter::once(rows).chain(tail).collect(), data));
        }
        Record { name, scene, ctx, outcome: None, frames: Vec::new(), f, i, frame_row: None, state_row: None }
    }

    pub fn revision(&self) -> u64 {
        self.ctx["recorder_revision"].as_u64().unwrap_or(0)
    }

    fn has(&self, n: &str) -> bool {
        self.f.contains_key(n) || self.i.contains_key(n)
    }

    fn fa(&self, n: &str) -> Result<&(Vec<usize>, Vec<f32>), String> {
        self.f.get(n).ok_or_else(|| format!("{}: array {n} missing", self.name))
    }

    fn ia(&self, n: &str) -> Result<&(Vec<usize>, Vec<i32>), String> {
        self.i.get(n).ok_or_else(|| format!("{}: array {n} missing", self.name))
    }

    /// The stored row of frame (or frame boundary) `r` of array `n` in a thinned event;
    /// `r` itself otherwise and for arrays kept whole.
    fn row(&self, n: &str, r: usize) -> Result<usize, String> {
        let map = if n.starts_with("state_") {
            &self.state_row
        } else if n.starts_with("frame_") || n.starts_with("qs_") {
            &self.frame_row
        } else {
            &None
        };
        match map {
            Some(m) => m.get(&r).copied().ok_or_else(|| format!("{}: frame {r} of {n} was not kept (learned_impact_outcome_record --regenerate)", self.name)),
            None => Ok(r),
        }
    }

    /// The frames whose arrays the record holds.
    pub fn frames_held(&self) -> Vec<usize> {
        match &self.frame_row {
            Some(m) => {
                let mut v: Vec<usize> = m.keys().copied().collect();
                v.sort_unstable();
                v
            }
            None => (0..self.rows("frame_phase")).collect(),
        }
    }

    /// Row `r` of an f32 array (all trailing dimensions).
    fn rf(&self, n: &str, r: usize) -> Result<&[f32], String> {
        let r = self.row(n, r)?;
        let (shape, d) = self.fa(n)?;
        let w: usize = shape[1..].iter().product();
        d.get(r * w..(r + 1) * w).ok_or_else(|| format!("{}: {n} row {r} past {}", self.name, shape[0]))
    }

    fn ri(&self, n: &str, r: usize) -> Result<&[i32], String> {
        let r = self.row(n, r)?;
        let (shape, d) = self.ia(n)?;
        let w: usize = shape[1..].iter().product();
        d.get(r * w..(r + 1) * w).ok_or_else(|| format!("{}: {n} row {r} past {}", self.name, shape[0]))
    }

    fn rows(&self, n: &str) -> usize {
        self.f.get(n).map(|x| x.0[0]).or_else(|| self.i.get(n).map(|x| x.0[0])).unwrap_or(0)
    }
}

/// A symmetric matrix from (xx, yy, zz, xy, xz, yz).
fn sym6(x: &[f32]) -> Mat3 {
    let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    Mat3 { m: [[x[0], x[3], x[4]], [x[3], x[1], x[5]], [x[4], x[5], x[2]]] }
}

fn v3(s: &[f32], k: usize) -> Vec3 {
    Vec3::new(s[3 * k] as f64, s[3 * k + 1] as f64, s[3 * k + 2] as f64)
}

fn quat(s: &[f32], k: usize) -> Quat {
    Quat::from_wxyz([s[4 * k] as f64, s[4 * k + 1] as f64, s[4 * k + 2] as f64, s[4 * k + 3] as f64]).normalized()
}

/// Rows of a world frame (t1, t2, n).
#[derive(Clone, Copy)]
struct Frame3 {
    t1: Vec3,
    t2: Vec3,
    n: Vec3,
}

impl Frame3 {
    fn of(t1: Vec3, n: Vec3) -> Frame3 {
        Frame3 { t1, t2: n.cross(t1), n }
    }
    fn reversed(&self) -> Frame3 {
        Frame3 { t1: -self.t1, t2: self.t2, n: -self.n }
    }
    fn local(&self, x: Vec3) -> [f64; 3] {
        [self.t1.dot(x), self.t2.dot(x), self.n.dot(x)]
    }
    fn rows(&self) -> [f32; 9] {
        let mut o = [0f32; 9];
        for (k, r) in [self.t1, self.t2, self.n].iter().enumerate() {
            o[3 * k] = r.x as f32;
            o[3 * k + 1] = r.y as f32;
            o[3 * k + 2] = r.z as f32;
        }
        o
    }
    /// First two columns of R_bodyᵀ·R_edge (R_edge's columns t1, t2, n): t1 and t2 in the body frame.
    fn rot6(&self, body: &Mat3) -> [f64; 6] {
        let bt = body.transpose();
        let (a, b) = (bt * self.t1, bt * self.t2);
        [a.x, a.y, a.z, b.x, b.y, b.z]
    }
}

/// A unit tangent ⟂ n from a body axis (rotation-equivariant), falling back to a second axis.
fn tangent(n: Vec3, body: &Mat3) -> Vec3 {
    for axis in [body.col(0), body.col(1)] {
        let t = axis - n * axis.dot(n);
        if t.norm() > 1e-6 {
            return t.normalized();
        }
    }
    let t = body.col(2) - n * body.col(2).dot(n);
    t.normalized()
}

/// Half-thickness of a box along unit n.
fn support(r: &Mat3, half: Vec3, n: Vec3) -> f64 {
    let l = r.transpose() * n;
    l.x.abs() * half.x + l.y.abs() * half.y + l.z.abs() * half.z
}

/// World rotation vector r with R(q1) = exp(r)·R(q0).
fn rotvec_between(q0: Quat, q1: Quat) -> Vec3 {
    (q1 * q0.conjugate()).normalized().to_rotation_vector()
}

/// Time within `[0, dt]` at which two bounding spheres (centres `x`, velocities `v`,
/// radii summing to `r`) first touch: 0 if they already overlap, `None` if not within the
/// frame. Rotation-invariant, unlike the routing sweep's axis-aligned bounds.
fn sphere_contact_time(dx: Vec3, dv: Vec3, r: f64, dt: f64) -> Option<f64> {
    let c = dx.norm2() - r * r;
    if c <= 0.0 {
        return Some(0.0);
    }
    let (a, b) = (dv.norm2(), dx.dot(dv));
    if a <= 0.0 || b >= 0.0 {
        return None;
    }
    let disc = b * b - a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / a;
    (t <= dt).then_some(t.max(0.0))
}

fn ln_ratio(a: f64, b: f64) -> f64 {
    (a.max(TINY) / b.max(TINY)).ln()
}

/// A built array.
#[derive(Clone, Debug, PartialEq)]
pub enum Arr {
    F32(Vec<usize>, Vec<f32>),
    I32(Vec<usize>, Vec<i32>),
    I64(Vec<usize>, Vec<i64>),
    F64(Vec<usize>, Vec<f64>),
}

fn f32a(shape: &[usize], d: Vec<f64>) -> Arr {
    Arr::F32(shape.to_vec(), d.into_iter().map(|x| x as f32).collect())
}

/// One call.
#[derive(Clone, Debug)]
pub struct Call {
    pub frame: usize,
    pub prev_call: Option<usize>,
    pub phase: String,
    pub transition: bool,
    pub anchored: bool,
    pub counts: BTreeMap<String, usize>,
    pub scales: BTreeMap<String, f64>,
    pub energy: BTreeMap<String, f64>,
    pub arrays: Vec<(String, Arr)>,
}

impl Call {
    pub fn array(&self, n: &str) -> Option<&Arr> {
        self.arrays.iter().find(|(k, _)| k == n).map(|(_, a)| a)
    }
}

/// Per-event static data the builder needs, from the record and its scene.
struct Statics {
    n: usize,
    e: usize,
    ni: usize,
    mass: Vec<f64>,
    /// Each chunk's inertia about its centre in its structure's body frame (`ChunkData.inertia`):
    /// the solver's world inertia is R_body I R_body^T with R_body its cluster's rotation.
    inertia: Vec<Mat3>,
    half: Vec<Vec3>,
    center: Vec<Vec3>,
    rest: Vec<Mat3>,
    structure: Vec<usize>,
    ends: Vec<(usize, usize)>,
    geom: Vec<[f64; 13]>,
    stiff: Vec<[f64; 5]>,
    strength: Vec<[f64; 8]>,
    weibull: Vec<f64>,
    rebar: Vec<bool>,
    ckey: Vec<i64>,
    bkey: Vec<i64>,
    e_mod: Vec<f64>,
    rho: Vec<f64>,
    tensile: Vec<f64>,
    gf_t: Vec<f64>,
    fric: Vec<f64>,
    fixed: Vec<bool>,
    pinned: Vec<bool>,
    imp_mass: Vec<f64>,
    imp_e: Vec<f64>,
    imp_fric: Vec<f64>,
    imp_r: Vec<f64>,
    imp_inertia: Vec<Vec3>,
    imp_half: Vec<Vec3>,
    imp_crush: Vec<bool>,
    dt: f64,
    gravity: Vec3,
    restitution: f64,
    contact_friction: Option<f64>,
    bond_pairs: HashSet<(usize, usize)>,
}

impl Statics {
    fn of(r: &Record) -> Result<Statics, String> {
        let sc = &r.scene;
        let n = r.ctx["n_chunks"].as_u64().ok_or("n_chunks")? as usize;
        let e = r.ctx["n_bonds"].as_u64().ok_or("n_bonds")? as usize;
        let ni = r.ctx["n_impactors"].as_u64().ok_or("n_impactors")? as usize;
        let scale = sc.sim.stiffness_scale;
        let m = r.rf("static_chunk_mass", 0)?;
        // The full matrix from recorder revision 9; earlier records hold its diagonal (exact
        // for every chunk they hold: all rest rotations are the identity).
        let inertia: Vec<Mat3> = match r.f.get("static_chunk_inertia") {
            Some(_) => {
                let x = r.rf("static_chunk_inertia", 0)?;
                (0..n).map(|k| sym6(&x[6 * k..6 * k + 6])).collect()
            }
            None => {
                let d = r.rf("static_chunk_inertia_diag", 0)?;
                (0..n).map(|k| Mat3::diag(v3(d, k))).collect()
            }
        };
        let hf = r.rf("static_chunk_half_extents", 0)?;
        let cl = r.rf("static_chunk_center_local", 0)?;
        let st = r.ri("static_chunk_structure", 0)?;
        let en = r.ri("static_bond_endpoints", 0)?;
        let ge = r.rf("static_bond_geometry", 0)?;
        let sf = r.rf("static_bond_stiffness", 0)?;
        let sg = r.rf("static_bond_strength", 0)?;
        let wb = r.rf("static_bond_weibull", 0)?;
        let chunk_desc: Vec<_> = sc.bodies.iter().flat_map(|b| b.chunks.iter()).collect();
        let bond_desc: Vec<_> = sc.bodies.iter().flat_map(|b| b.bonds.iter()).collect();
        if chunk_desc.len() != n || bond_desc.len() != e {
            return Err(format!("{}: scene has {} chunks / {} bonds, record {n} / {e}", r.name, chunk_desc.len(), bond_desc.len()));
        }
        let mat = |name: &str| sc.material(name).clone();
        let ck = r.ctx["chunk_keys"].as_array().ok_or("chunk_keys")?;
        let bk = r.ctx["bond_keys"].as_array().ok_or("bond_keys")?;
        let ckey: Vec<i64> = ck.iter().map(|x| keys::chunk_key(x[0].as_u64().unwrap() as usize, x[1].as_u64().unwrap() as usize)).collect();
        let bkey: Vec<i64> = bk.iter().map(|x| keys::bond_key(x[0].as_u64().unwrap() as usize, x[1].as_u64().unwrap() as usize)).collect();
        let mut imp = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for d in &sc.impactors {
            let im = mat(&d.material);
            let (r_face, inertia, half) = match d.shape {
                ImpactorShape::Sphere { radius } => (radius, Vec3::splat(0.4 * d.mass * radius * radius), Vec3::splat(radius)),
                ImpactorShape::Box { half_extents } => {
                    let h = Vec3::from_array(half_extents);
                    let (a, b, c) = (2.0 * h.x, 2.0 * h.y, 2.0 * h.z);
                    (h.x.min(h.y).min(h.z), Vec3::new(b * b + c * c, a * a + c * c, a * a + b * b) * (d.mass / 12.0), h)
                }
            };
            imp.0.push(d.mass);
            imp.1.push(im.youngs_modulus * scale);
            imp.2.push(im.friction);
            imp.3.push(r_face);
            imp.4.push(inertia);
            imp.5.push(half);
            imp.6.push(d.crush.is_some());
        }
        let mut bond_pairs = HashSet::new();
        let ends: Vec<(usize, usize)> = (0..e).map(|k| (en[2 * k] as usize, en[2 * k + 1] as usize)).collect();
        for &(a, b) in &ends {
            bond_pairs.insert((a.min(b), a.max(b)));
        }
        Ok(Statics {
            n,
            e,
            ni,
            mass: m.iter().map(|&x| x as f64).collect(),
            inertia,
            half: (0..n).map(|k| v3(hf, k)).collect(),
            center: (0..n).map(|k| v3(cl, k)).collect(),
            rest: chunk_desc.iter().map(|c| Quat::from_wxyz(c.orientation).to_mat3()).collect(),
            structure: st.iter().map(|&x| x as usize).collect(),
            ends,
            geom: (0..e).map(|k| std::array::from_fn(|j| ge[13 * k + j] as f64)).collect(),
            stiff: (0..e).map(|k| std::array::from_fn(|j| sf[5 * k + j] as f64)).collect(),
            strength: (0..e).map(|k| std::array::from_fn(|j| sg[8 * k + j] as f64)).collect(),
            weibull: wb.iter().map(|&x| x as f64).collect(),
            rebar: bond_desc.iter().map(|b| b.rebar.is_some()).collect(),
            ckey,
            bkey,
            e_mod: chunk_desc.iter().map(|c| mat(&c.material).youngs_modulus * scale).collect(),
            rho: chunk_desc.iter().map(|c| mat(&c.material).density).collect(),
            tensile: chunk_desc.iter().map(|c| mat(&c.material).tensile_strength).collect(),
            gf_t: chunk_desc.iter().map(|c| mat(&c.material).fracture_energy.tension).collect(),
            fric: chunk_desc.iter().map(|c| mat(&c.material).friction).collect(),
            fixed: chunk_desc.iter().map(|c| c.support == Support::Fixed).collect(),
            pinned: chunk_desc.iter().map(|c| c.support == Support::Pinned).collect(),
            imp_mass: imp.0,
            imp_e: imp.1,
            imp_fric: imp.2,
            imp_r: imp.3,
            imp_inertia: imp.4,
            imp_half: imp.5,
            imp_crush: imp.6,
            dt: sc.sim.frame_dt,
            gravity: Vec3::from_array(sc.gravity),
            restitution: sc.sim.contact_restitution,
            contact_friction: sc.sim.contact_friction,
            bond_pairs,
        })
    }

    fn pair_friction(&self, a: f64, b: f64) -> f64 {
        self.contact_friction.unwrap_or(a.min(b))
    }
}

fn median(mut x: Vec<f64>) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    x.sort_by(|a, b| a.total_cmp(b));
    let k = x.len() / 2;
    if x.len() % 2 == 1 { x[k] } else { 0.5 * (x[k - 1] + x[k]) }
}

fn bfs(n: usize, adj: &[Vec<usize>], sources: &[usize]) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; n];
    let mut q = VecDeque::new();
    for &s in sources {
        if d[s] != 0.0 {
            d[s] = 0.0;
            q.push_back(s);
        }
    }
    while let Some(u) = q.pop_front() {
        for &w in &adj[u] {
            if d[w].is_infinite() {
                d[w] = d[u] + 1.0;
                q.push_back(w);
            }
        }
    }
    d
}

/// Multi-source shortest arrival times over weighted edges (Dijkstra, O(n²): islands are small).
fn arrival(n: usize, adj: &[Vec<(usize, f64)>], init: &[(usize, f64)]) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; n];
    for &(s, t) in init {
        d[s] = d[s].min(t);
    }
    let mut done = vec![false; n];
    loop {
        let mut best = None;
        for u in 0..n {
            if !done[u] && d[u].is_finite() && best.map_or(true, |b: usize| d[u] < d[b]) {
                best = Some(u);
            }
        }
        let Some(u) = best else { break };
        done[u] = true;
        for &(w, c) in &adj[u] {
            if d[u] + c < d[w] {
                d[w] = d[u] + c;
            }
        }
    }
    d
}

fn cap4(x: f64) -> f64 {
    if x.is_finite() { x.min(4.0) } else { 4.0 }
}

/// Inputs-only or with targets.
pub struct BuildOptions {
    pub targets: bool,
}

/// Each island chunk's hidden state at frame boundary `b` (calls /4, P0's definitions), in
/// its own body's frame (its cluster: the fragment): u and theta exactly the solver's
/// `ChunkState` (`state_chunk_hidden`), and the hidden rates about the body's
/// momentum-exact rigid fit (COM, v = P/M, omega = I_w^-1 L; zero for an anchored body):
/// v_def = R^-1 (v - v_body - omega_body x r), omega_def = R^-1 (w - omega_body).
fn hidden_state(r: &Record, st: &Statics, b: usize, isl: &[usize]) -> Result<Vec<[f64; 12]>, String> {
    let pos = r.rf("state_chunk_pos", b)?;
    let vel = r.rf("state_chunk_vel", b)?;
    let ang = r.rf("state_chunk_angvel", b)?;
    let qc = r.rf("state_chunk_cluster_quat", b)?;
    let hid = r.rf("state_chunk_hidden", b)?;
    let cl = r.ri("state_chunk_cluster", b)?;
    let anch = r.ri("state_chunk_anchored", b)?;
    // Bodies among the island's chunks (a body lies wholly in one island).
    let mut bodies: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for &c in isl {
        bodies.entry(cl[c]).or_default().push(c);
    }
    let mut fit: HashMap<i32, (Vec3, Vec3, Vec3)> = HashMap::new();
    for (&id, members) in &bodies {
        let anchored = members.iter().any(|&c| anch[c] == 1);
        let m: f64 = members.iter().map(|&c| st.mass[c]).sum();
        let mut x = Vec3::ZERO;
        let mut p = Vec3::ZERO;
        for &c in members {
            x += v3(pos, c) * st.mass[c];
            p += v3(vel, c) * st.mass[c];
        }
        let x = x * (1.0 / m);
        if anchored {
            fit.insert(id, (x, Vec3::ZERO, Vec3::ZERO));
            continue;
        }
        let (mut iw, mut l) = (Mat3::diag(Vec3::ZERO), Vec3::ZERO);
        for &c in members {
            let rc = v3(pos, c) - x;
            let rot = quat(qc, c).to_mat3();
            let ic = rot * st.inertia[c] * rot.transpose();
            iw = iw + ic + (Mat3::diag(Vec3::splat(rc.norm2())) - Mat3::outer(rc, rc)) * st.mass[c];
            l += rc.cross(v3(vel, c) * st.mass[c]) + ic * v3(ang, c);
        }
        let w = iw.inverse().map_or(Vec3::ZERO, |inv| inv * l);
        fit.insert(id, (x, p * (1.0 / m), w));
    }
    Ok(isl
        .iter()
        .map(|&c| {
            let (x, vb, wb) = fit[&cl[c]];
            let rt = quat(qc, c).to_mat3().transpose();
            let r_c = v3(pos, c) - x;
            let vd = rt * (v3(vel, c) - vb - wb.cross(r_c));
            let wd = rt * (v3(ang, c) - wb);
            let h = |k: usize| hid[6 * c + k] as f64;
            [h(0), h(1), h(2), h(3), h(4), h(5), vd.x, vd.y, vd.z, wd.x, wd.y, wd.z]
        })
        .collect())
}

/// Each island chunk's deformation scale u_c (calls /4): the mean over its live bonds of the
/// tensile failure elongation f_t A w / k_n, floored at 1e-9 m; a chunk with no live bond
/// takes the island median.
fn deformation_scale(st: &Statics, conn: &[i32], isl: &[usize]) -> Vec<f64> {
    let index: HashMap<usize, usize> = isl.iter().enumerate().map(|(j, &c)| (c, j)).collect();
    let (mut sum, mut n) = (vec![0.0; isl.len()], vec![0usize; isl.len()]);
    for k in 0..st.e {
        if conn[k] != 1 {
            continue;
        }
        let e = (st.strength[k][0] * st.geom[k][9] * st.weibull[k] / st.stiff[k][0].max(TINY)).max(1e-9);
        for c in [st.ends[k].0, st.ends[k].1] {
            if let Some(&j) = index.get(&c) {
                sum[j] += e;
                n[j] += 1;
            }
        }
    }
    let have: Vec<f64> = (0..isl.len()).filter(|&j| n[j] > 0).map(|j| sum[j] / n[j] as f64).collect();
    let fallback = if have.is_empty() { 1e-9 } else { median(have) };
    (0..isl.len()).map(|j| if n[j] > 0 { (sum[j] / n[j] as f64).max(1e-9) } else { fallback }).collect()
}

/// The call of frame `f` (None when the routing put no chunk in an island).
fn build_call(r: &Record, st: &Statics, f: usize, prev_call: Option<usize>, history: &History, opts: &BuildOptions) -> Result<Option<Call>, String> {
    let (n, e) = (st.n, st.e);
    let routed = r.ri("frame_chunk_routed", f)?;
    let imp_routed = r.ri("frame_impactor_routed", f)?;
    let active = r.ri("state_chunk_active", f)?;
    let isl: Vec<usize> = (0..n).filter(|&c| routed[c] == 1 && active[c] == 1).collect();
    if isl.is_empty() {
        return Ok(None);
    }
    let iimp: Vec<usize> = (0..st.ni).filter(|&k| imp_routed[k] == 1).collect();
    let (nc, ni) = (isl.len(), iimp.len());
    let node_of: HashMap<usize, usize> = isl.iter().enumerate().map(|(j, &c)| (c, j)).collect();
    let pos = r.rf("state_chunk_pos", f)?;
    let vel = r.rf("state_chunk_vel", f)?;
    let ang = r.rf("state_chunk_angvel", f)?;
    let qw = r.rf("state_chunk_q_world", f)?;
    let conn = r.ri("state_bond_connected", f)?;
    let joint = r.rf("state_bond_joint", f)?;
    let margin = r.rf("state_bond_fail_margin", f)?;
    let impst = r.rf("state_impactor", f)?;
    let plug = r.ri("frame_chunk_punch_plug", f)?;
    let prior = r.rf("frame_bond_qs_prior_util", f)?;
    let g = st.gravity;
    let dt = st.dt;
    // node state
    let xc: Vec<Vec3> = isl.iter().map(|&c| v3(pos, c)).collect();
    let vc: Vec<Vec3> = isl.iter().map(|&c| v3(vel, c)).collect();
    let wc: Vec<Vec3> = isl.iter().map(|&c| v3(ang, c)).collect();
    let qc: Vec<Quat> = isl.iter().map(|&c| quat(qw, c)).collect();
    let rb: Vec<Mat3> = qc.iter().map(|q| q.to_mat3()).collect();
    // Each chunk's body (cluster) rotation: what its structure-frame inertia turns with.
    let qcl = r.rf("state_chunk_cluster_quat", f)?;
    let rcl: Vec<Mat3> = isl.iter().map(|&c| quat(qcl, c).to_mat3()).collect();
    // A chunk's world rotation without its rest rotation: what maps structure-local vectors.
    let rs: Vec<Mat3> = isl.iter().zip(&rb).map(|(&c, m)| *m * st.rest[c].transpose()).collect();
    let imp = |k: usize, o: usize| Vec3::new(impst[13 * k + o] as f64, impst[13 * k + o + 1] as f64, impst[13 * k + o + 2] as f64);
    let xi: Vec<Vec3> = iimp.iter().map(|&k| imp(k, 0)).collect();
    let qi: Vec<Quat> = iimp.iter().map(|&k| Quat::from_wxyz([impst[13 * k + 3] as f64, impst[13 * k + 4] as f64, impst[13 * k + 5] as f64, impst[13 * k + 6] as f64]).normalized()).collect();
    let ri: Vec<Mat3> = qi.iter().map(|q| q.to_mat3()).collect();
    let vi: Vec<Vec3> = iimp.iter().map(|&k| imp(k, 7)).collect();
    let wi: Vec<Vec3> = iimp.iter().map(|&k| imp(k, 10)).collect();
    let mc: Vec<f64> = isl.iter().map(|&c| st.mass[c]).collect();
    let mi: Vec<f64> = iimp.iter().map(|&k| st.imp_mass[k]).collect();
    let mall: f64 = mc.iter().sum::<f64>() + mi.iter().sum::<f64>();
    let mut com = Vec3::ZERO;
    let mut vcom = Vec3::ZERO;
    for j in 0..nc {
        com += xc[j] * mc[j];
        vcom += vc[j] * mc[j];
    }
    for j in 0..ni {
        com += xi[j] * mi[j];
        vcom += vi[j] * mi[j];
    }
    com = com * (1.0 / mall);
    vcom = vcom * (1.0 / mall);
    let in_isl = |c: usize| node_of.contains_key(&c);
    let live: Vec<usize> = (0..e).filter(|&k| conn[k] == 1 && in_isl(st.ends[k].0) && in_isl(st.ends[k].1)).collect();
    // this frame's predicted contacts within the island
    let crow: Vec<usize> = if r.rows("contact_i32") > 0 {
        (0..r.rows("contact_i32")).filter(|&q| {
            let c = r.ri("contact_i32", q).unwrap();
            c[0] as usize == f && in_isl(c[3] as usize) && iimp.contains(&(c[1] as usize))
        }).collect()
    } else {
        Vec::new()
    };
    let ci = |q: usize| r.ri("contact_i32", q).unwrap();
    let frame_rows: Vec<usize> = (0..r.rows("contact_i32")).filter(|&q| ci(q)[0] as usize == f).collect();
    let cf = |q: usize| r.rf("contact_f32", q).unwrap();
    // scales
    let l = median(isl.iter().map(|&c| 8.0 * st.half[c].x * st.half[c].y * st.half[c].z).collect()).cbrt();
    let m = median(mc.clone());
    // The reference speed (calls /3): the largest of the routing speed, any predicted
    // contact's closing speed, and every island body's speed relative to the island's centre
    // of mass and its rim speed (|w| x half-diagonal), so every velocity feature is O(1).
    let mut v = tiered_impact_speed();
    for &q in &crow {
        v = v.max(cf(q)[7] as f64);
    }
    for j in 0..nc {
        v = v.max((vc[j] - vcom).norm()).max(wc[j].norm() * st.half[isl[j]].norm());
    }
    for j in 0..ni {
        v = v.max((vi[j] - vcom).norm()).max(wi[j].norm() * st.imp_half[iimp[j]].norm());
    }
    let dom = {
        // The island's dominant material: the most massive chunk's.
        let j = (0..nc).max_by(|&a, &b| mc[a].total_cmp(&mc[b])).unwrap();
        isl[j]
    };
    let c_wave = (st.e_mod[dom] / st.rho[dom]).sqrt();
    let f_t = if live.is_empty() { st.tensile[dom] } else { median(live.iter().map(|&k| st.strength[k][0]).collect()) };
    let e_med = median(isl.iter().map(|&c| st.e_mod[c]).collect());
    let ga_b = |k: usize| st.strength[k][5] * st.geom[k][9];
    let ga = if live.is_empty() { st.gf_t[dom] * l * l } else { median(live.iter().map(|&k| ga_b(k)).collect()) };
    let ke = |mass: f64, inertia: Mat3, rot: &Mat3, v: Vec3, w: Vec3| {
        let iw = *rot * inertia * rot.transpose();
        0.5 * mass * v.norm2() + 0.5 * w.dot(iw * w)
    };
    let ke_in: f64 = (0..nc).map(|j| ke(mc[j], st.inertia[isl[j]], &rcl[j], vc[j], wc[j])).sum::<f64>() + (0..ni).map(|j| ke(mi[j], Mat3::diag(st.imp_inertia[iimp[j]]), &ri[j], vi[j], wi[j])).sum::<f64>();
    let ke_imp: f64 = (0..ni).map(|j| ke(mi[j], Mat3::diag(st.imp_inertia[iimp[j]]), &ri[j], vi[j], wi[j])).sum();
    // E_in as a scale is causal: gravity's work estimated ballistically.
    let wg_est: f64 = (0..nc).map(|j| mc[j] * (vc[j] * dt + g * (0.5 * dt * dt)).dot(g)).sum::<f64>() + (0..ni).map(|j| mi[j] * (vi[j] * dt + g * (0.5 * dt * dt)).dot(g)).sum::<f64>();
    let e_in = (ke_in + wg_est).max(0.5 * m * v * v);
    let mut scales = BTreeMap::new();
    for (k, x) in [("L", l), ("m", m), ("v", v), ("c", c_wave), ("f_t", f_t), ("E", e_med), ("GA", ga), ("dt", dt), ("E_in", e_in)] {
        if !(x.is_finite() && x > 0.0) {
            return Err(format!("{} frame {f}: scale {k} = {x}", r.name));
        }
        scales.insert(k.to_string(), x);
    }
    // graph distances over the island's live bonds
    let mut adj = vec![Vec::new(); nc];
    let mut adjw = vec![Vec::new(); nc];
    for &k in &live {
        let (a, b) = (node_of[&st.ends[k].0], node_of[&st.ends[k].1]);
        adj[a].push(b);
        adj[b].push(a);
        let d = (xc[a] - xc[b]).norm() / c_wave;
        adjw[a].push((b, d));
        adjw[b].push((a, d));
    }
    let entries: Vec<usize> = crow.iter().map(|&q| node_of[&(ci(q)[3] as usize)]).collect();
    // Each predicted contact's time within the frame, from bounding spheres (the routing
    // sweep's own time is taken on world axis-aligned bounds, not rotation-invariant).
    let t_contact: Vec<Option<f64>> = crow
        .iter()
        .map(|&q| {
            let c = ci(q);
            let k_imp = iimp.iter().position(|&k| k == c[1] as usize).unwrap();
            let jd = node_of[&(c[3] as usize)];
            let r_imp = st.imp_half[iimp[k_imp]].norm();
            sphere_contact_time(xc[jd] - xi[k_imp], vc[jd] - vi[k_imp], r_imp + st.half[c[3] as usize].norm(), dt)
        })
        .collect();
    let hops_c = bfs(nc, &adj, &entries);
    let arr = arrival(nc, &adjw, &crow.iter().zip(&t_contact).filter_map(|(&q, t)| t.map(|t| (node_of[&(ci(q)[3] as usize)], t))).collect::<Vec<_>>());
    let anchors: Vec<usize> = (0..nc).filter(|&j| st.fixed[isl[j]] || st.pinned[isl[j]]).collect();
    let hops_a = bfs(nc, &adj, &anchors);
    // per chunk bond summaries (over every pristine bond of the chunk)
    let (mut dmax, mut dsum, mut nlive, mut nbroken) = (vec![0.0; nc], vec![0.0; nc], vec![0.0; nc], vec![0.0; nc]);
    for k in 0..e {
        for c in [st.ends[k].0, st.ends[k].1] {
            if let Some(&j) = node_of.get(&c) {
                if conn[k] == 1 {
                    nlive[j] += 1.0;
                    dsum[j] += joint[4 * k] as f64;
                    dmax[j] = f64::max(dmax[j], joint[4 * k] as f64);
                } else {
                    nbroken[j] += 1.0;
                }
            }
        }
    }
    // edges: impactor contacts first (they decide "in contact")
    let plug_bonds: HashSet<usize> = live.iter().copied().filter(|&k| (plug[st.ends[k].0] > 0) != (plug[st.ends[k].1] > 0)).collect();
    let ga_plug: f64 = plug_bonds.iter().map(|&k| ga_b(k)).sum();
    let median_area = if live.is_empty() { l * l } else { median(live.iter().map(|&k| st.geom[k][9]).collect()) };
    let mut ic = EdgeSet::default();
    for (ci_row, &q) in crow.iter().enumerate() {
        let (c, x) = (ci(q), cf(q));
        let k_imp = iimp.iter().position(|&k| k == c[1] as usize).unwrap();
        let ch = c[3] as usize;
        let jd = node_of[&ch];
        let js = nc + k_imp;
        let nrm = Vec3::new(x[4] as f64, x[5] as f64, x[6] as f64).normalized();
        let fr = Frame3::of(tangent(nrm, &ri[k_imp]), nrm);
        let gap = (xc[jd] - xi[k_imp]).dot(nrm) - support(&ri[k_imp], st.imp_half[iimp[k_imp]], nrm) - support(&rb[jd], st.half[ch], nrm);
        let ek = 0.5 * x[8] as f64 * (x[7] as f64).powi(2);
        let ler = if ga_plug > 0.0 && plug[ch] > 0 { ln_ratio(ek, ga_plug) } else { 0.0 };
        let fric = st.pair_friction(st.imp_fric[iimp[k_imp]], st.fric[ch]);
        let t_con = match t_contact[ci_row] {
            Some(t) if t <= 0.0 => -1.0,
            Some(t) => t / dt,
            None => 1.0,
        };
        let ki = iimp[k_imp];
        for (s_, d_, fr, self_) in [(js, jd, fr, 1.0), (jd, js, fr.reversed(), 0.0)] {
            let node_v = |j: usize| if j >= nc { vi[j - nc] } else { vc[j] };
            let node_m = |j: usize| if j >= nc { mi[j - nc] } else { mc[j] };
            let node_r = |j: usize| if j >= nc { ri[j - nc] } else { rb[j] };
            let rel = fr.local(node_v(s_) - node_v(d_));
            let spin = fr.local(wi[k_imp] * (st.imp_r[ki] / v));
            let mut row = vec![rel[2] / v, rel[0] / v, rel[1] / v, rel[2] / c_wave, ln_ratio(node_m(d_), node_m(s_)), ln_ratio(st.imp_r[ki], l), ln_ratio(st.imp_e[ki], st.e_mod[ch])];
            row.extend(spin);
            row.extend([t_con, ler, ln_ratio(x[12] as f64, f_t * median_area), x[11] as f64 / dt, self_, (-gap).max(0.0) / l, fric, st.restitution]);
            row.extend(fr.rot6(&node_r(s_)));
            row.extend(fr.rot6(&node_r(d_)));
            ic.push(s_, d_, row);
        }
        ic.frames.push(fr.rows());
        ic.keys.push(keys::contact_key(keys::impactor_key(ki), st.ckey[ch]));
    }
    // structure contacts and cracked faces from the frame's near pairs
    let cl = r.ri("state_chunk_cluster", f)?;
    let mut sc = EdgeSet::default();
    let mut cr = EdgeSet::default();
    let pairs: Vec<(usize, usize)> = if r.rows("pair_i32") > 0 {
        (0..r.rows("pair_i32")).map(|q| r.ri("pair_i32", q).unwrap()).filter(|p| p[0] as usize == f).map(|p| (p[1] as usize, p[2] as usize)).collect()
    } else {
        Vec::new()
    };
    let mut cracked_pairs: HashSet<(usize, usize)> = HashSet::new();
    for k in 0..e {
        let (a, b) = st.ends[k];
        if conn[k] == 1 || !in_isl(a) || !in_isl(b) {
            continue;
        }
        let (ja, jb) = (node_of[&a], node_of[&b]);
        let mut nrm = rs[ja] * Vec3::new(st.geom[k][3], st.geom[k][4], st.geom[k][5]);
        let mut t1 = rs[ja] * Vec3::new(st.geom[k][6], st.geom[k][7], st.geom[k][8]);
        if nrm.dot(xc[jb] - xc[ja]) < 0.0 {
            nrm = -nrm;
            t1 = -t1;
        }
        // Kept only while the faces are still in or near contact: the same margin as a
        // structure contact along the normal, and overlapping across it.
        let d = xc[jb] - xc[ja];
        let gap = d.dot(nrm) - support(&rb[ja], st.half[a], nrm) - support(&rb[jb], st.half[b], nrm);
        let t2 = nrm.cross(t1);
        let across = |t: Vec3| d.dot(t).abs() <= support(&rb[ja], st.half[a], t) + support(&rb[jb], st.half[b], t);
        if gap > CONTACT_MARGIN * l || !across(t1) || !across(t2) {
            continue;
        }
        cracked_pairs.insert((a.min(b), a.max(b)));
        let p = xc[ja] + rs[ja] * (Vec3::new(st.geom[k][0], st.geom[k][1], st.geom[k][2]) - st.center[a]);
        let fr = Frame3::of(t1, nrm);
        let since = history.frames_since_broken(k, f).map_or(1.0, |x| (x as f64 / 8.0).min(1.0));
        let fric = st.pair_friction(st.fric[a], st.fric[b]);
        for (s_, d_, fr) in [(ja, jb, fr), (jb, ja, fr.reversed())] {
            let vs = vc[s_] + wc[s_].cross(p - xc[s_]);
            let vd = vc[d_] + wc[d_].cross(p - xc[d_]);
            let mut row = vec![ln_ratio(st.geom[k][9], l * l)];
            row.extend(fr.local(p - xc[s_]).map(|x| x / l));
            row.extend([gap / l, fric]);
            row.extend(fr.local(vd - vs).map(|x| x / v));
            row.push(since);
            row.extend(fr.rot6(&rb[s_]));
            row.extend(fr.rot6(&rb[d_]));
            cr.push(s_, d_, row);
        }
        cr.frames.push(fr.rows());
        cr.keys.push(st.bkey[k]);
    }
    for &(a, b) in &pairs {
        if !in_isl(a) || !in_isl(b) || cl[a] == cl[b] || st.bond_pairs.contains(&(a.min(b), a.max(b))) {
            continue;
        }
        let (ja, jb) = (node_of[&a], node_of[&b]);
        let d = xc[jb] - xc[ja];
        if d.norm() < 1e-12 {
            continue;
        }
        let nrm = d.normalized();
        let gap = d.norm() - support(&rb[ja], st.half[a], nrm) - support(&rb[jb], st.half[b], nrm);
        if gap > CONTACT_MARGIN * l {
            continue;
        }
        let fr = Frame3::of(tangent(nrm, &rb[ja]), nrm);
        let fric = st.pair_friction(st.fric[a], st.fric[b]);
        let same = f64::from(u8::from(st.structure[a] == st.structure[b]));
        for (s_, d_, fr) in [(ja, jb, fr), (jb, ja, fr.reversed())] {
            let rel = fr.local(vc[s_] - vc[d_]);
            let t_con = if gap <= 0.0 { -1.0 } else if rel[2] > 0.0 && gap / rel[2] < dt { gap / rel[2] / dt } else { 1.0 };
            let mut row = vec![rel[2] / v, rel[0] / v, rel[1] / v, rel[2] / c_wave, ln_ratio(mc[d_], mc[s_]), t_con, (-gap).max(0.0) / l, fric, 0.0, same];
            row.extend(fr.rot6(&rb[s_]));
            row.extend(fr.rot6(&rb[d_]));
            sc.push(s_, d_, row);
        }
        sc.frames.push(fr.rows());
        sc.keys.push(keys::contact_key(st.ckey[a], st.ckey[b]));
    }
    let _ = cracked_pairs;
    let mut in_contact = vec![false; nc + ni];
    for es in [&ic, &sc, &cr] {
        for &s_ in &es.src {
            in_contact[s_] = true;
        }
    }
    // the hidden deformation (calls /4)
    let def = hidden_state(r, st, f, &isl)?;
    let uc = deformation_scale(st, conn, &isl);
    // chunk features
    // Principal moments, descending: invariant to the chunk's rest rotation.
    let pm: HashMap<usize, [f64; 3]> = isl.iter().map(|&c| (c, stress_ref::snapshot::symmetric_eigenvalues(&st.inertia[c]))).collect();
    let mut chunk_feat = Vec::with_capacity(nc * 45);
    for j in 0..nc {
        let c = isl[j];
        let bt = rb[j].transpose();
        let gb = bt * g * (1.0 / G0);
        let vr = bt * (vc[j] - vcom) * (1.0 / v);
        let wb = bt * wc[j] * (l / v);
        let row = [
            ln_ratio(mc[j], m),
            pm[&c][0] / (mc[j] * l * l), pm[&c][1] / (mc[j] * l * l), pm[&c][2] / (mc[j] * l * l),
            st.half[c].x / l, st.half[c].y / l, st.half[c].z / l,
            f64::from(u8::from(st.fixed[c])), f64::from(u8::from(st.pinned[c])),
            if nlive[j] > 0.0 { dsum[j] / nlive[j] } else { 0.0 }, dmax[j], nlive[j] / 6.0, nbroken[j] / 6.0,
            cap4(hops_c[j] / 16.0), cap4(arr[j] / dt), cap4(hops_a[j] / 16.0),
            gb.x, gb.y, gb.z, vr.x, vr.y, vr.z, wb.x, wb.y, wb.z,
            f64::from(u8::from(plug[c] > 0)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            f64::from(u8::from(in_contact[j])),
        ];
        chunk_feat.extend(row);
        let (d, ri) = (&def[j], st.half[c].norm());
        chunk_feat.extend([d[0] / uc[j], d[1] / uc[j], d[2] / uc[j]]);
        chunk_feat.extend([d[3] * ri / uc[j], d[4] * ri / uc[j], d[5] * ri / uc[j]]);
        chunk_feat.extend([d[6] / v, d[7] / v, d[8] / v]);
        chunk_feat.extend([d[9] * ri / v, d[10] * ri / v, d[11] * ri / v]);
    }
    let mut imp_feat = Vec::with_capacity(ni * 16);
    let m_island: f64 = mc.iter().sum();
    for j in 0..ni {
        let k = iimp[j];
        let bt = ri[j].transpose();
        let gb = bt * g * (1.0 / G0);
        let vr = bt * (vi[j] - vcom);
        let r2 = st.imp_r[k] * st.imp_r[k];
        imp_feat.extend([
            ln_ratio(mi[j], m_island), ln_ratio(mi[j], m),
            st.imp_inertia[k].x / (mi[j] * r2), st.imp_inertia[k].y / (mi[j] * r2), st.imp_inertia[k].z / (mi[j] * r2),
            ln_ratio(st.imp_r[k], l), ln_ratio(st.imp_e[k], e_med), f64::from(u8::from(st.imp_crush[k])),
            gb.x, gb.y, gb.z, vr.x / v, vr.y / v, vr.z / v, (vi[j] - vcom).norm() / c_wave,
            f64::from(u8::from(in_contact[nc + j])),
        ]);
    }
    let anchored = !anchors.is_empty();
    let ga_live: f64 = live.iter().map(|&k| ga_b(k)).sum();
    let glob = vec![
        if ke_imp > 0.0 && ga_live > 0.0 { ln_ratio(ke_imp, ga_live) } else { 0.0 },
        if ke_imp > 0.0 && ga_plug > 0.0 { ln_ratio(ke_imp, ga_plug) } else { 0.0 },
        (dt * c_wave / l).ln(),
        dt * v / l,
        ni as f64 / 4.0,
        (nc as f64).ln() / 10.0,
        g.norm() / G0,
        f64::from(u8::from(anchored)),
        0.0, 0.0, 0.0,
        v / c_wave,
    ];
    // bond edges
    // The previous frame's peak utilization: the record's row f - 1, or a live record's own.
    let peak_prev = if r.has("live_bond_peak_prev") {
        Some(r.rf("live_bond_peak_prev", 0)?)
    } else if r.has("frame_bond_peak_util_prev") {
        Some(r.rf("frame_bond_peak_util_prev", f)?)
    } else if f > 0 {
        Some(r.rf("frame_bond_peak_util", f - 1)?)
    } else {
        None
    };
    let mut be = EdgeSet::default();
    for &k in &live {
        let (a, b) = st.ends[k];
        let (ja, jb) = (node_of[&a], node_of[&b]);
        let gk = &st.geom[k];
        let mut nrm = rs[ja] * Vec3::new(gk[3], gk[4], gk[5]);
        let mut t1 = rs[ja] * Vec3::new(gk[6], gk[7], gk[8]);
        if nrm.dot(xc[jb] - xc[ja]) < 0.0 {
            nrm = -nrm;
            t1 = -t1;
        }
        let fr = Frame3::of(t1, nrm);
        let p = xc[ja] + rs[ja] * (Vec3::new(gk[0], gk[1], gk[2]) - st.center[a]);
        let area = gk[9];
        let (kl, ka) = (e_med * area / l, e_med * area * l);
        let [kn, ks, kb1, kb2, kt] = st.stiff[k];
        let sk = &st.strength[k];
        let pr = prior[k] as f64;
        let in_plug = plug_bonds.contains(&k);
        // The plug's id is its contact's index among this frame's contacts (all of them).
        let ratio = if in_plug {
            let pid = plug[a].max(plug[b]) as usize;
            match frame_rows.get(pid.wrapping_sub(1)) {
                Some(&q) => ln_ratio(cf(q)[13] as f64, 1.0),
                None => return Err(format!("{} frame {f}: plug {pid} has no contact row", r.name)),
            }
        } else {
            0.0
        };
        let common = [
            st.weibull[k], joint[4 * k] as f64, joint[4 * k + 1] as f64, margin[k] as f64,
            (joint[4 * k + 3] as f64).max(0.0).ln_1p(), peak_prev.map_or(0.0, |p| (p[k] as f64).max(0.0).ln_1p()),
            f64::from(u8::from(st.rebar[k])), if pr >= 0.0 { pr.max(TINY).ln() } else { 0.0 },
            f64::from(u8::from(in_plug)), ratio,
        ];
        for (s_, d_, fr) in [(ja, jb, fr), (jb, ja, fr.reversed())] {
            let vs = vc[s_] + wc[s_].cross(p - xc[s_]);
            let vd = vc[d_] + wc[d_].cross(p - xc[d_]);
            let mut row = vec![ln_ratio(area, l * l)];
            row.extend(fr.local(p - xc[s_]).map(|x| x / l));
            row.extend([ln_ratio(kn, kl), ln_ratio(ks, kl), ln_ratio(ks, kl), ln_ratio(kb1, ka), ln_ratio(kb2, ka), ln_ratio(kt, ka)]);
            row.extend([ln_ratio(sk[0], f_t), ln_ratio(sk[2], f_t), ln_ratio(sk[1], f_t)]);
            row.extend((0..3).map(|i| ln_ratio(sk[5 + i] * area, ga)));
            row.extend(common);
            row.extend(fr.local(vd - vs).map(|x| x / v));
            row.extend(fr.rot6(&rb[s_]));
            row.extend(fr.rot6(&rb[d_]));
            be.push(s_, d_, row);
        }
        be.frames.push(fr.rows());
        be.keys.push(st.bkey[k]);
    }
    let nb = live.len();
    let pos_rel = |x: &[Vec3]| -> Vec<f64> { x.iter().flat_map(|p| (*p - com).to_array()).collect() };
    let flat = |x: &[Vec3]| -> Vec<f64> { x.iter().flat_map(|p| p.to_array()).collect() };
    let mut a: Vec<(String, Arr)> = Vec::new();
    let mut put = |k: &str, x: Arr| a.push((k.to_string(), x));
    put("chunk_key", Arr::I64(vec![nc], isl.iter().map(|&c| st.ckey[c]).collect()));
    put("chunk_feat", f32a(&[nc, 45], chunk_feat));
    put("chunk_def", f32a(&[nc, 12], def.iter().flatten().copied().collect()));
    put("chunk_def_uc", f32a(&[nc], uc.clone()));
    // Each chunk's body (its cluster) at the frame start, numbered within the call in order
    // of first appearance, and that body's world rotation (wxyz, as chunk_q_world).
    {
        let (cl0, q0) = (r.ri("state_chunk_cluster", f)?, r.rf("state_chunk_cluster_quat", f)?);
        let mut ids: HashMap<i32, i32> = HashMap::new();
        let body: Vec<i32> = isl.iter().map(|&c| { let k = ids.len() as i32; *ids.entry(cl0[c]).or_insert(k) }).collect();
        put("chunk_body", Arr::I32(vec![nc], body));
        put("chunk_body_q", f32a(&[nc, 4], isl.iter().flat_map(|&c| quat(q0, c).to_wxyz()).collect()));
    }
    put("chunk_mass", f32a(&[nc], mc.clone()));
    // The diagonal of the body-frame inertia (not principal for a chunk with a rest rotation),
    // kept for old readers; chunk_inertia_struct is the full matrix (xx, yy, zz, xy, xz, yz)
    // in the structure's body frame: I_world = R_body I R_body^T with chunk_body_q.
    put("chunk_inertia_body", f32a(&[nc, 3], isl.iter().flat_map(|&c| { let m = &st.inertia[c].m; [m[0][0], m[1][1], m[2][2]] }).collect()));
    put("chunk_inertia_struct", f32a(&[nc, 6], isl.iter().flat_map(|&c| { let m = &st.inertia[c].m; [m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2]] }).collect()));
    put("chunk_q_world", f32a(&[nc, 4], qc.iter().flat_map(|q| q.to_wxyz()).collect()));
    put("chunk_pos", f32a(&[nc, 3], pos_rel(&xc)));
    put("chunk_v", f32a(&[nc, 3], flat(&vc)));
    put("chunk_w", f32a(&[nc, 3], flat(&wc)));
    put("chunk_anchored", Arr::I32(vec![nc], isl.iter().map(|&c| i32::from(st.fixed[c] || st.pinned[c])).collect()));
    put("impactor_key", Arr::I64(vec![ni], iimp.iter().map(|&k| keys::impactor_key(k)).collect()));
    put("impactor_feat", f32a(&[ni, 16], imp_feat));
    put("impactor_mass", f32a(&[ni], mi.clone()));
    put("impactor_inertia_body", f32a(&[ni, 3], flat(&iimp.iter().map(|&k| st.imp_inertia[k]).collect::<Vec<_>>())));
    put("impactor_q_world", f32a(&[ni, 4], qi.iter().flat_map(|q| q.to_wxyz()).collect()));
    put("impactor_pos", f32a(&[ni, 3], pos_rel(&xi)));
    put("impactor_v", f32a(&[ni, 3], flat(&vi)));
    put("impactor_w", f32a(&[ni, 3], flat(&wi)));
    put("global_feat", f32a(&[1, 12], glob));
    for (t, es, w) in [("bond", &be, 41usize), ("cracked", &cr, 22), ("impactor_contact", &ic, 30), ("structure_contact", &sc, 22)] {
        let k = es.keys.len();
        put(&format!("{t}_src"), Arr::I32(vec![2 * k], es.src.iter().map(|&x| x as i32).collect()));
        put(&format!("{t}_dst"), Arr::I32(vec![2 * k], es.dst.iter().map(|&x| x as i32).collect()));
        put(&format!("{t}_key"), Arr::I64(vec![k], es.keys.clone()));
        if es.feat.len() != 2 * k * w {
            return Err(format!("{t}: {} feature values for {k} edges of width {w}", es.feat.len()));
        }
        put(&format!("{t}_feat"), f32a(&[2 * k, w], es.feat.clone()));
        put(&format!("{t}_frame"), Arr::F32(vec![k, 3, 3], es.frames.iter().flatten().copied().collect()));
    }
    // variant-B groups and baselines
    let has_qs = r.has("qs_bond_broken") && r.row("qs_bond_broken", f).is_ok_and(|row| row < r.rows("qs_bond_broken"));
    let (mut cqs, mut iqs, mut bqs) = (vec![0.0; nc * 6], vec![0.0; ni * 6], vec![0.0; 2 * nb * 3]);
    if has_qs {
        let (qv, qw_, qi_, qb, qp) = (r.rf("qs_chunk_vel", f)?, r.rf("qs_chunk_angvel", f)?, r.rf("qs_impactor", f)?, r.ri("qs_bond_broken", f)?, r.rf("qs_bond_peak_util", f)?);
        for j in 0..nc {
            let bt = rb[j].transpose();
            let dv = bt * (v3(qv, isl[j]) - vc[j]) * (1.0 / v);
            let dw = bt * (v3(qw_, isl[j]) - wc[j]) * (l / v);
            cqs[6 * j..6 * j + 6].copy_from_slice(&[dv.x, dv.y, dv.z, dw.x, dw.y, dw.z]);
        }
        for j in 0..ni {
            let k = iimp[j];
            let bt = ri[j].transpose();
            let at = |o: usize| Vec3::new(qi_[13 * k + o] as f64, qi_[13 * k + o + 1] as f64, qi_[13 * k + o + 2] as f64);
            let dv = bt * (at(7) - vi[j]) * (1.0 / v);
            let dw = bt * (at(10) - wi[j]) * (l / v);
            iqs[6 * j..6 * j + 6].copy_from_slice(&[dv.x, dv.y, dv.z, dw.x, dw.y, dw.z]);
        }
        for (j, &k) in live.iter().enumerate() {
            // The QS branch records its breaks, not their time within the frame.
            let row = [(qp[k] as f64).max(TINY).ln(), qb[k] as f64, 0.0];
            bqs[6 * j..6 * j + 3].copy_from_slice(&row);
            bqs[6 * j + 3..6 * j + 6].copy_from_slice(&row);
        }
        if opts.targets {
            put("base_qs_bond_broke", Arr::I32(vec![nb], live.iter().map(|&k| qb[k]).collect()));
            put("base_qs_bond_peak_util", f32a(&[nb], live.iter().map(|&k| qp[k] as f64).collect()));
            put("base_qs_chunk_v_end", f32a(&[nc, 3], isl.iter().flat_map(|&c| v3(qv, c).to_array()).collect()));
            put("base_qs_chunk_w_end", f32a(&[nc, 3], isl.iter().flat_map(|&c| v3(qw_, c).to_array()).collect()));
            put("base_qs_impactor_v_end", f32a(&[ni, 3], iimp.iter().flat_map(|&k| [qi_[13 * k + 7] as f64, qi_[13 * k + 8] as f64, qi_[13 * k + 9] as f64]).collect()));
        }
    }
    put("chunk_qs_feat", f32a(&[nc, 6], cqs));
    put("impactor_qs_feat", f32a(&[ni, 6], iqs));
    put("bond_qs_feat", f32a(&[2 * nb, 3], bqs));
    let mut energy = BTreeMap::new();
    if opts.targets {
        if let Some(pi) = history.punch_row(f) {
            let (pv, pw, pim, pb, pp) = (r.rf("punch_chunk_vel", pi)?, r.rf("punch_chunk_angvel", pi)?, r.rf("punch_impactor", pi)?, r.ri("punch_bond_broken", pi)?, r.rf("punch_bond_peak_util", pi)?);
            put("base_punch_bond_broke", Arr::I32(vec![nb], live.iter().map(|&k| pb[k]).collect()));
            put("base_punch_bond_peak_util", f32a(&[nb], live.iter().map(|&k| pp[k] as f64).collect()));
            put("base_punch_chunk_v_end", f32a(&[nc, 3], isl.iter().flat_map(|&c| v3(pv, c).to_array()).collect()));
            put("base_punch_chunk_w_end", f32a(&[nc, 3], isl.iter().flat_map(|&c| v3(pw, c).to_array()).collect()));
            put("base_punch_impactor_v_end", f32a(&[ni, 3], iimp.iter().flat_map(|&k| [pim[13 * k + 7] as f64, pim[13 * k + 8] as f64, pim[13 * k + 9] as f64]).collect()));
        }
        // targets from the frame-end state
        let bins = r.ri("frame_bond_break_bin", f)?;
        let peak = r.rf("frame_bond_peak_util", f)?;
        let conn1 = r.ri("state_bond_connected", f + 1)?;
        let joint1 = r.rf("state_bond_joint", f + 1)?;
        let margin1 = r.rf("state_bond_fail_margin", f + 1)?;
        let mode1 = r.ri("state_bond_mode", f + 1)?;
        let en0 = r.rf("state_bond_energy", f)?;
        let en1 = r.rf("state_bond_energy", f + 1)?;
        let dis: Vec<[f64; 2]> = if r.has("frame_bond_dissipated") {
            let d = r.rf("frame_bond_dissipated", f)?;
            (0..e).map(|k| [d[2 * k] as f64, d[2 * k + 1] as f64]).collect()
        } else {
            (0..e).map(|k| [(en1[3 * k] - en0[3 * k]) as f64, (en1[3 * k + 1] - en0[3 * k + 1]) as f64]).collect()
        };
        let mut broke = Vec::with_capacity(nb);
        for &k in &live {
            let b = bins[k] >= 0;
            // The rule the break target rests on: the margin reaches 1 exactly when the bond breaks.
            if (margin1[k] >= 1.0) != b || b != (conn1[k] == 0) {
                return Err(format!("{} frame {f} bond {}: margin {} broke {b} connected {}", r.name, st.bkey[k], margin1[k], conn1[k]));
            }
            broke.push(b);
        }
        put("y_bond_peak_util", f32a(&[nb], live.iter().map(|&k| peak[k] as f64).collect()));
        put("y_bond_fail_margin", f32a(&[nb], live.iter().map(|&k| margin1[k] as f64).collect()));
        put("y_bond_broke", Arr::I32(vec![nb], broke.iter().map(|&b| i32::from(b)).collect()));
        put("y_bond_break_bin", Arr::I32(vec![nb], live.iter().map(|&k| if bins[k] >= 0 { bins[k] + 1 } else { 0 }).collect()));
        put("y_bond_fail_mode", Arr::I32(vec![nb], live.iter().zip(&broke).map(|(&k, &b)| if b { mode1[k] } else { 0 }).collect()));
        put("y_bond_ddamage", f32a(&[nb], live.iter().zip(&broke).map(|(&k, &b)| if b { 0.0 } else { (joint1[4 * k] as f64 - joint[4 * k] as f64).max(0.0) }).collect()));
        let ydis: Vec<f32> = live.iter().map(|&k| (dis[k][0] + dis[k][1]) as f32).collect();
        let d_bond: f64 = ydis.iter().map(|&x| x as f64).sum();
        put("y_bond_dissipated", Arr::F32(vec![nb], ydis));
        let pos1 = r.rf("state_chunk_pos", f + 1)?;
        let vel1 = r.rf("state_chunk_vel", f + 1)?;
        let ang1 = r.rf("state_chunk_angvel", f + 1)?;
        let qw1 = r.rf("state_chunk_q_world", f + 1)?;
        let cl1 = r.ri("state_chunk_cluster", f + 1)?;
        let imp1 = r.rf("state_impactor", f + 1)?;
        put("y_chunk_v_end", f32a(&[nc, 3], isl.iter().flat_map(|&c| v3(vel1, c).to_array()).collect()));
        put("y_chunk_w_end", f32a(&[nc, 3], isl.iter().flat_map(|&c| v3(ang1, c).to_array()).collect()));
        put("y_chunk_dx", f32a(&[nc, 3], isl.iter().enumerate().flat_map(|(j, &c)| (v3(pos1, c) - xc[j]).to_array()).collect()));
        put("y_chunk_drot", f32a(&[nc, 3], isl.iter().enumerate().flat_map(|(j, &c)| rotvec_between(qc[j], quat(qw1, c)).to_array()).collect()));
        let mut ids: HashMap<i32, i32> = HashMap::new();
        let frag: Vec<i32> = isl.iter().map(|&c| { let k = ids.len() as i32; *ids.entry(cl1[c]).or_insert(k) }).collect();
        put("y_chunk_fragment", Arr::I32(vec![nc], frag));
        // The frame-end hidden state, exactly (a fit from f32 world arrays loses um-scale u).
        let def1 = hidden_state(r, st, f + 1, &isl)?;
        put("y_chunk_def_end", f32a(&[nc, 12], def1.iter().flatten().copied().collect()));
        let i1 = |k: usize, o: usize| Vec3::new(imp1[13 * k + o] as f64, imp1[13 * k + o + 1] as f64, imp1[13 * k + o + 2] as f64);
        let q1 = |k: usize| Quat::from_wxyz([imp1[13 * k + 3] as f64, imp1[13 * k + 4] as f64, imp1[13 * k + 5] as f64, imp1[13 * k + 6] as f64]).normalized();
        put("y_impactor_v_end", f32a(&[ni, 3], iimp.iter().flat_map(|&k| i1(k, 7).to_array()).collect()));
        put("y_impactor_w_end", f32a(&[ni, 3], iimp.iter().flat_map(|&k| i1(k, 10).to_array()).collect()));
        put("y_impactor_dx", f32a(&[ni, 3], iimp.iter().enumerate().flat_map(|(j, &k)| (i1(k, 0) - xi[j]).to_array()).collect()));
        put("y_impactor_drot", f32a(&[ni, 3], iimp.iter().enumerate().flat_map(|(j, &k)| rotvec_between(qi[j], q1(k)).to_array()).collect()));
        put("y_impactor_regime", Arr::I32(vec![ni], iimp.iter().map(|&k| history.regime(k)).collect()));
        // energy over the frame, island bodies
        let qcl1 = r.rf("state_chunk_cluster_quat", f + 1)?;
        let ke_out: f64 = (0..nc).map(|j| ke(mc[j], st.inertia[isl[j]], &quat(qcl1, isl[j]).to_mat3(), v3(vel1, isl[j]), v3(ang1, isl[j]))).sum::<f64>()
            + (0..ni).map(|j| ke(mi[j], Mat3::diag(st.imp_inertia[iimp[j]]), &q1(iimp[j]).to_mat3(), i1(iimp[j], 7), i1(iimp[j], 10))).sum::<f64>();
        let w_g: f64 = (0..nc).map(|j| mc[j] * (v3(pos1, isl[j]) - xc[j]).dot(g)).sum::<f64>() + (0..ni).map(|j| mi[j] * (i1(iimp[j], 0) - xi[j]).dot(g)).sum::<f64>();
        let el0: f64 = live.iter().map(|&k| en0[3 * k + 2] as f64).sum();
        let el1: f64 = live.iter().map(|&k| en1[3 * k + 2] as f64).sum();
        for (k, x) in [
            ("KE_in", ke_in), ("KE_out", ke_out), ("W_gravity", w_g), ("W_external", 0.0), ("E_elastic_start", el0), ("E_elastic_end", el1), ("D_bond", d_bond),
            ("D_bond_joint", live.iter().map(|&k| dis[k][0]).sum()), ("D_bond_damping", live.iter().map(|&k| dis[k][1]).sum()),
            ("D_fracture_floor", live.iter().zip(&broke).filter(|(_, &b)| b).map(|(&k, _)| ga_b(k)).sum()),
            ("D_contact_booked", history.booked_contact(f)),
            ("ledger_residual", (ke_out + el1 + d_bond - ke_in - el0 - w_g) / e_in),
        ] {
            energy.insert(k.to_string(), x);
        }
    } else {
        energy.insert("KE_in".to_string(), ke_in);
    }
    let phase_idx = r.ri("frame_phase", f).map(|p| p[0] as usize).unwrap_or(0);
    let phase = PHASES.get(phase_idx).copied().unwrap_or("pre_contact").to_string();
    let mut counts = BTreeMap::new();
    for (k, x) in [("chunks", nc), ("impactors", ni), ("bond", nb), ("cracked", cr.keys.len()), ("impactor_contact", ic.keys.len()), ("structure_contact", sc.keys.len())] {
        counts.insert(k.to_string(), x);
    }
    Ok(Some(Call { frame: f, prev_call, phase, transition: history.transition(f), anchored, counts, scales, energy, arrays: a }))
}

#[derive(Default)]
struct EdgeSet {
    src: Vec<usize>,
    dst: Vec<usize>,
    feat: Vec<f64>,
    frames: Vec<[f32; 9]>,
    keys: Vec<i64>,
}

impl EdgeSet {
    fn push(&mut self, s: usize, d: usize, row: Vec<f64>) {
        self.src.push(s);
        self.dst.push(d);
        self.feat.extend(row);
    }
}

/// What a call needs from frames other than its own.
struct History {
    /// Frame in which each bond broke (from the record's break bins), if it did.
    broke_in: Vec<Option<usize>>,
    since_live: Option<Vec<i32>>,
    phase: Vec<i32>,
    first_break_frame: Option<usize>,
    punch_frames: Vec<i32>,
    regimes: Vec<i32>,
    booked: Vec<f64>,
}

impl History {
    fn of(r: &Record, e: usize) -> Result<History, String> {
        let mut broke_in = vec![None; e];
        let mut first_break_frame = None;
        // Thinned events keep every frame with a break; a live record has no outcome rows.
        let held = if r.has("frame_bond_break_bin") { r.frames_held() } else { Vec::new() };
        for f in held {
            let b = r.ri("frame_bond_break_bin", f)?;
            for k in 0..e {
                if b[k] >= 0 && broke_in[k].is_none() {
                    broke_in[k] = Some(f);
                    first_break_frame.get_or_insert(f);
                }
            }
        }
        // A live one-frame record carries, per broken bond, the frames since it broke.
        let since_live = if r.has("live_bond_frames_since_broken") { Some(r.ri("live_bond_frames_since_broken", 0)?.to_vec()) } else { None };
        // Every frame's phase: frames.json (always whole), else the array (a live record).
        let phase = if r.frames.is_empty() {
            (0..r.rows("frame_phase")).map(|f| r.ri("frame_phase", f).map(|p| p[0])).collect::<Result<Vec<_>, _>>()?
        } else {
            r.frames.iter().map(|x| PHASES.iter().position(|p| Some(*p) == x["phase"].as_str()).unwrap_or(0) as i32).collect()
        };
        let punch_frames = (0..r.rows("punch_frame")).map(|f| r.ri("punch_frame", f).map(|p| p[0])).collect::<Result<Vec<_>, _>>()?;
        let regime = |s: &str| match s {
            "arrest" => 0,
            "perforate" => 1,
            "rebound" => 2,
            "deflect" => 3,
            _ => -1,
        };
        let regimes = r.outcome.as_ref().and_then(|o| o["impactors"].as_array()).map(|a| a.iter().map(|x| regime(x["regime"].as_str().unwrap_or(""))).collect()).unwrap_or_default();
        let booked = r.frames.iter().map(|x| x["energy"]["contact_dissipated"].as_f64().unwrap_or(0.0)).collect();
        Ok(History { broke_in, since_live, phase, first_break_frame, punch_frames, regimes, booked })
    }

    fn frames_since_broken(&self, k: usize, f: usize) -> Option<usize> {
        match &self.since_live {
            Some(s) => (s[k] >= 0).then_some(s[k] as usize),
            None => self.broke_in[k].map(|b| f.saturating_sub(b)),
        }
    }

    fn punch_row(&self, f: usize) -> Option<usize> {
        self.punch_frames.iter().position(|&x| x as usize == f)
    }

    fn regime(&self, k: usize) -> i32 {
        self.regimes.get(k).copied().unwrap_or(-1)
    }

    fn transition(&self, f: usize) -> bool {
        (f > 0 && self.phase.get(f) != self.phase.get(f - 1)) || self.first_break_frame == Some(f)
    }

    /// Contact dissipation booked during frame `f` (frames.json holds the ledger after each frame).
    fn booked_contact(&self, f: usize) -> f64 {
        self.booked.get(f).copied().unwrap_or(0.0) - if f > 0 { self.booked.get(f - 1).copied().unwrap_or(0.0) } else { 0.0 }
    }
}

// ------------------------------------------------------------------ events and writing

pub struct CallEvent {
    pub manifest: Value,
    pub calls: Vec<Call>,
    /// Per structure: chunk keys, bond keys, bond ends (local), rest positions.
    pub structures: Vec<(usize, Vec<i64>, Vec<i64>, Vec<i32>, Vec<f32>)>,
    /// Event-level arrays (`/5`: the full state's static columns), in the manifest's
    /// `event_arrays`.
    pub event_arrays: Vec<(String, Arr)>,
}

fn structures_of(st: &Statics, r: &Record) -> Vec<(usize, Vec<i64>, Vec<i64>, Vec<i32>, Vec<f32>)> {
    let ns = r.scene.bodies.len();
    let bk = r.ctx["bond_keys"].as_array().cloned().unwrap_or_default();
    (0..ns)
        .map(|s| {
            let cm: Vec<usize> = (0..st.n).filter(|&c| st.structure[c] == s).collect();
            let local: HashMap<usize, usize> = cm.iter().enumerate().map(|(j, &c)| (c, j)).collect();
            let bm: Vec<usize> = (0..st.e).filter(|&k| bk[k][0].as_u64() == Some(s as u64)).collect();
            (
                s,
                cm.iter().map(|&c| st.ckey[c]).collect(),
                bm.iter().map(|&k| st.bkey[k]).collect(),
                bm.iter().flat_map(|&k| [local[&st.ends[k].0] as i32, local[&st.ends[k].1] as i32]).collect(),
                cm.iter().flat_map(|&c| st.center[c].to_array().map(|x| x as f32)).collect(),
            )
        })
        .collect()
}

fn manifest_of(r: &Record, st: &Statics, calls: &[Call], truncated: bool) -> Value {
    let ctx = &r.ctx;
    let ens = &ctx["ensemble"];
    let cols: serde_json::Map<String, Value> = feature_columns().into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
    let outcome = r.outcome.as_ref().map(|o| {
        json!({
            "impactors": o["impactors"].as_array().map(|a| a.iter().enumerate().map(|(k, x)| {
                let v = x["v_end"].as_array().map(|v| v.iter().map(|y| y.as_f64().unwrap_or(0.0)).collect::<Vec<_>>()).unwrap_or_default();
                json!({"key": k, "v_exit": v, "speed_exit": v.iter().map(|y| y * y).sum::<f64>().sqrt(), "regime": x["regime"]})
            }).collect::<Vec<_>>()),
            "n_fragments": o["fragments"], "largest_fraction": o["largest_fraction"], "sim_time": o["sim_seconds"], "frames": o["frames"],
        })
    });
    json!({
        "schema_version": CALLS_SCHEMA, "n_time_bins": N_TIME_BINS, "columns": cols,
        "event_id": r.name, "recipe_id": ctx["event_id"], "family": ctx["family"], "layout_family": ctx["layout_group"],
        "split_group": ctx["split_group"], "split": ctx["split"].as_str().unwrap_or("unassigned"),
        "ensemble_id": ens["id"], "ensemble_member": ens["member"], "ensemble_size": ens["size"].as_u64().unwrap_or(1),
        "perturbation": ens["perturbation"], "seed": r.scene.sim.seed,
        "oracle": {"config": ctx["oracle_config"]["name"], "methods": ctx["oracle_config"]["methods"], "contact": if r.scene.sim.methods.layer_contact { "layer" } else { "penalty" }, "hash": ctx["scene_hash"]},
        "scene_hash": ctx["scene_hash"], "recorder_revision": format!("p1-rev{}", r.revision()), "builder_revision": BUILDER_REVISION,
        "routing": {"name": "untiered", "source": "the recorder's frame_chunk_routed / frame_impactor_routed (World::predict_island_members, select_islands with the oracle configuration's defaults: routing_tiered off)"},
        "frame_dt": st.dt, "gravity": st.gravity.to_array(), "truncated": truncated, "n_calls": calls.len(),
        "available": {"qs_hertz": true, "punch": true, "qs_frame": r.has("qs_bond_broken"), "punch_frame": r.rows("punch_frame") > 0, "blast": false,
            "cracked_edges": true, "structure_contacts": true, "bond_energy_split": true},
        "outcome": outcome,
    })
}

/// Every call of a recorded event, with targets.
pub fn build_event(dir: &Path) -> Result<CallEvent, String> {
    let r = Record::load(dir)?;
    if r.revision() < 6 {
        return Err(format!("{}: recorder revision {} has no failure margin (revision 6 on)", r.name, r.revision()));
    }
    let st = Statics::of(&r)?;
    let hist = History::of(&r, st.e)?;
    let mut calls: Vec<Call> = Vec::new();
    let mut last: Option<(usize, usize)> = None;
    for f in r.frames_held() {
        let prev = last.filter(|&(lf, _)| lf + 1 == f).map(|(_, i)| i);
        if let Some(c) = build_call(&r, &st, f, prev, &hist, &BuildOptions { targets: true })? {
            last = Some((f, calls.len()));
            calls.push(c);
        }
    }
    let truncated = r.outcome.as_ref().and_then(|o| o["truncated"].as_bool()).unwrap_or(false);
    let full = crate::full_state_calls::augment(dir, &r.ctx, &mut calls)?;
    let mut manifest = manifest_of(&r, &st, &calls, truncated);
    let mut event_arrays = Vec::new();
    if let Some((cols, statics)) = full {
        manifest["schema_version"] = json!(CALLS_SCHEMA_FULL);
        manifest["full_state_columns"] = cols;
        event_arrays = statics;
    }
    Ok(CallEvent { manifest, calls, structures: structures_of(&st, &r), event_arrays })
}

/// The call of a live world's coming frame for the island of the given chunk and impactor
/// keys (the closed loop): its inputs captured exactly as the recorder captures them. No
/// targets. The caller must have reset the bonds' peak utilizations at the previous frame's
/// start (`ReferenceSolver::reset_peak_utilizations`), as the recorder does, for
/// `log1p_peak_util_prev`; `frame` is the engine frame index. With `qs`, the variant-B
/// groups come from the `blast_parity` QS frame of the same state (as recorded); without,
/// they are zero and `available.qs_frame` is false.
pub fn live_call(world: &mut World, name: &str, ctx_from: &Value, chunk_keys: &[i64], impactor_keys: &[i64], frame: usize, qs: bool) -> Result<CallEvent, String> {
    live_call_labelled(world, name, ctx_from, chunk_keys, impactor_keys, frame, qs, false)
}

/// `live_call`, and with `targets` the oracle's frame from that state as its targets: the
/// world is stepped one frame (`outcome::labelled_frame`, exactly as the recorder steps and
/// records a frame) and left at the frame's end. DAgger relabelling (`relabel`).
#[allow(clippy::too_many_arguments)]
pub fn live_call_labelled(world: &mut World, name: &str, ctx_from: &Value, chunk_keys: &[i64], impactor_keys: &[i64], frame: usize, qs: bool, targets: bool) -> Result<CallEvent, String> {
    let ix = Index::new(world);
    let (mut statics, mut a) = (Arrays::default(), Arrays::default());
    capture_static(world, &ix, &mut statics)?;
    a.declare_i("contact_i32", &[5]);
    a.declare_f("contact_f32", &[crate::outcome::CONTACT_F32]);
    a.declare_i("pair_i32", &[3]);
    capture_state(world, &ix, &mut a)?;
    let want_c: HashSet<i64> = chunk_keys.iter().copied().collect();
    let want_i: HashSet<i64> = impactor_keys.iter().copied().collect();
    let mut chunk_routed = vec![0i32; ix.n];
    for s in 0..world.solver.structures.len() {
        for c in 0..world.solver.structures[s].chunks.len() {
            chunk_routed[ix.chunk0[s] + c] = i32::from(want_c.contains(&keys::chunk_key(s, c)));
        }
    }
    let imp_routed: Vec<i32> = (0..world.impactors.len()).map(|k| i32::from(want_i.contains(&keys::impactor_key(k)))).collect();
    if chunk_routed.iter().filter(|&&x| x == 1).count() != want_c.len() || imp_routed.iter().filter(|&&x| x == 1).count() != want_i.len() {
        return Err("live_call: a requested key is not in the world".into());
    }
    // Rows are written as frame 0 of a one-frame record.
    let start_connected: Vec<i32> = world.solver.bonds.iter().flatten().map(|b| i32::from(b.connected())).collect();
    capture_inputs(world, &ix, 0, Some((chunk_routed, imp_routed)), &mut a)?;
    if qs {
        // The blast_parity QS frame of the same state, on a standalone copy (a coupled
        // world's engine is never stepped here), exactly as the recorder runs it.
        let standalone = world.standalone_copy();
        let template = crate::oracle_config::qs_template(&standalone.scene);
        let q = crate::outcome::branch_frame(&standalone, &template, &[])?;
        crate::outcome::capture_branch("qs", &start_connected, &q, &ix, &mut a)?;
    }
    // The previous frame's peak utilization (the caller reset it at that frame's start).
    a.f("live_bond_peak_prev", &[ix.e], &world.solver.bonds.iter().flatten().map(|b| b.peak_utilization as f32).collect::<Vec<_>>())?;
    // When each disconnected bond broke, from the solver's events.
    let dt = world.scene.sim.frame_dt;
    let mut since = vec![-1i32; ix.e];
    for ev in &world.solver.events {
        if let stress_ref::solver::SolverEvent::Broken { time, structure, bond, .. } = ev {
            let g = ix.bond0[*structure] + bond;
            if since[g] < 0 {
                since[g] = frame as i32 - (time / dt).floor() as i32;
            }
        }
    }
    a.i("live_bond_frames_since_broken", &[ix.e], &since)?;
    a.i("frame_phase", &[], &[0])?;
    if targets {
        let peak_prev: Vec<f32> = world.solver.bonds.iter().flatten().map(|b| b.peak_utilization as f32).collect();
        crate::outcome::labelled_frame(world, &ix, 0, &peak_prev, &mut a)?;
    }
    for (k, x) in statics.f32s {
        a.f32s.insert(k, x);
    }
    for (k, x) in statics.i32s {
        a.i32s.insert(k, x);
    }
    let chunk_keys_json: Vec<Value> = world.solver.structures.iter().enumerate().flat_map(|(s, st)| (0..st.chunks.len()).map(move |c| json!([s, c]))).collect();
    let bond_keys_json: Vec<Value> = world.solver.bonds.iter().enumerate().flat_map(|(s, bs)| bs.iter().map(move |b| json!([s, b.source, b.geometry.a, b.geometry.b]))).collect();
    let mut ctx = ctx_from.clone();
    ctx["schema"] = json!(crate::outcome::OUTCOME_SCHEMA);
    ctx["recorder_revision"] = json!(crate::outcome::RECORDER_REVISION);
    ctx["n_chunks"] = json!(ix.n);
    ctx["n_bonds"] = json!(ix.e);
    ctx["n_impactors"] = json!(world.impactors.len());
    ctx["chunk_keys"] = json!(chunk_keys_json);
    ctx["bond_keys"] = json!(bond_keys_json);
    let r = Record::from_arrays(name.to_string(), world.scene.clone(), ctx, a);
    let st = Statics::of(&r)?;
    let hist = History::of(&r, st.e)?;
    let mut call = build_call(&r, &st, 0, None, &hist, &BuildOptions { targets })?.ok_or("live_call: empty island")?;
    call.frame = frame;
    let calls = vec![call];
    let manifest = manifest_of(&r, &st, &calls, false);
    Ok(CallEvent { manifest, calls, structures: structures_of(&st, &r), event_arrays: Vec::new() })
}

/// `live_call_labelled` as a `learned-impact-outcome/5` call: the island's full state by field
/// path (`full_state_calls::augment_live`, the recorder's own rule) from the frame start, and
/// with `targets` from the frame end. `columns`: the training manifest's
/// `full_state_columns` (its columns and static split; None: the call's own).
#[allow(clippy::too_many_arguments)]
pub fn live_call_full(world: &mut World, name: &str, ctx_from: &Value, chunk_keys: &[i64], impactor_keys: &[i64], frame: usize, qs: bool, targets: bool, columns: Option<&Value>) -> Result<CallEvent, String> {
    let start = world.clone();
    let mut ev = live_call_labelled(world, name, ctx_from, chunk_keys, impactor_keys, frame, qs, targets)?;
    let bond_keys: Vec<Value> = start.solver.bonds.iter().enumerate().flat_map(|(s, bs)| bs.iter().map(move |b| json!([s, b.source, b.geometry.a, b.geometry.b]))).collect();
    let mut ctx = ctx_from.clone();
    ctx["bond_keys"] = json!(bond_keys);
    let (cols, statics) = crate::full_state_calls::augment_live(&start, targets.then_some(&*world), &ctx, &mut ev.calls, columns)?;
    ev.manifest["schema_version"] = json!(CALLS_SCHEMA_FULL);
    ev.manifest["full_state_columns"] = cols;
    ev.event_arrays = statics;
    Ok(ev)
}

/// The `learned-impact-outcome/3` record of a `/4` event, for models trained on `/3`: `/4`
/// only added to `/3` (the hidden deformation: chunk_feat's last 12 columns, `chunk_def`,
/// `chunk_def_uc`, `y_chunk_def_end`; the bodies: `chunk_body`, `chunk_body_q`), so dropping
/// the additions is the `/3` record exactly.
///
/// Exact for builder revisions 4 and 5 only: revision 6 changed `/3`-era quantities (the
/// chunk inertia features are principal moments, and the KE in `E_in` turns the
/// structure-frame inertia with the cluster), so a `/3` model is refused rather than fed
/// shifted inputs; run `/3` checkpoints with a revision-5 build.
pub fn downgrade_to_v3(ev: &mut CallEvent) -> Result<(), String> {
    const V3_CHUNK_FEAT: usize = 33;
    if ev.manifest["schema_version"] != CALLS_SCHEMA {
        return Err(format!("downgrade_to_v3: schema {} is not {CALLS_SCHEMA}", ev.manifest["schema_version"]));
    }
    let rev = ev.manifest["builder_revision"].as_u64().unwrap_or(0);
    if !(4..=5).contains(&rev) {
        return Err(format!(
            "downgrade_to_v3: builder revision {rev} is not exactly /3 plus additions (revision 6 changed the inertia features and E_in); \
             run a /3 checkpoint with a revision-5 build"
        ));
    }
    for call in &mut ev.calls {
        call.arrays.retain(|(k, _)| !matches!(k.as_str(), "chunk_def" | "chunk_def_uc" | "y_chunk_def_end" | "chunk_body" | "chunk_body_q"));
        for (k, a) in &mut call.arrays {
            if k == "chunk_feat" {
                let Arr::F32(shape, d) = a else { return Err("chunk_feat is not f32".into()) };
                let (n, w) = (shape[0], shape[1]);
                if w != V3_CHUNK_FEAT + 12 {
                    return Err(format!("chunk_feat width {w}, not {}", V3_CHUNK_FEAT + 12));
                }
                *d = (0..n).flat_map(|r| d[r * w..r * w + V3_CHUNK_FEAT].to_vec()).collect();
                shape[1] = V3_CHUNK_FEAT;
            }
        }
    }
    ev.manifest["schema_version"] = json!("learned-impact-outcome/3");
    ev.manifest["builder_revision"] = json!(3);
    ev.manifest["downgraded_from"] = json!(CALLS_SCHEMA);
    if let Some(cols) = ev.manifest["columns"]["chunk_feat"].as_array_mut() {
        cols.truncate(V3_CHUNK_FEAT);
    }
    Ok(())
}

/// Write an event in P2's on-disk format: `manifest.json`, `calls.json` and the arrays,
/// 8-byte aligned, as `arrays.bin.zst` (one zstd frame of exactly the `arrays.bin` bytes)
/// or plain `arrays.bin`; identical arrays share their bytes (two entries, one offset).
pub fn write_event(dir: &Path, ev: &CallEvent, compress: bool) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut blob: Vec<u8> = Vec::new();
    // Identical arrays are stored once and share an offset (lossless; readers address every
    // array by its own dtype, shape and offset): a call's frame-start state is mostly the
    // previous call's frame-end state.
    let mut stored: HashMap<(&'static str, Vec<u8>), usize> = HashMap::new();
    let mut add = |a: &Arr| -> Value {
        let mut bytes: Vec<u8> = Vec::new();
        let (dtype, shape) = match a {
            Arr::F32(s, d) => {
                for x in d {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                ("f32", s)
            }
            Arr::I32(s, d) => {
                for x in d {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                ("i32", s)
            }
            Arr::I64(s, d) => {
                for x in d {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                ("i64", s)
            }
            Arr::F64(s, d) => {
                for x in d {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                ("f64", s)
            }
        };
        let off = match (bytes.is_empty(), stored.get(&(dtype, bytes.clone()))) {
            (false, Some(&off)) => off,
            _ => {
                let off = blob.len();
                blob.extend_from_slice(&bytes);
                while blob.len() % 8 != 0 {
                    blob.push(0);
                }
                if !bytes.is_empty() {
                    stored.insert((dtype, bytes), off);
                }
                off
            }
        };
        json!({"dtype": dtype, "shape": shape, "offset": off})
    };
    let structures: Vec<Value> = ev
        .structures
        .iter()
        .map(|(s, ck, bk, ends, rest)| {
            json!({"structure_id": s, "n_chunks": ck.len(), "n_bonds": bk.len(), "arrays": {
                "chunk_key": add(&Arr::I64(vec![ck.len()], ck.clone())),
                "bond_key": add(&Arr::I64(vec![bk.len()], bk.clone())),
                "bond_ends": add(&Arr::I32(vec![bk.len(), 2], ends.clone())),
                "chunk_rest_pos": add(&Arr::F32(vec![ck.len(), 3], rest.clone())),
            }})
        })
        .collect();
    let entries: Vec<Value> = ev
        .calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let arrays: serde_json::Map<String, Value> = c.arrays.iter().map(|(k, a)| (k.clone(), add(a))).collect();
            json!({"call_index": i, "frame": c.frame, "island": 0, "prev_call": c.prev_call, "phase": c.phase, "transition": c.transition,
                "routed": true, "anchored": c.anchored, "counts": c.counts, "scales": c.scales, "energy": c.energy, "arrays": arrays})
        })
        .collect();
    let event_arrays: serde_json::Map<String, Value> = ev.event_arrays.iter().map(|(k, a)| (k.clone(), add(a))).collect();
    let mut manifest = ev.manifest.clone();
    manifest["structures"] = json!(structures);
    if !event_arrays.is_empty() {
        manifest["event_arrays"] = Value::Object(event_arrays);
    }
    let write = |n: &str, b: &[u8]| fs::write(dir.join(n), b).map_err(|e| format!("{n}: {e}"));
    write("manifest.json", &serde_json::to_vec_pretty(&manifest).unwrap())?;
    write("calls.json", &serde_json::to_vec(&entries).unwrap())?;
    let (plain, zst) = (dir.join("arrays.bin"), dir.join("arrays.bin.zst"));
    if compress {
        let z = zstd::bulk::compress(&blob, 19).map_err(|e| format!("zstd: {e}"))?;
        write("arrays.bin.zst", &z)?;
        if plain.exists() {
            fs::remove_file(&plain).map_err(|e| e.to_string())?;
        }
    } else {
        write("arrays.bin", &blob)?;
        if zst.exists() {
            fs::remove_file(&zst).map_err(|e| e.to_string())?;
        }
    }
    Ok(dir.to_path_buf())
}
