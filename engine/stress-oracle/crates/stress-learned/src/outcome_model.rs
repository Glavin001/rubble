//! CPU reference of P2's impact-outcome GNN (`learned_impact/outcome/model.py`, P3's port):
//! the forward pass on one call of a `learned-impact-outcome/3` record, from a
//! `learned-impact-weights/1` export, in f32 as PyTorch runs it on the CPU.
//!
//! The model: encoders per node and edge type (on standardised inputs), typed MeshGraphNets
//! rounds at the fine level, a BSMS bi-stride U on the masked bond graph (the hierarchy is
//! built per pristine structure, `Hierarchy`), a GraphNet global node throughout, then the
//! bond / chunk / impactor / energy heads. `forward` follows the PyTorch code step by step.
//! `Options::node_projection` turns on the GPU port's reformulation of the edge MLPs' first
//! layer on `[e, h_src, h_dst]` as `W_e·e + (W_s·h)[src] + (W_d·h)[dst]`, the two node
//! projections computed once per node instead of once per edge (the same for the coarse
//! rounds' `[fibre, h_i, h_j]`); it gives the same result up to f32 rounding.
//!
//! The input standardisation stays a separate step. Folded into the encoders' first layer
//! (`W/σ`, `b − W·μ/σ`) it is not lossless in f32: some bond columns have σ ≈ 1e-5 against
//! |μ| ≈ 10, and the folded dot product loses ~4e-3 to cancellation (measured on P2's
//! `sweep-all3`); standardising costs one subtraction and multiply per input.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::outcome_calls::{Arr, Call, CallEvent};

pub const WEIGHTS_SCHEMA: &str = "learned-impact-weights/1";
/// Δv(3) Δω(3) Δx(3) Δθ(3), body frame, nondimensional.
pub const NODE_OUT: usize = 12;
/// `bsms_graph.TIE_TOL`: rotation-stable seeds.
pub const TIE_TOL: f64 = 1e-5;

// ------------------------------------------------------------------ reading

fn read_json(p: &Path) -> Result<Value, String> {
    let s = fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_slice(&s).map_err(|e| format!("{}: {e}", p.display()))
}

fn shape_of(v: &Value) -> Vec<usize> {
    v.as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default()
}

fn f32s(blob: &[u8], off: usize, n: usize) -> Result<Vec<f32>, String> {
    let b = blob.get(off..off + 4 * n).ok_or_else(|| format!("array at {off} ({n} f32) past the end ({})", blob.len()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn arr_of(blob: &[u8], spec: &Value) -> Result<Arr, String> {
    let shape = shape_of(&spec["shape"]);
    let n: usize = shape.iter().product();
    let off = spec["offset"].as_u64().ok_or("array without offset")? as usize;
    let get = |w: usize| blob.get(off..off + w * n).ok_or_else(|| format!("array at {off} past the end ({})", blob.len()));
    Ok(match spec["dtype"].as_str() {
        Some("f32") => Arr::F32(shape, f32s(blob, off, n)?),
        Some("i32") => Arr::I32(shape, get(4)?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        Some("i64") => Arr::I64(shape, get(8)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()),
        other => return Err(format!("dtype {other:?}")),
    })
}

/// Read an event written by `outcome_calls::write_event` (P2's on-disk format, plain
/// `arrays.bin` or `arrays.bin.zst`).
pub fn read_event(dir: &Path) -> Result<CallEvent, String> {
    let mut manifest = read_json(&dir.join("manifest.json"))?;
    let entries = read_json(&dir.join("calls.json"))?;
    let blob = if dir.join("arrays.bin.zst").exists() {
        let z = fs::read(dir.join("arrays.bin.zst")).map_err(|e| e.to_string())?;
        zstd::stream::decode_all(&z[..]).map_err(|e| format!("zstd: {e}"))?
    } else {
        fs::read(dir.join("arrays.bin")).map_err(|e| format!("{}: {e}", dir.display()))?
    };
    let mut structures = Vec::new();
    for s in manifest["structures"].as_array().cloned().unwrap_or_default() {
        let a = &s["arrays"];
        let i64s = |k: &str| -> Result<Vec<i64>, String> {
            match arr_of(&blob, &a[k])? {
                Arr::I64(_, d) => Ok(d),
                Arr::I32(_, d) => Ok(d.into_iter().map(i64::from).collect()),
                _ => Err(format!("structure {k}: not integer")),
            }
        };
        let ends = match arr_of(&blob, &a["bond_ends"])? {
            Arr::I32(_, d) => d,
            Arr::I64(_, d) => d.into_iter().map(|x| x as i32).collect(),
            _ => return Err("bond_ends: not integer".into()),
        };
        let rest = match arr_of(&blob, &a["chunk_rest_pos"])? {
            Arr::F32(_, d) => d,
            _ => return Err("chunk_rest_pos: not f32".into()),
        };
        structures.push((s["structure_id"].as_u64().unwrap_or(0) as usize, i64s("chunk_key")?, i64s("bond_key")?, ends, rest));
    }
    let map_f = |v: &Value| -> BTreeMap<String, f64> { v.as_object().map(|o| o.iter().map(|(k, x)| (k.clone(), x.as_f64().unwrap_or(0.0))).collect()).unwrap_or_default() };
    let mut calls = Vec::new();
    for c in entries.as_array().ok_or("calls.json: not a list")? {
        let mut arrays = Vec::new();
        for (k, spec) in c["arrays"].as_object().ok_or("call without arrays")? {
            arrays.push((k.clone(), arr_of(&blob, spec)?));
        }
        calls.push(Call {
            frame: c["frame"].as_u64().unwrap_or(0) as usize,
            prev_call: c["prev_call"].as_u64().map(|x| x as usize),
            phase: c["phase"].as_str().unwrap_or("").to_string(),
            transition: c["transition"].as_bool().unwrap_or(false),
            anchored: c["anchored"].as_bool().unwrap_or(false),
            counts: c["counts"].as_object().map(|o| o.iter().map(|(k, x)| (k.clone(), x.as_u64().unwrap_or(0) as usize)).collect()).unwrap_or_default(),
            scales: map_f(&c["scales"]),
            energy: map_f(&c["energy"]),
            arrays,
        });
    }
    if let Some(o) = manifest.as_object_mut() {
        o.remove("structures");
    }
    Ok(CallEvent { manifest, calls, structures })
}

// ------------------------------------------------------------------ dense rows

/// Row-major `rows × cols` f32.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub d: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, d: vec![0.0; rows * cols] }
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.d[i * self.cols..(i + 1) * self.cols]
    }
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.d[i * self.cols..(i + 1) * self.cols]
    }
    fn gather(&self, idx: &[usize]) -> Mat {
        let mut m = Mat::zeros(idx.len(), self.cols);
        for (r, &i) in idx.iter().enumerate() {
            m.row_mut(r).copy_from_slice(self.row(i));
        }
        m
    }
    /// Row-wise concatenation of column blocks.
    fn hcat(parts: &[&Mat]) -> Mat {
        let rows = parts[0].rows;
        let cols = parts.iter().map(|p| p.cols).sum();
        let mut m = Mat::zeros(rows, cols);
        for r in 0..rows {
            let mut c = 0;
            for p in parts {
                debug_assert_eq!(p.rows, rows);
                m.d[r * cols + c..r * cols + c + p.cols].copy_from_slice(p.row(r));
                c += p.cols;
            }
        }
        m
    }
    fn add(&mut self, o: &Mat) {
        for (a, b) in self.d.iter_mut().zip(&o.d) {
            *a += b;
        }
    }
    /// `out[index[r]] += self[r]` (PyTorch's `index_add_`, rows in order).
    fn scatter_sum(&self, index: &[usize], n: usize) -> Mat {
        let mut out = Mat::zeros(n, self.cols);
        for (r, &i) in index.iter().enumerate() {
            for (a, b) in out.row_mut(i).iter_mut().zip(self.row(r)) {
                *a += b;
            }
        }
        out
    }
}

pub(crate) fn mat_of(a: &Arr) -> Result<Mat, String> {
    match a {
        Arr::F32(s, d) => {
            let rows = s.first().copied().unwrap_or(0);
            let cols = if s.len() > 1 { s[1..].iter().product() } else { 1 };
            Ok(Mat { rows, cols, d: d.clone() })
        }
        _ => Err("not an f32 array".into()),
    }
}

pub(crate) fn idx_of(a: &Arr) -> Result<Vec<usize>, String> {
    match a {
        Arr::I32(_, d) => Ok(d.iter().map(|&x| x as usize).collect()),
        Arr::I64(_, d) => Ok(d.iter().map(|&x| x as usize).collect()),
        _ => Err("not an index array".into()),
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `torch.nn.functional.softplus` (β 1, threshold 20).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ------------------------------------------------------------------ layers

#[derive(Clone, Debug)]
pub(crate) struct Linear {
    pub(crate) inp: usize,
    pub(crate) out: usize,
    /// `out × inp`, PyTorch's layout.
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
}

impl Linear {
    fn forward(&self, x: &Mat) -> Mat {
        assert_eq!(x.cols, self.inp, "linear input width");
        let mut y = Mat::zeros(x.rows, self.out);
        for r in 0..x.rows {
            let xr = x.row(r);
            for o in 0..self.out {
                let wr = &self.w[o * self.inp..(o + 1) * self.inp];
                let mut s = 0.0f32;
                for k in 0..self.inp {
                    s += wr[k] * xr[k];
                }
                y.d[r * self.out + o] = s + self.b[o];
            }
        }
        y
    }
    /// Columns `[c0, c0 + n)` of the weight as a bias-free layer.
    fn block(&self, c0: usize, n: usize, bias: bool) -> Linear {
        let mut w = Vec::with_capacity(self.out * n);
        for o in 0..self.out {
            w.extend_from_slice(&self.w[o * self.inp + c0..o * self.inp + c0 + n]);
        }
        Linear { inp: n, out: self.out, w, b: if bias { self.b.clone() } else { vec![0.0; self.out] } }
    }
}

/// `MeshGraphMLP`: Linear · SiLU · [Linear · SiLU]×(h−1) · Linear · [LayerNorm].
#[derive(Clone, Debug)]
pub(crate) struct Mlp {
    pub(crate) layers: Vec<Linear>,
    pub(crate) ln: Option<(Vec<f32>, Vec<f32>)>,
}

impl Mlp {
    fn forward(&self, x: &Mat) -> Mat {
        self.after_first(self.layers[0].forward(x))
    }
    /// The rest of the MLP from the first layer's pre-activation.
    fn after_first(&self, mut y: Mat) -> Mat {
        for l in &self.layers[1..] {
            for v in &mut y.d {
                *v = silu(*v);
            }
            y = l.forward(&y);
        }
        if let Some((g, b)) = &self.ln {
            let n = y.cols;
            for r in 0..y.rows {
                let row = y.row_mut(r);
                let mean = row.iter().sum::<f32>() / n as f32;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
                let inv = 1.0 / (var + 1e-5).sqrt();
                for (k, v) in row.iter_mut().enumerate() {
                    *v = (*v - mean) * inv * g[k] + b[k];
                }
            }
        }
        y
    }
}

/// An MLP on `[e, x[a], x[b], extra…]` with its first layer split by input block
/// (`Options::node_projection`): node blocks are projected once per node, then gathered.
struct SplitFirst {
    blocks: Vec<Linear>,
}

impl SplitFirst {
    fn of(mlp: &Mlp, widths: &[usize]) -> SplitFirst {
        let mut c = 0;
        let blocks = widths
            .iter()
            .enumerate()
            .map(|(k, &w)| {
                let l = mlp.layers[0].block(c, w, k == 0);
                c += w;
                l
            })
            .collect();
        assert_eq!(c, mlp.layers[0].inp);
        SplitFirst { blocks }
    }
}

// ------------------------------------------------------------------ weights

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub width: usize,
    pub hidden_layers: usize,
    pub rounds: Vec<usize>,
    pub levels: usize,
    pub variant: String,
    pub n_time_bins: usize,
    pub use_global: bool,
    pub input_norm: bool,
    pub edge_types: Vec<String>,
}

/// A loaded `learned-impact-weights/1` export.
pub struct Weights {
    pub manifest: Value,
    pub config: ModelConfig,
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    pub fn load(dir: &Path) -> Result<Weights, String> {
        let manifest = read_json(&dir.join("weights.json"))?;
        if manifest["schema"].as_str() != Some(WEIGHTS_SCHEMA) {
            return Err(format!("weights schema {:?}, expected {WEIGHTS_SCHEMA}", manifest["schema"]));
        }
        let blob = fs::read(dir.join("weights.bin")).map_err(|e| format!("weights.bin: {e}"))?;
        let mut tensors = HashMap::new();
        for t in manifest["tensors"].as_array().ok_or("weights.json: no tensors")? {
            let shape = shape_of(&t["shape"]);
            let n = t["count"].as_u64().unwrap_or(0) as usize;
            let off = t["offset"].as_u64().unwrap_or(0) as usize;
            tensors.insert(t["name"].as_str().unwrap_or("").to_string(), (shape, f32s(&blob, off, n)?));
        }
        let c = &manifest["model_config"];
        let us = |k: &str| c[k].as_u64().map(|x| x as usize).ok_or_else(|| format!("model_config.{k}"));
        let config = ModelConfig {
            width: us("width")?,
            hidden_layers: us("mlp_hidden_layers")?,
            rounds: c["rounds"].as_array().ok_or("model_config.rounds")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect(),
            levels: us("levels")?,
            variant: c["variant"].as_str().unwrap_or("B").to_string(),
            n_time_bins: us("n_time_bins")?,
            use_global: c["use_global"].as_bool().unwrap_or(true),
            input_norm: c["input_norm"].as_bool().unwrap_or(true),
            edge_types: c["edge_types"].as_array().ok_or("model_config.edge_types")?.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect(),
        };
        if config.rounds.len() != 2 * config.levels + 1 {
            return Err(format!("rounds {:?} need {} stages", config.rounds, 2 * config.levels + 1));
        }
        Ok(Weights { manifest, config, tensors })
    }

    fn get(&self, name: &str) -> Result<&(Vec<usize>, Vec<f32>), String> {
        self.tensors.get(name).ok_or_else(|| format!("weights: no tensor {name}"))
    }

    fn mlp(&self, prefix: &str, norm: bool) -> Result<Mlp, String> {
        // Sequential order: Linear (0), act (1), [Linear, act]…, Linear, [LayerNorm].
        let mut layers = Vec::new();
        for i in 0..=self.config.hidden_layers {
            let (ws, w) = self.get(&format!("{prefix}.model.{}.weight", 2 * i))?;
            let (_, b) = self.get(&format!("{prefix}.model.{}.bias", 2 * i))?;
            layers.push(Linear { inp: ws[1], out: ws[0], w: w.clone(), b: b.clone() });
        }
        let ln = if norm {
            let k = 2 * self.config.hidden_layers + 1;
            Some((self.get(&format!("{prefix}.model.{k}.weight"))?.1.clone(), self.get(&format!("{prefix}.model.{k}.bias"))?.1.clone()))
        } else {
            None
        };
        Ok(Mlp { layers, ln })
    }

    fn vec(&self, name: &str) -> Result<Vec<f32>, String> {
        Ok(self.get(name)?.1.clone())
    }
}

// ------------------------------------------------------------------ BSMS hierarchy

/// Level l+1 built from level l (`bsms_graph.Level`).
#[derive(Clone, Debug, PartialEq)]
pub struct Level {
    /// Level-l nodes kept.
    pub ids: Vec<usize>,
    /// Level-(l+1) directed edges (level-(l+1) ids), lexicographic.
    pub edges: Vec<(usize, usize)>,
    /// Edge `k` is supported by the level-l paths `(sup_a[s], sup_b[s])` with `sup_owner[s] == k`.
    pub sup_owner: Vec<usize>,
    pub sup_a: Vec<usize>,
    pub sup_b: Vec<usize>,
}

/// Per pristine structure: level-0 directed edges plus the coarse levels (`bsms_graph.Hierarchy`).
#[derive(Clone, Debug, PartialEq)]
pub struct Hierarchy {
    pub n_nodes: usize,
    pub edges0: Vec<(usize, usize)>,
    pub edge0_bond: Vec<usize>,
    pub levels: Vec<Level>,
}

const UNREACHED: i64 = -1;

fn adjacency(edges: &[(usize, usize)], n: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(a, b) in edges {
        adj[a].push(b);
    }
    adj
}

fn bfs_dist(adj: &[Vec<usize>], seed: usize) -> Vec<i64> {
    let mut d = vec![UNREACHED; adj.len()];
    d[seed] = 0;
    let mut q = VecDeque::from([seed]);
    while let Some(f) = q.pop_front() {
        for &n in &adj[f] {
            if d[n] == UNREACHED {
                d[n] = d[f] + 1;
                q.push_back(n);
            }
        }
    }
    d
}

fn find_clusters(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut label = vec![usize::MAX; adj.len()];
    let mut out = Vec::new();
    for s in 0..adj.len() {
        if label[s] != usize::MAX {
            continue;
        }
        let c = out.len();
        label[s] = c;
        let mut q = VecDeque::from([s]);
        let mut members = vec![s];
        while let Some(f) = q.pop_front() {
            for &n in &adj[f] {
                if label[n] == usize::MAX {
                    label[n] = c;
                    q.push_back(n);
                    members.push(n);
                }
            }
        }
        members.sort_unstable();
        out.push(members);
    }
    out
}

/// The node nearest each cluster's centroid, ties (within `tie_tol` × the cluster radius plus
/// f32 rounding at the coordinates' magnitude) to the lowest index (`nearest_center_seed`).
fn nearest_center_seed(pos: &[[f32; 3]], clusters: &[Vec<usize>], tie_tol: f64) -> Vec<usize> {
    clusters
        .iter()
        .map(|c| {
            let p: Vec<[f64; 3]> = c.iter().map(|&i| pos[i].map(f64::from)).collect();
            let n = p.len() as f64;
            let center = [0, 1, 2].map(|k| p.iter().map(|x| x[k]).sum::<f64>() / n);
            let dist: Vec<f64> = p.iter().map(|x| ((x[0] - center[0]).powi(2) + (x[1] - center[1]).powi(2) + (x[2] - center[2]).powi(2)).sqrt()).collect();
            let dmin = dist.iter().copied().fold(f64::INFINITY, f64::min);
            if tie_tol > 0.0 {
                let dmax = dist.iter().copied().fold(0.0, f64::max).max(1e-30);
                let amax = p.iter().flat_map(|x| x.iter().map(|v| v.abs())).fold(0.0, f64::max);
                let slack = tie_tol * dmax + 16.0 * f64::from(f32::EPSILON) * amax;
                c[dist.iter().position(|&d| d <= dmin + slack).unwrap()]
            } else {
                c[dist.iter().position(|&d| d == dmin).unwrap()]
            }
        })
        .collect()
}

fn bstride_selection(edges: &[(usize, usize)], pos: &[[f32; 3]], n: usize, tie_tol: f64) -> Level {
    let adj = adjacency(edges, n);
    let clusters = find_clusters(&adj);
    let seeds = nearest_center_seed(pos, &clusters, tie_tol);
    let mut kept: HashSet<usize> = HashSet::new();
    for &seed in &seeds {
        let dist = bfs_dist(&adj, seed);
        let even: Vec<usize> = (0..n).filter(|&i| dist[i] != UNREACHED && dist[i] % 2 == 0).collect();
        let odd: Vec<usize> = (0..n).filter(|&i| dist[i] != UNREACHED && dist[i] % 2 == 1).collect();
        kept.extend(if even.len() <= odd.len() || odd.is_empty() { even } else { odd });
    }
    let mut ids: Vec<usize> = kept.into_iter().collect();
    ids.sort_unstable();
    let mut new_id = vec![usize::MAX; n];
    for (k, &i) in ids.iter().enumerate() {
        new_id[i] = k;
    }
    let eidx: HashMap<(usize, usize), usize> = edges.iter().enumerate().map(|(k, &e)| (e, k)).collect();
    // Insertion-ordered support lists (Python dicts keep insertion order).
    let mut support: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for &a in &ids {
        for &b in &adj[a] {
            if new_id[b] != usize::MAX && b != a {
                support.entry((a, b)).or_default().push((eidx[&(a, b)], eidx[&(a, b)]));
            }
        }
        for &m in &adj[a] {
            for &b in &adj[m] {
                if b != a && new_id[b] != usize::MAX {
                    support.entry((a, b)).or_default().push((eidx[&(a, m)], eidx[&(m, b)]));
                }
            }
        }
    }
    let mut keys: Vec<(usize, usize)> = support.keys().copied().collect();
    keys.sort_unstable_by_key(|&(a, b)| (new_id[a], new_id[b]));
    let (mut owner, mut sa, mut sb) = (Vec::new(), Vec::new(), Vec::new());
    for (k, ab) in keys.iter().enumerate() {
        for &(p, q) in &support[ab] {
            owner.push(k);
            sa.push(p);
            sb.push(q);
        }
    }
    Level { ids, edges: keys.iter().map(|&(a, b)| (new_id[a], new_id[b])).collect(), sup_owner: owner, sup_a: sa, sup_b: sb }
}

impl Hierarchy {
    /// `build_hierarchy`: `bond_ends` undirected pristine bonds (local chunk indices).
    pub fn build(n_nodes: usize, bond_ends: &[(usize, usize)], rest_pos: &[[f32; 3]], levels: usize, tie_tol: f64) -> Hierarchy {
        let nb = bond_ends.len();
        let edges0: Vec<(usize, usize)> = bond_ends.iter().copied().chain(bond_ends.iter().map(|&(a, b)| (b, a))).collect();
        let edge0_bond: Vec<usize> = (0..nb).chain(0..nb).collect();
        let mut lv = Vec::new();
        let (mut g, mut p, mut n) = (edges0.clone(), rest_pos.to_vec(), n_nodes);
        for _ in 0..levels {
            let l = bstride_selection(&g, &p, n, tie_tol);
            g = l.edges.clone();
            p = l.ids.iter().map(|&i| p[i]).collect();
            n = l.ids.len();
            lv.push(l);
        }
        Hierarchy { n_nodes, edges0, edge0_bond, levels: lv }
    }

    /// `live_edges`: per level, the live mask of its directed edges and the present mask of
    /// its nodes. A coarse edge is live while some supporting fine path is entirely live.
    pub fn live_edges(&self, bond_live: &[bool], node_present: &[bool]) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        let mut live: Vec<bool> = self.edges0.iter().zip(&self.edge0_bond).map(|(&(a, b), &k)| bond_live[k] && node_present[a] && node_present[b]).collect();
        let mut present = node_present.to_vec();
        let (mut em, mut nm) = (vec![live.clone()], vec![present.clone()]);
        for lv in &self.levels {
            let mut nxt = vec![false; lv.edges.len()];
            for s in 0..lv.sup_owner.len() {
                if live[lv.sup_a[s]] && live[lv.sup_b[s]] {
                    nxt[lv.sup_owner[s]] = true;
                }
            }
            present = lv.ids.iter().map(|&i| present[i]).collect();
            for (k, &(a, b)) in lv.edges.iter().enumerate() {
                nxt[k] &= present[a] && present[b];
            }
            em.push(nxt.clone());
            nm.push(present.clone());
            live = nxt;
        }
        (em, nm)
    }
}

/// One call's masked hierarchy in compact ids (`batch.collate`): per level `l` its directed
/// edges; per `l ≥ 1` the level-(l−1) id of each node (`ids`) and its call chunk (`chunk`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CallHierarchy {
    pub edges: Vec<Vec<(usize, usize)>>,
    pub ids: Vec<Vec<usize>>,
    pub chunk: Vec<Vec<usize>>,
}

/// Build the hierarchy of every pristine structure of `ev` once (`HierarchyCache`).
pub fn structure_hierarchies(ev: &CallEvent, levels: usize) -> Vec<Hierarchy> {
    ev.structures
        .iter()
        .map(|(_, ck, _, ends, rest)| {
            let e: Vec<(usize, usize)> = ends.chunks_exact(2).map(|p| (p[0] as usize, p[1] as usize)).collect();
            let p: Vec<[f32; 3]> = rest.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
            Hierarchy::build(ck.len(), &e, &p, levels, TIE_TOL)
        })
        .collect()
}

fn i64s(a: &Arr) -> Vec<i64> {
    match a {
        Arr::I64(_, d) => d.clone(),
        Arr::I32(_, d) => d.iter().map(|&x| i64::from(x)).collect(),
        Arr::F32(_, d) => d.iter().map(|&x| x as i64).collect(),
    }
}

pub fn call_hierarchy(ev: &CallEvent, hier: &[Hierarchy], call: &Call, levels: usize) -> Result<CallHierarchy, String> {
    let ck = i64s(call.array("chunk_key").ok_or("call: no chunk_key")?);
    let pos_of: HashMap<i64, usize> = ck.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let bonds: HashSet<i64> = i64s(call.array("bond_key").ok_or("call: no bond_key")?).into_iter().collect();
    let mut out = CallHierarchy { edges: vec![Vec::new(); levels + 1], ids: vec![Vec::new(); levels + 1], chunk: vec![Vec::new(); levels + 1] };
    out.chunk[0] = (0..ck.len()).collect();
    let mut count = vec![0usize; levels + 1];
    let mut order: Vec<usize> = (0..ev.structures.len()).collect();
    order.sort_by_key(|&i| ev.structures[i].0);
    for si in order {
        let (_, sk, sb, _, _) = &ev.structures[si];
        let h = &hier[si];
        let call_idx: Vec<Option<usize>> = sk.iter().map(|k| pos_of.get(k).copied()).collect();
        let present: Vec<bool> = call_idx.iter().map(Option::is_some).collect();
        if !present.iter().any(|&p| p) {
            continue;
        }
        let live: Vec<bool> = sb.iter().map(|k| bonds.contains(k)).collect();
        let (em, nm) = h.live_edges(&live, &present);
        let mut comp: Vec<Vec<Option<usize>>> = vec![call_idx.clone()];
        let mut chunk_of: Vec<Vec<Option<usize>>> = vec![call_idx.clone()];
        for (k, &(a, b)) in h.edges0.iter().enumerate() {
            if em[0][k] {
                out.edges[0].push((comp[0][a].unwrap(), comp[0][b].unwrap()));
            }
        }
        for (li, lv) in h.levels.iter().enumerate().map(|(i, l)| (i + 1, l)) {
            let pres = &nm[li];
            let mut cid = vec![None; lv.ids.len()];
            for (k, &p) in pres.iter().enumerate() {
                if p {
                    cid[k] = Some(count[li]);
                    count[li] += 1;
                }
            }
            let prev: Vec<Option<usize>> = lv.ids.iter().map(|&i| comp[li - 1][i]).collect();
            let ch: Vec<Option<usize>> = lv.ids.iter().map(|&i| chunk_of[li - 1][i]).collect();
            for (k, &p) in pres.iter().enumerate() {
                if p {
                    out.ids[li].push(prev[k].unwrap());
                    out.chunk[li].push(ch[k].unwrap());
                }
            }
            for (k, &(a, b)) in lv.edges.iter().enumerate() {
                if em[li][k] {
                    out.edges[li].push((cid[a].unwrap(), cid[b].unwrap()));
                }
            }
            comp.push(cid);
            chunk_of.push(ch);
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------ model

/// The GPU port's reformulation (same result up to f32 rounding).
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub node_projection: bool,
    /// The GPU program's MLPs of many rows as GEMM tiles (`gnn_mlp_tiled`; the same
    /// arithmetic, another summation order).
    pub tiled_mlp: bool,
    /// The GPU program's edge sums staged into the node MLPs (`SEG_CSR`) instead of their
    /// own dispatches (the same sums, in the same order).
    pub fused_sums: bool,
}

pub(crate) struct FineRound {
    pub(crate) edge: Vec<Mlp>,
    pub(crate) node: Mlp,
    pub(crate) glob: Option<Mlp>,
}

pub(crate) struct CoarseRound {
    pub(crate) mlp_edge: Mlp,
    pub(crate) mlp_node: Mlp,
    pub(crate) glob: Option<Mlp>,
}

pub(crate) struct Norm {
    pub(crate) mean: Vec<f32>,
    pub(crate) std: Vec<f32>,
}

/// P2's `ImpactOutcomeModel` with its weights.
pub struct Model {
    pub config: ModelConfig,
    pub options: Options,
    pub(crate) enc_chunk: Mlp,
    pub(crate) enc_imp: Mlp,
    pub(crate) enc_glob: Mlp,
    pub(crate) enc_edge: Vec<Mlp>,
    pub(crate) norm: HashMap<String, Norm>,
    pub(crate) fine_pre: Vec<FineRound>,
    pub(crate) fine_post: Vec<FineRound>,
    pub(crate) down: Vec<Vec<CoarseRound>>,
    pub(crate) up: Vec<Vec<CoarseRound>>,
    pub(crate) bond_head: Mlp,
    pub(crate) chunk_head: Mlp,
    pub(crate) imp_head: Mlp,
    pub(crate) energy_head: Mlp,
}

/// Every head output of one call (`ImpactOutcomeModel._heads`), PyTorch's names.
#[derive(Clone, Debug, Default)]
pub struct Outputs {
    pub bond_log_r: Vec<f32>,
    pub bond_break_logit: Vec<f32>,
    pub bond_log_q: Vec<f32>,
    pub bond_time_logits: Mat,
    pub bond_ddamage: Vec<f32>,
    pub chunk_out: Mat,
    pub imp_out: Mat,
    pub imp_perf_logit: Vec<f32>,
    pub energy_d_bond: f32,
    pub energy_frac_joint: f32,
    pub energy_e_elastic: f32,
    pub energy_ke_loss: f32,
    pub latent_global: Vec<f32>,
}

impl Default for Mat {
    fn default() -> Mat {
        Mat::zeros(0, 0)
    }
}

impl Outputs {
    /// From `(shape, values)` per output name (`outputs.npz`'s arrays).
    pub fn from_named(m: &HashMap<String, (Vec<usize>, Vec<f32>)>) -> Result<Outputs, String> {
        let get = |k: &str| m.get(k).ok_or_else(|| format!("outputs: no {k}"));
        let v = |k: &str| -> Result<Vec<f32>, String> { Ok(get(k)?.1.clone()) };
        let mat = |k: &str| -> Result<Mat, String> {
            let (s, d) = get(k)?;
            Ok(Mat { rows: s.first().copied().unwrap_or(0), cols: s.get(1).copied().unwrap_or(1), d: d.clone() })
        };
        let s = |k: &str| -> Result<f32, String> { get(k)?.1.first().copied().ok_or_else(|| format!("outputs: empty {k}")) };
        Ok(Outputs {
            bond_log_r: v("bond_log_r")?,
            bond_break_logit: v("bond_break_logit")?,
            bond_log_q: v("bond_log_q")?,
            bond_time_logits: mat("bond_time_logits")?,
            bond_ddamage: v("bond_ddamage")?,
            chunk_out: mat("chunk_out")?,
            imp_out: mat("imp_out")?,
            imp_perf_logit: v("imp_perf_logit")?,
            energy_d_bond: s("energy_d_bond")?,
            energy_frac_joint: s("energy_frac_joint")?,
            energy_e_elastic: s("energy_e_elastic")?,
            energy_ke_loss: s("energy_ke_loss")?,
            latent_global: v("latent_global")?,
        })
    }

    /// `(name, shape, values)` of every output, as `outputs.npz` holds them.
    pub fn named(&self) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
        let v = |x: &Vec<f32>| (vec![x.len()], x.clone());
        let m = |x: &Mat| (vec![x.rows, x.cols], x.d.clone());
        let s = |x: f32| (vec![1], vec![x]);
        let mut o = Vec::new();
        for (k, (sh, d)) in [
            ("bond_log_r", v(&self.bond_log_r)),
            ("bond_break_logit", v(&self.bond_break_logit)),
            ("bond_log_q", v(&self.bond_log_q)),
            ("bond_time_logits", m(&self.bond_time_logits)),
            ("bond_ddamage", v(&self.bond_ddamage)),
            ("chunk_out", m(&self.chunk_out)),
            ("imp_out", m(&self.imp_out)),
            ("imp_perf_logit", v(&self.imp_perf_logit)),
            ("energy_d_bond", s(self.energy_d_bond)),
            ("energy_frac_joint", s(self.energy_frac_joint)),
            ("energy_e_elastic", s(self.energy_e_elastic)),
            ("energy_ke_loss", s(self.energy_ke_loss)),
            ("latent_global", (vec![1, self.latent_global.len()], self.latent_global.clone())),
        ] {
            o.push((k, sh, d));
        }
        o
    }
}

/// `quat_rotate_inv`: world → body for a w-first unit quaternion.
fn quat_rotate_inv(q: &[f32], v: [f32; 3]) -> [f32; 3] {
    let (w, u) = (q[0], [-q[1], -q[2], -q[3]]);
    let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let t = cross(u, v).map(|x| 2.0 * x);
    let c = cross(u, t);
    [v[0] + w * t[0] + c[0], v[1] + w * t[1] + c[1], v[2] + w * t[2] + c[2]]
}

/// `WeightedEdgeConv.cal_ew`: per-edge pooling weights and the next level's node weights.
pub(crate) fn cal_ew(w: &[f32], edges: &[(usize, usize)]) -> (Vec<f32>, Vec<f32>) {
    let n = w.len();
    let mut deg = vec![0.0f32; n];
    for &(i, _) in edges {
        deg[i] += 1.0;
    }
    let normed: Vec<f32> = w.iter().zip(&deg).map(|(w, d)| w / d.max(1.0)).collect();
    let mut aggr = vec![0.0f32; n];
    for &(i, j) in edges {
        aggr[j] += normed[i];
    }
    for a in &mut aggr {
        *a += 1e-12;
    }
    (edges.iter().map(|&(i, j)| normed[i] / aggr[j]).collect(), aggr)
}

/// `WeightedEdgeConv.forward`: aggregating, `out[j] += x[i]·ew`; else `out[i] += x[j]·ew`.
fn edge_conv(x: &Mat, edges: &[(usize, usize)], ew: &[f32], aggregating: bool) -> Mat {
    let mut out = Mat::zeros(x.rows, x.cols);
    for (k, &(i, j)) in edges.iter().enumerate() {
        let (from, to) = if aggregating { (i, j) } else { (j, i) };
        for c in 0..x.cols {
            out.d[to * x.cols + c] += x.d[from * x.cols + c] * ew[k];
        }
    }
    out
}

/// Mean over rows (one call: every node belongs to it).
fn mean_rows(h: &Mat) -> Mat {
    let mut m = Mat::zeros(1, h.cols);
    for r in 0..h.rows {
        for (a, b) in m.d.iter_mut().zip(h.row(r)) {
            *a += b;
        }
    }
    let n = (h.rows.max(1)) as f32;
    for a in &mut m.d {
        *a /= n;
    }
    m
}

fn broadcast(g: &Mat, rows: usize) -> Mat {
    let mut m = Mat::zeros(rows, g.cols);
    for r in 0..rows {
        m.row_mut(r).copy_from_slice(g.row(0));
    }
    m
}

impl Model {
    pub fn new(w: &Weights, options: Options) -> Result<Model, String> {
        let c = w.config.clone();
        let fine = |p: &str| -> Result<FineRound, String> {
            Ok(FineRound {
                edge: c.edge_types.iter().map(|t| w.mlp(&format!("{p}.edge.{t}.edge_mlp"), true)).collect::<Result<_, _>>()?,
                node: w.mlp(&format!("{p}.node.node_mlp"), true)?,
                glob: if c.use_global { Some(w.mlp(&format!("{p}.glob.mlp"), true)?) } else { None },
            })
        };
        let coarse = |p: &str| -> Result<CoarseRound, String> {
            Ok(CoarseRound {
                mlp_edge: w.mlp(&format!("{p}.gmp.mlp_edge"), true)?,
                mlp_node: w.mlp(&format!("{p}.gmp.mlp_node"), true)?,
                glob: if c.use_global { Some(w.mlp(&format!("{p}.glob.mlp"), true)?) } else { None },
            })
        };
        let r = &c.rounds;
        let mut norm = HashMap::new();
        for k in ["chunk".to_string(), "imp".into(), "glob".into()].into_iter().chain(c.edge_types.iter().map(|t| format!("edge_{t}"))) {
            norm.insert(k.clone(), Norm { mean: w.vec(&format!("in_mean_{k}"))?, std: w.vec(&format!("in_std_{k}"))? });
        }
        Ok(Model {
            options,
            enc_chunk: w.mlp("enc_chunk", true)?,
            enc_imp: w.mlp("enc_imp", true)?,
            enc_glob: w.mlp("enc_glob", true)?,
            enc_edge: c.edge_types.iter().map(|t| w.mlp(&format!("enc_edge.{t}"), true)).collect::<Result<_, _>>()?,
            norm,
            fine_pre: (0..r[0]).map(|i| fine(&format!("fine_pre.{i}"))).collect::<Result<_, _>>()?,
            fine_post: (0..r[r.len() - 1]).map(|i| fine(&format!("fine_post.{i}"))).collect::<Result<_, _>>()?,
            down: (0..c.levels).map(|l| (0..r[1 + l]).map(|i| coarse(&format!("down.{l}.{i}"))).collect::<Result<_, _>>()).collect::<Result<_, _>>()?,
            up: (0..c.levels.saturating_sub(1)).map(|l| (0..r[c.levels + 1 + l]).map(|i| coarse(&format!("up.{l}.{i}"))).collect::<Result<_, _>>()).collect::<Result<_, _>>()?,
            bond_head: w.mlp("bond_head", false)?,
            chunk_head: w.mlp("chunk_head", false)?,
            imp_head: w.mlp("imp_head", false)?,
            energy_head: w.mlp("energy_head", false)?,
            config: c,
        })
    }

    fn encode(&self, key: &str, mlp: &Mlp, mut x: Mat) -> Mat {
        if self.config.input_norm {
            let n = &self.norm[key];
            assert_eq!(n.mean.len(), x.cols, "{key}: input width");
            for r in 0..x.rows {
                for (k, v) in x.row_mut(r).iter_mut().enumerate() {
                    *v = (*v - n.mean[k]) / n.std[k];
                }
            }
        }
        mlp.forward(&x)
    }

    /// `mlp([e, h[a], h[b], tail…])`: the edge MLP of a round, by concatenation or (with
    /// `node_projection`) with `h` projected once per node.
    fn edge_mlp(&self, mlp: &Mlp, e: &Mat, h: &Mat, a: &[usize], b: &[usize], tail: Option<&Mat>) -> Mat {
        if !self.options.node_projection {
            let (ha, hb) = (h.gather(a), h.gather(b));
            let mut parts = vec![e, &ha, &hb];
            if let Some(t) = tail {
                parts.push(t);
            }
            return mlp.forward(&Mat::hcat(&parts));
        }
        let mut widths = vec![e.cols, h.cols, h.cols];
        if let Some(t) = tail {
            widths.push(t.cols);
        }
        let s = SplitFirst::of(mlp, &widths);
        let mut pre = s.blocks[0].forward(e);
        let (pa, pb) = (s.blocks[1].forward(h), s.blocks[2].forward(h));
        for r in 0..pre.rows {
            let row = &mut pre.d[r * pa.cols..(r + 1) * pa.cols];
            for c in 0..pa.cols {
                row[c] += pa.d[a[r] * pa.cols + c] + pb.d[b[r] * pb.cols + c];
            }
        }
        if let Some(t) = tail {
            pre.add(&s.blocks[3].forward(t));
        }
        mlp.after_first(pre)
    }

    fn fine_round(&self, rnd: &FineRound, h: &Mat, e: &mut [Mat], g: &Mat, edges: &[(Vec<usize>, Vec<usize>)]) -> (Mat, Mat) {
        for (k, (src, dst)) in edges.iter().enumerate() {
            if src.is_empty() {
                continue;
            }
            let upd = self.edge_mlp(&rnd.edge[k], &e[k], h, src, dst, None);
            e[k].add(&upd);
        }
        let n = h.rows;
        let aggs: Vec<Mat> = edges.iter().enumerate().map(|(k, (_, dst))| e[k].scatter_sum(dst, n)).collect();
        let gb = broadcast(g, n);
        let mut parts: Vec<&Mat> = aggs.iter().collect();
        parts.push(h);
        parts.push(&gb);
        let mut h2 = rnd.node.forward(&Mat::hcat(&parts));
        h2.add(h);
        let g2 = self.glob(rnd.glob.as_ref(), g, &h2);
        (h2, g2)
    }

    fn glob(&self, mlp: Option<&Mlp>, g: &Mat, h: &Mat) -> Mat {
        match mlp {
            Some(m) => {
                let mut g2 = m.forward(&Mat::hcat(&[g, &mean_rows(h)]));
                g2.add(g);
                g2
            }
            None => g.clone(),
        }
    }

    /// `GraphMessagePassing.forward_fiber` and the global update.
    fn coarse_round(&self, rnd: &CoarseRound, h: &Mat, edges: &[(usize, usize)], fib: &Mat, g: &Mat) -> (Mat, Mat) {
        let (i, j): (Vec<usize>, Vec<usize>) = edges.iter().copied().unzip();
        let m = if edges.is_empty() {
            Mat::zeros(0, h.cols)
        } else if self.options.node_projection {
            // [fibre, h_i, h_j]: the fibre block first, then the two node blocks.
            let s = SplitFirst::of(&rnd.mlp_edge, &[fib.cols, h.cols, h.cols]);
            let mut pre = s.blocks[0].forward(fib);
            let (pa, pb) = (s.blocks[1].forward(h), s.blocks[2].forward(h));
            for r in 0..pre.rows {
                for c in 0..pa.cols {
                    pre.d[r * pa.cols + c] += pa.d[i[r] * pa.cols + c] + pb.d[j[r] * pb.cols + c];
                }
            }
            rnd.mlp_edge.after_first(pre)
        } else {
            rnd.mlp_edge.forward(&Mat::hcat(&[fib, &h.gather(&i), &h.gather(&j)]))
        };
        let agg = m.scatter_sum(&j, h.rows);
        let gb = broadcast(g, h.rows);
        let mut h2 = rnd.mlp_node.forward(&Mat::hcat(&[h, &agg, &gb]));
        h2.add(h);
        let g2 = self.glob(rnd.glob.as_ref(), g, &h2);
        (h2, g2)
    }

    /// The forward pass on call `ci` of `ev` (`hier` from `structure_hierarchies`).
    pub fn forward(&self, ev: &CallEvent, hier: &[Hierarchy], ci: usize) -> Result<Outputs, String> {
        let call = &ev.calls[ci];
        let cfg = &self.config;
        let arr = |n: &str| call.array(n).ok_or_else(|| format!("call {ci}: no {n}"));
        let b = cfg.variant == "B";
        let cat = |a: &str, q: &str| -> Result<Mat, String> {
            let x = mat_of(arr(a)?)?;
            if b {
                Ok(Mat::hcat(&[&x, &mat_of(arr(q)?)?]))
            } else {
                Ok(x)
            }
        };
        let cx = cat("chunk_feat", "chunk_qs_feat")?;
        let ix = cat("impactor_feat", "impactor_qs_feat")?;
        let (nc, ni) = (cx.rows, ix.rows);
        let n = nc + ni;
        let w = cfg.width;
        let mut h = Mat::zeros(n, w);
        let hc = self.encode("chunk", &self.enc_chunk, cx);
        let hi = self.encode("imp", &self.enc_imp, ix);
        h.d[..nc * w].copy_from_slice(&hc.d);
        h.d[nc * w..].copy_from_slice(&hi.d);
        let mut g = self.encode("glob", &self.enc_glob, mat_of(arr("global_feat")?)?);
        let mut edges = Vec::new();
        let mut e = Vec::new();
        for (k, t) in cfg.edge_types.iter().enumerate() {
            let src = idx_of(arr(&format!("{t}_src"))?)?;
            let dst = idx_of(arr(&format!("{t}_dst"))?)?;
            let x = if t == "bond" { cat("bond_feat", "bond_qs_feat")? } else { mat_of(arr(&format!("{t}_feat"))?)? };
            e.push(if src.is_empty() { Mat::zeros(0, w) } else { self.encode(&format!("edge_{t}"), &self.enc_edge[k], x) });
            edges.push((src, dst));
        }

        for rnd in &self.fine_pre {
            let (h2, g2) = self.fine_round(rnd, &h, &mut e, &g, &edges);
            h = h2;
            g = g2;
        }

        if cfg.levels > 0 {
            let ch = call_hierarchy(ev, hier, call, cfg.levels)?;
            let q = mat_of(arr("chunk_q_world")?)?;
            let l_scale = *call.scales.get("L").ok_or("call: no L scale")? as f32;
            let fiber = |pos: &Mat, chunk: &[usize], es: &[(usize, usize)]| -> Mat {
                let mut f = Mat::zeros(es.len(), 4);
                for (k, &(i, j)) in es.iter().enumerate() {
                    let d = [0, 1, 2].map(|c| pos.d[i * 3 + c] - pos.d[j * 3 + c]);
                    let r = quat_rotate_inv(q.row(chunk[i]), d).map(|x| x / l_scale);
                    f.row_mut(k).copy_from_slice(&[r[0], r[1], r[2], (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt()]);
                }
                f
            };
            let skip0 = h.clone();
            let mut hl = Mat { rows: nc, cols: w, d: h.d[..nc * w].to_vec() };
            let mut pos = mat_of(arr("chunk_pos")?)?;
            let mut wt = vec![1.0f32; nc];
            let (mut skips, mut ews, mut poss) = (Vec::new(), Vec::new(), Vec::new());
            for li in 0..cfg.levels {
                let gl = &ch.edges[li];
                let (ew, aggr) = cal_ew(&wt, gl);
                if li > 0 {
                    skips.push(hl.clone());
                }
                poss.push(pos.clone());
                hl = edge_conv(&hl, gl, &ew, true).gather(&ch.ids[li + 1]);
                pos = edge_conv(&pos, gl, &ew, true).gather(&ch.ids[li + 1]);
                wt = ch.ids[li + 1].iter().map(|&k| aggr[k]).collect();
                ews.push(ew);
                let fib = fiber(&pos, &ch.chunk[li + 1], &ch.edges[li + 1]);
                for rnd in &self.down[li] {
                    let (h2, g2) = self.coarse_round(rnd, &hl, &ch.edges[li + 1], &fib, &g);
                    hl = h2;
                    g = g2;
                }
            }
            for (k, li) in (0..cfg.levels).rev().enumerate() {
                let gl = &ch.edges[li];
                let n_prev = poss[li].rows;
                let mut un = Mat::zeros(n_prev, w);
                for (r, &i) in ch.ids[li + 1].iter().enumerate() {
                    un.row_mut(i).copy_from_slice(hl.row(r));
                }
                hl = edge_conv(&un, gl, &ews[li], false);
                if li > 0 {
                    let fib = fiber(&poss[li], &ch.chunk[li], gl);
                    for rnd in &self.up[k] {
                        let (h2, g2) = self.coarse_round(rnd, &hl, gl, &fib, &g);
                        hl = h2;
                        g = g2;
                    }
                    hl.add(&skips[li - 1]);
                }
            }
            h = skip0;
            for (a, b) in h.d[..nc * w].iter_mut().zip(&hl.d) {
                *a += b;
            }
        }

        for rnd in &self.fine_post {
            let (h2, g2) = self.fine_round(rnd, &h, &mut e, &g, &edges);
            h = h2;
            g = g2;
        }
        self.heads(call, &h, &e, &g, &edges, nc, ni)
    }

    #[allow(clippy::too_many_arguments)]
    fn heads(&self, call: &Call, h: &Mat, e: &[Mat], g: &Mat, edges: &[(Vec<usize>, Vec<usize>)], nc: usize, ni: usize) -> Result<Outputs, String> {
        let cfg = &self.config;
        let nt = cfg.n_time_bins;
        let w = cfg.width;
        let bk = cfg.edge_types.iter().position(|t| t == "bond").ok_or("no bond edges")?;
        let (src, dst) = &edges[bk];
        let nb = src.len() / 2;
        let fwd: Vec<usize> = (0..nb).map(|k| 2 * k).collect();
        let rev: Vec<usize> = (0..nb).map(|k| 2 * k + 1).collect();
        let (ef, er) = (e[bk].gather(&fwd), e[bk].gather(&rev));
        let hs = h.gather(&fwd.iter().map(|&k| src[k]).collect::<Vec<_>>());
        let hd = h.gather(&fwd.iter().map(|&k| dst[k]).collect::<Vec<_>>());
        let gb = broadcast(g, nb);
        let o1 = self.bond_head.forward(&Mat::hcat(&[&ef, &er, &hs, &hd, &gb]));
        let o2 = self.bond_head.forward(&Mat::hcat(&[&er, &ef, &hd, &hs, &gb]));
        let ow = o1.cols;
        let out: Vec<f32> = o1.d.iter().zip(&o2.d).map(|(a, b)| 0.5 * (a + b)).collect();
        let col = |c: usize| -> Vec<f32> { (0..nb).map(|r| out[r * ow + c]).collect() };
        let mut log_r = col(0);
        let b = cfg.variant == "B";
        if b {
            let q = mat_of(call.array("bond_qs_feat").ok_or("no bond_qs_feat")?)?;
            for (r, v) in log_r.iter_mut().enumerate() {
                *v += q.d[2 * r * q.cols];
            }
        }
        let mut times = Mat::zeros(nb, nt);
        for r in 0..nb {
            times.row_mut(r).copy_from_slice(&out[r * ow + 4..r * ow + 4 + nt]);
        }
        let hc = Mat { rows: nc, cols: w, d: h.d[..nc * w].to_vec() };
        let hi = Mat { rows: ni, cols: w, d: h.d[nc * w..].to_vec() };
        let mut c_out = self.chunk_head.forward(&Mat::hcat(&[&hc, &broadcast(g, nc)]));
        let i_raw = self.imp_head.forward(&Mat::hcat(&[&hi, &broadcast(g, ni)]));
        let mut i_out = Mat::zeros(ni, NODE_OUT);
        for r in 0..ni {
            i_out.row_mut(r).copy_from_slice(&i_raw.row(r)[..NODE_OUT]);
        }
        if b {
            for (m, k) in [(&mut c_out, "chunk_qs_feat"), (&mut i_out, "impactor_qs_feat")] {
                let q = mat_of(call.array(k).ok_or_else(|| format!("no {k}"))?)?;
                for r in 0..m.rows {
                    for c in 0..q.cols.min(6) {
                        m.d[r * NODE_OUT + c] += q.d[r * q.cols + c];
                    }
                }
            }
        }
        let en = self.energy_head.forward(g);
        Ok(Outputs {
            bond_log_r: log_r,
            bond_break_logit: col(1),
            bond_log_q: col(3),
            bond_time_logits: times,
            bond_ddamage: col(2).into_iter().map(softplus).collect(),
            chunk_out: c_out,
            imp_out: i_out,
            imp_perf_logit: (0..ni).map(|r| i_raw.row(r)[NODE_OUT]).collect(),
            energy_d_bond: softplus(en.d[0]),
            energy_frac_joint: sigmoid(en.d[1]),
            energy_e_elastic: softplus(en.d[2]),
            energy_ke_loss: en.d[3],
            latent_global: g.d.clone(),
        })
    }
}
