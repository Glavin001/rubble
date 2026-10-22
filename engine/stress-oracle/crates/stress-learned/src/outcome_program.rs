//! P2's model as a GPU program (P3): one call's forward pass compiled into a list of ops over
//! three flat buffers, which the Slang kernels (`stress-gpu/shaders/outcome_gnn.slang`, run
//! by `stress_gpu::outcome_gnn`) execute, and which `Program::run_cpu` interprets op by op
//! with the same arithmetic, so the program is checked against `outcome_model` without a GPU.
//!
//! Buffers: the `arena` (every input and activation, f32 rows), the packed `weights` (each MLP
//! k-major, so a lane reads its channel's column contiguously across rows), and the `table`
//! (u32: gather indices, CSR row offsets and `(arena offset, f32 weight)` entries, fibre
//! triples). Three kernels:
//! - `MLP`: per row, an input assembled from up to six segments (by row, gathered through
//!   the table, or a constant row), each optionally standardised, through Linear · SiLU ·
//!   Linear · [LayerNorm], times a scale, plus an optional residual row, written or
//!   accumulated. A *projection* segment adds a precomputed first-layer row to the
//!   pre-activation instead (the node-projection lever), and an op without a second layer
//!   is that projection (`W_block·h` per node);
//! - `CSR`: `out[r] (+)= Σ_k w_k · arena[o_k ..]`, every sum of the model (edge sums, BSMS
//!   pooling and unpooling with their weights, means, skips);
//! - `FIBRE`: a coarse edge's `(R_bodyᵀ(p_i − p_j)/L, |·|)`.
//!
//! Host-side, from the inputs only: the hierarchy, the pooling weights (`cal_ew`), the gather
//! tables, the variant-B residual rows; after the run, the heads' elementwise maps
//! (softplus, sigmoid) on the read-back outputs.

use crate::outcome_calls::CallEvent;
use crate::outcome_model::{cal_ew, call_hierarchy, idx_of, mat_of, sigmoid, softplus, CallHierarchy, Hierarchy, Mat, Mlp, Model, Outputs, NODE_OUT};

pub const PARAM_WORDS: usize = 64;
pub const NONE: u32 = u32::MAX;
pub const THREADS: usize = 128;
pub const MAX_GROUPS_X: usize = 65_535;
/// Largest staged MLP input (six segments of width 64).
pub const MAX_K: usize = 384;

pub const KERNEL_MLP: u32 = 0;
pub const KERNEL_CSR: u32 = 1;
pub const KERNEL_FIBRE: u32 = 2;

/// Segment modes of the MLP kernel.
pub const SEG_ROW: u32 = 0;
pub const SEG_GATHER: u32 = 1;
pub const SEG_CONST: u32 = 2;
pub const SEG_PROJ_GATHER: u32 = 3;
pub const SEG_PROJ_ROW: u32 = 4;
pub const MAX_SEGS: usize = 6;

/// MLP parameter words (after the common header `kernel, rows, grid_x, lanes`).
pub mod mlp_p {
    pub const N_SEG: usize = 4;
    /// Segment `s` at `SEG0 + 5 s`: base, idx, width, mode, norm.
    pub const SEG0: usize = 8;
    pub const W1: usize = 40;
    pub const K_IN: usize = 41;
    pub const HIDDEN: usize = 42;
    pub const OUT: usize = 43;
    pub const W2: usize = 44;
    pub const LN: usize = 45;
    pub const OUT_BASE: usize = 46;
    pub const OUT_STRIDE: usize = 47;
    pub const RES_BASE: usize = 48;
    pub const RES_STRIDE: usize = 49;
    pub const FLAGS: usize = 50;
    pub const SCALE: usize = 51;
    pub const B1: usize = 52;
    pub const B2: usize = 53;
}

/// CSR parameter words.
pub mod csr_p {
    pub const WIDTH: usize = 4;
    pub const OFFSETS: usize = 5;
    pub const ENTRIES: usize = 6;
    pub const OUT_BASE: usize = 7;
    pub const OUT_STRIDE: usize = 8;
    pub const FLAGS: usize = 9;
}

/// FIBRE parameter words.
pub mod fibre_p {
    pub const POS: usize = 4;
    pub const TRIPLES: usize = 5;
    pub const Q: usize = 6;
    pub const OUT_BASE: usize = 7;
    // Each edge's (i, j, chunk of i, L as f32 bits) at TRIPLES + 4 k: divided by its
    // call's length scale, as the reference does.
}

pub const FLAG_ACCUMULATE: u32 = 1;

/// One dispatch: `groups` workgroups of `THREADS` threads, `params` its uniform.
#[derive(Clone, Debug)]
pub struct Op {
    pub label: String,
    pub params: [u32; PARAM_WORDS],
    pub groups: (u32, u32),
}

impl Op {
    pub fn kernel(&self) -> u32 {
        self.params[0]
    }
    pub fn rows(&self) -> usize {
        self.params[1] as usize
    }
}

/// A packed MLP (one hidden layer): offsets into the weights.
#[derive(Clone, Copy, Debug)]
struct PackedMlp {
    /// k-major `k_in × hidden`, then the bias at `b1`.
    w1: u32,
    b1: u32,
    k_in: usize,
    hidden: usize,
    /// k-major `hidden × out`, then the bias at `b2`.
    w2: u32,
    b2: u32,
    out: usize,
    ln: u32,
}

/// Where one call's outputs are in the arena.
#[derive(Clone, Debug, Default)]
pub struct OutputLayout {
    pub nb: usize,
    pub nc: usize,
    pub ni: usize,
    pub bond: usize,
    pub chunk: usize,
    pub imp: usize,
    pub energy: usize,
    pub glob: usize,
    pub width: usize,
    pub n_time_bins: usize,
    /// Per call: chunks, impactors, bonds (the batch's rows are in call order).
    pub calls: Vec<(usize, usize, usize)>,
}

/// One call's forward pass as ops over the three buffers.
#[derive(Clone, Debug)]
pub struct Program {
    /// Initial arena: the inputs, zero elsewhere.
    pub arena: Vec<f32>,
    pub weights: Vec<f32>,
    pub table: Vec<u32>,
    pub ops: Vec<Op>,
    pub layout: OutputLayout,
}

struct Builder {
    arena: Vec<f32>,
    weights: Vec<f32>,
    table: Vec<u32>,
    ops: Vec<Op>,
    lanes: usize,
}

fn groups(rows: usize, per_group: usize) -> (u32, u32) {
    let n = rows.div_ceil(per_group).max(1);
    if n <= MAX_GROUPS_X {
        (n as u32, 1)
    } else {
        (MAX_GROUPS_X as u32, n.div_ceil(MAX_GROUPS_X) as u32)
    }
}

/// An MLP input segment.
#[derive(Clone, Copy, Debug)]
struct Seg {
    base: usize,
    width: usize,
    mode: u32,
    /// Gather table offset (`SEG_GATHER`, `SEG_PROJ_GATHER`) or the row (`SEG_CONST`).
    idx: u32,
    /// Weights offset of `mean ‖ std` (width each), or `NONE`.
    norm: u32,
}

impl Seg {
    fn row(base: usize, width: usize) -> Seg {
        Seg { base, width, mode: SEG_ROW, idx: NONE, norm: NONE }
    }
    fn gather(base: usize, width: usize, idx: u32) -> Seg {
        Seg { base, width, mode: SEG_GATHER, idx, norm: NONE }
    }
    fn konst(base: usize, width: usize, row: u32) -> Seg {
        Seg { base, width, mode: SEG_CONST, idx: row, norm: NONE }
    }
}

struct MlpOut {
    base: usize,
    stride: usize,
    res: Option<(usize, usize)>,
    accumulate: bool,
    scale: f32,
}

impl MlpOut {
    fn to(base: usize, stride: usize) -> MlpOut {
        MlpOut { base, stride, res: None, accumulate: false, scale: 1.0 }
    }
    /// In place with the residual: `x ← x + y`.
    fn residual(base: usize, stride: usize) -> MlpOut {
        MlpOut { base, stride, res: Some((base, stride)), accumulate: false, scale: 1.0 }
    }
}

impl Builder {
    fn alloc(&mut self, rows: usize, width: usize) -> usize {
        let b = self.arena.len();
        self.arena.resize(b + rows * width, 0.0);
        b
    }
    fn input(&mut self, m: &Mat) -> usize {
        let b = self.arena.len();
        self.arena.extend_from_slice(&m.d);
        b
    }
    fn table(&mut self, v: &[u32]) -> u32 {
        let b = self.table.len() as u32;
        self.table.extend_from_slice(v);
        b
    }
    fn wvec(&mut self, v: &[f32]) -> u32 {
        let b = self.weights.len() as u32;
        self.weights.extend_from_slice(v);
        b
    }
    fn norm(&mut self, mean: &[f32], std: &[f32]) -> u32 {
        let b = self.wvec(mean);
        self.wvec(std);
        b
    }

    fn pack(&mut self, m: &Mlp) -> PackedMlp {
        assert_eq!(m.layers.len(), 2, "the GPU program packs one hidden layer (mlp_hidden_layers 1)");
        let (l1, l2) = (&m.layers[0], &m.layers[1]);
        let (k_in, hidden, out) = (l1.inp, l1.out, l2.out);
        let mut w1 = vec![0.0; k_in * hidden];
        for c in 0..hidden {
            for k in 0..k_in {
                w1[k * hidden + c] = l1.w[c * k_in + k];
            }
        }
        let w1o = self.wvec(&w1);
        let b1 = self.wvec(&l1.b);
        let mut w2 = vec![0.0; hidden * out];
        for c in 0..out {
            for k in 0..hidden {
                w2[k * out + c] = l2.w[c * hidden + k];
            }
        }
        let w2o = self.wvec(&w2);
        let b2 = self.wvec(&l2.b);
        let ln = match &m.ln {
            Some((g, b)) => {
                let o = self.wvec(g);
                self.wvec(b);
                o
            }
            None => NONE,
        };
        PackedMlp { w1: w1o, b1, k_in, hidden, w2: w2o, b2, out, ln }
    }

    fn header(&self, kernel: u32, rows: usize, per_group: usize) -> ([u32; PARAM_WORDS], (u32, u32)) {
        let g = groups(rows, per_group);
        let mut p = [0u32; PARAM_WORDS];
        p[0] = kernel;
        p[1] = rows as u32;
        p[2] = g.0;
        p[3] = self.lanes as u32;
        (p, g)
    }

    /// `out ← scale·mlp(segments) (+ residual)`. With `proj`, the segments listed as
    /// projection (`SEG_PROJ_*`) contribute their precomputed first-layer rows; the staged
    /// segments then use the first-layer block starting at `w1_col0` (input column).
    fn mlp(&mut self, label: &str, rows: usize, m: &PackedMlp, segs: &[Seg], out: MlpOut, w1_col0: usize) {
        if rows == 0 {
            return;
        }
        assert!(segs.len() <= MAX_SEGS);
        let (mut p, g) = self.header(KERNEL_MLP, rows, THREADS / self.lanes);
        use mlp_p::*;
        p[N_SEG] = segs.len() as u32;
        let mut k = 0;
        for (s, sg) in segs.iter().enumerate() {
            let o = SEG0 + 5 * s;
            p[o..o + 5].copy_from_slice(&[sg.base as u32, sg.idx, sg.width as u32, sg.mode, sg.norm]);
            if sg.mode < SEG_PROJ_GATHER {
                k += sg.width;
            } else {
                assert_eq!(sg.width, m.hidden, "{label}: a projection row is a pre-activation");
            }
        }
        assert!(k <= m.k_in.max(w1_col0 + k) && k <= MAX_K && m.hidden <= self.lanes && m.out <= self.lanes, "{label}: k {k}, hidden {}, out {}", m.hidden, m.out);
        p[W1] = m.w1 + (w1_col0 * m.hidden) as u32;
        p[K_IN] = k as u32;
        p[HIDDEN] = m.hidden as u32;
        p[OUT] = m.out as u32;
        p[W2] = m.w2;
        p[B1] = m.b1;
        p[B2] = m.b2;
        p[LN] = m.ln;
        if m.ln != NONE {
            assert_eq!(m.out, self.lanes, "{label}: LayerNorm spans every lane");
        }
        p[OUT_BASE] = out.base as u32;
        p[OUT_STRIDE] = out.stride as u32;
        let (rb, rs) = out.res.map(|(b, s)| (b as u32, s as u32)).unwrap_or((NONE, 0));
        p[RES_BASE] = rb;
        p[RES_STRIDE] = rs;
        p[FLAGS] = if out.accumulate { FLAG_ACCUMULATE } else { 0 };
        p[SCALE] = out.scale.to_bits();
        self.ops.push(Op { label: label.to_string(), params: p, groups: g });
    }

    /// `out[r] ← W1[block]·src[r]` for every row: the first-layer projection of a node block.
    fn projection(&mut self, label: &str, rows: usize, m: &PackedMlp, col0: usize, src: usize, width: usize, out: usize) {
        if rows == 0 {
            return;
        }
        let (mut p, g) = self.header(KERNEL_MLP, rows, THREADS / self.lanes);
        use mlp_p::*;
        p[N_SEG] = 1;
        p[SEG0..SEG0 + 5].copy_from_slice(&[src as u32, NONE, width as u32, SEG_ROW, NONE]);
        p[W1] = m.w1 + (col0 * m.hidden) as u32;
        p[K_IN] = width as u32;
        p[HIDDEN] = m.hidden as u32;
        p[OUT] = m.hidden as u32;
        p[W2] = NONE;
        p[B1] = NONE;
        p[B2] = NONE;
        p[LN] = NONE;
        p[OUT_BASE] = out as u32;
        p[OUT_STRIDE] = m.hidden as u32;
        p[RES_BASE] = NONE;
        p[SCALE] = 1.0f32.to_bits();
        self.ops.push(Op { label: label.to_string(), params: p, groups: g });
    }

    /// `out[r] (+)= Σ_k w_k · arena[o_k ..]`; `rows[r]` lists row `r`'s `(offset, weight)`.
    fn csr(&mut self, label: &str, rows: &[Vec<(usize, f32)>], width: usize, out: usize, accumulate: bool) {
        if rows.is_empty() {
            return;
        }
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut entries = Vec::new();
        offsets.push(0u32);
        for r in rows {
            for &(o, w) in r {
                entries.push(o as u32);
                entries.push(w.to_bits());
            }
            offsets.push((entries.len() / 2) as u32);
        }
        let ot = self.table(&offsets);
        let et = self.table(&entries);
        let (mut p, g) = self.header(KERNEL_CSR, rows.len(), THREADS / self.lanes);
        assert!(width <= self.lanes);
        p[csr_p::WIDTH] = width as u32;
        p[csr_p::OFFSETS] = ot;
        p[csr_p::ENTRIES] = et;
        p[csr_p::OUT_BASE] = out as u32;
        p[csr_p::OUT_STRIDE] = width as u32;
        p[csr_p::FLAGS] = if accumulate { FLAG_ACCUMULATE } else { 0 };
        self.ops.push(Op { label: label.to_string(), params: p, groups: g });
    }

    /// `l_scale[k]`: edge `k`'s call's length scale.
    fn fibre(&mut self, label: &str, edges: &[(usize, usize)], chunk: &[usize], pos: usize, q: usize, out: usize, l_scale: &[f32]) {
        if edges.is_empty() {
            return;
        }
        let tri: Vec<u32> = edges.iter().zip(l_scale).flat_map(|(&(i, j), l)| [i as u32, j as u32, chunk[i] as u32, l.to_bits()]).collect();
        let t = self.table(&tri);
        let (mut p, g) = self.header(KERNEL_FIBRE, edges.len(), THREADS);
        p[fibre_p::POS] = pos as u32;
        p[fibre_p::TRIPLES] = t;
        p[fibre_p::Q] = q as u32;
        p[fibre_p::OUT_BASE] = out as u32;
        self.ops.push(Op { label: label.to_string(), params: p, groups: g });
    }

    fn mean(&mut self, label: &str, src: usize, rows: usize, width: usize, out: usize) {
        let inv = 1.0 / rows.max(1) as f32;
        let row: Vec<(usize, f32)> = (0..rows).map(|r| (src + r * width, inv)).collect();
        self.csr(label, &[row], width, out, false);
    }
}

/// Every packed MLP of the model, in `Model` order.
struct PackedModel {
    enc_chunk: PackedMlp,
    enc_imp: PackedMlp,
    enc_glob: PackedMlp,
    enc_edge: Vec<PackedMlp>,
    fine_pre: Vec<(Vec<PackedMlp>, PackedMlp, Option<PackedMlp>)>,
    fine_post: Vec<(Vec<PackedMlp>, PackedMlp, Option<PackedMlp>)>,
    down: Vec<Vec<(PackedMlp, PackedMlp, Option<PackedMlp>)>>,
    up: Vec<Vec<(PackedMlp, PackedMlp, Option<PackedMlp>)>>,
    bond_head: PackedMlp,
    chunk_head: PackedMlp,
    imp_head: PackedMlp,
    energy_head: PackedMlp,
}

fn pack_model(b: &mut Builder, m: &Model) -> PackedModel {
    let fine = |b: &mut Builder, r: &crate::outcome_model::FineRound| (r.edge.iter().map(|x| b.pack(x)).collect::<Vec<_>>(), b.pack(&r.node), r.glob.as_ref().map(|x| b.pack(x)));
    let coarse = |b: &mut Builder, r: &crate::outcome_model::CoarseRound| (b.pack(&r.mlp_edge), b.pack(&r.mlp_node), r.glob.as_ref().map(|x| b.pack(x)));
    PackedModel {
        enc_chunk: b.pack(&m.enc_chunk),
        enc_imp: b.pack(&m.enc_imp),
        enc_glob: b.pack(&m.enc_glob),
        enc_edge: m.enc_edge.iter().map(|x| b.pack(x)).collect(),
        fine_pre: m.fine_pre.iter().map(|r| fine(b, r)).collect(),
        fine_post: m.fine_post.iter().map(|r| fine(b, r)).collect(),
        down: m.down.iter().map(|l| l.iter().map(|r| coarse(b, r)).collect()).collect(),
        up: m.up.iter().map(|l| l.iter().map(|r| coarse(b, r)).collect()).collect(),
        bond_head: b.pack(&m.bond_head),
        chunk_head: b.pack(&m.chunk_head),
        imp_head: b.pack(&m.imp_head),
        energy_head: b.pack(&m.energy_head),
    }
}

/// One call of a batch: the event, its structures' hierarchies, the call's index.
pub type BatchCall<'a> = (&'a CallEvent, &'a [Hierarchy], usize);

/// Compile call `ci` of `ev` (`hier` from `outcome_model::structure_hierarchies`).
pub fn compile(model: &Model, ev: &CallEvent, hier: &[Hierarchy], ci: usize) -> Result<Program, String> {
    compile_batch(model, &[(ev, hier, ci)])
}

fn cat_rows(ms: &[Mat]) -> Mat {
    let cols = ms.iter().map(|m| m.cols).find(|&c| c > 0).unwrap_or(0);
    let mut out = Mat::zeros(0, cols);
    for m in ms {
        if m.rows > 0 {
            assert_eq!(m.cols, cols, "batched rows of different widths");
            out.d.extend_from_slice(&m.d);
            out.rows += m.rows;
        }
    }
    out
}

/// Compile several calls into one program (P3's batch layout): node rows are every call's
/// chunks, in call order, then every call's impactors; edges, bonds and coarse levels are
/// concatenated in call order; each call keeps its own global latent row and means.
pub fn compile_batch(model: &Model, calls: &[BatchCall]) -> Result<Program, String> {
    let cfg = &model.config;
    let w = cfg.width;
    let lanes = w;
    assert!(lanes.is_power_of_two() && (32..=64).contains(&lanes), "width {w}: the program runs widths 32 and 64");
    assert!(!calls.is_empty());
    let lever = model.options.node_projection;
    let vb = cfg.variant == "B";
    let ng = calls.len();
    let mut b = Builder { arena: Vec::new(), weights: Vec::new(), table: Vec::new(), ops: Vec::new(), lanes };
    let pm = pack_model(&mut b, model);

    // ---- inputs, concatenated
    let get = |k: usize, n: &str| -> Result<Mat, String> {
        let (ev, _, ci) = calls[k];
        mat_of(ev.calls[ci].array(n).ok_or_else(|| format!("call {ci}: no {n}"))?)
    };
    let per = |n: &str| -> Result<Vec<Mat>, String> { (0..ng).map(|k| get(k, n)).collect() };
    let cfs = per("chunk_feat")?;
    let ifs = per("impactor_feat")?;
    let ncs: Vec<usize> = cfs.iter().map(|m| m.rows).collect();
    let nis: Vec<usize> = ifs.iter().map(|m| m.rows).collect();
    let (nc, ni) = (ncs.iter().sum::<usize>(), nis.iter().sum::<usize>());
    let n = nc + ni;
    let c_off: Vec<usize> = (0..ng).map(|k| ncs[..k].iter().sum()).collect();
    let i_off: Vec<usize> = (0..ng).map(|k| nc + nis[..k].iter().sum::<usize>()).collect();
    // node index of call k's local node `x` (chunks then impactors locally)
    let node = |k: usize, x: usize| if x < ncs[k] { c_off[k] + x } else { i_off[k] + x - ncs[k] };
    let cf = cat_rows(&cfs);
    let iff = cat_rows(&ifs);
    let gf = cat_rows(&per("global_feat")?);
    let cqs = per("chunk_qs_feat")?;
    let iqs = per("impactor_qs_feat")?;
    let cq = cat_rows(&cqs);
    let iq = cat_rows(&iqs);
    let cf_b = b.input(&cf);
    let cq_b = b.input(&cq);
    let if_b = b.input(&iff);
    let iq_b = b.input(&iq);
    let gf_b = b.input(&gf);
    let nrm = |key: &str| &model.norm[key];
    let seg_norm = |b: &mut Builder, key: &str, c0: usize, width: usize| -> u32 {
        if !cfg.input_norm {
            return NONE;
        }
        let n = nrm(key);
        b.norm(&n.mean[c0..c0 + width], &n.std[c0..c0 + width])
    };
    // the call of each node; per call its node rows (for the means)
    let mut node_call = vec![0u32; n];
    for k in 0..ng {
        for x in 0..ncs[k] + nis[k] {
            node_call[node(k, x)] = k as u32;
        }
    }
    let node_call_t = b.table(&node_call);
    let chunk_call_t = b.table(&node_call[..nc]);
    let imp_call_t = b.table(&node_call[nc..]);

    let h = b.alloc(n, w);
    let g = b.alloc(ng, w);
    {
        let mut s = vec![Seg { norm: seg_norm(&mut b, "chunk", 0, cf.cols), ..Seg::row(cf_b, cf.cols) }];
        if vb {
            s.push(Seg { norm: seg_norm(&mut b, "chunk", cf.cols, cq.cols), ..Seg::row(cq_b, cq.cols) });
        }
        b.mlp("enc_chunk", nc, &pm.enc_chunk, &s, MlpOut::to(h, w), 0);
        let mut s = vec![Seg { norm: seg_norm(&mut b, "imp", 0, iff.cols), ..Seg::row(if_b, iff.cols) }];
        if vb {
            s.push(Seg { norm: seg_norm(&mut b, "imp", iff.cols, iq.cols), ..Seg::row(iq_b, iq.cols) });
        }
        b.mlp("enc_imp", ni, &pm.enc_imp, &s, MlpOut::to(h + nc * w, w), 0);
        let s = [Seg { norm: seg_norm(&mut b, "glob", 0, gf.cols), ..Seg::row(gf_b, gf.cols) }];
        b.mlp("enc_glob", ng, &pm.enc_glob, &s, MlpOut::to(g, w), 0);
    }
    // per edge type: src, dst (node rows), their tables, the features' base, the edge's call
    let mut edges: Vec<(Vec<usize>, Vec<usize>, u32, u32, usize)> = Vec::new();
    let mut bond_call: Vec<u32> = Vec::new();
    for (k, t) in cfg.edge_types.iter().enumerate() {
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        let mut xs = Vec::new();
        let mut qs = Vec::new();
        for c in 0..ng {
            let (ev, _, ci) = calls[c];
            let call = &ev.calls[ci];
            let s0 = idx_of(call.array(&format!("{t}_src")).ok_or("no src")?)?;
            let d0 = idx_of(call.array(&format!("{t}_dst")).ok_or("no dst")?)?;
            if t == "bond" {
                bond_call.extend(std::iter::repeat_n(c as u32, s0.len() / 2));
            }
            if !s0.is_empty() {
                xs.push(get(c, &format!("{t}_feat"))?);
                if t == "bond" && vb {
                    qs.push(get(c, "bond_qs_feat")?);
                }
            }
            src.extend(s0.into_iter().map(|x| node(c, x)));
            dst.extend(d0.into_iter().map(|x| node(c, x)));
        }
        let st = b.table(&src.iter().map(|&x| x as u32).collect::<Vec<_>>());
        let dt = b.table(&dst.iter().map(|&x| x as u32).collect::<Vec<_>>());
        let ne = src.len();
        let e = b.alloc(ne, w);
        if ne > 0 {
            let x = cat_rows(&xs);
            let xb = b.input(&x);
            let key = format!("edge_{t}");
            let mut s = vec![Seg { norm: seg_norm(&mut b, &key, 0, x.cols), ..Seg::row(xb, x.cols) }];
            if t == "bond" && vb {
                let q = cat_rows(&qs);
                let qb = b.input(&q);
                s.push(Seg { norm: seg_norm(&mut b, &key, x.cols, q.cols), ..Seg::row(qb, q.cols) });
            }
            b.mlp(&format!("enc_edge.{t}"), ne, &pm.enc_edge[k], &s, MlpOut::to(e, w), 0);
        }
        edges.push((src, dst, st, dt, e));
    }
    let max_e = edges.iter().map(|x| x.0.len()).max().unwrap_or(0);
    let aggs: Vec<usize> = (0..edges.len()).map(|_| b.alloc(n, w)).collect();
    let gmean = b.alloc(ng, w);

    // the hierarchy, concatenated: level-0 ids are global chunk rows, level-l ids offset by
    // the earlier calls' level-l counts
    let levels = cfg.levels;
    let mut ch = CallHierarchy { edges: vec![Vec::new(); levels + 1], ids: vec![Vec::new(); levels + 1], chunk: vec![Vec::new(); levels + 1] };
    let mut level_call: Vec<Vec<u32>> = vec![Vec::new(); levels + 1];
    if levels > 0 {
        let mut count = vec![0usize; levels + 1];
        for k in 0..ng {
            let (ev, hier, ci) = calls[k];
            let one = call_hierarchy(ev, hier, &ev.calls[ci], levels)?;
            count[0] = c_off[k];
            for l in 0..=levels {
                let off = count[l];
                ch.edges[l].extend(one.edges[l].iter().map(|&(a, b)| (a + off, b + off)));
                if l > 0 {
                    let prev = count[l - 1];
                    ch.ids[l].extend(one.ids[l].iter().map(|&i| i + prev));
                    ch.chunk[l].extend(one.chunk[l].iter().map(|&c| c + c_off[k]));
                    level_call[l].extend(std::iter::repeat_n(k as u32, one.ids[l].len()));
                }
            }
            for l in 1..=levels {
                count[l] += one.ids[l].len();
            }
        }
        ch.chunk[0] = (0..nc).collect();
        level_call[0] = node_call[..nc].to_vec();
    }
    let n_l: Vec<usize> = (0..=levels).map(|l| if l == 0 { nc } else { ch.ids[l].len() }).collect();
    let level_call_t: Vec<u32> = level_call.iter().map(|lc| b.table(lc)).collect();
    let max_rows = n_l.iter().copied().chain([n]).max().unwrap();
    let (pa, pb) = (b.alloc(max_rows, w), b.alloc(max_rows, w));

    // per-call means of `rows` rows at `src` whose calls are `calls_of`
    let call_mean = |b: &mut Builder, label: &str, src: usize, calls_of: &[u32]| {
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); ng];
        for (r, &c) in calls_of.iter().enumerate() {
            rows[c as usize].push(r);
        }
        let ent: Vec<Vec<(usize, f32)>> = rows
            .iter()
            .map(|rs| {
                let inv = 1.0 / rs.len().max(1) as f32;
                rs.iter().map(|&r| (src + r * w, inv)).collect()
            })
            .collect();
        b.csr(label, &ent, w, gmean, false);
    };

    let fine_round = |b: &mut Builder, label: &str, r: &(Vec<PackedMlp>, PackedMlp, Option<PackedMlp>)| {
        for (k, (src, _, st, dt, e)) in edges.iter().enumerate() {
            let ne = src.len();
            if ne == 0 {
                continue;
            }
            let m = &r.0[k];
            let lbl = format!("{label}.edge.{}", cfg.edge_types[k]);
            // The projection pays off only where an edge type has more edges than the
            // graph has nodes (it moves the node blocks' work from per edge to per node).
            if lever && ne > n {
                b.projection(&format!("{lbl}.proj_src"), n, m, w, h, w, pa);
                b.projection(&format!("{lbl}.proj_dst"), n, m, 2 * w, h, w, pb);
                let s = [Seg::row(*e, w), Seg { mode: SEG_PROJ_GATHER, ..Seg::gather(pa, w, *st) }, Seg { mode: SEG_PROJ_GATHER, ..Seg::gather(pb, w, *dt) }];
                b.mlp(&lbl, ne, m, &s, MlpOut::residual(*e, w), 0);
            } else {
                let s = [Seg::row(*e, w), Seg::gather(h, w, *st), Seg::gather(h, w, *dt)];
                b.mlp(&lbl, ne, m, &s, MlpOut::residual(*e, w), 0);
            }
        }
        for (k, (_, dst, _, _, e)) in edges.iter().enumerate() {
            if dst.is_empty() {
                continue;
            }
            let mut rows = vec![Vec::new(); n];
            for (ei, &d) in dst.iter().enumerate() {
                rows[d].push((e + ei * w, 1.0));
            }
            b.csr(&format!("{label}.sum.{}", cfg.edge_types[k]), &rows, w, aggs[k], false);
        }
        let mut s: Vec<Seg> = aggs.iter().map(|&a| Seg::row(a, w)).collect();
        s.push(Seg::row(h, w));
        s.push(Seg::gather(g, w, node_call_t));
        b.mlp(&format!("{label}.node"), n, &r.1, &s, MlpOut::residual(h, w), 0);
        if let Some(gm) = &r.2 {
            call_mean(b, &format!("{label}.mean"), h, &node_call);
            b.mlp(&format!("{label}.glob"), ng, gm, &[Seg::row(g, w), Seg::row(gmean, w)], MlpOut::residual(g, w), 0);
        }
    };
    for (i, r) in pm.fine_pre.iter().enumerate() {
        fine_round(&mut b, &format!("fine_pre.{i}"), r);
    }

    if levels > 0 {
        let q_b = b.input(&cat_rows(&per("chunk_q_world")?));
        let pos0 = b.input(&cat_rows(&per("chunk_pos")?));
        // each chunk's call's length scale (fibres of a coarse edge use its chunk's call)
        let l_of_call: Vec<f32> = calls.iter().map(|&(ev, _, ci)| ev.calls[ci].scales.get("L").copied().ok_or("call: no L scale").map(|x| x as f32)).collect::<Result<_, _>>()?;
        let max_ce = ch.edges.iter().map(Vec::len).max().unwrap_or(0);
        let msg = b.alloc(max_ce.max(max_e), w);
        let cagg = b.alloc(max_rows, w);
        let mut hl = vec![h];
        let mut pos = vec![pos0];
        let mut fib = vec![0usize];
        let mut ews = Vec::new();
        let mut wt = vec![1.0f32; nc];
        for l in 1..=levels {
            hl.push(b.alloc(n_l[l], w));
            pos.push(b.alloc(n_l[l], 3));
            fib.push(b.alloc(ch.edges[l].len(), 4));
        }
        let coarse_round = |b: &mut Builder, label: &str, r: &(PackedMlp, PackedMlp, Option<PackedMlp>), x: usize, l: usize, fb: usize| {
            let (rows, es) = (n_l[l], &ch.edges[l]);
            if !es.is_empty() {
                let it: Vec<u32> = es.iter().map(|e| e.0 as u32).collect();
                let jt: Vec<u32> = es.iter().map(|e| e.1 as u32).collect();
                let (it, jt) = (b.table(&it), b.table(&jt));
                if lever && es.len() > rows {
                    b.projection(&format!("{label}.proj_i"), rows, &r.0, 4, x, w, pa);
                    b.projection(&format!("{label}.proj_j"), rows, &r.0, 4 + w, x, w, pb);
                    let s = [Seg::row(fb, 4), Seg { mode: SEG_PROJ_GATHER, ..Seg::gather(pa, w, it) }, Seg { mode: SEG_PROJ_GATHER, ..Seg::gather(pb, w, jt) }];
                    b.mlp(&format!("{label}.edge"), es.len(), &r.0, &s, MlpOut::to(msg, w), 0);
                } else {
                    let s = [Seg::row(fb, 4), Seg::gather(x, w, it), Seg::gather(x, w, jt)];
                    b.mlp(&format!("{label}.edge"), es.len(), &r.0, &s, MlpOut::to(msg, w), 0);
                }
            }
            let mut agg = vec![Vec::new(); rows];
            for (k, &(_, j)) in es.iter().enumerate() {
                agg[j].push((msg + k * w, 1.0));
            }
            b.csr(&format!("{label}.sum"), &agg, w, cagg, false);
            b.mlp(&format!("{label}.node"), rows, &r.1, &[Seg::row(x, w), Seg::row(cagg, w), Seg::gather(g, w, level_call_t[l])], MlpOut::residual(x, w), 0);
            if let Some(gm) = &r.2 {
                call_mean(b, &format!("{label}.mean"), x, &level_call[l]);
                b.mlp(&format!("{label}.glob"), ng, gm, &[Seg::row(g, w), Seg::row(gmean, w)], MlpOut::residual(g, w), 0);
            }
        };
        for li in 0..levels {
            let gl = &ch.edges[li];
            let (ew, aggr) = cal_ew(&wt, gl);
            let ids = &ch.ids[li + 1];
            let mut pool: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ids.len()];
            let mut row_of = vec![usize::MAX; n_l[li]];
            for (r, &j) in ids.iter().enumerate() {
                row_of[j] = r;
            }
            let mut ppos = pool.clone();
            for (k, &(i, j)) in gl.iter().enumerate() {
                if row_of[j] != usize::MAX {
                    pool[row_of[j]].push((hl[li] + i * w, ew[k]));
                    ppos[row_of[j]].push((pos[li] + i * 3, ew[k]));
                }
            }
            b.csr(&format!("pool.{li}"), &pool, w, hl[li + 1], false);
            b.csr(&format!("pool_pos.{li}"), &ppos, 3, pos[li + 1], false);
            wt = ids.iter().map(|&k| aggr[k]).collect();
            ews.push(ew);
            let ls: Vec<f32> = ch.edges[li + 1].iter().map(|&(i, _)| l_of_call[node_call[ch.chunk[li + 1][i]] as usize]).collect();
            b.fibre(&format!("fibre.{}", li + 1), &ch.edges[li + 1], &ch.chunk[li + 1], pos[li + 1], q_b, fib[li + 1], &ls);
            for (i, r) in pm.down[li].iter().enumerate() {
                coarse_round(&mut b, &format!("down.{li}.{i}"), r, hl[li + 1], li + 1, fib[li + 1]);
            }
        }
        let mut cur = hl[levels];
        for (k, li) in (0..levels).rev().enumerate() {
            let gl = &ch.edges[li];
            let ids = &ch.ids[li + 1];
            let mut row_of = vec![usize::MAX; n_l[li]];
            for (r, &j) in ids.iter().enumerate() {
                row_of[j] = r;
            }
            let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_l[li]];
            for (e, &(i, j)) in gl.iter().enumerate() {
                if row_of[j] != usize::MAX {
                    rows[i].push((cur + row_of[j] * w, ews[li][e]));
                }
            }
            let up = b.alloc(n_l[li], w);
            b.csr(&format!("unpool.{li}"), &rows, w, up, false);
            if li > 0 {
                for (i, r) in pm.up[k].iter().enumerate() {
                    coarse_round(&mut b, &format!("up.{k}.{i}"), r, up, li, fib[li]);
                }
                let skip: Vec<Vec<(usize, f32)>> = (0..n_l[li]).map(|r| vec![(hl[li] + r * w, 1.0)]).collect();
                b.csr(&format!("skip.{li}"), &skip, w, up, true);
                cur = up;
            } else {
                let add: Vec<Vec<(usize, f32)>> = (0..nc).map(|r| vec![(up + r * w, 1.0)]).collect();
                b.csr("unpool.add", &add, w, h, true);
            }
        }
    }

    for (i, r) in pm.fine_post.iter().enumerate() {
        fine_round(&mut b, &format!("fine_post.{i}"), r);
    }

    // ---- heads
    let nt = cfg.n_time_bins;
    let bk = cfg.edge_types.iter().position(|t| t == "bond").ok_or("no bond edges")?;
    let (bsrc, bdst, _, _, be) = &edges[bk];
    let nb = bsrc.len() / 2;
    let bw = pm.bond_head.out;
    assert_eq!(bw, 4 + nt);
    let bond = b.alloc(nb, bw);
    let chunk = b.alloc(nc, NODE_OUT);
    let iw = pm.imp_head.out;
    let imp = b.alloc(ni, iw);
    let energy = b.alloc(ng, pm.energy_head.out);
    if nb > 0 {
        let fwd = b.table(&(0..nb).map(|k| (2 * k) as u32).collect::<Vec<_>>());
        let rev = b.table(&(0..nb).map(|k| (2 * k + 1) as u32).collect::<Vec<_>>());
        let hs = b.table(&(0..nb).map(|k| bsrc[2 * k] as u32).collect::<Vec<_>>());
        let hd = b.table(&(0..nb).map(|k| bdst[2 * k] as u32).collect::<Vec<_>>());
        let bc = b.table(&bond_call);
        let res = if vb {
            let qs: Vec<Mat> = (0..ng).map(|k| get(k, "bond_qs_feat")).collect::<Result<_, _>>()?;
            let q = cat_rows(&qs);
            let mut r = Mat::zeros(nb, bw);
            for k in 0..nb {
                r.d[k * bw] = q.d[2 * k * q.cols];
            }
            Some((b.input(&r), bw))
        } else {
            None
        };
        let s1 = [Seg::gather(*be, w, fwd), Seg::gather(*be, w, rev), Seg::gather(h, w, hs), Seg::gather(h, w, hd), Seg::gather(g, w, bc)];
        b.mlp("bond_head.fwd", nb, &pm.bond_head, &s1, MlpOut { base: bond, stride: bw, res, accumulate: false, scale: 0.5 }, 0);
        let s2 = [Seg::gather(*be, w, rev), Seg::gather(*be, w, fwd), Seg::gather(h, w, hd), Seg::gather(h, w, hs), Seg::gather(g, w, bc)];
        b.mlp("bond_head.rev", nb, &pm.bond_head, &s2, MlpOut { base: bond, stride: bw, res: None, accumulate: true, scale: 0.5 }, 0);
    }
    let pad = |b: &mut Builder, q: &Mat, width: usize| -> Option<(usize, usize)> {
        if !vb {
            return None;
        }
        let mut r = Mat::zeros(q.rows, width);
        for i in 0..q.rows {
            for c in 0..q.cols.min(6) {
                r.d[i * width + c] = q.d[i * q.cols + c];
            }
        }
        Some((b.input(&r), width))
    };
    let cres = pad(&mut b, &cq, NODE_OUT);
    b.mlp("chunk_head", nc, &pm.chunk_head, &[Seg::row(h, w), Seg::gather(g, w, chunk_call_t)], MlpOut { base: chunk, stride: NODE_OUT, res: cres, accumulate: false, scale: 1.0 }, 0);
    let ires = pad(&mut b, &iq, iw);
    b.mlp("imp_head", ni, &pm.imp_head, &[Seg::row(h + nc * w, w), Seg::gather(g, w, imp_call_t)], MlpOut { base: imp, stride: iw, res: ires, accumulate: false, scale: 1.0 }, 0);
    b.mlp("energy_head", ng, &pm.energy_head, &[Seg::row(g, w)], MlpOut::to(energy, pm.energy_head.out), 0);

    if b.table.is_empty() {
        b.table.push(0);
    }
    let nb_of: Vec<usize> = (0..ng).map(|k| bond_call.iter().filter(|&&c| c as usize == k).count()).collect();
    let layout = OutputLayout { nb, nc, ni, bond, chunk, imp, energy, glob: g, width: w, n_time_bins: nt, calls: (0..ng).map(|k| (ncs[k], nis[k], nb_of[k])).collect() };
    Ok(Program { arena: b.arena, weights: b.weights, table: b.table, ops: b.ops, layout })
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

impl Program {
    /// Execute every op on the CPU (the kernels' arithmetic, row by row); the final arena.
    pub fn run_cpu(&self) -> Vec<f32> {
        let mut a = self.arena.clone();
        for op in &self.ops {
            match op.kernel() {
                KERNEL_MLP => self.mlp_cpu(op, &mut a),
                KERNEL_CSR => self.csr_cpu(op, &mut a),
                KERNEL_FIBRE => self.fibre_cpu(op, &mut a),
                k => panic!("kernel {k}"),
            }
        }
        a
    }

    fn mlp_cpu(&self, op: &Op, a: &mut [f32]) {
        use mlp_p::*;
        let p = &op.params;
        let (wt, t) = (&self.weights, &self.table);
        let (hid, out, k_in) = (p[HIDDEN] as usize, p[OUT] as usize, p[K_IN] as usize);
        let mut x = vec![0.0f32; k_in];
        let mut pre = vec![0.0f32; hid];
        let mut hv = vec![0.0f32; hid];
        let mut y = vec![0.0f32; out];
        for r in 0..op.rows() {
            let mut k = 0;
            for v in pre.iter_mut() {
                *v = 0.0;
            }
            for s in 0..p[N_SEG] as usize {
                let o = SEG0 + 5 * s;
                let (base, idx, width, mode, norm) = (p[o] as usize, p[o + 1], p[o + 2] as usize, p[o + 3], p[o + 4]);
                let row = match mode {
                    SEG_ROW | SEG_PROJ_ROW => r,
                    SEG_GATHER | SEG_PROJ_GATHER => t[idx as usize + r] as usize,
                    _ => idx as usize,
                };
                let src = &a[base + row * width..base + (row + 1) * width];
                if mode >= SEG_PROJ_GATHER {
                    for c in 0..hid {
                        pre[c] += src[c];
                    }
                    continue;
                }
                for c in 0..width {
                    x[k + c] = if norm == NONE { src[c] } else { (src[c] - wt[norm as usize + c]) / wt[norm as usize + width + c] };
                }
                k += width;
            }
            let w1 = p[W1] as usize;
            for c in 0..hid {
                let mut s = if p[B1] == NONE { 0.0 } else { wt[p[B1] as usize + c] };
                for (kk, xv) in x.iter().enumerate() {
                    s += wt[w1 + kk * hid + c] * xv;
                }
                pre[c] += s;
            }
            if p[W2] == NONE {
                y.copy_from_slice(&pre[..out]);
            } else {
                for c in 0..hid {
                    hv[c] = silu(pre[c]);
                }
                let w2 = p[W2] as usize;
                for c in 0..out {
                    let mut s = wt[p[B2] as usize + c];
                    for (kk, hk) in hv.iter().enumerate() {
                        s += wt[w2 + kk * out + c] * hk;
                    }
                    y[c] = s;
                }
                if p[LN] != NONE {
                    let mean = y.iter().sum::<f32>() / out as f32;
                    let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / out as f32;
                    let inv = 1.0 / (var + 1e-5).sqrt();
                    let ln = p[LN] as usize;
                    for c in 0..out {
                        y[c] = (y[c] - mean) * inv * wt[ln + c] + wt[ln + out + c];
                    }
                }
            }
            let scale = f32::from_bits(p[SCALE]);
            let ob = p[OUT_BASE] as usize + r * p[OUT_STRIDE] as usize;
            for c in 0..out {
                let mut v = scale * y[c];
                if p[RES_BASE] != NONE {
                    v += a[p[RES_BASE] as usize + r * p[RES_STRIDE] as usize + c];
                }
                if p[FLAGS] & FLAG_ACCUMULATE != 0 {
                    a[ob + c] += v;
                } else {
                    a[ob + c] = v;
                }
            }
        }
    }

    fn csr_cpu(&self, op: &Op, a: &mut [f32]) {
        use csr_p::*;
        let p = &op.params;
        let t = &self.table;
        let width = p[WIDTH] as usize;
        for r in 0..op.rows() {
            let (e0, e1) = (t[p[OFFSETS] as usize + r] as usize, t[p[OFFSETS] as usize + r + 1] as usize);
            let ob = p[OUT_BASE] as usize + r * p[OUT_STRIDE] as usize;
            for c in 0..width {
                let mut s = 0.0f32;
                for e in e0..e1 {
                    let o = t[p[ENTRIES] as usize + 2 * e] as usize;
                    s += a[o + c] * f32::from_bits(t[p[ENTRIES] as usize + 2 * e + 1]);
                }
                if p[FLAGS] & FLAG_ACCUMULATE != 0 {
                    a[ob + c] += s;
                } else {
                    a[ob + c] = s;
                }
            }
        }
    }

    fn fibre_cpu(&self, op: &Op, a: &mut [f32]) {
        use fibre_p::*;
        let p = &op.params;
        let t = &self.table;
        for r in 0..op.rows() {
            let o = p[TRIPLES] as usize + 4 * r;
            let (i, j, c) = (t[o] as usize, t[o + 1] as usize, t[o + 2] as usize);
            let l_scale = f32::from_bits(t[o + 3]);
            let pb = p[POS] as usize;
            let d = [0, 1, 2].map(|k| a[pb + 3 * i + k] - a[pb + 3 * j + k]);
            let q = &a[p[Q] as usize + 4 * c..p[Q] as usize + 4 * c + 4];
            let (qw, u) = (q[0], [-q[1], -q[2], -q[3]]);
            let cross = |x: [f32; 3], y: [f32; 3]| [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
            let tt = cross(u, d).map(|x| 2.0 * x);
            let cc = cross(u, tt);
            let rr = [0, 1, 2].map(|k| (d[k] + qw * tt[k] + cc[k]) / l_scale);
            let ob = p[OUT_BASE] as usize + 4 * r;
            a[ob..ob + 3].copy_from_slice(&rr);
            a[ob + 3] = (rr[0] * rr[0] + rr[1] * rr[1] + rr[2] * rr[2]).sqrt();
        }
    }

    /// The heads' outputs of a single-call program from its final arena.
    pub fn outputs(&self, a: &[f32]) -> Outputs {
        self.outputs_of(a, 0)
    }

    /// Call `k`'s heads' outputs from a final arena (the elementwise maps applied here).
    pub fn outputs_of(&self, a: &[f32], k: usize) -> Outputs {
        let l = &self.layout;
        let (nc, ni, nb) = l.calls[k];
        let c0: usize = l.calls[..k].iter().map(|c| c.0).sum();
        let i0: usize = l.calls[..k].iter().map(|c| c.1).sum();
        let b0: usize = l.calls[..k].iter().map(|c| c.2).sum();
        let bw = 4 + l.n_time_bins;
        let bond = l.bond + b0 * bw;
        let col = |c: usize| -> Vec<f32> { (0..nb).map(|r| a[bond + r * bw + c]).collect() };
        let mut times = Mat::zeros(nb, l.n_time_bins);
        for r in 0..nb {
            times.row_mut(r).copy_from_slice(&a[bond + r * bw + 4..bond + r * bw + bw]);
        }
        let iw = NODE_OUT + 1;
        let imp = l.imp + i0 * iw;
        let mut im = Mat::zeros(ni, NODE_OUT);
        for r in 0..ni {
            im.row_mut(r).copy_from_slice(&a[imp + r * iw..imp + r * iw + NODE_OUT]);
        }
        let en = &a[l.energy + 4 * k..l.energy + 4 * k + 4];
        let chunk = l.chunk + c0 * NODE_OUT;
        Outputs {
            bond_log_r: col(0),
            bond_break_logit: col(1),
            bond_log_q: col(3),
            bond_time_logits: times,
            bond_ddamage: col(2).into_iter().map(softplus).collect(),
            chunk_out: Mat { rows: nc, cols: NODE_OUT, d: a[chunk..chunk + nc * NODE_OUT].to_vec() },
            imp_out: im,
            imp_perf_logit: (0..ni).map(|r| a[imp + r * iw + NODE_OUT]).collect(),
            energy_d_bond: softplus(en[0]),
            energy_frac_joint: sigmoid(en[1]),
            energy_e_elastic: softplus(en[2]),
            energy_ke_loss: en[3],
            latent_global: a[l.glob + k * l.width..l.glob + (k + 1) * l.width].to_vec(),
        }
    }
}
