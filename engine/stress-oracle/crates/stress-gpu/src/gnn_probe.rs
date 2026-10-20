//! P0 cost probe of the learned impact-outcome plan (`docs/learned-stress-solver/
//! learned-stress-solver-plan.md`, P0 step 4): a MeshGraphNet of the target shape with
//! random weights on wall-shaped bond graphs, timed on this device. Nothing is trained.
//!
//! The network: an encoder, a U-shaped processor of message-passing rounds over a
//! bi-stride hierarchy (BSMS-GNN's pooling: every other BFS level of the bond graph is
//! kept, coarse edges join kept nodes within two hops), pool and unpool between levels,
//! and a decoder. One fused dispatch per round (`shaders/gnn_probe.slang`); all weights in
//! one buffer. A batch of islands is their disjoint union (CSR concatenation), so it costs
//! the same dispatches as one island.

use crate::gpu::{Binding, Gpu};
use crate::ShaderSet;

/// A directed graph in CSR by destination: node `i`'s incoming edges are
/// `offsets[i]..offsets[i + 1]`, edge `e` comes from `src[e]`.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: usize,
    pub offsets: Vec<u32>,
    pub src: Vec<u32>,
}

impl Graph {
    pub fn edges(&self) -> usize {
        self.src.len()
    }

    fn from_adjacency(adj: &[Vec<u32>]) -> Graph {
        let mut offsets = vec![0u32];
        let mut src = Vec::new();
        for a in adj {
            let mut a = a.clone();
            a.sort_unstable();
            a.dedup();
            src.extend(a);
            offsets.push(src.len() as u32);
        }
        Graph { nodes: adj.len(), offsets, src }
    }

    fn adjacency(&self) -> Vec<Vec<u32>> {
        (0..self.nodes).map(|i| self.src[self.offsets[i] as usize..self.offsets[i + 1] as usize].to_vec()).collect()
    }

    /// The bond graph of an `nx x ny x nz` grid of chunks (face neighbours, both ways).
    pub fn grid(nx: usize, ny: usize, nz: usize) -> Graph {
        let id = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
        let mut adj = vec![Vec::new(); nx * ny * nz];
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let n = id(i, j, k);
                    let mut link = |m: usize| {
                        adj[n].push(m as u32);
                        adj[m].push(n as u32);
                    };
                    if i + 1 < nx {
                        link(id(i + 1, j, k));
                    }
                    if j + 1 < ny {
                        link(id(i, j + 1, k));
                    }
                    if k + 1 < nz {
                        link(id(i, j, k + 1));
                    }
                }
            }
        }
        Graph::from_adjacency(&adj)
    }

    /// The disjoint union of `parts` (a batch of islands).
    pub fn union(parts: &[Graph]) -> Graph {
        let mut g = Graph { nodes: 0, offsets: vec![0], src: Vec::new() };
        for p in parts {
            let base = g.nodes as u32;
            for i in 0..p.nodes {
                g.src.extend(p.src[p.offsets[i] as usize..p.offsets[i + 1] as usize].iter().map(|s| s + base));
                g.offsets.push(g.src.len() as u32);
            }
            g.nodes += p.nodes;
        }
        g
    }
}

/// One coarsening (BSMS bi-stride): the coarse graph, every fine node's coarse parent, and
/// every coarse node's members (CSR).
#[derive(Clone, Debug)]
pub struct Coarsening {
    pub coarse: Graph,
    pub parent: Vec<u32>,
    pub member_offsets: Vec<u32>,
    pub members: Vec<u32>,
}

/// Bi-stride pooling: BFS from the lowest node of each component, keep the even levels;
/// an odd-level node's parent is its BFS parent (kept, one level up). Coarse edges join
/// kept nodes within two fine hops.
pub fn bistride(g: &Graph) -> Coarsening {
    let adj = g.adjacency();
    let mut depth = vec![u32::MAX; g.nodes];
    let mut bfs_parent = vec![u32::MAX; g.nodes];
    for root in 0..g.nodes {
        if depth[root] != u32::MAX {
            continue;
        }
        depth[root] = 0;
        let mut queue = std::collections::VecDeque::from([root as u32]);
        while let Some(n) = queue.pop_front() {
            for &m in &adj[n as usize] {
                if depth[m as usize] == u32::MAX {
                    depth[m as usize] = depth[n as usize] + 1;
                    bfs_parent[m as usize] = n;
                    queue.push_back(m);
                }
            }
        }
    }
    let mut index = vec![u32::MAX; g.nodes];
    let mut kept = Vec::new();
    for n in 0..g.nodes {
        if depth[n] % 2 == 0 {
            index[n] = kept.len() as u32;
            kept.push(n);
        }
    }
    let parent: Vec<u32> = (0..g.nodes).map(|n| if index[n] != u32::MAX { index[n] } else { index[bfs_parent[n] as usize] }).collect();
    let mut cadj = vec![Vec::new(); kept.len()];
    for (ci, &n) in kept.iter().enumerate() {
        for &m in &adj[n] {
            if index[m as usize] != u32::MAX && m as usize != n {
                cadj[ci].push(index[m as usize]);
            }
            for &q in &adj[m as usize] {
                if index[q as usize] != u32::MAX && q as usize != n {
                    cadj[ci].push(index[q as usize]);
                }
            }
        }
    }
    let mut member_lists = vec![Vec::new(); kept.len()];
    for (n, &p) in parent.iter().enumerate() {
        member_lists[p as usize].push(n as u32);
    }
    let mut member_offsets = vec![0u32];
    let mut members = Vec::new();
    for l in member_lists {
        members.extend(l);
        member_offsets.push(members.len() as u32);
    }
    Coarsening { coarse: Graph::from_adjacency(&cadj), parent, member_offsets, members }
}

/// One probe configuration: `islands` copies of a wall-shaped island of `chunks` chunks.
#[derive(Clone, Debug)]
pub struct Config {
    pub islands: usize,
    pub chunks: usize,
    pub width: usize,
    pub precision: Precision,
    pub schedule_name: &'static str,
    pub schedule: &'static [usize],
    pub kernel: RoundKernel,
}

/// The 12-round U (3 fine, 2 on each coarse level down, 2 up, 3 fine) and an 8-round one.
pub const SCHEDULES: [(&str, &[usize]); 2] = [("u12", &[3, 2, 2, 2, 3]), ("u8", &[2, 1, 2, 1, 2])];

/// A wall-shaped island of about `chunks` chunks (the M5 walls' 0.1 m grid).
pub fn island(chunks: usize) -> Graph {
    match chunks {
        220 => Graph::grid(10, 2, 11),
        1000 => Graph::grid(20, 2, 25),
        2000 => Graph::grid(20, 4, 25),
        10000 => Graph::grid(50, 4, 50),
        n => panic!("no island shape for {n} chunks"),
    }
}

/// The plan's sweeps. `size`: islands of 220, 1,000 and 10,000 chunks, widths 32/64/128,
/// each precision, both schedules. `batch`: 1, 5, 20 and 50 islands of 220, 1,000 and 2,000
/// chunks, widths 32/64/128, f32, the 12-round U. Each for every kernel in `kernels`.
pub fn sweep(name: &str, precisions: &[Precision], kernels: &[RoundKernel]) -> Vec<Config> {
    let mut out = Vec::new();
    for &kernel in kernels {
        if name == "size" || name == "all" {
            for &chunks in &[220usize, 1000, 10000] {
                for &width in &[32usize, 64, 128] {
                    for &precision in precisions {
                        for &(schedule_name, schedule) in &SCHEDULES {
                            out.push(Config { islands: 1, chunks, width, precision, schedule_name, schedule, kernel });
                        }
                    }
                }
            }
        }
        if name == "batch" || name == "all" {
            for &chunks in &[220usize, 1000, 2000] {
                for &islands in &[1usize, 5, 20, 50] {
                    for &width in &[32usize, 64, 128] {
                        out.push(Config { islands, chunks, width, precision: Precision::F32, schedule_name: "u12", schedule: SCHEDULES[0].1, kernel });
                    }
                }
            }
        }
    }
    out
}

/// Build and time one configuration: its JSON result line.
pub fn run(gpu: &Gpu, c: &Config, warmup: usize, repeats: usize, shaders: &str) -> serde_json::Value {
    let parts: Vec<Graph> = (0..c.islands).map(|_| island(c.chunks)).collect();
    let g = Graph::union(&parts);
    let probe = Probe::with_kernel(gpu, &g, c.width, c.precision, c.schedule, c.kernel);
    let (wall, gt) = probe.time(gpu, warmup, repeats);
    serde_json::json!({
        "adapter": gpu.adapter_info.name, "backend": format!("{:?}", gpu.adapter_info.backend), "shaders": shaders,
        "islands": c.islands, "chunks_per_island": c.chunks, "nodes": g.nodes, "edges": g.edges(), "levels": probe.levels,
        "width": c.width, "precision": c.precision.name(), "kernel": c.kernel.name(), "schedule": c.schedule_name,
        "rounds": c.schedule.iter().sum::<usize>(), "dispatches": probe.dispatches,
        "gmacs": probe.macs * 1e-9, "wall_ms": stats(&wall), "gpu_ms": stats(&gt),
    })
}

/// Median, p10, p90, min and max of `v` (null when empty).
pub fn stats(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    serde_json::json!({"median": q(0.5), "p10": q(0.1), "p90": q(0.9), "min": s[0], "max": s[s.len() - 1], "n": s.len()})
}

/// How a message-passing round is dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundKernel {
    /// One dispatch: each node runs the edge MLP on its incoming edges in turn, sums them and
    /// runs the node MLP (`gnn_round` mode 0).
    Fused,
    /// Two: the edge MLP edge-parallel, then the sum and node MLP (modes 5, 6).
    Split,
    /// Three: the edge MLP as GEMM tiles, the CSR sum, the node MLP as GEMM tiles
    /// (`gnn_mlp_tiled`, `gnn_csr_sum`).
    Tiled,
}

impl RoundKernel {
    pub fn name(self) -> &'static str {
        match self {
            RoundKernel::Fused => "fused",
            RoundKernel::Split => "split",
            RoundKernel::Tiled => "tiled",
        }
    }
}

/// Element type of the network's buffers and multiply-adds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F32,
    F16,
}

impl Precision {
    fn set(self) -> &'static ShaderSet {
        match self {
            Precision::F32 => &crate::shaders::GNN_PROBE,
            Precision::F16 => &crate::shaders::GNN_PROBE_F16,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    nodes: u32,
    grid_x: u32,
    weights: u32,
    edges: u32,
    mode: u32,
    in_width: u32,
    max_degree: u32,
}

const THREADS: usize = 128;
const MAX_GROUPS_X: usize = 65_535;
/// Input features per node and outputs per node (the plan's heads are a few per node).
pub const IN_FEATURES: usize = 16;
pub const OUT_FEATURES: usize = 8;

fn mlp_size(k_in: usize, w: usize) -> usize {
    k_in * w + w + w * w + w + 2 * w
}

/// The probe's network on one (batched) graph, built on the device.
pub struct Probe {
    pub width: usize,
    pub precision: Precision,
    /// Nodes and directed edges of each level, fine first.
    pub levels: Vec<(usize, usize)>,
    /// Rounds per stage of the U (down then up: `[3, 2, 2, 2, 3]` is 3 fine, 2 on level 1,
    /// 2 on level 2, 2 on level 1, 3 fine).
    pub schedule: Vec<usize>,
    pub dispatches: usize,
    /// Multiply-adds of one call (both MLPs of every round, encoder and decoder).
    pub macs: f64,
    kernels: [crate::gpu::Kernel; 3],
    steps: Vec<(usize, wgpu::BindGroup, u32, u32)>,
    timestamps: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer)>,
    _buffers: Vec<wgpu::Buffer>,
    outputs_index: usize,
    host: Host,
}

/// The inputs as uploaded, for the CPU reference.
struct Host {
    wts: Vec<f32>,
    features: Vec<f32>,
    e0: Vec<Vec<f32>>,
    graphs: Vec<Graph>,
    coarse: Vec<Coarsening>,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The shader's MLP (`mlp` in gnn_probe.slang) for one input vector: every output channel.
fn mlp_ref(wts: &[f32], base: usize, x: &[f32], w: usize) -> Vec<f32> {
    let k_in = x.len();
    let hid: Vec<f32> = (0..w)
        .map(|c| {
            let mut a = wts[base + k_in * w + c];
            for (k, xk) in x.iter().enumerate() {
                a += wts[base + k * w + c] * xk;
            }
            silu(a)
        })
        .collect();
    let b2 = base + k_in * w + w;
    let o: Vec<f32> = (0..w)
        .map(|c| {
            let mut a = wts[b2 + w * w + c];
            for (k, hk) in hid.iter().enumerate() {
                a += wts[b2 + k * w + c] * hk;
            }
            a
        })
        .collect();
    let mean = o.iter().sum::<f32>() / w as f32;
    let var = o.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
    let g = b2 + w * w + w;
    (0..w).map(|c| wts[g + c] * (o[c] - mean) / (var + 1e-5).sqrt() + wts[g + w + c]).collect()
}

/// Deterministic pseudo-random values in [-1, 1).
fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut z = seed;
    (0..n)
        .map(|_| {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// f32 to IEEE half bits (round to nearest even; the probe's values are small and finite).
fn half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let r = ((exp as u32) << 10) | (man >> 13);
    let round = (man >> 12) & 1 != 0 && ((man & 0xfff) != 0 || r & 1 != 0);
    sign | (r + round as u32) as u16
}

fn upload(gpu: &Gpu, label: &str, data: &[f32], p: Precision) -> wgpu::Buffer {
    match p {
        Precision::F32 => gpu.storage(label, data),
        Precision::F16 => {
            let mut h: Vec<u16> = data.iter().map(|&x| half_bits(x)).collect();
            if h.len() % 2 == 1 {
                h.push(0);
            }
            gpu.storage(label, &h)
        }
    }
}

impl Probe {
    pub fn new(gpu: &Gpu, graph: &Graph, width: usize, precision: Precision, schedule: &[usize]) -> Probe {
        Probe::with_kernel(gpu, graph, width, precision, schedule, RoundKernel::Fused)
    }

    /// `split`: each round as two dispatches, the edge MLP edge-parallel then the node update
    /// (no per-node loop over its edges); else one fused dispatch per round.
    pub fn with_kernel(gpu: &Gpu, graph: &Graph, width: usize, precision: Precision, schedule: &[usize], round_kernel: RoundKernel) -> Probe {
        let split = round_kernel == RoundKernel::Split;
        assert!(matches!(width, 32 | 64 | 128), "width {width}: 32, 64 or 128");
        assert!(schedule.len() % 2 == 1, "a U schedule has an odd number of stages");
        let depth = schedule.len() / 2;
        let mut graphs = vec![graph.clone()];
        let mut coarse = Vec::new();
        for _ in 0..depth {
            let c = bistride(graphs.last().unwrap());
            graphs.push(c.coarse.clone());
            coarse.push(c);
        }
        let levels: Vec<(usize, usize)> = graphs.iter().map(|g| (g.nodes, g.edges())).collect();
        let w = width;
        // Weights: encoder, each round's edge and node MLPs, decoder.
        let rounds: usize = schedule.iter().sum();
        let enc = mlp_size(IN_FEATURES, w);
        let round = mlp_size(3 * w, w) + mlp_size(2 * w, w);
        let dec = mlp_size(w, w);
        let total = enc + rounds * round + dec;
        let mut wts = noise(1, total);
        for x in &mut wts {
            *x *= (1.0 / (2.0 * w as f32)).sqrt();
        }
        let set = precision.set();
        let layout = [Binding::Uniform, Binding::Storage, Binding::Storage, Binding::StorageRw, Binding::Storage, Binding::StorageRw, Binding::Storage, Binding::Storage];
        let kernels = [gpu.compute(set, "gnn_round", &layout), gpu.compute(set, "gnn_mlp_tiled", &layout), gpu.compute(set, "gnn_csr_sum", &layout)];
        let mut buffers = Vec::new();
        let wbuf = upload(gpu, "gnn weights", &wts, precision);
        // Per level: two node-feature buffers and two edge-feature buffers (ping-pong),
        // and the CSR. Edge buffers hold at least one element (bindings must not be empty).
        let mut h = Vec::new();
        let mut e = Vec::new();
        let (mut host_h, mut host_e) = (Vec::new(), Vec::new());
        let mut dsts = Vec::new();
        let mut aggs = Vec::new();
        let mut csr = Vec::new();
        for (l, g) in graphs.iter().enumerate() {
            let hn = (g.nodes * w).max(2);
            let en = (g.edges() * w).max(2);
            h.push([upload(gpu, "h0", &noise(10 + l as u64, hn), precision), upload(gpu, "h1", &vec![0.0; hn], precision)]);
            // The level's initial edge features (never written, so every call starts from
            // them), and two ping-pong buffers.
            let e0 = noise(20 + l as u64, en);
            e.push([upload(gpu, "e init", &e0, precision), upload(gpu, "e1", &vec![0.0; en], precision), upload(gpu, "e2", &vec![0.0; en], precision)]);
            host_e.push(e0);
            host_h.push(noise(10 + l as u64, hn));
            let src = if g.src.is_empty() { vec![0u32] } else { g.src.clone() };
            let mut dst = Vec::with_capacity(g.edges().max(1));
            for n in 0..g.nodes {
                dst.extend(std::iter::repeat_n(n as u32, (g.offsets[n + 1] - g.offsets[n]) as usize));
            }
            if dst.is_empty() {
                dst.push(0);
            }
            dsts.push(gpu.storage("dst", &dst));
            aggs.push(upload(gpu, "agg", &vec![0.0; hn], precision));
            csr.push((gpu.storage("offsets", &g.offsets), gpu.storage("src", &src)));
        }
        let host_features = noise(5, (graph.nodes * IN_FEATURES).max(2));
        let features = upload(gpu, "features", &host_features, precision);
        let outputs = upload(gpu, "outputs", &vec![0.0; (graph.nodes * OUT_FEATURES).max(2)], precision);
        let pools: Vec<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)> =
            coarse.iter().map(|c| (gpu.storage("member offsets", &c.member_offsets), gpu.storage("members", &c.members), gpu.storage("parent", &c.parent))).collect();
        let dummy = gpu.storage("dummy", &[0u32; 4]);
        let dummy_w = gpu.storage("dummy rw", &[0u32; 4]);

        let mut steps = Vec::new();
        let mut macs = 0.0;
        let groups = |nodes: usize, per: usize| -> (u32, u32) {
            let n = nodes.div_ceil(per).max(1);
            if n <= MAX_GROUPS_X {
                (n as u32, 1)
            } else {
                (MAX_GROUPS_X as u32, n.div_ceil(MAX_GROUPS_X) as u32)
            }
        };
        let per = THREADS / w;
        let add_k = |gpu: &Gpu, k: usize, p: Params, bufs: [&wgpu::Buffer; 7], gx: u32, gy: u32, steps: &mut Vec<(usize, wgpu::BindGroup, u32, u32)>, buffers: &mut Vec<wgpu::Buffer>| {
            let u = gpu.uniform("gnn params", &p);
            let bind = gpu.bind(&kernels[k].group, &[&u, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6]]);
            buffers.push(u);
            steps.push((k, bind, gx, gy));
        };
        let add = |gpu: &Gpu, p: Params, bufs: [&wgpu::Buffer; 7], gx: u32, gy: u32, steps: &mut Vec<(usize, wgpu::BindGroup, u32, u32)>, buffers: &mut Vec<wgpu::Buffer>| add_k(gpu, 0, p, bufs, gx, gy, steps, buffers);
        // Which of each level's ping-pong buffers holds its current features.
        let mut cur = vec![0usize; graphs.len()];
        let mut ecur = vec![0usize; graphs.len()];
        // Encoder into the fine level's buffer 0.
        let (gx, gy) = groups(graph.nodes, per);
        add(gpu, Params { width: w as u32, nodes: graph.nodes as u32, grid_x: gx, weights: 0, edges: 0, mode: 1, in_width: IN_FEATURES as u32, max_degree: 0 }, [&wbuf, &features, &h[0][0], &dummy, &dummy_w, &csr[0].0, &csr[0].1], gx, gy, &mut steps, &mut buffers);
        macs += graph.nodes as f64 * (IN_FEATURES * w + w * w) as f64;
        let mut woff = enc;
        let mut level = 0usize;
        for (stage, &n) in schedule.iter().enumerate() {
            if stage > 0 {
                let down = stage <= depth;
                if down {
                    // Pool fine `level` into `level + 1`.
                    let c = &pools[level];
                    let nodes = graphs[level + 1].nodes;
                    let (gx, gy) = groups(nodes, per);
                    let (fine, coarse_buf) = (&h[level][cur[level]], &h[level + 1][cur[level + 1]]);
                    add(gpu, Params { width: w as u32, nodes: nodes as u32, grid_x: gx, weights: 0, edges: 0, mode: 3, in_width: 0, max_degree: 0 }, [&wbuf, fine, coarse_buf, &dummy, &dummy_w, &c.0, &c.1], gx, gy, &mut steps, &mut buffers);
                    level += 1;
                } else {
                    // Unpool `level` into `level - 1` (adds the coarse parent's features).
                    let c = &pools[level - 1];
                    let nodes = graphs[level - 1].nodes;
                    let (gx, gy) = groups(nodes, per);
                    let (coarse_buf, fine) = (&h[level][cur[level]], &h[level - 1][cur[level - 1]]);
                    add(gpu, Params { width: w as u32, nodes: nodes as u32, grid_x: gx, weights: 0, edges: 0, mode: 4, in_width: 0, max_degree: 0 }, [&wbuf, coarse_buf, fine, &dummy, &dummy_w, &dummy, &c.2], gx, gy, &mut steps, &mut buffers);
                    level -= 1;
                }
            }
            let g = &graphs[level];
            for _ in 0..n {
                let (gx, gy) = groups(g.nodes, per);
                let (a, b) = (cur[level], 1 - cur[level]);
                let (ea, eb) = (ecur[level], if ecur[level] == 1 { 2 } else { 1 });
                if round_kernel == RoundKernel::Tiled {
                    let tr = 2048 / w;
                    let (ex, ey) = groups(g.edges(), tr);
                    add_k(
                        gpu,
                        1,
                        Params { width: w as u32, nodes: g.edges() as u32, grid_x: ex, weights: woff as u32, edges: g.edges() as u32, mode: 7, in_width: 0, max_degree: 0 },
                        [&wbuf, &e[level][ea], &e[level][eb], &h[level][a], &dummy_w, &dsts[level], &csr[level].1],
                        ex,
                        ey,
                        &mut steps,
                        &mut buffers,
                    );
                    add_k(
                        gpu,
                        2,
                        Params { width: w as u32, nodes: g.nodes as u32, grid_x: gx, weights: 0, edges: g.edges() as u32, mode: 0, in_width: 0, max_degree: 0 },
                        [&wbuf, &dummy, &aggs[level], &e[level][eb], &dummy_w, &csr[level].0, &csr[level].1],
                        gx,
                        gy,
                        &mut steps,
                        &mut buffers,
                    );
                    let (nx, ny) = groups(g.nodes, tr);
                    add_k(
                        gpu,
                        1,
                        Params { width: w as u32, nodes: g.nodes as u32, grid_x: nx, weights: woff as u32, edges: g.edges() as u32, mode: 8, in_width: 0, max_degree: 0 },
                        [&wbuf, &h[level][a], &h[level][b], &aggs[level], &dummy_w, &csr[level].0, &csr[level].1],
                        nx,
                        ny,
                        &mut steps,
                        &mut buffers,
                    );
                    cur[level] = b;
                    ecur[level] = eb;
                    woff += round;
                    macs += g.edges() as f64 * (3 * w * w + w * w) as f64 + g.nodes as f64 * (2 * w * w + w * w) as f64;
                    continue;
                }
                if split {
                    let (ex, ey) = groups(g.edges(), per);
                    add(
                        gpu,
                        Params { width: w as u32, nodes: g.nodes as u32, grid_x: ex, weights: woff as u32, edges: g.edges() as u32, mode: 5, in_width: 0, max_degree: 0 },
                        [&wbuf, &h[level][a], &dummy_w, &e[level][ea], &e[level][eb], &dsts[level], &csr[level].1],
                        ex,
                        ey,
                        &mut steps,
                        &mut buffers,
                    );
                    add(
                        gpu,
                        Params { width: w as u32, nodes: g.nodes as u32, grid_x: gx, weights: woff as u32, edges: g.edges() as u32, mode: 6, in_width: 0, max_degree: 0 },
                        [&wbuf, &h[level][a], &h[level][b], &e[level][eb], &dummy_w, &csr[level].0, &csr[level].1],
                        gx,
                        gy,
                        &mut steps,
                        &mut buffers,
                    );
                    cur[level] = b;
                    ecur[level] = eb;
                    woff += round;
                    macs += g.edges() as f64 * (3 * w * w + w * w) as f64 + g.nodes as f64 * (2 * w * w + w * w) as f64;
                    continue;
                }
                add(
                    gpu,
                    Params { width: w as u32, nodes: g.nodes as u32, grid_x: gx, weights: woff as u32, edges: g.edges() as u32, mode: 0, in_width: 0, max_degree: (0..g.nodes).map(|n| g.offsets[n + 1] - g.offsets[n]).max().unwrap_or(0) },
                    [&wbuf, &h[level][a], &h[level][b], &e[level][ea], &e[level][eb], &csr[level].0, &csr[level].1],
                    gx,
                    gy,
                    &mut steps,
                    &mut buffers,
                );
                cur[level] = b;
                ecur[level] = eb;
                woff += round;
                macs += g.edges() as f64 * (3 * w * w + w * w) as f64 + g.nodes as f64 * (2 * w * w + w * w) as f64;
            }
        }
        assert_eq!(level, 0);
        let (gx, gy) = groups(graph.nodes, per);
        add(gpu, Params { width: w as u32, nodes: graph.nodes as u32, grid_x: gx, weights: woff as u32, edges: 0, mode: 2, in_width: OUT_FEATURES as u32, max_degree: 0 }, [&wbuf, &h[0][cur[0]], &outputs, &dummy, &dummy_w, &csr[0].0, &csr[0].1], gx, gy, &mut steps, &mut buffers);
        macs += graph.nodes as f64 * (2 * w * w) as f64;
        let timestamps = gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            let q = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("gnn time"), ty: wgpu::QueryType::Timestamp, count: 2 });
            let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("gnn resolve"), size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("gnn read"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            (q, resolve, read)
        });
        let outputs_index = buffers.len() + 2;
        buffers.push(wbuf);
        buffers.push(features);
        buffers.push(outputs);
        buffers.push(dummy);
        buffers.push(dummy_w);
        for x in h {
            buffers.extend(x);
        }
        for x in e {
            buffers.extend(x);
        }
        buffers.extend(dsts);
        buffers.extend(aggs);
        for (a, b) in csr {
            buffers.push(a);
            buffers.push(b);
        }
        for (a, b, c) in pools {
            buffers.extend([a, b, c]);
        }
        let dispatches = steps.len();
        let host = Host { wts, features: host_features, e0: host_e, graphs, coarse };
        Probe { width, precision, levels, schedule: schedule.to_vec(), dispatches, macs, kernels, steps, timestamps, _buffers: buffers, outputs_index, host }
    }

    /// Record one call: every dispatch in one compute pass (WebGPU orders them, each seeing
    /// the last one's writes), with timestamps around the pass when the device has them.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let tw = self.timestamps.as_ref().map(|(q, _, _)| wgpu::ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gnn call"), timestamp_writes: tw });
            for (k, bind, gx, gy) in &self.steps {
                pass.set_pipeline(&self.kernels[*k].pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
        }
        if let Some((q, resolve, read)) = &self.timestamps {
            encoder.resolve_query_set(q, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, read, 0, 16);
        }
    }

    /// The last call's GPU time (ms) from its timestamps, after the device is idle.
    pub fn gpu_ms(&self, gpu: &Gpu) -> Option<f64> {
        let (_, _, read) = self.timestamps.as_ref()?;
        let slice = read.slice(..);
        let mapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = mapped.clone();
        slice.map_async(wgpu::MapMode::Read, move |_| flag.store(true, std::sync::atomic::Ordering::Release));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        engine_gpu_wait::until(|| mapped.load(std::sync::atomic::Ordering::Acquire));
        let t: [u64; 2] = {
            let data = slice.get_mapped_range().expect("mapped timestamps");
            let v: &[u64] = bytemuck::cast_slice(&data);
            [v[0], v[1]]
        };
        read.unmap();
        let period = gpu.queue.get_timestamp_period() as f64;
        Some((t[1].saturating_sub(t[0])) as f64 * period * 1e-6)
    }

    /// Run the call `repeats` times (after `warmup`), each submitted alone and waited for:
    /// wall times (ms, submit to the work-done callback) and GPU times (ms, timestamps, when
    /// the device has them). In the browser this must run under `WebAssembly.promising`.
    pub fn time(&self, gpu: &Gpu, warmup: usize, repeats: usize) -> (Vec<f64>, Vec<f64>) {
        let mut wall = Vec::new();
        let mut gpu_t = Vec::new();
        for r in 0..warmup + repeats {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            self.encode(&mut enc);
            let t = web_time::Instant::now();
            gpu.queue.submit([enc.finish()]);
            // Natively the poll blocks until the work is done; in the browser the callback
            // runs on the event loop, which `until` yields to (JSPI, `engine-gpu-wait`).
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = done.clone();
            gpu.queue.on_submitted_work_done(move || flag.store(true, std::sync::atomic::Ordering::Release));
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            engine_gpu_wait::until(|| done.load(std::sync::atomic::Ordering::Acquire));
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let g = self.gpu_ms(gpu);
            if r >= warmup {
                wall.push(ms);
                if let Some(g) = g {
                    gpu_t.push(g);
                }
            }
        }
        (wall, gpu_t)
    }

    /// The decoder's outputs of the last call (f32 only).
    pub fn outputs(&self, gpu: &Gpu) -> Vec<f32> {
        assert_eq!(self.precision, Precision::F32, "outputs are read in f32");
        let mut v: Vec<f32> = gpu.read(&self._buffers[self.outputs_index]);
        v.truncate(self.host.graphs[0].nodes * OUT_FEATURES);
        v
    }

    /// The same call on the CPU, from the same inputs: the decoder's outputs.
    pub fn reference(&self) -> Vec<f32> {
        let w = self.width;
        let hst = &self.host;
        let g0 = &hst.graphs[0];
        let depth = self.schedule.len() / 2;
        let mut h: Vec<Vec<f32>> = hst.graphs.iter().map(|g| vec![0.0; g.nodes * w]).collect();
        for n in 0..g0.nodes {
            let y = mlp_ref(&hst.wts, 0, &hst.features[n * IN_FEATURES..(n + 1) * IN_FEATURES], w);
            h[0][n * w..(n + 1) * w].copy_from_slice(&y);
        }
        let enc = mlp_size(IN_FEATURES, w);
        let round = mlp_size(3 * w, w) + mlp_size(2 * w, w);
        let mut woff = enc;
        let mut e: Vec<Vec<f32>> = hst.e0.to_vec();
        let mut level = 0;
        for (stage, &rounds) in self.schedule.iter().enumerate() {
            if stage > 0 {
                if stage <= depth {
                    let c = &hst.coarse[level];
                    let nodes = hst.graphs[level + 1].nodes;
                    for n in 0..nodes {
                        let (a, b) = (c.member_offsets[n] as usize, c.member_offsets[n + 1] as usize);
                        for ch in 0..w {
                            let s: f32 = (a..b).map(|i| h[level][c.members[i] as usize * w + ch]).sum();
                            h[level + 1][n * w + ch] = s / (b - a).max(1) as f32;
                        }
                    }
                    level += 1;
                } else {
                    let c = &hst.coarse[level - 1];
                    for n in 0..hst.graphs[level - 1].nodes {
                        let p = c.parent[n] as usize;
                        for ch in 0..w {
                            h[level - 1][n * w + ch] += h[level][p * w + ch];
                        }
                    }
                    level -= 1;
                }
            }
            let g = &hst.graphs[level];
            for _ in 0..rounds {
                let mut hn = h[level].clone();
                let mut en = e[level].clone();
                for n in 0..g.nodes {
                    let mut agg = vec![0.0f32; w];
                    let hd = &h[level][n * w..(n + 1) * w];
                    for ei in g.offsets[n] as usize..g.offsets[n + 1] as usize {
                        let s = g.src[ei] as usize;
                        let mut x = e[level][ei * w..(ei + 1) * w].to_vec();
                        x.extend_from_slice(&h[level][s * w..(s + 1) * w]);
                        x.extend_from_slice(hd);
                        let y = mlp_ref(&hst.wts, woff, &x, w);
                        for ch in 0..w {
                            let v = e[level][ei * w + ch] + y[ch];
                            en[ei * w + ch] = v;
                            agg[ch] += v;
                        }
                    }
                    let mut x = hd.to_vec();
                    x.extend_from_slice(&agg);
                    let y = mlp_ref(&hst.wts, woff + mlp_size(3 * w, w), &x, w);
                    for ch in 0..w {
                        hn[n * w + ch] = hd[ch] + y[ch];
                    }
                }
                h[level] = hn;
                e[level] = en;
                woff += round;
            }
        }
        let mut out = vec![0.0; g0.nodes * OUT_FEATURES];
        for n in 0..g0.nodes {
            let y = mlp_ref(&hst.wts, woff, &h[0][n * w..(n + 1) * w], w);
            out[n * OUT_FEATURES..(n + 1) * OUT_FEATURES].copy_from_slice(&y[..OUT_FEATURES]);
        }
        out
    }
}
