// P0 cost probe of the learned impact-outcome plan (docs/learned-stress-solver/
// learned-stress-solver-plan.md, P0 step 4): a MeshGraphNet of the target shape with
// random weights, timed, never trained. One dispatch per message-passing round, fused:
// each node gathers its incoming edges (CSR by destination), runs the edge MLP on each
// (e' = e + LN(MLP([e, h_src, h_dst]))), sums them, then runs the node MLP
// (h' = h + LN(MLP([h, sum e']))). MLPs are two layers with SiLU; every weight of the
// network is in one storage buffer (WebGPU's 10 storage buffers per stage).
//
// A workgroup of 128 threads holds 128 / W nodes, W threads (one channel each) per node.
// `REAL` is the storage and arithmetic type (float, or half in gnn_probe_f16.slang);
// LayerNorm statistics are accumulated in float.

#ifndef REAL
#define REAL float
#endif

#define THREADS 128
#define MAXW 128

struct Params
{
    uint width;        // W: 32, 64 or 128
    uint nodes;        // nodes of this graph level
    uint grid_x;       // workgroups along x (2D dispatch past 65,535 groups)
    uint weights;      // offset of this round's weights in `wts` (REAL elements)
    uint edges;        // directed edges of this level
    uint mode;         // 0 message passing (fused), 1 encoder, 2 decoder, 3 pool, 4 unpool,
                       // 5 edge update (edge-parallel), 6 node update (after 5)
    uint in_width;     // encoder: input features per node; decoder: outputs per node
    uint max_degree;   // the level's largest in-degree (message passing, mode 0)
};

[[vk::binding(0, 0)]] ConstantBuffer<Params> params;
[[vk::binding(1, 0)]] StructuredBuffer<REAL> wts;
[[vk::binding(2, 0)]] StructuredBuffer<REAL> h_in;
[[vk::binding(3, 0)]] RWStructuredBuffer<REAL> h_out;
[[vk::binding(4, 0)]] StructuredBuffer<REAL> e_in;
[[vk::binding(5, 0)]] RWStructuredBuffer<REAL> e_out;
// Incoming edges of node i: offsets[i] .. offsets[i + 1] (edge ids sorted by destination,
// so an edge's id is its position); `src` holds each edge's source node. Pool and unpool
// reuse them as the coarse node's member list and the fine node's parent.
[[vk::binding(6, 0)]] StructuredBuffer<uint> offsets;
[[vk::binding(7, 0)]] StructuredBuffer<uint> src;

groupshared REAL xs[THREADS / 32][3 * MAXW];
groupshared REAL hid[THREADS];
groupshared float red[THREADS];

float silu(float x)
{
    return x / (1.0 + exp(-x));
}

// Sum of `v` over the W lanes of this slot (all threads take part: barriers).
float slot_sum(float v, uint tid, uint lane, uint w)
{
    red[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = w / 2; s > 0; s >>= 1)
    {
        if (lane < s)
            red[tid] += red[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float r = red[tid - lane];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Two-layer MLP with LayerNorm on its output, for this lane's channel. Its input (`k_in`
// values) is in xs[slot]; `base` is its weights: W1 (k-major, k_in x W), b1, W2 (k-major,
// W x W), b2, gamma, beta.
float mlp(uint base, uint k_in, uint tid, uint slot, uint lane, uint w, bool live)
{
    // The multiply-adds in REAL (half in the f16 variant), the activation and the
    // statistics in float.
    REAL a = REAL(0.0);
    if (live)
    {
        a = wts[base + k_in * w + lane];
        for (uint k = 0; k < k_in; k++)
            a += wts[base + k * w + lane] * xs[slot][k];
        a = REAL(silu(float(a)));
    }
    hid[tid] = a;
    GroupMemoryBarrierWithGroupSync();
    uint b2 = base + k_in * w + w;
    REAL acc = REAL(0.0);
    if (live)
    {
        acc = wts[b2 + w * w + lane];
        for (uint k = 0; k < w; k++)
            acc += wts[b2 + k * w + lane] * hid[tid - lane + k];
    }
    float o = float(acc);
    GroupMemoryBarrierWithGroupSync();
    float mean = slot_sum(o, tid, lane, w) / float(w);
    float d = o - mean;
    float var = slot_sum(d * d, tid, lane, w) / float(w);
    uint g = b2 + w * w + w;
    return float(wts[g + lane]) * d * rsqrt(var + 1e-5) + float(wts[g + w + lane]);
}

uint mlp_size(uint k_in, uint w)
{
    return k_in * w + w + w * w + w + 2 * w;
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void gnn_round(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint w = params.width;
    uint tid = lid.x;
    uint slot = tid / w;
    uint lane = tid % w;
    uint group = gid.y * params.grid_x + gid.x;
    uint node = group * (THREADS / w) + slot;
    bool live = node < params.nodes;

    if (params.mode == 3 || params.mode == 4)
    {
        // Pool (3): a coarse node is the mean of its members. Unpool (4): a fine node adds
        // its coarse parent's features (the U-net's skip is the fine features themselves).
        // (No early return on `live`: WGSL's uniformity analysis would then reject the
        // barriers of the other modes.)
        if (live && params.mode == 3)
        {
            uint a = offsets[node], b = offsets[node + 1];
            float s = 0.0;
            for (uint i = a; i < b; i++)
                s += float(h_in[src[i] * w + lane]);
            h_out[node * w + lane] = REAL(s / float(max(b - a, 1u)));
        }
        else if (live)
        {
            uint p = src[node];
            h_out[node * w + lane] = REAL(float(h_out[node * w + lane]) + float(h_in[p * w + lane]));
        }
        return;
    }
    if (params.mode == 1 || params.mode == 2)
    {
        // Encoder (1): `in_width` node features -> W. Decoder (2): W -> `in_width` outputs
        // (lanes past it idle). Both a two-layer MLP; the encoder's with LayerNorm.
        uint k_in = params.mode == 1 ? params.in_width : w;
        if (live && lane < k_in)
            xs[slot][lane] = h_in[node * k_in + lane];
        GroupMemoryBarrierWithGroupSync();
        float y = mlp(params.weights, k_in, tid, slot, lane, w, live);
        if (live && (params.mode == 1 || lane < params.in_width))
            h_out[node * (params.mode == 1 ? w : params.in_width) + lane] = REAL(y);
        return;
    }

    if (params.mode == 5)
    {
        // Edge-parallel edge update: one slot per edge (`node` is the edge id here; the
        // `offsets` binding holds each edge's destination).
        uint e = node;
        bool on = e < params.edges;
        if (on)
        {
            xs[slot][lane] = e_in[e * w + lane];
            xs[slot][w + lane] = h_in[src[e] * w + lane];
            xs[slot][2 * w + lane] = h_in[offsets[e] * w + lane];
        }
        GroupMemoryBarrierWithGroupSync();
        float y = mlp(params.weights, 3 * w, tid, slot, lane, w, on);
        if (on)
            e_out[e * w + lane] = REAL(float(xs[slot][lane]) + y);
        return;
    }
    if (params.mode == 6)
    {
        // Node update after mode 5: the sum of the new incoming edges (bound as `e_in`), then
        // the node MLP.
        float agg = 0.0;
        float hd = 0.0;
        if (live)
        {
            hd = float(h_in[node * w + lane]);
            for (uint i = offsets[node]; i < offsets[node + 1]; i++)
                agg += float(e_in[i * w + lane]);
            xs[slot][lane] = REAL(hd);
            xs[slot][w + lane] = REAL(agg);
        }
        GroupMemoryBarrierWithGroupSync();
        float y = mlp(params.weights + mlp_size(3 * w, w), 2 * w, tid, slot, lane, w, live);
        if (live)
            h_out[node * w + lane] = REAL(hd + y);
        return;
    }

    // Message passing: the edge MLP on each incoming edge, the sum, the node MLP.
    uint edge_w = params.weights;
    uint node_w = edge_w + mlp_size(3 * w, w);
    uint a = live ? offsets[node] : 0;
    uint b = live ? offsets[node + 1] : 0;
    // Every slot loops to the level's largest in-degree (barriers inside; a uniform bound, as
    // WGSL's uniformity analysis requires).
    uint most = params.max_degree;
    float agg = 0.0;
    float hd = live ? float(h_in[node * w + lane]) : 0.0;
    for (uint j = 0; j < most; j++)
    {
        uint e = a + j;
        bool on = live && e < b;
        if (on)
        {
            uint s = src[e];
            xs[slot][lane] = e_in[e * w + lane];
            xs[slot][w + lane] = h_in[s * w + lane];
            xs[slot][2 * w + lane] = REAL(hd);
        }
        GroupMemoryBarrierWithGroupSync();
        float y = mlp(edge_w, 3 * w, tid, slot, lane, w, on);
        if (on)
        {
            float en = float(xs[slot][lane]) + y;
            e_out[e * w + lane] = REAL(en);
            agg += en;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (live)
    {
        xs[slot][lane] = REAL(hd);
        xs[slot][w + lane] = REAL(agg);
    }
    GroupMemoryBarrierWithGroupSync();
    float y = mlp(node_w, 2 * w, tid, slot, lane, w, live);
    if (live)
        h_out[node * w + lane] = REAL(hd + y);
}

// ---------------------------------------------------------------------------------------
// Tiled kernels (`Probe::with_kernel(.., Kernel::Tiled)`): each round as an edge MLP, a CSR
// sum and a node MLP, the MLPs as GEMM tiles. A workgroup takes TR = 2048 / W rows and all W
// output channels; each thread a 4 x 4 block of them. The input rows are gathered: segment
// A from `h_in` by row (the edge's own features, or the node's), segment B from `e_in` by
// `src[row]` (an edge's source node) or by row (a node's sum), segment C (edges only) from
// `e_in` by `offsets[row]` (the destination). k is taken 8 at a time through group memory.
// The output is A + LN(MLP(x)) (the residual), written to `h_out`. Arithmetic in float.

#define KT 8
groupshared float tx[KT][64];
groupshared float tw[KT][MAXW];
groupshared float th[2048];
groupshared float tstat[2][64];

// params: width, nodes = rows, grid_x, weights, edges (unused), mode: 7 edge MLP (3 segments,
// B by src, C by dst), 8 node MLP (2 segments, B by row), in_width unused.
float gather(uint row, uint k, uint w, uint mode)
{
    uint seg = k / w;
    uint c = k - seg * w;
    if (seg == 0)
        return float(h_in[row * w + c]);
    if (mode == 8)
        return float(e_in[row * w + c]);
    uint n = seg == 1 ? src[row] : offsets[row];
    return float(e_in[n * w + c]);
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void gnn_mlp_tiled(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint w = params.width;
    uint tr = 2048 / w;
    uint t = lid.x;
    uint cg = w / 4;
    uint rb = (t / cg) * 4;   // first of this thread's 4 rows (in the tile)
    uint cb = (t % cg) * 4;   // first of its 4 channels
    uint group = gid.y * params.grid_x + gid.x;
    uint row0 = group * tr;
    uint rows = params.nodes;
    uint segs = params.mode == 7 ? 3 : 2;
    uint k_in = segs * w;
    uint base = params.weights + (params.mode == 8 ? mlp_size(3 * w, w) : 0);
    // The 4 x 4 block as four row vectors (registers: no dynamically indexed local array,
    // which the WGSL path spills).
    float4 bias = float4(float(wts[base + k_in * w + cb]), float(wts[base + k_in * w + cb + 1]), float(wts[base + k_in * w + cb + 2]), float(wts[base + k_in * w + cb + 3]));
    float4 r0 = bias, r1 = bias, r2 = bias, r3 = bias;
    // Layer 1: x (tr x k_in) W1 (k_in x w).
    for (uint k0 = 0; k0 < k_in; k0 += KT)
    {
        for (uint q = t; q < KT * tr; q += THREADS)
        {
            uint kk = q / tr, r = q - kk * tr;
            uint row = row0 + r;
            tx[kk][r] = row < rows ? gather(row, k0 + kk, w, params.mode) : 0.0;
        }
        for (uint q = t; q < KT * w; q += THREADS)
        {
            uint kk = q / w, c = q - kk * w;
            tw[kk][c] = float(wts[base + (k0 + kk) * w + c]);
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint kk = 0; kk < KT; kk++)
        {
            float4 b = float4(tw[kk][cb], tw[kk][cb + 1], tw[kk][cb + 2], tw[kk][cb + 3]);
            r0 += tx[kk][rb] * b;
            r1 += tx[kk][rb + 1] * b;
            r2 += tx[kk][rb + 2] * b;
            r3 += tx[kk][rb + 3] * b;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float4 h0 = float4(silu(r0.x), silu(r0.y), silu(r0.z), silu(r0.w));
    float4 h1 = float4(silu(r1.x), silu(r1.y), silu(r1.z), silu(r1.w));
    float4 h2 = float4(silu(r2.x), silu(r2.y), silu(r2.z), silu(r2.w));
    float4 h3 = float4(silu(r3.x), silu(r3.y), silu(r3.z), silu(r3.w));
    [ForceUnroll]
    for (uint j = 0; j < 4; j++)
    {
        th[(rb + 0) * w + cb + j] = h0[j];
        th[(rb + 1) * w + cb + j] = h1[j];
        th[(rb + 2) * w + cb + j] = h2[j];
        th[(rb + 3) * w + cb + j] = h3[j];
    }
    GroupMemoryBarrierWithGroupSync();
    // Layer 2: hidden (tr x w, in th) W2 (w x w).
    uint b2 = base + k_in * w + w;
    bias = float4(float(wts[b2 + w * w + cb]), float(wts[b2 + w * w + cb + 1]), float(wts[b2 + w * w + cb + 2]), float(wts[b2 + w * w + cb + 3]));
    r0 = bias;
    r1 = bias;
    r2 = bias;
    r3 = bias;
    for (uint k0 = 0; k0 < w; k0 += KT)
    {
        for (uint q = t; q < KT * w; q += THREADS)
        {
            uint kk = q / w, c = q - kk * w;
            tw[kk][c] = float(wts[b2 + (k0 + kk) * w + c]);
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint kk = 0; kk < KT; kk++)
        {
            float4 b = float4(tw[kk][cb], tw[kk][cb + 1], tw[kk][cb + 2], tw[kk][cb + 3]);
            r0 += th[(rb + 0) * w + k0 + kk] * b;
            r1 += th[(rb + 1) * w + k0 + kk] * b;
            r2 += th[(rb + 2) * w + k0 + kk] * b;
            r3 += th[(rb + 3) * w + k0 + kk] * b;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // LayerNorm per row: the outputs through group memory, one thread per row for the stats.
    [ForceUnroll]
    for (uint j = 0; j < 4; j++)
    {
        th[(rb + 0) * w + cb + j] = r0[j];
        th[(rb + 1) * w + cb + j] = r1[j];
        th[(rb + 2) * w + cb + j] = r2[j];
        th[(rb + 3) * w + cb + j] = r3[j];
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < tr)
    {
        float m = 0.0;
        for (uint c = 0; c < w; c++)
            m += th[t * w + c];
        m /= float(w);
        float v = 0.0;
        for (uint c = 0; c < w; c++)
        {
            float d = th[t * w + c] - m;
            v += d * d;
        }
        tstat[0][t] = m;
        tstat[1][t] = rsqrt(v / float(w) + 1e-5);
    }
    GroupMemoryBarrierWithGroupSync();
    uint g = b2 + w * w + w;
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        uint row = row0 + rb + i;
        float4 o = i == 0 ? r0 : (i == 1 ? r1 : (i == 2 ? r2 : r3));
        if (row < rows)
        {
            [ForceUnroll]
            for (uint j = 0; j < 4; j++)
            {
                uint c = cb + j;
                float y = float(wts[g + c]) * (o[j] - tstat[0][rb + i]) * tstat[1][rb + i] + float(wts[g + w + c]);
                h_out[row * w + c] = REAL(float(h_in[row * w + c]) + y);
            }
        }
    }
}

// The sum of each node's incoming edge features (CSR by destination) into `h_out`.
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void gnn_csr_sum(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint w = params.width;
    uint slot = lid.x / w, lane = lid.x % w;
    uint node = (gid.y * params.grid_x + gid.x) * (THREADS / w) + slot;
    if (node >= params.nodes)
        return;
    float s = 0.0;
    for (uint i = offsets[node]; i < offsets[node + 1]; i++)
        s += float(e_in[i * w + lane]);
    h_out[node * w + lane] = REAL(s);
}
