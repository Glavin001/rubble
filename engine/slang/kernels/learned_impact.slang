// Lean MGN-T inference for the Fracture lab (Slang → WGSL → WebGPU / Metal).
// Matches train defaults: H=48, P=16, NODE_IN=46, EDGE_IN=28, 2+2 MP, 1 attn.
// Weight blob layout: learned-impact-slang/1 (see learned_impact/export.py pack_slang).
// Bond mode is compute + viz: predictions are read back for HUD; they do not drive breaks.

static const uint H = 48;
static const uint P = 16;
static const uint H2 = 192;
static const uint NODE_IN = 46;
static const uint EDGE_IN = 28;
static const uint FAIL_BINS = 5; // K+1
static const uint OUT_PER_EDGE = 6; // FAIL_BINS logits + util
static const uint N_HEADS = 4;
static const uint HD = 12;
static const uint WG = 256;
static const uint WG_TOK = 16;

struct Params
{
    uint n_nodes;
    uint n_edges;
    uint n_islands;
    uint block;
    uint mp_pre;
    uint mp_post;
    uint attn_blocks;
    uint n_heads;
    // graph_u layout (uints): islands as 4×uint × n_islands, then csr, then edge pairs, then node_island
    uint off_islands;
    uint off_csr;
    uint off_edges;
    uint off_node_island;
    // work layout (floats)
    uint off_tokens;
    uint off_slice;
    uint off_qkv;
    uint off_output;
    uint off_ffn;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<Params> params;
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;
[[vk::binding(2, 0)]] StructuredBuffer<uint> graph_u;
[[vk::binding(3, 0)]] StructuredBuffer<float> node_in;
[[vk::binding(4, 0)]] StructuredBuffer<float> edge_in;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> node_x;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> edge_x;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> work;

float tok_get(uint i) { return work[params.off_tokens + i]; }
void tok_set(uint i, float v) { work[params.off_tokens + i] = v; }
float slice_get(uint i) { return work[params.off_slice + i]; }
void slice_set(uint i, float v) { work[params.off_slice + i] = v; }
float qkv_get(uint i) { return work[params.off_qkv + i]; }
void qkv_set(uint i, float v) { work[params.off_qkv + i] = v; }
float out_get(uint i) { return work[params.off_output + i]; }
void out_set(uint i, float v) { work[params.off_output + i] = v; }
float ffn_get(uint i) { return work[params.off_ffn + i]; }
void ffn_set(uint i, float v) { work[params.off_ffn + i] = v; }

uint isl_node_begin(uint i) { return graph_u[params.off_islands + i * 4u + 0u]; }
uint isl_node_end(uint i) { return graph_u[params.off_islands + i * 4u + 1u]; }
uint csr_at(uint i) { return graph_u[params.off_csr + i]; }
uint2 edge_ab(uint e) { uint b = params.off_edges + e * 2u; return uint2(graph_u[b], graph_u[b + 1u]); }
uint node_isl(uint n) { return graph_u[params.off_node_island + n]; }

float silu(float x) { return x / (1.0 + exp(-clamp(x, -20.0, 20.0))); }
float softplus(float x) { return x > 20.0 ? x : log(1.0 + exp(x)); }

uint node_enc_bytes() { return H * NODE_IN + H + H * H + H; }
uint edge_enc_bytes() { return H * EDGE_IN + H + H * H + H; }
uint mp_edge_bytes() { return H * 3u * H + H + H * H + H; }
uint mp_node_bytes() { return H * 2u * H + H + H * H + H; }
uint mp_layer_bytes() { return mp_edge_bytes() + mp_node_bytes(); }
uint mp_layers() { return params.mp_pre + params.mp_post; }
uint off_node_enc() { return 0; }
uint off_edge_enc() { return node_enc_bytes(); }
uint off_mp_edge() { return node_enc_bytes() + edge_enc_bytes() + params.block * mp_layer_bytes(); }
uint off_mp_node() { return off_mp_edge() + mp_edge_bytes(); }
// assign_w[P,H] + assign_b[P] + tau_w[H] + tau_b + qkv_w + qkv_b + out_w + out_b + ffn_up + ffn_down
uint attn_stride()
{
    return P * H + P + H + 1u + 3u * H * H + 3u * H + H * H + H + H2 * H + H2 + H * H2 + H;
}
uint off_attn() { return node_enc_bytes() + edge_enc_bytes() + mp_layers() * mp_layer_bytes(); }
uint off_decode() { return off_attn() + params.attn_blocks * attn_stride(); }

float dot_h(uint row_base, in float x[H])
{
    float s = 0.0;
    [unroll]
    for (uint i = 0; i < H; i++)
        s += weights[row_base + i] * x[i];
    return s;
}

float dot_node(uint row_base, in float x[NODE_IN])
{
    float s = 0.0;
    [unroll]
    for (uint i = 0; i < NODE_IN; i++)
        s += weights[row_base + i] * x[i];
    return s;
}

float dot_edge(uint row_base, in float x[EDGE_IN])
{
    float s = 0.0;
    [unroll]
    for (uint i = 0; i < EDGE_IN; i++)
        s += weights[row_base + i] * x[i];
    return s;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void encode_nodes_l1(uint3 id: SV_DispatchThreadID)
{
    uint n = id.x;
    if (n >= params.n_nodes)
        return;
    uint base = off_node_enc();
    float xin[NODE_IN];
    [unroll]
    for (uint i = 0; i < NODE_IN; i++)
        xin[i] = node_in[n * NODE_IN + i];
    [unroll]
    for (uint m = 0; m < H; m++)
        out_set(n * H + m, silu(weights[base + H * NODE_IN + m] + dot_node(base + m * NODE_IN, xin)));
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void encode_nodes_l2(uint3 id: SV_DispatchThreadID)
{
    uint total = params.n_nodes * H;
    uint idx = id.x;
    if (idx >= total)
        return;
    uint n = idx / H;
    uint o = idx - n * H;
    uint base = off_node_enc();
    uint b2 = base + H * NODE_IN + H;
    float mid[H];
    [unroll]
    for (uint i = 0; i < H; i++)
        mid[i] = out_get(n * H + i);
    node_x[n * H + o] = silu(weights[b2 + H * H + o] + dot_h(b2 + o * H, mid));
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void encode_edges_l1(uint3 id: SV_DispatchThreadID)
{
    uint e = id.x;
    if (e >= params.n_edges)
        return;
    uint base = off_edge_enc();
    float xin[EDGE_IN];
    [unroll]
    for (uint i = 0; i < EDGE_IN; i++)
        xin[i] = edge_in[e * EDGE_IN + i];
    [unroll]
    for (uint m = 0; m < H; m++)
        ffn_set(e * H + m, silu(weights[base + H * EDGE_IN + m] + dot_edge(base + m * EDGE_IN, xin)));
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void encode_edges_l2(uint3 id: SV_DispatchThreadID)
{
    uint total = params.n_edges * H;
    uint idx = id.x;
    if (idx >= total)
        return;
    uint e = idx / H;
    uint o = idx - e * H;
    uint base = off_edge_enc();
    uint b2 = base + H * EDGE_IN + H;
    float mid[H];
    [unroll]
    for (uint i = 0; i < H; i++)
        mid[i] = ffn_get(e * H + i);
    edge_x[e * H + o] = silu(weights[b2 + H * H + o] + dot_h(b2 + o * H, mid));
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void mp_edges(uint3 id: SV_DispatchThreadID)
{
    uint e = id.x;
    if (e >= params.n_edges)
        return;
    uint2 ab = edge_ab(e);
    uint in_dim = 3u * H;
    uint base = off_mp_edge();
    float ea[H];
    float na[H];
    float nb[H];
    [unroll]
    for (uint i = 0; i < H; i++)
    {
        ea[i] = edge_x[e * H + i];
        na[i] = node_x[ab.x * H + i];
        nb[i] = node_x[ab.y * H + i];
    }
    float mid[H];
    [unroll]
    for (uint m = 0; m < H; m++)
    {
        // PyTorch concat order: [node_a, node_b, edge]
        float s = weights[base + H * in_dim + m];
        [unroll]
        for (uint i = 0; i < H; i++)
        {
            s += weights[base + m * in_dim + i] * na[i];
            s += weights[base + m * in_dim + H + i] * nb[i];
            s += weights[base + m * in_dim + 2u * H + i] * ea[i];
        }
        mid[m] = silu(s);
    }
    uint b2 = base + H * in_dim + H;
    [unroll]
    for (uint o = 0; o < H; o++)
        edge_x[e * H + o] = ea[o] + silu(weights[b2 + H * H + o] + dot_h(b2 + o * H, mid));
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void mp_nodes(uint3 id: SV_DispatchThreadID)
{
    uint n = id.x;
    if (n >= params.n_nodes)
        return;
    float agg[H];
    [unroll]
    for (uint k = 0; k < H; k++)
        agg[k] = 0.0;
    uint begin = csr_at(n);
    uint end = csr_at(n + 1u);
    for (uint at = begin; at < end; at++)
    {
        uint e = csr_at(at) >> 1;
        [unroll]
        for (uint i = 0; i < H; i++)
            agg[i] += edge_x[e * H + i];
    }
    float nx[H];
    [unroll]
    for (uint i = 0; i < H; i++)
        nx[i] = node_x[n * H + i];
    uint base = off_mp_node();
    uint in_dim = 2u * H;
    float mid[H];
    [unroll]
    for (uint m = 0; m < H; m++)
    {
        float s = weights[base + H * in_dim + m];
        [unroll]
        for (uint i = 0; i < H; i++)
        {
            s += weights[base + m * in_dim + i] * nx[i];
            s += weights[base + m * in_dim + H + i] * agg[i];
        }
        mid[m] = silu(s);
    }
    uint b2 = base + H * in_dim + H;
    float y[H];
    [unroll]
    for (uint o2 = 0; o2 < H; o2++)
        y[o2] = nx[o2] + silu(weights[b2 + H * H + o2] + dot_h(b2 + o2 * H, mid));
    float mean = 0.0;
    [unroll]
    for (uint m = 0; m < H; m++)
        mean += y[m];
    mean /= (float)H;
    float var = 0.0;
    [unroll]
    for (uint m2 = 0; m2 < H; m2++)
    {
        float d = y[m2] - mean;
        var += d * d;
    }
    float inv = rsqrt(var / (float)H + 1e-5);
    [unroll]
    for (uint m3 = 0; m3 < H; m3++)
        node_x[n * H + m3] = (y[m3] - mean) * inv;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attn_slice(uint3 id: SV_DispatchThreadID)
{
    uint n = id.x;
    if (n >= params.n_nodes)
        return;
    uint base = off_attn() + params.block * attn_stride();
    float nx[H];
    [unroll]
    for (uint t = 0; t < H; t++)
        nx[t] = node_x[n * H + t];
    uint tau_base = base + P * H + P;
    float tau_logit = weights[tau_base + H];
    [unroll]
    for (uint t = 0; t < H; t++)
        tau_logit += weights[tau_base + t] * nx[t];
    float tau = 0.5 + softplus(tau_logit);
    float logits[P];
    float mmax = -1e30;
    [unroll]
    for (uint p = 0; p < P; p++)
    {
        float s = weights[base + P * H + p];
        [unroll]
        for (uint i = 0; i < H; i++)
            s += weights[base + p * H + i] * nx[i];
        logits[p] = s / max(tau, 1e-3);
        mmax = max(mmax, logits[p]);
    }
    float sum = 0.0;
    [unroll]
    for (uint p2 = 0; p2 < P; p2++)
    {
        logits[p2] = exp(logits[p2] - mmax);
        sum += logits[p2];
    }
    float inv = 1.0 / max(sum, 1e-9);
    [unroll]
    for (uint p3 = 0; p3 < P; p3++)
        slice_set(n * P + p3, logits[p3] * inv);
}

[shader("compute")]
[numthreads(WG_TOK, 1, 1)]
void attn_tokens(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID)
{
    if (gid.x >= params.n_islands || tid.x >= P)
        return;
    uint nb = isl_node_begin(gid.x); uint ne = isl_node_end(gid.x);
    uint p = tid.x;
    uint tok_base = gid.x * P * H;
    float acc[H];
    [unroll]
    for (uint h = 0; h < H; h++)
        acc[h] = 0.0;
    float wsum = 0.0;
    for (uint n = nb; n < ne; n++)
    {
        float w = slice_get(n * P + p);
        wsum += w;
        [unroll]
        for (uint h2 = 0; h2 < H; h2++)
            acc[h2] += w * node_x[n * H + h2];
    }
    float inv = 1.0 / max(wsum, 1e-9);
    [unroll]
    for (uint h3 = 0; h3 < H; h3++)
        tok_set(tok_base + p * H + h3, acc[h3] * inv);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attn_qkv(uint3 id: SV_DispatchThreadID)
{
    uint total = params.n_islands * P * 3u * H;
    uint idx = id.x;
    if (idx >= total)
        return;
    uint per_tok = 3u * H;
    uint flat_tok = idx / per_tok;
    uint rem = idx - flat_tok * per_tok;
    uint part = rem / H;
    uint o = rem - part * H;
    uint island = flat_tok / P;
    uint p = flat_tok - island * P;
    uint base = off_attn() + params.block * attn_stride();
    uint qkv = base + P * H + P + H + 1u;
    float tok[H];
    [unroll]
    for (uint i = 0; i < H; i++)
        tok[i] = tok_get(island * P * H + p * H + i);
    float s = weights[qkv + 3u * H * H + part * H + o];
    [unroll]
    for (uint i = 0; i < H; i++)
        s += weights[qkv + part * H * H + o * H + i] * tok[i];
    qkv_set(island * P * 3u * H + p * 3u * H + part * H + o, s);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attn_mha(uint3 id: SV_DispatchThreadID)
{
    uint total = params.n_islands * P;
    uint idx = id.x;
    if (idx >= total)
        return;
    uint island = idx / P;
    uint qi = idx - island * P;
    float scale = rsqrt((float)HD);
    uint qbase = island * P * 3u * H;
    uint base = off_attn() + params.block * attn_stride();
    uint out_w = base + P * H + P + H + 1u + 3u * H * H + 3u * H;
    float out_row[H];
    [unroll]
    for (uint z = 0; z < H; z++)
        out_row[z] = 0.0;
    [unroll]
    for (uint head = 0; head < N_HEADS; head++)
    {
        uint ho = head * HD;
        float scores[P];
        float mmax = -1e30;
        [unroll]
        for (uint kj = 0; kj < P; kj++)
        {
            float s = 0.0;
            [unroll]
            for (uint d = 0; d < HD; d++)
                s += qkv_get(qbase + qi * 3u * H + ho + d) * qkv_get(qbase + kj * 3u * H + H + ho + d);
            scores[kj] = s * scale;
            mmax = max(mmax, scores[kj]);
        }
        float sum = 0.0;
        [unroll]
        for (uint kj2 = 0; kj2 < P; kj2++)
        {
            scores[kj2] = exp(scores[kj2] - mmax);
            sum += scores[kj2];
        }
        float inv = 1.0 / max(sum, 1e-9);
        [unroll]
        for (uint d2 = 0; d2 < HD; d2++)
        {
            float acc = 0.0;
            [unroll]
            for (uint kj3 = 0; kj3 < P; kj3++)
                acc += scores[kj3] * inv * qkv_get(qbase + kj3 * 3u * H + 2u * H + ho + d2);
            out_row[ho + d2] = acc;
        }
    }
    [unroll]
    for (uint o2 = 0; o2 < H; o2++)
        tok_set(island * P * H + qi * H + o2, weights[out_w + H * H + o2] + dot_h(out_w + o2 * H, out_row));
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attn_deslice_mix(uint3 id: SV_DispatchThreadID)
{
    uint n = id.x;
    if (n >= params.n_nodes)
        return;
    uint island = node_isl(n);
    uint tok_base = island * P * H;
    float y[H];
    [unroll]
    for (uint h = 0; h < H; h++)
        y[h] = node_x[n * H + h];
    [unroll]
    for (uint p = 0; p < P; p++)
    {
        float w = slice_get(n * P + p);
        [unroll]
        for (uint h2 = 0; h2 < H; h2++)
            y[h2] += w * tok_get(tok_base + p * H + h2);
    }
    [unroll]
    for (uint h3 = 0; h3 < H; h3++)
        out_set(n * H + h3, y[h3]);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attn_ffn_up(uint3 id: SV_DispatchThreadID)
{
    uint total = params.n_nodes * H2;
    uint idx = id.x;
    if (idx >= total)
        return;
    uint n = idx / H2;
    uint o = idx - n * H2;
    float y[H];
    [unroll]
    for (uint i = 0; i < H; i++)
        y[i] = out_get(n * H + i);
    uint base = off_attn() + params.block * attn_stride();
    uint ffn = base + P * H + P + H + 1u + 3u * H * H + 3u * H + H * H + H;
    float s = weights[ffn + H2 * H + o];
    [unroll]
    for (uint i = 0; i < H; i++)
        s += weights[ffn + o * H + i] * y[i];
    ffn_set(n * H2 + o, silu(s));
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void attn_ffn_down(uint3 id: SV_DispatchThreadID)
{
    uint total = params.n_nodes * H;
    uint idx = id.x;
    if (idx >= total)
        return;
    uint n = idx / H;
    uint o = idx - n * H;
    uint base = off_attn() + params.block * attn_stride();
    uint ffn = base + P * H + P + H + 1u + 3u * H * H + 3u * H + H * H + H;
    uint ffn2 = ffn + H2 * H + H2;
    float s2 = weights[ffn2 + H * H2 + o];
    [unroll]
    for (uint i = 0; i < H2; i++)
        s2 += weights[ffn2 + o * H2 + i] * ffn_get(n * H2 + i);
    node_x[n * H + o] = out_get(n * H + o) + s2;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void decode_bonds(uint3 id: SV_DispatchThreadID)
{
    uint e = id.x;
    if (e >= params.n_edges)
        return;
    uint base = off_decode();
    float xin[H];
    [unroll]
    for (uint i = 0; i < H; i++)
        xin[i] = edge_x[e * H + i];
    [unroll]
    for (uint c = 0; c < FAIL_BINS; c++)
        out_set(e * OUT_PER_EDGE + c, weights[base + FAIL_BINS * H + c] + dot_h(base + c * H, xin));
    uint ub = base + FAIL_BINS * H + FAIL_BINS;
    out_set(e * OUT_PER_EDGE + FAIL_BINS, weights[ub + H] + dot_h(ub, xin));
}
