// P3: P2's impact-outcome GNN (learned_impact/outcome/model.py) as three kernels over three
// buffers, driven by the op list `stress_learned::outcome_program::compile` builds (see that
// module for the layout of every parameter word; `Program::run_cpu` is these kernels'
// arithmetic on the CPU). f32 throughout.
//
// - gnn_mlp: per row, an input assembled from up to six segments (by row, gathered through
//   the table, or a constant row; optionally standardised), Linear · SiLU · Linear ·
//   [LayerNorm], times a scale, plus an optional residual row, written or accumulated. A
//   projection segment adds a precomputed first-layer row to the pre-activation instead (the
//   node-projection lever); an op without a second layer outputs the pre-activation.
//   Weights are k-major, so lane c reads column c contiguously across rows.
// - gnn_csr: out[r] (+)= sum_k w_k * arena[o_k + c], entries (offset, f32 weight bits).
// - gnn_fibre: a coarse edge's (R_body^T (p_i - p_j) / L, |.|), L its call's.
//
// A workgroup of 128 threads holds 128 / lanes rows, `lanes` threads (one channel each) per
// row. Control flow around barriers depends on parameters only (uniform), as WGSL's
// uniformity analysis requires.

#define THREADS 128
#define MAX_K 384
#define NONE 0xffffffffu

struct Params
{
    uint4 p[16];
};

[[vk::binding(0, 0)]] ConstantBuffer<Params> params;
[[vk::binding(1, 0)]] StructuredBuffer<float> wts;
[[vk::binding(2, 0)]] StructuredBuffer<uint> table;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> arena;

uint P(uint i)
{
    return params.p[i >> 2][i & 3];
}

groupshared float xs[THREADS / 32][MAX_K];
groupshared float hid[THREADS];
groupshared float red[THREADS];

float silu(float x)
{
    return x / (1.0 + exp(-x));
}

// Sum of `v` over the `lanes` lanes of this slot (every thread takes part: barriers).
float slot_sum(float v, uint tid, uint lane, uint lanes)
{
    red[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = lanes / 2; s > 0; s >>= 1)
    {
        if (lane < s)
            red[tid] += red[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float r = red[tid - lane];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Parameter words (outcome_program::mlp_p).
#define N_SEG 4
#define SEG0 8
#define W1 40
#define K_IN 41
#define HIDDEN 42
#define OUT 43
#define W2 44
#define LN 45
#define OUT_BASE 46
#define OUT_STRIDE 47
#define RES_BASE 48
#define RES_STRIDE 49
#define FLAGS 50
#define SCALE 51
#define B1 52
#define B2 53

#define SEG_ROW 0
#define SEG_GATHER 1
#define SEG_CONST 2
#define SEG_PROJ_GATHER 3
#define SEG_PROJ_ROW 4

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void gnn_mlp(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint lanes = P(3);
    uint tid = lid.x;
    uint slot = tid / lanes;
    uint lane = tid % lanes;
    uint row = (gid.y * P(2) + gid.x) * (THREADS / lanes) + slot;
    bool live = row < P(1);
    uint hidden = P(HIDDEN);
    uint out = P(OUT);

    // Stage the input row; projection rows go straight into this lane's pre-activation.
    float pre = 0.0;
    uint k = 0;
    uint nseg = P(N_SEG);
    for (uint s = 0; s < nseg; s++)
    {
        uint o = SEG0 + 5 * s;
        uint base = P(o), idx = P(o + 1), width = P(o + 2), mode = P(o + 3), norm = P(o + 4);
        if (live)
        {
            uint r = row;
            if (mode == SEG_GATHER || mode == SEG_PROJ_GATHER)
                r = table[idx + row];
            else if (mode == SEG_CONST)
                r = idx;
            if (mode >= SEG_PROJ_GATHER)
            {
                if (lane < hidden)
                    pre += arena[base + r * width + lane];
            }
            else
            {
                for (uint c = lane; c < width; c += lanes)
                {
                    float v = arena[base + r * width + c];
                    if (norm != NONE)
                        v = (v - wts[norm + c]) / wts[norm + width + c];
                    xs[slot][k + c] = v;
                }
            }
        }
        if (mode < SEG_PROJ_GATHER)
            k += width;
    }
    GroupMemoryBarrierWithGroupSync();

    uint k_in = P(K_IN);
    uint w1 = P(W1);
    if (live && lane < hidden)
    {
        float a = P(B1) == NONE ? 0.0 : wts[P(B1) + lane];
        for (uint kk = 0; kk < k_in; kk++)
            a += wts[w1 + kk * hidden + lane] * xs[slot][kk];
        pre += a;
    }

    float y = 0.0;
    if (P(W2) == NONE)
    {
        y = pre;
    }
    else
    {
        hid[tid] = (live && lane < hidden) ? silu(pre) : 0.0;
        GroupMemoryBarrierWithGroupSync();
        if (live && lane < out)
        {
            uint w2 = P(W2);
            float a = wts[P(B2) + lane];
            for (uint kk = 0; kk < hidden; kk++)
                a += wts[w2 + kk * out + lane] * hid[tid - lane + kk];
            y = a;
        }
        if (P(LN) != NONE)
        {
            // LayerNorm spans every lane (the compiler checks out == lanes).
            float mean = slot_sum(y, tid, lane, lanes) / float(out);
            float d = y - mean;
            float var = slot_sum(d * d, tid, lane, lanes) / float(out);
            y = d * (1.0 / sqrt(var + 1e-5)) * wts[P(LN) + lane] + wts[P(LN) + out + lane];
        }
    }

    if (live && lane < out)
    {
        float v = asfloat(P(SCALE)) * y;
        if (P(RES_BASE) != NONE)
            v += arena[P(RES_BASE) + row * P(RES_STRIDE) + lane];
        uint o = P(OUT_BASE) + row * P(OUT_STRIDE) + lane;
        if ((P(FLAGS) & 1u) != 0)
            arena[o] = arena[o] + v;
        else
            arena[o] = v;
    }
}

// outcome_program::csr_p
#define C_WIDTH 4
#define C_OFFSETS 5
#define C_ENTRIES 6
#define C_OUT_BASE 7
#define C_OUT_STRIDE 8
#define C_FLAGS 9

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void gnn_csr(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint lanes = P(3);
    uint slot = lid.x / lanes, lane = lid.x % lanes;
    uint row = (gid.y * P(2) + gid.x) * (THREADS / lanes) + slot;
    if (row >= P(1) || lane >= P(C_WIDTH))
        return;
    uint e0 = table[P(C_OFFSETS) + row], e1 = table[P(C_OFFSETS) + row + 1];
    uint ent = P(C_ENTRIES);
    float s = 0.0;
    for (uint e = e0; e < e1; e++)
        s += arena[table[ent + 2 * e] + lane] * asfloat(table[ent + 2 * e + 1]);
    uint o = P(C_OUT_BASE) + row * P(C_OUT_STRIDE) + lane;
    if ((P(C_FLAGS) & 1u) != 0)
        arena[o] = arena[o] + s;
    else
        arena[o] = s;
}

// outcome_program::fibre_p
#define F_POS 4
#define F_TRIPLES 5
#define F_Q 6
#define F_OUT_BASE 7

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void gnn_fibre(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint r = (gid.y * P(2) + gid.x) * THREADS + lid.x;
    if (r >= P(1))
        return;
    // (i, j, chunk of i, its call's length scale L as f32 bits)
    uint t = P(F_TRIPLES) + 4 * r;
    uint i = table[t], j = table[t + 1], c = table[t + 2];
    uint pb = P(F_POS);
    float3 d = float3(arena[pb + 3 * i] - arena[pb + 3 * j], arena[pb + 3 * i + 1] - arena[pb + 3 * j + 1], arena[pb + 3 * i + 2] - arena[pb + 3 * j + 2]);
    uint qb = P(F_Q) + 4 * c;
    float qw = arena[qb];
    float3 u = -float3(arena[qb + 1], arena[qb + 2], arena[qb + 3]);
    float3 tt = 2.0 * cross(u, d);
    float L = asfloat(table[t + 3]);
    float3 rr = (d + qw * tt + cross(u, tt)) / L;
    uint o = P(F_OUT_BASE) + 4 * r;
    arena[o] = rr.x;
    arena[o + 1] = rr.y;
    arena[o + 2] = rr.z;
    arena[o + 3] = sqrt(rr.x * rr.x + rr.y * rr.y + rr.z * rr.z);
}

// ---------------------------------------------------------------------------------------
// gnn_mlp_tiled: gnn_mlp's computation (the same parameter words) as GEMM tiles, for ops
// with many rows (the P0 probe's layout). A workgroup takes TR = 2048 / lanes rows and all
// `lanes` channels; each thread a 4 x 4 block (rows rb..rb+3, channels cb..cb+3). The input
// is staged KT columns at a time, segment by segment, through group memory; projection
// segments add their rows to the pre-activation block. Requires hidden == lanes and
// out <= lanes; LayerNorm (out == lanes) per row through group memory.

#define KT 8
groupshared float tx[KT][64];
groupshared float tw[KT][64];
groupshared float th[2048];
groupshared float tstat[2][64];

float seg_value(uint base, uint idx, uint width, uint mode, uint norm, uint row, uint c)
{
    if (c >= width)
        return 0.0;
    uint r = row;
    if (mode == SEG_GATHER)
        r = table[idx + row];
    else if (mode == SEG_CONST)
        r = idx;
    float v = arena[base + r * width + c];
    if (norm != NONE)
        v = (v - wts[norm + c]) / wts[norm + width + c];
    return v;
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void gnn_mlp_tiled(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint lanes = P(3);
    uint tr = 2048 / lanes;
    uint t = lid.x;
    uint cg = lanes / 4;
    uint rb = (t / cg) * 4;
    uint cb = (t % cg) * 4;
    uint row0 = (gid.y * P(2) + gid.x) * tr;
    uint rows = P(1);
    uint hidden = P(HIDDEN);
    uint out = P(OUT);
    uint w1 = P(W1);

    float4 bias = float4(0.0, 0.0, 0.0, 0.0);
    if (P(B1) != NONE)
        bias = float4(wts[P(B1) + cb], wts[P(B1) + cb + 1], wts[P(B1) + cb + 2], wts[P(B1) + cb + 3]);
    float4 r0 = bias, r1 = bias, r2 = bias, r3 = bias;

    // Layer 1 over the staged segments, KT columns at a time.
    uint nseg = P(N_SEG);
    uint koff = 0;
    for (uint s = 0; s < nseg; s++)
    {
        uint o = SEG0 + 5 * s;
        uint base = P(o), idx = P(o + 1), width = P(o + 2), mode = P(o + 3), norm = P(o + 4);
        if (mode >= SEG_PROJ_GATHER)
            continue;
        for (uint k0 = 0; k0 < width; k0 += KT)
        {
            for (uint q = t; q < KT * tr; q += THREADS)
            {
                uint kk = q / tr, r = q - kk * tr;
                uint row = row0 + r;
                tx[kk][r] = row < rows ? seg_value(base, idx, width, mode, norm, row, k0 + kk) : 0.0;
            }
            for (uint q = t; q < KT * lanes; q += THREADS)
            {
                uint kk = q / lanes, c = q - kk * lanes;
                tw[kk][c] = (k0 + kk < width) ? wts[w1 + (koff + k0 + kk) * hidden + c] : 0.0;
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
        koff += width;
    }
    // Projection rows (pre-activations of node blocks) added per row.
    for (uint s = 0; s < nseg; s++)
    {
        uint o = SEG0 + 5 * s;
        uint base = P(o), idx = P(o + 1), mode = P(o + 3);
        if (mode < SEG_PROJ_GATHER)
            continue;
        [ForceUnroll]
        for (uint i = 0; i < 4; i++)
        {
            uint row = row0 + rb + i;
            if (row < rows)
            {
                uint r = mode == SEG_PROJ_GATHER ? table[idx + row] : row;
                float4 pv = float4(arena[base + r * hidden + cb], arena[base + r * hidden + cb + 1], arena[base + r * hidden + cb + 2], arena[base + r * hidden + cb + 3]);
                if (i == 0) r0 += pv;
                else if (i == 1) r1 += pv;
                else if (i == 2) r2 += pv;
                else r3 += pv;
            }
        }
    }

    float4 y0 = r0, y1 = r1, y2 = r2, y3 = r3;
    if (P(W2) != NONE)
    {
        // SiLU, then layer 2 from group memory.
        [ForceUnroll]
        for (uint j = 0; j < 4; j++)
        {
            th[(rb + 0) * lanes + cb + j] = silu(r0[j]);
            th[(rb + 1) * lanes + cb + j] = silu(r1[j]);
            th[(rb + 2) * lanes + cb + j] = silu(r2[j]);
            th[(rb + 3) * lanes + cb + j] = silu(r3[j]);
        }
        GroupMemoryBarrierWithGroupSync();
        uint w2 = P(W2);
        float4 b2 = float4(0.0, 0.0, 0.0, 0.0);
        [ForceUnroll]
        for (uint j = 0; j < 4; j++)
            b2[j] = cb + j < out ? wts[P(B2) + cb + j] : 0.0;
        y0 = b2;
        y1 = b2;
        y2 = b2;
        y3 = b2;
        for (uint k0 = 0; k0 < hidden; k0 += KT)
        {
            for (uint q = t; q < KT * lanes; q += THREADS)
            {
                uint kk = q / lanes, c = q - kk * lanes;
                tw[kk][c] = c < out ? wts[w2 + (k0 + kk) * out + c] : 0.0;
            }
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint kk = 0; kk < KT; kk++)
            {
                float4 b = float4(tw[kk][cb], tw[kk][cb + 1], tw[kk][cb + 2], tw[kk][cb + 3]);
                y0 += th[(rb + 0) * lanes + k0 + kk] * b;
                y1 += th[(rb + 1) * lanes + k0 + kk] * b;
                y2 += th[(rb + 2) * lanes + k0 + kk] * b;
                y3 += th[(rb + 3) * lanes + k0 + kk] * b;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (P(LN) != NONE)
        {
            [ForceUnroll]
            for (uint j = 0; j < 4; j++)
            {
                th[(rb + 0) * lanes + cb + j] = y0[j];
                th[(rb + 1) * lanes + cb + j] = y1[j];
                th[(rb + 2) * lanes + cb + j] = y2[j];
                th[(rb + 3) * lanes + cb + j] = y3[j];
            }
            GroupMemoryBarrierWithGroupSync();
            if (t < tr)
            {
                float m = 0.0;
                for (uint c = 0; c < out; c++)
                    m += th[t * lanes + c];
                m /= float(out);
                float v = 0.0;
                for (uint c = 0; c < out; c++)
                {
                    float d = th[t * lanes + c] - m;
                    v += d * d;
                }
                tstat[0][t] = m;
                tstat[1][t] = 1.0 / sqrt(v / float(out) + 1e-5);
            }
            GroupMemoryBarrierWithGroupSync();
            uint g = P(LN);
            [ForceUnroll]
            for (uint j = 0; j < 4; j++)
            {
                uint c = cb + j;
                y0[j] = (y0[j] - tstat[0][rb + 0]) * tstat[1][rb + 0] * wts[g + c] + wts[g + out + c];
                y1[j] = (y1[j] - tstat[0][rb + 1]) * tstat[1][rb + 1] * wts[g + c] + wts[g + out + c];
                y2[j] = (y2[j] - tstat[0][rb + 2]) * tstat[1][rb + 2] * wts[g + c] + wts[g + out + c];
                y3[j] = (y3[j] - tstat[0][rb + 3]) * tstat[1][rb + 3] * wts[g + c] + wts[g + out + c];
            }
        }
    }

    float scale = asfloat(P(SCALE));
    uint res = P(RES_BASE);
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        uint row = row0 + rb + i;
        float4 y = i == 0 ? y0 : (i == 1 ? y1 : (i == 2 ? y2 : y3));
        if (row < rows)
        {
            [ForceUnroll]
            for (uint j = 0; j < 4; j++)
            {
                uint c = cb + j;
                if (c < out)
                {
                    float v = scale * y[j];
                    if (res != NONE)
                        v += arena[res + row * P(RES_STRIDE) + c];
                    uint o = P(OUT_BASE) + row * P(OUT_STRIDE) + c;
                    if ((P(FLAGS) & 1u) != 0)
                        arena[o] = arena[o] + v;
                    else
                        arena[o] = v;
                }
            }
        }
    }
}
