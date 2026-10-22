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
