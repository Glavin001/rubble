// The resident (native) Blast solve of a large island over many groups: blast_native.slang's
// recurrence, step for step (its comments and stress-ref blast_bond/native.rs give the math),
// with each step's work spread over the island's nodes or bonds in blocks of BLAST_THREADS and
// each sum a fixed-order reduction of the blocks' partial sums (no float atomics).
// blast_native.slang solves the islands of at most `wide_min` nodes, one group each; this the
// larger ones (stressgpu's large components take its cooperative kernel; CuMetal's one group).
//
// Dispatch: a large island is a row (w_classify lists them); block kernels run over (blocks,
// rows), row kernels over rows. Each iteration is a fixed sequence the host records
// `max_iterations` times; a row that converged or stopped makes every later kernel a no-op.
//
// Wide buffer: per row its state (ROW_WORDS float4), then per row and block its partial sums
// (PARTIAL float4). Meta: the large rows' count and indirect sizes, then each row's island.

import common;
import blast_common;

#include "blast_native_ops.slangh"

[[vk::binding(10, 0)]] RWStructuredBuffer<float4> wide;
[[vk::binding(11, 0)]] RWStructuredBuffer<Atomic<uint>> meta;
// blast_topology.slang's `args` (word 3: the island count).
[[vk::binding(12, 0)]] StructuredBuffer<uint> topo_args;

static const uint ROW_WORDS = 16;
static const uint PARTIAL = 6;
// State slots of a row.
static const uint S_SCALARS = 0;  // (delta^2, num, gamma_prev, best)
static const uint S_STEP = 1;     // (gamma, alpha, beta, force step)
static const uint S_COUNT = 2;    // (force start, travel, iteration bits, flags bits)
static const uint S_CENTRE = 3;   // (centre, best iteration bits)
static const uint S_COEF = 4;     // 4, 5: the projection's coefficients (w, t)
static const uint S_GRAM = 6;     // 6..14: the rigid modes' Gram matrix (36 floats)
// Flags.
static const uint F_ACTIVE = 1;
static const uint F_CONVERGED = 2;
static const uint F_FAILED = 4;
static const uint F_FREE = 8;
static const uint F_ZERO = 16;
static const uint F_HOLD = 32;
static const uint F_REBUILD = 64;
static const uint F_SKIPPED = 128;
// Meta words.
static const uint M_COUNT = 0;
static const uint M_NODE_BLOCKS = 1;
static const uint M_BOND_BLOCKS = 2;
static const uint M_ARGS = 4;     // node grid (4..6), bond grid (8..10), rows (12..14)
static const uint M_ROWS = 16;

uint st(uint row, uint slot) { return row * ROW_WORDS + slot; }
uint pt(uint row, uint block, uint k) { return params.wide_rows * ROW_WORDS + (row * params.wide_blocks + block) * PARTIAL + k; }
uint row_island(uint row) { return meta[M_ROWS + row].load(); }
uint flags_of(uint row) { return asuint(wide[st(row, S_COUNT)].w); }
void set_flags(uint row, uint f) { float4 c = wide[st(row, S_COUNT)]; c.w = asfloat(f); wide[st(row, S_COUNT)] = c; }
uint iteration_of(uint row) { return asuint(wide[st(row, S_COUNT)].z); }
uint blocks(uint n) { return (n + BLAST_THREADS - 1) / BLAST_THREADS; }

// ------------------------------------------------------------------ classification

[shader("compute")]
[numthreads(1, 1, 1)]
void w_reset(uint3 id : SV_DispatchThreadID)
{
    meta[M_COUNT].store(0);
    meta[M_NODE_BLOCKS].store(0);
    meta[M_BOND_BLOCKS].store(0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void w_classify(uint3 id : SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= topo_args[3])
        return;
    uint4 r = islands[k];
    if (r.y - r.x <= params.wide_min)
        return;
    uint row = meta[M_COUNT].add(1);
    if (row < params.wide_rows)
        meta[M_ROWS + row].store(k);
    meta[M_NODE_BLOCKS].max(blocks(r.y - r.x));
    meta[M_BOND_BLOCKS].max(blocks(r.w - r.z));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void w_args(uint3 id : SV_DispatchThreadID)
{
    uint rows = min(meta[M_COUNT].load(), params.wide_rows);
    uint nb = rows > 0 ? meta[M_NODE_BLOCKS].load() : 0;
    uint bb = rows > 0 ? meta[M_BOND_BLOCKS].load() : 0;
    meta[M_ARGS].store(nb);
    meta[M_ARGS + 1].store(rows);
    meta[M_ARGS + 2].store(1);
    meta[M_ARGS + 4].store(bb);
    meta[M_ARGS + 5].store(rows);
    meta[M_ARGS + 6].store(1);
    meta[M_ARGS + 8].store(rows);
    meta[M_ARGS + 9].store(1);
    meta[M_ARGS + 10].store(1);
}

// ------------------------------------------------------------------ blocks and sums

// A block of a row's node (or bond) list: its range, uniform over the group.
struct Block
{
    uint row;
    uint island;
    uint begin;
    uint end;
    uint flags;
};

Block node_block(uint3 group, uint tid)
{
    Block b;
    b.row = group.y;
    uint4 u = group_uniform4(uint4(row_island(b.row), 0, 0, flags_of(b.row)), tid);
    b.island = u.x;
    b.flags = u.w;
    uint4 r = islands[b.island];
    uint4 g = group_uniform4(r, tid);
    b.begin = g.x + group.x * BLAST_THREADS;
    b.end = min(b.begin + BLAST_THREADS, g.y);
    return b;
}

Block bond_block(uint3 group, uint tid)
{
    Block b = node_block(group, tid);
    uint4 g = group_uniform4(islands[b.island], tid);
    b.begin = g.z + group.x * BLAST_THREADS;
    b.end = min(b.begin + BLAST_THREADS, g.w);
    return b;
}

// The group's sums of two float4 (thread values), written to the block's partial slot `k`.
void write_partial(uint tid, uint row, uint block, uint k, float4 v)
{
    float2 a = blast_group_sum(tid, v.xy);
    float2 b = blast_group_sum(tid, v.zw);
    if (tid == 0)
        wide[pt(row, block, k)] = float4(a, b);
}

// A row's sum of partial slot `k` over its `count` blocks, the same on every thread.
float4 row_sum(uint tid, uint row, uint count, uint k)
{
    float4 s = float4(0.0);
    for (uint b = tid; b < count; b += BLAST_THREADS)
        s += wide[pt(row, b, k)];
    float2 x = blast_group_sum(tid, s.xy);
    float2 y = blast_group_sum(tid, s.zw);
    return float4(x, y);
}

groupshared float g_wide_max[BLAST_THREADS];
groupshared float g_wide_max_out;

float wide_group_max(uint tid, float v)
{
    g_wide_max[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = BLAST_THREADS / 2; k > 0; k >>= 1)
    {
        if (tid < k)
            g_wide_max[tid] = max(g_wide_max[tid], g_wide_max[tid + k]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0)
        g_wide_max_out = g_wide_max[0];
    float m;
    __target_switch
    {
    case wgsl:
        m = workgroupUniformLoad(g_wide_max_out);
    default:
        GroupMemoryBarrierWithGroupSync();
        m = g_wide_max_out;
    }
    GroupMemoryBarrierWithGroupSync();
    return m;
}

uint4 row_range(uint tid, uint row)
{
    return group_uniform4(islands[group_uniform(row_island(row), tid)], tid);
}

// ------------------------------------------------------------------ set-up

// b, the residual's start, the input kept, the blocks; partials: 0 (target, singular, differs,
// 0), 1 (sum w p, sum w) for the rigid modes' centre.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_init_nodes(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end)
        return;
    float4 p0 = float4(0.0);
    float4 p1 = float4(0.0);
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        float4 a_ang = accel[i * 2];
        float4 a_lin = accel[i * 2 + 1];
        if (any(a_ang != scratch[ns(i, N_LAST)]) || any(a_lin != scratch[ns(i, N_LAST) + 1]))
            p0.z = 1.0;
        float3 ba = -a_ang.xyz * (n.w_ang > 0.0 ? 1.0 / n.w_ang : 1.0);
        float3 bl = -a_lin.xyz * (n.w_lin > 0.0 ? params.recip_length / n.w_lin : params.recip_length);
        p0.x = norm2(ba, bl);
        set_v(i, N_B, ba, bl);
        set_v(i, N_R, ba, bl);
        scratch[ns(i, N_LAST)] = a_ang;
        scratch[ns(i, N_LAST) + 1] = a_lin;
        set_v(i, N_U, float3(0.0), float3(0.0));
        set_v(i, N_P, float3(0.0), float3(0.0));
        if (!build_block(i))
            p0.y = 1.0;
        float w = 1.0 / (n.w_lin * n.w_lin);
        p1 = float4(n.position.xyz * w, w);
    }
    write_partial(tid, b.row, group.x, 0, p0);
    write_partial(tid, b.row, group.x, 1, p1);
}

// Partial 2: (whether a live bond reaches a support, 0, 0, 0); a cold start's x cleared.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_init_bonds(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = bond_block(group, tid);
    if (b.begin >= b.end)
        return;
    float anchor = 0.0;
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint j = lists[p];
        BlastBond bd = bonds[j];
        if (bd.health > 0.0 && (!(nodes[bd.n0].w_lin > 0.0) || !(nodes[bd.n1].w_lin > 0.0)))
            anchor = 1.0;
        if (params.warm == 0)
        {
            x[j * 2] = float4(0.0);
            x[j * 2 + 1] = float4(0.0);
        }
    }
    write_partial(tid, b.row, group.x, 2, float4(anchor, 0.0, 0.0, 0.0));
}

// The row's start: the certificate's skip, the target, a zero load, whether free, the centre.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_setup(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint row = group.x;
    uint island = group_uniform(row_island(row), tid);
    uint4 r = group_uniform4(islands[island], tid);
    uint nb = blocks(r.y - r.x);
    uint bb = blocks(r.w - r.z);
    float4 s0 = row_sum(tid, row, nb, 0);
    float4 s1 = row_sum(tid, row, nb, 1);
    float4 s2 = row_sum(tid, row, bb, 2);
    if (tid != 0)
        return;
    uint4 last = report[island];
    uint f = F_ACTIVE;
    // The settled certificate (beginNativeSettledReuse).
    if (params.warm != 0 && (last.y & REPORT_SETTLED) != 0 && s0.z == 0.0)
        f = F_SKIPPED;
    float delta2 = s0.x * params.tolerance * params.tolerance;
    if (f == F_ACTIVE)
    {
        if (s0.y > 0.0)
            f = F_FAILED;
        else if (!(delta2 > 0.0))
            f = F_ZERO | F_CONVERGED;
    }
    if (s2.x == 0.0)
        f |= F_FREE;
    wide[st(row, S_SCALARS)] = float4(delta2, 0.0, 0.0, float_bits(0x7f800000u));
    wide[st(row, S_STEP)] = float4(0.0, 0.0, 0.0, float_bits(0x7f800000u));
    wide[st(row, S_COUNT)] = float4(0.0, 0.0, asfloat(0u), asfloat(f));
    wide[st(row, S_CENTRE)] = float4(s1.y > 0.0 ? s1.xyz / s1.w : float3(0.0), asfloat(0u));
}

float3 wide_rel(uint row, uint n)
{
    return nodes[n].position.xyz - wide[st(row, S_CENTRE)].xyz;
}

// The Gram matrix's 22 sums (blast_native.slang `build_modes`), partials 0..5.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_gram_nodes(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end || (b.flags & F_FREE) == 0 || (b.flags & F_ACTIVE) == 0)
        return;
    float rr[6];
    float3 rt[3];
    for (uint z = 0; z < 6; z++)
        rr[z] = 0.0;
    for (uint z = 0; z < 3; z++)
        rt[z] = float3(0.0);
    float tt = 0.0;
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        float wa = 1.0 / (n.w_ang * n.w_ang);
        float wl = 1.0 / (n.w_lin * n.w_lin);
        float3 q = wide_rel(b.row, i);
        float3 c[3] = { cross(q, float3(1, 0, 0)), cross(q, float3(0, 1, 0)), cross(q, float3(0, 0, 1)) };
        rr[0] = wa + wl * dot(c[0], c[0]);
        rr[1] = wa + wl * dot(c[1], c[1]);
        rr[2] = wa + wl * dot(c[2], c[2]);
        rr[3] = wl * dot(c[0], c[1]);
        rr[4] = wl * dot(c[0], c[2]);
        rr[5] = wl * dot(c[1], c[2]);
        for (uint kk = 0; kk < 3; kk++)
            rt[kk] = wl * c[kk];
        tt = wl;
    }
    write_partial(tid, b.row, group.x, 0, float4(rr[0], rr[1], rr[2], rr[3]));
    write_partial(tid, b.row, group.x, 1, float4(rr[4], rr[5], rt[0].x, rt[0].y));
    write_partial(tid, b.row, group.x, 2, float4(rt[0].z, rt[1].x, rt[1].y, rt[1].z));
    write_partial(tid, b.row, group.x, 3, float4(rt[2], tt));
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void w_gram(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint row = group.x;
    uint f = group_uniform(flags_of(row), tid);
    if ((f & F_FREE) == 0 || (f & F_ACTIVE) == 0)
        return;
    uint4 r = row_range(tid, row);
    uint nb = blocks(r.y - r.x);
    float4 a = row_sum(tid, row, nb, 0);
    float4 b = row_sum(tid, row, nb, 1);
    float4 c = row_sum(tid, row, nb, 2);
    float4 d = row_sum(tid, row, nb, 3);
    if (tid != 0)
        return;
    float m[36];
    for (uint z = 0; z < 36; z++)
        m[z] = 0.0;
    m[0] = a.x; m[7] = a.y; m[14] = a.z;
    m[1] = a.w; m[6] = a.w;
    m[2] = b.x; m[12] = b.x;
    m[8] = b.y; m[13] = b.y;
    float3 r0 = float3(b.z, b.w, c.x);
    float3 r1 = float3(c.y, c.z, c.w);
    float3 r2 = d.xyz;
    for (uint j = 0; j < 3; j++)
    {
        m[0 * 6 + 3 + j] = r0[j];
        m[1 * 6 + 3 + j] = r1[j];
        m[2 * 6 + 3 + j] = r2[j];
        m[(3 + j) * 6 + 0] = r0[j];
        m[(3 + j) * 6 + 1] = r1[j];
        m[(3 + j) * 6 + 2] = r2[j];
        m[(3 + j) * 6 + 3 + j] = d.w;
    }
    for (uint z = 0; z < 9; z++)
        wide[st(row, S_GRAM + z)] = float4(m[4 * z], m[4 * z + 1], m[4 * z + 2], m[4 * z + 3]);
}

// A warm start's residual b - D C S x0.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_warm(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end || params.warm == 0 || (b.flags & F_ACTIVE) == 0)
        return;
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint i = lists[p];
        float3 a;
        float3 l;
        gather(i, false, false, a, l);
        set_v(i, N_R, va(i, N_B) - a, vl(i, N_B) - l);
    }
}

// ------------------------------------------------------------------ projection

bool runs(uint flags, bool only_rebuild)
{
    return (flags & F_ACTIVE) != 0 && (!only_rebuild || (flags & F_REBUILD) != 0);
}

// Partials 0, 1: the projection's right-hand side for node vector v.
void proj_part(uint3 group, uint tid, uint v, bool only_rebuild)
{
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, only_rebuild) || (b.flags & F_FREE) == 0)
        return;
    float3 sr = float3(0.0);
    float3 stt = float3(0.0);
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        float3 lin = vl(i, v) / n.w_lin;
        sr = va(i, v) / n.w_ang - cross(wide_rel(b.row, i), lin);
        stt = lin;
    }
    write_partial(tid, b.row, group.x, 0, float4(sr, stt.x));
    write_partial(tid, b.row, group.x, 1, float4(stt.yz, 0.0, 0.0));
}

void proj_solve(uint3 group, uint tid, bool only_rebuild)
{
    uint row = group.x;
    uint f = group_uniform(flags_of(row), tid);
    if (!runs(f, only_rebuild) || (f & F_FREE) == 0)
        return;
    uint4 r = row_range(tid, row);
    uint nb = blocks(r.y - r.x);
    float4 a = row_sum(tid, row, nb, 0);
    float4 b = row_sum(tid, row, nb, 1);
    if (tid != 0)
        return;
    float g[6][6];
    float rhs[6] = { a.x, a.y, a.z, a.w, b.x, b.y };
    for (uint z = 0; z < 36; z++)
    {
        float4 w = wide[st(row, S_GRAM + z / 4)];
        g[z / 6][z % 6] = w[z % 4];
    }
    // Gaussian elimination with partial pivoting (blast_native.slang `solve_gram`).
    float c[6];
    bool ok = true;
    for (uint col = 0; col < 6; col++)
    {
        uint piv = col;
        for (uint rr = col + 1; rr < 6; rr++)
            if (abs(g[rr][col]) > abs(g[piv][col]))
                piv = rr;
        for (uint k = 0; k < 6; k++)
        {
            float t = g[col][k];
            g[col][k] = g[piv][k];
            g[piv][k] = t;
        }
        float tb = rhs[col];
        rhs[col] = rhs[piv];
        rhs[piv] = tb;
        if (!(abs(g[col][col]) > 0.0))
        {
            ok = false;
            break;
        }
        for (uint rr = col + 1; rr < 6; rr++)
        {
            float fct = g[rr][col] / g[col][col];
            for (uint k = col; k < 6; k++)
                g[rr][k] -= fct * g[col][k];
            rhs[rr] -= fct * rhs[col];
        }
    }
    for (uint z = 0; z < 6; z++)
        c[z] = 0.0;
    if (ok)
    {
        for (int rr = 5; rr >= 0; rr--)
        {
            float s = rhs[rr];
            for (uint k = uint(rr) + 1; k < 6; k++)
                s -= g[rr][k] * c[k];
            c[rr] = s / g[rr][rr];
        }
    }
    wide[st(row, S_COEF)] = float4(c[0], c[1], c[2], 0.0);
    wide[st(row, S_COEF + 1)] = float4(c[3], c[4], c[5], 0.0);
}

void proj_apply(uint3 group, uint tid, uint v, bool only_rebuild)
{
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, only_rebuild) || (b.flags & F_FREE) == 0)
        return;
    uint p = b.begin + tid;
    if (p >= b.end)
        return;
    uint i = lists[p];
    BlastNode n = nodes[i];
    float3 w = wide[st(b.row, S_COEF)].xyz;
    float3 t = wide[st(b.row, S_COEF + 1)].xyz;
    set_v(i, v, va(i, v) - w / n.w_ang, vl(i, v) - (t + cross(wide_rel(b.row, i), w)) / n.w_lin);
}

[shader("compute")] [numthreads(1024, 1, 1)] void w_proj_part_r(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { proj_part(g, t.x, N_R, false); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_proj_solve(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { proj_solve(g, t.x, false); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_proj_apply_r(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { proj_apply(g, t.x, N_R, false); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_proj_part_rb(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { proj_part(g, t.x, N_R, true); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_proj_solve_rb(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { proj_solve(g, t.x, true); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_proj_apply_rb(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { proj_apply(g, t.x, N_R, true); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_proj_part_g(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { proj_part(g, t.x, N_G, false); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_proj_apply_g(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { proj_apply(g, t.x, N_G, false); }

// ------------------------------------------------------------------ the tests

// t = B^T R into the bond scratch: partial 0 (|t|^2, |x0 + B^T u|^2 for the force test).
void bond_r(uint3 group, uint tid, bool only_rebuild)
{
    Block b = bond_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, only_rebuild))
        return;
    float z = 0.0;
    float f = 0.0;
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint j = lists[p];
        BlastBond bd = bonds[j];
        float3 ta = float3(0.0);
        float3 tl = float3(0.0);
        if (bd.health > 0.0)
        {
            bond_term(bd, N_R, ta, tl);
            if (params.force_tolerance > 0.0)
            {
                float3 ua;
                float3 ul;
                bond_term(bd, N_U, ua, ul);
                f = norm2(x[j * 2].xyz + ua, x[j * 2 + 1].xyz + ul);
            }
        }
        z = norm2(ta, tl);
        scratch[bs(j)] = float4(ta, 0.0);
        scratch[bs(j) + 1] = float4(tl, 0.0);
    }
    write_partial(tid, b.row, group.x, 2, float4(z, f, 0.0, 0.0));
}

// Partial 3: |r|^2 (the relative residual).
void node_rn(uint3 group, uint tid, bool only_rebuild)
{
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, only_rebuild))
        return;
    float s = 0.0;
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint i = lists[p];
        s = norm2(va(i, N_R), vl(i, N_R));
    }
    write_partial(tid, b.row, group.x, 3, float4(s, 0.0, 0.0, 0.0));
}

[shader("compute")] [numthreads(1024, 1, 1)] void w_bond_r(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { bond_r(g, t.x, false); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_node_rn(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { node_rn(g, t.x, false); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_bond_rb(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { bond_r(g, t.x, true); }
[shader("compute")] [numthreads(1024, 1, 1)] void w_node_rnb(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) { node_rn(g, t.x, true); }

// The first test of an iteration: num and the hold; whether to rebuild the residual.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_test(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint row = group.x;
    uint f = group_uniform(flags_of(row), tid);
    if ((f & F_ACTIVE) == 0)
        return;
    uint4 r = row_range(tid, row);
    float4 z = row_sum(tid, row, blocks(r.w - r.z), 2);
    float4 rn = row_sum(tid, row, blocks(r.y - r.x), 3);
    if (tid != 0)
        return;
    float4 s0 = wide[st(row, S_SCALARS)];
    float delta2 = s0.x;
    bool hold = z.x <= delta2 && !(rn.x <= delta2);
    uint it = iteration_of(row);
    f &= ~(F_HOLD | F_REBUILD);
    if (hold)
        f |= F_HOLD;
    if ((it > 0 || params.warm != 0) && z.x <= delta2 && !hold)
        f |= F_REBUILD;
    s0.y = z.x;
    wide[st(row, S_SCALARS)] = s0;
    float4 c = wide[st(row, S_STEP)];
    wide[st(row, S_STEP)] = c;
    set_flags(row, f);
    // The exact force norm for the force test, from this pass.
    float4 cnt = wide[st(row, S_COUNT)];
    wide[st(row, S_CENTRE)] = float4(wide[st(row, S_CENTRE)].xyz, wide[st(row, S_CENTRE)].w);
    wide[pt(row, 0, 5)] = float4(z.y, 0.0, 0.0, 0.0);
    wide[st(row, S_COUNT)] = cnt;
}

// The true residual b - D C S (x0 + B^T u) (reliable residual), where the test asked for it.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_rebuild(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, true))
        return;
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint i = lists[p];
        float3 a;
        float3 l;
        gather(i, false, true, a, l);
        set_v(i, N_R, va(i, N_B) - a, vl(i, N_B) - l);
    }
}

// The iteration's decision: the rebuilt residual's tests, stagnation, convergence, the force
// test; a row that stops leaves every later kernel of the frame.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_decide(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint row = group.x;
    uint f = group_uniform(flags_of(row), tid);
    if ((f & F_ACTIVE) == 0)
        return;
    uint4 r = row_range(tid, row);
    float4 z = float4(0.0);
    float4 rn = float4(0.0);
    if ((f & F_REBUILD) != 0)
    {
        z = row_sum(tid, row, blocks(r.w - r.z), 2);
        rn = row_sum(tid, row, blocks(r.y - r.x), 3);
    }
    if (tid != 0)
        return;
    float4 s0 = wide[st(row, S_SCALARS)];
    float4 s1 = wide[st(row, S_STEP)];
    float4 cnt = wide[st(row, S_COUNT)];
    float4 cen = wide[st(row, S_CENTRE)];
    float delta2 = s0.x;
    float num = s0.y;
    bool hold = (f & F_HOLD) != 0;
    if ((f & F_REBUILD) != 0)
    {
        num = z.x;
        hold = num <= delta2 && !(rn.x <= delta2);
        s0.z = 0.0;
    }
    f &= ~(F_HOLD | F_REBUILD);
    uint it = asuint(cnt.z);
    uint best_it = asuint(cen.w);
    bool stop = false;
    if (num < s0.w * 0.99)
    {
        s0.w = num;
        best_it = it;
    }
    else if (it - best_it >= STAGNATION)
        stop = true;
    if (!stop && num <= delta2 && !hold)
    {
        f |= F_CONVERGED;
        stop = true;
    }
    if (!stop && params.force_tolerance > 0.0 && it > 0 && s1.w <= params.force_tolerance * (cnt.x + cnt.y))
    {
        // The exact force norm |x0 + B^T u| from the first pass of this iteration (u has not
        // changed since).
        if (s1.w <= params.force_tolerance * sqrt(wide[pt(row, 0, 5)].x))
        {
            f |= F_CONVERGED;
            stop = true;
        }
    }
    if (stop || it >= params.max_iterations)
        f &= ~F_ACTIVE;
    s0.y = num;
    wide[st(row, S_SCALARS)] = s0;
    wide[st(row, S_CENTRE)] = float4(cen.xyz, asfloat(best_it));
    set_flags(row, f);
}

// ------------------------------------------------------------------ the direction

// g = r on iteration 0; else M r into LOC.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_pre1(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, false))
        return;
    uint p = b.begin + tid;
    if (p >= b.end)
        return;
    uint i = lists[p];
    if (iteration_of(b.row) == 0)
    {
        set_v(i, N_G, va(i, N_R), vl(i, N_R));
        return;
    }
    float3 a;
    float3 l;
    inverse_apply(i, va(i, N_R), vl(i, N_R), a, l);
    set_v(i, N_LOC, a, l);
}

// t = B^T LOC (iterations after the first).
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_bond_loc(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = bond_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, false) || iteration_of(b.row) == 0)
        return;
    uint p = b.begin + tid;
    if (p >= b.end)
        return;
    uint j = lists[p];
    BlastBond bd = bonds[j];
    float3 ta = float3(0.0);
    float3 tl = float3(0.0);
    if (bd.health > 0.0)
        bond_term(bd, N_LOC, ta, tl);
    scratch[bs(j)] = float4(ta, 0.0);
    scratch[bs(j) + 1] = float4(tl, 0.0);
}

// G = (a+b-ab) LOC - ab M (L LOC - block LOC).
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_pre2(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, false) || iteration_of(b.row) == 0)
        return;
    uint p = b.begin + tid;
    if (p >= b.end)
        return;
    uint i = lists[p];
    float3 oa;
    float3 ol;
    gather(i, true, false, oa, ol);
    float3 sa;
    float3 sl;
    block_apply(i, va(i, N_LOC), vl(i, N_LOC), sa, sl);
    float3 ma;
    float3 ml;
    inverse_apply(i, oa - sa, ol - sl, ma, ml);
    float diagonal = LOW_WEIGHT + HIGH_WEIGHT - LOW_WEIGHT * HIGH_WEIGHT;
    float coupling = LOW_WEIGHT * HIGH_WEIGHT;
    set_v(i, N_G, diagonal * va(i, N_LOC) - coupling * ma, diagonal * vl(i, N_LOC) - coupling * ml);
}

// Partial 4: the largest component of G.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_max(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, false))
        return;
    float m = 0.0;
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint i = lists[p];
        float3 a = abs(va(i, N_G));
        float3 l = abs(vl(i, N_G));
        m = max(max(max(a.x, a.y), max(a.z, l.x)), max(l.y, l.z));
    }
    float mm = wide_group_max(tid, m);
    if (tid == 0)
        wide[pt(b.row, group.x, 4)] = float4(mm, 0.0, 0.0, 0.0);
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void w_mag(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint row = group.x;
    uint f = group_uniform(flags_of(row), tid);
    if ((f & F_ACTIVE) == 0)
        return;
    uint4 r = row_range(tid, row);
    uint nb = blocks(r.y - r.x);
    float m = 0.0;
    for (uint blk = tid; blk < nb; blk += BLAST_THREADS)
        m = max(m, wide[pt(row, blk, 4)].x);
    float mag = wide_group_max(tid, m);
    if (tid != 0)
        return;
    if (!(mag > 0.0) || !isfinite(mag))
    {
        set_flags(row, (f & ~F_ACTIVE) | F_FAILED);
        return;
    }
    float4 s1 = wide[st(row, S_STEP)];
    s1.y = 1.0 / mag;
    wide[st(row, S_STEP)] = s1;
}

// G scaled by 1 / its largest component; partial 0: r.g.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_scale(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, false))
        return;
    float gl = 0.0;
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint i = lists[p];
        float inv = wide[st(b.row, S_STEP)].y;
        float3 ga = va(i, N_G) * inv;
        float3 gn = vl(i, N_G) * inv;
        set_v(i, N_G, ga, gn);
        gl = dot(va(i, N_R), ga) + dot(vl(i, N_R), gn);
    }
    write_partial(tid, b.row, group.x, 0, float4(gl, 0.0, 0.0, 0.0));
}

// gamma, and beta = gamma / gamma_prev from iteration 2.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_gamma(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint row = group.x;
    uint f = group_uniform(flags_of(row), tid);
    if ((f & F_ACTIVE) == 0)
        return;
    uint4 r = row_range(tid, row);
    float gamma = row_sum(tid, row, blocks(r.y - r.x), 0).x;
    if (tid != 0)
        return;
    if (!(gamma > 0.0) || !isfinite(gamma))
    {
        set_flags(row, (f & ~F_ACTIVE) | F_FAILED);
        return;
    }
    float4 s0 = wide[st(row, S_SCALARS)];
    uint it = iteration_of(row);
    float beta = it > 1 && s0.z > 0.0 ? gamma / s0.z : 0.0;
    float4 s1 = wide[st(row, S_STEP)];
    s1.x = gamma;
    s1.z = beta;
    wide[st(row, S_STEP)] = s1;
}

// p = g + beta p.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_dir(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, false))
        return;
    uint p = b.begin + tid;
    if (p >= b.end)
        return;
    uint i = lists[p];
    float beta = wide[st(b.row, S_STEP)].z;
    if (beta == 0.0)
        set_v(i, N_P, va(i, N_G), vl(i, N_G));
    else
        set_v(i, N_P, va(i, N_G) + va(i, N_P) * beta, vl(i, N_G) + vl(i, N_P) * beta);
}

// t = B^T p; partial 0: den = |t|^2.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_bond_p(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = bond_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, false))
        return;
    float z = 0.0;
    uint p = b.begin + tid;
    if (p < b.end)
    {
        uint j = lists[p];
        BlastBond bd = bonds[j];
        float3 ta = float3(0.0);
        float3 tl = float3(0.0);
        if (bd.health > 0.0)
            bond_term(bd, N_P, ta, tl);
        z = norm2(ta, tl);
        scratch[bs(j)] = float4(ta, 0.0);
        scratch[bs(j) + 1] = float4(tl, 0.0);
    }
    write_partial(tid, b.row, group.x, 1, float4(z, 0.0, 0.0, 0.0));
}

// q = L p.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_node_q(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    if (b.begin >= b.end || !runs(b.flags, false))
        return;
    uint p = b.begin + tid;
    if (p >= b.end)
        return;
    uint i = lists[p];
    float3 a;
    float3 l;
    gather(i, true, false, a, l);
    set_v(i, N_Q, a, l);
}

// den, alpha, the force step; the iteration counted.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_den(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint row = group.x;
    uint f = group_uniform(flags_of(row), tid);
    if ((f & F_ACTIVE) == 0)
        return;
    uint4 r = row_range(tid, row);
    float den = row_sum(tid, row, blocks(r.w - r.z), 1).x;
    if (tid != 0)
        return;
    float4 s0 = wide[st(row, S_SCALARS)];
    float4 s1 = wide[st(row, S_STEP)];
    float4 cnt = wide[st(row, S_COUNT)];
    float gamma = s1.x;
    if (params.force_tolerance > 0.0)
    {
        s1.w = den > 0.0 && gamma > 0.0 ? gamma / sqrt(den) : float_bits(0x7f800000u);
        if (isfinite(s1.w))
            cnt.y += s1.w;
    }
    s0.z = gamma;
    uint it = asuint(cnt.z) + 1;
    cnt.z = asfloat(it);
    if (!(den > 0.0) || !isfinite(den))
    {
        f &= ~F_ACTIVE;
        s1.y = 0.0;
    }
    else
        s1.y = gamma / den;
    wide[st(row, S_SCALARS)] = s0;
    wide[st(row, S_STEP)] = s1;
    cnt.w = asfloat(f);
    wide[st(row, S_COUNT)] = cnt;
}

// u += alpha p, r -= alpha q.
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_update(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = node_block(group, tid);
    // A row that retired in w_den keeps its step out.
    if (b.begin >= b.end || !runs(b.flags, false))
        return;
    uint p = b.begin + tid;
    if (p >= b.end)
        return;
    uint i = lists[p];
    float alpha = wide[st(b.row, S_STEP)].y;
    set_v(i, N_U, va(i, N_U) + va(i, N_P) * alpha, vl(i, N_U) + vl(i, N_P) * alpha);
    set_v(i, N_R, va(i, N_R) - va(i, N_Q) * alpha, vl(i, N_R) - vl(i, N_Q) * alpha);
}

// ------------------------------------------------------------------ the end

// x = x0 + B^T u over the row's live bonds (zero under a zero load; kept when skipped).
[shader("compute")]
[numthreads(1024, 1, 1)]
void w_final(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    Block b = bond_block(group, tid);
    if (b.begin >= b.end || (b.flags & F_SKIPPED) != 0)
        return;
    uint p = b.begin + tid;
    if (p >= b.end)
        return;
    uint j = lists[p];
    BlastBond bd = bonds[j];
    if (!(bd.health > 0.0) || (b.flags & F_ZERO) != 0)
    {
        x[j * 2] = float4(0.0);
        x[j * 2 + 1] = float4(0.0);
        return;
    }
    float3 ta;
    float3 tl;
    bond_term(bd, N_U, ta, tl);
    x[j * 2] = float4(x[j * 2].xyz + ta, 0.0);
    x[j * 2 + 1] = float4(x[j * 2 + 1].xyz + tl, 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void w_report(uint3 group : SV_GroupID)
{
    uint row = group.x;
    uint island = row_island(row);
    uint f = flags_of(row);
    uint4 last = report[island];
    if ((f & F_SKIPPED) != 0)
    {
        report[island] = uint4(0, REPORT_SKIPPED | REPORT_SETTLED | REPORT_CONVERGED, last.z, last.w);
        return;
    }
    bool ok = (f & F_CONVERGED) != 0 && (f & F_FAILED) == 0;
    uint it = iteration_of(row);
    uint flags = (ok ? REPORT_CONVERGED : 0u) | (ok && params.warm != 0 && it == 0 ? REPORT_SETTLED : 0u);
    float4 s0 = wide[st(row, S_SCALARS)];
    report[island] = uint4(it, flags, asuint(s0.y), asuint(s0.x));
}
