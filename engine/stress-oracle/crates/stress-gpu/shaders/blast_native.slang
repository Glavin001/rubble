// The resident (native) Blast solve of PhysX's GPU destruction stage, one group per island
// (stressgpu/detail/StressComponentIteration.cuh `componentStressSolve`, without rotational or
// shear stiffness; vibe-land's `native` backend). Its f64/f32 reference, with the full recurrence
// in its docs, is stress-ref blast_bond/native.rs.
//
// Projected PCG in node space on L = B B^T (B = D C S, D the native mass weighting):
//   r = b - B x0 (warm) or b;  delta^2 = tol^2 |b|^2;  a zero load retires with zero forces;
//   each iteration: r projected off a free island's rigid modes; num = |B^T r|^2 with the
//   relative residual |r|^2 <= delta^2 also required; passing after an update or on a warm
//   start, the true residual rebuilt (reliable residual) and both tests taken again; 512
//   iterations without a 1% gain stop; converged; the force test (optional); g = r on iteration
//   0, else the two-step Chebyshev block-Jacobi polynomial, projected and scaled by its largest
//   component; gamma = r.g > 0; beta = gamma / gamma_prev from iteration 2; p = g + beta p;
//   q = L p, den = |B^T p|^2 > 0; u += alpha p, r -= alpha q;  then x = x0 + B^T u.
//
// Each node's 6x6 block of L is [[A, -[k]x], [[k]x, c I]] (StressNativeRigidInverse.cuh):
// A = dA^2 sum s^2 ((1 + |o|^2) I - o o^T), k = dA dL sum s^2 o, c = dL^2 sum s^2 over its live
// bonds (o its own offset); its inverse by the Schur complement S = A + [k]x^2 / c.
//
// The island's lists as blast_cgls.slang's (blast_topology.slang builds them); one group per
// island (`args` sizes the dispatch).

import common;
import blast_common;

[[vk::binding(0, 0)]] ConstantBuffer<BlastParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> islands;
[[vk::binding(2, 0)]] StructuredBuffer<BlastNode> nodes;
[[vk::binding(3, 0)]] StructuredBuffer<uint> refs;
[[vk::binding(4, 0)]] RWStructuredBuffer<BlastBond> bonds;
[[vk::binding(5, 0)]] StructuredBuffer<float4> accel;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> x;
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> scratch;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint4> report;
[[vk::binding(9, 0)]] StructuredBuffer<uint> lists;
// Per node: its position over the length scale (the rigid modes' frame).
[[vk::binding(10, 0)]] StructuredBuffer<float4> positions;

// Scratch per node (float4s): vectors, angular then linear.
static const uint NV = 24;
static const uint N_R = 0;     // residual
static const uint N_P = 2;     // direction
static const uint N_Q = 4;     // L p
static const uint N_U = 6;     // solution in node space
static const uint N_G = 8;     // preconditioned residual
static const uint N_LOC = 10;  // polynomial scratch: M r
static const uint N_OFF = 12;  // polynomial scratch: E M r
static const uint N_B = 14;    // right-hand side
static const uint N_LAST = 16; // the input, for the settled certificate
static const uint N_BLK = 18;  // the block: (A xx, yy, zz, xy), (A xz, yz, c, 0), (k, 0)
static const uint N_INV = 21;  // its inverse: (S^-1 xx, yy, zz, xy), (S^-1 xz, yz, 1/c, 0), (k / c, 0)
uint ns(uint n, uint v) { return n * NV + v; }
uint bs(uint j) { return params.node_count * NV + 2 * j; }

static const float LOW_WEIGHT = 0.5779388123770052;
static const float HIGH_WEIGHT = 2.6335678180143502;
static const uint STAGNATION = 512;

float3 va(uint n, uint v) { return scratch[ns(n, v)].xyz; }
float3 vl(uint n, uint v) { return scratch[ns(n, v) + 1].xyz; }
void set_v(uint n, uint v, float3 a, float3 l)
{
    scratch[ns(n, v)] = float4(a, 0.0);
    scratch[ns(n, v) + 1] = float4(l, 0.0);
}

// t_j = S C^T D v for node vector `v` (a support's weights are 0; its scratch stays 0).
void bond_term(BlastBond b, uint v, out float3 ta, out float3 tl)
{
    BlastNode m0 = nodes[b.n0];
    BlastNode m1 = nodes[b.n1];
    float3 a0 = va(b.n0, v) * m0.w_ang;
    float3 l0 = vl(b.n0, v) * m0.w_lin;
    float3 a1 = va(b.n1, v) * m1.w_ang;
    float3 l1 = vl(b.n1, v) * m1.w_lin;
    ta = (a0 - a1) * b.s;
    tl = ((l0 - l1) + (cross(b.o0.xyz, a0) - cross(b.o1.xyz, a1))) * b.s;
}

// (D C S y)_node over live bonds: y the bond scratch (from_t) or x0 + (from_u ? B^T u : 0).
void gather(uint node, bool from_t, bool with_u, out float3 ang, out float3 lin)
{
    BlastNode n = nodes[node];
    float3 acc_a = float3(0.0);
    float3 acc_l = float3(0.0);
    for (uint r = n.ref_begin; r < n.ref_end; r++)
    {
        uint ref = refs[r];
        uint j = ref & ~REF_SECOND;
        BlastBond b = bonds[j];
        if (!(b.health > 0.0))
            continue;
        float3 ya;
        float3 yl;
        if (from_t)
        {
            ya = scratch[bs(j)].xyz;
            yl = scratch[bs(j) + 1].xyz;
        }
        else
        {
            ya = x[j * 2].xyz;
            yl = x[j * 2 + 1].xyz;
            if (with_u)
            {
                float3 ta;
                float3 tl;
                bond_term(b, N_U, ta, tl);
                ya += ta;
                yl += tl;
            }
        }
        ya *= b.s;
        yl *= b.s;
        if ((ref & REF_SECOND) != 0)
        {
            acc_a += cross(b.o1.xyz, yl) - ya;
            acc_l -= yl;
        }
        else
        {
            acc_a += ya - cross(b.o0.xyz, yl);
            acc_l += yl;
        }
    }
    ang = acc_a * n.w_ang;
    lin = acc_l * n.w_lin;
}

float3 sym_mul(float4 s0, float4 s1, float3 v)
{
    // (xx, yy, zz, xy), (xz, yz, ...)
    return float3(s0.x * v.x + s0.w * v.y + s1.x * v.z, s0.w * v.x + s0.y * v.y + s1.y * v.z, s1.x * v.x + s1.y * v.y + s0.z * v.z);
}

// The node's block applied to (a, l): (A a - k x l, k x a + c l).
void block_apply(uint n, float3 a, float3 l, out float3 oa, out float3 ol)
{
    float4 b0 = scratch[ns(n, N_BLK)];
    float4 b1 = scratch[ns(n, N_BLK) + 1];
    float3 k = scratch[ns(n, N_BLK) + 2].xyz;
    oa = sym_mul(b0, b1, a) - cross(k, l);
    ol = cross(k, a) + b1.z * l;
}

// The block's inverse applied: x = S^-1 (a + k' x l), y = l / c - k' x x.
void inverse_apply(uint n, float3 a, float3 l, out float3 oa, out float3 ol)
{
    float4 s0 = scratch[ns(n, N_INV)];
    float4 s1 = scratch[ns(n, N_INV) + 1];
    float3 kc = scratch[ns(n, N_INV) + 2].xyz;
    oa = sym_mul(s0, s1, a + cross(kc, l));
    ol = l * s1.z - cross(kc, oa);
}

// Build node n's block and its inverse; false when the block is singular.
bool build_block(uint node)
{
    BlastNode n = nodes[node];
    float3 diag = float3(0.0);
    float3 offd = float3(0.0); // xy, xz, yz of sum s^2 ((1+|o|^2) I - o o^T)
    float3 ks = float3(0.0);
    float cs = 0.0;
    for (uint r = n.ref_begin; r < n.ref_end; r++)
    {
        uint ref = refs[r];
        BlastBond b = bonds[ref & ~REF_SECOND];
        if (!(b.health > 0.0))
            continue;
        float3 o = (ref & REF_SECOND) != 0 ? b.o1.xyz : b.o0.xyz;
        float s2 = b.s * b.s;
        float oo = 1.0 + dot(o, o);
        diag += s2 * (float3(oo) - o * o);
        offd += s2 * -float3(o.x * o.y, o.x * o.z, o.y * o.z);
        ks += s2 * o;
        cs += s2;
    }
    float da = n.w_ang * n.w_ang;
    float dad = n.w_ang * n.w_lin;
    float c = n.w_lin * n.w_lin * cs;
    float3 k = ks * dad;
    float4 a0 = float4(diag * da, offd.x * da);
    float4 a1 = float4(offd.y * da, offd.z * da, c, 0.0);
    scratch[ns(node, N_BLK)] = a0;
    scratch[ns(node, N_BLK) + 1] = a1;
    scratch[ns(node, N_BLK) + 2] = float4(k, 0.0);
    if (!(c > 0.0))
        return false;
    // S = A + [k]x^2 / c = A + (k k^T - |k|^2 I) / c.
    float ic = 1.0 / c;
    float kk = dot(k, k);
    float sxx = a0.x + (k.x * k.x - kk) * ic;
    float syy = a0.y + (k.y * k.y - kk) * ic;
    float szz = a0.z + (k.z * k.z - kk) * ic;
    float sxy = a0.w + k.x * k.y * ic;
    float sxz = a1.x + k.x * k.z * ic;
    float syz = a1.y + k.y * k.z * ic;
    // Symmetric 3x3 inverse by its adjugate.
    float cxx = syy * szz - syz * syz;
    float cxy = sxz * syz - sxy * szz;
    float cxz = sxy * syz - sxz * syy;
    float det = sxx * cxx + sxy * cxy + sxz * cxz;
    if (!(det > 0.0) || !isfinite(det))
        return false;
    float id = 1.0 / det;
    float cyy = sxx * szz - sxz * sxz;
    float cyz = sxy * sxz - sxx * syz;
    float czz = sxx * syy - sxy * sxy;
    scratch[ns(node, N_INV)] = float4(cxx * id, cyy * id, czz * id, cxy * id);
    scratch[ns(node, N_INV) + 1] = float4(cxz * id, cyz * id, ic, 0.0);
    scratch[ns(node, N_INV) + 2] = float4(k * ic, 0.0);
    return true;
}

// The rigid modes' Gram matrix and frame, per group (thread 0 solves with them).
groupshared float g_gram[6][6];
groupshared float g_rhs[6];
groupshared float g_coef[6];
groupshared float3 g_centre;

// A free island's rigid modes at node n: rot k = (e_k / dA, (q x e_k) / dL), trans j = (0, e_j / dL).
float3 rel(uint n)
{
    return positions[n].xyz - g_centre;
}

// Solve g_gram c = g_rhs (6x6) into g_coef, thread 0 (Gaussian elimination, partial pivoting).
void solve_gram()
{
    float a[6][6];
    float b[6];
    for (uint r = 0; r < 6; r++)
    {
        b[r] = g_rhs[r];
        for (uint c = 0; c < 6; c++)
            a[r][c] = g_gram[r][c];
    }
    for (uint c = 0; c < 6; c++)
    {
        uint p = c;
        for (uint r = c + 1; r < 6; r++)
            if (abs(a[r][c]) > abs(a[p][c]))
                p = r;
        for (uint k = 0; k < 6; k++)
        {
            float t = a[c][k];
            a[c][k] = a[p][k];
            a[p][k] = t;
        }
        float tb = b[c];
        b[c] = b[p];
        b[p] = tb;
        float piv = a[c][c];
        if (!(abs(piv) > 0.0))
        {
            for (uint z = 0; z < 6; z++)
                g_coef[z] = 0.0;
            return;
        }
        for (uint r = c + 1; r < 6; r++)
        {
            float f = a[r][c] / piv;
            for (uint k = c; k < 6; k++)
                a[r][k] -= f * a[c][k];
            b[r] -= f * b[c];
        }
    }
    for (int r = 5; r >= 0; r--)
    {
        float s = b[r];
        for (uint k = uint(r) + 1; k < 6; k++)
            s -= a[r][k] * g_coef[k];
        g_coef[r] = s / a[r][r];
    }
}

// Project node vector `v` of the island's nodes [n0, n1) off its rigid modes (`free` uniform).
void project(uint tid, uint n0, uint n1, uint v, bool free)
{
    if (!free)
        return;
    float3 sr = float3(0.0);
    float3 st = float3(0.0);
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        float3 lin = vl(i, v) / n.w_lin;
        sr += va(i, v) / n.w_ang - cross(rel(i), lin);
        st += lin;
    }
    float2 a = blast_group_sum(tid, sr.xy);
    float2 b = blast_group_sum(tid, float2(sr.z, st.x));
    float2 c = blast_group_sum(tid, st.yz);
    if (tid == 0)
    {
        g_rhs[0] = a.x;
        g_rhs[1] = a.y;
        g_rhs[2] = b.x;
        g_rhs[3] = b.y;
        g_rhs[4] = c.x;
        g_rhs[5] = c.y;
        solve_gram();
    }
    GroupMemoryBarrierWithGroupSync();
    float3 w = float3(g_coef[0], g_coef[1], g_coef[2]);
    float3 t = float3(g_coef[3], g_coef[4], g_coef[5]);
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        set_v(i, v, va(i, v) - w / n.w_ang, vl(i, v) - (t + cross(rel(i), w)) / n.w_lin);
    }
    device_group_barrier();
}

// The rigid modes' frame and Gram matrix of a free island.
void build_modes(uint tid, uint n0, uint n1)
{
    float4 cw = float4(0.0);
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        float w = 1.0 / (nodes[i].w_lin * nodes[i].w_lin);
        cw += float4(positions[i].xyz * w, w);
    }
    float2 c0 = blast_group_sum(tid, cw.xy);
    float2 c1 = blast_group_sum(tid, cw.zw);
    if (tid == 0)
        g_centre = float3(c0.x, c0.y, c1.x) / c1.y;
    GroupMemoryBarrierWithGroupSync();
    // rot-rot: sum e_k.e_l / dA^2 + (q x e_k).(q x e_l) / dL^2; rot-trans: sum (q x e_k)_j / dL^2;
    // trans-trans: sum 1 / dL^2 on the diagonal.
    float rr[6];   // xx, yy, zz, xy, xz, yz
    float3 rt[3];  // rot k against trans j
    for (uint z = 0; z < 6; z++)
        rr[z] = 0.0;
    for (uint z = 0; z < 3; z++)
        rt[z] = float3(0.0);
    float tt = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        float wa = 1.0 / (n.w_ang * n.w_ang);
        float wl = 1.0 / (n.w_lin * n.w_lin);
        float3 q = rel(i);
        float3 c[3] = { cross(q, float3(1, 0, 0)), cross(q, float3(0, 1, 0)), cross(q, float3(0, 0, 1)) };
        rr[0] += wa + wl * dot(c[0], c[0]);
        rr[1] += wa + wl * dot(c[1], c[1]);
        rr[2] += wa + wl * dot(c[2], c[2]);
        rr[3] += wl * dot(c[0], c[1]);
        rr[4] += wl * dot(c[0], c[2]);
        rr[5] += wl * dot(c[1], c[2]);
        for (uint kk = 0; kk < 3; kk++)
            rt[kk] += wl * c[kk];
        tt += wl;
    }
    float2 s0 = blast_group_sum(tid, float2(rr[0], rr[1]));
    float2 s1 = blast_group_sum(tid, float2(rr[2], rr[3]));
    float2 s2 = blast_group_sum(tid, float2(rr[4], rr[5]));
    float2 s3 = blast_group_sum(tid, rt[0].xy);
    float2 s4 = blast_group_sum(tid, float2(rt[0].z, rt[1].x));
    float2 s5 = blast_group_sum(tid, rt[1].yz);
    float2 s6 = blast_group_sum(tid, rt[2].xy);
    float2 s7 = blast_group_sum(tid, float2(rt[2].z, tt));
    if (tid == 0)
    {
        float m[6][6];
        m[0][0] = s0.x; m[1][1] = s0.y; m[2][2] = s1.x;
        m[0][1] = s1.y; m[0][2] = s2.x; m[1][2] = s2.y;
        m[1][0] = m[0][1]; m[2][0] = m[0][2]; m[2][1] = m[1][2];
        float3 r0 = float3(s3.x, s3.y, s4.x);
        float3 r1 = float3(s4.y, s5.x, s5.y);
        float3 r2 = float3(s6.x, s6.y, s7.x);
        for (uint j = 0; j < 3; j++)
        {
            m[0][3 + j] = r0[j];
            m[1][3 + j] = r1[j];
            m[2][3 + j] = r2[j];
            m[3 + j][0] = r0[j];
            m[3 + j][1] = r1[j];
            m[3 + j][2] = r2[j];
            for (uint l = 0; l < 3; l++)
                m[3 + j][3 + l] = j == l ? s7.y : 0.0;
        }
        for (uint r = 0; r < 6; r++)
            for (uint c2 = 0; c2 < 6; c2++)
                g_gram[r][c2] = m[r][c2];
    }
    GroupMemoryBarrierWithGroupSync();
}

// t into the bond scratch for node vector v over the island's bonds, and sum |t|^2.
float bond_pass(uint tid, uint b0, uint b1, uint v)
{
    float z = 0.0;
    for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
    {
        uint j = lists[q];
        BlastBond b = bonds[j];
        float3 ta = float3(0.0);
        float3 tl = float3(0.0);
        if (b.health > 0.0)
            bond_term(b, v, ta, tl);
        z += norm2(ta, tl);
        scratch[bs(j)] = float4(ta, 0.0);
        scratch[bs(j) + 1] = float4(tl, 0.0);
    }
    return blast_group_sum(tid, float2(z, 0.0)).x;
}

// out = L v over the island's nodes (bond_pass for v first, then a barrier), and |out|^2.
float node_pass(uint tid, uint n0, uint n1, uint out_v)
{
    float s = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        float3 a;
        float3 l;
        gather(i, true, false, a, l);
        set_v(i, out_v, a, l);
        s += norm2(a, l);
    }
    return s;
}

// The true residual b - D C S (x0 + (with_u ? B^T u : 0)) into R, and |r|^2.
float rebuild(uint tid, uint n0, uint n1, bool with_u)
{
    float s = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        float3 a;
        float3 l;
        gather(i, false, with_u, a, l);
        float3 ra = va(i, N_B) - a;
        float3 rl = vl(i, N_B) - l;
        set_v(i, N_R, ra, rl);
        s += norm2(ra, rl);
    }
    device_group_barrier();
    return blast_group_sum(tid, float2(s, 0.0)).x;
}

float residual_norm(uint tid, uint n0, uint n1)
{
    float s = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        s += norm2(va(i, N_R), vl(i, N_R));
    }
    return blast_group_sum(tid, float2(s, 0.0)).x;
}

groupshared float g_max[BLAST_THREADS];
groupshared float g_max_out;

float group_max(uint tid, float v)
{
    g_max[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = BLAST_THREADS / 2; k > 0; k >>= 1)
    {
        if (tid < k)
            g_max[tid] = max(g_max[tid], g_max[tid + k]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0)
        g_max_out = g_max[0];
    // As a uniform value on WGSL (it steers the group's barriers).
    float m;
    __target_switch
    {
    case wgsl:
        m = workgroupUniformLoad(g_max_out);
    default:
        GroupMemoryBarrierWithGroupSync();
        m = g_max_out;
    }
    GroupMemoryBarrierWithGroupSync();
    return m;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void blast_native(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    uint tid = thread.x;
    uint k = group.x;
    if (k >= params.island_count)
        return;
    uint4 range = islands[k];
    uint n0 = range.x;
    uint n1 = range.y;
    uint b0 = range.z;
    uint b1 = range.w;
    uint4 last = report[k];
    uint last_flags = group_uniform(last.y, tid);
    bool warm = params.warm != 0;

    // The settled certificate (beginNativeSettledReuse): the last solve verified its stored
    // output and the input repeats bit for bit.
    if (warm && (last_flags & REPORT_SETTLED) != 0)
    {
        float differ = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            if (any(accel[i * 2] != scratch[ns(i, N_LAST)]) || any(accel[i * 2 + 1] != scratch[ns(i, N_LAST) + 1]))
                differ = 1.0;
        }
        if (blast_group_sum(tid, float2(differ, 0.0)).x == 0.0)
        {
            if (tid == 0)
                report[k] = uint4(0, REPORT_SKIPPED | REPORT_SETTLED | REPORT_CONVERGED, last.z, last.w);
            return;
        }
    }

    // b = -D^-1 a; the blocks and their inverses; whether the island is anchored.
    float target = 0.0;
    float singular = 0.0;
    for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
    {
        uint i = lists[p];
        BlastNode n = nodes[i];
        float4 a_ang = accel[i * 2];
        float4 a_lin = accel[i * 2 + 1];
        float3 ba = -a_ang.xyz * (n.w_ang > 0.0 ? 1.0 / n.w_ang : 1.0);
        float3 bl = -a_lin.xyz * (n.w_lin > 0.0 ? params.recip_length / n.w_lin : params.recip_length);
        target += norm2(ba, bl);
        set_v(i, N_B, ba, bl);
        set_v(i, N_R, ba, bl);
        scratch[ns(i, N_LAST)] = a_ang;
        scratch[ns(i, N_LAST) + 1] = a_lin;
        set_v(i, N_U, float3(0.0), float3(0.0));
        set_v(i, N_P, float3(0.0), float3(0.0));
        if (!build_block(i))
            singular = 1.0;
    }
    float anchor = 0.0;
    for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
    {
        BlastBond b = bonds[lists[q]];
        if (b.health > 0.0 && (!(nodes[b.n0].w_lin > 0.0) || !(nodes[b.n1].w_lin > 0.0)))
            anchor = 1.0;
    }
    if (!warm)
    {
        for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
        {
            uint j = lists[q];
            x[j * 2] = float4(0.0);
            x[j * 2 + 1] = float4(0.0);
        }
    }
    float2 tsum = blast_group_sum(tid, float2(target, singular));
    float delta2 = tsum.x * params.tolerance * params.tolerance;
    bool failed = tsum.y > 0.0;
    bool free = blast_group_sum(tid, float2(anchor, 0.0)).x == 0.0;
    device_group_barrier();

    // A zero load: zero forces at once (retireHomogeneousTreeComponent).
    if (!(delta2 > 0.0))
    {
        for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
        {
            uint j = lists[q];
            x[j * 2] = float4(0.0);
            x[j * 2 + 1] = float4(0.0);
        }
        if (tid == 0)
            report[k] = uint4(0, REPORT_CONVERGED | (warm ? REPORT_SETTLED : 0u), asuint(0.0), asuint(delta2));
        return;
    }
    if (free)
        build_modes(tid, n0, n1);
    if (warm)
        rebuild(tid, n0, n1, false);

    float diagonal = LOW_WEIGHT + HIGH_WEIGHT - LOW_WEIGHT * HIGH_WEIGHT;
    float coupling = LOW_WEIGHT * HIGH_WEIGHT;
    float previous = 0.0;
    float best = float_bits(0x7f800000u);
    uint best_iteration = 0;
    bool converged = false;
    uint iteration = 0;
    float num = 0.0;
    float force_start = 0.0;
    float travel = 0.0;
    float step = float_bits(0x7f800000u);
    if (params.force_tolerance > 0.0)
    {
        float f = 0.0;
        for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
        {
            uint j = lists[q];
            f += norm2(x[j * 2].xyz, x[j * 2 + 1].xyz);
        }
        force_start = sqrt(blast_group_sum(tid, float2(f, 0.0)).x);
    }
    while (!failed && iteration < params.max_iterations)
    {
        project(tid, n0, n1, N_R, free);
        num = bond_pass(tid, b0, b1, N_R);
        bool hold = num <= delta2 && !(residual_norm(tid, n0, n1) <= delta2);
        if ((iteration > 0 || warm) && num <= delta2 && !hold)
        {
            rebuild(tid, n0, n1, true);
            previous = 0.0;
            project(tid, n0, n1, N_R, free);
            num = bond_pass(tid, b0, b1, N_R);
            hold = num <= delta2 && !(residual_norm(tid, n0, n1) <= delta2);
        }
        if (num < best * 0.99)
        {
            best = num;
            best_iteration = iteration;
        }
        else if (iteration - best_iteration >= STAGNATION)
            break;
        if (num <= delta2 && !hold)
        {
            converged = true;
            break;
        }
        if (params.force_tolerance > 0.0 && iteration > 0 && step <= params.force_tolerance * (force_start + travel))
        {
            float f = 0.0;
            for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
            {
                uint j = lists[q];
                BlastBond b = bonds[j];
                float3 ta;
                float3 tl;
                bond_term(b, N_U, ta, tl);
                f += norm2(x[j * 2].xyz + ta, x[j * 2 + 1].xyz + tl);
            }
            if (step <= params.force_tolerance * sqrt(blast_group_sum(tid, float2(f, 0.0)).x))
            {
                converged = true;
                break;
            }
        }
        device_group_barrier();

        // g: r on iteration 0, else P r = (a+b-ab) M r - ab M E M r, with E = L - the blocks.
        if (iteration == 0)
        {
            for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
            {
                uint i = lists[p];
                set_v(i, N_G, va(i, N_R), vl(i, N_R));
            }
        }
        else
        {
            for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
            {
                uint i = lists[p];
                float3 a;
                float3 l;
                inverse_apply(i, va(i, N_R), vl(i, N_R), a, l);
                set_v(i, N_LOC, a, l);
            }
            device_group_barrier();
            bond_pass(tid, b0, b1, N_LOC);
            device_group_barrier();
            node_pass(tid, n0, n1, N_OFF);
            for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
            {
                uint i = lists[p];
                float3 sa;
                float3 sl;
                block_apply(i, va(i, N_LOC), vl(i, N_LOC), sa, sl);
                float3 ma;
                float3 ml;
                inverse_apply(i, va(i, N_OFF) - sa, vl(i, N_OFF) - sl, ma, ml);
                set_v(i, N_G, diagonal * va(i, N_LOC) - coupling * ma, diagonal * vl(i, N_LOC) - coupling * ml);
            }
        }
        device_group_barrier();
        project(tid, n0, n1, N_G, free);
        float mloc = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            float3 a = abs(va(i, N_G));
            float3 l = abs(vl(i, N_G));
            mloc = max(mloc, max(max(max(a.x, a.y), max(a.z, l.x)), max(l.y, l.z)));
        }
        float mag = group_max(tid, mloc);
        if (!(mag > 0.0) || !isfinite(mag))
        {
            failed = true;
            break;
        }
        float inv = 1.0 / mag;
        float gl = 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            float3 ga = va(i, N_G) * inv;
            float3 gn = vl(i, N_G) * inv;
            set_v(i, N_G, ga, gn);
            gl += dot(va(i, N_R), ga) + dot(vl(i, N_R), gn);
        }
        float gamma = blast_group_sum(tid, float2(gl, 0.0)).x;
        if (!(gamma > 0.0) || !isfinite(gamma))
        {
            failed = true;
            break;
        }
        float beta = iteration > 1 && previous > 0.0 ? gamma / previous : 0.0;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            if (beta == 0.0)
                set_v(i, N_P, va(i, N_G), vl(i, N_G));
            else
                set_v(i, N_P, va(i, N_G) + va(i, N_P) * beta, vl(i, N_G) + vl(i, N_P) * beta);
        }
        device_group_barrier();
        float den = bond_pass(tid, b0, b1, N_P);
        device_group_barrier();
        node_pass(tid, n0, n1, N_Q);
        if (params.force_tolerance > 0.0)
        {
            step = den > 0.0 && gamma > 0.0 ? gamma / sqrt(den) : float_bits(0x7f800000u);
            if (isfinite(step))
                travel += step;
        }
        previous = gamma;
        iteration++;
        if (!(den > 0.0) || !isfinite(den))
            break;
        float alpha = gamma / den;
        for (uint p = n0 + tid; p < n1; p += BLAST_THREADS)
        {
            uint i = lists[p];
            set_v(i, N_U, va(i, N_U) + va(i, N_P) * alpha, vl(i, N_U) + vl(i, N_P) * alpha);
            set_v(i, N_R, va(i, N_R) - va(i, N_Q) * alpha, vl(i, N_R) - vl(i, N_Q) * alpha);
        }
        device_group_barrier();
    }
    device_group_barrier();

    // x = x0 + B^T u over the island's live bonds.
    for (uint q = b0 + tid; q < b1; q += BLAST_THREADS)
    {
        uint j = lists[q];
        BlastBond b = bonds[j];
        if (!(b.health > 0.0))
        {
            x[j * 2] = float4(0.0);
            x[j * 2 + 1] = float4(0.0);
            continue;
        }
        float3 ta;
        float3 tl;
        bond_term(b, N_U, ta, tl);
        x[j * 2] = float4(x[j * 2].xyz + ta, 0.0);
        x[j * 2 + 1] = float4(x[j * 2 + 1].xyz + tl, 0.0);
    }
    bool ok = converged && !failed;
    // The certificate (commitNativeSettledReuse): a warm solve that converged with no update.
    uint flags = (ok ? REPORT_CONVERGED : 0u) | (ok && warm && iteration == 0 ? REPORT_SETTLED : 0u);
    if (tid == 0)
        report[k] = uint4(iteration, flags, asuint(num), asuint(delta2));
}
