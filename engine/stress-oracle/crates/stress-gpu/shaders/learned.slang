// The learned impact-outcome model's engine side (`STRESS_IMPACT_PUNCH=learned`, off by
// default; docs/learned-stress-solver/learned-stress-solver-plan.md, P3). The network itself
// is P0's kernel (`impact_learned.slang`) or a stub (`learned_stub`); these kernels feed it
// and apply what it predicts, in the resident frame's one submission:
//
//   learned_decode  one group per call: the deterministic decode of P2's
//                   `learned_impact.outcome.decode` (network outputs, body frame and
//                   nondimensional, to world frame-end velocities and displacements; the
//                   exact momentum projection of a free call; the kinetic-energy bound;
//                   the failure-margin break rule; survivors' damage kept below 1)
//
// Layouts (`learned.rs` mirrors them):
//   calls  CALL_F4 float4 per call:
//          0 scales (L, m, v, c)        1 scales (f_t, E, GA, dt)
//          2 (E_in, flags bits, first node bits, nodes bits)
//          3 (gravity xyz, chunk nodes bits)
//          4 (first bond bits, bonds bits, decode flags bits, fallback reason bits)
//   nodes  NODE_F4 float4 per node (a call's chunks, then its impactors, contiguous):
//          0 position relative to the call's centre of mass (world), mass
//          1 world orientation (x, y, z, w; body -> world)
//          2 principal inertia (body), anchored (0/1)
//          3 velocity (world), call bits        4 angular velocity (world), kind bits
//          5..7 the network's 12 body-frame outputs (dv, dw, dx, dtheta; nondimensional)
//          8 v_end   9 w_end   10 dx   11 drot  (decoded, world)
//   bonds  BOND_F4 float4 per live bond of a call (undirected):
//          0 (damage at frame start, log q, Δdamage, time bin bits 1..8)
//          1 (decoded damage, broke bits, engine bond bits, -)
// Frames: every vector here is the stress solver's (Z-up).

import common;

static const uint CALL_F4 = 8;
static const uint NODE_F4 = 12;
static const uint BOND_F4 = 2;
static const uint G = 256;

// calls[2].y flags
static const uint CALL_FREE = 1;      // momentum conserved (causal: no anchored chunk, no load, no gravity)
static const uint CALL_EXTERNAL = 2;  // a pressure load does work the inputs cannot bound: no KE bound
// calls[4].z decode flags
static const uint DEC_BOUND_ACTIVE = 1;
static const uint DEC_BOUND_INFEASIBLE = 2;
static const uint DEC_NONFINITE = 4;

struct DecodeParams
{
    uint call_count;
    uint pad0;
    float log_tau;    // ln τ on the failure margin
    float tolerance;  // KE bound tolerance, a fraction of E_in
};

[[vk::binding(0, 0)]] ConstantBuffer<DecodeParams> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> calls;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> nodes;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> bonds;

// ------------------------------------------------------------------ math

float3x3 quat_matrix(float4 q)
{
    float w = q.w, x = q.x, y = q.y, z = q.z;
    return float3x3(
        float3(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        float3(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        float3(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)));
}

float3x3 skew(float3 r)
{
    return float3x3(float3(0, -r.z, r.y), float3(r.z, 0, -r.x), float3(-r.y, r.x, 0));
}

// Rodrigues, with the first-order series below 1e-6 rad (geometry.py `rotvec_to_matrix`).
float3x3 rotvec_matrix(float3 r)
{
    float th = length(r);
    float3x3 eye = float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
    if (th < 1e-6)
        return eye + skew(r);
    float3x3 k = skew(r / th);
    return eye + sin(th) * k + (1.0 - cos(th)) * mul(k, k);
}

float3x3 diag3(float3 d)
{
    return float3x3(float3(d.x, 0, 0), float3(0, d.y, 0), float3(0, 0, d.z));
}

float3x3 outer3(float3 a, float3 b)
{
    return float3x3(a * b.x, a * b.y, a * b.z);
}

float3x3 world_inertia(float4 q, float3 principal)
{
    float3x3 r = quat_matrix(q);
    return mul(mul(r, diag3(principal)), transpose(r));
}

float3 solve3(float3x3 a, float3 b)
{
    float3 c0 = float3(a[0][0], a[1][0], a[2][0]);
    float3 c1 = float3(a[0][1], a[1][1], a[2][1]);
    float3 c2 = float3(a[0][2], a[1][2], a[2][2]);
    float det = dot(c0, cross(c1, c2));
    // Cramer's rule
    return float3(dot(b, cross(c1, c2)), dot(c0, cross(b, c2)), dot(c0, cross(c1, b))) / det;
}

// ------------------------------------------------------------------ group sums (fixed order)

groupshared float4 g_red[4][G];

void reduce_store(uint t, float4 a, float4 b, float4 c, float4 d)
{
    g_red[0][t] = a;
    g_red[1][t] = b;
    g_red[2][t] = c;
    g_red[3][t] = d;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            g_red[0][t] += g_red[0][t + s];
            g_red[1][t] += g_red[1][t + s];
            g_red[2][t] += g_red[2][t + s];
            g_red[3][t] += g_red[3][t + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// After `reduce_store`, every thread reads the totals, then a barrier before the next sum.
void reduce_read(out float4 a, out float4 b, out float4 c, out float4 d)
{
    a = g_red[0][0];
    b = g_red[1][0];
    c = g_red[2][0];
    d = g_red[3][0];
    GroupMemoryBarrierWithGroupSync();
}

float4 node(uint n, uint k)
{
    return nodes[NODE_F4 * n + k];
}

groupshared float g_lambda[6];

// ------------------------------------------------------------------ decode

[shader("compute")]
[numthreads(256, 1, 1)]
void learned_decode(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (g >= params.call_count)
        return;
    float4 s0 = calls[CALL_F4 * g];
    float4 s1 = calls[CALL_F4 * g + 1];
    float4 c2 = calls[CALL_F4 * g + 2];
    float4 c3 = calls[CALL_F4 * g + 3];
    float4 c4 = calls[CALL_F4 * g + 4];
    float L = s0.x;
    float vref = s0.z;
    float dt = s1.w;
    float e_in = c2.x;
    uint4 u = group_uniform4(uint4(asuint(c2.y), asuint(c2.z), asuint(c2.w), asuint(c4.y)), t);
    uint flags = u.x;
    uint first = u.y;
    uint n = u.z;
    uint nbonds = u.w;
    uint bfirst = asuint(c4.x);
    float3 grav = c3.xyz;
    bool free = (flags & CALL_FREE) != 0;
    // 1. to physical (decode.py `to_physical`)
    for (uint j = t; j < n; j += G)
    {
        uint k = first + j;
        float4 q = node(k, 1);
        float3 v = node(k, 3).xyz;
        float3 w = node(k, 4).xyz;
        float4 o0 = node(k, 5), o1 = node(k, 6), o2 = node(k, 7);
        Quat qq = quat_of(q);
        float3 v_end = v + rotate(qq, o0.xyz) * vref;
        float3 w_end = w + rotate(qq, float3(o0.w, o1.x, o1.y)) * vref / L;
        float3 dx = v * dt + 0.5 * grav * dt * dt + rotate(qq, float3(o1.z, o1.w, o2.x)) * vref * dt;
        float3 dr = w * dt + rotate(qq, o2.yzw) * vref * dt / L;
        if (node(k, 2).w != 0.0)
        {
            v_end = float3(0.0);
            w_end = float3(0.0);
            dx = float3(0.0);
            dr = float3(0.0);
        }
        nodes[NODE_F4 * k + 8] = float4(v_end, 0.0);
        nodes[NODE_F4 * k + 9] = float4(w_end, 0.0);
        nodes[NODE_F4 * k + 10] = float4(dx, 0.0);
        nodes[NODE_F4 * k + 11] = float4(dr, 0.0);
    }
    DeviceMemoryBarrierWithGroupSync();
    float4 a, b, c, d;
    // Sums every later step needs: mass, Σ m x, Σ m v (frame start), Σ m dx.
    {
        float4 sa = float4(0.0), sb = float4(0.0), sc = float4(0.0), sd = float4(0.0);
        for (uint j = t; j < n; j += G)
        {
            uint k = first + j;
            float4 p = node(k, 0);
            float m = p.w;
            sa += float4(p.xyz * m, m);
            sb += float4(node(k, 3).xyz * m, 0.0);
            sc += float4(node(k, 10).xyz * m, 0.0);
        }
        reduce_store(t, sa, sb, sc, sd);
        reduce_read(a, b, c, d);
    }
    float mt = a.w;
    float3 com0 = a.xyz / mt;
    float3 p0 = b.xyz;
    float3 have = c.xyz;
    // 2. exact momentum projection of a free call (decode.py `project`)
    float3 dx_shift = float3(0.0);
    if (free)
    {
        float3 want = p0 * dt + 0.5 * mt * grav * dt * dt;
        dx_shift = (want - have) / mt;
    }
    // frame-end centre of mass (with the shifted displacements), and l0 about com0
    {
        float4 sa = float4(0.0), sb = float4(0.0), sc = float4(0.0), sd = float4(0.0);
        for (uint j = t; j < n; j += G)
        {
            uint k = first + j;
            float4 p = node(k, 0);
            float m = p.w;
            float3 dx = node(k, 10).xyz;
            if (free)
                dx += dx_shift;
            sa += float4((p.xyz + dx) * m, 0.0);
            float3 r0 = p.xyz - com0;
            float3x3 i0 = world_inertia(node(k, 1), node(k, 2).xyz);
            sb += float4(cross(r0, m * node(k, 3).xyz) + mul(i0, node(k, 4).xyz), 0.0);
        }
        reduce_store(t, sa, sb, sc, sd);
        reduce_read(a, b, c, d);
    }
    float3 com1 = a.xyz / mt;
    float3 l0 = b.xyz;
    if (free)
    {
        for (uint j = t; j < n; j += G)
        {
            uint k = first + j;
            nodes[NODE_F4 * k + 10] = float4(node(k, 10).xyz + dx_shift, 0.0);
        }
        // ax, S = Σ m [r]×, J = Σ m(|r|²E - r rᵀ) + I1 (rows), at the frame-end poses
        float4 sa = float4(0.0), sb = float4(0.0), sc = float4(0.0), sd = float4(0.0);
        float4 se = float4(0.0), sf = float4(0.0), sg = float4(0.0);
        for (uint j = t; j < n; j += G)
        {
            uint k = first + j;
            float4 p = node(k, 0);
            float m = p.w;
            // (node 10 was shifted above by this same thread, for this same node)
            float3 r = p.xyz + node(k, 10).xyz - com1;
            float3x3 rd = rotvec_matrix(node(k, 11).xyz);
            float3x3 i1 = mul(mul(rd, world_inertia(node(k, 1), node(k, 2).xyz)), transpose(rd));
            float3 v = node(k, 8).xyz;
            float3 w = node(k, 9).xyz;
            sa += float4(m * v, 0.0);
            sb += float4(cross(r, m * v) + mul(i1, w), 0.0);
            sc += float4(m * r, 0.0); // Σ m r: S = [Σ m r]×
            float3x3 jj = m * (dot(r, r) * float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1)) - outer3(r, r)) + i1;
            sd += float4(jj[0], 0.0);
            se += float4(jj[1], 0.0);
            sf += float4(jj[2], 0.0);
        }
        reduce_store(t, sa, sb, sc, sd);
        float4 ax_p, ax_l, smr, j0;
        reduce_read(ax_p, ax_l, smr, j0);
        reduce_store(t, se, sf, sg, float4(0.0));
        float4 j1, j2, unused0, unused1;
        reduce_read(j1, j2, unused0, unused1);
        if (t == 0)
        {
            float3 rp = p0 + mt * grav * dt - ax_p.xyz;
            float3 rl = l0 - ax_l.xyz;
            float3x3 S = skew(smr.xyz);
            // [[mt I, -S], [S, J]] λ = resid; Gaussian elimination, partial pivoting
            float m6[6][7];
            for (uint r = 0; r < 6; r++)
                for (uint q = 0; q < 7; q++)
                    m6[r][q] = 0.0;
            float3 jr[3] = { j0.xyz, j1.xyz, j2.xyz };
            for (uint r = 0; r < 3; r++)
            {
                m6[r][r] = mt;
                for (uint q = 0; q < 3; q++)
                {
                    m6[r][3 + q] = -S[r][q];
                    m6[3 + r][q] = S[r][q];
                    m6[3 + r][3 + q] = jr[r][q];
                }
                m6[r][6] = rp[r];
                m6[3 + r][6] = rl[r];
            }
            for (uint col = 0; col < 6; col++)
            {
                uint piv = col;
                for (uint r = col + 1; r < 6; r++)
                    if (abs(m6[r][col]) > abs(m6[piv][col]))
                        piv = r;
                for (uint q = 0; q < 7; q++)
                {
                    float tmp = m6[col][q];
                    m6[col][q] = m6[piv][q];
                    m6[piv][q] = tmp;
                }
                for (uint r = col + 1; r < 6; r++)
                {
                    float f = m6[r][col] / m6[col][col];
                    for (uint q = col; q < 7; q++)
                        m6[r][q] -= f * m6[col][q];
                }
            }
            float x6[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            for (int r = 5; r >= 0; r--)
            {
                float acc = m6[r][6];
                for (uint q = uint(r) + 1; q < 6; q++)
                    acc -= m6[r][q] * x6[q];
                x6[r] = acc / m6[r][r];
            }
            for (uint r = 0; r < 6; r++)
                g_lambda[r] = x6[r];
        }
        GroupMemoryBarrierWithGroupSync();
        float3 lp = float3(g_lambda[0], g_lambda[1], g_lambda[2]);
        float3 ll = float3(g_lambda[3], g_lambda[4], g_lambda[5]);
        for (uint j = t; j < n; j += G)
        {
            uint k = first + j;
            float3 r = node(k, 0).xyz + node(k, 10).xyz - com1;
            nodes[NODE_F4 * k + 8] = float4(node(k, 8).xyz + lp + cross(ll, r), 0.0);
            nodes[NODE_F4 * k + 9] = float4(node(k, 9).xyz + ll, 0.0);
        }
        DeviceMemoryBarrierWithGroupSync();
    }
    // 3. kinetic-energy bound (decode.py `energy_bound`): KE_out <= KE_in + W_g + tol E_in
    uint dec = 0;
    if ((flags & CALL_EXTERNAL) == 0)
    {
        float4 sa = float4(0.0), sb = float4(0.0), sc = float4(0.0), sd = float4(0.0);
        for (uint j = t; j < n; j += G)
        {
            uint k = first + j;
            float m = node(k, 0).w;
            float3x3 i0 = world_inertia(node(k, 1), node(k, 2).xyz);
            float3x3 rd = rotvec_matrix(node(k, 11).xyz);
            float3x3 i1 = mul(mul(rd, i0), transpose(rd));
            float3 v0 = node(k, 3).xyz, w0 = node(k, 4).xyz;
            float3 v = node(k, 8).xyz, w = node(k, 9).xyz;
            float ke_in = 0.5 * (m * dot(v0, v0) + dot(w0, mul(i0, w0)));
            float wg = m * dot(grav, node(k, 10).xyz);
            float ke = 0.5 * (m * dot(v, v) + dot(w, mul(i1, w)));
            sa += float4(ke_in, wg, ke, 0.0);
            sb += float4(m * v, 0.0);
        }
        reduce_store(t, sa, sb, sc, sd);
        reduce_read(a, b, c, d);
        float bound = a.x + a.y + params.tolerance * e_in;
        float ke = a.z;
        bool active = group_uniform(ke > bound ? 1u : 0u, t) != 0;
        if (active)
        {
            dec |= DEC_BOUND_ACTIVE;
            if (free)
            {
                float3 vc = b.xyz / mt;
                // I_tot (rows) and L_rel about com1
                float4 ta = float4(0.0), tb = float4(0.0), tc = float4(0.0), td = float4(0.0);
                for (uint j = t; j < n; j += G)
                {
                    uint k = first + j;
                    float m = node(k, 0).w;
                    float3 r = node(k, 0).xyz + node(k, 10).xyz - com1;
                    float3x3 rd = rotvec_matrix(node(k, 11).xyz);
                    float3x3 i1 = mul(mul(rd, world_inertia(node(k, 1), node(k, 2).xyz)), transpose(rd));
                    float3x3 jj = m * (dot(r, r) * float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1)) - outer3(r, r)) + i1;
                    float3 vr = node(k, 8).xyz - vc;
                    ta += float4(jj[0], 0.0);
                    tb += float4(jj[1], 0.0);
                    tc += float4(jj[2], 0.0);
                    td += float4(cross(r, m * vr) + mul(i1, node(k, 9).xyz), 0.0);
                }
                reduce_store(t, ta, tb, tc, td);
                reduce_read(a, b, c, d);
                float3x3 itot = float3x3(a.xyz, b.xyz, c.xyz);
                float3 lrel = d.xyz;
                float3 omega = solve3(itot, lrel);
                float ke_base = 0.5 * (mt * dot(vc, vc) + dot(omega, lrel));
                float4 ra = float4(0.0);
                for (uint j = t; j < n; j += G)
                {
                    uint k = first + j;
                    float m = node(k, 0).w;
                    float3 r = node(k, 0).xyz + node(k, 10).xyz - com1;
                    float3x3 rd = rotvec_matrix(node(k, 11).xyz);
                    float3x3 i1 = mul(mul(rd, world_inertia(node(k, 1), node(k, 2).xyz)), transpose(rd));
                    float3 uu = node(k, 8).xyz - vc - cross(omega, r);
                    float3 wr = node(k, 9).xyz - omega;
                    ra.x += 0.5 * (m * dot(uu, uu) + dot(wr, mul(i1, wr)));
                }
                reduce_store(t, ra, float4(0.0), float4(0.0), float4(0.0));
                reduce_read(a, b, c, d);
                float ke_res = a.x;
                float sc2 = clamp((bound - ke_base) / max(ke_res, 1e-30), 0.0, 1.0);
                float s = sqrt(sc2);
                if (ke_base > bound)
                    dec |= DEC_BOUND_INFEASIBLE;
                for (uint j = t; j < n; j += G)
                {
                    uint k = first + j;
                    float3 r = node(k, 0).xyz + node(k, 10).xyz - com1;
                    float3 uu = node(k, 8).xyz - vc - cross(omega, r);
                    float3 wr = node(k, 9).xyz - omega;
                    nodes[NODE_F4 * k + 8] = float4(vc + cross(omega, r) + s * uu, 0.0);
                    nodes[NODE_F4 * k + 9] = float4(omega + s * wr, 0.0);
                }
            }
            else
            {
                float s = sqrt(min(max(bound, 0.0) / max(ke, 1e-30), 1.0));
                for (uint j = t; j < n; j += G)
                {
                    uint k = first + j;
                    nodes[NODE_F4 * k + 8] = float4(node(k, 8).xyz * s, 0.0);
                    nodes[NODE_F4 * k + 9] = float4(node(k, 9).xyz * s, 0.0);
                }
            }
        }
    }
    DeviceMemoryBarrierWithGroupSync();
    // 4. breaks (failure margin q̂ >= τ) and survivors' damage (< 1)
    for (uint j = t; j < nbonds; j += G)
    {
        uint k = bfirst + j;
        float4 b0 = bonds[BOND_F4 * k];
        float4 b1 = bonds[BOND_F4 * k + 1];
        bool broke = b0.y >= params.log_tau;
        float dmg = min(b0.x + b0.z, 1.0 - 1e-6);
        bonds[BOND_F4 * k + 1] = float4(dmg, asfloat(broke ? 1u : 0u), b1.z, b1.w);
    }
    // non-finite output anywhere: flagged (the fallback gate reads it)
    uint bad = 0;
    for (uint j = t; j < n; j += G)
    {
        uint k = first + j;
        float s = dot(node(k, 8), float4(1.0)) + dot(node(k, 9), float4(1.0)) + dot(node(k, 10), float4(1.0)) + dot(node(k, 11), float4(1.0));
        if (!(abs(s) < 3.0e38))
            bad = 1;
    }
    reduce_store(t, float4(float(bad), 0.0, 0.0, 0.0), float4(0.0), float4(0.0), float4(0.0));
    reduce_read(a, b, c, d);
    if (a.x > 0.0)
        dec |= DEC_NONFINITE;
    if (t == 0)
        calls[CALL_F4 * g + 4] = float4(c4.x, c4.y, asfloat(dec), c4.w);
}
