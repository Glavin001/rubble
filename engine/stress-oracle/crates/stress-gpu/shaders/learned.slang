// The learned impact-outcome model's engine side (`STRESS_IMPACT_PUNCH=learned`, off by
// default; docs/learned-stress-solver/learned-stress-solver-plan.md, P3). The network itself
// is P0's kernel (`impact_learned.slang`) or a stub (`learned_stub`); these kernels feed it
// and apply what it predicts, in the resident frame's one submission. P0's validated
// boundary (findings/p0-feasibility-2026-10-20.md, step 3): one call per routed impact
// island per frame, inputs the frame-start state, applied at window end with the island's
// contacts filtered for that frame (`window_reduced`).
//
// At the end of frame N (after the cascade), in order:
//   learned_apply_bonds   the call decoded at the end of N-1, at its window end: its breaks
//   learned_apply_bodies  and damage into the bond records, its islands marked disconnected;
//                         its bodies released (they were held static through frame N: the
//                         call's contacts filtered), impactors and unsplit islands placed at
//                         start + dx with the momentum-exact fit (stress-ref `apply_decoded`)
//   (the split pass)      with `learned_place` after its spawn: each child of a call's
//                         island placed and fitted from its chunks' decoded motion
//   (the routing check)
//   learned_select        per island: the impactors that reach it within the next frame,
//   learned_debris        (STRESS_DEBRIS_ROUTING=1) and the islands the launched debris reaches
//   learned_compact       one group: the calls (impactors joined by shared islands), their
//                         islands, nodes and bonds laid out; capacities checked
//   learned_gather_chunks / _bonds / _impactors   the frame-start state of every node and bond
//   learned_finish        per call: centre of mass, scales, freedom
//   (the network: P0's kernel, or learned_stub)
//   learned_decode        per call (decode.py)
//   learned_gate          per call: the fallback gate (non-finite, infeasible bound): a call
//                         that fails it goes to the GPU punch (logged by the host); the others'
//                         bodies are held static for frame N+1
//   (the punch, skipping the islands a call took; its split pass)
//
// Layouts (`learned.rs` mirrors them):
//   calls  CALL_F4 float4 per call:
//          0 scales (L, m, v, c)        1 scales (f_t, E, GA, dt)
//          2 (E_in, flags bits, first node bits, nodes bits)
//          3 (gravity xyz, chunk nodes bits)
//          4 (first bond bits, bonds bits, decode flags bits, fallback reason bits)
//          5 (centre of mass (world), impactor mask bits)
//          7 window-end counts (`learned_apply_bonds`)
//          8 (first contact row bits, contact rows bits, -, largest contact speed) (learned_feat)
//          9 (median live bond area, live G_f·A, plug G_f·A, -) (learned_feat)
//          6 (first call island bits, call islands bits, chunk base bits, impactor base bits):
//            the call's first chunk and impactor in the network's global order (every
//            call's chunks, then every call's impactors: `batch.collate`'s)
//   nodes  NODE_F4 float4 per node (a call's chunks, then its impactors, contiguous):
//          0 position relative to the call's centre of mass (world), mass
//          1 world orientation (x, y, z, w; body -> world)
//          2 principal inertia (body), anchored (0/1)
//          3 velocity (world), call bits        4 angular velocity (world), kind bits (0 chunk, 1 impactor)
//          5..7 the network's 12 body-frame outputs (dv, dw, dx, dtheta; nondimensional)
//          8 v_end   9 w_end   10 dx   11 drot  (decoded, world)
//          12 (layout chunk or impactor index bits, engine body bits, structure bits, chunk bits)
//          13 frame-start position (world, absolute), -
//          14 rest centre (body frame), I_xy   15 I_xx, I_yy, I_zz, I_xz   16 I_yz, -  (the
//             chunk's inertia about its centre, body frame: `ChunkStatic.inertia`)
//          17 its body's frame-start orientation (x, y, z, w)  (14..17: the window-end fit's
//             inputs, which `learned_place` cannot bind: chunks only)
//   bonds  BOND_F4 float4 per live bond of a call (undirected):
//          0 (damage at frame start, log q, Δdamage, time bin bits 1..8)
//          1 (decoded damage, broke bits, layout bond bits, call bits)
//   cisl   CISL_U4 uint4 per call island: (island, first node, chunks, call),
//          (engine body, first bond, bonds, anchored)
//   ctl    control words (CTL_*), read back with the frame's status
// Frames: every vector here is the stress solver's (Z-up); the engine's bodies are Y-up:
// (x, y, z) -> (x, -z, y) into the solver.

import common;
import joint;
import bonds;

static const uint CALL_F4 = 10;
static const uint NODE_F4 = 18;
static const uint BOND_F4 = 2;
static const uint CISL_U4 = 2;
static const uint ISL_U4 = 5;
static const uint IMP_F4 = 4;
// 64 threads per group (see learned_feat.slang: Metal caps a heavy pipeline's group size).
static const uint G = 64;
static const uint NONE = 0xffffffffu;

// calls[2].y flags
static const uint CALL_FREE = 1;      // momentum conserved (causal: no anchored chunk, no load, no gravity)
static const uint CALL_EXTERNAL = 2;  // a pressure load does work the inputs cannot bound: no KE bound
static const uint CALL_ANCHORED = 4;
// calls[4].z decode flags
static const uint DEC_BOUND_ACTIVE = 1;
static const uint DEC_BOUND_INFEASIBLE = 2;
static const uint DEC_NONFINITE = 4;
// calls[4].w fallback reasons (bits; 0: the call stands)
static const uint FB_NONFINITE = 1;
static const uint FB_INFEASIBLE = 2;
static const uint FB_RANGE = 4;       // an input outside the model's range (speed over c, size)
static const uint FB_DRY = 16;        // STRESS_LEARNED_DRY: not applied (an export's mode)

// ctl words
static const uint CTL_PENDING = 0;    // 1: a decoded batch waits for its window end
static const uint CTL_CALLS = 1;
static const uint CTL_NODES = 2;
static const uint CTL_BONDS = 3;
static const uint CTL_FROZEN = 4;     // bodies held static (in `frozen`)
static const uint CTL_FALLBACKS = 5;  // calls sent to the punch this frame
static const uint CTL_BOUND = 6;      // calls whose KE bound bound this frame
static const uint CTL_INFEASIBLE = 7;
static const uint CTL_OVERFLOW = 8;   // bits: 1 calls, 2 nodes, 4 bonds, 8 call islands, 16 a call's chunks
static const uint CTL_CISL = 9;
static const uint CTL_STALE = 10;     // applied calls whose islands changed under them (not applied)
static const uint CTL_APPLIED = 11;   // calls applied at window end this frame
static const uint CTL_BROKEN = 12;    // bonds the applied calls broke
static const uint CTL_PLACED = 13;    // children placed
static const uint CTL_FB_NONFINITE = 14;
static const uint CTL_FB_INFEASIBLE = 15;
static const uint CTL_FB_RANGE = 16;
static const uint CTL_ROUTED = 17;    // islands routed into calls this frame
static const uint CTL_PHASE = 18;
static const uint CTL_JOINED = 19;   // islands joined to a call by their approach (`learned_debris`)     // 1: between a batch's window end and the next batch (`learned_place` acts)
static const uint CTL_WORDS = 32;

static const uint ST_DISCONNECTED = 1;
static const uint ST_ISLANDS = 16;
static const uint DISCONNECTED_BIT = 8;
static const uint LEARNED_BIT = 4096; // the island is a learned call's this frame (the punch skips it)

struct LearnedParams
{
    uint impactor_count;
    uint engine_base;     // ctab: per island 3 float4 (resident.slang)
    uint island_cap;      // islands the tables hold
    uint stub;            // 0 the network ran; 1 zero outputs; 2 the punch-like plug
    float dt;             // the frame
    float speed;          // the tiered impact speed
    float log_tau;        // ln τ on the failure margin
    float tolerance;      // KE bound tolerance, a fraction of E_in
    float4 gravity;       // xyz; w: |g|
    uint max_calls;
    uint max_nodes;
    uint max_bonds;
    uint max_call_chunks;
    uint debris;          // 1: STRESS_DEBRIS_ROUTING (islands launched debris reaches join)
    uint max_cisl;
    uint chunk_count;     // layout chunks (`chunk_node`)
    float speed_ref;      // the tiered impact speed, exact (the reference speed's floor)
    // The network's outputs (`lout`, P2's outputs.npz shapes, global node order): float
    // offsets of chunk_out [Nc,12], imp_out [Ni,12], bond_log_q [E], bond_ddamage [E];
    // and of bond_time_logits [E,8].
    uint4 out_bases;
    uint4 out_bases2;
};

// One table for every kernel of this module (each entry point binds at most 10 storage
// buffers: `binding_limits.rs`).
[[vk::binding(0, 0)]] ConstantBuffer<LearnedParams> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> ctl;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> calls;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> nodes;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> bonds;
// Per island ISL_U4 uint4: (impactor mask, call, live bonds, joined), then as float bits its
// bounds lo, hi, its launch (debris routing: 2 (v_impactor - v_island) dt, launched), and
// (its velocity, the mask the impactors gave it: what `learned_debris` reads of others).
[[vk::binding(5, 0)]] RWStructuredBuffer<uint4> isl;
// Per call island 2 float4: (centre of mass (world), mass), (velocity, anchored) (learned_feat).
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> cislf;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint4> cisl;
// Per layout chunk: its call node + 1 (0: none), while a decoded batch waits.
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> chunk_node;
// Per held body 2 float4: (engine body bits, inverse mass, -, -), (inverse inertia, flags bits).
[[vk::binding(9, 0)]] RWStructuredBuffer<float4> frozen;
// Per impactor IMP_F4 float4: (engine body bits, bounding radius, face radius, mass),
// (principal inertia, plane-strain modulus), (half-extent norm, -, -, -).
[[vk::binding(10, 0)]] StructuredBuffer<float4> imps;
[[vk::binding(11, 0)]] StructuredBuffer<Island> islands;
[[vk::binding(12, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(13, 0)]] StructuredBuffer<float4> state;
[[vk::binding(14, 0)]] StructuredBuffer<float4> ctab;
// The engine's bodies (rubble `BodyState`, 7 float4: position and inverse mass, rotation
// (x, y, z, w), inverse inertia and flags, velocity and sleep timer, angular velocity, ...).
[[vk::binding(15, 0)]] RWStructuredBuffer<float4> bodies;
[[vk::binding(16, 0)]] StructuredBuffer<BondStatic> bond_static;
[[vk::binding(17, 0)]] RWStructuredBuffer<BondDyn> bond_dyn;
// cascade.slang status: [1] a pass disconnected, [16 + i] island i's bits.
[[vk::binding(18, 0)]] RWStructuredBuffer<uint> status;
// The split pass's counters: [0] live islands.
[[vk::binding(19, 0)]] StructuredBuffer<uint> counters;
// The split pass (gpu_pass.rs): parents, work (labels; members at the total), child slots,
// control words, the parents' first chunks (`islands_of`).
[[vk::binding(20, 0)]] StructuredBuffer<uint4> split_parents;
[[vk::binding(21, 0)]] StructuredBuffer<uint> split_work;
[[vk::binding(22, 0)]] StructuredBuffer<uint4> split_slots;
[[vk::binding(23, 0)]] StructuredBuffer<uint> split_control;
[[vk::binding(24, 0)]] StructuredBuffer<uint4> split_firsts;
// routing.slang's flag of this frame: [0] 1 when an impactor may reach an island.
[[vk::binding(25, 0)]] StructuredBuffer<uint> route;
// The split pass's child records (split.slang `out`, 8 float4 per child slot).
[[vk::binding(26, 0)]] StructuredBuffer<float4> split_out;
// Indirect dispatch arguments over the batch (`learned_compact`): [0] one group per call,
// [1] one per call island, [2] call islands in groups of 64.
[[vk::binding(27, 0)]] RWStructuredBuffer<uint4> largs;
// The network's outputs (P0's kernel; `params.out_bases`).
[[vk::binding(28, 0)]] StructuredBuffer<float> lout;

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
groupshared uint g_scan[G];

void reduce_store(uint t, float4 a, float4 b, float4 c, float4 d)
{
    // A barrier first: the compiler may sink the last call's read of the result past its
    // trailing barrier, so no thread may overwrite the array until every thread has read it.
    GroupMemoryBarrierWithGroupSync();
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
[numthreads(64, 1, 1)]
void learned_decode(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (g >= ctl[CTL_CALLS])
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

// ------------------------------------------------------------------ frames and bodies

float3 to_zup(float3 v)
{
    return float3(v.x, -v.z, v.y);
}

float3 to_yup(float3 v)
{
    return float3(v.x, v.z, -v.y);
}

// The unit quaternion of a rotation vector (x, y, z, w).
float4 rotvec_quat(float3 r)
{
    float th = length(r);
    if (th < 1e-12)
        return normalize(float4(0.5 * r, 1.0));
    float3 a = r / th;
    return float4(a * sin(0.5 * th), cos(0.5 * th));
}

float4 qmul(float4 a, float4 b)
{
    return quat_vec(quat_mul(quat_of(a), quat_of(b)));
}

// A rotation matrix's quaternion (x, y, z, w).
float4 qconj(float4 q)
{
    return float4(-q.xyz, q.w);
}

float4 matrix_quat(float3x3 m)
{
    float tr = m[0][0] + m[1][1] + m[2][2];
    float4 q;
    if (tr > 0.0)
    {
        float s = sqrt(tr + 1.0) * 2.0;
        q = float4((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s);
    }
    else if (m[0][0] > m[1][1] && m[0][0] > m[2][2])
    {
        float s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q = float4(0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s);
    }
    else if (m[1][1] > m[2][2])
    {
        float s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q = float4((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s);
    }
    else
    {
        float s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q = float4((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s);
    }
    return normalize(q);
}

static const uint BODY_F4 = 7;

// The engine's rotation (Y-up, principal axes) to the solver's frame, as routing.slang does:
// axes⁻¹ · q_rubble · principal⁻¹ (`p_inv` given).
float4 body_rotation_zup(float4 q_rubble, float4 p_inv)
{
    float h = 0.70710678118;
    float4 a_inv = float4(h, 0.0, 0.0, h);
    return normalize(qmul(qmul(a_inv, q_rubble), p_inv));
}

// A world rotation (solver frame) applied to an engine orientation: the same rotation in
// Y-up axes, on the left.
float4 turn_rubble(float4 q_rubble, float3 rotvec_zup)
{
    return normalize(qmul(rotvec_quat(to_yup(rotvec_zup)), q_rubble));
}

struct Frame
{
    float4 q;      // solver frame
    float3 origin; // the structure frame's origin
    float3 com;    // the island's centre of mass (world)
    float3 v;
    float3 w;
    uint body;
};

Frame island_frame(uint i)
{
    Island isl = islands[i];
    Frame f;
    f.body = asuint(ctab[params.engine_base + 3 * i].x);
    f.q = isl.rotation;
    f.origin = isl.position.xyz;
    f.v = isl.velocity.xyz;
    f.w = isl.angular_velocity.xyz;
    if (f.body != NONE)
    {
        float4 p_inv = ctab[params.engine_base + 3 * i + 1];
        float3 com = ctab[params.engine_base + 3 * i + 2].xyz;
        float4 rot = bodies[BODY_F4 * f.body + 1];
        f.q = body_rotation_zup(rot, p_inv);
        f.com = to_zup(bodies[BODY_F4 * f.body].xyz);
        f.origin = f.com - rotate(quat_of(f.q), com);
        f.v = to_zup(bodies[BODY_F4 * f.body + 3].xyz);
        f.w = to_zup(bodies[BODY_F4 * f.body + 4].xyz);
    }
    else
    {
        f.com = f.origin + rotate(quat_of(f.q), isl.com.xyz);
    }
    return f;
}

float3 chunk_world(Frame f, uint k)
{
    return f.origin + rotate(quat_of(f.q), chunks[k].center.xyz + state[4 * k].xyz);
}

// ------------------------------------------------------------------ learned_clear

// The last batch's chunk map emptied and its pending flag cleared (after its window end).
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_clear(uint3 thread: SV_GroupThreadID)
{
    uint t = thread.x;
    uint n = ctl[CTL_NODES];
    for (uint j = t; j < n; j += G)
    {
        float4 w = nodes[NODE_F4 * j + 12];
        if (asuint(nodes[NODE_F4 * j + 4].w) == 0 && asuint(w.x) < params.chunk_count)
            chunk_node[asuint(w.x)] = 0;
    }
    if (t == 0)
    {
        ctl[CTL_PENDING] = 0;
        ctl[CTL_PHASE] = 0;
    }
}

// ------------------------------------------------------------------ learned_select

groupshared float3 g_lo[G];
groupshared float3 g_hi[G];

// Per island: the impactors that reach it within the frame closing faster than the tiered
// impact speed (the punch's sweep, punch.slang), as a mask; its bounds and, for debris
// routing, its launch: 2 (v_impactor - v_island) dt (`select_islands`' elastic bound).
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_select(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    uint t = thread.x;
    uint live = group_uniform(counters[0], t);
    if (i >= live)
        return;
    if (group_uniform(route[0], t) == 0)
    {
        if (t == 0)
        {
            isl[ISL_U4 * i] = uint4(0, NONE, 0, 0);
            isl[ISL_U4 * i + 3] = uint4(0);
            isl[ISL_U4 * i + 4] = uint4(0);
        }
        return;
    }
    Island rec = islands[i];
    uint n = group_uniform(rec.range.y - rec.range.x, t);
    Frame f = island_frame(i);
    float inf = float_bits(0x7f800000u);
    float3 lo = float3(inf);
    float3 hi = float3(-inf);
    for (uint c = t; c < n; c += G)
    {
        uint k = rec.range.x + c;
        float3 p = chunk_world(f, k);
        lo = min(lo, p - chunks[k].half.w);
        hi = max(hi, p + chunks[k].half.w);
    }
    g_lo[t] = lo;
    g_hi[t] = hi;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            g_lo[t] = min(g_lo[t], g_lo[t + s]);
            g_hi[t] = max(g_hi[t], g_hi[t + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t != 0)
        return;
    lo = g_lo[0];
    hi = g_hi[0];
    uint mask = 0;
    float3 launch = float3(0.0);
    bool launched = false;
    for (uint k = 0; k < min(params.impactor_count, 32u); k++)
    {
        float4 a = imps[IMP_F4 * k];
        uint b = asuint(a.x);
        float3 c = to_zup(bodies[BODY_F4 * b].xyz);
        float3 rel = to_zup(bodies[BODY_F4 * b + 3].xyz) - f.v;
        float r = a.y;
        float sp = length(rel);
        if (!(sp > params.speed) || n == 0)
            continue;
        float3 elo = lo - float3(r);
        float3 ehi = hi + float3(r);
        float t0 = 0.0;
        float t1 = params.dt;
        for (uint ax = 0; ax < 3; ax++)
        {
            if (abs(rel[ax]) < 1e-12)
            {
                if (c[ax] < elo[ax] || c[ax] > ehi[ax])
                    t0 = inf;
                continue;
            }
            float ta = (elo[ax] - c[ax]) / rel[ax];
            float tb = (ehi[ax] - c[ax]) / rel[ax];
            t0 = max(t0, min(ta, tb));
            t1 = min(t1, max(ta, tb));
        }
        if (t0 <= t1)
        {
            mask |= 1u << k;
            if (!launched && (rec.info.x & ISLAND_ANCHORED) == 0)
            {
                launched = true;
                launch = 2.0 * rel * params.dt;
            }
        }
    }
    isl[ISL_U4 * i] = uint4(mask, NONE, 0, 0);
    isl[ISL_U4 * i + 1] = asuint(float4(lo, 0.0));
    isl[ISL_U4 * i + 2] = asuint(float4(hi, 0.0));
    isl[ISL_U4 * i + 3] = asuint(float4(launch, launched ? 1.0 : 0.0));
    isl[ISL_U4 * i + 4] = uint4(asuint(f.v), mask);
}

// ------------------------------------------------------------------ learned_debris

// An island no impactor reaches joins a call (one level; only the impactors' masks are
// read of others, `isl[ISL_U4 * i + 4].w`, so the result does not depend on thread order):
//   - STRESS_DEBRIS_ROUTING=1: when a launched island's bounds, swept by its launch, overlap
//     its own (the host routing's elastic bound);
//   - always: when its bounds, swept by its velocity relative to a struck island over the
//     frame, overlap that island's. A called island's bodies are held static through the
//     next frame (the call's contacts filtered), so a body that would meet them must be in
//     the call, or it would meet an immovable object (counted: `joined`).
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_debris(uint3 id: SV_DispatchThreadID)
{
    uint j = id.x;
    uint live = counters[0];
    if (j >= live || route[0] == 0 || isl[ISL_U4 * j + 4].w != 0)
        return;
    float3 lo = asfloat(isl[ISL_U4 * j + 1]).xyz;
    float3 hi = asfloat(isl[ISL_U4 * j + 2]).xyz;
    if (!(lo.x <= hi.x))
        return;
    float3 vj = asfloat(isl[ISL_U4 * j + 4].xyz);
    uint mask = 0;
    bool approach = false;
    for (uint i = 0; i < live; i++)
    {
        uint mi = isl[ISL_U4 * i + 4].w;
        if (mi == 0 || i == j)
            continue;
        float3 a = asfloat(isl[ISL_U4 * i + 1]).xyz;
        float3 b = asfloat(isl[ISL_U4 * i + 2]).xyz;
        float4 l = asfloat(isl[ISL_U4 * i + 3]);
        if (params.debris != 0 && l.w != 0.0)
        {
            float3 slo = min(a, a + l.xyz) - float3(1e-3);
            float3 shi = max(b, b + l.xyz) + float3(1e-3);
            if (all(slo <= hi) && all(lo <= shi))
                mask |= mi;
        }
        float3 rel = (vj - asfloat(isl[ISL_U4 * i + 4].xyz)) * params.dt;
        float3 jlo = min(lo, lo + rel) - float3(1e-3);
        float3 jhi = max(hi, hi + rel) + float3(1e-3);
        if (all(jlo <= b) && all(a <= jhi))
        {
            mask |= mi;
            approach = true;
        }
    }
    if (mask != 0)
        isl[ISL_U4 * j] = uint4(mask, NONE, 0, approach ? 1u : 0u);
}

// ------------------------------------------------------------------ learned_compact

groupshared uint g_label[32];
groupshared uint g_used;
groupshared uint g_changed;
groupshared uint g_call_of[32];
groupshared uint g_cnt_isl[64];
groupshared uint g_cnt_chunk[64];
groupshared uint g_cnt_bond[64];
groupshared uint g_cnt_imp[64];
groupshared uint g_mask_of[64];

uint lowest_bit(uint m)
{
    return firstbitlow(m);
}

// One group: impactors joined into calls by the islands they share (minimum labels, a
// fixed point: order-independent), calls in the order of their lowest impactor; each call's
// islands in island order, its nodes (the islands' chunks, then its impactors) and its live
// bonds laid out; capacities checked (a batch over one is not made: `CTL_OVERFLOW`).
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_compact(uint3 thread: SV_GroupThreadID)
{
    uint t = thread.x;
    uint live = group_uniform(counters[0], t);
    uint nimp = min(params.impactor_count, 32u);
    for (uint i = t; i < live; i += G)
        status[ST_ISLANDS + i] &= ~LEARNED_BIT;
    if (t < 32)
        g_label[t] = t;
    if (t == 0)
    {
        g_used = 0;
        ctl[CTL_CALLS] = 0;
        ctl[CTL_NODES] = 0;
        ctl[CTL_BONDS] = 0;
        ctl[CTL_CISL] = 0;
        ctl[CTL_OVERFLOW] = 0;
        ctl[CTL_FALLBACKS] = 0;
        ctl[CTL_BOUND] = 0;
        ctl[CTL_INFEASIBLE] = 0;
        ctl[CTL_FB_NONFINITE] = 0;
        ctl[CTL_FB_INFEASIBLE] = 0;
        ctl[CTL_FB_RANGE] = 0;
        ctl[CTL_ROUTED] = 0;
    }
    if (t == 0)
    {
        largs[0] = uint4(0, 1, 1, 0);
        largs[1] = uint4(0, 1, 1, 0);
        largs[2] = uint4(0, 1, 1, 0);
    }
    GroupMemoryBarrierWithGroupSync();
    if (group_uniform(route[0], t) == 0)
        return;
    // labels: the minimum over each island's impactors, to a fixed point (one thread: few
    // islands are struck by more than one impactor)
    if (t == 0)
    {
        uint used = 0;
        for (uint i = 0; i < live; i++)
            used |= isl[ISL_U4 * i].x;
        g_used = used;
        for (uint iter = 0; iter < 32; iter++)
        {
            bool changed = false;
            for (uint i = 0; i < live; i++)
            {
                uint m = isl[ISL_U4 * i].x;
                if (countbits(m) < 2)
                    continue;
                uint lo = 32;
                for (uint mm = m; mm != 0; mm &= mm - 1)
                    lo = min(lo, g_label[lowest_bit(mm)]);
                for (uint mm = m; mm != 0; mm &= mm - 1)
                {
                    uint b = lowest_bit(mm);
                    if (g_label[b] > lo)
                    {
                        g_label[b] = lo;
                        changed = true;
                    }
                }
            }
            if (!changed)
                break;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // calls: the roots with a used impactor in their component, ascending
    if (t == 0)
    {
        uint used = g_used;
        uint calls_n = 0;
        for (uint r = 0; r < nimp; r++)
        {
            g_call_of[r] = NONE;
            if (g_label[r] != r)
                continue;
            uint comp = 0;
            for (uint k = 0; k < nimp; k++)
                if (g_label[k] == r)
                    comp |= 1u << k;
            if ((comp & used) == 0)
                continue;
            if (calls_n < 64)
            {
                g_mask_of[calls_n] = comp & used;
                g_cnt_isl[calls_n] = 0;
                g_cnt_chunk[calls_n] = 0;
                g_cnt_bond[calls_n] = 0;
                g_cnt_imp[calls_n] = countbits(comp & used);
            }
            g_call_of[r] = calls_n;
            calls_n++;
        }
        ctl[CTL_CALLS] = calls_n;
    }
    GroupMemoryBarrierWithGroupSync();
    uint ncalls = group_uniform(ctl[CTL_CALLS], t);
    // each routed island's call (its live bonds: `learned_count_bonds`)
    if (t == 0)
    {
        for (uint i = 0; i < live; i++)
        {
            uint4 e = isl[ISL_U4 * i];
            if (e.x != 0)
                isl[ISL_U4 * i] = uint4(e.x, g_call_of[g_label[lowest_bit(e.x)]], e.z, e.w);
        }
    }
    if (t != 0)
        return;
    uint over = 0;
    if (ncalls > min(params.max_calls, 64u))
        over |= 1;
    uint routed = 0, joined = 0;
    for (uint i = 0; i < live; i++)
    {
        uint4 e = isl[ISL_U4 * i];
        if (e.x == 0 || e.y >= 64)
            continue;
        Island rec = islands[i];
        g_cnt_isl[e.y] += 1;
        g_cnt_chunk[e.y] += rec.range.y - rec.range.x;
        g_cnt_bond[e.y] += e.z;
        routed++;
        joined += e.w;
    }
    // prefix over calls
    uint node_at = 0, bond_at = 0, cisl_at = 0;
    uint call_first_cisl[64];
    for (uint g = 0; g < min(ncalls, 64u); g++)
    {
        uint chunks_n = g_cnt_chunk[g];
        uint nodes_n = chunks_n + g_cnt_imp[g];
        if (chunks_n > params.max_call_chunks)
            over |= 16;
        calls[CALL_F4 * g + 2] = float4(0.0, 0.0, asfloat(node_at), asfloat(nodes_n));
        calls[CALL_F4 * g + 3] = float4(params.gravity.xyz, asfloat(chunks_n));
        calls[CALL_F4 * g + 4] = float4(asfloat(bond_at), asfloat(g_cnt_bond[g]), 0.0, 0.0);
        calls[CALL_F4 * g + 5] = float4(0.0, 0.0, 0.0, asfloat(g_mask_of[g]));
        calls[CALL_F4 * g + 6] = float4(asfloat(cisl_at), asfloat(g_cnt_isl[g]), 0.0, 0.0);
        call_first_cisl[g] = cisl_at;
        // reuse the counters as running offsets
        g_cnt_chunk[g] = node_at;
        g_cnt_bond[g] = bond_at;
        g_cnt_isl[g] = cisl_at;
        node_at += nodes_n;
        bond_at += g_cnt_bond[g] == bond_at ? 0 : 0;
        cisl_at += call_first_cisl[g] == cisl_at ? 0 : 0;
    }
    // the per-call totals again (the loop above reused the arrays): from the call records
    bond_at = 0;
    cisl_at = 0;
    uint chunk_base = 0, imp_base = 0, contact_at = 0;
    for (uint g = 0; g < min(ncalls, 64u); g++)
    {
        uint nimp = countbits(asuint(calls[CALL_F4 * g + 5].w));
        calls[CALL_F4 * g + 8] = float4(asfloat(contact_at), asfloat(0u), 0.0, 0.0);
        calls[CALL_F4 * g + 9] = float4(0.0);
        contact_at += nimp * asuint(calls[CALL_F4 * g + 6].y);
        if (asuint(calls[CALL_F4 * g + 4].y) > 131072u)
            over |= 64;
        calls[CALL_F4 * g + 4].x = asfloat(bond_at);
        calls[CALL_F4 * g + 6] = float4(asfloat(cisl_at), calls[CALL_F4 * g + 6].y, asfloat(chunk_base), asfloat(imp_base));
        chunk_base += asuint(calls[CALL_F4 * g + 3].w);
        imp_base += countbits(asuint(calls[CALL_F4 * g + 5].w));
        g_cnt_bond[g] = bond_at;
        g_cnt_isl[g] = cisl_at;
        bond_at += asuint(calls[CALL_F4 * g + 4].y);
        cisl_at += asuint(calls[CALL_F4 * g + 6].y);
    }
    if (node_at > params.max_nodes)
        over |= 2;
    if (bond_at > params.max_bonds)
        over |= 4;
    if (cisl_at > params.max_cisl)
        over |= 8;
    if (over != 0)
    {
        ctl[CTL_OVERFLOW] = over;
        ctl[CTL_CALLS] = 0;
        return;
    }
    // the call islands, in island order within each call
    for (uint i = 0; i < live; i++)
    {
        uint4 e = isl[ISL_U4 * i];
        if (e.x == 0 || e.y >= 64)
            continue;
        Island rec = islands[i];
        uint g = e.y;
        uint at = g_cnt_isl[g];
        uint nchunks = rec.range.y - rec.range.x;
        cisl[CISL_U4 * at] = uint4(i, g_cnt_chunk[g], nchunks, g);
        cisl[CISL_U4 * at + 1] = uint4(asuint(ctab[params.engine_base + 3 * i].x), g_cnt_bond[g], e.z, (rec.info.x & ISLAND_ANCHORED) != 0 ? 1u : 0u);
        g_cnt_isl[g] = at + 1;
        g_cnt_chunk[g] += nchunks;
        g_cnt_bond[g] += e.z;
    }
    ctl[CTL_NODES] = node_at;
    ctl[CTL_BONDS] = bond_at;
    ctl[CTL_CISL] = cisl_at;
    ctl[CTL_ROUTED] = routed;
    ctl[CTL_JOINED] = joined;
    largs[0] = uint4(ncalls, 1, 1, 0);
    largs[1] = uint4(cisl_at, 1, 1, 0);
    largs[2] = uint4((cisl_at + 63) / 64, 1, 1, 0);
}

// One group per routed island: its live bonds counted (`isl[ISL_U4 * i].z`).
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_count_bonds(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    uint t = thread.x;
    uint live = group_uniform(counters[0], t);
    if (i >= live)
        return;
    uint4 e = isl[ISL_U4 * i];
    if (group_uniform(route[0] == 0 || e.x == 0 ? 1u : 0u, t) != 0)
        return;
    Island rec = islands[i];
    uint nb = 0;
    for (uint bi = rec.range.z + t; bi < rec.range.w; bi += G)
    {
        if (connected(bond_dyn[bi].js, bond_static[bi].law.stiff1.w != 0.0))
            nb++;
    }
    GroupMemoryBarrierWithGroupSync();
    g_scan[t] = nb;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_scan[t] += g_scan[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
        isl[ISL_U4 * i] = uint4(e.x, e.y, g_scan[0], e.w);
}

// ------------------------------------------------------------------ gathers

// One group per call island: its chunks' frame-start state (world), and the chunk map.
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_gather_chunks(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint e = group.x;
    uint t = thread.x;
    if (e >= ctl[CTL_CISL])
        return;
    uint4 c0 = cisl[CISL_U4 * e];
    uint4 c1 = cisl[CISL_U4 * e + 1];
    uint i = c0.x;
    Island rec = islands[i];
    Frame f = island_frame(i);
    if (t == 0)
    {
        cislf[2 * e] = float4(f.com, rec.com.w);
        cislf[2 * e + 1] = float4(f.v, c1.w != 0 ? 1.0 : 0.0);
    }
    for (uint c = t; c < c0.z; c += G)
    {
        uint k = rec.range.x + c;
        uint n = c0.y + c;
        ChunkStatic ch = chunks[k];
        float3 x = chunk_world(f, k);
        float3x3 cr = float3x3(ch.crot0.xyz, ch.crot1.xyz, ch.crot2.xyz);
        float3x3 inr = float3x3(ch.inertia0.xyz, ch.inertia1.xyz, ch.inertia2.xyz);
        // the chunk's frame: its rotation rows map chunk axes into the structure's
        float3x3 rest = transpose(cr);
        float3x3 ic = mul(mul(transpose(rest), inr), rest);
        // the chunk's own orientation and exact velocity (cluster × hidden × rest, the rigid
        // motion plus the hidden velocity: the builder's `state_chunk_q_world` / `_vel`)
        float4 q = normalize(qmul(qmul(f.q, rotvec_quat(state[4 * k + 1].xyz)), matrix_quat(rest)));
        float3 v = f.v + cross(f.w, x - f.com) + rotate(quat_of(f.q), state[4 * k + 2].xyz);
        float3 w_c = f.w + rotate(quat_of(f.q), state[4 * k + 3].xyz);
        bool anchored = ch.info.x != SUPPORT_NONE;
        nodes[NODE_F4 * n] = float4(x, ch.center.w);
        nodes[NODE_F4 * n + 1] = q;
        nodes[NODE_F4 * n + 2] = float4(ic[0][0], ic[1][1], ic[2][2], anchored ? 1.0 : 0.0);
        nodes[NODE_F4 * n + 3] = float4(v, asfloat(c0.w));
        nodes[NODE_F4 * n + 4] = float4(w_c, asfloat(0u));
        nodes[NODE_F4 * n + 12] = float4(asfloat(k), asfloat(c1.x), asfloat(ch.info.z), asfloat(ch.info.w));
        nodes[NODE_F4 * n + 14] = float4(ch.center.xyz, inr[0][1]);
        nodes[NODE_F4 * n + 15] = float4(inr[0][0], inr[1][1], inr[2][2], inr[0][2]);
        nodes[NODE_F4 * n + 16] = float4(inr[1][2], 0.0, 0.0, 0.0);
        nodes[NODE_F4 * n + 17] = f.q;
        chunk_node[k] = n + 1;
    }
}

// One group per call island: its live bonds, in bond order (a group scan per block).
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_gather_bonds(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint e = group.x;
    uint t = thread.x;
    if (group_uniform(e >= ctl[CTL_CISL] ? 1u : 0u, t) != 0)
        return;
    uint4 c0 = cisl[CISL_U4 * e];
    uint4 c1 = cisl[CISL_U4 * e + 1];
    Island rec = islands[c0.x];
    uint z = rec.range.z;
    uint w = group_uniform(rec.range.w, t);
    uint carry = 0;
    for (uint base = group_uniform(z, t); base < w; base += G)
    {
        uint bi = base + t;
        bool live = bi < w && connected(bond_dyn[bi].js, bond_static[bi].law.stiff1.w != 0.0);
        GroupMemoryBarrierWithGroupSync();
        g_scan[t] = live ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 1; step < G; step <<= 1)
        {
            uint add = t >= step ? g_scan[t - step] : 0;
            GroupMemoryBarrierWithGroupSync();
            g_scan[t] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (live)
        {
            uint at = c1.y + carry + g_scan[t] - 1;
            bonds[BOND_F4 * at] = float4(bond_dyn[bi].js.damage, -30.0, 0.0, asfloat(1u));
            bonds[BOND_F4 * at + 1] = float4(0.0, asfloat(0u), asfloat(bi), asfloat(c0.w));
        }
        uint total = g_scan[G - 1];
        GroupMemoryBarrierWithGroupSync();
        carry += total;
    }
}

// One group per call: its impactors' nodes, after its chunks.
[shader("compute")]
[numthreads(32, 1, 1)]
void learned_gather_impactors(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint k = thread.x;
    if (g >= ctl[CTL_CALLS])
        return;
    uint mask = asuint(calls[CALL_F4 * g + 5].w);
    if ((mask & (1u << k)) == 0)
        return;
    uint rank = countbits(mask & ((1u << k) - 1u));
    uint first = asuint(calls[CALL_F4 * g + 2].z);
    uint nchunks = asuint(calls[CALL_F4 * g + 3].w);
    uint n = first + nchunks + rank;
    float4 a = imps[IMP_F4 * k];
    float4 b = imps[IMP_F4 * k + 1];
    uint body = asuint(a.x);
    float4 rot = bodies[BODY_F4 * body + 1];
    float4 q = body_rotation_zup(rot, float4(0.0, 0.0, 0.0, 1.0));
    nodes[NODE_F4 * n] = float4(to_zup(bodies[BODY_F4 * body].xyz), a.w);
    nodes[NODE_F4 * n + 1] = q;
    nodes[NODE_F4 * n + 2] = float4(b.xyz, 0.0);
    nodes[NODE_F4 * n + 3] = float4(to_zup(bodies[BODY_F4 * body + 3].xyz), asfloat(g));
    nodes[NODE_F4 * n + 4] = float4(to_zup(bodies[BODY_F4 * body + 4].xyz), asfloat(1u));
    nodes[NODE_F4 * n + 12] = float4(asfloat(k), asfloat(body), asfloat(NONE), asfloat(NONE));
}

// ------------------------------------------------------------------ learned_finish

groupshared float g_sort[4096];
groupshared float4 g_scalars;

// Sorts g_sort[0..n) ascending (n <= 4096; padded with +inf to a power of two).
void sort_group(uint t, uint n)
{
    uint m = 1;
    while (m < n)
        m <<= 1;
    float inf = float_bits(0x7f800000u);
    for (uint j = n + t; j < m; j += G)
        g_sort[j] = inf;
    GroupMemoryBarrierWithGroupSync();
    for (uint size = 2; size <= m; size <<= 1)
    {
        for (uint stride = size >> 1; stride > 0; stride >>= 1)
        {
            for (uint j = t; j < m; j += G)
            {
                uint p = j ^ stride;
                if (p > j)
                {
                    bool up = (j & size) == 0;
                    float a = g_sort[j];
                    float b = g_sort[p];
                    if ((a > b) == up)
                    {
                        g_sort[j] = b;
                        g_sort[p] = a;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}

float median_sorted(uint n)
{
    uint k = n / 2;
    return (n % 2 == 1) ? g_sort[k] : 0.5 * (g_sort[k - 1] + g_sort[k]);
}

// One group per call: its centre of mass (positions made relative to it), the scales the
// decode uses (L, m, v, dt, E_in; the calls schema's definitions), its freedom.
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_finish(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (group_uniform(g >= ctl[CTL_CALLS] ? 1u : 0u, t) != 0)
        return;
    uint first = asuint(calls[CALL_F4 * g + 2].z);
    uint n = group_uniform(asuint(calls[CALL_F4 * g + 2].w), t);
    uint nc = group_uniform(asuint(calls[CALL_F4 * g + 3].w), t);
    float4 a, b, c, d;
    {
        float4 sa = float4(0.0), sb = float4(0.0), sc = float4(0.0);
        for (uint j = t; j < n; j += G)
        {
            uint k = first + j;
            float4 p = node(k, 0);
            sa += float4(p.xyz * p.w, p.w);
            sb += float4(node(k, 3).xyz * p.w, node(k, 2).w);
        }
        reduce_store(t, sa, sb, sc, float4(0.0));
        reduce_read(a, b, c, d);
    }
    float mt = a.w;
    float3 com = a.xyz / mt;
    float3 vcom = b.xyz / mt;
    bool anchored = b.w > 0.0;
    // reference speed: the routing speed, every body's speed relative to the centre of
    // mass and its rim speed, the impactors' closing speeds on the call's chunks
    // (the impactors' closing speeds on the call's islands: `feat_contacts`, calls[8].w)
    float vmax = max(params.speed_ref, calls[CALL_F4 * g + 8].w);
    float ke_in = 0.0;
    float wg = 0.0;
    for (uint j = t; j < n; j += G)
    {
        uint k = first + j;
        float4 p = node(k, 0);
        float3 v = node(k, 3).xyz;
        float3 w = node(k, 4).xyz;
        bool imp = asuint(node(k, 4).w) != 0;
        uint id = asuint(node(k, 12).x);
        float rim = imp ? imps[IMP_F4 * id + 2].x : length(chunks[id].half.xyz);
        vmax = max(vmax, max(length(v - vcom), length(w) * rim));
        float3x3 i0 = world_inertia(node(k, 1), node(k, 2).xyz);
        ke_in += 0.5 * (p.w * dot(v, v) + dot(w, mul(i0, w)));
        wg += p.w * dot(v * params.dt + params.gravity.xyz * (0.5 * params.dt * params.dt), params.gravity.xyz);
    }
    // the largest of every thread's
    GroupMemoryBarrierWithGroupSync();
    g_sort[t] = vmax;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_sort[t] = max(g_sort[t], g_sort[t + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    vmax = g_sort[0];
    GroupMemoryBarrierWithGroupSync();
    reduce_store(t, float4(ke_in, wg, 0.0, 0.0), float4(0.0), float4(0.0), float4(0.0));
    reduce_read(a, b, c, d);
    ke_in = a.x;
    wg = a.y;
    // L: the cube root of the median chunk volume; m: the median chunk mass
    uint ns = min(nc, 4096u);
    for (uint j = t; j < ns; j += G)
    {
        uint id = asuint(node(first + j, 12).x);
        float3 h = chunks[id].half.xyz;
        g_sort[j] = 8.0 * h.x * h.y * h.z;
    }
    sort_group(t, ns);
    // medians kept in their own words (a read of the sorted array may be moved past a
    // barrier by the compiler, and the array is reused)
    if (t == 0)
        g_scalars.x = pow(median_sorted(ns), 1.0 / 3.0);
    GroupMemoryBarrierWithGroupSync();
    for (uint j = t; j < ns; j += G)
        g_sort[j] = node(first + j, 0).w;
    sort_group(t, ns);
    if (t == 0)
        g_scalars.y = median_sorted(ns);
    GroupMemoryBarrierWithGroupSync();
    float L = g_scalars.x;
    float mmed = g_scalars.y;
    for (uint j = t; j < n; j += G)
    {
        uint k = first + j;
        float4 p = node(k, 0);
        nodes[NODE_F4 * k + 13] = float4(p.xyz, node(k, 13).w);
        nodes[NODE_F4 * k] = float4(p.xyz - com, p.w);
    }
    if (t == 0)
    {
        float e_in = max(ke_in + wg, 0.5 * mmed * vmax * vmax);
        uint flags = anchored ? CALL_ANCHORED : 0u;
        if (!anchored && params.gravity.w == 0.0)
            flags |= CALL_FREE;
        float4 c2 = calls[CALL_F4 * g + 2];
        calls[CALL_F4 * g] = float4(L, mmed, vmax, 0.0);
        calls[CALL_F4 * g + 1] = float4(0.0, 0.0, 0.0, params.dt);
        calls[CALL_F4 * g + 2] = float4(e_in, asfloat(flags), c2.z, c2.w);
        float4 c5 = calls[CALL_F4 * g + 5];
        calls[CALL_F4 * g + 5] = float4(com, c5.w);
    }
}

// ------------------------------------------------------------------ learned_stub

// The network's place before its weights exist (`STRESS_LEARNED_STUB`):
//   1 zero: every output 0 (ballistic motion, no break);
//   3 nonfinite: NaN outputs (a test of the fallback gate);
//   5 punch-overlap: the punch's outcome with the impactor placed 5 mm into the plug;
//   4 punch-wobble: the punch's outcome plus a fixed per-chunk pattern (non-rigid pieces);
//   2 punch: the punch's idea as an outcome: per impactor, the chunks within its face
//     radius of its path from the struck depth on are a plug; the bonds holding the plug
//     break; the plug and the impactor leave at their common (inelastic) velocity.
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_stub(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (group_uniform(g >= ctl[CTL_CALLS] || (params.stub & 0xffu) == 0 ? 1u : 0u, t) != 0)
        return;
    uint first = asuint(calls[CALL_F4 * g + 2].z);
    uint n = group_uniform(asuint(calls[CALL_F4 * g + 2].w), t);
    uint nc = group_uniform(asuint(calls[CALL_F4 * g + 3].w), t);
    uint bfirst = asuint(calls[CALL_F4 * g + 4].x);
    uint nb = asuint(calls[CALL_F4 * g + 4].y);
    float vref = calls[CALL_F4 * g].z;
    for (uint j = t; j < n; j += G)
    {
        uint k = first + j;
        nodes[NODE_F4 * k + 5] = float4(0.0);
        nodes[NODE_F4 * k + 6] = float4(0.0);
        nodes[NODE_F4 * k + 7] = float4(0.0);
    }
    for (uint j = t; j < nb; j += G)
    {
        uint k = bfirst + j;
        float4 b0 = bonds[BOND_F4 * k];
        bonds[BOND_F4 * k] = float4(b0.x, -30.0, 0.0, asfloat(1u));
    }
    DeviceMemoryBarrierWithGroupSync();
    // 3 nonfinite (a test of the fallback gate): every output NaN
    if (group_uniform((params.stub & 0xffu) == 3 ? 1u : 0u, t) != 0)
    {
        float nan = float_bits(0x7fc00000u);
        for (uint j = t; j < n; j += G)
            nodes[NODE_F4 * (first + j) + 5] = float4(nan);
        return;
    }
    uint mode = params.stub & 0xffu;
    if (group_uniform(mode == 2 || mode == 4 || mode == 5 ? 1u : 0u, t) == 0)
        return;
    // the plug of the call's first impactor (the call's nodes after its chunks)
    uint ki = first + nc;
    if (group_uniform(n > nc ? 1u : 0u, t) == 0)
        return;
    float3 xi = node(ki, 0).xyz;
    float3 vi = node(ki, 3).xyz;
    float mi = node(ki, 0).w;
    float face = imps[IMP_F4 * asuint(node(ki, 12).x)].z;
    float3 dir = normalize(vi - calls[CALL_F4 * g + 5].xyz * 0.0);
    // the struck depth: the nearest chunk along the path within the face
    float inf = float_bits(0x7f800000u);
    float best = inf;
    for (uint j = t; j < nc; j += G)
    {
        uint k = first + j;
        float3 r = node(k, 0).xyz - xi;
        float along = dot(r, dir);
        float h = chunks[asuint(node(k, 12).x)].half.w;
        if (length(r - dir * along) <= face + h && along > 0.0)
            best = min(best, along);
    }
    GroupMemoryBarrierWithGroupSync();
    g_sort[t] = best;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_sort[t] = min(g_sort[t], g_sort[t + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float entry = g_sort[0];
    GroupMemoryBarrierWithGroupSync();
    // plug mass and momentum
    float4 sa = float4(0.0);
    for (uint j = t; j < nc; j += G)
    {
        uint k = first + j;
        float3 r = node(k, 0).xyz - xi;
        float along = dot(r, dir);
        float h = chunks[asuint(node(k, 12).x)].half.w;
        bool plug = entry < inf && along >= entry - h && length(r - dir * along) <= face && node(k, 2).w == 0.0;
        nodes[NODE_F4 * k + 12].y = node(k, 12).y; // (unchanged)
        if (plug)
            sa += float4(node(k, 3).xyz * node(k, 0).w, node(k, 0).w);
    }
    float4 a, b, c, d;
    reduce_store(t, sa, float4(0.0), float4(0.0), float4(0.0));
    reduce_read(a, b, c, d);
    float mp = a.w;
    if (group_uniform(mp > 0.0 ? 1u : 0u, t) == 0)
        return;
    float3 vc = (a.xyz + mi * vi) / (mp + mi);
    for (uint j = t; j < nc; j += G)
    {
        uint k = first + j;
        float3 r = node(k, 0).xyz - xi;
        float along = dot(r, dir);
        float h = chunks[asuint(node(k, 12).x)].half.w;
        bool plug = along >= entry - h && length(r - dir * along) <= face && node(k, 2).w == 0.0;
        if (plug)
        {
            Quat q = quat_of(node(k, 1));
            float3 o = inverse_rotate(q, vc - node(k, 3).xyz) / vref;
            nodes[NODE_F4 * k + 5] = float4(o, 0.0);
            nodes[NODE_F4 * k + 6] = float4(0.0, 0.0, 0.5 * o.x, 0.5 * o.y);
            nodes[NODE_F4 * k + 7] = float4(0.5 * o.z, 0.0, 0.0, 0.0);
            nodes[NODE_F4 * k + 4] = float4(node(k, 4).xyz, asfloat(2u)); // plug mark (kind 0 | 2)
        }
    }
    if (t == 0)
    {
        Quat q = quat_of(node(ki, 1));
        float3 o = inverse_rotate(q, vc - vi) / vref;
        float3 ox = 0.5 * o;
        // 5 punch-overlap (a test of the window end's handover): the impactor placed 5 mm
        // further along its path than the punch's outcome, into the plug ahead of it
        if (mode == 5)
            ox += inverse_rotate(q, dir * 0.005) / (vref * params.dt);
        nodes[NODE_F4 * ki + 5] = float4(o, 0.0);
        nodes[NODE_F4 * ki + 6] = float4(0.0, 0.0, ox.x, ox.y);
        nodes[NODE_F4 * ki + 7] = float4(ox.z, 0.0, 0.0, 0.0);
    }
    DeviceMemoryBarrierWithGroupSync();
    // the bonds holding the plug
    for (uint j = t; j < nb; j += G)
    {
        uint k = bfirst + j;
        uint bi = asuint(bonds[BOND_F4 * k + 1].z);
        JointBond law = bond_static[bi].law;
        uint na = chunk_node[law.ids.y];
        uint nb2 = chunk_node[law.ids.z];
        bool pa = na != 0 && (asuint(nodes[NODE_F4 * (na - 1) + 4].w) & 2u) != 0;
        bool pb = nb2 != 0 && (asuint(nodes[NODE_F4 * (nb2 - 1) + 4].w) & 2u) != 0;
        if (pa != pb)
        {
            float4 b0 = bonds[BOND_F4 * k];
            bonds[BOND_F4 * k] = float4(b0.x, 1.0, 0.0, asfloat(1u));
        }
    }
    DeviceMemoryBarrierWithGroupSync();
    for (uint j = t; j < nc; j += G)
    {
        uint k = first + j;
        nodes[NODE_F4 * k + 4] = float4(node(k, 4).xyz, asfloat(0u));
    }
    // 4 punch-wobble (a test of the kept-deformation boundary): the punch's outcome plus a
    // small fixed pattern on every chunk's outputs, so the chunks of a piece no longer move
    // rigidly
    if (mode == 4)
    {
        DeviceMemoryBarrierWithGroupSync();
        for (uint j = t; j < nc; j += G)
        {
            uint k = first + j;
            float a = float(j);
            float3 wob = 2e-3 * float3(sin(1.3 * a), cos(0.7 * a), sin(0.9 * a + 1.0));
            nodes[NODE_F4 * k + 5] = node(k, 5) + float4(wob, 0.0);
            nodes[NODE_F4 * k + 6] = node(k, 6) + float4(0.0, 0.0, wob.x, wob.y);
            nodes[NODE_F4 * k + 7] = node(k, 7) + float4(wob.z, 0.0, 0.0, 0.0);
        }
    }
}

// ------------------------------------------------------------------ learned_gate

// One group per call, after the decode: the fallback gate (a non-finite outcome, or a KE
// bound that cannot hold without breaking momentum: the call goes to the punch, counted and
// logged by the host); a call that stands takes its islands (the punch skips them) and its
// bodies are held static for the next frame (the call's contacts filtered: P0's boundary),
// their state kept to release them at its window end.
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_gate(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (g >= ctl[CTL_CALLS])
        return;
    float4 c4 = calls[CALL_F4 * g + 4];
    uint dec = asuint(c4.z);
    uint fb = 0;
    if ((dec & DEC_NONFINITE) != 0)
        fb |= FB_NONFINITE;
    if ((dec & DEC_BOUND_INFEASIBLE) != 0)
        fb |= FB_INFEASIBLE;
    // STRESS_LEARNED_DRY=1: no call stands (a feature export's mode)
    if ((params.stub & 0x100u) != 0)
        fb |= FB_DRY;
    uint e0 = asuint(calls[CALL_F4 * g + 6].x);
    uint ne = asuint(calls[CALL_F4 * g + 6].y);
    uint mask = asuint(calls[CALL_F4 * g + 5].w);
    if (t == 0)
        calls[CALL_F4 * g + 4] = float4(c4.x, c4.y, c4.z, asfloat(fb));
    if (fb == FB_DRY)
    {
        // a dry call still takes its islands from the punch (the state stays the batch's)
        for (uint e = e0 + t; e < e0 + ne; e += 64)
            status[ST_ISLANDS + cisl[CISL_U4 * e].x] |= LEARNED_BIT;
    }
    if (fb != 0)
    {
        uint first = asuint(calls[CALL_F4 * g + 2].z);
        uint nc = asuint(calls[CALL_F4 * g + 3].w);
        for (uint j = t; j < nc; j += 64)
            chunk_node[asuint(nodes[NODE_F4 * (first + j) + 12].x)] = 0;
        return;
    }
    for (uint e = e0 + t; e < e0 + ne; e += 64)
    {
        uint4 c0 = cisl[CISL_U4 * e];
        uint4 c1 = cisl[CISL_U4 * e + 1];
        status[ST_ISLANDS + c0.x] |= LEARNED_BIT;
        uint body = c1.x;
        if (body == NONE || c1.w != 0)
        {
            frozen[4 * e] = float4(asfloat(NONE), 0.0, 0.0, 0.0);
            continue;
        }
        uint at = BODY_F4 * body;
        frozen[4 * e] = float4(asfloat(body), bodies[at].w, 0.0, 0.0);
        frozen[4 * e + 1] = bodies[at + 2];
        frozen[4 * e + 2] = bodies[at + 3];
        frozen[4 * e + 3] = bodies[at + 4];
        bodies[at] = float4(bodies[at].xyz, 0.0);
        bodies[at + 2] = float4(0.0, 0.0, 0.0, bodies[at + 2].w);
        bodies[at + 3] = float4(0.0, 0.0, 0.0, bodies[at + 3].w);
        bodies[at + 4] = float4(0.0, 0.0, 0.0, bodies[at + 4].w);
    }
    if (t < 32 && (mask & (1u << t)) != 0)
    {
        uint e = params.max_cisl + t;
        uint body = asuint(imps[IMP_F4 * t].x);
        uint at = BODY_F4 * body;
        frozen[4 * e] = float4(asfloat(body), bodies[at].w, 0.0, 0.0);
        frozen[4 * e + 1] = bodies[at + 2];
        frozen[4 * e + 2] = bodies[at + 3];
        frozen[4 * e + 3] = bodies[at + 4];
        bodies[at] = float4(bodies[at].xyz, 0.0);
        bodies[at + 2] = float4(0.0, 0.0, 0.0, bodies[at + 2].w);
        bodies[at + 3] = float4(0.0, 0.0, 0.0, bodies[at + 3].w);
        bodies[at + 4] = float4(0.0, 0.0, 0.0, bodies[at + 4].w);
    }
}

// ------------------------------------------------------------------ window end

static const uint FB_STALE = 8; // the call's islands changed under it (not applied)

// Whether call island `e` is still as the call saw it: every chunk node's layout chunk
// holds the same chunk, all in one island of the same size.
bool island_current(uint e)
{
    uint4 c0 = cisl[CISL_U4 * e];
    uint first = c0.y;
    uint island = NONE;
    for (uint j = 0; j < c0.z; j++)
    {
        float4 w = nodes[NODE_F4 * (first + j) + 12];
        uint k = asuint(w.x);
        if (k >= params.chunk_count)
            return false;
        uint4 info = chunks[k].info;
        if (info.z != asuint(w.z) || info.w != asuint(w.w))
            return false;
        if (j == 0)
            island = info.y;
        else if (info.y != island)
            return false;
    }
    if (c0.z == 0)
        return false;
    Island rec = islands[island];
    return rec.range.y - rec.range.x == c0.z;
}

groupshared uint g_stale;
groupshared uint g_broke;


// One group per call (pending): a call that stood is checked current at its window end
// (its islands as it saw them); a stale one is marked (not applied, its chunks unmapped).
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_check(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (group_uniform(ctl[CTL_PENDING] == 0 || g >= ctl[CTL_CALLS] ? 1u : 0u, t) != 0)
        return;
    float4 c4 = calls[CALL_F4 * g + 4];
    if (group_uniform(asuint(c4.w), t) != 0)
        return;
    uint e0 = asuint(calls[CALL_F4 * g + 6].x);
    uint ne = asuint(calls[CALL_F4 * g + 6].y);
    if (t == 0)
        g_stale = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint e = e0 + t; e < e0 + ne; e += G)
        if (!island_current(e))
            g_stale = 1;
    GroupMemoryBarrierWithGroupSync();
    if (group_uniform(g_stale, t) == 0)
        return;
    if (t == 0)
    {
        calls[CALL_F4 * g + 4] = float4(c4.x, c4.y, c4.z, asfloat(FB_STALE));
        calls[CALL_F4 * g + 7] = float4(asfloat(1u), asfloat(0u), asfloat(0u), 0.0);
    }
    uint first = asuint(calls[CALL_F4 * g + 2].z);
    uint nc = asuint(calls[CALL_F4 * g + 3].w);
    for (uint j = t; j < nc; j += G)
    {
        uint k = asuint(nodes[NODE_F4 * (first + j) + 12].x);
        if (k < params.chunk_count)
            chunk_node[k] = 0;
    }
}

// One group per call (pending, after `learned_check`): the window end of a call that stood:
// its breaks (the bond records as a break leaves them, as the punch's), each survivor's
// damage raised to the decoded value (kept below 1), its broken islands marked
// disconnected for the split pass.
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_apply_bonds(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (group_uniform(ctl[CTL_PENDING] == 0 || g >= ctl[CTL_CALLS] ? 1u : 0u, t) != 0)
        return;
    float4 c4 = calls[CALL_F4 * g + 4];
    if (group_uniform(asuint(c4.w), t) != 0)
        return;
    if (t == 0)
        g_broke = 0;
    GroupMemoryBarrierWithGroupSync();
    uint bfirst = asuint(c4.x);
    uint nb = asuint(c4.y);
    uint count = 0;
    for (uint j = t; j < nb; j += G)
    {
        uint k = bfirst + j;
        float4 b1 = bonds[BOND_F4 * k + 1];
        uint bi = asuint(b1.z);
        BondDyn dyn = bond_dyn[bi];
        if (asuint(b1.y) != 0)
        {
            if (connected(dyn.js, bond_static[bi].law.stiff1.w != 0.0))
            {
                dyn.js.damage = 1.0;
                dyn.js.rebar_broken = 1.0;
                dyn.events.z = max(dyn.events.z, 1u);
                bond_dyn[bi] = dyn;
                count++;
                uint island = chunks[bond_static[bi].law.ids.y].info.y;
                status[ST_ISLANDS + island] |= DISCONNECTED_BIT;
                g_broke = 1;
            }
        }
        else
        {
            dyn.js.damage = max(dyn.js.damage, min(b1.x, 1.0 - 1e-6));
            bond_dyn[bi] = dyn;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    g_scan[t] = count;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_scan[t] += g_scan[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
    {
        calls[CALL_F4 * g + 7] = float4(asfloat(0u), asfloat(g_scan[0]), asfloat(1u), 0.0);
        if (g_broke != 0)
            status[ST_DISCONNECTED] = 1;
    }
}

// The rigid fit of chunk nodes (stress-ref `apply_decoded`): the centre of mass moves to
// the mass-weighted mean of the chunks' start + dx, the body turns by their mass-weighted
// mean drot, and its velocity and spin are the momentum-exact fit to their decoded v and w
// at the turned pose.
struct Fit
{
    float3 com0;
    float3 com1;
    float4 turn;
    float3 v;
    float3 w;
};

// Sums over nodes `idx(j)`, j < n, by one thread (a call island's or a child's chunks): the
// body fit of stress-ref's `apply_decoded_with` (P0's definition). COM1 the mass mean of the
// decoded positions, R1 = exp(mass-mean drot) · R0 (`turn` · the body's frame-start rotation),
// v the mass-mean velocity, ω = I_w⁻¹ Σ [(x1 − COM1) × m v1 + R1 I_c R1ᵀ w1] with I_w =
// R1 I_body R1ᵀ, I_body the chunks' rest inertia about their rest centre of mass (I_c body
// frame, as `ChunkStatic.inertia`).
Fit fit_nodes(uint first, uint n, bool listed, uint list_base, uint list_first)
{
    // positions about the first chunk's frame-start position (f32: small arms)
    uint k0 = listed ? chunk_node[list_first + split_work[list_base]] - 1 : first;
    float3 ref = node(k0, 13).xyz;
    float m = 0.0;
    float3 x0 = float3(0.0), x1 = float3(0.0), rot = float3(0.0), p = float3(0.0), c = float3(0.0);
    for (uint j = 0; j < n; j++)
    {
        uint k = listed ? chunk_node[list_first + split_work[list_base + j]] - 1 : first + j;
        float mc = node(k, 0).w;
        float3 xs = node(k, 13).xyz - ref;
        m += mc;
        x0 += xs * mc;
        x1 += (xs + node(k, 10).xyz) * mc;
        rot += node(k, 11).xyz * mc;
        p += node(k, 8).xyz * mc;
        c += node(k, 14).xyz * mc;
    }
    Fit f;
    float3 rc1 = x1 / m;
    f.com0 = ref + x0 / m;
    f.com1 = ref + rc1;
    f.turn = rotvec_quat(rot / m);
    f.v = p / m;
    c = c / m;
    // the body's frame-start rotation (any of its chunks': a child's is its parent's)
    float4 r0 = node(k0, 17);
    float3x3 r1 = quat_matrix(normalize(qmul(f.turn, r0)));
    float3x3 eye = float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
    float3 l = float3(0.0);
    float3x3 ibody = float3x3(float3(0.0), float3(0.0), float3(0.0));
    for (uint j = 0; j < n; j++)
    {
        uint k = listed ? chunk_node[list_first + split_work[list_base + j]] - 1 : first + j;
        float mc = node(k, 0).w;
        float4 i14 = node(k, 14), i15 = node(k, 15);
        float i16 = node(k, 16).x;
        float3x3 icb = float3x3(float3(i15.x, i14.w, i15.w), float3(i14.w, i15.y, i16), float3(i15.w, i16, i15.z));
        float3 d = i14.xyz - c;
        ibody += icb + mc * (dot(d, d) * eye - outer3(d, d));
        // (Σ m r = 0: the velocities about their mean, the same sum with less cancellation)
        float3 r = (node(k, 13).xyz - ref) + node(k, 10).xyz - rc1;
        l += cross(r, mc * (node(k, 8).xyz - f.v)) + mul(mul(mul(r1, icb), transpose(r1)), node(k, 9).xyz);
    }
    f.w = solve3(mul(mul(r1, ibody), transpose(r1)), l);
    return f;
}

// Writes a body's placed state (engine frame), its mass as `held` kept it.
void place_body(uint body, Fit f, float4 held_mass, float4 held_inertia)
{
    uint at = BODY_F4 * body;
    float3 pos = to_zup(bodies[at].xyz);
    float3 p1 = f.com1 + rotate(quat_of(f.turn), pos - f.com0);
    bodies[at] = float4(to_yup(p1), held_mass.y);
    bodies[at + 1] = turn_rubble(bodies[at + 1], length(f.turn.xyz) > 0.0 ? 2.0 * atan2(length(f.turn.xyz), f.turn.w) * normalize(f.turn.xyz) : float3(0.0));
    bodies[at + 2] = held_inertia;
    bodies[at + 3] = float4(to_yup(f.v), 0.0);
    bodies[at + 4] = float4(to_yup(f.w), bodies[at + 4].w);
}

// One thread per call island (pending): its body released; a call that stood places an
// unsplit, unanchored island by its fit (a split one stays at its frame-start pose: the
// split takes it, and `learned_place` places its children).
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_apply_bodies(uint3 id: SV_DispatchThreadID)
{
    uint e = id.x;
    if (ctl[CTL_PENDING] == 0 || e >= ctl[CTL_CISL])
        return;
    uint4 c0 = cisl[CISL_U4 * e];
    uint4 c1 = cisl[CISL_U4 * e + 1];
    float4 h0 = frozen[4 * e];
    uint body = asuint(h0.x);
    uint g = c0.w;
    bool stood = asuint(calls[CALL_F4 * g + 4].w) == 0;
    if (body == NONE)
        return;
    uint at = BODY_F4 * body;
    if (!stood)
    {
        // released where it is held, with its frame-start motion (a stale or fallen-back call)
        uint fb = asuint(calls[CALL_F4 * g + 4].w);
        if ((fb & FB_STALE) != 0)
        {
            bodies[at] = float4(bodies[at].xyz, h0.y);
            bodies[at + 2] = frozen[4 * e + 1];
            bodies[at + 3] = frozen[4 * e + 2];
            bodies[at + 4] = frozen[4 * e + 3];
        }
        return;
    }
    uint island = chunks[asuint(nodes[NODE_F4 * c0.y + 12].x)].info.y;
    bool split = (status[ST_ISLANDS + island] & DISCONNECTED_BIT) != 0;
    if (split)
    {
        bodies[at] = float4(bodies[at].xyz, h0.y);
        bodies[at + 2] = frozen[4 * e + 1];
        bodies[at + 3] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    Fit f = fit_nodes(c0.y, c0.z, false, 0, 0);
    place_body(body, f, h0, frozen[4 * e + 1]);
}

// One thread per impactor of a call (pending): released at start + dx, turned by drot,
// with the decoded v and w (a stale call: as held, with its frame-start motion).
[shader("compute")]
[numthreads(32, 1, 1)]
void learned_apply_impactors(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint k = thread.x;
    if (ctl[CTL_PENDING] == 0 || g >= ctl[CTL_CALLS])
        return;
    uint mask = asuint(calls[CALL_F4 * g + 5].w);
    uint fb = asuint(calls[CALL_F4 * g + 4].w);
    if ((mask & (1u << k)) == 0 || (fb != 0 && (fb & FB_STALE) == 0))
        return;
    uint e = params.max_cisl + k;
    float4 h0 = frozen[4 * e];
    uint body = asuint(h0.x);
    uint at = BODY_F4 * body;
    bodies[at] = float4(bodies[at].xyz, h0.y);
    bodies[at + 2] = frozen[4 * e + 1];
    if (fb != 0)
    {
        bodies[at + 3] = frozen[4 * e + 2];
        bodies[at + 4] = frozen[4 * e + 3];
        return;
    }
    uint rank = countbits(mask & ((1u << k) - 1u));
    uint n = asuint(calls[CALL_F4 * g + 2].z) + asuint(calls[CALL_F4 * g + 3].w) + rank;
    float3 dx = node(n, 10).xyz;
    float3 dr = node(n, 11).xyz;
    bodies[at] = float4(bodies[at].xyz + to_yup(dx), h0.y);
    bodies[at + 1] = turn_rubble(bodies[at + 1], dr);
    bodies[at + 3] = float4(to_yup(node(n, 8).xyz), 0.0);
    bodies[at + 4] = float4(to_yup(node(n, 9).xyz), bodies[at + 4].w);
}

// One thread per child slot of the split pass (`gpu_pass.rs`, after its spawn): a child of
// a call's island placed by its chunks' fit.
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_place(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (ctl[CTL_PHASE] == 0 || split_control[CTL_ABORT] != 0)
        return;
    uint np = split_control[CTL_PARENTS];
    if (np == 0)
        return;
    uint total = split_parents[np].y;
    if (slot >= total)
        return;
    uint lo = 0, hi = np - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi + 1) / 2;
        if (split_parents[mid].y <= slot)
            lo = mid;
        else
            hi = mid - 1;
    }
    uint4 parent = split_parents[lo];
    if (slot - parent.y >= parent.z)
        return;
    uint4 s = split_slots[slot];
    if (s.w == 0)
        return;
    float4 head = split_out[8 * slot];
    if (asuint(head.w) != 0)
        return;
    uint offset = asuint(head.y);
    uint count = asuint(head.z);
    uint base = total + parent.y + offset;
    uint first = split_firsts[lo].x;
    // every member a chunk of a call that stood (`learned_gate` unmaps a fallen-back
    // call's chunks; `learned_apply_bonds` a stale one's)
    if (count == 0)
        return;
    for (uint j = 0; j < count; j++)
    {
        if (chunk_node[first + split_work[base + j]] == 0)
            return;
    }
    Fit f = fit_nodes(0, count, true, base, first);
    uint at = BODY_F4 * s.x;
    place_body(s.x, f, float4(0.0, bodies[at].w, 0.0, 0.0), bodies[at + 2]);
}

// One thread: the window end's counts (`learned_apply_bonds`' per call), for the frame read.
[shader("compute")]
[numthreads(1, 1, 1)]
void learned_tally_apply()
{
    if (ctl[CTL_PENDING] == 0)
        return;
    uint stale = 0, broken = 0, applied = 0;
    for (uint g = 0; g < ctl[CTL_CALLS]; g++)
    {
        if (asuint(calls[CALL_F4 * g + 4].w) != 0 && asuint(calls[CALL_F4 * g + 4].w) != FB_STALE)
            continue;
        float4 c7 = calls[CALL_F4 * g + 7];
        stale += asuint(c7.x);
        broken += asuint(c7.y);
        applied += asuint(c7.z);
    }
    ctl[CTL_PHASE] = 1;
    ctl[CTL_STALE] = stale;
    ctl[CTL_BROKEN] = broken;
    ctl[CTL_APPLIED] = applied;
}

// One thread: the batch's counts after the gate (fallbacks by reason, bound activations)
// and whether any call stood (its window end is pending).
[shader("compute")]
[numthreads(1, 1, 1)]
void learned_tally()
{
    uint n = ctl[CTL_CALLS];
    uint fallbacks = 0, bound = 0, infeasible = 0, nonfinite = 0, fb_inf = 0, range = 0, stood = 0;
    for (uint g = 0; g < n; g++)
    {
        uint dec = asuint(calls[CALL_F4 * g + 4].z);
        uint fb = asuint(calls[CALL_F4 * g + 4].w);
        bound += (dec & DEC_BOUND_ACTIVE) != 0 ? 1u : 0u;
        infeasible += (dec & DEC_BOUND_INFEASIBLE) != 0 ? 1u : 0u;
        if (fb != 0)
        {
            fallbacks++;
            nonfinite += (fb & FB_NONFINITE) != 0 ? 1u : 0u;
            fb_inf += (fb & FB_INFEASIBLE) != 0 ? 1u : 0u;
            range += (fb & FB_RANGE) != 0 ? 1u : 0u;
        }
        else
        {
            stood++;
        }
        calls[CALL_F4 * g + 7] = float4(0.0);
    }
    ctl[CTL_FALLBACKS] = fallbacks;
    ctl[CTL_BOUND] = bound;
    ctl[CTL_INFEASIBLE] = infeasible;
    ctl[CTL_FB_NONFINITE] = nonfinite;
    ctl[CTL_FB_INFEASIBLE] = fb_inf;
    ctl[CTL_FB_RANGE] = range;
    ctl[CTL_PENDING] = stood > 0 ? 1u : 0u;
}

// ------------------------------------------------------------------ learned_adopt

// One group per call: the network's outputs, written in P2's outputs.npz shapes in the
// global node order (P0's kernel), into the slots the decode reads: each node's 12
// body-frame outputs, each live bond's log q̂, Δdamage and time bin (argmax + 1; ties: the
// first, as numpy's).
[shader("compute")]
[numthreads(64, 1, 1)]
void learned_adopt(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (g >= ctl[CTL_CALLS])
        return;
    uint first = asuint(calls[CALL_F4 * g + 2].z);
    uint n = asuint(calls[CALL_F4 * g + 2].w);
    uint nc = asuint(calls[CALL_F4 * g + 3].w);
    uint bfirst = asuint(calls[CALL_F4 * g + 4].x);
    uint nb = asuint(calls[CALL_F4 * g + 4].y);
    uint cbase = asuint(calls[CALL_F4 * g + 6].z);
    uint ibase = asuint(calls[CALL_F4 * g + 6].w);
    for (uint j = t; j < n; j += G)
    {
        uint at = j < nc ? params.out_bases.x + 12 * (cbase + j) : params.out_bases.y + 12 * (ibase + j - nc);
        uint k = first + j;
        nodes[NODE_F4 * k + 5] = float4(lout[at], lout[at + 1], lout[at + 2], lout[at + 3]);
        nodes[NODE_F4 * k + 6] = float4(lout[at + 4], lout[at + 5], lout[at + 6], lout[at + 7]);
        nodes[NODE_F4 * k + 7] = float4(lout[at + 8], lout[at + 9], lout[at + 10], lout[at + 11]);
    }
    for (uint j = t; j < nb; j += G)
    {
        uint b = bfirst + j;
        uint tl = params.out_bases2.x + 8 * b;
        uint best = 0;
        for (uint i = 1; i < 8; i++)
            if (lout[tl + i] > lout[tl + best])
                best = i;
        float4 b0 = bonds[BOND_F4 * b];
        bonds[BOND_F4 * b] = float4(b0.x, lout[params.out_bases.z + b], lout[params.out_bases.w + b], asfloat(best + 1));
    }
}
