// The learned impact model's input features on the GPU (`STRESS_IMPACT_PUNCH=learned`):
// the calls schema's arrays (learned-impact/OUTCOME_CALLS_SCHEMA.md, `/3`) for every call
// of the batch, computed from the frame-start state `learned.slang` gathered, as P1's Rust
// call builder (stress-learned `outcome_calls.rs` `build_call`) defines them. Column
// positions come from a map (`cols`) the host builds from the schema's column lists, so a
// schema change moves columns without touching these kernels; a column these kernels do not
// compute is an error on the host.
//
//   feat_map       per layout bond: its static row (by its chunks' stable keys), the row's
//                  layout bond, and the frame it was first seen broken
//   feat_keys      per call chunk node: its stable key's node; per-node static copies
//   feat_brow      per call bond: (node a, node b, static row, layout bond)
//   feat_contacts  per call: predicted impactor contacts (stress-ref `predict_contacts`),
//                  with the punch's plug per contact
//   feat_scales    per call: c, f_t, E, GA (medians as the builder's)
//   feat_summ      per call chunk node: its pristine bonds' damage summaries
//   feat_graph     per call: hops to contact, wave arrival, hops to anchor
//   feat_ic / feat_cracked / feat_sc  per call: the edge sets (count, then write)
//   feat_prefix    one thread: the edge sets' places in the batch
//   feat_nodes     per call: chunk, impactor and global features, raw node arrays
//   feat_bonds     per call bond: bond features
//
// Frames: the solver's (Z-up); positions world, absolute until `learned_finish` (feat_keys,
// feat_contacts run before it), and `nodes[13]` keeps them after.

import common;
import joint;
import bonds;

static const uint CALL_F4 = 10;
static const uint NODE_F4 = 14;
static const uint BOND_F4 = 2;
static const uint CISL_U4 = 2;
static const uint IMP_F4 = 4;
static const uint SC_F4 = 6;
static const uint SB_F4 = 9;
static const uint CON_F4 = 6;
static const uint WORK_F = 24;
// 64 threads per group: Metal caps a pipeline's threads per group by its registers (a heavy
// kernel's cap can be below 256, and a larger dispatch then runs as unsynchronised parts).
static const uint G = 64;
static const uint NONE = 0xffffffffu;
static const float G0 = 9.81;
static const float TINY = 1e-6;
static const float CONTACT_MARGIN = 0.25;

// work words per node
static const uint W_HOPS_C = 0;
static const uint W_ARRIVAL = 1;
static const uint W_HOPS_A = 2;
static const uint W_PLUG = 3;       // the last contact row whose plug holds it, + 1 (0: none)
static const uint W_CONTACT = 4;    // 1: a contact edge's source
static const uint W_DSUM = 5;
static const uint W_DMAX = 6;
static const uint W_NLIVE = 7;
static const uint W_NBROKEN = 8;
static const uint W_HALF = 9;       // 9..11 half extents
static const uint W_HNORM = 12;
static const uint W_FRIC = 13;
static const uint W_FIXED = 14;     // 1 fixed, 2 pinned
static const uint W_EMOD = 15;     // scaled modulus
static const uint W_ROW = 16;      // stable chunk row (bits)
static const uint W_REST = 17;     // 17..20 rest rotation (x, y, z, w)
static const uint W_CENTER = 21;   // 21..23 centre (structure frame)
// Cracked-face candidates per call (`feat_cracked_find`): CRACKED_PER_NODE per chunk node,
// after the batch's bonds in `bedge`.
static const uint CRACKED_PER_NODE = 8;

// The column map (`cols`): per section, its stride, then for each column these kernels
// compute (in their order, `K_*` below) its index in the section, or NONE.
static const uint S_CHUNK = 0;
static const uint S_IMP = 1;
static const uint S_GLOB = 2;
static const uint S_BOND = 3;
static const uint S_CRACKED = 4;
static const uint S_IC = 5;
static const uint S_SC = 6;
static const uint SECTIONS = 7;
static const uint COLS_MAX = 64;    // per section in the map

struct FeatParams
{
    uint impactor_count;
    uint frame;
    uint layout_bonds;     // bonds in the layout (`feat_map`)
    uint rows;             // static bond rows
    float dt;
    float speed;           // the tiered impact speed
    float restitution;
    float contact_friction; // < 0: none (pairs take the smaller friction)
    float4 gravity;
    uint4 bases;           // feat floats: chunk rows, impactor rows, global rows, bond rows
    uint4 bases2;          // cracked rows, impactor-contact rows, structure-contact rows, raw
    uint4 caps;            // rows (directed) the edge sections hold: cracked, ic, sc; max calls
    uint count_only;       // 1: the edge kernels count, 0: they write
    uint max_nodes;
    uint layout_cap;       // `bmap`'s layout part
    uint max_bonds_cap;    // `bedge`'s bond part (cracked candidates after it)
    uint prev_frame;       // the batch `feat_deform` applies (its key map's frame)
    uint engine_base;      // ctab: per island 3 float4
    uint chunk_count;      // layout chunks
    uint pad1;
};

[[vk::binding(0, 0)]] ConstantBuffer<FeatParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint> ctl;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> calls;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> nodes;
[[vk::binding(4, 0)]] StructuredBuffer<float4> bonds;
[[vk::binding(5, 0)]] StructuredBuffer<uint4> cisl;
// Per call island 2 float4: (centre of mass, mass), (velocity, anchored).
[[vk::binding(6, 0)]] StructuredBuffer<float4> cislf;
// Per stable chunk (SC_F4): (mass, scaled modulus, density, tensile strength), (G_f
// tension, friction, fixed, pinned), (half extents, |half|), (principal inertia,
// plane-strain modulus), (centre (structure frame), -), rest rotation (x, y, z, w).
[[vk::binding(7, 0)]] StructuredBuffer<float4> sc;
// Per stable bond row (SB_F4): (centroid, area), (normal, width0), (t1, width1), (length, kn,
// ks, kb_t1), (kb_t2, kt, Weibull, rebar), strengths 0..3, 4..7 (-1: infinite), (chunk a,
// chunk b (stable chunk rows) bits, structure bits, source bits), (ductile, softening,
// rebar rupture work, -).
[[vk::binding(8, 0)]] StructuredBuffer<float4> sb;
// [0] structures S, [1] stable chunks N; [2..2+S) each structure's first chunk row; then
// N+1 incidence offsets; then the incident bond rows.
[[vk::binding(9, 0)]] StructuredBuffer<uint> sinc;
// Bond rows sorted by (lower chunk row, higher chunk row): (lo, hi, row, -).
[[vk::binding(10, 0)]] StructuredBuffer<uint4> slook;
// [0, layout_cap): layout bond -> row; then per row: its layout bond; then per row: the
// frame it was first seen broken (NONE: not yet).
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> bmap;
// Per stable chunk: (node + 1, frame of the batch).
[[vk::binding(12, 0)]] RWStructuredBuffer<uint2> key_node;
[[vk::binding(13, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(14, 0)]] StructuredBuffer<BondStatic> bond_static;
[[vk::binding(15, 0)]] StructuredBuffer<BondDyn> bond_dyn;
// learned.slang's impactors, IMP_F4: (body, bound, face, mass), (inertia, plane-strain
// modulus), (|half|, half extents), (scaled modulus, friction, face radius, crushable).
[[vk::binding(16, 0)]] StructuredBuffer<float4> imps;
// Per call contact row CON_F4: (impactor bits, entry node bits, call island bits, time),
// (point, speed), (direction, reduced mass), (modulus, face, Hertz time, peak force),
// (plug ratio, plug mass, plug marked, cluster chunks), (valid, -, -, -).
[[vk::binding(17, 0)]] RWStructuredBuffer<float4> contacts;
[[vk::binding(18, 0)]] RWStructuredBuffer<float> work;
[[vk::binding(19, 0)]] RWStructuredBuffer<float> feat;
// Per directed edge row of each type: (src, dst) global node indices; per undirected edge:
// key words (bond: structure, source; contact: the two node keys).
[[vk::binding(20, 0)]] RWStructuredBuffer<uint> edges;
[[vk::binding(21, 0)]] StructuredBuffer<uint> cols;
// Per call bond: (node a, node b, static row, layout bond).
[[vk::binding(22, 0)]] RWStructuredBuffer<uint4> bedge;
// Feature control: [0..4) edge totals (cracked, ic, sc, bonds) directed rows, [8..] per-call
// (cracked, ic, sc) counts and bases.
[[vk::binding(23, 0)]] RWStructuredBuffer<uint> fctl;
// The solver's tables and the engine's bodies (`feat_deform`).
[[vk::binding(24, 0)]] StructuredBuffer<Island> islands;
[[vk::binding(25, 0)]] StructuredBuffer<float4> ctab;
[[vk::binding(26, 0)]] StructuredBuffer<float4> bodies;
[[vk::binding(27, 0)]] RWStructuredBuffer<float4> state;

// ------------------------------------------------------------------ helpers

float4 node(uint n, uint k)
{
    return nodes[NODE_F4 * n + k];
}

uint call_first(uint g) { return asuint(calls[CALL_F4 * g + 2].z); }
uint call_nodes(uint g) { return asuint(calls[CALL_F4 * g + 2].w); }
uint call_chunks(uint g) { return asuint(calls[CALL_F4 * g + 3].w); }
uint call_bfirst(uint g) { return asuint(calls[CALL_F4 * g + 4].x); }
uint call_nbonds(uint g) { return asuint(calls[CALL_F4 * g + 4].y); }
uint call_mask(uint g) { return asuint(calls[CALL_F4 * g + 5].w); }
uint call_e0(uint g) { return asuint(calls[CALL_F4 * g + 6].x); }
uint call_ne(uint g) { return asuint(calls[CALL_F4 * g + 6].y); }
uint call_cbase(uint g) { return asuint(calls[CALL_F4 * g + 6].z); }
uint call_ibase(uint g) { return asuint(calls[CALL_F4 * g + 6].w); }

// A node's index in the network's global order (every call's chunks, then every call's impactors).
uint global_node(uint g, uint j)
{
    uint nc = call_chunks(g);
    if (j < nc)
        return call_cbase(g) + j;
    return fctl[4] + call_ibase(g) + (j - nc);
}

float3x3 quat_matrix(float4 q)
{
    float w = q.w, x = q.x, y = q.y, z = q.z;
    return float3x3(
        float3(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        float3(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        float3(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)));
}

float3 col3(float3x3 m, uint c)
{
    return float3(m[0][c], m[1][c], m[2][c]);
}

float ln_ratio(float a, float b)
{
    return log(max(a, TINY) / max(b, TINY));
}

float cap4(float x)
{
    return x < 3.0e38 ? min(x, 4.0) : 4.0;
}

// A unit tangent ⟂ n from a body axis (outcome_calls.rs `tangent`).
float3 tangent_of(float3 n, float3x3 body)
{
    for (uint a = 0; a < 2; a++)
    {
        float3 axis = col3(body, a);
        float3 t = axis - n * dot(axis, n);
        if (length(t) > 1e-6)
            return normalize(t);
    }
    float3 axis = col3(body, 2);
    return normalize(axis - n * dot(axis, n));
}

// Half-thickness of a box along unit n.
float support_of(float3x3 r, float3 half, float3 n)
{
    float3 l = mul(transpose(r), n);
    return abs(l.x) * half.x + abs(l.y) * half.y + abs(l.z) * half.z;
}

uint col_of(uint section, uint k)
{
    return cols[section * (COLS_MAX + 1) + 1 + k];
}

uint stride_of(uint section)
{
    return cols[section * (COLS_MAX + 1)];
}

void put(uint base, uint section, uint row, uint k, float v)
{
    uint c = col_of(section, k);
    if (c != NONE)
        feat[base + row * stride_of(section) + c] = v;
}

// The 6-D orientation of an edge frame in a body frame: t1 and t2 in body axes.
void put_rot6(uint base, uint section, uint row, uint k0, float3 t1, float3 t2, float3x3 body)
{
    float3 a = mul(transpose(body), t1);
    float3 b = mul(transpose(body), t2);
    put(base, section, row, k0, a.x);
    put(base, section, row, k0 + 1, a.y);
    put(base, section, row, k0 + 2, a.z);
    put(base, section, row, k0 + 3, b.x);
    put(base, section, row, k0 + 4, b.y);
    put(base, section, row, k0 + 5, b.z);
}

uint chunk_row_of(uint layout_chunk)
{
    uint4 info = chunks[layout_chunk].info;
    return sinc[2 + info.z] + info.w;
}

float hertz_duration(float m, float r, float e, float v)
{
    float vv = max(abs(v), 1e-3);
    return 2.87 * pow(m * m / (r * e * e * vv), 0.2);
}

// Time within [0, dt] two bounding spheres first touch (0: overlapping), or -1.
float sphere_contact_time(float3 dx, float3 dv, float r, float dt)
{
    float c = dot(dx, dx) - r * r;
    if (c <= 0.0)
        return 0.0;
    float a = dot(dv, dv);
    float b = dot(dx, dv);
    if (a <= 0.0 || b >= 0.0)
        return -1.0;
    float disc = b * b - a * c;
    if (disc < 0.0)
        return -1.0;
    float t = (-b - sqrt(disc)) / a;
    return t <= dt ? max(t, 0.0) : -1.0;
}

uint bond_row_lookup(uint ga, uint gb)
{
    uint lo_k = min(ga, gb), hi_k = max(ga, gb);
    uint lo = 0, hi = params.rows;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        uint4 e = slook[mid];
        if (e.x < lo_k || (e.x == lo_k && e.y < hi_k))
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo < params.rows && slook[lo].x == lo_k && slook[lo].y == hi_k)
        return slook[lo].z;
    return NONE;
}

uint node_of_key(uint row)
{
    uint2 k = key_node[row];
    return (k.x != 0 && k.y == params.frame) ? k.x - 1 : NONE;
}

// ------------------------------------------------------------------ group sums

groupshared float4 g_red[2][G];
groupshared uint g_flag;
groupshared float4 g_scalars;

float4 group_sum(uint t, float4 a)
{
    // A barrier first: the compiler may sink the last call's read of the result past its
    // trailing barrier, so no thread may overwrite the array until every thread has read it.
    GroupMemoryBarrierWithGroupSync();
    g_red[0][t] = a;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_red[0][t] += g_red[0][t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float4 r = g_red[0][0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float4 group_min(uint t, float4 a)
{
    // A barrier first: the compiler may sink the last call's read of the result past its
    // trailing barrier, so no thread may overwrite the array until every thread has read it.
    GroupMemoryBarrierWithGroupSync();
    g_red[0][t] = a;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_red[0][t] = min(g_red[0][t], g_red[0][t + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float4 r = g_red[0][0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float4 group_max(uint t, float4 a)
{
    // A barrier first: the compiler may sink the last call's read of the result past its
    // trailing barrier, so no thread may overwrite the array until every thread has read it.
    GroupMemoryBarrierWithGroupSync();
    g_red[0][t] = a;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
            g_red[0][t] = max(g_red[0][t], g_red[0][t + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float4 r = g_red[0][0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// (value, index) minimum, ties to the lower index.
groupshared float g_mv[G];
groupshared uint g_mi[G];

uint group_argmin(uint t, float v, uint i)
{
    // A barrier first: the compiler may sink the last call's read of the result past its
    // trailing barrier, so no thread may overwrite the array until every thread has read it.
    GroupMemoryBarrierWithGroupSync();
    g_mv[t] = v;
    g_mi[t] = i;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s && (g_mv[t + s] < g_mv[t] || (g_mv[t + s] == g_mv[t] && g_mi[t + s] < g_mi[t])))
        {
            g_mv[t] = g_mv[t + s];
            g_mi[t] = g_mi[t + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = g_mi[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Inclusive group scan of one uint per thread (fixed tree).
groupshared uint g_scan[G];

uint group_scan(uint t, uint v)
{
    // A barrier first: the compiler may sink the last call's read of the result past its
    // trailing barrier, so no thread may overwrite the array until every thread has read it.
    GroupMemoryBarrierWithGroupSync();
    g_scan[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 1; step < G; step <<= 1)
    {
        uint add = t >= step ? g_scan[t - step] : 0;
        GroupMemoryBarrierWithGroupSync();
        g_scan[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = g_scan[t];
    return r;
}

uint group_total()
{
    uint r = g_scan[G - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// ------------------------------------------------------------------ feat_map

// One thread per layout bond: its static row (by its chunks' stable rows), the row's layout
// bond, and the frame it was first seen disconnected.
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_map(uint3 id: SV_DispatchThreadID)
{
    uint l = id.x;
    if (l >= params.layout_bonds)
        return;
    JointBond law = bond_static[l].law;
    uint row = bond_row_lookup(chunk_row_of(law.ids.y), chunk_row_of(law.ids.z));
    bmap[l] = row;
    if (row == NONE)
        return;
    bmap[params.layout_cap + row] = l;
    bool live = connected(bond_dyn[l].js, law.stiff1.w != 0.0);
    uint at = params.layout_cap + params.rows + row;
    if (!live && bmap[at] == NONE)
        bmap[at] = params.frame;
}

// ------------------------------------------------------------------ feat_keys

// One thread per call node: a chunk's stable row maps to its node (this batch); per-node
// static copies (half extents, friction, supports, plane-strain modulus) for the kernels
// after; per-node words cleared.
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_keys(uint3 id: SV_DispatchThreadID)
{
    uint n = id.x;
    if (n >= ctl[2])
        return;
    for (uint k = 0; k < WORK_F; k++)
        work[WORK_F * n + k] = 0.0;
    if (asuint(node(n, 4).w) != 0)
        return;
    float4 w12 = node(n, 12);
    uint row = sinc[2 + asuint(w12.z)] + asuint(w12.w);
    key_node[row] = uint2(n + 1, params.frame);
    float4 s1 = sc[SC_F4 * row + 1];
    float4 s2 = sc[SC_F4 * row + 2];
    float4 s3 = sc[SC_F4 * row + 3];
    work[WORK_F * n + W_HALF] = s2.x;
    work[WORK_F * n + W_HALF + 1] = s2.y;
    work[WORK_F * n + W_HALF + 2] = s2.z;
    work[WORK_F * n + W_HNORM] = s2.w;
    work[WORK_F * n + W_FRIC] = s1.y;
    work[WORK_F * n + W_FIXED] = s1.z != 0.0 ? 1.0 : (s1.w != 0.0 ? 2.0 : 0.0);
    float4 s0 = sc[SC_F4 * row];
    float4 s4 = sc[SC_F4 * row + 4];
    float4 s5 = sc[SC_F4 * row + 5];
    work[WORK_F * n + W_EMOD] = s0.y;
    work[WORK_F * n + W_ROW] = asfloat(row);
    for (uint k = 0; k < 4; k++)
        work[WORK_F * n + W_REST + k] = s5[k];
    for (uint k = 0; k < 3; k++)
        work[WORK_F * n + W_CENTER + k] = s4[k];
    float4 p13 = node(n, 13);
    nodes[NODE_F4 * n + 13] = float4(p13.xyz, s3.w);
    // the builder's chunk inertia: the static diagonal
    nodes[NODE_F4 * n + 2] = float4(s3.xyz, node(n, 2).w);
}

// ------------------------------------------------------------------ feat_brow

// One thread per call bond (after feat_keys, feat_map): its two nodes, static row and layout
// bond, oriented as the static row's (chunk a, chunk b).
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_brow(uint3 id: SV_DispatchThreadID)
{
    uint b = id.x;
    if (b >= ctl[3])
        return;
    uint l = asuint(bonds[BOND_F4 * b + 1].z);
    uint row = bmap[l];
    uint4 ends = asuint(sb[SB_F4 * row + 7]);
    bedge[b] = uint4(node_of_key(ends.x), node_of_key(ends.y), row, l);
}

// ------------------------------------------------------------------ feat_contacts

groupshared uint g_plug[4096];

// One group per call: for each of its impactors (ascending), each of its islands (call
// island order: the world's cluster order), the predicted contact of stress-ref
// `predict_contacts` (min speed 0): the swept bounds' first time, the struck chunk, the
// punch's plug (the 45° cone's chunks bonded to the struck one), Hertz duration and peak
// force, and the plug's force over capacity. Contact rows in the call's order; the plug's
// chunks marked with the row (the last row wins, as the recorder's).
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_contacts(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (group_uniform(g >= ctl[1] ? 1u : 0u, t) != 0)
        return;
    uint mask = group_uniform(call_mask(g), t);
    uint e0 = group_uniform(call_e0(g), t);
    uint ne = group_uniform(call_ne(g), t);
    uint first = call_first(g);
    uint nc = call_chunks(g);
    // rows: the call's places (impactors × islands), first at calls[7].x
    uint row0 = asuint(calls[CALL_F4 * g + 8].x);
    uint rows = 0;
    float vmax = 0.0;
    for (uint k = 0; k < 32; k++)
    {
        if ((mask & (1u << k)) == 0)
            continue;
        uint rank = countbits(mask & ((1u << k) - 1u));
        uint inode = first + nc + rank;
        float3 c = node(inode, 0).xyz;
        float3 vi = node(inode, 3).xyz;
        float r = imps[IMP_F4 * k + 2].x;
        float face = imps[IMP_F4 * k + 3].z;
        float m_imp = imps[IMP_F4 * k].w;
        float e_imp = imps[IMP_F4 * k + 1].w;
        for (uint e = e0; e < e0 + ne; e++)
        {
            uint4 c0 = cisl[CISL_U4 * e];
            uint4 c1 = cisl[CISL_U4 * e + 1];
            float4 f1 = cislf[2 * e + 1];
            uint at = row0 + rows;
            rows++;
            if (t == 0)
                contacts[CON_F4 * at + 5] = float4(0.0);
            uint n_isl = group_uniform(c0.z, t);
            if (n_isl == 0 || c1.z == 0)
                continue;
            float3 v = vi - f1.xyz;
            float speed = length(v);
            if (group_uniform(speed > 0.0 ? 1u : 0u, t) == 0)
                continue;
            float inf = float_bits(0x7f800000u);
            float4 lo = float4(inf), hi = float4(-inf);
            float msum = 0.0;
            for (uint j = t; j < n_isl; j += G)
            {
                uint n = c0.y + j;
                float3 p = node(n, 0).xyz;
                float h = work[WORK_F * n + W_HNORM];
                lo = min(lo, float4(p - float3(h + r), 0.0));
                hi = max(hi, float4(p + float3(h + r), 0.0));
                msum += node(n, 0).w;
            }
            lo = group_min(t, lo);
            hi = -group_min(t, -hi);
            float cl_mass = group_sum(t, float4(msum, 0, 0, 0)).x;
            float t0 = 0.0, t1 = params.dt;
            for (uint a = 0; a < 3; a++)
            {
                if (abs(v[a]) < 1e-12)
                {
                    if (c[a] < lo[a] || c[a] > hi[a])
                        t0 = inf;
                    continue;
                }
                float ta = (lo[a] - c[a]) / v[a];
                float tb = (hi[a] - c[a]) / v[a];
                t0 = max(t0, min(ta, tb));
                t1 = min(t1, max(ta, tb));
            }
            if (group_uniform(t0 > t1 ? 1u : 0u, t) != 0)
                continue;
            float3 d = v / speed;
            float3 hit = c + v * t0;
            // the struck chunk: the nearest to the path (ties: the lower index)
            float best = inf;
            uint best_j = NONE;
            for (uint j = t; j < n_isl; j += G)
            {
                float3 q = node(c0.y + j, 0).xyz - hit;
                float along = dot(q, d);
                float dist = length(q - d * along) + abs(along) * 1e-3;
                if (dist < best)
                {
                    best = dist;
                    best_j = j;
                }
            }
            uint entry = group_argmin(t, best, best_j);
            uint entry_n = c0.y + entry;
            float entry_depth = dot(node(entry_n, 0).xyz - hit, d);
            // the plug: the cone's chunks bonded to the struck one (label propagation)
            bool small = n_isl <= 4096;
            for (uint j = t; j < min(n_isl, 4096u); j += G)
                g_plug[j] = j == entry ? 1u : 0u;
            GroupMemoryBarrierWithGroupSync();
            for (uint iter = 0; iter < n_isl && small; iter++)
            {
                if (t == 0)
                    g_flag = 0;
                GroupMemoryBarrierWithGroupSync();
                for (uint bj = t; bj < c1.z; bj += G)
                {
                    uint4 be = bedge[c1.y + bj];
                    if (be.x == NONE || be.y == NONE)
                        continue;
                    uint la = be.x - c0.y, lb = be.y - c0.y;
                    if (la >= n_isl || lb >= n_isl || g_plug[la] == g_plug[lb])
                        continue;
                    uint o = g_plug[la] != 0 ? lb : la;
                    float3 q = node(c0.y + o, 0).xyz - hit;
                    float along = dot(q, d);
                    float h = work[WORK_F * (c0.y + o) + W_HNORM];
                    float lateral = length(q - d * along);
                    if (along >= entry_depth - h && lateral <= face + max(along - entry_depth, 0.0))
                    {
                        g_plug[o] = 1;
                        g_flag = 1;
                    }
                }
                GroupMemoryBarrierWithGroupSync();
                if (group_uniform(g_flag, t) == 0)
                    break;
                GroupMemoryBarrierWithGroupSync();
            }
            // plug mass, count, and the capacity of its interface's bonds
            float4 acc = float4(0.0);
            for (uint j = t; j < n_isl; j += G)
                if (g_plug[j] != 0)
                    acc += float4(node(c0.y + j, 0).w, 1.0, 0.0, 0.0);
            for (uint bj = t; bj < c1.z; bj += G)
            {
                uint4 be = bedge[c1.y + bj];
                if (be.x == NONE || be.y == NONE)
                    continue;
                uint la = be.x - c0.y, lb = be.y - c0.y;
                if (la < n_isl && lb < n_isl && (g_plug[la] != 0) != (g_plug[lb] != 0))
                {
                    float4 s0 = sb[SB_F4 * be.z];
                    float4 s4 = sb[SB_F4 * be.z + 4];
                    float4 s5 = sb[SB_F4 * be.z + 5];
                    acc.z += s5.x * s0.w * s4.z;
                }
            }
            acc = group_sum(t, acc);
            float m_plug = acc.x;
            bool has_plug = acc.y != float(n_isl);
            float capacity = acc.z;
            float modulus = 1.0 / (1.0 / e_imp + 1.0 / node(entry_n, 13).w);
            bool anchored = f1.w != 0.0;
            float m_red = anchored ? m_imp : cl_mass * m_imp / (cl_mass + m_imp);
            float hertz = hertz_duration(m_red, face, modulus, speed);
            float peak = 3.14159265359 * (2.0 * m_red * speed) / (2.0 * hertz);
            float ratio = 0.0;
            if (has_plug)
            {
                float m_eff = m_plug * m_imp / (m_plug + m_imp);
                float tau = hertz_duration(m_eff, face, modulus, speed);
                ratio = (m_eff * speed / tau) / max(capacity, 1e-30);
            }
            bool marked = has_plug && speed > params.speed;
            if (marked)
            {
                for (uint j = t; j < n_isl; j += G)
                    if (g_plug[j] != 0)
                        work[WORK_F * (c0.y + j) + W_PLUG] = float(at - row0 + 1);
            }
            if (t == 0)
            {
                contacts[CON_F4 * at] = float4(asfloat(k), asfloat(entry_n), asfloat(e), t0);
                contacts[CON_F4 * at + 1] = float4(hit, speed);
                contacts[CON_F4 * at + 2] = float4(d, m_red);
                contacts[CON_F4 * at + 3] = float4(modulus, face, hertz, peak);
                contacts[CON_F4 * at + 4] = float4(ratio, has_plug ? m_plug : 0.0, marked ? 1.0 : 0.0, float(n_isl));
                contacts[CON_F4 * at + 5] = float4(1.0, 0.0, 0.0, 0.0);
            }
            vmax = max(vmax, speed);
            GroupMemoryBarrierWithGroupSync();
        }
    }
    // the largest contact speed, for the reference speed (`learned_finish`)
    if (t == 0)
        calls[CALL_F4 * g + 8] = float4(calls[CALL_F4 * g + 8].x, asfloat(rows), 0.0, vmax);
}

// ------------------------------------------------------------------ feat_scales

// Sorts work[base .. base + n) ascending in one group (bitonic over the next power of two,
// padded with +inf in work[base + n ..)): the builder's medians.
void sort_work(uint t, uint base, uint n)
{
    uint m = 1;
    while (m < n)
        m <<= 1;
    float inf = float_bits(0x7f800000u);
    for (uint j = n + t; j < m; j += G)
        work[base + j] = inf;
    DeviceMemoryBarrierWithGroupSync();
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
                    float a = work[base + j];
                    float b = work[base + p];
                    if ((a > b) == up)
                    {
                        work[base + j] = b;
                        work[base + p] = a;
                    }
                }
            }
            DeviceMemoryBarrierWithGroupSync();
        }
    }
}

float median_work(uint base, uint n)
{
    uint k = n / 2;
    return (n % 2 == 1) ? work[base + k] : 0.5 * (work[base + k - 1] + work[base + k]);
}

static const uint SCRATCH = 131072; // per region: a call's live bonds or chunks at most

// One group, the calls in turn (one scratch for the sorts): c (the dominant chunk's wave
// speed: the most massive, the last on ties), f_t (median live bond tensile strength), E
// (median chunk modulus), GA (median live bond G_f·A), the median live bond area, and the
// G_f·A sums of the live bonds and of the punch plugs' bonds, as the builder's.
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_scales(uint3 thread: SV_GroupThreadID)
{
    uint t = thread.x;
    uint ncalls = group_uniform(ctl[1], t);
    uint base = WORK_F * params.max_nodes;
    for (uint g = 0; g < ncalls; g++)
    {
        uint first = call_first(g);
        uint nc = group_uniform(call_chunks(g), t);
        uint bfirst = call_bfirst(g);
        uint nb = group_uniform(call_nbonds(g), t);
        float L = calls[CALL_F4 * g].x;
        float best = -1.0;
        uint best_j = 0;
        for (uint j = t; j < nc; j += G)
        {
            float m = node(first + j, 0).w;
            if (m >= best)
            {
                best = m;
                best_j = j;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        g_mv[t] = best;
        g_mi[t] = best_j;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = G / 2; s > 0; s >>= 1)
        {
            if (t < s && (g_mv[t + s] > g_mv[t] || (g_mv[t + s] == g_mv[t] && g_mi[t + s] > g_mi[t])))
            {
                g_mv[t] = g_mv[t + s];
                g_mi[t] = g_mi[t + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        uint dom = first + g_mi[0];
        GroupMemoryBarrierWithGroupSync();
        float4 w12 = node(dom, 12);
        uint drow = sinc[2 + asuint(w12.z)] + asuint(w12.w);
        float4 d0 = sc[SC_F4 * drow];
        float4 d1 = sc[SC_F4 * drow + 1];
        float c_wave = sqrt(d0.y / d0.z);
        // the live bonds' tensile strengths, G_f·A and areas; the G_f·A sums
        uint carry = 0;
        float4 ga = float4(0.0);
        for (uint b0 = 0; b0 < nb; b0 += G)
        {
            uint j = b0 + t;
            uint4 be = j < nb ? bedge[bfirst + j] : uint4(NONE);
            uint ok = j < nb && be.x != NONE && be.y != NONE ? 1u : 0u;
            uint at = carry + group_scan(t, ok) - ok;
            if (ok != 0)
            {
                float area = sb[SB_F4 * be.z].w;
                float gab = sb[SB_F4 * be.z + 6].y * area;
                work[base + at] = sb[SB_F4 * be.z + 5].x;
                work[base + SCRATCH + at] = gab;
                work[base + 2 * SCRATCH + at] = area;
                ga.x += gab;
                bool pa = work[WORK_F * be.x + W_PLUG] != 0.0, pb = work[WORK_F * be.y + W_PLUG] != 0.0;
                if (pa != pb)
                    ga.y += gab;
            }
            carry += group_total();
        }
        uint nlive = group_uniform(carry, t);
        ga = group_sum(t, ga);
        DeviceMemoryBarrierWithGroupSync();
        float f_t = d0.w;
        float GA = d1.x * L * L;
        float med_area = L * L;
        if (nlive > 0)
        {
            sort_work(t, base, nlive);
            sort_work(t, base + SCRATCH, nlive);
            sort_work(t, base + 2 * SCRATCH, nlive);
            // kept in their own words before the scratch is reused (see learned.slang)
            if (t == 0)
                g_scalars = float4(median_work(base, nlive), median_work(base + SCRATCH, nlive), median_work(base + 2 * SCRATCH, nlive), 0.0);
            DeviceMemoryBarrierWithGroupSync();
            f_t = g_scalars.x;
            GA = g_scalars.y;
            med_area = g_scalars.z;
            DeviceMemoryBarrierWithGroupSync();
        }
        for (uint j = t; j < nc; j += G)
        {
            float4 w = node(first + j, 12);
            work[base + j] = sc[SC_F4 * (sinc[2 + asuint(w.z)] + asuint(w.w))].y;
        }
        DeviceMemoryBarrierWithGroupSync();
        sort_work(t, base, nc);
        if (t == 0)
            g_scalars.w = median_work(base, nc);
        DeviceMemoryBarrierWithGroupSync();
        float e_med = g_scalars.w;
        DeviceMemoryBarrierWithGroupSync();
        if (t == 0)
        {
            float4 s0 = calls[CALL_F4 * g];
            float4 s1 = calls[CALL_F4 * g + 1];
            calls[CALL_F4 * g] = float4(s0.x, s0.y, s0.z, c_wave);
            calls[CALL_F4 * g + 1] = float4(f_t, e_med, GA, s1.w);
            calls[CALL_F4 * g + 9] = float4(med_area, ga.x, ga.y, 0.0);
        }
        DeviceMemoryBarrierWithGroupSync();
    }
}

// ------------------------------------------------------------------ feat_summ

// One thread per call chunk node: over every pristine bond of its chunk, the live ones'
// damage mean and maximum and their count, and the broken ones' count.
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_summ(uint3 id: SV_DispatchThreadID)
{
    uint n = id.x;
    if (n >= ctl[2] || asuint(node(n, 4).w) != 0)
        return;
    float4 w12 = node(n, 12);
    uint S = sinc[0], N = sinc[1];
    uint row = sinc[2 + asuint(w12.z)] + asuint(w12.w);
    uint off = 2 + S;
    uint a = sinc[off + row], b = sinc[off + row + 1];
    float dsum = 0.0, dmax = 0.0, nlive = 0.0, nbroken = 0.0;
    for (uint i = a; i < b; i++)
    {
        uint br = sinc[off + N + 1 + i];
        uint l = bmap[params.layout_cap + br];
        BondDyn dyn = bond_dyn[l];
        if (connected(dyn.js, bond_static[l].law.stiff1.w != 0.0))
        {
            nlive += 1.0;
            dsum += dyn.js.damage;
            dmax = max(dmax, dyn.js.damage);
        }
        else
        {
            nbroken += 1.0;
        }
    }
    work[WORK_F * n + W_DSUM] = dsum;
    work[WORK_F * n + W_DMAX] = dmax;
    work[WORK_F * n + W_NLIVE] = nlive;
    work[WORK_F * n + W_NBROKEN] = nbroken;
}

// ------------------------------------------------------------------ feat_graph

// One group per call: over its live bonds, bond-graph hops from the contacts' struck chunks
// and from the anchored chunks, and the wave's arrival (contact time + geodesic / c), by
// relaxation to the fixed point (the same sums a breadth-first search and Dijkstra make).
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_graph(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (group_uniform(g >= ctl[1] ? 1u : 0u, t) != 0)
        return;
    uint first = call_first(g);
    uint nc = group_uniform(call_chunks(g), t);
    uint bfirst = call_bfirst(g);
    uint nb = group_uniform(call_nbonds(g), t);
    uint row0 = asuint(calls[CALL_F4 * g + 8].x);
    uint rows = group_uniform(asuint(calls[CALL_F4 * g + 8].y), t);
    float c_wave = calls[CALL_F4 * g].w;
    float inf = float_bits(0x7f800000u);
    float dt = params.dt;
    for (uint j = t; j < nc; j += G)
    {
        uint n = first + j;
        work[WORK_F * n + W_HOPS_C] = inf;
        work[WORK_F * n + W_ARRIVAL] = inf;
        work[WORK_F * n + W_HOPS_A] = work[WORK_F * n + W_FIXED] != 0.0 ? 0.0 : inf;
    }
    DeviceMemoryBarrierWithGroupSync();
    if (t == 0)
    {
        for (uint r = 0; r < rows; r++)
        {
            float4 c0 = contacts[CON_F4 * (row0 + r)];
            if (contacts[CON_F4 * (row0 + r) + 5].x == 0.0)
                continue;
            uint en = asuint(c0.y);
            uint k = asuint(c0.x);
            work[WORK_F * en + W_HOPS_C] = 0.0;
            uint inode = first + nc + countbits(call_mask(g) & ((1u << k) - 1u));
            float r_imp = imps[IMP_F4 * k + 2].x;
            float tc = sphere_contact_time(node(en, 13).xyz - node(inode, 13).xyz, node(en, 3).xyz - node(inode, 3).xyz, r_imp + work[WORK_F * en + W_HNORM], dt);
            if (tc >= 0.0)
                work[WORK_F * en + W_ARRIVAL] = min(work[WORK_F * en + W_ARRIVAL], tc);
        }
    }
    DeviceMemoryBarrierWithGroupSync();
    for (uint iter = 0; iter < nc + 1; iter++)
    {
        if (t == 0)
            g_flag = 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint j = t; j < nb; j += G)
        {
            uint4 be = bedge[bfirst + j];
            if (be.x == NONE || be.y == NONE)
                continue;
            for (uint side = 0; side < 2; side++)
            {
                uint u = side == 0 ? be.x : be.y;
                uint w = side == 0 ? be.y : be.x;
                float hu = work[WORK_F * u + W_HOPS_C] + 1.0;
                if (hu < work[WORK_F * w + W_HOPS_C])
                {
                    work[WORK_F * w + W_HOPS_C] = hu;
                    g_flag = 1;
                }
                float au = work[WORK_F * u + W_HOPS_A] + 1.0;
                if (au < work[WORK_F * w + W_HOPS_A])
                {
                    work[WORK_F * w + W_HOPS_A] = au;
                    g_flag = 1;
                }
                float cost = length(node(u, 13).xyz - node(w, 13).xyz) / c_wave;
                float tu = work[WORK_F * u + W_ARRIVAL] + cost;
                if (tu < work[WORK_F * w + W_ARRIVAL])
                {
                    work[WORK_F * w + W_ARRIVAL] = tu;
                    g_flag = 1;
                }
            }
        }
        DeviceMemoryBarrierWithGroupSync();
        if (group_uniform(g_flag, t) == 0)
            break;
        GroupMemoryBarrierWithGroupSync();
    }
}

// ------------------------------------------------------------------ edges

// Writes one directed edge row's endpoints (global node order).
void put_edge(uint section, uint row, uint src, uint dst)
{
    uint base = section == S_CRACKED ? 0u : (section == S_IC ? 2u * params.caps.x : 2u * (params.caps.x + params.caps.y));
    edges[base + 2 * row] = src;
    edges[base + 2 * row + 1] = dst;
}

void put_key(uint section, uint k, uint a, uint b)
{
    uint base = 2u * 2u * (params.caps.x + params.caps.y + params.caps.z) + (section == S_CRACKED ? 0u : (section == S_IC ? params.caps.x : params.caps.x + params.caps.y));
    edges[base + 2 * k] = a;
    edges[base + 2 * k + 1] = b;
}

// The per-call places of each edge set (`feat_prefix`): fctl[8 + 6 g ..] counts, bases.
uint edge_base(uint g, uint s)
{
    return fctl[8 + 6 * g + 3 + s];
}

void edge_count(uint g, uint s, uint n)
{
    fctl[8 + 6 * g + s] = n;
}

// One group per call: impactor-contact edges, one per valid contact row (twins: the
// impactor's side first).
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_ic(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (g >= ctl[1] || t != 0)
        return;
    uint first = call_first(g);
    uint nc = call_chunks(g);
    uint row0 = asuint(calls[CALL_F4 * g + 8].x);
    uint rows = asuint(calls[CALL_F4 * g + 8].y);
    float4 s0 = calls[CALL_F4 * g];
    float4 s1 = calls[CALL_F4 * g + 1];
    float L = s0.x, vref = s0.z, c_wave = s0.w, f_t = s1.x;
    uint count = 0;
    uint basek = params.count_only != 0 ? 0u : edge_base(g, 1);
    uint fb = params.bases2.y;
    // the median live bond area (the Hertz force's scale): from the call's scales record
    float med_area = calls[CALL_F4 * g + 9].x;
    for (uint r = 0; r < rows; r++)
    {
        uint at = row0 + r;
        if (contacts[CON_F4 * at + 5].x == 0.0)
            continue;
        if (params.count_only == 0)
        {
            float4 c0 = contacts[CON_F4 * at];
            float4 c1 = contacts[CON_F4 * at + 1];
            float4 c2 = contacts[CON_F4 * at + 2];
            float4 c3 = contacts[CON_F4 * at + 3];
            float4 c4 = contacts[CON_F4 * at + 4];
            uint k = asuint(c0.x);
            uint en = asuint(c0.y);
            uint inode = first + nc + countbits(call_mask(g) & ((1u << k) - 1u));
            float3x3 ri = quat_matrix(node(inode, 1));
            float3x3 rb = quat_matrix(node(en, 1));
            float3 nrm = normalize(c2.xyz);
            float3 t1 = tangent_of(nrm, ri);
            float3 t2 = cross(nrm, t1);
            float3 half_c = float3(work[WORK_F * en + W_HALF], work[WORK_F * en + W_HALF + 1], work[WORK_F * en + W_HALF + 2]);
            // the impactor's half extents: its box (a sphere's: its radius on every axis)
            float hn = imps[IMP_F4 * k + 2].x;
            float3 half_i = float3(imps[IMP_F4 * k + 2].y, imps[IMP_F4 * k + 2].z, imps[IMP_F4 * k + 2].w);
            float gap = dot(node(en, 13).xyz - node(inode, 13).xyz, nrm) - support_of(ri, half_i, nrm) - support_of(rb, half_c, nrm);
            // E_k of the contact: ½ reduced mass × closing speed²
            float ek = 0.5 * c2.w * c1.w * c1.w;
            uint plug_row = uint(work[WORK_F * en + W_PLUG]);
            float ga_plug = calls[CALL_F4 * g + 9].z;
            float ler = (ga_plug > 0.0 && plug_row != 0) ? ln_ratio(ek, ga_plug) : 0.0;
            float fr_imp = imps[IMP_F4 * k + 3].y;
            float fr_ch = work[WORK_F * en + W_FRIC];
            float fric = params.contact_friction >= 0.0 ? params.contact_friction : min(fr_imp, fr_ch);
            float r_face = imps[IMP_F4 * k + 3].z;
            float tc = sphere_contact_time(node(en, 13).xyz - node(inode, 13).xyz, node(en, 3).xyz - node(inode, 3).xyz, hn + work[WORK_F * en + W_HNORM], params.dt);
            float t_con = tc < 0.0 ? 1.0 : (tc <= 0.0 ? -1.0 : tc / params.dt);
            float e_imp = imps[IMP_F4 * k + 3].x;
            float e_ch = work[WORK_F * en + W_EMOD];
            for (uint side = 0; side < 2; side++)
            {
                uint row = basek + count + side;
                uint sn = side == 0 ? inode : en;
                uint dn = side == 0 ? en : inode;
                float3 ft1 = side == 0 ? t1 : -t1;
                float3 ft2 = t2;
                float3 fn = side == 0 ? nrm : -nrm;
                float3 rel = node(sn, 3).xyz - node(dn, 3).xyz;
                float3 rl = float3(dot(ft1, rel), dot(ft2, rel), dot(fn, rel));
                float3 wi = node(inode, 4).xyz * (r_face / vref);
                float3 spin = float3(dot(ft1, wi), dot(ft2, wi), dot(fn, wi));
                uint gs = global_node(g, sn - first), gd = global_node(g, dn - first);
                put_edge(S_IC, row, gs, gd);
                put(fb, S_IC, row, 0, rl.z / vref);
                put(fb, S_IC, row, 1, rl.x / vref);
                put(fb, S_IC, row, 2, rl.y / vref);
                put(fb, S_IC, row, 3, rl.z / c_wave);
                put(fb, S_IC, row, 4, ln_ratio(node(dn, 0).w, node(sn, 0).w));
                put(fb, S_IC, row, 5, ln_ratio(r_face, L));
                put(fb, S_IC, row, 6, ln_ratio(e_imp, e_ch));
                put(fb, S_IC, row, 7, spin.x);
                put(fb, S_IC, row, 8, spin.y);
                put(fb, S_IC, row, 9, spin.z);
                put(fb, S_IC, row, 10, t_con);
                put(fb, S_IC, row, 11, ler);
                put(fb, S_IC, row, 12, ln_ratio(c3.w, f_t * med_area));
                put(fb, S_IC, row, 13, c3.z / params.dt);
                put(fb, S_IC, row, 14, side == 0 ? 1.0 : 0.0);
                put(fb, S_IC, row, 15, max(-gap, 0.0) / L);
                put(fb, S_IC, row, 16, fric);
                put(fb, S_IC, row, 17, params.restitution);
                put_rot6(fb, S_IC, row, 18, ft1, ft2, quat_matrix(node(sn, 1)));
                put_rot6(fb, S_IC, row, 24, ft1, ft2, quat_matrix(node(dn, 1)));
                work[WORK_F * sn + W_CONTACT] = 1.0;
            }
            put_key(S_IC, (basek + count) / 2, asuint(node(en, 12).z), asuint(node(en, 12).w) | (k << 24));
        }
        count += 2;
    }
    if (params.count_only != 0)
        edge_count(g, 1, count);
}

// One group per call (one thread): the broken bonds between two of its chunks, each once
// (from its chunk a's node, in node then incidence order): (row, node a, node b, frame first
// seen broken) into the call's candidate slots (CRACKED_PER_NODE per chunk node, after the
// batch's bonds in `bedge`), after a header (count, overflow).
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_cracked_find(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (g >= ctl[1] || t != 0)
        return;
    uint first = call_first(g);
    uint nc = call_chunks(g);
    uint S = sinc[0], N = sinc[1];
    uint off = 2 + S;
    // the call's region: a header (count, overflow), then the candidates
    uint slots = params.max_bonds_cap + CRACKED_PER_NODE * first;
    uint cap = CRACKED_PER_NODE * nc - 1;
    uint count = 0;
    for (uint j = 0; j < nc; j++)
    {
        uint na = first + j;
        uint rowa = sinc[2 + asuint(node(na, 12).z)] + asuint(node(na, 12).w);
        for (uint i = sinc[off + rowa]; i < sinc[off + rowa + 1]; i++)
        {
            uint br = sinc[off + N + 1 + i];
            uint4 ends = asuint(sb[SB_F4 * br + 7]);
            if (ends.x != rowa)
                continue;
            uint nb = node_of_key(ends.y);
            if (nb == NONE || nb < first || nb >= first + nc)
                continue;
            uint l = bmap[params.layout_cap + br];
            if (connected(bond_dyn[l].js, bond_static[l].law.stiff1.w != 0.0))
                continue;
            if (count < cap)
                bedge[slots + 1 + count] = uint4(br, na, nb, bmap[params.layout_cap + params.rows + br]);
            count++;
        }
    }
    bedge[slots] = uint4(min(count, cap), count > cap ? 1u : 0u, 0, 0);
}

// One group per call (one thread): cracked edges, the candidates whose faces are still
// within the contact margin and overlap across the face (counting, or writing).
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_cracked(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (g >= ctl[1] || t != 0)
        return;
    uint first = call_first(g);
    float4 s0 = calls[CALL_F4 * g];
    float L = s0.x, vref = s0.z;
    uint count = 0;
    uint basek = params.count_only != 0 ? 0u : edge_base(g, 0);
    uint fb = params.bases2.x;
    uint slots = params.max_bonds_cap + CRACKED_PER_NODE * first;
    uint n_cand = bedge[slots].x;
    for (uint ci = 0; ci < n_cand; ci++)
    {
        uint4 cand = bedge[slots + 1 + ci];
        uint br = cand.x, na = cand.y, nb = cand.z, stamp = cand.w;
        float3x3 ra = quat_matrix(node(na, 1));
        float3x3 rbm = quat_matrix(node(nb, 1));
        float4 rest = float4(work[WORK_F * na + W_REST], work[WORK_F * na + W_REST + 1], work[WORK_F * na + W_REST + 2], work[WORK_F * na + W_REST + 3]);
        float3x3 rs = mul(ra, transpose(quat_matrix(rest)));
        float4 g0 = sb[SB_F4 * br];
        float4 g1 = sb[SB_F4 * br + 1];
        float4 g2 = sb[SB_F4 * br + 2];
        uint4 ends = asuint(sb[SB_F4 * br + 7]);
        float3 nrm = mul(rs, g1.xyz);
        float3 t1 = mul(rs, g2.xyz);
        float3 xa = node(na, 13).xyz, xb = node(nb, 13).xyz;
        if (dot(nrm, xb - xa) < 0.0)
        {
            nrm = -nrm;
            t1 = -t1;
        }
        float3 ha = float3(work[WORK_F * na + W_HALF], work[WORK_F * na + W_HALF + 1], work[WORK_F * na + W_HALF + 2]);
        float3 hb = float3(work[WORK_F * nb + W_HALF], work[WORK_F * nb + W_HALF + 1], work[WORK_F * nb + W_HALF + 2]);
        float3 d = xb - xa;
        float gap = dot(d, nrm) - support_of(ra, ha, nrm) - support_of(rbm, hb, nrm);
        float3 t2 = cross(nrm, t1);
        bool across1 = abs(dot(d, t1)) <= support_of(ra, ha, t1) + support_of(rbm, hb, t1);
        bool across2 = abs(dot(d, t2)) <= support_of(ra, ha, t2) + support_of(rbm, hb, t2);
        if (gap > CONTACT_MARGIN * L || !across1 || !across2)
            continue;
        if (params.count_only == 0)
        {
            float3 center_a = float3(work[WORK_F * na + W_CENTER], work[WORK_F * na + W_CENTER + 1], work[WORK_F * na + W_CENTER + 2]);
            float3 p = xa + mul(rs, g0.xyz - center_a);
            float since = stamp == NONE ? 1.0 : min(float(params.frame - stamp) / 8.0, 1.0);
            float fa = work[WORK_F * na + W_FRIC], fbr = work[WORK_F * nb + W_FRIC];
            float fric = params.contact_friction >= 0.0 ? params.contact_friction : min(fa, fbr);
            for (uint side = 0; side < 2; side++)
            {
                uint row = basek + count + side;
                uint sn = side == 0 ? na : nb;
                uint dn = side == 0 ? nb : na;
                float3 ft1 = side == 0 ? t1 : -t1;
                float3 fn = side == 0 ? nrm : -nrm;
                float3 vs = node(sn, 3).xyz + cross(node(sn, 4).xyz, p - node(sn, 13).xyz);
                float3 vd = node(dn, 3).xyz + cross(node(dn, 4).xyz, p - node(dn, 13).xyz);
                float3 q = p - node(sn, 13).xyz;
                float3 dv = vd - vs;
                put_edge(S_CRACKED, row, global_node(g, sn - first), global_node(g, dn - first));
                put(fb, S_CRACKED, row, 0, ln_ratio(g0.w, L * L));
                put(fb, S_CRACKED, row, 1, dot(ft1, q) / L);
                put(fb, S_CRACKED, row, 2, dot(t2, q) / L);
                put(fb, S_CRACKED, row, 3, dot(fn, q) / L);
                put(fb, S_CRACKED, row, 4, gap / L);
                put(fb, S_CRACKED, row, 5, fric);
                put(fb, S_CRACKED, row, 6, dot(ft1, dv) / vref);
                put(fb, S_CRACKED, row, 7, dot(t2, dv) / vref);
                put(fb, S_CRACKED, row, 8, dot(fn, dv) / vref);
                put(fb, S_CRACKED, row, 9, since);
                put_rot6(fb, S_CRACKED, row, 10, ft1, t2, quat_matrix(node(sn, 1)));
                put_rot6(fb, S_CRACKED, row, 16, ft1, t2, quat_matrix(node(dn, 1)));
                work[WORK_F * sn + W_CONTACT] = 1.0;
            }
            put_key(S_CRACKED, (basek + count) / 2, ends.z, ends.w);
        }
        count += 2;
    }
    if (params.count_only != 0)
        edge_count(g, 0, count);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void feat_sc(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (g >= ctl[1] || t != 0)
        return;
    uint first = call_first(g);
    uint nc = call_chunks(g);
    uint e0 = call_e0(g), ne = call_ne(g);
    float4 s0 = calls[CALL_F4 * g];
    float L = s0.x, vref = s0.z, c_wave = s0.w;
    uint count = 0;
    uint basek = params.count_only != 0 ? 0u : edge_base(g, 2);
    uint fb = params.bases2.z;
    // pairs of chunks of different call islands, each once (the lower stable row first)
    for (uint ea = e0; ea < e0 + ne; ea++)
    {
        uint4 ca = cisl[CISL_U4 * ea];
        for (uint ja = 0; ja < ca.z; ja++)
        {
            uint na = ca.y + ja;
            uint ra = asuint(work[WORK_F * na + W_ROW]);
            float3 xa = node(na, 13).xyz;
            float hna = work[WORK_F * na + W_HNORM];
            for (uint e = e0; e < e0 + ne; e++)
            {
                if (e == ea)
                    continue;
                uint4 c0 = cisl[CISL_U4 * e];
                for (uint j = 0; j < c0.z; j++)
                {
                    uint nb = c0.y + j;
                    uint rb = asuint(work[WORK_F * nb + W_ROW]);
                    if (rb <= ra)
                        continue;
                    float3 xb = node(nb, 13).xyz;
                    float3 d = xb - xa;
                    if (!(length(d) < hna + work[WORK_F * nb + W_HNORM]))
                        continue;
                    if (bond_row_lookup(ra, rb) != NONE)
                        continue;
                    if (length(d) < 1e-12)
                        continue;
                    float3 nrm = normalize(d);
                    float3x3 rma = quat_matrix(node(na, 1));
                    float3x3 rmb = quat_matrix(node(nb, 1));
                    float3 ha = float3(work[WORK_F * na + W_HALF], work[WORK_F * na + W_HALF + 1], work[WORK_F * na + W_HALF + 2]);
                    float3 hb = float3(work[WORK_F * nb + W_HALF], work[WORK_F * nb + W_HALF + 1], work[WORK_F * nb + W_HALF + 2]);
                    float gap = length(d) - support_of(rma, ha, nrm) - support_of(rmb, hb, nrm);
                    if (gap > CONTACT_MARGIN * L)
                        continue;
                    if (params.count_only == 0)
                    {
                        float3 t1 = tangent_of(nrm, rma);
                        float3 t2 = cross(nrm, t1);
                        float fa = work[WORK_F * na + W_FRIC], fbr = work[WORK_F * nb + W_FRIC];
                        float fric = params.contact_friction >= 0.0 ? params.contact_friction : min(fa, fbr);
                        uint sa = asuint(node(na, 12).z), sb2 = asuint(node(nb, 12).z);
                        for (uint side = 0; side < 2; side++)
                        {
                            uint row = basek + count + side;
                            uint sn = side == 0 ? na : nb;
                            uint dn = side == 0 ? nb : na;
                            float3 ft1 = side == 0 ? t1 : -t1;
                            float3 fn = side == 0 ? nrm : -nrm;
                            float3 rel = node(sn, 3).xyz - node(dn, 3).xyz;
                            float3 rl = float3(dot(ft1, rel), dot(t2, rel), dot(fn, rel));
                            float t_con = gap <= 0.0 ? -1.0 : ((rl.z > 0.0 && gap / rl.z < params.dt) ? gap / rl.z / params.dt : 1.0);
                            put_edge(S_SC, row, global_node(g, sn - first), global_node(g, dn - first));
                            put(fb, S_SC, row, 0, rl.z / vref);
                            put(fb, S_SC, row, 1, rl.x / vref);
                            put(fb, S_SC, row, 2, rl.y / vref);
                            put(fb, S_SC, row, 3, rl.z / c_wave);
                            put(fb, S_SC, row, 4, ln_ratio(node(dn, 0).w, node(sn, 0).w));
                            put(fb, S_SC, row, 5, t_con);
                            put(fb, S_SC, row, 6, max(-gap, 0.0) / L);
                            put(fb, S_SC, row, 7, fric);
                            put(fb, S_SC, row, 8, 0.0);
                            put(fb, S_SC, row, 9, sa == sb2 ? 1.0 : 0.0);
                            put_rot6(fb, S_SC, row, 10, ft1, t2, quat_matrix(node(sn, 1)));
                            put_rot6(fb, S_SC, row, 16, ft1, t2, quat_matrix(node(dn, 1)));
                            work[WORK_F * sn + W_CONTACT] = 1.0;
                        }
                        put_key(S_SC, (basek + count) / 2, ra, rb);
                    }
                    count += 2;
                }
            }
        }
    }
    if (params.count_only != 0)
        edge_count(g, 2, count);
}

// One thread: each edge set's per-call bases (calls in order) and totals; capacities checked
// (fctl[3] bits: 1 cracked, 2 impactor contacts, 4 structure contacts over their capacity).
[shader("compute")]
[numthreads(1, 1, 1)]
void feat_prefix()
{
    uint n = ctl[1];
    uint tot0 = 0, tot1 = 0, tot2 = 0;
    for (uint g = 0; g < n; g++)
    {
        fctl[8 + 6 * g + 3] = tot0;
        fctl[8 + 6 * g + 4] = tot1;
        fctl[8 + 6 * g + 5] = tot2;
        tot0 += fctl[8 + 6 * g];
        tot1 += fctl[8 + 6 * g + 1];
        tot2 += fctl[8 + 6 * g + 2];
    }
    uint over = 0;
    if (tot0 > 2 * params.caps.x)
        over |= 1;
    if (tot1 > 2 * params.caps.y)
        over |= 2;
    if (tot2 > 2 * params.caps.z)
        over |= 4;
    // [5]: a call's broken bonds overflowed its cracked-face candidates
    uint cand_over = 0;
    for (uint g = 0; g < n; g++)
        cand_over |= bedge[params.max_bonds_cap + CRACKED_PER_NODE * call_first(g)].y;
    fctl[5] = cand_over;
    fctl[0] = tot0;
    fctl[1] = tot1;
    fctl[2] = tot2;
    fctl[3] = over;
    // [4]: chunk nodes in the batch (impactors follow them in the global order)
    uint chunks_n = 0;
    for (uint g = 0; g < n; g++)
        chunks_n += call_chunks(g);
    fctl[4] = chunks_n;
}

// ------------------------------------------------------------------ feat_nodes

// One group per call: chunk features, impactor features, the global node, and the raw node
// arrays (positions relative to the call's centre of mass, w-first orientations).
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_nodes(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint g = group.x;
    uint t = thread.x;
    if (group_uniform(g >= ctl[1] ? 1u : 0u, t) != 0)
        return;
    uint first = call_first(g);
    uint n = group_uniform(call_nodes(g), t);
    uint nc = group_uniform(call_chunks(g), t);
    uint bfirst = call_bfirst(g);
    uint nb = group_uniform(call_nbonds(g), t);
    float4 s0 = calls[CALL_F4 * g];
    float4 s1 = calls[CALL_F4 * g + 1];
    float L = s0.x, m_med = s0.y, vref = s0.z, c_wave = s0.w;
    float e_med = s1.y;
    float dt = params.dt;
    float3 grav = params.gravity.xyz;
    // the call's velocity centre (chunks and impactors), the island's mass
    float4 a = float4(0.0), b = float4(0.0);
    for (uint j = t; j < n; j += G)
    {
        float m = node(first + j, 0).w;
        a += float4(node(first + j, 3).xyz * m, m);
        b += float4(j < nc ? m : 0.0, j < nc && node(first + j, 2).w != 0.0 ? 1.0 : 0.0, 0.0, 0.0);
    }
    a = group_sum(t, a);
    b = group_sum(t, b);
    float3 vcom = a.xyz / a.w;
    float m_island = b.x;
    bool anchored = b.y > 0.0;
    float4 ga = float4(calls[CALL_F4 * g + 9].y, calls[CALL_F4 * g + 9].z, 0.0, 0.0);
    uint fbc = params.bases.x, fbi = params.bases.y, fbg = params.bases.z;
    uint raw = params.bases2.w;
    float ke_imp = 0.0;
    for (uint j = t; j < n; j += G)
    {
        uint k = first + j;
        float3x3 rb = quat_matrix(node(k, 1));
        float3x3 bt = transpose(rb);
        float m = node(k, 0).w;
        float3 gb = mul(bt, grav) / G0;
        float4 q = node(k, 1);
        uint gn = global_node(g, j);
        // raw: position (relative to the call's centre of mass), orientation (w, x, y, z)
        feat[raw + 7 * gn] = node(k, 0).x;
        feat[raw + 7 * gn + 1] = node(k, 0).y;
        feat[raw + 7 * gn + 2] = node(k, 0).z;
        feat[raw + 7 * gn + 3] = q.w;
        feat[raw + 7 * gn + 4] = q.x;
        feat[raw + 7 * gn + 5] = q.y;
        feat[raw + 7 * gn + 6] = q.z;
        if (j < nc)
        {
            uint row = call_cbase(g) + j;
            float3 inr = node(k, 2).xyz;
            float3 half = float3(work[WORK_F * k + W_HALF], work[WORK_F * k + W_HALF + 1], work[WORK_F * k + W_HALF + 2]);
            float fixed = work[WORK_F * k + W_FIXED];
            float nlive = work[WORK_F * k + W_NLIVE];
            float3 vr = mul(bt, node(k, 3).xyz - vcom) / vref;
            float3 wb = mul(bt, node(k, 4).xyz) * (L / vref);
            put(fbc, S_CHUNK, row, 0, ln_ratio(m, m_med));
            put(fbc, S_CHUNK, row, 1, inr.x / (m * L * L));
            put(fbc, S_CHUNK, row, 2, inr.y / (m * L * L));
            put(fbc, S_CHUNK, row, 3, inr.z / (m * L * L));
            put(fbc, S_CHUNK, row, 4, half.x / L);
            put(fbc, S_CHUNK, row, 5, half.y / L);
            put(fbc, S_CHUNK, row, 6, half.z / L);
            put(fbc, S_CHUNK, row, 7, fixed == 1.0 ? 1.0 : 0.0);
            put(fbc, S_CHUNK, row, 8, fixed == 2.0 ? 1.0 : 0.0);
            put(fbc, S_CHUNK, row, 9, nlive > 0.0 ? work[WORK_F * k + W_DSUM] / nlive : 0.0);
            put(fbc, S_CHUNK, row, 10, work[WORK_F * k + W_DMAX]);
            put(fbc, S_CHUNK, row, 11, nlive / 6.0);
            put(fbc, S_CHUNK, row, 12, work[WORK_F * k + W_NBROKEN] / 6.0);
            put(fbc, S_CHUNK, row, 13, cap4(work[WORK_F * k + W_HOPS_C] / 16.0));
            put(fbc, S_CHUNK, row, 14, cap4(work[WORK_F * k + W_ARRIVAL] / dt));
            put(fbc, S_CHUNK, row, 15, cap4(work[WORK_F * k + W_HOPS_A] / 16.0));
            put(fbc, S_CHUNK, row, 16, gb.x);
            put(fbc, S_CHUNK, row, 17, gb.y);
            put(fbc, S_CHUNK, row, 18, gb.z);
            put(fbc, S_CHUNK, row, 19, vr.x);
            put(fbc, S_CHUNK, row, 20, vr.y);
            put(fbc, S_CHUNK, row, 21, vr.z);
            put(fbc, S_CHUNK, row, 22, wb.x);
            put(fbc, S_CHUNK, row, 23, wb.y);
            put(fbc, S_CHUNK, row, 24, wb.z);
            put(fbc, S_CHUNK, row, 25, work[WORK_F * k + W_PLUG] != 0.0 ? 1.0 : 0.0);
            for (uint c = 26; c < 32; c++)
                put(fbc, S_CHUNK, row, c, 0.0);
            put(fbc, S_CHUNK, row, 32, work[WORK_F * k + W_CONTACT]);
            // variant B's QS-frame group: not available in the engine (zeros)
            for (uint c = 33; c < 39; c++)
                put(fbc, S_CHUNK, row, c, 0.0);
        }
        else
        {
            uint ki = asuint(node(k, 12).x);
            uint row = call_ibase(g) + (j - nc);
            float r_face = imps[IMP_F4 * ki + 3].z;
            float3 inr = node(k, 2).xyz;
            float r2 = r_face * r_face;
            float3 vr = mul(bt, node(k, 3).xyz - vcom);
            float3 v = node(k, 3).xyz, w = node(k, 4).xyz;
            float3x3 iw = mul(mul(rb, float3x3(float3(inr.x, 0, 0), float3(0, inr.y, 0), float3(0, 0, inr.z))), bt);
            ke_imp += 0.5 * m * dot(v, v) + 0.5 * dot(w, mul(iw, w));
            put(fbi, S_IMP, row, 0, ln_ratio(m, m_island));
            put(fbi, S_IMP, row, 1, ln_ratio(m, m_med));
            put(fbi, S_IMP, row, 2, inr.x / (m * r2));
            put(fbi, S_IMP, row, 3, inr.y / (m * r2));
            put(fbi, S_IMP, row, 4, inr.z / (m * r2));
            put(fbi, S_IMP, row, 5, ln_ratio(r_face, L));
            put(fbi, S_IMP, row, 6, ln_ratio(imps[IMP_F4 * ki + 3].x, e_med));
            put(fbi, S_IMP, row, 7, imps[IMP_F4 * ki + 3].w);
            put(fbi, S_IMP, row, 8, gb.x);
            put(fbi, S_IMP, row, 9, gb.y);
            put(fbi, S_IMP, row, 10, gb.z);
            put(fbi, S_IMP, row, 11, vr.x / vref);
            put(fbi, S_IMP, row, 12, vr.y / vref);
            put(fbi, S_IMP, row, 13, vr.z / vref);
            put(fbi, S_IMP, row, 14, length(node(k, 3).xyz - vcom) / c_wave);
            put(fbi, S_IMP, row, 15, work[WORK_F * k + W_CONTACT]);
            for (uint c = 16; c < 22; c++)
                put(fbi, S_IMP, row, c, 0.0);
        }
    }
    ke_imp = group_sum(t, float4(ke_imp, 0, 0, 0)).x;
    if (t == 0)
    {
        uint ni = n - nc;
        put(fbg, S_GLOB, g, 0, ke_imp > 0.0 && ga.x > 0.0 ? ln_ratio(ke_imp, ga.x) : 0.0);
        put(fbg, S_GLOB, g, 1, ke_imp > 0.0 && ga.y > 0.0 ? ln_ratio(ke_imp, ga.y) : 0.0);
        put(fbg, S_GLOB, g, 2, log(dt * c_wave / L));
        put(fbg, S_GLOB, g, 3, dt * vref / L);
        put(fbg, S_GLOB, g, 4, float(ni) / 4.0);
        put(fbg, S_GLOB, g, 5, log(float(nc)) / 10.0);
        put(fbg, S_GLOB, g, 6, length(grav) / G0);
        put(fbg, S_GLOB, g, 7, anchored ? 1.0 : 0.0);
        put(fbg, S_GLOB, g, 8, 0.0);
        put(fbg, S_GLOB, g, 9, 0.0);
        put(fbg, S_GLOB, g, 10, 0.0);
        put(fbg, S_GLOB, g, 11, vref / c_wave);
    }
}

// ------------------------------------------------------------------ feat_bonds

// The failure margin from the joint's state: per family κ over κ_fail (stress-ref
// `kappa_fail`: brittle max(r, 1), ductile (r + 1) / 2, 1 with softening off), the larger
// family; with rebar, the smaller of that and the rebar's consumed rupture work. Kept below
// 1 for a connected bond (the joint law's rule).
float fail_margin_of(JointState js, float4 s8, bool rebar, bool live)
{
    bool ductile = s8.x != 0.0;
    bool softening = s8.y != 0.0;
    float kf_ts = !softening ? 1.0 : (ductile ? 0.5 * (js.ductility + 1.0) : max(js.ductility, 1.0));
    float kf_c = !softening ? 1.0 : (ductile ? 0.5 * (js.ductility_c + 1.0) : max(js.ductility_c, 1.0));
    float q = max(js.kappa / kf_ts, js.kappa_c / kf_c);
    if (rebar && s8.z > 0.0)
        q = min(q, js.rebar_work / s8.z);
    if (live && q >= 1.0)
        q = 1.0 - 5.96e-8;
    return q;
}

// One thread per call bond whose both chunks are in the call (the live bonds): its two
// directed rows.
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_bonds(uint3 id: SV_DispatchThreadID)
{
    uint b = id.x;
    if (b >= ctl[3])
        return;
    uint4 be = bedge[b];
    if (be.x == NONE || be.y == NONE)
        return;
    uint g = asuint(node(be.x, 3).w);
    uint first = call_first(g);
    float4 s0 = calls[CALL_F4 * g];
    float4 s1 = calls[CALL_F4 * g + 1];
    float L = s0.x, vref = s0.z;
    float f_t = s1.x, e_med = s1.y, GA = s1.z;
    uint row = be.z;
    float4 g0 = sb[SB_F4 * row];
    float4 g1 = sb[SB_F4 * row + 1];
    float4 g2 = sb[SB_F4 * row + 2];
    float4 k3 = sb[SB_F4 * row + 3];
    float4 k4 = sb[SB_F4 * row + 4];
    float4 st0 = sb[SB_F4 * row + 5];
    float4 st1 = sb[SB_F4 * row + 6];
    float4 s8 = sb[SB_F4 * row + 8];
    uint4 ends = asuint(sb[SB_F4 * row + 7]);
    uint na = be.x, nb = be.y;
    float3x3 ra = quat_matrix(node(na, 1));
    float4 rest = float4(work[WORK_F * na + W_REST], work[WORK_F * na + W_REST + 1], work[WORK_F * na + W_REST + 2], work[WORK_F * na + W_REST + 3]);
    float3x3 rs = mul(ra, transpose(quat_matrix(rest)));
    float3 xa = node(na, 13).xyz, xb = node(nb, 13).xyz;
    float3 nrm = mul(rs, g1.xyz);
    float3 t1 = mul(rs, g2.xyz);
    if (dot(nrm, xb - xa) < 0.0)
    {
        nrm = -nrm;
        t1 = -t1;
    }
    float3 t2 = cross(nrm, t1);
    float3 center_a = float3(work[WORK_F * na + W_CENTER], work[WORK_F * na + W_CENTER + 1], work[WORK_F * na + W_CENTER + 2]);
    float3 p = xa + mul(rs, g0.xyz - center_a);
    float area = g0.w;
    float kl = e_med * area / L, ka = e_med * area * L;
    BondDyn dyn = bond_dyn[be.w];
    bool rebar = k4.w != 0.0;
    float margin = fail_margin_of(dyn.js, s8, rebar, true);
    uint pa = uint(work[WORK_F * na + W_PLUG]), pb = uint(work[WORK_F * nb + W_PLUG]);
    bool in_plug = (pa != 0) != (pb != 0);
    float ratio = 0.0;
    if (in_plug)
    {
        uint pid = max(pa, pb);
        uint row0 = asuint(calls[CALL_F4 * g + 8].x);
        ratio = ln_ratio(contacts[CON_F4 * (row0 + pid - 1) + 4].x, 1.0);
    }
    float strength[8] = { st0.x, st0.y, st0.z, st0.w, st1.x, st1.y, st1.z, st1.w };
    uint fb = params.bases.w;
    for (uint side = 0; side < 2; side++)
    {
        uint r = 2 * b + side;
        uint sn = side == 0 ? na : nb;
        uint dn = side == 0 ? nb : na;
        float3 ft1 = side == 0 ? t1 : -t1;
        float3 fn = side == 0 ? nrm : -nrm;
        float3 vs = node(sn, 3).xyz + cross(node(sn, 4).xyz, p - node(sn, 13).xyz);
        float3 vd = node(dn, 3).xyz + cross(node(dn, 4).xyz, p - node(dn, 13).xyz);
        float3 q = p - node(sn, 13).xyz;
        float3 dv = vd - vs;
        put(fb, S_BOND, r, 0, ln_ratio(area, L * L));
        put(fb, S_BOND, r, 1, dot(ft1, q) / L);
        put(fb, S_BOND, r, 2, dot(t2, q) / L);
        put(fb, S_BOND, r, 3, dot(fn, q) / L);
        put(fb, S_BOND, r, 4, ln_ratio(k3.y, kl));
        put(fb, S_BOND, r, 5, ln_ratio(k3.z, kl));
        put(fb, S_BOND, r, 6, ln_ratio(k3.z, kl));
        put(fb, S_BOND, r, 7, ln_ratio(k3.w, ka));
        put(fb, S_BOND, r, 8, ln_ratio(k4.x, ka));
        put(fb, S_BOND, r, 9, ln_ratio(k4.y, ka));
        put(fb, S_BOND, r, 10, ln_ratio(strength[0], f_t));
        put(fb, S_BOND, r, 11, ln_ratio(strength[2], f_t));
        put(fb, S_BOND, r, 12, ln_ratio(strength[1], f_t));
        put(fb, S_BOND, r, 13, ln_ratio(strength[5] * area, GA));
        put(fb, S_BOND, r, 14, ln_ratio(strength[6] * area, GA));
        put(fb, S_BOND, r, 15, ln_ratio(strength[7] * area, GA));
        put(fb, S_BOND, r, 16, k4.z);
        put(fb, S_BOND, r, 17, dyn.js.damage);
        put(fb, S_BOND, r, 18, dyn.js.crush);
        put(fb, S_BOND, r, 19, margin);
        put(fb, S_BOND, r, 20, log(1.0 + max(dyn.js.utilization, 0.0)));
        // the previous frame's peak utilisation: the frame-end utilisation (the engine keeps
        // no per-frame peak; documented)
        put(fb, S_BOND, r, 21, log(1.0 + max(dyn.js.utilization, 0.0)));
        put(fb, S_BOND, r, 22, rebar ? 1.0 : 0.0);
        put(fb, S_BOND, r, 23, 0.0);
        put(fb, S_BOND, r, 24, in_plug ? 1.0 : 0.0);
        put(fb, S_BOND, r, 25, ratio);
        put(fb, S_BOND, r, 26, dot(ft1, dv) / vref);
        put(fb, S_BOND, r, 27, dot(t2, dv) / vref);
        put(fb, S_BOND, r, 28, dot(fn, dv) / vref);
        put_rot6(fb, S_BOND, r, 29, ft1, t2, quat_matrix(node(sn, 1)));
        put_rot6(fb, S_BOND, r, 35, ft1, t2, quat_matrix(node(dn, 1)));
        // variant B's QS-frame group: not available in the engine
        put(fb, S_BOND, r, 41, 0.0);
        put(fb, S_BOND, r, 42, 0.0);
        put(fb, S_BOND, r, 43, 0.0);
    }
}

// ------------------------------------------------------------------ feat_deform

float4 qmulf(float4 a, float4 b)
{
    return quat_vec(quat_mul(quat_of(a), quat_of(b)));
}

float4 qconj(float4 q)
{
    return float4(-q.xyz, q.w);
}

float4 rotvec_quat_f(float3 r)
{
    float th = length(r);
    if (th < 1e-12)
        return normalize(float4(0.5 * r, 1.0));
    return float4(r / th * sin(0.5 * th), cos(0.5 * th));
}

float3 quat_rotvec(float4 q)
{
    float4 p = q.w < 0.0 ? -q : q;
    float s = length(p.xyz);
    if (s < 1e-12)
        return 2.0 * p.xyz;
    return p.xyz / s * (2.0 * atan2(s, p.w));
}

float3 to_zup_f(float3 v)
{
    return float3(v.x, -v.z, v.y);
}

// One thread per layout chunk, after the split pass of a frame whose batch was applied
// (`STRESS_DECODED_DEFORMATION=1`): each chunk of a call that stood keeps its decoded
// departure from its body's fitted rigid motion as its hidden state (stress-ref
// `apply_decoded_with(keep_deformation)`): displacement and rotation in the body's frame,
// velocity and spin against the body's (an anchored body's: against its spin only).
[shader("compute")]
[numthreads(64, 1, 1)]
void feat_deform(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count)
        return;
    uint4 info = chunks[k].info;
    uint row = sinc[2 + info.z] + info.w;
    uint2 kn = key_node[row];
    if (kn.x == 0 || kn.y != params.prev_frame)
        return;
    uint n = kn.x - 1;
    uint g = asuint(node(n, 3).w);
    if (asuint(calls[CALL_F4 * g + 4].w) != 0)
        return;
    uint i = info.y;
    Island isl = islands[i];
    bool anchored = (isl.info.x & 1u) != 0;
    // the body's frame now (placed by the window end)
    float4 q = isl.rotation;
    float3 com_w;
    float3 com_b = isl.com.xyz;
    float3 vb = float3(0.0), wb = float3(0.0);
    uint body = asuint(ctab[params.engine_base + 3 * i].x);
    if (body != NONE)
    {
        float4 p_inv = ctab[params.engine_base + 3 * i + 1];
        float h = 0.70710678118;
        q = normalize(qmulf(qmulf(float4(h, 0.0, 0.0, h), bodies[7 * body + 1]), p_inv));
        com_w = to_zup_f(bodies[7 * body].xyz);
        com_b = ctab[params.engine_base + 3 * i + 2].xyz;
        vb = to_zup_f(bodies[7 * body + 3].xyz);
        wb = to_zup_f(bodies[7 * body + 4].xyz);
    }
    else
    {
        com_w = isl.position.xyz + rotate(quat_of(q), com_b);
    }
    float3 origin = com_w - rotate(quat_of(q), com_b);
    float3 center = chunks[k].center.xyz;
    float3 x1 = node(n, 13).xyz + node(n, 10).xyz;
    float3 u = inverse_rotate(quat_of(q), x1 - origin) - center;
    // the chunk's end orientation without its rest rotation: exp(drot) · (q_node · rest⁻¹)
    float4 rest = float4(work[WORK_F * n + W_REST], work[WORK_F * n + W_REST + 1], work[WORK_F * n + W_REST + 2], work[WORK_F * n + W_REST + 3]);
    float4 q1 = normalize(qmulf(rotvec_quat_f(node(n, 11).xyz), qmulf(node(n, 1), qconj(rest))));
    float3 th = quat_rotvec(normalize(qmulf(qconj(q), q1)));
    float3 r = rotate(quat_of(q), center + u - com_b);
    float3 v1 = node(n, 8).xyz, w1 = node(n, 9).xyz;
    float3 vh = anchored ? inverse_rotate(quat_of(q), v1 - cross(wb, r)) : inverse_rotate(quat_of(q), v1 - vb - cross(wb, r));
    float3 wh = anchored ? inverse_rotate(quat_of(q), w1) : inverse_rotate(quat_of(q), w1 - wb);
    state[4 * k] = float4(u, state[4 * k].w);
    state[4 * k + 1] = float4(th, state[4 * k + 1].w);
    state[4 * k + 2] = float4(vh, state[4 * k + 2].w);
    state[4 * k + 3] = float4(wh, state[4 * k + 3].w);
}
