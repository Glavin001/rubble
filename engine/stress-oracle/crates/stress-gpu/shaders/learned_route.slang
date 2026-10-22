// The learned impact path's routing: stress-ref's `World::select_islands` on the GPU, so a
// learned call's members are the bodies the CPU routing would put in islands from the same
// state (P1's recorder: `predict_island_members`). Per body a part: its swept bounds, the
// material and bond data of the rigid-assumption test, its motion. Then, as the CPU:
//   - debris reach (`STRESS_DEBRIS_ROUTING=1`): a struck cluster an impactor reaches closing
//     fast grows its bounds by twice the closing velocity times dt;
//   - every overlapping pair of parts (a sort along x and a sweep: the CPU's set, found in
//     any order) joins its two bodies unless both are immovable (or, tiered with
//     `tiered_fast_joins`, it is a slow contact between clusters, neither debris); an owned
//     pair (tiered: an impactor's, fast) whose rigid assumption fails seeds both;
//   - against the ground (a half-space) an owned contact seeds the body; it joins nothing;
//   - the routed bodies: the components (union-find over the joining pairs) holding a seed.
// Components do not depend on the order pairs are taken in, so the result is the CPU's.
// Atomics only on the labels and the edge count (`Atomic<uint>`: WGSL's atomic<u32>);
// labels converge to each component's least part index, whatever the order.
//
//   lr_islands     one group per island: its part
//   lr_impactors   one thread per impactor: its part
//   lr_debris      one thread per island: debris reach
//   lr_keys        one thread per padded slot: sort keys (lo.x) and the label / seed reset
//   lr_bitonic     one thread per pair of slots: a bitonic step (k, j from the pass table)
//   lr_sweep       one thread per sorted part: its overlaps with later parts; ground
//   lr_hook        one thread per edge: labels min-joined (a round)
//   lr_jump        one thread per part: pointer jumping (the same round)
//   lr_roots       one thread per part: a seed marks its component's root
//   lr_finish      one thread per part: routed islands into `isl`, routed impactors' mask
// Frames: the stress solver's (Z-up); the engine's bodies are Y-up.

import common;
import joint;
import bonds;

static const uint G = 64;
static const uint NONE = 0xffffffffu;
static const uint BODY_F4 = 7;
static const uint ISL_U4 = 5;
static const uint IMP_F4 = 4;
static const uint PART_F4 = 8;
// part flags (word 0.w bits)
static const uint P_VALID = 1;
static const uint P_IMPACTOR = 2;
static const uint P_STRUCK = 4;
static const uint P_DEBRIS = 8;
static const uint P_IMMOVABLE = 16;
// atomic control words
static const uint A_EDGES = 0;
static const uint A_OVER = 1;     // edges past capacity (1)
static const uint A_ROUTED_IMP = 2; // routed impactors' mask (OR)
static const uint A_CHANGED = 8;  // + round: a label changed in that round
static const uint ROUNDS = 48;
static const uint ROUTED = 1u << 31;

struct RouteParams
{
    uint island_cap;   // dispatch bound
    uint impactor_count;
    uint engine_base;
    uint n_pad;        // sort slots (a power of two >= island_cap + impactors)
    float dt;
    float speed;       // the tiered impact speed
    uint tiered;       // qs islands under the quasi-static solve
    uint fast_joins;   // `tiered_fast_joins`
    uint debris;       // `debris_reach_routing`
    uint ground;       // a ground half-space
    float ground_height;
    float ground_modulus; // plane strain, scaled
    uint edge_cap;
    uint always;       // ignore the routing check's flag (untiered: any contact may seed)
    uint pad0;
    uint pad1;
};

struct MaterialTable
{
    JointMaterial m[64];
};

[[vk::binding(0, 0)]] ConstantBuffer<RouteParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<Island> islands;
[[vk::binding(2, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(3, 0)]] StructuredBuffer<float4> state;
[[vk::binding(4, 0)]] StructuredBuffer<float4> ctab;
[[vk::binding(5, 0)]] StructuredBuffer<float4> bodies;
[[vk::binding(6, 0)]] StructuredBuffer<BondStatic> bond_static;
[[vk::binding(7, 0)]] ConstantBuffer<MaterialTable> materials;
// the chunk rows (`[0]` structures, `[1]` rows, `[2 + structure]` its first row), then per
// stable chunk row its wave speed sqrt(E scale / rho) as bits (at `2 + structures + row`)
[[vk::binding(8, 0)]] StructuredBuffer<uint> rtab;
// per impactor (learned.rs `set_impactors`): (body bits, bounding radius, face radius,
// mass), (inertia, plane-strain modulus), ...
[[vk::binding(9, 0)]] StructuredBuffer<float4> imps;
[[vk::binding(10, 0)]] StructuredBuffer<uint> counters;
[[vk::binding(11, 0)]] StructuredBuffer<uint> route;
[[vk::binding(12, 0)]] RWStructuredBuffer<float4> parts;
[[vk::binding(13, 0)]] RWStructuredBuffer<uint2> sorted;
[[vk::binding(14, 0)]] RWStructuredBuffer<uint2> edges;
[[vk::binding(15, 0)]] RWStructuredBuffer<Atomic<uint>> lab;
[[vk::binding(16, 0)]] RWStructuredBuffer<Atomic<uint>> actl;
[[vk::binding(17, 0)]] RWStructuredBuffer<uint> seed;
[[vk::binding(18, 0)]] RWStructuredBuffer<uint4> isl;
// this dispatch's pass: (k, j) of a sort step, or (-, -, round) of the labels (a small
// buffer per pass, made once)
[[vk::binding(19, 0)]] StructuredBuffer<uint4> pass;
// learned.slang's control words: [CTL_ROUTE] bits 1 edges past capacity, 2 labels not
// converged in ROUNDS; [CTL_ROUTE_EDGES] the joining pairs
[[vk::binding(20, 0)]] RWStructuredBuffer<uint> ctl;
static const uint CTL_ROUTE = 20;
static const uint CTL_ROUTE_EDGES = 21;

float3 to_zup(float3 v)
{
    return float3(v.x, -v.z, v.y);
}

float4 qmul(float4 a, float4 b)
{
    return quat_vec(quat_mul(quat_of(a), quat_of(b)));
}

float float_inf()
{
    return asfloat(0x7f800000u);
}

uint live_islands()
{
    return counters[0];
}

uint part_count()
{
    return live_islands() + min(params.impactor_count, 32u);
}

bool gate()
{
    return params.always != 0 || route[0] != 0;
}

// ------------------------------------------------------------------ the parts

struct Part
{
    float3 lo;
    float3 hi;
    float3 box_lo;
    float3 box_hi;
    float3 center;
    float3 v;
    float3 w;
    float reach;
    float mass;     // inf when immovable
    float modulus;
    float radius;   // contact radius
    float round_trip;
    float capacity;
    uint flags;
};

void store_part(uint i, Part p)
{
    parts[PART_F4 * i] = float4(p.lo, asfloat(p.flags));
    parts[PART_F4 * i + 1] = float4(p.hi, p.mass);
    parts[PART_F4 * i + 2] = float4(p.box_lo, p.reach);
    parts[PART_F4 * i + 3] = float4(p.box_hi, p.modulus);
    parts[PART_F4 * i + 4] = float4(p.center, p.radius);
    parts[PART_F4 * i + 5] = float4(p.v, p.round_trip);
    parts[PART_F4 * i + 6] = float4(p.w, p.capacity);
}

Part load_part(uint i)
{
    Part p;
    float4 a = parts[PART_F4 * i];
    float4 b = parts[PART_F4 * i + 1];
    float4 c = parts[PART_F4 * i + 2];
    float4 d = parts[PART_F4 * i + 3];
    float4 e = parts[PART_F4 * i + 4];
    float4 f = parts[PART_F4 * i + 5];
    float4 g = parts[PART_F4 * i + 6];
    p.lo = a.xyz;
    p.flags = asuint(a.w);
    p.hi = b.xyz;
    p.mass = b.w;
    p.box_lo = c.xyz;
    p.reach = c.w;
    p.box_hi = d.xyz;
    p.modulus = d.w;
    p.center = e.xyz;
    p.radius = e.w;
    p.v = f.xyz;
    p.round_trip = f.w;
    p.w = g.xyz;
    p.capacity = g.w;
    return p;
}

struct Frame
{
    float4 q;
    float3 origin;
    float3 com;
    float3 v;
    float3 w;
};

Frame island_frame(uint i)
{
    Island isl_rec = islands[i];
    Frame f;
    uint body = asuint(ctab[params.engine_base + 3 * i].x);
    f.q = isl_rec.rotation;
    f.origin = isl_rec.position.xyz;
    f.v = isl_rec.velocity.xyz;
    f.w = isl_rec.angular_velocity.xyz;
    if (body != NONE)
    {
        float4 p_inv = ctab[params.engine_base + 3 * i + 1];
        float3 com = ctab[params.engine_base + 3 * i + 2].xyz;
        float h = 0.70710678118;
        f.q = normalize(qmul(qmul(float4(h, 0.0, 0.0, h), bodies[BODY_F4 * body + 1]), p_inv));
        f.com = to_zup(bodies[BODY_F4 * body].xyz);
        f.origin = f.com - rotate(quat_of(f.q), com);
        f.v = to_zup(bodies[BODY_F4 * body + 3].xyz);
        f.w = to_zup(bodies[BODY_F4 * body + 4].xyz);
    }
    else
    {
        f.com = f.origin + rotate(quat_of(f.q), isl_rec.com.xyz);
    }
    return f;
}

groupshared float3 g_lo[G];
groupshared float3 g_hi[G];
groupshared float4 g_a[G]; // reach (max), modulus, speed, radius (mins)
groupshared float g_cap[G];

// One group per island (`select_islands`' cluster part): bounds of the chunk centres +- their
// bounding radius, reach about the centre of mass, the least plane-strain modulus, wave speed
// and half extent over its chunks, the least tensile capacity over its bonds (struck when it
// has any), the sweep over dt.
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_islands(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    uint t = thread.x;
    if (i >= live_islands() || !gate())
        return;
    Island rec = islands[i];
    Frame f = island_frame(i);
    uint n = rec.range.y - rec.range.x;
    float inf = float_inf();
    float3 lo = float3(inf), hi = float3(-inf);
    float4 a = float4(0.0, inf, inf, inf);
    for (uint c = t; c < n; c += G)
    {
        uint k = rec.range.x + c;
        ChunkStatic ch = chunks[k];
        float3 p = f.origin + rotate(quat_of(f.q), ch.center.xyz + state[4 * k].xyz);
        float r = ch.half.w;
        lo = min(lo, p - float3(r));
        hi = max(hi, p + float3(r));
        a.x = max(a.x, length(p - f.com) + r);
        a.y = min(a.y, ch.cmat.x);
        a.z = min(a.z, asfloat(rtab[2 + rtab[0] + rtab[2 + ch.info.z] + ch.info.w]));
        a.w = min(a.w, min(ch.half.x, min(ch.half.y, ch.half.z)));
    }
    float cap = inf;
    for (uint b = rec.range.z + t; b < rec.range.w; b += G)
    {
        JointBond law = bond_static[b].law;
        cap = min(cap, materials.m[law.ids.x].strength.x * law.geom0.x * law.geom1.w);
    }
    // (a leading barrier: Slang may sink a later shared read past the last one)
    GroupMemoryBarrierWithGroupSync();
    g_lo[t] = lo;
    g_hi[t] = hi;
    g_a[t] = a;
    g_cap[t] = cap;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            g_lo[t] = min(g_lo[t], g_lo[t + s]);
            g_hi[t] = max(g_hi[t], g_hi[t + s]);
            float4 x = g_a[t], y = g_a[t + s];
            g_a[t] = float4(max(x.x, y.x), min(x.yzw, y.yzw));
            g_cap[t] = min(g_cap[t], g_cap[t + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t != 0)
        return;
    Part p;
    p.flags = 0;
    if (n == 0)
    {
        store_part(i, p);
        return;
    }
    lo = g_lo[0];
    hi = g_hi[0];
    a = g_a[0];
    bool anchored = (rec.info.x & ISLAND_ANCHORED) != 0;
    bool struck = rec.range.w > rec.range.z;
    float extent = max(hi.x - lo.x, max(hi.y - lo.y, hi.z - lo.z));
    float3 sweep = abs(f.v) * params.dt + float3(length(f.w) * a.x * params.dt);
    p.flags = P_VALID | (struck ? P_STRUCK : 0u) | (anchored ? P_IMMOVABLE : 0u);
    p.lo = lo - sweep;
    p.hi = hi + sweep;
    p.box_lo = lo;
    p.box_hi = hi;
    p.center = f.com;
    p.v = f.v;
    p.w = f.w;
    p.reach = a.x;
    p.mass = anchored ? inf : rec.com.w;
    p.modulus = a.y;
    p.radius = a.w;
    p.round_trip = 2.0 * extent / a.z;
    p.capacity = g_cap[0];
    store_part(i, p);
}

// One thread per impactor: its bounding sphere swept over dt.
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_impactors(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= min(params.impactor_count, 32u) || !gate())
        return;
    float4 a = imps[IMP_F4 * k];
    float4 b = imps[IMP_F4 * k + 1];
    uint body = asuint(a.x);
    float r = a.y;
    float3 c = to_zup(bodies[BODY_F4 * body].xyz);
    float3 v = to_zup(bodies[BODY_F4 * body + 3].xyz);
    float3 w = to_zup(bodies[BODY_F4 * body + 4].xyz);
    float3 sweep = abs(v) * params.dt + float3(length(w) * r * params.dt);
    Part p;
    p.flags = P_VALID | P_IMPACTOR;
    p.lo = c - float3(r) - sweep;
    p.hi = c + float3(r) + sweep;
    p.box_lo = c - float3(r);
    p.box_hi = c + float3(r);
    p.center = c;
    p.v = v;
    p.w = w;
    p.reach = r;
    p.mass = a.w;
    p.modulus = b.w;
    p.radius = a.z;
    p.round_trip = 0.0;
    p.capacity = 0.0;
    store_part(live_islands() + k, p);
}

bool overlaps(Part a, Part b)
{
    return all(a.lo <= b.hi) && all(b.lo <= a.hi);
}

// `closing_speed` between bodies (neither the ground).
float closing_speed(Part a, Part b)
{
    float spin = length(a.w) * a.reach + length(b.w) * b.reach;
    float3 rel = a.v - b.v;
    float3 n0 = clamp(a.center, b.box_lo, b.box_hi) - a.center;
    float3 n1 = clamp(b.center, a.box_lo, a.box_hi) - b.center;
    float l0 = length(n0), l1 = length(n1);
    if (!(l0 > 0.0) || !(l1 > 0.0))
        return length(rel) + spin;
    return max(dot(rel, n0 / l0), dot(rel, -n1 / l1)) + spin;
}

// api.rs `hertz_duration`.
float hertz_duration(float m, float r, float e, float v)
{
    float vv = max(abs(v), 1e-3);
    return 2.87 * pow(m * m / (r * e * e * vv), 0.2);
}

// `rigid_assumption_fails(a, b, false)`, with the closing speed given.
bool rigid_fails(Part a, Part b, float speed)
{
    if ((b.flags & P_STRUCK) == 0)
        return false;
    bool fa = (a.flags & P_IMMOVABLE) == 0, fb = (b.flags & P_IMMOVABLE) == 0;
    float m;
    if (fa && fb)
        m = a.mass * b.mass / (a.mass + b.mass);
    else if (fa)
        m = a.mass;
    else if (fb)
        m = b.mass;
    else
        return false;
    if (!(speed > 0.0))
        return false;
    float modulus = 1.0 / (1.0 / a.modulus + 1.0 / b.modulus);
    float duration = hertz_duration(m, min(a.radius, b.radius), modulus, speed);
    float peak = 3.14159265358979 * (2.0 * m * speed) / (2.0 * duration);
    return duration < b.round_trip || peak > b.capacity;
}

// One thread per island: debris reach (a struck cluster an impactor reaches closing faster
// than the tiered speed grows by twice the closing velocity times dt, from zero).
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_debris(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (params.debris == 0 || c >= live_islands() || !gate())
        return;
    Part pc = load_part(c);
    if ((pc.flags & P_VALID) == 0 || (pc.flags & P_STRUCK) == 0)
        return;
    float3 g0 = float3(0.0), g1 = float3(0.0);
    uint live = live_islands();
    for (uint k = 0; k < min(params.impactor_count, 32u); k++)
    {
        Part pk = load_part(live + k);
        if (!overlaps(pk, pc) || !(closing_speed(pk, pc) > params.speed))
            continue;
        float3 d = (pk.v - pc.v) * (2.0 * params.dt);
        g0 = min(g0, d);
        g1 = max(g1, d);
    }
    if (any(g0 != float3(0.0)) || any(g1 != float3(0.0)))
    {
        parts[PART_F4 * c] = float4(pc.lo + g0, asfloat(pc.flags | P_DEBRIS));
        parts[PART_F4 * c + 1] = float4(pc.hi + g1, pc.mass);
    }
}

// ------------------------------------------------------------------ sort and sweep

// A float's bits ordered as the float (for the sort).
uint sortable(float x)
{
    uint u = asuint(x);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

// One thread per padded slot: the sort's keys (invalid parts and padding last), the labels
// and seeds reset, the control words cleared.
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_keys(uint3 id: SV_DispatchThreadID)
{
    uint s = id.x;
    if (s >= params.n_pad)
        return;
    if (s < 8 + ROUNDS)
        actl[s].store(0);
    uint n = part_count();
    bool used = s < n && gate() && (asuint(parts[PART_F4 * s].w) & P_VALID) != 0;
    sorted[s] = uint2(used ? sortable(parts[PART_F4 * s].x) : 0xffffffffu, used ? s : NONE);
    if (s < n)
    {
        lab[s].store(s);
        seed[s] = 0;
    }
}

// One thread per pair of slots: a bitonic compare-exchange (k, j: `pass[r]`).
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_bitonic(uint3 id: SV_DispatchThreadID)
{
    uint4 kj = pass[0];
    uint k = kj.x, j = kj.y;
    uint t = id.x;
    if (t >= params.n_pad / 2)
        return;
    // the lower slot of the pair: t with a zero inserted at bit log2(j)
    uint lo = ((t & ~(j - 1)) << 1) | (t & (j - 1));
    uint hi = lo | j;
    uint2 a = sorted[lo], b = sorted[hi];
    bool up = (lo & k) == 0;
    bool swap = up ? (a.x > b.x || (a.x == b.x && a.y > b.y)) : (a.x < b.x || (a.x == b.x && a.y < b.y));
    if (swap)
    {
        sorted[lo] = b;
        sorted[hi] = a;
    }
}

// Ground: a half-space at `ground_height`, below everything.
bool ground_owned(Part p)
{
    // `rigid_assumption_fails(ground, p)` (the other way round never: the ground is not
    // struck); the closing speed down onto it.
    float spin = length(p.w) * p.reach;
    float speed = p.v.z * -1.0 + spin;
    Part g;
    g.flags = P_VALID | P_IMMOVABLE;
    g.mass = float_inf();
    g.modulus = params.ground_modulus;
    g.radius = float_inf();
    return rigid_fails(g, p, speed);
}

// One thread per sorted part: every later part whose swept bounds overlap its own (the
// sweep stops at the first whose lower x bound is past its upper one), as `select_islands`
// takes each pair; and the ground.
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_sweep(uint3 id: SV_DispatchThreadID)
{
    uint s = id.x;
    if (s >= params.n_pad || !gate())
        return;
    uint ia = sorted[s].y;
    if (ia == NONE)
        return;
    Part a = load_part(ia);
    bool tiered = params.tiered != 0;
    uint live = live_islands();
    if (params.ground != 0 && a.lo.z <= params.ground_height)
    {
        // (tiered: only an impactor's contact is owned, and an impactor is never struck)
        if (!tiered && ground_owned(a))
            seed[ia] = 1;
    }
    for (uint q = s + 1; q < params.n_pad; q++)
    {
        uint ib = sorted[q].y;
        if (ib == NONE)
            break;
        Part b = load_part(ib);
        if (b.lo.x > a.hi.x)
            break;
        if (!overlaps(a, b))
            continue;
        bool impactor = ((a.flags | b.flags) & P_IMPACTOR) != 0;
        float sab = closing_speed(a, b), sba = closing_speed(b, a);
        bool fast = !tiered || max(sab, sba) > params.speed;
        bool owned = (!tiered || impactor) && fast && (rigid_fails(a, b, sab) || rigid_fails(b, a, sba));
        bool both_immovable = (a.flags & P_IMMOVABLE) != 0 && (b.flags & P_IMMOVABLE) != 0;
        if (both_immovable)
            continue;
        if (tiered && params.fast_joins != 0 && !impactor && !fast && ((a.flags | b.flags) & P_DEBRIS) == 0)
            continue;
        uint e = actl[A_EDGES].add(1);
        if (e < params.edge_cap)
            edges[e] = uint2(ia, ib);
        else
            actl[A_OVER].store(1);
        if (owned)
        {
            seed[ia] = 1;
            seed[ib] = 1;
        }
    }
}

// ------------------------------------------------------------------ components

uint edge_count()
{
    return min(actl[A_EDGES].load(), params.edge_cap);
}

// One thread per edge, round r (`pass[r].x`; a round after one that changed nothing does
// nothing): both ends' labels and their labels' labels min-joined.
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_hook(uint3 id: SV_DispatchThreadID)
{
    uint r = pass[0].z;
    uint e = id.x;
    if (!gate() || (r > 0 && actl[A_CHANGED + r - 1].load() == 0) || e >= edge_count())
        return;
    uint2 ab = edges[e];
    uint la = lab[ab.x].load(), lb = lab[ab.y].load();
    if (la == lb)
        return;
    uint m = min(la, lb);
    lab[ab.x].min(m);
    lab[ab.y].min(m);
    lab[max(la, lb)].min(m);
    actl[A_CHANGED + r].store(1);
}

// One thread per part, round r: a label's label (pointer jumping; labels only decrease).
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_jump(uint3 id: SV_DispatchThreadID)
{
    uint r = pass[0].z;
    uint i = id.x;
    if (!gate() || (r > 0 && actl[A_CHANGED + r - 1].load() == 0) || i >= part_count())
        return;
    uint l = lab[i].load();
    uint ll = lab[l].load();
    if (ll < l)
    {
        lab[i].min(ll);
        actl[A_CHANGED + r].store(1);
    }
}

// One thread per part: a seed marks its component's root (the same value from every writer).
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_roots(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (!gate() || i >= part_count())
        return;
    if ((seed[i] & 1) != 0)
        seed[lab[i].load()] |= 2;
}

// One thread per part: a routed island's `isl` entry (ROUTED: the learned batch's call), a
// routed impactor in the mask; an island not routed cleared.
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_finish(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    uint live = live_islands();
    if (i >= part_count())
        return;
    bool valid = gate() && (asuint(parts[PART_F4 * i].w) & P_VALID) != 0;
    bool routed = valid && (seed[lab[i].load()] & 2) != 0;
    if (i < live)
    {
        isl[ISL_U4 * i] = uint4(routed ? ROUTED : 0u, NONE, 0, 0);
        isl[ISL_U4 * i + 3] = uint4(0);
        isl[ISL_U4 * i + 4] = uint4(0);
    }
    else if (routed)
    {
        actl[A_ROUTED_IMP].or(1u << (i - live));
    }
    if (i == 0)
    {
        uint err = actl[A_OVER].load() | (actl[A_CHANGED + ROUNDS - 1].load() != 0 ? 2u : 0u);
        ctl[CTL_ROUTE] = gate() ? err : 0u;
        ctl[CTL_ROUTE_EDGES] = gate() ? actl[A_EDGES].load() : 0u;
    }
}

// One thread per island: a routed island's entry carries the routed impactors' mask
// (`learned_compact` takes the call's impactors from it; at most 31 impactors).
[shader("compute")]
[numthreads(64, 1, 1)]
void lr_mark(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= live_islands() || !gate())
        return;
    uint4 e = isl[ISL_U4 * i];
    if (e.x != 0)
        isl[ISL_U4 * i] = uint4(ROUTED | (actl[A_ROUTED_IMP].load() & ~ROUTED), e.yzw);
}
