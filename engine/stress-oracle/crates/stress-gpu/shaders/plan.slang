// The contact plan on the GPU (docs/stress-fracture-gpu-resident.md phase 6a): contacts.rs
// `plan` from the GPU's own state, so an island segment needs no download to plan.
//
//   plan_motion      one thread per chunk: position, velocity and spin, bounding radius,
//                    travel budget over the horizon, whether it falls, start orientation;
//   plan_reach       one group per island: the whole-cluster test's centre, spread,
//                    largest radius + budget and largest speed;
//   plan_pair_count  one thread per job (an owned cluster pair and a chunk of the first):
//                    its chunks of the second cluster that pass the sweep and box tests;
//   plan_scan        one group: the jobs' output offsets (the host loop's order);
//   plan_pair_write  the jobs again, writing their pairs at those offsets;
//   plan_side        one thread per chunk: ground and impactor candidates (flags).
//
// f32 throughout, with the box test's wider margin (world.slang `may_overlap`): a plan
// may hold a borderline candidate the host's f64 plan leaves out, never the reverse
// beyond rounding.

import common;

struct PlanParams
{
    uint chunk_count;
    uint island_count;
    uint job_count;
    uint impactor_count;
    float duration;
    float g;           // |gravity|
    float gz;          // gravity.z
    float ground_height;
    uint has_ground;
    uint reach_base;   // pm: per island 2 float4
    uint cp_base;      // pu: owned cluster pairs (ia, ib)
    uint job_base;     // pu: jobs (pair index, chunk)
    uint count_base;   // pu: per job its count, then offsets (job_count + 1)
    uint pair_base;    // pu: pairs (ka, kb)
    uint flag_base;    // pu: per chunk: bit 0 rotation checked, bit 1 ground, bit 2 + i impactor i
    uint owned_base;   // pu: per island 1 when an impact island owns it (not engine-driven)
};

struct Impactor
{
    float4 position;
    float4 position_err;
    float4 velocity;
    float4 velocity_err;
    float4 angular_velocity;
    float4 rotation;
    float4 inertia0;
    float4 inertia1;
    float4 inertia2;
    float4 inv0;
    float4 inv1;
    float4 inv2;
    float4 shape;
    float4 half;
    float4 mat;
    float4 crush;
    float4 geom;
    float4 rotation_err;
    float4 ledger;
    uint4 cand;
    float4 momentum;
    float4 momentum_err;
};

[[vk::binding(0, 0)]] ConstantBuffer<PlanParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(2, 0)]] StructuredBuffer<Island> islands;
[[vk::binding(3, 0)]] StructuredBuffer<float4> state;
[[vk::binding(4, 0)]] StructuredBuffer<Impactor> impactors;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> pm;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> pu;

struct Box
{
    float3 center;
    float3 axis0;
    float3 axis1;
    float3 axis2;
    float3 half;
};

// world.slang `may_overlap`.
bool may_overlap(Box a, Box b)
{
    float3 d = b.center - a.center;
    float margin = 1e-5 * (length(a.half) + length(b.half));
    float3 axes[6] = { a.axis0, a.axis1, a.axis2, b.axis0, b.axis1, b.axis2 };
    for (uint i = 0; i < 15; i++)
    {
        float3 l = i < 6 ? axes[i] : cross(axes[(i - 6) / 3], axes[3 + (i - 6) % 3]);
        float len = length(l);
        if (len <= 1e-6)
            continue;
        float ra = a.half.x * abs(dot(a.axis0, l)) + a.half.y * abs(dot(a.axis1, l)) + a.half.z * abs(dot(a.axis2, l));
        float rb = b.half.x * abs(dot(b.axis0, l)) + b.half.y * abs(dot(b.axis1, l)) + b.half.z * abs(dot(b.axis2, l));
        if (abs(dot(d, l)) > ra + rb + margin * len)
            return false;
    }
    return true;
}

// contacts.rs `grown_box`: the chunk's box (cluster rotation x hidden rotation x chunk
// rotation) grown by `grow`, centred at `center`.
Box grown_box(uint c, float3 center, float grow)
{
    ChunkStatic ch = chunks[c];
    Island isl = islands[ch.info.y];
    Quat q = quat_of(isl.rotation);
    float3 th = state[4 * c + 1].xyz;
    Quat hidden = from_axis_angle(th, length(th));
    Box b;
    b.center = center;
    b.axis0 = rotate(q, rotate(hidden, float3(ch.crot0.x, ch.crot1.x, ch.crot2.x)));
    b.axis1 = rotate(q, rotate(hidden, float3(ch.crot0.y, ch.crot1.y, ch.crot2.y)));
    b.axis2 = rotate(q, rotate(hidden, float3(ch.crot0.z, ch.crot1.z, ch.crot2.z)));
    b.half = ch.half.xyz + grow;
    return b;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void plan_motion(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count)
        return;
    ChunkStatic ch = chunks[k];
    Island isl = islands[ch.info.y];
    Quat q = quat_of(isl.rotation);
    float3 local = ch.center.xyz + state[4 * k].xyz;
    float3 rel = rotate(q, local);
    float3 pos = (isl.position.xyz + isl.position_err.xyz) + rel;
    // The same position as a compensated pair (hi + lo; contacts.rs `chunk_plan`'s
    // `split(chunk_position)`): two-sums of position + error + offset.
    float3 a = isl.position.xyz;
    float3 s1 = a + rel;
    float3 bb = s1 - a;
    float3 e1 = (a - (s1 - bb)) + (rel - bb);
    float3 hi = s1 + (e1 + isl.position_err.xyz);
    float3 lo = (e1 + isl.position_err.xyz) - (hi - s1);
    float3 r = rotate(q, local - isl.com.xyz);
    float3 v = (isl.velocity.xyz + isl.velocity_err.xyz) + cross(isl.angular_velocity.xyz, r) + rotate(q, state[4 * k + 2].xyz);
    float3 w = isl.angular_velocity.xyz + rotate(q, state[4 * k + 3].xyz);
    float3 com_world = (isl.position.xyz + isl.position_err.xyz) + rotate(q, isl.com.xyz);
    float radius = length(ch.half.xyz);
    float arm = length(pos - com_world);
    float d = params.duration;
    float budget = 2.0 * length(w) * (arm + radius) * d + params.g * d * d + 0.05 * radius + 1e-3;
    bool falls = (isl.info.x & ISLAND_ANCHORED) == 0;
    pm[5 * k] = float4(pos, radius);
    pm[5 * k + 1] = float4(v, budget);
    pm[5 * k + 2] = float4(w, falls ? 1.0 : 0.0);
    // The chunk's orientation at the plan's start (cluster rotation x hidden rotation;
    // contacts.rs `chunk_plan`'s `start_rot`).
    float3 th = state[4 * k + 1].xyz;
    pm[5 * k + 3] = quat_vec(quat_mul(q, from_axis_angle(th, length(th))));
    pm[5 * k + 4] = float4(hi, 0.0);
    pm[params.reach_base + 2 * params.island_count + k] = float4(lo, 0.0);
}

groupshared float g_spread[64];
groupshared float g_rb[64];
groupshared float g_v[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void plan_reach(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    uint tid = thread.x;
    if (i >= params.island_count)
        return;
    Island isl = islands[i];
    float3 center = isl.range.y > isl.range.x ? pm[5 * isl.range.x].xyz : float3(0.0);
    float spread = 0.0;
    float rb = 0.0;
    float vmax = 0.0;
    for (uint k = isl.range.x + tid; k < isl.range.y; k += 64)
    {
        float4 p = pm[5 * k];
        float4 v = pm[5 * k + 1];
        spread = max(spread, length(p.xyz - center));
        rb = max(rb, p.w + v.w);
        vmax = max(vmax, length(v.xyz));
    }
    g_spread[tid] = spread;
    g_rb[tid] = rb;
    g_v[tid] = vmax;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            g_spread[tid] = max(g_spread[tid], g_spread[tid + s]);
            g_rb[tid] = max(g_rb[tid], g_rb[tid + s]);
            g_v[tid] = max(g_v[tid], g_v[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0)
    {
        pm[params.reach_base + 2 * i] = float4(center, g_spread[0]);
        pm[params.reach_base + 2 * i + 1] = float4(g_rb[0], g_v[0], 0.0, 0.0);
    }
}

// The whole-cluster test of an owned cluster pair (contacts.rs `reach`).
bool clusters_may_touch(uint ia, uint ib)
{
    float4 a0 = pm[params.reach_base + 2 * ia];
    float4 a1 = pm[params.reach_base + 2 * ia + 1];
    float4 b0 = pm[params.reach_base + 2 * ib];
    float4 b1 = pm[params.reach_base + 2 * ib + 1];
    bool anch_a = (islands[ia].info.x & ISLAND_ANCHORED) != 0;
    bool anch_b = (islands[ib].info.x & ISLAND_ANCHORED) != 0;
    float d = params.duration;
    float dg = anch_a == anch_b ? 0.0 : params.g;
    float travel = (a1.y + b1.y) * d;
    float near = length(b0.xyz - a0.xyz) - a0.w - b0.w - 0.5 * travel;
    float limit = a1.x + b1.x + 0.5 * travel + 0.5 * dg * d * d;
    return near <= limit + 1e-5 * (abs(near) + limit + a0.w + b0.w + 1.0);
}

// Whether chunks a and b pass the sweep test, and (when they do) the box test.
bool pair_passes(uint a, uint b, out bool swept)
{
    float4 pa = pm[5 * a];
    float4 va = pm[5 * a + 1];
    float4 pb = pm[5 * b];
    float4 vb = pm[5 * b + 1];
    float d = params.duration;
    float3 dv = vb.xyz - va.xyz;
    float dg = pm[5 * a + 2].w == pm[5 * b + 2].w ? 0.0 : params.g;
    float3 middle = pb.xyz - pa.xyz + dv * (0.5 * d);
    float sweep = 0.5 * length(dv) * d + 0.5 * dg * d * d;
    swept = length(middle) <= pa.w + pb.w + va.w + vb.w + sweep;
    if (!swept)
        return false;
    Box box_a = grown_box(a, pa.xyz, va.w);
    Box box_b = grown_box(b, pa.xyz + middle, vb.w + sweep);
    return may_overlap(box_a, box_b);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void plan_pair_count(uint3 id: SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= params.job_count)
        return;
    uint p = pu[params.job_base + 2 * j];
    uint a = pu[params.job_base + 2 * j + 1];
    uint ia = pu[params.cp_base + 2 * p];
    uint ib = pu[params.cp_base + 2 * p + 1];
    uint count = 0;
    if (clusters_may_touch(ia, ib))
    {
        Island isl = islands[ib];
        for (uint b = isl.range.x; b < isl.range.y; b++)
        {
            bool swept;
            bool kept = pair_passes(a, b, swept);
            if (swept)
            {
                pu[params.flag_base + a] |= 1u;
                pu[params.flag_base + b] |= 1u;
            }
            if (kept)
                count++;
        }
    }
    pu[params.count_base + j] = count;
}

groupshared uint g_scan[1024];

[shader("compute")]
[numthreads(1024, 1, 1)]
void plan_scan(uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint n = params.job_count;
    uint carry = 0;
    for (uint base = 0; base < n; base += 1024)
    {
        uint k = base + tid;
        uint v = k < n ? pu[params.count_base + k] : 0;
        g_scan[tid] = v;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 1; step < 1024; step <<= 1)
        {
            uint add = tid >= step ? g_scan[tid - step] : 0;
            GroupMemoryBarrierWithGroupSync();
            g_scan[tid] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (k < n)
            pu[params.count_base + n + 1 + k] = carry + g_scan[tid] - v;
        uint total = g_scan[1023];
        GroupMemoryBarrierWithGroupSync();
        carry += total;
    }
    if (tid == 0)
        pu[params.count_base + n] = carry;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void plan_pair_write(uint3 id: SV_DispatchThreadID)
{
    uint j = id.x;
    uint n = params.job_count;
    if (j >= n || pu[params.count_base + j] == 0)
        return;
    uint p = pu[params.job_base + 2 * j];
    uint a = pu[params.job_base + 2 * j + 1];
    uint ib = pu[params.cp_base + 2 * p + 1];
    uint at = pu[params.count_base + n + 1 + j];
    Island isl = islands[ib];
    for (uint b = isl.range.x; b < isl.range.y; b++)
    {
        bool swept;
        if (pair_passes(a, b, swept))
        {
            pu[params.pair_base + 2 * at] = a;
            pu[params.pair_base + 2 * at + 1] = b;
            at++;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void plan_side(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.chunk_count)
        return;
    ChunkStatic ch = chunks[k];
    Island isl = islands[ch.info.y];
    bool anchored = (isl.info.x & ISLAND_ANCHORED) != 0;
    if (pu[params.owned_base + ch.info.y] == 0)
        return;
    float4 p = pm[5 * k];
    float4 v = pm[5 * k + 1];
    bool falls = pm[5 * k + 2].w != 0.0;
    float d = params.duration;
    uint flags = 0;
    if (params.has_ground != 0 && !anchored)
    {
        float z0 = p.z;
        float vz = v.z;
        float az = falls ? params.gz : 0.0;
        float t_low = az != 0.0 ? clamp(-vz / az, 0.0, d) : 0.0;
        float lowest = min(min(z0, z0 + vz * d + 0.5 * az * d * d), z0 + vz * t_low + 0.5 * az * t_low * t_low);
        if (lowest - p.w - v.w <= params.ground_height)
            flags |= 2u;
    }
    float chunk_reach = length(v.xyz) * d + (falls ? 0.5 * params.g * d * d : 0.0) + v.w;
    for (uint i = 0; i < params.impactor_count && i < 30; i++)
    {
        Impactor imp = impactors[i];
        if (imp.cand.z != 0)
            continue;
        float reach = imp.shape.x == 0.0 ? imp.shape.y * 1.7320508 : length(imp.half.xyz);
        float imp_budget = 2.0 * (length(imp.velocity.xyz) + length(imp.angular_velocity.xyz) * reach) * d + params.g * d * d + 1e-3;
        float3 ip = imp.position.xyz + imp.position_err.xyz;
        if (length(p.xyz - ip) <= reach + p.w + chunk_reach + imp_budget)
            flags |= 4u << i;
    }
    pu[params.flag_base + k] |= flags;
}
