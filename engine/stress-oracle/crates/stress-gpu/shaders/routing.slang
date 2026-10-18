// The routed resident frame's check (engine/docs/stress-baseline-goal.md, M5): at the end of
// a resident frame, whether the next frame may route an island, from the engine's bodies
// where the frame left them. Under the tiered coupling only an impactor's contact seeds an
// island (world_coupling.rs `select_islands`): a pair whose swept bounds overlap, closing
// faster than the tiered impact speed, with a bonded cluster on one side. This flags any
// (impactor, island) pair that passes a superset of that test:
//   - the island's bounds as the host's `Part` takes them (each chunk's centre, displaced,
//     at the island's pose, +- its bounding radius), its sweep with twice its reach;
//   - the impactor's as a sphere about its body's centre;
//   - the closing speed bounded by the relative speed plus both spins' surface speeds
//     (`closing_speed` projects the relative velocity on a normal);
//   - the rigid-assumption test (which can only reject a pair) left out.
// A frame after a flag takes the coupled path, whose host routing decides exactly.
//
//   route_islands  one group per live island (indirect, `island_args`): its bounds and
//                  reach reduced over its chunks; thread 0 tests it against every impactor
//
// Frames: the engine is Y-up, the stress solver Z-up: (x, y, z) -> (x, -z, y).

import common;

struct RouteParams
{
    uint island_count;   // dispatch bound (the island capacity)
    uint impactor_count;
    uint engine_base;    // ctab: per island 3 float4 (resident.slang)
    uint pad0;
    float dt;            // the next frame's sweep (the frame)
    float speed;         // the tiered impact speed, with a margin
    float margin;        // m, added to every bound
    float pad1;
};

struct EngineBody
{
    float4 pos_inv_mass;
    float4 rot;
    float4 inv_inertia_flags;
    float4 vel_sleep;
    float4 ang_vel_gen;
    float4 initial_pos;
    float4 initial_rot;
};

[[vk::binding(0, 0)]] ConstantBuffer<RouteParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<Island> islands;
[[vk::binding(2, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(3, 0)]] StructuredBuffer<float4> state;
[[vk::binding(4, 0)]] StructuredBuffer<float4> ctab;
[[vk::binding(5, 0)]] StructuredBuffer<EngineBody> bodies;
// Per impactor: (engine body as bits, bounding radius about the body's centre, -, -).
[[vk::binding(6, 0)]] StructuredBuffer<float4> impactors;
// [0]: 1 when a pair may seed an island (cleared before the check).
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> route;

static const uint NONE = 0xffffffffu;

float3 to_zup(float3 v)
{
    return float3(v.x, -v.z, v.y);
}

groupshared float3 g_lo[64];
groupshared float3 g_hi[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void route_islands(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    uint t = thread.x;
    if (i >= params.island_count || params.impactor_count == 0)
        return;
    Island isl = islands[i];
    // Only a bonded cluster is struck (`Part::struck`); blank records have no bonds.
    if (isl.range.z >= isl.range.w)
        return;
    // Its pose and motion: a driven island's from its engine body now (resident.slang
    // `resident_poses`), any other's from its record.
    uint body = asuint(ctab[params.engine_base + 3 * i].x);
    Quat q = quat_of(isl.rotation);
    float3 origin = isl.position.xyz;
    float3 centre = origin + rotate(q, isl.com.xyz);
    float3 v = isl.velocity.xyz;
    float3 w = isl.angular_velocity.xyz;
    if (body != NONE)
    {
        Quat p_inv = quat_of(ctab[params.engine_base + 3 * i + 1]);
        float3 com = ctab[params.engine_base + 3 * i + 2].xyz;
        EngineBody b = bodies[body];
        float h = 0.70710678118;
        Quat a_inv;
        a_inv.w = h;
        a_inv.x = h;
        a_inv.y = 0.0;
        a_inv.z = 0.0;
        q = normalized(quat_mul(quat_mul(a_inv, quat_of(b.rot)), p_inv));
        centre = to_zup(b.pos_inv_mass.xyz);
        origin = centre - rotate(q, com);
        v = to_zup(b.vel_sleep.xyz);
        w = to_zup(b.ang_vel_gen.xyz);
    }
    // The bounds (`select_islands`: chunk centres +- bounding radius) and the reach
    // about the centre, as a box whose corner distance bounds it.
    float inf = float_bits(0x7f800000u);
    float3 lo = float3(inf);
    float3 hi = float3(-inf);
    for (uint c = isl.range.x + t; c < isl.range.y; c += 64)
    {
        float3 p = origin + rotate(q, chunks[c].center.xyz + state[4 * c].xyz);
        float r = chunks[c].half.w;
        lo = min(lo, p - r);
        hi = max(hi, p + r);
    }
    g_lo[t] = lo;
    g_hi[t] = hi;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32; s > 0; s >>= 1)
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
    // Every chunk sphere lies in the box, so its corner farthest from the centre bounds
    // the reach (`Part::reach`, about the cluster's own centre: twice this covers that).
    float reach = length(max(abs(hi - centre), abs(centre - lo)));
    float dt = params.dt;
    float m = params.margin;
    float3 sweep = abs(v) * dt + float3(length(w) * 2.0 * reach * dt) + float3(m);
    float3 ilo = lo - sweep;
    float3 ihi = hi + sweep;
    uint found = 0;
    for (uint k = 0; k < params.impactor_count; k++)
    {
        float4 imp = impactors[k];
        EngineBody b = bodies[asuint(imp.x)];
        float3 c = to_zup(b.pos_inv_mass.xyz);
        float3 vk = to_zup(b.vel_sleep.xyz);
        float3 wk = to_zup(b.ang_vel_gen.xyz);
        float r = imp.y;
        float3 ext = float3(r) + abs(vk) * dt + float3(length(wk) * r * dt) + float3(m);
        bool overlap = all(c - ext <= ihi) && all(ilo <= c + ext);
        float closing = length(vk - v) + length(wk) * r + length(w) * 2.0 * reach;
        if (overlap && closing > params.speed)
            found++;
    }
    // Every group that finds one writes the same word (no atomics: WGSL has none on a
    // plain buffer, and the check needs a flag).
    if (found > 0)
        route[0] = 1;
}
