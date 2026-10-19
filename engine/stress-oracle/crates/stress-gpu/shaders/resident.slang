// The resident quasi-static frame (docs/stress-fracture-gpu-resident.md, phases 2b and
// 4): from the engine's contact manifolds (coupling.slang) to the static solve's loads
// without the host.
//
//   resident_poses        one thread per island: a driven cluster's pose and velocities
//                         at the frame's start, from the engine's body (world_coupling.rs
//                         `set_driven_motion` at the frame start, rubble's `motion_of`);
//   resident_filter       one thread per chunk: coupled.rs `LoadFilter::ingest` and
//                         `frame_average_terms` (sustained loads, impact pulses averaged
//                         over the frame, pulses carried to later frames) and
//                         world.rs `host_frame_loads` (force, and its moment about the
//                         chunk's displaced centre, world frame);
//   resident_static_loads one threadgroup per island flagged ISLAND_SOLVE: statics.rs
//                         `static_loads` (gravity, the rigid inertial loads, the body
//                         frame) into the static solve's load slots (`sload_base`).
//
// Frames: the engine is Y-up, the stress solver Z-up: (x, y, z) -> (x, -z, y).

import common;

struct ResidentParams
{
    uint chunk_count;
    uint island_count;
    uint sload_base;      // scratch: the static solve's loads (2 float4 per chunk)
    uint rml;             // rigid_motion_loads
    float4 gravity;       // xyz; w: |g| dt (the filter's resolution band)
    float dt;             // the frame
    uint props_base;      // ctab: per chunk (E, poisson, cbrt(volume), -)
    uint engine_base;     // ctab: per island 3 float4: (body or NONE as bits, -, -, -), P^-1 (quaternion x, y, z, w), com (body frame)
    uint eload_base;      // ctab: per chunk 2 float4: world force, world moment about the displaced centre
    uint pulse_base;      // ctab: per chunk `pulse_cap` pulses of 3 float4
    uint pulse_count_base;// ctab: per chunk its pulse count (x, as bits)
    uint pulse_cap;
    uint status_base;     // ctab: x = chunks whose carried pulses overflowed (bits)
    uint impact_peak;     // 1: this frame's impacts at their Hertz peak force (a measurement)
    uint pad0;
    uint pad1;
};

struct Record
{
    uint4 ids;
    float4 point;
    float4 impulse;
    float4 other;
    uint4 key; // coupling.slang `Record::key`
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

[[vk::binding(0, 0)]] ConstantBuffer<ResidentParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<Record> manifolds;
[[vk::binding(2, 0)]] StructuredBuffer<uint> sort;
[[vk::binding(3, 0)]] RWStructuredBuffer<Island> islands;
[[vk::binding(4, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(5, 0)]] StructuredBuffer<float4> state;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> scratch;
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> ctab;
[[vk::binding(8, 0)]] StructuredBuffer<EngineBody> start_bodies;

static const uint NONE = 0xffffffffu;
static const uint ISLAND_SOLVE = 8;
static const float PI = 3.14159265358979;

float3 to_zup(float3 v)
{
    return float3(v.x, -v.z, v.y);
}

float3 mat_mul(float4 r0, float4 r1, float4 r2, float3 v)
{
    return float3(dot(r0.xyz, v), dot(r1.xyz, v), dot(r2.xyz, v));
}

// R M R^T v for a body-frame matrix M (rows).
float3 world_mul(Quat q, float4 r0, float4 r1, float4 r2, float3 v)
{
    return rotate(q, mat_mul(r0, r1, r2, inverse_rotate(q, v)));
}

Quat island_rotation(Island isl)
{
    return quat_of(isl.rotation);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void resident_poses(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.island_count)
        return;
    float4 e = ctab[params.engine_base + 3 * i];
    uint body = asuint(e.x);
    if (body == NONE)
        return;
    Quat p_inv = quat_of(ctab[params.engine_base + 3 * i + 1]);
    float3 com = ctab[params.engine_base + 3 * i + 2].xyz;
    EngineBody b = start_bodies[body];
    // rubble's `rot_from_rubble`: axes()^-1 q (axes: -90 degrees about x).
    float h = 0.70710678118;
    Quat a_inv;
    a_inv.w = h;
    a_inv.x = h;
    a_inv.y = 0.0;
    a_inv.z = 0.0;
    Quat q = normalized(quat_mul(quat_mul(a_inv, quat_of(b.rot)), p_inv));
    float3 com_world = to_zup(b.pos_inv_mass.xyz);
    Island isl = islands[i];
    isl.rotation = quat_vec(q);
    isl.rotation_err = float4(0.0);
    isl.position = float4(com_world - rotate(q, com), 0.0);
    isl.position_err = float4(0.0);
    isl.velocity = float4(to_zup(b.vel_sleep.xyz), 0.0);
    isl.velocity_err = float4(0.0);
    float3 w = to_zup(b.ang_vel_gen.xyz);
    isl.angular_velocity = float4(w, 0.0);
    isl.momentum = float4(world_mul(q, isl.inertia0, isl.inertia1, isl.inertia2, w), 0.0);
    isl.momentum_err = float4(0.0);
    islands[i] = isl;
}

// api.rs `hertz_duration`.
float hertz_duration(float m, float r, float e, float v)
{
    float va = max(abs(v), 1e-3);
    return 2.87 * pow(m * m / (r * e * e * va), 0.2);
}

// coupled.rs `pulse_integral` with no plateau over [a, b], times relative to the frame.
float pulse_mean(float start, float plateau, float plateau_time, float peak, float duration, float dt)
{
    float sum = plateau * max(min(start + plateau_time, dt) - max(start, 0.0), 0.0);
    float s0 = start + plateau_time;
    float lo = max(0.0, s0);
    float hi = min(dt, s0 + duration);
    if (hi > lo && duration > 0.0)
    {
        float w = PI / duration;
        sum += peak / w * (cos(w * (lo - s0)) - cos(w * (hi - s0)));
    }
    return sum / dt;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void resident_filter(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    uint n = params.chunk_count;
    if (k >= n)
        return;
    ChunkStatic ch = chunks[k];
    Island isl = islands[ch.info.y];
    bool anchored = (isl.info.x & ISLAND_ANCHORED) != 0;
    Quat q = island_rotation(isl);
    float3 centre = isl.position.xyz + rotate(q, ch.center.xyz + state[4 * k].xyz);
    float4 props = ctab[params.props_base + k];
    float m_body = anchored ? float_bits(0x7f800000u) : isl.com.w;
    float dt = params.dt;
    float3 f_total = float3(0.0);
    float3 t_total = float3(0.0);
    uint begin = sort[k];
    uint count = sort[n + 1 + k];
    // Sustained loads in the order of their key (other body; static geometry first),
    // as the host's BTreeMap holds them.
    uint last = 0;
    bool first = true;
    for (uint step = 0; step < count; step++)
    {
        uint best = NONE;
        uint best_key = 0;
        for (uint q2 = 0; q2 < count; q2++)
        {
            uint key = manifolds[begin + q2].ids.y + 1; // NONE -> 0
            if ((first || key > last) && (best == NONE || key < best_key))
            {
                best = q2;
                best_key = key;
            }
        }
        if (best == NONE)
            break;
        first = false;
        last = best_key;
        Record r = manifolds[begin + best];
        float j = length(r.impulse.xyz);
        if (j == 0.0)
            continue;
        float m_other = r.other.x;
        float m = isinf(m_body) ? (isinf(m_other) ? ch.center.w : m_other) : (isinf(m_other) ? m_body : m_body * m_other / (m_body + m_other));
        float j_impact = min(j, 2.0 * m * max(r.impulse.w - params.gravity.w, 0.0));
        if (j_impact < j)
        {
            float3 f = r.impulse.xyz / j * ((j - j_impact) / dt);
            f_total += f;
            t_total += cross(r.point.xyz - centre, f);
        }
    }
    // Pulses carried from earlier frames, in their order; then this frame's, in
    // manifold order. Survivors are written back in the same order.
    uint at = params.pulse_base + 3 * params.pulse_cap * k;
    uint carried = asuint(ctab[params.pulse_count_base + k].x);
    uint kept = 0;
    bool overflow = false;
    for (uint p = 0; p < carried; p++)
    {
        float4 a = ctab[at + 3 * p];
        float4 b = ctab[at + 3 * p + 1];
        float4 c = ctab[at + 3 * p + 2];
        float mean = pulse_mean(a.x, a.y, a.z, a.w, b.w, dt);
        if (mean != 0.0)
        {
            float3 f = c.xyz * mean;
            f_total += f;
            t_total += cross(b.xyz - centre, f);
        }
        if (a.x + a.z + b.w > dt)
        {
            a.x -= dt;
            ctab[at + 3 * kept] = a;
            ctab[at + 3 * kept + 1] = b;
            ctab[at + 3 * kept + 2] = c;
            kept++;
        }
    }
    for (uint q3 = 0; q3 < count; q3++)
    {
        Record r = manifolds[begin + q3];
        float j = length(r.impulse.xyz);
        if (j == 0.0)
            continue;
        float m_other = r.other.x;
        float m = isinf(m_body) ? (isinf(m_other) ? ch.center.w : m_other) : (isinf(m_other) ? m_body : m_body * m_other / (m_body + m_other));
        float j_impact = min(j, 2.0 * m * max(r.impulse.w - params.gravity.w, 0.0));
        if (j_impact <= 0.0)
            continue;
        float e_star = 1.0 / ((1.0 - props.y * props.y) / props.x + 1.0 / r.other.y);
        float radius = min(r.other.z, props.z);
        float duration = hertz_duration(m, radius, e_star, r.impulse.w);
        float peak = PI * j_impact / (2.0 * duration);
        float3 dir = r.impulse.xyz / j;
        float mean = params.impact_peak != 0 ? peak : pulse_mean(0.0, 0.0, 0.0, peak, duration, dt);
        if (mean != 0.0)
        {
            float3 f = dir * mean;
            f_total += f;
            t_total += cross(r.point.xyz - centre, f);
        }
        if (duration > dt)
        {
            if (kept < params.pulse_cap)
            {
                ctab[at + 3 * kept] = float4(-dt, 0.0, 0.0, peak);
                ctab[at + 3 * kept + 1] = float4(r.point.xyz, duration);
                ctab[at + 3 * kept + 2] = float4(dir, 0.0);
                kept++;
            }
            else
            {
                overflow = true;
            }
        }
    }
    ctab[params.pulse_count_base + k] = float4(asfloat(kept), 0.0, 0.0, 0.0);
    if (overflow)
        ctab[params.status_base] = float4(asfloat(1u), 0.0, 0.0, 0.0);
    ctab[params.eload_base + 2 * k] = float4(f_total, 0.0);
    ctab[params.eload_base + 2 * k + 1] = float4(t_total, 0.0);
}

groupshared float3 g_f[256];
groupshared float3 g_t[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void resident_static_loads(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint i = group.x;
    uint tid = thread.x;
    if (i >= params.island_count)
        return;
    Island isl = islands[i];
    // The exit as a uniform value: barriers follow.
    if (group_uniform((isl.info.z & ISLAND_SOLVE) == 0 ? 1u : 0u, tid) != 0)
        return;
    bool anchored = (isl.info.x & ISLAND_ANCHORED) != 0;
    Quat q = island_rotation(isl);
    float3 w = isl.angular_velocity.xyz;
    float3 g = params.gravity.xyz;
    float3 com_w = isl.position.xyz + rotate(q, isl.com.xyz);
    // statics.rs `net_load`: the cluster's net force and torque about its centre of mass.
    float3 f = float3(0.0);
    float3 t = float3(0.0);
    if (!anchored)
    {
        for (uint c = isl.range.x + tid; c < isl.range.y; c += 256)
        {
            ChunkStatic ch = chunks[c];
            float3 fc = ctab[params.eload_base + 2 * c].xyz + g * ch.center.w;
            float3 x = isl.position.xyz + rotate(q, ch.center.xyz + state[4 * c].xyz);
            f += fc;
            t += cross(x - com_w, fc) + ctab[params.eload_base + 2 * c + 1].xyz;
        }
    }
    g_f[tid] = f;
    g_t[tid] = t;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            g_f[tid] += g_f[tid + s];
            g_t[tid] += g_t[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float3 a = float3(0.0);
    float3 alpha = float3(0.0);
    if (!anchored)
    {
        float3 iw_w = world_mul(q, isl.inertia0, isl.inertia1, isl.inertia2, w);
        alpha = world_mul(q, isl.inv0, isl.inv1, isl.inv2, g_t[0] - cross(w, iw_w));
        a = g_f[0] / isl.com.w;
    }
    bool rml = params.rml != 0;
    float3 wb = inverse_rotate(q, w);
    for (uint c = isl.range.x + tid; c < isl.range.y; c += 256)
    {
        ChunkStatic ch = chunks[c];
        float mass = ch.center.w;
        float3 u = state[4 * c].xyz;
        float3 r_world = rotate(q, ch.center.xyz + u - isl.com.xyz);
        float3 f_world = ctab[params.eload_base + 2 * c].xyz + g * mass;
        float3 t_world = ctab[params.eload_base + 2 * c + 1].xyz;
        if (rml)
        {
            f_world -= (a + cross(alpha, r_world) + cross(w, cross(w, r_world))) * mass;
            t_world -= world_mul(q, ch.inertia0, ch.inertia1, ch.inertia2, alpha) + cross(w, world_mul(q, ch.inertia0, ch.inertia1, ch.inertia2, w));
        }
        float3 fb = inverse_rotate(q, f_world);
        float3 mo = inverse_rotate(q, t_world);
        if (rml)
        {
            float3 v = state[4 * c + 2].xyz;
            float3 sw = state[4 * c + 3].xyz;
            fb -= cross(wb, v) * (2.0 * mass);
            mo -= cross(wb, mat_mul(ch.inertia0, ch.inertia1, ch.inertia2, sw)) + cross(sw, mat_mul(ch.inertia0, ch.inertia1, ch.inertia2, wb)) + cross(sw, mat_mul(ch.inertia0, ch.inertia1, ch.inertia2, sw));
        }
        scratch[params.sload_base + 2 * c] = float4(fb, 0.0);
        scratch[params.sload_base + 2 * c + 1] = float4(mo, 0.0);
    }
}
