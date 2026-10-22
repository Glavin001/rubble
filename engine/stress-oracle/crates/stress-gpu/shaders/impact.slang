// Impact pulses as the coupling turns contact impulses into loads (resident.slang,
// fracture.slang): a Hertz contact's duration, and a half-sine pulse's mean over a frame.

import common;

public static const float PI = 3.14159265358979;
static const uint NO_MANIFOLD = 0xffffffffu;

// api.rs `hertz_duration`.
public float hertz_duration(float m, float r, float e, float v)
{
    float va = max(abs(v), 1e-3);
    return 2.87 * pow(m * m / (r * e * e * va), 0.2);
}

// coupled.rs `pulse_integral` with no plateau over [a, b], times relative to the frame.
public float pulse_mean(float start, float plateau, float plateau_time, float peak, float duration, float dt)
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

// A chunk's contact manifold (coupling.slang `Record`).
public struct Record
{
    public uint4 ids;
    public float4 point;
    public float4 impulse;
    public float4 other;
    public uint4 key; // coupling.slang `Record::key`
};

// A chunk's engine load over the frame (coupled.rs `LoadFilter::ingest` and
// `frame_average_terms`, world.rs `host_frame_loads`): from its contact manifolds
// (coupling.slang, `sort`'s ranges), the sustained part of each impulse as a force, and
// each impact as a Hertz pulse: at its peak (`impact_peak`) or its mean over the frame.
// Pulses that outlast the frame are carried to the next in `store` (3 float4 each): the
// `carried` from word `src`, the survivors and this frame's to `dst` (room for `cap`; the
// same range as `src` when carried in place: a survivor is never written past the one it
// came from). The force and its moment about `centre` (world), the pulses kept, and
// whether one found no room.
[ForceInline]
public void chunk_engine_load(StructuredBuffer<Record> manifolds, StructuredBuffer<uint> sort, RWStructuredBuffer<float4> store, uint n, uint k, float m_body, float chunk_mass, float3 centre, float4 props, float band, float dt, bool impact_peak, uint src, uint carried, uint dst, uint cap, out float3 f_total, out float3 t_total, out uint kept, out bool overflow)
{
    f_total = float3(0.0);
    t_total = float3(0.0);
    uint begin = sort[k];
    uint count = sort[n + 1 + k];
    // Sustained loads in the order of their key (other body; static geometry first),
    // as the host's BTreeMap holds them.
    uint last = 0;
    bool first = true;
    for (uint step = 0; step < count; step++)
    {
        uint best = NO_MANIFOLD;
        uint best_key = 0;
        for (uint q2 = 0; q2 < count; q2++)
        {
            uint key = manifolds[begin + q2].ids.y + 1; // NO_MANIFOLD -> 0
            if ((first || key > last) && (best == NO_MANIFOLD || key < best_key))
            {
                best = q2;
                best_key = key;
            }
        }
        if (best == NO_MANIFOLD)
            break;
        first = false;
        last = best_key;
        Record r = manifolds[begin + best];
        float j = length(r.impulse.xyz);
        if (j == 0.0)
            continue;
        float m_other = r.other.x;
        float m = isinf(m_body) ? (isinf(m_other) ? chunk_mass : m_other) : (isinf(m_other) ? m_body : m_body * m_other / (m_body + m_other));
        float j_impact = min(j, 2.0 * m * max(r.impulse.w - band, 0.0));
        if (j_impact < j)
        {
            float3 f = r.impulse.xyz / j * ((j - j_impact) / dt);
            f_total += f;
            t_total += cross(r.point.xyz - centre, f);
        }
    }
    // Pulses carried from earlier frames, in their order; then this frame's, in
    // manifold order. Survivors are written back in the same order.
    kept = 0;
    overflow = false;
    for (uint p = 0; p < carried; p++)
    {
        float4 a = store[src + 3 * p];
        float4 b = store[src + 3 * p + 1];
        float4 c = store[src + 3 * p + 2];
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
            store[dst + 3 * kept] = a;
            store[dst + 3 * kept + 1] = b;
            store[dst + 3 * kept + 2] = c;
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
        float m = isinf(m_body) ? (isinf(m_other) ? chunk_mass : m_other) : (isinf(m_other) ? m_body : m_body * m_other / (m_body + m_other));
        float j_impact = min(j, 2.0 * m * max(r.impulse.w - band, 0.0));
        if (j_impact <= 0.0)
            continue;
        float e_star = 1.0 / ((1.0 - props.y * props.y) / props.x + 1.0 / r.other.y);
        float radius = min(r.other.z, props.z);
        float duration = hertz_duration(m, radius, e_star, r.impulse.w);
        float peak = PI * j_impact / (2.0 * duration);
        float3 dir = r.impulse.xyz / j;
        float mean = impact_peak ? peak : pulse_mean(0.0, 0.0, 0.0, peak, duration, dt);
        if (mean != 0.0)
        {
            float3 f = dir * mean;
            f_total += f;
            t_total += cross(r.point.xyz - centre, f);
        }
        if (duration > dt)
        {
            if (kept < cap)
            {
                store[dst + 3 * kept] = float4(-dt, 0.0, 0.0, peak);
                store[dst + 3 * kept + 1] = float4(r.point.xyz, duration);
                store[dst + 3 * kept + 2] = float4(dir, 0.0);
                kept++;
            }
            else
            {
                overflow = true;
            }
        }
    }
}
