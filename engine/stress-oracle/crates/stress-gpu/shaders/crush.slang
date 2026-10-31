// Chunk crushing's law (engine/docs/destruction-reactions.md section 1; stress-ref
// crush.rs, whose `crush_step_f32` mirrors `crush_step` here operation for operation):
//
//   sigma = (1/V) sum sym(branch (x) force)     p = -tr(sigma)/3     q = sqrt(1.5 s:s)
//   crushing needs p > 0
//   excess = max(q - (cohesion + frictionSlope p), p - capPressure)
//   D += excess^2 dt / (crushViscosity crushEnergy)      crushed at D >= 1
//
// Material: m0 = (capPressure, cohesion, frictionSlope, crushEnergy), viscosity.

import common;

// A symmetric tensor: diagonal (xx, yy, zz), off-diagonal (xy, xz, yz).
public struct Sym
{
    public float3 d;
    public float3 o;
};

public Sym sym_zero()
{
    Sym s;
    s.d = float3(0.0);
    s.o = float3(0.0);
    return s;
}

// sym(r (x) f).
public Sym sym_outer(float3 r, float3 f)
{
    Sym s;
    s.d = r * f;
    s.o = 0.5 * float3(r.x * f.y + r.y * f.x, r.x * f.z + r.z * f.x, r.y * f.z + r.z * f.y);
    return s;
}

public Sym sym_add(Sym a, Sym b)
{
    Sym s;
    s.d = a.d + b.d;
    s.o = a.o + b.o;
    return s;
}

// (p, q) of a virial over a volume.
public float2 crush_invariants(Sym v, float volume)
{
    float inv = 1.0 / volume;
    float xx = v.d.x * inv, yy = v.d.y * inv, zz = v.d.z * inv;
    float xy = v.o.x * inv, xz = v.o.y * inv, yz = v.o.z * inv;
    float p = -(xx + yy + zz) / 3.0;
    float dx = xx + p, dy = yy + p, dz = zz + p;
    float ss = dx * dx + dy * dy + dz * dz + 2.0 * (xy * xy + xz * xz + yz * yz);
    return float2(p, sqrt(1.5 * max(ss, 0.0)));
}

// The overstress at (p, q); 0 where nothing accumulates (tension, inside the surface).
public float crush_excess(float2 pq, float4 m0)
{
    if (!(pq.x > 0.0))
        return 0.0;
    float excess = max(pq.y - (m0.y + m0.z * pq.x), pq.x - m0.x);
    return excess > 0.0 ? excess : 0.0;
}

// One step: (damage, p, q, 1 if crushed).
public float4 crush_step(Sym v, float volume, float dt, float4 m0, float viscosity, float damage)
{
    float2 pq = crush_invariants(v, volume);
    float excess = crush_excess(pq, m0);
    if (!(excess > 0.0))
        return float4(damage, pq, 0.0);
    float d = damage + excess * excess * dt / (viscosity * m0.w);
    if (d >= 1.0)
        return float4(1.0, pq, 1.0);
    return float4(d, pq, 0.0);
}
