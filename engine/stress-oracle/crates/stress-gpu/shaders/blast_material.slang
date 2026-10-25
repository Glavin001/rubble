// Blast's bond stresses and damage after a solve (blast_cgls.slang): the shared formula of the
// PhysX fork (NvBlastExtStressFormula.h `extStressCalcBondStress`, NvBlastExtStressMaterialFormula.h
// `extStressFibre` and `extStressBondDamage`), one bond slot a thread. Its f64 reference is
// stress-ref blast_bond.rs `bond_stress` and `bond_damage`.
//
//   blast_bond_stress   the bond's physical impulse (x s, times length mass or length^2
//                       mass), its stresses over the remaining area, and its damage verdict;
//                       with apply_damage, the damage is taken from the bond's health, and a
//                       bond at 0 is broken (its impulse cleared).

import blast_common;

[[vk::binding(0, 0)]] ConstantBuffer<BlastParams> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<BlastBond> bonds;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> x;
// Per bond slot: (unit normal from node 0 to node 1, node distance), (authored area,
// material index as bits, 0, 0).
[[vk::binding(3, 0)]] StructuredBuffer<float4> geometry;
// Per material: (compression elastic, fatal, tension elastic, fatal), (shear elastic, fatal,
// residual area fraction, 0).
[[vk::binding(4, 0)]] StructuredBuffer<float4> materials;
// Per bond slot: impulse angular, linear; (normal, shear, bend, compression);
// (tension, damage, multiplier, command).
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> out;

// |v - (v.n) n| with fused components (extStressPerpendicularMagnitude).
float perpendicular(float3 v, float3 n, float along)
{
    float px = fma(-along, n.x, v.x);
    float py = fma(-along, n.y, v.y);
    float pz = fma(-along, n.z, v.z);
    return sqrt(fma(pz, pz, fma(py, py, px * px)));
}

// extStressDot.
float fdot(float3 a, float3 b)
{
    return fma(a.z, b.z, fma(a.y, b.y, a.x * b.x));
}

// The excess of `value` over `elastic` in the band to `fatal` (a denominator <= 0 is 1).
float excess(float value, float elastic, float fatal)
{
    float d = fatal - elastic;
    return (value - elastic) / (d > 0.0 ? d : 1.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void blast_bond_stress(uint3 id : SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= params.bond_count)
        return;
    BlastBond b = bonds[j];
    float3 ia = x[j * 2].xyz * (b.s * params.angular_scale);
    float3 il = x[j * 2 + 1].xyz * (b.s * params.linear_scale);
    out[j * 4] = float4(ia, 0.0);
    out[j * 4 + 1] = float4(il, 0.0);
    float area = b.health;
    if (!(area > 0.0))
    {
        out[j * 4 + 2] = float4(0.0);
        out[j * 4 + 3] = float4(0.0);
        return;
    }
    float4 g0 = geometry[j * 2];
    float4 g1 = geometry[j * 2 + 1];
    float3 n = g0.xyz;
    float node_distance = g0.w;

    // extStressCalcBondStress.
    float linear_normal = fdot(il, n);
    float normal = linear_normal / area;
    float shear = perpendicular(il, n, linear_normal) / area;
    float angular_normal = fdot(ia, n);
    float twist = abs(angular_normal) / area;
    float bend = perpendicular(ia, n, angular_normal) / area;
    float gain_max = params.bend_gain_max;
    float bend_stress = 0.0;
    if (gain_max > 0.0)
    {
        float torsion = 4.81 / sqrt(area > 1.0e-6 ? area : 1.0e-6);
        shear = fma(twist, torsion < gain_max ? torsion : gain_max, shear);
        float section = 6.0 / sqrt(area > 1.0e-6 ? area : 1.0e-6);
        bend_stress = bend * (section < gain_max ? section : gain_max);
    }
    else
    {
        shear += twist * 2.0 / node_distance;
        // copysign(bend 2 / d, normal), with the sign of a signed zero kept.
        float folded = bend * 2.0 / node_distance;
        normal += (asuint(normal) & 0x80000000u) != 0 ? -folded : folded;
    }
    // extStressFibre.
    float compression;
    float tension;
    if (params.fibres == 0 || gain_max <= 0.0)
    {
        float combined = normal + ((asuint(normal) & 0x80000000u) != 0 ? -bend_stress : bend_stress);
        compression = combined <= 0.0 ? -combined : 0.0;
        tension = combined > 0.0 ? combined : 0.0;
    }
    else
    {
        float t = normal + bend_stress;
        float c = bend_stress - normal;
        tension = t > 0.0 ? t : 0.0;
        compression = c > 0.0 ? c : 0.0;
    }
    out[j * 4 + 2] = float4(normal, shear, bend_stress, compression);

    // extStressBondDamage.
    uint m = asuint(g1.y);
    float4 m0 = materials[m * 2];
    float4 m1 = materials[m * 2 + 1];
    float axial = 0.0;
    if (compression > m0.x)
        axial = excess(compression, m0.x, m0.y);
    if (tension > m0.z)
    {
        float t = excess(tension, m0.z, m0.w);
        axial = axial < t ? t : axial;
    }
    float multiplier = axial;
    if (shear > m1.x)
        multiplier += excess(shear, m1.x, m1.y);
    float damage = 0.0;
    bool command = multiplier > 0.0;
    if (command)
    {
        if (multiplier >= 1.0)
            damage = area;
        else if (params.dt > 0.0)
        {
            float fraction = multiplier * params.dt * params.damage_rate;
            damage = area * (fraction < 1.0 ? fraction : 1.0);
        }
        else
            damage = area * multiplier;
        if (multiplier < 1.0 && m1.z > 0.0)
        {
            float floor = g1.x * m1.z;
            if (area - damage < floor)
                damage = area > floor ? area - floor : 0.0;
        }
    }
    out[j * 4 + 3] = float4(tension, damage, multiplier, command ? 1.0 : 0.0);
    if (params.apply_damage != 0 && command)
    {
        float health = area - damage;
        bonds[j].health = health > 0.0 ? health : 0.0;
        if (!(health > 0.0))
        {
            x[j * 2] = float4(0.0);
            x[j * 2 + 1] = float4(0.0);
        }
    }
}
