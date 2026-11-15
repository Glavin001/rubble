// The stress view of Blast's stress solver in the GPU fracture's frame (`Profile::BlastGpu`,
// opt-in: `GpuFracture::stress_view`), after blast_material.slang's bond stresses and before
// the frame's breaks: per node, the largest stress over its fatal limit among its intact
// bonds (compression, tension, shear as `blast_bond_stress` leaves them in `out`; the joint
// friction's grading of the shear is not applied). For a host to draw; nothing reads it
// back into the solve, so the frame's damage and breaks are the same with it or without.
//
//   blast_view_clear   one thread per node;
//   blast_view         one thread per bond.

import blast_common;

[[vk::binding(0, 0)]] ConstantBuffer<BlastParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<BlastBond> bonds;
// Per bond 4 float4 (blast_material.slang): ..., (normal, shear, bend, compression),
// (tension, damage, multiplier, command).
[[vk::binding(2, 0)]] StructuredBuffer<float4> out;
// Per bond 2 float4; the material index as bits in the second's y.
[[vk::binding(3, 0)]] StructuredBuffer<float4> geometry;
// Per material 3 float4: (compression elastic, fatal, tension elastic, fatal); (shear
// elastic, fatal, ...).
[[vk::binding(4, 0)]] StructuredBuffer<float4> materials;
// Per node: the ratio as ordered float bits.
[[vk::binding(5, 0)]] RWStructuredBuffer<Atomic<uint>> view;

[shader("compute")]
[numthreads(64, 1, 1)]
void blast_view_clear(uint3 id : SV_DispatchThreadID)
{
    if (id.x < params.node_count)
        view[id.x].store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void blast_view(uint3 id : SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= params.bond_count)
        return;
    BlastBond bond = bonds[j];
    if (!(bond.health > 0.0))
        return;
    float4 s = out[j * 4 + 2];
    float tension = out[j * 4 + 3].x;
    uint m = asuint(geometry[j * 2 + 1].y);
    float4 m0 = materials[m * 3];
    float4 m1 = materials[m * 3 + 1];
    float u = 0.0;
    if (m0.y > 0.0)
        u = max(u, s.w / m0.y);
    if (m0.w > 0.0)
        u = max(u, tension / m0.w);
    if (m1.y > 0.0)
        u = max(u, s.y / m1.y);
    uint bits = asuint(max(u, 0.0));
    view[bond.n0].max(bits);
    view[bond.n1].max(bits);
}
