// The chunks' bond virial from Blast's solved forces, for chunk crushing in Blast's frame
// (PhysX's evaluateChunkMaterials: sum over a chunk's live bonds of sym((c - p) (x) F), c the
// bond's centroid, p the chunk's position, F the bond's linear force on the chunk: +lambda on
// its first node, -lambda on its second). Recorded after the solve and before
// blast_material.slang applies the frame's damage, so over the bonds that were live as the
// solve met them; fracture.slang `fr_crush_loads` adds its contacts' virial and steps the law.
//
//   blast_virial   one thread per node: its virial, structure frame, into the fracture's
//                  `crush_rw` at params.r_virial (2 float4: diagonal, off-diagonal xy xz yz).

import blast_common;

[[vk::binding(0, 0)]] ConstantBuffer<BlastParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<BlastNode> nodes;
[[vk::binding(2, 0)]] StructuredBuffer<uint> refs;
[[vk::binding(3, 0)]] StructuredBuffer<BlastBond> bonds;
[[vk::binding(4, 0)]] StructuredBuffer<float4> x;
// blast_native_ops.slangh's per-bond stiffness (A, W, Al, Wl; params.rotation).
[[vk::binding(5, 0)]] StructuredBuffer<float4> stiffness;
// Per bond its centroid (structure frame, m); per node (Blast's position, mass) first of its
// NODE_FRAME rows (blast_fracture.slang `node_frame`).
[[vk::binding(6, 0)]] StructuredBuffer<float4> centroids;
[[vk::binding(7, 0)]] StructuredBuffer<float4> node_frame;
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> crush_rw;

static const uint NODE_FRAME = 6;

// Bond j's Al times v (the shear stiffness's linear block; params.rotation 2).
float3 linear_block(uint j, float3 v)
{
    float m[6];
    for (uint k = 0; k < 6; k++)
    {
        float4 q = stiffness[j * 6 + (12 + k) / 4];
        m[k] = q[(12 + k) % 4];
    }
    return float3(m[0] * v.x + m[3] * v.y + m[4] * v.z, m[3] * v.x + m[1] * v.y + m[5] * v.z, m[4] * v.x + m[5] * v.y + m[2] * v.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void blast_virial(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.node_count)
        return;
    float3 d = float3(0.0);
    float3 o = float3(0.0);
    float4 node = node_frame[NODE_FRAME * k];
    BlastNode n = nodes[k];
    if (node.w > 0.0)
    {
        for (uint r = n.ref_begin; r < n.ref_end; r++)
        {
            uint ref = refs[r];
            uint j = ref & ~REF_SECOND;
            BlastBond b = bonds[j];
            if (!(b.health > 0.0))
                continue;
            float3 xl = x[j * 2 + 1].xyz;
            if (params.rotation == 2)
                xl = linear_block(j, xl);
            float3 f = xl * (b.s * params.linear_scale);
            if ((ref & REF_SECOND) != 0)
                f = -f;
            float3 a = centroids[j].xyz - node.xyz;
            d += a * f;
            o += 0.5 * float3(a.x * f.y + a.y * f.x, a.x * f.z + a.z * f.x, a.y * f.z + a.z * f.y);
        }
    }
    crush_rw[params.r_virial + 2 * k] = float4(d, 0.0);
    crush_rw[params.r_virial + 2 * k + 1] = float4(o, 0.0);
}
