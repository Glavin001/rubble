// Blast's stress solver in the GPU fracture's frame (fracture.slang, `Profile::BlastGpu`): the
// fracture's chunks and bonds are Blast's nodes and bonds, index for index.
//
//   blast_loads    one thread per chunk, after fr_loads: Blast's input for its node, in its
//                  structure's frame, as vibe-land feeds ExtStressSolver every tick
//                  (NvBlastExtStressPhysX.cpp): gravity (`addGravity`), the engine's contact
//                  force over the node's mass (`addForceAt`, FORCE mode; its moment is not an
//                  input), and the centrifugal load of its body's spin about its centre of
//                  mass, -w x (w x r) (`addCentrifugalAcceleration`). Supports take none (their
//                  weights in D are 0);
//   blast_sync     one thread per bond, first: a bond the fracture broke itself (a predicted
//                  strike, a crushed chunk) is broken in Blast too: one bond state, as the
//                  target has (its health 0, its impulse cleared, the islands marked for
//                  rebuilding);
//   blast_breaks   one thread per bond, after blast_material.slang applied the damage: a bond
//                  whose health reached 0 this frame breaks in the fracture (fr_breaks' marks:
//                  alive cleared, its body dirty, the counters), whose split follows.

import common;
import rigid_frame;
import blast_common;

[[vk::binding(0, 0)]] ConstantBuffer<BlastParams> params;
// fracture.slang's: `work` (each chunk's body slot, the bonds' alive flags), `atom` (dirty marks,
// counters), `frames` (per slot: centre of mass in the structure frame; principal rotation), the
// engine's `bodies` (7 float4 each: position and inverse mass, rotation, inverse inertia,
// linear velocity, angular velocity, ...; Y-up), `loads` (per chunk: |force|, force (world)).
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> work;
[[vk::binding(2, 0)]] RWStructuredBuffer<Atomic<uint>> atom;
[[vk::binding(3, 0)]] StructuredBuffer<float4> frames;
[[vk::binding(4, 0)]] StructuredBuffer<float4> bodies;
[[vk::binding(5, 0)]] StructuredBuffer<float4> loads;
// Per node NODE_FRAME float4: Blast's position (structure frame) and mass; (inertia, 0, 0, 0);
// then the host's chunk load (world): force, impulse, torque, angular impulse (params.chunk_loads).
[[vk::binding(6, 0)]] StructuredBuffer<float4> node_frame;
static const uint NODE_FRAME = 6;
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> accel;
[[vk::binding(8, 0)]] RWStructuredBuffer<BlastBond> bonds;
// blast_topology.slang's `topo` (its pending-rebuild word) and the solver variable.
[[vk::binding(9, 0)]] RWStructuredBuffer<Atomic<uint>> topo;
[[vk::binding(10, 0)]] RWStructuredBuffer<float4> x;

static const uint BODY_WORDS = 7;
static const uint C_FRAME_BROKEN = 0;
static const uint C_BROKEN = 1;

[shader("compute")]
[numthreads(64, 1, 1)]
void blast_loads(uint3 id: SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= params.node_count)
        return;
    uint slot = work[k];
    uint b = BODY_WORDS * slot;
    // The structure frame's rotation in the world (fracture.slang `structure_pose`).
    Quat p_inv = conj(quat_of(frames[2 * slot + 1]));
    Quat q = normalized(quat_mul(rot_from_rubble(quat_of(bodies[b + 1])), p_inv));
    float4 node = node_frame[NODE_FRAME * k];
    float3 f = loads[k].yzw;
    float3 a = params.gravity.xyz + (node.w > 0.0 ? f / node.w : float3(0.0));
    // A host chunk load (prepareLoads): its force over the mass, and the angular input
    // -torque / inertia (Blast's angular coordinates are of the opposite hand); impulses over dt.
    float3 alpha = float3(0.0);
    if (params.chunk_loads != 0 && node.w > 0.0)
    {
        float3 force = node_frame[NODE_FRAME * k + 2].xyz + node_frame[NODE_FRAME * k + 3].xyz / params.dt;
        float3 torque = node_frame[NODE_FRAME * k + 4].xyz + node_frame[NODE_FRAME * k + 5].xyz / params.dt;
        a += force / node.w;
        float inertia = node_frame[NODE_FRAME * k + 1].x;
        if (inertia > 0.0)
            alpha = -torque / inertia;
    }
    float3 a_local = inverse_rotate(q, a);
    float3 w = inverse_rotate(q, from_rubble(bodies[b + 4].xyz));
    float3 r = node.xyz - frames[2 * slot].xyz;
    a_local -= cross(w, cross(w, r));
    accel[2 * k] = float4(inverse_rotate(q, alpha), 0.0);
    accel[2 * k + 1] = float4(a_local, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void blast_sync(uint3 id: SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= params.bond_count || work[params.w_alive + j] != 0)
        return;
    BlastBond bond = bonds[j];
    if (!(bond.health > 0.0))
        return;
    bonds[j].health = 0.0;
    x[j * 2] = float4(0.0);
    x[j * 2 + 1] = float4(0.0);
    topo[topo_dirty(params.node_count)].store(1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void blast_breaks(uint3 id: SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= params.bond_count || work[params.w_alive + j] == 0)
        return;
    BlastBond bond = bonds[j];
    if (bond.health > 0.0)
        return;
    work[params.w_alive + j] = 0;
    atom[params.a_dirty + work[bond.n0]].or(1u);
    atom[params.a_counters + C_FRAME_BROKEN].add(1u);
    atom[params.a_counters + C_BROKEN].add(1u);
}
