// Fracture topology on the GPU (engine/docs/stress-baseline-goal.md, milestone M2): a
// disconnected island's children, as stress-ref `split_cluster` / `add_cluster` make them,
// from the GPU's own tables and state. It is PhysX's native stage's topology pass and
// candidate-body preparation (PxgDestructionTopology.cu, `prepareCandidateBodies`) on our
// layout:
//   split_components  one group per parent island: connected components over its still
//                     connected bonds (minimum-label propagation; a component's label is
//                     its first chunk in the island's order), then the children in order
//                     of their first chunk (`components`' order), each child's chunks in
//                     the island's order;
//   split_children    one thread per child: its mass, centre of mass and inertia (body
//                     frame of the structure, about its centre of mass), whether a
//                     support anchors it, and its rigid motion from its chunks' exact
//                     world velocities (`split_cluster`'s momentum transfer).
// Every sum runs over a fixed order, so results repeat bit for bit.

import common;
import joint;
import bonds;

struct SplitParams
{
    uint parent_count;
    uint out_base;   // float4 `out` index of the first child record
    uint pad0;
    uint pad1;
};

// Per parent (`parents`): x its island, y the base of its labels, members and child
// records (its first chunk's offset among all the parents' chunks); the GPU writes z,
// its child count.
[[vk::binding(0, 0)]] ConstantBuffer<SplitParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(2, 0)]] StructuredBuffer<uint> index;
[[vk::binding(3, 0)]] StructuredBuffer<BondStatic> bonds;
[[vk::binding(4, 0)]] StructuredBuffer<BondDyn> bond_dyn;
[[vk::binding(5, 0)]] StructuredBuffer<Island> islands;
[[vk::binding(6, 0)]] StructuredBuffer<float4> state;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint4> parents;
// Per parent chunk (at its parent's base): its label (island-local), then (at
// `labels + total`) the chunks of each child in order (island-local indices).
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> work;
// Per child (at its parent's base + its number): 8 float4 (`ChildRecord`).
[[vk::binding(9, 0)]] RWStructuredBuffer<float4> out;
// Per parent, its rigid frame at the split: rotation (x, y, z, w), centre-of-mass
// velocity, angular velocity (world), centre of mass (body frame; the cluster's own, not
// the island record's mass-scaled one), pose position (world; spawn.slang). The engine's motion at the moment of the split,
// which an island record (its frame-start pose) does not hold.
[[vk::binding(10, 0)]] RWStructuredBuffer<float4> frames;
// The engine's bodies (rubble `BodyState`, 7 float4 each) and, per parent, its body (x as
// bits; NONE: none), its centre of mass in its body frame (y-w as floats) and its
// principal rotation (the next float4): for `split_frames`.
[[vk::binding(11, 0)]] StructuredBuffer<float4> engine_bodies;
[[vk::binding(12, 0)]] StructuredBuffer<float4> parent_bodies;

static const uint THREADS = 256;
static const uint CHILD_WORDS = 8;

groupshared uint g_changed;

uint total_chunks()
{
    // The last parent's base plus its chunk count (`parents[n].y`, written by the host).
    return parents[params.parent_count].y;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void split_components(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint p = group.x;
    if (p >= params.parent_count)
        return;
    uint4 parent = parents[p];
    Island isl = islands[parent.x];
    uint first = isl.range.x;
    uint n = isl.range.y - isl.range.x;
    uint base = parent.y;
    uint members = total_chunks();
    for (uint c = tid; c < n; c += THREADS)
        work[base + c] = c;
    device_group_barrier();
    // Labels fall to their component's first chunk; a pass reads whatever its
    // neighbours' labels are by then (every value read is an upper bound of the
    // final one), so the result does not depend on the order threads run in.
    for (uint pass = 0; pass <= n; pass++)
    {
        if (tid == 0)
            g_changed = 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint c = tid; c < n; c += THREADS)
        {
            uint l = work[base + c];
            uint gc = first + c;
            for (uint e = index[gc]; e < index[gc + 1]; e++)
            {
                uint entry = index[e];
                uint bond = entry >> 1;
                if (!connected(bond_dyn[bond].js, bonds[bond].law.stiff1.w != 0.0))
                    continue;
                uint other = (entry & 1) == 0 ? bonds[bond].law.ids.z : bonds[bond].law.ids.y;
                if (other < first || other >= first + n)
                    continue;
                l = min(l, work[base + other - first]);
            }
            l = min(l, work[base + l]);
            if (l < work[base + c])
            {
                work[base + c] = l;
                g_changed = 1;
            }
        }
        device_group_barrier();
        bool done = g_changed == 0;
        GroupMemoryBarrierWithGroupSync();
        if (done)
            break;
    }
    // Children in order of their first chunk, each child's chunks in the island's
    // order: one thread, a fixed order (n is at most an island's chunks).
    if (tid != 0)
        return;
    uint children = 0;
    for (uint c = 0; c < n; c++)
    {
        if (work[base + c] == c)
        {
            // A root: its child number, kept in its member slot until placement.
            work[members + base + c] = children;
            children++;
        }
    }
    // Counts, then offsets (in the child records' first word, temporarily).
    for (uint k = 0; k < children; k++)
        out[params.out_base + CHILD_WORDS * (base + k)] = float4(asfloat(p), asfloat(0u), asfloat(0u), asfloat(0u));
    for (uint c = 0; c < n; c++)
    {
        uint k = work[members + base + work[base + c]];
        float4 rec = out[params.out_base + CHILD_WORDS * (base + k)];
        rec.z = asfloat(asuint(rec.z) + 1);
        out[params.out_base + CHILD_WORDS * (base + k)] = rec;
    }
    uint at = 0;
    for (uint k = 0; k < children; k++)
    {
        float4 rec = out[params.out_base + CHILD_WORDS * (base + k)];
        rec.y = asfloat(at);
        at += asuint(rec.z);
        rec.z = asfloat(0u);
        out[params.out_base + CHILD_WORDS * (base + k)] = rec;
    }
    // Child numbers of every chunk, before the member list overwrites them.
    for (uint c = 0; c < n; c++)
        work[base + c] = work[members + base + work[base + c]];
    for (uint c = 0; c < n; c++)
    {
        uint k = work[base + c];
        float4 rec = out[params.out_base + CHILD_WORDS * (base + k)];
        uint slot = asuint(rec.y) + asuint(rec.z);
        work[members + base + slot] = c;
        rec.z = asfloat(asuint(rec.z) + 1);
        out[params.out_base + CHILD_WORDS * (base + k)] = rec;
    }
    parents[p] = uint4(parent.x, parent.y, children, 0);
}

// 3x3 inverse of a symmetric matrix given by rows (adjugate over determinant).
void inverse3(float3 r0, float3 r1, float3 r2, out float3 i0, out float3 i1, out float3 i2)
{
    float3 c0 = cross(r1, r2);
    float3 c1 = cross(r2, r0);
    float3 c2 = cross(r0, r1);
    float det = dot(r0, c0);
    float inv = det != 0.0 ? 1.0 / det : 0.0;
    i0 = float3(c0.x, c1.x, c2.x) * inv;
    i1 = float3(c0.y, c1.y, c2.y) * inv;
    i2 = float3(c0.z, c1.z, c2.z) * inv;
}

float3 mat_mul(float3 r0, float3 r1, float3 r2, float3 v)
{
    return float3(dot(r0, v), dot(r1, v), dot(r2, v));
}

// R M R^T for a body-frame matrix M (rows), as rows.
void rotate_matrix(Quat q, float3 m0, float3 m1, float3 m2, out float3 o0, out float3 o1, out float3 o2)
{
    // Columns of R M R^T: R M R^T e_k.
    float3 k0 = rotate(q, mat_mul(m0, m1, m2, inverse_rotate(q, float3(1, 0, 0))));
    float3 k1 = rotate(q, mat_mul(m0, m1, m2, inverse_rotate(q, float3(0, 1, 0))));
    float3 k2 = rotate(q, mat_mul(m0, m1, m2, inverse_rotate(q, float3(0, 0, 1))));
    o0 = float3(k0.x, k1.x, k2.x);
    o1 = float3(k0.y, k1.y, k2.y);
    o2 = float3(k0.z, k1.z, k2.z);
}

// A child record (8 float4 at `out_base + 8 * (parent base + child)`):
//   0: parent (its `parents` index), member offset, member count, anchored (uint bits)
//   1: centre of mass (body frame of the structure), mass
//   2-4: inertia rows about the centre of mass (body frame)
//   5: velocity (world), 6: angular velocity (world)
//   7: -
[shader("compute")]
[numthreads(64, 1, 1)]
void split_children(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    uint total = total_chunks();
    if (slot >= total)
        return;
    // Which parent this slot is in (parents are few: a scan).
    uint p = 0;
    while (p + 1 < params.parent_count && parents[p + 1].y <= slot)
        p++;
    uint4 parent = parents[p];
    uint k = slot - parent.y;
    if (k >= parent.z)
        return;
    uint at = params.out_base + CHILD_WORDS * slot;
    float4 head = out[at];
    uint offset = asuint(head.y);
    uint count = asuint(head.z);
    Island isl = islands[parent.x];
    uint first = isl.range.x;
    uint members = total;
    Quat q = quat_of(frames[5 * p]);
    float3 vel = frames[5 * p + 1].xyz;
    float3 omega = frames[5 * p + 2].xyz;
    float3 parent_com = frames[5 * p + 3].xyz;
    // Mass and centre of mass (`add_cluster`), in member order.
    float mass = 0.0;
    float3 moment = float3(0.0);
    bool anchored = false;
    for (uint j = 0; j < count; j++)
    {
        uint c = first + work[members + parent.y + offset + j];
        ChunkStatic ch = chunks[c];
        mass += ch.center.w;
        moment += ch.center.xyz * ch.center.w;
        anchored = anchored || ch.info.x != SUPPORT_NONE;
    }
    float3 com = moment / mass;
    // Inertia about it, and the momentum of the chunks' exact world velocities
    // (`split_cluster`): v = the parent's rigid motion at the chunk plus its hidden
    // velocity, w likewise.
    float3 i0 = float3(0.0), i1 = float3(0.0), i2 = float3(0.0);
    float3 pm = float3(0.0);
    float3 l = float3(0.0);
    for (uint j = 0; j < count; j++)
    {
        uint c = first + work[members + parent.y + offset + j];
        ChunkStatic ch = chunks[c];
        float m = ch.center.w;
        float3 r = ch.center.xyz - com;
        float rr = dot(r, r);
        i0 += ch.inertia0.xyz + (float3(rr, 0, 0) - r.x * r) * m;
        i1 += ch.inertia1.xyz + (float3(0, rr, 0) - r.y * r) * m;
        i2 += ch.inertia2.xyz + (float3(0, 0, rr) - r.z * r) * m;
        float3 u = state[4 * c].xyz;
        float3 v = vel + cross(omega, rotate(q, ch.center.xyz + u - parent_com)) + rotate(q, state[4 * c + 2].xyz);
        float3 w = omega + rotate(q, state[4 * c + 3].xyz);
        pm += v * m;
        float3 rw = rotate(q, r);
        float3 ci0, ci1, ci2;
        rotate_matrix(q, ch.inertia0.xyz, ch.inertia1.xyz, ch.inertia2.xyz, ci0, ci1, ci2);
        l += cross(rw, v) * m + mat_mul(ci0, ci1, ci2, w);
    }
    float3 cv = float3(0.0);
    float3 cw = float3(0.0);
    if (!anchored)
    {
        float3 w0, w1, w2;
        rotate_matrix(q, i0, i1, i2, w0, w1, w2);
        float3 v0, v1, v2;
        inverse3(w0, w1, w2, v0, v1, v2);
        cv = pm / mass;
        cw = mat_mul(v0, v1, v2, l);
    }
    out[at] = float4(asfloat(p), asfloat(offset), asfloat(count), asfloat(anchored ? 1u : 0u));
    out[at + 1] = float4(com, mass);
    out[at + 2] = float4(i0, 0.0);
    out[at + 3] = float4(i1, 0.0);
    out[at + 4] = float4(i2, 0.0);
    out[at + 5] = float4(cv, 0.0);
    out[at + 6] = float4(cw, 0.0);
    out[at + 7] = float4(0.0);
}

// Each parent's rigid frame from its engine body (stress-rubble `motion_of`), when the
// host gives its body (`parent_bodies`): rotation (x, y, z, w), centre-of-mass velocity,
// angular velocity, centre of mass (body frame), pose position. The engine is Y-up.
[shader("compute")]
[numthreads(64, 1, 1)]
void split_frames(uint3 id: SV_DispatchThreadID)
{
    uint p = id.x;
    if (p >= params.parent_count)
        return;
    float4 head = parent_bodies[2 * p];
    uint body = asuint(head.x);
    if (body == 0xffffffffu)
        return;
    float3 com = head.yzw;
    Quat principal = quat_of(parent_bodies[2 * p + 1]);
    float4 b0 = engine_bodies[7 * body];
    Quat orientation = quat_of(engine_bodies[7 * body + 1]);
    float3 lin = engine_bodies[7 * body + 3].xyz;
    float3 ang = engine_bodies[7 * body + 4].xyz;
    // rot_from_rubble: (axes^-1 q); then times the principal's inverse.
    float h = 0.70710678118654752;
    Quat axes_inv;
    axes_inv.w = h;
    axes_inv.x = h;
    axes_inv.y = 0.0;
    axes_inv.z = 0.0;
    Quat p_inv;
    p_inv.w = principal.w;
    p_inv.x = -principal.x;
    p_inv.y = -principal.y;
    p_inv.z = -principal.z;
    Quat rot = normalized(quat_mul(normalized(quat_mul(axes_inv, orientation)), p_inv));
    float3 com_world = float3(b0.x, -b0.z, b0.y);
    frames[5 * p] = quat_vec(rot);
    frames[5 * p + 1] = float4(lin.x, -lin.z, lin.y, 0.0);
    frames[5 * p + 2] = float4(ang.x, -ang.z, ang.y, 0.0);
    frames[5 * p + 3] = float4(com, 0.0);
    frames[5 * p + 4] = float4(com_world - rotate(rot, com), 0.0);
}
