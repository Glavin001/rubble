// New rigid bodies for a split's children, made on the GPU (engine/docs/stress-baseline-goal.md
// M3): each child's rubble body and colliders as stress-rubble's `cook_with` and
// `create_body` make them, from the per-chunk part table (`Part`: each collider's shape in
// the structure's frame) and the split's child records (split.slang). PhysX's native stage
// does this in `prepareCandidateBodies` and its body initialization; here the records are
// rubble's `BodyState` and `Collider`.
//   spawn_offsets  one thread: each child's first body and collider slot, in child order
//                  (bodies and colliders are appended in the order the host would add them)
//   spawn_bodies   one thread per child: mass, centre of mass and inertia of its parts,
//                  principal axes (rubble's Jacobi, `convex::principal_axes`), its body and
//                  its colliders
// Frames: the oracle is Z-up, rubble Y-up: (x, y, z) -> (x, z, -y), rotations by -90
// degrees about x.

import common;

struct SpawnParams
{
    uint parent_count;
    uint first_body;     // the body slot the first child takes
    uint first_collider; // likewise for colliders
    uint live;           // 1: `bodies` and `colliders` are the engine's own buffers (absolute slots;
                         // bodies in both regions, the second at `max_bodies`); 0: a copy from slot 0
    uint max_bodies;
    uint max_colliders;  // the engine's collider capacity
    uint island_free;    // islands the stress tables can still take
    uint pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<SpawnParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> parents;   // split.slang `parents` (z: children)
[[vk::binding(2, 0)]] StructuredBuffer<uint> work;       // split.slang `work` (members)
[[vk::binding(3, 0)]] StructuredBuffer<float4> children; // split.slang `out` (child records)
[[vk::binding(4, 0)]] StructuredBuffer<float4> frames;   // per parent: rotation, v, w, com, position
[[vk::binding(5, 0)]] StructuredBuffer<uint4> part_ranges; // per chunk (`chunk_key`): first part, count
[[vk::binding(6, 0)]] StructuredBuffer<float4> parts;    // 8 float4 per part (`Part`)
[[vk::binding(7, 0)]] StructuredBuffer<ChunkStatic> chunks;
[[vk::binding(8, 0)]] StructuredBuffer<float4> state;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint4> slots;   // per child slot: body, first collider, colliders, valid
[[vk::binding(10, 0)]] RWStructuredBuffer<float4> bodies;    // 7 float4 per body (`BodyState`)
[[vk::binding(11, 0)]] RWStructuredBuffer<float4> colliders; // 4 float4 per collider (`Collider`)
[[vk::binding(12, 0)]] StructuredBuffer<uint4> islands_of; // per parent: its island's first chunk (x); its engine body (y), first collider (z), collider end (w)
// Per child slot: centre of mass (the structure's frame), mass; principal rotation (x, y, z, w).
[[vk::binding(13, 0)]] RWStructuredBuffer<float4> extras;

static const uint CHILD_WORDS = 8;
static const uint PART_WORDS = 8;
static const uint BODY_WORDS = 7;
static const uint FLAG_DYNAMIC = 1;
static const uint FLAG_AWAKE = 8;

// Per structure: its first chunk's index among every structure's chunks (`part_ranges`).
[[vk::binding(14, 0)]] StructuredBuffer<uint> structure_base;
// [0]: 1 when the split does not fit (islands, bodies or colliders past a capacity):
// nothing is made, retired or laid out, and the host splits instead.
[[vk::binding(15, 0)]] RWStructuredBuffer<uint> control;

// A layout chunk's part range: by its structure and chunk, so the table does not change
// with the layout.
uint4 part_range(uint layout_chunk)
{
    uint4 info = chunks[layout_chunk].info;
    return part_ranges[structure_base[info.z] + info.w];
}

uint total_slots()
{
    return parents[params.parent_count].y;
}

// The child records' member list of child slot `slot` of parent `p`: (offset, count).
uint2 child_members(uint slot)
{
    float4 head = children[CHILD_WORDS * slot];
    return uint2(asuint(head.y), asuint(head.z));
}

uint parent_of(uint slot)
{
    uint p = 0;
    while (p + 1 < params.parent_count && parents[p + 1].y <= slot)
        p++;
    return p;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void spawn_offsets(uint3 id: SV_DispatchThreadID)
{
    uint body = params.first_body;
    uint collider = params.first_collider;
    uint total = total_slots();
    // Room first: islands (one more per child past the first), bodies, colliders.
    uint islands_needed = 0;
    uint bodies_needed = 0;
    uint colliders_needed = 0;
    for (uint p = 0; p < params.parent_count; p++)
    {
        uint4 parent = parents[p];
        if (parent.z < 2)
            continue;
        islands_needed += parent.z - 1;
        bodies_needed += parent.z;
        uint first = islands_of[p].x;
        for (uint k = 0; k < parent.z; k++)
        {
            uint2 m = child_members(parent.y + k);
            for (uint j = 0; j < m.y; j++)
                colliders_needed += part_range(first + work[total + parent.y + m.x + j]).y;
        }
    }
    bool fits = islands_needed <= params.island_free && (params.live == 0 || (body + bodies_needed <= params.max_bodies && collider + colliders_needed <= params.max_colliders));
    control[0] = fits ? 0u : 1u;
    if (!fits)
        return;
    for (uint p = 0; p < params.parent_count; p++)
    {
        uint4 parent = parents[p];
        uint first = islands_of[p].x;
        // A disconnection that left the island whole makes no body.
        if (parent.z < 2)
            continue;
        for (uint k = 0; k < parent.z; k++)
        {
            uint slot = parent.y + k;
            uint2 m = child_members(slot);
            uint count = 0;
            for (uint j = 0; j < m.y; j++)
                count += part_range(first + work[total + parent.y + m.x + j]).y;
            slots[slot] = uint4(body, collider, count, 1);
            body++;
            collider += count;
        }
    }
}

// Quaternion of a rotation matrix given by columns (glam `Quat::from_rotation_axes`).
Quat quat_from_columns(float3 c0, float3 c1, float3 c2)
{
    float m00 = c0.x, m01 = c0.y, m02 = c0.z;
    float m10 = c1.x, m11 = c1.y, m12 = c1.z;
    float m20 = c2.x, m21 = c2.y, m22 = c2.z;
    Quat q;
    if (m22 <= 0.0)
    {
        float dif10 = m11 - m00;
        float omm22 = 1.0 - m22;
        if (dif10 <= 0.0)
        {
            float f = omm22 - dif10;
            float inv = 0.5 / sqrt(f);
            q.x = f * inv; q.y = (m01 + m10) * inv; q.z = (m02 + m20) * inv; q.w = (m12 - m21) * inv;
        }
        else
        {
            float f = omm22 + dif10;
            float inv = 0.5 / sqrt(f);
            q.x = (m01 + m10) * inv; q.y = f * inv; q.z = (m12 + m21) * inv; q.w = (m20 - m02) * inv;
        }
    }
    else
    {
        float sum10 = m11 + m00;
        float opm22 = 1.0 + m22;
        if (sum10 <= 0.0)
        {
            float f = opm22 - sum10;
            float inv = 0.5 / sqrt(f);
            q.x = (m02 + m20) * inv; q.y = (m12 + m21) * inv; q.z = f * inv; q.w = (m01 - m10) * inv;
        }
        else
        {
            float f = opm22 + sum10;
            float inv = 0.5 / sqrt(f);
            q.x = (m12 - m21) * inv; q.y = (m20 - m02) * inv; q.z = (m01 - m10) * inv; q.w = f * inv;
        }
    }
    return q;
}

// engine-schema `convex::principal_axes`: Jacobi rotations on the largest off-diagonal
// entry, at most 32, the same pivot order; the rotation and the moments.
void principal_axes(float3 r0, float3 r1, float3 r2, out Quat rot, out float3 moments)
{
    float a[3][3] = { { r0.x, r0.y, r0.z }, { r1.x, r1.y, r1.z }, { r2.x, r2.y, r2.z } };
    float v[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
    for (uint it = 0; it < 32; it++)
    {
        uint p = 0, q = 1;
        if (abs(a[0][2]) > abs(a[p][q])) { p = 0; q = 2; }
        if (abs(a[1][2]) > abs(a[p][q])) { p = 1; q = 2; }
        float scale = max(max(abs(a[0][0]), abs(a[1][1])), max(abs(a[2][2]), 1e-30));
        if (abs(a[p][q]) <= 1e-14 * scale)
            break;
        float theta = 0.5 * atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
        float s = sin(theta);
        float c = cos(theta);
        float app = a[p][p], aqq = a[q][q], apq = a[p][q];
        for (uint k = 0; k < 3; k++)
        {
            if (k != p && k != q)
            {
                float kp = a[k][p], kq = a[k][q];
                a[k][p] = c * kp - s * kq;
                a[p][k] = a[k][p];
                a[k][q] = s * kp + c * kq;
                a[q][k] = a[k][q];
            }
        }
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for (uint row = 0; row < 3; row++)
        {
            float kp = v[row][p], kq = v[row][q];
            v[row][p] = c * kp - s * kq;
            v[row][q] = s * kp + c * kq;
        }
    }
    float3 x = float3(v[0][0], v[1][0], v[2][0]);
    float3 y = float3(v[0][1], v[1][1], v[2][1]);
    float3 z = float3(v[0][2], v[1][2], v[2][2]);
    if (dot(cross(x, y), z) < 0.0)
        z = -z;
    rot = normalized(quat_from_columns(x, y, z));
    moments = float3(a[0][0], a[1][1], a[2][2]);
}

float3 to_rubble(float3 v)
{
    return float3(v.x, v.z, -v.y);
}

// `rot_to_rubble`: -90 degrees about x, then q.
Quat rot_to_rubble(Quat q)
{
    float h = 0.70710678118654752;
    Quat axes;
    axes.w = h;
    axes.x = -h;
    axes.y = 0.0;
    axes.z = 0.0;
    return normalized(quat_mul(axes, q));
}

Quat conj(Quat q)
{
    Quat c;
    c.w = q.w;
    c.x = -q.x;
    c.y = -q.y;
    c.z = -q.z;
    return c;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_bodies(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    uint total = total_slots();
    if (slot >= total)
        return;
    uint p = parent_of(slot);
    uint4 parent = parents[p];
    if (slot - parent.y >= parent.z || slots[slot].w == 0)
        return;
    uint first = islands_of[p].x;
    uint2 m = child_members(slot);
    float4 rec0 = children[CHILD_WORDS * slot];
    bool anchored = asuint(rec0.w) != 0;
    // Mass, centre and inertia of the child's parts (`cook_with`), in member order.
    float mass = 0.0;
    float3 moment = float3(0.0);
    for (uint j = 0; j < m.y; j++)
    {
        uint4 r = part_range(first + work[total + parent.y + m.x + j]);
        for (uint k = r.x; k < r.x + r.y; k++)
        {
            float4 c = parts[PART_WORDS * k];
            mass += c.w;
            moment += c.xyz * c.w;
        }
    }
    float3 com = moment / max(mass, 1e-30);
    float3 i0 = float3(0.0), i1 = float3(0.0), i2 = float3(0.0);
    for (uint j = 0; j < m.y; j++)
    {
        uint4 r = part_range(first + work[total + parent.y + m.x + j]);
        for (uint k = r.x; k < r.x + r.y; k++)
        {
            float4 c = parts[PART_WORDS * k];
            float3 d = c.xyz - com;
            float dd = dot(d, d);
            i0 += parts[PART_WORDS * k + 1].xyz + (float3(dd, 0, 0) - d.x * d) * c.w;
            i1 += parts[PART_WORDS * k + 2].xyz + (float3(0, dd, 0) - d.y * d) * c.w;
            i2 += parts[PART_WORDS * k + 3].xyz + (float3(0, 0, dd) - d.z * d) * c.w;
        }
    }
    Quat principal;
    float3 moments;
    principal_axes(i0, i1, i2, principal, moments);
    Quat to_body = conj(principal);
    // The child's pose (`split_cluster`): the parent's, a free child's moved by its
    // chunks' mean hidden displacement (stress masses).
    Quat pose_rot = quat_of(frames[5 * p]);
    float3 pose_pos = frames[5 * p + 4].xyz;
    if (!anchored)
    {
        float sm = 0.0;
        float3 su = float3(0.0);
        for (uint j = 0; j < m.y; j++)
        {
            uint c = first + work[total + parent.y + m.x + j];
            float cm = chunks[c].center.w;
            sm += cm;
            su += state[4 * c].xyz * cm;
        }
        pose_pos += rotate(pose_rot, su / sm);
    }
    // The body (`create_body`).
    uint4 at = slots[slot];
    Quat orientation = rot_to_rubble(normalized(quat_mul(pose_rot, principal)));
    float inv_mass = anchored || !(mass > 0.0) ? 0.0 : 1.0 / mass;
    float3 inv_inertia = float3(0.0);
    if (!anchored)
        inv_inertia = float3(moments.x > 0.0 ? 1.0 / moments.x : 0.0, moments.y > 0.0 ? 1.0 / moments.y : 0.0, moments.z > 0.0 ? 1.0 / moments.z : 0.0);
    float3 velocity = anchored ? float3(0.0) : to_rubble(children[CHILD_WORDS * slot + 5].xyz);
    float3 spin = anchored ? float3(0.0) : to_rubble(children[CHILD_WORDS * slot + 6].xyz);
    uint flags = anchored ? 0u : (FLAG_DYNAMIC | FLAG_AWAKE);
    float4 record[BODY_WORDS] = {
        float4(to_rubble(pose_pos + rotate(pose_rot, com)), inv_mass),
        quat_vec(orientation),
        float4(inv_inertia, asfloat(flags)),
        float4(velocity, 0.0),
        float4(spin, 0.0),
        float4(0.0),
        float4(0.0, 0.0, 0.0, 1.0),
    };
    uint b = BODY_WORDS * (params.live != 0 ? at.x : at.x - params.first_body);
    for (uint w = 0; w < BODY_WORDS; w++)
        bodies[b + w] = record[w];
    // The previous-pose region (`add_bodies` writes both).
    if (params.live != 0)
        for (uint w = 0; w < BODY_WORDS; w++)
            bodies[BODY_WORDS * (params.max_bodies + at.x) + w] = record[w];
    extras[2 * slot] = float4(com, mass);
    extras[2 * slot + 1] = quat_vec(principal);
    // Its colliders, in part order: at `to_body (origin - com)`, turned `to_body rotation`.
    uint col = at.y;
    for (uint j = 0; j < m.y; j++)
    {
        uint4 r = part_range(first + work[total + parent.y + m.x + j]);
        for (uint k = r.x; k < r.x + r.y; k++)
        {
            float4 o = parts[PART_WORDS * k + 4];
            Quat rot = quat_of(parts[PART_WORDS * k + 5]);
            float4 ids = parts[PART_WORDS * k + 6];
            float4 shape_params = parts[PART_WORDS * k + 7];
            uint cw = 4 * (params.live != 0 ? col : col - params.first_collider);
            colliders[cw] = float4(asfloat(at.x), ids.x, ids.y, ids.z);
            colliders[cw + 1] = float4(rotate(to_body, o.xyz - com), o.w);
            colliders[cw + 2] = quat_vec(normalized(quat_mul(to_body, rot)));
            colliders[cw + 3] = shape_params;
            col++;
        }
    }
}

// A split parent's engine body retired where it stands (stress-rubble `remove_body`):
// no mass, no motion, no flags, in both regions; its colliders in no pair (a category no
// mask includes, a mask that admits none: `CollisionFilter { 0x8000, 0, 0 }`). Live only.
[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_retire(uint3 id: SV_DispatchThreadID)
{
    uint p = id.x;
    if (p >= params.parent_count || params.live == 0 || parents[p].z < 2 || control[0] != 0)
        return;
    uint4 e = islands_of[p];
    // The current record, modified, into both regions (`write_bodies` writes one record twice).
    uint b = BODY_WORDS * e.y;
    float4 record[BODY_WORDS];
    for (uint w = 0; w < BODY_WORDS; w++)
        record[w] = bodies[b + w];
    record[0].w = 0.0;
    record[2] = float4(0.0, 0.0, 0.0, asfloat(0u));
    record[3].xyz = float3(0.0);
    record[4].xyz = float3(0.0);
    for (uint region = 0; region < 2; region++)
        for (uint w = 0; w < BODY_WORDS; w++)
            bodies[BODY_WORDS * (e.y + region * params.max_bodies) + w] = record[w];
    for (uint c = e.z; c < e.w; c++)
    {
        float4 head = colliders[4 * c];
        uint material = asuint(head.z) & 0xFFFFu;
        colliders[4 * c] = float4(head.x, head.y, asfloat(material), asfloat(0x8000u));
    }
}
