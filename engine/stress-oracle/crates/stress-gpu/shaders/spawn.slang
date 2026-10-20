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
    uint island_cap;     // islands the stress tables can take (`CTL_ISLANDS` of them live)
    // Where the phased kernels (`spawn_*_mass`, `_pose`, `_record`, `spawn_collider_members`)
    // keep what the fused ones hold in registers: `extras` from this float4, 2 per child
    // slot (principal moments, pose position); `member_offsets` from this word, 4 per
    // member (child slot, body, first part, parts).
    uint moments_base;
    uint info_base;
    uint pad0;
    uint pad1;
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
// From `params.moments_base`, the phased kernels' per child slot: principal moments; pose position.
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
// Per member (split.slang's member order: parent, then child): its first collider slot.
// From `params.info_base`, `spawn_collider_members`'s per member: (child slot or NONE, body,
// first part, parts).
[[vk::binding(16, 0)]] RWStructuredBuffer<uint> member_offsets;

// A layout chunk's part range: by its structure and chunk, so the table does not change
// with the layout.
uint4 part_range(uint layout_chunk)
{
    uint4 info = chunks[layout_chunk].info;
    return part_ranges[structure_base[info.z] + info.w];
}

uint total_slots()
{
    return parents[control[CTL_PARENTS]].y;
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
    while (p + 1 < control[CTL_PARENTS] && parents[p + 1].y <= slot)
        p++;
    return p;
}

// The parent of member or child slot i: the last parent whose base is at most i.
uint parent_at(uint i)
{
    uint lo = 0, hi = control[CTL_PARENTS] - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi + 1) / 2;
        if (parents[mid].y <= i)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// A member's collider count: its chunk's parts, when its parent splits into children.
uint member_parts(uint i, uint total)
{
    uint p = parent_at(i);
    if (parents[p].z < 2)
        return 0;
    return part_range(islands_of[p].x + work[total + i]).y;
}

// Child slot s makes a body: its parent splits, and it is one of the children.
bool slot_valid(uint s)
{
    uint4 parent = parents[parent_at(s)];
    return parent.z >= 2 && s - parent.y < parent.z;
}

groupshared uint g_scan[1024];

// An exclusive scan of `v` over the group, in blocks of 1024 (the caller keeps the carry).
uint block_scan(uint tid, uint v)
{
    g_scan[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 1; step < 1024; step <<= 1)
    {
        uint add = tid >= step ? g_scan[tid - step] : 0u;
        GroupMemoryBarrierWithGroupSync();
        g_scan[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    return g_scan[tid] - v;
}

groupshared uint g_total;
groupshared uint g_split_parents;

// One group: each child its first body and collider slot, in child order (bodies and
// colliders appended in the order the host would add them), by scans over the members'
// collider counts and the valid child slots; room checked first (nothing is made when it
// does not fit). The same slots as one thread walking them in that order.
[shader("compute")]
[numthreads(1024, 1, 1)]
void spawn_offsets(uint3 thread: SV_GroupThreadID)
{
    uint tid = thread.x;
    uint body0 = control[CTL_FIRST_BODY];
    uint collider0 = control[CTL_FIRST_COLLIDER];
    // The exits and loop bounds as uniform values: barriers follow.
    if (group_uniform(control[CTL_PARENTS], tid) == 0)
    {
        if (tid == 0)
        {
            control[CTL_ABORT] = 0;
            control[CTL_BODIES_AFTER] = body0;
            control[CTL_COLLIDERS_AFTER] = collider0;
        }
        return;
    }
    uint total = group_uniform(total_slots(), tid);
    // Members' first colliders.
    uint carry = 0;
    for (uint block = 0; block < total; block += 1024)
    {
        uint i = block + tid;
        uint v = i < total ? member_parts(i, total) : 0u;
        uint at = block_scan(tid, v);
        if (i < total)
            member_offsets[i] = collider0 + carry + at;
        uint t = g_scan[1023];
        GroupMemoryBarrierWithGroupSync();
        carry += t;
    }
    uint colliders_needed = carry;
    // Valid child slots' bodies (a first pass for the count, the slots after the check).
    uint bodies_needed = 0;
    for (uint block = 0; block < total; block += 1024)
    {
        uint s2 = block + tid;
        uint v = s2 < total && slot_valid(s2) ? 1u : 0u;
        block_scan(tid, v);
        uint t = g_scan[1023];
        GroupMemoryBarrierWithGroupSync();
        bodies_needed += t;
    }
    if (tid == 0)
    {
        uint split = 0;
        for (uint p = 0; p < control[CTL_PARENTS]; p++)
            if (parents[p].z >= 2)
                split++;
        g_split_parents = split;
    }
    GroupMemoryBarrierWithGroupSync();
    uint islands_needed = bodies_needed - g_split_parents;
    bool fits = islands_needed <= params.island_cap - min(params.island_cap, control[CTL_ISLANDS]) && (params.live == 0 || (body0 + bodies_needed <= params.max_bodies && collider0 + colliders_needed <= params.max_colliders));
    if (tid == 0)
    {
        control[CTL_ABORT] = fits ? 0u : 1u;
        control[CTL_BODIES_AFTER] = fits ? body0 + bodies_needed : body0;
        control[CTL_COLLIDERS_AFTER] = fits ? collider0 + colliders_needed : collider0;
    }
    if (group_uniform(fits ? 1u : 0u, tid) == 0)
        return;
    // The slots: a valid child its body, first collider and collider count; every other
    // slot cleared (the slots are kept between split passes).
    uint bodies = 0;
    for (uint block = 0; block < total; block += 1024)
    {
        uint s2 = block + tid;
        bool valid = s2 < total && slot_valid(s2);
        uint at = block_scan(tid, valid ? 1u : 0u);
        if (s2 < total)
        {
            if (valid)
            {
                uint4 parent = parents[parent_at(s2)];
                uint2 m = child_members(s2);
                uint first = parent.y + m.x;
                uint end = first + m.y;
                uint c_first = member_offsets[first];
                // The members' offsets are a prefix over the pass: a child's colliders end
                // where the next member's begin (the pass's last at its colliders).
                uint c_end = end < total ? member_offsets[end] : collider0 + colliders_needed;
                slots[s2] = uint4(body0 + bodies + at, c_first, c_end - c_first, 1);
            }
            else
            {
                slots[s2] = uint4(0);
            }
        }
        uint t = g_scan[1023];
        GroupMemoryBarrierWithGroupSync();
        bodies += t;
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

// A child slot whose body `spawn_bodies` makes: its parent's, when that parent splits, the
// slot is one of its children and the split fits (`spawn_offsets` gives exactly these slots
// a body: `slots[slot].w`, which the phases before the body's record do not read).
bool child_spawns(uint slot, uint total, out uint p, out uint4 parent)
{
    p = 0;
    parent = uint4(0);
    if (slot >= total || control[CTL_ABORT] != 0)
        return false;
    p = parent_of(slot);
    parent = parents[p];
    return parent.z >= 2 && slot - parent.y < parent.z;
}

// Mass, centre and inertia of a child's parts (`cook_with`), in member order (`members`:
// its first member's index in `work`).
void child_mass(uint first, uint2 m, uint members, out float mass, out float3 com, out float3 i0, out float3 i1, out float3 i2)
{
    mass = 0.0;
    float3 moment = float3(0.0);
    for (uint j = 0; j < m.y; j++)
    {
        uint4 r = part_range(first + work[members + j]);
        for (uint k = r.x; k < r.x + r.y; k++)
        {
            float4 c = parts[PART_WORDS * k];
            mass += c.w;
            moment += c.xyz * c.w;
        }
    }
    com = moment / max(mass, 1e-30);
    i0 = float3(0.0); i1 = float3(0.0); i2 = float3(0.0);
    for (uint j = 0; j < m.y; j++)
    {
        uint4 r = part_range(first + work[members + j]);
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
}

// The child's pose position (`split_cluster`): the parent's, a free child's moved by its
// chunks' mean hidden displacement (stress masses).
float3 child_pose(uint p, uint first, uint2 m, uint members, bool anchored)
{
    float3 pose_pos = frames[5 * p + 4].xyz;
    if (!anchored)
    {
        Quat pose_rot = quat_of(frames[5 * p]);
        float sm = 0.0;
        float3 su = float3(0.0);
        for (uint j = 0; j < m.y; j++)
        {
            uint c = first + work[members + j];
            float cm = chunks[c].center.w;
            sm += cm;
            su += state[4 * c].xyz * cm;
        }
        pose_pos += rotate(pose_rot, su / sm);
    }
    return pose_pos;
}

// The child's body (`create_body`) at its slot, and its extras (centre of mass, mass,
// principal rotation).
void child_record(uint slot, uint p, bool anchored, float mass, float3 com, Quat principal, float3 moments, float3 pose_pos)
{
    Quat pose_rot = quat_of(frames[5 * p]);
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
    uint b = BODY_WORDS * (params.live != 0 ? at.x : at.x - control[CTL_FIRST_BODY]);
    for (uint w = 0; w < BODY_WORDS; w++)
        bodies[b + w] = record[w];
    // The previous-pose region (`add_bodies` writes both).
    if (params.live != 0)
        for (uint w = 0; w < BODY_WORDS; w++)
            bodies[BODY_WORDS * (params.max_bodies + at.x) + w] = record[w];
    extras[2 * slot] = float4(com, mass);
    extras[2 * slot + 1] = quat_vec(principal);
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
    if (slot - parent.y >= parent.z || slots[slot].w == 0 || control[CTL_ABORT] != 0)
        return;
    uint first = islands_of[p].x;
    uint2 m = child_members(slot);
    float4 rec0 = children[CHILD_WORDS * slot];
    bool anchored = asuint(rec0.w) != 0;
    uint members = total + parent.y + m.x;
    float mass;
    float3 com, i0, i1, i2;
    child_mass(first, m, members, mass, com, i0, i1, i2);
    Quat principal;
    float3 moments;
    principal_axes(i0, i1, i2, principal, moments);
    child_record(slot, p, anchored, mass, com, principal, moments, child_pose(p, first, m, members, anchored));
}

// `spawn_bodies` in three dispatches, for a device that binds fewer storage buffers per
// stage than it uses (WebGPU's 10): the parts' mass properties and principal axes (into
// `extras`), the pose, then the body.
[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_bodies_mass(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    uint p;
    uint4 parent;
    if (!child_spawns(slot, total_slots(), p, parent))
        return;
    uint2 m = child_members(slot);
    float mass;
    float3 com, i0, i1, i2;
    child_mass(islands_of[p].x, m, total_slots() + parent.y + m.x, mass, com, i0, i1, i2);
    Quat principal;
    float3 moments;
    principal_axes(i0, i1, i2, principal, moments);
    extras[2 * slot] = float4(com, mass);
    extras[2 * slot + 1] = quat_vec(principal);
    extras[params.moments_base + 2 * slot] = float4(moments, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_bodies_pose(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    uint p;
    uint4 parent;
    if (!child_spawns(slot, total_slots(), p, parent))
        return;
    uint2 m = child_members(slot);
    bool anchored = asuint(children[CHILD_WORDS * slot].w) != 0;
    extras[params.moments_base + 2 * slot + 1] = float4(child_pose(p, islands_of[p].x, m, total_slots() + parent.y + m.x, anchored), 0.0);
}

// The body from the phases' extras (with `slots[slot].w`, as `spawn_bodies` checks it).
void phased_record(uint slot, uint p)
{
    float4 cm = extras[2 * slot];
    bool anchored = asuint(children[CHILD_WORDS * slot].w) != 0;
    child_record(slot, p, anchored, cm.w, cm.xyz, quat_of(extras[2 * slot + 1]), extras[params.moments_base + 2 * slot].xyz, extras[params.moments_base + 2 * slot + 1].xyz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_bodies_record(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    uint p;
    uint4 parent;
    if (!child_spawns(slot, total_slots(), p, parent) || slots[slot].w == 0)
        return;
    phased_record(slot, p);
}


// `spawn_bodies` with one group per child slot (`STRESS_GPU_SPLIT_TREE`): each thread over
// every 64th member, the sums combined in a fixed tree (f32 rounding apart, the same body).
groupshared float4 g_s0[64];
groupshared float4 g_s1[64];
groupshared float4 g_s2[64];
groupshared float4 g_s3[64];
groupshared float4 g_s4[64];

void spawn_tree(uint tid)
{
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            g_s0[tid] += g_s0[tid + s];
            g_s1[tid] += g_s1[tid + s];
            g_s2[tid] += g_s2[tid + s];
            g_s3[tid] += g_s3[tid + s];
            g_s4[tid] += g_s4[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// The group's slot, as uniform values (barriers follow); false: no body to make.
bool group_child(uint slot, uint tid, out uint p, out uint4 parent)
{
    p = 0;
    parent = uint4(0);
    uint total = total_slots();
    if (group_uniform(slot >= total || control[CTL_PARENTS] == 0 || control[CTL_ABORT] != 0 ? 1u : 0u, tid) != 0)
        return false;
    p = parent_at(slot);
    parent = parents[p];
    return group_uniform(slot - parent.y >= parent.z || parent.z < 2 ? 1u : 0u, tid) == 0;
}

// The group's first tree (`spawn_bodies_group`): sums of the child's parts' mass moments
// (`mm`, g_s0) and of its chunks' hidden displacements by stress mass (`su`, g_s4).
void group_first_tree(uint first, uint2 m, uint members, uint tid)
{
    float4 mm = float4(0.0);
    float4 su = float4(0.0);
    for (uint j = tid; j < m.y; j += 64)
    {
        uint c = first + work[members + j];
        uint4 r = part_range(c);
        for (uint k = r.x; k < r.x + r.y; k++)
        {
            float4 pc = parts[PART_WORDS * k];
            mm += float4(pc.xyz * pc.w, pc.w);
        }
        float cm = chunks[c].center.w;
        su += float4(state[4 * c].xyz * cm, cm);
    }
    tree_of(tid, mm, float4(0.0), float4(0.0), float4(0.0), su);
}

// The first tree's mass moments alone, and its displacement sums alone (each lane's sums
// as there: the tree adds each array on its own).
void group_mass_tree(uint first, uint2 m, uint members, uint tid)
{
    float4 mm = float4(0.0);
    for (uint j = tid; j < m.y; j += 64)
    {
        uint4 r = part_range(first + work[members + j]);
        for (uint k = r.x; k < r.x + r.y; k++)
        {
            float4 pc = parts[PART_WORDS * k];
            mm += float4(pc.xyz * pc.w, pc.w);
        }
    }
    tree_of(tid, mm, float4(0.0), float4(0.0), float4(0.0), float4(0.0));
}

void group_su_tree(uint first, uint2 m, uint members, uint tid)
{
    float4 su = float4(0.0);
    for (uint j = tid; j < m.y; j += 64)
    {
        uint c = first + work[members + j];
        float cm = chunks[c].center.w;
        su += float4(state[4 * c].xyz * cm, cm);
    }
    tree_of(tid, float4(0.0), float4(0.0), float4(0.0), float4(0.0), su);
}

// The group's second tree: the inertia about `com` (g_s1..g_s3), `su_total` carried in
// lane 0 of g_s4.
void group_inertia_tree(uint first, uint2 m, uint members, uint tid, float3 com, float4 su_total)
{
    float3 i0 = float3(0.0), i1 = float3(0.0), i2 = float3(0.0);
    for (uint j = tid; j < m.y; j += 64)
    {
        uint4 r = part_range(first + work[members + j]);
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
    tree_of(tid, float4(0.0), float4(i0, 0.0), float4(i1, 0.0), float4(i2, 0.0), tid == 0 ? su_total : float4(0.0));
}

// The second tree with only `su_total` carried (its value after that tree's additions).
void group_carry_tree(uint tid, float4 su_total)
{
    tree_of(tid, float4(0.0), float4(0.0), float4(0.0), float4(0.0), tid == 0 ? su_total : float4(0.0));
}

// Each lane's five sums into the shared arrays, then the fixed tree.
void tree_of(uint tid, float4 a, float4 b, float4 c, float4 d, float4 e)
{
    g_s0[tid] = a;
    g_s1[tid] = b;
    g_s2[tid] = c;
    g_s3[tid] = d;
    g_s4[tid] = e;
    spawn_tree(tid);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_bodies_group(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint slot = group.x;
    uint tid = thread.x;
    uint total = total_slots();
    // The exits as uniform values: barriers follow.
    if (group_uniform(slot >= total || control[CTL_PARENTS] == 0 || control[CTL_ABORT] != 0 ? 1u : 0u, tid) != 0)
        return;
    uint p = parent_at(slot);
    uint4 parent = parents[p];
    if (group_uniform(slot - parent.y >= parent.z || slots[slot].w == 0 ? 1u : 0u, tid) != 0)
        return;
    uint first = islands_of[p].x;
    uint2 m = child_members(slot);
    float4 rec0 = children[CHILD_WORDS * slot];
    bool anchored = asuint(rec0.w) != 0;
    uint members = total + parent.y + m.x;
    group_first_tree(first, m, members, tid);
    float mass = g_s0[0].w;
    float3 com = g_s0[0].xyz / max(mass, 1e-30);
    float4 su_total = g_s4[0];
    GroupMemoryBarrierWithGroupSync();
    group_inertia_tree(first, m, members, tid, com, su_total);
    if (tid != 0)
        return;
    float3 i0 = g_s1[0].xyz, i1 = g_s2[0].xyz, i2 = g_s3[0].xyz;
    Quat principal;
    float3 moments;
    principal_axes(i0, i1, i2, principal, moments);
    // The child's pose (`split_cluster`): the parent's, a free child's moved by its
    // chunks' mean hidden displacement (stress masses).
    Quat pose_rot = quat_of(frames[5 * p]);
    float3 pose_pos = frames[5 * p + 4].xyz;
    if (!anchored)
        pose_pos += rotate(pose_rot, g_s4[0].xyz / g_s4[0].w);
    child_record(slot, p, anchored, mass, com, principal, moments, pose_pos);
}

// `spawn_bodies_group` in three dispatches (a group per child slot), for a device that
// binds fewer storage buffers per stage than it uses (WebGPU's 10): the same trees, the
// mass properties first, the hidden displacements next (carried through the second tree
// as there), then the body.
[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_bodies_group_mass(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint slot = group.x;
    uint tid = thread.x;
    uint p;
    uint4 parent;
    if (!group_child(slot, tid, p, parent))
        return;
    uint first = islands_of[p].x;
    uint2 m = child_members(slot);
    uint members = total_slots() + parent.y + m.x;
    group_mass_tree(first, m, members, tid);
    float mass = g_s0[0].w;
    float3 com = g_s0[0].xyz / max(mass, 1e-30);
    GroupMemoryBarrierWithGroupSync();
    group_inertia_tree(first, m, members, tid, com, float4(0.0));
    if (tid != 0)
        return;
    Quat principal;
    float3 moments;
    principal_axes(g_s1[0].xyz, g_s2[0].xyz, g_s3[0].xyz, principal, moments);
    extras[2 * slot] = float4(com, mass);
    extras[2 * slot + 1] = quat_vec(principal);
    extras[params.moments_base + 2 * slot] = float4(moments, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_bodies_group_pose(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint slot = group.x;
    uint tid = thread.x;
    uint p;
    uint4 parent;
    if (!group_child(slot, tid, p, parent))
        return;
    uint first = islands_of[p].x;
    uint2 m = child_members(slot);
    uint members = total_slots() + parent.y + m.x;
    group_su_tree(first, m, members, tid);
    float4 su_total = g_s4[0];
    GroupMemoryBarrierWithGroupSync();
    group_carry_tree(tid, su_total);
    if (tid != 0)
        return;
    bool anchored = asuint(children[CHILD_WORDS * slot].w) != 0;
    Quat pose_rot = quat_of(frames[5 * p]);
    float3 pose_pos = frames[5 * p + 4].xyz;
    if (!anchored)
        pose_pos += rotate(pose_rot, g_s4[0].xyz / g_s4[0].w);
    extras[params.moments_base + 2 * slot + 1] = float4(pose_pos, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_bodies_group_record(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint slot = group.x;
    uint p;
    uint4 parent;
    if (thread.x != 0 || !child_spawns(slot, total_slots(), p, parent) || slots[slot].w == 0)
        return;
    phased_record(slot, p);
}

// A member's child slot (NONE: no body) and part range: the last child whose members
// start at or before it, and its chunk's parts.
uint4 member_info(uint i, uint total, out uint col)
{
    col = 0;
    uint p = parent_at(i);
    uint4 parent = parents[p];
    if (parent.z < 2)
        return uint4(NONE_SLOT, 0, 0, 0);
    // Its child: the last whose members start at or before it.
    uint local = i - parent.y;
    uint lo = 0, hi = parent.z - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi + 1) / 2;
        if (child_members(parent.y + mid).x <= local)
            lo = mid;
        else
            hi = mid - 1;
    }
    uint slot = parent.y + lo;
    uint4 at = slots[slot];
    if (at.w == 0)
        return uint4(NONE_SLOT, 0, 0, 0);
    uint4 r = part_range(islands_of[p].x + work[total + i]);
    col = member_offsets[i];
    return uint4(slot, at.x, r.x, r.y);
}

// A member's colliders (`info`: its child slot, body, first part, parts) from collider
// slot `col`.
void member_colliders(uint4 info, uint col)
{
    uint slot = info.x;
    float3 com = extras[2 * slot].xyz;
    Quat to_body = conj(quat_of(extras[2 * slot + 1]));
    for (uint k = info.z; k < info.z + info.w; k++)
    {
        float4 o = parts[PART_WORDS * k + 4];
        Quat rot = quat_of(parts[PART_WORDS * k + 5]);
        float4 ids = parts[PART_WORDS * k + 6];
        float4 shape_params = parts[PART_WORDS * k + 7];
        uint cw = 4 * (params.live != 0 ? col : col - control[CTL_FIRST_COLLIDER]);
        colliders[cw] = float4(asfloat(info.y), ids.x, ids.y, ids.z);
        colliders[cw + 1] = float4(rotate(to_body, o.xyz - com), o.w);
        colliders[cw + 2] = quat_vec(normalized(quat_mul(to_body, rot)));
        colliders[cw + 3] = shape_params;
        col++;
    }
}

static const uint NONE_SLOT = 0xffffffffu;

// Per member: its chunk's colliders for its child's body (`create_body`), in part order:
// at `to_body (origin - com)`, turned `to_body rotation` (the child's centre of mass and
// principal rotation from `spawn_bodies`), at the member's slots (`spawn_offsets`).
[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_colliders(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    uint total = total_slots();
    if (i >= total || control[CTL_PARENTS] == 0 || control[CTL_ABORT] != 0)
        return;
    uint col;
    uint4 info = member_info(i, total, col);
    if (info.x != NONE_SLOT)
        member_colliders(info, col);
}

// `spawn_colliders` in two dispatches, for a device that binds fewer storage buffers per
// stage than it uses (WebGPU's 10): each member's child slot and parts (into
// `member_offsets` from `params.info_base`), then its colliders.
[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_collider_members(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    uint total = total_slots();
    if (i >= total || control[CTL_PARENTS] == 0 || control[CTL_ABORT] != 0)
        return;
    uint col;
    uint4 info = member_info(i, total, col);
    for (uint w = 0; w < 4; w++)
        member_offsets[params.info_base + 4 * i + w] = info[w];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_colliders_phased(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    uint total = total_slots();
    if (i >= total || control[CTL_PARENTS] == 0 || control[CTL_ABORT] != 0)
        return;
    uint4 info = uint4(member_offsets[params.info_base + 4 * i], member_offsets[params.info_base + 4 * i + 1], member_offsets[params.info_base + 4 * i + 2], member_offsets[params.info_base + 4 * i + 3]);
    if (info.x != NONE_SLOT)
        member_colliders(info, member_offsets[i]);
}

// A split parent's engine body retired where it stands (stress-rubble `remove_body`):
// no mass, no motion, no flags, in both regions; its colliders in no pair (a category no
// mask includes, a mask that admits none: `CollisionFilter { 0x8000, 0, 0 }`). Live only.
[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_retire(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint p = group.x;
    uint tid = thread.x;
    if (p >= control[CTL_PARENTS] || params.live == 0 || parents[p].z < 2 || control[0] != 0)
        return;
    uint4 e = islands_of[p];
    // The current record, modified, into both regions (`write_bodies` writes one record twice).
    if (tid == 0)
    {
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
    }
    for (uint c = e.z + tid; c < e.w; c += 64)
    {
        float4 head = colliders[4 * c];
        uint material = asuint(head.z) & 0xFFFFu;
        colliders[4 * c] = float4(head.x, head.y, asfloat(material), asfloat(0x8000u));
    }
}
