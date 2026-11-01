// The engine's chunk colliders moved with their chunks' deformation on the GPU, each
// resident frame before the engine steps (the host-coupled frame's `World::shape_poses` and
// stress-rubble `move_shapes`): where the stress solve has the chunk, its hidden
// displacement u and rotation th included.
//   shapes_deform  per engine collider with a rest record: its chunk (the coupling's
//                  collider table) moved by u and turned by th about the chunk's centre
//
// A collider's undeformed placement in its body frame is (p0, r0), the body frame being
// `to_body (x - com)`, and its chunk's centre there is cb = to_body (c - com). Turned by
// th about c and moved by u in the cluster frame:
//   p = cb + to_body u + R (p0 - cb),  r = R r0,  R = to_body rot(th) to_body^-1.

import common;

struct ShapeParams
{
    uint collider_count;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<ShapeParams> params;
// Per engine collider (4 float4): chunk centre in the body frame (w: 1 for a chunk's
// collider, 0 otherwise); to_body (x, y, z, w); undeformed local position (w: -); its
// undeformed local rotation (x, y, z, w).
[[vk::binding(1, 0)]] StructuredBuffer<float4> rest;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> collider_info; // two uint4 per collider (`ColliderInfo` a, b)
[[vk::binding(3, 0)]] StructuredBuffer<float4> state;        // 4 float4 per layout chunk (u, th, ...)
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> colliders;  // the engine's (4 float4 per `Collider`)

static const uint NONE = 0xffffffffu;
static const uint KIND_BODY = 2;

[shader("compute")]
[numthreads(64, 1, 1)]
void shapes_deform(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.collider_count)
        return;
    float4 cb = rest[4 * i];
    uint4 info = collider_info[2 * i];
    // Static geometry, an impactor, a retired body's (the GPU's split marks them static).
    if (cb.w == 0.0 || info.z != KIND_BODY || info.x == NONE)
        return;
    uint k = info.x;
    float3 u = state[4 * k].xyz;
    float3 th = state[4 * k + 1].xyz;
    Quat to_body = quat_of(rest[4 * i + 1]);
    Quat turn = quat_mul(quat_mul(to_body, from_axis_angle(th, length(th))), conj_of(to_body));
    float3 p0 = rest[4 * i + 2].xyz;
    Quat r0 = quat_of(rest[4 * i + 3]);
    float3 p = cb.xyz + rotate(to_body, u) + rotate(turn, p0 - cb.xyz);
    float density = colliders[4 * i + 1].w;
    colliders[4 * i + 1] = float4(p, density);
    colliders[4 * i + 2] = quat_vec(normalized(quat_mul(turn, r0)));
}

Quat conj_of(Quat q)
{
    Quat c;
    c.w = q.w;
    c.x = -q.x;
    c.y = -q.y;
    c.z = -q.z;
    return c;
}
