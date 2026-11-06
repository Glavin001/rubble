// The projectile layer: every live projectile as a tracer (a box stretched along its
// velocity), read from the world's projectile pool records (position, velocity, flags: bit
// 0 alive) at gFrame.pool.x + slot × gFrame.pool.y. Unlit.

#include "common.slang"

ConstantBuffer<Frame> gFrame;
StructuredBuffer<float4> gPool;

struct VsOut
{
    float4 pos : SV_Position;
    float3 color : COLOR;
};

// The unit cube's corners by vertex index (the rigid layer's order).
void cubeCorner(uint vid, out float3 corner)
{
    uint face = vid / 6u;
    uint k = vid % 6u;
    uint3 axes[3] = { uint3(0u, 1u, 2u), uint3(1u, 2u, 0u), uint3(2u, 0u, 1u) };
    uint3 a = axes[face / 2u];
    float sign = (face % 2u == 0u) ? -1.0 : 1.0;
    uint pos[6] = { 0u, 1u, 2u, 0u, 2u, 3u };
    uint neg[6] = { 0u, 2u, 1u, 0u, 3u, 2u };
    uint c = sign > 0.0 ? pos[k] : neg[k];
    float2 ab[4] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, 1) };
    float3 p = float3(0.0);
    p[a.x] = ab[c].x * 0.5;
    p[a.y] = ab[c].y * 0.5;
    p[a.z] = sign * 0.5;
    corner = p;
}

[shader("vertex")]
VsOut vs(uint vid : SV_VertexID, uint inst : SV_InstanceID)
{
    float3 corner;
    cubeCorner(vid, corner);
    uint rec = gFrame.pool.x + inst * gFrame.pool.y;
    float4 p = gPool[rec];
    float4 v = gPool[rec + 1u];
    bool alive = (asuint(gPool[rec + 2u].x) & 1u) != 0u;
    float3 dir = normalize(v.xyz + float3(0.0, 1e-6, 0.0));
    float len = clamp(length(v.xyz) * 0.004, 0.2, 3.0);
    float3 up = abs(dir.y) > 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    float3 x = normalize(cross(up, dir));
    float3 y = cross(dir, x);
    float3 w = p.xyz + x * corner.x * 0.06 + y * corner.y * 0.06 - dir * (corner.z + 0.5) * len;
    VsOut o;
    o.pos = alive ? clip(gFrame, w) : float4(0.0, 0.0, 2.0, 1.0);
    o.color = float3(1.0, 0.85, 0.3);
    return o;
}

[shader("fragment")]
float4 fs(VsOut i) : SV_Target { return float4(i.color, 1.0); }
