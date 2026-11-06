// The heightfield layer: a heightfield collider drawn straight from its physics records in
// the hull arena (engine_schema::heightfield: header (magic, nx, nz, flags), (dx, dz, min,
// max), then the heights, four per record), at gFrame.style.w. Its triangles split as the
// collider's; normals from central differences; the terrain page's grass, soil and grid.

#include "common.slang"

ConstantBuffer<Frame> gFrame;
StructuredBuffer<uint4> gHulls;

struct VsOut
{
    float4 pos : SV_Position;
    float3 normal : NORMAL;
    float3 world : POSITION1;
};

float hfH(uint base, uint i, uint j, uint nx)
{
    uint k = j * nx + i;
    uint4 r = gHulls[base + 2u + (k >> 2u)];
    return asfloat(r[k & 3u]);
}

[shader("vertex")]
VsOut vs(uint vi : SV_VertexID)
{
    uint base = gFrame.style.w;
    uint4 head = gHulls[base];
    uint nx = head.y, nz = head.z;
    float4 d = asfloat(gHulls[base + 1u]);
    uint cell = vi / 6u;
    uint ci = cell % (nx - 1u), cj = cell / (nx - 1u);
    // Triangles (00, 11, 10) and (00, 01, 11): the collider's diagonal split.
    uint2 corners[6] = { uint2(0u, 0u), uint2(1u, 1u), uint2(1u, 0u), uint2(0u, 0u), uint2(0u, 1u), uint2(1u, 1u) };
    uint2 corner = corners[vi % 6u];
    uint i = ci + corner.x, j = cj + corner.y;
    float x0 = -0.5 * float(nx - 1u) * d.x, z0 = -0.5 * float(nz - 1u) * d.y;
    float h = hfH(base, i, j, nx);
    float hx0 = hfH(base, max(i, 1u) - 1u, j, nx), hx1 = hfH(base, min(i + 1u, nx - 1u), j, nx);
    float hz0 = hfH(base, i, max(j, 1u) - 1u, nx), hz1 = hfH(base, i, min(j + 1u, nz - 1u), nx);
    VsOut o;
    o.world = float3(x0 + float(i) * d.x, h, z0 + float(j) * d.y);
    o.normal = normalize(float3(-(hx1 - hx0) / (2.0 * d.x), 1.0, -(hz1 - hz0) / (2.0 * d.y)));
    o.pos = clip(gFrame, o.world);
    return o;
}

[shader("fragment")]
float4 fs(VsOut i) : SV_Target
{
    float3 n = normalize(i.normal);
    float2 g = abs(frac(i.world.xz) - 0.5);
    float line = smoothstep(0.47, 0.5, max(g.x, g.y));
    float slope = 1.0 - n.y;
    float3 grass = lerp(float3(0.33, 0.42, 0.25), float3(0.42, 0.37, 0.27), clamp(slope * 4.0, 0.0, 1.0));
    float low = clamp(-i.world.y * 0.4, 0.0, 1.0);
    float3 base = lerp(grass, float3(0.38, 0.3, 0.22), low);
    return float4(shade(gFrame, lerp(base, base * 1.25, line * 0.4), n, i.world), 1.0);
}
