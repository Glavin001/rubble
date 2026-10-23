// Soft-body render-mesh skinning (engine_softbody::SkinBinding), standalone:
// run by an integration after the step, outside the physics plan. It reads the
// body buffer (binding 0 of the step) and writes a vertex stream any renderer
// can draw: positions, smooth normals, and per-triangle/per-vertex crumple
// strain (for damage shading) and tears.
//
//   soft_skin_vertices  p = c + a·(x − c) + b·(y − c) + d·((x − c) × (y − c));
//                       torn when the frame stretched past 1.8 × its rest edge
//   soft_skin_faces     per triangle: normal (area-weighted), strain against the
//                       rest triangle, −1 when torn
//   soft_skin_normals   per vertex: normalised sum of its triangles' normals and
//                       the largest strain around it
//
// Records: binding (2 × uint4 per vertex: c, x, y, flags | coords xyz, rest
// edge as float bits), rest position (float4 per vertex), triangle (uint4: a, b,
// c, user), adjacency (vertex count + 1 starts, then triangle indices).

struct BodyState
{
    float3 position;
    float invMass;
    float4 orientation;
    float3 invInertia;
    uint flags;
    float3 linearVel;
    float sleepTimer;
    float3 angularVel;
    uint generation;
    float3 initialPos;
    float initialPad;
    float4 initialRot;
};

struct SkinParams
{
    uint vertices;
    uint triangles;
    float tearRatio; // frame stretch that tears (1.8)
    float pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<BodyState> gBodies;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> gSkinBind;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gSkinRest;
[[vk::binding(3, 0)]] StructuredBuffer<uint4> gSkinTris;
[[vk::binding(4, 0)]] StructuredBuffer<uint> gSkinAdj;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> gSkinPos;   // xyz, w = 1 when torn
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> gSkinFace;  // normal, w = strain or −1 (torn)
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> gSkinNormal; // normal, w = max strain around
[[vk::binding(8, 0)]] ConstantBuffer<SkinParams> gSkin;

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_skin_vertices(uint3 tid : SV_DispatchThreadID)
{
    uint v = tid.x;
    if (v >= gSkin.vertices) return;
    uint4 b0 = gSkinBind[v * 2u];
    float4 b1 = asfloat(gSkinBind[v * 2u + 1u]);
    float3 c = gBodies[b0.x].position;
    float3 ex = gBodies[b0.y].position - c;
    float3 ey = gBodies[b0.z].position - c;
    float stretch = max(length(ex), length(ey)) / max(b1.w, 1e-4);
    float3 p = c + ex * b1.x + ey * b1.y + cross(ex, ey) * b1.z;
    bool torn = b1.w > 0.0 && stretch > gSkin.tearRatio;
    gSkinPos[v] = float4(p, torn ? 1.0 : 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_skin_faces(uint3 tid : SV_DispatchThreadID)
{
    uint t = tid.x;
    if (t >= gSkin.triangles) return;
    uint4 f = gSkinTris[t];
    float4 a = gSkinPos[f.x];
    float4 b = gSkinPos[f.y];
    float4 c = gSkinPos[f.z];
    float3 ra = gSkinRest[f.x].xyz;
    float3 rb = gSkinRest[f.y].xyz;
    float3 rc = gSkinRest[f.z].xyz;
    float la = length(a.xyz - b.xyz) / max(length(ra - rb), 1e-4);
    float lb = length(b.xyz - c.xyz) / max(length(rb - rc), 1e-4);
    float lc = length(c.xyz - a.xyz) / max(length(rc - ra), 1e-4);
    float strain = max(max(abs(la - 1.0), abs(lb - 1.0)), abs(lc - 1.0));
    bool torn = a.w + b.w + c.w > 0.0 || max(max(la, lb), lc) > 2.5;
    float3 n = cross(b.xyz - a.xyz, c.xyz - a.xyz);
    gSkinFace[t] = float4(n, torn ? -1.0 : strain);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_skin_normals(uint3 tid : SV_DispatchThreadID)
{
    uint v = tid.x;
    if (v >= gSkin.vertices) return;
    float3 n = float3(0.0, 0.0, 0.0);
    float strain = 0.0;
    uint end = gSkinAdj[v + 1u];
    for (uint k = gSkinAdj[v]; k < end; k++)
    {
        float4 face = gSkinFace[gSkinAdj[k]];
        if (face.w < 0.0) continue;
        n += face.xyz;
        strain = max(strain, face.w);
    }
    float l = length(n);
    gSkinNormal[v] = float4(l > 1e-12 ? n / l : float3(0.0, 1.0, 0.0), strain);
}
