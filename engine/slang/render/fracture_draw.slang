// Fracture viewer draw: render vertices are body-local and read the live AVBD pose
// buffer directly (no CPU transform readback). Vertex = 10 floats:
// position.xyz, normal.xyz, color.rgb, body id.
struct BodyPose { float4 p; float4 q; float4 invI; float4 v; float4 w; float4 ip; float4 iq; };
struct Camera { float4 c0; float4 c1; float4 c2; float4 c3; };

StructuredBuffer<BodyPose> gBodies;
StructuredBuffer<float> gVerts;
ConstantBuffer<Camera> gCamera;

struct VsOut
{
    float4 pos : SV_Position;
    float3 n : NORMAL;
    float3 color : COLOR;
    float3 world : POSITION1;
};

float3 rotate(float4 q, float3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }

[shader("vertex")]
VsOut vs(uint vid : SV_VertexID)
{
    uint o = vid * 10u;
    float3 p = float3(gVerts[o], gVerts[o + 1u], gVerts[o + 2u]);
    float3 n = float3(gVerts[o + 3u], gVerts[o + 4u], gVerts[o + 5u]);
    BodyPose body = gBodies[uint(gVerts[o + 9u])];
    float3 world = body.p.xyz + rotate(body.q, p);
    VsOut r;
    r.pos = gCamera.c0 * world.x + gCamera.c1 * world.y + gCamera.c2 * world.z + gCamera.c3;
    r.n = rotate(body.q, n);
    r.color = float3(gVerts[o + 6u], gVerts[o + 7u], gVerts[o + 8u]);
    r.world = world;
    return r;
}

[shader("fragment")]
float4 fs(VsOut i) : SV_Target
{
    float light = 0.38 + 0.62 * max(0.0, dot(normalize(i.n), normalize(float3(-0.5, 1.0, 0.7))));
    return float4(pow(i.color * light, float3(0.8)), 1.0);
}

[shader("fragment")]
float4 ground(VsOut i) : SV_Target
{
    float2 coord = i.world.xz;
    float2 grid = abs(frac(coord - 0.5) - 0.5) / max(fwidth(coord), float2(0.0001));
    float line = 1.0 - min(min(grid.x, grid.y), 1.0);
    return float4(lerp(float3(0.075, 0.1, 0.125), float3(0.13, 0.18, 0.21), line * 0.55), 1.0);
}

[shader("fragment")]
float4 wire(VsOut i) : SV_Target { return float4(i.color, 1.0); }
