// Fracture lab draw, shared by the browser (WGSL) and the Metal viewer (MSL).
// Render vertices are body-local and read the live AVBD pose buffer directly
// (no CPU transform readback). Vertex = 10 floats: position.xyz, normal.xyz,
// color.rgb, body id (a bond line's normal: see `vs`). Instanced house copies: copy k draws the same mesh on
// bodies id + k * instance stride (body 0, the static ground, is shared).
struct BodyPose { float4 p; float4 q; float4 invI; float4 v; float4 w; float4 ip; float4 iq; };
// Column-major view-projection; mode.x = instance stride, mode.y = colour mode
// (0 materials, 1 sleep state: awake orange, asleep blue, static grey, 2 bodies: one
// colour per body, hues a golden-ratio step apart), mode.z = joints
// per copy (0: the scene runs without joints).
struct View { float4 c0; float4 c1; float4 c2; float4 c3; uint4 mode; };

StructuredBuffer<BodyPose> gBodies;
StructuredBuffer<float> gVerts;
ConstantBuffer<View> gView;
// The engine's joint records (engine-schema Joint, JOINT_WORDS words each; flags at word 2,
// bit 0 broken). Bond line k (vertices 2k, 2k + 1) is joint k of its copy.
StructuredBuffer<uint> gJoints;
static const uint JOINT_WORDS = 72u;
static const uint JOINT_FLAG_BROKEN = 1u;

struct VsOut
{
    float4 pos : SV_Position;
    float3 n : NORMAL;
    float3 color : COLOR;
    float3 world : POSITION1;
};

float3 rotate(float4 q, float3 v) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }

// Colour `slot` of the bodies view (as stress-rubble render.rs `slot_color`).
float3 slotColor(uint slot)
{
    float h = frac(float(slot) * 0.6180339887) * 6.0;
    float s = 0.62, v = 0.92, f = frac(h);
    float p = v * (1.0 - s), q = v * (1.0 - s * f), t = v * (1.0 - s * (1.0 - f));
    uint k = uint(h);
    return k == 0u ? float3(v, t, p) : k == 1u ? float3(q, v, p) : k == 2u ? float3(p, v, t)
         : k == 3u ? float3(p, q, v) : k == 4u ? float3(t, p, v) : float3(v, p, q);
}

[shader("vertex")]
VsOut vs(uint vid : SV_VertexID, uint copy : SV_InstanceID)
{
    uint o = vid * 10u;
    float3 p = float3(gVerts[o], gVerts[o + 1u], gVerts[o + 2u]);
    float3 n = float3(gVerts[o + 3u], gVerts[o + 4u], gVerts[o + 5u]);
    uint id = uint(gVerts[o + 9u]);
    uint index = id == 0u ? 0u : id + copy * gView.mode.x;
    BodyPose body = gBodies[index];
    float3 world = body.p.xyz + rotate(body.q, p);
    VsOut r;
    r.pos = gView.c0 * world.x + gView.c1 * world.y + gView.c2 * world.z + gView.c3;
    // A bond line (the bond graph): normal = (-(partner + 2), rest distance, rest height),
    // below any unit normal. With joints it is hidden once its joint breaks. Without them
    // (pieces sleep until hit) it is hidden once its pieces part (their centres more than
    // a slack past the rest distance), or, for an anchor (partner 0), once the piece
    // leaves its rest height. Either way a broken bond stops showing as a connection.
    if (n.x < -1.5)
    {
        bool parted;
        if (gView.mode.z > 0u)
        {
            uint joint = vid / 2u + copy * gView.mode.z;
            parted = (gJoints[joint * JOINT_WORDS + 2u] & JOINT_FLAG_BROKEN) != 0u;
        }
        else
        {
            uint partner = uint(-n.x - 2.0 + 0.5);
            float slack = 0.02 + 0.02 * n.y;
            parted = partner == 0u
                ? abs(body.p.y - n.z) > slack
                : abs(length(gBodies[partner + copy * gView.mode.x].p.xyz - body.p.xyz) - n.y) > slack;
        }
        if (parted)
            r.pos = float4(0.0, 0.0, 2.0, 1.0); // past the far plane: clipped
        n = float3(0.0, 1.0, 0.0);
    }
    r.n = rotate(body.q, n);
    r.color = float3(gVerts[o + 6u], gVerts[o + 7u], gVerts[o + 8u]);
    if (gView.mode.y == 1u && index != 0u)
    {
        // Body flags live in invI.w: dynamic 1, sleeping 4, awake 8.
        uint flags = asuint(body.invI.w);
        bool asleep = (flags & 4u) != 0u;
        r.color = (flags & 1u) != 0u && !asleep ? float3(1.0, 0.45, 0.12) : asleep ? float3(0.25, 0.45, 0.85) : float3(0.45, 0.45, 0.48);
    }
    else if (gView.mode.y == 2u && index != 0u)
        r.color = slotColor(index - 1u);
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
