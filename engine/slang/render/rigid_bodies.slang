// The rigid-body layer (engine/docs/playground-plan.md phase 4): every body's collider
// drawn from the live body and collider buffers, one instance per body after the ground
// (instance k draws body and collider k + 1). Boxes are drawn as boxes; spheres and
// capsules as their bounds box, ray-hit in the fragment so nothing sticks out. The ground
// is a plane. Shared by every page that draws rigid bodies (WGSL via wgsl_export).

struct Body
{
    float3 position; float invMass;
    float4 orientation;
    float3 invInertia; uint flags;
    float3 linearVel; float sleepTimer;
    float3 angularVel; uint generation;
    float3 initialPos; float initialPad;
    float4 initialRot;
};

struct Collider
{
    uint body; uint shape; uint material; uint generation;
    float3 localPos; float density;
    float4 localRot;
    float4 params;
};

// The view-projection by columns; mode: 0 position tint, 1 awake/asleep, 2 speed;
// projectile: the scene's ball (0xffffffff: every sphere is a ball); the ground plane.
struct Frame
{
    float4 c0; float4 c1; float4 c2; float4 c3;
    uint mode; uint projectile;
    // The ground square: its half side and height.
    float groundHalf; float groundY;
};
struct Eye { float3 pos; float pad; };

StructuredBuffer<Body> gBodies;
ConstantBuffer<Frame> gFrame;
StructuredBuffer<Collider> gCols;
ConstantBuffer<Eye> gEye;

float3 quatRotate(float4 q, float3 v)
{
    float3 u = q.xyz;
    return u * (2.0 * dot(u, v)) + v * (q.w * q.w - dot(u, u)) + cross(u, v) * (2.0 * q.w);
}

float4 clip(float3 w) { return gFrame.c0 * w.x + gFrame.c1 * w.y + gFrame.c2 * w.z + gFrame.c3; }

struct VsOut
{
    float4 pos : SV_Position;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float3 world : POSITION1;
    float3 center : POSITION2;
    float radius : RADIUS;
    // Capsule: world-space half segment (axis · half length); zero for spheres.
    float3 axis : AXIS;
};

// The unit cube's 36 corners (half extent 0.5) and their face normals, by vertex index,
// in the pages' order: faces along axis w = z, x, y (u, v the other two), the negative face
// first; corners (−1,−1) (1,−1) (1,1) (−1,1) in (u, v); triangles (0,1,2) (0,2,3) on a
// positive face, (0,2,1) (0,3,2) on a negative one (counter-clockwise from outside).
void cubeCorner(uint vid, out float3 corner, out float3 normal)
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
    float3 n = float3(0.0);
    n[a.z] = sign;
    corner = p;
    normal = n;
}

[shader("vertex")]
VsOut vs(uint vid : SV_VertexID, uint inst : SV_InstanceID)
{
    float3 corner, normal;
    cubeCorner(vid, corner, normal);
    Body b = gBodies[inst + 1u];
    Collider col = gCols[inst + 1u];
    float3 half = col.params.xyz;
    float radius = 0.0;
    float3 axis = float3(0.0);
    if (col.shape == 0u)
    {
        // Bounds cube of half extent r; the fragment ray-hits the sphere.
        half = float3(col.params.x);
        radius = col.params.x;
    }
    else if (col.shape == 1u)
    {
        // Capsule along local Y (radius, half length): its bounds box, ray-hit in the fragment.
        half = float3(col.params.x, col.params.y + col.params.x, col.params.x);
        radius = col.params.x;
        axis = quatRotate(b.orientation, float3(0.0, col.params.y, 0.0));
    }
    float3 world = b.position + quatRotate(b.orientation, corner * (half / 0.5));
    VsOut o;
    o.pos = clip(world);
    o.normal = quatRotate(b.orientation, normal);
    o.world = world;
    o.center = b.position;
    o.radius = radius;
    o.axis = axis;
    // The ball stays grey: the projectile when the scene names one, else every sphere.
    bool isBall = col.shape == 0u && (gFrame.projectile == 0xffffffffu || col.body == gFrame.projectile);
    if (isBall)
        o.color = float3(0.75, 0.78, 0.82);
    else if (gFrame.mode == 1u)
    {
        bool sleeping = (b.flags & 4u) != 0u;
        o.color = sleeping ? float3(0.28, 0.48, 0.82) : float3(0.95, 0.58, 0.16);
    }
    else if (gFrame.mode == 2u)
    {
        float t = clamp(length(b.linearVel) / 3.0, 0.0, 1.0);
        o.color = lerp(float3(0.22, 0.32, 0.48), float3(0.95, 0.22, 0.16), t);
    }
    else
    {
        float t = frac(b.position.y * 0.08 + b.position.x * 0.05);
        o.color = float3(0.72, 0.38 + 0.35 * t, 0.22);
    }
    return o;
}

float4 shade(float3 color, float3 n)
{
    float3 light = normalize(float3(0.35, 0.85, 0.4));
    float ndl = max(dot(n, light), 0.0);
    return float4(color * (0.28 + 0.72 * ndl), 1.0);
}

[shader("fragment")]
float4 fs(VsOut i) : SV_Target
{
    float3 n = normalize(i.normal);
    float3 eye = gEye.pos;
    if (i.radius > 0.0 && dot(i.axis, i.axis) > 0.0)
    {
        // Ray vs capsule (segment centre ± axis, radius).
        float3 rd = normalize(i.world - eye);
        float3 pa = i.center - i.axis;
        float3 ba = 2.0 * i.axis;
        float3 oa = eye - pa;
        float baba = dot(ba, ba);
        float bard = dot(ba, rd);
        float baoa = dot(ba, oa);
        float rdoa = dot(rd, oa);
        float oaoa = dot(oa, oa);
        float qa = baba - bard * bard;
        float qb = baba * rdoa - baoa * bard;
        float qc = baba * oaoa - baoa * baoa - i.radius * i.radius * baba;
        float tHit = -1.0;
        float h = qb * qb - qa * qc;
        if (h >= 0.0 && qa > 1e-8)
        {
            float t = (-qb - sqrt(h)) / qa;
            float y = baoa + t * bard;
            if (y > 0.0 && y < baba)
                tHit = t;
        }
        if (tHit < 0.0)
        {
            // End caps: the nearer sphere.
            float3 ca = eye - pa;
            float3 cb = eye - (pa + ba);
            float ba0 = dot(rd, ca);
            float ha = ba0 * ba0 - (dot(ca, ca) - i.radius * i.radius);
            float bb0 = dot(rd, cb);
            float hb = bb0 * bb0 - (dot(cb, cb) - i.radius * i.radius);
            if (ha >= 0.0)
                tHit = -ba0 - sqrt(ha);
            if (hb >= 0.0)
            {
                float tb = -bb0 - sqrt(hb);
                if (tHit < 0.0 || (tb > 0.0 && tb < tHit))
                    tHit = tb;
            }
        }
        if (tHit < 0.0)
            discard;
        float3 p = eye + rd * tHit;
        float k = clamp(dot(p - pa, ba) / baba, 0.0, 1.0);
        n = normalize(p - (pa + ba * k));
    }
    else if (i.radius > 0.0)
    {
        float3 rd = normalize(i.world - eye);
        float3 oc = eye - i.center;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - i.radius * i.radius;
        float h = b * b - c;
        if (h < 0.0)
            discard;
        float tHit = -b - sqrt(h);
        if (tHit < 0.0)
            tHit = -b + sqrt(h);
        if (tHit < 0.0)
            discard;
        n = normalize((eye + rd * tHit) - i.center);
    }
    return shade(i.color, n);
}

// The ground: a square of half side groundHalf at groundY, two triangles by vertex index.
[shader("vertex")]
VsOut vsGround(uint vid : SV_VertexID)
{
    float2 q[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    float3 corner = float3(q[vid].x * gFrame.groundHalf, gFrame.groundY, q[vid].y * gFrame.groundHalf);
    VsOut o;
    o.pos = clip(corner);
    o.normal = float3(0.0, 1.0, 0.0);
    // Keep the bodies bound in this entry point (one layout for both pipelines).
    o.color = float3(0.22, 0.24, 0.28) + gBodies[0].position * 0.0 + float3(gCols[0].density * 0.0);
    o.world = corner;
    o.center = corner;
    o.radius = 0.0;
    o.axis = float3(0.0);
    return o;
}

[shader("fragment")]
float4 fsGround(VsOut i) : SV_Target { return shade(i.color, normalize(i.normal)); }
