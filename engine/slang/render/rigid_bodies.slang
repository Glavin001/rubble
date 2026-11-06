// The rigid-body layer (engine/docs/playground-plan.md phase 4): every collider drawn from
// the live body and collider buffers (instance k: collider k + 1 on its body), boxes as
// boxes, spheres and capsules as their bounds box ray-hit in the fragment. A preset picks
// its look as data (gFrame.style): the colours (the demo's position / sleep / speed tints,
// or the terrain page's per-shape colours) and the lighting (the demo's, or sky, specular
// and fog). Plus a ground square (the demo's). WGSL via wgsl_export.

#include "common.slang"

StructuredBuffer<Body> gBodies;
ConstantBuffer<Frame> gFrame;
StructuredBuffer<Collider> gCols;

struct VsOut
{
    float4 pos : SV_Position;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float3 world : POSITION1;
    float3 center : POSITION2;
    float radius : RADIUS;
    // Demo look: the capsule's world half segment. Terrain look: its unit axis (hh: half length).
    float3 axis : AXIS;
    float hh : HALFLEN;
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
    Collider col = gCols[inst + 1u];
    Body b = gBodies[col.body];
    float4 q = b.orientation;
    bool terrainLook = gFrame.style.y == LIGHT_SKY;
    VsOut o;
    o.radius = 0.0;
    o.hh = 0.0;
    o.axis = float3(0.0);
    float3 half = col.params.xyz;
    if (col.shape == 0u)
    {
        half = float3(col.params.x);
        o.radius = col.params.x;
    }
    else if (col.shape == 1u)
    {
        half = float3(col.params.x, col.params.y + col.params.x, col.params.x);
        o.radius = col.params.x;
        o.hh = col.params.y;
    }
    float3 local = col.localPos + quatRotate(col.localRot, corner * half * gFrame.params.z);
    float3 world = b.position + quatRotate(q, local);
    o.world = world;
    o.pos = clip(gFrame, world);
    // Parked out of sight (below light.w when set): not drawn.
    if (gFrame.light.w != 0.0 && b.position.y < gFrame.light.w)
        o.pos = float4(0.0, 0.0, 2.0, 1.0);
    o.normal = quatRotate(q, quatRotate(col.localRot, normal));
    o.center = b.position + quatRotate(q, col.localPos);
    if (col.shape == 1u)
    {
        o.axis = terrainLook ? quatRotate(q, quatRotate(col.localRot, float3(0.0, 1.0, 0.0)))
                             : quatRotate(q, float3(0.0, col.params.y, 0.0));
    }
    uint scheme = gFrame.style.x;
    if (scheme == COLOR_MOTION)
    {
        bool dyn = b.invMass > 0.0 && (b.flags & 1u) != 0u;
        bool kin = (b.flags & 3u) == 2u;
        bool ccd = (b.flags & 64u) != 0u;
        if (col.shape == 1u && kin) o.color = float3(0.2, 0.75, 0.95);          // characters
        else if (kin) o.color = float3(0.35, 0.45, 0.85);                       // kinematic platforms
        else if (ccd) o.color = float3(0.95, 0.25, 0.2);                        // CCD bodies
        else if (!dyn && col.shape == 0u) o.color = float3(0.12, 0.12, 0.14);   // parked balls
        else if (!dyn) o.color = float3(0.6, 0.61, 0.65);
        else if (col.shape == 0u) o.color = float3(0.15, 0.16, 0.18);           // control balls
        else
        {
            o.color = lerp(float3(0.85, 0.55, 0.25), float3(0.9, 0.78, 0.4), hash(col.body));
            if ((b.flags & 4u) != 0u) o.color = o.color * 0.7;
        }
        return o;
    }
    if (scheme == COLOR_SHAPES)
    {
        float h = hash(col.body);
        bool asleep = (b.flags & 8u) == 0u;
        if (col.shape == 0u) o.color = lerp(float3(0.25, 0.45, 0.8), float3(0.35, 0.6, 0.9), h);
        else if (col.shape == 1u) o.color = lerp(float3(0.75, 0.3, 0.3), float3(0.9, 0.45, 0.35), h);
        else o.color = lerp(float3(0.85, 0.6, 0.25), float3(0.95, 0.8, 0.45), h);
        if (asleep) o.color = o.color * 0.65;
        return o;
    }
    // The demo's colours. The ball stays grey: the projectile when the scene names one,
    // else every sphere.
    bool isBall = col.shape == 0u && (gFrame.style.z == 0xffffffffu || col.body == gFrame.style.z);
    if (isBall)
        o.color = float3(0.75, 0.78, 0.82);
    else if (scheme == COLOR_SLEEP)
        o.color = (b.flags & 4u) != 0u ? float3(0.28, 0.48, 0.82) : float3(0.95, 0.58, 0.16);
    else if (scheme == COLOR_SPEED)
        o.color = lerp(float3(0.22, 0.32, 0.48), float3(0.95, 0.22, 0.16), clamp(length(b.linearVel) / 3.0, 0.0, 1.0));
    else
    {
        float t = frac(b.position.y * 0.08 + b.position.x * 0.05);
        o.color = float3(0.72, 0.38 + 0.35 * t, 0.22);
    }
    return o;
}

// The terrain page's capsule (or sphere, hh = 0) hit.
float capsuleHit(float3 ro, float3 rd, float3 pa, float3 pb, float r)
{
    float3 ba = pb - pa;
    float3 oa = ro - pa;
    float baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa), rdoa = dot(rd, oa), oaoa = dot(oa, oa);
    float a = baba - bard * bard, b = baba * rdoa - baoa * bard, c = baba * oaoa - baoa * baoa - r * r * baba;
    float hh = b * b - a * c;
    if (hh >= 0.0 && a > 1e-9)
    {
        float t = (-b - sqrt(hh)) / a;
        float y = baoa + t * bard;
        if (y > 0.0 && y < baba)
            return t;
        float3 oc = y <= 0.0 ? oa : ro - pb;
        float b2 = dot(rd, oc), c2 = dot(oc, oc) - r * r;
        hh = b2 * b2 - c2;
        if (hh > 0.0)
            return -b2 - sqrt(hh);
    }
    else if (a <= 1e-9)
    {
        float b2 = dot(rd, oa), c2 = oaoa - r * r, h2 = b2 * b2 - c2;
        if (h2 > 0.0)
            return -b2 - sqrt(h2);
    }
    return -1.0;
}

[shader("fragment")]
float4 fs(VsOut i) : SV_Target
{
    float3 n = normalize(i.normal);
    float3 eye = gFrame.eye.xyz;
    if (gFrame.style.y == LIGHT_SKY)
    {
        if (i.radius > 0.0)
        {
            float3 rd = normalize(i.world - eye);
            float3 pa = i.center - i.axis * i.hh;
            float3 pb = i.center + i.axis * i.hh;
            float t = capsuleHit(eye, rd, pa, pb, i.radius);
            if (t < 0.0)
                discard;
            float3 p = eye + rd * t;
            float s = clamp(dot(p - pa, i.axis), 0.0, 2.0 * i.hh);
            n = normalize(p - (pa + i.axis * s));
        }
        return float4(shade(gFrame, i.color, n, i.world), 1.0);
    }
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
    return float4(shade(gFrame, i.color, n, i.world), 1.0);
}

// The ground: a square of half side params.x at params.y, two triangles by vertex index.
[shader("vertex")]
VsOut vsGround(uint vid : SV_VertexID)
{
    float2 q[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    float3 corner = float3(q[vid].x * gFrame.params.x, gFrame.params.y, q[vid].y * gFrame.params.x);
    VsOut o;
    o.pos = clip(gFrame, corner);
    o.normal = float3(0.0, 1.0, 0.0);
    // Keep every buffer bound in this entry point (one layout for both pipelines).
    o.color = float3(0.22, 0.24, 0.28) + gBodies[0].position * 0.0 + float3(gCols[0].density * 0.0);
    o.world = corner;
    o.center = corner;
    o.radius = 0.0;
    o.axis = float3(0.0);
    o.hh = 0.0;
    return o;
}

[shader("fragment")]
float4 fsGround(VsOut i) : SV_Target
{
    if (gFrame.style.y == LIGHT_SKY)
    {
        // The motion page's floor: a 2 m grid on a checker.
        float2 g = abs(frac(i.world.xz / 2.0) - 0.5);
        float line = smoothstep(0.47, 0.5, max(g.x, g.y));
        float checker = step(0.0, sin(i.world.x * 0.785) * sin(i.world.z * 0.785));
        float3 base = lerp(float3(0.26, 0.29, 0.27), float3(0.3, 0.33, 0.31), checker);
        return float4(shade(gFrame, lerp(base, float3(0.45, 0.48, 0.46), line * 0.6), float3(0.0, 1.0, 0.0), i.world), 1.0);
    }
    return float4(shade(gFrame, i.color, normalize(i.normal), i.world), 1.0);
}
