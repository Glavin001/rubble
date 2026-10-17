#include "types.slang"
#include "math3d.slang"

public float3 supportSphere(float3 center, float radius, float3 dir)
{
    float3 d = normalize(dir);
    return center + d * radius;
}

public float3 supportBox(float3 center, float4 rot, float3 half, float3 dir)
{
    float3 localDir = quatRotate(quatConjugate(rot), dir);
    float3 local = float3(
        localDir.x >= 0.0 ? half.x : -half.x,
        localDir.y >= 0.0 ? half.y : -half.y,
        localDir.z >= 0.0 ? half.z : -half.z);
    return transformPoint(center, rot, local);
}

public float3 supportCapsule(float3 center, float4 rot, float radius, float halfLen, float3 dir)
{
    float3 axis = quatRotate(rot, float3(0, 1, 0));
    float3 d = normalize(dir);
    float s = dot(axis, d) >= 0.0 ? halfLen : -halfLen;
    return center + axis * s + d * radius;
}

public float3 supportConvex(
    float3 center, float4 rot,
    StructuredBuffer<float3> verts, uint start, uint count,
    float3 dir)
{
    float3 localDir = quatRotate(quatConjugate(rot), dir);
    float best = -1e30;
    float3 bestV = float3(0, 0, 0);
    for (uint i = 0; i < count; i++)
    {
        float3 v = verts[start + i];
        float s = dot(v, localDir);
        if (s > best) { best = s; bestV = v; }
    }
    return transformPoint(center, rot, bestV);
}

public void worldAabbSphere(float3 c, float r, out float3 mn, out float3 mx)
{
    float3 e = float3(r, r, r);
    mn = c - e; mx = c + e;
}

public void worldAabbBox(float3 c, float4 q, float3 half, out float3 mn, out float3 mx)
{
    float3x3 R = quatToMatrix(q);
    float3 e = abs(float3(R[0][0], R[1][0], R[2][0])) * half.x
             + abs(float3(R[0][1], R[1][1], R[2][1])) * half.y
             + abs(float3(R[0][2], R[1][2], R[2][2])) * half.z;
    mn = c - e; mx = c + e;
}

public void worldAabbCapsule(float3 c, float4 q, float radius, float halfLen, out float3 mn, out float3 mx)
{
    float3 axis = quatRotate(q, float3(0, 1, 0)) * halfLen;
    float3 e = abs(axis) + float3(radius, radius, radius);
    mn = c - e; mx = c + e;
}

public bool sphereSphere(
    float3 cA, float rA, float3 cB, float rB, float margin,
    out float3 normal, out float penetration, out float3 pA, out float3 pB)
{
    float3 d = cB - cA;
    float dist = length(d);
    float r = rA + rB + margin;
    if (dist > r && dist > 1e-8) return false;
    normal = dist > 1e-8 ? d / dist : float3(0, 1, 0);
    penetration = r - dist;
    pA = cA + normal * rA;
    pB = cB - normal * rB;
    return true;
}

// Coupled Coulomb friction disk projection (not two independent tangent clamps).
public float3 projectContactForce(float3 f, float mu)
{
    // f.x = normal (negative = compression), f.yz = tangent
    f.x = min(f.x, 0.0);
    float ft = length(f.yz);
    float maxT = mu * (-f.x);
    if (ft > maxT && ft > 1e-12)
    {
        f.yz *= maxT / ft;
    }
    return f;
}
