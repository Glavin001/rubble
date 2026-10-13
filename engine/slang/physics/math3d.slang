// Rigid transforms, quaternions (xyzw), and small linear algebra.

public float4 quatIdentity()
{
    return float4(0.0, 0.0, 0.0, 1.0);
}

public float4 quatNormalize(float4 q)
{
    float n = length(q);
    return n > 1e-12 ? q / n : quatIdentity();
}

public float4 quatMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

public float4 quatConjugate(float4 q)
{
    return float4(-q.x, -q.y, -q.z, q.w);
}

public float3 quatRotate(float4 q, float3 v)
{
    float3 u = q.xyz;
    float s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

// Integrate local rotation increment (Rodrigues-style chart) onto a unit quaternion.
public float4 quatApplyDelta(float4 q, float3 dTheta)
{
    float angle = length(dTheta);
    float4 dq;
    if (angle < 1e-8)
    {
        dq = float4(0.5 * dTheta, 1.0);
    }
    else
    {
        float half = 0.5 * angle;
        float s = sin(half) / angle;
        dq = float4(dTheta * s, cos(half));
    }
    return quatNormalize(quatMul(dq, q));
}

public float3x3 quatToMatrix(float4 q)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return float3x3(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy));
}

public float3 transformPoint(float3 pos, float4 rot, float3 local)
{
    return pos + quatRotate(rot, local);
}

public void orthonormalBasis(float3 n, out float3 t1, out float3 t2)
{
    float3 a = abs(n.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
    t1 = normalize(cross(n, a));
    t2 = cross(n, t1);
}

public float3x3 mat3Diagonal(float3 d)
{
    return float3x3(d.x, 0, 0, 0, d.y, 0, 0, 0, d.z);
}

public float3x3 mat3Add(float3x3 a, float3x3 b)
{
    return float3x3(
        a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2],
        a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2],
        a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]);
}

public float3 mat3MulVec(float3x3 m, float3 v)
{
    return float3(dot(float3(m[0][0], m[0][1], m[0][2]), v),
                  dot(float3(m[1][0], m[1][1], m[1][2]), v),
                  dot(float3(m[2][0], m[2][1], m[2][2]), v));
}
