// Rigid body frames shared by the kernels that make engine bodies from chunk sets
// (spawn.slang, fracture.slang): principal axes (engine-schema `convex::principal_axes`),
// a rotation matrix's quaternion, and the frames' conversions: the stress solver is Z-up,
// rubble Y-up, (x, y, z) -> (x, z, -y).

import common;

// Quaternion of a rotation matrix given by columns (glam `Quat::from_rotation_axes`).
public Quat quat_from_columns(float3 c0, float3 c1, float3 c2)
{
    float m00 = c0.x, m01 = c0.y, m02 = c0.z;
    float m10 = c1.x, m11 = c1.y, m12 = c1.z;
    float m20 = c2.x, m21 = c2.y, m22 = c2.z;
    Quat q;
    if (m22 <= 0.0)
    {
        float dif10 = m11 - m00;
        float omm22 = 1.0 - m22;
        if (dif10 <= 0.0)
        {
            float f = omm22 - dif10;
            float inv = 0.5 / sqrt(f);
            q.x = f * inv; q.y = (m01 + m10) * inv; q.z = (m02 + m20) * inv; q.w = (m12 - m21) * inv;
        }
        else
        {
            float f = omm22 + dif10;
            float inv = 0.5 / sqrt(f);
            q.x = (m01 + m10) * inv; q.y = f * inv; q.z = (m12 + m21) * inv; q.w = (m20 - m02) * inv;
        }
    }
    else
    {
        float sum10 = m11 + m00;
        float opm22 = 1.0 + m22;
        if (sum10 <= 0.0)
        {
            float f = opm22 - sum10;
            float inv = 0.5 / sqrt(f);
            q.x = (m02 + m20) * inv; q.y = (m12 + m21) * inv; q.z = f * inv; q.w = (m01 - m10) * inv;
        }
        else
        {
            float f = opm22 + sum10;
            float inv = 0.5 / sqrt(f);
            q.x = (m12 - m21) * inv; q.y = (m20 - m02) * inv; q.z = (m01 - m10) * inv; q.w = f * inv;
        }
    }
    return q;
}

// engine-schema `convex::principal_axes`: Jacobi rotations on the largest off-diagonal
// entry, at most 32, the same pivot order; the rotation and the moments.
public void principal_axes(float3 r0, float3 r1, float3 r2, out Quat rot, out float3 moments)
{
    float a[3][3] = { { r0.x, r0.y, r0.z }, { r1.x, r1.y, r1.z }, { r2.x, r2.y, r2.z } };
    float v[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
    for (uint it = 0; it < 32; it++)
    {
        uint p = 0, q = 1;
        if (abs(a[0][2]) > abs(a[p][q])) { p = 0; q = 2; }
        if (abs(a[1][2]) > abs(a[p][q])) { p = 1; q = 2; }
        float scale = max(max(abs(a[0][0]), abs(a[1][1])), max(abs(a[2][2]), 1e-30));
        if (abs(a[p][q]) <= 1e-14 * scale)
            break;
        float theta = 0.5 * atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
        float s = sin(theta);
        float c = cos(theta);
        float app = a[p][p], aqq = a[q][q], apq = a[p][q];
        for (uint k = 0; k < 3; k++)
        {
            if (k != p && k != q)
            {
                float kp = a[k][p], kq = a[k][q];
                a[k][p] = c * kp - s * kq;
                a[p][k] = a[k][p];
                a[k][q] = s * kp + c * kq;
                a[q][k] = a[k][q];
            }
        }
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for (uint row = 0; row < 3; row++)
        {
            float kp = v[row][p], kq = v[row][q];
            v[row][p] = c * kp - s * kq;
            v[row][q] = s * kp + c * kq;
        }
    }
    float3 x = float3(v[0][0], v[1][0], v[2][0]);
    float3 y = float3(v[0][1], v[1][1], v[2][1]);
    float3 z = float3(v[0][2], v[1][2], v[2][2]);
    if (dot(cross(x, y), z) < 0.0)
        z = -z;
    rot = normalized(quat_from_columns(x, y, z));
    moments = float3(a[0][0], a[1][1], a[2][2]);
}

public float3 to_rubble(float3 v)
{
    return float3(v.x, v.z, -v.y);
}

// `rot_to_rubble`: -90 degrees about x, then q.
public Quat rot_to_rubble(Quat q)
{
    float h = 0.70710678118654752;
    Quat axes;
    axes.w = h;
    axes.x = -h;
    axes.y = 0.0;
    axes.z = 0.0;
    return normalized(quat_mul(axes, q));
}

public Quat conj(Quat q)
{
    Quat c;
    c.w = q.w;
    c.x = -q.x;
    c.y = -q.y;
    c.z = -q.z;
    return c;
}

// The inverse of `to_rubble`: rubble (Y-up) to the stress solver's frame (Z-up).
public float3 from_rubble(float3 v)
{
    return float3(v.x, -v.z, v.y);
}

// The inverse of `rot_to_rubble`: +90 degrees about x, then q.
public Quat rot_from_rubble(Quat q)
{
    float h = 0.70710678118654752;
    Quat axes;
    axes.w = h;
    axes.x = h;
    axes.y = 0.0;
    axes.z = 0.0;
    return normalized(quat_mul(axes, q));
}
