// Scale-aware 6×6 LDLT with pivot floor. Matches production AVBD local solve needs.

public struct Mat6
{
    float m[36];
};

public struct Vec6
{
    float v[6];
};

public Mat6 mat6Zero()
{
    Mat6 a;
    for (int i = 0; i < 36; i++) a.m[i] = 0.0;
    return a;
}

public Vec6 vec6Zero()
{
    Vec6 a;
    for (int i = 0; i < 6; i++) a.v[i] = 0.0;
    return a;
}

public void mat6AddOuter(inout Mat6 a, Vec6 j, float k)
{
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++)
            a.m[r * 6 + c] += k * j.v[r] * j.v[c];
}

public void mat6AddDiag(inout Mat6 a, float3 lin, float3 ang)
{
    a.m[0] += lin.x; a.m[7] += lin.y; a.m[14] += lin.z;
    a.m[21] += ang.x; a.m[28] += ang.y; a.m[35] += ang.z;
}

public void vec6AddScaled(inout Vec6 a, Vec6 j, float s)
{
    for (int i = 0; i < 6; i++) a.v[i] += j.v[i] * s;
}

// Solve A x = b for SPD A via LDLT. Returns pivot-floor hit count in floorHits.
public Vec6 solve6x6(Mat6 A, Vec6 b, float pivotFloor, inout uint floorHits)
{
    float a[36];
    for (int i = 0; i < 36; i++) a[i] = A.m[i];

    // Jacobi precondition scale
    float scale[6];
    for (int i = 0; i < 6; i++)
    {
        float d = a[i * 6 + i];
        scale[i] = d > 1e-12 ? rsqrt(d) : 1.0;
    }
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++)
            a[r * 6 + c] *= scale[r] * scale[c];

    float L[36];
    float D[6];
    for (int i = 0; i < 36; i++) L[i] = 0.0;
    for (int i = 0; i < 6; i++)
    {
        L[i * 6 + i] = 1.0;
        float diag = a[i * 6 + i];
        for (int k = 0; k < i; k++)
        {
            float lik = L[i * 6 + k];
            diag -= lik * lik * D[k];
        }
        if (diag < pivotFloor)
        {
            diag = pivotFloor;
            floorHits += 1u;
        }
        D[i] = diag;
        for (int j = i + 1; j < 6; j++)
        {
            float v = a[j * 6 + i];
            for (int k = 0; k < i; k++)
                v -= L[j * 6 + k] * L[i * 6 + k] * D[k];
            L[j * 6 + i] = v / diag;
        }
    }

    float y[6];
    for (int i = 0; i < 6; i++)
    {
        float s = b.v[i] * scale[i];
        for (int k = 0; k < i; k++) s -= L[i * 6 + k] * y[k];
        y[i] = s;
    }
    float z[6];
    for (int i = 0; i < 6; i++) z[i] = y[i] / D[i];
    float x[6];
    for (int i = 5; i >= 0; i--)
    {
        float s = z[i];
        for (int k = i + 1; k < 6; k++) s -= L[k * 6 + i] * x[k];
        x[i] = s;
    }

    Vec6 outv = vec6Zero();
    for (int i = 0; i < 6; i++) outv.v[i] = x[i] * scale[i];
    return outv;
}
