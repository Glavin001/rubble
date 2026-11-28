// Continuous contact's arithmetic (docs/continuous-contact-design.md, "The GPU port"): the
// twins of crates/engine-softbody/src/explicit_contact/ccd.rs's closest, accd, sin_cos,
// inverse3 and to_fixed_vel, in strict float (the CPU's order, operation for operation), so
// GPU == CPU bit for bit with bit-exact on. Needs strict.slang and feature_query.slang.

// Units per m/s of the per-node velocity sums (ccd.rs VEL_FIXED, 2²²).
static const float CC_VEL_FIXED = 4194304.0;

// x in 1/CC_VEL_FIXED m/s units, clamped to ±2·10⁹, rounded to nearest, ties to even
// (ccd.rs to_fixed_vel; spelt out as toFixedShift).
int ccToFixedVel(float x)
{
    float y = clamp(nf(x * CC_VEL_FIXED), -2.0e9, 2.0e9);
    float f = floor(y);
    float d = nf(y - f);
    if (d > 0.5 || (d == 0.5 && floor(f * 0.5) * 2.0 != f)) f = f + 1.0;
    return int(f);
}

float ccLen(float3 v) { return ssqrtCr(sdot(v, v)); }

// A pair's closest points at positions p (the first nn points one side: a triangle and a
// point, nn 3, or two edges, nn 2): the weights signed by side (+ on the second) and the gap
// vector from the first side to the second (ccd.rs closest).
float3 ccClosest(uint nn, float3 p[4], out float4 w)
{
    if (nn == 3u)
    {
        float3 b;
        float3 c = fqClosestTri(p[3], p[0], p[1], p[2], b);
        w = float4(-b.x, -b.y, -b.z, 1.0);
        return ssub3(p[3], c);
    }
    float2 st = fqSegSeg(p[0], p[1], p[2], p[3]);
    float3 c1 = sadd3(p[0], sscale(ssub3(p[1], p[0]), st.x));
    float3 c2 = sadd3(p[2], sscale(ssub3(p[3], p[2]), st.y));
    w = float4(-nf(1.0 - st.x), -st.x, nf(1.0 - st.y), st.y);
    return ssub3(c2, c1);
}

float ccGap(uint nn, float3 p[4])
{
    float4 w;
    return ccLen(ccClosest(nn, p, w));
}

// (d² − ξ²) / (d + ξ), as accd spells it.
float ccRoom(float d, float xi) { return sdiv(nf(smul(d, d) - smul(xi, xi)), nf(d + xi)); }

// Additive CCD (ccd.rs accd): the first time in [0, 1] the gap of the pair at p moving by dp
// closes to xi; -1 when it stays above. At most 32 iterations; at the cap the time reached.
float ccAccd(uint nn, float3 p[4], float3 dp[4], float xi)
{
    const float S = 0.1;
    float3 mean = sscale(sadd3(sadd3(sadd3(dp[0], dp[1]), dp[2]), dp[3]), 0.25);
    float3 d[4];
    for (uint j = 0u; j < 4u; j++) d[j] = ssub3(dp[j], mean);
    float m0 = 0.0;
    float m1 = 0.0;
    for (uint j = 0u; j < 4u; j++)
    {
        float l = ccLen(d[j]);
        if (j < nn) m0 = max(m0, l); else m1 = max(m1, l);
    }
    float lp = nf(m0 + m1);
    if (lp <= 0.0) return -1.0;
    float3 x[4] = p;
    float dist = ccGap(nn, x);
    if (dist <= xi) return 0.0;
    float g = sdiv(smul(S, nf(smul(dist, dist) - smul(xi, xi))), nf(dist + xi));
    float t = 0.0;
    float tl = sdiv(smul(nf(1.0 - S), nf(smul(dist, dist) - smul(xi, xi))), smul(nf(dist + xi), lp));
    for (uint it = 0u; it < 32u; it++)
    {
        for (uint j = 0u; j < 4u; j++) x[j] = sadd3(x[j], sscale(d[j], tl));
        float dd = ccGap(nn, x);
        if (it > 0u && ccRoom(dd, xi) < g) return t;
        t = nf(t + tl);
        if (t > 1.0) return -1.0;
        tl = sdiv(smul(0.9, nf(smul(dd, dd) - smul(xi, xi))), smul(nf(dd + xi), lp));
    }
    return t;
}

// sin and cos of t (|t| ≤ 1) by their Taylor series, Horner (ccd.rs sin_cos).
float2 ccSinCos(float t)
{
    float t2 = smul(t, t);
    float s = smul(t, nf(1.0 + smul(t2, nf(-1.0 / 6.0 + smul(t2, nf(1.0 / 120.0 + smul(t2, nf(-1.0 / 5040.0
              + smul(t2, nf(1.0 / 362880.0 + smul(t2, -1.0 / 39916800.0)))))))))));
    float c = nf(1.0 + smul(t2, nf(-0.5 + smul(t2, nf(1.0 / 24.0 + smul(t2, nf(-1.0 / 720.0
              + smul(t2, nf(1.0 / 40320.0 + smul(t2, -1.0 / 3628800.0))))))))));
    return float2(s, c);
}

// The inverse of the 3×3 matrix m (rows) by its adjugate (ccd.rs inverse3); false when
// singular.
bool ccInverse3(float3 m[3], out float3 r[3])
{
    float c00 = nf(smul(m[1].y, m[2].z) - smul(m[1].z, m[2].y));
    float c01 = nf(smul(m[1].z, m[2].x) - smul(m[1].x, m[2].z));
    float c02 = nf(smul(m[1].x, m[2].y) - smul(m[1].y, m[2].x));
    float det = nf(nf(smul(m[0].x, c00) + smul(m[0].y, c01)) + smul(m[0].z, c02));
    r[0] = float3(0.0, 0.0, 0.0);
    r[1] = float3(0.0, 0.0, 0.0);
    r[2] = float3(0.0, 0.0, 0.0);
    if (abs(det) <= 1e-12) return false;
    float k = sdiv(1.0, det);
    float c10 = nf(smul(m[0].z, m[2].y) - smul(m[0].y, m[2].z));
    float c11 = nf(smul(m[0].x, m[2].z) - smul(m[0].z, m[2].x));
    float c12 = nf(smul(m[0].y, m[2].x) - smul(m[0].x, m[2].y));
    float c20 = nf(smul(m[0].y, m[1].z) - smul(m[0].z, m[1].y));
    float c21 = nf(smul(m[0].z, m[1].x) - smul(m[0].x, m[1].z));
    float c22 = nf(smul(m[0].x, m[1].y) - smul(m[0].y, m[1].x));
    r[0] = float3(smul(c00, k), smul(c10, k), smul(c20, k));
    r[1] = float3(smul(c01, k), smul(c11, k), smul(c21, k));
    r[2] = float3(smul(c02, k), smul(c12, k), smul(c22, k));
    return true;
}
