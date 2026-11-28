// Strict single-precision arithmetic (shared: the explicit vehicles' kernel, the shared
// closest-feature queries in feature_query.slang, and any caller that needs the CPU
// reference's bits).
//
// Strict float arithmetic where a decision or a bit trick depends on rounding.
// Metal compiles with fast math (fused multiply-add, reassociation, approximate
// division and square root). Then a contacter lying in a triangle's plane can see
// the other sign of δ than the CPU reference and be pushed out the other side, and
// the bit-trick exponential (a·x + b to integer bits) can land on other bits. These
// helpers give IEEE single precision in the CPU's order: every intermediate passes
// through an XOR with a uniform the compiler cannot know is zero (no fusing or
// reassociation across it); division and square root take one exact fma
// correction.
//
// The includer defines `uint noFuseBits()`: a value that is 0 at run time but comes
// from a uniform (the explicit kernel: gP.vehicles >> 31).
#ifndef ENGINE_PHYSICS_STRICT_SLANG
#define ENGINE_PHYSICS_STRICT_SLANG

uint noFuseBits();
float nf(float x) { return asfloat(asuint(x) ^ noFuseBits()); }
float3 nf3(float3 v) { return float3(nf(v.x), nf(v.y), nf(v.z)); }
float smul(float a, float b) { return nf(a * b); }
float sdot(float3 a, float3 b) { return nf(nf(smul(a.x, b.x) + smul(a.y, b.y)) + smul(a.z, b.z)); }
float3 scross(float3 a, float3 b)
{
    return float3(nf(smul(a.y, b.z) - smul(b.y, a.z)), nf(smul(a.z, b.x) - smul(b.z, a.x)),
                  nf(smul(a.x, b.y) - smul(b.x, a.y)));
}
float3 sscale(float3 v, float k) { return float3(smul(v.x, k), smul(v.y, k), smul(v.z, k)); }
float sdiv(float a, float b)
{
    float r = 1.0 / b;
    float q = nf(a * r);
    return nf(fma(fma(-q, b, a), r, q));
}
float ssqrt(float x)
{
    if (x <= 0.0) return sqrt(x);
    float s = nf(sqrt(x));
    return nf(fma(fma(-s, s, x), 0.5 / s, s));
}
// ssqrt correctly rounded everywhere: its fma step can land one ulp off where the ulp
// changes (sqrt(0.99999994) gave 1.0, the CPU 0.99999994), which a normalisation of a
// near-unit sum meets (contact offsets' and skin edges' vertex normals). Keep s only if x
// lies between the squares of the midpoints to its neighbours: r = x - s² is exact, s·u
// too (u a power of two), and the differences near a midpoint are exact (Sterbenz). Only
// where needed: ssqrt is inlined a hundred times a kernel (compile_cost.py).
float ssqrtCr(float x)
{
    if (x <= 0.0) return sqrt(x);
    float s = ssqrt(x);
    float r = fma(-s, s, x);
    float uh = nf(asfloat(asuint(s) + 1u) - s);
    float ul = nf(s - asfloat(asuint(s) - 1u));
    if (nf(r - smul(s, uh)) > smul(smul(uh, uh), 0.25)) return asfloat(asuint(s) + 1u);
    if (nf(nf(-r) - smul(s, ul)) >= -smul(smul(ul, ul), 0.25)) return asfloat(asuint(s) - 1u);
    return s;
}

float3 sadd3(float3 a, float3 b) { return nf3(a + b); }
float3 ssub3(float3 a, float3 b) { return nf3(a - b); }
float3 sdiv3(float3 v, float k) { return float3(sdiv(v.x, k), sdiv(v.y, k), sdiv(v.z, k)); }

#endif
