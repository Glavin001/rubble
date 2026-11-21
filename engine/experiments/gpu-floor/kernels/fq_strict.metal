// The shared closest-feature queries' arithmetic (slang/physics/strict.slang's names) for
// a hand-written Metal kernel: the strict build is fast math off with fp contract off, so
// each operation is already IEEE single precision in the written order, and a division
// and a square root are correctly rounded. (The Slang original fences every result with
// an XOR against a uniform, for compilers that would fuse or reorder.) Then the queries
// themselves: slang/physics/feature_query.slang, converted as text when the kernel is
// assembled (floor.rs fq_msl), not copied.
#ifndef EMU
#define FQXYZ(v) ((v).xyz)
#endif
inline float nf(float x) { return x; }
inline float3 nf3(float3 v) { return v; }
inline float smul(float a, float b) { return a * b; }
inline float sdot(float3 a, float3 b) { return ((a.x * b.x) + (a.y * b.y)) + (a.z * b.z); }
inline float3 scross(float3 a, float3 b)
{
    return float3((a.y * b.z) - (b.y * a.z), (a.z * b.x) - (b.z * a.x), (a.x * b.y) - (b.x * a.y));
}
inline float3 sscale(float3 v, float k) { return float3(v.x * k, v.y * k, v.z * k); }
inline float sdiv(float a, float b) { return a / b; }
inline float ssqrt(float x) { return sqrt(x); }
inline float3 sadd3(float3 a, float3 b) { return float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline float3 ssub3(float3 a, float3 b) { return float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline float3 sdiv3(float3 v, float k) { return float3(v.x / k, v.y / k, v.z / k); }
