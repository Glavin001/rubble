// The GPU floor of one explicit node-beam car: a purpose-built step for the lab's
// generated sedan (experiments/gpu-floor). One threadgroup a car runs every substep of a
// frame; counts are compile-time constants for this car (the host prepends #defines).
//
// The per-element rules are ports of the CPU reference
// (engine-softbody/src/explicit_beams.rs, explicit_jbeam_ext.rs, explicit_contact.rs),
// written in glam's operation order (dot = (x·x' + y·y') + z·z', cross as glam's), so that
// with fast math off (IEEE single, no contraction: see mathprobe) the results are the
// reference's bits. Host defines: NN (nodes), NR (beam records), NT (triangles), NG
// (break groups + 1), ND (deform groups), RS (record slots a lane), LANES, STEPS,
// NODE_MAX_ADD, NODE_MAX_CEN, plus feature switches.
#include <metal_stdlib>
using namespace metal;

#define H 0.0005f
#define GRAVITY (-9.807f)
#define MIN_REST 0.1f
#define NONE 0xFFFFFFFFu
#define SENT 0x7FC0DEADu // an add entry of a broken record: skipped
#define PI_F 3.14159265358979323846f
#define TAU_F 6.28318530717958647692f

// Kinds (ExplicitKind::code)
#define K_PLAIN 0u
#define K_JPLAIN 6u
#define K_JSUPPORT 7u
#define K_JBOUNDED 8u
#define K_JCOUPLER 9u
#define K_JTORSION 10u
#define K_JTORSION2 11u
#define K_JANISO 12u
#define K_JATRI 18u
#define K_JATRI2 19u
#define TY_DEFORMABLE 0u
#define NODE_IMMOVABLE 2u
#define EXT_DEFORM 2u

// ---- glam-ordered vector arithmetic --------------------------------------------------
struct V3 { float x, y, z; };
inline V3 v3(float x, float y, float z) { V3 r; r.x = x; r.y = y; r.z = z; return r; }
inline V3 v3(float4 a) { return v3(a.x, a.y, a.z); }
inline float4 f4(V3 a, float w) { return float4(a.x, a.y, a.z, w); }
inline V3 operator+(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline V3 operator-(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline V3 operator-(V3 a) { return v3(-a.x, -a.y, -a.z); }
inline V3 operator*(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
inline V3 operator*(float s, V3 a) { return v3(s * a.x, s * a.y, s * a.z); }
inline V3 operator/(V3 a, float s) { return v3(a.x / s, a.y / s, a.z / s); }
inline float dot3(V3 a, V3 b) { return ((a.x * b.x) + (a.y * b.y)) + (a.z * b.z); }
inline V3 cross3(V3 a, V3 b)
{
    return v3(a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y);
}
inline float len3(V3 a) { return sqrt(dot3(a, a)); }
inline bool isZero3(V3 a) { return a.x == 0.0f && a.y == 0.0f && a.z == 0.0f; }

// ---- the reference's exact functions -------------------------------------------------
inline float exactInvSqrt(float x)
{
    int i = 0x5f3759df - (as_type<int>(x) >> 1);
    float y = as_type<float>(i);
    float hx = 0.5f * x;
    y = y * (1.5f - (hx * y) * y);
    y = y * (1.5f - (hx * y) * y);
    y = y * (1.5f - (hx * y) * y);
    return y * (1.5f - (hx * y) * y);
}
// (length, inverse length)
inline float2 exactLength(float q)
{
    float y = exactInvSqrt(q);
    float l = q * y;
    return float2(fma(fma(-l, l, q), 0.5f * y, l), y);
}
constant float EXP_C[8] = {1.0f, 1.0f, 0.5f, 0.16666667f, 0.041666668f, 0.008333334f, 0.0013888889f, 0.0001984127f};
#define LN2_HI 0.6933594f
#define LN2_LO (-2.1219444e-4f)
inline float exactExp(float x)
{
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return 3.4e38f;
    float k = floor(x * 1.44269504088896340736f + 0.5f);
    float r = (x - k * LN2_HI) - k * LN2_LO;
    float p = EXP_C[7];
    for (int i = 6; i >= 0; i--) p = p * r + EXP_C[i];
    return p * as_type<float>(uint(int(k) + 127) << 23);
}
inline float exactLn(float x)
{
    if (!(x >= 1.17549435e-38f)) return -1000.0f;
    uint b = as_type<uint>(x);
    float e = float(int(b >> 23) - 127);
    float m = as_type<float>((b & 0x007FFFFFu) | 0x3F800000u);
    if (m > 1.41421356237309504880f) { m *= 0.5f; e += 1.0f; }
    float s = (m - 1.0f) / (m + 1.0f);
    float s2 = s * s;
    float q = 1.0f / 11.0f;
    q = q * s2 + 1.0f / 9.0f;
    q = q * s2 + 1.0f / 7.0f;
    q = q * s2 + 1.0f / 5.0f;
    q = q * s2 + 1.0f / 3.0f;
    q = q * s2 + 1.0f;
    float lnm = (s + s) * q;
    return e * LN2_HI + (lnm + e * LN2_LO);
}
inline float exactPow(float x, float y)
{
    if (y == 0.0f) return 1.0f;
    if (x == 0.0f && y > 0.0f) return 0.0f;
    return exactExp(y * exactLn(x));
}
constant uint ATAN_C[12] = {0x3f800000u, 0xbeaaaaabu, 0x3e4ccccdu, 0xbe124925u, 0x3de38e39u, 0xbdba2e8cu,
                            0x3d9d89d9u, 0xbd888889u, 0x3d70f0f1u, 0xbd579436u, 0x3d430c31u, 0xbd321643u};
inline float exactAtan2(float y, float x)
{
    float ax = fabs(x), ay = fabs(y);
    bool swp = ay > ax;
    float lo = swp ? ax : ay, hi = swp ? ay : ax;
    if (hi == 0.0f) return 0.0f;
    float a = lo / hi;
    float t = a, base = 0.0f;
    if (a > as_type<float>(0x3ed413cdu)) { t = (a - 1.0f) / (a + 1.0f); base = 0.78539816339744830962f; }
    float t2 = t * t;
    float p = as_type<float>(ATAN_C[11]);
    for (int k = 10; k >= 0; k--) p = p * t2 + as_type<float>(ATAN_C[k]);
    float r = base + t * p;
    if (swp) r = 1.57079632679489661923f - r;
    if (x < 0.0f) r = PI_F - r;
    if (y < 0.0f) r = -r;
    return r;
}
inline float wrapAngle(float a)
{
    if (a > PI_F) return a - TAU_F;
    if (a <= -PI_F) return a + TAU_F;
    return a;
}
// Rust f32::min / max on non-NaN inputs.
inline float rmin(float a, float b) { return fmin(a, b); }
inline float rmax(float a, float b) { return fmax(a, b); }
inline float rclamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// ---- contact laws --------------------------------------------------------------------
struct Surface { float va, ms, mc, t2, vs, alpha, strength; };
constant Surface GROUND = {3.0f, 0.85f, 0.6f, 0.006f, 3.0f, 2.0f, 1.0f};
constant Surface BOXS = {3.0f, 1.2f, 0.75f, 0.01f, 6.0f, 2.0f, 1.0f};

// contact_friction, exact exponentials.
inline V3 contactFriction(float r, V3 f, V3 v, V3 n, Surface s, float friction)
{
    float vn = dot3(v, n);
    float fn = dot3(f, n);
    V3 sf = f - fn * n;
    V3 sl = v - vn * n;
    float sigma = len3(sl);
    if (sigma > 0.0f) sl = sl / sigma;
    float gr = (r * s.strength) * friction;
    float ms = s.ms * gr;
    if (sigma < s.va && gr > 0.0f && dot3(sf, sf) <= ms * ms)
    {
        float e = exactExp(-sigma / s.va);
        return (r * n - (ms * (1.0f - e)) * sl) - sf;
    }
    float e = exactExp(-exactPow(sigma / s.vs, s.alpha));
    float g = s.mc + (s.ms - s.mc) * e;
    return r * n - ((g + rmin(s.t2 * sigma, 5.0f)) * gr) * sl;
}

inline V3 noPositiveWork(V3 c, float r, V3 n, V3 f, V3 v, float m)
{
    if (m <= 0.0f) return c;
    float hm = H / m;
    V3 rn = n * r;
    V3 t = c - rn;
    V3 u = (v + v) + f * hm;
    float a0 = dot3(rn, u) + dot3(rn, rn) * hm;
    float a1 = dot3(t, u) + (2.0f * hm) * dot3(rn, t);
    float b = dot3(t, t) * hm;
    float s = 1.0f;
    bool whole = true;
    if ((a0 + a1) + b > 0.0f && b > 0.0f)
    {
        whole = false;
        float disc = a1 * a1 - (4.0f * b) * a0;
        s = disc < 0.0f ? 0.0f : rclamp((sqrt(disc) - a1) / (2.0f * b), 0.0f, 1.0f);
    }
    V3 vt = v - n * dot3(v, n);
    float g = dot3(vt, vt) + hm * dot3(f, vt);
    float d = hm * dot3(t, vt);
    if (d < 0.0f && g + d * s < 0.0f)
    {
        s = rclamp(g / -d, 0.0f, s);
        whole = false;
    }
    return whole ? c : rn + t * s;
}

// contact_response on a dissipative, exact surface: (force, budget in .w).
inline V3 contactResponse(V3 f, V3 v, float m, V3 n, Surface s, float p, float friction, thread float& budget)
{
    budget = 0.0f;
    if (p < 0.0f) return v3(0, 0, 0);
    float vn = dot3(v, n);
    float fn = dot3(f, n);
    float r = -fn - ((vn + rmax(vn, 0.0f)) * m) / H;
    if (vn < 0.0f) budget = rmax((0.5f * m) * (vn * vn) + (0.5f * H) * (fn * vn), 0.0f);
    if (r <= 0.0f) { budget = 0.0f; return v3(0, 0, 0); }
    V3 c = contactFriction(r, f, v, n, s, friction);
    return noPositiveWork(c, r, n, f, v, m);
}

inline float shiftDepth(float p, float budget, float fn)
{
    budget = budget * 1.0f;
    float full = 0.2f * p;
    if (fn >= 0.0f || -fn * full <= budget) return full;
    return budget / -fn;
}

// ---- beam elements -------------------------------------------------------------------
struct BeamOut { V3 d; float inv; float f; float slope; bool broke; bool check; };

// The JBeam yield rule (keep: no strength drop).
inline float jbeamYield(thread float4& s0, float strength, float shortB, float longB, float plastic, float limitK,
                        uint ty, float len, float e, float k, float damp, float f, thread bool& check)
{
    // s0 = (rest, tc, tt, gate)
    float tau = fabs(f);
    bool tear = false;
    if (tau > s0.w && ty == TY_DEFORMABLE && k != 0.0f)
    {
        if (f > s0.y && e < 0.0f && e + (s0.y / k) * (1.0f - plastic) < 0.0f)
        {
            float delta = e + (s0.y / k) * (1.0f - plastic);
            float old = s0.x;
            float flo = rmax(shortB, MIN_REST);
            float fr = s0.x + delta;
            if (fr < flo)
            {
                s0.x = rmin(flo, old);
                f = -k * (len - s0.x) - damp;
            }
            else
            {
                s0.x = fr;
                f -= (f - s0.y) / 2.0f;
            }
            tau = fabs(f);
            if (s0.x < old)
            {
                s0.y = rmin((s0.y * old) / s0.x, rmax(limitK, s0.y));
                s0.w = rmin(rmin(s0.y, s0.z), strength);
            }
        }
        else if (f < -s0.z && e > 0.0f && e - (s0.z / k) * (1.0f - plastic) > 0.0f)
        {
            float delta = e - (s0.z / k) * (1.0f - plastic);
            float old = s0.x;
            float ceil = fabs(longB);
            float fr = s0.x + delta;
            if (fr > ceil)
            {
                s0.x = rmax(ceil, old);
                f = -k * (len - s0.x) - damp;
                tear = longB < 0.0f;
            }
            else
            {
                s0.x = fr;
                f -= (f + s0.z) / 2.0f;
            }
            tau = fabs(f);
            if (s0.x > old)
            {
                s0.z = rmin((s0.z * s0.x) / old, rmax(limitK, s0.z));
                s0.w = rmin(rmin(s0.y, s0.z), strength);
            }
        }
    }
    check = tau > strength || tear;
    return f;
}

// jbeam_force_filtered (no filter) and jbeam_ext_force_with (no ramp, no rebound) for
// JbeamPlain / JbeamSupport / JbeamBounded / JbeamAniso.
inline BeamOut jbeamForce(uint kind, uint ty, float4 kc, float4 bb, float4 e1, float4 e2, bool ext,
                          thread float4& s0, thread float4& s1, V3 pa, V3 pb, V3 va, V3 vb)
{
    BeamOut o;
    o.d = pa - pb;
    float q = dot3(o.d, o.d);
    float2 li = exactLength(q);
    float len = li.x;
    o.inv = li.y;
    float e = len - s0.x;
    float u = dot3(va - vb, o.d) * o.inv;
    o.broke = false;
    o.check = false;
    float k = kc.x, c = kc.y;
    float damp = c * u;
    float slope = c;
    float shortB = bb.x, longB = bb.y, plastic = bb.z;
    if (kind == K_JBOUNDED)
    {
        float split = s0.y;
        float au = fabs(u);
        if (au > split)
        {
            damp = copysign(c * split + s0.z * (au - split), u);
            slope = s0.z;
        }
        float lng = longB * s0.x;
        float sht = shortB * s0.x;
        float ov = e > lng ? e - lng : (e < -sht ? -e - sht : 0.0f);
        if (ov > 0.0f)
        {
            float zone = plastic;
            float s = zone > 0.0f ? rmin(ov / zone, 1.0f) : 1.0f;
            k += (kc.z - k) * s;
            damp += (kc.w * u - damp) * s;
            slope += (kc.w - slope) * s;
        }
    }
    else if (kind == K_JSUPPORT)
    {
        if (e > 0.0f)
        {
            if (len > longB)
            {
                o.broke = true;
                o.f = 0.0f;
                o.slope = 0.0f;
                s1.y = 0.0f;
                return o;
            }
            k = 0.0f;
            damp = 0.0f;
            slope = 0.0f;
        }
    }
    else if (kind == K_JANISO)
    {
        if (len > e1.w)
        {
            o.broke = true;
            o.f = 0.0f;
            o.slope = 0.0f;
            s1.y = 0.0f;
            return o;
        }
        float sw = s0.x + (e2.y + e2.z);
        float past = len - sw;
        if (past > 0.0f)
        {
            float zone = e1.z * sw;
            float s = zone > 0.0f ? rmin(past / zone, 1.0f) : 1.0f;
            k += (e1.x - k) * s;
            slope = c + (e1.y - c) * s;
            damp = slope * u;
        }
    }
    float f = -k * e - damp;
    s1.y = f;
    bool check;
    f = jbeamYield(s0, s1.x, shortB, longB, plastic, kc.z, ty, len, e, k, damp, f, check);
    o.f = f;
    o.check = check;
    o.slope = slope;
    return o;
}

// beam_force for a non-JBeam Plain beam (exact lengths).
inline BeamOut plainForce(float4 kc, float4 bb, thread float4& s0, thread float4& s1, V3 pa, V3 pb, V3 va, V3 vb)
{
    BeamOut o;
    o.d = pa - pb;
    float2 li = exactLength(dot3(o.d, o.d));
    o.inv = li.y;
    float e = li.x - s0.x;
    float u = dot3(va - vb, o.d) * o.inv;
    float k = kc.x, c = kc.y;
    bool torn = false;
    o.broke = false;
    if (k == 0.0f && bb.y > 0.0f && e > s0.x * bb.y) { torn = true; o.broke = true; }
    float f = torn ? 0.0f : -k * e - c * u;
    s1.y = f;
    // (Virtual / actuator: no yield; the host checks the strength is infinite.)
    o.f = f;
    o.check = false;
    o.slope = 0.0f; // not a JBeam kind: no centred slope
    return o;
}

// aero_tri_force
inline V3 aeroTri(V3 x0, V3 x1, V3 x2, V3 v0, V3 v1, V3 v2, float kd, float kl, float stall)
{
    V3 vm = ((v0 + v1) + v2) * (1.0f / 3.0f);
    float s2 = dot3(vm, vm);
    V3 cr = cross3(x1 - x0, x2 - x0);
    float cl = sqrt(dot3(cr, cr));
    if (s2 <= 1e-12f || cl <= 0.0f) return v3(0, 0, 0);
    float s = sqrt(s2);
    V3 u = vm * (-1.0f / s);
    V3 n = cr * (1.0f / cl);
    float c = dot3(n, u);
    float sa = fabs(c);
    float ca = sqrt(rmax(1.0f - c * c, 0.0f));
    float q = (0.5f * cl) * s2;
    V3 f = u * ((kd * q) * sa);
    V3 l = n - u * c;
    float ll = sqrt(dot3(l, l));
    if (ll > 0.0f)
    {
        float cla = sa <= stall ? sa * ca : stall * ca;
        float m = (kl * q) * cla;
        f = f + l * ((c < 0.0f ? -m : m) / ll);
    }
    return f * (1.0f / 3.0f);
}

struct Jac { V3 g[4]; };
inline Jac dihedralJacobian(V3 x0, V3 x1, V3 x2, V3 x3)
{
    Jac J;
    V3 b1 = x1 - x0, b2 = x2 - x1, b3 = x3 - x2;
    V3 n1 = cross3(b1, b2), n2 = cross3(b2, b3);
    float l1 = dot3(n1, n1), l2 = dot3(n2, n2);
    float ll = dot3(b2, b2);
    if (l1 < 1e-12f || l2 < 1e-12f || ll < 1e-12f)
    {
        for (int q = 0; q < 4; q++) J.g[q] = v3(0, 0, 0);
        return J;
    }
    float lb = sqrt(ll);
    V3 g1 = n1 * (-lb / l1);
    V3 g4 = n2 * (lb / l2);
    float p = -dot3(b1, b2) / ll;
    float q = -dot3(b3, b2) / ll;
    J.g[0] = g1;
    J.g[1] = g1 * (p - 1.0f) - g4 * q;
    J.g[2] = g4 * (q - 1.0f) - g1 * p;
    J.g[3] = g4;
    return J;
}
inline float dihedral(V3 x0, V3 x1, V3 x2, V3 x3)
{
    V3 b1 = x1 - x0, b2 = x2 - x1, b3 = x3 - x2;
    V3 n1 = cross3(b1, b2), n2 = cross3(b2, b3);
    return exactAtan2(len3(b2) * dot3(b1, n2), dot3(n1, n2));
}


// ---- self-collision (explicit_beams.rs collision_pass, explicit_contact.rs clip_*) ----
#define CLIP_DEPTH 0.1f
#define CLIP_EMPTY 0x7FFFFFFFu
#define CLIP_SELF 1023u
// A triangle's frame (TriFrame / ContactFrame): p0, n, and the rows of the inverse.
struct Frame { V3 p0; V3 n; V3 ra, rb, rd; bool ok; };
inline Frame triFrame(V3 p0, V3 p1, V3 p2)
{
    Frame F;
    V3 u = p1 - p0;
    V3 w = p2 - p0;
    V3 c = cross3(u, w);
    float len = len3(c);
    F.p0 = p0;
    F.n = len > 0.0f ? c / len : c;
    float det = dot3(u, cross3(w, F.n));
    F.ok = fabs(det) > 1e-6f;
    if (F.ok)
    {
        float r = 1.0f / det;
        F.ra = cross3(w, F.n) * r;
        F.rb = cross3(F.n, u) * r;
        F.rd = cross3(u, w) * r;
    }
    else { F.ra = v3(0, 0, 0); F.rb = F.ra; F.rd = F.ra; }
    return F;
}
// (α, β, γ, δ)
inline float4 frameLocal(V3 p0, V3 ra, V3 rb, V3 rd, bool ok, V3 p)
{
    V3 q = p - p0;
    float alpha = 0.0f, beta = 0.0f, delta = 0.0f;
    if (ok) { alpha = dot3(ra, q); beta = dot3(rb, q); delta = dot3(rd, q); }
    return float4(alpha, beta, (1.0f - alpha) - beta, delta);
}
inline int toFixed(float x) { return int(rint(clamp(x * 32.0f, -2.0e9f, 2.0e9f))); }
inline int toFixedShift(float x) { return int(rint(clamp(x * 16777216.0f, -2.0e9f, 2.0e9f))); }
// Floats as ordered ints (atomic min / max of signed values).
inline int ordF(float f) { int i = as_type<int>(f); return i >= 0 ? i : i ^ 0x7FFFFFFF; }
inline float unordF(int i) { return as_type<float>(i >= 0 ? i : i ^ 0x7FFFFFFF); }
inline float tbInvMass(float im1, float im2, float im0, float a, float b, float g)
{
    float t0 = a > 0.0f ? (a * a) * im1 : 0.0f;
    float t1 = b > 0.0f ? (b * b) * im2 : 0.0f;
    float t2 = g > 0.0f ? (g * g) * im0 : 0.0f;
    return (t0 + t1) + t2;
}

// ---- buffers -------------------------------------------------------------------------
struct Params
{
    uint cars;
    uint frames;
    float ground;
    uint skip; // timing only: phases left out (bit k: phase k)
    // RUNTIME_SIZES (a ledger measurement): the counts as uniforms, as a general kernel has them.
    uint rtLanes, rtNN, rtNT, rtNC, rtSteps, rtNG, rtRS, pad2;
};

#define TG_FLAGS 8
#define SKIP(k) ((P.skip & (1u << (k))) != 0u)
#if RUNTIME_SIZES
#define Z_LANES P.rtLanes
#define Z_NN P.rtNN
#define Z_NT P.rtNT
#define Z_NC P.rtNC
#define Z_STEPS P.rtSteps
#define Z_NG P.rtNG
#define Z_RS P.rtRS
#else
#define Z_LANES LANES
#define Z_NN NN
#define Z_NT NT
#define Z_NC NC
#define Z_STEPS STEPS
#define Z_NG NG
#define Z_RS RS
#endif
#ifndef FAST_SUMS
#define FAST_SUMS 0
#endif
#if FAST_SUMS
// Unordered sums: device atomic float adds into per-node accumulators.
#define PUT_ADD(e, node, val) { V3 _v = (val); atomic_fetch_add_explicit(&fAcc[(node) * 3], _v.x, memory_order_relaxed); atomic_fetch_add_explicit(&fAcc[(node) * 3 + 1], _v.y, memory_order_relaxed); atomic_fetch_add_explicit(&fAcc[(node) * 3 + 2], _v.z, memory_order_relaxed); }
#define PUT_CEN(e, node, val) { V3 _v = (val); atomic_fetch_add_explicit(&cAcc[(node) * 3], _v.x, memory_order_relaxed); atomic_fetch_add_explicit(&cAcc[(node) * 3 + 1], _v.y, memory_order_relaxed); atomic_fetch_add_explicit(&cAcc[(node) * 3 + 2], _v.z, memory_order_relaxed); }
#define PUT_SENT(e)
#else
// Ordered sums: each add in its own entry, summed per node in the CPU's order.
#define PUT_ADD(e, node, val) addEnt[e] = f4((val), 0)
#define PUT_CEN(e, node, val) cenEnt[e] = f4((val), 0)
#define PUT_SENT(e) addEnt[e] = as_type<float4>(uint4(SENT))
#endif

kernel void floorFrame(
    constant Params& P [[buffer(0)]],
    device float4* nodeState [[buffer(1)]],          // NN*3 a car: pos, vel, force
    device const float4* nodeStatic [[buffer(2)]],   // NN: mass, friction, flags bits, 0
    device const uint4* nodeRange [[buffer(3)]],     // NN: add start, add count, cen start, cen count
    device const uint4* recA [[buffer(4)]],          // NR: a, b, kind | ty << 8 | ext << 16, group (int)
    device const float4* recK [[buffer(5)]],         // NR: k, c, limit_k, limit_c
    device const float4* recB [[buffer(6)]],         // NR: short, long, plastic, aux bits (n2, c...)
    device const float4* recE [[buffer(7)]],         // NR*2: e1; (e2.x, e2.y, ramp offset, trigger)
    device float4* recState [[buffer(8)]],           // NR*3 a car: s0, s1 (strength, force, slope, broken), aux
    device const uint4* recEnt [[buffer(9)]],        // NR: add entries (a, b, c/n2, d)
    device const uint4* recCen [[buffer(10)]],       // NR: centred entries (j end 0, j end 1, j+1 end 0, j+1 end 1)
    device float4* addEnt [[buffer(11)]],            // NADD a car
    device float4* cenEnt [[buffer(12)]],            // NCEN a car
    device const uint2* addInfo [[buffer(13)]],      // NADD: record, group (int)
    device const uint* perm [[buffer(14)]],          // RS*LANES: the records a lane owns (NONE past the end)
    device uint* groupsLast [[buffer(15)]],          // NG a car: fired in the last beam pass
    device uint* deform [[buffer(16)]],              // ND a car
    device const uint4* triBreaks [[buffer(17)]],    // NT
    device uint* triDead [[buffer(18)]],             // NT a car
    device atomic_int* acc [[buffer(19)]],           // NN*3 a car: contact sums (fixed point)
    device int* shift [[buffer(20)]],                // NN*3 a car: contact position steps
    device uint* misc [[buffer(21)]],                // 16 a car: [0] shifts pending, [1] status, [2] pairs, [3] rebuild, [4] hits last step, [5] rebuilds, [6] most hits
    device const uint* cands [[buffer(22)]],         // NC: the contacters (node indices)
    device const uint4* tris [[buffer(23)]],         // NT: i0, i1, i2, 0
    device const uint* allowed [[buffer(24)]],       // NT*NCW: pair (t, c) not excluded (vertex, self group, exclusions)
    device float4* triGeo [[buffer(25)]],            // NT*GEO a car
    device uint* pairs [[buffer(26)]],               // PAIR_CAP a car: t << 16 | node
    device float4* pBuild [[buffer(27)]],            // NN a car: positions at the list's build
    device atomic_uint* clip [[buffer(28)]],         // NN*4 a car: anti-clip side memory
    device float4* hitBuf [[buffer(29)]],            // HIT_CAP*HITW a car
    device uint* scratch [[buffer(30)]],             // NN*12 a car: stage (the first pass's sums, the residual's),
                                                     // then (FAST_SUMS) the beam pass's and the centred sums
    uint lid [[thread_position_in_threadgroup]],
    uint car [[threadgroup_position_in_grid]])
{
    threadgroup float tgPos[NN * 3];
    threadgroup float tgVel[NN * 3];
    threadgroup float tgDv[NN * 3];
    threadgroup atomic_uint tgMaster[NG];
    threadgroup uint tgFlags[TG_FLAGS];
#if COLLIDE
    threadgroup float tgF[NN * 3];
    threadgroup atomic_uint tgSplit[NN];
    threadgroup atomic_uint tgCount[4]; // [0] speed max (bits), [1] pairs, [2] hits
    threadgroup atomic_int tgDisp[6];   // per axis min, max of the displacement since the list's build
#endif

    nodeState += car * NN * 3;
    recState += car * NR * 3;
    addEnt += car * NADD;
    cenEnt += car * NCEN;
    groupsLast += car * NG;
    deform += car * ND;
    triDead += car * NT;
    acc += car * NN * 3;
    shift += car * NN * 3;
    misc += car * 16;
    scratch += car * NN * 12;
    device atomic_int* stage = (device atomic_int*)scratch;
    device atomic_float* fAcc = (device atomic_float*)(scratch + NN * 6);
    device atomic_float* cAcc = (device atomic_float*)(scratch + NN * 9);
#if COLLIDE
    triGeo += car * NT * GEO;
    pairs += car * PAIR_CAP;
    pBuild += car * NN;
    clip += car * NN * 4;
    hitBuf += car * HIT_CAP * HITW;
#endif

    const bool isNode = lid < Z_NN;
    const uint ni = isNode ? lid : 0u;
    // Node registers.
    V3 P0 = v3(nodeState[ni * 3]), V0 = v3(nodeState[ni * 3 + 1]), F0 = v3(nodeState[ni * 3 + 2]);
    float4 ns = nodeStatic[ni];
    const float mass = ns.x, friction = ns.y;
    const uint nflags = as_type<uint>(ns.z);
    uint4 nr = nodeRange[ni];

    // Record registers.
    uint rec[RS];
    float4 s0[RS], s1[RS], ax[RS];
    for (uint s = 0; s < Z_RS; s++)
    {
        rec[s] = perm[s * Z_LANES + lid];
        uint r = rec[s] == NONE ? 0u : rec[s];
        s0[s] = recState[r * 3];
        s1[s] = recState[r * 3 + 1];
        ax[s] = recState[r * 3 + 2];
    }
    for (uint g = lid; g < Z_NG; g += Z_LANES) atomic_store_explicit(&tgMaster[g], NONE, memory_order_relaxed);
    if (lid == 0)
    {
        tgFlags[0] = misc[0]; // shifts pending from the last collision pass
        tgFlags[1] = 0u;
        tgFlags[3] = misc[4]; // hits last step (stage sums to clear)
        tgFlags[4] = misc[5]; // list builds
        tgFlags[5] = misc[6]; // most hits in a step
        tgFlags[6] = misc[1]; // status
    }
#if COLLIDE
    for (uint q = lid; q < 4; q += Z_LANES) atomic_store_explicit(&tgCount[q], 0u, memory_order_relaxed);
    if (lid == 0) atomic_store_explicit(&tgCount[1], misc[2], memory_order_relaxed);
    if (lid == 0) tgFlags[2] = misc[3]; // the list must be built
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint frame = 0; frame < P.frames; frame++)
    for (uint step = 0; step < Z_STEPS; step++)
    {
        // ---- S0: the node pass ----
        bool shiftsPending = tgFlags[0] != 0u;
        if ((frame != 0u || step != 0u) && !SKIP(11))
        {
            // The groups the last beam pass set off (torsion watches read them).
            for (uint g = lid; g < Z_NG; g += Z_LANES)
            {
                groupsLast[g] = atomic_load_explicit(&tgMaster[g], memory_order_relaxed) != NONE ? 1u : 0u;
                atomic_store_explicit(&tgMaster[g], NONE, memory_order_relaxed);
            }
        }
        if (isNode && !SKIP(0))
        {
            // Fixed-point contact sums (taken and cleared) become part of the force.
            int ax_ = atomic_exchange_explicit(&acc[ni * 3], 0, memory_order_relaxed);
            int ay_ = atomic_exchange_explicit(&acc[ni * 3 + 1], 0, memory_order_relaxed);
            int az_ = atomic_exchange_explicit(&acc[ni * 3 + 2], 0, memory_order_relaxed);
            F0 = F0 + v3(float(ax_), float(ay_), float(az_)) * (1.0f / 32.0f);
            V3 dshift = v3(0, 0, 0);
            if (shiftsPending)
            {
                dshift = v3(float(shift[ni * 3]), float(shift[ni * 3 + 1]), float(shift[ni * 3 + 2])) * (1.0f / 16777216.0f);
                shift[ni * 3] = 0;
                shift[ni * 3 + 1] = 0;
                shift[ni * 3 + 2] = 0;
            }
#if COLLIDE
            // Anti-clip's memory: last step's contacts are held now.
            uint m2 = atomic_load_explicit(&clip[ni * 4 + 2], memory_order_relaxed);
            uint m3 = atomic_load_explicit(&clip[ni * 4 + 3], memory_order_relaxed);
            atomic_store_explicit(&clip[ni * 4], m2, memory_order_relaxed);
            atomic_store_explicit(&clip[ni * 4 + 1], m3, memory_order_relaxed);
            atomic_store_explicit(&clip[ni * 4 + 2], CLIP_EMPTY, memory_order_relaxed);
            atomic_store_explicit(&clip[ni * 4 + 3], CLIP_EMPTY, memory_order_relaxed);
            if (tgFlags[3] != 0u)
                for (uint q = 0; q < 6; q++) atomic_store_explicit(&stage[ni * 6 + q], 0, memory_order_relaxed);
            atomic_store_explicit(&tgSplit[ni], 0u, memory_order_relaxed);
#endif
            float lift = 0.0f;
            V3 x = v3(0, 0, 0) + P0;
            if (P.ground > x.y)
            {
                float budget;
                float depth = P.ground - x.y;
                V3 c = contactResponse(F0, V0, mass, v3(0, 1, 0), GROUND, depth, friction, budget);
                if (!isZero3(c)) lift = shiftDepth(depth, budget, F0.y);
                F0 = F0 + c;
            }
#if NBOX > 0
            // Static boxes (box_face, contact_force: the box surface, no depth term).
            for (uint b = 0; b < NBOX; b++)
            {
                float3 lo = BOX_LO[b], hi = BOX_HI[b];
                if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
                {
                    V3 normal = v3(0.0f, 0.0f, -1.0f);
                    float dist = x.z - lo.z;
                    float dd[5] = {hi.z - x.z, x.x - lo.x, hi.x - x.x, x.y - lo.y, hi.y - x.y};
                    V3 nn[5] = {v3(0.0f, 0.0f, 1.0f), v3(-1.0f, 0.0f, 0.0f), v3(1.0f, 0.0f, 0.0f), v3(0.0f, -1.0f, 0.0f), v3(0.0f, 1.0f, 0.0f)};
                    for (uint q = 0; q < 5; q++)
                        if (dd[q] < dist) { dist = dd[q]; normal = nn[q]; }
                    float budget;
                    V3 c = contactResponse(F0, V0, mass, normal, BOXS, 0.0f, friction, budget);
                    F0 = F0 + c;
                }
            }
#endif
            V3 vBefore = V0;
            if ((nflags & NODE_IMMOVABLE) == 0u)
            {
                V0 = V0 + (F0 / mass) * H;
                P0 = P0 + V0 * H;
                if (shiftsPending) P0 = P0 + dshift;
                if (lift != 0.0f) P0.y = P0.y + lift;
            }
            F0 = v3(0.0f, mass * GRAVITY, 0.0f);
            V3 dv = V0 - vBefore;
            tgPos[ni * 3] = P0.x; tgPos[ni * 3 + 1] = P0.y; tgPos[ni * 3 + 2] = P0.z;
            tgVel[ni * 3] = V0.x; tgVel[ni * 3 + 1] = V0.y; tgVel[ni * 3 + 2] = V0.z;
            tgDv[ni * 3] = dv.x; tgDv[ni * 3 + 1] = dv.y; tgDv[ni * 3 + 2] = dv.z;
        }
        // Triangle breaks (spec §18): from the broken flags as the beam pass starts.
        for (uint t = lid; t < Z_NT && !SKIP(9); t += Z_LANES)
        {
            if (triDead[t] == 0u)
            {
                uint4 tb = triBreaks[t];
                bool dead = false;
                for (uint q = 0; q < 4; q++)
                {
                    uint j = tb[q];
                    if (j != NONE && recState[j * 3 + 1].w != 0.0f) dead = true;
                }
                if (dead) triDead[t] = 1u;
            }
        }
#if COLLIDE
        if (lid < 6) atomic_store_explicit(&tgDisp[lid], (lid & 1u) == 0u ? 0x7FFFFFFF : int(0x80000000), memory_order_relaxed);
        if (lid == 0) { atomic_store_explicit(&tgCount[0], 0u, memory_order_relaxed); atomic_store_explicit(&tgCount[2], 0u, memory_order_relaxed); }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (lid == 0) { tgFlags[0] = 0u; tgFlags[1] = 0u; tgFlags[3] = 0u; }

        // ---- S1: the centred damping's slots (last beam pass's slopes) ----
        for (uint s = 0; s < Z_RS && !SKIP(1); s++)
        {
            uint r = rec[s];
            if (r == NONE) continue;
            uint4 a = recA[r];
            uint kind = a.z & 0xFFu;
            if (kind == K_JATRI) continue; // aero: no slope, its entries stay zero
            uint4 ce = recCen[r];
            float slope = s1[s].z;
            if (kind == K_JTORSION)
            {
                uint4 a2 = recA[r + 1];
                V3 x[4], v[4];
                uint ids[4] = {a.x, a.y, a2.x, a2.y};
                for (uint q = 0; q < 4; q++)
                {
                    x[q] = v3(tgPos[ids[q] * 3], tgPos[ids[q] * 3 + 1], tgPos[ids[q] * 3 + 2]);
                    v[q] = v3(tgVel[ids[q] * 3], tgVel[ids[q] * 3 + 1], tgVel[ids[q] * 3 + 2]);
                }
                V3 o[4] = {v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0)};
                if (slope != 0.0f)
                {
                    Jac J = dihedralJacobian(x[0], x[1], x[2], x[3]);
                    float dr = (((dot3(J.g[0], v[0]) - ax[s].x) + (dot3(J.g[1], v[1]) - ax[s].y)) +
                                (dot3(J.g[2], v[2]) - ax[s].z)) + (dot3(J.g[3], v[3]) - ax[s].w);
                    float t = (-0.5f * slope) * dr;
                    for (uint q = 0; q < 4; q++) o[q] = J.g[q] * t;
                }
                PUT_CEN(ce.x, ids[0], o[0]); PUT_CEN(ce.y, ids[1], o[1]);
                PUT_CEN(ce.z, ids[2], o[2]); PUT_CEN(ce.w, ids[3], o[3]);
                continue;
            }
            V3 push = v3(0, 0, 0);
            if (slope != 0.0f)
            {
                if (kind == K_JCOUPLER)
                {
                    V3 da = v3(tgDv[a.x * 3], tgDv[a.x * 3 + 1], tgDv[a.x * 3 + 2]);
                    V3 db = v3(tgDv[a.y * 3], tgDv[a.y * 3 + 1], tgDv[a.y * 3 + 2]);
                    push = (da - db) * (-0.5f * slope);
                }
                else
                {
                    V3 pa = v3(tgPos[a.x * 3], tgPos[a.x * 3 + 1], tgPos[a.x * 3 + 2]);
                    V3 pb = v3(tgPos[a.y * 3], tgPos[a.y * 3 + 1], tgPos[a.y * 3 + 2]);
                    V3 va = v3(tgVel[a.x * 3], tgVel[a.x * 3 + 1], tgVel[a.x * 3 + 2]);
                    V3 vb = v3(tgVel[a.y * 3], tgVel[a.y * 3 + 1], tgVel[a.y * 3 + 2]);
                    V3 d = pa - pb;
                    float inv = exactLength(dot3(d, d)).y;
                    float du = dot3(va - vb, d) * inv - ax[s].x;
                    push = d * (((-0.5f * slope) * du) * inv);
                }
            }
            PUT_CEN(ce.x, a.x, push);
            PUT_CEN(ce.y, a.y, -push);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // ---- S2: the centred correction per node ----
        if (isNode && !SKIP(2))
        {
            V3 c = v3(0, 0, 0);
            uint n = min(nr.w, (uint)NODE_MAX_CEN);
#if FAST_SUMS
            (void)n;
            c = v3(atomic_exchange_explicit(&cAcc[ni * 3], 0.0f, memory_order_relaxed), atomic_exchange_explicit(&cAcc[ni * 3 + 1], 0.0f, memory_order_relaxed), atomic_exchange_explicit(&cAcc[ni * 3 + 2], 0.0f, memory_order_relaxed));
#else
            for (uint k = 0; k < n; k++) c = c + v3(cenEnt[nr.z + k]);
#endif
            if ((nflags & NODE_IMMOVABLE) == 0u)
            {
                V3 dv = (c / mass) * H;
                V0 = V0 + dv;
                P0 = P0 + dv * H;
            }
            tgPos[ni * 3] = P0.x; tgPos[ni * 3 + 1] = P0.y; tgPos[ni * 3 + 2] = P0.z;
            tgVel[ni * 3] = V0.x; tgVel[ni * 3 + 1] = V0.y; tgVel[ni * 3 + 2] = V0.z;
#if COLLIDE
            atomic_fetch_max_explicit(&tgCount[0], as_type<uint>(len3(V0)), memory_order_relaxed);
            V3 dp = P0 - v3(pBuild[ni]);
            atomic_fetch_min_explicit(&tgDisp[0], ordF(dp.x), memory_order_relaxed);
            atomic_fetch_max_explicit(&tgDisp[1], ordF(dp.x), memory_order_relaxed);
            atomic_fetch_min_explicit(&tgDisp[2], ordF(dp.y), memory_order_relaxed);
            atomic_fetch_max_explicit(&tgDisp[3], ordF(dp.y), memory_order_relaxed);
            atomic_fetch_min_explicit(&tgDisp[4], ordF(dp.z), memory_order_relaxed);
            atomic_fetch_max_explicit(&tgDisp[5], ordF(dp.z), memory_order_relaxed);
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // ---- S3: the beam pass ----
        bool deformReq[RS];
#if FAST_SUMS
        V3 lastPush[RS];
        for (uint s = 0; s < Z_RS; s++) lastPush[s] = v3(0, 0, 0);
#endif
        for (uint s = 0; s < Z_RS; s++) deformReq[s] = false;
        for (uint s = 0; s < Z_RS && !SKIP(3); s++)
        {
            uint r = rec[s];
            if (r == NONE) continue;
            uint4 a = recA[r];
            uint4 ent = recEnt[r];
            uint kind = a.z & 0xFFu;
            uint ty = (a.z >> 8) & 0xFFu;
            bool ext = ((a.z >> 16) & 1u) != 0u;
            int group = as_type<int>(a.w);
            if (s1[s].w != 0.0f)
            {
                // Broken: its entries are skipped from now on.
                PUT_SENT(ent.x);
                PUT_SENT(ent.y);
                if (ent.z != NONE) PUT_SENT(ent.z);
                if (ent.w != NONE) PUT_SENT(ent.w);
                s1[s].z = 0.0f;
                continue;
            }
            V3 pa = v3(tgPos[a.x * 3], tgPos[a.x * 3 + 1], tgPos[a.x * 3 + 2]);
            V3 pb = v3(tgPos[a.y * 3], tgPos[a.y * 3 + 1], tgPos[a.y * 3 + 2]);
            V3 va = v3(tgVel[a.x * 3], tgVel[a.x * 3 + 1], tgVel[a.x * 3 + 2]);
            V3 vb = v3(tgVel[a.y * 3], tgVel[a.y * 3 + 1], tgVel[a.y * 3 + 2]);
            float4 kc = recK[r];
            float4 bb = recB[r];
            if (kind == K_JATRI)
            {
                float4 e1 = recE[r * 2];
                uint n2 = as_type<uint>(e1.x);
                V3 p2 = v3(tgPos[n2 * 3], tgPos[n2 * 3 + 1], tgPos[n2 * 3 + 2]);
                V3 v2 = v3(tgVel[n2 * 3], tgVel[n2 * 3 + 1], tgVel[n2 * 3 + 2]);
                V3 f = aeroTri(pa, pb, p2, va, vb, v2, e1.y, e1.z, e1.w);
                PUT_ADD(ent.x, a.x, f);
                PUT_ADD(ent.y, a.y, f);
                PUT_ADD(ent.z, n2, f);
                s1[s].z = 0.0f;
                continue;
            }
            if (kind == K_JTORSION)
            {
                uint4 a2 = recA[r + 1];
                V3 x[4] = {pa, pb, v3(tgPos[a2.x * 3], tgPos[a2.x * 3 + 1], tgPos[a2.x * 3 + 2]),
                           v3(tgPos[a2.y * 3], tgPos[a2.y * 3 + 1], tgPos[a2.y * 3 + 2])};
                V3 v[4] = {va, vb, v3(tgVel[a2.x * 3], tgVel[a2.x * 3 + 1], tgVel[a2.x * 3 + 2]),
                           v3(tgVel[a2.y * 3], tgVel[a2.y * 3 + 1], tgVel[a2.y * 3 + 2])};
                // The watch: a group set off in the last beam pass breaks it.
                float4 we = recE[r * 2];
                uint3 w = uint3(as_type<uint>(we.x), as_type<uint>(we.y), as_type<uint>(we.z));
                bool watch = (w.x != 0u && groupsLast[w.x] != 0u) || (w.y != 0u && groupsLast[w.y] != 0u) ||
                             (w.z != 0u && groupsLast[w.z] != 0u);
                V3 o[4] = {v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0)};
                bool broke = false;
                Jac J = dihedralJacobian(x[0], x[1], x[2], x[3]);
                float gv[4];
                for (uint q = 0; q < 4; q++) gv[q] = dot3(J.g[q], v[q]);
                if (watch)
                {
                    broke = true;
                    s1[s].y = 0.0f;
                }
                else
                {
                    float phi = dihedral(x[0], x[1], x[2], x[3]);
                    float rate = ((gv[0] + gv[1]) + gv[2]) + gv[3];
                    float d = wrapAngle(phi - s0[s].x);
                    float k = kc.x, c = kc.y;
                    if (k * fabs(d) > s0[s].y)
                    {
                        d = copysign(s0[s].y / k, d);
                        s0[s].x = wrapAngle(phi - d);
                    }
                    float t = -k * d - c * rate;
                    s1[s].y = t;
                    if (fabs(t) > s1[s].x) broke = true;
                    else for (uint q = 0; q < 4; q++) o[q] = J.g[q] * t;
                }
                ax[s] = float4(gv[0], gv[1], gv[2], gv[3]);
                if (broke) s1[s].w = 1.0f;
                s1[s].z = broke ? 0.0f : kc.y;
                PUT_ADD(ent.x, a.x, o[0]); PUT_ADD(ent.y, a.y, o[1]);
                PUT_ADD(ent.z, a2.x, o[2]); PUT_ADD(ent.w, a2.y, o[3]);
                if (broke && group > 0) { atomic_fetch_min_explicit(&tgMaster[group], r, memory_order_relaxed); tgFlags[1] = 1u; }
                continue;
            }
            if (kind == K_JCOUPLER)
            {
                V3 f = (-kc.x) * (pa - pb) - kc.y * (va - vb);
                float m = len3(f);
                s1[s].y = m;
                bool broke = m > s1[s].x;
                V3 push = broke ? v3(0, 0, 0) : f;
                if (broke) s1[s].w = 1.0f;
                s1[s].z = broke ? 0.0f : kc.y;
                PUT_ADD(ent.x, a.x, push);
                PUT_ADD(ent.y, a.y, -push);
#if FAST_SUMS
                lastPush[s] = push;
#endif
                if (broke && group > 0) { atomic_fetch_min_explicit(&tgMaster[group], r, memory_order_relaxed); tgFlags[1] = 1u; }
                continue;
            }
            BeamOut o;
            if (kind == K_PLAIN)
            {
                o = plainForce(kc, bb, s0[s], s1[s], pa, pb, va, vb);
            }
            else
            {
                float4 e1 = ext ? recE[r * 2] : float4(0);
                float4 e2 = ext ? recE[r * 2 + 1] : float4(0);
                o = jbeamForce(kind, ty, kc, bb, e1, e2, ext, s0[s], s1[s], pa, pb, va, vb);
            }
            // The extension rate for the next centred correction.
            ax[s].x = dot3(va - vb, o.d) * o.inv;
            float f = o.f;
            bool brokeNow = o.broke;
            if (o.check)
            {
                brokeNow = true;
                f = 0.0f;
            }
            if (brokeNow) s1[s].w = 1.0f;
            s1[s].z = (brokeNow || kind == K_PLAIN) ? 0.0f : o.slope;
            if (ext)
            {
                float4 e2 = recE[r * 2 + 1];
                uint mask = as_type<uint>(bb.w);
                bool yields = kind == K_JPLAIN || kind == K_JSUPPORT || kind == K_JANISO;
                if ((mask & EXT_DEFORM) != 0u && (brokeNow || (yields && fabs(s0[s].x - e2.x) > e2.w * e2.x)))
                    deformReq[s] = true;
            }
            if (brokeNow && group > 0) { atomic_fetch_min_explicit(&tgMaster[group], r, memory_order_relaxed); tgFlags[1] = 1u; }
            V3 push = o.d * (f * o.inv);
            PUT_ADD(ent.x, a.x, push);
            PUT_ADD(ent.y, a.y, -push);
#if FAST_SUMS
            lastPush[s] = push;
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // ---- S4: node sums of the beam pass; break groups set off ----
        const bool anyFired = tgFlags[1] != 0u;
#if FAST_SUMS
        if (anyFired)
        {
            // Beams after a group's first break in this pass were skipped: take theirs back.
            for (uint s = 0; s < Z_RS; s++)
            {
                uint r = rec[s];
                if (r == NONE) continue;
                uint4 a = recA[r];
                int group = as_type<int>(a.w);
                if (group == 0) continue;
                uint m = atomic_load_explicit(&tgMaster[abs(group)], memory_order_relaxed);
                if (m != NONE && r > m && (a.z & 0xFFu) != K_JTORSION)
                {
                    PUT_ADD(0, a.x, -lastPush[s]);
                    PUT_ADD(0, a.y, lastPush[s]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
#endif
        if (isNode && !SKIP(4))
        {
            uint n = min(nr.y, (uint)NODE_MAX_ADD);
            V3 f = F0;
#if FAST_SUMS
            n = 0;
            f = f + v3(atomic_exchange_explicit(&fAcc[ni * 3], 0.0f, memory_order_relaxed), atomic_exchange_explicit(&fAcc[ni * 3 + 1], 0.0f, memory_order_relaxed), atomic_exchange_explicit(&fAcc[ni * 3 + 2], 0.0f, memory_order_relaxed));
#endif
            for (uint k = 0; k < n; k++)
            {
                float4 e = addEnt[nr.x + k];
                if (as_type<uint>(e.x) == SENT) continue;
                if (anyFired)
                {
                    uint2 info = addInfo[nr.x + k];
                    int g = as_type<int>(info.y);
                    if (g != 0)
                    {
                        uint m = atomic_load_explicit(&tgMaster[abs(g)], memory_order_relaxed);
                        if (m != NONE && info.x > m) continue;
                    }
                }
                f = f + v3(e);
            }
            F0 = f;
#if COLLIDE
            tgF[ni * 3] = F0.x; tgF[ni * 3 + 1] = F0.y; tgF[ni * 3 + 2] = F0.z;
#endif
        }
        for (uint s = 0; s < Z_RS && !SKIP(10); s++)
        {
            uint r = rec[s];
            if (r == NONE) continue;
            uint4 a = recA[r];
            int group = as_type<int>(a.w);
            bool ext = ((a.z >> 16) & 1u) != 0u;
            uint emask = ext ? as_type<uint>(recB[r].w) : 0u;
            bool trig = deformReq[s];
            if (anyFired && group != 0)
            {
                uint m = atomic_load_explicit(&tgMaster[abs(group)], memory_order_relaxed);
                if (m != NONE)
                {
                    // cascade(): every member broken; one after the first to break this pass
                    // was skipped (no force, no slope, no yield-triggered deform).
                    if (r > m) { s1[s].z = 0.0f; trig = false; }
                    s1[s].w = 1.0f;
                    // cascade_ext(): a member with an extension record triggers its deform groups.
                    if ((emask & EXT_DEFORM) != 0u) trig = true;
                }
            }
            if (trig)
            {
                uint d0 = (emask >> 8) & 0xFFu, d1 = (emask >> 16) & 0xFFu;
                if (d0 != 0u) deform[d0 - 1u] = 1u;
                if (d1 != 0u) deform[d1 - 1u] = 1u;
            }
            // The broken flag for the next step's triangle breaks.
            recState[r * 3 + 1].w = s1[s].w;
            if ((a.z & 0xFFu) == K_JTORSION) recState[(r + 1) * 3 + 1].w = s1[s].w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

#if COLLIDE
        // ---- S5: self-collision ----
        // S5a: each triangle's frames and boxes; whether the pair list still holds.
        const float vmax = as_type<float>(atomic_load_explicit(&tgCount[0], memory_order_relaxed));
        for (uint t = lid; t < Z_NT && !SKIP(5); t += Z_LANES)
        {
            uint4 tn = tris[t];
            V3 p[3], v[3];
            for (uint q = 0; q < 3; q++)
            {
                uint i = tn[q];
                p[q] = v3(0, 0, 0) + v3(tgPos[i * 3], tgPos[i * 3 + 1], tgPos[i * 3 + 2]);
                v[q] = v3(tgVel[i * 3], tgVel[i * 3 + 1], tgVel[i * 3 + 2]);
            }
            Frame F = triFrame(p[0], p[1], p[2]);
            V3 b[3];
            for (uint q = 0; q < 3; q++) b[q] = p[q] - v[q] * H;
            Frame B = triFrame(b[0], b[1], b[2]);
            float tv = rmax(rmax(rmax(0.0f, len3(v[0])), len3(v[1])), len3(v[2]));
            float reach = (CR + (tv + vmax) * H) + CLIP_DEPTH;
            V3 mn = v3(rmin(rmin(p[0].x, p[1].x), p[2].x), rmin(rmin(p[0].y, p[1].y), p[2].y), rmin(rmin(p[0].z, p[1].z), p[2].z));
            V3 mx = v3(rmax(rmax(p[0].x, p[1].x), p[2].x), rmax(rmax(p[0].y, p[1].y), p[2].y), rmax(rmax(p[0].z, p[1].z), p[2].z));
            device float4* g = triGeo + t * GEO;
            g[0] = f4(F.p0, F.ok ? 1.0f : 0.0f);
            g[1] = f4(F.n, 0);
            g[2] = f4(F.ra, 0);
            g[3] = f4(F.rb, 0);
            g[4] = f4(F.rd, 0);
            g[5] = f4(B.p0, B.ok ? 1.0f : 0.0f);
            g[6] = f4(B.ra, 0);
            g[7] = f4(B.rb, 0);
            g[8] = f4(B.rd, 0);
            g[9] = f4(mn - v3(reach, reach, reach), triDead[t] != 0u ? 1.0f : 0.0f);
            g[10] = f4(mx + v3(reach, reach, reach), 0);
            g[11] = f4(mn - v3(LIST_REACH, LIST_REACH, LIST_REACH), 0);
            g[12] = f4(mx + v3(LIST_REACH, LIST_REACH, LIST_REACH), 0);
        }
        if (lid == 0)
        {
            // The list holds every pair the box test can pass while each node has moved at
            // most D (per axis, about the common translation) since its build and the reach
            // is at most r + 2·vmax·h + CLIP_DEPTH: D + D + that ≤ LIST_REACH.
            float D = 0.0f;
            for (uint a = 0; a < 3; a++)
            {
                float lo = unordF(atomic_load_explicit(&tgDisp[a * 2], memory_order_relaxed));
                float hi = unordF(atomic_load_explicit(&tgDisp[a * 2 + 1], memory_order_relaxed));
                D = rmax(D, 0.5f * (hi - lo));
            }
            float need = ((CR + 2.0f * vmax * H) + CLIP_DEPTH) + 2.0f * D + LIST_EPS;
            if (!(need <= LIST_REACH)) tgFlags[2] = 1u;
            if (tgFlags[2] != 0u) atomic_store_explicit(&tgCount[1], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (tgFlags[2] != 0u)
        {
            // S5b: the list built: every allowed pair inside the list box.
            for (uint k = lid; k < Z_NT * Z_NC; k += Z_LANES)
            {
                uint t = k / Z_NC, c = k - t * Z_NC;
                if ((allowed[t * NCW + (c >> 5)] & (1u << (c & 31u))) == 0u) continue;
                device float4* g = triGeo + t * GEO;
                if (g[9].w != 0.0f) continue; // a dead triangle stays dead
                uint h = cands[c];
                V3 p = v3(0, 0, 0) + v3(tgPos[h * 3], tgPos[h * 3 + 1], tgPos[h * 3 + 2]);
                float4 lo = g[11], hi = g[12];
                if (p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z) continue;
                uint i = atomic_fetch_add_explicit(&tgCount[1], 1u, memory_order_relaxed);
                if (i < PAIR_CAP) pairs[i] = t << 16 | h;
            }
            if (isNode) pBuild[ni] = f4(P0, 0);
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            if (lid == 0)
            {
                tgFlags[2] = 0u;
                tgFlags[4] += 1u;
                if (atomic_load_explicit(&tgCount[1], memory_order_relaxed) > PAIR_CAP) tgFlags[6] |= 1u;
            }
        }
        // S5c: the listed pairs: the box test, then anti-clip's hit test.
        const uint npairs = min(atomic_load_explicit(&tgCount[1], memory_order_relaxed), (uint)PAIR_CAP);
        for (uint k = lid; k < npairs && !SKIP(7); k += Z_LANES)
        {
            uint pr = pairs[k];
            uint t = pr >> 16, h = pr & 0xFFFFu;
            device float4* g = triGeo + t * GEO;
            float4 lo = g[9], hi = g[10];
            if (lo.w != 0.0f) continue; // dead
            V3 p = v3(0, 0, 0) + v3(tgPos[h * 3], tgPos[h * 3 + 1], tgPos[h * 3 + 2]);
            if (p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z) continue;
            float4 g0 = g[0];
            V3 n = v3(g[1]);
            float4 loc = frameLocal(v3(g0), v3(g[2]), v3(g[3]), v3(g[4]), g0.w != 0.0f, p);
            // The held side for this triangle.
            uint key = CLIP_SELF << 20 | t;
            float side = 0.0f;
            uint c0 = atomic_load_explicit(&clip[h * 4], memory_order_relaxed);
            uint c1 = atomic_load_explicit(&clip[h * 4 + 1], memory_order_relaxed);
            if (c0 != CLIP_EMPTY && (c0 >> 1) == key) side = (c0 & 1u) != 0u ? 1.0f : -1.0f;
            else if (c1 != CLIP_EMPTY && (c1 >> 1) == key) side = (c1 & 1u) != 0u ? 1.0f : -1.0f;
            bool over = loc.x >= 0.0f && loc.y >= 0.0f && loc.z >= 0.0f;
            bool haveW = false;
            float3 w = float3(0);
            if (side != 0.0f)
            {
                haveW = over;
                w = loc.xyz;
            }
            else
            {
                // The swept crossing over the step.
                uint4 tn = tris[t];
                (void)tn;
                V3 vh = v3(tgVel[h * 3], tgVel[h * 3 + 1], tgVel[h * 3 + 2]);
                V3 pp = p - vh * H;
                float4 b0 = g[5];
                float4 l0 = frameLocal(v3(b0), v3(g[6]), v3(g[7]), v3(g[8]), b0.w != 0.0f, pp);
                float d0 = l0.w, d1 = loc.w;
                bool crossed = false;
                if ((d0 > 0.0f && d1 < 0.0f) || (d0 < 0.0f && d1 > 0.0f))
                {
                    float s = d0 / (d0 - d1);
                    float a = l0.x + (loc.x - l0.x) * s;
                    float b = l0.y + (loc.y - l0.y) * s;
                    float3 cw = float3(a, b, (1.0f - a) - b);
                    if (cw.x >= 0.0f && cw.y >= 0.0f && cw.z >= 0.0f)
                    {
                        crossed = true;
                        side = d0 > 0.0f ? 1.0f : -1.0f;
                        haveW = true;
                        w = cw;
                    }
                }
                if (!crossed)
                {
                    if (over && fabs(loc.w) <= CR)
                    {
                        side = loc.w < 0.0f ? -1.0f : 1.0f;
                        haveW = true;
                        w = loc.xyz;
                    }
                    else continue;
                }
            }
            float d = side * loc.w;
            if (!(haveW && d <= CR && d >= -CLIP_DEPTH)) continue;
            // A contact: its side recorded (the two smallest words), the hit kept.
            uint word = CLIP_SELF << 21 | t << 1 | (side > 0.0f ? 1u : 0u);
            uint old = atomic_fetch_min_explicit(&clip[h * 4 + 2], word, memory_order_relaxed);
            atomic_fetch_min_explicit(&clip[h * 4 + 3], max(old, word), memory_order_relaxed);
            float pen = rmin(CR - d, CR);
            V3 nn = n * side;
            uint i = atomic_fetch_add_explicit(&tgCount[2], 1u, memory_order_relaxed);
            uint4 tn = tris[t];
            if (i < HIT_CAP)
            {
                hitBuf[i * HITW] = as_type<float4>(uint4(h, tn.x, tn.y, tn.z));
                hitBuf[i * HITW + 1] = float4(w.x, w.y, w.z, pen);
                hitBuf[i * HITW + 2] = f4(nn, 0);
            }
            atomic_fetch_add_explicit(&tgSplit[h], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&tgSplit[tn.x], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&tgSplit[tn.y], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&tgSplit[tn.z], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        const uint nhits = atomic_load_explicit(&tgCount[2], memory_order_relaxed);
        if (nhits != 0u && !SKIP(8))
        {
            if (lid == 0)
            {
                tgFlags[3] = 1u;
                tgFlags[5] = max(tgFlags[5], nhits);
                if (nhits > HIT_CAP) tgFlags[6] |= 2u;
            }
            const uint nh = min(nhits, (uint)HIT_CAP);
            // S6: the two-body rule's first pass, from the forces at the pass's start.
            for (uint k = lid; k < nh; k += Z_LANES)
            {
                uint4 id = as_type<uint4>(hitBuf[k * HITW]);
                float4 wp = hitBuf[k * HITW + 1];
                V3 n = v3(hitBuf[k * HITW + 2]);
                uint hn = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                float alpha = wp.x, beta = wp.y, gamma = wp.z, depth = wp.w;
                #define NP(i) v3(tgPos[(i) * 3], tgPos[(i) * 3 + 1], tgPos[(i) * 3 + 2])
                #define NV(i) v3(tgVel[(i) * 3], tgVel[(i) * 3 + 1], tgVel[(i) * 3 + 2])
                #define NF(i) (v3(tgF[(i) * 3], tgF[(i) * 3 + 1], tgF[(i) * 3 + 2]) + v3(0, 0, 0))
                #define NM(i) nodeStatic[i].x
                #define NIMM(i) ((as_type<uint>(nodeStatic[i].z) & NODE_IMMOVABLE) != 0u)
                #define ACC(f, i) (NIMM(i) ? v3(0, 0, 0) : (f) / NM(i))
                #define IM(i, kk) (NIMM(i) ? 0.0f : float(kk) / NM(i))
                V3 v = NV(hn) - ((NV(i1) * alpha + NV(i2) * beta) + NV(i0) * gamma);
                float imh = IM(hn, atomic_load_explicit(&tgSplit[hn], memory_order_relaxed));
                float imt = tbInvMass(IM(i1, atomic_load_explicit(&tgSplit[i1], memory_order_relaxed)),
                                      IM(i2, atomic_load_explicit(&tgSplit[i2], memory_order_relaxed)),
                                      IM(i0, atomic_load_explicit(&tgSplit[i0], memory_order_relaxed)), alpha, beta, gamma);
                float inv = imh + imt;
                float m = inv > 0.0f ? 1.0f / inv : 0.0f;
                V3 at = (ACC(NF(i1), i1) * alpha + ACC(NF(i2), i2) * beta) + ACC(NF(i0), i0) * gamma;
                V3 fh = (ACC(NF(hn), hn) - at) * m;
                float budget;
                V3 f = contactResponse(fh, v, m, n, BOXS, depth, nodeStatic[hn].y, budget);
                if (!isZero3(f))
                {
                    float delta = shiftDepth(depth, budget, dot3(fh, n));
                    if (delta > 0.0f)
                    {
                        float q = delta * m;
                        V3 dh = n * (q * IM(hn, 1u));
                        for (uint a = 0; a < 3; a++) atomic_fetch_add_explicit((device atomic_int*)&shift[hn * 3 + a], toFixedShift(a == 0 ? dh.x : (a == 1 ? dh.y : dh.z)), memory_order_relaxed);
                        uint ti[3] = {i1, i2, i0};
                        float tw[3] = {alpha, beta, gamma};
                        for (uint e = 0; e < 3; e++)
                        {
                            if (tw[e] > 0.0f)
                            {
                                V3 dt = n * -(tw[e] * (q * IM(ti[e], 1u)));
                                for (uint a = 0; a < 3; a++) atomic_fetch_add_explicit((device atomic_int*)&shift[ti[e] * 3 + a], toFixedShift(a == 0 ? dt.x : (a == 1 ? dt.y : dt.z)), memory_order_relaxed);
                            }
                        }
                        tgFlags[0] = 1u;
                    }
                }
                hitBuf[k * HITW + 3] = f4(f, 0);
                V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                uint ns[4] = {hn, i1, i2, i0};
                for (uint e = 0; e < 4; e++)
                {
                    atomic_fetch_add_explicit(&stage[ns[e] * 6], toFixed(fs[e].x), memory_order_relaxed);
                    atomic_fetch_add_explicit(&stage[ns[e] * 6 + 1], toFixed(fs[e].y), memory_order_relaxed);
                    atomic_fetch_add_explicit(&stage[ns[e] * 6 + 2], toFixed(fs[e].z), memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            // S7: the residual pass, from the forces with the first pass's sums.
            #define NOW1(i) (v3(tgF[(i) * 3], tgF[(i) * 3 + 1], tgF[(i) * 3 + 2]) + v3(float(atomic_load_explicit(&stage[(i) * 6], memory_order_relaxed)), float(atomic_load_explicit(&stage[(i) * 6 + 1], memory_order_relaxed)), float(atomic_load_explicit(&stage[(i) * 6 + 2], memory_order_relaxed))) * (1.0f / 32.0f))
            for (uint k = lid; k < nh; k += Z_LANES)
            {
                uint4 id = as_type<uint4>(hitBuf[k * HITW]);
                float4 wp = hitBuf[k * HITW + 1];
                V3 n = v3(hitBuf[k * HITW + 2]);
                uint hn = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                float alpha = wp.x, beta = wp.y, gamma = wp.z;
                V3 v = NV(hn) - ((NV(i1) * alpha + NV(i2) * beta) + NV(i0) * gamma);
                V3 at = (ACC(NOW1(i1), i1) * alpha + ACC(NOW1(i2), i2) * beta) + ACC(NOW1(i0), i0) * gamma;
                float a = dot3(ACC(NOW1(hn), hn) - at, n);
                float vn = dot3(v, n);
                float x = vn + rmax(vn, 0.0f);
                float r = 0.0f;
                if (vn < 0.0f && a * H + x < 0.0f)
                {
                    float inv = IM(hn, atomic_load_explicit(&tgSplit[hn], memory_order_relaxed)) +
                                tbInvMass(IM(i1, atomic_load_explicit(&tgSplit[i1], memory_order_relaxed)),
                                          IM(i2, atomic_load_explicit(&tgSplit[i2], memory_order_relaxed)),
                                          IM(i0, atomic_load_explicit(&tgSplit[i0], memory_order_relaxed)), alpha, beta, gamma);
                    if (inv > 0.0f)
                    {
                        float m = 1.0f / inv;
                        float rr = -(a * m) - (x * m) / H;
                        if (rr > 0.0f) r = rr;
                    }
                }
                float4 h3 = hitBuf[k * HITW + 3];
                hitBuf[k * HITW + 3] = float4(h3.xyz, r);
                if (r > 0.0f)
                {
                    V3 f = n * r;
                    V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                    uint ns[4] = {hn, i1, i2, i0};
                    for (uint e = 0; e < 4; e++)
                    {
                        atomic_fetch_add_explicit(&stage[ns[e] * 6 + 3], toFixed(fs[e].x), memory_order_relaxed);
                        atomic_fetch_add_explicit(&stage[ns[e] * 6 + 4], toFixed(fs[e].y), memory_order_relaxed);
                        atomic_fetch_add_explicit(&stage[ns[e] * 6 + 5], toFixed(fs[e].z), memory_order_relaxed);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            // S8: the work guard, from the forces with both passes' sums; the totals added.
            #define SUM2(i, a) (atomic_load_explicit(&stage[(i) * 6 + (a)], memory_order_relaxed) + atomic_load_explicit(&stage[(i) * 6 + 3 + (a)], memory_order_relaxed))
            #define NOW2(i) (v3(tgF[(i) * 3], tgF[(i) * 3 + 1], tgF[(i) * 3 + 2]) + v3(float(SUM2(i, 0)), float(SUM2(i, 1)), float(SUM2(i, 2))) * (1.0f / 32.0f))
            for (uint k = lid; k < nh; k += Z_LANES)
            {
                uint4 id = as_type<uint4>(hitBuf[k * HITW]);
                float4 wp = hitBuf[k * HITW + 1];
                V3 n = v3(hitBuf[k * HITW + 2]);
                float4 h3 = hitBuf[k * HITW + 3];
                uint hn = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                float alpha = wp.x, beta = wp.y, gamma = wp.z;
                V3 t = v3(h3);
                if (h3.w > 0.0f) t = t + n * h3.w;
                V3 f = t;
                if (!isZero3(t))
                {
                    V3 v = NV(hn) - ((NV(i1) * alpha + NV(i2) * beta) + NV(i0) * gamma);
                    V3 a = ACC(NOW2(hn), hn) - ((ACC(NOW2(i1), i1) * alpha + ACC(NOW2(i2), i2) * beta) + ACC(NOW2(i0), i0) * gamma);
                    float inv = IM(hn, 1u) + tbInvMass(IM(i1, 1u), IM(i2, 1u), IM(i0, 1u), alpha, beta, gamma);
                    V3 aRest = a - t * inv;
                    float aa = dot3(t, (v + v) + aRest * H);
                    float bb = (H * dot3(t, t)) * inv;
                    if (aa + bb > 0.0f && bb > 0.0f)
                    {
                        float s = aa < 0.0f ? rmin(-aa / bb, 1.0f) : 0.0f;
                        f = t + t * -(1.0f - s);
                    }
                }
                V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                uint ns[4] = {hn, i1, i2, i0};
                for (uint e = 0; e < 4; e++)
                {
                    atomic_fetch_add_explicit(&acc[ns[e] * 3], toFixed(fs[e].x), memory_order_relaxed);
                    atomic_fetch_add_explicit(&acc[ns[e] * 3 + 1], toFixed(fs[e].y), memory_order_relaxed);
                    atomic_fetch_add_explicit(&acc[ns[e] * 3 + 2], toFixed(fs[e].z), memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
#endif
    }

    // ---- write back ----
    if (isNode)
    {
        nodeState[ni * 3] = f4(P0, 0);
        nodeState[ni * 3 + 1] = f4(V0, 0);
        nodeState[ni * 3 + 2] = f4(F0, 0);
    }
    for (uint s = 0; s < Z_RS; s++)
    {
        uint r = rec[s];
        if (r == NONE) continue;
        recState[r * 3] = s0[s];
        recState[r * 3 + 1] = s1[s];
        recState[r * 3 + 2] = ax[s];
    }
    for (uint g = lid; g < Z_NG; g += Z_LANES)
        groupsLast[g] = atomic_load_explicit(&tgMaster[g], memory_order_relaxed) != NONE ? 1u : 0u;
    if (lid == 0)
    {
        misc[0] = tgFlags[0];
        misc[1] = tgFlags[6];
        misc[4] = tgFlags[3];
        misc[5] = tgFlags[4];
        misc[6] = tgFlags[5];
#if COLLIDE
        misc[2] = atomic_load_explicit(&tgCount[1], memory_order_relaxed);
        misc[3] = tgFlags[2];
#endif
    }
}
