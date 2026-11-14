// Strict closest-feature queries (Rust twin: engine_schema::feature_query): a point, a
// segment or a triangle against a rigid shape (box, convex hull, capsule, vertical
// cylinder, sphere, a single triangle: a trimesh's or a heightfield cell's), each
// contact with barycentric weights on the element's vertices, its normal, gap and point.
// Used by the explicit vehicles' surface contact (jbeam-semantics-spec.md §38,
// explicit_beams_surface.slang); the rigid engine's SAT manifolds (convex.slang,
// box_collide.slang) are separate and unchanged.
//
// Every operation is strict single precision (strict.slang) in the twin's order, so the
// GPU and the CPU agree bit for bit. No workgroup memory.
//
// Hulls: the shape holds its pose (c, ax0..ax2) and a handle; the includer defines the
// accessors of its local records (vertices and planes with the collider's scale
// applied, edges as vertex pairs), as the World's hull arena (engine_schema::convex)
// stores them:
//   float3 fqHullVertex(FqShape s, uint i);
//   uint2 fqHullEdge(FqShape s, uint e);
//   float4 fqHullPlane(FqShape s, uint f);   // (n, d): n·x ≤ d inside
// Loops over a hull's records have constant bounds (FQ_HULL_* below) and break at its
// counts.
#ifndef ENGINE_PHYSICS_FEATURE_QUERY_SLANG
#define ENGINE_PHYSICS_FEATURE_QUERY_SLANG

#include "strict.slang"

static const float FQ_TIE = 1e-4;
static const float FQ_PIERCE = 1e-5;
static const uint FQ_BOX = 0u;
static const uint FQ_CYL = 2u;
static const uint FQ_CAP = 3u;
static const uint FQ_SPH = 4u;
static const uint FQ_HULL = 5u;
static const uint FQ_TRI = 6u;
static const uint FQ_HULL_VERTS = 64u;
static const uint FQ_HULL_FACES = 128u;
static const uint FQ_HULL_EDGES = 192u;

struct FqShape
{
    uint kind;   // FQ_*
    float3 c;    // box, cylinder, sphere centre; capsule a; hull position; triangle vertex 0
    float3 b;    // capsule b; triangle vertex 1
    float3 ax0;  // box and hull axes (local x, y, z in world); triangle vertex 2 in ax0
    float3 ax1;
    float3 ax2;
    float3 half; // box half extents; .x radius, .y half height (cylinder); .x radius (capsule, sphere)
    uint hull;   // hull: the includer's handle of its records
    uint nv;     // hull: vertices, faces, edges
    uint nf;
    uint ne;
    uint moved;  // hull: cast by `off` (fqTranslate), its world records then moved
    float3 off;
};

float3 fqHullVertex(FqShape s, uint i);
uint2 fqHullEdge(FqShape s, uint e);
float4 fqHullPlane(FqShape s, uint f);

// Candidate pair: weights and point on the element, point on the shape, distance.
struct FqPair
{
    float3 w;
    float3 q;
    float3 cs;
    float d;
};

// The running reduction over candidates: pass 0 the nearest gap, pass 1 the sums
// of those within FQ_TIE of it; touching or crossing (the twin's `deepest`), along
// the separating axis n pass 2 the deepest of those with shared weights, pass 3 the
// sums of those within FQ_TIE of it, pass 4 (a beam) the deepest at the least and
// the most parameter.
struct FqAcc
{
    uint pass;
    float gx;
    float3 w;
    float3 q;
    float3 cs;
    uint k;
    float3 n;
    float top;
    uint any;
    float3 lw;
    float3 lq;
    float3 hw;
    float3 hq;
    float lim;  // pass 0: candidates past min(gx, lim) cannot matter (fqFar)
};

// --- Strict poses (the twin's rot, qmul, normalize4, shape_pose, axes, to_world) ---

float3 fqRot(float4 q, float3 v)
{
    float3 t = sscale(scross(q.xyz, v), 2.0);
    return sadd3(sadd3(v, sscale(t, q.w)), scross(q.xyz, t));
}

float4 fqQmul(float4 a, float4 b)
{
    float3 v = sadd3(sadd3(sscale(b.xyz, a.w), sscale(a.xyz, b.w)), scross(a.xyz, b.xyz));
    return float4(v, nf(smul(a.w, b.w) - sdot(a.xyz, b.xyz)));
}

float4 fqNormalize4(float4 q)
{
    float l = ssqrt(nf(nf(nf(smul(q.x, q.x) + smul(q.y, q.y)) + smul(q.z, q.z)) + smul(q.w, q.w)));
    return float4(sdiv(q.x, l), sdiv(q.y, l), sdiv(q.z, l), sdiv(q.w, l));
}

// A collider's world position (.xyz of p) and orientation (r) from its body's centre of
// mass c, rotation q and its own pose (lp, lq) in the body's frame.
float3 fqPose(float3 c, float4 q, float3 lp, float4 lq, out float4 r)
{
    r = fqNormalize4(fqQmul(q, lq));
    return sadd3(c, fqRot(q, lp));
}

float3 fqToWorld(FqShape s, float3 l)
{
    return sadd3(sadd3(sadd3(s.c, sscale(s.ax0, l.x)), sscale(s.ax1, l.y)), sscale(s.ax2, l.z));
}

float3 fqTurn(FqShape s, float3 n) { return sadd3(sadd3(sscale(s.ax0, n.x), sscale(s.ax1, n.y)), sscale(s.ax2, n.z)); }

// A hull's world vertex i and plane f.
float3 fqHullV(FqShape s, uint i)
{
    float3 v = fqToWorld(s, fqHullVertex(s, i));
    return s.moved != 0u ? sadd3(v, s.off) : v;
}
float4 fqHullP(FqShape s, uint f)
{
    float4 p = fqHullPlane(s, f);
    float3 wn = fqTurn(s, p.xyz);
    float w = nf(p.w + sdot(wn, s.c));
    return float4(wn, s.moved != 0u ? nf(w + sdot(wn, s.off)) : w);
}

// Triangle shape vertex k.
float3 fqTriV(FqShape s, uint k) { return k == 0u ? s.c : (k == 1u ? s.b : s.ax0); }

// --- The reduction ---

bool fqShared(float3 w) { return (w.x > 0.0 ? 1u : 0u) + (w.y > 0.0 ? 1u : 0u) + (w.z > 0.0 ? 1u : 0u) >= 2u; }

void fqVisit(inout FqAcc a, FqPair p, float g)
{
    if (a.pass == 0u)
    {
        a.gx = min(a.gx, g);
        return;
    }
    if (!(g <= nf(a.gx + FQ_TIE))) return;
    if (a.pass == 2u)
    {
        if (!fqShared(p.w)) return;
        float d = sdot(a.n, p.q);
        a.top = a.any == 0u ? d : max(a.top, d);
        a.any = 1u;
        return;
    }
    if (a.pass >= 3u && !(fqShared(p.w) && sdot(a.n, p.q) >= nf(a.top - FQ_TIE))) return;
    if (a.pass == 4u)
    {
        if (a.k == 0u || p.w.y < a.lw.y) { a.lw = p.w; a.lq = p.q; }
        if (a.k == 0u || p.w.y > a.hw.y) { a.hw = p.w; a.hq = p.q; }
        a.k = 1u;
        return;
    }
    a.w = float3(nf(a.w.x + p.w.x), nf(a.w.y + p.w.y), nf(a.w.z + p.w.z));
    a.q = sadd3(a.q, p.q);
    a.cs = sadd3(a.cs, p.cs);
    a.k = a.k + 1u;
}

FqAcc fqAccNew()
{
    FqAcc a;
    a.pass = 0u;
    a.gx = 1e30;
    a.w = float3(0.0, 0.0, 0.0);
    a.q = float3(0.0, 0.0, 0.0);
    a.cs = float3(0.0, 0.0, 0.0);
    a.k = 0u;
    a.n = float3(0.0, 0.0, 0.0);
    a.top = 0.0;
    a.any = 0u;
    a.lw = float3(0.0, 0.0, 0.0);
    a.lq = a.lw;
    a.hw = a.lw;
    a.hq = a.lw;
    a.lim = 3.0e38;
    return a;
}

// --- Lossless culls inside the candidate walks (no twin: the same candidates count) ---
//
// A pass keeps the candidates of gap at most its threshold: pass 0 the least so far
// (with a.lim, the band of a contact query: a least gap past it is no contact either
// way), later passes gx + FQ_TIE. Every candidate of a shape edge pairs a point of the
// edge with a point of the element, so its gap (a box's, a hull's or a triangle
// shape's: the points' distance) is at least the gap between their boxes, less the
// rounding of the computed points (within 1e-5 of the largest coordinate and 1e-4 m,
// far past it). An edge whose box is farther than the threshold by that is skipped.
// The includer's switch (tests and measurement): false, every candidate is walked.
bool fqCullsOn();
float fqPrune(FqAcc a)
{
    if (!fqCullsOn()) return 3.0e38;
    return a.pass == 0u ? min(a.gx, a.lim) : a.gx + FQ_TIE;
}

bool fqFar(float3 e0, float3 e1, float3 tlo, float3 thi, float thr)
{
    float3 elo = min(e0, e1);
    float3 ehi = max(e0, e1);
    float3 gap = max(max(elo - thi, tlo - ehi), float3(0.0, 0.0, 0.0));
    float3 m3 = max(max(abs(elo), abs(ehi)), max(abs(tlo), abs(thi)));
    float t = thr + 1e-4 + 1e-5 * max(max(m3.x, m3.y), m3.z);
    return dot(gap, gap) > t * t;
}

void fqAccDeep(inout FqAcc a, float3 n, uint pass)
{
    a.pass = pass;
    a.n = n;
    a.w = float3(0.0, 0.0, 0.0);
    a.q = float3(0.0, 0.0, 0.0);
    a.cs = float3(0.0, 0.0, 0.0);
    a.k = 0u;
}

float fqLen(float3 v) { return ssqrt(sdot(v, v)); }
float fqClamp01(float x) { return min(max(x, 0.0), 1.0); }

// Closest point on triangle a b c to p (Ericson §5.1.5): closest_on_triangle.
float3 fqClosestTri(float3 p, float3 a, float3 b, float3 c, out float3 w)
{
    float3 ab = ssub3(b, a);
    float3 ac = ssub3(c, a);
    float3 ap = ssub3(p, a);
    float d1 = sdot(ab, ap);
    float d2 = sdot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { w = float3(1.0, 0.0, 0.0); return a; }
    float3 bp = ssub3(p, b);
    float d3 = sdot(ab, bp);
    float d4 = sdot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { w = float3(0.0, 1.0, 0.0); return b; }
    float vc = nf(smul(d1, d4) - smul(d3, d2));
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        float v = sdiv(d1, nf(d1 - d3));
        w = float3(nf(1.0 - v), v, 0.0);
        return sadd3(a, sscale(ab, v));
    }
    float3 cp = ssub3(p, c);
    float d5 = sdot(ab, cp);
    float d6 = sdot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { w = float3(0.0, 0.0, 1.0); return c; }
    float vb = nf(smul(d5, d2) - smul(d1, d6));
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        float t = sdiv(d2, nf(d2 - d6));
        w = float3(nf(1.0 - t), 0.0, t);
        return sadd3(a, sscale(ac, t));
    }
    float va = nf(smul(d3, d6) - smul(d5, d4));
    float e43 = nf(d4 - d3);
    float e56 = nf(d5 - d6);
    if (va <= 0.0 && e43 >= 0.0 && e56 >= 0.0)
    {
        float t = sdiv(e43, nf(e43 + e56));
        w = float3(0.0, nf(1.0 - t), t);
        return sadd3(b, sscale(ssub3(c, b), t));
    }
    float denom = sdiv(1.0, nf(nf(va + vb) + vc));
    float v = smul(vb, denom);
    float u = smul(vc, denom);
    w = float3(nf(nf(1.0 - v) - u), v, u);
    return sadd3(sadd3(a, sscale(ab, v)), sscale(ac, u));
}

// Closest points of segments p1–q1 and p2–q2 (Ericson §5.1.9): closest_seg_seg.
float2 fqSegSeg(float3 p1, float3 q1, float3 p2, float3 q2)
{
    float3 d1 = ssub3(q1, p1);
    float3 d2 = ssub3(q2, p2);
    float3 r = ssub3(p1, p2);
    float a = sdot(d1, d1);
    float e = sdot(d2, d2);
    float f = sdot(d2, r);
    if (a <= 1e-12 && e <= 1e-12) return float2(0.0, 0.0);
    if (a <= 1e-12) return float2(0.0, fqClamp01(sdiv(f, e)));
    float c = sdot(d1, r);
    if (e <= 1e-12) return float2(fqClamp01(sdiv(-c, a)), 0.0);
    float b = sdot(d1, d2);
    float denom = nf(smul(a, e) - smul(b, b));
    float s = denom > 1e-12 ? fqClamp01(sdiv(nf(smul(b, f) - smul(c, e)), denom)) : 0.0;
    float t = sdiv(nf(smul(b, s) + f), e);
    if (t < 0.0)
    {
        t = 0.0;
        s = fqClamp01(sdiv(-c, a));
    }
    else if (t > 1.0)
    {
        t = 1.0;
        s = fqClamp01(sdiv(nf(b - c), a));
    }
    return float2(s, t);
}

float3 fqEdgeW(uint k, float t)
{
    float3 w = float3(0.0, 0.0, 0.0);
    float u = nf(1.0 - t);
    if (k == 0u) w = float3(u, t, 0.0);
    else if (k == 1u) w = float3(0.0, u, t);
    else w = float3(t, 0.0, u);
    return w;
}

float3 fqX(float3 x0, float3 x1, float3 x2, uint k) { return k == 0u ? x0 : (k == 1u ? x1 : x2); }

// The point of the solid vertical cylinder nearest x (cylinder_point).
float3 fqCylPoint(float3 c, float r, float hh, float3 x)
{
    float3 d = ssub3(x, c);
    float l = ssqrt(nf(smul(d.x, d.x) + smul(d.z, d.z)));
    float px = d.x;
    float pz = d.z;
    if (l > r)
    {
        px = smul(d.x, sdiv(r, l));
        pz = smul(d.z, sdiv(r, l));
    }
    float py = min(max(d.y, -hh), hh);
    return sadd3(c, float3(px, py, pz));
}

// A pair as the shape takes it (the twin's `map` closures): its gap.
float fqMap(FqShape s, inout FqPair p)
{
    if (s.kind == FQ_CAP) return nf(p.d - s.half.x);
    if (s.kind == FQ_CYL)
    {
        p.cs = fqCylPoint(s.c, s.half.x, s.half.y, p.q);
        p.d = fqLen(ssub3(p.q, p.cs));
    }
    return p.d;
}

// Segment p0–p1 crossing triangle t0 t1 t2 strictly at a point inside it (pierce):
// .w 1 with the point (.xyz) and its parameter s.
float4 fqPierce(float3 p0, float3 p1, float3 t0, float3 t1, float3 t2, out float s)
{
    s = 0.0;
    float3 n = scross(ssub3(t1, t0), ssub3(t2, t0));
    float d0 = sdot(ssub3(p0, t0), n);
    float d1 = sdot(ssub3(p1, t0), n);
    if (!((d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0))) return float4(0.0, 0.0, 0.0, 0.0);
    s = sdiv(d0, nf(d0 - d1));
    float3 c = sadd3(p0, sscale(ssub3(p1, p0), s));
    float3 w;
    float3 q = fqClosestTri(c, t0, t1, t2, w);
    float3 qc = ssub3(q, c);
    if (!(sdot(qc, qc) < 1e-10)) return float4(0.0, 0.0, 0.0, 0.0);
    return float4(c, 1.0);
}

// seg_triangle: segment p0–p1's candidate pairs against triangle x0 x1 x2.
void fqSegTri(FqShape s, float3 p0, float3 p1, float3 x0, float3 x1, float3 x2, inout FqAcc a)
{
    for (uint e = 0u; e < 2u; e++)
    {
        float3 pe = e == 0u ? p0 : p1;
        FqPair p;
        p.q = fqClosestTri(pe, x0, x1, x2, p.w);
        p.cs = pe;
        p.d = fqLen(ssub3(p.q, pe));
        float g = fqMap(s, p);
        fqVisit(a, p, g);
    }
    for (uint k = 0u; k < 3u; k++)
    {
        float3 ea = fqX(x0, x1, x2, k);
        float3 eb = fqX(x0, x1, x2, (k + 1u) % 3u);
        float2 st = fqSegSeg(p0, p1, ea, eb);
        float3 pa = sadd3(p0, sscale(ssub3(p1, p0), st.x));
        float3 y = sadd3(ea, sscale(ssub3(eb, ea), st.y));
        FqPair p;
        p.w = fqEdgeW(k, st.y);
        p.q = y;
        p.cs = pa;
        p.d = fqLen(ssub3(pa, y));
        float g = fqMap(s, p);
        fqVisit(a, p, g);
    }
    float3 n = scross(ssub3(x1, x0), ssub3(x2, x0));
    float d0 = sdot(ssub3(p0, x0), n);
    float d1 = sdot(ssub3(p1, x0), n);
    if ((d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0))
    {
        float3 c = sadd3(p0, sscale(ssub3(p1, p0), sdiv(d0, nf(d0 - d1))));
        FqPair p;
        p.q = fqClosestTri(c, x0, x1, x2, p.w);
        float3 qc = ssub3(p.q, c);
        if (sdot(qc, qc) < 1e-10)
        {
            p.cs = c;
            p.d = 0.0;
            float g = fqMap(s, p);
            fqVisit(a, p, g);
        }
    }
}

float3 fqAxis(FqShape s, uint i) { return i == 0u ? s.ax0 : (i == 1u ? s.ax1 : s.ax2); }
float fqHalf(FqShape s, uint i) { return i == 0u ? s.half.x : (i == 1u ? s.half.y : s.half.z); }

// box_planes, plane j (−x, +x, −y, +y, −z, +z): (n, d), inside n·x ≤ d.
float4 fqBoxPlane(FqShape s, uint j)
{
    uint i = j / 2u;
    float3 a = fqAxis(s, i);
    float ca = sdot(a, s.c);
    float h = fqHalf(s, i);
    if ((j & 1u) == 0u) return float4(-a, nf(h - ca));
    return float4(a, nf(ca + h));
}

// The planes of a box (6) or a hull (its faces, world).
uint fqPlaneCount(FqShape s) { return s.kind == FQ_HULL ? s.nf : 6u; }
float4 fqPlane(FqShape s, uint j) { return s.kind == FQ_HULL ? fqHullP(s, j) : fqBoxPlane(s, j); }

// clip_planes: segment p0–p1's run strictly inside the shape's planes (t0 < t1).
bool fqClip(FqShape s, float3 p0, float3 p1, out float t0, out float t1)
{
    t0 = 0.0;
    t1 = 1.0;
    uint np = fqPlaneCount(s);
    for (uint j = 0u; j < FQ_HULL_FACES; j++)
    {
        if (j >= np) break;
        float4 pl = fqPlane(s, j);
        float s0 = nf(sdot(pl.xyz, p0) - pl.w);
        float s1 = nf(sdot(pl.xyz, p1) - pl.w);
        if (s0 >= 0.0 && s1 >= 0.0) return false;
        if (s0 < 0.0 && s1 < 0.0) continue;
        float t = sdiv(s0, nf(s0 - s1));
        if (s0 >= 0.0) t0 = max(t0, t);
        else t1 = min(t1, t);
    }
    return t0 < t1;
}

// tri_edges_inside for a box or a hull.
void fqEdgesInside(FqShape s, float3 x0, float3 x1, float3 x2, inout FqAcc a)
{
    for (uint k = 0u; k < 3u; k++)
    {
        float3 p0 = fqX(x0, x1, x2, k);
        float3 p1 = fqX(x0, x1, x2, (k + 1u) % 3u);
        float t0;
        float t1;
        if (!fqClip(s, p0, p1, t0, t1)) continue;
        for (uint e = 0u; e < 2u; e++)
        {
            float t = e == 0u ? t0 : t1;
            FqPair p;
            p.w = fqEdgeW(k, t);
            p.q = sadd3(p0, sscale(ssub3(p1, p0), t));
            p.cs = p.q;
            p.d = 0.0;
            fqVisit(a, p, 0.0);
        }
    }
}

// x_candidates.
void fqCandidates(FqShape s, float3 x0, float3 x1, float3 x2, inout FqAcc a)
{
    if (s.kind == FQ_SPH)
    {
        FqPair p;
        p.q = fqClosestTri(s.c, x0, x1, x2, p.w);
        p.cs = s.c;
        p.d = fqLen(ssub3(s.c, p.q));
        fqVisit(a, p, nf(p.d - s.half.x));
        return;
    }
    if (s.kind == FQ_CAP)
    {
        fqSegTri(s, s.c, s.b, x0, x1, x2, a);
        return;
    }
    if (s.kind == FQ_CYL)
    {
        float3 h = float3(0.0, s.half.y, 0.0);
        fqSegTri(s, ssub3(s.c, h), sadd3(s.c, h), x0, x1, x2, a);
        return;
    }
    float3 tlo = min(min(x0, x1), x2);
    float3 thi = max(max(x0, x1), x2);
    if (s.kind == FQ_TRI)
    {
        for (uint k = 0u; k < 3u; k++)
        {
            float3 e0 = fqTriV(s, k);
            float3 e1 = fqTriV(s, (k + 1u) % 3u);
            if (fqFar(e0, e1, tlo, thi, fqPrune(a))) continue;
            fqSegTri(s, e0, e1, x0, x1, x2, a);
        }
        for (uint k = 0u; k < 3u; k++)
        {
            float3 p0 = fqX(x0, x1, x2, k);
            float3 p1 = fqX(x0, x1, x2, (k + 1u) % 3u);
            float t;
            float4 c = fqPierce(p0, p1, s.c, s.b, s.ax0, t);
            if (c.w == 0.0) continue;
            FqPair p;
            p.w = fqEdgeW(k, t);
            p.q = c.xyz;
            p.cs = c.xyz;
            p.d = 0.0;
            fqVisit(a, p, 0.0);
        }
        return;
    }
    if (s.kind == FQ_HULL)
    {
        for (uint e = 0u; e < FQ_HULL_EDGES; e++)
        {
            if (e >= s.ne) break;
            uint2 ve = fqHullEdge(s, e);
            float3 e0 = fqHullV(s, ve.x);
            float3 e1 = fqHullV(s, ve.y);
            if (fqFar(e0, e1, tlo, thi, fqPrune(a))) continue;
            fqSegTri(s, e0, e1, x0, x1, x2, a);
        }
        fqEdgesInside(s, x0, x1, x2, a);
        return;
    }
    for (uint i = 0u; i < 3u; i++)
    {
        uint u = (i + 1u) % 3u;
        uint v = (i + 2u) % 3u;
        for (uint k = 0u; k < 4u; k++)
        {
            float su = (k & 1u) != 0u ? fqHalf(s, u) : -fqHalf(s, u);
            float sv = (k & 2u) != 0u ? fqHalf(s, v) : -fqHalf(s, v);
            float3 m = sadd3(sadd3(s.c, sscale(fqAxis(s, u), su)), sscale(fqAxis(s, v), sv));
            float3 e = sscale(fqAxis(s, i), fqHalf(s, i));
            float3 e0 = ssub3(m, e);
            float3 e1 = sadd3(m, e);
            if (fqFar(e0, e1, tlo, thi, fqPrune(a))) continue;
            fqSegTri(s, e0, e1, x0, x1, x2, a);
        }
    }
    fqEdgesInside(s, x0, x1, x2, a);
}

// point_distance: x's distance to the shape when outside (.y 1), else .y 0.
float2 fqPointDist(FqShape s, float3 x)
{
    if (s.kind == FQ_BOX)
    {
        float3 d = ssub3(x, s.c);
        float3 l = float3(sdot(s.ax0, d), sdot(s.ax1, d), sdot(s.ax2, d));
        float3 h = s.half;
        if (l.x > -h.x && l.x < h.x && l.y > -h.y && l.y < h.y && l.z > -h.z && l.z < h.z) return float2(0.0, 0.0);
        float3 k = float3(min(max(l.x, -h.x), h.x), min(max(l.y, -h.y), h.y), min(max(l.z, -h.z), h.z));
        return float2(fqLen(ssub3(l, k)), 1.0);
    }
    if (s.kind == FQ_SPH)
    {
        float d = nf(fqLen(ssub3(x, s.c)) - s.half.x);
        return float2(d, d > 0.0 ? 1.0 : 0.0);
    }
    if (s.kind == FQ_CAP)
    {
        float3 ab = ssub3(s.b, s.c);
        float l2 = sdot(ab, ab);
        float t = l2 > 0.0 ? fqClamp01(sdiv(sdot(ssub3(x, s.c), ab), l2)) : 0.0;
        float d = nf(fqLen(ssub3(x, sadd3(s.c, sscale(ab, t)))) - s.half.x);
        return float2(d, d > 0.0 ? 1.0 : 0.0);
    }
    if (s.kind == FQ_HULL)
    {
        float m = -3.0e38;
        for (uint f = 0u; f < FQ_HULL_FACES; f++)
        {
            if (f >= s.nf) break;
            float4 pl = fqHullP(s, f);
            m = max(m, nf(sdot(pl.xyz, x) - pl.w));
        }
        return float2(m, m >= 0.0 ? 1.0 : 0.0);
    }
    if (s.kind == FQ_TRI)
    {
        float3 w;
        float3 q = fqClosestTri(x, s.c, s.b, s.ax0, w);
        return float2(fqLen(ssub3(x, q)), 1.0);
    }
    float3 d = ssub3(x, s.c);
    float dh = nf(ssqrt(nf(smul(d.x, d.x) + smul(d.z, d.z))) - s.half.x);
    float dy = nf(abs(d.y) - s.half.y);
    if (dh < 0.0 && dy < 0.0) return float2(0.0, 0.0);
    float a = max(dh, 0.0);
    float b = max(dy, 0.0);
    return float2(ssqrt(nf(smul(a, a) + smul(b, b))), 1.0);
}

// project: the shape's (lo, hi) on unit axis u.
float2 fqProject(FqShape s, float3 u)
{
    if (s.kind == FQ_HULL || s.kind == FQ_TRI)
    {
        float lo = 3.0e38;
        float hi = -3.0e38;
        uint n = s.kind == FQ_HULL ? s.nv : 3u;
        for (uint i = 0u; i < FQ_HULL_VERTS; i++)
        {
            if (i >= n) break;
            float p = sdot(u, s.kind == FQ_HULL ? fqHullV(s, i) : fqTriV(s, i));
            lo = min(lo, p);
            hi = max(hi, p);
        }
        return float2(lo, hi);
    }
    float cu;
    float e;
    if (s.kind == FQ_BOX)
    {
        cu = sdot(u, s.c);
        e = nf(nf(smul(abs(sdot(u, s.ax0)), s.half.x) + smul(abs(sdot(u, s.ax1)), s.half.y))
               + smul(abs(sdot(u, s.ax2)), s.half.z));
    }
    else if (s.kind == FQ_SPH)
    {
        cu = sdot(u, s.c);
        e = s.half.x;
    }
    else if (s.kind == FQ_CAP)
    {
        float3 m = sscale(sadd3(s.c, s.b), 0.5);
        float3 h = sscale(ssub3(s.b, s.c), 0.5);
        cu = sdot(u, m);
        e = nf(abs(sdot(u, h)) + s.half.x);
    }
    else
    {
        cu = sdot(u, s.c);
        e = nf(smul(abs(u.y), s.half.y) + smul(s.half.x, ssqrt(nf(smul(u.x, u.x) + smul(u.z, u.z)))));
    }
    return float2(nf(cu - e), nf(cu + e));
}

// One axis of overlap_axis.
void fqAxisTest(FqShape s, float3 u, float3 x0, float3 x1, float3 x2, inout float best, inout float3 axis,
                inout bool first)
{
    float t0 = sdot(u, x0);
    float t1 = sdot(u, x1);
    float t2 = sdot(u, x2);
    float tlo = min(min(t0, t1), t2);
    float thi = max(max(t0, t1), t2);
    float2 sp = fqProject(s, u);
    float plus = nf(thi - sp.x);
    float minus = nf(sp.y - tlo);
    float ov = plus <= minus ? plus : minus;
    float3 ax = plus <= minus ? u : -u;
    if (first || ov < nf(best - 1e-5))
    {
        best = ov;
        axis = ax;
        first = false;
    }
}

// The shape's edge directions (the twin's `dirs`): a box's 3 axes, a cylinder's y, a
// capsule's axis, a hull's edges (world), a triangle's 3 edges.
uint fqDirCount(FqShape s)
{
    if (s.kind == FQ_BOX || s.kind == FQ_TRI) return 3u;
    if (s.kind == FQ_SPH) return 0u;
    if (s.kind == FQ_HULL) return s.ne;
    return 1u;
}
float3 fqDir(FqShape s, uint j)
{
    if (s.kind == FQ_BOX) return fqAxis(s, j);
    if (s.kind == FQ_CYL) return float3(0.0, 1.0, 0.0);
    if (s.kind == FQ_TRI) return ssub3(fqTriV(s, (j + 1u) % 3u), fqTriV(s, j));
    if (s.kind == FQ_HULL)
    {
        uint2 e = fqHullEdge(s, j);
        return ssub3(fqHullV(s, e.y), fqHullV(s, e.x));
    }
    return ssub3(s.b, s.c);
}

// A triangle shape's unit normal (.w 1), none when degenerate (tri_normal).
float4 fqTriNormal(FqShape s)
{
    float3 n = scross(ssub3(s.b, s.c), ssub3(s.ax0, s.c));
    float l = fqLen(n);
    if (!(l > 1e-8)) return float4(0.0, 0.0, 0.0, 0.0);
    return float4(sdiv3(n, l), 1.0);
}

// overlap_axis.
float3 fqOverlapAxis(FqShape s, float3 x0, float3 x1, float3 x2, float3 n)
{
    float best = 0.0;
    float3 axis = n;
    bool first = true;
    fqAxisTest(s, n, x0, x1, x2, best, axis, first);
    if (s.kind == FQ_BOX)
    {
        fqAxisTest(s, s.ax0, x0, x1, x2, best, axis, first);
        fqAxisTest(s, s.ax1, x0, x1, x2, best, axis, first);
        fqAxisTest(s, s.ax2, x0, x1, x2, best, axis, first);
    }
    else if (s.kind == FQ_CYL)
        fqAxisTest(s, float3(0.0, 1.0, 0.0), x0, x1, x2, best, axis, first);
    else if (s.kind == FQ_HULL)
    {
        for (uint f = 0u; f < FQ_HULL_FACES; f++)
        {
            if (f >= s.nf) break;
            fqAxisTest(s, fqHullP(s, f).xyz, x0, x1, x2, best, axis, first);
        }
    }
    else if (s.kind == FQ_TRI)
    {
        float4 tn = fqTriNormal(s);
        if (tn.w != 0.0) fqAxisTest(s, tn.xyz, x0, x1, x2, best, axis, first);
    }
    uint nd = fqDirCount(s);
    for (uint k = 0u; k < 3u; k++)
    {
        float3 e = ssub3(fqX(x0, x1, x2, (k + 1u) % 3u), fqX(x0, x1, x2, k));
        for (uint j = 0u; j < FQ_HULL_EDGES; j++)
        {
            if (j >= nd) break;
            float3 d = fqDir(s, j);
            float3 c = scross(e, d);
            float l2 = sdot(c, c);
            if (!(l2 > smul(1e-12, smul(sdot(e, e), sdot(d, d))))) continue;
            fqAxisTest(s, sdiv3(c, ssqrt(l2)), x0, x1, x2, best, axis, first);
        }
    }
    return axis;
}

// triangle_contact: a hit (weights w, normal n, gap g, point q).
bool fqContact(FqShape s, float3 x0, float3 x1, float3 x2, float r, out float3 w, out float3 n, out float g,
               out float3 q)
{
    w = float3(0.0, 0.0, 0.0);
    n = float3(0.0, 0.0, 0.0);
    g = 0.0;
    q = float3(0.0, 0.0, 0.0);
    float3 nc = scross(ssub3(x1, x0), ssub3(x2, x0));
    float nl = fqLen(nc);
    if (!(nl > 1e-8)) return false;
    float3 tn = sdiv3(nc, nl);
    FqAcc a = fqAccNew();
    a.lim = r;
    fqCandidates(s, x0, x1, x2, a);
    float gx = a.gx;
    if (!(gx <= r)) return false;
    for (uint k = 0u; k < 3u; k++)
    {
        float2 pd = fqPointDist(s, fqX(x0, x1, x2, k));
        if (pd.y != 0.0 && pd.x <= nf(gx + FQ_TIE)) return false;
    }
    a.pass = 1u;
    fqCandidates(s, x0, x1, x2, a);
    float kf = float(a.k);
    w = float3(sdiv(a.w.x, kf), sdiv(a.w.y, kf), sdiv(a.w.z, kf));
    q = sdiv3(a.q, kf);
    float3 cs = sdiv3(a.cs, kf);
    uint nz = (w.x > 0.0 ? 1u : 0u) + (w.y > 0.0 ? 1u : 0u) + (w.z > 0.0 ? 1u : 0u);
    if (nz <= 1u) return false;
    float3 dv = ssub3(cs, q);
    float dl = fqLen(dv);
    g = gx;
    if (gx > FQ_PIERCE && dl > FQ_PIERCE)
    {
        n = sdiv3(dv, dl);
        return true;
    }
    n = fqOverlapAxis(s, x0, x1, x2, tn);
    fqAccDeep(a, n, 2u);
    fqCandidates(s, x0, x1, x2, a);
    if (a.any == 0u) return true;
    fqAccDeep(a, n, 3u);
    fqCandidates(s, x0, x1, x2, a);
    kf = float(a.k);
    w = float3(sdiv(a.w.x, kf), sdiv(a.w.y, kf), sdiv(a.w.z, kf));
    q = sdiv3(a.q, kf);
    return true;
}

// --- Segments (the twin's segment_contact) ---

float fqSegParam(float3 p0, float3 d, float dd, float3 x) { return fqClamp01(sdiv(sdot(ssub3(x, p0), d), dd)); }

// edge_beam: shape edge e0–e1 against beam p0–p1 (d = p1 − p0, dd = d·d).
void fqEdgeBeam(FqShape s, float3 e0, float3 e1, float3 p0, float3 p1, float3 d, float dd, inout FqAcc a)
{
    float3 de = ssub3(e1, e0);
    float3 c = scross(d, de);
    if (sdot(c, c) > smul(1e-12, smul(dd, sdot(de, de))))
    {
        float2 st = fqSegSeg(p0, p1, e0, e1);
        FqPair p;
        p.q = sadd3(p0, sscale(d, st.x));
        p.cs = sadd3(e0, sscale(de, st.y));
        p.w = float3(nf(1.0 - st.x), st.x, 0.0);
        p.d = fqLen(ssub3(p.q, p.cs));
        float g = fqMap(s, p);
        fqVisit(a, p, g);
    }
    for (uint e = 0u; e < 2u; e++)
    {
        float3 pe = e == 0u ? e0 : e1;
        float t = fqSegParam(p0, d, dd, pe);
        FqPair p;
        p.q = sadd3(p0, sscale(d, t));
        p.cs = pe;
        p.w = float3(nf(1.0 - t), t, 0.0);
        p.d = fqLen(ssub3(p.q, pe));
        float g = fqMap(s, p);
        fqVisit(a, p, g);
    }
}

// beam_inside: the beam's run inside a box's or hull's planes, both ends.
void fqBeamInside(FqShape s, float3 p0, float3 p1, inout FqAcc a)
{
    float t0;
    float t1;
    if (!fqClip(s, p0, p1, t0, t1)) return;
    for (uint e = 0u; e < 2u; e++)
    {
        float t = e == 0u ? t0 : t1;
        FqPair p;
        p.q = sadd3(p0, sscale(ssub3(p1, p0), t));
        p.cs = p.q;
        p.w = float3(nf(1.0 - t), t, 0.0);
        p.d = 0.0;
        fqVisit(a, p, 0.0);
    }
}

// seg_candidates.
void fqSegCandidates(FqShape s, float3 p0, float3 p1, inout FqAcc a)
{
    float3 d = ssub3(p1, p0);
    float dd = sdot(d, d);
    if (s.kind == FQ_SPH)
    {
        float t = fqSegParam(p0, d, dd, s.c);
        FqPair p;
        p.q = sadd3(p0, sscale(d, t));
        p.cs = s.c;
        p.w = float3(nf(1.0 - t), t, 0.0);
        p.d = fqLen(ssub3(s.c, p.q));
        fqVisit(a, p, nf(p.d - s.half.x));
        return;
    }
    if (s.kind == FQ_CAP)
    {
        fqEdgeBeam(s, s.c, s.b, p0, p1, d, dd, a);
        return;
    }
    if (s.kind == FQ_CYL)
    {
        float3 h = float3(0.0, s.half.y, 0.0);
        fqEdgeBeam(s, ssub3(s.c, h), sadd3(s.c, h), p0, p1, d, dd, a);
        return;
    }
    float3 tlo = min(p0, p1);
    float3 thi = max(p0, p1);
    if (s.kind == FQ_TRI)
    {
        for (uint k = 0u; k < 3u; k++)
        {
            float3 e0 = fqTriV(s, k);
            float3 e1 = fqTriV(s, (k + 1u) % 3u);
            if (fqFar(e0, e1, tlo, thi, fqPrune(a))) continue;
            fqEdgeBeam(s, e0, e1, p0, p1, d, dd, a);
        }
        float t;
        float4 c = fqPierce(p0, p1, s.c, s.b, s.ax0, t);
        if (c.w != 0.0)
        {
            FqPair p;
            p.q = c.xyz;
            p.cs = c.xyz;
            p.w = float3(nf(1.0 - t), t, 0.0);
            p.d = 0.0;
            fqVisit(a, p, 0.0);
        }
        return;
    }
    if (s.kind == FQ_HULL)
    {
        for (uint e = 0u; e < FQ_HULL_EDGES; e++)
        {
            if (e >= s.ne) break;
            uint2 ve = fqHullEdge(s, e);
            float3 e0 = fqHullV(s, ve.x);
            float3 e1 = fqHullV(s, ve.y);
            if (fqFar(e0, e1, tlo, thi, fqPrune(a))) continue;
            fqEdgeBeam(s, e0, e1, p0, p1, d, dd, a);
        }
        fqBeamInside(s, p0, p1, a);
        return;
    }
    for (uint i = 0u; i < 3u; i++)
    {
        uint u = (i + 1u) % 3u;
        uint v = (i + 2u) % 3u;
        for (uint k = 0u; k < 4u; k++)
        {
            float su = (k & 1u) != 0u ? fqHalf(s, u) : -fqHalf(s, u);
            float sv = (k & 2u) != 0u ? fqHalf(s, v) : -fqHalf(s, v);
            float3 m = sadd3(sadd3(s.c, sscale(fqAxis(s, u), su)), sscale(fqAxis(s, v), sv));
            float3 e = sscale(fqAxis(s, i), fqHalf(s, i));
            float3 e0 = ssub3(m, e);
            float3 e1 = sadd3(m, e);
            if (fqFar(e0, e1, tlo, thi, fqPrune(a))) continue;
            fqEdgeBeam(s, e0, e1, p0, p1, d, dd, a);
        }
    }
    fqBeamInside(s, p0, p1, a);
}

// One axis of seg_overlap_axis.
void fqSegAxisTest(FqShape s, float3 u, float3 x0, float3 x1, inout float best, inout float3 axis, inout bool first)
{
    float t0 = sdot(u, x0);
    float t1 = sdot(u, x1);
    float tlo = min(t0, t1);
    float thi = max(t0, t1);
    float2 sp = fqProject(s, u);
    float plus = nf(thi - sp.x);
    float minus = nf(sp.y - tlo);
    float ov = plus <= minus ? plus : minus;
    float3 ax = plus <= minus ? u : -u;
    if (first || ov < nf(best - 1e-5))
    {
        best = ov;
        axis = ax;
        first = false;
    }
}

// seg_overlap_axis.
float3 fqSegOverlapAxis(FqShape s, float3 x0, float3 x1)
{
    float3 d = ssub3(x1, x0);
    float best = 0.0;
    float3 axis = float3(0.0, 0.0, 0.0);
    bool first = true;
    if (s.kind == FQ_BOX)
    {
        fqSegAxisTest(s, s.ax0, x0, x1, best, axis, first);
        fqSegAxisTest(s, s.ax1, x0, x1, best, axis, first);
        fqSegAxisTest(s, s.ax2, x0, x1, best, axis, first);
    }
    else if (s.kind == FQ_CYL)
        fqSegAxisTest(s, float3(0.0, 1.0, 0.0), x0, x1, best, axis, first);
    else if (s.kind == FQ_HULL)
    {
        for (uint f = 0u; f < FQ_HULL_FACES; f++)
        {
            if (f >= s.nf) break;
            fqSegAxisTest(s, fqHullP(s, f).xyz, x0, x1, best, axis, first);
        }
    }
    else if (s.kind == FQ_TRI)
    {
        float4 tn = fqTriNormal(s);
        if (tn.w != 0.0) fqSegAxisTest(s, tn.xyz, x0, x1, best, axis, first);
    }
    uint nd = fqDirCount(s);
    for (uint j = 0u; j < FQ_HULL_EDGES; j++)
    {
        if (j >= nd) break;
        float3 e = fqDir(s, j);
        float3 c = scross(d, e);
        float l2 = sdot(c, c);
        if (!(l2 > smul(1e-12, smul(sdot(d, d), sdot(e, e))))) continue;
        fqSegAxisTest(s, sdiv3(c, ssqrt(l2)), x0, x1, best, axis, first);
    }
    if (first)
    {
        float3 ax = abs(d.x) <= max(abs(d.y), abs(d.z)) ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
        float3 c = scross(d, ax);
        fqSegAxisTest(s, sdiv3(c, fqLen(c)), x0, x1, best, axis, first);
    }
    return axis;
}

// segment_contact: a hit of beam x0–x1 (weights (w0, w1, 0), normal n, gap g, point q),
// and when `two` a second (w2, q2; the same normal and gap).
bool fqSegContact(FqShape s, float3 x0, float3 x1, float r, out float3 w, out float3 n, out float g, out float3 q,
                  out bool two, out float3 w2, out float3 q2)
{
    two = false;
    w2 = float3(0.0, 0.0, 0.0);
    q2 = w2;
    w = float3(0.0, 0.0, 0.0);
    n = float3(0.0, 0.0, 0.0);
    g = 0.0;
    q = float3(0.0, 0.0, 0.0);
    float3 d = ssub3(x1, x0);
    if (!(sdot(d, d) > 1e-12)) return false;
    FqAcc a = fqAccNew();
    a.lim = r;
    fqSegCandidates(s, x0, x1, a);
    float gx = a.gx;
    if (!(gx <= r)) return false;
    float2 pd = fqPointDist(s, x0);
    if (pd.y != 0.0 && pd.x <= nf(gx + FQ_TIE)) return false;
    pd = fqPointDist(s, x1);
    if (pd.y != 0.0 && pd.x <= nf(gx + FQ_TIE)) return false;
    a.pass = 1u;
    fqSegCandidates(s, x0, x1, a);
    float kf = float(a.k);
    w = float3(sdiv(a.w.x, kf), sdiv(a.w.y, kf), 0.0);
    q = sdiv3(a.q, kf);
    float3 cs = sdiv3(a.cs, kf);
    if (!(w.x > 0.0 && w.y > 0.0)) return false;
    float3 dv = ssub3(cs, q);
    float dl = fqLen(dv);
    g = gx;
    if (gx > FQ_PIERCE && dl > FQ_PIERCE)
    {
        n = sdiv3(dv, dl);
        return true;
    }
    n = fqSegOverlapAxis(s, x0, x1);
    fqAccDeep(a, n, 2u);
    fqSegCandidates(s, x0, x1, a);
    if (a.any == 0u) return true;
    fqAccDeep(a, n, 4u);
    fqSegCandidates(s, x0, x1, a);
    w = a.lw;
    q = a.lq;
    if (a.hw.y > a.lw.y)
    {
        two = true;
        w2 = a.hw;
        q2 = a.hq;
    }
    return true;
}

// --- Points ---

// point_nearest: the shape's point nearest x outside it (.w 1), else .w 0.
float4 fqPointNearest(FqShape s, float3 x)
{
    if (s.kind == FQ_BOX)
    {
        float3 d = ssub3(x, s.c);
        float3 l = float3(sdot(s.ax0, d), sdot(s.ax1, d), sdot(s.ax2, d));
        float3 h = s.half;
        if (l.x > -h.x && l.x < h.x && l.y > -h.y && l.y < h.y && l.z > -h.z && l.z < h.z) return float4(0.0, 0.0, 0.0, 0.0);
        float3 k = float3(min(max(l.x, -h.x), h.x), min(max(l.y, -h.y), h.y), min(max(l.z, -h.z), h.z));
        return float4(sadd3(sadd3(sadd3(s.c, sscale(s.ax0, k.x)), sscale(s.ax1, k.y)), sscale(s.ax2, k.z)), 1.0);
    }
    if (s.kind == FQ_SPH)
    {
        float3 d = ssub3(x, s.c);
        float l = fqLen(d);
        if (!(l > s.half.x)) return float4(0.0, 0.0, 0.0, 0.0);
        return float4(sadd3(s.c, sscale(d, sdiv(s.half.x, l))), 1.0);
    }
    if (s.kind == FQ_CAP)
    {
        float3 ab = ssub3(s.b, s.c);
        float l2 = sdot(ab, ab);
        float t = l2 > 0.0 ? fqClamp01(sdiv(sdot(ssub3(x, s.c), ab), l2)) : 0.0;
        float3 m = sadd3(s.c, sscale(ab, t));
        float3 d = ssub3(x, m);
        float l = fqLen(d);
        if (!(l > s.half.x)) return float4(0.0, 0.0, 0.0, 0.0);
        return float4(sadd3(m, sscale(d, sdiv(s.half.x, l))), 1.0);
    }
    if (s.kind == FQ_HULL)
    {
        float m = -3.0e38;
        float3 n = float3(0.0, 0.0, 0.0);
        for (uint f = 0u; f < FQ_HULL_FACES; f++)
        {
            if (f >= s.nf) break;
            float4 pl = fqHullP(s, f);
            float e = nf(sdot(pl.xyz, x) - pl.w);
            if (e > m)
            {
                m = e;
                n = pl.xyz;
            }
        }
        if (!(m >= 0.0)) return float4(0.0, 0.0, 0.0, 0.0);
        return float4(ssub3(x, sscale(n, m)), 1.0);
    }
    if (s.kind == FQ_TRI)
    {
        float3 w;
        return float4(fqClosestTri(x, s.c, s.b, s.ax0, w), 1.0);
    }
    if (fqPointDist(s, x).y == 0.0) return float4(0.0, 0.0, 0.0, 0.0);
    return float4(fqCylPoint(s.c, s.half.x, s.half.y, x), 1.0);
}

// point_contact: the band contact of a node at x (normal n, gap g).
bool fqPointContact(FqShape s, float3 x, float r, out float3 n, out float g)
{
    n = float3(0.0, 0.0, 0.0);
    g = 0.0;
    float4 p = fqPointNearest(s, x);
    if (p.w == 0.0) return false;
    float3 dv = ssub3(p.xyz, x);
    g = fqLen(dv);
    if (!(g <= r) || !(g > FQ_PIERCE)) return false;
    n = sdiv3(dv, g);
    return true;
}


// --- Casts (continuous contact for fast shapes; the twin's translated, distance, toi) ---

// The shape (unmoved) moved by d: every point plus d, a hull's world vertices plus d and
// plane offsets plus n·d (after its pose, as the twin's).
FqShape fqTranslate(FqShape s, float3 d)
{
    FqShape t = s;
    if (s.kind == FQ_HULL)
    {
        t.moved = 1u;
        t.off = d;
        return t;
    }
    t.c = sadd3(s.c, d);
    if (s.kind == FQ_CAP || s.kind == FQ_TRI) t.b = sadd3(s.b, d);
    if (s.kind == FQ_TRI) t.ax0 = sadd3(s.ax0, d);
    return t;
}

// The element's distance to the shape (0 when a vertex is inside): kind 0 a triangle
// x0 x1 x2, 1 a segment x0–x1, 2 a point x0.
float fqDistance(FqShape s, float3 x0, float3 x1, float3 x2, uint kind)
{
    FqAcc a = fqAccNew();
    if (kind == 0u) fqCandidates(s, x0, x1, x2, a);
    else if (kind == 1u) fqSegCandidates(s, x0, x1, a);
    float g = a.gx;
    uint nv = kind == 0u ? 3u : (kind == 1u ? 2u : 1u);
    for (uint k = 0u; k < 3u; k++)
    {
        if (k >= nv) break;
        float2 pd = fqPointDist(s, fqX(x0, x1, x2, k));
        if (pd.y == 0.0) return 0.0;
        g = min(g, pd.x);
    }
    return g;
}

static const uint FQ_CAST_ITERS = 24u;

// toi: the earliest fraction t of the relative motion d at which the element comes
// within r of the shape (conservative advancement), else −1.
float fqToi(FqShape s, float3 x0, float3 x1, float3 x2, uint kind, float3 d, float r)
{
    float len = fqLen(d);
    float t = 0.0;
    for (uint i = 0u; i < FQ_CAST_ITERS; i++)
    {
        float g = fqDistance(fqTranslate(s, sscale(d, t)), x0, x1, x2, kind);
        if (g <= r) return t;
        if (!(len > 0.0)) return -1.0;
        t = nf(t + sdiv(nf(g - r), len));
        if (t > 1.0) return -1.0;
    }
    return -1.0;
}

// --- Culls ---

// bounds: a world box holding the shape, 1 cm wide of it.
void fqBounds(FqShape s, out float3 lo, out float3 hi)
{
    if (s.kind == FQ_HULL || s.kind == FQ_TRI)
    {
        lo = float3(3.0e38, 3.0e38, 3.0e38);
        hi = -lo;
        uint n = s.kind == FQ_HULL ? s.nv : 3u;
        for (uint i = 0u; i < FQ_HULL_VERTS; i++)
        {
            if (i >= n) break;
            float3 v = s.kind == FQ_HULL ? fqHullV(s, i) : fqTriV(s, i);
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    else if (s.kind == FQ_BOX)
    {
        float3 e = sadd3(sadd3(sscale(abs(s.ax0), s.half.x), sscale(abs(s.ax1), s.half.y)), sscale(abs(s.ax2), s.half.z));
        lo = ssub3(s.c, e);
        hi = sadd3(s.c, e);
    }
    else if (s.kind == FQ_SPH)
    {
        float3 r3 = float3(s.half.x, s.half.x, s.half.x);
        lo = ssub3(s.c, r3);
        hi = sadd3(s.c, r3);
    }
    else if (s.kind == FQ_CAP)
    {
        float3 r3 = float3(s.half.x, s.half.x, s.half.x);
        lo = ssub3(min(s.c, s.b), r3);
        hi = sadd3(max(s.c, s.b), r3);
    }
    else
    {
        float3 e = float3(s.half.x, s.half.y, s.half.x);
        lo = ssub3(s.c, e);
        hi = sadd3(s.c, e);
    }
    float3 m = float3(0.01, 0.01, 0.01);
    lo = ssub3(lo, m);
    hi = sadd3(hi, m);
}

// near: a triangle's box padded by r overlapping lo..hi.
bool fqNear(float3 x0, float3 x1, float3 x2, float r, float3 lo, float3 hi)
{
    float3 tlo = ssub3(min(min(x0, x1), x2), float3(r, r, r));
    float3 thi = sadd3(max(max(x0, x1), x2), float3(r, r, r));
    return all(tlo <= hi) && all(thi >= lo);
}

// seg_near: a beam's box padded by r overlapping lo..hi.
bool fqSegNear(float3 x0, float3 x1, float r, float3 lo, float3 hi)
{
    float3 tlo = ssub3(min(x0, x1), float3(r, r, r));
    float3 thi = sadd3(max(x0, x1), float3(r, r, r));
    return all(tlo <= hi) && all(thi >= lo);
}

#endif
