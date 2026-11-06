// Surface contact (jbeam-semantics-spec.md §38; CPU reference
// engine_softbody::explicit_surface): a vehicle's collision triangles against its
// static box slots (axis-aligned boxes, oriented boxes, cylinders, capsules, spheres)
// by their surface, beside node contact. Included by explicit_beams.slang.
//
// Off (no vehicle of the upload has FLAG_SURFACE: gP.surfSplit 0, a uniform) nothing
// here runs. On, per step after self-collision and before the slide nodes, three
// phases with a barrier before each (every lane reaches them; a vehicle without the
// flag does no work between them). No workgroup memory (Metal's threadgroup slots
// are spent by the contact modules):
//   snap   per node its force at the pass's start (beam sum + accumulator) into gBeams
//          at surfSnap, its hit count at surfSplit cleared
//   find   per collision triangle the narrow phase against every slot near it; each
//          hit counts its nodes (weight > 0) at surfSplit
//   apply  the same hits recomputed (positions do not change), the two-body
//          dissipative response from the snapshot and the counts, into the
//          accumulators (fixed point, order free) and the position steps
// Every operation strict, in the CPU reference's order.

static const uint FLAG_SURFACE = 131072u;
static const float SURF_TIE = 1e-4;
static const float SURF_PIERCE = 1e-5;
static const uint SURF_BOX = 0u;
static const uint SURF_CYL = 2u;
static const uint SURF_CAP = 3u;
static const uint SURF_SPH = 4u;

struct SurfShape
{
    uint kind; // SURF_BOX (oriented or not), SURF_CYL, SURF_CAP, SURF_SPH
    float3 c;  // box, cylinder, sphere centre; capsule a
    float3 b;  // capsule b
    float3 ax0;
    float3 ax1;
    float3 ax2;
    float3 half; // box half extents; .x radius, .y half height (cylinder); .x radius (capsule, sphere)
};

// Candidate pair: weights and point on the triangle, point on the shape, distance.
struct SurfPair
{
    float3 w;
    float3 q;
    float3 cs;
    float d;
};

// The running reduction over candidates: pass 0 the nearest gap, pass 1 the sums
// of those within SURF_TIE of it; touching or crossing (explicit_surface.rs
// deepest), along the separating axis n pass 2 the deepest of those with shared
// weights, pass 3 the sums of those within SURF_TIE of it.
struct SurfAcc
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
    float3 lw; // pass 4 (a beam): the deepest at the least and the most parameter
    float3 lq;
    float3 hw;
    float3 hq;
};

bool surfShared(float3 w) { return (w.x > 0.0 ? 1u : 0u) + (w.y > 0.0 ? 1u : 0u) + (w.z > 0.0 ? 1u : 0u) >= 2u; }

void surfVisit(inout SurfAcc a, SurfPair p, float g)
{
    if (a.pass == 0u)
    {
        a.gx = min(a.gx, g);
        return;
    }
    if (!(g <= nf(a.gx + SURF_TIE))) return;
    if (a.pass == 2u)
    {
        if (!surfShared(p.w)) return;
        float d = sdot(a.n, p.q);
        a.top = a.any == 0u ? d : max(a.top, d);
        a.any = 1u;
        return;
    }
    if (a.pass >= 3u && !(surfShared(p.w) && sdot(a.n, p.q) >= nf(a.top - SURF_TIE))) return;
    if (a.pass == 4u)
    {
        // deepest_ends: the least and the most parameter along the beam (w.y).
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

SurfAcc surfAccNew()
{
    SurfAcc a;
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
    return a;
}

// Passes 2 and 3 set up (normal n): the sums cleared.
void surfAccDeep(inout SurfAcc a, float3 n, uint pass)
{
    a.pass = pass;
    a.n = n;
    a.w = float3(0.0, 0.0, 0.0);
    a.q = float3(0.0, 0.0, 0.0);
    a.cs = float3(0.0, 0.0, 0.0);
    a.k = 0u;
}

float surfLen(float3 v) { return ssqrt(sdot(v, v)); }
float surfClamp01(float x) { return min(max(x, 0.0), 1.0); }

// Closest point on triangle a b c to p (Ericson §5.1.5): explicit_surface.rs
// closest_on_triangle.
float3 surfClosestTri(float3 p, float3 a, float3 b, float3 c, out float3 w)
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
float2 surfSegSeg(float3 p1, float3 q1, float3 p2, float3 q2)
{
    float3 d1 = ssub3(q1, p1);
    float3 d2 = ssub3(q2, p2);
    float3 r = ssub3(p1, p2);
    float a = sdot(d1, d1);
    float e = sdot(d2, d2);
    float f = sdot(d2, r);
    if (a <= 1e-12 && e <= 1e-12) return float2(0.0, 0.0);
    if (a <= 1e-12) return float2(0.0, surfClamp01(sdiv(f, e)));
    float c = sdot(d1, r);
    if (e <= 1e-12) return float2(surfClamp01(sdiv(-c, a)), 0.0);
    float b = sdot(d1, d2);
    float denom = nf(smul(a, e) - smul(b, b));
    float s = denom > 1e-12 ? surfClamp01(sdiv(nf(smul(b, f) - smul(c, e)), denom)) : 0.0;
    float t = sdiv(nf(smul(b, s) + f), e);
    if (t < 0.0)
    {
        t = 0.0;
        s = surfClamp01(sdiv(-c, a));
    }
    else if (t > 1.0)
    {
        t = 1.0;
        s = surfClamp01(sdiv(nf(b - c), a));
    }
    return float2(s, t);
}

float3 surfEdgeW(uint k, float t)
{
    float3 w = float3(0.0, 0.0, 0.0);
    float u = nf(1.0 - t);
    if (k == 0u) w = float3(u, t, 0.0);
    else if (k == 1u) w = float3(0.0, u, t);
    else w = float3(t, 0.0, u);
    return w;
}

float3 surfTriX(float3 x0, float3 x1, float3 x2, uint k) { return k == 0u ? x0 : (k == 1u ? x1 : x2); }

// The point of the solid vertical cylinder nearest x (cylinder_point).
float3 surfCylPoint(float3 c, float r, float hh, float3 x)
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

// A pair as the shape takes it (the closure of seg_triangle): its gap.
float surfMap(SurfShape s, inout SurfPair p)
{
    if (s.kind == SURF_CAP) return nf(p.d - s.half.x);
    if (s.kind == SURF_CYL)
    {
        p.cs = surfCylPoint(s.c, s.half.x, s.half.y, p.q);
        p.d = surfLen(ssub3(p.q, p.cs));
    }
    return p.d;
}

// seg_triangle: segment p0–p1's candidate pairs against triangle x0 x1 x2.
void surfSegTri(SurfShape s, float3 p0, float3 p1, float3 x0, float3 x1, float3 x2, inout SurfAcc a)
{
    for (uint e = 0u; e < 2u; e++)
    {
        float3 pe = e == 0u ? p0 : p1;
        SurfPair p;
        p.q = surfClosestTri(pe, x0, x1, x2, p.w);
        p.cs = pe;
        p.d = surfLen(ssub3(p.q, pe));
        float g = surfMap(s, p);
        surfVisit(a, p, g);
    }
    for (uint k = 0u; k < 3u; k++)
    {
        float3 ea = surfTriX(x0, x1, x2, k);
        float3 eb = surfTriX(x0, x1, x2, (k + 1u) % 3u);
        float2 st = surfSegSeg(p0, p1, ea, eb);
        float3 pa = sadd3(p0, sscale(ssub3(p1, p0), st.x));
        float3 y = sadd3(ea, sscale(ssub3(eb, ea), st.y));
        SurfPair p;
        p.w = surfEdgeW(k, st.y);
        p.q = y;
        p.cs = pa;
        p.d = surfLen(ssub3(pa, y));
        float g = surfMap(s, p);
        surfVisit(a, p, g);
    }
    float3 n = scross(ssub3(x1, x0), ssub3(x2, x0));
    float d0 = sdot(ssub3(p0, x0), n);
    float d1 = sdot(ssub3(p1, x0), n);
    if ((d0 < 0.0 && d1 > 0.0) || (d0 > 0.0 && d1 < 0.0))
    {
        float3 c = sadd3(p0, sscale(ssub3(p1, p0), sdiv(d0, nf(d0 - d1))));
        SurfPair p;
        p.q = surfClosestTri(c, x0, x1, x2, p.w);
        float3 qc = ssub3(p.q, c);
        if (sdot(qc, qc) < 1e-10)
        {
            p.cs = c;
            p.d = 0.0;
            float g = surfMap(s, p);
            surfVisit(a, p, g);
        }
    }
}

float3 surfAxis(SurfShape s, uint i) { return i == 0u ? s.ax0 : (i == 1u ? s.ax1 : s.ax2); }
float surfHalf(SurfShape s, uint i) { return i == 0u ? s.half.x : (i == 1u ? s.half.y : s.half.z); }

// box_planes, plane j (−x, +x, −y, +y, −z, +z): (n, d), inside n·x ≤ d.
float4 surfBoxPlane(SurfShape s, uint j)
{
    uint i = j / 2u;
    float3 a = surfAxis(s, i);
    float ca = sdot(a, s.c);
    float h = surfHalf(s, i);
    if ((j & 1u) == 0u) return float4(-a, nf(h - ca));
    return float4(a, nf(ca + h));
}

// tri_edges_inside for a box.
void surfEdgesInBox(SurfShape s, float3 x0, float3 x1, float3 x2, inout SurfAcc a)
{
    for (uint k = 0u; k < 3u; k++)
    {
        float3 p0 = surfTriX(x0, x1, x2, k);
        float3 p1 = surfTriX(x0, x1, x2, (k + 1u) % 3u);
        float t0 = 0.0;
        float t1 = 1.0;
        bool out = false;
        for (uint j = 0u; j < 6u; j++)
        {
            float4 pl = surfBoxPlane(s, j);
            float s0 = nf(sdot(pl.xyz, p0) - pl.w);
            float s1 = nf(sdot(pl.xyz, p1) - pl.w);
            if (s0 >= 0.0 && s1 >= 0.0) { out = true; break; }
            if (s0 < 0.0 && s1 < 0.0) continue;
            float t = sdiv(s0, nf(s0 - s1));
            if (s0 >= 0.0) t0 = max(t0, t);
            else t1 = min(t1, t);
        }
        if (out || !(t0 < t1)) continue;
        for (uint e = 0u; e < 2u; e++)
        {
            float t = e == 0u ? t0 : t1;
            SurfPair p;
            p.w = surfEdgeW(k, t);
            p.q = sadd3(p0, sscale(ssub3(p1, p0), t));
            p.cs = p.q;
            p.d = 0.0;
            surfVisit(a, p, 0.0);
        }
    }
}

// clip_planes against box s: segment p0–p1's run strictly inside (t0 < t1).
bool surfClipBox(SurfShape s, float3 p0, float3 p1, out float t0, out float t1)
{
    t0 = 0.0;
    t1 = 1.0;
    for (uint j = 0u; j < 6u; j++)
    {
        float4 pl = surfBoxPlane(s, j);
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

// x_candidates.
void surfCandidates(SurfShape s, float3 x0, float3 x1, float3 x2, inout SurfAcc a)
{
    if (s.kind == SURF_SPH)
    {
        SurfPair p;
        p.q = surfClosestTri(s.c, x0, x1, x2, p.w);
        p.cs = s.c;
        p.d = surfLen(ssub3(s.c, p.q));
        surfVisit(a, p, nf(p.d - s.half.x));
        return;
    }
    if (s.kind == SURF_CAP)
    {
        surfSegTri(s, s.c, s.b, x0, x1, x2, a);
        return;
    }
    if (s.kind == SURF_CYL)
    {
        float3 h = float3(0.0, s.half.y, 0.0);
        surfSegTri(s, ssub3(s.c, h), sadd3(s.c, h), x0, x1, x2, a);
        return;
    }
    for (uint i = 0u; i < 3u; i++)
    {
        uint u = (i + 1u) % 3u;
        uint v = (i + 2u) % 3u;
        for (uint k = 0u; k < 4u; k++)
        {
            float su = (k & 1u) != 0u ? surfHalf(s, u) : -surfHalf(s, u);
            float sv = (k & 2u) != 0u ? surfHalf(s, v) : -surfHalf(s, v);
            float3 m = sadd3(sadd3(s.c, sscale(surfAxis(s, u), su)), sscale(surfAxis(s, v), sv));
            float3 e = sscale(surfAxis(s, i), surfHalf(s, i));
            surfSegTri(s, ssub3(m, e), sadd3(m, e), x0, x1, x2, a);
        }
    }
    surfEdgesInBox(s, x0, x1, x2, a);
}

// point_distance: x's distance to the shape when outside (.y 1), else .y 0.
float2 surfPointDist(SurfShape s, float3 x)
{
    if (s.kind == SURF_BOX)
    {
        float3 d = ssub3(x, s.c);
        float3 l = float3(sdot(s.ax0, d), sdot(s.ax1, d), sdot(s.ax2, d));
        float3 h = s.half;
        if (l.x > -h.x && l.x < h.x && l.y > -h.y && l.y < h.y && l.z > -h.z && l.z < h.z) return float2(0.0, 0.0);
        float3 k = float3(min(max(l.x, -h.x), h.x), min(max(l.y, -h.y), h.y), min(max(l.z, -h.z), h.z));
        return float2(surfLen(ssub3(l, k)), 1.0);
    }
    if (s.kind == SURF_SPH)
    {
        float d = nf(surfLen(ssub3(x, s.c)) - s.half.x);
        return float2(d, d > 0.0 ? 1.0 : 0.0);
    }
    if (s.kind == SURF_CAP)
    {
        float3 ab = ssub3(s.b, s.c);
        float l2 = sdot(ab, ab);
        float t = l2 > 0.0 ? surfClamp01(sdiv(sdot(ssub3(x, s.c), ab), l2)) : 0.0;
        float d = nf(surfLen(ssub3(x, sadd3(s.c, sscale(ab, t)))) - s.half.x);
        return float2(d, d > 0.0 ? 1.0 : 0.0);
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
float2 surfProject(SurfShape s, float3 u)
{
    float cu;
    float e;
    if (s.kind == SURF_BOX)
    {
        cu = sdot(u, s.c);
        e = nf(nf(smul(abs(sdot(u, s.ax0)), s.half.x) + smul(abs(sdot(u, s.ax1)), s.half.y))
               + smul(abs(sdot(u, s.ax2)), s.half.z));
    }
    else if (s.kind == SURF_SPH)
    {
        cu = sdot(u, s.c);
        e = s.half.x;
    }
    else if (s.kind == SURF_CAP)
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
void surfAxisTest(SurfShape s, float3 u, float3 x0, float3 x1, float3 x2, inout float best, inout float3 axis,
                  inout bool first)
{
    float t0 = sdot(u, x0);
    float t1 = sdot(u, x1);
    float t2 = sdot(u, x2);
    float tlo = min(min(t0, t1), t2);
    float thi = max(max(t0, t1), t2);
    float2 sp = surfProject(s, u);
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

uint surfDirCount(SurfShape s) { return s.kind == SURF_BOX ? 3u : (s.kind == SURF_SPH ? 0u : 1u); }
float3 surfDir(SurfShape s, uint j)
{
    if (s.kind == SURF_BOX) return surfAxis(s, j);
    if (s.kind == SURF_CYL) return float3(0.0, 1.0, 0.0);
    return ssub3(s.b, s.c);
}

// overlap_axis.
float3 surfOverlapAxis(SurfShape s, float3 x0, float3 x1, float3 x2, float3 n)
{
    float best = 0.0;
    float3 axis = n;
    bool first = true;
    surfAxisTest(s, n, x0, x1, x2, best, axis, first);
    if (s.kind == SURF_BOX)
    {
        surfAxisTest(s, s.ax0, x0, x1, x2, best, axis, first);
        surfAxisTest(s, s.ax1, x0, x1, x2, best, axis, first);
        surfAxisTest(s, s.ax2, x0, x1, x2, best, axis, first);
    }
    else if (s.kind == SURF_CYL)
        surfAxisTest(s, float3(0.0, 1.0, 0.0), x0, x1, x2, best, axis, first);
    uint nd = surfDirCount(s);
    for (uint k = 0u; k < 3u; k++)
    {
        float3 e = ssub3(surfTriX(x0, x1, x2, (k + 1u) % 3u), surfTriX(x0, x1, x2, k));
        for (uint j = 0u; j < 3u; j++)
        {
            if (j >= nd) break;
            float3 d = surfDir(s, j);
            float3 c = scross(e, d);
            float l2 = sdot(c, c);
            if (!(l2 > smul(1e-12, smul(sdot(e, e), sdot(d, d))))) continue;
            surfAxisTest(s, sdiv3(c, ssqrt(l2)), x0, x1, x2, best, axis, first);
        }
    }
    return axis;
}

// triangle_contact: .w 1 a hit (weights w, normal n, gap g, point q), else 0.
bool surfContact(SurfShape s, float3 x0, float3 x1, float3 x2, float r, out float3 w, out float3 n, out float g,
                 out float3 q)
{
    w = float3(0.0, 0.0, 0.0);
    n = float3(0.0, 0.0, 0.0);
    g = 0.0;
    q = float3(0.0, 0.0, 0.0);
    float3 nc = scross(ssub3(x1, x0), ssub3(x2, x0));
    float nl = surfLen(nc);
    if (!(nl > 1e-8)) return false;
    float3 tn = sdiv3(nc, nl);
    SurfAcc a = surfAccNew();
    surfCandidates(s, x0, x1, x2, a);
    float gx = a.gx;
    if (!(gx <= r)) return false;
    for (uint k = 0u; k < 3u; k++)
    {
        float2 pd = surfPointDist(s, surfTriX(x0, x1, x2, k));
        if (pd.y != 0.0 && pd.x <= nf(gx + SURF_TIE)) return false;
    }
    a.pass = 1u;
    surfCandidates(s, x0, x1, x2, a);
    float kf = float(a.k);
    w = float3(sdiv(a.w.x, kf), sdiv(a.w.y, kf), sdiv(a.w.z, kf));
    q = sdiv3(a.q, kf);
    float3 cs = sdiv3(a.cs, kf);
    uint nz = (w.x > 0.0 ? 1u : 0u) + (w.y > 0.0 ? 1u : 0u) + (w.z > 0.0 ? 1u : 0u);
    if (nz <= 1u) return false;
    float3 dv = ssub3(cs, q);
    float dl = surfLen(dv);
    g = gx;
    if (gx > SURF_PIERCE && dl > SURF_PIERCE)
    {
        n = sdiv3(dv, dl);
        return true;
    }
    n = surfOverlapAxis(s, x0, x1, x2, tn);
    // The deepest point along n (deepest).
    surfAccDeep(a, n, 2u);
    surfCandidates(s, x0, x1, x2, a);
    if (a.any == 0u) return true;
    surfAccDeep(a, n, 3u);
    surfCandidates(s, x0, x1, x2, a);
    kf = float(a.k);
    w = float3(sdiv(a.w.x, kf), sdiv(a.w.y, kf), sdiv(a.w.z, kf));
    q = sdiv3(a.q, kf);
    return true;
}

// --- Beam-edge contact: the segments (explicit_surface.rs segment_contact) ---

float surfSegParam(float3 p0, float3 d, float dd, float3 x) { return surfClamp01(sdiv(sdot(ssub3(x, p0), d), dd)); }

// edge_beam: shape edge e0–e1 against beam p0–p1 (d = p1 − p0, dd = d·d).
void surfEdgeBeam(SurfShape s, float3 e0, float3 e1, float3 p0, float3 p1, float3 d, float dd, inout SurfAcc a)
{
    float3 de = ssub3(e1, e0);
    float3 c = scross(d, de);
    if (sdot(c, c) > smul(1e-12, smul(dd, sdot(de, de))))
    {
        float2 st = surfSegSeg(p0, p1, e0, e1);
        SurfPair p;
        p.q = sadd3(p0, sscale(d, st.x));
        p.cs = sadd3(e0, sscale(de, st.y));
        p.w = float3(nf(1.0 - st.x), st.x, 0.0);
        p.d = surfLen(ssub3(p.q, p.cs));
        float g = surfMap(s, p);
        surfVisit(a, p, g);
    }
    for (uint e = 0u; e < 2u; e++)
    {
        float3 pe = e == 0u ? e0 : e1;
        float t = surfSegParam(p0, d, dd, pe);
        SurfPair p;
        p.q = sadd3(p0, sscale(d, t));
        p.cs = pe;
        p.w = float3(nf(1.0 - t), t, 0.0);
        p.d = surfLen(ssub3(p.q, pe));
        float g = surfMap(s, p);
        surfVisit(a, p, g);
    }
}

// seg_candidates.
void surfSegCandidates(SurfShape s, float3 p0, float3 p1, inout SurfAcc a)
{
    float3 d = ssub3(p1, p0);
    float dd = sdot(d, d);
    if (s.kind == SURF_SPH)
    {
        float t = surfSegParam(p0, d, dd, s.c);
        SurfPair p;
        p.q = sadd3(p0, sscale(d, t));
        p.cs = s.c;
        p.w = float3(nf(1.0 - t), t, 0.0);
        p.d = surfLen(ssub3(s.c, p.q));
        surfVisit(a, p, nf(p.d - s.half.x));
        return;
    }
    if (s.kind == SURF_CAP)
    {
        surfEdgeBeam(s, s.c, s.b, p0, p1, d, dd, a);
        return;
    }
    if (s.kind == SURF_CYL)
    {
        float3 h = float3(0.0, s.half.y, 0.0);
        surfEdgeBeam(s, ssub3(s.c, h), sadd3(s.c, h), p0, p1, d, dd, a);
        return;
    }
    for (uint i = 0u; i < 3u; i++)
    {
        uint u = (i + 1u) % 3u;
        uint v = (i + 2u) % 3u;
        for (uint k = 0u; k < 4u; k++)
        {
            float su = (k & 1u) != 0u ? surfHalf(s, u) : -surfHalf(s, u);
            float sv = (k & 2u) != 0u ? surfHalf(s, v) : -surfHalf(s, v);
            float3 m = sadd3(sadd3(s.c, sscale(surfAxis(s, u), su)), sscale(surfAxis(s, v), sv));
            float3 e = sscale(surfAxis(s, i), surfHalf(s, i));
            surfEdgeBeam(s, ssub3(m, e), sadd3(m, e), p0, p1, d, dd, a);
        }
    }
    float t0;
    float t1;
    if (surfClipBox(s, p0, p1, t0, t1))
    {
        for (uint e = 0u; e < 2u; e++)
        {
            float t = e == 0u ? t0 : t1;
            SurfPair p;
            p.q = sadd3(p0, sscale(ssub3(p1, p0), t));
            p.cs = p.q;
            p.w = float3(nf(1.0 - t), t, 0.0);
            p.d = 0.0;
            surfVisit(a, p, 0.0);
        }
    }
}

// One axis of seg_overlap_axis.
void surfSegAxisTest(SurfShape s, float3 u, float3 x0, float3 x1, inout float best, inout float3 axis, inout bool first)
{
    float t0 = sdot(u, x0);
    float t1 = sdot(u, x1);
    float tlo = min(t0, t1);
    float thi = max(t0, t1);
    float2 sp = surfProject(s, u);
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
float3 surfSegOverlapAxis(SurfShape s, float3 x0, float3 x1)
{
    float3 d = ssub3(x1, x0);
    float best = 0.0;
    float3 axis = float3(0.0, 0.0, 0.0);
    bool first = true;
    if (s.kind == SURF_BOX)
    {
        surfSegAxisTest(s, s.ax0, x0, x1, best, axis, first);
        surfSegAxisTest(s, s.ax1, x0, x1, best, axis, first);
        surfSegAxisTest(s, s.ax2, x0, x1, best, axis, first);
    }
    else if (s.kind == SURF_CYL)
        surfSegAxisTest(s, float3(0.0, 1.0, 0.0), x0, x1, best, axis, first);
    uint nd = surfDirCount(s);
    for (uint j = 0u; j < 3u; j++)
    {
        if (j >= nd) break;
        float3 e = surfDir(s, j);
        float3 c = scross(d, e);
        float l2 = sdot(c, c);
        if (!(l2 > smul(1e-12, smul(sdot(d, d), sdot(e, e))))) continue;
        surfSegAxisTest(s, sdiv3(c, ssqrt(l2)), x0, x1, best, axis, first);
    }
    if (first)
    {
        float3 ax = abs(d.x) <= max(abs(d.y), abs(d.z)) ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
        float3 c = scross(d, ax);
        surfSegAxisTest(s, sdiv3(c, surfLen(c)), x0, x1, best, axis, first);
    }
    return axis;
}

// segment_contact: a hit of beam x0–x1 (weights (w0, w1, 0), normal n, gap g, point q),
// and when `two` a second (w2, q2; the same normal and gap).
bool surfSegContact(SurfShape s, float3 x0, float3 x1, float r, out float3 w, out float3 n, out float g, out float3 q,
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
    SurfAcc a = surfAccNew();
    surfSegCandidates(s, x0, x1, a);
    float gx = a.gx;
    if (!(gx <= r)) return false;
    float2 pd = surfPointDist(s, x0);
    if (pd.y != 0.0 && pd.x <= nf(gx + SURF_TIE)) return false;
    pd = surfPointDist(s, x1);
    if (pd.y != 0.0 && pd.x <= nf(gx + SURF_TIE)) return false;
    a.pass = 1u;
    surfSegCandidates(s, x0, x1, a);
    float kf = float(a.k);
    w = float3(sdiv(a.w.x, kf), sdiv(a.w.y, kf), 0.0);
    q = sdiv3(a.q, kf);
    float3 cs = sdiv3(a.cs, kf);
    if (!(w.x > 0.0 && w.y > 0.0)) return false;
    float3 dv = ssub3(cs, q);
    float dl = surfLen(dv);
    g = gx;
    if (gx > SURF_PIERCE && dl > SURF_PIERCE)
    {
        n = sdiv3(dv, dl);
        return true;
    }
    n = surfSegOverlapAxis(s, x0, x1);
    surfAccDeep(a, n, 2u);
    surfSegCandidates(s, x0, x1, a);
    if (a.any == 0u) return true;
    surfAccDeep(a, n, 4u);
    surfSegCandidates(s, x0, x1, a);
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

// A beam's box padded by r overlapping lo..hi (seg_near).
bool surfSegNear(float3 x0, float3 x1, float r, float3 lo, float3 hi)
{
    float3 tlo = ssub3(min(x0, x1), float3(r, r, r));
    float3 thi = sadd3(max(x0, x1), float3(r, r, r));
    return all(tlo <= hi) && all(thi >= lo);
}

// Box slot b of a vehicle (boxBase records: (lo, kind), (hi, at)) as a shape, and its
// cull box.
SurfShape surfSlot(uint boxOff, uint b, out float3 lo, out float3 hi)
{
    uint4 L = gStatic[gP.boxBase + (boxOff + b) * 2u];
    uint4 H = gStatic[gP.boxBase + (boxOff + b) * 2u + 1u];
    lo = asfloat(L.xyz);
    hi = asfloat(H.xyz);
    SurfShape s;
    s.kind = SURF_BOX;
    s.c = sscale(sadd3(lo, hi), 0.5);
    s.b = float3(0.0, 0.0, 0.0);
    s.ax0 = float3(1.0, 0.0, 0.0);
    s.ax1 = float3(0.0, 1.0, 0.0);
    s.ax2 = float3(0.0, 0.0, 1.0);
    s.half = sscale(ssub3(hi, lo), 0.5);
    if (L.w == 0u) return s;
    uint at = gP.boxBase + H.w;
    float4 p0 = asfloat(gStatic[at]);
    float4 p1 = asfloat(gStatic[at + 1u]);
    if (L.w == SHAPE_OBOX)
    {
        float4 p2 = asfloat(gStatic[at + 2u]);
        float4 p3 = asfloat(gStatic[at + 3u]);
        s.c = p0.xyz;
        s.ax0 = p1.xyz;
        s.ax1 = p2.xyz;
        s.ax2 = p3.xyz;
        s.half = float3(p0.w, p1.w, p2.w);
    }
    else if (L.w == SHAPE_CYLINDER)
    {
        s.kind = SURF_CYL;
        s.c = p0.xyz;
        s.half = float3(p0.w, p1.x, 0.0);
    }
    else if (L.w == SHAPE_CAPSULE)
    {
        s.kind = SURF_CAP;
        s.c = p0.xyz;
        s.b = p1.xyz;
        s.half = float3(p0.w, 0.0, 0.0);
    }
    else
    {
        s.kind = SURF_SPH;
        s.c = p0.xyz;
        s.half = float3(p0.w, 0.0, 0.0);
    }
    return s;
}

// A triangle's box padded by r overlapping lo..hi (near).
bool surfNear(float3 x0, float3 x1, float3 x2, float r, float3 lo, float3 hi)
{
    float3 tlo = ssub3(min(min(x0, x1), x2), float3(r, r, r));
    float3 thi = sadd3(max(max(x0, x1), x2), float3(r, r, r));
    return all(tlo <= hi) && all(thi >= lo);
}

float surfInvMass(uint x)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return 0.0;
    return sdiv(float(uint(gAcc[gP.surfSplit + x].load())), gNodes[x * 3u].w);
}

float3 surfAcc(uint x)
{
    if ((gStatic[gP.nodeInfoBase + x].x & NODE_IMMOVABLE) != 0u) return float3(0.0, 0.0, 0.0);
    return sdiv3(gBeams[gP.surfSnap + x].xyz, gNodes[x * 3u].w);
}

float surfTerm(float w, float im) { return w > 0.0 ? smul(smul(w, w), im) : 0.0; }

// apply_surface_hits for one hit on triangle tn (global nodes) of vehicle v.
void surfApply(uint v, uint4 tn, float3 w, float3 nn, float gap, float r)
{
    float im0 = surfInvMass(tn.x);
    float im1 = surfInvMass(tn.y);
    float im2 = surfInvMass(tn.z);
    float imt = nf(nf(surfTerm(w.x, im0) + surfTerm(w.y, im1)) + surfTerm(w.z, im2));
    float3 vt = sadd3(sadd3(sscale(gNodes[tn.x * 3u + 1u].xyz, w.x), sscale(gNodes[tn.y * 3u + 1u].xyz, w.y)),
                      sscale(gNodes[tn.z * 3u + 1u].xyz, w.z));
    float3 at = sadd3(sadd3(sscale(surfAcc(tn.x), w.x), sscale(surfAcc(tn.y), w.y)), sscale(surfAcc(tn.z), w.z));
    float inv = nf(0.0 + imt);
    float m = inv > 0.0 ? sdiv(1.0, inv) : 0.0;
    float3 fh = sscale(ssub3(float3(0.0, 0.0, 0.0), at), m);
    float fr = nf(nf(smul(gNodes[tn.x * 3u + 1u].w, w.x) + smul(gNodes[tn.y * 3u + 1u].w, w.y))
                  + smul(gNodes[tn.z * 3u + 1u].w, w.z));
    float p = min(nf(r - gap), r);
    Surface surf = surfaceV(boxSurface(), v);
    surf.dissipative = 1u;
    float budget;
    float3 fc = contactResponse(fh, ssub3(float3(0.0, 0.0, 0.0), vt), m, nn, surf, p, fr, budget);
    if (all(fc == float3(0.0, 0.0, 0.0))) return;
    float delta = shiftDepth(p, budget, sdot(fh, nn));
    if (delta > 0.0)
    {
        float qd = smul(delta, m);
        if (w.x > 0.0) shiftAdd(tn.x, sscale(nn, -smul(w.x, smul(qd, im0))));
        if (w.y > 0.0) shiftAdd(tn.y, sscale(nn, -smul(w.y, smul(qd, im1))));
        if (w.z > 0.0) shiftAdd(tn.z, sscale(nn, -smul(w.z, smul(qd, im2))));
    }
    accAdd(tn.x, -sscale(fc, w.x));
    accAdd(tn.y, -sscale(fc, w.y));
    accAdd(tn.z, -sscale(fc, w.z));
    cfAdd(tn.x, sscale(fc, w.x));
    cfAdd(tn.y, sscale(fc, w.y));
    cfAdd(tn.z, sscale(fc, w.z));
}

// Triangle gt (of vehicle v) against every box slot: apply 0 counts its hits, 1
// applies them.
void surfTriangle(uint v, uint gt, float3 origin, float r, uint boxOff, uint boxCount, bool apply)
{
    if (gP.triBreak != 0u && triangleDead(gt)) return;
    if ((gMisc[gP.vehicles + gt].z & TRI_ANTI_ONLY) != 0u) return;
    uint4 tn = gStatic[gP.triBase + gt];
    float3 x0 = nf3(origin + gNodes[tn.x * 3u].xyz);
    float3 x1 = nf3(origin + gNodes[tn.y * 3u].xyz);
    float3 x2 = nf3(origin + gNodes[tn.z * 3u].xyz);
    for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
    {
        float3 lo;
        float3 hi;
        SurfShape s = surfSlot(boxOff, b, lo, hi);
        if (!surfNear(x0, x1, x2, r, lo, hi)) continue;
        float3 w;
        float3 nn;
        float g;
        float3 q;
        if (!surfContact(s, x0, x1, x2, r, w, nn, g, q)) continue;
        if (!apply)
        {
            if (w.x > 0.0) gAcc[gP.surfSplit + tn.x].add(1);
            if (w.y > 0.0) gAcc[gP.surfSplit + tn.y].add(1);
            if (w.z > 0.0) gAcc[gP.surfSplit + tn.z].add(1);
        }
        else
            surfApply(v, tn, w, nn, g, r);
    }
}

// Beam j is broken as the CPU sees it after this step's beam pass: its flag, or a
// detacher-group break this step (the next node phase marks those). (The slide
// nodes' brokenNow, which only the EB_AERO modules have.)
bool surfBroken(uint j, uint groupOff, uint parity)
{
    if (gBeams[j * 2u + 1u].z != 0.0) return true;
    int g = asint(gStatic[gP.beamBase + j * 3u].w);
    if (g == 0) return false;
    return gGroups[groupSlot(parity, groupOff, uint(abs(g)))].load() != NO_GROUP;
}

// Beam gj of vehicle v (a segment: beam-edge contact) against every box slot: apply
// 0 counts its hits, 1 applies them (its nodes (a, b, a), w.z 0).
void surfSegment(uint v, uint gj, float3 origin, float r, uint boxOff, uint boxCount, uint groupOff, uint parity,
                 bool apply)
{
    if (surfBroken(gj, groupOff, parity)) return;
    uint4 bn = gStatic[gP.beamBase + gj * 3u];
    float3 x0 = nf3(origin + gNodes[bn.x * 3u].xyz);
    float3 x1 = nf3(origin + gNodes[bn.y * 3u].xyz);
    for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
    {
        float3 lo;
        float3 hi;
        SurfShape s = surfSlot(boxOff, b, lo, hi);
        if (!surfSegNear(x0, x1, r, lo, hi)) continue;
        float3 w;
        float3 nn;
        float g;
        float3 q;
        bool two;
        float3 w2;
        float3 q2;
        if (!surfSegContact(s, x0, x1, r, w, nn, g, q, two, w2, q2)) continue;
        uint hits = two ? 2u : 1u;
        if (!apply)
        {
            gAcc[gP.surfSplit + bn.x].add(int(hits));
            gAcc[gP.surfSplit + bn.y].add(int(hits));
        }
        else
        {
            surfApply(v, uint4(bn.x, bn.y, bn.x, 0u), w, nn, g, r);
            if (two) surfApply(v, uint4(bn.x, bn.y, bn.x, 0u), w2, nn, g, r);
        }
    }
}

// point_nearest: the shape's point nearest x outside it (.w 1), else .w 0.
float4 surfPointNearest(SurfShape s, float3 x)
{
    if (s.kind == SURF_BOX)
    {
        float3 d = ssub3(x, s.c);
        float3 l = float3(sdot(s.ax0, d), sdot(s.ax1, d), sdot(s.ax2, d));
        float3 h = s.half;
        if (l.x > -h.x && l.x < h.x && l.y > -h.y && l.y < h.y && l.z > -h.z && l.z < h.z) return float4(0.0, 0.0, 0.0, 0.0);
        float3 k = float3(min(max(l.x, -h.x), h.x), min(max(l.y, -h.y), h.y), min(max(l.z, -h.z), h.z));
        return float4(sadd3(sadd3(sadd3(s.c, sscale(s.ax0, k.x)), sscale(s.ax1, k.y)), sscale(s.ax2, k.z)), 1.0);
    }
    if (s.kind == SURF_SPH)
    {
        float3 d = ssub3(x, s.c);
        float l = surfLen(d);
        if (!(l > s.half.x)) return float4(0.0, 0.0, 0.0, 0.0);
        return float4(sadd3(s.c, sscale(d, sdiv(s.half.x, l))), 1.0);
    }
    if (s.kind == SURF_CAP)
    {
        float3 ab = ssub3(s.b, s.c);
        float l2 = sdot(ab, ab);
        float t = l2 > 0.0 ? surfClamp01(sdiv(sdot(ssub3(x, s.c), ab), l2)) : 0.0;
        float3 m = sadd3(s.c, sscale(ab, t));
        float3 d = ssub3(x, m);
        float l = surfLen(d);
        if (!(l > s.half.x)) return float4(0.0, 0.0, 0.0, 0.0);
        return float4(sadd3(m, sscale(d, sdiv(s.half.x, l))), 1.0);
    }
    if (surfPointDist(s, x).y == 0.0) return float4(0.0, 0.0, 0.0, 0.0);
    return float4(surfCylPoint(s.c, s.half.x, s.half.y, x), 1.0);
}

// point_contact: the band contact of a node at x (normal n, gap g).
bool surfPointContact(SurfShape s, float3 x, float r, out float3 n, out float g)
{
    n = float3(0.0, 0.0, 0.0);
    g = 0.0;
    float4 p = surfPointNearest(s, x);
    if (p.w == 0.0) return false;
    float3 dv = ssub3(p.xyz, x);
    g = surfLen(dv);
    if (!(g <= r) || !(g > SURF_PIERCE)) return false;
    n = sdiv3(dv, g);
    return true;
}

// Node gi of vehicle v (point contact in the band) against every box slot: apply 0
// counts its hits, 1 applies them.
void surfPoint(uint v, uint gi, float3 origin, float r, uint boxOff, uint boxCount, bool apply)
{
    if ((gStatic[gP.nodeInfoBase + gi].x & NODE_NO_GROUND) != 0u) return;
    float3 x = nf3(origin + gNodes[gi * 3u].xyz);
    for (uint b = 0u, bStop = EB_END(b, boxCount, 1u); b < bStop; b++)
    {
        float3 lo;
        float3 hi;
        SurfShape s = surfSlot(boxOff, b, lo, hi);
        if (!surfSegNear(x, x, r, lo, hi)) continue;
        float3 nn;
        float g;
        if (!surfPointContact(s, x, r, nn, g)) continue;
        if (!apply) gAcc[gP.surfSplit + gi].add(1);
        else surfApply(v, uint4(gi, gi, gi, 0u), float3(1.0, 0.0, 0.0), nn, g, r);
    }
}

// Whether any vehicle of the upload has surface contact (uniform: a parameter).
uint surfaceOn() { return gP.surfSplit != 0u ? 1u : 0u; }

// Step's surface pass of vehicle v (uniform `on`), after its self-collision and
// before its slide nodes (parity: the step's): the barrier, the snapshot, the find,
// the apply. A vehicle without the flag reaches the barriers only. The caller's
// barrier follows.
void surfaceStep(uint v, uint lane, uint on, uint parity)
{
    if (on == 0u) return;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    bool mine = (r3.x & FLAG_SURFACE) != 0u;
    uint nodeCount = mine ? r0.y : 0u;
    uint triCount = mine ? r1.y : 0u;
    float3 origin = asfloat(gMisc[v].xyz);
    float r = asfloat(r3.z);
    uint4 sr = gStatic[gP.surfSeg + v];
    uint segCount = mine ? sr.y : 0u;
    sync();
    for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
    {
        uint gi = r0.x + i;
        float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + gi], r0.z, r2.z, r2.w, parity, false);
        gBeams[gP.surfSnap + gi] = float4(sadd3(f, accForce(gi)), 0.0);
        gAcc[gP.surfSplit + gi].store(0);
    }
    sync();
    for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES)
        surfTriangle(v, r1.x + t, origin, r, r1.z, r1.w, false);
    for (uint j = lane, jStop = EB_END(j, segCount, LANES); j < jStop; j += LANES)
        surfSegment(v, gLists[sr.x + j], origin, r, r1.z, r1.w, r2.z, parity, false);
    for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
        surfPoint(v, r0.x + i, origin, r, r1.z, r1.w, false);
    sync();
    for (uint t = lane, tStop = EB_END(t, triCount, LANES); t < tStop; t += LANES)
        surfTriangle(v, r1.x + t, origin, r, r1.z, r1.w, true);
    for (uint j = lane, jStop = EB_END(j, segCount, LANES); j < jStop; j += LANES)
        surfSegment(v, gLists[sr.x + j], origin, r, r1.z, r1.w, r2.z, parity, true);
    for (uint i = lane, iStop = EB_END(i, nodeCount, LANES); i < iStop; i += LANES)
        surfPoint(v, r0.x + i, origin, r, r1.z, r1.w, true);
    if (lane == 0u && mine) gsHit = 1u;
}
