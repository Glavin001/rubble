// The shape's own axes overlap_axis tests after the element's (the twin's order): a
// box's 3, a cylinder's y, a hull's face normals, a triangle's normal (.w 0 when it is
// degenerate: skipped). A capsule and a sphere have none.
static const uint FQ_OWN_AXES = 128u; // FQ_HULL_FACES, the most
uint fqOwnAxes(FqShape s)
{
    if (s.kind == FQ_BOX) return 3u;
    if (s.kind == FQ_CYL || s.kind == FQ_TRI) return 1u;
    if (s.kind == FQ_HULL) return min(s.nf, FQ_HULL_FACES);
    return 0u;
}
float4 fqOwnAxis(FqShape s, uint j)
{
    if (s.kind == FQ_BOX) return float4(fqAxis(s, j), 1.0);
    if (s.kind == FQ_CYL) return float4(0.0, 1.0, 0.0, 1.0);
    if (s.kind == FQ_HULL) return float4(fqHullP(s, j).xyz, 1.0);
    return fqTriNormal(s);
}

// overlap_axis: the axes in the twin's order through one fqAxisTest call site: the
// triangle's normal n (j = 0), the shape's own (1..), then each triangle edge crossed
// with each of the shape's edge directions.
float3 fqOverlapAxis(FqShape s, float3 x0, float3 x1, float3 x2, float3 n)
{
    float best = 0.0;
    float3 axis = n;
    bool first = true;
    uint own = fqOwnAxes(s);
    uint nd = min(fqDirCount(s), FQ_HULL_EDGES);
    uint count = 1u + own + 3u * nd;
    for (uint j = 0u; j < 1u + FQ_OWN_AXES + 3u * FQ_HULL_EDGES; j++)
    {
        if (j >= count) break;
        float3 u = n;
        if (j > own)
        {
            uint k = (j - 1u - own) / nd;
            float3 e = ssub3(fqX(x0, x1, x2, (k + 1u) % 3u), fqX(x0, x1, x2, k));
            float3 d = fqDir(s, (j - 1u - own) % nd);
            float3 c = scross(e, d);
            float l2 = sdot(c, c);
            if (!(l2 > smul(1e-12, smul(sdot(e, e), sdot(d, d))))) continue;
            u = sdiv3(c, ssqrt(l2));
        }
        else if (j > 0u)
        {
            float4 o = fqOwnAxis(s, j - 1u);
            if (o.w == 0.0) continue;
            u = o.xyz;
        }
        fqAxisTest(s, u, x0, x1, x2, best, axis, first);
    }
    return axis;
}

// triangle_contact: a hit (weights w, normal n, gap g, point q). The candidate walk
// runs up to four times (the least gap, the pairs at it, then for a deep contact the
// support along the overlap axis and the pairs at it): one loop, one fqCandidates
// call site, the twin's steps between the walks.
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
    float gx = 0.0;
    for (uint walk = 0u; walk < 4u; walk++)
    {
        fqCandidates(s, x0, x1, x2, a);
        if (walk == 0u)
        {
            gx = a.gx;
            if (!(gx <= r)) return false;
            for (uint k = 0u; k < 3u; k++)
            {
                float2 pd = fqPointDist(s, fqX(x0, x1, x2, k));
                if (pd.y != 0.0 && pd.x <= nf(gx + FQ_TIE)) return false;
            }
            a.pass = 1u;
        }
        else if (walk == 1u)
        {
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
        }
        else if (walk == 2u)
        {
            if (a.any == 0u) return true;
            fqAccDeep(a, n, 3u);
        }
        else
        {
            float kf = float(a.k);
            w = float3(sdiv(a.w.x, kf), sdiv(a.w.y, kf), sdiv(a.w.z, kf));
            q = sdiv3(a.q, kf);
        }
    }
    return true;
}

// --- Segments (the twin's segment_contact) ---

float fqSegParam(float3 p0, float3 d, float dd, float3 x) { return fqClamp01(sdiv(sdot(ssub3(x, p0), d), dd)); }

// edge_beam: shape edge e0–e1 against beam p0–p1 (d = p1 − p0, dd = d·d): the two
// lines' closest points unless parallel (i = 0), then the edge's two ends on the beam
// (1, 2), in the twin's order; one call site of each query.
void fqEdgeBeam(FqShape s, float3 e0, float3 e1, float3 p0, float3 p1, float3 d, float dd, inout FqAcc a)
{
    float3 de = ssub3(e1, e0);
    for (uint i = 0u; i < 3u; i++)
    {
        FqPair p;
        if (i == 0u)
        {
            float3 c = scross(d, de);
            if (!(sdot(c, c) > smul(1e-12, smul(dd, sdot(de, de))))) continue;
            float2 st = fqSegSeg(p0, p1, e0, e1);
            p.q = sadd3(p0, sscale(d, st.x));
            p.cs = sadd3(e0, sscale(de, st.y));
            p.w = float3(nf(1.0 - st.x), st.x, 0.0);
            p.d = fqLen(ssub3(p.q, p.cs));
        }
        else
        {
            float3 pe = i == 1u ? e0 : e1;
            float t = fqSegParam(p0, d, dd, pe);
            p.q = sadd3(p0, sscale(d, t));
            p.cs = pe;
            p.w = float3(nf(1.0 - t), t, 0.0);
            p.d = fqLen(ssub3(p.q, pe));
        }
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
    float3 tlo = min(p0, p1);
    float3 thi = max(p0, p1);
    bool culls = !fqWalkAxisOnly(s);
    uint ne = fqWalkEdges(s);
    for (uint e = 0u; e < FQ_WALK_EDGES; e++)
    {
        if (e >= ne) break;
        float3 e0;
        float3 e1;
        fqWalkEdge(s, e, e0, e1);
        if (culls && fqFar(e0, e1, tlo, thi, fqPrune(a))) continue;
        fqEdgeBeam(s, e0, e1, p0, p1, d, dd, a);
    }
    if (s.kind == FQ_TRI)
    {
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
    if (!fqWalkAxisOnly(s)) fqBeamInside(s, p0, p1, a);
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

// seg_overlap_axis: the axes in the twin's order through one fqSegAxisTest call site:
// the shape's own (fqOwnAxis), the beam crossed with each of the shape's edge
// directions, and, when none of those was a valid axis, a perpendicular of the beam.
float3 fqSegOverlapAxis(FqShape s, float3 x0, float3 x1)
{
    float3 d = ssub3(x1, x0);
    float best = 0.0;
    float3 axis = float3(0.0, 0.0, 0.0);
    bool first = true;
    uint own = fqOwnAxes(s);
    uint nd = min(fqDirCount(s), FQ_HULL_EDGES);
    uint count = own + nd + 1u;
    for (uint j = 0u; j < FQ_OWN_AXES + FQ_HULL_EDGES + 1u; j++)
    {
        if (j >= count) break;
        float3 u;
        if (j < own)
        {
            float4 o = fqOwnAxis(s, j);
            if (o.w == 0.0) continue;
            u = o.xyz;
        }
        else if (j < own + nd)
        {
            float3 e = fqDir(s, j - own);
            float3 c = scross(d, e);
            float l2 = sdot(c, c);
            if (!(l2 > smul(1e-12, smul(sdot(d, d), sdot(e, e))))) continue;
            u = sdiv3(c, ssqrt(l2));
        }
        else
        {
            if (!first) break;
            float3 ax = abs(d.x) <= max(abs(d.y), abs(d.z)) ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
            float3 c = scross(d, ax);
            u = sdiv3(c, fqLen(c));
        }
        fqSegAxisTest(s, u, x0, x1, best, axis, first);
    }
    return axis;
}

// segment_contact: a hit of beam x0–x1 (weights (w0, w1, 0), normal n, gap g, point q),
// and when `two` a second (w2, q2; the same normal and gap). The candidate walk runs up
// to four times, as fqContact's: one loop, one fqSegCandidates call site.
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
    float gx = 0.0;
    for (uint walk = 0u; walk < 4u; walk++)
    {
        fqSegCandidates(s, x0, x1, a);
        if (walk == 0u)
        {
            gx = a.gx;
            if (!(gx <= r)) return false;
            for (uint k = 0u; k < 2u; k++)
            {
                float2 pd = fqPointDist(s, k == 0u ? x0 : x1);
                if (pd.y != 0.0 && pd.x <= nf(gx + FQ_TIE)) return false;
            }
            a.pass = 1u;
        }
        else if (walk == 1u)
        {
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
        }
        else if (walk == 2u)
        {
            if (a.any == 0u) return true;
            fqAccDeep(a, n, 4u);
        }
        else
        {
            w = a.lw;
            q = a.lq;
            if (a.hw.y > a.lw.y)
            {
                two = true;
                w2 = a.hw;
                q2 = a.hq;
            }
        }
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
// The separation cull (lossless; engine_schema::feature_query::separated, the same
// planes and allowance): whether the element's points (a segment passes x1 twice) lie
// wholly outside one of the shape's outward face planes by more than r and the rounding
// allowance (1e-3 m and 1e-4 of the largest coordinate read). The plane then holds every
// point of the element farther than r from the shape, and the narrow phase (its gaps
// within 1e-5 of the largest coordinate and 1e-4 m) finds no contact. A box's six planes,
// a hull's, a cylinder's ends; for a sphere, a capsule and a cylinder's side the gap from
// the element's box to their core (fqApart). A turned shape's bounding box
// (fqNear) can hold a whole vehicle: a slab on a slope. Off with the includer's
// fqCullsOn (tests and measurement).
static const float FQ_SEP_ABS = 1e-3;
static const float FQ_SEP_REL = 1e-4;
bool fqOutside(float4 p, float3 x0, float3 x1, float3 x2, float r, float big)
{
    float thr = r + FQ_SEP_ABS + FQ_SEP_REL * max(big, abs(p.w));
    return dot(p.xyz, x0) - p.w > thr && dot(p.xyz, x1) - p.w > thr && dot(p.xyz, x2) - p.w > thr;
}
// A round shape's core (a sphere's or cylinder's centre, a capsule's segment) in lo..hi:
// the gap from the element's box to it, over the axes `mask` keeps, farther than the
// shape's radius, r and the allowance.
bool fqApart(float3 x0, float3 x1, float3 x2, float3 lo, float3 hi, float3 mask, float radius, float r, float big)
{
    float3 elo = min(min(x0, x1), x2);
    float3 ehi = max(max(x0, x1), x2);
    float3 gap = max(max(elo - hi, lo - ehi), float3(0.0, 0.0, 0.0)) * mask;
    float3 m = max(abs(lo), abs(hi));
    float thr = radius + r + FQ_SEP_ABS + FQ_SEP_REL * max(big, max(max(m.x, m.y), m.z));
    return dot(gap, gap) > thr * thr;
}
bool fqSeparated(FqShape s, float3 x0, float3 x1, float3 x2, float r)
{
    if (!fqCullsOn()) return false;
    float3 m = max(max(abs(x0), abs(x1)), abs(x2));
    float big = max(max(m.x, m.y), m.z);
    if (s.kind == FQ_BOX)
    {
        for (uint j = 0u; j < 6u; j++)
        {
            if (fqOutside(fqBoxPlane(s, j), x0, x1, x2, r, big)) return true;
        }
        return false;
    }
    if (s.kind == FQ_HULL)
    {
        for (uint f = 0u; f < FQ_HULL_FACES; f++)
        {
            if (f >= s.nf) break;
            if (fqOutside(fqHullP(s, f), x0, x1, x2, r, big)) return true;
        }
        return false;
    }
    if (s.kind == FQ_CYL)
    {
        return fqOutside(float4(0.0, -1.0, 0.0, s.half.y - s.c.y), x0, x1, x2, r, big)
            || fqOutside(float4(0.0, 1.0, 0.0, s.c.y + s.half.y), x0, x1, x2, r, big)
            || fqApart(x0, x1, x2, s.c, s.c, float3(1.0, 0.0, 1.0), s.half.x, r, big);
    }
    if (s.kind == FQ_SPH) return fqApart(x0, x1, x2, s.c, s.c, float3(1.0, 1.0, 1.0), s.half.x, r, big);
    if (s.kind == FQ_CAP) return fqApart(x0, x1, x2, min(s.c, s.b), max(s.c, s.b), float3(1.0, 1.0, 1.0), s.half.x, r, big);
    return false;
}

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

