// ---- casts (translation along a unit direction) -------------------------------
// Results: t along the direction to first contact and the terrain-side normal
// at the contact, pointing toward the moving shape. A start already touching
// or inside reports t = 0 (the callers' "entry behind" convention: t <= 0).

// Distance between segment p0p1 and triangle abc (Ericson 5.1.10): an end
// point against the face, or the segment against an edge, unless the segment
// crosses the triangle (0). qs on the segment, qt on the triangle.
float hfSegTriDistance(float3 p0, float3 p1, float3 a, float3 b, float3 c, out float3 qs, out float3 qt)
{
    float3 nt = cross(b - a, c - a);
    float d0 = dot(p0 - a, nt), d1 = dot(p1 - a, nt);
    if (d0 * d1 < 0.0)
    {
        float3 x = p0 + (p1 - p0) * (d0 / (d0 - d1));
        if (dot(cross(b - a, x - a), nt) >= 0.0 && dot(cross(c - b, x - b), nt) >= 0.0 && dot(cross(a - c, x - c), nt) >= 0.0)
        {
            qs = x; qt = x;
            return 0.0;
        }
    }
    qs = p0; qt = hfClosestOnTriangle(p0, a, b, c);
    float best = dot(qs - qt, qs - qt);
    float3 q = hfClosestOnTriangle(p1, a, b, c);
    float d = dot(p1 - q, p1 - q);
    if (d < best) { best = d; qs = p1; qt = q; }
    for (uint e = 0u; e < 3u; e++)
    {
        float3 u = e == 0u ? a : e == 1u ? b : c;
        float3 v = e == 0u ? b : e == 1u ? c : a;
        float s, t;
        hfClosestSegSeg(p0, p1, u, v, s, t);
        float3 x = p0 + (p1 - p0) * s, y = u + (v - u) * t;
        d = dot(x - y, x - y);
        if (d < best) { best = d; qs = x; qt = y; }
    }
    return sqrt(best);
}

// Signed clearance between a capsule's axis p0p1 (local; p0 = p1 for a ball)
// and the surface, with the outward contact normal. Overlapping: the deepest
// point's depth along its triangle's normal (negative). Along the axis both its
// height and the surface's are linear between the points where its x/z track
// crosses a grid line or a cell diagonal, so that depth is found at one of those
// points or an end. Apart: the exact distance to the nearest triangle within
// `reach` (`reach` when none is).
float hfCapsuleClearance(HeightFieldRef hf, float3 p0, float3 p1, float reach, out float3 n)
{
    n = float3(0, 1, 0);
    float deepest = 0.0;
    float fx0 = (p0.x - hf.x0) / hf.dx, fx1 = (p1.x - hf.x0) / hf.dx;
    float fz0 = (p0.z - hf.z0) / hf.dz, fz1 = (p1.z - hf.z0) / hf.dz;
    for (uint family = 0u; family < 4u; family++)
    {
        float a = family == 0u ? fx0 : family == 1u ? fz0 : fx0 - fz0;
        float b = family == 0u ? fx1 : family == 1u ? fz1 : fx1 - fz1;
        // Family 3: the two ends.
        float lo = family == 3u ? 0.0 : ceil(min(a, b));
        float hi = family == 3u ? 1.0 : floor(max(a, b));
        uint count = family == 3u ? 2u : (abs(b - a) < 1e-9 ? 0u : uint(max(hi - lo + 1.0, 0.0)));
        for (uint k = 0u; k < count; k++)
        {
            float s = family == 3u ? float(k) : (lo + float(k) - a) / (b - a);
            if (s < 0.0 || s > 1.0) continue;
            float3 ps = p0 + (p1 - p0) * s;
            float h; float3 nn;
            if (!hfSurface(hf, ps.x, ps.z, h, nn)) continue;
            float depth = (ps.y - h) * nn.y;
            if (depth < deepest) { deepest = depth; n = nn; }
        }
    }
    if (deepest < 0.0) return deepest;
    float3 mn = min(p0, p1) - reach;
    float3 mx = max(p0, p1) + reach;
    uint i0, i1, j0, j1;
    if (mn.y > hf.maxH || !hfVertexRange(hf, mn, mx, i0, i1, j0, j1)) return reach;
    float best = reach;
    for (uint j = j0; j < j1; j++)
    for (uint i = i0; i < i1; i++)
    {
        float3 v00 = hfVertex(hf, i, j);
        float3 v11 = hfVertex(hf, i + 1u, j + 1u);
        float3 v10 = hfVertex(hf, i + 1u, j);
        float3 v01 = hfVertex(hf, i, j + 1u);
        for (uint tri = 0u; tri < 2u; tri++)
        {
            // Triangles (00, 10, 11) and (00, 11, 01), one call site.
            float3 qs, qt;
            float d = hfSegTriDistance(p0, p1, v00, tri == 0u ? v10 : v11, tri == 0u ? v11 : v01, qs, qt);
            if (d < best)
            {
                best = d;
                if (d > 1e-9) n = (qs - qt) / d;
            }
        }
    }
    return best;
}

// Sphere (radius r, r = 0 a point) or capsule (half length > 0, unit axis) swept
// from o along unit d by conservative advancement: each step moves by the exact
// clearance, which no point of the shape can cross sooner. The march starts
// where the path enters the field's bounds (padded by the shape) and ends where
// it leaves them, so a cast from far off the field takes no extra steps.
bool hfRoundCast(HeightFieldRef hf, float3 o, float3 axis, float halfLen, float r, float3 d, float maxT, out float t, out float3 n)
{
    t = 0.0;
    n = float3(0, 1, 0);
    float probe = 4.0 * max(hf.dx, hf.dz) + r;
    // Slab entry and exit of the bounds every centre on the path can touch from.
    float pad = halfLen + r + 1.0e-3;
    float3 bmn = float3(hf.x0 - pad, hf.minH - pad, hf.z0 - pad);
    float3 bmx = float3(-hf.x0 + pad, hf.maxH + pad, -hf.z0 + pad);
    float tIn = 0.0, tOut = maxT;
    for (uint ax = 0u; ax < 3u; ax++)
    {
        float oa = ax == 0u ? o.x : ax == 1u ? o.y : o.z;
        float da = ax == 0u ? d.x : ax == 1u ? d.y : d.z;
        float lo = ax == 0u ? bmn.x : ax == 1u ? bmn.y : bmn.z;
        float hi = ax == 0u ? bmx.x : ax == 1u ? bmx.y : bmx.z;
        if (abs(da) < 1e-12)
        {
            if (oa < lo || oa > hi) return false;
            continue;
        }
        float t0 = (lo - oa) / da, t1 = (hi - oa) / da;
        tIn = max(tIn, min(t0, t1));
        tOut = min(tOut, max(t0, t1));
    }
    if (tIn > tOut) return false;
    t = tIn;
    // Each step clears at least min(gap, probe); the count covers the path
    // through the bounds in probe-sized steps plus the slow approach near a
    // surface (bounded: a data-dependent count the shader compilers do not unroll).
    uint iterations = min(512u, 4u * (hf.nx + hf.nz) + uint(min((tOut - tIn) / probe, 256.0)) + 16u);
    for (uint k = 0u; k < iterations; k++)
    {
        float3 c = o + d * t;
        float3 nk;
        float dist = hfCapsuleClearance(hf, c - axis * halfLen, c + axis * halfLen, probe, nk);
        float gap = dist - r;
        // Absolute contact tolerance, plus f32 resolution at distance t.
        if (gap <= 1.0e-4 + 1.0e-6 * t)
        {
            n = nk;
            return t <= maxT; // t = 0 with gap < 0: started inside
        }
        t += gap;
        if (t > tOut) return false;
    }
    return false;
}

// Entry of the ray o + s·d into the hull (local frame of the hull), as rayHull
// does: false when it misses or starts inside.
bool hfRayEntersHull(HullRef h, float3 o, float3 d, float maxT, out float s, out float3 nHull)
{
    s = 0.0;
    nHull = -d;
    float tNear = -1e30, tFar = 1e30;
    for (uint f = 0u; f < h.nf; f++)
    {
        float4 pl = hullPlane(h, f);
        float denom = dot(pl.xyz, d);
        float num = pl.w - dot(pl.xyz, o);
        if (abs(denom) < 1e-12)
        {
            if (num < 0.0) return false;
            continue;
        }
        float tt = num / denom;
        if (denom < 0.0) { if (tt > tNear) { tNear = tt; nHull = pl.xyz; } }
        else tFar = min(tFar, tt);
        if (tNear > tFar) return false;
    }
    if (tNear < 0.0 || tNear > maxT || tNear <= -1e29) return false;
    s = tNear;
    return true;
}

// A box or hull swept by translation along unit d (heightfield-local; the hull
// at (tRel, qRel) in that frame). Exact for polyhedra: first contact is a hull
// vertex reaching a terrain triangle, a terrain vertex reaching a hull face, or
// a hull edge crossing a terrain edge.
bool hfHullCast(HeightFieldRef hf, HullRef h, float3 tRel, float4 qRel, float3 d, float maxT, out float t, out float3 n)
{
    t = maxT;
    n = float3(0, 1, 0);
    bool hit = false;
    float3 mn = float3(1e30, 1e30, 1e30);
    float3 mx = float3(-1e30, -1e30, -1e30);
    for (uint k = 0u; k < h.nv; k++)
    {
        float3 v = tRel + quatRotate(qRel, hullVertex(h, k));
        mn = min(mn, min(v, v + d * maxT));
        mx = max(mx, max(v, v + d * maxT));
        // A vertex already under the surface: started inside.
        float hs; float3 ns;
        if (hfSurface(hf, v.x, v.z, hs, ns) && v.y < hs) { t = 0.0; n = ns; return true; }
        float tv; float3 nv;
        if (heightFieldRayLocal(hf, v, d, t, tv, nv) && tv < t) { t = tv; n = nv; hit = true; }
    }
    if (mn.y > hf.maxH) return hit;
    uint i0, i1, j0, j1;
    if (!hfVertexRange(hf, mn, mx, i0, i1, j0, j1)) return hit;
    float4 qInv = quatConjugate(qRel);
    float3 dh = quatRotate(qInv, -d);
    for (uint j = j0; j <= j1; j++)
    for (uint i = i0; i <= i1; i++)
    {
        float3 v = hfVertex(hf, i, j);
        float s; float3 nh;
        if (hfRayEntersHull(h, quatRotate(qInv, v - tRel), dh, t, s, nh) && s < t)
        {
            t = s;
            n = -quatRotate(qRel, nh);
            hit = true;
        }
    }
    // Hull edge (a, b) moving along d against terrain edge (p, q):
    // a + u·(b - a) + s·d = p + w·(q - p) with u, w in [0, 1], s in [0, t].
    for (uint e = 0u; e < h.ne; e++)
    {
        uint4 ed = hullEdge(h, e);
        float3 a = tRel + quatRotate(qRel, hullVertex(h, ed.x));
        float3 b = tRel + quatRotate(qRel, hullVertex(h, ed.y));
        float3 emn = min(min(a, b), min(a, b) + d * t);
        float3 emx = max(max(a, b), max(a, b) + d * t);
        if (emn.y > hf.maxH) continue;
        uint ei0, ei1, ej0, ej1;
        if (!hfVertexRange(hf, emn, emx, ei0, ei1, ej0, ej1)) continue;
        float3 ea = b - a;
        for (uint j = ej0; j <= ej1; j++)
        for (uint i = ei0; i <= ei1; i++)
        {
            float3 p = hfVertex(hf, i, j);
            for (uint dd = 0u; dd < 3u; dd++)
            {
                uint ii = i + (dd != 1u ? 1u : 0u);
                uint jj = j + (dd != 0u ? 1u : 0u);
                if (ii > ei1 || jj > ej1) continue;
                float3 eq = hfVertex(hf, ii, jj) - p;
                // Columns (ea, -eq, d) · (u, w, s) = p - a (Cramer's rule).
                float3 rhs = p - a;
                float det = dot(ea, cross(-eq, d));
                if (abs(det) < 1e-12) continue;
                float u = dot(rhs, cross(-eq, d)) / det;
                float w = dot(ea, cross(rhs, d)) / det;
                float s = dot(ea, cross(-eq, rhs)) / det;
                if (u < 0.0 || u > 1.0 || w < 0.0 || w > 1.0 || s < 0.0 || s >= t) continue;
                float3 nn = cross(ea, eq);
                float len = length(nn);
                if (len < 1e-12) continue;
                nn /= len;
                if (dot(nn, d) > 0.0) nn = -nn;
                t = s;
                n = nn;
                hit = true;
            }
        }
    }
    return hit;
}

// Any supported shape (`cs` posed at p, q) swept along unit world d against a
// heightfield collider at frame fHf. n is world, toward the caster.
bool heightFieldShapeCast(HeightFieldRef hf, Frame fHf, Collider cs, float3 p, float4 q, float3 d, float maxT, out float t, out float3 n)
{
    t = 0.0;
    n = -d;
    float3 o = frameToLocal(fHf, p);
    float3 dl = quatRotate(quatConjugate(fHf.q), d);
    float3 nl;
    bool hit;
    if (cs.shape == SHAPE_SPHERE || cs.shape == SHAPE_CAPSULE)
    {
        // One call site for both (each is inlined whole by the shader compilers).
        bool capsule = cs.shape == SHAPE_CAPSULE;
        float3 axis = capsule ? quatRotate(quatConjugate(fHf.q), quatRotate(q, float3(0, 1, 0))) : float3(0, 1, 0);
        hit = hfRoundCast(hf, o, axis, capsule ? cs.params.y : 0.0, cs.params.x, dl, maxT, t, nl);
    }
#if !defined(HF_CAST_ROUND_ONLY)
    else if (shapeHasHull(cs.shape))
    {
        float4 qRel = quatMul(quatConjugate(fHf.q), q);
        hit = hfHullCast(hf, hullOf(cs), o, qRel, dl, maxT, t, nl);
    }
#endif
    else return false;
    if (hit) n = quatRotate(fHf.q, nl);
    return hit;
}

