// Heightfield collision (deformable terrain T1; engine/docs/deformable-terrain.md).
//
// A heightfield is a static grid of nx × nz height samples in the collider's
// local frame, centred on its origin: vertex (i, j) sits at
// (x0 + i·dx, h(i, j), z0 + j·dz) with x0 = -(nx-1)·dx/2, z0 = -(nz-1)·dz/2.
// Each cell is two triangles split along the (i, j)–(i+1, j+1) diagonal; the
// surface is their union and its outward side is +y.
//
// The samples live in the hull arena (gHulls), so heightfield pairs take the
// convex stages' path (CONVEX_DEFER → convex_classify) and need no extra
// binding. Record layout at `base` (engine-schema HeightField::pack):
//   0: (HF_MAGIC, nx, nz, flags)
//   1: (dx, dz, minH, maxH) as f32 bits
//   2..: heights, four per record, row-major (k = j·nx + i)
// Collider params: x = asfloat(base), w = bounding radius (broadphase level).
//
// The rigid solver takes one normal per manifold. A body touching terrain whose
// normal varies under it gets the depth-weighted mean of its contact normals,
// and every point's gap is measured along that normal (contactAccum). Spheres
// have one contact and so are exact.
#ifndef ENGINE_PHYSICS_HEIGHTFIELD_SLANG
#define ENGINE_PHYSICS_HEIGHTFIELD_SLANG

static const uint HF_MAGIC = 0x48460001u;
// Feature IDs keep bits 30..31 clear (they never seed a SAT hint). Bit 29 tags
// a heightfield contact; bits 24..26 the kind; the rest an index.
static const uint HF_FEATURE = 0x20000000u;
static const uint HF_KIND_SHAPE_VERTEX = 0u;
static const uint HF_KIND_TERRAIN_VERTEX = 1u;
static const uint HF_KIND_EDGE = 2u;
static const uint HF_KIND_ROUND = 3u;

uint hfFeature(uint kind, uint index) { return HF_FEATURE | (kind << 24) | (index & 0x00FFFFFFu); }

struct HeightFieldRef
{
    uint base;
    uint nx;
    uint nz;
    float dx;
    float dz;
    float minH;
    float maxH;
    float x0;
    float z0;
};

HeightFieldRef heightFieldOf(Collider c)
{
    HeightFieldRef hf;
    hf.base = asuint(c.params.x) & HULL_BASE_MASK;
    uint4 head = gHulls[hf.base];
    float4 dims = asfloat(gHulls[hf.base + 1u]);
    hf.nx = head.y;
    hf.nz = head.z;
    hf.dx = dims.x;
    hf.dz = dims.y;
    hf.minH = dims.z;
    hf.maxH = dims.w;
    hf.x0 = -0.5 * float(hf.nx - 1u) * hf.dx;
    hf.z0 = -0.5 * float(hf.nz - 1u) * hf.dz;
    return hf;
}

float hfHeight(HeightFieldRef hf, uint i, uint j)
{
    uint k = j * hf.nx + i;
    uint4 r = gHulls[hf.base + 2u + (k >> 2)];
    return asfloat(r[k & 3u]);
}

float3 hfVertex(HeightFieldRef hf, uint i, uint j)
{
    return float3(hf.x0 + float(i) * hf.dx, hfHeight(hf, i, j), hf.z0 + float(j) * hf.dz);
}

// Local bounds of the whole field.
void hfLocalBounds(HeightFieldRef hf, out float3 mn, out float3 mx)
{
    mn = float3(hf.x0, hf.minH, hf.z0);
    mx = float3(-hf.x0, hf.maxH, -hf.z0);
}

// Vertex index range [i0, i1] × [j0, j1] covering local x/z bounds. False when
// the bounds miss the field.
bool hfVertexRange(HeightFieldRef hf, float3 mn, float3 mx, out uint i0, out uint i1, out uint j0, out uint j1)
{
    i0 = 0u; i1 = 0u; j0 = 0u; j1 = 0u;
    float fx0 = (mn.x - hf.x0) / hf.dx;
    float fx1 = (mx.x - hf.x0) / hf.dx;
    float fz0 = (mn.z - hf.z0) / hf.dz;
    float fz1 = (mx.z - hf.z0) / hf.dz;
    float lastX = float(hf.nx - 1u);
    float lastZ = float(hf.nz - 1u);
    if (fx1 < 0.0 || fz1 < 0.0 || fx0 > lastX || fz0 > lastZ) return false;
    i0 = uint(clamp(floor(fx0), 0.0, lastX));
    i1 = uint(clamp(ceil(fx1), 0.0, lastX));
    j0 = uint(clamp(floor(fz0), 0.0, lastZ));
    j1 = uint(clamp(ceil(fz1), 0.0, lastZ));
    return true;
}

// Surface height and outward normal of the triangle under local (x, z). False
// outside the field.
bool hfSurface(HeightFieldRef hf, float x, float z, out float h, out float3 n)
{
    h = 0.0;
    n = float3(0, 1, 0);
    float fx = (x - hf.x0) / hf.dx;
    float fz = (z - hf.z0) / hf.dz;
    float lastX = float(hf.nx - 1u);
    float lastZ = float(hf.nz - 1u);
    if (fx < 0.0 || fz < 0.0 || fx > lastX || fz > lastZ) return false;
    uint i = min(uint(fx), hf.nx - 2u);
    uint j = min(uint(fz), hf.nz - 2u);
    float u = fx - float(i);
    float v = fz - float(j);
    float h00 = hfHeight(hf, i, j);
    float h10 = hfHeight(hf, i + 1u, j);
    float h01 = hfHeight(hf, i, j + 1u);
    float h11 = hfHeight(hf, i + 1u, j + 1u);
    float sx, sz;
    if (u >= v)
    {
        // Triangle (00, 10, 11).
        sx = h10 - h00;
        sz = h11 - h10;
    }
    else
    {
        // Triangle (00, 01, 11).
        sx = h11 - h01;
        sz = h01 - h00;
    }
    h = h00 + u * sx + v * sz;
    n = normalize(float3(-sx / hf.dx, 1.0, -sz / hf.dz));
    return true;
}

// Closest point on triangle abc to p (Ericson 5.1.5).
float3 hfClosestOnTriangle(float3 p, float3 a, float3 b, float3 c)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + ab * (d1 / (d1 - d3));
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + ac * (d2 / (d2 - d6));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    float denom = 1.0 / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

// Closest points of segments p1q1 and p2q2 (Ericson 5.1.9): parameters s, t.
void hfClosestSegSeg(float3 p1, float3 q1, float3 p2, float3 q2, out float s, out float t)
{
    float3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    s = 0.0; t = 0.0;
    if (a <= 1e-12 && e <= 1e-12) return;
    if (a <= 1e-12) { t = clamp(f / e, 0.0, 1.0); return; }
    float c = dot(d1, r);
    if (e <= 1e-12) { s = clamp(-c / a, 0.0, 1.0); return; }
    float b = dot(d1, d2);
    float denom = a * e - b * b;
    s = denom > 1e-12 ? clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
    t = (b * s + f) / e;
    if (t < 0.0) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); }
    else if (t > 1.0) { t = 1.0; s = clamp((b - c) / a, 0.0, 1.0); }
}

// Whether pt, on the terrain edge from vertex (i, j) in direction d (0: +x,
// 1: +z, 2: the cell diagonal), is the closest point of the triangles beside
// that edge to p. Otherwise a face is closer and the edge is no contact (flat
// ground and creases), and its tilted normal must not enter the manifold.
bool hfEdgeIsClosest(HeightFieldRef hf, uint i, uint j, uint d, float3 p, float3 pt)
{
    float tol2 = 1.0e-6 * max(hf.dx * hf.dx, hf.dz * hf.dz);
    for (uint side = 0u; side < 2u; side++)
    {
        int ci, cj;
        uint tri;
        if (d == 0u) { ci = int(i); cj = int(j) - int(side); tri = side == 0u ? 0u : 1u; }
        else if (d == 1u) { ci = int(i) - int(side); cj = int(j); tri = side == 0u ? 1u : 0u; }
        else { ci = int(i); cj = int(j); tri = side; }
        if (ci < 0 || cj < 0 || ci > int(hf.nx) - 2 || cj > int(hf.nz) - 2) continue;
        uint ui = uint(ci), uj = uint(cj);
        float3 v00 = hfVertex(hf, ui, uj);
        float3 v11 = hfVertex(hf, ui + 1u, uj + 1u);
        float3 q = tri == 0u
            ? hfClosestOnTriangle(p, v00, hfVertex(hf, ui + 1u, uj), v11)
            : hfClosestOnTriangle(p, v00, v11, hfVertex(hf, ui, uj + 1u));
        if (dot(q - pt, q - pt) > tol2) return false;
    }
    return true;
}

// Whether terrain vertex (i, j) is the closest point to p of every triangle
// that has it as a corner (a peak under p): both triangles of the cells where
// it is corner 00 or 11, triangle (00, 10, 11) where it is 10 and
// (00, 11, 01) where it is 01.
bool hfVertexIsClosest(HeightFieldRef hf, uint i, uint j, float3 p)
{
    float tol2 = 1.0e-6 * max(hf.dx * hf.dx, hf.dz * hf.dz);
    float3 v = hfVertex(hf, i, j);
    for (int dj = -1; dj <= 0; dj++)
    for (int di = -1; di <= 0; di++)
    {
        int ci = int(i) + di, cj = int(j) + dj;
        if (ci < 0 || cj < 0 || ci > int(hf.nx) - 2 || cj > int(hf.nz) - 2) continue;
        uint ui = uint(ci), uj = uint(cj);
        // v's corner of this cell: (-di, -dj).
        bool in0 = !(di == 0 && dj == -1); // not corner 01
        bool in1 = !(di == -1 && dj == 0); // not corner 10
        float3 v00 = hfVertex(hf, ui, uj);
        float3 v11 = hfVertex(hf, ui + 1u, uj + 1u);
        float3 qa = hfClosestOnTriangle(p, v00, hfVertex(hf, ui + 1u, uj), v11);
        float3 qb = hfClosestOnTriangle(p, v00, v11, hfVertex(hf, ui, uj + 1u));
        if ((in0 && dot(qa - v, qa - v) > tol2) || (in1 && dot(qb - v, qb - v) > tol2)) return false;
    }
    return true;
}

// Closest terrain point to a ball's centre c (local) within its footprint.
// Returns the outward normal at the contact and the signed distance from c to
// the surface (negative when c is under it). False when nothing is within
// `reach` of c or c is off the field.
bool hfBallQuery(HeightFieldRef hf, float3 c, float reach, out float3 q, out float3 n, out float dist, out uint feature)
{
    q = c; n = float3(0, 1, 0); dist = 1e30; feature = 0u;
    float hc; float3 nc;
    bool over = hfSurface(hf, c.x, c.z, hc, nc);
    // A centre under the surface: the triangle under it gives the way out.
    if (over && c.y < hc)
    {
        dist = (c.y - hc) * nc.y;
        n = nc;
        q = c - nc * dist;
        uint i = min(uint((c.x - hf.x0) / hf.dx), hf.nx - 2u);
        uint j = min(uint((c.z - hf.z0) / hf.dz), hf.nz - 2u);
        feature = j * hf.nx + i;
        return true;
    }
    uint i0, i1, j0, j1;
    if (!hfVertexRange(hf, c - reach, c + reach, i0, i1, j0, j1)) return false;
    if (c.y - reach > hf.maxH) return false;
    float best = reach * reach;
    bool found = false;
    for (uint j = j0; j < j1; j++)
    for (uint i = i0; i < i1; i++)
    {
        float3 v00 = hfVertex(hf, i, j);
        float3 v10 = hfVertex(hf, i + 1u, j);
        float3 v01 = hfVertex(hf, i, j + 1u);
        float3 v11 = hfVertex(hf, i + 1u, j + 1u);
        for (uint tri = 0u; tri < 2u; tri++)
        {
            float3 qa = hfClosestOnTriangle(c, v00, tri == 0u ? v10 : v11, tri == 0u ? v11 : v01);
            float da = dot(c - qa, c - qa);
            if (da < best) { best = da; q = qa; feature = j * hf.nx + i; found = true; }
        }
    }
    if (!found) return false;
    dist = sqrt(best);
    if (dist > 1e-9) n = (c - q) / dist;
    else
    {
        float h; float3 ns;
        hfSurface(hf, q.x, q.z, h, ns);
        n = ns;
    }
    return true;
}

// ---- manifolds --------------------------------------------------------------
// All functions return a manifold whose normal points A → B; `shapeIsA` says
// which side the non-terrain shape is on. `spec` is the speculative gap.

BoxManifold hfFinish(ContactAccum acc, Frame fHf, float3 nLocal, bool shapeIsA)
{
    BoxManifold mf;
    mf.count = 0;
    float3 up = quatRotate(fHf.q, nLocal);
    // Reference = terrain side; incident = the shape.
    contactAccumEmit(acc, fHf, nLocal, !shapeIsA, shapeIsA ? -up : up, mf);
    return mf;
}

BoxManifold heightFieldSphereManifold(HeightFieldRef hf, Frame fHf, float3 center, float radius, bool sphereIsA, float spec)
{
    BoxManifold mf;
    mf.count = 0;
    mf.normal = float3(0, 1, 0);
    float3 c = frameToLocal(fHf, center);
    float3 q, n; float dist; uint feature;
    if (!hfBallQuery(hf, c, radius + spec, q, n, dist, feature)) return mf;
    if (dist - radius > spec) return mf;
    float3 nW = quatRotate(fHf.q, n);
    float3 pTerrain = frameToWorld(fHf, q);
    float3 pSphere = center - nW * radius;
    mf.normal = sphereIsA ? -nW : nW;
    uint f = hfFeature(HF_KIND_ROUND, feature);
    if (sphereIsA) manifoldAdd(mf, pSphere, pTerrain, f);
    else manifoldAdd(mf, pTerrain, pSphere, f);
    return mf;
}

// Candidate points gathered before the shared normal is known.
static const int HF_MAX_CANDIDATES = 24;
struct HfCandidates
{
    float3 p[HF_MAX_CANDIDATES];   // shape-side point (local)
    float3 n[HF_MAX_CANDIDATES];   // outward terrain normal of that contact (local)
    float sep[HF_MAX_CANDIDATES];  // signed gap along n
    uint feature[HF_MAX_CANDIDATES];
    int count;
    float3 nSum;                   // depth-weighted normal sum
    float deepest;                 // smallest gap seen
    int deepestIndex;
};

HfCandidates hfCandidatesInit()
{
    HfCandidates c;
    c.count = 0;
    c.nSum = float3(0, 0, 0);
    c.deepest = 1e30;
    c.deepestIndex = -1;
    for (int k = 0; k < HF_MAX_CANDIDATES; k++)
    {
        c.p[k] = float3(0, 0, 0); c.n[k] = float3(0, 1, 0); c.sep[k] = 0.0; c.feature[k] = 0u;
    }
    return c;
}

// Every candidate weighs the normal; the deepest ones are kept. When the buffer
// is full a new candidate replaces the shallowest one if it is deeper.
void hfCandidateAdd(inout HfCandidates c, float3 p, float3 n, float sep, uint feature, float spec)
{
    if (sep > spec) return;
    c.nSum += n * (spec - sep + 1.0e-6);
    int slot = c.count;
    if (slot >= HF_MAX_CANDIDATES)
    {
        slot = 0;
        for (int k = 1; k < HF_MAX_CANDIDATES; k++)
            if (c.sep[k] > c.sep[slot] || (c.sep[k] == c.sep[slot] && c.feature[k] > c.feature[slot])) slot = k;
        if (sep >= c.sep[slot]) return;
    }
    else c.count++;
    c.p[slot] = p; c.n[slot] = n; c.sep[slot] = sep; c.feature[slot] = feature;
    if (sep < c.deepest) { c.deepest = sep; c.deepestIndex = slot; }
}

// Shared normal, then each candidate's gap along it into the reduction.
BoxManifold hfReduce(HfCandidates c, Frame fHf, bool shapeIsA, float spec)
{
    BoxManifold mf;
    mf.count = 0;
    mf.normal = float3(0, 1, 0);
    if (c.count == 0) return mf;
    float len = length(c.nSum);
    float3 nL = len > 1e-12 ? c.nSum / len : c.n[max(c.deepestIndex, 0)];
    ContactAccum acc = contactAccumInit(nL, spec);
    for (int k = 0; k < c.count; k++)
    {
        // Gap along nL of the terrain point under this candidate.
        float3 onTerrain = c.p[k] - c.n[k] * c.sep[k];
        contactAccumAdd(acc, c.p[k], dot(c.p[k] - onTerrain, nL), c.feature[k]);
    }
    return hfFinish(acc, fHf, nL, shapeIsA);
}

// Shape-side test of one point: its gap to the triangle under it.
void hfAddPointBelow(HeightFieldRef hf, inout HfCandidates c, float3 p, uint feature, float spec)
{
    float h; float3 n;
    if (!hfSurface(hf, p.x, p.z, h, n)) return;
    hfCandidateAdd(c, p, n, (p.y - h) * n.y, feature, spec);
}

// The capsule's axis p0p1 against the plane of the triangle under each point.
// Along the segment both its height and the surface's are linear between the
// points where its x/z track crosses a grid line or a cell diagonal, so the
// smallest gap lies at one of those crossings or an end: testing them finds the
// deepest point at any depth, including an axis that has sunk through a ridge
// with both ends still above the surface.
void hfAddSegmentBelow(HeightFieldRef hf, inout HfCandidates c, float3 p0, float3 p1, float radius, float spec)
{
    float fx0 = (p0.x - hf.x0) / hf.dx, fx1 = (p1.x - hf.x0) / hf.dx;
    float fz0 = (p0.z - hf.z0) / hf.dz, fz1 = (p1.z - hf.z0) / hf.dz;
    // Lines x = i, z = j and diagonals fx - fz = k, in grid units.
    for (uint family = 0u; family < 3u; family++)
    {
        float a = family == 0u ? fx0 : family == 1u ? fz0 : fx0 - fz0;
        float b = family == 0u ? fx1 : family == 1u ? fz1 : fx1 - fz1;
        if (abs(b - a) < 1e-9) continue;
        float lo = ceil(min(a, b)), hi = floor(max(a, b));
        uint n = uint(max(hi - lo + 1.0, 0.0));
        for (uint k = 0u; k < n; k++)
        {
            float s = (lo + float(k) - a) / (b - a);
            if (s <= 0.0 || s >= 1.0) continue;
            float3 ps = p0 + (p1 - p0) * s;
            float h; float3 nn;
            if (!hfSurface(hf, ps.x, ps.z, h, nn)) continue;
            float sep = (ps.y - h) * nn.y - radius;
            hfCandidateAdd(c, ps - nn * radius, nn, sep, hfFeature(HF_KIND_ROUND, 2u + (family << 12) + (uint(lo + float(k)) & 4095u)), spec);
        }
    }
}

BoxManifold heightFieldCapsuleManifold(HeightFieldRef hf, Frame fHf, Frame cap, float radius, float halfLen, bool capsuleIsA, float spec)
{
    float3 axis = quatRotate(cap.q, float3(0, 1, 0)) * halfLen;
    float3 p0 = frameToLocal(fHf, cap.p - axis);
    float3 p1 = frameToLocal(fHf, cap.p + axis);
    HfCandidates c = hfCandidatesInit();
    float reach = radius + spec;
    // End caps: the closest terrain point to each end.
    for (uint e = 0u; e < 2u; e++)
    {
        float3 pe = e == 0u ? p0 : p1;
        float3 q, n; float dist; uint feature;
        if (hfBallQuery(hf, pe, reach, q, n, dist, feature))
            hfCandidateAdd(c, pe - n * radius, n, dist - radius, hfFeature(HF_KIND_ROUND, e), spec);
    }
    hfAddSegmentBelow(hf, c, p0, p1, radius, spec);
    // The segment against every terrain edge in its footprint (ridges and
    // peaks between the caps).
    float3 mn = min(p0, p1) - reach;
    float3 mx = max(p0, p1) + reach;
    uint i0, i1, j0, j1;
    if (mn.y <= hf.maxH && hfVertexRange(hf, mn, mx, i0, i1, j0, j1))
    {
        for (uint j = j0; j <= j1; j++)
        for (uint i = i0; i <= i1; i++)
        {
            float3 v = hfVertex(hf, i, j);
            for (uint d = 0u; d < 3u; d++)
            {
                uint ii = i + (d != 1u ? 1u : 0u);
                uint jj = j + (d != 0u ? 1u : 0u);
                if (ii > i1 || jj > j1) continue;
                float3 w = hfVertex(hf, ii, jj);
                float s, t;
                hfClosestSegSeg(p0, p1, v, w, s, t);
                // A segment end is a cap contact, found exactly above.
                if (s <= 1e-5 || s >= 1.0 - 1e-5) continue;
                float3 ps = p0 + (p1 - p0) * s;
                float3 pt = v + (w - v) * t;
                float h; float3 ns;
                if (!hfSurface(hf, ps.x, ps.z, h, ns) || ps.y < h) continue; // the caps cover it
                float3 dv = ps - pt;
                float dist = length(dv);
                if (dist <= 1e-9 || dist - radius > spec) continue;
                // Only where this terrain feature is locally the closest to
                // the segment (a ridge or a peak); otherwise a face is closer.
                bool closest = t <= 1e-5 ? hfVertexIsClosest(hf, i, j, ps)
                    : t >= 1.0 - 1e-5 ? hfVertexIsClosest(hf, ii, jj, ps)
                    : hfEdgeIsClosest(hf, i, j, d, ps, pt);
                if (!closest) continue;
                float3 n = dv / dist;
                hfCandidateAdd(c, ps - n * radius, n, dist - radius,
                    hfFeature(HF_KIND_EDGE, ((j * hf.nx + i) * 3u + d) << 1), spec);
            }
        }
    }
    return hfReduce(c, fHf, capsuleIsA, spec);
}

// Boxes and convex hulls (a box is the unit cube scaled, as in convex.slang).
BoxManifold heightFieldHullManifold(HeightFieldRef hf, Frame fHf, HullRef h, Frame fHull, bool hullIsA, float spec)
{
    BoxManifold none;
    none.count = 0;
    none.normal = float3(0, 1, 0);
    // Hull → terrain-local transform, and the hull's local bounds there.
    float4 qRel = quatMul(quatConjugate(fHf.q), fHull.q);
    float3 tRel = frameToLocal(fHf, fHull.p);
    float3 mn = float3(1e30, 1e30, 1e30);
    float3 mx = float3(-1e30, -1e30, -1e30);
    for (uint k = 0u; k < h.nv; k++)
    {
        float3 v = tRel + quatRotate(qRel, hullVertex(h, k));
        mn = min(mn, v);
        mx = max(mx, v);
    }
    mn -= spec; mx += spec;
    if (mn.y > hf.maxH) return none;
    uint i0, i1, j0, j1;
    if (!hfVertexRange(hf, mn, mx, i0, i1, j0, j1)) return none;
    HfCandidates c = hfCandidatesInit();
    // 1. Hull vertices against the triangle under each.
    for (uint k = 0u; k < h.nv; k++)
    {
        float3 v = tRel + quatRotate(qRel, hullVertex(h, k));
        hfAddPointBelow(hf, c, v, hfFeature(HF_KIND_SHAPE_VERTEX, k), spec);
    }
    // 2. Terrain vertices inside (or within spec of) the hull: the least-deep
    //    face is the way out, its plane the contact.
    float4 qInv = quatConjugate(qRel);
    for (uint j = j0; j <= j1; j++)
    for (uint i = i0; i <= i1; i++)
    {
        float3 v = hfVertex(hf, i, j);
        if (v.y < mn.y) continue;
        float3 pl = quatRotate(qInv, v - tRel);
        float best = -1e30;
        uint face = 0u;
        for (uint f = 0u; f < h.nf; f++)
        {
            float4 plane = hullPlane(h, f);
            float s = dot(plane.xyz, pl) - plane.w;
            if (s > best) { best = s; face = f; }
        }
        if (best > spec) continue;
        float3 nFace = quatRotate(qRel, hullPlane(h, face).xyz);
        // Terrain-outward normal is minus the hull face's; the hull point sits
        // on that face above the terrain vertex.
        hfCandidateAdd(c, v - nFace * best, -nFace, best,
            hfFeature(HF_KIND_TERRAIN_VERTEX, j * hf.nx + i), spec);
    }
    // 3. Hull edges against terrain edges, for crossings neither vertex test
    //    sees (an edge resting across a ridge).
    for (uint e = 0u; e < h.ne; e++)
    {
        uint4 ed = hullEdge(h, e);
        float3 a = tRel + quatRotate(qRel, hullVertex(h, ed.x));
        float3 b = tRel + quatRotate(qRel, hullVertex(h, ed.y));
        float3 emn = min(a, b) - spec;
        float3 emx = max(a, b) + spec;
        if (emn.y > hf.maxH) continue;
        uint ei0, ei1, ej0, ej1;
        if (!hfVertexRange(hf, emn, emx, ei0, ei1, ej0, ej1)) continue;
        for (uint j = ej0; j <= ej1; j++)
        for (uint i = ei0; i <= ei1; i++)
        {
            float3 v = hfVertex(hf, i, j);
            for (uint d = 0u; d < 3u; d++)
            {
                uint ii = i + (d != 1u ? 1u : 0u);
                uint jj = j + (d != 0u ? 1u : 0u);
                if (ii > ei1 || jj > ej1) continue;
                float3 w = hfVertex(hf, ii, jj);
                float s, t;
                hfClosestSegSeg(a, b, v, w, s, t);
                // End points are vertex contacts (tests 1 and 2).
                if (s <= 1e-4 || s >= 1.0 - 1e-4 || t <= 1e-4 || t >= 1.0 - 1e-4) continue;
                float3 ps = a + (b - a) * s;
                float3 pt = v + (w - v) * t;
                float hs; float3 ns;
                if (!hfSurface(hf, ps.x, ps.z, hs, ns)) continue;
                float3 axis = cross(b - a, w - v);
                float al = length(axis);
                if (al <= 1e-9) continue;
                axis /= al;
                if (axis.y < 0.0) axis = -axis;
                float sep = dot(ps - pt, axis);
                // The hull's interior must lie on the far side of the terrain
                // edge from below, or this axis is not a contact.
                float3 inside = tRel + quatRotate(qRel, hullInterior(h));
                if (dot(inside - pt, axis) < sep - 1e-6) continue;
                // Only a ridge (the surface falls away on both sides of the
                // terrain edge) has an edge contact; a crease is the faces'.
                float3 side = normalize(float3(-(w - v).z, 0.0, (w - v).x) + float3(1e-12, 0, 0));
                float probe = 0.25 * min(hf.dx, hf.dz);
                float ha, hb; float3 nn;
                bool ridge = true;
                if (hfSurface(hf, pt.x + side.x * probe, pt.z + side.z * probe, ha, nn))
                    ridge = ridge && dot(float3(pt.x + side.x * probe, ha, pt.z + side.z * probe) - pt, axis) <= 1e-5;
                if (hfSurface(hf, pt.x - side.x * probe, pt.z - side.z * probe, hb, nn))
                    ridge = ridge && dot(float3(pt.x - side.x * probe, hb, pt.z - side.z * probe) - pt, axis) <= 1e-5;
                if (!ridge) continue;
                hfCandidateAdd(c, ps, axis, sep,
                    hfFeature(HF_KIND_EDGE, (((j * hf.nx + i) * 3u + d) << 8) | (e & 255u)), spec);
            }
        }
    }
    return hfReduce(c, fHf, hullIsA, spec);
}

// Any supported shape against a heightfield collider. The pair is (ca, cb); one
// of them is the heightfield.
BoxManifold heightFieldPairManifold(Collider ca, BodyState ba, Collider cb, BodyState bb, float spec)
{
    bool hfIsA = ca.shape == SHAPE_HEIGHTFIELD;
    Collider ch = hfIsA ? ca : cb;
    Collider cs = hfIsA ? cb : ca;
    BodyState bh = hfIsA ? ba : bb;
    BodyState bs = hfIsA ? bb : ba;
    HeightFieldRef hf = heightFieldOf(ch);
    Frame fh = colliderFrame(ch, bh);
    Frame fs = colliderFrame(cs, bs);
    bool shapeIsA = !hfIsA;
    if (cs.shape == SHAPE_SPHERE)
        return heightFieldSphereManifold(hf, fh, fs.p, cs.params.x, shapeIsA, spec);
    if (cs.shape == SHAPE_CAPSULE)
        return heightFieldCapsuleManifold(hf, fh, fs, cs.params.x, cs.params.y, shapeIsA, spec);
    if (shapeHasHull(cs.shape))
        return heightFieldHullManifold(hf, fh, hullOf(cs), fs, shapeIsA, spec);
    BoxManifold none;
    none.count = 0;
    none.normal = float3(0, 1, 0);
    return none;
}

// World bounds of a heightfield collider (build_aabbs).
void heightFieldWorldAabb(Collider c, float3 worldPos, float4 worldRot, float pad, out float3 mn, out float3 mx)
{
    HeightFieldRef hf = heightFieldOf(c);
    float3 lmn, lmx;
    hfLocalBounds(hf, lmn, lmx);
    mn = float3(1e30, 1e30, 1e30);
    mx = float3(-1e30, -1e30, -1e30);
    for (uint k = 0u; k < 8u; k++)
    {
        float3 l = float3((k & 1u) != 0u ? lmx.x : lmn.x, (k & 2u) != 0u ? lmx.y : lmn.y, (k & 4u) != 0u ? lmx.z : lmn.z);
        float3 p = worldPos + quatRotate(worldRot, l);
        mn = min(mn, p);
        mx = max(mx, p);
    }
    mn -= pad;
    mx += pad;
}

// Ray against the surface (local frame): march cell by cell along the ray's
// x/z footprint (2D DDA) and test the two triangles of each cell. Returns the
// distance along unit `dir` and the outward normal.
bool hfRayTriangle(float3 o, float3 d, float3 a, float3 b, float3 c, inout float tBest, inout float3 nBest)
{
    float3 e1 = b - a, e2 = c - a;
    float3 p = cross(d, e2);
    float det = dot(e1, p);
    if (abs(det) < 1e-12) return false;
    float inv = 1.0 / det;
    float3 s = o - a;
    float u = dot(s, p) * inv;
    if (u < 0.0 || u > 1.0) return false;
    float3 qv = cross(s, e1);
    float v = dot(d, qv) * inv;
    if (v < 0.0 || u + v > 1.0) return false;
    float t = dot(e2, qv) * inv;
    if (t < 0.0 || t >= tBest) return false;
    float3 n = normalize(cross(e1, e2));
    if (n.y < 0.0) n = -n;
    tBest = t;
    nBest = n;
    return true;
}

bool heightFieldRayLocal(HeightFieldRef hf, float3 o, float3 d, float maxT, out float tHit, out float3 nHit)
{
    tHit = maxT;
    nHit = float3(0, 1, 0);
    // Clip the ray to the field's box.
    float3 bmn, bmx;
    hfLocalBounds(hf, bmn, bmx);
    float t0 = 0.0, t1 = maxT;
    for (int k = 0; k < 3; k++)
    {
        if (abs(d[k]) < 1e-12)
        {
            if (o[k] < bmn[k] || o[k] > bmx[k]) return false;
            continue;
        }
        float inv = 1.0 / d[k];
        float ta = (bmn[k] - o[k]) * inv;
        float tb = (bmx[k] - o[k]) * inv;
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    }
    if (t0 > t1) return false;
    float3 start = o + d * t0;
    int ci = int(clamp(floor((start.x - hf.x0) / hf.dx), 0.0, float(hf.nx - 2u)));
    int cj = int(clamp(floor((start.z - hf.z0) / hf.dz), 0.0, float(hf.nz - 2u)));
    int stepI = d.x > 0.0 ? 1 : -1;
    int stepJ = d.z > 0.0 ? 1 : -1;
    float tdx = abs(d.x) > 1e-12 ? hf.dx / abs(d.x) : 1e30;
    float tdz = abs(d.z) > 1e-12 ? hf.dz / abs(d.z) : 1e30;
    float nextX = hf.x0 + float(ci + (stepI > 0 ? 1 : 0)) * hf.dx;
    float nextZ = hf.z0 + float(cj + (stepJ > 0 ? 1 : 0)) * hf.dz;
    float tmx = abs(d.x) > 1e-12 ? (nextX - o.x) / d.x : 1e30;
    float tmz = abs(d.z) > 1e-12 ? (nextZ - o.z) / d.z : 1e30;
    bool hit = false;
    uint cells = (hf.nx + hf.nz) * 2u;
    for (uint k = 0u; k < cells; k++)
    {
        uint i = uint(ci), j = uint(cj);
        float3 v00 = hfVertex(hf, i, j);
        float3 v10 = hfVertex(hf, i + 1u, j);
        float3 v01 = hfVertex(hf, i, j + 1u);
        float3 v11 = hfVertex(hf, i + 1u, j + 1u);
        hit = hfRayTriangle(o, d, v00, v10, v11, tHit, nHit) || hit;
        hit = hfRayTriangle(o, d, v00, v11, v01, tHit, nHit) || hit;
        float tExit = min(tmx, tmz);
        if (hit && tHit <= tExit) return true;
        if (tExit > t1) break;
        if (tmx < tmz) { ci += stepI; tmx += tdx; }
        else { cj += stepJ; tmz += tdz; }
        if (ci < 0 || cj < 0 || ci > int(hf.nx) - 2 || cj > int(hf.nz) - 2) break;
    }
    return hit;
}

#include "heightfield_casts.slang"
#endif
