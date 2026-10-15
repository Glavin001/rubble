// Convex hull collision helpers (packed hull ABI v2; see engine-schema/src/convex.rs).
// Gauss-map edge pruning and incident-face selection adapted from three-avbd (MIT),
// Copyright (c) 2026 Steven Bobyn; see ../THIRD_PARTY_LICENSES.txt.
//
// Requires `gHulls` (uint4 arena) and the common/box_collide types. All geometry is
// evaluated in a hull's local (scaled) frame where possible: one transform per
// query point instead of one per vertex. Boxes reuse the unit cube at record 0
// with their half extents as a per-axis scale, so box-hull pairs need no extra path.
#ifndef ENGINE_PHYSICS_CONVEX_SLANG
#define ENGINE_PHYSICS_CONVEX_SLANG

static const uint HULL_FLAG_CLOSED = 1u;
static const uint HULL_BASE_MASK = 0x1FFFFFFFu;
static const uint HULL_UNIT_BOX = 0u;
static const uint HULL_NO_FACE = 255u;
static const float CONVEX_PLANE_EPS = 1.0e-5;
// Hysteresis for the previous frame's SAT feature: keeps feature IDs (and so
// warm starts) stable when two axes are within round-off of each other.
static const float CONVEX_HYSTERESIS = 2.0e-4;
// three-avbd face bias: an edge axis must be clearly better than the best face.
static const float CONVEX_EDGE_REL = 0.95;
static const float CONVEX_EDGE_ABS = 0.01;
#ifndef MAX_CONVEX_CONTACTS
#define MAX_CONVEX_CONTACTS 4
#endif
#if MAX_CONVEX_CONTACTS > 4
static const int CONVEX_DIRS = 8;
#else
static const int CONVEX_DIRS = 4;
#endif
static const int CONVEX_SLOTS = CONVEX_DIRS + 1;

// Contact feature IDs. Bits 30..31: 1 = face of A is the reference, 2 = face of
// B, 3 = edge/edge. Face: ref face 23..29, incident face 16..22, kind 14..15
// (0 incident vertex, 1 incident edge x reference side, 2 reference vertex),
// a 7..13, b 0..6. Edge: A edge 8..15, B edge 0..7. Other hull pairs keep bits
// 30..31 clear so they never seed a SAT hint.
static const uint CONVEX_FEATURE_FACE_A = 1u;
static const uint CONVEX_FEATURE_FACE_B = 2u;
static const uint CONVEX_FEATURE_EDGE = 3u;
static const uint CONVEX_FEATURE_PLANE = 0x08000000u;
static const uint CONVEX_FEATURE_SPHERE = 0x0A000000u;
static const uint CONVEX_FEATURE_CAPSULE = 0x0C000000u;

// Candidate `_pad` after narrowphase: deferred hull pair plus the previous
// frame's SAT feature (type 16..17, first index 8..15, second index 0..7).
static const uint CONVEX_DEFER = 0x80000000u;

uint convexFeatureFace(uint type, uint refFace, uint incFace, uint kind, uint a, uint b)
{
    return (type << 30) | ((refFace & 127u) << 23) | ((incFace & 127u) << 16) |
        ((kind & 3u) << 14) | ((a & 127u) << 7) | (b & 127u);
}

uint convexFeatureEdge(uint ea, uint eb)
{
    return (CONVEX_FEATURE_EDGE << 30) | ((ea & 255u) << 8) | (eb & 255u);
}

// Previous contact feature -> compact SAT hint (0 = none).
uint convexHintFromFeature(uint feature)
{
    uint type = feature >> 30;
    if (type == 0u) return 0u;
    if (type == CONVEX_FEATURE_EDGE)
        return (type << 16) | (((feature >> 8) & 255u) << 8) | (feature & 255u);
    return (type << 16) | (((feature >> 23) & 127u) << 8);
}

bool shapeHasHull(uint shape) { return shape == SHAPE_CONVEX || shape == SHAPE_BOX; }

struct HullRef
{
    uint base;
    uint nv;
    uint nf;
    uint ne;
    uint flags;
    uint indexWord;
    uint edgeWord;
    uint adjWord;
    float3 scale;
};

HullRef hullOf(Collider c)
{
    HullRef h;
    if (c.shape == SHAPE_BOX)
    {
        h.base = HULL_UNIT_BOX;
        h.scale = c.params.xyz;
    }
    else
    {
        h.base = asuint(c.params.x) & HULL_BASE_MASK;
        float s = c.params.y > 0.0 ? c.params.y : 1.0;
        h.scale = float3(s, s, s);
    }
    uint4 head = gHulls[h.base];
    h.nv = head.x & 255u;
    h.nf = (head.x >> 8) & 255u;
    h.ne = (head.x >> 16) & 255u;
    h.flags = head.x >> 24;
    h.indexWord = head.y;
    h.edgeWord = head.z;
    h.adjWord = head.w;
    return h;
}

bool hullClosed(HullRef h) { return (h.flags & HULL_FLAG_CLOSED) != 0u; }

uint hullWord(HullRef h, uint word)
{
    uint4 r = gHulls[h.base + (word >> 2)];
    return r[word & 3u];
}

uint hullByte(HullRef h, uint sectionWord, uint b)
{
    return (hullWord(h, sectionWord + (b >> 2)) >> ((b & 3u) * 8u)) & 255u;
}

float3 hullVertex(HullRef h, uint i) { return asfloat(gHulls[h.base + 4u + i].xyz) * h.scale; }
uint hullVertexAdj(HullRef h, uint i) { return gHulls[h.base + 4u + i].w; }

// Outward plane (n, d) of the scaled hull: (S⁻¹n)·x = d, renormalised.
float4 hullPlane(HullRef h, uint f)
{
    float4 p = asfloat(gHulls[h.base + 4u + h.nv + f]);
    float3 n = p.xyz / h.scale;
    float inv = rsqrt(max(dot(n, n), 1e-30));
    return float4(n * inv, p.w * inv);
}

uint2 hullFace(HullRef h, uint f)
{
    uint entry = hullWord(h, (4u + h.nv + h.nf) * 4u + f);
    return uint2(entry & 0xFFFFu, entry >> 16);
}

uint hullFaceVertex(HullRef h, uint2 face, uint k) { return hullByte(h, h.indexWord, face.x + k); }

// (a, b, faceL, faceR); faceL walks a -> b.
uint4 hullEdge(HullRef h, uint e)
{
    uint x = hullWord(h, h.edgeWord + e);
    return uint4(x & 255u, (x >> 8) & 255u, (x >> 16) & 255u, x >> 24);
}

float3 hullInterior(HullRef h) { return asfloat(gHulls[h.base + 3u].xyz) * h.scale; }
float3 hullLocalCenter(HullRef h) { return asfloat(gHulls[h.base + 1u].xyz) * h.scale; }
float3 hullLocalHalf(HullRef h) { return asfloat(gHulls[h.base + 2u].xyz) * abs(h.scale); }

// Index of the vertex maximising dot(v, dir) in the scaled local frame. Closed
// hulls hill-climb the vertex graph (convex: a local maximum is global);
// uncertified hulls scan every vertex.
uint hullSupportIndex(HullRef h, float3 dir)
{
    if (!hullClosed(h))
    {
        uint best = 0u;
        float bestD = -1e30;
        for (uint i = 0u; i < h.nv; i++)
        {
            float d = dot(hullVertex(h, i), dir);
            if (d > bestD) { bestD = d; best = i; }
        }
        return best;
    }
    uint cur = 0u;
    float curD = dot(hullVertex(h, 0u), dir);
    for (uint step = 0u; step < h.nv; step++)
    {
        uint adj = hullVertexAdj(h, cur);
        uint start = adj & 0xFFFFu;
        uint deg = adj >> 16;
        uint next = cur;
        float nextD = curD;
        for (uint k = 0u; k < deg; k++)
        {
            uint4 e = hullEdge(h, hullByte(h, h.adjWord, start + k));
            uint other = e.x == cur ? e.y : e.x;
            float d = dot(hullVertex(h, other), dir);
            if (d > nextD) { nextD = d; next = other; }
        }
        if (next == cur) break;
        cur = next;
        curD = nextD;
    }
    return cur;
}

// Face around vertex `v` whose normal is most anti-parallel to `refN` (local).
// three-avbd: the incident face must touch the deepest vertex on irregular hulls.
uint hullIncidentFace(HullRef h, uint v, float3 refN)
{
    uint best = 0u;
    float bestD = 1e30;
    if (hullClosed(h))
    {
        uint adj = hullVertexAdj(h, v);
        uint start = adj & 0xFFFFu;
        uint deg = adj >> 16;
        for (uint k = 0u; k < deg; k++)
        {
            uint4 e = hullEdge(h, hullByte(h, h.adjWord, start + k));
            float dl = dot(hullPlane(h, e.z).xyz, refN);
            if (dl < bestD) { bestD = dl; best = e.z; }
            if (e.w != HULL_NO_FACE)
            {
                float dr = dot(hullPlane(h, e.w).xyz, refN);
                if (dr < bestD) { bestD = dr; best = e.w; }
            }
        }
        if (bestD < 1e29) return best;
    }
    bool touched = false;
    for (uint f = 0u; f < h.nf; f++)
    {
        uint2 face = hullFace(h, f);
        bool touches = false;
        for (uint k = 0u; k < face.y; k++)
            if (hullFaceVertex(h, face, k) == v) touches = true;
        float d = dot(hullPlane(h, f).xyz, refN);
        // Faces touching v win; otherwise fall back to the most anti-parallel face.
        if (touches && (!touched || d < bestD))
        {
            touched = true;
            bestD = d;
            best = f;
        }
        else if (!touched && !touches && d < bestD)
        {
            bestD = d;
            best = f;
        }
    }
    return best;
}

// ---- streaming contact reduction --------------------------------------------
// Clipping streams candidate points; only the deepest point and the extremes
// along CONVEX_DIRS in-plane directions are retained. No polygon buffers.

struct ContactAccum
{
    float3 u;
    float3 v;
    float3 point[CONVEX_SLOTS];  // incident-side point (accumulator frame)
    float depth[CONVEX_SLOTS];   // signed distance below the reference plane
    float score[CONVEX_SLOTS];
    uint feature[CONVEX_SLOTS];
};

ContactAccum contactAccumInit(float3 n)
{
    ContactAccum acc;
    orthonormalBasis(n, acc.u, acc.v);
    for (int k = 0; k < CONVEX_SLOTS; k++)
    {
        acc.point[k] = float3(0, 0, 0);
        acc.depth[k] = 0.0;
        acc.score[k] = -1e30;
        acc.feature[k] = 0u;
    }
    return acc;
}

float3 contactAccumDir(ContactAccum acc, int k)
{
    if (k == 0) return acc.u;
    if (k == 1) return -acc.u;
    if (k == 2) return acc.v;
    if (k == 3) return -acc.v;
    if (k == 4) return acc.u + acc.v;
    if (k == 5) return -acc.u - acc.v;
    if (k == 6) return acc.u - acc.v;
    return acc.v - acc.u;
}

void contactAccumAdd(inout ContactAccum acc, float3 p, float depth, uint feature)
{
    if (depth > CONVEX_PLANE_EPS) return;
    if (-depth > acc.score[0])
    {
        acc.score[0] = -depth; acc.point[0] = p; acc.depth[0] = depth; acc.feature[0] = feature;
    }
    for (int k = 0; k < CONVEX_DIRS; k++)
    {
        float s = dot(contactAccumDir(acc, k), p);
        if (s > acc.score[k + 1])
        {
            acc.score[k + 1] = s; acc.point[k + 1] = p; acc.depth[k + 1] = depth; acc.feature[k + 1] = feature;
        }
    }
}

bool contactAccumEmpty(ContactAccum acc) { return acc.score[0] < -1e29; }

// Pick at most MAX_CONVEX_CONTACTS of the retained slots: deepest, farthest from
// it, then the largest triangle on each side (Jolt PruneContactPoints shape).
// Returns slot indices in `pick` and their count.
int contactAccumSelect(ContactAccum acc, float3 n, out int pick[CONVEX_SLOTS])
{
    for (int k = 0; k < CONVEX_SLOTS; k++) pick[k] = -1;
    if (contactAccumEmpty(acc)) return 0;
    // Unique live slots (slots can share a point).
    int live[CONVEX_SLOTS];
    for (int k = 0; k < CONVEX_SLOTS; k++) live[k] = 0;
    int count = 0;
    for (int k = 0; k < CONVEX_SLOTS; k++)
    {
        if (acc.score[k] < -1e29) continue;
        bool dup = false;
        for (int j = 0; j < count; j++)
        {
            float3 d = acc.point[live[j]] - acc.point[k];
            if (dot(d, d) < BOX_MERGE_DIST_SQ) dup = true;
        }
        if (!dup) { live[count] = k; count++; }
    }
    if (count <= MAX_CONVEX_CONTACTS)
    {
        for (int k = 0; k < count; k++) pick[k] = live[k];
        return count;
    }
    // live[0] is slot 0, the deepest point.
    float3 p0 = acc.point[live[0]];
    int i1 = 1;
    float best = -1.0;
    for (int k = 1; k < count; k++)
    {
        float3 d = acc.point[live[k]] - p0;
        d -= n * dot(d, n);
        float dd = dot(d, d);
        if (dd > best) { best = dd; i1 = k; }
    }
    float3 p1 = acc.point[live[i1]];
    int i2 = -1, i3 = -1;
    float pos = 0.0, neg = 0.0;
    for (int k = 1; k < count; k++)
    {
        if (k == i1) continue;
        float area = dot(cross(p1 - p0, acc.point[live[k]] - p0), n);
        if (area > pos) { pos = area; i2 = k; }
        if (area < neg) { neg = area; i3 = k; }
    }
    int picked = 0;
    pick[picked++] = live[0];
    pick[picked++] = live[i1];
    if (i2 >= 0 && picked < MAX_CONVEX_CONTACTS) pick[picked++] = live[i2];
    if (i3 >= 0 && picked < MAX_CONVEX_CONTACTS) pick[picked++] = live[i3];
    // MAX_CONVEX_CONTACTS > 4: append the remaining extremes in slot order.
    for (int k = 1; k < count && picked < MAX_CONVEX_CONTACTS; k++)
        if (k != i1 && k != i2 && k != i3) pick[picked++] = live[k];
    return picked;
}

void manifoldAdd(inout BoxManifold mf, float3 pA, float3 pB, uint feature)
{
    if (mf.count >= 8) return;
    mf.pts[mf.count].pA = pA;
    mf.pts[mf.count].pB = pB;
    mf.pts[mf.count].feature = feature;
    mf.count++;
}

// Transform r -> world for accumulator points stored in a hull's local frame.
struct Frame { float3 p; float4 q; };
float3 frameToWorld(Frame f, float3 x) { return f.p + quatRotate(f.q, x); }
float3 frameToLocal(Frame f, float3 x) { return quatRotate(quatConjugate(f.q), x - f.p); }

// Emit the reduced accumulator as a manifold. Points are on the incident
// surface in `frame`; the reference point is the projection on the plane n·x=d.
void contactAccumEmit(
    ContactAccum acc, Frame frame, float3 nLocal, bool referenceIsA,
    float3 normalAB, inout BoxManifold mf)
{
    int pick[CONVEX_SLOTS];
    int count = contactAccumSelect(acc, nLocal, pick);
    mf.normal = normalAB;
    for (int k = 0; k < count; k++)
    {
        int s = pick[k];
        float3 inc = frameToWorld(frame, acc.point[s]);
        float3 ref = frameToWorld(frame, acc.point[s] - nLocal * acc.depth[s]);
        if (referenceIsA) manifoldAdd(mf, ref, inc, acc.feature[s]);
        else manifoldAdd(mf, inc, ref, acc.feature[s]);
    }
}

// ---- face / edge manifolds for a SAT result ------------------------------------

Frame colliderFrame(Collider c, BodyState b)
{
    Frame f;
    f.p = transformPoint(b.position, b.orientation, c.localPos);
    f.q = quatMul(b.orientation, c.localRot);
    return f;
}

void closestSegments(float3 p1, float3 q1, float3 p2, float3 q2, out float3 c1, out float3 c2)
{
    float3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    float a = dot(d1, d1), e = dot(d2, d2), b = dot(d1, d2), c = dot(d1, r), f = dot(d2, r);
    float s = 0.0, t = 0.0;
    if (a > 1e-12 && e > 1e-12)
    {
        float denom = a * e - b * b;
        s = denom > 1e-12 ? clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
        t = (b * s + f) / e;
        if (t < 0.0) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); }
        else if (t > 1.0) { t = 1.0; s = clamp((b - c) / a, 0.0, 1.0); }
    }
    else if (e > 1e-12) t = clamp(f / e, 0.0, 1.0);
    else if (a > 1e-12) s = clamp(-c / a, 0.0, 1.0);
    c1 = p1 + s * d1;
    c2 = p2 + t * d2;
}

// Uncertified hulls: the SAT edge pair fixes the axis, not the supporting edges
// (any parallel edge yields the same cross product). Pick the parallel edge that
// is extreme along n: `sign` +1 maximises (A), -1 minimises (B).
uint4 hullExtremeParallelEdge(HullRef h, Frame f, uint4 e, float3 n, float sign)
{
    float3 dir = normalize(hullVertex(h, e.y) - hullVertex(h, e.x));
    float3 nLocal = quatRotate(quatConjugate(f.q), n) * sign;
    uint4 best = e;
    float bestD = -1e30;
    for (uint k = 0u; k < h.ne; k++)
    {
        uint4 c = hullEdge(h, k);
        float3 x = hullVertex(h, c.x);
        float3 y = hullVertex(h, c.y);
        float3 d = y - x;
        float len = length(d);
        if (len < 1e-12 || abs(dot(d / len, dir)) < 0.999) continue;
        float s = dot(nLocal, x + y);
        if (s > bestD) { bestD = s; best = c; }
    }
    return best;
}

// World A->B axis of edge pair (ea, eb), oriented away from A's interior.
float3 convexEdgeAxis(HullRef ha, Frame fa, HullRef hb, Frame fb, uint ea, uint eb)
{
    uint4 a = hullEdge(ha, ea);
    uint4 b = hullEdge(hb, eb);
    float3 pa = frameToWorld(fa, hullVertex(ha, a.x));
    float3 da = frameToWorld(fa, hullVertex(ha, a.y)) - pa;
    float3 pb = frameToWorld(fb, hullVertex(hb, b.x));
    float3 db = frameToWorld(fb, hullVertex(hb, b.y)) - pb;
    float3 n = normalize(cross(da, db));
    float3 ref = hullClosed(ha) && hullClosed(hb)
        ? pa - frameToWorld(fa, hullInterior(ha))
        : frameToWorld(fb, hullInterior(hb)) - frameToWorld(fa, hullInterior(ha));
    return dot(n, ref) < 0.0 ? -n : n;
}

BoxManifold convexEdgeManifold(HullRef ha, Frame fa, HullRef hb, Frame fb, uint ea, uint eb, float3 n)
{
    BoxManifold mf;
    mf.count = 0;
    mf.normal = n;
    uint4 a = hullEdge(ha, ea);
    uint4 b = hullEdge(hb, eb);
    if (!hullClosed(ha) || !hullClosed(hb))
    {
        a = hullExtremeParallelEdge(ha, fa, a, n, 1.0);
        b = hullExtremeParallelEdge(hb, fb, b, n, -1.0);
    }
    float3 pa, pb;
    closestSegments(
        frameToWorld(fa, hullVertex(ha, a.x)), frameToWorld(fa, hullVertex(ha, a.y)),
        frameToWorld(fb, hullVertex(hb, b.x)), frameToWorld(fb, hullVertex(hb, b.y)), pa, pb);
    manifoldAdd(mf, pa, pb, convexFeatureEdge(ea, eb));
    return mf;
}

// Thread-scalar Gauss-map edge query (closed hulls only), for the rare cached-
// face patch that clips empty. Returns the best edge separation (A->B normal in
// world) or -1e30 when it cannot be evaluated (uncertified adjacency).
float convexEdgeQueryScalar(HullRef ha, Frame fa, HullRef hb, Frame fb, out uint bestA, out uint bestB, out float3 bestN)
{
    bestA = 0u; bestB = 0u; bestN = float3(0, 1, 0);
    if (!hullClosed(ha) || !hullClosed(hb)) return -1e30;
    float4 qBi = quatConjugate(fb.q);
    float4 qRel = quatMul(qBi, fa.q);
    float3 tRel = quatRotate(qBi, fa.p - fb.p);
    float3 interiorA = tRel + quatRotate(qRel, hullInterior(ha));
    float best = -1e30;
    for (uint ea = 0u; ea < ha.ne; ea++)
    {
        uint4 e1 = hullEdge(ha, ea);
        float3 na = quatRotate(qRel, hullPlane(ha, e1.z).xyz);
        float3 nb = quatRotate(qRel, hullPlane(ha, e1.w).xyz);
        float3 bxa = cross(nb, na);
        float3 pa = tRel + quatRotate(qRel, hullVertex(ha, e1.x));
        float3 da = tRel + quatRotate(qRel, hullVertex(ha, e1.y)) - pa;
        for (uint eb = 0u; eb < hb.ne; eb++)
        {
            uint4 e2 = hullEdge(hb, eb);
            float3 c = -hullPlane(hb, e2.z).xyz;
            float3 d = -hullPlane(hb, e2.w).xyz;
            float cba = dot(c, bxa);
            float dba = dot(d, bxa);
            if (cba * dba >= 0.0) continue;
            float3 dxc = cross(d, c);
            float adc = dot(na, dxc);
            float bdc = dot(nb, dxc);
            if (adc * bdc >= 0.0 || cba * bdc <= 0.0) continue;
            float3 pb = hullVertex(hb, e2.x);
            float3 db = hullVertex(hb, e2.y) - pb;
            float3 n = cross(da, db);
            float len = length(n);
            if (len < 1e-5 * length(da) * length(db) || len < 1e-12) continue;
            n /= len;
            if (dot(n, pa - interiorA) < 0.0) n = -n;
            float sep = dot(n, pb - pa);
            if (sep > best) { best = sep; bestA = ea; bestB = eb; bestN = quatRotate(fb.q, n); }
        }
    }
    return best;
}

// Reference face `rf` of `ref` against `inc`. `normalAB` is the A->B SAT normal.
// Clipping runs in the reference hull's local frame (Cyrus-Beck per incident
// edge against the reference side planes, plus reference vertices inside the
// incident face), streaming points into the reduction accumulator.
BoxManifold convexFaceManifold(
    HullRef ref, Frame fr, HullRef inc, Frame fi, uint rf, bool referenceIsA,
    float3 normalAB, bool allowFallback)
{
    BoxManifold mf;
    mf.count = 0;
    mf.normal = normalAB;
    float4 plane = hullPlane(ref, rf);
    float3 n = plane.xyz;
    // Incident hull relative to the reference frame.
    float4 qRel = quatMul(quatConjugate(fr.q), fi.q);
    float3 tRel = quatRotate(quatConjugate(fr.q), fi.p - fr.p);
    float3 nInc = quatRotate(quatConjugate(qRel), -n); // deepest direction, incident local
    uint deep = hullSupportIndex(inc, nInc);
    uint incFace = hullIncidentFace(inc, deep, -nInc);
    uint2 rFace = hullFace(ref, rf);
    uint2 iFace = hullFace(inc, incFace);
    uint type = referenceIsA ? CONVEX_FEATURE_FACE_A : CONVEX_FEATURE_FACE_B;
    ContactAccum acc = contactAccumInit(n);

    // Incident edges clipped against every reference side plane.
    float3 x0 = tRel + quatRotate(qRel, hullVertex(inc, hullFaceVertex(inc, iFace, 0u)));
    float3 first = x0;
    for (uint k = 0u; k < iFace.y; k++)
    {
        float3 x1 = k + 1u < iFace.y
            ? tRel + quatRotate(qRel, hullVertex(inc, hullFaceVertex(inc, iFace, k + 1u)))
            : first;
        float t0 = 0.0, t1 = 1.0;
        uint enter = 127u, leave = 127u;
        float3 r0 = hullVertex(ref, hullFaceVertex(ref, rFace, 0u));
        float3 rFirst = r0;
        bool rejected = false;
        for (uint j = 0u; j < rFace.y && !rejected; j++)
        {
            float3 r1 = j + 1u < rFace.y ? hullVertex(ref, hullFaceVertex(ref, rFace, j + 1u)) : rFirst;
            float3 side = cross(r1 - r0, n);
            float len = length(side);
            if (len > 1e-12)
            {
                side /= len;
                float da = dot(side, x0 - r0);
                float db = dot(side, x1 - r0);
                if (da > CONVEX_PLANE_EPS && db > CONVEX_PLANE_EPS) rejected = true;
                else if (da > CONVEX_PLANE_EPS)
                {
                    float t = da / (da - db);
                    if (t > t0) { t0 = t; enter = j; }
                }
                else if (db > CONVEX_PLANE_EPS)
                {
                    float t = da / (da - db);
                    if (t < t1) { t1 = t; leave = j; }
                }
            }
            r0 = r1;
        }
        if (!rejected && t0 <= t1)
        {
            uint kNext = k + 1u < iFace.y ? k + 1u : 0u;
            float3 p0 = lerp(x0, x1, t0);
            float3 p1 = lerp(x0, x1, t1);
            uint f0 = enter == 127u
                ? convexFeatureFace(type, rf, incFace, 0u, k, 0u)
                : convexFeatureFace(type, rf, incFace, 1u, k, enter);
            uint f1 = leave == 127u
                ? convexFeatureFace(type, rf, incFace, 0u, kNext, 0u)
                : convexFeatureFace(type, rf, incFace, 1u, k, leave);
            contactAccumAdd(acc, p0, dot(n, p0) - plane.w, f0);
            contactAccumAdd(acc, p1, dot(n, p1) - plane.w, f1);
        }
        x0 = x1;
    }

    // Reference vertices inside the incident face (tested in the incident frame).
    float4 iPlane = hullPlane(inc, incFace);
    float3 nRefInc = quatRotate(quatConjugate(qRel), n);
    float denom = dot(iPlane.xyz, nRefInc);
    if (denom < -1e-3)
    {
        for (uint j = 0u; j < rFace.y; j++)
        {
            float3 r = hullVertex(ref, hullFaceVertex(ref, rFace, j));
            float3 rInc = quatRotate(quatConjugate(qRel), r - tRel);
            // Project along the reference normal onto the incident plane, then
            // test that point (not r) against the incident polygon's sides.
            float t = (dot(iPlane.xyz, rInc) - iPlane.w) / denom;
            float3 xInc = rInc - nRefInc * t;
            bool inside = true;
            float3 y0 = hullVertex(inc, hullFaceVertex(inc, iFace, 0u));
            float3 yFirst = y0;
            for (uint k = 0u; k < iFace.y && inside; k++)
            {
                float3 y1 = k + 1u < iFace.y ? hullVertex(inc, hullFaceVertex(inc, iFace, k + 1u)) : yFirst;
                if (dot(cross(y1 - y0, iPlane.xyz), xInc - y0) > CONVEX_PLANE_EPS) inside = false;
                y0 = y1;
            }
            if (!inside) continue;
            float3 x = tRel + quatRotate(qRel, xInc);
            contactAccumAdd(acc, x, dot(n, x) - plane.w, convexFeatureFace(type, rf, incFace, 2u, j, 0u));
        }
    }

    contactAccumEmit(acc, fr, n, referenceIsA, normalAB, mf);
    if (mf.count == 0 && allowFallback)
    {
        // Round-off or an imperfect baked loop emptied the patch after SAT proved
        // overlap: keep a supporting contact (three-avbd fallback).
        float3 deepLocal = tRel + quatRotate(qRel, hullVertex(inc, deep));
        float depth = dot(n, deepLocal) - plane.w;
        float3 incW = frameToWorld(fr, deepLocal);
        float3 refW = frameToWorld(fr, deepLocal - n * depth);
        uint feat = convexFeatureFace(type, rf, incFace, 3u, 127u, 127u);
        if (referenceIsA) manifoldAdd(mf, refW, incW, feat);
        else manifoldAdd(mf, incW, refW, feat);
    }
    return mf;
}

// ---- hull versus plane / sphere / capsule ---------------------------------------

// Plane n·x = w (world). Hull touching or below the plane.
BoxManifold convexPlaneManifold(HullRef h, Frame f, float3 pn, float pw, bool hullIsA)
{
    BoxManifold mf;
    mf.count = 0;
    pn = normalize(pn);
    mf.normal = hullIsA ? -pn : pn;
    float3 nL = quatRotate(quatConjugate(f.q), pn);
    float dL = pw - dot(pn, f.p);
    // Hill-climbing early-out: lowest vertex above the plane means no contact.
    uint low = hullSupportIndex(h, -nL);
    if (dot(nL, hullVertex(h, low)) - dL > CONVEX_PLANE_EPS) return mf;
    ContactAccum acc = contactAccumInit(nL);
    for (uint i = 0u; i < h.nv; i++)
    {
        float3 v = hullVertex(h, i);
        contactAccumAdd(acc, v, dot(nL, v) - dL, CONVEX_FEATURE_PLANE | i);
    }
    // Reference = plane side; incident = hull.
    contactAccumEmit(acc, f, nL, !hullIsA, mf.normal, mf);
    return mf;
}

// Closest point on the polygon `face` (local) to p, with p's plane distance s.
float3 hullClosestOnFace(HullRef h, uint fIndex, float4 plane, float3 p)
{
    uint2 face = hullFace(h, fIndex);
    float3 q = p - plane.xyz * (dot(plane.xyz, p) - plane.w);
    bool inside = true;
    float3 best = q;
    float bestD = 1e30;
    float3 v0 = hullVertex(h, hullFaceVertex(h, face, 0u));
    float3 vFirst = v0;
    for (uint k = 0u; k < face.y; k++)
    {
        float3 v1 = k + 1u < face.y ? hullVertex(h, hullFaceVertex(h, face, k + 1u)) : vFirst;
        float3 e = v1 - v0;
        if (dot(cross(e, plane.xyz), q - v0) > 0.0)
        {
            inside = false;
            float t = clamp(dot(p - v0, e) / max(dot(e, e), 1e-20), 0.0, 1.0);
            float3 c = v0 + e * t;
            float d = dot(p - c, p - c);
            if (d < bestD) { bestD = d; best = c; }
        }
        v0 = v1;
    }
    return inside ? q : best;
}

BoxManifold convexSphereManifold(HullRef h, Frame f, float3 center, float radius, bool hullIsA)
{
    BoxManifold mf;
    mf.count = 0;
    mf.normal = float3(0, 1, 0);
    float3 c = frameToLocal(f, center);
    float maxPlane = -1e30;
    uint maxFace = 0u;
    for (uint k = 0u; k < h.nf; k++)
    {
        float4 pl = hullPlane(h, k);
        float s = dot(pl.xyz, c) - pl.w;
        if (s > radius) return mf; // separating face axis
        if (s > maxPlane) { maxPlane = s; maxFace = k; }
    }
    float3 nLocal;
    float3 onHull;
    uint feature = CONVEX_FEATURE_SPHERE | maxFace;
    if (maxPlane <= 0.0)
    {
        nLocal = hullPlane(h, maxFace).xyz;
        onHull = c - nLocal * maxPlane;
    }
    else
    {
        // Outside: the closest surface point lies on a face whose plane faces c.
        float bestD = 1e30;
        onHull = c;
        for (uint k = 0u; k < h.nf; k++)
        {
            float4 pl = hullPlane(h, k);
            if (dot(pl.xyz, c) - pl.w <= 0.0) continue;
            float3 q = hullClosestOnFace(h, k, pl, c);
            float d = dot(c - q, c - q);
            if (d < bestD) { bestD = d; onHull = q; feature = CONVEX_FEATURE_SPHERE | k; }
        }
        if (bestD > radius * radius) return mf;
        nLocal = bestD > 1e-16 ? (c - onHull) * rsqrt(bestD) : hullPlane(h, maxFace).xyz;
    }
    float3 n = quatRotate(f.q, nLocal);
    float3 pHull = frameToWorld(f, onHull);
    float3 pSphere = center - n * radius;
    mf.normal = hullIsA ? n : -n;
    if (hullIsA) manifoldAdd(mf, pHull, pSphere, feature);
    else manifoldAdd(mf, pSphere, pHull, feature);
    return mf;
}

// Capsule = segment (local Y, half length) swept by a radius.
BoxManifold convexCapsuleManifold(HullRef h, Frame f, Frame cap, float radius, float halfLen, bool hullIsA)
{
    BoxManifold mf;
    mf.count = 0;
    mf.normal = float3(0, 1, 0);
    float3 axis = quatRotate(cap.q, float3(0, 1, 0)) * halfLen;
    float3 p0 = frameToLocal(f, cap.p - axis);
    float3 p1 = frameToLocal(f, cap.p + axis);
    float3 seg = p1 - p0;
    float segLen = length(seg);
    float3 u = segLen > 1e-9 ? seg / segLen : float3(0, 1, 0);
    // Face axes of the hull.
    float faceSep = -1e30;
    uint face = 0u;
    for (uint k = 0u; k < h.nf; k++)
    {
        float4 pl = hullPlane(h, k);
        float s = min(dot(pl.xyz, p0), dot(pl.xyz, p1)) - pl.w - radius;
        if (s > 0.0) return mf;
        if (s > faceSep) { faceSep = s; face = k; }
    }
    // Edge x segment axes (Gauss-map arc crosses the great circle normal to u).
    float edgeSep = -1e30;
    float3 edgeN = float3(0, 1, 0);
    uint edge = 0u;
    bool closed = hullClosed(h);
    float3 interior = hullInterior(h);
    if (segLen > 1e-9)
    {
        for (uint e = 0u; e < h.ne; e++)
        {
            uint4 ed = hullEdge(h, e);
            if (closed)
            {
                float sl = dot(hullPlane(h, ed.z).xyz, u);
                float sr = dot(hullPlane(h, ed.w).xyz, u);
                if (sl * sr >= 0.0) continue;
            }
            float3 a = hullVertex(h, ed.x);
            float3 n = cross(hullVertex(h, ed.y) - a, u);
            float len = length(n);
            if (len < 1e-9) continue;
            n /= len;
            if (dot(n, a - interior) < 0.0) n = -n;
            float hullMax = dot(n, a);
            if (!closed)
                for (uint i = 0u; i < h.nv; i++) hullMax = max(hullMax, dot(n, hullVertex(h, i)));
            float s = min(dot(n, p0), dot(n, p1)) - hullMax - radius;
            if (s > 0.0) return mf;
            if (s > edgeSep) { edgeSep = s; edgeN = n; edge = e; }
        }
    }
    float3 nLocal;
    if (edgeSep > CONVEX_EDGE_REL * faceSep + CONVEX_EDGE_ABS)
    {
        uint4 ed = hullEdge(h, edge);
        float3 cs, ch;
        closestSegments(p0, p1, hullVertex(h, ed.x), hullVertex(h, ed.y), cs, ch);
        nLocal = edgeN;
        float3 n = quatRotate(f.q, nLocal);
        float3 pHull = frameToWorld(f, ch);
        float3 pCap = frameToWorld(f, cs) - n * radius;
        mf.normal = hullIsA ? n : -n;
        uint feat = CONVEX_FEATURE_CAPSULE | 0x00800000u | edge;
        if (hullIsA) manifoldAdd(mf, pHull, pCap, feat);
        else manifoldAdd(mf, pCap, pHull, feat);
        return mf;
    }
    // Face: clip the segment to the face's side planes (Cyrus-Beck).
    float4 pl = hullPlane(h, face);
    nLocal = pl.xyz;
    uint2 fc = hullFace(h, face);
    float t0 = 0.0, t1 = 1.0;
    float3 v0 = hullVertex(h, hullFaceVertex(h, fc, 0u));
    float3 vFirst = v0;
    for (uint k = 0u; k < fc.y; k++)
    {
        float3 v1 = k + 1u < fc.y ? hullVertex(h, hullFaceVertex(h, fc, k + 1u)) : vFirst;
        float3 side = cross(v1 - v0, nLocal);
        float da = dot(side, p0 - v0);
        float db = dot(side, p1 - v0);
        if (da > 0.0 && db > 0.0) { t0 = 1.0; t1 = 0.0; }
        else if (da > 0.0) t0 = max(t0, da / (da - db));
        else if (db > 0.0) t1 = min(t1, da / (da - db));
        v0 = v1;
    }
    float3 n = quatRotate(f.q, nLocal);
    mf.normal = hullIsA ? n : -n;
    uint added = 0u;
    if (t0 <= t1)
    {
        for (uint k = 0u; k < 2u; k++)
        {
            if (k == 1u && t1 - t0 < 1e-6) break;
            float3 p = lerp(p0, p1, k == 0u ? t0 : t1);
            float dist = dot(nLocal, p) - pl.w;
            if (dist - radius > CONVEX_PLANE_EPS) continue;
            float3 pHull = frameToWorld(f, p - nLocal * dist);
            float3 pCap = frameToWorld(f, p) - n * radius;
            uint feat = CONVEX_FEATURE_CAPSULE | (face << 8) | k;
            if (hullIsA) manifoldAdd(mf, pHull, pCap, feat);
            else manifoldAdd(mf, pCap, pHull, feat);
            added++;
        }
    }
    if (added == 0u)
    {
        float3 p = dot(nLocal, p0) <= dot(nLocal, p1) ? p0 : p1;
        float dist = dot(nLocal, p) - pl.w;
        float3 pHull = frameToWorld(f, p - nLocal * dist);
        float3 pCap = frameToWorld(f, p) - n * radius;
        uint feat = CONVEX_FEATURE_CAPSULE | (face << 8) | 2u;
        if (hullIsA) manifoldAdd(mf, pHull, pCap, feat);
        else manifoldAdd(mf, pCap, pHull, feat);
    }
    return mf;
}

#endif
