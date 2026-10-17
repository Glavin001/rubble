// Box-box SAT + face clipping, ported from avbd-demo3d/collide.cpp (and three-avbd ref).
// Contact normal is A→B (normalAB). Points are world-space on each surface.

#ifndef ENGINE_PHYSICS_BOX_COLLIDE_SLANG
#define ENGINE_PHYSICS_BOX_COLLIDE_SLANG

// The intersection of two rotated rectangular faces can have eight vertices.
// Keep the whole support polygon: taking its first four biases forces and topples rings.
static const int BOX_MAX_CONTACTS = 8;
// Clipping a convex quad against four half-planes adds at most four
// vertices. Match three-avbd: retain all eight without 16-vertex copies.
static const int BOX_MAX_POLY = 8;
static const float BOX_SAT_EPS = 1.0e-6;
static const float BOX_PLANE_EPS = 1.0e-5;
static const float BOX_MERGE_DIST_SQ = 1.0e-6;
static const int BOX_AXIS_FACE_A = 0;
static const int BOX_AXIS_FACE_B = 1;
static const int BOX_AXIS_EDGE = 2;

struct BoxObb
{
    float3 center;
    float3 half;
    float3 axis0;
    float3 axis1;
    float3 axis2;
};

// three-avbd demo brick [1, 0.5, 0.5] → half (0.5, 0.25, 0.25). Used to gate Ring-only
// manifold caps so random Box Pile sizes keep full side contacts.
bool isDemoBrickHalf(float3 h)
{
    return abs(h.x - 0.5) < 0.05 && abs(h.y - 0.25) < 0.05 && abs(h.z - 0.25) < 0.05;
}

struct BoxManifoldPt
{
    float3 pA;
    float3 pB;
    uint feature;
};

struct BoxManifold
{
    float3 normal; // A → B
    int count;
    BoxManifoldPt pts[8];
};

float boxHalfComp(float3 h, int i)
{
    if (i == 0) return h.x;
    if (i == 1) return h.y;
    return h.z;
}

float3 boxAxis(BoxObb b, int i)
{
    if (i == 0) return b.axis0;
    if (i == 1) return b.axis1;
    return b.axis2;
}

BoxObb makeBoxObb(float3 center, float4 rot, float3 half)
{
    BoxObb b;
    b.center = center;
    b.half = half;
    b.axis0 = quatRotate(rot, float3(1, 0, 0));
    b.axis1 = quatRotate(rot, float3(0, 1, 0));
    b.axis2 = quatRotate(rot, float3(0, 0, 1));
    return b;
}

float3 boxSupportPoint(BoxObb box, float3 dir)
{
    float sx = dot(dir, box.axis0) >= 0.0 ? 1.0 : -1.0;
    float sy = dot(dir, box.axis1) >= 0.0 ? 1.0 : -1.0;
    float sz = dot(dir, box.axis2) >= 0.0 ? 1.0 : -1.0;
    return box.center
        + box.axis0 * (box.half.x * sx)
        + box.axis1 * (box.half.y * sy)
        + box.axis2 * (box.half.z * sz);
}

void boxFaceAxes(BoxObb box, int axisIndex, out float3 u, out float3 v, out float extentU, out float extentV)
{
    if (axisIndex == 0)
    {
        u = box.axis1; v = box.axis2; extentU = box.half.y; extentV = box.half.z;
    }
    else if (axisIndex == 1)
    {
        u = box.axis0; v = box.axis2; extentU = box.half.x; extentV = box.half.z;
    }
    else
    {
        u = box.axis0; v = box.axis1; extentU = box.half.x; extentV = box.half.y;
    }
}

bool boxTestAxis(
    BoxObb boxA, BoxObb boxB, float3 delta, float3 axis,
    int type, int indexA, int indexB, float margin,
    inout bool bestValid, inout float bestSep, inout float3 bestN,
    inout int bestType, inout int bestIA, inout int bestIB)
{
    float lenSq = dot(axis, axis);
    if (lenSq < BOX_SAT_EPS) return true;
    float3 n = axis * (1.0 / sqrt(lenSq));
    if (dot(n, delta) < 0.0) n = -n;
    float distance = abs(dot(delta, n));
    float rA = boxA.half.x * abs(dot(n, boxA.axis0))
             + boxA.half.y * abs(dot(n, boxA.axis1))
             + boxA.half.z * abs(dot(n, boxA.axis2));
    float rB = boxB.half.x * abs(dot(n, boxB.axis0))
             + boxB.half.y * abs(dot(n, boxB.axis1))
             + boxB.half.z * abs(dot(n, boxB.axis2));
    float separation = distance - (rA + rB);
    // three-avbd testAxis: reject sep > 0. Margin belongs in C0 at emit only —
    // keeping sep∈(0,margin] doubled Ring contact pairs vs the oracle.
    if (separation > 0.0) return false;
    // Prefer strictly larger separation. On near-ties, keep the first axis (FACE_A
    // before FACE_B) so feature prefixes don't flip-flop on stacked boxes and kill warm-start.
    bool better = !bestValid || separation > bestSep + 1.0e-5;
    if (better)
    {
        bestValid = true;
        bestSep = separation;
        bestN = n;
        bestType = type;
        bestIA = indexA;
        bestIB = indexB;
    }
    return true;
}

// inout lowers to a WGSL function pointer instead of an array value. The
// source is read-only here; callers alternate two distinct scratch arrays.
int boxClipPoly(inout float3 inV[8], int inCount, float3 planeN, float planeOffset, out float3 outV[8])
{
    int outCount = 0;
    if (inCount <= 0) return 0;
    float3 a = inV[inCount - 1];
    float da = dot(planeN, a) - planeOffset;
    for (int i = 0; i < inCount; i++)
    {
        float3 b = inV[i];
        float db = dot(planeN, b) - planeOffset;
        bool aIn = da <= BOX_PLANE_EPS;
        bool bIn = db <= BOX_PLANE_EPS;
        if (aIn != bIn)
        {
            float t = 0.0;
            float denom = da - db;
            if (abs(denom) > BOX_SAT_EPS) t = clamp(da / denom, 0.0, 1.0);
            if (outCount < BOX_MAX_POLY) outV[outCount++] = a + (b - a) * t;
        }
        if (bIn && outCount < BOX_MAX_POLY) outV[outCount++] = b;
        a = b;
        da = db;
    }
    return outCount;
}

bool boxAddPt(inout BoxManifold m, inout float3 mids[8], float3 xA, float3 xB, uint feature)
{
    float3 mid = 0.5 * (xA + xB);
    for (int i = 0; i < m.count; i++)
    {
        float3 d = mid - mids[i];
        if (dot(d, d) < BOX_MERGE_DIST_SQ) return false;
    }
    if (m.count >= BOX_MAX_CONTACTS) return false;
    m.pts[m.count].pA = xA;
    m.pts[m.count].pB = xB;
    m.pts[m.count].feature = feature;
    mids[m.count] = mid;
    m.count++;
    return true;
}

void boxBuildFaceManifold(
    BoxObb boxA, BoxObb boxB, bool referenceIsA, int referenceAxis, float3 normalAB,
    inout BoxManifold m)
{
    BoxObb referenceBox = referenceIsA ? boxA : boxB;
    BoxObb incidentBox = referenceIsA ? boxB : boxA;
    float3 referenceOutward = referenceIsA ? normalAB : -normalAB;

    float sign = dot(referenceOutward, boxAxis(referenceBox, referenceAxis)) >= 0.0 ? 1.0 : -1.0;
    float3 refN = boxAxis(referenceBox, referenceAxis) * sign;
    float3 refC = referenceBox.center + refN * boxHalfComp(referenceBox.half, referenceAxis);
    float3 refU, refV;
    float extU, extV;
    boxFaceAxes(referenceBox, referenceAxis, refU, refV, extU, extV);

    int incidentAxis = 0;
    float best = -1e30;
    for (int i = 0; i < 3; i++)
    {
        float d = abs(dot(boxAxis(incidentBox, i), refN));
        if (d > best) { best = d; incidentAxis = i; }
    }

    float isign = dot(boxAxis(incidentBox, incidentAxis), refN) > 0.0 ? -1.0 : 1.0;
    float3 faceN = boxAxis(incidentBox, incidentAxis) * isign;
    float3 faceC = incidentBox.center + faceN * boxHalfComp(incidentBox.half, incidentAxis);
    float3 iu, iv;
    float ieU, ieV;
    boxFaceAxes(incidentBox, incidentAxis, iu, iv, ieU, ieV);

    float3 clip0[8];
    float3 clip1[8];
    clip0[0] = faceC + iu * ieU + iv * ieV;
    clip0[1] = faceC - iu * ieU + iv * ieV;
    clip0[2] = faceC - iu * ieU - iv * ieV;
    clip0[3] = faceC + iu * ieU - iv * ieV;
    int count = 4;

    count = boxClipPoly(clip0, count, refU, dot(refU, refC) + extU, clip1);
    if (count == 0) return;
    count = boxClipPoly(clip1, count, -refU, dot(-refU, refC) + extU, clip0);
    if (count == 0) return;
    count = boxClipPoly(clip0, count, refV, dot(refV, refC) + extV, clip1);
    if (count == 0) return;
    count = boxClipPoly(clip1, count, -refV, dot(-refV, refC) + extV, clip0);
    if (count == 0) return;

    float3 mids[8];
    uint featurePrefix = uint((referenceIsA ? BOX_AXIS_FACE_A : BOX_AXIS_FACE_B) << 24)
        | (uint(referenceAxis) << 16) | (uint(incidentAxis) << 8);
    // Keep every clipped corner, matching three-avbd, including rotated stacking faces.
    int maxPts = BOX_MAX_CONTACTS;

    for (int i = 0; i < count && m.count < maxPts; i++)
    {
        float3 pIncident = clip0[i];
        float distance = dot(pIncident - refC, refN);
        if (distance > BOX_PLANE_EPS) continue;
        float3 pReference = pIncident - refN * distance;
        float3 xA = referenceIsA ? pReference : pIncident;
        float3 xB = referenceIsA ? pIncident : pReference;
        // Geometry-derived feature id (not clip loop order) — stops warm-start churn.
        float3 rel = pReference - refC;
        float u = dot(rel, refU) / max(extU, 1e-8);
        float v = dot(rel, refV) / max(extV, 1e-8);
        uint qu = uint(clamp(floor((u + 1.0) * 2.0), 0.0, 3.0));
        uint qv = uint(clamp(floor((v + 1.0) * 2.0), 0.0, 3.0));
        boxAddPt(m, mids, xA, xB, featurePrefix | (qu << 2) | qv);
    }
    if (m.count == 0)
    {
        boxAddPt(m, mids, boxSupportPoint(boxA, normalAB), boxSupportPoint(boxB, -normalAB), featurePrefix);
    }
}

void boxBuildEdgeContact(
    BoxObb boxA, BoxObb boxB, int axisA, int axisB, float3 normalAB, inout BoxManifold m)
{
    int a1 = (axisA + 1) % 3;
    int a2 = (axisA + 2) % 3;
    float s1 = dot(normalAB, boxAxis(boxA, a1)) >= 0.0 ? 1.0 : -1.0;
    float s2 = dot(normalAB, boxAxis(boxA, a2)) >= 0.0 ? 1.0 : -1.0;
    float3 eC = boxA.center
        + boxAxis(boxA, a1) * (boxHalfComp(boxA.half, a1) * s1)
        + boxAxis(boxA, a2) * (boxHalfComp(boxA.half, a2) * s2);
    float3 a0 = eC - boxAxis(boxA, axisA) * boxHalfComp(boxA.half, axisA);
    float3 a1p = eC + boxAxis(boxA, axisA) * boxHalfComp(boxA.half, axisA);

    float3 nB = -normalAB;
    int b1 = (axisB + 1) % 3;
    int b2 = (axisB + 2) % 3;
    float t1 = dot(nB, boxAxis(boxB, b1)) >= 0.0 ? 1.0 : -1.0;
    float t2 = dot(nB, boxAxis(boxB, b2)) >= 0.0 ? 1.0 : -1.0;
    float3 fC = boxB.center
        + boxAxis(boxB, b1) * (boxHalfComp(boxB.half, b1) * t1)
        + boxAxis(boxB, b2) * (boxHalfComp(boxB.half, b2) * t2);
    float3 b0 = fC - boxAxis(boxB, axisB) * boxHalfComp(boxB.half, axisB);
    float3 b1p = fC + boxAxis(boxB, axisB) * boxHalfComp(boxB.half, axisB);

    float3 d1 = a1p - a0;
    float3 d2 = b1p - b0;
    float3 r = a0 - b0;
    float aa = dot(d1, d1);
    float ee = dot(d2, d2);
    float ff = dot(d2, r);
    float s = 0.0;
    float t = 0.0;
    if (aa > BOX_SAT_EPS || ee > BOX_SAT_EPS)
    {
        if (aa <= BOX_SAT_EPS) t = clamp(ff / ee, 0.0, 1.0);
        else
        {
            float c = dot(d1, r);
            if (ee <= BOX_SAT_EPS) s = clamp(-c / aa, 0.0, 1.0);
            else
            {
                float b = dot(d1, d2);
                float denom = aa * ee - b * b;
                if (abs(denom) > BOX_SAT_EPS) s = clamp((b * ff - c * ee) / denom, 0.0, 1.0);
                t = (b * s + ff) / ee;
                if (t < 0.0) { t = 0.0; s = clamp(-c / aa, 0.0, 1.0); }
                else if (t > 1.0) { t = 1.0; s = clamp((b - c) / aa, 0.0, 1.0); }
            }
        }
    }
    float3 xA = a0 + d1 * s;
    float3 xB = b0 + d2 * t;
    float3 mids[8];
    uint feature = uint(BOX_AXIS_EDGE << 24) | (uint(axisA) << 8) | uint(axisB);
    if (!boxAddPt(m, mids, xA, xB, feature))
        boxAddPt(m, mids, boxSupportPoint(boxA, normalAB), boxSupportPoint(boxB, -normalAB), feature);
}

// Returns manifold with normal A→B; count==0 if separated (sep > 0). Margin is unused
// in the SAT reject (kept for call-site parity); it is applied to C0 at emit.
BoxManifold collideBoxesCached(BoxObb boxA, BoxObb boxB, float margin)
{
    BoxManifold m;
    m.normal = float3(0, 1, 0);
    m.count = 0;
    float3 delta = boxB.center - boxA.center;

    bool faceValid = false;
    float faceSep = -1e30;
    float3 faceN = float3(0, 1, 0);
    int faceType = BOX_AXIS_FACE_A;
    int faceIA = 0;
    int faceIB = 0;

    bool edgeValid = false;
    float edgeSep = -1e30;
    float3 edgeN = float3(0, 1, 0);
    int edgeType = BOX_AXIS_EDGE;
    int edgeIA = 0;
    int edgeIB = 0;

    for (int i = 0; i < 3; i++)
        if (!boxTestAxis(boxA, boxB, delta, boxAxis(boxA, i), BOX_AXIS_FACE_A, i, -1, margin,
                         faceValid, faceSep, faceN, faceType, faceIA, faceIB))
            return m;
    for (int i = 0; i < 3; i++)
        if (!boxTestAxis(boxA, boxB, delta, boxAxis(boxB, i), BOX_AXIS_FACE_B, -1, i, margin,
                         faceValid, faceSep, faceN, faceType, faceIA, faceIB))
            return m;
    // Always evaluate edge axes (three-avbd). Skipping them when faceSep<0.01 was a
    // perf shortcut that forced 4-point face manifolds for corner contacts.
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (!boxTestAxis(boxA, boxB, delta, cross(boxAxis(boxA, i), boxAxis(boxB, j)),
                             BOX_AXIS_EDGE, i, j, margin, edgeValid, edgeSep, edgeN, edgeType, edgeIA, edgeIB))
                return m;

    if (!faceValid) return m;

    bool useEdge = false;
    float3 bestN = faceN;
    int bestType = faceType;
    int bestIA = faceIA;
    int bestIB = faceIB;
    // three-avbd FLAG_FACE_BIAS (default on). Non-face-bias no longer cuts Ring
    // contacts after smash/seating fixes (~100k either way); keep oracle pick.
    if (edgeValid)
    {
        if (edgeSep > 0.95 * faceSep + 0.01)
        {
            useEdge = true;
            bestN = edgeN;
            bestType = edgeType;
            bestIA = edgeIA;
            bestIB = edgeIB;
        }
    }
    m.normal = bestN; // A → B
    if (useEdge || bestType == BOX_AXIS_EDGE)
        boxBuildEdgeContact(boxA, boxB, bestIA, bestIB, bestN, m);
    else if (bestType == BOX_AXIS_FACE_A)
        boxBuildFaceManifold(boxA, boxB, true, bestIA, bestN, m);
    else
        boxBuildFaceManifold(boxA, boxB, false, bestIB, bestN, m);
    return m;
}

BoxManifold collideBoxes(float3 cA, float4 qA, float3 hA, float3 cB, float4 qB, float3 hB, float margin)
{
    return collideBoxesCached(makeBoxObb(cA, qA, hA), makeBoxObb(cB, qB, hB), margin);
}

// Up to 8 corner contacts of an OBB against an infinite plane (normal pointing into free space).
// Geometric touch/penetration only (gap ≤ 0); margin goes into C0 at emit (three-avbd).
BoxManifold collideBoxPlane(float3 cA, float4 qA, float3 hA, float3 planeN, float planeW, float margin)
{
    BoxManifold m;
    m.count = 0;
    float3 pn = normalize(planeN);
    m.normal = -pn; // A (box) → B (plane)
    float3 mids[8];
    for (int ix = 0; ix < 2; ix++)
    for (int iy = 0; iy < 2; iy++)
    for (int iz = 0; iz < 2; iz++)
    {
        float3 local = float3(
            ix == 0 ? -hA.x : hA.x,
            iy == 0 ? -hA.y : hA.y,
            iz == 0 ? -hA.z : hA.z);
        float3 corner = transformPoint(cA, qA, local);
        float gap = dot(corner, pn) - planeW;
        if (gap <= 0.0)
        {
            float3 pB = corner - pn * gap; // on the geometric plane
            uint feature = uint(ix) | (uint(iy) << 1) | (uint(iz) << 2) | 0x04000000u;
            boxAddPt(m, mids, corner, pB, feature);
        }
    }
    return m;
}

static const uint BOX_CAPSULE_FEATURE = 0x06000000u;
static const uint CAPSULE_PLANE_FEATURE = 0x05000000u;

float boxSegmentDist2(float3 p, float3 h)
{
    float3 g = p - clamp(p, -h, h);
    return dot(g, g);
}

// Parameter t ∈ [0, 1] of the point of segment p0 + t·d closest to the box
// |x| ≤ h. dist²(t) is a convex piecewise quadratic whose pieces break where a
// coordinate crosses ±h, so the minimum is a piece's stationary point (or a
// breakpoint). Ties keep the smallest t, so the result is deterministic.
float boxSegmentClosestT(float3 p0, float3 d, float3 h)
{
    float ts[8];
    ts[0] = 0.0;
    int n = 1;
    for (int k = 0; k < 3; k++)
    {
        if (abs(d[k]) < 1e-12) continue;
        for (int sgn = -1; sgn <= 1; sgn += 2)
        {
            float t = (float(sgn) * h[k] - p0[k]) / d[k];
            if (t > 0.0 && t < 1.0) { ts[n] = t; n++; }
        }
    }
    ts[n] = 1.0;
    n++;
    for (int i = 1; i < n; i++)
    {
        float v = ts[i];
        int j = i - 1;
        while (j >= 0 && ts[j] > v) { ts[j + 1] = ts[j]; j--; }
        ts[j + 1] = v;
    }
    float bestT = 0.0;
    float bestF = boxSegmentDist2(p0, h);
    for (int i = 0; i + 1 < n; i++)
    {
        float a = ts[i];
        float b = ts[i + 1];
        float3 pm = p0 + d * (0.5 * (a + b));
        float num = 0.0;
        float den = 0.0;
        for (int k = 0; k < 3; k++)
        {
            if (abs(pm[k]) <= h[k]) continue;
            float c = pm[k] > 0.0 ? h[k] : -h[k];
            num += d[k] * (p0[k] - c);
            den += d[k] * d[k];
        }
        float t = den > 0.0 ? clamp(-num / den, a, b) : a;
        float f = boxSegmentDist2(p0 + d * t, h);
        if (f < bestF) { bestF = f; bestT = t; }
    }
    return bestT;
}

// OBB A vs capsule B (segment along local +Y, half length halfLen, swept by
// radius). As box/sphere, but for the segment: the exact closest segment point
// decides the feature. On a face (one axis outside) or through the box (the
// shallowest face for the whole segment) the segment is clipped to the face's
// side slabs and both ends of the clipped part become contacts; on an edge or
// a corner the closest pair is the single contact. Normal A → B. Contacts within
// `spec` of touching are kept (0 unless contact-gap mode).
BoxManifold collideBoxCapsule(float3 cA, float4 qA, float3 hA, float3 cB, float4 qB, float radius, float halfLen, float spec)
{
    BoxManifold m;
    m.count = 0;
    m.normal = float3(0, 1, 0);
    float4 qInv = quatConjugate(qA);
    float3 axis = quatRotate(qB, float3(0, 1, 0)) * halfLen;
    float3 p0 = quatRotate(qInv, cB - axis - cA);
    float3 p1 = quatRotate(qInv, cB + axis - cA);
    float3 d = p1 - p0;
    float3 s = p0 + d * boxSegmentClosestT(p0, d, hA);
    float3 q = clamp(s, -hA, hA);
    float3 gap = s - q;
    float dist = length(gap);
    if (dist > radius + spec) return m;
    float3 mids[8];
    int k = 0;
    float sgn = 1.0;
    if (dist > 1e-8)
    {
        int outside = 0;
        uint code = 0u;
        uint place = 1u;
        for (int j = 0; j < 3; j++)
        {
            if (gap[j] != 0.0) { outside++; k = j; sgn = gap[j] > 0.0 ? 1.0 : -1.0; }
            code += place * (gap[j] < 0.0 ? 0u : (gap[j] > 0.0 ? 2u : 1u));
            place *= 3u;
        }
        if (outside > 1)
        {
            float3 nLocal = gap / dist;
            float3 n = quatRotate(qA, nLocal);
            m.normal = n;
            boxAddPt(m, mids, transformPoint(cA, qA, q), transformPoint(cA, qA, s) - n * radius,
                BOX_CAPSULE_FEATURE | 0x100u | code);
            return m;
        }
    }
    else
    {
        float best = -1e30;
        for (int j = 0; j < 3; j++)
        {
            for (int side = -1; side <= 1; side += 2)
            {
                float sg = float(side);
                float sep = min(sg * p0[j], sg * p1[j]) - hA[j];
                if (sep > best) { best = sep; k = j; sgn = sg; }
            }
        }
    }
    // Face k (sign sgn): clip the segment to the slabs of the other two axes.
    float t0 = 0.0;
    float t1 = 1.0;
    for (int j = 0; j < 3; j++)
    {
        if (j == k) continue;
        if (abs(d[j]) < 1e-12)
        {
            if (abs(p0[j]) > hA[j]) { t0 = 1.0; t1 = 0.0; }
            continue;
        }
        float ta = (-hA[j] - p0[j]) / d[j];
        float tb = (hA[j] - p0[j]) / d[j];
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
    }
    float3 nLocal = float3(0, 0, 0);
    nLocal[k] = sgn;
    float3 n = quatRotate(qA, nLocal);
    m.normal = n;
    uint faceId = uint(k) * 2u + (sgn > 0.0 ? 1u : 0u);
    if (t0 > t1)
    {
        // Round-off at a slab boundary: keep the closest point on the face.
        t0 = boxSegmentClosestT(p0, d, hA);
        t1 = t0;
    }
    for (int e = 0; e < 2; e++)
    {
        if (e == 1 && t1 - t0 < 1e-6) break;
        float3 p = p0 + d * (e == 0 ? t0 : t1);
        float depth = sgn * p[k] - hA[k];
        if (depth > radius + spec) continue;
        float3 onBox = clamp(p, -hA, hA);
        onBox[k] = sgn * hA[k];
        boxAddPt(m, mids, transformPoint(cA, qA, onBox), transformPoint(cA, qA, p) - n * radius,
            BOX_CAPSULE_FEATURE | (faceId << 4) | uint(e));
    }
    return m;
}

// Closest points c1 on segment p1q1 and c2 on segment p2q2 (Ericson 5.1.9).
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

static const uint CAPSULE_CAPSULE_FEATURE = 0x07000000u;
// Segments closer than this to parallel (sine of the angle) share a face-like
// overlap and get two contacts; otherwise they cross at one point.
static const float CAPSULE_PARALLEL_SIN = 0.05;

// Capsule A vs capsule B (segments along local +Y, half lengths hA / hB).
// The closest pair of segment points gives the normal (A → B). Near-parallel
// segments whose projections overlap get one contact at each end of the
// overlap (each paired with its closest point on B), plus the closest pair
// when the segments cross inside the overlap; crossed segments, or capsules
// that are spheres, get the single closest pair. Contacts within
// `spec` of touching are kept (0 unless contact-gap mode).
BoxManifold collideCapsules(float3 cA, float4 qA, float rA, float hA, float3 cB, float4 qB, float rB, float hB, float spec)
{
    BoxManifold m;
    m.count = 0;
    m.normal = float3(0, 1, 0);
    float3 axA = quatRotate(qA, float3(0, 1, 0));
    float3 axB = quatRotate(qB, float3(0, 1, 0));
    float3 a0 = cA - axA * hA;
    float3 a1 = cA + axA * hA;
    float3 b0 = cB - axB * hB;
    float3 b1 = cB + axB * hB;
    float3 pa, pb;
    closestSegments(a0, a1, b0, b1, pa, pb);
    float3 d = pb - pa;
    float dist = length(d);
    float reach = rA + rB;
    if (dist > reach + spec) return m;
    float3 n;
    if (dist > 1e-6) n = d / dist;
    else
    {
        // Axes meet: any normal perpendicular to both, toward B.
        float3 c = cross(axA, axB);
        if (dot(c, c) < 1e-12)
        {
            float3 t2;
            orthonormalBasis(axA, c, t2);
        }
        n = normalize(c);
        if (dot(n, cB - cA) < 0.0) n = -n;
    }
    m.normal = n;
    float3 mids[8];
    if (hA > 1e-6 && hB > 1e-6 && length(cross(axA, axB)) < CAPSULE_PARALLEL_SIN)
    {
        // Overlap of B's projection on A's segment.
        float lenA = 2.0 * hA;
        float tb0 = dot(b0 - a0, axA);
        float tb1 = dot(b1 - a0, axA);
        float t0 = max(0.0, min(tb0, tb1));
        float t1 = min(lenA, max(tb0, tb1));
        if (t1 - t0 > 1e-6)
        {
            float endGap = 1e30;
            for (int e = 0; e < 2; e++)
            {
                float3 p = a0 + axA * (e == 0 ? t0 : t1);
                float3 q, dummy;
                closestSegments(p, p, b0, b1, dummy, q);
                float g = dot(n, q - p) - reach;
                endGap = min(endGap, g);
                if (g > spec) continue;
                boxAddPt(m, mids, p + n * rA, q - n * rB, CAPSULE_CAPSULE_FEATURE | uint(e));
            }
            // Segments crossing inside the overlap are deepest there: keep that pair too.
            if (m.count > 0 && dist - reach >= endGap - 1e-5) return m;
        }
    }
    boxAddPt(m, mids, pa + n * rA, pb - n * rB, CAPSULE_CAPSULE_FEATURE | 0x100u);
    return m;
}

// Capsule A vs plane B: one contact per end-cap sphere closer than `margin`
// (as sphere/plane). Normal A → B (−plane normal).
BoxManifold collideCapsulePlane(float3 cA, float4 qA, float radius, float halfLen, float3 planeN, float planeW, float margin)
{
    BoxManifold m;
    m.count = 0;
    float3 pn = normalize(planeN);
    m.normal = -pn;
    float3 axis = quatRotate(qA, float3(0, 1, 0)) * halfLen;
    float3 mids[8];
    for (int i = 0; i < 2; i++)
    {
        float3 e = i == 0 ? cA - axis : cA + axis;
        float gap = dot(e, pn) - planeW - radius;
        if (gap >= margin) continue;
        float3 pA = e - pn * radius;
        boxAddPt(m, mids, pA, pA - pn * gap, CAPSULE_PLANE_FEATURE | uint(i));
    }
    return m;
}

// Swap a manifold's sides (B → A normal, points exchanged).
BoxManifold flipManifold(BoxManifold mf)
{
    mf.normal = -mf.normal;
    for (int k = 0; k < mf.count; k++)
    {
        float3 tmp = mf.pts[k].pA;
        mf.pts[k].pA = mf.pts[k].pB;
        mf.pts[k].pB = tmp;
    }
    return mf;
}

#endif
