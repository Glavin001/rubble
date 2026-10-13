// Box-box SAT + face clipping, ported from avbd-demo3d/collide.cpp (and three-avbd ref).
// Contact normal is A→B (normalAB). Points are world-space on each surface.

#ifndef ENGINE_PHYSICS_BOX_COLLIDE_SLANG
#define ENGINE_PHYSICS_BOX_COLLIDE_SLANG

// Cap at 4 (face corners). three-avbd allows 8; 4 matches stack manifolds and cuts
// Brick Ring contact traffic ~2× when clipping produced 5–8 points.
static const int BOX_MAX_CONTACTS = 4;
static const int BOX_MAX_POLY = 16;
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
    // three-avbd: keep pairs within collision margin so resting stacks don't drop
    // all contacts on a 1e-4 gap (Jacobi soft contacts often sit slightly open).
    if (separation > margin) return false;
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

int boxClipPoly(float3 inV[16], int inCount, float3 planeN, float planeOffset, out float3 outV[16])
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

    float3 clip0[16];
    float3 clip1[16];
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

    for (int i = 0; i < count && m.count < BOX_MAX_CONTACTS; i++)
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

// Returns manifold with normal A→B; count==0 if separated by more than `margin`.
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
    // three-avbd FLAG_FACE_BIAS (default on). Shallow-edge overrides that cut Ring
    // contacts (~110k→43k) also drift Stack / prevent Columns sleep — leave alone.
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
// Accepts corners within `margin` of the plane; points are geometric (margin goes into C0 at emit).
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
        if (gap < margin)
        {
            float3 pB = corner - pn * gap; // on the geometric plane
            uint feature = uint(ix) | (uint(iy) << 1) | (uint(iz) << 2) | 0x04000000u;
            boxAddPt(m, mids, corner, pB, feature);
        }
    }
    return m;
}

#endif
