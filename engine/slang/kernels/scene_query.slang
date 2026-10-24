// Scene queries shared by the in-step query kernels (wheels, characters,
// projectiles, raycast_batch): ray and convex casts against any collider and
// the grid walks that find their candidates. Include after world_step.slang.
#ifndef ENGINE_KERNELS_SCENE_QUERY_SLANG
#define ENGINE_KERNELS_SCENE_QUERY_SLANG

#include "../physics/vehicle_layout.slang"

// ---------------------------------------------------------------------------
// Ray casts against world colliders (shared by wheels and raycast_batch)
//
// A cast whose origin is already inside a shape reports where the line entered
// it (t ≤ 0) and the surface normal there, not a made-up t = 0 along -d: a wheel
// mount inside a crate must see the crate's real face, which the tyre cone then
// accepts (geometry under the tyre) or leaves to the wheel shell (beside it).
// ---------------------------------------------------------------------------
bool raySphere(float3 o, float3 d, float3 c, float r, out float t, out float3 n)
{
    t = 0.0; n = -d;
    float3 oc = o - c;
    float b = dot(oc, d);
    float cc = dot(oc, oc) - r * r;
    float h = b * b - cc;
    if (cc > 0.0 && (h < 0.0 || b > 0.0)) return false;
    t = -b - sqrt(max(h, 0.0)); // origin inside (cc ≤ 0): the entry behind it, t ≤ 0
    float3 p = o + d * t - c;
    n = dot(p, p) > 1e-12 ? normalize(p) : -d;
    return true;
}

bool rayObb(float3 o, float3 d, BoxObb box, out float t, out float3 n)
{
    t = 0.0; n = -d;
    float3 rel = o - box.center;
    float3 lo = float3(dot(rel, box.axis0), dot(rel, box.axis1), dot(rel, box.axis2));
    float3 ld = float3(dot(d, box.axis0), dot(d, box.axis1), dot(d, box.axis2));
    float tNear = -1e30;
    float tFar = 1e30;
    int nearAxis = -1;
    float nearSign = 0.0;
    for (int k = 0; k < 3; k++)
    {
        float h = box.half[k];
        if (abs(ld[k]) < 1e-9)
        {
            if (abs(lo[k]) > h) return false;
            continue;
        }
        float inv = 1.0 / ld[k];
        float t0 = (-h - lo[k]) * inv;
        float t1 = (h - lo[k]) * inv;
        float s = -1.0;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; s = 1.0; }
        if (t0 > tNear) { tNear = t0; nearAxis = k; nearSign = s; }
        tFar = min(tFar, t1);
        if (tNear > tFar) return false;
    }
    if (tFar < 0.0 || nearAxis < 0) return false;
    t = tNear; // origin inside: the entry behind it, t < 0
    float3 ax = nearAxis == 0 ? box.axis0 : (nearAxis == 1 ? box.axis1 : box.axis2);
    n = ax * nearSign;
    return true;
}

// Ray vs box of half extents `h` rounded by `rad` (sphere sweep vs OBB), box space.
// Inigo Quilez's rounded-box intersector; returns t (< 0 = miss).
float roundedBoxT(float3 ro, float3 rd, float3 h, float rad)
{
    float3 safeD = float3(abs(rd.x) < 1e-9 ? 1e-9 : rd.x, abs(rd.y) < 1e-9 ? 1e-9 : rd.y, abs(rd.z) < 1e-9 ? 1e-9 : rd.z);
    float3 m = 1.0 / safeD;
    float3 nn = m * ro;
    float3 k = abs(m) * (h + rad);
    float3 t1 = -nn - k;
    float3 t2 = -nn + k;
    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);
    if (tN > tF || tF < 0.0) return -1.0;
    float t = tN;
    float3 pos = ro + t * rd;
    float3 sg = float3(pos.x < 0.0 ? -1.0 : 1.0, pos.y < 0.0 ? -1.0 : 1.0, pos.z < 0.0 ? -1.0 : 1.0);
    float3 o = ro * sg;
    float3 dd0 = rd * sg;
    pos *= sg;
    pos -= h;
    pos = max(pos.xyz, pos.yzx);
    if (min(min(pos.x, pos.y), pos.z) < 0.0) return t;
    float3 oc = o - h;
    float3 dd = dd0 * dd0;
    float3 oo = oc * oc;
    float3 od = oc * dd0;
    float ra2 = rad * rad;
    t = 1e20;
    {
        float b = od.x + od.y + od.z;
        float c = oo.x + oo.y + oo.z - ra2;
        float hh = b * b - c;
        if (hh > 0.0) t = -b - sqrt(hh);
    }
    {
        float a = dd.y + dd.z; float b = od.y + od.z; float c = oo.y + oo.z - ra2;
        float hh = b * b - a * c;
        if (hh > 0.0 && a > 1e-12) { hh = (-b - sqrt(hh)) / a; if (hh > 0.0 && hh < t && abs(o.x + dd0.x * hh) < h.x) t = hh; }
    }
    {
        float a = dd.z + dd.x; float b = od.z + od.x; float c = oo.z + oo.x - ra2;
        float hh = b * b - a * c;
        if (hh > 0.0 && a > 1e-12) { hh = (-b - sqrt(hh)) / a; if (hh > 0.0 && hh < t && abs(o.y + dd0.y * hh) < h.y) t = hh; }
    }
    {
        float a = dd.x + dd.y; float b = od.x + od.y; float c = oo.x + oo.y - ra2;
        float hh = b * b - a * c;
        if (hh > 0.0 && a > 1e-12) { hh = (-b - sqrt(hh)) / a; if (hh > 0.0 && hh < t && abs(o.z + dd0.z * hh) < h.z) t = hh; }
    }
    return t > 1e19 ? -1.0 : t;
}

bool sweepObb(float3 o, float3 d, BoxObb box, float rad, out float t, out float3 n)
{
    t = 0.0; n = -d;
    float3 rel = o - box.center;
    float3 lo = float3(dot(rel, box.axis0), dot(rel, box.axis1), dot(rel, box.axis2));
    float3 ld = float3(dot(d, box.axis0), dot(d, box.axis1), dot(d, box.axis2));
    // Origin already inside the rounded box: cast from a point behind it (beyond
    // the box's bounding radius) to find where the line entered, t < 0.
    float3 q = abs(lo) - box.half;
    float back = 0.0;
    if (length(max(q, float3(0, 0, 0))) + min(max(q.x, max(q.y, q.z)), 0.0) < rad)
        back = 2.0 * (length(box.half) + rad);
    float tt = roundedBoxT(lo - ld * back, ld, box.half, rad);
    if (tt < 0.0) return false;
    t = tt - back;
    float3 p = lo + ld * t;
    float3 ln = float3(p.x < 0.0 ? -1.0 : 1.0, p.y < 0.0 ? -1.0 : 1.0, p.z < 0.0 ? -1.0 : 1.0)
        * max(abs(p) - box.half, float3(0, 0, 0));
    if (dot(ln, ln) < 1e-12)
    {
        float3 a = abs(p) - box.half;
        ln = a.x > a.y && a.x > a.z ? float3(p.x < 0.0 ? -1.0 : 1.0, 0, 0)
            : (a.y > a.z ? float3(0, p.y < 0.0 ? -1.0 : 1.0, 0) : float3(0, 0, p.z < 0.0 ? -1.0 : 1.0));
    }
    ln = normalize(ln);
    n = box.axis0 * ln.x + box.axis1 * ln.y + box.axis2 * ln.z;
    return true;
}

bool rayCapsuleOutside(float3 o, float3 d, float3 c, float3 axis, float halfLen, float r, out float t, out float3 n)
{
    // Closest approach between the ray and the segment, then a sphere test there.
    float3 pa = c - axis * halfLen;
    float3 ba = axis * (2.0 * halfLen);
    float3 oa = o - pa;
    float baba = dot(ba, ba);
    float bard = dot(ba, d);
    float baoa = dot(ba, oa);
    float rdoa = dot(d, oa);
    float oaoa = dot(oa, oa);
    float a = baba - bard * bard;
    float b = baba * rdoa - baoa * bard;
    float cc = baba * oaoa - baoa * baoa - r * r * baba;
    float h = b * b - a * cc;
    t = 0.0; n = -d;
    if (h >= 0.0 && a > 1e-12)
    {
        float tt = (-b - sqrt(h)) / a;
        float y = baoa + tt * bard;
        if (y > 0.0 && y < baba && tt >= 0.0)
        {
            t = tt;
            float3 p = o + d * t;
            n = normalize(p - (pa + ba * (y / baba)));
            return true;
        }
    }
    float tA, tB;
    float3 nA, nB;
    bool hA = raySphere(o, d, pa, r, tA, nA);
    bool hB = raySphere(o, d, pa + ba, r, tB, nB);
    if (hA && (!hB || tA <= tB)) { t = tA; n = nA; return true; }
    if (hB) { t = tB; n = nB; return true; }
    return false;
}

bool rayCapsule(float3 o0, float3 d, float3 c, float3 axis, float halfLen, float r, out float t, out float3 n)
{
    // Origin inside: cast from behind the capsule and report the entry (t < 0).
    float s0 = clamp(dot(o0 - c, axis), -halfLen, halfLen);
    float3 off = o0 - (c + axis * s0);
    float back = dot(off, off) < r * r ? 2.0 * (halfLen + r) : 0.0;
    float3 o = o0 - d * back;
    bool hit = rayCapsuleOutside(o, d, c, axis, halfLen, r, t, n);
    t -= back;
    return hit;
}

// Sweep a sphere of radius `rad` (0 = ray) from o along d against a collider.
bool rayCollider(uint ci, float3 o, float3 d, float rad, out float t, out float3 n)
{
    t = 0.0; n = -d;
    Collider col = gColliders[ci];
    BodyState b = gBodies[col.body];
    float3 c = transformPoint(b.position, b.orientation, col.localPos);
    if (col.shape == SHAPE_SPHERE) return raySphere(o, d, c, col.params.x + rad, t, n);
    if (col.shape == SHAPE_BOX)
        return rad > 0.0 ? sweepObb(o, d, worldObbFromCache(ci), rad, t, n) : rayObb(o, d, worldObbFromCache(ci), t, n);
    if (col.shape == SHAPE_CAPSULE)
    {
        float4 q = quatMul(b.orientation, col.localRot);
        return rayCapsule(o, d, c, quatRotate(q, float3(0, 1, 0)), col.params.y, col.params.x + rad, t, n);
    }
    if (col.shape == SHAPE_PLANE)
    {
        float3 pn = normalize(col.params.xyz);
        float denom = dot(pn, d);
        float dist = dot(pn, o) - col.params.w - rad;
        n = pn;
        if (denom >= -1e-9)
        {
            // Parallel or leaving: behind the plane the whole line is inside.
            t = -1e9;
            return dist <= 0.0;
        }
        t = -dist / denom; // origin behind the plane: t ≤ 0
        n = pn;
        return true;
    }
    // Convex hulls: exact against the face planes (ray), or a GJK sphere cast.
    float4 q = quatMul(b.orientation, col.localRot);
    if (rad <= 0.0) return rayHull(o, d, col, c, q, t, n);
    Collider ball = col;
    ball.shape = SHAPE_SPHERE;
    ball.params = float4(rad, 0.0, 0.0, rad);
    // Any hit lies within the centre distance plus both bounds.
    float reach = length(c - o) + col.params.w + rad + 1e-3;
    return shapeCastCollider(ball, o, float4(0, 0, 0, 1), d, reach, ci, t, n);
}

// Ray against a convex hull collider at pose (pos, rot): the slab test over its
// outward face planes (n·x ≤ w inside). An origin inside reports the entry
// behind it (t ≤ 0), as the other casts do.
bool rayHull(float3 o, float3 d, Collider c, float3 pos, float4 rot, out float t, out float3 n)
{
    t = 0.0;
    n = -d;
    HullRef h = hullOf(c);
    float4 qi = quatConjugate(rot);
    float3 lo = quatRotate(qi, o - pos);
    float3 ld = quatRotate(qi, d);
    float tNear = -1e30, tFar = 1e30;
    float3 nNear = -ld;
    for (uint f = 0u; f < h.nf; f++)
    {
        float4 pl = hullPlane(h, f);
        float denom = dot(pl.xyz, ld);
        float num = pl.w - dot(pl.xyz, lo);
        if (abs(denom) < 1e-12)
        {
            if (num < 0.0) return false; // parallel and outside this face
        }
        else
        {
            float tt = num / denom;
            if (denom < 0.0)
            {
                if (tt > tNear) { tNear = tt; nNear = pl.xyz; }
            }
            else tFar = min(tFar, tt);
        }
        if (tNear > tFar) return false;
    }
    if (tFar < 0.0 || tNear <= -1e29) return false;
    t = tNear;
    n = quatRotate(rot, nNear);
    return true;
}

// Sweep a convex shape (`cs`: a sphere, capsule, box or hull collider record
// placed at pose (p, q)) along unit d, up to maxT, against collider ci: the GJK
// cast of ccd_step.slang. A start already inside reports the entry behind it
// (t ≤ 0): the cast starts beyond both shapes' reach back along -d. n is the
// target's surface normal at the hit (toward the caster).
bool shapeCastCollider(Collider cs, float3 p, float4 q, float3 d, float maxT, uint ci, out float t, out float3 n)
{
    t = 0.0;
    n = -d;
    Collider col = gColliders[ci];
    BodyState b = gBodies[col.body];
    if (col.shape == SHAPE_PLANE)
    {
        float3 pn = normalize(col.params.xyz);
        float dist = dot(pn, shapeSupport(cs, p, q, -pn)) - col.params.w;
        float denom = dot(pn, d);
        n = pn;
        if (denom >= -1e-9)
        {
            t = -1e9;
            return dist <= 0.0;
        }
        t = -dist / denom;
        return true;
    }
    float3 pc; float4 qc; colliderPose(col, b, pc, qc);
    float mu;
    float3 nAB;
    float eps = 1.0e-4 + 1.0e-6 * maxT;
    uint status = gjkCast(cs, float3(0, 0, 0), q, col, pc - p, qc, float3(0, 0, 0), d * maxT, 1.0, eps, mu, nAB);
    if (status == CAST_HIT)
    {
        t = mu * maxT;
        n = -nAB;
        return true;
    }
    if (status != CAST_OVERLAP) return false;
    // Started inside: cast again from beyond both shapes' reach back along -d
    // for the entry (t < 0).
    float reachA = length(shapeSupport(cs, float3(0, 0, 0), q, d)) + length(shapeSupport(cs, float3(0, 0, 0), q, -d));
    float boundB = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    if (col.shape == SHAPE_CAPSULE) boundB = col.params.x + col.params.y;
    float back = reachA + boundB + length(pc - p) + 1e-3;
    float span = maxT + back;
    status = gjkCast(cs, float3(0, 0, 0), q, col, pc - p, qc, -d * back, d * span, 1.0, eps + 1.0e-6 * span, mu, nAB);
    if (status != CAST_HIT) { t = 0.0; n = -d; return true; }
    t = mu * span - back;
    n = -nAB;
    return true;
}

bool segmentHitsAabb(float3 mn, float3 mx, Aabb ab)
{
    return !(mx.x < ab.mn.x || ab.mx.x < mn.x || mx.y < ab.mn.y || ab.mx.y < mn.y ||
             mx.z < ab.mn.z || ab.mx.z < mn.z);
}

// Whether a hit can carry the tyre: its normal lies within the tyre cone around
// the suspension axis (coneCos; ≤ -1 accepts every hit) and, for a sphere cast
// hit above the full-bump position (t < lmin), the contact also lies inside that
// cone seen from the wheel centre at full bump. Hits beside the tyre (crate
// faces, kerb walls, a mount inside a crate) belong to the wheel shell, and must
// not hide the support further down the suspension travel.
bool tyreAccepts(float3 o, float3 d, float t, float3 n, float rad, float coneCos, float lmin)
{
    if (dot(n, d) > 0.0) n = -n;
    if (dot(n, -d) < coneCos) return false;
    if (rad > 0.0 && t < lmin)
    {
        float3 v = (o + d * t - n * rad) - (o + d * lmin);
        float lv = length(v);
        if (lv > 1e-4 && dot(v / lv, d) < coneCos) return false;
    }
    return true;
}

// Negative collision group of the body a query starts from (0 = none): a soft
// vehicle's rigid core shares it with its node spheres, so its wheels ignore
// their own bodywork. Set per query before scanning.
static int gRayExcludeGroup = 0;

int bodyNegativeGroup(uint body)
{
    uint c = firstColliderOfBody(body);
    if (c >= gCounters[0].colliderCount || gColliders[c].body != body) return 0;
    int g = int(gColliders[c].material) >> 16;
    return g < 0 ? g : 0;
}

bool rayIgnores(Collider col, uint excludeBody)
{
    if (col.body == excludeBody) return true;
    return gRayExcludeGroup != 0 && (int(col.material) >> 16) == gRayExcludeGroup;
}

// One workgroup per query: lanes stride the colliders, then a min-reduction.
groupshared float gsRayT[64];
groupshared uint gsRayC[64];

void rayLaneScan(uint lane, bool live, float3 o, float3 d, float maxT, uint excludeBody, float rad,
    float coneCos, float lmin)
{
    float bestT = 1e30;
    uint bestC = NO_INDEX;
    uint nCols = live ? gCounters[0].colliderCount : 0u;
    float3 e = o + d * maxT;
    float3 mn = min(o, e) - rad;
    float3 mx = max(o, e) + rad;
    uint base = HOST_S(1);
    for (uint ci = lane; ci < nCols; ci += 64u)
    {
        Collider col = gColliders[ci];
        if (rayIgnores(col, excludeBody)) continue;
        if (col.shape != SHAPE_PLANE && !segmentHitsAabb(mn, mx, gAabbs[base + ci])) continue;
        float t;
        float3 n;
        if (rayCollider(ci, o, d, rad, t, n) && t <= maxT && t < bestT && tyreAccepts(o, d, t, n, rad, coneCos, lmin))
        {
            bestT = t;
            bestC = ci;
        }
    }
    gsRayT[lane] = bestT;
    gsRayC[lane] = bestC;
}

void rayTestCollider(uint ci, float3 o, float3 d, float maxT, uint excludeBody, float rad,
    float coneCos, float lmin, float3 mn, float3 mx, uint base, inout float bestT, inout uint bestC)
{
    Collider col = gColliders[ci];
    if (rayIgnores(col, excludeBody)) return;
    if (col.shape != SHAPE_PLANE && !segmentHitsAabb(mn, mx, gAabbs[base + ci])) return;
    float t;
    float3 n;
    if (rayCollider(ci, o, d, rad, t, n) && t <= maxT && t < bestT && tyreAccepts(o, d, t, n, rad, coneCos, lmin))
    {
        bestT = t;
        bestC = ci;
    }
}

// Normal of the surface a contact point lies on (Jolt's surface normal, as
// opposed to the contact normal): for a box or hull, the face whose plane
// holds the point (within tol) that is most aligned with `up` among those
// facing the contact normal's side; at an edge or corner the contact normal
// tilts, but the face the point lies on does not. Smooth shapes and planes:
// the contact normal.
float3 surfaceNormal(uint ci, float3 p, float3 contactN, float3 up, float tol)
{
    Collider col = gColliders[ci];
    if (col.shape != SHAPE_BOX && col.shape != SHAPE_CONVEX) return contactN;
    BodyState b = gBodies[col.body];
    float3 pc; float4 qc; colliderPose(col, b, pc, qc);
    float3 l = quatRotate(quatConjugate(qc), p - pc);
    float3 best = contactN;
    float bestUp = -2.0;
    if (col.shape == SHAPE_BOX)
    {
        float3 h = col.params.xyz;
        for (uint k = 0u; k < 6u; k++)
        {
            uint axis = k >> 1;
            float sgn = (k & 1u) != 0u ? 1.0 : -1.0;
            float3 nl = float3(0, 0, 0);
            nl[axis] = sgn;
            if (abs(sgn * l[axis] - h[axis]) > tol) continue;
            float3 nw = quatRotate(qc, nl);
            if (dot(nw, contactN) <= 0.0) continue;
            if (dot(nw, up) > bestUp) { bestUp = dot(nw, up); best = nw; }
        }
        return best;
    }
    HullRef hr = hullOf(col);
    for (uint f = 0u; f < hr.nf; f++)
    {
        float4 pl = hullPlane(hr, f);
        if (abs(dot(pl.xyz, l) - pl.w) > tol) continue;
        float3 nw = quatRotate(qc, pl.xyz);
        if (dot(nw, contactN) <= 0.0) continue;
        if (dot(nw, up) > bestUp) { bestUp = dot(nw, up); best = nw; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Grid walk: every collider whose bounds can touch a swept shape, through this
// step's broadphase (valid between broadphase_query and the contact hash, which
// reuses gConstraintIds). Lanes take the cells within reach of the segment on
// the uniform grid (both split tables) or every occupied hierarchical level
// (both rest tables), then the plane, oversized and fast lists. Each collider
// is visited once (true cell and level); too many cells fall back to the full
// scan, so no collider is ever missed.
// ---------------------------------------------------------------------------
interface IColliderVisitor
{
    [mutating] void visit(uint ci);
};

static const uint WALK_MAX_CELLS = 4096u;

int3 walkSpan(float3 mn, float3 mx, float cs, float pad, out int3 lo)
{
    lo = int3(floor((mn - pad) / cs));
    int3 hi = int3(floor((mx + pad) / cs));
    return hi - lo + int3(1, 1, 1);
}

uint walkLevelCells(float3 mn, float3 mx, uint level)
{
    float cs = hierCellSize(level);
    int3 lo;
    int3 sp = walkSpan(mn, mx, cs, 0.5 * cs, lo);
    return uint(sp.x) * uint(sp.y) * uint(sp.z);
}

void sceneWalk<V : IColliderVisitor>(uint lane, bool live, float3 a, float3 b, float rad, bool fullScan, inout V v)
{
    uint nColliders = gCounters[0].colliderCount;
    float3 mn = min(a, b) - rad;
    float3 mx = max(a, b) + rad;
    bool hier = HOST_S(18) != 0u;
    uint occupied = hier ? ALOAD(gScratchAtomics[HIER_LEVEL_MASK_SLOT]) : 1u;
    uint levels = hier ? HOST_S(18) : 1u;
    // Cells per level (uniform: one level padded by a cell, the entries' reach).
    float csU = max(PARAMS.gridCellSize, 0.5);
    uint total = 0u;
    if (live && !fullScan)
    {
        if (!hier)
        {
            int3 lo;
            int3 sp = walkSpan(mn, mx, csU, csU, lo);
            total = uint(sp.x) * uint(sp.y) * uint(sp.z);
        }
        else
        {
            for (uint L = 0u; L < levels; L++)
                if (((occupied >> L) & 1u) != 0u) total += walkLevelCells(mn, mx, L);
        }
    }
    if (live && (fullScan || total > WALK_MAX_CELLS))
    {
        for (uint ci = lane; ci < nColliders; ci += 64u) v.visit(ci);
        return;
    }
    uint base = HOST_S(1);
    if (live)
    {
        for (uint k = lane; k < total; k += 64u)
        {
            // Which level and cell this lane's index names.
            uint rem = k, level = 0u;
            float cs = csU, pad = csU;
            if (hier)
            {
                for (; level < levels; level++)
                {
                    if (((occupied >> level) & 1u) == 0u) continue;
                    uint n = walkLevelCells(mn, mx, level);
                    if (rem < n) break;
                    rem -= n;
                }
                cs = hierCellSize(level);
                pad = 0.5 * cs;
            }
            int3 lo;
            int3 sp = walkSpan(mn, mx, cs, pad, lo);
            int3 cell = lo + int3(int(rem % uint(sp.x)), int((rem / uint(sp.x)) % uint(sp.y)), int(rem / uint(sp.x * sp.y)));
            float3 cc = (float3(cell) + 0.5) * cs;
            if (segmentPointDistance(a, b, cc) > rad + 0.8660254 * cs + 1.7320508 * pad) continue;
            if (!hier)
            {
                uint tables = gridSplit() ? 2u : 1u;
                for (uint t = 0u; t < tables; t++)
                {
                    uint link = ALOAD(gConstraintIds[t * gridHashSize() + gridHash(cell.x, cell.y, cell.z)]);
                    uint guard = 0u;
                    while (link != 0xFFFFFFFFu && guard < 4096u)
                    {
                        uint cur = link & GRID_COLLIDER_MASK;
                        Aabb ab = gAabbs[base + cur];
                        link = gridSplit() ? asuint(ab._pad1) : gColorLists[cur];
                        guard++;
                        if (all(int3(floor((0.5 * (ab.mn + ab.mx)) / cs)) == cell)) v.visit(cur);
                    }
                }
            }
            else
            {
                uint passes = restTables() ? 2u : 1u;
                for (uint t = 0u; t < passes; t++)
                {
                    bool rest = restTables() && t == 1u;
                    uint heads = restTables() ? (rest ? restHeads() : awakeHeads()) : 0u;
                    uint j = ALOAD(gConstraintIds[heads + hierHash(cell, level)]);
                    uint guard = 0u;
                    while (j != 0xFFFFFFFFu && guard < nColliders)
                    {
                        guard++;
                        Aabb ab = gAabbs[base + j];
                        uint next = rest ? ALOAD(gConstraintIds[restLinks() + j]) : asuint(ab._pad1);
                        if (hierLevel(ab._pad0) == level && all(hierCell(ab, level) == cell) &&
                            !(rest && bodyMovingOrKinematic(gBodies[gColliders[j].body])))
                            v.visit(j);
                        j = next;
                    }
                }
            }
        }
        // Lists: planes, oversized colliders, fast colliders (not in the grid).
        uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
        uint nLargeDyn = hier ? 0u : min(ALOAD(gScratchAtomics[25]), 16u);
        uint nLarge = hier ? min(ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT]), HIER_LARGE_MAX) : largeStaticCount();
        uint nFast = fastCount();
        uint nList = nPlanes + nLargeDyn + nLarge + nFast;
        for (uint k = lane; k < nList; k += 64u)
        {
            uint ci;
            if (k < nPlanes) ci = ALOAD(gScratchAtomics[43u + k]);
            else if (k < nPlanes + nLargeDyn) ci = ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k - nPlanes]);
            else if (k < nPlanes + nLargeDyn + nLarge)
            {
                uint q = k - nPlanes - nLargeDyn;
                ci = hier ? ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + q]) : largeStaticCollider(q);
            }
            else ci = ALOAD(gScratchAtomics[FAST_BASE + k - nPlanes - nLargeDyn - nLarge]);
            if (ci < nColliders) v.visit(ci);
        }
    }
}

// Collision filter of a query: a body to ignore (the caster's own) and the
// collider categories it hits (bits 0..15 of Collider.generation when the
// scene has filters; 0 = every category).
struct QueryFilter
{
    uint excludeBody;
    uint mask;
};

bool queryAccepts(Collider col, QueryFilter f)
{
    if (col.body == f.excludeBody) return false;
    if (f.mask == 0u || (HOST_S(30) & 1u) == 0u) return true;
    uint cat = col.generation == 0u ? 0xFFFFu : (col.generation & 0xFFFFu);
    return (cat & f.mask) != 0u;
}

struct RayVisitor : IColliderVisitor
{
    float3 o; float3 d; float maxT; float minT; float rad; float coneCos; float lmin;
    float3 mn; float3 mx;
    QueryFilter filter;
    float bestT; uint bestC;
    [mutating] void visit(uint ci)
    {
        Collider col = gColliders[ci];
        if (!queryAccepts(col, filter)) return;
        if (col.shape != SHAPE_PLANE && !segmentHitsAabb(mn, mx, gAabbs[HOST_S(1) + ci])) return;
        float t;
        float3 n;
        if (rayCollider(ci, o, d, rad, t, n) && t <= maxT && t >= minT && t < bestT && tyreAccepts(o, d, t, n, rad, coneCos, lmin))
        {
            bestT = t;
            bestC = ci;
        }
    }
};

// A ray (rad 0) or sphere cast through the grid walk, filtered; lanes write
// their best hit to gsRayT / gsRayC (rayReduce then finds the nearest).
// minT: hits nearer than this are ignored (projectiles: 0, so a cast leaving a
// collider it is inside does not hit it again; wheels: −∞, the entry behind).
void rayLaneWalk(uint lane, bool live, float3 o, float3 d, float maxT, float rad, float coneCos, float lmin,
    QueryFilter filter, bool fullScan, float minT = -1e30)
{
    RayVisitor v;
    v.o = o; v.d = d; v.maxT = maxT; v.minT = minT; v.rad = rad; v.coneCos = coneCos; v.lmin = lmin;
    float3 e = o + d * maxT;
    v.mn = min(o, e) - rad;
    v.mx = max(o, e) + rad;
    v.filter = filter;
    v.bestT = 1e30;
    v.bestC = NO_INDEX;
    sceneWalk(lane, live, o, e, rad, fullScan, v);
    gsRayT[lane] = v.bestT;
    gsRayC[lane] = v.bestC;
}

// Same query through this step's broadphase grid (valid between broadphase_query
// and the contact hash, which reuses gConstraintIds): lanes take the cells under
// the swept segment (padded by one cell, the largest small radius), then the
// plane / large-dynamic / large-static lists and the endpoint probes. Duplicates
// from hash aliasing only re-test a collider. Falls back to the full scan when
// the segment spans too many cells.
static const uint RAY_GRID_MAX_CELLS = 512u;

void rayLaneScanGrid(uint lane, bool live, float3 o, float3 d, float maxT, uint excludeBody, float rad,
    float coneCos, float lmin)
{
    QueryFilter f;
    f.excludeBody = excludeBody;
    f.mask = 0u;
    rayLaneWalk(lane, live, o, d, maxT, rad, coneCos, lmin, f, false);
}

void rayReduce(uint lane)
{
    GROUP_SYNC();
    for (uint s = 32u; s > 0u; s >>= 1)
    {
        if (lane < s && gsRayT[lane + s] < gsRayT[lane])
        {
            gsRayT[lane] = gsRayT[lane + s];
            gsRayC[lane] = gsRayC[lane + s];
        }
        GROUP_SYNC();
    }
}

#endif
