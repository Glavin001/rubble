// Vehicle controller kernels (raycast wheels solved inside AVBD).
//
// Attached wheels are not bodies. Each wheel contributes rows that act only on
// its chassis: a soft one-sided suspension (implicit spring-damper), a compliant
// bump stop, an anti-roll pair row, and hard AVBD longitudinal/lateral friction
// rows bounded by a friction ellipse μ(slip)·Fn. Wheel spin is an extra 1-DoF
// variable y (rotation this step) coupled through the longitudinal row; it is
// condensed into the chassis 6×6 block with an exact Schur complement and
// back-substituted after every primal pass. The block is added to the chassis
// primal by primalAccumulate (BODY_VEHICLE), so vehicles add no graph edges.
//
// Step placement (host):
//   apply_commands → vehicle_input → build_aabbs → wheel_query → … → predict_bodies
//   → vehicle_prepare → per iteration { primal colors → solve_dual → vehicle_iterate }
//   → extract_velocity_sleep → vehicle_post → sleep gate/commit
//
// The ground is treated as moving-but-given during the step (its displacement is
// explicit); vehicle_post hands the reaction back to dynamic grounds as a velocity
// change for the next step. Brakes, handbrake and rolling resistance share one
// bounded servo row on y, so locked wheels and parked cars hold positionally.

#include "world_step.slang"
#include "../physics/vehicle_layout.slang"

BIND(31) RWStructuredBuffer<float4> gVehicleData; // 29, 30: convex_step

// ---------------------------------------------------------------------------
// Record access
// ---------------------------------------------------------------------------
float4 VD(uint q) { return gVehicleData[q]; }
uint4 VU(uint q) { return asuint(gVehicleData[q]); }
void VSet(uint q, float4 v) { gVehicleData[q] = v; }
void VSetU(uint q, uint4 v) { gVehicleData[q] = asfloat(v); }

uint vehicleCount() { return VU(VH_COUNTS).z; }
uint wheelCountTotal() { return VU(VH_COUNTS).w; }
uint vehRec(uint v) { return VU(VH_BASES).x + v * VEH_STRIDE; }
uint wheelRec(uint w) { return VU(VH_BASES).y + w * WHEEL_STRIDE; }
uint vehicleAabbBlock(uint v) { return VU(VH_BASES).w + 4u * v; }
uint partRec(uint p) { return VU(VH_POOLS).z + p * PART_STRIDE; }

static const float TWO_PI = 6.28318530718;
static const float RAD_PER_S_TO_RPM = 9.54929658551; // 60 / 2π

float4 quatAxisAngle(float3 axis, float angle)
{
    float h = 0.5 * angle;
    return float4(axis * sin(h), cos(h));
}

float moveToward(float cur, float target, float maxDelta)
{
    if (maxDelta <= 0.0) return target;
    float d = target - cur;
    if (abs(d) <= maxDelta) return target;
    return cur + sign(d) * maxDelta;
}

// World inverse inertia applied to a vector (diagonal body-space inverse inertia).
float3 applyInvInertiaWorld(BodyState b, float3 v)
{
    float3 local = quatRotate(quatConjugate(b.orientation), v);
    return quatRotate(b.orientation, local * b.invInertia);
}

// ---------------------------------------------------------------------------
// Piecewise-linear curves: handle = startPoint | count << 16 (NO_INDEX = 1.0)
// ---------------------------------------------------------------------------
float2 curvePoint(uint i)
{
    float4 q = VD(VU(VH_BASES).z + (i >> 1));
    return (i & 1u) == 0u ? q.xy : q.zw;
}

float curveEval(uint h, float x)
{
    if (h == NO_INDEX) return 1.0;
    uint start = h & 0xFFFFu;
    uint count = h >> 16;
    if (count == 0u) return 1.0;
    float2 prev = curvePoint(start);
    if (x <= prev.x) return prev.y;
    for (uint i = 1u; i < count; i++)
    {
        float2 p = curvePoint(start + i);
        if (x <= p.x)
        {
            float t = (x - prev.x) / max(p.x - prev.x, 1e-9);
            return lerp(prev.y, p.y, t);
        }
        prev = p;
    }
    return prev.y;
}

// x of the curve's maximum (the slip where grip peaks).
float curvePeakX(uint h)
{
    if (h == NO_INDEX) return 0.0;
    uint start = h & 0xFFFFu;
    uint count = h >> 16;
    float m = -1e30, x = 0.0;
    for (uint i = 0u; i < count; i++)
    {
        float2 p = curvePoint(start + i);
        if (p.y > m) { m = p.y; x = p.x; }
    }
    return x;
}

float curvePeak(uint h)
{
    if (h == NO_INDEX) return 1.0;
    uint start = h & 0xFFFFu;
    uint count = h >> 16;
    float m = 0.0;
    for (uint i = 0u; i < count; i++) m = max(m, curvePoint(start + i).y);
    return count == 0u ? 1.0 : m;
}

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
    // Convex hulls (fracture path): bounding sphere.
    return raySphere(o, d, c, col.params.w + rad, t, n);
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
    float bestT = 1e30;
    uint bestC = NO_INDEX;
    float3 e = o + d * maxT;
    float3 mn = min(o, e) - rad;
    float3 mx = max(o, e) + rad;
    uint base = HOST_S(1);
    float cs = max(PARAMS.gridCellSize, 0.5);
    int3 lo = int3(floor((mn - cs) / cs));
    int3 hi = int3(floor((mx + cs) / cs));
    int3 span = hi - lo + int3(1, 1, 1);
    uint nCells = live ? uint(span.x * span.y * span.z) : 0u;
    if (nCells > RAY_GRID_MAX_CELLS)
    {
        uint nCols = gCounters[0].colliderCount;
        for (uint ci = lane; ci < nCols; ci += 64u)
            rayTestCollider(ci, o, d, maxT, excludeBody, rad, coneCos, lmin, mn, mx, base, bestT, bestC);
    }
    else
    {
        uint tables = gridSplit() ? 2u : 1u;
        uint total = nCells * tables;
        for (uint k = lane; k < total; k += 64u)
        {
            uint table = k / max(nCells, 1u);
            uint idx = k - table * nCells;
            int3 cell = lo + int3(int(idx % uint(span.x)), int((idx / uint(span.x)) % uint(span.y)),
                int(idx / uint(span.x * span.y)));
            uint link = ALOAD(gConstraintIds[table * gridHashSize() + gridHash(cell.x, cell.y, cell.z)]);
            uint guard = 0u;
            while (link != 0xFFFFFFFFu && guard < 4096u)
            {
                uint cur = link & GRID_COLLIDER_MASK;
                link = gridSplit() ? asuint(gAabbs[base + cur]._pad1) : gColorLists[cur];
                guard++;
                rayTestCollider(cur, o, d, maxT, excludeBody, rad, coneCos, lmin, mn, mx, base, bestT, bestC);
            }
        }
        if (live)
        {
            uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
            uint nLarge = min(ALOAD(gScratchAtomics[25]), 16u);
            uint nStatic = largeStaticCount();
            uint nList = nPlanes + nLarge + nStatic + 2u;
            uint nColliders = gCounters[0].colliderCount;
            for (uint k = lane; k < nList; k += 64u)
            {
                uint ci;
                if (k < nPlanes) ci = ALOAD(gScratchAtomics[43u + k]);
                else if (k < nPlanes + nLarge) ci = ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k - nPlanes]);
                else if (k < nPlanes + nLarge + nStatic) ci = largeStaticCollider(k - nPlanes - nLarge);
                else ci = k + 2u == nList ? 0u : nColliders - 1u;
                if (ci < nColliders)
                    rayTestCollider(ci, o, d, maxT, excludeBody, rad, coneCos, lmin, mn, mx, base, bestT, bestC);
            }
        }
    }
    gsRayT[lane] = bestT;
    gsRayC[lane] = bestC;
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

// ---------------------------------------------------------------------------
// Wheel frame helpers
// ---------------------------------------------------------------------------
float3 vehicleUpLocal(uint vr) { return VD(vr + V_UP).xyz; }
float3 vehicleFwdLocal(uint vr) { return VD(vr + V_FWD).xyz; }
float3 axleLocal(uint vr) { return normalize(cross(vehicleUpLocal(vr), vehicleFwdLocal(vr))); }

float4 wheelSteerRot(uint wr, float steer)
{
    return quatAxisAngle(normalize(VD(wr + W_STEERAX).xyz), steer);
}

// Mass properties removed from the chassis when a wheel or part leaves it.
void chassisRemoveMass(uint chassis, uint vr, float m, float3 iLocal, float3 com)
{
    float4 mass = VD(vr + V_MASS);
    float3 d2 = com * com;
    float3 parallel = m * float3(d2.y + d2.z, d2.x + d2.z, d2.x + d2.y);
    float newM = max(mass.x - m, 0.05 * mass.x);
    float3 newI = max(mass.yzw - iLocal - parallel, 0.05 * mass.yzw);
    VSet(vr + V_MASS, float4(newM, newI));
    gBodies[chassis].invMass = 1.0 / newM;
    gBodies[chassis].invInertia = 1.0 / newI;
}

void chassisAddMass(uint chassis, uint vr, float m, float3 iLocal, float3 com)
{
    float4 mass = VD(vr + V_MASS);
    float3 d2 = com * com;
    float3 parallel = m * float3(d2.y + d2.z, d2.x + d2.z, d2.x + d2.y);
    float newM = mass.x + m;
    float3 newI = mass.yzw + iLocal + parallel;
    VSet(vr + V_MASS, float4(newM, newI));
    gBodies[chassis].invMass = 1.0 / newM;
    gBodies[chassis].invInertia = 1.0 / newI;
}

float3 wheelInertiaLocal(uint vr, uint wr)
{
    float wheelMass = VD(wr + W_MISC).x;
    float radius = VD(wr + W_HP).w;
    float width = VD(wr + W_DIR).w;
    float im = wheelMass * (3.0 * radius * radius + width * width) / 12.0;
    float ia = 0.5 * wheelMass * radius * radius;
    float3 axl = axleLocal(vr);
    return float3(im, im, im) + (ia - im) * axl * axl;
}

// Severed pieces ignore their former chassis until they have separated.
static const float SEVER_GRACE = 0.5;

void setColliderFilter(uint start, uint count, uint value)
{
    for (uint k = 0u; k < count; k++) gColliders[start + k].generation = value;
}

// aabbOverlap: world_step.slang.

// True while any collider in [start, start+count) overlaps an attached chassis part.
bool overlapsChassis(uint vr, uint start, uint count)
{
    uint base = HOST_S(1);
    uint4 pool2 = VU(vr + V_POOL2);
    for (uint k = 0u; k < count; k++)
    {
        Aabb a = gAabbs[base + start + k];
        for (uint pi = 0u; pi < pool2.y; pi++)
        {
            uint pr = partRec(pool2.x + pi);
            if (asuint(VD(pr + P_COM).w) != PS_ATTACHED) continue;
            uint4 pc = VU(pr + P_CFG);
            for (uint c = 0u; c < pc.y; c++)
                if (aabbOverlap(a, gAabbs[base + pc.x + c])) return true;
        }
    }
    return false;
}

void updateSeverGrace(uint vr, uint4 cfg, float dt)
{
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        float4 gv = VD(wr + W_GV);
        if ((VU(wr + W_C0).z & WD_DETACHED) == 0u || gv.w <= 0.0) continue;
        float grace = gv.w - dt;
        uint4 park = VU(wr + W_PARK);
        if (grace <= 0.0)
        {
            if (overlapsChassis(vr, park.y, park.z)) grace = dt;
            else setColliderFilter(park.y, park.z, 0u);
        }
        VSet(wr + W_GV, float4(gv.xyz, max(grace, 0.0)));
    }
    uint4 pool2 = VU(vr + V_POOL2);
    for (uint pi = 0u; pi < pool2.y; pi++)
    {
        uint pr = partRec(pool2.x + pi);
        float4 park = VD(pr + P_PARK);
        if (park.z <= 0.0 || asuint(VD(pr + P_COM).w) != PS_DETACHED) continue;
        float grace = park.z - dt;
        uint4 pc = VU(pr + P_CFG);
        if (grace <= 0.0)
        {
            if (overlapsChassis(vr, pc.x, pc.y)) grace = dt;
            else setColliderFilter(pc.x, pc.y, 0u);
        }
        VSet(pr + P_PARK, float4(park.xy, max(grace, 0.0), park.w));
    }
}

// Activate a parked body as a free rigid body with the given pose and velocity.
void activateParkedBody(uint pb, float3 pos, float4 rot, float3 v, float3 w)
{
    BodyState b = gBodies[pb];
    b.position = pos;
    b.orientation = rot;
    b.initialPos = pos;
    b.initialRot = rot;
    b.linearVel = v;
    b.angularVel = w;
    b.flags = BODY_DYNAMIC | BODY_AWAKE;
    b.sleepTimer = 0.0;
    b.generation = HOST_S(17);
    gBodies[pb] = b;
    // Adaptive warm start compares against the previous step's velocity.
    BODY_PREV(pb).linearVel = v;
}

void wakeBody(uint i)
{
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b)) return;
    if (bodyIsSleeping(b) || b.sleepTimer > 0.0)
    {
        gBodies[i].flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        gBodies[i].sleepTimer = 0.0;
    }
}

// ---------------------------------------------------------------------------
// vehicle_input: controls, functional state, severing, steering, wake
// ---------------------------------------------------------------------------
void detachWheel(uint v, uint vr, uint chassis, uint wi, bool spawnBody)
{
    uint w = VU(vr + V_CFG).y + wi;
    uint wr = wheelRec(w);
    uint4 dyn = VU(wr + W_C0);
    if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u) return;
    BodyState c = gBodies[chassis];
    float wheelMass = VD(wr + W_MISC).x;
    float3 hp = VD(wr + W_HP).xyz;
    float3 axl = axleLocal(vr);
    float3 iLocal = wheelInertiaLocal(vr, wr);
    chassisRemoveMass(chassis, vr, wheelMass, iLocal, hp + VD(wr + W_DIR).xyz * VD(wr + W_SUSP).y);
    if (spawnBody)
    {
        uint pb = VU(wr + W_PARK).x;
        float3 center = VD(wr + W_O_POS).xyz;
        float4 rot = VD(wr + W_O_ROT);
        if (dot(rot, rot) < 0.5)
        {
            center = transformPoint(c.position, c.orientation, hp + VD(wr + W_DIR).xyz * VD(wr + W_SUSP).y);
            rot = c.orientation;
        }
        float omega = VD(wr + W_W).x;
        float3 axleWorld = quatRotate(rot, axl);
        float3 vel = c.linearVel + cross(c.angularVel, center - c.position);
        float3 ang = c.angularVel + axleWorld * omega;
        activateParkedBody(pb, center, rot, vel, ang);
        setColliderFilter(VU(wr + W_PARK).y, VU(wr + W_PARK).z, chassis + 1u);
        float4 gv = VD(wr + W_GV);
        VSet(wr + W_GV, float4(gv.xyz, SEVER_GRACE));
    }
    dyn.z = (dyn.z & ~(WD_CONTACT | WD_STICK_L | WD_STICK_T | WD_SUSP | WD_BUMP | WD_SLIDE))
        | (spawnBody ? WD_DETACHED : WD_DISABLED);
    VSetU(wr + W_C0, dyn);
    gBodies[chassis].generation = HOST_S(17);
}

void detachPart(uint vr, uint chassis, uint p, bool spawnBody)
{
    uint pr = partRec(p);
    float4 com = VD(pr + P_COM);
    if (asuint(com.w) != PS_ATTACHED) return;
    uint4 cfg = VU(pr + P_CFG);
    float4 mass = VD(pr + P_MASS);
    BodyState c = gBodies[chassis];
    chassisRemoveMass(chassis, vr, mass.x, mass.yzw, com.xyz);
    uint target = VU(vr + V_POOL2).z;
    float3 offset = float3(0, 0, 0);
    if (spawnBody)
    {
        target = VU(pr + P_PARK).x;
        float3 pos = transformPoint(c.position, c.orientation, com.xyz);
        float3 vel = c.linearVel + cross(c.angularVel, pos - c.position);
        activateParkedBody(target, pos, c.orientation, vel, c.angularVel);
        offset = com.xyz;
    }
    for (uint k = 0u; k < cfg.y; k++)
    {
        uint ci = cfg.x + k;
        gColliders[ci].body = target;
        gColliders[ci].localPos = gColliders[ci].localPos - offset;
        gColliders[ci].generation = spawnBody ? chassis + 1u : 0u;
    }
    if (spawnBody)
    {
        float4 park = VD(pr + P_PARK);
        VSet(pr + P_PARK, float4(park.xy, SEVER_GRACE, park.w));
    }
    VSet(pr + P_COM, float4(com.xyz, asfloat(spawnBody ? PS_DETACHED : PS_DISABLED)));
    gBodies[chassis].generation = HOST_S(17);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_input(uint3 tid : SV_DispatchThreadID)
{
    uint v = dispatchIndex(tid);
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    uint4 cfg = VU(vr + V_CFG);
    if ((cfg.w & VF_ENABLED) == 0u) return;
    uint chassis = cfg.x;
    AvbdParams p = PARAMS;
    float dt = p.dt;

    // --- severing requests (applied once; masks are latched in V_D4) ---
    // Host requests, plus mounts vehicle_loads found overloaded last step.
    uint4 applied = VU(vr + V_D4);
    uint4 in1 = VU(vr + V_IN1);
    uint4 in3 = VU(vr + V_IN3);
    uint4 brk = VU(vr + V_BRK);
    bool changed = false;
    uint newDetach = (in1.z | brk.y) & ~applied.x;
    uint newDisable = in1.w & ~applied.y & ~(in1.z | brk.y);
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint bit = 1u << wi;
        if ((newDetach & bit) != 0u) { detachWheel(v, vr, chassis, wi, true); changed = true; }
        else if ((newDisable & bit) != 0u) { detachWheel(v, vr, chassis, wi, false); changed = true; }
    }
    applied.x |= newDetach;
    applied.y |= newDisable;
    uint4 pool2 = VU(vr + V_POOL2);
    uint newPartDetach = (in3.x | brk.x) & ~applied.z;
    uint newPartDisable = in3.y & ~applied.w & ~(in3.x | brk.x);
    for (uint pi = 0u; pi < pool2.y && pi < 32u; pi++)
    {
        uint bit = 1u << pi;
        if ((newPartDetach & bit) != 0u) { detachPart(vr, chassis, pool2.x + pi, true); changed = true; }
        else if ((newPartDisable & bit) != 0u) { detachPart(vr, chassis, pool2.x + pi, false); changed = true; }
    }
    applied.z |= newPartDetach;
    applied.w |= newPartDisable;
    VSetU(vr + V_D4, applied);
    if ((brk.x | brk.y) != 0u)
    {
        VSetU(vr + V_BRK, uint4(0u, 0u, brk.z | brk.x, brk.w | brk.y));
        changed = true;
    }

    // --- wheels attached at runtime by the host (spare slots) ---
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & WD_PENDING_ADD) == 0u) continue;
        float3 hp = VD(wr + W_HP).xyz + VD(wr + W_DIR).xyz * VD(wr + W_SUSP).y;
        chassisAddMass(chassis, vr, VD(wr + W_MISC).x, wheelInertiaLocal(vr, wr), hp);
        dyn.z &= ~(WD_PENDING_ADD | WD_DISABLED | WD_DETACHED);
        VSetU(wr + W_C0, dyn);
        changed = true;
    }

    // --- functional state: derived from part ownership, then host overrides ---
    uint wheelMask = 0u, driveMask = 0u, steerMask = 0u, brakeMask = 0u;
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint wf = VU(wr + W_CFG).y;
        uint wd = VU(wr + W_C0).z;
        if ((wf & WF_ENABLED) == 0u || (wd & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
        uint bit = 1u << wi;
        wheelMask |= bit;
        brakeMask |= bit;
        if ((wf & WF_DRIVEN) != 0u) driveMask |= bit;
        if ((wf & WF_STEER) != 0u) steerMask |= bit;
    }
    bool engineConnected = (in3.z & 1u) != 0u;
    uint partsLost = 0u;
    for (uint pi = 0u; pi < pool2.y && pi < 32u; pi++)
    {
        uint pr = partRec(pool2.x + pi);
        uint4 pc = VU(pr + P_CFG);
        float4 com = VD(pr + P_COM);
        uint state = asuint(com.w);
        if (state == PS_ATTACHED && pc.y > 0u && gColliders[pc.x].body != chassis)
        {
            // Someone else (fracture, gameplay) moved the part off the chassis.
            float4 m = VD(pr + P_MASS);
            chassisRemoveMass(chassis, vr, m.x, m.yzw, com.xyz);
            state = PS_LOST;
            VSet(pr + P_COM, float4(com.xyz, asfloat(state)));
            changed = true;
        }
        if (state == PS_ATTACHED) continue;
        partsLost |= 1u << pi;
        if (pc.z == PR_ENGINE) engineConnected = false;
        else if (pc.z == PR_DRIVESHAFT) driveMask &= ~pc.w;
        else if (pc.z == PR_STEERING) steerMask &= ~pc.w;
        else if (pc.z == PR_BRAKE) brakeMask &= ~pc.w;
    }
    uint4 over = VU(vr + V_IN2);
    wheelMask &= over.x;
    driveMask &= over.y & wheelMask;
    steerMask &= over.z & wheelMask;
    brakeMask &= over.w & wheelMask;

    uint4 prevMasks = VU(vr + V_O2);
    if (prevMasks.x != wheelMask || prevMasks.y != driveMask || prevMasks.z != steerMask) changed = true;
    uint4 o3 = VU(vr + V_O3);
    if ((o3.x != 0u) != engineConnected) changed = true;
    VSetU(vr + V_O2, uint4(wheelMask, driveMask, steerMask, prevMasks.w + (changed ? 1u : 0u)));
    VSetU(vr + V_O3, uint4(engineConnected ? 1u : 0u, partsLost, o3.z, o3.w));

    // --- input smoothing ---
    float4 in0 = VD(vr + V_IN0);
    float4 d1 = VD(vr + V_D1);
    float4 d2 = VD(vr + V_D2);
    float4 resp = VD(vr + V_RESP);
    BodyState c = gBodies[chassis];
    float3 fwdW = quatRotate(c.orientation, vehicleFwdLocal(vr));
    float speed = dot(c.linearVel, fwdW);
    float forwardIn = clamp(in0.x, -1.0, 1.0);
    float brakeIn = clamp(in0.z, 0.0, 1.0);
    // Throttle against the motion brakes first. Not for a steering tank: a pivot
    // swings the hull's forward speed negative and would brake the turn away.
    bool steeringTank = (cfg.w & VF_TRACKED) != 0u && abs(in0.y) > 0.1;
    if ((cfg.w & VF_REVERSE_BRAKE) != 0u && !steeringTank && forwardIn * speed < 0.0 && abs(speed) > 1.0)
    {
        brakeIn = max(brakeIn, abs(forwardIn));
        forwardIn = 0.0;
    }
    float steerTarget = clamp(in0.y, -1.0, 1.0);
    if ((cfg.w & VF_TRACKED) != 0u)
    {
        // Arcade skid steer: each track gets throttle ± steer (left = f + s, right =
        // f − s), so steer always yaws the hull the same way: a pivot from rest, a
        // track-ratio turn while driving, the same rotation sense in reverse. The
        // faster track sets the throttle and drive direction; the ratios are each
        // track's share of it (negative = counter-rotating), feeding Jolt's
        // tracked drivetrain unchanged.
        float4 tr = VD(vr + V_TRACK);
        float turnAW = VD(vr + V_TRACK_D).w;
        float s = steerTarget * tr.z;
        float trackL = forwardIn + s, trackR = forwardIn - s;
        float mag = max(abs(trackL), abs(trackR));
        float ratioL = 1.0, ratioR = 1.0;
        if (mag > 0.02)
        {
            float dir = trackL + trackR < -1e-4 ? -1.0 : 1.0;
            ratioL = trackL / (dir * mag);
            ratioR = trackR / (dir * mag);
            forwardIn = dir * min(mag, 1.0);
            // Both tracks the same way at speed: cap the turn's lateral acceleration
            // (yaw ≈ (1 − ratio)·v / W, lateral accel ≈ (1 − ratio)·v² / W).
            if (ratioL * ratioR >= 0.0 && turnAW > 0.0 && speed * speed > 1e-3)
            {
                float floorRatio = 1.0 - turnAW / (speed * speed);
                ratioL = max(ratioL, floorRatio);
                ratioR = max(ratioR, floorRatio);
            }
        }
        VSet(vr + V_TRACK_D, float4(ratioL, ratioR, forwardIn, turnAW));
        steerTarget = 0.0;
    }
    float forward = moveToward(d1.z, forwardIn, resp.x * dt);
    float brake = moveToward(d1.w, brakeIn, resp.y * dt);
    float handbrake = moveToward(d2.x, clamp(in0.w, 0.0, 1.0), resp.z * dt);
    float steerRate = VD(vr + V_STEER).w;
    uint steerCurve = asuint(VD(vr + V_AERO0).w);
    steerTarget *= curveEval(steerCurve, abs(speed));
    float steer = moveToward(d2.y, steerTarget, steerRate * dt);
    VSet(vr + V_D1, float4(d1.x, d1.y, forward, brake));
    VSet(vr + V_D2, float4(handbrake, steer, asfloat(wheelMask), asfloat(driveMask)));
    uint4 d3 = VU(vr + V_D3);
    d3.x = steerMask;
    d3.z = brakeMask;

    // --- wake on intent (inputs never wake a parked car by themselves; intent does) ---
    bool intent = abs(forwardIn) > 1e-4 || abs(steerTarget - steer) > 1e-4 || abs(steer - d2.y) > 1e-5
        || changed || in3.w != d3.w || ((cfg.w & VF_TRACKED) != 0u && abs(in0.y) > 0.02);
    d3.w = in3.w;
    VSetU(vr + V_D3, d3);
    if (intent) wakeBody(chassis);

    // --- per-wheel steer angle (Ackermann; motorcycles cap it by lean) ---
    float4 st = VD(vr + V_STEER);
    float4 bike0 = VD(vr + V_BIKE0);
    bool leanLimit = bike0.x > 0.0 && VD(vr + V_BIKE1).z > 0.5;
    float gScale = VD(vr + V_AERO0).z;
    float g = abs(PARAMS.gravityY) * (gScale > 0.0 ? gScale : 1.0);
    for (uint wi = 0u; wi < cfg.z; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        float maxSteer = VD(wr + W_STEERAX).w;
        float side = VD(wr + W_FWD).w;
        float angle = 0.0;
        if ((steerMask & (1u << wi)) != 0u)
        {
            angle = -steer * maxSteer; // + = left (Jolt convention); + input = right
            float v2 = speed * speed;
            float cosCaster = dot(normalize(VD(wr + W_STEERAX).xyz), vehicleUpLocal(vr));
            if (leanLimit && v2 > 1e-6 && cosCaster > 1e-6 && st.y > 0.0)
            {
                // SteerAngle = asin(wheelbase·tan(maxLean)·g / (v²·cos caster)) (Jolt).
                float lim = asin(clamp(st.y * tan(bike0.x) * g / (v2 * cosCaster), 0.0, 1.0));
                angle = clamp(angle, -lim, lim);
            }
            if (st.x > 0.0 && side != 0.0 && abs(angle) > 1e-4 && st.y > 0.0)
            {
                float turnR = st.y / tan(abs(angle));
                bool inner = (angle > 0.0) == (side > 0.0);
                float ack = atan(st.y / max(turnR + (inner ? -0.5 : 0.5) * st.z, 1e-3));
                angle = sign(angle) * lerp(abs(angle), ack, st.x);
            }
        }
        float4 c5 = VD(wr + W_C5);
        VSet(wr + W_C5, float4(c5.xyz, angle));
    }
}

// ---------------------------------------------------------------------------
// wheel_query: one workgroup per wheel casts along the suspension at x⁻
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void wheel_query(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint w = gid.x + gid.y * 32768u;
    uint lane = lid.x;
    bool live = w < wheelCountTotal();
    uint wr = live ? wheelRec(w) : 0u;
    uint v = live ? VU(wr + W_CFG).x : 0u;
    uint vr = live ? vehRec(v) : 0u;
    uint chassis = live ? VU(vr + V_CFG).x : 0u;
    if (live)
    {
        uint wd = VU(wr + W_C0).z;
        uint wf = VU(wr + W_CFG).y;
        BodyState c = gBodies[chassis];
        live = (VU(vr + V_CFG).w & VF_ENABLED) != 0u && (wf & WF_ENABLED) != 0u
            && (wd & (WD_DETACHED | WD_DISABLED)) == 0u && bodyIsDynamic(c) && !bodyIsSleeping(c);
    }
    float3 o = float3(0, 0, 0);
    float3 d = float3(0, -1, 0);
    float maxT = 0.0;
    float sweep = 0.0; // sphere-cast radius (0 = ray)
    float coneCos = 1.0;
    float lmin = 0.0;
    // Query rate (Jolt mNumStepsBetweenCollisionTest): between full probes a wheel
    // that had a contact slides it along the last hit plane, which moves with its
    // ground body. Staggered so a vehicle's wheels do not probe on the same step.
    bool probe = live;
    if (live)
    {
        BodyState c = gBodies[chassis];
        o = transformPoint(c.position, c.orientation, VD(wr + W_HP).xyz);
        d = normalize(quatRotate(c.orientation, VD(wr + W_DIR).xyz));
        bool sphere = (VU(vr + V_CFG).w & VF_SPHERE_CAST) != 0u;
        sweep = sphere ? VD(wr + W_HP).w : 0.0;
        // The tyre rows own the bottom cone (≤ 35° off the suspension axis for a
        // sphere cast, which also finds things beside the tyre); the wheel shell
        // (spheres from 60°) owns the rest, so the suspension never pushes along
        // its axis against a side hit.
        coneCos = sphere ? max(VD(vr + V_UP).w, SWEEP_CONE_COS) : VD(vr + V_UP).w;
        lmin = VD(wr + W_SUSP).x;
        maxT = VD(wr + W_SUSP).y + (sphere ? 0.0 : VD(wr + W_HP).w);
        uint interval = VU(vr + V_QUERY).x;
        bool hadHit = (VU(wr + W_C0).z & WD_CONTACT) != 0u && VD(wr + W_LHIT).w > 0.5;
        if (interval > 1u && hadHit && ((HOST_S(17) + w) % interval) != 0u) probe = false;
    }
    gRayExcludeGroup = (live && HOST_S(30) != 0u) ? bodyNegativeGroup(chassis) : 0;
    // Query mode in the vehicle header: 0 = broadphase grid walk, 1 = full scan.
    if (VU(VH_MISC).w == 1u) rayLaneScan(lane, probe, o, d, maxT, chassis, sweep, coneCos, lmin);
    else rayLaneScanGrid(lane, probe, o, d, maxT, chassis, sweep, coneCos, lmin);
    rayReduce(lane);
    if (!live || lane != 0u) return;

    BodyState c = gBodies[chassis];
    float radius = VD(wr + W_HP).w;
    uint4 dyn = VU(wr + W_C0);
    bool contact = false;
    float3 n = float3(0, 1, 0);
    float tHit = 0.0;
    uint best = NO_INDEX;
    uint ground = NO_INDEX;
    if (probe)
    {
        best = gsRayC[0];
        tHit = gsRayT[0];
        if (best != NO_INDEX)
        {
            // The scan only keeps hits the tyre cone accepts (tyreAccepts).
            float t2;
            float3 n2;
            rayCollider(best, o, d, sweep, t2, n2);
            n = dot(n2, d) > 0.0 ? -n2 : n2;
            contact = true;
        }
        ground = contact ? gColliders[best].body : NO_INDEX;
    }
    else
    {
        // Extrapolate on the stored hit plane (ground-body space).
        ground = dyn.x;
        best = dyn.y;
        BodyState g = gBodies[ground];
        float3 pc = transformPoint(g.position, g.orientation, VD(wr + W_LHIT).xyz);
        n = normalize(quatRotate(g.orientation, VD(wr + W_LN).xyz));
        float dn = dot(n, d);
        if (dn < -0.1)
        {
            // Sphere: centre one radius above the plane; ray: on the plane.
            tHit = (sweep - dot(n, o - pc)) / dn;
            contact = tHit >= 0.0 && tHit <= maxT && dot(n, -d) >= VD(vr + V_UP).w;
        }
        if (!contact) { ground = NO_INDEX; best = NO_INDEX; }
    }
    if (!contact || ground != dyn.x)
    {
        // New ground (or none): forget friction anchors and multipliers.
        float4 lam = VD(wr + W_LAM);
        VSet(wr + W_LAM, float4(0.0, 0.0, lam.z, lam.w));
        float4 k2 = VD(wr + W_K2);
        VSet(wr + W_K2, float4(k2.x, k2.y, 0.0, 0.0));
        dyn.z &= ~(WD_STICK_L | WD_STICK_T);
    }
    dyn.x = ground;
    dyn.y = contact ? best : NO_INDEX;
    dyn.z = contact ? (dyn.z | WD_CONTACT) : (dyn.z & ~(WD_CONTACT | WD_SUSP | WD_BUMP | WD_SLIDE));
    VSetU(wr + W_C0, dyn);
    if (!contact)
    {
        VSet(wr + W_C1, float4(0, 0, 0, VD(wr + W_SUSP).y));
        VSet(wr + W_HIT, float4(o + d * maxT, maxT));
        VSet(wr + W_LHIT, float4(0, 0, 0, 0));
        return;
    }
    // Ray: the hit is the contact. Sphere: the hit is the wheel centre; the
    // contact lies one radius along -n (edges give edge normals).
    float3 hit = sweep > 0.0 ? o + d * tHit - n * radius : o + d * tHit;
    float invCos = 1.0 / max(0.3, -dot(d, n));
    VSet(wr + W_C1, float4(hit - c.position, sweep > 0.0 ? tHit : tHit - radius));
    VSet(wr + W_C2, float4(n, invCos));
    VSet(wr + W_HIT, float4(hit, tHit));
    BodyState g = gBodies[ground];
    if (probe)
    {
        float4 gi = quatConjugate(g.orientation);
        VSet(wr + W_LHIT, float4(quatRotate(gi, hit - g.position), 1.0));
        VSet(wr + W_LN, float4(quatRotate(gi, n), 0.0));
    }
    float3 gv = float3(0, 0, 0);
    if (bodyIsDynamic(g) && !bodyIsSleeping(g))
        gv = g.linearVel + cross(g.angularVel, hit - g.position);
    VSet(wr + W_GV, float4(gv, VD(wr + W_GV).w));
}

// ---------------------------------------------------------------------------
// Drivetrain (Jolt WheeledVehicleController::PostCollide port) → per-wheel τ_drive
// ---------------------------------------------------------------------------
bool gaussSolve(inout float a[81], inout float b[9], uint n)
{
    for (uint col = 0u; col < n; col++)
    {
        uint piv = col;
        float best = abs(a[col * 9u + col]);
        for (uint r = col + 1u; r < n; r++)
        {
            float vv = abs(a[r * 9u + col]);
            if (vv > best) { best = vv; piv = r; }
        }
        if (best < 1e-12) return false;
        if (piv != col)
        {
            for (uint k = 0u; k < n; k++)
            {
                float tmp = a[col * 9u + k];
                a[col * 9u + k] = a[piv * 9u + k];
                a[piv * 9u + k] = tmp;
            }
            float tb = b[col]; b[col] = b[piv]; b[piv] = tb;
        }
        float inv = 1.0 / a[col * 9u + col];
        for (uint r = 0u; r < n; r++)
        {
            if (r == col) continue;
            float f = a[r * 9u + col] * inv;
            if (f == 0.0) continue;
            for (uint k = col; k < n; k++) a[r * 9u + k] -= f * a[col * 9u + k];
            b[r] -= f * b[col];
        }
    }
    for (uint r = 0u; r < n; r++) b[r] /= a[r * 9u + r];
    return true;
}

float gearRatio(uint vr, int gear)
{
    uint nF = asuint(VD(vr + V_TR1).y);
    uint nR = asuint(VD(vr + V_TR1).z);
    int idx = -1;
    if (gear > 0 && uint(gear) <= nF) idx = gear - 1;
    else if (gear < 0 && uint(-gear) <= nR) idx = int(nF) + (-gear) - 1;
    if (idx < 0) return 0.0;
    float4 g = VD(vr + (idx < 4 ? V_GEARS0 : V_GEARS1));
    uint k = uint(idx) & 3u;
    return k == 0u ? g.x : (k == 1u ? g.y : (k == 2u ? g.z : g.w));
}

void computeDrivetrain(uint vr, uint4 cfg, float dt, float speedFwd, inout float tauDrive[MAX_VEHICLE_WHEELS])
{
    for (uint wi = 0u; wi < MAX_VEHICLE_WHEELS; wi++) tauDrive[wi] = 0.0;
    float4 d1 = VD(vr + V_D1);
    float forward = d1.z;
    uint driveMask = asuint(VD(vr + V_D2).w);
    bool engineConnected = VU(vr + V_O3).x != 0u;

    if ((cfg.w & VF_DIRECT_DRIVE) != 0u)
    {
        float maxT = VD(vr + V_AERO1).w;
        float top = VD(vr + V_RESP).w;
        float s = abs(speedFwd);
        float scale = 1.0;
        if (top > 0.0 && forward * speedFwd > 0.0)
            scale = s < top / 3.0 ? 1.0 : clamp((top - s) / (top * 2.0 / 3.0), 0.0, 1.0);
        if (!engineConnected) scale = 0.0;
        for (uint wi = 0u; wi < cfg.z && wi < MAX_VEHICLE_WHEELS; wi++)
        {
            if ((driveMask & (1u << wi)) == 0u) continue;
            float mult = VD(wheelRec(cfg.y + wi) + W_UP).w;
            tauDrive[wi] = forward * maxT * scale * mult;
        }
        return;
    }

    float4 eng0 = VD(vr + V_ENG0);
    float4 eng1 = VD(vr + V_ENG1);
    float4 d0 = VD(vr + V_D0);
    int gear = int(round(d0.y));
    float ratio = gearRatio(vr, gear);
    float clutchFriction = d0.z;
    float omegaE = d0.x;
    float minW = eng0.y / RAD_PER_S_TO_RPM;
    float maxW = eng0.z / RAD_PER_S_TO_RPM;
    // Remember RPM before this step for the upshift gate.
    float4 refp = VD(vr + V_REFP);
    VSet(vr + V_REFP, float4(refp.xyz, omegaE * RAD_PER_S_TO_RPM));

    if ((cfg.w & VF_TRACKED) != 0u)
    {
        // Jolt TrackedVehicleController::PostCollide: the engine is rigidly tied
        // to the fastest track; each track gets diffRatio·ratio·transmission torque
        // up to the speed the engine allows, then the tracks are synced to the
        // left/right ratio conserving angular momentum.
        uint4 tl = VU(vr + V_TRACK);
        float4 td = VD(vr + V_TRACK_D);
        uint lead[2] = { tl.x, tl.y };
        float tr[2] = { td.x, td.y };
        bool live[2];
        for (uint i = 0u; i < 2u; i++)
            live[i] = lead[i] < cfg.z && (driveMask & (1u << lead[i])) != 0u;
        bool connected = gear != 0 && clutchFriction > 1e-3 && engineConnected && (live[0] || live[1]);
        if (connected)
        {
            bool fwdGear = ratio >= 0.0;
            float fastest = fwdGear ? -1e30 : 1e30;
            for (uint i = 0u; i < 2u; i++)
            {
                if (!live[i]) continue;
                uint wr = wheelRec(cfg.y + lead[i]);
                float s = VD(wr + W_W).x * VD(wr + W_UP).w;
                fastest = fwdGear ? max(fastest, s) : min(fastest, s);
            }
            omegaE = clamp(fastest * ratio, minW, maxW);
        }
        else
        {
            omegaE = clamp(omegaE * max(0.0, 1.0 - eng1.x * dt), minW, maxW);
            float freeThrottle = (cfg.w & VF_MANUAL_TRANS) != 0u && engineConnected ? abs(forward) : 0.0;
            float tq = freeThrottle * eng0.x * curveEval(asuint(eng1.y), omegaE * RAD_PER_S_TO_RPM / max(eng0.z, 1.0));
            omegaE = clamp(omegaE + tq * dt / max(eng0.w, 1e-6), minW, maxW);
        }
        float rpm = omegaE * RAD_PER_S_TO_RPM;
        float transmission = engineConnected
            ? clutchFriction * ratio * abs(forward) * eng0.x * curveEval(asuint(eng1.y), rpm / max(eng0.z, 1.0)) : 0.0;
        if (transmission != 0.0)
        {
            for (uint i = 0u; i < 2u; i++)
            {
                if (!live[i] || tr[i] == 0.0) continue;
                uint wr = wheelRec(cfg.y + lead[i]);
                float diff = VD(wr + W_UP).w;
                float trackMax = rpm / (ratio * diff * tr[i] * RAD_PER_S_TO_RPM) * 1.001;
                float w0 = VD(wr + W_W).x;
                if (w0 * trackMax < 0.0 || abs(w0) < abs(trackMax)) tauDrive[lead[i]] = diff * tr[i] * transmission;
            }
        }
        // Regenerative steering: the torque the inner track's steering brake took
        // last step drives the outer track (a geared steering differential), so
        // steering trades speed between the tracks instead of only braking.
        if (live[0] && live[1] && abs(tr[0]) != abs(tr[1]) && td.z != 0.0)
        {
            uint inner = abs(tr[0]) < abs(tr[1]) ? 0u : 1u;
            float lamB = VD(wheelRec(cfg.y + lead[inner]) + W_LAM).z;
            tauDrive[lead[1u - inner]] += lamB;
        }
        if (live[0] && live[1])
        {
            uint wl = wheelRec(cfg.y + lead[0]);
            uint wrr = wheelRec(cfg.y + lead[1]);
            float4 sl = VD(wl + W_W);
            float4 sr = VD(wrr + W_W);
            float il = max(VD(wl + W_INER).x, 1e-6);
            float ir = max(VD(wrr + W_INER).x, 1e-6);
            if (tr[0] * tr[1] > 0.0)
            {
                // Same direction: Jolt's momentum-conserving exchange.
                float denom = tr[0] * ir + tr[1] * il;
                if (abs(denom) > 1e-9)
                {
                    float impulse = (tr[0] * sr.x - tr[1] * sl.x) / denom;
                    sl.x += impulse * il;
                    sr.x -= impulse * ir;
                }
            }
            else
            {
                // Counter-rotating: Jolt's opposite-sign formula sends two equal track
                // speeds to 0/0 (any ratio "holds"), so a partial counter-rotation from
                // rolling never turns. Project onto right·ωl = left·ωr instead, with
                // the least inertia-weighted change.
                float g = tr[1] * tr[1] / il + tr[0] * tr[0] / ir;
                if (g > 1e-12)
                {
                    float lambda = (tr[0] * sr.x - tr[1] * sl.x) / g;
                    sl.x += lambda * tr[1] / il;
                    sr.x -= lambda * tr[0] / ir;
                }
            }
            VSet(wl + W_W, sl);
            VSet(wrr + W_W, sr);
        }
        VSet(vr + V_D0, float4(omegaE, d0.yzw));
        return;
    }

    float throttle = abs(forward);
    if ((cfg.w & VF_MANUAL_TRANS) == 0u) throttle *= clutchFriction;
    if (!engineConnected) throttle = 0.0;
    if ((cfg.w & VF_TRACTION_CONTROL) != 0u)
    {
        // Traction control on last step's slip, normalised by each tyre's curve peak:
        // full torque to 0.8× peak slip, down to a fifth by 1.6× (past the peak a
        // tyre trades its side grip for wheelspin). The slip ratio is regularised as
        // slip speed / max(|v|, 3 m/s): at a standstill κ = slip / |v| blows up and
        // would choke every launch.
        float kMax = 0.0;
        float vReg = abs(speedFwd) / max(abs(speedFwd), 3.0);
        for (uint wi = 0u; wi < cfg.z && wi < MAX_VEHICLE_WHEELS; wi++)
        {
            uint wr = wheelRec(cfg.y + wi);
            if ((driveMask & (1u << wi)) != 0u && (VU(wr + W_C0).z & WD_CONTACT) != 0u)
                kMax = max(kMax, VD(wr + W_C3).w * vReg / max(curvePeakX(asuint(VD(wr + W_TIRE).x)), 0.03));
        }
        // Low-pass the cut (τ 0.1 s): applied raw it chattered around the threshold
        // every step, which also amplified tiny backend (Metal vs WebGPU) differences.
        float4 q = VD(vr + V_QUERY);
        float cutTarget = 1.0 - clamp(1.0 - (kMax - 0.8) / 1.0, 0.2, 1.0);
        float cut = lerp(q.y, cutTarget, 1.0 - exp(-dt / 0.1));
        VSet(vr + V_QUERY, float4(q.x, cut, q.zw));
        throttle *= 1.0 - cut;
    }
    omegaE = clamp(omegaE * max(0.0, 1.0 - eng1.x * dt), minW, maxW);
    float rpm = omegaE * RAD_PER_S_TO_RPM;
    float engineTorque = throttle * eng0.x * curveEval(asuint(eng1.y), rpm / max(eng0.z, 1.0));

    // Driven differentials (missing / severed wheels drop out; ratios renormalise).
    uint4 pool = VU(vr + V_POOL);
    uint dIdx[MAX_DRIVEN_WHEELS];
    float dRatio[MAX_DRIVEN_WHEELS];
    float dTorque[MAX_DRIVEN_WHEELS];
    float dEst[MAX_DRIVEN_WHEELS];
    uint nd = 0u;
    float diffOmega[8];
    float diffRatio[8];
    uint diffL[8];
    uint diffR[8];
    float diffShare[8];
    uint nDiff = 0u;
    float omegaMin = 1e30, omegaMax = 0.0;
    float shareSum = 0.0;
    uint diffBase = VU(VH_POOLS).x;
    for (uint di = 0u; di < pool.y && di < 8u; di++)
    {
        uint dr = diffBase + (pool.x + di) * DIFF_STRIDE;
        uint4 du = VU(dr);
        float4 df = VD(dr);
        float4 df2 = VD(dr + 1u);
        uint l = du.x, r = du.y;
        if (l != NO_INDEX && (driveMask & (1u << l)) == 0u) l = NO_INDEX;
        if (r != NO_INDEX && (driveMask & (1u << r)) == 0u) r = NO_INDEX;
        if (l == NO_INDEX && r == NO_INDEX) continue;
        float avg = 0.0;
        float cnt = 0.0;
        if (l != NO_INDEX) { avg += VD(wheelRec(cfg.y + l) + W_W).x; cnt += 1.0; }
        if (r != NO_INDEX) { avg += VD(wheelRec(cfg.y + r) + W_W).x; cnt += 1.0; }
        avg = abs(avg * df.z / cnt);
        diffOmega[nDiff] = avg;
        diffRatio[nDiff] = df.z;
        diffL[nDiff] = l;
        diffR[nDiff] = r;
        diffShare[nDiff] = df2.y;
        shareSum += df2.y;
        omegaMin = min(omegaMin, avg);
        omegaMax = max(omegaMax, avg);
        nDiff++;
    }
    for (uint di = 0u; di < nDiff; di++) diffShare[di] = shareSum > 0.0 ? diffShare[di] / shareSum : 1.0 / float(nDiff);
    float interLs = eng1.w;
    if (nDiff > 1u && interLs < 1e29 && omegaMax > omegaMin)
    {
        float sumF = 0.0;
        float tmpF[8];
        for (uint di = 0u; di < nDiff; di++) { tmpF[di] = (omegaMax - diffOmega[di]) / (omegaMax - omegaMin); sumF += tmpF[di]; }
        float lo = max(1e-3, omegaMin), hi = max(1e-3, omegaMax);
        float a = min((hi / lo - 1.0) / (interLs - 1.0), 1.0);
        for (uint di = 0u; di < nDiff; di++) diffShare[di] = (1.0 - a) * diffShare[di] + a * tmpF[di] / sumF;
    }
    for (uint di = 0u; di < nDiff; di++)
    {
        uint dr = diffBase + (pool.x + di) * DIFF_STRIDE;
        float split = VD(dr).w;
        float ls = VD(dr + 1u).x;
        float c2w = ratio * diffRatio[di];
        uint l = diffL[di], r = diffR[di];
        if (l != NO_INDEX && r != NO_INDEX)
        {
            float wl = VD(wheelRec(cfg.y + l) + W_W).x;
            float wrr = VD(wheelRec(cfg.y + r) + W_W).x;
            float fl = 1.0 - split, fr = split;
            if (ls < 1e29)
            {
                float ol = max(1e-3, abs(wl)), orr = max(1e-3, abs(wrr));
                float a = min((max(ol, orr) / min(ol, orr) - 1.0) / (ls - 1.0), 1.0);
                if (ol < orr) { fl = fl * (1.0 - a) + a; fr = fr * (1.0 - a); }
                else { fl = fl * (1.0 - a); fr = fr * (1.0 - a) + a; }
            }
            if (nd < MAX_DRIVEN_WHEELS) { dIdx[nd] = l; dRatio[nd] = c2w; dTorque[nd] = diffShare[di] * fl; nd++; }
            if (nd < MAX_DRIVEN_WHEELS) { dIdx[nd] = r; dRatio[nd] = c2w; dTorque[nd] = diffShare[di] * fr; nd++; }
        }
        else if (nd < MAX_DRIVEN_WHEELS)
        {
            dIdx[nd] = l != NO_INDEX ? l : r; dRatio[nd] = c2w; dTorque[nd] = diffShare[di]; nd++;
        }
    }

    bool solved = false;
    if (nd > 0u)
    {
        uint n = nd + 1u;
        uint engine = nd;
        float a[81];
        float b[9];
        for (uint k = 0u; k < 81u; k++) a[k] = 0.0;
        for (uint k = 0u; k < 9u; k++) b[k] = 0.0;
        float S = ratio != 0.0 && engineConnected ? clutchFriction * eng1.z : 0.0;
        float dtDivIe = dt / max(eng0.w, 1e-6);
        float nf = float(nd);
        for (uint i = 0u; i < nd; i++)
        {
            uint wr = wheelRec(cfg.y + dIdx[i]);
            float inertia = max(VD(wr + W_INER).x, 1e-6);
            float sr = S * dRatio[i];
            float k = dt * sr * dTorque[i] / inertia;
            for (uint j = 0u; j < nd; j++) a[i * 9u + j] = k * dRatio[j] / nf;
            a[i * 9u + i] += 1.0;
            a[i * 9u + engine] = -k;
            // External wheel torque estimate: last step's tire force and brake.
            float R = VD(wr + W_HP).w;
            float4 lam = VD(wr + W_LAM);
            bool contactW = (VU(wr + W_C0).z & WD_CONTACT) != 0u;
            float dtTw = dt * ((contactW ? R * lam.x : 0.0) - lam.z);
            dEst[i] = dtTw;
            b[i] = VD(wr + W_W).x + dtTw / inertia;
            a[engine * 9u + i] = -dtDivIe * sr / nf;
        }
        a[engine * 9u + engine] = 1.0 + dtDivIe * S;
        b[engine] = omegaE + dtDivIe * engineTorque;
        if (gaussSolve(a, b, n))
        {
            for (uint i = 0u; i < nd; i++)
            {
                uint wr = wheelRec(cfg.y + dIdx[i]);
                float inertia = max(VD(wr + W_INER).x, 1e-6);
                float omegaJ = b[i] - dEst[i] / inertia;
                tauDrive[dIdx[i]] = inertia * (omegaJ - VD(wr + W_W).x) / dt;
            }
            omegaE = clamp(b[engine], minW, maxW);
            solved = true;
        }
    }
    if (!solved) omegaE = clamp(omegaE + engineTorque * dt / max(eng0.w, 1e-6), minW, maxW);
    VSet(vr + V_D0, float4(omegaE, d0.yzw));
}

// ---------------------------------------------------------------------------
// Row assembly (shared by prepare and iterate)
// ---------------------------------------------------------------------------
Vec6 jac(float3 e, float3 r)
{
    Vec6 j;
    j.v[0] = e.x; j.v[1] = e.y; j.v[2] = e.z;
    float3 a = cross(r, e);
    j.v[3] = a.x; j.v[4] = a.y; j.v[5] = a.z;
    return j;
}

float jdot(Vec6 j, float3 dl, float3 da)
{
    return j.v[0] * dl.x + j.v[1] * dl.y + j.v[2] * dl.z + j.v[3] * da.x + j.v[4] * da.y + j.v[5] * da.z;
}

// Suspension rows act at the contact, or at a fixed chassis point (Jolt
// mSuspensionForcePoint) when the wheel has one.
float3 suspensionLever(uint wr, float3 r, float4 q0)
{
    float4 fp = VD(wr + W_FP);
    return fp.w > 0.5 ? quatRotate(q0, fp.xyz) : r;
}

float3 lateralLever(uint wr, float3 r, float3 upW)
{
    float rollInf = VD(wr + W_MISC).w;
    return r - upW * (dot(r, upW) * (1.0 - rollInf));
}

// ---------------------------------------------------------------------------
// Two-body tyre rows. A wheel on a dynamic body (vehicle_link sets W_TJ.z) sees
// that body's current solver iterate at the contact instead of a given ground
// motion, and writes the body's share of every row (J_g = −[e; r_g × e], same
// stiffness and force as the chassis side) into its ground block, which the
// body's primal adds through the wheel's tyre joint (JOINT_FLAG_TYRE).
// ---------------------------------------------------------------------------
uint wheelGround(uint wr) { return VU(wr + W_TJ).z; }

uint groundBlockIndex(uint wheelSlot)
{
    return VU(VH_BASES).w + 4u * VU(VH_COUNTS).x + 4u * wheelSlot;
}

// Ground displacement at the contact since x⁻: the body's current iterate for a
// two-body wheel, else the given motion v_g·dt (static / kinematic ground).
float3 wheelGroundDisp(uint wr)
{
    uint gi = wheelGround(wr);
    if (gi == NO_INDEX) return VD(wr + W_C5).xyz;
    BodyState gb = gBodies[gi];
    if (bodyIsSleeping(gb)) return float3(0, 0, 0);
    float4 q = gb.orientation;
    if (dot(q, gb.initialRot) < 0.0) q = -q;
    return (gb.position - gb.initialPos) + cross(quatSubVec(q, gb.initialRot), VD(wr + W_HIT).xyz - gb.initialPos);
}

Vec6 groundJac(uint wr, float3 e)
{
    BodyState gb = gBodies[wheelGround(wr)];
    Vec6 j = jac(e, VD(wr + W_HIT).xyz - gb.initialPos);
    for (int a = 0; a < 6; a++) j.v[a] = -j.v[a];
    return j;
}

// Add k·JJᵀ and f·J to a ground block (lower triangle + rhs, writeVehicleBlock layout).
void groundAdd(uint wheelSlot, Vec6 j, float k, float f)
{
    uint at = groundBlockIndex(wheelSlot);
    float b[32];
    for (uint e = 0u; e < 4u; e++)
    {
        Aabb a = gAabbs[at + e];
        b[e * 8u] = a.mn.x; b[e * 8u + 1u] = a.mn.y; b[e * 8u + 2u] = a.mn.z; b[e * 8u + 3u] = a._pad0;
        b[e * 8u + 4u] = a.mx.x; b[e * 8u + 5u] = a.mx.y; b[e * 8u + 6u] = a.mx.z; b[e * 8u + 7u] = a._pad1;
    }
    uint idx = 0u;
    for (uint r = 0u; r < 6u; r++)
        for (uint c2 = 0u; c2 <= r; c2++) { b[idx] += k * j.v[r] * j.v[c2]; idx++; }
    for (uint r = 0u; r < 6u; r++) b[21u + r] += f * j.v[r];
    b[27] = 1.0;
    for (uint e = 0u; e < 4u; e++)
    {
        Aabb a;
        a.mn = float3(b[e * 8u], b[e * 8u + 1u], b[e * 8u + 2u]);
        a._pad0 = b[e * 8u + 3u];
        a.mx = float3(b[e * 8u + 4u], b[e * 8u + 5u], b[e * 8u + 6u]);
        a._pad1 = b[e * 8u + 7u];
        gAabbs[at + e] = a;
    }
}

void groundClear(uint wheelSlot)
{
    uint at = groundBlockIndex(wheelSlot);
    Aabb z;
    z.mn = float3(0, 0, 0); z._pad0 = 0.0; z.mx = float3(0, 0, 0); z._pad1 = 0.0;
    for (uint e = 0u; e < 3u; e++) gAabbs[at + e] = z;
    z._pad0 = 1.0; // active (empty until rows add to it)
    gAabbs[at + 3u] = z;
}

// Assemble every wheel row at chassis displacement (dl, da) from x⁻.
// When dual is set, first update multipliers/penalties at this iterate.
// N: per-wheel scratch size (≥ the vehicle's wheels; see assembleVehicle).
// assembleVehicleN's passes, one function each (the serial path calls them in
// order; soft_primal_bodies' vehicle lanes split the per-wheel ones).
// Pass 1: suspension / bump stop of wheel wi.
void vehSuspension(uint wi, uint vr, uint4 cfg, BodyState c, float3 dl, float3 da, inout Mat6 H, inout Vec6 g,
    out float outL, out float outFn, out bool contact)
{
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));

        outL = 0.0; outFn = 0.0; contact = false;
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & WD_CONTACT) == 0u) return;
        contact = true;
        float4 c1 = VD(wr + W_C1);
        float4 c2 = VD(wr + W_C2);
        float3 eg = wheelGroundDisp(wr);
        float4 k = VD(wr + W_K);
        float4 susp = VD(wr + W_SUSP);
        Vec6 jn = jac(c2.xyz, suspensionLever(wr, c1.xyz, c.initialRot));
        float un = jdot(jn, dl, da) - dot(c2.xyz, eg);
        float L = c1.w + c2.w * un;
        outL = L;
        float Lnat = susp.y + susp.z;
        float fS = k.x * (L - Lnat) + k.y * (L - c1.w);
        uint flags = dyn.z & ~(WD_SUSP | WD_BUMP);
        float fsTotal = 0.0;
        float kTotal = 0.0;
        if (fS < 0.0) { fsTotal += fS; kTotal += k.x + k.y; flags |= WD_SUSP; }
        if (L < susp.x)
        {
            float fB = k.z * (L - susp.x) + k.w * (L - c1.w);
            if (fB < 0.0) { fsTotal += fB; kTotal += k.z + k.w; flags |= WD_BUMP; }
        }
        // Optional physical limit (bump_max_force): past it the row is a constant force.
        float bumpMax = VD(wr + W_MISC).z;
        if (bumpMax > 0.0 && -fsTotal * c2.w > bumpMax)
        {
            fsTotal = -bumpMax / max(c2.w, 1e-6);
            kTotal = 0.0;
        }
        if (fsTotal != 0.0)
        {
            Vec6 js = jn;
            for (int a = 0; a < 6; a++) js.v[a] *= c2.w;
            mat6AddOuter(H, js, kTotal);
            vec6AddScaled(g, js, fsTotal);
            if (wheelGround(wr) != NO_INDEX)
            {
                Vec6 jg = groundJac(wr, c2.xyz);
                for (int a = 0; a < 6; a++) jg.v[a] *= c2.w;
                groundAdd(cfg.y + wi, jg, kTotal, fsTotal);
            }
        }
        outFn = max(0.0, -fsTotal * c2.w);
        dyn.z = flags;
        VSetU(wr + W_C0, dyn);
    }

void vehAntiRoll<let N : uint>(uint vr, uint4 cfg, BodyState c, uint nW, float Lw[N], inout float FnW[N],
    inout Mat6 H, inout Vec6 g)
{
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));
    // Pass 2: anti-roll bars (only when both wheels touch the ground, as Jolt).
    uint4 pool = VU(vr + V_POOL);
    uint arBase = VU(VH_POOLS).y;
    for (uint ai = 0u; ai < pool.w; ai++)
    {
        float4 ar = VD(arBase + (pool.z + ai) * ANTIROLL_STRIDE);
        uint l = asuint(ar.x), r = asuint(ar.y);
        if (l >= nW || r >= nW) continue;
        uint wl = wheelRec(cfg.y + l), wrr = wheelRec(cfg.y + r);
        if ((VU(wl + W_C0).z & WD_CONTACT) == 0u || (VU(wrr + W_C0).z & WD_CONTACT) == 0u) continue;
        float4 cl = VD(wl + W_C1), cr = VD(wrr + W_C1);
        float4 nl = VD(wl + W_C2), nr = VD(wrr + W_C2);
        Vec6 jl = jac(nl.xyz, suspensionLever(wl, cl.xyz, c.initialRot));
        Vec6 jr = jac(nr.xyz, suspensionLever(wrr, cr.xyz, c.initialRot));
        Vec6 gj;
        for (int a = 0; a < 6; a++) gj.v[a] = nl.w * jl.v[a] - nr.w * jr.v[a];
        float diff = Lw[l] - Lw[r];
        mat6AddOuter(H, gj, ar.z);
        vec6AddScaled(g, gj, ar.z * diff);
        // Each wheel's ground carries its end of the bar (same row, its own Jacobian).
        if (wheelGround(wl) != NO_INDEX)
        {
            Vec6 jg = groundJac(wl, nl.xyz);
            for (int a = 0; a < 6; a++) jg.v[a] *= nl.w;
            groundAdd(cfg.y + l, jg, ar.z, ar.z * diff);
        }
        if (wheelGround(wrr) != NO_INDEX)
        {
            Vec6 jg = groundJac(wrr, nr.xyz);
            for (int a = 0; a < 6; a++) jg.v[a] *= -nr.w;
            groundAdd(cfg.y + r, jg, ar.z, ar.z * diff);
        }
        FnW[l] = max(0.0, FnW[l] - ar.z * diff * nl.w);
        FnW[r] = max(0.0, FnW[r] + ar.z * diff * nr.w);
    }

}

    // Pass 3: friction rows + wheel spin. Each spin group (one wheel, or every
    // wheel of a track) shares one rotation y, solved exactly for the current
    // chassis iterate (1-D, monotone, piecewise linear in the tyre and brake
    // limits), then folded in with the Schur complement of the regime it lands
    // in: rolling (coupled) or saturated (decoupled). The group's inertia, brake
    // and drive live on its leader (the driven sprocket for a track).
void vehSpinGroup<let N : uint>(uint li, uint vr, uint4 cfg, BodyState c, float3 dl, float3 da, bool dual, uint nW,
    float FnW[N], inout Mat6 H, inout Vec6 g)
{
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));

        uint lr = wheelRec(cfg.y + li);
        uint leadOf = VU(lr + W_CFG).w;
        if (leadOf != NO_INDEX && leadOf != li) return; // a member: its leader handles it
        if ((VU(lr + W_C0).z & (WD_DETACHED | WD_DISABLED)) != 0u) return;
        float4 yv = VD(lr + W_Y);     // y, yTilde, mY, tauB (group)
        float4 lamLead = VD(lr + W_LAM);
        float4 k2Lead = VD(lr + W_K2);
        bool servo = yv.w > 0.0;

        uint mWheel[N];
        float mR[N];
        float mCL0[N];
        float mCT[N];
        float mKL[N];
        float mLamL[N];
        float mAL[N];
        float mAT[N];
        float mFT[N];
        bool mClampT[N];
        bool mContact[N];
        bool mClampL[N];
        float mFL[N];
        uint nm = 0u;
        float rAsum = 0.0;
        for (uint wi = 0u; wi < nW; wi++)
        {
            uint wr = wheelRec(cfg.y + wi);
            uint lead = VU(wr + W_CFG).w;
            if ((lead == NO_INDEX ? wi : lead) != li) continue;
            uint4 dyn = VU(wr + W_C0);
            if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
            float R = VD(wr + W_HP).w;
            float4 lam = VD(wr + W_LAM);
            float4 k2 = VD(wr + W_K2);
            float4 mu = VD(wr + W_MU);
            bool contact = (dyn.z & WD_CONTACT) != 0u;
            float Fn = FnW[wi];
            float CL0 = 0.0, CT = 0.0, aL = 0.0, aT = 0.0;
            if (contact)
            {
                float4 c1 = VD(wr + W_C1);
                float3 eg = wheelGroundDisp(wr);
                float3 t = VD(wr + W_C3).xyz;
                float3 s = VD(wr + W_C4).xyz;
                CL0 = alphaStab * k2.z + jdot(jac(t, c1.xyz), dl, da) - dot(t, eg);
                CT = alphaStab * k2.w + jdot(jac(s, lateralLever(wr, c1.xyz, upW)), dl, da) - dot(s, eg);
                aL = ((dyn.z & WD_STICK_L) != 0u ? mu.z : mu.x) * Fn;
                aT = ((dyn.z & WD_STICK_T) != 0u ? mu.w : mu.y) * Fn;
            }
            // Friction ellipse: split the budget by the raw forces at the previous y.
            float fTraw = k2.x * CT + lam.y;
            float fLprev = lam.w * (CL0 - R * yv.x) + lam.x;
            float fT = fTraw;
            float aLeff = aL;
            float tLim = aT; // lateral bound the dual update re-clamps to
            bool clampT = false;
            if (contact)
            {
                float e = (aL > 1e-9 && aT > 1e-9)
                    ? (fLprev / aL) * (fLprev / aL) + (fTraw / aT) * (fTraw / aT) : 2.0;
                if (aT <= 1e-9) { fT = 0.0; clampT = true; }
                else if ((VU(wr + W_CFG).y & WF_BOX_FRICTION) != 0u)
                {
                    fT = clamp(fTraw, -aT, aT);
                    clampT = abs(fTraw) > aT;
                }
                else if (e > 1.0)
                {
                    // Saturated. Near the grip peak split the ellipse by the raw row
                    // forces; well past it (a locked or spinning wheel) a sliding
                    // tyre's force opposes its slip velocity, so share it along the
                    // peak-normalised slip (κ/κ̂, tan α/tan α̂), Pacejka's similarity
                    // method. Raw forces alone let the stiff lateral row keep its grip
                    // on a locked wheel (no handbrake turns); the slip direction alone
                    // stripped a driven tyre's side grip at κ ≈ 0.05 on a straight
                    // (bikes swapped ends under throttle).
                    float sc = rsqrt(max(e, 1e-12));
                    float tProp = abs(fTraw) * sc;
                    float lProp = abs(fLprev) * sc;
                    float4 tire = VD(wr + W_TIRE);
                    float kPk = max(curvePeakX(asuint(tire.x)), 0.03);
                    float aPk = tan(radians(max(curvePeakX(asuint(tire.y)), 1.5)));
                    float nx = VD(wr + W_C3).w / kPk;
                    float ny = tan(min(VD(wr + W_C4).w, 1.45)) / aPk;
                    float nn = sqrt(nx * nx + ny * ny);
                    float wSlip = nn > 1e-6 ? clamp(max(nx, ny) - 1.0, 0.0, 1.0) : 0.0;
                    float tCap = lerp(tProp, nn > 1e-6 ? aT * ny / nn : tProp, wSlip);
                    aLeff = lerp(lProp, nn > 1e-6 ? aL * nx / nn : lProp, wSlip);
                    clampT = abs(fTraw) > tCap;
                    fT = clampT ? sign(fTraw) * tCap : fTraw;
                    tLim = max(tCap, abs(fT));
                }
                else aLeff = aL * sqrt(max(0.0, 1.0 - (fTraw / aT) * (fTraw / aT)));
            }
            mWheel[nm] = wi; mR[nm] = R; mCL0[nm] = CL0; mCT[nm] = CT; mKL[nm] = lam.w;
            mLamL[nm] = lam.x; mAL[nm] = contact ? aLeff : 0.0; mAT[nm] = tLim; mFT[nm] = fT;
            mClampT[nm] = clampT; mContact[nm] = contact; mClampL[nm] = false; mFL[nm] = 0.0;
            rAsum += R * mAL[nm];
            nm++;
        }

        float kB = k2Lead.y;
        float lamB = lamLead.z;
        float yStar = VD(lr + W_LN).w; // servo target (0 = brake to a stop this step)
        float y = yv.x;
        float fB = 0.0;
        bool clampB = false;
        for (uint pass = 0u; pass < (dual ? 2u : 1u); pass++)
        {
            // Solve G(y) = mY(y − ỹ) − Σ R·fL(y) + fB(y) = 0 (monotone increasing).
            float spread = (rAsum + yv.w) / max(yv.z, 1e-9) + 1e-6;
            float lo = yv.y - spread, hi = yv.y + spread;
            y = clamp(y, lo, hi);
            for (uint it = 0u; it < 24u; it++)
            {
                float G = yv.z * (y - yv.y);
                float slope = yv.z;
                for (uint m = 0u; m < nm; m++)
                {
                    if (!mContact[m]) continue;
                    float fLr = mKL[m] * (mCL0[m] - mR[m] * y) + mLamL[m];
                    mFL[m] = clamp(fLr, -mAL[m], mAL[m]);
                    mClampL[m] = abs(fLr) > mAL[m];
                    G -= mR[m] * mFL[m];
                    if (!mClampL[m]) slope += mR[m] * mR[m] * mKL[m];
                }
                float fBr = kB * (y - yStar) + lamB;
                fB = servo ? clamp(fBr, -yv.w, yv.w) : 0.0;
                clampB = servo && abs(fBr) > yv.w;
                G += fB;
                if (!clampB && servo) slope += kB;
                if (abs(G) <= 1e-7 * (yv.z * abs(yv.y) + rAsum + yv.w + 1.0)) break;
                if (G > 0.0) hi = y; else lo = y;
                float yn = y - G / slope;
                if (yn <= lo || yn >= hi) yn = 0.5 * (lo + hi);
                y = yn;
            }
            if (!dual || pass == 1u) break;
            // AVBD dual update at this iterate, then re-solve with the new multipliers.
            AvbdParams pp = PARAMS;
            for (uint m = 0u; m < nm; m++)
            {
                uint wr = wheelRec(cfg.y + mWheel[m]);
                float4 lam = VD(wr + W_LAM);
                float4 k2 = VD(wr + W_K2);
                if (mContact[m])
                {
                    float kCap = 1e3 * max(VD(wr + W_W).w, 1.0) / (pp.dt * pp.dt);
                    float CL = mCL0[m] - mR[m] * y;
                    lam.x = mFL[m];
                    lam.y = mFT[m];
                    if (!mClampL[m] && !mClampT[m])
                    {
                        lam.w = min(lam.w + pp.betaLin * abs(CL), kCap);
                        k2.x = min(k2.x + pp.betaLin * abs(mCT[m]), kCap);
                    }
                    uint4 dyn = VU(wr + W_C0);
                    uint stick = (!mClampL[m] ? WD_STICK_L : 0u) | (!mClampT[m] ? WD_STICK_T : 0u);
                    dyn.z = (dyn.z & ~(WD_STICK_L | WD_STICK_T | WD_SLIDE)) | stick
                        | ((mClampL[m] || mClampT[m]) ? WD_SLIDE : 0u);
                    VSetU(wr + W_C0, dyn);
                    mKL[m] = lam.w;
                    mLamL[m] = lam.x;
                    mFT[m] = clamp(k2.x * mCT[m] + lam.y, -mAT[m], mAT[m]);
                }
                if (mWheel[m] == li)
                {
                    lam.z = servo ? fB : 0.0;
                    if (servo && !clampB) k2.y = min(k2.y + pp.betaAng * abs(y - yStar), 1e3 * k2.y);
                    kB = k2.y;
                    lamB = lam.z;
                }
                VSet(wr + W_LAM, lam);
                VSet(wr + W_K2, k2);
            }
        }

        // Schur complement of the shared spin: H += Σ k JᵀJ − u uᵀ / A, u = Σ R·k·Jt.
        float kBa = (servo && !clampB) ? kB : 0.0;
        float A = yv.z + kBa;
        for (uint m = 0u; m < nm; m++)
            if (mContact[m] && !mClampL[m]) A += mR[m] * mR[m] * mKL[m];
        Vec6 u = vec6Zero();
        bool coupled = false;
        for (uint m = 0u; m < nm; m++)
        {
            uint wr = wheelRec(cfg.y + mWheel[m]);
            VSet(wr + W_Y, float4(y, yv.yzw));
            VSet(wr + W_F, float4(FnW[mWheel[m]], mFL[m], mFT[m], VD(wr + W_F).w));
            if (!mContact[m]) continue;
            float4 c1 = VD(wr + W_C1);
            Vec6 jt = jac(VD(wr + W_C3).xyz, c1.xyz);
            Vec6 js = jac(VD(wr + W_C4).xyz, lateralLever(wr, c1.xyz, upW));
            // Saturated rows keep their penalty as a chassis-only proximal term.
            mat6AddOuter(H, jt, mKL[m]);
            vec6AddScaled(g, jt, mFL[m]);
            if (!mClampL[m])
            {
                vec6AddScaled(u, jt, mR[m] * mKL[m]);
                coupled = true;
            }
            mat6AddOuter(H, js, VD(wr + W_K2).x);
            vec6AddScaled(g, js, mFT[m]);
            // The ground's share, with wheel spin held at its solved y.
            if (wheelGround(wr) != NO_INDEX)
            {
                groundAdd(cfg.y + mWheel[m], groundJac(wr, VD(wr + W_C3).xyz), mKL[m], mFL[m]);
                groundAdd(cfg.y + mWheel[m], groundJac(wr, VD(wr + W_C4).xyz), VD(wr + W_K2).x, mFT[m]);
            }
        }
        if (coupled) mat6AddOuter(H, u, -1.0 / A);
        VSet(lr + W_BS, float4(0.0, A, 0.0, 0.0));
    }

void vehChassisRows<let N : uint>(uint vr, uint4 cfg, BodyState c, float3 dl, float3 da, bool dual, uint nW,
    float FnW[N], uint contacts, inout Mat6 H, inout Vec6 g)
{
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));
    // Pitch/roll limit (Jolt mMaxPitchRollAngle): one-sided angular AVBD row that
    // keeps the chassis up axis within a cone around world up.
    float cosMax = VD(vr + V_FWD).w;
    if (cosMax > -0.999)
    {
        float3 worldUp = float3(0.0, p.gravityY < 0.0 ? 1.0 : -1.0, 0.0);
        float cu = clamp(dot(upW, worldUp), -1.0, 1.0);
        float theta = acos(cu);
        float thetaMax = acos(clamp(cosMax, -1.0, 1.0));
        float4 o3 = VD(vr + V_O3);
        float lamPR = o3.z;
        float kPR = o3.w;
        float3 axis = cross(upW, worldUp);
        float al = length(axis);
        if (theta > thetaMax - 0.05 && al > 1e-6)
        {
            axis /= al;
            Vec6 j = vec6Zero();
            j.v[3] = -axis.x; j.v[4] = -axis.y; j.v[5] = -axis.z;
            float C = theta - thetaMax + jdot(j, dl, da);
            float4 m = VD(vr + V_MASS);
            float iAx = dot(abs(quatRotate(quatConjugate(c.initialRot), axis)), m.yzw);
            float kFloor = iAx / (p.dt * p.dt);
            kPR = max(kPR, kFloor);
            float fPR = max(0.0, kPR * C + lamPR);
            if (dual)
            {
                lamPR = fPR;
                if (fPR > 0.0) kPR = min(kPR + p.betaAng * abs(C), 1e3 * kFloor);
                fPR = max(0.0, kPR * C + lamPR);
            }
            if (fPR > 0.0)
            {
                mat6AddOuter(H, j, kPR);
                vec6AddScaled(g, j, fPR);
            }
        }
        else if (dual) { lamPR = 0.0; }
        if (dual) VSet(vr + V_O3, float4(o3.xy, lamPR, kPR));
    }

    // Motorcycle lean (Jolt MotorcycleController): a soft angular row about the
    // forward axis toward the target lean, with implicit damping and an explicit
    // integral term; only while every wheel is loaded. Like Jolt, the lean torque
    // comes with a linear force that cancels the load-weighted acceleration it
    // would give the contact patches, so it rolls the bike about its tyres
    // instead of dragging them sideways (that friction would feed back into the
    // target lean).
    float4 bike0 = VD(vr + V_BIKE0);
    if (bike0.x > 0.0 && contacts > 0u)
    {
        uint attached = 0u;
        uint loaded = 0u;
        float3 rBar = float3(0, 0, 0);
        float fnSum = 0.0;
        for (uint wi = 0u; wi < nW; wi++)
        {
            uint wr = wheelRec(cfg.y + wi);
            uint wd = VU(wr + W_C0).z;
            if ((wd & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
            attached++;
            if ((wd & WD_CONTACT) == 0u || FnW[wi] <= 0.0) continue;
            loaded++;
            rBar += FnW[wi] * VD(wr + W_C1).xyz;
            fnSum += FnW[wi];
        }
        if (loaded == attached)
        {
            float3 fwd0 = quatRotate(c.initialRot, vehicleFwdLocal(vr));
            float4 bd = VD(vr + V_BIKE_D);
            float3 tgt = bd.xyz;
            float dAng = -sign(dot(cross(tgt, upW), fwd0)) * acos(clamp(dot(tgt, upW), -1.0, 1.0));
            Vec6 j = vec6Zero();
            j.v[3] = fwd0.x; j.v[4] = fwd0.y; j.v[5] = fwd0.z;
            float phi = jdot(j, dl, da);
            float k = bike0.y;
            float dd = bike0.z / p.dt;
            float f = -k * (dAng - phi) + dd * phi - bike0.w * bd.w;
            mat6AddOuter(H, j, k + dd);
            vec6AddScaled(g, j, f);
            // Torque −f·fwd gives Δω ∝ −f·I⁻¹fwd; the matching linear force is
            // −m·(Δω × r̄), i.e. a gradient of −m·f·(I⁻¹fwd × r̄).
            float4 mI = VD(vr + V_MASS);
            float3 invIf = quatRotate(c.initialRot, quatRotate(quatConjugate(c.initialRot), fwd0) / max(mI.yzw, float3(1e-6, 1e-6, 1e-6)));
            float3 comp = -mI.x * f * cross(invIf, rBar / fnSum);
            g.v[0] += comp.x; g.v[1] += comp.y; g.v[2] += comp.z;
        }
    }

    // Arcade assists (opt-in, explicit torques): air control while airborne and an
    // upright stabiliser, both scaled by the chassis inertia.
    float4 assist = VD(vr + V_ASSIST);
    if (assist.x != 0.0 || assist.y != 0.0 || assist.z != 0.0)
    {
        float3 alphaW = float3(0, 0, 0);
        if (contacts == 0u)
        {
            float forwardIn = VD(vr + V_D1).z;
            float steerIn = VD(vr + V_D2).y;
            float3 axleW = quatRotate(c.initialRot, axleLocal(vr));
            alphaW += axleW * (forwardIn * assist.x) - upW * (steerIn * assist.y);
        }
        if (assist.z > 0.0)
        {
            float3 worldUp = float3(0.0, p.gravityY < 0.0 ? 1.0 : -1.0, 0.0);
            float3 wPerp = c.angularVel - upW * dot(c.angularVel, upW);
            alphaW += cross(upW, worldUp) * assist.z - wPerp * assist.w;
        }
        float3 Ib = VD(vr + V_MASS).yzw;
        float3 local = quatRotate(quatConjugate(c.initialRot), alphaW) * Ib;
        float3 tau = quatRotate(c.initialRot, local);
        g.v[3] -= tau.x; g.v[4] -= tau.y; g.v[5] -= tau.z;
    }

    // Aerodynamics (explicit): drag and downforce at the aero center.
    float4 aero = VD(vr + V_AERO0);
    if (aero.x > 0.0 || aero.y > 0.0)
    {
        float3 vel = c.linearVel;
        float3 fwdW = quatRotate(c.initialRot, vehicleFwdLocal(vr));
        float vf = dot(vel, fwdW);
        float3 F = -aero.x * length(vel) * vel - aero.y * vf * vf * upW;
        float3 a = quatRotate(c.initialRot, VD(vr + V_AERO1).xyz);
        Vec6 jF = jac(F, a);
        vec6AddScaled(g, jF, -1.0);
    }
    // Per-vehicle gravity scale (0 = default world gravity; else extra (scale−1)·g).
    float gScale = aero.z;
    if (gScale != 0.0 && gScale != 1.0)
    {
        float m = VD(vr + V_MASS).x;
        g.v[1] -= (gScale - 1.0) * m * p.gravityY;
    }
}

void assembleVehicleN<let N : uint>(uint v, uint vr, uint4 cfg, BodyState c, float3 dl, float3 da,
    bool dual, out Mat6 H, out Vec6 g)
{
    H = mat6Zero();
    g = vec6Zero();
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));
    uint nW = min(cfg.z, N);

    // Ground blocks are rebuilt with the chassis block, at the same iterate.
    for (uint wi = 0u; wi < nW; wi++)
        if (wheelGround(wheelRec(cfg.y + wi)) != NO_INDEX) groundClear(cfg.y + wi);

    float Lw[N];
    float FnW[N];
    uint contacts = 0u;
    for (uint wi = 0u; wi < nW; wi++)
    {
        float L, Fn;
        bool ct;
        vehSuspension(wi, vr, cfg, c, dl, da, H, g, L, Fn, ct);
        Lw[wi] = L;
        FnW[wi] = Fn;
        if (ct) contacts++;
    }
    vehAntiRoll<N>(vr, cfg, c, nW, Lw, FnW, H, g);
    for (uint li = 0u; li < nW; li++) vehSpinGroup<N>(li, vr, cfg, c, dl, da, dual, nW, FnW, H, g);
    vehChassisRows<N>(vr, cfg, c, dl, da, dual, nW, FnW, contacts, H, g);
}

// Per-wheel scratch sized to the vehicle: thread-private arrays live in memory on
// GPUs, and the spin solve walks them every Newton step, so a car pays for 4
// wheels, not MAX_VEHICLE_WHEELS (identical arithmetic either way).
void assembleVehicle(uint v, uint vr, uint4 cfg, BodyState c, float3 dl, float3 da,
    bool dual, out Mat6 H, out Vec6 g)
{
    if (cfg.z <= 4u) assembleVehicleN<4>(v, vr, cfg, c, dl, da, dual, H, g);
    else assembleVehicleN<MAX_VEHICLE_WHEELS>(v, vr, cfg, c, dl, da, dual, H, g);
}

void writeVehicleBlock(uint at, Mat6 H, Vec6 g, bool active)
{
    float f[32];
    static const uint LOWER[21] = { 0, 6, 7, 12, 13, 14, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35 };
    for (uint k = 0u; k < 21u; k++) f[k] = H.m[LOWER[k]];
    for (uint k = 0u; k < 6u; k++) f[21u + k] = g.v[k];
    f[27] = active ? 1.0 : 0.0;
    for (uint k = 28u; k < 32u; k++) f[k] = 0.0;
    for (uint e = 0u; e < 4u; e++)
    {
        Aabb a;
        a.mn = float3(f[e * 8u], f[e * 8u + 1u], f[e * 8u + 2u]);
        a._pad0 = f[e * 8u + 3u];
        a.mx = float3(f[e * 8u + 4u], f[e * 8u + 5u], f[e * 8u + 6u]);
        a._pad1 = f[e * 8u + 7u];
        gAabbs[at + e] = a;
    }
}

void chassisDisplacement(BodyState c, out float3 dl, out float3 da)
{
    dl = c.position - c.initialPos;
    float4 q = c.orientation;
    if (dot(q, c.initialRot) < 0.0) q = -q;
    da = quatSubVec(q, c.initialRot);
}

// Wheel shell: SHELL_SPHERES spheres on the chassis around the tyre, at 60°…300°
// from straight down (the tyre rows own the bottom), following suspension travel
// L and steering. A detached or removed wheel parks its shell on the static anchor.
void parkWheelShell(uint wr, uint anchor)
{
    uint first = VU(wr + W_TJ).y;
    if (first == NO_INDEX || gColliders[first].body == anchor) return;
    for (uint k = 0u; k < SHELL_SPHERES; k++)
    {
        Collider col = gColliders[first + k];
        col.body = anchor;
        gColliders[first + k] = col;
    }
}

// Separate from parkWheelShell: inlining both paths into one call with constant
// arguments folded the steer quaternion into a constant that naga's SPIR-V
// front end cannot swizzle.
void placeWheelShell(uint wr, uint chassis, uint anchor, bool attached, float L, float steer)
{
    uint first = VU(wr + W_TJ).y;
    if (first == NO_INDEX) return;
    if (!attached)
    {
        parkWheelShell(wr, anchor);
        return;
    }
    float4 hp = VD(wr + W_HP);
    float4 dir = VD(wr + W_DIR);
    float3 down = normalize(dir.xyz);
    float3 fwd = quatRotate(wheelSteerRot(wr, steer), VD(wr + W_FWD).xyz);
    fwd = normalize(fwd - down * dot(fwd, down));
    float3 center = hp.xyz + down * L;
    float rs = max(min(0.5 * dir.w, 0.4 * hp.w), 1e-3);
    float ring = max(hp.w - rs, 0.0);
    for (uint k = 0u; k < SHELL_SPHERES; k++)
    {
        float th = TWO_PI * float(k + 1u) / float(SHELL_SPHERES + 1u);
        // Whole-record store: a partial float3 store mis-translates through naga.
        Collider col = gColliders[first + k];
        col.body = chassis;
        col.localPos = center + (down * cos(th) + fwd * sin(th)) * ring;
        gColliders[first + k] = col;
    }
}

// Track belts (V_QUERY.zw = left/right pod colliders): the pod's surface runs at the
// sprocket's rim speed (world_step reads it from the collider's density as a
// conveyor). A track whose sprocket is gone parks its pod on the static anchor.
void updateTrackBelts(uint vr, uint4 cfg, uint anchor)
{
    uint4 q = VU(vr + V_QUERY);
    uint4 leaders = VU(vr + V_TRACK);
    for (uint side = 0u; side < 2u; side++)
    {
        uint pod = side == 0u ? q.z : q.w;
        uint leader = side == 0u ? leaders.x : leaders.y;
        if (pod == NO_INDEX || leader == NO_INDEX) continue;
        uint wr = wheelRec(cfg.y + leader);
        bool lost = (VU(wr + W_C0).z & (WD_DETACHED | WD_DISABLED)) != 0u;
        float speed = VD(wr + W_W).x * VD(wr + W_HP).w;
        uint first = pod & ((1u << POD_COUNT_SHIFT) - 1u);
        for (uint k = 0u; k < (pod >> POD_COUNT_SHIFT); k++)
        {
            // Whole-record store: a partial store mis-translates through naga.
            Collider col = gColliders[first + k];
            col.body = lost ? anchor : cfg.x;
            col.density = lost ? 0.0 : speed;
            gColliders[first + k] = col;
        }
    }
}

// ---------------------------------------------------------------------------
// vehicle_link: after wheel_query, before the constraint graph. A wheel standing
// on a dynamic body points its tyre joint (JOINT_FLAG_TYRE) at (chassis, ground) so the two
// get different colours and the ground's primal gathers the wheel's ground block;
// the pair sleeps and wakes as one island (Jolt VehicleConstraint::BuildIslands).
// Otherwise the joint parks on the vehicle's static anchor and adds no graph edge.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_link(uint3 tid : SV_DispatchThreadID)
{
    uint w = dispatchIndex(tid);
    if (w >= wheelCountTotal()) return;
    uint wr = wheelRec(w);
    uint4 tj = VU(wr + W_TJ);
    if (tj.x == NO_INDEX) return;
    uint vr = vehRec(VU(wr + W_CFG).x);
    uint chassis = VU(vr + V_CFG).x;
    uint anchor = VU(vr + V_POOL2).z;
    uint4 dyn = VU(wr + W_C0);
    BodyState c = gBodies[chassis];
    uint ground = NO_INDEX;
    if ((dyn.z & WD_CONTACT) != 0u && (dyn.z & (WD_DETACHED | WD_DISABLED)) == 0u
        && dyn.x != NO_INDEX && dyn.x != chassis && bodyIsDynamic(c))
    {
        BodyState g = gBodies[dyn.x];
        // Vehicle and ground form one island: both sleep, or both are awake and
        // linked. A sleeping car on a moving truck (or the reverse) would otherwise
        // stop transmitting its weight.
        bool cs = bodyIsSleeping(c), gs = bodyIsSleeping(g);
        if (bodyIsDynamic(g) && !(cs && gs))
        {
            ground = dyn.x;
            if (cs) wakeBody(chassis);
            if (gs) wakeBody(ground);
        }
    }
    tj.z = ground;
    VSetU(wr + W_TJ, tj);
    // D6 record: JOINT_FLAG_TYRE, ground block index in `motion` (header words
    // only; the vehicle kernels own the rest of the record).
    gJoints[tj.x].bodyA = ground != NO_INDEX ? chassis : anchor;
    gJoints[tj.x].bodyB = ground != NO_INDEX ? ground : anchor;
    gJoints[tj.x].flags = JOINT_FLAG_TYRE;
    gJoints[tj.x].motion = groundBlockIndex(w);
}

// ---------------------------------------------------------------------------
// vehicle_prepare: per-step wheel setup, drivetrain, first block
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_prepare(uint3 tid : SV_DispatchThreadID)
{
    uint v = dispatchIndex(tid);
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    uint4 cfg = VU(vr + V_CFG);
    uint chassis = cfg.x;
    BodyState c = gBodies[chassis];
    uint at = vehicleAabbBlock(v);
    uint4 d3 = VU(vr + V_D3);
    bool active = (cfg.w & VF_ENABLED) != 0u && bodyIsDynamic(c) && !bodyIsSleeping(c);
    d3.y = active ? (d3.y | VS_ACTIVE) : (d3.y & ~VS_ACTIVE);
    VSetU(vr + V_D3, d3);
    if (!active)
    {
        Mat6 H0 = mat6Zero();
        Vec6 g0 = vec6Zero();
        writeVehicleBlock(at, H0, g0, false);
        return;
    }
    // Step-start velocity for vehicle_loads (BODY_PREV's fields are sleep scratch).
    VSet(vr + V_V0, float4(c.linearVel, 0.0));
    VSet(vr + V_W0, float4(c.angularVel, 0.0));
    AvbdParams p = PARAMS;
    float dt = p.dt;
    float4 q0 = c.initialRot;
    float3 fwdW = quatRotate(q0, vehicleFwdLocal(vr));
    float3 upW = quatRotate(q0, vehicleUpLocal(vr));
    float speedFwd = dot(c.linearVel, fwdW);
    float mass = VD(vr + V_MASS).x;

    float tauDrive[MAX_VEHICLE_WHEELS];
    computeDrivetrain(vr, cfg, dt, speedFwd, tauDrive);

    uint nW = min(cfg.z, MAX_VEHICLE_WHEELS);
    uint nContact = 0u;
    for (uint wi = 0u; wi < nW; wi++)
        if ((VU(wheelRec(cfg.y + wi) + W_C0).z & WD_CONTACT) != 0u) nContact++;
    float mShare = mass / float(max(nContact, 1u));
    float brake = VD(vr + V_D1).w;
    float handbrake = VD(vr + V_D2).x;
    uint brakeMask = VU(vr + V_D3).z;

    for (uint wi = 0u; wi < nW; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
        uint wf = VU(wr + W_CFG).y;
        float R = VD(wr + W_HP).w;
        float4 iner = VD(wr + W_INER);
        float4 w = VD(wr + W_W);
        bool contact = (dyn.z & WD_CONTACT) != 0u;
        float inertia = max(iner.x, 1e-6);

        // Wheel spin inertial target (Jolt angular damping, drive torque).
        float omega0 = w.x;
        float yTilde = dt * omega0 * max(0.0, 1.0 - iner.y * dt) + dt * dt * tauDrive[wi] / inertia;
        float mY = inertia / (dt * dt);
        float Fn = VD(wr + W_F).x;
        float tauB = 0.0;
        if ((brakeMask & (1u << wi)) != 0u)
        {
            tauB = brake * iner.z;
            if ((wf & WF_HANDBRAKE) != 0u) tauB += handbrake * iner.w;
        }
        if (contact) tauB += VD(wr + W_MISC).y * Fn * R;
        // Tracked steering: the inner track's brake holds it at ratio × the outer
        // track's speed (a steering brake). Jolt only syncs the spins, which the
        // implicit tyre rows here pull straight back to ground speed.
        float yStar = 0.0;
        if ((cfg.w & VF_TRACKED) != 0u)
        {
            uint4 tl = VU(vr + V_TRACK);
            float4 td = VD(vr + V_TRACK_D);
            bool leftInner = abs(td.x) < abs(td.y);
            uint inner = leftInner ? tl.x : tl.y;
            uint outer = leftInner ? tl.y : tl.x;
            if (abs(td.x) != abs(td.y) && wi == inner && outer < cfg.z && abs(td.z) > 0.0)
            {
                float rat = leftInner ? td.x / td.y : td.y / td.x;
                uint orr = wheelRec(cfg.y + outer);
                float wOut = VD(orr + W_W).x;
                yStar = dt * wOut * rat;
                // Fade in with track speed (full at 1 m/s) so a standing tank can
                // pull away on both tracks instead of pivoting on a locked one.
                tauB += iner.z * saturate(abs(wOut) * VD(orr + W_HP).w);
            }
        }
        VSet(wr + W_LN, float4(VD(wr + W_LN).xyz, yStar));
        VSet(wr + W_Y, float4(yTilde, yTilde, mY, tauB));
        VSet(wr + W_W, float4(w.x, w.y, tauDrive[wi], mShare));

        // Servo warm start / penalty floor.
        float4 lam = VD(wr + W_LAM);
        float4 k2 = VD(wr + W_K2);
        float decay = p.alpha * p.gamma;
        float kFloor = mShare / (dt * dt);
        float kBFloor = (inertia + mShare * R * R) / (dt * dt);
        lam.z = tauB > 0.0 ? clamp(lam.z * decay, -tauB, tauB) : 0.0;
        k2.y = clamp(k2.y * p.gamma, kBFloor, 1e3 * kBFloor);

        if (!contact)
        {
            lam.x = 0.0; lam.y = 0.0;
            lam.w = kFloor; k2.x = kFloor;
            VSet(wr + W_LAM, lam);
            VSet(wr + W_K2, k2);
            VSet(wr + W_C3, float4(VD(wr + W_C3).xyz, 0.0));
            VSet(wr + W_C4, float4(VD(wr + W_C4).xyz, 0.0));
            continue;
        }

        float3 n = VD(wr + W_C2).xyz;
        float invCos = VD(wr + W_C2).w;
        float3 r = VD(wr + W_C1).xyz;
        float steer = VD(wr + W_C5).w;
        float3 fwdLocal = quatRotate(wheelSteerRot(wr, steer), VD(wr + W_FWD).xyz);
        float3 fw = quatRotate(q0, fwdLocal);
        float3 t = fw - n * dot(n, fw);
        t = dot(t, t) > 1e-12 ? normalize(t) : normalize(cross(quatRotate(q0, axleLocal(vr)), n));
        float3 s = cross(n, t);

        // Spring constants (Jolt): K = m_eff ω², D = 2 m_eff ζ ω, m_eff at the contact along n.
        float4 spr = VD(wr + W_SPR);
        uint mode = asuint(VD(wr + W_SUSP).w);
        float3 rn = cross(suspensionLever(wr, r, q0), n);
        float invMeff = c.invMass + dot(rn, applyInvInertiaWorld(c, rn));
        float meff = invMeff > 1e-12 ? 1.0 / invMeff : mShare;
        float K, D;
        if (mode == 0u)
        {
            float om = TWO_PI * spr.x;
            K = meff * om * om;
            D = 2.0 * meff * spr.y * om;
        }
        else { K = spr.x; D = spr.y; }
        float Kb = 0.0, Db = 0.0;
        if ((wf & WF_RIGID_BUMP) != 0u) { Kb = 100.0 * meff / (dt * dt); Db = 0.0; }
        else if (spr.z > 0.0)
        {
            float ob = TWO_PI * spr.z;
            Kb = meff * ob * ob;
            Db = 2.0 * meff * spr.w * ob;
        }
        VSet(wr + W_K, float4(K, D / dt, Kb, Db / dt));

        // Ground motion over the step (explicit).
        float3 gv = VD(wr + W_GV).xyz;
        VSet(wr + W_C5, float4(gv * dt, steer));
        VSet(wr + W_C3, float4(t, 0.0));
        VSet(wr + W_C4, float4(s, 0.0));

        // Slip (Jolt WheelWV::Update) from step-start velocities.
        float3 vp = c.linearVel + cross(c.angularVel, r) - gv;
        vp -= n * dot(n, vp);
        float vLong = dot(vp, t);
        float denom = sign(vLong) * max(1e-3, abs(vLong));
        float kappa = abs((omega0 * R - vLong) / (vLong == 0.0 ? 1e-3 : denom));
        float vLen = length(vp);
        float alphaSlip = vLen < 1e-3 ? 0.0 : acos(clamp(abs(vLong) / vLen, 0.0, 1.0));
        VSet(wr + W_C3, float4(t, kappa));
        VSet(wr + W_C4, float4(s, alphaSlip));
        float4 tire = VD(wr + W_TIRE);
        uint hit = VU(wr + W_C0).y;
        float surf = gMaterials[gColliders[hit].material & 0x0FFFu].friction;
        // Bound by the curve's falling branch: below the peak slip the AVBD row's
        // penalty already supplies the elastic slip stiffness, so a rolling tyre
        // that starts to slip mid-step keeps its peak grip instead of μ(0) = 0.
        uint hL = asuint(tire.x), hT = asuint(tire.y);
        float muLs = sqrt(max(curveEval(hL, max(kappa, curvePeakX(hL))) * tire.z, 0.0) * surf);
        float muTs = sqrt(max(curveEval(hT, max(degrees(alphaSlip), curvePeakX(hT))) * tire.w, 0.0) * surf);
        float muLp = sqrt(max(curvePeak(asuint(tire.x)) * tire.z, 0.0) * surf);
        float muTp = sqrt(max(curvePeak(asuint(tire.y)) * tire.w, 0.0) * surf);
        VSet(wr + W_MU, float4(muLs, muTs, muLp, muTp));

        // Friction warm start (three-avbd: λ ← λ·α·γ, k ← clamp(k·γ)).
        lam.x *= decay; lam.y *= decay;
        lam.w = clamp(lam.w * p.gamma, kFloor, 1e3 * kFloor);
        k2.x = clamp(k2.x * p.gamma, kFloor, 1e3 * kFloor);
        VSet(wr + W_LAM, lam);
        VSet(wr + W_K2, k2);
    }

    // Motorcycle target lean: direction of last step's ground force (normal and
    // lateral), smoothed, sideways only, clamped to the max lean (Jolt PreCollide).
    float4 bike0 = VD(vr + V_BIKE0);
    if (bike0.x > 0.0)
    {
        float3 worldUp = float3(0.0, p.gravityY < 0.0 ? 1.0 : -1.0, 0.0);
        float3 tgt = float3(0, 0, 0);
        bool all = true;
        for (uint wi = 0u; wi < nW; wi++)
        {
            uint wr = wheelRec(cfg.y + wi);
            uint wd = VU(wr + W_C0).z;
            if ((wd & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
            if ((wd & WD_CONTACT) == 0u) { all = false; continue; }
            float4 f = VD(wr + W_F);
            tgt += VD(wr + W_C2).xyz * f.x - VD(wr + W_C4).xyz * f.z;
        }
        tgt = dot(tgt, tgt) > 1e-12 ? normalize(tgt) : worldUp;
        float4 bd = VD(vr + V_BIKE_D);
        float4 bike1 = VD(vr + V_BIKE1);
        float3 T = bike1.y * bd.xyz + (1.0 - bike1.y) * tgt;
        T -= fwdW * dot(T, fwdW);
        T = dot(T, T) > 1e-12 ? normalize(T) : worldUp;
        float3 adj = worldUp - fwdW * dot(worldUp, fwdW);
        adj = dot(adj, adj) > 1e-12 ? normalize(adj) : worldUp;
        float wAng = -sign(dot(cross(T, adj), fwdW)) * acos(clamp(dot(T, adj), -1.0, 1.0));
        if (abs(wAng) > bike0.x) T = quatRotate(quatAxisAngle(fwdW, sign(wAng) * bike0.x), adj);
        float dAng = -sign(dot(cross(T, upW), fwdW)) * acos(clamp(dot(T, upW), -1.0, 1.0));
        float integ = all ? bd.w + dAng * dt : bd.w * max(0.0, 1.0 - bike1.x * dt);
        VSet(vr + V_BIKE_D, float4(T, integ));
    }

    float3 dl, da;
    chassisDisplacement(c, dl, da);
    Mat6 H;
    Vec6 g;
    assembleVehicle(v, vr, cfg, c, dl, da, false, H, g);
    writeVehicleBlock(at, H, g, true);
    VSet(vr + V_REFP, float4(c.position, VD(vr + V_REFP).w));
    VSet(vr + V_REFQ, c.orientation);
}

// ---------------------------------------------------------------------------
// vehicle_iterate: back-substitute wheel spin, dual update, rebuild block
// ---------------------------------------------------------------------------
// One vehicle's iterate (vehicle_iterate, or soft_iterate's vehicle workgroups).
void vehicleIterateOne(uint v)
{
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    if ((VU(vr + V_D3).y & VS_ACTIVE) == 0u) return;
    uint4 cfg = VU(vr + V_CFG);
    BodyState c = gBodies[cfg.x];
    if (!bodyIsDynamic(c) || bodyIsSleeping(c)) return;

    // Wheel spins are re-solved exactly inside assembleVehicle at this iterate.
    float3 dl, da;
    chassisDisplacement(c, dl, da);
    Mat6 H;
    Vec6 g;
    assembleVehicle(v, vr, cfg, c, dl, da, true, H, g);
    writeVehicleBlock(vehicleAabbBlock(v), H, g, true);
    VSet(vr + V_REFP, float4(c.position, VD(vr + V_REFP).w));
    VSet(vr + V_REFQ, c.orientation);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_iterate(uint3 tid : SV_DispatchThreadID)
{
    vehicleIterateOne(dispatchIndex(tid));
}

// ---------------------------------------------------------------------------
// vehicle_loads: what each part's / wheel's mount carried this step
// ---------------------------------------------------------------------------
// After the final dual (λ = this step's converged contact forces) and
// extract_velocity_sleep (the solved velocities). A part is rigid with the
// chassis, so the wrench through its mount is what the part's own motion needs
// minus what its contacts supply: F = m·(a_p − g) − ΣF_contact, and the moment
// about the mount anchor likewise. A wheel's mount carries its tyre force and
// its shell contacts. Past a limit the part (wheel) is flagged in V_BRK and
// vehicle_input severs it before the next step solves: forward-only, as joints
// break on the final dual and leave the next step's solve.
//
// One thread per vehicle walks the chassis's CSR row in its sorted order, so
// the sums are deterministic (no float atomics). Contacts come through
// READ_CONTACT: canonical and compact hosts publish final λ to gContacts; the
// persistent host compiles vehicle_loads_persistent (vehicle_step_persistent.slang).
static const uint EVENT_MOUNT_BREAK = 6u;
static const uint MAX_VEHICLE_PARTS = 32u;

[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_loads(uint3 tid : SV_DispatchThreadID)
{
    uint v = dispatchIndex(tid);
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    uint4 cfg = VU(vr + V_CFG);
    if ((cfg.w & (VF_ENABLED | VF_BREAKABLE)) != (VF_ENABLED | VF_BREAKABLE)) return;
    // A sleeping (inactive) vehicle carries its resting loads: nothing new to check.
    if ((VU(vr + V_D3).y & VS_ACTIVE) == 0u) return;
    uint chassis = cfg.x;
    BodyState c = gBodies[chassis];
    AvbdParams p = PARAMS;
    float dt = p.dt;
    float4 qi = quatConjugate(c.orientation);
    uint4 pool2 = VU(vr + V_POOL2);
    uint nP = min(pool2.y, MAX_VEHICLE_PARTS);
    uint nW = min(cfg.z, MAX_VEHICLE_WHEELS);
    // Chassis-local contact force and moment (about the origin) per part / wheel shell.
    float3 partF[MAX_VEHICLE_PARTS], partM[MAX_VEHICLE_PARTS];
    float3 wheelF[MAX_VEHICLE_WHEELS], wheelM[MAX_VEHICLE_WHEELS];
    for (uint k = 0u; k < MAX_VEHICLE_PARTS; k++) { partF[k] = float3(0, 0, 0); partM[k] = float3(0, 0, 0); }
    for (uint k = 0u; k < MAX_VEHICLE_WHEELS; k++) { wheelF[k] = float3(0, 0, 0); wheelM[k] = float3(0, 0, 0); }

    // ---- contacts on the chassis (final λ), in its CSR order ----
    uint nJ = gCounters[0].jointCount;
    uint lo = SOLVE_INDEX(chassis);
    uint hi = SOLVE_INDEX(chassis + 1u);
    for (uint e = lo; e < hi; e++)
    {
        uint id = SOLVE_ID(e);
        if (id < nJ) continue; // tyre / user joints: not part mounts
#if defined(ENGINE_PERSISTENT_CONTACTS)
        // Persistent hosts key graph edges to their solver pairs; each contact's
        // colliders live in its cold record.
        SolverPair pair = gSolverPairs[id - nJ];
        uint contactStart = pair.contactStart;
        uint contactCount = pair.countAndSphere & SOLVER_COUNT_MASK;
        bool isA = pair.bodyA == chassis;
#else
        Manifold mf = gManifolds[id - nJ];
        uint contactStart = mf.contactStart;
        uint contactCount = mf.contactCount;
        bool isA = gColliders[mf.colliderA].body == chassis;
        uint ci = isA ? mf.colliderA : mf.colliderB;
#endif
        for (uint k = 0u; k < contactCount; k++)
        {
            Contact ct = READ_CONTACT(contactStart + k);
            if (ct.alive == 0u) continue;
#if defined(ENGINE_PERSISTENT_CONTACTS)
            uint ci = isA ? ct.colliderA : ct.colliderB;
#endif
            // n = A→B; the force on B is −(λN n + λT1 t1 + λT2 t2), on A its opposite.
            float3 fB = -(ct.normal * ct.lambdaN + ct.tangent1 * ct.lambdaT1 + ct.tangent2 * ct.lambdaT2);
            float3 fW = isA ? -fB : fB;
            // Anchors are body-local offsets from the centre of mass: for the
            // chassis that is already the chassis-local point.
            float3 xL = isA ? ct.anchorA : ct.anchorB;
            float3 fL = quatRotate(qi, fW);
            // The part (collider range) or wheel shell that owns the collider.
            bool found = false;
            for (uint pi = 0u; pi < nP && !found; pi++)
            {
                uint4 pc = VU(partRec(pool2.x + pi) + P_CFG);
                if (ci >= pc.x && ci < pc.x + pc.y)
                {
                    partF[pi] += fL;
                    partM[pi] += cross(xL, fL);
                    found = true;
                }
            }
            for (uint wi = 0u; wi < nW && !found; wi++)
            {
                uint shell = VU(wheelRec(cfg.y + wi) + W_TJ).y;
                if (shell != NO_INDEX && ci >= shell && ci < shell + SHELL_SPHERES)
                {
                    wheelF[wi] += fL;
                    wheelM[wi] += cross(xL, fL);
                    found = true;
                }
            }
        }
    }

    // ---- chassis kinematics this step (chassis-local) ----
    float3 aL = quatRotate(qi, (c.linearVel - VD(vr + V_V0).xyz) / dt);
    float3 wL = quatRotate(qi, c.angularVel);
    float3 alL = quatRotate(qi, (c.angularVel - VD(vr + V_W0).xyz) / dt);
    float gScale = VD(vr + V_AERO0).z;
    float3 gL = quatRotate(qi, float3(0.0, p.gravityY * (gScale != 0.0 ? gScale : 1.0), 0.0));

    uint4 brk = VU(vr + V_BRK);
    for (uint pi = 1u; pi < nP; pi++)
    {
        uint pr = partRec(pool2.x + pi);
        float4 com = VD(pr + P_COM);
        if (asuint(com.w) != PS_ATTACHED) continue;
        float4 mount = VD(pr + P_MOUNT);
        float4 load = VD(pr + P_LOAD);
        if (mount.w >= 1e29 && load.x >= 1e29) continue;
        float4 mass = VD(pr + P_MASS);
        float3 r = com.xyz;
        float3 ap = aL + cross(alL, r) + cross(wL, cross(wL, r));
        float3 fm = mass.x * (ap - gL) - partF[pi];
        float3 tauCom = mass.yzw * alL + cross(wL, mass.yzw * wL);
        float3 mContactCom = partM[pi] - cross(r, partF[pi]);
        float3 tm = tauCom - cross(mount.xyz - r, fm) - mContactCom;
        float F = length(fm), Tq = length(tm);
        float ratio = max(F / mount.w, Tq / load.x);
        VSet(pr + P_LOAD, float4(load.x, F, Tq, max(load.w, ratio)));
        uint bit = 1u << pi;
        if (ratio > 1.0 && (brk.z & bit) == 0u && (brk.x & bit) == 0u)
        {
            brk.x |= bit;
            emitEvent(EVENT_MOUNT_BREAK, (v << 8u) | pi, asuint(F), asuint(Tq));
        }
    }

    float3 axl = axleLocal(vr);
    for (uint wi = 0u; wi < nW; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
        float4 wm = VD(wr + W_MOUNT);
        if (wm.x >= 1e29 && wm.y >= 1e29) continue;
        float4 wm2 = VD(wr + W_MOUNT2);
        // Tyre force from the ground on the chassis, as vehicle_post reports it.
        float3 fT = float3(0, 0, 0);
        float3 hitL = float3(0, 0, 0);
        if ((dyn.z & WD_CONTACT) != 0u)
        {
            float4 f = VD(wr + W_F);
            float3 fw = VD(wr + W_C2).xyz * f.x - VD(wr + W_C3).xyz * f.y - VD(wr + W_C4).xyz * f.z;
            fT = quatRotate(qi, fw);
            hitL = quatRotate(qi, VD(wr + W_HIT).xyz - c.position);
        }
        float3 center = VD(wr + W_HP).xyz + VD(wr + W_DIR).xyz * VD(wr + W_O_POS).w;
        float mw = VD(wr + W_MISC).x;
        float3 aw = aL + cross(alL, center) + cross(wL, cross(wL, center));
        float3 fm = mw * (aw - gL) - fT - wheelF[wi];
        float3 mContact = cross(hitL - center, fT) + wheelM[wi] - cross(center, wheelF[wi]);
        float3 tm = -cross(wm2.xyz - center, fm) - mContact;
        // Torque about the spin axis goes to the drive and brakes, not the mount.
        tm -= axl * dot(tm, axl);
        float F = length(fm), Tq = length(tm);
        float ratio = max(F / wm.x, Tq / wm.y);
        VSet(wr + W_MOUNT, float4(wm.x, wm.y, F, Tq));
        VSet(wr + W_MOUNT2, float4(wm2.xyz, max(wm2.w, ratio)));
        uint bit = 1u << wi;
        if (ratio > 1.0 && (brk.w & bit) == 0u && (brk.y & bit) == 0u)
        {
            brk.y |= bit;
            emitEvent(EVENT_MOUNT_BREAK, (v << 8u) | 0x80u | wi, asuint(F), asuint(Tq));
        }
    }
    VSetU(vr + V_BRK, brk);
}

// ---------------------------------------------------------------------------
// vehicle_post: integrate spin, outputs, ground reaction, transmission, sleep
// ---------------------------------------------------------------------------
void updateTransmission(uint vr, uint4 cfg, float dt, float forward, float rpm, float oldRpm, bool slipping)
{
    float4 d0 = VD(vr + V_D0);
    float4 d1 = VD(vr + V_D1);
    int gear = int(round(d0.y));
    int oldGear = gear;
    float clutch = d0.z;
    float switchLeft = d0.w;
    float releaseLeft = d1.x;
    float latencyLeft = d1.y;
    float4 tr0 = VD(vr + V_TR0);
    float4 tr1 = VD(vr + V_TR1);
    int nF = int(asuint(tr1.y));
    int nR = int(asuint(tr1.z));
    uint req = VU(vr + V_IN1).x;
    bool manual = (cfg.w & VF_MANUAL_TRANS) != 0u && req != GEAR_AUTO;
    if (manual)
    {
        gear = clamp(int(req), -nR, nF);
    }
    else
    {
        bool canShiftUp = !slipping && rpm >= oldRpm;
        if (gear == 0 || forward * float(gear) < 0.0)
            gear = forward > 0.0 ? 1 : (forward < 0.0 ? -1 : 0);
        else if (latencyLeft == 0.0)
        {
            if (canShiftUp && rpm > tr0.w)
            {
                if (gear < 0) { if (gear > -nR) gear--; }
                else if (gear < nF) gear++;
            }
            else if (rpm < tr1.x)
            {
                if (gear < 0) { int maxG = forward != 0.0 ? -1 : 0; if (gear < maxG) gear++; }
                else { int minG = forward != 0.0 ? 1 : 0; if (gear > minG) gear--; }
            }
        }
    }
    if (oldGear != gear)
    {
        switchLeft = oldGear != 0 ? tr0.x : 0.0;
        releaseLeft = tr0.y;
        latencyLeft = tr0.z;
        clutch = 0.0;
    }
    else if (switchLeft > 0.0) { switchLeft = max(0.0, switchLeft - dt); clutch = 0.0; }
    else if (releaseLeft > 0.0)
    {
        releaseLeft = max(0.0, releaseLeft - dt);
        clutch = tr0.y > 0.0 ? 1.0 - releaseLeft / tr0.y : 1.0;
    }
    else
    {
        clutch = 1.0;
        latencyLeft = max(0.0, latencyLeft - dt);
    }
    VSet(vr + V_D0, float4(d0.x, float(gear), clutch, switchLeft));
    VSet(vr + V_D1, float4(releaseLeft, latencyLeft, d1.z, d1.w));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_post(uint3 tid : SV_DispatchThreadID)
{
    uint v = dispatchIndex(tid);
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    uint4 cfg = VU(vr + V_CFG);
    uint chassis = cfg.x;
    BodyState c = gBodies[chassis];
    bool active = (VU(vr + V_D3).y & VS_ACTIVE) != 0u;
    AvbdParams p = PARAMS;
    float dt = p.dt;
    float3 fwdLocal = vehicleFwdLocal(vr);
    float3 upLocal = vehicleUpLocal(vr);
    float3 axl = axleLocal(vr);
    float3 fwdW = quatRotate(c.orientation, fwdLocal);
    float3 upW = quatRotate(c.orientation, upLocal);
    uint nW = min(cfg.z, MAX_VEHICLE_WHEELS);
    bool keepAwake = false;
    bool slipping = false;
    uint nContact = 0u;
    uint driveMask = asuint(VD(vr + V_D2).w);
    float3 dl, da;
    chassisDisplacement(c, dl, da);
    float alphaStab = 1.0 - p.alpha;
    updateSeverGrace(vr, cfg, dt);

    uint anchor = VU(vr + V_POOL2).z;
    for (uint wi = 0u; wi < nW; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u)
        {
            parkWheelShell(wr, anchor);
            continue;
        }
        float4 w = VD(wr + W_W);
        float4 hp = VD(wr + W_HP);
        float4 susp = VD(wr + W_SUSP);
        float R = hp.w;
        bool contact = (dyn.z & WD_CONTACT) != 0u;
        if (active)
        {
            float y = VD(wr + W_Y).x;
            w.x = y / dt;
            w.y = fmod(w.y + y, TWO_PI);
            VSet(wr + W_W, w);
        }
        if (abs(w.x) > 0.2) keepAwake = true;

        // Suspension length at the solved pose.
        float L = susp.y;
        float3 F = float3(0, 0, 0); // force from the ground on the chassis
        float4 f = VD(wr + W_F);
        if (contact)
        {
            nContact++;
            float4 c1 = VD(wr + W_C1);
            float4 c2 = VD(wr + W_C2);
            float3 eg = wheelGroundDisp(wr);
            float3 t = VD(wr + W_C3).xyz;
            float3 s = VD(wr + W_C4).xyz;
            Vec6 jn = jac(c2.xyz, suspensionLever(wr, c1.xyz, c.initialRot));
            float un = jdot(jn, dl, da) - dot(c2.xyz, eg);
            L = clamp(c1.w + c2.w * un, susp.x, susp.y);
            F = c2.xyz * f.x - t * f.y - s * f.z;
            if (active)
            {
                // Stick anchors persist across steps (zero-creep parking).
                float3 upS = quatRotate(c.initialRot, upLocal);
                float4 k2 = VD(wr + W_K2);
                float y = VD(wr + W_Y).x;
                float CL = alphaStab * k2.z + jdot(jac(t, c1.xyz), dl, da) - dot(t, eg) - R * y;
                float CT = alphaStab * k2.w + jdot(jac(s, lateralLever(wr, c1.xyz, upS)), dl, da) - dot(s, eg);
                k2.z = (dyn.z & WD_STICK_L) != 0u ? clamp(CL, -0.05, 0.05) : 0.0;
                k2.w = (dyn.z & WD_STICK_T) != 0u ? clamp(CT, -0.05, 0.05) : 0.0;
                VSet(wr + W_K2, k2);
            }
            if ((driveMask & (1u << wi)) != 0u && VD(wr + W_C3).w > 0.1) slipping = true;
            // A moving ground body keeps the vehicle awake (its reaction is already
            // in the solve: two-body tyre rows).
            uint gi = wheelGround(wr);
            if (gi != NO_INDEX && length(gBodies[gi].linearVel) > p.sleepLin) keepAwake = true;
        }
        else
        {
            dyn.z &= ~(WD_STICK_L | WD_STICK_T | WD_SLIDE);
            VSetU(wr + W_C0, dyn);
            if ((driveMask & (1u << wi)) != 0u) slipping = true;
        }
        if (active || !bodyIsSleeping(c))
        {
            // Visual transform at the solved pose.
            float steer = VD(wr + W_C5).w;
            float3 center = transformPoint(c.position, c.orientation, hp.xyz + VD(wr + W_DIR).xyz * L);
            float4 rot = quatNormalize(quatMul(quatMul(c.orientation, wheelSteerRot(wr, steer)), quatAxisAngle(axl, w.y)));
            VSet(wr + W_O_POS, float4(center, L));
            VSet(wr + W_O_ROT, rot);
            placeWheelShell(wr, chassis, anchor, (VU(wr + W_CFG).y & WF_ENABLED) != 0u, L, steer);
            float3 hpW = transformPoint(c.position, c.orientation, hp.xyz);
            float3 hit = VD(wr + W_HIT).xyz;
            float4 qi = quatConjugate(c.orientation);
            float3 torque = cross(hit - hpW, F);
            VSet(wr + W_O_FM, float4(quatRotate(qi, F), f.x));
            VSet(wr + W_O_TM, float4(quatRotate(qi, torque), f.y));
            VSet(wr + W_O_CP, float4(hit, f.z));
            uint outFlags = dyn.z;
            VSet(wr + W_O_ST, float4(w.x, VD(wr + W_C3).w, degrees(VD(wr + W_C4).w), asfloat(outFlags)));
            VSet(wr + W_O_N, float4(VD(wr + W_C2).xyz, asfloat(contact ? dyn.x : NO_INDEX)));
        }
    }

    if ((cfg.w & VF_TRACKED) != 0u) updateTrackBelts(vr, cfg, anchor);

    float speedFwd = dot(c.linearVel, fwdW);
    float4 d0 = VD(vr + V_D0);
    float rpm = d0.x * RAD_PER_S_TO_RPM;
    if (active && (cfg.w & VF_DIRECT_DRIVE) == 0u)
    {
        float oldRpm = VD(vr + V_REFP).w;
        if ((cfg.w & VF_TRACKED) != 0u)
        {
            float4 td = VD(vr + V_TRACK_D);
            // Jolt: the sprocket is usually off the ground, so ignore wheel slip and
            // shift up whenever both tracks roll the same way.
            slipping = td.x * td.y <= 0.0;
        }
        updateTransmission(vr, cfg, dt, VD(vr + V_D1).z, rpm, oldRpm, slipping);
        d0 = VD(vr + V_D0);
    }
    float3 latW = cross(upW, fwdW);
    VSet(vr + V_O0, float4(speedFwd, rpm, d0.y, asfloat(VU(vr + V_D3).y)));
    VSet(vr + V_O1, float4(dot(c.linearVel, latW), dot(c.angularVel, upW), d0.z, asfloat(nContact)));

    // Sleep gating: intent, spinning wheels, shifting, or moving ground keep it awake.
    if (!active) return;
    float forward = VD(vr + V_D1).z;
    float4 d1 = VD(vr + V_D1);
    bool shifting = d0.w > 0.0 || d1.x > 0.0;
    float minRpm = VD(vr + V_ENG0).y;
    bool engineIdle = (cfg.w & VF_DIRECT_DRIVE) != 0u || rpm <= 1.01 * minRpm || VU(vr + V_O3).x == 0u;
    if (abs(forward) > 1e-4 || shifting || !engineIdle || (cfg.w & VF_CAN_SLEEP) == 0u) keepAwake = true;
    if (keepAwake)
    {
        gCcdFlags[chassis] = 0u;
        gBodies[chassis].sleepTimer = 0.0;
    }
}

// ---------------------------------------------------------------------------
// raycast_batch: public scene query (one workgroup per ray)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void raycast_batch(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint ray = gid.x + gid.y * 32768u;
    uint lane = lid.x;
    uint4 misc = VU(VH_MISC);
    bool live = ray < misc.x;
    uint rr = VU(VH_POOLS).w + ray * RAY_STRIDE;
    float4 a = live ? VD(rr) : float4(0, 0, 0, 0);
    float4 b = live ? VD(rr + 1u) : float4(0, -1, 0, 0);
    float3 dir = live ? normalize(b.xyz) : float3(0, -1, 0);
    rayLaneScan(lane, live, a.xyz, dir, a.w, asuint(b.w), 0.0, -2.0, 0.0);
    rayReduce(lane);
    if (!live || lane != 0u) return;
    uint best = gsRayC[0];
    if (best == NO_INDEX)
    {
        VSet(rr + 2u, float4(0, 0, 0, -1.0));
        VSet(rr + 3u, float4(0, 0, 0, asfloat(NO_INDEX)));
        return;
    }
    float t;
    float3 n;
    rayCollider(best, a.xyz, dir, 0.0, t, n);
    // A ray starting inside a collider hits it at its origin.
    float tHit = max(gsRayT[0], 0.0);
    if (gsRayT[0] < 0.0) n = -dir;
    VSet(rr + 2u, float4(a.xyz + dir * tHit, tHit));
    VSet(rr + 3u, float4(n, asfloat(best)));
}
