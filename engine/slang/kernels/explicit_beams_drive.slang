// Raycast wheels for explicit vehicles (CPU reference engine_softbody::explicit_drive,
// records engine_exec::explicit_plan::drive). Included by explicit_beams.slang.
//
// Off (the drive header's .x is 0: no vehicle has raycast wheels) nothing here runs.
// On, at the start of every step, before the node phase: lane 0 runs each vehicle's
// wheels (rays, suspension, tyres, spin) in the CPU's order and adds the wheel forces
// to the hub nodes' contact accumulators (fixed point, as self-collision's), sets
// gsHit so the node phase reads them, and a barrier follows. The barrier is under
// gsUniformDrive (workgroupUniformLoad, uniform). Strict arithmetic in the CPU's order.

groupshared uint gsUniformDrive;

static const uint DRV_VEH = 20u;
static const uint DRV_WHEEL = 23u;
static const uint DRV_MAX_WHEELS = 16u;
static const float DRV_PARALLEL = 1e-12;
static const float DRV_FLT_MAX = 3.40282347e38;

uint driveRec() { return coupleRec() + 2u; }
uint driveBase(uint v) { return gMisc[driveRec()].y + v * gMisc[driveRec()].z; }
float4 DF(uint r) { return asfloat(gMisc[r]); }
void DSet(uint r, float4 x) { gMisc[r] = asuint(x); }

float drvSgn(float x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }
float drvClamp(float x, float lo, float hi) { return min(max(x, lo), hi); }
float drvLerp(float a, float b, float t) { return nf(a + smul(nf(b - a), t)); }
float3 drvNrm(float3 v)
{
    float l = ssqrt(sdot(v, v));
    if (l > 0.0) return sdiv3(v, l);
    return v;
}
float drvMoveToward(float cur, float target, float maxDelta)
{
    if (maxDelta <= 0.0) return target;
    float d = nf(target - cur);
    if (abs(d) <= maxDelta) return target;
    return nf(cur + smul(drvSgn(d), maxDelta));
}

// Curve of `count` points from record `at` (two per record): explicit_drive::curve_eval.
float2 drvPoint(uint at, uint k)
{
    float4 r = DF(at + k / 2u);
    return (k % 2u) == 0u ? r.xy : r.zw;
}
float drvCurve(uint at, uint count, float x)
{
    if (count == 0u) return 1.0;
    float2 p0 = drvPoint(at, 0u);
    if (x <= p0.x) return p0.y;
    for (uint i = 1u, iStop = EB_END(288, i, count, 1u); i < iStop; i++)
    {
        float2 a = drvPoint(at, i - 1u);
        float2 b = drvPoint(at, i);
        if (x <= b.x)
        {
            float t = sdiv(nf(x - a.x), max(nf(b.x - a.x), 1e-9));
            return nf(a.y + smul(nf(b.y - a.y), t));
        }
    }
    return drvPoint(at, count - 1u).y;
}

// explicit_drive::sin_cos.
float2 drvSinCos(float x)
{
    float x2 = smul(x, x);
    float s = asfloat(0xb2d7322bu);
    s = nf(smul(s, x2) + asfloat(0x3638ef1du));
    s = nf(smul(s, x2) + asfloat(0xb9500d01u));
    s = nf(smul(s, x2) + asfloat(0x3c088889u));
    s = nf(smul(s, x2) + asfloat(0xbe2aaaabu));
    s = nf(smul(s, x2) + 1.0);
    float c = asfloat(0x310f76c7u);
    c = nf(smul(c, x2) + asfloat(0xb493f27eu));
    c = nf(smul(c, x2) + asfloat(0x37d00d01u));
    c = nf(smul(c, x2) + asfloat(0xbab60b61u));
    c = nf(smul(c, x2) + asfloat(0x3d2aaaabu));
    c = nf(smul(c, x2) + asfloat(0xbf000000u));
    c = nf(smul(c, x2) + 1.0);
    return float2(smul(x, s), c);
}

// --- rays (t, normal); .w < 0: no hit ---
float4 drvNone() { return float4(0.0, 0.0, 0.0, -1.0); }

// explicit_drive::slab: the ray (pl, dl) in a frame of axes a0..a2 against lo..hi.
float4 drvSlab(float3 pl, float3 dl, float3 lo, float3 hi, float3 a0, float3 a1, float3 a2)
{
    float tIn = -DRV_FLT_MAX;
    float tOut = DRV_FLT_MAX;
    float3 n = float3(0.0, 0.0, 0.0);
    for (uint a = 0u; a < 3u; a++)
    {
        float p = pl[a];
        float d = dl[a];
        float l = lo[a];
        float h = hi[a];
        if (abs(d) < DRV_PARALLEL)
        {
            if (p < l || p > h) return drvNone();
            continue;
        }
        float t1 = sdiv(nf(l - p), d);
        float t2 = sdiv(nf(h - p), d);
        float nearT = t1 < t2 ? t1 : t2;
        float farT = t1 < t2 ? t2 : t1;
        float sg = t1 < t2 ? -1.0 : 1.0;
        if (nearT > tIn)
        {
            tIn = nearT;
            float3 ax = a == 0u ? a0 : (a == 1u ? a1 : a2);
            n = sscale(ax, sg);
        }
        if (farT < tOut) tOut = farT;
    }
    if (tIn >= 0.0 && tIn <= tOut) return float4(n, tIn);
    return drvNone();
}

float4 drvSphere(float3 p, float3 d, float3 c, float r)
{
    float3 oc = ssub3(p, c);
    float b = sdot(oc, d);
    float cc = nf(sdot(oc, oc) - smul(r, r));
    if (cc <= 0.0) return drvNone();
    float disc = nf(smul(b, b) - cc);
    if (disc < 0.0) return drvNone();
    float t = nf(-b - ssqrt(disc));
    if (t < 0.0) return drvNone();
    return float4(drvNrm(ssub3(sadd3(p, sscale(d, t)), c)), t);
}

// explicit_drive::ray_tube: (t, axial coordinate) in .xy, normal in .yzw of a pair.
bool drvTube(float3 p, float3 d, float3 c, float3 a, float r, out float t, out float y, out float3 n)
{
    t = 0.0; y = 0.0; n = float3(0.0, 0.0, 0.0);
    float3 oc = ssub3(p, c);
    float da = sdot(d, a);
    float oa = sdot(oc, a);
    float3 dp = ssub3(d, sscale(a, da));
    float3 op = ssub3(oc, sscale(a, oa));
    float qa = sdot(dp, dp);
    if (qa < DRV_PARALLEL) return false;
    float qb = sdot(op, dp);
    float qc = nf(sdot(op, op) - smul(r, r));
    if (qc <= 0.0) return false;
    float disc = nf(smul(qb, qb) - smul(qa, qc));
    if (disc < 0.0) return false;
    t = sdiv(nf(-qb - ssqrt(disc)), qa);
    if (t < 0.0) return false;
    y = nf(oa + smul(da, t));
    n = drvNrm(sadd3(op, sscale(dp, t)));
    return true;
}

float4 drvCloser(float4 best, float4 h)
{
    if (h.w >= 0.0 && (best.w < 0.0 || h.w < best.w)) return h;
    return best;
}

// One box slot (the frame's slots: boxes, then shapes; explicit_drive::ray_cast).
float4 drvSlot(uint slot, float3 p, float3 d)
{
    uint4 lo4 = gStatic[gP.boxBase + slot * 2u];
    uint4 hi4 = gStatic[gP.boxBase + slot * 2u + 1u];
    float3 X = float3(1.0, 0.0, 0.0), Y = float3(0.0, 1.0, 0.0), Z = float3(0.0, 0.0, 1.0);
    if (lo4.w == 0u) return drvSlab(p, d, asfloat(lo4.xyz), asfloat(hi4.xyz), X, Y, Z);
    uint at = gP.boxBase + hi4.w;
    float4 p0 = asfloat(gStatic[at]);
    float4 p1 = asfloat(gStatic[at + 1u]);
    if (lo4.w == SHAPE_OBOX)
    {
        float4 p2 = asfloat(gStatic[at + 2u]);
        float4 p3 = asfloat(gStatic[at + 3u]);
        float3 h = float3(p0.w, p1.w, p2.w);
        float3 o = ssub3(p, p0.xyz);
        float3 pl = float3(sdot(p1.xyz, o), sdot(p2.xyz, o), sdot(p3.xyz, o));
        float3 dl = float3(sdot(p1.xyz, d), sdot(p2.xyz, d), sdot(p3.xyz, d));
        return drvSlab(pl, dl, -h, h, p1.xyz, p2.xyz, p3.xyz);
    }
    if (lo4.w == SHAPE_CYLINDER)
    {
        float3 c = p0.xyz;
        float radius = p0.w;
        float hh = p1.x;
        float4 best = drvNone();
        if (abs(d.y) >= DRV_PARALLEL)
        {
            float y = d.y < 0.0 ? nf(c.y + hh) : nf(c.y - hh);
            float ny = d.y < 0.0 ? 1.0 : -1.0;
            float t = sdiv(nf(y - p.y), d.y);
            float3 q = ssub3(sadd3(p, sscale(d, t)), c);
            if (t >= 0.0 && nf(smul(q.x, q.x) + smul(q.z, q.z)) <= smul(radius, radius)) best = float4(0.0, ny, 0.0, t);
        }
        float t, y;
        float3 n;
        if (drvTube(p, d, c, Y, radius, t, y, n) && y >= -hh && y <= hh) best = drvCloser(best, float4(n, t));
        return best;
    }
    if (lo4.w == SHAPE_CAPSULE)
    {
        float3 a = p0.xyz;
        float3 b = p1.xyz;
        float radius = p0.w;
        float4 best = drvSphere(p, d, a, radius);
        best = drvCloser(best, drvSphere(p, d, b, radius));
        float3 ab = ssub3(b, a);
        float l = ssqrt(sdot(ab, ab));
        if (l > 0.0)
        {
            float t, y;
            float3 n;
            if (drvTube(p, d, a, sdiv3(ab, l), radius, t, y, n) && y >= 0.0 && y <= l) best = drvCloser(best, float4(n, t));
        }
        return best;
    }
    return drvSphere(p, d, p0.xyz, p0.w);
}

// `p` in the vehicle's frame (origin `ow`): the ground's height in that frame; the box
// slots in the World's (the ray's start moved to the World).
float4 drvRay(float3 p, float3 ow, float3 d, float tMax, float worldGround, uint boxOff, uint boxCount)
{
    float4 best = drvNone();
    float ground = nf(worldGround - ow.y);
    if (d.y < -DRV_PARALLEL && p.y >= ground) best = float4(0.0, 1.0, 0.0, sdiv(nf(ground - p.y), d.y));
    float3 pw = sadd3(ow, p);
    for (uint b = 0u, bStop = EB_END(289, b, boxCount, 1u); b < bStop; b++) best = drvCloser(best, drvSlot(boxOff + b, pw, d));
    if (best.w > tMax) return drvNone();
    return best;
}

// explicit_drive::spin_solve: (new spin, tyre force).
float2 drvSpin(float m, float u, float ft, float j, float r, float wd, float b, float a)
{
    float h = STEP;
    float f0 = drvClamp(nf(sdiv(smul(m, nf(smul(0.0, r) - u)), h) - ft), -a, a);
    float g0 = nf(sdiv(smul(j, nf(0.0 - wd)), h) + smul(r, f0));
    if (nf(g0 - b) <= 0.0 && nf(g0 + b) >= 0.0) return float2(0.0, f0);
    float s = nf(g0 + b) < 0.0 ? 1.0 : -1.0;
    float num = nf(nf(nf(smul(j, wd) + smul(smul(r, m), u)) + smul(smul(h, r), ft)) - smul(smul(h, s), b));
    float den = nf(j + smul(smul(m, r), r));
    float wu = sdiv(num, den);
    float fu = nf(sdiv(smul(m, nf(smul(wu, r) - u)), h) - ft);
    float w = wu;
    float f = fu;
    if (!(abs(fu) <= a))
    {
        f = smul(drvSgn(fu), a);
        w = nf(wd - sdiv(smul(h, nf(smul(r, f) + smul(s, b))), j));
    }
    if (smul(s, w) < 0.0) w = 0.0;
    return float2(w, f);
}

// The node phase's force on node gi before the wheels (beamSum + contact sums).
float3 drvForceNow(uint gi, uint beamOff, uint groupOff, uint groupCount, uint step)
{
    uint4 info = gStatic[gP.nodeInfoBase + gi];
    float3 f = beamSum(gNodes[gi * 3u + 2u].xyz, info, beamOff, groupOff, groupCount, (step + 1u) & 1u, false);
    return f + accForce(gi);
}

// One vehicle's wheels (explicit_drive::drive_pass) in four parts, each in the CPU's
// order: drvCtx (frame, speed, inputs: section 1), drvRays (one wheel's ray and
// suspension: section 2), drvBars (the anti-roll bars), drvTyre (one wheel's tyre,
// spin and hub forces: sections 3 and 4) and drvEnd. driveWheelsSplit runs them on
// one lane, or a wheel per lane when no two wheels share a hub node (a
// wheel's tyre reads its hub nodes' accumulators, which only its own forces then
// touch, and its own records), else on one lane: the same arithmetic, so the same
// bits. Between the parts each wheel's (length, load) and hit are kept in its scratch
// records (DRV_SCRATCH).
static const uint DRV_SCRATCH = 21u;

struct DrvCtx
{
    uint v;
    uint step; // the step (driveWheelsSplit's; 0 on the ahead schedules)
    uint vr;
    uint nw;
    uint nodeOff;
    uint beamOff;
    uint boxOff;
    uint boxCount;
    uint groupOff;
    uint groupCount;
    uint c5y;
    uint arCount;
    float ground;
    // The wheels' frame: the vehicle's (o is 0: positions relative to its origin); ow
    // its origin, added at the World's boundary (ground, box slots, reported centres).
    float3 o;
    float3 ow;
    float3 fw;
    float3 up;
    float3 left;
    float speed;
    float forward;
    float brake;
    float hand;
    float steer;
    float scale;
    float top;
    float cut;
    float4 c3;
    float4 c4;
    float4 c5;
    float4 c6;
};

uint drvWheel(DrvCtx c, uint k) { return c.vr + DRV_VEH + k * DRV_WHEEL; }

// Wheel k's hub: (mass, position, velocity) (section 1's per-wheel terms).
void drvHub(DrvCtx c, uint k, out float m, out float3 x, out float3 vel)
{
    uint4 w0 = gMisc[drvWheel(c, k)];
    uint g1 = c.nodeOff + w0.x;
    uint g2 = c.nodeOff + w0.y;
    float4 P1 = gNodes[g1 * 3u];
    float3 V1 = gNodes[g1 * 3u + 1u].xyz;
    if (w0.x == w0.y)
    {
        m = P1.w;
        x = sadd3(c.o, P1.xyz);
        vel = V1;
    }
    else
    {
        float4 P2 = gNodes[g2 * 3u];
        float3 V2 = gNodes[g2 * 3u + 1u].xyz;
        m = nf(P1.w + P2.w);
        x = sdiv3(sadd3(sscale(sadd3(c.o, P1.xyz), P1.w), sscale(sadd3(c.o, P2.xyz), P2.w)), m);
        vel = sdiv3(sadd3(sscale(V1, P1.w), sscale(V2, P2.w)), m);
    }
}

// Section 1: the frame, speed and inputs (false: no wheels).
bool drvCtx(uint v, out DrvCtx c)
{
    c = (DrvCtx)0;
    uint vr = driveBase(v);
    c.v = v;
    c.vr = vr;
    uint4 c2 = gMisc[vr + 2u];
    uint nw = min(c2.x, DRV_MAX_WHEELS);
    c.nw = nw;
    if (nw == 0u) return false;
    uint4 r0 = gStatic[v * 4u];
    uint4 r1 = gStatic[v * 4u + 1u];
    uint4 r2 = gStatic[v * 4u + 2u];
    uint4 r3 = gStatic[v * 4u + 3u];
    c.nodeOff = r0.x;
    c.beamOff = r0.z;
    c.boxOff = r1.z;
    c.boxCount = r1.w;
    c.groupOff = r2.z;
    c.groupCount = r2.w;
    c.ground = asfloat(r3.y);
    c.ow = asfloat(gMisc[v].xyz);
    c.o = float3(0.0, 0.0, 0.0);
    float h = STEP;
    float3 xc = sadd3(c.o, gNodes[(c.nodeOff + c2.y) * 3u].xyz);
    float3 xb = sadd3(c.o, gNodes[(c.nodeOff + c2.z) * 3u].xyz);
    float3 xl = sadd3(c.o, gNodes[(c.nodeOff + c2.w) * 3u].xyz);
    c.fw = drvNrm(ssub3(xc, xb));
    c.up = drvNrm(scross(c.fw, ssub3(xl, xc)));
    c.left = scross(c.up, c.fw);
    float msum = 0.0;
    float3 psum = float3(0.0, 0.0, 0.0);
    for (uint k = 0u, kStop = EB_END(290, k, nw, 1u); k < kStop; k++)
    {
        float m;
        float3 x, vel;
        drvHub(c, k, m, x, vel);
        msum = nf(msum + m);
        psum = sadd3(psum, sscale(vel, m));
    }
    float3 vm = sdiv3(psum, msum);
    float speed = sdot(vm, c.fw);
    c.speed = speed;
    float4 inp = DF(vr);
    float4 sm = DF(vr + 1u);
    c.c3 = DF(vr + 3u);
    c.c4 = DF(vr + 4u);
    uint4 c5u = gMisc[vr + 5u];
    c.c5 = asfloat(c5u);
    c.c5y = c5u.y;
    c.c6 = DF(vr + 6u);
    c.arCount = min(gMisc[vr + 6u].w, 8u);
    uint speedPts = gMisc[vr + 7u].x;
    float fin = drvClamp(inp.x, -1.0, 1.0);
    float sin_ = drvClamp(inp.y, -1.0, 1.0);
    float bin = drvClamp(inp.z, 0.0, 1.0);
    float hin = drvClamp(inp.w, 0.0, 1.0);
    if ((c5u.y & 1u) != 0u && smul(fin, speed) < 0.0 && abs(speed) > 1.0)
    {
        bin = max(bin, abs(fin));
        fin = 0.0;
    }
    c.forward = drvMoveToward(sm.x, fin, smul(c.c4.z, h));
    c.brake = drvMoveToward(sm.y, bin, smul(c.c4.w, h));
    c.hand = drvMoveToward(sm.z, hin, smul(c.c5.x, h));
    float target = smul(sin_, drvCurve(vr + 8u, speedPts, abs(speed)));
    c.steer = drvMoveToward(sm.w, target, smul(c.c3.z, h));
    float sp = abs(speed);
    c.scale = 1.0;
    c.top = c.c3.y;
    if (c.top > 0.0 && smul(c.forward, speed) > 0.0)
        c.scale = sp < sdiv(c.top, 3.0) ? 1.0 : drvClamp(sdiv(nf(c.top - sp), sdiv(smul(c.top, 2.0), 3.0)), 0.0, 1.0);
    c.cut = c.c6.y;
    if ((c5u.y & 2u) != 0u)
    {
        float vReg = sdiv(sp, max(sp, 3.0));
        float kMax = 0.0;
        for (uint k = 0u, kStop = EB_END(291, k, nw, 1u); k < kStop; k++)
        {
            uint wr = drvWheel(c, k);
            if ((gMisc[wr].z & 1u) != 0u && DF(wr + 16u).w != 0.0)
                kMax = max(kMax, sdiv(smul(DF(wr + 16u).y, vReg), max(DF(wr + 5u).z, 0.03)));
        }
        float cutTarget = nf(1.0 - drvClamp(nf(1.0 - sdiv(nf(kMax - 0.8), 1.0)), 0.2, 1.0));
        c.cut = drvLerp(c.cut, cutTarget, c.c6.x);
    }
    return true;
}

// --- Wheels on rigid bodies (engine_softbody::explicit_drive DriveConfig::see_bodies;
// the drive flags' bit 8) ---
// With the vehicle's coupling on, the rays also meet its proxy shapes at their bodies'
// poses now (after the step's integration: coupleStep runs before driveStep), the
// nearest of every surface; a wheel on a body takes the two-body form (the hub's
// velocity relative to the body's surface at the contact point, the reduced mass and
// the share of the hub's force, the body's inverse mass at the point split among this
// vehicle's wheels on it and the vehicles holding it), and the wheel force's reaction
// goes onto the shape's step sums at the contact point (fixed point, as node contact's).
// Positions are in the vehicle frame, as the coupling's records. The scratch record's
// .z holds the shape + 1 (0: a static surface). Its own functions behind early returns.
bool drvBodiesFlag(DrvCtx c) { return (c.c5y & 8u) != 0u && (gMisc[coupleRec()].x != 0u || gMisc[driveRec()].w != 0u); }
uint drvPartnerBase();
// (The coupling on, or other vehicles' records: wheelsPartners.)
bool drvSeeBodies(DrvCtx c) { return (c.c5y & 8u) != 0u && (gsCpl[0] != 0u || drvPartnerBase() != 0u); }

// --- Wheels on other vehicles (a contact group's members; explicit_contact.rs
// ExplicitBeamsWorld::wheels_begin, explicit_drive.rs ray_triangle) ---
// The drive header's .w: the partner records' base (0: none), DRV_PARTNER a wheel at
// base + (v · wheels room + k) · DRV_PARTNER (explicit_plan::drive::PARTNER_RECS):
//   0  the hit (normal, t), or during the rays the ray's start (World) and its reach
//   1  the triangle's weights (w0, w1, w2) and its global index
//   2  (the step it is for + 1, the partner vehicle, 0, 0)
//   3  the wheel's staged reaction on the triangle (force, 1: staged), applied before
//      the next step (the vehicle-contact pass's timing)
//   4  the triangle point's velocity and two-body inverse mass (unsplit), or during the
//      rays the ray's direction
// A group's members' rays run at each step's start where every member's state is final
// (wheelsPartners: contactGroup's step start, the split path's own dispatch), so a
// member's step reads only its own records. The scratch record's .z is DRV_PARTNER_HIT
// for a wheel on another vehicle.
static const uint DRV_PARTNER = 5u;
static const uint DRV_PARTNER_HIT = 0xFFFFFFFFu;
uint drvPartnerBase() { return gMisc[driveRec()].w; }
uint drvPartnerAt(uint v, uint k)
{
    uint nwMax = (gMisc[driveRec()].z - DRV_VEH) / (DRV_WHEEL + DRV_PRE);
    return drvPartnerBase() + (v * nwMax + k) * DRV_PARTNER;
}

// explicit_drive::ray_triangle: (normal facing the ray's start, t), .w < 0 none; the
// weights in `w`.
float4 drvTriangle(float3 p, float3 d, float3 x0, float3 x1, float3 x2, out float3 w)
{
    w = float3(0.0, 0.0, 0.0);
    float3 e1 = ssub3(x1, x0);
    float3 e2 = ssub3(x2, x0);
    float3 pv = scross(d, e2);
    float det = sdot(e1, pv);
    if (abs(det) < DRV_PARALLEL) return drvNone();
    float3 tv = ssub3(p, x0);
    float u = sdiv(sdot(tv, pv), det);
    if (!(u >= 0.0 && u <= 1.0)) return drvNone();
    float3 qv = scross(tv, e1);
    float v = sdiv(sdot(d, qv), det);
    if (!(v >= 0.0 && nf(u + v) <= 1.0)) return drvNone();
    float t = sdiv(sdot(e2, qv), det);
    if (!(t >= 0.0)) return drvNone();
    float3 n = drvNrm(scross(e1, e2));
    if (sdot(n, d) > 0.0) n = -n;
    w = float3(nf(nf(1.0 - u) - v), u, v);
    return float4(n, t);
}

// The ray p + t·d (vehicle frame) against convex planes from outside (explicit_drive::ray_planes).
float4 drvHullRay(uint v, uint off, uint count, float4 r, float3 c, float3 p, float3 d)
{
    float tIn = -DRV_FLT_MAX;
    float tOut = DRV_FLT_MAX;
    float3 n = float3(0.0, 0.0, 0.0);
    for (uint i = 0u, iStop = EB_END(292, i, count, 1u); i < iStop; i++)
    {
        float4 pl = asfloat(gMisc[cplPlane(v, off + i)]);
        float3 wn = cplRot(r, pl.xyz);
        float o = nf(pl.w + sdot(wn, c));
        float dist = nf(sdot(wn, p) - o);
        float den = sdot(wn, d);
        if (abs(den) < DRV_PARALLEL)
        {
            if (dist > 0.0) return drvNone();
            continue;
        }
        float t = sdiv(-dist, den);
        if (den < 0.0)
        {
            if (t > tIn) { tIn = t; n = wn; }
        }
        else if (t < tOut) tOut = t;
    }
    if (tIn >= 0.0 && tIn <= tOut) return float4(n, tIn);
    return drvNone();
}

// The ray against vehicle v's proxy shape k (explicit_drive::ray_proxy).
float4 drvProxy(uint v, uint k, float3 p, float3 d)
{
    uint at = cplShape(v, k);
    uint4 h = gMisc[at];
    uint b = cplBody(v, h.y);
    float4 c0 = asfloat(gMisc[b]);
    float4 q = asfloat(gMisc[b + 1u]);
    float4 lp = asfloat(gMisc[at + 1u]);
    float4 lq = asfloat(gMisc[at + 2u]);
    float4 pr = asfloat(gMisc[at + 3u]);
    float3 c = sadd3(c0.xyz, cplRot(q, lp.xyz));
    float4 r = cplQmul(q, lq);
    if (h.x == CPL_BOX)
    {
        float3 a0 = cplRot(r, float3(1.0, 0.0, 0.0));
        float3 a1 = cplRot(r, float3(0.0, 1.0, 0.0));
        float3 a2 = cplRot(r, float3(0.0, 0.0, 1.0));
        float3 o = ssub3(p, c);
        float3 hh = float3(lp.w, pr.x, pr.y);
        return drvSlab(float3(sdot(a0, o), sdot(a1, o), sdot(a2, o)), float3(sdot(a0, d), sdot(a1, d), sdot(a2, d)), -hh, hh, a0, a1, a2);
    }
    if (h.x == CPL_SPHERE) return drvSphere(p, d, c, lp.w);
    if (h.x == CPL_CAPSULE)
    {
        float3 ax = cplRot(r, float3(0.0, 1.0, 0.0));
        float3 a = ssub3(c, sscale(ax, pr.x));
        float3 e = sadd3(c, sscale(ax, pr.x));
        float4 best = drvSphere(p, d, a, lp.w);
        best = drvCloser(best, drvSphere(p, d, e, lp.w));
        float l = smul(2.0, pr.x);
        if (l > 0.0)
        {
            float t, y;
            float3 n;
            if (drvTube(p, d, a, ax, lp.w, t, y, n) && y >= 0.0 && y <= l) best = drvCloser(best, float4(n, t));
        }
        return best;
    }
    return drvHullRay(v, h.w & 0xFFFFu, h.w >> 16, r, c, p, d);
}

// The velocity of shape k's body at x (vehicle frame).
float3 drvBodyVel(uint v, uint k, float3 x)
{
    uint b = cplBody(v, gMisc[cplShape(v, k)].y);
    float3 c = asfloat(gMisc[b].xyz);
    return sadd3(asfloat(gMisc[b + 2u].xyz), scross(asfloat(gMisc[b + 3u].xyz), ssub3(x, c)));
}

// drvRays with the proxies: the nearest of the static surfaces' hit and every shape's.
void drvRaysBodies(DrvCtx c, uint k)
{
    uint wr = drvWheel(c, k);
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 hp = sadd3(xh, sscale(c.up, w2.z));
    float tMax = nf(w2.y + w1.x);
    float3 d = -c.up;
    float4 ht = drvRay(hp, c.ow, d, tMax, c.ground, c.boxOff, c.boxCount);
    uint shape = 0u;
    uint4 hdr = gsCpl[0] != 0u ? cplVeh(c.v) : uint4(0u, 0u, 0u, 0u);
    bool cull = cplCullOn();
    for (uint s = 0u, sStop = EB_END(293, s, hdr.y, 1u); s < sStop; s++)
    {
        // The shape's grown bounding sphere this step (cplSpheresBy): a segment that
        // misses it misses the shape (lossless).
        if (cull)
        {
            uint sw = cplSphere(c.v, s);
            float3 sc = asfloat(uint3(gNext[sw], gNext[sw + 1u], gNext[sw + 2u]));
            float tc = clamp(dot(sc - hp, d), 0.0, tMax);
            float3 off = sc - (hp + d * tc);
            if (dot(off, off) >= asfloat(gNext[sw + 3u])) continue;
        }
        float4 hs = drvProxy(c.v, s, hp, d);
        if (hs.w >= 0.0 && (ht.w < 0.0 || hs.w < ht.w))
        {
            ht = hs;
            shape = s + 1u;
        }
    }
    // Another vehicle's triangle (its record for this step), after every other surface.
    uint pa = 0u;
    if (drvPartnerBase() != 0u)
    {
        pa = drvPartnerAt(c.v, k);
        float4 hq = DF(pa);
        if (gMisc[pa + 2u].x == c.step + 1u && hq.w >= 0.0 && (ht.w < 0.0 || hq.w < ht.w))
        {
            ht = hq;
            shape = DRV_PARTNER_HIT;
        }
    }
    if (ht.w > tMax)
    {
        ht = drvNone();
        shape = 0u;
    }
    float len = w2.y;
    float load = 0.0;
    float4 hit = drvNone();
    uint on = 0u;
    if (ht.w >= 0.0)
    {
        float cosT = sdot(ht.xyz, c.up);
        if (cosT >= c.c5.z)
        {
            float l = nf(ht.w - w1.x);
            float3 vr = vh;
            if (shape == DRV_PARTNER_HIT) vr = ssub3(vh, DF(pa + 4u).xyz);
            else if (shape != 0u) vr = ssub3(vh, drvBodyVel(c.v, shape - 1u, ssub3(hp, sscale(c.up, ht.w))));
            float rate = sdiv(sdot(vr, ht.xyz), cosT);
            float fs = nf(smul(w2.w, nf(l - w2.y)) + smul(w3.x, rate));
            float f = fs < 0.0 ? -fs : 0.0;
            if (l < w2.x && w3.y > 0.0)
            {
                float fb = nf(smul(w3.y, nf(l - w2.x)) + smul(w3.z, rate));
                if (fb < 0.0) f = nf(f - fb);
            }
            len = l;
            load = f;
            hit = ht;
            on = shape;
        }
    }
    gMisc[wr + DRV_SCRATCH] = uint4(asuint(len), asuint(load), on, 0u);
    DSet(wr + DRV_SCRATCH + 1u, hit);
}

// Wheel k's surface on another vehicle's triangle (explicit_drive::partner_surface):
// (effective mass, share), the point's velocity in vb.
float2 drvPartnerSurface(DrvCtx c, uint k, float m, out float3 vb)
{
    uint pa = drvPartnerAt(c.v, k);
    float4 p4 = DF(pa + 4u);
    vb = p4.xyz;
    uint pv = gMisc[pa + 2u].y;
    uint same = 0u;
    for (uint j = 0u, jStop = EB_END(294, j, c.nw, 1u); j < jStop; j++)
    {
        if (gMisc[drvWheel(c, j) + DRV_SCRATCH].z == DRV_PARTNER_HIT && gMisc[drvPartnerAt(c.v, j) + 2u].y == pv) same++;
    }
    float wb = smul(float(same), p4.w);
    float wn = sdiv(1.0, m);
    return float2(sdiv(1.0, nf(wn + wb)), sdiv(wn, nf(wn + wb)));
}

// Wheel k's surface on its body (explicit_drive::surface): (effective mass, share) and
// the body's velocity at the contact point `at`.
float2 drvSurface(DrvCtx c, uint k, uint shape, float3 n, float3 at, float m, out float3 vb)
{
    uint body = gMisc[cplShape(c.v, shape)].y;
    uint b = cplBody(c.v, body);
    float4 c0 = asfloat(gMisc[b]);
    float4 q = asfloat(gMisc[b + 1u]);
    float4 c2 = asfloat(gMisc[b + 2u]);
    float3 rr = ssub3(at, c0.xyz);
    vb = sadd3(c2.xyz, scross(asfloat(gMisc[b + 3u].xyz), rr));
    if (c0.w == 0.0) return float2(m, 1.0);
    uint same = 0u;
    for (uint j = 0u, jStop = EB_END(295, j, c.nw, 1u); j < jStop; j++)
    {
        uint sj = gMisc[drvWheel(c, j) + DRV_SCRATCH].z;
        if (sj != 0u && gMisc[cplShape(c.v, sj - 1u)].y == body) same++;
    }
    float3 rn = scross(rr, n);
    float3 ii = asfloat(gMisc[b + 4u].xyz);
    float3 irn = cplRot(q, ii * cplRot(float4(-q.xyz, q.w), rn));
    float split = smul(float(max(same, 1u)), max(c2.w, 1.0));
    float wb = smul(split, nf(c0.w + sdot(rn, irn)));
    float wn = sdiv(1.0, m);
    return float2(sdiv(1.0, nf(wn + wb)), sdiv(wn, nf(wn + wb)));
}

// Wheel k's axle (section 2).
float3 drvAxle(DrvCtx c, uint k)
{
    uint wr = drvWheel(c, k);
    uint4 w0 = gMisc[wr];
    bool hubAxles = (c.c5y & 4u) != 0u;
    float3 a = c.left;
    if (w0.x != w0.y && hubAxles)
        a = drvNrm(ssub3(sadd3(c.o, gNodes[(c.nodeOff + w0.y) * 3u].xyz), sadd3(c.o, gNodes[(c.nodeOff + w0.x) * 3u].xyz)));
    if (sdot(a, c.left) < 0.0) a = -a;
    // Less the spawn's toe and camber (in the vehicle frame).
    float3 ar = hubAxles ? DF(wr + 20u).xyz : float3(0.0, 0.0, 1.0);
    return drvNrm(sadd3(sadd3(sscale(c.fw, nf(sdot(a, c.fw) - ar.x)), sscale(c.up, nf(sdot(a, c.up) - ar.y))),
                        sscale(c.left, nf(nf(sdot(a, c.left) - ar.z) + 1.0))));
}

// Section 2 for wheel k: its ray and suspension into its scratch records.
void drvRaysBodies(DrvCtx c, uint k);
bool drvSeeBodies(DrvCtx c);
void drvRays(DrvCtx c, uint k)
{
    // Wheels on rigid bodies: their own function (Metal's codegen of the path below
    // stays as it was).
    if (drvSeeBodies(c))
    {
        drvRaysBodies(c, k);
        return;
    }
    uint wr = drvWheel(c, k);
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 hp = sadd3(xh, sscale(c.up, w2.z));
    float4 ht = drvRay(hp, c.ow, -c.up, nf(w2.y + w1.x), c.ground, c.boxOff, c.boxCount);
    float len = w2.y;
    float load = 0.0;
    float4 hit = drvNone();
    if (ht.w >= 0.0)
    {
        float cosT = sdot(ht.xyz, c.up);
        if (cosT >= c.c5.z)
        {
            float l = nf(ht.w - w1.x);
            float rate = sdiv(sdot(vh, ht.xyz), cosT);
            float fs = nf(smul(w2.w, nf(l - w2.y)) + smul(w3.x, rate));
            float f = fs < 0.0 ? -fs : 0.0;
            if (l < w2.x && w3.y > 0.0)
            {
                float fb = nf(smul(w3.y, nf(l - w2.x)) + smul(w3.z, rate));
                if (fb < 0.0) f = nf(f - fb);
            }
            len = l;
            load = f;
            hit = ht;
        }
    }
    DSet(wr + DRV_SCRATCH, float4(len, load, 0.0, 0.0));
    DSet(wr + DRV_SCRATCH + 1u, hit);
}

// The anti-roll bars, in order, on the wheels' scratch loads.
void drvBars(DrvCtx c)
{
    for (uint i = 0u, iStop = EB_END(296, i, c.arCount, 1u); i < iStop; i++)
    {
        uint4 ar = gMisc[c.vr + 12u + i];
        uint l = ar.x, r = ar.y;
        uint sl = drvWheel(c, l) + DRV_SCRATCH;
        uint sr = drvWheel(c, r) + DRV_SCRATCH;
        if (DF(sl + 1u).w >= 0.0 && DF(sr + 1u).w >= 0.0)
        {
            float4 a = DF(sl);
            float4 b = DF(sr);
            float kar = asfloat(ar.z);
            float diff = nf(a.x - b.x);
            DSet(sl, float4(a.x, max(nf(a.y - smul(kar, diff)), 0.0), a.zw));
            b = DF(sr);
            DSet(sr, float4(b.x, max(nf(b.y + smul(kar, diff)), 0.0), b.zw));
        }
    }
}

// Sections 3 and 4 for wheel k: tyre, spin, state records, hub forces.
void drvTyreBody(DrvCtx c, uint k, uint step);
void drvTyre(DrvCtx c, uint k, uint step)
{
    // On a rigid body (the ray's proxy shape in the scratch record's .z): its own function.
    if (gMisc[drvWheel(c, k) + DRV_SCRATCH].z != 0u)
    {
        drvTyreBody(c, k, step);
        return;
    }
    float h = STEP;
    float wb = c.c4.x;
    float track = c.c4.y;
    float ack = c.c3.w;
    uint wr = drvWheel(c, k);
    uint4 w0 = gMisc[wr];
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float4 w4 = DF(wr + 4u);
    float4 w5 = DF(wr + 5u);
    float4 w6 = DF(wr + 6u);
    float4 st = DF(wr + 15u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 a = drvAxle(c, k);
    float4 sc0 = DF(wr + DRV_SCRATCH);
    float4 hitK = DF(wr + DRV_SCRATCH + 1u);
    float lenK = sc0.x;
    float loadK = sc0.y;
    float angle = 0.0;
    if (w3.w > 0.0)
    {
        angle = smul(-c.steer, w3.w);
        if (ack > 0.0 && w6.y != 0.0 && abs(angle) > 1e-4 && wb > 0.0)
        {
            float2 sc = drvSinCos(abs(angle));
            float turnR = sdiv(wb, sdiv(sc.x, sc.y));
            bool inner = (angle > 0.0) == (w6.y > 0.0);
            float off = inner ? -0.5 : 0.5;
            float ak = sAtan2(wb, max(nf(turnR + smul(off, track)), 1e-3));
            angle = smul(drvSgn(angle), drvLerp(abs(angle), ak, ack));
        }
    }
    float3 uw = drvNrm(ssub3(c.up, sscale(a, sdot(a, c.up))));
    float3 f0 = scross(a, uw);
    float2 sca = drvSinCos(angle);
    float3 fwd = sadd3(sscale(f0, sca.y), sscale(scross(uw, f0), sca.x));
    float j = w1.z;
    float r = w1.x;
    float tau = 0.0;
    if ((w0.z & 1u) != 0u) tau = smul(smul(smul(smul(c.forward, c.c3.x), c.scale), w4.x), nf(1.0 - c.cut));
    if (c.top > 0.0 && smul(smul(drvSgn(c.forward), st.x), r) >= c.top) tau = 0.0;
    float wd = nf(smul(st.x, max(nf(1.0 - smul(w1.w, h)), 0.0)) + sdiv(smul(h, tau), j));
    float b = smul(c.brake, w4.y);
    if ((w0.z & 2u) != 0u) b = nf(b + smul(c.hand, w4.z));
    float wNew;
    float kappa = 0.0;
    float alpha = 0.0;
    float fnOut = 0.0;
    bool contact = hitK.w >= 0.0;
    float3 force = float3(0.0, 0.0, 0.0);
    if (contact)
    {
        float3 n = hitK.xyz;
        float fn_ = loadK;
        b = nf(b + smul(smul(w4.w, fn_), r));
        float3 t = drvNrm(ssub3(fwd, sscale(n, sdot(n, fwd))));
        float3 sd = scross(n, t);
        uint g1 = c.nodeOff + w0.x;
        float3 fpre = drvForceNow(g1, c.beamOff, c.groupOff, c.groupCount, step);
        if (w0.x != w0.y) fpre = sadd3(fpre, drvForceNow(c.nodeOff + w0.y, c.beamOff, c.groupOff, c.groupCount, step));
        float vt = sdot(vh, t);
        float vs = sdot(vh, sd);
        float ft = sdot(fpre, t);
        float fs = sdot(fpre, sd);
        kappa = sdiv(abs(nf(smul(st.x, r) - vt)), max(abs(vt), 1e-3));
        float vlen = ssqrt(nf(smul(vt, vt) + smul(vs, vs)));
        alpha = vlen < 1e-3 ? 0.0 : smul(sAtan2(abs(vs), abs(vt)), asfloat(0x42652ee1u));
        uint4 counts = gMisc[wr];
        uint nl = counts.w & 0xFFFFu;
        uint nt = counts.w >> 16;
        float muL = max(smul(drvCurve(wr + 7u, nl, max(kappa, w5.z)), w5.x), 0.0);
        muL = ssqrt(smul(muL, c.c5.w));
        float muT = max(smul(drvCurve(wr + 11u, nt, max(alpha, w5.w)), w5.y), 0.0);
        muT = ssqrt(smul(muT, c.c5.w));
        float al = smul(muL, fn_);
        float at = smul(muT, fn_);
        float ftRaw = -nf(fs + sdiv(smul(m, vs), h));
        float flRaw = drvSpin(m, vt, ft, j, r, wd, b, DRV_FLT_MAX).y;
        float alEff = al;
        float fT = ftRaw;
        if (at <= 1e-9)
        {
            fT = 0.0;
        }
        else
        {
            float e = 2.0;
            if (al > 1e-9)
            {
                float ql = sdiv(flRaw, al);
                float qt = sdiv(ftRaw, at);
                e = nf(smul(ql, ql) + smul(qt, qt));
            }
            if (e > 1.0)
            {
                float sc = sdiv(1.0, ssqrt(e));
                float tProp = smul(abs(ftRaw), sc);
                float lProp = smul(abs(flRaw), sc);
                float nx = sdiv(kappa, max(w5.z, 0.03));
                float tanA = asfloat(0x4103cf3au);
                if (vlen < 1e-3) tanA = 0.0;
                else if (abs(vt) > 0.0) tanA = min(sdiv(abs(vs), abs(vt)), asfloat(0x4103cf3au));
                float ny = sdiv(tanA, w6.x);
                float nn = ssqrt(nf(smul(nx, nx) + smul(ny, ny)));
                float ws = nn > 1e-6 ? drvClamp(nf(max(nx, ny) - 1.0), 0.0, 1.0) : 0.0;
                float tCap = drvLerp(tProp, nn > 1e-6 ? sdiv(smul(at, ny), nn) : tProp, ws);
                alEff = drvLerp(lProp, nn > 1e-6 ? sdiv(smul(al, nx), nn) : lProp, ws);
                fT = drvClamp(ftRaw, -tCap, tCap);
            }
            else
            {
                float qt = sdiv(ftRaw, at);
                alEff = smul(al, ssqrt(max(nf(1.0 - smul(qt, qt)), 0.0)));
            }
        }
        float2 sol = drvSpin(m, vt, ft, j, r, wd, b, alEff);
        wNew = sol.x;
        force = sadd3(sadd3(sscale(t, sol.y), sscale(sd, fT)), sscale(n, fn_));
        fnOut = fn_;
    }
    else
    {
        wNew = drvSpin(m, 0.0, 0.0, j, r, wd, b, 0.0).x;
    }
    float spin = nf(st.y + smul(wNew, h));
    if (spin > asfloat(0x40490fdbu)) spin = nf(spin - asfloat(0x40c90fdbu));
    else if (spin <= -asfloat(0x40490fdbu)) spin = nf(spin + asfloat(0x40c90fdbu));
    DSet(wr + 15u, float4(wNew, spin, angle, lenK));
    DSet(wr + 16u, float4(fnOut, kappa, alpha, contact ? 1.0 : 0.0));
    DSet(wr + 17u, float4(sadd3(c.ow, sadd3(xh, sscale(c.up, nf(w2.z - lenK)))), 0.0));
    DSet(wr + 18u, float4(scross(uw, fwd), 0.0));
    DSet(wr + 19u, float4(uw, 0.0));
    // 4. Onto the hub nodes, by mass (fixed point, read by the node phase).
    if (contact)
    {
        uint g1 = c.nodeOff + w0.x;
        if (w0.x == w0.y) accAdd(g1, force);
        else
        {
            uint g2 = c.nodeOff + w0.y;
            accAdd(g1, sscale(force, sdiv(gNodes[g1 * 3u].w, m)));
            accAdd(g2, sscale(force, sdiv(gNodes[g2 * 3u].w, m)));
        }
    }
}

void drvTyreBody(DrvCtx c, uint k, uint step)
{
    float h = STEP;
    float wb = c.c4.x;
    float track = c.c4.y;
    float ack = c.c3.w;
    uint wr = drvWheel(c, k);
    uint4 w0 = gMisc[wr];
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float4 w4 = DF(wr + 4u);
    float4 w5 = DF(wr + 5u);
    float4 w6 = DF(wr + 6u);
    float4 st = DF(wr + 15u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 a = drvAxle(c, k);
    float4 sc0 = DF(wr + DRV_SCRATCH);
    float4 hitK = DF(wr + DRV_SCRATCH + 1u);
    uint shape = gMisc[wr + DRV_SCRATCH].z - 1u;
    float lenK = sc0.x;
    float loadK = sc0.y;
    float angle = 0.0;
    if (w3.w > 0.0)
    {
        angle = smul(-c.steer, w3.w);
        if (ack > 0.0 && w6.y != 0.0 && abs(angle) > 1e-4 && wb > 0.0)
        {
            float2 sc = drvSinCos(abs(angle));
            float turnR = sdiv(wb, sdiv(sc.x, sc.y));
            bool inner = (angle > 0.0) == (w6.y > 0.0);
            float off = inner ? -0.5 : 0.5;
            float ak = sAtan2(wb, max(nf(turnR + smul(off, track)), 1e-3));
            angle = smul(drvSgn(angle), drvLerp(abs(angle), ak, ack));
        }
    }
    float3 uw = drvNrm(ssub3(c.up, sscale(a, sdot(a, c.up))));
    float3 f0 = scross(a, uw);
    float2 sca = drvSinCos(angle);
    float3 fwd = sadd3(sscale(f0, sca.y), sscale(scross(uw, f0), sca.x));
    float j = w1.z;
    float r = w1.x;
    float tau = 0.0;
    if ((w0.z & 1u) != 0u) tau = smul(smul(smul(smul(c.forward, c.c3.x), c.scale), w4.x), nf(1.0 - c.cut));
    if (c.top > 0.0 && smul(smul(drvSgn(c.forward), st.x), r) >= c.top) tau = 0.0;
    float wd = nf(smul(st.x, max(nf(1.0 - smul(w1.w, h)), 0.0)) + sdiv(smul(h, tau), j));
    float b = smul(c.brake, w4.y);
    if ((w0.z & 2u) != 0u) b = nf(b + smul(c.hand, w4.z));
    float wNew;
    float kappa = 0.0;
    float alpha = 0.0;
    float fnOut = 0.0;
    bool contact = hitK.w >= 0.0;
    float3 force = float3(0.0, 0.0, 0.0);
    if (contact)
    {
        float3 n = hitK.xyz;
        // The two-body form against the body's surface at the contact point.
        float3 cp = ssub3(sadd3(xh, sscale(c.up, w2.z)), sscale(c.up, hitK.w));
        float3 vb;
        float2 ms;
        bool onPartner = gMisc[wr + DRV_SCRATCH].z == DRV_PARTNER_HIT;
        if (onPartner) ms = drvPartnerSurface(c, k, m, vb);
        else ms = drvSurface(c, k, shape, n, cp, m, vb);
        float mHub = m;
        m = ms.x;
        vh = ssub3(vh, vb);
        float fn_ = loadK;
        b = nf(b + smul(smul(w4.w, fn_), r));
        float3 t = drvNrm(ssub3(fwd, sscale(n, sdot(n, fwd))));
        float3 sd = scross(n, t);
        uint g1 = c.nodeOff + w0.x;
        float3 fpre = drvForceNow(g1, c.beamOff, c.groupOff, c.groupCount, step);
        if (w0.x != w0.y) fpre = sadd3(fpre, drvForceNow(c.nodeOff + w0.y, c.beamOff, c.groupOff, c.groupCount, step));
        fpre = sscale(fpre, ms.y);
        float vt = sdot(vh, t);
        float vs = sdot(vh, sd);
        float ft = sdot(fpre, t);
        float fs = sdot(fpre, sd);
        kappa = sdiv(abs(nf(smul(st.x, r) - vt)), max(abs(vt), 1e-3));
        float vlen = ssqrt(nf(smul(vt, vt) + smul(vs, vs)));
        alpha = vlen < 1e-3 ? 0.0 : smul(sAtan2(abs(vs), abs(vt)), asfloat(0x42652ee1u));
        uint4 counts = gMisc[wr];
        uint nl = counts.w & 0xFFFFu;
        uint nt = counts.w >> 16;
        float muL = max(smul(drvCurve(wr + 7u, nl, max(kappa, w5.z)), w5.x), 0.0);
        muL = ssqrt(smul(muL, c.c5.w));
        float muT = max(smul(drvCurve(wr + 11u, nt, max(alpha, w5.w)), w5.y), 0.0);
        muT = ssqrt(smul(muT, c.c5.w));
        float al = smul(muL, fn_);
        float at = smul(muT, fn_);
        float ftRaw = -nf(fs + sdiv(smul(m, vs), h));
        float flRaw = drvSpin(m, vt, ft, j, r, wd, b, DRV_FLT_MAX).y;
        float alEff = al;
        float fT = ftRaw;
        if (at <= 1e-9)
        {
            fT = 0.0;
        }
        else
        {
            float e = 2.0;
            if (al > 1e-9)
            {
                float ql = sdiv(flRaw, al);
                float qt = sdiv(ftRaw, at);
                e = nf(smul(ql, ql) + smul(qt, qt));
            }
            if (e > 1.0)
            {
                float sc = sdiv(1.0, ssqrt(e));
                float tProp = smul(abs(ftRaw), sc);
                float lProp = smul(abs(flRaw), sc);
                float nx = sdiv(kappa, max(w5.z, 0.03));
                float tanA = asfloat(0x4103cf3au);
                if (vlen < 1e-3) tanA = 0.0;
                else if (abs(vt) > 0.0) tanA = min(sdiv(abs(vs), abs(vt)), asfloat(0x4103cf3au));
                float ny = sdiv(tanA, w6.x);
                float nn = ssqrt(nf(smul(nx, nx) + smul(ny, ny)));
                float ws = nn > 1e-6 ? drvClamp(nf(max(nx, ny) - 1.0), 0.0, 1.0) : 0.0;
                float tCap = drvLerp(tProp, nn > 1e-6 ? sdiv(smul(at, ny), nn) : tProp, ws);
                alEff = drvLerp(lProp, nn > 1e-6 ? sdiv(smul(al, nx), nn) : lProp, ws);
                fT = drvClamp(ftRaw, -tCap, tCap);
            }
            else
            {
                float qt = sdiv(ftRaw, at);
                alEff = smul(al, ssqrt(max(nf(1.0 - smul(qt, qt)), 0.0)));
            }
        }
        float2 sol = drvSpin(m, vt, ft, j, r, wd, b, alEff);
        wNew = sol.x;
        force = sadd3(sadd3(sscale(t, sol.y), sscale(sd, fT)), sscale(n, fn_));
        fnOut = fn_;
        m = mHub;
        // Equal and opposite on the body, at the contact point (its step sums); on
        // another vehicle's triangle staged for its nodes before the next step.
        float3 react = -force;
        if (onPartner)
        {
            DSet(drvPartnerAt(c.v, k) + 3u, float4(react, asfloat(1u)));
        }
        else
        {
        uint sat = cplShape(c.v, shape);
        float3 mo = scross(ssub3(cp, asfloat(gMisc[sat + 6u].xyz)), react);
        uint sacc = cplAcc(c.v) + cplB() + 6u * shape;
        gAcc[sacc].add(toFixed(react.x));
        gAcc[sacc + 1u].add(toFixed(react.y));
        gAcc[sacc + 2u].add(toFixed(react.z));
        gAcc[sacc + 3u].add(toFixed(mo.x));
        gAcc[sacc + 4u].add(toFixed(mo.y));
        gAcc[sacc + 5u].add(toFixed(mo.z));
        }
    }
    else
    {
        wNew = drvSpin(m, 0.0, 0.0, j, r, wd, b, 0.0).x;
    }
    float spin = nf(st.y + smul(wNew, h));
    if (spin > asfloat(0x40490fdbu)) spin = nf(spin - asfloat(0x40c90fdbu));
    else if (spin <= -asfloat(0x40490fdbu)) spin = nf(spin + asfloat(0x40c90fdbu));
    DSet(wr + 15u, float4(wNew, spin, angle, lenK));
    DSet(wr + 16u, float4(fnOut, kappa, alpha, contact ? 1.0 : 0.0));
    DSet(wr + 17u, float4(sadd3(c.ow, sadd3(xh, sscale(c.up, nf(w2.z - lenK)))), 0.0));
    DSet(wr + 18u, float4(scross(uw, fwd), 0.0));
    DSet(wr + 19u, float4(uw, 0.0));
    // 4. Onto the hub nodes, by mass (fixed point, read by the node phase).
    if (contact)
    {
        uint g1 = c.nodeOff + w0.x;
        if (w0.x == w0.y) accAdd(g1, force);
        else
        {
            uint g2 = c.nodeOff + w0.y;
            accAdd(g1, sscale(force, sdiv(gNodes[g1 * 3u].w, m)));
            accAdd(g2, sscale(force, sdiv(gNodes[g2 * 3u].w, m)));
        }
    }
}

// The vehicle's input and cut state (after its wheels).
void drvEnd(DrvCtx c)
{
    DSet(c.vr + 1u, float4(c.forward, c.brake, c.hand, c.steer));
    DSet(c.vr + 6u, float4(c.c6.x, c.cut, c.speed, asfloat(c.arCount)));
}

// Whether no two wheels share a hub node (then a wheel per lane gives the same bits).
bool drvWheelsApart(DrvCtx c)
{
    bool apart = true;
    for (uint a = 0u, aStop = EB_END(297, a, c.nw, 1u); a < aStop; a++)
    {
        uint4 wa = gMisc[drvWheel(c, a)];
        for (uint b = a + 1u, bStop = EB_END(298, b, c.nw, 1u); b < bStop; b++)
        {
            uint4 wb = gMisc[drvWheel(c, b)];
            if (wa.x == wb.x || wa.x == wb.y || wa.y == wb.x || wa.y == wb.y) apart = false;
        }
    }
    return apart;
}

// The frame's start (lane 0, before the barrier the uniform read follows).
void driveBegin()
{
    gsUniformDrive = gMisc[driveRec()].x;
}

// Wheel `w` of vehicle v (live: the lane has one) at step `step`, the three parts of
// a step, each followed by a barrier (every lane reaches them). Wheels apart: wheel w
// on lane w; else lane 0 runs every wheel, in explicit_drive::drive_pass's order (the
// rays, the bars, the tyres, the end: the barriers between change nothing on one
// lane). One call site per part (the compilers inline every call).
void driveWheelsSplit(uint v, uint w, bool live, uint step)
{
    DrvCtx c;
    bool has = live && w < DRV_MAX_WHEELS && drvCtx(v, c);
    c.step = step;
    bool apart = has && drvWheelsApart(c);
    uint k0 = apart ? w : 0u;
    uint k1 = !has ? 0u : (apart ? min(w + 1u, c.nw) : (w == 0u ? c.nw : 0u));
    for (uint k = k0, kStop = EB_END(299, k, k1, 1u); k < kStop; k++) drvRays(c, k);
    sync();
    if (has && w == 0u) drvBars(c);
    sync();
    for (uint k = k0, kStop = EB_END(300, k, k1, 1u); k < kStop; k++) drvTyre(c, k, step);
    if (has && w == 0u) drvEnd(c);
}

// The wheels ahead (perfFlags bit 2, the solo frame): one part of driveWheelsSplit for
// wheel lane `w` (0: the lane of the bars and the end), no barrier here. The rays and the
// bars read the nodes' positions and velocities (final from a step's node phase to the
// next), static shapes and the drive's records, so step s + 1's run beside step s's beam
// phase and after it; the tyres (which read their hubs' forces: the last beam phase's
// sums and the accumulators after selfApply) stay at the step's start, beside the
// coupling's first phase, which reads and writes none of it. The same calls on the same
// values as driveWheelsSplit.
// (The context once for a step's three parts where they run in one invocation: drvCtx
// reads the records drvEnd wrote and the nodes, neither of which changes between a
// step's rays and its tyres.)
void driveAheadCtx(uint v, uint w, out DrvCtx c, out bool has, out bool apart)
{
    has = w < DRV_MAX_WHEELS && drvCtx(v, c);
    apart = has && drvWheelsApart(c);
}
void driveAheadPart(DrvCtx c, bool has, bool apart, uint w, uint part, uint step)
{
    // The ahead schedules read the proxies before the step's integration: a host turns
    // them off while wheels see bodies (ExplicitBeamsGpu); a frame that still takes them
    // fails loudly, never with stale bodies.
    if (has && w == 0u && part == 0u && drvBodiesFlag(c)) ebLoopCapHit(301);
    uint k0 = apart ? w : 0u;
    uint k1 = !has ? 0u : (apart ? min(w + 1u, c.nw) : (w == 0u ? c.nw : 0u));
    if (part == 1u)
    {
        if (has && w == 0u) drvBars(c);
        return;
    }
    // (One call site for the rays and one for the tyres, as driveWheelsSplit.)
    for (uint k = k0, kStop = EB_END(302, k, k1, 1u); k < kStop; k++)
    {
        if (part == 0u) drvRays(c, k);
        else drvTyre(c, k, step);
    }
    if (part == 2u && has && w == 0u) drvEnd(c);
}

// --- The tyres split at their hubs' forces (perfFlags 262144, PERF_TYRE_AHEAD, with
// PERF_AHEAD_APART and PERF_WIDE_FUSED; explicit_plan::PERF_TYRE_AHEAD) ---
// drvTyre reads its hubs' forces (drvForceNow: the beam sums and the contact
// accumulators after the step's hits) and nothing else that the step's walk, pairs and
// hits change. drvTyrePre runs everything before that read and every value it does not
// feed (the frame, the hub, the axle, the steer angle, the tyre's frame, spin drive and
// brake, the slip ratio and angle, the friction curves, the friction ellipse's shape;
// the outputs 17..19; the end), on the coupling's and wheels' own workgroup after the
// bars (wideAheadApart); drvTyrePost, on the hits kernel's lanes after selfApply, reads
// the hubs' forces and runs the rest: the same functions on the same values in the same
// order (the values cross in float records, exactly), so the same bits as drvTyre. The
// records: DRV_PRE a wheel after the vehicle's wheels (DriveLayout::stride_for), cleared
// by drvTyrePost (a frame leaves them zero, as without the split).
static const uint DRV_PRE = 8u;
uint drvPreAt(uint vr, uint k)
{
    uint nwMax = (gMisc[driveRec()].z - DRV_VEH) / (DRV_WHEEL + DRV_PRE);
    return vr + DRV_VEH + DRV_WHEEL * nwMax + DRV_PRE * k;
}
void drvTyrePre(DrvCtx c, uint k, bool apart)
{
    if (drvBodiesFlag(c)) ebLoopCapHit(303);
    float h = STEP;
    float wb = c.c4.x;
    float track = c.c4.y;
    float ack = c.c3.w;
    uint wr = drvWheel(c, k);
    uint4 w0 = gMisc[wr];
    float4 w1 = DF(wr + 1u);
    float4 w2 = DF(wr + 2u);
    float4 w3 = DF(wr + 3u);
    float4 w4 = DF(wr + 4u);
    float4 w5 = DF(wr + 5u);
    float4 w6 = DF(wr + 6u);
    float4 st = DF(wr + 15u);
    float m;
    float3 xh, vh;
    drvHub(c, k, m, xh, vh);
    float3 a = drvAxle(c, k);
    float4 sc0 = DF(wr + DRV_SCRATCH);
    float4 hitK = DF(wr + DRV_SCRATCH + 1u);
    float lenK = sc0.x;
    float loadK = sc0.y;
    float angle = 0.0;
    if (w3.w > 0.0)
    {
        angle = smul(-c.steer, w3.w);
        if (ack > 0.0 && w6.y != 0.0 && abs(angle) > 1e-4 && wb > 0.0)
        {
            float2 sc = drvSinCos(abs(angle));
            float turnR = sdiv(wb, sdiv(sc.x, sc.y));
            bool inner = (angle > 0.0) == (w6.y > 0.0);
            float off = inner ? -0.5 : 0.5;
            float ak = sAtan2(wb, max(nf(turnR + smul(off, track)), 1e-3));
            angle = smul(drvSgn(angle), drvLerp(abs(angle), ak, ack));
        }
    }
    float3 uw = drvNrm(ssub3(c.up, sscale(a, sdot(a, c.up))));
    float3 f0 = scross(a, uw);
    float2 sca = drvSinCos(angle);
    float3 fwd = sadd3(sscale(f0, sca.y), sscale(scross(uw, f0), sca.x));
    float j = w1.z;
    float r = w1.x;
    float tau = 0.0;
    if ((w0.z & 1u) != 0u) tau = smul(smul(smul(smul(c.forward, c.c3.x), c.scale), w4.x), nf(1.0 - c.cut));
    if (c.top > 0.0 && smul(smul(drvSgn(c.forward), st.x), r) >= c.top) tau = 0.0;
    float wd = nf(smul(st.x, max(nf(1.0 - smul(w1.w, h)), 0.0)) + sdiv(smul(h, tau), j));
    float b = smul(c.brake, w4.y);
    if ((w0.z & 2u) != 0u) b = nf(b + smul(c.hand, w4.z));
    bool contact = hitK.w >= 0.0;
    uint p = drvPreAt(c.vr, k);
    if (contact)
    {
        float3 n = hitK.xyz;
        float fn_ = loadK;
        b = nf(b + smul(smul(w4.w, fn_), r));
        float3 t = drvNrm(ssub3(fwd, sscale(n, sdot(n, fwd))));
        float3 sd = scross(n, t);
        float vt = sdot(vh, t);
        float vs = sdot(vh, sd);
        float kappa = sdiv(abs(nf(smul(st.x, r) - vt)), max(abs(vt), 1e-3));
        float vlen = ssqrt(nf(smul(vt, vt) + smul(vs, vs)));
        float alpha = vlen < 1e-3 ? 0.0 : smul(sAtan2(abs(vs), abs(vt)), asfloat(0x42652ee1u));
        uint4 counts = gMisc[wr];
        uint nl = counts.w & 0xFFFFu;
        uint nt = counts.w >> 16;
        float muL = max(smul(drvCurve(wr + 7u, nl, max(kappa, w5.z)), w5.x), 0.0);
        muL = ssqrt(smul(muL, c.c5.w));
        float muT = max(smul(drvCurve(wr + 11u, nt, max(alpha, w5.w)), w5.y), 0.0);
        muT = ssqrt(smul(muT, c.c5.w));
        float al = smul(muL, fn_);
        float at = smul(muT, fn_);
        // (drvTyre's friction ellipse's shape: there only when the ellipse binds, here
        // always; nothing else reads it.)
        float nx = sdiv(kappa, max(w5.z, 0.03));
        float tanA = asfloat(0x4103cf3au);
        if (vlen < 1e-3) tanA = 0.0;
        else if (abs(vt) > 0.0) tanA = min(sdiv(abs(vs), abs(vt)), asfloat(0x4103cf3au));
        float ny = sdiv(tanA, w6.x);
        float nn = ssqrt(nf(smul(nx, nx) + smul(ny, ny)));
        float ws = nn > 1e-6 ? drvClamp(nf(max(nx, ny) - 1.0), 0.0, 1.0) : 0.0;
        DSet(p, float4(t, vt));
        DSet(p + 1u, float4(sd, vs));
        DSet(p + 2u, float4(n, fn_));
        DSet(p + 3u, float4(m, j, r, wd));
        DSet(p + 4u, float4(b, al, at, nx));
        DSet(p + 5u, float4(ny, nn, ws, 0.0));
        DSet(p + 6u, float4(st.y, angle, lenK, kappa));
        DSet(p + 7u, float4(alpha, 1.0, 0.0, apart ? 1.0 : 0.0));
    }
    else
    {
        float wNew = drvSpin(m, 0.0, 0.0, j, r, wd, b, 0.0).x;
        DSet(p + 6u, float4(st.y, angle, lenK, 0.0));
        DSet(p + 7u, float4(0.0, 0.0, wNew, apart ? 1.0 : 0.0));
    }
    DSet(wr + 17u, float4(sadd3(c.ow, sadd3(xh, sscale(c.up, nf(w2.z - lenK)))), 0.0));
    DSet(wr + 18u, float4(scross(uw, fwd), 0.0));
    DSet(wr + 19u, float4(uw, 0.0));
}
void drvTyrePost(uint v, uint vr, uint k, uint step)
{
    float h = STEP;
    uint wr = vr + DRV_VEH + k * DRV_WHEEL;
    uint p = drvPreAt(vr, k);
    float4 p6 = DF(p + 6u);
    float4 p7 = DF(p + 7u);
    bool contact = p7.y != 0.0;
    float wNew;
    float fnOut = 0.0;
    float m = 0.0;
    float3 force = float3(0.0, 0.0, 0.0);
    uint4 w0 = gMisc[wr];
    uint nodeOff = gStatic[v * 4u].x;
    if (contact)
    {
        float4 p0 = DF(p);
        float4 p1 = DF(p + 1u);
        float4 p2 = DF(p + 2u);
        float4 p3 = DF(p + 3u);
        float4 p4 = DF(p + 4u);
        float4 p5 = DF(p + 5u);
        float3 t = p0.xyz;
        float vt = p0.w;
        float3 sd = p1.xyz;
        float vs = p1.w;
        float3 n = p2.xyz;
        float fn_ = p2.w;
        m = p3.x;
        float j = p3.y;
        float r = p3.z;
        float wd = p3.w;
        float b = p4.x;
        float al = p4.y;
        float at = p4.z;
        float nx = p4.w;
        float ny = p5.x;
        float nn = p5.y;
        float ws = p5.z;
        uint beamOff = gStatic[v * 4u].z;
        uint4 r2 = gStatic[v * 4u + 2u];
        uint g1 = nodeOff + w0.x;
        float3 fpre = drvForceNow(g1, beamOff, r2.z, r2.w, step);
        if (w0.x != w0.y) fpre = sadd3(fpre, drvForceNow(nodeOff + w0.y, beamOff, r2.z, r2.w, step));
        float ft = sdot(fpre, t);
        float fs = sdot(fpre, sd);
        float ftRaw = -nf(fs + sdiv(smul(m, vs), h));
        float flRaw = drvSpin(m, vt, ft, j, r, wd, b, DRV_FLT_MAX).y;
        float alEff = al;
        float fT = ftRaw;
        if (at <= 1e-9)
        {
            fT = 0.0;
        }
        else
        {
            float e = 2.0;
            if (al > 1e-9)
            {
                float ql = sdiv(flRaw, al);
                float qt = sdiv(ftRaw, at);
                e = nf(smul(ql, ql) + smul(qt, qt));
            }
            if (e > 1.0)
            {
                float sc = sdiv(1.0, ssqrt(e));
                float tProp = smul(abs(ftRaw), sc);
                float lProp = smul(abs(flRaw), sc);
                float tCap = drvLerp(tProp, nn > 1e-6 ? sdiv(smul(at, ny), nn) : tProp, ws);
                alEff = drvLerp(lProp, nn > 1e-6 ? sdiv(smul(al, nx), nn) : lProp, ws);
                fT = drvClamp(ftRaw, -tCap, tCap);
            }
            else
            {
                float qt = sdiv(ftRaw, at);
                alEff = smul(al, ssqrt(max(nf(1.0 - smul(qt, qt)), 0.0)));
            }
        }
        float2 sol = drvSpin(m, vt, ft, j, r, wd, b, alEff);
        wNew = sol.x;
        force = sadd3(sadd3(sscale(t, sol.y), sscale(sd, fT)), sscale(n, fn_));
        fnOut = fn_;
    }
    else
    {
        wNew = p7.z;
    }
    float spin = nf(p6.x + smul(wNew, h));
    if (spin > asfloat(0x40490fdbu)) spin = nf(spin - asfloat(0x40c90fdbu));
    else if (spin <= -asfloat(0x40490fdbu)) spin = nf(spin + asfloat(0x40c90fdbu));
    DSet(wr + 15u, float4(wNew, spin, p6.y, p6.z));
    DSet(wr + 16u, float4(fnOut, p6.w, p7.x, contact ? 1.0 : 0.0));
    for (uint i = 0u; i < DRV_PRE; i++) gMisc[p + i] = uint4(0u, 0u, 0u, 0u);
    if (contact)
    {
        uint g1 = nodeOff + w0.x;
        if (w0.x == w0.y) accAdd(g1, force);
        else
        {
            uint g2 = nodeOff + w0.y;
            accAdd(g1, sscale(force, sdiv(gNodes[g1 * 3u].w, m)));
            accAdd(g2, sscale(force, sdiv(gNodes[g2 * 3u].w, m)));
        }
    }
}
// The tyres' first part (wheel lane w; drvTyrePre), then the end (lane 0), as
// driveAheadPart's part 2 takes the wheels.
void driveTyresPre(DrvCtx c, bool has, bool apart, uint w)
{
    uint k0 = apart ? w : 0u;
    uint k1 = !has ? 0u : (apart ? min(w + 1u, c.nw) : (w == 0u ? c.nw : 0u));
    for (uint k = k0, kStop = EB_END(304, k, k1, 1u); k < kStop; k++) drvTyrePre(c, k, apart);
    if (has && w == 0u) drvEnd(c);
}
// The tyres' second part (wheel lane w; drvTyrePost) for vehicle v's wheels as
// driveTyresPre took them: the wheels-apart decision from wheel w's record (wheel w is
// this lane's when apart; otherwise lane 0 takes every wheel, and any other lane reads
// 0, whether or not lane 0 has cleared the record).
void driveTyresPost(uint v, uint w, uint step)
{
    if (w >= DRV_MAX_WHEELS) return;
    uint vr = driveBase(v);
    uint nw = min(gMisc[vr + 2u].x, DRV_MAX_WHEELS);
    if (w >= nw) return;
    bool apart = DF(drvPreAt(vr, w) + 7u).w != 0.0;
    uint k0 = apart ? w : 0u;
    uint k1 = apart ? w + 1u : (w == 0u ? nw : 0u);
    for (uint k = k0, kStop = EB_END(305, k, k1, 1u); k < kStop; k++) drvTyrePost(v, vr, k, step);
}

// --- perfFlags bit 2097152 (explicit_plan::PERF_TYRE_EARLY, with PERF_TYRE_AHEAD): a
// wheel's second part loaded before the step's hits ---
// drvTyrePost reads, besides its hubs' accumulators (accForce), only what nothing in the
// hits kernel after its first barrier writes: its first part's records (the list
// dispatch's), the wheel's record, the hub nodes' masses and beam sums (the beam phase's
// incidence entries and group state; resolvePending ran before that barrier, and selfApply
// writes neither). With the wheels apart (wheel w on lane 255 - w, its own records), the
// lane loads all of that while the other lanes apply the hits (drvTyreLoad), and after
// them reads the accumulators and runs the rest (drvTyreFinish): drvTyrePost's arithmetic
// on the same bits in the same order (drvForceNow's `beam sum + accForce`, as written
// there). Not apart: drvTyrePost as before.
struct DrvPostIn
{
    bool ok;
    uint p;
    uint wr;
    uint4 w0;
    uint nodeOff;
    float4 p0;
    float4 p1;
    float4 p2;
    float4 p3;
    float4 p4;
    float4 p5;
    float4 p6;
    float4 p7;
    float3 b1;
    float3 b2;
    float m1;
    float m2;
};
DrvPostIn drvTyreLoad(uint v, uint w, uint step)
{
    DrvPostIn d = (DrvPostIn)0;
    d.ok = false;
    if (w >= DRV_MAX_WHEELS) return d;
    uint vr = driveBase(v);
    uint nw = min(gMisc[vr + 2u].x, DRV_MAX_WHEELS);
    if (w >= nw) return d;
    d.p = drvPreAt(vr, w);
    d.p7 = DF(d.p + 7u);
    if (d.p7.w == 0.0) return d;
    d.ok = true;
    d.wr = vr + DRV_VEH + w * DRV_WHEEL;
    d.p6 = DF(d.p + 6u);
    d.w0 = gMisc[d.wr];
    d.nodeOff = gStatic[v * 4u].x;
    if (d.p7.y == 0.0) return d;
    d.p0 = DF(d.p);
    d.p1 = DF(d.p + 1u);
    d.p2 = DF(d.p + 2u);
    d.p3 = DF(d.p + 3u);
    d.p4 = DF(d.p + 4u);
    d.p5 = DF(d.p + 5u);
    uint beamOff = gStatic[v * 4u].z;
    uint4 r2 = gStatic[v * 4u + 2u];
    uint g1 = d.nodeOff + d.w0.x;
    d.b1 = beamSum(gNodes[g1 * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + g1], beamOff, r2.z, r2.w, (step + 1u) & 1u, false);
    d.m1 = gNodes[g1 * 3u].w;
    if (d.w0.x != d.w0.y)
    {
        uint g2 = d.nodeOff + d.w0.y;
        d.b2 = beamSum(gNodes[g2 * 3u + 2u].xyz, gStatic[gP.nodeInfoBase + g2], beamOff, r2.z, r2.w, (step + 1u) & 1u,
                       false);
        d.m2 = gNodes[g2 * 3u].w;
    }
    return d;
}
float3 drvForceFrom(float3 f, uint gi)
{
    return f + accForce(gi);
}
void drvTyreFinish(DrvPostIn d)
{
    float h = STEP;
    uint p = d.p;
    uint wr = d.wr;
    float4 p6 = d.p6;
    float4 p7 = d.p7;
    bool contact = p7.y != 0.0;
    float wNew;
    float fnOut = 0.0;
    float m = 0.0;
    float3 force = float3(0.0, 0.0, 0.0);
    uint4 w0 = d.w0;
    uint nodeOff = d.nodeOff;
    if (contact)
    {
        float3 t = d.p0.xyz;
        float vt = d.p0.w;
        float3 sd = d.p1.xyz;
        float vs = d.p1.w;
        float3 n = d.p2.xyz;
        float fn_ = d.p2.w;
        m = d.p3.x;
        float j = d.p3.y;
        float r = d.p3.z;
        float wd = d.p3.w;
        float b = d.p4.x;
        float al = d.p4.y;
        float at = d.p4.z;
        float nx = d.p4.w;
        float ny = d.p5.x;
        float nn = d.p5.y;
        float ws = d.p5.z;
        uint g1 = nodeOff + w0.x;
        float3 fpre = drvForceFrom(d.b1, g1);
        if (w0.x != w0.y) fpre = sadd3(fpre, drvForceFrom(d.b2, nodeOff + w0.y));
        float ft = sdot(fpre, t);
        float fs = sdot(fpre, sd);
        float ftRaw = -nf(fs + sdiv(smul(m, vs), h));
        float flRaw = drvSpin(m, vt, ft, j, r, wd, b, DRV_FLT_MAX).y;
        float alEff = al;
        float fT = ftRaw;
        if (at <= 1e-9)
        {
            fT = 0.0;
        }
        else
        {
            float e = 2.0;
            if (al > 1e-9)
            {
                float ql = sdiv(flRaw, al);
                float qt = sdiv(ftRaw, at);
                e = nf(smul(ql, ql) + smul(qt, qt));
            }
            if (e > 1.0)
            {
                float sc = sdiv(1.0, ssqrt(e));
                float tProp = smul(abs(ftRaw), sc);
                float lProp = smul(abs(flRaw), sc);
                float tCap = drvLerp(tProp, nn > 1e-6 ? sdiv(smul(at, ny), nn) : tProp, ws);
                alEff = drvLerp(lProp, nn > 1e-6 ? sdiv(smul(al, nx), nn) : lProp, ws);
                fT = drvClamp(ftRaw, -tCap, tCap);
            }
            else
            {
                float qt = sdiv(ftRaw, at);
                alEff = smul(al, ssqrt(max(nf(1.0 - smul(qt, qt)), 0.0)));
            }
        }
        float2 sol = drvSpin(m, vt, ft, j, r, wd, b, alEff);
        wNew = sol.x;
        force = sadd3(sadd3(sscale(t, sol.y), sscale(sd, fT)), sscale(n, fn_));
        fnOut = fn_;
    }
    else
    {
        wNew = p7.z;
    }
    float spin = nf(p6.x + smul(wNew, h));
    if (spin > asfloat(0x40490fdbu)) spin = nf(spin - asfloat(0x40c90fdbu));
    else if (spin <= -asfloat(0x40490fdbu)) spin = nf(spin + asfloat(0x40c90fdbu));
    DSet(wr + 15u, float4(wNew, spin, p6.y, p6.z));
    DSet(wr + 16u, float4(fnOut, p6.w, p7.x, contact ? 1.0 : 0.0));
    for (uint i = 0u; i < DRV_PRE; i++) gMisc[p + i] = uint4(0u, 0u, 0u, 0u);
    if (contact)
    {
        uint g1 = nodeOff + w0.x;
        if (w0.x == w0.y) accAdd(g1, force);
        else
        {
            uint g2 = nodeOff + w0.y;
            accAdd(g1, sscale(force, sdiv(d.m1, m)));
            accAdd(g2, sscale(force, sdiv(d.m2, m)));
        }
    }
}

// A contact group's wheels on each other (members `off`, k of them in vehicle order) at
// substep s's start, where every member's state is final (explicit_contact.rs
// ExplicitBeamsWorld::wheels_begin and wheels_end): the last step's staged reactions onto
// the partners' nodes' accumulators (read by the coming node phases), then (`rays`) every
// member's wheels' rays against the other members' live collision triangles, the nearest
// hit per wheel kept in its partner records for step gMisc[v].w + s. Every lane; barriers
// in uniform control flow (k uniform). `rays` false: the reactions only (after a frame's
// last step).
// The rays in rounds of WP_ROUND members (their wheels the round's slots): a slot's ray
// against the members whose allowed box (the frame's bound on its nodes, vvRec
// VV_ALLOWED_*: a node outside it is reported, vvReportOutside) its segment's box meets;
// the rest cannot hold a hit (lossless). Those (slot, member) pairs are listed with the
// prefix of their triangles, then every (pair, triangle) runs across the lanes: the least
// t a slot, then the first triangle at it (its global index: the CPU's order). Workgroup
// words in gsCount's first ones (free between a step's phases): the minima, the pairs'
// count and list.
static const uint WP_ROUND = 2u;
static const uint WP_SLOTS = WP_ROUND * DRV_MAX_WHEELS;
static const uint WP_COUNT = 2u * WP_SLOTS;
static const uint WP_LIST = WP_COUNT + 1u;
// Pairs a round may list (gsCount's words past the minima and the count); more fails
// the frame loudly (never a pair dropped). A round lists at most 32 slots × the members
// whose box a ray's meets.
static const uint WP_PAIRS = GRID_BUCKETS + REUSE_WORDS - WP_LIST - 1u;
// A listed pair: its triangles' prefix << 13 | slot << 7 | member.

// The ray of round slot q (member r0 + q / 16's wheel q % 16): its partner records `pa`,
// start and reach, direction. False: no such wheel, or it does not see bodies.
bool wpRay(uint off, uint k, uint r0, uint q, out uint pa, out float4 ray, out float3 d)
{
    pa = 0u;
    ray = float4(0.0, 0.0, 0.0, -1.0);
    d = float3(0.0, 0.0, 0.0);
    uint m = r0 + q / DRV_MAX_WHEELS;
    uint w = q % DRV_MAX_WHEELS;
    if (m >= k) return false;
    uint va = vvMember(off, m);
    if ((gMisc[driveBase(va) + 5u].y & 8u) == 0u || w >= min(gMisc[driveBase(va) + 2u].x, DRV_MAX_WHEELS)) return false;
    pa = drvPartnerAt(va, w);
    ray = DF(pa);
    d = DF(pa + 4u).xyz;
    return true;
}

void wheelsPartners(uint off, uint k, uint lane, uint s, bool rays)
{
    for (uint i = lane, iStop = EB_END(306, i, k * DRV_MAX_WHEELS, LANES); i < iStop; i += LANES)
    {
        uint v = vvMember(off, i / DRV_MAX_WHEELS);
        uint w = i % DRV_MAX_WHEELS;
        if (w >= min(gMisc[driveBase(v) + 2u].x, DRV_MAX_WHEELS)) continue;
        uint pa = drvPartnerAt(v, w);
        uint4 r = gMisc[pa + 3u];
        if (r.w == 0u) continue;
        float3 f = asfloat(r.xyz);
        float3 wt = DF(pa + 1u).xyz;
        uint4 tn = gStatic[gP.triBase + gMisc[pa + 1u].w];
        if (wt.x > 0.0) accAdd(tn.x, sscale(f, wt.x));
        if (wt.y > 0.0) accAdd(tn.y, sscale(f, wt.y));
        if (wt.z > 0.0) accAdd(tn.z, sscale(f, wt.z));
        gMisc[pa + 3u] = uint4(0u, 0u, 0u, 0u);
    }
    if (!rays)
    {
        sync();
        return;
    }
    // Every member's rays (World): start and reach in record 0, direction in record 4.
    for (uint i = lane, iStop = EB_END(307, i, k * DRV_MAX_WHEELS, LANES); i < iStop; i += LANES)
    {
        uint va = vvMember(off, i / DRV_MAX_WHEELS);
        uint w = i % DRV_MAX_WHEELS;
        if ((gMisc[driveBase(va) + 5u].y & 8u) == 0u) continue;
        DrvCtx c;
        if (!drvCtx(va, c) || w >= c.nw) continue;
        uint wr = drvWheel(c, w);
        float4 w1 = DF(wr + 1u);
        float4 w2 = DF(wr + 2u);
        float mh;
        float3 xh, vh;
        drvHub(c, w, mh, xh, vh);
        float3 hp = sadd3(xh, sscale(c.up, w2.z));
        uint pa = drvPartnerAt(va, w);
        DSet(pa, float4(sadd3(c.ow, hp), nf(w2.y + w1.x)));
        DSet(pa + 4u, float4(-c.up, 0.0));
    }
    sync();
    for (uint r0 = 0u, r0Stop = EB_END_UNIFORM(308, r0, k, WP_ROUND, EB_MAX_MEMBERS); r0 < r0Stop; r0 += WP_ROUND)
    {
        if (lane < WP_COUNT) gsCount[lane].store(0xFFFFFFFFu);
        if (lane == 0u) gsCount[WP_COUNT].store(0u);
        sync();
        // The pairs: (slot, member) whose boxes meet.
        for (uint i = lane, iStop = EB_END(309, i, WP_SLOTS * k, LANES); i < iStop; i += LANES)
        {
            uint q = i / k;
            uint j = i % k;
            uint pa;
            float4 ray;
            float3 d;
            if (!wpRay(off, k, r0, q, pa, ray, d) || r0 + q / DRV_MAX_WHEELS == j) continue;
            uint vb = vvMember(off, j);
            float3 e = sadd3(ray.xyz, sscale(d, ray.w));
            float3 lo = min(ray.xyz, e);
            float3 hi = max(ray.xyz, e);
            if (any(hi < asfloat(gMisc[vvRec(vb, VV_ALLOWED_LO)].xyz)) || any(lo > asfloat(gMisc[vvRec(vb, VV_ALLOWED_HI)].xyz))) continue;
            uint n = gsCount[WP_COUNT].add(1u);
            if (n < WP_PAIRS) gsCount[WP_LIST + n].store(q << 7 | j);
            else ebLoopCapHit(310);
        }
        sync();
        // Their triangles' prefix (lane 0, in list order: the sums matter, not the order).
        uint pairs = min(gsCount[WP_COUNT].load(), WP_PAIRS);
        if (lane == 0u)
        {
            uint run = 0u;
            for (uint p = 0u, pStop = EB_END(311, p, pairs, 1u); p < pStop; p++)
            {
                uint e = gsCount[WP_LIST + p].load();
                gsCount[WP_LIST + p].store(run << 13 | (e & 0x1FFFu));
                run += gStatic[vvMember(off, e & 0x7Fu) * 4u + 1u].y;
            }
            gsCount[WP_LIST + pairs].store(run << 13);
        }
        sync();
        uint total = gsCount[WP_LIST + pairs].load() >> 13;
        // The least t of each slot, then the first triangle at it.
        for (uint pass = 0u; pass < 2u; pass++)
        {
            for (uint i = lane, iStop = EB_END(312, i, total, LANES); i < iStop; i += LANES)
            {
                // The pair holding item i: the last whose prefix is at most i.
                uint lo = 0u;
                uint hi = pairs;
                for (uint g = 0u; g < 16u; g++)
                {
                    if (hi - lo <= 1u) break;
                    uint mid = (lo + hi) / 2u;
                    if ((gsCount[WP_LIST + mid].load() >> 13) <= i) lo = mid;
                    else hi = mid;
                }
                uint e = gsCount[WP_LIST + lo].load();
                uint q = (e >> 7) & 0x3Fu;
                uint vb = vvMember(off, e & 0x7Fu);
                uint pa;
                float4 ray;
                float3 d;
                wpRay(off, k, r0, q, pa, ray, d);
                uint gt = gStatic[vb * 4u + 1u].x + (i - (e >> 13));
                uint z = gMisc[gP.vehicles + gt].z;
                if ((z & TRI_ANTI_ONLY) != 0u || ((z & TRI_BREAKABLE) != 0u && triangleDead(gt))) continue;
                float3 ob = asfloat(gMisc[vb].xyz);
                uint4 tn = gStatic[gP.triBase + gt];
                float3 wt;
                float4 h = drvTriangle(ray.xyz, d, sadd3(ob, gNodes[tn.x * 3u].xyz), sadd3(ob, gNodes[tn.y * 3u].xyz),
                                       sadd3(ob, gNodes[tn.z * 3u].xyz), wt);
                if (h.w < 0.0 || h.w > ray.w) continue;
                uint key = asuint(h.w == 0.0 ? 0.0 : h.w);
                if (pass == 0u) gsCount[q].min(key);
                else if (key == gsCount[q].load()) gsCount[WP_SLOTS + q].min(gt);
            }
            sync();
        }
        // The winners' records.
        if (lane < WP_SLOTS)
        {
            uint pa;
            float4 ray;
            float3 d;
            if (wpRay(off, k, r0, lane, pa, ray, d))
            {
                uint va = vvMember(off, r0 + lane / DRV_MAX_WHEELS);
                uint step1 = gMisc[va].w + s + 1u;
                uint gt = gsCount[WP_SLOTS + lane].load();
                if (gt == 0xFFFFFFFFu)
                {
                    DSet(pa, drvNone());
                    gMisc[pa + 2u] = uint4(step1, 0xFFFFFFFFu, 0u, 0u);
                }
                else
                {
                    uint4 tn = gStatic[gP.triBase + gt];
                    float3 ob = asfloat(gMisc[tn.w].xyz);
                    float3 wt;
                    float4 h = drvTriangle(ray.xyz, d, sadd3(ob, gNodes[tn.x * 3u].xyz), sadd3(ob, gNodes[tn.y * 3u].xyz),
                                           sadd3(ob, gNodes[tn.z * 3u].xyz), wt);
                    // The point's velocity and inverse mass (explicit_contact::two_body_inverse_mass).
                    float3 v0 = gNodes[tn.x * 3u + 1u].xyz;
                    float3 v1 = gNodes[tn.y * 3u + 1u].xyz;
                    float3 v2 = gNodes[tn.z * 3u + 1u].xyz;
                    float3 vel = sadd3(sadd3(sscale(v0, wt.x), sscale(v1, wt.y)), sscale(v2, wt.z));
                    float3 im;
                    im.x = (gStatic[gP.nodeInfoBase + tn.x].x & NODE_IMMOVABLE) != 0u ? 0.0 : sdiv(1.0, gNodes[tn.x * 3u].w);
                    im.y = (gStatic[gP.nodeInfoBase + tn.y].x & NODE_IMMOVABLE) != 0u ? 0.0 : sdiv(1.0, gNodes[tn.y * 3u].w);
                    im.z = (gStatic[gP.nodeInfoBase + tn.z].x & NODE_IMMOVABLE) != 0u ? 0.0 : sdiv(1.0, gNodes[tn.z * 3u].w);
                    float t0 = wt.x > 0.0 ? smul(smul(wt.x, wt.x), im.x) : 0.0;
                    float t1 = wt.y > 0.0 ? smul(smul(wt.y, wt.y), im.y) : 0.0;
                    float t2 = wt.z > 0.0 ? smul(smul(wt.z, wt.z), im.z) : 0.0;
                    DSet(pa, h);
                    gMisc[pa + 1u] = uint4(asuint(wt), gt);
                    gMisc[pa + 2u] = uint4(step1, tn.w, 0u, 0u);
                    DSet(pa + 4u, float4(vel, nf(nf(t0 + t1) + t2)));
                }
            }
        }
        sync();
    }
}

// The drive's part of step s of vehicle v (uniform `on`): its wheels, then a barrier.
void driveStep(uint v, uint lane, uint s, uint on)
{
    if (on == 0u) return;
    driveWheelsSplit(v, lane, true, gMisc[v].w + s);
    if (lane == 0u) gsHit = 1u;
    sync();
}

// driveStep for a contact group's k members (their node phases read the
// accumulators every step), DRV_MAX_WHEELS lanes a member.
void driveStepMembers(uint off, uint k, uint lane, uint s, uint on)
{
    if (on == 0u) return;
    uint per = LANES / DRV_MAX_WHEELS;
    for (uint p = 0u, pStop = EB_END_UNIFORM(313, p, k, per, EB_MAX_MEMBERS); p < pStop; p += per)
    {
        uint m = p + lane / DRV_MAX_WHEELS;
        bool live = m < k;
        uint v = live ? vvMember(off, m) : 0u;
        driveWheelsSplit(v, lane % DRV_MAX_WHEELS, live, live ? gMisc[v].w + s : 0u);
    }
    sync();
}
