// Soft-body kernels: 3-DOF particles joined by plastic, breakable beams
// (Rigs of Rods / BeamNG node–beam vehicles) solved inside the AVBD step.
//
// Particles are BodyState records flagged BODY_PARTICLE (zero inverse inertia,
// identity orientation). They keep every rigid pipeline stage (prediction,
// broadphase, node-sphere contacts, CSR, velocity extraction, sleep) but are
// left out of the rigid colour lists: each iteration sweeps the rigid colours,
// then `soft_primal` sweeps the particle colours computed per vehicle at spawn
// (Newton-style alternating sweeps). A particle's block reuses primalAccumulate
// for its inertia, contacts and joints (rotation frozen, so only the linear 3×3
// block is solved) and adds its beams from the soft arena.
//
// Beam energy ½k(ℓ − L)² with an axial dashpot (c/h)(ℓ − ℓ⁻); one-sided supports
// and ropes; bounded (shock) beams with stiff limit springs; hydros whose rest
// length follows the steering input. After the iterations, `soft_plastic`
// applies the Rigs of Rods plasticity and breaking rules at the converged
// state, once per beam (no colouring needed), and `soft_break_groups` breaks
// every beam of a triggered detacher group.
//
// Step placement (engine_exec::rigid_plan):
//   vehicle_input → soft_wake → … → predict_bodies → vehicle_prepare → soft_begin
//   → per iteration { rigid colours → soft_primal × particle colours → soft_chebyshev
//                     → solve_dual → soft_dual → vehicle_iterate }
//   → soft_plastic → soft_break_groups → extract_velocity_sleep → vehicle_post
//   → soft_sleep → sleep gate → soft_sleep → sleep commit
//
// Arena layout (gSoftData, float4 slots): engine-softbody/src/layout.rs.

#include "world_step.slang"

BIND(32) RWStructuredBuffer<float4> gSoftData;


static const uint BEAM_NORMAL = 0u;
static const uint BEAM_SUPPORT = 1u;
static const uint BEAM_ROPE = 2u;
static const uint BEAM_BOUNDED = 3u;
static const uint BEAM_HYDRO = 4u;

static const uint BEAM_BROKEN = 16u;
static const uint BEAM_HARD = 32u;
static const uint BEAM_YIELDED = 128u;

static const uint SOFT_PLASTICITY = 1u;
static const uint SOFT_BREAKING = 2u;
static const uint SOFT_SLEEP_SYNC = 4u;
static const uint SOFT_CONTACTS = 8u;
static const uint ENTRY_CONTACTER = 65536u;
// Node contacts a triangle records (the triangle side of node–triangle contacts).
static const uint TRI_NODE_CONTACTS = 4u;
static const uint TRI_CONTACT_STRIDE = 3u + 3u * TRI_NODE_CONTACTS;
static const uint NODE_SLOT_STRIDE = 3u;
// "No index" marker stored in float4 slots: FLT_MAX bits (an all-ones NaN pattern
// cannot appear as a WGSL constant).
static const uint SOFT_NONE = 0x7F7FFFFFu;

static const uint EVENT_BEAM_BREAK = 5u;
static const uint SOFT_NO_BODY = 0x7F7FFFFFu;

float4 SD(uint q) { return gSoftData[q]; }
uint4 SU(uint q) { return asuint(gSoftData[q]); }

uint softBodyCount() { return SU(0u).x; }
uint softBeamCount() { return SU(0u).y; }
uint softColorCount() { return SU(0u).z; }
uint softListCount() { return SU(0u).w; }
uint softBodyBase() { return SU(1u).x; }
uint softBeamBase() { return SU(1u).y; }
uint softListBase() { return SU(1u).z; }
uint softAdjBase() { return SU(1u).w; }
uint softColorBase() { return SU(2u).x; }
uint softGroupBase() { return SU(2u).y; }
uint softGroupCount() { return SU(2u).z; }
uint softFlags() { return SU(2u).w; }
float softDampScale() { return SD(3u).x; }
float softBeta() { return SD(3u).y; }
float softMinLength() { return SD(3u).z; }
float softThresholdScale() { return SD(4u).x; }
float softChebyshevRho() { return SD(4u).z; }
float softContactStiffness() { return SD(4u).w; }
uint softChebBase() { return SU(5u).x; }
uint softTriBase() { return SU(5u).y; }
uint softTriCount() { return SU(5u).z; }
uint softTriContactBase() { return SU(5u).w; }
uint softNodeSlotBase() { return SU(6u).x; }
uint softFlagBase() { return SU(6u).y; }
uint softObstacleBase() { return SU(6u).z; }
uint softObstacleCount() { return SU(6u).w; }
uint softAabbBase() { return SU(7u).x; }
uint softParticleTotal() { return SU(7u).y; }
float softContactFriction() { return SD(7u).z; }
uint softTriRestBase() { return SU(7u).w; }
// A skin triangle stretched past this × its rest edges is torn (the parts it
// joined have separated): it stops colliding, as renderers stop drawing it.
static const float TRI_TEAR = 2.5;
bool triTorn(uint t, float3 a, float3 b, float3 c)
{
    float3 r = SD(softTriRestBase() + t).xyz;
    return length(a - b) > TRI_TEAR * r.x || length(b - c) > TRI_TEAR * r.y || length(c - a) > TRI_TEAR * r.z;
}
bool softContactsOn() { return (softFlags() & SOFT_CONTACTS) != 0u; }

struct SoftBeam
{
    uint a;
    uint b;
    float rest;
    float rest0;
    float spring;
    float damp;
    float deformC;
    float strength;
    float coef;
    float shortB;
    float longB;
    uint meta;
    float penalty;
    float lambda;
    float deformT;
    float force;
};

uint beamKind(SoftBeam e) { return e.meta & 15u; }
bool beamBroken(SoftBeam e) { return (e.meta & BEAM_BROKEN) != 0u; }
bool beamHard(SoftBeam e) { return (e.meta & BEAM_HARD) != 0u; }
uint beamSoftBody(SoftBeam e) { return (e.meta >> 8) & 255u; }
uint beamGroup(SoftBeam e) { return e.meta >> 16; }

SoftBeam loadBeam(uint e)
{
    uint q = softBeamBase() + 4u * e;
    uint4 s0 = SU(q);
    float4 s1 = SD(q + 1u);
    float4 s2 = SD(q + 2u);
    float4 s3 = SD(q + 3u);
    SoftBeam r;
    r.a = s0.x; r.b = s0.y; r.rest = asfloat(s0.z); r.rest0 = asfloat(s0.w);
    r.spring = s1.x; r.damp = s1.y; r.deformC = s1.z; r.strength = s1.w;
    r.coef = s2.x; r.shortB = s2.y; r.longB = s2.z; r.meta = asuint(s2.w);
    r.penalty = s3.x; r.lambda = s3.y; r.deformT = s3.z; r.force = s3.w;
    return r;
}

void storeBeamState(uint e, SoftBeam r)
{
    uint q = softBeamBase() + 4u * e;
    gSoftData[q] = float4(asfloat(r.a), asfloat(r.b), r.rest, r.rest0);
    gSoftData[q + 1u] = float4(r.spring, r.damp, r.deformC, r.strength);
    gSoftData[q + 2u] = float4(r.coef, r.shortB, r.longB, asfloat(r.meta));
    gSoftData[q + 3u] = float4(r.penalty, r.lambda, r.deformT, r.force);
}

// φ'(ℓ), φ''(ℓ) of the elastic energy with stiffness k (engine-softbody model.rs).
float2 beamElastic(SoftBeam e, float len, float k)
{
    float delta = len - e.rest;
    uint kind = beamKind(e);
    if (kind == BEAM_SUPPORT && delta > 0.0) return float2(0.0, 0.0);
    if (kind == BEAM_ROPE && delta < 0.0) return float2(0.0, 0.0);
    if (kind == BEAM_BOUNDED)
    {
        float2 r = float2(k * delta, k);
        float lo = e.rest * (1.0 - e.shortB);
        float hi = e.rest * (1.0 + e.longB);
        float kl = e.deformC; // limit spring
        if (len < lo) r += float2(kl * (len - lo), kl);
        else if (len > hi) r += float2(kl * (len - hi), kl);
        return r;
    }
    if (beamHard(e)) return float2(k * delta + e.lambda, k);
    return float2(k * delta, k);
}

float beamDampCoef(SoftBeam e, float len)
{
    float delta = len - e.rest;
    uint kind = beamKind(e);
    bool slack = (kind == BEAM_SUPPORT && delta > 0.0) || (kind == BEAM_ROPE && delta < 0.0);
    return e.damp * softDampScale() * (slack ? 0.1 : 1.0);
}

// Signed axial force, + = compression (Rigs of Rods `slen`), material stiffness.
float beamReadout(SoftBeam e, float len, float len0, float h)
{
    float spring = beamHard(e) ? (e.penalty * (len - e.rest) + e.lambda) : beamElastic(e, len, e.spring).x;
    return -(spring + beamDampCoef(e, len) * (len - len0) / h);
}

// ---------------------------------------------------------------------------
// Soft body records: [firstBody, particles, firstBeam, beams],
// [steer, group, core, ordinal base], [firstTri, tris, δ, 0]
// ---------------------------------------------------------------------------
uint4 softBodyRec(uint s) { return SU(softBodyBase() + 3u * s); }
float4 softBodyRec1(uint s) { return SD(softBodyBase() + 3u * s + 1u); }
uint4 softBodyRec2(uint s) { return SU(softBodyBase() + 3u * s + 2u); }

// Ordinal (index across every soft particle) of a particle body.
uint particleOrdinal(uint s, uint body) { return asuint(softBodyRec1(s).w) + body - softBodyRec(s).x; }

void wakeBody(uint i)
{
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b) || !bodyIsSleeping(b)) return;
    gBodies[i].flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
    gBodies[i].sleepTimer = 0.0;
}

// A soft body sleeps and wakes as a whole: when any of its particles (or its
// rigid core) is awake, wake every sleeping one before the step builds its lists.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_wake(uint3 tid : SV_DispatchThreadID)
{
    uint s = dispatchIndex(tid);
    if (s >= softBodyCount() || (softFlags() & SOFT_SLEEP_SYNC) == 0u) return;
    uint4 r = softBodyRec(s);
    uint core = asuint(softBodyRec1(s).z);
    bool anyAwake = false, anyAsleep = false;
    for (uint k = 0u; k < r.y; k++)
    {
        BodyState b = gBodies[r.x + k];
        if (!bodyIsDynamic(b)) continue;
        if (bodyIsSleeping(b)) anyAsleep = true; else anyAwake = true;
    }
    if (core != SOFT_NO_BODY)
    {
        BodyState c = gBodies[core];
        if (bodyIsDynamic(c)) { if (bodyIsSleeping(c)) anyAsleep = true; else anyAwake = true; }
    }
    if (!(anyAwake && anyAsleep)) return;
    for (uint k = 0u; k < r.y; k++) wakeBody(r.x + k);
    if (core != SOFT_NO_BODY) wakeBody(core);
}

// Sleep candidates (gCcdFlags ≠ 0 after extract_velocity_sleep and the island
// gate) only stand when every particle and the core of the soft body agree.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_sleep(uint3 tid : SV_DispatchThreadID)
{
    uint s = dispatchIndex(tid);
    if (s >= softBodyCount() || (softFlags() & SOFT_SLEEP_SYNC) == 0u) return;
    uint4 r = softBodyRec(s);
    uint core = asuint(softBodyRec1(s).z);
    bool all = true;
    for (uint k = 0u; k < r.y && all; k++)
    {
        BodyState b = gBodies[r.x + k];
        if (bodyIsDynamic(b) && !bodyIsSleeping(b) && gCcdFlags[r.x + k] == 0u) all = false;
    }
    if (all && core != SOFT_NO_BODY)
    {
        BodyState c = gBodies[core];
        if (bodyIsDynamic(c) && !bodyIsSleeping(c) && gCcdFlags[core] == 0u) all = false;
    }
    if (all) return;
    for (uint k = 0u; k < r.y; k++) gCcdFlags[r.x + k] = 0u;
    if (core != SOFT_NO_BODY) gCcdFlags[core] = 0u;
}

// Per step, after prediction: hydro rest lengths from the steering input, clear
// the break-group flags, set each particle's initial guess and seed its
// Chebyshev history with it.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_begin(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i < softBeamCount())
    {
        SoftBeam e = loadBeam(i);
        if (beamKind(e) == BEAM_HYDRO && !beamBroken(e))
        {
            float steer = softBodyRec1(beamSoftBody(e)).x;
            float target = e.rest0 * (1.0 - steer * e.coef);
            if (target != e.rest)
            {
                e.rest = max(target, 0.01);
                storeBeamState(i, e);
            }
        }
    }
    if (i < softGroupCount()) gSoftData[softGroupBase() + i] = float4(0, 0, 0, 0);
    if (i < softParticleTotal()) gSoftData[softFlagBase() + i] = float4(0, 0, 0, 0);
    if (i < softListCount())
    {
        // Particle initial guess (VBD adaptive initialisation): x⁻ + h v + h² g ã,
        // ã = last step's acceleration along gravity, clamped to [0, 1]. The
        // rigid predictor's at-rest rule (no h v below 0.1 m/s) would cap a
        // falling lattice at a few cm/s, since a Gauss–Seidel sweep over stiff
        // beams moves the whole structure only slowly; a full h² g drop at rest
        // would leave an under-converged sweep sagging (and its supports
        // over-loaded) every step.
        uint body = SU(softListBase() + i).x;
        uint q = softChebBase() + 3u * i;
        BodyState b = gBodies[body];
        if (bodyIsDynamic(b) && !bodyIsSleeping(b) && !wakePending(b))
        {
            AvbdParams p = PARAMS;
            float h = p.dt;
            float3 gv = float3(0.0, p.gravityY, 0.0);
            float3 vPrev = SD(q + 2u).xyz;
            float3 a = (b.linearVel - vPrev) / max(h, 1e-6);
            float weight = clamp(dot(a, gv) / max(dot(gv, gv), 1e-6), 0.0, 1.0);
            float3 x = b.initialPos + b.linearVel * h + gv * (h * h * weight);
            gBodies[body].position = x;
            gSoftData[q] = float4(x, 0.0);
            gSoftData[q + 1u] = float4(x, 0.0);
            gSoftData[q + 2u] = float4(b.linearVel, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Soft contacts: node–triangle (between soft bodies) and triangle–obstacle
// (spheres, capsules and boxes registered as obstacles: poles and bollards that
// would pass between nodes, rams whose edges would slip between them). Detected
// once per step at the initial guess, normals and barycentrics frozen for the
// step, the gap re-evaluated every iteration as a penalty ½k·min(C, 0)² with
// lagged Coulomb friction. Each particle reads only its own slots: a node its
// node slot, a triangle vertex the slots of its adjacent triangles.
// ---------------------------------------------------------------------------

// Closest point on triangle abc to p (Ericson 5.1.5); w = barycentrics.
float3 closestOnTriangle(float3 p, float3 a, float3 b, float3 c, out float3 w)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { w = float3(1, 0, 0); return a; }
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { w = float3(0, 1, 0); return b; }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        float v = d1 / (d1 - d3);
        w = float3(1.0 - v, v, 0.0);
        return a + v * ab;
    }
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { w = float3(0, 0, 1); return c; }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        float t = d2 / (d2 - d6);
        w = float3(1.0 - t, 0.0, t);
        return a + t * ac;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        w = float3(0.0, 1.0 - t, t);
        return b + t * (c - b);
    }
    float denom = 1.0 / (va + vb + vc);
    float v = vb * denom, u = vc * denom;
    w = float3(1.0 - v - u, v, u);
    return a + ab * v + ac * u;
}

// Closest points of segments p1q1 and p2q2 (Ericson 5.1.9): s, t parameters.
void closestSegSeg(float3 p1, float3 q1, float3 p2, float3 q2, out float s, out float t)
{
    float3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    if (a <= 1e-12 && e <= 1e-12) { s = 0.0; t = 0.0; return; }
    if (a <= 1e-12) { s = 0.0; t = clamp(f / e, 0.0, 1.0); return; }
    float c = dot(d1, r);
    if (e <= 1e-12) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); return; }
    float b = dot(d1, d2);
    float denom = a * e - b * b;
    s = denom > 1e-12 ? clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
    t = (b * s + f) / e;
    if (t < 0.0) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); }
    else if (t > 1.0) { t = 1.0; s = clamp((b - c) / a, 0.0, 1.0); }
}

// Closest points between segment p0p1 and triangle abc: q on the triangle
// (barycentrics w), cs on the segment.
void closestSegTriangle(float3 p0, float3 p1, float3 a, float3 b, float3 c, out float3 q, out float3 cs, out float3 w)
{
    float best = 1e30;
    float3 wq;
    float3 qq = closestOnTriangle(p0, a, b, c, wq);
    best = dot(qq - p0, qq - p0); q = qq; cs = p0; w = wq;
    qq = closestOnTriangle(p1, a, b, c, wq);
    float d = dot(qq - p1, qq - p1);
    if (d < best) { best = d; q = qq; cs = p1; w = wq; }
    float3 ea[3] = { a, b, c };
    float3 eb[3] = { b, c, a };
    [unroll]
    for (uint k = 0u; k < 3u; k++)
    {
        float s, t;
        closestSegSeg(p0, p1, ea[k], eb[k], s, t);
        float3 x = p0 + (p1 - p0) * s;
        float3 y = ea[k] + (eb[k] - ea[k]) * t;
        d = dot(x - y, x - y);
        if (d < best)
        {
            best = d; q = y; cs = x;
            w = k == 0u ? float3(1.0 - t, t, 0.0) : (k == 1u ? float3(0.0, 1.0 - t, t) : float3(t, 0.0, 1.0 - t));
        }
    }
    // The segment piercing the triangle.
    float3 n = cross(b - a, c - a);
    float d0 = dot(p0 - a, n), d1 = dot(p1 - a, n);
    if (d0 * d1 < 0.0)
    {
        float3 x = p0 + (p1 - p0) * (d0 / (d0 - d1));
        float3 wx;
        float3 qx = closestOnTriangle(x, a, b, c, wx);
        if (dot(qx - x, qx - x) < 1e-10) { q = qx; cs = x; w = wx; }
    }
}

uint triNode(uint t, uint k) { return SU(softTriBase() + t)[k]; }

// Per soft body: bounds of its particles at the initial guess, padded by this
// step's travel, for culling contact pairs.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_bounds(uint3 tid : SV_DispatchThreadID)
{
    uint s = dispatchIndex(tid);
    if (s >= softBodyCount()) return;
    uint4 r = softBodyRec(s);
    float3 mn = float3(1e30, 1e30, 1e30), mx = float3(-1e30, -1e30, -1e30);
    float travel = 0.0;
    for (uint k = 0u; k < r.y; k++)
    {
        BodyState b = gBodies[r.x + k];
        mn = min(mn, b.position);
        mx = max(mx, b.position);
        travel = max(travel, length(b.position - b.initialPos));
    }
    float pad = 2.0 * travel + 0.1;
    gSoftData[softAabbBase() + 2u * s] = float4(mn - pad, 0.0);
    gSoftData[softAabbBase() + 2u * s + 1u] = float4(mx + pad, 0.0);
}

bool softAabbHas(uint s, float3 p)
{
    float3 mn = SD(softAabbBase() + 2u * s).xyz, mx = SD(softAabbBase() + 2u * s + 1u).xyz;
    return all(p >= mn) && all(p <= mx);
}

bool softAabbOverlaps(uint s, float3 lo, float3 hi)
{
    float3 mn = SD(softAabbBase() + 2u * s).xyz, mx = SD(softAabbBase() + 2u * s + 1u).xyz;
    return all(lo <= mx) && all(hi >= mn);
}

// Per solved particle: the closest other-vehicle collision triangle it is in
// front of (or slightly behind) within the collision range.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_collide_nodes(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softListCount()) return;
    uint4 ent = SU(softListBase() + i);
    uint body = ent.x;
    uint sb = ent.w & 0xFFFFu;
    uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * particleOrdinal(sb, body);
    gSoftData[ns] = float4(0, 0, 0, asfloat(SOFT_NONE));
    if (!softContactsOn() || (ent.w & ENTRY_CONTACTER) == 0u) return;
    BodyState b = gBodies[body];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    float3 x = b.position, x0 = b.initialPos;
    float bestScore = 1e30;
    uint bestTri = SOFT_NONE;
    float3 bestN = float3(0, 0, 0), bestW = float3(0, 0, 0);
    float bestDelta = 0.0;
    float bestGap = 0.0;
    uint nb = softBodyCount();
    for (uint s2 = 0u; s2 < nb; s2++)
    {
        if (s2 == sb || !softAabbHas(s2, x)) continue;
        uint4 r2 = softBodyRec2(s2);
        float delta = asfloat(r2.z);
        for (uint t = r2.x; t < r2.x + r2.y; t++)
        {
            uint4 tri = SU(softTriBase() + t);
            if (((tri.w >> 8) & 1u) == 0u) continue;
            BodyState ba = gBodies[tri.x], bb = gBodies[tri.y], bc = gBodies[tri.z];
            float3 a = ba.position, bpos = bb.position, c = bc.position;
            if (triTorn(t, a, bpos, c)) continue;
            float3 nr = cross(bpos - a, c - a);
            float area = length(nr);
            if (area < 1e-8) continue;
            float3 n = nr / area;
            float d = dot(x - a, n);
            float reach = delta + length(x - x0) + 0.05;
            if (d > reach || d < -0.3) continue;
            // Step-start side: the node must come from the front face.
            float3 a0 = ba.initialPos, b0 = bb.initialPos, c0 = bc.initialPos;
            float3 n0 = cross(b0 - a0, c0 - a0);
            float l0 = length(n0);
            float d0 = l0 > 1e-8 ? dot(x0 - a0, n0 / l0) : d;
            if (d0 < -0.05) continue;
            float3 w;
            float3 q = closestOnTriangle(x - n * d, a, bpos, c, w);
            // Prism test: the node projects inside the triangle.
            if (dot(q - (x - n * d), q - (x - n * d)) > 1e-6) continue;
            float score = abs(d);
            if (score < bestScore)
            {
                bestScore = score; bestTri = t; bestN = n; bestW = w; bestDelta = delta;
                // Gap at the step start (frozen normal): the stabilised constraint
                // corrects only (1 − α) of an existing overlap per step.
                bestGap = dot(x0 - (w.x * a0 + w.y * b0 + w.z * c0), n) - delta;
            }
        }
    }
    if (bestTri == SOFT_NONE) return;
    gSoftData[ns] = float4(bestN, asfloat(bestTri));
    gSoftData[ns + 1u] = float4(bestW, bestDelta);
    gSoftData[ns + 2u] = float4(bestGap, 0.0, 0.0, 0.0);
    gSoftData[softFlagBase() + particleOrdinal(sb, body)] = float4(1, 0, 0, 0);
}

// Closest points between an (oriented) box and triangle abc: q on the triangle
// (barycentrics w), cs on the box surface. A triangle corner inside the box gives
// `inside`, with cs its exit point through the nearest face and n that face's
// normal. Otherwise the closest pair is a triangle corner against the box or a box
// edge against the triangle (which covers face–corner and edge–edge).
void boxTriangle(Collider col, BodyState ob, float3 a, float3 b, float3 c, out float3 q, out float3 cs, out float3 w,
    out bool inside, out float3 n)
{
    float4 rot = quatMul(ob.orientation, col.localRot);
    float3 centre = transformPoint(ob.position, ob.orientation, col.localPos);
    float3 h = col.params.xyz;
    inside = false;
    n = float3(0, 1, 0);
    float best = 1e30;
    float deepest = 0.0;
    float3 corners[3] = { a, b, c };
    float3 bary[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };
    [unroll]
    for (uint k = 0u; k < 3u; k++)
    {
        float3 l = quatRotate(quatConjugate(rot), corners[k] - centre);
        float3 clamped = clamp(l, -h, h);
        if (all(abs(l) < h))
        {
            // Inside: leave through the nearest face.
            float3 room = h - abs(l);
            uint ax = room.x < room.y ? (room.x < room.z ? 0u : 2u) : (room.y < room.z ? 1u : 2u);
            float3 out = l;
            out[ax] = h[ax] * (l[ax] >= 0.0 ? 1.0 : -1.0);
            if (room[ax] > deepest)
            {
                deepest = room[ax];
                inside = true;
                q = corners[k];
                w = bary[k];
                cs = centre + quatRotate(rot, out);
                float3 fn = float3(0, 0, 0);
                fn[ax] = l[ax] >= 0.0 ? 1.0 : -1.0;
                n = quatRotate(rot, fn);
            }
            continue;
        }
        float3 p = centre + quatRotate(rot, clamped);
        float d = dot(corners[k] - p, corners[k] - p);
        if (!inside && d < best) { best = d; q = corners[k]; w = bary[k]; cs = p; }
    }
    if (inside) return;
    // The twelve box edges against the triangle.
    [unroll]
    for (uint e = 0u; e < 12u; e++)
    {
        uint axis = e / 4u;
        uint u = (axis + 1u) % 3u, v = (axis + 2u) % 3u;
        float3 lo = float3(0, 0, 0), hi = float3(0, 0, 0);
        lo[axis] = -h[axis]; hi[axis] = h[axis];
        lo[u] = (e & 1u) != 0u ? h[u] : -h[u]; hi[u] = lo[u];
        lo[v] = (e & 2u) != 0u ? h[v] : -h[v]; hi[v] = lo[v];
        float3 qq, cc, ww;
        closestSegTriangle(centre + quatRotate(rot, lo), centre + quatRotate(rot, hi), a, b, c, qq, cc, ww);
        float d = dot(qq - cc, qq - cc);
        if (d < best) { best = d; q = qq; cs = cc; w = ww; }
    }
}

// Per collision triangle: its deepest thin obstacle, and the (up to four) nodes
// that chose it in soft_collide_nodes (the triangle side of their contacts).
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_collide_tris(uint3 tid : SV_DispatchThreadID)
{
    uint t = dispatchIndex(tid);
    if (t >= softTriCount()) return;
    uint tc = softTriContactBase() + TRI_CONTACT_STRIDE * t;
    gSoftData[tc] = float4(0, 0, 0, asfloat(SOFT_NONE));
    for (uint m = 0u; m < TRI_NODE_CONTACTS; m++) gSoftData[tc + 3u + 3u * m] = float4(0, 0, 0, asfloat(SOFT_NONE));
    if (!softContactsOn()) return;
    uint4 tri = SU(softTriBase() + t);
    if (((tri.w >> 8) & 1u) == 0u) return;
    uint sb = tri.w & 255u;
    BodyState ba = gBodies[tri.x], bb = gBodies[tri.y], bc = gBodies[tri.z];
    if (bodyIsSleeping(ba) && bodyIsSleeping(bb) && bodyIsSleeping(bc)) return;
    float3 a = ba.position, b = bb.position, c = bc.position;
    if (triTorn(t, a, b, c)) return;
    float delta = asfloat(softBodyRec2(sb).z);
    float travel = max(max(length(a - ba.initialPos), length(b - bb.initialPos)), length(c - bc.initialPos));
    float3 lo = min(min(a, b), c), hi = max(max(a, b), c);
    bool marked = false;

    // Thin obstacles.
    float bestGap = 1e30;
    uint nObs = softObstacleCount();
    for (uint k = 0u; k < nObs; k++)
    {
        uint ci = SU(softObstacleBase() + k).x;
        Collider col = gColliders[ci];
        BodyState ob = gBodies[col.body];
        float r = col.params.x;
        float3 q, cs, w;
        bool inside = false;
        float3 insideN = float3(0, 1, 0);
        if (col.shape == SHAPE_SPHERE)
        {
            cs = transformPoint(ob.position, ob.orientation, col.localPos);
            q = closestOnTriangle(cs, a, b, c, w);
        }
        else if (col.shape == SHAPE_CAPSULE)
        {
            float3 ax = quatRotate(quatMul(ob.orientation, col.localRot), float3(0.0, col.params.y, 0.0));
            float3 mid = transformPoint(ob.position, ob.orientation, col.localPos);
            closestSegTriangle(mid - ax, mid + ax, a, b, c, q, cs, w);
        }
        else if (col.shape == SHAPE_BOX)
        {
            r = 0.0;
            boxTriangle(col, ob, a, b, c, q, cs, w, inside, insideN);
        }
        else continue;
        float3 dv = q - cs;
        float dist = length(dv);
        float gap = inside ? -dist : dist - r;
        if (gap > delta + travel + 0.05 || gap >= bestGap) continue;
        float3 n = inside ? insideN : (dist > 1e-5 ? dv / dist : normalize(cross(b - a, c - a)));
        // Touching or crossing (no direction from the gap): point away from the obstacle.
        if (!inside && dist <= 1e-5 && dot(n, (a + b + c) / 3.0 - cs) < 0.0) n = -n;
        bestGap = gap;
        // Gap at the step start (triangle at x⁻) for the stabilised constraint.
        float3 q0 = w.x * ba.initialPos + w.y * bb.initialPos + w.z * bc.initialPos;
        gSoftData[tc] = float4(n, asfloat(ci));
        gSoftData[tc + 1u] = float4(cs + n * r, dot(q0 - (cs + n * r), n) - delta);
        gSoftData[tc + 2u] = float4(w, delta);
        marked = true;
    }

    // Nodes of other soft bodies that chose this triangle.
    uint found = 0u;
    uint nb = softBodyCount();
    for (uint s2 = 0u; s2 < nb && found < TRI_NODE_CONTACTS; s2++)
    {
        if (s2 == sb || !softAabbOverlaps(s2, lo - 0.3, hi + 0.3)) continue;
        uint4 r2 = softBodyRec(s2);
        uint ordBase = asuint(softBodyRec1(s2).w);
        for (uint k = 0u; k < r2.y && found < TRI_NODE_CONTACTS; k++)
        {
            uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * (ordBase + k);
            float4 s0 = SD(ns);
            if (asuint(s0.w) != t) continue;
            gSoftData[tc + 3u + 3u * found] = float4(s0.xyz, asfloat(r2.x + k));
            gSoftData[tc + 4u + 3u * found] = SD(ns + 1u);
            gSoftData[tc + 5u + 3u * found] = SD(ns + 2u);
            found++;
            marked = true;
        }
    }
    if (marked)
    {
        uint ob = asuint(softBodyRec1(sb).w) - softBodyRec(sb).x;
        gSoftData[softFlagBase() + ob + tri.x] = float4(1, 0, 0, 0);
        gSoftData[softFlagBase() + ob + tri.y] = float4(1, 0, 0, 0);
        gSoftData[softFlagBase() + ob + tri.z] = float4(1, 0, 0, 0);
    }
}

// Penalty contact on one particle: Jacobian coefficient s (1 for a node, ±w for a
// triangle vertex), gap C along n, gap C0 at the step start, tangential relative
// travel u this step. As rigid contacts (AVBD α), only (1 − α) of an overlap
// present at the step start is corrected per step, so a deep overlap (a node
// freed inside a crushed panel) is not fired out.
void softContactTerm(inout float3 g, inout float3x3 H, float s, float C, float C0, float3 n, float3 u)
{
    C -= PARAMS.alpha * min(C0, 0.0);
    if (C >= 0.0) return;
    float k = softContactStiffness();
    float3x3 nn = float3x3(n * n.x, n * n.y, n * n.z);
    g += n * (s * k * C);
    H += nn * (s * s * k);
    // Lagged Coulomb friction: a tangential spring capped at μ·|f_n| (secant stiffness).
    float3 ut = u - n * dot(u, n);
    float ul = length(ut);
    if (ul < 1e-9) { H += (float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1) - nn) * (s * s * k); return; }
    float cap = softContactFriction() * k * (-C);
    float kt = min(k, cap / ul);
    g += ut * (s * kt);
    H += (float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1) - nn) * (s * s * kt);
}

float3 triTravel(uint4 tri, float3 w)
{
    BodyState a = gBodies[tri.x], b = gBodies[tri.y], c = gBodies[tri.z];
    return w.x * (a.position - a.initialPos) + w.y * (b.position - b.initialPos) + w.z * (c.position - c.initialPos);
}

float3 triPoint(uint4 tri, float3 w)
{
    return w.x * gBodies[tri.x].position + w.y * gBodies[tri.y].position + w.z * gBodies[tri.z].position;
}

// Soft contact terms of one particle (node side and triangle-vertex side).
void softContactsFor(uint body, uint sb, float3 x, float3 x0, uint triStart, uint triCount, inout float3 g, inout float3x3 H)
{
    if (!softContactsOn()) return;
    // Detection flags every particle in a soft contact this step (node or
    // triangle vertex); the rest have nothing to read.
    uint ord = particleOrdinal(sb, body);
    if (SD(softFlagBase() + ord).x == 0.0) return;
    uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * ord;
    float4 s0 = SD(ns);
    uint t = asuint(s0.w);
    if (t != SOFT_NONE)
    {
        float4 s1 = SD(ns + 1u);
        uint4 tri = SU(softTriBase() + t);
        float C = dot(s0.xyz, x - triPoint(tri, s1.xyz)) - s1.w;
        softContactTerm(g, H, 1.0, C, SD(ns + 2u).x, s0.xyz, (x - x0) - triTravel(tri, s1.xyz));
    }
    uint adj = softAdjBase();
    for (uint k = 0u; k < triCount; k++)
    {
        uint wi = triStart + k;
        uint tt = SU(adj + (wi >> 2))[wi & 3u];
        uint4 tri = SU(softTriBase() + tt);
        uint tc = softTriContactBase() + TRI_CONTACT_STRIDE * tt;
        // This particle's weight in the triangle.
        uint j = tri.x == body ? 0u : (tri.y == body ? 1u : 2u);
        float4 o0 = SD(tc);
        if (asuint(o0.w) != SOFT_NONE)
        {
            float4 o1 = SD(tc + 1u);
            float4 o2 = SD(tc + 2u);
            float wj = o2[j];
            float C = dot(o0.xyz, triPoint(tri, o2.xyz) - o1.xyz) - o2.w;
            softContactTerm(g, H, wj, C, o1.w, o0.xyz, triTravel(tri, o2.xyz));
        }
        for (uint m = 0u; m < TRI_NODE_CONTACTS; m++)
        {
            float4 c0 = SD(tc + 3u + 3u * m);
            uint node = asuint(c0.w);
            if (node == SOFT_NONE) continue;
            float4 c1 = SD(tc + 4u + 3u * m);
            BodyState nb = gBodies[node];
            float C = dot(c0.xyz, nb.position - triPoint(tri, c1.xyz)) - c1.w;
            float3 u = (nb.position - nb.initialPos) - triTravel(tri, c1.xyz);
            softContactTerm(g, H, -c1[j], C, SD(tc + 5u + 3u * m).x, c0.xyz, u);
        }
    }
}

// VBD block solve of one particle (colour-list entry `ent`).
void softSolveEntry(uint4 ent)
{
    uint body = ent.x;
    BodyState b = gBodies[body];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b)) return;
    AvbdParams p = PARAMS;
    float h = max(p.dt, 1e-6);

    // Inertia, contacts and joints: the rigid accumulation, rotation frozen.
    Mat6 A;
    Vec6 rhs;
    primalAccumulate(body, 0u, 1u, A, rhs);
    float3x3 H = float3x3(
        A.m[0], A.m[6], A.m[12],
        A.m[6], A.m[7], A.m[13],
        A.m[12], A.m[13], A.m[14]);
    float3 g = float3(rhs.v[0], rhs.v[1], rhs.v[2]);

    float3 xa = b.position;
    float3 xa0 = b.initialPos;
    uint adj = softAdjBase();
    uint nBeams = ent.z & 0xFFFFu;
    bool plastic = (softFlags() & SOFT_PLASTICITY) != 0u;
    float ts = softThresholdScale();
    for (uint k = 0u; k < nBeams; k++)
    {
        uint w = ent.y + k;
        uint e = SU(adj + (w >> 2))[w & 3u];
        SoftBeam sb = loadBeam(e);
        if (beamBroken(sb)) continue;
        uint other = sb.a == body ? sb.b : sb.a;
        BodyState o = gBodies[other];
        float3 d = xa - o.position;
        float len = length(d);
        if (len < 1e-6) continue;
        float3 n = d / len;
        float2 el = beamElastic(sb, len, sb.penalty);
        float c = beamDampCoef(sb, len);
        float len0 = length(xa0 - o.initialPos);
        float d1 = el.x + c / h * (len - len0);
        float d2 = el.y + c / h;
        float tension = el.x;
        if (plastic && beamKind(sb) == BEAM_NORMAL && sb.spring > 0.0 && !beamHard(sb))
        {
            // In-solve plasticity (engine_softbody::model::yield_cap): past its
            // threshold the beam flows at it. The curvature stays elastic, an upper
            // bound that keeps the undamped block step from overshooting.
            d1 = clamp(d1, -sb.deformC * ts, sb.deformT * ts);
            tension = min(tension, max(d1, 0.0));
        }
        float lat = max(tension / len, 0.0);
        g += n * d1;
        float3x3 nn = float3x3(n * n.x, n * n.y, n * n.z);
        H += nn * (d2 - lat);
        H[0][0] += lat; H[1][1] += lat; H[2][2] += lat;
    }
    softContactsFor(body, ent.w & 0xFFFFu, xa, xa0, ent.y + nBeams, ent.z >> 16, g, H);

    // Closed-form 3×3 solve (adjugate); skip near-singular blocks (VBD).
    float3 c0 = float3(H[0][0], H[1][0], H[2][0]);
    float3 c1 = float3(H[0][1], H[1][1], H[2][1]);
    float3 c2 = float3(H[0][2], H[1][2], H[2][2]);
    float3 r0 = cross(c1, c2);
    float det = dot(c0, r0);
    float scale = max(max(abs(H[0][0]), abs(H[1][1])), abs(H[2][2]));
    if (!(abs(det) > 1e-12 * scale * scale * scale)) return;
    float3 r1 = cross(c2, c0);
    float3 r2 = cross(c0, c1);
    float3 dx = -(r0 * g.x + r1 * g.y + r2 * g.z) / det;
    float maxLin = p.trustLin * 0.5;
    float l = length(dx);
    if (l > maxLin && l > 1e-12) dx *= maxLin / l;
    if (dx.x != dx.x || dx.y != dx.y || dx.z != dx.z) return;
    gBodies[body].position = xa + dx;
}

// One particle colour (PASS_U): VBD block solve of each particle.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_primal(uint3 tid : SV_DispatchThreadID)
{
    uint color = PASS_U & 255u;
    gStabilize = false;
    uint t = dispatchIndex(tid);
    uint4 cr = SU(softColorBase() + color);
    if (color >= softColorCount() || t >= cr.y) return;
    softSolveEntry(SU(softListBase() + cr.x + t));
}

// Chebyshev semi-iterative acceleration of the particle sweep (VBD §3.6),
// PASS_U = iteration index. Particles with any contact or joint edge this step
// are left alone.
// Chebyshev step of colour-list entry `i` after iteration `it`.
void softChebyshevEntry(uint i, uint it)
{
    float rho = softChebyshevRho();
    uint body = SU(softListBase() + i).x;
    uint q = softChebBase() + 3u * i;
    float3 prev2 = SD(q).xyz;
    float3 prev1 = SD(q + 1u).xyz;
    BodyState b = gBodies[body];
    float3 x = b.position;
    uint4 ent = SU(softListBase() + i);
    bool soft = softContactsOn() && SD(softFlagBase() + particleOrdinal(ent.w & 0xFFFFu, body)).x != 0.0;
    if (it > 0u && bodyIsDynamic(b) && !bodyIsSleeping(b) && !wakePending(b) && !soft
        && SOLVE_INDEX(body + 1u) == SOLVE_INDEX(body))
    {
        float rho2 = rho * rho;
        float omega = 2.0 / (2.0 - rho2);
        for (uint k = 2u; k <= it; k++) omega = 4.0 / (4.0 - rho2 * omega);
        x = prev2 + (x - prev2) * omega;
        gBodies[body].position = x;
    }
    gSoftData[q] = float4(prev1, 0.0);
    gSoftData[q + 1u] = float4(x, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_chebyshev(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softListCount() || softChebyshevRho() <= 0.0) return;
    softChebyshevEntry(i, PASS_U);
}

// One whole particle iteration in a single 256-lane workgroup: every colour in
// order with a device barrier between colours (the same Gauss–Seidel order as
// one soft_primal dispatch per colour), then the Chebyshev step. Hosts use it
// while every colour fits in one workgroup pass, where dispatch latency, not
// arithmetic, dominates a small scene. PASS_U = iteration index.
[shader("compute")]
[numthreads(256, 1, 1)]
void soft_primal_fused(uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    gStabilize = false;
    if (local == 0u) gsUniformLanes = softColorCount();
    GROUP_SYNC();
    uint colors = gsUniformLanes;
    for (uint c = 0u; c < colors; c++)
    {
        uint4 cr = SU(softColorBase() + c);
        for (uint t = local; t < cr.y; t += 256u) softSolveEntry(SU(softListBase() + cr.x + t));
        DeviceMemoryBarrierWithGroupSync();
    }
    if (softChebyshevRho() > 0.0)
    {
        uint n = softListCount();
        for (uint i = local; i < n; i += 256u) softChebyshevEntry(i, PASS_U);
    }
}

// Dual update per beam: the finite-stiffness ramp k ← min(k*, k + β|C|), and the
// multiplier of hard (augmented-Lagrangian) beams.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_dual(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softBeamCount()) return;
    SoftBeam e = loadBeam(i);
    if (beamBroken(e)) return;
    float len = length(gBodies[e.a].position - gBodies[e.b].position);
    float cv = len - e.rest;
    if (beamHard(e))
    {
        e.lambda += e.penalty * cv;
        e.penalty = min(e.penalty + softBeta() * abs(cv), PARAMS.kMaxN);
    }
    else
    {
        if (e.penalty >= e.spring) return;
        e.penalty = min(e.penalty + softBeta() * abs(cv), e.spring);
    }
    storeBeamState(i, e);
}

// Post-step plasticity and breaking at the converged state (Rigs of Rods rules).
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_plastic(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softBeamCount()) return;
    SoftBeam e = loadBeam(i);
    if (beamBroken(e)) return;
    BodyState ba = gBodies[e.a];
    BodyState bb = gBodies[e.b];
    // Both ends asleep (or static): nothing moved this step.
    bool movingA = bodyIsDynamic(ba) && !bodyIsSleeping(ba);
    bool movingB = bodyIsDynamic(bb) && !bodyIsSleeping(bb);
    if (!movingA && !movingB) return;
    float h = max(PARAMS.dt, 1e-6);
    float len = length(ba.position - bb.position);
    float len0 = length(ba.initialPos - bb.initialPos);
    float force = beamReadout(e, len, len0, h);
    uint flags = softFlags();
    float delta = len - e.rest;
    float ts = softThresholdScale();
    if ((flags & SOFT_PLASTICITY) != 0u && beamKind(e) == BEAM_NORMAL && e.spring > 0.0)
    {
        float k = e.spring;
        float dc = e.deformC * ts;
        float dt = e.deformT * ts;
        if (force > dc && delta < 0.0)
        {
            // The solve capped it at the threshold: that is the force it carried.
            force = dc;
            float old = e.rest;
            float deform = delta + dc * (1.0 - e.coef) / k;
            e.rest = max(e.rest + deform, softMinLength());
            if (e.shortB > 0.0) e.rest = max(e.rest, e.rest0 * e.shortB);
            e.deformC *= old / e.rest;
            e.meta |= BEAM_YIELDED;
        }
        else if (-force > dt && delta > 0.0)
        {
            // Tension: it carried the threshold, and the plastic stretch uses up
            // its strength until it tears.
            force = -dt;
            float old = e.rest;
            float deform = delta - dt * (1.0 - e.coef) / k;
            e.rest += deform;
            if (e.longB > 0.0) e.rest = min(e.rest, e.rest0 * e.longB);
            e.deformT *= e.rest / old;
            e.strength -= deform * k / ts;
            e.meta |= BEAM_YIELDED;
        }
    }
    e.force = force;
    if ((flags & SOFT_BREAKING) != 0u)
    {
        bool broke = beamKind(e) == BEAM_SUPPORT
            ? (e.longB > 0.0 && delta > e.rest * e.longB)
            : abs(force) > e.strength * ts;
        if (broke)
        {
            e.meta |= BEAM_BROKEN;
            uint g = beamGroup(e);
            if (g != 0u && g < softGroupCount()) gSoftData[softGroupBase() + g] = float4(1, 0, 0, 0);
            emitEvent(EVENT_BEAM_BREAK, i, asuint(abs(force)), beamSoftBody(e));
        }
    }
    storeBeamState(i, e);
}

// Detacher groups: a beam breaking breaks every beam of its group.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_break_groups(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softBeamCount()) return;
    SoftBeam e = loadBeam(i);
    uint g = beamGroup(e);
    if (g == 0u || g >= softGroupCount() || beamBroken(e)) return;
    if (SD(softGroupBase() + g).x == 0.0) return;
    e.meta |= BEAM_BROKEN;
    storeBeamState(i, e);
    emitEvent(EVENT_BEAM_BREAK, i, asuint(abs(e.force)), beamSoftBody(e));
}
