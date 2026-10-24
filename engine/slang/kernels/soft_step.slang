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

#include "vehicle_step.slang" // world_step.slang, and vehicleIterateOne for soft_iterate

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
static const uint SOFT_RIGID_MODE = 16u;
static const uint SOFT_BDF2 = 32u;
// Self-collision skips triangles nearer than this to the node in the rest body.
static const float SELF_REST_GAP = 0.12;
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
uint softBodyColorBase() { return SU(8u).x; }
uint softBodyColorStride() { return SU(8u).y; }
uint softTriPlaneBase() { return SU(8u).z; }
uint softPairBase() { return SU(8u).w; }
uint softBdfBase() { return SU(9u).x; }
bool softBdf2On() { return (softFlags() & SOFT_BDF2) != 0u; }
// BDF2 for this particle this step: on, and it has a position one step back.
bool softBdf2For(uint ord) { return softBdf2On() && SD(softBdfBase() + ord).w == 1.0; }
static const uint SOFT_PAIR_STRIDE = 8u; // engine_softbody::SOFT_PAIR_STRIDE
static const uint SOFT_PAIR_MAX = 31u;
static const uint SOFT_PAIR_ALL = 0xFFFFFFFFu;
// Soft bodies near body s this step (soft_pairs): how many to walk, and the k-th
// (ascending, s itself included). Too many for the list: every body.
uint softPairCount(uint s)
{
    uint n = SU(softPairBase() + SOFT_PAIR_STRIDE * s).x;
    return n == SOFT_PAIR_ALL ? softBodyCount() : n;
}
uint softPairAt(uint s, uint k)
{
    uint n = SU(softPairBase() + SOFT_PAIR_STRIDE * s).x;
    if (n == SOFT_PAIR_ALL) return k;
    uint w = k + 1u;
    return SU(softPairBase() + SOFT_PAIR_STRIDE * s + (w >> 2))[w & 3u];
}
// A skin triangle stretched past this × its rest edges is torn (the parts it
// joined have separated): it stops colliding, as renderers stop drawing it.
static const float TRI_TEAR = 2.5;
bool triTorn(uint t, float3 a, float3 b, float3 c)
{
    float3 r = SD(softTriRestBase() + t).xyz;
    return length(a - b) > TRI_TEAR * r.x || length(b - c) > TRI_TEAR * r.y || length(c - a) > TRI_TEAR * r.z;
}
bool softContactsOn() { return (softFlags() & SOFT_CONTACTS) != 0u; }

// Particle position reads and writes of the particle solve.
float3 softPos(uint b) { return gBodies[b].position; }
float3 softPos0(uint b) { return gBodies[b].initialPos; }
void softSetPos(uint b, float3 x) { gBodies[b].position = x; }
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
uint4 softBodyRec(uint s) { return SU(softBodyBase() + 6u * s); }
float4 softBodyRec1(uint s) { return SD(softBodyBase() + 6u * s + 1u); }
uint4 softBodyRec2(uint s) { return SU(softBodyBase() + 6u * s + 2u); }
// Rigid mode: [mode, quiet steps, flags (bit 0 has core) | joints << 8 | quiet ms << 16, first core joint].
uint softModeSlot(uint s) { return softBodyBase() + 6u * s + 3u; }
bool softIsRigid(uint s) { return SU(softModeSlot(s)).x == 1u; }
uint softRigidBase() { return SU(3u).w; }
float3 quatRotateInv(float4 q, float3 v) { return quatRotate(float4(-q.xyz, q.w), v); }

// Ordinal (index across every soft particle) of a particle body.
uint particleOrdinal(uint s, uint body) { return asuint(softBodyRec1(s).w) + body - softBodyRec(s).x; }

void softWakeBody(uint i)
{
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b) || !bodyIsSleeping(b)) return;
    gBodies[i].flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
    gBodies[i].sleepTimer = 0.0;
}

// A soft body sleeps and wakes as a whole: when any of its particles (or its
// rigid core) is awake, wake every sleeping one before the step builds its lists.
groupshared uint gsBodyFlags[2];
groupshared float4 gsBodyLo[64];
groupshared float4 gsBodyHi[64];
// One 64-lane workgroup per soft body (hosts size it as bodies × 64 threads).
uint softGroupBody(uint3 gid) { return gid.y * 32768u + gid.x; }

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_wake(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint s = softGroupBody(gid), local = lid.x;
    bool live = s < softBodyCount() && (softFlags() & SOFT_SLEEP_SYNC) != 0u;
    uint4 r = live ? softBodyRec(s) : uint4(0, 0, 0, 0);
    uint core = live ? asuint(softBodyRec1(s).z) : SOFT_NO_BODY;
    if (local == 0u) { gsBodyFlags[0] = 0u; gsBodyFlags[1] = 0u; }
    GroupMemoryBarrierWithGroupSync();
    // [0]: something awake, [1]: something asleep (particles and the core).
    for (uint k = local; k < r.y + 1u; k += 64u)
    {
        uint body = k < r.y ? r.x + k : core;
        if (body == SOFT_NO_BODY) continue;
        BodyState b = gBodies[body];
        if (!bodyIsDynamic(b)) continue;
        gsBodyFlags[bodyIsSleeping(b) ? 1u : 0u] = 1u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (!live || gsBodyFlags[0] == 0u || gsBodyFlags[1] == 0u) return;
    for (uint k = local; k < r.y; k += 64u) softWakeBody(r.x + k);
    if (local == 0u && core != SOFT_NO_BODY) softWakeBody(core);
}

// Sleep candidates (gCcdFlags ≠ 0 after extract_velocity_sleep and the island
// gate) only stand when every particle and the core of the soft body agree.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_sleep(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint s = softGroupBody(gid), local = lid.x;
    bool live = s < softBodyCount() && (softFlags() & SOFT_SLEEP_SYNC) != 0u;
    uint4 r = live ? softBodyRec(s) : uint4(0, 0, 0, 0);
    uint core = live ? asuint(softBodyRec1(s).z) : SOFT_NO_BODY;
    if (local == 0u) gsBodyFlags[0] = 0u;
    GroupMemoryBarrierWithGroupSync();
    // [0]: an awake particle (or the core) that is not a sleep candidate.
    for (uint k = local; k < r.y + 1u; k += 64u)
    {
        uint body = k < r.y ? r.x + k : core;
        if (body == SOFT_NO_BODY) continue;
        BodyState b = gBodies[body];
        if (bodyIsDynamic(b) && !bodyIsSleeping(b) && gCcdFlags[body] == 0u) gsBodyFlags[0] = 1u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (!live || gsBodyFlags[0] == 0u) return;
    for (uint k = local; k < r.y; k += 64u) gCcdFlags[r.x + k] = 0u;
    if (local == 0u && core != SOFT_NO_BODY) gCcdFlags[core] = 0u;
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
        uint4 ent = SU(softListBase() + i);
        uint body = ent.x;
        uint q = softChebBase() + 3u * i;
        BodyState b = gBodies[body];
        uint sb = ent.w & 0xFFFFu;
        uint core = asuint(softBodyRec1(sb).z);
        uint ord = particleOrdinal(sb, body);
        if (bodyIsDynamic(b) && !bodyIsSleeping(b) && !wakePending(b) && core != SOFT_NO_BODY && softIsRigid(sb))
        {
            // Rigid mode: the particle rides its core (predicted pose).
            BodyState c = gBodies[core];
            float3 x = c.position + quatRotate(c.orientation, SD(softRigidBase() + 2u * ord).xyz);
            gBodies[body].position = x;
            gSoftData[q] = float4(x, 0.0);
            gSoftData[q + 1u] = float4(x, 0.0);
            gSoftData[q + 2u] = float4(b.linearVel, 0.0);
        }
        else if (bodyIsDynamic(b) && !bodyIsSleeping(b) && !wakePending(b))
        {
            // Soft: remember where the particle sits in its core's frame at the
            // start of the step (what rigid mode would hold it to).
            if (core != SOFT_NO_BODY)
            {
                BodyState c = gBodies[core];
                gSoftData[softRigidBase() + 2u * ord] = float4(quatRotateInv(c.initialRot, b.initialPos - c.initialPos), 0.0);
            }
            AvbdParams p = PARAMS;
            float h = p.dt;
            float3 gv = float3(0.0, p.gravityY, 0.0);
            float3 vPrev = SD(q + 2u).xyz;
            float3 a = (b.linearVel - vPrev) / max(h, 1e-6);
            float weight = clamp(dot(a, gv) / max(dot(gv, gv), 1e-6), 0.0, 1.0);
            float3 x = b.initialPos + b.linearVel * h + gv * (h * h * weight);
            if (softBdf2For(ord))
            {
                // BDF2 inertial target: (4/3)xₙ − (1/3)xₙ₋₁ + (2h/3)((4/3)vₙ − (1/3)vₙ₋₁)
                // + (4/9)h²g, weighted (9/4)·m/h² in the solve.
                float3 xp = SD(softBdfBase() + ord).xyz;
                float3 y = (4.0 / 3.0) * b.initialPos - (1.0 / 3.0) * xp
                    + (2.0 * h / 3.0) * ((4.0 / 3.0) * b.linearVel - (1.0 / 3.0) * vPrev)
                    + gv * (4.0 / 9.0 * h * h);
                gAabbs[body].mn = y;
            }
            gBodies[body].position = x;
            gSoftData[q] = float4(x, 0.0);
            gSoftData[q + 1u] = float4(x, 0.0);
            gSoftData[q + 2u] = float4(b.linearVel, 0.0);
        }
    }
}

// After velocity extraction: BDF2 velocity v = (3x − 4xₙ + xₙ₋₁)/(2h) where the
// particle has its history, then xₙ becomes the history (none for a sleeping or
// static particle, or under backward Euler).
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_velocity(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softListCount()) return;
    uint4 ent = SU(softListBase() + i);
    uint body = ent.x;
    uint ord = particleOrdinal(ent.w & 0xFFFFu, body);
    uint slot = softBdfBase() + ord;
    BodyState b = gBodies[body];
    if (!softBdf2On() || !bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b))
    {
        gSoftData[slot] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    float4 xp = SD(slot);
    // This step was BDF2 (w = 1, or 2: a beam of it yielded or broke since).
    if (xp.w >= 1.0)
        gBodies[body].linearVel = (3.0 * b.position - 4.0 * b.initialPos + xp.xyz) / (2.0 * max(PARAMS.dt, 1e-6));
    // Next step BDF2 only if nothing interrupted it: no yield or break of its
    // beams, no contact (an impact's velocity jump would be extrapolated).
    bool contact = SOLVE_INDEX(body + 1u) != SOLVE_INDEX(body)
        || (softContactsOn() && SD(softFlagBase() + ord).x != 0.0);
    bool smooth = xp.w != 2.0 && !contact;
    gSoftData[slot] = float4(b.initialPos, smooth ? 1.0 : 0.0);
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
void soft_bounds(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint s = softGroupBody(gid), local = lid.x;
    bool live = s < softBodyCount();
    uint4 r = live ? softBodyRec(s) : uint4(0, 0, 0, 0);
    float3 mn = float3(1e30, 1e30, 1e30), mx = float3(-1e30, -1e30, -1e30);
    float travel = 0.0;
    for (uint k = local; k < r.y; k += 64u)
    {
        float3 x = gBodies[r.x + k].position;
        mn = min(mn, x);
        mx = max(mx, x);
        travel = max(travel, length(x - gBodies[r.x + k].initialPos));
    }
    gsBodyLo[local] = float4(mn, travel);
    gsBodyHi[local] = float4(mx, 0.0);
    GroupMemoryBarrierWithGroupSync();
    if (!live || local != 0u) return;
    for (uint k = 1u; k < 64u; k++)
    {
        mn = min(mn, gsBodyLo[k].xyz);
        mx = max(mx, gsBodyHi[k].xyz);
        travel = max(travel, gsBodyLo[k].w);
    }
    float pad = 2.0 * travel + 0.1;
    gSoftData[softAabbBase() + 2u * s] = float4(mn - pad, 0.0);
    gSoftData[softAabbBase() + 2u * s + 1u] = float4(mx + pad, 0.0);
}

// Per soft body, after soft_bounds: the soft bodies (itself included, ascending)
// whose bounds come within both skins + 0.3 m of its own. Every later test that
// walks other bodies (node contacts, a triangle's nodes, rigid-mode promotion)
// asks for less, so walking this list instead of every body finds the same.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_pairs(uint3 tid : SV_DispatchThreadID)
{
    uint s = dispatchIndex(tid);
    uint nb = softBodyCount();
    if (s >= nb) return;
    float m = asfloat(softBodyRec2(s).z) + 0.3;
    float3 lo = SD(softAabbBase() + 2u * s).xyz - m, hi = SD(softAabbBase() + 2u * s + 1u).xyz + m;
    uint base = softPairBase() + SOFT_PAIR_STRIDE * s;
    uint n = 0u;
    uint4 slot = uint4(0, 0, 0, 0);
    for (uint s2 = 0u; s2 < nb && n <= SOFT_PAIR_MAX; s2++)
    {
        float m2 = asfloat(softBodyRec2(s2).z);
        float3 lo2 = SD(softAabbBase() + 2u * s2).xyz - m2, hi2 = SD(softAabbBase() + 2u * s2 + 1u).xyz + m2;
        if (s2 != s && !(all(lo <= hi2) && all(lo2 <= hi))) continue;
        n++;
        if (n > SOFT_PAIR_MAX) break;
        slot[n & 3u] = s2;
        if ((n & 3u) == 3u) { gSoftData[base + (n >> 2)] = asfloat(slot); slot = uint4(0, 0, 0, 0); }
    }
    if ((n & 3u) != 3u && n <= SOFT_PAIR_MAX) gSoftData[base + (n >> 2)] = asfloat(slot);
    uint4 head = SU(base);
    head.x = n > SOFT_PAIR_MAX ? SOFT_PAIR_ALL : n;
    gSoftData[base] = asfloat(head);
}

bool softAabbHas(uint s, float3 p)
{
    float3 mn = SD(softAabbBase() + 2u * s).xyz, mx = SD(softAabbBase() + 2u * s + 1u).xyz;
    return all(p >= mn) && all(p <= mx);
}

// World AABB of a soft obstacle: sphere, capsule, oriented box (other shapes:
// the bounding radius in params.w).
void obstacleBounds(Collider col, BodyState ob, out float3 lo, out float3 hi)
{
    float3 c = transformPoint(ob.position, ob.orientation, col.localPos);
    float4 q = quatMul(ob.orientation, col.localRot);
    float3 e;
    if (col.shape == SHAPE_SPHERE) e = float3(col.params.x, col.params.x, col.params.x);
    else if (col.shape == SHAPE_CAPSULE) e = abs(quatRotate(q, float3(0.0, col.params.y, 0.0))) + col.params.x;
    else if (col.shape == SHAPE_BOX)
    {
        e = abs(quatRotate(q, float3(col.params.x, 0.0, 0.0))) + abs(quatRotate(q, float3(0.0, col.params.y, 0.0)))
            + abs(quatRotate(q, float3(0.0, 0.0, col.params.z)));
    }
    else e = float3(col.params.w, col.params.w, col.params.w);
    lo = c - e;
    hi = c + e;
}

bool softAabbOverlaps(uint s, float3 lo, float3 hi)
{
    float3 mn = SD(softAabbBase() + 2u * s).xyz, mx = SD(softAabbBase() + 2u * s + 1u).xyz;
    return all(lo <= mx) && all(hi >= mn);
}

// Per solved particle: the closest other-vehicle collision triangle it is in
// front of (or slightly behind) within the collision range.
// SOFT_NODE_LANES lanes per node: each walks every LANES-th candidate triangle and
// keeps its first best; the node's best is the lowest (score, triangle) over its
// lanes, the triangle the one-lane walk (bodies, then triangles, ascending) keeps.
static const uint SOFT_NODE_LANES = 8u;
groupshared float gsNodeScore[64];
groupshared uint gsNodeTri[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_collide_nodes(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint gi = dispatchIndex(tid);
    uint i = gi / SOFT_NODE_LANES, lane = gi % SOFT_NODE_LANES;
    bool live = i < softListCount();
    uint4 ent = live ? SU(softListBase() + i) : uint4(0, 0, 0, 0);
    uint body = ent.x;
    uint sb = ent.w & 0xFFFFu;
    uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * particleOrdinal(sb, body);
    if (live && lane == 0u) gSoftData[ns] = float4(0, 0, 0, asfloat(SOFT_NONE));
    live = live && softContactsOn() && (ent.w & ENTRY_CONTACTER) != 0u;
    BodyState b = gBodies[body];
    live = live && bodyIsDynamic(b) && !bodyIsSleeping(b);
    float3 x = b.position, x0 = b.initialPos;
    float bestScore = 1e30;
    uint bestTri = SOFT_NONE;
    float3 bestN = float3(0, 0, 0), bestW = float3(0, 0, 0);
    float bestDelta = 0.0;
    float bestGap = 0.0;
    uint nb = softBodyCount();
    // Self-collision: a node against its own body's triangles too (a torn door,
    // a folding bonnet), but not triangles it is a corner of, nor any within
    // SELF_REST_GAP of it in the undeformed body (its own panel).
    uint ordSelf = particleOrdinal(sb, body);
    float3 restSelf = SD(softRigidBase() + 2u * ordSelf + 1u).xyz;
    uint selfOrdBase = asuint(softBodyRec1(sb).w) - softBodyRec(sb).x;
    bool selfOn = !softIsRigid(sb);
    uint np = live ? softPairCount(sb) : 0u;
    for (uint pk = 0u; pk < np; pk++)
    {
        uint s2 = softPairAt(sb, pk);
        bool self = s2 == sb;
        if (self ? !selfOn : !softAabbHas(s2, x)) continue;
        uint4 r2 = softBodyRec2(s2);
        float delta = asfloat(r2.z);
        for (uint t = r2.x + lane; t < r2.x + r2.y; t += SOFT_NODE_LANES)
        {
            uint4 tri = SU(softTriBase() + t);
            if (((tri.w >> 8) & 1u) == 0u) continue;
            if (self && (tri.x == body || tri.y == body || tri.z == body)) continue;
            // soft_tri_planes: first vertex and cull radius, then normal and area
            // (the same values as from the vertices, which load only past the cull).
            float reach = delta + length(x - x0) + 0.05;
            float4 p1 = SD(softTriPlaneBase() + 2u * t + 1u);
            float3 a = p1.xyz;
            // Inside the prism at |d| ≤ max(reach, 0.3) puts the node within the
            // triangle's reach of its first vertex.
            float far = p1.w + max(reach, 0.3) + 1e-3;
            if (dot(x - a, x - a) > far * far) continue;
            float4 p0 = SD(softTriPlaneBase() + 2u * t);
            if (p0.w < 1e-8) continue; // torn or degenerate
            float3 n = p0.xyz;
            float d = dot(x - a, n);
            if (d > reach || d < -0.3) continue;
            float3 bpos = gBodies[tri.y].position, c = gBodies[tri.z].position;
            // Step-start side: the node must come from the front face.
            float3 a0 = gBodies[tri.x].initialPos, b0 = gBodies[tri.y].initialPos, c0 = gBodies[tri.z].initialPos;
            float3 n0 = cross(b0 - a0, c0 - a0);
            float l0 = length(n0);
            float d0 = l0 > 1e-8 ? dot(x0 - a0, n0 / l0) : d;
            if (d0 < -0.05) continue;
            float3 w;
            float3 q = closestOnTriangle(x - n * d, a, bpos, c, w);
            // Prism test: the node projects inside the triangle.
            if (dot(q - (x - n * d), q - (x - n * d)) > 1e-6) continue;
            if (self)
            {
                // Checked last (it costs a closest-point query): excluded only if the
                // node sat over this triangle's face, within SELF_REST_GAP, in the
                // rest body: its own panel, or the shell under a bolt-on part. A
                // neighbour beside it (a fender next to the bonnet) still collides
                // once the part comes loose.
                float3 wr;
                float3 ra = SD(softRigidBase() + 2u * (selfOrdBase + tri.x) + 1u).xyz;
                float3 rb = SD(softRigidBase() + 2u * (selfOrdBase + tri.y) + 1u).xyz;
                float3 rc = SD(softRigidBase() + 2u * (selfOrdBase + tri.z) + 1u).xyz;
                float3 rn = cross(rb - ra, rc - ra);
                float rl = length(rn);
                float dPlane = rl > 1e-8 ? abs(dot(restSelf - ra, rn / rl)) : 0.0;
                float dRest = length(closestOnTriangle(restSelf, ra, rb, rc, wr) - restSelf);
                if (dRest < SELF_REST_GAP && dRest - dPlane < 0.02) continue;
            }
            float score = abs(d);
            if (score < bestScore || (score == bestScore && t < bestTri))
            {
                bestScore = score; bestTri = t; bestN = n; bestW = w; bestDelta = delta;
                // Gap at the step start (frozen normal): the stabilised constraint
                // corrects only (1 − α) of an existing overlap per step.
                bestGap = dot(x0 - (w.x * a0 + w.y * b0 + w.z * c0), n) - delta;
            }
        }
    }
    gsNodeScore[lid.x] = bestScore;
    gsNodeTri[lid.x] = bestTri;
    GroupMemoryBarrierWithGroupSync();
    if (!live || bestTri == SOFT_NONE) return;
    uint first = lid.x - lane;
    for (uint k = 0u; k < SOFT_NODE_LANES; k++)
    {
        float sk = gsNodeScore[first + k];
        uint tk = gsNodeTri[first + k];
        if (k != lane && tk != SOFT_NONE && (sk < bestScore || (sk == bestScore && tk < bestTri))) return;
    }
    gSoftData[ns] = float4(bestN, asfloat(bestTri));
    gSoftData[ns + 1u] = float4(bestW, bestDelta);
    gSoftData[ns + 2u] = float4(bestGap, 0.0, 0.0, 0.0);
    gSoftData[softFlagBase() + particleOrdinal(sb, body)] = float4(1, 0, 0, 0);
}

// Per collision triangle, before node contacts: its unit normal and area (area 0
// when torn or degenerate) and its first vertex with the distance to the farthest
// other vertex, so soft_collide_nodes culls a triangle from two reads.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_tri_planes(uint3 tid : SV_DispatchThreadID)
{
    uint t = dispatchIndex(tid);
    if (t >= softTriCount()) return;
    uint4 tri = SU(softTriBase() + t);
    float3 a = gBodies[tri.x].position, b = gBodies[tri.y].position, c = gBodies[tri.z].position;
    float3 nr = cross(b - a, c - a);
    float area = length(nr);
    bool dead = triTorn(t, a, b, c) || area < 1e-8;
    gSoftData[softTriPlaneBase() + 2u * t] = dead ? float4(0.0, 0.0, 0.0, 0.0) : float4(nr / area, area);
    gSoftData[softTriPlaneBase() + 2u * t + 1u] = float4(a, max(length(b - a), length(c - a)));
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
        // Boxes apart by more than the contact reach on an axis cannot touch.
        float reach = delta + travel + 0.05;
        float3 olo, ohi;
        obstacleBounds(col, ob, olo, ohi);
        if (any(olo > hi + reach) || any(ohi < lo - reach)) continue;
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
    uint np = softPairCount(sb);
    for (uint pk = 0u; pk < np && found < TRI_NODE_CONTACTS; pk++)
    {
        uint s2 = softPairAt(sb, pk);
        // Own nodes too (self-collision, filtered on the node side).
        if (s2 == sb ? softIsRigid(sb) : !softAabbOverlaps(s2, lo - 0.3, hi + 0.3)) continue;
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
    return w.x * (softPos(tri.x) - softPos0(tri.x))
        + w.y * (softPos(tri.y) - softPos0(tri.y))
        + w.z * (softPos(tri.z) - softPos0(tri.z));
}

float3 triPoint(uint4 tri, float3 w)
{
    return w.x * softPos(tri.x) + w.y * softPos(tri.y) + w.z * softPos(tri.z);
}

// Soft contact terms of one particle (node side and triangle-vertex side).
// Triangle-side contact terms of a particle at corner j of triangle tt: the
// triangle's thin obstacle (kind 0) or the node in its slot m (kind m + 1).
void softTriObstacleTerm(uint4 tri, uint tc, uint j, inout float3 g, inout float3x3 H)
{
    float4 o0 = SD(tc);
    float4 o1 = SD(tc + 1u);
    float4 o2 = SD(tc + 2u);
    float C = dot(o0.xyz, triPoint(tri, o2.xyz) - o1.xyz) - o2.w;
    softContactTerm(g, H, o2[j], C, o1.w, o0.xyz, triTravel(tri, o2.xyz));
}
void softTriNodeTerm(uint4 tri, uint tc, uint j, uint m, inout float3 g, inout float3x3 H)
{
    float4 c0 = SD(tc + 3u + 3u * m);
    uint node = asuint(c0.w);
    float4 c1 = SD(tc + 4u + 3u * m);
    float3 np = softPos(node);
    float C = dot(c0.xyz, np - triPoint(tri, c1.xyz)) - c1.w;
    float3 u = (np - softPos0(node)) - triTravel(tri, c1.xyz);
    softContactTerm(g, H, -c1[j], C, SD(tc + 5u + 3u * m).x, c0.xyz, u);
}

uint softContactListBase() { return SU(9u).y; }
static const uint CONTACT_LIST_STRIDE = 4u; // engine_softbody::CONTACT_LIST_STRIDE
static const uint CONTACT_LIST_MAX = 15u;
static const uint CONTACT_LIST_ALL = 0xFFFFFFFFu;

// Per particle, after soft_collide_tris: its triangle-side contacts this step,
// (triangle << 5 | corner << 3 | kind), in the order softContactsFor's walk of
// its triangles visits them. The sweep then reads only real contacts instead of
// every slot of every triangle around it.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_contact_lists(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softListCount()) return;
    uint4 ent = SU(softListBase() + i);
    uint body = ent.x;
    uint ord = particleOrdinal(ent.w & 0xFFFFu, body);
    uint base = softContactListBase() + CONTACT_LIST_STRIDE * ord;
    uint n = 0u;
    uint4 slot = uint4(0, 0, 0, 0);
    if (softContactsOn() && SD(softFlagBase() + ord).x != 0.0)
    {
        uint adj = softAdjBase();
        uint triStart = ent.y + 2u * (ent.z & 0xFFFFu), triCount = ent.z >> 16;
        for (uint k = 0u; k < triCount && n <= CONTACT_LIST_MAX; k++)
        {
            uint wi = triStart + k;
            uint tt = SU(adj + (wi >> 2))[wi & 3u];
            uint4 tri = SU(softTriBase() + tt);
            uint tc = softTriContactBase() + TRI_CONTACT_STRIDE * tt;
            uint j = tri.x == body ? 0u : (tri.y == body ? 1u : 2u);
            for (uint kind = 0u; kind <= TRI_NODE_CONTACTS && n <= CONTACT_LIST_MAX; kind++)
            {
                uint w = kind == 0u ? asuint(SD(tc).w) : asuint(SD(tc + 3u * kind).w);
                if (w == SOFT_NONE) continue;
                n++;
                if (n > CONTACT_LIST_MAX) break;
                slot[n & 3u] = (tt << 5) | (j << 3) | kind;
                if ((n & 3u) == 3u) { gSoftData[base + (n >> 2)] = asfloat(slot); slot = uint4(0, 0, 0, 0); }
            }
        }
    }
    if ((n & 3u) != 3u && n <= CONTACT_LIST_MAX) gSoftData[base + (n >> 2)] = asfloat(slot);
    uint4 head = SU(base);
    head.x = n > CONTACT_LIST_MAX ? CONTACT_LIST_ALL : n;
    gSoftData[base] = asfloat(head);
}

void softContactsFor(uint body, uint ord, float3 x, float3 x0, uint triStart, uint triCount, uint lane, uint lanes,
    inout float3 g, inout float3x3 H)
{
    if (!softContactsOn()) return;
    // Detection flags every particle in a soft contact this step (node or
    // triangle vertex); the rest have nothing to read.
    if (SD(softFlagBase() + ord).x == 0.0) return;
    uint ns = softNodeSlotBase() + NODE_SLOT_STRIDE * ord;
    float4 s0 = SD(ns);
    uint t = asuint(s0.w);
    if (t != SOFT_NONE && lane == 0u)
    {
        float4 s1 = SD(ns + 1u);
        uint4 tri = SU(softTriBase() + t);
        float C = dot(s0.xyz, x - triPoint(tri, s1.xyz)) - s1.w;
        softContactTerm(g, H, 1.0, C, SD(ns + 2u).x, s0.xyz, (x - x0) - triTravel(tri, s1.xyz));
    }
    uint cl = softContactListBase() + CONTACT_LIST_STRIDE * ord;
    uint n = SU(cl).x;
    if (n != CONTACT_LIST_ALL)
    {
        for (uint k = lane; k < n; k += lanes)
        {
            uint w = k + 1u;
            uint code = SU(cl + (w >> 2))[w & 3u];
            uint tt = code >> 5, kind = code & 7u;
            uint4 tri = SU(softTriBase() + tt);
            uint tc = softTriContactBase() + TRI_CONTACT_STRIDE * tt;
            if (kind == 0u) softTriObstacleTerm(tri, tc, (code >> 3) & 3u, g, H);
            else softTriNodeTerm(tri, tc, (code >> 3) & 3u, kind - 1u, g, H);
        }
        return;
    }
    // Too many for the list: walk every triangle around the particle.
    uint adj = softAdjBase();
    for (uint k = lane; k < triCount; k += lanes)
    {
        uint wi = triStart + k;
        uint tt = SU(adj + (wi >> 2))[wi & 3u];
        uint4 tri = SU(softTriBase() + tt);
        uint tc = softTriContactBase() + TRI_CONTACT_STRIDE * tt;
        // This particle's weight in the triangle.
        uint j = tri.x == body ? 0u : (tri.y == body ? 1u : 2u);
        if (asuint(SD(tc).w) != SOFT_NONE) softTriObstacleTerm(tri, tc, j, g, H);
        for (uint m = 0u; m < TRI_NODE_CONTACTS; m++)
            if (asuint(SD(tc + 3u + 3u * m).w) != SOFT_NONE) softTriNodeTerm(tri, tc, j, m, g, H);
    }
}

static const uint SOFT_SKIP = 0u;
static const uint SOFT_FOLLOW = 1u;
static const uint SOFT_SOLVE = 2u;

// VBD block of one particle (colour-list entry `ent`), split over `lanes` lanes:
// this lane's share (lane 0: inertia, rigid edges and its node contact; beams and
// contact triangles strided over the lanes). The block is the sum of the shares.
// Returns SOFT_SKIP (asleep, static), SOFT_FOLLOW (rigid mode) or SOFT_SOLVE.
uint softAccumulate(uint4 ent, uint lane, uint lanes, out float3x3 H, out float3 g)
{
    H = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    g = float3(0.0, 0.0, 0.0);
    uint body = ent.x;
    BodyState b = gBodies[body];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b)) return SOFT_SKIP;
    if (softIsRigid(ent.w & 0xFFFFu)) return SOFT_FOLLOW;
    AvbdParams p = PARAMS;
    float h = max(p.dt, 1e-6);

    // Inertia, contacts and joints: the rigid accumulation, rotation frozen.
    bool edges = SOLVE_INDEX(body) != SOLVE_INDEX(body + 1u) || (b.flags & BODY_VEHICLE) != 0u;
    if (!edges && lane != 0u) {}
    else if (!edges)
    {
        // No contact or joint edge: primalAccumulate's linear block is the inertia
        // term alone (same arithmetic), without its 6×6 accumulation.
        float k = (b.invMass > 1e-12 ? (1.0 / b.invMass) : 0.0) / (p.dt * p.dt);
        if (softBdf2For(particleOrdinal(ent.w & 0xFFFFu, body))) k *= 2.25;
        float3 dp = b.position - gAabbs[body].mn;
        H = float3x3(k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, k);
        g = float3(k * dp.x, k * dp.y, k * dp.z);
    }
    else
    {
        // Its contact and joint edges strided over the particle's lanes (lane 0
        // also adds the inertia), as the rigid colour solve splits a body.
        Mat6 A;
        Vec6 rhs;
        primalAccumulate(body, lane, lanes, A, rhs);
        H = float3x3(
            A.m[0], A.m[6], A.m[12],
            A.m[6], A.m[7], A.m[13],
            A.m[12], A.m[13], A.m[14]);
        g = float3(rhs.v[0], rhs.v[1], rhs.v[2]);
        if (lane == 0u && softBdf2For(particleOrdinal(ent.w & 0xFFFFu, body)))
        {
            // BDF2 inertia weight 9/4: add the other 5/4 of primalAccumulate's term.
            float k = 1.25 * (b.invMass > 1e-12 ? (1.0 / b.invMass) : 0.0) / (p.dt * p.dt);
            H += float3x3(k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, k);
            g += k * (b.position - gAabbs[body].mn);
        }
    }

    float3 xa = b.position;
    float3 xa0 = b.initialPos;
    uint adj = softAdjBase();
    uint nBeams = ent.z & 0xFFFFu;
    bool plastic = (softFlags() & SOFT_PLASTICITY) != 0u;
    float ts = softThresholdScale();
    for (uint k = lane; k < nBeams; k += lanes)
    {
        // (beam, other end) pairs: the neighbour's position loads beside the beam.
        uint w = ent.y + 2u * k, w1 = w + 1u;
        uint e = SU(adj + (w >> 2))[w & 3u];
        uint other = SU(adj + (w1 >> 2))[w1 & 3u];
        // Fields, not the 112-byte record: WGSL copies a loaded struct whole.
        float3 oPos = softPos(other);
        float3 oPos0 = softPos0(other);
        SoftBeam sb = loadBeam(e);
        if (beamBroken(sb)) continue;
        float3 d = xa - oPos;
        float len = length(d);
        if (len < 1e-6) continue;
        float3 n = d / len;
        float2 el = beamElastic(sb, len, sb.penalty);
        float c = beamDampCoef(sb, len);
        float len0 = length(xa0 - oPos0);
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
    softContactsFor(body, particleOrdinal(ent.w & 0xFFFFu, body), xa, xa0, ent.y + 2u * nBeams, ent.z >> 16, lane, lanes, g, H);
    return SOFT_SOLVE;
}

// Apply a particle's summed block: follow the core (rigid mode) or take the
// block step.
void softFinish(uint4 ent, uint mode, float3x3 H, float3 g)
{
    uint body = ent.x;
    if (mode == SOFT_SKIP) return;
    uint sb0 = ent.w & 0xFFFFu;
    if (mode == SOFT_FOLLOW)
    {
        // Rigid mode: follow the core's current iterate, no particle solve.
        BodyState c = gBodies[asuint(softBodyRec1(sb0).z)];
        softSetPos(body, c.position + quatRotate(c.orientation, SD(softRigidBase() + 2u * particleOrdinal(sb0, body)).xyz));
        return;
    }
    AvbdParams p = PARAMS;
    float3 xa = gBodies[body].position;
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
    softSetPos(body, xa + dx);
}

// VBD block solve of one particle on one lane.
void softSolveEntry(uint4 ent)
{
    float3x3 H;
    float3 g;
    uint mode = softAccumulate(ent, 0u, 1u, H, g);
    softFinish(ent, mode, H, g);
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
    if (softIsRigid(ent.w & 0xFFFFu)) soft = true; // rigid mode: no acceleration
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

// One whole particle iteration per soft body, one 256-lane workgroup each: the
// body's share of every colour in order with a barrier between colours (the same
// Gauss–Seidel order within the body as one soft_primal dispatch per colour),
// then its Chebyshev step. Bodies couple only through contacts frozen at the step
// start, read Jacobi-style as within a colour, so bodies need not wait for each
// other: one dispatch per iteration at any vehicle count. A colour holds a few
// dozen of a car's particles, so each particle gets up to 16 lanes: its beams and
// contact triangles strided over them, the 3×3 blocks summed in workgroup memory
// (a particle's solve is a chain of dependent reads; the lanes overlap them).
// PASS_U = iteration index.
static const uint SOFT_LANES_MAX = 16u;
groupshared float4 gsSoftRedA[256]; // H00 H01 H02 H11
groupshared float4 gsSoftRedB[256]; // H12 H22 g.x g.y
groupshared float gsSoftRedC[256];  // g.z
groupshared uint gsSoftAwake;
static const uint SOFT_ROLE_FOLLOW = 0xFFFFFFFEu;
static const uint SOFT_ROLE_VEHICLE = 0xFFFFFFFFu;

groupshared float gsVehL[256];  // per vehicle lane: its wheel's suspension length
groupshared float gsVehF[256];  // its wheel's normal load (after the anti-roll bars)
groupshared uint gsVehCt[256];  // its wheel touches the ground

// Lane-summed vehicle block: every lane adds its share of H's lower triangle and g
// into lane 0's, 9 values a round through gsSoftRed (soft workgroups' buffers).
static const uint VEH_LOWER[21] = { 0, 6, 7, 12, 13, 14, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35 };
float vehBlockValue(Mat6 H, Vec6 g, uint k) { return k < 21u ? H.m[VEH_LOWER[k]] : g.v[k - 21u]; }

// vehicle_iterate for 64 vehicles, four lanes each (one wheel or spin group per
// lane): suspension per wheel, anti-roll bars and chassis rows on lane 0, spin
// groups per lane, then the lanes' blocks summed. The same rows as
// assembleVehicle in its order per lane; vehicles over four wheels take one lane.
void softVehicleLanes(uint group, uint local)
{
    uint slot = local / 4u, lane = local % 4u, first = slot * 4u;
    uint v = group * 64u + slot;
    bool act = v < vehicleCount();
    uint vr = act ? vehRec(v) : 0u;
    act = act && (VU(vr + V_D3).y & VS_ACTIVE) != 0u;
    uint4 cfg = act ? VU(vr + V_CFG) : uint4(0, 0, 0, 0);
    BodyState c = gBodies[cfg.x];
    act = act && bodyIsDynamic(c) && !bodyIsSleeping(c);
    bool split = act && cfg.z <= 4u;
    if (act && !split && lane == 0u) vehicleIterateOne(v);
    uint nW = split ? cfg.z : 0u;
    float3 dl = float3(0, 0, 0), da = float3(0, 0, 0);
    if (split) chassisDisplacement(c, dl, da);
    Mat6 H = mat6Zero();
    Vec6 g = vec6Zero();
    gsVehL[local] = 0.0;
    gsVehF[local] = 0.0;
    gsVehCt[local] = 0u;
    if (lane < nW)
    {
        if (wheelGround(wheelRec(cfg.y + lane)) != NO_INDEX) groundClear(cfg.y + lane);
        float L, Fn;
        bool ct;
        vehSuspension(lane, vr, cfg, c, dl, da, H, g, L, Fn, ct);
        gsVehL[local] = L;
        gsVehF[local] = Fn;
        gsVehCt[local] = ct ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    float Lw[4];
    float FnW[4];
    for (uint k = 0u; k < 4u; k++) { Lw[k] = gsVehL[first + k]; FnW[k] = gsVehF[first + k]; }
    if (split && lane == 0u)
    {
        vehAntiRoll<4>(vr, cfg, c, nW, Lw, FnW, H, g);
        for (uint k = 0u; k < 4u; k++) gsVehF[first + k] = FnW[k];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 0u; k < 4u; k++) FnW[k] = gsVehF[first + k];
    if (lane < nW) vehSpinGroup<4>(lane, vr, cfg, c, dl, da, true, nW, FnW, H, g);
    if (split && lane == 0u)
    {
        uint contacts = gsVehCt[first] + gsVehCt[first + 1u] + gsVehCt[first + 2u] + gsVehCt[first + 3u];
        vehChassisRows<4>(vr, cfg, c, dl, da, true, nW, FnW, contacts, H, g);
    }
    // 27 values (21 lower-triangle entries of H, 6 of g) in three rounds of 9.
    for (uint round = 0u; round < 3u; round++)
    {
        uint k0 = round * 9u;
        gsSoftRedA[local] = float4(vehBlockValue(H, g, k0), vehBlockValue(H, g, k0 + 1u), vehBlockValue(H, g, k0 + 2u), vehBlockValue(H, g, k0 + 3u));
        gsSoftRedB[local] = float4(vehBlockValue(H, g, k0 + 4u), vehBlockValue(H, g, k0 + 5u), vehBlockValue(H, g, k0 + 6u), vehBlockValue(H, g, k0 + 7u));
        gsSoftRedC[local] = vehBlockValue(H, g, k0 + 8u);
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u)
        {
            float vals[9];
            for (uint q = 0u; q < 9u; q++) vals[q] = 0.0;
            for (uint o = 0u; o < 4u; o++)
            {
                float4 a = gsSoftRedA[first + o], b = gsSoftRedB[first + o];
                vals[0] += a.x; vals[1] += a.y; vals[2] += a.z; vals[3] += a.w;
                vals[4] += b.x; vals[5] += b.y; vals[6] += b.z; vals[7] += b.w;
                vals[8] += gsSoftRedC[first + o];
            }
            for (uint q = 0u; q < 9u; q++)
            {
                uint k = k0 + q;
                if (k < 21u) H.m[VEH_LOWER[k]] = vals[q]; else g.v[k - 21u] = vals[q];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (split && lane == 0u)
    {
        writeVehicleBlock(vehicleAabbBlock(v), H, g, true);
        VSet(vr + V_REFP, float4(c.position, VD(vr + V_REFP).w));
        VSet(vr + V_REFQ, c.orientation);
    }
}

uint softLanesFor(uint n)
{
    uint q = 256u / max(n, 1u);
    return q >= 16u ? 16u : (q >= 8u ? 8u : (q >= 4u ? 4u : (q >= 2u ? 2u : 1u)));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void soft_primal_bodies(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint local = lid.x;
    gStabilize = false;
    uint nb = softBodyCount();
    bool isSoft = gid.x < nb;
    uint base = softBodyColorBase() + gid.x * softBodyColorStride();
    uint allColors = softColorCount();
    // Anything of this body to solve? (A soft body sleeps and wakes whole; a
    // sleeping particle's solve is a no-op, and soft_begin reseeds the Chebyshev
    // history every step.)
    if (local == 0u) gsSoftAwake = 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint c = 0u; c < (isSoft ? allColors : 0u); c++)
    {
        uint4 slot = SU(base + c / 2u);
        uint2 cr = (c & 1u) == 0u ? slot.xy : slot.zw;
        for (uint t = local; t < cr.y; t += 256u)
        {
            BodyState pb = gBodies[SU(softListBase() + cr.x + t).x];
            if (bodyIsDynamic(pb) && !bodyIsSleeping(pb) && !wakePending(pb)) gsSoftAwake = 1u;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Role: SOFT_ROLE_VEHICLE (workgroups past the bodies), nothing awake (0),
    // SOFT_ROLE_FOLLOW (rigid mode: particles take the core's pose once; the
    // colour order cannot matter, each writes only itself), or the colour count.
    if (local == 0u)
        gsUniformLanes = !isSoft ? SOFT_ROLE_VEHICLE
            : (gsSoftAwake == 0u ? 0u : (softIsRigid(gid.x) ? SOFT_ROLE_FOLLOW : allColors));
    GROUP_SYNC();
    uint role = gsUniformLanes;
    if (role == SOFT_ROLE_FOLLOW)
    {
        for (uint c = 0u; c < allColors; c++)
        {
            uint4 slot = SU(base + c / 2u);
            uint2 cr = (c & 1u) == 0u ? slot.xy : slot.zw;
            for (uint t = local; t < cr.y; t += 256u) softSolveEntry(SU(softListBase() + cr.x + t));
        }
    }
    uint colors = role < SOFT_ROLE_FOLLOW ? role : 0u;
    for (uint c = 0u; c < colors; c++)
    {
        uint4 slot = SU(base + c / 2u);
        uint2 cr = (c & 1u) == 0u ? slot.xy : slot.zw;
        uint lanes = softLanesFor(cr.y);
        uint t = local / lanes, lane = local % lanes;
        uint mode = SOFT_SKIP;
        uint4 ent = uint4(0, 0, 0, 0);
        if (lanes > 1u)
        {
            float3x3 H = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            float3 g = float3(0.0, 0.0, 0.0);
            if (t < cr.y)
            {
                ent = SU(softListBase() + cr.x + t);
                mode = softAccumulate(ent, lane, lanes, H, g);
            }
            gsSoftRedA[local] = float4(H[0][0], H[0][1], H[0][2], H[1][1]);
            gsSoftRedB[local] = float4(H[1][2], H[2][2], g.x, g.y);
            gsSoftRedC[local] = g.z;
        }
        GroupMemoryBarrierWithGroupSync();
        if (lanes > 1u)
        {
            if (lane == 0u && t < cr.y && mode != SOFT_SKIP)
            {
                float4 a = float4(0.0, 0.0, 0.0, 0.0), b = float4(0.0, 0.0, 0.0, 0.0);
                float gz = 0.0;
                for (uint k = 0u; k < lanes; k++)
                {
                    a += gsSoftRedA[local + k];
                    b += gsSoftRedB[local + k];
                    gz += gsSoftRedC[local + k];
                }
                float3x3 H = float3x3(a.x, a.y, a.z, a.y, a.w, b.x, a.z, b.x, b.y);
                softFinish(ent, mode, H, float3(b.z, b.w, gz));
            }
        }
        else
        {
            for (uint k = local; k < cr.y; k += 256u) softSolveEntry(SU(softListBase() + cr.x + k));
        }
        DeviceMemoryBarrierWithGroupSync();
    }
    if (softChebyshevRho() > 0.0)
    {
        for (uint c = 0u; c < colors; c++)
        {
            uint4 slot = SU(base + c / 2u);
            uint2 cr = (c & 1u) == 0u ? slot.xy : slot.zw;
            for (uint t = local; t < cr.y; t += 256u) softChebyshevEntry(cr.x + t, PASS_U);
        }
    }
    // Workgroups past the soft bodies (SOFT_ROLE_VEHICLE): vehicle_iterate, 64 vehicles each. It
    // reads only the cores' iterate (solved before this pass), and the sweep reads
    // nothing it writes, so the two run side by side instead of one after the
    // other. (A wheel standing on another car's particle reads that particle
    // mid-sweep, as cross-body contacts already do.)
    if (role == SOFT_ROLE_VEHICLE) softVehicleLanes(gid.x - nb, local);
}

// A yield or a break: both ends take a backward-Euler step next (BDF2's history
// no longer describes them). Marks the history slot (w = 2); soft_velocity reads it.
void softBdf2Interrupt(uint s, uint a, uint b)
{
    if (!softBdf2On()) return;
    uint ia = softBdfBase() + particleOrdinal(s, a), ib = softBdfBase() + particleOrdinal(s, b);
    if (SD(ia).w == 1.0) gSoftData[ia].w = 2.0;
    if (SD(ib).w == 1.0) gSoftData[ib].w = 2.0;
}

// A yield or a break keeps its body soft: restart its quiet count.
void softQuietReset(uint s)
{
    uint4 st = SU(softModeSlot(s));
    if (st.y != 0u) gSoftData[softModeSlot(s)] = asfloat(uint4(st.x, 0u, st.z, st.w));
}

// Core mass properties and core joints for a mode (soft: the core alone, joined
// to the axle nodes; rigid: the whole vehicle, joints off).
void softSetMode(uint s, uint core, uint4 st, bool rigid)
{
    float4 m = SD(softBodyBase() + 6u * s + (rigid ? 5u : 4u));
    gBodies[core].invMass = m.x;
    gBodies[core].invInertia = m.yzw;
    uint nj = (st.z >> 8) & 255u;
    for (uint j = st.w; j < st.w + nj; j++)
    {
        uint f = gJoints[j].flags;
        gJoints[j].flags = rigid ? (f | JOINT_FLAG_BROKEN) : (f & ~JOINT_FLAG_BROKEN);
    }
}

// A rigid contact edge (solver edge `id`, past the joints) that can push this
// step: its gap at the step start closed by twice this step's approach along the
// normal (C = c0N + n·(dB − dA), as the solve), or a force carried over from the
// last step. A speculative contact the body only glides past (the ground under a
// moving car's sills) cannot, and must not keep the car soft.
bool softContactEdgeLive(uint id)
{
    float h2 = 2.0 * max(PARAMS.dt, 1e-6);
#if defined(ENGINE_COMPACT_SOLVE)
    SolverPair pair = gSolverPairs[id];
    uint count = pair.countAndSphere & SOLVER_COUNT_MASK;
    uint start = pair.contactStart;
    BodyState bA = gBodies[pair.bodyA];
    BodyState bB = gBodies[pair.bodyB];
#else
    Manifold mf = gManifolds[id];
    uint count = mf.contactCount;
    uint start = mf.contactStart;
    BodyState bA = gBodies[gColliders[mf.colliderA].body];
    BodyState bB = gBodies[gColliders[mf.colliderB].body];
#endif
    for (uint ci = 0u; ci < count; ci++)
    {
#if defined(ENGINE_COMPACT_SOLVE)
        Contact c = solverContact(start + ci - HOST_S(11), pair.normal, solverPairFriction(pair));
#else
        Contact c = READ_CONTACT(start + ci);
#endif
        if (c.alive == 0u) continue;
        if (c.lambdaN < 0.0) return true;
        float3 vA = bA.linearVel + cross(bA.angularVel, quatRotate(bA.orientation, c.anchorA));
        float3 vB = bB.linearVel + cross(bB.angularVel, quatRotate(bB.orientation, c.anchorB));
        if (c.c0N + h2 * min(dot(c.normal, vB - vA), 0.0) < 0.01) return true;
    }
    return false;
}

// Rigid mode (design: "rigid mode with promotion"), once per soft body per step
// after the contact passes and before the solve. Promote (turn soft) on any
// contact of a particle (a contact edge in its constraint list) or bounds that
// near another soft body or an obstacle; demote (turn rigid) after `quiet ms`
// with neither and no yield. Both take effect this step.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_mode(uint3 tid : SV_DispatchThreadID)
{
    uint s = dispatchIndex(tid);
    if (s >= softBodyCount()) return;
    uint4 st = SU(softModeSlot(s));
    if ((st.z & 1u) == 0u) return; // no rigid core
    uint core = asuint(softBodyRec1(s).z);
    if (core == SOFT_NO_BODY) return;
    bool enabled = (softFlags() & SOFT_RIGID_MODE) != 0u;
    uint4 r = softBodyRec(s);
    bool trigger = !enabled;
    uint nJ = gCounters[0].jointCount;
    for (uint k = 0u; k < r.y && !trigger; k++)
    {
        uint b = r.x + k;
        uint hi = SOLVE_INDEX(b + 1u);
        for (uint e = SOLVE_INDEX(b); e < hi; e++)
        {
            if (SOLVE_ID(e) >= nJ && softContactEdgeLive(SOLVE_ID(e) - nJ)) { trigger = true; break; }
        }
    }
    if (!trigger && softContactsOn())
    {
        // Contacts this step form only inside the bounds soft_bounds padded by
        // twice this step's travel (+ 0.1 m), so bounds within both skins
        // (+ 0.1 m) of each other are the earliest a contact can come: the car
        // turns soft that step, before any contact force. A wider margin only
        // kept cars soft in traffic for nothing.
        float own = asfloat(softBodyRec2(s).z) + 0.1;
        float3 lo = SD(softAabbBase() + 2u * s).xyz - own, hi = SD(softAabbBase() + 2u * s + 1u).xyz + own;
        uint nb = softBodyCount();
        uint np = softPairCount(s);
        for (uint pk = 0u; pk < np && !trigger; pk++)
        {
            uint s2 = softPairAt(s, pk);
            if (s2 == s) continue;
            float m2 = asfloat(softBodyRec2(s2).z);
            float3 lo2 = SD(softAabbBase() + 2u * s2).xyz - m2, hi2 = SD(softAabbBase() + 2u * s2 + 1u).xyz + m2;
            if (all(lo <= hi2) && all(lo2 <= hi)) trigger = true;
        }
        uint nObs = softObstacleCount();
        float h2 = 2.0 * max(PARAMS.dt, 1e-6);
        for (uint k = 0u; k < nObs && !trigger; k++)
        {
            Collider col = gColliders[SU(softObstacleBase() + k).x];
            BodyState ob = gBodies[col.body];
            float3 olo, ohi;
            obstacleBounds(col, ob, olo, ohi);
            // A moving obstacle (a ram, a crusher): its own travel this step too.
            float mo = h2 * (length(ob.linearVel) + length(ob.angularVel) * length(ohi - olo) * 0.5);
            if (all(ohi + mo >= lo) && all(olo - mo <= hi)) trigger = true;
        }
    }
    float h = max(PARAMS.dt, 1e-6);
    uint quiet = trigger ? 0u : min(st.y + 1u, 1u << 24);
    uint mode = st.x;
    if (mode == 1u && trigger)
    {
        mode = 0u;
        softSetMode(s, core, st, false);
    }
    else if (mode == 0u && enabled && float(quiet) * h * 1000.0 >= float(st.z >> 16))
    {
        mode = 1u;
        softSetMode(s, core, st, true);
    }
    gSoftData[softModeSlot(s)] = asfloat(uint4(mode, quiet, st.z, st.w));
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
    if (softIsRigid(beamSoftBody(e))) return; // rigid mode: the shape does not change
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
            softQuietReset(beamSoftBody(e));
            softBdf2Interrupt(beamSoftBody(e), e.a, e.b);
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
            softQuietReset(beamSoftBody(e));
            softBdf2Interrupt(beamSoftBody(e), e.a, e.b);
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
