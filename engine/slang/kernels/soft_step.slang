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

static const uint EVENT_BEAM_BREAK = 5u;
static const uint SOFT_NO_BODY = 0xFFFFFFFFu;

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
uint softChebBase() { return SU(5u).x; }

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
// Soft body records: [firstBody, particles, firstBeam, beams], [steer, group, core, 0]
// ---------------------------------------------------------------------------
uint4 softBodyRec(uint s) { return SU(softBodyBase() + 2u * s); }
float4 softBodyRec1(uint s) { return SD(softBodyBase() + 2u * s + 1u); }

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
    if (i < softListCount())
    {
        // Particle initial guess (VBD adaptive initialisation): x⁻ + h v + h² g ã.
        // The rigid predictor's at-rest rule holds slow bodies still, which a
        // Gauss–Seidel sweep over stiff beams cannot undo in a few iterations
        // (a lattice would "fall" at a few cm/s). Free particles take the full
        // inertial step (ã = 1); particles with contact or joint edges use the
        // acceleration they had along gravity last step.
        uint body = SU(softListBase() + i).x;
        uint q = softChebBase() + 3u * i;
        BodyState b = gBodies[body];
        if (bodyIsDynamic(b) && !bodyIsSleeping(b) && !wakePending(b))
        {
            AvbdParams p = PARAMS;
            float h = p.dt;
            float3 gv = float3(0.0, p.gravityY, 0.0);
            float weight = 1.0;
            if (SOLVE_INDEX(body + 1u) != SOLVE_INDEX(body))
            {
                float3 vPrev = SD(q + 2u).xyz;
                float3 a = (b.linearVel - vPrev) / max(h, 1e-6);
                weight = clamp(dot(a, gv) / max(dot(gv, gv), 1e-6), 0.0, 1.0);
            }
            float3 x = b.initialPos + b.linearVel * h + gv * (h * h * weight);
            gBodies[body].position = x;
            gSoftData[q] = float4(x, 0.0);
            gSoftData[q + 1u] = float4(x, 0.0);
            gSoftData[q + 2u] = float4(b.linearVel, 0.0);
        }
    }
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
    uint4 ent = SU(softListBase() + cr.x + t);
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
    for (uint k = 0u; k < ent.z; k++)
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
        float lat = max(el.x / len, 0.0);
        g += n * d1;
        float3x3 nn = float3x3(n * n.x, n * n.y, n * n.z);
        H += nn * (d2 - lat);
        H[0][0] += lat; H[1][1] += lat; H[2][2] += lat;
    }

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

// Chebyshev semi-iterative acceleration of the particle sweep (VBD §3.6),
// PASS_U = iteration index. Particles with any contact or joint edge this step
// are left alone.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_chebyshev(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    float rho = softChebyshevRho();
    if (i >= softListCount() || rho <= 0.0) return;
    uint it = PASS_U;
    uint body = SU(softListBase() + i).x;
    uint q = softChebBase() + 3u * i;
    float3 prev2 = SD(q).xyz;
    float3 prev1 = SD(q + 1u).xyz;
    BodyState b = gBodies[body];
    float3 x = b.position;
    if (it > 0u && bodyIsDynamic(b) && !bodyIsSleeping(b) && !wakePending(b)
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
    e.force = force;
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
            float old = e.rest;
            float deform = delta + dc * (1.0 - e.coef) / k;
            e.rest = max(e.rest + deform, softMinLength());
            if (e.shortB > 0.0) e.rest = max(e.rest, e.rest0 * e.shortB);
            e.deformC *= old / e.rest;
            e.meta |= BEAM_YIELDED;
        }
        else if (-force > dt && delta > 0.0)
        {
            float old = e.rest;
            float deform = delta - dt * (1.0 - e.coef) / k;
            e.rest += deform;
            if (e.longB > 0.0) e.rest = min(e.rest, e.rest0 * e.longB);
            e.deformT *= e.rest / old;
            e.strength -= deform * k;
            e.meta |= BEAM_YIELDED;
        }
    }
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
