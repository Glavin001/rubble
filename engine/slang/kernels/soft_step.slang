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
// Normal, with its own spring and damping while longer than rest (beamExtraBase).
static const uint BEAM_ANISOTROPIC = 5u;

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
// The node collides with its own body's triangles (engine_softbody::ENTRY_SELF).
static const uint ENTRY_SELF = 131072u;
// Rest slot .w: the node's self-collision group, with this bit when its body has
// groups (engine_softbody::NODE_GROUPED).
static const uint NODE_GROUPED = 0x80000000u;
// Node contacts a triangle records (the triangle side of node–triangle contacts).
static const uint TRI_NODE_CONTACTS = 4u;
// Obstacle (3 slots), node contacts (3 each), then the nearest node's distance in
// front of the face this step (the displacement bound of its vertices).
static const uint TRI_CONTACT_STRIDE = 4u + 3u * TRI_NODE_CONTACTS;
static const uint TRI_BOUND_SLOT = 3u + 3u * TRI_NODE_CONTACTS;
// Node contacts per node (its side-cached triangles), 3 slots each.
static const uint NODE_CONTACTS = 4u;
static const uint NODE_SLOT_STRIDE = 3u * NODE_CONTACTS;
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
uint softTriExtraBase() { return SU(9u).w; }
uint softPairCacheBase() { return SU(10u).x; }
uint softLocalBase() { return SU(10u).y; }
uint softOriginBase() { return SU(10u).z; }
// Per beam: an anisotropic beam's expansion spring and damping.
uint softBeamExtraBase() { return SU(11u).y; }
uint softElementCount() { return SU(11u).z; }
uint softElementBase() { return SU(11u).w; }
uint softRailNodeBase() { return SU(12u).x; }
uint softIslandBase() { return SU(12u).y; }
uint softCrashSubsteps() { return SU(12u).z; }
uint softSweeps() { return max(SU(12u).w, 1u); }
// Substeps soft body s takes this step (crash islands; 1 otherwise) and its
// substep length.
uint softSubsteps(uint s) { return max(SU(softIslandBase() + s).x, 1u); }
float softH(uint s) { return max(PARAMS.dt, 1e-6) / float(softSubsteps(s)); }
// Least speed (m/s) at which a soft contact pushes out an overlap it started the
// step with (SoftSettings::contact_push_out).
float softPushOut() { return SD(10u).w; }
// Displacement bound relaxation γ (SoftSettings::displacement_bound; 0 = off).
float softBoundGamma() { return SD(11u).x; }
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
bool beamIdBroken(uint e)
{
    return e != SOFT_NONE && (SU(softBeamBase() + 4u * e + 2u).w & BEAM_BROKEN) != 0u;
}
// A triangle stops colliding once torn, once a beam along one of its edges has
// broken, or once its break group has (BeamNG's rule).
bool triDead(uint t, float3 a, float3 b, float3 c)
{
    uint4 e = SU(softTriExtraBase() + t);
    return beamIdBroken(e.x) || beamIdBroken(e.y) || beamIdBroken(e.z) || beamIdBroken(e.w)
        || triTorn(t, a, b, c);
}
// Contact thickness of triangle t: its body's δ plus the triangle's bias
// (BeamNG externalCollisionBias); 0 for an anti-clip-only triangle.
float triThickness(uint t, uint4 tri)
{
    if (((tri.w >> 8) & 1u) == 0u) return 0.0;
    return asfloat(softBodyRec2(tri.w & 255u).z) + SD(softTriRestBase() + t).w;
}
bool softContactsOn() { return (softFlags() & SOFT_CONTACTS) != 0u; }

// Particle position reads and writes of the particle solve.
float3 softPos(uint b) { return gBodies[b].position; }
float3 softPos0(uint b) { return gBodies[b].initialPos; }
void softSetPos(uint b, float3 x) { gBodies[b].position = x; }

// Vehicle-local frames (f32 precision far from the origin). Each soft body has an
// origin on a 1 m grid; its particles' positions relative to it are the solver's
// state: beams, inertia, Chebyshev and BDF2 history, plasticity and velocity are
// all computed from them, so sub-millimetre corrections survive at any distance.
// gBodies keeps the world position (origin + local) for the rigid pipeline and
// the contacts between bodies, which are geometric and tolerate its rounding.
// Per particle ordinal: [local], [local at the step start], [inertial target].
float3 softOrigin(uint s) { return SD(softOriginBase() + s).xyz; }
static const uint LOCAL_STRIDE = 4u;
float3 softLocal(uint o) { return SD(softLocalBase() + LOCAL_STRIDE * o).xyz; }
float3 softLocal0(uint o) { return SD(softLocalBase() + LOCAL_STRIDE * o + 1u).xyz; }
float3 softTargetLocal(uint o) { return SD(softLocalBase() + LOCAL_STRIDE * o + 2u).xyz; }
// Where contacts were detected this step (the initial guess, local).
float3 softDetectLocal(uint o) { return SD(softLocalBase() + LOCAL_STRIDE * o + 3u).xyz; }
// Writes a particle's local position and its world copy.
void softSetLocal(uint body, uint o, uint s, float3 xl)
{
    gSoftData[softLocalBase() + LOCAL_STRIDE * o] = float4(xl, 0.0);
    gBodies[body].position = softOrigin(s) + xl;
}
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

bool beamPlastic(uint kind) { return kind == BEAM_NORMAL || kind == BEAM_ANISOTROPIC; }
// An anisotropic beam's expansion spring and damping (others: their own).
float2 beamExpansion(SoftBeam e, uint i)
{
    return beamKind(e) == BEAM_ANISOTROPIC ? SD(softBeamExtraBase() + i).xy : float2(e.spring, e.damp);
}

// φ'(ℓ), φ''(ℓ) of the elastic energy with stiffness k (engine-softbody model.rs).
// `i` is the beam's index (for an anisotropic beam's expansion rates).
float2 beamElastic(SoftBeam e, uint i, float len, float k)
{
    float delta = len - e.rest;
    uint kind = beamKind(e);
    if (kind == BEAM_ANISOTROPIC && delta > 0.0 && e.spring > 0.0)
    {
        float ke = k * beamExpansion(e, i).x / e.spring;
        return float2(ke * delta, ke);
    }
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

float beamDampCoef(SoftBeam e, uint i, float len)
{
    float delta = len - e.rest;
    uint kind = beamKind(e);
    bool slack = (kind == BEAM_SUPPORT && delta > 0.0) || (kind == BEAM_ROPE && delta < 0.0);
    float c = kind == BEAM_ANISOTROPIC && delta > 0.0 ? beamExpansion(e, i).y : e.damp;
    return c * softDampScale() * (slack ? 0.1 : 1.0);
}

// Signed axial force, + = compression (Rigs of Rods `slen`), material stiffness.
float beamReadout(SoftBeam e, uint i, float len, float len0, float h)
{
    float spring = beamHard(e) ? (e.penalty * (len - e.rest) + e.lambda) : beamElastic(e, i, len, e.spring).x;
    return -(spring + beamDampCoef(e, i, len) * (len - len0) / h);
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

groupshared float4 gsShift;

// Start of the step, per soft body: resynchronise a particle whose world
// position was written from outside (a spawn, a teleport) with its local one,
// then move the origin (by whole metres) when the body has wandered more than
// 32 m from it, shifting every local position and the BDF2 history to match.
// Every lane of the workgroup calls it (its barriers need uniform control flow);
// `live` = s is a soft body.
void softRecentre(uint s, uint local, bool live)
{
    uint4 r = live ? softBodyRec(s) : uint4(0, 0, 0, 0);
    uint ob = live ? asuint(softBodyRec1(s).w) : 0u;
    float3 o = live ? softOrigin(s) : float3(0.0, 0.0, 0.0);
    for (uint k = local; k < r.y; k += 64u)
    {
        float3 xw = gBodies[r.x + k].position;
        float3 xl = softLocal(ob + k);
        if (any(abs(o + xl - xw) > 0.01))
        {
            gSoftData[softLocalBase() + LOCAL_STRIDE * (ob + k)] = float4(xw - o, 0.0);
            gSoftData[softBdfBase() + ob + k] = float4(0.0, 0.0, 0.0, 0.0);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0u)
    {
        float3 x0 = r.y > 0u ? softLocal(ob) : float3(0.0, 0.0, 0.0);
        float3 shift = any(abs(x0) > 32.0) ? round(x0) : float3(0.0, 0.0, 0.0);
        gsShift = float4(shift, 0.0);
        if (live && any(shift != 0.0)) gSoftData[softOriginBase() + s] = float4(o + shift, 0.0);
    }
    GroupMemoryBarrierWithGroupSync();
    float3 shift = gsShift.xyz;
    uint n = any(shift != 0.0) ? r.y : 0u;
    for (uint k = local; k < n; k += 64u)
    {
        uint q = softLocalBase() + LOCAL_STRIDE * (ob + k);
        gSoftData[q] = float4(SD(q).xyz - shift, 0.0);
        float4 hist = SD(softBdfBase() + ob + k);
        if (hist.w != 0.0) gSoftData[softBdfBase() + ob + k] = float4(hist.xyz - shift, hist.w);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_wake(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint s = softGroupBody(gid), local = lid.x;
    softRecentre(s, local, s < softBodyCount());
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
            float3 o = softOrigin(sb);
            float3 xl = x - o;
            // Start and inertial target too: the body may turn soft within this step
            // (soft_mode), and then solves from them.
            float3 x0l = softLocal(ord);
            float h = PARAMS.dt;
            gSoftData[softLocalBase() + LOCAL_STRIDE * ord + 1u] = float4(x0l, 0.0);
            gSoftData[softLocalBase() + LOCAL_STRIDE * ord + 2u] = float4(
                x0l + b.linearVel * h + float3(0.0, PARAMS.gravityY, 0.0) * (h * h), 0.0);
            gSoftData[softLocalBase() + LOCAL_STRIDE * ord + 3u] = float4(xl, 0.0);
            softSetLocal(body, ord, sb, xl);
            gSoftData[q] = float4(xl, 0.0);
            gSoftData[q + 1u] = float4(xl, 0.0);
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
            // In the body's local frame: x⁻ is where the last step left the particle.
            float3 o = softOrigin(sb);
            float3 x0 = softLocal(ord);
            float3 x = x0 + b.linearVel * h + gv * (h * h * weight);
            // Backward Euler's inertial target (predict_bodies's, locally).
            float3 y = x0 + b.linearVel * h + gv * (h * h);
            if (softBdf2For(ord))
            {
                // BDF2 inertial target: (4/3)xₙ − (1/3)xₙ₋₁ + (2h/3)((4/3)vₙ − (1/3)vₙ₋₁)
                // + (4/9)h²g, weighted (9/4)·m/h² in the solve.
                float3 xp = SD(softBdfBase() + ord).xyz;
                y = (4.0 / 3.0) * x0 - (1.0 / 3.0) * xp
                    + (2.0 * h / 3.0) * ((4.0 / 3.0) * b.linearVel - (1.0 / 3.0) * vPrev)
                    + gv * (4.0 / 9.0 * h * h);
            }
            // The world copies the rigid pipeline reads (primalAccumulate's target).
            gAabbs[body].mn = o + y;
            gBodies[body].initialPos = o + x0;
            uint lq = softLocalBase() + LOCAL_STRIDE * ord;
            gSoftData[lq + 1u] = float4(x0, 0.0);
            gSoftData[lq + 2u] = float4(y, 0.0);
            gSoftData[lq + 3u] = float4(x, 0.0);
            softSetLocal(body, ord, sb, x);
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
    bool moving = bodyIsDynamic(b) && !bodyIsSleeping(b) && !wakePending(b);
    // The last substep's length for a crash island.
    float h = softH(ent.w & 0xFFFFu);
    float3 xl = softLocal(ord), xl0 = softLocal0(ord);
    // The velocity from the local positions: extraction's (x − x⁻)/h from the
    // world copies loses precision far from the origin. Not for a particle that
    // started this step asleep (its local x⁻ was not refreshed).
    if (moving && !softIsRigid(ent.w & 0xFFFFu)) gBodies[body].linearVel = (xl - xl0) / h;
    if (!softBdf2On() || !moving)
    {
        gSoftData[slot] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    float4 xp = SD(slot);
    // This step was BDF2 (w = 1, or 2: a beam of it yielded or broke since).
    if (xp.w >= 1.0)
        gBodies[body].linearVel = (3.0 * xl - 4.0 * xl0 + xp.xyz) / (2.0 * h);
    // Next step BDF2 only if nothing interrupted it: no yield or break of its
    // beams, no contact (an impact's velocity jump would be extrapolated).
    bool contact = SOLVE_INDEX(body + 1u) != SOLVE_INDEX(body)
        || (softContactsOn() && SD(softFlagBase() + ord).x != 0.0);
    bool smooth = xp.w != 2.0 && !contact;
    gSoftData[slot] = float4(xl0, smooth ? 1.0 : 0.0);
}

// ---------------------------------------------------------------------------
// Soft contacts: node–triangle (between soft bodies) and triangle–obstacle
// (spheres, capsules and boxes registered as obstacles: poles and bollards that
// would pass between nodes, rams whose edges would slip between them). Detected
// once per step at the initial guess, normals and barycentrics frozen for the
// step, the gap re-evaluated every iteration as a penalty ½k·min(C, 0)² with
// lagged Coulomb friction. Each particle reads only its own slots: a node its
// node slot, a triangle vertex the slots of its adjacent triangles.
#include "soft_step_closest.slang"
#include "soft_step_contacts.slang"
#include "soft_step_primal.slang"
