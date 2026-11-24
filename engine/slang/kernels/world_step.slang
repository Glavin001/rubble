// Production phase entry points — one Slang source for Metal / SPIR-V / CUDA.
#include "../physics/common.slang"
#include "../physics/atomics.slang"
#include "../physics/box_collide.slang"
#include "../physics/joints.slang"

BIND(0) RWStructuredBuffer<BodyState> gBodies;
#if !defined(ENGINE_WGSL)
RWStructuredBuffer<BodyState> gBodiesPrev;
#endif
BIND(2) RWStructuredBuffer<Collider> gColliders;
BIND(3) RWStructuredBuffer<Material> gMaterials;
#if !defined(ENGINE_PERSISTENT_CONTACTS)
BIND(4) RWStructuredBuffer<Contact> gContacts;
#endif
BIND(5) RWStructuredBuffer<Joint> gJoints;
#if !defined(ENGINE_WGSL)
RWStructuredBuffer<AvbdParams> gParams;
#endif
BIND(7) RWStructuredBuffer<GpuCounters> gCounters;
BIND(8) RWStructuredBuffer<GpuCommand> gCommands;
BIND(9) RWStructuredBuffer<Aabb> gAabbs;
BIND(10) RWStructuredBuffer<CandidatePair> gCandidates;
BIND(11) RWStructuredBuffer<uint> gAwakeList;
BIND(12) RWStructuredBuffer<AUint> gBodyColor;
#if !defined(ENGINE_WGSL)
RWStructuredBuffer<AUint> gConstraintIndex;
#endif // CSR: body -> contact/joint indices
BIND(14) RWStructuredBuffer<AUint> gConstraintIds;
BIND(15) RWStructuredBuffer<uint> gColorLists;      // packed colors + optional history/compacted-history rows
BIND(16) RWStructuredBuffer<uint> gColorOffsets;    // stable packing: per-color tile counts/exclusive starts
BIND(17) RWStructuredBuffer<AUint> gColorCounts;
BIND(18) RWStructuredBuffer<uint4> gHulls;           // packed convex hull arena (ABI v2)
BIND(19) RWStructuredBuffer<AUint> gEvents;          // ring: type, id, ...
BIND(20) RWStructuredBuffer<uint> gCcdFlags;        // compact recolor list, then per-body sleep/CCD flags
// Binding order must match engine-metal `bind_all` (declaration order → Metal [[buffer(N)]]).
BIND(21) RWStructuredBuffer<AUint> gScratchAtomics;
#if !defined(ENGINE_WGSL)
RWStructuredBuffer<uint> gPassUniform;
#endif     // length-1 view: host binds offset into 0..63 table
BIND(23) RWStructuredBuffer<uint> gIndirectArgs;    // Metal MTLDispatchThreadgroupsIndirectArguments × maxColors
// Per-collider world OBB (filled in build_aabbs) — avoids rebuild in BP face-reject + NP.
struct WorldObbGpu
{
    float4 center;
    float4 half;
    float4 axis0;
    float4 axis1;
    float4 axis2;
};
BIND(24) RWStructuredBuffer<WorldObbGpu> gWorldObbs;
// Appended (buffer 25+): do not insert earlier — Metal bind_all is declaration order.
BIND(25) RWStructuredBuffer<Manifold> gManifolds;
// Per-body reuse reference pose (three-avbd refsWGSL). pos.w unused; rot = (x,y,z,w).
struct RefPoseGpu { float4 pos; float4 rot; };
BIND(26) RWStructuredBuffer<RefPoseGpu> gRefPoses;

#if defined(ENGINE_WGSL)
// Chrome caps storage buffers at 10 per stage: read-only host data rides in uniforms,
// prev bodies live in gBodies[maxBodies..], CSR index in gConstraintIds[HOST_S(24)..].
struct HostUniforms { AvbdParams params; uint4 s[8]; };
struct PassUniform { uint v; uint _p0; uint _p1; uint _p2; };
BIND(6) ConstantBuffer<HostUniforms> gHost;
BIND(22) ConstantBuffer<PassUniform> gPass;
#define PARAMS (gHost.params)
#define HOST_S(k) (gHost.s[(k) >> 2][(k) & 3])
#define PASS_U (gPass.v)
#define BODY_PREV(i) gBodies[HOST_S(1) + (i)]
#define CINDEX(i) gConstraintIds[HOST_S(24) + (i)]
// The solve follows topology construction in a separate dispatch. Its immutable
// topology views use ordinary loads; each entry retains only one view per binding.
// RW matches the existing host layout, but solve never writes these resources.
BIND(14) RWStructuredBuffer<uint> gSolveConstraintIds;
BIND(17) RWStructuredBuffer<uint> gSolveColorCounts;
#define SOLVE_INDEX(i) gSolveConstraintIds[HOST_S(24) + (i)]
#define SOLVE_ID(i) gSolveConstraintIds[i]
#define SOLVE_COUNT(i) gSolveColorCounts[i]
#else
#define PARAMS (gParams[0])
#define HOST_S(k) (gScratchAtomics[k])
#define PASS_U (gPassUniform[0])
#define BODY_PREV(i) gBodiesPrev[i]
#define CINDEX(i) gConstraintIndex[i]
#define SOLVE_INDEX(i) gConstraintIndex[i]
#define SOLVE_ID(i) gConstraintIds[i]
#define SOLVE_COUNT(i) gColorCounts[i]
#endif


#include "../physics/contact_storage.slang"
// The sleep archive's rows (archiveMeta): canonical contacts, and persistent ones on
// WGSL hosts (the only hosts that enable it; Metal and CUDA keep their codegen).
#if !defined(ENGINE_PERSISTENT_CONTACTS) || defined(ENGINE_WGSL)
#define ENGINE_SLEEP_ARCHIVE 1
#endif
#include "../physics/convex.slang"
#include "../physics/heightfield.slang"

// Round shapes roll: their contact point moves over the surface, so the solver
// takes their lever arms at the step-start rotation (spheres, and capsules
// rolling about their axis).
bool shapeIsRound(uint shape) { return shape == SHAPE_SPHERE || shape == SHAPE_CAPSULE; }

// WebGPU limits each dispatch dimension to 65535 workgroups. Large workloads
// use rows of 32768 groups (2^21 threads); never clamp away contacts/hash slots.
// Metal's direct 1-D dispatches retain their original indices (tid.y == 0).
uint dispatchIndex(uint3 tid) { return tid.x + tid.y * (32768u * 64u); }


static const uint EVENT_SLEEP = 1u;
static const uint EVENT_WAKE = 2u;
static const uint EVENT_BREAK = 3u;
static const uint EVENT_OVERFLOW = 4u;

// Event ring in gScratchAtomics after its 96 host/list words (engine-schema
// EVENT_RING_*; gEvents belongs to the joint-pair hash, which grows with the
// joint count): a monotonically increasing write cursor, then 4-word records
// [type, id, a, b]. Hosts read the records since their last read; never cleared.
static const uint EVENT_RING_BASE = 96u;
static const uint EVENT_RING_CAPACITY = 1024u;

void emitEvent(uint type, uint id, uint a, uint b)
{
    uint n;
    AADD3(gScratchAtomics[EVENT_RING_BASE], 1u, n);
    uint r = EVENT_RING_BASE + 4u + 4u * (n % EVENT_RING_CAPACITY);
    ASTORE(gScratchAtomics[r], type);
    ASTORE(gScratchAtomics[r + 1u], id);
    ASTORE(gScratchAtomics[r + 2u], a);
    ASTORE(gScratchAtomics[r + 3u], b);
}

bool bodyIsDynamic(BodyState b) { return b.invMass > 0.0 && (b.flags & BODY_DYNAMIC) != 0u; }
bool bodyIsSleeping(BodyState b) { return (b.flags & BODY_SLEEPING) != 0u; }
// Kinematic: moved by its given velocity (predict), never by the solver. It
// counts as moving for contacts, bounds, the broadphase and wakes, while its
// target is moving.
bool bodyIsKinematic(BodyState b) { return (b.flags & (BODY_KINEMATIC | BODY_DYNAMIC)) == BODY_KINEMATIC; }
bool kinematicMoving(BodyState b)
{
    return bodyIsKinematic(b) && !bodyIsSleeping(b) &&
        (dot(b.linearVel, b.linearVel) > 0.0 || dot(b.angularVel, b.angularVel) > 0.0);
}
// Continuous collision (ccd_step.slang): an awake dynamic body flagged BODY_CCD.
// BODY_CCD_FAST: continuous collision for the next step only if it moves far enough
// this step to pass through something (CCD_FAST_REACH contact offsets).
uint ccdFastLatch(uint flags, float3 v, float dt)
{
    bool fast = length(v) * dt > CCD_FAST_REACH * PARAMS.contactOffset;
    return fast ? flags | BODY_CCD : flags & ~BODY_CCD;
}

bool bodyIsCcd(BodyState b)
{
    return (b.flags & BODY_CCD) != 0u && bodyIsDynamic(b) && !bodyIsSleeping(b) && (b.flags & BODY_VEHICLE) == 0u;
}
// HOST_S(30) bit 2: some body carries BODY_CCD (the CCD passes are scheduled).
// Bit 3: some body is kinematic.
bool sceneHasCcd() { return (HOST_S(30) & 4u) != 0u; }
bool sceneHasKinematic() { return (HOST_S(30) & 8u) != 0u; }

// World-frame inverse inertia times v: R diag(invI) R^T v.
float3 invInertiaWorld(BodyState b, float3 v)
{
    float4 q = b.orientation;
    float3 local = quatRotate(float4(-q.xyz, q.w), v);
    return quatRotate(q, b.invInertia * local);
}

// The host's `wake`: dynamic and kinematic bodies, sleep timer reset.
void wakeBody(inout BodyState b)
{
    if (b.invMass > 0.0 || bodyIsKinematic(b))
    {
        b.flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        b.sleepTimer = 0.0;
    }
}

// A run of `n` commands on one body, in submission order, stored once into both
// rows (as `write_bodies` does): the same state a host read-modify-write gives.
void applyCommandRun(uint first, uint n, uint body, float dt)
{
    BodyState b = gBodies[body];
    for (uint k = 0u; k < n; k++)
    {
        GpuCommand c = gCommands[first + k];
        // Box3D's `wake = false`: a sleeping body ignores the command.
        if ((c.b & CMD_NO_WAKE) != 0u && (b.flags & BODY_SLEEPING) != 0u) continue;
        if (c.op == CMD_WAKE_BODY)
        {
            wakeBody(b);
        }
        else if (c.op == CMD_TELEPORT)
        {
            b.position = c.f0.xyz;
            b.orientation = c.f1;
            wakeBody(b);
        }
        else if (c.op == CMD_SET_VELOCITY)
        {
            if ((c.b & CMD_ANGULAR_ONLY) == 0u) b.linearVel = c.f0.xyz;
            if ((c.b & CMD_LINEAR_ONLY) == 0u) b.angularVel = c.f1.xyz;
            wakeBody(b);
        }
        else if (c.op == CMD_APPLY_IMPULSE && b.invMass > 0.0)
        {
            float3 j = c.f0.xyz;
            float3 r = (c.b & CMD_AT_CENTER) != 0u ? float3(0.0, 0.0, 0.0) : c.f1.xyz - b.position;
            b.linearVel = b.linearVel + j * b.invMass;
            b.angularVel = b.angularVel + invInertiaWorld(b, cross(r, j));
            wakeBody(b);
        }
        else if (c.op == CMD_ADD_FORCE && b.invMass > 0.0)
        {
            b.linearVel = b.linearVel + c.f0.xyz * (b.invMass * dt);
            b.angularVel = b.angularVel + invInertiaWorld(b, c.f1.xyz * dt);
            wakeBody(b);
        }
        else if (c.op == CMD_ADD_FORCE_AT && b.invMass > 0.0)
        {
            float3 f = c.f0.xyz;
            b.linearVel = b.linearVel + f * (b.invMass * dt);
            b.angularVel = b.angularVel + invInertiaWorld(b, cross(c.f1.xyz - b.position, f) * dt);
            wakeBody(b);
        }
        else if (c.op == CMD_ANGULAR_IMPULSE && b.invMass > 0.0)
        {
            b.angularVel = b.angularVel + invInertiaWorld(b, c.f0.xyz);
            wakeBody(b);
        }
        else if (c.op == CMD_SET_KINEMATIC_TARGET && bodyIsKinematic(b))
        {
            // The velocity that reaches the target pose over this step, from
            // where the body is now (the inverse of prediction's integration).
            float h = max(dt, 1e-8);
            b.linearVel = (c.f0.xyz - b.position) / h;
            float4 q1 = c.f1;
            if (dot(b.orientation, q1) < 0.0) q1 = -q1;
            float4 dq = quatMul(q1, quatConjugate(b.orientation));
            float3 w = 2.0 * dq.xyz / h;
            b.angularVel = dq.w < 0.0 ? -w : w;
            wakeBody(b);
        }
        else if (c.op == CMD_DAMP && b.invMass > 0.0)
        {
            b.linearVel = b.linearVel * c.f0.w + c.f0.xyz;
            b.angularVel = b.angularVel * c.f1.x;
        }
        else if (c.op == CMD_SET_LOCKS)
        {
            b.flags = (b.flags & ~BODY_LOCK_MASK) | ((asuint(c.f0.x) << BODY_LOCK_SHIFT) & BODY_LOCK_MASK);
        }
        else if (c.op == CMD_SET_MASS_FRAME)
        {
            // The body origin stays put: the solver pose moves to the new
            // centre and principal axes; the centre's velocity picks up w x d.
            float3 d = quatRotate(b.orientation, c.f0.xyz);
            b.position = b.position + d;
            b.linearVel = b.linearVel + cross(b.angularVel, d);
            b.orientation = quatNormalize(quatMul(b.orientation, c.f1));
            b.invInertia = c.f2.xyz;
            b.invMass = c.f2.w;
            b.initialPos = b.position;
            b.initialRot = b.orientation;
        }
    }
    gBodies[body] = b;
    BODY_PREV(body) = b;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_commands(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    AvbdParams p = PARAMS;
    // Commands are densely packed; the host supplies their count separately.
    uint cmdCount = HOST_S(0);
    if (i >= cmdCount) return;
    GpuCommand cmd = gCommands[i];
    if (cmd.c == CMD_RUN_MEMBER) return;
    if (cmd.c > 0u)
    {
        applyCommandRun(i, min(cmd.c, cmdCount - i), cmd.a, p.dt);
        return;
    }
    if (cmd.op == CMD_WAKE_BODY)
    {
        BodyState b = gBodies[cmd.a];
        b.flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        b.sleepTimer = 0.0;
        gBodies[cmd.a] = b;
        AADD2(gCounters[0].wakeEvents, 1);
    }
    else if (cmd.op == CMD_TELEPORT)
    {
        BodyState b = gBodies[cmd.a];
        b.position = cmd.f0.xyz;
        b.orientation = cmd.f1;
        b.flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        b.sleepTimer = 0.0;
        gBodies[cmd.a] = b;
    }
    else if (cmd.op == CMD_SET_VELOCITY)
    {
        BodyState b = gBodies[cmd.a];
        b.linearVel = cmd.f0.xyz;
        b.angularVel = cmd.f1.xyz;
        b.flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        gBodies[cmd.a] = b;
    }
    else if (cmd.op == CMD_REPARENT)
    {
        Collider col = gColliders[cmd.a];
        uint oldBody = col.body;
        col.body = cmd.b;
        col.localPos = cmd.f0.xyz;
        col.localRot = cmd.f1;
        gColliders[cmd.a] = col;
        BodyState a = gBodies[oldBody];
        a.flags = (a.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        gBodies[oldBody] = a;
        BodyState nb = gBodies[cmd.b];
        nb.flags = (nb.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        gBodies[cmd.b] = nb;
        // Invalidate contacts involving this collider (current ping-pong half).
        uint nContacts = ALOAD(gCounters[0].contactCount);
        uint curOff = HOST_S(11);
        for (uint ci = 0; ci < nContacts; ci++)
        {
            Contact ct = READ_CONTACT(curOff + ci);
            if (ct.colliderA == cmd.a || ct.colliderB == cmd.a)
            {
                ct.alive = 0;
                STORE_CONTACT(curOff + ci, ct);
            }
        }
    }
}

void predictOne(uint i)
{
    AvbdParams p = PARAMS;
    BodyState b = gBodies[i];
    if (bodyIsKinematic(b))
    {
        // Kinematic: x⁻ for the contact rows, then the given motion (no
        // gravity, no solve). A stopped kinematic is a static this step.
        gBodies[i].initialPos = b.position;
        gBodies[i].initialRot = b.orientation;
        if (kinematicMoving(b))
        {
            gBodies[i].position = b.position + b.linearVel * p.dt;
            gBodies[i].orientation = quatNormalize(
                b.orientation + quatMul(float4(b.angularVel * p.dt, 0.0), b.orientation) * 0.5);
        }
        return;
    }
    // Sleeping/static: do not touch prev/pose (three-avbd mover compaction).
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    float dt = p.dt;
    // Adaptive warm-start: (v - v_prev)/dt along up (three-avbd / VBD). gBodiesPrev still
    // holds the previous step's snapshot (incl. that step's incoming velocity).
    float3 prevVel = BODY_PREV(i).linearVel;
    float3 accel = (b.linearVel - prevVel) / max(dt, 1e-8);
    float accelExt = accel.y * sign(p.gravityY);
    float accelWeight = clamp(accelExt / max(abs(p.gravityY), 1e-6), 0.0, 1.0);
    // three-avbd REST_SPEED: |v| + |ω|·radius (moment.w). Use max collider bound.
    float radius = 0.5;
    // Cheap proxy: AABB diagonal of first collider with this body (params.w for boxes).
    // Full scan is O(colliders); keep 0.5 if unknown — close for pile cubes.
    float slow = length(b.linearVel) + length(b.angularVel) * radius;
    bool startAtRest = slow < 0.1;

    // three-avbd inertial target + qadd (not exp-map).
    float3 inertialLin = b.position + b.linearVel * dt + float3(0, p.gravityY, 0) * (dt * dt);
    float4 inertialAng = quatNormalize(
        b.orientation + quatMul(float4(b.angularVel * dt, 0.0), b.orientation) * 0.5);
    // x⁻ snapshot for velocity extract (three-avbd initialPos). The previous-body
    // row is only read for this step's incoming velocity; the pose lives in
    // initialPos/initialRot, so store just that field.
    float3 x0 = b.position;
    float4 q0 = b.orientation;
    BODY_PREV(i).linearVel = b.linearVel;

    // A deferred wake keeps its pose this step (it is solved from the next).
    if (!startAtRest && !wakePending(b))
    {
        // CCD body with a time of impact (ccd_toi): start the iterate at the
        // TOI pose, short of what it hits, so the trust region never has to
        // pull it back through the wall (CCD audit row 2). The inertial target
        // below is unchanged, so its momentum is kept.
        float toi = ccdStartFraction(i, b);
        if (toi < 1.0)
        {
            b.position = b.position + b.linearVel * (toi * dt) + float3(0, p.gravityY, 0) * (toi * accelWeight * dt * dt);
            b.orientation = quatNormalize(
                b.orientation + quatMul(float4(b.angularVel * (toi * dt), 0.0), b.orientation) * 0.5);
        }
        else
        {
            b.position = b.position + b.linearVel * dt + float3(0, p.gravityY, 0) * (accelWeight * dt * dt);
            b.orientation = quatNormalize(
                b.orientation + quatMul(float4(b.angularVel * dt, 0.0), b.orientation) * 0.5);
        }
    }
    Aabb box;
    box.mn = inertialLin;
    box.mx = float3(inertialAng.x, inertialAng.y, inertialAng.z);
    box._pad0 = inertialAng.w;
    box._pad1 = 0.0;
    gAabbs[i] = box;
    // Store only the fields prediction changes (velocities/mass/flags untouched).
    gBodies[i].position = b.position;
    gBodies[i].orientation = b.orientation;
    gBodies[i].initialPos = x0;
    gBodies[i].initialRot = q0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void predict_bodies(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    // PASS_U == 1: one thread per entry of this step's awake list (the only
    // bodies predictOne changes); hosts dispatch it from the awake-count slot.
    if (PASS_U == 1u)
    {
        if (i >= ALOAD(gCounters[0].awakeCount)) return;
        predictOne(gAwakeList[i]);
        return;
    }
    if (i >= gCounters[0].bodyCount) return;
    predictOne(i);
}

// Previous-step awake list (host sets gScratchAtomics[7] = prev awake count).
[shader("compute")]
[numthreads(64, 1, 1)]
void predict_bodies_listed(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    uint n = HOST_S(7);
    if (li >= n) return;
    predictOne(gAwakeList[li]);
}

// First collider of `body` (colliders are sorted by body; host-checked).
uint firstColliderOfBody(uint body)
{
    uint lo = 0u, hi = gCounters[0].colliderCount;
    while (lo < hi)
    {
        uint mid = (lo + hi) >> 1;
        if (gColliders[mid].body < body) lo = mid + 1u; else hi = mid;
    }
    return lo;
}

// Moved-only bounds (HOST_S(19) != 0, JS host): one thread per body that was
// awake in the previous step (gAwakeList still holds that list, its length is
// copied to HOST_S(6) before the counters reset). Only those bodies moved, so
// every other collider's stored AABB/OBB is already exact. Colliders are
// sorted by body (host-checked), so a body's run is found by binary search.
// Hosts fall back to the full pass after any host-side pose or collider write.
[shader("compute")]
[numthreads(64, 1, 1)]
void build_aabbs(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (HOST_S(19) != 0u)
    {
        if (i >= HOST_S(6)) return;
        uint body = gAwakeList[i];
        BodyState b = gBodies[body];
        // A body that fell asleep at the end of last step keeps its previous
        // bounds, exactly as in the full pass below.
        if (bodyIsSleeping(b) && bodyIsDynamic(b)) return;
        uint n = gCounters[0].colliderCount;
        for (uint c = firstColliderOfBody(body); c < n && gColliders[c].body == body; c++)
            buildColliderBounds(c, gColliders[c], b);
        return;
    }
    if (i >= gCounters[0].colliderCount) return;
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    // Keep a sleeping body's previous AABB/OBB, but build them on the first step:
    // scenes may start asleep (pre-fractured rubble) and still need bounds.
    if (bodyIsSleeping(b) && bodyIsDynamic(b) && HOST_S(17) > 0u)
    {
        return;
    }
    buildColliderBounds(i, col, b);
}

void buildColliderBounds(uint i, Collider col, BodyState b)
{
    float3 worldPos = transformPoint(b.position, b.orientation, col.localPos);
    float4 worldRot = quatMul(b.orientation, col.localRot);
    float3 mn, mx;
    // three-avbd testPair pads world AABBs by 1e-4 only (f32 round-off). Inflating by
    // contactOffset (~1 cm) pulled in ring-adjacent bricks that faceSeparated should
    // still drop — but the fatter boxes also bloated grid occupancy. Margin stays in C0,
    // except under FLAG_CONTACT_GAP, where contacts are speculative within it.
    // Look-ahead sweep: the linear motion v·h extends the box along v only
    // (as Box2D sweeps bullets); spin and gravity pad it all round. `reach`
    // stays the scalar bound for the face tests and the grid level.
    float reach = colliderReach(b, col);
    float3 sweep = reach > 0.0 ? b.linearVel * PARAMS.dt : float3(0, 0, 0);
    float aabbPad = max(1.0e-4, speculativeMargin()) + (reach > 0.0 ? max(reach - length(sweep), 0.0) : 0.0);
    // A CCD body can be deflected off its straight sweep within the step (a
    // struck piece that spins carries it sideways): pad it all round by the
    // farthest the solve can move it from its predicted pose (the trust
    // region's per-iteration step times the iterations), at most the sweep.
    if (bodyIsCcd(b)) aabbPad += ccdDeflectionBound(length(sweep));
    if (col.shape == SHAPE_SPHERE)
    {
        worldAabbSphere(worldPos, col.params.x + aabbPad, mn, mx);
    }
    else if (col.shape == SHAPE_BOX)
    {
        worldAabbBox(worldPos, worldRot, col.params.xyz + aabbPad, mn, mx);
    }
    else if (col.shape == SHAPE_CAPSULE)
    {
        worldAabbCapsule(worldPos, worldRot, col.params.x + aabbPad, col.params.y, mn, mx);
    }
    else if (col.shape == SHAPE_PLANE)
    {
        mn = float3(-1e6, -1e6, -1e6);
        mx = float3(1e6, 1e6, 1e6);
    }
    else if (col.shape == SHAPE_HEIGHTFIELD)
    {
        heightFieldWorldAabb(col, worldPos, worldRot, aabbPad, mn, mx);
    }
    else
    {
        // Exact bounds from every transformed vertex: one pass per collider, not
        // per pair, and far tighter than a radius about the collider origin.
        // One matrix per collider: 9 FMAs per vertex instead of a quaternion
        // rotation (the hull may be shared, but its pose is per collider).
        HullRef h = hullOf(col);
        float3 ax = quatRotate(worldRot, float3(1, 0, 0));
        float3 ay = quatRotate(worldRot, float3(0, 1, 0));
        float3 az = quatRotate(worldRot, float3(0, 0, 1));
        mn = float3(1e30, 1e30, 1e30);
        mx = float3(-1e30, -1e30, -1e30);
        for (uint v = 0u; v < h.nv; v++)
        {
            float3 l = hullVertex(h, v);
            float3 p = ax * l.x + ay * l.y + az * l.z;
            mn = min(mn, p);
            mx = max(mx, p);
        }
        mn += worldPos - aabbPad;
        mx += worldPos + aabbPad;
    }
    mn = min(mn, mn + sweep);
    mx = max(mx, mx + sweep);
    Aabb box;
    box.mn = mn; box.mx = mx;
    float boundR = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    // The hierarchical grid needs every AABB half extent ≤ _pad0; a capsule's
    // reaches radius + half height, beyond length(radius, half height).
    if (col.shape == SHAPE_CAPSULE) boundR = col.params.x + col.params.y;
    // The swept reach counts as size, so the grid level still bounds the box:
    // the whole sweep, not the capped reach (the box spans all of it). Sweeps
    // too long for the grid take the fast list, where it is scheduled.
    box._pad0 = boundR + (fastListScheduled() ? max(reach, length(sweep)) : reach);
    box._pad1 = asfloat(col.body);
    // Store at collider index offset by max_bodies — host allocates gAabbs sized max_bodies+max_colliders
    // Convention: collider AABBs start at index max_bodies. Host sets gScratchAtomics[1] = max_bodies.
    uint base = HOST_S(1);
    gAabbs[base + i] = box;
    // World OBB cache for box colliders (BP face-reject + narrowphase). Convex
    // hulls cache their local AABB as a conservative OBB for the same rejection.
    if (col.shape == SHAPE_CONVEX)
    {
        HullRef h = hullOf(col);
        WorldObbGpu w;
        w.center = float4(transformPoint(worldPos, worldRot, hullLocalCenter(h)), max(reach, length(sweep)));
        w.half = float4(hullLocalHalf(h), 0.0);
        w.axis0 = float4(quatRotate(worldRot, float3(1, 0, 0)), 0.0);
        w.axis1 = float4(quatRotate(worldRot, float3(0, 1, 0)), 0.0);
        w.axis2 = float4(quatRotate(worldRot, float3(0, 0, 1)), 0.0);
        gWorldObbs[i] = w;
    }
    if (col.shape == SHAPE_BOX)
    {
        float4 q = quatMul(b.orientation, col.localRot);
        BoxObb obb = makeBoxObb(worldPos, q, col.params.xyz);
        WorldObbGpu w;
        w.center = float4(obb.center, max(reach, length(sweep)));
        w.half = float4(obb.half, 0.0);
        w.axis0 = float4(obb.axis0, 0.0);
        w.axis1 = float4(obb.axis1, 0.0);
        w.axis2 = float4(obb.axis2, 0.0);
        gWorldObbs[i] = w;
    }
}

static const uint GRID_HASH_SIZE = 131072u;
// Oversized statics (wider than half a grid cell) cannot be found from the
// uniform grid's 27-cell stencil, so broadphase_insert lists their collider ids:
// count in gScratchAtomics[80], ids after the event ring (engine-schema
// LARGE_STATIC_*). Order follows the atomic append; every listed static is
// tested, so the result does not depend on it.
static const uint LARGE_STATIC_COUNT = 80u;
static const uint LARGE_STATIC_CAP = 1024u;
static const uint LARGE_STATIC_BASE = EVENT_RING_BASE + 4u + 4u * EVENT_RING_CAPACITY;
uint largeStaticCount() { return min(ALOAD(gScratchAtomics[LARGE_STATIC_COUNT]), LARGE_STATIC_CAP); }
uint largeStaticCollider(uint k) { return ALOAD(gScratchAtomics[LARGE_STATIC_BASE + k]); }

// Continuous-collision scratch (engine-schema CCD_*), after the oversized-static
// list. Counts are reset by broadphase_clear (CCD bodies, fast colliders) and
// build_awake_list (impulses, after apply_impulses consumed them).
//  - CCD bodies: 4 words per slot [body, min TOI bits, winning candidate, flags];
//  - fast colliders: swept boxes longer than a grid cell (row 1), queried by
//    broadphase_fast instead of the grid;
//  - impulses: 8 words per record [body, mode, J.xyz, point.xyz] (row 4).
static const uint CCD_BASE = LARGE_STATIC_BASE + LARGE_STATIC_CAP;
static const uint CCD_BODY_COUNT = CCD_BASE;
static const uint FAST_COUNT = CCD_BASE + 1u;
static const uint IMPULSE_COUNT = CCD_BASE + 2u;
// TOI casts that fell back to look-ahead this step (GJK budget, degenerate).
static const uint CCD_FALLBACKS = CCD_BASE + 3u;
// Impulse records from the host (the late apply_impulses pass starts after them).
static const uint IMPULSE_FIRST = CCD_BASE + 4u;
static const uint CCD_BODY_CAP = 1024u;
static const uint CCD_BODY_BASE = CCD_BASE + 8u;
static const uint FAST_CAP = 16384u;
static const uint FAST_BASE = CCD_BODY_BASE + 4u * CCD_BODY_CAP;
static const uint IMPULSE_CAP = 1024u;
static const uint IMPULSE_WORDS = 8u;
static const uint IMPULSE_BASE = FAST_BASE + FAST_CAP;
static const uint CCD_CARRY_BASE = IMPULSE_BASE + IMPULSE_WORDS * IMPULSE_CAP;
static const uint CCD_CARRY_WORDS = 32u;
static const uint CCD_PAIR_COUNT = CCD_BASE + 5u;
// Largest |v|·h of this step's CCD bodies (asuint, ccd_register; ccd_trust).
static const uint CCD_MAX_REACH = CCD_BASE + 6u;
static const uint CCD_PAIR_CAP = 131072u;
static const uint CCD_PAIR_BASE = CCD_CARRY_BASE + CCD_CARRY_WORDS * CCD_BODY_CAP;
static const uint CCD_NO_PAIR = 0xFFFFFFFFu;
static const uint CCD_TOI_NONE = 0x7F800000u; // +inf

uint ccdBodyCount() { return min(ALOAD(gScratchAtomics[CCD_BODY_COUNT]), CCD_BODY_CAP); }
uint fastCount() { return min(ALOAD(gScratchAtomics[FAST_COUNT]), FAST_CAP); }

// CCD slot of `body` this step, or CCD_NO_PAIR (initialPad carries the slot
// while registered; the record's body word confirms it).
uint ccdSlotOf(uint body, BodyState b)
{
    if (!bodyIsCcd(b)) return CCD_NO_PAIR;
    uint slot = asuint(b.initialPad);
    if (slot >= ccdBodyCount()) return CCD_NO_PAIR;
    return ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * slot]) == body ? slot : CCD_NO_PAIR;
}

// HOST_S(30) bit 4: this step schedules broadphase_fast. Step graphs that do
// not (the legacy Metal schedule) keep the grid exactly as before: no fast
// list, and the grid level sized by the capped reach.
bool fastListScheduled() { return (HOST_S(30) & 16u) != 0u; }
// The fast list exists where the broadphase uses a grid (hierarchical, or the
// uniform grid past the exact pairwise path's 512 colliders).
bool fastListOn() { return fastListScheduled() && (HOST_S(18) != 0u || gCounters[0].colliderCount > 512u); }

// A swept box the grid cannot hold: a moving collider whose sweep is longer
// than a cell (the 27-cell stencil covers boxes within a cell), or whose swept
// size would put it past the hierarchical grid's top level (the oversized
// list, which only movers query). broadphase_insert and broadphase_query
// evaluate this on the same inputs, so both agree.
bool colliderIsFast(uint i, Collider col, BodyState b)
{
    if (col.shape == SHAPE_PLANE || !fastListOn() || !bodyMovingOrKinematic(b)) return false;
    if (colliderReach(b, col) <= 0.0) return false; // unswept box
    float sweep = length(b.linearVel) * PARAMS.dt;
    if (HOST_S(18) == 0u) return sweep > max(PARAMS.gridCellSize, 0.5);
    return sweep > PARAMS.gridCellSize || hierLevel(gAabbs[HOST_S(1) + i]._pad0) >= HOST_S(18);
}

void fastListAppend(uint i)
{
    uint slot;
    AADD3(gScratchAtomics[FAST_COUNT], 1u, slot);
    if (slot < FAST_CAP) ASTORE(gScratchAtomics[FAST_BASE + slot], i);
    else AOR2(gCounters[0].overflowFlags, OVERFLOW_FAST_LIST);
}
// Heads/links carry immutable broadphase eligibility alongside the collider ID.
// Collider IDs occupy 31 bits; 0xffffffff remains the empty-list sentinel.
static const uint GRID_AWAKE_DYNAMIC = 0x80000000u;
static const uint GRID_COLLIDER_MASK = 0x7FFFFFFFu;

// Power-of-two grid table size from the host (HOST_S(16)); 0 keeps the fixed
// table for hosts that do not size it (Metal). Heads live in gConstraintIds
// before the contact hash and CSR reuse that region later in the step.
uint gridHashSize()
{
    uint n = HOST_S(16);
    return n != 0u ? n : GRID_HASH_SIZE;
}

// Opt-in split grids (HOST_S(20) != 0): awake dynamics and everything else
// (sleepers, small statics) get separate head tables so awake queriers can walk
// only their own cell plus the 13 forward neighbours of the awake table; each
// awake pair is still found once and emitted as (min, max) like before.
bool gridSplit() { return HOST_S(20) != 0u; }

// Opt-in resting-pair carry (8ep, HOST_S(28) != 0): a pair whose bodies are
// both at rest (static or sleeping) is re-emitted from last step's manifolds
// instead of the broadphase, so its warm-start λ/k survive sleep and a woken
// body does not restart its resting contacts cold. Sleepers then emit no
// rest pairs themselves (no duplicates); awake bodies still emit theirs.
// WGSL hosts only: Metal's HOST_S(k) reads gScratchAtomics, whose words 26+
// hold the large-body list.
#if defined(ENGINE_WGSL)
bool carryRestPairs() { return HOST_S(28) != 0u; }
#else
bool carryRestPairs() { return false; }
#endif

uint gridHash(int x, int y, int z)
{
    uint h = (uint)x * 73856093u ^ (uint)y * 19349663u ^ (uint)z * 83492791u;
    return h & (gridHashSize() - 1u);
}

// Open-addressed joint-pair set in gEvents (three-avbd / demo3d constrainedTo).
// Replaces O(joints) scan that dominated Chain Mail / Jointed Drop broadphase.
// Each slot is claimed with its joint's index; the inserter then stores the
// body pair in two parallel arrays (gEvents[size + slot], [2 size + slot]), so
// lookups compare the real pair (no false "jointed" answers from hash
// collisions) without binding gJoints. The table is at least 65536 slots and at
// least twice the joint count; hosts size gEvents to 3 x jointHashSize() words.
// Broken bonds are left out, so their bodies collide again.
static const uint JOINT_HASH_MIN = 65536u;
static const uint JOINT_HASH_EMPTY = 0xFFFFFFFFu;
static const uint JOINT_HASH_PROBES = 64u;

uint jointHashSize()
{
    uint v = max(2u * gCounters[0].jointCount, JOINT_HASH_MIN) - 1u;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1u;
}

#include "world_step_pairs.slang"
#include "world_step_oversize.slang"
#include "world_step_awake_queries.slang"
#include "world_step_rest_tables.slang"
#include "world_step_rest_inserts.slang"
#include "world_step_margins.slang"
#include "world_step_warmstart.slang"
#include "world_step_persist.slang"
#include "world_step_csr.slang"
#include "world_step_coloring.slang"
#include "world_step_primal.slang"
#include "world_step_sleep.slang"
