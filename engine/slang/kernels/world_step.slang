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
#include "../physics/convex.slang"

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
        b.position = b.position + b.linearVel * dt + float3(0, p.gravityY, 0) * (accelWeight * dt * dt);
        b.orientation = quatNormalize(
            b.orientation + quatMul(float4(b.angularVel * dt, 0.0), b.orientation) * 0.5);
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
    // The swept reach counts as size, so the grid level still bounds the box.
    box._pad0 = boundR + reach;
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
        w.center = float4(transformPoint(worldPos, worldRot, hullLocalCenter(h)), reach);
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
        w.center = float4(obb.center, reach);
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

uint jointPairHash(uint bodyA, uint bodyB)
{
    uint lo = min(bodyA, bodyB);
    uint hi = max(bodyA, bodyB);
    return (lo * 73856093u) ^ (hi * 19349663u) ^ (lo << 16) ^ (hi >> 3);
}


[shader("compute")]
[numthreads(64, 1, 1)]
void joint_hash_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= jointHashSize()) return;
    ASTORE(gEvents[i], JOINT_HASH_EMPTY);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void joint_hash_insert(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint nJ = gCounters[0].jointCount;
    if (i >= nJ) return;
    Joint jj = gJoints[i];
    // Broken joints stop filtering; COLLIDE joints never filter (PhysX flag);
    // a wheel standing on a crate must still let the chassis collide with it.
    if ((jj.flags & (JOINT_FLAG_BROKEN | JOINT_FLAG_COLLIDE | JOINT_FLAG_TYRE)) != 0u) return;
    uint mask = jointHashSize() - 1u;
    uint h = jointPairHash(jj.bodyA, jj.bodyB) & mask;
    for (uint probe = 0; probe < JOINT_HASH_PROBES; probe++)
    {
        uint slot = (h + probe) & mask;
        // Retry the same slot while WGSL's weak CAS fails spuriously: advancing
        // would leave an EMPTY hole and break open-address lookups.
        uint owner = JOINT_HASH_EMPTY;
        for (uint spin = 0u; spin < 64u && owner == JOINT_HASH_EMPTY; spin++)
        {
            uint prev;
            ACAS4(gEvents[slot], JOINT_HASH_EMPTY, i, prev);
            owner = ALOAD(gEvents[slot]);
        }
        if (owner == i)
        {
            uint size = mask + 1u;
            ASTORE(gEvents[size + slot], min(jj.bodyA, jj.bodyB));
            ASTORE(gEvents[2u * size + slot], max(jj.bodyA, jj.bodyB));
            return;
        }
    }
    AOR2(gCounters[0].overflowFlags, OVERFLOW_JOINTS);
}

bool bodiesJointed(uint bodyA, uint bodyB)
{
    // Ring/Wall have zero joints — skip the open-address probe entirely (and do not
    // require a clear of gEvents each step).
    if (gCounters[0].jointCount == 0u) return false;
    uint lo = min(bodyA, bodyB);
    uint hi = max(bodyA, bodyB);
    uint mask = jointHashSize() - 1u;
    uint h = jointPairHash(lo, hi) & mask;
    for (uint probe = 0; probe < JOINT_HASH_PROBES; probe++)
    {
        uint slot = (h + probe) & mask;
        if (ALOAD(gEvents[slot]) == JOINT_HASH_EMPTY) return false;
        if (ALOAD(gEvents[mask + 1u + slot]) == lo && ALOAD(gEvents[2u * (mask + 1u) + slot]) == hi)
            return true;
    }
    return false;
}

// Collision filtering (per collider; HOST_S(30) != 0 when any filter is set).
// Collider.generation: categories (bits 0..15) | mask (bits 16..31); 0 means
// "every category, collides with everything". Collider.material bits 16..31:
// signed group; a shared positive group always collides and a shared negative
// group never does, overriding categories (Box2D-style), for e.g. ragdolls or
// the chunks of one fractured compound.
bool collidersFilteredOut(uint i, uint j)
{
    Collider a = gColliders[i];
    Collider b = gColliders[j];
    int ga = int(a.material) >> 16;
    int gb = int(b.material) >> 16;
    if (ga != 0 && ga == gb) return ga < 0;
    uint fa = a.generation == 0u ? 0xFFFFFFFFu : a.generation;
    uint fb = b.generation == 0u ? 0xFFFFFFFFu : b.generation;
    return ((fa & 0xFFFFu) & (fb >> 16)) == 0u || ((fb & 0xFFFFu) & (fa >> 16)) == 0u;
}

uint materialIndex(Collider c) { return c.material & 0xFFFFu; }

// Look-ahead pairs are tested with their relative sweep: each box is swept by
// its own v·h for the broadphase, so pieces carried along together (debris in
// front of the ball) overlap their neighbours' boxes although they cannot
// close on them. Undo both sweeps and sweep one box by (vi − vj)·h instead.
bool lookAheadSeparated(uint i, uint j)
{
    if ((PARAMS.flags & FLAG_SPEC_VELOCITY) == 0u) return false;
    Collider ci = gColliders[i];
    Collider cj = gColliders[j];
    BodyState bi = gBodies[ci.body];
    BodyState bj = gBodies[cj.body];
    float3 si = colliderReach(bi, ci) > 0.0 ? bi.linearVel * PARAMS.dt : float3(0, 0, 0);
    float3 sj = colliderReach(bj, cj) > 0.0 ? bj.linearVel * PARAMS.dt : float3(0, 0, 0);
    if (all(si == float3(0, 0, 0)) && all(sj == float3(0, 0, 0))) return false;
    uint base = HOST_S(1);
    Aabb ai = gAabbs[base + i];
    Aabb aj = gAabbs[base + j];
    float3 imn = ai.mn - min(si, float3(0, 0, 0)), imx = ai.mx - max(si, float3(0, 0, 0));
    float3 jmn = aj.mn - min(sj, float3(0, 0, 0)), jmx = aj.mx - max(sj, float3(0, 0, 0));
    float3 rel = si - sj;
    imn = min(imn, imn + rel);
    imx = max(imx, imx + rel);
    return any(imx < jmn) || any(jmx < imn);
}

// Vehicle sever grace (HOST_S(30) bit 1, set by vehicle hosts when the scene has
// no category filters): Collider.generation = ignored body + 1 (0 = none), so a
// severed wheel or part does not explode out of the chassis it was attached
// to. The vehicle kernels write it on the GPU and clear it once they separate.
bool collidersIgnoreBody(uint i, uint j)
{
    Collider a = gColliders[i];
    Collider b = gColliders[j];
    return (a.generation != 0u && a.generation - 1u == b.body) ||
           (b.generation != 0u && b.generation - 1u == a.body);
}

void emitCandidate(uint i, uint j)
{
    if (i == j) return;
    uint filterMode = HOST_S(30);
    if ((filterMode & 1u) != 0u ? collidersFilteredOut(i, j)
        : (filterMode & 2u) != 0u && collidersIgnoreBody(i, j))
        return;
    if (lookAheadSeparated(i, j)) return;
    uint a = min(i, j);
    uint b = max(i, j);
    uint slot;
    AADD3(gCounters[0].candidateCount, 1, slot);
    uint maxCand = HOST_S(2);
    if (slot >= maxCand)
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_CANDIDATES);
        return;
    }
    CandidatePair p;
    p.colliderA = a;
    p.colliderB = b;
    p.key = (a << 16) ^ b;
    p._pad = 0;
    gCandidates[slot] = p;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    // Split grids (HOST_S(20) != 0): awake heads [0, n), other heads [n, 2n).
    if (i >= gridHashSize() * (gridSplit() ? 2u : 1u)) return;
    ASTORE(gConstraintIds[i], 0xFFFFFFFFu);
    if (i == 0u && gridSplit()) ASTORE(gScratchAtomics[52], 0u);
    // Metal hosts do not rewrite GPU-owned scratch words each step: reset the
    // large-dynamic (25), plane (42) and large-static (53) list counts here.
    if (i == 0u)
    {
        ASTORE(gScratchAtomics[25], 0u);
        ASTORE(gScratchAtomics[42], 0u);
        ASTORE(gScratchAtomics[LARGE_STATIC_COUNT], 0u);
    }
}

BoxObb worldObbFromCache(uint colliderIdx)
{
    WorldObbGpu w = gWorldObbs[colliderIdx];
    BoxObb b;
    b.center = w.center.xyz;
    b.half = w.half.xyz;
    b.axis0 = w.axis0.xyz;
    b.axis1 = w.axis1.xyz;
    b.axis2 = w.axis2.xyz;
    return b;
}

// three-avbd Stage 8h: reject AABB-overlapping box pairs separated on a face axis
// by more than `gap`. `probe` OBB is built once per broadphase thread.
bool boxesSeparatedOnFaceAxis(BoxObb probe, uint otherCollider, float gap)
{
    Collider cb = gColliders[otherCollider];
    if (!shapeHasHull(cb.shape)) return false;
    BoxObb other = worldObbFromCache(otherCollider);
    gap += cachedReach(otherCollider);
    float3 d = other.center - probe.center;
    float3 hA = probe.half;
    float3 hB = other.half;
    float3 a0 = probe.axis0;
    float3 a1 = probe.axis1;
    float3 a2 = probe.axis2;
    float3 b0 = other.axis0;
    float3 b1 = other.axis1;
    float3 b2 = other.axis2;
    {
        float rB = hB.x * abs(dot(a0, b0)) + hB.y * abs(dot(a0, b1)) + hB.z * abs(dot(a0, b2));
        if (abs(dot(d, a0)) > hA.x + rB + gap) return true;
        rB = hB.x * abs(dot(a1, b0)) + hB.y * abs(dot(a1, b1)) + hB.z * abs(dot(a1, b2));
        if (abs(dot(d, a1)) > hA.y + rB + gap) return true;
        rB = hB.x * abs(dot(a2, b0)) + hB.y * abs(dot(a2, b1)) + hB.z * abs(dot(a2, b2));
        if (abs(dot(d, a2)) > hA.z + rB + gap) return true;
    }
    {
        float rA = hA.x * abs(dot(b0, a0)) + hA.y * abs(dot(b0, a1)) + hA.z * abs(dot(b0, a2));
        if (abs(dot(d, b0)) > hB.x + rA + gap) return true;
        rA = hA.x * abs(dot(b1, a0)) + hA.y * abs(dot(b1, a1)) + hA.z * abs(dot(b1, a2));
        if (abs(dot(d, b1)) > hB.y + rA + gap) return true;
        rA = hA.x * abs(dot(b2, a0)) + hA.y * abs(dot(b2, a1)) + hA.z * abs(dot(b2, a2));
        if (abs(dot(d, b2)) > hB.z + rA + gap) return true;
    }
    return false;
}

// ---- Hierarchical size-level grid (HOST_S(18) = level count; 0 = uniform grid) ----
// Geometric levels in the spirit of three-avbd's fine/coarse ownership. Each
// collider is inserted once, at the level whose cell (base·2^level) covers its
// bound diameter, so compound children and mixed hull sizes never inflate the
// cell of the whole scene. A collider queries its own and every coarser level's
// 27 cells; fixed/sleeping fine colliders still query coarse movers (coarse
// movers never visit fine cells). Colliders beyond the top level (oversized
// statics) go to the explicit large list, tested by every mover.
static const uint HIER_LEVEL_MASK_SLOT = 51u;
// Uniform-grid large dynamics: count at 25, up to 16 IDs at 64..79. Metal reads
// HOST_S(k) from these same scratch words, so the IDs must stay clear of every
// host word (0..31): at 26..41 they overwrote HOST_S(26..30) mid-step.
static const uint LEGACY_LARGE_BASE = 64u;
// Levels holding at least one moving collider (scratch 31, zeroed by every host
// each step). Sleeping/static queriers only pair with movers, so they skip
// every level without one, and return at once when none is coarser than them.
static const uint HIER_MOVING_MASK_SLOT = 31u;
// Oversized statics: count at 53, up to 10 collider IDs at 54..63.
static const uint HIER_LARGE_COUNT_SLOT = 53u;
static const uint HIER_LARGE_MAX = 10u;
// Moving colliders above the finest level (count at 32, up to 9 IDs at
// 33..41; host-zeroed each step). Sleeping/static colliders only pair with
// coarser movers, so while the list is complete a sleeper that overlaps none
// of them skips its query: the same pairs, without 27 hash probes per level.
static const uint HIER_COARSE_COUNT_SLOT = 32u;
static const uint HIER_COARSE_MAX = 9u;

uint hierLevel(float radius)
{
    float diameter = 2.0 * radius + 0.0004;
    return uint(max(0.0, ceil(log2(max(diameter / PARAMS.gridCellSize, 1.0)))));
}

float hierCellSize(uint level) { return PARAMS.gridCellSize * exp2(float(level)); }

int3 hierCell(Aabb a, uint level)
{
    return int3(floor((a.mn + a.mx) * (0.5 / hierCellSize(level))));
}

uint hierHash(int3 c, uint level)
{
    return (gridHash(c.x, c.y, c.z) ^ (level * 2654435769u)) & (gridHashSize() - 1u);
}

bool aabbOverlap(Aabb a, Aabb b)
{
    return all(a.mn <= b.mx) && all(b.mn <= a.mx);
}

bool bodyMoving(BodyState b) { return bodyIsDynamic(b) && !bodyIsSleeping(b); }

// AABB and plane n·x = w: true when the box lies entirely above the plane.
bool aabbAbovePlane(Aabb a, float4 plane)
{
    float3 n = normalize(plane.xyz);
    float3 c = 0.5 * (a.mn + a.mx);
    float3 e = 0.5 * (a.mx - a.mn);
    return dot(n, c) - dot(abs(n), e) - plane.w > 0.0;
}

void hierInsert(uint i)
{
    Collider col = gColliders[i];
    if (col.shape == SHAPE_PLANE)
    {
        uint slot;
        AADD3(gScratchAtomics[42], 1, slot);
        if (slot < 8u) ASTORE(gScratchAtomics[43u + slot], i);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    Aabb box = gAabbs[HOST_S(1) + i];
    uint level = hierLevel(box._pad0);
    if (level >= HOST_S(18))
    {
        uint slot;
        AADD3(gScratchAtomics[HIER_LARGE_COUNT_SLOT], 1, slot);
        if (slot < HIER_LARGE_MAX) ASTORE(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + slot], i);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    // Occupied-level mask (scratch 51, zeroed by the host each step): queries
    // skip empty levels, so the host can keep slack levels for f32 rounding.
    AOR2(gScratchAtomics[HIER_LEVEL_MASK_SLOT], 1u << level);
    if (bodyMoving(gBodies[col.body]))
    {
        AOR2(gScratchAtomics[HIER_MOVING_MASK_SLOT], 1u << level);
        if (level > 0u)
        {
            uint k;
            AADD3(gScratchAtomics[HIER_COARSE_COUNT_SLOT], 1u, k);
            if (k < HIER_COARSE_MAX) ASTORE(gScratchAtomics[HIER_COARSE_COUNT_SLOT + 1u + k], i);
        }
    }
    uint prev;
    AXCHG3(gConstraintIds[hierHash(hierCell(box, level), level)], i, prev);
    // The chain link lives in the AABB's spare word, as in the split grid:
    // a query hop then fetches link and bounds with one load.
    gAabbs[HOST_S(1) + i]._pad1 = asfloat(prev);
}

// ---- Persistent resting table (JS host; HOST_S(25) bit 0) ----
// Static and sleeping colliders live in a resting table (R) that persists
// across steps; moving colliders go into an awake table (W) each step, and
// only they query (broadphase_query_awake). Both tables sit past the CSR
// index in gConstraintIds (restBase): a 32-word header (occupied-level mask,
// planes, oversized statics), R heads, W heads, then R chain links (one word
// per collider; W links stay in the AABB spare word). R is rebuilt from
// scratch on the step after any body fell asleep (or when the host asks):
// otherwise no collider stops moving, so every resting collider is already
// in R at its current cell. A woken collider stays in R as a tombstone: it
// moves, so R visits skip it, and it is found in W instead. W is emptied after
// the queries by the same moving colliders, so no step clears whole tables.
static const uint HIER_LINK_SPARE = 0xFFFFFFFFu;
static const uint REST_HEADER = 32u;
// Header words: 0 level mask, 1 plane count, 2..9 planes, 10 large count,
// 11..20 oversized statics.

bool restTables() { return (HOST_S(25) & 1u) != 0u; }
uint restBase() { return HOST_S(24) + HOST_S(1) + 1u + HOST_S(9); } // past the tagged warm hash
uint restHeads() { return restBase() + REST_HEADER; }
uint awakeHeads() { return restHeads() + gridHashSize(); }
uint restLinks() { return awakeHeads() + gridHashSize(); }

// One cell's chain (heads at headBase; links in the AABB spare word, or at
// linkBase + j for R) as seen by querier i: the pair rules of hierQuery.
void hierVisitCell(uint i, Collider a, bool active, uint own, Aabb aa, bool probeHasObb,
    BoxObb probe, float faceGap, int3 cell, uint level, uint headBase, uint linkBase, uint count)
{
    uint base = HOST_S(1);
    bool rest = linkBase != HIER_LINK_SPARE;
    uint j = ALOAD(gConstraintIds[headBase + hierHash(cell, level)]);
    uint guard = 0u;
    while (j != 0xFFFFFFFFu && guard < count)
    {
        guard++;
        Aabb ab = gAabbs[base + j];
        uint next = rest ? ALOAD(gConstraintIds[linkBase + j]) : asuint(ab._pad1);
        // Hash buckets alias: accept only colliders whose true cell/level
        // is this one, so no pair is visited twice. The overlap test runs
        // first: it rejects most entries before the log2/floor checks.
        if (j != i && aabbOverlap(aa, ab) && hierLevel(ab._pad0) == level &&
            all(hierCell(ab, level) == cell))
        {
            Collider b = gColliders[j];
            if (b.body != a.body)
            {
                bool other = bodyMoving(gBodies[b.body]);
                bool owner = level > own || (active && (!other || i < j));
                // A moving collider in R is a tombstone (found in W).
                if ((active || other) && owner && !(rest && other) &&
                    !bodiesJointed(a.body, b.body) &&
                    !(probeHasObb && boxesSeparatedOnFaceAxis(probe, j, faceGap)))
                    emitCandidate(i, j);
            }
        }
        j = next;
    }
    if (j != 0xFFFFFFFFu) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
}

void hierQuery(uint i)
{
    Collider a = gColliders[i];
    if (a.shape == SHAPE_PLANE) return;
    BodyState ba = gBodies[a.body];
    bool active = bodyMoving(ba);
    uint base = HOST_S(1);
    Aabb aa = gAabbs[base + i];
    uint own = hierLevel(aa._pad0);
    uint levels = HOST_S(18);
    // Oversized colliders are static by host contract; movers find them.
    if (own >= levels) return;
    bool probeHasObb = shapeHasHull(a.shape);
    BoxObb probe = probeHasObb ? worldObbFromCache(i)
                               : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    float faceGap = max(0.001, speculativeMargin()) + (probeHasObb ? cachedReach(i) : 0.0);
    if (active)
    {
        uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
        for (uint pi = 0u; pi < nPlanes; pi++)
        {
            uint j = ALOAD(gScratchAtomics[43u + pi]);
            if (!aabbAbovePlane(aa, gColliders[j].params)) emitCandidate(i, j);
        }
        uint nLarge = min(ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT]), HIER_LARGE_MAX);
        for (uint k = 0u; k < nLarge; k++)
        {
            uint j = ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + k]);
            Collider b = gColliders[j];
            if (b.body == a.body || bodiesJointed(a.body, b.body)) continue;
            if (!aabbOverlap(aa, gAabbs[base + j])) continue;
            if (probeHasObb && boxesSeparatedOnFaceAxis(probe, j, faceGap)) continue;
            emitCandidate(i, j);
        }
    }
    uint count = gCounters[0].colliderCount;
    // Movers may pair with anything; others only with movers at coarser levels.
    uint occupied = active ? ALOAD(gScratchAtomics[HIER_LEVEL_MASK_SLOT])
                           : ALOAD(gScratchAtomics[HIER_MOVING_MASK_SLOT]) & ~((2u << own) - 1u);
    if ((occupied >> own) == 0u) return;
    // Params flag 512 (diagnostic) disables the prefilter, to compare outputs.
    if (!active && (PARAMS.flags & 512u) == 0u)
    {
        uint nCoarse = ALOAD(gScratchAtomics[HIER_COARSE_COUNT_SLOT]);
        if (nCoarse <= HIER_COARSE_MAX)
        {
            bool near = false;
            for (uint k = 0u; k < nCoarse; k++)
            {
                Aabb mb = gAabbs[base + ALOAD(gScratchAtomics[HIER_COARSE_COUNT_SLOT + 1u + k])];
                if (hierLevel(mb._pad0) > own && aabbOverlap(aa, mb)) near = true;
            }
            if (!near) return;
        }
    }
    for (uint level = own; level < levels; level++)
    {
        if (((occupied >> level) & 1u) == 0u) continue;
        // A level-L collider's AABB centre lies in its cell and its half extent
        // is at most half a cell (the same bound the 27-cell scan relies on), so
        // only centres within aa ± cell/2 can overlap. That range is a subset of
        // the 27 cells: about 8 at the own level and 1-8 at coarser levels.
        float cs = hierCellSize(level);
        int3 lo = int3(floor((aa.mn - 0.5 * cs) / cs));
        int3 hi = int3(floor((aa.mx + 0.5 * cs) / cs));
        int3 center = hierCell(aa, level);
        lo = max(lo, center - 1);
        hi = min(hi, center + 1);
        for (int z = lo.z; z <= hi.z; z++)
        for (int y = lo.y; y <= hi.y; y++)
        for (int x = lo.x; x <= hi.x; x++)
        {
            int3 cell = int3(x, y, z);
            if (restTables())
            {
                hierVisitCell(i, a, active, own, aa, probeHasObb, probe, faceGap, cell, level,
                    awakeHeads(), HIER_LINK_SPARE, count);
                hierVisitCell(i, a, active, own, aa, probeHasObb, probe, faceGap, cell, level,
                    restHeads(), restLinks(), count);
            }
            else
                hierVisitCell(i, a, active, own, aa, probeHasObb, probe, faceGap, cell, level,
                    0u, HIER_LINK_SPARE, count);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_insert(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    if (HOST_S(18) != 0u)
    {
        hierInsert(i);
        return;
    }
    Collider col = gColliders[i];
    if (col.shape == SHAPE_PLANE)
    {
        uint slot;
        AADD3(gScratchAtomics[42], 1, slot);
        if (slot < 8u) ASTORE(gScratchAtomics[43u + slot], i);
        return;
    }
    // Insert sleeping too — awake movers must find them (island wake on impact).
    uint base = HOST_S(1);
    float cellSize = max(PARAMS.gridCellSize, 0.5);
    Aabb box = gAabbs[base + i];
    float3 ext = box.mx - box.mn;
    // Oversized statics are probed separately; large dynamics use the explicit list.
    BodyState bIns = gBodies[col.body];
    // Statics wider than a cell (ramps, walls, ground) cannot be found from the
    // 27-cell stencil: list them. Past LARGE_STATIC_CAP
    // flag the overflow and keep the legacy behaviour (grid, or endpoint probe).
    float boundR = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    if (!bodyIsDynamic(bIns) && boundR > 0.5 * cellSize)
    {
        uint slot;
        AADD3(gScratchAtomics[LARGE_STATIC_COUNT], 1, slot);
        if (slot < LARGE_STATIC_CAP)
        {
            ASTORE(gScratchAtomics[LARGE_STATIC_BASE + slot], i);
            return;
        }
        AOR2(gCounters[0].overflowFlags, OVERFLOW_LARGE_STATICS);
    }
    if (!bodyIsDynamic(bIns) &&
        (ext.x >= cellSize * 4.0 || ext.y >= cellSize * 4.0 || ext.z >= cellSize * 4.0))
        return;
    float radius = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    // Bigger than the cell (smash ball vs bricks): test from every dynamic, not one hash slot.
    if (bodyIsDynamic(bIns) && radius > cellSize)
    {
        uint slot;
        AADD3(gScratchAtomics[25], 1, slot);
        if (slot < 16u) ASTORE(gScratchAtomics[LEGACY_LARGE_BASE + slot], i);
        return;
    }
    float3 center = 0.5 * (box.mn + box.mx);
    int3 cc = int3(floor(center / cellSize));
    uint h = gridHash(cc.x, cc.y, cc.z);
    uint prev;
    bool awake = bodyIsDynamic(bIns) && !bodyIsSleeping(bIns);
    uint entry = i | (awake ? GRID_AWAKE_DYNAMIC : 0u);
    if (gridSplit() && !awake)
    {
        h += gridHashSize();
        AADD2(gScratchAtomics[52], 1u); // non-empty other table
    }
    AXCHG3(gConstraintIds[h], entry, prev);
    // Split grids keep the next link in the AABB's spare word (8eo): queries
    // then fetch link and bounds with one load. The legacy grid keeps gColorLists.
    if (gridSplit()) gAabbs[base + i]._pad1 = asfloat(prev);
    else gColorLists[i] = prev;
}

// Own cell (index 0), then the 13 neighbours lexicographically after it.
int3 gridForwardOffset(uint c)
{
    if (c == 0u) return int3(0, 0, 0);
    if (c <= 9u)
    {
        uint k = c - 1u; // dz = +1: all nine (dx, dy)
        return int3(int(k % 3u) - 1, int(k / 3u) - 1, 1);
    }
    if (c <= 12u) return int3(int(c - 10u) - 1, 1, 0); // dz = 0, dy = +1
    return int3(1, 0, 0);                              // dz = 0, dy = 0, dx = +1
}

// One cell of a split table. In the awake table every visited pair belongs to
// this querier (forward cell) or to the higher index (own cell); the other
// table keeps the legacy sleeper/static rule. Geometry tests match the full scan.
void gridScanCell(uint i, Collider col, BodyState b, Aabb box, float3 cA, float rA,
    bool probeIsBox, BoxObb probeObb, float faceGap, float cellSize, int3 cell,
    uint tableBase, bool ownCell)
{
    uint base = HOST_S(1);
    bool sleeper = bodyIsSleeping(b);
    uint link = ALOAD(gConstraintIds[tableBase + gridHash(cell.x, cell.y, cell.z)]);
    uint guard = 0;
    while (link != 0xFFFFFFFFu && guard < 4096u)
    {
        uint cur = link & GRID_COLLIDER_MASK;
        bool awakeEntry = (link & GRID_AWAKE_DYNAMIC) != 0u;
        Aabb ab = gAabbs[base + cur];
        link = asuint(ab._pad1);
        guard++;
        if (guard == 4096u)
            AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        if (cur == i || (ownCell && awakeEntry && cur <= i)) continue;
        int3 candidateCell = int3(floor((0.5 * (ab.mn + ab.mx)) / cellSize));
        if (any(candidateCell != cell)) continue;
        float rB = ab._pad0;
        if (rB > 0.0)
        {
            float3 d = cA - 0.5 * (ab.mn + ab.mx);
            float rr = rA + rB;
            if (dot(d, d) > rr * rr) continue;
        }
        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x ||
            box.mx.y < ab.mn.y || ab.mx.y < box.mn.y ||
            box.mx.z < ab.mn.z || ab.mx.z < box.mn.z)
            continue;
        Collider other = gColliders[cur];
        bool keep;
        if (awakeEntry)
            keep = true;
        else
        {
            BodyState ob = gBodies[other.body];
            bool otherSleep = bodyIsDynamic(ob) && bodyIsSleeping(ob);
            keep = cur > i || (!sleeper && otherSleep);
        }
        if (!keep || other.body == col.body || bodiesJointed(col.body, other.body) ||
            other.shape == SHAPE_PLANE)
            continue;
        if (probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)) continue;
        // Awake pairs keep the (lower, higher) order the full scan emitted.
        if (awakeEntry) emitCandidate(min(i, cur), max(i, cur));
        else emitCandidate(i, cur);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_query(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    if (HOST_S(18) != 0u)
    {
        hierQuery(i);
        return;
    }
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    if (!bodyIsDynamic(b)) return;
    if (col.shape == SHAPE_PLANE) return;
    uint base = HOST_S(1);
    Aabb box = gAabbs[base + i];
    bool sleeper = bodyIsSleeping(b);
    // Probe OBB from build_aabbs cache (three-avbd findPairs loads querier once).
    bool probeIsBox = shapeHasHull(col.shape);
    BoxObb probeObb = probeIsBox ? worldObbFromCache(i) : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    float faceGap = max(0.001, speculativeMargin()) + (probeIsBox ? cachedReach(i) : 0.0);

    // Small scenes: exact pairwise (correctness). Large: uniform-grid hash.
    // Awake dynamics test everyone; sleepers only keep static contacts (ground) so a
    // later impact can wake the island without the floor dropping out for a frame.
    if (gCounters[0].colliderCount <= 512u)
    {
        // while + nested tests, not for/continue: SPIR-V→naga turns `continue` into a
        // switch-break that never exits the loop (GPU hang on WebGPU).
        uint j = 0u;
        while (j < gCounters[0].colliderCount)
        {
            if (j != i)
            {
                Collider other = gColliders[j];
                if (other.body != col.body)
                {
                    BodyState ob = gBodies[other.body];
                    bool otherStatic = !bodyIsDynamic(ob);
                    bool keep = true;
                    if (sleeper && (!otherStatic || carryRestPairs())) keep = false;
                    if (bodyIsSleeping(ob) && sleeper) keep = false;
                    // Lower-index dynamics own the pair, unless asleep: sleepers
                    // never query, so an awake body must emit its sleeping
                    // neighbours at any index (as the grid path does).
                    if (!otherStatic && j < i && !(bodyIsSleeping(ob) && !sleeper)) keep = false;
                    if (otherStatic && !bodyIsDynamic(b)) keep = false;
                    if (bodiesJointed(col.body, other.body)) keep = false;
                    if (keep && other.shape != SHAPE_PLANE)
                    {
                        Aabb ab = gAabbs[base + j];
                        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) keep = false;
                        if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) keep = false;
                        if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) keep = false;
                    }
                    if (keep && probeIsBox && boxesSeparatedOnFaceAxis(probeObb, j, faceGap)) keep = false;
                    if (keep) emitCandidate(i, j);
                }
            }
            j++;
        }
        return;
    }

    float cellSize = max(PARAMS.gridCellSize, 0.5);
    // Cell is 2·maxSmallRadius, so the centre cell ±1 (27) covers every small overlap.
    float3 center = 0.5 * (box.mn + box.mx);
    int3 cc = int3(floor(center / cellSize));

    // Sleepers emit nothing: frozen pose needs no ground manifold, and 32k sleeper–
    // plane candidates dominated settled-pile ms. Awake neighbours still insert into
    // the grid and wake sleepers on impact; a woken body re-acquires the plane next step.
    float rA = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    bool queryLarge = rA > cellSize;
    rA += colliderReach(b, col);
    uint nLarge = min(ALOAD(gScratchAtomics[25]), 16u);
    for (uint k = 0u; k < nLarge; k++)
    {
        uint j = ALOAD(gScratchAtomics[LEGACY_LARGE_BASE + k]);
        if (j == i) continue;
        Collider other = gColliders[j];
        if (other.body == col.body || bodiesJointed(col.body, other.body)) continue;
        BodyState ob = gBodies[other.body];
        // Small→large owns this pair regardless of indices. Only large→large
        // needs index ordering; a large centre-cell query cannot cover its surface.
        if (queryLarge && bodyIsDynamic(ob) && j < i) continue;
        if (sleeper && carryRestPairs() && bodyIsSleeping(ob)) continue; // carried
        Aabb ab = gAabbs[base + j];
        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
        if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
        if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
        emitCandidate(i, j);
    }
    if (sleeper)
        return;
    uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
    for (uint pi = 0u; pi < nPlanes; pi++)
        emitCandidate(i, ALOAD(gScratchAtomics[43u + pi]));
    float3 cA = 0.5 * (box.mn + box.mx);
    if (!queryLarge && gridSplit())
    {
        // Awake table: own cell (higher index owns) + 13 forward cells (all).
        for (uint c = 0u; c < 14u; c++)
        {
            int3 off = gridForwardOffset(c);
            gridScanCell(i, col, b, box, cA, rA, probeIsBox, probeObb, faceGap, cellSize,
                cc + off, 0u, c == 0u);
        }
        // Other table (sleepers, small statics): full stencil, skipped when empty.
        if (ALOAD(gScratchAtomics[52]) != 0u)
        {
            for (int z = cc.z - 1; z <= cc.z + 1; z++)
            for (int y = cc.y - 1; y <= cc.y + 1; y++)
            for (int x = cc.x - 1; x <= cc.x + 1; x++)
                gridScanCell(i, col, b, box, cA, rA, probeIsBox, probeObb, faceGap, cellSize,
                    int3(x, y, z), gridHashSize(), false);
        }
    }
    else if (!queryLarge)
    {
    for (int z = cc.z - 1; z <= cc.z + 1; z++)
    for (int y = cc.y - 1; y <= cc.y + 1; y++)
    for (int x = cc.x - 1; x <= cc.x + 1; x++)
    {
        uint h = gridHash(x, y, z);
        uint link = ALOAD(gConstraintIds[h]);
        // Dense pile hash slots can exceed a few hundred links (32k in ~2 m cells).
        // Truncating at 256 silently dropped pairs → deep C0. Cap high; flag if hit.
        uint guard = 0;
        while (link != 0xFFFFFFFFu && guard < 4096u)
        {
            uint cur = link & GRID_COLLIDER_MASK;
            bool ownedElsewhere = (link & GRID_AWAKE_DYNAMIC) != 0u && cur <= i;
            // Fetch the next dependent link before this candidate's geometry.
            // Lower-index awake dynamics own the reverse query; sleepers do not.
            link = gColorLists[cur];
            guard++;
            if (guard == 4096u)
                AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
            if (ownedElsewhere) continue;
            // Sphere test from the cached AABB (center + radius in _pad0) before
            // touching the collider or body. Same geometric radius as the old test.
            Aabb ab = gAabbs[base + cur];
            // Hash buckets are not cells: two of the 27 queried cells can alias.
            // Visiting only the candidate's true cell prevents duplicate constraints.
            int3 candidateCell = int3(floor((0.5 * (ab.mn + ab.mx)) / cellSize));
            if (candidateCell.x != x || candidateCell.y != y || candidateCell.z != z)
            {
                continue;
            }
            float rB = ab._pad0;
            if (rB > 0.0)
            {
                float3 d = cA - 0.5 * (ab.mn + ab.mx);
                float rr = rA + rB;
                if (dot(d, d) > rr * rr)
                {
                    continue;
                }
            }
            if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x ||
                box.mx.y < ab.mn.y || ab.mx.y < box.mn.y ||
                box.mx.z < ab.mn.z || ab.mx.z < box.mn.z)
            {
                continue;
            }
            // Dynamic-dynamic once (higher index), PLUS awake→sleeping at any index:
            // smash balls are last, so cur>i alone never pairs them with sleeping
            // bricks (sleepers only keep static contacts and would not emit).
            Collider other = gColliders[cur];
            BodyState ob = gBodies[other.body];
            bool otherSleep = bodyIsDynamic(ob) && bodyIsSleeping(ob);
            bool pairOnce = cur > i || (!sleeper && otherSleep);
            if (pairOnce && cur != i)
            {
                if (other.body != col.body &&
                    !bodiesJointed(col.body, other.body) &&
                    other.shape != SHAPE_PLANE)
                {
                    if (!(probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)))
                        emitCandidate(i, cur);
                }
            }
        }
    }
    }
    if (queryLarge)
    {
        // Large dynamics (vehicle chassis, smash balls) do not use the 27-cell
        // stencil. Small dynamics and sleepers find them through the large list;
        // small statics never query, so walk the grid cells under this AABB for them.
        int3 lo = int3(floor(box.mn / cellSize));
        int3 hi = int3(floor(box.mx / cellSize));
        int3 span = hi - lo + int3(1, 1, 1);
        if (span.x * span.y * span.z <= 512)
        {
            uint tableBase = gridSplit() ? gridHashSize() : 0u;
            for (int z = lo.z; z <= hi.z; z++)
            for (int y = lo.y; y <= hi.y; y++)
            for (int x = lo.x; x <= hi.x; x++)
            {
                uint link = ALOAD(gConstraintIds[tableBase + gridHash(x, y, z)]);
                uint guard = 0u;
                while (link != 0xFFFFFFFFu && guard < 4096u)
                {
                    uint cur = link & GRID_COLLIDER_MASK;
                    Aabb ab = gAabbs[base + cur];
                    link = gridSplit() ? asuint(ab._pad1) : gColorLists[cur];
                    guard++;
                    int3 cell = int3(floor((0.5 * (ab.mn + ab.mx)) / cellSize));
                    if (cell.x != x || cell.y != y || cell.z != z) continue;
                    if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x ||
                        box.mx.y < ab.mn.y || ab.mx.y < box.mn.y ||
                        box.mx.z < ab.mn.z || ab.mx.z < box.mn.z)
                        continue;
                    Collider other = gColliders[cur];
                    if (other.body == col.body || other.shape == SHAPE_PLANE) continue;
                    if (bodyIsDynamic(gBodies[other.body])) continue;
                    if (bodiesJointed(col.body, other.body)) continue;
                    if (probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)) continue;
                    emitCandidate(i, cur);
                }
            }
        }
    }
    // Oversized statics are not in the grid. broadphase_insert lists up to
    // LARGE_STATIC_CAP of them; on overflow the legacy endpoint probe (ground at
    // collider 0 or last) applies. Large dynamics were handled exactly once above.
    uint nLargeStatic = ALOAD(gScratchAtomics[LARGE_STATIC_COUNT]);
    uint nListed = largeStaticCount();
    bool listOverflow = nLargeStatic > LARGE_STATIC_CAP;
    uint nProbe = nListed + (listOverflow ? 2u : 0u);
    for (uint t = 0u; t < nProbe; t++)
    {
        if (gCounters[0].colliderCount == 0u) break;
        bool endpoint = t >= nListed;
        uint j = !endpoint ? largeStaticCollider(t)
            : ((t == nListed) ? 0u : (gCounters[0].colliderCount - 1u));
        if (j == i || (endpoint && t > nListed && j == 0u)) continue;
        if (endpoint)
        {
            bool listed = false;
            for (uint u = 0u; u < nListed; u++)
                if (largeStaticCollider(u) == j) listed = true;
            if (listed) continue;
        }
        Collider other = gColliders[j];
        if (other.shape == SHAPE_PLANE || other.body == col.body) continue;
        Aabb ab = gAabbs[base + j];
        float3 ext = ab.mx - ab.mn;
        if (endpoint && ext.x < cellSize * 4.0 && ext.y < cellSize * 4.0 && ext.z < cellSize * 4.0)
            continue;
        // Dynamic-static always from the mover.
        BodyState ob = gBodies[other.body];
        if (bodyIsDynamic(ob)) continue;
        if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
        if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
        if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
        if (probeIsBox && shapeHasHull(other.shape) && boxesSeparatedOnFaceAxis(probeObb, j, faceGap))
            continue;
        // Sphere large (smash ball): radius test.
        if (other.shape == SHAPE_SPHERE)
        {
            float3 d = b.position - ob.position;
            float rr = other.params.x + (col.params.w > 0.0 ? col.params.w : length(col.params.xyz))
                + colliderReach(b, col) + colliderReach(ob, other);
            if (dot(d, d) > rr * rr) continue;
        }
        emitCandidate(i, j);
    }
}

// ---- Awake-only queries (JS host, moved-only bounds: HOST_S(19) != 0) ----
// Every moving collider then belongs to a body on last step's awake list
// (HOST_S(6) entries; nothing wakes between that list and this broadphase
// without a host write, which turns moved-only bounds off), so only those
// query: hierQuery with an active querier emits exactly the pairs it always
// did. The pairs a resting (sleeping or static) collider found from its own
// thread, with a coarser mover, are found by that mover instead: it scans the
// finer cells for resting colliders (hierScanFiner), replaying the resting
// collider's own acceptance test. emitCandidate orders every pair (min, max),
// so the candidate set is the same; only its append order differs, and every
// consumer is order-independent (contacts are keyed and CSR rows sorted).
// A coarse mover's finer cells are split into chunks of 64 (one lane each):
// the awake query appends one work item (mover, chunk) per chunk to a list in
// gConstraintIds between the grid heads (HOST_S(16) words) and the CSR base
// (HOST_S(24)), unused until the CSR build; the item count is scratch word 5
// (host-zeroed; HOST_S(5) is unused). broadphase_query_finer runs one
// workgroup per item, so a large mover's scan spreads over many SMs.
static const uint FINER_COUNT_SLOT = 5u;
static const uint FINER_CHUNK = 64u;
static const uint FINER_SKIP = 0xFFFFFFFFu;

uint finerItemCapacity() { return (HOST_S(24) - HOST_S(16)) / 2u; }

// Cells of `level` that can hold the centre of a collider overlapping `ma`.
// Level-L colliders reach at most cs/2 (plus the AABB pad, which _pad0 leaves
// out) from their centre.
uint3 finerSpan(Aabb ma, uint level, out int3 lo)
{
    float cs = hierCellSize(level);
    float reach = 0.5 * cs + max(1.0e-4, speculativeMargin());
    lo = int3(floor((ma.mn - reach) / cs));
    int3 hi = int3(floor((ma.mx + reach) / cs));
    return uint3(hi - lo + 1);
}

uint finerLevelMask(uint own)
{
    return ALOAD(gScratchAtomics[HIER_LEVEL_MASK_SLOT]) & ((1u << own) - 1u);
}

uint hierFinerCellCount(Aabb ma, uint own)
{
    uint mask = finerLevelMask(own), total = 0u;
    for (uint level = 0u; level < own; level++)
    {
        if (((mask >> level) & 1u) == 0u) continue;
        int3 lo;
        uint3 span = finerSpan(ma, level, lo);
        total += span.x * span.y * span.z;
    }
    return total;
}

// Pairs of coarse mover m with resting colliders at finer levels, over the
// finer cells [first, end) (every level's cells in turn) with step `stride`.
// A resting collider j at level L emitted (j, m) from hierQuery iff they
// overlap and m's cell lies in j's range at m's level; the same test runs
// here on the same values, so the accepted pairs are identical. Every j that
// overlaps m has its centre in one of the cells finerSpan lists.
void hierScanFiner(uint m, uint first, uint stride, uint end)
{
    uint base = HOST_S(1);
    Aabb ma = gAabbs[base + m];
    uint own = hierLevel(ma._pad0);
    Collider cm = gColliders[m];
    bool probeHasObb = shapeHasHull(cm.shape);
    BoxObb probe = probeHasObb ? worldObbFromCache(m)
                               : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    float faceGap = max(0.001, speculativeMargin()) + (probeHasObb ? cachedReach(m) : 0.0);
    uint mask = finerLevelMask(own);
    uint count = gCounters[0].colliderCount;
    float csM = hierCellSize(own);
    int3 cellM = hierCell(ma, own);
    for (uint t = first; t < end; t += stride)
    {
        uint rem = t, level = 0u;
        int3 lo = int3(0, 0, 0);
        uint3 span = uint3(1, 1, 1);
        for (; level < own; level++)
        {
            if (((mask >> level) & 1u) == 0u) continue;
            span = finerSpan(ma, level, lo);
            uint n = span.x * span.y * span.z;
            if (rem < n) break;
            rem -= n;
        }
        if (level >= own) break;
        int3 cell = lo + int3(uint3(rem % span.x, (rem / span.x) % span.y, rem / (span.x * span.y)));
        // Resting colliders: the grid, or only R (W holds movers alone).
        bool rest = restTables();
        uint j = ALOAD(gConstraintIds[(rest ? restHeads() : 0u) + hierHash(cell, level)]);
        uint guard = 0u;
        while (j != 0xFFFFFFFFu && guard < count)
        {
            guard++;
            Aabb ab = gAabbs[base + j];
            uint next = rest ? ALOAD(gConstraintIds[restLinks() + j]) : asuint(ab._pad1);
            if (aabbOverlap(ma, ab) && hierLevel(ab._pad0) == level &&
                all(hierCell(ab, level) == cell))
            {
                // j's query range at m's level, as hierQuery computes it.
                int3 jc = hierCell(ab, own);
                int3 jlo = max(int3(floor((ab.mn - 0.5 * csM) / csM)), jc - 1);
                int3 jhi = min(int3(floor((ab.mx + 0.5 * csM) / csM)), jc + 1);
                if (all(cellM >= jlo) && all(cellM <= jhi))
                {
                    Collider b = gColliders[j];
                    if (b.body != cm.body && !bodyMoving(gBodies[b.body]) &&
                        !bodiesJointed(cm.body, b.body) &&
                        !(probeHasObb && boxesSeparatedOnFaceAxis(probe, j, faceGap)))
                        emitCandidate(j, m);
                }
            }
            j = next;
        }
        if (j != 0xFFFFFFFFu) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
    }
}

// One thread per listed awake collider (indirect, work slot 81).
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_query_awake(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    uint levels = HOST_S(18);
    hierQuery(c);
    if (gColliders[c].shape == SHAPE_PLANE) return;
    Aabb aa = gAabbs[HOST_S(1) + c];
    uint own = hierLevel(aa._pad0);
    if (own == 0u || own >= levels) return;
    uint cells = hierFinerCellCount(aa, own);
    if (cells == 0u) return;
    uint chunks = (cells + FINER_CHUNK - 1u) / FINER_CHUNK;
    uint k;
    AADD3(gScratchAtomics[FINER_COUNT_SLOT], chunks, k);
    uint cap = finerItemCapacity();
    bool fits = k + chunks <= cap;
    for (uint q = 0u; q < chunks && k + q < cap; q++)
    {
        ASTORE(gConstraintIds[HOST_S(16) + 2u * (k + q)], fits ? c : FINER_SKIP);
        ASTORE(gConstraintIds[HOST_S(16) + 2u * (k + q) + 1u], q);
    }
    if (!fits) hierScanFiner(c, 0u, 1u, cells); // list full: scan in this thread
}

// Colliders of moving bodies on last step's awake list (one thread per entry,
// work slot 78), so the awake insert, query and clear run one thread per
// collider: a body with many colliders no longer serializes them in one thread.
// Listed in gConstraintIds after the resting links (awakeColliderBase); the
// count is scratch word 6 (host-zeroed; HOST_S(6) lives in the uniform).
static const uint AWAKE_COLLIDER_COUNT_SLOT = 6u;
uint awakeColliderBase() { return restLinks() + gCounters[0].colliderCount; }
uint awakeColliderCount() { return ALOAD(gScratchAtomics[AWAKE_COLLIDER_COUNT_SLOT]); }
uint awakeCollider(uint li) { return ALOAD(gConstraintIds[awakeColliderBase() + li]); }

[shader("compute")]
[numthreads(64, 1, 1)]
void list_awake_colliders(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= HOST_S(6)) return;
    uint body = gAwakeList[li];
    if (!bodyMoving(gBodies[body])) return;
    uint n = gCounters[0].colliderCount;
    uint first = firstColliderOfBody(body), end = first;
    while (end < n && gColliders[end].body == body) end++;
    if (end == first) return;
    uint base;
    AADD3(gScratchAtomics[AWAKE_COLLIDER_COUNT_SLOT], end - first, base);
    for (uint c = first; c < end; c++) ASTORE(gConstraintIds[awakeColliderBase() + base + (c - first)], c);
}

// Work slot 81: one thread per listed awake collider.
[shader("compute")]
[numthreads(64, 1, 1)]
void write_awake_collider_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    writeDispatchArgs(81u, awakeColliderCount());
}

// Work slot 77: one workgroup per listed chunk.
[shader("compute")]
[numthreads(64, 1, 1)]
void write_finer_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    writeDispatchArgs(77u, min(ALOAD(gScratchAtomics[FINER_COUNT_SLOT]), finerItemCapacity()) * 64u);
}

// Work slot 78: last step's awake list (HOST_S(6) entries), for moved-only
// bounds and awake-only queries. Slot 67 is reused by the sleep pass.
// Slots 79/80: the resting-table rebuild (clear, insert), dispatched only
// when last step put bodies to sleep (HOST_S(5) = its sleep events) or the
// host asks (HOST_S(25) bit 1).
[shader("compute")]
[numthreads(64, 1, 1)]
void write_moved_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    writeDispatchArgs(78u, HOST_S(6));
    bool rebuild = restTables() && (HOST_S(5) != 0u || (HOST_S(25) & 2u) != 0u);
    writeDispatchArgs(79u, rebuild ? REST_HEADER + 2u * gridHashSize() : 0u);
    writeDispatchArgs(80u, rebuild ? gCounters[0].colliderCount : 0u);
}

// Rebuild 1: empty both tables and the header.
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= REST_HEADER + 2u * gridHashSize()) return;
    ASTORE(gConstraintIds[restBase() + i], i < REST_HEADER ? 0u : 0xFFFFFFFFu);
}

// Rebuild 2: every collider not moving now goes into R (planes and oversized
// statics into the header lists, as hierInsert lists them each step).
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_insert(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    Collider col = gColliders[i];
    if (bodyMoving(gBodies[col.body])) return;
    uint hdr = restBase();
    if (col.shape == SHAPE_PLANE)
    {
        uint slot;
        AADD3(gConstraintIds[hdr + 1u], 1u, slot);
        if (slot < 8u) ASTORE(gConstraintIds[hdr + 2u + slot], i);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    Aabb box = gAabbs[HOST_S(1) + i];
    uint level = hierLevel(box._pad0);
    if (level >= HOST_S(18))
    {
        uint slot;
        AADD3(gConstraintIds[hdr + 10u], 1u, slot);
        if (slot < HIER_LARGE_MAX) ASTORE(gConstraintIds[hdr + 11u + slot], i);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    AOR2(gConstraintIds[hdr], 1u << level);
    uint prev;
    AXCHG3(gConstraintIds[restHeads() + hierHash(hierCell(box, level), level)], i, prev);
    ASTORE(gConstraintIds[restLinks() + i], prev);
}

// Each step, before the awake inserts: R's levels, planes and oversized
// statics into the scratch words hierQuery reads (hierInsert's lists).
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_publish(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    uint hdr = restBase();
    AOR2(gScratchAtomics[HIER_LEVEL_MASK_SLOT], ALOAD(gConstraintIds[hdr]));
    uint planes = ALOAD(gConstraintIds[hdr + 1u]);
    ASTORE(gScratchAtomics[42], planes);
    for (uint k = 0u; k < min(planes, 8u); k++)
        ASTORE(gScratchAtomics[43u + k], ALOAD(gConstraintIds[hdr + 2u + k]));
    uint large = ALOAD(gConstraintIds[hdr + 10u]);
    ASTORE(gScratchAtomics[HIER_LARGE_COUNT_SLOT], large);
    for (uint k = 0u; k < min(large, HIER_LARGE_MAX); k++)
        ASTORE(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + k], ALOAD(gConstraintIds[hdr + 11u + k]));
}

// Listed awake colliders into W (one thread each, slot 81).
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_insert_awake(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    if (gColliders[c].shape == SHAPE_PLANE) return;
    Aabb box = gAabbs[HOST_S(1) + c];
    uint level = hierLevel(box._pad0);
    if (level >= HOST_S(18))
    {
        uint slot;
        AADD3(gScratchAtomics[HIER_LARGE_COUNT_SLOT], 1, slot);
        if (slot < HIER_LARGE_MAX) ASTORE(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + slot], c);
        else AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        return;
    }
    AOR2(gScratchAtomics[HIER_LEVEL_MASK_SLOT], 1u << level);
    uint prev;
    AXCHG3(gConstraintIds[awakeHeads() + hierHash(hierCell(box, level), level)], c, prev);
    gAabbs[HOST_S(1) + c]._pad1 = asfloat(prev);
}

// After the queries: empty the W heads the listed colliders filled (slot 81).
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_clear_awake(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    if (gColliders[c].shape == SHAPE_PLANE) return;
    Aabb box = gAabbs[HOST_S(1) + c];
    uint level = hierLevel(box._pad0);
    if (level < HOST_S(18))
        ASTORE(gConstraintIds[awakeHeads() + hierHash(hierCell(box, level), level)], 0xFFFFFFFFu);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_query_finer(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint g = gid.x + gid.y * 32768u;
    if (g >= min(ALOAD(gScratchAtomics[FINER_COUNT_SLOT]), finerItemCapacity())) return;
    uint m = ALOAD(gConstraintIds[HOST_S(16) + 2u * g]);
    if (m == FINER_SKIP) return;
    uint t = ALOAD(gConstraintIds[HOST_S(16) + 2u * g + 1u]) * FINER_CHUNK + lid.x;
    hierScanFiner(m, t, FINER_CHUNK, t + 1u);
}

// Back-compat name used by CUDA/WebGPU build lists.
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_grid(uint3 tid : SV_DispatchThreadID)
{
    broadphase_query(tid);
}

// Emit one contact. C0 matches three-avbd: n·(xB−xA)+margin with n = A→B (compression ⇒ C0 < 0),
// or the bare gap under FLAG_CONTACT_GAP (see contactC0N).
Contact makeContactRow(
    CandidatePair pair, BodyState ba, BodyState bb, Collider ca, Collider cb,
    float3 nrm, float3 pA, float3 pB, uint featureId, float margin, float mu)
{
    float3 t1, t2;
    orthonormalBasis(nrm, t1, t2);
    // three-avbd: C0_n = n·(xB−xA) + margin (A→B). Equilibrium sits ~margin deep.
    float3 d0 = pB - pA;
    float C0n = contactC0N(dot(nrm, d0), margin);
    float C0t1 = dot(t1, d0);
    float C0t2 = dot(t2, d0);

    Contact ct;
    ct.colliderA = pair.colliderA;
    ct.colliderB = pair.colliderB;
    ct.featureId = featureId;
    ct.alive = 1;
    ct.normal = nrm;
    ct.penetration = C0n;
    ct.anchorA = quatRotate(quatConjugate(ba.orientation), pA - ba.position);
    ct.c0N = C0n;
    ct.anchorB = quatRotate(quatConjugate(bb.orientation), pB - bb.position);
    ct.friction = mu;
    ct.tangent1 = t1;
    ct.lambdaN = 0.0;
    ct.tangent2 = t2;
    ct.lambdaT1 = 0.0;
    // Cold start like three-avbd: PENALTY_MIN=1, then persist applies massPenalty
    // (k ≥ m/dt²). Flooring with kStart=1000 over-stiffened light Box Pile boxes
    // on every warm-start miss and pumped energy.
    ct.kN = 1.0;
    ct.kT = 1.0;
    ct.lambdaT2 = 0.0;
    ct.c0T1 = C0t1;
    ct.c0T2 = C0t2;
    ct._pad0 = 0.0; // stick flag (1 = static friction held)
    ct._pad1 = asfloat(HOST_S(17)); // generated step (reuseContacts)
    ct._pad2 = 1.0; // independent second tangent penalty
    return ct;
}

// Track belt (conveyor) surface: a collider on a vehicle chassis with a non-zero
// `density` is a belt whose surface runs at `density` m/s around its local +X axle
// (vehicle chassis colliders carry no density; vehicle_post writes the speed).
// The surface velocity at a contact with outward normal nOut is speed·(axle × nOut):
// backward on the bottom, down the front, forward on top — a track loop.
bool colliderIsBelt(Collider col, BodyState b)
{
    return (b.flags & BODY_VEHICLE) != 0u && col.density != 0.0;
}

// Same test loading the body only for a collider with density (rare: belts, and
// colliders the inertia pass reads).
bool colliderIsBeltLazy(Collider col)
{
    return col.density != 0.0 && (gBodies[col.body].flags & BODY_VEHICLE) != 0u;
}

float3 beltSurfaceDisp(Collider col, BodyState b, float3 nOut, float dt)
{
    if (!colliderIsBelt(col, b)) return float3(0, 0, 0);
    float3 axle = quatRotate(b.initialRot, quatRotate(col.localRot, float3(1, 0, 0)));
    return col.density * dt * cross(axle, nOut);
}

// Relative belt travel of a pair this step (n = A→B): the surfaces move by
// d + belt, so the tangent rows hold t·(dB − dA + sB − sA) at zero.
float3 pairBeltDisp(Collider ca, Collider cb, BodyState ba, BodyState bb, float3 n)
{
    float dt = PARAMS.dt;
    return beltSurfaceDisp(cb, bb, -n, dt) - beltSurfaceDisp(ca, ba, n, dt);
}

void wakeFromContact(BodyState ba, BodyState bb, Collider ca, Collider cb)
{
    // Wake only a sleeping body hit by an awake neighbour with meaningful relative
    // speed (island wake). Resting pile contacts used to thrash sleep every NP frame
    // (~1.5k sleeping / 31k awake despite KE~50). Never clear SLEEPING on calm pairs.
    bool awakeA = bodyIsDynamic(ba) && !bodyIsSleeping(ba);
    bool awakeB = bodyIsDynamic(bb) && !bodyIsSleeping(bb);
    // Displacement sleep: a body that has itself stayed put for sleepTime (a
    // sleep candidate) only jitters in place; it cannot wake its neighbours.
    awakeA = awakeA && canWake(ba);
    awakeB = awakeB && canWake(bb);
    if (!awakeA && !awakeB) return;
    float3 vRel = ba.linearVel - bb.linearVel;
    float wRel = length(ba.angularVel) + length(bb.angularVel);
    float thresh = PARAMS.sleepLin;
    if (pointSpeedSleep())
    {
        float pointSpeed = length(vRel) + length(ba.angularVel) * colliderExtent(ca) + length(bb.angularVel) * colliderExtent(cb);
        if (pointSpeed < thresh) return;
    }
    else if (dot(vRel, vRel) < thresh * thresh && wRel < PARAMS.sleepAng)
        return;
    // Store only the wake fields: a whole-body store from this thread's
    // narrowphase-start copy could overwrite fields other work updated.
    if (bodyIsDynamic(ba) && bodyIsSleeping(ba) && awakeB)
    {
        gBodies[ca.body].flags = wokenFlags(ba.flags);
        gBodies[ca.body].sleepTimer = 0.0;
    }
    if (bodyIsDynamic(bb) && bodyIsSleeping(bb) && awakeA)
    {
        gBodies[cb.body].flags = wokenFlags(bb.flags);
        gBodies[cb.body].sleepTimer = 0.0;
    }
}

// three-avbd Stage 8b: one manifold record + contiguous contact points.
#if defined(ENGINE_REUSE_COPY)
// Opt-in persist work list (8em, HOST_S(22) != 0). Contacts that still need the
// persistence pass (fresh points and reused pairs at rest) are listed, eight
// local indices per record, in the unused top of gManifolds: manifold slots
// never exceed the candidate count, so records [n, maxM) are free during
// narrowphase. gCounters._pad2 counts listed contacts. Once it exceeds the
// capacity the list is ignored and the full per-contact scan runs instead, so
// every contact is persisted exactly once in either mode.
bool persistListEnabled() { return HOST_S(22) != 0u; }

uint persistListCapacity()
{
    uint maxM = HOST_S(10);
    uint capacity = (maxM - min(ALOAD(gCounters[0].candidateCount), maxM)) * 8u;
    // HOST_S(23) != 0 caps the list (tests exercise the overflow scan).
    return HOST_S(23) != 0u ? min(capacity, HOST_S(23)) : capacity;
}

void persistListWrite(uint slot, uint value)
{
    uint rec = HOST_S(10) - 1u - (slot >> 3u);
    switch (slot & 7u)
    {
    case 0u: gManifolds[rec].colliderA = value; break;
    case 1u: gManifolds[rec].colliderB = value; break;
    case 2u: gManifolds[rec].contactStart = value; break;
    case 3u: gManifolds[rec].contactCount = value; break;
    case 4u: gManifolds[rec].normal.x = asfloat(value); break;
    case 5u: gManifolds[rec].normal.y = asfloat(value); break;
    case 6u: gManifolds[rec].normal.z = asfloat(value); break;
    default: gManifolds[rec].friction = asfloat(value); break;
    }
}

uint persistListRead(uint slot)
{
    Manifold m = gManifolds[HOST_S(10) - 1u - (slot >> 3u)];
    switch (slot & 7u)
    {
    case 0u: return m.colliderA;
    case 1u: return m.colliderB;
    case 2u: return m.contactStart;
    case 3u: return m.contactCount;
    case 4u: return asuint(m.normal.x);
    case 5u: return asuint(m.normal.y);
    case 6u: return asuint(m.normal.z);
    default: return asuint(m.friction);
    }
}

// Lists local contact indices [first, first + count) for the persistence pass.
void persistListAppend(uint first, uint count)
{
    if (!persistListEnabled()) return;
    uint k;
    AADD3(gCounters[0]._pad2, count, k);
    if (k + count > persistListCapacity()) return; // overflow: the scan pass runs
    for (uint c = 0u; c < count; c++)
        persistListWrite(k + c, first + c);
}
#endif

// Look-ahead keeps a separated point only if the pair can close that gap this
// step: the relative normal velocity at the point, (vA + ωA×rA − vB − ωB×rB)·n,
// over h. Points moving apart or sliding past each other get no speculative
// contact (and no graph edge); touching points (gap ≤ contactOffset) are kept.
BoxManifold closingPoints(BoxManifold points, BodyState ba, Collider ca, BodyState bb, Collider cb)
{
    bool moveA = bodyIsDynamic(ba) && !bodyIsSleeping(ba);
    bool moveB = bodyIsDynamic(bb) && !bodyIsSleeping(bb);
    float h = PARAMS.dt;
    float3 n = points.normal;
    BoxManifold kept = points;
    kept.count = 0;
    for (int k = 0; k < points.count; k++)
    {
        float3 pA = points.pts[k].pA, pB = points.pts[k].pB;
        float3 vA = moveA ? ba.linearVel + cross(ba.angularVel, pA - ba.position) : float3(0, 0, 0);
        float3 vB = moveB ? bb.linearVel + cross(bb.angularVel, pB - bb.position) : float3(0, 0, 0);
        float reach = PARAMS.contactOffset + max(dot(vA - vB, n), 0.0) * h + abs(PARAMS.gravityY) * h * h;
        if (dot(n, pB - pA) <= reach) { kept.pts[kept.count] = points.pts[k]; kept.count++; }
    }
    return kept;
}

void emitContactManifold(
    CandidatePair pair, BodyState ba, BodyState bb, Collider ca, Collider cb,
    BoxManifold points, float margin)
{
    if ((PARAMS.flags & FLAG_SPEC_VELOCITY) != 0u) points = closingPoints(points, ba, ca, bb, cb);
    int count = points.count;
    float3 nrm = points.normal;
    if (count <= 0) return;
    uint maxC = HOST_S(3);
    uint maxM = HOST_S(10);
    uint mslot;
    AADD3(gCounters[0].manifoldCount, 1, mslot);
    if (mslot >= maxM)
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_MANIFOLDS);
        return;
    }
    uint cStart;
    AADD3(gCounters[0].contactCount, (uint)count, cStart);
    uint curOff = HOST_S(11);
    if (cStart >= maxC || cStart + (uint)count > maxC)
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_CONTACTS);
        // Zero any slots this thread reserved so persist/CSR never see garbage.
        for (uint k = 0u; k < (uint)count && cStart + k < maxC; k++)
        {
            Contact z;
            z.colliderA = 0; z.colliderB = 0; z.featureId = 0; z.alive = 0;
            z.normal = float3(0, 0, 0); z.penetration = 0.0;
            z.anchorA = float3(0, 0, 0); z.c0N = 0.0;
            z.anchorB = float3(0, 0, 0); z.friction = 0.0;
            z.tangent1 = float3(0, 0, 0); z.lambdaN = 0.0;
            z.tangent2 = float3(0, 0, 0); z.lambdaT1 = 0.0;
            z.kN = 0.0; z.kT = 0.0; z.lambdaT2 = 0.0;
            z.c0T1 = 0.0; z.c0T2 = 0.0; z._pad0 = 0.0; z._pad1 = 0.0; z._pad2 = 0.0;
            STORE_NEW_CONTACT(curOff + cStart + k, mslot, z);
        }
        Manifold dead;
        dead.colliderA = pair.colliderA;
        dead.colliderB = pair.colliderB;
        dead.contactStart = 0;
        dead.contactCount = 0;
        dead.normal = nrm;
        dead.friction = 0.0;
        gManifolds[mslot] = dead;
        return;
    }

    Material ma = gMaterials[materialIndex(ca)];
    Material mb = gMaterials[materialIndex(cb)];
    float mu = sqrt(max(ma.friction, 0.0) * max(mb.friction, 0.0));

    for (int k = 0; k < count; k++)
    {
        BoxManifoldPt pt = points.pts[k];
        STORE_NEW_CONTACT(curOff + cStart + k, mslot, makeContactRow(pair, ba, bb, ca, cb, nrm, pt.pA, pt.pB, pt.feature, margin, mu));
    }
#if defined(ENGINE_REUSE_COPY)
    persistListAppend(cStart, (uint)count);
#endif

    Manifold mf;
    mf.colliderA = pair.colliderA;
    mf.colliderB = pair.colliderB;
    mf.contactStart = curOff + cStart;
    mf.contactCount = (uint)count;
    mf.normal = nrm;
    mf.friction = mu;
    gManifolds[mslot] = mf;
    wakeFromContact(ba, bb, ca, cb);
}

void emitManifold(CandidatePair pair, BodyState ba, BodyState bb, Collider ca, Collider cb, BoxManifold mf, float margin)
{
    emitContactManifold(pair, ba, bb, ca, cb, mf, margin);
}

void emitContact(
    CandidatePair pair, BodyState ba, BodyState bb, Collider ca, Collider cb,
    float3 nrm, float3 pA, float3 pB, uint featureId, float margin)
{
    BoxManifold points;
    points.normal = nrm;
    points.count = 1;
    points.pts[0].pA = pA;
    points.pts[0].pB = pB;
    points.pts[0].feature = featureId;
    emitContactManifold(pair, ba, bb, ca, cb, points, margin);
}

// Contact warm-start hash helpers (pair → first prev-contact index).
static const uint CONTACT_HASH_EMPTY = 0xFFFFFFFFu;

// Tick-tagged warm hash (HOST_S(25) bit 2, JS host): the per-contact hash gets
// its own region past the CSR index (the default one, gConstraintIds[0..),
// is overwritten by the grid and the CSR every step), and each slot word is
// (tag << vb) | j with vb = log2(max contacts) and a tag per step. A word
// without this step's tag reads as EMPTY, so the table behaves exactly as one
// cleared this step; the host clears it only when the tags wrap around.
bool warmTagged() { return (HOST_S(25) & 4u) != 0u; }
uint warmBase() { return warmTagged() ? HOST_S(24) + HOST_S(1) + 1u : 0u; }
uint warmValueBits() { return firstbithigh(HOST_S(3)); }
uint warmTag()
{
    uint period = (1u << (32u - warmValueBits())) - 2u;
    return 1u + HOST_S(17) % period;
}
// Slot word -> previous-contact index, or EMPTY.
uint warmSlotValue(uint w)
{
    if (!warmTagged()) return w;
    uint vb = warmValueBits();
    return (w >> vb) == warmTag() ? (w & ((1u << vb) - 1u)) : CONTACT_HASH_EMPTY;
}
uint warmSlotWord(uint j) { return warmTagged() ? (warmTag() << warmValueBits()) | j : j; }
static const float NEAREST_FRACTION = 0.05;

uint pairWarmKey(uint colA, uint colB)
{
    // Finalize with a full-avalanche mixer (murmur3 fmix32): the slot is the low
    // bits, and the plain prime-xor key left Morton-ordered neighbour pairs
    // (three-avbd's spatialSort builders) in clusters longer than the 64-slot
    // linear probe even at 25% load, overflowing the record hash on gables506k.
    uint h = colA * 0x9E3779B1u ^ (colB + 0x7F4A7C15u) * 0x85EBCA77u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h == CONTACT_HASH_EMPTY ? (colA ^ colB ^ 0xA5A5A5A5u) : h;
}

// Record warm hash (HOST_S(15) != 0): HOST_S(9) slots of four words in
// gConstraintIds — claim (previous manifold + 1, unique so weak-CAS success is
// unambiguous), collider A, collider B, and run start | run length << 28. A
// lookup reads one slot line per probe instead of loading previous contacts.
static const uint WARM_RUN_SHIFT = 28u;
bool warmRecords() { return HOST_S(15) != 0u; }

bool findWarmRecord(uint colA, uint colB, out uint startJ, out uint count)
{
    startJ = CONTACT_HASH_EMPTY; count = 0u;
    uint slots = HOST_S(9);
    uint h = pairWarmKey(colA, colB) & (slots - 1u);
    // Scan to EMPTY and keep the lowest run start, matching the per-contact
    // hash's minimum-head merge if a pair ever owned two previous manifolds.
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint w = ((h + probe) & (slots - 1u)) * 4u;
        if (ALOAD(gConstraintIds[w]) == CONTACT_HASH_EMPTY) break;
        if (ALOAD(gConstraintIds[w + 1u]) == colA && ALOAD(gConstraintIds[w + 2u]) == colB)
        {
            uint run = ALOAD(gConstraintIds[w + 3u]);
            uint j = run & ((1u << WARM_RUN_SHIFT) - 1u);
            if (j < startJ)
            {
                startJ = j;
                count = run >> WARM_RUN_SHIFT;
            }
        }
    }
    return startJ != CONTACT_HASH_EMPTY;
}

float colliderMinSide(Collider col)
{
    if (col.shape == SHAPE_SPHERE)
        return max(col.params.x * 2.0, 1e-6);
    // Convex params.xyz are (tagged base, scale, 0); params.w bounds the hull.
    if (col.shape == SHAPE_CONVEX)
        return max(col.params.w, 1e-6);
    float3 h = col.params.xyz;
    return max(2.0 * min(h.x, min(h.y, h.z)), 1e-6);
}

// Farthest point of a collider from its body's origin (centre of mass).
float colliderExtent(Collider col)
{
    float r = col.shape == SHAPE_SPHERE ? col.params.x
        : col.shape == SHAPE_CAPSULE ? col.params.x + col.params.y
        : col.shape == SHAPE_CONVEX ? col.params.w
        : length(col.params.xyz);
    return length(col.localPos) + r;
}

static const uint FLAG_REUSE_CONTACTS = 1u;
// Preload: bonds carry load but never break (hosts settle a structure under its
// own weight before it is put to sleep, so it wakes in static equilibrium).
static const uint FLAG_BOND_PRELOAD = 2u;
// params.flags bit 5: contacts target zero gap. The margin (contactOffset) only
// widens detection, so a contact exists before touch and acts at touch. Without
// it (three-avbd), C0 = gap + margin and resting bodies sit ~margin deep.
static const uint FLAG_CONTACT_GAP = 32u;

// Detection margin for contacts that exist before touch (FLAG_CONTACT_GAP):
// bounds, SAT and clipping keep pairs/points whose gap is at most this.
float speculativeMargin()
{
    return contactGapMode() ? PARAMS.contactOffset : 0.0;
}

// params.flags bit 14: look-ahead (speculative) contacts sized to motion. Each
// awake collider's bounds are swept by how far it can move this step, and a
// pair keeps contacts whose gap is under the sum of both reaches, so pieces
// meet at contact depth instead of passing up to |v|·h into each other first
// (PhysX speculative CCD; Box2D/Jolt speculative distance). Rest depth is
// unchanged; a positive C0 is clearance the solver lets the pair close, so it
// is never scaled down by α (contactC0NReg).
static const uint FLAG_SPEC_VELOCITY = 16384u;
static const float SPEC_REACH_MAX = 0.5;
// The cap a host sets (HOST_S(31), float bits; 0: SPEC_REACH_MAX).
float specReachMax()
{
    uint v = HOST_S(31);
    return v != 0u ? asfloat(v) : SPEC_REACH_MAX;
}
bool contactGapMode() { return (PARAMS.flags & FLAG_CONTACT_GAP) != 0u; }

// How far any point of collider `c` on body `b` moves this step:
// (|v| + |ω|·r)·h + |g|·h², from the velocity at the start of the step. Motion
// within the contact depth (contactOffset) is already caught by ordinary
// contacts, so slower colliders get no look-ahead and rest exactly as before.
float colliderReach(BodyState b, Collider c)
{
    if ((PARAMS.flags & FLAG_SPEC_VELOCITY) == 0u || !bodyIsDynamic(b) || bodyIsSleeping(b)) return 0.0;
    // Spin pads the box all round (|ω|·r·h); with the rotation limit
    // (FLAG_MAX_ROTATION) that is at most 0.785 r. Without spin, tumbling
    // debris jammed into neighbours (7x7 grid: 3,869 awake at 30 s vs 6).
    float h = PARAMS.dt;
    float linear = length(b.linearVel) * h;
    float motion = linear + length(b.angularVel) * colliderExtent(c) * h;
    if (motion <= PARAMS.contactOffset) return 0.0;
    return min(motion + abs(PARAMS.gravityY) * h * h, specReachMax());
}
// Reach of a hull or box collider, cached with its world OBB by build_aabbs.
// Bounds and early-outs use these per-collider reaches (a conservative sweep).
float cachedReach(uint c) { return gWorldObbs[c].center.w; }
// How far the pair can close this step: relative speed plus both spins, so
// pieces carried along together (debris pushed by the ball) get no look-ahead.
float pairReach(BodyState ba, Collider ca, BodyState bb, Collider cb)
{
    if ((PARAMS.flags & FLAG_SPEC_VELOCITY) == 0u) return 0.0;
    bool moveA = bodyIsDynamic(ba) && !bodyIsSleeping(ba);
    bool moveB = bodyIsDynamic(bb) && !bodyIsSleeping(bb);
    float3 vA = moveA ? ba.linearVel : float3(0, 0, 0);
    float3 vB = moveB ? bb.linearVel : float3(0, 0, 0);
    // Spin counts here (the narrowphase margin): closingPoints keeps only the
    // points whose ω×r velocity actually closes the gap. The broadphase sweep
    // stays linear (colliderReach), so spin does not flood the candidates.
    float spin = (moveA ? length(ba.angularVel) * colliderExtent(ca) : 0.0)
               + (moveB ? length(bb.angularVel) * colliderExtent(cb) : 0.0);
    float h = PARAMS.dt;
    float motion = (length(vB - vA) + spin) * h;
    if (motion <= PARAMS.contactOffset) return 0.0;
    return min(motion + abs(PARAMS.gravityY) * h * h, 2.0 * specReachMax());
}
// Gap under which a pair keeps (speculative) contacts.
float pairMargin(BodyState ba, Collider ca, BodyState bb, Collider cb)
{
    return speculativeMargin() + pairReach(ba, ca, bb, cb);
}
float pairMarginIds(uint a, uint b)
{
    Collider ca = gColliders[a];
    Collider cb = gColliders[b];
    return pairMargin(gBodies[ca.body], ca, gBodies[cb.body], cb);
}
// params.flags bit 6: the inertial block uses the world inertia R·diag(I)·Rᵀ
// (R at the inertial target). Without it (three-avbd) the principal moments are
// applied on world axes, which is only right for isotropic or axis-aligned bodies.
static const uint FLAG_WORLD_INERTIA = 64u;
// Experiments (sleep investigation, 2026-10-16):
// 128 Coulomb stick: a contact keeps its friction anchors while its unclamped
//     friction force is inside the cone and the anchor offset is under STICK_TOL
//     (avbd-demo2d), instead of requiring a 10 µm end-of-step tangential residual.
// 256 Tangential penalties floored at the pair's mass stiffness, as the normal is.
// 1024 Sliding rows use the secant stiffness μ|f_N|/|C_T| in the Hessian.
static const uint FLAG_COULOMB_STICK = 128u;
static const uint FLAG_TANGENT_MASS_FLOOR = 256u;
static const uint FLAG_SLIDE_HESSIAN = 1024u;
static const float STICK_TOL = 0.01;
// 2048 Load-aware penalty floor: a warm-started contact's resting offset is
//     ≈ |λ|/k (λ decays by αγ each step and k·C rebuilds it), so k ≥ |λ|/REST_SLOP
//     bounds the resting overlap (normal) and anchor give (tangent) by REST_SLOP.
static const uint FLAG_LOAD_PENALTY = 2048u;
static const float REST_SLOP = 0.001;
// 4096 Post-stabilization (avbd-demo2d): warm start keeps λ whole, the
//     iterations run with α = 1 (no correction of step-start error), velocity is
//     taken after them, then one extra primal pass with α = 0 removes the whole
//     position error without turning it into velocity. Replaces α's decay of λ,
//     whose steady state rests every loaded contact ≈ |λ|/k deep. Contacts
//     only: joints and elastic bonds keep α (a spring's force comes from its
//     whole deformation, not this step's change; post-stabilizing bonds blew
//     bonded houses apart).
static const uint FLAG_POST_STAB = 4096u;
// 8192 Point-speed sleep test (Box2D v3): a body is slow when its fastest point,
//     |v| + |ω|·r_max, is under sleepLin; sleepAng is not used. A fixed rad/s
//     threshold is 20x stricter than the linear one for a 3 cm piece.
static const uint FLAG_POINT_SPEED_SLEEP = 8192u;
bool pointSpeedSleep() { return (PARAMS.flags & FLAG_POINT_SPEED_SLEEP) != 0u; }
// 32768 Displacement sleep test (Jolt's sleep test spheres): a body is slow while
//     its farthest point stays within sleepLin·sleepTime of the pose it had when
//     its sleep timer started (r_max for rotation), by both a low-pass pose
//     and the velocity integral Σv·h; motion must show in both. Jitter and one-tick blips
//     that go nowhere no longer reset the timer; steady motion still does. The
//     reference pose lives in the previous-step body record (BODY_PREV), whose
//     pose fields are otherwise unused.
static const uint FLAG_SLEEP_DISPLACEMENT = 32768u;
static const float SLEEP_POSE_BETA = 0.2;
// 4194304 Push-out limit (Box2D contactPushMaxVelocity, PhysX
//     maxDepenetrationVelocity): overlap is corrected at most PUSH_MAX_SPEED·h
//     per step. Post-stabilization otherwise removed a jammed piece's whole
//     1.6-4 cm overlap in one pass, the iterations pushed it back, and it
//     cycled by ~3.5 cm every few ticks with zero reported velocity.
static const uint FLAG_PUSH_LIMIT = 4194304u;
// 8388608 (removed) Rotational armature (MuJoCo `armature`): measured no gain
//     on the bonded suburb (3,275 awake vs 2,191) and read colliders inside
//     the primal, so it was dropped. The bit is unused.
static const float PUSH_MAX_SPEED = 0.1;
bool displacementSleep() { return (PARAMS.flags & FLAG_SLEEP_DISPLACEMENT) != 0u; }
// A body that may wake its neighbours (contacts or bonds): awake, and under the
// displacement test not itself a sleep candidate (those only jitter in place).
bool canWake(BodyState b)
{
    return bodyIsDynamic(b) && !bodyIsSleeping(b) && (!displacementSleep() || b.sleepTimer < PARAMS.sleepTime);
}
// 65536 Stabilization (PhysX PxSceneFlag::eENABLE_STABILIZATION): a body whose
//     fastest point is under STAB_SPEEDS·sleepLin keeps only the fraction
//     speed/(STAB_SPEEDS·sleepLin) of its velocity for the next step, so
//     near-rest jitter cannot build momentum; faster motion is untouched.
static const uint FLAG_STABILIZATION = 65536u;
// 131072 Deferred wake: a body woken during a step (contact or bond wake, in the
//     narrowphase, after the broadphase skipped its sleeper pairs) stays
//     immovable for the rest of that step and is solved from the next one, when
//     it has its whole contact set. Woken immediately, it was solved for a step
//     without its supports and started with a 0.1-0.6 m/s kick that woke its
//     neighbours in turn (Box2D instead restores a woken island's contacts).
static const uint FLAG_DEFER_WAKE = 131072u;
// 524288 Bond geometric stiffness (AVBD Sec. 3.5, avbd-demo3d Joint): the
//     anchor rows' lumped |F|·|r| term in the angular Hessian. Without it the
//     per-body Newton step overshoots under large bond forces and the iteration
//     diverges (light bonded slivers spun to 240 rad/s; worse with more
//     iterations).
static const uint FLAG_BOND_GEOMETRIC = 524288u;
// 1048576 Max rotation per step (Box2D b2_maxRotation = 0.25π; Jolt clamps
//     angular velocity to 0.25π/dt; PhysX maxAngularVelocity): |ω|·h ≤ 0.25π.
//     Past it the small-angle Jacobians and the quaternion update break down
//     (light bonded slivers saturated at 2/h = 240 rad/s and kept flipping).
static const uint FLAG_MAX_ROTATION = 1048576u;
// 2097152 Rolling resistance (Box2D v3 shape rollingResistance; Bullet rolling
//     friction): per manifold, a torque against the pair's relative rolling
//     (rotation about axes in the contact plane) capped at μr·N·r, N the
//     manifold's normal force and r the contact's lever arm, so a rolling body
//     comes to rest instead of rolling forever on rigid ground.
static const uint FLAG_ROLLING_RESISTANCE = 2097152u;
static const float ROLLING_RESISTANCE = 0.02;

static const float MAX_ROTATION = 0.785398;
static const uint BODY_WAKE_PENDING = 16u;
uint wokenFlags(uint flags)
{
    uint pending = (PARAMS.flags & FLAG_DEFER_WAKE) != 0u ? BODY_WAKE_PENDING : 0u;
    return (flags & ~BODY_SLEEPING) | BODY_AWAKE | pending;
}
bool wakePending(BodyState b) { return (b.flags & BODY_WAKE_PENDING) != 0u; }
static const float STAB_SPEEDS = 3.0;
// Set by the primal kernels from the pass value (bit 8): the stabilization pass.
static bool gStabilize = false;
bool postStabilize() { return (PARAMS.flags & FLAG_POST_STAB) != 0u; }
float warmDecay() { return postStabilize() ? 1.0 : PARAMS.alpha * PARAMS.gamma; }
float primalAlpha() { return postStabilize() ? (gStabilize ? 0.0 : 1.0) : PARAMS.alpha; }
float dualAlpha() { return postStabilize() ? 1.0 : PARAMS.alpha; }

// Step-start normal offset of a contact whose signed gap is n·(xB−xA).
float contactC0N(float gap, float margin)
{
    return contactGapMode() ? gap : gap + margin;
}

// α-regularized step-start normal offset. α absorbs initial penetration; with
// FLAG_CONTACT_GAP a positive gap is real clearance, not error, so it is kept
// whole (the row then stops the bodies exactly at touch).
float contactC0NReg(float c0N, float keep)
{
    // Push-out limit: overlap is removed at most PUSH_MAX_SPEED·h per step.
    if ((PARAMS.flags & FLAG_PUSH_LIMIT) != 0u)
        return max(contactC0NRegFull(c0N, keep), -PUSH_MAX_SPEED * PARAMS.dt);
    return contactC0NRegFull(c0N, keep);
}
float contactC0NRegFull(float c0N, float keep)
{
    if (contactGapMode()) return c0N > 0.0 ? c0N : c0N * keep;
    // Look-ahead without FLAG_CONTACT_GAP: C0 = gap + margin. Only a real gap
    // (C0 beyond the margin) is clearance kept whole; the margin band below it
    // behaves exactly as without look-ahead.
    if ((PARAMS.flags & FLAG_SPEC_VELOCITY) != 0u)
    {
        float margin = PARAMS.contactOffset;
        return max(c0N - margin, 0.0) + min(c0N, margin) * keep;
    }
    return c0N * keep;
}
static const float REUSE_LIN_TOL = 0.002;
static const float REUSE_ANG_TOL = 0.002;

float3 quatSubVec(float4 a, float4 b)
{
    float4 d = quatMul(a, quatConjugate(b));
    return 2.0 * d.xyz;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_refs(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    // Moved-only steps (HOST_S(19), JS host): only bodies on last step's awake
    // list changed pose, so every other body would find itself unmoved below.
    if (HOST_S(19) != 0u)
    {
        if (i >= HOST_S(6)) return;
        i = gAwakeList[i];
    }
    else if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b)) return;
    RefPoseGpu last = gRefPoses[i];
    uint step = HOST_S(17);
    bool noRef = dot(last.rot, last.rot) < 1e-12;
    // Same hemisphere as last.rot — q and -q are the same attitude; without this,
    // a solver sign flip looks like a ~2π turn and bumps generation every frame.
    float4 rot = b.orientation;
    if (dot(rot, last.rot) < 0.0) rot = -rot;
    bool linMove = length(b.position - last.pos.xyz) > REUSE_LIN_TOL;
    bool angMove = length(quatSubVec(rot, last.rot)) > REUSE_ANG_TOL;
    bool moved = noRef || linMove || angMove;
    if (moved)
    {
        RefPoseGpu r;
        r.pos = float4(b.position, 0.0);
        r.rot = rot;
        gRefPoses[i] = r;
        gBodies[i].generation = step;
    }
}

// three-avbd reuseContacts. Opt-in via params.flags bit0.
// Copy one previous contact run (contiguous, same collider pair) into this
// step as a manifold whose points persistence resolves from the previous half.
void emitReusedRun(uint colA, uint colB, uint startJ, uint count, BodyState ba, BodyState bb)
{
    uint prevOff = HOST_S(8);
    uint maxC = HOST_S(3);
    uint maxM = HOST_S(10);
    uint mslot;
    AADD3(gCounters[0].manifoldCount, 1, mslot);
    if (mslot >= maxM)
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_MANIFOLDS);
        return;
    }
    uint cStart;
    AADD3(gCounters[0].contactCount, count, cStart);
    uint curOff = HOST_S(11);
    if (cStart >= maxC || cStart + count > maxC)
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_CONTACTS);
        Manifold dead;
        dead.colliderA = colA;
        dead.colliderB = colB;
        dead.contactStart = 0;
        dead.contactCount = 0;
        dead.normal = float3(0, 1, 0);
        dead.friction = 0.0;
        gManifolds[mslot] = dead;
        return;
    }

    Contact head = READ_CONTACT(prevOff + startJ);
#if defined(ENGINE_REUSE_COPY)
    // Moving pairs: finish the reused copy here (exactly persistContact's reused
    // path) and mark it final. Pairs at rest defer so later wakeups still count.
    bool atRest = contactPairAtRest(ba, bb);
#endif
    for (uint k = 0u; k < count; k++)
    {
#if defined(ENGINE_REUSE_COPY)
        if (!atRest)
        {
            Contact c = READ_CONTACT(prevOff + startJ + k);
            c.alive = 1;
            storeFinalPersistentContact(curOff + cStart + k, mslot,
                finishWarmContact(refreshReusedContact(c, ba, bb), ba, bb));
            continue;
        }
#endif
        // Defer the full copy/transform to the coalesced per-contact persistence
        // pass. Encode the exact source index in the transient alive word.
        MARK_REUSED_CONTACT(curOff + cStart + k, mslot, startJ + k + 2u);
    }
#if defined(ENGINE_REUSE_COPY)
    if (atRest) persistListAppend(cStart, count);
#endif
    Manifold mf;
    mf.colliderA = colA;
    mf.colliderB = colB;
    mf.contactStart = curOff + cStart;
    mf.contactCount = count;
    mf.normal = head.normal;
    mf.friction = head.friction;
    gManifolds[mslot] = mf;
}

bool tryReuseContacts(CandidatePair pair, BodyState ba, BodyState bb, float margin)
{
    uint prevFeature;
    return tryReuseContactsHint(pair, ba, bb, margin, prevFeature);
}

// As tryReuseContacts; also reports the previous run's first feature ID (0 if
// the pair had no previous contacts) so convex SAT can start from that axis.
bool tryReuseContactsHint(CandidatePair pair, BodyState ba, BodyState bb, float margin, out uint prevFeature)
{
    prevFeature = 0u;
    uint prevOff = HOST_S(8);
    uint prevN = HOST_S(7);
    uint hashN = HOST_S(9);
    uint startJ = CONTACT_HASH_EMPTY;
    uint count = 0u;
    if (warmRecords())
    {
        // Run start and length were resolved once per manifold at insert time.
        if (!findWarmRecord(pair.colliderA, pair.colliderB, startJ, count)) return false;
        // _pad1 = asfloat(generated step). Skip if either body moved after that.
        Contact first = READ_CONTACT(prevOff + startJ);
        prevFeature = first.featureId;
        uint generated = asuint(first._pad1);
        if (generated < max(ba.generation, bb.generation)) return false;
    }
    else
    {
    uint key = pairWarmKey(pair.colliderA, pair.colliderB);
    uint h = key & (hashN - 1u);
    uint table = warmBase();
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = (h + probe) & (hashN - 1u);
        uint j = warmSlotValue(ALOAD(gConstraintIds[table + slot]));
        if (j == CONTACT_HASH_EMPTY) break;
        Contact prev = READ_CONTACT(prevOff + j);
        if (prev.alive != 0 &&
            prev.colliderA == pair.colliderA &&
            prev.colliderB == pair.colliderB)
        {
            startJ = j;
            break;
        }
    }
    if (startJ == CONTACT_HASH_EMPTY)
    {
        return false;
    }
    // _pad1 = asfloat(generated step). Skip if either body moved after that.
    Contact first = READ_CONTACT(prevOff + startJ);
    prevFeature = first.featureId;
    uint generated = asuint(first._pad1);
    if (generated < max(ba.generation, bb.generation))
    {
        return false;
    }

    for (uint k = 0u; k < 8u; k++)
    {
        uint j = startJ + k;
        if (j >= prevN) break;
        Contact prev = READ_CONTACT(prevOff + j);
        if (prev.alive == 0 ||
            prev.colliderA != pair.colliderA ||
            prev.colliderB != pair.colliderB)
            break;
        count++;
    }
    }
    if (count == 0u) return false;
    emitReusedRun(pair.colliderA, pair.colliderB, startJ, count, ba, bb);
    wakeFromContact(ba, bb, gColliders[pair.colliderA], gColliders[pair.colliderB]);
    return true;
}

// Whole-step manifold reuse (three-avbd reuseManifolds): skip bounds, broadphase
// and narrowphase, and carry every previous contact run forward; persistence
// re-evaluates C0 from the anchors. One thread per previous contact (slot 73);
// each run's head emits it. Hosts only schedule this while the previous step's
// manifolds are valid (no scene, filter, hull or command changes since).
[shader("compute")]
[numthreads(64, 1, 1)]
void reuse_all_manifolds(uint3 tid : SV_DispatchThreadID)
{
    uint j = dispatchIndex(tid);
    uint prevN = min(HOST_S(7), HOST_S(3));
    if (j >= prevN) return;
    uint prevOff = HOST_S(8);
    Contact c = READ_CONTACT(prevOff + j);
    if (c.alive == 0u) return;
    if (j > 0u)
    {
        Contact before = READ_CONTACT(prevOff + j - 1u);
        if (before.alive != 0u && before.colliderA == c.colliderA && before.colliderB == c.colliderB)
            return;
    }
    uint count = 1u;
    while (count < 8u && j + count < prevN)
    {
        Contact next = READ_CONTACT(prevOff + j + count);
        if (next.alive == 0u || next.colliderA != c.colliderA || next.colliderB != c.colliderB) break;
        count++;
    }
    BodyState ba = gBodies[gColliders[c.colliderA].body];
    BodyState bb = gBodies[gColliders[c.colliderB].body];
    emitReusedRun(c.colliderA, c.colliderB, j, count, ba, bb);
}

// Hull pairs run in convex_classify / convex_sat_* / convex_manifold. Those
// entries bind the hull arena and the convex work list; the narrowphase entries
// stay within the WebGPU storage-buffer budget. Carry the previous SAT feature.
bool deferConvexPair(uint i, Collider ca, Collider cb, uint prevFeature)
{
    if (ca.shape != SHAPE_CONVEX && cb.shape != SHAPE_CONVEX) return false;
    gCandidates[i]._pad = CONVEX_DEFER | convexHintFromFeature(prevFeature);
    return true;
}

// Shape tests and emission for one candidate that did not reuse contacts.
[ForceInline]
void narrowphaseShapes(CandidatePair pair, Collider ca, Collider cb, BodyState ba, BodyState bb)
{
    float3 posA = transformPoint(ba.position, ba.orientation, ca.localPos);
    float3 posB = transformPoint(bb.position, bb.orientation, cb.localPos);
    float4 rotA = quatMul(ba.orientation, ca.localRot);
    float4 rotB = quatMul(bb.orientation, cb.localRot);
    // Contacts sit `margin` deep (C0 = gap + margin); they are detected up to
    // `detect`, which adds how far the pair can close this step (look-ahead).
    float margin = PARAMS.contactOffset;
    float detect = margin + pairReach(ba, ca, bb, cb);
    float3 nrm; float pen; float3 pA; float3 pB;
    bool hit = false;

    // Multi-point manifolds (demo3d / three-avbd) — required for stable stacks.
    if (ca.shape == SHAPE_BOX && cb.shape == SHAPE_BOX)
    {
        // World OBBs from build_aabbs cache (skip per-pair quat axis rebuild).
        BoxManifold mf = collideBoxesCached(
            worldObbFromCache(pair.colliderA), worldObbFromCache(pair.colliderB), detect);
        if (mf.count > 0)
            emitManifold(pair, ba, bb, ca, cb, mf, margin);
        return;
    }
    if (ca.shape == SHAPE_BOX && cb.shape == SHAPE_PLANE)
    {
        emitManifold(pair, ba, bb, ca, cb,
            collideBoxPlane(posA, rotA, ca.params.xyz, cb.params.xyz, cb.params.w, detect), margin);
        return;
    }
    if (ca.shape == SHAPE_PLANE && cb.shape == SHAPE_BOX)
    {
        // Swap to box-vs-plane then flip normal / points for A=plane, B=box.
        BoxManifold mf = collideBoxPlane(posB, rotB, cb.params.xyz, ca.params.xyz, ca.params.w, detect);
        if (mf.count > 0)
            emitManifold(pair, ba, bb, ca, cb, flipManifold(mf), margin);
        return;
    }
    if (ca.shape == SHAPE_BOX && cb.shape == SHAPE_CAPSULE)
    {
        emitManifold(pair, ba, bb, ca, cb,
            collideBoxCapsule(posA, rotA, ca.params.xyz, posB, rotB, cb.params.x, cb.params.y, speculativeMargin()), margin);
        return;
    }
    if (ca.shape == SHAPE_CAPSULE && cb.shape == SHAPE_BOX)
    {
        BoxManifold mf = collideBoxCapsule(posB, rotB, cb.params.xyz, posA, rotA, ca.params.x, ca.params.y, speculativeMargin());
        if (mf.count > 0)
            emitManifold(pair, ba, bb, ca, cb, flipManifold(mf), margin);
        return;
    }
    if (ca.shape == SHAPE_CAPSULE && cb.shape == SHAPE_CAPSULE)
    {
        emitManifold(pair, ba, bb, ca, cb,
            collideCapsules(posA, rotA, ca.params.x, ca.params.y, posB, rotB, cb.params.x, cb.params.y, speculativeMargin()), margin);
        return;
    }
    if (ca.shape == SHAPE_CAPSULE && cb.shape == SHAPE_PLANE)
    {
        emitManifold(pair, ba, bb, ca, cb,
            collideCapsulePlane(posA, rotA, ca.params.x, ca.params.y, cb.params.xyz, cb.params.w, margin), margin);
        return;
    }
    if (ca.shape == SHAPE_PLANE && cb.shape == SHAPE_CAPSULE)
    {
        BoxManifold mf = collideCapsulePlane(posB, rotB, cb.params.x, cb.params.y, ca.params.xyz, ca.params.w, margin);
        if (mf.count > 0)
            emitManifold(pair, ba, bb, ca, cb, flipManifold(mf), margin);
        return;
    }

    if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_SPHERE)
    {
        hit = sphereSphere(posA, ca.params.x, posB, cb.params.x, detect, nrm, pen, pA, pB);
    }
    else if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_PLANE)
    {
        // Geometric points: sphere surface toward plane, plane projection.
        float3 pn = normalize(cb.params.xyz);
        float gap = dot(posA, pn) - cb.params.w - ca.params.x;
        if (gap < detect)
        {
            hit = true; nrm = -pn; pen = margin - gap;
            pA = posA - pn * ca.params.x;
            pB = pA - pn * gap; // on the geometric plane
        }
    }
    else if (ca.shape == SHAPE_PLANE && cb.shape == SHAPE_SPHERE)
    {
        float3 pn = normalize(ca.params.xyz);
        float gap = dot(posB, pn) - ca.params.w - cb.params.x;
        if (gap < detect)
        {
            hit = true; nrm = pn; pen = margin - gap;
            pB = posB - pn * cb.params.x;
            pA = pB - pn * gap;
        }
    }
    else if (ca.shape == SHAPE_BOX && cb.shape == SHAPE_SPHERE)
    {
        // three-avbd sphereBox: closest point, or shallowest face when centre is inside.
        float3 local = quatRotate(quatConjugate(rotA), posB - posA);
        float3 half = ca.params.xyz;
        float3 onBox = clamp(local, -half, half);
        float3 d = local - onBox;
        float dist = length(d);
        // three-avbd sphereBox: geometric radius only; margin goes into C0 at emit.
        float rGeo = cb.params.x;
        float3 nLocal;
        if (dist > 1e-8)
        {
            if (dist <= rGeo)
            {
                hit = true;
                nLocal = d / dist;
                pen = rGeo - dist;
                nrm = quatRotate(rotA, nLocal);
                pA = transformPoint(posA, rotA, onBox);
                pB = posB - nrm * rGeo;
            }
        }
        else
        {
            float3 depth = half - abs(local);
            int k = 0;
            if (depth.y < depth.x) k = 1;
            if (depth.z < ((k == 0) ? depth.x : depth.y)) k = 2;
            float pk = (k == 0) ? local.x : ((k == 1) ? local.y : local.z);
            float hk = (k == 0) ? half.x : ((k == 1) ? half.y : half.z);
            float dk = (k == 0) ? depth.x : ((k == 1) ? depth.y : depth.z);
            float s = pk >= 0.0 ? 1.0 : -1.0;
            nLocal = float3(0, 0, 0);
            if (k == 0) { nLocal.x = s; onBox.x = s * hk; }
            else if (k == 1) { nLocal.y = s; onBox.y = s * hk; }
            else { nLocal.z = s; onBox.z = s * hk; }
            hit = true;
            pen = rGeo + dk;
            nrm = quatRotate(rotA, nLocal);
            pA = transformPoint(posA, rotA, onBox);
            pB = posB - nrm * rGeo;
        }
    }
    else if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_BOX)
    {
        // Same geometry as box→sphere; flip normal so nrm is A→B (sphere→box).
        float3 local = quatRotate(quatConjugate(rotB), posA - posB);
        float3 half = cb.params.xyz;
        float3 onBox = clamp(local, -half, half);
        float3 d = local - onBox;
        float dist = length(d);
        float rGeo = ca.params.x;
        float3 nLocal;
        if (dist > 1e-8)
        {
            if (dist <= rGeo)
            {
                hit = true;
                nLocal = d / dist;
                pen = rGeo - dist;
                nrm = -quatRotate(rotB, nLocal);
                pB = transformPoint(posB, rotB, onBox);
                pA = posA + nrm * rGeo;
            }
        }
        else
        {
            float3 depth = half - abs(local);
            int k = 0;
            if (depth.y < depth.x) k = 1;
            if (depth.z < ((k == 0) ? depth.x : depth.y)) k = 2;
            float pk = (k == 0) ? local.x : ((k == 1) ? local.y : local.z);
            float hk = (k == 0) ? half.x : ((k == 1) ? half.y : half.z);
            float dk = (k == 0) ? depth.x : ((k == 1) ? depth.y : depth.z);
            float s = pk >= 0.0 ? 1.0 : -1.0;
            nLocal = float3(0, 0, 0);
            if (k == 0) { nLocal.x = s; onBox.x = s * hk; }
            else if (k == 1) { nLocal.y = s; onBox.y = s * hk; }
            else { nLocal.z = s; onBox.z = s * hk; }
            hit = true;
            pen = rGeo + dk;
            nrm = -quatRotate(rotB, nLocal);
            pB = transformPoint(posB, rotB, onBox);
            pA = posA + nrm * rGeo;
        }
    }
    else if (ca.shape == SHAPE_CAPSULE && cb.shape == SHAPE_SPHERE)
    {
        float3 axis = quatRotate(rotA, float3(0, 1, 0));
        float half = ca.params.y;
        float3 pa = posA - axis * half;
        float3 pb = posA + axis * half;
        float3 ab = pb - pa;
        float t = clamp(dot(posB - pa, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
        float3 closest = pa + ab * t;
        hit = sphereSphere(closest, ca.params.x, posB, cb.params.x, detect, nrm, pen, pA, pB);
    }
    else if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_CAPSULE)
    {
        float3 axis = quatRotate(rotB, float3(0, 1, 0));
        float half = cb.params.y;
        float3 pa = posB - axis * half;
        float3 pb = posB + axis * half;
        float3 ab = pb - pa;
        float t = clamp(dot(posA - pa, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
        float3 closest = pa + ab * t;
        hit = sphereSphere(posA, ca.params.x, closest, cb.params.x, detect, nrm, pen, pA, pB);
    }
    else
    {
        float rA = ca.params.w > 0.0 ? ca.params.w : length(ca.params.xyz);
        float rB = cb.params.w > 0.0 ? cb.params.w : length(cb.params.xyz);
        hit = sphereSphere(posA, rA, posB, rB, detect, nrm, pen, pA, pB);
    }

    if (!hit) return;
    emitContact(pair, ba, bb, ca, cb, nrm, pA, pB, pair.key, margin);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void narrowphase(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = ALOAD(gCounters[0].candidateCount);
    if (i >= n) return;
    CandidatePair pair = gCandidates[i];
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    // three-avbd: reuseContacts before any shape SAT (ground + box-box + spheres).
    uint prevFeature = 0u;
    if ((PARAMS.flags & FLAG_REUSE_CONTACTS) != 0u &&
        tryReuseContactsHint(pair, ba, bb, PARAMS.contactOffset, prevFeature))
        return;
    if (deferConvexPair(i, ca, cb, prevFeature)) return;
    narrowphaseShapes(pair, ca, cb, ba, bb);
}

// Opt-in two-pass narrowphase (8ek): reuse first, then shape tests over a
// compacted list, so warps no longer run both the reuse copy and box SAT.
// Deferred pairs are copied to the unused tail of gCandidates (counted in
// gCounters._pad1); when the tail is full a pair is tested inline instead.
// HOST_S(21) != 0 caps the tail (tests exercise the inline fallback).
uint narrowphaseDeferCapacity(uint n)
{
    uint cap = HOST_S(2);
    uint tail = cap - min(n, cap);
    return HOST_S(21) != 0u ? min(tail, HOST_S(21)) : tail;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void narrowphase_defer(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = ALOAD(gCounters[0].candidateCount);
    if (i >= n) return;
    CandidatePair pair = gCandidates[i];
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    uint prevFeature = 0u;
    if ((PARAMS.flags & FLAG_REUSE_CONTACTS) != 0u &&
        tryReuseContactsHint(pair, ba, bb, PARAMS.contactOffset, prevFeature))
        return;
    // Hull pairs never enter the shape tail; the convex stages take them.
    if (deferConvexPair(i, ca, cb, prevFeature)) return;
    uint k;
    AADD3(gCounters[0]._pad1, 1u, k);
    if (k < narrowphaseDeferCapacity(n))
    {
        gCandidates[HOST_S(2) - 1u - k] = pair;
        return;
    }
    narrowphaseShapes(pair, ca, cb, ba, bb);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void narrowphase_fresh(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    uint n = ALOAD(gCounters[0].candidateCount);
    if (k >= min(ALOAD(gCounters[0]._pad1), narrowphaseDeferCapacity(n))) return;
    CandidatePair pair = gCandidates[HOST_S(2) - 1u - k];
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    narrowphaseShapes(pair, ca, cb, gBodies[ca.body], gBodies[cb.body]);
}

JointBall loadJointBall(uint i)
{
    JointBall v;
    v.bodyA = gJoints[i].bodyA;
    v.bodyB = gJoints[i].bodyB;
    v.anchorA = gJoints[i].anchorA;
    v.anchorB = gJoints[i].anchorB;
    v.linStiffness = gJoints[i].linStiffness;
    v.breakForce = gJoints[i].breakForce;
    v.c0 = gJoints[i].c0Lo.xyz;
    v.lambda = gJoints[i].lambdaLo.xyz;
    v.penalty = gJoints[i].penaltyLo.xyz;
    return v;
}

bool jointHeaderIsBall(uint flags, uint motion)
{
    return (motion & 0xFFFu) == 0xA80u && (flags & (JOINT_FLAG_RADIAL | JOINT_FLAG_CONE | JOINT_FLAG_DRIVE)) == 0u;
}

// demo3d Joint::initialize — store C0, decay λ/k (Eq. 19).
// Wake through intact joints (one hop per dispatch; hosts run a few rounds): a
// sleeping body bonded or jointed to an awake one must move with it, or it would act as a
// fixed anchor and overload the bond. Bonded islands sleep together (gate).
[shader("compute")]
[numthreads(64, 1, 1)]
void bond_wake(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].jointCount) return;
    Joint j = gJoints[i];
    if (jointBroken(j)) return;
    BodyState a = gBodies[j.bodyA];
    BodyState b = gBodies[j.bodyB];
    bool sleepA = bodyIsDynamic(a) && bodyIsSleeping(a);
    bool sleepB = bodyIsDynamic(b) && bodyIsSleeping(b);
    if (sleepA && canWake(b))
    {
        gBodies[j.bodyA].flags = wokenFlags(a.flags);
        gBodies[j.bodyA].sleepTimer = 0.0;
    }
    if (sleepB && canWake(a))
    {
        gBodies[j.bodyB].flags = wokenFlags(b.flags);
        gBodies[j.bodyB].sleepTimer = 0.0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void warmstart_joints(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].jointCount;
    if (i >= n) return;
    Joint j = gJoints[i];
    // Broken joints are inert; the vehicle kernels own tyre joints.
    if ((j.flags & (JOINT_FLAG_BROKEN | JOINT_FLAG_TYRE)) != 0u) return;
    AvbdParams p = PARAMS;
    BodyState ja = gBodies[j.bodyA];
    BodyState jb = gBodies[j.bodyB];
    // A joint never links a sleeping body to an awake one: wake the sleeper so
    // the island moves together (contacts wake in the narrowphase).
    if (bodyIsDynamic(jb) && bodyIsSleeping(jb) && canWake(ja))
    {
        gBodies[j.bodyB].flags = wokenFlags(jb.flags);
        gBodies[j.bodyB].sleepTimer = 0.0;
    }
    if (bodyIsDynamic(ja) && bodyIsSleeping(ja) && canWake(jb))
    {
        gBodies[j.bodyA].flags = wokenFlags(ja.flags);
        gBodies[j.bodyA].sleepTimer = 0.0;
    }
    // Stasis while both ends rest (asleep or static), like the sleeping
    // contacts: keep λ and k for the wake.
    if (!bodyMoving(gBodies[j.bodyA]) && !bodyMoving(gBodies[j.bodyB])) return;
    if (jointHeaderIsBall(j.flags, j.motion))
    {
        float3 C = (ja.position + quatRotate(ja.orientation, j.anchorA)) - (jb.position + quatRotate(jb.orientation, j.anchorB));
        float ag = p.alpha * p.gamma;
        gJoints[i].c0Lo.xyz = C;
        gJoints[i].lambdaLo.xyz = j.linStiffness >= JOINT_HARD ? j.lambdaLo.xyz * ag : float3(0.0);
        gJoints[i].penaltyLo.xyz = clamp(j.penaltyLo.xyz * p.gamma, float3(p.kStart), float3(p.kMaxN));
        return;
    }
    jointWarmstart(j, ja, jb, p.alpha, p.gamma, p.kStart, p.kMaxN);
    gJoints[i].lambdaLo = j.lambdaLo; gJoints[i].lambdaHi = j.lambdaHi;
    gJoints[i].penaltyLo = j.penaltyLo; gJoints[i].penaltyHi = j.penaltyHi;
    gJoints[i].c0Lo = j.c0Lo; gJoints[i].c0Hi = j.c0Hi;
}

// Prev-contact warm-start hash in gConstraintIds (open address). Built before NP
// (reuseContacts + persist), then overwritten by CSR. Replaces O(n²) linear scan.

[shader("compute")]
[numthreads(64, 1, 1)]
void contact_hash_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = HOST_S(9) * (warmRecords() ? 4u : 1u); // hash table word count
    if (i >= n) return;
    ASTORE(gConstraintIds[warmBase() + i], CONTACT_HASH_EMPTY);
}

// Publish history index j (a live block head) for pair (colA, colB).
void insertWarmHead(uint j, uint colA, uint colB, uint prevOff)
{
    uint key = pairWarmKey(colA, colB);
    uint n = HOST_S(9);
    uint h = key & (n - 1u); // n is power of two
    uint table = warmBase();
    // Slot words (tagged: tag | index); untagged, words are the indices.
    uint mine = warmSlotWord(j);
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = table + ((h + probe) & (n - 1u));
        // An empty slot holds EMPTY or (tagged) a word from an earlier step;
        // other inserts only ever write this step's words.
        uint expect = warmTagged() ? ALOAD(gConstraintIds[slot]) : CONTACT_HASH_EMPTY;
        uint was = expect;
        // Spin on the same slot when WGSL weak CAS spuriously fails. Advancing
        // the probe would leave an EMPTY hole and break open-address lookups.
        for (uint spin = 0u; spin < 64u; spin++)
        {
            if (warmSlotValue(expect) != CONTACT_HASH_EMPTY) { was = expect; break; }
            ACAS4(gConstraintIds[slot], expect, mine, was);
            if (was != expect) { expect = was; continue; }
            if (ALOAD(gConstraintIds[slot]) == mine) return;
        }
        uint occupied = warmSlotValue(was);
        if (occupied == CONTACT_HASH_EMPTY) return;
        Contact occupant = READ_CONTACT(prevOff + occupied);
        if (occupant.alive != 0 &&
            occupant.colliderA == colA &&
            occupant.colliderB == colB)
        {
            // Same pair: keep the minimum index (start of the contiguous block).
            uint cur = was;
            while (j < warmSlotValue(cur))
            {
                uint got;
                for (uint spin = 0u; spin < 64u; spin++)
                {
                    ACAS4(gConstraintIds[slot], cur, mine, got);
                    if (got != cur) break;
                    if (ALOAD(gConstraintIds[slot]) == mine) return;
                }
                if (got == cur) return;
                cur = got;
            }
            return;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void contact_hash_insert(uint3 tid : SV_DispatchThreadID)
{
    // Slot value = lowest prev-contact index of that pair (manifold block start).
    uint j = dispatchIndex(tid);
    uint prevN = HOST_S(7);
    if (j >= prevN) return;
    uint prevOff = HOST_S(8);
    Contact prev = READ_CONTACT(prevOff + j);
    if (prev.alive == 0) return;
    // Contacts are emitted as contiguous manifold blocks. Only the head needs
    // to claim a hash slot; having all eight corners contend adds no information.
    // Retain the minimum-index CAS below for separate blocks of the same pair.
    if (j > 0u)
    {
        Contact before = READ_CONTACT(prevOff + j - 1u);
        if (before.alive != 0 && before.colliderA == prev.colliderA &&
            before.colliderB == prev.colliderB) return;
    }
    insertWarmHead(j, prev.colliderA, prev.colliderB, prevOff);
}

// three-avbd hashInsert: one thread per previous manifold instead of per point.
// Publishes the same value as contact_hash_insert (first live point of each
// block), so lookups are unchanged. Opt-in via HOST_S(14); HOST_S(13) holds the
// previous manifold count. Precondition: gManifolds still holds the previous
// step, i.e. dispatch before narrowphase (contact reuse enabled).
[shader("compute")]
[numthreads(64, 1, 1)]
void contact_hash_insert_manifolds(uint3 tid : SV_DispatchThreadID)
{
    uint m = dispatchIndex(tid);
    if (m >= min(HOST_S(13), HOST_S(10))) return;
    Manifold mf = gManifolds[m];
    uint prevOff = HOST_S(8);
    uint prevN = HOST_S(7);
    if (mf.contactCount == 0u || mf.contactStart < prevOff) return;
    uint start = mf.contactStart - prevOff;
    for (uint k = 0u; k < min(mf.contactCount, 8u); k++)
    {
        uint j = start + k;
        if (j >= prevN) return;
        if (CONTACT_STATE(prevOff + j) != 0u)
        {
            // Persistence marks every point of a live manifold alive and overflow
            // only zeroes points of zero-count manifolds, so the run from j is the
            // rest of this block (at most eight, within the previous set).
            if (warmRecords())
                insertWarmRecord(m, j, min(min(start + mf.contactCount - j, 8u), prevN - j),
                    mf.colliderA, mf.colliderB);
            else insertWarmHead(j, mf.colliderA, mf.colliderB, prevOff);
            return;
        }
    }
}

void insertWarmRecord(uint m, uint j, uint count, uint colA, uint colB)
{
    uint claim = m + 1u;
    uint slots = HOST_S(9);
    uint h = pairWarmKey(colA, colB) & (slots - 1u);
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint w = ((h + probe) & (slots - 1u)) * 4u;
        uint was;
        // ACAS4 retries spurious weak-CAS failures; the claim is unique, so the
        // slot holding it after an EMPTY result proves this thread won it.
        ACAS4(gConstraintIds[w], CONTACT_HASH_EMPTY, claim, was);
        if (was == CONTACT_HASH_EMPTY && ALOAD(gConstraintIds[w]) == claim)
        {
            ASTORE(gConstraintIds[w + 1u], colA);
            ASTORE(gConstraintIds[w + 2u], colB);
            ASTORE(gConstraintIds[w + 3u], j | (count << WARM_RUN_SHIFT));
            return;
        }
    }
    AOR2(gCounters[0].overflowFlags, OVERFLOW_WARM_HASH);
}

// Step-start constraint offsets of a reused point at current poses.
Contact refreshReusedContact(Contact c, BodyState ba, BodyState bb)
{
    AvbdParams p = PARAMS;
    float3 wa = transformPoint(ba.position, ba.orientation, c.anchorA);
    float3 wb = transformPoint(bb.position, bb.orientation, c.anchorB);
    float3 d = wb - wa;
    c.c0N = contactC0N(dot(c.normal, d), p.contactOffset);
    c.c0T1 = dot(c.tangent1, d);
    c.c0T2 = dot(c.tangent2, d);
    c.penetration = c.c0N;
    return c;
}

// Mass-floored normal penalty, independent tangent ramps and λ decay.
Contact finishWarmContact(Contact c, BodyState ba, BodyState bb)
{
    AvbdParams p = PARAMS;
    c.alive = 1;

    float dt = max(p.dt, 1e-8);
    float mA = ba.invMass > 1e-12 ? (1.0 / ba.invMass) : 1e20;
    float mB = bb.invMass > 1e-12 ? (1.0 / bb.invMass) : 1e20;
    // Normal stiffness is floored at m/dt²; each tangent ramps independently.
    float kMass = min(mA, mB) / (dt * dt);
    c.kN = max(max(1.0, kMass), p.gamma * c.kN);
    float kTMin = (p.flags & FLAG_TANGENT_MASS_FLOOR) != 0u ? clamp(kMass, 1.0, p.kMaxN) : 1.0;
    c.kT = max(kTMin, p.gamma * c.kT);
    c._pad2 = max(kTMin, p.gamma * c._pad2);
    c.lambdaN *= warmDecay();
    c.lambdaT1 *= warmDecay();
    c.lambdaT2 *= warmDecay();
    if ((p.flags & FLAG_LOAD_PENALTY) != 0u)
    {
        c.kN = min(max(c.kN, abs(c.lambdaN) / REST_SLOP), p.kMaxN);
        c.kT = min(max(c.kT, abs(c.lambdaT1) / REST_SLOP), p.kMaxN);
        c._pad2 = min(max(c._pad2, abs(c.lambdaT2) / REST_SLOP), p.kMaxN);
    }
    return c;
}

bool contactPairAtRest(BodyState ba, BodyState bb)
{
    return (!bodyIsDynamic(ba) || bodyIsSleeping(ba)) &&
        (!bodyIsDynamic(bb) || bodyIsSleeping(bb));
}

// Shared point update; callers choose contact or manifold scheduling. Pair bodies
// are immutable during persistence, so manifold scheduling can load them once.
Contact persistContact(uint i, Contact c, bool reused, BodyState ba, BodyState bb)
{
    AvbdParams p = PARAMS;
    uint curOff = HOST_S(11);
    c.alive = 1;
    if (contactPairAtRest(ba, bb))
    {
        STORE_CONTACT(curOff + i, c);
        return c;
    }
    if (reused)
        c = refreshReusedContact(c, ba, bb);
    else
    {
        uint prevOff = HOST_S(8);
        uint prevN = HOST_S(7);
        uint hashN = HOST_S(9);
        uint key = pairWarmKey(c.colliderA, c.colliderB);
        uint h = key & (hashN - 1u);
        uint startJ = CONTACT_HASH_EMPTY;
        uint runCount;
        if (warmRecords())
            findWarmRecord(c.colliderA, c.colliderB, startJ, runCount);
        else
        for (uint probe = 0; probe < 64u; probe++)
        {
            uint slot = (h + probe) & (hashN - 1u);
            uint j = warmSlotValue(ALOAD(gConstraintIds[warmBase() + slot]));
            if (j == CONTACT_HASH_EMPTY) break;
            Contact prev = READ_CONTACT(prevOff + j);
            if (prev.alive != 0 &&
                prev.colliderA == c.colliderA &&
                prev.colliderB == c.colliderB)
            {
                startJ = j;
                break;
            }
        }
        uint bestJ = CONTACT_HASH_EMPTY;
        if (startJ != CONTACT_HASH_EMPTY)
        {
            // Feature IDs are quantized on the reference face, so all four corners
            // of a small box on a large slab can share an ID. Pick the nearest
            // anchor among matches, not the first: reusing one sticking corner's
            // anchor/force for the entire manifold removes the support polygon.
            float bestFeatureDist = 1e30;
            for (uint k = 0u; k < 8u; k++)
            {
                uint j = startJ + k;
                if (j >= prevN) break;
                Contact prev = READ_CONTACT(prevOff + j);
                if (prev.alive == 0 ||
                    prev.colliderA != c.colliderA ||
                    prev.colliderB != c.colliderB)
                    break;
                if (prev.featureId == c.featureId)
                {
                    float3 deltaA = prev.anchorA - c.anchorA;
                    float3 deltaB = prev.anchorB - c.anchorB;
                    float dist = dot(deltaA, deltaA) + dot(deltaB, deltaB);
                    if (dist < bestFeatureDist)
                    {
                        bestFeatureDist = dist;
                        bestJ = j;
                    }
                }
            }
            // matchNearest: nearest prev A-local anchor within 5% of smaller side.
            if (bestJ == CONTACT_HASH_EMPTY)
            {
                Collider colA = gColliders[c.colliderA];
                Collider colB = gColliders[c.colliderB];
                float minSide = min(colliderMinSide(colA), colliderMinSide(colB));
                float bestDist = NEAREST_FRACTION * minSide;
                for (uint k = 0u; k < 8u; k++)
                {
                    uint j = startJ + k;
                    if (j >= prevN) break;
                    Contact prev = READ_CONTACT(prevOff + j);
                    if (prev.alive == 0 ||
                        prev.colliderA != c.colliderA ||
                        prev.colliderB != c.colliderB)
                        break;
                    float dist = length(prev.anchorA - c.anchorA);
                    if (dist <= bestDist)
                    {
                        bestDist = dist;
                        bestJ = j;
                    }
                }
            }
        }
        if (bestJ != CONTACT_HASH_EMPTY)
        {
            Contact prev = READ_CONTACT(prevOff + bestJ);
            // Refuse warm-start across SAT axis flips: λ from the old normal on a new
            // normal injects energy (random Box Pile). three-avbd avoids this via
            // reuseContacts keeping the manifold; without reuse, require n·n_prev>0.9.
            float nAgree = dot(prev.normal, c.normal);
            if (nAgree > 0.9)
            {
                c.lambdaN = prev.lambdaN;
                c.lambdaT1 = prev.lambdaT1;
                c.lambdaT2 = prev.lambdaT2;
                c.kN = prev.kN;
                c.kT = prev.kT;
                c._pad2 = prev._pad2;
                // three-avbd: keep old anchors while static friction held (stick).
                // Not for spheres — contact point slides over the surface as they roll;
                // pinned anchors rotate with the body and glue/stop smash balls.
                Collider colA = gColliders[c.colliderA];
                Collider colB = gColliders[c.colliderB];
                bool anySphere = colA.shape == SHAPE_SPHERE || colB.shape == SHAPE_SPHERE;
                // Nor for track belts: the surface runs past the contact every step.
                bool belt = colliderIsBeltLazy(colA) || colliderIsBeltLazy(colB);
                if (prev._pad0 > 0.5 && !anySphere && !belt)
                {
                    c.anchorA = prev.anchorA;
                    c.anchorB = prev.anchorB;
                    float3 wa = transformPoint(ba.position, ba.orientation, c.anchorA);
                    float3 wb = transformPoint(bb.position, bb.orientation, c.anchorB);
                    float3 d = wb - wa;
                    c.c0N = contactC0N(dot(c.normal, d), p.contactOffset);
                    c.c0T1 = dot(c.tangent1, d);
                    c.c0T2 = dot(c.tangent2, d);
                    c.penetration = c.c0N;
                    c._pad0 = 1.0;
                }
            }
        }
    }
    c = finishWarmContact(c, ba, bb);
    STORE_CONTACT(curOff + i, c);
    return c;
}

void persistContactAt(uint i)
{
    uint curOff = HOST_S(11);
    uint state = CONTACT_STATE(curOff + i);
    if (state == 0u) return;
    bool reused = state >= 2u;
    Contact c = reused ? READ_CONTACT(HOST_S(8) + state - 2u) : READ_CONTACT(curOff + i);
    BodyState ba = gBodies[gColliders[c.colliderA].body];
    BodyState bb = gBodies[gColliders[c.colliderB].body];
    persistContact(i, c, reused, ba, bb);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void persist_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= ALOAD(gCounters[0].contactCount)) return;
#if defined(ENGINE_REUSE_COPY)
    if ((gSolverPoints[HOST_S(11) + i].pairAndStick & PERSIST_FINAL) != 0u) return;
#endif
    persistContactAt(i);
}

#if defined(ENGINE_REUSE_COPY)
// Persist only the listed contacts; ignored when the list overflowed.
[shader("compute")]
[numthreads(64, 1, 1)]
void persist_list(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    // Dispatched over work slot 70, which is zero when the list overflowed.
    if (k >= ALOAD(gCounters[0]._pad2)) return;
    persistContactAt(persistListRead(k));
}
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void persist_manifolds(uint3 tid : SV_DispatchThreadID)
{
    uint mi = dispatchIndex(tid);
    if (mi >= ALOAD(gCounters[0].manifoldCount)) return;
    Manifold mf = gManifolds[mi];
#if defined(ENGINE_COMPACT_SOLVE)
    packSolverPair(mi, mf);
#endif
    if (mf.contactCount == 0u) return;
    BodyState ba = gBodies[gColliders[mf.colliderA].body];
    BodyState bb = gBodies[gColliders[mf.colliderB].body];
    uint curOff = HOST_S(11);
    for (uint k = 0u; k < mf.contactCount; k++)
    {
        uint absolute = mf.contactStart + k;
        uint state = CONTACT_STATE(absolute);
        if (state == 0u) continue;
        bool reused = state >= 2u;
        Contact c = reused ? READ_CONTACT(HOST_S(8) + state - 2u) : READ_CONTACT(absolute);
        Contact persisted = persistContact(absolute - curOff, c, reused, ba, bb);
#if defined(ENGINE_COMPACT_SOLVE) && !defined(ENGINE_PERSISTENT_CONTACTS)
        packSolverPoint(absolute - curOff, mi, persisted);
#endif
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_awake_list(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (bodyIsDynamic(b) && !bodyIsSleeping(b))
    {
        uint slot;
        AADD3(gCounters[0].awakeCount, 1, slot);
        gAwakeList[slot] = i;
    }
}

// --- CSR adjacency (three-avbd: degree → scan → scatter → sort) ----------------------------
// Constraint id: [0, jointCount) = joints; jointCount + mi = manifold mi (Stage 8b).
// CINDEX(0..bodyCount): degree then exclusive starts; [bodyCount] = total.
// gBodyColor reused as fill cursors during scatter (init_colors overwrites afterward).
// gConstraintIds holds the packed adjacency list (after broadphase hash is done).

uint2 constraintEndpoints(uint id)
{
    uint nJ = gCounters[0].jointCount;
    if (id < nJ)
        return uint2(gJoints[id].bodyA, gJoints[id].bodyB);
    Manifold m = gManifolds[id - nJ];
    return uint2(gColliders[m.colliderA].body, gColliders[m.colliderB].body);
}

#if defined(ENGINE_WGSL)
// CSR edge partners, parallel to the adjacency ids. The candidate buffer is dead
// after narrowphase until the next broadphase and holds four words per candidate,
// at least the 2·(manifolds + joints) adjacency edges. CSR scatter writes each
// edge's partner once; sort permutes it with the ids; coloring then reads one
// word per edge instead of deriving manifold → colliders → bodies every pass.
BIND(10) RWStructuredBuffer<uint> gEdgePartner;
uint csrPartner(uint body, uint e) { return gEdgePartner[e]; }
#else
uint csrPartner(uint body, uint e)
{
    uint2 ends = constraintEndpoints(ALOAD(gConstraintIds[e]));
    return ends.x == body ? ends.y : ends.x;
}
#endif

void csrCountEndpoint(uint body)
{
    if (!bodyIsDynamic(gBodies[body])) return;
    AADD2(CINDEX(body), 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount + 1u;
    if (i < n) ASTORE(CINDEX(i), 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_degree_contacts(uint3 tid : SV_DispatchThreadID)
{
    // Name kept for host pipeline map; degrees manifolds (one edge per pair).
    uint i = dispatchIndex(tid);
    uint n = ALOAD(gCounters[0].manifoldCount);
    if (i >= n) return;
#if defined(ENGINE_PERSISTENT_CONTACTS)
    // Persistent hosts pack solver pairs before the graph (8el): read the two
    // body ids from the 32-byte pair instead of manifold -> two colliders.
    SolverPair sp = gSolverPairs[i];
    if ((sp.countAndSphere & SOLVER_COUNT_MASK) == 0u) return;
    uint2 e = uint2(sp.bodyA, sp.bodyB);
#else
    Manifold m = gManifolds[i];
    if (m.contactCount == 0u) return;
    uint2 e = constraintEndpoints(gCounters[0].jointCount + i);
#endif
    csrCountEndpoint(e.x);
    csrCountEndpoint(e.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_degree_joints(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].jointCount;
    if (i >= n) return;
    if ((gJoints[i].flags & JOINT_FLAG_BROKEN) != 0u) return;
    uint2 e = constraintEndpoints(i);
    csrCountEndpoint(e.x);
    csrCountEndpoint(e.y);
}

// Three-pass blocked scan. Each 64-thread workgroup scans 256 bodies; the
// adjacency-list storage is scratch until scatter, so it holds block offsets.
// All neighbours are read before any in-place publication within a workgroup.
groupshared uint csrScanShared[64];

uint csrScanThreadPrefix(uint local, uint value)
{
    csrScanShared[local] = value;
    GROUP_SYNC();
    for (uint offset = 1u; offset < 64u; offset <<= 1u)
    {
        uint addend = local >= offset ? csrScanShared[local - offset] : 0u;
        GROUP_SYNC();
        csrScanShared[local] += addend;
        GROUP_SYNC();
    }
    return csrScanShared[local] - value;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_blocks(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint thread = dispatchIndex(tid);
    uint base = thread * 4u;
    uint n = gCounters[0].bodyCount;
    uint4 d = uint4(0u, 0u, 0u, 0u);
    if (base < n) d.x = ALOAD(CINDEX(base));
    if (base + 1u < n) d.y = ALOAD(CINDEX(base + 1u));
    if (base + 2u < n) d.z = ALOAD(CINDEX(base + 2u));
    if (base + 3u < n) d.w = ALOAD(CINDEX(base + 3u));
    uint sum = d.x + d.y + d.z + d.w;
    uint prefix = csrScanThreadPrefix(lid.x, sum);
    if (base < n) ASTORE(CINDEX(base), prefix);
    if (base + 1u < n) ASTORE(CINDEX(base + 1u), prefix + d.x);
    if (base + 2u < n) ASTORE(CINDEX(base + 2u), prefix + d.x + d.y);
    if (base + 3u < n) ASTORE(CINDEX(base + 3u), prefix + d.x + d.y + d.z);
    if (lid.x == 63u) ASTORE(gConstraintIds[thread / 64u], prefix + sum);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_block_sums(uint3 lid : SV_GroupThreadID)
{
    uint n = gCounters[0].bodyCount;
    uint blocks = (n + 255u) / 256u;
    uint perLane = (blocks + 63u) / 64u;
    uint start = lid.x * perLane;
    uint end = min(start + perLane, blocks);
    uint sum = 0u;
    for (uint b = start; b < end; b++) sum += ALOAD(gConstraintIds[b]);
    uint prefix = csrScanThreadPrefix(lid.x, sum);
    if (lid.x == 63u) ASTORE(CINDEX(n), prefix + sum);
    // Each lane owns a disjoint interval; no lane reads another lane's stores.
    for (uint b = start; b < end; b++)
    {
        uint degree = ALOAD(gConstraintIds[b]);
        ASTORE(gConstraintIds[b], prefix);
        prefix += degree;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_add(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    uint prefix = ALOAD(CINDEX(i)) + ALOAD(gConstraintIds[i / 256u]);
    ASTORE(CINDEX(i), prefix);
    ASTORE(gBodyColor[i], prefix);
}

// Parallel exclusive prefix (Hillis–Steele). Host binds gPassUniform = 1,2,4,... via
// pass_table (no mid-encoder CPU scratch writes). gBodyColor holds the running sum.
[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_init(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (i >= n) return;
    ASTORE(gBodyColor[i], ALOAD(CINDEX(i))); // degree
}

// Ping-pong Hillis–Steele step. Host packs pass_table[48+log] = offset | (ping<<31)
// (0⇒read bodyColor/write constraintIndex, 1⇒opposite). Fuses old scan+swap
// (~14 fewer launches on Ring 9k).
[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_scan(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (i >= n) return;
    uint packed = PASS_U;
    uint offset = packed & 0x7fffffffu;
    uint ping = packed >> 31;
    uint v;
    if (ping == 0u)
    {
        v = ALOAD(gBodyColor[i]);
        if (i >= offset) v += ALOAD(gBodyColor[i - offset]);
        ASTORE(CINDEX(i), v);
    }
    else
    {
        v = ALOAD(CINDEX(i));
        if (i >= offset) v += ALOAD(CINDEX(i - offset));
        ASTORE(gBodyColor[i], v);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_swap(uint3 tid : SV_DispatchThreadID)
{
    // Finalize wrote exclusive offsets to the buffer opposite the inclusive
    // scan. Publish them to both adjacency starts and scatter cursors only
    // after that dispatch completes; no thread now reads a neighbour we write.
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (i >= n) return;
    if (HOST_S(12) == 0u)
        ASTORE(CINDEX(i), ALOAD(gBodyColor[i]));
    else
        ASTORE(gBodyColor[i], ALOAD(CINDEX(i)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_finalize(uint3 tid : SV_DispatchThreadID)
{
    // Inclusive scan in constraintIndex if last ping was 0, else bodyColor.
    // Host sets gScratchAtomics[12] = last ping (write_scratch).
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (i > n) return;
    uint ping = HOST_S(12);
    if (i == n)
    {
        uint last = 0u;
        if (n > 0u)
            last = (ping == 0u) ? ALOAD(CINDEX(n - 1u)) : ALOAD(gBodyColor[n - 1u]);
        ASTORE(CINDEX(n), last);
        return;
    }
    uint excl = (i == 0u) ? 0u : ((ping == 0u) ? ALOAD(CINDEX(i - 1u)) : ALOAD(gBodyColor[i - 1u]));
    // Never shift the inclusive scan in place: another workgroup may still
    // need index i as its predecessor. Use the other ping-pong buffer, then
    // csr_prefix_swap copies the result in a separate dispatch.
    if (ping == 0u) ASTORE(gBodyColor[i], excl);
    else ASTORE(CINDEX(i), excl);
}

// Back-compat single-thread path for tiny worlds (tests).
[shader("compute")]
[numthreads(1, 1, 1)]
void csr_prefix(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0) return;
    uint n = gCounters[0].bodyCount;
    uint sum = 0;
    for (uint i = 0; i < n; i++)
    {
        uint d = ALOAD(CINDEX(i));
        ASTORE(CINDEX(i), sum);
        ASTORE(gBodyColor[i], sum);
        sum += d;
    }
    ASTORE(CINDEX(n), sum);
}

void csrFillEndpoint(uint body, uint partner, uint id)
{
    if (!bodyIsDynamic(gBodies[body])) return;
    uint slot;
    AADD3(gBodyColor[body], 1, slot);
    ASTORE(gConstraintIds[slot], id);
#if defined(ENGINE_WGSL)
    if (slot < HOST_S(10) * 4u) gEdgePartner[slot] = partner;
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_scatter_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = ALOAD(gCounters[0].manifoldCount);
    if (i >= n) return;
    uint id = gCounters[0].jointCount + i;
#if defined(ENGINE_PERSISTENT_CONTACTS)
    SolverPair sp = gSolverPairs[i];
    if ((sp.countAndSphere & SOLVER_COUNT_MASK) == 0u) return;
    uint2 e = uint2(sp.bodyA, sp.bodyB);
#else
    Manifold m = gManifolds[i];
    if (m.contactCount == 0u) return;
    uint2 e = constraintEndpoints(id);
#endif
    csrFillEndpoint(e.x, e.y, id);
    csrFillEndpoint(e.y, e.x, id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_scatter_joints(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].jointCount;
    if (i >= n) return;
    if ((gJoints[i].flags & JOINT_FLAG_BROKEN) != 0u) return;
    uint2 e = constraintEndpoints(i);
    csrFillEndpoint(e.x, e.y, i);
    csrFillEndpoint(e.y, e.x, i);
}

// Diagnostic control: old append-ID ordering. Atomic emission changes these IDs
// between runs, so this ordering is not a physical/deterministic summation order.
[shader("compute")]
[numthreads(64, 1, 1)]
void csr_sort_append(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    uint awakeN = ALOAD(gCounters[0].awakeCount);
    if (li >= awakeN) return;
    uint b = gAwakeList[li];
    uint lo = ALOAD(CINDEX(b));
    uint hi = ALOAD(CINDEX(b + 1u));
    uint nJ = gCounters[0].jointCount;
    for (uint i = lo + 1u; i < hi; i++)
    {
        uint id = ALOAD(gConstraintIds[i]);
        uint key0 = id < nJ ? 0u : 1u;
        uint key1 = id < nJ ? id : (id - nJ);
        uint j = i;
        while (j > lo)
        {
            uint prev = ALOAD(gConstraintIds[j - 1u]);
            uint pk0 = prev < nJ ? 0u : 1u;
            uint pk1 = prev < nJ ? prev : (prev - nJ);
            if (key0 > pk0 || (key0 == pk0 && key1 >= pk1)) break;
            ASTORE(gConstraintIds[j], prev);
            j--;
        }
        ASTORE(gConstraintIds[j], id);
    }
    if (HOST_S(26) != 0u)
        gColorLists[HOST_S(1) * (HOST_S(4) + 1u) + b] = compactPreviousColor(b);
}

// Joints have stable identities. Contacts are ordered by the neighboring body,
// then the unordered collider pair (compound bodies can have multiple colliders).
// Never use contactStart or manifold append ID as a physical key.
uint4 csrPhysicalKey(uint body, uint id)
{
    uint nJ = gCounters[0].jointCount;
    if (id < nJ) return uint4(0u, id, 0u, 0u);
    Manifold mf = gManifolds[id - nJ];
    uint a = gColliders[mf.colliderA].body;
    uint b = gColliders[mf.colliderB].body;
    return uint4(1u, a == body ? b : a,
        min(mf.colliderA, mf.colliderB), max(mf.colliderA, mf.colliderB));
}

bool csrKeyLess(uint4 a, uint4 b)
{
    return a.x < b.x || (a.x == b.x && (a.y < b.y ||
        (a.y == b.y && (a.z < b.z || (a.z == b.z && a.w < b.w)))));
}

// Same order as csrKeyLess(csrPhysicalKey(...)) but reads the cached partner:
// joints (id order) before manifolds, manifolds by partner body; only edges
// sharing a partner load collider ids for the tie-break.
bool csrEdgeLess(uint body, uint idA, uint partnerA, uint idB, uint partnerB)
{
    uint nJ = gCounters[0].jointCount;
    bool jointA = idA < nJ, jointB = idB < nJ;
    if (jointA != jointB) return jointA;
    if (jointA) return idA < idB;
    if (partnerA != partnerB) return partnerA < partnerB;
    return csrKeyLess(csrPhysicalKey(body, idA), csrPhysicalKey(body, idB));
}

// Max-heap sift. A large impactor may have thousands of incident manifolds;
// unlike insertion sort, heap construction/extraction cannot become quadratic.
void csrSiftDown(uint body, uint lo, uint count, uint root)
{
    uint id = ALOAD(gConstraintIds[lo + root]);
    uint partner = csrPartner(body, lo + root);
    for (uint depth = 0u; depth < 32u; depth++)
    {
        if (root >= count / 2u) break; // includes count 0/1, avoids child overflow
        uint child = root * 2u + 1u;
        uint childId = ALOAD(gConstraintIds[lo + child]);
        uint childPartner = csrPartner(body, lo + child);
        if (child + 1u < count)
        {
            uint rightId = ALOAD(gConstraintIds[lo + child + 1u]);
            uint rightPartner = csrPartner(body, lo + child + 1u);
            if (csrEdgeLess(body, childId, childPartner, rightId, rightPartner))
            {
                child++; childId = rightId; childPartner = rightPartner;
            }
        }
        if (!csrEdgeLess(body, id, partner, childId, childPartner)) break;
        ASTORE(gConstraintIds[lo + root], childId);
        csrSetPartner(lo + root, childPartner);
        root = child;
    }
    ASTORE(gConstraintIds[lo + root], id);
    csrSetPartner(lo + root, partner);
}

void csrSetPartner(uint e, uint partner)
{
#if defined(ENGINE_WGSL)
    gEdgePartner[e] = partner;
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_sort(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= ALOAD(gCounters[0].awakeCount)) return;
    uint body = gAwakeList[li];
    uint lo = ALOAD(CINDEX(body));
    uint hi = ALOAD(CINDEX(body + 1u));
    uint count = hi - lo;
    if (count <= 32u)
    {
        // Most bricks have a short list; insertion avoids heap setup there.
        for (uint i = lo + 1u; i < hi; i++)
        {
            uint id = ALOAD(gConstraintIds[i]);
            uint partner = csrPartner(body, i);
            uint j = i;
            while (j > lo)
            {
                uint prev = ALOAD(gConstraintIds[j - 1u]);
                uint prevPartner = csrPartner(body, j - 1u);
                if (!csrEdgeLess(body, id, partner, prev, prevPartner)) break;
                ASTORE(gConstraintIds[j], prev);
                csrSetPartner(j, prevPartner);
                j--;
            }
            ASTORE(gConstraintIds[j], id);
            csrSetPartner(j, partner);
        }
    }
    else
    {
        // The WGSL safety pass caps each loop at 65,536 iterations. Chunk both
        // sweeps so a high-degree body cannot silently leave an unsorted tail.
        uint parent = count / 2u;
        for (uint batch = 0u; batch < (count + 32767u) / 32768u; batch++)
        {
            uint work = min(parent, 32768u);
            for (uint k = 0u; k < work; k++)
            {
                parent--;
                csrSiftDown(body, lo, count, parent);
            }
            if (parent == 0u) break;
        }
        uint end = count - 1u;
        for (uint batch = 0u; batch < (count + 32767u) / 32768u; batch++)
        {
            uint work = min(end, 32768u);
            for (uint k = 0u; k < work; k++)
            {
                uint largest = ALOAD(gConstraintIds[lo]);
                uint last = ALOAD(gConstraintIds[lo + end]);
                uint largestPartner = csrPartner(body, lo);
                uint lastPartner = csrPartner(body, lo + end);
                ASTORE(gConstraintIds[lo], last);
                ASTORE(gConstraintIds[lo + end], largest);
                csrSetPartner(lo, lastPartner);
                csrSetPartner(lo + end, largestPartner);
                csrSiftDown(body, lo, end, 0u);
                end--;
            }
            if (end == 0u) break;
        }
    }
    if (HOST_S(26) != 0u)
        gColorLists[HOST_S(1) * (HOST_S(4) + 1u) + body] = compactPreviousColor(body);
}

// three-avbd `priority`: hash32(i + 0x9e3779b9) — independent of round / height.
uint jpPriority(uint body)
{
    uint x = body + 0x9e3779b9u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Like the reference's colorCompact: only local priority maxima may reduce
// their retained color, so no adjacent bodies compact simultaneously. Write a
// separate snapshot row; initialization must not observe partial updates.
uint compactPreviousColor(uint body)
{
    uint history = HOST_S(1) * HOST_S(4);
    uint previous = HOST_S(17) == 0u ? 0u : gColorLists[history + body];
    if (previous == 0u || previous > HOST_S(4)) return 0u;
    uint prio = jpPriority(body);
    uint used = 0u;
    uint lo = ALOAD(CINDEX(body));
    uint hi = ALOAD(CINDEX(body + 1u));
    for (uint e = lo; e < hi; e++)
    {
        uint other = csrPartner(body, e);
        if (!bodyIsDynamic(gBodies[other]) || bodyIsSleeping(gBodies[other])) continue;
        uint op = jpPriority(other);
        if (op > prio || (op == prio && other > body)) return previous;
        uint encoded = gColorLists[history + other];
        if (encoded > 0u && encoded <= 32u) used |= 1u << (encoded - 1u);
    }
    uint free = 0u;
    uint bits = ~used;
    while (free < 32u && (bits & 1u) == 0u) { bits >>= 1u; free++; }
    return min(previous, free + 1u);
}

// One Jones-Plassmann attempt for an uncolored body: false while a
// higher-priority uncolored neighbour blocks it.
bool jpTryColor(uint body)
{
    uint prio = jpPriority(body);
    bool blocked = false;
    uint usedLo = 0u;
    uint usedHi = 0u;
    uint lo = ALOAD(CINDEX(body));
    uint hi = ALOAD(CINDEX(body + 1u));
    for (uint e = lo; e < hi; e++)
    {
        uint other = csrPartner(body, e);
        if (!bodyIsDynamic(gBodies[other]) || bodyIsSleeping(gBodies[other])) continue;
        uint oc = ALOAD(gBodyColor[other]);
        if (oc == 0xFFFFFFFFu)
        {
            uint op = jpPriority(other);
            if (op > prio || (op == prio && other > body)) blocked = true;
        }
        else if (oc < 32u) usedLo |= 1u << oc;
        else if (oc < 64u) usedHi |= 1u << (oc - 32u);
    }
    if (blocked) return false;

    // Smallest colour not used by already-coloured neighbours (three-avbd smallestFree).
    uint free = 63u;
    if (~usedLo != 0u)
    {
        free = 0u;
        uint bit = ~usedLo;
        while ((bit & 1u) == 0u) { bit >>= 1u; free++; }
    }
    else if (~usedHi != 0u)
    {
        free = 32u;
        uint bit = ~usedHi;
        while ((bit & 1u) == 0u) { bit >>= 1u; free++; }
    }
    ASTORE(gBodyColor[body], free);
    AMAX2(gCounters[0].colorCount, free + 1);
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void color_bodies(uint3 tid : SV_DispatchThreadID)
{
    // Jones-Plassmann: winners take the *smallest free* colour (three-avbd), not the
    // round index — assigning round forced O(rounds) colours and killed parallelism.
    uint li = dispatchIndex(tid);
    bool incremental = HOST_S(26) != 0u;
    uint workN = incremental ? ALOAD(gCounters[0].recolorCount) : ALOAD(gCounters[0].awakeCount);
    if (li >= workN) return;
    uint body = incremental ? gCcdFlags[li] : gAwakeList[li];
    if (ALOAD(gBodyColor[body]) != 0xFFFFFFFFu) return;
    // PASS_U is the round index: a body still blocked records the round, so
    // the host can size the next step's rounds and the tail knows whether the
    // dispatched rounds finished.
    if (!jpTryColor(body)) AMAX2(gCounters[0].colorRounds, PASS_U + 1u);
}

// Rounds past the host's estimate (PASS_U = rounds dispatched): one workgroup
// keeps running Jones-Plassmann rounds over the same work list while the last
// round still left a body blocked, up to gCounters.colorRoundLimit rounds in
// all (the global rounds' limit), with a storage barrier between rounds.
[shader("compute")]
[numthreads(256, 1, 1)]
void color_bodies_tail(uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    // Publish the round limit (0: the dispatched rounds finished) through
    // workgroup memory so the loop and its barriers stay in uniform control flow.
    if (local == 0u) gsUniformLanes = ALOAD(gCounters[0].colorRounds) >= PASS_U ? ALOAD(gCounters[0].colorRoundLimit) : 0u;
    GROUP_SYNC();
    uint limit = gsUniformLanes;
    if (limit == 0u) return;
    GROUP_SYNC();
    bool incremental = HOST_S(26) != 0u;
    uint workN = incremental ? ALOAD(gCounters[0].recolorCount) : ALOAD(gCounters[0].awakeCount);
    for (uint round = PASS_U; round < limit; round++)
    {
        if (local == 0u) ASTORE(gsColorRemaining, 0u);
        GROUP_SYNC();
        for (uint li = local; li < workN; li += 256u)
        {
            uint body = incremental ? gCcdFlags[li] : gAwakeList[li];
            if (ALOAD(gBodyColor[body]) != 0xFFFFFFFFu) continue;
            if (!jpTryColor(body)) { AOR2(gsColorRemaining, 1u); AMAX2(gCounters[0].colorRounds, round + 1u); }
        }
        DeviceMemoryBarrierWithGroupSync();
        if (local == 0u) gsUniformLanes = ALOAD(gsColorRemaining);
        GROUP_SYNC();
        if (gsUniformLanes == 0u) break;
    }
}

// Opt-in local repair (8en, HOST_S(27) != 0): a small incremental repair list
// is colored by one workgroup looping JP rounds behind storage barriers, and
// write_work_args zeroes slot 69 so the 32 global rounds launch nothing. The
// highest-priority uncolored body always colors, and every uncolored neighbour
// is itself on the list, so each round progresses.
static const uint LOCAL_COLOR_LIMIT = 4096u;
bool localColoring()
{
#if defined(ENGINE_WGSL)
    // WGSL hosts only: Metal's HOST_S(27) is a large-body list word.
    return HOST_S(27) != 0u && HOST_S(26) != 0u &&
        ALOAD(gCounters[0].recolorCount) <= LOCAL_COLOR_LIMIT;
#else
    return false;
#endif
}

groupshared AUint gsColorRemaining;

[shader("compute")]
[numthreads(256, 1, 1)]
void color_bodies_local(uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    if (local == 0u) gsUniformLanes = localColoring() ? 1u : 0u;
    GROUP_SYNC();
    if (gsUniformLanes == 0u) return;
    uint workN = ALOAD(gCounters[0].recolorCount);
    for (uint round = 0u; round < LOCAL_COLOR_LIMIT; round++)
    {
        if (local == 0u) ASTORE(gsColorRemaining, 0u);
        GROUP_SYNC();
        for (uint li = local; li < workN; li += 256u)
        {
            uint body = gCcdFlags[li];
            if (ALOAD(gBodyColor[body]) != 0xFFFFFFFFu) continue;
            if (!jpTryColor(body)) AOR2(gsColorRemaining, 1u);
        }
        DeviceMemoryBarrierWithGroupSync();
        if (local == 0u) gsUniformLanes = ALOAD(gsColorRemaining);
        GROUP_SYNC();
        if (gsUniformLanes == 0u) break;
    }
}

// Adaptive 1/2/4/8 threads per body by colour size. This tuning differs from
// pinned three-avbd, whose one-lane threshold is 2^31 rather than 65536.
uint primalLanesFor(uint bodies)
{
    // Never one lane per body: gables' ~184k-body colors measured 1/2/4/8 lanes
    // at 32.2/31.1/30.9/31.4 ms (decision 8eg); the ring is neutral.
    if (bodies == 0u) return 1u;
    // Convex scenes (HOST_S(29)): hull bodies walk several manifolds each, and
    // 8 lanes hide that latency (49-house grid peak: 4/8/16 lanes measured
    // 11.3/10.0/11.7 ms); box scenes keep their validated 4.
    if (bodies >= 4096u) return HOST_S(29) != 0u ? 8u : 4u;
    return 8u;
}

// Degree-aware lanes (opt-in, HOST_S(29) != 0; host sizes gColorCounts to 128
// words): gColorCounts[64 + c] holds colour c's largest CSR degree. A compound
// body with hundreds of manifolds otherwise holds its whole colour dispatch
// while 8 lanes walk its edges serially. Widen to 16/32/64 lanes when the
// colour is small enough that the extra threads stay under 64k.
static const uint COLOR_DEGREE_BASE = 64u;
uint primalLanesForDegree(uint bodies, uint maxDegree)
{
    uint lanes = primalLanesFor(bodies);
    if (HOST_S(29) == 0u || bodies == 0u) return lanes;
    uint want = lanes;
    while (want < 64u && maxDegree > want * 8u) want *= 2u;
    while (want > lanes && bodies * want > 65536u) want /= 2u;
    return max(lanes, want);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void record_color_degrees(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= ALOAD(gCounters[0].awakeCount)) return;
    uint body = gAwakeList[li];
    uint color = ALOAD(gBodyColor[body]);
    if (color >= min(HOST_S(4), COLOR_DEGREE_BASE)) return;
    uint degree = ALOAD(CINDEX(body + 1u)) - ALOAD(CINDEX(body));
    AMAX2(gColorCounts[COLOR_DEGREE_BASE + color], degree);
}

// Slots 0..63 are colour dispatches. Slots 64..67 are the *current* step's
// candidate, contact/joint, manifold and awake counts; slot 68 is previous contacts.
// Hosts opting into incremental coloring (HOST_S(26)) also allocate slot 69.
// CPU estimates from a prior
// frame can miss most of the first ring/gables contacts and leave them unsolved.
void writeDispatchArgs(uint slot, uint threads)
{
    uint groups = (threads + 63u) / 64u;
    gIndirectArgs[slot * 3u] = min(groups, 32768u);
    gIndirectArgs[slot * 3u + 1u] = max(1u, (groups + 32767u) / 32768u);
    gIndirectArgs[slot * 3u + 2u] = 1u;
}

// Re-emit last step's resting pairs as candidates (8ep). Runs after the
// broadphase and before narrowphase overwrites gManifolds; dispatched over
// work slot 72 (HOST_S(13) = previous GPU manifold count). Narrowphase reuses
// them through the at-rest path, which copies the contacts unchanged.
[shader("compute")]
[numthreads(64, 1, 1)]
void carry_rest_pairs(uint3 tid : SV_DispatchThreadID)
{
    if (!carryRestPairs() || HOST_S(17) == 0u) return;
    uint m = dispatchIndex(tid);
    if (m >= min(HOST_S(13), HOST_S(10))) return;
    Manifold mf = gManifolds[m];
    if (mf.contactCount == 0u) return;
    BodyState ba = gBodies[gColliders[mf.colliderA].body];
    BodyState bb = gBodies[gColliders[mf.colliderB].body];
    if (!bodyIsDynamic(ba) && !bodyIsDynamic(bb)) return;
    if (!contactPairAtRest(ba, bb)) return;
    emitCandidate(mf.colliderA, mf.colliderB);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void write_work_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    writeDispatchArgs(64u, min(ALOAD(gCounters[0].candidateCount), HOST_S(2)));
    writeDispatchArgs(65u, max(min(ALOAD(gCounters[0].contactCount), HOST_S(3)), gCounters[0].jointCount));
    writeDispatchArgs(66u, min(ALOAD(gCounters[0].manifoldCount), HOST_S(10)));
    writeDispatchArgs(67u, ALOAD(gCounters[0].awakeCount));
    // Slot 68 = warm-hash insert work: previous manifolds (opt-in) or contacts.
    writeDispatchArgs(68u, HOST_S(14) != 0u ? min(HOST_S(13), HOST_S(10))
                                            : min(HOST_S(7), HOST_S(3)));
    // Opt-in keeps hosts without a history row / slot 69 compatible.
    // Local repair (HOST_S(27)) colors small lists itself; slot 69 then stays empty.
    if (HOST_S(26) != 0u)
        writeDispatchArgs(69u, localColoring() ? 0u : ALOAD(gCounters[0].recolorCount));
    // Resting-pair carry (8ep): previous manifolds, published by the host.
    if (carryRestPairs()) writeDispatchArgs(72u, min(HOST_S(13), HOST_S(10)));
    // Slot 73: previous contacts, for whole-step manifold reuse (any hash mode).
    writeDispatchArgs(73u, min(HOST_S(7), HOST_S(3)));
#if defined(ENGINE_REUSE_COPY)
    // Persist work list (8em): slot 70 = listed contacts, slot 71 = the full
    // scan, which runs only when the list overflowed.
    if (persistListEnabled())
    {
        uint listed = ALOAD(gCounters[0]._pad2);
        bool complete = listed <= persistListCapacity();
        writeDispatchArgs(70u, complete ? listed : 0u);
        writeDispatchArgs(71u, complete ? 0u : min(ALOAD(gCounters[0].contactCount), HOST_S(3)));
    }
#endif
}

// Write Metal indirect threadgroup counts for each colour (0 groups ⇒ skipped).
[shader("compute")]
[numthreads(64, 1, 1)]
void write_color_args(uint3 tid : SV_DispatchThreadID)
{
    uint c = dispatchIndex(tid);
    // Hosts launch a full workgroup, but allocate only maxColors counts (usually
    // 32). Metal has no WGSL-style robust buffer access: guard before the load.
    // Slots 64+ belong to work dispatches and must never be overwritten here.
    if (c >= min(HOST_S(4), 64u)) return;
    uint n = ALOAD(gColorCounts[c]);
    if (n != 0u) AMAX2(gCounters[0].colorCount, c + 1u);
    uint lanes = primalLanesForDegree(
        n, HOST_S(29) != 0u ? ALOAD(gColorCounts[COLOR_DEGREE_BASE + c]) : 0u);
    uint threads = n * lanes;
    writeDispatchArgs(c, threads);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init_colors(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    uint awakeN = ALOAD(gCounters[0].awakeCount);
    if (li >= awakeN) return;
    uint body = gAwakeList[li];
    uint color = 0xFFFFFFFFu;
    if (HOST_S(26) != 0u)
    {
        // The extra row lies beyond both packed colors and broadphase scratch.
        // History encodes color+1; zeroed/new bodies have no valid history.
        uint history = HOST_S(1) * (HOST_S(4) + 1u);
        uint previous = HOST_S(17) == 0u ? 0u : gColorLists[history + body];
        if (previous > 0u && previous <= HOST_S(4))
        {
            color = previous - 1u;
            uint prio = jpPriority(body);
            uint lo = ALOAD(CINDEX(body));
            uint hi = ALOAD(CINDEX(body + 1u));
            for (uint e = lo; e < hi; e++)
            {
                uint other = csrPartner(body, e);
                if (!bodyIsDynamic(gBodies[other]) || bodyIsSleeping(gBodies[other])) continue;
                // Read immutable history, never this dispatch's partially
                // initialized colors. A new conflict clears its lower-priority
                // endpoint even if the winner itself is pending elsewhere.
                if (gColorLists[history + other] == previous)
                {
                    uint op = jpPriority(other);
                    if (op > prio || (op == prio && other > body))
                    {
                        color = 0xFFFFFFFFu;
                        break;
                    }
                }
            }
        }
        if (color == 0xFFFFFFFFu)
        {
            uint slot;
            AADD3(gCounters[0].recolorCount, 1u, slot);
            // extract_velocity_sleep clears every body slot before CCD/sleep
            // uses this buffer later in the step.
            gCcdFlags[slot] = body;
        }
    }
    ASTORE(gBodyColor[body], color);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pack_color_lists(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    uint awakeN = ALOAD(gCounters[0].awakeCount);
    if (li >= awakeN) return;
    uint body = gAwakeList[li];
    if ((gBodies[body].flags & BODY_PARTICLE) != 0u) return; // soft_primal solves particles
    uint color = ALOAD(gBodyColor[body]);
    // Never turn an unfinished graph into racing colour-zero solves.
    if (color >= HOST_S(4))
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_COLORS);
        return;
    }
    uint slot;
    AADD3(gColorCounts[color], 1, slot);
    // Second pass needed for offsets — host compact or prefix. Store body at atomic slot in scratch region.
    uint maxB = HOST_S(1);
    gColorLists[color * maxB + slot] = body;
    if (HOST_S(26) != 0u) gColorLists[maxB * HOST_S(4) + body] = color + 1u;
}

// Stable body-index buckets preserve locality for every primal iteration. The
// old atomic append above remains available to hosts with the original buffers.
// gColorOffsets holds color-major per-64-body tile counts, then exclusive starts.
// Coloring has finished, so its pending-list use of gCcdFlags is over; keep each
// body's tile rank there until scatter. Velocity extraction clears it before sleep.
groupshared AUint gsPackHistogram[64];
groupshared uint gsPackColors[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void count_color_tiles(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint body = dispatchIndex(tid);
    uint local = lid.x;
    uint palette = min(HOST_S(4), 64u);
    uint n = gCounters[0].bodyCount;
    uint groups = (n + 63u) / 64u;
    ASTORE(gsPackHistogram[local], 0u);
    uint color = 0xFFFFFFFFu;
    if (body < n)
    {
        BodyState b = gBodies[body];
        if (bodyIsDynamic(b) && !bodyIsSleeping(b) && (b.flags & BODY_PARTICLE) == 0u)
        {
            color = ALOAD(gBodyColor[body]);
            if (color >= palette)
            {
                AOR2(gCounters[0].overflowFlags, OVERFLOW_COLORS);
                color = 0xFFFFFFFFu;
            }
        }
    }
    gsPackColors[local] = color;
    GROUP_SYNC();
    if (color != 0xFFFFFFFFu)
    {
        uint rank = 0u;
        for (uint j = 0u; j < local; j++)
            if (gsPackColors[j] == color) rank++;
        gCcdFlags[body] = rank;
        AADD2(gsPackHistogram[color], 1u);
    }
    GROUP_SYNC();
    uint group = body / 64u;
    if (local < palette && group < groups)
        gColorOffsets[local * groups + group] = ALOAD(gsPackHistogram[local]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prefix_color_tiles(uint3 tid : SV_DispatchThreadID)
{
    uint color = dispatchIndex(tid);
    if (color >= min(HOST_S(4), 64u)) return;
    uint groups = (gCounters[0].bodyCount + 63u) / 64u;
    uint sum = 0u;
    for (uint group = 0u; group < groups; group++)
    {
        uint index = color * groups + group;
        // Read before the store (see prefix_color_tiles_parallel).
        uint next = sum + gColorOffsets[index];
        gColorOffsets[index] = sum;
        sum = next;
    }
    ASTORE(gColorCounts[color], sum);
}

// One complete workgroup per color. Each lane owns a contiguous tile range,
// scans its range twice, and exchanges only one block sum with its peers.
// In-place output is safe because ranges are disjoint; empty ranges contribute 0.
groupshared uint gsPackBlockSums[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void prefix_color_tiles_parallel(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint color = gid.x;
    if (color >= min(HOST_S(4), 64u)) return; // uniform for the entire workgroup
    uint local = lid.x;
    uint groups = (gCounters[0].bodyCount + 63u) / 64u;
    uint chunk = (groups + 63u) / 64u;
    uint start = min(local * chunk, groups);
    uint end = min(start + chunk, groups);
    uint sum = 0u;
    for (uint group = start; group < end; group++)
        sum += gColorOffsets[color * groups + group];
    gsPackBlockSums[local] = sum;
    GROUP_SYNC();
    for (uint offset = 1u; offset < 64u; offset *= 2u)
    {
        uint add = local >= offset ? gsPackBlockSums[local - offset] : 0u;
        GROUP_SYNC();
        gsPackBlockSums[local] += add;
        GROUP_SYNC();
    }
    uint prefix = gsPackBlockSums[local] - sum;
    for (uint group = start; group < end; group++)
    {
        uint index = color * groups + group;
        // Use the count before overwriting it: Slang's CUDA output otherwise
        // keeps `count` as a pointer and reads it back after the store.
        uint next = prefix + gColorOffsets[index];
        gColorOffsets[index] = prefix;
        prefix = next;
    }
    if (local == 63u) ASTORE(gColorCounts[color], gsPackBlockSums[63]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter_color_tiles(uint3 tid : SV_DispatchThreadID)
{
    uint body = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (body >= n) return;
    BodyState b = gBodies[body];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b) || (b.flags & BODY_PARTICLE) != 0u) return;
    uint color = ALOAD(gBodyColor[body]);
    if (color >= min(HOST_S(4), 64u)) return; // count pass reports overflow
    uint groups = (n + 63u) / 64u;
    uint slot = gColorOffsets[color * groups + body / 64u] + gCcdFlags[body];
    uint maxB = HOST_S(1);
    gColorLists[color * maxB + slot] = body;
    if (HOST_S(26) != 0u) gColorLists[maxB * HOST_S(4) + body] = color + 1u;
}

#if defined(ENGINE_COMPACT_SOLVE)
// Compact solver pairs cache collider-to-body indirection once per step.
// Canonical manifolds retain collider identity for compound shapes/persistence.
struct SolverPair
{
    uint bodyA; uint bodyB; uint contactStart; uint countAndSphere;
    float3 normal; float friction;
};
BIND(28) RWStructuredBuffer<SolverPair> gSolverPairs;
static const uint SOLVER_SPHERE = 0x80000000u;
// A track-belt pair (pairBeltDisp). The solve kernels have no binding to spare for
// colliders, so its belt travel is packed here, once per step, in 16-bit fixed
// point (no half intrinsics: Slang's CUDA output lacks cuda_fp16): μ moves to bits
// 14..29 of countAndSphere, and `friction` holds the travel along (t1, t2).
// Other pairs keep the plain layout.
static const uint SOLVER_BELT = 0x40000000u;
static const uint SOLVER_COUNT_MASK = 0x3FFFu;
static const float BELT_MU_SCALE = 4096.0;   // μ in [0, 16), steps of 2.4e-4
static const float BELT_TRAVEL_MAX = 2.0;    // m per step, steps of 6e-5 m

uint packUnorm16(float v, float scale) { return (uint)clamp(round(v * scale), 0.0, 65535.0); }
uint packSnorm16(float v) { return (uint)(clamp(round(v / BELT_TRAVEL_MAX * 32767.0), -32767.0, 32767.0) + 32768.0); }
float unpackSnorm16(uint q) { return (float(q & 0xFFFFu) - 32768.0) / 32767.0 * BELT_TRAVEL_MAX; }

void packSolverPair(uint mi, Manifold mf)
{
    Collider a = gColliders[mf.colliderA];
    Collider b = gColliders[mf.colliderB];
    SolverPair pair;
    pair.bodyA = a.body; pair.bodyB = b.body;
    pair.contactStart = mf.contactStart;
    pair.countAndSphere = mf.contactCount |
        ((a.shape == SHAPE_SPHERE || b.shape == SHAPE_SPHERE) ? SOLVER_SPHERE : 0u);
    pair.normal = mf.normal; pair.friction = mf.friction;
    if (colliderIsBeltLazy(a) || colliderIsBeltLazy(b))
    {
        float3 belt = pairBeltDisp(a, b, gBodies[a.body], gBodies[b.body], mf.normal);
        float3 t1, t2;
        orthonormalBasis(mf.normal, t1, t2);
        pair.countAndSphere |= SOLVER_BELT | (packUnorm16(mf.friction, BELT_MU_SCALE) << 14u);
        pair.friction = asfloat(packSnorm16(dot(belt, t1)) | (packSnorm16(dot(belt, t2)) << 16u));
    }
    gSolverPairs[mi] = pair;
}

float solverPairFriction(SolverPair pair)
{
    return (pair.countAndSphere & SOLVER_BELT) != 0u
        ? float((pair.countAndSphere >> 14u) & 0xFFFFu) / BELT_MU_SCALE : pair.friction;
}

// World belt travel of a pair this step (pairBeltDisp), from its packed halves.
float3 solverPairBelt(SolverPair pair)
{
    if ((pair.countAndSphere & SOLVER_BELT) == 0u) return float3(0, 0, 0);
    float3 t1, t2;
    orthonormalBasis(pair.normal, t1, t2);
    uint bits = asuint(pair.friction);
    return t1 * unpackSnorm16(bits) + t2 * unpackSnorm16(bits >> 16u);
}

Contact solverContact(uint index, float3 normal, float friction)
{
    SolverPoint pt = loadSolverPoint(index);
    Contact c = (Contact)0;
    c.alive = (pt.pairAndStick & SOLVER_DEAD) != SOLVER_DEAD ? 1u : 0u;
    c.normal = normal; c.friction = friction;
    orthonormalBasis(c.normal, c.tangent1, c.tangent2);
    c.anchorA = pt.anchorA; c.anchorB = pt.anchorB;
    c.c0N = pt.c0N; c.c0T1 = pt.c0T1; c.c0T2 = pt.c0T2;
    c.lambdaN = pt.lambda.x; c.lambdaT1 = pt.lambda.y; c.lambdaT2 = pt.lambda.z;
    c.kN = pt.penalty.x; c.kT = pt.penalty.y; c._pad2 = pt.penalty.z;
    c._pad0 = (pt.pairAndStick & SOLVER_STICK) != 0u ? 1.0 : 0.0;
    return c;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_solver_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i < ALOAD(gCounters[0].contactCount)) gSolverPoints[i].pairAndStick = SOLVER_DEAD;
}

void packSolverPoint(uint index, uint mi, Contact c)
{
    SolverPoint pt;
    pt.anchorA = c.anchorA; pt.c0N = c.c0N;
    pt.anchorB = c.anchorB; pt.c0T1 = c.c0T1;
    pt.lambda = float3(c.lambdaN, c.lambdaT1, c.lambdaT2); pt.c0T2 = c.c0T2;
    pt.penalty = float3(c.kN, c.kT, c._pad2);
    pt.pairAndStick = (c.alive != 0u ? mi : SOLVER_DEAD) |
        (c._pad0 != 0.0 ? SOLVER_STICK : 0u);
    gSolverPoints[index] = pt;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pack_solver_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint mi = dispatchIndex(tid);
    if (mi >= ALOAD(gCounters[0].manifoldCount)) return;
    Manifold mf = gManifolds[mi];
#if defined(ENGINE_COMPACT_SOLVE)
    packSolverPair(mi, mf);
#endif
#if !defined(ENGINE_PERSISTENT_CONTACTS)
    uint cur = HOST_S(11);
    for (uint k = 0u; k < mf.contactCount; k++)
    {
        uint index = mf.contactStart + k;
        Contact c = READ_CONTACT(index);
        packSolverPoint(index - cur, mi, c);
    }
#endif
}
#endif

// Vehicle block: 4 Aabb records (32 floats) = lower-triangle H (21, primalReduceStep
// order), gradient g (6), active flag. Written by vehicle_step.slang kernels.
void addVehicleBlock(inout Mat6 A, inout Vec6 rhs, uint at)
{
    Aabb e3 = gAabbs[at + 3u];
    if (e3._pad0 == 0.0) return; // float 27: block inactive
    Aabb e0 = gAabbs[at];
    Aabb e1 = gAabbs[at + 1u];
    Aabb e2 = gAabbs[at + 2u];
    A.m[0] += e0.mn.x;
    A.m[6] += e0.mn.y;
    A.m[7] += e0.mn.z;
    A.m[12] += e0._pad0;
    A.m[13] += e0.mx.x;
    A.m[14] += e0.mx.y;
    A.m[18] += e0.mx.z;
    A.m[19] += e0._pad1;
    A.m[20] += e1.mn.x;
    A.m[21] += e1.mn.y;
    A.m[24] += e1.mn.z;
    A.m[25] += e1._pad0;
    A.m[26] += e1.mx.x;
    A.m[27] += e1.mx.y;
    A.m[28] += e1.mx.z;
    A.m[30] += e1._pad1;
    A.m[31] += e2.mn.x;
    A.m[32] += e2.mn.y;
    A.m[33] += e2.mn.z;
    A.m[34] += e2._pad0;
    A.m[35] += e2.mx.x;
    rhs.v[0] += e2.mx.y;
    rhs.v[1] += e2.mx.z;
    rhs.v[2] += e2._pad1;
    rhs.v[3] += e3.mn.x;
    rhs.v[4] += e3.mn.y;
    rhs.v[5] += e3.mn.z;
}

// Accumulate body Newton's system for adjacency entries lane, lane+lanes, ...
// Lane 0 also adds the inertial mass block (three-avbd accumulate).
void primalAccumulate(uint bodyIdx, uint lane, uint lanes, out Mat6 A, out Vec6 rhs)
{
    A = mat6Zero();
    rhs = vec6Zero();
    BodyState b = gBodies[bodyIdx];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    AvbdParams p = PARAMS;
    float dt = p.dt;
    float dt2 = dt * dt;

    if (lane == 0u)
    {
        float3 inertial = gAabbs[bodyIdx].mn;
        float mass = b.invMass > 1e-12 ? (1.0 / b.invMass) : 0.0;
        float3 inertia = float3(
            b.invInertia.x > 1e-12 ? (1.0 / b.invInertia.x) : 0.0,
            b.invInertia.y > 1e-12 ? (1.0 / b.invInertia.y) : 0.0,
            b.invInertia.z > 1e-12 ? (1.0 / b.invInertia.z) : 0.0);
        float3 dp = b.position - inertial;
        rhs.v[0] = (mass / dt2) * dp.x;
        rhs.v[1] = (mass / dt2) * dp.y;
        rhs.v[2] = (mass / dt2) * dp.z;
        Aabb stash = gAabbs[bodyIdx];
        float4 inertialAng = quatNormalize(float4(stash.mx.x, stash.mx.y, stash.mx.z, stash._pad0));
        float4 q = b.orientation;
        if (dot(q, inertialAng) < 0.0) inertialAng = -inertialAng;
        // three-avbd qsub(rot, inertialRot) = 2·vec(rot·conj(inertial)) — world-frame
        // chart matching left-multiply quatApplyDelta / qadd. The conj-first order
        // exploded randomly oriented Box Piles while axis-aligned stacks stayed fine.
        float4 dq = quatMul(q, quatConjugate(inertialAng));
        float3 dangTarget = 2.0 * dq.xyz;
        if (dq.w < 0.0) dangTarget = -dangTarget;
        float dangLen = length(dangTarget);
        if (dangLen > 0.5) dangTarget *= 0.5 / dangLen;
        if ((p.flags & FLAG_WORLD_INERTIA) != 0u)
        {
            mat6AddDiag(A, float3(mass, mass, mass) / dt2, float3(0, 0, 0));
            float3 e0 = quatRotate(inertialAng, float3(1, 0, 0));
            float3 e1 = quatRotate(inertialAng, float3(0, 1, 0));
            float3 e2 = quatRotate(inertialAng, float3(0, 0, 1));
            float3 i = inertia / dt2;
            float3 Iw[3];
            for (int r = 0; r < 3; r++)
                Iw[r] = i.x * e0[r] * e0 + i.y * e1[r] * e1 + i.z * e2[r] * e2;
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++) A.m[(3 + r) * 6 + 3 + c] += Iw[r][c];
                rhs.v[3 + r] = dot(Iw[r], dangTarget);
            }
        }
        else
        {
            mat6AddDiag(A, float3(mass, mass, mass) / dt2, inertia / dt2);
            rhs.v[3] = (inertia.x / dt2) * dangTarget.x;
            rhs.v[4] = (inertia.y / dt2) * dangTarget.y;
            rhs.v[5] = (inertia.z / dt2) * dangTarget.z;
        }
        // Vehicle chassis: wheel rows are unary, so vehicle_prepare/iterate fold them
        // (wheel spin condensed) into one 6×6 block at this exact pose.
        if ((b.flags & BODY_VEHICLE) != 0u) addVehicleBlock(A, rhs, asuint(b.initialPad));
    }

    uint nJ = gCounters[0].jointCount;
    uint alo = SOLVE_INDEX(bodyIdx);
    uint ahi = SOLVE_INDEX(bodyIdx + 1u);
    for (uint e = alo + lane; e < ahi; e += lanes)
    {
        uint id = SOLVE_ID(e);
        if (id >= nJ)
        {
            // Manifold edge: load both bodies once, fold every contact point (three-avbd).
#if defined(ENGINE_COMPACT_SOLVE)
            SolverPair pair = gSolverPairs[id - nJ];
            uint contactCount = pair.countAndSphere & SOLVER_COUNT_MASK;
            uint contactStart = pair.contactStart;
            if (contactCount == 0u) continue;
            uint ba = pair.bodyA, bb = pair.bodyB;
            bool anySphere = (pair.countAndSphere & SOLVER_SPHERE) != 0u;
#else
            Manifold mf = gManifolds[id - nJ];
            uint contactCount = mf.contactCount;
            uint contactStart = mf.contactStart;
            if (contactCount == 0u) continue;
            Collider colA = gColliders[mf.colliderA];
            Collider colB = gColliders[mf.colliderB];
            uint ba = colA.body, bb = colB.body;
            bool anySphere = colA.shape == SHAPE_SPHERE || colB.shape == SHAPE_SPHERE;
#endif
            bool isA = ba == bodyIdx;
            BodyState bA = gBodies[ba];
            BodyState bB = gBodies[bb];
            // Track belt: the pair's tangential target this step (zero off belts).
            float3 belt = float3(0, 0, 0);
#if defined(ENGINE_COMPACT_SOLVE)
            belt = solverPairBelt(pair);
#else
            belt = pairBeltDisp(colA, colB, bA, bB, mf.normal);
#endif
            // Sphere lever arms use step-start rotation; static poses stay current.
            float4 rotA = (anySphere && bodyIsDynamic(bA)) ? bA.initialRot : bA.orientation;
            float4 rotB = (anySphere && bodyIsDynamic(bB)) ? bB.initialRot : bB.orientation;
            float4 rotSelf = isA ? rotA : rotB;
            // Displacement from x⁻ stored on the body (predict). Static/sleeping stay at 0.
            bool moveA = bodyIsDynamic(bA) && !bodyIsSleeping(bA);
            bool moveB = bodyIsDynamic(bB) && !bodyIsSleeping(bB);
            float3 dLinA = moveA ? (bA.position - bA.initialPos) : float3(0, 0, 0);
            float3 dLinB = moveB ? (bB.position - bB.initialPos) : float3(0, 0, 0);
            float3 dAngA = float3(0, 0, 0);
            float3 dAngB = float3(0, 0, 0);
            if (moveA)
            {
                float4 qA = bA.orientation;
                if (dot(qA, bA.initialRot) < 0.0) qA = -qA;
                dAngA = quatSubVec(qA, bA.initialRot);
            }
            if (moveB)
            {
                float4 qB = bB.orientation;
                if (dot(qB, bB.initialRot) < 0.0) qB = -qB;
                dAngB = quatSubVec(qB, bB.initialRot);
            }
            float keep = 1.0 - primalAlpha();
            float rollN = 0.0, rollArm = 0.0;
            float3 rollNormal = float3(0, 0, 0);
            for (uint ci = 0u; ci < contactCount; ci++)
            {
                #if defined(ENGINE_COMPACT_SOLVE)
                Contact c = solverContact(contactStart + ci - HOST_S(11), pair.normal, solverPairFriction(pair));
#else
                Contact c = READ_CONTACT(contactStart + ci);
#endif
                if (c.alive == 0) continue;
                float3 rAW = quatRotate(rotA, c.anchorA);
                float3 rBW = quatRotate(rotB, c.anchorB);
                // n = A→B. three-avbd stores n=B→A; signs flip so
                // C = c0*(1-α) + n·(dB-dA) - (rA×n)·dAngA + (rB×n)·dAngB.
                float3 n = c.normal;
                float3 t1 = c.tangent1;
                float3 t2 = c.tangent2;
                float3 C = float3(
                    contactC0NReg(c.c0N, keep) + dot(n, dLinB - dLinA) - dot(cross(rAW, n), dAngA) + dot(cross(rBW, n), dAngB),
                    c.c0T1 * keep + dot(t1, dLinB - dLinA + belt) - dot(cross(rAW, t1), dAngA) + dot(cross(rBW, t1), dAngB),
                    c.c0T2 * keep + dot(t2, dLinB - dLinA + belt) - dot(cross(rAW, t2), dAngA) + dot(cross(rBW, t2), dAngB));
                c.penetration = C.x;
                // C is already Ĉ; alpha=0 + margin=0 ⇒ contributeContact uses F = k·C + λ.
                if ((p.flags & FLAG_SLIDE_HESSIAN) != 0u)
                {
                    // A sliding row's force is fixed at μ|f_N| along the slip, so its
                    // stiffness is the secant μ|f_N|/|C_T| (IPC's lagged friction
                    // Hessian), not the stick penalty that would damp the step.
                    float3 fRaw = float3(c.kN, c.kT, c._pad2) * C + float3(c.lambdaN, c.lambdaT1, c.lambdaT2);
                    float fN = max(-min(fRaw.x, 0.0), 0.0);
                    float slip = length(C.yz);
                    if (length(fRaw.yz) > c.friction * fN && slip > 1.0e-9)
                    {
                        float kSec = c.friction * fN / slip;
                        // Keep the force (k·C + λ, then the cone) unchanged: only the
                        // Hessian uses the secant. contributeContact uses the same k for
                        // both, so pass the force through λ with k = kSec.
                        float3 f = projectContactForce(fRaw, c.friction);
                        c.lambdaT1 = f.y - min(c.kT, kSec) * C.y;
                        c.lambdaT2 = f.z - min(c._pad2, kSec) * C.z;
                        c.kT = min(c.kT, kSec);
                        c._pad2 = min(c._pad2, kSec);
                    }
                }
                if ((p.flags & FLAG_ROLLING_RESISTANCE) != 0u)
                {
                    float fN = max(-(c.kN * C.x + c.lambdaN), 0.0);
                    rollN += fN;
                    rollArm = max(rollArm, length(isA ? rAW : rBW));
                    rollNormal = n;
                }
                contributeContact(A, rhs, c, isA, 0.0, b.position, rotSelf, C, 0.0,
                    isA ? rAW : rBW);
            }
            if ((p.flags & FLAG_ROLLING_RESISTANCE) != 0u && rollN > 0.0)
            {
                // Relative rolling this step (twist about the normal excluded).
                float3 roll = dAngB - dAngA;
                roll -= rollNormal * dot(roll, rollNormal);
                float angle = length(roll);
                float cap = ROLLING_RESISTANCE * rollN * rollArm;
                if (angle > 1.0e-9 && cap > 0.0)
                {
                    float3 axis = roll / angle;
                    // At most stop this step's relative rolling, never reverse it
                    // (as Box2D clamps the impulse): stiffness I_axis/h² of this
                    // body about the axis, the torque capped at μr·N·r.
                    float3 local = quatRotate(quatConjugate(b.orientation), axis);
                    float invIAxis = max(dot(local * local, b.invInertia), 1.0e-12);
                    float kHold = 1.0 / (invIAxis * p.dt * p.dt);
                    float torque = min(kHold * angle, cap);
                    float kRow = torque / angle;
                    Vec6 ja = vec6Zero();
                    float sgn = isA ? -1.0 : 1.0;
                    ja.v[3] = sgn * axis.x; ja.v[4] = sgn * axis.y; ja.v[5] = sgn * axis.z;
                    mat6AddOuter(A, ja, kRow);
                    vec6AddScaled(rhs, ja, torque);
                }
            }
        }
        else
        {
            uint jflags = gJoints[id].flags;
            if ((jflags & JOINT_FLAG_BROKEN) != 0u) continue;
            if ((jflags & JOINT_FLAG_TYRE) != 0u)
            {
                // Wheel on this body: its share of the tyre rows, assembled by the
                // vehicle kernels at both bodies' current iterates (ground block
                // index in motion). The chassis side is already in the chassis's
                // own vehicle block.
                uint ta = gJoints[id].bodyA;
                if (gJoints[id].bodyB == bodyIdx && ta != bodyIdx) addVehicleBlock(A, rhs, gJoints[id].motion);
                continue;
            }
            if (jointHeaderIsBall(jflags, gJoints[id].motion))
            {
                JointBall jb = loadJointBall(id);
                jointContributeBall(A, rhs, jb, gBodies[jb.bodyA], gBodies[jb.bodyB], jb.bodyA == bodyIdx, p.alpha, p.kStart);
            }
            else
            {
                Joint j = gJoints[id];
                jointContributePrimal(A, rhs, j, gBodies[j.bodyA], gBodies[j.bodyB], j.bodyA == bodyIdx, p.alpha, p.dt, p.kStart,
                                      (p.flags & FLAG_BOND_GEOMETRIC) != 0u);
            }
        }
    }
}

// BDF1 velocity from the step-start pose x⁻ saved by prediction.
float3 bodyLinearVelocity(BodyState b, float dt)
{
    return (b.position - b.initialPos) / max(dt, 1e-8);
}
float3 bodyAngularVelocity(BodyState b, float dt)
{
    float4 q0 = b.initialRot;
    float4 q1 = b.orientation;
    if (dot(q0, q1) < 0.0) q1 = -q1; // same hemisphere
    float4 dq = quatMul(q1, quatConjugate(q0));
    float3 av = 2.0 * dq.xyz / max(dt, 1e-8);
    return dq.w < 0.0 ? -av : av;
}

void primalApply(uint bodyIdx, Mat6 A, Vec6 rhs)
{
    BodyState b = gBodies[bodyIdx];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b)) return;
    AvbdParams p = PARAMS;
    uint floors = 0;
    Vec6 neg = rhs;
    for (int k = 0; k < 6; k++) neg.v[k] = -rhs.v[k];
    Vec6 dz = solve6x6(A, neg, p.pivotFloor, floors);
    if (floors > 0) AADD2(gCounters[0].pivotFloorHits, floors);
    float3 dlin = float3(dz.v[0], dz.v[1], dz.v[2]);
    float3 dang = float3(dz.v[3], dz.v[4], dz.v[5]);
    float r = 0.5;
    float maxLin = p.trustLin * r;
    float len = length(dlin);
    if (len > maxLin && len > 1e-12) dlin *= maxLin / len;
    float alen = length(dang);
    if (alen > p.trustAng && alen > 1e-12) dang *= p.trustAng / alen;
    if (dlin.x != dlin.x || dlin.y != dlin.y || dlin.z != dlin.z) return;
    if (dang.x != dang.x || dang.y != dang.y || dang.z != dang.z) return;
    b.position += dlin;
    // three-avbd qadd: normalize(q + ½·(δω,0)·q) — first-order chart matching
    // Taylor C / qsub. Full exp-map quatApplyDelta disagrees for large δω.
    b.orientation = quatNormalize(b.orientation + quatMul(float4(dang, 0.0), b.orientation) * 0.5);
    gBodies[bodyIdx].position = b.position;
    gBodies[bodyIdx].orientation = b.orientation;
}

void solvePrimalForBody(uint bodyIdx)
{
    Mat6 A;
    Vec6 rhs;
    primalAccumulate(bodyIdx, 0u, 1u, A, rhs);
    primalApply(bodyIdx, A, rhs);
}

// Only the sending half of each reduction needs shared storage. The LDLT
// solver reads the symmetric lower triangle, so transfer 21 matrix values.
groupshared float gsA[32 * 21];
groupshared float gsRhs[32 * 6];
groupshared uint gsUniformLanes;

void primalReduceStep(uint s, bool live, uint lane, uint slot, inout Mat6 A, inout Vec6 rhs)
{
    if (live && lane >= s && lane < 2u * s)
    {
        uint target = slot * s + lane - s;
        gsA[target * 21 + 0] = A.m[0];
        gsA[target * 21 + 1] = A.m[6];
        gsA[target * 21 + 2] = A.m[7];
        gsA[target * 21 + 3] = A.m[12];
        gsA[target * 21 + 4] = A.m[13];
        gsA[target * 21 + 5] = A.m[14];
        gsA[target * 21 + 6] = A.m[18];
        gsA[target * 21 + 7] = A.m[19];
        gsA[target * 21 + 8] = A.m[20];
        gsA[target * 21 + 9] = A.m[21];
        gsA[target * 21 + 10] = A.m[24];
        gsA[target * 21 + 11] = A.m[25];
        gsA[target * 21 + 12] = A.m[26];
        gsA[target * 21 + 13] = A.m[27];
        gsA[target * 21 + 14] = A.m[28];
        gsA[target * 21 + 15] = A.m[30];
        gsA[target * 21 + 16] = A.m[31];
        gsA[target * 21 + 17] = A.m[32];
        gsA[target * 21 + 18] = A.m[33];
        gsA[target * 21 + 19] = A.m[34];
        gsA[target * 21 + 20] = A.m[35];
        gsRhs[target * 6 + 0] = rhs.v[0];
        gsRhs[target * 6 + 1] = rhs.v[1];
        gsRhs[target * 6 + 2] = rhs.v[2];
        gsRhs[target * 6 + 3] = rhs.v[3];
        gsRhs[target * 6 + 4] = rhs.v[4];
        gsRhs[target * 6 + 5] = rhs.v[5];
    }
    GROUP_SYNC();
    if (live && lane < s)
    {
        uint source = slot * s + lane;
        A.m[0] += gsA[source * 21 + 0];
        A.m[6] += gsA[source * 21 + 1];
        A.m[7] += gsA[source * 21 + 2];
        A.m[12] += gsA[source * 21 + 3];
        A.m[13] += gsA[source * 21 + 4];
        A.m[14] += gsA[source * 21 + 5];
        A.m[18] += gsA[source * 21 + 6];
        A.m[19] += gsA[source * 21 + 7];
        A.m[20] += gsA[source * 21 + 8];
        A.m[21] += gsA[source * 21 + 9];
        A.m[24] += gsA[source * 21 + 10];
        A.m[25] += gsA[source * 21 + 11];
        A.m[26] += gsA[source * 21 + 12];
        A.m[27] += gsA[source * 21 + 13];
        A.m[28] += gsA[source * 21 + 14];
        A.m[30] += gsA[source * 21 + 15];
        A.m[31] += gsA[source * 21 + 16];
        A.m[32] += gsA[source * 21 + 17];
        A.m[33] += gsA[source * 21 + 18];
        A.m[34] += gsA[source * 21 + 19];
        A.m[35] += gsA[source * 21 + 20];
        rhs.v[0] += gsRhs[source * 6 + 0];
        rhs.v[1] += gsRhs[source * 6 + 1];
        rhs.v[2] += gsRhs[source * 6 + 2];
        rhs.v[3] += gsRhs[source * 6 + 3];
        rhs.v[4] += gsRhs[source * 6 + 4];
        rhs.v[5] += gsRhs[source * 6 + 5];
    }
    GROUP_SYNC();
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_primal_color(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint color = PASS_U & 255u;
    gStabilize = (PASS_U & 256u) != 0u;
    uint count = SOLVE_COUNT(color);
    uint gid = dispatchIndex(tid);
    uint local = lid.x;
    // Storage loads are non-uniform in WGSL. Publish lanes through workgroup
    // memory; the WGSL post-pass rewrites the read as workgroupUniformLoad.
    if (local == 0u)
        gsUniformLanes = primalLanesForDegree(
            count, HOST_S(29) != 0u ? SOLVE_COUNT(COLOR_DEGREE_BASE + color) : 0u);
    GROUP_SYNC();
    uint lanes = gsUniformLanes;
    uint threads = count * lanes;
    uint bodySlot = gid / lanes;
    uint lane = local % lanes;
    uint slot = local / lanes; // body slot within the workgroup
    bool live = gid < threads && bodySlot < count;

    Mat6 A = mat6Zero();
    Vec6 rhs = vec6Zero();
    uint bodyIdx = 0u;
    uint maxB = HOST_S(1);
    if (live)
    {
        bodyIdx = gColorLists[color * maxB + bodySlot];
        primalAccumulate(bodyIdx, lane, lanes, A, rhs);
    }

    // lanes is uniform (a function of the colour's body count). Barriers stay in
    // uniform control flow: Chrome rejects a barrier inside a loop or after a return.
    if (lanes == 1u)
    {
        if (live) primalApply(bodyIdx, A, rhs);
    }
    else
    {
        // lanes is 2..64. Unrolled halving (three-avbd primalColor).
        if (lanes >= 64u) primalReduceStep(32u, live, lane, slot, A, rhs);
        if (lanes >= 32u) primalReduceStep(16u, live, lane, slot, A, rhs);
        if (lanes >= 16u) primalReduceStep(8u, live, lane, slot, A, rhs);
        if (lanes >= 8u) primalReduceStep(4u, live, lane, slot, A, rhs);
        if (lanes >= 4u) primalReduceStep(2u, live, lane, slot, A, rhs);
        primalReduceStep(1u, live, lane, slot, A, rhs);
        if (live && lane == 0u) primalApply(bodyIdx, A, rhs);
    }
}

// Experimental single-workgroup tail. Every color is completed before the next,
// and every bucket is chunked, so no stale CPU count can truncate the solve.
[shader("compute")]
[numthreads(64, 1, 1)]
void solve_primal_sparse(uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    uint lane = local % 8u;
    uint slot = local / 8u;
    uint maxB = HOST_S(1);
    // Every colour the palette can hold (gColorCounts has 64 colour slots).
    if (local == 0u) gsUniformLanes = min(ALOAD(gCounters[0].colorCount), min(HOST_S(4), COLOR_DEGREE_BASE));
    GROUP_SYNC();
    uint colorLimit = gsUniformLanes;
    gStabilize = (PASS_U & 256u) != 0u;
    for (uint color = PASS_U & 255u; color < colorLimit; color++)
    {
        if (local == 0u) gsUniformLanes = SOLVE_COUNT(color);
        GROUP_SYNC();
        uint count = gsUniformLanes;
        for (uint base = 0u; base < count; base += 8u)
        {
            uint bodySlot = base + slot;
            bool live = bodySlot < count;
            uint bodyIdx = 0u;
            Mat6 A = mat6Zero();
            Vec6 rhs = vec6Zero();
            if (live)
            {
                bodyIdx = gColorLists[color * maxB + bodySlot];
                primalAccumulate(bodyIdx, lane, 8u, A, rhs);
            }
            primalReduceStep(4u, live, lane, slot, A, rhs);
            primalReduceStep(2u, live, lane, slot, A, rhs);
            primalReduceStep(1u, live, lane, slot, A, rhs);
            if (live && lane == 0u) primalApply(bodyIdx, A, rhs);
            // Protect scratch reuse before the next chunk, including partial tails.
            GROUP_SYNC();
        }
        // Graph colors are Gauss-Seidel: publish every pose before the next color.
        if (count > 0u) DeviceMemoryBarrierWithGroupSync();
    }
}

// Small-scene GS: one body per dispatch (gPassUniform = body index). Host walks
// bodies high→low like three-avbd CPU so impacts propagate through stacks.
[shader("compute")]
[numthreads(1, 1, 1)]
void solve_primal_one(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0) return;
    solvePrimalForBody(PASS_U);
}


[shader("compute")]
[numthreads(64, 1, 1)]
void solve_dual(uint3 tid : SV_DispatchThreadID)
{
    // Per-contact dual (Metal occupancy). CSR/primal still use manifolds (Stage 8b).
    uint i = dispatchIndex(tid);
    uint n = ALOAD(gCounters[0].contactCount);
    AvbdParams p = PARAMS;
    if (i < n)
    {
        uint curOff = HOST_S(11);
        #if defined(ENGINE_COMPACT_SOLVE)
        uint pairIndex = loadSolverPoint(i).pairAndStick & SOLVER_DEAD;
        Contact c = (Contact)0;
        SolverPair pair;
        if (pairIndex != SOLVER_DEAD)
        {
            pair = gSolverPairs[pairIndex];
            c = solverContact(i, pair.normal, solverPairFriction(pair));
        }
#else
        Contact c = READ_CONTACT(curOff + i);
#endif
        if (c.alive != 0)
        {
#if defined(ENGINE_COMPACT_SOLVE)
            uint ba = pair.bodyA, bb = pair.bodyB;
#else
            uint ba = gColliders[c.colliderA].body;
            uint bb = gColliders[c.colliderB].body;
#endif
            BodyState bA = gBodies[ba];
            BodyState bB = gBodies[bb];
            // beat-box / three-avbd: dual stasis when both ends are asleep or static.
            bool sleepA = !bodyIsDynamic(bA) || bodyIsSleeping(bA);
            bool sleepB = !bodyIsDynamic(bB) || bodyIsSleeping(bB);
            if (sleepA && sleepB) { /* leave λ/k */ }
            else
            {
#if defined(ENGINE_COMPACT_SOLVE)
            bool anySphere = (pair.countAndSphere & SOLVER_SPHERE) != 0u;
#else
            Collider colA = gColliders[c.colliderA];
            Collider colB = gColliders[c.colliderB];
            bool anySphere = colA.shape == SHAPE_SPHERE || colB.shape == SHAPE_SPHERE;
#endif
            float3 belt = float3(0, 0, 0);
#if defined(ENGINE_COMPACT_SOLVE)
            belt = solverPairBelt(pair);
#else
            belt = pairBeltDisp(colA, colB, bA, bB, c.normal);
#endif
            float4 rotA = (anySphere && bodyIsDynamic(bA)) ? bA.initialRot : bA.orientation;
            float4 rotB = (anySphere && bodyIsDynamic(bB)) ? bB.initialRot : bB.orientation;
            // three-avbd evalContact: Ĉ = c0*(1-α) + J·(x-x⁻), not absolute geometric C.
            bool moveA = bodyIsDynamic(bA) && !bodyIsSleeping(bA);
            bool moveB = bodyIsDynamic(bB) && !bodyIsSleeping(bB);
            float3 dLinA = moveA ? (bA.position - bA.initialPos) : float3(0, 0, 0);
            float3 dLinB = moveB ? (bB.position - bB.initialPos) : float3(0, 0, 0);
            float3 dAngA = float3(0, 0, 0);
            float3 dAngB = float3(0, 0, 0);
            if (moveA)
            {
                float4 qA = bA.orientation;
                if (dot(qA, bA.initialRot) < 0.0) qA = -qA;
                dAngA = quatSubVec(qA, bA.initialRot);
            }
            if (moveB)
            {
                float4 qB = bB.orientation;
                if (dot(qB, bB.initialRot) < 0.0) qB = -qB;
                dAngB = quatSubVec(qB, bB.initialRot);
            }
            float3 rAW = quatRotate(rotA, c.anchorA);
            float3 rBW = quatRotate(rotB, c.anchorB);
            float keep = 1.0 - dualAlpha();
            float3 n = c.normal;
            float3 t1 = c.tangent1;
            float3 t2 = c.tangent2;
            float3 cReg = float3(
                contactC0NReg(c.c0N, keep) + dot(n, dLinB - dLinA) - dot(cross(rAW, n), dAngA) + dot(cross(rBW, n), dAngB),
                c.c0T1 * keep + dot(t1, dLinB - dLinA + belt) - dot(cross(rAW, t1), dAngA) + dot(cross(rBW, t1), dAngB),
                c.c0T2 * keep + dot(t2, dLinB - dLinA + belt) - dot(cross(rAW, t2), dAngA) + dot(cross(rBW, t2), dAngB));
            c.penetration = cReg.x;
            float3 f = float3(c.kN, c.kT, c._pad2) * cReg + float3(c.lambdaN, c.lambdaT1, c.lambdaT2);
            // Eq. 16 tests the *unclamped* tangential force. Testing after cone
            // projection always accepts sliding rows and ramps friction without bound.
            float rawFriction = length(f.yz);
            float bound = c.friction * max(-f.x, 0.0);
            f = projectContactForce(f, c.friction);
            c.lambdaN = f.x;
            c.lambdaT1 = f.y;
            c.lambdaT2 = f.z;
            if (f.x < 0.0) c.kN = min(c.kN + p.betaLin * abs(cReg.x), p.kMaxN);
            if (rawFriction <= bound)
            {
                c.kT = min(c.kT + p.betaLin * abs(cReg.y), max(p.kMaxT, c.kT));
                c._pad2 = min(c._pad2 + p.betaLin * abs(cReg.z), max(p.kMaxT, c._pad2));
                c._pad0 = length(cReg.yz) < 1.0e-5 ? 1.0 : 0.0;
            }
            if ((p.flags & FLAG_COULOMB_STICK) != 0u)
                c._pad0 = (rawFriction <= bound && length(float2(c.c0T1, c.c0T2)) < STICK_TOL) ? 1.0 : 0.0;
            // The dual changes forces/penalties, not geometry. Avoid writing the
            // full 128-byte record for every contact on every solver iteration.
#if defined(ENGINE_COMPACT_SOLVE)
            gSolverPoints[solverPointIndex(i)].lambda = float3(c.lambdaN, c.lambdaT1, c.lambdaT2);
            gSolverPoints[solverPointIndex(i)].penalty = float3(c.kN, c.kT, c._pad2);
            gSolverPoints[solverPointIndex(i)].pairAndStick = solverPointTag(pairIndex, c._pad0 != 0.0);
#endif
#if !defined(ENGINE_PERSISTENT_CONTACTS)
#if defined(ENGINE_COMPACT_SOLVE)
            // The final iteration publishes the canonical warm-start/debug state.
            // Earlier iterations touch only the compact hot records.
            if (PASS_U != 0u)
#endif
            {
            gContacts[curOff + i].penetration = c.penetration;
            gContacts[curOff + i].lambdaN = c.lambdaN;
            gContacts[curOff + i].lambdaT1 = c.lambdaT1;
            gContacts[curOff + i].lambdaT2 = c.lambdaT2;
            gContacts[curOff + i].kN = c.kN;
            gContacts[curOff + i].kT = c.kT;
            gContacts[curOff + i]._pad0 = c._pad0;
            gContacts[curOff + i]._pad2 = c._pad2;
            }
#endif
            }
        }
    }
    uint nJ = gCounters[0].jointCount;
    if (i < nJ)
    {
        uint jflags = gJoints[i].flags;
        // Broken joints are inert, the vehicle kernels own tyre joints, and a
        // joint whose ends both rest (asleep or static) keeps its λ and k.
        bool live = (jflags & (JOINT_FLAG_BROKEN | JOINT_FLAG_TYRE)) == 0u
            && (bodyMoving(gBodies[gJoints[i].bodyA]) || bodyMoving(gBodies[gJoints[i].bodyB]));
        // On the final iteration (PASS_U != 0) λ is this step's converged
        // constraint force: breaking is decided there (bonds never while a host
        // preloads a structure), and a joint whose multipliers still move is
        // marked unsettled so its bodies cannot fall asleep on it.
        bool last = PASS_U != 0u;
        float4 lamLo0 = gJoints[i].lambdaLo;
        float4 lamHi0 = gJoints[i].lambdaHi;
        bool broke = false;
        float force = 0.0, torque = 0.0;
        if (live && jointHeaderIsBall(jflags, gJoints[i].motion))
        {
            JointBall jb = loadJointBall(i);
            jointDualBall(jb, gBodies[jb.bodyA], gBodies[jb.bodyB], p.alpha, p.kStart, p.kMaxN, p.betaLin);
            force = length(jb.lambda);
            broke = last && force > jb.breakForce;
            gJoints[i].lambdaLo.xyz = broke ? float3(0.0) : jb.lambda;
            gJoints[i].penaltyLo.xyz = broke ? float3(0.0) : jb.penalty;
        }
        else if (live)
        {
            Joint j = gJoints[i];
            BodyState a = gBodies[j.bodyA];
            BodyState b = gBodies[j.bodyB];
            float4 frameB0 = j.frameB;
            bool mayBreak = last && (!jointIsBond(j) || (p.flags & FLAG_BOND_PRELOAD) == 0u);
            broke = jointDual(j, a, b, p.alpha, p.dt, p.kStart, p.kMaxN, p.betaLin, p.betaAng, mayBreak, force, torque);
            gJoints[i].lambdaLo = j.lambdaLo; gJoints[i].lambdaHi = j.lambdaHi;
            gJoints[i].penaltyLo = j.penaltyLo; gJoints[i].penaltyHi = j.penaltyHi;
            if (any(j.frameB != frameB0))
            {
                gJoints[i].frameB = j.frameB;
                gJoints[i].c0Lo = j.c0Lo; gJoints[i].c0Hi = j.c0Hi;
            }
        }
        if (live && last)
        {
            uint flags = jflags;
            if (broke)
            {
                // Leaves the hash, CSR and solve from the next step; its bodies
                // collide again. Hosts read the event ring (or scan flags).
                flags |= JOINT_FLAG_BROKEN;
                emitEvent(EVENT_BREAK, i, asuint(force), asuint(torque));
            }
            else
            {
                float4 lo = gJoints[i].lambdaLo, hi = gJoints[i].lambdaHi;
                float3 lin = lo.xyz, ang = float3(lo.w, hi.x, hi.y);
                float3 lin0 = lamLo0.xyz, ang0 = float3(lamLo0.w, lamHi0.x, lamHi0.y);
                // Converged when the last iteration moved the multipliers by
                // < 0.01 % (plus 0.1 N / 0.1 N·m of noise floor).
                bool moving = length(lin - lin0) > 1.0e-4 * length(lin) + 0.1
                    || length(ang - ang0) > 1.0e-4 * length(ang) + 0.1;
                flags = moving ? (flags | JOINT_FLAG_UNSETTLED) : (flags & ~JOINT_FLAG_UNSETTLED);
            }
            if (flags != jflags) gJoints[i].flags = flags;
        }
    }
}


// Post-stabilization (FLAG_POST_STAB): the velocity is the pose after the
// α = 1 iterations, recorded here just before the α = 0 pass removes the
// step-start error, so the primal kernels keep writing pose fields only.
// PASS_U == 1: this step's awake list only, as extract_velocity_sleep.
[shader("compute")]
[numthreads(64, 1, 1)]
void record_stab_velocity(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (PASS_U == 1u)
    {
        if (i >= ALOAD(gCounters[0].awakeCount)) return;
        i = gAwakeList[i];
    }
    else if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b)) return;
    gBodies[i].linearVel = bodyLinearVelocity(b, PARAMS.dt);
    gBodies[i].angularVel = bodyAngularVelocity(b, PARAMS.dt);
}

// gCcdFlags reused pre-CCD: 1 = sleep candidate (slow + timer elapsed) after extract_velocity.
[shader("compute")]
[numthreads(64, 1, 1)]
void extract_velocity_sleep(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    // PASS_U == 1: this step's awake list only. Every other body returns below
    // without effect, and gCcdFlags is read only for awake dynamic bodies
    // (sleep gate, commit), so leaving theirs untouched is equivalent.
    if (PASS_U == 1u)
    {
        if (i >= ALOAD(gCounters[0].awakeCount)) return;
        i = gAwakeList[i];
    }
    else if (i >= gCounters[0].bodyCount) return;
    AvbdParams p = PARAMS;
    BodyState b = gBodies[i];
    gCcdFlags[i] = 0;
    if (!bodyIsDynamic(b)) return;
    if (bodyIsSleeping(b)) return;
    if (wakePending(b))
    {
        // Held still this step; solved from the next with a fresh sleep timer.
        gBodies[i].flags = b.flags & ~BODY_WAKE_PENDING;
        gBodies[i].linearVel = float3(0, 0, 0);
        gBodies[i].angularVel = float3(0, 0, 0);
        gBodies[i].sleepTimer = 0.0;
        return;
    }

    // Step-start pose x⁻ was saved on the body by prediction (initialPos/Rot).
    // Post-stabilization stored the velocity before its pass moved the body.
    float dt = max(p.dt, 1e-8);
    if (!postStabilize())
    {
        b.linearVel = bodyLinearVelocity(b, p.dt);
        b.angularVel = bodyAngularVelocity(b, p.dt);
    }

    if ((p.flags & FLAG_MAX_ROTATION) != 0u)
    {
        float wMax = MAX_ROTATION / dt;
        float w = length(b.angularVel);
        if (w > wMax) b.angularVel *= wMax / w;
    }
    float lin = length(b.linearVel);
    float ang = length(b.angularVel);
    bool slow = lin < p.sleepLin && ang < p.sleepAng;
    float extent = 0.0;
    if (pointSpeedSleep() || displacementSleep() || (p.flags & FLAG_STABILIZATION) != 0u)
    {
        uint n = gCounters[0].colliderCount;
        for (uint c = firstColliderOfBody(i); c < n && gColliders[c].body == i; c++)
            extent = max(extent, colliderExtent(gColliders[c]));
        if (pointSpeedSleep()) slow = lin + ang * extent < p.sleepLin;
        if ((p.flags & FLAG_STABILIZATION) != 0u)
        {
            float band = STAB_SPEEDS * p.sleepLin;
            float speed = lin + ang * extent;
            if (speed < band)
            {
                float keepV = speed / band;
                b.linearVel *= keepV;
                b.angularVel *= keepV;
            }
        }
        if (displacementSleep())
        {
            // A body moves only if both measures of this window say so:
            //  - pose: a low-pass (β = SLEEP_POSE_BETA) pose against the pose
            //    at the window start. It ignores velocity that never moves the
            //    body (one-tick blips that post-stabilization undoes);
            //  - velocity: D = Σ v·h, Θ = Σ ω·h, as mature engines judge sleep
            //    (Box2D, Jolt, PhysX). It ignores push-out, which is not
            //    velocity (a jammed pair slowly pushed apart).
            // Storage: the previous-step record's spare fields (only its
            // linearVel is used elsewhere).
            bool fresh = b.sleepTimer <= 0.0;
            float3 emaPos = fresh ? b.position : BODY_PREV(i).initialPos;
            float4 emaRot = fresh ? b.orientation : BODY_PREV(i).initialRot;
            float4 qNow = b.orientation;
            if (dot(qNow, emaRot) < 0.0) qNow = -qNow;
            emaPos += SLEEP_POSE_BETA * (b.position - emaPos);
            emaRot = quatNormalize(emaRot + SLEEP_POSE_BETA * (qNow - emaRot));
            float3 refPos = fresh ? emaPos : BODY_PREV(i).position;
            float4 refRot = fresh ? emaRot : BODY_PREV(i).orientation;
            float turn = 2.0 * acos(min(abs(dot(refRot, emaRot)), 1.0));
            float movedPose = length(emaPos - refPos) + turn * extent;
            float3 dLin = (fresh ? float3(0, 0, 0) : BODY_PREV(i).angularVel) + b.linearVel * dt;
            float3 dAng = (fresh ? float3(0, 0, 0) : BODY_PREV(i).invInertia) + b.angularVel * dt;
            float movedVel = length(dLin) + length(dAng) * extent;
            float moved = min(movedPose, movedVel);
            slow = fresh || moved < p.sleepLin * p.sleepTime;
            // Diagnostics (sleep-diag.mjs): this window's displacement. A vehicle
            // chassis keeps its wheel-block index in initialPad.
            if ((b.flags & BODY_VEHICLE) == 0u) gBodies[i].initialPad = moved;
            // A candidate held awake (by its island) restarts the window every
            // sleepTime, so the limit is drift per window, like a speed.
            bool rewindow = slow && b.sleepTimer >= 2.0 * p.sleepTime;
            if (rewindow) b.sleepTimer = p.sleepTime;
            if (!slow || rewindow)
            {
                refPos = emaPos;
                refRot = emaRot;
                dLin = float3(0, 0, 0);
                dAng = float3(0, 0, 0);
            }
            BODY_PREV(i).initialPos = emaPos;
            BODY_PREV(i).initialRot = emaRot;
            BODY_PREV(i).position = refPos;
            BODY_PREV(i).orientation = refRot;
            BODY_PREV(i).angularVel = dLin;
            BODY_PREV(i).invInertia = dAng;
        }
    }
    if (slow)
    {
        b.sleepTimer += dt;
        if (b.sleepTimer >= p.sleepTime)
        {
            gCcdFlags[i] = 1; // candidate — sleep_island_commit after island gate
            AOR2(gScratchAtomics[13], 1);
        }
    }
    else
    {
        b.sleepTimer = 0.0;
        b.flags |= BODY_AWAKE;
    }
    gBodies[i].linearVel = b.linearVel;
    gBodies[i].angularVel = b.angularVel;
    gBodies[i].sleepTimer = b.sleepTimer;
    gBodies[i].flags = b.flags;
}

// If any dynamic contact neighbor is not a sleep candidate, revoke both. Repeating
// this (host) propagates through the island so a stack sleeps together or not at all.
// At this point the solver no longer needs manifold/awake dispatch slots. Reuse
// them for sleep gate/commit, avoiding millions of no-op threads on active scenes.
[shader("compute")]
[numthreads(64, 1, 1)]
void write_sleep_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    bool anyCandidate = ALOAD(gScratchAtomics[13]) != 0u;
    // Manifolds, then joints (joints and bonds join islands), one thread each.
    writeDispatchArgs(66u, anyCandidate ? min(ALOAD(gCounters[0].manifoldCount), HOST_S(10)) + gCounters[0].jointCount : 0u);
    writeDispatchArgs(67u, anyCandidate ? gCounters[0].bodyCount : 0u);
}

static const uint SLEEP_REVOKED = 0x80000000u;
bool sleepGateExact() { return (HOST_S(25) & 8u) != 0u; }
// Candidate at the start of `round`: never revoked, or revoked in this round.
bool gateCandidate(uint flag, uint round) { return flag == 1u || flag == (SLEEP_REVOKED | round); }

[shader("compute")]
[numthreads(64, 1, 1)]
void sleep_island_gate(uint3 tid : SV_DispatchThreadID)
{
    // Opt-in deterministic gate (HOST_S(25) bit 3, JS host, PASS_U = round
    // 1..n): a candidate revoked in round r is marked SLEEP_REVOKED | r and
    // still counts as a candidate for the rest of round r, so a revocation
    // travels exactly one hop per round. In place (default), a revocation
    // written earlier in the same dispatch may already be read, so how far it
    // travels in n rounds depends on thread scheduling.
    bool roundExact = sleepGateExact() && PASS_U != 0u;
    uint round = PASS_U;
    // One thread per manifold (same connectivity as contacts, ~2–4× fewer).
    // Ring smash: no sleep candidates → scratch[13]==0 → entire gate is a no-op.
    if (ALOAD(gScratchAtomics[13]) == 0u) return;
    uint i = dispatchIndex(tid);
    uint n = min(ALOAD(gCounters[0].manifoldCount), HOST_S(10));
    uint ba, bb;
    if (i < n)
    {
        Manifold m = gManifolds[i];
        if (m.contactCount == 0u) return;
        ba = gColliders[m.colliderA].body;
        bb = gColliders[m.colliderB].body;
    }
    bool driven = false;
    if (i >= n)
    {
        // Intact joints and bonds tie their bodies into one island, like contacts.
        if (i - n >= gCounters[0].jointCount) return;
        Joint j = gJoints[i - n];
        if (jointBroken(j)) return;
        ba = j.bodyA;
        bb = j.bodyB;
        // A running drive keeps its bodies awake, even against a static hub, and
        // so do multipliers that are still converging (a stiff chain can be
        // nearly still long before its forces settle). Tyre joints (chassis on a
        // dynamic ground body) just join the island.
        driven = ((j.flags & JOINT_FLAG_DRIVE) != 0u && (j.driveVelocity != 0.0 || j.driveStiffness > 0.0))
            || (j.flags & JOINT_FLAG_UNSETTLED) != 0u;
    }
    BodyState a = gBodies[ba];
    BodyState b = gBodies[bb];
    bool dynA = bodyIsDynamic(a);
    bool dynB = bodyIsDynamic(b);
    if (driven)
    {
        if (dynA && !bodyIsSleeping(a)) gCcdFlags[ba] = 0;
        if (dynB && !bodyIsSleeping(b)) gCcdFlags[bb] = 0;
        return;
    }
    if (!dynA || !dynB) return;
    // Sleeping neighbor is fine (already out of the island). Awake non-candidate blocks.
    uint fa = gCcdFlags[ba];
    uint fb = gCcdFlags[bb];
    bool okA = bodyIsSleeping(a) || (roundExact ? gateCandidate(fa, round) : fa != 0);
    bool okB = bodyIsSleeping(b) || (roundExact ? gateCandidate(fb, round) : fb != 0);
    if (!okA || !okB)
    {
        if (roundExact)
        {
            if (dynA && !bodyIsSleeping(a) && fa == 1u) gCcdFlags[ba] = SLEEP_REVOKED | round;
            if (dynB && !bodyIsSleeping(b) && fb == 1u) gCcdFlags[bb] = SLEEP_REVOKED | round;
        }
        else
        {
            if (dynA && !bodyIsSleeping(a)) gCcdFlags[ba] = 0;
            if (dynB && !bodyIsSleeping(b)) gCcdFlags[bb] = 0;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sleep_island_commit(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    // PASS_U == 1: this step's awake list (others return below anyway).
    if (PASS_U == 1u)
    {
        if (i >= ALOAD(gCounters[0].awakeCount)) return;
        i = gAwakeList[i];
    }
    else if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    // Exact gate: only unrevoked candidates (1) sleep; revoked ones hold a mark.
    if (sleepGateExact() ? gCcdFlags[i] != 1u : gCcdFlags[i] == 0) return;
    b.flags = (b.flags & ~BODY_AWAKE) | BODY_SLEEPING;
    b.linearVel = float3(0, 0, 0);
    b.angularVel = float3(0, 0, 0);
    AADD2(gCounters[0].sleepEvents, 1);
    gBodies[i] = b;
}

// --- CCD stretch: classify fast bodies (linear + angular rim) ---
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_classify(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    gCcdFlags[i] = 0;
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    AvbdParams p = PARAMS;
    float speed = length(b.linearVel);
    float aspeed = length(b.angularVel);
    float radius = 0.5; // conservative; refined per-collider in full CCD
    float rim = aspeed * radius;
    float thresh = radius * 0.5 / max(p.dt, 1e-8);
    if (speed + rim > thresh)
        gCcdFlags[i] = 1;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_sweep(uint3 tid : SV_DispatchThreadID)
{
    // Stretch goal: swept sphere-sphere / sphere-plane TOI for flagged bodies.
    // Exhaustion and misses set CCD_* flags — never report certified no-hit on miss.
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    Collider col = gColliders[i];
    if (gCcdFlags[col.body] == 0) return;
    BodyState b = gBodies[col.body];
    AvbdParams p = PARAMS;
    float3 v = b.linearVel;
    float maxIter = 8.0;
    // Against planes only in this stretch slice
    for (uint j = 0; j < gCounters[0].colliderCount; j++)
    {
        if (j == i) continue;
        Collider other = gColliders[j];
        if (other.shape != SHAPE_PLANE) continue;
        float3 n = normalize(other.params.xyz);
        float r = col.shape == SHAPE_SPHERE ? col.params.x : col.params.w;
        float3 x0 = transformPoint(b.position, b.orientation, col.localPos);
        float denom = dot(v, n);
        if (denom >= -1e-8) continue;
        float dist = dot(x0, n) - other.params.w - r;
        float toi = dist / -denom;
        if (toi < 0.0 || toi > p.dt) continue;
        // Advance and mark — budget check
        if (maxIter <= 0.0)
        {
            AOR2(gCounters[0].overflowFlags, CCD_BUDGET_EXHAUSTED);
            return;
        }
        maxIter -= 1.0;
        b.position += v * toi;
        b.linearVel -= n * max(0.0, -dot(b.linearVel, n));
        gBodies[col.body] = b;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void recompute_inertia(uint3 tid : SV_DispatchThreadID)
{
    // Parallel-axis compound inertia from attached colliders (deterministic by collider index).
    uint body = dispatchIndex(tid);
    if (body >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[body];
    // Vehicle chassis mass comes from the packer; their belt colliders reuse density.
    if (!bodyIsDynamic(b) || (b.flags & BODY_VEHICLE) != 0u) return;
    float mass = 0.0;
    float3 com = float3(0, 0, 0);
    // Pass 1: mass / COM in body frame
    for (uint i = 0; i < gCounters[0].colliderCount; i++)
    {
        Collider col = gColliders[i];
        if (col.body != body) continue;
        float m = col.density;
        if (col.shape == SHAPE_SPHERE)
        {
            float r = col.params.x;
            m *= 4.0 / 3.0 * 3.14159265 * r * r * r;
        }
        else if (col.shape == SHAPE_BOX)
        {
            float3 h = col.params.xyz;
            m *= 8.0 * h.x * h.y * h.z;
        }
        else if (col.shape == SHAPE_CAPSULE)
        {
            float r = col.params.x;
            float hh = col.params.y;
            m *= 3.14159265 * r * r * (2.0 * hh) + 4.0 / 3.0 * 3.14159265 * r * r * r;
        }
        else
        {
            m *= max(col.params.w, 0.1);
        }
        mass += m;
        com += col.localPos * m;
    }
    if (mass <= 1e-8) return;
    com /= mass;
    float3 I = float3(0, 0, 0);
    for (uint i = 0; i < gCounters[0].colliderCount; i++)
    {
        Collider col = gColliders[i];
        if (col.body != body) continue;
        float m = col.density;
        float3 localI = float3(0.1, 0.1, 0.1);
        if (col.shape == SHAPE_SPHERE)
        {
            float r = col.params.x;
            m *= 4.0 / 3.0 * 3.14159265 * r * r * r;
            float ii = 0.4 * m * r * r;
            localI = float3(ii, ii, ii);
        }
        else if (col.shape == SHAPE_BOX)
        {
            float3 h = col.params.xyz;
            m *= 8.0 * h.x * h.y * h.z;
            localI = float3(
                (1.0 / 12.0) * m * (4.0 * h.y * h.y + 4.0 * h.z * h.z),
                (1.0 / 12.0) * m * (4.0 * h.x * h.x + 4.0 * h.z * h.z),
                (1.0 / 12.0) * m * (4.0 * h.x * h.x + 4.0 * h.y * h.y));
        }
        float3 d = col.localPos - com;
        I += localI + m * float3(d.y * d.y + d.z * d.z, d.x * d.x + d.z * d.z, d.x * d.x + d.y * d.y);
    }
    b.invMass = 1.0 / mass;
    b.invInertia = float3(
        I.x > 1e-8 ? 1.0 / I.x : 0.0,
        I.y > 1e-8 ? 1.0 / I.y : 0.0,
        I.z > 1e-8 ? 1.0 / I.z : 0.0);
    // Shift body origin note: we keep authoring frame; COM offset absorbed into colliders later if needed.
    gBodies[body] = b;
}
