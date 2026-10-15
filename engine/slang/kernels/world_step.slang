// Production phase entry points — one Slang source for Metal / SPIR-V / CUDA.
#include "../physics/common.slang"
#include "../physics/atomics.slang"
#include "../physics/box_collide.slang"

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

    if (!startAtRest)
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

[shader("compute")]
[numthreads(64, 1, 1)]
void build_aabbs(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    // Keep a sleeping body's previous AABB/OBB, but build them on the first step:
    // scenes may start asleep (pre-fractured rubble) and still need bounds.
    if (bodyIsSleeping(b) && bodyIsDynamic(b) && HOST_S(17) > 0u)
    {
        return;
    }
    float3 worldPos = transformPoint(b.position, b.orientation, col.localPos);
    float4 worldRot = quatMul(b.orientation, col.localRot);
    float3 mn, mx;
    // three-avbd testPair pads world AABBs by 1e-4 only (f32 round-off). Inflating by
    // contactOffset (~1 cm) pulled in ring-adjacent bricks that faceSeparated should
    // still drop — but the fatter boxes also bloated grid occupancy. Margin stays in C0.
    const float aabbPad = 1.0e-4;
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
        HullRef h = hullOf(col);
        mn = float3(1e30, 1e30, 1e30);
        mx = float3(-1e30, -1e30, -1e30);
        for (uint v = 0u; v < h.nv; v++)
        {
            float3 p = transformPoint(worldPos, worldRot, hullVertex(h, v));
            mn = min(mn, p);
            mx = max(mx, p);
        }
        mn -= aabbPad;
        mx += aabbPad;
    }
    Aabb box;
    box.mn = mn; box.mx = mx;
    float boundR = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    box._pad0 = boundR;
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
        w.center = float4(transformPoint(worldPos, worldRot, hullLocalCenter(h)), 0.0);
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
        w.center = float4(obb.center, 0.0);
        w.half = float4(obb.half, 0.0);
        w.axis0 = float4(obb.axis0, 0.0);
        w.axis1 = float4(obb.axis1, 0.0);
        w.axis2 = float4(obb.axis2, 0.0);
        gWorldObbs[i] = w;
    }
}

static const uint GRID_HASH_SIZE = 131072u;
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

uint gridHash(int x, int y, int z)
{
    uint h = (uint)x * 73856093u ^ (uint)y * 19349663u ^ (uint)z * 83492791u;
    return h & (gridHashSize() - 1u);
}

// Open-addressed joint-pair set in gEvents (three-avbd / demo3d constrainedTo).
// Replaces O(joints) scan that dominated Chain Mail / Jointed Drop broadphase.
static const uint JOINT_HASH_SIZE = 65536u;
static const uint JOINT_HASH_EMPTY = 0xFFFFFFFFu;

uint jointPairKey(uint bodyA, uint bodyB)
{
    uint lo = min(bodyA, bodyB);
    uint hi = max(bodyA, bodyB);
    uint h = (lo * 73856093u) ^ (hi * 19349663u) ^ (lo << 16) ^ (hi >> 3);
    // Must never equal EMPTY sentinel.
    return h == JOINT_HASH_EMPTY ? (lo ^ 0xA5A5A5A5u) : h;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void joint_hash_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= JOINT_HASH_SIZE) return;
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
    uint key = jointPairKey(jj.bodyA, jj.bodyB);
    uint h = key & (JOINT_HASH_SIZE - 1u);
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = (h + probe) & (JOINT_HASH_SIZE - 1u);
        uint prev;
        // Spin on the same slot when WGSL weak CAS spuriously fails. Advancing
        // the probe would leave an EMPTY hole and break open-address lookups.
        for (uint spin = 0u; spin < 64u; spin++)
        {
            ACAS4(gEvents[slot], JOINT_HASH_EMPTY, key, prev);
            if (prev != JOINT_HASH_EMPTY) break;
            if (ALOAD(gEvents[slot]) == key) return;
        }
        if (prev == JOINT_HASH_EMPTY || prev == key) return;
    }
}

bool bodiesJointed(uint bodyA, uint bodyB)
{
    // Ring/Wall have zero joints — skip the open-address probe entirely (and do not
    // require a 64k clear of gEvents each step).
    if (gCounters[0].jointCount == 0u) return false;
    uint key = jointPairKey(bodyA, bodyB);
    uint h = key & (JOINT_HASH_SIZE - 1u);
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = (h + probe) & (JOINT_HASH_SIZE - 1u);
        uint v = ALOAD(gEvents[slot]);
        if (v == JOINT_HASH_EMPTY) return false;
        if (v == key) return true;
    }
    return false;
}

// Collision filtering (per collider; HOST_S(23) != 0 when any filter is set).
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

void emitCandidate(uint i, uint j)
{
    if (i == j) return;
    if (HOST_S(23) != 0u && collidersFilteredOut(i, j)) return;
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
// Oversized statics: count at 53, up to 10 collider IDs at 54..63. Kept clear of
// the legacy large list (25..41), which overlaps Metal's HOST_S(26).
static const uint HIER_LARGE_COUNT_SLOT = 53u;
static const uint HIER_LARGE_MAX = 10u;

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
    uint prev;
    AXCHG3(gConstraintIds[hierHash(hierCell(box, level), level)], i, prev);
    gColorLists[i] = prev;
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
    const float faceGap = 0.001;
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
    uint occupied = ALOAD(gScratchAtomics[HIER_LEVEL_MASK_SLOT]);
    for (uint level = own; level < levels; level++)
    {
        if (!active && level == own) continue;
        if (((occupied >> level) & 1u) == 0u) continue;
        int3 center = hierCell(aa, level);
        for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++)
        {
            int3 cell = center + int3(x, y, z);
            uint j = ALOAD(gConstraintIds[hierHash(cell, level)]);
            uint guard = 0u;
            while (j != 0xFFFFFFFFu && guard < count)
            {
                guard++;
                uint next = gColorLists[j];
                Aabb ab = gAabbs[base + j];
                // Hash buckets alias: accept only colliders whose true cell/level
                // is this one, so no pair is visited twice.
                if (j != i && hierLevel(ab._pad0) == level &&
                    all(hierCell(ab, level) == cell) && aabbOverlap(aa, ab))
                {
                    Collider b = gColliders[j];
                    if (b.body != a.body)
                    {
                        bool other = bodyMoving(gBodies[b.body]);
                        bool owner = level > own || (active && (!other || i < j));
                        if ((active || other) && owner &&
                            !bodiesJointed(a.body, b.body) &&
                            !(probeHasObb && boxesSeparatedOnFaceAxis(probe, j, faceGap)))
                            emitCandidate(i, j);
                    }
                }
                j = next;
            }
            if (j != 0xFFFFFFFFu) AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
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
    if (!bodyIsDynamic(bIns) &&
        (ext.x >= cellSize * 4.0 || ext.y >= cellSize * 4.0 || ext.z >= cellSize * 4.0))
        return;
    float radius = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    // Bigger than the cell (smash ball vs bricks): test from every dynamic, not one hash slot.
    if (bodyIsDynamic(bIns) && radius > cellSize)
    {
        uint slot;
        AADD3(gScratchAtomics[25], 1, slot);
        if (slot < 16u) ASTORE(gScratchAtomics[26u + slot], i);
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
    gColorLists[i] = prev;
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
        link = gColorLists[cur];
        guard++;
        if (guard == 4096u)
            AOR2(gCounters[0].overflowFlags, OVERFLOW_BP_CHAIN);
        if (cur == i || (ownCell && awakeEntry && cur <= i)) continue;
        Aabb ab = gAabbs[base + cur];
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
    const float faceGap = 0.001;

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
                    if (sleeper && !otherStatic) keep = false;
                    if (bodyIsSleeping(ob) && sleeper) keep = false;
                    if (!otherStatic && j < i) keep = false;
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
    uint nLarge = min(ALOAD(gScratchAtomics[25]), 16u);
    for (uint k = 0u; k < nLarge; k++)
    {
        uint j = ALOAD(gScratchAtomics[26u + k]);
        if (j == i) continue;
        Collider other = gColliders[j];
        if (other.body == col.body || bodiesJointed(col.body, other.body)) continue;
        BodyState ob = gBodies[other.body];
        // Small→large owns this pair regardless of indices. Only large→large
        // needs index ordering; a large centre-cell query cannot cover its surface.
        if (queryLarge && bodyIsDynamic(ob) && j < i) continue;
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
    // Oversized statics are not in the grid/list. The scene builder puts ground
    // at an endpoint. Large dynamics were already handled exactly once above.
    for (uint t = 0u; t < 2u; t++)
    {
        if (gCounters[0].colliderCount == 0u) break;
        uint j = (t == 0u) ? 0u : (gCounters[0].colliderCount - 1u);
        if (j == i || (t == 1u && j == 0u)) continue;
        Collider other = gColliders[j];
        if (other.shape == SHAPE_PLANE || other.body == col.body) continue;
        Aabb ab = gAabbs[base + j];
        float3 ext = ab.mx - ab.mn;
        if (ext.x < cellSize * 4.0 && ext.y < cellSize * 4.0 && ext.z < cellSize * 4.0)
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
            float rr = other.params.x + (col.params.w > 0.0 ? col.params.w : length(col.params.xyz));
            if (dot(d, d) > rr * rr) continue;
        }
        emitCandidate(i, j);
    }
}

// Back-compat name used by CUDA/WebGPU build lists.
[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_grid(uint3 tid : SV_DispatchThreadID)
{
    broadphase_query(tid);
}

// Emit one contact. C0 matches three-avbd: n·(xB−xA)+margin with n = A→B (compression ⇒ C0 < 0).
Contact makeContactRow(
    CandidatePair pair, BodyState ba, BodyState bb, Collider ca, Collider cb,
    float3 nrm, float3 pA, float3 pB, uint featureId, float margin, float mu)
{
    float3 t1, t2;
    orthonormalBasis(nrm, t1, t2);
    // three-avbd: C0_n = n·(xB−xA) + margin (A→B). Equilibrium sits ~margin deep.
    float3 d0 = pB - pA;
    float C0n = dot(nrm, d0) + margin;
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

void wakeFromContact(BodyState ba, BodyState bb, Collider ca, Collider cb)
{
    // Wake only a sleeping body hit by an awake neighbour with meaningful relative
    // speed (island wake). Resting pile contacts used to thrash sleep every NP frame
    // (~1.5k sleeping / 31k awake despite KE~50). Never clear SLEEPING on calm pairs.
    bool awakeA = bodyIsDynamic(ba) && !bodyIsSleeping(ba);
    bool awakeB = bodyIsDynamic(bb) && !bodyIsSleeping(bb);
    if (!awakeA && !awakeB) return;
    float3 vRel = ba.linearVel - bb.linearVel;
    float wRel = length(ba.angularVel) + length(bb.angularVel);
    float thresh = PARAMS.sleepLin;
    if (dot(vRel, vRel) < thresh * thresh && wRel < PARAMS.sleepAng)
        return;
    // Store only the wake fields: a whole-body store from this thread's
    // narrowphase-start copy could overwrite fields other work updated.
    if (bodyIsDynamic(ba) && bodyIsSleeping(ba) && awakeB)
    {
        gBodies[ca.body].flags = (ba.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        gBodies[ca.body].sleepTimer = 0.0;
    }
    if (bodyIsDynamic(bb) && bodyIsSleeping(bb) && awakeA)
    {
        gBodies[cb.body].flags = (bb.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        gBodies[cb.body].sleepTimer = 0.0;
    }
}

// three-avbd Stage 8b: one manifold record + contiguous contact points.
void emitContactManifold(
    CandidatePair pair, BodyState ba, BodyState bb, Collider ca, Collider cb,
    BoxManifold points, float margin)
{
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
static const float NEAREST_FRACTION = 0.05;

uint pairWarmKey(uint colA, uint colB)
{
    uint h = colA * 73856093u ^ colB * 19349663u;
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

static const uint FLAG_REUSE_CONTACTS = 1u;
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
    if (i >= gCounters[0].bodyCount) return;
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
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = (h + probe) & (hashN - 1u);
        uint j = ALOAD(gConstraintIds[slot]);
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
    float margin = PARAMS.contactOffset;
    float3 nrm; float pen; float3 pA; float3 pB;
    bool hit = false;

    // Multi-point manifolds (demo3d / three-avbd) — required for stable stacks.
    if (ca.shape == SHAPE_BOX && cb.shape == SHAPE_BOX)
    {
        // World OBBs from build_aabbs cache (skip per-pair quat axis rebuild).
        BoxManifold mf = collideBoxesCached(
            worldObbFromCache(pair.colliderA), worldObbFromCache(pair.colliderB), margin);
        if (mf.count > 0)
            emitManifold(pair, ba, bb, ca, cb, mf, margin);
        return;
    }
    if (ca.shape == SHAPE_BOX && cb.shape == SHAPE_PLANE)
    {
        emitManifold(pair, ba, bb, ca, cb,
            collideBoxPlane(posA, rotA, ca.params.xyz, cb.params.xyz, cb.params.w, margin), margin);
        return;
    }
    if (ca.shape == SHAPE_PLANE && cb.shape == SHAPE_BOX)
    {
        // Swap to box-vs-plane then flip normal / points for A=plane, B=box.
        BoxManifold mf = collideBoxPlane(posB, rotB, cb.params.xyz, ca.params.xyz, ca.params.w, margin);
        if (mf.count > 0)
        {
            mf.normal = -mf.normal;
            for (int k = 0; k < mf.count; k++)
            {
                float3 tmp = mf.pts[k].pA;
                mf.pts[k].pA = mf.pts[k].pB;
                mf.pts[k].pB = tmp;
            }
            emitManifold(pair, ba, bb, ca, cb, mf, margin);
        }
        return;
    }

    if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_SPHERE)
    {
        hit = sphereSphere(posA, ca.params.x, posB, cb.params.x, margin, nrm, pen, pA, pB);
    }
    else if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_PLANE)
    {
        // Geometric points: sphere surface toward plane, plane projection.
        float3 pn = normalize(cb.params.xyz);
        float gap = dot(posA, pn) - cb.params.w - ca.params.x;
        if (gap < margin)
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
        if (gap < margin)
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
    else if (ca.shape == SHAPE_CAPSULE && cb.shape == SHAPE_CAPSULE)
    {
        float rA = ca.params.x;
        float rB = cb.params.x;
        float3 axA = quatRotate(rotA, float3(0, 1, 0));
        float3 axB = quatRotate(rotB, float3(0, 1, 0));
        float3 a0 = posA - axA * ca.params.y;
        float3 a1 = posA + axA * ca.params.y;
        float3 b0 = posB - axB * cb.params.y;
        float3 b1 = posB + axB * cb.params.y;
        float3 midA = 0.5 * (a0 + a1);
        float3 midB = 0.5 * (b0 + b1);
        float sep = length(midB - midA) - ca.params.y - cb.params.y;
        if (sep < rA + rB + margin)
            hit = sphereSphere(midA, rA + ca.params.y * 0.25, midB, rB + cb.params.y * 0.25, margin, nrm, pen, pA, pB);
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
        hit = sphereSphere(closest, ca.params.x, posB, cb.params.x, margin, nrm, pen, pA, pB);
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
        hit = sphereSphere(posA, ca.params.x, closest, cb.params.x, margin, nrm, pen, pA, pB);
    }
    else
    {
        float rA = ca.params.w > 0.0 ? ca.params.w : length(ca.params.xyz);
        float rB = cb.params.w > 0.0 ? cb.params.w : length(cb.params.xyz);
        hit = sphereSphere(posA, rA, posB, rB, margin, nrm, pen, pA, pB);
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

// demo3d Joint::initialize — store C0, decay λ/k (Eq. 19).
[shader("compute")]
[numthreads(64, 1, 1)]
void warmstart_joints(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].jointCount;
    if (i >= n) return;
    AvbdParams p = PARAMS;
    Joint j = gJoints[i];
    BodyState a = gBodies[j.bodyA];
    BodyState b = gBodies[j.bodyB];
    float3 wa = transformPoint(a.position, a.orientation, j.localAnchorA);
    float3 wb = transformPoint(b.position, b.orientation, j.localAnchorB);
    j.c0 = wa - wb;
    j.lambda0 *= p.alpha * p.gamma;
    j.lambda1 *= p.alpha * p.gamma;
    j.lambda2 *= p.alpha * p.gamma;
    j.lambda3 *= p.alpha * p.gamma;
    j.lambda4 *= p.alpha * p.gamma;
    j.lambda5 *= p.alpha * p.gamma;
    float kMin = p.kStart;
    float kMax = max(j.stiffness, kMin);
    j.k0 = clamp(j.k0 * p.gamma, kMin, kMax);
    j.k1 = clamp(j.k1 * p.gamma, kMin, kMax);
    j.k2 = clamp(j.k2 * p.gamma, kMin, kMax);
    j.k3 = clamp(j.k3 * p.gamma, kMin, kMax);
    j.k4 = clamp(j.k4 * p.gamma, kMin, kMax);
    j.k5 = clamp(j.k5 * p.gamma, kMin, kMax);
    gJoints[i] = j;
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
    ASTORE(gConstraintIds[i], CONTACT_HASH_EMPTY);
}

// Publish history index j (a live block head) for pair (colA, colB).
void insertWarmHead(uint j, uint colA, uint colB, uint prevOff)
{
    uint key = pairWarmKey(colA, colB);
    uint n = HOST_S(9);
    uint h = key & (n - 1u); // n is power of two
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = (h + probe) & (n - 1u);
        uint was;
        // Spin on the same slot when WGSL weak CAS spuriously fails. Advancing
        // the probe would leave an EMPTY hole and break open-address lookups.
        for (uint spin = 0u; spin < 64u; spin++)
        {
            ACAS4(gConstraintIds[slot], CONTACT_HASH_EMPTY, j, was);
            if (was != CONTACT_HASH_EMPTY) break;
            if (ALOAD(gConstraintIds[slot]) == j) return;
        }
        if (was == CONTACT_HASH_EMPTY) return;
        Contact occupant = READ_CONTACT(prevOff + was);
        if (occupant.alive != 0 &&
            occupant.colliderA == colA &&
            occupant.colliderB == colB)
        {
            // Same pair: keep the minimum index (start of the contiguous block).
            uint cur = was;
            while (j < cur)
            {
                uint got;
                for (uint spin = 0u; spin < 64u; spin++)
                {
                    ACAS4(gConstraintIds[slot], cur, j, got);
                    if (got != cur) break;
                    if (ALOAD(gConstraintIds[slot]) == j) return;
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
    c.c0N = dot(c.normal, d) + p.contactOffset;
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
    c.kT = max(1.0, p.gamma * c.kT);
    c._pad2 = max(1.0, p.gamma * c._pad2);
    c.lambdaN *= p.alpha * p.gamma;
    c.lambdaT1 *= p.alpha * p.gamma;
    c.lambdaT2 *= p.alpha * p.gamma;
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
            uint j = ALOAD(gConstraintIds[slot]);
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
                if (prev._pad0 > 0.5 && !anySphere)
                {
                    c.anchorA = prev.anchorA;
                    c.anchorB = prev.anchorB;
                    float3 wa = transformPoint(ba.position, ba.orientation, c.anchorA);
                    float3 wb = transformPoint(bb.position, bb.orientation, c.anchorB);
                    float3 d = wb - wa;
                    c.c0N = dot(c.normal, d) + p.contactOffset;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void persist_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= ALOAD(gCounters[0].contactCount)) return;
    uint curOff = HOST_S(11);
#if defined(ENGINE_REUSE_COPY)
    if ((gSolverPoints[curOff + i].pairAndStick & PERSIST_FINAL) != 0u) return;
#endif
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
    if ((sp.countAndSphere & ~SOLVER_SPHERE) == 0u) return;
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
    if ((sp.countAndSphere & ~SOLVER_SPHERE) == 0u) return;
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
    uint prio = jpPriority(body);

    if (ALOAD(gBodyColor[body]) != 0xFFFFFFFFu) return;

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
    if (blocked) return;

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
}

// Adaptive 1/2/4/8 threads per body by colour size. This tuning differs from
// pinned three-avbd, whose one-lane threshold is 2^31 rather than 65536.
uint primalLanesFor(uint bodies)
{
    // Never one lane per body: gables' ~184k-body colors measured 1/2/4/8 lanes
    // at 32.2/31.1/30.9/31.4 ms (decision 8eg); the ring is neutral.
    if (bodies == 0u) return 1u;
    if (bodies >= 4096u) return 4u;
    return 8u;
}

// Degree-aware lanes (opt-in, HOST_S(22) != 0; host sizes gColorCounts to 128
// words): gColorCounts[64 + c] holds colour c's largest CSR degree. A compound
// body with hundreds of manifolds otherwise holds its whole colour dispatch
// while 8 lanes walk its edges serially. Widen to 16/32/64 lanes when the
// colour is small enough that the extra threads stay under 64k.
static const uint COLOR_DEGREE_BASE = 64u;
uint primalLanesForDegree(uint bodies, uint maxDegree)
{
    uint lanes = primalLanesFor(bodies);
    if (HOST_S(22) == 0u || bodies == 0u) return lanes;
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
    if (HOST_S(26) != 0u) writeDispatchArgs(69u, ALOAD(gCounters[0].recolorCount));
    // Slot 73: previous contacts, for whole-step manifold reuse (any hash mode).
    writeDispatchArgs(73u, min(HOST_S(7), HOST_S(3)));
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
        n, HOST_S(22) != 0u ? ALOAD(gColorCounts[COLOR_DEGREE_BASE + c]) : 0u);
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
        if (bodyIsDynamic(b) && !bodyIsSleeping(b))
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
        uint count = gColorOffsets[index];
        gColorOffsets[index] = sum;
        sum += count;
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
        uint count = gColorOffsets[index];
        gColorOffsets[index] = prefix;
        prefix += count;
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
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
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
    gSolverPairs[mi] = pair;
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
        mat6AddDiag(A, float3(mass, mass, mass) / dt2, inertia / dt2);
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
        rhs.v[3] = (inertia.x / dt2) * dangTarget.x;
        rhs.v[4] = (inertia.y / dt2) * dangTarget.y;
        rhs.v[5] = (inertia.z / dt2) * dangTarget.z;
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
            uint contactCount = pair.countAndSphere & ~SOLVER_SPHERE;
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
            float keep = 1.0 - p.alpha;
            for (uint ci = 0u; ci < contactCount; ci++)
            {
                #if defined(ENGINE_COMPACT_SOLVE)
                Contact c = solverContact(contactStart + ci - HOST_S(11), pair.normal, pair.friction);
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
                    c.c0N * keep + dot(n, dLinB - dLinA) - dot(cross(rAW, n), dAngA) + dot(cross(rBW, n), dAngB),
                    c.c0T1 * keep + dot(t1, dLinB - dLinA) - dot(cross(rAW, t1), dAngA) + dot(cross(rBW, t1), dAngB),
                    c.c0T2 * keep + dot(t2, dLinB - dLinA) - dot(cross(rAW, t2), dAngA) + dot(cross(rBW, t2), dAngB));
                c.penetration = C.x;
                // C is already Ĉ; alpha=0 + margin=0 ⇒ contributeContact uses F = k·C + λ.
                contributeContact(A, rhs, c, isA, 0.0, b.position, rotSelf, C, 0.0,
                    isA ? rAW : rBW);
            }
        }
        else
        {
            Joint j = gJoints[id];
            bool isA = j.bodyA == bodyIdx;
            BodyState ba = gBodies[j.bodyA];
            BodyState bb = gBodies[j.bodyB];
            float3 wa = transformPoint(ba.position, ba.orientation, j.localAnchorA);
            float3 wb = transformPoint(bb.position, bb.orientation, j.localAnchorB);
            float3 C = wa - wb;
            float3 local = isA ? j.localAnchorA : j.localAnchorB;
            float3 lam = float3(j.lambda0, j.lambda1, j.lambda2);
            float k = max(j.k0, p.kStart);
            contributeBallJoint(A, rhs, b.position, b.orientation, local, C, j.c0, k, lam, isA, p.alpha);
            if (j.jointType == JOINT_FIXED || j.jointType == JOINT_HINGE || j.jointType == JOINT_D6)
            {
                BodyState other = isA ? bb : ba;
                float3 axis = isA ? j.axisA : j.axisB;
                float3 worldAxis = quatRotate(b.orientation, axis);
                float3 otherAxis = quatRotate(other.orientation, isA ? j.axisB : j.axisA);
                float3 Cang = cross(worldAxis, otherAxis);
                for (int ax = 0; ax < 3; ax++)
                {
                    if (j.jointType == JOINT_HINGE && ax == 0) continue;
                    Vec6 jj = vec6Zero();
                    jj.v[3 + ax] = isA ? 1.0 : -1.0;
                    float f = k * (1.0 - p.alpha) * Cang[ax] + (ax == 0 ? j.lambda3 : ax == 1 ? j.lambda4 : j.lambda5);
                    mat6AddOuter(A, jj, k);
                    vec6AddScaled(rhs, jj, f);
                }
            }
            if (j.jointType == JOINT_SLIDER)
            {
                float3 axis = normalize(j.axisA);
                float3 d = wa - wb;
                float3 radial = d - axis * dot(d, axis);
                for (int ax = 0; ax < 3; ax++)
                {
                    float3 n = float3(ax == 0 ? 1.0 : 0.0, ax == 1 ? 1.0 : 0.0, ax == 2 ? 1.0 : 0.0);
                    if (abs(dot(n, axis)) > 0.9) continue;
                    Vec6 jj = jacobianLin(n, isA ? (wa - b.position) : (wb - b.position), !isA);
                    float f = k * (1.0 - p.alpha) * radial[ax] + (ax == 0 ? j.lambda3 : ax == 1 ? j.lambda4 : j.lambda5);
                    mat6AddOuter(A, jj, k);
                    vec6AddScaled(rhs, jj, f);
                }
            }
        }
    }
}

void primalApply(uint bodyIdx, Mat6 A, Vec6 rhs)
{
    BodyState b = gBodies[bodyIdx];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
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
    uint color = PASS_U;
    uint count = SOLVE_COUNT(color);
    uint gid = dispatchIndex(tid);
    uint local = lid.x;
    // Storage loads are non-uniform in WGSL. Publish lanes through workgroup
    // memory; the WGSL post-pass rewrites the read as workgroupUniformLoad.
    if (local == 0u)
        gsUniformLanes = primalLanesForDegree(
            count, HOST_S(22) != 0u ? SOLVE_COUNT(COLOR_DEGREE_BASE + color) : 0u);
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
    if (local == 0u) gsUniformLanes = min(ALOAD(gCounters[0].colorCount), min(HOST_S(4), 32u));
    GROUP_SYNC();
    uint colorLimit = gsUniformLanes;
    for (uint color = PASS_U; color < colorLimit; color++)
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
            c = solverContact(i, pair.normal, pair.friction);
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
            float keep = 1.0 - p.alpha;
            float3 n = c.normal;
            float3 t1 = c.tangent1;
            float3 t2 = c.tangent2;
            float3 cReg = float3(
                c.c0N * keep + dot(n, dLinB - dLinA) - dot(cross(rAW, n), dAngA) + dot(cross(rBW, n), dAngB),
                c.c0T1 * keep + dot(t1, dLinB - dLinA) - dot(cross(rAW, t1), dAngA) + dot(cross(rBW, t1), dAngB),
                c.c0T2 * keep + dot(t2, dLinB - dLinA) - dot(cross(rAW, t2), dAngA) + dot(cross(rBW, t2), dAngB));
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
                c.kT = min(c.kT + p.betaLin * abs(cReg.y), p.kMaxT);
                c._pad2 = min(c._pad2 + p.betaLin * abs(cReg.z), p.kMaxT);
                c._pad0 = length(cReg.yz) < 1.0e-5 ? 1.0 : 0.0;
            }
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
        Joint j = gJoints[i];
        BodyState a = gBodies[j.bodyA];
        BodyState b = gBodies[j.bodyB];
        float3 wa = transformPoint(a.position, a.orientation, j.localAnchorA);
        float3 wb = transformPoint(b.position, b.orientation, j.localAnchorB);
        float3 C = wa - wb; // shared with primal
        float k = max(j.k0, p.kStart);
        float3 lam = float3(j.lambda0, j.lambda1, j.lambda2);
        // demo3d dual: C_hat = C - alpha * C0
        float3 Creg = C - p.alpha * j.c0;
        lam = k * Creg + lam;
        j.lambda0 = lam.x; j.lambda1 = lam.y; j.lambda2 = lam.z;
        j.k0 = min(k + p.betaLin * length(Creg), min(j.stiffness, p.kMaxN));
        j.k1 = j.k0; j.k2 = j.k0;
        gJoints[i] = j;
    }
}


// gCcdFlags reused pre-CCD: 1 = sleep candidate (slow + timer elapsed) after extract_velocity.
[shader("compute")]
[numthreads(64, 1, 1)]
void extract_velocity_sleep(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    AvbdParams p = PARAMS;
    BodyState b = gBodies[i];
    gCcdFlags[i] = 0;
    if (!bodyIsDynamic(b)) return;
    if (bodyIsSleeping(b)) return;

    // Step-start pose x⁻ was saved on the body by prediction (initialPos/Rot).
    float dt = max(p.dt, 1e-8);
    b.linearVel = (b.position - b.initialPos) / dt;
    // Angular velocity from quat delta
    float4 q0 = b.initialRot;
    float4 q1 = b.orientation;
    // Ensure same hemisphere
    if (dot(q0, q1) < 0.0) q1 = -q1;
    float4 dq = quatMul(q1, quatConjugate(q0));
    float3 av = 2.0 * dq.xyz / dt;
    if (dq.w < 0.0) av = -av;
    b.angularVel = av;

    float lin = length(b.linearVel);
    float ang = length(b.angularVel);
    if (lin < p.sleepLin && ang < p.sleepAng)
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
    writeDispatchArgs(66u, anyCandidate ? min(ALOAD(gCounters[0].manifoldCount), HOST_S(10)) : 0u);
    writeDispatchArgs(67u, anyCandidate ? gCounters[0].bodyCount : 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sleep_island_gate(uint3 tid : SV_DispatchThreadID)
{
    // One thread per manifold (same connectivity as contacts, ~2–4× fewer).
    // Ring smash: no sleep candidates → scratch[13]==0 → entire gate is a no-op.
    if (ALOAD(gScratchAtomics[13]) == 0u) return;
    uint i = dispatchIndex(tid);
    uint n = ALOAD(gCounters[0].manifoldCount);
    if (i >= n) return;
    Manifold m = gManifolds[i];
    if (m.contactCount == 0u) return;
    uint ba = gColliders[m.colliderA].body;
    uint bb = gColliders[m.colliderB].body;
    BodyState a = gBodies[ba];
    BodyState b = gBodies[bb];
    bool dynA = bodyIsDynamic(a);
    bool dynB = bodyIsDynamic(b);
    if (!dynA || !dynB) return;
    // Sleeping neighbor is fine (already out of the island). Awake non-candidate blocks.
    bool okA = bodyIsSleeping(a) || gCcdFlags[ba] != 0;
    bool okB = bodyIsSleeping(b) || gCcdFlags[bb] != 0;
    if (!okA || !okB)
    {
        if (dynA && !bodyIsSleeping(a)) gCcdFlags[ba] = 0;
        if (dynB && !bodyIsSleeping(b)) gCcdFlags[bb] = 0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sleep_island_commit(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    if (gCcdFlags[i] == 0) return;
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
    if (!bodyIsDynamic(b)) return;
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
