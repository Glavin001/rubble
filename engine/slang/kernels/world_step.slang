// Production phase entry points — one Slang source for Metal / SPIR-V / CUDA.
#include "../physics/common.slang"
#include "../physics/box_collide.slang"

RWStructuredBuffer<BodyState> gBodies;
RWStructuredBuffer<BodyState> gBodiesPrev;
RWStructuredBuffer<Collider> gColliders;
RWStructuredBuffer<Material> gMaterials;
RWStructuredBuffer<Contact> gContacts;
RWStructuredBuffer<Joint> gJoints;
RWStructuredBuffer<AvbdParams> gParams;
RWStructuredBuffer<StepCounters> gCounters;
RWStructuredBuffer<GpuCommand> gCommands;
RWStructuredBuffer<Aabb> gAabbs;
RWStructuredBuffer<CandidatePair> gCandidates;
RWStructuredBuffer<uint> gAwakeList;
RWStructuredBuffer<uint> gBodyColor;
RWStructuredBuffer<uint> gConstraintIndex; // CSR: body -> contact/joint indices
RWStructuredBuffer<uint> gConstraintIds;
RWStructuredBuffer<uint> gColorLists;      // packed body indices per color
RWStructuredBuffer<uint> gColorOffsets;    // color -> start in gColorLists
RWStructuredBuffer<uint> gColorCounts;
RWStructuredBuffer<float3> gHullVerts;
RWStructuredBuffer<uint> gEvents;          // ring: type, id, ...
RWStructuredBuffer<uint> gCcdFlags;        // per-body fast-motion flags
// Binding order must match engine-metal `bind_all` (declaration order → Metal [[buffer(N)]]).
RWStructuredBuffer<uint> gScratchAtomics;
RWStructuredBuffer<uint> gPassUniform;     // length-1 view: host binds offset into 0..63 table
RWStructuredBuffer<uint> gIndirectArgs;    // Metal MTLDispatchThreadgroupsIndirectArguments × maxColors
// Per-collider world OBB (filled in build_aabbs) — avoids rebuild in BP face-reject + NP.
struct WorldObbGpu
{
    float4 center;
    float4 half;
    float4 axis0;
    float4 axis1;
    float4 axis2;
};
RWStructuredBuffer<WorldObbGpu> gWorldObbs;
// Appended (buffer 25): do not insert earlier — Metal bind_all is declaration order.
RWStructuredBuffer<Manifold> gManifolds;

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
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.bodyCount && i >= 1) { /* commands counted separately */ }
    AvbdParams p = gParams[0];
    // Commands are densely packed; host sets command count in counters.iteration temporarily? use candidateCount as cmdCount staging — better: use overflow high bits.
    // Convention: gCounters[0].iteration holds command count for this kernel only before solve resets it.
    uint cmdCount = gScratchAtomics[0];
    if (i >= cmdCount) return;
    GpuCommand cmd = gCommands[i];
    if (cmd.op == CMD_WAKE_BODY)
    {
        BodyState b = gBodies[cmd.a];
        b.flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        b.sleepTimer = 0.0;
        gBodies[cmd.a] = b;
        InterlockedAdd(gCounters[0].wakeEvents, 1);
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
        uint nContacts = gCounters[0].contactCount;
        uint curOff = gScratchAtomics[11];
        for (uint ci = 0; ci < nContacts; ci++)
        {
            Contact ct = gContacts[curOff + ci];
            if (ct.colliderA == cmd.a || ct.colliderB == cmd.a)
            {
                ct.alive = 0;
                gContacts[curOff + ci] = ct;
            }
        }
    }
}

void predictOne(uint i)
{
    AvbdParams p = gParams[0];
    BodyState b = gBodies[i];
    // Sleeping/static: do not touch prev/pose (three-avbd mover compaction).
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    float dt = p.dt;
    // Adaptive warm-start: (v - v_prev)/dt along up (three-avbd / VBD). gBodiesPrev still
    // holds the previous step's snapshot (incl. that step's incoming velocity).
    float3 prevVel = gBodiesPrev[i].linearVel;
    float3 accel = (b.linearVel - prevVel) / max(dt, 1e-8);
    float accelExt = accel.y * sign(p.gravityY);
    float accelWeight = clamp(accelExt / max(abs(p.gravityY), 1e-6), 0.0, 1.0);
    // three-avbd FLAG_START_AT_REST: skip extrapolation when nearly still so few-iter
    // stacks do not pump from a bad warm start.
    float slow = length(b.linearVel) + length(b.angularVel) * 0.5;
    bool startAtRest = slow < 0.1;

    float3 inertialLin = b.position + b.linearVel * dt + float3(0, p.gravityY, 0) * (dt * dt);
    float4 inertialAng = quatNormalize(quatApplyDelta(b.orientation, b.angularVel * dt));
    // x⁻ snapshot for velocity extract (three-avbd initialPos).
    gBodiesPrev[i] = b;

    if (!startAtRest)
    {
        b.position = b.position + b.linearVel * dt + float3(0, p.gravityY, 0) * (accelWeight * dt * dt);
        b.orientation = quatNormalize(quatApplyDelta(b.orientation, b.angularVel * dt));
    }
    Aabb box;
    box.mn = inertialLin;
    box.mx = float3(inertialAng.x, inertialAng.y, inertialAng.z);
    box._pad0 = inertialAng.w;
    box._pad1 = 0.0;
    gAabbs[i] = box;
    gBodies[i] = b;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void predict_bodies(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCounters[0].bodyCount) return;
    predictOne(i);
}

// Previous-step awake list (host sets gScratchAtomics[7] = prev awake count).
[shader("compute")]
[numthreads(64, 1, 1)]
void predict_bodies_listed(uint3 tid : SV_DispatchThreadID)
{
    uint li = tid.x;
    uint n = gScratchAtomics[7];
    if (li >= n) return;
    predictOne(gAwakeList[li]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_aabbs(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.colliderCount) return;
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    if (bodyIsSleeping(b) && bodyIsDynamic(b))
    {
        // Keep previous AABB; skip refit
        return;
    }
    float3 worldPos = transformPoint(b.position, b.orientation, col.localPos);
    float4 worldRot = quatMul(b.orientation, col.localRot);
    float3 mn, mx;
    float margin = gParams[0].contactOffset;
    if (col.shape == SHAPE_SPHERE)
    {
        worldAabbSphere(worldPos, col.params.x + margin, mn, mx);
    }
    else if (col.shape == SHAPE_BOX)
    {
        worldAabbBox(worldPos, worldRot, col.params.xyz + margin, mn, mx);
    }
    else if (col.shape == SHAPE_CAPSULE)
    {
        worldAabbCapsule(worldPos, worldRot, col.params.x + margin, col.params.y, mn, mx);
    }
    else if (col.shape == SHAPE_PLANE)
    {
        mn = float3(-1e6, -1e6, -1e6);
        mx = float3(1e6, 1e6, 1e6);
    }
    else
    {
        // Convex: use sphere bound from params.w as radius hint
        worldAabbSphere(worldPos, col.params.w + margin, mn, mx);
    }
    Aabb box;
    box.mn = mn; box.mx = mx; box._pad0 = 0; box._pad1 = asfloat(col.body);
    // Store at collider index offset by max_bodies — host allocates gAabbs sized max_bodies+max_colliders
    // Convention: collider AABBs start at index max_bodies. Host sets gScratchAtomics[1] = max_bodies.
    uint base = gScratchAtomics[1];
    gAabbs[base + i] = box;
    // World OBB cache for box colliders (BP face-reject + narrowphase).
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

uint gridHash(int x, int y, int z)
{
    uint h = (uint)x * 73856093u ^ (uint)y * 19349663u ^ (uint)z * 83492791u;
    return h & (GRID_HASH_SIZE - 1u);
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
    uint i = tid.x;
    if (i >= JOINT_HASH_SIZE) return;
    gEvents[i] = JOINT_HASH_EMPTY;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void joint_hash_insert(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint nJ = gCounters[0].jointCount;
    if (i >= nJ) return;
    Joint jj = gJoints[i];
    uint key = jointPairKey(jj.bodyA, jj.bodyB);
    uint h = key & (JOINT_HASH_SIZE - 1u);
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = (h + probe) & (JOINT_HASH_SIZE - 1u);
        uint prev;
        InterlockedCompareExchange(gEvents[slot], JOINT_HASH_EMPTY, key, prev);
        if (prev == JOINT_HASH_EMPTY || prev == key) return;
    }
}

bool bodiesJointed(uint bodyA, uint bodyB)
{
    uint key = jointPairKey(bodyA, bodyB);
    uint h = key & (JOINT_HASH_SIZE - 1u);
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = (h + probe) & (JOINT_HASH_SIZE - 1u);
        uint v = gEvents[slot];
        if (v == JOINT_HASH_EMPTY) return false;
        if (v == key) return true;
    }
    return false;
}

void emitCandidate(uint i, uint j)
{
    if (i == j) return;
    uint a = min(i, j);
    uint b = max(i, j);
    uint slot;
    InterlockedAdd(gCounters[0].candidateCount, 1, slot);
    uint maxCand = gScratchAtomics[2];
    if (slot >= maxCand)
    {
        InterlockedOr(gCounters[0].overflowFlags, OVERFLOW_CANDIDATES);
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
    uint i = tid.x;
    if (i >= GRID_HASH_SIZE) return;
    gConstraintIds[i] = 0xFFFFFFFFu;
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
    if (cb.shape != SHAPE_BOX) return false;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_insert(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.colliderCount) return;
    Collider col = gColliders[i];
    if (col.shape == SHAPE_PLANE) return;
    // Insert sleeping too — awake movers must find them (island wake on impact).
    uint base = gScratchAtomics[1];
    float cellSize = max(gParams[0].contactOffset * 16.0, 0.5);
    Aabb box = gAabbs[base + i];
    float3 ext = box.mx - box.mn;
    // three-avbd isLarge: keep oversized statics out of the hash grid; dynamics
    // find them via the brute-force large-static pass in broadphase_query.
    BodyState b = gBodies[col.body];
    if (!bodyIsDynamic(b) &&
        (ext.x >= cellSize * 4.0 || ext.y >= cellSize * 4.0 || ext.z >= cellSize * 4.0))
        return;
    float3 center = 0.5 * (box.mn + box.mx);
    int3 cc = int3(floor(center / cellSize));
    uint h = gridHash(cc.x, cc.y, cc.z);
    uint prev;
    InterlockedExchange(gConstraintIds[h], i, prev);
    gColorLists[i] = prev;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_query(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.colliderCount) return;
    Collider col = gColliders[i];
    BodyState b = gBodies[col.body];
    if (!bodyIsDynamic(b)) return;
    if (col.shape == SHAPE_PLANE) return;
    uint base = gScratchAtomics[1];
    Aabb box = gAabbs[base + i];
    bool sleeper = bodyIsSleeping(b);
    // Probe OBB from build_aabbs cache (three-avbd findPairs loads querier once).
    bool probeIsBox = col.shape == SHAPE_BOX;
    BoxObb probeObb = probeIsBox ? worldObbFromCache(i) : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    const float faceGap = 0.001;

    // Small scenes: exact pairwise (correctness). Large: uniform-grid hash.
    // Awake dynamics test everyone; sleepers only keep static contacts (ground) so a
    // later impact can wake the island without the floor dropping out for a frame.
    if (c.colliderCount <= 512u)
    {
        for (uint j = 0; j < c.colliderCount; j++)
        {
            if (j == i) continue;
            Collider other = gColliders[j];
            if (other.body == col.body) continue;
            BodyState ob = gBodies[other.body];
            bool otherStatic = !bodyIsDynamic(ob);
            if (sleeper && !otherStatic) continue;
            if (bodyIsSleeping(ob) && sleeper) continue;
            // Emit each dynamic-dynamic pair once (lower index). Dynamic vs static always
            // emits from the dynamic collider (three-avbd box ground, planes).
            if (!otherStatic && j < i) continue;
            if (otherStatic && !bodyIsDynamic(b)) continue;
            if (bodiesJointed(col.body, other.body)) continue;
            if (other.shape != SHAPE_PLANE)
            {
                Aabb ab = gAabbs[base + j];
                if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
                if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
                if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
            }
            if (probeIsBox && boxesSeparatedOnFaceAxis(probeObb, j, faceGap)) continue;
            emitCandidate(i, j);
        }
        return;
    }

    float cellSize = max(gParams[0].contactOffset * 16.0, 0.5);
    int3 mn = int3(floor(box.mn / cellSize));
    int3 mx = int3(floor(box.mx / cellSize));
    mn = max(mn, mx - int3(1, 1, 1));
    mx = min(mx, mn + int3(1, 1, 1));

    for (uint p = 0; p < c.colliderCount; p++)
    {
        if (gColliders[p].shape != SHAPE_PLANE) continue;
        emitCandidate(i, p);
    }
    // Sleepers: keep contacts vs static boxes (thick ground) too — grid pair loop is
    // awake-only below, so scan statics here.
    if (sleeper)
    {
        for (uint j = 0; j < c.colliderCount; j++)
        {
            if (j == i) continue;
            Collider other = gColliders[j];
            if (other.body == col.body || other.shape == SHAPE_PLANE) continue;
            BodyState ob = gBodies[other.body];
            if (bodyIsDynamic(ob)) continue;
            if (bodiesJointed(col.body, other.body)) continue;
            Aabb ab = gAabbs[base + j];
            if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
            if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
            if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
            emitCandidate(i, j);
        }
        return;
    }
    for (int z = mn.z - 1; z <= mx.z + 1; z++)
    for (int y = mn.y - 1; y <= mx.y + 1; y++)
    for (int x = mn.x - 1; x <= mx.x + 1; x++)
    {
        uint h = gridHash(x, y, z);
        uint cur = gConstraintIds[h];
        uint guard = 0;
        while (cur != 0xFFFFFFFFu && guard < 256u)
        {
            guard++;
            // Dynamic-dynamic once (higher index). Statics are not in this hash (planes
            // skip insert; large box grounds use the O(1) probe below) — keep the
            // cur > i early-out so we do not touch every lower-index chain node.
            if (cur > i)
            {
                Collider other = gColliders[cur];
                if (other.body != col.body &&
                    !bodiesJointed(col.body, other.body) &&
                    other.shape != SHAPE_PLANE)
                {
                    // three-avbd testPair: bounding spheres before AABB/face (rings/piles
                    // otherwise send 48-72% of pairs to NP only to find them apart).
                    BodyState ob = gBodies[other.body];
                    float3 d = b.position - ob.position;
                    float rA = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
                    float rB = other.params.w > 0.0 ? other.params.w : length(other.params.xyz);
                    float rr = rA + rB + 2.0 * gParams[0].contactOffset;
                    if (dot(d, d) > rr * rr)
                    {
                        cur = gColorLists[cur];
                        continue;
                    }
                    Aabb ab = gAabbs[base + cur];
                    if (!(box.mx.x < ab.mn.x || ab.mx.x < box.mn.x ||
                          box.mx.y < ab.mn.y || ab.mx.y < box.mn.y ||
                          box.mx.z < ab.mn.z || ab.mx.z < box.mn.z))
                    {
                        if (!(probeIsBox && boxesSeparatedOnFaceAxis(probeObb, cur, faceGap)))
                            emitCandidate(i, cur);
                    }
                }
            }
            cur = gColorLists[cur];
        }
    }
    // three-avbd isLarge: ground slab is collider 0 and skipped from the hash grid
    // (one cell at its centre never covers the arena). Probe it O(1) per dynamic.
    if (c.colliderCount > 0u && i != 0u)
    {
        uint j = 0u;
        Collider other = gColliders[j];
        BodyState ob = gBodies[other.body];
        if (!bodyIsDynamic(ob) && other.shape != SHAPE_PLANE)
        {
            Aabb ab = gAabbs[base + j];
            float3 ext = ab.mx - ab.mn;
            if (ext.x >= cellSize * 4.0 || ext.y >= cellSize * 4.0 || ext.z >= cellSize * 4.0)
            {
                if (!(box.mx.x < ab.mn.x || ab.mx.x < box.mn.x ||
                      box.mx.y < ab.mn.y || ab.mx.y < box.mn.y ||
                      box.mx.z < ab.mn.z || ab.mx.z < box.mn.z))
                {
                    if (!(probeIsBox && boxesSeparatedOnFaceAxis(probeObb, j, faceGap)))
                        emitCandidate(i, j);
                }
            }
        }
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
    // Cold start at kStart; persist_contacts raises to massPenalty (m/dt²) before solve.
    ct.kN = gParams[0].kStart;
    ct.kT = gParams[0].kStart;
    ct.lambdaT2 = 0.0;
    ct.c0T1 = C0t1;
    ct.c0T2 = C0t2;
    ct._pad0 = 0.0; // stick flag (1 = static friction held)
    ct._pad1 = 0.0;
    ct._pad2 = 0.0;
    return ct;
}

void wakeFromContact(BodyState ba, BodyState bb, Collider ca, Collider cb)
{
    // Wake only a sleeping body hit by an awake neighbour (island wake). Never clear
    // SLEEPING on resting pairs — that used to re-awake stacks every NP and pump KE.
    bool awakeA = bodyIsDynamic(ba) && !bodyIsSleeping(ba);
    bool awakeB = bodyIsDynamic(bb) && !bodyIsSleeping(bb);
    if (bodyIsDynamic(ba) && bodyIsSleeping(ba) && awakeB)
    {
        ba.flags = (ba.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        ba.sleepTimer = 0.0;
        gBodies[ca.body] = ba;
    }
    if (bodyIsDynamic(bb) && bodyIsSleeping(bb) && awakeA)
    {
        bb.flags = (bb.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        bb.sleepTimer = 0.0;
        gBodies[cb.body] = bb;
    }
}

// three-avbd Stage 8b: one manifold record + contiguous contact points.
void emitContactManifold(
    CandidatePair pair, BodyState ba, BodyState bb, Collider ca, Collider cb,
    float3 nrm, float margin, int count,
    float3 pA0, float3 pB0, uint f0,
    float3 pA1, float3 pB1, uint f1,
    float3 pA2, float3 pB2, uint f2,
    float3 pA3, float3 pB3, uint f3)
{
    if (count <= 0) return;
    uint maxC = gScratchAtomics[3];
    uint maxM = gScratchAtomics[10];
    uint mslot;
    InterlockedAdd(gCounters[0].manifoldCount, 1, mslot);
    if (mslot >= maxM)
    {
        InterlockedOr(gCounters[0].overflowFlags, OVERFLOW_MANIFOLDS);
        return;
    }
    uint cStart;
    InterlockedAdd(gCounters[0].contactCount, (uint)count, cStart);
    uint curOff = gScratchAtomics[11];
    if (cStart >= maxC || cStart + (uint)count > maxC)
    {
        InterlockedOr(gCounters[0].overflowFlags, OVERFLOW_CONTACTS);
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
            gContacts[curOff + cStart + k] = z;
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

    Material ma = gMaterials[ca.material];
    Material mb = gMaterials[cb.material];
    float mu = sqrt(max(ma.friction, 0.0) * max(mb.friction, 0.0));

    gContacts[curOff + cStart] = makeContactRow(pair, ba, bb, ca, cb, nrm, pA0, pB0, f0, margin, mu);
    if (count > 1)
        gContacts[curOff + cStart + 1u] = makeContactRow(pair, ba, bb, ca, cb, nrm, pA1, pB1, f1, margin, mu);
    if (count > 2)
        gContacts[curOff + cStart + 2u] = makeContactRow(pair, ba, bb, ca, cb, nrm, pA2, pB2, f2, margin, mu);
    if (count > 3)
        gContacts[curOff + cStart + 3u] = makeContactRow(pair, ba, bb, ca, cb, nrm, pA3, pB3, f3, margin, mu);

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
    if (mf.count <= 0) return;
    // Feature IDs must be geometric only (three-avbd). Cap 4 — matches BOX_MAX_CONTACTS.
    int n = mf.count > 4 ? 4 : mf.count;
    emitContactManifold(
        pair, ba, bb, ca, cb, mf.normal, margin, n,
        mf.pts[0].pA, mf.pts[0].pB, mf.pts[0].feature,
        mf.pts[1].pA, mf.pts[1].pB, mf.pts[1].feature,
        mf.pts[2].pA, mf.pts[2].pB, mf.pts[2].feature,
        mf.pts[3].pA, mf.pts[3].pB, mf.pts[3].feature);
}

void emitContact(
    CandidatePair pair, BodyState ba, BodyState bb, Collider ca, Collider cb,
    float3 nrm, float3 pA, float3 pB, uint featureId, float margin)
{
    emitContactManifold(
        pair, ba, bb, ca, cb, nrm, margin, 1,
        pA, pB, featureId,
        float3(0, 0, 0), float3(0, 0, 0), 0u,
        float3(0, 0, 0), float3(0, 0, 0), 0u,
        float3(0, 0, 0), float3(0, 0, 0), 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void narrowphase(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].candidateCount;
    if (i >= n) return;
    CandidatePair pair = gCandidates[i];
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    float3 posA = transformPoint(ba.position, ba.orientation, ca.localPos);
    float3 posB = transformPoint(bb.position, bb.orientation, cb.localPos);
    float4 rotA = quatMul(ba.orientation, ca.localRot);
    float4 rotB = quatMul(bb.orientation, cb.localRot);
    float margin = gParams[0].contactOffset;
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
        float3 local = quatRotate(quatConjugate(rotA), posB - posA);
        float3 half = ca.params.xyz;
        float3 q = clamp(local, -half, half);
        float3 diff = local - q;
        float dist = length(diff);
        float r = cb.params.x + margin;
        if (dist < r)
        {
            hit = true;
            nrm = dist > 1e-8 ? quatRotate(rotA, diff / dist) : float3(0, 1, 0);
            pen = r - dist;
            pA = transformPoint(posA, rotA, q);
            pB = posB - nrm * cb.params.x;
        }
    }
    else if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_BOX)
    {
        float3 local = quatRotate(quatConjugate(rotB), posA - posB);
        float3 half = cb.params.xyz;
        float3 q = clamp(local, -half, half);
        float3 diff = local - q;
        float dist = length(diff);
        float r = ca.params.x + margin;
        if (dist < r)
        {
            hit = true;
            nrm = dist > 1e-8 ? -quatRotate(rotB, diff / dist) : float3(0, 1, 0);
            pen = r - dist;
            pB = transformPoint(posB, rotB, q);
            pA = posA + nrm * ca.params.x;
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

// demo3d Joint::initialize — store C0, decay λ/k (Eq. 19).
[shader("compute")]
[numthreads(64, 1, 1)]
void warmstart_joints(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].jointCount;
    if (i >= n) return;
    AvbdParams p = gParams[0];
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

// Prev-contact warm-start hash in gConstraintIds (open address). Built after NP,
// consumed by persist, then overwritten by CSR. Replaces O(n²) linear scan.
static const uint CONTACT_HASH_EMPTY = 0xFFFFFFFFu;

uint contactWarmKey(uint colA, uint colB, uint feature)
{
    uint h = colA * 73856093u ^ colB * 19349663u ^ feature * 83492791u;
    return h == CONTACT_HASH_EMPTY ? (colA ^ colB ^ feature ^ 0xA5A5A5A5u) : h;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void contact_hash_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gScratchAtomics[9]; // hash table word count
    if (i >= n) return;
    gConstraintIds[i] = CONTACT_HASH_EMPTY;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void contact_hash_insert(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x;
    uint prevN = gScratchAtomics[7];
    if (j >= prevN) return;
    uint prevOff = gScratchAtomics[8];
    Contact prev = gContacts[prevOff + j];
    if (prev.alive == 0) return;
    uint key = contactWarmKey(prev.colliderA, prev.colliderB, prev.featureId);
    uint n = gScratchAtomics[9];
    uint h = key & (n - 1u); // n is power of two
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = (h + probe) & (n - 1u);
        uint was;
        InterlockedCompareExchange(gConstraintIds[slot], CONTACT_HASH_EMPTY, j, was);
        if (was == CONTACT_HASH_EMPTY) return;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void persist_contacts(uint3 tid : SV_DispatchThreadID)
{
    // Exact featureId warm-start via contact hash (three-avbd hashInsert), then Eq. 19.
    uint i = tid.x;
    uint n = gCounters[0].contactCount;
    if (i >= n) return;
    AvbdParams p = gParams[0];
    uint curOff = gScratchAtomics[11];
    Contact c = gContacts[curOff + i];
    if (c.alive == 0) return;
    BodyState ba = gBodies[gColliders[c.colliderA].body];
    BodyState bb = gBodies[gColliders[c.colliderB].body];
    bool bothSleep =
        (!bodyIsDynamic(ba) || bodyIsSleeping(ba)) &&
        (!bodyIsDynamic(bb) || bodyIsSleeping(bb));
    if (bothSleep) return;

    uint prevOff = gScratchAtomics[8];
    uint hashN = gScratchAtomics[9];
    uint key = contactWarmKey(c.colliderA, c.colliderB, c.featureId);
    uint h = key & (hashN - 1u);
    uint bestJ = CONTACT_HASH_EMPTY;
    for (uint probe = 0; probe < 64u; probe++)
    {
        uint slot = (h + probe) & (hashN - 1u);
        uint j = gConstraintIds[slot];
        if (j == CONTACT_HASH_EMPTY) break;
        Contact prev = gContacts[prevOff + j];
        if (prev.alive != 0 &&
            prev.colliderA == c.colliderA &&
            prev.colliderB == c.colliderB &&
            prev.featureId == c.featureId)
        {
            bestJ = j;
            break;
        }
    }
    if (bestJ != CONTACT_HASH_EMPTY)
    {
        Contact prev = gContacts[prevOff + bestJ];
        c.lambdaN = prev.lambdaN;
        c.lambdaT1 = prev.lambdaT1;
        c.lambdaT2 = prev.lambdaT2;
        c.kN = prev.kN;
        c.kT = prev.kT;
        // three-avbd: keep old anchors while static friction held (stick).
        if (prev._pad0 > 0.5)
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

    float dt = max(p.dt, 1e-8);
    float mA = ba.invMass > 1e-12 ? (1.0 / ba.invMass) : 1e20;
    float mB = bb.invMass > 1e-12 ? (1.0 / bb.invMass) : 1e20;
    // three-avbd FLAG_MASS_PENALTY: kN ≥ min(m)/dt² so resting contacts hold under gravity
    // with Jacobi (was deferred to stick-only while cReg had a margin bug).
    float kMass = min(mA, mB) / (dt * dt);
    c.kN = max(max(p.kStart, kMass), p.gamma * c.kN);
    c.kT = max(max(p.kStart, kMass * 0.1), p.gamma * c.kT);
    c.lambdaN *= p.alpha * p.gamma;
    c.lambdaT1 *= p.alpha * p.gamma;
    c.lambdaT2 *= p.alpha * p.gamma;
    gContacts[curOff + i] = c;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_awake_list(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.bodyCount) return;
    BodyState b = gBodies[i];
    if (bodyIsDynamic(b) && !bodyIsSleeping(b))
    {
        uint slot;
        InterlockedAdd(gCounters[0].awakeCount, 1, slot);
        gAwakeList[slot] = i;
    }
}

// --- CSR adjacency (three-avbd: degree → scan → scatter → sort) ----------------------------
// Constraint id: [0, jointCount) = joints; jointCount + mi = manifold mi (Stage 8b).
// gConstraintIndex[0..bodyCount]: degree then exclusive starts; [bodyCount] = total.
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

void csrCountEndpoint(uint body)
{
    if (!bodyIsDynamic(gBodies[body])) return;
    InterlockedAdd(gConstraintIndex[body], 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].bodyCount + 1u;
    if (i < n) gConstraintIndex[i] = 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_degree_contacts(uint3 tid : SV_DispatchThreadID)
{
    // Name kept for host pipeline map; degrees manifolds (one edge per pair).
    uint i = tid.x;
    uint n = gCounters[0].manifoldCount;
    if (i >= n) return;
    Manifold m = gManifolds[i];
    if (m.contactCount == 0u) return;
    uint2 e = constraintEndpoints(gCounters[0].jointCount + i);
    csrCountEndpoint(e.x);
    csrCountEndpoint(e.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_degree_joints(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].jointCount;
    if (i >= n) return;
    uint2 e = constraintEndpoints(i);
    csrCountEndpoint(e.x);
    csrCountEndpoint(e.y);
}

// Parallel exclusive prefix (Hillis–Steele). Host binds gPassUniform = 1,2,4,... via
// pass_table (no mid-encoder CPU scratch writes). gBodyColor holds the running sum.
[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_init(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].bodyCount;
    if (i >= n) return;
    gBodyColor[i] = gConstraintIndex[i]; // degree
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_scan(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].bodyCount;
    if (i >= n) return;
    uint offset = gPassUniform[0];
    uint v = gBodyColor[i];
    if (i >= offset) v += gBodyColor[i - offset];
    gConstraintIndex[i] = v;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_swap(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].bodyCount;
    if (i >= n) return;
    gBodyColor[i] = gConstraintIndex[i];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_finalize(uint3 tid : SV_DispatchThreadID)
{
    // Inclusive scan is in gBodyColor; convert to exclusive starts in constraint_index
    // and leave fill cursors in gBodyColor (= exclusive start).
    uint i = tid.x;
    uint n = gCounters[0].bodyCount;
    if (i > n) return;
    if (i == n)
    {
        gConstraintIndex[n] = (n == 0u) ? 0u : gBodyColor[n - 1u];
        return;
    }
    uint incl = gBodyColor[i];
    uint excl = (i == 0u) ? 0u : gBodyColor[i - 1u];
    gConstraintIndex[i] = excl;
    gBodyColor[i] = excl; // fill cursor
    // Keep inclusive total on last element available via constraint_index[n] above.
    incl = incl; // silence
}

// Back-compat single-thread path for tiny worlds (tests).
[shader("compute")]
[numthreads(1, 1, 1)]
void csr_prefix(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    uint n = gCounters[0].bodyCount;
    uint sum = 0;
    for (uint i = 0; i < n; i++)
    {
        uint d = gConstraintIndex[i];
        gConstraintIndex[i] = sum;
        gBodyColor[i] = sum;
        sum += d;
    }
    gConstraintIndex[n] = sum;
}

void csrFillEndpoint(uint body, uint id)
{
    if (!bodyIsDynamic(gBodies[body])) return;
    uint slot;
    InterlockedAdd(gBodyColor[body], 1, slot);
    gConstraintIds[slot] = id;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_scatter_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].manifoldCount;
    if (i >= n) return;
    Manifold m = gManifolds[i];
    if (m.contactCount == 0u) return;
    uint id = gCounters[0].jointCount + i;
    uint2 e = constraintEndpoints(id);
    csrFillEndpoint(e.x, id);
    csrFillEndpoint(e.y, id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_scatter_joints(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].jointCount;
    if (i >= n) return;
    uint2 e = constraintEndpoints(i);
    csrFillEndpoint(e.x, i);
    csrFillEndpoint(e.y, i);
}

// Deterministic adjacency order (three-avbd sortAdjacency) — joints first by id, then manifolds.
// Indexed by awake list so sleeping fields with high body ids stay cheap.
[shader("compute")]
[numthreads(64, 1, 1)]
void csr_sort(uint3 tid : SV_DispatchThreadID)
{
    uint li = tid.x;
    uint awakeN = gCounters[0].awakeCount;
    if (li >= awakeN) return;
    uint b = gAwakeList[li];
    uint lo = gConstraintIndex[b];
    uint hi = gConstraintIndex[b + 1u];
    uint nJ = gCounters[0].jointCount;
    for (uint i = lo + 1u; i < hi; i++)
    {
        uint id = gConstraintIds[i];
        uint key0 = id < nJ ? 0u : 1u;
        uint key1 = id < nJ ? id : (id - nJ);
        uint j = i;
        while (j > lo)
        {
            uint prev = gConstraintIds[j - 1u];
            uint pk0 = prev < nJ ? 0u : 1u;
            uint pk1 = prev < nJ ? prev : (prev - nJ);
            if (key0 > pk0 || (key0 == pk0 && key1 >= pk1)) break;
            gConstraintIds[j] = prev;
            j--;
        }
        gConstraintIds[j] = id;
    }
}

uint jpPriority(uint body, uint round, bool heightBias)
{
    uint prio = heightBias
        ? ((~asuint(gBodies[body].position.y)) + round * 0x85ebca6bu + body * 0x9e3779b9u)
        : (body * 0x9e3779b9u ^ round * 0x85ebca6bu);
    return prio ^ (prio >> 16);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void color_bodies(uint3 tid : SV_DispatchThreadID)
{
    // Jones-Plassmann: winners take the *smallest free* colour (three-avbd), not the
    // round index — assigning round forced O(rounds) colours and killed parallelism.
    uint li = tid.x;
    uint awakeN = gCounters[0].awakeCount;
    if (li >= awakeN) return;
    uint body = gAwakeList[li];
    uint round = gPassUniform[0];
    // Height-biased JP priority: lower bodies win early rounds (bottom-up support).
    // Still assign smallest-free colours (not the round id) so the palette stays tight.
    bool heightBias = true;
    uint prio = jpPriority(body, round, heightBias);

    if (gBodyColor[body] != 0xFFFFFFFFu) return;

    bool blocked = false;
    uint usedLo = 0u;
    uint usedHi = 0u;
    uint lo = gConstraintIndex[body];
    uint hi = gConstraintIndex[body + 1u];
    for (uint e = lo; e < hi; e++)
    {
        uint2 ends = constraintEndpoints(gConstraintIds[e]);
        uint other = ends.x == body ? ends.y : ends.x;
        if (!bodyIsDynamic(gBodies[other]) || bodyIsSleeping(gBodies[other])) continue;
        uint oc = gBodyColor[other];
        if (oc == 0xFFFFFFFFu)
        {
            uint op = jpPriority(other, round, heightBias);
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
    gBodyColor[body] = free;
    InterlockedMax(gCounters[0].colorCount, free + 1);
}

// three-avbd primalLanes defaults → 1/2/4/8 threads per body by colour size.
uint primalLanesFor(uint bodies)
{
    if (bodies == 0u || bodies >= 65536u) return 1u;
    if (bodies >= 32768u) return 2u;
    if (bodies >= 4096u) return 4u;
    return 8u;
}

// Write Metal indirect threadgroup counts for each colour (0 groups ⇒ skipped).
[shader("compute")]
[numthreads(64, 1, 1)]
void write_color_args(uint3 tid : SV_DispatchThreadID)
{
    uint c = tid.x;
    if (c >= 64u) return;
    uint n = gColorCounts[c];
    uint lanes = primalLanesFor(n);
    uint threads = n * lanes;
    uint groups = threads == 0u ? 0u : (threads + 63u) / 64u;
    gIndirectArgs[c * 3u + 0u] = groups;
    gIndirectArgs[c * 3u + 1u] = 1u;
    gIndirectArgs[c * 3u + 2u] = 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init_colors(uint3 tid : SV_DispatchThreadID)
{
    uint li = tid.x;
    uint awakeN = gCounters[0].awakeCount;
    if (li >= awakeN) return;
    gBodyColor[gAwakeList[li]] = 0xFFFFFFFFu;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pack_color_lists(uint3 tid : SV_DispatchThreadID)
{
    uint li = tid.x;
    uint awakeN = gCounters[0].awakeCount;
    if (li >= awakeN) return;
    uint body = gAwakeList[li];
    uint color = gBodyColor[body];
    if (color == 0xFFFFFFFFu) color = 0;
    uint slot;
    InterlockedAdd(gColorCounts[color], 1, slot);
    // Second pass needed for offsets — host compact or prefix. Store body at atomic slot in scratch region.
    uint maxB = gScratchAtomics[1];
    gColorLists[color * maxB + slot] = body;
}

// Accumulate body Newton's system for adjacency entries lane, lane+lanes, ...
// Lane 0 also adds the inertial mass block (three-avbd accumulate).
void primalAccumulate(uint bodyIdx, uint lane, uint lanes, out Mat6 A, out Vec6 rhs)
{
    A = mat6Zero();
    rhs = vec6Zero();
    BodyState b = gBodies[bodyIdx];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    AvbdParams p = gParams[0];
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
        float4 dq = quatMul(quatConjugate(inertialAng), q);
        float3 dangTarget = 2.0 * dq.xyz;
        if (dq.w < 0.0) dangTarget = -dangTarget;
        float dangLen = length(dangTarget);
        if (dangLen > 0.5) dangTarget *= 0.5 / dangLen;
        rhs.v[3] = (inertia.x / dt2) * dangTarget.x;
        rhs.v[4] = (inertia.y / dt2) * dangTarget.y;
        rhs.v[5] = (inertia.z / dt2) * dangTarget.z;
    }

    uint nJ = gCounters[0].jointCount;
    uint alo = gConstraintIndex[bodyIdx];
    uint ahi = gConstraintIndex[bodyIdx + 1u];
    for (uint e = alo + lane; e < ahi; e += lanes)
    {
        uint id = gConstraintIds[e];
        if (id >= nJ)
        {
            // Manifold edge: load both bodies once, fold every contact point (three-avbd).
            Manifold mf = gManifolds[id - nJ];
            if (mf.contactCount == 0u) continue;
            uint ba = gColliders[mf.colliderA].body;
            uint bb = gColliders[mf.colliderB].body;
            bool isA = ba == bodyIdx;
            BodyState bA = gBodies[ba];
            BodyState bB = gBodies[bb];
            for (uint ci = 0u; ci < mf.contactCount; ci++)
            {
                Contact c = gContacts[mf.contactStart + ci];
                if (c.alive == 0) continue;
                float3 wa = transformPoint(bA.position, bA.orientation, c.anchorA);
                float3 wb = transformPoint(bB.position, bB.orientation, c.anchorB);
                float3 d = wb - wa;
                float3 C = float3(dot(c.normal, d), dot(c.tangent1, d), dot(c.tangent2, d));
                c.penetration = C.x;
                contributeContact(A, rhs, c, isA, p.alpha, b.position, b.orientation, C, p.contactOffset);
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
    AvbdParams p = gParams[0];
    uint floors = 0;
    Vec6 neg = rhs;
    for (int k = 0; k < 6; k++) neg.v[k] = -rhs.v[k];
    Vec6 dz = solve6x6(A, neg, p.pivotFloor, floors);
    if (floors > 0) InterlockedAdd(gCounters[0].pivotFloorHits, floors);
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
    b.orientation = quatApplyDelta(b.orientation, dang);
    gBodies[bodyIdx] = b;
}

void solvePrimalForBody(uint bodyIdx)
{
    Mat6 A;
    Vec6 rhs;
    primalAccumulate(bodyIdx, 0u, 1u, A, rhs);
    primalApply(bodyIdx, A, rhs);
}

// Multi-lane partial systems: ≤8 lanes ⇒ ≤8 bodies per workgroup.
groupshared float gsA[64 * 36];
groupshared float gsRhs[64 * 6];

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_primal_color(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint color = gPassUniform[0];
    uint count = gColorCounts[color];
    uint lanes = primalLanesFor(count);
    uint threads = count * lanes;
    uint gid = tid.x;
    uint local = lid.x;
    uint bodySlot = gid / lanes;
    uint lane = local % lanes;
    uint slot = local / lanes; // body slot within the workgroup
    bool live = gid < threads && bodySlot < count;

    Mat6 A = mat6Zero();
    Vec6 rhs = vec6Zero();
    uint bodyIdx = 0u;
    uint maxB = gScratchAtomics[1];
    if (live)
    {
        bodyIdx = gColorLists[color * maxB + bodySlot];
        primalAccumulate(bodyIdx, lane, lanes, A, rhs);
    }

    if (lanes == 1u)
    {
        if (live) primalApply(bodyIdx, A, rhs);
        return;
    }

    // Store partials by local id; reduce lanes (three-avbd primalColor).
    if (live)
    {
        for (int i = 0; i < 36; i++) gsA[local * 36 + i] = A.m[i];
        for (int i = 0; i < 6; i++) gsRhs[local * 6 + i] = rhs.v[i];
    }
    AllMemoryBarrierWithGroupSync();
    // Halving reduction across lanes for each body slot
    for (uint s = lanes >> 1; s > 0u; s >>= 1)
    {
        if (live && lane < s)
        {
            uint other = local + s;
            for (int i = 0; i < 36; i++) A.m[i] += gsA[other * 36 + i];
            for (int i = 0; i < 6; i++) rhs.v[i] += gsRhs[other * 6 + i];
            for (int i = 0; i < 36; i++) gsA[local * 36 + i] = A.m[i];
            for (int i = 0; i < 6; i++) gsRhs[local * 6 + i] = rhs.v[i];
        }
        AllMemoryBarrierWithGroupSync();
        if (live && lane < s)
        {
            for (int i = 0; i < 36; i++) A.m[i] = gsA[local * 36 + i];
            for (int i = 0; i < 6; i++) rhs.v[i] = gsRhs[local * 6 + i];
        }
        AllMemoryBarrierWithGroupSync();
    }
    if (live && lane == 0u) primalApply(bodyIdx, A, rhs);
}

// Small-scene GS: one body per dispatch (gPassUniform = body index). Host walks
// bodies high→low like three-avbd CPU so impacts propagate through stacks.
[shader("compute")]
[numthreads(1, 1, 1)]
void solve_primal_one(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0) return;
    solvePrimalForBody(gPassUniform[0]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_dual(uint3 tid : SV_DispatchThreadID)
{
    // Per-contact dual (Metal occupancy). CSR/primal still use manifolds (Stage 8b).
    uint i = tid.x;
    uint n = gCounters[0].contactCount;
    AvbdParams p = gParams[0];
    if (i < n)
    {
        uint curOff = gScratchAtomics[11];
        Contact c = gContacts[curOff + i];
        if (c.alive != 0)
        {
            uint ba = gColliders[c.colliderA].body;
            uint bb = gColliders[c.colliderB].body;
            BodyState bA = gBodies[ba];
            BodyState bB = gBodies[bb];
            // beat-box / three-avbd: dual stasis when both ends are asleep or static.
            bool sleepA = !bodyIsDynamic(bA) || bodyIsSleeping(bA);
            bool sleepB = !bodyIsDynamic(bB) || bodyIsSleeping(bB);
            if (sleepA && sleepB) { /* leave λ/k */ }
            else
            {
            float3 wa = transformPoint(bA.position, bA.orientation, c.anchorA);
            float3 wb = transformPoint(bB.position, bB.orientation, c.anchorB);
            float3 d = wb - wa;
            float3 C = float3(dot(c.normal, d), dot(c.tangent1, d), dot(c.tangent2, d));
            c.penetration = C.x;
            float3 C0 = float3(c.c0N, c.c0T1, c.c0T2);
            // Match contributeContact / three-avbd: margin is in C0, not in geometric C.
            float3 cReg = C - p.alpha * C0 + float3(p.contactOffset, 0.0, 0.0);
            float3 f = float3(c.kN, c.kT, c.kT) * cReg + float3(c.lambdaN, c.lambdaT1, c.lambdaT2);
            f = projectContactForce(f, c.friction);
            c.lambdaN = f.x;
            c.lambdaT1 = f.y;
            c.lambdaT2 = f.z;
            // Penalty ramp (three-avbd: ramp when force in bounds)
            if (f.x < 0.0) c.kN = min(c.kN + p.betaLin * abs(cReg.x), p.kMaxN);
            float ft = length(float2(f.y, f.z));
            float bound = c.friction * (-f.x);
            if (ft <= bound + 1e-8)
                c.kT = min(c.kT + p.betaLin * 0.1, p.kMaxT);
            // Stick only when friction force is inside the cone (three-avbd updateDual).
            float slip = length(float2(C.y, C.z));
            c._pad0 = (ft <= bound + 1e-8 && slip < 1.0e-5) ? 1.0 : 0.0;
            gContacts[curOff + i] = c;
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
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.bodyCount) return;
    AvbdParams p = gParams[0];
    BodyState b = gBodies[i];
    BodyState initB = gBodiesPrev[i];
    gCcdFlags[i] = 0;
    if (!bodyIsDynamic(b)) return;
    if (bodyIsSleeping(b)) return;

    float dt = max(p.dt, 1e-8);
    b.linearVel = (b.position - initB.position) / dt;
    // Angular velocity from quat delta
    float4 q0 = initB.orientation;
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
            gCcdFlags[i] = 1; // candidate — sleep_island_commit after island gate
    }
    else
    {
        b.sleepTimer = 0.0;
        b.flags |= BODY_AWAKE;
    }
    gBodies[i] = b;
}

// If any dynamic contact neighbor is not a sleep candidate, revoke both. Repeating
// this (host) propagates through the island so a stack sleeps together or not at all.
[shader("compute")]
[numthreads(64, 1, 1)]
void sleep_island_gate(uint3 tid : SV_DispatchThreadID)
{
    // One thread per manifold (same connectivity as contacts, ~2–4× fewer).
    uint i = tid.x;
    uint n = gCounters[0].manifoldCount;
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
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.bodyCount) return;
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    if (gCcdFlags[i] == 0) return;
    b.flags = (b.flags & ~BODY_AWAKE) | BODY_SLEEPING;
    b.linearVel = float3(0, 0, 0);
    b.angularVel = float3(0, 0, 0);
    InterlockedAdd(gCounters[0].sleepEvents, 1);
    gBodies[i] = b;
}

// --- CCD stretch: classify fast bodies (linear + angular rim) ---
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_classify(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.bodyCount) return;
    BodyState b = gBodies[i];
    gCcdFlags[i] = 0;
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    AvbdParams p = gParams[0];
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
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.colliderCount) return;
    Collider col = gColliders[i];
    if (gCcdFlags[col.body] == 0) return;
    BodyState b = gBodies[col.body];
    AvbdParams p = gParams[0];
    float3 v = b.linearVel;
    float maxIter = 8.0;
    // Against planes only in this stretch slice
    for (uint j = 0; j < c.colliderCount; j++)
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
            InterlockedOr(gCounters[0].overflowFlags, CCD_BUDGET_EXHAUSTED);
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
    uint body = tid.x;
    StepCounters c = gCounters[0];
    if (body >= c.bodyCount) return;
    BodyState b = gBodies[body];
    if (!bodyIsDynamic(b)) return;
    float mass = 0.0;
    float3 com = float3(0, 0, 0);
    // Pass 1: mass / COM in body frame
    for (uint i = 0; i < c.colliderCount; i++)
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
    for (uint i = 0; i < c.colliderCount; i++)
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
