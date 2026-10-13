// Production phase entry points — one Slang source for Metal / SPIR-V / CUDA.
#include "../physics/common.slang"

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

static const uint EVENT_SLEEP = 1u;
static const uint EVENT_WAKE = 2u;
static const uint EVENT_BREAK = 3u;
static const uint EVENT_OVERFLOW = 4u;

bool bodyIsDynamic(BodyState b) { return b.invMass > 0.0 && (b.flags & BODY_DYNAMIC) != 0u; }
bool bodyIsSleeping(BodyState b) { return (b.flags & BODY_SLEEPING) != 0u; }

RWStructuredBuffer<uint> gScratchAtomics;

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
        // Invalidate contacts involving this collider
        uint nContacts = gCounters[0].contactCount;
        for (uint ci = 0; ci < nContacts; ci++)
        {
            Contact ct = gContacts[ci];
            if (ct.colliderA == cmd.a || ct.colliderB == cmd.a)
            {
                ct.alive = 0;
                gContacts[ci] = ct;
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void predict_bodies(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.bodyCount) return;
    AvbdParams p = gParams[0];
    BodyState b = gBodies[i];
    gBodiesPrev[i] = b;
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;

    float dt = p.dt;
    float3 accel = (b.linearVel - float3(0, 0, 0)) / max(dt, 1e-8); // prev vel stored unused; adaptive weight
    // Adaptive warmstart weight from official demo
    float accelExt = accel.y * sign(p.gravityY);
    float accelWeight = clamp(accelExt / max(abs(p.gravityY), 1e-6), 0.0, 1.0);

    float3 inertialLin = b.position + b.linearVel * dt + float3(0, p.gravityY, 0) * (dt * dt);
    float4 inertialAng = quatNormalize(quatApplyDelta(b.orientation, b.angularVel * dt));

    // Store inertial in linearVel/angularVel temp? Use event buffer unused lanes — instead pack into prev
    // We keep inertial in gBodiesPrev velocities as stash:
    BodyState stash = b;
    stash.linearVel = inertialLin;
    stash.angularVel = inertialAng.xyz; // incomplete — use contacts unused
    // Better: write warmstarted pose into body; stash initial in prev
    BodyState initB = b;
    gBodiesPrev[i] = initB;

    b.position = b.position + b.linearVel * dt + float3(0, p.gravityY, 0) * (accelWeight * dt * dt);
    b.orientation = quatNormalize(quatApplyDelta(b.orientation, b.angularVel * dt));
    // Stash inertial target into sleepTimer unused? No — use hull or scratch.
    // Convention: gAabbs[i].mn = inertialLin, gAabbs[i].mx.x = inertialAng components via pad
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
}

static const uint GRID_HASH_SIZE = 32768u;

uint gridHash(int x, int y, int z)
{
    uint h = (uint)x * 73856093u ^ (uint)y * 19349663u ^ (uint)z * 83492791u;
    return h & (GRID_HASH_SIZE - 1u);
}

bool bodiesJointed(uint bodyA, uint bodyB)
{
    uint nJ = gCounters[0].jointCount;
    for (uint ji = 0; ji < nJ; ji++)
    {
        Joint jj = gJoints[ji];
        if ((jj.bodyA == bodyA && jj.bodyB == bodyB) ||
            (jj.bodyA == bodyB && jj.bodyB == bodyA))
            return true;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void broadphase_insert(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    StepCounters c = gCounters[0];
    if (i >= c.colliderCount) return;
    Collider col = gColliders[i];
    if (col.shape == SHAPE_PLANE) return;
    BodyState b = gBodies[col.body];
    if (bodyIsSleeping(b) && bodyIsDynamic(b)) return;
    uint base = gScratchAtomics[1];
    float cellSize = max(gParams[0].contactOffset * 16.0, 0.5);
    Aabb box = gAabbs[base + i];
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
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    if (col.shape == SHAPE_PLANE) return;
    uint base = gScratchAtomics[1];
    Aabb box = gAabbs[base + i];

    // Small scenes: exact pairwise (correctness). Large: uniform-grid hash.
    // Dynamic i must test all j (including static/planes with j < i).
    if (c.colliderCount <= 512u)
    {
        for (uint j = 0; j < c.colliderCount; j++)
        {
            if (j == i) continue;
            Collider other = gColliders[j];
            if (other.body == col.body) continue;
            // Emit each unordered pair once from the lower collider index.
            if (j < i && other.shape != SHAPE_PLANE) continue;
            BodyState ob = gBodies[other.body];
            if (bodyIsSleeping(ob) && bodyIsSleeping(b)) continue;
            if (!bodyIsDynamic(ob) && other.shape != SHAPE_PLANE) continue;
            if (bodiesJointed(col.body, other.body)) continue;
            if (other.shape != SHAPE_PLANE)
            {
                Aabb ab = gAabbs[base + j];
                if (box.mx.x < ab.mn.x || ab.mx.x < box.mn.x) continue;
                if (box.mx.y < ab.mn.y || ab.mx.y < box.mn.y) continue;
                if (box.mx.z < ab.mn.z || ab.mx.z < box.mn.z) continue;
            }
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
            if (cur > i)
            {
                Collider other = gColliders[cur];
                if (other.body != col.body &&
                    !bodiesJointed(col.body, other.body) &&
                    other.shape != SHAPE_PLANE)
                {
                    BodyState ob = gBodies[other.body];
                    if (!bodyIsSleeping(ob))
                    {
                        Aabb ab = gAabbs[base + cur];
                        if (!(box.mx.x < ab.mn.x || ab.mx.x < box.mn.x ||
                              box.mx.y < ab.mn.y || ab.mx.y < box.mn.y ||
                              box.mx.z < ab.mn.z || ab.mx.z < box.mn.z))
                            emitCandidate(i, cur);
                    }
                }
            }
            cur = gColorLists[cur];
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
    float margin = gParams[0].contactOffset;
    float3 nrm; float pen; float3 pA; float3 pB;
    bool hit = false;

    if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_SPHERE)
    {
        hit = sphereSphere(posA, ca.params.x, posB, cb.params.x, margin, nrm, pen, pA, pB);
    }
    else if (ca.shape == SHAPE_SPHERE && cb.shape == SHAPE_PLANE)
    {
        float3 pn = normalize(cb.params.xyz);
        float d = dot(posA, pn) - cb.params.w - ca.params.x - margin;
        if (d < 0.0)
        {
            hit = true; nrm = -pn; pen = -d;
            pA = posA + nrm * ca.params.x;
            pB = posA + nrm * (ca.params.x - pen);
        }
    }
    else if (ca.shape == SHAPE_PLANE && cb.shape == SHAPE_SPHERE)
    {
        float3 pn = normalize(ca.params.xyz);
        float d = dot(posB, pn) - ca.params.w - cb.params.x - margin;
        if (d < 0.0)
        {
            hit = true; nrm = pn; pen = -d;
            pB = posB - nrm * cb.params.x;
            pA = posB - nrm * (cb.params.x - pen);
        }
    }
    else if (ca.shape == SHAPE_BOX && cb.shape == SHAPE_SPHERE)
    {
        // Sphere vs OBB: closest point
        float4 rotA = quatMul(ba.orientation, ca.localRot);
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
        float4 rotB = quatMul(bb.orientation, cb.localRot);
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
        // Reject non-adjacent capsule pairs that only overlap via oversized bounds.
        // Adjacent rope segments are already filtered by bodiesJointed in broadphase.
        float rA = ca.params.x;
        float rB = cb.params.x;
        float4 rotA = quatMul(ba.orientation, ca.localRot);
        float4 rotB = quatMul(bb.orientation, cb.localRot);
        float3 axA = quatRotate(rotA, float3(0, 1, 0));
        float3 axB = quatRotate(rotB, float3(0, 1, 0));
        float3 a0 = posA - axA * ca.params.y;
        float3 a1 = posA + axA * ca.params.y;
        float3 b0 = posB - axB * cb.params.y;
        float3 b1 = posB + axB * cb.params.y;
        // Approximate: distance between segment midpoints minus half-lengths.
        float3 midA = 0.5 * (a0 + a1);
        float3 midB = 0.5 * (b0 + b1);
        float sep = length(midB - midA) - ca.params.y - cb.params.y;
        if (sep < rA + rB + margin)
            hit = sphereSphere(midA, rA + ca.params.y * 0.25, midB, rB + cb.params.y * 0.25, margin, nrm, pen, pA, pB);
    }
    else if (ca.shape == SHAPE_CAPSULE && cb.shape == SHAPE_SPHERE)
    {
        float4 rotA = quatMul(ba.orientation, ca.localRot);
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
        float4 rotB = quatMul(bb.orientation, cb.localRot);
        float3 axis = quatRotate(rotB, float3(0, 1, 0));
        float half = cb.params.y;
        float3 pa = posB - axis * half;
        float3 pb = posB + axis * half;
        float3 ab = pb - pa;
        float t = clamp(dot(posA - pa, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
        float3 closest = pa + ab * t;
        hit = sphereSphere(posA, ca.params.x, closest, cb.params.x, margin, nrm, pen, pA, pB);
    }
    else if (ca.shape == SHAPE_BOX && cb.shape == SHAPE_PLANE)
    {
        float3 pn = normalize(cb.params.xyz);
        float4 rotA = quatMul(ba.orientation, ca.localRot);
        float3 half = ca.params.xyz;
        // Support point of box along -plane normal
        float3 localN = quatRotate(quatConjugate(rotA), -pn);
        float3 localSupport = float3(localN.x >= 0 ? half.x : -half.x,
                                     localN.y >= 0 ? half.y : -half.y,
                                     localN.z >= 0 ? half.z : -half.z);
        float3 support = transformPoint(posA, rotA, localSupport);
        float d = dot(support, pn) - cb.params.w - margin;
        if (d < 0.0)
        {
            hit = true; nrm = -pn; pen = -d;
            pA = support; pB = support - pn * d;
        }
    }
    else if (ca.shape == SHAPE_PLANE && cb.shape == SHAPE_BOX)
    {
        float3 pn = normalize(ca.params.xyz);
        float4 rotB = quatMul(bb.orientation, cb.localRot);
        float3 half = cb.params.xyz;
        float3 localN = quatRotate(quatConjugate(rotB), -pn);
        float3 localSupport = float3(localN.x >= 0 ? half.x : -half.x,
                                     localN.y >= 0 ? half.y : -half.y,
                                     localN.z >= 0 ? half.z : -half.z);
        float3 support = transformPoint(posB, rotB, localSupport);
        float d = dot(support, pn) - ca.params.w - margin;
        if (d < 0.0)
        {
            hit = true; nrm = pn; pen = -d;
            pB = support; pA = support - pn * d;
        }
    }
    else if (ca.shape == SHAPE_BOX && cb.shape == SHAPE_BOX)
    {
        // Separating-axis on world AABB of OBBs (conservative); refine with sphere of half extents.
        float rA = length(ca.params.xyz);
        float rB = length(cb.params.xyz);
        hit = sphereSphere(posA, rA, posB, rB, margin, nrm, pen, pA, pB);
        if (hit)
        {
            // Clip penetration to OBB face estimate along nrm
            float4 rotA = quatMul(ba.orientation, ca.localRot);
            float4 rotB = quatMul(bb.orientation, cb.localRot);
            float3 la = abs(quatRotate(quatConjugate(rotA), nrm)) * ca.params.xyz;
            float3 lb = abs(quatRotate(quatConjugate(rotB), nrm)) * cb.params.xyz;
            float extent = la.x + la.y + la.z + lb.x + lb.y + lb.z;
            float centerSep = dot(posB - posA, nrm);
            pen = extent + margin - abs(centerSep);
            if (pen <= 0.0) hit = false;
            else if (centerSep < 0.0) nrm = -nrm;
        }
    }
    else
    {
        // Fallback: sphere bound (convex hulls use params.w)
        float rA = ca.params.w > 0.0 ? ca.params.w : length(ca.params.xyz);
        float rB = cb.params.w > 0.0 ? cb.params.w : length(cb.params.xyz);
        hit = sphereSphere(posA, rA, posB, rB, margin, nrm, pen, pA, pB);
    }

    if (!hit) return;

    uint slot;
    InterlockedAdd(gCounters[0].contactCount, 1, slot);
    uint maxC = gScratchAtomics[3];
    if (slot >= maxC)
    {
        InterlockedOr(gCounters[0].overflowFlags, OVERFLOW_CONTACTS);
        return;
    }

    Material ma = gMaterials[ca.material];
    Material mb = gMaterials[cb.material];
    float mu = sqrt(max(ma.friction, 0.0) * max(mb.friction, 0.0));
    float3 t1, t2;
    orthonormalBasis(nrm, t1, t2);

    Contact ct;
    ct.colliderA = pair.colliderA;
    ct.colliderB = pair.colliderB;
    ct.featureId = pair.key;
    ct.alive = 1;
    ct.normal = nrm;
    ct.penetration = pen;
    ct.anchorA = quatRotate(quatConjugate(ba.orientation), pA - ba.position);
    ct.c0N = pen;
    ct.anchorB = quatRotate(quatConjugate(bb.orientation), pB - bb.position);
    ct.friction = mu;
    ct.tangent1 = t1;
    ct.lambdaN = 0.0;
    ct.tangent2 = t2;
    ct.lambdaT1 = 0.0;
    ct.kN = gParams[0].kStart;
    ct.kT = gParams[0].kStart;
    ct.lambdaT2 = 0.0;
    ct.c0T1 = 0.0;
    gContacts[slot] = ct;

    // Wake both
    if (bodyIsDynamic(ba))
    {
        ba.flags = (ba.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        gBodies[ca.body] = ba;
    }
    if (bodyIsDynamic(bb))
    {
        bb.flags = (bb.flags & ~BODY_SLEEPING) | BODY_AWAKE;
        gBodies[cb.body] = bb;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void persist_contacts(uint3 tid : SV_DispatchThreadID)
{
    // Warm-start match against previous contacts in gContacts[old] — host double-buffers
    // For first full step: apply gamma/alpha to newly written contacts using feature id hash map in-place.
    uint i = tid.x;
    uint n = gCounters[0].contactCount;
    if (i >= n) return;
    AvbdParams p = gParams[0];
    Contact c = gContacts[i];
    if (c.alive == 0) return;
    // Decay warm start for new contacts already zero; when matched, host/kernel sets lambdas.
    c.kN = max(p.kStart, p.gamma * c.kN);
    c.kT = max(p.kStart, p.gamma * c.kT);
    c.lambdaN *= p.alpha * p.gamma;
    c.lambdaT1 *= p.alpha * p.gamma;
    c.lambdaT2 *= p.alpha * p.gamma;
    gContacts[i] = c;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void color_bodies(uint3 tid : SV_DispatchThreadID)
{
    // Jones-Plassmann style on awake bodies using contact graph (persistent body ids).
    uint li = tid.x;
    uint awakeN = gCounters[0].awakeCount;
    if (li >= awakeN) return;
    uint body = gAwakeList[li];
    uint round = gScratchAtomics[4];
    uint prio = body * 0x9e3779b9u + round * 0x85ebca6bu;
    prio ^= prio >> 16;

    bool blocked = false;
    uint nContacts = gCounters[0].contactCount;
    for (uint ci = 0; ci < nContacts; ci++)
    {
        Contact c = gContacts[ci];
        if (c.alive == 0) continue;
        uint ba = gColliders[c.colliderA].body;
        uint bb = gColliders[c.colliderB].body;
        uint other = 0xFFFFFFFFu;
        if (ba == body) other = bb;
        else if (bb == body) other = ba;
        else continue;
        if (!bodyIsDynamic(gBodies[other]) || bodyIsSleeping(gBodies[other])) continue;
        if (gBodyColor[other] != 0xFFFFFFFFu) continue; // already colored
        uint op = other * 0x9e3779b9u + round * 0x85ebca6bu;
        op ^= op >> 16;
        if (op > prio || (op == prio && other > body)) { blocked = true; break; }
    }
    // Joint edges
    uint nJoints = gCounters[0].jointCount;
    for (uint ji = 0; ji < nJoints && !blocked; ji++)
    {
        Joint j = gJoints[ji];
        uint other = j.bodyA == body ? j.bodyB : (j.bodyB == body ? j.bodyA : 0xFFFFFFFFu);
        if (other == 0xFFFFFFFFu) continue;
        if (!bodyIsDynamic(gBodies[other]) || bodyIsSleeping(gBodies[other])) continue;
        if (gBodyColor[other] != 0xFFFFFFFFu) continue;
        uint op = other * 0x9e3779b9u + round * 0x85ebca6bu;
        op ^= op >> 16;
        if (op > prio || (op == prio && other > body)) blocked = true;
    }

    if (!blocked && gBodyColor[body] == 0xFFFFFFFFu)
    {
        gBodyColor[body] = round;
        InterlockedMax(gCounters[0].colorCount, round + 1);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init_colors(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= gCounters[0].bodyCount) return;
    gBodyColor[i] = 0xFFFFFFFFu;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_primal_color(uint3 tid : SV_DispatchThreadID)
{
    uint color = gScratchAtomics[5];
    uint count = gColorCounts[color];
    uint li = tid.x;
    if (li >= count) return;
    uint maxB = gScratchAtomics[1];
    uint bodyIdx = gColorLists[color * maxB + li];
    BodyState b = gBodies[bodyIdx];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b)) return;
    AvbdParams p = gParams[0];
    float dt = p.dt;
    float dt2 = dt * dt;

    float3 inertial = gAabbs[bodyIdx].mn;
    // Build A and rhs. Official AVBD uses mass/inertia (not inverse) in M/h².
    Mat6 A = mat6Zero();
    Vec6 rhs = vec6Zero();
    float mass = b.invMass > 1e-12 ? (1.0 / b.invMass) : 0.0;
    float3 inertia = float3(
        b.invInertia.x > 1e-12 ? (1.0 / b.invInertia.x) : 0.0,
        b.invInertia.y > 1e-12 ? (1.0 / b.invInertia.y) : 0.0,
        b.invInertia.z > 1e-12 ? (1.0 / b.invInertia.z) : 0.0);
    mat6AddDiag(A, float3(mass, mass, mass) / dt2, inertia / dt2);
    // rhs = M/h² (z - y)
    float3 dp = b.position - inertial;
    rhs.v[0] = (mass / dt2) * dp.x;
    rhs.v[1] = (mass / dt2) * dp.y;
    rhs.v[2] = (mass / dt2) * dp.z;
    // Angular chart vs inertial orientation (stashed in AABB mx/_pad0 during predict).
    Aabb stash = gAabbs[bodyIdx];
    float4 inertialAng = quatNormalize(float4(stash.mx.x, stash.mx.y, stash.mx.z, stash._pad0));
    float4 q = b.orientation;
    if (dot(q, inertialAng) < 0.0) inertialAng = -inertialAng;
    // Small-angle rotation taking inertial → current: 2 * Im(conj(qi)*q)
    float4 dq = quatMul(quatConjugate(inertialAng), q);
    float3 dangTarget = 2.0 * dq.xyz;
    if (dq.w < 0.0) dangTarget = -dangTarget;
    // Clamp chart residual so a bad stash cannot explode the 6×6.
    float dangLen = length(dangTarget);
    if (dangLen > 0.5) dangTarget *= 0.5 / dangLen;
    rhs.v[3] = (inertia.x / dt2) * dangTarget.x;
    rhs.v[4] = (inertia.y / dt2) * dangTarget.y;
    rhs.v[5] = (inertia.z / dt2) * dangTarget.z;

    uint nContacts = gCounters[0].contactCount;
    for (uint ci = 0; ci < nContacts; ci++)
    {
        Contact c = gContacts[ci];
        if (c.alive == 0) continue;
        uint ba = gColliders[c.colliderA].body;
        uint bb = gColliders[c.colliderB].body;
        if (ba != bodyIdx && bb != bodyIdx) continue;
        bool isA = ba == bodyIdx;
        // Refresh penetration
        BodyState bo = gBodies[isA ? bb : ba];
        Collider ca = gColliders[c.colliderA];
        Collider cb = gColliders[c.colliderB];
        float3 wa = transformPoint(gBodies[ba].position, gBodies[ba].orientation, c.anchorA);
        float3 wb = transformPoint(gBodies[bb].position, gBodies[bb].orientation, c.anchorB);
        c.penetration = dot(c.normal, wb - wa);
        contributeContact(A, rhs, c, isA, p.alpha, b.position, b.orientation);
    }

    uint nJoints = gCounters[0].jointCount;
    for (uint ji = 0; ji < nJoints; ji++)
    {
        Joint j = gJoints[ji];
        if (j.bodyA != bodyIdx && j.bodyB != bodyIdx) continue;
        bool isA = j.bodyA == bodyIdx;
        BodyState ba = gBodies[j.bodyA];
        BodyState bb = gBodies[j.bodyB];
        float3 wa = transformPoint(ba.position, ba.orientation, j.localAnchorA);
        float3 wb = transformPoint(bb.position, bb.orientation, j.localAnchorB);
        float3 C = wa - wb; // shared residual
        float3 local = isA ? j.localAnchorA : j.localAnchorB;
        float3 lam = float3(j.lambda0, j.lambda1, j.lambda2);
        float k = max(j.stiffness, p.kStart);
        contributeBallJoint(A, rhs, b.position, b.orientation, local, C, k, lam, isA, p.alpha);
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
            // Lock two axes perpendicular to slide axis (simplified).
            float3 axis = normalize(j.axisA);
            float3 mid = 0.5 * (wa + wb);
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
            mid = mid; // silence unused in some targets
        }
    }

    uint floors = 0;
    // A Δz = -rhs  => solve A x = -rhs
    Vec6 neg = rhs;
    for (int k = 0; k < 6; k++) neg.v[k] = -rhs.v[k];
    Vec6 dz = solve6x6(A, neg, p.pivotFloor, floors);
    if (floors > 0) InterlockedAdd(gCounters[0].pivotFloorHits, floors);

    // Trust region
    float3 dlin = float3(dz.v[0], dz.v[1], dz.v[2]);
    float3 dang = float3(dz.v[3], dz.v[4], dz.v[5]);
    float r = 0.5; // characteristic size fallback
    float maxLin = p.trustLin * r;
    float len = length(dlin);
    if (len > maxLin && len > 1e-12) dlin *= maxLin / len;
    float alen = length(dang);
    if (alen > p.trustAng && alen > 1e-12) dang *= p.trustAng / alen;

    // Portable finite check (isnan is not available on the WGSL path).
    if (dlin.x != dlin.x || dlin.y != dlin.y || dlin.z != dlin.z) return;
    if (dang.x != dang.x || dang.y != dang.y || dang.z != dang.z) return;
    b.position += dlin;
    b.orientation = quatApplyDelta(b.orientation, dang);
    gBodies[bodyIdx] = b;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void solve_dual(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    uint n = gCounters[0].contactCount;
    AvbdParams p = gParams[0];
    if (i < n)
    {
        Contact c = gContacts[i];
        if (c.alive != 0)
        {
            uint ba = gColliders[c.colliderA].body;
            uint bb = gColliders[c.colliderB].body;
            float3 wa = transformPoint(gBodies[ba].position, gBodies[ba].orientation, c.anchorA);
            float3 wb = transformPoint(gBodies[bb].position, gBodies[bb].orientation, c.anchorB);
            float pen = dot(c.normal, wb - wa);
            c.penetration = pen;
            float cReg = (1.0 - p.alpha) * c.c0N + (pen - c.c0N);
            float3 f;
            f.x = c.kN * cReg + c.lambdaN;
            f.y = c.kT * (1.0 - p.alpha) * c.c0T1 + c.lambdaT1;
            f.z = c.lambdaT2;
            f = projectContactForce(f, c.friction);
            c.lambdaN = f.x;
            c.lambdaT1 = f.y;
            c.lambdaT2 = f.z;
            // Penalty ramp
            c.kN = min(c.kN + p.betaLin * abs(cReg), p.kMaxN);
            c.kT = min(c.kT + p.betaLin * 0.1, p.kMaxT);
            gContacts[i] = c;
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
        float k = max(j.stiffness, p.kStart);
        float3 lam = float3(j.lambda0, j.lambda1, j.lambda2);
        float3 Creg = (1.0 - p.alpha) * C;
        lam = k * Creg + lam;
        j.lambda0 = lam.x; j.lambda1 = lam.y; j.lambda2 = lam.z;
        j.k0 = min(k + p.betaLin * length(Creg), p.kMaxN);
        j.k1 = j.k0; j.k2 = j.k0;
        gJoints[i] = j;
    }
}

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
        {
            b.flags = (b.flags & ~BODY_AWAKE) | BODY_SLEEPING;
            b.linearVel = float3(0, 0, 0);
            b.angularVel = float3(0, 0, 0);
            InterlockedAdd(gCounters[0].sleepEvents, 1);
        }
    }
    else
    {
        b.sleepTimer = 0.0;
        b.flags |= BODY_AWAKE;
    }
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
