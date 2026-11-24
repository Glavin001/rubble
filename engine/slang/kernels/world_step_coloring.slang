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

// count_color_tiles and scatter_color_tiles reading a body's record only when its awake
// bit is set (both act only on listed bodies; the rest of each is as theirs).
[shader("compute")]
[numthreads(64, 1, 1)]
void count_color_tiles_bits(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint body = dispatchIndex(tid);
    uint local = lid.x;
    uint palette = min(HOST_S(4), 64u);
    uint n = gCounters[0].bodyCount;
    uint groups = (n + 63u) / 64u;
    ASTORE(gsPackHistogram[local], 0u);
    uint color = 0xFFFFFFFFu;
    if (body < n && awakeBit(body))
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
void scatter_color_tiles_bits(uint3 tid : SV_DispatchThreadID)
{
    uint body = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (body >= n || !awakeBit(body)) return;
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
        ((shapeIsRound(a.shape) || shapeIsRound(b.shape)) ? SOLVER_SPHERE : 0u);
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
// Crash-island substeps (soft_step.slang): a particle part way through the
// step carries the fraction f of the step its iterate stands for, as −f in
// initialPad (no particle uses that word for anything else during the solve;
// every other value reads as the whole step).
float bodyTimeFrac(BodyState b)
{
    return ((b.flags & BODY_PARTICLE) != 0u && b.initialPad < 0.0) ? -b.initialPad : 1.0;
}

// `o` as it stands at fraction f of the step, from its step-start pose and its
// iterate (which stands for its own fraction): a rigid core read by a particle
// mid-step is taken part way, a particle read by the core is carried on to the
// step's end. Unchanged when the fractions match (every row without islands).
BodyState bodyAtTime(BodyState o, float f)
{
    float fo = bodyTimeFrac(o);
    if (fo == f) return o;
    float r = f / fo;
    o.position = o.initialPos + (o.position - o.initialPos) * r;
    float4 q = o.orientation;
    if (dot(q, o.initialRot) < 0.0) q = -q;
    o.orientation = quatNormalize(o.initialRot + (q - o.initialRot) * r);
    return o;
}

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
            bool anySphere = shapeIsRound(colA.shape) || shapeIsRound(colB.shape);
#endif
            bool isA = ba == bodyIdx;
            // Both bodies at this body's time (crash-island substeps).
            float fSelf = bodyTimeFrac(b);
            BodyState bA = bodyAtTime(gBodies[ba], fSelf);
            BodyState bB = bodyAtTime(gBodies[bb], fSelf);
            // Track belt: the pair's tangential target this step (zero off belts).
            float3 belt = float3(0, 0, 0);
#if defined(ENGINE_COMPACT_SOLVE)
            belt = solverPairBelt(pair);
#else
            belt = pairBeltDisp(colA, colB, bA, bB, mf.normal);
#endif
            // Sphere lever arms use step-start rotation; static poses stay current.
            float4 rotA = (anySphere && (bodyIsDynamic(bA) || kinematicMoving(bA))) ? bA.initialRot : bA.orientation;
            float4 rotB = (anySphere && (bodyIsDynamic(bB) || kinematicMoving(bB))) ? bB.initialRot : bB.orientation;
            float4 rotSelf = isA ? rotA : rotB;
            // Displacement from x⁻ stored on the body (predict). Static/sleeping
            // stay at 0; a kinematic carries its given motion v·h.
            bool moveA = bodyMovingOrKinematic(bA);
            bool moveB = bodyMovingOrKinematic(bB);
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
            if ((jflags & (JOINT_FLAG_BROKEN | JOINT_FLAG_IDLE)) != 0u) continue;
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
                float fSelf = bodyTimeFrac(b);
                jointContributeBall(A, rhs, jb, bodyAtTime(gBodies[jb.bodyA], fSelf), bodyAtTime(gBodies[jb.bodyB], fSelf),
                                    jb.bodyA == bodyIdx, p.alpha, p.kStart);
            }
            else
            {
                Joint j = gJoints[id];
                float fSelf = bodyTimeFrac(b);
                jointContributePrimal(A, rhs, j, bodyAtTime(gBodies[j.bodyA], fSelf), bodyAtTime(gBodies[j.bodyB], fSelf),
                                      j.bodyA == bodyIdx, p.alpha, p.dt, p.kStart,
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

