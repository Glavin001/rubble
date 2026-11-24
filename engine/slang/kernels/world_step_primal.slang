void primalApply(uint bodyIdx, Mat6 A, Vec6 rhs)
{
    BodyState b = gBodies[bodyIdx];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b) || (b.flags & BODY_FOLLOW) != 0u) return;
    AvbdParams p = PARAMS;
    uint floors = 0;
    Vec6 neg = rhs;
    for (int k = 0; k < 6; k++) neg.v[k] = -rhs.v[k];
    Vec6 dz = solve6x6(A, neg, p.pivotFloor, floors);
    if (floors > 0) AADD2(gCounters[0].pivotFloorHits, floors);
    float3 dlin = float3(dz.v[0], dz.v[1], dz.v[2]);
    float3 dang = float3(dz.v[3], dz.v[4], dz.v[5]);
    dlin = ccdTrustStep(bodyIdx, b, dlin);
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

// Bodies per span of the sparse tail: 2^15 chunks of 8.
static const uint SPARSE_SPAN_BODIES = 8u << 15;

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
        // Spans keep each loop under the WGSL loop guard (WGSL_LOOP_CAP
        // iterations), so a colour of any size finishes here.
        for (uint span = 0u; span < count; span += SPARSE_SPAN_BODIES)
        for (uint base = span; base < min(count, span + SPARSE_SPAN_BODIES); base += 8u)
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

// The colour dispatches' headroom and the sparse tail in one workgroup (World's plan,
// pass value TAIL_PASS: first colour p in bits 0..7, the stabilization flag 256, and
// the first sparse colour q in bits 16..23). Colours [p, q) are solved as
// solve_primal_color solves them: the same lanes per body (primalLanesForDegree of the
// colour's count and degree), the same accumulation and the same halving reduction
// within each body's lanes, one batch of 64 / lanes bodies at a time; colours from q as
// solve_primal_sparse solves them. A body's update depends only on its lane count and
// the poses of earlier colours, so each body gets what its own dispatch gave it, without
// a dispatch (and a barrier) for each headroom colour, which is usually empty.
[shader("compute")]
[numthreads(64, 1, 1)]
void solve_primal_sparse_tail(uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    uint maxB = HOST_S(1);
    if (local == 0u) gsUniformLanes = min(ALOAD(gCounters[0].colorCount), min(HOST_S(4), COLOR_DEGREE_BASE));
    GROUP_SYNC();
    uint colorLimit = gsUniformLanes;
    GROUP_SYNC();
    gStabilize = (PASS_U & 256u) != 0u;
    uint sparseFrom = (PASS_U >> 16) & 255u;
    for (uint color = PASS_U & 255u; color < colorLimit; color++)
    {
        if (local == 0u) gsUniformLanes = SOLVE_COUNT(color);
        GROUP_SYNC();
        uint count = gsUniformLanes;
        GROUP_SYNC();
        uint lanes = 8u;
        if (color < sparseFrom)
        {
            if (local == 0u)
                gsUniformLanes = primalLanesForDegree(
                    count, HOST_S(29) != 0u ? SOLVE_COUNT(COLOR_DEGREE_BASE + color) : 0u);
            GROUP_SYNC();
            lanes = gsUniformLanes;
            GROUP_SYNC();
        }
        uint perBatch = 64u / lanes;
        uint lane = local % lanes;
        uint slot = local / lanes;
        uint spanBodies = perBatch << 15;
        for (uint span = 0u; span < count; span += spanBodies)
        for (uint base = span; base < min(count, span + spanBodies); base += perBatch)
        {
            uint bodySlot = base + slot;
            bool live = bodySlot < count;
            uint bodyIdx = 0u;
            Mat6 A = mat6Zero();
            Vec6 rhs = vec6Zero();
            if (live)
            {
                bodyIdx = gColorLists[color * maxB + bodySlot];
                primalAccumulate(bodyIdx, lane, lanes, A, rhs);
            }
            if (lanes == 1u)
            {
                if (live) primalApply(bodyIdx, A, rhs);
            }
            else
            {
                if (lanes >= 64u) primalReduceStep(32u, live, lane, slot, A, rhs);
                if (lanes >= 32u) primalReduceStep(16u, live, lane, slot, A, rhs);
                if (lanes >= 16u) primalReduceStep(8u, live, lane, slot, A, rhs);
                if (lanes >= 8u) primalReduceStep(4u, live, lane, slot, A, rhs);
                if (lanes >= 4u) primalReduceStep(2u, live, lane, slot, A, rhs);
                primalReduceStep(1u, live, lane, slot, A, rhs);
                if (live && lane == 0u) primalApply(bodyIdx, A, rhs);
            }
            GROUP_SYNC();
        }
        // Graph colors are Gauss-Seidel: publish every pose before the next color.
        if (count > 0u) DeviceMemoryBarrierWithGroupSync();
    }
}

// Every solver iteration of the step in one workgroup (World's plan, while few bodies are
// awake): pass value LOOP_PASS, bits 0..7 the first colour of the sparse tail q, bits
// 8..15 the iterations. Each iteration solves the colours in order, colours below q as
// solve_primal_color solves them (the same lanes per body, accumulation and halving
// reduction) and the rest as solve_primal_sparse does (8 lanes), then every contact's
// and joint's dual as solve_dual (solveDualAt), the last iteration flagged as its
// dispatch flags it. A body's and a contact's update depend only on their lanes and the
// state the previous colour or phase published (a storage barrier between them), so the
// bodies and multipliers come out as the dispatch per colour and per dual gives them.
[shader("compute")]
[numthreads(64, 1, 1)]
void solve_primal_sparse_loop(uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    uint maxB = HOST_S(1);
    uint sparseFrom = PASS_U & 255u;
    uint iterations = (PASS_U >> 8) & 255u;
    gStabilize = false;
    for (uint iter = 0u; iter < iterations; iter++)
    {
        if (local == 0u) gsUniformLanes = min(ALOAD(gCounters[0].colorCount), min(HOST_S(4), COLOR_DEGREE_BASE));
        GROUP_SYNC();
        uint colorLimit = gsUniformLanes;
        GROUP_SYNC();
        for (uint color = 0u; color < colorLimit; color++)
        {
            if (local == 0u) gsUniformLanes = SOLVE_COUNT(color);
            GROUP_SYNC();
            uint count = gsUniformLanes;
            GROUP_SYNC();
            uint lanes = 8u;
            if (color < sparseFrom)
            {
                if (local == 0u)
                    gsUniformLanes = primalLanesForDegree(
                        count, HOST_S(29) != 0u ? SOLVE_COUNT(COLOR_DEGREE_BASE + color) : 0u);
                GROUP_SYNC();
                lanes = gsUniformLanes;
                GROUP_SYNC();
            }
            uint perBatch = 64u / lanes;
            uint lane = local % lanes;
            uint slot = local / lanes;
            uint spanBodies = perBatch << 15;
            for (uint span = 0u; span < count; span += spanBodies)
            for (uint base = span; base < min(count, span + spanBodies); base += perBatch)
            {
                uint bodySlot = base + slot;
                bool live = bodySlot < count;
                uint bodyIdx = 0u;
                Mat6 A = mat6Zero();
                Vec6 rhs = vec6Zero();
                if (live)
                {
                    bodyIdx = gColorLists[color * maxB + bodySlot];
                    primalAccumulate(bodyIdx, lane, lanes, A, rhs);
                }
                if (lanes == 1u)
                {
                    if (live) primalApply(bodyIdx, A, rhs);
                }
                else
                {
                    if (lanes >= 64u) primalReduceStep(32u, live, lane, slot, A, rhs);
                    if (lanes >= 32u) primalReduceStep(16u, live, lane, slot, A, rhs);
                    if (lanes >= 16u) primalReduceStep(8u, live, lane, slot, A, rhs);
                    if (lanes >= 8u) primalReduceStep(4u, live, lane, slot, A, rhs);
                    if (lanes >= 4u) primalReduceStep(2u, live, lane, slot, A, rhs);
                    primalReduceStep(1u, live, lane, slot, A, rhs);
                    if (live && lane == 0u) primalApply(bodyIdx, A, rhs);
                }
                GROUP_SYNC();
            }
            if (count > 0u) DeviceMemoryBarrierWithGroupSync();
        }
        // The dual over max(contacts, joints), as solve_dual's dispatch (slot 65) covers.
        if (local == 0u) gsUniformLanes = max(min(ALOAD(gCounters[0].contactCount), HOST_S(3)), gCounters[0].jointCount);
        GROUP_SYNC();
        // Every thread of that dispatch (rounded up to its workgroups) runs solveDualAt.
        uint duals = (gsUniformLanes + 63u) / 64u * 64u;
        GROUP_SYNC();
        bool lastIter = iter + 1u == iterations;
        for (uint k = local; k < duals; k += 64u) solveDualAt(k, lastIter);
        DeviceMemoryBarrierWithGroupSync();
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
    solveDualAt(dispatchIndex(tid), PASS_U != 0u);
}

// Contact i's and joint i's dual update; `lastIter`: the step's final iteration.
void solveDualAt(uint i, bool lastIter)
{
    // Per-contact dual (Metal occupancy). CSR/primal still use manifolds (Stage 8b).
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
            bool anySphere = shapeIsRound(colA.shape) || shapeIsRound(colB.shape);
#endif
            float3 belt = float3(0, 0, 0);
#if defined(ENGINE_COMPACT_SOLVE)
            belt = solverPairBelt(pair);
#else
            belt = pairBeltDisp(colA, colB, bA, bB, c.normal);
#endif
            float4 rotA = (anySphere && (bodyIsDynamic(bA) || kinematicMoving(bA))) ? bA.initialRot : bA.orientation;
            float4 rotB = (anySphere && (bodyIsDynamic(bB) || kinematicMoving(bB))) ? bB.initialRot : bB.orientation;
            // three-avbd evalContact: Ĉ = c0*(1-α) + J·(x-x⁻), not absolute geometric C.
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
            if (lastIter)
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
        bool live = (jflags & (JOINT_FLAG_BROKEN | JOINT_FLAG_IDLE | JOINT_FLAG_TYRE)) == 0u
            && (bodyMoving(gBodies[gJoints[i].bodyA]) || bodyMoving(gBodies[gJoints[i].bodyB]));
        // On the final iteration (PASS_U != 0) λ is this step's converged
        // constraint force: breaking is decided there (bonds never while a host
        // preloads a structure), and a joint whose multipliers still move is
        // marked unsettled so its bodies cannot fall asleep on it.
        bool last = lastIter;
        float4 lamLo0 = gJoints[i].lambdaLo;
        float4 lamHi0 = gJoints[i].lambdaHi;
        bool broke = false;
        float force = 0.0, torque = 0.0;
        if (live && jointHeaderIsBall(jflags, gJoints[i].motion))
        {
            JointBall jb = loadJointBall(i);
            // Multipliers at the step's end (a crash-island particle carried on).
            jointDualBall(jb, bodyAtTime(gBodies[jb.bodyA], 1.0), bodyAtTime(gBodies[jb.bodyB], 1.0),
                          p.alpha, p.kStart, p.kMaxN, p.betaLin);
            force = length(jb.lambda);
            broke = last && force > jb.breakForce;
            gJoints[i].lambdaLo.xyz = broke ? float3(0.0) : jb.lambda;
            gJoints[i].penaltyLo.xyz = broke ? float3(0.0) : jb.penalty;
        }
        else if (live)
        {
            Joint j = gJoints[i];
            BodyState a = bodyAtTime(gBodies[j.bodyA], 1.0);
            BodyState b = bodyAtTime(gBodies[j.bodyB], 1.0);
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
    float3 clampedVel;
    if (ccdClampedVelocity(i, b, clampedVel)) b.linearVel = clampedVel;

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
    if ((b.flags & BODY_CCD_FAST) != 0u) b.flags = ccdFastLatch(b.flags, b.linearVel, dt);
    gBodies[i].linearVel = b.linearVel;
    gBodies[i].angularVel = b.angularVel;
    gBodies[i].sleepTimer = b.sleepTimer;
    gBodies[i].flags = b.flags;
}

