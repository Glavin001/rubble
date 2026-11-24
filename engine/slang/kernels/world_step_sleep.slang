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
    // Its resting rows are archived by the archive_* passes that follow.
    if (sleepArchive() && !archiveCompact())
    {
        ASTORE(gScratchAtomics[arAge(i)], ARCHIVE_JUST_SLEPT);
    }
}

#if defined(ENGINE_SLEEP_ARCHIVE)
// Sleep archive passes, after sleep_island_commit (see archiveMeta).
// archive_free, over this step's awake list: a body awake for a second step lets its
// archive go (its pairs formed again by now); one that fell asleep again lets the old one
// go before archiving anew. Pushes only (archive_alloc pops in a later dispatch).
void archiveRelease(uint body)
{
    uint k = ALOAD(gScratchAtomics[arHead(body)]);
    uint n = ALOAD(gScratchAtomics[arCount(body)]);
    uint blocks = (n + ARCHIVE_ROWS - 1u) / ARCHIVE_ROWS;
    uint top;
    AADD3(gScratchAtomics[arTop()], blocks, top);
    for (uint j = 0u; j < blocks && j < 4096u; j++)
    {
        ASTORE(gScratchAtomics[arStack(top + j)], k);
        k = ALOAD(gScratchAtomics[arNext(k)]);
    }
    ASTORE(gScratchAtomics[arHead(body)], ARCHIVE_NONE);
    ASTORE(gScratchAtomics[arCount(body)], 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void archive_free(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sleepArchive() || i >= ALOAD(gCounters[0].awakeCount)) return;
    uint body = gAwakeList[i];
    if (ALOAD(gScratchAtomics[arHead(body)]) == ARCHIVE_NONE) return;
    uint age = ALOAD(gScratchAtomics[arAge(body)]);
    if (age == ARCHIVE_JUST_SLEPT)
    {
        archiveRelease(body);
        return;
    }
    if (bodyIsSleeping(gBodies[body])) return;
    if (age + 1u < 2u)
    {
        ASTORE(gScratchAtomics[arAge(body)], age + 1u);
        return;
    }
    archiveRelease(body);
    ASTORE(gScratchAtomics[arAge(body)], 0u);
}

// archive_count, over this step's contact rows: each resting row's owner (archiveOwner)
// counts it (in its cursor word).
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_count(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sleepArchive() || ALOAD(gCounters[0].sleepEvents) == 0u) return;
    if (i >= min(ALOAD(gCounters[0].contactCount), HOST_S(3))) return;
    if (CONTACT_STATE(HOST_S(11) + i) == 0u) return;
    Contact c = READ_CONTACT(HOST_S(11) + i);
    uint owner = archiveOwner(c);
    if (owner != ARCHIVE_NONE)
        AADD2(gScratchAtomics[arCursor(owner)], 1u);
}

// archive_alloc, over this step's awake list: each body that fell asleep with rows takes
// its blocks off the free stack, chained.
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_alloc(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sleepArchive() || ALOAD(gCounters[0].sleepEvents) == 0u) return;
    if (i >= ALOAD(gCounters[0].awakeCount)) return;
    uint body = gAwakeList[i];
    if (ALOAD(gScratchAtomics[arAge(body)]) != ARCHIVE_JUST_SLEPT) return;
    uint n = ALOAD(gScratchAtomics[arCursor(body)]);
    ASTORE(gScratchAtomics[arCursor(body)], 0u);
    if (n == 0u) return;
    uint blocks = (n + ARCHIVE_ROWS - 1u) / ARCHIVE_ROWS;
    uint top;
    AADD3(gScratchAtomics[arTop()], 0u - blocks, top);
    if (top < blocks || top > archiveBlocks())
    {
        AOR2(gCounters[0].overflowFlags, OVERFLOW_SLEEP_ARCHIVE);
        return;
    }
    uint first = top - blocks;
    for (uint j = 0u; j < blocks && j < 4096u; j++)
    {
        uint k = ALOAD(gScratchAtomics[arStack(first + j)]);
        uint next = j + 1u < blocks ? ALOAD(gScratchAtomics[arStack(first + j + 1u)]) : ARCHIVE_NONE;
        ASTORE(gScratchAtomics[arNext(k)], next);
    }
    ASTORE(gScratchAtomics[arHead(body)], ALOAD(gScratchAtomics[arStack(first)]));
    ASTORE(gScratchAtomics[arCount(body)], n);
}

// archive_scatter, over this step's contact rows: each resting row into its owner's
// blocks, tagged with its colliders' local poses (archiveKey).
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_scatter(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sleepArchive() || ALOAD(gCounters[0].sleepEvents) == 0u) return;
    if (i >= min(ALOAD(gCounters[0].contactCount), HOST_S(3))) return;
    if (CONTACT_STATE(HOST_S(11) + i) == 0u) return;
    Contact c = READ_CONTACT(HOST_S(11) + i);
    uint owner = archiveOwner(c);
    if (owner == ARCHIVE_NONE) return;
    uint k = ALOAD(gScratchAtomics[arHead(owner)]);
    if (k == ARCHIVE_NONE) return;
    uint slot;
    AADD3(gScratchAtomics[arCursor(owner)], 1u, slot);
    if (slot >= ALOAD(gScratchAtomics[arCount(owner)])) return;
    for (uint h = slot / ARCHIVE_ROWS; h > 0u && h < 4096u; h--)
        k = ALOAD(gScratchAtomics[arNext(k)]);
    c._pad1 = asfloat(archiveKey(gColliders[c.colliderA], gColliders[c.colliderB]));
    c.alive = 1u;
    STORE_NEW_CONTACT(archiveRow(k, slot % ARCHIVE_ROWS), 0u, c);
}

// archive_emit, after the broadphase (4,096 threads striding over this step's touched
// sleepers): their archived resting pairs join the candidates. It also empties the other
// step's touched count (the next step lists into it).
// archive_spread (measurement variant, scheduled before archive_emit in rounds): every
// listed touched sleeper lists its archived partners that are asleep too, one archived
// hop a round, so their resting pairs are emitted as well and a wake can chain through
// them within the step, as it could through carried resting pairs.
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_spread(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sleepArchive() || archiveCompact()) return;
    uint n = min(ALOAD(gScratchAtomics[arTouchedCount(HOST_S(17))]), HOST_S(1));
    for (uint k = i; k < n && k < 1048576u; k += 4096u)
    {
        uint body = ALOAD(gScratchAtomics[arTouchedList(k)]);
        uint m = min(ALOAD(gScratchAtomics[arCount(body)]), 2048u);
        for (uint slot = 0u; slot < m && slot < 2048u; slot++)
        {
            Contact r = archiveRowOf(body, slot);
            uint a = gColliders[r.colliderA].body;
            uint other = a == body ? gColliders[r.colliderB].body : a;
            BodyState ob = gBodies[other];
            if (bodyIsDynamic(ob) && bodyIsSleeping(ob))
                archiveMark(other);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void archive_emit(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sleepArchive()) return;
    uint step = HOST_S(17);
    if (i == 0u)
        ASTORE(gScratchAtomics[arTouchedCount(step + 1u)], 0u);
    // With the resting-pair carry on (EngineFeatures::carry_with_archive) the carried
    // pairs already include these: emitting them again doubled their contact rows.
    if (carryRestPairs()) return;
    uint n = min(ALOAD(gScratchAtomics[arTouchedCount(step)]), HOST_S(1));
    for (uint k = i; k < n && k < 1048576u; k += 4096u)
        archiveEmitResting(ALOAD(gScratchAtomics[arTouchedList(k)]));
}

// The touch query's passes (archive_* in rigid_plan order): snap, seed and spread (one
// hop), args, touch. Slot 83: a thread a listed sleeper; slot 84: a thread a collider,
// or none when nothing is listed.
//
// archive_touch_snap (one thread): this step's listed sleepers so far, for the spread.
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_touch_snap(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u || !sleepArchive()) return;
    uint n = min(ALOAD(gScratchAtomics[arTouchedCount(HOST_S(17))]), HOST_S(1));
    writeDispatchArgs(83u, n);
    writeDispatchArgs(84u, n > 0u ? gCounters[0].colliderCount : 0u);
}

// archive_touch_seed (slot 83): those sleepers spread this step (arSpread).
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_touch_seed(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    if (!sleepArchive()) return;
    if (k < min(ALOAD(gScratchAtomics[arTouchedCount(HOST_S(17))]), HOST_S(1)))
        ASTORE(gScratchAtomics[arSpread(ALOAD(gScratchAtomics[arTouchedList(k)]))], HOST_S(17) + 1u);
}

// archive_touch_spread (slot 84): the sleepers the seeded ones overlap are listed too. A
// touched sleeper usually wakes (wake on touch), and once awake it wakes the sleepers it
// rests on within the same step (its hull pairs' narrowphase runs after the first pass;
// a projectile or ccd_wake wakes it before the narrowphase), which are then solved with
// their whole contact set. One hop (sleepers listed here are not seeded): the carry's
// racier chains further in are not followed.
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_touch_spread(uint3 tid : SV_DispatchThreadID)
{
    uint c = dispatchIndex(tid);
    if (!sleepArchive() || c >= gCounters[0].colliderCount) return;
    if (ALOAD(gScratchAtomics[arSpread(gColliders[c].body)]) == HOST_S(17) + 1u)
        touchQueryCollider(c, true);
}

// archive_touch_args (one thread, after the spread): archive_touch's work (slot 84), and
// the next step's count word emptied (the next step lists into it).
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_touch_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u || !sleepArchive()) return;
    uint step = HOST_S(17);
    writeDispatchArgs(84u, ALOAD(gScratchAtomics[arTouchedCount(step)]) > 0u ? gCounters[0].colliderCount : 0u);
    ASTORE(gScratchAtomics[arTouchedCount(step + 1u)], 0u);
}

// archive_touch, in place of archive_emit with the touch query (slot 84): each listed
// sleeper's resting pairs.
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_touch(uint3 tid : SV_DispatchThreadID)
{
    uint c = dispatchIndex(tid);
    if (!sleepArchive() || c >= gCounters[0].colliderCount) return;
    if (archiveTouchedNow(gColliders[c].body))
        touchQueryCollider(c, false);
}

// archive_seal, over this step's awake list: the step's archiving is done.
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_seal(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sleepArchive() || i >= ALOAD(gCounters[0].awakeCount)) return;
    uint body = gAwakeList[i];
    if (ALOAD(gScratchAtomics[arAge(body)]) == ARCHIVE_JUST_SLEPT)
    {
        ASTORE(gScratchAtomics[arAge(body)], 0u);
        ASTORE(gScratchAtomics[arCursor(body)], 0u);
    }
}
#else
// Persistent rows off WGSL hosts (SPIR-V checks): no archive.
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_count(uint3 tid : SV_DispatchThreadID) {}
[shader("compute")]
[numthreads(64, 1, 1)]
void archive_scatter(uint3 tid : SV_DispatchThreadID) {}
#endif

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

// ----- contact events (P4) ----------------------------------------------------
// Opt-in pass (hosts add it while any shape reports contact events), right after
// contact persistence. One thread per manifold: a pair with a flagged collider
// whose deepest contact touches this step (gap within the contact offset, or
// closed by its approach within the step) is listed with
// that point, the A-to-B normal and the closing speed there from the step-start
// velocities, before the solve. Reads physics state only; the host diffs the
// set step to step into begin, end and hit events (engine-world contacts.rs).
static const uint COLLIDER_CONTACT_EVENTS = 0x1000u;
static const uint COLLIDER_SENSOR = 0x4000u;
// After the continuous-collision region (engine-schema CONTACT_EVENT_BASE).
static const uint CONTACT_EVENT_BASE = CCD_PAIR_BASE + 2u * CCD_PAIR_CAP;
static const uint CONTACT_EVENT_CAP = 2048u;
static const uint CONTACT_EVENT_WORDS = 12u;

[shader("compute")]
[numthreads(64, 1, 1)]
void contact_events(uint3 tid : SV_DispatchThreadID)
{
    uint mi = dispatchIndex(tid);
    if (mi >= ALOAD(gCounters[0].manifoldCount)) return;
    Manifold mf = gManifolds[mi];
    if (mf.contactCount == 0u) return;
    Collider ca = gColliders[mf.colliderA];
    Collider cb = gColliders[mf.colliderB];
    uint flags = ca.material | cb.material;
    if ((flags & (COLLIDER_CONTACT_EVENTS | COLLIDER_SENSOR)) == 0u) return;
    bool sensor = (flags & COLLIDER_SENSOR) != 0u;
    if (sensor)
    {
        // Sensors have no response: the pair leaves the solve and the islands
        // (the CSR graph skips empty manifolds and solver pairs).
        gManifolds[mi].contactCount = 0u;
#if defined(ENGINE_PERSISTENT_CONTACTS)
        gSolverPairs[mi].countAndSphere &= ~SOLVER_COUNT_MASK;
#endif
    }
    float best = 3.402823e38;
    Contact deepest = (Contact)0;
    for (uint k = 0u; k < mf.contactCount; k++)
    {
        // Absolute index after persistence; READ_CONTACT is the storage mode's.
        Contact c = READ_CONTACT(mf.contactStart + k);
        if (c.alive == 0u) continue;
        float gap = contactGapMode() ? c.c0N : c.c0N - PARAMS.contactOffset;
        if (gap < best)
        {
            best = gap;
            deepest = c;
        }
    }
    if (best > 3.0e38) return;
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    float3 rA = quatRotate(ba.orientation, deepest.anchorA);
    float3 rB = quatRotate(bb.orientation, deepest.anchorB);
    float3 vA = ba.linearVel + cross(ba.angularVel, rA);
    float3 vB = bb.linearVel + cross(bb.angularVel, rB);
    float3 n = mf.normal;
    float approach = dot(vA - vB, n);
    // Touching this step: within the contact offset, or closing that gap
    // within the step (a fast or continuous-collision impact reports its begin
    // on the impact step, at the speed before the solve).
    if (best > PARAMS.contactOffset + max(approach, 0.0) * PARAMS.dt) return;
    float3 point = 0.5 * (ba.position + rA + bb.position + rB);
    uint slot;
    AADD3(gScratchAtomics[CONTACT_EVENT_BASE], 1u, slot);
    if (slot >= CONTACT_EVENT_CAP) return;
    uint w = CONTACT_EVENT_BASE + 4u + CONTACT_EVENT_WORDS * slot;
    ASTORE(gScratchAtomics[w + 0u], mf.colliderA);
    ASTORE(gScratchAtomics[w + 1u], mf.colliderB);
    ASTORE(gScratchAtomics[w + 2u], asuint(approach));
    ASTORE(gScratchAtomics[w + 3u], sensor ? 1u : 0u);
    ASTORE(gScratchAtomics[w + 4u], asuint(point.x));
    ASTORE(gScratchAtomics[w + 5u], asuint(point.y));
    ASTORE(gScratchAtomics[w + 6u], asuint(point.z));
    ASTORE(gScratchAtomics[w + 8u], asuint(n.x));
    ASTORE(gScratchAtomics[w + 9u], asuint(n.y));
    ASTORE(gScratchAtomics[w + 10u], asuint(n.z));
}

// ----- overlap queries (P5) and explosions -------------------------------------
// Fixed regions after the contact events (engine-schema QUERY_* / EXPLODE_*).
static const uint QUERY_BASE = CONTACT_EVENT_BASE + 4u + CONTACT_EVENT_WORDS * CONTACT_EVENT_CAP;
static const uint QUERY_CAP = 32u;
static const uint QUERY_RESULT_BASE = QUERY_BASE + 4u + 8u * QUERY_CAP;
static const uint QUERY_RESULT_CAP = 8192u;
static const uint EXPLODE_BASE = QUERY_RESULT_BASE + 2u * QUERY_RESULT_CAP;

// A collider's filter: category bits in the low half, mask bits in the high
// half of `generation`; 0 = everything.
uint colliderCategories(Collider c) { return c.generation == 0u ? 0xFFFFu : (c.generation & 0xFFFFu); }
uint colliderMask(Collider c) { return c.generation == 0u ? 0xFFFFu : (c.generation >> 16u); }

// Shapes whose bounds overlap each queued box (Box3D b3World_OverlapAABB,
// with its query filter; sensors are not reported). One thread per collider,
// after build_aabbs refreshed the bounds for the step's final poses. Reads
// physics state only.
[shader("compute")]
[numthreads(64, 1, 1)]
void overlap_query(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].colliderCount) return;
    Collider col = gColliders[i];
    if ((col.material & COLLIDER_SENSOR) != 0u) return;
    if (col.shape == SHAPE_PLANE) return;
    Aabb box = gAabbs[HOST_S(1) + i];
    uint n = min(ALOAD(gScratchAtomics[QUERY_BASE]), QUERY_CAP);
    for (uint q = 0u; q < n; q++)
    {
        uint w = QUERY_BASE + 4u + 8u * q;
        float3 mn = float3(asfloat(ALOAD(gScratchAtomics[w])), asfloat(ALOAD(gScratchAtomics[w + 1u])),
                           asfloat(ALOAD(gScratchAtomics[w + 2u])));
        uint categories = ALOAD(gScratchAtomics[w + 3u]);
        float3 mx = float3(asfloat(ALOAD(gScratchAtomics[w + 4u])), asfloat(ALOAD(gScratchAtomics[w + 5u])),
                           asfloat(ALOAD(gScratchAtomics[w + 6u])));
        uint mask = ALOAD(gScratchAtomics[w + 7u]);
        if ((colliderCategories(col) & mask) == 0u || (colliderMask(col) & categories) == 0u) continue;
        if (any(box.mx < mn) || any(box.mn > mx)) continue;
        uint slot;
        AADD3(gScratchAtomics[QUERY_BASE + 1u], 1u, slot);
        if (slot >= QUERY_RESULT_CAP) return;
        ASTORE(gScratchAtomics[QUERY_RESULT_BASE + 2u * slot], q);
        ASTORE(gScratchAtomics[QUERY_RESULT_BASE + 2u * slot + 1u], i);
    }
}

// Radial impulses (Box3D b3World_Explode), at the start of a step that has
// them: every collider of a dynamic body within radius + falloff of a queued
// explosion takes impulse-per-area x its cross-section (pi r^2 of its bounding
// radius), full strength inside the radius and fading linearly to zero over
// the falloff, pushed away from the centre at the collider's centre. The body
// wakes. One thread per body (colliders are sorted by body).
[shader("compute")]
[numthreads(64, 1, 1)]
void explode(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if ((b.flags & BODY_DYNAMIC) == 0u || b.invMass <= 0.0 || bodyIsKinematic(b)) return;
    uint n = min(ALOAD(gScratchAtomics[EXPLODE_BASE]), 16u);
    bool hit = false;
    uint count = gCounters[0].colliderCount;
    for (uint c = firstColliderOfBody(i); c < count && gColliders[c].body == i; c++)
    {
        Collider col = gColliders[c];
        if ((col.material & COLLIDER_SENSOR) != 0u) continue;
        float3 centre = b.position + quatRotate(b.orientation, col.localPos);
        float r = col.params.w;
        for (uint e = 0u; e < n; e++)
        {
            uint w = EXPLODE_BASE + 4u + 8u * e;
            float3 at = float3(asfloat(ALOAD(gScratchAtomics[w])), asfloat(ALOAD(gScratchAtomics[w + 1u])),
                               asfloat(ALOAD(gScratchAtomics[w + 2u])));
            float radius = asfloat(ALOAD(gScratchAtomics[w + 3u]));
            float falloff = asfloat(ALOAD(gScratchAtomics[w + 4u]));
            float perArea = asfloat(ALOAD(gScratchAtomics[w + 5u]));
            uint mask = ALOAD(gScratchAtomics[w + 6u]);
            if ((colliderCategories(col) & mask) == 0u) continue;
            float3 off = centre - at;
            float dist = length(off);
            float d = max(dist - r, 0.0);
            float scale = d <= radius ? 1.0 : (falloff > 0.0 ? max(0.0, 1.0 - (d - radius) / falloff) : 0.0);
            if (scale <= 0.0) continue;
            float3 dir = dist > 1e-6 ? off / dist : float3(0.0, 1.0, 0.0);
            float3 j = dir * (scale * perArea * 3.14159265 * r * r);
            b.linearVel = b.linearVel + j * b.invMass;
            b.angularVel = b.angularVel + invInertiaWorld(b, cross(centre - b.position, j));
            hit = true;
        }
    }
    if (!hit) return;
    b.flags = (b.flags & ~BODY_SLEEPING) | BODY_AWAKE;
    b.sleepTimer = 0.0;
    gBodies[i].linearVel = b.linearVel;
    gBodies[i].angularVel = b.angularVel;
    gBodies[i].flags = b.flags;
    gBodies[i].sleepTimer = b.sleepTimer;
}
// ----- motion locks (P3) ------------------------------------------------------
// Scheduled only while some body has locks (World), so no other step changes:
// lock_predict after predict_bodies, lock_project after the solve. Each
// projects a locked body's pose onto its free axes, measured from the step
// start (initialPos / initialRot): the locked components of its displacement
// and of its rotation (world axes) are removed. The prediction and the
// inertial target the solve pulls toward respect the locks; the final
// projection makes the locked axes exact at the end of every step.
uint bodyLocks(BodyState b) { return (b.flags & BODY_LOCK_MASK) >> BODY_LOCK_SHIFT; }
float3 lockKeep(uint bits) { return float3((bits & 1u) == 0u ? 1.0 : 0.0, (bits & 2u) == 0u ? 1.0 : 0.0, (bits & 4u) == 0u ? 1.0 : 0.0); }

float3 lockPosition(float3 x, float3 x0, uint locks) { return x0 + (x - x0) * lockKeep(locks); }

float4 lockRotation(float4 q, float4 q0, uint locks)
{
    if (dot(q, q0) < 0.0) q = -q;
    float4 dq = quatMul(q, quatConjugate(q0)); // world-frame rotation since step start
    float3 v = dq.xyz * lockKeep(locks >> 3u);
    float w = sqrt(max(1.0 - dot(v, v), 0.0));
    return quatNormalize(quatMul(float4(v, w), q0));
}

bool lockedBody(BodyState b) { return bodyIsDynamic(b) && !bodyIsSleeping(b) && bodyLocks(b) != 0u; }

[shader("compute")]
[numthreads(64, 1, 1)]
void lock_predict(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (!lockedBody(b)) return;
    uint locks = bodyLocks(b);
    gBodies[i].position = lockPosition(b.position, b.initialPos, locks);
    gBodies[i].orientation = lockRotation(b.orientation, b.initialRot, locks);
    // The inertial target predict_bodies left in gAabbs[i].
    Aabb box = gAabbs[i];
    float4 target = lockRotation(float4(box.mx, box._pad0), b.initialRot, locks);
    box.mn = lockPosition(box.mn, b.initialPos, locks);
    box.mx = target.xyz;
    box._pad0 = target.w;
    gAabbs[i] = box;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lock_project(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (!lockedBody(b)) return;
    uint locks = bodyLocks(b);
    gBodies[i].position = lockPosition(b.position, b.initialPos, locks);
    gBodies[i].orientation = lockRotation(b.orientation, b.initialRot, locks);
}

#include "ccd_step.slang"
