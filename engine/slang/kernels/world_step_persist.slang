// Shared point update; callers choose contact or manifold scheduling. Pair bodies
// are immutable during persistence, so manifold scheduling can load them once.
#if defined(ENGINE_SLEEP_ARCHIVE)
// persistContact's warm start from an archived row (a woken pair's resting contact):
// the same rules as a previous-step row.
Contact archiveWarmContact(Contact c, BodyState ba, BodyState bb, AvbdParams p)
{
    Contact prev;
    // archiveCold (measurement variant): the pair starts cold.
    if (archiveCold() || !archiveFind(c, prev))
        return c;
    // Refuse warm-start across SAT axis flips: λ from the old normal on a new
    // normal injects energy (random Box Pile). three-avbd avoids this via
    // reuseContacts keeping the manifold; without reuse, require n·n_prev>0.9.
    float nAgree = dot(prev.normal, c.normal);
    // Nor from a time-of-impact row (ccd_emit's features): its force
    // stopped a fast body within one step, often rigidly against a
    // partner held still, and ccd_respond already exchanged that
    // impact's momentum. Carried over, the rigid stiffness and the
    // stopping force held the striker dead against the struck body the
    // next step while the body moved off (an edge hit lost 84% of the
    // momentum).
    uint prevTag = prev.featureId >> 16;
    bool prevToi = prevTag == 0x0CCDu || prevTag == 0x0CCEu || prevTag == 0x0CCFu;
    if (nAgree > 0.9 && !prevToi)
    {
        c.lambdaN = prev.lambdaN;
        c.lambdaT1 = prev.lambdaT1;
        c.lambdaT2 = prev.lambdaT2;
        c.kN = prev.kN;
        c.kT = prev.kT;
        c._pad2 = prev._pad2;
        // three-avbd: keep old anchors while static friction held (stick).
        // Not for round shapes (spheres, capsules) — the contact point
        // slides over their surface as they roll; pinned anchors rotate
        // with the body, glue/stop smash balls, and let a rolling capsule
        // sink through the ground (its pinned anchor rises off the true
        // contact point every step).
        Collider colA = gColliders[c.colliderA];
        Collider colB = gColliders[c.colliderB];
        bool anySphere = colA.shape == SHAPE_SPHERE || colB.shape == SHAPE_SPHERE ||
            colA.shape == SHAPE_CAPSULE || colB.shape == SHAPE_CAPSULE;
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
    return c;
}
// A fresh row of a pair at rest (a touched sleeper's, archive_touch) stores the
// archived row's warm start: stored cold, it would mask the archive when the pair
// wakes (the warm start finds the previous step's row first).
Contact archiveRestRow(Contact c, BodyState ba, BodyState bb)
{
    return archiveWarmContact(c, ba, bb, PARAMS);
}
#endif
Contact persistContact(uint i, Contact c, bool reused, BodyState ba, BodyState bb)
{
    AvbdParams p = PARAMS;
    uint curOff = HOST_S(11);
    c.alive = 1;
    if (contactPairAtRest(ba, bb))
    {
#if defined(ENGINE_SLEEP_ARCHIVE)
        // A touched sleeper's resting pair keeps its archived row (archiveRestRow).
        if (!reused && archiveTouchQuery())
            c = archiveRestRow(c, ba, bb);
#endif
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
            // Nor from a time-of-impact row (ccd_emit's features): its force
            // stopped a fast body within one step, often rigidly against a
            // partner held still, and ccd_respond already exchanged that
            // impact's momentum. Carried over, the rigid stiffness and the
            // stopping force held the striker dead against the struck body the
            // next step while the body moved off (an edge hit lost 84% of the
            // momentum).
            uint prevTag = prev.featureId >> 16;
            bool prevToi = prevTag == 0x0CCDu || prevTag == 0x0CCEu || prevTag == 0x0CCFu;
            if (nAgree > 0.9 && !prevToi)
            {
                c.lambdaN = prev.lambdaN;
                c.lambdaT1 = prev.lambdaT1;
                c.lambdaT2 = prev.lambdaT2;
                c.kN = prev.kN;
                c.kT = prev.kT;
                c._pad2 = prev._pad2;
                // three-avbd: keep old anchors while static friction held (stick).
                // Not for round shapes (spheres, capsules) — the contact point
                // slides over their surface as they roll; pinned anchors rotate
                // with the body, glue/stop smash balls, and let a rolling capsule
                // sink through the ground (its pinned anchor rises off the true
                // contact point every step).
                Collider colA = gColliders[c.colliderA];
                Collider colB = gColliders[c.colliderB];
                bool anySphere = colA.shape == SHAPE_SPHERE || colB.shape == SHAPE_SPHERE ||
                    colA.shape == SHAPE_CAPSULE || colB.shape == SHAPE_CAPSULE;
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
#if defined(ENGINE_SLEEP_ARCHIVE)
        // A warm-start miss: the sleep archive (archiveWarmContact). Kept out of the
        // block above so its codegen is unchanged without the archive.
        else if (sleepArchive())
            c = archiveWarmContact(c, ba, bb, p);
#endif
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
    // apply_impulses consumed this step's records at its start.
    if (i == 0u)
    {
        ASTORE(gScratchAtomics[IMPULSE_COUNT], 0u);
        ASTORE(gScratchAtomics[IMPULSE_FIRST], 0u);
    }
    if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (bodyIsDynamic(b) && !bodyIsSleeping(b))
    {
        uint slot;
        AADD3(gCounters[0].awakeCount, 1, slot);
        gAwakeList[slot] = i;
    }
}

// ---- Awake bitmap (World's plan, HOST_S(25) bit 0 hosts; awakeBits) ----
// A bit per body slot, set by build_awake_list_bits for every body it lists (dynamic and
// not asleep), past the incremental rest inserts' bucket list. The per-body passes that
// act only on such bodies, while no body's flags change between the list and them
// (count_color_tiles, scatter_color_tiles, ccd_unhold), read one word per 32 bodies
// first instead of each body's record: a body whose bit is clear is one they leave alone.
bool awakeBit(uint body) { return (ALOAD(gConstraintIds[awakeBitsBase() + (body >> 5)]) & (1u << (body & 31u))) != 0u; }

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_awake_bits(uint3 tid : SV_DispatchThreadID)
{
    uint w = dispatchIndex(tid);
    if (w < (gCounters[0].bodyCount + 31u) / 32u) ASTORE(gConstraintIds[awakeBitsBase() + w], 0u);
}

// build_awake_list, also setting each listed body's bit.
[shader("compute")]
[numthreads(64, 1, 1)]
void build_awake_list_bits(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i == 0u)
    {
        ASTORE(gScratchAtomics[IMPULSE_COUNT], 0u);
        ASTORE(gScratchAtomics[IMPULSE_FIRST], 0u);
    }
    if (i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (bodyIsDynamic(b) && !bodyIsSleeping(b))
    {
        uint slot;
        AADD3(gCounters[0].awakeCount, 1, slot);
        gAwakeList[slot] = i;
        AOR2(gConstraintIds[awakeBitsBase() + (i >> 5)], 1u << (i & 31u));
    }
}

