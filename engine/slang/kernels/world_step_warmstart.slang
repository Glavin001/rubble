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
    if ((j.flags & (JOINT_FLAG_BROKEN | JOINT_FLAG_IDLE | JOINT_FLAG_TYRE)) != 0u) return;
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
    return !bodyMovingOrKinematic(ba) && !bodyMovingOrKinematic(bb);
}

// ---- Sleep archive (HOST_S(30) bit 5; canonical contact rows) ----
// A pair that falls asleep stops producing contact rows, so a body woken later used to
// restart its resting contacts cold (λ = 0, k = k_start) and its first steps carried
// less load than had it stayed awake. When a body falls asleep, the rows it rests on
// (its partner static or asleep) are kept here; when the pair forms again, a warm-start
// miss finds them, and the body wakes with the contacts it fell asleep with. Nothing
// runs for a body while it sleeps. Rows live in the contact rows past the two halves, in
// blocks of ARCHIVE_ROWS; per body records and the free-block stack sit past the
// scratch words in gScratchAtomics:
//   [free top, blocks, touched count (two, by step parity)]
//   [row count, first block, age, cursor, touched step] x maxBodies [next] x blocks
//   [free stack] x blocks [touched list] x maxBodies [spread step] x maxBodies
// An archive that keeps no rows (cold wakes with the touch query, HOST_S(30) bit 9,
// archiveCompact) has only the touch query's records past the header:
//   [touched step] x maxBodies [touched list] x maxBodies [spread step] x maxBodies
static const uint ARCHIVE_ROWS = 8u;
static const uint ARCHIVE_NONE = 0xFFFFFFFFu;
static const uint ARCHIVE_JUST_SLEPT = 0xFFFFFFFEu;
static const uint OVERFLOW_SLEEP_ARCHIVE = 131072u;
bool sleepArchive() { return (HOST_S(30) & 32u) != 0u; }
// The touch query (archive_touch; HOST_S(30) bit 8, set only with the archive).
bool archiveTouchQuery() { return (HOST_S(30) & 256u) != 0u; }
// Measurement variants (HOST_S(30) bits 6 and 7; off by default): a woken pair starts
// cold instead of from its archived row; archive_spread lists the sleepers resting on a
// touched sleeper (over the archived adjacency) so archive_emit gives them rows too.
bool archiveCold() { return (HOST_S(30) & 64u) != 0u; }
// No rows, no blocks, no per-body row records (WgpuSim::enable_sleep_archive).
bool archiveCompact() { return (HOST_S(30) & 512u) != 0u; }
uint archiveBlocks() { return ALOAD(gScratchAtomics[archiveMeta() + 1u]); }
uint archiveMeta() { return EXPLODE_BASE + 4u + 8u * 16u; }
uint arTop() { return archiveMeta(); }
uint arCount(uint b) { return archiveMeta() + 4u + b; }
uint arHead(uint b) { return archiveMeta() + 4u + HOST_S(1) + b; }
uint arAge(uint b) { return archiveMeta() + 4u + 2u * HOST_S(1) + b; }
uint arCursor(uint b) { return archiveMeta() + 4u + 3u * HOST_S(1) + b; }
uint arTouched(uint b) { return archiveMeta() + 4u + (archiveCompact() ? 0u : 4u * HOST_S(1)) + b; }
uint arNext(uint k) { return archiveMeta() + 4u + 5u * HOST_S(1) + k; }
uint arStack(uint k) { return archiveMeta() + 4u + 5u * HOST_S(1) + archiveBlocks() + k; }
uint arTouchedCount(uint step) { return archiveMeta() + 2u + (step & 1u); }
uint arTouchedList(uint k)
{
    return archiveMeta() + 4u + (archiveCompact() ? HOST_S(1) : 5u * HOST_S(1) + 2u * archiveBlocks()) + k;
}
// The step a listed sleeper spreads in (archive_touch_seed), after the touched list.
uint arSpread(uint b)
{
    return archiveMeta() + 4u + (archiveCompact() ? 2u * HOST_S(1) : 6u * HOST_S(1) + 2u * archiveBlocks()) + b;
}
uint archiveRow(uint block, uint j) { return 2u * HOST_S(3) + block * ARCHIVE_ROWS + j; }

// Both colliders' local poses: a split re-parents a collider (new local pose), and its
// archived rows (anchors in the old body frame) no longer apply.
uint archiveKey(Collider a, Collider b)
{
    uint h = 2166136261u;
    float v[14] = { a.localPos.x, a.localPos.y, a.localPos.z, a.localRot.x, a.localRot.y, a.localRot.z, a.localRot.w,
                    b.localPos.x, b.localPos.y, b.localPos.z, b.localRot.x, b.localRot.y, b.localRot.z, b.localRot.w };
    for (uint k = 0u; k < 14u; k++)
        h = (h ^ asuint(v[k])) * 16777619u;
    return h;
}

#if defined(ENGINE_SLEEP_ARCHIVE)
// The body a resting row is archived under: one that fell asleep this step, its
// partner at rest (static or asleep); else none (the pair keeps forming).
uint archiveOwner(Contact c)
{
    uint a = gColliders[c.colliderA].body;
    uint b = gColliders[c.colliderB].body;
    BodyState ba = gBodies[a];
    BodyState bb = gBodies[b];
    bool slptA = bodyIsDynamic(ba) && bodyIsSleeping(ba) && ALOAD(gScratchAtomics[arAge(a)]) == ARCHIVE_JUST_SLEPT;
    bool slptB = bodyIsDynamic(bb) && bodyIsSleeping(bb) && ALOAD(gScratchAtomics[arAge(b)]) == ARCHIVE_JUST_SLEPT;
    if (slptA && !bodyMovingOrKinematic(bb)) return a;
    if (slptB && !bodyMovingOrKinematic(ba)) return b;
    return ARCHIVE_NONE;
}

// An archived row from before its collider was re-parented: its anchors (old frame)
// are not pinned (persistence keeps a row's anchors only while it sticks).
Contact archiveStaleRow(Contact prev)
{
    prev._pad0 = 0.0;
    return prev;
}

// A warm-start miss: the row this pair fell asleep with, in either body's archive
// (same feature, nearest anchors; else the nearest anchor within the fraction). With
// the touch query, a row whose collider has since been re-parented (a split: the key
// differs, its anchors are in the old body frame) still lends a same-feature match its
// impulses and stiffness, as the previous step's row does without the archive; its
// anchors are not kept (archiveStaleRow).
bool archiveFind(Contact c, out Contact found)
{
    Collider colA = gColliders[c.colliderA];
    Collider colB = gColliders[c.colliderB];
    uint key = archiveKey(colA, colB);
    float minSide = min(colliderMinSide(colA), colliderMinSide(colB));
    float bestFeature = 1e30;
    float bestNear = NEAREST_FRACTION * minSide;
    bool haveFeature = false;
    bool haveNear = false;
    bool haveStale = false;
    bool staleOk = archiveTouchQuery();
    found = c;
    Contact near = c;
    Contact stale = c;
    uint owners[2] = { colA.body, colB.body };
    for (uint side = 0u; side < 2u; side++)
    {
        uint body = owners[side];
        // A compact archive has no rows (and its words past the header are not records).
        uint k = archiveCompact() ? ARCHIVE_NONE : ALOAD(gScratchAtomics[arHead(body)]);
        if (k == ARCHIVE_NONE) continue;
        uint n = ALOAD(gScratchAtomics[arCount(body)]);
        for (uint slot = 0u; slot < n && slot < 8192u; slot++)
        {
            if (slot > 0u && slot % ARCHIVE_ROWS == 0u)
                k = ALOAD(gScratchAtomics[arNext(k)]);
            Contact prev = READ_CONTACT(archiveRow(k, slot % ARCHIVE_ROWS));
            if (prev.colliderA != c.colliderA || prev.colliderB != c.colliderB)
                continue;
            if (asuint(prev._pad1) != key)
            {
                if (staleOk && !haveStale && prev.featureId == c.featureId)
                {
                    stale = archiveStaleRow(prev);
                    haveStale = true;
                }
                continue;
            }
            float3 dA = prev.anchorA - c.anchorA;
            float3 dB = prev.anchorB - c.anchorB;
            if (prev.featureId == c.featureId)
            {
                float d = dot(dA, dA) + dot(dB, dB);
                if (d < bestFeature) { bestFeature = d; found = prev; haveFeature = true; }
            }
            else
            {
                float d = length(dA);
                if (d <= bestNear) { bestNear = d; near = prev; haveNear = true; }
            }
        }
    }
    if (!haveFeature && haveNear) found = near;
    else if (!haveFeature && haveStale) found = stale;
    return haveFeature || haveNear || haveStale;
}

// The archived row at `slot` of `body`'s archive (walking its block chain).
Contact archiveRowOf(uint body, uint slot)
{
    uint k = ALOAD(gScratchAtomics[arHead(body)]);
    for (uint h = slot / ARCHIVE_ROWS; h > 0u && h < 4096u; h--)
        k = ALOAD(gScratchAtomics[arNext(k)]);
    return READ_CONTACT(archiveRow(k, slot % ARCHIVE_ROWS));
}

// A moving body's pair reaches a sleeping one with an archive: the first such touch in
// a step lists the sleeper; archive_emit (after the broadphase) emits its archived
// resting pairs, so if the touch wakes it, its supports are in this step's contacts,
// warm, as they would be had it never slept. Marking reads no contact rows (the
// broadphase stays within WebGPU's storage-buffer budget).
void archiveMark(uint body)
{
    if (archiveCompact() || ALOAD(gScratchAtomics[arHead(body)]) == ARCHIVE_NONE) return;
    archiveList(body);
}

// List `body` once this step (the touch query, archiveTouchQuery, lists every touched
// sleeper, archive or not).
void archiveList(uint body)
{
    uint stamp = HOST_S(17) + 1u;
    uint old;
    AXCHG3(gScratchAtomics[arTouched(body)], stamp, old);
    if (old == stamp) return;
    uint at;
    AADD3(gScratchAtomics[arTouchedCount(HOST_S(17))], 1u, at);
    if (at < HOST_S(1))
        ASTORE(gScratchAtomics[arTouchedList(at)], body);
}

// The touch query lists a touched sleeper only if the touch can wake it this step
// (wakeFromContact's rule on the same velocities): one it cannot wake stays asleep, its
// pairs stay at rest, and nothing this step reads them.
bool touchMayWake(BodyState m, BodyState s, uint cm, uint cs)
{
    bool awake = (bodyIsDynamic(m) && !bodyIsSleeping(m) && canWake(m)) || kinematicMoving(m);
    if (!awake) return false;
    if (wakeOnTouch()) return true;
    float3 vRel = m.linearVel - s.linearVel;
    float thresh = PARAMS.sleepLin;
    if (pointSpeedSleep())
        return length(vRel) + length(m.angularVel) * colliderExtent(gColliders[cm]) +
            length(s.angularVel) * colliderExtent(gColliders[cs]) >= thresh;
    return dot(vRel, vRel) >= thresh * thresh || length(m.angularVel) + length(s.angularVel) >= PARAMS.sleepAng;
}

void archiveTouch(uint i, uint j)
{
    uint bi = gColliders[i].body;
    uint bj = gColliders[j].body;
    BodyState a = gBodies[bi];
    BodyState b = gBodies[bj];
    if (bodyMovingOrKinematic(a) && bodyIsDynamic(b) && bodyIsSleeping(b))
    {
        if (!archiveTouchQuery()) archiveMark(bj);
        else if (touchMayWake(a, b, i, j)) archiveList(bj);
    }
    else if (bodyMovingOrKinematic(b) && bodyIsDynamic(a) && bodyIsSleeping(a))
    {
        if (!archiveTouchQuery()) archiveMark(bi);
        else if (touchMayWake(b, a, j, i)) archiveList(bi);
    }
}

// A listed sleeper's archived resting pairs (each pair once; a pair whose partner
// moves is emitted by it).
void archiveEmitResting(uint body)
{
    if (archiveCompact()) return;
    uint n = min(ALOAD(gScratchAtomics[arCount(body)]), 2048u);
    for (uint slot = 0u; slot < n && slot < 2048u; slot++)
    {
        Contact r = archiveRowOf(body, slot);
        uint other = gColliders[r.colliderA].body == body ? gColliders[r.colliderB].body : gColliders[r.colliderA].body;
        if (bodyMovingOrKinematic(gBodies[other])) continue;
        bool first = true;
        for (uint e = 0u; e < slot && e < 2048u; e++)
        {
            Contact q = archiveRowOf(body, e);
            if (q.colliderA == r.colliderA && q.colliderB == r.colliderB) { first = false; break; }
        }
        if (first)
            emitCandidateCore(r.colliderA, r.colliderB);
    }
}

// ---- Touch query (HOST_S(30) bit 8, with the sleep archive) ----
// A sleeper touched by a moving body this step (archiveMark, from the moving body's
// candidate pairs, or woken after the broadphase by a projectile hit, apply_impulses) forms every resting pair it has, before this step's narrowphase:
// archive_touch queries each listed sleeper's colliders as if it were awake, against
// the colliders at rest (static, or asleep: a mover's pairs are its own query's), so
// when the touch wakes it its supports are in this step's contacts, as they would be
// had it never slept, whichever body's archive holds them; persistence then warms each
// from either body's archive (archiveFind). Each pair is emitted once: of two touched
// sleepers at one grid level the lower collider index emits it, across levels the
// finer one (the coarser one's finer scan skips touched partners).
bool archiveTouchedNow(uint body) { return ALOAD(gScratchAtomics[arTouched(body)]) == HOST_S(17) + 1u; }

// Touched collider i's pair with collider j (a candidate by bounds): j at rest, on
// another body, not jointed; `lowerOwns` (both scan each other): a touched partner's pair
// goes to the lower index; otherwise only i reaches j (a coarser level: j's finer scan
// skips touched colliders), so i emits it.
// `mark` (archive_touch_spread): list a sleeping partner instead of emitting the pair.
void touchEmit(uint i, Collider a, uint j, bool lowerOwns, bool probeHasObb, BoxObb probe, float faceGap, bool mark)
{
    if (j == i) return;
    Collider b = gColliders[j];
    if (b.body == a.body || b.shape == SHAPE_PLANE) return;
    if (mark)
    {
        BodyState ob = gBodies[b.body];
        if (bodyIsDynamic(ob) && bodyIsSleeping(ob) && !bodiesJointed(a.body, b.body) &&
            !(probeHasObb && boxesSeparatedOnFaceAxis(probe, j, faceGap)))
            archiveList(b.body);
        return;
    }
    // A touched partner (asleep at the broadphase, perhaps woken since by a projectile
    // hit) also queries; any other moving partner emitted the pair from its own query.
    if (archiveTouchedNow(b.body))
    {
        if (lowerOwns && j < i) return;
    }
    else if (bodyMovingOrKinematic(gBodies[b.body])) return;
    if (bodiesJointed(a.body, b.body)) return;
    if (probeHasObb && boxesSeparatedOnFaceAxis(probe, j, faceGap)) return;
    emitCandidateCore(i, j);
}

// Hierarchical grids: planes, oversized statics, the own and coarser levels' cells
// (hierQuery's visit), then the finer levels' (hierScanFiner's acceptance test).
void touchQueryHier(uint i, bool mark)
{
    Collider a = gColliders[i];
    if (a.shape == SHAPE_PLANE) return;
    uint base = HOST_S(1);
    Aabb aa = gAabbs[base + i];
    uint own = hierLevel(aa._pad0);
    uint levels = HOST_S(18);
    bool probeHasObb = shapeHasHull(a.shape);
    BoxObb probe = probeHasObb ? worldObbFromCache(i)
                               : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    float faceGap = max(0.001, speculativeMargin()) + (probeHasObb ? cachedReach(i) : 0.0);
    uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
    for (uint pi = 0u; pi < nPlanes && !mark; pi++)
    {
        uint j = ALOAD(gScratchAtomics[43u + pi]);
        if (!aabbAbovePlane(aa, gColliders[j].params)) emitCandidateCore(i, j);
    }
    uint nLarge = min(ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT]), HIER_LARGE_MAX);
    for (uint k = 0u; k < nLarge; k++)
    {
        uint j = ALOAD(gScratchAtomics[HIER_LARGE_COUNT_SLOT + 1u + k]);
        if (aabbOverlap(aa, gAabbs[base + j]))
            touchEmit(i, a, j, false, probeHasObb, probe, faceGap, mark);
    }
    if (own >= levels) return;
    bool rest = restTables();
    uint heads = rest ? restHeads() : 0u;
    uint count = gCounters[0].colliderCount;
    uint occupied = ALOAD(gScratchAtomics[HIER_LEVEL_MASK_SLOT]);
    for (uint level = own; level < levels; level++)
    {
        if (((occupied >> level) & 1u) == 0u) continue;
        float cs = hierCellSize(level);
        int3 center = hierCell(aa, level);
        int3 lo = max(int3(floor((aa.mn - 0.5 * cs) / cs)), center - 1);
        int3 hi = min(int3(floor((aa.mx + 0.5 * cs) / cs)), center + 1);
        for (int z = lo.z; z <= hi.z; z++)
        for (int y = lo.y; y <= hi.y; y++)
        for (int x = lo.x; x <= hi.x; x++)
        {
            int3 cell = int3(x, y, z);
            uint j = ALOAD(gConstraintIds[heads + hierHash(cell, level)]);
            uint guard = 0u;
            while (j != 0xFFFFFFFFu && guard < count)
            {
                guard++;
                Aabb ab = gAabbs[base + j];
                uint next = rest ? ALOAD(gConstraintIds[restLinks() + j]) : asuint(ab._pad1);
                if (j != i && aabbOverlap(aa, ab) && hierLevel(ab._pad0) == level &&
                    all(hierCell(ab, level) == cell))
                    touchEmit(i, a, j, level == own, probeHasObb, probe, faceGap, mark);
                j = next;
            }
        }
    }
    uint mask = occupied & ((1u << own) - 1u);
    float csM = hierCellSize(own);
    int3 cellM = hierCell(aa, own);
    for (uint level = 0u; level < own; level++)
    {
        if (((mask >> level) & 1u) == 0u) continue;
        int3 lo;
        uint3 span = finerSpan(aa, level, lo);
        uint cells = span.x * span.y * span.z;
        for (uint t = 0u; t < cells && t < 65536u; t++)
        {
            int3 cell = lo + int3(uint3(t % span.x, (t / span.x) % span.y, t / (span.x * span.y)));
            uint j = ALOAD(gConstraintIds[heads + hierHash(cell, level)]);
            uint guard = 0u;
            while (j != 0xFFFFFFFFu && guard < count)
            {
                guard++;
                Aabb ab = gAabbs[base + j];
                uint next = rest ? ALOAD(gConstraintIds[restLinks() + j]) : asuint(ab._pad1);
                if (aabbOverlap(aa, ab) && hierLevel(ab._pad0) == level &&
                    all(hierCell(ab, level) == cell))
                {
                    int3 jc = hierCell(ab, own);
                    int3 jlo = max(int3(floor((ab.mn - 0.5 * csM) / csM)), jc - 1);
                    int3 jhi = min(int3(floor((ab.mx + 0.5 * csM) / csM)), jc + 1);
                    // A touched finer partner emits the pair from its own coarser scan.
                    if (!archiveTouchedNow(gColliders[j].body) && all(cellM >= jlo) && all(cellM <= jhi))
                        touchEmit(i, a, j, false, probeHasObb, probe, faceGap, mark);
                }
                j = next;
            }
        }
    }
}

// Uniform grids: planes, listed large statics, then the resting colliders in the 27
// cells (the split grid's other table, or the one table's entries not awake). A large
// touched collider walks the cells under its box for small statics instead (small
// sleepers pair with it from their own query, as do large dynamics with everyone).
void touchQueryGrid(uint i, bool mark)
{
    Collider col = gColliders[i];
    if (col.shape == SHAPE_PLANE) return;
    uint base = HOST_S(1);
    Aabb box = gAabbs[base + i];
    float cellSize = max(PARAMS.gridCellSize, 0.5);
    bool probeIsBox = shapeHasHull(col.shape);
    BoxObb probeObb = probeIsBox ? worldObbFromCache(i) : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    float faceGap = max(0.001, speculativeMargin()) + (probeIsBox ? cachedReach(i) : 0.0);
    float rA = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
    bool queryLarge = rA > cellSize;
    uint nPlanes = min(ALOAD(gScratchAtomics[42]), 8u);
    for (uint pi = 0u; pi < nPlanes && !mark; pi++)
        emitCandidateCore(i, ALOAD(gScratchAtomics[43u + pi]));
    uint nListed = largeStaticCount();
    for (uint t = 0u; t < nListed; t++)
    {
        uint j = largeStaticCollider(t);
        if (aabbOverlap(box, gAabbs[base + j]))
            touchEmit(i, col, j, false, probeIsBox, probeObb, faceGap, mark);
    }
    bool split = gridSplit();
    // The uniform rest tables (restUniform): the other table is R, without its tombstones.
#if defined(ENGINE_WGSL)
    bool rest = restUniform();
#else
    bool rest = false;
#endif
    uint tableBase = rest ? restHeads() : split ? gridHashSize() : 0u;
    int3 lo, hi;
    if (queryLarge)
    {
        lo = int3(floor(box.mn / cellSize));
        hi = int3(floor(box.mx / cellSize));
        int3 span = hi - lo + int3(1, 1, 1);
        if (span.x * span.y * span.z > 512) return;
    }
    else
    {
        int3 cc = int3(floor((0.5 * (box.mn + box.mx)) / cellSize));
        lo = cc - 1;
        hi = cc + 1;
    }
    for (int z = lo.z; z <= hi.z; z++)
    for (int y = lo.y; y <= hi.y; y++)
    for (int x = lo.x; x <= hi.x; x++)
    {
        uint link = ALOAD(gConstraintIds[tableBase + gridHash(x, y, z)]);
        uint guard = 0u;
        while (link != 0xFFFFFFFFu && guard < 4096u)
        {
            uint cur = link & GRID_COLLIDER_MASK;
            bool awakeEntry = (link & GRID_AWAKE_DYNAMIC) != 0u;
            Aabb ab = gAabbs[base + cur];
            link = rest ? ALOAD(gConstraintIds[restLinks() + cur]) : split ? asuint(ab._pad1) : gColorLists[cur];
            guard++;
            if (awakeEntry || cur == i) continue;
            int3 cell = int3(floor((0.5 * (ab.mn + ab.mx)) / cellSize));
            if (any(cell != int3(x, y, z)) || !aabbOverlap(box, ab) || (rest && restTombstone(cur))) continue;
            if (queryLarge && bodyIsDynamic(gBodies[gColliders[cur].body])) continue;
            touchEmit(i, col, cur, true, probeIsBox, probeObb, faceGap, mark);
        }
    }
}

// Small worlds' exact pairwise scan: sleepers keep their static pairs there, so only
// pairs with other sleepers are missing.
void touchQuerySmall(uint i, bool mark)
{
    Collider col = gColliders[i];
    if (col.shape == SHAPE_PLANE) return;
    uint base = HOST_S(1);
    Aabb box = gAabbs[base + i];
    bool probeIsBox = shapeHasHull(col.shape);
    BoxObb probeObb = probeIsBox ? worldObbFromCache(i) : makeBoxObb(float3(0, 0, 0), float4(0, 0, 0, 1), float3(0, 0, 0));
    float faceGap = max(0.001, speculativeMargin()) + (probeIsBox ? cachedReach(i) : 0.0);
    uint n = gCounters[0].colliderCount;
    for (uint j = 0u; j < n && j < 512u; j++)
    {
        if (j == i || !aabbOverlap(box, gAabbs[base + j])) continue;
        uint bj = gColliders[j].body;
        BodyState ob = gBodies[bj];
        if (!archiveTouchedNow(bj) && (!bodyIsDynamic(ob) || !bodyIsSleeping(ob))) continue;
        touchEmit(i, col, j, true, probeIsBox, probeObb, faceGap, mark);
    }
}

// One collider of a listed sleeper (the passes scan the colliders: a threshold split
// re-parents colliders, so a body's colliders need not be contiguous).
void touchQueryCollider(uint c, bool mark)
{
    if (HOST_S(18) != 0u) touchQueryHier(c, mark);
    else if (gCounters[0].colliderCount <= 512u) touchQuerySmall(c, mark);
    else touchQueryGrid(c, mark);
}
#endif

