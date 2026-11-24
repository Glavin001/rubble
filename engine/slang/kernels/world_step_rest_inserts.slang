// ---- Incremental resting-table inserts (HOST_S(25) bit 5, both grids) ----
// After the sleep commit, instead of a rebuild the next step: the colliders listed as
// moving this step (slot 81) whose bodies are at rest now go into R, so R again holds
// every collider at rest at its current bucket, as a rebuild would. A collider already
// in R at that bucket (a tombstone of a body that woke and fell asleep again in the
// same cell, or one whose cell moved within the bucket) is valid again as it is. One in
// another bucket is unlinked from that chain first: each such bucket is claimed once
// (its W head, empty after the step's queries, serves as the claim), and one thread per
// claimed bucket splices its leaving colliders out (rest_unlink, which also empties that
// W head again); then every collider
// not in R is pushed (rest_push). A collider R cannot place incrementally (a
// hierarchical collider moving between a cell and the oversized list) asks for a
// rebuild at the next step (header word 31). Every chain then holds the colliders a
// rebuild would put in it, plus tombstones, in another order; walks test each entry's
// true cell and skip tombstones, so they find the same colliders.
static const uint REST_REBUILD_WORD = 31u;
static const uint REST_DIRTY_WORD = 30u;
static const uint REST_AT_LEAVING = 0xFFFFFFFDu;
// A W head claimed for rest_unlink (no grid entry has this value: ids are below 2^31-2).
static const uint REST_CLAIMED = 0xFFFFFFFEu;
bool restIncremental() { return (HOST_S(25) & 32u) != 0u; }
uint restDirtyBase() { return restLinks() + 4u * gCounters[0].colliderCount; }

// Bucket of a collider at rest in R: its centre's uniform cell, or its hierarchical
// cell and level; REST_AT_LISTED for the header lists (planes, uniform large dynamics,
// hierarchical oversized colliders).
uint restBucket(uint c, Collider col, BodyState b, out uint level)
{
    level = 0u;
    if (col.shape == SHAPE_PLANE) return REST_AT_LISTED;
    Aabb box = gAabbs[HOST_S(1) + c];
    if (restUniform())
    {
        float cellSize = max(PARAMS.gridCellSize, 0.5);
        float boundR = col.params.w > 0.0 ? col.params.w : length(col.params.xyz);
        if (bodyIsDynamic(b) && boundR > cellSize) return REST_AT_LISTED;
        int3 cc = uniformCell(box, cellSize);
        return gridHash(cc.x, cc.y, cc.z);
    }
    level = hierLevel(box._pad0);
    if (level >= HOST_S(18)) return REST_AT_LISTED;
    return hierHash(hierCell(box, level), level);
}

// A listed collider at rest now (the listing's test: hierarchical lists kinematics too).
bool restSleeperNow(Collider col)
{
    BodyState b = gBodies[col.body];
    return restUniform() ? !bodyMoving(b) : !bodyMovingOrKinematic(b);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rest_sleepers_mark(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    Collider col = gColliders[c];
    if (!restSleeperNow(col)) return;
    BodyState b = gBodies[col.body];
    uint level;
    uint h = restBucket(c, col, b, level);
    uint cur = ALOAD(gConstraintIds[restAtBase() + c]);
    uint hdr = restBase();
    // Its bounds now (an entry valid again keeps its place but may have grown).
    if (restUniform() && h != REST_AT_LISTED)
        AMAX2(gConstraintIds[hdr + REST_U_HALF], asuint(boxHalfMax(gAabbs[HOST_S(1) + c])));
    if (h == cur) return;
    if (h == REST_AT_LISTED || cur == REST_AT_LISTED)
    {
        // Uniform header lists: a plane or large dynamic not listed yet joins them
        // (they are listed by its shape, which never changes). Anything else moving
        // between a list and a chain: rebuild.
        if (restUniform() && cur == REST_AT_NONE)
        {
            bool plane = col.shape == SHAPE_PLANE;
            uint slot;
            AADD3(gConstraintIds[hdr + (plane ? 1u : 10u)], 1u, slot);
            if (slot < (plane ? 8u : REST_U_LARGE_MAX))
                ASTORE(gConstraintIds[hdr + (plane ? 2u : 11u) + slot], c);
            ASTORE(gConstraintIds[restAtBase() + c], REST_AT_LISTED);
        }
        else
            ASTORE(gConstraintIds[hdr + REST_REBUILD_WORD], 1u);
        return;
    }
    if (cur == REST_AT_NONE) return; // rest_push inserts it
    // In another chain: leave it (claim that bucket for one unlinking thread).
    ASTORE(gConstraintIds[restAtBase() + c], REST_AT_LEAVING);
    uint prev;
    AXCHG3(gConstraintIds[awakeHeads() + cur], REST_CLAIMED, prev);
    if (prev != REST_CLAIMED)
    {
        uint k;
        AADD3(gConstraintIds[hdr + REST_DIRTY_WORD], 1u, k);
        ASTORE(gConstraintIds[restDirtyBase() + k], cur);
    }
}

// Work slot 86: one thread per claimed bucket.
[shader("compute")]
[numthreads(64, 1, 1)]
void write_rest_dirty_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    writeDispatchArgs(86u, ALOAD(gConstraintIds[restBase() + REST_DIRTY_WORD]));
}

// One claimed bucket: its leaving colliders spliced out (no other thread walks it).
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_unlink(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    uint hdr = restBase();
    if (k >= ALOAD(gConstraintIds[hdr + REST_DIRTY_WORD])) return;
    uint h = ALOAD(gConstraintIds[restDirtyBase() + k]);
    uint count = gCounters[0].colliderCount;
    uint before = 0xFFFFFFFFu;
    uint j = ALOAD(gConstraintIds[restHeads() + h]);
    uint guard = 0u;
    while (j != 0xFFFFFFFFu && guard < count)
    {
        guard++;
        uint next = ALOAD(gConstraintIds[restLinks() + j]);
        if (ALOAD(gConstraintIds[restAtBase() + j]) == REST_AT_LEAVING)
        {
            if (before == 0xFFFFFFFFu) ASTORE(gConstraintIds[restHeads() + h], next);
            else ASTORE(gConstraintIds[restLinks() + before], next);
            ASTORE(gConstraintIds[restAtBase() + j], REST_AT_NONE);
            if (restUniform()) AADD2(gConstraintIds[hdr], 0xFFFFFFFFu);
        }
        else
            before = j;
        j = next;
    }
    ASTORE(gConstraintIds[awakeHeads() + h], 0xFFFFFFFFu);
}

// Listed colliders at rest and not in R: pushed onto their bucket's chain.
[shader("compute")]
[numthreads(64, 1, 1)]
void rest_push(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint hdr = restBase();
    if (li == 0u) ASTORE(gConstraintIds[hdr + REST_DIRTY_WORD], 0u);
    uint c = awakeCollider(li);
    if (ALOAD(gConstraintIds[restAtBase() + c]) != REST_AT_NONE) return;
    Collider col = gColliders[c];
    if (!restSleeperNow(col)) return;
    uint level;
    uint h = restBucket(c, col, gBodies[col.body], level);
    if (h == REST_AT_LISTED) return; // marked above (or a rebuild is due)
    uint prev;
    AXCHG3(gConstraintIds[restHeads() + h], c, prev);
    ASTORE(gConstraintIds[restLinks() + c], prev);
    ASTORE(gConstraintIds[restAtBase() + c], h);
    if (restUniform())
    {
        AADD2(gConstraintIds[hdr], 1u);
        AMAX2(gConstraintIds[hdr + REST_U_HALF], asuint(boxHalfMax(gAabbs[HOST_S(1) + c])));
    }
    else AOR2(gConstraintIds[hdr], 1u << level);
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
    awakeA = (awakeA && canWake(ba)) || kinematicMoving(ba);
    awakeB = (awakeB && canWake(bb)) || kinematicMoving(bb);
    if (!awakeA && !awakeB) return;
    float3 vRel = ba.linearVel - bb.linearVel;
    float wRel = length(ba.angularVel) + length(bb.angularVel);
    float thresh = PARAMS.sleepLin;
    if (wakeOnTouch())
    {
        // Any touch from an awake body wakes: a sleeper is never a wall to a slow one.
    }
    else if (pointSpeedSleep())
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
// Inlined: called (as WGSL lowers it, with its arguments by pointer), it corrupted its
// caller's candidate pair in narrowphase_fresh natively (wgpu on Metal; Chrome's Dawn was
// not seen to): with
// a CCD body in the scene (look-ahead on), the split narrowphase emitted resting box-plane
// manifolds and rows as colliders (0, 0); the carry then dropped the pair, and a box woken
// later had no ground contact and fell through the floor. Same code, inlined.
[ForceInline]
BoxManifold closingPoints(BoxManifold points, BodyState ba, Collider ca, BodyState bb, Collider cb)
{
    bool moveA = bodyMovingOrKinematic(ba);
    bool moveB = bodyMovingOrKinematic(bb);
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
    if (lookAheadOn()) points = closingPoints(points, ba, ca, bb, cb);
    emitManifoldRows(pair, ba, bb, ca, cb, points, margin, false, ba, bb);
}

// A CCD body's winning pair (ccd_emit): points found at the time-of-impact
// pose, which is the step-start pose moved by shiftA / shiftB. Anchors come
// from that pose; every row's step-start offset is then taken at x⁻, so C0 is
// the gap along the path (kept whole like any look-ahead gap) and the solver
// stops the body at the TOI. No closing-point filter: they close by construction.
void emitToiManifold(
    CandidatePair pair, BodyState ba, BodyState bb, Collider ca, Collider cb,
    BoxManifold points, float3 shiftA, float3 shiftB)
{
    BodyState aT = ba;
    BodyState bT = bb;
    aT.position += shiftA;
    bT.position += shiftB;
    emitManifoldRows(pair, aT, bT, ca, cb, points, PARAMS.contactOffset, true, ba, bb);
}

// Rows from bodies (rowA, rowB); `toi` re-evaluates their offsets at the
// step-start bodies (ba, bb), which also wake.
void emitManifoldRows(
    CandidatePair pair, BodyState rowA, BodyState rowB, Collider ca, Collider cb,
    BoxManifold points, float margin, bool toi, BodyState ba, BodyState bb)
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
        Contact row = makeContactRow(pair, rowA, rowB, ca, cb, nrm, pt.pA, pt.pB, pt.feature, margin, mu);
        if (toi) row = refreshReusedContact(row, ba, bb);
        // A time-of-impact contact with a body that cannot move this step
        // (static, kinematic, or asleep with its wake deferred) starts rigid
        // (k_max), not at the cold-start floor (m/h²) that ramps up over the
        // iterations: against the whole step's inertial target (a 576 m/s ball
        // is 4.8 m past the wall at 120 Hz) a soft start let the ball through
        // within the step. Against a free dynamic body it starts as usual: a
        // rigid row there made a 21 t ball yield to a 0.25 kg brick (it rode up
        // over the wall); ccd_respond exchanges the momentum, and the
        // post-solve sweep check (ccd_validate) stops any crossing of a still
        // body. A CCD body's other contacts start as usual too.
        // A body a fast CCD body strikes is solved this step even if it starts
        // it asleep (this contact wakes it; ccd_unhold lifts the deferral): a
        // rigid row between a 200 t ball and a 20 kg brick both solved took
        // 5-9 m/s off the ball per brick that the brick never received.
        bool heldA = !bodyIsDynamic(ba) || ((bodyIsSleeping(ba) || wakePending(ba)) && !ccdUnholds(bb));
        bool heldB = !bodyIsDynamic(bb) || ((bodyIsSleeping(bb) || wakePending(bb)) && !ccdUnholds(ba));
        if (toi && row.c0N >= 0.0 && (heldA || heldB)) row.kN = PARAMS.kMaxN;
        STORE_NEW_CONTACT(curOff + cStart + k, mslot, row);
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
    // Convex params.xyz are (tagged base, scale, 0); params.w bounds the hull
    // (a heightfield's params.w bounds the whole field).
    if (col.shape == SHAPE_CONVEX || col.shape == SHAPE_HEIGHTFIELD)
        return max(col.params.w, 1e-6);
    float3 h = col.params.xyz;
    return max(2.0 * min(h.x, min(h.y, h.z)), 1e-6);
}

// Farthest point of a collider from its body's origin (centre of mass).
float colliderExtent(Collider col)
{
    float r = col.shape == SHAPE_SPHERE ? col.params.x
        : col.shape == SHAPE_CAPSULE ? col.params.x + col.params.y
        : (col.shape == SHAPE_CONVEX || col.shape == SHAPE_HEIGHTFIELD) ? col.params.w
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
// Look-ahead semantics are on for the whole scene (bounds sweeps, closing
// points, positive gaps kept whole) when the flag is set or a body is CCD or
// kinematic (their contacts are speculative by construction).
bool lookAheadOn() { return (PARAMS.flags & FLAG_SPEC_VELOCITY) != 0u || sceneHasCcd() || sceneHasKinematic(); }

// gCandidates[i]._pad of a CCD pair only its deflection pad found (ccd_pick):
// asuint(1.0) + 2, above every TOI tag (asuint(t) + 1, t <= 1), bit 30 clear.
static const uint CCD_BEYOND = 0x3F800002u;

// A candidate pair whose bodies' bounding spheres stay apart along the CCD
// body's straight sweep (relative to the other) up to `fraction` of it: found
// only through the deflection pad (buildColliderBounds), or beyond the first
// impact, so it needs no contact row.
bool ccdApartAlongSweep(Collider ca, BodyState ba, Collider cb, BodyState bb, float fraction)
{
    bool ccdA = bodyIsCcd(ba), ccdB = bodyIsCcd(bb);
    if (!ccdA && !ccdB) return false;
    // A plane has no bounding sphere.
    if (ca.shape == SHAPE_PLANE || cb.shape == SHAPE_PLANE) return false;
    float3 pa = transformPoint(ba.position, ba.orientation, ca.localPos);
    float3 pb = transformPoint(bb.position, bb.orientation, cb.localPos);
    float3 va = bodyMovingOrKinematic(ba) ? ba.linearVel : float3(0, 0, 0);
    float3 vb = bodyMovingOrKinematic(bb) ? bb.linearVel : float3(0, 0, 0);
    float3 d = (va - vb) * (PARAMS.dt * fraction);
    float h = PARAMS.dt;
    float spin = (length(ba.angularVel) * colliderExtent(ca) + length(bb.angularVel) * colliderExtent(cb)) * h;
    float reach = colliderExtent(ca) + colliderExtent(cb) + spin + abs(PARAMS.gravityY) * h * h + 2.0 * PARAMS.contactOffset;
    // Closest approach of b's centre to a's centre swept by d.
    float3 w = pb - pa;
    float t = clamp(dot(w, d) / max(dot(d, d), 1e-20), 0.0, 1.0);
    return length(w - d * t) > reach;
}

// Farthest the solve moves a CCD body off its straight sweep in one step: the
// primal trust region (trustLin · 0.5 per iteration) over the iterations
// (primalApply holds a CCD body's sideways steps to it and its position along
// the sweep to the swept segment, padded by it), at most `sweep` (the bound for
// a CCD body's broadphase pad).
float ccdDeflectionBound(float sweep)
{
    float trust = float(PARAMS.iterations) * 0.5 * PARAMS.trustLin;
    return trust > 0.0 ? min(sweep, trust) : sweep;
}

// How far any point of collider `c` on body `b` moves this step:
// (|v| + |ω|·r)·h + |g|·h², from the velocity at the start of the step. Motion
// within the contact depth (contactOffset) is already caught by ordinary
// contacts, so slower colliders get no look-ahead and rest exactly as before.
float colliderReach(BodyState b, Collider c)
{
    // CCD bodies sweep with or without the global flag, and are never capped:
    // their reach is the whole motion (row 1 of the CCD audit).
    bool ccd = bodyIsCcd(b);
    bool kin = kinematicMoving(b);
    if (((PARAMS.flags & FLAG_SPEC_VELOCITY) == 0u && !ccd && !kin) || !(bodyMoving(b) || kin)) return 0.0;
    // Spin pads the box all round (|ω|·r·h); with the rotation limit
    // (FLAG_MAX_ROTATION) that is at most 0.785 r. Without spin, tumbling
    // debris jammed into neighbours (7x7 grid: 3,869 awake at 30 s vs 6).
    float h = PARAMS.dt;
    float linear = length(b.linearVel) * h;
    float motion = linear + spinSweep(b, c);
    if (motion <= PARAMS.contactOffset) return 0.0;
    float reach = motion + abs(PARAMS.gravityY) * h * h;
    return ccd || kin ? reach : min(reach, specReachMax());
}

