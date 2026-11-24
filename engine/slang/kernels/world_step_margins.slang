// How far rotation this step moves any point of collider `c`: |ω|·h·r, but
// never more than its diameter (2r: a turn cannot carry a point farther), and
// nothing for a sphere about its body's centre (it fills the same space). A
// CCD ball spun to 7,500 rad/s by contact friction had a box past the fast
// walk's cell limit.
float spinSweep(BodyState b, Collider c)
{
    if (c.shape == SHAPE_SPHERE && dot(c.localPos, c.localPos) < 1e-12) return 0.0;
    return min(length(b.angularVel) * PARAMS.dt, 2.0) * colliderExtent(c);
}
// Reach of a hull or box collider, cached with its world OBB by build_aabbs.
// Bounds and early-outs use these per-collider reaches (a conservative sweep).
float cachedReach(uint c) { return gWorldObbs[c].center.w; }
// How far the pair can close this step: relative speed plus both spins, so
// pieces carried along together (debris pushed by the ball) get no look-ahead.
float pairReach(BodyState ba, Collider ca, BodyState bb, Collider cb)
{
    bool uncapped = bodyIsCcd(ba) || bodyIsCcd(bb) || kinematicMoving(ba) || kinematicMoving(bb);
    if ((PARAMS.flags & FLAG_SPEC_VELOCITY) == 0u && !uncapped) return 0.0;
    bool moveA = bodyMovingOrKinematic(ba);
    bool moveB = bodyMovingOrKinematic(bb);
    float3 vA = moveA ? ba.linearVel : float3(0, 0, 0);
    float3 vB = moveB ? bb.linearVel : float3(0, 0, 0);
    // Spin counts here (the narrowphase margin): closingPoints keeps only the
    // points whose ω×r velocity actually closes the gap. The broadphase sweep
    // stays linear (colliderReach), so spin does not flood the candidates.
    float spin = (moveA ? spinSweep(ba, ca) : 0.0) + (moveB ? spinSweep(bb, cb) : 0.0);
    float h = PARAMS.dt;
    float motion = length(vB - vA) * h + spin;
    if (motion <= PARAMS.contactOffset) return 0.0;
    float reach = motion + abs(PARAMS.gravityY) * h * h;
    return uncapped ? reach : min(reach, 2.0 * specReachMax());
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
// 16777216 Wake on touch: a sleeping body wakes on any contact with an awake one, not
//     only above the sleep speeds. Sleep is then invisible to what the awake bodies
//     do: a sleeper never acts as a static wall to slow debris (threshold fracture
//     reads the loads such contacts carry). With the sleep archive (sleepArchive) the
//     woken body also keeps its resting contacts' warm start.
static const uint FLAG_WAKE_ON_TOUCH = 16777216u;
bool wakeOnTouch() { return (PARAMS.flags & FLAG_WAKE_ON_TOUCH) != 0u; }
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
// A CCD body moving more than four contact offsets a step: the bodies it is
// paired with are solved in the step they wake (ccd_unhold).
bool ccdUnholds(BodyState b) { return bodyIsCcd(b) && length(b.linearVel) * PARAMS.dt > 4.0 * PARAMS.contactOffset; }
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
    if (lookAheadOn())
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
    // Heightfield pairs read their samples from the hull arena too, so they
    // take the same path (convex_classify).
    if (ca.shape != SHAPE_CONVEX && cb.shape != SHAPE_CONVEX &&
        ca.shape != SHAPE_HEIGHTFIELD && cb.shape != SHAPE_HEIGHTFIELD) return false;
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
    // A CCD body's winning pair gets its manifold at the TOI pose (ccd_emit).
    if (ccdOwnsPair(pair)) return;
    // Only the CCD body's deflection pad found this pair (ccd_pick): no contact
    // row; the post-solve sweep check still sees the pair.
    if (pair._pad == CCD_BEYOND) return;
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
    if (ccdOwnsPair(pair)) return;
    // Only the CCD body's deflection pad found this pair (ccd_pick): no contact
    // row; the post-solve sweep check still sees the pair.
    if (pair._pad == CCD_BEYOND) return;
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

