[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_mark(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    if (k >= ccdBodyCount()) return;
    uint win = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k + 2u]);
    if (win == CCD_NO_PAIR) return;
    gCandidates[win]._pad |= CCD_WIN;
    // The impact response (ccd_respond): both bodies' step-start velocities.
    CandidatePair pair = gCandidates[win];
    uint fast = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
    uint other = gColliders[pair.colliderA].body == fast ? gColliders[pair.colliderB].body : gColliders[pair.colliderA].body;
    BodyState bo = gBodies[other];
    if (!bodyIsDynamic(bo)) return;
    float3 v = gBodies[fast].linearVel;
    float3 vo = bodyIsSleeping(bo) || wakePending(bo) ? float3(0, 0, 0) : bo.linearVel;
    uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * k;
    ASTORE(gScratchAtomics[r], asuint(v.x));
    ASTORE(gScratchAtomics[r + 1u], asuint(v.y));
    ASTORE(gScratchAtomics[r + 2u], asuint(v.z));
    bool fastA = gColliders[pair.colliderA].body == fast;
    ASTORE(gScratchAtomics[r + 3u], fastA ? pair.colliderA : pair.colliderB);
    ASTORE(gScratchAtomics[r + 9u], fastA ? pair.colliderB : pair.colliderA);
    ASTORE(gScratchAtomics[r + 4u], asuint(vo.x));
    ASTORE(gScratchAtomics[r + 5u], asuint(vo.y));
    ASTORE(gScratchAtomics[r + 6u], asuint(vo.z));
    ASTORE(gScratchAtomics[r + 7u], other);
    // Held still this step (asleep, or its wake deferred): the solve treated it
    // as immovable, so the response resolves the whole impact.
    ASTORE(gScratchAtomics[r + 15u], bodyIsSleeping(bo) || wakePending(bo) ? 1u : 0u);
    ASTORE(gScratchAtomics[CCD_BODY_BASE + 4u * k + 3u], CCD_RESPOND);
}

// ---------------------------------------------------------------------------
// Post-solve sweep check (after the solve, before extract_velocity_sleep). The
// TOI contact places the first hit and the look-ahead contacts hold the rest,
// but in dense rubble the solve can still push a CCD body into a piece it was
// paired with (wedged among rigid contacts, some against bodies held still
// this step). ccd_validate casts each CCD body's solved motion this step
// relative to each candidate partner, from both step-start poses; ccd_clamp
// pulls the body back along its own path to the first contact found. Its
// bounds are padded by the farthest the solve can deflect it, so every body
// it can reach is a candidate: it never ends a step past a surface.
// ---------------------------------------------------------------------------
// Whether the segment a + s·d (s in [0, 1]) enters collider c posed at
// (pos, rot) from outside, and where (s). Exact for boxes and hulls (the slab
// test over their faces) and spheres; other shapes fall back to a GJK point
// cast. A start inside is not an entry (false).
bool segmentEnters(Collider c, float3 pos, float4 rot, float3 a, float3 d, out float s)
{
    s = 1.0;
    if (c.shape == SHAPE_HEIGHTFIELD)
    {
        Frame f; f.p = pos; f.q = rot;
        HeightFieldRef hf = heightFieldOf(c);
        float3 lo = frameToLocal(f, a);
        float3 ld = quatRotate(quatConjugate(rot), d);
        float h; float3 nn;
        if (hfSurface(hf, lo.x, lo.z, h, nn) && lo.y < h) return false; // starts under it
        float len = length(ld);
        float th;
        if (len < 1e-12 || !heightFieldRayLocal(hf, lo, ld / len, len, th, nn)) return false;
        s = th / len;
        return s > 0.0 && s <= 1.0;
    }
    float4 qi = quatConjugate(rot);
    float3 lo = quatRotate(qi, a - pos);
    float3 ld = quatRotate(qi, d);
    float tNear = -1e30, tFar = 1e30;
    if (c.shape == SHAPE_BOX)
    {
        for (int k = 0; k < 3; k++)
        {
            float h = c.params[k];
            if (abs(ld[k]) < 1e-12)
            {
                if (abs(lo[k]) > h) return false;
                continue;
            }
            float t0 = (-h - lo[k]) / ld[k];
            float t1 = (h - lo[k]) / ld[k];
            tNear = max(tNear, min(t0, t1));
            tFar = min(tFar, max(t0, t1));
            if (tNear > tFar) return false;
        }
    }
    else if (c.shape == SHAPE_CONVEX)
    {
        HullRef hr = hullOf(c);
        for (uint f = 0u; f < hr.nf; f++)
        {
            float4 pl = hullPlane(hr, f);
            float denom = dot(pl.xyz, ld);
            float num = pl.w - dot(pl.xyz, lo);
            if (abs(denom) < 1e-12)
            {
                if (num < 0.0) return false;
                continue;
            }
            float tt = num / denom;
            if (denom < 0.0) tNear = max(tNear, tt);
            else tFar = min(tFar, tt);
            if (tNear > tFar) return false;
        }
    }
    else if (c.shape == SHAPE_SPHERE)
    {
        float r = c.params.x;
        float dd = dot(ld, ld);
        if (dd < 1e-20) return false;
        float b = dot(lo, ld);
        float cc = dot(lo, lo) - r * r;
        float disc = b * b - dd * cc;
        if (disc < 0.0) return false;
        tNear = (-b - sqrt(disc)) / dd;
        tFar = (-b + sqrt(disc)) / dd;
    }
    else
    {
        Collider probe = c;
        probe.shape = SHAPE_SPHERE;
        probe.params = float4(0.0, 0.0, 0.0, 0.0);
        float mu;
        float3 n;
        // The point (a) cast against c along d, in a's frame.
        uint st = gjkCast(probe, float3(0, 0, 0), float4(0, 0, 0, 1), c, pos - a, rot, float3(0, 0, 0), d, 1.0, 1.0e-4, mu, n);
        s = mu;
        return st == CAST_HIT;
    }
    // Entry from outside within the segment.
    if (tNear <= 0.0 || tNear > 1.0 || tFar < tNear) return false;
    s = tNear;
    return true;
}

// Smallest half extent of a collider (sphere and capsule: the radius; hull:
// half its bound radius, a conservative stand-in).
float colliderThickness(Collider c)
{
    if (c.shape == SHAPE_SPHERE || c.shape == SHAPE_CAPSULE) return c.params.x;
    if (c.shape == SHAPE_BOX) return min(c.params.x, min(c.params.y, c.params.z));
    if (c.shape == SHAPE_CONVEX) return 0.5 * c.params.w;
    return 1.0e30;
}

// Step-start pose of a body: x⁻ (predict saved it), or the pose itself for a
// body the step did not move (asleep, held, static).
void stepStartPose(BodyState b, out float3 x, out float4 q)
{
    bool moved = bodyMovingOrKinematic(b) && !wakePending(b);
    x = moved ? b.initialPos : b.position;
    q = moved ? b.initialRot : b.orientation;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_validate(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= min(ALOAD(gScratchAtomics[CCD_PAIR_COUNT]), CCD_PAIR_CAP)) return;
    uint colA = ALOAD(gScratchAtomics[CCD_PAIR_BASE + 2u * i]);
    uint colB = ALOAD(gScratchAtomics[CCD_PAIR_BASE + 2u * i + 1u]);
    for (uint side = 0u; side < 2u; side++)
    {
        Collider cf = gColliders[side == 0u ? colA : colB];
        Collider co = gColliders[side == 0u ? colB : colA];
        if (cf.body == co.body || co.shape == SHAPE_PLANE) continue;
        BodyState bf = gBodies[cf.body];
        if (!bodyIsCcd(bf)) continue;
        uint slot = ccdSlotOf(cf.body, bf);
        if (slot == CCD_NO_PAIR) continue;
        BodyState bo = gBodies[co.body];
        float3 xf0; float4 qf0; stepStartPose(bf, xf0, qf0);
        float3 xo0; float4 qo0; stepStartPose(bo, xo0, qo0);
        // A body that cannot move this step (static, kinematic, asleep or held,
        // its wake deferred): any real crossing.
        // Or one the solve did not move (no row moved it: a piece past the
        // first impact a 1,152 m/s ball reached within its step, woken by
        // ccd_wake but not yet pushed): it behaved as a fixed one did, so the
        // ball stops at its surface, not with its centre at it.
        float4 qoNow = dot(bo.orientation, bo.initialRot) < 0.0 ? -bo.orientation : bo.orientation;
        bool stayedStill = bodyMovingOrKinematic(bo) && !wakePending(bo)
            && length(bo.position - bo.initialPos) < 1e-3 && dot(qoNow, bo.initialRot) > 1.0 - 1e-6;
        bool fixedThisStep = !bodyIsDynamic(bo) || bodyIsSleeping(bo) || wakePending(bo) || stayedStill;
        // An awake body (moved or not): only the CCD body's centre entering it,
        // along their relative motion, counts. A centre inside a piece is a
        // pass-through in the making (it comes out the far side once the piece
        // stops); a heavy ball's overlap with bricks it pushes, or that the
        // solve has not pushed yet, is not.
        // Relative motion of the CCD body over the step, against the other at
        // its step-start pose.
        float3 d = (bf.position - xf0) - (bo.position - xo0);
        if (dot(d, d) < 1e-10) continue;
        BodyState sf = bf; sf.position = xf0; sf.orientation = qf0;
        BodyState so = bo; so.position = xo0; so.orientation = qo0;
        float3 pf; float4 qf; colliderPose(cf, sf, pf, qf);
        float3 po; float4 qo; colliderPose(co, so, po, qo);
        float mu = 1.0;
        float3 n = float3(0, 0, 0);
        float eps = 1.0e-4 + 1.0e-5 * length(d);
        // A body that cannot move: the whole shape, cast (GJK). Already
        // touching it (resting against it in rubble), or an awake body: its
        // centre must not go in (an exact segment test, not a cast: a GJK
        // point cast grazing a box edge can stall and miss).
        uint status = CAST_OVERLAP;
        if (fixedThisStep && co.shape == SHAPE_HEIGHTFIELD)
        {
            Frame fo; fo.p = po; fo.q = qo;
            float len = length(d);
            float tc; float3 nc;
            status = CAST_MISS;
            if (heightFieldShapeCast(heightFieldOf(co), fo, cf, pf, qf, d / len, len, tc, nc))
            {
                status = tc <= 0.0 ? CAST_OVERLAP : CAST_HIT;
                mu = tc / len;
                n = -nc;
            }
        }
        else if (fixedThisStep)
            status = gjkCast(cf, float3(0, 0, 0), qf, co, po - pf, qo, float3(0, 0, 0), d, 1.0, eps, mu, n);
        bool centreOnly = status == CAST_OVERLAP;
        // An awake body the CCD body is shoving along (it moved at least half as
        // far this step): the ball is pushing it, not passing it. Pulling the
        // ball back there froze it for a tick at each brick it ploughed (a
        // visible stutter) while its velocity carried on.
        if (!fixedThisStep && 2.0 * length(bo.position - xo0) >= length(bf.position - xf0)) continue;
        if (centreOnly)
        {
            if (!segmentEnters(co, po, qo, pf, d, mu)) continue;
            status = CAST_HIT;
        }
        if (status != CAST_HIT) continue;
        // Only a real crossing: deeper than a quarter of the thinner body's
        // smallest half extent (contact depth, rolling and resting sink stay
        // the solve's; a centre crossing is far past it).
        // Against a dynamic piece, a quarter of the CCD body's own thickness if
        // held, half if awake but still (its centre still never reaches it): the thinner body's let a 3 cm roof plate or
        // fence picket stop a 1.6 m ball at 4 mm of contact sink, a pull-back
        // nearly every tick in rubble (stop motion). Static geometry: as tight.
        float depth = bodyIsDynamic(bo)
            ? max(2.0 * PARAMS.contactOffset, (stayedStill ? 0.5 : 0.25) * colliderThickness(cf))
            : max(2.0 * PARAMS.contactOffset, 0.25 * min(colliderThickness(cf), colliderThickness(co)));
        if (!centreOnly && (1.0 - mu) * dot(d, n) <= depth) continue;
        AMIN2(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot + 8u], asuint(mu));
        // First contact with a body that cannot move (static, kinematic).
        if (!bodyIsDynamic(bo)) AMIN2(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot + 10u], asuint(mu));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_clamp(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    if (k >= ccdBodyCount()) return;
    uint bits = ALOAD(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * k + 8u]);
    if (bits == CCD_TOI_NONE) return;
    uint body = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
    BodyState b = gBodies[body];
    float3 x0; float4 q0; stepStartPose(b, x0, q0);
    float3 d = b.position - x0;
    float len = length(d);
    if (len < 1e-8) return;
    // Back to the first contact, a contact offset short of it.
    float t = clamp(asfloat(bits) - PARAMS.contactOffset / len, 0.0, 1.0);
    float3 x = x0 + d * t;
    float4 q1 = dot(q0, b.orientation) < 0.0 ? -b.orientation : b.orientation;
    gBodies[body].position = x;
    gBodies[body].orientation = quatNormalize(q0 + (q1 - q0) * t);
    // Stopped by a dynamic body (held still this step, its wake deferred, or
    // pinned): it keeps the velocity of its solved motion and meets that body,
    // awake, next step. Stopped by static geometry: the clamped motion is its
    // velocity (it presses no further).
    uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * k;
    bool byStatic = ALOAD(gScratchAtomics[r + 10u]) <= bits;
    float3 v = (byStatic ? x - x0 : d) / max(PARAMS.dt, 1e-8);
    ASTORE(gScratchAtomics[r + 12u], asuint(v.x));
    ASTORE(gScratchAtomics[r + 13u], asuint(v.y));
    ASTORE(gScratchAtomics[r + 14u], asuint(v.z));
    ASTORE(gScratchAtomics[r + 11u], 1u);
    if (postStabilize()) gBodies[body].linearVel = v;
}

// The velocity ccd_clamp left for a clamped CCD body (extract_velocity_sleep
// takes it instead of the one its clamped motion gives).
bool ccdClampedVelocity(uint body, BodyState b, out float3 v)
{
    v = float3(0, 0, 0);
    if (!sceneHasCcd()) return false;
    uint slot = ccdSlotOf(body, b);
    if (slot == CCD_NO_PAIR) return false;
    uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * slot;
    if (ALOAD(gScratchAtomics[r + 11u]) == 0u) return false;
    v = asfloat(uint3(ALOAD(gScratchAtomics[r + 12u]), ALOAD(gScratchAtomics[r + 13u]), ALOAD(gScratchAtomics[r + 14u])));
    return true;
}

// Whether body `b` lies within `reach` × the CCD collider's radius, plus its own
// extent, of the swept segment this step of some CCD body moving more than four
// contact offsets a step (ccdUnholds).
bool ccdNearFastSweepAt(uint body, BodyState b, float reach, bool atStart)
{
    float own = 0.0;
    uint nc = gCounters[0].colliderCount;
    for (uint c = firstColliderOfBody(body); c < nc && gColliders[c].body == body; c++)
        own = max(own, colliderExtent(gColliders[c]));
    uint n = ccdBodyCount();
    for (uint k = 0u; k < n; k++)
    {
        uint fast = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
        BodyState bf = gBodies[fast];
        if (!ccdUnholds(bf)) continue;
        float r = 0.0;
        for (uint c = firstColliderOfBody(fast); c < nc && gColliders[c].body == fast; c++)
            r = max(r, colliderExtent(gColliders[c]));
        float3 a = atStart ? bf.position : bf.initialPos;
        float3 d = bf.linearVel * PARAMS.dt;
        float t = clamp(dot(b.position - a, d) / max(dot(d, d), 1e-20), 0.0, 1.0);
        if (length(a + d * t - b.position) <= reach * r + own) return true;
    }
    return false;
}

// Before predict_bodies a CCD body's pose is its step start; after it, its
// initialPos is (predict saved it and moved the body).
bool ccdNearFastSweepStart(uint body, BodyState b, float reach) { return ccdNearFastSweepAt(body, b, reach, true); }

// Sleepers in a fast CCD body's path wake before the narrowphase (after
// ccd_pick), so they get their contacts and are solved this step. Left asleep,
// a ball reaching a sleeping piece past its first impact had no contact with it,
// the post-solve check stopped the ball at it as at a wall, and the next tick
// at the next one: in rubble the ball stalled every tick (stop motion). Within
// 1.25 radii of the CCD collider of its swept segment, plus the piece's own size.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_wake(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sceneHasCcd() || i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b) || !bodyIsSleeping(b)) return;
    // Step start: initialPos is not yet this step's (predict saves it).
    if (!ccdNearFastSweepStart(i, b, 1.25)) return;
    gBodies[i].flags = (b.flags & ~(BODY_SLEEPING | BODY_WAKE_PENDING)) | BODY_AWAKE;
    gBodies[i].sleepTimer = 0.0;
}

// Woken near a fast CCD body: solved this step (before predict_bodies). A body
// the narrowphase wakes keeps still for the rest of its step (FLAG_DEFER_WAKE:
// solved without its supports, a woken brick in a pile started with a kick).
// But where a fast CCD body ploughs in, those bodies stood like a fixed wall
// for that step: the bricks it struck were backed by neighbours woken in the
// same step and held still, so a 200 t ball took its impulse off them (99% of
// the momentum it lost in a suburb of brick houses went into held bodies) and
// was deflected, often upward over the wall. So a body whose wake is deferred
// is solved this step when it is within reach of a CCD body moving more than
// four contact offsets a step: three of the CCD collider's radii, plus its own,
// from the CCD body's swept segment this step. A CCD slot whose impact partner
// is released no longer treats it as held (ccd_respond: the solve exchanges the
// momentum). Pure function of the step-start state: every thread agrees.
bool ccdReleasesHeld(uint body, BodyState b)
{
    if (!wakePending(b) || bodyIsSleeping(b) || !bodyIsDynamic(b)) return false;
    return ccdNearFastSweepStart(body, b, 3.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_unhold(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sceneHasCcd() || i >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[i];
    if (ccdReleasesHeld(i, b)) gBodies[i].flags = b.flags & ~BODY_WAKE_PENDING;
    if (i < ccdBodyCount())
    {
        uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * i;
        if ((ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * i + 3u]) & CCD_RESPOND) != 0u)
        {
            uint other = ALOAD(gScratchAtomics[r + 7u]);
            // Read before any thread clears the partner's flag: its own thread
            // clears only the deferred-wake bit, which ccdReleasesHeld tests.
            BodyState bo = gBodies[other];
            if (ccdReleasesHeld(other, bo) || (!bodyIsSleeping(bo) && !wakePending(bo))) ASTORE(gScratchAtomics[r + 15u], 0u);
        }
    }
}

// ---- ccd_wake by collider (World's plan; WGSL hosts) ----
// The same test as ccd_wake without a collider search per body: a
// body lies within reach × r + own of a sweep (own its largest collider extent) iff
// one of its colliders c lies within reach × r + extent(c) of it (adding to the
// largest extent gives the largest bound), so a thread per collider tests its own
// extent, and the sweeping body's r (the largest extent of its colliders) is taken
// once per CCD slot (ccd_slot_reach, or ccd_slot_reach_listed over the listed movers).
// Both scan colliders, so colliders need not be sorted by body (threshold splits
// re-parent them), and both run only while some CCD body moves more than four contact
// offsets a step (ccdUnholds; work slot 87 is empty otherwise, when ccd_wake could wake
// nothing). A body without colliders is never woken here; hosts with such bodies keep
// ccd_wake. ccd_unhold stays per body (it searches colliders only for the few bodies
// whose wake is deferred).
// The slot's r: carry word 28 (ccd_respond's partner dw, written only after the
// solve; read here in ccd_wake's place).
static const uint CCD_REACH_WORD = 28u;

// Work slot 87 (a thread per collider) when a CCD body sweeps fast; slot r words zeroed.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_reach_args(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0u) return;
    uint n = ccdBodyCount();
    bool any = false;
    for (uint k = 0u; k < n; k++)
    {
        ASTORE(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * k + CCD_REACH_WORD], 0u);
        if (ccdUnholds(gBodies[ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k])])) any = true;
    }
    writeDispatchArgs(87u, sceneHasCcd() && any ? gCounters[0].colliderCount : 0u);
}

// Largest collider extent of each fast CCD body (non-negative floats order as their bits).
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_slot_reach(uint3 tid : SV_DispatchThreadID)
{
    uint c = dispatchIndex(tid);
    if (c >= gCounters[0].colliderCount) return;
    Collider col = gColliders[c];
    BodyState b = gBodies[col.body];
    uint slot = ccdSlotOf(col.body, b);
    if (slot == CCD_NO_PAIR || !ccdUnholds(b)) return;
    AMAX2(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot + CCD_REACH_WORD], asuint(colliderExtent(col)));
}

// ccd_slot_reach over the moving colliders the awake-indexed broadphase listed (slot 81):
// a CCD slot's body was moving when it registered, so all its colliders are listed.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_slot_reach_listed(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= awakeColliderCount()) return;
    uint c = awakeCollider(li);
    Collider col = gColliders[c];
    BodyState b = gBodies[col.body];
    uint slot = ccdSlotOf(col.body, b);
    if (slot == CCD_NO_PAIR || !ccdUnholds(b)) return;
    AMAX2(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * slot + CCD_REACH_WORD], asuint(colliderExtent(col)));
}

// ccdNearFastSweepStart's test for one collider of `b` (extent `own`).
bool ccdColliderNearFastSweep(BodyState b, float reach, float own)
{
    uint n = ccdBodyCount();
    for (uint k = 0u; k < n; k++)
    {
        BodyState bf = gBodies[ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k])];
        if (!ccdUnholds(bf)) continue;
        float r = asfloat(ALOAD(gScratchAtomics[CCD_CARRY_BASE + CCD_CARRY_WORDS * k + CCD_REACH_WORD]));
        float3 a = bf.position;
        float3 d = bf.linearVel * PARAMS.dt;
        float t = clamp(dot(b.position - a, d) / max(dot(d, d), 1e-20), 0.0, 1.0);
        if (length(a + d * t - b.position) <= reach * r + own) return true;
    }
    return false;
}

// ccd_wake, a thread per collider (slot 87).
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_wake_colliders(uint3 tid : SV_DispatchThreadID)
{
    uint c = dispatchIndex(tid);
    if (c >= gCounters[0].colliderCount) return;
    Collider col = gColliders[c];
    uint i = col.body;
    BodyState b = gBodies[i];
    if (!bodyIsDynamic(b) || !bodyIsSleeping(b)) return;
    if (!ccdColliderNearFastSweep(b, 1.25, colliderExtent(col))) return;
    gBodies[i].flags = (b.flags & ~(BODY_SLEEPING | BODY_WAKE_PENDING)) | BODY_AWAKE;
    gBodies[i].sleepTimer = 0.0;
}

// ccd_unhold reading a body's record only when its awake bit is set: it changes only
// awake bodies (a deferred wake), and the slot part runs in the threads below the CCD
// slot count whatever their bit.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_unhold_bits(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sceneHasCcd() || i >= gCounters[0].bodyCount) return;
    if (awakeBit(i))
    {
        BodyState b = gBodies[i];
        if (ccdReleasesHeld(i, b)) gBodies[i].flags = b.flags & ~BODY_WAKE_PENDING;
    }
    if (i < ccdBodyCount())
    {
        uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * i;
        if ((ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * i + 3u]) & CCD_RESPOND) != 0u)
        {
            uint other = ALOAD(gScratchAtomics[r + 7u]);
            BodyState bo = gBodies[other];
            if (ccdReleasesHeld(other, bo) || (!bodyIsSleeping(bo) && !wakePending(bo))) ASTORE(gScratchAtomics[r + 15u], 0u);
        }
    }
}

// Trust region of the bodies a CCD body can strike (after predict_bodies,
// before the solve). A free body struck by a fast one takes the impact through
// the solve, a position correction of up to the closing speed times h, but the
// primal trust region lets an ordinary body move only iterations · trustLin ·
// 0.5 (1.75 m) in a step: a 300 m/s ball into a free 2 kg box (inelastic
// 214 m/s, 3.6 m) kept 85% of the momentum. So for this step every awake
// dynamic body paired with a CCD body (ccd_pick's list) gets a trust region
// covering the largest CCD travel this step, read by primalApply from its
// inertial-target record's spare word (predict_bodies just zeroed it). Every
// write is the same value, so concurrent ones agree.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_trust(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (!sceneHasCcd() || i >= min(ALOAD(gScratchAtomics[CCD_PAIR_COUNT]), CCD_PAIR_CAP)) return;
    float reach = asfloat(ALOAD(gScratchAtomics[CCD_MAX_REACH]));
    for (uint side = 0u; side < 2u; side++)
    {
        uint body = gColliders[ALOAD(gScratchAtomics[CCD_PAIR_BASE + 2u * i + side])].body;
        BodyState b = gBodies[body];
        if (!bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b) || bodyIsCcd(b)) continue;
        gAabbs[body]._pad1 = reach;
    }
}

// The primal step of body i under the trust region (trustLin · 0.5 per
// iteration). A CCD body's iterate starts at its time of impact: held to that
// trust it could advance only iterations · 0.175 m (1.75 m) in a step, so a
// fast ball pushing a free box lost the rest of its motion and with it the
// momentum. So along its velocity it may move as far as its step's travel
// over the iterations, its position along the sweep staying within the swept
// segment padded by the ordinary bound (ccdDeflectionBound), and across it the
// ordinary trust still holds: its broadphase pad stays a hard bound. A body a
// CCD body can strike gets that reach all round (ccd_trust).
float3 ccdTrustStep(uint i, BodyState b, float3 dlin)
{
    float base = PARAMS.trustLin * 0.5;
    float iters = max(float(PARAMS.iterations), 1.0);
    float sp = length(b.linearVel);
    if (sceneHasCcd() && bodyIsCcd(b) && sp > 1e-6)
    {
        float3 u = b.linearVel / sp;
        float along = dot(dlin, u);
        float3 perp = dlin - u * along;
        float pl = length(perp);
        if (pl > base) perp *= base / pl;
        float reach = sp * PARAMS.dt;
        along = clamp(along, -max(base, 2.0 * reach / iters), max(base, 2.0 * reach / iters));
        // Position along the sweep: within [x⁻, x⁻ + v·h], padded.
        float pad = ccdDeflectionBound(reach);
        float s0 = dot(b.position - b.initialPos, u);
        along = clamp(s0 + along, min(s0, -pad), max(s0, reach + pad)) - s0;
        return perp + u * along;
    }
    float maxLin = base;
    if (sceneHasCcd() && !bodyIsCcd(b))
        maxLin = max(maxLin, 2.0 * gAabbs[i]._pad1 / iters);
    float len = length(dlin);
    if (len > maxLin && len > 1e-12) dlin *= maxLin / len;
    return dlin;
}

// ccd slot flags word (+3): set by ccd_mark when the TOI partner is dynamic.
static const uint CCD_RESPOND = 1u;

// ---------------------------------------------------------------------------
// Impact response (after the sleep pass). A rigid TOI contact stops a CCD body
// at its partner's surface, but the solve barely passes momentum through a
// rigid contact in one step's iterations, and a partner held still this step
// (asleep, its wake deferred) or pinned by such neighbours takes none: the
// striker would stop dead. So each CCD body that struck a dynamic body gets a
// perfectly inelastic, momentum-conserving exchange along the TOI normal from
// both bodies' step-start velocities, with both bodies' rotational effective
// mass (the partner also spins, from the contact point), and a held partner
// wakes with its new velocity. Against a free partner the solve's result is
// already close, so this changes little; kinetic energy never grows.
//
// Two passes, one thread per CCD slot: ccd_respond computes each slot's
// velocity changes from the bodies as the solve left them (reads only);
// ccd_respond_apply has one owner thread per affected body (its own slot if it
// struck, else the lowest slot that struck it), which sums every change
// addressed to it in slot order. (One thread looping over the slots cost 3 ms
// at 512 strikers.)
