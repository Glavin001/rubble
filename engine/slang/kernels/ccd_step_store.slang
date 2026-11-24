// ---------------------------------------------------------------------------
static const uint CCD_RESP = 16u; // carry words: flag, striker dv, dw, partner dv, dw

void ccdStoreVec(uint at, float3 v)
{
    ASTORE(gScratchAtomics[at], asuint(v.x));
    ASTORE(gScratchAtomics[at + 1u], asuint(v.y));
    ASTORE(gScratchAtomics[at + 2u], asuint(v.z));
}
float3 ccdLoadVec(uint at)
{
    return asfloat(uint3(ALOAD(gScratchAtomics[at]), ALOAD(gScratchAtomics[at + 1u]), ALOAD(gScratchAtomics[at + 2u])));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_respond(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    if (!sceneHasCcd() || k >= ccdBodyCount()) return;
    uint r = CCD_CARRY_BASE + CCD_CARRY_WORDS * k;
    ASTORE(gScratchAtomics[r + CCD_RESP], 0u);
    if ((ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k + 3u]) & CCD_RESPOND) == 0u) return;
    uint fast = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
    float3 v0 = ccdLoadVec(r);
    float3 vo0 = ccdLoadVec(r + 4u);
    uint other = ALOAD(gScratchAtomics[r + 7u]);
    // The winning pair's colliders (ccd_mark): the candidate buffer is the
    // constraint graph's by now.
    Collider cf = gColliders[ALOAD(gScratchAtomics[r + 3u])];
    Collider co = gColliders[ALOAD(gScratchAtomics[r + 9u])];
    BodyState bf = gBodies[fast];
    BodyState bo = gBodies[other];
    if (!bodyIsDynamic(bo) || !bodyIsDynamic(bf)) return;
    // The TOI normal again, from the step-start poses (a held partner did not
    // move) along the step's relative motion.
    BodyState bf0 = bf;
    bf0.position = bf.initialPos;
    bf0.orientation = bf.initialRot;
    float3 rel = v0 * PARAMS.dt;
    float t;
    float3 nrm;
    if (ccdCast(cf, bf0, co, bo, rel, t, nrm) != CAST_HIT || dot(nrm, nrm) < 0.5) nrm = normalize(rel);
    float closing = dot(v0 - vo0, nrm);
    if (closing <= 0.0) return;
    // The perfectly inelastic impulse along n at the contact point, with both
    // bodies' rotational effective mass (a linear-only impulse with the torque
    // added on top over-counts it off centre: struck bricks spun to 15,000 rad/s).
    float3 pf; float4 qf; colliderPose(cf, bf, pf, qf);
    float3 at = shapeSupport(cf, pf, qf, nrm);
    float3 rfn = cross(at - bf.position, nrm);
    float3 ron = cross(at - bo.position, nrm);
    float kf = bf.invMass + dot(rfn, applyInvInertia(bf, rfn));
    float ko = bo.invMass + dot(ron, applyInvInertia(bo, ron));
    float J = closing / max(kf + ko, 1e-20);
    // Each body's normal speed is only ever raised over the solve's result:
    // the striker to what the impulse leaves it, the partner to what it gives.
    // Only a partner held still this step needs it: the solve treated it as
    // immovable (the striker stopped against it as against a wall, its motion
    // across the normal too) and it took no momentum. A free partner's
    // contacts already exchange momentum in the solve; adjusting only one
    // side's speed there broke conservation (an edge hit that spun the box
    // into the ball again lost 57% of the momentum).
    if (ALOAD(gScratchAtomics[r + 15u]) == 0u) return;
    // The whole frictionless impact from the step-start velocities.
    ccdStoreVec(r + CCD_RESP + 1u, (v0 - nrm * (J * bf.invMass)) - bf.linearVel);
    ccdStoreVec(r + CCD_RESP + 4u, -applyInvInertia(bf, rfn * J));
    ccdStoreVec(r + CCD_RESP + 7u, (vo0 + nrm * (J * bo.invMass)) - bo.linearVel);
    ccdStoreVec(r + CCD_RESP + 10u, applyInvInertia(bo, ron * J));
    ASTORE(gScratchAtomics[r + CCD_RESP], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_respond_apply(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    uint n = ccdBodyCount();
    if (!sceneHasCcd() || k >= n) return;
    uint rk = CCD_CARRY_BASE + CCD_CARRY_WORDS * k;
    uint fast = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * k]);
    bool struck = ALOAD(gScratchAtomics[rk + CCD_RESP]) != 0u;
    uint other = ALOAD(gScratchAtomics[rk + 7u]);
    // Owner of `other` (it struck nothing itself, so no slot owns it as a
    // striker): this slot, if no lower slot struck it too.
    bool ownsOther = struck;
    for (uint j = 0u; j < n && ownsOther; j++)
    {
        uint rj = CCD_CARRY_BASE + CCD_CARRY_WORDS * j;
        if (ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * j]) == other) ownsOther = false;
        else if (j < k && ALOAD(gScratchAtomics[rj + CCD_RESP]) != 0u && ALOAD(gScratchAtomics[rj + 7u]) == other) ownsOther = false;
    }
    // This slot's striker always belongs to this thread; `other` too if owned.
    for (uint pass = 0u; pass < 2u; pass++)
    {
        uint b = pass == 0u ? fast : other;
        if (pass == 1u && !ownsOther) break;
        float3 dv = float3(0, 0, 0), dw = float3(0, 0, 0);
        bool any = false;
        for (uint j = 0u; j < n; j++)
        {
            uint rj = CCD_CARRY_BASE + CCD_CARRY_WORDS * j;
            if (ALOAD(gScratchAtomics[rj + CCD_RESP]) == 0u) continue;
            if (ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * j]) == b)
            {
                dv += ccdLoadVec(rj + CCD_RESP + 1u);
                dw += ccdLoadVec(rj + CCD_RESP + 4u);
                any = true;
            }
            if (ALOAD(gScratchAtomics[rj + 7u]) == b)
            {
                dv += ccdLoadVec(rj + CCD_RESP + 7u);
                dw += ccdLoadVec(rj + CCD_RESP + 10u);
                any = true;
            }
        }
        if (!any) continue;
        // This thread owns body b: one load, one store.
        BodyState bb = gBodies[b];
        bb.linearVel += dv;
        // Under the per-step rotation limit as extract_velocity_sleep is (it ran
        // before this pass): an edge hit spun a box to 6 rad a step.
        float3 w = bb.angularVel + dw;
        if ((PARAMS.flags & FLAG_MAX_ROTATION) != 0u)
        {
            float wMax = MAX_ROTATION / max(PARAMS.dt, 1e-8);
            if (length(w) > wMax) w *= wMax / length(w);
        }
        bb.angularVel = w;
        if (pass == 1u)
        {
            bb.flags = (bb.flags & ~(BODY_SLEEPING | BODY_WAKE_PENDING)) | BODY_AWAKE;
            bb.sleepTimer = 0.0;
        }
        gBodies[b] = bb;
    }
}

// World inverse inertia times v (diagonal in the body frame).
float3 applyInvInertia(BodyState b, float3 v)
{
    float3 local = quatRotate(quatConjugate(b.orientation), v);
    return quatRotate(b.orientation, local * b.invInertia);
}

// The narrowphase leaves each CCD body's winning pair to ccd_emit.
bool ccdOwnsPair(CandidatePair pair)
{
    return (pair._pad & (CCD_WIN | CONVEX_DEFER)) == CCD_WIN;
}

// Fraction of its motion a CCD body is predicted forward (1: no TOI).
float ccdStartFraction(uint body, BodyState b)
{
    if (!sceneHasCcd()) return 1.0;
    uint slot = ccdSlotOf(body, b);
    if (slot == CCD_NO_PAIR) return 1.0;
    uint bits = ALOAD(gScratchAtomics[CCD_BODY_BASE + 4u * slot + 1u]);
    return bits == CCD_TOI_NONE ? 1.0 : clamp(asfloat(bits), 0.0, 1.0);
}

// Support features of a collider at pose (pos, rot) along n: the points whose
// support value is within tol of the maximum (a face, an edge or a vertex),
// at most 8.
void supportFeature(Collider c, float3 pos, float4 rot, float3 n, float tol, inout float3 pts[8], inout uint count)
{
    count = 0u;
    if (c.shape == SHAPE_SPHERE)
    {
        pts[0] = pos + n * c.params.x;
        count = 1u;
        return;
    }
    if (c.shape == SHAPE_CAPSULE)
    {
        float3 axis = quatRotate(rot, float3(0, 1, 0));
        float3 e0 = pos - axis * c.params.y, e1 = pos + axis * c.params.y;
        float d0 = dot(e0, n), d1 = dot(e1, n);
        if (abs(d0 - d1) <= tol) { pts[0] = e0 + n * c.params.x; pts[1] = e1 + n * c.params.x; count = 2u; }
        else { pts[0] = (d0 > d1 ? e0 : e1) + n * c.params.x; count = 1u; }
        return;
    }
    float best = -1e30;
    if (c.shape == SHAPE_BOX)
    {
        float3 h = c.params.xyz;
        for (uint k = 0u; k < 8u; k++)
        {
            float3 l = float3((k & 1u) != 0u ? h.x : -h.x, (k & 2u) != 0u ? h.y : -h.y, (k & 4u) != 0u ? h.z : -h.z);
            best = max(best, dot(pos + quatRotate(rot, l), n));
        }
        for (uint k = 0u; k < 8u; k++)
        {
            float3 l = float3((k & 1u) != 0u ? h.x : -h.x, (k & 2u) != 0u ? h.y : -h.y, (k & 4u) != 0u ? h.z : -h.z);
            float3 w = pos + quatRotate(rot, l);
            if (dot(w, n) >= best - tol && count < 8u) { pts[count] = w; count++; }
        }
        return;
    }
    HullRef h = hullOf(c);
    for (uint k = 0u; k < h.nv; k++) best = max(best, dot(pos + quatRotate(rot, hullVertex(h, k)), n));
    for (uint k = 0u; k < h.nv; k++)
    {
        float3 w = pos + quatRotate(rot, hullVertex(h, k));
        if (dot(w, n) >= best - tol && count < 8u) { pts[count] = w; count++; }
    }
}

// Whether p lies within a collider's local bounds at pose (pos, rot), padded.
bool pointNearCollider(float3 p, Collider c, float3 pos, float4 rot, float pad)
{
    float3 l = quatRotate(quatConjugate(rot), p - pos);
    if (c.shape == SHAPE_SPHERE) return length(l) <= c.params.x + pad;
    if (c.shape == SHAPE_CAPSULE)
        return length(float3(l.x, max(abs(l.y) - c.params.y, 0.0), l.z)) <= c.params.x + pad;
    if (c.shape == SHAPE_BOX) return all(abs(l) <= c.params.xyz + pad);
    if (c.shape == SHAPE_HEIGHTFIELD)
    {
        float3 mn, mx;
        hfLocalBounds(heightFieldOf(c), mn, mx);
        return all(l >= mn - pad) && all(l <= mx + pad);
    }
    HullRef h = hullOf(c);
    return all(abs(l - hullLocalCenter(h)) <= hullLocalHalf(h) + pad);
}

// One thread per candidate (work slot 64); the CCD winners (ccd_mark) build
// their manifold at the TOI pose and emit it as a speculative contact whose
// rows hold the gap along the path.
[shader("compute")]
[numthreads(64, 1, 1)]
void ccd_emit(uint3 tid : SV_DispatchThreadID)
{
    uint win = dispatchIndex(tid);
    if (win >= min(ALOAD(gCounters[0].candidateCount), HOST_S(2))) return;
    CandidatePair pair = gCandidates[win];
    if (!ccdOwnsPair(pair)) return;
    Collider ca = gColliders[pair.colliderA];
    Collider cb = gColliders[pair.colliderB];
    BodyState ba = gBodies[ca.body];
    BodyState bb = gBodies[cb.body];
    bool fastIsA = ccdFastIsA(ba, ca.body, bb, cb.body);
    Collider cf = fastIsA ? ca : cb;
    Collider co = fastIsA ? cb : ca;
    BodyState bf = fastIsA ? ba : bb;
    BodyState bo = fastIsA ? bb : ba;
    float3 r = ccdRelativeMotion(bf, bo);
    float t;
    float3 n;
    // The narrowphase left this pair to us (ccd_mark): it must get a contact.
    // The cast that found it (ccd_toi) ran in another kernel; a touching pair
    // at TOI 0 is borderline in f32 and can come out differently here, so a
    // miss falls back to the TOI ccd_toi stored and a stopper normal along the
    // relative motion (the pair would otherwise have no contact at all).
    if (dot(r, r) < 1e-20 || ccdCast(cf, bf, co, bo, r, t, n) != CAST_HIT)
    {
        t = clamp(asfloat((pair._pad & ~CCD_WIN) - 1u), 0.0, 1.0);
        n = dot(r, r) > 1e-20 ? normalize(r) : float3(0, -1, 0);
    }
    // TOI pose: the fast body moved by t·r relative to the other (at x⁻).
    BodyState bfT = bf;
    bfT.position += r * t;
    float3 pf; float4 qf; colliderPose(cf, bfT, pf, qf);
    float tol = max(2.0 * PARAMS.contactOffset, 0.005);
    float3 fp[8];
    uint nf = 0u;
    supportFeature(cf, pf, qf, n, tol, fp, nf);
    BoxManifold mf;
    mf.normal = n; // fast → other
    mf.count = 0;
    if (co.shape == SHAPE_PLANE)
    {
        float3 pn = normalize(co.params.xyz);
        for (uint q = 0u; q < nf; q++)
        {
            mf.pts[mf.count].pA = fp[q];
            mf.pts[mf.count].pB = fp[q] - pn * (dot(pn, fp[q]) - co.params.w);
            mf.pts[mf.count].feature = 0x0CCD0000u | q;
            mf.count++;
        }
    }
    else if (co.shape == SHAPE_HEIGHTFIELD)
    {
        // The ordinary heightfield manifold at the TOI pose (fast = A), tagged
        // as time-of-impact rows so they never warm-start the next step.
        BodyState boT = bo;
        mf = heightFieldPairManifold(cf, bfT, co, boT, tol);
        for (int q = 0; q < mf.count; q++) mf.pts[q].feature = 0x0CCD0000u | uint(q);
        if (mf.count == 0)
        {
            mf.normal = n;
            mf.pts[0].pA = shapeSupport(cf, pf, qf, n);
            mf.pts[0].pB = mf.pts[0].pA;
            mf.pts[0].feature = 0x0CCF0000u;
            mf.count = 1;
        }
    }
    else
    {
        float3 po; float4 qo; colliderPose(co, bo, po, qo);
        float3 sO = shapeSupport(co, po, qo, -n);
        // The fast body's support points that land on the other's support
        // plane inside its bounds; else the other's points on the fast body's;
        // else the single touching pair of support points.
        for (uint q = 0u; q < nf; q++)
        {
            float3 pB = fp[q] + n * dot(n, sO - fp[q]);
            if (pointNearCollider(pB, co, po, qo, tol))
            {
                mf.pts[mf.count].pA = fp[q];
                mf.pts[mf.count].pB = pB;
                mf.pts[mf.count].feature = 0x0CCD0000u | q;
                mf.count++;
            }
        }
        if (mf.count == 0)
        {
            float3 op[8];
            uint no = 0u;
            supportFeature(co, po, qo, -n, tol, op, no);
            float3 sF = shapeSupport(cf, pf, qf, n);
            for (uint q = 0u; q < no; q++)
            {
                float3 pA = op[q] - n * dot(n, op[q] - sF);
                if (pointNearCollider(pA, cf, pf, qf, tol) && mf.count < 8)
                {
                    mf.pts[mf.count].pA = pA;
                    mf.pts[mf.count].pB = op[q];
                    mf.pts[mf.count].feature = 0x0CCE0000u | q;
                    mf.count++;
                }
            }
        }
        if (mf.count == 0)
        {
            mf.pts[0].pA = shapeSupport(cf, pf, qf, n);
            mf.pts[0].pB = sO;
            mf.pts[0].feature = 0x0CCF0000u;
            mf.count = 1;
        }
    }
    // Diagnostics (ball traces): the TOI normal, fast → other, in the slot's
    // carry words 29-31 (nothing in the step reads them).
    uint slotF = ccdSlotOf(fastIsA ? ca.body : cb.body, bf);
    if (slotF != CCD_NO_PAIR) ccdStoreVec(CCD_CARRY_BASE + CCD_CARRY_WORDS * slotF + 29u, n);
    // Pair order: rows run A → B.
    if (!fastIsA) mf = flipManifold(mf);
    emitToiManifold(pair, ba, bb, ca, cb, mf, fastIsA ? r * t : float3(0, 0, 0), fastIsA ? float3(0, 0, 0) : r * t);
}

// ---------------------------------------------------------------------------
// GPU impulses (row 4): [body, mode, J.xyz, point.xyz] records. Mode 0 adds an
// impulse at a world point (dynamic bodies); mode 1 sets a kinematic body's
// velocity (J = linear, point = angular). One thread per record; the first
// record of each body sums its run in record order (deterministic, no float
// atomics) and wakes the body.
// ---------------------------------------------------------------------------
static const uint IMPULSE_AT_POINT = 0u;
static const uint IMPULSE_SET_VELOCITY = 1u;
static const uint IMPULSE_LINEAR_ANGULAR = 2u; // J through the centre, angular impulse in `point`

// Mode word: mode (bits 0..7) | key << 8. A body's records are summed in
// ascending key order (host records: their queue index; projectile hits:
// after every host record, by projectile then hit), whatever order the
// atomic appends left them in. PASS_U 0 applies [0, count) at the start of
// the step (host records); PASS_U 1, after projectile_step, applies the
// records past the host's (IMPULSE_FIRST).
uint impulseKey(uint r) { return ALOAD(gScratchAtomics[r + 1u]) >> 8; }

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_impulses(uint3 tid : SV_DispatchThreadID)
{
    uint k = dispatchIndex(tid);
    uint count = ALOAD(gScratchAtomics[IMPULSE_COUNT]);
    if (count > IMPULSE_CAP)
    {
        if (k == 0u) AOR2(gCounters[0].overflowFlags, OVERFLOW_IMPULSES);
        count = IMPULSE_CAP;
    }
    uint first = PASS_U == 1u ? min(ALOAD(gScratchAtomics[IMPULSE_FIRST]), count) : 0u;
    k += first;
    if (k >= count) return;
    uint rk = IMPULSE_BASE + IMPULSE_WORDS * k;
    uint body = ALOAD(gScratchAtomics[rk]);
    uint myKey = impulseKey(rk);
    // Only the body's lowest-key record sums the run.
    for (uint q = first; q < count; q++)
    {
        uint rq = IMPULSE_BASE + IMPULSE_WORDS * q;
        if (q != k && ALOAD(gScratchAtomics[rq]) == body && impulseKey(rq) < myKey) return;
    }
    if (body >= gCounters[0].bodyCount) return;
    BodyState b = gBodies[body];
    float3 dv = float3(0, 0, 0);
    float3 dw = float3(0, 0, 0);
    bool setVel = false;
    float3 setLin = b.linearVel, setAng = b.angularVel;
    uint key = myKey;
    uint at = k;
    for (uint n = first; n < count; n++)
    {
        uint r = IMPULSE_BASE + IMPULSE_WORDS * at;
        uint mode = ALOAD(gScratchAtomics[r + 1u]) & 255u;
        float3 J = asfloat(uint3(ALOAD(gScratchAtomics[r + 2u]), ALOAD(gScratchAtomics[r + 3u]), ALOAD(gScratchAtomics[r + 4u])));
        float3 P = asfloat(uint3(ALOAD(gScratchAtomics[r + 5u]), ALOAD(gScratchAtomics[r + 6u]), ALOAD(gScratchAtomics[r + 7u])));
        if (mode == IMPULSE_SET_VELOCITY)
        {
            setVel = true;
            setLin = J;
            setAng = P;
        }
        else if (bodyIsDynamic(b) && mode == IMPULSE_LINEAR_ANGULAR)
        {
            dv += J * b.invMass;
            float3 local = quatRotate(quatConjugate(b.orientation), P);
            dw += quatRotate(b.orientation, local * b.invInertia);
        }
        else if (bodyIsDynamic(b))
        {
            dv += J * b.invMass;
            float3 torque = cross(P - b.position, J);
            float3 local = quatRotate(quatConjugate(b.orientation), torque);
            dw += quatRotate(b.orientation, local * b.invInertia);
        }
        // Next record of this body: the smallest key above the current one.
        uint nextKey = 0xFFFFFFFFu;
        uint next = count;
        for (uint q = first; q < count; q++)
        {
            uint rq = IMPULSE_BASE + IMPULSE_WORDS * q;
            if (ALOAD(gScratchAtomics[rq]) != body) continue;
            uint kq = impulseKey(rq);
            if (kq > key && kq < nextKey) { nextKey = kq; next = q; }
        }
        if (next == count) break;
        key = nextKey;
        at = next;
    }
    if (setVel && !bodyIsDynamic(b))
    {
        gBodies[body].linearVel = setLin;
        gBodies[body].angularVel = setAng;
        return;
    }
    if (setVel)
    {
        dv += setLin - b.linearVel;
        dw += setAng - b.angularVel;
    }
    if (!bodyIsDynamic(b)) return;
    gBodies[body].linearVel = b.linearVel + dv;
    gBodies[body].angularVel = b.angularVel + dw;
    gBodies[body].flags = (b.flags & ~(BODY_SLEEPING | BODY_WAKE_PENDING)) | BODY_AWAKE;
    gBodies[body].sleepTimer = 0.0;
#if defined(ENGINE_WGSL) && defined(ENGINE_SLEEP_ARCHIVE)
    // A sleeper a projectile hit wakes after the broadphase: list it for the sleep
    // archive's touch query (archive_touch), which forms its resting pairs this step.
    if (PASS_U == 1u && bodyIsSleeping(b) && archiveTouchQuery())
        archiveList(body);
#endif
}
