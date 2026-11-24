[shader("compute")]
[numthreads(256, 1, 1)]
void soft_primal_bodies(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint local = lid.x;
    gStabilize = false;
    uint nb = softBodyCount();
    bool isSoft = gid.x < nb;
    uint base = softBodyColorBase() + gid.x * softBodyColorStride();
    uint allColors = softColorCount();
    // Anything of this body to solve? (A soft body sleeps and wakes whole; a
    // sleeping particle's solve is a no-op, and soft_begin reseeds the Chebyshev
    // history every step.)
    if (local == 0u) gsSoftAwake = 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint c = 0u; c < (isSoft ? allColors : 0u); c++)
    {
        uint4 slot = SU(base + c / 2u);
        uint2 cr = (c & 1u) == 0u ? slot.xy : slot.zw;
        for (uint t = local; t < cr.y; t += 256u)
        {
            BodyState pb = gBodies[SU(softListBase() + cr.x + t).x];
            if (bodyIsDynamic(pb) && !bodyIsSleeping(pb) && !wakePending(pb)) gsSoftAwake = 1u;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Role: SOFT_ROLE_VEHICLE (workgroups past the bodies), nothing awake (0),
    // SOFT_ROLE_FOLLOW (rigid mode: particles take the core's pose once; the
    // colour order cannot matter, each writes only itself), or the colour count.
    if (local == 0u)
        gsUniformLanes = !isSoft ? SOFT_ROLE_VEHICLE
            : (gsSoftAwake == 0u ? 0u : (softIsRigid(gid.x) ? SOFT_ROLE_FOLLOW : allColors));
    GROUP_SYNC();
    uint role = gsUniformLanes;
    // Crash island: the step's passes split into its substeps. At the first pass
    // the first substep is set from the step's start; at each later boundary its
    // particles move on to the next substep (rest lengths move only at the step's
    // end). Chebyshev restarts with each substep. Lane 0 decides and publishes it
    // through gsUniformLanes: the boundary's barrier needs uniform control flow.
    GROUP_SYNC();
    if (local == 0u)
    {
        uint code = 0u;
        if (isSoft && role != 0u && role < SOFT_ROLE_FOLLOW && softSubsteps(gid.x) > 1u)
        {
            uint S = softSubsteps(gid.x);
            uint per = max(PARAMS.iterations * softSweeps() / S, 1u);
            uint j = min(PASS_U / per, S - 1u);
            code = 1u | (j == 0u ? 2u : 0u) | (j << 2) | ((PASS_U - j * per) << 8);
        }
        gsUniformLanes = code;
    }
    GROUP_SYNC();
    uint island = gsUniformLanes;
    uint chebPass = (island & 1u) != 0u ? island >> 8 : PASS_U;
    if ((island & 1u) != 0u && chebPass == 0u)
    {
        float hs = softH(gid.x);
        // The fraction of the step this substep ends at: what the particles'
        // iterates stand for in the rigid rows (world_step.slang bodyAtTime).
        float f = float(((island >> 2) & 63u) + 1u) / float(softSubsteps(gid.x));
        for (uint c = 0u; c < allColors; c++)
        {
            uint4 slot = SU(base + c / 2u);
            uint2 cr = (c & 1u) == 0u ? slot.xy : slot.zw;
            for (uint t = local; t < cr.y; t += 256u)
                softSubstepParticle(cr.x + t, hs, (island & 2u) != 0u, f);
        }
        DeviceMemoryBarrierWithGroupSync();
    }
    if (role == SOFT_ROLE_FOLLOW)
    {
        for (uint c = 0u; c < allColors; c++)
        {
            uint4 slot = SU(base + c / 2u);
            uint2 cr = (c & 1u) == 0u ? slot.xy : slot.zw;
            for (uint t = local; t < cr.y; t += 256u) softSolveEntry(SU(softListBase() + cr.x + t));
        }
    }
    uint colors = role < SOFT_ROLE_FOLLOW ? role : 0u;
    for (uint c = 0u; c < colors; c++)
    {
        uint4 slot = SU(base + c / 2u);
        uint2 cr = (c & 1u) == 0u ? slot.xy : slot.zw;
        uint lanes = softLanesFor(cr.y);
        uint t = local / lanes, lane = local % lanes;
        uint mode = SOFT_SKIP;
        uint4 ent = uint4(0, 0, 0, 0);
        if (lanes > 1u)
        {
            float3x3 H = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            float3 g = float3(0.0, 0.0, 0.0);
            if (t < cr.y)
            {
                ent = SU(softListBase() + cr.x + t);
                mode = softAccumulate(ent, lane, lanes, H, g);
            }
            gsSoftRedA[local] = float4(H[0][0], H[0][1], H[0][2], H[1][1]);
            gsSoftRedB[local] = float4(H[1][2], H[2][2], g.x, g.y);
            gsSoftRedC[local] = g.z;
        }
        GroupMemoryBarrierWithGroupSync();
        if (lanes > 1u)
        {
            if (lane == 0u && t < cr.y && mode != SOFT_SKIP)
            {
                float4 a = float4(0.0, 0.0, 0.0, 0.0), b = float4(0.0, 0.0, 0.0, 0.0);
                float gz = 0.0;
                for (uint k = 0u; k < lanes; k++)
                {
                    a += gsSoftRedA[local + k];
                    b += gsSoftRedB[local + k];
                    gz += gsSoftRedC[local + k];
                }
                float3x3 H = float3x3(a.x, a.y, a.z, a.y, a.w, b.x, a.z, b.x, b.y);
                softFinish(ent, mode, H, float3(b.z, b.w, gz));
            }
        }
        else
        {
            for (uint k = local; k < cr.y; k += 256u) softSolveEntry(SU(softListBase() + cr.x + k));
        }
        DeviceMemoryBarrierWithGroupSync();
    }
    if (softChebyshevRho() > 0.0)
    {
        for (uint c = 0u; c < colors; c++)
        {
            uint4 slot = SU(base + c / 2u);
            uint2 cr = (c & 1u) == 0u ? slot.xy : slot.zw;
            for (uint t = local; t < cr.y; t += 256u) softChebyshevEntry(cr.x + t, chebPass);
        }
    }
    // Workgroups past the soft bodies (SOFT_ROLE_VEHICLE): vehicle_iterate, 64 vehicles each. It
    // reads only the cores' iterate (solved before this pass), and the sweep reads
    // nothing it writes, so the two run side by side instead of one after the
    // other. (A wheel standing on another car's particle reads that particle
    // mid-sweep, as cross-body contacts already do.)
    if (role == SOFT_ROLE_VEHICLE) softVehicleLanes(gid.x - nb, local);
}

// A yield or a break: both ends take a backward-Euler step next (BDF2's history
// no longer describes them). Marks the history slot (w = 2); soft_velocity reads it.
void softBdf2Interrupt(uint s, uint a, uint b)
{
    if (!softBdf2On()) return;
    uint ia = softBdfBase() + particleOrdinal(s, a), ib = softBdfBase() + particleOrdinal(s, b);
    if (SD(ia).w == 1.0) gSoftData[ia].w = 2.0;
    if (SD(ib).w == 1.0) gSoftData[ib].w = 2.0;
}

// A yield or a break keeps its body soft: restart its quiet count.
void softQuietReset(uint s)
{
    uint4 st = SU(softModeSlot(s));
    if (st.y != 0u) gSoftData[softModeSlot(s)] = asfloat(uint4(st.x, 0u, st.z, st.w));
}

// Bolted part k of soft body s (layout.rs PART_STRIDE): [body, first joint,
// joints, 0], its position and orientation in the core's frame.
uint softPartBase() { return SU(9u).z; }
uint softPartCount(uint s) { return softBodyRec2(s).w; }
uint softPartSlot(uint s, uint k) { return softPartBase() + 3u * (32u * s + k); }

// A bolted part still on: some joint to its nodes holds.
bool softPartAttached(uint4 rec)
{
    for (uint j = rec.y; j < rec.y + rec.z; j++)
        if ((gJoints[j].flags & JOINT_FLAG_BROKEN) == 0u) return true;
    return false;
}

// Core mass properties and joints for a mode. Soft: the core alone, joined to
// its frame nodes; each bolted part a free body on its node joints. Rigid: the
// core carries the whole car (frame and attached parts), its node joints and the
// parts' joints idle, and the parts ride it (BODY_FOLLOW, placed by soft_mode).
// Joints switch with JOINT_FLAG_IDLE: they stay in this step's graph, so a car
// turning soft solves them this same step.
void softSetMode(uint s, uint core, uint4 st, bool rigid)
{
    float4 m = SD(softBodyBase() + 6u * s + (rigid ? 5u : 4u));
    gBodies[core].invMass = m.x;
    gBodies[core].invInertia = m.yzw;
    uint nj = (st.z >> 8) & 255u;
    for (uint j = st.w; j < st.w + nj; j++)
    {
        uint f = gJoints[j].flags;
        gJoints[j].flags = rigid ? (f | JOINT_FLAG_IDLE) : (f & ~JOINT_FLAG_IDLE);
    }
    for (uint k = 0u; k < softPartCount(s); k++)
    {
        uint4 rec = SU(softPartSlot(s, k));
        // A loose part (every joint broken) stays a free body.
        bool ride = rigid && softPartAttached(rec);
        uint bf = gBodies[rec.x].flags;
        gBodies[rec.x].flags = ride ? (bf | BODY_FOLLOW) : (bf & ~BODY_FOLLOW);
        for (uint j = rec.y; j < rec.y + rec.z; j++)
        {
            uint f = gJoints[j].flags;
            gJoints[j].flags = ride ? (f | JOINT_FLAG_IDLE) : (f & ~JOINT_FLAG_IDLE);
        }
    }
}

// Bolted parts, each step after the mode is decided: while soft, capture each
// one's pose in the core's frame at the step start; while rigid, place it at
// the core's predicted pose (as soft_begin places the particles). Its velocity
// comes from that motion (extract_velocity), so it carries the core's when the
// car turns soft.
void softPartsFollow(uint s, uint core, bool rigid)
{
    BodyState c = gBodies[core];
    for (uint k = 0u; k < softPartCount(s); k++)
    {
        uint slot = softPartSlot(s, k);
        uint4 rec = SU(slot);
        BodyState b = gBodies[rec.x];
        if (rigid)
        {
            if ((b.flags & BODY_FOLLOW) == 0u) continue;
            gBodies[rec.x].position = c.position + quatRotate(c.orientation, SD(slot + 1u).xyz);
            gBodies[rec.x].orientation = quatNormalize(quatMul(c.orientation, SD(slot + 2u)));
        }
        else
        {
            float4 qi = quatConjugate(c.initialRot);
            gSoftData[slot + 1u] = float4(quatRotate(qi, b.initialPos - c.initialPos), 0.0);
            gSoftData[slot + 2u] = quatMul(qi, b.initialRot);
        }
    }
}

// A body's live contact edges (see softContactEdgeLive).
bool softBodyContactLive(uint b, uint nJ)
{
    uint hi = SOLVE_INDEX(b + 1u);
    for (uint e = SOLVE_INDEX(b); e < hi; e++)
        if (SOLVE_ID(e) >= nJ && softContactEdgeLive(SOLVE_ID(e) - nJ)) return true;
    return false;
}

// A rigid contact edge (solver edge `id`, past the joints) that can push this
// step: its gap at the step start closed by twice this step's approach along the
// normal (C = c0N + n·(dB − dA), as the solve), or a force carried over from the
// last step. A speculative contact the body only glides past (the ground under a
// moving car's sills) cannot, and must not keep the car soft.
bool softContactEdgeLive(uint id)
{
    float h2 = 2.0 * max(PARAMS.dt, 1e-6);
#if defined(ENGINE_COMPACT_SOLVE)
    SolverPair pair = gSolverPairs[id];
    uint count = pair.countAndSphere & SOLVER_COUNT_MASK;
    uint start = pair.contactStart;
    BodyState bA = gBodies[pair.bodyA];
    BodyState bB = gBodies[pair.bodyB];
#else
    Manifold mf = gManifolds[id];
    uint count = mf.contactCount;
    uint start = mf.contactStart;
    BodyState bA = gBodies[gColliders[mf.colliderA].body];
    BodyState bB = gBodies[gColliders[mf.colliderB].body];
#endif
    for (uint ci = 0u; ci < count; ci++)
    {
#if defined(ENGINE_COMPACT_SOLVE)
        Contact c = solverContact(start + ci - HOST_S(11), pair.normal, solverPairFriction(pair));
#else
        Contact c = READ_CONTACT(start + ci);
#endif
        if (c.alive == 0u) continue;
        if (c.lambdaN < 0.0) return true;
        float3 vA = bA.linearVel + cross(bA.angularVel, quatRotate(bA.orientation, c.anchorA));
        float3 vB = bB.linearVel + cross(bB.angularVel, quatRotate(bB.orientation, c.anchorB));
        if (c.c0N + h2 * min(dot(c.normal, vB - vA), 0.0) < 0.01) return true;
    }
    return false;
}

// Rigid mode (design: "rigid mode with promotion"), once per soft body per step
// after the contact passes and before the solve. Promote (turn soft) on any
// contact of a particle (a contact edge in its constraint list) or bounds that
// near another soft body or an obstacle; demote (turn rigid) after `quiet ms`
// with neither and no yield. Both take effect this step.
// Bounds near another soft body or a registered obstacle (within both skins
// + 0.1 m, an obstacle's own travel this step besides).
bool softNear(uint s)
{
    bool trigger = false;
    if (softContactsOn())
    {
        // Contacts this step form only inside the bounds soft_bounds padded by
        // twice this step's travel (+ 0.1 m), so bounds within both skins
        // (+ 0.1 m) of each other are the earliest a contact can come: the car
        // turns soft that step, before any contact force. A wider margin only
        // kept cars soft in traffic for nothing.
        float own = asfloat(softBodyRec2(s).z) + 0.1;
        float3 lo = SD(softAabbBase() + 2u * s).xyz - own, hi = SD(softAabbBase() + 2u * s + 1u).xyz + own;
        uint nb = softBodyCount();
        uint np = softPairCount(s);
        for (uint pk = 0u; pk < np && !trigger; pk++)
        {
            uint s2 = softPairAt(s, pk);
            if (s2 == s) continue;
            float m2 = asfloat(softBodyRec2(s2).z);
            float3 lo2 = SD(softAabbBase() + 2u * s2).xyz - m2, hi2 = SD(softAabbBase() + 2u * s2 + 1u).xyz + m2;
            if (all(lo <= hi2) && all(lo2 <= hi)) trigger = true;
        }
        uint nObs = softObstacleCount();
        float h2 = 2.0 * max(PARAMS.dt, 1e-6);
        for (uint k = 0u; k < nObs && !trigger; k++)
        {
            Collider col = gColliders[SU(softObstacleBase() + k).x];
            BodyState ob = gBodies[col.body];
            float3 olo, ohi;
            obstacleBounds(col, ob, olo, ohi);
            // A moving obstacle (a ram, a crusher): its own travel this step too.
            float mo = h2 * (length(ob.linearVel) + length(ob.angularVel) * length(ohi - olo) * 0.5);
            if (all(ohi + mo >= lo) && all(olo - mo <= hi)) trigger = true;
        }
    }
    return trigger;
}

// Crash islands: a body that is not riding its core, with a live contact on a
// particle or a soft contact in the last 50 ms, takes crash_substeps substeps
// this step. Yields and nearness do not count: a quiet body hung on a rigid
// core, stepped in substeps, kept yielding a little, so it never went rigid.
void softIsland(uint s)
{
    uint S = softCrashSubsteps();
    uint4 isl = SU(softIslandBase() + s);
    bool hit = false;
    if (S > 1u && !softIsRigid(s))
    {
        uint4 r = softBodyRec(s);
        uint nJ = gCounters[0].jointCount;
        uint ob = asuint(softBodyRec1(s).w);
        for (uint k = 0u; k < r.y && !hit; k++)
            hit = softBodyContactLive(r.x + k, nJ) || (softContactsOn() && SD(softFlagBase() + ob + k).x != 0.0);
    }
    uint quiet = hit ? 0u : min(isl.y + 1u, 1u << 24);
    bool island = S > 1u && !softIsRigid(s) && float(quiet) * max(PARAMS.dt, 1e-6) <= 0.05;
    gSoftData[softIslandBase() + s] = asfloat(uint4(island ? S : 1u, quiet, 0u, 0u));
}

void softModeBody(uint s)
{
    uint4 st = SU(softModeSlot(s));
    if ((st.z & 1u) == 0u) return; // no rigid core
    uint core = asuint(softBodyRec1(s).z);
    if (core == SOFT_NO_BODY) return;
    bool enabled = (softFlags() & SOFT_RIGID_MODE) != 0u;
    uint4 r = softBodyRec(s);
    bool trigger = !enabled;
    uint nJ = gCounters[0].jointCount;
    for (uint k = 0u; k < r.y && !trigger; k++) trigger = softBodyContactLive(r.x + k, nJ);
    // A blow to an attached bolted part.
    for (uint k = 0u; k < softPartCount(s) && !trigger; k++)
    {
        uint4 rec = SU(softPartSlot(s, k));
        if (softPartAttached(rec)) trigger = softBodyContactLive(rec.x, nJ);
    }
    if (!trigger) trigger = softNear(s);
    float h = max(PARAMS.dt, 1e-6);
    uint quiet = trigger ? 0u : min(st.y + 1u, 1u << 24);
    uint mode = st.x;
    if (mode == 1u && trigger)
    {
        mode = 0u;
        softSetMode(s, core, st, false);
    }
    else if (mode == 0u && enabled && float(quiet) * h * 1000.0 >= float(st.z >> 16))
    {
        mode = 1u;
        softSetMode(s, core, st, true);
    }
    gSoftData[softModeSlot(s)] = asfloat(uint4(mode, quiet, st.z, st.w));
    softPartsFollow(s, core, mode == 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_mode(uint3 tid : SV_DispatchThreadID)
{
    uint s = dispatchIndex(tid);
    if (s >= softBodyCount()) return;
    softModeBody(s);
    softIsland(s);
}

// Dual update per beam: the finite-stiffness ramp k ← min(k*, k + β|C|), and the
// multiplier of hard (augmented-Lagrangian) beams.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_dual(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softBeamCount()) return;
    SoftBeam e = loadBeam(i);
    if (beamBroken(e)) return;
    uint ob = asuint(softBodyRec1(beamSoftBody(e)).w) - softBodyRec(beamSoftBody(e)).x;
    float len = length(softLocal(ob + e.a) - softLocal(ob + e.b));
    float cv = len - e.rest;
    if (beamHard(e))
    {
        e.lambda += e.penalty * cv;
        e.penalty = min(e.penalty + softBeta() * abs(cv), PARAMS.kMaxN);
    }
    else
    {
        if (e.penalty >= e.spring) return;
        e.penalty = min(e.penalty + softBeta() * abs(cv), e.spring);
    }
    storeBeamState(i, e);
}

// Post-step plasticity and breaking at the converged state (Rigs of Rods rules).
// Rigs of Rods yield at the converged length: past a threshold the rest length
// moves so the beam sits on it (hardening in compression, weakening in
// tension). Returns the force the beam carried.
float softBeamYield(uint i, inout SoftBeam e, float force, float delta)
{
    uint flags = softFlags();
    float ts = softThresholdScale();
    if ((flags & SOFT_PLASTICITY) != 0u && beamPlastic(beamKind(e)) && e.spring > 0.0)
    {
        float k = e.spring;
        // Tension on an anisotropic beam yields against its expansion spring.
        float kt = beamExpansion(e, i).x;
        float dc = e.deformC * ts;
        float dt = e.deformT * ts;
        if (force > dc && delta < 0.0)
        {
            // The solve capped it at the threshold: that is the force it carried.
            force = dc;
            float old = e.rest;
            float deform = delta + dc * (1.0 - e.coef) / k;
            e.rest = max(e.rest + deform, softMinLength());
            if (e.shortB > 0.0) e.rest = max(e.rest, e.rest0 * e.shortB);
            // Hardening stops at the beam's ceiling (JBeam deformLimitStress).
            e.deformC = min(e.deformC * (old / e.rest), SD(softBeamExtraBase() + i).z);
            e.meta |= BEAM_YIELDED;
            softQuietReset(beamSoftBody(e));
            softBdf2Interrupt(beamSoftBody(e), e.a, e.b);
        }
        else if (-force > dt && delta > 0.0)
        {
            // Tension: it carried the threshold, and the plastic stretch uses up
            // its strength until it tears.
            force = -dt;
            float old = e.rest;
            float deform = delta - dt * (1.0 - e.coef) / kt;
            e.rest += deform;
            if (e.longB > 0.0) e.rest = min(e.rest, e.rest0 * e.longB);
            e.deformT *= e.rest / old;
            e.strength -= deform * kt / ts;
            e.meta |= BEAM_YIELDED;
            softQuietReset(beamSoftBody(e));
            softBdf2Interrupt(beamSoftBody(e), e.a, e.b);
        }
    }
    return force;
}

// Rest lengths move only at the step's end: yield checks mid-step fired on
// unconverged forces (the structure, and the rigid core it hangs on, had only
// part of the iterations). The in-solve cap holds each beam at its threshold
// through the substeps.
// A crash island's particle at a substep boundary: `restart` (pass 0) sets the
// first substep from the step's start with hₛ; otherwise the substep that ended
// becomes the next one's start, at the velocity it reached.
void softSubstepParticle(uint li, float h, bool restart, float f)
{
    uint4 ent = SU(softListBase() + li);
    uint body = ent.x;
    BodyState b = gBodies[body];
    if (!bodyIsDynamic(b) || bodyIsSleeping(b) || wakePending(b)) return;
    uint sb = ent.w & 0xFFFFu;
    uint ord = particleOrdinal(sb, body);
    uint lq = softLocalBase() + LOCAL_STRIDE * ord;
    float3 x0 = softLocal0(ord), x = softLocal(ord);
    float3 v = restart ? b.linearVel : (x - x0) / h;
    float3 start = restart ? x0 : x;
    float3 gv = float3(0.0, PARAMS.gravityY, 0.0);
    float3 y = start + v * h + gv * (h * h);
    gSoftData[lq + 1u] = float4(start, 0.0);
    gSoftData[lq + 2u] = float4(y, 0.0);
    float3 o = softOrigin(sb);
    gAabbs[body].mn = o + y;
    softSetLocal(body, ord, sb, y);
    gBodies[body].initialPad = -f;
    // Chebyshev history restarts at the substep's guess.
    uint q = softChebBase() + 3u * li;
    gSoftData[q] = float4(y, 0.0);
    gSoftData[q + 1u] = float4(y, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void soft_plastic(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    // Elements ride the beams' threads (dispatched one per beam).
    for (uint k = i; k < softElementCount(); k += max(softBeamCount(), 1u)) softElementPlastic(k);
    if (i >= softBeamCount()) return;
    SoftBeam e = loadBeam(i);
    if (beamBroken(e)) return;
    if (softIsRigid(beamSoftBody(e))) return; // rigid mode: the shape does not change
    BodyState ba = gBodies[e.a];
    BodyState bb = gBodies[e.b];
    // Both ends asleep (or static): nothing moved this step.
    bool movingA = bodyIsDynamic(ba) && !bodyIsSleeping(ba);
    bool movingB = bodyIsDynamic(bb) && !bodyIsSleeping(bb);
    if (!movingA && !movingB) return;
    float h = softH(beamSoftBody(e));
    uint ob = asuint(softBodyRec1(beamSoftBody(e)).w) - softBodyRec(beamSoftBody(e)).x;
    float3 la = softLocal(ob + e.a), lb = softLocal(ob + e.b);
    float len = length(la - lb);
    // A sleeping end did not move: its step-start position is where it is.
    float3 la0 = movingA ? softLocal0(ob + e.a) : la, lb0 = movingB ? softLocal0(ob + e.b) : lb;
    float len0 = length(la0 - lb0);
    float force = beamReadout(e, i, len, len0, h);
    uint flags = softFlags();
    float delta = len - e.rest;
    float ts = softThresholdScale();
    force = softBeamYield(i, e, force, delta);
    e.force = force;
    if ((flags & SOFT_BREAKING) != 0u)
    {
        bool broke = beamKind(e) == BEAM_SUPPORT
            ? (e.longB > 0.0 && delta > e.rest * e.longB)
            : abs(force) > e.strength * ts;
        if (broke)
        {
            e.meta |= BEAM_BROKEN;
            uint g = beamGroup(e);
            // A passive member (breakGroupType 1) breaks with its group but does
            // not set it off.
            bool passive = SU(softBeamExtraBase() + i).w != 0u;
            if (g != 0u && g < softGroupCount() && !passive) gSoftData[softGroupBase() + g] = float4(1, 0, 0, 0);
            emitEvent(EVENT_BEAM_BREAK, i, asuint(abs(force)), beamSoftBody(e));
        }
    }
    storeBeamState(i, e);
}

// Detacher groups: a beam breaking breaks every beam of its group.
[shader("compute")]
[numthreads(64, 1, 1)]
void soft_break_groups(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= softBeamCount()) return;
    SoftBeam e = loadBeam(i);
    uint g = beamGroup(e);
    if (g == 0u || g >= softGroupCount() || beamBroken(e)) return;
    if (SD(softGroupBase() + g).x == 0.0) return;
    e.meta |= BEAM_BROKEN;
    storeBeamState(i, e);
    emitEvent(EVENT_BEAM_BREAK, i, asuint(abs(e.force)), beamSoftBody(e));
}
