// ---------------------------------------------------------------------------
// Compaction resistance (wheel_soil, last step) as a rolling-resistance torque
// on the spin servo, while the wheel still stands on the soil it was found on.
// Zero (no soil): the slot's collider tag is 0, never hit + 1.
float soilRollingTorque(uint wr, uint hit)
{
    uint tag = VU(wr + W_SOIL2).x;
    if (tag != hit + 1u) return 0.0;
    return VD(wr + W_SOIL2).y * VD(wr + W_HP).w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_prepare(uint3 tid : SV_DispatchThreadID)
{
    uint v = dispatchIndex(tid);
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    uint4 cfg = VU(vr + V_CFG);
    uint chassis = cfg.x;
    BodyState c = gBodies[chassis];
    uint at = vehicleAabbBlock(v);
    uint4 d3 = VU(vr + V_D3);
    bool active = (cfg.w & VF_ENABLED) != 0u && bodyIsDynamic(c) && !bodyIsSleeping(c);
    d3.y = active ? (d3.y | VS_ACTIVE) : (d3.y & ~VS_ACTIVE);
    VSetU(vr + V_D3, d3);
    if (!active)
    {
        Mat6 H0 = mat6Zero();
        Vec6 g0 = vec6Zero();
        writeVehicleBlock(at, H0, g0, false);
        return;
    }
    // Step-start velocity for vehicle_loads (BODY_PREV's fields are sleep scratch).
    VSet(vr + V_V0, float4(c.linearVel, 0.0));
    VSet(vr + V_W0, float4(c.angularVel, 0.0));
    AvbdParams p = PARAMS;
    float dt = p.dt;
    float4 q0 = c.initialRot;
    float3 fwdW = quatRotate(q0, vehicleFwdLocal(vr));
    float3 upW = quatRotate(q0, vehicleUpLocal(vr));
    float speedFwd = dot(c.linearVel, fwdW);
    float mass = VD(vr + V_MASS).x;

    float tauDrive[MAX_VEHICLE_WHEELS];
    computeDrivetrain(vr, cfg, dt, speedFwd, tauDrive);

    uint nW = min(cfg.z, MAX_VEHICLE_WHEELS);
    uint nContact = 0u;
    for (uint wi = 0u; wi < nW; wi++)
        if ((VU(wheelRec(cfg.y + wi) + W_C0).z & WD_CONTACT) != 0u) nContact++;
    float mShare = mass / float(max(nContact, 1u));
    float brake = VD(vr + V_D1).w;
    float handbrake = VD(vr + V_D2).x;
    uint brakeMask = VU(vr + V_D3).z;

    for (uint wi = 0u; wi < nW; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
        uint wf = VU(wr + W_CFG).y;
        float R = VD(wr + W_HP).w;
        float4 iner = VD(wr + W_INER);
        float4 w = VD(wr + W_W);
        bool contact = (dyn.z & WD_CONTACT) != 0u;
        float inertia = max(iner.x, 1e-6);

        // Wheel spin inertial target (Jolt angular damping, drive torque).
        float omega0 = w.x;
        float yTilde = dt * omega0 * max(0.0, 1.0 - iner.y * dt) + dt * dt * tauDrive[wi] / inertia;
        float mY = inertia / (dt * dt);
        float Fn = VD(wr + W_F).x;
        float tauB = 0.0;
        if ((brakeMask & (1u << wi)) != 0u)
        {
            tauB = brake * iner.z;
            if ((wf & WF_HANDBRAKE) != 0u) tauB += handbrake * iner.w;
        }
        if (contact) tauB += VD(wr + W_MISC).y * Fn * R;
        if (contact) tauB += soilRollingTorque(wr, dyn.y);
        // Tracked steering: the inner track's brake holds it at ratio × the outer
        // track's speed (a steering brake). Jolt only syncs the spins, which the
        // implicit tyre rows here pull straight back to ground speed.
        float yStar = 0.0;
        if ((cfg.w & VF_TRACKED) != 0u)
        {
            uint4 tl = VU(vr + V_TRACK);
            float4 td = VD(vr + V_TRACK_D);
            bool leftInner = abs(td.x) < abs(td.y);
            uint inner = leftInner ? tl.x : tl.y;
            uint outer = leftInner ? tl.y : tl.x;
            if (abs(td.x) != abs(td.y) && wi == inner && outer < cfg.z && abs(td.z) > 0.0)
            {
                float rat = leftInner ? td.x / td.y : td.y / td.x;
                uint orr = wheelRec(cfg.y + outer);
                float wOut = VD(orr + W_W).x;
                yStar = dt * wOut * rat;
                // Fade in with track speed (full at 1 m/s) so a standing tank can
                // pull away on both tracks instead of pivoting on a locked one.
                tauB += iner.z * saturate(abs(wOut) * VD(orr + W_HP).w);
            }
        }
        VSet(wr + W_LN, float4(VD(wr + W_LN).xyz, yStar));
        VSet(wr + W_Y, float4(yTilde, yTilde, mY, tauB));
        VSet(wr + W_W, float4(w.x, w.y, tauDrive[wi], mShare));

        // Servo warm start / penalty floor.
        float4 lam = VD(wr + W_LAM);
        float4 k2 = VD(wr + W_K2);
        float decay = p.alpha * p.gamma;
        float kFloor = mShare / (dt * dt);
        float kBFloor = (inertia + mShare * R * R) / (dt * dt);
        lam.z = tauB > 0.0 ? clamp(lam.z * decay, -tauB, tauB) : 0.0;
        k2.y = clamp(k2.y * p.gamma, kBFloor, 1e3 * kBFloor);

        if (!contact)
        {
            lam.x = 0.0; lam.y = 0.0;
            lam.w = kFloor; k2.x = kFloor;
            VSet(wr + W_LAM, lam);
            VSet(wr + W_K2, k2);
            VSet(wr + W_C3, float4(VD(wr + W_C3).xyz, 0.0));
            VSet(wr + W_C4, float4(VD(wr + W_C4).xyz, 0.0));
            continue;
        }

        float3 n = VD(wr + W_C2).xyz;
        float invCos = VD(wr + W_C2).w;
        float3 r = VD(wr + W_C1).xyz;
        float steer = VD(wr + W_C5).w;
        float3 fwdLocal = quatRotate(wheelSteerRot(wr, steer), VD(wr + W_FWD).xyz);
        float3 fw = quatRotate(q0, fwdLocal);
        float3 t = fw - n * dot(n, fw);
        t = dot(t, t) > 1e-12 ? normalize(t) : normalize(cross(quatRotate(q0, axleLocal(vr)), n));
        float3 s = cross(n, t);

        // Spring constants (Jolt): K = m_eff ω², D = 2 m_eff ζ ω, m_eff at the contact along n.
        float4 spr = VD(wr + W_SPR);
        uint mode = asuint(VD(wr + W_SUSP).w);
        float3 rn = cross(suspensionLever(wr, r, q0), n);
        float invMeff = c.invMass + dot(rn, applyInvInertiaWorld(c, rn));
        float meff = invMeff > 1e-12 ? 1.0 / invMeff : mShare;
        float K, D;
        if (mode == 0u)
        {
            float om = TWO_PI * spr.x;
            K = meff * om * om;
            D = 2.0 * meff * spr.y * om;
        }
        else { K = spr.x; D = spr.y; }
        float Kb = 0.0, Db = 0.0;
        if ((wf & WF_RIGID_BUMP) != 0u) { Kb = 100.0 * meff / (dt * dt); Db = 0.0; }
        else if (spr.z > 0.0)
        {
            float ob = TWO_PI * spr.z;
            Kb = meff * ob * ob;
            Db = 2.0 * meff * spr.w * ob;
        }
        VSet(wr + W_K, float4(K, D / dt, Kb, Db / dt));

        // Ground motion over the step (explicit).
        float3 gv = VD(wr + W_GV).xyz;
        VSet(wr + W_C5, float4(gv * dt, steer));
        VSet(wr + W_C3, float4(t, 0.0));
        VSet(wr + W_C4, float4(s, 0.0));

        // Slip (Jolt WheelWV::Update) from step-start velocities.
        float3 vp = c.linearVel + cross(c.angularVel, r) - gv;
        vp -= n * dot(n, vp);
        float vLong = dot(vp, t);
        float denom = sign(vLong) * max(1e-3, abs(vLong));
        float kappa = abs((omega0 * R - vLong) / (vLong == 0.0 ? 1e-3 : denom));
        float vLen = length(vp);
        float alphaSlip = vLen < 1e-3 ? 0.0 : acos(clamp(abs(vLong) / vLen, 0.0, 1.0));
        VSet(wr + W_C3, float4(t, kappa));
        VSet(wr + W_C4, float4(s, alphaSlip));
        float4 tire = VD(wr + W_TIRE);
        uint hit = VU(wr + W_C0).y;
        float surf = gMaterials[materialIndex(gColliders[hit])].friction;
        // Bound by the curve's falling branch: below the peak slip the AVBD row's
        // penalty already supplies the elastic slip stiffness, so a rolling tyre
        // that starts to slip mid-step keeps its peak grip instead of μ(0) = 0.
        uint hL = asuint(tire.x), hT = asuint(tire.y);
        float muLs = sqrt(max(curveEval(hL, max(kappa, curvePeakX(hL))) * tire.z, 0.0) * surf);
        float muTs = sqrt(max(curveEval(hT, max(degrees(alphaSlip), curvePeakX(hT))) * tire.w, 0.0) * surf);
        float muLp = sqrt(max(curvePeak(asuint(tire.x)) * tire.z, 0.0) * surf);
        float muTp = sqrt(max(curvePeak(asuint(tire.y)) * tire.w, 0.0) * surf);
        VSet(wr + W_MU, float4(muLs, muTs, muLp, muTp));

        // Friction warm start (three-avbd: λ ← λ·α·γ, k ← clamp(k·γ)).
        lam.x *= decay; lam.y *= decay;
        lam.w = clamp(lam.w * p.gamma, kFloor, 1e3 * kFloor);
        k2.x = clamp(k2.x * p.gamma, kFloor, 1e3 * kFloor);
        VSet(wr + W_LAM, lam);
        VSet(wr + W_K2, k2);
    }

    // Motorcycle target lean: direction of last step's ground force (normal and
    // lateral), smoothed, sideways only, clamped to the max lean (Jolt PreCollide).
    float4 bike0 = VD(vr + V_BIKE0);
    if (bike0.x > 0.0)
    {
        float3 worldUp = float3(0.0, p.gravityY < 0.0 ? 1.0 : -1.0, 0.0);
        float3 tgt = float3(0, 0, 0);
        bool all = true;
        for (uint wi = 0u; wi < nW; wi++)
        {
            uint wr = wheelRec(cfg.y + wi);
            uint wd = VU(wr + W_C0).z;
            if ((wd & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
            if ((wd & WD_CONTACT) == 0u) { all = false; continue; }
            float4 f = VD(wr + W_F);
            tgt += VD(wr + W_C2).xyz * f.x - VD(wr + W_C4).xyz * f.z;
        }
        tgt = dot(tgt, tgt) > 1e-12 ? normalize(tgt) : worldUp;
        float4 bd = VD(vr + V_BIKE_D);
        float4 bike1 = VD(vr + V_BIKE1);
        float3 T = bike1.y * bd.xyz + (1.0 - bike1.y) * tgt;
        T -= fwdW * dot(T, fwdW);
        T = dot(T, T) > 1e-12 ? normalize(T) : worldUp;
        float3 adj = worldUp - fwdW * dot(worldUp, fwdW);
        adj = dot(adj, adj) > 1e-12 ? normalize(adj) : worldUp;
        float wAng = -sign(dot(cross(T, adj), fwdW)) * acos(clamp(dot(T, adj), -1.0, 1.0));
        if (abs(wAng) > bike0.x) T = quatRotate(quatAxisAngle(fwdW, sign(wAng) * bike0.x), adj);
        float dAng = -sign(dot(cross(T, upW), fwdW)) * acos(clamp(dot(T, upW), -1.0, 1.0));
        float integ = all ? bd.w + dAng * dt : bd.w * max(0.0, 1.0 - bike1.x * dt);
        VSet(vr + V_BIKE_D, float4(T, integ));
    }

    float3 dl, da;
    chassisDisplacement(c, dl, da);
    Mat6 H;
    Vec6 g;
    assembleVehicle(v, vr, cfg, c, dl, da, false, H, g);
    writeVehicleBlock(at, H, g, true);
    VSet(vr + V_REFP, float4(c.position, VD(vr + V_REFP).w));
    VSet(vr + V_REFQ, c.orientation);
}

// ---------------------------------------------------------------------------
// vehicle_iterate: back-substitute wheel spin, dual update, rebuild block
// ---------------------------------------------------------------------------
// One vehicle's iterate (vehicle_iterate, or soft_iterate's vehicle workgroups).
void vehicleIterateOne(uint v)
{
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    if ((VU(vr + V_D3).y & VS_ACTIVE) == 0u) return;
    uint4 cfg = VU(vr + V_CFG);
    BodyState c = gBodies[cfg.x];
    if (!bodyIsDynamic(c) || bodyIsSleeping(c)) return;

    // Wheel spins are re-solved exactly inside assembleVehicle at this iterate.
    float3 dl, da;
    chassisDisplacement(c, dl, da);
    Mat6 H;
    Vec6 g;
    assembleVehicle(v, vr, cfg, c, dl, da, true, H, g);
    writeVehicleBlock(vehicleAabbBlock(v), H, g, true);
    VSet(vr + V_REFP, float4(c.position, VD(vr + V_REFP).w));
    VSet(vr + V_REFQ, c.orientation);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_iterate(uint3 tid : SV_DispatchThreadID)
{
    vehicleIterateOne(dispatchIndex(tid));
}

// ---------------------------------------------------------------------------
// vehicle_loads: what each part's / wheel's mount carried this step
// ---------------------------------------------------------------------------
// After the final dual (λ = this step's converged contact forces) and
// extract_velocity_sleep (the solved velocities). A part is rigid with the
// chassis, so the wrench through its mount is what the part's own motion needs
// minus what its contacts supply: F = m·(a_p − g) − ΣF_contact, and the moment
// about the mount anchor likewise. A wheel's mount carries its tyre force and
// its shell contacts. Past a limit the part (wheel) is flagged in V_BRK and
// vehicle_input severs it before the next step solves: forward-only, as joints
// break on the final dual and leave the next step's solve.
//
// One thread per vehicle walks the chassis's CSR row in its sorted order, so
// the sums are deterministic (no float atomics). Contacts come through
// READ_CONTACT: canonical and compact hosts publish final λ to gContacts; the
// persistent host compiles vehicle_loads_persistent (vehicle_step_persistent.slang).
static const uint EVENT_MOUNT_BREAK = 6u;
static const uint MAX_VEHICLE_PARTS = 32u;

[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_loads(uint3 tid : SV_DispatchThreadID)
{
    uint v = dispatchIndex(tid);
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    uint4 cfg = VU(vr + V_CFG);
    if ((cfg.w & (VF_ENABLED | VF_BREAKABLE)) != (VF_ENABLED | VF_BREAKABLE)) return;
    // A sleeping (inactive) vehicle carries its resting loads: nothing new to check.
    if ((VU(vr + V_D3).y & VS_ACTIVE) == 0u) return;
    uint chassis = cfg.x;
    BodyState c = gBodies[chassis];
    AvbdParams p = PARAMS;
    float dt = p.dt;
    float4 qi = quatConjugate(c.orientation);
    uint4 pool2 = VU(vr + V_POOL2);
    uint nP = min(pool2.y, MAX_VEHICLE_PARTS);
    uint nW = min(cfg.z, MAX_VEHICLE_WHEELS);
    // Chassis-local contact force and moment (about the origin) per part / wheel shell.
    float3 partF[MAX_VEHICLE_PARTS], partM[MAX_VEHICLE_PARTS];
    float3 wheelF[MAX_VEHICLE_WHEELS], wheelM[MAX_VEHICLE_WHEELS];
    for (uint k = 0u; k < MAX_VEHICLE_PARTS; k++) { partF[k] = float3(0, 0, 0); partM[k] = float3(0, 0, 0); }
    for (uint k = 0u; k < MAX_VEHICLE_WHEELS; k++) { wheelF[k] = float3(0, 0, 0); wheelM[k] = float3(0, 0, 0); }

    // ---- contacts on the chassis (final λ), in its CSR order ----
    uint nJ = gCounters[0].jointCount;
    uint lo = SOLVE_INDEX(chassis);
    uint hi = SOLVE_INDEX(chassis + 1u);
    for (uint e = lo; e < hi; e++)
    {
        uint id = SOLVE_ID(e);
        if (id < nJ) continue; // tyre / user joints: not part mounts
#if defined(ENGINE_PERSISTENT_CONTACTS)
        // Persistent hosts key graph edges to their solver pairs; each contact's
        // colliders live in its cold record.
        SolverPair pair = gSolverPairs[id - nJ];
        uint contactStart = pair.contactStart;
        uint contactCount = pair.countAndSphere & SOLVER_COUNT_MASK;
        bool isA = pair.bodyA == chassis;
#else
        Manifold mf = gManifolds[id - nJ];
        uint contactStart = mf.contactStart;
        uint contactCount = mf.contactCount;
        bool isA = gColliders[mf.colliderA].body == chassis;
        uint ci = isA ? mf.colliderA : mf.colliderB;
#endif
        for (uint k = 0u; k < contactCount; k++)
        {
            Contact ct = READ_CONTACT(contactStart + k);
            if (ct.alive == 0u) continue;
#if defined(ENGINE_PERSISTENT_CONTACTS)
            uint ci = isA ? ct.colliderA : ct.colliderB;
#endif
            // n = A→B; the force on B is −(λN n + λT1 t1 + λT2 t2), on A its opposite.
            float3 fB = -(ct.normal * ct.lambdaN + ct.tangent1 * ct.lambdaT1 + ct.tangent2 * ct.lambdaT2);
            float3 fW = isA ? -fB : fB;
            // Anchors are body-local offsets from the centre of mass: for the
            // chassis that is already the chassis-local point.
            float3 xL = isA ? ct.anchorA : ct.anchorB;
            float3 fL = quatRotate(qi, fW);
            // The part (collider range) or wheel shell that owns the collider.
            bool found = false;
            for (uint pi = 0u; pi < nP && !found; pi++)
            {
                uint4 pc = VU(partRec(pool2.x + pi) + P_CFG);
                if (ci >= pc.x && ci < pc.x + pc.y)
                {
                    partF[pi] += fL;
                    partM[pi] += cross(xL, fL);
                    found = true;
                }
            }
            for (uint wi = 0u; wi < nW && !found; wi++)
            {
                uint shell = VU(wheelRec(cfg.y + wi) + W_TJ).y;
                if (shell != NO_INDEX && ci >= shell && ci < shell + SHELL_SPHERES)
                {
                    wheelF[wi] += fL;
                    wheelM[wi] += cross(xL, fL);
                    found = true;
                }
            }
        }
    }

    // ---- chassis kinematics this step (chassis-local) ----
    float3 aL = quatRotate(qi, (c.linearVel - VD(vr + V_V0).xyz) / dt);
    float3 wL = quatRotate(qi, c.angularVel);
    float3 alL = quatRotate(qi, (c.angularVel - VD(vr + V_W0).xyz) / dt);
    float gScale = VD(vr + V_AERO0).z;
    float3 gL = quatRotate(qi, float3(0.0, p.gravityY * (gScale != 0.0 ? gScale : 1.0), 0.0));

    uint4 brk = VU(vr + V_BRK);
    // Children (higher indices) first: a part's mount also carries what its
    // attached children's mounts pass into it (force, and moment about the origin).
    float3 childF[MAX_VEHICLE_PARTS], childM[MAX_VEHICLE_PARTS];
    for (uint k = 0u; k < MAX_VEHICLE_PARTS; k++) { childF[k] = float3(0, 0, 0); childM[k] = float3(0, 0, 0); }
    for (uint step = 1u; step < nP; step++)
    {
        uint pi = nP - step;
        uint pr = partRec(pool2.x + pi);
        float4 com = VD(pr + P_COM);
        if (asuint(com.w) != PS_ATTACHED) continue;
        float4 mount = VD(pr + P_MOUNT);
        float4 load = VD(pr + P_LOAD);
        uint parent = VU(pr + P_TREE).x;
        if (mount.w >= 1e29 && load.x >= 1e29 && parent >= 32u) continue;
        float4 mass = VD(pr + P_MASS);
        float3 r = com.xyz;
        float3 ap = aL + cross(alL, r) + cross(wL, cross(wL, r));
        float3 fm = mass.x * (ap - gL) - partF[pi] + childF[pi];
        float3 tauCom = mass.yzw * alL + cross(wL, mass.yzw * wL);
        float3 mContactCom = partM[pi] - cross(r, partF[pi]);
        float3 mChildCom = childM[pi] - cross(r, childF[pi]);
        float3 tm = tauCom - cross(mount.xyz - r, fm) - mContactCom + mChildCom;
        if (parent < 32u)
        {
            childF[parent] += fm;
            childM[parent] += tm + cross(mount.xyz, fm);
        }
        if (mount.w >= 1e29 && load.x >= 1e29) continue;
        float F = length(fm), Tq = length(tm);
        float ratio = max(F / mount.w, Tq / load.x);
        VSet(pr + P_LOAD, float4(load.x, F, Tq, max(load.w, ratio)));
        uint bit = 1u << pi;
        if (ratio > 1.0 && (brk.z & bit) == 0u && (brk.x & bit) == 0u)
        {
            brk.x |= bit;
            emitEvent(EVENT_MOUNT_BREAK, (v << 8u) | pi, asuint(F), asuint(Tq));
        }
    }

    float3 axl = axleLocal(vr);
    for (uint wi = 0u; wi < nW; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
        float4 wm = VD(wr + W_MOUNT);
        if (wm.x >= 1e29 && wm.y >= 1e29) continue;
        float4 wm2 = VD(wr + W_MOUNT2);
        // Tyre force from the ground on the chassis, as vehicle_post reports it.
        float3 fT = float3(0, 0, 0);
        float3 hitL = float3(0, 0, 0);
        if ((dyn.z & WD_CONTACT) != 0u)
        {
            float4 f = VD(wr + W_F);
            float3 fw = VD(wr + W_C2).xyz * f.x - VD(wr + W_C3).xyz * f.y - VD(wr + W_C4).xyz * f.z;
            fT = quatRotate(qi, fw);
            hitL = quatRotate(qi, VD(wr + W_HIT).xyz - c.position);
        }
        float3 center = VD(wr + W_HP).xyz + VD(wr + W_DIR).xyz * VD(wr + W_O_POS).w;
        float mw = VD(wr + W_MISC).x;
        float3 aw = aL + cross(alL, center) + cross(wL, cross(wL, center));
        float3 fm = mw * (aw - gL) - fT - wheelF[wi];
        float3 mContact = cross(hitL - center, fT) + wheelM[wi] - cross(center, wheelF[wi]);
        float3 tm = -cross(wm2.xyz - center, fm) - mContact;
        // Torque about the spin axis goes to the drive and brakes, not the mount.
        tm -= axl * dot(tm, axl);
        float F = length(fm), Tq = length(tm);
        float ratio = max(F / wm.x, Tq / wm.y);
        VSet(wr + W_MOUNT, float4(wm.x, wm.y, F, Tq));
        VSet(wr + W_MOUNT2, float4(wm2.xyz, max(wm2.w, ratio)));
        uint bit = 1u << wi;
        if (ratio > 1.0 && (brk.w & bit) == 0u && (brk.y & bit) == 0u)
        {
            brk.y |= bit;
            emitEvent(EVENT_MOUNT_BREAK, (v << 8u) | 0x80u | wi, asuint(F), asuint(Tq));
        }
    }
    VSetU(vr + V_BRK, brk);
}

// ---------------------------------------------------------------------------
// vehicle_post: integrate spin, outputs, ground reaction, transmission, sleep
// ---------------------------------------------------------------------------
void updateTransmission(uint vr, uint4 cfg, float dt, float forward, float rpm, float oldRpm, bool slipping)
{
    float4 d0 = VD(vr + V_D0);
    float4 d1 = VD(vr + V_D1);
    int gear = int(round(d0.y));
    int oldGear = gear;
    float clutch = d0.z;
    float switchLeft = d0.w;
    float releaseLeft = d1.x;
    float latencyLeft = d1.y;
    float4 tr0 = VD(vr + V_TR0);
    float4 tr1 = VD(vr + V_TR1);
    int nF = int(asuint(tr1.y));
    int nR = int(asuint(tr1.z));
    uint req = VU(vr + V_IN1).x;
    bool manual = (cfg.w & VF_MANUAL_TRANS) != 0u && req != GEAR_AUTO;
    if (manual)
    {
        gear = clamp(int(req), -nR, nF);
    }
    else
    {
        bool canShiftUp = !slipping && rpm >= oldRpm;
        if (gear == 0 || forward * float(gear) < 0.0)
            gear = forward > 0.0 ? 1 : (forward < 0.0 ? -1 : 0);
        else if (latencyLeft == 0.0)
        {
            if (canShiftUp && rpm > tr0.w)
            {
                if (gear < 0) { if (gear > -nR) gear--; }
                else if (gear < nF) gear++;
            }
            else if (rpm < tr1.x)
            {
                if (gear < 0) { int maxG = forward != 0.0 ? -1 : 0; if (gear < maxG) gear++; }
                else { int minG = forward != 0.0 ? 1 : 0; if (gear > minG) gear--; }
            }
        }
    }
    if (oldGear != gear)
    {
        switchLeft = oldGear != 0 ? tr0.x : 0.0;
        releaseLeft = tr0.y;
        latencyLeft = tr0.z;
        clutch = 0.0;
    }
    else if (switchLeft > 0.0) { switchLeft = max(0.0, switchLeft - dt); clutch = 0.0; }
    else if (releaseLeft > 0.0)
    {
        releaseLeft = max(0.0, releaseLeft - dt);
        clutch = tr0.y > 0.0 ? 1.0 - releaseLeft / tr0.y : 1.0;
    }
    else
    {
        clutch = 1.0;
        latencyLeft = max(0.0, latencyLeft - dt);
    }
    VSet(vr + V_D0, float4(d0.x, float(gear), clutch, switchLeft));
    VSet(vr + V_D1, float4(releaseLeft, latencyLeft, d1.z, d1.w));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_post(uint3 tid : SV_DispatchThreadID)
{
    uint v = dispatchIndex(tid);
    if (v >= vehicleCount()) return;
    uint vr = vehRec(v);
    uint4 cfg = VU(vr + V_CFG);
    uint chassis = cfg.x;
    BodyState c = gBodies[chassis];
    bool active = (VU(vr + V_D3).y & VS_ACTIVE) != 0u;
    AvbdParams p = PARAMS;
    float dt = p.dt;
    float3 fwdLocal = vehicleFwdLocal(vr);
    float3 upLocal = vehicleUpLocal(vr);
    float3 axl = axleLocal(vr);
    float3 fwdW = quatRotate(c.orientation, fwdLocal);
    float3 upW = quatRotate(c.orientation, upLocal);
    uint nW = min(cfg.z, MAX_VEHICLE_WHEELS);
    bool keepAwake = false;
    bool slipping = false;
    uint nContact = 0u;
    uint driveMask = asuint(VD(vr + V_D2).w);
    float3 dl, da;
    chassisDisplacement(c, dl, da);
    float alphaStab = 1.0 - p.alpha;
    updateSeverGrace(vr, cfg, dt);

    uint anchor = VU(vr + V_POOL2).z;
    for (uint wi = 0u; wi < nW; wi++)
    {
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u)
        {
            parkWheelShell(wr, anchor);
            continue;
        }
        float4 w = VD(wr + W_W);
        float4 hp = VD(wr + W_HP);
        float4 susp = VD(wr + W_SUSP);
        float R = hp.w;
        bool contact = (dyn.z & WD_CONTACT) != 0u;
        if (active)
        {
            float y = VD(wr + W_Y).x;
            w.x = y / dt;
            w.y = fmod(w.y + y, TWO_PI);
            VSet(wr + W_W, w);
        }
        if (abs(w.x) > 0.2) keepAwake = true;

        // Suspension length at the solved pose.
        float L = susp.y;
        float3 F = float3(0, 0, 0); // force from the ground on the chassis
        float4 f = VD(wr + W_F);
        if (contact)
        {
            nContact++;
            float4 c1 = VD(wr + W_C1);
            float4 c2 = VD(wr + W_C2);
            float3 eg = wheelGroundDisp(wr);
            float3 t = VD(wr + W_C3).xyz;
            float3 s = VD(wr + W_C4).xyz;
            Vec6 jn = jac(c2.xyz, suspensionLever(wr, c1.xyz, c.initialRot));
            float un = jdot(jn, dl, da) - dot(c2.xyz, eg);
            L = clamp(c1.w + c2.w * un, susp.x, susp.y);
            F = c2.xyz * f.x - t * f.y - s * f.z;
            if (active)
            {
                // Stick anchors persist across steps (zero-creep parking).
                float3 upS = quatRotate(c.initialRot, upLocal);
                float4 k2 = VD(wr + W_K2);
                float y = VD(wr + W_Y).x;
                float CL = alphaStab * k2.z + jdot(jac(t, c1.xyz), dl, da) - dot(t, eg) - R * y;
                float CT = alphaStab * k2.w + jdot(jac(s, lateralLever(wr, c1.xyz, upS)), dl, da) - dot(s, eg);
                k2.z = (dyn.z & WD_STICK_L) != 0u ? clamp(CL, -0.05, 0.05) : 0.0;
                k2.w = (dyn.z & WD_STICK_T) != 0u ? clamp(CT, -0.05, 0.05) : 0.0;
                VSet(wr + W_K2, k2);
            }
            if ((driveMask & (1u << wi)) != 0u && VD(wr + W_C3).w > 0.1) slipping = true;
            // A moving ground body keeps the vehicle awake (its reaction is already
            // in the solve: two-body tyre rows).
            uint gi = wheelGround(wr);
            if (gi != NO_INDEX && length(gBodies[gi].linearVel) > p.sleepLin) keepAwake = true;
        }
        else
        {
            dyn.z &= ~(WD_STICK_L | WD_STICK_T | WD_SLIDE);
            VSetU(wr + W_C0, dyn);
            if ((driveMask & (1u << wi)) != 0u) slipping = true;
        }
        if (active || !bodyIsSleeping(c))
        {
            // Visual transform at the solved pose.
            float steer = VD(wr + W_C5).w;
            float3 center = transformPoint(c.position, c.orientation, hp.xyz + VD(wr + W_DIR).xyz * L);
            float4 rot = quatNormalize(quatMul(quatMul(c.orientation, wheelSteerRot(wr, steer)), quatAxisAngle(axl, w.y)));
            VSet(wr + W_O_POS, float4(center, L));
            VSet(wr + W_O_ROT, rot);
            placeWheelShell(wr, chassis, anchor, (VU(wr + W_CFG).y & WF_ENABLED) != 0u, L, steer);
            float3 hpW = transformPoint(c.position, c.orientation, hp.xyz);
            float3 hit = VD(wr + W_HIT).xyz;
            float4 qi = quatConjugate(c.orientation);
            float3 torque = cross(hit - hpW, F);
            VSet(wr + W_O_FM, float4(quatRotate(qi, F), f.x));
            VSet(wr + W_O_TM, float4(quatRotate(qi, torque), f.y));
            VSet(wr + W_O_CP, float4(hit, f.z));
            uint outFlags = dyn.z;
            VSet(wr + W_O_ST, float4(w.x, VD(wr + W_C3).w, degrees(VD(wr + W_C4).w), asfloat(outFlags)));
            VSet(wr + W_O_N, float4(VD(wr + W_C2).xyz, asfloat(contact ? dyn.x : NO_INDEX)));
        }
    }

    if ((cfg.w & VF_TRACKED) != 0u) updateTrackBelts(vr, cfg, anchor);

    float speedFwd = dot(c.linearVel, fwdW);
    float4 d0 = VD(vr + V_D0);
    float rpm = d0.x * RAD_PER_S_TO_RPM;
    if (active && (cfg.w & VF_DIRECT_DRIVE) == 0u)
    {
        float oldRpm = VD(vr + V_REFP).w;
        if ((cfg.w & VF_TRACKED) != 0u)
        {
            float4 td = VD(vr + V_TRACK_D);
            // Jolt: the sprocket is usually off the ground, so ignore wheel slip and
            // shift up whenever both tracks roll the same way.
            slipping = td.x * td.y <= 0.0;
        }
        updateTransmission(vr, cfg, dt, VD(vr + V_D1).z, rpm, oldRpm, slipping);
        d0 = VD(vr + V_D0);
    }
    float3 latW = cross(upW, fwdW);
    VSet(vr + V_O0, float4(speedFwd, rpm, d0.y, asfloat(VU(vr + V_D3).y)));
    VSet(vr + V_O1, float4(dot(c.linearVel, latW), dot(c.angularVel, upW), d0.z, asfloat(nContact)));

    // Sleep gating: intent, spinning wheels, shifting, or moving ground keep it awake.
    if (!active) return;
    float forward = VD(vr + V_D1).z;
    float4 d1 = VD(vr + V_D1);
    bool shifting = d0.w > 0.0 || d1.x > 0.0;
    float minRpm = VD(vr + V_ENG0).y;
    bool engineIdle = (cfg.w & VF_DIRECT_DRIVE) != 0u || rpm <= 1.01 * minRpm || VU(vr + V_O3).x == 0u;
    if (abs(forward) > 1e-4 || shifting || !engineIdle || (cfg.w & VF_CAN_SLEEP) == 0u) keepAwake = true;
    if (keepAwake)
    {
        gCcdFlags[chassis] = 0u;
        gBodies[chassis].sleepTimer = 0.0;
    }
}

// ---------------------------------------------------------------------------
// raycast_batch: public scene query (one workgroup per ray)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void raycast_batch(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint ray = gid.x + gid.y * 32768u;
    uint lane = lid.x;
    uint4 misc = VU(VH_MISC);
    bool live = ray < misc.x;
    uint rr = VU(VH_POOLS).w + ray * RAY_STRIDE;
    float4 a = live ? VD(rr) : float4(0, 0, 0, 0);
    float4 b = live ? VD(rr + 1u) : float4(0, -1, 0, 0);
    float3 dir = live ? normalize(b.xyz) : float3(0, -1, 0);
    rayLaneScan(lane, live, a.xyz, dir, a.w, asuint(b.w), 0.0, -2.0, 0.0);
    rayReduce(lane);
    if (!live || lane != 0u) return;
    uint best = gsRayC[0];
    if (best == NO_INDEX)
    {
        VSet(rr + 2u, float4(0, 0, 0, -1.0));
        VSet(rr + 3u, float4(0, 0, 0, asfloat(NO_INDEX)));
        return;
    }
    float t;
    float3 n;
    rayCollider(best, a.xyz, dir, 0.0, t, n);
    // A ray starting inside a collider hits it at its origin.
    float tHit = max(gsRayT[0], 0.0);
    if (gsRayT[0] < 0.0) n = -dir;
    VSet(rr + 2u, float4(a.xyz + dir * tHit, tHit));
    VSet(rr + 3u, float4(n, asfloat(best)));
}

// ---------------------------------------------------------------------------
// Soil ruts (deformable terrain T5; engine/docs/deformable-terrain.md "T5 ruts")
