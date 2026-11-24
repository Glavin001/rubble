// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void wheel_query(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint w = gid.x + gid.y * 32768u;
    uint lane = lid.x;
    bool live = w < wheelCountTotal();
    uint wr = live ? wheelRec(w) : 0u;
    uint v = live ? VU(wr + W_CFG).x : 0u;
    uint vr = live ? vehRec(v) : 0u;
    uint chassis = live ? VU(vr + V_CFG).x : 0u;
    if (live)
    {
        uint wd = VU(wr + W_C0).z;
        uint wf = VU(wr + W_CFG).y;
        BodyState c = gBodies[chassis];
        live = (VU(vr + V_CFG).w & VF_ENABLED) != 0u && (wf & WF_ENABLED) != 0u
            && (wd & (WD_DETACHED | WD_DISABLED)) == 0u && bodyIsDynamic(c) && !bodyIsSleeping(c);
    }
    float3 o = float3(0, 0, 0);
    float3 d = float3(0, -1, 0);
    float maxT = 0.0;
    float sweep = 0.0; // sphere-cast radius (0 = ray)
    float coneCos = 1.0;
    float lmin = 0.0;
    // Query rate (Jolt mNumStepsBetweenCollisionTest): between full probes a wheel
    // that had a contact slides it along the last hit plane, which moves with its
    // ground body. Staggered so a vehicle's wheels do not probe on the same step.
    bool probe = live;
    if (live)
    {
        BodyState c = gBodies[chassis];
        o = transformPoint(c.position, c.orientation, VD(wr + W_HP).xyz);
        d = normalize(quatRotate(c.orientation, VD(wr + W_DIR).xyz));
        bool sphere = (VU(vr + V_CFG).w & VF_SPHERE_CAST) != 0u;
        sweep = sphere ? VD(wr + W_HP).w : 0.0;
        // The tyre rows own the bottom cone (≤ 35° off the suspension axis for a
        // sphere cast, which also finds things beside the tyre); the wheel shell
        // (spheres from 60°) owns the rest, so the suspension never pushes along
        // its axis against a side hit.
        coneCos = sphere ? max(VD(vr + V_UP).w, SWEEP_CONE_COS) : VD(vr + V_UP).w;
        lmin = VD(wr + W_SUSP).x;
        maxT = VD(wr + W_SUSP).y + (sphere ? 0.0 : VD(wr + W_HP).w);
        uint interval = VU(vr + V_QUERY).x;
        bool hadHit = (VU(wr + W_C0).z & WD_CONTACT) != 0u && VD(wr + W_LHIT).w > 0.5;
        if (interval > 1u && hadHit && ((HOST_S(17) + w) % interval) != 0u) probe = false;
    }
    gRayExcludeGroup = (live && (HOST_S(30) & 3u) != 0u) ? bodyNegativeGroup(chassis) : 0;
    // Query mode in the vehicle header: 0 = broadphase grid walk, 1 = full scan.
    if (VU(VH_MISC).w == 1u) rayLaneScan(lane, probe, o, d, maxT, chassis, sweep, coneCos, lmin);
    else rayLaneScanGrid(lane, probe, o, d, maxT, chassis, sweep, coneCos, lmin);
    rayReduce(lane);
    if (!live || lane != 0u) return;

    BodyState c = gBodies[chassis];
    float radius = VD(wr + W_HP).w;
    uint4 dyn = VU(wr + W_C0);
    bool contact = false;
    float3 n = float3(0, 1, 0);
    float tHit = 0.0;
    uint best = NO_INDEX;
    uint ground = NO_INDEX;
    if (probe)
    {
        best = gsRayC[0];
        tHit = gsRayT[0];
        if (best != NO_INDEX)
        {
            // The scan only keeps hits the tyre cone accepts (tyreAccepts).
            float t2;
            float3 n2;
            rayCollider(best, o, d, sweep, t2, n2);
            n = dot(n2, d) > 0.0 ? -n2 : n2;
            contact = true;
        }
        ground = contact ? gColliders[best].body : NO_INDEX;
    }
    else
    {
        // Extrapolate on the stored hit plane (ground-body space).
        ground = dyn.x;
        best = dyn.y;
        BodyState g = gBodies[ground];
        float3 pc = transformPoint(g.position, g.orientation, VD(wr + W_LHIT).xyz);
        n = normalize(quatRotate(g.orientation, VD(wr + W_LN).xyz));
        float dn = dot(n, d);
        if (dn < -0.1)
        {
            // Sphere: centre one radius above the plane; ray: on the plane.
            tHit = (sweep - dot(n, o - pc)) / dn;
            contact = tHit >= 0.0 && tHit <= maxT && dot(n, -d) >= VD(vr + V_UP).w;
        }
        if (!contact) { ground = NO_INDEX; best = NO_INDEX; }
    }
    if (!contact || ground != dyn.x)
    {
        // New ground (or none): forget friction anchors and multipliers.
        float4 lam = VD(wr + W_LAM);
        VSet(wr + W_LAM, float4(0.0, 0.0, lam.z, lam.w));
        float4 k2 = VD(wr + W_K2);
        VSet(wr + W_K2, float4(k2.x, k2.y, 0.0, 0.0));
        dyn.z &= ~(WD_STICK_L | WD_STICK_T);
    }
    dyn.x = ground;
    dyn.y = contact ? best : NO_INDEX;
    dyn.z = contact ? (dyn.z | WD_CONTACT) : (dyn.z & ~(WD_CONTACT | WD_SUSP | WD_BUMP | WD_SLIDE));
    VSetU(wr + W_C0, dyn);
    if (!contact)
    {
        VSet(wr + W_C1, float4(0, 0, 0, VD(wr + W_SUSP).y));
        VSet(wr + W_HIT, float4(o + d * maxT, maxT));
        VSet(wr + W_LHIT, float4(0, 0, 0, 0));
        return;
    }
    // Ray: the hit is the contact. Sphere: the hit is the wheel centre; the
    // contact lies one radius along -n (edges give edge normals).
    float3 hit = sweep > 0.0 ? o + d * tHit - n * radius : o + d * tHit;
    float invCos = 1.0 / max(0.3, -dot(d, n));
    VSet(wr + W_C1, float4(hit - c.position, sweep > 0.0 ? tHit : tHit - radius));
    VSet(wr + W_C2, float4(n, invCos));
    VSet(wr + W_HIT, float4(hit, tHit));
    BodyState g = gBodies[ground];
    if (probe)
    {
        float4 gi = quatConjugate(g.orientation);
        VSet(wr + W_LHIT, float4(quatRotate(gi, hit - g.position), 1.0));
        VSet(wr + W_LN, float4(quatRotate(gi, n), 0.0));
    }
    float3 gv = float3(0, 0, 0);
    if (bodyIsDynamic(g) && !bodyIsSleeping(g))
        gv = g.linearVel + cross(g.angularVel, hit - g.position);
    VSet(wr + W_GV, float4(gv, VD(wr + W_GV).w));
}

// ---------------------------------------------------------------------------
// Drivetrain (Jolt WheeledVehicleController::PostCollide port) → per-wheel τ_drive
// ---------------------------------------------------------------------------
bool gaussSolve(inout float a[81], inout float b[9], uint n)
{
    for (uint col = 0u; col < n; col++)
    {
        uint piv = col;
        float best = abs(a[col * 9u + col]);
        for (uint r = col + 1u; r < n; r++)
        {
            float vv = abs(a[r * 9u + col]);
            if (vv > best) { best = vv; piv = r; }
        }
        if (best < 1e-12) return false;
        if (piv != col)
        {
            for (uint k = 0u; k < n; k++)
            {
                float tmp = a[col * 9u + k];
                a[col * 9u + k] = a[piv * 9u + k];
                a[piv * 9u + k] = tmp;
            }
            float tb = b[col]; b[col] = b[piv]; b[piv] = tb;
        }
        float inv = 1.0 / a[col * 9u + col];
        for (uint r = 0u; r < n; r++)
        {
            if (r == col) continue;
            float f = a[r * 9u + col] * inv;
            if (f == 0.0) continue;
            for (uint k = col; k < n; k++) a[r * 9u + k] -= f * a[col * 9u + k];
            b[r] -= f * b[col];
        }
    }
    for (uint r = 0u; r < n; r++) b[r] /= a[r * 9u + r];
    return true;
}

float gearRatio(uint vr, int gear)
{
    uint nF = asuint(VD(vr + V_TR1).y);
    uint nR = asuint(VD(vr + V_TR1).z);
    int idx = -1;
    if (gear > 0 && uint(gear) <= nF) idx = gear - 1;
    else if (gear < 0 && uint(-gear) <= nR) idx = int(nF) + (-gear) - 1;
    if (idx < 0) return 0.0;
    float4 g = VD(vr + (idx < 4 ? V_GEARS0 : V_GEARS1));
    uint k = uint(idx) & 3u;
    return k == 0u ? g.x : (k == 1u ? g.y : (k == 2u ? g.z : g.w));
}

void computeDrivetrain(uint vr, uint4 cfg, float dt, float speedFwd, inout float tauDrive[MAX_VEHICLE_WHEELS])
{
    for (uint wi = 0u; wi < MAX_VEHICLE_WHEELS; wi++) tauDrive[wi] = 0.0;
    float4 d1 = VD(vr + V_D1);
    float forward = d1.z;
    uint driveMask = asuint(VD(vr + V_D2).w);
    bool engineConnected = VU(vr + V_O3).x != 0u;

    if ((cfg.w & VF_DIRECT_DRIVE) != 0u)
    {
        float maxT = VD(vr + V_AERO1).w;
        float top = VD(vr + V_RESP).w;
        float s = abs(speedFwd);
        float scale = 1.0;
        if (top > 0.0 && forward * speedFwd > 0.0)
            scale = s < top / 3.0 ? 1.0 : clamp((top - s) / (top * 2.0 / 3.0), 0.0, 1.0);
        if (!engineConnected) scale = 0.0;
        for (uint wi = 0u; wi < cfg.z && wi < MAX_VEHICLE_WHEELS; wi++)
        {
            if ((driveMask & (1u << wi)) == 0u) continue;
            float mult = VD(wheelRec(cfg.y + wi) + W_UP).w;
            tauDrive[wi] = forward * maxT * scale * mult;
        }
        return;
    }

    float4 eng0 = VD(vr + V_ENG0);
    float4 eng1 = VD(vr + V_ENG1);
    float4 d0 = VD(vr + V_D0);
    int gear = int(round(d0.y));
    float ratio = gearRatio(vr, gear);
    float clutchFriction = d0.z;
    float omegaE = d0.x;
    float minW = eng0.y / RAD_PER_S_TO_RPM;
    float maxW = eng0.z / RAD_PER_S_TO_RPM;
    // Remember RPM before this step for the upshift gate.
    float4 refp = VD(vr + V_REFP);
    VSet(vr + V_REFP, float4(refp.xyz, omegaE * RAD_PER_S_TO_RPM));

    if ((cfg.w & VF_TRACKED) != 0u)
    {
        // Jolt TrackedVehicleController::PostCollide: the engine is rigidly tied
        // to the fastest track; each track gets diffRatio·ratio·transmission torque
        // up to the speed the engine allows, then the tracks are synced to the
        // left/right ratio conserving angular momentum.
        uint4 tl = VU(vr + V_TRACK);
        float4 td = VD(vr + V_TRACK_D);
        uint lead[2] = { tl.x, tl.y };
        float tr[2] = { td.x, td.y };
        bool live[2];
        for (uint i = 0u; i < 2u; i++)
            live[i] = lead[i] < cfg.z && (driveMask & (1u << lead[i])) != 0u;
        bool connected = gear != 0 && clutchFriction > 1e-3 && engineConnected && (live[0] || live[1]);
        if (connected)
        {
            bool fwdGear = ratio >= 0.0;
            float fastest = fwdGear ? -1e30 : 1e30;
            for (uint i = 0u; i < 2u; i++)
            {
                if (!live[i]) continue;
                uint wr = wheelRec(cfg.y + lead[i]);
                float s = VD(wr + W_W).x * VD(wr + W_UP).w;
                fastest = fwdGear ? max(fastest, s) : min(fastest, s);
            }
            omegaE = clamp(fastest * ratio, minW, maxW);
        }
        else
        {
            omegaE = clamp(omegaE * max(0.0, 1.0 - eng1.x * dt), minW, maxW);
            float freeThrottle = (cfg.w & VF_MANUAL_TRANS) != 0u && engineConnected ? abs(forward) : 0.0;
            float tq = freeThrottle * eng0.x * curveEval(asuint(eng1.y), omegaE * RAD_PER_S_TO_RPM / max(eng0.z, 1.0));
            omegaE = clamp(omegaE + tq * dt / max(eng0.w, 1e-6), minW, maxW);
        }
        float rpm = omegaE * RAD_PER_S_TO_RPM;
        float transmission = engineConnected
            ? clutchFriction * ratio * abs(forward) * eng0.x * curveEval(asuint(eng1.y), rpm / max(eng0.z, 1.0)) : 0.0;
        if (transmission != 0.0)
        {
            for (uint i = 0u; i < 2u; i++)
            {
                if (!live[i] || tr[i] == 0.0) continue;
                uint wr = wheelRec(cfg.y + lead[i]);
                float diff = VD(wr + W_UP).w;
                float trackMax = rpm / (ratio * diff * tr[i] * RAD_PER_S_TO_RPM) * 1.001;
                float w0 = VD(wr + W_W).x;
                if (w0 * trackMax < 0.0 || abs(w0) < abs(trackMax)) tauDrive[lead[i]] = diff * tr[i] * transmission;
            }
        }
        // Regenerative steering: the torque the inner track's steering brake took
        // last step drives the outer track (a geared steering differential), so
        // steering trades speed between the tracks instead of only braking.
        if (live[0] && live[1] && abs(tr[0]) != abs(tr[1]) && td.z != 0.0)
        {
            uint inner = abs(tr[0]) < abs(tr[1]) ? 0u : 1u;
            float lamB = VD(wheelRec(cfg.y + lead[inner]) + W_LAM).z;
            tauDrive[lead[1u - inner]] += lamB;
        }
        if (live[0] && live[1])
        {
            uint wl = wheelRec(cfg.y + lead[0]);
            uint wrr = wheelRec(cfg.y + lead[1]);
            float4 sl = VD(wl + W_W);
            float4 sr = VD(wrr + W_W);
            float il = max(VD(wl + W_INER).x, 1e-6);
            float ir = max(VD(wrr + W_INER).x, 1e-6);
            if (tr[0] * tr[1] > 0.0)
            {
                // Same direction: Jolt's momentum-conserving exchange.
                float denom = tr[0] * ir + tr[1] * il;
                if (abs(denom) > 1e-9)
                {
                    float impulse = (tr[0] * sr.x - tr[1] * sl.x) / denom;
                    sl.x += impulse * il;
                    sr.x -= impulse * ir;
                }
            }
            else
            {
                // Counter-rotating: Jolt's opposite-sign formula sends two equal track
                // speeds to 0/0 (any ratio "holds"), so a partial counter-rotation from
                // rolling never turns. Project onto right·ωl = left·ωr instead, with
                // the least inertia-weighted change.
                float g = tr[1] * tr[1] / il + tr[0] * tr[0] / ir;
                if (g > 1e-12)
                {
                    float lambda = (tr[0] * sr.x - tr[1] * sl.x) / g;
                    sl.x += lambda * tr[1] / il;
                    sr.x -= lambda * tr[0] / ir;
                }
            }
            VSet(wl + W_W, sl);
            VSet(wrr + W_W, sr);
        }
        VSet(vr + V_D0, float4(omegaE, d0.yzw));
        return;
    }

    float throttle = abs(forward);
    if ((cfg.w & VF_MANUAL_TRANS) == 0u) throttle *= clutchFriction;
    if (!engineConnected) throttle = 0.0;
    if ((cfg.w & VF_TRACTION_CONTROL) != 0u)
    {
        // Traction control on last step's slip, normalised by each tyre's curve peak:
        // full torque to 0.8× peak slip, down to a fifth by 1.6× (past the peak a
        // tyre trades its side grip for wheelspin). The slip ratio is regularised as
        // slip speed / max(|v|, 3 m/s): at a standstill κ = slip / |v| blows up and
        // would choke every launch.
        float kMax = 0.0;
        float vReg = abs(speedFwd) / max(abs(speedFwd), 3.0);
        for (uint wi = 0u; wi < cfg.z && wi < MAX_VEHICLE_WHEELS; wi++)
        {
            uint wr = wheelRec(cfg.y + wi);
            if ((driveMask & (1u << wi)) != 0u && (VU(wr + W_C0).z & WD_CONTACT) != 0u)
                kMax = max(kMax, VD(wr + W_C3).w * vReg / max(curvePeakX(asuint(VD(wr + W_TIRE).x)), 0.03));
        }
        // Low-pass the cut (τ 0.1 s): applied raw it chattered around the threshold
        // every step, which also amplified tiny backend (Metal vs WebGPU) differences.
        float4 q = VD(vr + V_QUERY);
        float cutTarget = 1.0 - clamp(1.0 - (kMax - 0.8) / 1.0, 0.2, 1.0);
        float cut = lerp(q.y, cutTarget, 1.0 - exp(-dt / 0.1));
        VSet(vr + V_QUERY, float4(q.x, cut, q.zw));
        throttle *= 1.0 - cut;
    }
    omegaE = clamp(omegaE * max(0.0, 1.0 - eng1.x * dt), minW, maxW);
    float rpm = omegaE * RAD_PER_S_TO_RPM;
    float engineTorque = throttle * eng0.x * curveEval(asuint(eng1.y), rpm / max(eng0.z, 1.0));

    // Driven differentials (missing / severed wheels drop out; ratios renormalise).
    uint4 pool = VU(vr + V_POOL);
    uint dIdx[MAX_DRIVEN_WHEELS];
    float dRatio[MAX_DRIVEN_WHEELS];
    float dTorque[MAX_DRIVEN_WHEELS];
    float dEst[MAX_DRIVEN_WHEELS];
    uint nd = 0u;
    float diffOmega[8];
    float diffRatio[8];
    uint diffL[8];
    uint diffR[8];
    float diffShare[8];
    uint nDiff = 0u;
    float omegaMin = 1e30, omegaMax = 0.0;
    float shareSum = 0.0;
    uint diffBase = VU(VH_POOLS).x;
    for (uint di = 0u; di < pool.y && di < 8u; di++)
    {
        uint dr = diffBase + (pool.x + di) * DIFF_STRIDE;
        uint4 du = VU(dr);
        float4 df = VD(dr);
        float4 df2 = VD(dr + 1u);
        uint l = du.x, r = du.y;
        if (l != NO_INDEX && (driveMask & (1u << l)) == 0u) l = NO_INDEX;
        if (r != NO_INDEX && (driveMask & (1u << r)) == 0u) r = NO_INDEX;
        if (l == NO_INDEX && r == NO_INDEX) continue;
        float avg = 0.0;
        float cnt = 0.0;
        if (l != NO_INDEX) { avg += VD(wheelRec(cfg.y + l) + W_W).x; cnt += 1.0; }
        if (r != NO_INDEX) { avg += VD(wheelRec(cfg.y + r) + W_W).x; cnt += 1.0; }
        avg = abs(avg * df.z / cnt);
        diffOmega[nDiff] = avg;
        diffRatio[nDiff] = df.z;
        diffL[nDiff] = l;
        diffR[nDiff] = r;
        diffShare[nDiff] = df2.y;
        shareSum += df2.y;
        omegaMin = min(omegaMin, avg);
        omegaMax = max(omegaMax, avg);
        nDiff++;
    }
    for (uint di = 0u; di < nDiff; di++) diffShare[di] = shareSum > 0.0 ? diffShare[di] / shareSum : 1.0 / float(nDiff);
    float interLs = eng1.w;
    if (nDiff > 1u && interLs < 1e29 && omegaMax > omegaMin)
    {
        float sumF = 0.0;
        float tmpF[8];
        for (uint di = 0u; di < nDiff; di++) { tmpF[di] = (omegaMax - diffOmega[di]) / (omegaMax - omegaMin); sumF += tmpF[di]; }
        float lo = max(1e-3, omegaMin), hi = max(1e-3, omegaMax);
        float a = min((hi / lo - 1.0) / (interLs - 1.0), 1.0);
        for (uint di = 0u; di < nDiff; di++) diffShare[di] = (1.0 - a) * diffShare[di] + a * tmpF[di] / sumF;
    }
    for (uint di = 0u; di < nDiff; di++)
    {
        uint dr = diffBase + (pool.x + di) * DIFF_STRIDE;
        float split = VD(dr).w;
        float ls = VD(dr + 1u).x;
        float c2w = ratio * diffRatio[di];
        uint l = diffL[di], r = diffR[di];
        if (l != NO_INDEX && r != NO_INDEX)
        {
            float wl = VD(wheelRec(cfg.y + l) + W_W).x;
            float wrr = VD(wheelRec(cfg.y + r) + W_W).x;
            float fl = 1.0 - split, fr = split;
            if (ls < 1e29)
            {
                float ol = max(1e-3, abs(wl)), orr = max(1e-3, abs(wrr));
                float a = min((max(ol, orr) / min(ol, orr) - 1.0) / (ls - 1.0), 1.0);
                if (ol < orr) { fl = fl * (1.0 - a) + a; fr = fr * (1.0 - a); }
                else { fl = fl * (1.0 - a); fr = fr * (1.0 - a) + a; }
            }
            if (nd < MAX_DRIVEN_WHEELS) { dIdx[nd] = l; dRatio[nd] = c2w; dTorque[nd] = diffShare[di] * fl; nd++; }
            if (nd < MAX_DRIVEN_WHEELS) { dIdx[nd] = r; dRatio[nd] = c2w; dTorque[nd] = diffShare[di] * fr; nd++; }
        }
        else if (nd < MAX_DRIVEN_WHEELS)
        {
            dIdx[nd] = l != NO_INDEX ? l : r; dRatio[nd] = c2w; dTorque[nd] = diffShare[di]; nd++;
        }
    }

    bool solved = false;
    if (nd > 0u)
    {
        uint n = nd + 1u;
        uint engine = nd;
        float a[81];
        float b[9];
        for (uint k = 0u; k < 81u; k++) a[k] = 0.0;
        for (uint k = 0u; k < 9u; k++) b[k] = 0.0;
        float S = ratio != 0.0 && engineConnected ? clutchFriction * eng1.z : 0.0;
        float dtDivIe = dt / max(eng0.w, 1e-6);
        float nf = float(nd);
        for (uint i = 0u; i < nd; i++)
        {
            uint wr = wheelRec(cfg.y + dIdx[i]);
            float inertia = max(VD(wr + W_INER).x, 1e-6);
            float sr = S * dRatio[i];
            float k = dt * sr * dTorque[i] / inertia;
            for (uint j = 0u; j < nd; j++) a[i * 9u + j] = k * dRatio[j] / nf;
            a[i * 9u + i] += 1.0;
            a[i * 9u + engine] = -k;
            // External wheel torque estimate: last step's tire force and brake.
            float R = VD(wr + W_HP).w;
            float4 lam = VD(wr + W_LAM);
            bool contactW = (VU(wr + W_C0).z & WD_CONTACT) != 0u;
            float dtTw = dt * ((contactW ? R * lam.x : 0.0) - lam.z);
            dEst[i] = dtTw;
            b[i] = VD(wr + W_W).x + dtTw / inertia;
            a[engine * 9u + i] = -dtDivIe * sr / nf;
        }
        a[engine * 9u + engine] = 1.0 + dtDivIe * S;
        b[engine] = omegaE + dtDivIe * engineTorque;
        if (gaussSolve(a, b, n))
        {
            for (uint i = 0u; i < nd; i++)
            {
                uint wr = wheelRec(cfg.y + dIdx[i]);
                float inertia = max(VD(wr + W_INER).x, 1e-6);
                float omegaJ = b[i] - dEst[i] / inertia;
                tauDrive[dIdx[i]] = inertia * (omegaJ - VD(wr + W_W).x) / dt;
            }
            omegaE = clamp(b[engine], minW, maxW);
            solved = true;
        }
    }
    if (!solved) omegaE = clamp(omegaE + engineTorque * dt / max(eng0.w, 1e-6), minW, maxW);
    VSet(vr + V_D0, float4(omegaE, d0.yzw));
}

// ---------------------------------------------------------------------------
// Row assembly (shared by prepare and iterate)
// ---------------------------------------------------------------------------
Vec6 jac(float3 e, float3 r)
{
    Vec6 j;
    j.v[0] = e.x; j.v[1] = e.y; j.v[2] = e.z;
    float3 a = cross(r, e);
    j.v[3] = a.x; j.v[4] = a.y; j.v[5] = a.z;
    return j;
}

float jdot(Vec6 j, float3 dl, float3 da)
{
    return j.v[0] * dl.x + j.v[1] * dl.y + j.v[2] * dl.z + j.v[3] * da.x + j.v[4] * da.y + j.v[5] * da.z;
}

// Suspension rows act at the contact, or at a fixed chassis point (Jolt
// mSuspensionForcePoint) when the wheel has one.
float3 suspensionLever(uint wr, float3 r, float4 q0)
{
    float4 fp = VD(wr + W_FP);
    return fp.w > 0.5 ? quatRotate(q0, fp.xyz) : r;
}

float3 lateralLever(uint wr, float3 r, float3 upW)
{
    float rollInf = VD(wr + W_MISC).w;
    return r - upW * (dot(r, upW) * (1.0 - rollInf));
}

// ---------------------------------------------------------------------------
// Two-body tyre rows. A wheel on a dynamic body (vehicle_link sets W_TJ.z) sees
// that body's current solver iterate at the contact instead of a given ground
// motion, and writes the body's share of every row (J_g = −[e; r_g × e], same
// stiffness and force as the chassis side) into its ground block, which the
// body's primal adds through the wheel's tyre joint (JOINT_FLAG_TYRE).
