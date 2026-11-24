// ---------------------------------------------------------------------------
uint wheelGround(uint wr) { return VU(wr + W_TJ).z; }

uint groundBlockIndex(uint wheelSlot)
{
    return VU(VH_BASES).w + 4u * VU(VH_COUNTS).x + 4u * wheelSlot;
}

// Ground displacement at the contact since x⁻: the body's current iterate for a
// two-body wheel, else the given motion v_g·dt (static / kinematic ground).
float3 wheelGroundDisp(uint wr)
{
    uint gi = wheelGround(wr);
    if (gi == NO_INDEX) return VD(wr + W_C5).xyz;
    BodyState gb = gBodies[gi];
    if (bodyIsSleeping(gb)) return float3(0, 0, 0);
    float4 q = gb.orientation;
    if (dot(q, gb.initialRot) < 0.0) q = -q;
    return (gb.position - gb.initialPos) + cross(quatSubVec(q, gb.initialRot), VD(wr + W_HIT).xyz - gb.initialPos);
}

Vec6 groundJac(uint wr, float3 e)
{
    BodyState gb = gBodies[wheelGround(wr)];
    Vec6 j = jac(e, VD(wr + W_HIT).xyz - gb.initialPos);
    for (int a = 0; a < 6; a++) j.v[a] = -j.v[a];
    return j;
}

// Add k·JJᵀ and f·J to a ground block (lower triangle + rhs, writeVehicleBlock layout).
void groundAdd(uint wheelSlot, Vec6 j, float k, float f)
{
    uint at = groundBlockIndex(wheelSlot);
    float b[32];
    for (uint e = 0u; e < 4u; e++)
    {
        Aabb a = gAabbs[at + e];
        b[e * 8u] = a.mn.x; b[e * 8u + 1u] = a.mn.y; b[e * 8u + 2u] = a.mn.z; b[e * 8u + 3u] = a._pad0;
        b[e * 8u + 4u] = a.mx.x; b[e * 8u + 5u] = a.mx.y; b[e * 8u + 6u] = a.mx.z; b[e * 8u + 7u] = a._pad1;
    }
    uint idx = 0u;
    for (uint r = 0u; r < 6u; r++)
        for (uint c2 = 0u; c2 <= r; c2++) { b[idx] += k * j.v[r] * j.v[c2]; idx++; }
    for (uint r = 0u; r < 6u; r++) b[21u + r] += f * j.v[r];
    b[27] = 1.0;
    for (uint e = 0u; e < 4u; e++)
    {
        Aabb a;
        a.mn = float3(b[e * 8u], b[e * 8u + 1u], b[e * 8u + 2u]);
        a._pad0 = b[e * 8u + 3u];
        a.mx = float3(b[e * 8u + 4u], b[e * 8u + 5u], b[e * 8u + 6u]);
        a._pad1 = b[e * 8u + 7u];
        gAabbs[at + e] = a;
    }
}

void groundClear(uint wheelSlot)
{
    uint at = groundBlockIndex(wheelSlot);
    Aabb z;
    z.mn = float3(0, 0, 0); z._pad0 = 0.0; z.mx = float3(0, 0, 0); z._pad1 = 0.0;
    for (uint e = 0u; e < 3u; e++) gAabbs[at + e] = z;
    z._pad0 = 1.0; // active (empty until rows add to it)
    gAabbs[at + 3u] = z;
}

// Assemble every wheel row at chassis displacement (dl, da) from x⁻.
// When dual is set, first update multipliers/penalties at this iterate.
// N: per-wheel scratch size (≥ the vehicle's wheels; see assembleVehicle).
// assembleVehicleN's passes, one function each (the serial path calls them in
// order; soft_primal_bodies' vehicle lanes split the per-wheel ones).
// Pass 1: suspension / bump stop of wheel wi.
void vehSuspension(uint wi, uint vr, uint4 cfg, BodyState c, float3 dl, float3 da, inout Mat6 H, inout Vec6 g,
    out float outL, out float outFn, out bool contact)
{
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));

        outL = 0.0; outFn = 0.0; contact = false;
        uint wr = wheelRec(cfg.y + wi);
        uint4 dyn = VU(wr + W_C0);
        if ((dyn.z & WD_CONTACT) == 0u) return;
        contact = true;
        float4 c1 = VD(wr + W_C1);
        float4 c2 = VD(wr + W_C2);
        float3 eg = wheelGroundDisp(wr);
        float4 k = VD(wr + W_K);
        float4 susp = VD(wr + W_SUSP);
        Vec6 jn = jac(c2.xyz, suspensionLever(wr, c1.xyz, c.initialRot));
        float un = jdot(jn, dl, da) - dot(c2.xyz, eg);
        float L = c1.w + c2.w * un;
        outL = L;
        float Lnat = susp.y + susp.z;
        float fS = k.x * (L - Lnat) + k.y * (L - c1.w);
        uint flags = dyn.z & ~(WD_SUSP | WD_BUMP);
        float fsTotal = 0.0;
        float kTotal = 0.0;
        if (fS < 0.0) { fsTotal += fS; kTotal += k.x + k.y; flags |= WD_SUSP; }
        if (L < susp.x)
        {
            float fB = k.z * (L - susp.x) + k.w * (L - c1.w);
            if (fB < 0.0) { fsTotal += fB; kTotal += k.z + k.w; flags |= WD_BUMP; }
        }
        // Optional physical limit (bump_max_force): past it the row is a constant force.
        float bumpMax = VD(wr + W_MISC).z;
        if (bumpMax > 0.0 && -fsTotal * c2.w > bumpMax)
        {
            fsTotal = -bumpMax / max(c2.w, 1e-6);
            kTotal = 0.0;
        }
        if (fsTotal != 0.0)
        {
            Vec6 js = jn;
            for (int a = 0; a < 6; a++) js.v[a] *= c2.w;
            mat6AddOuter(H, js, kTotal);
            vec6AddScaled(g, js, fsTotal);
            if (wheelGround(wr) != NO_INDEX)
            {
                Vec6 jg = groundJac(wr, c2.xyz);
                for (int a = 0; a < 6; a++) jg.v[a] *= c2.w;
                groundAdd(cfg.y + wi, jg, kTotal, fsTotal);
            }
        }
        outFn = max(0.0, -fsTotal * c2.w);
        dyn.z = flags;
        VSetU(wr + W_C0, dyn);
    }

void vehAntiRoll<let N : uint>(uint vr, uint4 cfg, BodyState c, uint nW, float Lw[N], inout float FnW[N],
    inout Mat6 H, inout Vec6 g)
{
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));
    // Pass 2: anti-roll bars (only when both wheels touch the ground, as Jolt).
    uint4 pool = VU(vr + V_POOL);
    uint arBase = VU(VH_POOLS).y;
    for (uint ai = 0u; ai < pool.w; ai++)
    {
        float4 ar = VD(arBase + (pool.z + ai) * ANTIROLL_STRIDE);
        uint l = asuint(ar.x), r = asuint(ar.y);
        if (l >= nW || r >= nW) continue;
        uint wl = wheelRec(cfg.y + l), wrr = wheelRec(cfg.y + r);
        if ((VU(wl + W_C0).z & WD_CONTACT) == 0u || (VU(wrr + W_C0).z & WD_CONTACT) == 0u) continue;
        float4 cl = VD(wl + W_C1), cr = VD(wrr + W_C1);
        float4 nl = VD(wl + W_C2), nr = VD(wrr + W_C2);
        Vec6 jl = jac(nl.xyz, suspensionLever(wl, cl.xyz, c.initialRot));
        Vec6 jr = jac(nr.xyz, suspensionLever(wrr, cr.xyz, c.initialRot));
        Vec6 gj;
        for (int a = 0; a < 6; a++) gj.v[a] = nl.w * jl.v[a] - nr.w * jr.v[a];
        float diff = Lw[l] - Lw[r];
        mat6AddOuter(H, gj, ar.z);
        vec6AddScaled(g, gj, ar.z * diff);
        // Each wheel's ground carries its end of the bar (same row, its own Jacobian).
        if (wheelGround(wl) != NO_INDEX)
        {
            Vec6 jg = groundJac(wl, nl.xyz);
            for (int a = 0; a < 6; a++) jg.v[a] *= nl.w;
            groundAdd(cfg.y + l, jg, ar.z, ar.z * diff);
        }
        if (wheelGround(wrr) != NO_INDEX)
        {
            Vec6 jg = groundJac(wrr, nr.xyz);
            for (int a = 0; a < 6; a++) jg.v[a] *= -nr.w;
            groundAdd(cfg.y + r, jg, ar.z, ar.z * diff);
        }
        FnW[l] = max(0.0, FnW[l] - ar.z * diff * nl.w);
        FnW[r] = max(0.0, FnW[r] + ar.z * diff * nr.w);
    }

}

    // Pass 3: friction rows + wheel spin. Each spin group (one wheel, or every
    // wheel of a track) shares one rotation y, solved exactly for the current
    // chassis iterate (1-D, monotone, piecewise linear in the tyre and brake
    // limits), then folded in with the Schur complement of the regime it lands
    // in: rolling (coupled) or saturated (decoupled). The group's inertia, brake
    // and drive live on its leader (the driven sprocket for a track).
void vehSpinGroup<let N : uint>(uint li, uint vr, uint4 cfg, BodyState c, float3 dl, float3 da, bool dual, uint nW,
    float FnW[N], inout Mat6 H, inout Vec6 g)
{
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));

        uint lr = wheelRec(cfg.y + li);
        uint leadOf = VU(lr + W_CFG).w;
        if (leadOf != NO_INDEX && leadOf != li) return; // a member: its leader handles it
        if ((VU(lr + W_C0).z & (WD_DETACHED | WD_DISABLED)) != 0u) return;
        float4 yv = VD(lr + W_Y);     // y, yTilde, mY, tauB (group)
        float4 lamLead = VD(lr + W_LAM);
        float4 k2Lead = VD(lr + W_K2);
        bool servo = yv.w > 0.0;

        uint mWheel[N];
        float mR[N];
        float mCL0[N];
        float mCT[N];
        float mKL[N];
        float mLamL[N];
        float mAL[N];
        float mAT[N];
        float mFT[N];
        bool mClampT[N];
        bool mContact[N];
        bool mClampL[N];
        float mFL[N];
        uint nm = 0u;
        float rAsum = 0.0;
        for (uint wi = 0u; wi < nW; wi++)
        {
            uint wr = wheelRec(cfg.y + wi);
            uint lead = VU(wr + W_CFG).w;
            if ((lead == NO_INDEX ? wi : lead) != li) continue;
            uint4 dyn = VU(wr + W_C0);
            if ((dyn.z & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
            float R = VD(wr + W_HP).w;
            float4 lam = VD(wr + W_LAM);
            float4 k2 = VD(wr + W_K2);
            float4 mu = VD(wr + W_MU);
            bool contact = (dyn.z & WD_CONTACT) != 0u;
            float Fn = FnW[wi];
            float CL0 = 0.0, CT = 0.0, aL = 0.0, aT = 0.0;
            if (contact)
            {
                float4 c1 = VD(wr + W_C1);
                float3 eg = wheelGroundDisp(wr);
                float3 t = VD(wr + W_C3).xyz;
                float3 s = VD(wr + W_C4).xyz;
                CL0 = alphaStab * k2.z + jdot(jac(t, c1.xyz), dl, da) - dot(t, eg);
                CT = alphaStab * k2.w + jdot(jac(s, lateralLever(wr, c1.xyz, upW)), dl, da) - dot(s, eg);
                aL = ((dyn.z & WD_STICK_L) != 0u ? mu.z : mu.x) * Fn;
                aT = ((dyn.z & WD_STICK_T) != 0u ? mu.w : mu.y) * Fn;
            }
            // Friction ellipse: split the budget by the raw forces at the previous y.
            float fTraw = k2.x * CT + lam.y;
            float fLprev = lam.w * (CL0 - R * yv.x) + lam.x;
            float fT = fTraw;
            float aLeff = aL;
            float tLim = aT; // lateral bound the dual update re-clamps to
            bool clampT = false;
            if (contact)
            {
                float e = (aL > 1e-9 && aT > 1e-9)
                    ? (fLprev / aL) * (fLprev / aL) + (fTraw / aT) * (fTraw / aT) : 2.0;
                if (aT <= 1e-9) { fT = 0.0; clampT = true; }
                else if ((VU(wr + W_CFG).y & WF_BOX_FRICTION) != 0u)
                {
                    fT = clamp(fTraw, -aT, aT);
                    clampT = abs(fTraw) > aT;
                }
                else if (e > 1.0)
                {
                    // Saturated. Near the grip peak split the ellipse by the raw row
                    // forces; well past it (a locked or spinning wheel) a sliding
                    // tyre's force opposes its slip velocity, so share it along the
                    // peak-normalised slip (κ/κ̂, tan α/tan α̂), Pacejka's similarity
                    // method. Raw forces alone let the stiff lateral row keep its grip
                    // on a locked wheel (no handbrake turns); the slip direction alone
                    // stripped a driven tyre's side grip at κ ≈ 0.05 on a straight
                    // (bikes swapped ends under throttle).
                    float sc = rsqrt(max(e, 1e-12));
                    float tProp = abs(fTraw) * sc;
                    float lProp = abs(fLprev) * sc;
                    float4 tire = VD(wr + W_TIRE);
                    float kPk = max(curvePeakX(asuint(tire.x)), 0.03);
                    float aPk = tan(radians(max(curvePeakX(asuint(tire.y)), 1.5)));
                    float nx = VD(wr + W_C3).w / kPk;
                    float ny = tan(min(VD(wr + W_C4).w, 1.45)) / aPk;
                    float nn = sqrt(nx * nx + ny * ny);
                    float wSlip = nn > 1e-6 ? clamp(max(nx, ny) - 1.0, 0.0, 1.0) : 0.0;
                    float tCap = lerp(tProp, nn > 1e-6 ? aT * ny / nn : tProp, wSlip);
                    aLeff = lerp(lProp, nn > 1e-6 ? aL * nx / nn : lProp, wSlip);
                    clampT = abs(fTraw) > tCap;
                    fT = clampT ? sign(fTraw) * tCap : fTraw;
                    tLim = max(tCap, abs(fT));
                }
                else aLeff = aL * sqrt(max(0.0, 1.0 - (fTraw / aT) * (fTraw / aT)));
            }
            mWheel[nm] = wi; mR[nm] = R; mCL0[nm] = CL0; mCT[nm] = CT; mKL[nm] = lam.w;
            mLamL[nm] = lam.x; mAL[nm] = contact ? aLeff : 0.0; mAT[nm] = tLim; mFT[nm] = fT;
            mClampT[nm] = clampT; mContact[nm] = contact; mClampL[nm] = false; mFL[nm] = 0.0;
            rAsum += R * mAL[nm];
            nm++;
        }

        float kB = k2Lead.y;
        float lamB = lamLead.z;
        float yStar = VD(lr + W_LN).w; // servo target (0 = brake to a stop this step)
        float y = yv.x;
        float fB = 0.0;
        bool clampB = false;
        for (uint pass = 0u; pass < (dual ? 2u : 1u); pass++)
        {
            // Solve G(y) = mY(y − ỹ) − Σ R·fL(y) + fB(y) = 0 (monotone increasing).
            float spread = (rAsum + yv.w) / max(yv.z, 1e-9) + 1e-6;
            float lo = yv.y - spread, hi = yv.y + spread;
            y = clamp(y, lo, hi);
            for (uint it = 0u; it < 24u; it++)
            {
                float G = yv.z * (y - yv.y);
                float slope = yv.z;
                for (uint m = 0u; m < nm; m++)
                {
                    if (!mContact[m]) continue;
                    float fLr = mKL[m] * (mCL0[m] - mR[m] * y) + mLamL[m];
                    mFL[m] = clamp(fLr, -mAL[m], mAL[m]);
                    mClampL[m] = abs(fLr) > mAL[m];
                    G -= mR[m] * mFL[m];
                    if (!mClampL[m]) slope += mR[m] * mR[m] * mKL[m];
                }
                float fBr = kB * (y - yStar) + lamB;
                fB = servo ? clamp(fBr, -yv.w, yv.w) : 0.0;
                clampB = servo && abs(fBr) > yv.w;
                G += fB;
                if (!clampB && servo) slope += kB;
                if (abs(G) <= 1e-7 * (yv.z * abs(yv.y) + rAsum + yv.w + 1.0)) break;
                if (G > 0.0) hi = y; else lo = y;
                float yn = y - G / slope;
                if (yn <= lo || yn >= hi) yn = 0.5 * (lo + hi);
                y = yn;
            }
            if (!dual || pass == 1u) break;
            // AVBD dual update at this iterate, then re-solve with the new multipliers.
            AvbdParams pp = PARAMS;
            for (uint m = 0u; m < nm; m++)
            {
                uint wr = wheelRec(cfg.y + mWheel[m]);
                float4 lam = VD(wr + W_LAM);
                float4 k2 = VD(wr + W_K2);
                if (mContact[m])
                {
                    float kCap = 1e3 * max(VD(wr + W_W).w, 1.0) / (pp.dt * pp.dt);
                    float CL = mCL0[m] - mR[m] * y;
                    lam.x = mFL[m];
                    lam.y = mFT[m];
                    if (!mClampL[m] && !mClampT[m])
                    {
                        lam.w = min(lam.w + pp.betaLin * abs(CL), kCap);
                        k2.x = min(k2.x + pp.betaLin * abs(mCT[m]), kCap);
                    }
                    uint4 dyn = VU(wr + W_C0);
                    uint stick = (!mClampL[m] ? WD_STICK_L : 0u) | (!mClampT[m] ? WD_STICK_T : 0u);
                    dyn.z = (dyn.z & ~(WD_STICK_L | WD_STICK_T | WD_SLIDE)) | stick
                        | ((mClampL[m] || mClampT[m]) ? WD_SLIDE : 0u);
                    VSetU(wr + W_C0, dyn);
                    mKL[m] = lam.w;
                    mLamL[m] = lam.x;
                    mFT[m] = clamp(k2.x * mCT[m] + lam.y, -mAT[m], mAT[m]);
                }
                if (mWheel[m] == li)
                {
                    lam.z = servo ? fB : 0.0;
                    if (servo && !clampB) k2.y = min(k2.y + pp.betaAng * abs(y - yStar), 1e3 * k2.y);
                    kB = k2.y;
                    lamB = lam.z;
                }
                VSet(wr + W_LAM, lam);
                VSet(wr + W_K2, k2);
            }
        }

        // Schur complement of the shared spin: H += Σ k JᵀJ − u uᵀ / A, u = Σ R·k·Jt.
        float kBa = (servo && !clampB) ? kB : 0.0;
        float A = yv.z + kBa;
        for (uint m = 0u; m < nm; m++)
            if (mContact[m] && !mClampL[m]) A += mR[m] * mR[m] * mKL[m];
        Vec6 u = vec6Zero();
        bool coupled = false;
        for (uint m = 0u; m < nm; m++)
        {
            uint wr = wheelRec(cfg.y + mWheel[m]);
            VSet(wr + W_Y, float4(y, yv.yzw));
            VSet(wr + W_F, float4(FnW[mWheel[m]], mFL[m], mFT[m], VD(wr + W_F).w));
            if (!mContact[m]) continue;
            float4 c1 = VD(wr + W_C1);
            Vec6 jt = jac(VD(wr + W_C3).xyz, c1.xyz);
            Vec6 js = jac(VD(wr + W_C4).xyz, lateralLever(wr, c1.xyz, upW));
            // Saturated rows keep their penalty as a chassis-only proximal term.
            mat6AddOuter(H, jt, mKL[m]);
            vec6AddScaled(g, jt, mFL[m]);
            if (!mClampL[m])
            {
                vec6AddScaled(u, jt, mR[m] * mKL[m]);
                coupled = true;
            }
            mat6AddOuter(H, js, VD(wr + W_K2).x);
            vec6AddScaled(g, js, mFT[m]);
            // The ground's share, with wheel spin held at its solved y.
            if (wheelGround(wr) != NO_INDEX)
            {
                groundAdd(cfg.y + mWheel[m], groundJac(wr, VD(wr + W_C3).xyz), mKL[m], mFL[m]);
                groundAdd(cfg.y + mWheel[m], groundJac(wr, VD(wr + W_C4).xyz), VD(wr + W_K2).x, mFT[m]);
            }
        }
        if (coupled) mat6AddOuter(H, u, -1.0 / A);
        VSet(lr + W_BS, float4(0.0, A, 0.0, 0.0));
    }

void vehChassisRows<let N : uint>(uint vr, uint4 cfg, BodyState c, float3 dl, float3 da, bool dual, uint nW,
    float FnW[N], uint contacts, inout Mat6 H, inout Vec6 g)
{
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));
    // Pitch/roll limit (Jolt mMaxPitchRollAngle): one-sided angular AVBD row that
    // keeps the chassis up axis within a cone around world up.
    float cosMax = VD(vr + V_FWD).w;
    if (cosMax > -0.999)
    {
        float3 worldUp = float3(0.0, p.gravityY < 0.0 ? 1.0 : -1.0, 0.0);
        float cu = clamp(dot(upW, worldUp), -1.0, 1.0);
        float theta = acos(cu);
        float thetaMax = acos(clamp(cosMax, -1.0, 1.0));
        float4 o3 = VD(vr + V_O3);
        float lamPR = o3.z;
        float kPR = o3.w;
        float3 axis = cross(upW, worldUp);
        float al = length(axis);
        if (theta > thetaMax - 0.05 && al > 1e-6)
        {
            axis /= al;
            Vec6 j = vec6Zero();
            j.v[3] = -axis.x; j.v[4] = -axis.y; j.v[5] = -axis.z;
            float C = theta - thetaMax + jdot(j, dl, da);
            float4 m = VD(vr + V_MASS);
            float iAx = dot(abs(quatRotate(quatConjugate(c.initialRot), axis)), m.yzw);
            float kFloor = iAx / (p.dt * p.dt);
            kPR = max(kPR, kFloor);
            float fPR = max(0.0, kPR * C + lamPR);
            if (dual)
            {
                lamPR = fPR;
                if (fPR > 0.0) kPR = min(kPR + p.betaAng * abs(C), 1e3 * kFloor);
                fPR = max(0.0, kPR * C + lamPR);
            }
            if (fPR > 0.0)
            {
                mat6AddOuter(H, j, kPR);
                vec6AddScaled(g, j, fPR);
            }
        }
        else if (dual) { lamPR = 0.0; }
        if (dual) VSet(vr + V_O3, float4(o3.xy, lamPR, kPR));
    }

    // Motorcycle lean (Jolt MotorcycleController): a soft angular row about the
    // forward axis toward the target lean, with implicit damping and an explicit
    // integral term; only while every wheel is loaded. Like Jolt, the lean torque
    // comes with a linear force that cancels the load-weighted acceleration it
    // would give the contact patches, so it rolls the bike about its tyres
    // instead of dragging them sideways (that friction would feed back into the
    // target lean).
    float4 bike0 = VD(vr + V_BIKE0);
    if (bike0.x > 0.0 && contacts > 0u)
    {
        uint attached = 0u;
        uint loaded = 0u;
        float3 rBar = float3(0, 0, 0);
        float fnSum = 0.0;
        for (uint wi = 0u; wi < nW; wi++)
        {
            uint wr = wheelRec(cfg.y + wi);
            uint wd = VU(wr + W_C0).z;
            if ((wd & (WD_DETACHED | WD_DISABLED)) != 0u) continue;
            attached++;
            if ((wd & WD_CONTACT) == 0u || FnW[wi] <= 0.0) continue;
            loaded++;
            rBar += FnW[wi] * VD(wr + W_C1).xyz;
            fnSum += FnW[wi];
        }
        if (loaded == attached)
        {
            float3 fwd0 = quatRotate(c.initialRot, vehicleFwdLocal(vr));
            float4 bd = VD(vr + V_BIKE_D);
            float3 tgt = bd.xyz;
            float dAng = -sign(dot(cross(tgt, upW), fwd0)) * acos(clamp(dot(tgt, upW), -1.0, 1.0));
            Vec6 j = vec6Zero();
            j.v[3] = fwd0.x; j.v[4] = fwd0.y; j.v[5] = fwd0.z;
            float phi = jdot(j, dl, da);
            float k = bike0.y;
            float dd = bike0.z / p.dt;
            float f = -k * (dAng - phi) + dd * phi - bike0.w * bd.w;
            mat6AddOuter(H, j, k + dd);
            vec6AddScaled(g, j, f);
            // Torque −f·fwd gives Δω ∝ −f·I⁻¹fwd; the matching linear force is
            // −m·(Δω × r̄), i.e. a gradient of −m·f·(I⁻¹fwd × r̄).
            float4 mI = VD(vr + V_MASS);
            float3 invIf = quatRotate(c.initialRot, quatRotate(quatConjugate(c.initialRot), fwd0) / max(mI.yzw, float3(1e-6, 1e-6, 1e-6)));
            float3 comp = -mI.x * f * cross(invIf, rBar / fnSum);
            g.v[0] += comp.x; g.v[1] += comp.y; g.v[2] += comp.z;
        }
    }

    // Arcade assists (opt-in, explicit torques): air control while airborne and an
    // upright stabiliser, both scaled by the chassis inertia.
    float4 assist = VD(vr + V_ASSIST);
    if (assist.x != 0.0 || assist.y != 0.0 || assist.z != 0.0)
    {
        float3 alphaW = float3(0, 0, 0);
        if (contacts == 0u)
        {
            float forwardIn = VD(vr + V_D1).z;
            float steerIn = VD(vr + V_D2).y;
            float3 axleW = quatRotate(c.initialRot, axleLocal(vr));
            alphaW += axleW * (forwardIn * assist.x) - upW * (steerIn * assist.y);
        }
        if (assist.z > 0.0)
        {
            float3 worldUp = float3(0.0, p.gravityY < 0.0 ? 1.0 : -1.0, 0.0);
            float3 wPerp = c.angularVel - upW * dot(c.angularVel, upW);
            alphaW += cross(upW, worldUp) * assist.z - wPerp * assist.w;
        }
        float3 Ib = VD(vr + V_MASS).yzw;
        float3 local = quatRotate(quatConjugate(c.initialRot), alphaW) * Ib;
        float3 tau = quatRotate(c.initialRot, local);
        g.v[3] -= tau.x; g.v[4] -= tau.y; g.v[5] -= tau.z;
    }

    // Aerodynamics (explicit): drag and downforce at the aero center.
    float4 aero = VD(vr + V_AERO0);
    if (aero.x > 0.0 || aero.y > 0.0)
    {
        float3 vel = c.linearVel;
        float3 fwdW = quatRotate(c.initialRot, vehicleFwdLocal(vr));
        float vf = dot(vel, fwdW);
        float3 F = -aero.x * length(vel) * vel - aero.y * vf * vf * upW;
        float3 a = quatRotate(c.initialRot, VD(vr + V_AERO1).xyz);
        Vec6 jF = jac(F, a);
        vec6AddScaled(g, jF, -1.0);
    }
    // Per-vehicle gravity scale (0 = default world gravity; else extra (scale−1)·g).
    float gScale = aero.z;
    if (gScale != 0.0 && gScale != 1.0)
    {
        float m = VD(vr + V_MASS).x;
        g.v[1] -= (gScale - 1.0) * m * p.gravityY;
    }
}

void assembleVehicleN<let N : uint>(uint v, uint vr, uint4 cfg, BodyState c, float3 dl, float3 da,
    bool dual, out Mat6 H, out Vec6 g)
{
    H = mat6Zero();
    g = vec6Zero();
    AvbdParams p = PARAMS;
    float alphaStab = 1.0 - p.alpha;
    float3 upW = quatRotate(c.initialRot, vehicleUpLocal(vr));
    uint nW = min(cfg.z, N);

    // Ground blocks are rebuilt with the chassis block, at the same iterate.
    for (uint wi = 0u; wi < nW; wi++)
        if (wheelGround(wheelRec(cfg.y + wi)) != NO_INDEX) groundClear(cfg.y + wi);

    float Lw[N];
    float FnW[N];
    uint contacts = 0u;
    for (uint wi = 0u; wi < nW; wi++)
    {
        float L, Fn;
        bool ct;
        vehSuspension(wi, vr, cfg, c, dl, da, H, g, L, Fn, ct);
        Lw[wi] = L;
        FnW[wi] = Fn;
        if (ct) contacts++;
    }
    vehAntiRoll<N>(vr, cfg, c, nW, Lw, FnW, H, g);
    for (uint li = 0u; li < nW; li++) vehSpinGroup<N>(li, vr, cfg, c, dl, da, dual, nW, FnW, H, g);
    vehChassisRows<N>(vr, cfg, c, dl, da, dual, nW, FnW, contacts, H, g);
}

// Per-wheel scratch sized to the vehicle: thread-private arrays live in memory on
// GPUs, and the spin solve walks them every Newton step, so a car pays for 4
// wheels, not MAX_VEHICLE_WHEELS (identical arithmetic either way).
void assembleVehicle(uint v, uint vr, uint4 cfg, BodyState c, float3 dl, float3 da,
    bool dual, out Mat6 H, out Vec6 g)
{
    if (cfg.z <= 4u) assembleVehicleN<4>(v, vr, cfg, c, dl, da, dual, H, g);
    else assembleVehicleN<MAX_VEHICLE_WHEELS>(v, vr, cfg, c, dl, da, dual, H, g);
}

void writeVehicleBlock(uint at, Mat6 H, Vec6 g, bool active)
{
    float f[32];
    static const uint LOWER[21] = { 0, 6, 7, 12, 13, 14, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35 };
    for (uint k = 0u; k < 21u; k++) f[k] = H.m[LOWER[k]];
    for (uint k = 0u; k < 6u; k++) f[21u + k] = g.v[k];
    f[27] = active ? 1.0 : 0.0;
    for (uint k = 28u; k < 32u; k++) f[k] = 0.0;
    for (uint e = 0u; e < 4u; e++)
    {
        Aabb a;
        a.mn = float3(f[e * 8u], f[e * 8u + 1u], f[e * 8u + 2u]);
        a._pad0 = f[e * 8u + 3u];
        a.mx = float3(f[e * 8u + 4u], f[e * 8u + 5u], f[e * 8u + 6u]);
        a._pad1 = f[e * 8u + 7u];
        gAabbs[at + e] = a;
    }
}

void chassisDisplacement(BodyState c, out float3 dl, out float3 da)
{
    dl = c.position - c.initialPos;
    float4 q = c.orientation;
    if (dot(q, c.initialRot) < 0.0) q = -q;
    da = quatSubVec(q, c.initialRot);
}

// Wheel shell: SHELL_SPHERES spheres on the chassis around the tyre, at 60°…300°
// from straight down (the tyre rows own the bottom), following suspension travel
// L and steering. A detached or removed wheel parks its shell on the static anchor.
void parkWheelShell(uint wr, uint anchor)
{
    uint first = VU(wr + W_TJ).y;
    if (first == NO_INDEX || gColliders[first].body == anchor) return;
    for (uint k = 0u; k < SHELL_SPHERES; k++)
    {
        Collider col = gColliders[first + k];
        col.body = anchor;
        gColliders[first + k] = col;
    }
}

// Separate from parkWheelShell: inlining both paths into one call with constant
// arguments folded the steer quaternion into a constant that naga's SPIR-V
// front end cannot swizzle.
void placeWheelShell(uint wr, uint chassis, uint anchor, bool attached, float L, float steer)
{
    uint first = VU(wr + W_TJ).y;
    if (first == NO_INDEX) return;
    if (!attached)
    {
        parkWheelShell(wr, anchor);
        return;
    }
    float4 hp = VD(wr + W_HP);
    float4 dir = VD(wr + W_DIR);
    float3 down = normalize(dir.xyz);
    float3 fwd = quatRotate(wheelSteerRot(wr, steer), VD(wr + W_FWD).xyz);
    fwd = normalize(fwd - down * dot(fwd, down));
    float3 center = hp.xyz + down * L;
    float rs = max(min(0.5 * dir.w, 0.4 * hp.w), 1e-3);
    float ring = max(hp.w - rs, 0.0);
    for (uint k = 0u; k < SHELL_SPHERES; k++)
    {
        float th = TWO_PI * float(k + 1u) / float(SHELL_SPHERES + 1u);
        // Whole-record store: a partial float3 store mis-translates through naga.
        Collider col = gColliders[first + k];
        col.body = chassis;
        col.localPos = center + (down * cos(th) + fwd * sin(th)) * ring;
        gColliders[first + k] = col;
    }
}

// Track belts (V_QUERY.zw = left/right pod colliders): the pod's surface runs at the
// sprocket's rim speed (world_step reads it from the collider's density as a
// conveyor). A track whose sprocket is gone parks its pod on the static anchor.
void updateTrackBelts(uint vr, uint4 cfg, uint anchor)
{
    uint4 q = VU(vr + V_QUERY);
    uint4 leaders = VU(vr + V_TRACK);
    for (uint side = 0u; side < 2u; side++)
    {
        uint pod = side == 0u ? q.z : q.w;
        uint leader = side == 0u ? leaders.x : leaders.y;
        if (pod == NO_INDEX || leader == NO_INDEX) continue;
        uint wr = wheelRec(cfg.y + leader);
        bool lost = (VU(wr + W_C0).z & (WD_DETACHED | WD_DISABLED)) != 0u;
        float speed = VD(wr + W_W).x * VD(wr + W_HP).w;
        uint first = pod & ((1u << POD_COUNT_SHIFT) - 1u);
        for (uint k = 0u; k < (pod >> POD_COUNT_SHIFT); k++)
        {
            // Whole-record store: a partial store mis-translates through naga.
            Collider col = gColliders[first + k];
            col.body = lost ? anchor : cfg.x;
            col.density = lost ? 0.0 : speed;
            gColliders[first + k] = col;
        }
    }
}

// ---------------------------------------------------------------------------
// vehicle_link: after wheel_query, before the constraint graph. A wheel standing
// on a dynamic body points its tyre joint (JOINT_FLAG_TYRE) at (chassis, ground) so the two
// get different colours and the ground's primal gathers the wheel's ground block;
// the pair sleeps and wakes as one island (Jolt VehicleConstraint::BuildIslands).
// Otherwise the joint parks on the vehicle's static anchor and adds no graph edge.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void vehicle_link(uint3 tid : SV_DispatchThreadID)
{
    uint w = dispatchIndex(tid);
    if (w >= wheelCountTotal()) return;
    uint wr = wheelRec(w);
    uint4 tj = VU(wr + W_TJ);
    if (tj.x == NO_INDEX) return;
    uint vr = vehRec(VU(wr + W_CFG).x);
    uint chassis = VU(vr + V_CFG).x;
    uint anchor = VU(vr + V_POOL2).z;
    uint4 dyn = VU(wr + W_C0);
    BodyState c = gBodies[chassis];
    uint ground = NO_INDEX;
    if ((dyn.z & WD_CONTACT) != 0u && (dyn.z & (WD_DETACHED | WD_DISABLED)) == 0u
        && dyn.x != NO_INDEX && dyn.x != chassis && bodyIsDynamic(c))
    {
        BodyState g = gBodies[dyn.x];
        // Vehicle and ground form one island: both sleep, or both are awake and
        // linked. A sleeping car on a moving truck (or the reverse) would otherwise
        // stop transmitting its weight.
        bool cs = bodyIsSleeping(c), gs = bodyIsSleeping(g);
        if (bodyIsDynamic(g) && !(cs && gs))
        {
            ground = dyn.x;
            if (cs) wakeBody(chassis);
            if (gs) wakeBody(ground);
        }
    }
    tj.z = ground;
    VSetU(wr + W_TJ, tj);
    // D6 record: JOINT_FLAG_TYRE, ground block index in `motion` (header words
    // only; the vehicle kernels own the rest of the record).
    gJoints[tj.x].bodyA = ground != NO_INDEX ? chassis : anchor;
    gJoints[tj.x].bodyB = ground != NO_INDEX ? ground : anchor;
    gJoints[tj.x].flags = JOINT_FLAG_TYRE;
    gJoints[tj.x].motion = groundBlockIndex(w);
}

// ---------------------------------------------------------------------------
// vehicle_prepare: per-step wheel setup, drivetrain, first block
