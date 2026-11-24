// ---------------------------------------------------------------------------
// After vehicle_post, on hosts with a soil heightfield (the plan's `soil` input):
//   wheel_soil        per wheel on soil: sinkage of its load, the patch it
//                     presses and the compaction resistance the next step's
//                     vehicle_prepare applies (soilRollingTorque)
//   wheel_soil_press  SOIL_LANES threads; lane L owns the heightfield records
//                     r ≡ L (mod SOIL_LANES) and presses them under every
//                     wheel's patch in wheel order. No two threads share a
//                     record, so the result does not depend on scheduling.
// Samples are lowered plastically to (uncompacted height − imprint depth) and
// never raised; the uncompacted plane only rises (to a sample raised by other
// edits). The header's min height already sits max_sinkage below the field
// (HeightField::bounds_range), so it needs no update here.
static const uint OVERFLOW_SOIL_SINKAGE = 65536u;

// Records in one plane of heightfield `hf`'s samples.
uint soilPlaneRecords(HeightFieldRef hf) { return (hf.nx * hf.nz + 3u) / 4u; }

// A plane of samples (first record `pb`) at local (x, z), by the field's own
// triangles (hfSurface). False off the field.
bool soilPlaneAt(HeightFieldRef hf, uint pb, float x, float z, out float h)
{
    h = 0.0;
    float fx = (x - hf.x0) / hf.dx;
    float fz = (z - hf.z0) / hf.dz;
    if (fx < 0.0 || fz < 0.0 || fx > float(hf.nx - 1u) || fz > float(hf.nz - 1u)) return false;
    uint i = min(uint(fx), hf.nx - 2u);
    uint j = min(uint(fz), hf.nz - 2u);
    float u = fx - float(i);
    float v = fz - float(j);
    uint k = j * hf.nx + i;
    float h00 = asfloat(gHulls[pb + (k >> 2)][k & 3u]);
    k += 1u;
    float h10 = asfloat(gHulls[pb + (k >> 2)][k & 3u]);
    k += hf.nx;
    float h11 = asfloat(gHulls[pb + (k >> 2)][k & 3u]);
    k -= 1u;
    float h01 = asfloat(gHulls[pb + (k >> 2)][k & 3u]);
    if (u >= v) h = h00 + u * (h10 - h00) + v * (h11 - h10);
    else h = h00 + u * (h11 - h01) + v * (h01 - h00);
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wheel_soil(uint3 tid : SV_DispatchThreadID)
{
    uint w = dispatchIndex(tid);
    if (w >= wheelCountTotal()) return;
    uint wr = wheelRec(w);
    uint vr = vehRec(VU(wr + W_CFG).x);
    BodyState c = gBodies[VU(vr + V_CFG).x];
    // A sleeping vehicle keeps its last resistance and presses nothing new.
    if (bodyIsDynamic(c) && bodyIsSleeping(c))
    {
        VSet(wr + W_SOIL1, float4(VD(wr + W_SOIL1).xyz, 0.0));
        return;
    }
    VSet(wr + W_SOIL1, float4(0, 0, 0, 0));
    VSet(wr + W_SOIL2, float4(0, 0, 0, 0));
    uint4 dyn = VU(wr + W_C0);
    bool live = (VU(vr + V_CFG).w & VF_ENABLED) != 0u && (VU(wr + W_CFG).y & WF_ENABLED) != 0u
        && (dyn.z & (WD_DETACHED | WD_DISABLED | WD_CONTACT)) == WD_CONTACT && bodyIsDynamic(c);
    if (!live) return;
    Collider col = gColliders[dyn.y];
    if (col.shape != SHAPE_HEIGHTFIELD) return;
    HeightFieldRef hf = heightFieldOf(col);
    if ((gHulls[hf.base].w & HF_SOIL) == 0u) return;
    uint plane = soilPlaneRecords(hf);
    float4 sp = asfloat(gHulls[hf.base + 2u + plane]);
    float maxZ = asfloat(gHulls[hf.base + 3u + plane].x);
    float b = VD(wr + W_DIR).w;
    float R = VD(wr + W_HP).w;
    float W = max(VD(wr + W_F).x, 0.0);
    if (!(b > 0.0) || !(R > 0.0))
    {
        // No contact area: the pressure–sinkage relation has no answer.
        AOR2(gCounters[0].overflowFlags, OVERFLOW_SOIL_SINKAGE);
        return;
    }
    // The contact at x⁻ (this step's query) and where the next query will find
    // it: the hit carried with the chassis to its solved pose.
    Frame f = colliderFrame(col, gBodies[col.body]);
    float4 hit = VD(wr + W_HIT);
    bool sphere = (VU(vr + V_CFG).w & VF_SPHERE_CAST) != 0u;
    float3 o1 = transformPoint(c.position, c.orientation, VD(wr + W_HP).xyz);
    float3 d1 = normalize(quatRotate(c.orientation, VD(wr + W_DIR).xyz));
    float3 next = o1 + d1 * hit.w - (sphere ? VD(wr + W_C2).xyz * R : float3(0, 0, 0));
    float2 P = frameToLocal(f, hit.xyz).xz;

    // Sinkage: the soil in series with the suspension (soilSeriesSinkage), from
    // the depth the wheel stands at, under the step's tyre load; at rest, the
    // static rigid-wheel sinkage of that load.
    float k = soilModulus(sp.x, sp.y, b);
    float hAt, hFreshAt;
    float zNow = 0.0;
    if (soilPlaneAt(hf, hf.base + 2u, P.x, P.y, hAt) && soilPlaneAt(hf, hf.base + 4u + plane, P.x, P.y, hFreshAt))
        zNow = max(hFreshAt - hAt, 0.0);
    float4 kd = VD(wr + W_K);
    float z = soilSeriesSinkage(W, zNow, max(kd.x + kd.y, 0.0), b, 2.0 * R, k, sp.z, sp.w);
    if (!(z <= maxZ))
    {
        // Deeper than the soil's limit (the field's bounds): the step fails.
        AOR2(gCounters[0].overflowFlags, OVERFLOW_SOIL_SINKAGE);
        z = maxZ;
    }
    float2 delta = frameToLocal(f, next).xz - P;
    float2 roll = quatRotate(quatConjugate(f.q), VD(wr + W_C3).xyz).xz;
    float moved = dot(delta, delta);
    if (dot(roll, roll) > 1e-8) roll = normalize(roll);
    else roll = moved > 1e-12 ? normalize(delta) : float2(0.0, 1.0);
    // Samples within half a cell beyond the tyre's edges, so a tyre narrower
    // than two cells still lowers the samples either side of its centre line.
    float cell = max(hf.dx, hf.dz);
    float hw = 0.5 * b + 0.5 * cell;
    // The floor (full depth) runs from this contact to the next one and
    // 1.5 cells past both, so every triangle a contact can stand on is level
    // at the sinkage: the imprint's circular ends, sampled one cell apart,
    // would otherwise tilt the contact normal back against the motion and
    // resist it a second time (the soil's resistance is R_c).
    float2 fwd = moved > 1e-12 ? normalize(delta) : roll;
    float level = 1.5 * cell;
    float2 from = P - fwd * level;
    float2 to = P + delta + fwd * level;

    // Compaction already in the soil this step newly compacts (an earlier pass,
    // or a wheel ahead on the same track): sampled just past the front of the
    // patch, clear of this wheel's own imprint.
    float ahead = sqrt(max(2.0 * R * z - z * z, 0.0)) + 1.5 * cell;
    float2 F = to + fwd * ahead;
    float hNow, hFresh;
    float z0 = 0.0;
    if (soilPlaneAt(hf, hf.base + 2u, F.x, F.y, hNow) && soilPlaneAt(hf, hf.base + 4u + plane, F.x, F.y, hFresh))
        z0 = max(hFresh - hNow, 0.0);
    float Rc = soilCompaction(b, k, sp.z, z, z0);

    VSet(wr + W_SOIL0, float4(from, to - from));
    VSet(wr + W_SOIL1, float4(roll, hw, z));
    VSetU(wr + W_SOIL2, uint4(dyn.y + 1u, asuint(Rc), asuint(W), asuint(z)));
    VSet(wr + W_SOIL3, float4(z0, 0, 0, 0));
    // The ground under the wheel moves: probe it next step rather than slide
    // along the stored hit plane (query rate, V_QUERY).
    VSet(wr + W_LHIT, float4(VD(wr + W_LHIT).xyz, 0.0));
}

// Press one record (four samples) of `hf` under one wheel's patch.
void soilPressRecord(HeightFieldRef hf, uint r, uint vb, float2 P, float2 D, float2 t, float hw, float z, float R)
{
    uint n = hf.nx * hf.nz;
    uint4 hr = gHulls[hf.base + 2u + r];
    uint4 ur = gHulls[vb + r];
    bool dh = false;
    bool du = false;
    float2 s = float2(-t.y, t.x);
    float dLat = dot(D, s);
    float dLon = dot(D, t);
    for (uint c = 0u; c < 4u; c++)
    {
        uint k = 4u * r + c;
        if (k >= n) break;
        float2 x = float2(hf.x0 + float(k % hf.nx) * hf.dx, hf.z0 + float(k / hf.nx) * hf.dz) - P;
        float d = soilImprintDepth(dot(x, s), dot(x, t), dLat, dLon, hw, z, R);
        if (d <= 0.0) continue;
        float h = asfloat(hr[c]);
        float fresh = asfloat(ur[c]);
        // A sample other edits raised above the soil's surface is fresh soil.
        if (h > fresh)
        {
            fresh = h;
            ur[c] = asuint(fresh);
            du = true;
        }
        float target = fresh - d;
        if (target < h)
        {
            hr[c] = asuint(target);
            dh = true;
        }
    }
    if (dh) gHulls[hf.base + 2u + r] = hr;
    if (du) gHulls[vb + r] = ur;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wheel_soil_press(uint3 tid : SV_DispatchThreadID)
{
    uint lane = dispatchIndex(tid);
    if (lane >= SOIL_LANES) return;
    uint nWheels = wheelCountTotal();
    for (uint w = 0u; w < nWheels; w++)
    {
        uint wr = wheelRec(w);
        uint tag = VU(wr + W_SOIL2).x;
        float4 s1 = VD(wr + W_SOIL1);
        if (tag == 0u || !(s1.w > 0.0)) continue;
        HeightFieldRef hf = heightFieldOf(gColliders[tag - 1u]);
        float4 s0 = VD(wr + W_SOIL0);
        float R = VD(wr + W_HP).w;
        float z = s1.w;
        float2 P = s0.xy;
        float2 D = s0.zw;
        float2 t = s1.xy;
        // Bounds of the swept imprint (its rectangle at both ends of the segment).
        float l = sqrt(max(2.0 * R * z - z * z, 0.0));
        float2 ext = abs(t) * l + abs(float2(-t.y, t.x)) * s1.z;
        float2 lo = min(P, P + D) - ext;
        float2 hi = max(P, P + D) + ext;
        float lastX = float(hf.nx - 1u);
        float lastZ = float(hf.nz - 1u);
        float fx0 = (lo.x - hf.x0) / hf.dx;
        float fx1 = (hi.x - hf.x0) / hf.dx;
        float fz0 = (lo.y - hf.z0) / hf.dz;
        float fz1 = (hi.y - hf.z0) / hf.dz;
        if (fx1 < 0.0 || fz1 < 0.0 || fx0 > lastX || fz0 > lastZ) continue;
        uint i0 = uint(clamp(floor(fx0), 0.0, lastX));
        uint i1 = uint(clamp(ceil(fx1), 0.0, lastX));
        uint j0 = uint(clamp(floor(fz0), 0.0, lastZ));
        uint j1 = uint(clamp(ceil(fz1), 0.0, lastZ));
        uint vb = hf.base + 4u + soilPlaneRecords(hf);
        for (uint j = j0; j <= j1; j++)
        {
            uint r0 = (j * hf.nx + i0) >> 2;
            uint r1 = (j * hf.nx + i1) >> 2;
            // This lane's first record in the row's range.
            uint r = r0 + (lane + SOIL_LANES - (r0 % SOIL_LANES)) % SOIL_LANES;
            for (; r <= r1; r += SOIL_LANES) soilPressRecord(hf, r, vb, P, D, t, s1.z, z, R);
        }
    }
}
