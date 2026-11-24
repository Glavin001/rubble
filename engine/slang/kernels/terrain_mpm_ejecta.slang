// ---- ejecta parcels -------------------------------------------------------------

uint mpmOrderedBits(float h)
{
    uint b = asuint(h);
    return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}

// A parcel landed at `xl` (heightfield-local, over the field): its quanta
// split onto the nine samples around it with the B-spline (mpmBsplineSplit,
// seeded by `seed`), as the bake splits moved material, all or none. A share
// on a sample past the field's edge escapes. Returns false, raising
// MPM_OVERFLOW_FIELD and depositing nothing, when a share's sample has no
// deposit slot.
bool mpmParcelDeposit(float3 xl, uint quanta, uint seed, MpmParams p, HeightFieldRef hf)
{
    int2 s0;
    uint parts[9];
    mpmBsplineSplit((xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz), quanta, seed, s0, parts);
    for (uint k = 0u; k < 9u; k++)
    {
        int2 s = s0 + int2(int(k % 3u), int(k / 3u));
        bool on = s.x >= 0 && s.y >= 0 && s.x < int(hf.nx) && s.y < int(hf.nz);
        if (parts[k] != 0u && on && uint(s.y) * hf.nx + uint(s.x) >= asuint(p.couple.w))
        {
            AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_FIELD);
            return false;
        }
    }
    for (uint k = 0u; k < 9u; k++)
    {
        if (parts[k] == 0u) continue;
        int2 s = s0 + int2(int(k % 3u), int(k / 3u));
        if (s.x < 0 || s.y < 0 || s.x >= int(hf.nx) || s.y >= int(hf.nz))
        {
            AADD2(gMpmCounters[MPM_C_ESCAPED], parts[k]);
            continue;
        }
        AADD2(gMpmDeposit[uint(s.y) * hf.nx + uint(s.x)], parts[k]);
        AADD2(gMpmCounters[MPM_C_DEPOSITED], parts[k]);
    }
    return true;
}

// A parcel back as a particle, in the slot of the one it left: no atomic, so
// particle ids (P2G's order) do not depend on the parcels' slots.
void mpmParcelReturn(uint slot, float4 xq, float4 vm, uint quanta, MpmParams p)
{
    MpmParticle q;
    float vol = float(quanta) * p.column.y;
    q.xm = float4(xq.xyz, vm.w);
    q.vj = float4(vm.xyz, 0.0);
    // Rock comes back as rubble: thrown fragments are broken.
    q.c0 = float4(0, 0, 0, p.counts.w == MPM_MATERIAL_ROCK ? 1.0 : 0.0);
    q.c1 = float4(0, 0, 0, 0);
    q.c2 = float4(0, 0, 0, 0);
    q.f0 = float4(1, 0, 0, vol > 0.0 ? vol : vm.w / max(p.column.w, 1e-30));
    q.f1 = float4(0, 1, 0, asfloat(quanta));
    q.f2 = float4(0, 0, 1, 0);
    gMpmParticles[slot] = q;
    AADD2(gMpmCounters[MPM_C_LIVE], 1u);
}

// Per parcel, per substep: ballistic flight (symplectic Euler, as particles), then
// back as a particle on entering an open area's box, else landing on the
// heightfield (bed collider): deposited on the samples around it; off the
// field's footprint, escaped.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_parcels(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (k >= ALOAD(gMpmCounters[MPM_C_FLYING])) return;
    float4 xq = gMpmParcels[2u * k], vm = gMpmParcels[2u * k + 1u];
    if (vm.w <= 0.0) return;
    float dt = p.step.x;
    vm.y += dt * p.step.y;
    xq.xyz += dt * vm.xyz;
    uint quanta = asuint(xq.w);
    bool landed = false;
    // The particle it left (its slot is dead while it flies).
    uint source = asuint(gMpmParcels[2u * asuint(p.couple.z) + k].x);
    if (mpmAreaOf(xq.xyz, p) != MPM_NO_AREA)
    {
        // Into an open area's box (one cell in): back as a particle, which
        // the MPM lands on the material there. (Landing it only at the
        // heightfield, which an open area lowers to its bed, buried it under
        // the material still there.)
        mpmParcelReturn(source, xq, vm, quanta, p);
        landed = true;
    }
    else if (p.dims.w != MPM_NO_BLOCK)
    {
        Collider col = gColliders[p.dims.w];
        Frame f = colliderFrame(col, gBodies[col.body]);
        HeightFieldRef hf = heightFieldOf(col);
        float3 xl = frameToLocal(f, xq.xyz);
        float h; float3 n;
        if (!hfSurface(hf, xl.x, xl.z, h, n))
        {
            AADD2(gMpmCounters[MPM_C_ESCAPED], quanta);
            landed = true;
        }
        else if (xl.y <= h) landed = mpmParcelDeposit(xl, quanta, ~source, p, hf);
    }
    if (!landed)
    {
        AADD2(gMpmCounters[MPM_C_AIRBORNE], 1u);
        mpmParcelReach(xq.xyz, vm.xyz, p);
    }
    gMpmParcels[2u * k] = xq;
    gMpmParcels[2u * k + 1u] = float4(vm.xyz, landed ? 0.0 : vm.w);
}

// Flights a forced close lands at most (substeps of the last frame's dt): far
// past any ballistic flight under the CFL bound; one not down by then stays
// in flight, counted airborne.
static const uint MPM_LAND_STEPS = 1u << 22;

// When the bake closing the last area runs with parcels still in flight (a
// forced close: the World bakes on its own only with nothing airborne): each
// flies on as mpm_parcels would fly it, substep by substep at the last
// frame's dt, until it lands, all in this one pass. No area is left to take it
// back, so it lands on the heightfield (mpmParcelDeposit, the same seed), or
// escapes off its footprint; one rising with no gravity to bring it back never
// lands and escapes. Its quanta are on the field when the pool empties, so a
// domain opened afresh starts with nothing of the old one in flight.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_parcels_land(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (k >= ALOAD(gMpmCounters[MPM_C_FLYING])) return;
    float4 xq = gMpmParcels[2u * k], vm = gMpmParcels[2u * k + 1u];
    if (vm.w <= 0.0) return;
    if (p.dims.w == MPM_NO_BLOCK) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    uint quanta = asuint(xq.w);
    uint source = asuint(gMpmParcels[2u * asuint(p.couple.z) + k].x);
    float dt = p.step.x, g = p.step.y;
    bool landed = false;
    if (!(dt > 0.0) || (!(g < 0.0) && vm.y >= 0.0))
    {
        AADD2(gMpmCounters[MPM_C_ESCAPED], quanta);
        landed = true;
    }
    for (uint n = 0u; n < MPM_LAND_STEPS && !landed; n++)
    {
        vm.y += dt * g;
        xq.xyz += dt * vm.xyz;
        float3 xl = frameToLocal(f, xq.xyz);
        float h; float3 nrm;
        if (!hfSurface(hf, xl.x, xl.z, h, nrm))
        {
            AADD2(gMpmCounters[MPM_C_ESCAPED], quanta);
            landed = true;
        }
        else if (xl.y <= h)
        {
            landed = mpmParcelDeposit(xl, quanta, ~source, p, hf);
            // No deposit slot (MPM_OVERFLOW_FIELD raised): it stays in flight.
            if (!landed) break;
        }
    }
    if (!landed) AADD2(gMpmCounters[MPM_C_AIRBORNE], 1u);
    gMpmParcels[2u * k] = xq;
    gMpmParcels[2u * k + 1u] = float4(vm.xyz, landed ? 0.0 : vm.w);
}

// Once per frame, per heightfield record: the quanta deposited on its samples,
// as height, and the field's max height raised over them.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_deposit_raise(uint3 tid : SV_DispatchThreadID)
{
    uint r = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (p.dims.w == MPM_NO_BLOCK) return;
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    if (r >= (min(hf.nx * hf.nz, asuint(p.couple.w)) + 3u) / 4u) return;
    uint4 rec = gHulls[hf.base + 2u + r];
    float area = mpmSampleArea(hf);
    bool changed = false;
    for (uint c = 0u; c < 4u; c++)
    {
        uint k = 4u * r + c;
        if (k >= min(hf.nx * hf.nz, asuint(p.couple.w))) continue;
        uint n;
        AXCHG3(gMpmDeposit[k], 0u, n);
        if (n == 0u) continue;
        float h = asfloat(rec[c]) + float(n) * p.column.y / area;
        rec[c] = asuint(h);
        AMAX2(gMpmCounters[MPM_C_BAKE_MAX], mpmOrderedBits(h));
        changed = true;
    }
    if (changed) gHulls[hf.base + 2u + r] = rec;
}

// After mpm_deposit_raise: the field's max height covers the deposits.
[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_deposit_header(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmParams();
    if (p.dims.w == MPM_NO_BLOCK) return;
    HeightFieldRef hf = heightFieldOf(gColliders[p.dims.w]);
    uint b;
    AXCHG3(gMpmCounters[MPM_C_BAKE_MAX], 0u, b);
    if (b == 0u) return;
    float top = asfloat((b & 0x80000000u) != 0u ? (b & 0x7FFFFFFFu) : ~b);
    uint4 dims = gHulls[hf.base + 1u];
    dims.w = asuint(max(asfloat(dims.w), top));
    gHulls[hf.base + 1u] = dims;
}

// ---- km/s sources (T4) -------------------------------------------------------------

// Maxwell's Z-model excavation velocity at offset d from the source's surface
// point (+y up, the target below): radial from the flow centre a depth below, with
// the tangential part that makes the flow divergence-free, speed α·r^−Z; inside
// the core a uniform radial expansion. Mirrors engine-terrain ZModel::velocity.
float3 mpmZModelVelocity(float3 d0, float z, float depth, float core, float alpha)
{
    float3 d = d0 + float3(0.0, depth, 0.0);
    float r = length(d);
    if (r < core) return d * (alpha * pow(core, -z) / core);
    float c = -d.y / r;
    float f = (z - 2.0) / (1.0 + c);
    float h = d.x * d.x + d.z * d.z;
    float3 tangential = float3(c * d.x / r, h / (r * r), c * d.z / r) * f;
    return (d / r + tangential) * (alpha * pow(r, -z));
}

// Before a source on material that has moved: the top of the material under
// the source point, the highest of the selected area's particles within a
// lattice spacing of its vertical line (plus half a spacing, the particle's
// own extent), or the heightfield's surface there if higher (bare bed), as
// order-preserving bits in MPM_C_SURFACE (which the host zeroes first).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_source_surface(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i == 0u && p.dims.w != MPM_NO_BLOCK)
    {
        Collider col = gColliders[p.dims.w];
        Frame f = colliderFrame(col, gBodies[col.body]);
        HeightFieldRef hf = heightFieldOf(col);
        float3 xl = frameToLocal(f, p.source0.xyz);
        float h;
        float3 n;
        if (hfSurface(hf, xl.x, xl.z, h, n))
            AMAX2(gMpmCounters[MPM_C_SURFACE], mpmOrderedBits(frameToWorld(f, float3(xl.x, h, xl.z)).y));
    }
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    float s = p.column.z;
    float2 d = q.xm.xz - p.source0.xz;
    if (abs(d.x) > s || abs(d.y) > s) return;
    if (mpmAreaOf(q.xm.xyz, p) != p.area.y) return;
    AMAX2(gMpmCounters[MPM_C_SURFACE], mpmOrderedBits(q.xm.y + 0.5 * s));
}

float mpmFromOrderedBits(uint b) { return asfloat((b & 0x80000000u) != 0u ? (b & 0x7FFFFFFFu) : ~b); }

// Per live particle of the selected area: the source's excavation velocity
// added to its own (the shock and contact stage replaced by its outcome; the
// MPM computes the rest). Several sources superpose: each adds its field to
// the material's velocity where it stands.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_source(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = mpmParams();
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    float3 s0 = p.source0.xyz;
    if ((p.area.w & MPM_DOMAIN_SURFACE) != 0u)
    {
        uint top = ALOAD(gMpmCounters[MPM_C_SURFACE]);
        if (top != 0u) s0.y = mpmFromOrderedBits(top);
    }
    float3 d = q.xm.xyz - s0;
    if (d.y > 0.0) return; // the field is the target's (y ≤ 0)
    if (length(d) > p.source1.w) return; // beyond the transient crater: at rest
    if (mpmAreaOf(q.xm.xyz, p) != p.area.y) return;
    q.vj.xyz += mpmZModelVelocity(d, p.source0.w, p.source1.x, p.source1.y, p.source1.z);
    // A source wakes what it moves (block sleep).
    q.f2.w = asfloat(asuint(q.f2.w) & ~(MPM_PARTICLE_ASLEEP | MPM_PARTICLE_QUIET));
    gMpmParticles[i] = q;
}

// ---- wheel loads ----------------------------------------------------------------


// Whether wheel record wr stood, in the rigid step before, on an open area's
// live surface: in contact, enabled and attached, on a static heightfield
// other than the area's bed (the area's surface patch), the contact one cell
// inside the area's box. Then the force it pressed on the ground there (the
// reaction of the ground's force on the chassis), the contact and the area.
bool mpmWheelOnSand(uint wr, MpmParams p, out float3 force, out float3 x, out uint area)
{
    force = float3(0, 0, 0);
    x = float3(0, 0, 0);
    area = MPM_NO_AREA;
    uint4 cfg = asuint(gVehicleData[wr + W_CFG]);
    uint4 dyn = asuint(gVehicleData[wr + W_C0]);
    if ((cfg.y & WF_ENABLED) == 0u) return false;
    if ((dyn.z & (WD_DETACHED | WD_DISABLED | WD_CONTACT)) != WD_CONTACT) return false;
    uint vr = asuint(gVehicleData[VH_BASES]).x + cfg.x * VEH_STRIDE;
    if ((asuint(gVehicleData[vr + V_CFG]).w & VF_ENABLED) == 0u) return false;
    uint ci = dyn.y;
    if (ci >= gCounters[0].colliderCount) return false;
    Collider col = gColliders[ci];
    if (col.shape != SHAPE_HEIGHTFIELD || bodyIsDynamic(gBodies[col.body])) return false;
    x = gVehicleData[wr + W_HIT].xyz;
    area = mpmAreaOf(x, p);
    if (area == MPM_NO_AREA) return false;
    if (ci == gMpmParams[1u + area].dims.w) return false;
    float4 f = gVehicleData[wr + W_F];
    float3 nrm = gVehicleData[wr + W_C2].xyz;
    float3 t = gVehicleData[wr + W_C3].xyz;
    float3 s = gVehicleData[wr + W_C4].xyz;
    force = -(nrm * f.x - t * f.y - s * f.z);
    // Finite by its exponent bits (not 255): isfinite's test, which the SPIR-V path
    // lowers to IsInf/IsNan that naga's SPIR-V reader rejects.
    return all((asuint(force) & 0x7F800000u) != 0x7F800000u) && dot(force, force) > 0.0;
}

// Once per coupled frame, one thread: the wheels on open sand, in wheel order
// (so runs agree bit for bit), as wheel-load records; past max wheel loads the
// frame fails by name (MPM_OVERFLOW_WHEELS).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_wheel_loads(uint3 tid : SV_DispatchThreadID)
{
    if (mpmIndex(tid) != 0u) return;
    MpmParams p = mpmParams();
    uint base = asuint(p.limits.z);
    uint maxLoads = asuint(p.limits.w);
    uint wheels = asuint(gVehicleData[VH_COUNTS]).w;
    uint wb = asuint(gVehicleData[VH_BASES]).y;
    uint n = 0u;
    for (uint w = 0u; w < wheels; w++)
    {
        float3 force, x;
        uint area;
        if (!mpmWheelOnSand(wb + w * WHEEL_STRIDE, p, force, x, area)) continue;
        if (n >= maxLoads)
        {
            AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_WHEELS);
            break;
        }
        // The contact's base node and offset in the area's frame, as a
        // particle there would have them (mpm_g2p).
        float4 o = gMpmParams[1u + area].origin;
        int3 local = int3(floor((x - o.xyz) / o.w - 0.5));
        float3 fx = (x - o.xyz) / o.w - float3(local);
        int3 b = local + mpmSlotOffset(area);
        uint r = base + MPM_WHEEL_RECORDS * n;
        gMpmCoupled[r] = float4(fx, asfloat(area));
        gMpmCoupled[r + 1u] = float4(asfloat(b.x), asfloat(b.y), asfloat(b.z), 0.0);
        gMpmCoupled[r + 2u] = float4(force, 0.0);
        n++;
    }
    ASTORE(gMpmCounters[MPM_C_WHEELS], n);
}

// Per wheel load, each substep after P2G: the material's mass at the contact,
// M = Σ w·m over the stencil's nodes that carry velocity (mpm_grid's).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_wheel_mass(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= ALOAD(gMpmCounters[MPM_C_WHEELS])) return;
    MpmParams p = mpmParams();
    uint r = asuint(p.limits.z) + MPM_WHEEL_RECORDS * i;
    float3 fx = gMpmCoupled[r].xyz;
    int3 base = asint(gMpmCoupled[r + 1u].xyz);
    float3 wx = mpmWeights(fx.x), wy = mpmWeights(fx.y), wz = mpmWeights(fx.z);
    int3 k0 = base >> 2;
    uint blk[8];
    mpmBlockCube(k0, p, blk);
    float mass = 0.0;
    for (uint o = 0u; o < 27u; o++)
    {
        uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
        uint c = mpmCellInCube(base + int3(off), k0, blk);
        if (c == MPM_NO_BLOCK) continue;
        float m = gMpmGrid[2u * c].w;
        if (m <= p.step.z) continue;
        mass += mpmWeightAt(wx, off.x) * mpmWeightAt(wy, off.y) * mpmWeightAt(wz, off.z) * m;
    }
    gMpmCoupled[r + 2u].w = mass;
}
