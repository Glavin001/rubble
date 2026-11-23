// mpm_g2p's body for thread tid, lane j of its workgroup (every lane of the
// workgroup calls it: it synchronises the workgroup).
void mpmG2pGroup(uint3 tid, uint j)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint speedBits = 0u, speedArea = MPM_NO_AREA;
    if (i < mpmParticleCount())
    {
        // Block sleep: its own function, ahead of the usual path (it takes
        // its maxima itself: this lane then adds none below).
        if (mpmSleepSpeed() > 0.0) mpmG2pSleep(i, p);
        else mpmG2p(i, p, speedBits, speedArea);
    }
    // The fastest speed, once per workgroup instead of once per particle (a
    // max of the same values: the counters end the same). f32 bits of speeds
    // order as the speeds; 0 is no speed. Every barrier is reached by every
    // lane (WGSL's uniformity).
    bool moved = speedArea != MPM_NO_AREA;
    gsMpmG2pSpeed[j] = speedBits;
    gsMpmG2pArea[j] = speedArea;
    gsMpmG2pTop[j] = speedBits;
    gsMpmG2pLo[j] = moved ? speedArea : MPM_NO_AREA;
    gsMpmG2pHi[j] = moved ? speedArea : 0u;
    GROUP_SYNC();
    for (uint w = 32u; w > 0u; w >>= 1)
    {
        if (j < w)
        {
            gsMpmG2pTop[j] = max(gsMpmG2pTop[j], gsMpmG2pTop[j + w]);
            gsMpmG2pLo[j] = min(gsMpmG2pLo[j], gsMpmG2pLo[j + w]);
            gsMpmG2pHi[j] = max(gsMpmG2pHi[j], gsMpmG2pHi[j + w]);
        }
        GROUP_SYNC();
    }
    uint top = gsMpmG2pTop[0];
    uint lo = gsMpmG2pLo[0], hi = gsMpmG2pHi[0];
    if (j == 0u && top != 0u) AMAX2(gMpmCounters[MPM_C_MAX_SPEED], top);
    // Per area only with more than slot 0 in use (one area's is the frame's:
    // TerrainPlan::check). One area in the workgroup: its maximum is the
    // workgroup's. Several: the first lane of each takes that area's.
    if (p.area.x <= 1u || lo > hi) return;
    if (lo == hi)
    {
        if (j == 0u) AMAX2(gMpmCounters[MPM_C_AREA_SPEED + lo], top);
        return;
    }
    if (!moved) return;
    uint mine = 0u;
    bool first = true;
    for (uint k = 0u; k < 64u; k++)
    {
        if (gsMpmG2pArea[k] != speedArea) continue;
        if (k < j) first = false;
        mine = max(mine, gsMpmG2pSpeed[k]);
    }
    if (first) AMAX2(gMpmCounters[MPM_C_AREA_SPEED + speedArea], mine);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_g2p(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    mpmG2pGroup(tid, lid.x);
}

// ---- activation and bake -----------------------------------------------------

// Area each heightfield sample stands for (its cell around it) and the window
// sample t's grid indices.
float mpmSampleArea(HeightFieldRef hf) { return hf.dx * hf.dz; }

uint2 mpmWindowSample(uint t, MpmParams p)
{
    return uint2(p.window.x + t % p.window.z, p.window.y + t / p.window.z);
}

// Particles a sample's column becomes: a lattice of spacing ≈ column.z over its
// cell and its height above the bed, never more particles than quanta.
uint3 mpmColumnLattice(HeightFieldRef hf, float dh, uint quanta, MpmParams p)
{
    if (quanta == 0u) return uint3(0, 0, 0);
    float s = p.column.z;
    uint3 n = uint3(max(1.0, round(hf.dx / s)), max(1.0, round(dh / s)), max(1.0, round(hf.dz / s)));
    while (n.x * n.y * n.z > quanta)
    {
        if (n.y > 1u) n.y--;
        else if (n.x >= n.z && n.x > 1u) n.x--;
        else n.z--;
    }
    return n;
}

// The area being taken over by a growing one (gMpmParams[0].area.z) whose
// window holds sample ij, or MPM_NO_AREA.
uint mpmMergedAt(uint2 ij)
{
    uint mask = gMpmParams[0].area.z;
    for (uint k = 0u; k < MPM_MAX_AREAS; k++)
    {
        if ((mask & (1u << k)) == 0u) continue;
        uint4 w = gMpmParams[1u + k].window;
        if (ij.x >= w.x && ij.x < w.x + w.z && ij.y >= w.y && ij.y < w.y + w.w) return k;
    }
    return MPM_NO_AREA;
}

// Per window sample: the quanta above the bed (rounded to whole quanta), to
// emit now ([3]), all it has given ([0]), and its particle count. A sample a
// growing area takes over is active already: it keeps the quanta it gave, and
// gives only the slab between its old bed and a deeper new one.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_activate_count(uint3 tid : SV_DispatchThreadID)
{
    uint t = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    if (t >= p.window.z * p.window.w) return;
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    uint2 ij = mpmWindowSample(t, p);
    float dh = hfHeight(hf, ij.x, ij.y) - p.column.x;
    float area = mpmSampleArea(hf);
    uint quanta = dh > 0.0 ? uint(floor(dh * area / p.column.y + 0.5)) : 0u;
    uint total = quanta;
    uint old = mpmMergedAt(ij);
    if (old != MPM_NO_AREA)
    {
        MpmParams o = mpmArea(old);
        uint to = (ij.y - o.window.y) * o.window.z + (ij.x - o.window.x);
        if (!(p.column.x < o.column.x)) quanta = 0u;
        total = ALOAD(gMpmColumns[mpmCol(o, to)]) + quanta;
    }
    uint3 n = mpmColumnLattice(hf, float(quanta) * p.column.y / area, quanta, p);
    ASTORE(gMpmColumns[mpmCol(p, t)], total);
    ASTORE(gMpmColumns[mpmCol(p, t) + 1u], n.x * n.y * n.z);
    ASTORE(gMpmColumns[mpmCol(p, t) + 3u], quanta);
}

// One workgroup: each sample's first particle (after the particles already in
// the region), in window order; all or nothing against max particles.
[shader("compute")]
[numthreads(256, 1, 1)]
void mpm_activate_scan(uint3 lid : SV_GroupThreadID)
{
    uint l = lid.x;
    MpmParams p = mpmSelArea();
    uint samples = p.window.z * p.window.w;
    uint chunk = (samples + 255u) / 256u;
    uint first = l * chunk, last = min(first + chunk, samples);
    uint sum = 0u;
    for (uint t = first; t < last; t++) sum += ALOAD(gMpmColumns[mpmCol(p, t) + 1u]);
    gsMpmScan[l] = sum;
    GROUP_SYNC();
    for (uint d = 1u; d < 256u; d <<= 1u)
    {
        uint add = l >= d ? gsMpmScan[l - d] : 0u;
        GROUP_SYNC();
        gsMpmScan[l] += add;
        GROUP_SYNC();
    }
    uint have = mpmParticleCount();
    uint start = have + gsMpmScan[l] - sum;
    for (uint t = first; t < last; t++)
    {
        ASTORE(gMpmColumns[mpmCol(p, t) + 2u], start);
        start += ALOAD(gMpmColumns[mpmCol(p, t) + 1u]);
    }
    GROUP_SYNC();
    if (l == 255u)
    {
        uint total = gsMpmScan[255];
        // Past the pool's end: the slots baked areas and landed parcels left
        // take them (mpm_pool_*), if enough are free (mpm_pool_finish decides).
        bool reuse = have + total > p.counts.x;
        if (reuse && total > p.counts.x)
            AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_PARTICLES);
        else
        {
            if (reuse) ASTORE(gMpmCounters[MPM_C_REUSE], total);
            else
            {
                ASTORE(gMpmCounters[MPM_C_PARTICLES], have + total);
                AADD2(gMpmCounters[MPM_C_LIVE], total);
            }
            // The area's first slot (a grown one keeps its oldest area's): the
            // bake's dithers count slots from it, so an area bakes as it
            // would alone however many particles other areas hold.
            uint first = have;
            uint mask = gMpmParams[0].area.z;
            for (uint k = 0u; k < MPM_MAX_AREAS; k++)
                if ((mask & (1u << k)) != 0u) first = min(first, ALOAD(gMpmCounters[MPM_C_AREA_FIRST + k]));
            ASTORE(gMpmCounters[MPM_C_AREA_FIRST + gMpmParams[0].area.y], first);
        }
    }
}

bool mpmActivationRefused() { return (ALOAD(gMpmCounters[MPM_C_FLAGS]) & MPM_OVERFLOW_PARTICLES) != 0u; }

// Per particle of the window (one thread each: a column's particles were a
// serial loop of one thread per sample before, the strike frame's longest
// activation kernel): the sample whose range holds it, by binary search over
// the samples' first particles, and its place in that sample's lattice. The
// first `quanta mod n` of a sample take one more quantum. Writes exactly what
// the per-sample loop wrote.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_activate_emit(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    uint samples = p.window.z * p.window.w;
    if (samples == 0u || mpmActivationRefused()) return;
    // Particles already in the pool before this activation are not its.
    uint have = ALOAD(gMpmColumns[mpmCol(p, 0u) + 2u]);
    uint reuse = ALOAD(gMpmCounters[MPM_C_REUSE]);
    if (i >= (reuse != 0u ? have + reuse : mpmParticleCount())) return;
    if (i < have) return;
    // The last sample whose first particle is at or before i (samples with
    // no particles share the next one's first, so this one holds i).
    uint lo = 0u, hi = samples - 1u;
    while (lo < hi)
    {
        uint mid = (lo + hi + 1u) / 2u;
        if (ALOAD(gMpmColumns[mpmCol(p, mid) + 2u]) <= i) lo = mid;
        else hi = mid - 1u;
    }
    uint t = lo;
    uint total = ALOAD(gMpmColumns[mpmCol(p, t) + 1u]);
    uint first = ALOAD(gMpmColumns[mpmCol(p, t) + 2u]);
    if (i >= first + total) return;
    uint k = i - first;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    uint2 ij = mpmWindowSample(t, p);
    // The quanta it gives now (a grown area's old sample: only a deeper slab).
    uint quanta = ALOAD(gMpmColumns[mpmCol(p, t) + 3u]);
    float area = mpmSampleArea(hf);
    float dh = float(quanta) * p.column.y / area;
    uint3 n = mpmColumnLattice(hf, dh, quanta, p);
    float3 lo3 = float3(hf.x0 + (float(ij.x) - 0.5) * hf.dx, p.column.x, hf.z0 + (float(ij.y) - 0.5) * hf.dz);
    float3 cell = float3(hf.dx / float(n.x), dh / float(n.y), hf.dz / float(n.z));
    uint each = quanta / total, extra = quanta % total;
    uint3 c = uint3(k % n.x, (k / n.x) % n.y, k / (n.x * n.y));
    float3 xl = lo3 + (float3(c) + 0.5) * cell;
    uint mine = each + (k < extra ? 1u : 0u);
    float vol = float(mine) * p.column.y;
    float3 xw = frameToWorld(f, xl);
    // Where the bake will see it if it never moves (the same transform
    // mpm_bake_bin applies), so its displacement is then exactly zero.
    float3 home = frameToLocal(f, xw);
    // Growing over an old area's wall: the ground there gave (mpmGrowGive).
    if (gMpmParams[0].area.z != 0u) xw = frameToWorld(f, xl + mpmGrowGive(xl, ij, p, hf));
    MpmParticle q;
    q.xm = float4(xw, vol * p.column.w);
    q.vj = float4(0, 0, 0, 0);
    q.c0 = float4(0, 0, 0, 0);
    q.c1 = float4(0, 0, 0, home.x);
    q.c2 = float4(0, 0, 0, home.z);
    q.f0 = float4(1, 0, 0, vol);
    q.f1 = float4(0, 1, 0, asfloat(mine));
    q.f2 = float4(0, 0, 1, asfloat(MPM_PARTICLE_HOME | MPM_PARTICLE_FRESH));
    // Reusing slots: the first ones take the free slots in order, the rest
    // go past the pool's end (i counts the activation's particles from it).
    uint slot = i;
    if (reuse != 0u)
    {
        uint v = i - have, used = ALOAD(gMpmCounters[MPM_C_HOLES]);
        slot = v < used ? gMpmSorted[v] : have + (v - used);
    }
    gMpmParticles[slot] = q;
}

// ---- pool reuse ---------------------------------------------------------------
// engine/docs/deformable-terrain.md, "Endless areas". A baked area's particles
// and a landed parcel's source stay in the pool as dead slots while other
// areas are open. Two passes take them back, neither moving a live particle:
// a bake that leaves areas open trims the pool's dead tail (mpm_pool_top,
// mpm_pool_trim) and packs the parcels still in flight (mpm_parcels_pack,
// mpm_parcels_unpack); an activation past the pool's end puts its particles
// in the free slots, in slot order (mpm_pool_holes_*, mpm_pool_finish). Both
// run only while MPM_C_REUSE is set. A dead slot a parcel in flight will come
// back to is held (mpm_pool_reserve).

bool mpmPoolReusing() { return ALOAD(gMpmCounters[MPM_C_REUSE]) != 0u; }

// Per parcel in flight: the slot it left is held.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_pool_reserve(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (!mpmPoolReusing() || k >= min(ALOAD(gMpmCounters[MPM_C_PARCELS]), asuint(p.couple.z))) return;
    if (gMpmParcels[2u * k + 1u].w <= 0.0) return;
    uint source = asuint(gMpmParcels[2u * asuint(p.couple.z) + k].x);
    gMpmParticles[source].f2.w = asfloat(asuint(gMpmParticles[source].f2.w) | MPM_PARTICLE_RESERVED);
}

// Whether slot i is held (live, or a dead slot a parcel comes back to);
// `release` clears the parcel's mark (the pass's last read of it).
bool mpmPoolHeld(uint i, bool release)
{
    uint flags = asuint(gMpmParticles[i].f2.w);
    if (release && (flags & MPM_PARTICLE_RESERVED) != 0u)
        gMpmParticles[i].f2.w = asfloat(flags & ~MPM_PARTICLE_RESERVED);
    return (flags & (MPM_PARTICLE_DEAD | MPM_PARTICLE_RESERVED)) != MPM_PARTICLE_DEAD;
}

// Per particle: one past the highest held slot.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_pool_top(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (!mpmPoolReusing() || i >= mpmParticleCount()) return;
    if (mpmPoolHeld(i, true)) AMAX2(gMpmCounters[MPM_C_POOL_TOP], i + 1u);
}

// After mpm_pool_top: the pool ends after its last held slot.
[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_pool_trim(uint3 tid : SV_DispatchThreadID)
{
    if (!mpmPoolReusing()) return;
    ASTORE(gMpmCounters[MPM_C_PARTICLES], ALOAD(gMpmCounters[MPM_C_POOL_TOP]));
    ASTORE(gMpmCounters[MPM_C_POOL_TOP], 0u);
    ASTORE(gMpmCounters[MPM_C_REUSE], 0u);
}

// Per 64 slots of the pool (a thread each): its free slots, counted after
// the slot list's room (gMpmSorted[counts.x + chunk]; between frames the
// sort's scratch).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_pool_holes_count(uint3 tid : SV_DispatchThreadID)
{
    uint c = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint have = mpmParticleCount();
    if (!mpmPoolReusing() || c >= (have + 63u) / 64u) return;
    uint n = 0u;
    for (uint i = 64u * c; i < min(64u * c + 64u, have); i++)
        if (!mpmPoolHeld(i, false)) n++;
    gMpmSorted[p.counts.x + c] = n;
}

// One workgroup: each chunk's first place in the slot list, and the count.
[shader("compute")]
[numthreads(256, 1, 1)]
void mpm_pool_holes_scan(uint3 lid : SV_GroupThreadID)
{
    uint l = lid.x;
    MpmParams p = mpmParams();
    // No early return before the barriers (WGSL needs uniform control flow):
    // with reuse off there are no chunks.
    bool on = mpmPoolReusing();
    uint chunks = on ? (mpmParticleCount() + 63u) / 64u : 0u;
    uint per = (chunks + 255u) / 256u;
    uint first = l * per, last = min(first + per, chunks);
    uint sum = 0u;
    for (uint c = first; c < last; c++) sum += gMpmSorted[p.counts.x + c];
    gsMpmScan[l] = sum;
    GROUP_SYNC();
    for (uint d = 1u; d < 256u; d <<= 1u)
    {
        uint add = l >= d ? gsMpmScan[l - d] : 0u;
        GROUP_SYNC();
        gsMpmScan[l] += add;
        GROUP_SYNC();
    }
    uint start = gsMpmScan[l] - sum;
    for (uint c = first; c < last; c++)
    {
        // The count used before the store: Slang's CUDA output would otherwise
        // load it after (engine-cuda's build rejects that).
        uint next = start + gMpmSorted[p.counts.x + c];
        gMpmSorted[p.counts.x + c] = start;
        start = next;
    }
    if (on && l == 255u) ASTORE(gMpmCounters[MPM_C_HOLES], gsMpmScan[255]);
}

// Per 64 slots: its free slots into the list, in slot order (gMpmSorted
// from 0), releasing the parcels' marks.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_pool_holes_list(uint3 tid : SV_DispatchThreadID)
{
    uint c = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint have = mpmParticleCount();
    if (!mpmPoolReusing() || c >= (have + 63u) / 64u) return;
    uint at = gMpmSorted[p.counts.x + c];
    for (uint i = 64u * c; i < min(64u * c + 64u, have); i++)
        if (!mpmPoolHeld(i, true)) gMpmSorted[at++] = i;
}

// After the list: the activation takes the free slots first, then the pool
// grows by the rest (all or nothing against max particles); the emit's
// workgroups cover every particle of the activation.
[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_pool_finish(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmParams();
    uint total = ALOAD(gMpmCounters[MPM_C_REUSE]);
    if (total == 0u) return;
    uint have = mpmParticleCount();
    uint used = min(ALOAD(gMpmCounters[MPM_C_HOLES]), total);
    if (have + (total - used) > p.counts.x)
    {
        AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_PARTICLES);
        return;
    }
    ASTORE(gMpmCounters[MPM_C_HOLES], used);
    ASTORE(gMpmCounters[MPM_C_PARTICLES], have + (total - used));
    AADD2(gMpmCounters[MPM_C_LIVE], total);
    gMpmArgs[3] = (have + total + MPM_GROUP - 1u) / MPM_GROUP;
}

// Packing the parcels (a bake leaving areas open): one workgroup, each lane
// a run of slots; those still in flight go, in slot order, to the staging
// records past the parcels' own (gMpmParcels[3·max ..]), and the count is
// theirs. Nothing reads a parcel's slot (landing, deposits and the slot it
// returns to go by its own record), so packing changes no result.
[shader("compute")]
[numthreads(256, 1, 1)]
void mpm_parcels_pack(uint3 lid : SV_GroupThreadID)
{
    uint l = lid.x;
    MpmParams p = mpmParams();
    // No early return before the barriers (uniform control flow).
    bool on = mpmPoolReusing();
    uint maxP = asuint(p.couple.z);
    uint n = on ? min(ALOAD(gMpmCounters[MPM_C_PARCELS]), maxP) : 0u;
    uint per = (n + 255u) / 256u;
    uint first = l * per, last = min(first + per, n);
    uint sum = 0u;
    for (uint k = first; k < last; k++) if (gMpmParcels[2u * k + 1u].w > 0.0) sum++;
    gsMpmScan[l] = sum;
    GROUP_SYNC();
    for (uint d = 1u; d < 256u; d <<= 1u)
    {
        uint add = l >= d ? gsMpmScan[l - d] : 0u;
        GROUP_SYNC();
        gsMpmScan[l] += add;
        GROUP_SYNC();
    }
    uint at = gsMpmScan[l] - sum;
    for (uint k = first; k < last; k++)
    {
        if (!(gMpmParcels[2u * k + 1u].w > 0.0)) continue;
        gMpmParcels[3u * maxP + 2u * at] = gMpmParcels[2u * k];
        gMpmParcels[3u * maxP + 2u * at + 1u] = gMpmParcels[2u * k + 1u];
        gMpmParcels[5u * maxP + at] = gMpmParcels[2u * maxP + k];
        at++;
    }
    GROUP_SYNC();
    if (on && l == 255u) ASTORE(gMpmCounters[MPM_C_PARCELS], gsMpmScan[255]);
}

// After mpm_parcels_pack, per parcel slot: the packed parcels back in place.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_parcels_unpack(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (!mpmPoolReusing()) return;
    uint maxP = asuint(p.couple.z);
    if (k >= min(ALOAD(gMpmCounters[MPM_C_PARCELS]), maxP)) return;
    gMpmParcels[2u * k] = gMpmParcels[3u * maxP + 2u * k];
    gMpmParcels[2u * k + 1u] = gMpmParcels[3u * maxP + 2u * k + 1u];
    gMpmParcels[2u * maxP + k] = gMpmParcels[5u * maxP + k];
}

// Heightfield records (four samples each) touching the window, as a range.
void mpmWindowRecords(HeightFieldRef hf, MpmParams p, out uint r0, out uint r1)
{
    r0 = (p.window.y * hf.nx + p.window.x) >> 2;
    r1 = ((p.window.y + p.window.w - 1u) * hf.nx + p.window.x + p.window.z - 1u) >> 2;
}

// Per heightfield record (its own four samples, so no two threads share one):
// window samples lowered by the quanta they gave (activate) or raised by the
// quanta baked onto them (bake).
void mpmAdjustRecord(uint r, bool raise, MpmParams p, HeightFieldRef hf)
{
    uint4 rec = gHulls[hf.base + 2u + r];
    float area = mpmSampleArea(hf);
    for (uint c = 0u; c < 4u; c++)
    {
        uint k = 4u * r + c;
        uint i = k % hf.nx, j = k / hf.nx;
        if (j >= hf.nz || i < p.window.x || i >= p.window.x + p.window.z || j < p.window.y || j >= p.window.y + p.window.w)
            continue;
        uint t = (j - p.window.y) * p.window.z + (i - p.window.x);
        float h = asfloat(rec[c]);
        if (raise)
        {
            h += float(ALOAD(gMpmColumns[mpmCol(p, t) + 3u])) * p.column.y / area;
            // Order-preserving bits of h (a float's sign flips the order).
            uint b = asuint(h);
            AMAX2(gMpmCounters[MPM_C_BAKE_MAX], (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u));
        }
        else
        {
            // The quanta it gave now; [3] then starts the bake's tally at 0.
            h -= float(ALOAD(gMpmColumns[mpmCol(p, t) + 3u])) * p.column.y / area;
            ASTORE(gMpmColumns[mpmCol(p, t) + 3u], 0u);
        }
        rec[c] = asuint(h);
    }
    gHulls[hf.base + 2u + r] = rec;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_activate_lower(uint3 tid : SV_DispatchThreadID)
{
    MpmParams p = mpmSelArea();
    if (mpmActivationRefused())
    {
        if (mpmIndex(tid) == 0u) ASTORE(gMpmCounters[MPM_C_REUSE], 0u);
        return;
    }
    Collider col = gColliders[p.dims.w];
    HeightFieldRef hf = heightFieldOf(col);
    uint r0, r1;
    mpmWindowRecords(hf, p, r0, r1);
    uint r = r0 + mpmIndex(tid);
    if (r <= r1) mpmAdjustRecord(r, false, p, hf);
    if (mpmIndex(tid) == 0u)
    {
        ASTORE(gMpmCounters[MPM_C_REUSE], 0u);
        // Lowered samples sit within half a quantum's height of the bed.
        uint4 dims = gHulls[hf.base + 1u];
        float lowest = p.column.x - p.column.y / mpmSampleArea(hf);
        dims.z = asuint(min(asfloat(dims.z), lowest));
        gHulls[hf.base + 1u] = dims;
    }
}

// The bake (mpm_bake_bin) and the parcels' deposits (mpm_parcels) split each
// particle's or parcel's quanta between heightfield samples, in whole quanta
// that sum exactly to the particle's (mpmTentSplit, mpmBsplineSplit).
//
// Two kernels, blended by how far a particle made by activation has moved,
// w = max(0, 1 - |x - x₀|/(fade·s)) (x₀ where it was emitted, in MpmParticle
// c1.w, c2.w; s the activation lattice's spacing, params column.z; fade =
// params limits.y, TerrainConfig::bake_home_fade, by default half a lattice
// spacing: the particle has left its own lattice cell):
// - The share w is the heightfield the particle came from: the tent (bilinear,
//   four samples) at its home sample plus its displacement. Activation lays a
//   sample's particles on a lattice across the sample's cell; at zero
//   displacement this puts all of it back on the home sample, so
//   activate-then-bake is the identity exactly.
// - The share 1 - w is material the MPM moved: the quadratic B-spline (nine
//   samples) at its own position, the kernel P2G gives the grid its mass
//   with. The simulation resolves no finer than that: particles may part
//   within a cell (a sub-cell crack, as along a crater's mirror planes through
//   the impact point, where nothing mixes across) without any force on the
//   grid to close it. A narrower kernel bakes such a gap as a trench a sample
//   wide; the grid's own kernel bakes the mass the simulation computed with.
// Particles not made by activation (loaded by the host, or landed parcels)
// have w = 0. Sand pushed off a buried body (mpmBakeColumn) is placed by that
// rule, not by the flow: it splats with the tent at its new place.
//
// On the absorbing ground the bake measures positions in the unactivated
// ground's frame (mpmGroundShift). That ground is a heightfield and cannot
// move, but the absorbing walls give, and the material against them moves
// out with them: the window's edge columns end displaced outward by the
// walls' give (metres at the browser demo's scale, where the ground is
// 50 MPa sand 350 m deep). Splatted in the region's frame, that shift crosses
// the window's edge and folds back onto its last samples (mpmBakeOnto): a
// ridge a sample wide, tens of metres high, over a deficit spread inside. In
// the ground's frame the walls are where they were: each particle's position
// has the walls' displacement taken out, interpolated linearly between
// opposite walls (the uniform strain an elastic layer between two displaced
// walls takes). The walls' displacement is their edge columns' mean outward
// displacement (mpm_bake_shift): material can only move out past a side's
// plane by moving the ground with it. (The faces' own records hold the grid's
// velocity at the plane, integrated, which the dashpot-held node layers there
// pull to a fraction of the material's a cell in.) Volume is kept exactly;
// the region's lateral give comes back as a uniform rise of its columns.

// Once before the bake on the absorbing ground, per sample along the window's
// edge (low-x then high-x side's rows, then low-z then high-z side's
// columns): its column's mean displacement across that side (quanta-weighted,
// the particles activation made there, in slot order), into the face records'
// space (the faces are done with).
// The list (in gMpmCells, three words each: count, start, cursor) of the
// window's edge sample t (window-local ti, tj), or MPM_NO_SLOT for an inner
// sample: low-z row, high-z row, then the low-x and high-x columns between.
uint mpmEdgeList(int ti, int tj, MpmParams p)
{
    int ni = int(p.window.z), nj = int(p.window.w);
    if (tj == 0) return uint(ti);
    if (tj == nj - 1) return uint(ni + ti);
    if (ti == 0) return uint(2 * ni + tj - 1);
    if (ti == ni - 1) return uint(2 * ni + nj - 2 + tj - 1);
    return MPM_NO_SLOT;
}

// The edge list of particle q's home sample in the selected area p (live,
// made by activation, inside p's box), or MPM_NO_SLOT.
uint mpmParticleEdgeList(MpmParticle q, MpmParams p, HeightFieldRef hf)
{
    if (mpmParticleDead(q) || (asuint(q.f2.w) & MPM_PARTICLE_HOME) == 0u) return MPM_NO_SLOT;
    if (!mpmBakeOwns(q.xm.xyz)) return MPM_NO_SLOT;
    int si = clamp(int(round((q.c1.w - hf.x0) / hf.dx)), int(p.window.x), int(p.window.x + p.window.z) - 1);
    int sj = clamp(int(round((q.c2.w - hf.z0) / hf.dz)), int(p.window.y), int(p.window.y + p.window.w) - 1);
    return mpmEdgeList(si - int(p.window.x), sj - int(p.window.y), p);
}

// The bake's edge lists: the particles activation made on each sample along
// the selected area's window edge, by slot (a counting sort, so the walls'
// give sums them in a fixed order however the pool's slots lie).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_edge_clear(uint3 tid : SV_DispatchThreadID)
{
    uint e = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    if (e >= 2u * (p.window.z + p.window.w)) return;
    for (uint w = 0u; w < 3u; w++) ASTORE(gMpmCells[3u * e + w], 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_edge_count(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = mpmSelArea();
    HeightFieldRef hf = heightFieldOf(gColliders[p.dims.w]);
    uint e = mpmParticleEdgeList(gMpmParticles[i], p, hf);
    if (e != MPM_NO_SLOT) AADD2(gMpmCells[3u * e], 1u);
}

// One workgroup: each list's start (exclusive prefix over the lists' counts).
[shader("compute")]
[numthreads(256, 1, 1)]
void mpm_bake_edge_scan(uint3 lid : SV_GroupThreadID)
{
    uint l = lid.x;
    MpmParams p = mpmSelArea();
    uint lists = 2u * (p.window.z + p.window.w);
    uint chunk = (lists + 255u) / 256u;
    uint first = l * chunk, last = min(first + chunk, lists);
    uint sum = 0u;
    for (uint e = first; e < last; e++) sum += ALOAD(gMpmCells[3u * e]);
    gsMpmScan[l] = sum;
    GROUP_SYNC();
    for (uint d = 1u; d < 256u; d <<= 1u)
    {
        uint add = l >= d ? gsMpmScan[l - d] : 0u;
        GROUP_SYNC();
        gsMpmScan[l] += add;
        GROUP_SYNC();
    }
    uint start = gsMpmScan[l] - sum;
    for (uint e = first; e < last; e++)
    {
        ASTORE(gMpmCells[3u * e + 1u], start);
        start += ALOAD(gMpmCells[3u * e]);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_edge_scatter(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= mpmParticleCount()) return;
    MpmParams p = mpmSelArea();
    HeightFieldRef hf = heightFieldOf(gColliders[p.dims.w]);
    uint e = mpmParticleEdgeList(gMpmParticles[i], p, hf);
    if (e == MPM_NO_SLOT) return;
    uint k;
    AADD3(gMpmCells[3u * e + 2u], 1u, k);
    gMpmSorted[ALOAD(gMpmCells[3u * e + 1u]) + k] = i;
}

// Per list: its range by slot (insertion sort; the order the sums take).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_edge_sort(uint3 tid : SV_DispatchThreadID)
{
    uint e = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    if (e >= 2u * (p.window.z + p.window.w)) return;
    uint first = ALOAD(gMpmCells[3u * e + 1u]), n = ALOAD(gMpmCells[3u * e]);
    for (uint a = 1u; a < n; a++)
    {
        uint v = gMpmSorted[first + a];
        uint k = a;
        while (k > 0u && gMpmSorted[first + k - 1u] > v)
        {
            gMpmSorted[first + k] = gMpmSorted[first + k - 1u];
            k--;
        }
        gMpmSorted[first + k] = v;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_bake_shift(uint3 tid : SV_DispatchThreadID)
{
    uint k = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    uint ni = p.window.z, nj = p.window.w;
    if (k >= 2u * (ni + nj)) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    uint2 ij;
    uint axis;
    if (k < 2u * nj)
    {
        ij = uint2(k < nj ? 0u : ni - 1u, k % nj);
        axis = 0u;
    }
    else
    {
        uint m = k - 2u * nj;
        ij = uint2(m % ni, m < ni ? 0u : nj - 1u);
        axis = 2u;
    }
    gMpmCoupled[p.boundary.y + k] = float4(mpmEdgeGive(k, ij, axis, p, f), 0, 0, 0);
}

