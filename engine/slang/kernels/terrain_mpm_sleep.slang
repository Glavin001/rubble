// ---- area clocks and block sleep ---------------------------------------------
// engine/docs/deformable-terrain.md, "Area clocks and block sleep".

// Area clocks: before each substep pass, the pass's dt and CFL bound into the
// domain record, which areas sit it out, and whether its fresh particles take
// their geostatic F first (each area group's first pass). The pass table
// follows the area records (TerrainPlan::area_passes); its last record puts
// the frame's own dt back.
[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_pass(uint3 tid : SV_DispatchThreadID)
{
    if (mpmIndex(tid) != 0u) return;
    uint k = ALOAD(gMpmCounters[MPM_C_PASS]);
    float4 e = gMpmParams[1u + MPM_MAX_AREAS + k].step;
    MpmParams d = gMpmParams[0];
    d.step.x = e.x;
    d.limits.x = e.y;
    gMpmParams[0] = d;
    ASTORE(gMpmCounters[MPM_C_SKIP], asuint(e.z));
    if (asuint(e.w) != 0u) ASTORE(gMpmCounters[MPM_C_GEOSTATIC], 1u);
    ASTORE(gMpmCounters[MPM_C_PASS], k + 1u);
}

// Block sleep, in mpm_touch: asleep particle i at x (area `area`) stays asleep
// unless a coupled collider's padded bounds reach its block cell (the body may
// press on it this frame); woken, it clears its flag and joins as awake.
bool mpmTouchStaysAsleep(uint i, float3 x, uint area, uint flags)
{
    MpmParams pa = mpmArea(area);
    float block = pa.origin.w * float(MPM_BLOCK);
    uint cell = mpmLatticeCell(int3(floor((x - pa.origin.xyz) / block)), pa);
    if (cell == MPM_NO_BLOCK || ALOAD(gMpmCouple[4u + cell * (1u + MPM_CELL_COLLIDERS)]) == 0u) return true;
    gMpmParticles[i].f2.w = asfloat(flags & ~(MPM_PARTICLE_ASLEEP | MPM_PARTICLE_QUIET));
    return false;
}

// Block sleep: each wheel load's stencil blocks are awake blocks (the load
// presses on what stands there, asleep or not).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_sleep_wheels(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    if (i >= ALOAD(gMpmCounters[MPM_C_WHEELS])) return;
    MpmParams p = mpmParams();
    uint r = asuint(p.limits.z) + MPM_WHEEL_RECORDS * i;
    if (mpmAreaSkips(asuint(gMpmCoupled[r].w))) return;
    int3 b = asint(gMpmCoupled[r + 1u].xyz);
    int3 lo = b >> 2, hi = (b + 2) >> 2;
    for (int bz = lo.z; bz <= hi.z; bz++)
    for (int by = lo.y; by <= hi.y; by++)
    for (int bx = lo.x; bx <= hi.x; bx++)
        if (mpmKeyOk(int3(bx, by, bz))) mpmInsertBlock(mpmBlockKey(int3(bx, by, bz)), mpmParams());
}

// Block sleep: the blocks so far are a ring (the first call: the awake
// blocks; the second: those and their neighbours). mpm_clear unsets both.
[shader("compute")]
[numthreads(1, 1, 1)]
void mpm_sleep_ring(uint3 tid : SV_DispatchThreadID)
{
    if (mpmIndex(tid) != 0u) return;
    uint blocks = min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), mpmParams().counts.z);
    if (ALOAD(gMpmCounters[MPM_C_SLEEP_RING0]) == MPM_NO_BLOCK) ASTORE(gMpmCounters[MPM_C_SLEEP_RING0], blocks);
    else ASTORE(gMpmCounters[MPM_C_SLEEP_RING1], blocks);
}

// Block sleep: the last ring's blocks insert their 26 neighbours.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_sleep_dilate(uint3 tid : SV_DispatchThreadID)
{
    uint b = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint ring1 = ALOAD(gMpmCounters[MPM_C_SLEEP_RING1]);
    uint ring = ring1 == MPM_NO_BLOCK ? ALOAD(gMpmCounters[MPM_C_SLEEP_RING0]) : ring1;
    if (b >= ring) return;
    int3 k = mpmKeyBlock(ALOAD(gMpmBlocks[4u * b]));
    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
    {
        int3 kn = k + int3(x, y, z);
        if (mpmKeyOk(kn)) mpmInsertBlock(mpmBlockKey(kn), p);
    }
}

// Block sleep: an asleep particle joins the substep, frozen, where its base
// block is in the rings (it then pushes on the awake material with its mass
// and its stress); elsewhere it takes no part.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_sleep_touch(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    uint flags = asuint(gMpmParticles[i].f2.w);
    if ((flags & (MPM_PARTICLE_DEAD | MPM_PARTICLE_ASLEEP)) != MPM_PARTICLE_ASLEEP) return;
    uint n = p.counts.x;
    // Woken in mpm_touch (its slot then holds its key).
    if (gMpmSorted[n + i] != MPM_NO_BLOCK) return;
    float3 x = gMpmParticles[i].xm.xyz;
    uint area = mpmAreaOf(x, p);
    if (area == MPM_NO_AREA || mpmAreaSkips(area)) return;
    int3 b = mpmBaseNodeIn(x, area);
    if (!mpmKeyOk(b >> 2)) return;
    uint key = mpmBlockKey(b >> 2);
    if (mpmFindBlock(key, p) == MPM_NO_BLOCK) return;
    uint3 l = uint3(b & 3);
    gMpmSorted[n + i] = key;
    gMpmSorted[2u * n + i] = l.x + 4u * l.y + 16u * l.z;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_count(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    uint n = p.counts.x;
    uint key = gMpmSorted[n + i];
    uint c = MPM_NO_BLOCK;
    if (key != MPM_NO_BLOCK)
    {
        uint b = mpmFindBlock(key, p);
        if (b != MPM_NO_BLOCK) c = b * MPM_BLOCK_NODES + gMpmSorted[2u * n + i];
    }
    // Its cell and its rank among the cell's particles, for mpm_scatter (the
    // rank's order is the atomics'; mpm_cell_sort fixes the order).
    gMpmSorted[n + i] = c;
    if (c == MPM_NO_BLOCK) return;
    uint k;
    AADD3(gMpmCells[4u * c], 1u, k);
    gMpmSorted[2u * n + i] = k;
}

// One workgroup per block: exclusive scan of its 64 cell counts.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_scan_cells(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint b = gid.x + gid.y * 32768u;
    uint l = lid.x;
    MpmParams p = mpmParams();
    bool live = b < min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z);
    uint c = b * MPM_BLOCK_NODES + l;
    uint n = live ? ALOAD(gMpmCells[4u * c]) : 0u;
    gsMpmScan[l] = n;
    GROUP_SYNC();
    // Hillis–Steele inclusive scan over 64 lanes.
    for (uint d = 1u; d < MPM_BLOCK_NODES; d <<= 1u)
    {
        uint add = l >= d ? gsMpmScan[l - d] : 0u;
        GROUP_SYNC();
        gsMpmScan[l] += add;
        GROUP_SYNC();
    }
    if (!live) return;
    ASTORE(gMpmCells[4u * c + 1u], gsMpmScan[l] - n);
    if (l == MPM_BLOCK_NODES - 1u)
    {
        // The block's range in gMpmSorted. Where it lands does not change any
        // sum (P2G adds by cell and particle id), so an atomic claim does.
        uint total = gsMpmScan[l];
        uint start;
        AADD3(gMpmCounters[MPM_C_SORTED], total, start);
        ASTORE(gMpmBlocks[4u * b + 1u], total);
        ASTORE(gMpmBlocks[4u * b + 2u], start);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_scatter(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    uint n = p.counts.x;
    uint c = gMpmSorted[n + i];
    if (c == MPM_NO_BLOCK) return;
    uint k = gMpmSorted[2u * n + i];
    uint b = c / MPM_BLOCK_NODES;
    gMpmSorted[ALOAD(gMpmBlocks[4u * b + 2u]) + ALOAD(gMpmCells[4u * c + 1u]) + k] = i;
}

// Range [first, first + n) of cell c in gMpmSorted.
void mpmCellRange(uint c, out uint first, out uint n)
{
    uint b = c / MPM_BLOCK_NODES;
    first = ALOAD(gMpmBlocks[4u * b + 2u]) + ALOAD(gMpmCells[4u * c + 1u]);
    n = ALOAD(gMpmCells[4u * c]);
}

// Insertion sort of each cell's (short) range by particle id.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_cell_sort(uint3 tid : SV_DispatchThreadID)
{
    uint c = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (c >= min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z) * MPM_BLOCK_NODES) return;
    uint first, n;
    mpmCellRange(c, first, n);
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

// Cohesive soil (T5): F returned to the stress-space Mohr–Coulomb surface and
// its apex, updating q's F and log Jp, then its Hencky stress.
float3x3 mpmSoilStress(inout MpmParticle q, MpmParams p)
{
    float3x3 zero = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    if (!mpmSvdInputValid(F) || !mpmSandInputValid(float3(1, 1, 1), q.vj.w))
    {
        AOR2(gMpmCounters[MPM_C_FLAGS], MPM_INVALID);
        return zero;
    }
    float3x3 U, V;
    float3 sigma;
    mpmSvd3(F, U, sigma, V);
    float3 s;
    float logJp;
    mpmSoilProjectMohrCoulomb(sigma, q.vj.w, p.sand.z, p.sand.w, p.sand.x, p.sand.y, s, logJp);
    F = mul(U, mul(float3x3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z), transpose(V)));
    q.f0.xyz = F[0];
    q.f1.xyz = F[1];
    q.f2.xyz = F[2];
    q.vj.w = logJp;
    return mpmHenckyKirchhoff(U, s, p.sand.x, p.sand.y);
}

// Rock (T5): F is elastic; the damage in q.c0.w grows with the largest
// principal stress, and the stress is Hencky's with its tensile part scaled by
// (1 − d). The crack band is the cell size.
float3x3 mpmRockStress(inout MpmParticle q, MpmParams p)
{
    float3x3 zero = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    if (!mpmSvdInputValid(F) || !(q.c0.w >= 0.0 && q.c0.w <= 1.0))
    {
        AOR2(gMpmCounters[MPM_C_FLAGS], MPM_INVALID);
        return zero;
    }
    float3x3 U, V;
    float3 sigma;
    mpmSvd3(F, U, sigma, V);
    float3 t;
    float d;
    mpmRockDamage(sigma, q.c0.w, p.sand.x, p.sand.y, p.sand.z, p.sand.w, p.origin.w, t, d);
    q.c0.w = d;
    return mpmPrincipalKirchhoff(U, t);
}

// Kirchhoff stress τ of particle q (zero for MPM_MATERIAL_NONE). Sand returns F
// to its yield surface first (Mohr–Coulomb, or Klár et al.'s Drucker–Prager
// cone), updating q's F and log Jp. Cohesive soil and rock have their own
// functions, ahead of the dry-sand path.
float3x3 mpmStress(inout MpmParticle q, MpmParams p)
{
    if (p.counts.w == MPM_MATERIAL_SOIL_MC) return mpmSoilStress(q, p);
    if (p.counts.w == MPM_MATERIAL_ROCK) return mpmRockStress(q, p);
    float3x3 zero = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (p.counts.w != MPM_MATERIAL_SAND && p.counts.w != MPM_MATERIAL_SAND_MC) return zero;
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    if (!mpmSvdInputValid(F) || !mpmSandInputValid(float3(1, 1, 1), q.vj.w))
    {
        AOR2(gMpmCounters[MPM_C_FLAGS], MPM_INVALID);
        return zero;
    }
    float3x3 U, V;
    float3 sigma;
    mpmSvd3(F, U, sigma, V);
    float3 s;
    float logJp;
    if (p.counts.w == MPM_MATERIAL_SAND_MC)
        mpmSandProjectMohrCoulomb(sigma, q.vj.w, p.sand.z, p.sand.w, p.sand.x, p.sand.y, s, logJp);
    else mpmSandProject(sigma, q.vj.w, p.sand.z, p.sand.w, p.sand.x, p.sand.y, s, logJp);
    F = mul(U, mul(float3x3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z), transpose(V)));
    q.f0.xyz = F[0];
    q.f1.xyz = F[1];
    q.f2.xyz = F[2];
    q.vj.w = logJp;
    return mpmHenckyKirchhoff(U, s, p.sand.x, p.sand.y);
}

// The payload of slot s, the first substep after a geostatic activation:
// particle q's geostatic F first (mpmGeostaticStart), then as mpm_payload.
// A function of its own behind mpm_payload's early return, so the usual
// path's code (and its fast-math results) is unchanged.
void mpmPayloadGeostatic(uint s, MpmParams p)
{
    uint id = gMpmSorted[s];
    MpmParticle q = gMpmParticles[id];
    mpmGeostaticFresh(q, p);
    float m = q.xm.w;
    float dx = p.origin.w;
    float3x3 C = float3x3(q.c0.xyz, q.c1.xyz, q.c2.xyz);
    float3x3 tau = mpmStress(q, p);
    gMpmParticles[id] = q;
    float3x3 A = C * m - tau * (p.step.x * q.f0.w * 4.0 / (dx * dx));
    MpmPayload o;
    o.xm = q.xm;
    o.p0 = float4(q.vj.xyz * m, A[0].x);
    o.p1 = float4(A[0].y, A[0].z, A[1].x, A[1].y);
    o.p2 = float4(A[1].z, A[2].x, A[2].y, A[2].z);
    gMpmPayload[s] = o;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_payload(uint3 tid : SV_DispatchThreadID)
{
    uint s = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (s >= ALOAD(gMpmCounters[MPM_C_SORTED])) return;
    // The first substep after a geostatic activation: its F first.
    if (ALOAD(gMpmCounters[MPM_C_GEOSTATIC]) != 0u)
    {
        mpmPayloadGeostatic(s, p);
        return;
    }
    uint id = gMpmSorted[s];
    MpmParticle q = gMpmParticles[id];
    float m = q.xm.w;
    float dx = p.origin.w;
    float3x3 C = float3x3(q.c0.xyz, q.c1.xyz, q.c2.xyz);
    float3x3 tau = mpmStress(q, p);
    if (p.counts.w != MPM_MATERIAL_NONE) gMpmParticles[id] = q;
    float3x3 A = C * m - tau * (p.step.x * q.f0.w * 4.0 / (dx * dx));
    MpmPayload o;
    o.xm = q.xm;
    o.p0 = float4(q.vj.xyz * m, A[0].x);
    o.p1 = float4(A[0].y, A[0].z, A[1].x, A[1].y);
    o.p2 = float4(A[1].z, A[2].x, A[2].y, A[2].z);
    gMpmPayload[s] = o;
}

// P2G's lanes per node: each sums the cells o = lane, lane + 4, ... of
// the 27, and the lanes' sums add in lane order: a fixed order, so runs and
// backends still agree bit for bit. Each node's gather is a chain of loads;
// more lanes per node run more of them at once.
static const uint MPM_P2G_LANES = 4u;
groupshared float4 gsMpmP2g[64u * MPM_P2G_LANES];

// One workgroup per block, MPM_P2G_LANES threads per node: the 27 cells whose
// particles' stencils cover the node, each cell's particles by id.
[shader("compute")]
[numthreads(64 * 4, 1, 1)]
void mpm_p2g(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint b = gid.x + gid.y * 32768u;
    uint l = lid.x % MPM_BLOCK_NODES;
    uint lane = lid.x / MPM_BLOCK_NODES;
    MpmParams p = mpmParams();
    bool live = b < min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z);
    float3 mv = float3(0, 0, 0);
    float m = 0.0;
    if (live)
    {
        uint key = ALOAD(gMpmBlocks[4u * b]);
        int3 g = mpmKeyNode(key, l);
        float dx = p.origin.w;
        // The node's area: every particle whose stencil reaches it is that
        // area's, in that area's frame (none: the node has no mass).
        uint area = mpmNodeArea(g, p);
        float3 ao = area == MPM_NO_AREA ? float3(0, 0, 0) : gMpmParams[1u + area].origin.xyz;
        int3 aoff = area == MPM_NO_AREA ? int3(0, 0, 0) : mpmSlotOffset(area);
        // Bases g - 2 .. g lie in the blocks from (g - 2) >> 2.
        int3 k0 = (g - 2) >> 2;
        uint blk[8];
        mpmBlockCube(k0, p, blk);
        for (uint o = lane; o < 27u && area != MPM_NO_AREA; o += MPM_P2G_LANES)
        {
            uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
            int3 base = g - int3(off);
            uint c = mpmCellInCube(base, k0, blk);
            if (c == MPM_NO_BLOCK) continue;
            uint first, count;
            mpmCellRange(c, first, count);
            for (uint k = 0u; k < count; k++)
            {
                MpmPayload q = gMpmPayload[first + k];
                float3 fx = (q.xm.xyz - ao) / dx - float3(base - aoff);
                float w = mpmWeightAt(mpmWeights(fx.x), off.x) * mpmWeightAt(mpmWeights(fx.y), off.y)
                    * mpmWeightAt(mpmWeights(fx.z), off.z);
                float3 dpos = (float3(off) - fx) * dx;
                float3x3 A = float3x3(float3(q.p0.w, q.p1.x, q.p1.y), float3(q.p1.z, q.p1.w, q.p2.x),
                    float3(q.p2.y, q.p2.z, q.p2.w));
                m += w * q.xm.w;
                mv += w * (q.p0.xyz + mul(A, dpos));
            }
        }
    }
    gsMpmP2g[lid.x] = float4(mv, m);
    GROUP_SYNC();
    if (!live || lane != 0u) return;
    float4 sum = gsMpmP2g[l];
    for (uint j = 1u; j < MPM_P2G_LANES; j++) sum += gsMpmP2g[l + j * MPM_BLOCK_NODES];
    uint n = b * MPM_BLOCK_NODES + l;
    gMpmGrid[2u * n] = sum;
    // The node's coupling record: impulse, coupled slot (a float, exact below
    // 2^24; -1 for none: an all-ones bit pattern is a NaN WGSL cannot spell).
    gMpmGrid[2u * n + 1u] = float4(0, 0, 0, -1.0);
}

// Signed distance from world point x to collider c (negative inside) and the
// outward normal there. Exact inside, which is all the grid contact uses; boxes and
// hulls by their face planes.
float mpmColliderSdf(Collider c, BodyState b, float3 x, out float3 n)
{
    Frame f = colliderFrame(c, b);
    float3 xl = frameToLocal(f, x);
    float d;
    float3 nl;
    if (c.shape == SHAPE_SPHERE || c.shape == SHAPE_CAPSULE)
    {
        float half = c.shape == SHAPE_CAPSULE ? c.params.y : 0.0;
        float3 core = float3(0.0, clamp(xl.y, -half, half), 0.0);
        float3 r = xl - core;
        float len = length(r);
        nl = len > 1e-12 ? r / len : float3(0, 1, 0);
        d = len - c.params.x;
    }
    else
    {
        HullRef h = hullOf(c);
        d = -1e30;
        nl = float3(0, 1, 0);
        for (uint k = 0u; k < h.nf; k++)
        {
            float4 pl = hullPlane(h, k);
            float s = dot(pl.xyz, xl) - pl.w;
            if (s > d) { d = s; nl = pl.xyz; }
        }
    }
    n = quatRotate(f.q, nl);
    return d;
}

// Bounds of collider c on body b padded by its motion over the frame and a cell.
void mpmColliderBounds(Collider c, BodyState b, MpmParams p, out float3 lo, out float3 hi)
{
    float3 x = transformPoint(b.position, b.orientation, c.localPos);
    float r = c.params.w;
    float pad = (length(b.linearVel) + length(b.angularVel) * r) * p.couple.x + p.origin.w;
    lo = x - (r + pad);
    hi = x + (r + pad);
}

// The block cell of an area's block k (area-local block coordinates) in
// gMpmCouple's cells (the area's run from its record's area.w), or MPM_NO_BLOCK.
uint mpmLatticeCell(int3 k, MpmParams p)
{
    int3 blocks = int3(p.dims.xyz / MPM_BLOCK);
    if (any(k < int3(0, 0, 0)) || any(k >= blocks)) return MPM_NO_BLOCK;
    uint c = p.area.w + uint(k.x + blocks.x * (k.y + blocks.y * k.z));
    return c < p.counts.z ? c : MPM_NO_BLOCK;
}

// Node n (area-local node g of area p) inside a coupled collider: hold it to
// the body's surface velocity (separating, Coulomb friction p.step.w) and record
// the momentum change for that body's reaction.
void mpmRigidContact(uint n, int3 g, float m, MpmParams p, inout float3 v)
{
    uint cell = mpmLatticeCell(g >> 2, p);
    if (cell == MPM_NO_BLOCK) return;
    uint base = 4u + cell * (1u + MPM_CELL_COLLIDERS);
    uint count = min(ALOAD(gMpmCouple[base]), MPM_CELL_COLLIDERS);
    if (count == 0u) return;
    float3 xw = p.origin.xyz + float3(g) * p.origin.w;
    float best = 0.0;
    uint bestSlot = MPM_NO_SLOT, bestCollider = 0xFFFFFFFFu;
    float3 bestN = float3(0, 1, 0);
    for (uint k = 0u; k < count; k++)
    {
        uint slot = ALOAD(gMpmCouple[base + 1u + k]);
        uint ci = asuint(gMpmCoupled[4u * slot].w);
        Collider col = gColliders[ci];
        float3 nn;
        float d = mpmColliderSdf(col, gBodies[col.body], xw, nn);
        // Deepest wins; ties go to the lower collider (order-independent).
        if (d < best || (d == best && d < 0.0 && ci < bestCollider))
        {
            best = d;
            bestSlot = slot;
            bestCollider = ci;
            bestN = nn;
        }
    }
    if (bestSlot == MPM_NO_SLOT) return;
    BodyState body = gBodies[gColliders[bestCollider].body];
    float3 vb = gMpmCoupled[4u * bestSlot + 2u].xyz
        + cross(gMpmCoupled[4u * bestSlot + 3u].xyz, xw - body.position);
    float3 rel = v - vb;
    float vn = dot(rel, bestN);
    if (vn >= 0.0) return;
    float3 vt = rel - bestN * vn;
    float st = length(vt);
    float mu = p.step.w;
    float3 relNew = st <= -mu * vn ? float3(0, 0, 0) : vt * (1.0 + mu * vn / st);
    float3 vNew = vb + relNew;
    // The impulse that would hold the node to the body as if the body were
    // immovable; mpm_couple_apply applies the share the body's mass allows.
    gMpmGrid[2u * n + 1u] = float4(m * (vNew - v), float(bestSlot));
}

// Wheel loads on domain node gd of area `area`: each listed wheel standing in
// the area moves the node by F·dt·w/M, w the node's B-spline weight at the
// contact and M the material's mass there (mpm_wheel_mass), so the nodes
// together take F·dt.
void mpmWheelLoads(int3 gd, uint area, MpmParams d, inout float3 v)
{
    uint count = ALOAD(gMpmCounters[MPM_C_WHEELS]);
    if (count == 0u) return;
    uint base = asuint(d.limits.z);
    for (uint i = 0u; i < count; i++)
    {
        uint r = base + MPM_WHEEL_RECORDS * i;
        float4 r0 = gMpmCoupled[r];
        if (asuint(r0.w) != area) continue;
        int3 off = gd - asint(gMpmCoupled[r + 1u].xyz);
        if (any(off < int3(0, 0, 0)) || any(off > int3(2, 2, 2))) continue;
        float4 fm = gMpmCoupled[r + 2u];
        if (!(fm.w > 0.0)) continue;
        uint3 o = uint3(off);
        float w = mpmWeightAt(mpmWeights(r0.x), o.x) * mpmWeightAt(mpmWeights(r0.y), o.y)
            * mpmWeightAt(mpmWeights(r0.z), o.z);
        v += fm.xyz * (d.step.x * w / fm.w);
    }
}

// Grid velocity: momentum over mass, gravity, then the bed (the heightfield
// under the region) as a separating Coulomb contact.
// Whether a grid node at xl (the bed's local frame) lies in the rigid bed, and
// the bed's outward normal there. With an activation window open, the rigid
// bed is the window's lowered floor and, around it, the ground left as it was:
// a node under that ground faces the pit through the window's side (a wall),
// never up through the ground's own surface, which would let the region's
// material squeeze out sideways beneath it. Without a window: the field's
// surface.
bool mpmBedNormal(HeightFieldRef hf, float3 xl, MpmParams p, out float3 nl)
{
    nl = float3(0, 1, 0);
    if (p.window.z == 0u || p.window.w == 0u)
    {
        float h;
        return hfSurface(hf, xl.x, xl.z, h, nl) && (xl.y - h) * nl.y <= 0.0;
    }
    int si = clamp(int(round((xl.x - hf.x0) / hf.dx)), 0, int(hf.nx) - 1);
    int sj = clamp(int(round((xl.z - hf.z0) / hf.dz)), 0, int(hf.nz) - 1);
    int2 lo = int2(p.window.xy), hi = int2(p.window.xy + p.window.zw) - 1;
    float wx = si < lo.x ? 1.0 : (si > hi.x ? -1.0 : 0.0);
    float wz = sj < lo.y ? 1.0 : (sj > hi.y ? -1.0 : 0.0);
    if (wx == 0.0 && wz == 0.0) return xl.y <= hfHeight(hf, uint(si), uint(sj));
    float h;
    if (xl.y < hfHeight(hf, uint(si), uint(sj)))
    {
        nl = normalize(float3(wx, 0.0, wz));
        return true;
    }
    return hfSurface(hf, xl.x, xl.z, h, nl) && (xl.y - h) * nl.y <= 0.0;
}

