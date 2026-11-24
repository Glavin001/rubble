// --- CSR adjacency (three-avbd: degree → scan → scatter → sort) ----------------------------
// Constraint id: [0, jointCount) = joints; jointCount + mi = manifold mi (Stage 8b).
// CINDEX(0..bodyCount): degree then exclusive starts; [bodyCount] = total.
// gBodyColor reused as fill cursors during scatter (init_colors overwrites afterward).
// gConstraintIds holds the packed adjacency list (after broadphase hash is done).

uint2 constraintEndpoints(uint id)
{
    uint nJ = gCounters[0].jointCount;
    if (id < nJ)
        return uint2(gJoints[id].bodyA, gJoints[id].bodyB);
    Manifold m = gManifolds[id - nJ];
    return uint2(gColliders[m.colliderA].body, gColliders[m.colliderB].body);
}

#if defined(ENGINE_WGSL)
// CSR edge partners, parallel to the adjacency ids. The candidate buffer is dead
// after narrowphase until the next broadphase and holds four words per candidate,
// at least the 2·(manifolds + joints) adjacency edges. CSR scatter writes each
// edge's partner once; sort permutes it with the ids; coloring then reads one
// word per edge instead of deriving manifold → colliders → bodies every pass.
BIND(10) RWStructuredBuffer<uint> gEdgePartner;
uint csrPartner(uint body, uint e) { return gEdgePartner[e]; }
#else
uint csrPartner(uint body, uint e)
{
    uint2 ends = constraintEndpoints(ALOAD(gConstraintIds[e]));
    return ends.x == body ? ends.y : ends.x;
}
#endif

void csrCountEndpoint(uint body)
{
    if (!bodyIsDynamic(gBodies[body])) return;
    AADD2(CINDEX(body), 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_clear(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount + 1u;
    if (i < n) ASTORE(CINDEX(i), 0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_degree_contacts(uint3 tid : SV_DispatchThreadID)
{
    // Name kept for host pipeline map; degrees manifolds (one edge per pair).
    uint i = dispatchIndex(tid);
    uint n = ALOAD(gCounters[0].manifoldCount);
    if (i >= n) return;
#if defined(ENGINE_PERSISTENT_CONTACTS)
    // Persistent hosts pack solver pairs before the graph (8el): read the two
    // body ids from the 32-byte pair instead of manifold -> two colliders.
    SolverPair sp = gSolverPairs[i];
    if ((sp.countAndSphere & SOLVER_COUNT_MASK) == 0u) return;
    uint2 e = uint2(sp.bodyA, sp.bodyB);
#else
    Manifold m = gManifolds[i];
    if (m.contactCount == 0u) return;
    uint2 e = constraintEndpoints(gCounters[0].jointCount + i);
#endif
    csrCountEndpoint(e.x);
    csrCountEndpoint(e.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_degree_joints(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].jointCount;
    if (i >= n) return;
    if ((gJoints[i].flags & JOINT_FLAG_BROKEN) != 0u) return;
    uint2 e = constraintEndpoints(i);
    csrCountEndpoint(e.x);
    csrCountEndpoint(e.y);
}

// Three-pass blocked scan. Each 64-thread workgroup scans 256 bodies; the
// adjacency-list storage is scratch until scatter, so it holds block offsets.
// All neighbours are read before any in-place publication within a workgroup.
groupshared uint csrScanShared[64];

uint csrScanThreadPrefix(uint local, uint value)
{
    csrScanShared[local] = value;
    GROUP_SYNC();
    for (uint offset = 1u; offset < 64u; offset <<= 1u)
    {
        uint addend = local >= offset ? csrScanShared[local - offset] : 0u;
        GROUP_SYNC();
        csrScanShared[local] += addend;
        GROUP_SYNC();
    }
    return csrScanShared[local] - value;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_blocks(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint thread = dispatchIndex(tid);
    uint base = thread * 4u;
    uint n = gCounters[0].bodyCount;
    uint4 d = uint4(0u, 0u, 0u, 0u);
    if (base < n) d.x = ALOAD(CINDEX(base));
    if (base + 1u < n) d.y = ALOAD(CINDEX(base + 1u));
    if (base + 2u < n) d.z = ALOAD(CINDEX(base + 2u));
    if (base + 3u < n) d.w = ALOAD(CINDEX(base + 3u));
    uint sum = d.x + d.y + d.z + d.w;
    uint prefix = csrScanThreadPrefix(lid.x, sum);
    if (base < n) ASTORE(CINDEX(base), prefix);
    if (base + 1u < n) ASTORE(CINDEX(base + 1u), prefix + d.x);
    if (base + 2u < n) ASTORE(CINDEX(base + 2u), prefix + d.x + d.y);
    if (base + 3u < n) ASTORE(CINDEX(base + 3u), prefix + d.x + d.y + d.z);
    if (lid.x == 63u) ASTORE(gConstraintIds[thread / 64u], prefix + sum);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_block_sums(uint3 lid : SV_GroupThreadID)
{
    uint n = gCounters[0].bodyCount;
    uint blocks = (n + 255u) / 256u;
    uint perLane = (blocks + 63u) / 64u;
    uint start = lid.x * perLane;
    uint end = min(start + perLane, blocks);
    uint sum = 0u;
    for (uint b = start; b < end; b++) sum += ALOAD(gConstraintIds[b]);
    uint prefix = csrScanThreadPrefix(lid.x, sum);
    if (lid.x == 63u) ASTORE(CINDEX(n), prefix + sum);
    // Each lane owns a disjoint interval; no lane reads another lane's stores.
    for (uint b = start; b < end; b++)
    {
        uint degree = ALOAD(gConstraintIds[b]);
        ASTORE(gConstraintIds[b], prefix);
        prefix += degree;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_add(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= gCounters[0].bodyCount) return;
    uint prefix = ALOAD(CINDEX(i)) + ALOAD(gConstraintIds[i / 256u]);
    ASTORE(CINDEX(i), prefix);
    ASTORE(gBodyColor[i], prefix);
}

// Parallel exclusive prefix (Hillis–Steele). Host binds gPassUniform = 1,2,4,... via
// pass_table (no mid-encoder CPU scratch writes). gBodyColor holds the running sum.
[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_init(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (i >= n) return;
    ASTORE(gBodyColor[i], ALOAD(CINDEX(i))); // degree
}

// Ping-pong Hillis–Steele step. Host packs pass_table[48+log] = offset | (ping<<31)
// (0⇒read bodyColor/write constraintIndex, 1⇒opposite). Fuses old scan+swap
// (~14 fewer launches on Ring 9k).
[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_scan(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (i >= n) return;
    uint packed = PASS_U;
    uint offset = packed & 0x7fffffffu;
    uint ping = packed >> 31;
    uint v;
    if (ping == 0u)
    {
        v = ALOAD(gBodyColor[i]);
        if (i >= offset) v += ALOAD(gBodyColor[i - offset]);
        ASTORE(CINDEX(i), v);
    }
    else
    {
        v = ALOAD(CINDEX(i));
        if (i >= offset) v += ALOAD(CINDEX(i - offset));
        ASTORE(gBodyColor[i], v);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_swap(uint3 tid : SV_DispatchThreadID)
{
    // Finalize wrote exclusive offsets to the buffer opposite the inclusive
    // scan. Publish them to both adjacency starts and scatter cursors only
    // after that dispatch completes; no thread now reads a neighbour we write.
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (i >= n) return;
    if (HOST_S(12) == 0u)
        ASTORE(CINDEX(i), ALOAD(gBodyColor[i]));
    else
        ASTORE(gBodyColor[i], ALOAD(CINDEX(i)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_prefix_finalize(uint3 tid : SV_DispatchThreadID)
{
    // Inclusive scan in constraintIndex if last ping was 0, else bodyColor.
    // Host sets gScratchAtomics[12] = last ping (write_scratch).
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].bodyCount;
    if (i > n) return;
    uint ping = HOST_S(12);
    if (i == n)
    {
        uint last = 0u;
        if (n > 0u)
            last = (ping == 0u) ? ALOAD(CINDEX(n - 1u)) : ALOAD(gBodyColor[n - 1u]);
        ASTORE(CINDEX(n), last);
        return;
    }
    uint excl = (i == 0u) ? 0u : ((ping == 0u) ? ALOAD(CINDEX(i - 1u)) : ALOAD(gBodyColor[i - 1u]));
    // Never shift the inclusive scan in place: another workgroup may still
    // need index i as its predecessor. Use the other ping-pong buffer, then
    // csr_prefix_swap copies the result in a separate dispatch.
    if (ping == 0u) ASTORE(gBodyColor[i], excl);
    else ASTORE(CINDEX(i), excl);
}

// Back-compat single-thread path for tiny worlds (tests).
[shader("compute")]
[numthreads(1, 1, 1)]
void csr_prefix(uint3 tid : SV_DispatchThreadID)
{
    if (dispatchIndex(tid) != 0) return;
    uint n = gCounters[0].bodyCount;
    uint sum = 0;
    for (uint i = 0; i < n; i++)
    {
        uint d = ALOAD(CINDEX(i));
        ASTORE(CINDEX(i), sum);
        ASTORE(gBodyColor[i], sum);
        sum += d;
    }
    ASTORE(CINDEX(n), sum);
}

void csrFillEndpoint(uint body, uint partner, uint id)
{
    if (!bodyIsDynamic(gBodies[body])) return;
    uint slot;
    AADD3(gBodyColor[body], 1, slot);
    ASTORE(gConstraintIds[slot], id);
#if defined(ENGINE_WGSL)
    if (slot < HOST_S(10) * 4u) gEdgePartner[slot] = partner;
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_scatter_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = ALOAD(gCounters[0].manifoldCount);
    if (i >= n) return;
    uint id = gCounters[0].jointCount + i;
#if defined(ENGINE_PERSISTENT_CONTACTS)
    SolverPair sp = gSolverPairs[i];
    if ((sp.countAndSphere & SOLVER_COUNT_MASK) == 0u) return;
    uint2 e = uint2(sp.bodyA, sp.bodyB);
#else
    Manifold m = gManifolds[i];
    if (m.contactCount == 0u) return;
    uint2 e = constraintEndpoints(id);
#endif
    csrFillEndpoint(e.x, e.y, id);
    csrFillEndpoint(e.y, e.x, id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_scatter_joints(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    uint n = gCounters[0].jointCount;
    if (i >= n) return;
    if ((gJoints[i].flags & JOINT_FLAG_BROKEN) != 0u) return;
    uint2 e = constraintEndpoints(i);
    csrFillEndpoint(e.x, e.y, i);
    csrFillEndpoint(e.y, e.x, i);
}

// Diagnostic control: old append-ID ordering. Atomic emission changes these IDs
// between runs, so this ordering is not a physical/deterministic summation order.
[shader("compute")]
[numthreads(64, 1, 1)]
void csr_sort_append(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    uint awakeN = ALOAD(gCounters[0].awakeCount);
    if (li >= awakeN) return;
    uint b = gAwakeList[li];
    uint lo = ALOAD(CINDEX(b));
    uint hi = ALOAD(CINDEX(b + 1u));
    uint nJ = gCounters[0].jointCount;
    for (uint i = lo + 1u; i < hi; i++)
    {
        uint id = ALOAD(gConstraintIds[i]);
        uint key0 = id < nJ ? 0u : 1u;
        uint key1 = id < nJ ? id : (id - nJ);
        uint j = i;
        while (j > lo)
        {
            uint prev = ALOAD(gConstraintIds[j - 1u]);
            uint pk0 = prev < nJ ? 0u : 1u;
            uint pk1 = prev < nJ ? prev : (prev - nJ);
            if (key0 > pk0 || (key0 == pk0 && key1 >= pk1)) break;
            ASTORE(gConstraintIds[j], prev);
            j--;
        }
        ASTORE(gConstraintIds[j], id);
    }
    if (HOST_S(26) != 0u)
        gColorLists[HOST_S(1) * (HOST_S(4) + 1u) + b] = compactPreviousColor(b);
}

// Joints have stable identities. Contacts are ordered by the neighboring body,
// then the unordered collider pair (compound bodies can have multiple colliders).
// Never use contactStart or manifold append ID as a physical key.
uint4 csrPhysicalKey(uint body, uint id)
{
    uint nJ = gCounters[0].jointCount;
    if (id < nJ) return uint4(0u, id, 0u, 0u);
    Manifold mf = gManifolds[id - nJ];
    uint a = gColliders[mf.colliderA].body;
    uint b = gColliders[mf.colliderB].body;
    return uint4(1u, a == body ? b : a,
        min(mf.colliderA, mf.colliderB), max(mf.colliderA, mf.colliderB));
}

bool csrKeyLess(uint4 a, uint4 b)
{
    return a.x < b.x || (a.x == b.x && (a.y < b.y ||
        (a.y == b.y && (a.z < b.z || (a.z == b.z && a.w < b.w)))));
}

// Same order as csrKeyLess(csrPhysicalKey(...)) but reads the cached partner:
// joints (id order) before manifolds, manifolds by partner body; only edges
// sharing a partner load collider ids for the tie-break.
bool csrEdgeLess(uint body, uint idA, uint partnerA, uint idB, uint partnerB)
{
    uint nJ = gCounters[0].jointCount;
    bool jointA = idA < nJ, jointB = idB < nJ;
    if (jointA != jointB) return jointA;
    if (jointA) return idA < idB;
    if (partnerA != partnerB) return partnerA < partnerB;
    return csrKeyLess(csrPhysicalKey(body, idA), csrPhysicalKey(body, idB));
}

// Max-heap sift. A large impactor may have thousands of incident manifolds;
// unlike insertion sort, heap construction/extraction cannot become quadratic.
void csrSiftDown(uint body, uint lo, uint count, uint root)
{
    uint id = ALOAD(gConstraintIds[lo + root]);
    uint partner = csrPartner(body, lo + root);
    for (uint depth = 0u; depth < 32u; depth++)
    {
        if (root >= count / 2u) break; // includes count 0/1, avoids child overflow
        uint child = root * 2u + 1u;
        uint childId = ALOAD(gConstraintIds[lo + child]);
        uint childPartner = csrPartner(body, lo + child);
        if (child + 1u < count)
        {
            uint rightId = ALOAD(gConstraintIds[lo + child + 1u]);
            uint rightPartner = csrPartner(body, lo + child + 1u);
            if (csrEdgeLess(body, childId, childPartner, rightId, rightPartner))
            {
                child++; childId = rightId; childPartner = rightPartner;
            }
        }
        if (!csrEdgeLess(body, id, partner, childId, childPartner)) break;
        ASTORE(gConstraintIds[lo + root], childId);
        csrSetPartner(lo + root, childPartner);
        root = child;
    }
    ASTORE(gConstraintIds[lo + root], id);
    csrSetPartner(lo + root, partner);
}

void csrSetPartner(uint e, uint partner)
{
#if defined(ENGINE_WGSL)
    gEdgePartner[e] = partner;
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csr_sort(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= ALOAD(gCounters[0].awakeCount)) return;
    uint body = gAwakeList[li];
    uint lo = ALOAD(CINDEX(body));
    uint hi = ALOAD(CINDEX(body + 1u));
    uint count = hi - lo;
    if (count <= 32u)
    {
        // Most bricks have a short list; insertion avoids heap setup there.
        for (uint i = lo + 1u; i < hi; i++)
        {
            uint id = ALOAD(gConstraintIds[i]);
            uint partner = csrPartner(body, i);
            uint j = i;
            while (j > lo)
            {
                uint prev = ALOAD(gConstraintIds[j - 1u]);
                uint prevPartner = csrPartner(body, j - 1u);
                if (!csrEdgeLess(body, id, partner, prev, prevPartner)) break;
                ASTORE(gConstraintIds[j], prev);
                csrSetPartner(j, prevPartner);
                j--;
            }
            ASTORE(gConstraintIds[j], id);
            csrSetPartner(j, partner);
        }
    }
    else
    {
        // The WGSL safety pass caps each loop at 65,536 iterations. Chunk both
        // sweeps so a high-degree body cannot silently leave an unsorted tail.
        uint parent = count / 2u;
        for (uint batch = 0u; batch < (count + 32767u) / 32768u; batch++)
        {
            uint work = min(parent, 32768u);
            for (uint k = 0u; k < work; k++)
            {
                parent--;
                csrSiftDown(body, lo, count, parent);
            }
            if (parent == 0u) break;
        }
        uint end = count - 1u;
        for (uint batch = 0u; batch < (count + 32767u) / 32768u; batch++)
        {
            uint work = min(end, 32768u);
            for (uint k = 0u; k < work; k++)
            {
                uint largest = ALOAD(gConstraintIds[lo]);
                uint last = ALOAD(gConstraintIds[lo + end]);
                uint largestPartner = csrPartner(body, lo);
                uint lastPartner = csrPartner(body, lo + end);
                ASTORE(gConstraintIds[lo], last);
                ASTORE(gConstraintIds[lo + end], largest);
                csrSetPartner(lo, lastPartner);
                csrSetPartner(lo + end, largestPartner);
                csrSiftDown(body, lo, end, 0u);
                end--;
            }
            if (end == 0u) break;
        }
    }
    if (HOST_S(26) != 0u)
        gColorLists[HOST_S(1) * (HOST_S(4) + 1u) + body] = compactPreviousColor(body);
}

// three-avbd `priority`: hash32(i + 0x9e3779b9) — independent of round / height.
uint jpPriority(uint body)
{
    uint x = body + 0x9e3779b9u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Like the reference's colorCompact: only local priority maxima may reduce
// their retained color, so no adjacent bodies compact simultaneously. Write a
// separate snapshot row; initialization must not observe partial updates.
uint compactPreviousColor(uint body)
{
    uint history = HOST_S(1) * HOST_S(4);
    uint previous = HOST_S(17) == 0u ? 0u : gColorLists[history + body];
    if (previous == 0u || previous > HOST_S(4)) return 0u;
    uint prio = jpPriority(body);
    uint used = 0u;
    uint lo = ALOAD(CINDEX(body));
    uint hi = ALOAD(CINDEX(body + 1u));
    for (uint e = lo; e < hi; e++)
    {
        uint other = csrPartner(body, e);
        if (!bodyIsDynamic(gBodies[other]) || bodyIsSleeping(gBodies[other])) continue;
        uint op = jpPriority(other);
        if (op > prio || (op == prio && other > body)) return previous;
        uint encoded = gColorLists[history + other];
        if (encoded > 0u && encoded <= 32u) used |= 1u << (encoded - 1u);
    }
    uint free = 0u;
    uint bits = ~used;
    while (free < 32u && (bits & 1u) == 0u) { bits >>= 1u; free++; }
    return min(previous, free + 1u);
}

// One Jones-Plassmann attempt for an uncolored body: false while a
// higher-priority uncolored neighbour blocks it.
bool jpTryColor(uint body)
{
    uint prio = jpPriority(body);
    bool blocked = false;
    uint usedLo = 0u;
    uint usedHi = 0u;
    uint lo = ALOAD(CINDEX(body));
    uint hi = ALOAD(CINDEX(body + 1u));
    for (uint e = lo; e < hi; e++)
    {
        uint other = csrPartner(body, e);
        if (!bodyIsDynamic(gBodies[other]) || bodyIsSleeping(gBodies[other])) continue;
        uint oc = ALOAD(gBodyColor[other]);
        if (oc == 0xFFFFFFFFu)
        {
            uint op = jpPriority(other);
            if (op > prio || (op == prio && other > body)) blocked = true;
        }
        else if (oc < 32u) usedLo |= 1u << oc;
        else if (oc < 64u) usedHi |= 1u << (oc - 32u);
    }
    if (blocked) return false;

    // Smallest colour not used by already-coloured neighbours (three-avbd smallestFree).
    uint free = 63u;
    if (~usedLo != 0u)
    {
        free = 0u;
        uint bit = ~usedLo;
        while ((bit & 1u) == 0u) { bit >>= 1u; free++; }
    }
    else if (~usedHi != 0u)
    {
        free = 32u;
        uint bit = ~usedHi;
        while ((bit & 1u) == 0u) { bit >>= 1u; free++; }
    }
    ASTORE(gBodyColor[body], free);
    AMAX2(gCounters[0].colorCount, free + 1);
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void color_bodies(uint3 tid : SV_DispatchThreadID)
{
    // Jones-Plassmann: winners take the *smallest free* colour (three-avbd), not the
    // round index — assigning round forced O(rounds) colours and killed parallelism.
    uint li = dispatchIndex(tid);
    bool incremental = HOST_S(26) != 0u;
    uint workN = incremental ? ALOAD(gCounters[0].recolorCount) : ALOAD(gCounters[0].awakeCount);
    if (li >= workN) return;
    uint body = incremental ? gCcdFlags[li] : gAwakeList[li];
    if (ALOAD(gBodyColor[body]) != 0xFFFFFFFFu) return;
    // PASS_U is the round index: a body still blocked records the round, so
    // the host can size the next step's rounds and the tail knows whether the
    // dispatched rounds finished.
    if (!jpTryColor(body)) AMAX2(gCounters[0].colorRounds, PASS_U + 1u);
}

// Rounds past the host's estimate (PASS_U = rounds dispatched): one workgroup
// keeps running Jones-Plassmann rounds over the same work list while the last
// round still left a body blocked, up to gCounters.colorRoundLimit rounds in
// all (the global rounds' limit), with a storage barrier between rounds.
[shader("compute")]
[numthreads(256, 1, 1)]
void color_bodies_tail(uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    // Publish the round limit (0: the dispatched rounds finished) through
    // workgroup memory so the loop and its barriers stay in uniform control flow.
    if (local == 0u) gsUniformLanes = ALOAD(gCounters[0].colorRounds) >= PASS_U ? ALOAD(gCounters[0].colorRoundLimit) : 0u;
    GROUP_SYNC();
    uint limit = gsUniformLanes;
    if (limit == 0u) return;
    GROUP_SYNC();
    bool incremental = HOST_S(26) != 0u;
    uint workN = incremental ? ALOAD(gCounters[0].recolorCount) : ALOAD(gCounters[0].awakeCount);
    for (uint round = PASS_U; round < limit; round++)
    {
        if (local == 0u) ASTORE(gsColorRemaining, 0u);
        GROUP_SYNC();
        for (uint li = local; li < workN; li += 256u)
        {
            uint body = incremental ? gCcdFlags[li] : gAwakeList[li];
            if (ALOAD(gBodyColor[body]) != 0xFFFFFFFFu) continue;
            if (!jpTryColor(body)) { AOR2(gsColorRemaining, 1u); AMAX2(gCounters[0].colorRounds, round + 1u); }
        }
        DeviceMemoryBarrierWithGroupSync();
        if (local == 0u) gsUniformLanes = ALOAD(gsColorRemaining);
        GROUP_SYNC();
        if (gsUniformLanes == 0u) break;
    }
}

// Opt-in local repair (8en, HOST_S(27) != 0): a small incremental repair list
// is colored by one workgroup looping JP rounds behind storage barriers, and
// write_work_args zeroes slot 69 so the 32 global rounds launch nothing. The
// highest-priority uncolored body always colors, and every uncolored neighbour
// is itself on the list, so each round progresses.
static const uint LOCAL_COLOR_LIMIT = 4096u;
bool localColoring()
{
#if defined(ENGINE_WGSL)
    // WGSL hosts only: Metal's HOST_S(27) is a large-body list word.
    return HOST_S(27) != 0u && HOST_S(26) != 0u &&
        ALOAD(gCounters[0].recolorCount) <= LOCAL_COLOR_LIMIT;
#else
    return false;
#endif
}

groupshared AUint gsColorRemaining;

[shader("compute")]
[numthreads(256, 1, 1)]
void color_bodies_local(uint3 lid : SV_GroupThreadID)
{
    uint local = lid.x;
    if (local == 0u) gsUniformLanes = localColoring() ? 1u : 0u;
    GROUP_SYNC();
    if (gsUniformLanes == 0u) return;
    uint workN = ALOAD(gCounters[0].recolorCount);
    for (uint round = 0u; round < LOCAL_COLOR_LIMIT; round++)
    {
        if (local == 0u) ASTORE(gsColorRemaining, 0u);
        GROUP_SYNC();
        for (uint li = local; li < workN; li += 256u)
        {
            uint body = gCcdFlags[li];
            if (ALOAD(gBodyColor[body]) != 0xFFFFFFFFu) continue;
            if (!jpTryColor(body)) AOR2(gsColorRemaining, 1u);
        }
        DeviceMemoryBarrierWithGroupSync();
        if (local == 0u) gsUniformLanes = ALOAD(gsColorRemaining);
        GROUP_SYNC();
        if (gsUniformLanes == 0u) break;
    }
}

// Adaptive 1/2/4/8 threads per body by colour size. This tuning differs from
// pinned three-avbd, whose one-lane threshold is 2^31 rather than 65536.
uint primalLanesFor(uint bodies)
{
    // Never one lane per body: gables' ~184k-body colors measured 1/2/4/8 lanes
    // at 32.2/31.1/30.9/31.4 ms (decision 8eg); the ring is neutral.
    if (bodies == 0u) return 1u;
    // Convex scenes (HOST_S(29)): hull bodies walk several manifolds each, and
    // 8 lanes hide that latency (49-house grid peak: 4/8/16 lanes measured
    // 11.3/10.0/11.7 ms); box scenes keep their validated 4.
    if (bodies >= 4096u) return HOST_S(29) != 0u ? 8u : 4u;
    return 8u;
}

// Degree-aware lanes (opt-in, HOST_S(29) != 0; host sizes gColorCounts to 128
// words): gColorCounts[64 + c] holds colour c's largest CSR degree. A compound
// body with hundreds of manifolds otherwise holds its whole colour dispatch
// while 8 lanes walk its edges serially. Widen to 16/32/64 lanes when the
// colour is small enough that the extra threads stay under 64k.
static const uint COLOR_DEGREE_BASE = 64u;
uint primalLanesForDegree(uint bodies, uint maxDegree)
{
    uint lanes = primalLanesFor(bodies);
    if (HOST_S(29) == 0u || bodies == 0u) return lanes;
    uint want = lanes;
    while (want < 64u && maxDegree > want * 8u) want *= 2u;
    while (want > lanes && bodies * want > 65536u) want /= 2u;
    return max(lanes, want);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void record_color_degrees(uint3 tid : SV_DispatchThreadID)
{
    uint li = dispatchIndex(tid);
    if (li >= ALOAD(gCounters[0].awakeCount)) return;
    uint body = gAwakeList[li];
    uint color = ALOAD(gBodyColor[body]);
    if (color >= min(HOST_S(4), COLOR_DEGREE_BASE)) return;
    uint degree = ALOAD(CINDEX(body + 1u)) - ALOAD(CINDEX(body));
    AMAX2(gColorCounts[COLOR_DEGREE_BASE + color], degree);
}

// Slots 0..63 are colour dispatches. Slots 64..67 are the *current* step's
// candidate, contact/joint, manifold and awake counts; slot 68 is previous contacts.
// Hosts opting into incremental coloring (HOST_S(26)) also allocate slot 69.
// CPU estimates from a prior
// frame can miss most of the first ring/gables contacts and leave them unsolved.
// Indirect args of a (x, y) grid of workgroups, each side within the 32,768 groups a
// dimension may hold (broadphase_sleeper_large_walk: chunks of 64 cells by large dynamic).
void writeGroupGrid(uint slot, uint x, uint y)
{
    gIndirectArgs[slot * 3u] = min(x, 32768u);
    gIndirectArgs[slot * 3u + 1u] = min(y, 32768u);
    gIndirectArgs[slot * 3u + 2u] = 1u;
}

void writeDispatchArgs(uint slot, uint threads)
{
    uint groups = (threads + 63u) / 64u;
    gIndirectArgs[slot * 3u] = min(groups, 32768u);
    gIndirectArgs[slot * 3u + 1u] = max(1u, (groups + 32767u) / 32768u);
    gIndirectArgs[slot * 3u + 2u] = 1u;
}

// Re-emit last step's resting pairs as candidates (8ep). Runs after the
// broadphase and before narrowphase overwrites gManifolds; dispatched over
// work slot 72 (HOST_S(13) = previous GPU manifold count). Narrowphase reuses
// them through the at-rest path, which copies the contacts unchanged.
[shader("compute")]
[numthreads(64, 1, 1)]
void carry_rest_pairs(uint3 tid : SV_DispatchThreadID)
{
    if (!carryRestPairs() || HOST_S(17) == 0u) return;
    uint m = dispatchIndex(tid);
    if (m >= min(HOST_S(13), HOST_S(10))) return;
    Manifold mf = gManifolds[m];
    if (mf.contactCount == 0u) return;
    BodyState ba = gBodies[gColliders[mf.colliderA].body];
    BodyState bb = gBodies[gColliders[mf.colliderB].body];
    if (!bodyIsDynamic(ba) && !bodyIsDynamic(bb)) return;
    if (!contactPairAtRest(ba, bb)) return;
    emitCandidate(mf.colliderA, mf.colliderB);
}

