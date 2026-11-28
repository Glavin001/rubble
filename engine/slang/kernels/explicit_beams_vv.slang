// --- Vehicle–vehicle contact (engine/docs/explicit-beams-vehicle-contact-spec.md,
// CPU reference engine_softbody::explicit_contact) ---
//
// A frame with vehicle contact on (more than one vehicle):
//   explicit_beams_vv_bounds  one workgroup per vehicle: its box over the node
//                             world positions; its frame bounds (the box grown by
//                             the travel of its fastest node over the steps the plan
//                             covers, twice, plus a margin) and the padded frame box
//                             (+ 0.05 m);
//                             the box into a hashed vehicle grid (or the big list)
//   explicit_beams_vv_pairs   one thread per vehicle: pairs of overlapping padded
//                             frame boxes
//   explicit_beams_vv_group   one workgroup: connected components of the pairs
//                             (label propagation; a vehicle without partners a
//                             group of one), the groups'
//                             member lists, the indirect dispatch arguments
//   (the plan passes run when the last plan's steps are used up: it covers at
//   least VV_PLAN_STEPS on the host, so at 120 fps every other frame)
//   explicit_beams_frame      (indirect) the solo vehicles, a whole frame each; a
//                             node leaving its frame bounds is reported (loud)
//   explicit_beams_contact    (indirect) one workgroup per group: every step, each
//                             member's step in vehicle order, then the contact pass
//                             (owners in vehicle order)
// Two vehicles whose padded frame boxes do not overlap, with every node inside
// its frame bounds, cannot be partners in any step of the frame: the solo
// vehicles need no contact pass. A node outside its bounds is reported to the
// host, which fails (the frame may have missed a contact).

static const int VV_INT_MAX = 2147483647;
static const int VV_INT_MIN = -2147483647 - 1;
static const float VV_BOX_MARGIN = 0.05;
static const uint VV_SLOTS = 8u;
static const uint VV_MAX_SPAN = 3u;
static const uint VV_MAX_CELLS = 27u;
static const float VV_APPROACH_SQ = 16.0;
// Members a contact group may have (one workgroup steps them one after another);
// a larger group's vehicles run alone that frame and the host fails.
static const uint VV_MAX_GROUP = 64u;
// Groups of at most VV_PART members step side by side (contactMembersStep).
static const uint VV_PART = 8u;
// Members a group steps side by side on one workgroup, and the one-workgroup path's
// small-group pass: at most VV_SIDE. The split group path's pass (its tests across
// workgroups, its levels) takes groups up to VV_PART, whose members past VV_SIDE each
// step on every lane of their own workgroup (contactMemberStep).
static const uint VV_SIDE = 4u;
static const int VV_MAX_WAIT = 12;

[[vk::binding(10, 0)]] RWStructuredBuffer<uint> gArgs;

groupshared Atomic<int> gsVvBox[12];
groupshared uint gsVvA[256];
groupshared uint gsVvB[256];
groupshared Atomic<uint> gsVvOffered[2];
groupshared Atomic<uint> gsVvListed[2];
groupshared Atomic<uint> gsVvCands[2];
groupshared Atomic<uint> gsVvHits[2];
groupshared uint gsVvReset[2];

// Box coordinates as integers ordered like the floats (atomic minimum, maximum).
int vvKey(float f)
{
    int i = asint(f);
    return i >= 0 ? i : (i ^ 0x7FFFFFFF);
}
float vvUnkey(int k) { return asfloat(k >= 0 ? k : (k ^ 0x7FFFFFFF)); }

uint vvBoxKey(uint v, uint parity) { return gP.vvAcc + (v * 2u + parity) * 6u; }
uint vvCpOff(uint v) { return gStatic[gP.vvStatic + v].x; }
uint vvCpCount(uint v) { return gStatic[gP.vvStatic + v].y; }

bool vvOverlap(float3 alo, float3 ahi, float3 blo, float3 bhi)
{
    return !(any(ahi < blo) || any(bhi < alo));
}

uint vvHash(int3 c)
{
    return (uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u) & gP.vvGridMask;
}

// The vehicle grid cells of a padded frame box: false when it spans too many (or
// is not finite).
bool vvCells(float3 lo, float3 hi, out int3 c0, out int3 span)
{
    float cell = asfloat(gP.vvCell);
    float3 f0 = floor(lo / cell);
    float3 fs = floor(hi / cell) - f0 + float3(1.0, 1.0, 1.0);
    c0 = int3(0, 0, 0);
    span = int3(0, 0, 0);
    if (!(all(fs <= float(VV_MAX_SPAN)) && all(fs >= 1.0) && fs.x * fs.y * fs.z <= float(VV_MAX_CELLS)))
        return false;
    c0 = int3(f0);
    span = int3(fs);
    return true;
}

// vvBounds' boxes, cells and flags for vehicle v with its own margin (wider than the
// fleet's): the same arithmetic with `margin` from the vehicle's record.
void vvBoundsRest(uint v, float3 blo, float3 bhi, float3 vl, float3 vh, float time, float margin)
{
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    float3 alo = blo - (-vl * time + float3(margin, margin, margin));
    float3 ahi = bhi + (vh * time + float3(margin, margin, margin));
    float3 plo = alo - m3;
    float3 phi = ahi + m3;
    gMisc[vvRec(v, VV_P_LO)] = uint4(asuint(plo), 0u);
    gMisc[vvRec(v, VV_P_HI)] = uint4(asuint(phi), 0u);
    gMisc[vvRec(v, VV_ALLOWED_LO)] = uint4(asuint(alo), 0u);
    gMisc[vvRec(v, VV_ALLOWED_HI)] = uint4(asuint(ahi), 0u);
    int3 c0;
    int3 span;
    // A box that is not finite (a node at infinity or NaN) partners no one
    // here (a GPU deviation: the specification's rule would partner it with
    // every vehicle and group the whole fleet).
    bool finite = all(abs(plo) < 3.0e38) && all(abs(phi) < 3.0e38);
    bool big = finite && !vvCells(plo, phi, c0, span);
    if (finite && !big)
    {
        for (int x = 0; x < min(span.x, int(VV_MAX_SPAN)); x++)
            for (int y = 0; y < min(span.y, int(VV_MAX_SPAN)); y++)
                for (int z = 0; z < min(span.z, int(VV_MAX_SPAN)); z++)
                {
                    uint b = vvHash(c0 + int3(x, y, z));
                    uint slot = uint(gAcc[gP.vvGridCount + b].add(1));
                    if (slot < VV_SLOTS) gNext[gP.vvGrid + b * VV_SLOTS + slot] = v;
                    else big = true;
                }
    }
    if (big)
    {
        uint at = uint(gAcc[vvCtl(VV_BIGS)].add(1));
        gNext[gP.vvBig + at] = v;
    }
    uint4 fl = gMisc[vvRec(v, VV_FLAGS)];
    gMisc[vvRec(v, VV_FLAGS)] = uint4(finite ? (big ? 1u : 0u) : 2u, 0u, 0u, fl.w);
}

[shader("compute")]
float contactOffMax(uint v);
[numthreads(256, 1, 1)]
void explicit_beams_vv_bounds(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint v = gid.x;
    uint lane = lid.x;
    uint4 r0 = gStatic[v * 4u];
    float3 origin = asfloat(gMisc[v].xyz);
    // Keys: position lo xyz, hi xyz; velocity lo xyz, hi xyz.
    if (lane < 12u) gsVvBox[lane].store((lane % 6u) < 3u ? VV_INT_MAX : VV_INT_MIN);
    sync();
    float3 lo = float3(3.0e38, 3.0e38, 3.0e38);
    float3 hi = -lo;
    float3 vlo = float3(0.0, 0.0, 0.0);
    float3 vhi = float3(0.0, 0.0, 0.0);
    bool has = false;
    for (uint i = lane, iStop = EB_END(115, i, r0.y, 256u); i < iStop; i += 256u)
    {
        uint gi = r0.x + i;
        float3 x = nf3(origin + gNodes[gi * 3u].xyz);
        float3 vel = gNodes[gi * 3u + 1u].xyz;
        lo = min(lo, x);
        hi = max(hi, x);
        vlo = min(vlo, vel);
        vhi = max(vhi, vel);
        has = true;
    }
    if (has)
    {
        gsVvBox[0].min(vvKey(lo.x));
        gsVvBox[1].min(vvKey(lo.y));
        gsVvBox[2].min(vvKey(lo.z));
        gsVvBox[3].max(vvKey(hi.x));
        gsVvBox[4].max(vvKey(hi.y));
        gsVvBox[5].max(vvKey(hi.z));
        gsVvBox[6].min(vvKey(vlo.x));
        gsVvBox[7].min(vvKey(vlo.y));
        gsVvBox[8].min(vvKey(vlo.z));
        gsVvBox[9].max(vvKey(vhi.x));
        gsVvBox[10].max(vvKey(vhi.y));
        gsVvBox[11].max(vvKey(vhi.z));
    }
    sync();
    if (lane == 0u)
    {
        float3 blo = float3(vvUnkey(gsVvBox[0].load()), vvUnkey(gsVvBox[1].load()), vvUnkey(gsVvBox[2].load()));
        float3 bhi = float3(vvUnkey(gsVvBox[3].load()), vvUnkey(gsVvBox[4].load()), vvUnkey(gsVvBox[5].load()));
        // Each face moves out by twice the farthest a node moving that way at its
        // velocity covers over the plan, plus the margin (min/max start at 0).
        float3 vl = float3(vvUnkey(gsVvBox[6].load()), vvUnkey(gsVvBox[7].load()), vvUnkey(gsVvBox[8].load()));
        float3 vh = float3(vvUnkey(gsVvBox[9].load()), vvUnkey(gsVvBox[10].load()), vvUnkey(gsVvBox[11].load()));
        // (Contact offsets: the triangles reach past the nodes by up to the largest.)
        float om = contactOffMax(v);
        blo = blo - float3(om, om, om);
        bhi = bhi + float3(om, om, om);
        float time = 2.0 * float(gP.vvPlanSteps) * STEP;
        float margin = asfloat(gP.vvMargin);
        float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
        // The positions the last frame left, against the bounds that held for it
        // (a solo frame checks each step's positions at the next step's start).
        if ((gP.vvOn & 4u) == 0u
            && (any(blo < asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz)) || any(bhi > asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz))))
            vvReportOutside(v);
        // This vehicle's own margin when wider than the fleet's (VV_MEMBER .y: the host
        // widens a vehicle's after it left its frame bounds; 0 until then): the rest in
        // its own function (the fleet's path below keeps its codegen).
        if (asfloat(gMisc[vvRec(v, VV_MEMBER)].y) > margin)
        {
            vvBoundsRest(v, blo, bhi, vl, vh, time, asfloat(gMisc[vvRec(v, VV_MEMBER)].y));
            return ebLoopCapFlush();
        }
        float3 alo = blo - (-vl * time + float3(margin, margin, margin));
        float3 ahi = bhi + (vh * time + float3(margin, margin, margin));
        float3 plo = alo - m3;
        float3 phi = ahi + m3;
        gMisc[vvRec(v, VV_P_LO)] = uint4(asuint(plo), 0u);
        gMisc[vvRec(v, VV_P_HI)] = uint4(asuint(phi), 0u);
        gMisc[vvRec(v, VV_ALLOWED_LO)] = uint4(asuint(alo), 0u);
        gMisc[vvRec(v, VV_ALLOWED_HI)] = uint4(asuint(ahi), 0u);
        int3 c0;
        int3 span;
        // A box that is not finite (a node at infinity or NaN) partners no one
        // here (a GPU deviation: the specification's rule would partner it with
        // every vehicle and group the whole fleet).
        bool finite = all(abs(plo) < 3.0e38) && all(abs(phi) < 3.0e38);
        bool big = finite && !vvCells(plo, phi, c0, span);
        if (finite && !big)
        {
            for (int x = 0; x < min(span.x, int(VV_MAX_SPAN)); x++)
                for (int y = 0; y < min(span.y, int(VV_MAX_SPAN)); y++)
                    for (int z = 0; z < min(span.z, int(VV_MAX_SPAN)); z++)
                    {
                        uint b = vvHash(c0 + int3(x, y, z));
                        uint slot = uint(gAcc[gP.vvGridCount + b].add(1));
                        if (slot < VV_SLOTS) gNext[gP.vvGrid + b * VV_SLOTS + slot] = v;
                        else big = true;
                    }
        }
        if (big)
        {
            uint at = uint(gAcc[vvCtl(VV_BIGS)].add(1));
            gNext[gP.vvBig + at] = v;
        }
        uint4 fl = gMisc[vvRec(v, VV_FLAGS)];
        gMisc[vvRec(v, VV_FLAGS)] = uint4(finite ? (big ? 1u : 0u) : 2u, 0u, 0u, fl.w);
    }
    ebLoopCapFlush();
}

// The frame's dispatch arguments: with no group the solo frame over the vehicles
// (words 0..2), else the frame with the groups over vehicles + groups (words 4..6);
// and every group again for the larger groups' module (words 12..14) when one has
// more than VV_PART members (`big`), else none.
void vvArgs(uint nv, uint groups, uint multi, bool big, bool larger, bool alone)
{
    gArgs[12] = big ? groups : 0u;
    gArgs[13] = 1u;
    gArgs[14] = 1u;
    // The solo module over the vehicles also when one runs alone (its group past
    // VV_MAX_GROUP members: the groups' module holds no solo frame; word 19 keeps it for
    // the awake list's rewrite).
    gArgs[0] = groups == 0u || alone ? nv : 0u;
    gArgs[1] = 1u;
    gArgs[2] = 1u;
    gArgs[19] = alone ? 1u : 0u;
    gArgs[4] = groups == 0u ? 0u : nv + groups;
    gArgs[5] = 1u;
    gArgs[6] = 1u;
    // The plan's groups (vehicle sleep's awake list rewrites the arguments every frame).
    gArgs[3] = groups;
    gArgs[15] = multi + 1u; // its groups of two or more members, plus one (explicit_vv_gate)
    // The split group path (explicit_beams_split.slang): the groups' passes (words
    // 16..18) and, with a group, the members' steps over every vehicle (20..22).
    gArgs[16] = groups;
    gArgs[17] = 1u;
    gArgs[18] = 1u;
    gArgs[20] = groups == 0u ? 0u : nv;
    // (Two rows: the second the members' wheels ahead in the self split's rest dispatch,
    // PERF_SPLIT_WHEELS_AHEAD; every other use of these words returns there at once.)
    gArgs[21] = 2u;
    gArgs[22] = 1u;
    // Its triangle tests across workgroups (explicit_beams_split.slang SPLIT_TEST_WGS a
    // group; words 24..26).
    gArgs[24] = groups * 64u;
    gArgs[25] = 1u;
    gArgs[26] = 1u;
    // Self-collision split in a world with vehicle contact (explicit_beams_split.slang,
    // EB_SELF_SPLIT): the members' steps over every vehicle in every frame, a group or
    // not (words 28..30).
    gArgs[28] = nv;
    gArgs[29] = 1u;
    gArgs[30] = 1u;
    // The contact pass's front as fleet-wide dispatches (explicit_beams_fleet.slang), none
    // without a group: a thread a pair (words 32..34, in workgroups of 256 at least) and
    // one workgroup (36..38: the first plan, which sizes the rest).
    uint any = groups == 0u ? 0u : 1u;
    // (With the kept candidates, PERF_FLEET_KEEP, also a thread a node: its motion since
    // the list's build, fkTrack.)
    uint scanThreads = gNext[flAdjFirst(nv)] / 2u;
    if ((gP.perfFlags & PERF_FLEET_KEEP) != 0u) scanThreads = max(scanThreads, gP.totalNodes);
    gArgs[32] = any * ((scanThreads + 255u) / 256u);
    gArgs[33] = 1u;
    gArgs[34] = 1u;
    gArgs[36] = any;
    gArgs[37] = 1u;
    gArgs[38] = 1u;
    // … and every group again when one is past VV_PART members (`larger`: its pass is
    // the larger groups', in the groups' own dispatch; words 44..46), else none.
    gArgs[44] = larger ? groups : 0u;
    gArgs[45] = 1u;
    gArgs[46] = 1u;
    // The plan's values of words 32 and 44, kept for vehicle sleep's awake list, which
    // rewrites those arguments every frame (explicit_beams_sleep.slang).
    gArgs[7] = gArgs[32];
    gArgs[11] = gArgs[44];
}

void vvEmitPair(uint a, uint b)
{
    uint at = uint(gAcc[vvCtl(VV_PAIRS)].add(1));
    if (at < gP.vvPairCap)
    {
        gNext[gP.vvPairs + 2u * at] = a;
        gNext[gP.vvPairs + 2u * at + 1u] = b;
    }
    else
    {
        gAcc[vvCtl(VV_OVERFLOW)].store(1);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_vv_pairs(uint3 id : SV_DispatchThreadID)
{
    if (id.x < gP.vehicles) vvFindPairs(id.x);
    ebLoopCapFlush();
}

// The overlapping pairs of vehicle v's padded frame box (each pair once).
void vvFindPairs(uint v)
{
    float3 plo = asfloat(gMisc[vvRec(v, VV_P_LO)].xyz);
    float3 phi = asfloat(gMisc[vvRec(v, VV_P_HI)].xyz);
    uint kind = gMisc[vvRec(v, VV_FLAGS)].x;
    if (kind == 2u) return;
    if (kind == 1u)
    {
        // Against every vehicle (two big ones once).
        for (uint w = 0u, wStop = EB_END(116, w, gP.vehicles, 1u); w < wStop; w++)
        {
            if (w == v) continue;
            uint kw = gMisc[vvRec(w, VV_FLAGS)].x;
            if (kw == 2u || (kw == 1u && w < v)) continue;
            if (vvOverlap(plo, phi, asfloat(gMisc[vvRec(w, VV_P_LO)].xyz), asfloat(gMisc[vvRec(w, VV_P_HI)].xyz)))
                vvEmitPair(v, w);
        }
        return;
    }
    int3 c0;
    int3 span;
    vvCells(plo, phi, c0, span);
    for (int x = 0; x < min(span.x, int(VV_MAX_SPAN)); x++)
        for (int y = 0; y < min(span.y, int(VV_MAX_SPAN)); y++)
            for (int z = 0; z < min(span.z, int(VV_MAX_SPAN)); z++)
            {
                int3 c = c0 + int3(x, y, z);
                uint b = vvHash(c);
                uint n = min(uint(gAcc[gP.vvGridCount + b].load()), VV_SLOTS);
                for (uint e = 0u, eStop = EB_END(117, e, n, 1u); e < eStop; e++)
                {
                    uint w = gNext[gP.vvGrid + b * VV_SLOTS + e];
                    // Each pair once: the higher index in the lower's thread, in
                    // the first cell both cover; big vehicles find their own.
                    if (w <= v || gMisc[vvRec(w, VV_FLAGS)].x != 0u) continue;
                    float3 wlo = asfloat(gMisc[vvRec(w, VV_P_LO)].xyz);
                    float3 whi = asfloat(gMisc[vvRec(w, VV_P_HI)].xyz);
                    if (!vvOverlap(plo, phi, wlo, whi)) continue;
                    int3 cw;
                    int3 sw;
                    vvCells(wlo, whi, cw, sw);
                    if (all(c == max(c0, cw))) vvEmitPair(v, w);
                }
            }
}

uint vvLabel(uint v) { return uint(gAcc[gP.vvLabel + v].load()); }

// The plan's overlapping pairs by vehicle (contact-groups-design.md §8.2: partners from
// the broadphase's pair list): per vehicle its entries, one a pair it is in (d = 2 · pair
// + its side), in gNext at the fleet state's adjacency. Every lane of the grouping
// kernel; three barriers. The entries' order within a vehicle follows an atomic counter:
// nothing read from them depends on it (boxes' unions, counts, a slot's number).
void vvAdjBuild(uint lane, uint nv, uint pairs)
{
    for (uint v = lane, vStop = EB_END(118, v, nv, 256u); v < vStop; v += 256u) gAcc[flAdjCount(v)].store(0);
    sync();
    for (uint p = lane, pStop = EB_END(119, p, pairs, 256u); p < pStop; p += 256u)
    {
        gAcc[flAdjCount(gNext[gP.vvPairs + 2u * p])].add(1);
        gAcc[flAdjCount(gNext[gP.vvPairs + 2u * p + 1u])].add(1);
    }
    sync();
    if (lane == 0u)
    {
        uint run = 0u;
        for (uint v = 0u, vStop = EB_END(120, v, nv, 1u); v < vStop; v++)
        {
            gNext[flAdjFirst(v)] = run;
            run += uint(gAcc[flAdjCount(v)].load());
            gAcc[flAdjCount(v)].store(0);
        }
        gNext[flAdjFirst(nv)] = run;
    }
    sync();
    for (uint p = lane, pStop = EB_END(121, p, pairs, 256u); p < pStop; p += 256u)
    {
        uint x = gNext[gP.vvPairs + 2u * p];
        uint y = gNext[gP.vvPairs + 2u * p + 1u];
        gNext[flAdjAt(gNext[flAdjFirst(x)] + uint(gAcc[flAdjCount(x)].add(1)))] = 2u * p;
        gNext[flAdjAt(gNext[flAdjFirst(y)] + uint(gAcc[flAdjCount(y)].add(1)))] = 2u * p + 1u;
    }
    sync();
    // The pair list may hold a pair twice (a vehicle twice in one bucket of the vehicle
    // grid, two of its cells there: vvFindPairs then meets it twice in a cell). The
    // grouping does not mind; the owners' partners must be each once, or a partner's
    // contact points would be candidates twice and its hits applied twice. A later copy
    // of a pair is marked in its bits words (4: flScanPair leaves it, never a partner).
    for (uint p = lane, pStop = EB_END(122, p, pairs, 256u); p < pStop; p += 256u)
    {
        uint x = gNext[gP.vvPairs + 2u * p];
        uint y = gNext[gP.vvPairs + 2u * p + 1u];
        uint dup = 0u;
        uint e1 = gNext[flAdjFirst(x + 1u)];
        for (uint e = gNext[flAdjFirst(x)], eStop = EB_END(123, e, e1, 1u); e < eStop; e++)
        {
            uint d = gNext[flAdjAt(e)];
            if ((d >> 1u) < p && flPartner(d) == y) dup = 4u;
        }
        gNext[flBitsAt(2u * p)] = dup;
        gNext[flBitsAt(2u * p + 1u)] = dup;
    }
}
uint vvSum(uint g, uint k) { return gP.vvLabel + gP.vehicles + g * 3u + k; }

[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_vv_group(uint3 lid : SV_GroupThreadID)
{
    uint lane = lid.x;
    uint nv = gP.vehicles;
    if (gP.vvFusedPairs != 0u)
    {
        for (uint v = lane, vStop = EB_END(124, v, nv, 256u); v < vStop; v += 256u) vvFindPairs(v);
        sync();
    }
    uint pairs = min(uint(gAcc[vvCtl(VV_PAIRS)].load()), gP.vvPairCap);
    // Every vehicle is a member of a group, one without partners a group of its own: the
    // member step is the solo frame's bit for bit, and the split path's vehicles' module
    // holds that step alone (explicit_beams_split_member; kernel-compile-time.md). A group
    // past VV_MAX_GROUP members is reported below; its vehicles run alone in the contact
    // module, and are not stepped that frame on the split group path.
    for (uint v = lane, vStop = EB_END(126, v, nv, 256u); v < vStop; v += 256u)
    {
        gAcc[gP.vvLabel + v].store(int(v));
        gMisc[vvRec(v, VV_FLAGS)].y = 1u;
    }
    sync();
    // The pairs by vehicle (the owners' state by vehicle reads its partners from them).
    if (gP.vvFleet != 0u) vvAdjBuild(lane, nv, pairs);
    // Connected components: every vehicle of a group ends with its smallest index.
    uint go = 1u;
    for (uint it = 0u; it < 64u; it++)
    {
        if (lane == 0u) gsUniformLanes = 0u;
        sync();
        bool changed = false;
        for (uint p = lane, pStop = EB_END(128, p, pairs, 256u); p < pStop; p += 256u)
        {
            uint a = gNext[gP.vvPairs + 2u * p];
            uint b = gNext[gP.vvPairs + 2u * p + 1u];
            uint la = vvLabel(a);
            uint lb = vvLabel(b);
            if (la != lb)
            {
                int m = int(min(la, lb));
                gAcc[gP.vvLabel + a].min(m);
                gAcc[gP.vvLabel + b].min(m);
                gAcc[gP.vvLabel + la].min(m);
                gAcc[gP.vvLabel + lb].min(m);
                changed = true;
            }
        }
        for (uint v = lane, vStop = EB_END(129, v, nv, 256u); v < vStop; v += 256u)
        {
            uint l = vvLabel(v);
            uint ll = vvLabel(l);
            if (ll < l) gAcc[gP.vvLabel + v].min(int(ll));
        }
        if (changed) gsUniformLanes = 1u;
        sync();
        go = gsUniformLanes;
        sync();
        if (go == 0u) break;
    }
    if (lane == 0u && go != 0u) gAcc[vvCtl(VV_UNCONVERGED)].store(1);
    // Groups (numbered by their smallest vehicle, in no particular order) and the
    // solo list.
    for (uint v = lane, vStop = EB_END(130, v, nv, 256u); v < vStop; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u && vvLabel(v) == v)
            gMisc[vvRec(v, VV_FLAGS)].z = uint(gAcc[vvCtl(VV_GROUPS)].add(1));
    }
    sync();
    for (uint v = lane, vStop = EB_END(131, v, nv, 256u); v < vStop; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u)
        {
            uint g = gMisc[vvRec(vvLabel(v), VV_FLAGS)].z;
            gMisc[vvRec(v, VV_FLAGS)].z = g;
            gAcc[vvSum(g, 0u)].add(1);
            gAcc[vvSum(g, 2u)].add(int(vvCpCount(v)));
        }

    }
    sync();
    // Offsets of the groups' members and contact point runs.
    uint groups = uint(gAcc[vvCtl(VV_GROUPS)].load());
    uint chunk = (groups + 255u) / 256u;
    uint g0 = min(lane * chunk, groups);
    uint g1 = min(g0 + chunk, groups);
    uint sm = 0u;
    uint sp = 0u;
    for (uint g = g0, gStop = EB_END(132, g, g1, 1u); g < gStop; g++)
    {
        sm += uint(gAcc[vvSum(g, 0u)].load());
        sp += uint(gAcc[vvSum(g, 2u)].load());
    }
    gsVvA[lane] = sm;
    gsVvB[lane] = sp;
    sync();
    if (lane == 0u)
    {
        uint ra = 0u;
        uint rb = 0u;
        for (uint k = 0u; k < 256u; k++)
        {
            uint a = gsVvA[k];
            uint b = gsVvB[k];
            gsVvA[k] = ra;
            gsVvB[k] = rb;
            ra += a;
            rb += b;
        }
    }
    sync();
    uint ra = gsVvA[lane];
    uint rb = gsVvB[lane];
    for (uint g = g0, gStop = EB_END(133, g, g1, 1u); g < gStop; g++)
    {
        uint m = uint(gAcc[vvSum(g, 0u)].load());
        uint pts = uint(gAcc[vvSum(g, 2u)].load());
        // Too many members: none stepped as a group (they run alone; reported).
        gMisc[gP.vvGroups + g] = m <= VV_MAX_GROUP ? uint4(ra, m, rb, pts) : uint4(ra, 0u, rb, 0u);
        ra += m;
        rb += pts;
    }
    sync();
    for (uint v = lane, vStop = EB_END(134, v, nv, 256u); v < vStop; v += 256u)
    {
        if (gMisc[vvRec(v, VV_FLAGS)].y != 0u)
        {
            uint g = gMisc[vvRec(v, VV_FLAGS)].z;
            if (gMisc[gP.vvGroups + g].y == 0u)
            {
                // Stepped alone (by the solo frame) this frame.
                gMisc[vvRec(v, VV_FLAGS)].y = 0u;
                gAcc[vvCtl(VV_TOO_BIG)].store(1);
                continue;
            }
            uint slot = uint(gAcc[vvSum(g, 1u)].add(1));
            gNext[gP.vvMembers + gMisc[gP.vvGroups + g].x + slot] = v;
        }
    }
    sync();
    if (lane == 0u)
    {
        // A group past VV_PART members (within VV_MAX_GROUP: a larger one's
        // vehicles run alone, its member count 0).
        bool big = false, larger = false, alone = false;
        uint multi = 0u; // groups of two or more members (a count of 0: past VV_MAX_GROUP)
        for (uint g = 0u, gStop = EB_END(135, g, groups, 1u); g < gStop; g++)
        {
            if (gMisc[gP.vvGroups + g].y > VV_SIDE) big = true;
            if (gMisc[gP.vvGroups + g].y > VV_PART) larger = true;
            if (gMisc[gP.vvGroups + g].y != 1u) multi++;
            if (gMisc[gP.vvGroups + g].y == 0u) alone = true;
        }
        vvArgs(nv, groups, multi, big, larger, alone);
    }
    // Clear for the next frame.
    for (uint b = lane, bGuard = 0u; b <= gP.vvGridMask; b += 256u)
    {
        EB_PASS(136, bGuard);
        gAcc[gP.vvGridCount + b].store(0);
    }
    for (uint g = lane, gStop = EB_END(137, g, groups, 256u); g < gStop; g += 256u)
    {
        gAcc[vvSum(g, 0u)].store(0);
        gAcc[vvSum(g, 1u)].store(0);
        gAcc[vvSum(g, 2u)].store(0);
    }
    sync();
    if (lane == 0u)
    {
        gAcc[vvCtl(VV_PAIRS)].store(0);
        gAcc[vvCtl(VV_BIGS)].store(0);
        gAcc[vvCtl(VV_SOLOS)].store(0);
        gAcc[vvCtl(VV_GROUPS)].store(0);
    }
    ebLoopCapFlush();
}

