// --- Edge vehicle contact's pass (the split group path: a dispatch a substep, after the
// groups' contact pass; one workgroup a contact group) ---
// The CPU reference (explicit_contact.rs contact_pass): every owner's turn of its edge
// hits after every owner's node turn, in the members' order, each turn's first pass from
// the forces after the node turns (one snapshot at the pass's start), its residual pass
// and work guard. An owner's hits are found for the group at once (edgeTestsSmall: the
// owner's split counts in its slot), then applied owner by owner.
// Groups of at most VV_PART members; a larger group with an edge-contact pair is reported
// (the contact status' FL_ERR_EDGE_GROUP), never run without its edge hits silently.
static const int FL_ERR_EDGE_GROUP = 6;

void edgePass(uint off, uint k, uint s, uint lane);

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_split_edge(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint lane = lid.x;
    uint4 G = gMisc[gP.vvGroups + gid.x];
    uint off = G.x;
    uint k = splitUniform(lane, edgeOn() ? splitAwake(G) : 0u);
    if (k == 0u) return ebLoopCapFlush();
    uint s = gP.pad2 & SPLIT_STEP;
    if (k > VV_PART)
    {
        if (lane == 0u)
        {
            for (uint a = 0u, aStop = EB_END(880, a, k, 1u); a < aStop; a++)
                if (edgeVeh(vvMember(off, a))) gAcc[vvCtl(VV_OVERFLOW)].max(FL_ERR_EDGE_GROUP);
        }
        splitGroupEndNow(off, k, lane, s);
        return ebLoopCapFlush();
    }
    edgePass(off, k, s, lane);
    // The frame's end after its last step's edge hits (the groups' pass leaves it here).
    splitGroupEndNow(off, k, lane, s);
    ebLoopCapFlush();
}

void edgePass(uint off, uint k, uint s, uint lane)
{
    uint parity = (gMisc[vvMember(off, 0u)].w + s) & 1u;
    // The owners' partners: their boxes overlap (the contact pass's test).
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    for (uint p = lane, pStop = EB_END(881, p, k * k, LANES); p < pStop; p += LANES)
    {
        uint a = p / k;
        uint m = p - a * k;
        uint ka = vvBoxKey(vvMember(off, a), parity);
        uint kb = vvBoxKey(vvMember(off, m), parity);
        float3 alo = ssub3(float3(vvUnkey(gAcc[ka].load()), vvUnkey(gAcc[ka + 1u].load()), vvUnkey(gAcc[ka + 2u].load())), m3);
        float3 ahi = sadd3(float3(vvUnkey(gAcc[ka + 3u].load()), vvUnkey(gAcc[ka + 4u].load()), vvUnkey(gAcc[ka + 5u].load())), m3);
        float3 blo = ssub3(float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())), m3);
        float3 bhi = sadd3(float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())), m3);
        gsVvPair[a * VV_PART + m] = (a != m && vvOverlap(alo, ahi, blo, bhi)) ? 1u : 0u;
    }
    gVvTag = 0u;
    sync();
    // (No pair with edge contact: nothing to do. The pairs are workgroup memory, read by
    // every lane alike.)
    bool any = false;
    for (uint p = 0u, pStop = EB_END(885, p, k * k, 1u); p < pStop; p++)
    {
        uint a = p / k;
        uint m = p - a * k;
        if (gsVvPair[a * VV_PART + m] != 0u && edgePairOn(vvMember(off, a), vvMember(off, m))) any = true;
    }
    if (splitUniform(lane, any ? 1u : 0u) == 0u) return;
    edgeTestsSmall(k, off, parity, lane);
    sync();
    // The forces after the node turns at the hits' nodes, before any edge hit applies (the
    // CPU's one snapshot of every node; two-body contact's first pass reads them).
    if (gP.vvSplit != 0u)
    {
        for (uint a = 0u, aStop = EB_END(886, a, k, 1u); a < aStop; a++) edgeSnapOwner(vvMember(off, a), parity, 0u, lane, LANES);
    }
    sync();
    // The owners by vehicle index, as the CPU's turns (a member keeps its slot).
    for (uint r = 0u, rStop = EB_END_UNIFORM(882, r, k, 1u, VV_PART); r < rStop; r++)
    {
        uint a = 0u;
        for (uint m = 0u, mStop = EB_END(883, m, k, 1u); m < mStop; m++)
        {
            uint below = 0u;
            for (uint q = 0u, qStop = EB_END(884, q, k, 1u); q < qStop; q++)
                if (vvMember(off, q) < vvMember(off, m)) below++;
            if (below == r) a = m;
        }
        uint va = vvMember(off, a);
        uint n = splitUniform(lane, edgeHits(va));
        if (n == 0u) continue;
        edgeApplyOwner(va, a, lane, LANES);
        if (gP.vvSplit != 0u)
        {
            for (uint ph = 0u; ph < 3u; ph++)
            {
                sync();
                edgeResidualPhase(va, a, parity, lane, LANES, ph);
            }
        }
        sync();
        edgeClearOwner(va, a, lane, LANES);
        sync();
    }
}
