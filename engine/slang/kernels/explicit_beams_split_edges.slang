// --- Edge vehicle contact's pass (the split group path: a dispatch a substep, after the
// groups' contact pass; one workgroup a contact group) ---
// The CPU reference (explicit_contact.rs contact_pass): every owner's turn of its edge
// hits after every owner's node turn, in the members' order, each turn's first pass from
// the forces after the node turns (one snapshot at the pass's start), its residual pass
// and work guard. An owner's hits are found for the group at once (edgeTestsSmall: the
// owner's split counts in its slot), then applied owner by owner.
// A group of at most VV_PART members finds every owner's hits at once (edgePass); a larger
// one finds them an owner at a time, slot 0.
void edgePass(uint off, uint k, uint s, uint lane);
void ccPass(uint off, uint k, uint s, uint lane);

[shader("compute")]
[numthreads(EB_LANES, 1, 1)]
void explicit_beams_split_edge(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint lane = lid.x;
    uint4 G = gMisc[gP.vvGroups + gid.x];
    uint off = G.x;
    // Continuous contact's pass between a step's two halves (both seam bits in the step
    // word: explicit_plan::continuous), a workgroup a group.
    if ((gP.pad2 & (SPLIT_CC_NODES | SPLIT_CC_REST)) == (SPLIT_CC_NODES | SPLIT_CC_REST))
    {
        uint kc = splitUniform(lane, gP.cc != 0u ? splitAwake(G) : 0u);
        if (kc != 0u) ccPass(off, kc, gP.pad2 & SPLIT_STEP, lane);
        return ebLoopCapFlush();
    }
    uint k = splitUniform(lane, edgeOn() ? splitAwake(G) : 0u);
    if (k == 0u) return ebLoopCapFlush();
    uint s = gP.pad2 & SPLIT_STEP;
    edgePass(off, k, s, lane);
    // The frame's end after its last step's edge hits (the groups' pass leaves it here).
    splitGroupEndNow(off, k, lane, s);
    ebLoopCapFlush();
}

// The contact pass's partner test: the members' boxes overlap.
bool edgePartner(uint off, uint a, uint m, uint parity)
{
    float3 m3 = float3(VV_BOX_MARGIN, VV_BOX_MARGIN, VV_BOX_MARGIN);
    uint ka = vvBoxKey(vvMember(off, a), parity);
    uint kb = vvBoxKey(vvMember(off, m), parity);
    float3 alo = ssub3(float3(vvUnkey(gAcc[ka].load()), vvUnkey(gAcc[ka + 1u].load()), vvUnkey(gAcc[ka + 2u].load())), m3);
    float3 ahi = sadd3(float3(vvUnkey(gAcc[ka + 3u].load()), vvUnkey(gAcc[ka + 4u].load()), vvUnkey(gAcc[ka + 5u].load())), m3);
    float3 blo = ssub3(float3(vvUnkey(gAcc[kb].load()), vvUnkey(gAcc[kb + 1u].load()), vvUnkey(gAcc[kb + 2u].load())), m3);
    float3 bhi = sadd3(float3(vvUnkey(gAcc[kb + 3u].load()), vvUnkey(gAcc[kb + 4u].load()), vvUnkey(gAcc[kb + 5u].load())), m3);
    return a != m && vvOverlap(alo, ahi, blo, bhi);
}

void edgePass(uint off, uint k, uint s, uint lane)
{
    uint parity = (gMisc[vvMember(off, 0u)].w + s) & 1u;
    bool big = k > VV_PART;
    gVvTag = 0u;
    if (!big)
    {
        for (uint p = lane, pStop = EB_END(881, p, k * k, LANES); p < pStop; p += LANES)
            gsVvPair[(p / k) * VV_PART + p % k] = edgePartner(off, p / k, p % k, parity) ? 1u : 0u;
        sync();
        // (No pair with edge contact: nothing to do.)
        bool any = false;
        for (uint p = 0u, pStop = EB_END(885, p, k * k, 1u); p < pStop; p++)
            if (gsVvPair[(p / k) * VV_PART + p % k] != 0u && edgePairOn(vvMember(off, p / k), vvMember(off, p % k))) any = true;
        if (splitUniform(lane, any ? 1u : 0u) == 0u) return;
        edgeTestsSmall(k, off, parity, lane);
        sync();
        // The forces after the node turns, at the hits' nodes, before any edge hit applies.
        if (gP.vvSplit != 0u)
        {
            for (uint a = 0u, aStop = EB_END(886, a, k, 1u); a < aStop; a++) edgeSnapOwner(vvMember(off, a), parity, 0u, lane, LANES);
        }
    }
    else
    {
        // (Its owners' hits are found one owner at a time: every member's forces.)
        edgeResetPrepare(k, off, lane, true);
        if (gP.vvSplit != 0u) vvSnapshot(off, k, parity, lane);
    }
    sync();
    // The owners' turns in the members' order (by vehicle index, the CPU's).
    for (uint a = 0u, aStop = EB_END_UNIFORM(882, a, k, 1u, EB_MAX_MEMBERS); a < aStop; a++)
    {
        uint va = vvMember(off, a);
        uint slot = big ? 0u : a;
        if (big)
        {
            for (uint m = lane, mStop = EB_END(888, m, k, LANES); m < mStop; m += LANES)
                gNext[vvPartnerAt(off, m)] = edgePartner(off, a, m, parity) ? 1u : 0u;
            sync();
            edgeOwnerBig(a, k, off, lane);
        }
        if (splitUniform(lane, edgeHits(va)) == 0u) continue;
        edgeApplyOwner(va, slot, lane, LANES);
        for (uint ph = 0u; ph < 3u; ph++)
        {
            sync();
            edgeResidualPhase(va, slot, parity, lane, LANES, ph);
        }
        sync();
        edgeClearOwner(va, slot, lane, LANES);
        sync();
    }
}

#include "explicit_beams_split_cc.slang"
