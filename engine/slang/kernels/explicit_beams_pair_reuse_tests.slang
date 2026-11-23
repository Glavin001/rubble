// Included by explicit_beams.slang: the reused pair list's tests (pairReuseTests).

// pairReuseTests up to its walk: true (nothing tested) past the list's room, the caller
// then walking the grid (one call site of the walk in the solo frame; pairReuseTestsEnd after).
uint pairReuseListed(uint mode) { return mode == 2u ? gsCount[GRID_BUCKETS + RU_COUNT].load() : gsPairCount.load(); }
bool pairReuseTestsOrWalk(uint v, uint lane, uint listed, uint triOff, float3 origin, float r, float cell, uint candOff,
                          uint candCount, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    if (listed > TRI_PAIR_CAP) return true;
    uint at = triPairsAt(v);
    for (uint i = lane, iStop = EB_END(35, i, listed, LANES); i < iStop; i += LANES)
        pairReuseTest(gNext[at + 2u * i], gNext[at + 2u * i + 1u], origin, r, cell, beamOff, groupOff, groupCount,
                      parity);
    piecesTests(v, lane, triOff, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount, parity);
    return false;
}
void pairReuseTestsEnd(uint lane, uint mode, uint listed)
{
    if (lane == 0u && mode == 1u)
    {
        if (gP.pieceBase != 0u) piecesClearBuild();
        gsCount[GRID_BUCKETS + RU_VALID].store(listed <= TRI_PAIR_CAP ? 1u : 0u);
        gsCount[GRID_BUCKETS + RU_COUNT].store(listed);
        gsCount[GRID_BUCKETS + RU_STATS].add(1u);
        gsCount[GRID_BUCKETS + RU_STATS + 2u].add(listed);
        gsCount[GRID_BUCKETS + RU_STATS + 3u].max(listed);
    }
}
void pairReuseTests(uint v, uint lane, uint mode, uint triOff, uint triCount, float3 origin, float r, float cell,
                    uint candOff, uint candCount, uint beamOff, uint groupOff, uint groupCount, uint parity)
{
    sync();
    uint listed = pairReuseListed(mode);
    if (pairReuseTestsOrWalk(v, lane, listed, triOff, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount,
                             parity))
    {
        uint tested = triCount <= TRI_LIST ? gsTestedCount[parity].load() : triCount;
        for (uint p = lane, pStop = EB_END(36, p, tested * GRID_SUBS, LANES); p < pStop; p += LANES)
        {
            uint t = triCount <= TRI_LIST ? gsTested[p / GRID_SUBS] : p / GRID_SUBS;
            triangleGrid(triOff + t, p % GRID_SUBS, origin, r, cell, candOff, candCount, beamOff, groupOff, groupCount,
                         parity, 0u);
        }
    }
    pairReuseTestsEnd(lane, mode, listed);
}
