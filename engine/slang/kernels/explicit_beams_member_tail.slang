// Included by explicit_beams.slang before contactMemberStepHead.

// A member's step's tail (contactMemberStep's, and memberRestStep's on the split group
// path's rest dispatch): the hits applied (selfApply), the slide nodes, the step's
// closing barrier, the kept pair list's save. One function both reach, so that the
// split group path's members' module, whose entry runs one or the other, holds one copy
// of selfApply (the compilers inline every call: a copy at each site was an eighth of
// that module and of its first-load compile). The same calls in the same order as the
// two did each.
// The decisions travel as the heads' return value (MT_* bits), not in the struct: a
// barrier's uniformity holds through a return value but not through an out parameter
// (Chrome rejects a barrier behind a branch on one).
static const uint MT_STEPPED = 1u;  // splitMemberHead: the member was stepped (its end)
static const uint MT_TAIL = 2u;     // run memberTail (else the rest is another dispatch's)
static const uint MT_SYNC = 4u;     // a barrier before the hits
static const uint MT_APPLY = 8u;    // selfApply
static const uint MT_LOCAL = 16u;   // selfApplyLocalKeep
static const uint MT_SAVE = 32u;    // memberListSave after
struct MemberTail
{
    uint v;
    uint s;
    float3 origin;
    float range;
    uint beamOff;
    uint groupOff;
    uint groupCount;
    uint parity;
    uint flagsV;
    uint stepBase;
};
MemberTail memberTailNone(uint v, uint s)
{
    MemberTail t;
    t.v = v;
    t.s = s;
    t.origin = float3(0.0, 0.0, 0.0);
    t.range = 0.0;
    t.beamOff = 0u;
    t.groupOff = 0u;
    t.groupCount = 0u;
    t.parity = 0u;
    t.flagsV = 0u;
    t.stepBase = 0u;
    return t;
}
void memberTail(MemberTail t, uint how, uint lane)
{
    if ((how & MT_SYNC) != 0u) sync();
    if ((how & MT_LOCAL) != 0u)
        selfApplyLocalKeep(t.v, lane, t.beamOff, t.groupOff, t.groupCount, t.parity);
    else if ((how & MT_APPLY) != 0u)
        selfApply(t.v, true, lane, LANES, lane, t.origin, t.range, t.beamOff, t.groupOff, t.groupCount, t.parity);
#ifdef EB_AERO
    uint4 aero2 = gStatic[gP.aeroBase + t.v * 3u + 2u];
    if (lane == LANES - 1u && (t.flagsV & FLAG_SLIDES) != 0u) slidePass(aero2.x, aero2.y, SLIDE_FRAME, t.groupOff, t.parity);
#endif
    sync();
#ifdef EB_SOLO_STEP
    if ((how & MT_SAVE) != 0u) memberListSave(t.v, lane, t.s);
#endif
}
