// --- A fleet with vehicle contact on the wide passes (EB_VV_WIDE) ---
// engine/docs/measurements/vehicle-perf-first-principles.md §8.57. With vehicle contact
// on, a frame in which the contact plan holds no group steps every vehicle alone
// (explicit_beams_frame under EB_VV_SOLO, or a step a dispatch: explicitBeamsFrameStep).
// The same frame as the wide passes (a thread an element across the GPU, as a fleet
// with contact off runs): the same per-element functions, plus the one thing the solo
// frame adds under EB_VV_SOLO, the frame-bounds check (wideVvBounds). The plan decides
// on the GPU whether a frame holds a group, so a host launches both schedules every
// frame and explicit_vv_gate sizes them: the wide passes get their workgroups when the
// plan holds no group, today's per-vehicle dispatches when it holds one.
//
// Three small modules, each one entry of this file: the gate, and the node and close
// passes with the bounds check (the wide modules themselves are unchanged: a fleet with
// contact off runs the code it ran).

// The gate's buffer (explicit_plan::schedule::ArgBuf::Wide): VW_SLOTS indirect
// arguments of four words (x, 1, 1, unused), read by the launches they size and bound
// by no launch but the gate's.
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> gVvWide;

// The slots (explicit_plan::vv_wide::WideSlot): the wide passes' workgroup counts as
// FramePlanner::wide_plan gives them, each 0 in a frame with a group; VW_OLD the
// vehicles in a frame with a group, 0 without (the per-vehicle dispatches that
// otherwise run every frame).
static const uint VW_NV = 0u; // a workgroup a vehicle: aux, hits, the list, the surface prep
static const uint VW_NV2 = 1u; // the list with a second workgroup a vehicle (PERF_AHEAD_APART)
static const uint VW_NV3 = 2u; // … and a third (PERF_WHEELS_APART)
static const uint VW_VEH = 3u; // a thread a vehicle: begin, close
static const uint VW_NODES = 4u; // a thread a node: rebase, the centred nodes
static const uint VW_NODES_ALL = 5u; // the node pass: nodes, group slots, grid buckets
static const uint VW_BEAMS = 6u; // a thread a beam: the centred beams
static const uint VW_BEAMS_ALL = 7u; // the beam pass: beams, candidates, triangles
static const uint VW_END = 8u; // the end pass: triangles, beams
static const uint VW_OLD = 9u;
// Words the gate's clear modes zero: the self split's arguments (two kinds, three step
// slots of three words: selfArgsAt, selfWalkArgsAt) and the surface pass's (five
// kinds: surfArgsAt).
static const uint VW_CLEAR_SELF = 18u;
static const uint VW_CLEAR_SURF = 45u;

// Workgroups of 256 threads for n elements (the plan's: at least one).
uint vwGroups(uint n) { return max((n + 255u) / 256u, 1u); }
void vwPut(uint slot, uint x)
{
    gVvWide[4u * slot] = x;
    gVvWide[4u * slot + 1u] = 1u;
    gVvWide[4u * slot + 2u] = 1u;
}

// One workgroup, after the contact plan's passes, every frame. The step word picks what it
// does (a parameter copy each):
//   0  binding 10 the contact plan's arguments: from the plan's group count (vvArgs,
//      word 3; with the arguments in every snapshot) the slots above.
//   1  binding 10 the self split's arguments: zeroed.
//   2  binding 10 the surface pass's arguments: zeroed.
// (1 and 2, at a frame's start where the wide frame's walk, pairs and surface dispatches
// are not today's path's own: a step's slot is otherwise cleared by the dispatch of the
// step before, which gets no workgroup in a frame with a group. Zeroing every slot at a
// frame's start changes nothing for a frame that writes them: the first step's is clear
// then already and each later one is cleared before its step.)
[shader("compute")]
[numthreads(64, 1, 1)]
void explicit_vv_gate(uint3 id : SV_DispatchThreadID)
{
    uint mode = gP.pad2;
    if (mode != 0u)
    {
        // (A thread a word: no loop.)
        if (id.x < (mode == 1u ? VW_CLEAR_SELF : VW_CLEAR_SURF)) gArgs[id.x] = 0u;
        return;
    }
    if (id.x != 0u) return;
    uint nv = gP.vehicles;
    uint wide = gArgs[3] == 0u ? 1u : 0u;
    vwPut(VW_NV, wide * nv);
    vwPut(VW_NV2, wide * 2u * nv);
    vwPut(VW_NV3, wide * 3u * nv);
    vwPut(VW_VEH, wide * vwGroups(nv));
    vwPut(VW_NODES, wide * vwGroups(gP.totalNodes));
    vwPut(VW_NODES_ALL, wide * vwGroups(gP.totalNodes + gP.groupTotal + nv * GRID_BUCKETS));
    vwPut(VW_BEAMS, wide * vwGroups(gP.totalBeams));
    vwPut(VW_BEAMS_ALL, wide * vwGroups(gP.totalBeams + gP.totalCands + gP.totalTris));
    vwPut(VW_END, wide * vwGroups(gP.totalTris + gP.totalBeams));
    vwPut(VW_OLD, (1u - wide) * nv);
}

// The solo frame's bounds check for node gi of vehicle v (nodeUpdateSolo and its
// variants under EB_VV_SOLO, and the frame's end): the position the last step left
// against the frame's allowed box, relative to the vehicle's origin; outside, the
// vehicle reported (an atomic maximum: in any order the same). The box as the solo
// frame's lane 0 keeps it (gsVvBounds: centre and half extents), here from the two
// records a thread.
void wideVvBounds(uint gi, uint v, float3 origin)
{
    float3 lo = asfloat(gMisc[vvRec(v, VV_ALLOWED_LO)].xyz) - origin;
    float3 hi = asfloat(gMisc[vvRec(v, VV_ALLOWED_HI)].xyz) - origin;
    float3 centre = 0.5 * (lo + hi);
    float3 half = 0.5 * (hi - lo);
    if (any(abs(gNodes[gi * 3u].xyz - centre) > half)) vvReportOutside(v);
}

// explicit_beams_wide_nodes with the bounds check: a node's thread checks the position
// its last step left before it updates it, as nodeUpdateSolo does (never a retired
// node's: its node phase returns before the check).
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_nodes_vv(uint3 id : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID)
{
    uint i = id.x;
    if (i < gP.totalNodes && !(gP.retireBase != 0u && nodeRetired(i)))
    {
        uint v = gStatic[gP.nodeInfoBase + i].w;
        wideVvBounds(i, v, vehicleOrigin(v));
    }
    explicit_beams_wide_nodes(id, lid);
}

// explicit_beams_wide_close with the frame's last bounds check: every node of the
// vehicle at the last step's position (the solo frame's end, retired nodes included),
// on the vehicle's thread.
[shader("compute")]
[numthreads(256, 1, 1)]
void explicit_beams_wide_close_vv(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v < gP.vehicles)
    {
        uint4 r0 = gStatic[v * 4u];
        float3 origin = asfloat(gMisc[v].xyz);
        for (uint i = 0u, iStop = EB_END(606, i, r0.y, 1u); i < iStop; i++) wideVvBounds(r0.x + i, v, origin);
    }
    explicit_beams_wide_close(id);
}
