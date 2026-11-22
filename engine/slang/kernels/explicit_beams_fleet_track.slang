// --- Vehicle contact's kept candidates (PERF_FLEET_KEEP): their constants, the vehicles on
// the fleet-wide passes and the tracking of a node's motion since the list's build ---
// (contact-groups-design.md §8.10; the passes: explicit_beams_fleet.slang fk*.) Included
// with the split group path (EB_VV_SPLIT) ahead of explicit_beams_split.slang, whose
// members' rest (memberRest) tracks its vehicle's nodes when the host says so
// (PERF_KEEP_TRACK_REST), and of explicit_beams_fleet.slang, whose scans' dispatch does
// otherwise (fkTrack).

static const float FK_CALM = 0.05; // a calm node's motion and travel since the build, at most
static const float FK_WILD = 0.35; // a wild node's (a spinning wheel's, a flying piece's)
static const float FK_BETWEEN = 0.05; // two vehicles' relative motion, with the rounding, at least;
static const float FK_BETWEEN_MOST = 0.3; // at most; a build's own: FK_BETWEEN_PASSES passes of its
static const float FK_BETWEEN_PASSES = 16.0; // fastest pair of partners' relative motion (fkBetween)
static const float FK_WILD_SPEED = 1.0; // a node this fast relative to its vehicle is wild
static const float FK_FAR = 0.3; // two vehicles with no pair this much past a kept pair's box: this much more
static const uint FK_NEAR = 0x80000000u; // a covered partner's entry: not far
static const float FK_PER_CELL = 1.0 / 0.6; // the build's grid: cells of 0.6 m (a multiply: the
// gather's cell of a candidate and the walk's are the same bits)
static const uint FK_MAX_CELLS = 216u; // a box over more cells walks every candidate
static const uint FK_CAP = TRI_PAIR_CAP - 1u;
static const uint FK_MAGIC = 0xFFFFFFFFu;
static const uint FK_BACKOFF = 136u;
static const uint FK_SHORT = 3u; // a list that lasted fewer passes: today's path for FK_THRASH_WAIT
static const uint FK_THRASH_WAIT = 68u;
// The pass's mode (fkCtl(FK_MODE), written by the first plan).
static const uint FK_OFF = 0u; // today's front
static const uint FK_REUSE = 1u; // the kept pairs' tests alone
static const uint FK_BUILD = 2u; // the padded front, then the kept pairs' tests
static const uint FK_OVER = 3u; // a build past an owner's room: its tests in the walk's own boxes (fkWalk)
// fkCtl's words.
static const uint FK_MODE = 0u;
static const uint FK_VALID = 1u;
static const uint FK_EPOCH = 2u;
static const uint FK_WAIT = 3u;
static const uint FK_SUBS = 4u;
static const uint FK_BUILDS = 5u; // measurement: builds, reused passes, pairs kept (summed over
static const uint FK_REUSED = 6u; // builds), passes on today's path after a build failed, the
static const uint FK_KEPT = 7u; // most pairs an owner kept
static const uint FK_FELL = 8u;
static const uint FK_MOST = 9u;
static const uint FK_BY_CALM = 10u; // builds because a calm node passed its allowance,
static const uint FK_BY_COVER = 11u; // … a pair the build did not cover (or no list),
static const uint FK_DEV_MOST = 12u; // the largest motion bound a vehicle had at a build for motion (float bits)
static const uint FK_BTW = 13u; // the list's allowance for two vehicles' relative motion (float bits)
static const uint FK_PASS = 14u; // the passes so far (a clock for the hot nodes)
static const uint FK_BY_BETWEEN = 15u; // … two vehicles' relative motion (the rest: a wild node's)
// A node that has used more than half a calm node's allowance since a build is hot: wild
// at the builds of the next FK_HOT · 256 passes (its record's .w: the class at the build
// in bit 0, the pass / 256 its heat ends above).
static const uint FK_HOT = 16u;
// flPrefix's rows and counts of the kept candidates.
static const uint FK_ROW_RUN = 16u;
static const uint FK_ROW_PAIRS = 17u;
static const uint FK_W_RUN = 100u;
static const uint FK_W_PAIRS = 101u;

// The kept list's allowance for two vehicles' relative motion: the build's choice, the
// same for its boxes and for the bound that holds the list to them.
float fkBetween() { return min(max(asfloat(gNext[fkCtl(FK_BTW)]), FK_BETWEEN), FK_BETWEEN_MOST); }
bool fkOn() { return (gP.perfFlags & PERF_FLEET_KEEP) != 0u && gP.vvTriPairs != 0u; }
uint fkMode() { return fkOn() ? gNext[fkCtl(FK_MODE)] : FK_OFF; }
// FK_WAIT: the passes left on today's path (low 16 bits) and how many builds past a room
// in a row (above): each one doubles the next wait, to FK_BACKOFF << 4 at most; a build
// that keeps its list clears it. (Choices of path alone.)
uint fkWaitLeft() { return gNext[fkCtl(FK_WAIT)] & 0xFFFFu; }
void fkBackoff()
{
    uint streak = gNext[fkCtl(FK_WAIT)] >> 16u;
    gNext[fkCtl(FK_WAIT)] = (min(streak + 1u, 4u) << 16u) | (FK_BACKOFF << min(streak, 4u));
}
// A vehicle whose contact is the fleet-wide passes' this frame: a member of a group the
// small groups' levels take, or with PERF_FLEET_ANY of any group (else a larger group's
// pass, vvPass, is its own).
bool flAnySize() { return (gP.perfFlags & PERF_FLEET_ANY) != 0u; }
bool flVehicleOn(uint v)
{
    uint4 fl = gMisc[vvRec(v, VV_FLAGS)];
    if (fl.y == 0u) return false;
    // (Vehicle sleep: a sleeping group's members are no one's partners; its levels do
    // not run. explicit_beams_split.slang splitAwake.)
    if (sleepAsleep(v)) return false;
    uint k = gMisc[gP.vvGroups + fl.z].y;
    return k != 0u && (k <= VV_PART || flAnySize());
}
// A float's bits as an ordered key of a bound: itself when finite and not negative, else
// +inf's (NaN and negative values: never under a margin).
uint fkBits(float x)
{
    uint b = asuint(x);
    return b >= 0x7F800000u ? 0x7F800000u : b;
}

// A vehicle's motion since the build as its nodes' records have it: a node's is its
// position (the vehicle's frame) less its recorded one plus the origin's change, so a
// rebase cancels. The vehicle's reference motion of the pass (fkMean 0..2, the same for
// every thread of the pass): its tracked nodes' mean motion of the pass before plus that
// pass's change of it (fkDecide). Any vector is sound: a node's bound is its motion
// relative to it, and two vehicles' relative motion is taken from the same two vectors.
float3 fkOriginShift(uint v)
{
    return asfloat(gMisc[v].xyz) - asfloat(uint3(gNext[fkOrigin(v)], gNext[fkOrigin(v) + 1u], gNext[fkOrigin(v) + 2u]));
}
float3 fkReference(uint v) { return asfloat(uint3(gNext[fkMean(v, 0u)], gNext[fkMean(v, 1u)], gNext[fkMean(v, 2u)])); }
bool fkWild(float4 ref) { return (asuint(ref.w) & 1u) != 0u; }


// Node n of vehicle v (every node: a contact point or a triangle's node is one): how far
// it has moved since the build relative to the vehicle's reference motion, and its travel
// over a step, into the vehicle's bounds of the node's class at the build; its motion
// into the vehicle's sum (fixed point, a metre at most: the mean is a reference, not a
// result).
void fkTrackNode(uint n, uint v)
{
    float4 ref = gBeams[fkRef(n)];
    float3 m = (gNodes[n * 3u].xyz - ref.xyz) + fkOriginShift(v);
    float3 d = m - fkReference(v);
    float3 w = gNodes[n * 3u + 1u].xyz;
    uint c = fkWild(ref) ? 2u : 0u;
    float dev = sqrt(dot(d, d));
    float trav = sqrt(dot(w, w)) * STEP;
    // (A bound only grows within a pass: no atomic for a value under it.)
    int db = int(fkBits(dev));
    int tb = int(fkBits(trav));
    if (db > gAcc[fkAcc(v, FK_DEV + c)].load()) gAcc[fkAcc(v, FK_DEV + c)].max(db);
    if (tb > gAcc[fkAcc(v, FK_TRAV + c)].load()) gAcc[fkAcc(v, FK_TRAV + c)].max(tb);
    // (The mean of one node in eight: any reference is sound, fkSampled counts them.)
    if ((n & 7u) == 0u)
    {
        int3 q = int3(clamp(m, float3(-1.0, -1.0, -1.0), float3(1.0, 1.0, 1.0)) * 4096.0);
        gAcc[fkAcc(v, FK_SUM)].add(q.x);
        gAcc[fkAcc(v, FK_SUM + 1u)].add(q.y);
        gAcc[fkAcc(v, FK_SUM + 2u)].add(q.z);
    }
    // (Hot from now on; every thread that tracks the node writes the same record.)
    if (fkBits(dev + trav) > asuint(0.5 * FK_CALM))
    {
        uint heat = (gNext[fkCtl(FK_PASS)] >> 8u) + FK_HOT;
        uint bits = asuint(ref.w);
        if ((bits >> 1u) != heat) gBeams[fkRef(n)] = float4(ref.xyz, asfloat((heat << 1u) | (bits & 1u)));
    }
}
// The tracking of the pass in the members' rest dispatch, not the scans' (the host's
// PERF_KEEP_TRACK_REST, set with the kept candidates, the fused scans and the members'
// rest dispatches: the scans then stay in the plan's workgroup, fuseScansOn). The rest is
// the last dispatch to write a member's nodes before the scans, so the same positions and
// velocities; the bounds are maxima and the sum an integer sum, so in any order the same.
bool fkTrackInRest() { return (gP.perfFlags & (PERF_FLEET_KEEP | PERF_KEEP_TRACK_REST)) == (PERF_FLEET_KEEP | PERF_KEEP_TRACK_REST); }
// Workgroup v of the members' rest, after its rest (every lane; no barrier inside): its
// vehicle's nodes, a lane each, as fkTrack's threads take them.
// A pass after one in which no owner was offered a partner (the last pass's offered total,
// which this pass's first plan replaces after the rest) is not tracked: its plan then reuses
// no list (fkDecide's `untracked`). Cars in a group but apart track nothing.
bool fkTrackIdle() { return gNext[flPrefixAt(0u, gP.vehicles)] == 0u; }
void fkTrackVehicle(uint v, uint lane)
{
    if (v >= gP.vehicles || gP.vvTriPairs == 0u || fkWaitLeft() > 1u || fkTrackIdle() || !flVehicleOn(v)) return;
    uint4 r0 = gStatic[v * 4u];
    // (fkTrackNode's values, each lane's maxima and sum kept in registers and put once: the
    // same maxima and the same integer sum, a quarter of the atomics on four words.)
    float3 shift = fkOriginShift(v);
    float3 refm = fkReference(v);
    int4 most = int4(0, 0, 0, 0);
    int3 sum = int3(0, 0, 0);
    bool sampled = false;
    uint hot = (gNext[fkCtl(FK_PASS)] >> 8u) + FK_HOT;
    for (uint i = lane, iStop = EB_END(446, i, r0.y, LANES); i < iStop; i += LANES)
    {
        uint n = r0.x + i;
        float4 ref = gBeams[fkRef(n)];
        float3 m = (gNodes[n * 3u].xyz - ref.xyz) + shift;
        float3 d = m - refm;
        float3 w = gNodes[n * 3u + 1u].xyz;
        float dev = sqrt(dot(d, d));
        float trav = sqrt(dot(w, w)) * STEP;
        int db = int(fkBits(dev));
        int tb = int(fkBits(trav));
        if (fkWild(ref))
        {
            most.z = max(most.z, db);
            most.w = max(most.w, tb);
        }
        else
        {
            most.x = max(most.x, db);
            most.y = max(most.y, tb);
        }
        if ((n & 7u) == 0u)
        {
            sum += int3(clamp(m, float3(-1.0, -1.0, -1.0), float3(1.0, 1.0, 1.0)) * 4096.0);
            sampled = true;
        }
        if (fkBits(dev + trav) > asuint(0.5 * FK_CALM))
        {
            uint bits = asuint(ref.w);
            if ((bits >> 1u) != hot) gBeams[fkRef(n)] = float4(ref.xyz, asfloat((hot << 1u) | (bits & 1u)));
        }
    }
    for (uint c = 0u; c < 4u; c++)
    {
        int b = c == 0u ? most.x : (c == 1u ? most.y : (c == 2u ? most.z : most.w));
        if (b > gAcc[fkAcc(v, c)].load()) gAcc[fkAcc(v, c)].max(b);
    }
    if (sampled)
    {
        gAcc[fkAcc(v, FK_SUM)].add(sum.x);
        gAcc[fkAcc(v, FK_SUM + 1u)].add(sum.y);
        gAcc[fkAcc(v, FK_SUM + 2u)].add(sum.z);
    }
}
// The kept pairs' places (explicit_beams_fleet.slang fkKeepPair): the fleet's pair regions
// as one room of vehicles · FK_CAP slots (slot g: region g / FK_CAP's pair g % FK_CAP, never
// its two signature words), each owner a share at the build (fkAlloc's prefix: its last
// build's count and an eighth more, the room left shared evenly; past the room, scaled
// down). An owner past its share: the list is void, as past the room before, and the next
// build's shares take its count. (Where a pair is kept decides nothing: fkTest tests every
// kept pair.)
static const uint FK_ROW_ALLOC = 18u;
static const uint FK_W_ALLOC = 102u;
uint fkRoom() { return gP.vehicles * FK_CAP; }
uint fkSlotAt(uint g) { return vvTriPairsAt(g / FK_CAP) + 2u * (1u + g % FK_CAP); }
// Owner v's want at a build: its last build's pairs and an eighth more, and 512 (a build
// past its shares counts every pair: fkKeepPair). After a build that skipped its listing
// (no counts), its share.
uint fkWant(uint v)
{
    uint last = min(uint(gAcc[fkAcc(v, FK_PAIRS)].load()), fkRoom());
    if (last == 0u && gNext[fkCtl(FK_VALID)] == 0u) return gNext[fkShareAt(v)];
    return min(last + (last >> 3u) + 512u, fkRoom());
}
// Owner v's first slot (v = vehicles: the room's end).
uint fkBase(uint v)
{
    uint nv = gP.vehicles;
    uint sum = gNext[fkAlloc(nv)];
    uint room = fkRoom();
    if (sum <= room) return v >= nv ? room : gNext[fkAlloc(v)] + v * ((room - sum) / nv);
    return v >= nv ? room : uint(float(gNext[fkAlloc(v)]) * (float(room) / float(sum)));
}
uint fkShare(uint v) { return fkBase(v + 1u) - fkBase(v); }
