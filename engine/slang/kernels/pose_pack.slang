// Render poses for hosts: after the step, every body's pose in a compact 32-byte
// record (binding 34), and a list of the bodies whose pose or sleep state changed
// this step (bindings 35-36). The pass only reads physics state (gBodies) and
// writes its own buffers, so a step with it computes exactly what a step without
// it computes. Sleeping bodies cost one read and no write.
#include "../physics/common.slang"
#include "../physics/atomics.slang"

// Matches engine-webgpu `PoseRecord` (32 B): position, a state or id word, rotation.
struct PoseRecord
{
    float3 position;
    uint word;
    float4 rotation;
};

// gPoses[i].word: the state last published for body i.
static const uint POSE_STATE_SLEEPING = 1u;
static const uint POSE_STATE_PUBLISHED = 2u; // cleared by the host when a slot is reused

// gMoved[k].word: body index in the low 28 bits, then what changed.
static const uint MOVED_BODY_MASK = 0x0FFFFFFFu;
static const uint MOVED_SLEPT = 1u << 28;
static const uint MOVED_WOKE = 1u << 29;
static const uint MOVED_FIRST = 1u << 30;

BIND(0) RWStructuredBuffer<BodyState> gBodies;
BIND(34) RWStructuredBuffer<PoseRecord> gPoses;
// [0] records appended this step (the host zeroes it), [1] bodies to scan.
BIND(35) RWStructuredBuffer<AUint> gMovedHeader;
BIND(36) RWStructuredBuffer<PoseRecord> gMoved;

uint dispatchIndex(uint3 tid) { return tid.x + tid.y * (32768u * 64u); }

[shader("compute")]
[numthreads(64, 1, 1)]
void pose_pack(uint3 tid : SV_DispatchThreadID)
{
    uint i = dispatchIndex(tid);
    if (i >= ALOAD(gMovedHeader[1])) return;
    BodyState b = gBodies[i];
    PoseRecord last = gPoses[i];
    uint sleeping = (b.flags & BODY_SLEEPING) != 0u ? POSE_STATE_SLEEPING : 0u;
    uint state = POSE_STATE_PUBLISHED | sleeping;
    bool first = (last.word & POSE_STATE_PUBLISHED) == 0u;
    bool moved = any(asuint(b.position) != asuint(last.position))
        || any(asuint(b.orientation) != asuint(last.rotation));
    if (!first && !moved && last.word == state) return;

    uint change = 0u;
    if (first)
    {
        change = MOVED_FIRST;
    }
    else if (sleeping != (last.word & POSE_STATE_SLEEPING))
    {
        change = sleeping != 0u ? MOVED_SLEPT : MOVED_WOKE;
    }
    PoseRecord now;
    now.position = b.position;
    now.word = state;
    now.rotation = b.orientation;
    gPoses[i] = now;

    uint slot;
    AADD3(gMovedHeader[0], 1u, slot);
    PoseRecord rec = now;
    rec.word = (i & MOVED_BODY_MASK) | change;
    gMoved[slot] = rec;
}
