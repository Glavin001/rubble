// Build-time atomic backend. WGSL (`-DENGINE_WGSL`) requires atomic<u32> storage, so
// atomic slots become Atomic<uint> and every touch goes through these macros. Metal /
// SPIR-V / CUDA keep plain uint + Interlocked*, so their generated code is unchanged.
#ifndef ENGINE_PHYSICS_ATOMICS_SLANG
#define ENGINE_PHYSICS_ATOMICS_SLANG

#if defined(ENGINE_WGSL)
typealias AUint = Atomic<uint>;
#define ALOAD(x) ((x).load())
#define ASTORE(x, v) (x).store(v)
#define AADD2(x, v) (x).add(v)
#define AADD3(x, v, o) o = (x).add(v)
#define AOR2(x, v) (x).or(v)
#define AMAX2(x, v) (x).max(v)
#define AXCHG3(x, v, o) o = (x).exchange(v)
// WGSL only exposes atomicCompareExchangeWeak. A spurious fail returns old==cmp
// without storing; callers that treat old==cmp as success then drop hash inserts
// (contact warm-start / joint hash) and pump KE on large piles. Retry until the
// value leaves cmp or the CAS reports a different occupant.
#define ACAS4(x, cmp, v, o)                                                 \
    {                                                                       \
        for (uint _casK = 0u; _casK < 32u; _casK++)                         \
        {                                                                   \
            (o) = (x).compareExchange((cmp), (v));                          \
            if ((o) != (cmp))                                               \
                break;                                                      \
            if ((x).load() != (cmp))                                        \
                break;                                                      \
        }                                                                   \
    }
// Groupshared-only exchange: All* lowers to textureBarrier(), which naga/wgpu reject.
#define GROUP_SYNC() GroupMemoryBarrierWithGroupSync()
#define BIND(n) [[vk::binding(n, 0)]]
#elif defined(ENGINE_CPU_TEST)
// CPU conformance harness (slangc -target cpp): threads run sequentially, so
// plain read-modify-writes are exact and barriers are no-ops. Kernels that need
// cross-lane barriers are instantiated with one lane per item for this target.
typealias AUint = uint;
#define ALOAD(x) (x)
#define ASTORE(x, v) (x) = (v)
#define AADD2(x, v) (x) += (v)
#define AADD3(x, v, o) do { (o) = (x); (x) += (v); } while (false)
#define AOR2(x, v) (x) |= (v)
#define AMAX2(x, v) (x) = max((x), (v))
#define AXCHG3(x, v, o) do { (o) = (x); (x) = (v); } while (false)
#define ACAS4(x, cmp, v, o) do { (o) = (x); if ((o) == (cmp)) (x) = (v); } while (false)
#define GROUP_SYNC()
#define BIND(n)
#else
#define BIND(n)
typealias AUint = uint;
#define ALOAD(x) (x)
#define ASTORE(x, v) (x) = (v)
#define AADD2(x, v) InterlockedAdd(x, v)
#define AADD3(x, v, o) InterlockedAdd(x, v, o)
#define AOR2(x, v) InterlockedOr(x, v)
#define AMAX2(x, v) InterlockedMax(x, v)
#define AXCHG3(x, v, o) InterlockedExchange(x, v, o)
#define ACAS4(x, cmp, v, o) InterlockedCompareExchange(x, cmp, v, o)
#define GROUP_SYNC() AllMemoryBarrierWithGroupSync()
#endif

// Same 64-byte layout as StepCounters; atomic fields typed AUint.
public struct GpuCounters
{
    uint bodyCount;
    uint colliderCount;
    AUint contactCount;
    uint jointCount;
    AUint candidateCount;
    AUint awakeCount;
    AUint colorCount;
    AUint overflowFlags;
    AUint pivotFloorHits;
    AUint sleepEvents;
    AUint wakeEvents;
    uint iteration;
    AUint manifoldCount;
    AUint recolorCount; // ABI byte 52: compact pending-color work count
    AUint convexSmall;  // ABI byte 56: convex SAT pairs, small bucket (front of list)
    AUint convexLarge;  // ABI byte 60: convex SAT pairs, large bucket (back of list)
};
#endif
