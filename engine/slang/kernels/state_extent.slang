// A buffer's extent for a compact snapshot (WgpuSim::measure_state): one plus the
// index of its last non-zero 32-bit word, 0 when every word is zero. Past the
// extent the buffer is zeros, so a snapshot copies the words below it and a restore
// clears the rest: the same bytes as a copy of the whole buffer.
//
// One dispatch per buffer, threads striding over its words; each keeps its last
// non-zero index and the largest is kept with an atomic max.

struct ExtentParams
{
    uint words;  // the buffer's size in 32-bit words
    uint slot;   // where its extent goes in gExtents
    uint stride; // threads in the dispatch
    uint pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> gData;
[[vk::binding(1, 0)]] RWStructuredBuffer<Atomic<uint>> gExtents;
[[vk::binding(2, 0)]] ConstantBuffer<ExtentParams> gExtent;

[shader("compute")]
[numthreads(64, 1, 1)]
void state_extent(uint3 tid : SV_DispatchThreadID)
{
    uint last = 0u;
    for (uint w = tid.x; w < gExtent.words; w += gExtent.stride)
    {
        if (gData[w] != 0u)
            last = w + 1u;
    }
    if (last != 0u)
        gExtents[gExtent.slot].max(last);
}
