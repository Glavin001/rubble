// The gate of blast_native_wide.slang's iterations. The host records `max_iterations`
// iterations; a row that converged or stopped makes each of their kernels a no-op, but each is
// still dispatched over every row's blocks. Before each iteration this zeroes the iterations'
// indirect sizes once no row is active, so the rest of the frame's iterations dispatch no
// groups. Lossless: those groups would have returned at their row's flags. The set-up and end
// kernels take their own copy of the sizes (w_final and w_report still run), and the next
// frame's set-up copies both again.
//
//   w_gate   one thread: the active rows counted; none, the iterations' grids zeroed.

import blast_common;

[[vk::binding(0, 0)]] ConstantBuffer<BlastParams> params;
// blast_native_wide.slang's `wide` (each row's state) and `meta` (word 0: the rows listed).
[[vk::binding(1, 0)]] StructuredBuffer<float4> wide;
[[vk::binding(2, 0)]] StructuredBuffer<uint> meta;
// The iterations' indirect sizes: node grid (0..2), bond grid (4..6), rows (8..10).
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> grids;

// blast_native_wide.slang `ROW_WORDS`, `S_COUNT` (its w: the flags bits), `F_ACTIVE`.
static const uint ROW_WORDS = 16;
static const uint S_COUNT = 2;
static const uint F_ACTIVE = 1;

[shader("compute")]
[numthreads(1, 1, 1)]
void w_gate(uint3 id : SV_DispatchThreadID)
{
    uint rows = min(meta[0], params.wide_rows);
    for (uint row = 0; row < rows; row++)
    {
        if ((asuint(wide[row * ROW_WORDS + S_COUNT].w) & F_ACTIVE) != 0)
            return;
    }
    grids[0] = 0;
    grids[4] = 0;
    grids[8] = 0;
}
