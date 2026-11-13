// Tests only: the explicit kernels' loop caps (explicit_loop_cap.slang) on a
// runaway, compiled for every target (engine-webgpu, engine-metal and engine-cuda
// tests explicit_loop_cap). One thread; gOut from the host:
// [1] 1 ends the first loop (0: it never ends), [3] the counted loop's bound,
// [4] the stepped loop's bound, [5] its step. Back: [0] the first loop's passes,
// [2] the constant loop's, [3] the counted loop's, [6] the stepped loop's.
// gAcc's last word (3 of 4) takes the flag.

[[vk::binding(0, 0)]] RWStructuredBuffer<Atomic<int>> gAcc;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> gOut;

#define EB_LOOP_CAP_WORD 3u
#include "explicit_loop_cap.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void explicit_loop_cap_test(uint3 tid : SV_DispatchThreadID)
{
    // A loop whose exit never comes (it reads a word the host sets to 0).
    uint n = 0u;
    uint nGuard = 0u;
    while (gOut[1] != 1u)
    {
        EB_PASS(nGuard); // loop-cap-test: pass
        n++;
    }
    // A constant loop: finite as written.
    uint i = 0u;
    for (uint k = 0u; k < 3u; k++)
        i++;
    // A counted loop up to a bound the host sets (huge for a runaway).
    uint m = gOut[3];
    uint c = 0u;
    for (uint j = 0u, jStop = EB_END(j, m, 1u); j < jStop; j++) // loop-cap-test: end
        c++;
    // A counted loop with a step the host sets (0 for a runaway: no pass).
    uint m2 = gOut[4];
    uint s = gOut[5];
    uint c2 = 0u;
    for (uint j = 0u, jStop = EB_END(j, m2, s); j < jStop; j += s) // loop-cap-test: step
        c2++;
    gOut[0] = n;
    gOut[2] = i;
    gOut[3] = c;
    gOut[6] = c2;
    ebLoopCapFlush();
}
