// Loop caps, in the source so every target has them (WGSL, MSL, CUDA). A kernel
// loop that never ends holds the GPU (on Apple Silicon it can stall the window
// server and the whole machine), so no loop of the explicit kernels runs unbounded:
// * a loop with a constant bound (a literal or a static const, a constant step)
//   is finite as written;
// * a counted loop `for (uint i = a; i < n; i += s)` with n and s loop-invariant
//   takes its end once: `for (uint i = a, iStop = EB_END(i, n, s); i < iStop;
//   i += s)`, n clamped to at most EB_LOOP_CAP passes from a and below the
//   counter's wrap (a zero step: no pass);
// * any other loop counts its passes, `EB_PASS(guard)` first in its body.
// EB_LOOP_CAP is far above any legitimate count, so a capped loop computes what
// an uncapped one does. A loop held to its cap (a clamped end, or a counter past
// the cap) raises the sticky flag gAcc[EB_LOOP_CAP_WORD] (atomic max of
// EB_LOOP_CAP_TAG | its source line, where the entry point ends: ebLoopCapFlush)
// and the host reports the frame as an error
// (engine_exec::explicit_plan::explicit_loop_cap_hit). engine-webgpu's
// tests/explicit_loop_cap.rs checks every loop of explicit_beams.slang and its
// includes has one of the three forms; explicit_loop_cap_test.slang runs a runaway
// on each target (engine-webgpu, engine-metal, engine-cuda tests explicit_loop_cap).
//
// The including file declares `gAcc` (RWStructuredBuffer<Atomic<int>>) and defines
// EB_LOOP_CAP_WORD, the flag's index in it (explicit_beams.slang: gP.loopCapWord,
// gAcc's last word).
static const uint EB_LOOP_CAP = 1u << 20;
static const uint EB_LOOP_CAP_TAG = 1u << 24;
// The stopped loop (the highest line), kept per invocation and written to gAcc
// once, where the entry point ends (a flag write at each loop changes the
// kernels' codegen: Metal's fast math then rounds differently).
static uint ebLoopHit = 0u;
void ebLoopCapHit(uint line)
{
    ebLoopHit = max(ebLoopHit, EB_LOOP_CAP_TAG | line);
}
// Every entry point with a capped loop ends with this: `ebLoopCapFlush();` last
// and `return ebLoopCapFlush();` for each return.
void ebLoopCapFlush()
{
    if (ebLoopHit != 0u)
        gAcc[EB_LOOP_CAP_WORD].max(int(ebLoopHit));
}
uint ebEnd(uint start, uint bound, uint step, uint line)
{
    uint end = min(bound, max(start + min(step, 0xFFFFFFFFu / EB_LOOP_CAP) * EB_LOOP_CAP, start));
    end = min(end, 0xFFFFFFFFu - max(step, 1u) + 1u);
    if (end < bound)
        ebLoopCapHit(line);
    return end;
}
#define EB_END(start, bound, step) ebEnd(start, bound, step, __LINE__)
#define EB_PASS(guard) if (++guard > EB_LOOP_CAP) { ebLoopCapHit(__LINE__); break; }
