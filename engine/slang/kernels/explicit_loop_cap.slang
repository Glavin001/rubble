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
    {
#ifdef EB_LOOP_CAP_STORE
        // An including file with its own status word (vehicle_coupling.slang).
        EB_LOOP_CAP_STORE(ebLoopHit);
#else
        gAcc[EB_LOOP_CAP_WORD].max(int(ebLoopHit));
#endif
    }
}
// The pass budget (an including file that defines EB_LOOP_BUDGET, the explicit frame
// kernels: gP.loopBudget, the host's bound on one invocation's legitimate passes in a
// frame; 0 none). The caps above bound each loop, but nested loops multiply them, so
// a corrupt count could hold the GPU for minutes. Every budgeted loop draws its passes
// from one per-invocation count (a register): EB_END its granted passes, EB_PASS one a
// pass. Past the budget the loops end at once and the sticky flag above is raised, so
// a bad state fails its frame in milliseconds. A loop that reaches a barrier must end
// alike on every lane, so it takes EB_END_UNIFORM / EB_PASS_UNIFORM, capped as above
// but not budgeted (engine/tools/explicit-2khz/barrier_loops.py lists them). The
// budget is far above any legitimate count, so a budgeted loop computes what an
// uncapped one does.
static uint ebSpent = 0u;
uint ebEnd(uint start, uint bound, uint step, uint line, bool budgeted)
{
    uint end = min(bound, max(start + min(step, 0xFFFFFFFFu / EB_LOOP_CAP) * EB_LOOP_CAP, start));
    end = min(end, 0xFFFFFFFFu - max(step, 1u) + 1u);
#ifdef EB_LOOP_BUDGET
    uint budget = EB_LOOP_BUDGET;
    if (budgeted && budget != 0u && end > start)
    {
        uint s = max(step, 1u);
        uint passes = (end - start - 1u) / s + 1u;
        uint left = budget - min(ebSpent, budget);
        if (passes > left)
        {
            end = start + left * s;
            ebSpent = budget;
        }
        else
            ebSpent += passes;
    }
#endif
    if (end < bound)
        ebLoopCapHit(line);
    return end;
}
// One pass of a counted loop from the budget (false: none left).
bool ebSpend()
{
#ifdef EB_LOOP_BUDGET
    uint budget = EB_LOOP_BUDGET;
    if (budget != 0u)
    {
        if (ebSpent >= budget) return false;
        ebSpent += 1u;
    }
#endif
    return true;
}
// The uniform form's end: the cap only, never the per-invocation budget. Its own
// function so its value depends on its arguments alone: Tint's uniformity analysis
// (Chrome) is not path-sensitive, and an end read through ebEnd (which reads ebSpent,
// a per-invocation value) is "possibly non-uniform", which fails every barrier the
// loop reaches. Same end as ebEnd(.., false).
uint ebEndUniform(uint start, uint bound, uint step, uint line)
{
    uint end = min(bound, max(start + min(step, 0xFFFFFFFFu / EB_LOOP_CAP) * EB_LOOP_CAP, start));
    end = min(end, 0xFFFFFFFFu - max(step, 1u) + 1u);
    if (end < bound)
        ebLoopCapHit(line);
    return end;
}
#define EB_END(start, bound, step) ebEnd(start, bound, step, __LINE__, true)
#define EB_END_UNIFORM(start, bound, step) ebEndUniform(start, bound, step, __LINE__)
#define EB_PASS(guard) if (++guard > EB_LOOP_CAP || !ebSpend()) { ebLoopCapHit(__LINE__); break; }
#define EB_PASS_UNIFORM(guard) if (++guard > EB_LOOP_CAP) { ebLoopCapHit(__LINE__); break; }
