// Runs the generated kernel (kernel.inc: the host's #defines and floor.metal, its entry
// renamed to a coroutine) over a blob of the 31 buffers: emu <blob in> <blob out> [reverse]
// `reverse` runs the lanes of each phase in the other order: a phase whose result depends
// on its lanes' order (a race on the GPU) then gives another answer.
#include "shim.h"
#include "kernel.inc"

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    bool reverse = argc > 3 && atoi(argv[3]) != 0;
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 3;
    std::vector<std::vector<char>> b;
    for (;;) {
        uint64_t n;
        if (fread(&n, 8, 1, f) != 1) break;
        std::vector<char> v(n + 64);
        if (n && fread(v.data(), 1, n, f) != n) return 4;
        v.resize(n);
        b.push_back(std::move(v));
    }
    fclose(f);
    if (b.size() != 31) { fprintf(stderr, "emu: %zu buffers\n", b.size()); return 5; }
    #define B(i, T) ((T)(void*)b[i].data())
    // Every workgroup's lanes (the first parameter word: how many groups).
    const uint groups = *B(0, const uint*);
    std::vector<std::vector<Lane>> lanes(groups);
    for (uint g = 0; g < groups; g++)
    for (uint lid = 0; lid < LANES; lid++)
#ifdef V2
        lanes[g].push_back(floorFrame(*B(0, const Params*), B(1, uint*), B(2, const float4*), B(4, const uint4*), B(5, const float4*),
            B(6, const float4*), B(7, const float4*), B(12, const uint*), B(14, const uint*), B(17, const uint4*),
            B(22, const uint*), B(23, const uint4*), B(24, const uint*), lid, g));
#else
        lanes[g].push_back(floorFrame(*B(0, const Params*), B(1, float4*), B(2, const float4*), B(3, const uint4*), B(4, const uint4*),
            B(5, const float4*), B(6, const float4*), B(7, const float4*), B(8, float4*), B(9, const uint4*), B(10, const uint4*),
            B(11, float4*), B(12, float4*), B(13, const uint2*), B(14, const uint*), B(15, uint*), B(16, uint*), B(17, const uint4*),
            B(18, uint*), B(19, atomic_int*), B(20, int*), B(21, uint*), B(22, const uint*), B(23, const uint4*), B(24, const uint*),
            B(25, float4*), B(26, uint*), B(27, float4*), B(28, atomic_uint*), B(29, float4*), B(30, uint*), lid, g));
#endif
    // A group runs phase after phase until its lanes stop at a cross-group barrier (or
    // finish); when every group is there, all go on.
    unsigned long phases = 0;
    std::vector<int> state(groups, 0); // 0 running, 1 at the cross barrier, 2 done
    for (;;) {
        bool progressed = false;
        for (uint gi = 0; gi < groups; gi++) {
            // (`reverse` also runs the groups in the other order.)
            const uint g = reverse ? groups - 1 - gi : gi;
            while (state[g] == 0) {
                uint done = 0, atx = 0;
                for (uint k = 0; k < LANES; k++) {
                    uint lid = reverse ? LANES - 1 - k : k;
                    if (lanes[g][lid].h.done()) { done++; continue; }
                    g_at_x = false;
                    lanes[g][lid].h.resume();
                    if (lanes[g][lid].h.done()) done++;
                    else if (g_at_x) atx++;
                }
                phases++;
                progressed = true;
                if (done == LANES) state[g] = 2;
                else if (atx == LANES) state[g] = 1;
                else if (done != 0 || atx != 0) { fprintf(stderr, "emu: group %u: lanes at different barriers (%u done, %u at the cross barrier)\n", g, done, atx); return 6; }
            }
        }
        bool all_done = true, all_wait = true;
        for (uint g = 0; g < groups; g++) { if (state[g] != 2) all_done = false; if (state[g] == 0) all_wait = false; }
        if (all_done) break;
        if (all_wait) {
            bool any = false;
            for (uint g = 0; g < groups; g++) if (state[g] == 1) { state[g] = 0; any = true; }
            if (!any) break;
            // (A group that finished while others wait at a cross barrier: a mismatch.)
            for (uint g = 0; g < groups; g++) if (state[g] == 2) { fprintf(stderr, "emu: group %u finished while others wait at a cross barrier\n", g); return 7; }
        }
        if (!progressed && !all_wait) break;
    }
    f = fopen(argv[2], "wb");
    for (auto& v : b) { uint64_t n = v.size(); fwrite(&n, 8, 1, f); fwrite(v.data(), 1, n, f); }
    fclose(f);
    fprintf(stderr, "emu: %lu phases\n", phases);
    return 0;
}
