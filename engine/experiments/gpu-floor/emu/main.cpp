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
    std::vector<Lane> lanes;
    for (uint lid = 0; lid < LANES; lid++)
        lanes.push_back(floorFrame(*B(0, const Params*), B(1, float4*), B(2, const float4*), B(3, const uint4*), B(4, const uint4*),
            B(5, const float4*), B(6, const float4*), B(7, const float4*), B(8, float4*), B(9, const uint4*), B(10, const uint4*),
            B(11, float4*), B(12, float4*), B(13, const uint2*), B(14, const uint*), B(15, uint*), B(16, uint*), B(17, const uint4*),
            B(18, uint*), B(19, atomic_int*), B(20, int*), B(21, uint*), B(22, const uint*), B(23, const uint4*), B(24, const uint*),
            B(25, float4*), B(26, uint*), B(27, float4*), B(28, atomic_uint*), B(29, float4*), B(30, uint*), lid, 0u));
    unsigned long phases = 0;
    for (;;) {
        uint done = 0;
        for (uint k = 0; k < LANES; k++) {
            uint lid = reverse ? LANES - 1 - k : k;
            if (lanes[lid].h.done()) { done++; continue; }
            lanes[lid].h.resume();
        }
        if (done == LANES) break;
        if (done != 0) { fprintf(stderr, "emu: lanes left the kernel at different barriers\n"); return 6; }
        phases++;
    }
    f = fopen(argv[2], "wb");
    for (auto& v : b) { uint64_t n = v.size(); fwrite(&n, 8, 1, f); fwrite(v.data(), 1, n, f); }
    fclose(f);
    fprintf(stderr, "emu: %lu phases\n", phases);
    return 0;
}
