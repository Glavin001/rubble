// CPU driver for the Slang convex entries (slangc -target cpp -DENGINE_CPU_TEST=1).
// Runs convex_classify -> convex_bucket -> convex_sat_{small,large} -> convex_manifold over a
// scene written by tests/convex_slang_cpu.rs and writes counters, contacts,
// manifolds and SAT work records back. Any structured-buffer index out of range
// aborts, so the harness also checks the kernels' memory bounds.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#define SLANG_BOUND_CHECK(index, count)                                                     \
    if ((size_t)(index) >= (size_t)(count))                                                 \
    {                                                                                       \
        std::fprintf(stderr, "out of bounds: %zu >= %zu\n", (size_t)(index), (size_t)(count)); \
        std::abort();                                                                       \
    }
#include CONVEX_GENERATED

template <typename T>
static void bind(RWStructuredBuffer<T>& b, std::vector<uint8_t>& storage, size_t count)
{
    storage.assign(count * sizeof(T) + sizeof(T), 0);
    b.data = reinterpret_cast<T*>(storage.data());
    b.count = count;
}

static void run(void (*entry)(ComputeVaryingInput*, void*, void*), GlobalParams_0* g, uint32_t gx, uint32_t gy)
{
    if (gx == 0 || gy == 0) return;
    ComputeVaryingInput in = {};
    in.endGroupID.x = gx;
    in.endGroupID.y = gy;
    in.endGroupID.z = 1;
    entry(&in, nullptr, g);
}

int main(int argc, char** argv)
{
    if (argc != 3) { std::fprintf(stderr, "usage: harness in.bin out.bin\n"); return 2; }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 3;
    std::vector<uint8_t> in;
    uint8_t buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) in.insert(in.end(), buf, buf + n);
    std::fclose(f);
    size_t at = 0;
    auto take = [&](void* dst, size_t bytes) { std::memcpy(dst, in.data() + at, bytes); at += bytes; };
    uint32_t head[8];
    take(head, sizeof head);
    if (head[0] != 0x31585643u) return 4; // "CVX1"
    uint32_t nBodies = head[1], nColliders = head[2], nCandidates = head[3], nHull = head[4], maxContacts = head[5];
    // mode 0: run the convex stages on the given candidates. mode 1: run the
    // whole collision front end (bounds, broadphase, narrowphase, convex) with
    // `nCandidates` as the candidate capacity and `head[7]` grid hash words.
    // mode 3: run the narrowphase (then the convex stages) on the given candidates.
    uint32_t mode = head[6], gridWords = head[7];

    GlobalParams_0 g = {};
    std::vector<uint8_t> sBodies, sColliders, sMaterials, sContacts, sParams, sCounters, sCandidates,
        sHulls, sScratch, sManifolds, sWork, sConstraintIds, sEvents, sIndirect, sObbs, sAabbs;
    bind(g.gParams_0, sParams, 1);
    take(&g.gParams_0[0], 80);
    bind(g.gScratchAtomics_0, sScratch, 96); // 64..79: uniform-grid large list
    take(&g.gScratchAtomics_0[0], 64 * 4);
    bind(g.gBodies_0, sBodies, nBodies);
    take(&g.gBodies_0[0], size_t(nBodies) * 112);
    bind(g.gColliders_0, sColliders, nColliders);
    take(&g.gColliders_0[0], size_t(nColliders) * 64);
    bind(g.gCandidates_0, sCandidates, nCandidates);
    if (mode == 0 || mode == 3) take(&g.gCandidates_0[0], size_t(nCandidates) * 16);
    bind(g.gHulls_0, sHulls, nHull);
    take(&g.gHulls_0[0], size_t(nHull) * 16);
    bind(g.gMaterials_0, sMaterials, 64);
    for (int i = 0; i < 64; i++) { g.gMaterials_0[i].friction_0 = 0.5f; g.gMaterials_0[i].contactOffset_0 = 0.01f; }
    bind(g.gContacts_0, sContacts, size_t(maxContacts) * 2);
    bind(g.gCounters_0, sCounters, 1);
    g.gCounters_0[0].bodyCount_0 = nBodies;
    g.gCounters_0[0].colliderCount_0 = nColliders;
    g.gCounters_0[0].candidateCount_0 = (mode == 0 || mode == 3) ? nCandidates : 0u;
    // Previous contacts live in the other half: [maxContacts, 2·maxContacts).
    bind(g.gManifolds_0, sManifolds, nCandidates);
    bind(g.gConvexWork_0, sWork, nCandidates);
    std::vector<uint8_t> sSatCache;
    // SAT separating-feature cache: next power of two >= max(1024, max candidates).
    size_t satSlots = 1024; while (satSlots < nCandidates) satSlots <<= 1;
    bind(g.gSatCache_0, sSatCache, satSlots);
    std::vector<uint8_t> sColorLists;
    bind(g.gConstraintIds_0, sConstraintIds, gridWords > (1u << 16) ? gridWords : (1u << 16));
    bind(g.gColorLists_0, sColorLists, nColliders + 64);
    bind(g.gEvents_0, sEvents, 65536);
    bind(g.gIndirectArgs_0, sIndirect, 80 * 3);
    bind(g.gWorldObbs_0, sObbs, nColliders);
    bind(g.gAabbs_0, sAabbs, nBodies + nColliders);

    if (mode == 3) run(narrowphase, &g, (nCandidates + 63) / 64, 1);
    if (mode == 1 || mode == 2)
    {
        run(build_aabbs, &g, (nColliders + 63) / 64, 1);
        run(broadphase_clear, &g, (gridWords + 63) / 64, 1);
        run(broadphase_insert, &g, (nColliders + 63) / 64, 1);
        run(broadphase_query, &g, (nColliders + 63) / 64, 1);
        uint32_t found = g.gCounters_0[0].candidateCount_0;
        run(narrowphase, &g, (found + 63) / 64, 1);
    }
    run(convex_classify, &g, (g.gCounters_0[0].candidateCount_0 + 63) / 64, 1);
    run(convex_bucket, &g, (g.gCounters_0[0].candidateCount_0 + 63) / 64, 1);
    uint32_t small = g.gCounters_0[0].convexSmall_0, large = g.gCounters_0[0].convexLarge_0;
    // Same workgroup geometry as the GPU (4 small / 1 large pairs per group);
    // the CPU entries emulate the 64 lanes phase by phase.
    run(convex_sat_small, &g, (small + 3) / 4, 1);
    run(convex_sat_large, &g, large, 1);
    run(convex_manifold, &g, (small + large + 63) / 64, 1);

    // mode 2: then reuse every manifold of that step (three-avbd reuseManifolds)
    // into the other contact half, as a host does on the next step.
    uint32_t prevContacts = g.gCounters_0[0].contactCount_0, prevManifolds = g.gCounters_0[0].manifoldCount_0;
    if (mode == 2)
    {
        g.gScratchAtomics_0[7] = prevContacts;
        g.gScratchAtomics_0[8] = g.gScratchAtomics_0[11];
        g.gScratchAtomics_0[11] = maxContacts - g.gScratchAtomics_0[11];
        g.gCounters_0[0].contactCount_0 = 0;
        g.gCounters_0[0].manifoldCount_0 = 0;
        run(reuse_all_manifolds, &g, (prevContacts + 63) / 64, 1);
    }

    FILE* o = std::fopen(argv[2], "wb");
    if (!o) return 5;
    GpuCounters_0 c = g.gCounters_0[0];
    std::fwrite(&c, sizeof(c), 1, o);
    uint32_t cur = g.gScratchAtomics_0[11];
    std::fwrite(&g.gContacts_0[cur], 128, c.contactCount_0, o);
    std::fwrite(&g.gManifolds_0[0], 32, c.manifoldCount_0, o);
    std::fwrite(&g.gConvexWork_0[0], 32, nCandidates, o);
    uint32_t found = c.candidateCount_0 < nCandidates ? c.candidateCount_0 : nCandidates;
    std::fwrite(&g.gCandidates_0[0], 16, found, o);
    std::fwrite(&g.gAabbs_0[nBodies], 32, nColliders, o);
    uint32_t tail[2] = {prevContacts, prevManifolds};
    std::fwrite(tail, 4, 2, o);
    if (mode == 2) std::fwrite(&g.gContacts_0[g.gScratchAtomics_0[8]], 128, prevContacts, o);
    std::fclose(o);
    return 0;
}
