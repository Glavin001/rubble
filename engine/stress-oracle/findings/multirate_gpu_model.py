#!/usr/bin/env python3
"""Frame-time model of multirate on the GPU (findings/multirate.md, "GPU design"). A model,
not a measurement: each substep is one launch (dispatch, graph node or persistent-kernel
barrier) of latency L plus the bonds and chunks it updates at the device's throughput.

    python3 findings/multirate_gpu_model.py

Inputs, all measured on the M3 Max (REALTIME_ROADMAP.md, stress-gpu PLAN.md):
- C_BOND: a fully active 24k-chunk island (78,196 bonds) costs 139 us per substep (G8),
  of which about 24 us is the full model's per-substep floor (A6): 1.47 ns per bond;
- CHUNK_PER_BOND: a chunk update (frame loads and integration) costs 0.58 of a bond
  evaluation (stress-ref profile, house-1story rammed, single thread);
- per pack: bonds, chunks, substeps per frame at A1's true step, and the bonds per rate and
  the chunk-update ratio of multirate's rule at that step (`stress-gpu-steps --levers`).
Multirate updates a chunk at its finest bond's rate (the design's chunk kernels), so the
chunk work falls by the chunk-update ratio. One house is latency-bound (the floor sets
every substep, SeisSol's 1.3x case); batched buildings and large packs are work-bound.
"""

C_BOND = 1.47e-3       # us per bond evaluation at full occupancy
CHUNK_PER_BOND = 0.58  # chunk update over bond evaluation

# name, bonds, chunks, substeps per frame (true step), {rate: bonds}, chunk-update ratio
PACKS = [
    ('house-1story', 877, 319, 1264, {1: 252, 2: 98, 4: 351, 8: 176}, 1.40),
    ('house-2story', 1652, 599, 1358, {1: 443, 2: 218, 4: 586, 8: 401, 16: 4}, 1.50),
    ('villa-savoye', 2464, 1045, 931, {1: 1062, 2: 611, 4: 438, 8: 353}, 1.31),
    ('minas-tirith', 169338, 49049, 6766, {1: 1513, 2: 4237, 4: 7921, 8: 12797, 16: 12633, 32: 51665, 64: 73744, 128: 4782, 256: 46}, 9.14),
]


def single(n, bonds, chunks, latency, copies):
    """Frame time (ms), every bond and chunk at every substep."""
    return n * (latency + (bonds + chunks * CHUNK_PER_BOND) * copies * C_BOND) / 1000.0


def multirate(n, rates, chunks, chunk_ratio, latency, copies):
    """Frame time (ms): substep i launches the bonds of every rate dividing i (a prefix of
    the level-ordered buckets, all buildings batched) and the chunks at their rates."""
    chunk_work = chunks * CHUNK_PER_BOND * copies / chunk_ratio
    t = 0.0
    for i in range(n):
        fired = sum(b for r, b in rates.items() if i % r == 0) * copies
        t += latency + (fired + chunk_work) * C_BOND
    return t / 1000.0


def main():
    print('| Pack | Buildings | Launch latency | Single rate | Multirate | Gain |')
    print('|---|---|---|---|---|---|')
    for name, bonds, chunks, n, rates, chunk_ratio in PACKS:
        for copies in ([1, 10, 100] if bonds < 10000 else [1]):
            for latency in [24.0, 5.0, 2.0]:
                s = single(n, bonds, chunks, latency, copies)
                m = multirate(n, rates, chunks, chunk_ratio, latency, copies)
                print('| %s | %d | %.0f us | %.1f ms | %.1f ms | x%.2f |' % (name, copies, latency, s, m, s / m))


if __name__ == '__main__':
    main()
