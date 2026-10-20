//! The P0 cost probe computes the network it claims to (so its timings are of real work):
//! the GPU's outputs match a CPU evaluation of the same random-weight network, at every
//! width, through the hierarchy (pool, coarse rounds, unpool), on a batch of two islands.

use stress_gpu::gnn_probe::{bistride, Graph, Precision, Probe, RoundKernel};
use stress_gpu::gpu::Gpu;

#[test]
fn probe_matches_its_cpu_reference() {
    let gpu = Gpu::new().expect("GPU");
    let g = Graph::union(&[Graph::grid(4, 2, 3), Graph::grid(3, 1, 5)]);
    for (w, split) in [32, 64, 128].into_iter().flat_map(|w| [RoundKernel::Fused, RoundKernel::Split, RoundKernel::Tiled].map(move |k| (w, k))) {
        let probe = Probe::with_kernel(&gpu, &g, w, Precision::F32, &[2, 1, 1, 1, 2], split);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        probe.encode(&mut enc);
        gpu.queue.submit([enc.finish()]);
        let got = probe.outputs(&gpu);
        let want = probe.reference();
        assert_eq!(got.len(), want.len());
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0f32, f32::max);
        assert!(got.iter().all(|x| x.is_finite()), "W {w} {split:?}: non-finite outputs");
        assert!(worst < 2e-3, "W {w} {split:?}: GPU and CPU differ by {worst:.3e} (relative)");
    }
}

#[test]
fn bistride_keeps_every_other_level_and_covers_every_node() {
    let g = Graph::grid(6, 1, 5);
    let c = bistride(&g);
    assert!(c.coarse.nodes > 0 && c.coarse.nodes < g.nodes);
    assert_eq!(c.parent.len(), g.nodes);
    assert_eq!(c.members.len(), g.nodes, "each fine node belongs to exactly one coarse node");
    // A connected fine graph stays connected: BFS over the coarse graph reaches every node.
    let mut seen = vec![false; c.coarse.nodes];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(n) = stack.pop() {
        for &m in &c.coarse.src[c.coarse.offsets[n] as usize..c.coarse.offsets[n + 1] as usize] {
            if !seen[m as usize] {
                seen[m as usize] = true;
                stack.push(m as usize);
            }
        }
    }
    assert!(seen.iter().all(|&s| s));
}
