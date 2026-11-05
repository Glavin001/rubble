//! GPU frame time of the explicit node–beam solver on the hosts of the shared plan
//! (`engine_exec::explicit_plan`): native wgpu (`ExplicitBeamsGpu`) and native Metal
//! (`MetalExplicitBeams`), at 1, 16 and 64 procedural sedans (`RUBBLE_BENCH_CARS`
//! overrides, comma separated), parked 8 m apart on a grid with vehicle contact on,
//! 60 fps frames of the solver's fixed steps. Prints the median GPU ms per frame.
//!
//!   cargo run -p engine-metal --release --example explicit_bench
//!
//! Run one GPU job at a time on an otherwise idle machine.

use engine_metal::MetalExplicitBeams;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, FrameClock};
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::Vec3;

const WARMUP: u32 = 30;
const TIMED: u32 = 120;

fn fleet(n: usize) -> Vec<ExplicitBeams> {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let side = (n as f32).sqrt().ceil() as usize;
    (0..n)
        .map(|i| {
            let at = Vec3::new(8.0 * (i % side) as f32, 0.01 - low, 8.0 * (i / side) as f32);
            ExplicitBeams::from_truck(&text, at, Vec3::ZERO, settings).unwrap()
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

fn main() {
    let counts: Vec<usize> = std::env::var("RUBBLE_BENCH_CARS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 16, 64]);
    let (_, device, queue) = engine_webgpu::default_device().expect("wgpu device");
    println!("cars  nodes   beams   wgpu ms/frame  Metal ms/frame  (median GPU time, 60 fps frames)");
    for n in counts {
        let f = fleet(n);
        let nodes: usize = f.iter().map(|v| v.nodes.len()).sum();
        let beams: usize = f.iter().map(|v| v.beams.len()).sum();
        // wgpu
        let wgpu_ms = {
            let mut g = engine_webgpu::ExplicitBeamsGpu::new(&device, &queue, &f);
            let mut clock = FrameClock::default();
            for _ in 0..WARMUP {
                g.frame(clock.next());
            }
            g.wait();
            if g.enable_frame_timing(TIMED) {
                for _ in 0..TIMED {
                    g.frame(clock.next());
                }
                g.wait();
                median(g.frame_times_ms())
            } else {
                f64::NAN
            }
        };
        // Metal
        let metal_ms = {
            let mut g = MetalExplicitBeams::new(&f).expect("Metal");
            let mut clock = FrameClock::default();
            for _ in 0..WARMUP {
                g.frame(clock.next());
            }
            g.wait();
            g.enable_frame_timing(TIMED);
            for _ in 0..TIMED {
                g.frame(clock.next());
            }
            median(g.frame_times_ms())
        };
        println!("{n:>4}  {nodes:>6}  {beams:>6}  {wgpu_ms:>13.3}  {metal_ms:>14.3}");
    }
}
