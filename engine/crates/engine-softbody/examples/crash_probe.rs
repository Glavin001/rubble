//! Print a CPU-reference wall crash of the procedural car over time.
//! `cargo run --release -p engine-softbody --example crash_probe -- [speed m/s] [Hz] [iterations]`
use engine_softbody::cpu::{CpuSoft, Plane};
use engine_softbody::procedural::{car_truck, CarParams};
use engine_softbody::truck::load_truck;
use glam::Vec3;
fn main() {
    let args: Vec<f32> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (speed, hz, iters) = (
        args.first().copied().unwrap_or(15.0),
        args.get(1).copied().unwrap_or(240.0),
        args.get(2).copied().unwrap_or(12.0) as u32,
    );
    let deform = args.get(4).copied().unwrap_or(CarParams::default().deform);
    let d = load_truck(&car_truck(&CarParams {
        deform,
        ..Default::default()
    }));
    let mut s = CpuSoft::from_desc(&d, Vec3::new(0.0, 0.05, 0.0), Vec3::new(0.0, 0.0, speed));
    s.h = 1.0 / hz;
    s.iterations = iters;
    s.chebyshev_rho = args.get(3).copied().unwrap_or(0.0);
    s.planes.push(Plane {
        normal: Vec3::Y,
        offset: 0.0,
        friction: 0.8,
    });
    s.planes.push(Plane {
        normal: -Vec3::Z,
        offset: -3.0,
        friction: 0.5,
    });
    let n = CarParams::default().stations;
    let len = |s: &CpuSoft| {
        let c = |k: usize| (0..6).map(|j| s.x[k * 6 + j]).sum::<Vec3>() / 6.0;
        (c(n - 1) - c(0)).length()
    };
    let l0 = len(&s);
    for k in 0..(hz as usize) {
        let r = s.step();
        if k % (hz as usize / 20) == 0 {
            let vz: f32 = s.v.iter().zip(&s.mass).map(|(v, m)| v.z * m).sum::<f32>()
                / s.mass.iter().sum::<f32>();
            let maxf = s.beams.iter().map(|b| b.force.abs()).fold(0.0, f32::max);
            println!(
                "t {:.3} len {:.3} vz {:.2} ke {:.0} yielded {} broke {} total_y {} maxF {:.0}",
                k as f32 / hz,
                len(&s),
                vz,
                s.kinetic_energy(),
                r.yielded,
                r.broke,
                s.yielded_beams(),
                maxf
            );
        }
    }
    println!("crush {:.3} m", l0 - len(&s));
}
