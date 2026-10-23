//! Behaviour of the CPU reference solver on imported content: stability at rest,
//! convergence with iterations and step rate, plastic crush and breaking.

use engine_softbody::cpu::{CpuSoft, Plane};
use engine_softbody::procedural::{box_truck, car_truck, CarParams};
use engine_softbody::truck::load_truck;
use glam::Vec3;

fn ground() -> Plane {
    Plane {
        normal: Vec3::Y,
        offset: 0.0,
        friction: 0.8,
    }
}

fn car(velocity: Vec3, hz: f32, iterations: u32) -> CpuSoft {
    let d = load_truck(&car_truck(&CarParams::default()));
    let mut s = CpuSoft::from_desc(&d, Vec3::new(0.0, 0.05, 0.0), velocity);
    s.h = 1.0 / hz;
    s.iterations = iterations;
    s.planes.push(ground());
    s
}

#[test]
fn car_rests_on_its_nodes_without_exploding() {
    let mut s = car(Vec3::ZERO, 120.0, 12);
    for _ in 0..240 {
        s.step();
        assert!(s.x.iter().all(|p| p.is_finite()));
    }
    let ke = s.kinetic_energy();
    assert!(ke < 1.0, "kinetic energy at rest {ke}");
    assert_eq!(s.broken_beams(), 0);
    assert_eq!(s.yielded_beams(), 0, "resting on the ground must not yield");
    let lowest = s.x.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    assert!(lowest > 0.0 && lowest < 0.1, "lowest node {lowest}");
}

/// Convergence check (design: do first). Sag of a supported box under its own
/// weight must approach the fine reference as iterations and step rate grow.
#[test]
fn sag_converges_with_iterations_and_rate() {
    let d = load_truck(&box_truck(
        6,
        2,
        2,
        0.4,
        400.0,
        3.0e6,
        f32::INFINITY,
        f32::INFINITY,
    ));
    let run_rho = |hz: f32, iters: u32, rho: f32| {
        let mut s = CpuSoft::from_desc(&d, Vec3::ZERO, Vec3::ZERO);
        s.chebyshev_rho = rho;
        s.settings.plasticity = false;
        s.settings.breaking = false;
        s.h = 1.0 / hz;
        s.iterations = iters;
        // Clamp the x = 0 end: a cantilever.
        for i in 0..s.x.len() {
            if s.x[i].x < 0.01 {
                s.fixed[i] = true;
                s.v[i] = Vec3::ZERO;
            }
        }
        let rest = s.x.clone();
        for _ in 0..(hz as usize * 5) {
            s.step();
        }
        let tip =
            s.x.iter()
                .zip(&rest)
                .filter(|(_, r)| r.x > 1.9)
                .map(|(p, r)| r.y - p.y)
                .sum::<f32>()
                / 4.0;
        (tip, s.kinetic_energy())
    };
    let run = |hz: f32, iters: u32| run_rho(hz, iters, 0.0);
    let (fine, ke) = run(480.0, 60);
    assert!(fine > 0.0 && ke < 0.5, "reference sag {fine} ke {ke}");
    let mut last = f32::MAX;
    for iters in [4, 10, 30] {
        let (sag, _) = run(120.0, iters);
        let err = (sag - fine).abs() / fine;
        println!(
            "120 Hz, {iters:>2} iterations: sag {sag:.5} (fine {fine:.5}) error {:.1}%",
            err * 100.0
        );
        assert!(err <= last + 1e-3, "error must shrink with iterations");
        last = err;
    }
    let (sag240, _) = run(240.0, 10);
    let (sag60, _) = run(60.0, 10);
    println!("10 iterations: 60 Hz {sag60:.5}, 240 Hz {sag240:.5}");
    assert!(
        (sag240 - fine).abs() <= (sag60 - fine).abs() + 1e-5,
        "higher rate converges closer"
    );
    assert!(last < 0.35, "30 iterations within 35% of the reference");
    for rho in [0.5, 0.8, 0.9, 0.95] {
        let (sag, _) = run_rho(120.0, 10, rho);
        println!(
            "120 Hz, 10 iterations, Chebyshev ρ={rho}: sag {sag:.5} error {:.1}%",
            (sag - fine).abs() / fine * 100.0
        );
    }
}

#[test]
fn wall_crash_crumples_the_front() {
    let mut s = car(Vec3::new(0.0, 0.0, 15.0), 240.0, 12);
    // Wall facing −Z at z = 3.0, just ahead of the bumper.
    s.planes.push(Plane {
        normal: -Vec3::Z,
        offset: -3.0,
        friction: 0.5,
    });
    // Body length between the rear and front rings (their six shell nodes).
    let stations = CarParams::default().stations;
    let ring = engine_softbody::procedural::CAR_RING;
    let body_len = |x: &[Vec3]| {
        let c = |k: usize| (0..6).map(|j| x[k * ring + j]).sum::<Vec3>() / 6.0;
        (c(stations - 1) - c(0)).length()
    };
    let len0 = body_len(&s.x);
    for _ in 0..240 {
        s.step();
        assert!(s.x.iter().all(|p| p.is_finite()), "solver blew up");
    }
    let yielded = s.yielded_beams();
    let broken = s.broken_beams();
    let crush = len0 - body_len(&s.x);
    println!(
        "yielded {yielded}, broken {broken}, crush {crush:.3} m, ke {}",
        s.kinetic_energy()
    );
    assert!(
        yielded > 50,
        "a 54 km/h wall hit must crumple the front ({yielded})"
    );
    // Pushed straight back, the bumper mounts carry only their yield force
    // (yield is applied inside the solve), so they crush rather than snap.
    let _ = broken;
    assert!(
        s.kinetic_energy() < 0.1 * 0.5 * 1300.0 * 225.0,
        "most energy absorbed"
    );
    assert!(
        crush > 0.25,
        "54 km/h folds the front by tens of cm ({crush})"
    );
}
