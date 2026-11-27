//! `pair.<a>_<b>.com_order_min` on a hand-built run.

use super::*;

#[test]
fn crash_metrics_com_order_of_a_pair() {
    // Car heading +Z (COM z = 1), barrier heading −Z ahead of it (COM z = 3 at the start).
    let car = desc(vec![node("a", [0.0, 0.5, 0.0], 1.0), node("b", [0.0, 0.5, 2.0], 1.0)], &[(0, 1)]);
    let bar = desc(vec![node("a", [0.0, 0.5, 0.5], 1.0), node("b", [0.0, 0.5, -0.5], 1.0)], &[(0, 1)]);
    let (t, b) = scene(&[("car", &car, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO), ("barrier", &bar, ZoneMap::default(), Vec3::new(0.0, 0.0, 3.0), std::f32::consts::PI, Vec3::ZERO)], vec![]);
    let x0 = sample(&b, 0.0);
    let com = |x: &Sample, k: usize| x.pos[k].iter().map(|p| f64::from(p.z)).sum::<f64>() / x.pos[k].len() as f64;
    let gap0 = com(&x0, 1) - com(&x0, 0);
    // Then the barrier is carried 2.5 m back past the car's centre of mass, and returns.
    let mut x1 = x0.clone();
    x1.t = 0.004;
    x1.pos[1].iter_mut().for_each(|p| p.z -= 2.5);
    let mut x2 = x0.clone();
    x2.t = 0.008;
    let m = run(&t, &b, &[x0, x1.clone(), x2]);
    let least = com(&x1, 1) - com(&x1, 0);
    assert!(least < 0.0 && gap0 > 0.0);
    // Along the car's heading (+Z): barrier − car; along the barrier's (−Z): car − barrier, the same number.
    near(&m, "pair.car_barrier.com_order_min", least, 1e-5);
    near(&m, "pair.barrier_car.com_order_min", least, 1e-5);
}
