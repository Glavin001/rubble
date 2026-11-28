//! The stability guard's tests (`stability.rs`).

use super::*;

const MASS: f64 = 1450.0;

/// A 1,450 kg car: centre of mass at `speed` m/s and `height` m, `size` m across,
/// with `extra` joules of kinetic energy besides (vibration, debris).
fn car(speed: f32, height: f64, size: f32, extra: f64) -> Sample {
    Sample {
        finite: true,
        mass: MASS,
        height,
        speed,
        size,
        kinetic: 0.5 * MASS * f64::from(speed) * f64::from(speed) + extra,
    }
}

/// The report of 2026-11-14 (vehicle lab, mountain, generated sedan): the car
/// stops on the slope (3 kJ), then tumbles on down it and leaves a torn-off part
/// behind, so its box is 104 m across at 441 kJ. The fall since its low (138 m of
/// it) explains that energy many times over: not a trip.
#[test]
fn a_part_left_behind_on_a_long_descent_does_not_trip() {
    let mut g = StabilityGuard::new(1);
    // Launched at 12 m/s, it comes to rest 20 m down, then tumbles 118 m further.
    g.launched(12.0);
    let mut t = 0.0;
    for k in 0..=60 {
        t = k as f64 / 60.0;
        let s = 12.0 * (1.0 - k as f32 / 60.0);
        assert_eq!(g.observe(t, &[car(s, 0.68 - 20.0 * k as f64 / 60.0, 5.39, 3.0e3)]), None, "t = {t}");
    }
    for k in 1..=660 {
        t = 1.0 + k as f64 / 60.0;
        let f = k as f64 / 660.0;
        // The part comes off a quarter of the way down and stays where it fell.
        let size = if f < 0.25 { 5.39 } else { 5.39 + (f - 0.25) as f32 * 132.0 };
        let sample = car((20.6 * f) as f32, -19.32 - 118.63 * f, size, 3.0e3 + 131.0e3 * f);
        assert_eq!(g.observe(t, &[sample]), None, "t = {t}, {size} m across");
    }
}

/// The same descent, but the car is 104 m across with far more energy than its
/// fall since the low can have given it: that still trips.
#[test]
fn a_blow_up_on_a_descent_still_trips() {
    let mut g = StabilityGuard::new(1);
    assert_eq!(g.observe(0.0, &[car(0.0, 0.68, 5.39, 3.0e3)]), None);
    // 5 m lower (71 kJ of fall), 104 m across, 2,000 kJ.
    let trip = g.observe(1.0, &[car(0.0, -4.32, 104.0, 2.0e6)]).cloned();
    assert!(trip.is_some_and(|t| t.reason.contains("104 m across")), "no trip");
}

/// The report of 2026-11-16 (vehicle lab, mountain, eight generated sedans): a car
/// slowed by a crash (47 kJ), 104 m across with parts left behind, is struck by a
/// faster one and leaves at 568 kJ. Its own fall does not explain the rise; the
/// other car lost that energy. Not a trip.
#[test]
fn a_car_struck_by_another_does_not_trip() {
    let mut g = StabilityGuard::new(2);
    assert_eq!(g.observe(0.0, &[car(8.0, -100.0, 5.39, 0.0), car(40.0, -100.0, 5.39, 0.0)]), None);
    // A second later, at the same height: the struck car at 28 m/s and 104 m
    // across, the striking car slowed to 20 m/s.
    assert_eq!(g.observe(1.0, &[car(28.0, -100.0, 104.4, 0.0), car(20.0, -100.0, 5.39, 0.0)]), None);
}

/// The same two cars, but the second keeps its speed: the first car's rise is
/// energy from nowhere, and still trips.
#[test]
fn a_blow_up_beside_another_car_still_trips() {
    let mut g = StabilityGuard::new(2);
    assert_eq!(g.observe(0.0, &[car(8.0, -100.0, 5.39, 0.0), car(40.0, -100.0, 5.39, 0.0)]), None);
    let trip = g.observe(1.0, &[car(28.0, -100.0, 104.4, 0.0), car(40.0, -100.0, 5.39, 0.0)]).cloned();
    assert!(trip.is_some_and(|t| t.reason.contains("104 m across")), "no trip");
}

/// The report of 2026-11-24 (vehicle lab, drive, four generated sedans, `surface=1`):
/// the player's car, parked at the start, is driven up to speed, hits a pole and
/// leaves a torn-off part there (heavier than the debris limit, so it stays), then
/// drives on at 66 km/h until its box is 104 m across at 302 kJ. The tyres gave it
/// that energy over the drive: not a trip. (The box check counted its rise from
/// the lowest energy since the start, 0 kJ parked, with no drive allowance.)
#[test]
fn a_part_left_behind_while_driving_on_does_not_trip() {
    let mut g = StabilityGuard::new(4);
    let parked = car(0.0, 0.38, 5.39, 0.0);
    let mut t = 0.0;
    let mut observe = |g: &mut StabilityGuard, t: f64, speed: f32, size: f32, extra: f64| {
        let s = [car(speed, 0.48, size, extra), parked, parked, parked];
        assert_eq!(g.observe(t, &s), None, "t = {t:.3}, {speed} m/s, {size} m across");
    };
    // Parked, then 10 s up to 20 m/s.
    for k in 0..=600 {
        t = k as f64 / 60.0;
        observe(&mut g, t, 20.0 * k as f32 / 600.0, 5.39, 0.0);
    }
    // The pole: 20 → 5 m/s in 0.2 s, crush vibration.
    for k in 1..=12 {
        t += 1.0 / 60.0;
        observe(&mut g, t, 20.0 - 15.0 * k as f32 / 12.0, 5.6, 20.0e3);
    }
    // Back up to 18.4 m/s (66 km/h) over 4 s and on: the part stays at the pole,
    // so the box grows with the distance driven, to 104 m.
    let mut dist = 0.0f32;
    let mut speed = 5.0f32;
    while dist < 99.0 {
        t += 1.0 / 60.0;
        speed = (speed + 13.4 / 240.0).min(18.4);
        dist += speed / 60.0;
        observe(&mut g, t, speed, 5.6 + dist, 58.0e3);
    }
}

/// The report of 2026-11-24 at 151 s (vehicle lab, drive, three generated sedans,
/// `surface=1`): the player's car, 144 m across with a torn-off part left behind,
/// accelerates at 0.55 g from 21 to 36 km/h (24 to 74 kJ inside the window), its
/// energy all the whole car's motion. Its tyres did that: not a trip.
#[test]
fn accelerating_away_from_a_part_left_behind_does_not_trip() {
    let mut g = StabilityGuard::new(3);
    let parked = car(0.0, 0.38, 5.4, 0.0);
    let mut t = 0.0;
    let mut speed = 0.0f32;
    // Driven about at 5 to 10 m/s for two minutes, the part torn off early.
    for k in 0..(120 * 60) {
        t = k as f64 / 60.0;
        speed = 7.5 + 2.5 * (t * 0.3).sin() as f32;
        let size = if t < 10.0 { 5.4 } else { 104.0 + 40.0 * (t / 120.0) as f32 };
        assert_eq!(g.observe(t, &[car(speed, 0.48, size, 2.0e3), parked, parked]), None, "t = {t:.3}");
    }
    // Slowed to 5.8 m/s, then 0.55 g up to 10.1 m/s, 144 m across.
    let target = 5.76f32;
    while speed > target {
        t += 1.0 / 60.0;
        speed = (speed - 0.1).max(target);
        assert_eq!(g.observe(t, &[car(speed, 0.48, 144.0, 2.0e3), parked, parked]), None, "t = {t:.3}");
    }
    while speed < 10.1 {
        t += 1.0 / 60.0;
        speed += 5.4 / 60.0;
        assert_eq!(g.observe(t, &[car(speed, 0.48, 144.0, 2.0e3), parked, parked]), None, "t = {t:.3}, {speed} m/s");
    }
}

/// The same car, 144 m across, with the rise in its parts' motion instead (the
/// centre at the same speed, 200 kJ of internal energy from nowhere): that trips.
#[test]
fn internal_energy_rising_past_the_box_still_trips() {
    let mut g = StabilityGuard::new(1);
    let mut t = 0.0;
    for k in 0..=60 {
        t = k as f64 / 60.0;
        assert_eq!(g.observe(t, &[car(6.0, 0.48, 144.0, 2.0e3)]), None);
    }
    let mut tripped = false;
    for k in 1..=30 {
        t += 1.0 / 60.0;
        if g.observe(t, &[car(6.0, 0.48, 144.0, 2.0e3 + 1.0e4 * k as f64)]).is_some() {
            tripped = true;
            break;
        }
    }
    assert!(tripped, "no trip");
}

/// The same drive, but at 66 km/h the car goes from 300 kJ to 2,000 kJ and 104 m
/// across in one readback: that still trips.
#[test]
fn a_blow_up_while_driving_still_trips() {
    let mut g = StabilityGuard::new(1);
    let mut t = 0.0;
    for k in 0..=600 {
        t = k as f64 / 60.0;
        assert_eq!(g.observe(t, &[car(18.4 * k as f32 / 600.0, 0.48, 5.39, 0.0)]), None);
    }
    t += 1.0 / 60.0;
    let trip = g.observe(t, &[car(18.4, 0.48, 104.0, 1.75e6)]).cloned();
    assert!(trip.is_some(), "no trip");
}

/// `n` nodes on a line `length` long, all at velocity `v`, 10 kg each.
fn nodes(n: usize, length: f32, v: [f32; 3]) -> Vec<u32> {
    (0..n)
        .flat_map(|i| {
            let x = length * i as f32 / (n - 1) as f32;
            [x, 0.5, 0.0, 10.0, v[0], v[1], v[2], 0.0, 0.0, 0.0, 0.0, 0.0].map(f32::to_bits)
        })
        .collect()
}

#[test]
fn sample_from_nodes() {
    let s = Sample::from_nodes(2.0, &nodes(5, 4.0, [30.0, 0.0, 40.0]));
    assert!(s.finite);
    assert!((s.speed - 50.0).abs() < 1e-3);
    assert!((s.size - 4.0).abs() < 1e-5);
    assert!((s.kinetic - 0.5 * 50.0 * 2500.0).abs() < 1e-6);
    assert!((s.mass - 50.0).abs() < 1e-9 && (s.height - 2.5).abs() < 1e-6);
    for (word, bad) in [(12 + 5, f32::NAN), (24, f32::INFINITY), (3, f32::NAN)] {
        let mut w = nodes(5, 4.0, [0.0; 3]);
        w[word] = bad.to_bits();
        assert!(!Sample::from_nodes(0.0, &w).finite, "word {word}");
    }
    assert!(!Sample::from_nodes(f32::NAN, &nodes(5, 4.0, [0.0; 3])).finite);
}

/// A 200 km/h pole crash as the explicit solver's runs look: the car stops in
/// 80 ms, parts fly 20 m, and the kinetic energy only falls. Then 10 s at rest.
#[test]
fn a_healthy_crash_at_200_kmh_does_not_trip() {
    let mut g = StabilityGuard::new(1);
    let v0 = 200.0 / 3.6;
    for frame in 0..660 {
        let t = frame as f64 / 60.0;
        let crash = ((t - 0.5) / 0.08).clamp(0.0, 1.0) as f32;
        let speed = v0 * (1.0 - crash);
        // A tenth of the lost energy in debris and vibration for a while.
        let extra = if crash > 0.0 && t < 1.5 { 0.1 * 0.5 * MASS * f64::from(v0 * v0) } else { 0.0 };
        let size = 5.0 + 20.0 * ((t - 0.5).max(0.0) as f32).min(1.0);
        assert_eq!(g.observe(t, &[car(speed, 0.6, size, extra)]), None, "t = {t}");
        g.damage(0, (600.0 * crash) as u32, (1900.0 * crash) as u32, 4228);
    }
    assert!(g.trip().is_none());
}

/// No speed limit: a valid run at any speed, a launch, driving and a long fall
/// all stay inside the budget.
#[test]
fn fast_runs_launches_driving_and_falls_do_not_trip() {
    // Spawned at 3,000 km/h, coasting.
    let mut g = StabilityGuard::new(1);
    for k in 0..100 {
        assert_eq!(g.observe(k as f64 / 60.0, &[car(3000.0 / 3.6, 0.6, 5.0, 0.0)]), None);
    }
    // At rest, then the page launches it at 1,000 km/h.
    let mut g = StabilityGuard::new(1);
    assert_eq!(g.observe(0.0, &[car(0.0, 0.6, 5.0, 0.0)]), None);
    g.launched(1000.0 / 3.6);
    for k in 1..100 {
        assert_eq!(g.observe(k as f64 / 60.0, &[car(1000.0 / 3.6, 0.6, 5.0, 0.0)]), None);
    }
    // 0 to 300 km/h in 10 s under its own power (700 kW at the end).
    let mut g = StabilityGuard::new(1);
    for k in 0..=600 {
        let t = k as f64 / 60.0;
        assert_eq!(g.observe(t, &[car((300.0 / 3.6 * t / 10.0) as f32, 0.6, 5.0, 0.0)]), None, "t = {t}");
    }
    // Off a 500 m cliff from rest: free fall, then the landing throws parts 40 m.
    let mut g = StabilityGuard::new(1);
    for k in 0..=660 {
        let t = k as f64 / 60.0;
        let fall = (0.5 * GRAVITY * t * t).min(500.0);
        let speed = if fall < 500.0 { (GRAVITY * t) as f32 } else { 0.0 };
        let size = if fall < 500.0 { 5.0 } else { 45.0 };
        assert_eq!(g.observe(t, &[car(speed, 500.6 - fall, size, 0.0)]), None, "t = {t}");
    }
    // A ram: car 0 at 200 km/h hits car 1 at rest; car 1 leaves at 150 km/h.
    let mut g = StabilityGuard::new(2);
    for k in 0..120 {
        let hit = k >= 30;
        let cars = [
            car(if hit { 40.0 / 3.6 } else { 200.0 / 3.6 }, 0.6, 5.0, 0.0),
            car(if hit { 150.0 / 3.6 } else { 0.0 }, 0.6, if hit { 70.0 } else { 5.0 }, 0.0),
        ];
        assert_eq!(g.observe(k as f64 / 60.0, &cars), None, "frame {k}");
    }
}

#[test]
fn a_nan_trips_at_once_and_for_good() {
    let mut g = StabilityGuard::new(1);
    assert_eq!(g.observe(0.0, &[Sample::from_nodes(0.0, &nodes(8, 4.0, [50.0, 0.0, 0.0]))]), None);
    let mut w = nodes(8, 4.0, [50.0, 0.0, 0.0]);
    w[3 * 12 + 1] = f32::NAN.to_bits();
    let trip = g.observe(1.05, &[Sample::from_nodes(0.0, &w)]).cloned().expect("tripped");
    assert!(trip.reason.contains("not a finite number"), "{}", trip.reason);
    let message = trip.message();
    assert!(message.starts_with("ENGINE ERROR: instability detected at t = 1.050 s"), "{message}");
    assert!(message.contains("This is a simulation bug") && message.contains("vehicle 0: finite false"), "{message}");
    // Final: a healthy sample afterwards changes nothing.
    assert_eq!(g.observe(1.1, &[car(50.0, 0.6, 4.0, 0.0)]), Some(&trip));
}

/// The 2026-11-13 side pole run: launched at 186 km/h; the HUD read 4,861 km/h
/// with 373 broken and 2,017 bent of 4,228 beams when the device was lost.
#[test]
fn energy_from_nowhere_trips() {
    let mut g = StabilityGuard::new(1);
    let v0 = 186.0 / 3.6;
    for k in 0..60 {
        assert_eq!(g.observe(k as f64 / 60.0, &[car(v0, 0.6, 5.0, 0.0)]), None);
    }
    g.damage(0, 373, 2017, 4228);
    // The energy grows 1.6 × a frame from the impact on (4,861 km/h is 680 × the start's).
    let mut energy = 0.5 * MASS * f64::from(v0 * v0);
    let mut tripped = None;
    for k in 60..80 {
        energy *= 1.6;
        let t = k as f64 / 60.0;
        if let Some(trip) = g.observe(t, &[car(0.0, 0.6, 6.0, energy)]) {
            tripped = Some((k, energy, trip.clone()));
            break;
        }
    }
    let (k, energy, trip) = tripped.expect("tripped");
    // The rebound check sees it first (nothing was lost, so nothing may come back):
    // tripped at the 3rd frame of the growth, at 4.1 × the start's energy, long
    // before 680 ×. (The whole-run budget, 3 × 1,935 kJ + the drive allowance +
    // the floor, would have tripped 2 frames later.)
    assert_eq!(k, 62, "{}", trip.message());
    assert!(energy < 4.2 * 0.5 * MASS * f64::from(v0 * v0));
    assert!(trip.reason.starts_with("the vehicles' kinetic energy rose from"), "{}", trip.reason);
    let message = trip.message();
    assert!(message.contains("373 broken and 2017 bent of 4228 beams"), "{message}");
    assert!(message.contains("kinetic energy of the vehicles read (t s: energy / budget)"), "{message}");
    // One readback past the budget (a reading mid-impact) is not a trip.
    let mut g = StabilityGuard::new(1);
    assert_eq!(g.observe(0.0, &[car(v0, 0.6, 5.0, 0.0)]), None);
    assert_eq!(g.observe(0.1, &[car(v0, 0.6, 5.0, 1.0e8)]), None);
    assert_eq!(g.observe(0.2, &[car(v0, 0.6, 5.0, 0.0)]), None);
    assert_eq!(g.observe(0.3, &[car(v0, 0.6, 5.0, 1.0e8)]), None);
    assert_eq!(g.observe(0.4, &[car(v0, 0.6, 5.0, 0.0)]), None);
}

/// The 2026-11-13 mountain run on the user's GPU: a 31 m drop at 189 km/h; after
/// the impact the kinetic energy fell to 681 kJ, then rose to 4,413 kJ in 0.4 s
/// (the whole-run budget, 22,000 kJ, was far above; the box rule caught it at
/// 112 m). The rebound check trips at about 3,000 kJ, a third of a second after the low.
#[test]
fn energy_rising_again_after_an_impact_trips() {
    let mut g = StabilityGuard::new(1);
    let v0 = 189.0 / 3.6;
    let extra = |kj: f64| kj * 1000.0;
    // The fall: 31 m in 2.5 s at 189 km/h.
    for k in 0..150 {
        let t = k as f64 / 60.0;
        let fall = (0.5 * GRAVITY * t * t).min(31.0);
        let vy = (2.0 * GRAVITY * fall).sqrt();
        assert_eq!(g.observe(t, &[car(v0, 40.0 - fall, 5.4, 0.5 * MASS * vy * vy)]), None, "t = {t}");
    }
    // The impact: down to 681 kJ in 0.2 s.
    let before = 0.5 * MASS * f64::from(v0 * v0) + MASS * GRAVITY * 31.0;
    for k in 0..12 {
        let t = 2.5 + k as f64 / 60.0;
        let e = before + (extra(681.0) - before) * (k as f64 + 1.0) / 12.0;
        assert_eq!(g.observe(t, &[car(0.0, 9.0, 6.0, e)]), None, "t = {t}");
    }
    // The blow-up: 681 kJ to 4,413 kJ in 0.4 s (24 readbacks).
    let mut tripped = None;
    for k in 1..=24 {
        let t = 2.7 + k as f64 / 60.0;
        let e = extra(681.0) * (4413.0f64 / 681.0).powf(k as f64 / 24.0);
        if let Some(trip) = g.observe(t, &[car(0.0, 9.0, 6.0 + 4.0 * k as f32, e)]) {
            tripped = Some((e, trip.clone()));
            break;
        }
    }
    let (e, trip) = tripped.expect("tripped");
    assert!(trip.reason.contains("its low at t = 2.683 s"), "{}", trip.reason);
    assert!(e < extra(3100.0), "tripped at {} kJ: {}", e / 1000.0, trip.message());
}

/// Energy coming back that is not an instability: a rebound off a wall, a bounce,
/// rolling downhill after a stop, driving off after a crash, a part thrown clear.
#[test]
fn rebounds_bounces_hills_driving_and_thrown_parts_do_not_trip() {
    // A 100 km/h wall hit, then a springy rebound at 30 % of the speed.
    let mut g = StabilityGuard::new(1);
    let v0 = 100.0f32 / 3.6;
    for k in 0..200 {
        let t = k as f64 / 60.0;
        let v = if k < 60 { v0 } else if k < 66 { v0 * (65 - k) as f32 / 6.0 } else { (0.3 * v0 * (k - 65) as f32 / 6.0).min(0.3 * v0) };
        assert_eq!(g.observe(t, &[car(v, 0.6, 5.0, 0.0)]), None, "rebound, t = {t}");
    }
    // Dropped from 10 m: bounces at a restitution of 0.6 until it rests.
    let mut g = StabilityGuard::new(1);
    let (mut h, mut vy, dt) = (10.0f64, 0.0f64, 1.0 / 60.0);
    for k in 0..900 {
        vy -= GRAVITY * dt;
        h += vy * dt;
        if h < 0.0 {
            h = 0.0;
            vy = -0.6 * vy;
        }
        assert_eq!(g.observe(k as f64 * dt, &[car(vy.abs() as f32, h + 0.6, 5.0, 0.0)]), None, "bounce, frame {k}");
    }
    // Stopped by a crash at the top of a hill, then rolls 60 m down it.
    let mut g = StabilityGuard::new(1);
    for k in 0..1200 {
        let t = k as f64 / 60.0;
        let (v, h) = if k < 60 { (20.0, 100.0) } else if k < 120 { (0.0, 100.0) } else {
            let drop = (0.5 * 2.0 * (t - 2.0) * (t - 2.0)).min(60.0);
            ((2.0 * GRAVITY * drop).sqrt() as f32 * 0.9, 100.0 - drop)
        };
        assert_eq!(g.observe(t, &[car(v, h, 5.0, 0.0)]), None, "hill, t = {t}");
    }
    // Crashes to a stop, then drives off: 0 to 150 km/h in 8 s.
    let mut g = StabilityGuard::new(1);
    for k in 0..700 {
        let t = k as f64 / 60.0;
        let v = if k < 60 { 30.0 } else if k < 120 { 0.0 } else { (150.0 / 3.6 * (t - 2.0) / 8.0).min(150.0 / 3.6) as f32 };
        assert_eq!(g.observe(t, &[car(v, 0.6, 5.0, 0.0)]), None, "driving, t = {t}");
    }
    // A part thrown clear in a crash: the box grows to 80 m, the energy only falls.
    let mut g = StabilityGuard::new(1);
    for k in 0..600 {
        let t = k as f64 / 60.0;
        let crash = ((t - 1.0) / 0.1).clamp(0.0, 1.0);
        let part = if crash > 0.0 { 0.5 * 20.0 * 40.0 * 40.0 * (-(t - 1.0)).exp() } else { 0.0 };
        let size = 5.0 + (40.0 * (t - 1.0).max(0.0) as f32).min(75.0);
        assert_eq!(g.observe(t, &[car(50.0 * (1.0 - crash) as f32, 0.6, size, part)]), None, "thrown part, t = {t}");
    }
    // The page launches a car that had come to rest.
    let mut g = StabilityGuard::new(1);
    for k in 0..300 {
        if k == 120 {
            g.launched(60.0);
        }
        let v = if k < 30 { 40.0 } else if k < 120 { 0.0 } else { 60.0 };
        assert_eq!(g.observe(k as f64 / 60.0, &[car(v, 0.6, 5.0, 0.0)]), None, "launch, frame {k}");
    }
}

/// The 2026-11-13 drive yard run (3 cars, 35 s of driving, a 25 km/h bump into a
/// parked car): the player's car went from 155 kJ to 7,569 kJ in 0.65 s. The old
/// budget (a 1 MW allowance per car) stood at 315,000 kJ; only the box check
/// caught it, at 161 m. Readbacks 0.1 s apart, as that page's were.
#[test]
fn energy_multiplying_in_a_long_drive_trips_within_three_doublings() {
    let mut g = StabilityGuard::new(3);
    let rest = car(0.0, 0.6, 5.4, 0.0);
    // 35 s of driving the player's car between 40 and 60 km/h.
    for k in 0..350 {
        let t = k as f64 * 0.1;
        let v = (50.0 + 10.0 * (t * 0.5).sin()) / 3.6;
        assert_eq!(g.observe(t, &[car(v as f32, 0.6, 5.4, 0.0), rest, rest]), None, "t = {t}");
    }
    // The bump: 25 km/h, the parked car shoved to 24 km/h (31 kJ), the player's car
    // at 155 kJ in all; then its energy as the report listed it.
    let bumped = car(23.7 / 3.6, 0.6, 6.4, 0.0);
    let v = 25.4f32 / 3.6;
    let own = 0.5 * MASS * f64::from(v * v);
    let mut tripped = None;
    for (k, kj) in [155.0, 230.0, 401.0, 760.0, 1510.0, 2900.0, 4271.0, 7569.0].into_iter().enumerate() {
        let t = 35.021 + k as f64 * 0.1;
        if let Some(trip) = g.observe(t, &[car(v, 0.52, 6.0 + 2.0 * k as f32, kj * 1000.0 - own), bumped, rest]) {
            tripped = Some((kj, trip.clone()));
            break;
        }
    }
    let (kj, trip) = tripped.expect("tripped");
    // Within the first three doublings: 760 kJ, the car still 12 m across (the
    // run went on to 7,569 kJ and 161 m).
    assert_eq!(kj, 760.0, "{}", trip.message());
    assert!(trip.reason.contains("what the tyres can have delivered"), "{}", trip.reason);
}

/// A minute of ordinary driving does not trip: three cars accelerating hard,
/// braking to a stop, cornering, bumping into each other at 30 km/h and pulling
/// away again, readbacks at 60 Hz and at 10 Hz.
#[test]
fn a_minute_of_driving_braking_and_bumps_does_not_trip() {
    for hz in [60.0, 10.0] {
        let mut g = StabilityGuard::new(3);
        let n = (60.0 * hz) as usize;
        for k in 0..n {
            let t = k as f64 / hz;
            // A 12 s cycle per car, out of phase: 0.9 g up to 130 km/h, 1.1 g
            // braking to rest, a pause; every third cycle ends in a 30 km/h bump
            // that throws 15 % of the energy into the other car and some debris.
            let cars: Vec<Sample> = (0..3)
                .map(|i| {
                    let phase = (t + 4.0 * i as f64) % 12.0;
                    let v = if phase < 4.0 { (0.9 * GRAVITY * phase).min(130.0 / 3.6) } else { (130.0 / 3.6 - 1.1 * GRAVITY * (phase - 4.0)).max(0.0) };
                    let bump = ((t + 4.0 * i as f64) / 12.0) as u32 % 3 == 2 && (8.0..8.3).contains(&phase);
                    let extra = if bump { 0.15 * 0.5 * MASS * (30.0f64 / 3.6).powi(2) } else { 0.0 };
                    car(if bump { 8.3 * (8.3 - phase) as f32 / 0.3 } else { v as f32 }, 0.6, if bump { 9.0 } else { 5.4 }, extra)
                })
                .collect();
            assert_eq!(g.observe(t, &cars), None, "{hz} Hz, t = {t}");
        }
    }
}

/// Inputs the guard cannot see are reported by the page: a shot's energy
/// (`supplied`), a hand's grab (`disturbed` each frame), a pulling rig
/// (`set_unbounded`).
#[test]
fn reported_inputs_do_not_trip() {
    // A 400 kJ shot hits a car at rest.
    let mut g = StabilityGuard::new(1);
    assert_eq!(g.observe(0.0, &[car(0.0, 0.6, 5.0, 0.0)]), None);
    g.supplied(4.0e5);
    for k in 1..60 {
        assert_eq!(g.observe(k as f64 / 60.0, &[car(20.0, 0.6, 5.0, 0.0)]), None, "shot, frame {k}");
    }
    // The same without the report is energy from nowhere.
    let mut g = StabilityGuard::new(1);
    assert_eq!(g.observe(0.0, &[car(0.0, 0.6, 5.0, 0.0)]), None);
    assert_eq!(g.observe(0.1, &[car(0.0, 0.6, 5.0, 3.0e5)]), None);
    assert!(g.observe(0.2, &[car(0.0, 0.6, 5.0, 3.0e5)]).is_some());
    // A hand drags a car from rest to 15 m/s over a second.
    let mut g = StabilityGuard::new(1);
    for k in 0..120 {
        g.disturbed();
        assert_eq!(g.observe(k as f64 / 60.0, &[car((15.0 * k as f32 / 60.0).min(15.0), 0.6, 5.0, 0.0)]), None, "grab, frame {k}");
    }
    // A rig pulls a car to 56 km/h; a boulder throws its parts 300 m. Unbounded:
    // only a non-finite value stops it.
    let mut g = StabilityGuard::new(1);
    g.set_unbounded(true);
    for k in 0..120 {
        assert_eq!(g.observe(k as f64 / 60.0, &[car((15.6 * k as f32 / 60.0).min(15.6), 0.6, 5.0, 0.0)]), None, "rig, frame {k}");
    }
    assert_eq!(g.observe(2.1, &[car(40.0, 0.6, 300.0, 2.0e6)]), None);
    let mut bad = car(40.0, 0.6, 300.0, 2.0e6);
    bad.finite = false;
    assert!(g.observe(2.2, &[bad]).is_some());
}

#[test]
fn an_exploding_box_with_rising_energy_trips() {
    let mut g = StabilityGuard::new(1);
    assert_eq!(g.observe(0.0, &[car(30.0, 0.6, 5.0, 0.0)]), None);
    assert_eq!(g.observe(0.5, &[car(5.0, 0.6, 6.0, 0.0)]), None);
    // Debris 240 m away with the energy falling: a torn car, not an instability.
    assert_eq!(g.observe(1.0, &[car(4.0, 0.6, 240.0, 0.0)]), None);
    // The same box with the energy up 20 × since its low (still inside the budget).
    let trip = g.observe(1.1, &[car(18.0, 0.6, 240.0, 0.0)]).cloned().expect("tripped");
    assert!(trip.reason.contains("240 m across (past 100 m"), "{}", trip.reason);
}

/// More vehicles than the guard reads: an unread one may hand a read one its
/// energy, so only the non-finite check runs.
#[test]
fn a_large_fleet_checks_only_for_non_finite_values() {
    let mut g = StabilityGuard::new(64);
    assert_eq!(g.tracked(), HEALTH_VEHICLES);
    let mut cars = vec![car(0.0, 0.6, 5.0, 0.0); HEALTH_VEHICLES];
    assert_eq!(g.observe(0.0, &cars), None);
    cars[3] = car(100.0, 0.6, 500.0, 1.0e9);
    for k in 1..10 {
        assert_eq!(g.observe(k as f64 * 0.1, &cars), None);
    }
    cars[5].finite = false;
    assert!(g.observe(1.0, &cars).is_some());
}

/// The guard is read-only: a run sampled by it every step is, bit for bit, the
/// run without it (the CPU reference; on the GPU the guard is a buffer copy and
/// a map, which write nothing the solver reads).
#[test]
fn the_guard_does_not_change_a_run() {
    use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
    let truck = crate::explicit::explicit_sedan_truck(true);
    let spawn = || {
        ExplicitBeams::from_truck(
            &truck,
            glam::Vec3::new(0.0, 0.3, 0.0),
            glam::Vec3::new(0.0, 0.0, 25.0),
            ExplicitBeamsSettings {
                fast_inv_sqrt: true,
                ..Default::default()
            },
        )
        .expect("the sedan")
    };
    let bits = |sim: &ExplicitBeams| -> Vec<u32> {
        sim.nodes
            .iter()
            .flat_map(|n| n.pos.to_array().into_iter().chain(n.vel.to_array()).chain(n.force.to_array()))
            .map(f32::to_bits)
            .chain(sim.origin.to_array().map(f32::to_bits))
            .collect()
    };
    let (mut plain, mut guarded) = (spawn(), spawn());
    let mut guard = StabilityGuard::new(1);
    for step in 0..400 {
        plain.step();
        guarded.step();
        let sample = Sample::from_points(
            guarded.origin.y,
            guarded.nodes.iter().map(|n| (n.pos.to_array(), n.mass, n.vel.to_array())),
        );
        assert_eq!(guard.observe(f64::from(step) / 2000.0, &[sample]), None, "step {step}");
        guard.damage(0, 0, 0, guarded.beams.len() as u32);
    }
    assert!(guard.trip().is_none());
    assert_eq!(bits(&plain), bits(&guarded));
}
