//! Physical and geometric invariants of the rigid benchmark scenes (`engine_scenes::web`,
//! Y-up), written from their definitions, and the specification of their GPU upload order.
//!
//! Ring: running-bond courses of 1 × 0.5 × 0.5 bricks on concentric rings, a stepped
//! profile (one ring fewer every `tier` courses) and a rolling impactor. Gables: stepped
//! brick gables on a grid, each course centred over the joints below, and two rolling
//! impactors. Upload: Morton order over the scene's bounds, the ground at slot 0.

use engine_scenes::web::{self, Body, Piece, Shape, BODY_RECORD};
use std::collections::BTreeMap;
use std::f64::consts::PI;

const BRICK: [f64; 3] = [1.0, 0.5, 0.5];
/// The ground slab's top.
const GROUND_TOP: f64 = 0.5;
const EPS: f64 = 1e-9;

fn near(a: f64, b: f64, tol: f64, msg: &str) {
    assert!((a - b).abs() <= tol, "{msg}: {a} vs {b}");
}

/// Rotate `v` by the unit quaternion `q` (x, y, z, w).
fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z, w] = q;
    let t = [
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ];
    [
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    ]
}

struct Brick {
    p: [f64; 3],
    angle: f64,
    mass: f64,
}

struct Ball {
    p: [f64; 3],
    r: f64,
    density: f64,
    v: [f64; 3],
    w: [f64; 3],
}

fn split(pieces: &[Piece]) -> (Vec<Brick>, Vec<Ball>) {
    let (mut bricks, mut balls) = (Vec::new(), Vec::new());
    for p in pieces {
        match *p {
            Piece::Brick { p, angle, mass, .. } => bricks.push(Brick { p, angle, mass }),
            Piece::Ball {
                p,
                radius,
                density,
                v,
                w,
            } => balls.push(Ball {
                p,
                r: radius,
                density,
                v,
                w,
            }),
            Piece::Box { .. } => panic!("a box in a brick scene"),
        }
    }
    (bricks, balls)
}

/// A brick's mass, a unit yaw-only orientation, and a solid box's inertia (from full
/// extents, written independently of `box_inertia`).
fn check_brick(b: &Brick, label: &str) -> [f64; 4] {
    assert!(
        b.p.iter().all(|x| x.is_finite()),
        "{label}: finite position"
    );
    near(b.mass, 0.25, EPS, &format!("{label}: brick mass"));
    let q = web::brick_quaternion(b.angle);
    near(
        (q.iter().map(|c| c * c).sum::<f64>()).sqrt(),
        1.0,
        1e-12,
        &format!("{label}: unit orientation"),
    );
    near(q[0], 0.0, EPS, &format!("{label}: yaw only (x)"));
    near(q[2], 0.0, EPS, &format!("{label}: yaw only (z)"));
    let i = web::box_inertia(b.mass, web::BRICK_HALF);
    let [a, bb, c] = BRICK;
    let want = [
        b.mass * (bb * bb + c * c) / 12.0,
        b.mass * (a * a + c * c) / 12.0,
        b.mass * (a * a + bb * bb) / 12.0,
    ];
    for k in 0..3 {
        near(
            i[k] / want[k],
            1.0,
            1e-12,
            &format!("{label}: inertia axis {k}"),
        );
    }
    q
}

fn check_ball(ball: &Ball, min_z: f64, label: &str) {
    let mass = web::sphere_mass(ball.r, ball.density);
    near(
        mass,
        ball.density * 4.0 / 3.0 * PI * ball.r.powi(3),
        1e-9 * mass,
        &format!("{label}: sphere mass"),
    );
    near(
        ball.p[1] - ball.r,
        GROUND_TOP,
        EPS,
        &format!("{label}: resting on the ground"),
    );
    // Rolls without slipping: v + w × (0, −r, 0) = 0 at the contact point.
    let contact = [
        ball.v[0] + ball.w[2] * ball.r,
        ball.v[1],
        ball.v[2] - ball.w[0] * ball.r,
    ];
    for (k, x) in contact.iter().enumerate() {
        near(
            *x,
            0.0,
            1e-9,
            &format!("{label}: contact velocity axis {k}"),
        );
    }
    assert!(
        ball.v[2] > 0.0 && ball.p[2] + ball.r < min_z,
        "{label}: starts outside the structure, rolling towards it"
    );
}

fn check_ring(name: &str, radius: f64, courses: u32, tier: u32, rows: u32, count: usize) {
    let (bricks, balls) = split(&web::ring(radius, courses, tier, rows));
    assert_eq!(
        bricks.len() + balls.len(),
        count,
        "{name}: changed benchmark workload"
    );
    assert_eq!(balls.len(), 1);
    let mut rings: BTreeMap<(i64, i64), Vec<f64>> = BTreeMap::new();
    let mut min_z = f64::INFINITY;
    for (i, b) in bricks.iter().enumerate() {
        let label = format!("{name} brick {i}");
        let q = check_brick(b, &label);
        let course = (b.p[1] - 0.75) / 0.5;
        near(
            course,
            course.round(),
            1e-9,
            &format!("{label}: course height"),
        );
        let c = course.round() as i64;
        assert!(c >= 0 && c < courses as i64, "{label}: course {c}");
        // The bottom course rests on the ground; each course stacks one brick height up.
        near(
            b.p[1] - BRICK[1] / 2.0,
            GROUND_TOP + BRICK[1] * c as f64,
            1e-9,
            &format!("{label}: stacked height"),
        );
        let r = b.p[0].hypot(b.p[2]);
        let j = ((r - radius - 0.25) / 0.52).round() as i64;
        near(
            r,
            radius + 0.25 + 0.52 * j as f64,
            1e-9,
            &format!("{label}: ring radius"),
        );
        // Stepped profile: course c keeps max(1, rows − c / tier) rings.
        let depth = (rows as i64 - c / tier as i64).max(1);
        assert!(
            j >= 0 && j < depth,
            "{label}: ring {j} beyond the step profile"
        );
        // Long axis tangent to the ring, short axis radial.
        let along = rotate(q, [1.0, 0.0, 0.0]);
        near(
            (along[0] * b.p[0] / r + along[2] * b.p[2] / r).abs(),
            0.0,
            1e-9,
            &format!("{label}: long axis tangent"),
        );
        rings.entry((c, j)).or_default().push(b.p[2].atan2(b.p[0]));
        min_z = min_z.min(b.p[2] - 0.5);
    }
    for (&(c, j), angles) in rings.iter_mut() {
        let n = angles.len() as f64;
        let r = radius + 0.25 + 0.52 * j as f64;
        let inner = r - BRICK[2] / 2.0;
        angles.sort_by(f64::total_cmp);
        let label = format!("{name} course {c} ring {j}");
        // Evenly spaced, and as many bricks as fit: the inner-face arc per brick is at
        // least 1.02 m (a 1 m brick and a 2 cm joint); one more would not fit.
        for k in 1..angles.len() {
            near(
                angles[k] - angles[k - 1],
                2.0 * PI / n,
                1e-9,
                &format!("{label}: even spacing"),
            );
        }
        assert!(
            2.0 * PI * inner / n >= 1.02 - 1e-9 && 2.0 * PI * inner / (n + 1.0) < 1.02,
            "{label}: {n} bricks is not the most that fit"
        );
        // Neighbouring centres are at least a brick length apart (no overlap).
        assert!(
            2.0 * r * (PI / n).sin() >= BRICK[0],
            "{label}: neighbours overlap"
        );
        // Absolute phase: even courses start at angle 0, odd ones half a brick on.
        let step = 2.0 * PI / n;
        let offset = ((angles[0] - (c % 2) as f64 * step / 2.0) % step + step) % step;
        near(
            offset.min(step - offset),
            0.0,
            1e-9,
            &format!("{label}: phase"),
        );
    }
    // Running bond: course c + 1 is turned half a brick against course c.
    for (&(c, j), angles) in &rings {
        if let Some(above) = rings.get(&(c + 1, j)).filter(|a| a.len() == angles.len()) {
            let step = 2.0 * PI / angles.len() as f64;
            let shift = ((above[0] - angles[0]) % step + step) % step;
            near(
                shift,
                PI / angles.len() as f64,
                1e-9,
                &format!("{name} course {c}->{} ring {j}: half-brick offset", c + 1),
            );
        }
    }
    // Every course has exactly its stepped set of rings 0 .. depth − 1.
    for c in 0..courses as i64 {
        let depth = (rows as i64 - c / tier as i64).max(1);
        for j in 0..=depth {
            assert_eq!(
                rings.contains_key(&(c, j)),
                j < depth,
                "{name} course {c}: ring {j} presence"
            );
        }
    }
    let ball = &balls[0];
    near(
        ball.r,
        courses as f64 / 5.0,
        EPS,
        &format!("{name}: impactor radius"),
    );
    near(
        ball.p[0],
        0.0,
        EPS,
        &format!("{name}: impactor aimed at the ring centre"),
    );
    check_ball(ball, min_z, &format!("{name} impactor"));
}

#[test]
fn ring_9k_bricks_sit_on_their_course_and_ring_radii_without_overlap() {
    check_ring("ring 9k", 12.0, 20, 2, 10, 9131);
}

#[test]
fn ring_110k_bricks_sit_on_their_course_and_ring_radii_without_overlap() {
    check_ring("ring 110k", 80.0, 40, 4, 10, 110441);
}

#[test]
fn gables_506k_stand_on_the_joints_below_never_overlap_and_line_up_their_impactors() {
    let (columns, rows, base) = (16u32, 68u32, 30u32);
    let (bricks, balls) = split(&web::gables(columns, rows, base));
    assert_eq!(
        bricks.len() as u32,
        columns * rows * base * (base + 1) / 2,
        "triangular gables: base + (base − 1) + … + 1 per gable"
    );
    assert_eq!(
        bricks.len() + balls.len(),
        505922,
        "changed benchmark workload"
    );
    let mut by_gable: BTreeMap<(i64, i64), Vec<(f64, i64)>> = BTreeMap::new();
    let mut min_z = f64::INFINITY;
    for (i, b) in bricks.iter().enumerate() {
        check_brick(b, &format!("gables brick {i}"));
        near(
            b.angle,
            0.0,
            EPS,
            &format!("gables brick {i}: aligned with X"),
        );
        let k = ((b.p[1] - 0.75) / 0.5).round() as i64;
        near(
            b.p[1] - BRICK[1] / 2.0,
            GROUND_TOP + BRICK[1] * k as f64,
            1e-9,
            &format!("gables brick {i}: stacked height"),
        );
        let column = (b.p[0] / (base as f64 + 2.0) + (columns as f64 - 1.0) / 2.0).round() as i64;
        by_gable
            .entry((column, (b.p[2] * 1e6).round() as i64))
            .or_default()
            .push((b.p[0], k));
        min_z = min_z.min(b.p[2] - 0.25);
    }
    assert_eq!(by_gable.len() as u32, columns * rows);
    let mut centres = Vec::new();
    for (key, list) in &by_gable {
        let mut courses: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
        for &(x, k) in list {
            courses.entry(k).or_default().push(x);
        }
        for xs in courses.values_mut() {
            xs.sort_by(f64::total_cmp);
        }
        let xs0 = &courses[&0];
        let centre = (xs0[0] + xs0[xs0.len() - 1]) / 2.0;
        centres.push(centre);
        for k in 0..base as i64 {
            let xs = courses.get(&k).cloned().unwrap_or_default();
            assert_eq!(
                xs.len() as i64,
                base as i64 - k,
                "gable {key:?} course {k}: brick count"
            );
            near(
                (xs[0] + xs[xs.len() - 1]) / 2.0,
                centre,
                1e-9,
                &format!("gable {key:?} course {k}: centred"),
            );
            for n in 1..xs.len() {
                near(
                    xs[n] - xs[n - 1],
                    1.01,
                    1e-9,
                    &format!("gable {key:?} course {k}: 1 cm joints"),
                );
            }
            // Every brick above course 0 bridges the joint between two bricks below.
            if k > 0 {
                let below = &courses[&(k - 1)];
                for &x in &xs {
                    let n = below
                        .iter()
                        .filter(|&&y| ((x - y).abs() - 0.505).abs() < 1e-9)
                        .count();
                    assert_eq!(
                        n, 2,
                        "gable {key:?} course {k}: brick at {x} is not on two bricks"
                    );
                }
            }
        }
    }
    // Neighbouring gables never overlap: the X pitch exceeds the widest course, the Z pitch
    // the depth.
    let width = (base as f64 - 1.0) * 1.01 + BRICK[0];
    assert!(
        base as f64 + 2.0 > width,
        "X pitch leaves a gap between gables"
    );
    assert!(5.0 > BRICK[2], "Z pitch leaves a gap between gable rows");
    assert_eq!(balls.len(), 2);
    for (i, ball) in balls.iter().enumerate() {
        near(
            ball.r,
            base as f64 / 7.0,
            EPS,
            &format!("gables impactor {i}: radius"),
        );
        assert!(
            centres.iter().any(|c| (c - ball.p[0]).abs() < 1e-9),
            "gables impactor {i}: not aligned with a gable column"
        );
        check_ball(ball, min_z, &format!("gables impactor {i}"));
    }
}

#[test]
fn the_ground_slab_covers_every_body_and_the_run_up_with_a_100_m_minimum() {
    near(
        web::ground_half_extent(10.0),
        100.0,
        0.0,
        "minimum half extent",
    );
    for (radius, courses, rows) in [(12.0, 20, 10), (80.0, 40, 10)] {
        let half = web::ground_half_extent(radius + 10.0);
        for p in web::ring(radius, courses, 2, rows) {
            let at = match p {
                Piece::Brick { p, .. } | Piece::Box { p, .. } | Piece::Ball { p, .. } => p,
            };
            assert!(
                at[0].abs() < half && at[2].abs() < half,
                "radius {radius}: body off the ground slab"
            );
        }
    }
}

#[test]
fn rolling_angular_velocity_spins_against_the_direction_of_travel() {
    for radius in [4.0, 8.0, 30.0 / 7.0] {
        let w = web::rolling_angular_velocity(30.0, radius);
        near(
            30.0 - w[0] * radius,
            0.0,
            1e-12,
            &format!("radius {radius}: contact-point speed"),
        );
        assert_eq!((w[1], w[2]), (0.0, 0.0));
    }
}

/// The upload order's specification, written independently of `upload_order` (bit by bit,
/// no magic-number spreading): quantise each axis to 10 bits over the scene's bounds
/// (q = min(1023, floor((p − lo) / max(hi − lo, 1e-9) × 1024))); interleave from the
/// least significant bit in the axis order X, Z, Y (bit b of X → 3b, of Z → 3b + 1, of Y
/// → 3b + 2) so vertical stacks vary slowest within a cell; stable-sort the dynamic bodies
/// by code (ties keep scene order); the static ground stays at slot 0.
fn morton_spec(positions: &[[f64; 3]]) -> Vec<u32> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let quant = |p: &[f64; 3], k: usize| {
        ((p[k] - lo[k]) / (hi[k] - lo[k]).max(1e-9) * 1024.0)
            .floor()
            .min(1023.0) as u64
    };
    let codes: Vec<u64> = positions
        .iter()
        .map(|p| {
            let q = [quant(p, 0), quant(p, 2), quant(p, 1)];
            let mut c = 0u64;
            for b in 0..10 {
                for (s, axis) in q.iter().enumerate() {
                    c += ((axis >> b) & 1) << (3 * b + s);
                }
            }
            c
        })
        .collect();
    let mut rest: Vec<u32> = (1..positions.len() as u32).collect();
    rest.sort_by(|&a, &b| codes[a as usize].cmp(&codes[b as usize]).then(a.cmp(&b)));
    std::iter::once(0).chain(rest).collect()
}

/// Bodies at `positions`, the first a static ground (unit boxes).
fn at(positions: &[[f64; 3]]) -> Vec<Body> {
    positions
        .iter()
        .enumerate()
        .map(|(i, &p)| Body {
            dynamic: i > 0,
            p,
            q: [0.0, 0.0, 0.0, 1.0],
            v: [0.0; 3],
            w: [0.0; 3],
            mass: if i > 0 { 1.0 } else { 0.0 },
            inertia: [1.0; 3],
            shape: Shape::Box { half: [0.5; 3] },
            density: 1.0,
            bullet: false,
        })
        .collect()
}

fn upload(positions: &[[f64; 3]]) -> Vec<u32> {
    web::upload_order(&at(positions), true).unwrap()
}

#[test]
fn morton_upload_handles_negative_coordinates_coincident_bodies_degenerate_axes_and_the_ground() {
    for positions in [
        vec![[0.0; 3]],
        vec![[0.0; 3]; 3],
        vec![
            [0.0, 0.0, 0.0],
            [12.0, 0.0, -8.0],
            [-12.0, 0.0, 8.0],
            [1.0, 0.0, 3.0],
            [1.0, 0.0, 3.0],
            [0.0, 0.0, -8.0],
        ],
    ] {
        assert_eq!(upload(&positions), morton_spec(&positions));
    }
    // Hand-checked: one body per corner of a unit cube. Each axis quantises to 0 or 1023,
    // so at every bit Y outranks Z outranks X: (0,0,0) (1,0,0) (0,0,1) (1,0,1) (0,1,0)
    // (1,1,0) (0,1,1) (1,1,1).
    let corners = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ];
    assert_eq!(
        upload(&corners),
        vec![0, 8, 4, 3, 7, 1, 5, 6, 2],
        "axis order X, Z, Y"
    );
    let mut moving_ground = at(&corners);
    moving_ground[0].dynamic = true;
    let e = web::upload_order(&moving_ground, true)
        .unwrap_err()
        .to_string();
    assert!(e.contains("static ground"), "{e}");
}

#[test]
fn the_large_tiers_upload_in_the_specified_order_with_their_impactors() {
    for (name, count) in [("ring110k", 110442), ("gables506k", 505923)] {
        let table = web::scenes();
        let ground = table.iter().find(|(n, _)| *n == name).unwrap().1.ground;
        let bodies = web::bodies(name, 0, ground, true).unwrap();
        assert_eq!(
            bodies.len(),
            count,
            "{name}: do not reduce scene scale to improve timing"
        );
        let order = web::upload_order(&bodies, true).unwrap();
        let positions: Vec<[f64; 3]> = bodies.iter().map(|b| b.p).collect();
        assert_eq!(order, morton_spec(&positions), "{name}");
        assert!(
            order.iter().enumerate().any(|(g, &s)| g as u32 != s),
            "{name}: the scene must exercise a non-identity permutation"
        );
    }
}

/// A deterministic pseudo-random cloud (an LCG: failures reproduce), the ground first.
fn cloud(n: usize, seed: u32, scale: f64, integer: bool) -> Vec<[f64; 3]> {
    let mut s = seed;
    let mut r = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s as f64 / 4294967296.0 - 0.5) * scale
    };
    let mut out = vec![[0.0; 3]];
    for _ in 1..n {
        let p = [r(), r() * 0.3, r()];
        out.push(if integer { p.map(f64::round) } else { p });
    }
    out
}

#[test]
fn the_upload_order_is_a_bijection_with_the_ground_at_slot_0() {
    for seed in [1, 7, 99] {
        let order = upload(&cloud(5000, seed, 40.0, false));
        assert_eq!(order[0], 0);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            (0..5000).collect::<Vec<u32>>(),
            "every body exactly once"
        );
    }
}

#[test]
fn the_order_depends_only_on_positions_relative_to_the_bounds() {
    let base = cloud(3000, 5, 64.0, true);
    let moved: Vec<[f64; 3]> = base
        .iter()
        .map(|p| [p[0] + 1024.0, p[1] - 512.0, p[2] + 256.0])
        .collect();
    assert_eq!(upload(&moved), upload(&base));
}

#[test]
fn a_flat_scene_orders_by_its_horizontal_axes() {
    let flat: Vec<[f64; 3]> = cloud(2000, 11, 40.0, false)
        .into_iter()
        .map(|p| [p[0], 3.0, p[2]])
        .collect();
    assert_eq!(upload(&flat), morton_spec(&flat));
}

#[test]
fn the_morton_upload_keeps_spatial_neighbours_adjacent_in_memory() {
    let positions = cloud(20000, 3, 40.0, false);
    // A random cloud's scene order is spatially incoherent; the Morton order must cut the
    // mean distance between consecutive GPU bodies several-fold.
    let mean_step = |order: &[u32]| {
        let mut d = 0.0;
        for i in 2..order.len() {
            let (a, b) = (
                positions[order[i - 1] as usize],
                positions[order[i] as usize],
            );
            d += ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        }
        d / (order.len() - 2) as f64
    };
    let scene = mean_step(&(0..positions.len() as u32).collect::<Vec<_>>());
    let morton = mean_step(&upload(&positions));
    assert!(
        morton * 5.0 < scene,
        "Morton mean step {morton:.3} vs scene order {scene:.3}"
    );
}

#[test]
fn records_keep_each_body_with_its_shape_in_upload_order() {
    for spatial in [false, true] {
        let mut bodies = at(&[
            [0.0, 0.0, 0.0],
            [8.0, 2.0, -9.0],
            [-9.0, 1.0, 8.0],
            [0.0, 3.0, -4.0],
        ]);
        bodies[0].shape = Shape::Box {
            half: [9.0, 1.0, 9.0],
        };
        bodies[0].density = 0.0;
        for (k, b) in bodies.iter_mut().enumerate().skip(1) {
            let k = k as f64;
            b.mass = k;
            b.inertia = [k, k + 1.0, k + 2.0];
            b.shape = if k == 3.0 {
                Shape::Sphere { radius: 0.5 }
            } else {
                Shape::Box {
                    half: [k, k + 1.0, k + 2.0],
                }
            };
            b.density = k + 3.0;
            b.v = [k - 1.0, 0.0, 0.0];
        }
        let order = web::upload_order(&bodies, spatial).unwrap();
        let r = web::records(&bodies, &order);
        assert_eq!(r.len(), bodies.len() * BODY_RECORD);
        assert_eq!(order[0], 0, "the ground stays at slot 0");
        for (slot, &scene) in order.iter().enumerate() {
            let (b, r) = (
                &bodies[scene as usize],
                &r[slot * BODY_RECORD..(slot + 1) * BODY_RECORD],
            );
            let f = |x: f64| x as f32;
            assert_eq!(
                &r[2..5],
                &b.p.map(f),
                "scene {scene}: wrong body at slot {slot}"
            );
            assert_eq!(r[0], if b.dynamic { 2.0 } else { 0.0 });
            let (shape, dims) = match b.shape {
                Shape::Sphere { radius } => (0.0, [radius, 0.0, 0.0]),
                Shape::Box { half } => (2.0, half),
            };
            assert_eq!(r[19], shape, "a body keeps its shape");
            assert_eq!(&r[20..23], &dims.map(f), "a body keeps its dimensions");
            assert_eq!(r[23], f(b.density));
            assert_eq!(r[15], f(b.mass));
            assert_eq!(&r[16..19], &b.inertia.map(f));
            assert_eq!(r[9], f(b.v[0]));
        }
    }
}
