//! Vehicle–vehicle contact of the explicit solver
//! (`engine/docs/explicit-beams-vehicle-contact-spec.md`), each rule checked from
//! its definition on tiny synthetic structures: contact points, the contact
//! throttle and the fast-approach reset, one node over one triangle (force and
//! reactions), the side vote, order dependence, and vehicles that never meet.

use engine_softbody::explicit_beams::*;
use engine_softbody::explicit_contact::*;
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
};

/// Unit-mass nodes at `points` with ground contact on (the ground far below), no
/// drag.
fn vehicle(points: &[Vec3]) -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -1.0e3;
    for &p in points {
        s.nodes.push(Node {
            pos: p,
            vel: Vec3::ZERO,
            force: Vec3::ZERO,
            mass: 1.0,
            friction: 1.0,
            flags: 0,
        });
    }
    s
}

fn triangle(nodes: [u32; 3]) -> Triangle {
    Triangle {
        nodes,
        wait: 0,
        idle: 0,
    }
}

/// An immovable triangle in y = 0 (x, z in [−1, 1]) plus an immovable node at
/// `extra` (stretching the box).
fn floor(extra: Vec3) -> ExplicitBeams {
    let mut s = vehicle(&[
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 2.0),
        extra,
    ]);
    for n in &mut s.nodes {
        n.flags |= NODE_IMMOVABLE;
    }
    s.triangles.push(triangle([0, 1, 2]));
    s.finish();
    s
}

/// One free node at `p` (a vehicle without triangles: its node is a contact
/// point).
fn point(p: Vec3, v: Vec3) -> ExplicitBeams {
    let mut s = vehicle(&[p]);
    s.nodes[0].vel = v;
    s.finish();
    s
}

const CONTACT_TRUCK: &str = "contact test
globals
100, 0, tracks/semi
nodes
0, 0, 0.5, 0
1, 0, 0.5, 0.3
2, 0.5, 0.5, 0.25
3, 0, 1.0, 0.15
4, 1, 1.0, 0.15, c
5, 1, 0.2, 0.15, c
6, 2, 1.0, 0.15
beams
0, 1
0, 2
1, 2
0, 3
1, 3
2, 3
3, 4
4, 5
5, 6
contacters
5
submesh
cab
0, 1, 4, c
wheels
0.4, 0.2, 6, 0, 1, 2, 1, 0, 9999, 30, 50000, 300
end
";

#[test]
fn contact_points_are_contacters_and_triangle_nodes_with_ground_contact() {
    let s = ExplicitBeams::from_truck(CONTACT_TRUCK, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    // Nodes 0, 1 on the triangle; 4 on it but `c` (no ground contact): out; 5 a
    // `c` contacter: in; 2, 3, 6 neither: out; every tyre node (7..19): in.
    let mut want = vec![0, 1, 5];
    want.extend(7..19);
    assert_eq!(s.contact_points, want);
    // Without triangles every node except the `c` ones (contacters still in).
    let src = CONTACT_TRUCK.replace("0, 1, 4, c\n", "");
    let s = ExplicitBeams::from_truck(&src, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let mut want = vec![0, 1, 2, 3, 5, 6];
    want.extend(7..19);
    assert_eq!(s.contact_points, want);
    // Without a contacters section a `c` node off the triangles is not one.
    let src = CONTACT_TRUCK.replace("contacters\n5\n", "");
    let s = ExplicitBeams::from_truck(&src, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let mut want = vec![0, 1];
    want.extend(7..19);
    assert_eq!(s.contact_points, want);
}

#[test]
fn neighbour_lists_follow_beam_order_and_keep_duplicates() {
    let mut s = vehicle(&[Vec3::ZERO, Vec3::X, Vec3::Y]);
    let b = |a, b| {
        Beam::new(
            a,
            b,
            ExplicitKind::Plain,
            ExplicitType::Deformable,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
        )
    };
    s.beams = vec![b(0, 1), b(2, 0), b(1, 0)];
    s.finish();
    let list =
        |i: usize| &s.neighbours[s.neighbour_start[i] as usize..s.neighbour_start[i + 1] as usize];
    assert_eq!(list(0), &[1, 2, 1]);
    assert_eq!(list(1), &[0, 0]);
    assert_eq!(list(2), &[0]);
}

#[test]
fn a_named_contact_surface_and_too_many_triangles_are_refused() {
    let src = CONTACT_TRUCK.replace("submesh\n", "submesh_groundmodel ice\nsubmesh\n");
    let e = ExplicitBeams::from_truck(&src, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap_err();
    assert!(e.contains("submesh_groundmodel"), "{e}");
    let src = CONTACT_TRUCK.replace("submesh\n", "submesh_groundmodel concrete\nsubmesh\n");
    assert!(ExplicitBeams::from_truck(&src, Vec3::ZERO, Vec3::ZERO, EXACT).is_ok());
    let many = "0, 1, 4, c\n".repeat(3001);
    let src = CONTACT_TRUCK.replace("0, 1, 4, c\n", &many);
    let e = ExplicitBeams::from_truck(&src, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap_err();
    assert!(e.contains("3000"), "{e}");
}

#[test]
fn a_solid_and_buoyant_cab_line_is_two_triangles() {
    let src = CONTACT_TRUCK.replace("0, 1, 4, c\n", "0, 1, 4, cD\n");
    let s = ExplicitBeams::from_truck(&src, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    assert_eq!(s.triangles.len(), 2);
    assert_eq!(s.triangles[0].nodes, s.triangles[1].nodes);
    assert_eq!(s.contact_throttle.len(), 2);
}

/// Triangle tests of owner 0 over `steps` steps (1-based), with its throttle after
/// each.
fn tested_steps(world: &mut ExplicitBeamsWorld, steps: usize) -> (Vec<usize>, Vec<[i32; 2]>) {
    let mut tested = Vec::new();
    let mut seen = Vec::new();
    for k in 1..=steps {
        world.step();
        if world.last.triangles_tested > 0 {
            tested.push(k);
        }
        seen.push(world.vehicles[0].contact_throttle[0]);
    }
    (tested, seen)
}

#[test]
fn a_triangle_without_candidates_backs_off_to_every_fourteenth_step() {
    // The partner's point is inside the owner's box (stretched by its 4th node)
    // but far from the triangle's.
    let owner = floor(Vec3::new(3.0, 3.0, 3.0));
    let mut p = point(Vec3::new(2.5, 2.5, 2.5), Vec3::ZERO);
    p.nodes[0].flags |= NODE_IMMOVABLE;
    let mut w = ExplicitBeamsWorld::new(vec![owner, p]);
    let (tested, seen) = tested_steps(&mut w, 120);
    assert_eq!(&tested[..6], &[1, 3, 6, 10, 15, 21]);
    assert_eq!(
        &seen[..6],
        &[[1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [3, 0]]
    );
    let gaps: Vec<usize> = tested.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(gaps.ends_with(&[14, 14]), "{gaps:?}");
    assert!(seen.iter().all(|t| t[0] >= 0 && t[0] <= 13));
}

#[test]
fn a_frozen_throttle_when_no_partner_and_a_kept_interval_with_candidates() {
    // No partner: the owner's triangle loop never runs, its throttle stays.
    let owner = floor(Vec3::new(3.0, 3.0, 3.0));
    let far = point(Vec3::new(50.0, 0.0, 0.0), Vec3::ZERO);
    let mut w = ExplicitBeamsWorld::new(vec![owner, far]);
    w.vehicles[0].contact_throttle[0] = [5, 2];
    let (tested, seen) = tested_steps(&mut w, 10);
    assert!(tested.is_empty());
    assert!(seen.iter().all(|&t| t == [5, 2]));
    // A candidate in the triangle's padded box but outside the band: the interval
    // set from cidle is kept (no increment).
    let owner = floor(Vec3::new(3.0, 3.0, 3.0));
    let mut p = point(Vec3::new(0.9, 0.01, 1.5), Vec3::ZERO);
    p.nodes[0].flags |= NODE_IMMOVABLE;
    let mut w = ExplicitBeamsWorld::new(vec![owner, p]);
    w.vehicles[0].contact_throttle[0] = [0, 4];
    let (tested, seen) = tested_steps(&mut w, 12);
    assert_eq!(&seen[..2], &[[4, 0], [3, 1]]);
    assert_eq!(&tested[..2], &[1, 6]);
    assert_eq!(seen[5], [4, 0], "re-tested at the same interval");
}

#[test]
fn a_hit_tests_every_step_and_never_goes_negative() {
    let owner = floor(Vec3::new(3.0, 3.0, 3.0));
    let p = point(Vec3::new(0.0, 0.01, 0.0), Vec3::ZERO);
    let mut w = ExplicitBeamsWorld::new(vec![owner, p]);
    let (tested, seen) = tested_steps(&mut w, 50);
    assert_eq!(tested, (1..=50).collect::<Vec<_>>());
    assert!(seen.iter().all(|&t| t == [0, 0]));
    // Held up on the far edge of the band (one range above the plane).
    let y = w.vehicles[1].nodes[0].pos.y;
    assert!(y > 0.0 && y <= 0.021, "{y}");
}

#[test]
fn a_fast_approach_resets_both_vehicles_throttles() {
    // Two vehicles whose boxes overlap, node 0 speeds 5 m/s apart, nowhere near
    // touching triangles.
    let mut a = floor(Vec3::new(3.0, 3.0, 3.0));
    let mut b = floor(Vec3::new(3.0, 3.0, 3.0));
    for n in &mut b.nodes {
        n.pos += Vec3::new(0.0, 2.0, 0.0);
    }
    // Node 0 of b moves (it is immovable: its position stays, its velocity counts).
    b.nodes[0].vel = Vec3::new(5.0, 0.0, 0.0);
    a.triangles[0].wait = 7;
    a.triangles[0].idle = 3;
    b.triangles[0].wait = -20000;
    let mut w = ExplicitBeamsWorld::new(vec![a, b]);
    w.vehicles[0].contact_throttle[0] = [9, 4];
    w.vehicles[1].contact_throttle[0] = [6, 1];
    // Without contact nothing resets (the self-collision throttles still run:
    // no candidates for self-collision, so they only count).
    w.contact = false;
    let mut off = w.clone();
    off.step();
    assert_eq!(off.vehicles[0].contact_throttle[0], [9, 4]);
    w.contact = true;
    w.step();
    assert_eq!(w.last.resets, 2, "in both scans");
    // Reset to 0 in a's scan, then a's loop tested it (no candidates: 0 + 1),
    // then b's scan set it to 0 again.
    assert_eq!(w.vehicles[0].contact_throttle[0], [0, 0]);
    // b's loop ran after both resets: min(cidle 1, 12), no candidates, + 1.
    assert_eq!(w.vehicles[1].contact_throttle[0], [2, 0]);
    // The self-collision throttles: wait ← 0 (−20000 erased), idle kept.
    assert_eq!(
        (
            w.vehicles[0].triangles[0].wait,
            w.vehicles[0].triangles[0].idle
        ),
        (0, 3)
    );
    assert_eq!(w.vehicles[1].triangles[0].wait, 0);
    // At 4 m/s exactly nothing resets.
    w.vehicles[1].nodes[0].vel = Vec3::new(4.0, 0.0, 0.0);
    w.vehicles[0].contact_throttle[0] = [9, 4];
    w.step();
    assert_eq!(w.last.resets, 0);
    assert_eq!(w.vehicles[0].contact_throttle[0], [8, 5]);
}

#[test]
fn one_node_over_one_triangle_gets_the_contact_force_and_the_triangle_its_reaction() {
    let owner = floor(Vec3::new(3.0, 3.0, 3.0));
    let mut p = point(Vec3::new(0.2, 0.012, 0.3), Vec3::new(0.3, -1.0, 0.1));
    p.nodes[0].mass = 2.0;
    p.nodes[0].friction = 0.8;
    let mut w = ExplicitBeamsWorld::new(vec![owner, p]);
    for v in &mut w.vehicles {
        v.step();
    }
    let before: Vec<Vec<Vec3>> = w
        .vehicles
        .iter()
        .map(|v| v.nodes.iter().map(|n| n.force).collect())
        .collect();
    let h = w.vehicles[1].nodes[0];
    let tri = w.vehicles[0].positions();
    let frame = ContactFrame::new(tri[0], tri[1], tri[2]);
    let (alpha, beta, gamma, delta) = frame.local(w.vehicles[1].origin + h.pos);
    assert!(alpha > 0.0 && beta > 0.0 && gamma > 0.0 && delta.abs() <= 0.02);
    w.contact_pass();
    assert_eq!(w.last.hits, 1);
    let t = &w.vehicles[0].nodes;
    // Triangle masses 1, weighted as the rule writes it.
    let mt = 1.0f32 * alpha + 1.0 * beta + 1.0 * gamma;
    let m = (2.0f32 * mt) / (2.0 + mt);
    let n = if delta >= 0.0 { frame.n } else { -frame.n };
    let f = contact_force(
        h.force,
        h.vel,
        m,
        n,
        STEP,
        &ContactSurface::BOX,
        0.02 - delta.abs(),
        0.8,
    );
    assert!(f.y > 0.0, "pushes up: {f}");
    assert_eq!(w.vehicles[1].nodes[0].force, before[1][0] + f);
    assert_eq!(t[1].force, before[0][1] - f * alpha);
    assert_eq!(t[2].force, before[0][2] - f * beta);
    assert_eq!(t[0].force, before[0][0] - f * gamma);
    // The contact conserves momentum: the weights sum to one.
    let reaction = f * alpha + f * beta + f * gamma;
    assert!((reaction - f).length() < 1e-4 * f.length());
}

#[test]
fn the_side_vote_follows_the_point_and_its_neighbours() {
    let n = Vec3::Y;
    let p0 = Vec3::ZERO;
    let below = Vec3::new(0.0, -1.0, 0.0);
    let above = Vec3::new(0.0, 1.0, 0.0);
    // Three neighbours or fewer: δ's sign alone (0 counts as front).
    assert!(vote_behind(-0.01, n, p0, [above, above, above].into_iter()));
    assert!(!vote_behind(0.0, n, p0, [below, below, below].into_iter()));
    assert!(!vote_behind(-0.0, n, p0, std::iter::empty()));
    // Behind the plane, four neighbours in front: −3 + 4 = 1, front.
    assert!(!vote_behind(-0.01, n, p0, [above; 4].into_iter()));
    // A tie (−3 + 4 − 1 = 0) counts as front.
    assert!(!vote_behind(
        -0.01,
        n,
        p0,
        [above, above, below, above, above].into_iter()
    ));
    assert!(vote_behind(
        -0.01,
        n,
        p0,
        [above, above, below, above, below].into_iter()
    ));
    // In front with four neighbours behind: 3 − 4 < 0, behind.
    assert!(vote_behind(0.01, n, p0, [below; 4].into_iter()));
    // A neighbour exactly in the plane votes front; NaN votes behind.
    assert!(!vote_behind(-0.01, n, p0, [Vec3::ZERO; 4].into_iter()));
    assert!(vote_behind(0.01, n, p0, [Vec3::NAN; 4].into_iter()));
}

#[test]
fn a_node_behind_the_plane_with_neighbours_in_front_is_pushed_to_the_front() {
    let owner = floor(Vec3::new(3.0, 3.0, 3.0));
    // A small pyramid whose apex (node 0) dipped 1 cm under the triangle: its four
    // neighbours are above.
    let mut p = vehicle(&[
        Vec3::new(0.0, -0.01, 0.0),
        Vec3::new(0.1, 0.1, 0.1),
        Vec3::new(-0.1, 0.1, 0.1),
        Vec3::new(0.1, 0.1, -0.1),
        Vec3::new(-0.1, 0.1, -0.1),
    ]);
    for i in 1..5 {
        p.beams.push(Beam::new(
            0,
            i,
            ExplicitKind::Plain,
            ExplicitType::Deformable,
            0.0,
            0.0,
            1.0,
            1.0,
            1e9,
            0.0,
        ));
    }
    p.finish();
    let mut w = ExplicitBeamsWorld::new(vec![owner, p]);
    for v in &mut w.vehicles {
        v.step();
    }
    let f0 = w.vehicles[1].nodes[0].force;
    w.contact_pass();
    let f = w.vehicles[1].nodes[0].force - f0;
    assert!(f.y > 0.0, "pushed up, not down: {f}");
}

/// A free plate: one triangle (corners `c`, x and z) in the plane y, unit masses.
fn plate(y: f32, c: [(f32, f32); 3], vy: f32) -> ExplicitBeams {
    let mut s = vehicle(&c.map(|(x, z)| Vec3::new(x, y, z)));
    for n in &mut s.nodes {
        n.vel = Vec3::new(0.0, vy, 0.0);
    }
    s.triangles.push(triangle([0, 1, 2]));
    s.finish();
    s
}

#[test]
fn swapping_vehicle_order_changes_the_result() {
    // Two overlapping plates 1 cm apart, each with a corner over the other's face:
    // in the later owner's turn, the earlier owner's reaction is already in the
    // force of its corner (a candidate).
    let a = plate(0.0, [(-1.0, -1.0), (1.0, -1.0), (0.0, 2.0)], 0.0);
    let b = plate(0.01, [(0.0, -0.5), (2.5, 2.5), (-2.5, 2.5)], -0.5);
    let mut ab = ExplicitBeamsWorld::new(vec![a.clone(), b.clone()]);
    let mut ba = ExplicitBeamsWorld::new(vec![b, a]);
    ab.step();
    ba.step();
    assert_eq!(ab.last.hits, 2);
    assert_eq!(ba.last.hits, 2);
    let fa = |w: &ExplicitBeamsWorld, i: usize| {
        w.vehicles[i]
            .nodes
            .iter()
            .map(|n| n.force)
            .collect::<Vec<_>>()
    };
    assert_ne!(fa(&ab, 0), fa(&ba, 1), "the order changes the result");
    assert_ne!(fa(&ab, 1), fa(&ba, 0));
}

#[test]
fn candidate_order_within_a_triangle_changes_the_result_by_rounding_only() {
    // Two partners' points under one triangle: partners visited 1 then 2, or 2
    // then 1 (swapping vehicles 1 and 2): each point's force reads only its own.
    let owner = floor(Vec3::new(3.0, 3.0, 3.0));
    let p1 = point(Vec3::new(0.1, 0.01, 0.1), Vec3::new(0.0, -0.7, 0.0));
    let p2 = point(Vec3::new(-0.1, 0.012, 0.2), Vec3::new(0.0, -0.4, 0.0));
    let mut x = ExplicitBeamsWorld::new(vec![owner.clone(), p1.clone(), p2.clone()]);
    let mut y = ExplicitBeamsWorld::new(vec![owner, p2, p1]);
    for _ in 0..20 {
        x.step();
        y.step();
    }
    let d1 = x.vehicles[1].nodes[0]
        .pos
        .distance(y.vehicles[2].nodes[0].pos);
    let d2 = x.vehicles[2].nodes[0]
        .pos
        .distance(y.vehicles[1].nodes[0].pos);
    assert!(d1 < 1e-6 && d2 < 1e-6, "{d1} {d2}");
}

/// A small sedan-like frame with a cab and tyres (synthetic).
fn frame_vehicle(at: Vec3, v: Vec3) -> ExplicitBeams {
    let mut s =
        ExplicitBeams::from_truck(CONTACT_TRUCK, at, v, ExplicitBeamsSettings::default()).unwrap();
    s.ground = 0.0;
    s
}

#[test]
fn vehicles_that_never_meet_are_bit_identical_to_running_alone() {
    let a = frame_vehicle(Vec3::new(0.0, 0.3, 0.0), Vec3::new(3.0, 0.0, 0.0));
    let b = frame_vehicle(Vec3::new(0.0, 0.3, 20.0), Vec3::new(-3.0, 0.0, 0.0));
    let mut alone = [a.clone(), b.clone()];
    let mut w = ExplicitBeamsWorld::new(vec![a, b]);
    for _ in 0..30 {
        w.frame(17);
        for s in &mut alone {
            s.check_wings();
            s.rebase_origin();
            for _ in 0..17 {
                s.step();
            }
        }
    }
    assert_eq!(w.total.partner_pairs, 0);
    for (x, y) in w.vehicles.iter().zip(&alone) {
        assert_eq!(x.nodes, y.nodes);
        assert_eq!(x.beams, y.beams);
    }
}

#[test]
fn the_contact_conserves_momentum() {
    // A plate falls onto another, both free in the air: gravity is the only
    // outside force, and the contact moves momentum between them.
    let a = plate(0.0, [(-1.0, -1.0), (1.0, -1.0), (0.0, 2.0)], 0.0);
    let b = plate(0.015, [(0.0, -0.5), (2.5, 2.5), (-2.5, 2.5)], -2.0);
    let momentum = |w: &ExplicitBeamsWorld| {
        w.vehicles
            .iter()
            .flat_map(|v| v.nodes.iter().map(|n| n.vel * n.mass))
            .fold(Vec3::ZERO, |s, p| s + p)
    };
    let mut w = ExplicitBeamsWorld::new(vec![a, b]);
    let mass: f32 = w
        .vehicles
        .iter()
        .flat_map(|v| v.nodes.iter().map(|n| n.mass))
        .sum();
    let m0 = momentum(&w);
    let steps = 40;
    for _ in 0..steps {
        w.step();
    }
    assert!(w.total.hits > 0, "they touched");
    let m1 = momentum(&w);
    // The first step integrates the spawn force (zero): gravity acts steps − 1.
    let gravity = Vec3::new(0.0, mass * GRAVITY * STEP * (steps - 1) as f32, 0.0);
    let err = (m1 - m0 - gravity).length();
    assert!(err < 1e-5 * mass, "momentum off by {err} (mass {mass})");
    // And the lower plate was pushed down.
    assert!(w.vehicles[0].nodes[2].vel.y < GRAVITY * STEP * steps as f32);
}
