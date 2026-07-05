use glam::Vec4;
use rubble_math::{
    Contact3D, CONTACT_FLAG_STICKING, CONTACT_FLAG_WARMSTARTED,
};
use rubble3d::gpu::{
    DebugPhaseHarness3D, INACTIVE_BODY_COLOR, UNCOLORED_BODY,
};
use std::collections::BTreeMap;

const FEATURE_SKIP_PREFIX: u32 = 0x3200_0000;

fn try_harness(max_bodies: usize) -> Option<DebugPhaseHarness3D> {
    DebugPhaseHarness3D::new(max_bodies).ok()
}

fn contact(body_a: u32, body_b: u32, feature_id: u32) -> Contact3D {
    Contact3D {
        point: Vec4::new(body_a as f32, body_b as f32, 0.0, 0.1),
        normal: Vec4::new(0.0, 1.0, 0.0, 0.0),
        tangent: Vec4::new(1.0, 0.0, 0.0, 0.0),
        local_anchor_a: Vec4::new(0.11, 0.12, 0.13, 0.0),
        local_anchor_b: Vec4::new(-0.11, -0.12, -0.13, 0.0),
        lambda: Vec4::new(2.0, 3.0, 4.0, 0.0),
        penalty: Vec4::new(5.0, 6.0, 7.0, 0.0),
        body_a,
        body_b,
        feature_id,
        flags: CONTACT_FLAG_STICKING,
    }
}

fn coloring_hash(key: u32, seed: u32) -> u32 {
    let mut h = key ^ seed;
    h = h.wrapping_mul(0xcc9e_2d51);
    h = h.rotate_left(15);
    h = h.wrapping_mul(0x1b87_3593);
    h
}

fn warm_hash_u32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

fn warm_key(body_a: u32, body_b: u32, feature_id: u32) -> u32 {
    let lo = body_a.min(body_b);
    let hi = body_a.max(body_b);
    let mut key = warm_hash_u32(lo ^ hi.wrapping_mul(0x9e37_79b9) ^ feature_id.wrapping_mul(0x85eb_ca6b));
    if key == u32::MAX {
        key = u32::MAX - 1;
    }
    key
}

fn adjacency_reference(
    num_bodies: u32,
    contacts: &[Contact3D],
) -> (Vec<u32>, Vec<[u32; 2]>, Vec<Vec<(u32, u32)>>) {
    let mut counts = vec![0u32; num_bodies as usize];
    let mut active = vec![0u32; num_bodies as usize];
    for c in contacts {
        counts[c.body_a as usize] += 1;
        counts[c.body_b as usize] += 1;
        active[c.body_a as usize] = 1;
        active[c.body_b as usize] = 1;
    }

    let mut offsets = vec![0u32; num_bodies as usize];
    let mut running = 0u32;
    for (idx, count) in counts.iter().copied().enumerate() {
        offsets[idx] = running;
        running += count;
    }

    let ranges = offsets
        .iter()
        .zip(counts.iter())
        .map(|(&offset, &count)| [offset, count])
        .collect::<Vec<_>>();

    let mut per_body = vec![Vec::new(); num_bodies as usize];
    for (ci, c) in contacts.iter().enumerate() {
        per_body[c.body_a as usize].push((ci as u32, c.body_b));
        per_body[c.body_b as usize].push((ci as u32, c.body_a));
    }

    (active, ranges, per_body)
}

fn solve_schedule_reference(sorted_colors: &[u32]) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    let mut idx = 0u32;
    while (idx as usize) < sorted_colors.len() {
        let color = sorted_colors[idx as usize];
        if color == INACTIVE_BODY_COLOR || color == UNCOLORED_BODY {
            break;
        }
        let start = idx;
        while (idx as usize) < sorted_colors.len() && sorted_colors[idx as usize] == color {
            idx += 1;
        }
        let count = idx - start;
        let dispatch_x = ((count + 63) / 64).max(1);
        out.push((start, count, dispatch_x));
    }
    out
}

fn collision_features(body_a: u32, body_b: u32, capacity: u32) -> (u32, u32) {
    let mut slots = BTreeMap::new();
    for feature_id in 1..10_000u32 {
        let slot = warm_key(body_a, body_b, feature_id) % capacity;
        if let Some(prev) = slots.insert(slot, feature_id) {
            return (prev, feature_id);
        }
    }
    panic!("failed to find a warmstart collision");
}

#[test]
fn coloring_reset_matches_reference_exactly() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let active = vec![1, 1, 0, 1, 0];
    let seed = 0x00C0_FFEE;
    let snapshot = harness.run_coloring_reset(active.len() as u32, &active, seed);

    let expected_colors = vec![
        UNCOLORED_BODY,
        UNCOLORED_BODY,
        INACTIVE_BODY_COLOR,
        UNCOLORED_BODY,
        INACTIVE_BODY_COLOR,
    ];
    let expected_priorities = (0..active.len() as u32)
        .map(|idx| coloring_hash(idx, seed))
        .collect::<Vec<_>>();
    let expected_order = (0..active.len() as u32).collect::<Vec<_>>();

    assert_eq!(snapshot.body_colors, expected_colors);
    assert_eq!(snapshot.body_priorities, expected_priorities);
    assert_eq!(snapshot.body_order, expected_order);
    assert_eq!(snapshot.active_body_flags, active);
}

#[test]
fn solve_graph_counts_ranges_and_neighbor_sets_match_reference() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let contacts = vec![
        contact(0, 1, 10),
        contact(1, 2, 11),
        contact(1, 3, 12),
        contact(3, 4, 13),
    ];
    let snapshot = harness.run_solve_graph(6, &contacts);
    let (expected_active, expected_ranges, expected_sets) = adjacency_reference(6, &contacts);

    assert_eq!(snapshot.active_body_flags, expected_active);
    assert_eq!(snapshot.body_contact_ranges, expected_ranges);

    for (body, expected) in expected_sets.iter().enumerate() {
        let [offset, count] = snapshot.body_contact_ranges[body];
        let mut actual = (offset..offset + count)
            .map(|slot| {
                (
                    snapshot.body_contact_indices[slot as usize],
                    snapshot.body_contact_neighbors[slot as usize],
                )
            })
            .collect::<Vec<_>>();
        actual.sort_unstable();

        let mut expected_sorted = expected.clone();
        expected_sorted.sort_unstable();
        assert_eq!(actual, expected_sorted, "body {body} adjacency set mismatch");
    }
}

#[test]
fn solve_schedule_matches_reference_exactly_for_sorted_colors() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let sorted_colors = vec![0, 0, 1, 1, 1, 3, INACTIVE_BODY_COLOR, INACTIVE_BODY_COLOR];
    let snapshot = harness.run_solve_schedule_from_sorted_colors(&sorted_colors);
    let expected = solve_schedule_reference(&sorted_colors);

    assert_eq!(snapshot.run_count as usize, expected.len());
    for (idx, &(offset, count, dispatch_x)) in expected.iter().enumerate() {
        let run = snapshot.solve_runs[idx];
        let args = snapshot.dispatch_args[idx];
        assert_eq!(run.offset, offset);
        assert_eq!(run.count, count);
        assert_eq!(args.x, dispatch_x);
        assert_eq!(args.y, 1);
        assert_eq!(args.z, 1);
    }
}

#[test]
fn warmstart_exact_match_transfers_state_exactly() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let prev = vec![contact(0, 1, 77)];
    let cold = Contact3D {
        lambda: Vec4::ZERO,
        penalty: Vec4::ZERO,
        flags: 0,
        ..contact(0, 1, 77)
    };
    let snapshot = harness.run_warmstart(&prev, &[cold], 0.99, 1.0e4);
    let warmed = snapshot.contacts[0];
    let expected_scale = 0.95 * 0.99;

    assert!((warmed.lambda.x - prev[0].lambda.x * expected_scale).abs() < 1.0e-6);
    assert!((warmed.lambda.y - prev[0].lambda.y * expected_scale).abs() < 1.0e-6);
    assert!((warmed.lambda.z - prev[0].lambda.z * expected_scale).abs() < 1.0e-6);
    assert_eq!(warmed.tangent, prev[0].tangent);
    assert_eq!(warmed.local_anchor_a, prev[0].local_anchor_a);
    assert_eq!(warmed.local_anchor_b, prev[0].local_anchor_b);
    assert_ne!(warmed.flags & CONTACT_FLAG_WARMSTARTED, 0);
}

#[test]
fn warmstart_mismatch_and_skip_prefix_leave_new_contacts_untouched() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let prev = vec![contact(0, 1, 77), contact(0, 1, FEATURE_SKIP_PREFIX | 12)];
    let mismatch = Contact3D {
        lambda: Vec4::new(9.0, 8.0, 7.0, 0.0),
        penalty: Vec4::new(6.0, 5.0, 4.0, 0.0),
        flags: 0,
        ..contact(0, 1, 78)
    };
    let skip = Contact3D {
        lambda: Vec4::new(3.0, 2.0, 1.0, 0.0),
        penalty: Vec4::new(7.0, 8.0, 9.0, 0.0),
        flags: 0,
        ..contact(0, 1, FEATURE_SKIP_PREFIX | 12)
    };
    let snapshot = harness.run_warmstart(&prev, &[mismatch, skip], 0.99, 1.0e4);

    assert_eq!(snapshot.contacts[0].lambda, mismatch.lambda);
    assert_eq!(snapshot.contacts[0].penalty, mismatch.penalty);
    assert_eq!(snapshot.contacts[0].flags & CONTACT_FLAG_WARMSTARTED, 0);

    assert_eq!(snapshot.contacts[1].lambda, skip.lambda);
    assert_eq!(snapshot.contacts[1].penalty, skip.penalty);
    assert_eq!(snapshot.contacts[1].flags & CONTACT_FLAG_WARMSTARTED, 0);
}

#[test]
fn warmstart_hash_collisions_still_resolve_exact_matches() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let (feature_a, feature_b) = collision_features(0, 1, 64);
    let prev_a = Contact3D {
        lambda: Vec4::new(1.0, 1.0, 1.0, 0.0),
        penalty: Vec4::new(2.0, 2.0, 2.0, 0.0),
        ..contact(0, 1, feature_a)
    };
    let prev_b = Contact3D {
        lambda: Vec4::new(4.0, 5.0, 6.0, 0.0),
        penalty: Vec4::new(7.0, 8.0, 9.0, 0.0),
        ..contact(0, 1, feature_b)
    };
    let cold_b = Contact3D {
        lambda: Vec4::ZERO,
        penalty: Vec4::ZERO,
        flags: 0,
        ..contact(0, 1, feature_b)
    };

    let snapshot = harness.run_warmstart(&[prev_a, prev_b], &[cold_b], 0.99, 1.0e4);
    let warmed = snapshot.contacts[0];
    let expected_scale = 0.95 * 0.99;

    assert!((warmed.lambda.x - 4.0 * expected_scale).abs() < 1.0e-6);
    assert!((warmed.lambda.y - 5.0 * expected_scale).abs() < 1.0e-6);
    assert!((warmed.lambda.z - 6.0 * expected_scale).abs() < 1.0e-6);
    assert_ne!(warmed.flags & CONTACT_FLAG_WARMSTARTED, 0);
}
