use glam::Vec4;
use rubble_math::{Contact3D, CONTACT_FLAG_STICKING, CONTACT_FLAG_WARMSTARTED};
use rubble3d::gpu::{
    DebugColoringSnapshot3D, DebugPhaseHarness3D, DebugSolveGraphSnapshot3D,
    DebugSolveScheduleSnapshot3D, INACTIVE_BODY_COLOR, UNCOLORED_BODY,
};

fn try_harness(max_bodies: usize) -> Option<DebugPhaseHarness3D> {
    DebugPhaseHarness3D::new(max_bodies).ok()
}

fn contact(body_a: u32, body_b: u32, feature_id: u32) -> Contact3D {
    Contact3D {
        point: Vec4::new(body_a as f32, body_b as f32, 0.0, 0.1),
        normal: Vec4::new(0.0, 1.0, 0.0, 0.0),
        tangent: Vec4::new(1.0, 0.0, 0.0, 0.0),
        local_anchor_a: Vec4::new(0.1, 0.2, 0.3, 0.0),
        local_anchor_b: Vec4::new(-0.1, -0.2, -0.3, 0.0),
        lambda: Vec4::new(1.0, 2.0, 3.0, 0.0),
        penalty: Vec4::new(4.0, 5.0, 6.0, 0.0),
        body_a,
        body_b,
        feature_id,
        flags: CONTACT_FLAG_STICKING,
    }
}

fn assert_graph_snapshot_consistent(snapshot: &DebugSolveGraphSnapshot3D, contacts: &[Contact3D]) {
    assert_eq!(snapshot.active_body_flags.len(), snapshot.num_bodies as usize);
    assert_eq!(snapshot.body_contact_ranges.len(), snapshot.num_bodies as usize);
    assert_eq!(
        snapshot.body_contact_indices.len(),
        (snapshot.contact_count * 2) as usize
    );
    assert_eq!(
        snapshot.body_contact_neighbors.len(),
        (snapshot.contact_count * 2) as usize
    );

    let mut total_refs = 0usize;
    let mut last_end = 0u32;
    for (body, &[offset, count]) in snapshot.body_contact_ranges.iter().enumerate() {
        assert!(
            offset >= last_end,
            "body {body} offset {} regressed below previous end {last_end}",
            offset
        );
        let end = offset + count;
        assert!(
            end as usize <= snapshot.body_contact_indices.len(),
            "body {body} range [{offset}, {end}) exceeds {} refs",
            snapshot.body_contact_indices.len()
        );
        let active = snapshot.active_body_flags[body] != 0;
        assert_eq!(
            active,
            count > 0,
            "body {body} active flag {} disagrees with count {count}",
            snapshot.active_body_flags[body]
        );
        for slot in offset..end {
            let ci = snapshot.body_contact_indices[slot as usize] as usize;
            assert!(ci < contacts.len(), "slot {slot} references contact {ci} out of bounds");
            let c = contacts[ci];
            assert!(
                c.body_a == body as u32 || c.body_b == body as u32,
                "slot {slot} body {body} points at unrelated contact {:?}",
                (c.body_a, c.body_b, c.feature_id)
            );
            let expected_neighbor = if c.body_a == body as u32 {
                c.body_b
            } else {
                c.body_a
            };
            assert_eq!(
                snapshot.body_contact_neighbors[slot as usize],
                expected_neighbor,
                "slot {slot} has wrong neighbor"
            );
        }
        total_refs += count as usize;
        last_end = end;
    }
    assert_eq!(total_refs, (snapshot.contact_count * 2) as usize);
}

fn assert_coloring_snapshot_consistent(snapshot: &DebugColoringSnapshot3D, contacts: &[Contact3D]) {
    assert_eq!(snapshot.body_colors.len(), snapshot.num_bodies as usize);
    assert_eq!(snapshot.body_order.len(), snapshot.num_bodies as usize);
    assert_eq!(snapshot.active_body_flags.len(), snapshot.num_bodies as usize);

    let mut seen = vec![false; snapshot.num_bodies as usize];
    for &body in &snapshot.body_order {
        assert!((body as usize) < seen.len(), "body_order contains out-of-range body {body}");
        assert!(!seen[body as usize], "body_order is not a permutation: repeats body {body}");
        seen[body as usize] = true;
    }
    assert!(seen.into_iter().all(|hit| hit), "body_order missed some bodies");

    for (body, &color) in snapshot.body_colors.iter().enumerate() {
        if snapshot.active_body_flags[body] == 0 {
            assert_eq!(color, INACTIVE_BODY_COLOR, "inactive body {body} should use sentinel");
        } else {
            assert_ne!(color, INACTIVE_BODY_COLOR, "active body {body} marked inactive");
            assert_ne!(color, UNCOLORED_BODY, "active body {body} left uncolored");
        }
    }

    for contact in contacts {
        let a = contact.body_a as usize;
        let b = contact.body_b as usize;
        assert_ne!(
            snapshot.body_colors[a], snapshot.body_colors[b],
            "adjacent bodies {} and {} share color {}",
            a, b, snapshot.body_colors[a]
        );
    }
}

fn assert_schedule_snapshot_consistent(snapshot: &DebugSolveScheduleSnapshot3D) {
    assert_eq!(snapshot.solve_runs.len(), snapshot.run_count as usize);
    assert_eq!(snapshot.dispatch_args.len(), snapshot.run_count as usize);

    for window in snapshot.solve_runs.windows(2) {
        let a = window[0];
        let b = window[1];
        assert_eq!(a.offset + a.count, b.offset, "solve runs are not contiguous");
    }

    for (idx, run) in snapshot.solve_runs.iter().enumerate() {
        let end = run.offset + run.count;
        assert!(
            end <= snapshot.num_bodies,
            "solve run {idx} [{}, {}) exceeds {} bodies",
            run.offset,
            end,
            snapshot.num_bodies
        );
        let expected_x = ((run.count + 63) / 64).max(1);
        let args = snapshot.dispatch_args[idx];
        assert_eq!(args.x, expected_x, "solve run {idx} indirect x mismatch");
        assert_eq!(args.y, 1, "solve run {idx} indirect y should be 1");
        assert_eq!(args.z, 1, "solve run {idx} indirect z should be 1");
    }
}

#[test]
fn solve_graph_invariants_hold_for_mixed_contact_graph() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let contacts = vec![contact(0, 1, 10), contact(1, 2, 11), contact(1, 3, 12)];
    let snapshot = harness.run_solve_graph(5, &contacts);
    assert_graph_snapshot_consistent(&snapshot, &contacts);
}

#[test]
fn solve_graph_clears_stale_state_when_contacts_drop_to_zero() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let contacts = vec![contact(0, 1, 10), contact(1, 2, 11), contact(2, 3, 12)];
    let _ = harness.run_solve_graph(4, &contacts);
    let cleared = harness.run_solve_graph(4, &[]);

    assert_eq!(cleared.contact_count, 0);
    assert!(cleared.active_body_flags.iter().all(|&flag| flag == 0));
    assert!(cleared.body_contact_ranges.iter().all(|&range| range == [0, 0]));
    assert!(cleared.body_contact_indices.is_empty());
    assert!(cleared.body_contact_neighbors.is_empty());
}

#[test]
fn coloring_invariants_hold_and_inactive_bodies_use_sentinel() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let contacts = vec![contact(0, 1, 10), contact(1, 2, 11), contact(1, 3, 12)];
    let snapshot = harness.run_coloring(5, &contacts, 0xC0FFEE);
    assert_coloring_snapshot_consistent(&snapshot, &contacts);
}

#[test]
fn solve_schedule_resets_cleanly_on_empty_sorted_colors() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let populated = harness.run_solve_schedule_from_sorted_colors(&[
        0,
        0,
        1,
        2,
        INACTIVE_BODY_COLOR,
        INACTIVE_BODY_COLOR,
    ]);
    assert!(populated.run_count > 0);
    assert_schedule_snapshot_consistent(&populated);

    let empty = harness.run_solve_schedule_from_sorted_colors(&[
        INACTIVE_BODY_COLOR,
        INACTIVE_BODY_COLOR,
        INACTIVE_BODY_COLOR,
    ]);
    assert_eq!(empty.run_count, 0);
    assert!(empty.solve_runs.is_empty());
    assert!(empty.dispatch_args.is_empty());
}

#[test]
fn warmstart_snapshot_reports_only_true_matches() {
    let Some(mut harness) = try_harness(8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let prev = vec![contact(0, 1, 99)];
    let matched = contact(0, 1, 99);
    let mismatched = contact(0, 1, 100);
    let snapshot = harness.run_warmstart(&prev, &[matched, mismatched], 0.99, 1.0e4);
    assert_eq!(snapshot.prev_contact_count, 1);
    assert_eq!(snapshot.new_contact_count, 2);
    assert_eq!(snapshot.contacts.len(), 2);
    assert_ne!(
        snapshot.contacts[0].flags & CONTACT_FLAG_WARMSTARTED,
        0,
        "exact match should be flagged warmstarted"
    );
    assert_eq!(
        snapshot.contacts[1].flags & CONTACT_FLAG_WARMSTARTED,
        0,
        "feature mismatch should stay cold"
    );
}
