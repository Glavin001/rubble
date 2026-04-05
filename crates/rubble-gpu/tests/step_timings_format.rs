use rubble_gpu::{BroadphaseBreakdownMs, StepTimingsMs};

#[test]
fn format_text_overlay_includes_all_broadphase_rows_even_when_zero() {
    let mut t = StepTimingsMs {
        upload_ms: 1.0,
        predict_aabb_ms: 2.0,
        narrowphase_ms: 3.0,
        contact_fetch_ms: 4.0,
        solve_ms: 5.0,
        extract_ms: 6.0,
        ..Default::default()
    };
    t.set_broadphase_breakdown(BroadphaseBreakdownMs::default());
    let s = t.format_text_overlay("Test", 0.5);
    assert!(s.contains("    Bounds"));
    assert!(s.contains("    Sort"));
    assert!(s.contains("    Build"));
    assert!(s.contains("    Traverse"));
    assert!(s.contains("    Readback"));
    assert!(s.contains("Render      (Test)"));
}
