//! CB3 — native unit tests of the `rubble-wasm` marshaling layer.
//!
//! `rubble-wasm` previously had **zero** tests; its hand-written transform
//! packing, shape-size/offset bookkeeping, and `remove_body` semantics were
//! exercised only indirectly by a few coarse Playwright specs. These run on the
//! native build (the wasm-binding methods are plain Rust; only `step`/`copy_*_into`
//! use JS types, which these tests avoid). They pin the boundary contract so a
//! marshaling regression — the exact "web integration layer is the root cause"
//! surface — fails a fast Rust test instead of shipping.
//!
//! Note: `create()` is async (WebGPU adapter negotiation); we drive it natively
//! with `pollster::block_on`. Tests skip gracefully without a GPU adapter.

#![cfg(not(target_arch = "wasm32"))]

use rubble_wasm::PhysicsWorld3D;

fn make() -> Option<PhysicsWorld3D> {
    pollster::block_on(PhysicsWorld3D::create(0.0, -9.81, 0.0, 1.0 / 60.0)).ok()
}

#[test]
fn shape_arrays_index_correctly_and_transforms_are_7_per_handle() {
    let Some(mut w) = make() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    // sphere (type 0, 1 size: radius), box (type 1, 3 sizes), capsule (type 2, 2 sizes).
    let _s = w.add_sphere(0.0, 5.0, 0.0, 0.5, 1.0);
    let _b = w.add_box(2.0, 5.0, 0.0, 0.4, 0.3, 0.2, 1.0);
    let _c = w.add_capsule(-2.0, 5.0, 0.0, 0.6, 0.25, 1.0);

    assert_eq!(w.handle_count(), 3);
    assert_eq!(w.body_count(), 3);

    // Transform array is exactly 7 floats (xyz + quat xyzw) per handle.
    assert_eq!(
        w.get_transforms().len(),
        3 * 7,
        "transform array must be 7 floats per handle"
    );

    // Shape-type codes.
    assert_eq!(w.get_shape_types(), vec![0u32, 1, 2]);

    // Offsets must index the flat sizes array correctly: sphere@0 (1), box@1 (3),
    // capsule@4 (2) -> total 6 size floats.
    let offsets = w.get_shape_size_offsets();
    let sizes = w.get_shape_sizes();
    assert_eq!(offsets, vec![0u32, 1, 4], "shape-size offsets misaligned");
    assert_eq!(sizes.len(), 6, "flat shape-size array length wrong");
    assert!((sizes[0] - 0.5).abs() < 1e-6, "sphere radius mis-stored");
    assert!(
        (sizes[1] - 0.4).abs() < 1e-6 && (sizes[3] - 0.2).abs() < 1e-6,
        "box half-extents mis-stored"
    );
    assert!(
        (sizes[4] - 0.6).abs() < 1e-6 && (sizes[5] - 0.25).abs() < 1e-6,
        "capsule half-height/radius mis-stored"
    );
}

#[test]
fn remove_body_keeps_slot_and_renders_removed_at_origin() {
    let Some(mut w) = make() else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let _a = w.add_sphere(0.0, 5.0, 0.0, 0.5, 1.0); // slot 0
    let _b = w.add_sphere(2.0, 5.0, 0.0, 0.5, 1.0); // slot 1
    assert_eq!(w.handle_count(), 2);
    assert_eq!(w.body_count(), 2);

    assert!(
        w.remove_body(1),
        "remove_body should succeed for a live slot"
    );

    // CURRENT (documented) behaviour: `remove_body` does NOT pop the handle/shape
    // arrays, so handle_count is unchanged while alive body_count drops, and the
    // removed slot's transform reads identity-at-origin (the `None -> ZERO/IDENTITY`
    // substitution). Pinning this means a change to removal semantics — e.g. when
    // remove_body is reworked for GPU-residency — is caught here rather than as a
    // body silently rendering at the world origin in the browser.
    assert_eq!(
        w.handle_count(),
        2,
        "handles array is not compacted on remove"
    );
    assert_eq!(w.body_count(), 1, "alive count must drop after remove");

    let t = w.get_transforms();
    assert_eq!(t.len(), 2 * 7);
    let removed = &t[7..14];
    assert_eq!(
        removed,
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "removed slot must read identity-at-origin (documents the aliasing behaviour)"
    );

    // Out-of-range removal is rejected.
    assert!(!w.remove_body(99), "out-of-range remove must return false");
}
