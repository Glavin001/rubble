//! Compile-time WGSL shader validation using naga.
//!
//! These tests parse and validate every WGSL shader string in rubble3d
//! through naga's WGSL frontend + validator. This catches shader bugs
//! before they reach WebGPU browsers at runtime.

fn validate_wgsl(name: &str, source: &str) {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(m) => m,
        Err(e) => panic!("{name}: WGSL parse error:\n{e}"),
    };

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    if let Err(e) = validator.validate(&module) {
        panic!("{name}: WGSL validation error:\n{e}");
    }
}

#[test]
fn validate_narrowphase_wgsl() {
    validate_wgsl("NARROWPHASE_WGSL", rubble3d::gpu::NARROWPHASE_WGSL);
}

#[test]
fn validate_predict_wgsl() {
    validate_wgsl("PREDICT_WGSL", rubble3d::gpu::PREDICT_WGSL);
}

#[test]
fn validate_extract_velocity_wgsl() {
    validate_wgsl(
        "EXTRACT_VELOCITY_WGSL",
        rubble3d::gpu::EXTRACT_VELOCITY_WGSL,
    );
}

#[test]
fn validate_avbd_primal_wgsl() {
    validate_wgsl("AVBD_PRIMAL_WGSL", rubble3d::gpu::AVBD_PRIMAL_WGSL);
}

#[test]
fn validate_avbd_dual_wgsl() {
    validate_wgsl("AVBD_DUAL_WGSL", rubble3d::gpu::AVBD_DUAL_WGSL);
}

#[test]
fn validate_warmstart_match_wgsl() {
    validate_wgsl("WARMSTART_MATCH_WGSL", rubble3d::gpu::WARMSTART_MATCH_WGSL);
}

#[test]
fn validate_contact_prep_wgsl() {
    validate_wgsl("CONTACT_PREP_WGSL", rubble3d::gpu::CONTACT_PREP_WGSL);
}

#[test]
fn validate_restitution_wgsl() {
    validate_wgsl("RESTITUTION_WGSL", rubble3d::gpu::RESTITUTION_WGSL);
}

#[test]
fn validate_coloring_reset_wgsl() {
    validate_wgsl("COLORING_RESET_WGSL", rubble3d::gpu::COLORING_RESET_WGSL);
}

#[test]
fn validate_coloring_step_wgsl() {
    validate_wgsl("COLORING_STEP_WGSL", rubble3d::gpu::COLORING_STEP_WGSL);
}

#[test]
fn validate_coloring_apply_wgsl() {
    validate_wgsl("COLORING_APPLY_WGSL", rubble3d::gpu::COLORING_APPLY_WGSL);
}

#[test]
fn validate_contact_stream_wgsl() {
    validate_wgsl(
        "SORT_KEY_EXTRACT_WGSL",
        rubble3d::gpu::SORT_KEY_EXTRACT_WGSL,
    );
    validate_wgsl("SORT_IOTA_WGSL", rubble3d::gpu::SORT_IOTA_WGSL);
    validate_wgsl("SORT_GATHER_WGSL", rubble3d::gpu::SORT_GATHER_WGSL);
    validate_wgsl("ADJ_CLEAR_WGSL", rubble3d::gpu::ADJ_CLEAR_WGSL);
    validate_wgsl("ADJ_COUNT_WGSL", rubble3d::gpu::ADJ_COUNT_WGSL);
    validate_wgsl("ADJ_RANGES_WGSL", rubble3d::gpu::ADJ_RANGES_WGSL);
    validate_wgsl("ADJ_INCIDENCE_WGSL", rubble3d::gpu::ADJ_INCIDENCE_WGSL);
}
