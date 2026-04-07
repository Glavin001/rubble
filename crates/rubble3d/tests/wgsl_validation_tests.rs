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
fn validate_warmstart_hashmap_clear_wgsl() {
    validate_wgsl(
        "WARMSTART_HASHMAP_CLEAR_WGSL",
        rubble3d::gpu::WARMSTART_HASHMAP_CLEAR_WGSL,
    );
}

#[test]
fn validate_warmstart_hashmap_insert_wgsl() {
    validate_wgsl(
        "WARMSTART_HASHMAP_INSERT_WGSL",
        rubble3d::gpu::WARMSTART_HASHMAP_INSERT_WGSL,
    );
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
fn validate_adjacency_reset_wgsl() {
    validate_wgsl("ADJACENCY_RESET_WGSL", rubble3d::gpu::ADJACENCY_RESET_WGSL);
}

#[test]
fn validate_adjacency_count_wgsl() {
    validate_wgsl("ADJACENCY_COUNT_WGSL", rubble3d::gpu::ADJACENCY_COUNT_WGSL);
}

#[test]
fn validate_adjacency_init_ranges_wgsl() {
    validate_wgsl(
        "ADJACENCY_INIT_RANGES_WGSL",
        rubble3d::gpu::ADJACENCY_INIT_RANGES_WGSL,
    );
}

#[test]
fn validate_adjacency_scatter_wgsl() {
    validate_wgsl(
        "ADJACENCY_SCATTER_WGSL",
        rubble3d::gpu::ADJACENCY_SCATTER_WGSL,
    );
}

#[test]
fn validate_event_pair_keys_wgsl() {
    validate_wgsl("EVENT_PAIR_KEYS_WGSL", rubble3d::gpu::EVENT_PAIR_KEYS_WGSL);
}

#[test]
fn validate_build_render_instances_wgsl() {
    validate_wgsl(
        "BUILD_RENDER_INSTANCES_WGSL",
        rubble3d::gpu::BUILD_RENDER_INSTANCES_WGSL,
    );
}
