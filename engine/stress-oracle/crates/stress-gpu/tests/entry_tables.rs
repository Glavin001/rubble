//! The hand-kept entry tables (`shaders::*`, what a pipeline's entry point and a direct
//! dispatch's workgroup count are looked up in) against the generated WGSL's own entry
//! points and workgroup sizes (build.rs `ENTRY_BINDINGS`). CPU only.
//!
//! An entry missing from its table cannot be found by the Metal library path, and one
//! with the wrong size dispatches the wrong number of workgroups; a host that takes the
//! sizes from its own build (CUDA: Slang's reflection) would then disagree with this one.

use stress_gpu::{reflection::ENTRY_BINDINGS, shaders};

#[test]
fn every_compute_entry_is_in_its_table_with_its_workgroup_size() {
    let mut wrong = Vec::new();
    for &(module, entry, _, _, size) in ENTRY_BINDINGS {
        // Render stages have no workgroup; the tables list them as (1, 1, 1).
        if size == (0, 0, 0) {
            continue;
        }
        let Some(set) = shaders::ALL.iter().find(|s| s.name == module) else {
            wrong.push(format!("{module}: no shader set"));
            continue;
        };
        match set.entries.iter().find(|e| e.0 == entry) {
            None => wrong.push(format!("{module}::{entry}: not in the table (workgroup {size:?})")),
            Some(e) if e.1 != size => wrong.push(format!("{module}::{entry}: the table says {:?}, the shader {size:?}", e.1)),
            Some(_) => {}
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
