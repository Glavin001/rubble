//! Every compute entry point binds at most WebGPU's storage buffers per stage (10, the
//! default and Chrome's limit), or has phases that do (`reflection::PHASED`, which
//! `Gpu::staged` runs on such a device). Static: from `build.rs`'s reflection of the
//! generated WGSL; no GPU work runs.

use stress_gpu::reflection::{entry_storage, phases, ENTRY_BINDINGS, PHASED};

/// WebGPU's default `maxStorageBuffersPerShaderStage`.
const WEBGPU_STORAGE: u32 = 10;

#[test]
fn every_entry_fits_webgpu_or_has_phases_that_do() {
    let mut over = Vec::new();
    for &(module, entry, _, storage, _) in ENTRY_BINDINGS {
        if storage <= WEBGPU_STORAGE {
            continue;
        }
        match phases(module, entry) {
            None => over.push(format!("{module}::{entry} uses {storage} storage buffers and has no phases")),
            Some(list) => {
                for phase in list {
                    match entry_storage(module, phase) {
                        None => over.push(format!("{module}::{entry}: phase {phase} is not an entry point")),
                        Some(n) if n > WEBGPU_STORAGE => over.push(format!("{module}::{entry}: phase {phase} uses {n} storage buffers")),
                        Some(_) => {}
                    }
                }
            }
        }
    }
    assert!(over.is_empty(), "entry points over WebGPU's {WEBGPU_STORAGE} storage buffers per stage:\n  {}", over.join("\n  "));
}

/// Each generated compute entry point is in its shader set, at the workgroup size the
/// shader declares (the Metal path passes both through: an entry missing from the set
/// cannot be built there).
#[test]
fn shader_sets_list_every_compute_entry() {
    let mut wrong = Vec::new();
    for &(module, entry, _, _, size) in ENTRY_BINDINGS {
        let Some(set) = stress_gpu::shaders::ALL.iter().find(|s| s.name == module) else {
            wrong.push(format!("{module}: no shader set"));
            continue;
        };
        // Render stages are listed at (1, 1, 1); Naga gives them (0, 0, 0).
        if size == (0, 0, 0) {
            continue;
        }
        match set.entries.iter().find(|e| e.0 == entry) {
            None => wrong.push(format!("{module}::{entry} is not in the set")),
            Some(e) if e.1 != size => wrong.push(format!("{module}::{entry}: the set says {:?}, the shader {size:?}", e.1)),
            Some(_) => {}
        }
    }
    assert!(wrong.is_empty(), "shader sets out of step with the generated shaders:\n  {}", wrong.join("\n  "));
}

#[test]
fn phase_table_names_real_entries() {
    for &(module, entry, list) in PHASED {
        assert!(entry_storage(module, entry).is_some(), "{module}::{entry} is not an entry point");
        assert!(!list.is_empty(), "{module}::{entry} has an empty phase list");
        for phase in list {
            assert!(entry_storage(module, phase).is_some(), "{module}::{entry}: phase {phase} is not an entry point");
        }
    }
}

/// The fracture without crushing binds at most 8 storage buffers per entry (WebGPU's
/// default limit): only the crush entries (`fracture::CRUSH_ENTRIES`, built only with
/// crushing) may bind more, and the fracture lists every entry the shader has.
#[test]
fn fracture_core_fits_eight_storage_buffers() {
    use stress_gpu::fracture::{entries, CRUSH_ENTRIES};
    let mut wrong = Vec::new();
    for &(module, entry, _, storage, _) in ENTRY_BINDINGS {
        if module != "fracture" {
            continue;
        }
        if !entries().contains(&entry) {
            wrong.push(format!("{entry} is not in fracture.rs ENTRIES"));
        }
        if storage > 8 && !CRUSH_ENTRIES.contains(&entry) {
            wrong.push(format!("{entry} uses {storage} storage buffers and is no crush entry"));
        }
    }
    for e in CRUSH_ENTRIES {
        if !entries().contains(&e) {
            wrong.push(format!("crush entry {e} is not in ENTRIES"));
        }
    }
    assert!(wrong.is_empty(), "fracture entries:\n  {}", wrong.join("\n  "));
}
