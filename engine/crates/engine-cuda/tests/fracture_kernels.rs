//! The fracture world's kernels as built for CUDA (feature `stress`, build.rs
//! `fracture_kernels`): every compute entry of the stress shaders the World's
//! destruction runs is there, once, with its globals bound. CPU only: the build's own
//! checks (Slang's deferred-load hazard, the `{entry}_gs` rewrite) ran to make the table.
//!
//!   cargo test -p engine-cuda --features stress --test fracture_kernels
#![cfg(feature = "stress")]

use engine_cuda::fracture::{kernel, CUDA_FRACTURE_KERNELS};
use std::collections::{BTreeMap, HashSet};

/// The modules and the count of compute entries each declares
/// (`grep -c '\[shader("compute")\]'`). A new entry in a shader changes its count here
/// on purpose: the plan that dispatches it is the next thing to change.
const MODULES: &[(&str, usize)] = &[
    ("coupling", 10),
    ("fracture", 96),
    ("fracture_log", 3),
    ("blast_fracture", 3),
    ("blast_topology", 13),
    ("blast_material", 1),
    ("blast_native", 1),
    ("blast_native_wide", 36),
    ("blast_wide_gate", 1),
    ("blast_motion", 2),
    ("blast_cgls", 1),
    ("blast_virial", 1),
    ("blast_view", 2),
];

#[test]
fn every_module_has_its_entries() {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for k in CUDA_FRACTURE_KERNELS {
        *counts.entry(k.module).or_default() += 1;
    }
    let want: BTreeMap<&str, usize> = MODULES.iter().copied().collect();
    assert_eq!(counts, want);
}

#[test]
fn entries_are_unique_and_bound() {
    let mut seen = HashSet::new();
    for k in CUDA_FRACTURE_KERNELS {
        assert!(
            seen.insert((k.module, k.entry)),
            "{}.{} twice",
            k.module,
            k.entry
        );
        assert!(k.cu_bytes > 0, "{}.{}: no CUDA source", k.module, k.entry);
        assert!(
            k.threads.iter().all(|&t| t >= 1),
            "{}.{}: threads {:?}",
            k.module,
            k.entry,
            k.threads
        );
        assert!(
            k.threads.iter().product::<u32>() <= 1024,
            "{}.{}: threads {:?}",
            k.module,
            k.entry,
            k.threads
        );
        // Globals do not overlap and fit the parameter block.
        let mut fields: Vec<_> = k.binds.to_vec();
        fields.sort();
        for w in fields.windows(2) {
            assert!(
                w[0].0 + w[0].1 <= w[1].0,
                "{}.{}: globals overlap",
                k.module,
                k.entry
            );
        }
        for &(offset, size, _) in k.binds {
            assert!(
                size == 8 || size == 16,
                "{}.{}: a global of {size} bytes",
                k.module,
                k.entry
            );
            assert!(offset + size <= k.params_bytes);
        }
        // One global a binding.
        let bindings: HashSet<u32> = k.binds.iter().map(|b| b.2).collect();
        assert_eq!(
            bindings.len(),
            k.binds.len(),
            "{}.{}: a binding bound twice",
            k.module,
            k.entry
        );
    }
}

#[test]
fn the_frame_s_entries_are_found_by_name() {
    for (module, entry) in [
        ("coupling", "ingest_rows"),
        ("coupling", "ingest_gate"),
        ("fracture", "fr_loads"),
        ("fracture", "fr_breaks"),
        ("fracture", "fr_gate"),
        ("fracture", "fr_quiet"),
        ("fracture", "fr_uf_union"),
        ("fracture", "fr_commit"),
        ("fracture_log", "log_bonds"),
        ("blast_wide_gate", "w_gate"),
        ("blast_native_wide", "w_bond_r"),
    ] {
        assert!(kernel(module, entry).is_some(), "{module}.{entry}");
    }
}

/// On a CUDA device: every module loads, with its entry, its `{entry}_gs` and its
/// parameter block. Skips where there is no device or the build had no nvcc.
#[test]
fn every_module_loads_on_the_device() {
    if !engine_cuda::CudaSim::available() {
        eprintln!("skipped: no CUDA device, or the build has no device code");
        return;
    }
    let loaded = engine_cuda::fracture::load_all().expect("the fracture kernels load");
    eprintln!("loaded {loaded} modules");
    assert_eq!(loaded, CUDA_FRACTURE_KERNELS.len());
}
