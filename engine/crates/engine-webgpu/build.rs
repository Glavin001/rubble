use anyhow::Result;
use engine_slang_build::{compile, compile_with_defines, find_slangc, par_map, SlangTarget};
use std::env;
use std::path::{Path, PathBuf};

/// Same world_step entries as `engine-metal`, minus `broadphase_grid` (thin wrapper;
/// slangc SIGSEGV on SPIR-V — runtime aliases to `broadphase_query`).
const WORLD_ENTRIES: &[&str] = &[
    "solve_primal_sparse",
    "solve_primal_sparse_compact",
    "solve_primal_sparse_persistent",
    "narrowphase_copy_persistent",
    "narrowphase_defer_copy_persistent",
    "narrowphase_fresh_copy_persistent",
    "csr_degree_contacts_persistent",
    "persist_list_copy_persistent",
    "write_work_args_copy_persistent",
    "csr_scatter_contacts_persistent",
    "persist_contacts_copy_persistent",
    "apply_commands",
    "apply_commands_persistent",
    "narrowphase_persistent",
    "contact_hash_insert_persistent",
    "contact_hash_insert_manifolds_persistent",
    "persist_contacts_persistent",
    "persist_manifolds_persistent",
    "pack_solver_pairs_persistent",
    "solve_primal_color_persistent",
    "solve_primal_one_persistent",
    "solve_dual_persistent",
    "predict_bodies",
    "build_aabbs",
    "joint_hash_clear",
    "joint_hash_insert",
    "broadphase_clear",
    "broadphase_insert",
    "broadphase_query",
    "broadphase_query_awake",
    "broadphase_query_finer",
    "write_finer_args",
    "write_moved_args",
    "list_awake_colliders",
    "write_awake_collider_args",
    "rest_clear",
    "rest_insert",
    "rest_publish",
    "broadphase_insert_awake",
    "broadphase_clear_awake",
    "narrowphase",
    "update_refs",
    "contact_hash_clear",
    "contact_hash_insert",
    "contact_hash_insert_manifolds",
    "persist_contacts",
    "persist_manifolds",
    "persist_manifolds_compact",
    "warmstart_joints",
    "bond_wake",
    "build_awake_list",
    "csr_clear",
    "csr_degree_contacts",
    "csr_degree_joints",
    "csr_prefix",
    "csr_prefix_blocks",
    "csr_prefix_block_sums",
    "csr_prefix_add",
    "csr_prefix_init",
    "csr_prefix_scan",
    "csr_prefix_swap",
    "csr_prefix_finalize",
    "csr_scatter_contacts",
    "csr_scatter_joints",
    "csr_sort",
    "csr_sort_append",
    "color_bodies",
    "color_bodies_local",
    "color_bodies_tail",
    "carry_rest_pairs",
    "init_colors",
    "pack_color_lists",
    "count_color_tiles",
    "prefix_color_tiles",
    "prefix_color_tiles_parallel",
    "scatter_color_tiles",
    "write_color_args",
    "write_work_args",
    "write_sleep_args",
    "clear_solver_contacts",
    "pack_solver_contacts",
    "solve_primal_color_compact",
    "solve_primal_one_compact",
    "solve_dual_compact",
    "solve_primal_color",
    "solve_primal_one",
    "solve_dual",
    "sleep_island_gate",
    "sleep_island_commit",
    "record_stab_velocity",
    "extract_velocity_sleep",
    "recompute_inertia",
    // Continuous collision and step-start impulses (ccd_step.slang).
    "apply_impulses",
    "ccd_register",
    "broadphase_fast",
    "ccd_toi",
    "ccd_pick",
    "ccd_wake",
    "ccd_mark",
    "ccd_respond",
    "ccd_respond_apply",
    "ccd_trust",
    "ccd_unhold",
    "ccd_validate",
    "ccd_clamp",
    "ccd_emit",
    "ccd_emit_persistent",
    "ccd_emit_copy_persistent",
    "record_color_degrees",
    "reuse_all_manifolds",
    "reuse_all_manifolds_persistent",
    "convex_classify",
    "convex_classify_persistent",
    "convex_bucket",
    "convex_write_args",
    "convex_sat_small",
    "convex_sat_large",
    "convex_manifold",
    "convex_manifold_persistent",
    // Vehicle controller (slang/kernels/vehicle_step.slang).
    "vehicle_input",
    "wheel_query",
    "vehicle_link",
    "vehicle_prepare",
    "vehicle_iterate",
    "vehicle_loads",
    "vehicle_loads_persistent",
    "vehicle_post",
    "raycast_batch",
    // Characters, in-step queries, projectiles (slang/kernels/character_step.slang).
    "char_move",
    "char_post",
    "query_batch",
    "projectile_step",
    // Soft bodies (slang/kernels/soft_step.slang).
    "soft_wake",
    "soft_begin",
    "soft_bounds",
    "soft_tri_planes",
    "soft_pairs",
    "soft_velocity",
    "soft_contact_lists",
    "soft_collide_nodes",
    "soft_collide_tris",
    "soft_primal",
    "soft_primal_fused",
    "soft_primal_bodies",
    "soft_chebyshev",
    "soft_mode",
    "soft_dual",
    "soft_plastic",
    "soft_break_groups",
    "soft_sleep",
];

const CHARACTER_ENTRIES: &[&str] = &["char_move", "char_post", "query_batch", "projectile_step"];

const VEHICLE_ENTRIES: &[&str] = &[
    "vehicle_input",
    "wheel_query",
    "vehicle_link",
    "vehicle_prepare",
    "vehicle_iterate",
    "vehicle_loads",
    "vehicle_post",
    "raycast_batch",
];

/// Slang source that defines `entry`.
fn source_for(root: &std::path::Path, entry: &str) -> PathBuf {
    let kernels = root.join("kernels");
    if entry == "vehicle_loads_persistent" {
        return kernels.join("vehicle_step_persistent.slang");
    }
    if VEHICLE_ENTRIES.contains(&entry) {
        return kernels.join("vehicle_step.slang");
    }
    if CHARACTER_ENTRIES.contains(&entry) {
        return kernels.join("character_step.slang");
    }
    if entry.starts_with("soft_") {
        return kernels.join("soft_step.slang");
    }
    if entry.starts_with("convex_") {
        return kernels.join(if entry.ends_with("_persistent") {
            "convex_persistent.slang"
        } else {
            "convex_step.slang"
        });
    }
    if entry.ends_with("_copy_persistent") {
        kernels.join("reuse_copy.slang")
    } else if entry.ends_with("_persistent") {
        kernels.join("persistent_solve.slang")
    } else if entry.ends_with("_compact")
        || entry == "pack_solver_contacts"
        || entry == "clear_solver_contacts"
    {
        kernels.join("compact_solve.slang")
    } else {
        kernels.join("world_step.slang")
    }
}

/// What a compile leaves besides the files in OUT_DIR: the `cargo:warning` lines and
/// the entries whose WGSL compiled (for `direct_wgsl.rs`, which names OUT_DIR paths).
struct Compiled {
    warnings: Vec<String>,
    wgsl_ok: Vec<String>,
}

fn main() -> Result<()> {
    engine_slang_build::report_slangc_for_cargo();
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let slang_root = manifest.join("../../slang");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    println!("cargo:rerun-if-changed={}", slang_root.display());
    println!("cargo:rerun-if-env-changed=RUBBLE_SLANG_CACHE");
    let _ = find_slangc()?;

    // `RUBBLE_SLANG_CACHE`: a directory keeping these outputs between builds, keyed by
    // everything they depend on (see `cache_key`). The SPIR-V and WGSL do not depend on
    // the cargo target, so the host and wasm32 builds share an entry.
    let cache = match env::var_os("RUBBLE_SLANG_CACHE") {
        Some(dir) => Some(PathBuf::from(dir).join(cache_key(&manifest, &slang_root)?)),
        None => None,
    };
    let compiled = match cache.as_deref().filter(|d| d.join("wgsl_ok.txt").is_file()) {
        Some(hit) => {
            for sub in ["spirv", "wgsl"] {
                copy_dir(&hit.join(sub), &out_dir.join(sub))?;
            }
            std::fs::copy(
                hit.join("spirv_entries.txt"),
                out_dir.join("spirv_entries.txt"),
            )?;
            Compiled {
                warnings: read_records(&hit.join("warnings.txt"))?,
                wgsl_ok: read_records(&hit.join("wgsl_ok.txt"))?,
            }
        }
        None => {
            let compiled = compile_all(&slang_root, &out_dir)?;
            if let Some(dir) = &cache {
                store(dir, &out_dir, &compiled)?;
            }
            compiled
        }
    };
    for w in &compiled.warnings {
        println!("cargo:warning={w}");
    }
    let wgsl_dir = out_dir.join("wgsl");
    let mut table = String::from("pub static DIRECT_WGSL: &[(&str, &str)] = &[\n");
    for entry in &compiled.wgsl_ok {
        table.push_str(&format!(
            "    ({entry:?}, include_str!({:?})),\n",
            wgsl_dir.join(format!("{entry}.wgsl")).display().to_string()
        ));
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("direct_wgsl.rs"), table)?;
    Ok(())
}

/// Every slangc compile of this crate into `out_dir` (`spirv/`, `spirv_entries.txt`,
/// `wgsl/`).
fn compile_all(slang_root: &Path, out_dir: &Path) -> Result<Compiled> {
    let out = out_dir.join("spirv");
    std::fs::create_dir_all(&out)?;
    let mut warnings = Vec::new();
    let abi = slang_root.join("kernels/abi_probe.slang");
    compile(
        &abi,
        "abi_roundtrip",
        SlangTarget::Spirv,
        &out.join("abi_roundtrip.spv"),
        &[],
    )?;

    let mut ok = vec!["abi_roundtrip".to_string()];
    let mut failed = Vec::new();
    let spirv = par_map(WORLD_ENTRIES, |entry| {
        let dest = out.join(format!("{entry}.spv"));
        compile(
            &source_for(slang_root, entry),
            entry,
            SlangTarget::Spirv,
            &dest,
            &[],
        )
    });
    for (entry, result) in WORLD_ENTRIES.iter().zip(spirv) {
        match result {
            Ok(()) => ok.push((*entry).to_string()),
            Err(e) => {
                warnings.push(format!("SPIR-V skip {entry}: {e}"));
                failed.push((*entry).to_string());
            }
        }
    }
    // Alias for phase-graph name used by Metal. Not conditional on the alias being
    // absent: a reused OUT_DIR already holds it, and the entry list must not change.
    let query = out.join("broadphase_query.spv");
    let grid = out.join("broadphase_grid.spv");
    if query.is_file() {
        std::fs::copy(&query, &grid)?;
        ok.push("broadphase_grid".to_string());
    }

    std::fs::write(out_dir.join("spirv_entries.txt"), ok.join("\n") + "\n")?;
    if !failed.is_empty() {
        warnings.push(format!(
            "SPIR-V failed for {} entries: {}",
            failed.len(),
            failed.join(", ")
        ));
    }

    // Direct Slang → WGSL. `ENGINE_WGSL` switches atomic slots to Atomic<uint>
    // (physics/atomics.slang); the WGSL is embedded so wasm32 needs no filesystem.
    let wgsl_dir = out_dir.join("wgsl");
    std::fs::create_dir_all(&wgsl_dir)?;
    let mut wgsl_ok = Vec::new();
    let mut wgsl_failed = Vec::new();
    let wgsl_entries: Vec<&str> = WORLD_ENTRIES
        .iter()
        .copied()
        .chain(std::iter::once("broadphase_grid"))
        .collect();
    let wgsl = par_map(&wgsl_entries, |entry| {
        compile_with_defines(
            &source_for(slang_root, entry),
            entry,
            SlangTarget::Wgsl,
            &wgsl_dir.join(format!("{entry}.wgsl")),
            &[],
            &[("ENGINE_WGSL", "1")],
        )
    });
    for (entry, result) in wgsl_entries.iter().zip(wgsl) {
        match result {
            Ok(()) => wgsl_ok.push((*entry).to_string()),
            Err(e) => {
                warnings.push(format!("WGSL skip {entry}: {e}"));
                wgsl_failed.push(*entry);
            }
        }
    }
    // Soft-body skinning: one standalone WGSL module, three entries.
    let skin = wgsl_dir.join("soft_skin.wgsl");
    let status = std::process::Command::new(find_slangc()?)
        .args(["-target", "wgsl"])
        .args(["-entry", "soft_skin_vertices", "-stage", "compute"])
        .args(["-entry", "soft_skin_faces", "-stage", "compute"])
        .args(["-entry", "soft_skin_normals", "-stage", "compute"])
        .arg("-o")
        .arg(&skin)
        .arg(slang_root.join("kernels/soft_skin.slang"))
        .status()?;
    anyhow::ensure!(status.success(), "slangc failed for soft_skin.slang");
    if !wgsl_failed.is_empty() {
        warnings.push(format!("direct WGSL failed: {}", wgsl_failed.join(", ")));
    }
    Ok(Compiled { warnings, wgsl_ok })
}

/// A hash of everything the outputs depend on: slangc's version, every file under
/// `slang/` (path and bytes), this script and engine-slang-build (which pass the
/// arguments). Equal keys mean slangc gets identical inputs, so a hit is the same bytes.
fn cache_key(manifest: &Path, slang_root: &Path) -> Result<String> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let version = std::process::Command::new(find_slangc()?)
        .arg("-version")
        .output()?;
    (version.stdout, version.stderr).hash(&mut h);
    let mut files = Vec::new();
    list_files(slang_root, &mut files)?;
    files.sort();
    for f in &files {
        (f.strip_prefix(slang_root)?, std::fs::read(f)?).hash(&mut h);
    }
    for f in ["build.rs", "../engine-slang-build/src/lib.rs"] {
        std::fs::read(manifest.join(f))?.hash(&mut h);
    }
    Ok(format!("engine-webgpu-{:016x}", h.finish()))
}

fn list_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            list_files(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        std::fs::copy(entry.path(), to.join(entry.file_name()))?;
    }
    Ok(())
}

/// Records are separated by U+001E, since a warning can span lines.
fn read_records(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .split('\u{1e}')
        .filter(|r| !r.is_empty())
        .map(String::from)
        .collect())
}

/// Write the entry under a temporary name and rename it, so a build that stops
/// half way never leaves an entry that looks complete.
fn store(dir: &Path, out_dir: &Path, compiled: &Compiled) -> Result<()> {
    let tmp = dir.with_extension(format!("tmp{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    for sub in ["spirv", "wgsl"] {
        copy_dir(&out_dir.join(sub), &tmp.join(sub))?;
    }
    std::fs::copy(
        out_dir.join("spirv_entries.txt"),
        tmp.join("spirv_entries.txt"),
    )?;
    std::fs::write(tmp.join("warnings.txt"), compiled.warnings.join("\u{1e}"))?;
    std::fs::write(tmp.join("wgsl_ok.txt"), compiled.wgsl_ok.join("\u{1e}"))?;
    if std::fs::rename(&tmp, dir).is_err() {
        // Another build stored the same key first.
        std::fs::remove_dir_all(&tmp)?;
    }
    // Keep only this key: entries for older slang/ trees would only grow the cache.
    if let (Some(parent), Some(name)) = (dir.parent(), dir.file_name()) {
        for entry in std::fs::read_dir(parent)? {
            let entry = entry?;
            let other = entry.file_name();
            if other != name && other.to_string_lossy().starts_with("engine-webgpu-") {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
    Ok(())
}
