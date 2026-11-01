//! Native devices run without wgpu's indirect-call validation (`instance_flags`): the
//! indirect arguments in binding 23 (`gIndirectArgs`) must always be within the device's
//! workgroup limits. They are while every write goes through `writeDispatchArgs`
//! (world_step.slang: at most 32,768 groups a row) or `writeGroupArgs` (convex_step.slang:
//! rows of at most 32,768). This test scans every Slang source and fails on any other
//! mention of `gIndirectArgs`, or binding 23 declared under another name.

use std::path::{Path, PathBuf};

/// (file name, function, reason) of each direct writer allowed besides the two helpers.
const ALLOWED: &[(&str, &str, &str)] = &[
    ("world_step.slang", "writeDispatchArgs", "the helper: x = min(groups, 32768), y = rows"),
    ("convex_step.slang", "writeGroupArgs", "the helper: rows of at most 32,768 groups"),
    (
        "fracture_step.slang",
        "fracture_write_sat_args",
        "the lab plan's SAT args: rows = ceil(n / 32768), width = ceil(n / rows) <= 32768",
    ),
];

#[test]
fn debug_builds_keep_the_indirect_call_validation() {
    if std::env::var_os("WGPU_VALIDATION_INDIRECT_CALL").is_some() {
        return; // The variable overrides the build's default.
    }
    let on = engine_webgpu::instance_flags().contains(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    assert_eq!(on, cfg!(debug_assertions), "validation on in debug builds only");
}

fn slang_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("slang dir") {
        let p = entry.expect("entry").path();
        if p.is_dir() {
            slang_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "slang") {
            out.push(p);
        }
    }
}

/// The function a line belongs to: the last top-level `name(` definition above it.
fn function_name(line: &str) -> Option<String> {
    if line.starts_with(char::is_whitespace) || line.starts_with("//") || line.starts_with('#') {
        return None;
    }
    let open = line.find('(')?;
    let name = line[..open].split_whitespace().last()?;
    name.chars()
        .all(|c| c.is_alphanumeric() || c == '_')
        .then(|| name.to_string())
}

#[test]
fn indirect_arguments_are_written_only_by_the_bounded_helpers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../slang");
    let mut files = Vec::new();
    slang_files(&root, &mut files);
    assert!(files.len() > 10, "no Slang sources under {}", root.display());
    let mut bad = Vec::new();
    let mut seen = Vec::new();
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(f).expect("read");
        let mut function = String::new();
        for (n, raw) in text.lines().enumerate() {
            if let Some(fname) = function_name(raw) {
                function = fname;
            }
            let code = raw.split("//").next().unwrap_or("");
            if code.contains("BIND(23)") && !code.contains("gIndirectArgs") {
                bad.push(format!("{name}:{}: binding 23 under another name", n + 1));
            }
            if !code.contains("gIndirectArgs") || code.contains("BIND(23)") {
                continue;
            }
            if ALLOWED.iter().any(|(file, func, _)| *file == name && *func == function) {
                seen.push((name.clone(), function.clone()));
            } else {
                bad.push(format!("{name}:{}: gIndirectArgs in {function}: {}", n + 1, code.trim()));
            }
        }
    }
    assert!(bad.is_empty(), "indirect arguments written outside the bounded helpers:\n{}", bad.join("\n"));
    for (file, func, _) in ALLOWED {
        assert!(
            seen.iter().any(|(f, g)| f == file && g == func),
            "{file}: {func} no longer writes gIndirectArgs (drop it from ALLOWED)"
        );
    }
}
