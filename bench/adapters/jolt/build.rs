//! Builds Jolt Physics (bench/third_party/JoltPhysics, pinned by third_party/fetch.sh) as a static
//! library with CMake, then compiles `src/jolt_shim.cpp` with exactly the preprocessor defines and
//! code-generation flags CMake used for the library (read back from `compile_commands.json`), so
//! the shim and the library agree on every `JPH_*` feature switch (Jolt also checks this at
//! runtime: `VerifyJoltVersionID()` in the shim).
//!
//! Env knobs: JOLT_DIR (Jolt source checkout; default ../../third_party/JoltPhysics),
//! JOLT_CMAKE (cmake executable), JOLT_BUILD_JOBS (parallel compile jobs; default cargo's NUM_JOBS).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// CMake options (Build/CMakeLists.txt). Everything a CPU benchmark does not need is off; GPU
/// compute back-ends (Metal/Vulkan/DX12/CPU-compute, new in 5.5+) are off.
const OPTIONS: &[(&str, &str)] = &[
    ("CMAKE_BUILD_TYPE", "Distribution"),
    ("TARGET_UNIT_TESTS", "OFF"),
    ("TARGET_HELLO_WORLD", "OFF"),
    ("TARGET_PERFORMANCE_TEST", "OFF"),
    ("TARGET_SAMPLES", "OFF"),
    ("TARGET_VIEWER", "OFF"),
    ("INTERPROCEDURAL_OPTIMIZATION", "ON"),
    ("DOUBLE_PRECISION", "OFF"),
    ("OBJECT_LAYER_BITS", "16"),
    ("CROSS_PLATFORM_DETERMINISTIC", "OFF"),
    ("USE_ASSERTS", "OFF"),
    ("DEBUG_RENDERER_IN_DISTRIBUTION", "OFF"),
    ("DEBUG_RENDERER_IN_DEBUG_AND_RELEASE", "OFF"),
    ("PROFILER_IN_DISTRIBUTION", "OFF"),
    ("PROFILER_IN_DEBUG_AND_RELEASE", "OFF"),
    ("GENERATE_DEBUG_SYMBOLS", "OFF"),
    ("ENABLE_ALL_WARNINGS", "OFF"),
    ("ENABLE_INSTALL", "OFF"),
    ("JPH_USE_DX12", "OFF"),
    ("JPH_USE_VK", "OFF"),
    ("JPH_USE_MTL", "OFF"),
    ("JPH_USE_CPU_COMPUTE", "OFF"),
    ("JPH_BUILD_SHARED_LIBS", "OFF"),
    ("CMAKE_EXPORT_COMPILE_COMMANDS", "ON"),
];

fn main() {
    for v in ["JOLT_DIR", "JOLT_CMAKE", "JOLT_BUILD_JOBS"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    println!("cargo:rerun-if-changed=src/jolt_shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = env::var_os("JOLT_DIR").map(PathBuf::from).unwrap_or_else(|| manifest.join("../../third_party/JoltPhysics"));
    let src = src.canonicalize().unwrap_or_else(|_| {
        panic!("no Jolt source at {} (run bench/third_party/fetch.sh, or set JOLT_DIR)", src.display())
    });
    assert!(src.join("Build/CMakeLists.txt").exists(), "{}: no Build/CMakeLists.txt", src.display());
    println!("cargo:rerun-if-changed={}", src.join("Jolt").display());

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build = out.join("jolt-build");
    std::fs::create_dir_all(&build).unwrap();
    let cmake = env::var("JOLT_CMAKE").unwrap_or_else(|_| "cmake".into());

    let mut cfg = Command::new(&cmake);
    cfg.arg("-S").arg(src.join("Build")).arg("-B").arg(&build);
    for (k, v) in OPTIONS {
        cfg.arg(format!("-D{k}={v}"));
    }
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let arch = if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") { "arm64" } else { "x86_64" };
        cfg.arg(format!("-DCMAKE_OSX_ARCHITECTURES={arch}"));
    }
    run(&mut cfg, "cmake configure");

    let jobs = env::var("JOLT_BUILD_JOBS").or_else(|_| env::var("NUM_JOBS")).unwrap_or_else(|_| "4".into());
    let mut b = Command::new(&cmake);
    b.arg("--build").arg(&build).arg("--target").arg("Jolt").arg("--config").arg("Distribution").arg("-j").arg(&jobs);
    run(&mut b, "cmake --build");

    let lib = find_file(&build, "libJolt.a").or_else(|| find_file(&build, "Jolt.lib")).expect("libJolt.a not found after the CMake build");
    println!("cargo:rustc-link-search=native={}", lib.parent().unwrap().display());
    println!("cargo:rustc-link-lib=static=Jolt");

    // The library's own compile line for one of its sources: copy its -D, -std and codegen flags
    let cc_json = std::fs::read_to_string(build.join("compile_commands.json")).expect("compile_commands.json");
    let cmds: serde_json::Value = serde_json::from_str(&cc_json).unwrap();
    let cmd = cmds
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["file"].as_str().map(|f| f.ends_with("PhysicsSystem.cpp")).unwrap_or(false))
        .expect("PhysicsSystem.cpp missing from compile_commands.json");
    let line = cmd["command"].as_str().map(|s| s.to_string()).unwrap_or_else(|| {
        cmd["arguments"].as_array().unwrap().iter().map(|a| a.as_str().unwrap().to_string()).collect::<Vec<_>>().join(" ")
    });
    let mut defines = Vec::new();
    let mut flags = Vec::new();
    for tok in line.split_whitespace() {
        if let Some(d) = tok.strip_prefix("-D") {
            defines.push(d.to_string());
        } else if tok.starts_with("-std=")
            || tok.starts_with("-ffp-contract")
            || tok == "-fno-rtti"
            || tok == "-frtti"
            || tok == "-fno-exceptions"
            || tok == "-fexceptions"
            || tok == "-faligned-allocation"
            || tok.starts_with("-m")
        {
            flags.push(tok.to_string());
        }
    }
    let mut c = cc::Build::new();
    c.cpp(true).file("src/jolt_shim.cpp").include(&src).opt_level(3).debug(false);
    for d in &defines {
        match d.split_once('=') {
            Some((k, v)) => c.define(k, v),
            None => c.define(d, None),
        };
    }
    for f in &flags {
        c.flag(f);
    }
    if !flags.iter().any(|f| f.starts_with("-std=")) {
        c.flag("-std=c++17");
    }
    c.compile("bench_jolt_shim");

    let rev = Command::new("git").arg("-C").arg(&src).args(["rev-parse", "HEAD"]).output().ok().filter(|o| o.status.success());
    let rev = rev.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=BENCH_JOLT_REV={rev}");
    println!("cargo:rustc-env=BENCH_JOLT_DEFINES={}", defines.join(" "));
    println!("cargo:rustc-env=BENCH_JOLT_FLAGS={}", flags.join(" "));
    println!(
        "cargo:rustc-env=BENCH_JOLT_CMAKE={}",
        OPTIONS.iter().filter(|(k, _)| *k != "CMAKE_EXPORT_COMPILE_COMMANDS").map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
    );
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd.status().unwrap_or_else(|e| panic!("{what}: {e} (is cmake installed?)"));
    assert!(status.success(), "{what} failed: {status}");
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(f) = find_file(&p, name) {
                return Some(f);
            }
        } else if p.file_name().map(|n| n == name).unwrap_or(false) {
            return Some(p);
        }
    }
    None
}
