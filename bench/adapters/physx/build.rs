//! Builds `src/physx_shim.cpp` and links PhysX.
//!
//! Two sources of PhysX (see README.md):
//! * default feature `vendored`: PhysX 5.1 compiled from source by the `physx-sys` crate. The
//!   shim is compiled against physx-sys's vendored headers with the same defines physx-sys
//!   uses; physx-sys itself emits the link lines for its static `physx` library.
//! * `PHYSX_SDK_DIR=<NVIDIA-Omniverse/PhysX>/physx` with `--no-default-features`: an official,
//!   already-built PhysX 5.x SDK. Headers from `$PHYSX_SDK_DIR/include`, libraries from
//!   `$PHYSX_LIB_DIR` or the first of `$PHYSX_SDK_DIR/bin/<platform>/{release,profile,checked}`
//!   that contains PhysX. Feature `gpu` (official SDK only) adds CUDA GPU dynamics.
//!
//! * `PHYSX_CUMETAL_DIR=<PhysX>/out/install/macos-cumetal/release` with `--no-default-features
//!   --features gpu`: the macOS PhysX 5.10 package whose CUDA GPU module runs on Apple Metal
//!   through CuMetal (sibling `cuda-metal` repo). Headers from `include/physx`, static libs and
//!   `libcumetal.dylib` from `lib/`, `PX_CUMETAL=1`; PhysX `dlopen`s the GPU module
//!   (`libPhysXGpuActivity_64.dylib`) through the binary's rpath to `lib/`. CPU runs use the
//!   same build with `--set gpu=off`, so CPU and GPU PhysX are the same version.
//!
//! Env knobs: PHYSX_CUMETAL_DIR, PHYSX_SDK_DIR, PHYSX_LIB_DIR, PHYSX_LIBS (comma-separated library names, in
//! link order, overriding the defaults; `static:name` / `dylib:name` force the link kind,
//! otherwise names containing "static" link statically), PHYSX_SYS_DIR (physx-sys source dir, if it cannot be
//! found automatically).

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    for v in ["PHYSX_CUMETAL_DIR", "PHYSX_SDK_DIR", "PHYSX_LIB_DIR", "PHYSX_LIBS", "PHYSX_SYS_DIR"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    println!("cargo:rerun-if-changed=src/physx_shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    let vendored = env::var_os("CARGO_FEATURE_VENDORED").is_some();
    let gpu = env::var_os("CARGO_FEATURE_GPU").is_some();
    let sdk = env::var_os("PHYSX_SDK_DIR").map(PathBuf::from);

    let mut b = cc::Build::new();
    b.cpp(true).file("src/physx_shim.cpp").opt_level(3).debug(false);
    if gpu {
        b.define("BENCH_PHYSX_GPU", None);
    }

    if let Some(dir) = env::var_os("PHYSX_CUMETAL_DIR").map(PathBuf::from) {
        return build_cumetal(b, &dir, vendored, gpu, sdk.is_some());
    }

    match (sdk, vendored) {
        (Some(_), true) => panic!(
            "PHYSX_SDK_DIR is set but the `vendored` feature (physx-sys) is on too; build with \
             `--no-default-features` (plus `--features gpu` for CUDA) to use the official SDK"
        ),
        (None, false) => panic!(
            "bench-physx needs either the default `vendored` feature (PhysX 5.1 via physx-sys) or \
             PHYSX_SDK_DIR pointing at a built NVIDIA-Omniverse/PhysX `physx/` directory"
        ),
        (None, true) => {
            if gpu {
                panic!("feature `gpu` needs the official SDK: set PHYSX_SDK_DIR and use --no-default-features");
            }
            let root = physx_sys_dir();
            let include = root.join("physx/physx/include");
            assert!(include.join("PxPhysicsAPI.h").exists(), "no PhysX headers at {}", include.display());
            // Same preprocessor setup as physx-sys's own build of PhysX (its build.rs).
            b.define("PX_PHYSX_STATIC_LIB", None)
                .define("DISABLE_CUDA_PHYSX", None)
                .define("NDEBUG", "1")
                .define("PX_SUPPORT_PVD", "1");
            add_header_dir(&mut b, &include);
            b.flag_if_supported("-std=c++14").flag_if_supported("/std:c++14").flag_if_supported("-fno-strict-aliasing");
            let version = root
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "physx-sys".into());
            println!("cargo:rustc-env=BENCH_PHYSX_SOURCE={version} (vendored, CPU only)");
            // physx-sys emits the PhysX archive and the C++ runtime; nothing more to link.
            b.compile("bench_physx_shim");
        }
        (Some(sdk), false) => {
            let include = sdk.join("include");
            assert!(
                include.join("PxPhysicsAPI.h").exists(),
                "PHYSX_SDK_DIR={}: no include/PxPhysicsAPI.h (point it at the `physx/` directory of NVIDIA-Omniverse/PhysX)",
                sdk.display()
            );
            let lib_dir = match env::var_os("PHYSX_LIB_DIR") {
                Some(d) => PathBuf::from(d),
                None => find_lib_dir(&sdk).unwrap_or_else(|| {
                    panic!(
                        "no built PhysX libraries under {}/bin/*/{{release,profile,checked}}; build the SDK or set PHYSX_LIB_DIR",
                        sdk.display()
                    )
                }),
            };
            let config = lib_dir.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            b.define("NDEBUG", "1");
            if config == "checked" {
                b.define("PX_CHECKED", "1");
            } else if config == "profile" {
                b.define("PX_PROFILE", "1");
            }
            add_header_dir(&mut b, &include);
            b.flag_if_supported("-std=c++17").flag_if_supported("/std:c++17").flag_if_supported("-fno-strict-aliasing");

            let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
            let windows = target_os == "windows";
            let libs: Vec<(String, bool)> = match env::var("PHYSX_LIBS") {
                Ok(list) => list
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .map(|name| match name.split_once(':') {
                        Some(("static", n)) => (n.to_string(), true),
                        Some(("dylib", n)) => (n.to_string(), false),
                        _ => {
                            let st = name.contains("static");
                            (name, st)
                        }
                    })
                    .collect(),
                // Link order matters for GNU ld: dependents first.
                // PhysXCooking: convex hulls of format-v2 scenes (PxCreateConvexMesh).
                Err(_) => ["PhysXExtensions", "PhysX", "PhysXCooking", "PhysXPvdSDK", "PhysXCommon", "PhysXFoundation"]
                    .iter()
                    .map(|base| resolve_lib(&lib_dir, base, windows))
                    .collect(),
            };
            // PX_PHYSX_STATIC_LIB (no dllimport) when the core PhysX library links statically;
            // for a custom PHYSX_LIBS list without a recognisable core, when every lib is static.
            let core_static = match libs.iter().find(|(n, _)| n == "PhysX" || n.starts_with("PhysX_")) {
                Some((_, st)) => *st,
                None => libs.iter().all(|(_, st)| *st),
            };
            if core_static {
                b.define("PX_PHYSX_STATIC_LIB", None);
            }
            b.cpp_link_stdlib(None);
            b.compile("bench_physx_shim");

            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            for (name, st) in &libs {
                println!("cargo:rustc-link-lib={}={name}", if *st { "static" } else { "dylib" });
            }
            match target_os.as_str() {
                "linux" | "android" => {
                    println!("cargo:rustc-link-lib=dylib=stdc++");
                    for l in ["dl", "pthread", "rt", "m"] {
                        println!("cargo:rustc-link-lib=dylib={l}");
                    }
                    // PhysX dlopen()s libPhysXGpu_64.so by bare name (and shared PhysX builds
                    // need their .so files): search next to the binary, then the SDK bin dir.
                    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN:{}", lib_dir.display());
                }
                "macos" => println!("cargo:rustc-link-lib=dylib=c++"),
                _ => {}
            }
            println!(
                "cargo:rustc-env=BENCH_PHYSX_SOURCE=official SDK {} ({}{})",
                sdk.display(),
                config,
                if core_static { ", static" } else { ", shared" }
            );
        }
    }
}

/// `-isystem` on gcc/clang so PhysX's own header warnings stay quiet; `-I` elsewhere.
fn add_header_dir(b: &mut cc::Build, dir: &Path) {
    let c = b.get_compiler();
    if c.is_like_gnu() || c.is_like_clang() {
        b.flag(format!("-isystem{}", dir.display()));
    } else {
        b.include(dir);
    }
}

/// `<base>_static_64` if present, else `<base>_64` (DLL import lib / shared object).
fn resolve_lib(dir: &Path, base: &str, windows: bool) -> (String, bool) {
    let file = |name: &str, st: bool| {
        if windows {
            dir.join(format!("{name}.lib"))
        } else if st {
            dir.join(format!("lib{name}.a"))
        } else {
            dir.join(format!("lib{name}.so"))
        }
    };
    for (name, st) in [(format!("{base}_static_64"), true), (format!("{base}_64"), false), (format!("{base}_static"), true), (base.to_string(), false)] {
        if file(&name, st).exists() {
            return (name, st);
        }
    }
    // Extensions/PvdSDK only ship static; fall back to the conventional static name.
    (format!("{base}_static_64"), true)
}

fn find_lib_dir(sdk: &Path) -> Option<PathBuf> {
    let bin = sdk.join("bin");
    let mut platforms: Vec<PathBuf> = std::fs::read_dir(&bin).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
    platforms.sort();
    for config in ["release", "profile", "checked"] {
        for p in &platforms {
            let d = p.join(config);
            let has = ["libPhysX_static_64.a", "libPhysX_64.so", "PhysX_static_64.lib", "PhysX_64.lib"].iter().any(|f| d.join(f).exists());
            if has {
                return Some(d);
            }
        }
    }
    None
}

/// physx-sys has no `links` key (so no DEP_* variables); find its source via `cargo metadata`,
/// falling back to the registry under CARGO_HOME.
fn physx_sys_dir() -> PathBuf {
    if let Some(d) = env::var_os("PHYSX_SYS_DIR") {
        return PathBuf::from(d);
    }
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = std::process::Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--offline", "--manifest-path"])
        .arg(Path::new(&manifest_dir).join("Cargo.toml"))
        .output();
    if let Ok(out) = out {
        if out.status.success() {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&out.stdout) {
                if let Some(pkgs) = v["packages"].as_array() {
                    for p in pkgs {
                        if p["name"] == "physx-sys" {
                            if let Some(m) = p["manifest_path"].as_str() {
                                let dir = Path::new(m).parent().unwrap().to_path_buf();
                                if dir.join("physx/physx/include").exists() {
                                    return dir;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Fallback: newest physx-sys-* in the registry source cache.
    let home = env::var_os("CARGO_HOME").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")).expect("no HOME")).join(".cargo")
    });
    let mut found: Vec<PathBuf> = Vec::new();
    if let Ok(regs) = std::fs::read_dir(home.join("registry/src")) {
        for r in regs.flatten() {
            if let Ok(crates) = std::fs::read_dir(r.path()) {
                for c in crates.flatten() {
                    let n = c.file_name().to_string_lossy().to_string();
                    if n.starts_with("physx-sys-") && c.path().join("physx/physx/include").exists() {
                        found.push(c.path());
                    }
                }
            }
        }
    }
    found.sort();
    found.pop().expect("could not locate the physx-sys sources; set PHYSX_SYS_DIR")
}

/// The CuMetal PhysX package (`PhysXDestruction::NativeScene` in its
/// lib/cmake/PhysXDestruction/PhysXDestructionConfig.cmake links exactly this).
fn build_cumetal(mut b: cc::Build, dir: &Path, vendored: bool, gpu: bool, sdk: bool) {
    assert!(!vendored && !sdk, "PHYSX_CUMETAL_DIR excludes the `vendored` feature and PHYSX_SDK_DIR; build with --no-default-features --features gpu");
    assert!(gpu, "PHYSX_CUMETAL_DIR needs --features gpu (one build serves gpu=on and gpu=off)");
    let include = dir.join("include/physx");
    let lib = dir.join("lib");
    assert!(include.join("PxPhysicsAPI.h").exists(), "PHYSX_CUMETAL_DIR={}: no include/physx/PxPhysicsAPI.h", dir.display());
    for f in ["libcumetal.dylib", "libPhysXGpuActivity_64.dylib", "libPhysX_static_64.a"] {
        assert!(lib.join(f).exists(), "PHYSX_CUMETAL_DIR={}: no lib/{f}", dir.display());
    }
    // PxPreprocessor.h exposes the GPU API on macOS only under PX_CUMETAL
    b.define("PX_PHYSX_STATIC_LIB", None).define("NDEBUG", "1").define("PX_CUMETAL", "1");
    add_header_dir(&mut b, &include);
    b.flag("-std=c++17").flag("-fno-strict-aliasing");
    b.cpp_link_stdlib(None);
    b.compile("bench_physx_shim");
    println!("cargo:rustc-link-search=native={}", lib.display());
    for name in ["PhysXExtensions", "PhysXCudaContextManager", "PhysX", "PhysXPvdSDK", "PhysXCooking", "PhysXCommon", "PhysXFoundation"] {
        println!("cargo:rustc-link-lib=static={name}_static_64");
    }
    println!("cargo:rustc-link-lib=dylib=cumetal");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", lib.display());
    let rev = std::fs::read_to_string(dir.join("sdk-artifacts.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .map(|v| format!("{} {}", v["feature_profile"].as_str().unwrap_or("?"), &v["source_revision"].as_str().unwrap_or("?")[..12.min(v["source_revision"].as_str().unwrap_or("?").len())]))
        .unwrap_or_default();
    println!("cargo:rustc-env=BENCH_PHYSX_SOURCE=CuMetal package {} ({rev})", dir.display());
    println!("cargo:rerun-if-changed={}", lib.join("libPhysX_static_64.a").display());
}
