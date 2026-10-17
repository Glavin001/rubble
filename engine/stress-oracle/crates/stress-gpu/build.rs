//! Compiles the checked-in Metal Shading Language (generated from Slang by
//! `scripts/compile_shaders.sh`) into `.metallib` libraries with Xcode's `metal`, so
//! shader errors surface at build time and nothing is compiled at run time. Without
//! Xcode (or off macOS) the libraries are left empty and the solver compiles the MSL
//! source at load instead.
//!
//! Each shader is built twice: the parity library without fast math (IEEE division and
//! square root, no reassociation), and `<name>.fast.metallib` with it, for production
//! (`STRESS_GPU_MATH=fast`). The shaders test NaN by its bits, so fast math (which
//! assumes no NaN) keeps their sentinels.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let generated = Path::new("shaders/generated");
    println!("cargo:rerun-if-changed=shaders/generated");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    let mut entries: Vec<_> = std::fs::read_dir(generated).expect("shaders/generated").flatten().map(|e| e.path()).collect();
    entries.sort();
    for metal in entries.iter().filter(|p| p.extension().is_some_and(|e| e == "metal")) {
        let name = metal.file_stem().unwrap().to_string_lossy().to_string();
        println!("cargo:rerun-if-changed={}", metal.display());
        for (suffix, fast) in [("", false), (".fast", true)] {
            let lib = out.join(format!("{name}{suffix}.metallib"));
            let built = macos && compile(metal, &out.join(format!("{name}{suffix}.air")), &lib, fast);
            if !built {
                std::fs::write(&lib, []).expect("write placeholder metallib");
            }
        }
    }
}

fn compile(metal: &Path, air: &Path, lib: &Path, fast: bool) -> bool {
    let run = |args: &[&std::ffi::OsStr]| match Command::new("xcrun").args(args).status() {
        Ok(s) if s.success() => true,
        Ok(s) => {
            println!("cargo:warning=xcrun {:?} failed ({s}); MSL will be compiled at load", args);
            false
        }
        Err(e) => {
            println!("cargo:warning=xcrun unavailable ({e}); MSL will be compiled at load");
            false
        }
    };
    let math = if fast { "-ffast-math" } else { "-fno-fast-math" };
    run(&["-sdk".as_ref(), "macosx".as_ref(), "metal".as_ref(), math.as_ref(), "-c".as_ref(), metal.as_os_str(), "-o".as_ref(), air.as_os_str()])
        && run(&["-sdk".as_ref(), "macosx".as_ref(), "metallib".as_ref(), air.as_os_str(), "-o".as_ref(), lib.as_os_str()])
}
