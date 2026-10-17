//! Builds Box3D (bench/third_party/box3d, pinned by third_party/fetch.sh) with its own CMake
//! in Release with heavy validation, profiling, samples, tests and benchmarks off, then the shim.
use std::path::PathBuf;

fn main() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../third_party/box3d");
    assert!(src.join("CMakeLists.txt").exists(), "Box3D sources missing: run bench/third_party/fetch.sh");
    println!("cargo:rerun-if-changed=src/box3d_shim.c");
    println!("cargo:rerun-if-changed={}", src.join("src").display());
    let dst = cmake::Config::new(&src)
        .profile("Release")
        .define("BOX3D_SAMPLES", "OFF")
        .define("BOX3D_UNIT_TESTS", "OFF")
        .define("BOX3D_BENCHMARKS", "OFF")
        .define("BOX3D_DOCS", "OFF")
        .define("BOX3D_VALIDATE", "OFF")
        .define("BOX3D_PROFILE", "OFF")
        .build_target("box3d")
        .build();
    let lib = dst.join("build/src");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=box3d");
    let rev = std::process::Command::new("git").args(["-C", src.to_str().unwrap(), "rev-parse", "--short=12", "HEAD"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    println!("cargo:rustc-env=BENCH_BOX3D_REV={rev}");
    cc::Build::new().file("src/box3d_shim.c").include(src.join("include")).flag("-std=c17").opt_level(3).compile("bench_box3d_shim");
}
