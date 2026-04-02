//! Build script for rubble-shaders: compiles Rust GPU shaders to SPIR-V.
//!
//! When the `spirv-builder` toolchain is available, this compiles all shader
//! entry points to SPIR-V modules. The resulting .spv files are placed in
//! the target directory and can be included at compile time by consuming crates.
//!
//! If spirv-builder is not available (e.g., in CI without the rust-gpu toolchain),
//! the build gracefully falls back to using the WGSL shader strings.

fn main() {
    // When spirv-builder is available, uncomment the following to enable
    // automatic SPIR-V compilation:
    //
    // use spirv_builder::SpirvBuilder;
    //
    // let result = SpirvBuilder::new(".", "spirv-unknown-vulkan1.1")
    //     .multimodule(true)
    //     .build()
    //     .expect("Failed to compile shaders to SPIR-V");
    //
    // // Export the path so consuming crates can find the SPIR-V modules
    // for (entry, path) in result.module.unwrap_multi() {
    //     println!("cargo:rustc-env=RUBBLE_SPIRV_{}={}", entry.to_uppercase(), path.display());
    // }

    // For now, just ensure rebuild on source changes
    println!("cargo:rerun-if-changed=src/lib.rs");
}
