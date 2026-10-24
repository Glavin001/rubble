//! Generate the C header into OUT_DIR; `tests/header.rs` checks the committed
//! `include/rubble.h` matches (bless with `RUBBLE_FFI_BLESS=1`).

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("out dir")).join("rubble.h");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    let config =
        cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml")).expect("cbindgen.toml");
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("generate the C header")
        .write_to_file(&out);
}
