//! The kernels of this build ready to run, and where they are: with the kernels compiled
//! at run time (the fast profile on native) this compiles the store for the current Slang
//! sources (what the first run after a kernel edit would do), so a timed or GPU-locked run
//! does not; with them embedded it prints the build's WGSL directory.
//!
//! `--copy-to DIR` (kernels at run time only): the store's WGSL and its lists into DIR,
//! replacing what was there: the fast web build's `generated/kernels/`, which its world
//! wasm reads (engine-world-wasm `kernel_file`). A kernel edit for the pages is then this
//! command and a reload:
//!   cargo run --profile fast -p engine-webgpu --bin kernels -- --copy-to crates/engine-webgpu/generated/kernels
fn main() {
    let wgsl = engine_webgpu::explicit_beams_wgsl_dir();
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--copy-to") {
        let to = std::path::PathBuf::from(args.get(i + 1).expect("--copy-to DIR"));
        let store = wgsl.parent().expect("the store");
        if !store.join("wgsl_ok.txt").is_file() {
            eprintln!("--copy-to: this build embeds its kernels (only the fast profile on native compiles them at run time)");
            std::process::exit(2);
        }
        let _ = std::fs::remove_dir_all(&to);
        std::fs::create_dir_all(to.join("wgsl")).unwrap();
        for e in std::fs::read_dir(&wgsl).unwrap() {
            let p = e.unwrap().path();
            std::fs::copy(&p, to.join("wgsl").join(p.file_name().unwrap())).unwrap();
        }
        for f in ["wgsl_ok.txt", "spirv_entries.txt"] {
            std::fs::copy(store.join(f), to.join(f)).unwrap();
        }
        println!("{}", to.display());
        return;
    }
    println!("{}", wgsl.display());
}
