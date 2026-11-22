//! Scan Slang-emitted CUDA files for the deferred-load miscompile
//! (`cuda_layout::deferred_load_hazards`) before a module joins a CUDA build.
//!
//!     cargo run -p engine-slang-build --example cuda_hazards -- FILE.cu ...
use engine_slang_build::cuda_layout::deferred_load_hazards;

fn main() {
    let mut total = 0;
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let h = deferred_load_hazards(&text);
        let entries = text.matches("__global__").count();
        println!("{path}: {entries} entries, {} hazards", h.len());
        for line in &h {
            println!("    {line}");
        }
        total += h.len();
    }
    std::process::exit(if total == 0 { 0 } else { 1 });
}
