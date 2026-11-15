//! Validate WGSL files with naga (parse and validate, every capability):
//! a CPU check for shaders that cannot be run here. Chrome (Tint) is stricter
//! about uniformity; run `explicit-wgsl-check.mjs`-style checks on a GPU too.
//!
//! `cargo run -p engine-webgpu --release --example wgsl_validate -- a.wgsl b.wgsl`
//! (the showcase skin: `node showcase-wgsl-dump.mjs out/` writes its modules).
fn main() {
    let mut bad = 0;
    for path in std::env::args().skip(1) {
        let src = std::fs::read_to_string(&path).expect("read");
        match naga::front::wgsl::parse_str(&src) {
            Err(e) => {
                bad += 1;
                eprintln!("{path}: parse error\n{}", e.emit_to_string(&src));
            }
            Ok(m) => {
                let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
                match v.validate(&m) {
                    Err(e) => {
                        bad += 1;
                        eprintln!("{path}: invalid\n{}", e.emit_to_string(&src));
                    }
                    Ok(_) => println!("{path}: ok ({} entry points)", m.entry_points.len()),
                }
            }
        }
    }
    std::process::exit(bad);
}
