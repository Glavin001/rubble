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
    // `--closures FILE`: every kernel compiled into a scratch directory with each compile's
    // include closure logged to FILE (RUBBLE_SLANG_CLOSURE_LOG), for
    // tools/build-time/fanout.py. Any build, any profile; cached compiles are not rerun.
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--closures") {
        let log = std::path::PathBuf::from(args.get(i + 1).expect("--closures FILE"));
        let _ = std::fs::remove_file(&log);
        std::env::set_var("RUBBLE_SLANG_CLOSURE_LOG", &log);
        let out = std::env::temp_dir().join(format!("rubble-closures-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        let r = engine_webgpu::compile_kernels_to(&out);
        let _ = std::fs::remove_dir_all(&out);
        if let Err(e) = r {
            eprintln!("kernels --closures: {e}");
            std::process::exit(1);
        }
        println!("{}", log.display());
        return;
    }
    // `--browser-wgsl DIR`: every module of this build as a host hands it to the browser
    // (the World's shaders through its own report: barriers, loop caps, uniform loads;
    // every other module through uniformize_workgroup_lanes), and NAME.fast.wgsl with the
    // fast setting's arithmetic: for explicit-wgsl-check.mjs (EXPLICIT_WGSL_MODULES_ONLY),
    // Chrome's own WGSL validation in CI.
    if let Some(i) = args.iter().position(|a| a == "--browser-wgsl") {
        let to = std::path::PathBuf::from(args.get(i + 1).expect("--browser-wgsl DIR"));
        let _ = std::fs::remove_dir_all(&to);
        std::fs::create_dir_all(&to).unwrap();
        let world: std::collections::HashMap<String, String> = engine_webgpu::WebGpuWorld::compile_shaders_report()
            .into_iter()
            .filter_map(|(n, r)| r.ok().map(|w| (n, w)))
            .collect();
        let dir = engine_webgpu::explicit_beams_wgsl_dir();
        let mut n = 0;
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            let Some(name) = p.file_name().and_then(|f| f.to_str()).and_then(|f| f.strip_suffix(".wgsl")) else { continue };
            let raw = std::fs::read_to_string(&p).unwrap();
            let strict = world.get(name).cloned().unwrap_or_else(|| engine_webgpu::uniformize_workgroup_lanes(&raw));
            let fast = engine_webgpu::fast_float_wgsl(&strict).into_owned();
            std::fs::write(to.join(format!("{name}.wgsl")), &strict).unwrap();
            if fast != strict {
                std::fs::write(to.join(format!("{name}.fast.wgsl")), fast).unwrap();
            }
            n += 1;
        }
        println!("{n} modules -> {}", to.display());
        return;
    }
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
