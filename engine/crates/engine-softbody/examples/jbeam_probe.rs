//! Load a JBeam vehicle from local files and report what was imported, then
//! drop it on the ground with the CPU reference solver.
//! `cargo run --release -p engine-softbody --example jbeam_probe -- <dir with .jbeam files> [config.pc]`
//! (Not run in CI: BeamNG content is not redistributable; point it at files you have.)
use engine_softbody::cpu::{CpuSoft, Plane};
use engine_softbody::jbeam::load_jbeam;
use glam::Vec3;

fn collect(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
    for e in std::fs::read_dir(dir).expect("read dir").flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|x| x == "jbeam") {
            out.push((
                p.display().to_string(),
                std::fs::read_to_string(&p).unwrap_or_default(),
            ));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut files = Vec::new();
    for a in args.iter().filter(|a| !a.ends_with(".pc")) {
        collect(std::path::Path::new(a), &mut files);
    }
    let pc = args
        .iter()
        .find(|a| a.ends_with(".pc"))
        .map(|p| std::fs::read_to_string(p).expect("pc"));
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let t = std::time::Instant::now();
    let d = load_jbeam(&refs, pc.as_deref()).expect("load");
    println!(
        "{}: {} files → {} nodes, {} beams, {} triangles, {} wheels ({} driven), {:.0} kg in {:.0} ms",
        d.name,
        files.len(),
        d.nodes.len(),
        d.beams.len(),
        d.triangles.len(),
        d.wheels.len(),
        d.wheels.iter().filter(|w| w.propulsion != 0).count(),
        d.total_mass(),
        t.elapsed().as_secs_f64() * 1e3
    );
    for w in d.warnings.iter().take(12) {
        println!("  warning: {w}");
    }
    let (mn, mx) = d.bounds();
    println!("  bounds {mn:?} .. {mx:?}");
    let mut s = CpuSoft::from_desc(&d, Vec3::new(0.0, 0.3 - mn[1], 0.0), Vec3::ZERO);
    s.h = 1.0 / 240.0;
    s.iterations = 10;
    s.chebyshev_rho = 0.9;
    s.planes.push(Plane {
        normal: Vec3::Y,
        offset: 0.0,
        friction: 0.8,
    });
    for k in 0..480 {
        s.step();
        if k % 120 == 119 {
            println!(
                "  t {:.1}s ke {:.1} J, lowest {:.3} m, yielded {}, broken {}, finite {}",
                (k + 1) as f32 / 240.0,
                s.kinetic_energy(),
                s.x.iter().map(|p| p.y).fold(f32::MAX, f32::min),
                s.yielded_beams(),
                s.broken_beams(),
                s.x.iter().all(|p| p.is_finite())
            );
        }
    }
}
