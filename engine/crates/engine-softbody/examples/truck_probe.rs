//! Import a Rigs of Rods `.truck`-family file, report it, and drop it on the
//! ground with the CPU reference solver.
//! `cargo run --release -p engine-softbody --example truck_probe -- file.truck`
//! (Not run in CI: RoR content is GPL; point it at files you have.)
use engine_softbody::cpu::{CpuSoft, Plane};
use engine_softbody::truck::load_truck;
use glam::Vec3;

fn main() {
    for path in std::env::args().skip(1) {
        let src = std::fs::read_to_string(&path).expect("read");
        let d = load_truck(&src);
        let (mn, mx) = d.bounds();
        println!(
            "{path}\n  {}: {} nodes, {} beams, {} triangles ({} colliding), {} wheels ({} driven, {} steer), {:.0} kg, refs {:?}",
            d.name,
            d.nodes.len(),
            d.beams.len(),
            d.triangles.len(),
            d.triangles.iter().filter(|t| t.collision).count(),
            d.wheels.len(),
            d.wheels.iter().filter(|w| w.propulsion != 0).count(),
            d.wheels.iter().filter(|w| w.steerable).count(),
            d.total_mass(),
            d.refs.is_some()
        );
        println!("  bounds {mn:?} .. {mx:?}");
        let mut kinds = std::collections::BTreeMap::new();
        for b in &d.beams {
            *kinds.entry(format!("{:?}", b.kind)).or_insert(0) += 1;
        }
        println!("  beam kinds {kinds:?}");
        for w in d.warnings.iter().take(10) {
            println!("  warning: {w}");
        }
        if d.warnings.len() > 10 {
            println!("  … {} more warnings", d.warnings.len() - 10);
        }
        let mut s = CpuSoft::from_desc(&d, Vec3::new(0.0, 0.3 - mn[1], 0.0), Vec3::ZERO);
        s.h = 1.0 / 240.0;
        s.iterations = 10;
        s.chebyshev_rho = 0.9;
        s.planes.push(Plane { normal: Vec3::Y, offset: 0.0, friction: 0.8 });
        for k in 0..480 {
            s.step();
            if k % 160 == 159 {
                println!(
                    "  t {:.2}s ke {:.1} J, lowest {:.3} m, yielded {}, broken {}, finite {}",
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
}
