//! A JBeam vehicle in the explicit solver's JBeam mode with pressure wheels
//! (`engine/docs/jbeam-semantics-spec.md` Part II), CPU reference: the census of
//! what it uses and does not simulate, at rest for `rest=` s, then into a rigid wall
//! at `kmh=` (default 56) for `crash=` s.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example jbeam_accept -- <vehicles-dir> <name> <config.pc> [rest=3] [kmh=56] [crash=1.5] [wheels=pressure|soft]
//! ```
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, StaticBox,
};
use glam::Vec3;

fn report(tag: &str, s: &ExplicitBeams, ke0: f32) {
    let ke: f32 = s
        .nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum();
    let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
    let nan = s.nodes.iter().any(|n| !n.pos.is_finite());
    let (broken, bent) = s.damage();
    let deform = s.deform_triggered().iter().filter(|&&d| d != 0).count();
    let dead = s.tri_dead.iter().filter(|&&d| d).count();
    println!(
        "{tag}: broken {broken} bent {bent} KE {ke:.1} J (spawn {ke0:.1}) max node speed {vmax:.2} m/s \
         NaN {nan} deform groups {deform}/{} dead triangles {dead}/{}",
        s.deform_state.len(),
        s.tri_dead.len()
    );
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (dir, name, pc) = (&a[0], &a[1], &a[2]);
    let opt = |k: &str| a.iter().find_map(|x| x.strip_prefix(k)).map(str::to_string);
    let f = |k: &str, d: f32| opt(k).map_or(d, |s| s.parse().unwrap());
    let (rest, kmh, crash) = (f("rest=", 3.0), f("kmh=", 56.0), f("crash=", 1.5));
    let dir = std::path::Path::new(dir);
    let texts = engine_softbody::jbeam::read_vehicle_dir(dir, name).unwrap();
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pcs = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
    let census = engine_softbody::jbeam::feature_census(&files, Some(&pcs)).unwrap();
    let unsupported = engine_softbody::jbeam::unsupported_features(&census);
    println!("unsupported physics features: {unsupported:?}");
    let desc = engine_softbody::jbeam::load_jbeam(&files, Some(&pcs)).unwrap();
    let wheels = if opt("wheels=").as_deref() == Some("soft") {
        JbeamWheels::Soft(engine_softbody::truck_export::TyreDefaults::for_vehicle(
            &desc,
        ))
    } else {
        JbeamWheels::Pressure
    };
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let spawn = |v: Vec3| {
        let (mut s, w) =
            ExplicitBeams::from_desc_jbeam_with(&desc, wheels, Vec3::ZERO, v, settings).unwrap();
        let low = s
            .nodes
            .iter()
            .map(|n| n.pos.y)
            .fold(f32::INFINITY, f32::min);
        for n in &mut s.nodes {
            n.pos.y -= low - 0.02;
        }
        s.roll_tyres(v);
        (s, w)
    };
    let (mut s, warnings) = spawn(Vec3::ZERO);
    for w in &warnings {
        println!("warning: {w}");
    }
    let mass: f32 = s.nodes.iter().map(|n| n.mass).sum();
    println!(
        "{name} {pc}: {} nodes ({} desc), {} beams ({} with extensions), {} collision triangles, {} kg, {} wheels",
        s.nodes.len(),
        desc.nodes.len(),
        s.beams.len(),
        s.beam_ext.len(),
        s.triangles.len(),
        mass,
        desc.pressure_wheels.len()
    );
    if opt("trace=").is_some() {
        for pw in &desc.pressure_wheels {
            println!(
                "wheel {} values {:?} flags {:?}",
                pw.name, pw.values, pw.flags
            );
        }
        let n0 = desc.nodes.len();
        for step in 0..4000 {
            s.step();
            let bad = s
                .nodes
                .iter()
                .position(|n| !n.pos.is_finite() || n.vel.length() > 30.0);
            if let Some(i) = bad {
                println!(
                    "step {step}: node {i} (wheel node {}) {:?} {}",
                    i >= n0,
                    s.nodes[i],
                    desc.nodes.get(i).map_or("", |n| n.name.as_str())
                );
                for (j, b) in s.beams.iter().enumerate() {
                    if b.a as usize == i || b.b as usize == i {
                        let o = if b.a as usize == i { b.b } else { b.a } as usize;
                        println!(
                            "  beam {j} other {} m {} {:?} ext {:?}",
                            desc.nodes.get(o).map_or("wheel", |n| n.name.as_str()),
                            s.nodes[o].mass,
                            b,
                            s.ext_of
                                .get(j)
                                .filter(|&&x| x != 0)
                                .map(|&x| s.beam_ext[x as usize - 1])
                        );
                    }
                }
                return;
            }
        }
        println!("no blow-up in 2 s");
        let mut ke: Vec<(f32, usize)> = s
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (0.5 * n.mass * n.vel.length_squared(), i))
            .collect();
        ke.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(e, i) in ke.iter().take(8) {
            println!(
                "  KE {e:.1} J node {i} {} m {} v {:?} y {:.3}",
                desc.nodes.get(i).map_or("wheel", |n| n.name.as_str()),
                s.nodes[i].mass,
                s.nodes[i].vel,
                s.nodes[i].pos.y
            );
        }
        return;
    }
    let t0 = std::time::Instant::now();
    for k in 1..=(rest.ceil() as usize) {
        s.run(1.0);
        report(&format!("rest {k} s"), &s, 0.0);
    }
    println!("rest: {:.1} s wall", t0.elapsed().as_secs_f32());
    let v = Vec3::new(0.0, 0.0, kmh / 3.6);
    let (mut s, _) = spawn(v);
    let front = s
        .nodes
        .iter()
        .map(|n| n.pos.z)
        .fold(f32::NEG_INFINITY, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.5, front + 0.3 + 1.0),
        Vec3::new(5.0, 3.0, 1.0),
    ));
    let ke0: f32 = s
        .nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum();
    let steps = (crash / 0.1).ceil() as usize;
    for k in 1..=steps {
        s.run(0.1);
        if k % 5 == 0 || k == steps {
            report(&format!("crash {:.1} s", 0.1 * k as f32), &s, ke0);
        }
    }
}
