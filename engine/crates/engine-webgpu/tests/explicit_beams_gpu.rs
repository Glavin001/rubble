//! Explicit beams on the GPU against the CPU reference
//! (`engine_softbody::explicit_beams`) on the procedural crash-test sedan: at rest
//! the two agree to millimetres, and a 56 km/h crash into a rigid box breaks and
//! bends the same number of beams. Turbulence is off on both (the GPU draws its
//! noise differently).

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, StaticBox,
};
use engine_softbody::procedural::{car_truck_bare, CarParams};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn sedan(velocity: Vec3) -> ExplicitBeams {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    // Lowest node 1 cm above the ground.
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, velocity, settings).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), velocity, settings).unwrap()
}

fn rms_max(a: &[Vec3], b: &[Vec3]) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    for (p, q) in a.iter().zip(b) {
        let d = p.distance(*q);
        sum += (d as f64).powi(2);
        max = max.max(d);
    }
    ((sum / a.len() as f64).sqrt() as f32, max)
}

/// Run both for `frames` 60 fps frames; returns the GPU host.
fn run_both(cpu: &mut ExplicitBeams, frames: u32) -> Option<ExplicitBeamsGpu> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(cpu));
    let mut clock = FrameClock::default();
    for _ in 0..frames {
        cpu.frame();
        gpu.frame(clock.next());
    }
    gpu.wait();
    Some(gpu)
}

#[test]
fn sedan_at_rest_matches_cpu() {
    let mut cpu = sedan(Vec3::ZERO);
    let Some(gpu) = run_both(&mut cpu, 60) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
    eprintln!(
        "at rest after 1 s: GPU vs CPU rms {:.3} mm, max {:.3} mm",
        rms * 1e3,
        max * 1e3
    );
    assert!(rms < 3e-3, "rms {rms} m");
    assert!(max < 1e-2, "max {max} m");
    // It is resting: nothing broke and the speed is small.
    assert_eq!(cpu.damage().0, 0);
    let vmax = gpu.velocities()[0]
        .iter()
        .map(|v| v.length())
        .fold(0.0, f32::max);
    assert!(vmax < 0.5, "GPU node speed {vmax} m/s at rest");
}

#[test]
fn sedan_crash_matches_cpu_counts() {
    let speed = 56.0 / 3.6;
    let mut cpu = sedan(Vec3::new(0.0, 0.0, speed));
    let front = cpu.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    // A rigid block 5 cm in front of the car, wider and taller than it.
    cpu.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
        Vec3::new(3.0, 2.0, 1.0),
    ));
    let Some(gpu) = run_both(&mut cpu, 36) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (c_broken, c_bent) = cpu.damage();
    let beams = &gpu.beams()[0];
    let g_broken = beams.iter().filter(|b| b.broken).count();
    let g_bent = beams.iter().filter(|b| b.bent()).count();
    let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
    eprintln!(
        "crash 0.6 s: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}; \
         positions rms {:.1} mm max {:.1} mm",
        rms * 1e3,
        max * 1e3
    );
    assert!(c_bent > 100, "the crash bends the front ({c_bent})");
    let close = |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.05 * a.max(b) as f32 + 1.0;
    assert!(close(c_broken, g_broken), "broken {c_broken} vs {g_broken}");
    assert!(close(c_bent, g_bent), "bent {c_bent} vs {g_bent}");
}

/// A box glider with a wing on its top face (the 4-node layout), a fuselage drag,
/// a self-centering command, a rope command and a hydro: the GPU's aero module
/// against the CPU. A synthetic airfoil table (no file needed).
const GLIDER: &str = "glider
globals
400, 0, x
enable_advanced_deformation
set_beam_defaults 2000000, 3000, 900000, 300000, -1, -1, 0.0
nodes
0, 0, 0, 0
1, 0, 0, -2
2, 1, 0, -2
3, 1, 0, 0
4, 0, 0.4, 0
5, 0, 0.4, -2
6, 1, 0.4, -2
7, 1, 0.4, 0
beams
0, 1
1, 2
2, 3
3, 0
4, 5
5, 6
6, 7
7, 4
0, 4
1, 5
2, 6
3, 7
0, 2
1, 3
4, 6
5, 7
0, 5
1, 4
2, 7
3, 6
0, 7
3, 4
1, 6
2, 5
hydros
0, 6, 0.0
commands2
3, 5, 0.5, 0.25, 0.8, 1.5, 3, 4, c
1, 7, 0.5, 0.5, 0.9, 1.2, 1, 2, r
wings
4, 5, 4, 5, 7, 6, 7, 6, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, n, 1.0, 0.0, 0.0, syn.afl 0.8
fusedrag
4, 6, 1.5, syn.afl
end
";

fn synthetic_afl() -> String {
    let mut s = String::from("A\n700 version\nalpha cl cd cm:\n");
    for a10 in -1800i32..=1800 {
        let a = a10 as f32 / 10.0;
        let ip = a10 / 10;
        let sign = if a10 < 0 && ip == 0 { "-" } else { "" };
        let r = a.to_radians();
        s.push_str(&format!(
            "{sign}{ip}.{}  {}  {}  {}\n",
            (a10 % 10).abs(),
            1.1 * (2.0 * r).sin(),
            0.01 + 1.2 * r.sin() * r.sin(),
            -0.05 * (2.0 * r).sin()
        ));
    }
    s
}

#[test]
fn glider_with_wings_fusedrag_and_commands_matches_cpu() {
    let afl = synthetic_afl();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let mut cpu = ExplicitBeams::from_truck_with_airfoils(
        GLIDER,
        Vec3::new(0.0, 1.5, 0.0),
        Vec3::new(-30.0, 2.0, 0.0),
        settings,
        &|_| Some(afl.clone()),
    )
    .unwrap();
    assert_eq!((cpu.wings.len(), cpu.centering.len()), (1, 1));
    assert!(cpu.fusedrag.is_some());
    let Some(gpu) = run_both(&mut cpu, 90) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
    let b = &gpu.beams()[0];
    let cmd = cpu.centering[0].beam as usize;
    let d = gpu.fuse_forces()[0];
    eprintln!(
        "glider 1.5 s: rms {:.4} mm max {:.4} mm; centering rest CPU {} GPU {}; D CPU {:?} GPU {:?}; \
         travel {:?}",
        rms * 1e3,
        max * 1e3,
        cpu.beams[cmd].rest,
        b[cmd].rest,
        cpu.fuse_force,
        d,
        cpu.positions()[0]
    );
    // The wing lifts and drags it, the fuselage drag slows it (a check that the
    // features act at all), and both engines agree.
    assert!(cpu.fuse_force.length() > 1.0);
    assert!((cpu.beams[cmd].rest - b[cmd].rest).abs() < 1e-5);
    assert!(cpu.beams[cmd].rest > cpu.beams[cmd].rest0 + 0.1);
    assert!((cpu.fuse_force - d).length() < 1e-3 * cpu.fuse_force.length());
    assert_eq!(gpu.wings_broken()[0], vec![cpu.wings[0].broken]);
    assert!(rms < 1e-3 && max < 3e-3, "rms {rms} max {max}");
}

/// A box-frame car on four `meshwheels2` tyres (with and without a rigidity
/// node) and four `shocks2` (hard bump at the front, soft at the rear, metric
/// bounds on one): the GPU's shocks2 rule and the mesh wheels' spawn against the
/// CPU.
fn shocks2_car() -> String {
    let mut t = String::from(
        "shocks2 car\nglobals\n600, 0, x\nenable_advanced_deformation\nset_beam_defaults 3000000, 2000, 25000, 60000\nnodes\n",
    );
    let mut n = 0;
    let mut node = |t: &mut String, x: f32, y: f32, z: f32| {
        t.push_str(&format!("{n}, {x}, {y}, {z}\n"));
        n += 1;
        n - 1
    };
    let mut frame = Vec::new();
    for &x in &[-1.2f32, 1.2] {
        for &y in &[0.35f32, 0.85] {
            for &z in &[-0.6f32, 0.6] {
                frame.push(node(&mut t, x, y, z));
            }
        }
    }
    // Per wheel: inner and outer axle node, and the frame corners it hangs from
    // (lower front/back, upper).
    let mut axles = Vec::new();
    for &x in &[-0.9f32, 0.9] {
        for &side in &[-1.0f32, 1.0] {
            let inner = node(&mut t, x, 0.32, side * 0.62);
            let outer = node(&mut t, x, 0.32, side * 0.85);
            axles.push((x, side, inner, outer));
        }
    }
    t.push_str("beams\n");
    for i in 0..frame.len() {
        for j in i + 1..frame.len() {
            t.push_str(&format!("{}, {}\n", frame[i], frame[j]));
        }
    }
    let corner = |x: f32, y: f32, side: f32| {
        let xi = (x > 0.0) as usize;
        let yi = (y > 0.5) as usize;
        let zi = (side > 0.0) as usize;
        frame[xi * 4 + yi * 2 + zi]
    };
    for &(x, side, inner, outer) in &axles {
        let low = corner(x, 0.35, side);
        let far = corner(-x, 0.35, side);
        let up = corner(x, 0.85, side);
        let other = corner(x, 0.35, -side);
        t.push_str(&format!(
            "{inner}, {outer}\n{inner}, {low}\n{inner}, {far}\n{outer}, {low}\n"
        ));
        t.push_str(&format!(
            "{outer}, {far}\n{inner}, {other}\n{outer}, {up}\n"
        ));
    }
    t.push_str("shocks2\n");
    for &(x, side, inner, _) in &axles {
        let up = corner(x, 0.85, side);
        let opts = match (x > 0.0, side > 0.0) {
            (false, false) => "n",
            (false, true) => "m",
            (true, false) => "s",
            (true, true) => "si",
        };
        let (sb, lb) = if opts == "m" {
            (0.1, 0.15)
        } else {
            (0.25, 0.35)
        };
        t.push_str(&format!(
            "{inner}, {up}, 40000, 3000, 1.5, 0.5, 30000, 2500, 0.8, 1.5, {sb}, {lb}, 0.95, {opts}\n"
        ));
    }
    t.push_str("set_beam_defaults 900000, 600, 400000, 3000000\nmeshwheels2\n");
    for (k, &(_, _, inner, outer)) in axles.iter().enumerate() {
        let rigidity = if k % 2 == 0 {
            frame[0].to_string()
        } else {
            "9999".into()
        };
        t.push_str(&format!(
            "0.3, 0.2, 0.2, 10, {inner}, {outer}, {rigidity}, 1, 0, {inner}, 20, 120000, 400, l, rim.mesh, mat\n"
        ));
    }
    t.push_str("end\n");
    t
}

#[test]
fn meshwheels2_and_shocks2_car_matches_cpu() {
    let text = shocks2_car();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let spawn = |vel: Vec3| {
        let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, vel, settings).unwrap();
        let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
        let mut s =
            ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), vel, settings)
                .unwrap();
        s.roll_tyres(vel);
        s
    };
    // Dropped and settling for 1 s.
    let mut cpu = spawn(Vec3::ZERO);
    assert_eq!(cpu.shocks2.len(), 4);
    assert_eq!(cpu.tyres.len(), 4);
    let Some(gpu) = run_both(&mut cpu, 60) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
    eprintln!(
        "shocks2 car at rest 1 s: rms {:.4} mm max {:.4} mm",
        rms * 1e3,
        max * 1e3
    );
    assert_eq!(cpu.damage().0, 0);
    assert!(rms < 1e-3 && max < 3e-3, "rms {rms} max {max}");

    // Driven into a wall at 20 m/s for 0.8 s: the same damage.
    let mut cpu = spawn(Vec3::new(-20.0, 0.0, 0.0));
    let front = cpu.positions().iter().map(|p| p.x).fold(f32::MAX, f32::min);
    cpu.boxes.push(StaticBox::centered(
        Vec3::new(front - 0.5 - 1.0, 1.0, 0.0),
        Vec3::new(1.0, 2.0, 3.0),
    ));
    let Some(gpu) = run_both(&mut cpu, 48) else {
        return;
    };
    let (c_broken, c_bent) = cpu.damage();
    let beams = &gpu.beams()[0];
    let g_broken = beams.iter().filter(|b| b.broken).count();
    let g_bent = beams.iter().filter(|b| b.bent()).count();
    let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
    eprintln!(
        "shocks2 car crash 0.8 s: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}; \
         rms {:.1} mm max {:.1} mm",
        rms * 1e3,
        max * 1e3
    );
    assert!(
        c_broken + c_bent > 10,
        "the crash damages it ({c_broken}, {c_bent})"
    );
    let close = |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.05 * a.max(b) as f32 + 2.0;
    assert!(close(c_broken, g_broken), "broken {c_broken} vs {g_broken}");
    assert!(close(c_bent, g_bent), "bent {c_bent} vs {g_bent}");
}
