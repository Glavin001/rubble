//! `rubble_tension_yield` (jbeam-semantics-spec.md §18.4) on the GPU against the
//! CPU reference, bit for bit: beams whose tension yield Tt differs from their
//! compression yield Tc, stretched and squeezed past both, in the `.truck` yield
//! rule (`Plain`) and JBeam's (`JbeamPlain` with plastic limits). The kernel's
//! separate Tc and Tt branches only ever ran with Tc = Tt before this extension.

use engine_softbody::explicit_beams::{
    Beam, ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, ExplicitType, Node,
};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
};

/// Pairs of 1 kg nodes 0.5 m apart, high above the ground, each pair joined by
/// one deformable beam (k 1e5 N/m, Tc 500 N, S 20 kN) and flung apart (even
/// pairs) or together (odd pairs). Pairs 0–1 take Tt 3 kN; pairs
/// 2–3 are JBeam NORMAL beams (deformLimitStress 4 kN) with Tt 3 kN; pairs 4–5
/// keep Tt = Tc.
fn pairs() -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -100.0;
    for p in 0..6u32 {
        // Stretched at 15 m/s each end (225 J against Tt²/2k = 45 J), squeezed
        // at 2 m/s (4 J against Tc²/2k = 1.25 J).
        let sign = if p % 2 == 0 { 15.0 } else { -2.0 };
        for x in [0.0f32, 0.5] {
            s.nodes.push(Node {
                pos: Vec3::new(x, 10.0, p as f32),
                vel: Vec3::new(sign * if x == 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
                force: Vec3::ZERO,
                mass: 1.0,
                friction: 1.0,
                flags: 0,
            });
        }
        s.beams.push(Beam::new(
            2 * p,
            2 * p + 1,
            ExplicitKind::Plain,
            ExplicitType::Deformable,
            1.0e5,
            50.0,
            0.5,
            500.0,
            20_000.0,
            0.0,
        ));
    }
    s.finish();
    s.set_jbeam_plastic(&[(4, 5, 4_000.0, 0.5, 0.5, false), (6, 7, 4_000.0, 0.5, 0.5, false)])
        .unwrap();
    s.set_tension_yield(&[
        (0, 1, 3_000.0),
        (2, 3, 3_000.0),
        (4, 5, 3_000.0),
        (6, 7, 3_000.0),
    ])
    .unwrap();
    s
}

#[test]
fn tension_yield_steps_like_the_cpu() {
    let mut cpu = pairs();
    assert_eq!(cpu.beams[0].tt, 3_000.0);
    assert_eq!(cpu.beams[0].tc, 500.0);
    assert_eq!(cpu.beams[4].tt, cpu.beams[4].tc);
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    let bits = |a: Vec3| a.to_array().map(f32::to_bits);
    // Each beam's largest and smallest plastic strain over the run: a stretched
    // beam yields in tension first (its rebound may yield it back in compression
    // after, as its Tc is low: yield follows the elastic force only, jbeam spec §3).
    let mut extreme = vec![(0.0f32, 0.0f32); cpu.beams.len()];
    for step in 0..400 {
        cpu.step();
        for (j, b) in cpu.beams.iter().enumerate() {
            let e = b.rest / b.rest0 - 1.0;
            extreme[j] = (extreme[j].0.max(e), extreme[j].1.min(e));
        }
        gpu.frame(1);
        gpu.wait();
        let (p, v) = (&gpu.positions()[0], &gpu.velocities()[0]);
        for i in 0..cpu.nodes.len() {
            assert!(
                bits(cpu.nodes[i].pos) == bits(p[i]) && bits(cpu.nodes[i].vel) == bits(v[i]),
                "step {step}: node {i} cpu {:?} {:?} gpu {:?} {:?}",
                cpu.nodes[i].pos,
                cpu.nodes[i].vel,
                p[i],
                v[i]
            );
        }
        let gb = &gpu.beams()[0];
        for (j, b) in cpu.beams.iter().enumerate() {
            assert!(
                b.rest.to_bits() == gb[j].rest.to_bits() && b.broken == gb[j].broken,
                "step {step}: beam {j} cpu rest {} broken {} gpu rest {} broken {}",
                b.rest,
                b.broken,
                gb[j].rest,
                gb[j].broken
            );
        }
    }
    // Each stretched beam yielded in tension, each squeezed one in compression.
    let strain: Vec<f32> = cpu.beams.iter().map(|b| b.rest / b.rest0 - 1.0).collect();
    eprintln!("plastic strain per beam after 0.2 s: {strain:?}");
    eprintln!("largest / smallest plastic strain per beam: {extreme:?}");
    for (j, (hi, lo)) in extreme.iter().enumerate() {
        if j % 2 == 0 {
            assert!(*hi > 0.0, "beam {j} did not yield in tension");
        } else {
            assert!(*lo < 0.0, "beam {j} did not yield in compression");
        }
    }
    // The tension side yields less with Tt 3 kN than at Tc = Tt = 500 N.
    assert!(extreme[0].0 < extreme[4].0);
    let _ = strain;
}

/// The BeamNG profile's ropes and tearing (jbeam-semantics-spec.md §36) on the GPU,
/// bit for bit: rope pairs (tension-only JBeam beams, `JbeamRope`) and plain pairs,
/// flung apart (even) or together (odd), half of them tearing at dₑ 0.2
/// (`breakAtExpansion`), none losing strength as they stretch.
#[test]
fn ropes_and_tearing_step_like_the_cpu() {
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -100.0;
    for p in 0..8u32 {
        let sign = if p % 2 == 0 { 15.0 } else { -2.0 };
        for x in [0.0f32, 0.5] {
            s.nodes.push(Node {
                pos: Vec3::new(x, 10.0, p as f32),
                vel: Vec3::new(sign * if x == 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
                force: Vec3::ZERO,
                mass: 1.0,
                friction: 1.0,
                flags: 0,
            });
        }
        let kind = if p < 4 { ExplicitKind::Rope } else { ExplicitKind::Plain };
        s.beams.push(Beam::new(2 * p, 2 * p + 1, kind, ExplicitType::Deformable, 1.0e5, 50.0, 0.5, 500.0, 50_000.0, 0.0));
    }
    s.finish();
    let lines: Vec<(u32, u32, f32, f32, f32, bool)> =
        (0..8u32).map(|p| (2 * p, 2 * p + 1, -1.0, -1.0, 0.2, p % 4 < 2)).collect();
    s.set_jbeam_plastic(&lines).unwrap();
    s.plastic_strength_drop = false;
    assert_eq!(s.beams[0].kind, ExplicitKind::JbeamRope);
    let mut cpu = s;
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    let bits = |a: Vec3| a.to_array().map(f32::to_bits);
    for step in 0..400 {
        cpu.step();
        gpu.frame(1);
        gpu.wait();
        let (p, v) = (&gpu.positions()[0], &gpu.velocities()[0]);
        for i in 0..cpu.nodes.len() {
            assert!(
                bits(cpu.nodes[i].pos) == bits(p[i]) && bits(cpu.nodes[i].vel) == bits(v[i]),
                "step {step}: node {i}"
            );
        }
        let gb = &gpu.beams()[0];
        for (j, b) in cpu.beams.iter().enumerate() {
            assert!(
                b.rest.to_bits() == gb[j].rest.to_bits() && b.broken == gb[j].broken,
                "step {step}: beam {j} cpu rest {} broken {} gpu rest {} broken {}",
                b.rest, b.broken, gb[j].rest, gb[j].broken
            );
        }
    }
    let broken: Vec<bool> = cpu.beams.iter().map(|b| b.broken).collect();
    eprintln!("broken: {broken:?}; strengths {:?}", cpu.beams.iter().map(|b| b.strength).collect::<Vec<_>>());
    // Flung apart with the tear flag: torn; without it: held at 1.2·L; squeezed: held.
    assert_eq!(broken, [true, false, false, false, true, false, false, false]);
    assert!(cpu.beams.iter().all(|b| b.strength == 50_000.0));
    assert!((cpu.beams[2].rest - 0.6).abs() < 1e-6);
}

