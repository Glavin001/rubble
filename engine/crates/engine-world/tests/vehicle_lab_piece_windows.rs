//! The coupling's window per piece (`vehicle_coupling.slang` pieceAccept,
//! `ExplicitBeamsGpu::set_piece_windows`): with the car's piece labels cpl_bounds takes each
//! piece's node box and a collider is a proxy only when it can reach one of them. The
//! procedural sedan launched down the lab's mountain with its front torn off and placed 30 m ahead:
//! the run with a window per piece is the same bit for bit as with one window, and the
//! coupling holds a small box per piece, 30 m apart. (The proxies left out on the lab's
//! mountain: lab-mountain-perf.md section 7.)
#![cfg(feature = "stress")]

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_webgpu::explicit_coupling::read_words;
use rubble::vehicle_lab::{Lab, LabOptions};
use rubble::*;
use serde_json::json;

/// The procedural sedan with its front 0.5 m torn off and dropped 30 m ahead.
fn car() -> ExplicitBeams {
    let text = engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
        minimass: 10.0,
        ..engine_softbody::procedural::CarParams::sedan()
    });
    let mut s = ExplicitBeams::from_truck_raycast(
        &text,
        None,
        glam::Vec3::ZERO,
        0.0,
        glam::Vec3::ZERO,
        ExplicitBeamsSettings { fast_inv_sqrt: true, ..Default::default() },
    )
    .unwrap()
    .0;
    let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let cut: Vec<bool> = s.nodes.iter().map(|n| n.pos.z > front - 0.5).collect();
    assert!(!cut[0]);
    for b in &mut s.beams {
        if cut[b.a as usize] != cut[b.b as usize] {
            b.broken = true;
        }
    }
    for (n, &c) in s.nodes.iter_mut().zip(&cut) {
        if c {
            n.pos += glam::Vec3::new(0.0, 3.0, 30.0);
        }
    }
    s
}

fn unordered(u: u32) -> f32 {
    f32::from_bits(if u & 0x8000_0000 != 0 { u & 0x7FFF_FFFF } else { !u })
}

/// 150 World steps; the most proxy shapes in a window, the pieces' boxes the coupling took
/// at the end (label, lo, hi), and the car's node and beam records and origin as words.
fn run(windows: bool) -> (u32, Vec<(usize, glam::Vec3, glam::Vec3)>, Vec<u32>) {
    let opts = LabOptions::from_json(&json!({ "scene": "mountain", "pieceWindows": windows, "debris": { "on": false } }));
    let (mut world, mut lab) = Lab::build(GpuContext::Default, &car(), opts, |_| {}).unwrap();
    lab.read_blocking(&world, true).unwrap();
    let mut most = 0u32;
    for _ in 0..150 {
        lab.before_step(&mut world).unwrap();
        world.step().unwrap();
        world.poll().unwrap();
        lab.read_blocking(&world, true).unwrap();
        most = most.max(lab.bench.car.proxies[1]);
    }
    world.poll_blocking().unwrap();
    assert!(!lab.piece_labels().is_empty(), "the torn piece is labelled");
    let x = world.explicit_vehicles().unwrap();
    let p = lab.read_plan();
    let (d, q) = (world.device(), world.queue());
    let rec = u64::from(x.layout().piece_box_rec) * 16;
    eprintln!("piece window records {:?}, layout {} {}", x.piece_window_records(), x.layout().label_rec, x.layout().piece_box_rec);
    let keys = read_words(d, q, x.misc_buffer(), rec, 13 * 16);
    let o = read_words(d, q, x.misc_buffer(), p.origin.0, p.origin.1);
    let origin = glam::Vec3::new(f32::from_bits(o[0]), f32::from_bits(o[1]), f32::from_bits(o[2]));
    let boxes = (0..8)
        .filter(|&g| keys[6 * g] != 0xFFFF_FFFF)
        .map(|g| {
            let k = |j: usize| unordered(keys[6 * g + j]);
            (g, glam::Vec3::new(k(0), k(1), k(2)) - origin, glam::Vec3::new(k(3), k(4), k(5)) - origin)
        })
        .collect();
    let mut w = read_words(d, q, x.nodes_buffer(), p.nodes.0, p.nodes.1);
    w.extend(read_words(d, q, x.beams_buffer(), p.beams.0, p.beams.1));
    w.extend(o);
    (most, boxes, w)
}

#[test]
fn a_window_per_piece_is_bit_for_bit_and_boxes_each_piece() {
    let (one, _, a) = run(false);
    let (per, boxes, b) = run(true);
    eprintln!("most proxy shapes in a window: one window {one}, a window per piece {per}; piece boxes {boxes:?}");
    let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert_eq!(diff, 0, "{diff} of {} words differ", a.len());
    assert!(per <= one);
    // The main body's box and the torn piece's, 30 m apart along z, each small.
    assert!(boxes.len() >= 2, "piece boxes: {boxes:?}");
    let span = |(_, lo, hi): &(usize, glam::Vec3, glam::Vec3)| (*hi - *lo).max_element();
    assert!(boxes.iter().all(|b| span(b) < 6.0), "every piece box is small: {boxes:?}");
    let zs: Vec<f32> = boxes.iter().map(|(_, lo, _)| lo.z).collect();
    let spread = zs.iter().fold(f32::MIN, |a, &z| a.max(z)) - zs.iter().fold(f32::MAX, |a, &z| a.min(z));
    assert!(spread > 20.0, "the boxes are apart: {boxes:?}");
}
