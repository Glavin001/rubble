//! Vehicle contact's candidates in a grid (`explicit_beams.slang` vvGridBuild,
//! vvTriangleCandsGrid; `ExplicitBeamsGpu::set_vv_grid`): in a small contact group each
//! listed triangle tests only the partner contact points in the cells its box covers,
//! not every candidate. Two sedans head-on, each frame from one saved state with the grid
//! off then on: every state word the same, and far fewer candidates visited.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn first_diff(a: &[(&str, Vec<u32>)], b: &[(&str, Vec<u32>)]) -> Option<String> {
    for ((name, x), (_, y)) in a.iter().zip(b) {
        if x.len() != y.len() {
            return Some(format!("{name}: lengths {} and {}", x.len(), y.len()));
        }
        if let Some(i) = x.iter().zip(y).position(|(p, q)| p != q) {
            let count = x.iter().zip(y).filter(|(p, q)| p != q).count();
            return Some(format!("{name}[{i}] ({:#x} vs {:#x}), {count} words differ", x[i], y[i]));
        }
    }
    None
}

fn sedan(at: Vec3, yaw: f32, vel: Vec3) -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    let mut s = if yaw == 0.0 {
        ExplicitBeams::from_truck_config(&text, None, at, vel, settings)
    } else {
        ExplicitBeams::from_truck_config_yaw(&text, None, at, yaw, vel, settings)
    }
    .expect("spawn");
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s.roll_tyres(vel);
    s
}

#[test]
fn the_candidates_grid_is_the_brute_test_bit_for_bit_with_fewer_visits() {
    // The sedan faces -x; the second, turned 180°, 1.5 m past the first's front.
    let a = sedan(Vec3::ZERO, 0.0, Vec3::new(-12.0, 0.0, 0.0));
    let front = a.nodes.iter().map(|n| n.pos.x).fold(f32::MAX, f32::min);
    let probe = sedan(Vec3::ZERO, 180.0, Vec3::ZERO);
    let back = probe.nodes.iter().map(|n| n.pos.x).fold(f32::MIN, f32::max);
    let b = sedan(Vec3::new(front - 1.5 - back, 0.0, 0.0), 180.0, Vec3::new(12.0, 0.0, 0.0));
    let fleet = vec![a, b];
    let Some((_, device, queue)) = default_device().ok() else { return };
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    gpu.set_vehicle_contact(true);
    gpu.set_wide(false);
    let (mut brute, mut grid, mut grouped) = (0u64, 0u64, 0u32);
    for f in 0..40 {
        let snap = gpu.snapshot();
        gpu.set_vv_grid(false);
        gpu.take_vv_visits();
        gpu.frame(17);
        gpu.wait();
        let off = gpu.state_words();
        let v_off = gpu.take_vv_visits();
        gpu.restore(&snap);
        gpu.set_vv_grid(true);
        gpu.frame(17);
        gpu.wait();
        let on = gpu.state_words();
        let v_on = gpu.take_vv_visits();
        if let Some(d) = first_diff(&off, &on) {
            panic!("frame {f}: the grid differs from every candidate at {d}");
        }
        brute += v_off;
        grid += v_on;
        grouped += u32::from(gpu.contact_groups() > 0);
    }
    eprintln!("40 frames ({grouped} with a contact group): candidates visited, every candidate {brute}, the grid {grid}");
    assert!(brute > 0, "the cars met (candidates were tested)");
    assert!(grid * 4 < brute, "the grid visited {grid} candidates against {brute}");
}
