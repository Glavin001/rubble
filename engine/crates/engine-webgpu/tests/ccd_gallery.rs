//! motion.html's CCD gallery natively (?scene=ccd): five lanes of walls, balls
//! and 49 + 49 falling 0.1 m cubes, parked then launched by a host write, in the
//! native host's contact modes and in the browser's (canonical contacts, the
//! control step modes of world-host.mjs). No CCD cube may end below the slab.
//! (In the browser a touching pair at TOI 0 once lost its contact: ccd_emit's
//! recast came out differently from ccd_toi's, and a cube fell through; the
//! headless-Chrome gallery check, motion-check.mjs ccd, guards that path.)
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity, BODY_CCD, FLAG_SPEC_VELOCITY};
use engine_webgpu::WgpuSim;

fn b(pos: [f32; 3], inv_mass: f32, inv_i: [f32; 3], flags: u32, v: [f32; 3]) -> BodyState {
    BodyState { position: pos, inv_mass, orientation: [0.0, 0.0, 0.0, 1.0], inv_inertia: inv_i, flags, linear_vel: v, sleep_timer: 0.0, angular_vel: [0.0; 3], generation: 0, initial_pos: pos, initial_pad: 0.0, initial_rot: [0.0, 0.0, 0.0, 1.0] }
}
fn col(body: u32, shape: ShapeType, p: [f32; 4], material: u32) -> Collider {
    Collider { body, shape: shape as u32, material, generation: 0, local_pos: [0.0; 3], density: 0.0, local_rot: [0.0, 0.0, 0.0, 1.0], params: p }
}
fn bx(sim: &mut WgpuSim, pos: [f32; 3], h: [f32; 3], mass: f32, flags: u32) -> u32 {
    let inv_i = if mass > 0.0 { [3.0 / (mass * (h[1]*h[1]+h[2]*h[2])), 3.0 / (mass * (h[0]*h[0]+h[2]*h[2])), 3.0 / (mass * (h[0]*h[0]+h[1]*h[1]))] } else { [0.0; 3] };
    let id = sim.add_body(b(pos, if mass > 0.0 { 1.0 / mass } else { 0.0 }, inv_i, flags, [0.0; 3]));
    sim.add_collider(col(id, ShapeType::Box, [h[0], h[1], h[2], (h[0]*h[0]+h[1]*h[1]+h[2]*h[2]).sqrt()], 1)); id
}
#[test]
fn no_ccd_debris_falls_through_the_gallery_slab() {
  for (browser, hier) in [(false, Some(false)), (false, Some(true)), (true, Some(false)), (true, Some(true))] {
    let pre = 5;
    let cap = WorldCapacity { max_bodies: 4000, max_colliders: 8000, max_contacts: 262144, max_candidates: 262144, max_colors: 32, ..WorldCapacity::small() };
    let mut sim = if browser { WgpuSim::new_with_compact_contacts(cap, false).unwrap() } else { WgpuSim::new_with_persistent_contacts(cap).unwrap() };
    let mut p = AvbdParams::default();
    p.dt = 1.0 / 60.0; p.iterations = 10; p.gravity_y = -10.0; p.flags = FLAG_SPEC_VELOCITY;
    sim.set_params(p);
    sim.set_grid_levels(hier);
    if browser { sim.use_control_modes(); }
    sim.set_material(0, Material { friction: 0.9, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    sim.set_material(1, Material { friction: 0.6, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    let g = sim.add_body(b([0.0, -5.0, 0.0], 0.0, [0.0; 3], 0, [0.0; 3])); sim.add_collider(col(g, ShapeType::Box, [400.0, 5.0, 400.0, 565.7], 0));
    let wz = 30.0f32;
    let mut balls = Vec::new();
    for (k, x) in [-16.0f32, -8.0, 0.0, 8.0, 16.0].into_iter().enumerate() {
        match k { 0 => { bx(&mut sim, [x, 1.5, wz + 0.025], [2.5, 1.5, 0.025], 0.0, 0); } 1 => { bx(&mut sim, [x, 1.5, wz + 0.1], [2.5, 1.5, 0.1], 0.0, 0); } 2 => { bx(&mut sim, [x, 1.5, wz + 0.5], [2.5, 1.5, 0.5], 0.0, 0); }
            3 => { for iy in 0..15 { for ix in 0..25 { bx(&mut sim, [x - 2.4 + ix as f32 * 0.2, 0.1 + iy as f32 * 0.2, wz + 0.1], [0.1; 3], 0.0, 0); } } }
            _ => { for iy in 0..12 { for ix in 0..20 { bx(&mut sim, [x - 1.9 + ix as f32 * 0.2, 0.1 + iy as f32 * 0.2, wz + 0.1], [0.1; 3], 4.0, 9); } } } }
        for (dx, ccd) in [(-1.0f32, false), (1.0, true)] {
            let id = sim.add_body(b([x + dx, -500.0, 0.0], 0.2, [250.0; 3], if ccd { BODY_CCD } else { 0 }, [0.0; 3]));
            sim.add_collider(col(id, ShapeType::Sphere, [0.15, 0.0, 0.0, 0.15], 1)); balls.push((id, x + dx));
        }
    }
    bx(&mut sim, [-6.0, 6.1, -14.0], [2.5, 0.1, 2.5], 0.0, 0);
    bx(&mut sim, [6.0, 6.1, -14.0], [2.5, 0.1, 2.5], 0.0, 0);
    let mut shower = Vec::new();
    for (x0, ccd) in [(-6.0f32, false), (6.0, true)] {
        for i in 0..49 {
            let m = 0.3f32; let ii = 3.0 / (m * 2.0 * 0.05 * 0.05);
            let id = sim.add_body(b([x0 + (i % 7) as f32 * 0.6 - 1.8, -600.0, -14.0 + (i / 7) as f32 * 0.6 - 1.8], 1.0 / m, [ii; 3], if ccd { BODY_CCD } else { 0 }, [0.0; 3]));
            sim.add_collider(col(id, ShapeType::Box, [0.05, 0.05, 0.05, 0.0866], 1));
            shower.push((id, ccd, [x0 + (i % 7) as f32 * 0.6 - 1.8, 9.0 + 0.13 * (i % 5) as f32, -14.0 + (i / 7) as f32 * 0.6 - 1.8]));
        }
    }
    for _ in 0..pre { sim.step().unwrap(); }
    let mut all = sim.download_bodies();
    let launch = |all: &mut Vec<BodyState>, id: u32, p: [f32; 3], v: [f32; 3]| { let s = &mut all[id as usize]; let ccd = s.flags & BODY_CCD; s.position = p; s.orientation = [0.0, 0.0, 0.0, 1.0]; s.flags = 1 | 8 | ccd; s.linear_vel = v; s.angular_vel = [0.0; 3]; s.initial_pos = p; s.initial_rot = [0.0, 0.0, 0.0, 1.0]; };
    for &(id, x) in &balls { launch(&mut all, id, [x, 1.4, 4.0], [0.0, 0.0, 60.0]); }
    for &(id, _, p) in &shower { launch(&mut all, id, p, [0.0, -40.0, 0.0]); }
    for &(id, _) in &balls { sim.write_bodies(id, &[all[id as usize]]); }
    for &(id, _, _) in &shower { sim.write_bodies(id, &[all[id as usize]]); }
    for _ in 0..90 { sim.step().unwrap(); }
    for ccd in [false, true] {
        let through: Vec<(u32, [f32; 3])> = shower.iter().filter(|s| s.1 == ccd).map(|s| (s.0, sim.read_body(s.0).position)).filter(|(_, p)| p[1] < 5.9 && p[1] > -100.0).collect();
        println!("browser modes {browser}, hierarchical {hier:?}, ccd {ccd}: {} of 49 below the slab {:?}", through.len(), &through[..through.len().min(3)]);
        if ccd {
            assert!(through.is_empty(), "CCD debris fell through the slab: {through:?}");
        } else {
            assert!(through.len() > 20, "control: plain debris falls through ({})", through.len());
        }
    }
  }
}
