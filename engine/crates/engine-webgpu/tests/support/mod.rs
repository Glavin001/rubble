use engine_schema::{BodyState, Collider};
use engine_webgpu::WgpuSim;

pub fn make_body_collider(
    id: u32,
    pos: [f32; 3],
    half: [f32; 3],
    mass: f32,
    q: [f32; 4],
    sphere: bool,
) -> (BodyState, Collider) {
    let mut inv_i = [0.0; 3];
    if mass > 0.0 {
        for a in 0..3 {
            inv_i[a] = if sphere {
                1.0 / (0.4 * mass * half[0].powi(2))
            } else {
                3.0 / (mass * (half[(a + 1) % 3].powi(2) + half[(a + 2) % 3].powi(2)))
            };
        }
    }
    let body = BodyState {
        position: pos,
        orientation: q,
        inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
        inv_inertia: inv_i,
        flags: if mass > 0.0 { 9 } else { 0 },
        initial_pos: pos,
        initial_rot: q,
        linear_vel: if sphere { [0.0, 0.0, 30.0] } else { [0.0; 3] },
        angular_vel: if sphere {
            [30.0 / half[0], 0.0, 0.0]
        } else {
            [0.0; 3]
        },
        ..bytemuck::Zeroable::zeroed()
    };
    let collider = Collider {
        body: id,
        shape: if sphere { 0 } else { 2 },
        density: 1.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: if sphere {
            [half[0], 0.0, 0.0, half[0]]
        } else {
            [
                half[0],
                half[1],
                half[2],
                half.iter().map(|v| v * v).sum::<f32>().sqrt(),
            ]
        },
        ..bytemuck::Zeroable::zeroed()
    };
    (body, collider)
}

#[allow(dead_code)] // Ring uses individual insertion; gables exercises bulk upload.
pub fn add(
    sim: &mut WgpuSim,
    pos: [f32; 3],
    half: [f32; 3],
    mass: f32,
    q: [f32; 4],
    sphere: bool,
) -> u32 {
    let (body, mut collider) = make_body_collider(0, pos, half, mass, q, sphere);
    let id = sim.add_body(body);
    collider.body = id;
    sim.add_collider(collider);
    id
}
