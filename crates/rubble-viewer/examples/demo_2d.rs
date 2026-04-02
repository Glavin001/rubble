use rand::Rng;
use rubble_viewer::Viewer2D;

fn main() {
    let mut viewer = Viewer2D::new(0.0, -9.81);

    let world_w = 30.0_f32;
    let world_h = 20.0_f32;
    let wall = 0.25;

    // Ground
    viewer.add_static_rect(world_w / 2.0, wall / 2.0, world_w / 2.0, wall / 2.0, 0.0);
    // Left wall
    viewer.add_static_rect(wall / 2.0, world_h / 2.0, wall / 2.0, world_h / 2.0, 0.0);
    // Right wall
    viewer.add_static_rect(
        world_w - wall / 2.0,
        world_h / 2.0,
        wall / 2.0,
        world_h / 2.0,
        0.0,
    );

    let mut rng = rand::rng();
    for _ in 0..50 {
        let x = 3.0 + rng.random::<f32>() * (world_w - 6.0);
        let y = 5.0 + rng.random::<f32>() * 15.0;

        match rng.random_range(0..3u32) {
            0 => viewer.add_circle(x, y, 0.3 + rng.random::<f32>() * 0.5),
            1 => {
                let hw = 0.3 + rng.random::<f32>() * 0.4;
                let hh = 0.3 + rng.random::<f32>() * 0.4;
                viewer.add_rect(x, y, hw, hh, 0.0, 1.0);
            }
            _ => viewer.add_capsule(x, y, 0.4, 0.25),
        }
    }

    viewer.run();
}
