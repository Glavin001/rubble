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

    let columns = 40;
    let rows = 25;
    let x_start = 1.25_f32;
    let y_start = 2.0_f32;
    let x_spacing = 27.5_f32 / (columns - 1) as f32;
    let y_spacing = 16.0_f32 / (rows - 1) as f32;

    let mut rng = rand::rng();
    for row in 0..rows {
        for col in 0..columns {
            let x = x_start + col as f32 * x_spacing + (rng.random::<f32>() - 0.5) * 0.05;
            let y = y_start + row as f32 * y_spacing + (rng.random::<f32>() - 0.5) * 0.05;

            match rng.random_range(0..3u32) {
                0 => viewer.add_circle(x, y, 0.12 + rng.random::<f32>() * 0.08),
                1 => {
                    let hw = 0.12 + rng.random::<f32>() * 0.08;
                    let hh = 0.12 + rng.random::<f32>() * 0.08;
                    let angle = (rng.random::<f32>() - 0.5) * 0.35;
                    viewer.add_rect(x, y, hw, hh, angle, 1.0);
                }
                _ => {
                    let half_height = 0.10 + rng.random::<f32>() * 0.06;
                    let radius = 0.06 + rng.random::<f32>() * 0.04;
                    viewer.add_capsule(x, y, half_height, radius);
                }
            }
        }
    }

    viewer.run();
}
