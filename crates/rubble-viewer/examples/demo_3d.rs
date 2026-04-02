use rand::Rng;
use rubble_viewer::Viewer3D;

fn main() {
    let mut viewer = Viewer3D::new(0.0, -9.81, 0.0);
    viewer.add_ground_plane(0.0);

    let mut rng = rand::rng();
    for _ in 0..100 {
        let x = (rng.random::<f32>() - 0.5) * 12.0;
        let y = 3.0 + rng.random::<f32>() * 15.0;
        let z = (rng.random::<f32>() - 0.5) * 12.0;

        match rng.random_range(0..3u32) {
            0 => viewer.add_sphere(x, y, z, 0.3 + rng.random::<f32>() * 0.4),
            1 => {
                let s = 0.2 + rng.random::<f32>() * 0.4;
                viewer.add_box(x, y, z, s, s, s);
            }
            _ => viewer.add_capsule(x, y, z, 0.3, 0.2),
        }
    }

    viewer.run();
}
