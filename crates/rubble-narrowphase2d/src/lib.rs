use glam::Vec2;
use rubble_shapes2d::Shape2D;

#[derive(Clone, Copy, Debug)]
pub struct Contact2D {
    pub a: u32,
    pub b: u32,
    pub normal: Vec2,
    pub depth: f32,
}

pub fn contacts(pairs: &[[u32; 2]], pos: &[Vec2], shapes: &[Shape2D]) -> Vec<Contact2D> {
    let mut out = Vec::new();
    for &[a, b] in pairs {
        if let (Shape2D::Circle { radius: ra }, Shape2D::Circle { radius: rb }) =
            (&shapes[a as usize], &shapes[b as usize])
        {
            let d = pos[b as usize] - pos[a as usize];
            let dist = d.length();
            let r = ra + rb;
            if dist < r {
                out.push(Contact2D {
                    a,
                    b,
                    normal: if dist > 1e-6 { d / dist } else { Vec2::X },
                    depth: r - dist,
                });
            }
        }
    }
    out
}
