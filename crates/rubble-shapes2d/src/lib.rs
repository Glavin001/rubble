use glam::Vec2;

#[derive(Clone, Debug)]
pub enum Shape2D {
    Circle { radius: f32 },
    Rect { half_extents: Vec2 },
}

#[derive(Clone, Copy, Debug)]
pub struct Aabb2D {
    pub min: Vec2,
    pub max: Vec2,
}

pub fn compute_aabb(shape: &Shape2D, pos: Vec2) -> Aabb2D {
    match shape {
        Shape2D::Circle { radius } => Aabb2D {
            min: pos - Vec2::splat(*radius),
            max: pos + Vec2::splat(*radius),
        },
        Shape2D::Rect { half_extents } => Aabb2D {
            min: pos - *half_extents,
            max: pos + *half_extents,
        },
    }
}
