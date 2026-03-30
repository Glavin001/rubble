use glam::Vec3;
use rubble_math::Contact3D;
use rubble_shapes3d::ShapeDesc;

pub fn generate_contacts(
    pairs: &[[u32; 2]],
    positions: &[Vec3],
    shapes: &[ShapeDesc],
) -> Vec<Contact3D> {
    let mut contacts = Vec::new();
    for &[a, b] in pairs {
        if let Some(c) = collide_pair(
            a,
            b,
            positions[a as usize],
            positions[b as usize],
            &shapes[a as usize],
            &shapes[b as usize],
        ) {
            contacts.push(c);
        }
    }
    contacts
}

pub fn collide_pair(
    a: u32,
    b: u32,
    pa: Vec3,
    pb: Vec3,
    sa: &ShapeDesc,
    sb: &ShapeDesc,
) -> Option<Contact3D> {
    match (sa, sb) {
        (ShapeDesc::Sphere { radius: ra }, ShapeDesc::Sphere { radius: rb }) => {
            let d = pb - pa;
            let dist = d.length();
            let r = ra + rb;
            if dist >= r {
                return None;
            }
            let n = if dist > 1e-6 { d / dist } else { Vec3::X };
            Some(Contact3D {
                point: (pa + n * *ra).extend(r - dist),
                normal: n.extend(0.0),
                body_a: a,
                body_b: b,
                feature_id: 0,
                pad: 0,
                lambda_n: 0.0,
                lambda_t1: 0.0,
                lambda_t2: 0.0,
                penalty_k: 1e4,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_sphere_contact_expected() {
        let c = collide_pair(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(1.5, 0.0, 0.0),
            &ShapeDesc::Sphere { radius: 1.0 },
            &ShapeDesc::Sphere { radius: 1.0 },
        )
        .expect("expected contact");
        assert!((c.point.w - 0.5).abs() < 1e-6);
        assert_eq!(c.normal.truncate(), Vec3::X);
    }
}
