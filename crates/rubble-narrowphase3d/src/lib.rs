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
            sphere_sphere(a, b, pa, pb, *ra, *rb)
        }
        (ShapeDesc::Sphere { radius }, ShapeDesc::Box { half_extents }) => {
            sphere_box(a, b, pa, pb, *radius, *half_extents)
        }
        (ShapeDesc::Box { half_extents }, ShapeDesc::Sphere { radius }) => {
            sphere_box(b, a, pb, pa, *radius, *half_extents).map(|mut c| {
                c.body_a = a;
                c.body_b = b;
                c.normal *= -1.0;
                c
            })
        }
        (ShapeDesc::Box { half_extents: ha }, ShapeDesc::Box { half_extents: hb }) => {
            box_box(a, b, pa, pb, *ha, *hb)
        }
    }
}

fn sphere_sphere(a: u32, b: u32, pa: Vec3, pb: Vec3, ra: f32, rb: f32) -> Option<Contact3D> {
    let d = pb - pa;
    let dist = d.length();
    let r = ra + rb;
    if dist >= r {
        return None;
    }
    let n = if dist > 1e-6 { d / dist } else { Vec3::X };
    Some(Contact3D {
        point: (pa + n * ra).extend(r - dist),
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

fn sphere_box(
    sphere_id: u32,
    box_id: u32,
    sphere_pos: Vec3,
    box_pos: Vec3,
    radius: f32,
    half_extents: Vec3,
) -> Option<Contact3D> {
    let local = sphere_pos - box_pos;
    let clamped = local.clamp(-half_extents, half_extents);
    let closest = box_pos + clamped;
    let d = sphere_pos - closest;
    let dist = d.length();
    if dist >= radius {
        return None;
    }
    let n = if dist > 1e-6 {
        d / dist
    } else {
        (sphere_pos - box_pos).normalize_or_zero()
    };
    let depth = radius - dist;
    Some(Contact3D {
        point: closest.extend(depth),
        normal: n.extend(0.0),
        body_a: sphere_id,
        body_b: box_id,
        feature_id: 1,
        pad: 0,
        lambda_n: 0.0,
        lambda_t1: 0.0,
        lambda_t2: 0.0,
        penalty_k: 1e4,
    })
}

fn box_box(a: u32, b: u32, pa: Vec3, pb: Vec3, ha: Vec3, hb: Vec3) -> Option<Contact3D> {
    let delta = pb - pa;
    let overlap = ha + hb - delta.abs();
    if overlap.x <= 0.0 || overlap.y <= 0.0 || overlap.z <= 0.0 {
        return None;
    }

    let (depth, normal) = if overlap.x <= overlap.y && overlap.x <= overlap.z {
        (overlap.x, Vec3::new(delta.x.signum(), 0.0, 0.0))
    } else if overlap.y <= overlap.z {
        (overlap.y, Vec3::new(0.0, delta.y.signum(), 0.0))
    } else {
        (overlap.z, Vec3::new(0.0, 0.0, delta.z.signum()))
    };

    Some(Contact3D {
        point: ((pa + pb) * 0.5).extend(depth),
        normal: normal.extend(0.0),
        body_a: a,
        body_b: b,
        feature_id: 2,
        pad: 0,
        lambda_n: 0.0,
        lambda_t1: 0.0,
        lambda_t2: 0.0,
        penalty_k: 1e4,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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

    #[test]
    fn box_box_overlap_produces_contact_depth() {
        let c = collide_pair(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(1.9, 0.0, 0.0),
            &ShapeDesc::Box {
                half_extents: Vec3::ONE,
            },
            &ShapeDesc::Box {
                half_extents: Vec3::ONE,
            },
        )
        .expect("expected overlap");
        assert_relative_eq!(c.point.w, 0.1, epsilon = 1e-6);
        assert_eq!(c.normal.truncate(), Vec3::X);
    }
}
