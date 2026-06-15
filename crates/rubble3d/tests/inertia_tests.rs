//! Inertia-tensor cross-validation.
//!
//! The engine builds each body's inverse inertia in `compute_inertia`. The
//! test-side metrics also need inertia, and `tests/support/mod.rs` keeps its own
//! duplicate formula — so a divergence between them would silently corrupt every
//! energy / angular-momentum oracle. Here we validate the engine's *actual*
//! tensor (via `World::get_inv_inertia`) against an **independent** oracle:
//! parry3d's `mass_properties`. This is exact analytic ground truth, not a
//! duplicated formula, and it runs instantly (no stepping).

use glam::Vec3;
use nalgebra::{Point3, Vector3};
use parry3d::shape::{Ball, Capsule, Cuboid, SharedShape};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

/// Sorted principal moments of the engine's inertia tensor for a body of `mass`
/// with `shape`. The engine stores the *inverse* inertia, diagonal in the body
/// frame, so we invert and read the diagonal.
fn engine_principal_inertia(shape: ShapeDesc, mass: f32) -> Option<[f64; 3]> {
    let mut world = World::new(SimConfig {
        max_bodies: 8,
        ..Default::default()
    })
    .ok()?;
    let h = world.add_body(&RigidBodyDesc {
        mass,
        shape,
        ..Default::default()
    });
    let inv = world.get_inv_inertia(h)?;
    let i = inv.inverse();
    let cols = i.to_cols_array_2d();
    let mut d = [cols[0][0] as f64, cols[1][1] as f64, cols[2][2] as f64];
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(d)
}

/// Sorted principal moments from parry3d's independent mass-properties, scaled to
/// the target mass (inertia ∝ mass at fixed geometry).
fn parry_principal_inertia(shape: &SharedShape, mass: f32) -> [f64; 3] {
    let mp = shape.mass_properties(1.0);
    let scale = mass as f64 / mp.mass() as f64;
    let m = mp.reconstruct_inertia_matrix();
    let mut d = [
        m[(0, 0)] as f64 * scale,
        m[(1, 1)] as f64 * scale,
        m[(2, 2)] as f64 * scale,
    ];
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d
}

fn rel_diff(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-9))
        .fold(0.0, f64::max)
}

#[test]
fn inertia_matches_parry_for_primitive_shapes() {
    let mass = 2.5;
    let cases: Vec<(&str, ShapeDesc, SharedShape)> = vec![
        (
            "sphere",
            ShapeDesc::Sphere { radius: 0.5 },
            SharedShape::new(Ball::new(0.5)),
        ),
        (
            "box",
            ShapeDesc::Box {
                half_extents: Vec3::new(0.5, 0.4, 0.6),
            },
            SharedShape::new(Cuboid::new(Vector3::new(0.5, 0.4, 0.6))),
        ),
        (
            "capsule",
            ShapeDesc::Capsule {
                half_height: 0.6,
                radius: 0.3,
            },
            SharedShape::new(Capsule::new(
                Point3::new(0.0, -0.6, 0.0),
                Point3::new(0.0, 0.6, 0.0),
                0.3,
            )),
        ),
    ];

    let mut worst = 0.0f64;
    let mut failures = Vec::new();
    let mut ran = false;
    for (name, shape, parry) in cases {
        let Some(engine) = engine_principal_inertia(shape, mass) else {
            eprintln!("SKIP: no GPU adapter");
            return;
        };
        ran = true;
        let expected = parry_principal_inertia(&parry, mass);
        let rd = rel_diff(engine, expected);
        worst = worst.max(rd);
        println!("{name}: engine={engine:?} parry={expected:?} rel_diff={rd:.2e}");
        // Derived bound: both are analytic; the only spread is f32 evaluation of
        // two algebraically-equal formulas. 1e-3 relative is generous on that.
        if rd > 1.0e-3 {
            failures.push(format!("{name}: rel_diff={rd:.3e}"));
        }
    }
    assert!(ran, "no shapes evaluated");
    assert!(
        failures.is_empty(),
        "engine inertia disagrees with parry (independent oracle): {failures:?}"
    );
    println!("worst relative inertia error: {worst:.2e}");
}

#[test]
fn convex_hull_inertia_uses_bbox_approximation() {
    // The engine approximates convex-hull inertia with its bounding box
    // (see compute_inertia). This test documents that *known approximation* by
    // showing it diverges from parry's exact convex inertia — so the divergence
    // is expected and explained, not a silent surprise in energy/momentum checks.
    let mass = 2.0;
    // An octahedron-ish hull whose true inertia differs from its bbox.
    let verts = vec![
        Vec3::new(0.6, 0.0, 0.0),
        Vec3::new(-0.6, 0.0, 0.0),
        Vec3::new(0.0, 0.6, 0.0),
        Vec3::new(0.0, -0.6, 0.0),
        Vec3::new(0.0, 0.0, 0.6),
        Vec3::new(0.0, 0.0, -0.6),
    ];
    let Some(engine) = engine_principal_inertia(
        ShapeDesc::ConvexHull {
            vertices: verts.clone(),
        },
        mass,
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let points: Vec<Point3<f32>> = verts.iter().map(|v| Point3::new(v.x, v.y, v.z)).collect();
    let parry = SharedShape::convex_hull(&points).expect("convex hull");
    let exact = parry_principal_inertia(&parry, mass);
    let rd = rel_diff(engine, exact);
    println!("convex hull: engine(bbox)={engine:?} parry(exact)={exact:?} rel_diff={rd:.2e}");
    // We assert the approximation is *present* (engine over-estimates inertia
    // because the bbox has more mass far from the axis). If this ever becomes
    // exact, the engine started computing real convex inertia — update the note.
    assert!(
        rd > 0.05,
        "convex-hull inertia now matches parry ({rd:.3e}); engine may compute exact \
         convex inertia now — revisit the bbox-approximation note"
    );
}
