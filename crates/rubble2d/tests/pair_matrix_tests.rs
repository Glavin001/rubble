mod support;

use glam::{Vec2, Vec4};
use rubble2d::{
    gpu::{GpuPipeline2D, ShapeInfo},
    ShapeDesc2D,
};
use rubble_gpu::{GpuBuffer, GpuContext};
use rubble_math::{Aabb2D, Contact2D, RigidBodyState2D};
use rubble_primitives::GpuLbvh;
use rubble_shapes2d::{CapsuleData2D, CircleData, ConvexPolygonData, ConvexVertex2D, RectData};
use support::{regular_polygon, should_skip_known_failure};

const SHAPE_CIRCLE_2D: u32 = 0;
const SHAPE_RECT_2D: u32 = 1;
const SHAPE_POLYGON_2D: u32 = 2;
const SHAPE_CAPSULE_2D: u32 = 3;
const INITIAL_PENALTY: f32 = 1.0e4;

#[derive(Clone)]
struct BodySpec2D {
    shape: ShapeDesc2D,
    position: Vec2,
    angle: f32,
    linear_velocity: Vec2,
    angular_velocity: f32,
    mass: f32,
    friction: f32,
}

struct PairCase2D {
    name: &'static str,
    a: BodySpec2D,
    b: BodySpec2D,
    expect_contact: bool,
}

impl PairCase2D {
    fn known_failure_reason(&self) -> Option<&'static str> {
        match self.name {
            "capsule-capsule-contact" => {
                Some("2D capsule-capsule narrowphase is currently missing this overlap")
            }
            "polygon-capsule-contact" => {
                Some("2D polygon-capsule narrowphase is currently missing this overlap")
            }
            _ => None,
        }
    }
}

fn inverse_inertia(shape: &ShapeDesc2D, mass: f32) -> f32 {
    if mass <= 0.0 {
        return 0.0;
    }

    let inertia = match shape {
        ShapeDesc2D::Circle { radius } => 0.5 * mass * radius * radius,
        ShapeDesc2D::Rect { half_extents } => {
            let w = 2.0 * half_extents.x;
            let h = 2.0 * half_extents.y;
            mass * (w * w + h * h) / 12.0
        }
        ShapeDesc2D::Capsule {
            half_height,
            radius,
        } => {
            let w = 2.0 * radius;
            let h = 2.0 * (half_height + radius);
            mass * (w * w + h * h) / 12.0
        }
        ShapeDesc2D::ConvexPolygon { vertices } => {
            let mut min = Vec2::splat(f32::MAX);
            let mut max = Vec2::splat(f32::NEG_INFINITY);
            for &vertex in vertices {
                min = min.min(vertex);
                max = max.max(vertex);
            }
            let size = max - min;
            mass * (size.x * size.x + size.y * size.y) / 12.0
        }
    };

    if inertia <= 1.0e-12 {
        0.0
    } else {
        1.0 / inertia
    }
}

fn body_state(spec: &BodySpec2D) -> RigidBodyState2D {
    let inv_mass = if spec.mass > 0.0 {
        1.0 / spec.mass
    } else {
        0.0
    };
    let mut state = RigidBodyState2D::new(
        spec.position.x,
        spec.position.y,
        spec.angle,
        inv_mass,
        spec.linear_velocity.x,
        spec.linear_velocity.y,
        spec.angular_velocity,
    );
    state._pad0 = Vec4::new(
        spec.friction,
        inverse_inertia(&spec.shape, spec.mass),
        0.0,
        0.0,
    );
    state
}

fn run_pair_case(case: &PairCase2D) -> Option<(Vec<RigidBodyState2D>, Vec<Contact2D>)> {
    let mut pipeline = GpuPipeline2D::try_new(2)?;
    let states = vec![body_state(&case.a), body_state(&case.b)];
    let mut shape_infos = Vec::with_capacity(2);
    let mut circles = Vec::new();
    let mut rects = Vec::new();
    let mut polygons = Vec::new();
    let mut polygon_vertices = Vec::new();
    let mut capsules = Vec::new();

    for shape in [&case.a.shape, &case.b.shape] {
        match shape {
            ShapeDesc2D::Circle { radius } => {
                shape_infos.push(ShapeInfo {
                    shape_type: SHAPE_CIRCLE_2D,
                    shape_index: circles.len() as u32,
                });
                circles.push(CircleData {
                    radius: *radius,
                    _pad: [0.0; 3],
                });
            }
            ShapeDesc2D::Rect { half_extents } => {
                shape_infos.push(ShapeInfo {
                    shape_type: SHAPE_RECT_2D,
                    shape_index: rects.len() as u32,
                });
                rects.push(RectData {
                    half_extents: half_extents.extend(0.0).extend(0.0),
                });
            }
            ShapeDesc2D::ConvexPolygon { vertices } => {
                shape_infos.push(ShapeInfo {
                    shape_type: SHAPE_POLYGON_2D,
                    shape_index: polygons.len() as u32,
                });
                let vertex_offset = polygon_vertices.len() as u32;
                for &vertex in vertices.iter().take(64) {
                    polygon_vertices.push(ConvexVertex2D {
                        x: vertex.x,
                        y: vertex.y,
                        _pad: [0.0; 2],
                    });
                }
                polygons.push(ConvexPolygonData {
                    vertex_offset,
                    vertex_count: vertices.len().min(64) as u32,
                    _pad: [0; 2],
                });
            }
            ShapeDesc2D::Capsule {
                half_height,
                radius,
            } => {
                shape_infos.push(ShapeInfo {
                    shape_type: SHAPE_CAPSULE_2D,
                    shape_index: capsules.len() as u32,
                });
                capsules.push(CapsuleData2D {
                    half_height: *half_height,
                    radius: *radius,
                    _pad: [0.0; 2],
                });
            }
        }
    }

    pipeline.upload(
        &states,
        &states,
        &shape_infos,
        &circles,
        &rects,
        &polygons,
        &polygon_vertices,
        &capsules,
        Vec2::ZERO,
        1.0 / 120.0,
        10,
        10.0,
        INITIAL_PENALTY,
        0.95,
    );
    Some(pipeline.step_with_contacts(2, 10, None))
}

fn contact_sanity_errors(case: &PairCase2D, contacts: &[Contact2D]) -> Vec<String> {
    let mut errors = Vec::new();
    for contact in contacts {
        let normal = contact.contact_normal();
        let tangent = contact.contact_tangent();
        if !(contact.point.is_finite() && contact.normal.is_finite()) {
            errors.push(format!(
                "{}: contact contains non-finite values: {:?}",
                case.name, contact
            ));
        }
        if (normal.length() - 1.0).abs() >= 1.0e-3 {
            errors.push(format!(
                "{}: contact normal is not unit length: {:?}",
                case.name, contact
            ));
        }
        if (tangent.length() - 1.0).abs() >= 1.0e-3 {
            errors.push(format!(
                "{}: contact tangent is not unit length: {:?}",
                case.name, contact
            ));
        }
        if normal.dot(tangent).abs() >= 1.0e-3 {
            errors.push(format!(
                "{}: contact normal and tangent are not orthogonal: {:?}",
                case.name, contact
            ));
        }
        if !(contact.depth().is_finite() && contact.depth().abs() < 5.0) {
            errors.push(format!(
                "{}: contact depth is out of range: {:?}",
                case.name, contact
            ));
        }
        if contact.feature_id == 0 {
            errors.push(format!(
                "{}: contact feature id should be non-zero: {:?}",
                case.name, contact
            ));
        }
        if contact.body_a == contact.body_b {
            errors.push(format!(
                "{}: contact references the same body twice: {:?}",
                case.name, contact
            ));
        }
    }
    errors
}

fn pair_cases_2d() -> Vec<PairCase2D> {
    let polygon = ShapeDesc2D::ConvexPolygon {
        vertices: regular_polygon(0.7, 6),
    };

    vec![
        PairCase2D {
            name: "circle-circle-contact",
            a: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 1.0 },
                position: Vec2::new(-0.8, 0.0),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 1.0 },
                position: Vec2::new(0.8, 0.0),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "circle-circle-miss",
            a: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 1.0 },
                position: Vec2::new(-1.2, 0.0),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 1.0 },
                position: Vec2::new(1.2, 0.0),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            expect_contact: false,
        },
        PairCase2D {
            name: "circle-rect-contact",
            a: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 0.8 },
                position: Vec2::new(-0.7, 0.0),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.3,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(0.9, 0.6),
                },
                position: Vec2::new(0.7, 0.0),
                angle: 0.1,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.3,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "circle-rect-miss",
            a: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 0.8 },
                position: Vec2::new(-1.5, 0.0),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.3,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(0.9, 0.6),
                },
                position: Vec2::new(1.5, 0.0),
                angle: 0.1,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.3,
            },
            expect_contact: false,
        },
        PairCase2D {
            name: "rect-rect-contact",
            a: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(1.0, 0.5),
                },
                position: Vec2::new(-0.75, 0.0),
                angle: 0.2,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.6,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(0.9, 0.6),
                },
                position: Vec2::new(0.75, 0.0),
                angle: -0.3,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.6,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "rect-rect-miss",
            a: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(1.0, 0.5),
                },
                position: Vec2::new(-1.8, 0.0),
                angle: 0.2,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.6,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(0.9, 0.6),
                },
                position: Vec2::new(1.8, 0.0),
                angle: -0.3,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.6,
            },
            expect_contact: false,
        },
        PairCase2D {
            name: "circle-capsule-contact",
            a: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 0.7 },
                position: Vec2::new(-0.6, 0.15),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.7,
                    radius: 0.25,
                },
                position: Vec2::new(0.55, 0.0),
                angle: 0.5,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "circle-capsule-miss",
            a: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 0.7 },
                position: Vec2::new(-1.5, 0.15),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.7,
                    radius: 0.25,
                },
                position: Vec2::new(1.5, 0.0),
                angle: 0.5,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: false,
        },
        PairCase2D {
            name: "rect-capsule-contact",
            a: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(0.9, 0.45),
                },
                position: Vec2::new(-0.65, 0.0),
                angle: 0.15,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.65,
                    radius: 0.3,
                },
                position: Vec2::new(0.55, 0.1),
                angle: -0.35,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "rect-capsule-miss",
            a: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(0.9, 0.45),
                },
                position: Vec2::new(-1.7, 0.0),
                angle: 0.15,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.65,
                    radius: 0.3,
                },
                position: Vec2::new(1.7, 0.1),
                angle: -0.35,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: false,
        },
        PairCase2D {
            name: "capsule-capsule-contact",
            a: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.7,
                    radius: 0.25,
                },
                position: Vec2::new(-0.8, 0.1),
                angle: 0.4,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.6,
                    radius: 0.3,
                },
                position: Vec2::new(0.8, -0.1),
                angle: -0.5,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "capsule-capsule-miss",
            a: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.7,
                    radius: 0.25,
                },
                position: Vec2::new(-1.8, 0.1),
                angle: 0.4,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.6,
                    radius: 0.3,
                },
                position: Vec2::new(1.8, -0.1),
                angle: -0.5,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            expect_contact: false,
        },
        PairCase2D {
            name: "polygon-polygon-contact",
            a: BodySpec2D {
                shape: polygon.clone(),
                position: Vec2::new(-0.65, 0.0),
                angle: 0.3,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: polygon.clone(),
                position: Vec2::new(0.65, 0.0),
                angle: -0.2,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "polygon-polygon-miss",
            a: BodySpec2D {
                shape: polygon.clone(),
                position: Vec2::new(-1.7, 0.0),
                angle: 0.3,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: polygon.clone(),
                position: Vec2::new(1.7, 0.0),
                angle: -0.2,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: false,
        },
        PairCase2D {
            name: "polygon-circle-contact",
            a: BodySpec2D {
                shape: polygon.clone(),
                position: Vec2::new(-0.7, 0.0),
                angle: 0.2,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 0.75 },
                position: Vec2::new(0.7, 0.0),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "polygon-circle-miss",
            a: BodySpec2D {
                shape: polygon.clone(),
                position: Vec2::new(-1.5, 0.0),
                angle: 0.2,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Circle { radius: 0.75 },
                position: Vec2::new(1.5, 0.0),
                angle: 0.0,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: false,
        },
        PairCase2D {
            name: "polygon-rect-contact",
            a: BodySpec2D {
                shape: polygon.clone(),
                position: Vec2::new(-0.7, 0.0),
                angle: 0.25,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(0.8, 0.55),
                },
                position: Vec2::new(0.7, 0.05),
                angle: -0.35,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "polygon-rect-miss",
            a: BodySpec2D {
                shape: polygon.clone(),
                position: Vec2::new(-1.7, 0.0),
                angle: 0.25,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::new(0.8, 0.55),
                },
                position: Vec2::new(1.7, 0.05),
                angle: -0.35,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.5,
            },
            expect_contact: false,
        },
        PairCase2D {
            name: "polygon-capsule-contact",
            a: BodySpec2D {
                shape: polygon.clone(),
                position: Vec2::new(-0.7, 0.0),
                angle: 0.15,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.65,
                    radius: 0.25,
                },
                position: Vec2::new(0.7, 0.0),
                angle: -0.4,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            expect_contact: true,
        },
        PairCase2D {
            name: "polygon-capsule-miss",
            a: BodySpec2D {
                shape: polygon,
                position: Vec2::new(-1.7, 0.0),
                angle: 0.15,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            b: BodySpec2D {
                shape: ShapeDesc2D::Capsule {
                    half_height: 0.65,
                    radius: 0.25,
                },
                position: Vec2::new(1.7, 0.0),
                angle: -0.4,
                linear_velocity: Vec2::ZERO,
                angular_velocity: 0.0,
                mass: 1.0,
                friction: 0.4,
            },
            expect_contact: false,
        },
    ]
}

#[test]
fn pair_matrix_contacts_match_geometry_2d() {
    let mut failures = Vec::new();
    for case in pair_cases_2d() {
        if let Some(reason) = case.known_failure_reason() {
            if should_skip_known_failure(case.name, reason) {
                continue;
            }
        }
        let Some((states, contacts)) = run_pair_case(&case) else {
            eprintln!("SKIP: No GPU adapter found");
            return;
        };

        if !states.iter().all(|state| {
            state.position().is_finite()
                && state.linear_velocity().is_finite()
                && state.angle().is_finite()
                && state.angular_velocity().is_finite()
        }) {
            failures.push(format!(
                "{}: pair solve produced non-finite state: {:?}",
                case.name, states
            ));
        }

        if case.expect_contact {
            if contacts.is_empty() {
                failures.push(format!("{}: expected contact(s) but got none", case.name));
            } else {
                failures.extend(contact_sanity_errors(&case, &contacts));
            }
        } else if !contacts.is_empty() {
            failures.push(format!(
                "{}: expected no contacts, got {:?}",
                case.name, contacts
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn broadphase_matrix_culls_and_keeps_expected_pairs_2d() {
    let Some(ctx) = pollster::block_on(GpuContext::new()).ok() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let mut lbvh = GpuLbvh::new(&ctx, 8);
    let mut buffer = GpuBuffer::new(&ctx, 4);
    let aabbs = vec![
        Aabb2D::new(Vec2::new(-2.0, -1.0), Vec2::new(-0.5, 1.0)),
        Aabb2D::new(Vec2::new(-1.0, -1.0), Vec2::new(0.75, 1.0)),
        Aabb2D::new(Vec2::new(0.7, -1.0), Vec2::new(2.0, 1.0)),
        Aabb2D::new(Vec2::new(4.0, -1.0), Vec2::new(5.0, 1.0)),
    ];
    buffer.upload(&ctx, &aabbs);

    let pairs = lbvh.build_and_query_raw(&ctx, buffer.buffer(), aabbs.len() as u32);
    assert_eq!(
        pairs,
        vec![[0, 1], [1, 2]],
        "unexpected 2D broadphase overlaps: {pairs:?}"
    );
}
