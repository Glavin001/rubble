//! The lab's camera view modes (orbit, first-person walk and fly): what each mode sees
//! and shoots along, and that switching back to orbit restores its view.

use engine_fracture::cuboid::{build_cuboid_house, CuboidOptions};
use engine_fracture::math::{dot, length, sub};
use engine_fracture::view::{Orbit, ViewMode, EYE_HEIGHT, FLY_FLOOR, SPRINT};

const ASPECT: f64 = 16.0 / 9.0;

fn camera() -> Orbit {
    Orbit::new(&build_cuboid_house(&CuboidOptions::default()).unwrap())
}

fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    length(sub(a, b)) <= tol
}

#[test]
fn orbit_looks_at_the_target() {
    let c = camera();
    let to_target = sub(c.target, c.eye(ASPECT));
    assert!(close(
        c.forward(ASPECT),
        engine_fracture::math::norm(to_target),
        1e-12
    ));
}

#[test]
fn walk_starts_on_the_floor_looking_at_the_scene_centre() {
    let mut c = camera();
    let orbit_eye = c.eye(ASPECT);
    c.set_mode(ViewMode::Walk, ASPECT);
    let eye = c.eye(ASPECT);
    assert_eq!(eye[1], EYE_HEIGHT);
    assert_eq!([eye[0], eye[2]], [orbit_eye[0], orbit_eye[2]]);
    let to_target = engine_fracture::math::norm(sub(c.target, eye));
    assert!(close(c.forward(ASPECT), to_target, 1e-9));
}

#[test]
fn centre_shot_flies_along_the_view() {
    for mode in [ViewMode::Orbit, ViewMode::Walk, ViewMode::Fly] {
        let mut c = camera();
        c.set_mode(mode, ASPECT);
        c.drag(37.0, -12.0);
        let ray = c.aim(0.5, 0.5, ASPECT, 0.25);
        assert!(close(ray.direction, c.forward(ASPECT), 1e-12), "{mode:?}");
        // Launched just in front of the eye.
        assert!((length(sub(ray.origin, c.eye(ASPECT))) - 0.5).abs() < 1e-12);
    }
}

#[test]
fn walking_stays_on_the_floor_and_diagonals_are_no_faster() {
    let mut c = camera();
    c.set_mode(ViewMode::Walk, ASPECT);
    c.mouse_look(0.0, 300.0); // look down: walking stays level
    let start = c.eye(ASPECT);
    c.travel(1.0, 1.0, 1.0, false, 0.5);
    let moved = sub(c.eye(ASPECT), start);
    assert_eq!(moved[1], 0.0);
    assert!((length(moved) - c.speed * 0.5).abs() < 1e-9);
    let start = c.eye(ASPECT);
    c.travel(-1.0, 0.0, 0.0, true, 0.25);
    assert!((length(sub(c.eye(ASPECT), start)) - c.speed * SPRINT * 0.25).abs() < 1e-9);
}

#[test]
fn flying_follows_the_look_and_stays_above_the_floor() {
    let mut c = camera();
    c.set_mode(ViewMode::Fly, ASPECT);
    let (start, forward) = (c.eye(ASPECT), c.forward(ASPECT));
    c.travel(1.0, 0.0, 0.0, false, 0.1);
    let moved = sub(c.eye(ASPECT), start);
    assert!((dot(moved, forward) - c.speed * 0.1).abs() < 1e-9);
    c.travel(0.0, 0.0, -1.0, true, 1e3);
    assert_eq!(c.eye(ASPECT)[1], FLY_FLOOR);
}

#[test]
fn mouse_look_turns_toward_the_motion() {
    let mut c = camera();
    c.set_mode(ViewMode::Fly, ASPECT);
    let right = c.aim(1.0, 0.5, ASPECT, 0.0).direction;
    let before = c.forward(ASPECT);
    c.mouse_look(10.0, 0.0);
    // The view turned toward the right edge of the screen.
    assert!(dot(c.forward(ASPECT), right) > dot(before, right));
    let below = c.aim(0.5, 1.0, ASPECT, 0.0).direction;
    let before = c.forward(ASPECT);
    c.mouse_look(0.0, 10.0);
    assert!(dot(c.forward(ASPECT), below) > dot(before, below));
}

#[test]
fn back_to_orbit_restores_the_view_and_orbit_does_not_travel() {
    let mut c = camera();
    let view = c.view_proj(ASPECT);
    c.set_mode(ViewMode::Walk, ASPECT);
    c.travel(1.0, 0.3, 0.0, false, 2.0);
    c.drag(50.0, 20.0);
    c.zoom(-200.0);
    c.set_mode(ViewMode::Orbit, ASPECT);
    assert_eq!(c.view_proj(ASPECT), view);
    c.travel(1.0, 1.0, 1.0, true, 1.0);
    assert_eq!(c.view_proj(ASPECT), view);
}

#[test]
fn walk_and_fly_hand_over_the_eye() {
    let mut c = camera();
    c.set_mode(ViewMode::Fly, ASPECT);
    c.travel(0.0, 0.0, 1.0, false, 1.0);
    let (eye, forward) = (c.eye(ASPECT), c.forward(ASPECT));
    c.set_mode(ViewMode::Walk, ASPECT);
    assert_eq!(c.eye(ASPECT), [eye[0], EYE_HEIGHT, eye[2]]);
    assert!(close(c.forward(ASPECT), forward, 1e-12));
    assert_eq!(ViewMode::from_index(ViewMode::Fly.index()), ViewMode::Fly);
    assert_eq!(ViewMode::from_index(9), ViewMode::Orbit);
}

#[test]
fn set_orbit_places_the_camera_as_a_drag_and_zoom_would() {
    let mut c = camera();
    c.set_mode(ViewMode::Fly, ASPECT);
    c.set_orbit(1.1, 0.4, c.min_distance * 2.0, Some([1.0, 2.0, 3.0]));
    assert_eq!(c.mode, ViewMode::Orbit);
    assert_eq!(
        c.orbit_state(),
        [1.1, 0.4, c.min_distance * 2.0, 1.0, 2.0, 3.0]
    );
    // Still looking at its target, from that distance.
    assert!(close(
        c.forward(ASPECT),
        engine_fracture::math::norm(sub(c.target, c.eye(ASPECT))),
        1e-12
    ));
    assert!((length(sub(c.target, c.eye(ASPECT))) - c.min_distance * 2.0).abs() < 1e-9);
    // Pitch keeps the drag's range and distance the zoom's; no target keeps the target.
    c.set_orbit(0.0, 9.0, 0.0, None);
    let [_, pitch, distance, x, y, z] = c.orbit_state();
    assert_eq!(
        (pitch, distance, [x, y, z]),
        (1.3, c.min_distance, [1.0, 2.0, 3.0])
    );
    c.set_orbit(0.0, -9.0, 1e9, None);
    assert_eq!(c.orbit_state()[1..3], [-0.1, c.extent * 8.0]);
}
