//! Shooting on the rolling mountain (CPU reference): a click's ray from a camera behind
//! and above the falling car (led by its travel) picks the car (or the mountain's rock beside it), the shot
//! flies (the canyon has no ceiling) and strikes the car; a ray at the slope
//! picks the mountain, not the air.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::DVec3;
use rubble::vehicle_lab::cpu::CpuLab;
use rubble::vehicle_lab::LabOptions;
use serde_json::{json, Value};

fn car() -> ExplicitBeams {
    let text = engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
        minimass: 10.0,
        ..engine_softbody::procedural::CarParams::sedan()
    });
    ExplicitBeams::from_truck_raycast(
        &text,
        None,
        glam::Vec3::ZERO,
        0.0,
        glam::Vec3::ZERO,
        ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            ..Default::default()
        },
    )
    .unwrap()
    .0
}

fn arr(v: &Value) -> DVec3 {
    let a = v.as_array().unwrap();
    DVec3::new(
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    )
}

#[test]
fn a_shot_on_the_rolling_mountain_picks_the_car_and_strikes_it() {
    let opts = LabOptions::from_json(&json!({
        "scene": "mountain",
        "surface": true,
        "mountain": { "terrain": "rolling", "rockslide": false },
    }));
    let mut lab = CpuLab::new(&car(), opts).unwrap();
    for _ in 0..480 {
        lab.step().unwrap();
    }
    let c = lab.bench.car.centre;
    // A camera 30 m up the fall line and 25 m over it: above the drawn walls (13 m), as
    // a camera zoomed out would be. A ray at the car picks the car.
    let th = 50f64.to_radians();
    let (up, normal) = (
        DVec3::new(0.0, th.sin(), -th.cos()),
        DVec3::new(0.0, th.cos(), th.sin()),
    );
    let eye = c + up * 30.0 + normal * 25.0;
    let d = (c - eye).normalize();
    let r = lab
        .command(
            &json!({ "fire": { "shape": "sphere", "radius": 0.1, "speed": 60,
            "pick": { "origin": [eye.x, eye.y, eye.z], "dir": [d.x, d.y, d.z] } } }),
        )
        .unwrap();
    assert_eq!(r["pick"]["on"], "car", "{r}");
    assert!((arr(&r["pick"]["target"]) - c).length() < 3.0, "{r}");
    // A ray at the slope well beside the car meets the mountain's rock.
    let beside = c + DVec3::new(9.0, 0.0, 0.0) - up * 8.0;
    let d = (beside - eye).normalize();
    let r = lab
        .command(
            &json!({ "fire": { "shape": "sphere", "radius": 0.1, "speed": 60,
            "pick": { "origin": [eye.x, eye.y, eye.z], "dir": [d.x, d.y, d.z] } } }),
        )
        .unwrap();
    assert_eq!(r["pick"]["on"], "mountain", "{r}");
    let t = arr(&r["pick"]["target"]);
    assert!(
        (t - eye).dot(normal) < -20.0,
        "the rock is below the walls' top: {r}"
    );

    // A steel ball fired after the car along its own path, from 6 m behind, strikes it.
    lab.command(&json!({ "clear": "shots" })).unwrap();
    let c = lab.bench.car.centre;
    let p = lab.bench.pose();
    let v = DVec3::new(p[3], p[4], p[5]);
    let dir = v.normalize();
    let from = c - dir * 6.0;
    lab.command(&json!({ "fire": { "shape": "sphere", "radius": 0.15, "material": "steel",
        "speed": v.length() + 30.0, "origin": [from.x, from.y, from.z], "dir": [dir.x, dir.y, dir.z] } }))
        .unwrap();
    for _ in 0..60 {
        lab.step().unwrap();
    }
    let s = lab.bench.stats();
    let shot = s["shots"].as_array().unwrap().last().unwrap().clone();
    assert!(
        shot["impactSpeed"].is_number(),
        "the shot never touched the car: {shot}"
    );
}
