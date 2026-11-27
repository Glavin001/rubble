//! Synthetic shapes with known answers, and the committed structure schemas.

use super::measures::{chord, creases, fold_height, MeasureDef};
use super::rubble::{derive, envelope, hull, to_instance, BodyView};
use super::*;
use serde_json::json;

fn schema(v: Value) -> Schema {
    serde_json::from_value(v).unwrap()
}

fn line(name: &str, pts: &[P]) -> Shape {
    let mut s = Shape::default();
    s.lines.insert(name.into(), pts.to_vec());
    s
}

fn measure(v: Value) -> MeasureDef {
    serde_json::from_value(v).unwrap()
}

#[test]
fn a_tent_gives_its_fold_height_and_chord() {
    // A flat hood of length 1 folds into a tent 0.2 high over a 0.8 chord.
    let before = [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]];
    let after = [[0.0, 0.0], [0.4, 0.2], [0.8, 0.0]];
    assert!((fold_height(&after, chord(&before)) - 0.2).abs() < 1e-12);
    // A dent (below the chord) is negative.
    assert!((fold_height(&[[0.0, 0.0], [0.4, -0.1], [0.8, 0.0]], 1.0) + 0.1).abs() < 1e-12);
    let (b, a) = (line("hood", &before), line("hood", &after));
    let m = measure(json!({ "id": "f", "kind": "fold_height", "line": "hood" }));
    assert!((m.eval(&b, &a).unwrap().v - 0.2).abs() < 1e-12);
    let m = measure(json!({ "id": "c", "kind": "chord_ratio", "line": "hood" }));
    assert!((m.eval(&b, &a).unwrap().v - 0.8).abs() < 1e-12);
}

#[test]
fn creases_count_sharp_turns_and_their_sign_changes() {
    // One peak: a single turn of about -53 degrees.
    assert_eq!(creases(&[[0.0, 0.0], [0.4, 0.2], [0.8, 0.0]], 20.0), (1, 0));
    // A zigzag: three sharp turns alternating in sign.
    let z = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]];
    assert_eq!(creases(&z, 20.0), (3, 2));
    // A gentle bow has none.
    assert_eq!(
        creases(&[[0.0, 0.0], [1.0, 0.05], [2.0, 0.08], [3.0, 0.05]], 20.0),
        (0, 0)
    );
}

#[test]
fn angles_tilts_and_headings() {
    let b = line("w", &[[0.0, 0.0], [1.0, 1.0]]);
    let a = line("w", &[[0.0, 0.0], [1.0, 3f64.sqrt()]]);
    let v = |k: Value| measure(k).eval(&b, &a).unwrap().v;
    assert!((v(json!({ "id": "a", "kind": "angle", "line": "w" })) - 45.0).abs() < 1e-9);
    assert!((v(json!({ "id": "a", "kind": "angle_change", "line": "w" })) - 15.0).abs() < 1e-9);
    assert!((v(json!({ "id": "a", "kind": "heading_change", "line": "w" })) - 15.0).abs() < 1e-9);
    // Rear end (smaller x) raised: + rear up.
    let b = line("r", &[[0.0, 0.0], [1.0, 0.0]]);
    let a = line("r", &[[0.0, 0.1], [1.0, 0.0]]);
    let t = measure(json!({ "id": "t", "kind": "tilt_change", "line": "r" }))
        .eval(&b, &a)
        .unwrap()
        .v;
    assert!((t - 0.1f64.atan().to_degrees()).abs() < 1e-9);
}

#[test]
fn tracings_flip_into_structure_axes() {
    let sc = schema(json!({ "id": "t", "elements": [
        { "name": "w", "type": "circle" }, { "name": "p", "type": "point" }, { "name": "l", "type": "polyline" } ] }));
    let inst: Instance = serde_json::from_value(json!({ "front": "left", "elements": {
        "w": { "c": [0.5, 0.5], "rim": [0.5, 0.75], "vis": "blurred" },
        "p": { "p": [0.25, 0.1] },
        "l": { "pts": [] } } }))
    .unwrap();
    let s = from_tracing(&inst, &sc, 200.0, 100.0);
    assert_eq!(
        s.circles["w"],
        Circle {
            c: [-100.0, -50.0],
            r: 25.0
        }
    );
    assert_eq!(s.points["p"], [-50.0, -10.0]);
    assert!(!s.has("l"), "an empty polyline is absent");
    assert_eq!(s.vis_of("w"), Vis::Blurred);
}

fn car(front: P, rear: P, r: f64, hood: &[P]) -> Shape {
    let mut s = line("hood", hood);
    s.circles
        .insert("front_wheel".into(), Circle { c: front, r });
    s.circles.insert("rear_wheel".into(), Circle { c: rear, r });
    s
}

#[test]
fn compare_normalises_each_source_by_its_own_before() {
    let side = schema(json!({ "id": "s",
        "elements": [{ "name": "front_wheel", "type": "circle" }, { "name": "rear_wheel", "type": "circle" }, { "name": "hood", "type": "polyline" }],
        "frames": [
            { "id": "wheelbase", "origin": { "centre": "rear_wheel" }, "unit": { "distance": ["front_wheel", "rear_wheel"] } },
            { "id": "radius", "origin": { "centre": "front_wheel" }, "unit": { "radius": "front_wheel" } } ],
        "measures": [
            { "id": "setback", "kind": "shift_x", "of": { "centre": "front_wheel" }, "sign": -1 },
            { "id": "ride", "kind": "shift_y", "of": { "centre": "front_wheel" }, "frames": ["wheelbase"] } ] }));
    // Source a in metres, source b in pixels at 100 px/m: the same crash.
    let a = Source {
        name: "a".into(),
        before: car([2.5, 0.3], [0.0, 0.3], 0.3, &[[3.0, 0.8], [2.0, 1.0]]),
        now: car([2.25, 0.3], [0.0, 0.3], 0.3, &[[2.8, 0.8], [2.0, 1.0]]),
    };
    let b = Source {
        name: "b".into(),
        before: car(
            [250.0, 30.0],
            [0.0, 30.0],
            30.0,
            &[[300.0, 80.0], [200.0, 100.0]],
        ),
        now: car(
            [225.0, 30.0],
            [0.0, 30.0],
            30.0,
            &[[280.0, 80.0], [200.0, 100.0]],
        ),
    };
    let out = compare(&side, &[a, b]);
    assert_eq!(out["frame"]["id"], "wheelbase");
    let row = &out["rows"][0]["values"];
    assert!(
        (row["a"]["v"].as_f64().unwrap() - 0.1).abs() < 1e-9,
        "{row}"
    );
    assert!(
        (row["b"]["v"].as_f64().unwrap() - 0.1).abs() < 1e-9,
        "{row}"
    );
    assert_eq!(out["rows"].as_array().unwrap().len(), 2);
    // Without a rear wheel the second frame is used and the wheelbase-only row is dropped.
    let mut c = car([2.5, 0.3], [0.0, 0.3], 0.3, &[]);
    c.circles.remove("rear_wheel");
    let mut d = c.clone();
    d.circles.get_mut("front_wheel").unwrap().c = [2.2, 0.3];
    let out = compare(
        &side,
        &[Source {
            name: "c".into(),
            before: c,
            now: d,
        }],
    );
    assert_eq!(out["frame"]["id"], "radius");
    assert_eq!(out["rows"].as_array().unwrap().len(), 1);
    // The front wheel is the origin of that frame, so its own setback is 0 by definition.
    assert_eq!(out["rows"][0]["values"]["c"]["v"].as_f64().unwrap(), 0.0);
}

#[test]
fn envelope_and_hull() {
    let pts = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 0.0],
        [1.0, 0.5],
    ];
    assert_eq!(envelope(&pts, 0, 1.0, 2), vec![[1.0, 2.0], [1.0, 2.0]]);
    let h = hull(&pts);
    assert_eq!(h.len(), 6, "five corners and the closing point: {h:?}");
    assert_eq!(h.first(), h.last());
    assert!(
        !h.contains(&[1.0, 0.5]),
        "the interior point is not on the hull"
    );
}

#[test]
fn rubble_camera_view_round_trips_to_an_instance() {
    let sc = schema(json!({ "id": "s", "elements": [
        { "name": "front_wheel", "type": "circle", "rubble": { "wheel": "wheel_fl" } },
        { "name": "rear_wheel", "type": "circle", "rubble": { "wheel": "wheel_rl" } },
        { "name": "roof", "type": "polyline", "rubble": { "envelope": { "groups": ["roof"], "along": "x", "take": "max", "bins": 2 } } },
        { "name": "corner", "type": "point", "rubble": { "ghost_corner": "door" } } ] }));
    // Pixels, front at the left (front wheel x < rear wheel x), y down.
    let body = json!({ "rear_pitch_deg": 1.5, "cameras": { "side": {
        "wheels": [ { "name": "wheel_fl", "now": { "c": [100.0, 400.0], "r": 50.0 } },
                    { "name": "wheel_rl", "now": { "c": [700.0, 400.0], "r": 50.0 } } ],
        "hood_centreline": { "now": [] },
        "parts": {
            "roof": { "now": { "lines": [[[300.0, 100.0], [500.0, 90.0], [600.0, 120.0], null]] }, "ghost": { "lines": [] } },
            "door": { "now": { "lines": [] }, "ghost": { "lines": [[[200.0, 150.0], [200.0, 380.0], [420.0, 385.0], [420.0, 150.0]]] } } } } } });
    let v = BodyView::new(&body, "camera:side").unwrap();
    assert_eq!(v.sx, -1.0);
    let s = derive(&sc, &v);
    // Structure axes: x forward (image left), y up.
    assert_eq!(s.circles["front_wheel"].c, [-100.0, -400.0]);
    assert_eq!(s.fit["pitch"], 1.5);
    // The door's lowest front corner: lowest band (y up: the larger pixel y), largest x forward = smallest pixel x.
    assert_eq!(s.points["corner"], [-200.0, -380.0]);
    assert_eq!(s.lines["roof"].len(), 2);
    let inst = to_instance(&sc, &s, &v, 800.0, 500.0);
    assert_eq!(inst["front"], "left");
    assert_eq!(inst["elements"]["front_wheel"]["c"], json!([0.125, 0.8]));
    assert_eq!(inst["elements"]["front_wheel"]["rim"], json!([0.1875, 0.8]));
    assert_eq!(inst["elements"]["corner"]["p"], json!([0.25, 0.76]));
}

#[test]
fn the_committed_structure_schemas_parse_and_are_consistent() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/crash-reference/studio/structures");
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let s: Schema = serde_json::from_str(&std::fs::read_to_string(&p).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert!(
            s.elements.len() <= 13,
            "{}: keep a schema to about a dozen elements",
            s.id
        );
        let names: Vec<&str> = s.elements.iter().map(|e| e.name.as_str()).collect();
        let known = |n: &str| assert!(names.contains(&n), "{}: {n} is not an element", s.id);
        for f in &s.frames {
            known(f.origin.element());
            match &f.unit {
                Unit::Distance([a, b]) => {
                    known(a);
                    known(b)
                }
                Unit::Radius(x) | Unit::Chord(x) => known(x),
            }
        }
        n += 1;
    }
    assert!(
        n >= 4,
        "vehicle-side, vehicle-top, vehicle-three-quarter, truss-side"
    );
}
