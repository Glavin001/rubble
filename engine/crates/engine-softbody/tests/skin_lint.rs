//! The binding-quality lint (`authoring::skin_lint`) on the showcase sedan:
//! its thresholds, the triangle binding it replaces measured for contrast,
//! and mutations proving each measure catches a broken binding
//! (engine/docs/measurements/skin-shards.md, "Binding lint").

use engine_softbody::authoring::skin::{
    raycast_visual, showcase_visual_lod, showcase_visual_triangles_lod, ShowcaseVisual,
};
use engine_softbody::authoring::skin_lint::{lint, LintReport};
use engine_softbody::authoring::{build, style, BuildParams, Built, Resolution};
use std::sync::OnceLock;

fn car() -> &'static (Built, ShowcaseVisual) {
    static CAR: OnceLock<(Built, ShowcaseVisual)> = OnceLock::new();
    CAR.get_or_init(|| {
        let b = build(
            &style("sedan").unwrap(),
            &BuildParams {
                resolution: Resolution::Medium,
                ..Default::default()
            },
        )
        .unwrap();
        let mut v = showcase_visual_lod(&b, 1.0).unwrap();
        raycast_visual(&mut v, &b.truck).unwrap();
        (b, v)
    })
}

/// Probe every this-many-th node (B3).
const PROBE_EVERY: usize = 7;

/// The thresholds (skin-shards.md, "Binding lint"):
/// - A1 0 vertices driven from outside their group (BeamNG: a flexbody is
///   deformed only by its own node groups);
/// - A3 no vertex extrapolates: Σ|w| ≤ 1.001 (a convex blend is 1);
/// - B3 gain ≤ [`GAIN_MAX`]: a vertex moves at most this × a node's own move
///   (a partition-of-unity blend of rigidly moved frames gives 1 plus the
///   frames' turn times the vertex's offset; past 1.5 it extrapolates: the
///   spike predictor);
/// - A6 0 incoherent edges (the saw-tooth predictor);
/// - A7 no part under 3 nodes or on a line (BeamNG's VX/VY errors);
/// - B1 0 leaks, rigid error ≤ 0.1 mm (f32 at car scale is ~10⁻⁶ m);
/// - B4 0 bridges.
fn check(r: &LintReport) {
    assert_eq!(r.cross_group, 0, "A1: {} vertices driven from outside their group", r.cross_group);
    assert!(r.gain_max <= GAIN_MAX, "B3: gain {} (worst {:?})", r.gain_max, r.gain_worst);
    assert_eq!(r.weight_norm_over, 0, "A3: {} vertices extrapolate (max Σ|w| {})", r.weight_norm_over, r.weight_norm_max);
    assert_eq!(r.incoherent_edges, 0, "A6: {} incoherent edges", r.incoherent_edges);
    assert!(r.bad_groups.is_empty(), "A7: {:?}", r.bad_groups);
    assert_eq!(r.leaks, 0, "B1: {} vertices move with another part", r.leaks);
    assert!(r.rigid_err <= 1e-4, "B1: rigid error {} m", r.rigid_err);
    assert_eq!(r.bridges, 0, "B4: {} triangles bridge a separated part", r.bridges);
}

const GAIN_MAX: f32 = 1.5;

#[test]
fn the_showcase_sedan_passes_the_binding_lint() {
    let (b, v) = car();
    let r = lint(v, b, PROBE_EVERY);
    eprintln!("{r:#?}");
    check(&r);
}

/// The triangle binding the node frames replaced fails it (the crash
/// artefacts' predictors).
#[test]
fn the_triangle_binding_fails_the_lint() {
    let (b, _) = car();
    let mut old = showcase_visual_triangles_lod(b, 1.0).unwrap();
    raycast_visual(&mut old, &b.truck).unwrap();
    let r = lint(&old, b, PROBE_EVERY);
    eprintln!("{r:#?}");
    assert!(r.cross_group > 0 && r.leaks > 0 && r.bridges > 0 && r.gain_max > GAIN_MAX && r.weight_norm_over > 0);
}

/// A vertex of a framed piece and a node of another part.
fn victim(v: &ShowcaseVisual, b: &Built) -> (usize, u32) {
    let i = (0..v.positions.len())
        .find(|&i| v.bl_node[i][0] != u32::MAX && v.part[i] != u16::MAX && v.rigid[i] == 0)
        .unwrap();
    let p = v.part[i];
    let other = (0..b.skeleton.nodes.len() as u32)
        .find(|&j| b.skeleton.nodes[j as usize].part != p && b.skeleton.nodes[j as usize].part != u16::MAX)
        .unwrap();
    (i, other)
}

/// Mutations: each broken binding is caught by its measure.
#[test]
fn the_lint_catches_broken_bindings() {
    let (b, v) = car();
    // Cross-part vertex: A1, B1 and B4.
    let mut m = v.clone();
    let (i, other) = victim(v, b);
    m.bl_node[i][0] = other;
    let r = lint(&m, b, 0);
    assert!(r.cross_group >= 1 && r.leaks >= 1 && r.bridges >= 1, "cross-part: {r:#?}");
    // Extrapolating vertex (weights 3, −2 on two of its nodes): A3.
    let mut m = v.clone();
    m.bl_w[i] = [0.0; 8];
    m.bl_w[i][0] = 3.0;
    m.bl_w[i][1] = -2.0;
    let r = lint(&m, b, 0);
    assert!(r.weight_norm_over >= 1 && r.weight_norm_max >= 4.9, "extrapolation: {r:?}");
    // Incoherent vertex (driven by its part's nodes farthest from it): A6.
    let mut m = v.clone();
    let p = v.part[i];
    let x = glam::Vec3::from_array(v.positions[i]);
    let mut far: Vec<u32> = (0..b.skeleton.nodes.len() as u32)
        .filter(|&j| b.skeleton.nodes[j as usize].part == p)
        .collect();
    far.sort_by(|&a, &c| {
        let d = |j: u32| glam::Vec3::from_array(v.node_rest[j as usize]).distance(x);
        d(c).total_cmp(&d(a))
    });
    m.bl_node[i] = [u32::MAX; 8];
    m.bl_node[i][0] = far[0];
    m.bl_w[i] = [0.0; 8];
    m.bl_w[i][0] = 1.0;
    let r = lint(&m, b, 0);
    assert!(r.incoherent_edges >= 1, "incoherent: {r:#?}");
    // A part on a line: A7.
    let mut m = v.clone();
    let pid = b.skeleton.parts.iter().position(|s| s == "mirror_l").or(Some(0)).unwrap() as u16;
    let ids: Vec<usize> = (0..b.skeleton.nodes.len()).filter(|&j| b.skeleton.nodes[j].part == pid).collect();
    for (k, &j) in ids.iter().enumerate() {
        m.node_rest[j] = [k as f32 * 0.1, 0.0, 0.0];
    }
    let r = lint(&m, b, 0);
    assert!(!r.bad_groups.is_empty(), "collinear: {r:#?}");
}

#[test]
#[ignore]
fn report_both_bindings() {
    let (b, v) = car();
    let t0 = std::time::Instant::now();
    let r = lint(v, b, PROBE_EVERY);
    eprintln!("node frames ({:.1} s): {r:#?}", t0.elapsed().as_secs_f32());
    let mut old = showcase_visual_triangles_lod(b, 1.0).unwrap();
    raycast_visual(&mut old, &b.truck).unwrap();
    let r: LintReport = lint(&old, b, PROBE_EVERY);
    eprintln!("triangles: {r:#?}");
}
