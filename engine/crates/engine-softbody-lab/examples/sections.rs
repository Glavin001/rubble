//! The thin-walled section calculator (`engine_vehiclegen::thin_wall`) on a fixed set of
//! sections, as JSON: each section's walls, its exact pieces (quadrilaterals), its
//! properties and (with a material) its strengths and crush forces. Read by the external
//! oracle `engine/tools/sections/oracle.py`, which computes the same sections with Python's
//! `sectionproperties` (local only, never a dependency).
//!
//! `cargo run --release -p engine-softbody-lab --example sections > engine/out/sections/ours.json`

use engine_vehiclegen::thin_wall::{crush, Material, ThinSection, Wall};
use serde_json::{json, Value};

fn wall(points: &[[f64; 2]], t: f64, closed: bool) -> Wall {
    Wall { points: points.to_vec(), t, closed }
}

/// The sections compared (m).
fn sections() -> Vec<(&'static str, ThinSection, Material)> {
    let hsla = Material::steel(340e6, 450e6);
    let mm = 1e-3;
    // A two-part closed rail: an outer hat (100 wide over its webs, 70 deep, 20 flanges,
    // 1.6) and an inner C-shaped stamping (1.4) closing it, their flanges spot-welded face to
    // face (the inner's flanges stacked under the hat's).
    let (t1, t2) = (1.6 * mm, 1.4 * mm);
    let (x, fl) = (50.0 * mm - 0.5 * t1, 20.0 * mm);
    let hat_y = 0.5 * t1;
    let outer = wall(&[[-x - fl, hat_y], [-x, hat_y], [-x, 70.0 * mm - 0.5 * t1], [x, 70.0 * mm - 0.5 * t1], [x, hat_y], [x + fl, hat_y]], t1, false);
    let yi = -0.5 * t2;
    // (its web just inside the hat's, below the flanges' plane: no overlap)
    let xi = x - 0.5 * t1 + 0.5 * t2;
    let inner = wall(&[[-x - fl, yi], [-xi, yi], [-xi, -30.0 * mm], [xi, -30.0 * mm], [xi, yi], [x + fl, yi]], t2, false);
    // An irregular open section: a lipped channel with a sloped web.
    let lipped = wall(&[[60.0 * mm, 20.0 * mm], [60.0 * mm, 0.0], [0.0, 0.0], [10.0 * mm, 120.0 * mm], [70.0 * mm, 120.0 * mm], [70.0 * mm, 100.0 * mm]], 2.0 * mm, false);
    vec![
        ("rect_tube_100x60x3", ThinSection::rect_tube(0.100, 0.060, 0.003), hsla),
        ("circular_tube_100x4", ThinSection::circular_tube(0.100, 0.004, 720), hsla),
        ("truck_c_266x90x8", ThinSection::c_channel(0.266, 0.090, 0.008), Material::steel(500e6, 600e6)),
        ("hat_100x60_fl20_t1.5_plate1.2", ThinSection::hat(0.100, 0.060, 0.020, 0.0015, Some(0.0012)), hsla),
        ("hat_100x60_fl20_t1.5_open", ThinSection::hat(0.100, 0.060, 0.020, 0.0015, None), hsla),
        ("two_part_rail", ThinSection { walls: vec![outer, inner] }, hsla),
        ("lipped_channel_sloped", ThinSection { walls: vec![lipped] }, hsla),
        ("camry_rail_92x166x1.69", ThinSection::rect_tube(0.092, 0.166, 0.00169), Material { flow_pa: Some(499e6), ..Material::steel(360e6, 600e6) }),
        ("sedan_crash_box_70x110x1.5", ThinSection::rect_tube(0.070, 0.110, 0.0015), engine_vehiclegen::structure::Grade::Hsla.into()),
        ("sedan_front_rail_70x110x1.8", ThinSection::rect_tube(0.070, 0.110, 0.0018), engine_vehiclegen::structure::Grade::Hsla.into()),
    ]
}

/// The outer size and wall of the rectangular tubes (m).
fn tube(name: &str) -> Option<[f64; 3]> {
    match name {
        "rect_tube_100x60x3" => Some([0.100, 0.060, 0.003]),
        "camry_rail_92x166x1.69" => Some([0.092, 0.166, 0.00169]),
        "sedan_crash_box_70x110x1.5" => Some([0.070, 0.110, 0.0015]),
        "sedan_front_rail_70x110x1.8" => Some([0.070, 0.110, 0.0018]),
        _ => None,
    }
}

fn main() -> Result<(), String> {
    let mut out: Vec<Value> = Vec::new();
    for (name, s, m) in sections() {
        let p = s.properties()?;
        let st = p.strengths(&m, 1.0);
        let pieces = s.pieces()?;
        // A box's mean crush force: static at σ_y and σ₀, and at 15.6 m/s with each
        // Cowper–Symonds pair (Abramowicz–Jones; `thin_wall::crush`).
        let mean_crush = match tube(name) {
            Some([b, h, t]) => {
                let c = 0.5 * (b + h);
                let s0 = crush::rect_tube_mean_crush(m.flow(), t, b, h);
                json!({
                    "static_yield_n": crush::rect_tube_mean_crush(m.yield_pa, t, b, h),
                    "static_flow_n": s0,
                    "wierzbicki_abramowicz_n": crush::wierzbicki_abramowicz(m.flow(), t, c),
                    "at_15_6_mps_camry_deck_n": s0 * crush::dynamic_factor(15.6, c, crush::CS_CAMRY_DECK),
                    "at_15_6_mps_mild_steel_n": s0 * crush::dynamic_factor(15.6, c, crush::CS_MILD_STEEL),
                    "stroke_fraction": crush::EFFECTIVE_CRUSH_FRACTION,
                })
            }
            None => Value::Null,
        };
        out.push(json!({
            "name": name,
            "mean_crush": mean_crush,
            "walls": s.walls,
            "pieces": pieces,
            "material": m,
            "properties": p,
            "strengths": st,
        }));
    }
    println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
    Ok(())
}
