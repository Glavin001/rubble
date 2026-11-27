//! The front load path's members through the section calculator (`thin_wall`): the 2012 Camry's as
//! read off its finite-element deck (our own reading, `engine/docs/front-load-path.md` §1; typed
//! here as numbers, nothing read from the deck) and the generated sedan's as defined. Prints a
//! Markdown table: section, gauge, yield and flow stress, length, steel mass, squash load, mean
//! crush force (Abramowicz–Jones) static and at crash rate (the Camry deck's Cowper–Symonds
//! constants at 15.6 m/s), the fully plastic moment about the section's strong and weak axes, and
//! the stroke at 0.73 of its length.
//!
//!     cargo run --release -p engine-vehiclegen --example front_load_path

use engine_vehiclegen::thin_wall::{crush, Material, ThinSection};

struct M {
    car: &'static str,
    name: &'static str,
    /// outer section across × high (m), wall (m)
    b: f64,
    h: f64,
    t: f64,
    sy: f64,
    s0: f64,
    len: f64,
}

fn main() {
    // (σ_y, flow σ₀ over plastic strain 0–0.2: the deck's curves, fe_load_path.py)
    let rows = [
        M { car: "Camry", name: "bumper beam (B-section, double cell; across the car)", b: 0.116, h: 0.061, t: 0.00116, sy: 1300e6, s0: 1428e6, len: 1.18 },
        M { car: "Camry", name: "bumper bracket (stub on the rail's end)", b: 0.069, h: 0.116, t: 0.00206, sy: 360e6, s0: 499e6, len: 0.090 },
        M { car: "Camry", name: "crash zone (rail front, x −240…−445)", b: 0.092, h: 0.163, t: 0.00170, sy: 360e6, s0: 499e6, len: 0.205 },
        M { car: "Camry", name: "mid rail (x −445…−760)", b: 0.083, h: 0.145, t: 0.00206, sy: 360e6, s0: 499e6, len: 0.315 },
        M { car: "Camry", name: "rear rail (x −760…−1210)", b: 0.075, h: 0.182, t: 0.00230, sy: 360e6, s0: 499e6, len: 0.450 },
        M { car: "Camry", name: "upper rail top (x −850…−1050, a hat on the apron)", b: 0.075, h: 0.075, t: 0.00102, sy: 250e6, s0: 302e6, len: 0.55 },
        M { car: "Camry", name: "subframe side member (x −450…−1370, one side)", b: 0.070, h: 0.048, t: 0.00230, sy: 360e6, s0: 499e6, len: 0.92 },
        M { car: "Camry", name: "under-floor rail (kick-down, x −1050…−1350)", b: 0.100, h: 0.110, t: 0.00180, sy: 360e6, s0: 499e6, len: 0.30 },
        M { car: "Camry", name: "sill (x −1800)", b: 0.141, h: 0.155, t: 0.00124, sy: 400e6, s0: 584e6, len: 1.10 },
        M { car: "sedan", name: "crash box (defined)", b: 0.070, h: 0.110, t: 0.0015, sy: 340e6, s0: 395e6, len: 0.18 },
        M { car: "sedan", name: "front rail (defined)", b: 0.070, h: 0.110, t: 0.0018, sy: 340e6, s0: 395e6, len: 1.04 },
        M { car: "sedan", name: "upper rail (defined)", b: 0.040, h: 0.060, t: 0.0012, sy: 340e6, s0: 395e6, len: 0.97 },
        M { car: "sedan", name: "sill (defined)", b: 0.065, h: 0.140, t: 0.0014, sy: 340e6, s0: 395e6, len: 1.8 },
    ];
    println!("| car | member | section (mm) | t (mm) | σ_y / σ₀ (MPa) | L (m) | steel (kg) | squash σ_y·A (kN) | P_m static (kN) | rate factor | P_m crash rate (kN) | M_p strong / weak (kN·m) | stroke 0.73 L (m) |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for m in rows {
        let p = ThinSection::rect_tube(m.b, m.h, m.t).properties().expect("section");
        let mat = Material { flow_pa: Some(m.s0), ..Material::steel(m.sy, m.s0) };
        let s = p.strengths(&mat, m.len);
        let pm = crush::rect_tube_mean_crush(m.s0, m.t, m.b, m.h);
        let f = crush::dynamic_factor(15.6, 0.5 * (m.b + m.h), crush::CS_CAMRY_DECK);
        let (strong, weak) = if s.mp_x > s.mp_y { (s.mp_x, s.mp_y) } else { (s.mp_y, s.mp_x) };
        println!(
            "| {} | {} | {:.0} × {:.0} | {:.2} | {:.0} / {:.0} | {:.3} | {:.2} | {:.0} | {:.1} | {:.2} | {:.1} | {:.1} / {:.1} | {:.3} |",
            m.car,
            m.name,
            m.b * 1e3,
            m.h * 1e3,
            m.t * 1e3,
            m.sy / 1e6,
            m.s0 / 1e6,
            m.len,
            s.mass_per_m * m.len,
            s.squash_n / 1e3,
            pm / 1e3,
            f,
            pm * f / 1e3,
            strong / 1e3,
            weak / 1e3,
            crush::EFFECTIVE_CRUSH_FRACTION * m.len
        );
    }
}
