//! The explicit solver's built-in test vehicles as text for the page (`softbody.html`): the
//! high-resolution JBeam test car and the procedural sedan's `.truck`.

use wasm_bindgen::prelude::*;

/// The high-resolution test car (`engine_softbody::hires`) as JBeam text, tuned for
/// the explicit solver's 0.5 ms step (2 kg nodes, 1.5 MN/m over 0.15 m beams, the
/// stiffness of beams shorter than 0.15 m capped), as its tests run it.
#[wasm_bindgen(js_name = hiresCarJbeam)]
pub fn hires_car_jbeam() -> String {
    use engine_softbody::hires::{hires_car_jbeam, HiresParams};
    hires_car_jbeam(&HiresParams {
        node_weight: 2.0,
        spring: 1.5e6,
        damp: 150.0,
        stiffness_length: 0.15,
        ..Default::default()
    })
    .0
}

/// The procedural sedan for the explicit solver, as `.truck` text: the acceptance
/// runs' car (`CarParams::sedan()`, 10 kg minimum node mass;
/// `docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck` is its bare
/// body), with its soft tyres when `tyres`.
#[wasm_bindgen(js_name = explicitSedanTruck)]
pub fn explicit_sedan_truck(tyres: bool) -> String {
    use engine_softbody::procedural::{car_truck, car_truck_bare, CarParams};
    let p = CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    };
    if tyres {
        car_truck(&p)
    } else {
        car_truck_bare(&p)
    }
}
