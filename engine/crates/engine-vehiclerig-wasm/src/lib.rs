//! wasm-bindgen face of the vehicle rigger. The vehicle lab's Design panel runs it in a
//! worker: a registered design's parameters in, the files of the soft-body mod out (the
//! JBeam parts, the `.pc`, the lighter render and its skins, `files.json`), the same bytes
//! the `vehicle_gen_rig` CLI writes for that input (`rig_files`).
//!
//! No design is named here: the list, each design's controls (with the ranges its
//! `validate()` enforces) and its defaults come from `engine-vehiclegen`'s registry.

use wasm_bindgen::prelude::*;

/// The registered designs, as JSON: `[{"name", "title"}]`, in registry order.
#[wasm_bindgen]
pub fn design_list() -> String {
    let rows: Vec<serde_json::Value> = engine_vehiclegen::design_titles()
        .iter()
        .map(|(n, t)| serde_json::json!({"name": n, "title": t}))
        .collect();
    serde_json::to_string(&rows).unwrap()
}

/// A design's default parameters as JSON.
#[wasm_bindgen]
pub fn default_params(design: &str) -> Result<String, JsError> {
    engine_vehiclegen::default_params_json(design).map_err(|e| JsError::new(&e))
}

/// A design's configurator controls as JSON (see `engine_vehiclegen::param_schema_json`).
#[wasm_bindgen]
pub fn param_schema(design: &str) -> Result<String, JsError> {
    engine_vehiclegen::param_schema_json(design).map_err(|e| JsError::new(&e))
}

/// Check parameters against the design's `validate()`; the error is its message.
#[wasm_bindgen]
pub fn validate_params(design: &str, params_json: &str) -> Result<(), JsError> {
    engine_vehiclegen::validate_params(design, params_json).map_err(|e| JsError::new(&e))
}

/// The mod's files (path, bytes), taken one at a time by the page.
#[wasm_bindgen]
pub struct RigFiles {
    files: Vec<(String, Vec<u8>)>,
}

#[wasm_bindgen]
impl RigFiles {
    pub fn count(&self) -> usize {
        self.files.len()
    }
    pub fn path(&self, i: usize) -> String {
        self.files[i].0.clone()
    }
    /// File `i`'s bytes, copied out (its copy here is freed: each is taken once).
    pub fn take(&mut self, i: usize) -> Vec<u8> {
        std::mem::take(&mut self.files[i].1)
    }
}

/// Rig `design` from JSON parameters: invalid parameters are the validator's message;
/// the rest (a design that rigs only in some ranges) fails with the rigger's own.
#[wasm_bindgen]
pub fn rig_vehicle(design: &str, params_json: &str) -> Result<RigFiles, JsError> {
    let files = rig_files(design, params_json).map_err(|e| JsError::new(&e))?;
    Ok(RigFiles { files })
}

/// The files `rig_vehicle` returns (the native form, which the tests compare to the CLI's).
pub fn rig_files(design: &str, params_json: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
    engine_vehiclegen::validate_params(design, params_json)?;
    std::panic::catch_unwind(|| {
        engine_softbody_authoring::authoring::gen2::vg_input::web_files(
            design,
            params_json,
            engine_softbody_authoring::authoring::gen2::vg_input::LITE_EDGE,
        )
    })
    .map_err(|_| "the rigger failed for these parameters".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_softbody_authoring::authoring::gen2::vg_input;

    /// Every registered design's files at its defaults are the bytes the CLI writes (the
    /// physics files and the lighter render), in the same code path; and two runs agree.
    #[test]
    fn rig_files_are_the_cli_bytes_for_every_design() {
        for (name, _) in engine_vehiclegen::design_titles() {
            let t = std::time::Instant::now();
            let web = rig_files(name, "").unwrap_or_else(|e| panic!("{name}: {e}"));
            eprintln!("{name}: {} files in {:.1} s", web.len(), t.elapsed().as_secs_f32());
            // the CLI: build (full art), then files_with the lighter render
            let b = vg_input::build(vg_input::VgInput::find(name, None, None).unwrap()).unwrap();
            let lite = vg_input::lite_art(&b, "", vg_input::LITE_EDGE).unwrap();
            let cli = vg_input::files_with(&b, Some(&lite));
            let art = [vg_input::glb_file(name), vg_input::skin_file(name)];
            for (p, bytes) in &web {
                if p == "files.json" {
                    continue;
                }
                // (a path the CLI lists twice is the later file, as on disk)
                let (_, c) = cli.iter().rev().find(|(q, _)| q == p).unwrap_or_else(|| panic!("{name}: {p} is not a CLI file"));
                if c != bytes {
                    let (a, b) = (String::from_utf8_lossy(c), String::from_utf8_lossy(bytes));
                    let at = a.lines().zip(b.lines()).position(|(x, y)| x != y);
                    let l = at.map(|i| (a.lines().nth(i).unwrap().to_string(), b.lines().nth(i).unwrap().to_string()));
                    panic!("{name}: {p} differs from the CLI's at line {at:?}: {l:?}");
                }
            }
            // nothing the CLI writes is missing but the full art and the index
            for (p, _) in &cli {
                let skip = p == "files.json" || art.iter().any(|a| p.ends_with(&format!("/{a}")));
                assert!(skip || web.iter().any(|(q, _)| q == p), "{name}: {p} is missing");
            }
            let list: Vec<String> = serde_json::from_slice(&web.iter().find(|(p, _)| p == "files.json").unwrap().1).unwrap();
            assert!(list.iter().any(|p| p.ends_with(".lite.rskin")) && !list.iter().any(|p| p.ends_with(&format!("/{}", art[1]))));
        }
    }

    #[test]
    fn two_runs_agree() {
        let a = rig_files("sedan", "").unwrap();
        let b = rig_files("sedan", "").unwrap();
        for ((p, x), (_, y)) in a.iter().zip(&b) {
            assert!(x == y, "{p} differs between two runs in one process");
        }
    }

    #[test]
    fn invalid_parameters_are_the_validators_message_and_do_not_rig() {
        let e = rig_files("sedan", r#"{"wheelbase": 9.0}"#).unwrap_err();
        assert!(e.contains("wheelbase") && e.contains("outside"), "{e}");
        let e = rig_files("mach1_1969", r#"{"rim_style": "chrome"}"#).unwrap_err();
        assert!(e.contains("rim_style"), "{e}");
    }
}
