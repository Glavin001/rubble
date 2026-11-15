//! Explicit soft-body vehicles in the World (`explicit-world.html`): the crash town
//! (`rubble::explicit_scene::crash_town`, the native benchmark's builder) on the page's
//! device, and the buffers a renderer reads in place (no readback).

use crate::{err, gpu_buffer, res, WebWorld};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use rubble::explicit_scene::{crash_town, town_coupling, town_layout};
use rubble::GpuContext;
use wasm_bindgen::prelude::*;

thread_local! {
    /// Diagnostic: an explicit-vehicles snapshot (`debugExplicitSnapshot`).
    static SNAP: std::cell::RefCell<Option<engine_webgpu::GpuSnapshot>> = const { std::cell::RefCell::new(None) };
}

fn num(v: &serde_json::Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}

fn js(v: serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

/// `{ houses, ground, centres: [[x, y, z], ...] }` of a crash town for `count` vehicles
/// (at least one house per four).
#[wasm_bindgen(js_name = vehicleTownLayout)]
pub fn vehicle_town_layout(houses: u32, count: u32) -> Result<JsValue, JsValue> {
    let (centres, ground) = res(town_layout(houses as usize, count as usize))?;
    Ok(js(serde_json::json!({ "houses": centres.len(), "ground": ground, "centres": centres })))
}

#[wasm_bindgen]
impl WebWorld {
    /// The crash town on a new device. `spec`: `{ truck: <.truck text>, count, houses,
    /// speed (m/s), runUp (m), drag (default true) }`: `count` copies of the vehicle into
    /// prefractured cuboid houses (threshold destructibles), one from each side of each,
    /// past a crate each.
    #[wasm_bindgen(js_name = createVehicleTown)]
    pub async fn create_vehicle_town(spec: JsValue, progress: Option<js_sys::Function>) -> Result<WebWorld, JsValue> {
        let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
        let truck = spec
            .get("truck")
            .and_then(|t| t.as_str())
            .ok_or_else(|| err("createVehicleTown: spec.truck (.truck text)"))?;
        let drag = spec.get("drag").and_then(|x| x.as_bool()).unwrap_or(true);
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            drag,
            turbulence: drag,
            ..Default::default()
        };
        let template = ExplicitBeams::from_truck(truck, glam::Vec3::ZERO, glam::Vec3::ZERO, settings).map_err(err)?;
        let (info, device, queue) = crate::web_device().await?;
        // Compile the vehicles' kernels without blocking first (as `explicitSimAsync`
        // does): created synchronously inside `crash_town`, the contact frame compiles
        // past a windowed browser's GPU watchdog and the device is lost.
        let count = num(&spec, "count", 4.0) as usize;
        let fleet = vec![template.clone(); count.max(1)];
        let specs = engine_webgpu::ExplicitBeamsGpu::compute_specs(&fleet, Some(town_coupling().capacity));
        crate::warm_pipelines(&device, &specs, crate::hear_fn(progress)).await.map_err(err)?;
        let town = res(crash_town(
            GpuContext::Shared { info, device, queue },
            &template,
            num(&spec, "houses", 0.0) as usize,
            num(&spec, "count", 4.0) as usize,
            num(&spec, "speed", 15.0) as f32,
            num(&spec, "runUp", 8.0) as f32,
            0,
            &town_coupling(),
        ))?;
        crate::creation_check(&town.world).await?;
        Ok(WebWorld { world: town.world })
    }

    /// The explicit vehicles' node state `GPUBuffer` (3 `vec4<f32>` per node: position
    /// relative to its vehicle's origin and mass, velocity, force).
    #[wasm_bindgen(js_name = explicitNodesBuffer)]
    pub fn explicit_nodes_buffer(&self) -> JsValue {
        self.world.explicit_vehicles().map_or(JsValue::NULL, |g| gpu_buffer(g.nodes_buffer()))
    }

    /// Per vehicle `vec4<u32>`: its origin's bits (xyz).
    #[wasm_bindgen(js_name = explicitMiscBuffer)]
    pub fn explicit_misc_buffer(&self) -> JsValue {
        self.world.explicit_vehicles().map_or(JsValue::NULL, |g| gpu_buffer(g.misc_buffer()))
    }

    /// Collision triangles: `vec4<u32>` per triangle (global nodes a, b, c, vehicle).
    #[wasm_bindgen(js_name = explicitTrianglesBuffer)]
    pub fn explicit_triangles_buffer(&self) -> JsValue {
        self.world.explicit_triangles().map_or(JsValue::NULL, |(b, _)| gpu_buffer(b))
    }

    #[wasm_bindgen(js_name = explicitTriangleCount)]
    pub fn explicit_triangle_count(&self) -> u32 {
        self.world.explicit_triangles().map_or(0, |(_, n)| n)
    }

    /// Run the explicit frame as the wide passes (`true`) or one workgroup per vehicle
    /// (an ablation knob; results are the same). Returns whether wide is now on.
    #[wasm_bindgen(js_name = setExplicitWide)]
    pub fn set_explicit_wide(&mut self, on: bool) -> bool {
        self.world.explicit_vehicles_mut().is_some_and(|g| g.set_wide(on))
    }

    /// Measurement: the reused self-collision pair list's margin (m; 0 off) and backoff
    /// (a list reused on fewer steps than `streak` sends the next grid steps to the walk).
    #[wasm_bindgen(js_name = setExplicitPairReuse)]
    pub fn set_explicit_pair_reuse(&mut self, margin: f32, streak: u32) -> bool {
        let Some(g) = self.world.explicit_vehicles_mut() else { return false };
        g.set_pair_reuse_margin(margin);
        g.set_pair_reuse_streak(streak);
        true
    }

    /// Measurement knobs of the explicit frame (ablations in a browser): `coupling` (the
    /// rigid coupling in the frame; off, the vehicles no longer feel the bodies),
    /// `couplingCull`, `couplingCache`, `couplingRanges`, `surfacePairs`, `surfaceCulls`, `pairReuse`
    /// (lossless paths on or off). Returns false for an unknown knob.
    #[wasm_bindgen(js_name = setExplicitKnob)]
    pub fn set_explicit_knob(&mut self, name: &str, on: bool) -> bool {
        let Some(g) = self.world.explicit_vehicles_mut() else { return false };
        match name {
            "coupling" => g.set_coupling_on(on),
            "couplingCull" => g.set_coupling_cull(on),
            "couplingCache" => g.set_coupling_workgroup_cache(on),
            "surfacePairs" => g.set_surface_pairs(on),
            "surfaceCulls" => g.set_surface_culls(on),
            "couplingRanges" => g.set_coupling_node_ranges(on),
            "pairReuse" => g.set_pair_reuse(on),
            _ => return false,
        }
        true
    }

    /// The explicit solver's errors since the last call (a loop at its cap, vehicle
    /// contact limits); empty: none.
    #[wasm_bindgen(js_name = explicitErrors)]
    pub fn explicit_errors(&mut self) -> Vec<String> {
        self.world.explicit_vehicles_mut().map_or(Vec::new(), |g| g.take_contact_errors())
    }

    /// Diagnostic: the explicit frame module's WGSL (per-vehicle workgroups).
    #[wasm_bindgen(js_name = debugExplicitFrameWgsl)]
    pub fn debug_explicit_frame_wgsl(&self) -> String {
        self.world.explicit_vehicles().map_or(String::new(), |g| g.debug_frame_wgsl())
    }

    /// Diagnostic: the explicit frame's state bindings' buffer sizes (bytes, bindings 0..=8).
    #[wasm_bindgen(js_name = debugExplicitBindingSizes)]
    pub fn debug_explicit_binding_sizes(&self) -> Vec<f64> {
        self.world.explicit_vehicles().map_or(Vec::new(), |g| g.debug_binding_sizes().into_iter().map(|x| x as f64).collect())
    }

    /// Diagnostic: extra bits in the coupling header word (an edited frame module's
    /// runtime switches).
    #[wasm_bindgen(js_name = debugExplicitCouplingBits)]
    pub fn debug_explicit_coupling_bits(&mut self, bits: u32) {
        if let Some(g) = self.world.explicit_vehicles_mut() {
            g.debug_coupling_bits(bits);
        }
    }

    /// Diagnostic: a contact module's WGSL (`which`: "solo", "groups" or "big").
    #[wasm_bindgen(js_name = debugExplicitContactWgsl)]
    pub fn debug_explicit_contact_wgsl(&self, which: &str) -> String {
        self.world.explicit_vehicles().map_or(String::new(), |g| g.debug_contact_wgsl(which))
    }

    /// Diagnostic: run `src` as a contact module.
    #[wasm_bindgen(js_name = debugSetExplicitContactWgsl)]
    pub fn debug_set_explicit_contact_wgsl(&mut self, which: &str, src: &str) {
        if let Some(g) = self.world.explicit_vehicles_mut() {
            g.debug_set_contact_wgsl(which, src);
        }
    }

    /// Diagnostic: the large contact groups' module's WGSL.
    #[wasm_bindgen(js_name = debugExplicitContactBigWgsl)]
    pub fn debug_explicit_contact_big_wgsl(&self) -> String {
        self.world.explicit_vehicles().map_or(String::new(), |g| g.debug_contact_big_wgsl())
    }

    /// Diagnostic: run `src` as the large contact groups' module.
    #[wasm_bindgen(js_name = debugSetExplicitContactBigWgsl)]
    pub fn debug_set_explicit_contact_big_wgsl(&mut self, src: &str) {
        if let Some(g) = self.world.explicit_vehicles_mut() {
            g.debug_set_contact_big_wgsl(src);
        }
    }

    /// Diagnostic: run `src` as the explicit frame module (an edited copy of
    /// [`Self::debug_explicit_frame_wgsl`]; ablations without a rebuild).
    #[wasm_bindgen(js_name = debugSetExplicitFrameWgsl)]
    pub fn debug_set_explicit_frame_wgsl(&mut self, src: &str) {
        if let Some(g) = self.world.explicit_vehicles_mut() {
            g.debug_set_frame_wgsl(src);
        }
    }

    /// Diagnostic: the explicit accumulators `GPUBuffer` (gAcc) and the word of its
    /// last proxy shape's sums (free while vehicle 0 has fewer proxy shapes than its
    /// room): `[buffer, word]`.
    #[wasm_bindgen(js_name = debugExplicitAcc)]
    pub fn debug_explicit_acc(&self) -> JsValue {
        let (Some(g), Some(c)) = (self.world.explicit_vehicles(), self.world.explicit_coupling_layout()) else {
            return JsValue::NULL;
        };
        let arr = js_sys::Array::new();
        arr.push(&gpu_buffer(g.acc_buffer()));
        arr.push(&JsValue::from(c.acc_base + c.cap.bodies + 6 * (c.cap.shapes - 1)));
        arr.into()
    }

    /// Per vehicle its frame-bounds margin (m; a vehicle's widens after it left its
    /// frame bounds).
    #[wasm_bindgen(js_name = explicitContactMargin)]
    pub fn explicit_contact_margin(&self) -> Vec<f64> {
        self.world.explicit_vehicles().map_or(Vec::new(), |g| {
            (0..g.vehicle_count()).map(|v| f64::from(g.vehicle_contact_margin(v))).collect()
        })
    }

    /// Diagnostic: the vehicle-contact records' first gMisc record (6 a vehicle: padded
    /// box lo, hi, allowed lo, hi, flags, member/own margin).
    #[wasm_bindgen(js_name = debugExplicitVvRecords)]
    pub fn debug_explicit_vv_records(&self) -> u32 {
        self.world.explicit_vehicles().map_or(0, |g| (g.vv_record_words().start / 4) as u32)
    }

    /// Diagnostic (determinism checks): keep a snapshot of the explicit vehicles' state.
    #[wasm_bindgen(js_name = debugExplicitSnapshot)]
    pub fn debug_explicit_snapshot(&self) {
        if let Some(g) = self.world.explicit_vehicles() {
            let snap = g.snapshot();
            SNAP.with(|s| *s.borrow_mut() = Some(snap));
        }
    }

    /// Diagnostic: restore the kept snapshot and run one explicit frame of `steps` steps
    /// (the vehicles alone, the coupling's records as they were).
    #[wasm_bindgen(js_name = debugExplicitRestoreAndFrame)]
    pub fn debug_explicit_restore_and_frame(&mut self, steps: u32) {
        if let Some(g) = self.world.explicit_vehicles_mut() {
            SNAP.with(|s| {
                if let Some(snap) = s.borrow().as_ref() {
                    g.restore(snap);
                }
            });
            g.frame(steps);
        }
    }

    /// Whether the explicit frame runs as the wide passes.
    #[wasm_bindgen(js_name = explicitWide)]
    pub fn explicit_wide(&self) -> bool {
        self.world.explicit_vehicles().is_some_and(|g| g.is_wide())
    }

    /// `[explicit steps, World steps]` the vehicles have run.
    #[wasm_bindgen(js_name = explicitSteps)]
    pub fn explicit_steps(&self) -> Vec<f64> {
        self.world
            .explicit_vehicle_steps()
            .map_or(vec![0.0, 0.0], |(s, w)| vec![s as f64, w as f64])
    }
}
