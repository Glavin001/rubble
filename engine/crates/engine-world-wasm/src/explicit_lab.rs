//! The crash-test lab's reads and the node grab on [`ExplicitSim`]
//! (`engine-webgpu/softbody.html?solver=explicit`, `crash-lab-render.mjs`).
//!
//! - **Grab** ([`ExplicitSim::set_grab`]): BeamNG-style node grabbing, the solver's
//!   [`Grab`] (an external spring–damper on one node, the CPU reference's node pass
//!   and the GPU kernel's `grabApply`), set per frame while the mouse drags.
//! - **Beam info** ([`ExplicitSim::lab_beams_buffer`]): per beam the constants the
//!   strain colours need (stiffness, break strength, yield threshold) and a part
//!   for colouring by part.
//! - **Frame state** ([`ExplicitSim::read_frame_state`]): node world positions and
//!   beam rest lengths of a frame (the page records them for its scrub replay and
//!   picks grabbed nodes from them), with whether the frame shows an impact: a
//!   node inside a static collider (or, for a vehicle spawned in the air, on the
//!   ground), two vehicles' boxes touching, or a beam broken. A bent beam alone is
//!   not one: a car settling on its tyres after spawning yields a few beams.

use super::ExplicitSim;
use crate::err;
use engine_softbody::explicit_beams::{
    box_face, shape_face, ExplicitBeams, ExplicitKind, Grab, StaticShape, NODE_TYRE, STEP,
};
use glam::Vec3;
use wasm_bindgen::prelude::*;

/// Part names of [`ExplicitSim::lab_beams_buffer`]'s part indices.
const PARTS: [&str; 6] = ["front", "rear", "roof", "cabin", "tyres", "suspension"];

/// Vehicles a frame-state read takes at most (its read copies every node and beam).
const FRAME_STATE_VEHICLES: usize = 8;

/// A vehicle spawned with its lowest node this far above the ground counts its first
/// ground contact as an impact (the drop test).
const AIRBORNE: f32 = 0.3;

/// The part of each beam (indices into [`PARTS`]): tyre beams, beams that are not
/// plain springs (shocks, supports, ropes, commands), then the body by where the
/// beam's middle sits in the vehicle at spawn: front and rear thirds, the roof (the
/// top 38 % of its height) and the cabin. A display grouping, not data from the
/// file.
fn beam_parts(sim: &ExplicitBeams, forward: Vec3) -> Vec<u32> {
    let p = |i: u32| sim.nodes[i as usize].pos;
    let (mut f0, mut f1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for n in &sim.nodes {
        if n.flags & NODE_TYRE != 0 {
            continue;
        }
        let f = n.pos.dot(forward);
        f0 = f0.min(f);
        f1 = f1.max(f);
        y0 = y0.min(n.pos.y);
        y1 = y1.max(n.pos.y);
    }
    let tyre = |i: u32| sim.nodes[i as usize].flags & NODE_TYRE != 0;
    sim.beams
        .iter()
        .map(|b| {
            if tyre(b.a) || tyre(b.b) {
                return 4;
            }
            if b.kind != ExplicitKind::Plain && b.kind != ExplicitKind::JbeamPlain {
                return 5;
            }
            let m = 0.5 * (p(b.a) + p(b.b));
            let t = (m.dot(forward) - f0) / (f1 - f0).max(1e-6);
            let h = (m.y - y0) / (y1 - y0).max(1e-6);
            if t > 0.68 {
                0
            } else if t < 0.3 {
                1
            } else if h > 0.62 {
                2
            } else {
                3
            }
        })
        .collect()
}

/// Whether world point `x` is inside one of the vehicle's static colliders (its
/// boxes and shapes, as its node pass tests them).
fn in_static(sim: &ExplicitBeams, x: Vec3) -> bool {
    sim.boxes.iter().any(|b| box_face(b, x).is_some())
        || sim
            .shapes
            .iter()
            .any(|s| shape_face(s, &StaticShape::bounds(s), x).is_some())
}

#[wasm_bindgen]
impl ExplicitSim {
    /// Grab node `node` of vehicle `vehicle`: from the next frame on every step pulls
    /// it toward the world point (x, y, z) with stiffness `k` (N/m) and damping `c`
    /// (N·s/m; default: critical for the node's own mass, at most half of what one
    /// step keeps stable). Call again each frame to move the target.
    #[wasm_bindgen(js_name = setGrab)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_grab(
        &mut self,
        vehicle: u32,
        node: u32,
        x: f32,
        y: f32,
        z: f32,
        k: f32,
        c: Option<f32>,
    ) -> Result<(), JsValue> {
        let v = self
            .vehicles
            .get(vehicle as usize)
            .ok_or_else(|| err(format!("setGrab: no vehicle {vehicle}")))?;
        let n = v
            .sim
            .nodes
            .get(node as usize)
            .ok_or_else(|| err(format!("setGrab: vehicle {vehicle} has no node {node}")))?;
        if !(k.is_finite() && k >= 0.0) {
            return Err(err(format!("setGrab: stiffness {k}")));
        }
        let c = c.unwrap_or_else(|| (2.0 * (k * n.mass).sqrt()).min(0.5 * n.mass / STEP));
        let target = Vec3::new(x, y, z);
        if !target.is_finite() {
            return Err(err("setGrab: target not finite"));
        }
        self.gpu
            .set_grab(Some((vehicle as usize, Grab { node, target, k, c })));
        Ok(())
    }

    /// Release the grab.
    #[wasm_bindgen(js_name = clearGrab)]
    pub fn clear_grab(&mut self) {
        self.gpu.set_grab(None);
    }

    /// Per beam (in beam-buffer order) a `vec4<u32>`: stiffness (N/m), break
    /// strength (N) and compression yield threshold (N, infinite when it does not
    /// yield) as `f32` bits, then its part ([`Self::lab_parts`]). Spawn values.
    #[wasm_bindgen(js_name = labBeamsBuffer)]
    pub fn lab_beams_buffer(&self) -> JsValue {
        crate::gpu_buffer(&self.lab_beams)
    }

    /// Per node (global order) its vehicle index | 1 << 16 for a tyre node, `u32`.
    #[wasm_bindgen(js_name = labNodeVehicles)]
    pub fn lab_node_vehicles(&self) -> js_sys::Uint32Array {
        let mut out = Vec::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            out.extend(
                v.sim
                    .nodes
                    .iter()
                    .map(|n| i as u32 | u32::from(n.flags & NODE_TYRE != 0) << 16),
            );
        }
        js_sys::Uint32Array::from(out.as_slice())
    }

    /// Part names, by the part index of [`Self::lab_beams_buffer`].
    #[wasm_bindgen(js_name = labParts)]
    pub fn lab_parts(&self) -> js_sys::Array {
        PARTS.iter().map(|s| JsValue::from_str(s)).collect()
    }

    /// One frame's state, resolved once mapped (reads every node and beam: at most
    /// 8 vehicles): `{ step, pos: Float32Array (world x, y, z per node), rest:
    /// Float32Array (per beam its rest length, negated when broken), bent, broken,
    /// contact, impact }`. `contact`: a node inside a static collider (for a vehicle
    /// spawned in the air, also on the ground) or two vehicles' node boxes
    /// touching; `impact`: contact or any beam broken (not bending alone: settling
    /// after the spawn yields a few beams). The state is the one
    /// after the frames submitted so far (`step` steps).
    #[wasm_bindgen(js_name = readFrameState)]
    pub fn read_frame_state(&self) -> Result<js_sys::Promise, JsValue> {
        if self.vehicles.len() > FRAME_STATE_VEHICLES {
            return Err(err(format!(
                "readFrameState: at most {FRAME_STATE_VEHICLES} vehicles, the scene has {}",
                self.vehicles.len()
            )));
        }
        let nodes = self.gpu.nodes_buffer().clone();
        let beams = self.gpu.beams_buffer().clone();
        let misc = self.gpu.misc_buffer().clone();
        let nv = self.vehicles.len();
        let mut ranges = vec![(misc, 0, nv as u64 * 16)];
        let mut statics = Vec::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            let off = u64::from(self.gpu.node_offset(i));
            ranges.push((nodes.clone(), off * 48, v.sim.nodes.len() as u64 * 48));
            let boff = u64::from(self.gpu.beam_offset(i));
            ranges.push((beams.clone(), boff * 32, v.sim.beams.len() as u64 * 32));
            let low = v.sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            // The CPU copy's colliders and spawn rest lengths.
            let mut s = ExplicitBeams::empty(v.sim.settings);
            s.boxes = v.sim.boxes.clone();
            s.shapes = v.sim.shapes.clone();
            let rest0: Vec<f32> = v.sim.beams.iter().map(|b| b.rest0).collect();
            let airborne = v.sim.origin.y + low - v.sim.ground > AIRBORNE;
            statics.push((
                s,
                rest0,
                v.sim.nodes.len(),
                airborne,
                v.sim.ground,
                v.sim.collision_range,
            ));
        }
        let step = self.steps as f64;
        Ok(crate::read::read_ranges_on(
            self.device.clone(),
            self.queue.clone(),
            ranges,
            move |w| {
                let f = |x: u32| f32::from_bits(x);
                let mut pos = Vec::new();
                let mut rest = Vec::new();
                let (mut bent, mut broken, mut contact) = (0u32, 0u32, false);
                let mut boxes = Vec::with_capacity(nv);
                let mut at = nv * 4;
                for (i, (s, rest0, n, airborne, ground, range)) in statics.iter().enumerate() {
                    let o = Vec3::new(f(w[i * 4]), f(w[i * 4 + 1]), f(w[i * 4 + 2]));
                    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
                    for k in 0..*n {
                        let x = o + Vec3::new(
                            f(w[at + k * 12]),
                            f(w[at + k * 12 + 1]),
                            f(w[at + k * 12 + 2]),
                        );
                        pos.extend_from_slice(&x.to_array());
                        lo = lo.min(x);
                        hi = hi.max(x);
                        contact = contact || in_static(s, x) || (*airborne && x.y < *ground);
                    }
                    boxes.push((lo - Vec3::splat(*range), hi + Vec3::splat(*range)));
                    at += n * 12;
                    for (k, &r0) in rest0.iter().enumerate() {
                        let r = f(w[at + k * 8]);
                        let is_broken = f(w[at + k * 8 + 6]) != 0.0;
                        let beam = engine_webgpu::GpuBeam {
                            broken: is_broken,
                            rest: r,
                            rest0: r0,
                        };
                        broken += is_broken as u32;
                        bent += beam.bent() as u32;
                        rest.push(if is_broken { -r.abs().max(1e-6) } else { r });
                    }
                    at += rest0.len() * 8;
                }
                for a in 0..boxes.len() {
                    for b in a + 1..boxes.len() {
                        let (alo, ahi) = boxes[a];
                        let (blo, bhi) = boxes[b];
                        if !(ahi.cmplt(blo).any() || bhi.cmplt(alo).any()) {
                            contact = true;
                        }
                    }
                }
                let out = js_sys::Object::new();
                let set =
                    |k: &str, v: JsValue| js_sys::Reflect::set(&out, &JsValue::from_str(k), &v);
                set("step", JsValue::from_f64(step))?;
                set("pos", js_sys::Float32Array::from(pos.as_slice()).into())?;
                set("rest", js_sys::Float32Array::from(rest.as_slice()).into())?;
                set("bent", JsValue::from(bent))?;
                set("broken", JsValue::from(broken))?;
                set("contact", JsValue::from(contact))?;
                set("impact", JsValue::from(contact || broken > 0))?;
                Ok(out.into())
            },
        ))
    }
}

/// The lab's per-beam records ([`ExplicitSim::lab_beams_buffer`]), in beam-buffer
/// order over every vehicle.
pub(super) fn lab_beam_words(vehicles: &[(&ExplicitBeams, Vec3)]) -> Vec<[u32; 4]> {
    let mut out = Vec::new();
    for (sim, forward) in vehicles {
        let parts = beam_parts(sim, *forward);
        for (b, part) in sim.beams.iter().zip(parts) {
            out.push([b.k.to_bits(), b.strength.to_bits(), b.tc.to_bits(), part]);
        }
    }
    out
}
