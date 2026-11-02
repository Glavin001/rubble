//! Soft bodies for renderers and HUDs: skin triangles, render-mesh bindings,
//! flexbody meshes, the soft arena buffer, obstacles, and reads of damage and
//! rigid mode that resolve later (never waiting in place).

use crate::{bits, err, res, WebWorld};
use engine_softbody::layout::{BEAM_STRIDE, BODY_STRIDE};
use rubble::*;
use wasm_bindgen::prelude::*;

fn js(v: serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

impl WebWorld {
    fn live_soft_entry(&self, soft: u32) -> Result<&engine_softbody::host::SoftEntry, JsValue> {
        if !self.world.sim().has_soft() {
            return Err(err("the world has no soft arena"));
        }
        self.world
            .sim()
            .soft_host()
            .entries
            .get(soft as usize)
            .ok_or_else(|| err(format!("no soft body {soft}")))
    }
}

#[wasm_bindgen]
impl WebWorld {
    /// Soft bodies spawned (0 without a soft arena).
    #[wasm_bindgen(js_name = softCount)]
    pub fn soft_count(&self) -> u32 {
        if self.world.sim().has_soft() {
            self.world.sim().soft_host().entries.len() as u32
        } else {
            0
        }
    }

    /// Let soft skins collide with this rigid shape (`World::soft_add_obstacle`).
    #[wasm_bindgen(js_name = softAddObstacle)]
    pub fn soft_add_obstacle(&mut self, shape: f64) -> Result<(), JsValue> {
        res(self
            .world
            .soft_add_obstacle(ShapeId::from_bits(bits(shape))))
    }

    /// The soft body's skin triangles, 3 body slots each (its particles).
    #[wasm_bindgen(js_name = softSkinTriangles)]
    pub fn soft_skin_triangles(&self, soft: u32) -> Result<Vec<u32>, JsValue> {
        let e = self.live_soft_entry(soft)?;
        Ok(e.desc
            .triangles
            .iter()
            .flat_map(|t| t.nodes.map(|n| e.first_body + n))
            .collect())
    }

    /// Rest edge lengths of each skin triangle (ab, bc, ca), from the desc's
    /// node positions.
    #[wasm_bindgen(js_name = softSkinRest)]
    pub fn soft_skin_rest(&self, soft: u32) -> Result<Vec<f32>, JsValue> {
        let e = self.live_soft_entry(soft)?;
        let d = &e.desc;
        let p = |i: u32| glam::Vec3::from_array(d.nodes[i as usize].position);
        Ok(d.triangles
            .iter()
            .flat_map(|t| {
                let [a, b, c] = t.nodes;
                [
                    (p(a) - p(b)).length(),
                    (p(b) - p(c)).length(),
                    (p(c) - p(a)).length(),
                ]
            })
            .collect())
    }

    /// Bind render-mesh vertices (xyz triples, the vehicle's own frame) to the
    /// soft body, only to part `part`'s nodes when given (`SoftHost::bind_mesh`):
    /// 8 words per vertex (`SkinBinding::to_words`) for the engine's skinning
    /// kernel (`soft_skin.wgsl`) or a renderer's own shader.
    #[wasm_bindgen(js_name = softBindMesh)]
    pub fn soft_bind_mesh(
        &self,
        soft: u32,
        vertices: &[f32],
        part: Option<String>,
    ) -> Result<Vec<u32>, JsValue> {
        self.live_soft_entry(soft)?;
        let verts: Vec<glam::Vec3> = vertices
            .chunks_exact(3)
            .map(|c| glam::Vec3::new(c[0], c[1], c[2]))
            .collect();
        let b = self
            .world
            .sim()
            .soft_host()
            .bind_mesh(SoftHandle(soft), &verts, part.as_deref())
            .map_err(err)?;
        Ok(b.iter().flat_map(|b| b.to_words(0)).collect())
    }

    /// The soft body's flexbody render mesh (`engine_softbody::flexbody::flexbody_words`):
    /// `[vertexCount, triCount, bindings (8 words per vertex)…, indices…]`.
    #[wasm_bindgen(js_name = softFlexbody)]
    pub fn soft_flexbody(&self, soft: u32, bulge: f32) -> Result<Vec<u32>, JsValue> {
        let e = self.live_soft_entry(soft)?;
        let nodes: Vec<glam::Vec3> = e
            .desc
            .nodes
            .iter()
            .map(|n| glam::Vec3::from_array(n.position))
            .collect();
        let tris: Vec<[u32; 3]> = e.desc.triangles.iter().map(|t| t.nodes).collect();
        Ok(engine_softbody::flexbody::flexbody_words(
            &nodes,
            &tris,
            e.first_body,
            soft,
            bulge,
        ))
    }

    /// The soft arena `GPUBuffer` (binding 32), for renderers (beam views).
    #[wasm_bindgen(js_name = softBuffer)]
    pub fn soft_buffer(&self) -> JsValue {
        crate::gpu_buffer(self.world.sim().soft_buffer())
    }

    /// Region bases (16-byte slots) of the soft arena: `{ bodyBase, beamBase,
    /// triBase, totalSlots }`.
    #[wasm_bindgen(js_name = softLayout)]
    pub fn soft_layout(&self) -> Result<JsValue, JsValue> {
        if !self.world.sim().has_soft() {
            return Err(err("the world has no soft arena"));
        }
        let l = self.world.sim().soft_host().capacity.layout();
        Ok(js(serde_json::json!({
            "bodyBase": l.body_base, "beamBase": l.beam_base, "triBase": l.tri_base, "totalSlots": l.total_slots,
        })))
    }

    /// Rigid mode of every soft body (`SoftSettings::rigid_mode`: 1 riding its
    /// core with the particle solve skipped, 0 soft), resolved once mapped.
    #[wasm_bindgen(js_name = readSoftModes)]
    pub fn read_soft_modes(&self) -> js_sys::Promise {
        let n = self.soft_count();
        let mut ranges = Vec::new();
        if n > 0 {
            let l = self.world.sim().soft_host().capacity.layout();
            let buf = self.world.sim().soft_buffer().clone();
            for h in 0..n {
                let slot = l.body_base + BODY_STRIDE * h + 3;
                ranges.push((buf.clone(), u64::from(slot) * 16, 16));
            }
        }
        crate::read::read_ranges_with(&self.world, ranges, |w| {
            let modes: Vec<u32> = w.chunks_exact(4).map(|c| c[0]).collect();
            Ok(js_sys::Uint32Array::from(modes.as_slice()).into())
        })
    }

    /// Damage of every soft body from one copy of their beam records, resolved
    /// once mapped: `[{ bent, broken, beams, maxBeamStrain, plasticWork }]` in
    /// soft-body order (`engine_softbody::damage_summary` without positions).
    #[wasm_bindgen(js_name = readSoftDamage)]
    pub fn read_soft_damage(&self) -> js_sys::Promise {
        let n = self.soft_count();
        let mut ranges = Vec::new();
        let mut counts = Vec::new();
        if n > 0 {
            let host = self.world.sim().soft_host();
            let l = host.capacity.layout();
            let buf = self.world.sim().soft_buffer().clone();
            for e in &host.entries {
                let slot = l.beam_base + BEAM_STRIDE * e.first_beam;
                ranges.push((
                    buf.clone(),
                    u64::from(slot) * 16,
                    u64::from(BEAM_STRIDE * e.beams) * 16,
                ));
                counts.push(e.beams as usize);
            }
        }
        crate::read::read_ranges_with(&self.world, ranges, move |w| {
            let words = (BEAM_STRIDE * 4) as usize;
            let mut at = 0;
            let mut out = Vec::with_capacity(counts.len());
            for n in counts {
                let beams: Vec<engine_softbody::model::Beam> = w[at..at + n * words]
                    .chunks_exact(words)
                    .map(engine_softbody::model::Beam::from_words)
                    .collect();
                at += n * words;
                let d = engine_softbody::damage_summary(&beams, None, None);
                out.push(serde_json::json!({
                    "bent": d.bent, "broken": d.broken, "beams": d.beams,
                    "maxBeamStrain": d.max_beam_strain, "plasticWork": d.plastic_work,
                }));
            }
            Ok(js(serde_json::Value::Array(out)))
        })
    }

    /// Every collider record's material word in use (slot in the low 16 bits,
    /// the collision group in the high 16), from the host's mirror: renderers
    /// pick what to draw (wheel rims and soft nodes are not).
    #[wasm_bindgen(js_name = colliderMaterials)]
    pub fn collider_materials(&self) -> Vec<u32> {
        let sim = self.world.sim();
        (0..sim.collider_count())
            .map(|c| sim.collider_record(c).material)
            .collect()
    }
}
