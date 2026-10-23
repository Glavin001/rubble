//! Soft-body (node–beam) vehicles for the browser: `engine_softbody::SoftHost`
//! with the same op-stream backend as the vehicle host, plus the `.truck`
//! importer and the procedural test content.

use crate::{err, parse, Recorder, WasmVehicleHost};
use engine_softbody::procedural::{box_truck, car_truck, CarParams};
use engine_softbody::truck::load_truck;
use engine_softbody::{
    SoftCapacity, SoftHandle, SoftHost, SoftSettings, SoftVehicleDesc, SpawnOptions,
};
use serde_json::json;
use wasm_bindgen::prelude::*;

/// [maxBodies, maxBeams, maxParticles, maxColors, maxGroups, maxTriangles, maxObstacles].
pub(crate) fn soft_capacity(v: &[u32]) -> SoftCapacity {
    let d = SoftCapacity::default();
    let g = |i: usize, def: u32| v.get(i).copied().unwrap_or(def);
    SoftCapacity {
        max_bodies: g(0, d.max_bodies),
        max_beams: g(1, d.max_beams),
        max_particles: g(2, d.max_particles),
        max_colors: g(3, d.max_colors),
        max_groups: g(4, d.max_groups),
        max_triangles: g(5, d.max_triangles),
        max_obstacles: g(6, d.max_obstacles),
    }
}

/// Parse a Rigs of Rods `.truck` file into a `SoftVehicleDesc` (JSON), masses
/// distributed and the file's warnings listed.
#[wasm_bindgen]
pub fn soft_load_truck(src: &str) -> String {
    serde_json::to_string(&load_truck(src)).expect("desc serialises")
}

/// The procedural hatchback as `.truck` text (`deform`, `strength` in N; 0 = default).
#[wasm_bindgen]
pub fn soft_procedural_car(deform: f32, strength: f32) -> String {
    let d = CarParams::default();
    car_truck(&CarParams {
        deform: if deform > 0.0 { deform } else { d.deform },
        strength: if strength > 0.0 { strength } else { d.strength },
        ..d
    })
}

/// A braced lattice crate as `.truck` text.
#[wasm_bindgen]
pub fn soft_box_truck(
    n: u32,
    step: f32,
    mass: f32,
    spring: f32,
    deform: f32,
    strength: f32,
) -> String {
    let n = n.max(2) as usize;
    box_truck(n, n, n, step, mass, spring, deform, strength)
}

/// Arena layout and record constants (JSON).
#[wasm_bindgen]
pub fn soft_constants() -> String {
    use engine_softbody::layout::*;
    json!({
        "BEAM_STRIDE": BEAM_STRIDE, "BEAM_BROKEN": BEAM_BROKEN, "BEAM_YIELDED": BEAM_YIELDED,
        "BEAM_INVISIBLE": BEAM_INVISIBLE, "BODY_PARTICLE": BODY_PARTICLE, "EVENT_BEAM_BREAK": EVENT_BEAM_BREAK,
    })
    .to_string()
}

#[wasm_bindgen]
pub struct WasmSoftHost {
    host: SoftHost,
}

#[wasm_bindgen]
impl WasmSoftHost {
    #[wasm_bindgen(constructor)]
    pub fn new(cap: &[u32]) -> WasmSoftHost {
        WasmSoftHost {
            host: SoftHost::new(soft_capacity(cap)),
        }
    }

    /// Region bases (float4 slots) inside gSoftData (JSON, camelCase).
    pub fn layout(&self) -> String {
        let l = self.host.capacity.layout();
        json!({"bodyBase": l.body_base, "beamBase": l.beam_base, "groupBase": l.group_base,
            "colorBase": l.color_base, "listBase": l.list_base, "adjBase": l.adj_base, "chebBase": l.cheb_base,
            "triBase": l.tri_base, "triContactBase": l.tri_contact_base, "nodeSlotBase": l.node_slot_base,
            "flagBase": l.flag_base, "obstacleBase": l.obstacle_base, "aabbBase": l.aabb_base,
            "totalSlots": l.total_slots, "bytes": self.host.capacity.bytes()})
        .to_string()
    }

    /// Spawn a soft vehicle; `counts` = [bodies, colliders, joints] already in the
    /// world. With wheels it also adds the rigid core to `vehicles`.
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        counts: &[u32],
        desc_json: &str,
        position: &[f32],
        rotation: &[f32],
        opts_json: &str,
        vehicles: &mut WasmVehicleHost,
    ) -> Result<Vec<u32>, JsError> {
        let desc: SoftVehicleDesc = parse(desc_json)?;
        let opts: SpawnOptions = if opts_json.is_empty() {
            SpawnOptions::default()
        } else {
            parse(opts_json)?
        };
        let mut r = Recorder::new([counts[0], counts[1], counts[2]], None);
        let pos = [position[0], position[1], position[2]];
        let rot = [rotation[0], rotation[1], rotation[2], rotation[3]];
        self.host
            .add(&mut r, Some(&mut vehicles.host), &desc, pos, rot, &opts)
            .map_err(err)?;
        Ok(r.ops)
    }

    /// Register a thin static obstacle (sphere or capsule collider).
    pub fn add_obstacle(&mut self, collider: u32) -> Result<Vec<u32>, JsError> {
        let mut r = Recorder::new([0; 3], None);
        self.host.add_obstacle(&mut r, collider).map_err(err)?;
        Ok(r.ops)
    }

    /// Replace the solver settings (JSON `SoftSettings`, missing fields default).
    pub fn set_settings(&mut self, json: &str) -> Result<Vec<u32>, JsError> {
        let st: SoftSettings = parse(json)?;
        let mut r = Recorder::new([0; 3], None);
        self.host.set_settings(&mut r, st);
        Ok(r.ops)
    }

    pub fn settings(&self) -> String {
        serde_json::to_string(&self.host.settings).expect("settings serialise")
    }

    pub fn set_steer(&mut self, h: u32, steer: f32) -> Vec<u32> {
        let mut r = Recorder::new([0; 3], None);
        self.host.set_steer(&mut r, SoftHandle(h), steer);
        r.ops
    }

    /// Plan inputs: [bodies, beams, colours, max colour, groups + 1, particles,
    /// chebyshev, particles total, triangles, contacts] (the dual flag is `dual()`).
    pub fn counts(&self) -> Vec<u32> {
        let c = self.host.counts();
        vec![
            c.bodies,
            c.beams,
            c.colors,
            c.max_color,
            c.groups + 1,
            c.particles,
            u32::from(self.host.settings.chebyshev_rho > 0.0),
            c.particles_total,
            c.triangles,
            u32::from(c.contacts),
        ]
    }

    /// The soft dual pass has work (plan input counts[16]).
    pub fn dual(&self) -> bool {
        self.host.counts().dual
    }

    #[wasm_bindgen(getter)]
    pub fn body_count(&self) -> u32 {
        self.host.entries.len() as u32
    }

    /// Bookkeeping the demo renders from (JSON): bodies, beams (global particle
    /// indices), triangles with rest edge lengths, the rigid core.
    pub fn entry(&self, h: u32) -> String {
        let e = self.host.entry(SoftHandle(h));
        let d = &e.desc;
        let tris: Vec<[u32; 3]> = d
            .triangles
            .iter()
            .map(|t| t.nodes.map(|n| e.first_body + n))
            .collect();
        let rest: Vec<[f32; 3]> = d
            .triangles
            .iter()
            .map(|t| {
                let p = |i: u32| glam::Vec3::from_array(d.nodes[i as usize].position);
                let [a, b, c] = t.nodes;
                [
                    (p(a) - p(b)).length(),
                    (p(b) - p(c)).length(),
                    (p(c) - p(a)).length(),
                ]
            })
            .collect();
        json!({
            "name": d.name,
            "firstBody": e.first_body, "particles": e.particles,
            "firstBeam": e.first_beam, "beams": e.beams,
            "firstTri": e.first_tri, "tris": e.tris,
            "triangles": tris, "triRest": rest,
            "core": e.core.map(|(v, chassis)| json!({"vehicle": v.0, "chassis": chassis})),
            "mass": d.total_mass(),
            "nodes": d.nodes.len(),
            "warnings": d.warnings,
        })
        .to_string()
    }

    pub fn save(&self) -> WasmSoftHost {
        WasmSoftHost {
            host: self.host.clone(),
        }
    }

    pub fn restore(&mut self, snap: &WasmSoftHost) {
        self.host = snap.host.clone();
    }
}

/// Load a BeamNG JBeam vehicle: `files_json` = [[name, source], …] of its
/// `.jbeam` files, `pc` an optional `.pc` configuration. Returns the
/// `SoftVehicleDesc` (JSON) or the loader's error.
#[wasm_bindgen]
pub fn soft_load_jbeam(files_json: &str, pc: Option<String>) -> Result<String, JsError> {
    let files: Vec<(String, String)> = parse(files_json)?;
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let d = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).map_err(err)?;
    Ok(serde_json::to_string(&d).expect("desc serialises"))
}

#[wasm_bindgen]
impl WasmSoftHost {
    /// Visual mesh of soft body `h` (flexbody skinning): its skin triangles
    /// subdivided and rounded (`bulge` × edge length), each vertex bound to the
    /// frame of the skin triangle it came from. Returns `[vertexCount, triCount, bindings…, indices…]`
    /// with bindings as 8 words per vertex: centre, x, y particle (global body
    /// indices), soft body, then the frame coordinates and the frame's longest
    /// rest edge (f32 bits). Unbound vertices collapse onto their nearest node.
    pub fn flexbody(&self, h: u32, bulge: f32) -> Vec<u32> {
        use engine_softbody::flexbody::{bind_to_triangles, rounded_skin};
        use glam::Vec3;
        let e = self.host.entry(SoftHandle(h));
        let nodes: Vec<Vec3> = e
            .desc
            .nodes
            .iter()
            .map(|n| Vec3::from_array(n.position))
            .collect();
        let tris: Vec<[u32; 3]> = e.desc.triangles.iter().map(|t| t.nodes).collect();
        let (verts, faces, sources) = rounded_skin(&nodes, &tris, bulge);
        let binds = bind_to_triangles(&verts, &nodes, &sources);
        let mut out = vec![verts.len() as u32, faces.len() as u32];
        for (v, b) in verts.iter().zip(&binds) {
            match b {
                Some(b) => {
                    let [c, x, y] = b.nodes;
                    let rest = (nodes[x as usize] - nodes[c as usize])
                        .length()
                        .max((nodes[y as usize] - nodes[c as usize]).length());
                    out.extend_from_slice(&[
                        e.first_body + c,
                        e.first_body + x,
                        e.first_body + y,
                        h,
                    ]);
                    out.extend(b.coords.iter().map(|f| f.to_bits()));
                    out.push(rest.to_bits());
                }
                None => {
                    let c = nodes
                        .iter()
                        .enumerate()
                        .min_by(|a, b| {
                            (*a.1 - *v)
                                .length_squared()
                                .total_cmp(&(*b.1 - *v).length_squared())
                        })
                        .map_or(0, |(i, _)| i as u32);
                    out.extend_from_slice(&[
                        e.first_body + c,
                        e.first_body + c,
                        e.first_body + c,
                        h,
                        0,
                        0,
                        0,
                        1f32.to_bits(),
                    ]);
                }
            }
        }
        out.extend(faces.iter().flatten());
        out
    }
}

/// `SoftSettings::calibrated_for(hz)` as JSON (threshold scale fitted per step rate).
#[wasm_bindgen]
pub fn soft_calibrated_settings(hz: f32) -> String {
    serde_json::to_string(&SoftSettings::calibrated_for(hz)).expect("settings serialise")
}
