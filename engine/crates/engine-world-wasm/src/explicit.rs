//! The explicit-beams solver in the browser (`ExplicitSim`): soft vehicles stepped by
//! `engine_webgpu::ExplicitBeamsGpu` (fixed 0.5 ms steps, the
//! `engine_softbody::explicit_beams` rules) on the World's device, so a page renders
//! them from the solver's own buffers in the same passes as the World's scene.
//!
//! The scenes are the explicit solver's test scenes: a vehicle at rest, coasting
//! with its tyres rolling, or coasting into a static wall. A spawn places the
//! vehicle with its forward axis along the yaw given (+Z at yaw 0), its node box
//! centred on the spawn point (or the middle of its front face there, `front`) and
//! its lowest node 2 cm above the flat ground.
//! Nothing here waits on the GPU: frames are submitted, and reads resolve later
//! (`mapAsync`).

use crate::{err, gpu_buffer, WebWorld};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, NODE_IMMOVABLE, NODE_TYRE,
};
use engine_softbody::SoftVehicleDesc;
use engine_webgpu::ExplicitBeamsGpu;
use glam::{Quat, Vec3};
use wasm_bindgen::prelude::*;

/// Steps per second.
const RATE: f64 = 2000.0;
/// The most steps one frame may take (the GPU host's parameter copies).
const MAX_FRAME_STEPS: u32 = 128;
/// Frames timed between reads of the GPU times.
const TIMED_FRAMES: u32 = 256;

fn js(v: serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

/// One spawned vehicle: its CPU copy at spawn (flags, tyres, beam ends) and what
/// the page reads.
struct Spawned {
    name: String,
    sim: ExplicitBeams,
    /// The node the pose reads follow (structure near the node box's centre at spawn).
    reference: u32,
    forward: Vec3,
    warnings: Vec<String>,
}

/// Explicit-beams vehicles on the World's device (`WebWorld::explicit_sim`).
#[wasm_bindgen]
pub struct ExplicitSim {
    gpu: ExplicitBeamsGpu,
    device: wgpu::Device,
    queue: wgpu::Queue,
    vehicles: Vec<Spawned>,
    boxes: Vec<(Vec3, Vec3)>,
    /// Render records: per collision triangle (nodes, vehicle) and its rest edges;
    /// per beam (ends, vehicle | tyre << 16, spawn rest bits).
    tris: wgpu::Buffer,
    beam_recs: wgpu::Buffer,
    tri_count: u32,
    steps: u64,
    timed: bool,
}

/// The spawn's vehicle: `.truck` text, JBeam files (and `.pc`), or a desc.
fn build(
    v: &serde_json::Value,
    settings: ExplicitBeamsSettings,
) -> Result<(ExplicitBeams, SoftVehicleDesc, Vec<String>), String> {
    if let Some(text) = v.get("truck").and_then(|t| t.as_str()) {
        let sim = ExplicitBeams::from_truck(text, Vec3::ZERO, Vec3::ZERO, settings)?;
        return Ok((sim, engine_softbody::truck::load_truck(text), Vec::new()));
    }
    if let Some(files) = v.get("jbeam").and_then(|t| t.as_array()) {
        let files: Vec<(String, String)> = files
            .iter()
            .map(|f| {
                let name = f.get(0).and_then(|x| x.as_str()).unwrap_or("").to_string();
                let text = f.get(1).and_then(|x| x.as_str()).unwrap_or("").to_string();
                (name, text)
            })
            .collect();
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let pc = v.get("pc").and_then(|t| t.as_str());
        let (sim, text, warnings) =
            ExplicitBeams::from_jbeam(&refs, pc, Vec3::ZERO, Vec3::ZERO, settings)?;
        return Ok((sim, engine_softbody::truck::load_truck(&text), warnings));
    }
    if let Some(d) = v.get("desc") {
        let desc: SoftVehicleDesc =
            serde_json::from_value(d.clone()).map_err(|e| format!("desc: {e}"))?;
        let sim = ExplicitBeams::new(&desc, Vec3::ZERO, Vec3::ZERO, settings);
        return Ok((sim, desc, Vec::new()));
    }
    Err("a vehicle needs `truck`, `jbeam` or `desc`".into())
}

/// Turn every node about the vertical axis through the origin by `angle`. A whole
/// number of quarter turns permutes and negates coordinates exactly.
fn turn(sim: &mut ExplicitBeams, angle: f32) {
    let quarter = angle / std::f32::consts::FRAC_PI_2;
    let k = quarter.round();
    let exact = (quarter - k).abs() < 1e-6;
    let q = Quat::from_rotation_y(angle);
    let rot = |p: Vec3| -> Vec3 {
        if !exact {
            return q * p;
        }
        // (x, z) → (z, −x) per quarter turn (+Y rotation).
        match (k as i32).rem_euclid(4) {
            0 => p,
            1 => Vec3::new(p.z, p.y, -p.x),
            2 => Vec3::new(-p.x, p.y, -p.z),
            _ => Vec3::new(-p.z, p.y, p.x),
        }
    };
    if exact && (k as i32).rem_euclid(4) == 0 {
        return;
    }
    for n in &mut sim.nodes {
        n.pos = rot(n.pos);
        n.vel = rot(n.vel);
    }
}

/// Node box (min, max).
fn bounds(sim: &ExplicitBeams) -> (Vec3, Vec3) {
    sim.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    })
}

/// Every node at `velocity` (immovable ones still), tyres rolling along it.
fn set_moving(sim: &mut ExplicitBeams, velocity: Vec3) {
    for n in &mut sim.nodes {
        n.vel = if n.flags & NODE_IMMOVABLE != 0 {
            Vec3::ZERO
        } else {
            velocity
        };
    }
    if velocity != Vec3::ZERO {
        sim.roll_tyres(velocity);
    }
}

fn num(v: &serde_json::Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}

#[wasm_bindgen]
impl WebWorld {
    /// Explicit-beams vehicles on this world's device. `spec`: `{ vehicles: [{
    /// truck | jbeam: [[name, text], …] (+ pc) | desc, at: [x, z], yaw, front }], scene:
    /// 'idle' | 'coast' | 'wall', speed (m/s, coast and wall), wallGap (m, default
    /// 1 + 0.5·speed), drag, turbulence, fastInvSqrt (default true), timing
    /// (default true: GPU timestamps when the device has them) }`.
    #[wasm_bindgen(js_name = explicitSim)]
    pub fn explicit_sim(&self, spec: JsValue) -> Result<ExplicitSim, JsValue> {
        let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
        ExplicitSim::new(self.world.device(), self.world.queue(), &spec).map_err(err)
    }
}

impl ExplicitSim {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spec: &serde_json::Value,
    ) -> Result<Self, String> {
        let flag = |k: &str| spec.get(k).and_then(|x| x.as_bool()).unwrap_or(true);
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: flag("fastInvSqrt"),
            drag: flag("drag"),
            turbulence: flag("turbulence") && flag("drag"),
        };
        let scene = spec.get("scene").and_then(|s| s.as_str()).unwrap_or("idle");
        if !matches!(scene, "idle" | "coast" | "wall") {
            return Err(format!(
                "unknown explicit scene {scene}: idle, coast or wall"
            ));
        }
        let speed = if scene == "idle" {
            0.0
        } else {
            num(spec, "speed", 27.78) as f32
        };
        let list = spec
            .get("vehicles")
            .and_then(|v| v.as_array())
            .ok_or("spec.vehicles: an array of vehicles")?;
        if list.is_empty() {
            return Err("spec.vehicles is empty".into());
        }
        let mut vehicles = Vec::new();
        let mut boxes = Vec::new();
        for v in list {
            let (mut sim, desc, warnings) = build(v, settings)?;
            if sim.nodes.is_empty() {
                return Err("the vehicle has no nodes".into());
            }
            // Its forward along the spawn yaw (+Z at 0).
            let f = engine_softbody::pack::forward_axis(&desc);
            let yaw = num(v, "yaw", 0.0) as f32;
            turn(&mut sim, yaw - f.x.atan2(f.z));
            let forward = Quat::from_rotation_y(yaw) * Vec3::Z;
            let at = v.get("at").and_then(|a| a.as_array());
            let ax = at
                .and_then(|a| a.first())
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0) as f32;
            let az = at
                .and_then(|a| a.get(1))
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0) as f32;
            // `front`: the spawn point is the middle of the front face (along the
            // forward axis), else the node box's centre.
            let at3 = Vec3::new(ax, 0.0, az);
            let lateral = Vec3::Y.cross(forward);
            let span = |axis: Vec3| {
                sim.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| {
                    (a.min(n.pos.dot(axis)), b.max(n.pos.dot(axis)))
                })
            };
            let (f0, f1) = span(forward);
            let (l0, l1) = span(lateral);
            let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            let front = v.get("front").and_then(|x| x.as_bool()).unwrap_or(false);
            let along = at3.dot(forward) - if front { f1 } else { 0.5 * (f0 + f1) };
            let across = at3.dot(lateral) - 0.5 * (l0 + l1);
            let shift = forward * along + lateral * across + Vec3::Y * (0.02 - lowest);
            for n in &mut sim.nodes {
                n.pos += shift;
            }
            let (lo, hi) = bounds(&sim);
            let centre = 0.5 * (lo + hi);
            // The camera's reference: of the non-tyre nodes nearest the centre (a
            // tenth, at least 8), the one with the most beams (body structure,
            // not a panel that may come off).
            let mut near: Vec<u32> = (0..sim.nodes.len() as u32)
                .filter(|&i| sim.nodes[i as usize].flags & NODE_TYRE == 0)
                .collect();
            near.sort_by(|&a, &b| {
                let da = sim.nodes[a as usize].pos.distance_squared(centre);
                let db = sim.nodes[b as usize].pos.distance_squared(centre);
                da.total_cmp(&db)
            });
            near.truncate((near.len() / 10).max(8));
            let reference = near
                .iter()
                .copied()
                .max_by_key(|&i| (sim.incidence[i as usize].len(), std::cmp::Reverse(i)))
                .unwrap_or(0);
            set_moving(&mut sim, forward * speed);
            if scene == "wall" {
                // A 4 m wall across the forward axis (snapped to x or z), `wallGap` ahead
                // of the front, a metre wider than the vehicle each side.
                let gap = num(spec, "wallGap", 1.0 + 0.5 * f64::from(speed)) as f32;
                let along_z = forward.z.abs() >= forward.x.abs();
                let dir = if along_z {
                    Vec3::Z * forward.z.signum()
                } else {
                    Vec3::X * forward.x.signum()
                };
                let front = sim
                    .nodes
                    .iter()
                    .map(|n| n.pos.dot(dir))
                    .fold(f32::MIN, f32::max);
                let half_t = 0.5;
                let side = 0.5 * if along_z { hi.x - lo.x } else { hi.z - lo.z } + 1.0;
                let c = dir * (front + gap + half_t)
                    + Vec3::Y * 2.0
                    + if along_z {
                        Vec3::X * centre.x
                    } else {
                        Vec3::Z * centre.z
                    };
                let half = if along_z {
                    Vec3::new(side, 2.0, half_t)
                } else {
                    Vec3::new(half_t, 2.0, side)
                };
                sim.boxes.push(StaticBox::centered(c, half));
                boxes.push((c, half));
            }
            vehicles.push(Spawned {
                name: desc.name.clone(),
                sim,
                reference,
                forward,
                warnings,
            });
        }
        let sims: Vec<ExplicitBeams> = vehicles.iter().map(|v| v.sim.clone()).collect();
        let mut gpu = ExplicitBeamsGpu::new(device, queue, &sims);
        let timed = spec.get("timing").and_then(|x| x.as_bool()).unwrap_or(true)
            && gpu.enable_frame_timing(TIMED_FRAMES);

        // Render records.
        let mut tris: Vec<[u32; 4]> = Vec::new();
        let mut recs: Vec<[u32; 4]> = Vec::new();
        for (vi, v) in vehicles.iter().enumerate() {
            let off = gpu.node_offset(vi);
            let p = |i: u32| v.sim.nodes[i as usize].pos;
            for t in &v.sim.triangles {
                let [a, b, c] = t.nodes;
                tris.push([off + a, off + b, off + c, vi as u32]);
                tris.push([
                    p(a).distance(p(b)).to_bits(),
                    p(b).distance(p(c)).to_bits(),
                    p(c).distance(p(a)).to_bits(),
                    0,
                ]);
            }
            let tyre = |i: u32| v.sim.nodes[i as usize].flags & NODE_TYRE != 0;
            for b in &v.sim.beams {
                let t = (tyre(b.a) || tyre(b.b)) as u32;
                recs.push([off + b.a, off + b.b, vi as u32 | t << 16, b.rest0.to_bits()]);
            }
        }
        use wgpu::util::DeviceExt;
        let init = |label: &str, words: &[[u32; 4]]| {
            let bytes: &[u8] = if words.is_empty() {
                &[0u8; 16]
            } else {
                bytemuck::cast_slice(words)
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let tris_buf = init("explicit-render-tris", &tris);
        let recs_buf = init("explicit-render-beams", &recs);
        Ok(Self {
            gpu,
            device: device.clone(),
            queue: queue.clone(),
            tri_count: (tris.len() / 2) as u32,
            vehicles,
            boxes,
            tris: tris_buf,
            beam_recs: recs_buf,
            steps: 0,
            timed,
        })
    }
}

#[wasm_bindgen]
impl ExplicitSim {
    /// Steps per second (2,000: 0.5 ms steps).
    #[wasm_bindgen(js_name = stepRate)]
    pub fn step_rate(&self) -> f64 {
        RATE
    }

    /// The most steps one [`Self::frame`] takes.
    #[wasm_bindgen(js_name = maxFrameSteps)]
    pub fn max_frame_steps(&self) -> u32 {
        MAX_FRAME_STEPS
    }

    /// Submit one frame of `steps` fixed steps for every vehicle (never waits).
    pub fn frame(&mut self, steps: u32) -> Result<(), JsValue> {
        if steps > MAX_FRAME_STEPS {
            return Err(err(format!(
                "at most {MAX_FRAME_STEPS} steps per frame, got {steps}"
            )));
        }
        if steps > 0 {
            self.gpu.frame(steps);
            self.steps += u64::from(steps);
        }
        Ok(())
    }

    /// Steps run since spawn.
    pub fn steps(&self) -> f64 {
        self.steps as f64
    }

    /// Whether GPU frame times are recorded (the device has timestamp queries).
    pub fn timed(&self) -> bool {
        self.timed
    }

    /// The node state `GPUBuffer`: 3 `vec4<f32>` per node (position relative to its
    /// vehicle's origin and mass, velocity, force), every vehicle's nodes in order.
    #[wasm_bindgen(js_name = nodesBuffer)]
    pub fn nodes_buffer(&self) -> JsValue {
        gpu_buffer(self.gpu.nodes_buffer())
    }

    /// The beam state `GPUBuffer`: 2 `vec4<f32>` per beam, (rest length, yield
    /// thresholds, gate) and (strength, force, broken 0/1, 0).
    #[wasm_bindgen(js_name = beamsBuffer)]
    pub fn beams_buffer(&self) -> JsValue {
        gpu_buffer(self.gpu.beams_buffer())
    }

    /// The per-vehicle `GPUBuffer`: `vec4<u32>` per vehicle, its origin's bits (xyz).
    #[wasm_bindgen(js_name = miscBuffer)]
    pub fn misc_buffer(&self) -> JsValue {
        gpu_buffer(self.gpu.misc_buffer())
    }

    /// Collision triangles for rendering: 2 `vec4` per triangle, (global node
    /// indices a, b, c, vehicle) as `u32` then its spawn edge lengths ab, bc, ca as
    /// `f32` bits.
    #[wasm_bindgen(js_name = trisBuffer)]
    pub fn tris_buffer(&self) -> JsValue {
        gpu_buffer(&self.tris)
    }

    /// Beams for rendering: `vec4<u32>` per beam in beam-buffer order, (global node
    /// a, b, vehicle | 1 << 16 for a tyre beam, spawn rest length bits).
    #[wasm_bindgen(js_name = beamRecordsBuffer)]
    pub fn beam_records_buffer(&self) -> JsValue {
        gpu_buffer(&self.beam_recs)
    }

    /// `{ nodes, beams, triangles, timed, vehicles: [{ name, nodes, beams,
    /// triangles, tyres, mass, nodeOffset, beamOffset, warnings }], boxes: [[cx,
    /// cy, cz, hx, hy, hz]] }` (boxes: the static boxes, for the scene's colliders).
    pub fn layout(&self) -> JsValue {
        let vehicles: Vec<serde_json::Value> = self
            .vehicles
            .iter()
            .enumerate()
            .map(|(i, v)| {
                serde_json::json!({
                    "name": v.name, "nodes": v.sim.nodes.len(), "beams": v.sim.beams.len(),
                    "triangles": v.sim.triangles.len(), "tyres": v.sim.tyres.len(),
                    "mass": v.sim.nodes.iter().map(|n| f64::from(n.mass)).sum::<f64>(),
                    "nodeOffset": self.gpu.node_offset(i), "beamOffset": self.gpu.beam_offset(i),
                    "forward": v.forward.to_array(), "warnings": v.warnings,
                })
            })
            .collect();
        let boxes: Vec<[f32; 6]> = self
            .boxes
            .iter()
            .map(|(c, h)| [c.x, c.y, c.z, h.x, h.y, h.z])
            .collect();
        js(serde_json::json!({
            "nodes": self.vehicles.iter().map(|v| v.sim.nodes.len()).sum::<usize>(),
            "beams": self.vehicles.iter().map(|v| v.sim.beams.len()).sum::<usize>(),
            "triangles": self.tri_count,
            "timed": self.timed,
            "wide": self.gpu.is_wide(),
            "vehicles": vehicles,
            "boxes": boxes,
        }))
    }

    /// Damage of every vehicle from one copy of its beams, resolved once mapped:
    /// `[{ broken, bent, beams }]` (bent: alive with its rest length off its spawn
    /// length by over 0.1 %, as `GpuBeam::bent`).
    #[wasm_bindgen(js_name = readDamage)]
    pub fn read_damage(&self) -> js_sys::Promise {
        let buf = self.gpu.beams_buffer().clone();
        let mut ranges = Vec::new();
        let mut rest0 = Vec::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            let off = self.gpu.beam_offset(i) as usize;
            let n = v.sim.beams.len();
            ranges.push((buf.clone(), off as u64 * 32, n as u64 * 32));
            rest0.push(self.gpu.rest0()[off..off + n].to_vec());
        }
        crate::read::read_ranges_on(self.device.clone(), self.queue.clone(), ranges, move |w| {
            let mut out = Vec::new();
            let mut at = 0;
            for r0 in rest0 {
                let (mut broken, mut bent) = (0u32, 0u32);
                for (k, &rest0) in r0.iter().enumerate() {
                    let rec = &w[at + k * 8..at + k * 8 + 8];
                    let rest = f32::from_bits(rec[0]);
                    let is_broken = f32::from_bits(rec[6]) != 0.0;
                    let beam = engine_webgpu::GpuBeam {
                        broken: is_broken,
                        rest,
                        rest0,
                    };
                    broken += is_broken as u32;
                    bent += beam.bent() as u32;
                }
                at += r0.len() * 8;
                out.push(serde_json::json!({ "broken": broken, "bent": bent, "beams": r0.len() }));
            }
            Ok(js(serde_json::Value::Array(out)))
        })
    }

    /// Per vehicle its reference node's world position and velocity, resolved once
    /// mapped: 6 floats per vehicle (x, y, z, vx, vy, vz).
    #[wasm_bindgen(js_name = readPoses)]
    pub fn read_poses(&self) -> js_sys::Promise {
        let nodes = self.gpu.nodes_buffer().clone();
        let misc = self.gpu.misc_buffer().clone();
        let mut ranges = Vec::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            ranges.push((misc.clone(), i as u64 * 16, 16));
            let node = u64::from(self.gpu.node_offset(i) + v.reference);
            ranges.push((nodes.clone(), node * 48, 32));
        }
        let n = self.vehicles.len();
        crate::read::read_ranges_on(self.device.clone(), self.queue.clone(), ranges, move |w| {
            let f = |x: u32| f32::from_bits(x);
            let mut out = Vec::with_capacity(n * 6);
            for k in 0..n {
                let r = &w[k * 12..k * 12 + 12];
                out.extend_from_slice(&[
                    f(r[0]) + f(r[4]),
                    f(r[1]) + f(r[5]),
                    f(r[2]) + f(r[6]),
                    f(r[8]),
                    f(r[9]),
                    f(r[10]),
                ]);
            }
            Ok(js_sys::Float32Array::from(out.as_slice()).into())
        })
    }

    /// GPU milliseconds of each frame since the last read (timestamps around the
    /// frame's passes), resolved once mapped; empty without timestamps or frames.
    #[wasm_bindgen(js_name = readFrameTimes)]
    pub fn read_frame_times(&mut self) -> js_sys::Promise {
        let period = self.gpu.timestamp_period_ms();
        let Some((staging, used)) = self.gpu.take_frame_times() else {
            return js_sys::Promise::resolve(&JsValue::from(
                js_sys::Float64Array::new_with_length(0),
            ));
        };
        wasm_bindgen_futures::future_to_promise(async move {
            let (tx, rx) = futures_channel::oneshot::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rx.await
                .map_err(|_| err("readFrameTimes: map dropped"))?
                .map_err(|e| err(format!("readFrameTimes: {e}")))?;
            let ms: Vec<f64> = {
                let data = staging.slice(..).get_mapped_range().expect("mapped");
                let ticks: &[u64] = bytemuck::cast_slice(&data[..used as usize * 16]);
                ticks
                    .chunks(2)
                    .map(|p| p[1].saturating_sub(p[0]) as f64 * period)
                    .collect()
            };
            staging.unmap();
            Ok(js_sys::Float64Array::from(ms.as_slice()).into())
        })
    }

    /// Launch every vehicle at `speed` m/s along its spawn forward axis, tyres
    /// rolling (`ExplicitBeams::roll_tyres`) about their axles where they are now:
    /// reads the node positions, then writes every node's velocity. Resolves once
    /// written (steps taken meanwhile keep their own velocities until then).
    pub fn launch(&self, speed: f32) -> js_sys::Promise {
        let nodes = self.gpu.nodes_buffer().clone();
        let mut ranges = Vec::new();
        let mut sims = Vec::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            let off = u64::from(self.gpu.node_offset(i));
            ranges.push((nodes.clone(), off * 48, v.sim.nodes.len() as u64 * 48));
            sims.push((off, v.sim.clone(), v.forward));
        }
        let queue = self.queue.clone();
        crate::read::read_ranges_on(self.device.clone(), self.queue.clone(), ranges, move |w| {
            let mut at = 0;
            for (off, mut sim, forward) in sims {
                for n in &mut sim.nodes {
                    n.pos = Vec3::new(
                        f32::from_bits(w[at]),
                        f32::from_bits(w[at + 1]),
                        f32::from_bits(w[at + 2]),
                    );
                    at += 12;
                }
                set_moving(&mut sim, forward * speed);
                for (k, n) in sim.nodes.iter().enumerate() {
                    let offset = ((off + k as u64) * 3 + 1) * 16;
                    queue.write_buffer(&nodes, offset, bytemuck::cast_slice(&n.vel.to_array()));
                }
            }
            Ok(JsValue::UNDEFINED)
        })
    }
}

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
