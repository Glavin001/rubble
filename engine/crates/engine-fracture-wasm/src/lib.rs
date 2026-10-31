//! wasm-bindgen surface of `engine-fracture` for the browser fracture lab.
//!
//! The JS side (`engine-webgpu/fracture/wasm.mjs`) only loads this module and
//! adapts types; asset parsing, scene building, hull cooking and the per-tick
//! dispatch plan all run here, from the same Rust the Metal viewer uses.
//! Large streams cross as typed arrays; small records as plain JS objects.

use engine_fracture::cuboid::{build_cuboid_house, CuboidOptions};
use engine_fracture::guidance as near_miss;
use engine_schema::guidance::{GuidanceSettings, ZoneDesc};
use engine_fracture::plan::{self, Limits, Planner};
use engine_fracture::samples::{self, Layout};
use engine_fracture::scene::{self, Collider};
use engine_fracture::{FractureAsset, SAMPLES};
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

fn to_js<T: Serialize + ?Sized>(v: &T) -> Result<JsValue, JsError> {
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(err)
}

fn from_js<T: serde::de::DeserializeOwned>(v: JsValue) -> Result<T, JsError> {
    serde_wasm_bindgen::from_value(v).map_err(err)
}

/// Merge a partial JS object into a serde value (`{...a, ...b}`).
fn merge<T: Serialize + serde::de::DeserializeOwned>(
    base: &T,
    partial: JsValue,
) -> Result<T, JsError> {
    let mut a = serde_json::to_value(base).map_err(err)?;
    let b: serde_json::Value = from_js(partial)?;
    if let (Some(a), Some(b)) = (a.as_object_mut(), b.as_object()) {
        for (k, v) in b {
            a.insert(k.clone(), v.clone());
        }
    }
    serde_json::from_value(a).map_err(err)
}

/// A parsed SHARDWRIGHT_fracture v1 GLB.
#[wasm_bindgen]
pub struct Asset(FractureAsset);

#[wasm_bindgen]
impl Asset {
    /// Parse (and bounds-check) a fracture GLB.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Asset, JsError> {
        engine_fracture::parse_fracture_glb(bytes)
            .map(Asset)
            .map_err(err)
    }
    /// Hierarchy levels present, ascending.
    pub fn levels(&self) -> Vec<u8> {
        self.0.levels()
    }
    #[wasm_bindgen(js_name = levelFragments)]
    pub fn level_fragments(&self, level: u8) -> u32 {
        self.0.level_fragments(level) as u32
    }
    #[wasm_bindgen(getter)]
    pub fn bytes(&self) -> u32 {
        self.0.bytes as u32
    }
    #[wasm_bindgen(getter, js_name = materialCount)]
    pub fn material_count(&self) -> u32 {
        self.0.material_count() as u32
    }
    /// Name, seed and counts of the physics payload.
    pub fn summary(&self) -> Result<JsValue, JsError> {
        let p = &self.0.physics;
        to_js(&serde_json::json!({
            "name": p.name, "seed": p.seed, "fragments": p.fragments.len(), "bonds": p.bonds.len(),
            "hulls": p.hulls.len(), "levelFragmentCount": p.level_fragment_count,
            "materials": self.0.material_count(),
            "maxHullVertices": p.hulls.iter().map(|h| h.vertices.len()).max().unwrap_or(0),
        }))
    }
    /// `{physics, materialLibrary}` JSON (the lab's metadata download).
    #[wasm_bindgen(js_name = metadataJson)]
    pub fn metadata_json(&self) -> Result<String, JsError> {
        self.0.metadata_json().map_err(err)
    }
    /// The embedded material library (`{material, interface_material, ...}`).
    pub fn library(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.library)
    }
    /// One fragment record (`{id, level, mass, materialIds, ...}`).
    pub fn fragment(&self, id: u32) -> Result<JsValue, JsError> {
        let f = self
            .0
            .physics
            .fragments
            .get(id as usize)
            .ok_or_else(|| err("fragment out of range"))?;
        to_js(f)
    }
    /// The level a sample opens at (preferred if present, else the first).
    #[wasm_bindgen(js_name = initialLevel)]
    pub fn initial_level(&self, preferred: Option<u8>) -> Option<u8> {
        samples::initial_level(&self.0, preferred)
    }
}

/// A simulated scene (bodies, colliders, joints, hull arena, render streams).
#[wasm_bindgen]
pub struct Scene(scene::Scene);

/// Scene fields other than the large arrays (those are typed-array getters).
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Meta<'a> {
    body_count: usize,
    collider_count: usize,
    joint_count: usize,
    hull_count: usize,
    hull_records: usize,
    params: &'a scene::SceneParams,
    capacity: &'a scene::Budget,
    skipped_hulls: &'a [scene::SkippedHull],
    projectile_start: u32,
    projectile_count: u32,
    projectile_radius: f64,
    projectile_mass: f64,
    projectile_speed: Option<f64>,
    projectile_arc: bool,
    level: Option<u8>,
    selected: &'a [u32],
    bond_count: u32,
    extent: f64,
    extent_x: f64,
    extent_z: f64,
    height: f64,
    center: [f64; 3],
    copies: u32,
    instance_stride: u32,
    columns: u32,
    rows: u32,
    row_pitch: f64,
    column_pitch: f64,
    row_length: f64,
    grid_width: f64,
    grid_cell_size: f64,
    camera: Option<scene::Camera>,
    row_shot: Option<scene::Ray>,
    preload: Option<&'a scene::Preload>,
    pieces: u32,
}

#[wasm_bindgen]
impl Scene {
    /// One hierarchy level of a baked asset (`maxHullVertices`: engine-wide limit 8..64).
    #[wasm_bindgen(js_name = fromAsset)]
    pub fn from_asset(
        asset: &Asset,
        level: u8,
        max_hull_vertices: Option<u32>,
    ) -> Result<Scene, JsError> {
        scene::build_scene(&asset.0, level, max_hull_vertices.map(|v| v as usize))
            .map(Scene)
            .map_err(err)
    }
    /// The procedural cuboid house; `hulls`: cook each box as an 8-vertex hull.
    #[wasm_bindgen(js_name = cuboidHouse)]
    pub fn cuboid_house(hulls: Option<bool>, pitch_deg: Option<f64>) -> Result<Scene, JsError> {
        let mut o = CuboidOptions {
            hulls: hulls.unwrap_or(false),
            ..CuboidOptions::default()
        };
        if let Some(p) = pitch_deg {
            o.pitch_deg = p;
        }
        build_cuboid_house(&o).map(Scene).map_err(err)
    }
    /// A scene pack (vibe-city / blast-stress-solver JSON; engine-fracture `pack.rs`).
    #[wasm_bindgen(js_name = fromPack)]
    pub fn from_pack(bytes: &[u8]) -> Result<Scene, JsError> {
        engine_fracture::pack::build_pack_scene(bytes)
            .map(|(s, _)| Scene(s))
            .map_err(err)
    }
    /// Collision-cost benchmark scene (`count` stacks of shape `a` under `b`).
    #[wasm_bindgen(js_name = pairs)]
    pub fn pairs(a: &str, b: &str, count: u32) -> Result<Scene, JsError> {
        engine_fracture::pairs::build_pair_scene(a, b, count, 2.5)
            .map(Scene)
            .map_err(err)
    }
    /// A scene from a plain JS object (`{bodies, colliders, joints?, hullData?,
    /// params?, capacity?, ...}`; tests and GPU conformance suites). `hullData`
    /// may be a flat Uint32Array of arena words.
    #[wasm_bindgen(js_name = fromJs)]
    pub fn from_js(value: JsValue, hull_words: Option<Vec<u32>>) -> Result<Scene, JsError> {
        let mut s: scene::Scene = from_js(value)?;
        if let Some(words) = hull_words {
            s.hull_data = words.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
        }
        if s.projectile_start == 0 {
            s.projectile_start = s.bodies.len() as u32;
        }
        if s.grid_cell_size == 0.0 {
            s.grid_cell_size = scene::grid_config(&s.colliders).0;
        }
        Ok(Scene(s))
    }
    /// Copies laid out as a row/grid (`{columns, rows}`) or suburb (`{suburb:true}`).
    pub fn layout(&self, layout: JsValue, procedural: bool) -> Result<Scene, JsError> {
        let layout: Option<Layout> = if layout.is_null() || layout.is_undefined() {
            None
        } else {
            Some(from_js(layout)?)
        };
        Ok(Scene(samples::layout_scene(
            self.0.clone(),
            layout,
            procedural,
        )))
    }
    /// `columns × rows` copies of this pack scene (one tower), `gap` metres of
    /// street between them, firing a `radius` m, `mass` kg ball (default: the
    /// pack's); the joints are kept (engine-fracture `towers.rs`).
    pub fn towers(&self, columns: u32, rows: u32, gap: f64, radius: Option<f64>, mass: Option<f64>) -> Scene {
        let ball = radius.zip(mass);
        Scene(engine_fracture::towers::towers_grid(&self.0, columns, rows, gap, ball))
    }
    /// The towers showcase's volley for this block: `[{step, origin, dir, label}]`,
    /// `raking` glancing shots then `base` at the lower floors, `every` ticks apart.
    pub fn volley(&self, raking: u32, base: u32, every: u32) -> Result<JsValue, JsError> {
        to_js(&engine_fracture::towers::volley(&self.0, raking, base, every))
    }
    /// The scene with the balls' radius times `scale` (their mass stays).
    #[wasm_bindgen(js_name = withProjectileRadius)]
    pub fn with_projectile_radius(&self, scale: f64) -> Result<Scene, JsError> {
        self.0.clone().scale_projectile_radius(scale).map(Scene).map_err(err)
    }
    /// The scene with every bond's break capacities times `scale`.
    #[wasm_bindgen(js_name = withBondStrength)]
    pub fn with_bond_strength(&self, scale: f64) -> Scene {
        Scene(self.0.clone().scale_bond_strength(scale))
    }
    #[wasm_bindgen(js_name = withoutJoints)]
    pub fn without_joints(&self) -> Scene {
        Scene(self.0.clone().without_joints())
    }
    /// Bond preload, step 1: every piece awake, breaking off.
    #[wasm_bindgen(js_name = preloadVariant)]
    pub fn preload_variant(&self) -> Scene {
        Scene(self.0.preload_variant())
    }
    /// Bond preload, step 2: start asleep at the settled poses (28-word body records).
    pub fn settled(&self, snapshot: &[f32], steps: u32, settled: bool) -> Result<Scene, JsError> {
        self.0
            .settled(snapshot, steps, settled)
            .map(Scene)
            .map_err(err)
    }
    /// Merge into the GPU budget (`{...capacity, ...partial}`).
    #[wasm_bindgen(js_name = withCapacity)]
    pub fn with_capacity(&self, partial: JsValue) -> Result<Scene, JsError> {
        let mut s = self.0.clone();
        s.capacity = merge(&s.capacity, partial)?;
        Ok(Scene(s))
    }
    /// Merge into the solver parameters.
    #[wasm_bindgen(js_name = withParams)]
    pub fn with_params(&self, partial: JsValue) -> Result<Scene, JsError> {
        let mut s = self.0.clone();
        s.params = merge(&s.params, partial)?;
        Ok(Scene(s))
    }
    /// Measurement aid: every dynamic body starts awake.
    #[wasm_bindgen(js_name = wakeAll)]
    pub fn wake_all(&self) -> Scene {
        Scene(self.0.clone().wake_all())
    }
    /// Benchmark aid: houses outside the row shot's column become static.
    #[wasm_bindgen(js_name = freezeUnhitCopies)]
    pub fn freeze_unhit_copies(&mut self) -> u32 {
        self.0.freeze_unhit_copies()
    }
    pub fn meta(&self) -> Result<JsValue, JsError> {
        let s = &self.0;
        to_js(&Meta {
            body_count: s.bodies.len(),
            collider_count: s.colliders.len(),
            joint_count: s.joints.len(),
            hull_count: s.hull_count(),
            hull_records: s.hull_data.len(),
            params: &s.params,
            capacity: &s.capacity,
            skipped_hulls: &s.skipped_hulls,
            projectile_start: s.projectile_start,
            projectile_count: s.projectile_count,
            projectile_radius: s.projectile_radius,
            projectile_mass: s.projectile_mass,
            projectile_speed: s.projectile_speed,
            projectile_arc: s.projectile_arc,
            level: s.level,
            selected: &s.selected,
            bond_count: s.bond_count,
            extent: s.extent,
            extent_x: s.extent_x,
            extent_z: s.extent_z,
            height: s.height,
            center: s.center,
            copies: s.copies,
            instance_stride: s.instance_stride,
            columns: s.columns,
            rows: s.rows,
            row_pitch: s.row_pitch,
            column_pitch: s.column_pitch,
            row_length: s.row_length,
            grid_width: s.grid_width,
            grid_cell_size: s.grid_cell_size,
            camera: s.camera,
            row_shot: s.row_shot,
            preload: s.preload.as_ref(),
            pieces: s.pieces,
        })
    }
    /// `[{p, q, v, w, invMass, invI, flags}]` (diagnostics and tests).
    pub fn bodies(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.bodies)
    }
    pub fn colliders(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.colliders)
    }
    pub fn joints(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.joints)
    }
    /// `[[fragmentId, body]]` of copy 0.
    #[wasm_bindgen(js_name = bodyOfFragment)]
    pub fn body_of_fragment(&self, id: u32) -> Option<u32> {
        self.0.body_of_fragment(id)
    }
    /// Packed hull arena words (4 per record).
    #[wasm_bindgen(js_name = hullData)]
    pub fn hull_data(&self) -> Vec<u32> {
        self.0.hull_data.iter().flatten().copied().collect()
    }
    pub fn vertices(&self) -> Vec<f32> {
        self.0.vertices.clone()
    }
    pub fn lines(&self) -> Vec<f32> {
        self.0.lines.clone()
    }
    #[wasm_bindgen(js_name = projectileVertices)]
    pub fn projectile_vertices(&self) -> Vec<f32> {
        self.0.projectile_vertices.clone()
    }
    #[wasm_bindgen(js_name = roadVertices)]
    pub fn road_vertices(&self) -> Vec<f32> {
        self.0.road_vertices.clone()
    }
    /// World vertices (flat xyz) of convex collider `index` at pose `p`, `q`
    /// (default: its body's scene pose).
    #[wasm_bindgen(js_name = colliderVertices)]
    pub fn collider_vertices(
        &self,
        index: u32,
        p: Option<Vec<f64>>,
        q: Option<Vec<f64>>,
    ) -> Vec<f64> {
        let c = &self.0.colliders[index as usize];
        let b = self.0.bodies[c.body as usize];
        let p = p.map_or(b.p, |p| [p[0], p[1], p[2]]);
        let q = q.map_or(b.q, |q| [q[0], q[1], q[2], q[3]]);
        scene::collider_vertices(&self.0.hull_data, c, p, q)
            .into_iter()
            .flatten()
            .collect()
    }
}

/// Bond preloading: true once the settling run may stop (steps so far, awake bodies).
#[wasm_bindgen(js_name = preloadDone)]
pub fn preload_done(steps: u32, awake: u32) -> bool {
    scene::Scene::preload_done(steps, awake)
}

/// Scale physical bond springs to solver penalties (`scale`/`cap`: defaults
/// BOND_PENALTY_SCALE and BOND_PENALTY_CAP).
#[wasm_bindgen(js_name = scaleBondSprings)]
pub fn scale_bond_springs(
    joints: JsValue,
    bodies: JsValue,
    dt: f64,
    scale: Option<f64>,
    cap: Option<f64>,
    effective_mass: Option<bool>,
) -> Result<JsValue, JsError> {
    let joints: Vec<scene::Joint> = from_js(joints)?;
    let bodies: Vec<scene::Body> = from_js(bodies)?;
    use engine_fracture::bonds::*;
    to_js(&scale_bond_springs_with(
        &joints,
        &bodies,
        dt,
        scale.unwrap_or(BOND_PENALTY_SCALE),
        cap.unwrap_or(BOND_PENALTY_CAP),
        effective_mass.unwrap_or(true),
    ))
}

/// The lab's camera (orbit, first-person walk and fly) and click-to-shoot rays (shared
/// with the Metal viewer).
#[wasm_bindgen]
pub struct Orbit(engine_fracture::view::Orbit);

#[wasm_bindgen]
impl Orbit {
    #[wasm_bindgen(constructor)]
    pub fn new(scene: &Scene) -> Orbit {
        Orbit(engine_fracture::view::Orbit::new(&scene.0))
    }
    /// The view mode (the Camera menu: 0 orbit, 1 walk, 2 fly).
    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> u32 {
        self.0.mode.index()
    }
    #[wasm_bindgen(js_name = setMode)]
    pub fn set_mode(&mut self, mode: u32, aspect: f64) {
        self.0
            .set_mode(engine_fracture::view::ViewMode::from_index(mode), aspect)
    }
    pub fn drag(&mut self, dx: f64, dy: f64) {
        self.0.drag(dx, dy)
    }
    /// Pointer-lock movement (CSS pixels): first-person mouse-look.
    #[wasm_bindgen(js_name = mouseLook)]
    pub fn mouse_look(&mut self, dx: f64, dy: f64) {
        self.0.mouse_look(dx, dy)
    }
    pub fn zoom(&mut self, delta_y: f64) {
        self.0.zoom(delta_y)
    }
    /// First-person travel for `dt` seconds (axes in -1..1: forward, right, up).
    pub fn travel(&mut self, forward: f64, right: f64, up: f64, sprint: bool, dt: f64) {
        self.0.travel(forward, right, up, sprint, dt)
    }
    /// Place the orbit camera (scripted takes): yaw, pitch (radians), distance (m) and,
    /// unless `target` is empty, the target `[x, y, z]`. Switches to orbit mode.
    #[wasm_bindgen(js_name = setOrbit)]
    pub fn set_orbit(&mut self, yaw: f64, pitch: f64, distance: f64, target: &[f64]) {
        let target = (target.len() == 3).then(|| [target[0], target[1], target[2]]);
        self.0.set_orbit(yaw, pitch, distance, target)
    }
    /// `[yaw, pitch, distance, target x, y, z]`, as `setOrbit` takes them.
    #[wasm_bindgen(js_name = orbitState)]
    pub fn orbit_state(&self) -> Vec<f64> {
        self.0.orbit_state().to_vec()
    }
    /// Column-major view-projection for the draw shader.
    #[wasm_bindgen(js_name = viewProj)]
    pub fn view_proj(&self, aspect: f64) -> Vec<f32> {
        self.0.view_proj(aspect).to_vec()
    }
    /// `{origin, direction}` through a viewport point (u, v in 0..1 from the top left).
    pub fn aim(
        &self,
        u: f64,
        v: f64,
        aspect: f64,
        projectile_radius: f64,
    ) -> Result<JsValue, JsError> {
        to_js(&self.0.aim(u, v, aspect, projectile_radius))
    }
}

/// The ground quad for a scene extent (10-float render vertices).
#[wasm_bindgen(js_name = floorVertices)]
pub fn floor_vertices(extent: f64) -> Vec<f32> {
    engine_fracture::view::floor_vertices(extent)
}

/// Indexed render mesh with exact duplicate corners merged.
#[wasm_bindgen]
pub struct IndexedMesh {
    vertices: Vec<f32>,
    index: Vec<u32>,
}

#[wasm_bindgen]
impl IndexedMesh {
    #[wasm_bindgen(constructor)]
    pub fn new(vertices: &[f32]) -> IndexedMesh {
        let (vertices, index) = engine_fracture::view::index_mesh(vertices);
        IndexedMesh { vertices, index }
    }
    pub fn vertices(&self) -> Vec<f32> {
        self.vertices.clone()
    }
    pub fn index(&self) -> Vec<u32> {
        self.index.clone()
    }
}

/// The bundled scene catalogue (Scene menu of both hosts).
#[wasm_bindgen]
pub fn samples() -> Result<JsValue, JsError> {
    to_js(SAMPLES)
}

/// The Scene menu's scene packs built in Rust (the Wrecking Yard; `samples::PACK_SAMPLES`).
#[wasm_bindgen(js_name = packSamples)]
pub fn pack_samples() -> Result<JsValue, JsError> {
    to_js(engine_fracture::samples::PACK_SAMPLES)
}

/// A pack sample's scene pack JSON bytes (what the lab loads as any scene pack).
#[wasm_bindgen(js_name = packSampleJson)]
pub fn pack_sample_json(key: &str) -> Result<Vec<u8>, JsError> {
    engine_fracture::samples::pack_sample_json(key).map_err(err)
}

/// Hull arena with an engine-wide vertex limit (tests, GPU conformance suites).
#[wasm_bindgen]
pub struct HullArena(engine_schema::convex::HullArena);

#[wasm_bindgen]
impl HullArena {
    #[wasm_bindgen(constructor)]
    pub fn new(max_vertices: Option<u32>) -> Result<HullArena, JsError> {
        scene::hull_arena(max_vertices.map(|v| v as usize))
            .map(HullArena)
            .map_err(err)
    }
    /// Cook baked faces (flat xyz `vertices`), recentre and insert. Returns
    /// `{base, params, localPos, radius, vertices}` for a convex collider.
    #[wasm_bindgen(js_name = appendHull)]
    pub fn append_hull(
        &mut self,
        vertices: &[f64],
        sizes: &[u32],
        indices: &[u32],
        source_scale: Option<f64>,
    ) -> Result<JsValue, JsError> {
        let v: Vec<[f64; 3]> = vertices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let (base, radius, local_pos) =
            scene::append_hull(&mut self.0, &v, sizes, indices, source_scale.unwrap_or(0.0))
                .map_err(err)?;
        let c = Collider::hull(0, 1.0, base, 1.0, radius, local_pos);
        let nv = self.0.records()[base as usize][0] & 255;
        to_js(
            &serde_json::json!({"base": base, "params": c.params, "localPos": local_pos, "radius": radius, "vertexCount": nv}),
        )
    }
    pub fn release(&mut self, base: u32) -> bool {
        self.0.release(base)
    }
    pub fn refs(&self, base: u32) -> u32 {
        self.0.refs(base)
    }
    /// Dirty record ranges since the last call, as flat `[start, end, ...]`.
    #[wasm_bindgen(js_name = takeDirty)]
    pub fn take_dirty(&mut self) -> Vec<u32> {
        self.0
            .take_dirty()
            .into_iter()
            .flat_map(|r| [r.start, r.end])
            .collect()
    }
    #[wasm_bindgen(js_name = markAllDirty)]
    pub fn mark_all_dirty(&mut self) {
        self.0.mark_all_dirty()
    }
    /// Arena words (4 per record).
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<u32> {
        self.0.records().iter().flatten().copied().collect()
    }
    #[wasm_bindgen(getter)]
    pub fn records(&self) -> u32 {
        self.0.len() as u32
    }
    #[wasm_bindgen(getter, js_name = freeRecords)]
    pub fn free_records(&self) -> u32 {
        self.0.free_records()
    }
    #[wasm_bindgen(getter, js_name = maxVertices)]
    pub fn max_vertices(&self) -> u32 {
        self.0.max_vertices() as u32
    }
}

/// Decode a packed hull record (mirrors the Slang accessors; tests, oracles).
#[wasm_bindgen(js_name = hullView)]
pub fn hull_view(data: &[u32], base: u32) -> Result<JsValue, JsError> {
    let w = |i: usize| data[base as usize * 4 + i];
    let f = |i: usize| f32::from_bits(w(i)) as f64;
    let head = w(0);
    let (nv, nf, ne, flags) = (
        (head & 255) as usize,
        (head >> 8 & 255) as usize,
        (head >> 16 & 255) as usize,
        head >> 24,
    );
    let byte = |word: usize, b: usize| w(word + b / 4) >> ((b % 4) * 8) & 255;
    let vertices: Vec<[f64; 3]> = (0..nv)
        .map(|i| [f((4 + i) * 4), f((4 + i) * 4 + 1), f((4 + i) * 4 + 2)])
        .collect();
    let planes: Vec<serde_json::Value> = (0..nf)
        .map(|k| {
            let o = (4 + nv + k) * 4;
            serde_json::json!({"n": [f(o), f(o + 1), f(o + 2)], "d": f(o + 3)})
        })
        .collect();
    let faces: Vec<Vec<u32>> = (0..nf)
        .map(|k| {
            let entry = w((4 + nv + nf) * 4 + k);
            (0..entry >> 16)
                .map(|i| byte(w(1) as usize, (entry & 0xffff) as usize + i as usize))
                .collect()
        })
        .collect();
    let edges: Vec<[u32; 4]> = (0..ne)
        .map(|k| {
            let x = w(w(2) as usize + k);
            [x & 255, x >> 8 & 255, x >> 16 & 255, x >> 24]
        })
        .collect();
    let adjacency: Vec<Vec<u32>> = (0..nv)
        .map(|i| {
            let x = w((4 + i) * 4 + 3);
            (0..x >> 16)
                .map(|j| byte(w(3) as usize, (x & 0xffff) as usize + j as usize))
                .collect()
        })
        .collect();
    to_js(
        &serde_json::json!({"nv": nv, "nf": nf, "ne": ne, "closed": flags & 1 != 0,
        "vertices": vertices, "planes": planes, "faces": faces, "edges": edges, "adjacency": adjacency}),
    )
}

/// World vertices (flat xyz) of a convex collider (`{shape, params, localPos?,
/// localRot?}`) at body pose `p`, `q`.
#[wasm_bindgen(js_name = colliderWorldVertices)]
pub fn collider_world_vertices(
    hull_data: &[u32],
    collider: JsValue,
    p: &[f64],
    q: &[f64],
) -> Result<Vec<f64>, JsError> {
    let c: Collider = from_js(collider)?;
    let data: Vec<[u32; 4]> = hull_data
        .chunks(4)
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect();
    Ok(
        scene::collider_vertices(&data, &c, [p[0], p[1], p[2]], [q[0], q[1], q[2], q[3]])
            .into_iter()
            .flatten()
            .collect(),
    )
}

/// Conservative world AABB `[lo.xyz, hi.xyz]` of a collider at a body pose (as `build_aabbs`).
#[wasm_bindgen(js_name = colliderWorldBounds)]
pub fn collider_world_bounds(
    hull_data: &[u32],
    collider: JsValue,
    p: &[f64],
    q: &[f64],
) -> Result<Vec<f64>, JsError> {
    let c: Collider = from_js(collider)?;
    let data: Vec<[u32; 4]> = hull_data
        .chunks(4)
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect();
    let (lo, hi) = scene::collider_bounds(&data, &c, [p[0], p[1], p[2]], [q[0], q[1], q[2], q[3]]);
    Ok(lo.into_iter().chain(hi).collect())
}

/// Cook baked faces and pack them (hull ABI v2) without recentring: the
/// words `engine-schema`'s cooker writes (hull-golden.json).
#[wasm_bindgen(js_name = packHull)]
pub fn pack_hull(vertices: &[f32], sizes: &[u32], indices: &[u32]) -> Result<Vec<u32>, JsError> {
    use engine_schema::convex::ConvexHull;
    let v: Vec<glam::Vec3> = vertices
        .chunks(3)
        .map(|c| glam::Vec3::new(c[0], c[1], c[2]))
        .collect();
    let mut out = Vec::new();
    ConvexHull::from_faces(&v, sizes, indices)
        .map_err(err)?
        .pack(&mut out)
        .map_err(err)?;
    Ok(out.into_iter().flatten().collect())
}

/// Arena base of hull collider params.
#[wasm_bindgen(js_name = hullBase)]
pub fn hull_base(params: &[f64]) -> u32 {
    Collider::new(
        0,
        scene::SHAPE_CONVEX,
        0.0,
        [params[0], params[1], params[2], params[3]],
    )
    .hull_base()
}

/// The per-tick host plan (see `engine_fracture::plan`).
#[wasm_bindgen]
pub struct Plan(Planner);

#[wasm_bindgen]
impl Plan {
    #[wasm_bindgen(constructor)]
    pub fn new(scene: &Scene, max_binding: f64, max_buffer: f64) -> Result<Plan, JsError> {
        Planner::new(
            &scene.0,
            Limits {
                max_binding: max_binding as u64,
                max_buffer: max_buffer as u64,
            },
        )
        .map(Plan)
        .map_err(err)
    }
    /// `[[dispatch name, exported entry]]`, indexed by command kernel ids.
    pub fn kernels() -> Result<JsValue, JsError> {
        to_js(plan::KERNELS)
    }
    /// `[[phase name, ends a slice]]`, indexed by phase command ids.
    pub fn phases() -> Result<JsValue, JsError> {
        to_js(plan::PHASES)
    }
    #[wasm_bindgen(js_name = passTable)]
    pub fn pass_table() -> Vec<u8> {
        Planner::pass_table()
    }
    #[wasm_bindgen(js_name = clearedOnReset)]
    pub fn cleared_on_reset(slot: u32) -> bool {
        Planner::cleared_on_reset(slot)
    }
    #[wasm_bindgen(js_name = isUniform)]
    pub fn is_uniform(slot: u32) -> bool {
        Planner::is_uniform(slot)
    }
    #[wasm_bindgen(js_name = materialTable)]
    pub fn material_table(&self) -> Vec<u8> {
        self.0.material_table()
    }
    #[wasm_bindgen(js_name = bufferSizes)]
    pub fn buffer_sizes(&self) -> Vec<f64> {
        self.0
            .buffer_sizes_bytes()
            .iter()
            .map(|&s| s as f64)
            .collect()
    }
    #[wasm_bindgen(js_name = bodyBytes)]
    pub fn body_bytes(&self) -> Vec<u8> {
        self.0.body_bytes().to_vec()
    }
    #[wasm_bindgen(js_name = colliderBytes)]
    pub fn collider_bytes(&self) -> Vec<u8> {
        self.0.collider_bytes().to_vec()
    }
    #[wasm_bindgen(js_name = jointBytes)]
    pub fn joint_bytes(&self) -> Vec<u8> {
        self.0.joint_bytes().to_vec()
    }
    #[wasm_bindgen(getter)]
    pub fn cap(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.cap)
    }
    #[wasm_bindgen(getter, js_name = bodyCount)]
    pub fn body_count(&self) -> u32 {
        self.0.body_count()
    }
    #[wasm_bindgen(getter, js_name = colliderCount)]
    pub fn collider_count(&self) -> u32 {
        self.0.collider_count()
    }
    #[wasm_bindgen(getter, js_name = jointCount)]
    pub fn joint_count(&self) -> u32 {
        self.0.joint_count()
    }
    #[wasm_bindgen(getter)]
    pub fn steps(&self) -> u32 {
        self.0.steps
    }
    #[wasm_bindgen(getter)]
    pub fn shots(&self) -> u32 {
        self.0.shots
    }
    #[wasm_bindgen(getter)]
    pub fn initialized(&self) -> bool {
        self.0.initialized
    }
    #[wasm_bindgen(getter)]
    pub fn counters(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.last)
    }
    /// Live solver parameters (packed again every step).
    #[wasm_bindgen(getter)]
    pub fn params(&self) -> Result<JsValue, JsError> {
        to_js(&self.0.params)
    }
    #[wasm_bindgen(js_name = setParams)]
    pub fn set_params(&mut self, partial: JsValue) -> Result<(), JsError> {
        self.0.params = merge(&self.0.params, partial)?;
        Ok(())
    }
    #[wasm_bindgen(setter, js_name = incrementalColoring)]
    pub fn set_incremental_coloring(&mut self, on: bool) {
        self.0.incremental = on;
    }
    #[wasm_bindgen(getter, js_name = awakeQueries)]
    pub fn awake_queries(&self) -> bool {
        self.0.awake_queries
    }
    pub fn reset(&mut self) {
        self.0.reset()
    }
    /// True (and `count` steps advanced) when a sleeping world needs no GPU work.
    #[wasm_bindgen(js_name = skipAsleep)]
    pub fn skip_asleep(&mut self, count: u32) -> bool {
        self.0.skip_asleep(count)
    }
    /// Encode one tick; returns the flat command words (see `hostUpload`).
    #[wasm_bindgen(js_name = encodeStep)]
    pub fn encode_step(&mut self, reuse_manifolds: bool) -> Vec<u32> {
        self.0.encode_step(reuse_manifolds).to_vec()
    }
    #[wasm_bindgen(js_name = encodeInspect)]
    pub fn encode_inspect(&mut self, flags: u32, awake_queries: bool) -> Vec<u32> {
        self.0.encode_inspect(flags, awake_queries).to_vec()
    }
    #[wasm_bindgen(js_name = afterInspect)]
    pub fn after_inspect(&mut self) {
        self.0.after_inspect()
    }
    /// The last encode's host upload (source of `HOST_UPLOAD` copies).
    #[wasm_bindgen(js_name = hostUpload)]
    pub fn host_upload(&self) -> Vec<u8> {
        self.0.host_upload().to_vec()
    }
    #[wasm_bindgen(js_name = markSubmitted)]
    pub fn mark_submitted(&mut self) {
        self.0.mark_submitted()
    }
    /// `[steps, wake serial]` captured when a counter copy is queued.
    #[wasm_bindgen(js_name = beginRead)]
    pub fn begin_read(&self) -> Vec<u32> {
        let (a, b) = self.0.begin_read();
        vec![a, b]
    }
    #[wasm_bindgen(js_name = finishRead)]
    pub fn finish_read(&mut self, token: &[u32], words: &[u32]) -> Result<JsValue, JsError> {
        to_js(&self.0.finish_read((token[0], token[1]), words))
    }
    /// A counter read that leaves the step counters unchanged (inspection passes).
    #[wasm_bindgen(js_name = countRead)]
    pub fn count_read(&mut self, token: &[u32], words: &[u32]) -> Result<JsValue, JsError> {
        to_js(&self.0.count_read((token[0], token[1]), words))
    }
    #[wasm_bindgen(js_name = overflowError)]
    pub fn overflow_error(&mut self) -> Result<(), JsError> {
        self.0.overflow_error().map_err(err)
    }
    #[wasm_bindgen(js_name = markStale)]
    pub fn mark_stale(&mut self, manifolds: bool) {
        self.0.mark_stale(manifolds)
    }
    /// Replace collider `index`; returns its 64-byte record.
    #[wasm_bindgen(js_name = setCollider)]
    pub fn set_collider(&mut self, index: u32, collider: JsValue) -> Result<Vec<u8>, JsError> {
        self.0.set_collider(index, from_js(collider)?).map_err(err)
    }
    /// The mass (kg) of the projectiles fired from now on.
    #[wasm_bindgen(js_name = setProjectileMass)]
    pub fn set_projectile_mass(&mut self, mass: f64) -> Result<(), JsError> {
        self.0.set_projectile_mass(mass).map_err(err)
    }
    /// The speed of the projectiles fired from now on, as a multiple of the scene's.
    #[wasm_bindgen(js_name = setProjectileSpeedScale)]
    pub fn set_projectile_speed_scale(&mut self, scale: f64) -> Result<(), JsError> {
        self.0.set_projectile_speed_scale(scale).map_err(err)
    }
    /// Continuous collision (time-of-impact contacts) for the shots fired from now on.
    #[wasm_bindgen(js_name = setProjectileCcd)]
    pub fn set_projectile_ccd(&mut self, on: bool) {
        self.0.set_projectile_ccd(on)
    }
    /// The scene's shot speed (m/s) before the scale.
    #[wasm_bindgen(js_name = shotSpeed)]
    pub fn shot_speed(&self) -> f64 {
        self.0.shot_speed()
    }
    /// Fire the next projectile: `{id, bytes, offsets}` (write `bytes` at both offsets of buffer 0).
    pub fn launch(
        &mut self,
        origin: &[f64],
        direction: &[f64],
        speed: Option<f64>,
        arc: Option<bool>,
    ) -> Result<JsValue, JsError> {
        let (id, bytes) = self
            .0
            .launch(
                [origin[0], origin[1], origin[2]],
                [direction[0], direction[1], direction[2]],
                speed,
                arc,
            )
            .map_err(err)?;
        to_js(&serde_json::json!({"id": id, "bytes": bytes, "offsets": self.0.body_offsets(id)}))
    }
    /// Contact record base of the encoded step (`previous`: of the last completed step).
    #[wasm_bindgen(js_name = contactBase)]
    pub fn contact_base(&self, previous: bool) -> u32 {
        self.0.contact_base(previous)
    }
}

/// Directed debris, "Near Miss" (`engine_fracture::guidance`).
#[wasm_bindgen]
impl Scene {
    /// One ring zone per bystander of the crowd downrange of a shot.
    #[wasm_bindgen(js_name = nearMissZonesFor)]
    pub fn near_miss_zones_for(&self, origin: &[f64], direction: &[f64]) -> Result<JsValue, JsError> {
        let v3 = |v: &[f64]| -> Result<[f64; 3], JsError> {
            v.try_into().map_err(|_| err("expected 3 numbers"))
        };
        to_js(&near_miss::zones_downrange(&self.0, v3(origin)?, v3(direction)?))
    }
    /// This scene with parked cars and pedestrians on the streets around its
    /// block of towers (`engine_fracture::props`).
    #[wasm_bindgen(js_name = withCity)]
    pub fn with_city(&self) -> Scene {
        Scene(engine_fracture::props::with_city(&self.0))
    }
    /// Zones on the scene's cars (land on the roof) and pedestrians (near misses).
    #[wasm_bindgen(js_name = cityZones)]
    pub fn city_zones(&self) -> Result<JsValue, JsError> {
        to_js(&engine_fracture::props::city_zones(&self.0))
    }
    /// The street props' render stream (10 floats per vertex, drawn once).
    #[wasm_bindgen(js_name = propVertices)]
    pub fn prop_vertices(&self) -> Vec<f32> {
        self.0.prop_vertices.clone()
    }
    /// A meteor onto the block's first tower: `{step, origin, dir, label, speed}`.
    #[wasm_bindgen(js_name = meteorShot)]
    pub fn meteor_shot(&self) -> Result<JsValue, JsError> {
        to_js(&engine_fracture::towers::meteor(&self.0))
    }
    /// Base shots that topple tower `tower` outward over its street.
    #[wasm_bindgen(js_name = toppleShots)]
    pub fn topple_shots(&self, tower: u32, shots: u32, every: u32) -> Result<JsValue, JsError> {
        to_js(&engine_fracture::towers::topple(&self.0, tower as usize, shots, every))
    }
    /// Line-list vertices outlining `zones` (drawn on body 0).
    #[wasm_bindgen(js_name = zoneOutline)]
    pub fn zone_outline(&self, zones: JsValue) -> Result<Vec<f32>, JsError> {
        let zones: Vec<ZoneDesc> = from_js(zones)?;
        let ground = self.0.bodies.first().map_or([0.0; 3], |b| b.p);
        Ok(near_miss::outline_vertices(&zones, ground))
    }
    /// The scene's pieces in a body snapshot that lie near each zone.
    #[wasm_bindgen(js_name = nearCounts)]
    pub fn near_counts(&self, snapshot: &[f32], zones: JsValue) -> Result<Vec<u32>, JsError> {
        let zones: Vec<ZoneDesc> = from_js(zones)?;
        Ok(near_miss::near_counts(snapshot, 1, self.0.projectile_start, &zones))
    }
}

/// The guidance settings of a Near Miss strength (`subtle`, `cinematic`, `exaggerated`).
#[wasm_bindgen(js_name = nearMissSettings)]
pub fn near_miss_settings(strength: JsValue) -> Result<JsValue, JsError> {
    let strength: near_miss::Strength = from_js(strength)?;
    to_js(&near_miss::settings(strength))
}

#[wasm_bindgen]
impl Plan {
    /// Enable, disable or retune directed debris: `partial` over the defaults.
    #[wasm_bindgen(js_name = setGuidance)]
    pub fn set_guidance(&mut self, partial: JsValue) -> Result<(), JsError> {
        let settings: GuidanceSettings = merge(&GuidanceSettings::default(), partial)?;
        self.0.set_guidance(&settings).map_err(err)
    }
    /// Validate `zones`; returns their records for buffer 56 (offset 0).
    #[wasm_bindgen(js_name = setGuidanceZones)]
    pub fn set_guidance_zones(&mut self, zones: JsValue) -> Result<Vec<u8>, JsError> {
        let zones: Vec<ZoneDesc> = from_js(zones)?;
        self.0.set_guidance_zones(&zones).map_err(err)
    }
    /// Every body's guidance record for buffer 57 (offset 0): the pieces armed
    /// with `tagMask`, or all off for 0.
    #[wasm_bindgen(js_name = guideRecords)]
    pub fn guide_records(&self, tag_mask: u32) -> Vec<u8> {
        self.0.guide_records(tag_mask)
    }
    /// Guidance buffers: `[params, zones, bodies, scratch, slots]`.
    #[wasm_bindgen(js_name = guideBindings)]
    pub fn guide_bindings() -> Vec<u32> {
        use engine_schema::guidance as gd;
        [
            gd::GUIDE_PARAMS_BINDING,
            gd::GUIDE_ZONES_BINDING,
            gd::GUIDE_BODIES_BINDING,
            gd::GUIDE_SCRATCH_BINDING,
            gd::GUIDE_SLOTS_BINDING,
        ]
        .map(|b| b as u32)
        .to_vec()
    }
    /// Bodies per guidance state `[off, armed, steering, done]` in a readback of the bodies buffer.
    #[wasm_bindgen(js_name = guideStateCounts)]
    pub fn guide_state_counts(records: &[u8]) -> Vec<u32> {
        Planner::guide_state_counts(records).to_vec()
    }
    /// Bytes of one body's guidance record in the bodies buffer.
    #[wasm_bindgen(js_name = guideRecordBytes)]
    pub fn guide_record_bytes() -> u32 {
        std::mem::size_of::<engine_schema::guidance::GuideBodyGpu>() as u32
    }
    /// `[byte offset, byte length]` of the zone claim counters in the scratch buffer.
    #[wasm_bindgen(js_name = guideClaimsRange)]
    pub fn guide_claims_range(&self) -> Vec<u32> {
        let (o, n) = self.0.guide_claims_range();
        vec![o, n]
    }
}
