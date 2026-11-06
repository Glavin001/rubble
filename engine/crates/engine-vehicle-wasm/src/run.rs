//! wasm-bindgen face of the "Meteor gauntlet" game
//! (`engine_vehicle::scripts::run`): the page (vehicle.html `?scene=run`)
//! builds the street from `layout`, carries out `tick`'s commands (launch a
//! meteor, blast a facade, park a spent meteor), reads back the bodies
//! `watched` lists for `judge`, and shows `status`.

use wasm_bindgen::prelude::*;

use engine_vehicle::scripts::run as script;
pub use engine_vehicle::scripts::run::*;

/// The run (`engine_vehicle::scripts::run::MeteorRun`).
#[wasm_bindgen]
pub struct MeteorRun {
    inner: script::MeteorRun,
}

impl Default for MeteorRun {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl MeteorRun {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MeteorRun {
        MeteorRun {
            inner: script::MeteorRun::new(),
        }
    }

    /// The street (`Layout` JSON) with the meteors' `{radius, mass, count}`
    /// and the finish line.
    pub fn layout(&self) -> String {
        let mut v = serde_json::to_value(self.inner.street().street()).unwrap();
        v["meteor"] =
            serde_json::json!({ "radius": METEOR_RADIUS, "mass": METEOR_MASS, "count": METEORS });
        v["finish"] = FINISH.into();
        v.to_string()
    }

    /// A fresh run on an unbroken street.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// `step` at time `t` with the car's `[x, y, z, vx, vy, vz]`:
    /// `[launches, (slot, px, py, pz, vx, vy, vz, tags) × launches,
    ///   blasts, (cx, cy, cz, n, (block, vx, vy, vz, wx, wy, wz) × n) × blasts,
    ///   parks, slot × parks]`.
    pub fn tick(&mut self, t: f32, dt: f32, car: &[f32], gravity_y: f32) -> Vec<f32> {
        let (launch, blasts, park) = self.inner.step(
            t,
            dt,
            [car[0], car[1], car[2]],
            [car[3], car[4], car[5]],
            gravity_y,
        );
        let mut out = vec![launch.len() as f32];
        for s in launch {
            out.push(s.slot as f32);
            out.extend_from_slice(&s.p);
            out.extend_from_slice(&s.v);
            out.push(s.tags as f32);
        }
        out.push(blasts.len() as f32);
        for (c, list) in blasts {
            out.extend_from_slice(&c);
            out.push(list.len() as f32);
            for (i, v, w) in list {
                out.push(i as f32);
                out.extend_from_slice(&v);
                out.extend_from_slice(&w);
            }
        }
        out.push(park.len() as f32);
        out.extend(park.iter().map(|&s| s as f32));
        out
    }

    /// `watched_list` flat: `(kind, index)` pairs.
    pub fn watched(&self) -> Vec<u32> {
        self.inner
            .watched_list()
            .into_iter()
            .flat_map(|(k, i)| [if k == Kind::Meteor { 1 } else { 0 }, i])
            .collect()
    }

    /// `judge_at` with the car `[px, py, pz, qx, qy, qz, qw, vx, vy, vz]` and
    /// the bodies `(kind, index, px, py, pz, vx, vy, vz)` flat: what happened
    /// as `(event, kind, index, x, y, z)` each (`Event` codes, kind 1 a meteor).
    pub fn judge(&mut self, t: f32, car: &[f32], bodies: &[f32]) -> Vec<f32> {
        let list: Vec<_> = bodies
            .chunks_exact(8)
            .map(|b| {
                let kind = if b[0] == 1.0 {
                    Kind::Meteor
                } else {
                    Kind::Block
                };
                (kind, b[1] as u32, [b[2], b[3], b[4]], [b[5], b[6], b[7]])
            })
            .collect();
        self.inner
            .judge_at(
                t,
                [car[0], car[1], car[2]],
                [car[3], car[4], car[5], car[6]],
                [car[7], car[8], car[9]],
                &list,
            )
            .into_iter()
            .flat_map(|(e, k, i, p)| {
                [
                    e as u32 as f32,
                    if k == Kind::Meteor { 1.0 } else { 0.0 },
                    i as f32,
                    p[0],
                    p[1],
                    p[2],
                ]
            })
            .collect()
    }

    /// `Status` JSON at time `t` with the car at `car_z`.
    pub fn status(&self, t: f32, car_z: f32) -> String {
        serde_json::to_string(&self.inner.status_at(t, car_z)).unwrap()
    }

    /// `zones` as `ZoneDesc` JSON.
    pub fn zones(&self, player: u32, chassis_y: f32, parked: &[u32]) -> String {
        serde_json::to_string(&zones(player, chassis_y, parked)).unwrap()
    }
}
