//! Convex-hull (Shardwright) scenes on rubble's native Metal fracture host:
//! `engine_viewer::fracture::FractureSim`, the native counterpart of the WebGPU fracture demo
//! (same Slang kernels, dispatch order and hull ABI).
//!
//! The scene file records which asset and level it was exported from. Rubble rebuilds that
//! level with its own importer (`engine_fracture::build_scene`), so its GPU keeps the baked
//! hull topology. The benchmark's settings are then patched in: iterations, sleeping off
//! (fragments awake, sleep timer out of reach; the demo starts them asleep), and the scripted
//! ball written into the first projectile slot. Checkpoint 0 is compared with the scene file
//! as for every engine, so any importer drift shows up as an initial-state mismatch.

use anyhow::{bail, ensure, Context, Result};
use bench_core::{Args, BodySample, Engine, EngineInfo, Scene, Shape};
use engine_viewer::fracture::FractureSim;
use serde_json::json;
use std::collections::BTreeMap;

pub struct Fracture {
    sim: FractureSim,
    bodies: usize,
    iterations: u32,
    sleeping: bool,
    device: String,
    config: BTreeMap<String, String>,
}

pub fn build(scene: &Scene, args: &mut Args, version: String) -> Result<Fracture> {
    let src = &scene.meta.source;
    let path = src["assetPath"].as_str().context("hull scene has no source.assetPath (re-export with tools/fracture-export)")?;
    let level = src["level"].as_u64().context("hull scene has no source.level")? as u8;
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let mut fs = engine_fracture::build_scene(&engine_fracture::parse_fracture_glb(&bytes)?, level)?;
    let ball = fs.projectile_start as usize;
    ensure!(scene.bodies.len() == ball + 1, "scene has {} bodies; rubble's importer gives {} fragments + ground + ball", scene.bodies.len(), ball);
    let b = &scene.bodies[ball];
    let Shape::Sphere(r) = b.shape else { bail!("last scene body is not the ball") };

    let iterations = args.iterations.unwrap_or(scene.meta.iterations);
    fs.params.iterations = iterations;
    fs.params.dt = scene.meta.dt;
    fs.params.gravity_y = scene.meta.gravity[1];
    fs.params.alpha = scene.meta.avbd.alpha;
    if !args.sleep {
        fs.params.sleep_time = 1.0e6;
        for f in &mut fs.bodies[1..ball] {
            f.flags = 9; // dynamic | awake (the demo starts fragments asleep: 5)
        }
    }
    fs.bodies[ball] = engine_fracture::Body {
        p: b.position,
        q: b.rotation,
        v: b.linvel,
        inv_mass: 1.0 / b.mass,
        inv_i: b.inertia.map(|x| 1.0 / x),
        flags: 9,
    };
    let col = fs.colliders.iter_mut().find(|c| c.body as usize == ball).context("no projectile collider")?;
    col.params[0] = r;

    let device = metal::Device::system_default().context("no Metal device")?;
    let queue = device.new_command_queue();
    let sim = FractureSim::new(&device, &queue, &fs)?;
    let mut config = BTreeMap::new();
    config.insert("host".into(), "engine-viewer FractureSim (native Metal, fracture_step.slang convex SAT)".into());
    config.insert("asset".into(), format!("{} level {level} ({} hulls)", src["asset"].as_str().unwrap_or("?"), fs.hull_count));
    config.insert("reuseContacts".into(), (fs.params.flags & 1 != 0).to_string());
    config.insert("alpha".into(), fs.params.alpha.to_string());
    config.insert("contactOffset".into(), fs.params.contact_offset.to_string());
    config.insert("submit".into(), "one tick per command buffer, at most one in flight".into());
    config.insert("version".into(), version);
    Ok(Fracture { sim, bodies: scene.bodies.len(), iterations, sleeping: args.sleep, device: device.name().to_string(), config })
}

impl Engine for Fracture {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            engine: "rubble".into(),
            version: self.config["version"].clone(),
            backend: "metal (fracture host)".into(),
            device: self.device.clone(),
            threads: 1,
            pipelined: true,
            solver: "AVBD".into(),
            iterations: self.iterations,
            iterations_meaning: "AVBD iterations (primal+dual) per step".into(),
            sleeping: self.sleeping,
            config: self.config.iter().filter(|(k, _)| *k != "version").map(|(k, v)| (k.clone(), v.clone())).collect(),
        }
    }

    fn step(&mut self) -> Result<()> {
        self.sim.submit(1).map(|_| ())
    }

    fn drain(&mut self) -> Result<()> {
        self.sim.wait()
    }

    fn read_state(&mut self, out: &mut [BodySample]) -> Result<()> {
        // 112-byte BodyState: position 0..3, orientation 4..8, linear_vel 12..15, angular_vel 16..19
        let w = self.sim.snapshot(self.bodies)?;
        for (i, s) in out.iter_mut().enumerate() {
            let b = &w[i * 28..i * 28 + 28];
            *s = BodySample { position: [b[0], b[1], b[2]], rotation: [b[4], b[5], b[6], b[7]], linvel: [b[12], b[13], b[14]], angvel: [b[16], b[17], b[18]] };
        }
        Ok(())
    }

    fn counters(&mut self) -> Result<serde_json::Value> {
        let c = self.sim.last;
        Ok(json!({ "awake": c.awake, "contacts": c.contact, "candidates": c.candidate, "manifolds": c.manifold, "colors": c.color, "overflow": c.overflow }))
    }
}
