//! The one-World showcase (`showcase_scene`) natively: a preset with a BeamNG JBeam
//! vehicle read from a converted vehicle folder (its `files.json`, `.jbeam` files and a
//! `.pc`), or the repo sedan; each tenth of a second its broken beams and
//! house bonds, the first car's centroid and speed; then the GPU ms per World step.
//!
//! cargo run --release -p engine-world --example showcase_probe -- house \
//!     engine/crates/engine-webgpu/__mods/etkk_tex etkk/kc6.pc [count] [secs]
//! (one at a time on a quiet GPU; bounded: run it under `perl -e 'alarm 600; exec @ARGV'`).
//! SP_SPEED (m/s) and SP_RUNUP (m): the preset's speed and run up; SP_STRENGTH the houses'
//! bond capacity scale; SP_PHASES=1 GPU time per segment; SP_MASK0=1 no coupling.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use rubble::explicit_scene::town_coupling;
use rubble::showcase_scene::{jbeam_vehicle, showcase, ShowcaseHouses, ShowcasePreset};
use rubble::*;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let preset = ShowcasePreset::parse(a.first().map_or("house", |s| s.as_str()))
        .ok_or_else(|| anyhow::anyhow!("preset: house, crates, twoway, convoy"))?;
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: true,
        turbulence: true,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let template = match a.get(1).filter(|d| !d.is_empty()) {
        Some(dir) => {
            let dir = std::path::Path::new(dir);
            let list: Vec<String> =
                serde_json::from_str(&std::fs::read_to_string(dir.join("files.json"))?)?;
            let mut files = Vec::new();
            for f in list.iter().filter(|f| f.ends_with(".jbeam")) {
                files.push((f.clone(), std::fs::read_to_string(dir.join(f))?));
            }
            let pc = a
                .get(2)
                .filter(|p| !p.is_empty())
                .map(|p| std::fs::read_to_string(dir.join(p)))
                .transpose()?;
            let refs: Vec<(&str, &str)> = files
                .iter()
                .map(|(n, t)| (n.as_str(), t.as_str()))
                .collect();
            let (sim, warnings) = jbeam_vehicle(&refs, pc.as_deref(), settings)?;
            println!("{} nodes, {} beams, {} triangles, {} tyres (first nodes {:?}), {} brakes, {} warnings", sim.nodes.len(), sim.beams.len(), sim.triangles.len(), sim.tyres.len(), sim.tyres.iter().map(|t| (t.first, t.count)).collect::<Vec<_>>(), sim.brakes.len(), warnings.len());
            sim
        }
        None => {
            // The repo sedan (`.truck`), when no vehicle folder is given.
            let text = std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
            ))?;
            ExplicitBeams::from_truck(&text, glam::Vec3::ZERO, glam::Vec3::ZERO, settings)
                .map_err(anyhow::Error::msg)?
        }
    };
    let count: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let secs: f32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let env = |k: &str| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0)
    };
    // SAFETY: set before any thread reads it.
    unsafe { std::env::set_var("RUBBLE_GPU_TIMING", "1") };
    let mut s = showcase(
        GpuContext::Default,
        std::slice::from_ref(&template),
        preset,
        count,
        env("SP_SPEED"),
        env("SP_RUNUP"),
        ShowcaseHouses {
            strength_scale: if env("SP_STRENGTH") > 0.0 {
                env("SP_STRENGTH")
            } else {
                ShowcaseHouses::showcase().strength_scale
            },
            ..ShowcaseHouses::showcase()
        },
        &{
            // SP_MASK0=1 (measurement): the vehicles couple with nothing.
            let mut def = town_coupling();
            if std::env::var("SP_MASK0").is_ok() {
                def.mask = 0;
            }
            def
        },
    )?;
    println!(
        "built in {:.1} s; impact at {:.2} s",
        t0.elapsed().as_secs_f32(),
        s.layout.impact_time
    );
    let w = &mut s.world;
    let dt = w.solver().dt;
    let steps = (secs / dt).round() as u32;
    let (mut bonds, mut ms) = (0usize, Vec::new());
    // SP_PHASES=1: the World's per-segment timestamps, their mean ms per step printed.
    let phases = std::env::var("SP_PHASES").as_deref() == Ok("1");
    if phases {
        w.set_gpu_timing(GpuTiming::Phases)?;
    }
    let mut prev: Option<Vec<Vec<glam::Vec3>>> = None;
    let mut seg: std::collections::BTreeMap<&'static str, (f64, u32)> = Default::default();
    for k in 1..=steps {
        w.step()?;
        for f in w.poll_blocking()? {
            for (n, t) in f.gpu_time.iter().flat_map(|t| t.segments.iter()) {
                let e = seg.entry(n).or_default();
                e.0 += t;
                e.1 += 1;
            }
            for e in &f.destruction_events {
                if let DestructionEvent::BondsBroken { bonds: b, .. } = e {
                    bonds += b.len();
                }
            }
        }
        ms.extend(w.last_step_gpu_ms());
        // SP_NAN=1: each step, the first non-finite node and its neighbours' state.
        if std::env::var("SP_NAN").is_ok() {
            let x = w.explicit_vehicles().expect("vehicles");
            let pos = x.positions();
            if let Some((v, i)) = pos
                .iter()
                .enumerate()
                .find_map(|(v, p)| p.iter().position(|q| !q.is_finite()).map(|i| (v, i)))
            {
                let bad = pos[v].iter().filter(|q| !q.is_finite()).count();
                println!("step {k}: vehicle {v} node {i} non-finite ({bad} nodes)");
                break;
            }
            prev = Some(pos);
        }
        // SP_GROUPS=1: every half second, the vehicle contact groups and margins.
        if std::env::var("SP_GROUPS").is_ok() && k % 30 == 0 {
            if let Some(x) = w.explicit_vehicles_mut() {
                let mut g = x.contact_group_sizes();
                g.sort_unstable();
                let m: Vec<f32> = (0..x.vehicle_count())
                    .map(|v| x.vehicle_contact_margin(v))
                    .collect();
                println!("step {k}: groups {g:?}, margins {m:?}");
                for e in x.take_contact_errors() {
                    println!("  error: {e}");
                }
            }
        }
        if k % 6 == 0 {
            let x = w.explicit_vehicles().expect("vehicles");
            let (pos, vel, beams) = (x.positions(), x.velocities(), x.beams());
            let o = x.origins();
            let broken: Vec<usize> = beams
                .iter()
                .map(|b| b.iter().filter(|b| b.broken).count())
                .collect();
            let c = pos[0].iter().copied().sum::<glam::Vec3>() / pos[0].len() as f32 + o[0];
            let v = vel[0].iter().copied().sum::<glam::Vec3>() / vel[0].len() as f32;
            let finite = pos.iter().flatten().all(|p| p.is_finite());
            println!(
                "t {:.2} s: car 0 at ({:.2}, {:.2}, {:.2}) {:.1} m/s; beams broken {:?}; bonds {}{}",
                k as f32 * dt,
                c.x,
                c.y,
                c.z,
                v.length(),
                broken,
                bonds,
                if finite { "" } else { " NON-FINITE" }
            );
        }
    }
    if phases {
        let mut v: Vec<(&str, f64)> = seg
            .iter()
            .map(|(n, (t, _))| (*n, t / f64::from(steps)))
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        let list: Vec<String> = v
            .iter()
            .take(12)
            .map(|(n, t)| format!("{n} {t:.2}"))
            .collect();
        println!("segments (mean ms per step): {}", list.join(", "));
    }
    ms.sort_by(f64::total_cmp);
    if !ms.is_empty() {
        println!(
            "GPU per World step: median {:.2} ms, p95 {:.2} ({} steps)",
            ms[ms.len() / 2],
            ms[ms.len() * 95 / 100],
            ms.len()
        );
    }
    Ok(())
}
