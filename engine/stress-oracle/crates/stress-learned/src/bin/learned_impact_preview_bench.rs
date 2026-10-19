//! Benchmark the exact causal preview at a saved scene's initial state.
//! Timing repetitions exclude profiling; a separate pass reports stage costs.
use std::{fs, path::PathBuf, time::Instant};
use serde_json::json;
use stress_ref::{scene::Scene, world::{World, LearnedFrameBins}};

fn bits(bins: &LearnedFrameBins) -> Vec<u64> {
    bins.impulse_force.iter().chain(&bins.impulse_torque)
        .flat_map(|bin| bin.iter().flat_map(|structure| structure.iter()))
        .flat_map(|v| v.to_array().map(f64::to_bits))
        .chain(bins.contacted.iter().flatten().map(|v| u64::from(*v)))
        .collect()
}

fn main() -> Result<(), String> {
    let mut args=std::env::args().skip(1);
    let mut scene_path=None;let mut out=None;let mut repeats=3usize;
    while let Some(arg)=args.next() {
        let value=args.next().ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--scene"=>scene_path=Some(PathBuf::from(value)),
            "--out"=>out=Some(PathBuf::from(value)),
            "--repeats"=>repeats=value.parse().map_err(|_|"invalid repeats")?,
            _=>return Err(format!("unknown argument {arg}")),
        }
    }
    if repeats==0 { return Err("repeats must be positive".into()); }
    let path=scene_path.ok_or("--scene required")?;
    let out=out.ok_or("--out required")?;
    let bins_path=out.with_extension("bins");
    if out.exists() || bins_path.exists() {return Err("refusing to overwrite benchmark output".into());}
    let scene: Scene=serde_json::from_slice(&fs::read(&path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let init=Instant::now();let world=World::new(&scene);let init_seconds=init.elapsed().as_secs_f64();
    let unprofiled=!stress_ref::profile::enabled();
    let expected=bits(&world.learned_rigid_preview(4)?);
    let mut times=Vec::new();let mut clone_times=Vec::new();
    for _ in 0..repeats {
        let t=Instant::now();let cloned=world.clone();clone_times.push(t.elapsed().as_secs_f64());drop(cloned);
        let t=Instant::now();let bins=world.learned_rigid_preview(4)?;times.push(t.elapsed().as_secs_f64());
        if bits(&bins)!=expected {return Err("preview output changed between repetitions".into());}
    }
    stress_ref::profile::reset();stress_ref::profile::enable();
    let profiled=world.learned_rigid_preview(4)?;
    if bits(&profiled)!=expected {return Err("profiling changed preview output".into());}
    let report=stress_ref::profile::report();
    fs::write(&bins_path,expected.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()).map_err(|e|e.to_string())?;
    let result=json!({"scene":path,"scene_name":scene.name,"frame_dt":scene.sim.frame_dt,
        "initial_substep_dt":world.substep_dt(),"initialization_seconds":init_seconds,
        "active_chunks":world.solver.chunks.iter().flatten().filter(|c|c.active).count(),
        "initial_clusters":world.solver.clusters.len(),
        "bonds":world.solver.bonds.iter().map(Vec::len).sum::<usize>(),
        "timing_without_profiling":unprofiled,"repeats":repeats,
        "preview_seconds":times,"clone_seconds":clone_times,
        "repeat_and_profile_output_bitwise_identical":true,"profile":report,"output_bits":bins_path,
        "output_bits_contract":"little-endian u64: force then torque [bin][structure][chunk][xyz] IEEE754 bits, then contact flags [structure][chunk] as 0/1",
        "scope":"Initial scene only; same contact laws, timestep, substeps and geometry as the existing causal preview. Warm-up excluded. Preview includes clone; separate clone timings are diagnostic. Concurrent host workloads may affect wall time. No neural inference or oracle step included."});
    fs::write(&out,serde_json::to_vec_pretty(&result).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    println!("{}",serde_json::to_string_pretty(&result).unwrap());
    Ok(())
}
