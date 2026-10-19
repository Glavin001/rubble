//! Re-record a saved scene into a fresh directory, accepting added history only
//! after every original static/frame array and replay field matches exactly.
use std::{fs,path::{Path,PathBuf},time::Instant};
use serde_json::{Value,json};
use stress_learned::{catalog::Catalog,oracle::{record_event_options,LoadBasis},validate_event,RecordMeta};

fn read(path:&Path)->Result<Value,String> {serde_json::from_slice(&fs::read(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())}
fn subset(a:&Value,b:&Value)->bool {
    match (a,b) {
        (Value::Object(a),Value::Object(b))=>a.iter().all(|(k,v)|b.get(k).is_some_and(|x|subset(v,x))),
        (Value::Array(a),Value::Array(b))=>a.len()==b.len() && a.iter().zip(b).all(|(x,y)|subset(x,y)),
        _=>a==b,
    }
}
// Recipe metadata only. Historical JSON parsing may move an f64 by one ULP;
// preserve the original metadata after physical arrays/replay match exactly.
// Never use this predicate for physics, history, model inputs or scene equality.
fn metadata_roundtrip_equivalent(a:&Value,b:&Value)->bool {
    if a==b {return true}
    match (a,b) {
        (Value::Object(a),Value::Object(b))=>a.len()==b.len() && a.iter().all(|(k,v)|b.get(k).is_some_and(|x|metadata_roundtrip_equivalent(v,x))),
        (Value::Array(a),Value::Array(b))=>a.len()==b.len() && a.iter().zip(b).all(|(x,y)|metadata_roundtrip_equivalent(x,y)),
        (Value::Number(a),Value::Number(b)) if a.is_f64() && b.is_f64()=>{
            let a=a.as_f64().unwrap();let b=b.as_f64().unwrap();
            a.is_finite() && b.is_finite() && a.is_sign_negative()==b.is_sign_negative() && a.to_bits().abs_diff(b.to_bits())<=1
        },
        _=>false,
    }
}
fn verify_existing_sidecar(source:&Path,out:&Path,name:&str)->Result<usize,String> {
    let old=source.join(name);if !old.exists() {return Ok(0);}
    let mut count=0;
    for entry in fs::read_dir(old).map_err(|e|e.to_string())? {
        let entry=entry.map_err(|e|e.to_string())?;
        if !entry.file_type().map_err(|e|e.to_string())?.is_file() {return Err(format!("unexpected non-file in {name}"));}
        let a=fs::read(entry.path()).map_err(|e|e.to_string())?;
        let b=fs::read(out.join(name).join(entry.file_name())).map_err(|e|e.to_string())?;
        if a!=b {return Err(format!("original {name} sidecar changed; enrichment rejected"));}
        count+=1;
    }
    Ok(count)
}
fn run(source:&Path,out:&Path)->Result<Value,String> {
    if out.exists() {return Err("output exists; enrichment requires a fresh directory".into());}
    let man=validate_event(source).map_err(|e|e.to_string())?;
    let context=read(&source.join("context.json"))?;
    if context["load_basis"]!="frame_start_cluster" {return Err("only explicit frame-start recordings can be enriched".into());}
    let qs=context["qs_probe"].as_bool().ok_or("missing explicit QS probe contract")?;
    let preview=match context.get("load_preview") {
        None=>false,Some(v) if v["format"]=="rigid-contact-preview/1"=>true,_=>return Err("unknown preview contract".into()),
    };
    let scene=serde_json::from_slice(&fs::read(source.join("scene.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    record_event_options(scene,out,RecordMeta {event_id:man.event_id,structure_type:man.structure_type,load_type:man.load_type,
        notes:format!("verified history enrichment from {}",source.display())},man.n_frames,qs,LoadBasis::FrameStart,preview)?;
    let mut arrays=0;let mut bytes=0;
    for rel in std::iter::once(PathBuf::new()).chain((0..man.n_frames).map(|i|PathBuf::from(format!("frames/{i:06}")))) {
        let old=Catalog::load(&source.join(&rel).join("catalog.json"))?;
        let new=Catalog::load(&out.join(&rel).join("catalog.json"))?;
        if old!=new {return Err(format!("array catalog changed at {rel:?}"));}
        for (name,entry) in old.arrays {
            let a=fs::read(source.join(&rel).join(&entry.path)).map_err(|e|e.to_string())?;
            let b=fs::read(out.join(&rel).join(&entry.path)).map_err(|e|e.to_string())?;
            if a!=b {return Err(format!("original array changed: {rel:?}/{name}; enrichment rejected"));}
            arrays+=1;bytes+=a.len();
        }
    }
    let joint_files=verify_existing_sidecar(source,out,"joint_history")?;
    let response_files=verify_existing_sidecar(source,out,"bond_response")?;
    let old=read(&source.join("replay.json"))?;let new=read(&out.join("replay.json"))?;
    if !subset(&old["frames"],&new["frames"]) {return Err("original replay field changed; enrichment rejected".into());}
    let mut new_context=read(&out.join("context.json"))?;
    // Older preview recording parsed context twice without float_roundtrip.
    // Preserve those original model inputs, rather than silently correcting them.
    // The actual initial world centers must still match the original replay exactly,
    // and every original array and replay frame was already checked above.
    let retained_context_centers=context["centers"]!=new_context["centers"];
    if retained_context_centers {
        if new_context["centers"]!=old["frames"][0]["centers"] {
            return Err("initial world centers changed; enrichment rejected".into());
        }
        new_context["centers"]=context["centers"].clone();
    }
    let retained_generator_metadata=context.get("generator")!=new_context.get("generator");
    if retained_generator_metadata {
        let original=context.get("generator").ok_or("original generator metadata missing")?;
        let regenerated=new_context.get("generator").ok_or("regenerated generator metadata missing")?;
        if !metadata_roundtrip_equivalent(original,regenerated) {return Err("generator metadata changed beyond one-ULP JSON roundtrip; enrichment rejected".into())}
        new_context["generator"]=original.clone();
    }
    let mut old_context=context;old_context.as_object_mut().ok_or("invalid context")?.remove("recorder_revision");
    if !subset(&old_context,&new_context) {return Err("original context changed; enrichment rejected".into());}
    fs::write(out.join("context.json"),serde_json::to_vec_pretty(&new_context).unwrap()).map_err(|e|e.to_string())?;
    validate_event(out).map_err(|e|e.to_string())?;
    let report=json!({"verified":true,"source":source.canonicalize().map_err(|e|e.to_string())?,
        "arrays_bit_identical":arrays,"array_bytes_identical":bytes,"original_replay_fields_identical":true,
        "history_format":stress_learned::joint_history::FORMAT,"frames":man.n_frames,
        "response_format":stress_learned::response_history::FORMAT,
        "existing_joint_history_files_bit_identical":joint_files,"existing_response_files_bit_identical":response_files,
        "original_context_inputs_identical":true,"retained_original_context_centers":retained_context_centers,
        "retained_original_generator_metadata":retained_generator_metadata,
        "scope":"Deterministic re-recording with all original arrays, QS/preview, replay fields and existing history sidecars verified; adds exact joint/response history. Original files unchanged."});
    fs::write(out.join("enrichment-verification.json"),serde_json::to_vec_pretty(&report).unwrap()).map_err(|e|e.to_string())?;
    Ok(report)
}
fn main() {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=2 {eprintln!("usage: learned_impact_enrich SOURCE_EVENT NEW_OUTPUT");std::process::exit(2);}
    let start=Instant::now();match run(Path::new(&args[0]),Path::new(&args[1])) {
        Ok(r)=>println!("{}",json!({"seconds":start.elapsed().as_secs_f64(),"verification":r})),
        Err(error)=>{eprintln!("enrichment failed: {error}");std::process::exit(1);}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_verification_allows_only_additions_not_changed_or_missing_fields() {
        let a=json!({"frames":[{"center":[0.,1.,2.],"active":true}]});
        assert!(subset(&a,&json!({"frames":[{"center":[0.,1.,2.],"active":true,"new":3}]})));
        assert!(!subset(&a,&json!({"frames":[{"center":[0.,1.,3.],"active":true}]})));
        assert!(!subset(&a,&json!({"frames":[{"center":[0.,1.,2.]}]})));
    }
    #[test]
    fn saved_scene_json_round_trip_retains_generated_float_values() {
        let job=stress_learned::layouts::layout_job(stress_learned::layouts::wall_layouts()[0],"00",2,101,8);
        let bytes=serde_json::to_vec(&job.scene).unwrap();
        let restored:stress_ref::scene::Scene=serde_json::from_slice(&bytes).unwrap();
        assert!(serde_json::to_value(&job.scene).unwrap()==serde_json::to_value(restored).unwrap(),"saved scene numbers changed when parsed");
    }
    #[test]
    fn recipe_metadata_retention_rejects_real_changes() {
        let a=json!({"recipe":"point","point":[0.99,1.,0.],"seed":101});
        let mut b=a.clone();b["point"][0]=json!(f64::from_bits(0.99f64.to_bits()+1));
        assert!(metadata_roundtrip_equivalent(&a,&b));assert!(!subset(&a,&b));
        b["point"][0]=json!(f64::from_bits(0.99f64.to_bits()+2));assert!(!metadata_roundtrip_equivalent(&a,&b));
        b=a.clone();b["seed"]=json!(102);assert!(!metadata_roundtrip_equivalent(&a,&b));
        b=a.clone();b["recipe"]=json!("different");assert!(!metadata_roundtrip_equivalent(&a,&b));
        b=a.clone();b["extra"]=json!(true);assert!(!metadata_roundtrip_equivalent(&a,&b));
        b=a.clone();b.as_object_mut().unwrap().remove("point");assert!(!metadata_roundtrip_equivalent(&a,&b));
    }
}
