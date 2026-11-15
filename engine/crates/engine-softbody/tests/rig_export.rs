//! The generated sedan's mod folder, as the web build makes it (build-web.sh runs the
//! `vehicle_gen_rig` example and caches its output by the sources' hash): the same
//! bytes on every run, and a files.json that lists every other file once.
use engine_softbody::authoring::rig;

#[test]
fn the_sedan_export_is_the_same_bytes_every_run_and_lists_its_files() {
    let a = rig::mod_files(&rig::rig_vehicle("sedan", None, None).unwrap());
    let b = rig::mod_files(&rig::rig_vehicle("sedan", None, None).unwrap());
    assert_eq!(a.len(), b.len());
    for ((pa, da), (pb, db)) in a.iter().zip(&b) {
        assert_eq!(pa, pb);
        assert!(da == db, "{pa} differs between two runs");
    }
    let index = &a.iter().find(|(p, _)| p == "files.json").unwrap().1;
    let mut listed: Vec<String> = serde_json::from_slice(index).unwrap();
    let mut files: Vec<String> = a
        .iter()
        .map(|(p, _)| p.clone())
        .filter(|p| p != "files.json")
        .collect();
    listed.sort();
    files.sort();
    assert_eq!(listed, files);
    for ext in [".jbeam", ".pc", ".glb", ".rskin"] {
        assert!(
            files.iter().any(|p| p.ends_with(ext)),
            "no {ext} in the export"
        );
    }
}
