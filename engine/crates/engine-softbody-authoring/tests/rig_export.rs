//! The generated sedan's mod folder, as the web build makes it (build-web.sh runs the
//! `vehicle_gen_rig` example and caches its output by the sources' hash): the same
//! bytes on every run, and a files.json that lists every other file once.
use engine_softbody_authoring::authoring::rig;

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

/// The lighter render (`vg_input::lite_art`, what a phone's vehicle lab draws): the same
/// meshes by name, fewer vertices and skin entries, bound to the same nodes; the mod
/// folder with it is the folder without it plus its two files (the physics files, the
/// art and its skins byte for byte the same).
#[test]
fn the_lighter_render_is_lighter_and_leaves_the_physics_files_alone() {
    use engine_softbody_authoring::authoring::gen2::vg_input;
    use engine_softbody_authoring::flexskin::Skin;
    let b = rig::rig_vehicle("sedan", None, None).unwrap();
    let lite = vg_input::lite_art(&b, "", vg_input::LITE_EDGE).unwrap();
    let art = &b.skin;
    let verts = |s: &Skin| s.meshes.iter().map(|m| m.verts.len()).sum::<usize>();
    let entries = |s: &Skin| {
        s.meshes
            .iter()
            .flat_map(|m| m.verts.iter())
            .map(|v| v.w.len())
            .sum::<usize>()
    };
    let names = |s: &Skin| s.meshes.iter().map(|m| m.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(&lite.1), names(art));
    assert_eq!(lite.1.nodes, art.nodes, "bound to the same nodes");
    assert!(
        verts(&lite.1) * 10 < verts(art) * 7,
        "vertices {} of {}",
        verts(&lite.1),
        verts(art)
    );
    assert!(
        entries(&lite.1) * 10 < entries(art) * 6,
        "entries {} of {}",
        entries(&lite.1),
        entries(art)
    );
    let without = vg_input::files(&b);
    let with = vg_input::files_with(&b, Some(&lite));
    let name = b.input.design().to_string();
    let extra = [
        format!("{name}/{}", vg_input::lite_glb_file(&name)),
        format!("{name}/{}", vg_input::lite_skin_file(&name)),
    ];
    let kept: Vec<_> = with
        .iter()
        .filter(|(p, _)| !extra.contains(p) && p != "files.json")
        .collect();
    let base: Vec<_> = without.iter().filter(|(p, _)| p != "files.json").collect();
    assert_eq!(kept.len(), base.len());
    for ((pa, da), (pb, db)) in kept.iter().zip(&base) {
        assert_eq!(pa, pb);
        assert!(da == db, "{pa} differs with the lighter render beside it");
    }
    for p in &extra {
        assert!(with.iter().any(|(q, _)| q == p), "{p} not written");
    }
}
