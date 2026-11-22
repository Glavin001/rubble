//! A preset run through the C scene API (`rbCreateSceneWorld`, `rbWorld_PlaceScene`,
//! `rbWorld_Act`, `rbWorld_TakeEvents`, `rbWorld_GetRenderManifest`) does what the same
//! preset does in Rust (`rubble::play::Playground`): the same scene JSON, the same bodies,
//! the same script events at the same steps, the same draw groups. GPU results are not
//! compared bit for bit (two native worlds drift); everything the script decides is.

use rubble::play::{Action, Aim, Playground, SceneDef};
use rubble::GpuContext;
use rubble_ffi::coords::RbCoordinates;
use rubble_ffi::*;
use std::ffi::{c_char, CStr, CString};

fn last_error() -> String {
    unsafe {
        CStr::from_ptr(rbGetLastError())
            .to_string_lossy()
            .into_owned()
    }
}

unsafe fn json(p: *const c_char) -> serde_json::Value {
    serde_json::from_str(CStr::from_ptr(p).to_str().unwrap()).unwrap()
}

macro_rules! ok {
    ($e:expr) => {
        assert_eq!($e, RbResult::RB_OK, "{}: {}", stringify!($e), last_error())
    };
}

const STEPS: u32 = 140;

fn fire() -> Action {
    Action::Fire {
        aim: Some(Aim {
            eye: [0.0, 1.5, -6.0],
            dir: [0.0, 0.0, 1.0],
        }),
    }
}

#[test]
fn a_preset_plays_the_same_through_c_and_rust() {
    // Rust.
    let scene = rubble::play::api::preset_scene("bullets").unwrap();
    let mut play = Playground::new(GpuContext::Default, scene.clone()).unwrap();
    let mut rust_events = Vec::new();
    for s in 0..STEPS {
        if s == 10 {
            play.act(&fire()).unwrap();
        }
        play.step(1).unwrap();
        rust_events.extend(play.director.take_events());
    }
    let rust_events = serde_json::to_value(&rust_events).unwrap();
    let rust_manifest = serde_json::to_value(play.world().render_manifest()).unwrap();
    let rust_stats = play.stats();
    drop(play);

    unsafe {
        // The preset's scene through C is the Rust one.
        let mut p = std::ptr::null();
        let name = CString::new("bullets").unwrap();
        ok!(rbPresetScene(name.as_ptr(), &mut p));
        let c_scene = json(p);
        assert_eq!(c_scene, serde_json::to_value(&scene).unwrap());
        let scene_text = CString::new(c_scene.to_string()).unwrap();

        let mut w: *mut RbWorld = std::ptr::null_mut();
        ok!(rbCreateSceneWorld(
            scene_text.as_ptr(),
            RbCoordinates::RB_COORDINATES_Y_UP_METRES,
            &mut w
        ));
        // Acting before a scene is placed says so.
        let fire_text = CString::new(serde_json::to_string(&fire()).unwrap()).unwrap();
        assert_ne!(rbWorld_Act(w, fire_text.as_ptr()), RbResult::RB_OK);
        assert!(
            last_error().contains("rbWorld_PlaceScene"),
            "{}",
            last_error()
        );

        ok!(rbWorld_PlaceScene(w, scene_text.as_ptr(), &mut p));
        assert!(json(p).is_object());

        let mut c_events = Vec::new();
        let mut frame = std::mem::zeroed::<RbFrame>();
        for s in 0..STEPS {
            if s == 10 {
                ok!(rbWorld_Act(w, fire_text.as_ptr()));
            }
            ok!(rbWorld_Step(w));
            ok!(rbWorld_Poll(w, true, true, &mut frame));
            ok!(rbWorld_TakeEvents(w, &mut p));
            c_events.extend(json(p).as_array().unwrap().iter().cloned());
        }
        assert_eq!(serde_json::Value::Array(c_events), rust_events);
        assert!(
            rust_events.as_array().unwrap().len() >= 2,
            "the run should hold the player's fire and the script's own: {rust_events}"
        );

        ok!(rbWorld_GetRenderManifest(w, &mut p));
        let mut c_manifest = json(p);
        let mut r_manifest = rust_manifest;
        // Groups and planes are the scene's; the version counts this world's edits.
        for m in [&mut c_manifest, &mut r_manifest] {
            m.as_object_mut().unwrap().remove("version");
        }
        assert_eq!(c_manifest, r_manifest);
        let mut version = 0u64;
        ok!(rbWorld_GetRenderVersion(w, &mut version));
        assert!(version > 0);

        let mut pending = true;
        ok!(rbWorld_ScriptPending(w, &mut pending));
        ok!(rbWorld_GetScriptState(w, &mut p));
        assert!(frame.has_frame);
        assert_eq!(rust_stats.steps, u64::from(STEPS));
        rbDestroyWorld(w);
    }
}

#[test]
fn presets_and_queries_through_c() {
    unsafe {
        let mut p = std::ptr::null();
        ok!(rbPresetList(&mut p));
        let list = json(p);
        let list = list.as_array().unwrap();
        assert_eq!(list.len(), rubble::play::presets().len());
        assert!(list
            .iter()
            .any(|x| x["name"] == "bullets" && x["placed"] == true));

        let fam = CString::new("motion").unwrap();
        let q = CString::new("scene=ccd").unwrap();
        ok!(rbSceneFromQuery(fam.as_ptr(), q.as_ptr(), &mut p));
        let scene: SceneDef = serde_json::from_value(json(p)).unwrap();
        assert_eq!(scene.family().name(), "motion");

        let bad = CString::new("lab").unwrap();
        assert_ne!(
            rbSceneFromQuery(bad.as_ptr(), q.as_ptr(), &mut p),
            RbResult::RB_OK
        );
        assert!(
            last_error().contains("not placed through SceneDef"),
            "{}",
            last_error()
        );
    }
}
