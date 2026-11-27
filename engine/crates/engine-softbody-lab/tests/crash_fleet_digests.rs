//! Every reference scenario's spawned fleet, word for word: a digest of each run's layout and
//! fleet (every field of every `ExplicitBeams`, floats by their exact shortest form) in all
//! eight runs and three contact settings, against the digests committed in
//! `tests/data/crash_fleet_digests.json`. Written before the scenarios moved onto scenario
//! families (`crash_reference::family`), so it proves each family preset builds what its
//! scenario built. `BLESS=1` rewrites the file (only after a change meant to change a fleet).
//! `--release` (rigging the designs).

use engine_softbody::crash_reference::{self as cr, Contact, Jitter};
use std::collections::BTreeMap;

fn fnv(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// `<id>/<contact>/<run>` to the digest of its layout and its fleet.
fn digests() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for s in cr::all() {
        let templates = engine_softbody_authoring::crash_templates::templates_for(&s).expect("templates");
        for c in Contact::ALL {
            for run in 0..cr::RUNS {
                let b = s.build(&templates, Jitter::run(run), c).expect("build");
                let text = format!("{:?}|{:?}", b.layout, b.fleet);
                out.insert(format!("{}/{}/{run}", s.id, c.name()), fnv(&text));
            }
        }
    }
    out
}

#[test]
fn every_scenario_spawns_the_committed_fleet() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/crash_fleet_digests.json");
    let got = digests();
    if std::env::var("BLESS").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_string_pretty(&got).unwrap() + "\n").unwrap();
        return;
    }
    let want: BTreeMap<String, String> = serde_json::from_str(&std::fs::read_to_string(&path).expect("tests/data/crash_fleet_digests.json")).unwrap();
    let diff: Vec<&String> = want.keys().filter(|k| got.get(*k) != want.get(*k)).collect();
    assert!(diff.is_empty() && got.len() == want.len(), "{} of {} fleets differ: {diff:?}", diff.len(), want.len());
}
