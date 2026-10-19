use std::path::PathBuf;

use stress_learned::recorder::write_synthetic_tiny;
use stress_learned::validate_event;

#[test]
fn synthetic_event_validates() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("learned_impact_tiny");
    let path = write_synthetic_tiny(&dir).expect("write");
    let man = validate_event(&path).expect("validate");
    assert_eq!(man.schema_version, stress_learned::SCHEMA_VERSION);
    assert_eq!(man.n_frames, 2);
    assert_eq!(man.n_chunks, 4);
    assert_eq!(man.n_bonds, 3);
    assert_eq!(man.k, 4);
}
