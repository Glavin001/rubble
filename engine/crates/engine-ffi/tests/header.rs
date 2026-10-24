//! The committed header is the one the current sources generate.

#[test]
fn committed_header_is_current() {
    let generated = std::fs::read_to_string(concat!(env!("OUT_DIR"), "/rubble.h")).unwrap();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/include/rubble.h");
    if std::env::var("RUBBLE_FFI_BLESS").is_ok() {
        std::fs::write(path, &generated).unwrap();
    }
    let committed = std::fs::read_to_string(path).unwrap_or_default();
    assert!(
        committed == generated,
        "include/rubble.h is stale: run `RUBBLE_FFI_BLESS=1 cargo test -p engine-ffi --test header`"
    );
}
