//! The Reference Studio's TypeScript is generated from the Rust types
//! (`crash_reference::types::typescript`) and checked in; this fails when it is stale.
//! `BLESS=1 cargo test -p engine-softbody --test studio_types` rewrites it.

use std::path::PathBuf;

fn path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../apps/reference-studio/src/generated/crash-reference.ts")
}

#[test]
fn studio_types_are_current() {
    let path = path();
    let want = engine_softbody::crash_reference::types::typescript();
    if std::env::var_os("BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &want).unwrap();
        return;
    }
    let have = std::fs::read_to_string(&path).unwrap_or_default().replace("\r\n", "\n");
    assert!(have == want, "{} is stale: run BLESS=1 cargo test -p engine-softbody --test studio_types", path.display());
}

/// Every capitalised identifier a declaration uses is declared in the file.
#[test]
fn every_referenced_type_is_declared() {
    let ts = engine_softbody::crash_reference::types::typescript();
    let declared: std::collections::HashSet<&str> = ts.lines().filter_map(|l| l.strip_prefix("export type ")).filter_map(|l| l.split([' ', '<', '=']).next()).collect();
    let mut code = String::new();
    let mut rest = ts.as_str();
    while let Some(i) = rest.find("/**") {
        code.push_str(&rest[..i]);
        rest = &rest[i..];
        rest = rest.find("*/").map_or("", |j| &rest[j + 2..]);
    }
    code.push_str(rest);
    for word in code.lines().filter(|l| !l.trim_start().starts_with("//")).flat_map(|l| l.split(|c: char| !c.is_alphanumeric() && c != '_')) {
        if word.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && !["Array", "Record", "Partial"].contains(&word) {
            assert!(declared.contains(word), "{word} is used but not declared");
        }
    }
}
