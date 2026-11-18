//! The TypeScript the web uses for the data crossing a binding is generated from the Rust
//! types (`rubble::play::api::typescript`) and checked in; this fails when it is stale.
//! `BLESS=1 cargo test -p engine-world --test api_types` rewrites it. Also: every scene
//! operation (`rubble::play::api::HOST_API`) has its method in the web module
//! (engine-world-wasm's source; the C ABI's check is engine-ffi's `tests/header.rs`).

use std::path::PathBuf;

#[test]
fn generated_types_are_current() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/types/rubble-play.d.ts");
    let want = rubble::play::api::typescript();
    if std::env::var_os("BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &want).unwrap();
        return;
    }
    // A Windows checkout (core.autocrlf) has CRLF line endings; compare the text.
    let have = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .replace("\r\n", "\n");
    assert!(
        have == want,
        "{} is stale: run BLESS=1 cargo test -p engine-world --test api_types",
        path.display()
    );
}

#[test]
fn every_referenced_type_is_declared() {
    let ts = rubble::play::api::typescript();
    let declared: std::collections::HashSet<&str> = ts
        .lines()
        .filter_map(|l| l.strip_prefix("export type "))
        .filter_map(|l| l.split([' ', '<', '=']).next())
        .collect();
    // Every capitalised identifier used in a type is declared in the file.
    // Code only: line comments and every `/** ... */` doc comment removed.
    let mut code = String::new();
    let mut rest = ts.as_str();
    while let Some(i) = rest.find("/**") {
        code.push_str(&rest[..i]);
        rest = &rest[i..];
        rest = rest.find("*/").map_or("", |j| &rest[j + 2..]);
    }
    code.push_str(rest);
    for word in code
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .flat_map(|l| l.split(|c: char| !c.is_alphanumeric() && c != '_'))
    {
        if word.chars().next().is_some_and(|c| c.is_ascii_uppercase())
            && !["Array", "ArrayLike", "Record", "Partial"].contains(&word)
        {
            assert!(declared.contains(word), "{word} is used but not declared");
        }
    }
}

/// Every scene operation has its JavaScript name in engine-world-wasm: a `js_name` (or a
/// plain method of that name) in the binding's source.
#[test]
fn every_scene_operation_is_bound_in_wasm() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine-world-wasm/src");
    let mut src = String::new();
    for e in std::fs::read_dir(&dir).unwrap() {
        src += &std::fs::read_to_string(e.unwrap().path()).unwrap();
    }
    let snake = |s: &str| {
        s.chars()
            .flat_map(|c| {
                if c.is_ascii_uppercase() {
                    vec!['_', c.to_ascii_lowercase()]
                } else {
                    vec![c]
                }
            })
            .collect::<String>()
    };
    let missing: Vec<_> = rubble::play::api::HOST_API
        .iter()
        .filter(|op| {
            !src.contains(&format!("js_name = {}", op.wasm))
                && !src.contains(&format!("pub fn {}(", snake(op.wasm)))
                && !src.contains(&format!("pub async fn {}(", snake(op.wasm)))
        })
        .map(|op| op.wasm)
        .collect();
    assert!(
        missing.is_empty(),
        "not bound in engine-world-wasm: {missing:?}"
    );
}
