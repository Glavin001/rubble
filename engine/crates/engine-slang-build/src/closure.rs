//! The files one compile reads: what its cache key covers.

use std::path::{Path, PathBuf};

/// The paths and texts of `source` and of every file it reaches by `#include "..."`
/// or `import name;` (None: a file missing, or one that uses `__include`). With `known`, an
/// include inside an `#ifdef NAME` block whose NAME is not in `known` is not followed.
pub(crate) fn include_closure(
    source: &Path,
    include_dirs: &[&Path],
    known: Option<&std::collections::HashSet<String>>,
) -> Option<Vec<(PathBuf, String)>> {
    let mut files = vec![source.canonicalize().ok()?];
    let mut texts = Vec::new();
    let mut at = 0;
    while at < files.len() {
        let file = files[at].clone();
        at += 1;
        let text = std::fs::read_to_string(&file).ok()?;
        // Per open conditional: whether its block may be compiled.
        let mut open: Vec<bool> = Vec::new();
        for line in text.lines() {
            let line = line.trim_start();
            if line.starts_with("__include") || line.starts_with("__import") {
                return None;
            }
            // `import name;`: the module `name.slang` (dots are directories), found as
            // slangc finds it: beside the importing file, then in the include
            // directories. One that is not there (slangc's own search would have to
            // find it) leaves the compile uncached.
            if let Some(rest) = line.strip_prefix("import ") {
                let name = rest.split(';').next()?.trim().replace('.', "/");
                let name = format!("{name}.slang");
                let found = std::iter::once(file.parent()?.join(&name))
                    .chain(include_dirs.iter().map(|d| d.join(&name)))
                    .find(|p| p.is_file())?;
                let found = found.canonicalize().ok()?;
                if !files.contains(&found) {
                    files.push(found);
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("#ifdef") {
                let name = rest.split_whitespace().next().unwrap_or("");
                open.push(known.is_none_or(|k| k.contains(name)));
            } else if line.starts_with("#if") {
                open.push(true);
            } else if line.starts_with("#else") || line.starts_with("#elif") {
                if let Some(last) = open.last_mut() {
                    *last = true;
                }
            } else if line.starts_with("#endif") {
                open.pop();
            }
            let Some(rest) = line.strip_prefix("#include") else {
                continue;
            };
            if open.iter().any(|live| !live) {
                continue;
            }
            let name = rest.trim().strip_prefix('"')?.split('"').next()?;
            // Relative to the including file, then to the include directories.
            let found = std::iter::once(file.parent()?.join(name))
                .chain(include_dirs.iter().map(|d| d.join(name)))
                .find(|p| p.is_file())?;
            let found = found.canonicalize().ok()?;
            if !files.contains(&found) {
                files.push(found);
            }
        }
        texts.push((file, text));
    }
    Some(texts)
}
