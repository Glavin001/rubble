//! Array catalog + little-endian raw binary IO.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArrayEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Catalog {
    pub arrays: BTreeMap<String, ArrayEntry>,
}

impl Catalog {
    pub fn new() -> Self {
        Self { arrays: BTreeMap::new() }
    }

    pub fn insert(&mut self, name: &str, dtype: &str, shape: &[usize], rel: &str) {
        self.arrays.insert(
            name.to_string(),
            ArrayEntry { dtype: dtype.to_string(), shape: shape.to_vec(), path: rel.to_string() },
        );
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("catalog {}: {e}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, text).map_err(|e| e.to_string())
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn write_f32(root: &Path, rel: &str, shape: &[usize], data: &[f32]) -> Result<(), String> {
    if data.len() != numel(shape) {
        return Err(format!("{rel}: len {} != shape {:?}", data.len(), shape));
    }
    let path = root.join(rel);
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let mut f = File::create(&path).map_err(|e| e.to_string())?;
    for &x in data {
        f.write_all(&x.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn write_i32(root: &Path, rel: &str, shape: &[usize], data: &[i32]) -> Result<(), String> {
    if data.len() != numel(shape) {
        return Err(format!("{rel}: len {} != shape {:?}", data.len(), shape));
    }
    let path = root.join(rel);
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let mut f = File::create(&path).map_err(|e| e.to_string())?;
    for &x in data {
        f.write_all(&x.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn read_f32(root: &Path, entry: &ArrayEntry) -> Result<Vec<f32>, String> {
    if entry.dtype != "f32" {
        return Err(format!("{}: expected f32, got {}", entry.path, entry.dtype));
    }
    let n = numel(&entry.shape);
    let mut buf = vec![0u8; n * 4];
    File::open(root.join(&entry.path))
        .and_then(|mut f| f.read_exact(&mut buf))
        .map_err(|e| format!("{}: {e}", entry.path))?;
    Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

pub fn read_i32(root: &Path, entry: &ArrayEntry) -> Result<Vec<i32>, String> {
    if entry.dtype != "i32" {
        return Err(format!("{}: expected i32, got {}", entry.path, entry.dtype));
    }
    let n = numel(&entry.shape);
    let mut buf = vec![0u8; n * 4];
    File::open(root.join(&entry.path))
        .and_then(|mut f| f.read_exact(&mut buf))
        .map_err(|e| format!("{}: {e}", entry.path))?;
    Ok(buf.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
}

pub fn frame_dir(event: &Path, index: u32) -> PathBuf {
    event.join("frames").join(format!("{index:06}"))
}
