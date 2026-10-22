//! The oracle's complete dynamic state at a frame boundary, recorded mechanically
//! (`learned-impact-full-state/1`).
//!
//! The state contract is not written by hand. It is every field `stress_ref::world::World`
//! serializes (serde derives on the world and its solver; static data, the scene and the
//! engine coupling, is skipped and rebuilt from the scene). The serialized tree is shredded
//! into one typed column per field path:
//!
//! - a struct field is `parent.field`;
//! - the elements of a list are `parent[]`, and the list's lengths are its `lens`;
//! - a map (a `HashMap`, such as the contact sticks keyed by pair) is a variable-length
//!   table: keys under `parent{k}` and values under `parent{v}`, its entries sorted by
//!   key, its sizes in `lens`;
//! - every occurrence's kind (nil, bool, int, float, string, list, struct, map) is kept,
//!   run-length encoded, so options and enum variants round-trip.
//!
//! `World` -> columns -> `World` is exact (`restore`), and the completeness test steps a
//! restored frame and requires the next recorded frame bit for bit.
use std::collections::BTreeMap;

use rmpv::Value;
use stress_ref::scene::Scene;
use stress_ref::world::World;

pub const FULL_STATE_SCHEMA: &str = "learned-impact-full-state/1";

const NIL: u8 = 0;
const BOOL: u8 = 1;
const INT: u8 = 2;
const UINT: u8 = 3;
const F32: u8 = 4;
const F64: u8 = 5;
const STR: u8 = 6;
const BIN: u8 = 7;
const LIST: u8 = 8;
const STRUCT: u8 = 9;
const MAP: u8 = 10;
const EXT: u8 = 11;

pub const KIND_NAMES: [&str; 12] = ["nil", "bool", "int", "uint", "f32", "f64", "str", "bin", "list", "struct", "map", "ext"];

/// One field path's values over all its occurrences in a frame, in tree order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Column {
    /// (kind, run length).
    pub kinds: Vec<(u8, u32)>,
    pub bools: Vec<u8>,
    pub ints: Vec<i64>,
    pub uints: Vec<u64>,
    pub f32s: Vec<f32>,
    pub f64s: Vec<f64>,
    pub strs: Vec<String>,
    pub bins: Vec<Vec<u8>>,
    /// Lengths of lists and maps.
    pub lens: Vec<u64>,
    /// The field-name set of each struct occurrence (index into `Frame::keysets`).
    pub keysets: Vec<u32>,
    pub exts: Vec<(i8, Vec<u8>)>,
}

impl Column {
    fn kind(&mut self, k: u8) {
        match self.kinds.last_mut() {
            Some((last, n)) if *last == k => *n += 1,
            _ => self.kinds.push((k, 1)),
        }
    }

    pub fn occurrences(&self) -> usize {
        self.kinds.iter().map(|&(_, n)| n as usize).sum()
    }
}

/// A frame boundary's shredded state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    pub keysets: Vec<Vec<String>>,
    pub columns: BTreeMap<String, Column>,
}

struct Shred<'a> {
    frame: &'a mut Frame,
    keyset_ids: BTreeMap<Vec<String>, u32>,
}

impl Shred<'_> {
    fn col(&mut self, path: &str) -> &mut Column {
        if !self.frame.columns.contains_key(path) {
            self.frame.columns.insert(path.to_string(), Column::default());
        }
        self.frame.columns.get_mut(path).unwrap()
    }

    fn value(&mut self, path: &str, v: &Value) -> Result<(), String> {
        match v {
            Value::Nil => self.col(path).kind(NIL),
            Value::Boolean(b) => {
                let c = self.col(path);
                c.kind(BOOL);
                c.bools.push(u8::from(*b));
            }
            Value::Integer(i) => {
                let c = self.col(path);
                if let Some(x) = i.as_i64() {
                    c.kind(INT);
                    c.ints.push(x);
                } else {
                    c.kind(UINT);
                    c.uints.push(i.as_u64().ok_or("integer out of range")?);
                }
            }
            Value::F32(x) => {
                let c = self.col(path);
                c.kind(F32);
                c.f32s.push(*x);
            }
            Value::F64(x) => {
                let c = self.col(path);
                c.kind(F64);
                c.f64s.push(*x);
            }
            Value::String(s) => {
                let c = self.col(path);
                c.kind(STR);
                c.strs.push(s.as_str().ok_or("non-UTF-8 string")?.to_string());
            }
            Value::Binary(b) => {
                let c = self.col(path);
                c.kind(BIN);
                c.bins.push(b.clone());
            }
            Value::Ext(t, b) => {
                let c = self.col(path);
                c.kind(EXT);
                c.exts.push((*t, b.clone()));
            }
            Value::Array(items) => {
                let c = self.col(path);
                c.kind(LIST);
                c.lens.push(items.len() as u64);
                let child = format!("{path}[]");
                for x in items {
                    self.value(&child, x)?;
                }
            }
            Value::Map(entries) if !entries.is_empty() && entries.iter().all(|(k, _)| k.is_str()) => {
                let keys: Vec<String> = entries.iter().map(|(k, _)| k.as_str().unwrap().to_string()).collect();
                let next = self.keyset_ids.len() as u32;
                let id = *self.keyset_ids.entry(keys.clone()).or_insert_with(|| {
                    self.frame.keysets.push(keys.clone());
                    next
                });
                let c = self.col(path);
                c.kind(STRUCT);
                c.keysets.push(id);
                for (k, x) in entries {
                    let child = if path.is_empty() { k.as_str().unwrap().to_string() } else { format!("{path}.{}", k.as_str().unwrap()) };
                    self.value(&child, x)?;
                }
            }
            Value::Map(entries) => {
                // A table keyed by value: entries in the order of their encoded keys, so a
                // frame's columns do not depend on hash order.
                let mut sorted: Vec<(Vec<u8>, &Value, &Value)> = entries
                    .iter()
                    .map(|(k, x)| {
                        let mut b = Vec::new();
                        rmpv::encode::write_value(&mut b, k).map(|_| (b, k, x)).map_err(|e| e.to_string())
                    })
                    .collect::<Result<_, _>>()?;
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                let c = self.col(path);
                c.kind(MAP);
                c.lens.push(sorted.len() as u64);
                let (kp, vp) = (format!("{path}{{k}}"), format!("{path}{{v}}"));
                for (_, k, x) in sorted {
                    self.value(&kp, k)?;
                    self.value(&vp, x)?;
                }
            }
        }
        Ok(())
    }
}

/// Shred a serialized tree.
pub fn shred_value(v: &Value) -> Result<Frame, String> {
    let mut frame = Frame::default();
    Shred { frame: &mut frame, keyset_ids: BTreeMap::new() }.value("", v)?;
    Ok(frame)
}

/// The world's serialized tree (`World::save_state`).
pub fn world_value(w: &World) -> Result<Value, String> {
    let bytes = w.save_state()?;
    rmpv::decode::read_value(&mut bytes.as_slice()).map_err(|e| format!("decode state: {e}"))
}

/// Pure caches: maps from a key to a pure function of it, filled as the solver goes (and by
/// calls outside the oracle's step, such as the routing's prediction). Left out of the
/// contract (recorded empty): a world restored with them empty steps bit for bit like one
/// with them full (the completeness gate checks it).
pub const CACHES: &[&str] = &[
    // `stable_dt`'s per-cluster bound, keyed by (bonds, chunks, inertia-scale bits).
    "solver.step_bounds",
];

/// The contract's tree of a world: its serialized state with the pure caches emptied.
pub fn contract_value(w: &World) -> Result<Value, String> {
    let mut v = world_value(w)?;
    for c in CACHES {
        reset_path(&mut v, c);
    }
    Ok(v)
}

pub fn shred_world(w: &World) -> Result<Frame, String> {
    shred_value(&contract_value(w)?)
}

#[derive(Default)]
struct Cursor {
    run: usize,
    in_run: u32,
    bools: usize,
    ints: usize,
    uints: usize,
    f32s: usize,
    f64s: usize,
    strs: usize,
    bins: usize,
    lens: usize,
    keysets: usize,
    exts: usize,
}

struct Build<'a> {
    frame: &'a Frame,
    cursors: BTreeMap<&'a str, Cursor>,
}

impl<'a> Build<'a> {
    fn next_kind(&mut self, path: &str) -> Result<(&'a Column, &mut Cursor, u8), String> {
        let (key, col) = self.frame.columns.get_key_value(path).ok_or_else(|| format!("no column {path:?}"))?;
        let cur = self.cursors.entry(key.as_str()).or_default();
        let (k, n) = *col.kinds.get(cur.run).ok_or_else(|| format!("{path}: more occurrences than recorded"))?;
        cur.in_run += 1;
        if cur.in_run == n {
            cur.run += 1;
            cur.in_run = 0;
        }
        Ok((col, cur, k))
    }

    fn value(&mut self, path: &str) -> Result<Value, String> {
        let (col, cur, k) = self.next_kind(path)?;
        let bad = || format!("{path}: short column");
        macro_rules! take {
            ($v:ident) => {{
                let i = cur.$v;
                cur.$v += 1;
                col.$v.get(i).ok_or_else(bad)?
            }};
        }
        Ok(match k {
            NIL => Value::Nil,
            BOOL => Value::Boolean(*take!(bools) != 0),
            INT => Value::from(*take!(ints)),
            UINT => Value::from(*take!(uints)),
            F32 => Value::F32(*take!(f32s)),
            F64 => Value::F64(*take!(f64s)),
            STR => Value::from(take!(strs).as_str()),
            BIN => Value::Binary(take!(bins).clone()),
            EXT => {
                let (t, b) = take!(exts);
                Value::Ext(*t, b.clone())
            }
            LIST => {
                let n = *take!(lens) as usize;
                let child = format!("{path}[]");
                Value::Array((0..n).map(|_| self.value(&child)).collect::<Result<_, _>>()?)
            }
            STRUCT => {
                let id = *take!(keysets) as usize;
                let keys = self.frame.keysets.get(id).ok_or_else(|| format!("{path}: keyset {id}"))?;
                let mut out = Vec::with_capacity(keys.len());
                for name in keys {
                    let child = if path.is_empty() { name.clone() } else { format!("{path}.{name}") };
                    out.push((Value::from(name.as_str()), self.value(&child)?));
                }
                Value::Map(out)
            }
            MAP => {
                let n = *take!(lens) as usize;
                let (kp, vp) = (format!("{path}{{k}}"), format!("{path}{{v}}"));
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    let key = self.value(&kp)?;
                    out.push((key, self.value(&vp)?));
                }
                Value::Map(out)
            }
            k => return Err(format!("{path}: kind {k}")),
        })
    }
}

/// The tree a frame was shredded from (map entries in key order).
pub fn unshred(frame: &Frame) -> Result<Value, String> {
    let mut b = Build { frame, cursors: BTreeMap::new() };
    let v = b.value("")?;
    for (path, col) in &frame.columns {
        let used = b.cursors.get(path.as_str()).map_or(0, |c| c.run);
        if used != col.kinds.len() {
            return Err(format!("{path}: {} of {} kind runs used", used, col.kinds.len()));
        }
    }
    Ok(v)
}

/// The world at a recorded frame boundary.
pub fn restore(scene: &Scene, frame: &Frame) -> Result<World, String> {
    let mut bytes = Vec::new();
    rmpv::encode::write_value(&mut bytes, &unshred(frame)?).map_err(|e| e.to_string())?;
    World::restore_state(scene, &bytes)
}

/// The oracle's frame as the recorder steps it: peaks and load impulses reset, one frame,
/// an observer per substep (the recorder's watch; it only reads).
pub fn oracle_frame(w: &mut World, observe: impl FnMut(&World)) -> Result<(), String> {
    w.solver.reset_peak_utilizations();
    w.solver.reset_load_impulses();
    w.step_frame_watched(observe)
}

fn le<T: Copy>(v: &[T], f: impl Fn(T) -> Vec<u8>) -> Value {
    Value::Binary(v.iter().flat_map(|&x| f(x)).collect())
}

/// A frame's on-disk encoding: one MessagePack map
/// `{schema, keysets, columns: {path: {kinds: [[kind, run]...], <part>: ...}}}`, numeric
/// parts as little-endian bytes (`bools` u8, `ints` i64, `uints` u64, `f32s`, `f64s`,
/// `lens` u64, `keysets` u32), `strs` a list of strings, `bins` a list of bytes, `exts`
/// a list of [type, bytes]. Empty parts are left out.
pub fn encode(frame: &Frame) -> Result<Vec<u8>, String> {
    let mut cols = Vec::with_capacity(frame.columns.len());
    for (path, c) in &frame.columns {
        let mut m: Vec<(Value, Value)> = vec![("kinds".into(), Value::Array(c.kinds.iter().map(|&(k, n)| Value::Array(vec![Value::from(k), Value::from(n)])).collect()))];
        let mut put = |name: &str, present: bool, v: Value| {
            if present {
                m.push((name.into(), v));
            }
        };
        put("bools", !c.bools.is_empty(), Value::Binary(c.bools.clone()));
        put("ints", !c.ints.is_empty(), le(&c.ints, |x| x.to_le_bytes().to_vec()));
        put("uints", !c.uints.is_empty(), le(&c.uints, |x| x.to_le_bytes().to_vec()));
        put("f32s", !c.f32s.is_empty(), le(&c.f32s, |x| x.to_le_bytes().to_vec()));
        put("f64s", !c.f64s.is_empty(), le(&c.f64s, |x| x.to_le_bytes().to_vec()));
        put("lens", !c.lens.is_empty(), le(&c.lens, |x| x.to_le_bytes().to_vec()));
        put("keysets", !c.keysets.is_empty(), le(&c.keysets, |x| x.to_le_bytes().to_vec()));
        put("strs", !c.strs.is_empty(), Value::Array(c.strs.iter().map(|s| Value::from(s.as_str())).collect()));
        put("bins", !c.bins.is_empty(), Value::Array(c.bins.iter().map(|b| Value::Binary(b.clone())).collect()));
        put("exts", !c.exts.is_empty(), Value::Array(c.exts.iter().map(|(t, b)| Value::Array(vec![Value::from(*t), Value::Binary(b.clone())])).collect()));
        cols.push((Value::from(path.as_str()), Value::Map(m)));
    }
    let v = Value::Map(vec![
        ("schema".into(), FULL_STATE_SCHEMA.into()),
        ("keysets".into(), Value::Array(frame.keysets.iter().map(|k| Value::Array(k.iter().map(|s| Value::from(s.as_str())).collect())).collect())),
        ("columns".into(), Value::Map(cols)),
    ]);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).map_err(|e| e.to_string())?;
    Ok(out)
}

fn chunks<const N: usize, T>(b: &[u8], f: impl Fn([u8; N]) -> T) -> Result<Vec<T>, String> {
    if b.len() % N != 0 {
        return Err(format!("{} bytes is not a multiple of {N}", b.len()));
    }
    Ok(b.chunks_exact(N).map(|c| f(c.try_into().unwrap())).collect())
}

pub fn decode(bytes: &[u8]) -> Result<Frame, String> {
    let v = rmpv::decode::read_value(&mut &bytes[..]).map_err(|e| e.to_string())?;
    let get = |m: &Value, k: &str| -> Option<Value> { m.as_map()?.iter().find(|(x, _)| x.as_str() == Some(k)).map(|(_, y)| y.clone()) };
    if get(&v, "schema").and_then(|s| s.as_str().map(String::from)).as_deref() != Some(FULL_STATE_SCHEMA) {
        return Err("not a full-state frame".into());
    }
    let strs = |x: &Value| -> Vec<String> { x.as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default() };
    let keysets = get(&v, "keysets").and_then(|k| k.as_array().map(|a| a.iter().map(strs).collect())).unwrap_or_default();
    let mut columns = BTreeMap::new();
    for (path, m) in get(&v, "columns").ok_or("columns")?.as_map().ok_or("columns map")? {
        let mut c = Column::default();
        let bin = |k: &str| -> Vec<u8> { get(m, k).and_then(|b| b.as_slice().map(|s| s.to_vec())).unwrap_or_default() };
        c.kinds = get(m, "kinds").and_then(|k| k.as_array().map(|a| a.iter().map(|r| (r[0].as_u64().unwrap_or(0) as u8, r[1].as_u64().unwrap_or(0) as u32)).collect())).unwrap_or_default();
        c.bools = bin("bools");
        c.ints = chunks::<8, _>(&bin("ints"), i64::from_le_bytes)?;
        c.uints = chunks::<8, _>(&bin("uints"), u64::from_le_bytes)?;
        c.f32s = chunks::<4, _>(&bin("f32s"), f32::from_le_bytes)?;
        c.f64s = chunks::<8, _>(&bin("f64s"), f64::from_le_bytes)?;
        c.lens = chunks::<8, _>(&bin("lens"), u64::from_le_bytes)?;
        c.keysets = chunks::<4, _>(&bin("keysets"), u32::from_le_bytes)?;
        c.strs = get(m, "strs").map(|x| strs(&x)).unwrap_or_default();
        c.bins = get(m, "bins").and_then(|x| x.as_array().map(|a| a.iter().filter_map(|b| b.as_slice().map(|s| s.to_vec())).collect())).unwrap_or_default();
        c.exts = get(m, "exts").and_then(|x| x.as_array().map(|a| a.iter().map(|e| (e[0].as_i64().unwrap_or(0) as i8, e[1].as_slice().unwrap_or(&[]).to_vec())).collect())).unwrap_or_default();
        columns.insert(path.as_str().ok_or("column path")?.to_string(), c);
    }
    Ok(Frame { keysets, columns })
}

/// Bitwise column comparison (f64 by bits: -0.0 and NaN payloads count), skipping the
/// paths `skip` returns true for. The differing paths, sorted.
pub fn differing_paths(a: &Frame, b: &Frame, skip: impl Fn(&str) -> bool) -> Vec<String> {
    let mut out = Vec::new();
    let paths: std::collections::BTreeSet<&String> = a.columns.keys().chain(b.columns.keys()).collect();
    for p in paths {
        if skip(p) {
            continue;
        }
        let same = match (a.columns.get(p), b.columns.get(p)) {
            (Some(x), Some(y)) => {
                x.kinds == y.kinds
                    && x.bools == y.bools
                    && x.ints == y.ints
                    && x.uints == y.uints
                    && x.f32s.iter().map(|v| v.to_bits()).eq(y.f32s.iter().map(|v| v.to_bits()))
                    && x.f64s.iter().map(|v| v.to_bits()).eq(y.f64s.iter().map(|v| v.to_bits()))
                    && x.strs == y.strs
                    && x.bins == y.bins
                    && x.lens == y.lens
                    && x.exts == y.exts
                    && x.keysets.iter().map(|&i| &a.keysets[i as usize]).eq(y.keysets.iter().map(|&i| &b.keysets[i as usize]))
            }
            _ => false,
        };
        if !same {
            out.push(p.clone());
        }
    }
    out
}

/// `p` or below `p`.
pub fn under(p: &str, root: &str) -> bool {
    p == root || p.strip_prefix(root).is_some_and(|r| r.starts_with('.') || r.starts_with('[') || r.starts_with('{'))
}

/// Every occurrence of `path`'s value reset to its kind's zero (0, false, "", an empty
/// list or map; a struct keeps its fields, reset below): the probe the field
/// classification uses.
pub fn reset_path(v: &mut Value, path: &str) {
    fn walk(v: &mut Value, here: &str, target: &str) {
        if here == target {
            zero(v);
            return;
        }
        if !target.starts_with(here) {
            return;
        }
        match v {
            Value::Array(items) => {
                let child = format!("{here}[]");
                for x in items {
                    walk(x, &child, target);
                }
            }
            Value::Map(entries) if !entries.is_empty() && entries.iter().all(|(k, _)| k.is_str()) => {
                for (k, x) in entries.iter_mut() {
                    let child = if here.is_empty() { k.as_str().unwrap().to_string() } else { format!("{here}.{}", k.as_str().unwrap()) };
                    walk(x, &child, target);
                }
            }
            Value::Map(entries) => {
                let (kp, vp) = (format!("{here}{{k}}"), format!("{here}{{v}}"));
                for (k, x) in entries.iter_mut() {
                    walk(k, &kp, target);
                    walk(x, &vp, target);
                }
            }
            _ => {}
        }
    }
    fn zero(v: &mut Value) {
        if let Value::Map(entries) = v {
            if !entries.is_empty() && entries.iter().all(|(k, _)| k.is_str()) {
                for (_, x) in entries.iter_mut() {
                    zero(x);
                }
                return;
            }
        }
        let z = match &*v {
            Value::Boolean(_) => Value::Boolean(false),
            Value::Integer(_) => Value::from(0),
            Value::F32(_) => Value::F32(0.0),
            Value::F64(_) => Value::F64(0.0),
            Value::String(_) => Value::from(""),
            Value::Binary(_) => Value::Binary(Vec::new()),
            Value::Array(_) => Value::Array(Vec::new()),
            Value::Map(_) => Value::Map(Vec::new()),
            other => other.clone(),
        };
        *v = z;
    }
    walk(v, "", path);
}

/// Writes an event's boundaries as `full_state/<boundary>.fs.zst`: every `every`-th one
/// self-contained, the others zstd frames with the previous boundary's encoding as their
/// dictionary (lossless; consecutive states share most bytes), so a read decodes at most
/// `every - 1` deltas (`load_frame`).
pub struct FullStateWriter {
    every: usize,
    level: i32,
    prev: Option<(usize, Vec<u8>)>,
    pub entries: Vec<serde_json::Value>,
    pub raw_bytes: u64,
    pub stored_bytes: u64,
}

impl FullStateWriter {
    pub fn new(every: usize, level: i32) -> FullStateWriter {
        FullStateWriter { every: every.max(1), level, prev: None, entries: Vec::new(), raw_bytes: 0, stored_bytes: 0 }
    }

    pub fn push(&mut self, dir: &std::path::Path, boundary: usize, w: &World) -> Result<(), String> {
        let bytes = encode(&shred_world(w)?)?;
        let rel = format!("full_state/{boundary:06}.fs.zst");
        let key = boundary % self.every == 0 || self.prev.is_none();
        let z = match (&self.prev, key) {
            (Some((_, prev)), false) => zstd::bulk::Compressor::with_dictionary(self.level, prev).and_then(|mut c| c.compress(&bytes)),
            _ => zstd::bulk::compress(&bytes, self.level),
        }
        .map_err(|e| format!("{rel}: zstd {e}"))?;
        std::fs::create_dir_all(dir.join("full_state")).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(&rel), &z).map_err(|e| format!("{rel}: {e}"))?;
        let delta_of = if key { None } else { self.prev.as_ref().map(|p| p.0) };
        self.entries.push(serde_json::json!({"boundary": boundary, "path": rel, "bytes": bytes.len(), "delta_of": delta_of}));
        self.raw_bytes += bytes.len() as u64;
        self.stored_bytes += z.len() as u64;
        self.prev = Some((boundary, bytes));
        Ok(())
    }
}

/// The encoded bytes of `event`'s boundary (resolving its delta chain).
pub fn load_frame_bytes(event: &std::path::Path, boundary: usize) -> Result<Vec<u8>, String> {
    let ctx: serde_json::Value = serde_json::from_slice(&std::fs::read(event.join("context.json")).map_err(|e| format!("context.json: {e}"))?).map_err(|e| e.to_string())?;
    let list = ctx["full_state"]["boundaries"].as_array().ok_or("no full_state in context.json")?;
    let entry = |b: usize| list.get(b).filter(|s| s["boundary"].as_u64() == Some(b as u64)).or_else(|| list.iter().find(|s| s["boundary"].as_u64() == Some(b as u64))).ok_or_else(|| format!("no full state at boundary {b}"));
    let mut chain = vec![entry(boundary)?];
    while let Some(d) = chain.last().unwrap()["delta_of"].as_u64() {
        chain.push(entry(d as usize)?);
    }
    let mut prev: Option<Vec<u8>> = None;
    for s in chain.iter().rev() {
        let rel = s["path"].as_str().ok_or("full state path")?;
        let z = std::fs::read(event.join(rel)).map_err(|e| format!("{rel}: {e}"))?;
        let cap = s["bytes"].as_u64().ok_or("full state bytes")? as usize;
        let raw = match &prev {
            Some(p) => zstd::bulk::Decompressor::with_dictionary(p).and_then(|mut d| d.decompress(&z, cap)),
            None => zstd::bulk::decompress(&z, cap),
        }
        .map_err(|e| format!("{rel}: zstd {e}"))?;
        prev = Some(raw);
    }
    Ok(prev.unwrap())
}

pub fn load_frame(event: &std::path::Path, boundary: usize) -> Result<Frame, String> {
    decode(&load_frame_bytes(event, boundary)?)
}
