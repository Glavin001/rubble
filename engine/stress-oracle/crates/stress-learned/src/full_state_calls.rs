//! Calls `/5`: each call's island with the oracle's complete state at its frame's start
//! and end, from the recorder's full state (revision 9 on), next to the `/4` arrays.
//!
//! Each island entity's serialized fields are flattened, mechanically, into named f64
//! columns. A struct field becomes `a.b`, a list element `a[i]`, an enum's unit variant
//! the one-hot column `a=Variant`, and `None` NaN. Lists longer than `MAX_LIST`
//! (topology such as a cluster's chunk list) are left out; the call's graph carries them.
//! The arrays per call, `X` in {in, out} (boundary f, f + 1):
//!
//! - `fs_chunk_X [nc, Kc]`: `ChunkState` of the call's chunks (`chunk_key` order) plus the
//!   world's per-chunk arrays (`solver.f_int`, `loads`, `frame_loads`, `implicit_loads`,
//!   `contact_hits`, `chunk_contact_dts`, `exposure`), under their own prefixes;
//! - `fs_bond_X [nbf, Kb]`: `RtBond` of every bond with both ends in the island, live or
//!   broken (`fs_bond_key`, `keys::bond_key`);
//! - `fs_cluster_X [ncl, Kcl]`: the clusters of the island's chunks at that boundary
//!   (`fs_cluster_id_X`; `fs_chunk_cluster_X` is each chunk's row);
//! - `fs_impactor_X [ni, Ki]`: `Impactor` of the call's impactors;
//! - `fs_pair_key_X [P, 5]` (structure a, chunk a, structure b, chunk b, slot) and
//!   `fs_pair_X [P, 4]` (offset, then x, y, z): every persistent contact sample point
//!   (`pair_offsets`) with a chunk in the island.
//!
//! The column names of each kind, the union over the event's calls, are in the manifest's
//! `full_state_columns` and identical for `in` and `out`.
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use rmpv::Value;
use serde_json::{json, Value as Json};

use crate::full_state::{load_frame, unshred};
use crate::outcome_calls::{Arr, Call};

pub const MAX_LIST: usize = 16;
/// Entity fields left out: topology (the call's graph carries it) and names.
const SKIP: &[&str] = &["chunks", "bonds", "name", "children"];
const PER_CHUNK: &[&[&str]] = &[&["solver", "f_int"], &["loads", "force"], &["loads", "torque"], &["frame_loads", "force"], &["frame_loads", "torque"], &["implicit_loads", "force"], &["implicit_loads", "torque"], &["contact_hits"], &["chunk_contact_dts"], &["exposure"]];

fn field<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    v.as_map()?.iter().find(|(x, _)| x.as_str() == Some(k)).map(|(_, y)| y)
}

fn at<'a>(v: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter().try_fold(v, |v, k| field(v, k))
}

fn index(v: &Value, i: usize) -> Option<&Value> {
    v.as_array()?.get(i)
}

/// `v`'s leaves as (name, value).
pub fn flatten(v: &Value, prefix: &str, out: &mut Vec<(String, f64)>) {
    let join = |k: &str| if prefix.is_empty() { k.to_string() } else { format!("{prefix}.{k}") };
    match v {
        Value::F64(x) => out.push((prefix.to_string(), *x)),
        Value::F32(x) => out.push((prefix.to_string(), *x as f64)),
        Value::Integer(i) => out.push((prefix.to_string(), i.as_f64().unwrap_or(f64::NAN))),
        Value::Boolean(b) => out.push((prefix.to_string(), f64::from(u8::from(*b)))),
        Value::Nil => out.push((prefix.to_string(), f64::NAN)),
        Value::String(s) => out.push((format!("{prefix}={}", s.as_str().unwrap_or("?")), 1.0)),
        Value::Array(items) if items.len() <= MAX_LIST => {
            for (i, x) in items.iter().enumerate() {
                flatten(x, &format!("{prefix}[{i}]"), out);
            }
        }
        Value::Map(entries) if entries.iter().all(|(k, _)| k.is_str()) => {
            for (k, x) in entries {
                let k = k.as_str().unwrap();
                if prefix.is_empty() && SKIP.contains(&k) {
                    continue;
                }
                flatten(x, &join(k), out);
            }
        }
        _ => {}
    }
}

/// Rows of named values for one entity kind, before the event-wide column union.
#[derive(Default)]
struct Rows(Vec<Vec<(String, f64)>>);

impl Rows {
    fn names(&self, into: &mut BTreeSet<String>) {
        for r in &self.0 {
            into.extend(r.iter().map(|(n, _)| n.clone()));
        }
    }

    /// Missing one-hot columns are 0, other missing values NaN.
    fn matrix(&self, cols: &[String]) -> Vec<f64> {
        let at: HashMap<&str, usize> = cols.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
        let mut out = Vec::with_capacity(self.0.len() * cols.len());
        for r in &self.0 {
            let mut row: Vec<f64> = cols.iter().map(|c| if c.contains('=') { 0.0 } else { f64::NAN }).collect();
            for (n, x) in r {
                if let Some(&i) = at.get(n.as_str()) {
                    row[i] = *x;
                }
            }
            out.extend(row);
        }
        out
    }
}

struct Side {
    chunk_keys: Vec<i64>,
    bond_keys: Vec<i64>,
    impactor_keys: Vec<i64>,
    chunk: Rows,
    bond: Rows,
    cluster: Rows,
    cluster_id: Vec<i64>,
    chunk_cluster: Vec<i32>,
    impactor: Rows,
    pair_key: Vec<i64>,
    pair: Vec<f64>,
}

struct Pending {
    bond_key: Vec<i64>,
    sides: [Side; 2],
}

fn side(tree: &Value, chunks: &[(usize, usize)], bonds: &[(usize, usize)], impactors: &[usize]) -> Result<Side, String> {
    let solver = field(tree, "solver").ok_or("no solver")?;
    let (cs, bs, cls, imps) = (field(solver, "chunks").ok_or("chunks")?, field(solver, "bonds").ok_or("bonds")?, field(solver, "clusters").ok_or("clusters")?, field(tree, "impactors").ok_or("impactors")?);
    let mut chunk = Rows::default();
    let mut cluster_of = Vec::new();
    for &(s, c) in chunks {
        let st = index(cs, s).and_then(|x| index(x, c)).ok_or_else(|| format!("chunk {s}/{c}"))?;
        let mut row = Vec::new();
        flatten(st, "", &mut row);
        for p in PER_CHUNK {
            if let Some(v) = at(tree, p).and_then(|x| index(x, s)).and_then(|x| index(x, c)) {
                flatten(v, &format!("@{}", p.join(".")), &mut row);
            }
        }
        cluster_of.push(field(st, "cluster").and_then(|x| x.as_u64()).ok_or("chunk cluster")? as usize);
        chunk.0.push(row);
    }
    let mut ids: Vec<usize> = cluster_of.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    ids.sort_unstable();
    let row_of: HashMap<usize, i32> = ids.iter().enumerate().map(|(r, &ci)| (ci, r as i32)).collect();
    let mut cluster = Rows::default();
    let mut cluster_id = Vec::new();
    for &ci in &ids {
        let cl = index(cls, ci).ok_or_else(|| format!("cluster {ci}"))?;
        let mut row = Vec::new();
        flatten(cl, "", &mut row);
        cluster_id.push(field(cl, "id").and_then(|x| x.as_i64()).unwrap_or(ci as i64));
        cluster.0.push(row);
    }
    let mut bond = Rows::default();
    for &(s, b) in bonds {
        let v = index(bs, s).and_then(|x| index(x, b)).ok_or_else(|| format!("bond {s}/{b}"))?;
        let mut row = Vec::new();
        flatten(v, "", &mut row);
        bond.0.push(row);
    }
    let mut impactor = Rows::default();
    for &k in impactors {
        let mut row = Vec::new();
        flatten(index(imps, k).ok_or_else(|| format!("impactor {k}"))?, "", &mut row);
        impactor.0.push(row);
    }
    let island: HashSet<(usize, usize)> = chunks.iter().copied().collect();
    let (mut pair_key, mut pair) = (Vec::new(), Vec::new());
    if let Some(Value::Map(entries)) = field(tree, "pair_offsets") {
        let mut sorted: Vec<([usize; 4], &Value)> = entries
            .iter()
            .filter_map(|(k, v)| {
                let a = k.as_array()?;
                let key: Vec<usize> = a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect();
                (key.len() == 4).then(|| ([key[0], key[1], key[2], key[3]], v))
            })
            .filter(|(k, _)| island.contains(&(k[0], k[1])) || island.contains(&(k[2], k[3])))
            .collect();
        sorted.sort_by_key(|(k, _)| *k);
        for (k, v) in sorted {
            for (slot, pt) in v.as_array().into_iter().flatten().enumerate() {
                let mut row = Vec::new();
                flatten(pt, "", &mut row);
                pair_key.extend(k.iter().map(|&x| x as i64).chain([slot as i64]));
                pair.extend(row.iter().map(|(_, x)| *x).chain(std::iter::repeat(f64::NAN)).take(4));
            }
        }
    }
    Ok(Side {
        chunk_keys: chunks.iter().map(|&(s, c)| stress_ref::keys::chunk_key(s, c)).collect(),
        bond_keys: bonds.iter().map(|&(s, b)| ((s as i64) << 32) | b as i64).collect(),
        impactor_keys: impactors.iter().map(|&k| k as i64).collect(),
        chunk,
        bond,
        cluster, cluster_id, chunk_cluster: cluster_of.iter().map(|ci| row_of[ci]).collect(), impactor, pair_key, pair })
}

fn i64s(c: &Call, name: &str) -> Result<Vec<i64>, String> {
    match c.arrays.iter().find(|(k, _)| k == name) {
        Some((_, Arr::I64(_, d))) => Ok(d.clone()),
        _ => Err(format!("call lacks {name}")),
    }
}

/// Adds the `/5` arrays to every call of a recorded event with a full state; returns the
/// column names for the manifest (None: the event has no full state).
/// Columns whose value is the same for each entity at every boundary of the event (static
/// for this event), and each entity's static values.
fn static_columns(rows: &[(&[i64], &Rows)], cols: &[String]) -> (Vec<bool>, BTreeMap<i64, Vec<f64>>) {
    let at: HashMap<&str, usize> = cols.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
    let mut fixed = vec![true; cols.len()];
    let mut values: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (keys, r) in rows {
        let m = r.matrix(cols);
        for (j, &k) in keys.iter().enumerate() {
            let row = &m[j * cols.len()..(j + 1) * cols.len()];
            let v = values.entry(k).or_insert_with(|| row.to_vec());
            for (i, (a, b)) in v.iter().zip(row).enumerate() {
                if a.to_bits() != b.to_bits() {
                    fixed[i] = false;
                }
            }
        }
        let _ = &at;
    }
    (fixed, values)
}

pub fn augment(dir: &Path, ctx: &Json, calls: &mut [Call]) -> Result<Option<(Json, Vec<(String, Arr)>)>, String> {
    if ctx["full_state"].is_null() {
        return Ok(None);
    }
    // bond (structure, source) -> its index in the structure's bond list; endpoints as chunk keys.
    let mut source_of: HashMap<(usize, usize), usize> = HashMap::new();
    let mut by_structure: BTreeMap<usize, Vec<(usize, usize, usize)>> = BTreeMap::new();
    let mut count: HashMap<usize, usize> = HashMap::new();
    for k in ctx["bond_keys"].as_array().ok_or("bond_keys")? {
        let (s, src, a, b) = (k[0].as_u64().unwrap_or(0) as usize, k[1].as_u64().unwrap_or(0) as usize, k[2].as_u64().unwrap_or(0) as usize, k[3].as_u64().unwrap_or(0) as usize);
        let n = count.entry(s).or_default();
        source_of.insert((s, *n), src);
        by_structure.entry(s).or_default().push((*n, a, b));
        *n += 1;
    }
    let mut cache: HashMap<usize, Value> = HashMap::new();
    let mut tree = |b: usize| -> Result<Value, String> {
        if !cache.contains_key(&b) {
            if cache.len() > 4 {
                cache.clear();
            }
            cache.insert(b, unshred(&load_frame(dir, b)?)?);
        }
        Ok(cache[&b].clone())
    };
    let mut pending: Vec<Pending> = Vec::with_capacity(calls.len());
    for c in calls.iter() {
        let chunks: Vec<(usize, usize)> = i64s(c, "chunk_key")?.iter().map(|&k| ((k >> 32) as usize, (k & 0xffff_ffff) as usize)).collect();
        let impactors: Vec<usize> = i64s(c, "impactor_key")?.iter().map(|&k| k as usize).collect();
        let island: HashSet<(usize, usize)> = chunks.iter().copied().collect();
        let mut bonds = Vec::new();
        let mut bond_key = Vec::new();
        for (&s, list) in &by_structure {
            for &(bi, a, b) in list {
                if island.contains(&(s, a)) && island.contains(&(s, b)) {
                    bonds.push((s, bi));
                    bond_key.push(stress_ref::keys::bond_key(s, source_of[&(s, bi)]));
                }
            }
        }
        let t0 = tree(c.frame)?;
        let t1 = tree(c.frame + 1)?;
        pending.push(Pending { bond_key, sides: [side(&t0, &chunks, &bonds, &impactors)?, side(&t1, &chunks, &bonds, &impactors)?] });
    }
    let mut names: [BTreeSet<String>; 4] = Default::default();
    for p in &pending {
        for s in &p.sides {
            s.chunk.names(&mut names[0]);
            s.bond.names(&mut names[1]);
            s.cluster.names(&mut names[2]);
            s.impactor.names(&mut names[3]);
        }
    }
    let all: Vec<Vec<String>> = names.iter().map(|n| n.iter().cloned().collect()).collect();
    // Static columns (chunks, bonds, impactors; clusters split and merge, so all theirs are
    // per call) go to event-level tables once; the calls keep the rest.
    let mut statics: Vec<(String, Arr)> = Vec::new();
    let mut static_names: Vec<Vec<String>> = vec![Vec::new(); 4];
    let mut cols: Vec<Vec<String>> = all.clone();
    for (kind, name) in [(0usize, "chunk"), (1, "bond"), (3, "impactor")] {
        let rows: Vec<(&[i64], &Rows)> = pending
            .iter()
            .flat_map(|p| p.sides.iter())
            .map(|s| match kind {
                0 => (s.chunk_keys.as_slice(), &s.chunk),
                1 => (s.bond_keys.as_slice(), &s.bond),
                _ => (s.impactor_keys.as_slice(), &s.impactor),
            })
            .collect();
        let (fixed, values) = static_columns(&rows, &all[kind]);
        static_names[kind] = all[kind].iter().zip(&fixed).filter(|(_, &f)| f).map(|(c, _)| c.clone()).collect();
        cols[kind] = all[kind].iter().zip(&fixed).filter(|(_, &f)| !f).map(|(c, _)| c.clone()).collect();
        // bonds by `keys::bond_key` (structure, source), as fs_bond_key and the calls' bond_key
        let keys: Vec<i64> = values.keys().map(|&k| if kind == 1 { stress_ref::keys::bond_key((k >> 32) as usize, source_of[&((k >> 32) as usize, (k & 0xffff_ffff) as usize)]) } else { k }).collect();
        let data: Vec<f64> = values.values().flat_map(|v| v.iter().zip(&fixed).filter(|(_, &f)| f).map(|(x, _)| *x)).collect();
        statics.push((format!("fs_static_{name}_key"), Arr::I64(vec![keys.len()], keys.clone())));
        statics.push((format!("fs_static_{name}"), Arr::F64(vec![keys.len(), static_names[kind].len()], data)));
    }
    for (c, p) in calls.iter_mut().zip(pending) {
        c.arrays.push(("fs_bond_key".into(), Arr::I64(vec![p.bond_key.len()], p.bond_key)));
        for (x, s) in ["in", "out"].iter().zip(p.sides) {
            let put = |c: &mut Call, name: &str, rows: &Rows, cols: &[String]| {
                c.arrays.push((format!("fs_{name}_{x}"), Arr::F64(vec![rows.0.len(), cols.len()], rows.matrix(cols))));
            };
            put(c, "chunk", &s.chunk, &cols[0]);
            put(c, "bond", &s.bond, &cols[1]);
            put(c, "cluster", &s.cluster, &cols[2]);
            put(c, "impactor", &s.impactor, &cols[3]);
            c.arrays.push((format!("fs_cluster_id_{x}"), Arr::I64(vec![s.cluster_id.len()], s.cluster_id)));
            c.arrays.push((format!("fs_chunk_cluster_{x}"), Arr::I64(vec![s.chunk_cluster.len()], s.chunk_cluster.iter().map(|&r| r as i64).collect())));
            let np = s.pair_key.len() / 5;
            c.arrays.push((format!("fs_pair_key_{x}"), Arr::I64(vec![np, 5], s.pair_key)));
            c.arrays.push((format!("fs_pair_{x}"), Arr::F64(vec![np, 4], s.pair)));
        }
    }
    Ok(Some((json!({"chunk": cols[0], "bond": cols[1], "cluster": cols[2], "impactor": cols[3], "pair": ["offset", "x", "y", "z"],
        "static_chunk": static_names[0], "static_bond": static_names[1], "static_impactor": static_names[3],
        "static_rule": "columns with the same value for each entity at every boundary of the event: one row per entity in event_arrays fs_static_<kind> (keys fs_static_<kind>_key: chunk_key, bond_key, impactor index), left out of the calls", "pair_key": ["structure_a", "chunk_a", "structure_b", "chunk_b", "slot"],
        "max_list": MAX_LIST, "rule": "every serialized field of the entity at the boundary (in: the frame's start, out: its end), flattened: a.b struct fields, a[i] list elements (lists longer than max_list left out), a=Variant one-hot enum variants (missing: 0), None and absent NaN; chunk columns prefixed @ are the world's per-chunk arrays"}), statics)))
}
