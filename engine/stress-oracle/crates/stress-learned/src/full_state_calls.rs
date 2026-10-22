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

use crate::full_state::{contract_value, load_frame, unshred};
use stress_ref::world::World;
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
    /// Per slot row: the call's node index of chunk a and chunk b (-1 outside the island).
    pair_local: Vec<i64>,
}

struct Pending {
    bond_key: Vec<i64>,
    sides: Vec<Side>,
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
    let node: HashMap<(usize, usize), i64> = chunks.iter().enumerate().map(|(j, &k)| (k, j as i64)).collect();
    let (mut pair_key, mut pair, mut pair_local) = (Vec::new(), Vec::new(), Vec::new());
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
                pair_local.extend([node.get(&(k[0], k[1])).copied().unwrap_or(-1), node.get(&(k[2], k[3])).copied().unwrap_or(-1)]);
                pair.extend(row.iter().map(|(_, x)| *x).chain(std::iter::repeat(f64::NAN)).take(4));
            }
        }
    }
    Ok(Side {
        chunk_keys: chunks.iter().map(|&(s, c)| stress_ref::keys::chunk_key(s, c)).collect(),
        bond_keys: bonds.iter().map(|&(s, b)| ((s as i64) << 32) | b as i64).collect(),
        impactor_keys: impactors.iter().map(|&k| k as i64).collect(),
        pair_local,
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

/// Adds the `/5` arrays to every call of a recorded event with a full state; returns the
/// column names for the manifest and the event-level static tables (None: the event has no
/// full state).
pub fn augment(dir: &Path, ctx: &Json, calls: &mut [Call]) -> Result<Option<(Json, Vec<(String, Arr)>)>, String> {
    if ctx["full_state"].is_null() {
        return Ok(None);
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
    augment_trees(ctx, calls, &mut |c: &Call| Ok((tree(c.frame)?, Some(tree(c.frame + 1)?))), None).map(Some)
}

/// The `/5` arrays of calls on a live world (harness A's provider calls, DAgger relabels):
/// `start` is the frame's start (`fs_*_in`), `end`, when given, its end (`fs_*_out`).
/// With `columns` (a training manifest's `full_state_columns`), exactly those columns and
/// that static split are written (NaN for a column the world lacks), so a live call matches
/// the recorded calls a model was trained on; without, the split is this call's own
/// (columns constant from start to end are static). Every call is on the same frame.
pub fn augment_live(start: &World, end: Option<&World>, ctx: &Json, calls: &mut [Call], columns: Option<&Json>) -> Result<(Json, Vec<(String, Arr)>), String> {
    let t0 = contract_value(start)?;
    let t1 = end.map(contract_value).transpose()?;
    augment_trees(ctx, calls, &mut |_: &Call| Ok((t0.clone(), t1.clone())), columns)
}

type Trees<'a> = dyn FnMut(&Call) -> Result<(Value, Option<Value>), String> + 'a;

fn names_of(columns: &Json, k: &str) -> Vec<String> {
    columns[k].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn augment_trees(ctx: &Json, calls: &mut [Call], trees: &mut Trees, columns: Option<&Json>) -> Result<(Json, Vec<(String, Arr)>), String> {
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
        let (t0, t1) = trees(c)?;
        let mut sides = vec![side(&t0, &chunks, &bonds, &impactors)?];
        if let Some(t1) = t1 {
            sides.push(side(&t1, &chunks, &bonds, &impactors)?);
        }
        pending.push(Pending { bond_key, sides });
    }
    let rows_of = |kind: usize| -> Vec<(&[i64], &Rows)> {
        pending
            .iter()
            .flat_map(|p| p.sides.iter())
            .map(|s| match kind {
                0 => (s.chunk_keys.as_slice(), &s.chunk),
                1 => (s.bond_keys.as_slice(), &s.bond),
                2 => (&[][..], &s.cluster),
                _ => (s.impactor_keys.as_slice(), &s.impactor),
            })
            .collect()
    };
    // Columns: given (a training manifest's), or the union over these calls with the
    // static split their boundaries show (static: the same value for each entity at every
    // boundary; clusters split and merge, so all theirs are per call).
    let (cols, static_names): (Vec<Vec<String>>, Vec<Vec<String>>) = match columns {
        Some(c) => (
            ["chunk", "bond", "cluster", "impactor"].iter().map(|k| names_of(c, k)).collect(),
            ["static_chunk", "static_bond", "", "static_impactor"].iter().map(|k| if k.is_empty() { Vec::new() } else { names_of(c, k) }).collect(),
        ),
        None => {
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
            let mut cols = all.clone();
            let mut stat = vec![Vec::new(); 4];
            for kind in [0usize, 1, 3] {
                let (fixed, _) = static_columns(&rows_of(kind), &all[kind]);
                stat[kind] = all[kind].iter().zip(&fixed).filter(|(_, &f)| f).map(|(c, _)| c.clone()).collect();
                cols[kind] = all[kind].iter().zip(&fixed).filter(|(_, &f)| !f).map(|(c, _)| c.clone()).collect();
            }
            (cols, stat)
        }
    };
    let mut statics: Vec<(String, Arr)> = Vec::new();
    for (kind, name) in [(0usize, "chunk"), (1, "bond"), (3, "impactor")] {
        // each entity's static values, at the first boundary it appears in
        let mut values: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
        let k = static_names[kind].len();
        for (keys, r) in rows_of(kind) {
            let m = r.matrix(&static_names[kind]);
            for (j, &key) in keys.iter().enumerate() {
                values.entry(key).or_insert_with(|| m[j * k..(j + 1) * k].to_vec());
            }
        }
        // bonds by `keys::bond_key` (structure, source), as fs_bond_key and the calls' bond_key
        let keys: Vec<i64> = values.keys().map(|&key| if kind == 1 { stress_ref::keys::bond_key((key >> 32) as usize, source_of[&((key >> 32) as usize, (key & 0xffff_ffff) as usize)]) } else { key }).collect();
        let data: Vec<f64> = values.values().flatten().copied().collect();
        statics.push((format!("fs_static_{name}_key"), Arr::I64(vec![keys.len()], keys.clone())));
        if kind == 1 {
            // Per-bond scales for nondimensional inputs, from its static columns.
            let col = |n: &str| static_names[1].iter().position(|c| c == n);
            let get = |row: &[f64], n: &str| col(n).map_or(f64::NAN, |i| row[i]);
            let scales: Vec<f64> = data
                .chunks(k.max(1))
                .take(keys.len())
                .flat_map(|row| {
                    let (ft, a, kn, st1, kb) = (get(row, "strength.tensile"), get(row, "geometry.area"), get(row, "stiffness.kn"), get(row, "geometry.s_t1"), get(row, "stiffness.kb_t1"));
                    [ft * a / kn, ft * st1 / kb, get(row, "strength.g_tension") * a, get(row, "strength.g_shear") * a, get(row, "strength.g_compression") * a, ft]
                })
                .collect();
            statics.push(("fs_static_bond_scale".into(), Arr::F64(vec![keys.len(), 6], scales)));
        }
        statics.push((format!("fs_static_{name}"), Arr::F64(vec![keys.len(), k], data)));
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
            c.arrays.push((format!("fs_pair_local_{x}"), Arr::I64(vec![np, 2], s.pair_local)));
        }
    }
    Ok((json!({"chunk": cols[0], "bond": cols[1], "cluster": cols[2], "impactor": cols[3], "pair": ["offset", "x", "y", "z"],
        "pair_local": ["node_a", "node_b"],
        "static_bond_scale": ["u_c", "theta_c", "gf_area_tension", "gf_area_shear", "gf_area_compression", "f_t"],
        "scale_notes": "u_c = f_t A / k_n (m), theta_c = f_t S_t1 / k_b_t1 (rad), G_f A (J) per mode, f_t (Pa); kappa_fail is the joint law's kappa_fail(kind, r) with r = joint.ductility (tension/shear) or joint.ductility_c (compression): brittle max(r, 1), ductile (r + 1) / 2 (stress_ref::joint::kappa_fail); infinite ductility (at rest) -> infinite kappa_fail",
        "static_chunk": static_names[0], "static_bond": static_names[1], "static_impactor": static_names[3],
        "static_rule": "columns with the same value for each entity at every boundary of the event: one row per entity in event_arrays fs_static_<kind> (keys fs_static_<kind>_key: chunk_key, bond_key, impactor index), left out of the calls; augment_live with a training manifest's columns writes that same split", "pair_key": ["structure_a", "chunk_a", "structure_b", "chunk_b", "slot"],
        "max_list": MAX_LIST, "rule": "every serialized field of the entity at the boundary (in: the frame's start, out: its end), flattened: a.b struct fields, a[i] list elements (lists longer than max_list left out), a=Variant one-hot enum variants (missing: 0), None and absent NaN; chunk columns prefixed @ are the world's per-chunk arrays"}), statics))
}

/// The island `chunks` (stable keys) and `impactors` (indices) of a call as members.
pub fn members_of(chunks: &[i64], impactors: &[i64]) -> stress_ref::world::IslandMembers {
    stress_ref::world::IslandMembers { chunks: chunks.iter().map(|&k| ((k >> 32) as usize, (k & 0xffff_ffff) as usize)).collect(), impactors: impactors.iter().map(|&k| k as usize).collect() }
}

/// Each recorded call's island penalty contact energy at its frame's start and end
/// (`energy`'s `E_contact_start`, `E_contact_end`; P1's builder revision 9, 11460bd7): the
/// World rebuilt from the recorded full state at both boundaries,
/// `World::island_contact_energy` over the call's chunks and impactors (the contact law the
/// host's ledger keeps, `contact_bound`). Exact; no re-simulation.
pub fn add_contact_energy(dir: &Path, scene: &stress_ref::scene::Scene, calls: &mut [Call]) -> Result<(), String> {
    let mut worlds: HashMap<usize, World> = HashMap::new();
    for c in calls.iter_mut() {
        let m = members_of(&i64s(c, "chunk_key")?, &i64s(c, "impactor_key")?);
        for (b, key) in [(c.frame, "E_contact_start"), (c.frame + 1, "E_contact_end")] {
            if !worlds.contains_key(&b) {
                worlds.retain(|&k, _| k + 1 >= c.frame);
                worlds.insert(b, crate::full_state::restore(scene, &load_frame(dir, b)?)?);
            }
            c.energy.insert(key.to_string(), worlds[&b].island_contact_energy(&m));
        }
    }
    Ok(())
}
