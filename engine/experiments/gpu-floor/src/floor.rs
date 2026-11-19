//! The floor kernel's host: the reference car packed into the kernel's layout, frames run
//! on the GPU, and the state compared word for word with the CPU reference.
use crate::gpu::{read, write, Gpu};
use engine_softbody::explicit_beams::{exact_length, dihedral_jacobian, ExplicitBeams, ExplicitKind, Node};
use engine_softbody::explicit_jbeam_ext::{EXT_DEFORM, EXT_GROUPS, EXT_RAMP, EXT_REBOUND};
use glam::Vec3;
use metal::{Buffer, ComputePipelineState};

pub const LANES: usize = 1024;
pub const STEPS: usize = 34;
const NONE: u32 = u32::MAX;
const SENT: u32 = 0x7FC0_DEAD;

pub struct Layout {
    pub nn: usize,
    pub nr: usize,
    pub nt: usize,
    pub ng: usize,
    pub nd: usize,
    pub rs: usize,
    pub nadd: usize,
    pub ncen: usize,
    pub max_add: usize,
    pub max_cen: usize,
    /// Contacters (0: no self-collision) and their words a triangle's bitset.
    pub nc: usize,
    pub ncw: usize,
}

pub const GEO: usize = 13;
pub const HITW: usize = 4;
pub const PAIR_CAP: usize = 1 << 16;
pub const HIT_CAP: usize = 4096;
/// The pair list's box: the contact band, anti-clip's depth and a skin.
pub const LIST_SKIN: f32 = 0.05;
/// The wide list's skin.
pub const LIST_SKIN_OUT: f32 = 0.30;

/// Static data, the same for every car.
pub struct Statics {
    pub node_static: Vec<[f32; 4]>,
    pub node_range: Vec<[u32; 4]>,
    pub rec_a: Vec<[u32; 4]>,
    pub rec_k: Vec<[f32; 4]>,
    pub rec_b: Vec<[f32; 4]>,
    pub rec_e: Vec<[f32; 4]>,
    pub rec_ent: Vec<[u32; 4]>,
    pub rec_cen: Vec<[u32; 4]>,
    pub add_info: Vec<[u32; 2]>,
    pub perm: Vec<u32>,
    pub tri_breaks: Vec<[u32; 4]>,
    pub cands: Vec<u32>,
    pub tris: Vec<[u32; 4]>,
    pub allowed: Vec<u32>,
    pub list_reach: f32,
    /// The wide list's reach (the working list is filtered from it); the working reach or
    /// less: one list, built from the clusters every time.
    pub list_reach_out: f32,
    pub cr: f32,
    pub boxes: Vec<([f32; 3], [f32; 3])>,
    /// The second layout's table (lanes, node entry lists, vector indices) and defines;
    /// empty: the first layout.
    pub v2_tab: Vec<u32>,
    pub v2_defs: String,
    pub v2_off: V2Off,
    /// Bodies stepped in lockstep, their contact pass on their own workgroups (0: none;
    /// 2: pairs of workgroups).
    pub pair: usize,
    /// The fast variant's layout: each node's lane owns three of its plain beams (slots 0
    /// to 2), whose force on that node it adds from registers; the node's sums then start
    /// from its own beams (another order than the reference's).
    pub own: bool,
}

/// The second layout's one buffer a car: each part's offset in 4-byte words.
#[derive(Clone, Default, Debug)]
pub struct V2Off {
    pub touch: usize,
    pub pairs_out: usize,
    pub pbuild_out: usize,
    pub node: usize,
    pub rec: usize,
    pub groups: usize,
    pub deform: usize,
    pub tridead: usize,
    pub acc: usize,
    pub shift: usize,
    pub misc: usize,
    pub pairs: usize,
    pub pbuild: usize,
    pub clip: usize,
    pub hits: usize,
    pub scratch: usize,
    pub hot: usize,
    pub cand: usize,
    pub clu: usize,
    pub rig: usize,
    pub rig_words: usize,
    pub words: usize,
}

fn put_at(big: &mut [u8], word: usize, data: &[u8]) {
    big[word * 4..word * 4 + data.len()].copy_from_slice(data);
}
fn get_at<T: bytemuck::Pod>(big: &[u8], word: usize, n: usize) -> Vec<T> {
    let sz = std::mem::size_of::<T>();
    big[word * 4..word * 4 + n * sz].chunks_exact(sz).map(bytemuck::pod_read_unaligned).collect()
}
/// A car's state into its block (every scratch word, the pair rows and the hot image zero).
pub fn v2_put(big: &mut [u8], o: &V2Off, s: &State) {
    big.fill(0);
    put_at(big, o.node, bytemuck::cast_slice(&s.node));
    put_at(big, o.rec, bytemuck::cast_slice(&s.rec));
    put_at(big, o.groups, bytemuck::cast_slice(&s.groups_last));
    put_at(big, o.deform, bytemuck::cast_slice(&s.deform));
    put_at(big, o.tridead, bytemuck::cast_slice(&s.tri_dead));
    put_at(big, o.acc, bytemuck::cast_slice(&s.acc));
    put_at(big, o.shift, bytemuck::cast_slice(&s.shift));
    put_at(big, o.misc, bytemuck::cast_slice(&s.misc));
    put_at(big, o.clip, bytemuck::cast_slice(&s.clip));
    // (An uploaded state's pending sums are any node's: every node's bit set.)
    put_at(big, o.touch, &[0xFFu8; 512]);
    if o.rig_words != 0 && s.rig.len() == o.rig_words {
        put_at(big, o.rig, bytemuck::cast_slice(&s.rig));
    }
}
pub fn v2_get(big: &[u8], o: &V2Off, l: &Layout) -> State {
    State {
        node: get_at(big, o.node, l.nn * 3),
        rec: get_at(big, o.rec, l.nr * 3),
        add: Vec::new(),
        groups_last: get_at(big, o.groups, l.ng),
        deform: get_at(big, o.deform, l.nd),
        tri_dead: get_at(big, o.tridead, l.nt),
        acc: get_at(big, o.acc, l.nn * 3),
        shift: get_at(big, o.shift, l.nn * 3),
        misc: get_at(big, o.misc, 16),
        clip: get_at(big, o.clip, l.nn * 4),
        rig: get_at(big, o.rig, o.rig_words),
    }
}

/// One car's state in the kernel's layout.
#[derive(Clone)]
pub struct State {
    pub node: Vec<[f32; 4]>,
    pub rec: Vec<[f32; 4]>,
    pub add: Vec<[f32; 4]>,
    pub groups_last: Vec<u32>,
    pub deform: Vec<u32>,
    pub tri_dead: Vec<u32>,
    pub acc: Vec<i32>,
    pub shift: Vec<i32>,
    pub misc: Vec<u32>,
    pub clip: Vec<u32>,
    /// The rigid proxies' block (mechanism 3; empty without).
    pub rig: Vec<u32>,
}

pub const BDW: usize = 5;
pub const SHW: usize = 11;

/// A proxy set as the kernel's rigid block: body rows, counts, shape rows, sums.
pub fn rigid_state(r: &engine_softbody::explicit_rigid::ProxySet) -> anyhow::Result<Vec<u32>> {
    use engine_softbody::explicit_rigid::ProxyGeom;
    let (nb, nsh) = (r.bodies.len(), r.shapes.len());
    anyhow::ensure!(nb <= 64 && nsh <= 64 && !r.frozen, "the rigid block holds 64 bodies and 64 shapes");
    let mut w = vec![0u32; nb * BDW * 4 + 64 + nsh * SHW * 4 + nsh * 6];
    let mut put = |at: usize, v: [f32; 4]| {
        for k in 0..4 {
            w[at + k] = v[k].to_bits();
        }
    };
    for (i, b) in r.bodies.iter().enumerate() {
        anyhow::ensure!(b.inv_mass > 0.0, "a static proxy body (not ported)");
        let at = i * BDW * 4;
        put(at, [b.pos.x, b.pos.y, b.pos.z, b.inv_mass]);
        put(at + 4, [b.rot.x, b.rot.y, b.rot.z, b.rot.w]);
        put(at + 8, [b.vel.x, b.vel.y, b.vel.z, b.share]);
        put(at + 12, [b.ang_vel.x, b.ang_vel.y, b.ang_vel.z, 0.0]);
        put(at + 16, [b.inv_inertia.x, b.inv_inertia.y, b.inv_inertia.z, f32::from_bits(b.count)]);
    }
    let base = nb * BDW * 4 + 64;
    for (k, s) in r.shapes.iter().enumerate() {
        anyhow::ensure!(s.local_pos == Vec3::ZERO && s.local_rot == glam::Quat::IDENTITY, "a shape off its body's frame (not ported)");
        let at = base + k * SHW * 4;
        let (kind, radius, half) = match &s.geom {
            ProxyGeom::Box { half } => (0u32, 0.0, *half),
            ProxyGeom::Sphere { radius } => (1, *radius, Vec3::ZERO),
            g => anyhow::bail!("proxy geometry {g:?} not ported"),
        };
        put(at, [f32::from_bits(s.body), f32::from_bits(kind), radius, s.peak]);
        put(at + 4, [half.x, half.y, half.z, 0.0]);
        put(at + 8, [s.reference.x, s.reference.y, s.reference.z, 0.0]);
        put(at + 12, [s.impulse.x, s.impulse.y, s.impulse.z, 0.0]);
        put(at + 16, [s.moment.x, s.moment.y, s.moment.z, 0.0]);
    }
    Ok(w)
}

/// The kernel's rigid block against the reference's proxies: differing words and the first few.
pub fn compare_rigid(r: &engine_softbody::explicit_rigid::ProxySet, w: &[u32]) -> (usize, Vec<String>) {
    let (nb, mut bad, mut notes) = (r.bodies.len(), 0usize, Vec::new());
    let mut chk = |name: String, g: u32, c: f32| {
        if g != c.to_bits() {
            bad += 1;
            if notes.len() < 10 {
                notes.push(format!("{name} gpu {:e} cpu {c:e}", f32::from_bits(g)));
            }
        }
    };
    for (i, b) in r.bodies.iter().enumerate() {
        let at = i * BDW * 4;
        for (name, off, v) in [("pos", 0, [b.pos.x, b.pos.y, b.pos.z]), ("vel", 8, [b.vel.x, b.vel.y, b.vel.z]), ("ang", 12, [b.ang_vel.x, b.ang_vel.y, b.ang_vel.z])] {
            for k in 0..3 {
                chk(format!("body {i} {name}[{k}]"), w[at + off + k], v[k]);
            }
        }
        for (k, v) in [b.rot.x, b.rot.y, b.rot.z, b.rot.w].into_iter().enumerate() {
            chk(format!("body {i} rot[{k}]"), w[at + 4 + k], v);
        }
        chk(format!("body {i} count"), w[at + 19], f32::from_bits(b.count));
    }
    let base = nb * BDW * 4 + 64;
    for (k, s) in r.shapes.iter().enumerate() {
        let at = base + k * SHW * 4;
        for (name, off, v) in [("impulse", 12, s.impulse), ("moment", 16, s.moment), ("reference", 8, s.reference)] {
            for q in 0..3 {
                chk(format!("shape {k} {name}[{q}]"), w[at + off + q], v[q]);
            }
        }
        chk(format!("shape {k} peak"), w[at + 3], s.peak);
    }
    (bad, notes)
}

/// The GPU's car against the reference's by size, for a build whose sums run in another
/// order: (farthest node m, largest velocity difference m/s, largest force difference
/// over the largest force, beams and triangles whose broken state differs).
pub fn compare_tol(cpu: &ExplicitBeams, g: &State) -> (f64, f64, f64, usize) {
    let (mut dp, mut dv, mut df, mut fmax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let nan = |x: f64| if x.is_nan() { f64::INFINITY } else { x };
    for (i, n) in cpu.nodes.iter().enumerate() {
        let v = |k: usize| Vec3::new(g.node[i * 3 + k][0], g.node[i * 3 + k][1], g.node[i * 3 + k][2]);
        dp = dp.max(nan((v(0) - n.pos).length() as f64));
        dv = dv.max(nan((v(1) - n.vel).length() as f64));
        df = df.max(nan((v(2) - n.force).length() as f64));
        fmax = fmax.max(n.force.length() as f64);
    }
    let mut discrete = 0;
    for (j, b) in cpu.beams.iter().enumerate() {
        if (g.rec[j * 3 + 1][3] != 0.0) != b.broken {
            discrete += 1;
        }
    }
    (dp, dv, df / fmax.max(1.0), discrete)
}

fn fb(x: u32) -> f32 {
    f32::from_bits(x)
}

/// The order a beam record adds forces to nodes in the CPU's beam pass.
fn adds_of(s: &ExplicitBeams, j: usize) -> Vec<u32> {
    let b = &s.beams[j];
    match b.kind {
        ExplicitKind::JbeamTorsionPair | ExplicitKind::JbeamAeroTri2 => vec![],
        ExplicitKind::JbeamTorsion => vec![b.a, b.b, s.beams[j + 1].a, s.beams[j + 1].b],
        ExplicitKind::JbeamAeroTri => {
            let x = s.ext_of[j] as usize - 1;
            vec![b.a, b.b, s.beam_ext[x].index(0)]
        }
        _ => vec![b.a, b.b],
    }
}

pub fn layout_and_statics(s: &ExplicitBeams) -> anyhow::Result<(Layout, Statics)> {
    let nn = s.nodes.len();
    let nr = s.beams.len();
    let nt = s.triangles.len();
    anyhow::ensure!(nn <= LANES, "more nodes than lanes");
    anyhow::ensure!(s.damp_filters.is_empty() && s.shocks2.is_empty() && s.triggers.is_empty());
    anyhow::ensure!(s.settings.centred_damping && s.settings.centred_objective && !s.settings.fast_inv_sqrt);
    anyhow::ensure!(!s.settings.drag && !s.plastic_strength_drop && s.fixed_contact && s.dissipative_contact && s.exact_friction);
    anyhow::ensure!(s.damp_scheme == 0 && s.shapes.is_empty() && !s.surface_contact);
    anyhow::ensure!(s.contact_force.is_none() && s.wings.is_empty() && s.fusedrag.is_none() && s.slides.is_empty());
    let ng = s.beams.iter().map(|b| b.group.unsigned_abs() as usize).max().unwrap_or(0).max(
        s.torsion_watch.iter().flatten().copied().max().unwrap_or(0) as usize,
    ) + 1;
    let nd = s.deform_state.len().max(1);
    // Add entries, node-major in the CPU's order.
    let mut adds: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nn];
    for j in 0..nr {
        for (q, &n) in adds_of(s, j).iter().enumerate() {
            adds[n as usize].push((j, q));
        }
    }
    let mut rec_ent = vec![[NONE; 4]; nr];
    let mut add_info = Vec::new();
    let mut node_range = vec![[0u32; 4]; nn];
    for i in 0..nn {
        node_range[i][0] = add_info.len() as u32;
        node_range[i][1] = adds[i].len() as u32;
        for &(j, q) in &adds[i] {
            rec_ent[j][q] = add_info.len() as u32;
            add_info.push([j as u32, s.beams[j].group as u32]);
        }
    }
    let nadd = add_info.len();
    // Centred entries: each node's incidence list, the end as the CPU assigns it.
    let mut rec_cen_end = vec![[NONE; 2]; nr];
    let mut ncen = 0usize;
    for i in 0..nn {
        node_range[i][2] = ncen as u32;
        let mut seen: Option<u32> = None;
        for &j in &s.incidence[i] {
            let b = &s.beams[j as usize];
            let other = b.b == i as u32 && b.a != i as u32;
            let end = if other || (b.a == b.b && seen == Some(j)) { 1 } else { 0 };
            if b.a == b.b {
                seen = Some(j);
            }
            anyhow::ensure!(rec_cen_end[j as usize][end] == NONE, "record {j} end {end} twice");
            rec_cen_end[j as usize][end] = ncen as u32;
            ncen += 1;
        }
        node_range[i][3] = ncen as u32 - node_range[i][2];
    }
    let mut rec_cen = vec![[NONE; 4]; nr];
    for j in 0..nr {
        rec_cen[j][0] = rec_cen_end[j][0];
        rec_cen[j][1] = rec_cen_end[j][1];
        if s.beams[j].kind == ExplicitKind::JbeamTorsion {
            rec_cen[j][2] = rec_cen_end[j + 1][0];
            rec_cen[j][3] = rec_cen_end[j + 1][1];
        }
    }
    let max_add = adds.iter().map(|v| v.len()).max().unwrap_or(0);
    let max_cen = (0..nn).map(|i| node_range[i][3] as usize).max().unwrap_or(0);
    // Records.
    let mut rec_a = Vec::with_capacity(nr);
    let mut rec_k = Vec::with_capacity(nr);
    let mut rec_b = Vec::with_capacity(nr);
    let mut rec_e = Vec::with_capacity(nr * 2);
    for (j, b) in s.beams.iter().enumerate() {
        let x = if s.ext_of.is_empty() { 0 } else { s.ext_of[j] };
        let ext = x != 0;
        rec_a.push([b.a, b.b, b.kind.code() | b.ty.code() << 8 | (ext as u32) << 16, b.group as u32]);
        rec_k.push([b.k, b.c, b.limit_k, b.limit_c]);
        let mut w = 0.0f32;
        let mut e = [[0.0f32; 4]; 2];
        if ext {
            let r = &s.beam_ext[x as usize - 1];
            anyhow::ensure!(r.mask & (EXT_GROUPS | EXT_REBOUND | EXT_RAMP) == 0 && r.ramp_left == 0, "record {j}: extension features not ported (mask {})", r.mask);
            anyhow::ensure!(r.deform[0] < 256 && r.deform[1] < 256);
            w = fb(r.mask | r.deform[0] << 8 | r.deform[1] << 16);
            e[0] = r.e1;
            e[1] = [r.e2[0], r.e2[1], r.ramp_delta * r.ramp_left as f32, r.trigger];
            anyhow::ensure!(matches!(b.kind, ExplicitKind::JbeamPlain | ExplicitKind::JbeamSupport | ExplicitKind::JbeamAniso | ExplicitKind::JbeamAeroTri | ExplicitKind::JbeamAeroTri2), "record {j}: ext kind {:?}", b.kind);
            let _ = EXT_DEFORM;
        }
        match b.kind {
            ExplicitKind::JbeamTorsion => {
                let wch = s.torsion_watch[b.shock as usize - 1];
                e[0] = [fb(wch[0]), fb(wch[1]), fb(wch[2]), 0.0];
            }
            ExplicitKind::Plain => {
                anyhow::ensure!(b.ty.code() != 0 && b.strength.is_infinite(), "record {j}: a breakable or deformable plain beam");
            }
            ExplicitKind::JbeamPlain | ExplicitKind::JbeamSupport | ExplicitKind::JbeamBounded | ExplicitKind::JbeamAniso
            | ExplicitKind::JbeamCoupler | ExplicitKind::JbeamTorsionPair | ExplicitKind::JbeamAeroTri | ExplicitKind::JbeamAeroTri2 => {}
            k => anyhow::bail!("record {j}: kind {k:?} not ported"),
        }
        if b.kind == ExplicitKind::JbeamCoupler {
            anyhow::ensure!(!ext);
        }
        rec_b.push([b.short_bound, b.long_bound, b.plastic, w]);
        rec_e.push(e[0]);
        rec_e.push(e[1]);
    }
    // Lanes: records sorted by kind (a SIMD group runs one kind's branch).
    let mut active: Vec<usize> = (0..nr)
        .filter(|&j| !matches!(s.beams[j].kind, ExplicitKind::JbeamTorsionPair | ExplicitKind::JbeamAeroTri2))
        .collect();
    active.sort_by_key(|&j| (s.beams[j].kind.code(), j));
    let rs = active.len().div_ceil(LANES);
    let mut perm = vec![NONE; rs * LANES];
    for (i, &j) in active.iter().enumerate() {
        perm[i] = j as u32;
    }
    let tri_breaks = (0..nt).map(|t| s.tri_breaks.get(t).copied().unwrap_or([NONE; 4])).collect();
    let node_static = s.nodes.iter().map(|n| [n.mass, n.friction, fb(n.flags), 0.0]).collect();
    // Self-collision: the contacters, the triangles, the pairs no rule excludes.
    let nc = s.candidates.len();
    let ncw = nc.div_ceil(32).max(1);
    anyhow::ensure!(!s.anti_clip || (s.anti_clip && s.two_body_contact && !s.own_travel_reach && !s.settings.triangle_throttle), "self-collision rules not ported");
    anyhow::ensure!(nc == 0 || s.anti_clip, "only anti-clip self-collision is ported");
    anyhow::ensure!(s.anti_only_tris.iter().all(|&x| !x) && s.retired.is_empty() && nn < 65536 && nt < 65536);
    let tris: Vec<[u32; 4]> = s.triangles.iter().map(|t| [t.nodes[0], t.nodes[1], t.nodes[2], 0]).collect();
    let mut allowed = vec![0u32; nt * ncw];
    for t in 0..nt {
        let [i0, i1, i2] = s.triangles[t].nodes;
        let tg = s.triangle_self_group(t);
        for (c, &h) in s.candidates.iter().enumerate() {
            if h == i0 || h == i1 || h == i2 {
                continue;
            }
            if tg != 0 && s.self_groups[h as usize] == tg {
                continue;
            }
            if !s.tri_excl.is_empty() && s.tri_excl[t].contains(&h) {
                continue;
            }
            allowed[t * ncw + c / 32] |= 1 << (c % 32);
        }
    }
    let cr = s.collision_range;
    let list_reach = cr + engine_softbody::explicit_contact::CLIP_DEPTH + LIST_SKIN;
    Ok((
        Layout { nn, nr, nt, ng, nd, rs, nadd, ncen, max_add, max_cen, nc, ncw },
        Statics { node_static, node_range, rec_a, rec_k, rec_b, rec_e, rec_ent, rec_cen, add_info, perm, tri_breaks, cands: s.candidates.clone(), tris, allowed, list_reach, list_reach_out: list_reach + LIST_SKIN_OUT - LIST_SKIN, cr, boxes: s.boxes.iter().map(|b| (b.lo.to_array(), b.hi.to_array())).collect(), v2_tab: Vec::new(), v2_defs: String::new(), v2_off: V2Off::default(), pair: 0, own: false },
    ))
}

/// The car's state now, as the kernel holds it (the aux words: what the last beam pass
/// computed for the centred correction, recomputed from the same positions).
pub fn state_of(s: &ExplicitBeams, l: &Layout, st: &Statics) -> State {
    let mut node = Vec::with_capacity(l.nn * 3);
    for n in &s.nodes {
        node.push([n.pos.x, n.pos.y, n.pos.z, 0.0]);
        node.push([n.vel.x, n.vel.y, n.vel.z, 0.0]);
        node.push([n.force.x, n.force.y, n.force.z, 0.0]);
    }
    let mut rec = Vec::with_capacity(l.nr * 3);
    for (j, b) in s.beams.iter().enumerate() {
        let slope = s.damp_slopes.get(j).copied().unwrap_or(0.0);
        rec.push([b.rest, b.tc, b.tt, b.gate]);
        rec.push([b.strength, b.force, slope, if b.broken { 1.0 } else { 0.0 }]);
        let mut aux = [0.0f32; 4];
        let p = |i: u32| s.nodes[i as usize].pos;
        let v = |i: u32| s.nodes[i as usize].vel;
        match b.kind {
            ExplicitKind::JbeamTorsion => {
                let ids = [b.a, b.b, s.beams[j + 1].a, s.beams[j + 1].b];
                let g = dihedral_jacobian(ids.map(p));
                for q in 0..4 {
                    aux[q] = g[q].dot(v(ids[q]));
                }
            }
            _ => {
                let d = p(b.a) - p(b.b);
                let inv = exact_length(d.dot(d)).1;
                aux[0] = (v(b.a) - v(b.b)).dot(d) * inv;
            }
        }
        rec.push(aux);
    }
    let mut add = vec![[0.0f32; 4]; l.nadd];
    for (k, info) in st.add_info.iter().enumerate() {
        if s.beams[info[0] as usize].broken {
            add[k] = [fb(SENT); 4];
        }
    }
    let mut groups_last = vec![0u32; l.ng];
    for &g in &s.groups_now {
        groups_last[g as usize] = 1;
    }
    let mut deform = vec![0u32; l.nd];
    for (i, &d) in s.deform_state.iter().enumerate() {
        deform[i] = d;
    }
    let tri_dead = (0..l.nt).map(|t| s.tri_dead.get(t).copied().unwrap_or(false) as u32).collect();
    let mut acc = vec![0i32; l.nn * 3];
    if s.contact_acc.len() == l.nn {
        for (i, a) in s.contact_acc.iter().enumerate() {
            acc[i * 3..i * 3 + 3].copy_from_slice(a);
        }
    }
    let mut shift = vec![0i32; l.nn * 3];
    let pending = !s.contact_shift.is_empty();
    if pending {
        for (i, a) in s.contact_shift.iter().enumerate() {
            shift[i * 3..i * 3 + 3].copy_from_slice(a);
        }
    }
    let mut misc = vec![0u32; 16];
    misc[0] = pending as u32;
    misc[3] = 1; // the pair list is built at the first step
    // (The second layout's event flags, set so that its first step looks: contact sums and
    // stage words, side memory, the groups the last beam pass set off, triangle breaks.)
    misc[4] = 1;
    misc[7] = 1;
    misc[8] = 1;
    misc[9] = if s.groups_now.is_empty() { 0 } else { 2 };
    misc[10] = 1;
    let mut clip = vec![engine_softbody::explicit_contact::CLIP_EMPTY; l.nn * 4];
    if s.clip_sides.len() == l.nn {
        for (i, c) in s.clip_sides.iter().enumerate() {
            clip[i * 4..i * 4 + 4].copy_from_slice(c);
        }
    }
    let rig = match &s.rigid {
        Some(r) => rigid_state(r).expect("the rigid block"),
        None => Vec::new(),
    };
    State { node, rec, add, groups_last, deform, tri_dead, acc, shift, misc, clip, rig }
}

pub fn msl(l: &Layout, st: &Statics, extra: &str, per_frame: usize) -> String {
    let f3 = |v: [f32; 3]| format!("float3(as_type<float>({}u), as_type<float>({}u), as_type<float>({}u))", v[0].to_bits(), v[1].to_bits(), v[2].to_bits());
    let boxes = if st.boxes.is_empty() {
        "#define NBOX 0\n".to_string()
    } else {
        format!(
            "#define NBOX {}\nconstant float3 BOX_LO[{}] = {{{}}};\nconstant float3 BOX_HI[{}] = {{{}}};\n",
            st.boxes.len(), st.boxes.len(), st.boxes.iter().map(|b| f3(b.0)).collect::<Vec<_>>().join(", "),
            st.boxes.len(), st.boxes.iter().map(|b| f3(b.1)).collect::<Vec<_>>().join(", ")
        )
    };
    let extra = format!("{extra}\n{boxes}");
    if !st.v2_tab.is_empty() {
        let v1 = include_str!("../kernels/floor.metal");
        let helpers = &v1[..v1.find("// ---- buffers ---").expect("the marker")];
        return format!(
            "#define NN {}\n#define NR {}\n#define NT {}\n#define NG {}\n#define ND {}\n#define LANES {}\n#define STEPS {}\n#define COLLIDE {}\n#define NC {}\n#define NCW {}\n#define HITW {HITW}\n#define HIT_CAP {HIT_CAP}\n#define CR as_type<float>({}u)\n#define LIST_REACH as_type<float>({}u)\n#define LIST_EPS 0.001f\n#ifndef RUNTIME_SIZES\n#define RUNTIME_SIZES 0\n#endif\n{}\n{}\n{}\n{}",
            l.nn, l.nr, l.nt, l.ng, l.nd, LANES, STEPS / per_frame, (l.nc > 0) as u32, l.nc.max(1), l.ncw, st.cr.to_bits(), st.list_reach.to_bits(),
            extra, st.v2_defs, helpers,
            unroll_slots(&if st.pair != 0 {
                format!("#define PAIR 1\n#define NCARS {}\n#define NGROUPS {}\n#define XCAP 1048576u\n{}", st.pair, st.pair, include_str!("../kernels/floor2.metal").replace("//@PAIR_PASS", include_str!("../kernels/pair.metal")))
            } else {
                include_str!("../kernels/floor2.metal").to_string()
            })
        );
    }
    format!(
        "#define NN {}\n#define NR {}\n#define NT {}\n#define NG {}\n#define ND {}\n#define RS {}\n#define LANES {}\n#define STEPS {}\n#define NADD {}\n#define NCEN {}\n#define NODE_MAX_ADD {}\n#define NODE_MAX_CEN {}\n#define COLLIDE {}\n#define NC {}\n#define NCW {}\n#define GEO {GEO}\n#define HITW {HITW}\n#define PAIR_CAP {PAIR_CAP}\n#define HIT_CAP {HIT_CAP}\n#define CR as_type<float>({}u)\n#define LIST_REACH as_type<float>({}u)\n#define LIST_EPS 0.001f\n{}\n{}",
        l.nn, l.nr, l.nt, l.ng, l.nd, l.rs, LANES, STEPS / per_frame, l.nadd, l.ncen, l.max_add, l.max_cen,
        (l.nc > 0) as u32, l.nc.max(1), l.ncw, st.cr.to_bits(), st.list_reach.to_bits(), extra,
        include_str!("../kernels/floor.metal")
    )
}

/// The GPU side: N cars' buffers.
pub struct Floor {
    pub gpu: Gpu,
    /// Dispatches a frame (1: all substeps in one; STEPS: one a substep).
    pub per_frame: usize,
    v2: bool,
    off: V2Off,
    /// The second layout: the beams' two deferred words written to their records each
    /// dispatch (to read the beams; off for timing).
    pub flush: bool,
    /// With a kernel of one substep a dispatch (`PER_STEP`): a run is one dispatch.
    pub single: bool,
    pair: usize,
    pub l: Layout,
    pub cars: usize,
    pso: ComputePipelineState,
    params: Buffer,
    stat: Vec<Buffer>,
    node: Buffer,
    rec: Buffer,
    add: Buffer,
    cen: Buffer,
    groups: Buffer,
    deform: Buffer,
    tri_dead: Buffer,
    acc: Buffer,
    shift: Buffer,
    misc: Buffer,
    tri_geo: Buffer,
    pairs: Buffer,
    pbuild: Buffer,
    clip: Buffer,
    hits: Buffer,
    stage: Buffer,
}

impl Floor {
    pub fn new(gpu: Gpu, l: Layout, st: &Statics, cars: usize, fast: bool, extra: &str) -> anyhow::Result<Self> {
        let per_frame = if extra.contains("PER_STEP") { STEPS } else { 1 };
        // Strict: fast math off still lets Metal fuse a·b + c (llvm.fmuladd); the pragma
        // turns contraction off, which leaves IEEE single operations in source order.
        let pragma = if fast { "" } else { "#pragma clang fp contract(off)\n" };
        let pso = gpu.pipeline(&format!("{pragma}{}", msl(&l, st, extra, per_frame)), "floorFrame", fast)?;
        let stat = vec![
            gpu.buffer(&st.node_static),
            gpu.buffer(&st.node_range),
            gpu.buffer(&st.rec_a),
            gpu.buffer(&st.rec_k),
            gpu.buffer(&st.rec_b),
            gpu.buffer(&st.rec_e),
            gpu.buffer(&st.rec_ent),
            gpu.buffer(&st.rec_cen),
            gpu.buffer(&st.add_info),
            gpu.buffer(&st.perm),
            gpu.buffer(&st.tri_breaks),
            gpu.buffer(&st.cands),
            gpu.buffer(&st.tris),
            gpu.buffer(&st.allowed),
        ];
        let params = gpu.zeros(64);
        let z = |bytes: usize| gpu.zeros(bytes * cars);
        Ok(Self {
            node: if st.v2_tab.is_empty() { z(l.nn * 3 * 16) } else { z(st.v2_off.words * 4) },
            off: st.v2_off.clone(),
            flush: true,
            single: false,
            pair: st.pair,
            rec: z(l.nr * 3 * 16),
            add: z(l.nadd * 16),
            cen: if st.v2_tab.is_empty() { z(l.ncen * 16) } else { gpu.buffer(&st.v2_tab) },
            v2: !st.v2_tab.is_empty(),
            groups: z(l.ng * 4),
            deform: z(l.nd * 4),
            tri_dead: z(l.nt * 4),
            acc: z(l.nn * 3 * 4),
            shift: z(l.nn * 3 * 4),
            misc: z(16 * 4),
            tri_geo: z(l.nt * GEO * 16),
            pairs: z(PAIR_CAP * 4),
            pbuild: z(l.nn * 16),
            clip: z(l.nn * 16),
            hits: z(HIT_CAP * HITW * 16),
            stage: z(l.nn * 48),
            gpu,
            per_frame,
            l,
            cars,
            pso,
            params,
            stat,
        })
    }

    pub fn threads_ok(&self) -> (u64, u64, u64) {
        (self.pso.max_total_threads_per_threadgroup(), self.pso.thread_execution_width(), self.pso.static_threadgroup_memory_length())
    }

    pub fn upload(&self, car: usize, s: &State) {
        let l = &self.l;
        if self.v2 {
            let n = self.off.words * 4;
            let big = unsafe { std::slice::from_raw_parts_mut((self.node.contents() as *mut u8).add(car * n), n) };
            v2_put(big, &self.off, s);
            return;
        }
        let put = |b: &Buffer, off: usize, data: &[u8]| unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), (b.contents() as *mut u8).add(off), data.len())
        };
        put(&self.node, car * l.nn * 48, bytemuck::cast_slice(&s.node));
        put(&self.rec, car * l.nr * 48, bytemuck::cast_slice(&s.rec));
        put(&self.add, car * l.nadd * 16, bytemuck::cast_slice(&s.add));
        if !self.v2 {
            put(&self.cen, car * l.ncen * 16, &vec![0u8; l.ncen * 16]);
        }
        put(&self.groups, car * l.ng * 4, bytemuck::cast_slice(&s.groups_last));
        put(&self.deform, car * l.nd * 4, bytemuck::cast_slice(&s.deform));
        put(&self.tri_dead, car * l.nt * 4, bytemuck::cast_slice(&s.tri_dead));
        put(&self.acc, car * l.nn * 12, bytemuck::cast_slice(&s.acc));
        put(&self.shift, car * l.nn * 12, bytemuck::cast_slice(&s.shift));
        put(&self.misc, car * 64, bytemuck::cast_slice(&s.misc));
        put(&self.clip, car * l.nn * 16, bytemuck::cast_slice(&s.clip));
        put(&self.stage, car * l.nn * 48, &vec![0u8; l.nn * 48]);
    }

    pub fn download(&self, car: usize) -> State {
        let l = &self.l;
        if self.v2 {
            let n = self.off.words * 4;
            let big = unsafe { std::slice::from_raw_parts((self.node.contents() as *const u8).add(car * n), n) };
            return v2_get(big, &self.off, l);
        }
        fn get<T: bytemuck::Pod + Default + Clone>(b: &Buffer, off: usize, n: usize) -> Vec<T> {
            let all: Vec<T> = read(b, off + n);
            all[off..].to_vec()
        }
        State {
            node: get(&self.node, car * l.nn * 3, l.nn * 3),
            rec: get(&self.rec, car * l.nr * 3, l.nr * 3),
            add: get(&self.add, car * l.nadd, l.nadd),
            groups_last: get(&self.groups, car * l.ng, l.ng),
            deform: get(&self.deform, car * l.nd, l.nd),
            tri_dead: get(&self.tri_dead, car * l.nt, l.nt),
            acc: get(&self.acc, car * l.nn * 3, l.nn * 3),
            shift: get(&self.shift, car * l.nn * 3, l.nn * 3),
            misc: get(&self.misc, car * 16, 16),
            clip: get(&self.clip, car * l.nn * 4, l.nn * 4),
            rig: Vec::new(),
        }
    }

    /// Every per-car buffer with its size a car.
    fn per_car(&self) -> Vec<(&Buffer, usize)> {
        let l = &self.l;
        if self.v2 {
            return vec![(&self.node, self.off.words * 4)];
        }
        vec![
            (&self.node, l.nn * 48), (&self.rec, l.nr * 48), (&self.add, l.nadd * 16), (&self.cen, if self.v2 { 0 } else { l.ncen * 16 }), (&self.groups, l.ng * 4),
            (&self.deform, l.nd * 4), (&self.tri_dead, l.nt * 4), (&self.acc, l.nn * 12), (&self.shift, l.nn * 12), (&self.misc, 64),
            (&self.tri_geo, l.nt * GEO * 16), (&self.pairs, PAIR_CAP * 4), (&self.pbuild, l.nn * 16), (&self.clip, l.nn * 16),
            (&self.hits, HIT_CAP * HITW * 16), (&self.stage, l.nn * 48),
        ]
    }

    /// All of car `car`'s buffers (its state with the pair list and every scratch word).
    pub fn snapshot(&self, car: usize) -> Vec<Vec<u8>> {
        self.per_car().iter().map(|(b, n)| unsafe { std::slice::from_raw_parts((b.contents() as *const u8).add(car * n), *n).to_vec() }).collect()
    }

    pub fn restore(&self, car: usize, snap: &[Vec<u8>]) {
        for ((b, n), s) in self.per_car().iter().zip(snap) {
            unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), (b.contents() as *mut u8).add(car * n), *n) };
        }
    }

    /// `frames` frames of every car in one dispatch; the GPU ms.
    pub fn run(&self, frames: u32) -> f64 {
        self.run_skip(frames, 0)
    }

    /// Timing only: `skip` leaves phases out (bit k: phase k of the kernel).
    pub fn run_skip(&self, frames: u32, skip: u32) -> f64 {
        let l = &self.l;
        let steps = STEPS / self.per_frame;
        if self.pair != 0 {
            // (Each pair's cross-group barrier counter starts at zero each dispatch.)
            for g in (0..self.cars).step_by(2) {
                unsafe { *((self.node.contents() as *mut u32).add(g * self.off.words + self.off.misc + 14)) = 0 };
            }
        }
        write(&self.params, &[self.cars as u32, if self.per_frame > 1 { 1 } else { frames }, 0.0f32.to_bits(), skip, LANES as u32, l.nn as u32, l.nt as u32, l.nc.max(1) as u32, steps as u32, l.ng as u32, l.rs as u32, self.flush as u32]);
        let mut bufs: Vec<&Buffer> = vec![&self.params, &self.node, &self.stat[0], &self.stat[1], &self.stat[2], &self.stat[3], &self.stat[4], &self.stat[5]];
        bufs.extend([&self.rec, &self.stat[6], &self.stat[7], &self.add, &self.cen, &self.stat[8], &self.stat[9], &self.groups, &self.deform, &self.stat[10], &self.tri_dead, &self.acc, &self.shift, &self.misc]);
        bufs.extend([&self.stat[11], &self.stat[12], &self.stat[13], &self.tri_geo, &self.pairs, &self.pbuild, &self.clip, &self.hits, &self.stage]);
        let n = if self.single { 1 } else if self.per_frame > 1 { self.per_frame * frames as usize } else { 1 };
        self.gpu.run_n(&self.pso, &bufs, self.cars as u64, LANES as u64, n)
    }
}

/// Where a frame's time goes, without the confounds of leaving one phase out of a whole
/// frame (the later substeps then run from another state). The kernel is one substep a
/// dispatch (`PER_STEP`, `single`); after `frame` frames, each of the next 34 substeps is
/// run from its own saved state with its phases enabled one group more at a time (a
/// substep cut short after group k: nothing after it runs, nothing before it changes), the
/// median of `reps` dispatches each. A group's cost is the difference; the sum over the
/// 34 substeps is its ms a frame. `groups` are (name, the skip bits that group clears), in
/// the kernel's order.
pub fn profile(f: &Floor, blocks: usize, frame: usize, reps: usize, groups: &[(&str, u32)]) {
    let all: u32 = groups.iter().fold(0, |m, g| m | g.1);
    for _ in 0..frame * STEPS {
        f.run(1);
    }
    let mut t = vec![0.0f64; groups.len() + 1];
    for _ in 0..STEPS {
        let snap: Vec<Vec<Vec<u8>>> = (0..blocks).map(|g| f.snapshot(g)).collect();
        let back = || {
            for g in 0..blocks {
                f.restore(g, &snap[g]);
            }
        };
        let mut mask = all;
        for k in 0..=groups.len() {
            if k > 0 {
                mask &= !groups[k - 1].1;
            }
            let mut ms: Vec<f64> = (0..reps)
                .map(|_| {
                    back();
                    f.run_skip(1, mask)
                })
                .collect();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[k] += ms[ms.len() / 2];
        }
        back();
        f.run(1);
    }
    println!("profile, frame {} (ms a frame; each substep cut short after one more group, {reps} dispatches each):", frame + 1);
    println!("  {:30} {:.4}  (34 dispatches of an empty substep: the state's load and write-back)", "nothing", t[0]);
    for (k, g) in groups.iter().enumerate() {
        println!("  {:30} {:.4}", g.0, t[k + 1] - t[k]);
    }
    println!("  {:30} {:.4}", "all groups", t[groups.len()] - t[0]);
}

/// The one-body substep's groups for [`profile`], in the kernel's order.
pub const GROUPS: [(&str, u32); 9] = [
    ("S0 node pass", 1 << 0),
    ("S1 centred scalars", 1 << 1),
    ("C1+C2 centred gather", 1 << 2),
    ("S3 beams, list check", 1 << 3 | 1 << 6),
    ("pair list rebuilds", 1 << 14),
    ("S4 force gather", 1 << 4),
    ("S4 pair rows", 1 << 5),
    ("self candidates' rule", 1 << 15),
    ("self hit passes", 1 << 8),
];

/// Every state word of the GPU's car against the CPU reference's: (differing words, a
/// description of the first few).
pub fn compare(cpu: &ExplicitBeams, g: &State) -> (usize, Vec<String>) {
    let mut bad = 0;
    let mut first = Vec::new();
    let mut note = |what: String| {
        if first.len() < 12 {
            first.push(what);
        }
    };
    let words = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
    for (i, n) in cpu.nodes.iter().enumerate() {
        let n: &Node = n;
        for (k, (name, c)) in [("pos", n.pos), ("vel", n.vel), ("force", n.force)].into_iter().enumerate() {
            let gw = g.node[i * 3 + k];
            let cw = words(c);
            for a in 0..3 {
                if gw[a].to_bits() != cw[a] {
                    bad += 1;
                    note(format!("node {i} {name}[{a}] gpu {:e} cpu {:e}", gw[a], f32::from_bits(cw[a])));
                }
            }
        }
    }
    for (j, b) in cpu.beams.iter().enumerate() {
        let s0 = g.rec[j * 3];
        let s1 = g.rec[j * 3 + 1];
        let gb = s1[3] != 0.0;
        if gb != b.broken {
            bad += 1;
            note(format!("beam {j} ({:?}) broken gpu {gb} cpu {}", b.kind, b.broken));
            continue;
        }
        if b.broken || matches!(b.kind, ExplicitKind::JbeamTorsionPair | ExplicitKind::JbeamAeroTri2) {
            continue;
        }
        let slope = cpu.damp_slopes.get(j).copied().unwrap_or(0.0);
        let v2 = g.misc[15] == 2;
        for (name, gv, cv) in [("rest", s0[0], b.rest), ("tc", s0[1], b.tc), ("tt", s0[2], b.tt), ("gate", s0[3], b.gate), ("strength", s1[0], b.strength), ("force", s1[1], b.force), ("slope", s1[2], slope)] {
            // (The second layout keeps no last force: an output no rule reads.)
            if v2 && name == "force" {
                continue;
            }
            if gv.to_bits() != cv.to_bits() {
                bad += 1;
                note(format!("beam {j} ({:?}) {name} gpu {gv:e} cpu {cv:e}", b.kind));
            }
        }
    }
    for (i, &d) in cpu.deform_state.iter().enumerate() {
        if g.deform[i] != d {
            bad += 1;
            note(format!("deform {i}"));
        }
    }
    for (t, &d) in cpu.tri_dead.iter().enumerate() {
        if (g.tri_dead[t] != 0) != d {
            bad += 1;
            note(format!("triangle {t} dead gpu {} cpu {d}", g.tri_dead[t]));
        }
    }
    for gi in 1..g.groups_last.len() {
        let c = cpu.groups_now.contains(&(gi as u32));
        if (g.groups_last[gi] != 0) != c {
            bad += 1;
            note(format!("group {gi} fired gpu {} cpu {c}", g.groups_last[gi]));
        }
    }
    let acc_cpu = |i: usize| if cpu.contact_acc.len() == cpu.nodes.len() { cpu.contact_acc[i] } else { [0; 3] };
    for i in 0..cpu.nodes.len() {
        let a = acc_cpu(i);
        for k in 0..3 {
            if g.acc[i * 3 + k] != a[k] {
                bad += 1;
                note(format!("acc node {i}[{k}] gpu {} cpu {}", g.acc[i * 3 + k], a[k]));
            }
        }
    }
    let empty = engine_softbody::explicit_contact::CLIP_EMPTY;
    for i in 0..cpu.nodes.len() {
        let c = cpu.clip_sides.get(i).copied().unwrap_or([empty; 4]);
        for k in 0..4 {
            if g.clip[i * 4 + k] != c[k] {
                bad += 1;
                note(format!("clip node {i}[{k}] gpu {:#x} cpu {:#x}", g.clip[i * 4 + k], c[k]));
            }
        }
    }
    let pending = !cpu.contact_shift.is_empty();
    if (g.misc[0] != 0) != pending {
        bad += 1;
        note(format!("shifts pending gpu {} cpu {pending}", g.misc[0]));
    }
    if pending {
        for (i, a) in cpu.contact_shift.iter().enumerate() {
            for k in 0..3 {
                if g.shift[i * 3 + k] != a[k] {
                    bad += 1;
                    note(format!("shift node {i}[{k}] gpu {} cpu {}", g.shift[i * 3 + k], a[k]));
                }
            }
        }
    }
    if g.misc[1] != 0 {
        bad += 1;
        note(format!("kernel status {:#x}", g.misc[1]));
    }
    (bad, first)
}

/// The kernel on the CPU (emu/): the same source, each lane a coroutine. One car.
pub struct Emu {
    pub l: Layout,
    exe: std::path::PathBuf,
    dir: std::path::PathBuf,
    /// The 31 buffers in binding order.
    pub bufs: Vec<Vec<u8>>,
    pub reverse: bool,
    v2: bool,
    off: V2Off,
    /// Workgroups (bodies, then a pair's contact group): each a block of the one buffer.
    pub groups: usize,
    /// Substeps a run (RUNTIME_SIZES kernels only; else STEPS).
    pub steps: usize,
}

impl Emu {
    pub fn new(l: Layout, st: &Statics, extra: &str, dir: &std::path::Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let src = msl(&l, st, extra, 1).replace("kernel void floorFrame(", "Lane floorFrame(");
        std::fs::write(dir.join("kernel.inc"), src)?;
        let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("emu");
        let exe = dir.join("emu");
        let out = std::process::Command::new("xcrun")
            .args(["clang++", "-std=c++20", "-O1", "-ffp-contract=off", "-w", if std::env::var("EMU_TRACE").is_ok() { "-DEMU_TRACE" } else { "-DEMU_QUIET" }, if st.v2_tab.is_empty() { "-DV1" } else { "-DV2" }, "-I"])
            .arg(&here)
            .arg("-I")
            .arg(here.join("metal_stdlib_dir"))
            .arg("-I")
            .arg(dir)
            .arg(here.join("main.cpp"))
            .arg("-o")
            .arg(&exe)
            .output()?;
        anyhow::ensure!(out.status.success(), "emu compile:\n{}", String::from_utf8_lossy(&out.stderr));
        let b = |v: &[u8]| v.to_vec();
        let z = |n: usize| vec![0u8; n.max(16)];
        let mut bufs: Vec<Vec<u8>> = vec![Vec::new(); 31];
        bufs[0] = z(64);
        bufs[2] = b(bytemuck::cast_slice(&st.node_static));
        bufs[3] = b(bytemuck::cast_slice(&st.node_range));
        bufs[4] = b(bytemuck::cast_slice(&st.rec_a));
        bufs[5] = b(bytemuck::cast_slice(&st.rec_k));
        bufs[6] = b(bytemuck::cast_slice(&st.rec_b));
        bufs[7] = b(bytemuck::cast_slice(&st.rec_e));
        bufs[9] = b(bytemuck::cast_slice(&st.rec_ent));
        bufs[10] = b(bytemuck::cast_slice(&st.rec_cen));
        bufs[13] = b(bytemuck::cast_slice(&st.add_info));
        bufs[14] = b(bytemuck::cast_slice(&st.perm));
        bufs[17] = b(bytemuck::cast_slice(&st.tri_breaks));
        bufs[22] = if st.cands.is_empty() { z(16) } else { b(bytemuck::cast_slice(&st.cands)) };
        bufs[23] = b(bytemuck::cast_slice(&st.tris));
        bufs[24] = b(bytemuck::cast_slice(&st.allowed));
        bufs[12] = if st.v2_tab.is_empty() { z(l.ncen * 16) } else { b(bytemuck::cast_slice(&st.v2_tab)) };
        bufs[25] = z(l.nt * GEO * 16);
        bufs[26] = z(PAIR_CAP * 4);
        bufs[27] = z(l.nn * 16);
        bufs[29] = z(HIT_CAP * HITW * 16);
        bufs[30] = z(l.nn * 48);
        Ok(Self { l, exe, dir: dir.to_path_buf(), bufs, reverse: false, v2: !st.v2_tab.is_empty(), off: st.v2_off.clone(), groups: if st.pair != 0 { st.pair } else { 1 }, steps: STEPS })
    }

    pub fn upload(&mut self, s: &State) {
        let b = |v: &[u8]| v.to_vec();
        if self.v2 {
            self.upload_at(0, s);
            return;
        }
        self.bufs[1] = b(bytemuck::cast_slice(&s.node));
        self.bufs[8] = b(bytemuck::cast_slice(&s.rec));
        self.bufs[11] = b(bytemuck::cast_slice(&s.add));
        if !self.v2 {
            self.bufs[12].fill(0);
        }
        self.bufs[15] = b(bytemuck::cast_slice(&s.groups_last));
        self.bufs[16] = b(bytemuck::cast_slice(&s.deform));
        self.bufs[18] = b(bytemuck::cast_slice(&s.tri_dead));
        self.bufs[19] = b(bytemuck::cast_slice(&s.acc));
        self.bufs[20] = b(bytemuck::cast_slice(&s.shift));
        self.bufs[21] = b(bytemuck::cast_slice(&s.misc));
        self.bufs[28] = b(bytemuck::cast_slice(&s.clip));
        self.bufs[30].fill(0);
    }

    /// Body `car`'s state into its block (the second layout).
    pub fn upload_at(&mut self, car: usize, s: &State) {
        let n = self.off.words * 4;
        if self.bufs[1].len() != self.groups * n {
            self.bufs[1] = vec![0u8; self.groups * n];
            for i in [8, 11, 15, 16, 18, 19, 20, 21, 28] {
                self.bufs[i] = vec![0u8; 16];
            }
        }
        v2_put(&mut self.bufs[1][car * n..(car + 1) * n], &self.off, s);
    }

    pub fn download_at(&self, car: usize) -> State {
        let n = self.off.words * 4;
        v2_get(&self.bufs[1][car * n..(car + 1) * n], &self.off, &self.l)
    }

    pub fn download(&self) -> State {
        if self.v2 {
            return self.download_at(0);
        }
        fn g<T: bytemuck::Pod>(v: &[u8]) -> Vec<T> {
            v.chunks_exact(std::mem::size_of::<T>()).map(bytemuck::pod_read_unaligned).collect()
        }
        State {
            node: g(&self.bufs[1]),
            rec: g(&self.bufs[8]),
            add: g(&self.bufs[11]),
            groups_last: g(&self.bufs[15]),
            deform: g(&self.bufs[16]),
            tri_dead: g(&self.bufs[18]),
            acc: g(&self.bufs[19]),
            shift: g(&self.bufs[20]),
            misc: g(&self.bufs[21]),
            clip: g(&self.bufs[28]),
            rig: Vec::new(),
        }
    }

    pub fn run(&mut self, frames: u32) -> anyhow::Result<()> {
        let l = &self.l;
        if self.v2 && self.groups > 1 {
            // (The cross-group barrier's counter starts at zero each dispatch.)
            let n = self.off.words * 4;
            for g in (2..self.groups).step_by(3) {
                let at = g * n + (self.off.misc + 14) * 4;
                self.bufs[1][at..at + 4].fill(0);
            }
        }
        let p: [u32; 12] = [self.groups as u32, frames, 0.0f32.to_bits(), 0, LANES as u32, l.nn as u32, l.nt as u32, l.nc.max(1) as u32, self.steps as u32, l.ng as u32, l.rs as u32, 1];
        self.bufs[0] = bytemuck::cast_slice(&p).to_vec();
        self.bufs[0].resize(64, 0);
        let (fin, fout) = (self.dir.join("in.blob"), self.dir.join("out.blob"));
        let mut blob = Vec::new();
        for b in &self.bufs {
            blob.extend_from_slice(&(b.len() as u64).to_le_bytes());
            blob.extend_from_slice(b);
        }
        std::fs::write(&fin, blob)?;
        let out = std::process::Command::new(&self.exe).arg(&fin).arg(&fout).arg(if self.reverse { "1" } else { "0" }).output()?;
        anyhow::ensure!(out.status.success(), "emu run: {:?}\n{}", out.status, String::from_utf8_lossy(&out.stderr));
        if std::env::var("EMU_TRACE").is_ok() {
            eprint!("{}", String::from_utf8_lossy(&out.stderr));
        }
        let blob = std::fs::read(&fout)?;
        let mut at = 0;
        for b in &mut self.bufs {
            let n = u64::from_le_bytes(blob[at..at + 8].try_into().unwrap()) as usize;
            *b = blob[at + 8..at + 8 + n].to_vec();
            at += 8 + n;
        }
        Ok(())
    }
}


/// `--skin M` and `--skin-out M` of the command line: the working and the wide pair
/// lists' skins (m; defaults 0.05 and 0.30; a wide skin no larger than the working one:
/// one list, built from the clusters every time).
pub fn list_args(st: &mut Statics) {
    let a: Vec<String> = std::env::args().collect();
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).map(|x| x.parse::<f32>().unwrap());
    let base = st.cr + engine_softbody::explicit_contact::CLIP_DEPTH;
    if let Some(sk) = arg("--skin") {
        st.list_reach = base + sk;
        st.list_reach_out = base + sk + LIST_SKIN_OUT - LIST_SKIN;
    }
    if let Some(sk) = arg("--skin-out") {
        st.list_reach_out = base + sk;
    }
}

/// The second layout (kernels/floor2.metal) for this car: which lane holds which node,
/// triangle and beam slots, each node's entry list in the reference's add order, and the
/// vector slots of the elements whose force is not along a beam.
pub fn make_v2(s: &ExplicitBeams, l: &mut Layout, st: &mut Statics) -> anyhow::Result<()> {
    use ExplicitKind as K;
    let (nn, nr, nt) = (l.nn, l.nr, l.nt);
    anyhow::ensure!(nn <= 1023 && nr < (1 << 24));
    let scalar = |k: K| matches!(k, K::JbeamPlain | K::JbeamSupport | K::JbeamBounded | K::JbeamAniso | K::Plain);
    for b in &s.beams {
        if matches!(b.kind, K::JbeamPlain | K::JbeamSupport) {
            anyhow::ensure!(b.ty.code() == 0 && b.a != b.b, "a plain or support beam the fast path does not cover");
        }
    }
    // Scalar owners: plain by yield gate (a lane's slots near each other), supports, the rest.
    let mut plain: Vec<usize> = (0..nr).filter(|&j| s.beams[j].kind == K::JbeamPlain).collect();
    plain.sort_by(|&x, &y| s.beams[y].gate.min(s.beams[y].strength).total_cmp(&s.beams[x].gate.min(s.beams[x].strength)).then(x.cmp(&y)));
    // Nodes on lanes by entry count (a warp's lanes loop alike).
    let mut nadds = vec![0usize; nn];
    for j in 0..nr {
        for &n in &adds_of(s, j) {
            nadds[n as usize] += 1;
        }
    }
    let mut order: Vec<usize> = (0..nn).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(nadds[i]), i));
    // The owning layout: every node the owner of three of its plain beams, a beam one
    // owner (augmenting paths); they take slots 0 to 2 of the node's lane.
    let mut owner_of = vec![usize::MAX; nr];
    let mut plain_front: Vec<usize> = Vec::new();
    if st.own {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nn];
        for &j in &plain {
            adj[s.beams[j].a as usize].push(j);
            adj[s.beams[j].b as usize].push(j);
        }
        fn aug(i: usize, adj: &[Vec<usize>], owner: &mut [usize], seen: &mut [bool]) -> bool {
            for &k in &adj[i] {
                if seen[k] {
                    continue;
                }
                seen[k] = true;
                if owner[k] == usize::MAX || aug(owner[k], adj, owner, seen) {
                    owner[k] = i;
                    return true;
                }
            }
            false
        }
        for _ in 0..3 {
            for i in 0..nn {
                let mut seen = vec![false; nr];
                anyhow::ensure!(aug(i, &adj, &mut owner_of, &mut seen), "node {i} cannot own three plain beams");
            }
        }
        let mut owned: Vec<Vec<usize>> = vec![Vec::new(); nn];
        for &j in &plain {
            if owner_of[j] != usize::MAX {
                owned[owner_of[j]].push(j);
            }
        }
        for lane in 0..nn {
            let mut o = owned[order[lane]].clone();
            o.sort();
            anyhow::ensure!(o.len() == 3);
            plain_front.extend(o);
        }
        plain.retain(|&j| owner_of[j] == usize::MAX);
    }
    let mut owners = plain_front;
    owners.extend(plain);
    owners.extend((0..nr).filter(|&j| s.beams[j].kind == K::JbeamSupport));
    owners.extend((0..nr).filter(|&j| scalar(s.beams[j].kind) && !matches!(s.beams[j].kind, K::JbeamPlain | K::JbeamSupport)));
    let full = 3 * LANES;
    let tail = owners.len().saturating_sub(full);
    let aero_lo = tail.div_ceil(32) * 32;
    let ngs = if owners.len() > full { owners.len() } else { full };
    anyhow::ensure!(ngs <= 4096);
    let mut perm = vec![NONE; 4 * LANES];
    let mut gidx = vec![NONE; nr];
    for (j, &r) in owners.iter().enumerate() {
        let (slot, lane) = if j < full { (j % 3, j / 3) } else { (3, j - full) };
        perm[slot * LANES + lane] = r as u32;
        gidx[r] = (slot * LANES + lane) as u32;
    }
    // Vector owners in slot 3 from AERO_LO: aero triangles, couplers, torsion bars.
    let aero: Vec<usize> = (0..nr).filter(|&j| s.beams[j].kind == K::JbeamAeroTri).collect();
    let cpl: Vec<usize> = (0..nr).filter(|&j| s.beams[j].kind == K::JbeamCoupler).collect();
    let tor: Vec<usize> = (0..nr).filter(|&j| s.beams[j].kind == K::JbeamTorsion).collect();
    anyhow::ensure!(aero_lo + aero.len() + cpl.len() + tor.len() <= LANES, "no room for the vector elements");
    let mut vidx = vec![0u32; nr];
    let mut lane = aero_lo;
    for (i, &r) in aero.iter().enumerate() {
        perm[3 * LANES + lane] = r as u32;
        vidx[r] = i as u32;
        lane += 1;
    }
    let cpl_base = aero.len();
    for (i, &r) in cpl.iter().enumerate() {
        perm[3 * LANES + lane] = r as u32;
        vidx[r] = (cpl_base + i) as u32;
        lane += 1;
    }
    for (i, &r) in tor.iter().enumerate() {
        perm[3 * LANES + lane] = r as u32;
        vidx[r] = (cpl_base + cpl.len() + 4 * i) as u32;
        lane += 1;
    }
    let nv = cpl_base + cpl.len() + 4 * tor.len();
    anyhow::ensure!(nv <= 4096 && cpl.len() * 6 <= nn);
    // Each node's entries in the reference's add order.
    let mut ents: Vec<Vec<u32>> = vec![Vec::new(); nn];
    for j in 0..nr {
        let b = &s.beams[j];
        for (q, &n) in adds_of(s, j).iter().enumerate() {
            // (The owner adds its own beam's force from registers: no entry.)
            if owner_of[j] == n as usize {
                continue;
            }
            let e = match b.kind {
                K::JbeamAeroTri => 1 << 31 | 1 << 29 | vidx[j],
                K::JbeamCoupler => 1 << 31 | (q as u32) << 30 | vidx[j],
                K::JbeamTorsion => 1 << 31 | (vidx[j] + q as u32),
                _ => {
                    anyhow::ensure!(scalar(b.kind) && gidx[j] != NONE, "record {j}: no slot");
                    let other = if q == 0 { b.b } else { b.a };
                    other | gidx[j] << 11
                }
            };
            ents[n as usize].push(e);
        }
    }
    // Triangles on lanes by their pair count in this pose.
    let count_at = |t: usize, reach: f32| -> usize {
        let p = s.triangles[t].nodes.map(|i| s.nodes[i as usize].pos + s.origin);
        let lo = p[0].min(p[1]).min(p[2]) - Vec3::splat(reach);
        let hi = p[0].max(p[1]).max(p[2]) + Vec3::splat(reach);
        (0..l.nc).filter(|&c| st.allowed[t * l.ncw + c / 32] >> (c % 32) & 1 != 0).filter(|&c| {
            let x = s.nodes[s.candidates[c] as usize].pos + s.origin;
            !(x.cmplt(lo).any() || x.cmpgt(hi).any())
        }).count()
    };
    let counts: Vec<usize> = (0..nt).map(|t| count_at(t, st.list_reach)).collect();
    // (A row holds the wide list's pairs: twice this pose's longest, for a crushed car's.)
    let list2 = st.list_reach_out > st.list_reach;
    let widest = (0..nt).map(|t| count_at(t, st.list_reach_out.max(st.list_reach))).max().unwrap_or(0);
    let mut tri_order: Vec<usize> = (0..nt).collect();
    tri_order.sort_by_key(|&t| (std::cmp::Reverse(counts[t]), t));
    anyhow::ensure!(nt <= LANES);
    let prow = (((PAIR_CAP / nt.max(1)) & !3).min(124)).max((2 * widest + 16 + 3) & !3);
    anyhow::ensure!(prow <= 1020 && widest + 2 < prow, "a triangle's pairs past its row ({widest} of {prow})");
    // The contacters in clusters of at most 28 along the car's longest extent in this pose
    // (the pair list's build tests a cluster's box first), and each triangle's allowed
    // contacters a word a cluster.
    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
    for n in &s.nodes {
        lo = lo.min(n.pos);
        hi = hi.max(n.pos);
    }
    let ext = hi - lo;
    let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
    let mut by_axis: Vec<usize> = (0..l.nc).collect();
    by_axis.sort_by(|&x, &y| s.nodes[s.candidates[x] as usize].pos[axis].total_cmp(&s.nodes[s.candidates[y] as usize].pos[axis]).then(x.cmp(&y)));
    let nclu = l.nc.div_ceil(28).max(1);
    anyhow::ensure!(nclu <= LANES);
    let mut clu = vec![NONE; nclu * 32];
    let mut allowed_c = vec![0u32; nt * nclu];
    for (i, &c) in by_axis.iter().enumerate() {
        let (k, j) = (i / 28, i % 28);
        clu[k * 32 + j] = s.candidates[c];
        for t in 0..nt {
            if st.allowed[t * l.ncw + c / 32] >> (c % 32) & 1 != 0 {
                allowed_c[t * nclu + k] |= 1 << j;
            }
        }
    }
    let t_lane = 0usize;
    let t_vidx = t_lane + LANES * 4;
    let t_clu = (t_vidx + nr + 3) & !3;
    // For a contact pair: every contact point in clusters (the partner's cull), and each
    // node's neighbours (the side vote), as the reference's `finish_contact` lists them.
    let mut cps: Vec<u32> = s.contact_points.clone();
    cps.sort_by(|&x, &y| s.nodes[x as usize].pos[axis].total_cmp(&s.nodes[y as usize].pos[axis]).then(x.cmp(&y)));
    let nclp = cps.len().div_ceil(28).max(1);
    let mut pcl = vec![NONE; nclp * 32];
    for (i, &h) in cps.iter().enumerate() {
        pcl[(i / 28) * 32 + i % 28] = h;
    }
    let t_pcl = t_clu + nclu * 32;
    let t_nb = t_pcl + nclp * 32;
    let t_nbl = t_nb + nn + 1;
    let nbl = if s.neighbour_start.len() == nn + 1 { s.neighbours.len() } else { 0 };
    let t_ent = (t_nbl + nbl + 3) & !3;
    let mut tab = vec![0u32; t_ent];
    tab[t_clu..t_clu + nclu * 32].copy_from_slice(&clu);
    tab[t_pcl..t_pcl + nclp * 32].copy_from_slice(&pcl);
    if nbl != 0 || s.neighbour_start.len() == nn + 1 {
        tab[t_nb..t_nb + nn + 1].copy_from_slice(&s.neighbour_start);
        tab[t_nbl..t_nbl + nbl].copy_from_slice(&s.neighbours);
    }
    tab[t_vidx..t_vidx + nr].copy_from_slice(&vidx);
    let mut maxent = 0;
    for lane in 0..LANES {
        let (mut node, mut start, mut cnt) = (NONE, 0u32, 0u32);
        if lane < nn {
            let i = order[lane];
            node = i as u32;
            start = (tab.len() - t_ent) as u32;
            cnt = ents[i].len() as u32;
            maxent = maxent.max(ents[i].len());
            tab.extend_from_slice(&ents[i]);
            while tab.len() % 4 != 0 {
                tab.push(0);
            }
        }
        tab[t_lane + lane * 4] = node;
        tab[t_lane + lane * 4 + 1] = start;
        // (The leading entries with no vector among them, then all of them.)
        let lead = if lane < nn { ents[order[lane]].iter().take_while(|&&e| e >> 31 == 0).count() as u32 } else { 0 };
        // (An aero entry is never followed by another kind: aero records are the last.)
        let pre = if lane < nn { ents[order[lane]].iter().take_while(|&&e| !(e >> 31 == 1 && e >> 29 & 1 == 1)).count() as u32 } else { 0 };
        if lane < nn {
            anyhow::ensure!(ents[order[lane]][pre as usize..].iter().all(|&e| e >> 31 == 1 && e >> 29 & 1 == 1), "an entry after an aero triangle's");
        }
        anyhow::ensure!(cnt < 256 && nn < 2048);
        tab[t_lane + lane * 4 + 2] = lead | pre << 8 | cnt << 16;
        tab[t_lane + lane * 4 + 3] = if lane < nt { tri_order[lane] as u32 } else { NONE };
    }
    tab.extend_from_slice(&[0; 8]);
    let mut o = V2Off::default();
    let mut at = 0usize;
    let mut take = |words: usize| {
        let x = at;
        at = (at + words + 3) & !3;
        x
    };
    o.node = take(nn * 12);
    o.rec = take(nr * 12);
    o.groups = take(l.ng);
    o.deform = take(l.nd);
    o.tridead = take(nt);
    o.acc = take(nn * 3);
    o.shift = take(nn * 3);
    o.misc = take(16);
    o.pairs = take(nt * prow);
    o.pbuild = take(nn * 4);
    o.pairs_out = take(nt * prow);
    o.pbuild_out = take(nn * 4 + 4);
    o.clip = take(nn * 4);
    o.hits = take(HIT_CAP * HITW * 4);
    o.scratch = take(nn * 12);
    o.hot = take(LANES * 24);
    o.cand = take(4096);
    o.clu = take(nclu * 8);
    let (nb, nsh) = s.rigid.as_ref().map_or((0, 0), |r| (r.bodies.len(), r.shapes.len()));
    o.rig_words = if nb != 0 { nb * BDW * 4 + 64 + nsh * SHW * 4 + nsh * 6 } else { 0 };
    o.rig = take(o.rig_words);
    // A contact pair's pass (pair.metal): what a body publishes for its partner (box, top
    // speed, hit count); the owner's staging sums and hit counts for both bodies' nodes;
    // its lists (triangles, triangle and cluster, candidates) and its hits.
    let o_touch = take(128);
    o.touch = o_touch;
    let xcnd_cap = 8192usize;
    let x_pub = take(16);
    let x_stg = take(nn * 12);
    let x_spl = take(nn * 2);
    let x_tri = take(nt);
    let x_tc = take(nt * nclp);
    let x_cnd = take(xcnd_cap);
    let x_hit = take(1024 * 8 * 4);
    anyhow::ensure!(nn * 4 + nclp * 6 <= ngs && nclp < 256 && nt < 65536, "the partner's nodes do not fit the scalars' array ({} of {ngs})", nn * 4 + nclp * 6);
    o.words = at;
    st.perm = perm;
    st.allowed = allowed_c;
    st.v2_tab = tab;
    st.v2_off = o.clone();
    st.v2_defs = format!("#define RIGID {}\n#define NB {}\n#define NSH {}\n#define O_RIG {}\n#define OWN {}\n", (nb != 0) as u32, nb.max(1), nsh.max(1), o.rig, st.own as u32) + &format!(
        "#define LIST2 {}\n#define LIST_REACH_OUT as_type<float>({}u)\n#define O_PAIRSO {}\n#define O_PBUILDO {}\n#define O_TOUCH {o_touch}\n#define O_XPUB {x_pub}\n#define O_XSTG {x_stg}\n#define O_XSPL {x_spl}\n#define O_XTRI {x_tri}\n#define O_XTC {x_tc}\n#define O_XCND {x_cnd}\n#define XCND_CAP {xcnd_cap}u\n#define O_XHIT {x_hit}\n",
        list2 as u32, st.list_reach_out.to_bits(), o.pairs_out, o.pbuild_out
    ) + &format!(
        "#define O_NODE {}\n#define O_REC {}\n#define O_GROUPS {}\n#define O_DEFORM {}\n#define O_TRIDEAD {}\n#define O_ACC {}\n#define O_SHIFT {}\n#define O_MISC {}\n#define O_PAIRS {}\n#define O_PBUILD {}\n#define O_CLIP {}\n#define O_HITS {}\n#define O_SCRATCH {}\n#define O_HOT {}\n#define O_CAND {}\n#define CAND_CAP 2048\n#define O_CLU {}\n#define CAR_WORDS {}\n",
        o.node, o.rec, o.groups, o.deform, o.tridead, o.acc, o.shift, o.misc, o.pairs, o.pbuild, o.clip, o.hits, o.scratch, o.hot, o.cand, o.clu, o.words
    ) + &format!(
        "#define NGS {ngs}\n#define NV {}\n#define AERO_LO {aero_lo}\n#define CPL_BASE {cpl_base}\n#define T_LANE {t_lane}\n#define T_VIDX {t_vidx}\n#define T_ENT {t_ent}\n#define MAXENT {maxent}\n#define PROW {prow}\n#define REFNODE 0\n#define HAS_IMMOVABLE {}\n#define FULL_PLAIN_SLOTS {}\n#define NCLU {nclu}\n#define T_CLU {t_clu}\n#define NCLP {nclp}\n#define T_PCL {t_pcl}\n#define T_NB {t_nb}\n#define T_NBL {t_nbl}\n",
        nv.max(1), s.nodes.iter().any(|n| n.flags & 2 != 0) as u32,
        (0..3).take_while(|&sl| (0..LANES).all(|ln| st.perm[sl * LANES + ln] != NONE && s.beams[st.perm[sl * LANES + ln] as usize].kind == K::JbeamPlain)).count()
    );
    l.rs = 4;
    Ok(())
}


/// Every `for (uint s = 0; s < RS ...; s++) { body }` of the kernel written out once a slot
/// with `s` a constant. Metal's compiler does not unroll these loops (their bodies are
/// large), and an array indexed by a loop variable stays in stack memory: the slots' hot
/// words were then loaded and stored through memory in every phase instead of living in
/// registers. A `continue` in a body leaves its copy (the copies are `do { } while (0)`).
pub fn unroll_slots(src: &str) -> String {
    const HEAD: &str = "for (uint s = 0; s < RS";
    let mut out = String::with_capacity(src.len() * 3);
    let mut rest = src;
    while let Some(at) = rest.find(HEAD) {
        out.push_str(&rest[..at]);
        let after = &rest[at + HEAD.len()..];
        let close = after.find("; s++)").expect("the slot loop's header");
        let cond = after[..close].trim().trim_start_matches("&&").trim().to_string();
        let body_from = close + "; s++)".len();
        let open = body_from + after[body_from..].find('{').expect("the slot loop's body");
        assert!(after[body_from..open].trim().is_empty(), "a slot loop without braces");
        let (mut depth, mut end) = (0i32, 0usize);
        for (i, ch) in after[open..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        assert!(end != 0, "the slot loop's closing brace");
        let body = &after[open..end];
        for k in 0..4 {
            let guard = if cond.is_empty() { String::new() } else { format!("if ({cond}) ") };
            out.push_str(&format!("{{ const uint s = {k}u; {guard}do {body} while (0); }}\n"));
        }
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}
