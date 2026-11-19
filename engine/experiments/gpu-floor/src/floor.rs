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
    pub cr: f32,
    pub boxes: Vec<([f32; 3], [f32; 3])>,
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
    anyhow::ensure!(s.damp_scheme == 0 && s.shapes.is_empty() && s.rigid.is_none() && !s.surface_contact);
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
        Statics { node_static, node_range, rec_a, rec_k, rec_b, rec_e, rec_ent, rec_cen, add_info, perm, tri_breaks, cands: s.candidates.clone(), tris, allowed, list_reach, cr, boxes: s.boxes.iter().map(|b| (b.lo.to_array(), b.hi.to_array())).collect() },
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
    let mut clip = vec![engine_softbody::explicit_contact::CLIP_EMPTY; l.nn * 4];
    if s.clip_sides.len() == l.nn {
        for (i, c) in s.clip_sides.iter().enumerate() {
            clip[i * 4..i * 4 + 4].copy_from_slice(c);
        }
    }
    State { node, rec, add, groups_last, deform, tri_dead, acc, shift, misc, clip }
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
            node: z(l.nn * 3 * 16),
            rec: z(l.nr * 3 * 16),
            add: z(l.nadd * 16),
            cen: z(l.ncen * 16),
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
        let put = |b: &Buffer, off: usize, data: &[u8]| unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), (b.contents() as *mut u8).add(off), data.len())
        };
        put(&self.node, car * l.nn * 48, bytemuck::cast_slice(&s.node));
        put(&self.rec, car * l.nr * 48, bytemuck::cast_slice(&s.rec));
        put(&self.add, car * l.nadd * 16, bytemuck::cast_slice(&s.add));
        put(&self.cen, car * l.ncen * 16, &vec![0u8; l.ncen * 16]);
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
        }
    }

    /// Every per-car buffer with its size a car.
    fn per_car(&self) -> Vec<(&Buffer, usize)> {
        let l = &self.l;
        vec![
            (&self.node, l.nn * 48), (&self.rec, l.nr * 48), (&self.add, l.nadd * 16), (&self.cen, l.ncen * 16), (&self.groups, l.ng * 4),
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
        write(&self.params, &[self.cars as u32, if self.per_frame > 1 { 1 } else { frames }, 0.0f32.to_bits(), skip, LANES as u32, l.nn as u32, l.nt as u32, l.nc.max(1) as u32, steps as u32, l.ng as u32, l.rs as u32, 0]);
        let mut bufs: Vec<&Buffer> = vec![&self.params, &self.node, &self.stat[0], &self.stat[1], &self.stat[2], &self.stat[3], &self.stat[4], &self.stat[5]];
        bufs.extend([&self.rec, &self.stat[6], &self.stat[7], &self.add, &self.cen, &self.stat[8], &self.stat[9], &self.groups, &self.deform, &self.stat[10], &self.tri_dead, &self.acc, &self.shift, &self.misc]);
        bufs.extend([&self.stat[11], &self.stat[12], &self.stat[13], &self.tri_geo, &self.pairs, &self.pbuild, &self.clip, &self.hits, &self.stage]);
        let n = if self.per_frame > 1 { self.per_frame * frames as usize } else { 1 };
        self.gpu.run_n(&self.pso, &bufs, self.cars as u64, LANES as u64, n)
    }
}

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
        for (name, gv, cv) in [("rest", s0[0], b.rest), ("tc", s0[1], b.tc), ("tt", s0[2], b.tt), ("gate", s0[3], b.gate), ("strength", s1[0], b.strength), ("force", s1[1], b.force), ("slope", s1[2], slope)] {
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
            .args(["clang++", "-std=c++20", "-O1", "-ffp-contract=off", "-w", "-I"])
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
        bufs[12] = z(l.ncen * 16);
        bufs[25] = z(l.nt * GEO * 16);
        bufs[26] = z(PAIR_CAP * 4);
        bufs[27] = z(l.nn * 16);
        bufs[29] = z(HIT_CAP * HITW * 16);
        bufs[30] = z(l.nn * 48);
        Ok(Self { l, exe, dir: dir.to_path_buf(), bufs, reverse: false, steps: STEPS })
    }

    pub fn upload(&mut self, s: &State) {
        let b = |v: &[u8]| v.to_vec();
        self.bufs[1] = b(bytemuck::cast_slice(&s.node));
        self.bufs[8] = b(bytemuck::cast_slice(&s.rec));
        self.bufs[11] = b(bytemuck::cast_slice(&s.add));
        self.bufs[12].fill(0);
        self.bufs[15] = b(bytemuck::cast_slice(&s.groups_last));
        self.bufs[16] = b(bytemuck::cast_slice(&s.deform));
        self.bufs[18] = b(bytemuck::cast_slice(&s.tri_dead));
        self.bufs[19] = b(bytemuck::cast_slice(&s.acc));
        self.bufs[20] = b(bytemuck::cast_slice(&s.shift));
        self.bufs[21] = b(bytemuck::cast_slice(&s.misc));
        self.bufs[28] = b(bytemuck::cast_slice(&s.clip));
        self.bufs[30].fill(0);
    }

    pub fn download(&self) -> State {
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
        }
    }

    pub fn run(&mut self, frames: u32) -> anyhow::Result<()> {
        let l = &self.l;
        let p: [u32; 12] = [1, frames, 0.0f32.to_bits(), 0, LANES as u32, l.nn as u32, l.nt as u32, l.nc.max(1) as u32, self.steps as u32, l.ng as u32, l.rs as u32, 0];
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
