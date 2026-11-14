//! **Measurement only**: the visual binding as it was before the flexbody
//! rebinding (4e9db94a), for a visual A/B on identical physics
//! (`showcaseCar({ bind: 'legacy' })`, `crash-lab-record.mjs --query
//! bind=legacy`). Each vertex rides its voted part's single-part skin
//! triangles; glass panes, lamps and handles ride one rigid triangle. Not
//! for production: its vertices lie up to 0.37 m off their triangles.
use super::*;

pub(super) fn build_visual_legacy(built: &Built) -> Result<ShowcaseVisual, String> {
    let def = &built.def;
    let sk = &built.skeleton;
    let paint = built.params.paint.unwrap_or(def.look.paint);
    // The solver's nodes (tyre nodes included), in definition coordinates.
    let sim = ExplicitBeams::from_truck(
        &built.truck,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )?;
    let node_pos: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos - built.shift).collect();
    let mut pieces = assemble(def, &sedan_styling(def));
    {
        let mut names: Vec<&String> = built.bind_tris.keys().collect();
        names.sort();
        let gi = |n: &str| names.iter().position(|g| *g == n).map(|i| i as u16);
        pieces.extend(crate::authoring::parts::interior::interior(
            &gi,
            crate::authoring::parts::interior::Cabin::of(def),
        ));
        let nodes = |n: &str| -> Option<Vec<Vec3>> {
            Some(built.groups.get(n)?.iter().map(|&i| sk.pos(i)).collect())
        };
        pieces.extend(crate::authoring::parts::bay::engine_bay(&gi, &nodes));
    }
    // The pieces' bindings as they were: tempered glass, lamps and handles
    // rigid on one triangle; a structure piece on its first group only.
    for p in pieces.iter_mut() {
        let n = p.name.as_str();
        let rigid = n == "backlight"
            || n.starts_with("glass_side")
            || ["headlight_", "taillight_", "foglamp_"]
                .iter()
                .any(|k| n.starts_with(k))
            || n == "lightbar"
            || n == "handle";
        if rigid && p.bind == Bind::Vote {
            p.bind = Bind::Rigid;
        }
        if let Bind::Groups([g, ..]) = p.bind {
            p.bind = Bind::Group(g);
        }
    }
    // Binders: per part, all skin, per group, per tyre and hub.
    let mut by_part: HashMap<u16, Vec<[u32; 3]>> = HashMap::new();
    for t in &sk.skin {
        let p = sk.nodes[t[0] as usize].part;
        if t.iter().all(|&i| sk.nodes[i as usize].part == p) {
            by_part.entry(p).or_default().push(*t);
        }
    }
    let fallback = Binder::new(sk.skin.clone(), &node_pos, 0.15);
    // Single-part skin triangles of every part (votes and reach checks).
    let pure = Binder::new(
        by_part.values().flatten().copied().collect(),
        &node_pos,
        0.15,
    );
    let mut binders: HashMap<u16, Binder> = HashMap::new();
    let groups: Vec<String> = {
        let mut g: Vec<String> = built.bind_tris.keys().cloned().collect();
        g.sort();
        g
    };
    let tyre_centre: Vec<Vec3> = sim
        .tyres
        .iter()
        .map(|t| 0.5 * (node_pos[t.axle[0] as usize] + node_pos[t.axle[1] as usize]))
        .collect();
    let susp: Vec<u32> = built.groups.get("suspension").cloned().unwrap_or_default();
    let mut out = ShowcaseVisual {
        node_count: node_pos.len() as u32,
        node_rest: sim.nodes.iter().map(|n| n.pos.to_array()).collect(),
        ..Default::default()
    };
    for piece in pieces.iter_mut() {
        piece.mesh.orient_by_normals();
    }
    // A voted piece that reaches well past its physics part (its part's skin
    // 8 cm farther than another's) is cut there, along the smooth zero line of
    // that distance difference (marching triangles), and the far side rides
    // the other part: it parts along a clean line instead of stretching.
    let vote = |m: &Mesh| -> Option<u16> {
        let mut votes: HashMap<u16, usize> = HashMap::new();
        let step = (m.len() / 600).max(1);
        for k in (0..m.len()).step_by(step) {
            if let Some((t, _, _)) = pure.bind(m.pos[k], m.nrm[k]) {
                *votes.entry(sk.nodes[t[0] as usize].part).or_default() += 1;
            }
        }
        votes
            .into_iter()
            .filter(|(p, _)| by_part.contains_key(p))
            .max_by_key(|&(p, n)| (n, std::cmp::Reverse(p)))
            .map(|(p, _)| p)
    };
    let mut queue: Vec<Piece> = pieces;
    let mut pieces: Vec<Piece> = Vec::new();
    let mut rounds = 0;
    while let Some(piece) = queue.pop() {
        rounds += 1;
        if let Bind::Inset(mm) = piece.bind {
            // Vote at the outer skin (the inner panel's normals face in).
            let mut probe = piece.mesh.clone();
            for k in 0..probe.len() {
                probe.pos[k] -= probe.nrm[k] * (mm as f32 * 1e-3);
                probe.nrm[k] = -probe.nrm[k];
            }
            if let Some(pt) = vote(&probe) {
                pieces.push(Piece {
                    bind: Bind::PartInset(pt, mm),
                    ..piece
                });
                continue;
            }
        }
        let pt = match piece.bind {
            Bind::Vote if rounds < 400 => vote(&piece.mesh),
            _ => None,
        };
        let Some(pt) = pt else {
            pieces.push(piece);
            continue;
        };
        let own = Binder::new(by_part[&pt].clone(), &node_pos, 0.15);
        let g: Vec<f32> = (0..piece.mesh.len())
            .map(|k| {
                let (p, n) = (piece.mesh.pos[k], piece.mesh.nrm[k]);
                match (own.bind_d(p, n), pure.bind_d(p, n)) {
                    (Some(a), Some(z)) if sk.nodes[z.0[0] as usize].part != pt => a.3 - z.3 - 0.08,
                    _ => -1.0,
                }
            })
            .collect();
        if g.iter().all(|&v| v < 0.0) {
            pieces.push(Piece {
                bind: Bind::Part(pt),
                ..piece
            });
            continue;
        }
        let main = crate::authoring::parts::trim::clip(&piece.mesh, &g, 0.0).mesh;
        let neg: Vec<f32> = g.iter().map(|v| -v).collect();
        let rest = crate::authoring::parts::trim::clip(&piece.mesh, &neg, 0.0).mesh;
        if !main.is_empty() {
            pieces.push(Piece {
                name: piece.name.clone(),
                mesh: main,
                bind: Bind::Part(pt),
            });
        }
        if !rest.is_empty() {
            queue.push(Piece {
                name: format!("{}+", piece.name),
                mesh: rest,
                bind: Bind::Vote,
            });
        }
    }
    for piece in &pieces {
        let m = &piece.mesh;
        if m.is_empty() {
            continue;
        }
        let first = out.positions.len() as u32;
        let centroid = m.pos.iter().copied().sum::<Vec3>() / m.len() as f32;
        let nearest_tyre = || {
            (0..tyre_centre.len())
                .min_by(|&a, &b| {
                    tyre_centre[a]
                        .distance(centroid)
                        .total_cmp(&tyre_centre[b].distance(centroid))
                })
                .unwrap_or(0)
        };
        // The binder for this piece.
        let own: Option<Binder> = match piece.bind {
            Bind::Wheel(_) => {
                let t = &sim.tyres[nearest_tyre()];
                let rays = (t.count / 2) as usize;
                let ring =
                    |i: usize, outer: bool| t.first + 2 * (i % rays) as u32 + u32::from(!outer);
                let mut tris = Vec::new();
                for i in 0..rays {
                    tris.push([t.axle[0], ring(i, true), ring(i + 1, true)]);
                    tris.push([t.axle[1], ring(i, false), ring(i + 1, false)]);
                }
                Some(Binder::new(tris, &node_pos, 0.1))
            }
            Bind::Hub(_) => {
                let t = &sim.tyres[nearest_tyre()];
                let (a0, a1) = (t.axle[0], t.axle[1]);
                let axis = (node_pos[a1 as usize] - node_pos[a0 as usize]).normalize_or_zero();
                let c = tyre_centre[nearest_tyre()];
                let k = susp
                    .iter()
                    .copied()
                    .filter(|&i| i != a0 && i != a1)
                    .filter(|&i| {
                        let d = node_pos[i as usize] - c;
                        (d - axis * d.dot(axis)).length() > 0.03
                    })
                    .min_by(|&a, &b| {
                        node_pos[a as usize]
                            .distance(c)
                            .total_cmp(&node_pos[b as usize].distance(c))
                    });
                k.map(|k| Binder::new(vec![[a0, a1, k]], &node_pos, 0.5))
            }
            Bind::Group(g) => groups
                .get(g as usize)
                .and_then(|n| built.bind_tris.get(n))
                .map(|t| Binder::new(t.clone(), &node_pos, 0.25)),
            _ => None,
        };
        // A vote for the part.
        let group_rigid = match piece.bind {
            Bind::GroupRigid(g) => groups
                .get(g as usize)
                .and_then(|n| built.bind_tris.get(n))
                .map(|t| Binder::new(t.clone(), &node_pos, 0.25)),
            _ => None,
        };
        let rigid = piece.bind == Bind::Rigid || group_rigid.is_some();
        // An inner stamping binds where the outer panel over it binds: each
        // vertex picks its triangle at the outer skin (moved back out by its
        // depth), then is placed on that triangle at its own depth.
        let inset = if let Bind::PartInset(_, mm) = piece.bind {
            mm as f32 * 1e-3
        } else {
            0.0
        };
        let part: Option<u16> = if let Bind::Part(p) | Bind::PartInset(p, _) = piece.bind {
            Some(p)
        } else if piece.bind == Bind::Vote || rigid {
            let mut votes: HashMap<u16, usize> = HashMap::new();
            let step = (m.len() / 600).max(1);
            for k in (0..m.len()).step_by(step) {
                if let Some((t, _, _)) = pure.bind(m.pos[k], m.nrm[k]) {
                    *votes.entry(sk.nodes[t[0] as usize].part).or_default() += 1;
                }
            }
            votes
                .into_iter()
                .filter(|(p, _)| by_part.contains_key(p))
                .max_by_key(|&(p, n)| (n, std::cmp::Reverse(p)))
                .map(|(p, _)| p)
        } else {
            None
        };
        if rigid {
            // One triangle (the voted part's nearest to the centroid) carries
            // the whole part rigidly.
            let b = group_rigid
                .as_ref()
                .and_then(|g| g.bind(centroid, Vec3::Y))
                .or_else(|| {
                    part.and_then(|pt| {
                        binders
                            .entry(pt)
                            .or_insert_with(|| Binder::new(by_part[&pt].clone(), &node_pos, 0.15))
                            .bind(centroid, Vec3::Y)
                    })
                })
                .or_else(|| pure.bind(centroid, Vec3::Y));
            let Some((t, _, _)) = b else {
                return Err(format!(
                    "no skin triangle for the rigid part {}",
                    piece.name
                ));
            };
            let (p0, p1, p2) = (
                node_pos[t[0] as usize],
                node_pos[t[1] as usize],
                node_pos[t[2] as usize],
            );
            let tt = (p1 - p0).normalize();
            let nn = (p1 - p0).cross(p2 - p0).normalize();
            let bb = nn.cross(tt);
            for k in 0..m.len() {
                let d = m.pos[k] - p0;
                let n = m.nrm[k];
                out.bind_nodes.push(t);
                out.bind_coords.push([d.dot(tt), d.dot(bb), d.dot(nn)]);
                out.bind_normal.push([n.dot(tt), n.dot(bb), n.dot(nn)]);
                out.bind_rest.push(rest_edges(&node_pos, t));
                out.positions.push((m.pos[k] + built.shift).to_array());
                out.normals.push(n.to_array());
                out.uvs.push(m.uv[k]);
                out.material.push(m.mat[k]);
                out.colors.push(color(m.mat[k], paint));
                out.ao.push(m.ao[k].clamp(0.0, 1.0));
                out.smooth.push(0);
                out.rigid.push(1);
            }
            out.indices.extend(m.idx.iter().map(|i| i + first));
            out.pieces.push((piece.name.clone(), first, m.len() as u32));
            continue;
        }
        // Per vertex: its binding and its part. A voted piece rides its part,
        // except where that part's skin is much farther than another's (the
        // piece reaches past the physics panel): those vertices ride the
        // nearer part, and triangles are split along that boundary (vertices
        // duplicated), so the piece parts there instead of stretching.
        type B = ([u32; 3], [f32; 3], [f32; 3]);
        let mut vb: Vec<(B, u16)> = Vec::with_capacity(m.len());
        for k in 0..m.len() {
            let (p, n) = (m.pos[k], m.nrm[k]);
            let r: Option<(B, u16)> = if let Some(o) = &own {
                o.bind(p, n).map(|b| (b, u16::MAX))
            } else if let Some(pt) = part {
                let (pp, np) = if inset > 0.0 {
                    (p - n * inset, -n)
                } else {
                    (p, n)
                };
                let at = |r: Option<([u32; 3], [f32; 3], [f32; 3], f32)>| {
                    r.map(|(t, c, nl, d)| {
                        if inset > 0.0 {
                            let (c, nl) = frame_coords(&node_pos, t, p, n);
                            (t, c, nl, d)
                        } else {
                            (t, c, nl, d)
                        }
                    })
                };
                let near = at(pure.bind_d(pp, np));
                let b = binders
                    .entry(pt)
                    .or_insert_with(|| Binder::new(by_part[&pt].clone(), &node_pos, 0.15));
                let mine = at(b.bind_d(pp, np));
                match (mine, near) {
                    (Some(a), Some(z)) => {
                        let zp = sk.nodes[z.0[0] as usize].part;
                        // 1.5 cm more than the vote's cut (8 cm): a piece cut
                        // there has vertices on the cut line itself, which
                        // must stay on their side, or the line opens as a
                        // sawtooth of triangles riding alternate parts.
                        if a.3 > z.3 + 0.095 && zp != pt {
                            Some(((z.0, z.1, z.2), zp))
                        } else {
                            Some(((a.0, a.1, a.2), pt))
                        }
                    }
                    (Some(a), None) => Some(((a.0, a.1, a.2), pt)),
                    (None, Some(z)) => Some(((z.0, z.1, z.2), u16::MAX)),
                    _ => None,
                }
            } else {
                None
            };
            let Some(r) = r.or_else(|| fallback.bind(p, n).map(|b| (b, u16::MAX))) else {
                return Err(format!("no skin triangle for a vertex at {p:?}"));
            };
            vb.push(r);
        }
        let mut idx: Vec<u32> = m.idx.clone();
        let mut extra: Vec<(usize, B, u16)> = Vec::new();
        if part.is_some() {
            let mut dup: HashMap<(u32, u16), u32> = HashMap::new();
            for t in idx.chunks_mut(3) {
                let parts = [
                    vb[t[0] as usize].1,
                    vb[t[1] as usize].1,
                    vb[t[2] as usize].1,
                ];
                if parts[0] == parts[1] && parts[1] == parts[2] {
                    continue;
                }
                let tp = if parts[1] == parts[2] {
                    parts[1]
                } else {
                    parts[0]
                };
                let Some(bp) = binders.get(&tp).map(|_| ()).or_else(|| {
                    by_part.get(&tp).map(|tris| {
                        binders.insert(tp, Binder::new(tris.clone(), &node_pos, 0.15));
                    })
                }) else {
                    continue;
                };
                let _ = bp;
                for v in t.iter_mut() {
                    if vb[*v as usize].1 == tp {
                        continue;
                    }
                    let key = (*v, tp);
                    let nv = if let Some(&nv) = dup.get(&key) {
                        nv
                    } else {
                        let k = *v as usize;
                        let b = binders[&tp].bind(m.pos[k], m.nrm[k]).unwrap_or(vb[k].0);
                        let nv = (m.len() + extra.len()) as u32;
                        extra.push((k, b, tp));
                        dup.insert(key, nv);
                        nv
                    };
                    *v = nv;
                }
            }
        }
        let all = (0..m.len())
            .map(|k| (k, vb[k].0))
            .chain(extra.iter().map(|&(k, b, _)| (k, b)));
        for (k, (t, c, nl)) in all {
            out.bind_nodes.push(t);
            out.bind_coords.push(c);
            out.bind_normal.push(nl);
            out.bind_rest.push(rest_edges(&node_pos, t));
            out.positions.push((m.pos[k] + built.shift).to_array());
            out.normals.push(m.nrm[k].to_array());
            out.uvs.push(m.uv[k]);
            out.material.push(m.mat[k]);
            out.colors.push(color(m.mat[k], paint));
            out.ao.push(m.ao[k].clamp(0.0, 1.0));
            // Smooth shading from node normals only where the piece rides the
            // body's skin (not on wheels or frames).
            out.smooth.push(if own.is_none() { m.smooth[k] } else { 0 });
            out.rigid.push(0);
        }
        let count = out.positions.len() as u32 - first;
        out.indices.extend(idx.iter().map(|i| i + first));
        out.pieces.push((piece.name.clone(), first, count));
    }
    // Node normals for smooth shading.
    out.normal_tris = sk.skin.clone();
    let nn = node_pos.len();
    let mut count = vec![0u32; nn + 1];
    for t in &out.normal_tris {
        for &i in t {
            count[i as usize + 1] += 1;
        }
    }
    for i in 0..nn {
        count[i + 1] += count[i];
    }
    let mut fill = count.clone();
    let mut list = vec![0u32; *count.last().unwrap() as usize];
    for (ti, t) in out.normal_tris.iter().enumerate() {
        for &i in t {
            list[fill[i as usize] as usize] = ti as u32;
            fill[i as usize] += 1;
        }
    }
    out.node_tri_start = count;
    out.node_tri_list = list;
    out.node_rest_normal = node_normals(&out, &node_pos)
        .iter()
        .map(|n| n.to_array())
        .collect();
    damage_units(built, &sim, &node_pos, &pure, &mut out);
    Ok(out)
}
