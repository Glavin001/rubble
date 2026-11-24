//! The sedan's B-pillar solid (split from `sedan.rs`).

use super::*;

/// numpy.arange(a, b, step) for floats.
/// The B-pillar: a closed box section down the body side at z0, from the
/// sill to the roof rail, behind the doors. Its outer face sits 72 mm in
/// from the skin (the doors' inner panels are 50 mm deep), reaches out to
/// the sill's inner face at the foot and stops under the roof rail; each
/// jamb face has a seal land, a step in 16 mm down; the inner face is
/// 132 mm in. 72 mm wide: the slot between the door cards.
pub(crate) fn b_pillar(tube: &Tube, z0: f64, side: f64) -> Mesh {
    // (v: 1 the sill, 2 the doors' bottom edge, 7 the roof rail)
    let (v0, v1) = (1.60, 6.95);
    let at = |v: f64, dz: f64, depth: f64| {
        // (in from the skin level with the ground, not along the normal:
        // a section this deep would fold behind the shoulder's bend)
        let p = tube.eval(z0 + dz, if side > 0.0 { v } else { tube.vm - v });
        p - DVec3::X * (side * depth)
    };
    // stations every 22 mm up the side
    let fine: Vec<f64> = linspace(v0, v1, 280);
    let mut len = vec![0.0];
    for k in 1..fine.len() {
        let d = (at(fine[k], 0.0, 0.0) - at(fine[k - 1], 0.0, 0.0)).length();
        len.push(len[k - 1] + d);
    }
    let total = len[len.len() - 1];
    let ns = (total / 0.022).round() as usize;
    let vs: Vec<f64> = (0..=ns).map(|i| interp(total * i as f64 / ns as f64, &len, &fine)).collect();
    // the section: (along the car, below the skin), outer face first
    let (hw, land, face, step, back) = (0.036, 0.029, 0.072, 0.088, 0.132);
    let ring = |v: f64| -> Vec<DVec3> {
        // the outer face meets the sill's inner face
        let up = smoothstep(1.95, v0, v);
        let o = face + (0.031 - face) * up;
        let s = step + (0.047 - step) * up;
        [
            (-hw, o),
            (-0.012, o),
            (0.012, o),
            (hw, o),
            (hw, s),
            (land, s + 0.002),
            (land, 0.5 * (s + back)),
            (land, back),
            (0.010, back),
            (-0.010, back),
            (-land, back),
            (-land, 0.5 * (s + back)),
            (-land, s + 0.002),
            (-hw, s),
        ]
        .iter()
        .map(|&(dz, d)| at(v, dz, d))
        .collect()
    };
    // (the head stops at the doors' top edge, under the roof rail and short
    // of the headliner)
    let rings: Vec<Vec<DVec3>> = vs.iter().map(|&v| ring(v)).collect();
    let nr = rings[0].len();
    // its own UV island (metres: round the section, up the pillar), the end
    // caps beside it
    let mut round = vec![0.0; nr + 1];
    for j in 0..nr {
        round[j + 1] = round[j] + (rings[2][(j + 1) % nr] - rings[2][j]).length();
    }
    let island = if side > 0.0 { 5_000 } else { 5_010 };
    let mut m = Mesh::new();
    for r in &rings {
        m.v.extend_from_slice(r);
    }
    let id = |i: usize, j: usize| (i * nr + j % nr) as u32;
    for i in 0..rings.len() - 1 {
        let (v0, v1) = (0.022 * i as f64, 0.022 * (i + 1) as f64);
        for j in 0..nr {
            m.q.push([id(i, j), id(i, j + 1), id(i + 1, j + 1), id(i + 1, j)]);
            m.quv.push([DVec2::new(round[j], v0), DVec2::new(round[j + 1], v0), DVec2::new(round[j + 1], v1), DVec2::new(round[j], v1)]);
            m.qm.push(Mat::Paint);
            m.qi.push(island);
        }
    }
    // end caps: the section cut into well shaped cells (the wide part as
    // three, the stem as two with a triangle each side of its foot)
    let section: [DVec2; 14] = {
        let (o, s) = (face, step);
        [(-hw, o), (-0.012, o), (0.012, o), (hw, o), (hw, s), (land, s + 0.002), (land, 0.5 * (s + back)), (land, back), (0.010, back), (-0.010, back), (-land, back), (-land, 0.5 * (s + back)), (-land, s + 0.002), (-hw, s)].map(|(a, b)| DVec2::new(a, b))
    };
    for (e, i) in [0, rings.len() - 1].into_iter().enumerate() {
        let uv = |j: usize| section[j] + DVec2::new(0.0, 0.1 * e as f64);
        for f in [[0, 1, 12, 13], [1, 2, 5, 12], [2, 3, 4, 5], [12, 5, 6, 11], [11, 6, 8, 9]] {
            let (mut q, mut u) = (f.map(|j| id(i, j)), f.map(uv));
            if i == 0 {
                q.reverse();
                u.reverse();
            }
            m.q.push(q);
            m.qm.push(Mat::Paint);
            m.quv.push(u);
            m.qi.push(island + 1);
        }
        for f in [[6, 7, 8], [11, 9, 10]] {
            let (mut q, mut u) = (f.map(|j| id(i, j)), f.map(uv));
            if i == 0 {
                q.reverse();
                u.reverse();
            }
            m.t.push(q);
            m.tm.push(Mat::Paint);
            m.tuv.push(u);
            m.ti.push(island + 1);
        }
    }
    let mut m = orient_outward(m);
    // not skin: the rigger and the collision skin leave it alone
    m.surf = Some(vec![crate::mesh::Surf::Edge; m.v.len()]);
    m
}
