//! Hull cooking as the scenes use it (formerly fracture/convex.test.mjs; the
//! cooker, packer and arena themselves are covered by engine-schema).

mod common;
use common::*;
use engine_fracture::math::*;
use engine_fracture::scene::*;

fn prism(sides: usize, radius: f64, height: f64) -> (Vec<[f64; 3]>, Vec<u32>, Vec<u32>) {
    use std::f64::consts::PI;
    let mut v = Vec::new();
    for y in [-height / 2.0, height / 2.0] {
        for i in 0..sides {
            let t = i as f64 * 2.0 * PI / sides as f64;
            v.push([radius * t.cos(), y, radius * t.sin()]);
        }
    }
    let n = sides as u32;
    let mut faces: Vec<Vec<u32>> = vec![(0..n).collect(), (n..2 * n).collect()];
    for i in 0..n {
        faces.push(vec![i, (i + 1) % n, (i + 1) % n + n, i + n]);
    }
    (v, faces.iter().map(|f| f.len() as u32).collect(), faces.concat())
}

#[test]
fn offset_compound_children_are_recentred_with_tight_bounds() {
    let mut arena = hull_arena(None).unwrap();
    let (v, s, i) = prism(8, 1.0, 2.0);
    let moved: Vec<_> = v.iter().map(|p| add(*p, [50.0, 0.0, 0.0])).collect();
    let (base, radius, local_pos) = append_hull(&mut arena, &moved, &s, &i, 0.0).unwrap();
    assert!(radius < 2.0, "bound radius stays local to the child");
    assert!((local_pos[0] - 50.0).abs() < 1e-5);
    let c = Collider::hull(0, 1.0, base, 1.0, radius, local_pos);
    let data = arena.records().to_vec();
    for k in 0..20 {
        let t = k as f64 * 0.1;
        let (p, q) = ([1.0, 2.0, 3.0], [0.0, t.sin(), 0.0, t.cos()]);
        let (lo, hi) = collider_bounds(&data, &c, p, q);
        for x in &moved {
            let w = add(p, rotate(q, *x));
            assert!((0..3).all(|k| w[k] >= lo[k] - 1e-4 && w[k] <= hi[k] + 1e-4));
        }
        assert!((0..3).all(|k| hi[k] - lo[k] < 3.1));
        assert_eq!(collider_vertices(&data, &c, p, q).len(), 16);
    }
}

#[test]
fn convexity_is_judged_at_the_f32_precision_of_the_input_frame() {
    let cube = |center: [f64; 3], half: f64, lift: f64| {
        let mut v: Vec<[f64; 3]> = (0..8).map(|i| [0, 1, 2].map(|k| center[k] + if (i >> k) & 1 != 0 { half } else { -half })).collect();
        v.push([center[0], center[1] + half + lift, center[2]]);
        v
    };
    let faces = engine_fracture::cuboid::BOX_FACES;
    let sizes = [4u32; 6];
    let indices: Vec<u32> = faces.iter().flatten().copied().collect();
    let cook = |v: &[[f64; 3]], scale: f64| {
        let mut arena = hull_arena(None).unwrap();
        append_hull(&mut arena, v, &sizes, &indices, scale).map(|_| ())
    };
    assert!(cook(&cube([6.0, 0.0, 0.0], 0.25, 3e-6), 0.0).is_ok());
    assert!(cook(&cube([6.0, 0.0, 0.0], 0.25, 1e-3), 0.0).unwrap_err().to_string().contains("non-convex"));
    let small = cube([0.0; 3], 0.25, 3e-6);
    assert!(cook(&small, 0.0).unwrap_err().to_string().contains("non-convex"));
    // Data stored as f32 in a 6 m frame, cooked after moving to the origin.
    assert!(cook(&small, 6.25).is_ok());
}

#[test]
fn filter_words_match_the_collision_filter_encoding() {
    let mut c = Collider::new(0, SHAPE_BOX, 1.0, [1.0; 4]);
    c.material = 3;
    assert_eq!(c.filter_words(), (0, 3));
    c.material = 5;
    c.filter = Some(Filter { categories: 1, mask: 2, group: -3 });
    assert_eq!(c.filter_words(), (1 | 2 << 16, 5 | 0xfffd << 16));
    assert!(c.has_filter());
}

/// Mirrors hullSupportIndex (physics/convex.slang). Some cooked house hulls
/// keep a redundant vertex on no face loop as vertex 0; a climb starting there
/// never moved and missed the ground by up to 15 cm, so plates sank.
#[test]
fn hill_climb_supports_match_brute_force_on_every_certified_house_hull() {
    let dirs: Vec<[f64; 3]> = (0..64)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / 64.0;
            let r = (1.0 - y * y).sqrt();
            let t = i as f64 * 2.399963;
            [r * t.cos(), y, r * t.sin()]
        })
        .collect();
    let mut isolated_start = 0;
    for level in [1u8, 2, 3] {
        let s = engine_fracture::build_scene(house(), level, None).unwrap();
        let h = &s.hull_data;
        let mut seen = std::collections::HashSet::new();
        for c in s.colliders.iter().filter(|c| c.shape == SHAPE_CONVEX) {
            let base = c.hull_base() as usize;
            if !seen.insert(base) {
                continue;
            }
            let head = h[base];
            let (nv, closed) = ((head[0] & 255) as usize, head[0] >> 24 & 1 != 0);
            if !closed {
                continue;
            }
            let word = |w: usize| h[base + w / 4][w % 4];
            let byte = |w: usize, b: usize| (word(w + b / 4) >> ((b % 4) * 8)) & 255;
            let vertex = |i: usize| {
                let r = h[base + 4 + i];
                [f32::from_bits(r[0]) as f64, f32::from_bits(r[1]) as f64, f32::from_bits(r[2]) as f64]
            };
            let edge = |e: usize| {
                let x = word(head[2] as usize + e);
                ((x & 255) as usize, (x >> 8 & 255) as usize)
            };
            let adjacency = |i: usize| {
                let x = h[base + 4 + i][3];
                let (start, deg) = ((x & 0xffff) as usize, (x >> 16) as usize);
                (0..deg).map(|j| byte(head[3] as usize, start + j) as usize).collect::<Vec<_>>()
            };
            if adjacency(0).is_empty() {
                isolated_start += 1;
            }
            for d in &dirs {
                let mut cur = edge(0).0;
                let mut cur_d = dot(vertex(cur), *d);
                for _ in 0..nv {
                    let (mut next, mut next_d) = (cur, cur_d);
                    for e in adjacency(cur) {
                        let (a, b) = edge(e);
                        let o = if a == cur { b } else { a };
                        let od = dot(vertex(o), *d);
                        if od > next_d {
                            next_d = od;
                            next = o;
                        }
                    }
                    if next == cur {
                        break;
                    }
                    cur = next;
                    cur_d = next_d;
                }
                let best = (0..nv).map(|i| dot(vertex(i), *d)).fold(f64::NEG_INFINITY, f64::max);
                assert!(best - cur_d <= 1e-6, "level {level} hull {base}: climb stopped {} short", best - cur_d);
            }
        }
    }
    assert!(isolated_start > 0, "the asset still exercises hulls whose vertex 0 is on no face");
}
