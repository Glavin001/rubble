//! CPU-side 2D LBVH using Karras 2012 algorithm with 30-bit Morton codes.

use glam::Vec2;
use rubble_math::Aabb2D;

// ---------------------------------------------------------------------------
// Morton codes (30-bit, 15 bits per axis for 2D)
// ---------------------------------------------------------------------------

fn expand_bits_15(mut v: u32) -> u32 {
    v &= 0x7FFF;
    v = (v | (v << 16)) & 0x0000FFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    v
}

pub fn morton_encode_2d(x: f32, y: f32) -> u32 {
    let x = (x.clamp(0.0, 1.0) * 32767.0) as u32;
    let y = (y.clamp(0.0, 1.0) * 32767.0) as u32;
    (expand_bits_15(x) << 1) | expand_bits_15(y)
}

// ---------------------------------------------------------------------------
// LBVH internals
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Child {
    Internal(usize),
    Leaf(usize),
}

fn delta(codes: &[u32], i: usize, j: usize) -> i32 {
    if j >= codes.len() {
        return -1;
    }
    let xor = codes[i] ^ codes[j];
    if xor == 0 {
        32 + ((i as u32 ^ j as u32).leading_zeros() as i32)
    } else {
        xor.leading_zeros() as i32
    }
}

fn karras_node(codes: &[u32], i: usize) -> (usize, usize, usize) {
    let n = codes.len();
    let d_left = if i == 0 { -1 } else { delta(codes, i, i - 1) };
    let d_right = if i + 1 >= n {
        -1
    } else {
        delta(codes, i, i + 1)
    };
    let d: i32 = if d_right > d_left { 1 } else { -1 };
    let delta_min = if d > 0 { d_left } else { d_right };

    let mut l_max: usize = 2;
    loop {
        let j = i as i64 + l_max as i64 * d as i64;
        if j < 0 || j >= n as i64 || delta(codes, i, j as usize) <= delta_min {
            break;
        }
        l_max *= 2;
    }

    let mut l: usize = 0;
    let mut t = l_max >> 1;
    while t >= 1 {
        let j = i as i64 + (l + t) as i64 * d as i64;
        if j >= 0 && j < n as i64 && delta(codes, i, j as usize) > delta_min {
            l += t;
        }
        t >>= 1;
    }

    let j_other = (i as i64 + l as i64 * d as i64) as usize;
    let range_left = i.min(j_other);
    let range_right = i.max(j_other);

    let delta_node = delta(codes, range_left, range_right);
    let mut s: usize = 0;
    let mut t = ((range_right - range_left + 1) as u64).next_power_of_two() as usize / 2;
    if t == 0 {
        t = 1;
    }
    loop {
        let candidate = range_left + s + t;
        if candidate < range_right && delta(codes, range_left, candidate) > delta_node {
            s += t;
        }
        if t == 1 {
            break;
        }
        t = t.div_ceil(2);
    }

    (range_left, range_right, range_left + s)
}

#[inline]
fn aabb_overlap(a: &Aabb2D, b: &Aabb2D) -> bool {
    a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y
}

fn aabb_union(a: &Aabb2D, b: &Aabb2D) -> Aabb2D {
    Aabb2D::new(
        a.min_point().min(b.min_point()),
        a.max_point().max(b.max_point()),
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct Lbvh2D {
    sorted_indices: Vec<u32>,
    internal_left: Vec<Child>,
    internal_right: Vec<Child>,
    internal_aabbs: Vec<Aabb2D>,
    #[allow(dead_code)]
    leaf_aabbs: Vec<Aabb2D>,
    leaf_count: usize,
}

impl Lbvh2D {
    pub fn build(aabbs: &[Aabb2D]) -> Self {
        let n = aabbs.len();
        if n <= 1 {
            return Self {
                sorted_indices: if n == 1 { vec![0] } else { vec![] },
                internal_left: vec![],
                internal_right: vec![],
                internal_aabbs: vec![],
                leaf_aabbs: aabbs.to_vec(),
                leaf_count: n,
            };
        }

        let mut scene_min = Vec2::splat(f32::MAX);
        let mut scene_max = Vec2::splat(f32::NEG_INFINITY);
        for aabb in aabbs {
            scene_min = scene_min.min(aabb.min_point());
            scene_max = scene_max.max(aabb.max_point());
        }
        let extent = scene_max - scene_min;
        let inv_extent = Vec2::new(
            if extent.x > 1e-10 {
                1.0 / extent.x
            } else {
                0.0
            },
            if extent.y > 1e-10 {
                1.0 / extent.y
            } else {
                0.0
            },
        );

        let mut indexed: Vec<(u32, u32)> = aabbs
            .iter()
            .enumerate()
            .map(|(i, aabb)| {
                let center = (aabb.min_point() + aabb.max_point()) * 0.5;
                let norm = (center - scene_min) * inv_extent;
                (morton_encode_2d(norm.x, norm.y), i as u32)
            })
            .collect();
        indexed.sort_unstable_by_key(|&(code, idx)| (code, idx));

        let codes: Vec<u32> = indexed.iter().map(|&(c, _)| c).collect();
        let sorted_indices: Vec<u32> = indexed.iter().map(|&(_, i)| i).collect();

        let num_internal = n - 1;
        let mut internal_left = Vec::with_capacity(num_internal);
        let mut internal_right = Vec::with_capacity(num_internal);

        for i in 0..num_internal {
            let (range_left, range_right, split) = karras_node(&codes, i);
            let left = if split == range_left {
                Child::Leaf(split)
            } else {
                Child::Internal(split)
            };
            let right = if split + 1 == range_right {
                Child::Leaf(split + 1)
            } else {
                Child::Internal(split + 1)
            };
            internal_left.push(left);
            internal_right.push(right);
        }

        let leaf_aabbs: Vec<Aabb2D> = sorted_indices
            .iter()
            .map(|&idx| aabbs[idx as usize])
            .collect();

        let mut internal_aabbs = vec![Aabb2D::new(Vec2::ZERO, Vec2::ZERO); num_internal];
        fn refit(
            idx: usize,
            left: &[Child],
            right: &[Child],
            internal: &mut [Aabb2D],
            leaves: &[Aabb2D],
        ) -> Aabb2D {
            let la = match left[idx] {
                Child::Leaf(i) => leaves[i],
                Child::Internal(i) => refit(i, left, right, internal, leaves),
            };
            let ra = match right[idx] {
                Child::Leaf(i) => leaves[i],
                Child::Internal(i) => refit(i, left, right, internal, leaves),
            };
            let combined = aabb_union(&la, &ra);
            internal[idx] = combined;
            combined
        }
        refit(
            0,
            &internal_left,
            &internal_right,
            &mut internal_aabbs,
            &leaf_aabbs,
        );

        Self {
            sorted_indices,
            internal_left,
            internal_right,
            internal_aabbs,
            leaf_aabbs,
            leaf_count: n,
        }
    }

    pub fn find_overlapping_pairs(&self, aabbs: &[Aabb2D]) -> Vec<[u32; 2]> {
        let n = self.leaf_count;
        if n <= 1 || self.internal_aabbs.is_empty() {
            return Vec::new();
        }

        let leaf_aabbs: Vec<Aabb2D> = self
            .sorted_indices
            .iter()
            .map(|&idx| aabbs[idx as usize])
            .collect();

        let mut pairs = Vec::new();
        let mut stack: Vec<Child> = Vec::with_capacity(64);

        for leaf_idx in 0..n {
            let body_i = self.sorted_indices[leaf_idx];
            let aabb_i = &leaf_aabbs[leaf_idx];

            stack.clear();
            stack.push(Child::Internal(0));

            while let Some(child) = stack.pop() {
                match child {
                    Child::Leaf(j) => {
                        if j != leaf_idx {
                            let body_j = self.sorted_indices[j];
                            if body_i < body_j && aabb_overlap(aabb_i, &leaf_aabbs[j]) {
                                pairs.push([body_i, body_j]);
                            }
                        }
                    }
                    Child::Internal(idx) => {
                        if aabb_overlap(aabb_i, &self.internal_aabbs[idx]) {
                            stack.push(self.internal_left[idx]);
                            stack.push(self.internal_right[idx]);
                        }
                    }
                }
            }
        }

        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle_aabb(cx: f32, cy: f32, r: f32) -> Aabb2D {
        Aabb2D::new(Vec2::new(cx - r, cy - r), Vec2::new(cx + r, cy + r))
    }

    #[test]
    fn test_morton_2d_origin() {
        assert_eq!(morton_encode_2d(0.0, 0.0), 0);
    }

    #[test]
    fn test_morton_2d_max() {
        assert_eq!(morton_encode_2d(1.0, 1.0), 0x3FFFFFFF);
    }

    #[test]
    fn test_two_overlapping_2d() {
        let aabbs = vec![circle_aabb(0.0, 0.0, 1.0), circle_aabb(0.5, 0.0, 1.0)];
        let lbvh = Lbvh2D::build(&aabbs);
        let pairs = lbvh.find_overlapping_pairs(&aabbs);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], [0, 1]);
    }

    #[test]
    fn test_two_separated_2d() {
        let aabbs = vec![circle_aabb(0.0, 0.0, 1.0), circle_aabb(10.0, 0.0, 1.0)];
        let lbvh = Lbvh2D::build(&aabbs);
        let pairs = lbvh.find_overlapping_pairs(&aabbs);
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_brute_force_cross_validation_2d() {
        let mut rng: u64 = 54321;
        let mut next_f32 = || -> f32 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng & 0xFFFF) as f32 / 65535.0
        };

        let n = 100;
        let aabbs: Vec<Aabb2D> = (0..n)
            .map(|_| {
                let x = next_f32() * 20.0 - 10.0;
                let y = next_f32() * 20.0 - 10.0;
                let r = next_f32() * 1.5 + 0.5;
                circle_aabb(x, y, r)
            })
            .collect();

        let mut bf: Vec<[u32; 2]> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if aabb_overlap(&aabbs[i], &aabbs[j]) {
                    bf.push([i as u32, j as u32]);
                }
            }
        }
        bf.sort();

        let lbvh = Lbvh2D::build(&aabbs);
        let mut lbvh_pairs = lbvh.find_overlapping_pairs(&aabbs);
        lbvh_pairs.sort();

        assert_eq!(lbvh_pairs, bf);
    }

    #[test]
    fn test_empty_2d() {
        let lbvh = Lbvh2D::build(&[]);
        assert!(lbvh.find_overlapping_pairs(&[]).is_empty());
    }
}
