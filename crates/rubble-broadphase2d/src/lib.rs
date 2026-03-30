//! 2D broadphase collision detection using an LBVH (Linear Bounding Volume Hierarchy)
//! built from Morton codes (Karras 2012). CPU reference implementation.

use glam::Vec2;
use rubble_math::Aabb2D;

// ---------------------------------------------------------------------------
// Morton codes (30-bit, 15 bits per axis)
// ---------------------------------------------------------------------------

/// Insert one zero bit between each of the lower 15 bits of `v`.
fn expand_bits_15(mut v: u32) -> u32 {
    v &= 0x7FFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    v
}

/// Compute 30-bit Morton code from normalized [0,1]^2 position (15 bits per axis).
///
/// Bits are interleaved X-Y with X in the most significant position.
pub fn morton_encode_2d(x: f32, y: f32) -> u32 {
    let x = (x.clamp(0.0, 1.0) * 32767.0) as u32;
    let y = (y.clamp(0.0, 1.0) * 32767.0) as u32;
    (expand_bits_15(x) << 1) | expand_bits_15(y)
}

// ---------------------------------------------------------------------------
// Scene AABB
// ---------------------------------------------------------------------------

/// Compute the bounding box that encloses all given 2D AABBs.
pub fn compute_scene_aabb(aabbs: &[Aabb2D]) -> Aabb2D {
    let mut scene_min = Vec2::splat(f32::MAX);
    let mut scene_max = Vec2::splat(f32::NEG_INFINITY);
    for aabb in aabbs {
        scene_min = scene_min.min(aabb.min_point());
        scene_max = scene_max.max(aabb.max_point());
    }
    Aabb2D::new(scene_min, scene_max)
}

// ---------------------------------------------------------------------------
// AABB helpers
// ---------------------------------------------------------------------------

#[inline]
fn aabb_overlap(a: &Aabb2D, b: &Aabb2D) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
}

#[inline]
fn aabb_union(a: &Aabb2D, b: &Aabb2D) -> Aabb2D {
    Aabb2D::new(
        Vec2::new(a.min.x.min(b.min.x), a.min.y.min(b.min.y)),
        Vec2::new(a.max.x.max(b.max.x), a.max.y.max(b.max.y)),
    )
}

// ---------------------------------------------------------------------------
// LBVH child pointer
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum LbvhChild {
    Internal(usize),
    Leaf(usize),
}

// ---------------------------------------------------------------------------
// Karras 2012 helpers
// ---------------------------------------------------------------------------

fn delta(sorted_codes: &[u32], i: usize, j: usize) -> i32 {
    let n = sorted_codes.len();
    if j >= n {
        return -1;
    }
    let xor = sorted_codes[i] ^ sorted_codes[j];
    if xor == 0 {
        32 + ((i as u32 ^ j as u32).leading_zeros() as i32)
    } else {
        xor.leading_zeros() as i32
    }
}

fn karras_node(sorted_codes: &[u32], i: usize) -> (usize, usize, usize) {
    let n = sorted_codes.len();

    let d_left = if i == 0 {
        -1
    } else {
        delta(sorted_codes, i, i - 1)
    };
    let d_right = if i + 1 >= n {
        -1
    } else {
        delta(sorted_codes, i, i + 1)
    };

    let d: i32 = if d_right > d_left { 1 } else { -1 };
    let delta_min = if d > 0 { d_left } else { d_right };

    let mut l_max: usize = 2;
    loop {
        let j = i as i64 + l_max as i64 * d as i64;
        if j < 0 || j >= n as i64 {
            break;
        }
        if delta(sorted_codes, i, j as usize) <= delta_min {
            break;
        }
        l_max *= 2;
    }

    let mut l: usize = 0;
    let mut t = l_max >> 1;
    while t >= 1 {
        let j = i as i64 + (l + t) as i64 * d as i64;
        if j >= 0 && j < n as i64 && delta(sorted_codes, i, j as usize) > delta_min {
            l += t;
        }
        t >>= 1;
    }

    let j_other = (i as i64 + l as i64 * d as i64) as usize;
    let range_left = i.min(j_other);
    let range_right = i.max(j_other);

    let delta_node = delta(sorted_codes, range_left, range_right);
    let mut s: usize = 0;
    let mut t = ((range_right - range_left + 1) as u64)
        .next_power_of_two() as usize
        / 2;
    if t == 0 {
        t = 1;
    }
    loop {
        let candidate = range_left + s + t;
        if candidate < range_right
            && delta(sorted_codes, range_left, candidate) > delta_node
        {
            s += t;
        }
        if t == 1 {
            break;
        }
        t = (t + 1) / 2;
    }

    (range_left, range_right, range_left + s)
}

// ---------------------------------------------------------------------------
// LBVH
// ---------------------------------------------------------------------------

/// Result of broadphase overlap detection.
pub struct BroadphaseResult {
    /// Overlapping body index pairs, with `a < b` for each `[a, b]`.
    pub pairs: Vec<[u32; 2]>,
}

/// A 2D Linear Bounding Volume Hierarchy built from Morton codes (Karras 2012).
pub struct Lbvh {
    /// Internal BVH nodes. For N leaves there are N-1 internal nodes.
    pub nodes: Vec<rubble_math::BvhNode>,
    /// Number of leaves (bodies).
    pub leaf_count: u32,
    /// Body indices sorted by Morton code.
    pub sorted_indices: Vec<u32>,
    /// Internal representation for traversal.
    internal_left: Vec<LbvhChild>,
    internal_right: Vec<LbvhChild>,
    internal_aabbs: Vec<Aabb2D>,
}

impl Lbvh {
    /// Build an LBVH from a slice of 2D body AABBs.
    pub fn build(aabbs: &[Aabb2D]) -> Self {
        let n = aabbs.len();
        if n == 0 {
            return Self {
                nodes: vec![],
                leaf_count: 0,
                sorted_indices: vec![],
                internal_left: vec![],
                internal_right: vec![],
                internal_aabbs: vec![],
            };
        }
        if n == 1 {
            return Self {
                nodes: vec![],
                leaf_count: 1,
                sorted_indices: vec![0],
                internal_left: vec![],
                internal_right: vec![],
                internal_aabbs: vec![],
            };
        }

        let scene = compute_scene_aabb(aabbs);
        let scene_min = scene.min_point();
        let scene_extent = scene.max_point() - scene_min;
        let inv_extent = Vec2::new(
            if scene_extent.x > 1e-10 {
                1.0 / scene_extent.x
            } else {
                0.0
            },
            if scene_extent.y > 1e-10 {
                1.0 / scene_extent.y
            } else {
                0.0
            },
        );

        let mut indexed_codes: Vec<(u32, u32)> = aabbs
            .iter()
            .enumerate()
            .map(|(i, aabb)| {
                let center = (aabb.min_point() + aabb.max_point()) * 0.5;
                let norm = (center - scene_min) * inv_extent;
                (morton_encode_2d(norm.x, norm.y), i as u32)
            })
            .collect();

        indexed_codes.sort_unstable_by_key(|&(code, idx)| (code, idx));

        let sorted_codes: Vec<u32> = indexed_codes.iter().map(|&(c, _)| c).collect();
        let sorted_indices: Vec<u32> = indexed_codes.iter().map(|&(_, i)| i).collect();

        let num_internal = n - 1;
        let mut internal_left = Vec::with_capacity(num_internal);
        let mut internal_right = Vec::with_capacity(num_internal);

        for i in 0..num_internal {
            let (range_left, range_right, split) = karras_node(&sorted_codes, i);
            let left = if split == range_left {
                LbvhChild::Leaf(split)
            } else {
                LbvhChild::Internal(split)
            };
            let right = if split + 1 == range_right {
                LbvhChild::Leaf(split + 1)
            } else {
                LbvhChild::Internal(split + 1)
            };
            internal_left.push(left);
            internal_right.push(right);
        }

        let leaf_aabbs: Vec<Aabb2D> = sorted_indices
            .iter()
            .map(|&idx| aabbs[idx as usize])
            .collect();

        let mut internal_aabbs = vec![Aabb2D::new(Vec2::ZERO, Vec2::ZERO); num_internal];

        fn compute_aabb_rec(
            node_idx: usize,
            left: &[LbvhChild],
            right: &[LbvhChild],
            internal_aabbs: &mut [Aabb2D],
            leaf_aabbs: &[Aabb2D],
        ) -> Aabb2D {
            let left_aabb = match left[node_idx] {
                LbvhChild::Leaf(i) => leaf_aabbs[i],
                LbvhChild::Internal(i) => {
                    compute_aabb_rec(i, left, right, internal_aabbs, leaf_aabbs)
                }
            };
            let right_aabb = match right[node_idx] {
                LbvhChild::Leaf(i) => leaf_aabbs[i],
                LbvhChild::Internal(i) => {
                    compute_aabb_rec(i, left, right, internal_aabbs, leaf_aabbs)
                }
            };
            let combined = aabb_union(&left_aabb, &right_aabb);
            internal_aabbs[node_idx] = combined;
            combined
        }

        compute_aabb_rec(
            0,
            &internal_left,
            &internal_right,
            &mut internal_aabbs,
            &leaf_aabbs,
        );

        let nodes: Vec<rubble_math::BvhNode> = (0..num_internal)
            .map(|i| {
                let aabb = &internal_aabbs[i];
                let min2 = aabb.min_point();
                let max2 = aabb.max_point();
                let left_idx = match internal_left[i] {
                    LbvhChild::Internal(idx) => idx as i32,
                    LbvhChild::Leaf(idx) => -(idx as i32 + 1),
                };
                let right_idx = match internal_right[i] {
                    LbvhChild::Internal(idx) => idx as i32,
                    LbvhChild::Leaf(idx) => -(idx as i32 + 1),
                };
                rubble_math::BvhNode::internal(
                    glam::Vec3::new(min2.x, min2.y, 0.0),
                    glam::Vec3::new(max2.x, max2.y, 0.0),
                    left_idx,
                    right_idx,
                )
            })
            .collect();

        Self {
            nodes,
            leaf_count: n as u32,
            sorted_indices,
            internal_left,
            internal_right,
            internal_aabbs,
        }
    }

    /// Find all overlapping AABB pairs by traversing the BVH.
    pub fn find_overlapping_pairs(&self, aabbs: &[Aabb2D]) -> BroadphaseResult {
        let n = self.leaf_count as usize;
        let mut pairs = Vec::new();

        if n <= 1 || self.internal_aabbs.is_empty() {
            return BroadphaseResult { pairs };
        }

        let leaf_aabbs: Vec<Aabb2D> = self
            .sorted_indices
            .iter()
            .map(|&idx| aabbs[idx as usize])
            .collect();

        let mut stack: Vec<LbvhChild> = Vec::with_capacity(64);

        for leaf_idx in 0..n {
            let body_i = self.sorted_indices[leaf_idx] as u32;
            let aabb_i = &leaf_aabbs[leaf_idx];

            stack.clear();
            stack.push(LbvhChild::Internal(0));

            while let Some(child) = stack.pop() {
                match child {
                    LbvhChild::Leaf(j) => {
                        if j != leaf_idx {
                            let body_j = self.sorted_indices[j] as u32;
                            if body_i < body_j && aabb_overlap(aabb_i, &leaf_aabbs[j]) {
                                pairs.push([body_i, body_j]);
                            }
                        }
                    }
                    LbvhChild::Internal(idx) => {
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

        BroadphaseResult { pairs }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    // -- Morton code tests --

    #[test]
    fn test_morton_2d_origin() {
        assert_eq!(morton_encode_2d(0.0, 0.0), 0);
    }

    #[test]
    fn test_morton_2d_all_ones() {
        assert_eq!(morton_encode_2d(1.0, 1.0), 0x3FFFFFFF);
    }

    #[test]
    fn test_morton_2d_x_only() {
        assert_eq!(morton_encode_2d(1.0, 0.0), 0x2AAAAAAA);
    }

    #[test]
    fn test_morton_2d_y_only() {
        assert_eq!(morton_encode_2d(0.0, 1.0), 0x15555555);
    }

    #[test]
    fn test_morton_2d_distinct() {
        let positions = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.5, 0.5),
        ];
        let codes: Vec<u32> = positions
            .iter()
            .map(|&(x, y)| morton_encode_2d(x, y))
            .collect();
        let mut unique = codes.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), codes.len());
    }

    // -- Overlap tests --

    fn make_circle_aabb(cx: f32, cy: f32, r: f32) -> Aabb2D {
        Aabb2D::new(Vec2::new(cx - r, cy - r), Vec2::new(cx + r, cy + r))
    }

    #[test]
    fn test_two_overlapping_circles() {
        let aabbs = vec![
            make_circle_aabb(0.0, 0.0, 1.0),
            make_circle_aabb(0.5, 0.0, 1.0),
        ];
        let lbvh = Lbvh::build(&aabbs);
        let result = lbvh.find_overlapping_pairs(&aabbs);
        assert_eq!(result.pairs.len(), 1);
        assert_eq!(result.pairs[0], [0, 1]);
    }

    #[test]
    fn test_two_separated() {
        let aabbs = vec![
            make_circle_aabb(0.0, 0.0, 1.0),
            make_circle_aabb(10.0, 0.0, 1.0),
        ];
        let lbvh = Lbvh::build(&aabbs);
        let result = lbvh.find_overlapping_pairs(&aabbs);
        assert_eq!(result.pairs.len(), 0);
    }

    #[test]
    fn test_brute_force_cross_validation_2d() {
        let mut rng_state: u64 = 67890;
        let mut next_f32 = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state & 0xFFFF) as f32 / 65535.0
        };

        let n = 50;
        let aabbs: Vec<Aabb2D> = (0..n)
            .map(|_| {
                let x = next_f32() * 20.0 - 10.0;
                let y = next_f32() * 20.0 - 10.0;
                let r = next_f32() * 1.5 + 0.5;
                make_circle_aabb(x, y, r)
            })
            .collect();

        // Brute force.
        let mut bf_pairs: Vec<[u32; 2]> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if aabb_overlap(&aabbs[i], &aabbs[j]) {
                    bf_pairs.push([i as u32, j as u32]);
                }
            }
        }
        bf_pairs.sort();

        let lbvh = Lbvh::build(&aabbs);
        let mut lbvh_pairs = lbvh.find_overlapping_pairs(&aabbs).pairs;
        lbvh_pairs.sort();

        assert_eq!(
            lbvh_pairs, bf_pairs,
            "LBVH ({}) vs brute force ({}) mismatch",
            lbvh_pairs.len(),
            bf_pairs.len()
        );
    }

    #[test]
    fn test_empty_input() {
        let aabbs: Vec<Aabb2D> = vec![];
        let lbvh = Lbvh::build(&aabbs);
        assert_eq!(lbvh.leaf_count, 0);
        assert!(lbvh.find_overlapping_pairs(&aabbs).pairs.is_empty());
    }

    #[test]
    fn test_single_body() {
        let aabbs = vec![make_circle_aabb(0.0, 0.0, 1.0)];
        let lbvh = Lbvh::build(&aabbs);
        assert_eq!(lbvh.leaf_count, 1);
        assert!(lbvh.find_overlapping_pairs(&aabbs).pairs.is_empty());
    }
}
