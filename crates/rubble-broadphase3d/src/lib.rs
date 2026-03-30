//! 3D broadphase collision detection using an LBVH (Linear Bounding Volume Hierarchy)
//! built from Morton codes (Karras 2012). CPU reference implementation.

use glam::Vec3;
use rubble_math::Aabb3D;

// ---------------------------------------------------------------------------
// Morton codes (30-bit, 10 bits per axis)
// ---------------------------------------------------------------------------

/// Insert two zero bits between each of the lower 10 bits of `v`.
fn expand_bits_10(mut v: u32) -> u32 {
    v &= 0x3FF;
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    v
}

/// Compute 30-bit Morton code from normalized [0,1]^3 position (10 bits per axis).
///
/// Bits are interleaved X-Y-Z with X in the most significant position.
pub fn morton_encode_3d(x: f32, y: f32, z: f32) -> u32 {
    let x = (x.clamp(0.0, 1.0) * 1023.0) as u32;
    let y = (y.clamp(0.0, 1.0) * 1023.0) as u32;
    let z = (z.clamp(0.0, 1.0) * 1023.0) as u32;
    (expand_bits_10(x) << 2) | (expand_bits_10(y) << 1) | expand_bits_10(z)
}

// ---------------------------------------------------------------------------
// Scene AABB
// ---------------------------------------------------------------------------

/// Compute the bounding box that encloses all given AABBs.
pub fn compute_scene_aabb(aabbs: &[Aabb3D]) -> Aabb3D {
    let mut scene_min = Vec3::splat(f32::MAX);
    let mut scene_max = Vec3::splat(f32::NEG_INFINITY);
    for aabb in aabbs {
        scene_min = scene_min.min(aabb.min_point());
        scene_max = scene_max.max(aabb.max_point());
    }
    Aabb3D::new(scene_min, scene_max)
}

// ---------------------------------------------------------------------------
// AABB helpers
// ---------------------------------------------------------------------------

#[inline]
fn aabb_overlap(a: &Aabb3D, b: &Aabb3D) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
        && a.min.z <= b.max.z
        && a.max.z >= b.min.z
}

#[inline]
fn aabb_union(a: &Aabb3D, b: &Aabb3D) -> Aabb3D {
    Aabb3D::new(
        Vec3::new(
            a.min.x.min(b.min.x),
            a.min.y.min(b.min.y),
            a.min.z.min(b.min.z),
        ),
        Vec3::new(
            a.max.x.max(b.max.x),
            a.max.y.max(b.max.y),
            a.max.z.max(b.max.z),
        ),
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

/// Longest common prefix length between sorted Morton codes at positions i and j.
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

/// For internal node `i`, determine the range [left, right] and split position.
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

    // Upper bound for range length.
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

    // Binary search for exact range length.
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

    // Find split position.
    let delta_node = delta(sorted_codes, range_left, range_right);
    let mut s: usize = 0;
    let mut t = ((range_right - range_left + 1) as u64).next_power_of_two() as usize / 2;
    if t == 0 {
        t = 1;
    }
    loop {
        let candidate = range_left + s + t;
        if candidate < range_right && delta(sorted_codes, range_left, candidate) > delta_node {
            s += t;
        }
        if t == 1 {
            break;
        }
        t = t.div_ceil(2);
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

/// A Linear Bounding Volume Hierarchy built from Morton codes (Karras 2012).
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
    internal_aabbs: Vec<Aabb3D>,
}

impl Lbvh {
    /// Build an LBVH from a slice of body AABBs.
    pub fn build(aabbs: &[Aabb3D]) -> Self {
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

        // 1. Scene AABB and normalisation.
        let scene = compute_scene_aabb(aabbs);
        let scene_min = scene.min_point();
        let scene_extent = scene.max_point() - scene_min;
        let inv_extent = Vec3::new(
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
            if scene_extent.z > 1e-10 {
                1.0 / scene_extent.z
            } else {
                0.0
            },
        );

        // 2. Morton codes for centroids.
        let mut indexed_codes: Vec<(u32, u32)> = aabbs
            .iter()
            .enumerate()
            .map(|(i, aabb)| {
                let center = (aabb.min_point() + aabb.max_point()) * 0.5;
                let norm = (center - scene_min) * inv_extent;
                (morton_encode_3d(norm.x, norm.y, norm.z), i as u32)
            })
            .collect();

        // 3. Sort by Morton code.
        indexed_codes.sort_unstable_by_key(|&(code, idx)| (code, idx));

        let sorted_codes: Vec<u32> = indexed_codes.iter().map(|&(c, _)| c).collect();
        let sorted_indices: Vec<u32> = indexed_codes.iter().map(|&(_, i)| i).collect();

        // 4. Build binary radix tree (Karras 2012).
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

        // 5. Leaf AABBs in sorted order.
        let leaf_aabbs: Vec<Aabb3D> = sorted_indices
            .iter()
            .map(|&idx| aabbs[idx as usize])
            .collect();

        // 6. Compute internal AABBs bottom-up.
        let mut internal_aabbs = vec![Aabb3D::new(Vec3::ZERO, Vec3::ZERO); num_internal];

        fn compute_aabb_rec(
            node_idx: usize,
            left: &[LbvhChild],
            right: &[LbvhChild],
            internal_aabbs: &mut [Aabb3D],
            leaf_aabbs: &[Aabb3D],
        ) -> Aabb3D {
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

        // 7. Build BvhNode array for public API.
        let nodes: Vec<rubble_math::BvhNode> = (0..num_internal)
            .map(|i| {
                let aabb = &internal_aabbs[i];
                let left_idx = match internal_left[i] {
                    LbvhChild::Internal(idx) => idx as i32,
                    LbvhChild::Leaf(idx) => -(idx as i32 + 1),
                };
                let right_idx = match internal_right[i] {
                    LbvhChild::Internal(idx) => idx as i32,
                    LbvhChild::Leaf(idx) => -(idx as i32 + 1),
                };
                rubble_math::BvhNode::internal(
                    aabb.min_point(),
                    aabb.max_point(),
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
    ///
    /// For each leaf, descend the tree with a stack. Only pairs where
    /// `body_a < body_b` are emitted to avoid duplicates.
    pub fn find_overlapping_pairs(&self, aabbs: &[Aabb3D]) -> BroadphaseResult {
        let n = self.leaf_count as usize;
        let mut pairs = Vec::new();

        if n <= 1 || self.internal_aabbs.is_empty() {
            return BroadphaseResult { pairs };
        }

        // Precompute leaf AABBs in sorted order.
        let leaf_aabbs: Vec<Aabb3D> = self
            .sorted_indices
            .iter()
            .map(|&idx| aabbs[idx as usize])
            .collect();

        let mut stack: Vec<LbvhChild> = Vec::with_capacity(64);

        for leaf_idx in 0..n {
            let body_i = self.sorted_indices[leaf_idx];
            let aabb_i = &leaf_aabbs[leaf_idx];

            stack.clear();
            stack.push(LbvhChild::Internal(0));

            while let Some(child) = stack.pop() {
                match child {
                    LbvhChild::Leaf(j) => {
                        if j != leaf_idx {
                            let body_j = self.sorted_indices[j];
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
// Plane broadphase
// ---------------------------------------------------------------------------

/// Test each body's AABB against each plane's half-space.
///
/// Returns `(plane_index, body_index)` for each body whose AABB straddles the plane.
pub fn find_plane_pairs(planes: &[rubble_shapes3d::Plane], aabbs: &[Aabb3D]) -> Vec<(usize, u32)> {
    let mut results = Vec::new();
    for (pi, plane) in planes.iter().enumerate() {
        let n = plane.normal;
        let d = plane.distance;
        for (bi, aabb) in aabbs.iter().enumerate() {
            let min_pt = aabb.min_point();
            let max_pt = aabb.max_point();

            let near = Vec3::new(
                if n.x >= 0.0 { min_pt.x } else { max_pt.x },
                if n.y >= 0.0 { min_pt.y } else { max_pt.y },
                if n.z >= 0.0 { min_pt.z } else { max_pt.z },
            );
            let far = Vec3::new(
                if n.x >= 0.0 { max_pt.x } else { min_pt.x },
                if n.y >= 0.0 { max_pt.y } else { min_pt.y },
                if n.z >= 0.0 { max_pt.z } else { min_pt.z },
            );

            let dist_near = n.dot(near) - d;
            let dist_far = n.dot(far) - d;

            if dist_near <= 0.0 && dist_far >= 0.0 {
                results.push((pi, bi as u32));
            }
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    // -- Morton code tests --

    #[test]
    fn test_morton_code_origin() {
        assert_eq!(morton_encode_3d(0.0, 0.0, 0.0), 0);
    }

    #[test]
    fn test_morton_code_all_ones() {
        assert_eq!(morton_encode_3d(1.0, 1.0, 1.0), 0x3FFFFFFF);
    }

    #[test]
    fn test_morton_code_x_only() {
        assert_eq!(morton_encode_3d(1.0, 0.0, 0.0), 0x24924924);
    }

    #[test]
    fn test_morton_code_y_only() {
        assert_eq!(morton_encode_3d(0.0, 1.0, 0.0), 0x12492492);
    }

    #[test]
    fn test_morton_code_z_only() {
        assert_eq!(morton_encode_3d(0.0, 0.0, 1.0), 0x09249249);
    }

    #[test]
    fn test_morton_code_half() {
        let v = (0.5_f32 * 1023.0) as u32;
        let expected = (expand_bits_10(v) << 2) | (expand_bits_10(v) << 1) | expand_bits_10(v);
        assert_eq!(morton_encode_3d(0.5, 0.5, 0.5), expected);
    }

    #[test]
    fn test_morton_known_positions_distinct() {
        let positions = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.25, 0.25, 0.25),
            (0.75, 0.75, 0.75),
            (1.0, 1.0, 1.0),
        ];
        let codes: Vec<u32> = positions
            .iter()
            .map(|&(x, y, z)| morton_encode_3d(x, y, z))
            .collect();
        let mut unique = codes.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), codes.len());
    }

    // -- Overlap tests --

    fn make_sphere_aabb(cx: f32, cy: f32, cz: f32, r: f32) -> Aabb3D {
        Aabb3D::new(
            Vec3::new(cx - r, cy - r, cz - r),
            Vec3::new(cx + r, cy + r, cz + r),
        )
    }

    #[test]
    fn test_two_overlapping() {
        let aabbs = vec![
            make_sphere_aabb(0.0, 0.0, 0.0, 1.0),
            make_sphere_aabb(0.5, 0.0, 0.0, 1.0),
        ];
        let lbvh = Lbvh::build(&aabbs);
        let result = lbvh.find_overlapping_pairs(&aabbs);
        assert_eq!(result.pairs.len(), 1);
        assert_eq!(result.pairs[0], [0, 1]);
    }

    #[test]
    fn test_two_separated() {
        let aabbs = vec![
            make_sphere_aabb(0.0, 0.0, 0.0, 1.0),
            make_sphere_aabb(10.0, 0.0, 0.0, 1.0),
        ];
        let lbvh = Lbvh::build(&aabbs);
        let result = lbvh.find_overlapping_pairs(&aabbs);
        assert_eq!(result.pairs.len(), 0);
    }

    #[test]
    fn test_no_self_pairs_no_duplicates() {
        let aabbs: Vec<Aabb3D> = (0..10)
            .map(|i| {
                let x = i as f32 * 0.5;
                make_sphere_aabb(x, 0.0, 0.0, 1.0)
            })
            .collect();
        let lbvh = Lbvh::build(&aabbs);
        let result = lbvh.find_overlapping_pairs(&aabbs);
        for pair in &result.pairs {
            assert_ne!(pair[0], pair[1], "self-pair detected");
            assert!(pair[0] < pair[1], "pair not canonically ordered");
        }
        let mut sorted = result.pairs.clone();
        sorted.sort();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), before, "duplicate pairs detected");
    }

    #[test]
    fn test_brute_force_cross_validation() {
        let mut rng_state: u64 = 12345;
        let mut next_f32 = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state & 0xFFFF) as f32 / 65535.0
        };

        let n = 100;
        let aabbs: Vec<Aabb3D> = (0..n)
            .map(|_| {
                let x = next_f32() * 20.0 - 10.0;
                let y = next_f32() * 20.0 - 10.0;
                let z = next_f32() * 20.0 - 10.0;
                let r = next_f32() * 1.5 + 0.5;
                make_sphere_aabb(x, y, z, r)
            })
            .collect();

        // Brute force O(N^2).
        let mut bf_pairs: Vec<[u32; 2]> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if aabb_overlap(&aabbs[i], &aabbs[j]) {
                    bf_pairs.push([i as u32, j as u32]);
                }
            }
        }
        bf_pairs.sort();

        // LBVH.
        let lbvh = Lbvh::build(&aabbs);
        let mut lbvh_pairs = lbvh.find_overlapping_pairs(&aabbs).pairs;
        lbvh_pairs.sort();

        assert_eq!(
            lbvh_pairs,
            bf_pairs,
            "LBVH ({}) vs brute force ({}) mismatch",
            lbvh_pairs.len(),
            bf_pairs.len()
        );
    }

    // -- Plane broadphase tests --

    #[test]
    fn test_plane_body_above_no_pair() {
        let plane = rubble_shapes3d::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        };
        let aabbs = vec![Aabb3D::new(
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 3.0, 1.0),
        )];
        let pairs = find_plane_pairs(&[plane], &aabbs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_plane_body_intersecting() {
        let plane = rubble_shapes3d::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        };
        let aabbs = vec![Aabb3D::new(
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
        )];
        let pairs = find_plane_pairs(&[plane], &aabbs);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 0));
    }

    // -- Edge cases --

    #[test]
    fn test_empty_input() {
        let aabbs: Vec<Aabb3D> = vec![];
        let lbvh = Lbvh::build(&aabbs);
        assert_eq!(lbvh.leaf_count, 0);
        assert!(lbvh.find_overlapping_pairs(&aabbs).pairs.is_empty());
    }

    #[test]
    fn test_single_body() {
        let aabbs = vec![make_sphere_aabb(0.0, 0.0, 0.0, 1.0)];
        let lbvh = Lbvh::build(&aabbs);
        assert_eq!(lbvh.leaf_count, 1);
        assert!(lbvh.find_overlapping_pairs(&aabbs).pairs.is_empty());
    }

    #[test]
    fn test_identical_positions() {
        // All bodies at the same position produce degenerate (identical) Morton codes.
        // The LBVH must still build correctly and find all N*(N-1)/2 overlapping pairs.
        let n = 5;
        let aabbs: Vec<Aabb3D> = (0..n)
            .map(|_| make_sphere_aabb(0.0, 0.0, 0.0, 1.0))
            .collect();
        let lbvh = Lbvh::build(&aabbs);
        let result = lbvh.find_overlapping_pairs(&aabbs);

        let expected_pair_count = n * (n - 1) / 2;
        assert_eq!(
            result.pairs.len(),
            expected_pair_count,
            "Expected {} pairs for {} identical bodies, got {}",
            expected_pair_count,
            n,
            result.pairs.len()
        );

        // Verify all pairs are canonical (a < b) and unique.
        for pair in &result.pairs {
            assert!(
                pair[0] < pair[1],
                "pair not canonically ordered: {:?}",
                pair
            );
        }
        let mut sorted = result.pairs.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            expected_pair_count,
            "duplicate pairs detected"
        );
    }
}
