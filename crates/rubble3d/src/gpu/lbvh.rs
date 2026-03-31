//! CPU-side LBVH (Linear Bounding Volume Hierarchy) using Karras 2012 algorithm.
//!
//! Computes Morton codes, sorts, builds binary radix tree, refits AABBs, and
//! traverses to find overlapping pairs. Used by the GPU pipeline between
//! AABB compute and narrowphase passes.

use glam::Vec3;
use rubble_math::Aabb3D;

// ---------------------------------------------------------------------------
// Morton codes (30-bit, 10 bits per axis)
// ---------------------------------------------------------------------------

fn expand_bits_10(mut v: u32) -> u32 {
    v &= 0x3FF;
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    v
}

pub fn morton_encode_3d(x: f32, y: f32, z: f32) -> u32 {
    let x = (x.clamp(0.0, 1.0) * 1023.0) as u32;
    let y = (y.clamp(0.0, 1.0) * 1023.0) as u32;
    let z = (z.clamp(0.0, 1.0) * 1023.0) as u32;
    (expand_bits_10(x) << 2) | (expand_bits_10(y) << 1) | expand_bits_10(z)
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
fn aabb_overlap(a: &Aabb3D, b: &Aabb3D) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
        && a.min.z <= b.max.z
        && a.max.z >= b.min.z
}

fn aabb_union(a: &Aabb3D, b: &Aabb3D) -> Aabb3D {
    Aabb3D::new(
        a.min_point().min(b.min_point()),
        a.max_point().max(b.max_point()),
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct Lbvh {
    sorted_indices: Vec<u32>,
    internal_left: Vec<Child>,
    internal_right: Vec<Child>,
    internal_aabbs: Vec<Aabb3D>,
    #[allow(dead_code)]
    leaf_aabbs: Vec<Aabb3D>,
    leaf_count: usize,
}

impl Lbvh {
    pub fn build(aabbs: &[Aabb3D]) -> Self {
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

        // Scene AABB + normalization
        let mut scene_min = Vec3::splat(f32::MAX);
        let mut scene_max = Vec3::splat(f32::NEG_INFINITY);
        for aabb in aabbs {
            scene_min = scene_min.min(aabb.min_point());
            scene_max = scene_max.max(aabb.max_point());
        }
        let extent = scene_max - scene_min;
        let inv_extent = Vec3::new(
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
            if extent.z > 1e-10 {
                1.0 / extent.z
            } else {
                0.0
            },
        );

        // Morton codes for centroids
        let mut indexed: Vec<(u32, u32)> = aabbs
            .iter()
            .enumerate()
            .map(|(i, aabb)| {
                let center = (aabb.min_point() + aabb.max_point()) * 0.5;
                let norm = (center - scene_min) * inv_extent;
                (morton_encode_3d(norm.x, norm.y, norm.z), i as u32)
            })
            .collect();
        indexed.sort_unstable_by_key(|&(code, idx)| (code, idx));

        let codes: Vec<u32> = indexed.iter().map(|&(c, _)| c).collect();
        let sorted_indices: Vec<u32> = indexed.iter().map(|&(_, i)| i).collect();

        // Karras 2012 tree construction
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

        // Leaf AABBs in sorted order
        let leaf_aabbs: Vec<Aabb3D> = sorted_indices
            .iter()
            .map(|&idx| aabbs[idx as usize])
            .collect();

        // Bottom-up AABB refit
        let mut internal_aabbs = vec![Aabb3D::new(Vec3::ZERO, Vec3::ZERO); num_internal];
        fn refit(
            idx: usize,
            left: &[Child],
            right: &[Child],
            internal: &mut [Aabb3D],
            leaves: &[Aabb3D],
        ) -> Aabb3D {
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

    /// Find all overlapping AABB pairs. Returns `(body_a, body_b)` with `a < b`.
    pub fn find_overlapping_pairs(&self, aabbs: &[Aabb3D]) -> Vec<[u32; 2]> {
        let n = self.leaf_count;
        if n <= 1 || self.internal_aabbs.is_empty() {
            return Vec::new();
        }

        let leaf_aabbs: Vec<Aabb3D> = self
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

// ---------------------------------------------------------------------------
// Plane broadphase
// ---------------------------------------------------------------------------

/// Test each body AABB against plane half-spaces.
pub fn find_plane_pairs(normals: &[Vec3], distances: &[f32], aabbs: &[Aabb3D]) -> Vec<(u32, u32)> {
    let mut results = Vec::new();
    for (pi, (n, &d)) in normals.iter().zip(distances.iter()).enumerate() {
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
            if n.dot(near) - d <= 0.0 && n.dot(far) - d >= 0.0 {
                results.push((pi as u32, bi as u32));
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

    fn sphere_aabb(cx: f32, cy: f32, cz: f32, r: f32) -> Aabb3D {
        Aabb3D::new(
            Vec3::new(cx - r, cy - r, cz - r),
            Vec3::new(cx + r, cy + r, cz + r),
        )
    }

    #[test]
    fn test_morton_origin() {
        assert_eq!(morton_encode_3d(0.0, 0.0, 0.0), 0);
    }

    #[test]
    fn test_morton_all_ones() {
        assert_eq!(morton_encode_3d(1.0, 1.0, 1.0), 0x3FFFFFFF);
    }

    #[test]
    fn test_two_overlapping() {
        let aabbs = vec![
            sphere_aabb(0.0, 0.0, 0.0, 1.0),
            sphere_aabb(0.5, 0.0, 0.0, 1.0),
        ];
        let lbvh = Lbvh::build(&aabbs);
        let pairs = lbvh.find_overlapping_pairs(&aabbs);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], [0, 1]);
    }

    #[test]
    fn test_two_separated() {
        let aabbs = vec![
            sphere_aabb(0.0, 0.0, 0.0, 1.0),
            sphere_aabb(10.0, 0.0, 0.0, 1.0),
        ];
        let lbvh = Lbvh::build(&aabbs);
        let pairs = lbvh.find_overlapping_pairs(&aabbs);
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_no_self_pairs_no_duplicates() {
        let aabbs: Vec<Aabb3D> = (0..10)
            .map(|i| sphere_aabb(i as f32 * 0.5, 0.0, 0.0, 1.0))
            .collect();
        let lbvh = Lbvh::build(&aabbs);
        let pairs = lbvh.find_overlapping_pairs(&aabbs);
        for pair in &pairs {
            assert_ne!(pair[0], pair[1]);
            assert!(pair[0] < pair[1]);
        }
        let mut sorted = pairs.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), pairs.len());
    }

    #[test]
    fn test_brute_force_cross_validation() {
        let mut rng: u64 = 12345;
        let mut next_f32 = || -> f32 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng & 0xFFFF) as f32 / 65535.0
        };

        let n = 100;
        let aabbs: Vec<Aabb3D> = (0..n)
            .map(|_| {
                let x = next_f32() * 20.0 - 10.0;
                let y = next_f32() * 20.0 - 10.0;
                let z = next_f32() * 20.0 - 10.0;
                let r = next_f32() * 1.5 + 0.5;
                sphere_aabb(x, y, z, r)
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

        let lbvh = Lbvh::build(&aabbs);
        let mut lbvh_pairs = lbvh.find_overlapping_pairs(&aabbs);
        lbvh_pairs.sort();

        assert_eq!(lbvh_pairs, bf);
    }

    #[test]
    fn test_empty() {
        let lbvh = Lbvh::build(&[]);
        assert!(lbvh.find_overlapping_pairs(&[]).is_empty());
    }

    #[test]
    fn test_single() {
        let aabbs = vec![sphere_aabb(0.0, 0.0, 0.0, 1.0)];
        let lbvh = Lbvh::build(&aabbs);
        assert!(lbvh.find_overlapping_pairs(&aabbs).is_empty());
    }

    #[test]
    fn test_plane_broadphase() {
        let normals = vec![Vec3::Y];
        let distances = vec![0.0];
        let aabbs = vec![
            Aabb3D::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)), // intersects
            Aabb3D::new(Vec3::new(-1.0, 1.0, -1.0), Vec3::new(1.0, 3.0, 1.0)),  // above
        ];
        let pairs = find_plane_pairs(&normals, &distances, &aabbs);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 0));
    }
}
