//! GPU-native Linear BVH broadphase using compute shaders.
//!
//! Pipeline: compute Morton codes → radix sort → build Karras tree → refit AABBs → find pairs.
//! The Morton code computation and pair finding run as compute shaders. The tree build
//! and refit are done on CPU after downloading the sorted codes, which keeps the
//! implementation simple while the sort (the most expensive part) runs fully on GPU.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rubble_gpu::{round_up_workgroups, ComputeKernel, GpuAtomicCounter, GpuBuffer, GpuContext};
use rubble_math::Aabb3D;

use crate::GpuRadixSort;

const WG: u32 = 256;

// ---------------------------------------------------------------------------
// GPU types
// ---------------------------------------------------------------------------

/// Output pair from broadphase.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BroadPair {
    pub a: u32,
    pub b: u32,
}

/// Internal BVH node stored in a GPU buffer for the pair-finding kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BvhNodeGpu {
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
    pub left: i32, // negative → leaf index -(i+1)
    pub right: i32,
    pub _pad0: u32,
    pub _pad1: u32,
}

// ---------------------------------------------------------------------------
// Morton code compute shader (3D, 30-bit)
// ---------------------------------------------------------------------------

const MORTON_3D_WGSL: &str = r#"
struct Aabb {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct Params {
    num_bodies: u32,
    scene_min_x: f32,
    scene_min_y: f32,
    scene_min_z: f32,
    inv_extent_x: f32,
    inv_extent_y: f32,
    inv_extent_z: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(1) var<storage, read_write> morton_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> body_indices: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn expand10(v_in: u32) -> u32 {
    var v = v_in & 0x3FFu;
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v << 8u))  & 0x0300F00Fu;
    v = (v | (v << 4u))  & 0x030C30C3u;
    v = (v | (v << 2u))  & 0x09249249u;
    return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_bodies { return; }
    let aabb = aabbs[idx];
    let cx = (aabb.min_pt.x + aabb.max_pt.x) * 0.5;
    let cy = (aabb.min_pt.y + aabb.max_pt.y) * 0.5;
    let cz = (aabb.min_pt.z + aabb.max_pt.z) * 0.5;
    let nx = clamp((cx - params.scene_min_x) * params.inv_extent_x, 0.0, 1.0);
    let ny = clamp((cy - params.scene_min_y) * params.inv_extent_y, 0.0, 1.0);
    let nz = clamp((cz - params.scene_min_z) * params.inv_extent_z, 0.0, 1.0);
    let ix = u32(nx * 1023.0);
    let iy = u32(ny * 1023.0);
    let iz = u32(nz * 1023.0);
    morton_keys[idx] = (expand10(ix) << 2u) | (expand10(iy) << 1u) | expand10(iz);
    body_indices[idx] = idx;
}
"#;

// ---------------------------------------------------------------------------
// Pair-finding compute shader
// ---------------------------------------------------------------------------

const FIND_PAIRS_WGSL: &str = r#"
struct Aabb {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct BvhNode {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    left: i32,
    right: i32,
    _pad0: u32,
    _pad1: u32,
};

struct Pair {
    a: u32,
    b: u32,
};

@group(0) @binding(0) var<storage, read> leaf_aabbs: array<Aabb>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(3) var<storage, read_write> pairs: array<Pair>;
@group(0) @binding(4) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x = num_leaves, y = max_pairs

fn aabb_overlap(a_min: vec3<f32>, a_max: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> bool {
    return a_min.x <= b_max.x && a_max.x >= b_min.x
        && a_min.y <= b_max.y && a_max.y >= b_min.y
        && a_min.z <= b_max.z && a_max.z >= b_min.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_idx = gid.x;
    let n = params.x;
    let max_pairs = params.y;
    if leaf_idx >= n { return; }

    let body_i = sorted_indices[leaf_idx];
    let ai_min = leaf_aabbs[leaf_idx].min_pt.xyz;
    let ai_max = leaf_aabbs[leaf_idx].max_pt.xyz;

    var stack: array<i32, 64>;
    var sp = 0u;
    stack[0] = 0;
    sp = 1u;

    while sp > 0u {
        sp = sp - 1u;
        let node_idx = stack[sp];
        if node_idx < 0 {
            let j = u32(-node_idx - 1);
            if j != leaf_idx {
                let body_j = sorted_indices[j];
                if body_i < body_j {
                    if aabb_overlap(ai_min, ai_max, leaf_aabbs[j].min_pt.xyz, leaf_aabbs[j].max_pt.xyz) {
                        let slot = atomicAdd(&pair_count, 1u);
                        if slot < max_pairs {
                            pairs[slot].a = body_i;
                            pairs[slot].b = body_j;
                        }
                    }
                }
            }
        } else {
            let nd = nodes[u32(node_idx)];
            if aabb_overlap(ai_min, ai_max, nd.aabb_min.xyz, nd.aabb_max.xyz) {
                if sp + 2u <= 64u {
                    stack[sp] = nd.left; sp = sp + 1u;
                    stack[sp] = nd.right; sp = sp + 1u;
                }
            }
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Morton params uniform layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MortonParams {
    num_bodies: u32,
    scene_min_x: f32,
    scene_min_y: f32,
    scene_min_z: f32,
    inv_extent_x: f32,
    inv_extent_y: f32,
    inv_extent_z: f32,
    _pad: u32,
}

// ---------------------------------------------------------------------------
// CPU-side Karras tree build (reused from lbvh.rs pattern)
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

fn aabb_union(a: &Aabb3D, b: &Aabb3D) -> Aabb3D {
    Aabb3D::new(
        a.min_point().min(b.min_point()),
        a.max_point().max(b.max_point()),
    )
}

fn validate_internal_tree(left: &[Child], right: &[Child]) -> bool {
    if left.is_empty() {
        return true;
    }

    let n = left.len();
    let mut state = vec![0u8; n];
    let mut seen = 0usize;
    let mut stack = vec![(0usize, false)];

    while let Some((idx, expanded)) = stack.pop() {
        if idx >= n {
            return false;
        }

        if expanded {
            if state[idx] != 1 {
                return false;
            }
            state[idx] = 2;
            continue;
        }

        match state[idx] {
            0 => {
                state[idx] = 1;
                seen += 1;
                stack.push((idx, true));
                if let Child::Internal(ci) = right[idx] {
                    stack.push((ci, false));
                }
                if let Child::Internal(ci) = left[idx] {
                    stack.push((ci, false));
                }
            }
            1 => return false,
            2 => return false,
            _ => unreachable!(),
        }
    }

    seen == n
}

fn refit_iterative(
    left: &[Child],
    right: &[Child],
    leaves: &[Aabb3D],
) -> Option<Vec<Aabb3D>> {
    if left.is_empty() {
        return Some(Vec::new());
    }

    let n = left.len();
    let mut internal = vec![Aabb3D::new(Vec3::ZERO, Vec3::ZERO); n];
    let mut state = vec![0u8; n];
    let mut stack = vec![(0usize, false)];

    while let Some((idx, expanded)) = stack.pop() {
        if idx >= n {
            return None;
        }

        if expanded {
            let la = match left[idx] {
                Child::Leaf(i) => *leaves.get(i)?,
                Child::Internal(i) => {
                    if state.get(i).copied()? != 2 {
                        return None;
                    }
                    internal[i]
                }
            };
            let ra = match right[idx] {
                Child::Leaf(i) => *leaves.get(i)?,
                Child::Internal(i) => {
                    if state.get(i).copied()? != 2 {
                        return None;
                    }
                    internal[i]
                }
            };
            internal[idx] = aabb_union(&la, &ra);
            state[idx] = 2;
            continue;
        }

        match state[idx] {
            0 => {
                state[idx] = 1;
                stack.push((idx, true));
                if let Child::Internal(ci) = right[idx] {
                    stack.push((ci, false));
                }
                if let Child::Internal(ci) = left[idx] {
                    stack.push((ci, false));
                }
            }
            1 => return None,
            2 => {}
            _ => unreachable!(),
        }
    }

    Some(internal)
}

fn children_to_gpu_nodes(
    left: &[Child],
    right: &[Child],
    internal_aabbs: &[Aabb3D],
) -> Vec<BvhNodeGpu> {
    (0..left.len())
        .map(|i| {
            let left_val = match left[i] {
                Child::Leaf(l) => -(l as i32 + 1),
                Child::Internal(l) => l as i32,
            };
            let right_val = match right[i] {
                Child::Leaf(r) => -(r as i32 + 1),
                Child::Internal(r) => r as i32,
            };
            let aabb = internal_aabbs[i];
            BvhNodeGpu {
                aabb_min: [
                    aabb.min_point().x,
                    aabb.min_point().y,
                    aabb.min_point().z,
                    0.0,
                ],
                aabb_max: [
                    aabb.max_point().x,
                    aabb.max_point().y,
                    aabb.max_point().z,
                    0.0,
                ],
                left: left_val,
                right: right_val,
                _pad0: 0,
                _pad1: 0,
            }
        })
        .collect()
}

fn build_balanced_tree_cpu(leaf_aabbs: &[Aabb3D]) -> Vec<BvhNodeGpu> {
    if leaf_aabbs.len() <= 1 {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct Task {
        start: usize,
        end: usize,
        parent: Option<(usize, bool)>,
    }

    let mut left = Vec::<Child>::new();
    let mut right = Vec::<Child>::new();
    let mut stack = vec![Task {
        start: 0,
        end: leaf_aabbs.len(),
        parent: None,
    }];

    while let Some(task) = stack.pop() {
        let len = task.end - task.start;
        if len == 0 {
            continue;
        }

        if len == 1 {
            if let Some((parent, is_left)) = task.parent {
                let child = Child::Leaf(task.start);
                if is_left {
                    left[parent] = child;
                } else {
                    right[parent] = child;
                }
            }
            continue;
        }

        let node_idx = left.len();
        left.push(Child::Leaf(task.start));
        right.push(Child::Leaf(task.start));

        if let Some((parent, is_left)) = task.parent {
            let child = Child::Internal(node_idx);
            if is_left {
                left[parent] = child;
            } else {
                right[parent] = child;
            }
        }

        let mid = task.start + len / 2;
        stack.push(Task {
            start: mid,
            end: task.end,
            parent: Some((node_idx, false)),
        });
        stack.push(Task {
            start: task.start,
            end: mid,
            parent: Some((node_idx, true)),
        });
    }

    let mut internal_aabbs = vec![Aabb3D::new(Vec3::ZERO, Vec3::ZERO); left.len()];
    for idx in (0..left.len()).rev() {
        let la = match left[idx] {
            Child::Leaf(i) => leaf_aabbs[i],
            Child::Internal(i) => internal_aabbs[i],
        };
        let ra = match right[idx] {
            Child::Leaf(i) => leaf_aabbs[i],
            Child::Internal(i) => internal_aabbs[i],
        };
        internal_aabbs[idx] = aabb_union(&la, &ra);
    }

    children_to_gpu_nodes(&left, &right, &internal_aabbs)
}

/// Build Karras tree + refit on CPU. Returns (internal_nodes, leaf_aabbs_sorted).
fn build_tree_cpu(
    sorted_codes: &[u32],
    sorted_indices: &[u32],
    aabbs: &[Aabb3D],
) -> (Vec<BvhNodeGpu>, Vec<Aabb3D>) {
    let n = sorted_codes.len();
    let num_internal = n - 1;

    let mut internal_left: Vec<Child> = Vec::with_capacity(num_internal);
    let mut internal_right: Vec<Child> = Vec::with_capacity(num_internal);

    for i in 0..num_internal {
        let (range_left, range_right, split) = karras_node(sorted_codes, i);
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

    let leaf_aabbs: Vec<Aabb3D> = sorted_indices
        .iter()
        .map(|&idx| aabbs[idx as usize])
        .collect();

    let gpu_nodes = if validate_internal_tree(&internal_left, &internal_right) {
        if let Some(internal_aabbs) = refit_iterative(&internal_left, &internal_right, &leaf_aabbs) {
            children_to_gpu_nodes(&internal_left, &internal_right, &internal_aabbs)
        } else {
            build_balanced_tree_cpu(&leaf_aabbs)
        }
    } else {
        build_balanced_tree_cpu(&leaf_aabbs)
    };

    (gpu_nodes, leaf_aabbs)
}

// ---------------------------------------------------------------------------
// GpuLbvh
// ---------------------------------------------------------------------------

/// GPU-accelerated Linear BVH broadphase.
///
/// Morton code computation and radix sort run on GPU.
/// Tree construction and refit run on CPU (downloaded sorted codes).
/// Pair finding runs on GPU via parallel BVH traversal.
pub struct GpuLbvh {
    morton_kernel: ComputeKernel,
    find_pairs_kernel: ComputeKernel,
    radix_sort: GpuRadixSort,
    morton_keys: GpuBuffer<u32>,
    body_indices: GpuBuffer<u32>,
    internal_nodes: GpuBuffer<BvhNodeGpu>,
    leaf_aabbs_sorted: GpuBuffer<Aabb3D>,
    sorted_indices_buf: GpuBuffer<u32>,
    pairs_out: GpuBuffer<BroadPair>,
    pair_counter: GpuAtomicCounter,
    morton_params_buf: wgpu::Buffer,
    find_params_buf: wgpu::Buffer,
}

impl GpuLbvh {
    /// Create a new GPU LBVH broadphase for up to `max_bodies` bodies.
    pub fn new(ctx: &GpuContext, max_bodies: usize) -> Self {
        let morton_kernel = ComputeKernel::from_wgsl(ctx, MORTON_3D_WGSL, "main");
        let find_pairs_kernel = ComputeKernel::from_wgsl(ctx, FIND_PAIRS_WGSL, "main");
        let radix_sort = GpuRadixSort::new(ctx, max_bodies);

        let morton_keys = GpuBuffer::new(ctx, max_bodies.max(1));
        let body_indices = GpuBuffer::new(ctx, max_bodies.max(1));
        let internal_nodes = GpuBuffer::new(ctx, max_bodies.max(1));
        let leaf_aabbs_sorted = GpuBuffer::new(ctx, max_bodies.max(1));
        let sorted_indices_buf = GpuBuffer::new(ctx, max_bodies.max(1));
        let max_pairs = max_bodies * 8;
        let pairs_out = GpuBuffer::new(ctx, max_pairs.max(1));
        let pair_counter = GpuAtomicCounter::new(ctx);

        let morton_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morton params"),
            size: std::mem::size_of::<MortonParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let find_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("find_pairs params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            morton_kernel,
            find_pairs_kernel,
            radix_sort,
            morton_keys,
            body_indices,
            internal_nodes,
            leaf_aabbs_sorted,
            sorted_indices_buf,
            pairs_out,
            pair_counter,
            morton_params_buf,
            find_params_buf,
        }
    }

    /// Run the full GPU broadphase pipeline on a 3D AABB buffer.
    pub fn build_and_query(
        &mut self,
        ctx: &GpuContext,
        aabbs: &GpuBuffer<Aabb3D>,
        num_bodies: u32,
    ) -> Vec<[u32; 2]> {
        self.build_and_query_raw(ctx, aabbs.buffer(), &aabbs.download(ctx), num_bodies)
    }

    /// Run the full GPU broadphase pipeline on any AABB buffer.
    ///
    /// `aabb_buf` is the raw GPU buffer (must have the same layout as `array<Aabb>` in WGSL:
    /// each element is `{min_pt: vec4<f32>, max_pt: vec4<f32>}`).
    /// `cpu_aabbs` are the same AABBs on CPU (used for scene bounds computation).
    /// Both `Aabb3D` and `Aabb2D` satisfy this layout (32 bytes, Vec4 min + Vec4 max).
    pub fn build_and_query_raw(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        cpu_aabbs: &[Aabb3D],
        num_bodies: u32,
    ) -> Vec<[u32; 2]> {
        if num_bodies <= 1 {
            return Vec::new();
        }
        let n = num_bodies as usize;

        // Step 0: Compute scene bounds from CPU-side AABBs for Morton normalization.
        let mut scene_min = Vec3::splat(f32::MAX);
        let mut scene_max = Vec3::splat(f32::NEG_INFINITY);
        for aabb in cpu_aabbs {
            scene_min = scene_min.min(aabb.min_point());
            scene_max = scene_max.max(aabb.max_point());
        }
        let extent = scene_max - scene_min;
        let inv_x = if extent.x > 1e-10 {
            1.0 / extent.x
        } else {
            0.0
        };
        let inv_y = if extent.y > 1e-10 {
            1.0 / extent.y
        } else {
            0.0
        };
        let inv_z = if extent.z > 1e-10 {
            1.0 / extent.z
        } else {
            0.0
        };

        // Step 1: Compute Morton codes on GPU
        let params = MortonParams {
            num_bodies,
            scene_min_x: scene_min.x,
            scene_min_y: scene_min.y,
            scene_min_z: scene_min.z,
            inv_extent_x: inv_x,
            inv_extent_y: inv_y,
            inv_extent_z: inv_z,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.morton_params_buf, 0, bytemuck::bytes_of(&params));
        self.morton_keys.grow_if_needed(ctx, n);
        self.body_indices.grow_if_needed(ctx, n);
        self.morton_keys.set_len(num_bodies);
        self.body_indices.set_len(num_bodies);

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton"),
            layout: self.morton_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: aabb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.morton_keys.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.body_indices.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.morton_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morton"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.morton_kernel.pipeline());
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, WG), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        // Step 2: Radix sort Morton codes entirely on GPU, then download the
        // sorted keys/indices for the lightweight CPU tree build.
        self.radix_sort
            .sort_key_value_in_place(ctx, &mut self.morton_keys, &mut self.body_indices);
        let sorted_codes = self.morton_keys.download(ctx);
        let sorted_indices = self.body_indices.download(ctx);

        // Step 3: Build Karras tree on CPU (tree build is lightweight; sort was the expensive part)
        let (gpu_nodes, leaf_aabbs) = build_tree_cpu(&sorted_codes, &sorted_indices, cpu_aabbs);
        if gpu_nodes.is_empty() {
            return Vec::new();
        }

        // Step 4: Upload tree + sorted data for GPU pair finding
        self.internal_nodes.upload(ctx, &gpu_nodes);
        self.leaf_aabbs_sorted.upload(ctx, &leaf_aabbs);
        self.sorted_indices_buf.upload(ctx, &sorted_indices);

        // Step 5: Find pairs on GPU via parallel BVH traversal
        self.pair_counter.reset(ctx);
        let max_pairs = (n * 8) as u32;
        self.pairs_out.grow_if_needed(ctx, max_pairs as usize);

        let find_params: [u32; 4] = [num_bodies, max_pairs, 0, 0];
        ctx.queue
            .write_buffer(&self.find_params_buf, 0, bytemuck::cast_slice(&find_params));

        let bg2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("find_pairs"),
            layout: self.find_pairs_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.leaf_aabbs_sorted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.sorted_indices_buf.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.internal_nodes.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.pairs_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pair_counter.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.find_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("find_pairs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.find_pairs_kernel.pipeline());
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, 64), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let pair_count = self.pair_counter.read(ctx).min(max_pairs);
        if pair_count == 0 {
            return Vec::new();
        }
        self.pairs_out.set_len(pair_count);
        let pairs = self.pairs_out.download(ctx);

        let mut result: Vec<[u32; 2]> = pairs
            .iter()
            .take(pair_count as usize)
            .map(|p| {
                let (a, b) = if p.a < p.b { (p.a, p.b) } else { (p.b, p.a) };
                [a, b]
            })
            .collect();
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Async variant for WASM/WebGPU callers that already have CPU AABBs.
    #[cfg(target_arch = "wasm32")]
    pub async fn build_and_query_raw_async(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        cpu_aabbs: &[Aabb3D],
        num_bodies: u32,
    ) -> Vec<[u32; 2]> {
        if num_bodies <= 1 {
            return Vec::new();
        }
        let n = num_bodies as usize;

        let mut scene_min = Vec3::splat(f32::MAX);
        let mut scene_max = Vec3::splat(f32::NEG_INFINITY);
        for aabb in cpu_aabbs {
            scene_min = scene_min.min(aabb.min_point());
            scene_max = scene_max.max(aabb.max_point());
        }
        let extent = scene_max - scene_min;
        let inv_x = if extent.x > 1e-10 {
            1.0 / extent.x
        } else {
            0.0
        };
        let inv_y = if extent.y > 1e-10 {
            1.0 / extent.y
        } else {
            0.0
        };
        let inv_z = if extent.z > 1e-10 {
            1.0 / extent.z
        } else {
            0.0
        };

        let params = MortonParams {
            num_bodies,
            scene_min_x: scene_min.x,
            scene_min_y: scene_min.y,
            scene_min_z: scene_min.z,
            inv_extent_x: inv_x,
            inv_extent_y: inv_y,
            inv_extent_z: inv_z,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.morton_params_buf, 0, bytemuck::bytes_of(&params));
        self.morton_keys.grow_if_needed(ctx, n);
        self.body_indices.grow_if_needed(ctx, n);
        self.morton_keys.set_len(num_bodies);
        self.body_indices.set_len(num_bodies);

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton"),
            layout: self.morton_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: aabb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.morton_keys.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.body_indices.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.morton_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morton"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.morton_kernel.pipeline());
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, WG), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        self.radix_sort
            .sort_key_value_in_place(ctx, &mut self.morton_keys, &mut self.body_indices);
        let sorted_codes = self.morton_keys.download_async(ctx).await;
        let sorted_indices = self.body_indices.download_async(ctx).await;

        let (gpu_nodes, leaf_aabbs) = build_tree_cpu(&sorted_codes, &sorted_indices, cpu_aabbs);
        if gpu_nodes.is_empty() {
            return Vec::new();
        }

        self.internal_nodes.upload(ctx, &gpu_nodes);
        self.leaf_aabbs_sorted.upload(ctx, &leaf_aabbs);
        self.sorted_indices_buf.upload(ctx, &sorted_indices);

        self.pair_counter.reset(ctx);
        let max_pairs = (n * 8) as u32;
        self.pairs_out.grow_if_needed(ctx, max_pairs as usize);

        let find_params: [u32; 4] = [num_bodies, max_pairs, 0, 0];
        ctx.queue
            .write_buffer(&self.find_params_buf, 0, bytemuck::cast_slice(&find_params));

        let bg2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("find_pairs"),
            layout: self.find_pairs_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.leaf_aabbs_sorted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.sorted_indices_buf.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.internal_nodes.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.pairs_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pair_counter.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.find_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("find_pairs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.find_pairs_kernel.pipeline());
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, 64), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let pair_count = self.pair_counter.read_async(ctx).await.min(max_pairs);
        if pair_count == 0 {
            return Vec::new();
        }

        self.pairs_out.set_len(pair_count);
        let pairs = self.pairs_out.download_async(ctx).await;

        let mut result: Vec<[u32; 2]> = pairs
            .iter()
            .take(pair_count as usize)
            .map(|p| {
                let (a, b) = if p.a < p.b { (p.a, p.b) } else { (p.b, p.a) };
                [a, b]
            })
            .collect();
        result.sort_unstable();
        result.dedup();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rubble_gpu::GpuBuffer;
    use rubble_math::Aabb3D;

    fn try_ctx() -> Option<GpuContext> {
        crate::test_gpu()
    }

    fn aabb(min: [f32; 3], max: [f32; 3]) -> Aabb3D {
        Aabb3D::new(Vec3::from(min), Vec3::from(max))
    }

    #[test]
    fn single_body_returns_no_pairs() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 1);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 1);
        assert!(pairs.is_empty());
    }

    #[test]
    fn two_overlapping_bodies() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            aabb([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 2);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 2);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], [0, 1]);
    }

    #[test]
    fn two_separated_bodies_no_pairs() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            aabb([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 2);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 2);
        assert!(pairs.is_empty());
    }

    #[test]
    fn three_bodies_partial_overlap() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        // A overlaps B, B overlaps C, A does NOT overlap C
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), // A
            aabb([1.5, 1.5, 1.5], [3.5, 3.5, 3.5]), // B
            aabb([3.0, 3.0, 3.0], [5.0, 5.0, 5.0]), // C
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 3);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 3);
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&[0, 1]));
        assert!(pairs.contains(&[1, 2]));
        // A-C should NOT be a pair
        assert!(!pairs.contains(&[0, 2]));
    }

    #[test]
    fn all_overlapping_returns_all_pairs() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        // All overlap each other (big box)
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
            aabb([1.0, 1.0, 1.0], [11.0, 11.0, 11.0]),
            aabb([2.0, 2.0, 2.0], [12.0, 12.0, 12.0]),
            aabb([3.0, 3.0, 3.0], [13.0, 13.0, 13.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 4);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 4);
        // 4 choose 2 = 6 pairs
        assert_eq!(pairs.len(), 6);
    }

    #[test]
    fn no_duplicate_pairs() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [5.0, 5.0, 5.0]),
            aabb([1.0, 1.0, 1.0], [6.0, 6.0, 6.0]),
            aabb([2.0, 2.0, 2.0], [7.0, 7.0, 7.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 3);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 3);
        // All pairs should be unique and ordered (a < b)
        for p in &pairs {
            assert!(p[0] < p[1], "pair should be ordered: {:?}", p);
        }
        let mut deduped = pairs.clone();
        deduped.dedup();
        assert_eq!(pairs.len(), deduped.len(), "no duplicates expected");
    }

    #[test]
    fn build_and_query_raw_works() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            aabb([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 2);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query_raw(&ctx, buf.buffer(), &aabbs, 2);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], [0, 1]);
    }

    #[test]
    fn many_bodies_stress() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let n = 64;
        let mut lbvh = GpuLbvh::new(&ctx, n);
        // Line of non-overlapping unit cubes along X
        let aabbs: Vec<Aabb3D> = (0..n)
            .map(|i| {
                let x = i as f32 * 3.0; // gap of 2 between each
                aabb([x, 0.0, 0.0], [x + 1.0, 1.0, 1.0])
            })
            .collect();
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, n);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, n as u32);
        assert!(
            pairs.is_empty(),
            "separated bodies should have no pairs, got {}",
            pairs.len()
        );
    }

    #[test]
    fn touching_aabbs_overlap() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        // Edge-touching (min.x of B == max.x of A)
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            aabb([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 2);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 2);
        // Touching AABBs satisfy <=/>= so they overlap
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn karras_tree_build_correctness() {
        // Test CPU tree build directly with known Morton codes
        let codes = vec![0u32, 1, 2, 3];
        let indices = vec![0u32, 1, 2, 3];
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            aabb([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
            aabb([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]),
            aabb([3.0, 0.0, 0.0], [4.0, 1.0, 1.0]),
        ];
        let (nodes, leaf_aabbs) = build_tree_cpu(&codes, &indices, &aabbs);

        // 4 leaves → 3 internal nodes
        assert_eq!(nodes.len(), 3);
        assert_eq!(leaf_aabbs.len(), 4);

        // Root node AABB should encompass all leaves
        let root = &nodes[0];
        assert!(root.aabb_min[0] <= 0.0);
        assert!(root.aabb_max[0] >= 4.0);
        assert!(root.aabb_min[1] <= 0.0);
        assert!(root.aabb_max[1] >= 1.0);
    }
}
