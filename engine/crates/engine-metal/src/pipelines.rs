use metal::*;
use std::collections::HashMap;

macro_rules! msl {
    ($name:expr) => {
        include_str!(concat!(env!("OUT_DIR"), "/msl/", $name, ".metal"))
    };
}

pub struct Pipelines {
    pub map: HashMap<&'static str, ComputePipelineState>,
}

impl Pipelines {
    pub fn new(device: &Device) -> anyhow::Result<Self> {
        let entries: &[(&str, &str)] = &[
            ("abi_roundtrip", msl!("abi_roundtrip")),
            ("apply_commands", msl!("apply_commands")),
            ("predict_bodies", msl!("predict_bodies")),
            ("build_aabbs", msl!("build_aabbs")),
            ("joint_hash_clear", msl!("joint_hash_clear")),
            ("joint_hash_insert", msl!("joint_hash_insert")),
            ("broadphase_clear", msl!("broadphase_clear")),
            ("broadphase_insert", msl!("broadphase_insert")),
            ("broadphase_query", msl!("broadphase_query")),
            ("broadphase_grid", msl!("broadphase_grid")),
            ("narrowphase", msl!("narrowphase")),
            ("update_refs", msl!("update_refs")),
            ("contact_hash_clear", msl!("contact_hash_clear")),
            ("contact_hash_insert", msl!("contact_hash_insert")),
            ("persist_contacts", msl!("persist_contacts")),
            ("persist_manifolds", msl!("persist_manifolds")),
            ("warmstart_joints", msl!("warmstart_joints")),
            ("build_awake_list", msl!("build_awake_list")),
            ("csr_clear", msl!("csr_clear")),
            ("csr_degree_contacts", msl!("csr_degree_contacts")),
            ("csr_degree_joints", msl!("csr_degree_joints")),
            ("csr_prefix", msl!("csr_prefix")),
            ("csr_prefix_blocks", msl!("csr_prefix_blocks")),
            ("csr_prefix_block_sums", msl!("csr_prefix_block_sums")),
            ("csr_prefix_add", msl!("csr_prefix_add")),
            ("csr_prefix_init", msl!("csr_prefix_init")),
            ("csr_prefix_scan", msl!("csr_prefix_scan")),
            ("csr_prefix_swap", msl!("csr_prefix_swap")),
            ("csr_prefix_finalize", msl!("csr_prefix_finalize")),
            ("csr_scatter_contacts", msl!("csr_scatter_contacts")),
            ("csr_scatter_joints", msl!("csr_scatter_joints")),
            ("csr_sort", msl!("csr_sort")),
            ("csr_sort_append", msl!("csr_sort_append")),
            ("color_bodies", msl!("color_bodies")),
            ("init_colors", msl!("init_colors")),
            ("pack_color_lists", msl!("pack_color_lists")),
            ("count_color_tiles", msl!("count_color_tiles")),
            ("prefix_color_tiles", msl!("prefix_color_tiles")),
            ("scatter_color_tiles", msl!("scatter_color_tiles")),
            ("write_color_args", msl!("write_color_args")),
            ("write_work_args", msl!("write_work_args")),
            ("write_sleep_args", msl!("write_sleep_args")),
            ("clear_solver_contacts", msl!("clear_solver_contacts")),
            ("pack_solver_contacts", msl!("pack_solver_contacts")),
            (
                "solve_primal_color_compact",
                msl!("solve_primal_color_compact"),
            ),
            ("solve_primal_one_compact", msl!("solve_primal_one_compact")),
            ("solve_dual_compact", msl!("solve_dual_compact")),
            ("solve_primal_color", msl!("solve_primal_color")),
            ("solve_primal_one", msl!("solve_primal_one")),
            ("solve_dual", msl!("solve_dual")),
            ("extract_velocity_sleep", msl!("extract_velocity_sleep")),
            ("sleep_island_gate", msl!("sleep_island_gate")),
            ("sleep_island_commit", msl!("sleep_island_commit")),
            ("ccd_classify", msl!("ccd_classify")),
            ("ccd_sweep", msl!("ccd_sweep")),
            ("recompute_inertia", msl!("recompute_inertia")),
            ("record_color_degrees", msl!("record_color_degrees")),
            ("reuse_all_manifolds", msl!("reuse_all_manifolds")),
            ("convex_classify", msl!("convex_classify")),
            ("convex_bucket", msl!("convex_bucket")),
            ("convex_write_args", msl!("convex_write_args")),
            ("convex_sat_small", msl!("convex_sat_small")),
            ("convex_sat_large", msl!("convex_sat_large")),
            ("convex_manifold", msl!("convex_manifold")),
            ("vehicle_input", msl!("vehicle_input")),
            ("wheel_query", msl!("wheel_query")),
            ("vehicle_link", msl!("vehicle_link")),
            ("vehicle_prepare", msl!("vehicle_prepare")),
            ("vehicle_iterate", msl!("vehicle_iterate")),
            ("vehicle_post", msl!("vehicle_post")),
            ("raycast_batch", msl!("raycast_batch")),
        ];
        let mut map = HashMap::new();
        let opts = CompileOptions::new();
        for (name, source) in entries {
            let lib = device
                .new_library_with_source(source, &opts)
                .map_err(|e| anyhow::anyhow!("compile {name}: {e}"))?;
            let func = lib
                .get_function(name, None)
                .map_err(|e| anyhow::anyhow!("function {name}: {e}"))?;
            let pso = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow::anyhow!("pso {name}: {e}"))?;
            map.insert(*name, pso);
        }
        Ok(Self { map })
    }

    pub fn get(&self, name: &str) -> anyhow::Result<&ComputePipelineState> {
        self.map
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing pipeline {name}"))
    }
}
