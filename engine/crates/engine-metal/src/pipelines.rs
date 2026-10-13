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
            ("persist_contacts", msl!("persist_contacts")),
            ("warmstart_joints", msl!("warmstart_joints")),
            ("build_awake_list", msl!("build_awake_list")),
            ("csr_clear", msl!("csr_clear")),
            ("csr_degree_contacts", msl!("csr_degree_contacts")),
            ("csr_degree_joints", msl!("csr_degree_joints")),
            ("csr_prefix", msl!("csr_prefix")),
            ("csr_scatter_contacts", msl!("csr_scatter_contacts")),
            ("csr_scatter_joints", msl!("csr_scatter_joints")),
            ("csr_sort", msl!("csr_sort")),
            ("color_bodies", msl!("color_bodies")),
            ("init_colors", msl!("init_colors")),
            ("pack_color_lists", msl!("pack_color_lists")),
            ("write_color_args", msl!("write_color_args")),
            ("solve_primal_color", msl!("solve_primal_color")),
            ("solve_dual", msl!("solve_dual")),
            ("extract_velocity_sleep", msl!("extract_velocity_sleep")),
            ("ccd_classify", msl!("ccd_classify")),
            ("ccd_sweep", msl!("ccd_sweep")),
            ("recompute_inertia", msl!("recompute_inertia")),
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
