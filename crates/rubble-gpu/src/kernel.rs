use crate::GpuContext;

/// A compiled compute pipeline with its bind group layout.
pub struct ComputeKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputeKernel {
    /// Create a compute kernel from a WGSL source string.
    pub fn from_wgsl(ctx: &GpuContext, wgsl: &str, entry_point: &str) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ComputeKernel"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        Self::from_module(ctx, &module, entry_point)
    }

    /// Create a compute kernel from SPIR-V bytes. The SPIR-V is converted to
    /// WGSL via naga before being loaded.
    pub fn from_spirv(ctx: &GpuContext, spirv_bytes: &[u8], entry_point: &str) -> Self {
        let opts = naga::front::spv::Options::default();
        let module =
            naga::front::spv::parse_u8_slice(spirv_bytes, &opts).expect("failed to parse SPIR-V");

        let mut validator =
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), Default::default());
        let info = validator
            .validate(&module)
            .expect("SPIR-V validation failed");

        let wgsl =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .expect("failed to convert SPIR-V to WGSL");

        Self::from_wgsl(ctx, &wgsl, entry_point)
    }

    fn from_module(ctx: &GpuContext, module: &wgpu::ShaderModule, entry_point: &str) -> Self {
        // Use auto layout so the bind group layout is inferred from the shader.
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ComputeKernel pipeline"),
                layout: None, // auto layout
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Reference to the underlying compute pipeline.
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Reference to bind group layout (group 0) inferred from the shader.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}

/// Round up `total` to the next multiple of `workgroup_size`, then divide.
pub fn round_up_workgroups(total: u32, workgroup_size: u32) -> u32 {
    total.div_ceil(workgroup_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuBuffer;

    #[test]
    fn test_compute_kernel_wgsl() {
        let Some(ctx) = crate::test_gpu() else { eprintln!("SKIP: No GPU"); return; };

        let wgsl = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&data) {
        data[id.x] = data[id.x] * 2.0;
    }
}
"#;
        let kernel = ComputeKernel::from_wgsl(&ctx, wgsl, "main");

        let mut buf = GpuBuffer::<f32>::new(&ctx, 4);
        buf.upload(&ctx, &[1.0f32, 2.0, 3.0, 4.0]);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: kernel.bind_group_layout(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.buffer().as_entire_binding(),
            }],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(kernel.pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(round_up_workgroups(4, 64), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let result = buf.download(&ctx);
        assert_eq!(result, vec![2.0f32, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_round_up_workgroups() {
        assert_eq!(round_up_workgroups(1, 64), 1);
        assert_eq!(round_up_workgroups(64, 64), 1);
        assert_eq!(round_up_workgroups(65, 64), 2);
        assert_eq!(round_up_workgroups(128, 64), 2);
        assert_eq!(round_up_workgroups(0, 64), 0);
        assert_eq!(round_up_workgroups(256, 256), 1);
    }
}
