// src/renderers/texture_heatmap.rs
//
// Creates a heatmap texture by sampling and blurring an input particle texture
// Much more efficient than the particle-based approach since it operates in O(pixels) instead of O(particles × pixels)
//
// Usage workflow:
// 1. ParticleRenderer renders particles to a texture
// 2. TextureHeatmapRenderer takes that texture as input and generates a heatmap
// 3. The heatmap can be rendered to screen or used for further processing

use nannou::wgpu;
use std::collections::HashMap;

use crate::TextureConfig;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextureHeatmapParams {
    pub resolution: [f32; 2],
    pub blur_radius: f32,
    pub intensity_scale: f32,
    pub _padding: [f32; 4],
}

pub struct TextureHeatmapRenderer {
    // Compute pipeline for heatmap generation
    heatmap_pipeline: wgpu::ComputePipeline,

    // Buffers
    params_buffer: wgpu::Buffer,

    // Bind group layouts
    bind_group_layout: wgpu::BindGroupLayout,

    // Cache bind groups per input texture view to avoid recreation
    #[allow(dead_code)]
    bind_groups: std::cell::RefCell<HashMap<*const wgpu::TextureView, wgpu::BindGroup>>,

    // Parameters
    width: u32,
    height: u32,
}

impl TextureHeatmapRenderer {
    pub fn new(device: &wgpu::Device, config: TextureConfig) -> Self {
        // Create parameters buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Heatmap Params Buffer"),
            size: std::mem::size_of::<TextureHeatmapParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create shader
        let heatmap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Texture Heatmap Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/compute/texture_heatmap.wgsl").into(),
            ),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Heatmap Bind Group Layout"),
            entries: &[
                // Input particle texture (sampled)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Output heatmap texture (storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Parameters (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create compute pipeline
        let heatmap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Texture Heatmap Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let heatmap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Texture Heatmap Compute Pipeline"),
            layout: Some(&heatmap_pipeline_layout),
            module: &heatmap_shader,
            entry_point: "main",
        });

        Self {
            heatmap_pipeline,
            params_buffer,
            bind_group_layout,
            bind_groups: std::cell::RefCell::new(HashMap::new()),
            width: config.width,
            height: config.height,
        }
    }

    /// Encodes a compute pass to generate a heatmap from a particle texture
    #[allow(clippy::too_many_arguments)]
    pub fn encode_heatmap_into(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        particle_texture_view: &wgpu::TextureView,
        target_view: &wgpu::TextureView,
        blur_radius: f32,
        intensity_scale: f32,
    ) {
        // Update parameters
        let params = TextureHeatmapParams {
            resolution: [self.width as f32, self.height as f32],
            blur_radius,
            intensity_scale,
            _padding: [0.0; 4],
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Get or create bind group for this input texture
        let bind_group = self.get_or_create_bind_group(device, particle_texture_view, target_view);

        // Single compute pass to generate heatmap
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Texture Heatmap Compute Pass"),
            });

            compute_pass.set_pipeline(&self.heatmap_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch in 8x8 work groups
            let workgroup_size = 8;
            let dispatch_x = self.width.div_ceil(workgroup_size);
            let dispatch_y = self.height.div_ceil(workgroup_size);

            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    fn get_or_create_bind_group(
        &self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        // For simplicity, create a new bind group each time
        // In a real optimization, we'd cache based on both input and output views

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Heatmap Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        })
    }
}
