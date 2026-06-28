//! # Brightness Extraction Component
//!
//! Extracts bright areas from a texture by zeroing pixels whose luminance falls
//! below a threshold. This is typically the first stage in a bloom pipeline: only
//! the bright parts survive, then get blurred and composited back.
//!
//! ## Parameters
//!
//! | Parameter | Range | Description |
//! |-----------|-------|-------------|
//! | `threshold` | 0.0–1.0 | Luminance cutoff — pixels below this go to black |
//!
//! ## Pipeline Builder Usage
//!
//! ```rust,ignore
//! // Bloom: extract brights → downsample → blur → composite
//! PipelineBuilder::new()
//!     .brightness_extract(lo_config, 0.6) // keep pixels > 60% luminance
//!     .downsample(lo_config)
//!     .gaussian_blur_passes(lo_config, 2, 2.0, 10.0)
//!     .bloom_composite_with_curve(hi_config, 3.0, 3.0)
//!     .build(device)?;
//! ```

use crate::pipeline::{ComponentType, PipelineComponent, SimpleComponent, TextureConfig};
use nannou::prelude::*;
use nannou::wgpu;

/// Brightness extraction component that masks out pixels below a luminance threshold.
///
/// Output pixels above the threshold retain their original color; everything else
/// becomes `(0, 0, 0, 0)`. This isolates the bright areas of a scene for
/// subsequent bloom processing.
///
/// # Runtime Parameters
///
/// - [`set_threshold`](BrightnessComponent::set_threshold) — adjust the luminance cutoff
pub struct BrightnessComponent {
    enabled: bool,
    threshold: f32,

    // GPU resources
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    threshold_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl BrightnessComponent {
    pub fn new(device: &wgpu::Device, output_config: TextureConfig, threshold: f32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Brightness Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/post/brightness.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Brightness Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu_types::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Brightness Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Brightness Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Brightness sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let threshold_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Brightness Threshold Buffer"),
            contents: bytemuck::cast_slice(&[threshold]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create a dummy bind group (will be updated when processing)
        let dummy_texture = nannou::wgpu::TextureBuilder::new()
            .size([2, 2])
            .format(output_config.format)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .build(device);
        let dummy_view = dummy_texture.view().build();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brightness Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        threshold_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        Self {
            enabled: true,
            threshold,
            pipeline,
            bind_group,
            bind_group_layout,
            threshold_buffer,
            sampler,
        }
    }

    /// Change the brightness threshold at runtime. Lower values keep more of the image.
    pub fn set_threshold(&mut self, queue: &wgpu::Queue, threshold: f32) {
        self.threshold = threshold;
        queue.write_buffer(
            &self.threshold_buffer,
            0,
            bytemuck::cast_slice(&[threshold]),
        );
    }
}

impl SimpleComponent for BrightnessComponent {}

impl PipelineComponent for BrightnessComponent {
    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        _effect_view: Option<&wgpu::TextureView>,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brightness Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        self.threshold_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });
    }

    fn encode_pass(&mut self, encoder: &mut wgpu::CommandEncoder, output_view: &wgpu::TextureView) {
        if !self.enabled {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Brightness pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    fn name(&self) -> &str {
        "Brightness"
    }

    fn update_parameters(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.threshold_buffer,
            0,
            bytemuck::cast_slice(&[self.threshold]),
        );
    }

    fn component_type(&self) -> ComponentType {
        ComponentType::Simple
    }
}
