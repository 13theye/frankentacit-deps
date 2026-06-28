//! # Resample Component
//!
//! Bilinear resample to an arbitrary target resolution and/or texture format.
//! Unlike [`DownsampleComponent`] which is optimized for halving, this component
//! can scale to any size — use it to upscale from a downsampled blur back to
//! full resolution, or to convert between texture formats mid-pipeline.
//!
//! ## Pipeline Builder Usage
//!
//! ```rust,ignore
//! // Upscale a half-resolution blur back to full resolution
//! PipelineBuilder::new()
//!     .downsample(lo_config)
//!     .gaussian_blur_passes(lo_config, 2, 2.0, 5.0)
//!     .resample(hi_config) // back to original dimensions
//!     .build(device)?;
//! ```
//!
//! ## No Runtime Parameters
//!
//! This component has no configurable parameters — the target resolution and
//! format are fixed at construction time via `TextureConfig`.

use crate::pipeline::{ComponentType, PipelineComponent, SimpleComponent, TextureConfig};
use nannou::wgpu;

/// Bilinear resample component for resolution and format conversion.
///
/// Draws the input texture onto a fullscreen triangle at the output resolution
/// using linear filtering. The output dimensions and format are determined by
/// the `TextureConfig` passed to the constructor.
pub struct ResampleComponent {
    enabled: bool,

    // GPU resources
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl ResampleComponent {
    pub fn new(device: &wgpu::Device, output_config: TextureConfig) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Resample Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fragment/resample.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Resample Bind Group Layout"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Resample Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Resample Pipeline"),
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
            label: Some("Resample sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create a dummy bind group (will be updated when processing)
        let dummy_texture = nannou::wgpu::TextureBuilder::new()
            .size([2, 2])
            .format(output_config.format)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .build(device);
        let dummy_view = dummy_texture.view().build();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resample Bind Group"),
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
            ],
        });

        Self {
            enabled: true,
            pipeline,
            bind_group,
            bind_group_layout,
            sampler,
        }
    }
}

impl SimpleComponent for ResampleComponent {}

impl PipelineComponent for ResampleComponent {
    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        _effect_view: Option<&wgpu::TextureView>,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resample Bind Group"),
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
            ],
        });
    }

    fn encode_pass(&mut self, encoder: &mut wgpu::CommandEncoder, output_view: &wgpu::TextureView) {
        if !self.enabled {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Resample pass"),
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
        "Resample"
    }

    fn update_parameters(&mut self, _queue: &wgpu::Queue) {
        // No parameters to update for basic resampling
    }

    fn component_type(&self) -> ComponentType {
        ComponentType::Simple
    }
}
