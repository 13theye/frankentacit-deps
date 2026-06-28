//! # Inversion Component
//!
//! Monochromatic luminance inversion — inverts the brightness of each pixel while
//! preserving hue and saturation. The `darken_darks` parameter provides artistic
//! control over how dark areas are rendered in the inverted output.
//!
//! ## Parameters
//!
//! | Parameter | Range | Description |
//! |-----------|-------|-------------|
//! | `darken_darks` | 0.0–2.0 | How aggressively dark areas are pushed darker after inversion |
//!
//! ## Pipeline Builder Usage
//!
//! ```rust,ignore
//! PipelineBuilder::new()
//!     .inversion(config, 1.0) // standard luminance inversion
//!     .build(device)?;
//! ```

use crate::pipeline::{ComponentType, PipelineComponent, SimpleComponent, TextureConfig};
use nannou::prelude::*;
use nannou::wgpu;

/// Monochromatic luminance inversion component.
///
/// Inverts pixel luminance while preserving hue/saturation. The `darken_darks`
/// parameter controls post-inversion darkening — at `0.0` the inversion is pure,
/// at higher values dark regions in the output become even darker.
///
/// # Runtime Parameters
///
/// - [`set_darken_darks`](InversionComponent::set_darken_darks) — adjust post-inversion darkening
/// - [`set_enabled`](InversionComponent::set_enabled) / [`is_enabled`](InversionComponent::is_enabled) — toggle bypass
pub struct InversionComponent {
    enabled: bool,
    darken_darks: f32,

    // GPU resources
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    darken_darks_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl InversionComponent {
    pub fn new(device: &wgpu::Device, output_config: TextureConfig, darken_darks: f32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Inversion Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/post/inversion.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Inversion Bind Group Layout"),
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
            label: Some("Inversion Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Inversion Pipeline"),
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
            label: Some("Inversion sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let darken_darks_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Inversion Darken Darks Buffer"),
            contents: bytemuck::cast_slice(&[darken_darks]),
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
            label: Some("Inversion Bind Group"),
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
                        darken_darks_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        Self {
            enabled: true,
            darken_darks,
            pipeline,
            bind_group,
            bind_group_layout,
            darken_darks_buffer,
            sampler,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_darken_darks(&mut self, queue: &wgpu::Queue, darken_darks: f32) {
        self.darken_darks = darken_darks;
        queue.write_buffer(
            &self.darken_darks_buffer,
            0,
            bytemuck::cast_slice(&[darken_darks]),
        );
    }
}

impl SimpleComponent for InversionComponent {}

impl PipelineComponent for InversionComponent {
    fn encode_pass(&mut self, encoder: &mut wgpu::CommandEncoder, output_view: &wgpu::TextureView) {
        if !self.enabled {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Inversion pass"),
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

    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        _effect_view: Option<&wgpu::TextureView>,
    ) {
        // Simple stages ignore effect_view and use input_view as their single input
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Inversion Bind Group"),
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
                        self.darken_darks_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });
    }

    fn name(&self) -> &str {
        "Monochromatic Inversion"
    }

    fn component_type(&self) -> ComponentType {
        ComponentType::Simple
    }

    fn update_parameters(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.darken_darks_buffer,
            0,
            bytemuck::cast_slice(&[self.darken_darks]),
        );
    }
}
