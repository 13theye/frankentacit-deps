//! # Color Key Extraction Component
//!
//! Isolates pixels that match a target color within a configurable distance
//! threshold. Pixels outside the threshold are zeroed out, creating a color-based
//! mask. Useful for extracting specific hues from a scene for selective effects.
//!
//! ## Parameters
//!
//! | Parameter | Range | Description |
//! |-----------|-------|-------------|
//! | `target_color` | `[R, G, B]` 0.0–1.0 | The color to match against |
//! | `threshold` | 0.0–1.0 | Maximum Euclidean distance in RGB space for a match |
//! | `intensity` | 0.0–2.0 | Output brightness multiplier for matched pixels |
//!
//! ## Pipeline Builder Usage
//!
//! ```rust,ignore
//! // Extract reddish pixels and blur them for a selective glow
//! PipelineBuilder::new()
//!     .color_key_extract(config, [1.0, 0.2, 0.1], 0.3, 1.5)
//!     .gaussian_blur_passes(config, 2, 2.0, 5.0)
//!     .simple_additive_composite(config, 1.0)
//!     .build(device)?;
//! ```

use crate::pipeline::{ComponentType, PipelineComponent, SimpleComponent, TextureConfig};
use nannou::prelude::*;
use nannou::wgpu;

/// GPU uniform layout for color key parameters. Padded to 16-byte alignment.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct ColorKeyUniforms {
    target_color: [f32; 3],
    threshold: f32,
    intensity: f32,
    _padding: [u32; 3], // Align to 16 bytes
}

// Manual bytemuck implementation since we have non-standard padding
unsafe impl bytemuck::Pod for ColorKeyUniforms {}
unsafe impl bytemuck::Zeroable for ColorKeyUniforms {}

/// Color key extraction component that isolates pixels matching a target color.
///
/// Compares each pixel's RGB value against `target_color` using Euclidean distance.
/// Pixels within `threshold` distance are kept (scaled by `intensity`); the rest
/// become transparent black.
///
/// # Runtime Parameters
///
/// - [`set_target_color`](ColorKeyComponent::set_target_color) — change the color to match
/// - [`set_threshold`](ColorKeyComponent::set_threshold) — widen or narrow the match range
/// - [`set_intensity`](ColorKeyComponent::set_intensity) — scale the output brightness
pub struct ColorKeyComponent {
    enabled: bool,
    target_color: [f32; 3],
    threshold: f32,
    intensity: f32,

    // GPU resources
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl ColorKeyComponent {
    pub fn new(
        device: &wgpu::Device,
        output_config: TextureConfig,
        target_color: [f32; 3],
        threshold: f32,
        intensity: f32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Color Key Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/post/color_key.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Color Key Bind Group Layout"),
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
            label: Some("Color Key Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Color Key Pipeline"),
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
            label: Some("Color Key sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniforms = ColorKeyUniforms {
            target_color,
            threshold,
            intensity,
            _padding: [0; 3],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Color Key Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
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
            label: Some("Color Key Bind Group"),
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
                        uniform_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        Self {
            enabled: true,
            target_color,
            threshold,
            intensity,
            pipeline,
            bind_group,
            bind_group_layout,
            uniform_buffer,
            sampler,
        }
    }

    /// Change the target color at runtime. Accepts RGB in `[0.0, 1.0]` range.
    pub fn set_target_color(&mut self, queue: &wgpu::Queue, target_color: [f32; 3]) {
        self.target_color = target_color;
        self.update_uniforms(queue);
    }

    /// Change the color-distance threshold at runtime. Larger values match more pixels.
    pub fn set_threshold(&mut self, queue: &wgpu::Queue, threshold: f32) {
        self.threshold = threshold;
        self.update_uniforms(queue);
    }

    /// Change the output intensity at runtime. Values > 1.0 overbrighten matched pixels.
    pub fn set_intensity(&mut self, queue: &wgpu::Queue, intensity: f32) {
        self.intensity = intensity;
        self.update_uniforms(queue);
    }

    fn update_uniforms(&self, queue: &wgpu::Queue) {
        let uniforms = ColorKeyUniforms {
            target_color: self.target_color,
            threshold: self.threshold,
            intensity: self.intensity,
            _padding: [0; 3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }
}

impl SimpleComponent for ColorKeyComponent {}

impl PipelineComponent for ColorKeyComponent {
    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        _effect_view: Option<&wgpu::TextureView>,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Color Key Bind Group"),
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
                        self.uniform_buffer.as_entire_buffer_binding(),
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
            label: Some("Color Key pass"),
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
        "ColorKey"
    }

    fn update_parameters(&mut self, queue: &wgpu::Queue) {
        self.update_uniforms(queue);
    }

    fn component_type(&self) -> ComponentType {
        ComponentType::Simple
    }
}
