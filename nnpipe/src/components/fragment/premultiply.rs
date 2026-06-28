//! # Premultiply Component
//!
//! Converts textures from straight alpha to premultiplied alpha format.
//!
//! ## When to Use
//!
//! Use this component to convert any texture from straight alpha to premultiplied
//! alpha before compositing. All nnpipe renderers (particles, segments) and
//! external sources (Nannou Draw) output straight alpha. This component marks
//! the boundary between rendering and compositing.
//!
//! Common use cases:
//! - Particle and segment renderer output
//! - Nannou Draw output (UI, text, shapes, masks)
//! - Loaded image files (PNG, etc.)
//! - Any texture before entering compositing operations
//!
//! ## Pipeline Builder Usage
//!
//! ```rust,ignore
//! // Particles and masks render to same texture (both straight alpha)
//! particle_renderer.encode_only(&mut encoder, count, particles_view);
//! mask.draw(&rendering.draw);
//! rendering.encode_draw_commands_into(device, &mut encoder, "particles");
//!
//! // Premultiply before compositing
//! PipelineBuilder::new()
//!     .input_texture("particles")           // straight alpha
//!     .output_texture("particles_premult")  // premultiplied alpha
//!     .premultiply(config)
//!     .build(device)
//! ```

use crate::pipeline::{ComponentType, PipelineComponent, SimpleComponent, TextureConfig};
use nannou::wgpu;

/// Alpha premultiplication component — converts `(R, G, B, A)` to `(R*A, G*A, B*A, A)`.
///
/// Uses `BlendState::REPLACE` so the output is an exact format conversion with no
/// double-blending. This component has no runtime parameters.
///
/// Note: In most cases you do not need this component explicitly. The composite
/// shaders ([`SimpleCompositeComponent`], [`BloomCompositeComponent`]) handle
/// premultiplication internally when accepting straight-alpha inputs. Use this
/// component only when feeding textures to external consumers that expect
/// premultiplied alpha.
pub struct PremultiplyComponent {
    enabled: bool,

    // GPU resources
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl PremultiplyComponent {
    pub fn new(device: &wgpu::Device, output_config: TextureConfig) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Premultiply Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fragment/premultiply.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Premultiply Bind Group Layout"),
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
            label: Some("Premultiply Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Use REPLACE blend state since we're converting alpha format, not compositing
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Premultiply Pipeline"),
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
                    blend: Some(wgpu::BlendState::REPLACE), // Direct write, no blending
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
            label: Some("Premultiply sampler"),
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
            label: Some("Premultiply Bind Group"),
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

impl SimpleComponent for PremultiplyComponent {}

impl PipelineComponent for PremultiplyComponent {
    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        _effect_view: Option<&wgpu::TextureView>,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Premultiply Bind Group"),
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
            label: Some("Premultiply pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
        "Premultiply"
    }

    fn update_parameters(&mut self, _queue: &wgpu::Queue) {
        // No parameters to update
    }

    fn component_type(&self) -> ComponentType {
        ComponentType::Simple
    }
}
