//! # Blur Component
//!
//! A separable Gaussian blur that processes one axis per pass. For a proper 2D
//! Gaussian blur, chain a horizontal pass followed by a vertical pass — this is
//! mathematically equivalent to a full 2D kernel but runs in O(n) instead of O(n^2).
//!
//! ## Parameters
//!
//! | Parameter | Range | Description |
//! |-----------|-------|-------------|
//! | `direction` | `[1,0]` or `[0,1]` | Blur axis — horizontal or vertical |
//! | `adaptive_scaling` | 0.0+ | Multiplier for the blur radius based on resolution |
//! | `max_radius` | 1.0+ | Upper bound on the kernel radius in texels |
//!
//! ## Pipeline Builder Usage
//!
//! ```rust,ignore
//! // Two-pass blur (horizontal + vertical) at half resolution
//! PipelineBuilder::new()
//!     .downsample(lo_config)
//!     .gaussian_blur_passes(lo_config, 2, 2.0, 10.0) // 2 H+V pass pairs
//!     .resample(hi_config)
//!     .build(device)?;
//! ```

use crate::pipeline::{ComponentType, PipelineComponent, SimpleComponent, TextureConfig};
use nannou::prelude::*;
use nannou::wgpu;

/// Separable Gaussian blur component that processes one axis per pass.
///
/// Each instance blurs along a single direction. Chain horizontal and vertical
/// instances for a full 2D blur. [`PipelineBuilder::gaussian_blur_passes`]
/// handles this automatically.
///
/// # Runtime Parameters
///
/// - [`set_direction`](BlurComponent::set_direction) — change the blur axis
/// - [`set_adaptive_scaling`](BlurComponent::set_adaptive_scaling) — adjust resolution scaling
/// - [`set_max_radius`](BlurComponent::set_max_radius) — cap the kernel size
pub struct BlurComponent {
    enabled: bool,
    direction: [f32; 2],
    adaptive_scaling: f32,
    max_radius: f32,

    // GPU resources
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    direction_buffer: wgpu::Buffer,
    adaptive_scaling_buffer: wgpu::Buffer,
    max_radius_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl BlurComponent {
    pub fn new(
        device: &wgpu::Device,
        output_config: TextureConfig,
        direction: [f32; 2],
        adaptive_scaling: f32,
        max_radius: f32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/post/blur.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blur Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("Blur Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur Pipeline"),
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
            label: Some("Blur sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let direction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blur Direction Buffer"),
            contents: bytemuck::cast_slice(&direction),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let adaptive_scaling_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Adaptive Scaling Buffer"),
                contents: bytemuck::cast_slice(&[adaptive_scaling]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let max_radius_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Max Radius Buffer"),
            contents: bytemuck::cast_slice(&[max_radius]),
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
            label: Some("Blur Bind Group"),
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
                        direction_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        adaptive_scaling_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        max_radius_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        Self {
            enabled: true,
            direction,
            adaptive_scaling,
            max_radius,
            pipeline,
            bind_group,
            bind_group_layout,
            direction_buffer,
            adaptive_scaling_buffer,
            max_radius_buffer,
            sampler,
        }
    }

    /// Create a horizontal blur pass (direction `[1.0, 0.0]`).
    pub fn horizontal(
        device: &wgpu::Device,
        output_config: TextureConfig,
        adaptive_scaling: f32,
        max_radius: f32,
    ) -> Self {
        Self::new(
            device,
            output_config,
            [1.0, 0.0],
            adaptive_scaling,
            max_radius,
        )
    }

    /// Create a vertical blur pass (direction `[0.0, 1.0]`).
    pub fn vertical(
        device: &wgpu::Device,
        output_config: TextureConfig,
        adaptive_scaling: f32,
        max_radius: f32,
    ) -> Self {
        Self::new(
            device,
            output_config,
            [0.0, 1.0],
            adaptive_scaling,
            max_radius,
        )
    }

    /// Change the blur direction at runtime. Writes directly to the GPU uniform buffer.
    pub fn set_direction(&mut self, queue: &wgpu::Queue, direction: [f32; 2]) {
        self.direction = direction;
        queue.write_buffer(&self.direction_buffer, 0, bytemuck::cast_slice(&direction));
    }

    /// Change the adaptive scaling factor at runtime. Higher values increase blur spread.
    pub fn set_adaptive_scaling(&mut self, queue: &wgpu::Queue, scaling: f32) {
        self.adaptive_scaling = scaling;
        queue.write_buffer(
            &self.adaptive_scaling_buffer,
            0,
            bytemuck::cast_slice(&[scaling]),
        );
    }

    /// Change the maximum kernel radius at runtime. Caps how far the blur samples extend.
    pub fn set_max_radius(&mut self, queue: &wgpu::Queue, radius: f32) {
        self.max_radius = radius;
        queue.write_buffer(&self.max_radius_buffer, 0, bytemuck::cast_slice(&[radius]));
    }
}

impl SimpleComponent for BlurComponent {}

impl PipelineComponent for BlurComponent {
    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        _effect_view: Option<&wgpu::TextureView>,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur Bind Group"),
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
                        self.direction_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        self.adaptive_scaling_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        self.max_radius_buffer.as_entire_buffer_binding(),
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
            label: Some("Blur pass"),
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
        "Gaussian Blur"
    }

    fn update_parameters(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.direction_buffer,
            0,
            bytemuck::cast_slice(&self.direction),
        );
        queue.write_buffer(
            &self.adaptive_scaling_buffer,
            0,
            bytemuck::cast_slice(&[self.adaptive_scaling]),
        );
        queue.write_buffer(
            &self.max_radius_buffer,
            0,
            bytemuck::cast_slice(&[self.max_radius]),
        );
    }

    fn component_type(&self) -> ComponentType {
        ComponentType::Simple
    }
}
