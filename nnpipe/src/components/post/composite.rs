//! # Composite Components
//!
//! Pipeline components for combining multiple textures using various blend modes.
//!
//! ## Alpha Handling
//!
//! - **Input**: Straight alpha (RGB and A are independent)
//!   This is the format output by nnpipe's particle/segment renderers and Nannou Draw.
//!
//! - **Output**: Premultiplied alpha (RGB × A)
//!   Ready for subsequent compositing or final display.
//!
//! The composite shader premultiplies inputs internally before blending. This allows
//! direct compositing of straight-alpha textures without a separate premultiply pass.
//!
//! ## Components
//!
//! - `SimpleCompositeComponent`: General-purpose compositor with selectable blend modes
//! - `BloomCompositeComponent`: Specialized compositor for bloom effects with tone mapping
//!
//! ## Blend Modes
//!
//! | Mode | Use Case | Formula |
//! |------|----------|---------|
//! | Over | Standard compositing | effect + scene × (1 - effect.a) |
//! | Add | Glowing/light effects | scene + effect |
//! | Lighten | Combine particles, brightest wins | max by luminance |
//! | Screen | Lighten without oversaturation | 1 - (1-A)(1-B) |
//! | Multiply | Darken/shadow effects | A × B |
//! | Overlay | Contrast enhancement | Multiply or Screen based on value |
//!
//! ## Pipeline Builder Usage
//!
//! ```rust,ignore
//! // Combine two particle layers, brightest pixel wins
//! // Input: straight alpha, Output: premultiplied alpha
//! PipelineBuilder::new()
//!     .input_textures(&["particles_a", "particles_b"])
//!     .output_texture("combined")
//!     .simple_lighten_composite(config, 1.0)
//!     .build(device)
//!
//! // Standard alpha compositing for UI overlay
//! PipelineBuilder::new()
//!     .input_textures(&["scene", "ui_overlay"])
//!     .simple_over_composite(config, 1.0)
//!     .build(device)
//! ```

use crate::pipeline::{ComponentType, CompositorComponent, PipelineComponent, TextureConfig};
use nannou::prelude::*;
use nannou::wgpu;

/// Blend modes for texture compositing.
///
/// All modes accept straight alpha inputs and output premultiplied alpha.
#[derive(Clone, Copy, Debug)]
pub enum BlendMode {
    /// Porter-Duff "over" - standard alpha compositing
    Over,
    /// Additive blending - layers brighten when overlapping
    Add,
    /// Screen blending - lightens without oversaturation
    Screen,
    /// Multiply blending - darkens by multiplying colors
    Multiply,
    /// Overlay blending - enhances contrast
    Overlay,
    /// Lighten - takes the brighter pixel by luminance
    Lighten,
}

impl BlendMode {
    pub fn to_u32(self) -> u32 {
        match self {
            BlendMode::Over => 0,
            BlendMode::Add => 1,
            BlendMode::Screen => 2,
            BlendMode::Multiply => 3,
            BlendMode::Overlay => 4,
            BlendMode::Lighten => 5,
        }
    }
}

/// Specialized bloom compositor that additively blends an effect layer onto a scene
/// with intensity and curve controls for tone-mapped glow.
///
/// The `intensity` parameter scales the bloom contribution, while `intensity_curve`
/// applies a power curve to the bloom brightness, allowing fine control over how
/// bright areas glow versus dim areas.
///
/// Implements [`CompositorComponent`] — requires two input textures (scene + bloom effect).
///
/// # Runtime Parameters
///
/// - [`set_intensity`](BloomCompositeComponent::set_intensity) — scale bloom brightness
/// - [`set_intensity_curve`](BloomCompositeComponent::set_intensity_curve) — adjust the power curve
pub struct BloomCompositeComponent {
    enabled: bool,
    intensity: f32,
    intensity_curve: f32,

    // GPU resources
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    intensity_buffer: wgpu::Buffer,
    intensity_curve_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl BloomCompositeComponent {
    pub fn new(
        device: &wgpu::Device,
        output_config: TextureConfig,
        intensity: f32,
        intensity_curve: f32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/post/composite_bloom.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Composite Bind Group Layout"),
            entries: &[
                // Scene texture
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
                // Effect texture (bloom, etc.)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu_types::SamplerBindingType::Filtering),
                    count: None,
                },
                // Intensity uniform
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
                // Intensity curve uniform
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
            label: Some("Bloom Composite Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Composite Pipeline"),
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
            label: Some("Bloom Composite sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let intensity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bloom Composite Intensity Buffer"),
            contents: bytemuck::cast_slice(&[intensity]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let intensity_curve_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bloom Composite Intensity Curve Buffer"),
            contents: bytemuck::cast_slice(&[intensity_curve]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create dummy bind group (will be updated when processing)
        let dummy_texture = nannou::wgpu::TextureBuilder::new()
            .size([2, 2])
            .format(output_config.format)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .build(device);
        let dummy_view = dummy_texture.view().build();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom Composite Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        intensity_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        intensity_curve_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        Self {
            enabled: true,
            intensity,
            intensity_curve,
            pipeline,
            bind_group,
            bind_group_layout,
            intensity_buffer,
            intensity_curve_buffer,
            sampler,
        }
    }

    /// Create a bloom compositor with linear (no curve) intensity. Shorthand for `curve = 1.0`.
    pub fn additive(device: &wgpu::Device, output_config: TextureConfig, intensity: f32) -> Self {
        Self::new(device, output_config, intensity, 1.0)
    }

    /// Create a bloom compositor with a custom intensity curve for tone-mapped glow.
    pub fn with_curve(
        device: &wgpu::Device,
        output_config: TextureConfig,
        intensity: f32,
        intensity_curve: f32,
    ) -> Self {
        Self::new(device, output_config, intensity, intensity_curve)
    }

    pub fn set_intensity(&mut self, queue: &wgpu::Queue, intensity: f32) {
        self.intensity = intensity;
        queue.write_buffer(
            &self.intensity_buffer,
            0,
            bytemuck::cast_slice(&[intensity]),
        );
    }

    pub fn set_intensity_curve(&mut self, queue: &wgpu::Queue, curve: f32) {
        self.intensity_curve = curve;
        queue.write_buffer(
            &self.intensity_curve_buffer,
            0,
            bytemuck::cast_slice(&[curve]),
        );
    }
}

impl PipelineComponent for BloomCompositeComponent {
    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        effect_view: Option<&wgpu::TextureView>,
    ) {
        let effect = effect_view.expect("BloomCompositeComponent requires effect view");
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom Composite Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view), // scene
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(effect), // effect
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        self.intensity_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        self.intensity_curve_buffer.as_entire_buffer_binding(),
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
            label: Some("Bloom Composite pass"),
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
        "Bloom Composite"
    }

    fn component_type(&self) -> ComponentType {
        ComponentType::Compositor
    }

    fn update_parameters(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.intensity_buffer,
            0,
            bytemuck::cast_slice(&[self.intensity]),
        );
        queue.write_buffer(
            &self.intensity_curve_buffer,
            0,
            bytemuck::cast_slice(&[self.intensity_curve]),
        );
    }
}

impl CompositorComponent for BloomCompositeComponent {}

/// General-purpose texture compositor with selectable blend modes.
///
/// Blends two input textures (scene + effect) using the configured [`BlendMode`],
/// scaled by an `intensity` parameter. Accepts straight alpha inputs and outputs
/// premultiplied alpha — the conversion happens inside the shader.
///
/// This is the most commonly used compositor in Nnpipe, suitable for combining
/// particle layers, applying post-processed effects back onto the scene, or
/// any standard compositing operation.
///
/// Implements [`CompositorComponent`] — requires two input textures.
///
/// # Runtime Parameters
///
/// - [`set_intensity`](SimpleCompositeComponent::set_intensity) — scale the effect contribution
/// - [`set_blend_mode`](SimpleCompositeComponent::set_blend_mode) — switch blend modes without rebuilding
pub struct SimpleCompositeComponent {
    enabled: bool,
    blend_mode: BlendMode,
    intensity: f32,

    // GPU resources
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    intensity_buffer: wgpu::Buffer,
    blend_mode_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl SimpleCompositeComponent {
    pub fn new(
        device: &wgpu::Device,
        blend_mode: BlendMode,
        output_config: TextureConfig,
        intensity: f32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Simple Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/post/composite_simple.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Simple Composite Bind Group Layout"),
            entries: &[
                // Scene texture
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
                // Effect texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu_types::SamplerBindingType::Filtering),
                    count: None,
                },
                // Intensity uniform
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
                // Blend mode uniform
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
            label: Some("Simple Composite Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Simple Composite Pipeline"),
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
            label: Some("Simple Composite sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let intensity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simple Composite Intensity Buffer"),
            contents: bytemuck::cast_slice(&[intensity]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let blend_mode_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simple Composite Blend Mode Buffer"),
            contents: bytemuck::cast_slice(&[blend_mode.to_u32()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create dummy bind group (will be updated when processing)
        let dummy_texture = nannou::wgpu::TextureBuilder::new()
            .size([2, 2])
            .format(output_config.format)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .build(device);
        let dummy_view = dummy_texture.view().build();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Simple Composite Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dummy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        intensity_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        blend_mode_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        Self {
            enabled: true,
            blend_mode,
            intensity,
            pipeline,
            bind_group,
            bind_group_layout,
            intensity_buffer,
            blend_mode_buffer,
            sampler,
        }
    }

    /// Convenience constructor for additive blending (good for glowing/light effects).
    pub fn additive(device: &wgpu::Device, output_config: TextureConfig, intensity: f32) -> Self {
        Self::new(device, BlendMode::Add, output_config, intensity)
    }

    /// Convenience constructor for screen blending (lightens without oversaturation).
    pub fn screen(device: &wgpu::Device, output_config: TextureConfig, intensity: f32) -> Self {
        Self::new(device, BlendMode::Screen, output_config, intensity)
    }

    /// Convenience constructor for multiply blending (darkens/shadow effects).
    pub fn multiply(device: &wgpu::Device, output_config: TextureConfig, intensity: f32) -> Self {
        Self::new(device, BlendMode::Multiply, output_config, intensity)
    }

    /// Convenience constructor for overlay blending (contrast enhancement).
    pub fn overlay(device: &wgpu::Device, output_config: TextureConfig, intensity: f32) -> Self {
        Self::new(device, BlendMode::Overlay, output_config, intensity)
    }

    /// Change the effect intensity at runtime. Scales the effect layer's contribution.
    pub fn set_intensity(&mut self, queue: &wgpu::Queue, intensity: f32) {
        self.intensity = intensity;
        queue.write_buffer(
            &self.intensity_buffer,
            0,
            bytemuck::cast_slice(&[intensity]),
        );
    }

    /// Switch the blend mode at runtime without rebuilding the pipeline.
    /// The mode is passed to the shader as a uniform integer.
    pub fn set_blend_mode(&mut self, queue: &wgpu::Queue, blend_mode: BlendMode) {
        self.blend_mode = blend_mode;
        queue.write_buffer(
            &self.blend_mode_buffer,
            0,
            bytemuck::cast_slice(&[blend_mode.to_u32()]),
        );
    }
}

impl PipelineComponent for SimpleCompositeComponent {
    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        effect_view: Option<&wgpu::TextureView>,
    ) {
        let effect = effect_view.expect("SimpleCompositeComponent requires effect view");
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Simple Composite Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view), // scene
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(effect), // effect
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        self.intensity_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(
                        self.blend_mode_buffer.as_entire_buffer_binding(),
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
            label: Some("Simple Composite pass"),
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
        "Simple Composite"
    }

    fn component_type(&self) -> ComponentType {
        ComponentType::Compositor
    }

    fn update_parameters(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.intensity_buffer,
            0,
            bytemuck::cast_slice(&[self.intensity]),
        );
        queue.write_buffer(
            &self.blend_mode_buffer,
            0,
            bytemuck::cast_slice(&[self.blend_mode.to_u32()]),
        );
    }
}

impl CompositorComponent for SimpleCompositeComponent {}
