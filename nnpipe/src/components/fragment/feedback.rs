//! # Feedback / Trail Component
//!
//! Temporal feedback effect that blends the current frame with up to 8 previous
//! frames, creating motion trails, ghosting, and echo effects. Each frame the
//! history buffer shifts — the newest frame enters slot 1, and all others age by
//! one position (slot 8 gets discarded).
//!
//! ## How It Works
//!
//! 1. The shader composites the current input with the weighted history frames
//! 2. The history buffer shifts: each frame moves to the next older slot
//! 3. The *clean* current input (not the composited result) is copied into slot 1,
//!    preventing infinite feedback loops
//!
//! ## Parameters
//!
//! | Parameter | Range | Description |
//! |-----------|-------|-------------|
//! | `persistence` | 0.0–1.0 | Opacity of the oldest visible frames |
//! | `frame_history` | 0.0–1.0 | How many of the 8 history slots contribute |
//!
//! ## Pipeline Builder Usage
//!
//! ```rust,ignore
//! PipelineBuilder::new()
//!     .feedback(config, 0.95, 1.0) // strong trails, full history depth
//!     .build(device)?;
//! ```
//!
//! ## Memory
//!
//! This component allocates 8 history textures plus an output texture at
//! construction time, so it uses ~9x the memory of a single intermediate texture
//! at the configured resolution.

use crate::pipeline::{ComponentType, PipelineComponent, SimpleComponent, TextureConfig};
use nannou::prelude::*;
use nannou::wgpu;

/// GPU uniform layout for feedback parameters. Padded to 16-byte alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FeedbackParams {
    /// How visible the oldest frames are (0.0 = invisible, 1.0 = full strength)
    persistence: f32,
    /// How many of the 8 history frames contribute (0.0 = no trail, 1.0 = all 8)
    frame_history: f32,
    _padding1: u32,
    _padding2: u32,
}

unsafe impl bytemuck::Pod for FeedbackParams {}
unsafe impl bytemuck::Zeroable for FeedbackParams {}

/// Temporal feedback trail component with an 8-frame history buffer.
///
/// Blends the current input with previous frames to create motion trails and
/// ghosting effects. The history buffer shifts each frame, and only the clean
/// (uncomposited) input is stored to prevent infinite feedback loops.
///
/// # Runtime Parameters
///
/// - [`set_persistence`](FeedbackComponent::set_persistence) — control oldest-frame visibility
/// - [`set_frame_history`](FeedbackComponent::set_frame_history) — control history depth
/// - [`set_enabled`](FeedbackComponent::set_enabled) / [`is_enabled`](FeedbackComponent::is_enabled) — toggle bypass
/// - [`clear_history`](FeedbackComponent::clear_history) — reset all history frames to black
pub struct FeedbackComponent {
    enabled: bool,
    persistence: f32,
    frame_history: f32,

    // GPU resources
    pipeline: wgpu::RenderPipeline, // Main feedback compositing pipeline
    passthrough_pipeline: wgpu::RenderPipeline, // Simple copy pipeline for clean input->history
    bind_group_output: wgpu::BindGroup, // For rendering to output (reads from history)
    bind_group_history: wgpu::BindGroup, // For rendering to history (reads from history)
    passthrough_bind_group: wgpu::BindGroup, // For copying input to history
    bind_group_layout: wgpu::BindGroupLayout,
    passthrough_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,

    // Frame history buffers (8 previous frames)
    history_textures: Vec<wgpu::Texture>,
    history_views: Vec<wgpu::TextureView>,

    // Own output texture that we can copy from
    #[allow(dead_code)]
    output_texture: wgpu::Texture,
    #[allow(dead_code)]
    output_view: wgpu::TextureView,
    output_config: TextureConfig,
}

impl FeedbackComponent {
    pub fn new(
        device: &wgpu::Device,
        output_config: TextureConfig,
        persistence: f32,
        frame_history: f32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Feedback Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fragment/feedback.wgsl").into(),
            ),
        });

        let mut entries = Vec::new();

        // Current frame texture (binding 0)
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });

        // 8 history frame textures (bindings 1-8)
        for i in 1..=8 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }

        // Sampler (binding 9)
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu_types::SamplerBindingType::Filtering),
            count: None,
        });

        // Parameters (binding 10)
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Feedback Bind Group Layout"),
            entries: &entries,
        });

        // Create simple passthrough layout (just input texture + sampler)
        let passthrough_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Passthrough Layout"),
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
            label: Some("Feedback Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let passthrough_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Passthrough Pipeline Layout"),
                bind_group_layouts: &[&passthrough_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Feedback Pipeline"),
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

        // Create passthrough shader and pipeline
        let passthrough_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Passthrough Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/passthrough.wgsl").into()),
        });

        let passthrough_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Passthrough Pipeline"),
            layout: Some(&passthrough_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &passthrough_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &passthrough_shader,
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
            label: Some("Feedback sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let params = FeedbackParams {
            persistence,
            frame_history,
            _padding1: 0,
            _padding2: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Feedback Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create 8 history textures for storing previous frames
        let mut history_textures = Vec::with_capacity(8);
        let mut history_views = Vec::with_capacity(8);

        for _i in 0..8 {
            let texture = nannou::wgpu::TextureBuilder::new()
                .size([output_config.width, output_config.height])
                .format(output_config.format)
                .usage(
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC,
                )
                .build(device);
            let view = texture.view().build();
            history_textures.push(texture);
            history_views.push(view);
        }

        // Create our own output texture that we can copy from
        let output_texture = nannou::wgpu::TextureBuilder::new()
            .size([output_config.width, output_config.height])
            .format(output_config.format)
            .usage(
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .build(device);
        let output_view = output_texture.view().build();

        // Create dummy textures for initial bind group
        let dummy_texture = nannou::wgpu::TextureBuilder::new()
            .size([2, 2])
            .format(output_config.format)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .build(device);
        let dummy_view = dummy_texture.view().build();

        // Create initial dummy bind groups (will be updated in finalize_bind_groups)
        let mut bind_entries = Vec::new();

        // Current frame (binding 0)
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&dummy_view),
        });

        // 8 history frames (bindings 1-8)
        for (i, history_view) in history_views.iter().enumerate().take(8) {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: wgpu::BindingResource::TextureView(history_view),
            });
        }

        // Sampler (binding 9)
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Sampler(&sampler),
        });

        // Parameters (binding 10)
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 10,
            resource: wgpu::BindingResource::Buffer(params_buffer.as_entire_buffer_binding()),
        });

        let bind_group_output = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback Output Bind Group"),
            layout: &bind_group_layout,
            entries: &bind_entries,
        });

        let bind_group_history = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback History Bind Group"),
            layout: &bind_group_layout,
            entries: &bind_entries,
        });

        // Create passthrough bind group (will be updated in finalize_bind_groups)
        let passthrough_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Passthrough Bind Group"),
            layout: &passthrough_layout,
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
            persistence,
            frame_history,
            pipeline,
            passthrough_pipeline,
            bind_group_output,
            bind_group_history,
            passthrough_bind_group,
            bind_group_layout,
            passthrough_layout,
            params_buffer,
            sampler,
            history_textures,
            history_views,
            output_texture,
            output_view,
            output_config,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set the persistence factor. Higher values make older frames more visible.
    /// Clamped to `[0.0, 1.0]`.
    pub fn set_persistence(&mut self, persistence: f32) {
        self.persistence = persistence.clamp(0.0, 1.0);
    }

    pub fn get_persistence(&self) -> f32 {
        self.persistence
    }

    /// Set how many history frames contribute. `0.0` = no trail, `1.0` = all 8 frames.
    /// Clamped to `[0.0, 1.0]`.
    pub fn set_frame_history(&mut self, frame_history: f32) {
        self.frame_history = frame_history.clamp(0.0, 1.0);
    }

    pub fn get_frame_history(&self) -> f32 {
        self.frame_history
    }

    /// Clear all 8 history textures to black, erasing all trail state.
    /// Creates and submits its own command encoder.
    pub fn clear_history(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Clear all 8 history textures to black
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Clear Feedback History"),
        });

        for (i, view) in self.history_views.iter().enumerate() {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("Clear History {}", i)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
        }

        queue.submit(Some(encoder.finish()));
    }

    fn update_params_buffer(&self, queue: &wgpu::Queue) {
        let params = FeedbackParams {
            persistence: self.persistence,
            frame_history: self.frame_history,
            _padding1: 0,
            _padding2: 0,
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }
}

impl SimpleComponent for FeedbackComponent {}

impl PipelineComponent for FeedbackComponent {
    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        _effect_view: Option<&wgpu::TextureView>,
    ) {
        // Create bind entries for current input + 8 history textures
        let mut bind_entries = Vec::new();

        // Current frame (binding 0)
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(input_view),
        });

        // 8 history frames (bindings 1-8)
        for i in 0..8 {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: wgpu::BindingResource::TextureView(&self.history_views[i]),
            });
        }

        // Sampler (binding 9)
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        });

        // Parameters (binding 10)
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 10,
            resource: wgpu::BindingResource::Buffer(self.params_buffer.as_entire_buffer_binding()),
        });

        // Create bind group for output pass
        self.bind_group_output = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback Output Bind Group"),
            layout: &self.bind_group_layout,
            entries: &bind_entries,
        });

        // Create bind group for history pass (same entries)
        self.bind_group_history = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Feedback History Bind Group"),
            layout: &self.bind_group_layout,
            entries: &bind_entries,
        });

        // Create passthrough bind group (just input + sampler for copying clean input to history)
        self.passthrough_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Passthrough Bind Group"),
            layout: &self.passthrough_layout,
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

    fn encode_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_output_view: &wgpu::TextureView,
    ) {
        if !self.enabled {
            return;
        }

        // Step 1: Render the feedback effect to the pipeline output first
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Feedback pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: pipeline_output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group_output, &[]);
            pass.draw(0..3, 0..1);
        }

        // Step 2: Shift all history frames down by one position (age them)
        // history_frame_1 -> history_frame_2, history_frame_2 -> history_frame_3, etc.
        // history_frame_8 gets discarded, and history_frame_1 gets the new input
        for i in (1..8).rev() {
            // Copy history_textures[i-1] to history_textures[i] (aging the frames)
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.history_textures[i - 1],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.history_textures[i],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.output_config.width,
                    height: self.output_config.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Step 3: Copy ONLY the clean current frame (input) to history_frame_1 (newest position)
        // NOT the composited result, to avoid infinite feedback loops
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Copy clean input to history_frame_1"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.history_views[0], // Always write to position 0 (history_frame_1)
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            // Use passthrough pipeline to copy just the clean input
            pass.set_pipeline(&self.passthrough_pipeline);
            pass.set_bind_group(0, &self.passthrough_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    fn name(&self) -> &str {
        "Feedback Trail"
    }

    fn update_parameters(&mut self, queue: &wgpu::Queue) {
        self.update_params_buffer(queue);
    }

    fn component_type(&self) -> ComponentType {
        ComponentType::Simple
    }
}
