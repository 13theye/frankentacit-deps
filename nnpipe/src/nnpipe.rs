// src/nnpipe.rs
//
// Modular texture rendering and post-processing pipeline

use crate::{Pipeline, TextureConfig};
use nannou::prelude::*;
use nannou::wgpu;
use std::collections::HashMap;

pub struct Nnpipe {
    /// A Nannou Draw API object for drawing primitives to the scene texture.
    pub draw: nannou::draw::Draw,
    /// Nannou's Draw API renderer for drawing to the scene texture.
    /// Instead of having the Nannou API draw directly to the output texture,
    /// Nnpipe grabs Nannou::draw's command encoder, adds its own render pass(es), and handles
    /// submission in order to save on API calls.
    pub draw_renderer: nannou::draw::Renderer,

    // Primary textures
    //
    /// The default render target for content.
    /// Single pipeline systems default to writing to this texture
    pub scene_texture: wgpu::Texture,
    ///
    /// The texture that is wired to draw to the screen. The final node of a pipeline should
    /// write to this texture.
    pub output_texture: wgpu::Texture,

    // Texture views
    scene_view: wgpu::TextureView,
    output_view: wgpu::TextureView,

    // Modular effects pipeline system - the main pipeline chain for effects
    /// Consists of a sequence of PipelineComponents, all of which
    /// are designed to accept an input texture and write to an intermediate texture.
    /// Part of the legacy single-pipeline system
    pub effects_pipeline: Option<Pipeline>,

    // Simple passthrough pipeline for direct rendering
    passthrough_pipeline: wgpu::RenderPipeline,
    passthrough_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    sampler: wgpu::Sampler,

    // Named texture registry for multi-pipeline coordination
    named_textures: HashMap<String, wgpu::Texture>,
    named_views: HashMap<String, wgpu::TextureView>,

    // Named pipeline registry for multi-pipeline workflows
    multi_pipeline: HashMap<String, Pipeline>,
}

impl Nnpipe {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, samples: u32) -> Self {
        // Define formats
        let hi_format = wgpu::TextureFormat::Rgba16Float;

        // Create primary textures
        let scene_texture = create_render_texture(device, width, height, samples, hi_format);
        let composite_texture = create_render_texture(device, width, height, 1, hi_format);

        // Create draw and draw renderer
        let draw = nannou::draw::Draw::new();
        let draw_renderer = nannou::draw::RendererBuilder::new()
            .glyph_cache_size([2048, 2048])
            .build_from_texture_descriptor(device, scene_texture.descriptor());

        // Create texture views
        let scene_view = scene_texture.view().build();
        let composite_view = composite_texture.view().build();

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Pipeline sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // No default effects in the effects pipeline
        let effects_pipeline = None;

        // Create shader module for passthrough
        let passthrough_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Passthrough Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/passthrough.wgsl").into()),
        });

        // Create bind group layout for passthrough
        let passthrough_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Passthrough Bind Group Layout"),
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

        // Create pipeline layout
        let passthrough_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Passthrough Pipeline Layout"),
                bind_group_layouts: &[&passthrough_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create render pipeline
        let passthrough_pipeline = create_render_pipeline(
            device,
            &passthrough_pipeline_layout,
            &passthrough_shader,
            "Passthrough Pipeline",
            hi_format,
        );

        // Create bind group
        let passthrough_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Passthrough Bind Group"),
            layout: &passthrough_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            draw,
            draw_renderer,
            scene_texture,
            output_texture: composite_texture,
            scene_view,
            output_view: composite_view,
            effects_pipeline,
            passthrough_pipeline,
            passthrough_bind_group,
            sampler,
            named_textures: HashMap::new(),
            named_views: HashMap::new(),
            multi_pipeline: HashMap::new(),
        }
    }

    /***************** Typical rendering pipeline steps ***********************/

    /// Step 1: Creates a shared command encoder for Nannou Draw, Nnpipe Renders,
    /// and Nnpipe Post-processing to write to.
    pub fn create_command_encoder(&self, device: &wgpu::Device) -> wgpu::CommandEncoder {
        let ce_desc = wgpu::CommandEncoderDescriptor {
            label: Some("Scene renderer"),
        };

        device.create_command_encoder(&ce_desc)
    }

    /// Step 2: Create encoder commands for the internal Draw object to internal scene texture
    pub fn encode_draw_commands(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.draw_renderer.encode_render_pass(
            device,
            encoder,
            &self.draw,
            1.0,
            self.scene_texture.size(),
            &self.scene_view,
            None,
        );
    }

    pub fn encode_draw_commands_into(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture_view_name: &str,
    ) {
        let Some(texture_view) = self.get_named_texture(texture_view_name) else {
            return;
        };

        self.draw_renderer.encode_render_pass(
            device,
            encoder,
            &self.draw,
            1.0,
            texture_view.size(),
            &texture_view.clone(),
            None,
        );
    }

    // (Step 3: call encode_into on any NNpipe renderers)

    /// Step 4: Encode post-processing into an existing command encoder
    /// The method accesses the default behavior of using the internal scene and output views as input and output
    pub fn encode_post_process(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if let Some(effects_pipeline) = &mut self.effects_pipeline {
            effects_pipeline.encode_into(device, encoder, &self.scene_view, &self.output_view);
        } else {
            self.encode_passthrough_to_view(encoder, &self.output_view);
        }
    }

    /// Last step: Submit encoder commands to the GPU
    pub fn submit_command_encoder(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: wgpu::CommandEncoder,
    ) {
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Poll);
    }

    /// Experimental support for multiple command encoders to be submitted in parallel.
    /// This is generally not faster than a single encoder per frame.
    pub fn submit_command_encoders_par(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoders: Vec<wgpu::CommandEncoder>,
    ) {
        let mut buffers = Vec::new();
        for encoder in encoders {
            buffers.push(encoder.finish());
        }
        queue.submit(buffers);
        device.poll(wgpu::Maintain::Poll);
    }

    /***************** Named Texture Management ***********************/

    /// Get a named texture view, or None if it doesn't exist
    pub fn get_named_texture(&self, name: &str) -> Option<&wgpu::TextureView> {
        self.named_views.get(name)
    }

    /// Create a new named texture with the given configuration
    pub fn create_named_texture(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        config: TextureConfig,
    ) -> &wgpu::TextureView {
        let mut usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

        // Only add STORAGE_BINDING for formats that support it
        if supports_storage_binding(config.format) {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }

        let texture = nannou::wgpu::TextureBuilder::new()
            .size([config.width, config.height])
            .format(config.format)
            .usage(usage)
            .build(device);

        let view = texture.view().build();

        // Store both texture and view
        self.named_textures.insert(name.to_string(), texture);
        self.named_views.insert(name.to_string(), view);

        // Return reference to the view
        self.named_views.get(name).unwrap()
    }

    /// Get all named texture names
    pub fn list_named_textures(&self) -> Vec<&String> {
        self.named_textures.keys().collect()
    }

    /// Clear a specific named texture by adding a clear pass to the encoder
    pub fn encode_clear_named_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        clear_color: wgpu::Color,
    ) -> Result<(), String> {
        let texture_view = self
            .named_views
            .get(name)
            .ok_or_else(|| format!("Named texture '{}' not found", name))?;

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("Clear Named Texture '{}' Pass", name)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
        }

        Ok(())
    }

    /// Clear all textures by adding clear passes to the encoder
    pub fn encode_clear_all_textures(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        clear_color: wgpu::Color,
    ) {
        // Create a vector of all texture views to clear
        let mut views_to_clear = Vec::new();

        // Add all named texture views
        for view in self.named_views.values() {
            views_to_clear.push(view);
        }

        // Add internal views
        views_to_clear.push(&self.scene_view);
        views_to_clear.push(&self.output_view);

        // Clear each texture view
        for (i, view) in views_to_clear.iter().enumerate() {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("Clear All Textures Pass {}", i)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
        }
    }

    /// Execute a pipeline using named textures from the registry
    /// Falls back to explicit input/output views if named textures are not specified
    pub fn encode_pipeline_with_named_textures(
        &self,
        pipeline: &mut Pipeline,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        // Determine input view
        let input_view = if let Some(input_name) = pipeline.input_texture_name() {
            if let Some(view) = self.get_named_texture(input_name) {
                view
            } else {
                return Err(format!(
                    "Input texture '{}' not found in registry",
                    input_name
                ));
            }
        } else {
            &self.scene_view
        };

        // Determine output view
        let output_view = if let Some(output_name) = pipeline.output_texture_name() {
            if let Some(view) = self.get_named_texture(output_name) {
                view
            } else {
                return Err(format!(
                    "Output texture '{}' not found in registry",
                    output_name
                ));
            }
        } else {
            &self.output_view
        };

        // Execute the pipeline
        pipeline.encode_into(device, encoder, input_view, output_view);
        Ok(())
    }

    /// Execute multiple pipelines in sequence using named textures
    pub fn encode_pipelines_with_named_textures(
        &self,
        pipelines: &mut [Pipeline],
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        for pipeline in pipelines {
            self.encode_pipeline_with_named_textures(pipeline, device, encoder)?;
        }
        Ok(())
    }

    /***************** Named Pipeline Management ***********************/

    /// Add a named pipeline to the registry
    pub fn add_multi_pipeline(&mut self, pipeline_name: &str, pipeline: Pipeline) {
        self.multi_pipeline
            .insert(pipeline_name.to_owned(), pipeline);
    }

    /// Get a reference to a named pipeline
    pub fn get_named_pipeline(&mut self, pipeline_name: &str) -> Option<&mut Pipeline> {
        self.multi_pipeline.get_mut(pipeline_name)
    }

    /// Get all named pipeline names
    pub fn list_named_pipelines(&self) -> Vec<&String> {
        self.multi_pipeline.keys().collect()
    }

    /// Execute a specific named pipeline
    pub fn execute_named_pipeline(
        &mut self,
        name: &str,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        // Check if pipeline exists first
        if !self.multi_pipeline.contains_key(name) {
            return Err(format!("Named pipeline '{}' not found", name));
        }

        // Get the pipeline and execute it
        let pipeline = self.multi_pipeline.get_mut(name).unwrap();

        // Determine input view
        let input_view = if let Some(input_name) = pipeline.input_texture_name() {
            if let Some(view) = self.named_views.get(input_name) {
                view
            } else {
                return Err(format!(
                    "Input texture '{}' not found in registry",
                    input_name
                ));
            }
        } else {
            &self.scene_view
        };

        // Determine output view
        let output_view = if let Some(output_name) = pipeline.output_texture_name() {
            if let Some(view) = self.named_views.get(output_name) {
                view
            } else {
                return Err(format!(
                    "Output texture '{}' not found in registry",
                    output_name
                ));
            }
        } else {
            &self.output_view
        };

        // Execute the pipeline - use named texture support if the pipeline has named textures
        if pipeline.input_texture_name().is_some()
            || pipeline.output_texture_name().is_some()
            || !pipeline.input_texture_names().is_empty()
        {
            pipeline.encode_into_with_named_textures(
                device,
                encoder,
                &self.named_views,
                &self.scene_view,
                &self.output_view,
            )?;
        } else {
            pipeline.encode_into(device, encoder, input_view, output_view);
        }
        Ok(())
    }

    /// Execute multiple named pipelines in sequence
    pub fn execute_pipeline_sequence(
        &mut self,
        pipeline_names: &[&str],
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        for name in pipeline_names {
            self.execute_named_pipeline(name, device, encoder)?;
        }
        Ok(())
    }

    /// Execute all named pipelines in the registry
    /// Pipelines are executed in insertion order (HashMap preserves insertion order in Rust)
    pub fn execute_all_pipelines(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        let pipeline_names: Vec<String> = self.multi_pipeline.keys().cloned().collect();

        for name in &pipeline_names {
            self.execute_named_pipeline(name, device, encoder)?;
        }
        Ok(())
    }

    /// Encode all multi-pipelines using the client-owned encoder pattern
    /// This is the main entry point for multi-pipeline workflows
    pub fn encode_all_multi_pipelines(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        self.execute_all_pipelines(device, encoder)
    }

    /// Render the internal Draw object to internal scene texture and return a reference to the view.
    /// - The main entry point from the client app
    pub fn render_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> &wgpu::TextureView {
        let ce_desc = wgpu::CommandEncoderDescriptor {
            label: Some("Scene renderer"),
        };
        let mut encoder = device.create_command_encoder(&ce_desc);

        self.draw_renderer.encode_render_pass(
            device,
            &mut encoder,
            &self.draw,
            1.0,
            self.scene_texture.size(),
            &self.scene_view,
            None,
        );

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Poll);

        &self.scene_view
    }

    /// Take an already-rendered scene view and directly copy it to the output,
    /// skipping the effects pipeline
    pub fn direct_to_view(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.passthrough_to_view(device, queue, &self.output_view);
    }

    /// Post-process an already-rendered scene if a pipeline chain is enabled,
    /// or simply copy the scene texture directly to the output.
    /// - automatically uses the Nnpipe's output view
    pub fn post_process(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if let Some(effects_pipeline) = &mut self.effects_pipeline {
            effects_pipeline.process(device, queue, &self.scene_view, &self.output_view);
        } else {
            self.passthrough_to_view(device, queue, &self.output_view);
        }
    }

    /// Process with pipeline chain - runs all enabled effects in sequence
    /// - allows user to specify the output texture view
    pub fn process_effects(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
    ) {
        if let Some(effects_pipeline) = &mut self.effects_pipeline {
            effects_pipeline.process(device, queue, &self.scene_view, output_view);
        } else {
            self.passthrough_to_view(device, queue, &self.output_view);
        }
    }

    /// Get access to the raw scene texture view (if already rendered)
    pub fn get_scene_view(&self) -> &wgpu::TextureView {
        &self.scene_view
    }

    /// Get access the post-processed view
    pub fn get_post_processed_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// Add an effect to the pipeline chain
    pub fn add_effect(&mut self, effect: Pipeline) {
        self.effects_pipeline = Some(effect);
    }

    /// Clear the scene texture
    pub fn clear_scene(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Clear Scene Encoder"),
        });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Scene Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,
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
        device.poll(wgpu::Maintain::Poll);
    }

    /// Encode passthrough rendering into an existing command encoder
    pub fn encode_passthrough_to_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Passthrough pass"),
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

        pass.set_pipeline(&self.passthrough_pipeline);
        pass.set_bind_group(0, &self.passthrough_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Simple passthrough - copy scene texture directly to output
    pub fn passthrough_to_view(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Passthrough Encoder"),
        });

        self.encode_passthrough_to_view(&mut encoder, output_view);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Poll);
    }

    /// Original process method for backward compatibility
    pub fn process(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
    ) {
        self.render_scene(device, queue);
        self.process_effects(device, queue, output_view);
    }

    /// Helper to create a TextureReshaper for displaying a texture in a window
    pub fn create_reshaper_for_window(
        &self,
        device: &wgpu::Device,
        window: &nannou::window::Window,
        view: &wgpu::TextureView,
    ) -> wgpu::TextureReshaper {
        let sample_count = window.msaa_samples();
        let dst_format = Frame::TEXTURE_FORMAT;

        wgpu::TextureReshaper::new(
            device,
            view,
            1, // Assuming non-multisampled texture
            wgpu::TextureSampleType::Float { filterable: true },
            sample_count,
            dst_format,
        )
    }

    /// Helper to create a reshaper for the post-processed view
    pub fn create_reshaper_for_post_processed(
        &self,
        device: &wgpu::Device,
        window: &nannou::window::Window,
    ) -> wgpu::TextureReshaper {
        self.create_reshaper_for_window(device, window, &self.output_view)
    }

    /// Helper to create a reshaper for the raw scene
    pub fn create_reshaper_for_raw_scene(
        &self,
        device: &wgpu::Device,
        window: &nannou::window::Window,
    ) -> wgpu::TextureReshaper {
        self.create_reshaper_for_window(device, window, &self.scene_view)
    }

    /// Helper to draw a texture to a frame
    pub fn draw_to_frame(&self, reshaper: &wgpu::TextureReshaper, frame: &Frame) {
        let mut encoder = frame.command_encoder();
        reshaper.encode_render_pass(frame.texture_view(), &mut encoder);
    }

    /// Access the raw scene texture
    pub fn scene_view(&self) -> &wgpu::TextureView {
        &self.scene_view
    }

    /// Access the post-processed texture
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }
}

/// Helper function to create render texture
fn create_render_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    samples: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

    // Only add STORAGE_BINDING for formats that support it
    if supports_storage_binding(format) {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }

    wgpu::TextureBuilder::new()
        .size([width, height])
        .usage(usage)
        .sample_count(samples)
        .format(format)
        .build(device)
}

/// Helper function to create render pipeline
fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
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
    })
}

/// Check if a texture format supports storage binding
fn supports_storage_binding(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rgba16Float
            | wgpu::TextureFormat::Rgba32Float
            | wgpu::TextureFormat::R32Float
            | wgpu::TextureFormat::Rg32Float
            | wgpu::TextureFormat::R16Float
            | wgpu::TextureFormat::Rg16Float
            | wgpu::TextureFormat::R32Uint
            | wgpu::TextureFormat::Rg32Uint
            | wgpu::TextureFormat::Rgba32Uint
            | wgpu::TextureFormat::R32Sint
            | wgpu::TextureFormat::Rg32Sint
            | wgpu::TextureFormat::Rgba32Sint
    )
}
