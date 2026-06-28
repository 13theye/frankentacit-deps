// src/nnpipeline.rs
//
// Texture rendering and post-processing

use nannou::prelude::*;
use nannou::wgpu;

#[allow(dead_code)]
pub struct Nnpipe {
    // Draw renderer
    pub draw_renderer: nannou::draw::Renderer,

    // Textures for the pipeline
    pub scene_texture: wgpu::Texture,
    pub brightness_texture: wgpu::Texture,
    pub blur_h_texture: wgpu::Texture,
    pub blur_v_texture: wgpu::Texture,
    pub composite_texture: wgpu::Texture,

    // Texture views
    pub scene_view: wgpu::TextureView,
    pub brightness_view: wgpu::TextureView,
    pub blur_h_view: wgpu::TextureView,
    pub blur_v_view: wgpu::TextureView,
    pub composite_view: wgpu::TextureView,
    pub downsample_view: wgpu::TextureView,

    // Render pipelines for each pass
    brightness_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    downsample_pipeline: wgpu::RenderPipeline,
    passthrough_pipeline: wgpu::RenderPipeline,

    // Adaptive bloom
    pub adaptive_blur_scaling: f32,
    pub max_blur_radius: f32,
    pub intensity_curve: f32,

    // Pipeline parameters
    pub brightness_threshold: f32,
    pub bloom_intensity: f32,

    // Shader bind groups
    pub brightness_bind_group: wgpu::BindGroup,
    pub blur_h_bind_group: wgpu::BindGroup,
    pub blur_v_bind_group: wgpu::BindGroup,
    pub composite_bind_group: wgpu::BindGroup,
    pub downsample_bind_group: wgpu::BindGroup,
    passthrough_bind_group: wgpu::BindGroup,

    // Sampler for texture sampling
    sampler: wgpu::Sampler,

    // Uniform buffers for parameters
    threshold_buffer: wgpu::Buffer,
    blur_h_buffer: wgpu::Buffer,
    blur_v_buffer: wgpu::Buffer,
    intensity_buffer: wgpu::Buffer,

    adaptive_scaling_buffer: wgpu::Buffer,
    max_radius_buffer: wgpu::Buffer,
    intensity_curve_buffer: wgpu::Buffer,
}

impl Nnpipe {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, samples: u32) -> Self {
        // Define formats
        let hi_format = wgpu::TextureFormat::Rgba16Float;
        let lo_format = wgpu::TextureFormat::Rgba8UnormSrgb;

        // Define half-widths for blur textures
        let half_width = width / 2;
        let half_height = height / 2;

        // Create textures
        let scene_texture = create_render_texture(device, width, height, samples, hi_format);
        let composite_texture = create_render_texture(device, width, height, 1, hi_format);
        let brightness_texture = create_render_texture(device, width, height, 1, lo_format);
        let downsample_texture =
            create_render_texture(device, half_width, half_height, samples, lo_format);
        let blur_h_texture = create_render_texture(device, half_width, half_height, 1, lo_format);
        let blur_v_texture = create_render_texture(device, half_width, half_height, 1, lo_format);

        // Create draw renderer
        let draw_renderer = nannou::draw::RendererBuilder::new()
            .build_from_texture_descriptor(device, scene_texture.descriptor());

        // Create texture views
        let scene_view = scene_texture.view().build();
        let brightness_view = brightness_texture.view().build();
        let downsample_view = downsample_texture.view().build();
        let blur_h_view = blur_h_texture.view().build();
        let blur_v_view = blur_v_texture.view().build();
        let composite_view = composite_texture.view().build();

        // Create a sampler for texture sampling
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bloom sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create uniform buffers
        let brightness_threshold = 0.55f32;
        let threshold_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Threshold Buffer"),
            contents: bytemuck::cast_slice(&[brightness_threshold]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Horizontal blur direction (1.0, 0.0)
        let blur_h_direction = [1.0f32, 0.0f32];
        let blur_h_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Horizontal Blur Buffer"),
            contents: bytemuck::cast_slice(&blur_h_direction),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Vertical blur direction (0.0, 1.0)
        let blur_v_direction = [0.0f32, 0.7f32];
        let blur_v_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertical Blur Buffer"),
            contents: bytemuck::cast_slice(&blur_v_direction),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bloom intensity
        let bloom_intensity = 3.0f32;
        let intensity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Intensity Buffer"),
            contents: bytemuck::cast_slice(&[bloom_intensity]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Additional buffers for adaptive bloom
        let adaptive_blur_scaling = 2.0f32; // working at half resolution so equivalent to 2x in final
        let adaptive_scaling_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Adaptive Scaling Buffer"),
                contents: bytemuck::cast_slice(&[adaptive_blur_scaling]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let max_blur_radius = 10.0f32; // working at half resolution so equivalent to 2x in final
        let max_radius_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Max Radius Buffer"),
            contents: bytemuck::cast_slice(&[max_blur_radius]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let intensity_curve = 3.0f32;
        let intensity_curve_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Intensity Curve Buffer"),
            contents: bytemuck::cast_slice(&[intensity_curve]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create shader modules
        let brightness_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Brightness Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brightness.wgsl").into()),
        });

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
        });

        let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Downsample Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/downsample.wgsl").into()),
        });

        let passthrough_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Passthrough Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/passthrough.wgsl").into()),
        });

        // Create bind group layouts
        let brightness_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Brightness Bind Group Layout"),
                entries: &[
                    // Texture binding
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
                    // Sampler binding
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu_types::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Threshold uniform binding
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

        // Similar bind group layouts for blur and composite passes...
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blur Bind Group Layout"),
                entries: &[
                    // Texture binding
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
                    // Sampler binding
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu_types::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Direction uniform binding
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
                        binding: 3, // This would be the next available binding
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

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    // Scene texture binding
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
                    // Bloom texture binding
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
                    // Sampler binding
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu_types::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Intensity uniform binding
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
                        binding: 4, // This would be the next available binding
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

        let downsample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Downsample Bind Group Layout"),
                entries: &[
                    // Texture binding
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
                    // Sampler binding
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu_types::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let passthrough_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Passthrough Bind Group Layout"),
                entries: &[
                    // Texture binding
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
                    // Sampler binding
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu_types::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create bind groups
        let brightness_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Brightness Bind Group"),
            layout: &brightness_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
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

        let blur_h_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Horizontal Blur Bind Group"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&downsample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        blur_h_buffer.as_entire_buffer_binding(),
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

        let blur_v_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vertical Blur Bind Group"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blur_h_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(
                        blur_v_buffer.as_entire_buffer_binding(),
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

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Bind Group"),
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&blur_v_view),
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

        let downsample_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Downsample Bind Group"),
            layout: &downsample_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&brightness_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

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

        // Create render pipeline layouts
        let brightness_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Brightness Pipeline Layout"),
                bind_group_layouts: &[&brightness_bind_group_layout],
                push_constant_ranges: &[],
            });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Pipeline Layout"),
            bind_group_layouts: &[&blur_bind_group_layout],
            push_constant_ranges: &[],
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        let downsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Downsample Pipeline Layout"),
                bind_group_layouts: &[&downsample_bind_group_layout],
                push_constant_ranges: &[],
            });

        let passthrough_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Passthrough Pipeline Layout"),
                bind_group_layouts: &[&passthrough_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create render pipelines
        let brightness_pipeline = create_render_pipeline(
            device,
            &brightness_pipeline_layout,
            &brightness_shader,
            "Brightness Pipeline",
            lo_format,
        );

        let blur_pipeline = create_render_pipeline(
            device,
            &blur_pipeline_layout,
            &blur_shader,
            "Blur Pipeline",
            lo_format,
        );

        let composite_pipeline = create_render_pipeline(
            device,
            &composite_pipeline_layout,
            &composite_shader,
            "Composite Pipeline",
            hi_format,
        );

        let downsample_pipeline = create_render_pipeline(
            device,
            &downsample_pipeline_layout,
            &downsample_shader,
            "Downsample Pipeline",
            lo_format,
        );

        let passthrough_pipeline = create_render_pipeline(
            device,
            &passthrough_pipeline_layout,
            &passthrough_shader,
            "Passthrough Pipeline",
            hi_format, // Output high-quality format like the scene
        );

        // Return the fully initialized PostProcessing struct
        Self {
            draw_renderer,
            scene_texture,
            brightness_texture,
            blur_h_texture,
            blur_v_texture,
            composite_texture,
            scene_view,
            brightness_view,
            blur_h_view,
            blur_v_view,
            composite_view,
            downsample_view,
            sampler,
            brightness_pipeline,
            blur_pipeline,
            composite_pipeline,
            downsample_pipeline,
            passthrough_pipeline,
            threshold_buffer,
            blur_h_buffer,
            blur_v_buffer,
            intensity_buffer,

            adaptive_scaling_buffer,
            max_radius_buffer,
            intensity_curve_buffer,

            brightness_threshold,
            bloom_intensity,
            adaptive_blur_scaling,
            max_blur_radius,
            intensity_curve,

            brightness_bind_group,
            blur_h_bind_group,
            blur_v_bind_group,
            composite_bind_group,
            downsample_bind_group,
            passthrough_bind_group,
        }
    }

    // original process method for backward compatibility
    pub fn process(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        draw: &nannou::Draw,
    ) {
        self.render_scene(device, queue, draw);
        self.post_process_to_view(device, queue, output_view);
    }

    // Render a Draw object to internal scene texture and return a reference to the view
    pub fn render_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        draw: &nannou::Draw,
    ) -> &wgpu::TextureView {
        // Render the scene to the scene texture
        let ce_desc = wgpu::CommandEncoderDescriptor {
            label: Some("Scene renderer"),
        };
        let mut encoder = device.create_command_encoder(&ce_desc);

        self.draw_renderer.encode_render_pass(
            device,
            &mut encoder,
            draw,
            1.0,
            self.scene_texture.size(),
            &self.scene_view,
            None,
        );

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Return a reference to the scene texture view
        &self.scene_view
    }

    // Get access to the raw scene texture view (if already rendered)
    pub fn get_scene_view(&self) -> &wgpu::TextureView {
        &self.scene_view
    }

    // Get access the post-processed view
    pub fn get_post_processed_view(&self) -> &wgpu::TextureView {
        // Return a reference to the composite texture view
        &self.composite_view
    }

    // Apply post-processing and return the post-processed view
    pub fn post_process(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Apply post-processing to internal scene texture
        // Use the composite_texture as the destination

        // [Implementation of all post-processing passes]
        // 1. Brightness extraction pass to brightness_texture
        // 2. Horizontal blur pass to blur_h_texture
        // 3. Vertical blur pass to blur_v_texture
        // 4. Final composite pass to composite_texture

        self.post_process_to_view(device, queue, &self.composite_view);
    }

    /// Call this to take the rendered scene texture and draw it to the screen,
    /// skipping all post-processing
    pub fn direct_to_view(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.passthrough_to_view(device, queue, &self.composite_view);
    }

    // applies post-processing to a scene texture
    pub fn post_process_to_view(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
    ) {
        // Now execute the post-processing passes

        let ce_desc = wgpu::CommandEncoderDescriptor {
            label: Some("Post-processing Encoder"),
        };

        let mut encoder = device.create_command_encoder(&ce_desc);

        // 1. Brightness extraction pass
        {
            /*
            let ce_desc = wgpu::CommandEncoderDescriptor {
                label: Some("Brightness extraction"),
            };
            let mut encoder = device.create_command_encoder(&ce_desc);
            */

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Brightness pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.brightness_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&self.brightness_pipeline);
            pass.set_bind_group(0, &self.brightness_bind_group, &[]);
            pass.draw(0..3, 0..1); // Draw a fullscreen triangle

            drop(pass);
            //queue.submit(Some(encoder.finish()));
        }

        // 2. Downsample pass
        {
            /*
            let ce_desc = wgpu::CommandEncoderDescriptor {
                label: Some("Horizontal blur"),
            };
            let mut encoder = device.create_command_encoder(&ce_desc);
            */

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Horizontal blur pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.downsample_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            //pass.set_pipeline(&self.blur_pipeline);
            //pass.set_bind_group(0, &self.blur_h_bind_group, &[]);
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &self.downsample_bind_group, &[]);

            pass.draw(0..3, 0..1); // Draw a fullscreen triangle

            drop(pass);
            //queue.submit(Some(encoder.finish()));
        }

        // 3. Horizontal blur pass (at half resolution)
        // Create a temporary texture view for the horizontal pass output
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Horizontal blur pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.blur_h_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_h_bind_group, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
        }

        // 4. Vertical blur pass
        {
            /*
            let ce_desc = wgpu::CommandEncoderDescriptor {
                label: Some("Vertical blur"),
            };
            let mut encoder = device.create_command_encoder(&ce_desc);
             */

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Vertical blur pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.blur_v_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_v_bind_group, &[]);
            pass.draw(0..3, 0..1); // Draw a fullscreen triangle

            drop(pass);
            //queue.submit(Some(encoder.finish()));
        }

        // 5. Final composite pass to the output texture
        {
            /*            let ce_desc = wgpu::CommandEncoderDescriptor {
                label: Some("Final composite"),
            };
            let mut encoder = device.create_command_encoder(&ce_desc);
             */

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view, // Render directly to the output
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bind_group, &[]);
            pass.draw(0..3, 0..1); // Draw a fullscreen triangle

            drop(pass);
            //queue.submit(Some(encoder.finish()));
        }

        // Submit the encoder
        queue.submit(Some(encoder.finish()));

        // Make sure all commands are completed
        device.poll(wgpu::Maintain::Wait);
    }

    pub fn passthrough_to_view(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
    ) {
        // Simple passthrough - copy scene texture directly to output
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Passthrough Encoder"),
        });

        {
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

            // Use simple passthrough pipeline to copy scene -> output
            pass.set_pipeline(&self.passthrough_pipeline);
            pass.set_bind_group(0, &self.passthrough_bind_group, &[]);
            pass.draw(0..3, 0..1); // Draw a fullscreen triangle
        }

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    /******************* Helper methods for drawing to windows ****************** */

    // Helper to create a TextureReshaper for displaying a texture in a window
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

    // Helper to create a reshaper for the post-processed view
    pub fn create_reshaper_for_post_processed(
        &self,
        device: &wgpu::Device,
        window: &nannou::window::Window,
    ) -> wgpu::TextureReshaper {
        self.create_reshaper_for_window(device, window, &self.composite_view)
    }

    // Helper to create a reshaper for the raw scene
    pub fn create_reshaper_for_raw_scene(
        &self,
        device: &wgpu::Device,
        window: &nannou::window::Window,
    ) -> wgpu::TextureReshaper {
        self.create_reshaper_for_window(device, window, &self.scene_view)
    }

    // Helper to draw a texture to a frame
    pub fn draw_to_frame(&self, reshaper: &wgpu::TextureReshaper, frame: &Frame) {
        let mut encoder = frame.command_encoder();
        reshaper.encode_render_pass(frame.texture_view(), &mut encoder);
    }

    /******************* Helper methods for updating parameters ****************** */

    pub fn set_brightness_threshold(&mut self, queue: &wgpu::Queue, threshold: f32) {
        self.brightness_threshold = threshold;
        queue.write_buffer(
            &self.threshold_buffer,
            0,
            bytemuck::cast_slice(&[threshold]),
        );
    }

    pub fn set_bloom_intensity(&mut self, queue: &wgpu::Queue, intensity: f32) {
        self.bloom_intensity = intensity;
        queue.write_buffer(
            &self.intensity_buffer,
            0,
            bytemuck::cast_slice(&[intensity]),
        );
    }

    pub fn set_adaptive_blur_scaling(&mut self, queue: &wgpu::Queue, scaling: f32) {
        self.adaptive_blur_scaling = scaling;
        queue.write_buffer(
            &self.adaptive_scaling_buffer,
            0,
            bytemuck::cast_slice(&[scaling]),
        );
    }

    pub fn set_max_blur_radius(&mut self, queue: &wgpu::Queue, radius: f32) {
        self.max_blur_radius = radius;
        queue.write_buffer(&self.max_radius_buffer, 0, bytemuck::cast_slice(&[radius]));
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

// Helper function to create render texture
fn create_render_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    samples: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    wgpu::TextureBuilder::new()
        .size([width, height])
        .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
        .sample_count(samples)
        .format(format)
        .build(device)
}

// Helper function to create render pipeline
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
