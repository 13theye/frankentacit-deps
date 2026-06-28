//! # Particle Renderer
//!
//! Renders point particles as instanced quads with per-particle color.
//!
//! ## Alpha Handling
//!
//! The particle renderer outputs **straight alpha** (RGB and A are independent).
//! This allows particles and other straight-alpha content (e.g., Nannou Draw)
//! to render to the same texture before a single premultiply step.
//!
//! **Pipeline flow:**
//! 1. Particles render with straight alpha (this renderer)
//! 2. Other straight-alpha content renders to the same texture
//! 3. A `PremultiplyComponent` converts the texture to premultiplied alpha
//! 4. Compositing operations use the premultiplied texture
//!
//! ## Blend State
//!
//! Uses `ALPHA_BLENDING` (standard straight alpha blending):
//!
//! ```text
//! result.rgb = src.rgb × src.a + dst.rgb × (1 - src.a)
//! result.a   = src.a   + dst.a   × (1 - src.a)
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! let renderer = ParticleRenderer::new(device, texture_config, max_particles);
//!
//! // Each frame:
//! let particles: Vec<ParticleGpu> = assemble_particles();
//! let count = renderer.update_buffer(queue, &particles);
//! renderer.encode_only(&mut encoder, count, target_texture);
//!
//! // Then premultiply before compositing
//! premultiply_pipeline.encode(&mut encoder);
//! ```

use bytemuck::{Pod, Zeroable};
use nannou::wgpu;
use wgpu::util::DeviceExt;

use crate::TextureConfig;

// Maximum buffer size in bytes (256 MB - WebGPU max buffer size limit)
const MAX_BUFFER_SIZE: u64 = 268_435_456;

/// Base quad vertex positions (will be instanced)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2], // Local quad coordinates: [-1,-1] to [1,1]
}

/// Uniform data for particles
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ParticleRendererUniforms {
    texture_width: f32,
    texture_height: f32,
    _padding: [u32; 2], // Maintain 16-byte alignment for GPU uniforms
}

/// A simple struct containing only what is needed to draw a particle to texture
///
/// Memory Layout (32 bytes total):
/// - position: [f32; 2] = 8 bytes
/// - color_packed: u32 = 4 bytes (RGBA8 format)
/// - _padding: [u32; 5] = 20 bytes (for 16-byte alignment)
///
/// Previous layout: 48 bytes (position=8, rgb=12, alpha=4, padding=12)
/// New layout: 32 bytes (position=8, color=4, padding=20)
/// Savings: 16 bytes per particle (-33%)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleGpu {
    pub position: [f32; 2],
    pub color_packed: u32, // RGBA8 packed color (0xAABBGGRR)
    _padding: [u32; 5],    // Maintain 16-byte alignment for GPU
}

impl ParticleGpu {
    /// Create a new particle with separate RGB and alpha values
    ///
    /// # Arguments
    /// * `position` - Particle position in world coordinates
    /// * `rgb` - RGB color values (0.0-1.0), will be clamped
    /// * `alpha` - Alpha value (0.0-1.0), will be clamped
    pub fn new(position: [f32; 2], rgb: [f32; 3], alpha: f32) -> Self {
        use crate::renderers::color_utils::pack_color_rgba;

        Self {
            position,
            color_packed: pack_color_rgba(rgb[0], rgb[1], rgb[2], alpha),
            _padding: [0; 5],
        }
    }

    /// Create a new particle with a pre-packed color
    ///
    /// # Arguments
    /// * `position` - Particle position in world coordinates
    /// * `color_packed` - Pre-packed RGBA8 color (0xAABBGGRR)
    #[inline]
    pub fn new_packed(position: [f32; 2], color_packed: u32) -> Self {
        Self {
            position,
            color_packed,
            _padding: [0; 5],
        }
    }
}

/// Multi-buffer configuration for handling large particle counts
struct BufferGroup {
    buffer: wgpu::Buffer,
    capacity: usize, // Number of particles this buffer can hold
}

pub struct ParticleRenderer {
    render_pipeline: wgpu::RenderPipeline,
    quad_vertex_buffer: wgpu::Buffer,   // Static quad vertices
    index_buffer: wgpu::Buffer,         // Quad indices
    instance_buffers: Vec<BufferGroup>, // Multiple instance buffers to support large particle counts
    #[allow(dead_code)]
    uniforms_buffer: wgpu::Buffer, // Texture size uniforms
    bind_group: wgpu::BindGroup,        // Bind group for uniforms
    max_particles: usize,
    #[allow(dead_code)]
    particles_per_buffer: usize, // Max particles that fit in one buffer
    engine_debug: bool, // Enable performance timing output
}

impl ParticleRenderer {
    pub fn new(device: &wgpu::Device, config: TextureConfig, max_particles: usize) -> Self {
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fragment/particles.wgsl").into(),
            ),
        });

        // Create bind group layout for uniforms
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create uniforms buffer
        let uniforms = ParticleRendererUniforms {
            texture_width: config.width as f32,
            texture_height: config.height as f32,
            _padding: [0; 2],
        };

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Uniforms Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
        });

        // Create render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    // Quad vertex buffer (static)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    // Instance buffer (particle data)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleGpu>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // position
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // color_packed (RGBA8 u32)
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Uint32,
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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

        // Create static quad vertex buffer (single quad that will be instanced)
        let quad_vertices = [
            QuadVertex {
                position: [-1.0, -1.0],
            }, // Bottom-left
            QuadVertex {
                position: [1.0, -1.0],
            }, // Bottom-right
            QuadVertex {
                position: [1.0, 1.0],
            }, // Top-right
            QuadVertex {
                position: [-1.0, 1.0],
            }, // Top-left
        ];

        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create index buffer for quad (2 triangles)
        let quad_indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Quad Index Buffer"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Calculate how many particles fit in one buffer (respecting MAX_BUFFER_SIZE)
        let particle_size = std::mem::size_of::<ParticleGpu>() as u64;
        let particles_per_buffer = (MAX_BUFFER_SIZE / particle_size) as usize;

        // Calculate how many buffers we need
        let num_buffers = max_particles.div_ceil(particles_per_buffer);

        // Create multiple instance buffers to support large particle counts
        let mut instance_buffers = Vec::with_capacity(num_buffers);

        for i in 0..num_buffers {
            // Calculate capacity for this buffer (last buffer may be smaller)
            let remaining_particles = max_particles - (i * particles_per_buffer);
            let buffer_capacity = remaining_particles.min(particles_per_buffer);
            let buffer_size = (buffer_capacity * std::mem::size_of::<ParticleGpu>()) as u64;

            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Particle Instance Buffer {}", i)),
                size: buffer_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            instance_buffers.push(BufferGroup {
                buffer,
                capacity: buffer_capacity,
            });
        }

        Self {
            render_pipeline,
            quad_vertex_buffer,
            index_buffer,
            instance_buffers,
            uniforms_buffer,
            bind_group,
            max_particles,
            particles_per_buffer,
            engine_debug: false,
        }
    }

    /// Update the instance buffers without creating a new render pass
    /// Returns the number of particles
    /// Splits data across multiple buffers to respect WebGPU buffer size limits
    pub fn update_buffer(&self, queue: &wgpu::Queue, particles: &[ParticleGpu]) -> usize {
        let particle_count = particles.len().min(self.max_particles);
        if particle_count == 0 {
            return 0;
        }

        // Write particles to multiple buffers
        let mut offset = 0;
        for buffer_group in &self.instance_buffers {
            if offset >= particle_count {
                break;
            }

            let count_for_this_buffer = (particle_count - offset).min(buffer_group.capacity);
            let slice = &particles[offset..offset + count_for_this_buffer];

            queue.write_buffer(&buffer_group.buffer, 0, bytemuck::cast_slice(slice));

            offset += count_for_this_buffer;
        }

        particle_count
    }

    /// Get direct access to the first instance buffer for zero-copy writes
    /// DEPRECATED: Use write_particles_direct instead for multi-buffer support
    /// OPTIMIZED: For Apple Silicon unified memory - write directly to GPU-visible memory
    #[deprecated(note = "Use write_particles_direct for proper multi-buffer support")]
    pub fn get_instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffers[0].buffer
    }

    /// Get the maximum number of particles this renderer can handle
    pub fn max_particles(&self) -> usize {
        self.max_particles
    }

    /// Write particles directly to GPU memory using mapped buffer (zero-copy on unified memory)
    /// Returns the number of particles written
    /// Handles multi-buffer architecture transparently to the caller
    pub fn write_particles_direct<F>(&self, queue: &wgpu::Queue, count: usize, writer: F) -> usize
    where
        F: FnOnce(&mut [ParticleGpu]),
    {
        use std::time::Instant;

        let particle_count = count.min(self.max_particles);
        if particle_count == 0 {
            return 0;
        }

        let start_alloc = Instant::now();
        // Create a staging vector for the writer to populate
        let mut staging_buffer =
            vec![ParticleGpu::new([0.0, 0.0], [0.0, 0.0, 0.0], 0.0); particle_count];
        let alloc_time = start_alloc.elapsed();

        // Let the caller write to the staging buffer (caller will time this)
        writer(&mut staging_buffer[..particle_count]);

        // Now distribute the staging buffer across multiple GPU buffers
        let start_write = Instant::now();
        let mut offset = 0;
        for buffer_group in &self.instance_buffers {
            if offset >= particle_count {
                break;
            }

            let count_for_this_buffer = (particle_count - offset).min(buffer_group.capacity);
            let buffer_size =
                (count_for_this_buffer * std::mem::size_of::<ParticleGpu>()) as wgpu::BufferAddress;

            // Write directly to GPU-mapped memory (zero-copy on Apple Silicon)
            if let Ok(non_zero_size) = std::num::NonZeroU64::try_from(buffer_size) {
                if let Some(mut view) =
                    queue.write_buffer_with(&buffer_group.buffer, 0, non_zero_size)
                {
                    let particle_slice = bytemuck::cast_slice_mut::<u8, ParticleGpu>(&mut view);
                    let source_slice = &staging_buffer[offset..offset + count_for_this_buffer];
                    particle_slice[..count_for_this_buffer].copy_from_slice(source_slice);
                    // View is dropped here, scheduling the write
                }
            }

            offset += count_for_this_buffer;
        }
        let write_time = start_write.elapsed();

        if self.engine_debug {
            let particle_bytes = particle_count * std::mem::size_of::<ParticleGpu>();
            println!(
                "    [Nnpipe] Particle buffer alloc: {:.3}ms\n    write: {:.3}ms ({:.2} MB, {:.2} GB/s)",
                alloc_time.as_secs_f64() * 1000.0,
                write_time.as_secs_f64() * 1000.0,
                particle_bytes as f64 / 1_048_576.0,
                (particle_bytes as f64 / 1_073_741_824.0) / write_time.as_secs_f64()
            );
        }

        particle_count
    }

    /// Encode particle rendering without updating buffer (for zero-copy path)
    /// Call this when buffer was already updated via write_particles_direct
    /// Handles multi-buffer rendering with multiple draw calls
    pub fn encode_only(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        particle_count: usize,
        target_view: &wgpu::TextureView,
    ) {
        if particle_count == 0 {
            return;
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Particle Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Render from multiple buffers with separate draw calls
            let mut offset = 0;
            for buffer_group in &self.instance_buffers {
                if offset >= particle_count {
                    break;
                }

                let count_for_this_buffer = (particle_count - offset).min(buffer_group.capacity);

                render_pass.set_vertex_buffer(1, buffer_group.buffer.slice(..));
                render_pass.draw_indexed(0..6, 0, 0..count_for_this_buffer as u32);

                offset += count_for_this_buffer;
            }
        }
    }

    /// Encode particle rendering into an existing command encoder
    /// Handles multi-buffer architecture with multiple draw calls
    pub fn encode_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        particles: &[ParticleGpu],
        target_view: &wgpu::TextureView,
    ) {
        let particle_count = self.update_buffer(queue, particles);

        if particle_count == 0 {
            return;
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Particle Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't clear, add to existing content
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Render from multiple buffers with separate draw calls
            let mut offset = 0;
            for buffer_group in &self.instance_buffers {
                if offset >= particle_count {
                    break;
                }

                let count_for_this_buffer = (particle_count - offset).min(buffer_group.capacity);

                render_pass.set_vertex_buffer(1, buffer_group.buffer.slice(..));
                render_pass.draw_indexed(0..6, 0, 0..count_for_this_buffer as u32); // 6 indices per quad

                offset += count_for_this_buffer;
            }
        }
    }

    /// Set engine debug logging messages
    pub fn set_engine_debug(&mut self, engine_debug: bool) {
        self.engine_debug = engine_debug;
    }
}
