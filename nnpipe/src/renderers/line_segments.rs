//! # Line Segment Renderer
//!
//! Renders line segments as instanced oriented quads connecting point pairs.
//!
//! ## Alpha Handling
//!
//! The segment renderer outputs **straight alpha** (RGB and A are independent).
//! This allows segments and other straight-alpha content (e.g., Nannou Draw)
//! to render to the same texture before a single premultiply step.
//!
//! **Pipeline flow:**
//! 1. Segments render with straight alpha (this renderer)
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
//! let renderer = SegmentRenderer::new(device, texture_config, max_segments);
//!
//! // Each frame:
//! let segments: Vec<SegmentInstance> = assemble_segments();
//! let (_, count) = renderer.update_buffer(queue, &segments);
//! renderer.encode_only(&mut encoder, count, target_texture);
//!
//! // Then premultiply before compositing
//! premultiply_pipeline.encode(&mut encoder);
//! ```

use bytemuck::{Pod, Zeroable};
use nannou::wgpu;
use wgpu::util::DeviceExt;

use crate::TextureConfig;

const FEEDBACK_POSITIONS: usize = 64;

// Maximum buffer size in bytes (256 MB - WebGPU max buffer size limit)
const MAX_BUFFER_SIZE: u64 = 268_435_456;

/// Base quad vertex positions (will be instanced)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2], // Local quad coordinates: [-1,-1] to [1,1]
}

/// Instance data containing the actual line segment points and properties
///
/// Memory Layout (32 bytes total):
/// - start_pos: [f32; 2] = 8 bytes
/// - end_pos: [f32; 2] = 8 bytes
/// - color_packed: u32 = 4 bytes (RGBA8 format, alpha included)
/// - line_width: f32 = 4 bytes
/// - _padding: [u32; 2] = 8 bytes (for 16-byte alignment)
///
/// Previous layout: 48 bytes (start=8, end=8, color=12, alpha=4, width=4, padding=12)
/// New layout: 32 bytes (start=8, end=8, color=4, width=4, padding=8)
/// Savings: 16 bytes per instance (-33%)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SegmentInstance {
    start_pos: [f32; 2], // Start position of line segment
    end_pos: [f32; 2],   // End position of line segment
    color_packed: u32,   // RGBA8 packed color with alpha (0xAABBGGRR)
    line_width: f32,     // Line thickness in pixels for this segment
    _padding: [u32; 2],  // Maintain 16-byte alignment for GPU
}

/// A segment defined by a sequence of points
///
/// Memory Layout (1,552 bytes total):
/// - points: [[f32; 2]; 128] = 1,024 bytes
/// - colors: [u32; 128] = 512 bytes (RGBA8 format)
/// - alpha: f32 = 4 bytes
/// - segment_length: f32 = 4 bytes
/// - line_width: f32 = 4 bytes
/// - actual_history_length: u32 = 4 bytes
///
/// Previous layout: 2,576 bytes (points=1024, colors=1536, alpha=4, length=4, width=4, history=4)
/// New layout: 1,552 bytes (points=1024, colors=512, alpha=4, length=4, width=4, history=4)
/// Savings: 1,024 bytes per segment (-40%)
/// Buffer capacity increase: ~104k → ~172k segments (+66%)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SegmentGpu {
    pub points: [[f32; 2]; FEEDBACK_POSITIONS], // All segment points
    pub colors: [u32; FEEDBACK_POSITIONS],      // RGBA8 packed colors for each point (0xAABBGGRR)
    pub alpha: f32,                             // Alpha multiplier for the first segment
    pub segment_length: f32,                    // 0.0-1.0, controls how many line segments to draw
    pub line_width: f32,                        // Line thickness in pixels
    pub actual_history_length: u32,             // Number of valid history positions (1-128)
    pub _padding: [u32; 0],                     // Ensure optimal GPU performance alignment
}

impl SegmentGpu {
    /// Create a new segment with unpacked RGB colors (will be converted to packed format)
    ///
    /// # Arguments
    /// * `points` - Array of segment point positions
    /// * `colors` - Array of RGB colors (0.0-1.0), will be packed with full alpha
    /// * `alpha` - Alpha multiplier for the first segment (0.0-1.0)
    /// * `segment_length` - Controls how many segments to draw (0.0-1.0)
    /// * `line_width` - Line thickness in pixels
    /// * `actual_history_length` - Number of valid history positions (1-128)
    pub fn new(
        points: [[f32; 2]; FEEDBACK_POSITIONS],
        colors: [[f32; 3]; FEEDBACK_POSITIONS],
        alpha: f32,
        segment_length: f32,
        line_width: f32,
        actual_history_length: u32,
    ) -> Self {
        use crate::renderers::color_utils::pack_color_rgb;

        // Pack RGB colors to u32 format
        let mut colors_packed = [0u32; FEEDBACK_POSITIONS];
        for (i, color) in colors.iter().enumerate() {
            colors_packed[i] = pack_color_rgb(color[0], color[1], color[2]);
        }

        Self {
            points,
            colors: colors_packed,
            alpha: alpha.clamp(0.0, 1.0),
            segment_length: segment_length.clamp(0.0, 1.0),
            line_width: line_width.max(0.1),
            actual_history_length: actual_history_length.clamp(1, FEEDBACK_POSITIONS as u32),
            _padding: [],
        }
    }

    /// Create a new segment with pre-packed colors (more efficient)
    ///
    /// # Arguments
    /// * `points` - Array of segment point positions
    /// * `colors_packed` - Array of pre-packed RGBA8 colors (0xAABBGGRR)
    /// * `alpha` - Alpha multiplier for the first segment (0.0-1.0)
    /// * `segment_length` - Controls how many segments to draw (0.0-1.0)
    /// * `line_width` - Line thickness in pixels
    /// * `actual_history_length` - Number of valid history positions (1-128)
    #[inline]
    pub fn new_packed(
        points: [[f32; 2]; FEEDBACK_POSITIONS],
        colors_packed: [u32; FEEDBACK_POSITIONS],
        alpha: f32,
        segment_length: f32,
        line_width: f32,
        actual_history_length: u32,
    ) -> Self {
        Self {
            points,
            colors: colors_packed,
            alpha: alpha.clamp(0.0, 1.0),
            segment_length: segment_length.clamp(0.0, 1.0),
            line_width: line_width.max(0.1),
            actual_history_length: actual_history_length.clamp(1, FEEDBACK_POSITIONS as u32),
            _padding: [],
        }
    }
}

impl Default for SegmentGpu {
    fn default() -> Self {
        Self {
            points: [[0.0; 2]; FEEDBACK_POSITIONS],
            colors: [0u32; FEEDBACK_POSITIONS], // Black with full alpha
            alpha: 0.0,
            segment_length: 0.0,
            line_width: 1.0,
            actual_history_length: 1,
            _padding: [],
        }
    }
}

/// Multi-buffer configuration for handling large segment instance counts
struct SegmentBufferGroup {
    buffer: wgpu::Buffer,
    capacity: usize, // Number of segment instances this buffer can hold
}

pub struct SegmentRenderer {
    render_pipeline: wgpu::RenderPipeline,
    quad_vertex_buffer: wgpu::Buffer, // Static quad vertices
    instance_buffers: Vec<SegmentBufferGroup>, // Multiple instance buffers to support large segment counts
    index_buffer: wgpu::Buffer,                // Quad indices
    max_start_points: usize,
    max_instances_capacity: usize, // Maximum instances across all buffers
    #[allow(dead_code)]
    instances_per_buffer: usize, // Max instances that fit in one buffer
    engine_debug: bool,            // Enable performance timing output
}

impl SegmentRenderer {
    pub fn new(device: &wgpu::Device, config: TextureConfig, max_start_points: usize) -> Self {
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line Segment Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fragment/line_segments.wgsl").into(),
            ),
        });

        // Create render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Line Segment Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Segment Pipeline"),
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
                    // Instance buffer (dynamic)
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<SegmentInstance>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            // start_pos
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // end_pos
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            // color_packed (RGBA8 u32)
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Uint32,
                            },
                            // line_width
                            wgpu::VertexAttribute {
                                offset: 20,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32,
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
            label: Some("Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create index buffer for quad (2 triangles)
        let quad_indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Index Buffer"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Calculate maximum instances needed (max 127 segments per segment group)
        let max_instances_per_segment = FEEDBACK_POSITIONS - 1; // 127 max line segments per start point
        let max_instances_capacity = max_start_points * max_instances_per_segment;

        // Calculate how many instances fit in one buffer (respecting MAX_BUFFER_SIZE)
        let instance_size = std::mem::size_of::<SegmentInstance>() as u64;
        let instances_per_buffer = (MAX_BUFFER_SIZE / instance_size) as usize;

        // Calculate how many buffers we need
        let num_buffers = max_instances_capacity.div_ceil(instances_per_buffer);

        // Create multiple instance buffers to support large segment counts
        let mut instance_buffers = Vec::with_capacity(num_buffers);

        for i in 0..num_buffers {
            // Calculate capacity for this buffer (last buffer may be smaller)
            let remaining_instances = max_instances_capacity - (i * instances_per_buffer);
            let buffer_capacity = remaining_instances.min(instances_per_buffer);
            let buffer_size = (buffer_capacity * std::mem::size_of::<SegmentInstance>()) as u64;

            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Segment Instance Buffer {}", i)),
                size: buffer_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            instance_buffers.push(SegmentBufferGroup {
                buffer,
                capacity: buffer_capacity,
            });
        }

        Self {
            render_pipeline,
            quad_vertex_buffer,
            instance_buffers,
            index_buffer,
            max_start_points,
            max_instances_capacity,
            instances_per_buffer,
            engine_debug: false,
        }
    }

    /// Generate segment instances from segment data
    fn generate_instances(&self, segments: &[SegmentGpu]) -> Vec<SegmentInstance> {
        let mut instances = Vec::with_capacity(self.max_instances_capacity);

        for segment in segments.iter().take(self.max_start_points) {
            // Calculate how many line segments to draw based on actual history length and segment_length
            let max_possible_segments = (segment.actual_history_length - 1).max(1) as usize;
            let requested_segments =
                (segment.segment_length * (max_possible_segments as f32)).ceil() as usize;
            let active_segments = requested_segments.min(max_possible_segments);

            // Pre-calculate values used in the loop to avoid redundant calculations
            let active_segments_minus_one = (active_segments - 1).max(1) as f32;
            let alpha_step = segment.alpha / active_segments_minus_one;

            // Create line segments connecting consecutive points
            // Only create segments within the active range
            for i in 0..active_segments {
                let start_pos = segment.points[i];
                let end_pos = segment.points[i + 1];

                // Calculate line segment properties using optimized distance calculation
                let dx = end_pos[0] - start_pos[0];
                let dy = end_pos[1] - start_pos[1];
                let line_length_squared = dx * dx + dy * dy;

                // Skip zero-length segments using squared distance (avoid sqrt)
                if line_length_squared < 0.000001 {
                    continue;
                }

                // Alpha fading along the segment - optimized calculation
                let segment_alpha = segment.alpha - (alpha_step * i as f32);

                // Unpack the stored color and repack with faded alpha
                use crate::renderers::color_utils::{pack_color_rgba, unpack_color};
                let color_unpacked = unpack_color(segment.colors[i]);
                let color_with_alpha = pack_color_rgba(
                    color_unpacked[0],
                    color_unpacked[1],
                    color_unpacked[2],
                    segment_alpha,
                );

                instances.push(SegmentInstance {
                    start_pos,
                    end_pos,
                    color_packed: color_with_alpha,
                    line_width: segment.line_width,
                    _padding: [0; 2],
                });
            }
        }

        instances
    }

    /// Update buffers with new segment data
    /// Returns (number of segments, total number of line instances)
    /// Splits data across multiple buffers to respect WebGPU buffer size limits
    pub fn update_buffer(&self, queue: &wgpu::Queue, segments: &[SegmentGpu]) -> (usize, usize) {
        let segment_count = segments.len().min(self.max_start_points);

        if segment_count == 0 {
            return (0, 0);
        }

        let instances = self.generate_instances(&segments[..segment_count]);
        let instance_count = instances.len().min(self.max_instances_capacity);

        if instance_count > 0 {
            // Write instances to multiple buffers
            let mut offset = 0;
            for buffer_group in &self.instance_buffers {
                if offset >= instance_count {
                    break;
                }

                let count_for_this_buffer = (instance_count - offset).min(buffer_group.capacity);
                let slice = &instances[offset..offset + count_for_this_buffer];

                queue.write_buffer(&buffer_group.buffer, 0, bytemuck::cast_slice(slice));

                offset += count_for_this_buffer;
            }
        }

        (segment_count, instance_count)
    }

    /// Write segments directly to GPU memory (optimized for Apple Silicon unified memory)
    /// Returns (segment_count, instance_count)
    /// Handles multi-buffer architecture transparently to the caller
    pub fn write_segments_direct<F>(
        &self,
        queue: &wgpu::Queue,
        count: usize,
        writer: F,
    ) -> (usize, usize)
    where
        F: FnOnce(&mut [SegmentGpu]),
    {
        use std::time::Instant;

        let segment_count = count.min(self.max_start_points);
        if segment_count == 0 {
            return (0, 0);
        }

        let start_alloc = Instant::now();
        // Create temporary buffer for segments
        let mut segments_vec = vec![SegmentGpu::default(); segment_count];
        let alloc_time = start_alloc.elapsed();

        // Let caller write segments directly
        writer(&mut segments_vec[..segment_count]);

        let start_generate = Instant::now();
        // Generate instances from segments
        let instances = self.generate_instances(&segments_vec[..segment_count]);
        let instance_count = instances.len().min(self.max_instances_capacity);
        let generate_time = start_generate.elapsed();

        let segment_bytes = segment_count * std::mem::size_of::<SegmentGpu>();
        let instance_bytes = instance_count * std::mem::size_of::<SegmentInstance>();

        if self.engine_debug {
            println!(
                "    [Nnpipe] {} instances from {} segments",
                instance_count, segment_count
            );
            println!(
                "    [Nnpipe] Segment allocation: {:.3}ms",
                alloc_time.as_secs_f64() * 1000.0,
            );
            println!(
                "    [Nnpipe] Instance generation: {:.3}ms",
                generate_time.as_secs_f64() * 1000.0,
            );
            println!(
                "    [Nnpipe] Data sizes: SegmentGpu = {:.2}MB, Instances = {:.2}MB",
                segment_bytes as f64 / 1_048_576.0,
                instance_bytes as f64 / 1_048_576.0
            );
        }

        if instance_count > 0 {
            let start_write = Instant::now();
            // Write instances to multiple buffers
            let mut offset = 0;
            for buffer_group in &self.instance_buffers {
                if offset >= instance_count {
                    break;
                }

                let count_for_this_buffer = (instance_count - offset).min(buffer_group.capacity);
                let buffer_size = (count_for_this_buffer * std::mem::size_of::<SegmentInstance>())
                    as wgpu::BufferAddress;

                // Write directly to GPU-mapped memory (zero-copy on Apple Silicon)
                if let Ok(non_zero_size) = std::num::NonZeroU64::try_from(buffer_size) {
                    if let Some(mut view) =
                        queue.write_buffer_with(&buffer_group.buffer, 0, non_zero_size)
                    {
                        let instance_slice =
                            bytemuck::cast_slice_mut::<u8, SegmentInstance>(&mut view);
                        let source_slice = &instances[offset..offset + count_for_this_buffer];
                        instance_slice[..count_for_this_buffer].copy_from_slice(source_slice);
                        // View is dropped here, scheduling the write
                    }
                }

                offset += count_for_this_buffer;
            }
            let write_time = start_write.elapsed();
            if self.engine_debug {
                println!(
                    "    [GPU] Buffer write: {:.3}ms ({:.2} GB/s effective bandwidth)",
                    write_time.as_secs_f64() * 1000.0,
                    (instance_bytes as f64 / 1_073_741_824.0) / write_time.as_secs_f64()
                );
            }
        }

        (segment_count, instance_count)
    }

    /// Get the maximum number of segment start points this renderer can handle
    pub fn max_segments(&self) -> usize {
        self.max_start_points
    }

    /// Encode segment rendering without updating buffer (for zero-copy path)
    /// Call this when buffer was already updated via write_segments_direct
    /// Handles multi-buffer rendering with multiple draw calls
    pub fn encode_only(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        instance_count: usize,
        target_view: &wgpu::TextureView,
    ) {
        if instance_count == 0 {
            return;
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Line Segment Render Pass"),
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
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Render from multiple buffers with separate draw calls
            let mut offset = 0;
            for buffer_group in &self.instance_buffers {
                if offset >= instance_count {
                    break;
                }

                let count_for_this_buffer = (instance_count - offset).min(buffer_group.capacity);

                render_pass.set_vertex_buffer(1, buffer_group.buffer.slice(..));
                render_pass.draw_indexed(0..6, 0, 0..count_for_this_buffer as u32);

                offset += count_for_this_buffer;
            }
        }
    }

    /// Encode segment rendering into an existing command encoder
    /// This is the new method for client-owned encoder pattern
    /// Handles multi-buffer architecture with multiple draw calls
    pub fn encode_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        segments: &[SegmentGpu],
        target_view: &wgpu::TextureView,
    ) {
        let (segment_count, total_line_instances) = self.update_buffer(queue, segments);

        if segment_count == 0 {
            return;
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Line Segment Render Pass"),
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
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Render from multiple buffers with separate draw calls
            let mut offset = 0;
            for buffer_group in &self.instance_buffers {
                if offset >= total_line_instances {
                    break;
                }

                let count_for_this_buffer =
                    (total_line_instances - offset).min(buffer_group.capacity);

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
