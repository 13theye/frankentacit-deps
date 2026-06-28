// Line Segment Renderer Shader
//
// Renders instanced line segments as oriented quads. Each segment has a start
// position, end position, color, and line width.
//
// ## Alpha Handling
//
// This shader outputs straight alpha (RGB and A are independent). This allows
// segments to be rendered to the same texture as other straight-alpha content
// (e.g., Nannou Draw masks) before a single premultiply step prior to compositing.
//
// The pipeline flow is:
//   1. Segments render with straight alpha (this shader)
//   2. Other straight-alpha content renders to the same texture
//   3. A premultiply pass converts the texture to premultiplied alpha
//   4. Compositing operations use the premultiplied texture
//
// ## Blend State
//
// This shader expects ALPHA_BLENDING (standard straight alpha blending):
//   result.rgb = src.rgb × src.a + dst.rgb × (1 - src.a)
//   result.a   = src.a   + dst.a   × (1 - src.a)

struct VertexInput {
    @location(0) position: vec2<f32>, // Local quad position [-1, 1]
}

struct InstanceInput {
    @location(1) start_pos: vec2<f32>,  // Start position of line segment
    @location(2) end_pos: vec2<f32>,    // End position of line segment
    @location(3) color_packed: u32,     // RGBA8 packed color with alpha
    @location(4) line_width: f32,       // Line thickness in pixels for this segment
}

// Unpack RGBA8 u32 color to vec4<f32>
// Layout: 0xAABBGGRR (little-endian)
fn unpack_color(packed: u32) -> vec4<f32> {
    let r = f32((packed >> 0u) & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}


@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Calculate line properties from instance data
    let dx = instance.end_pos.x - instance.start_pos.x;
    let dy = instance.end_pos.y - instance.start_pos.y;
    let line_length = sqrt(dx * dx + dy * dy);
    
    // Skip zero-length segments
    if (line_length < 0.001) {
        out.clip_position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    let rotation = atan2(dy, dx);
    
    // Scale the quad vertex position
    var scaled_pos = vertex.position;
    scaled_pos.x = scaled_pos.x * line_length * 0.5; // Half length in each direction
    scaled_pos.y = scaled_pos.y * instance.line_width * 0.5; // Half width
    
    // Rotate the line to align with the direction between points
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);
    
    let rotated_x = scaled_pos.x * cos_r - scaled_pos.y * sin_r;
    let rotated_y = scaled_pos.x * sin_r + scaled_pos.y * cos_r;
    scaled_pos = vec2<f32>(rotated_x, rotated_y);
    
    // Translate to midpoint of the line
    let mid_x = (instance.start_pos.x + instance.end_pos.x) * 0.5;
    let mid_y = (instance.start_pos.y + instance.end_pos.y) * 0.5;
    let world_pos = scaled_pos + vec2<f32>(mid_x, mid_y);
    
    // Convert from Nannou's coordinate system to clip space
    let texture_width = 3840.0;
    let texture_height = 2160.0;
    
    let clip_x = world_pos.x / (texture_width * 0.5);
    let clip_y = world_pos.y / (texture_height * 0.5);
    
    out.clip_position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    out.color = unpack_color(instance.color_packed);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Output straight alpha: RGB and alpha are independent
    // This allows mixing with other straight-alpha content before premultiplying
    return in.color;
}