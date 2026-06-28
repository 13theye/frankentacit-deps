// Particle Renderer Shader
//
// Renders instanced point particles as small quads. Each particle has a position
// and a packed RGBA color.
//
// ## Alpha Handling
//
// This shader outputs straight alpha (RGB and A are independent). This allows
// particles to be rendered to the same texture as other straight-alpha content
// (e.g., Nannou Draw masks) before a single premultiply step prior to compositing.
//
// The pipeline flow is:
//   1. Particles render with straight alpha (this shader)
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
    @location(0) position: vec2<f32>,
}

struct InstanceInput {
    @location(1) particle_pos: vec2<f32>,
    @location(2) color_packed: u32,
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

struct ParticleUniforms {
    texture_width: f32,
    texture_height: f32,
    _padding0: f32,
    _padding1: f32,
}

@group(0) @binding(0) var<uniform> uniforms: ParticleUniforms;

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;

    // Scale the quad to particle size and position it
    let particle_size = 2.0; // 2x2 pixel particles
    let scaled_pos = vertex.position * particle_size;
    let world_pos = scaled_pos + instance.particle_pos;

    // Convert from Nannou's coordinate system to clip space
    // Nannou: (0,0) at center, Y+ is up
    // Clip space: (0,0) at center, range [-1, 1], Y+ is up
    // World extends from [-width/2, width/2] x [-height/2, height/2]

    let clip_x = world_pos.x / (uniforms.texture_width * 0.5);  // [-width/2, width/2] -> [-1, 1]
    let clip_y = world_pos.y / (uniforms.texture_height * 0.5); // [-height/2, height/2] -> [-1, 1]

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