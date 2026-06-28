// Simple passthrough shader - copies input texture to output
// No processing, just direct texture copy

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate fullscreen triangle vertices
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),  // Bottom-left
        vec2<f32>( 3.0, -1.0),  // Bottom-right (extended)
        vec2<f32>(-1.0,  3.0)   // Top-left (extended)
    );
    
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),    // Bottom-left UV
        vec2<f32>(2.0, 1.0),    // Bottom-right UV (extended)
        vec2<f32>(0.0, -1.0)    // Top-left UV (extended)
    );
    
    return VertexOutput(
        vec4<f32>(pos[vertex_index], 0.0, 1.0),
        uv[vertex_index]
    );
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Direct texture sample - no processing
    return textureSample(input_texture, texture_sampler, in.uv);
}