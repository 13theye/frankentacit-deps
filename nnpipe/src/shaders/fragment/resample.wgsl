// Vertex shader for a fullscreen triangle with UV coordinates
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

// General resample fragment shader - automatically handles scaling and format conversion
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Use linear filtering for smooth scaling between any input/output sizes
    // The UV coordinates from the vertex shader map (0,0) to (1,1) across the output
    // The sampler's linear filtering automatically handles resampling between different sizes
    return textureSample(tex, tex_sampler, in.uv);
}