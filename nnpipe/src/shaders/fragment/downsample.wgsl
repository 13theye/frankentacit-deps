// Vertex shader for a fullscreen triangle
@vertex
fn vs_main(@builtin(vertex_index) vert_id: u32) -> @builtin(position) vec4<f32> {
    // Create a fullscreen triangle with just the vertex id
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    
    return vec4<f32>(positions[vert_id], 0.0, 1.0);
}

// High-quality downsample fragment shader with 2x2 box filter
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(tex));
    let out_size = tex_size * 0.5;
    
    // Calculate texture coordinates for full-size input
    // Maps half-res position to full-res texture
    let tex_coord = pos.xy / out_size;
    
    // Apply 2x2 box filter for higher quality downsampling
    let step_size = vec2<f32>(1.0) / tex_size;
    var color = vec4<f32>(0.0);
    
    // 2x2 box filter
    color += textureSample(tex, tex_sampler, tex_coord);
    color += textureSample(tex, tex_sampler, tex_coord + vec2<f32>(step_size.x, 0.0));
    color += textureSample(tex, tex_sampler, tex_coord + vec2<f32>(0.0, step_size.y));
    color += textureSample(tex, tex_sampler, tex_coord + step_size);
    
    return color * 0.25; // Average the samples
}