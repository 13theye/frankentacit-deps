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

// Color key extraction fragment shader
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct ColorKeyUniforms {
    target_color: vec3<f32>,
    threshold: f32,
    intensity: f32,
    _padding: array<u32, 3>, // Align to 16 bytes
}

@group(0) @binding(2) var<uniform> uniforms: ColorKeyUniforms;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(tex));
    let tex_coord = pos.xy / tex_size;
    
    let color = textureSample(tex, tex_sampler, tex_coord);
    
    // Calculate color distance from target color
    let color_diff = distance(color.rgb, uniforms.target_color);
    
    // Convert distance to similarity (closer = higher similarity)
    let threshold = uniforms.threshold;
    let knee = 0.1; // Softness of the threshold
    
    // Soft thresholding - closer colors get higher values
    let similarity = 1.0 - smoothstep(0.0, threshold + knee, color_diff);
    
    // Apply intensity scaling
    let bloom_intensity = pow(similarity, 1.2) * uniforms.intensity;
    
    // Apply to color and store similarity in alpha for later stages
    return vec4<f32>(color.rgb * bloom_intensity, similarity);
}