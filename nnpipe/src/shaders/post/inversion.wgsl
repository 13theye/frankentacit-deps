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

// Monochromatic inversion fragment shader with "darken darks" contrast control
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> darken_darks: f32;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(tex));
    let tex_coord = pos.xy / tex_size;
    
    let color = textureSample(tex, tex_sampler, tex_coord);
    
    // Convert to grayscale using luminance weights
    let luminance = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    
    // Apply "darken darks" enhancement - only affects dark regions
    // Formula: luminance * (1.0 - luminance) creates a curve that peaks at 0.5 and is 0 at extremes
    let dark_enhancement = luminance * (1.0 - luminance) * darken_darks;
    
    // Invert the enhanced luminance
    let inverted_luminance = 1.0 - (luminance + dark_enhancement);
    
    // Create monochromatic output by using inverted luminance for all channels
    return vec4<f32>(inverted_luminance, inverted_luminance, inverted_luminance, color.a);
}