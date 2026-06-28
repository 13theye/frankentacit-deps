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

// Composite fragment shader
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> intensity_uniform: f32;
@group(0) @binding(4) var<uniform> intensity_curve: f32;


@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(scene_tex));
    let tex_coord = pos.xy / tex_size;
    
    // Sample original scene
    let scene_color = textureSample(scene_tex, tex_sampler, tex_coord);
    
    // Sample bloom texture
    let bloom_color = textureSample(bloom_tex, tex_sampler, tex_coord);
    
    // Get scene brightness
    let scene_luminance = dot(scene_color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    
    // Get bloom brightness info passed through the pipeline
    let bloom_brightness = bloom_color.a;
    
    // Enhanced adaptive intensity calculation
    // Base scaling on both scene brightness and bloom brightness
    let base_intensity = intensity_uniform;
    let min_intensity = 0.3;
    let max_intensity = 2.0;
    
    // Scale intensity non-linearly with brightness
    // This creates a more dramatic effect for very bright areas
    let brightness_factor = pow(max(scene_luminance, bloom_brightness), intensity_curve);
    let adaptive_intensity = mix(min_intensity, max_intensity, brightness_factor);
    
    // Apply HDR-like tone mapping to prevent over-saturation
    let bloom_contribution = bloom_color.rgb * base_intensity * adaptive_intensity;
    let combined = scene_color.rgb + bloom_contribution;
    
    // Basic tone mapping to prevent excessive brightness
    let mapped = combined / (combined + 1.0);
    
    return vec4<f32>(mapped, scene_color.a);
}