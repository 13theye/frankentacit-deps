// Texture-based heatmap generation compute shader
// Samples an input particle texture and applies Gaussian blur to create heatmap
// Much more efficient than particle-based approach: O(pixels) instead of O(particles × pixels)

struct TextureHeatmapParams {
    resolution: vec2<f32>,
    blur_radius: f32,
    intensity_scale: f32,
    _padding: vec4<f32>,
}

@group(0) @binding(0) var particle_texture: texture_2d<f32>;
@group(0) @binding(1) var heatmap_output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: TextureHeatmapParams;

// Grayscale gradient function for heatmap (matches original)
fn color_from_intensity(intensity: f32) -> vec4<f32> {
    // Allow intensity to go beyond 1.0 for HDR
    let gray_value = max(0.0, 1.0 - intensity);
    let alpha = min(1.0, 0.7 + intensity * 0.2);
    return vec4<f32>(gray_value, gray_value, gray_value, alpha);
}

// Gaussian weight function
fn gaussian_weight(distance: f32, sigma: f32) -> f32 {
    return exp(-(distance * distance) / (2.0 * sigma * sigma));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coords = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let resolution = vec2<i32>(i32(params.resolution.x), i32(params.resolution.y));
    
    // Check bounds
    if (pixel_coords.x >= resolution.x || pixel_coords.y >= resolution.y) {
        return;
    }
    
    var total_intensity = 0.0;
    let sample_radius = i32(min(params.blur_radius, 5.0)); // Cap at 5 pixels for performance
    
    // Sample in a circular region (skip corners)
    for (var dy = -sample_radius; dy <= sample_radius; dy++) {
        for (var dx = -sample_radius; dx <= sample_radius; dx++) {
            let distance = length(vec2<f32>(f32(dx), f32(dy)));
            
            // Skip samples outside circular radius
            if (distance > params.blur_radius) {
                continue;
            }
            
            let sample_coords = pixel_coords + vec2<i32>(dx, dy);
            
            // Skip samples outside texture bounds
            if (sample_coords.x < 0 || sample_coords.x >= resolution.x || 
                sample_coords.y < 0 || sample_coords.y >= resolution.y) {
                continue;
            }
            
            // Sample the particle texture
            let particle_color = textureLoad(particle_texture, sample_coords, 0);
            
            // Add particle contribution with falloff
            let particle_contribution = particle_color.a;
            if (particle_contribution > 0.001) {
                let falloff = max(0.0, 1.0 - (distance / params.blur_radius));
                total_intensity += particle_contribution * falloff;
                
                // Early termination for performance
                if (total_intensity > 2.0) {
                    break;
                }
            }
        }
        
        // Early exit from outer loop too
        if (total_intensity > 2.0) {
            break;
        }
    }
    
    // Output the heatmap
    if (total_intensity < 0.01) {
        textureStore(heatmap_output, pixel_coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    } else {
        let intensity = min(total_intensity * params.intensity_scale, 1.0);
        // Simple white heatmap for now
        textureStore(heatmap_output, pixel_coords, vec4<f32>(intensity, intensity, intensity, intensity));
    }
}