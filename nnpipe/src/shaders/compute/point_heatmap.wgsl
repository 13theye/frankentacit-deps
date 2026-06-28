// Heatmap generation compute shader
// Uses spatial binning to efficiently process points for heatmap generation

struct Point {
    position: vec2<f32>,
    padding: vec2<f32>,
}

struct HeatmapParams {
    resolution: vec2<f32>,
    bounds_min: vec2<f32>,
    bounds_max: vec2<f32>,
    grid_size: vec2<u32>,
    bin_cell_size: vec2<f32>,
    max_influence_radius: f32,
    intensity_scale: f32,
    point_count: u32,
    padding: u32,
}

@group(0) @binding(0) var heatmap_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<storage, read> point: array<Point>;
@group(0) @binding(2) var<storage, read> bin_counts: array<u32>;
@group(0) @binding(3) var<storage, read> bin_data: array<u32>;
@group(0) @binding(4) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(5) var<uniform> params: HeatmapParams;

// Deep red/magenta/orange gradient function
fn magenta_range_from_intensity(intensity: f32) -> vec4<f32> {
    let clamped_intensity = clamp(intensity, 0.0, 1.0);
    
    if (clamped_intensity < 0.25) {
        // Dark red to deep magenta
        let t = clamped_intensity / 0.25;
        let dark_red = vec3<f32>(0.2, 0.0, 0.0);
        let deep_magenta = vec3<f32>(0.4, 0.0, 0.2);
        let rgb = dark_red + (deep_magenta - dark_red) * t;
        return vec4<f32>(rgb, 0.7 + clamped_intensity * 0.2);
    } else if (clamped_intensity < 0.5) {
        // Deep magenta to bright magenta
        let t = (clamped_intensity - 0.25) / 0.25;
        let deep_magenta = vec3<f32>(0.4, 0.0, 0.2);
        let bright_magenta = vec3<f32>(0.8, 0.0, 0.4);
        let rgb = deep_magenta + (bright_magenta - deep_magenta) * t;
        return vec4<f32>(rgb, 0.8 + clamped_intensity * 0.1);
    } else if (clamped_intensity < 0.75) {
        // Bright magenta to red-orange
        let t = (clamped_intensity - 0.5) / 0.25;
        let bright_magenta = vec3<f32>(0.8, 0.0, 0.4);
        let red_orange = vec3<f32>(1.0, 0.3, 0.0);
        let rgb = bright_magenta + (red_orange - bright_magenta) * t;
        return vec4<f32>(rgb, 0.85 + clamped_intensity * 0.1);
    } else {
        // Red-orange to bright orange-yellow
        let t = (clamped_intensity - 0.75) / 0.25;
        let red_orange = vec3<f32>(1.0, 0.3, 0.0);
        let bright_orange = vec3<f32>(1.0, 0.6, 0.0);
        let yellow_orange = vec3<f32>(1.0, 0.8, 0.2);
        
        var rgb: vec3<f32>;
        if (t < 0.5) {
            rgb = red_orange + (bright_orange - red_orange) * (t * 2.0);
        } else {
            rgb = bright_orange + (yellow_orange - bright_orange) * ((t - 0.5) * 2.0);
        }
        return vec4<f32>(rgb, 0.9);
    }
}

// Grayscale gradient function for HDR
fn color_from_intensity(intensity: f32) -> vec4<f32> {
    // Allow intensity to go beyond 1.0 for HDR
    let gray_value = max(0.0, 1.0 - intensity);
    let alpha = min(1.0, 0.7 + intensity * 0.2);
    return vec4<f32>(gray_value, gray_value, gray_value, alpha);
}


// Check if a bin coordinate is valid
fn is_valid_bin(bin_coord: vec2<i32>) -> bool {
    return bin_coord.x >= 0 && bin_coord.x < i32(params.grid_size.x) &&
           bin_coord.y >= 0 && bin_coord.y < i32(params.grid_size.y);
}

// Convert 2D bin coordinate to 1D index
fn bin_coord_to_index(bin_coord: vec2<i32>) -> u32 {
    return u32(bin_coord.y) * params.grid_size.x + u32(bin_coord.x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coords = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let resolution = vec2<i32>(i32(params.resolution.x), i32(params.resolution.y));
    
    // Check bounds
    if (pixel_coords.x >= resolution.x || pixel_coords.y >= resolution.y) {
        return;
    }
    
    // Convert pixel coordinates to world coordinates (match CPU implementation)
    let bounds_width = params.bounds_max.x - params.bounds_min.x;
    let bounds_height = params.bounds_max.y - params.bounds_min.y;
    let pixel_size_x = bounds_width / params.resolution.x;
    let pixel_size_y = bounds_height / params.resolution.y;
    
    let pixel_world_x = params.bounds_min.x + (f32(pixel_coords.x) + 0.5) * pixel_size_x;
    // Flip Y coordinate to match Nannou's coordinate system
    let flipped_y = params.resolution.y - 1.0 - f32(pixel_coords.y);
    let pixel_world_y = params.bounds_min.y + (flipped_y + 0.5) * pixel_size_y;
    let pixel_pos = vec2<f32>(pixel_world_x, pixel_world_y);
    
    var total_intensity = 0.0;
    let max_influence_radius_sq = params.max_influence_radius * params.max_influence_radius;
    
    // Process particles in nearby bins (3x3 neighborhood)
    // Unroll the loop to avoid variable array indexing
    let normalized_pos = (pixel_pos - params.bounds_min) / (params.bounds_max - params.bounds_min);
    let bin_pos = normalized_pos * vec2<f32>(params.grid_size);
    let center_bin = vec2<i32>(i32(bin_pos.x), i32(bin_pos.y));
    
    // Process 3x3 neighborhood around the pixel's bin
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let bin_coord = center_bin + vec2<i32>(dx, dy);
            
            if (!is_valid_bin(bin_coord)) {
                continue;
            }
            
            let bin_linear_idx = bin_coord_to_index(bin_coord);
            let point_count = bin_counts[bin_linear_idx];
            let bin_start_offset = bin_offsets[bin_linear_idx];
            
            // Process particles in this bin
            for (var i = 0u; i < point_count; i++) {
                let point_idx = bin_data[bin_start_offset + i];
                
                // Bounds check
                if (point_idx >= params.point_count) {
                    continue;
                }
                
                let point_pos = point[point_idx].position;
                
                // Skip invalid particles
                if (point_pos.x > 1e30 || point_pos.y > 1e30) {
                    continue;
                }
                
                let dx_p = pixel_pos.x - point_pos.x;
                let dy_p = pixel_pos.y - point_pos.y;
                let distance_sq = dx_p * dx_p + dy_p * dy_p;
                
                // Early exit for distant particles
                if (distance_sq > max_influence_radius_sq) {
                    continue;
                }
                
                // Gaussian falloff - matches original sigma = 25.0
                let sigma = 25.0;
                let influence = exp(-distance_sq / (2.0 * sigma * sigma));
                total_intensity += influence;
                
                // Early exit if we have enough intensity
                if (total_intensity > 3.0) {
                    break;
                }
            }
            
            // Early exit if we already have enough intensity
            if (total_intensity > 3.0) {
                break;
            }
        }
        
        // Early exit if we already have enough intensity
        if (total_intensity > 3.0) {
            break;
        }
    }
    
    // Skip pixels with very low intensity
    if (total_intensity < 0.001) {
        textureStore(heatmap_texture, pixel_coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    
    // For HDR, don't clamp intensity - let it go above 1.0
    let intensity = total_intensity * params.intensity_scale;
    var color = color_from_intensity(intensity);

    // Smooth alpha transition for very low intensities to reduce pop-in
    if (total_intensity < 0.05) {
        let alpha_multiplier = smoothstep(0.001, 0.05, total_intensity);
        color.a *= alpha_multiplier;
    }
    
    textureStore(heatmap_texture, pixel_coords, color);
}