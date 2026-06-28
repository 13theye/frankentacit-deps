// Point spatial binning compute shader
// Sorts points into spatial grid bins for efficient heatmap computation

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

@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> bin_counts: array<atomic<u32>>;      // Count per bin
@group(0) @binding(2) var<storage, read_write> bin_data: array<u32>;               // Particle indices per bin
@group(0) @binding(3) var<storage, read_write> bin_offsets: array<u32>;            // Start offset for each bin
@group(0) @binding(4) var<uniform> params: HeatmapParams;
@group(0) @binding(5) var<storage, read_write> bin_fill_counts: array<atomic<u32>>; // Separate counter for fill stage

// Helper function to calculate bin index for a particle
// Returns bin_idx if valid, or u32(-1) if particle should be skipped
fn calculate_bin_index(point: Point) -> u32 {
    // Skip invalid particles
    if (point.position.x > 1e30 || point.position.y > 1e30) {
        return 0xFFFFFFFFu; // u32(-1)
    }
    
    // Calculate which bin this point belongs to
    // Use direct coordinate transformation to match heatmap shader exactly
    let normalized_pos = (point.position - params.bounds_min) / (params.bounds_max - params.bounds_min);
    
    // Skip particles outside bounds
    if (normalized_pos.x < 0.0 || normalized_pos.x > 1.0 || normalized_pos.y < 0.0 || normalized_pos.y > 1.0) {
        return 0xFFFFFFFFu; // u32(-1)
    }
    
    // Direct bin coordinate calculation without precision-losing round trips
    let bin_x_f = normalized_pos.x * f32(params.grid_size.x);
    let bin_y_f = normalized_pos.y * f32(params.grid_size.y);
    
    // Final bounds check and conversion
    if (bin_x_f >= 0.0 && bin_x_f < f32(params.grid_size.x) && bin_y_f >= 0.0 && bin_y_f < f32(params.grid_size.y)) {
        let bin_x = u32(bin_x_f);
        let bin_y = u32(bin_y_f);
        return bin_y * params.grid_size.x + bin_x;
    }
    
    return 0xFFFFFFFFu; // u32(-1)
}

// First pass: count particles per bin
@compute @workgroup_size(64, 1, 1)
fn count_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_idx = global_id.x;
    
    if (point_idx >= params.point_count) {
        return;
    }
    
    let point = points[point_idx];
    let bin_idx = calculate_bin_index(point);
    
    if (bin_idx != 0xFFFFFFFFu) {
        // Atomically increment the count for this bin
        atomicAdd(&bin_counts[bin_idx], 1u);
    }
}

// Second pass: calculate prefix sums to get bin offsets
@compute @workgroup_size(1, 1, 1) 
fn calculate_offsets(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Single-threaded prefix sum calculation to avoid race conditions
    if (global_id.x != 0u) {
        return;
    }
    
    let total_bins = params.grid_size.x * params.grid_size.y;
    var running_offset = 0u;
    
    // Sequential prefix sum - only thread 0 does this work
    for (var bin_idx = 0u; bin_idx < total_bins; bin_idx++) {
        bin_offsets[bin_idx] = running_offset;
        running_offset += atomicLoad(&bin_counts[bin_idx]);
    }
}

// Third pass: fill bin data with particle indices
@compute @workgroup_size(64, 1, 1)
fn fill_bins(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let point_idx = global_id.x;
    
    if (point_idx >= params.point_count) {
        return;
    }
    
    let point = points[point_idx];
    let bin_idx = calculate_bin_index(point);
    
    if (bin_idx != 0xFFFFFFFFu) {
        // Get the current write position for this bin using separate fill counter
        let write_pos = bin_offsets[bin_idx] + atomicAdd(&bin_fill_counts[bin_idx], 1u);
        
        // Bounds check to prevent overflow
        if (write_pos < arrayLength(&bin_data)) {
            bin_data[write_pos] = point_idx;
        }
    }
}