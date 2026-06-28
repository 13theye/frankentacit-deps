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

// Feedback/trail parameters
struct FeedbackParams {
    persistence: f32,   // How strong the oldest frames can be seen (0.0 = invisible, 1.0 = full strength)
    frame_history: f32, // Number of previous frames to show (0.0 = no trail, 1.0 = max trail length)
    _padding1: u32,
    _padding2: u32,
}

// Feedback/trail fragment shader with 8 previous frames
@group(0) @binding(0) var current_frame: texture_2d<f32>;   // Current frame input
@group(0) @binding(1) var history_frame_1: texture_2d<f32>; // Previous frame 1 (most recent)
@group(0) @binding(2) var history_frame_2: texture_2d<f32>; // Previous frame 2
@group(0) @binding(3) var history_frame_3: texture_2d<f32>; // Previous frame 3
@group(0) @binding(4) var history_frame_4: texture_2d<f32>; // Previous frame 4
@group(0) @binding(5) var history_frame_5: texture_2d<f32>; // Previous frame 5
@group(0) @binding(6) var history_frame_6: texture_2d<f32>; // Previous frame 6
@group(0) @binding(7) var history_frame_7: texture_2d<f32>; // Previous frame 7
@group(0) @binding(8) var history_frame_8: texture_2d<f32>; // Previous frame 8 (oldest)
@group(0) @binding(9) var tex_sampler: sampler;
@group(0) @binding(10) var<uniform> params: FeedbackParams;

// Create ONLY trail blur, don't preserve original particles
fn create_trail_blur_only(frame: texture_2d<f32>, coord: vec2<f32>, tex_size: vec2<f32>) -> vec4<f32> {
    let pixel_size = 1.0 / tex_size;
    let r = 2.0; // blur radius in pixels
    
    // Sample center and nearby pixels in 8 directions
    let r0 = textureSample(frame, tex_sampler, coord);
    let r1 = textureSample(frame, tex_sampler, coord + vec2<f32>(r * pixel_size.x, 0.0));
    let r2 = textureSample(frame, tex_sampler, coord + vec2<f32>(-r * pixel_size.x, 0.0));
    let r3 = textureSample(frame, tex_sampler, coord + vec2<f32>(0.0, r * pixel_size.y));
    let r4 = textureSample(frame, tex_sampler, coord + vec2<f32>(0.0, -r * pixel_size.y));
    let r5 = textureSample(frame, tex_sampler, coord + vec2<f32>(r * pixel_size.x * 0.707, r * pixel_size.y * 0.707));
    let r6 = textureSample(frame, tex_sampler, coord + vec2<f32>(-r * pixel_size.x * 0.707, -r * pixel_size.y * 0.707));
    let r7 = textureSample(frame, tex_sampler, coord + vec2<f32>(r * pixel_size.x * 0.707, -r * pixel_size.y * 0.707));
    let r8 = textureSample(frame, tex_sampler, coord + vec2<f32>(-r * pixel_size.x * 0.707, r * pixel_size.y * 0.707));
    
    // Create trail effect by averaging all samples including center
    let trail_blur = (r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8) / 9.0;
    
    // Return the blur effect
    return trail_blur * 0.5; // Dim the trail effect
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(current_frame));
    let tex_coord = pos.xy / tex_size;
    
    // Sample current frame (bright particles)
    let current_color = textureSample(current_frame, tex_sampler, tex_coord);
    
    // Create blurred trail from history frames (ONLY blur, no particles)
    let h1 = create_trail_blur_only(history_frame_1, tex_coord, tex_size);
    let h2 = create_trail_blur_only(history_frame_2, tex_coord, tex_size);
    let h3 = create_trail_blur_only(history_frame_3, tex_coord, tex_size);
    let h4 = create_trail_blur_only(history_frame_4, tex_coord, tex_size);
    let h5 = create_trail_blur_only(history_frame_5, tex_coord, tex_size);
    let h6 = create_trail_blur_only(history_frame_6, tex_coord, tex_size);
    let h7 = create_trail_blur_only(history_frame_7, tex_coord, tex_size);
    let h8 = create_trail_blur_only(history_frame_8, tex_coord, tex_size);
    
    // Combine all history trails into one blurred trail effect
    let base_weight = params.persistence * params.frame_history;
    var trail = vec4<f32>(0.0);
    trail += h1 * base_weight * 0.7;
    trail += h2 * base_weight * 0.6;
    trail += h3 * base_weight * 0.5;
    trail += h4 * base_weight * 0.4;
    trail += h5 * base_weight * 0.3;
    trail += h6 * base_weight * 0.2;
    trail += h7 * base_weight * 0.15;
    trail += h8 * base_weight * 0.1;
    
    // Final result: current particles + blurred trail
    var result = current_color + trail;
    
    // Clamp to prevent overbrightening
    return vec4<f32>(min(result.rgb, vec3<f32>(1.0)), result.a);
}