// Simple passthrough shader to copy input texture to output
// Used for copying clean input frames to history buffer

@vertex
fn vs_main(@builtin(vertex_index) vert_id: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    
    return vec4<f32>(positions[vert_id], 0.0, 1.0);
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(input_texture));
    let tex_coord = pos.xy / tex_size;
    
    // Simple passthrough - just copy input to output
    return textureSample(input_texture, tex_sampler, tex_coord);
}