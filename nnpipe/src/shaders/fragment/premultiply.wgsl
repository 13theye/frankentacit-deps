// Premultiply Conversion Shader
//
// Converts a texture from straight alpha to premultiplied alpha.
//
// ## Purpose
//
// This shader marks the boundary between RENDERING and COMPOSITING stages.
// All nnpipe renderers (particles, segments) and external sources (Nannou Draw)
// output straight alpha. This shader converts to premultiplied alpha before
// compositing operations.
//
// ## Alpha Formats
//
// - STRAIGHT ALPHA: RGB and A are independent
//   "This pixel is red (1,0,0) at 50% opacity (0.5)"
//   Stored as: (1.0, 0.0, 0.0, 0.5)
//
// - PREMULTIPLIED ALPHA: RGB is pre-multiplied by A
//   Same pixel stored as: (0.5, 0.0, 0.0, 0.5)
//   Transparent pixels are always (0, 0, 0, 0)
//
// ## Conversion
//
// output.rgb = input.rgb × input.a
// output.a   = input.a
//
// ## Usage
//
// ```
// // Particles and masks render to same texture (both straight alpha)
// particle_renderer.encode_only(&mut encoder, count, straight_texture);
// mask.draw(&rendering.draw);
// rendering.encode_draw_commands_into(device, &mut encoder, "straight_texture");
//
// // Premultiply before compositing
// PipelineBuilder::new()
//     .input_texture("straight_texture")
//     .output_texture("premultiplied_texture")
//     .premultiply(config)
//     .build(device)
// ```

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate fullscreen triangle vertices
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );

    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );

    return VertexOutput(
        vec4<f32>(pos[vertex_index], 0.0, 1.0),
        uv[vertex_index]
    );
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(input_texture, texture_sampler, in.uv);

    // Convert straight alpha to premultiplied alpha
    // RGB values are multiplied by alpha
    return vec4<f32>(color.rgb * color.a, color.a);
}
