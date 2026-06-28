// Simple Composite Shader
//
// Composites two textures using various blend modes.
//
// ## Alpha Handling
//
// - INPUT: Straight alpha (RGB and A are independent)
//   This is the format output by nnpipe's particle/segment renderers and Nannou Draw.
//
// - OUTPUT: Premultiplied alpha (RGB × A)
//   Ready for subsequent compositing operations or final display.
//
// The shader premultiplies inputs internally before blending. This allows a single
// shader pass to replace what would otherwise require separate premultiply passes.
//
// ## Available Blend Modes
//
// - OVER (0): Porter-Duff over compositing
//   Effect layer composites over scene layer based on alpha.
//   Formula: result = effect + scene × (1 - effect.a)
//
// - ADD (1): Additive blending
//   Layers add together, causing overlapping areas to brighten.
//   Formula: result = scene + effect
//   Good for: glowing particles, light effects
//
// - SCREEN (2): Screen blending
//   Lightens the image, similar to projecting two slides.
//   Formula: 1 - (1 - A) × (1 - B)
//
// - MULTIPLY (3): Multiply blending
//   Darkens the image by multiplying colors.
//   Formula: A × B
//
// - OVERLAY (4): Overlay blending
//   Combines multiply and screen based on base color.
//
// - LIGHTEN (5): Lighten blending
//   Takes the brighter pixel by luminance.
//   Transparent areas automatically lose (luminance = 0).
//   Good for: combining particle layers where brightest wins
//
// ## Intensity Parameter
//
// The intensity uniform scales the effect layer's contribution before blending.

@vertex
fn vs_main(@builtin(vertex_index) vert_id: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    return vec4<f32>(positions[vert_id], 0.0, 1.0);
}

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var effect_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> intensity: f32;
@group(0) @binding(4) var<uniform> blend_mode: u32;

// Blend mode constants
const BLEND_OVER: u32 = 0u;
const BLEND_ADD: u32 = 1u;
const BLEND_SCREEN: u32 = 2u;
const BLEND_MULTIPLY: u32 = 3u;
const BLEND_OVERLAY: u32 = 4u;
const BLEND_LIGHTEN: u32 = 5u;

// Helper: Convert straight to premultiplied alpha
fn premultiply(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * color.a, color.a);
}

// Helper: Calculate luminance (for premultiplied input, luminance is already alpha-weighted)
fn luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

// Blend two straight-alpha colors and output premultiplied alpha
fn blend_colors(scene_straight: vec4<f32>, effect_straight: vec4<f32>, mode: u32, intensity: f32) -> vec4<f32> {
    // Scale effect by intensity (in straight alpha space, only scale RGB)
    let scaled_effect = vec4<f32>(effect_straight.rgb * intensity, effect_straight.a * intensity);

    if mode == BLEND_OVER {
        // Porter-Duff Over: premultiply inputs, then blend
        let scene_pre = premultiply(scene_straight);
        let effect_pre = premultiply(scaled_effect);
        let result_rgb = effect_pre.rgb + scene_pre.rgb * (1.0 - effect_pre.a);
        let result_a = effect_pre.a + scene_pre.a * (1.0 - effect_pre.a);
        return vec4<f32>(result_rgb, result_a);

    } else if mode == BLEND_ADD {
        // Additive blend: premultiply inputs, then add
        let scene_pre = premultiply(scene_straight);
        let effect_pre = premultiply(scaled_effect);
        let result_rgb = scene_pre.rgb + effect_pre.rgb;
        let result_a = min(scene_pre.a + effect_pre.a, 1.0);
        return vec4<f32>(result_rgb, result_a);

    } else if mode == BLEND_SCREEN {
        // Screen blend: operate on straight alpha RGB, then premultiply result
        // Screen formula: 1 - (1 - A) * (1 - B)  =  A + B - A*B
        let blended_rgb = scene_straight.rgb + scaled_effect.rgb
                        - (scene_straight.rgb * scaled_effect.rgb);
        // Alpha: standard over compositing
        let blended_a = scaled_effect.a + scene_straight.a * (1.0 - scaled_effect.a);
        // Output premultiplied
        return vec4<f32>(blended_rgb * blended_a, blended_a);

    } else if mode == BLEND_MULTIPLY {
        // Multiply blend: operate on straight alpha RGB, then premultiply result
        let blended_rgb = scene_straight.rgb * scaled_effect.rgb;
        let blended_a = scaled_effect.a + scene_straight.a * (1.0 - scaled_effect.a);
        return vec4<f32>(blended_rgb * blended_a, blended_a);

    } else if mode == BLEND_OVERLAY {
        // Overlay: operate on straight alpha RGB, then premultiply result
        var result: vec3<f32>;
        // Overlay formula: multiply if < 0.5, screen if >= 0.5
        if scene_straight.r < 0.5 {
            result.r = 2.0 * scene_straight.r * scaled_effect.r;
        } else {
            result.r = 1.0 - 2.0 * (1.0 - scene_straight.r) * (1.0 - scaled_effect.r);
        }
        if scene_straight.g < 0.5 {
            result.g = 2.0 * scene_straight.g * scaled_effect.g;
        } else {
            result.g = 1.0 - 2.0 * (1.0 - scene_straight.g) * (1.0 - scaled_effect.g);
        }
        if scene_straight.b < 0.5 {
            result.b = 2.0 * scene_straight.b * scaled_effect.b;
        } else {
            result.b = 1.0 - 2.0 * (1.0 - scene_straight.b) * (1.0 - scaled_effect.b);
        }
        let blended_a = scaled_effect.a + scene_straight.a * (1.0 - scaled_effect.a);
        return vec4<f32>(result * blended_a, blended_a);

    } else if mode == BLEND_LIGHTEN {
        // Lighten: premultiply inputs, compare luminance, output winner
        let scene_pre = premultiply(scene_straight);
        let effect_pre = premultiply(scaled_effect);
        // Luminance of premultiplied values is already alpha-weighted
        let scene_lum = luminance(scene_pre.rgb);
        let effect_lum = luminance(effect_pre.rgb);
        if effect_lum > scene_lum {
            return effect_pre;
        } else {
            return scene_pre;
        }

    } else {
        // Fallback to additive
        let scene_pre = premultiply(scene_straight);
        let effect_pre = premultiply(scaled_effect);
        let result_rgb = scene_pre.rgb + effect_pre.rgb;
        let result_a = min(scene_pre.a + effect_pre.a, 1.0);
        return vec4<f32>(result_rgb, result_a);
    }
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(scene_tex));
    let tex_coord = pos.xy / tex_size;

    let scene_color = textureSample(scene_tex, tex_sampler, tex_coord);
    let effect_color = textureSample(effect_tex, tex_sampler, tex_coord);

    return blend_colors(scene_color, effect_color, blend_mode, intensity);
}
