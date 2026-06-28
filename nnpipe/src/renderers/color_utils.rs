// Color packing utilities for GPU renderers
//
// Optimizes memory bandwidth by converting RGB(A) float colors to packed RGBA8 u32 format.
// This reduces color storage from 12-16 bytes to 4 bytes per color while maintaining
// standard 8-bit color precision (more than sufficient for visual applications).

/// Pack RGB floats (0.0-1.0) into RGBA8 u32 format with full alpha
///
/// Layout: 0xAABBGGRR (little-endian):
/// - Bits 0-7: Red
/// - Bits 8-15: Green
/// - Bits 16-23: Blue
/// - Bits 24-31: Alpha (always 255)
///
/// # Arguments
/// * `r` - Red channel (0.0-1.0), will be clamped
/// * `g` - Green channel (0.0-1.0), will be clamped
/// * `b` - Blue channel (0.0-1.0), will be clamped
///
/// # Returns
/// Packed u32 color with full alpha (255)
#[inline]
pub fn pack_color_rgb(r: f32, g: f32, b: f32) -> u32 {
    pack_color_rgba(r, g, b, 1.0)
}

/// Pack RGBA floats (0.0-1.0) into RGBA8 u32 format
///
/// Layout: 0xAABBGGRR (little-endian):
/// - Bits 0-7: Red
/// - Bits 8-15: Green
/// - Bits 16-23: Blue
/// - Bits 24-31: Alpha
///
/// # Arguments
/// * `r` - Red channel (0.0-1.0), will be clamped
/// * `g` - Green channel (0.0-1.0), will be clamped
/// * `b` - Blue channel (0.0-1.0), will be clamped
/// * `a` - Alpha channel (0.0-1.0), will be clamped
///
/// # Returns
/// Packed u32 color in RGBA8 format
#[inline]
pub fn pack_color_rgba(r: f32, g: f32, b: f32, a: f32) -> u32 {
    let r_byte = (r.clamp(0.0, 1.0) * 255.0) as u32;
    let g_byte = (g.clamp(0.0, 1.0) * 255.0) as u32;
    let b_byte = (b.clamp(0.0, 1.0) * 255.0) as u32;
    let a_byte = (a.clamp(0.0, 1.0) * 255.0) as u32;

    // Pack into little-endian RGBA8 format: 0xAABBGGRR
    (a_byte << 24) | (b_byte << 16) | (g_byte << 8) | r_byte
}

/// Unpack u32 RGBA8 to vec4<f32> equivalent [r, g, b, a]
///
/// Primarily for testing/debugging. GPU shaders should use the WGSL unpack function.
///
/// # Arguments
/// * `packed` - Packed u32 color in RGBA8 format (0xAABBGGRR)
///
/// # Returns
/// Array of [r, g, b, a] floats in range 0.0-1.0
#[inline]
pub fn unpack_color(packed: u32) -> [f32; 4] {
    let r = ((packed) & 0xFF) as f32 / 255.0;
    let g = ((packed >> 8) & 0xFF) as f32 / 255.0;
    let b = ((packed >> 16) & 0xFF) as f32 / 255.0;
    let a = ((packed >> 24) & 0xFF) as f32 / 255.0;
    [r, g, b, a]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_rgb_pure_colors() {
        // Pure red
        let red = pack_color_rgb(1.0, 0.0, 0.0);
        assert_eq!(red, 0xFF0000FF); // 0xAABBGGRR

        // Pure green
        let green = pack_color_rgb(0.0, 1.0, 0.0);
        assert_eq!(green, 0xFF00FF00);

        // Pure blue
        let blue = pack_color_rgb(0.0, 0.0, 1.0);
        assert_eq!(blue, 0xFFFF0000);

        // White
        let white = pack_color_rgb(1.0, 1.0, 1.0);
        assert_eq!(white, 0xFFFFFFFF);

        // Black
        let black = pack_color_rgb(0.0, 0.0, 0.0);
        assert_eq!(black, 0xFF000000);
    }

    #[test]
    fn test_pack_rgba_with_alpha() {
        // Semi-transparent red
        let red_half = pack_color_rgba(1.0, 0.0, 0.0, 0.5);
        assert_eq!(red_half, 0x7F0000FF); // Alpha ~0.5 = 127

        // Fully transparent
        let transparent = pack_color_rgba(1.0, 1.0, 1.0, 0.0);
        assert_eq!(transparent, 0x00FFFFFF);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let original = [0.25, 0.5, 0.75, 1.0];
        let packed = pack_color_rgba(original[0], original[1], original[2], original[3]);
        let unpacked = unpack_color(packed);

        // Should be within 1/255 precision
        for i in 0..4 {
            assert!(
                (unpacked[i] - original[i]).abs() < 0.01,
                "Channel {} roundtrip failed: {} != {}",
                i,
                unpacked[i],
                original[i]
            );
        }
    }

    #[test]
    fn test_clamping() {
        // Values outside [0.0, 1.0] should be clamped
        let clamped = pack_color_rgba(-0.5, 1.5, 0.5, 2.0);
        let unpacked = unpack_color(clamped);

        assert_eq!(unpacked[0], 0.0); // -0.5 clamped to 0.0
        assert_eq!(unpacked[1], 1.0); // 1.5 clamped to 1.0
        assert!((unpacked[2] - 0.5).abs() < 0.01); // 0.5 unchanged
        assert_eq!(unpacked[3], 1.0); // 2.0 clamped to 1.0
    }

    #[test]
    fn test_memory_savings() {
        // Demonstrate memory savings
        let _old_format: [f32; 3] = [1.0, 0.5, 0.25]; // 12 bytes
        let _new_format: u32 = pack_color_rgb(1.0, 0.5, 0.25); // 4 bytes

        assert_eq!(std::mem::size_of::<[f32; 3]>(), 12);
        assert_eq!(std::mem::size_of::<u32>(), 4);
        // 67% reduction in memory usage per color!
    }
}
