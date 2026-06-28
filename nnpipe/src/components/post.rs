//! # Post-Processing Components
//!
//! Full-screen fragment-shader effects that analyze, filter, or combine pixel data.
//! Each component implements [`PipelineComponent`] and renders a fullscreen triangle
//! that samples from its input texture(s) and writes to an output texture.
//!
//! ## Simple Components (single input)
//!
//! - [`BlurComponent`] — Separable Gaussian blur with configurable direction,
//!   adaptive scaling, and max radius. Use horizontal + vertical passes for
//!   proper 2D blur.
//! - [`BrightnessComponent`] — Extracts pixels whose luminance exceeds a threshold,
//!   outputting black for everything below. Used as the first stage of bloom.
//! - [`DarknessComponent`] — The inverse of brightness extraction; keeps dark areas
//!   and blacks out bright ones.
//! - [`ColorKeyComponent`] — Extracts pixels within a color-distance threshold of a
//!   target RGB color. Useful for isolating specific hues.
//! - [`InversionComponent`] — Inverts luminance while preserving hue/saturation,
//!   with a configurable `darken_darks` parameter for artistic control.
//!
//! ## Compositor Components (two inputs)
//!
//! - [`SimpleCompositeComponent`] — General-purpose texture compositor with selectable
//!   blend modes (Over, Add, Screen, Multiply, Overlay, Lighten). Accepts straight
//!   alpha inputs, outputs premultiplied alpha.
//! - [`BloomCompositeComponent`] — Specialized additive compositor for bloom effects
//!   with intensity and curve parameters for tone-mapped glow.
//!
//! ## Pipeline Builder Integration
//!
//! These components are typically created through [`PipelineBuilder`] rather than
//! constructed directly:
//!
//! ```rust,ignore
//! PipelineBuilder::new()
//!     .name("Glow")
//!     .brightness_extract(lo_config, 0.7)   // BrightnessComponent
//!     .downsample(lo_config)                 // DownsampleComponent
//!     .gaussian_blur_passes(lo_config, 2, 2.0, 5.0) // 4x BlurComponent
//!     .bloom_composite_with_curve(hi_config, 2.0, 3.0) // BloomCompositeComponent
//!     .build(device)?;
//! ```

mod blur;
mod brightness;
mod color_key;
mod composite;
mod darkness;
mod inversion;

pub use blur::*;
pub use brightness::*;
pub use color_key::*;
pub use composite::*;
pub use darkness::*;
pub use inversion::*;
