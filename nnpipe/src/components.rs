//! # Components
//!
//! Concrete [`PipelineComponent`] implementations — the building blocks of every
//! post-processing pipeline in Nnpipe.
//!
//! Components are organized into two submodules by function:
//!
//! ## `post` — Image Post-Processing
//!
//! Full-screen fragment-shader effects that analyze or transform pixel data:
//!
//! | Component | Type | Description |
//! |-----------|------|-------------|
//! | [`BlurComponent`] | Simple | Separable Gaussian blur (horizontal or vertical) |
//! | [`BrightnessComponent`] | Simple | Extract pixels above a luminance threshold |
//! | [`DarknessComponent`] | Simple | Extract pixels below a luminance threshold |
//! | [`ColorKeyComponent`] | Simple | Extract pixels matching a target color |
//! | [`InversionComponent`] | Simple | Monochromatic luminance inversion |
//! | [`SimpleCompositeComponent`] | Compositor | Blend two textures with selectable blend mode |
//! | [`BloomCompositeComponent`] | Compositor | Additive bloom compositing with tone mapping |
//!
//! ## `fragment` — Texture Operations
//!
//! Components that change texture resolution, format, or temporal state:
//!
//! | Component | Type | Description |
//! |-----------|------|-------------|
//! | [`DownsampleComponent`] | Simple | 2x2 box-filter downsample to half resolution |
//! | [`ResampleComponent`] | Simple | Bilinear resample to arbitrary resolution/format |
//! | [`FeedbackComponent`] | Simple | Temporal feedback trail with 8-frame history |
//! | [`PremultiplyComponent`] | Simple | Convert straight alpha to premultiplied alpha |
//!
//! ## Common Patterns
//!
//! All components follow the same lifecycle (see [`PipelineComponent`]):
//! 1. Construct with `::new(device, config, ...)` — creates shader, pipeline, uniform buffers
//! 2. Pipeline calls `finalize_bind_groups(...)` — creates bind groups with real textures
//! 3. Pipeline calls `encode_pass(...)` each frame — writes a render pass
//! 4. Optionally call `update_parameters(queue)` when runtime values change
//!
//! Components are not used directly — they are added to a [`Pipeline`] via
//! [`PipelineBuilder`] methods like `.gaussian_blur_passes(...)`,
//! `.brightness_extract(...)`, `.simple_lighten_composite(...)`, etc.

mod fragment;
mod post;

pub use fragment::*;
pub use post::*;
