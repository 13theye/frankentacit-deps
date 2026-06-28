//! # Pipeline Module
//!
//! The core execution engine for GPU post-processing in Nnpipe. A [`Pipeline`] chains
//! [`PipelineComponent`]s into a sequential processing graph, automatically managing
//! intermediate textures and bind groups between stages.
//!
//! ## Module Contents
//!
//! - [`PipelineComponent`] / [`SimpleComponent`] / [`CompositorComponent`]: Trait
//!   hierarchy that all pipeline stages implement
//! - [`ComponentType`]: Discriminant used during finalization to wire inputs correctly
//! - [`TextureConfig`]: Lightweight descriptor for intermediate texture creation
//! - [`Pipeline`]: The ordered chain of components with automatic texture plumbing
//!
//! ## Architecture
//!
//! A `Pipeline` owns a `Vec<Box<dyn PipelineComponent>>` and a parallel set of
//! intermediate textures. During finalization each component receives its input
//! view(s) and creates the bind groups it needs. During encoding each component
//! writes a render pass whose output feeds the next stage:
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐           ┌─────────────┐
//! │  Component 0 │────▶│  Component 1 │──── ··· ──▶│  Component N │
//! └─────────────┘     └─────────────┘           └─────────────┘
//!   input_view ──▶ intermediate[0] ──▶ ··· ──▶ intermediate[N-1] ──▶ output_view
//! ```
//!
//! Compositors (`ComponentType::Compositor`) are special: during finalization they
//! receive *two* input views — the scene (or a replacement) and the previous stage's
//! output — so they can blend layers together.
//!
//! ## Named Texture Support
//!
//! Pipelines can reference textures by name rather than by position, enabling
//! multi-pipeline coordination through [`Nnpipe`]'s named texture registry:
//!
//! - `input_texture_name` — single-input override (replaces the default scene view)
//! - `input_texture_names` — multi-input list for compositors (first = scene, second = effect)
//! - `output_texture_name` — redirect final output to a named texture
//!
//! When a pipeline has named textures configured, use
//! [`Pipeline::encode_into_with_named_textures`] to resolve them from a
//! `HashMap<String, TextureView>` at encode time.
//!
//! ## Usage
//!
//! Pipelines are typically constructed via [`PipelineBuilder`] rather than assembled
//! manually:
//!
//! ```rust,ignore
//! let bloom = PipelineBuilder::new()
//!     .name("Bloom")
//!     .brightness_extract(lo_config, 0.6)
//!     .downsample(lo_config)
//!     .gaussian_blur_passes(lo_config, 2, 2.0, 10.0)
//!     .bloom_composite_with_curve(hi_config, 3.0, 3.0)
//!     .build(device)?;
//!
//! // Single-encoder pattern (recommended):
//! bloom.encode_into(device, &mut encoder, &scene_view, &output_view);
//! ```
//!
//! For multi-pipeline workflows with named textures:
//!
//! ```rust,ignore
//! let combine = PipelineBuilder::new()
//!     .input_textures(&["voice_0", "voice_1"])
//!     .output_texture("combined")
//!     .simple_lighten_composite(hi_config, 1.0)
//!     .build(device)?;
//!
//! rendering.add_multi_pipeline("combine_voices", combine);
//! rendering.execute_named_pipeline("combine_voices", device, &mut encoder)?;
//! ```

mod components;
mod pipe;

pub use components::*;
pub use pipe::*;
