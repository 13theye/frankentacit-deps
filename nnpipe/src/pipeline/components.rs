//! # Pipeline Component Traits
//!
//! Defines the trait hierarchy that every pipeline stage must implement, plus the
//! supporting types used during pipeline construction and finalization.
//!
//! ## Trait Hierarchy
//!
//! ```text
//! PipelineComponent            (base trait — encode, finalize, name, type, update)
//!     ├── SimpleComponent      (marker — single input → single output)
//!     └── CompositorComponent  (marker — two inputs → single output)
//! ```
//!
//! All concrete components (blur, brightness, composite, etc.) implement one of the
//! marker traits, which is also a super-trait of `PipelineComponent`. The marker
//! traits carry no additional methods — they exist so that [`PipelineBuilder`] and
//! [`Pipeline`] can accept stages with the correct semantics through distinct
//! `add_simple_stage` / `add_compositor` entry points.
//!
//! ## Component Lifecycle
//!
//! 1. **Construction** — the component creates its shader module, render pipeline,
//!    and any parameter uniform buffers. Bind groups are *not* created yet because
//!    input textures are unknown.
//!
//! 2. **Finalization** ([`PipelineComponent::finalize_bind_groups`]) — called once
//!    by [`Pipeline::finalize`] after all intermediate textures exist. The component
//!    receives its input view (and, for compositors, a second "effect" view) and
//!    creates the bind group(s) it needs.
//!
//! 3. **Encoding** ([`PipelineComponent::encode_pass`]) — called every frame.
//!    The component writes a render pass into the provided command encoder,
//!    reading from its bound input(s) and writing to the supplied `output_view`.
//!
//! 4. **Parameter update** ([`PipelineComponent::update_parameters`]) — called when
//!    runtime parameters (e.g. blur radius, blend intensity) need to be written to
//!    their uniform buffers via `queue.write_buffer`.
//!
//! ## Implementing a Custom Component
//!
//! ```rust,ignore
//! use nnpipe::{PipelineComponent, SimpleComponent, ComponentType, TextureConfig};
//!
//! pub struct MyEffect { /* shader, pipeline, bind_group, uniforms … */ }
//!
//! impl PipelineComponent for MyEffect {
//!     fn name(&self) -> &str { "My Effect" }
//!     fn component_type(&self) -> ComponentType { ComponentType::Simple }
//!
//!     fn finalize_bind_groups(
//!         &mut self, device: &wgpu::Device,
//!         input_view: &wgpu::TextureView,
//!         _effect_view: Option<&wgpu::TextureView>,
//!     ) {
//!         // Create bind group referencing input_view + sampler + uniforms
//!     }
//!
//!     fn encode_pass(
//!         &mut self, encoder: &mut wgpu::CommandEncoder,
//!         output_view: &wgpu::TextureView,
//!     ) {
//!         // Begin render pass writing to output_view, draw fullscreen triangle
//!     }
//!
//!     fn update_parameters(&mut self, queue: &wgpu::Queue) {
//!         // Write updated uniforms to GPU buffer
//!     }
//! }
//!
//! impl SimpleComponent for MyEffect {}
//! ```

use nannou::wgpu;

/// Discriminant used by [`Pipeline`] during finalization to determine how many
/// input views a component requires.
///
/// - `Simple` — receives one input (the previous stage's output, or the pipeline's
///   input view for the first stage).
/// - `Compositor` — receives two inputs: the *scene* view (or a scene-replacement
///   view) and the previous stage's output. Named textures can override both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComponentType {
    Simple,
    Compositor,
}

/// The base trait for all pipeline stages.
///
/// Every component in a [`Pipeline`] must implement this trait. The pipeline calls
/// its methods in order: `finalize_bind_groups` once at setup, then `encode_pass`
/// every frame, with optional `update_parameters` calls in between.
pub trait PipelineComponent {
    /// Encode the render pass into the provided command encoder.
    ///
    /// `output_view` is the intermediate texture (or final output) that this
    /// stage should write to. The component's *input* was bound during
    /// [`finalize_bind_groups`](PipelineComponent::finalize_bind_groups).
    fn encode_pass(&mut self, encoder: &mut wgpu::CommandEncoder, output_view: &wgpu::TextureView);

    /// Create bind groups referencing the resolved input texture(s).
    ///
    /// Called once by [`Pipeline::finalize`] after all intermediate textures are
    /// allocated. `input_view` is the primary input; `effect_view` is `Some` only
    /// for `ComponentType::Compositor` stages and provides the second texture to blend.
    fn finalize_bind_groups(
        &mut self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        effect_view: Option<&wgpu::TextureView>,
    );

    /// A human-readable name for this component (used in debug logs).
    fn name(&self) -> &str;

    /// Return the component type so the pipeline can wire inputs correctly.
    fn component_type(&self) -> ComponentType;

    /// Write updated parameter values to GPU uniform buffers.
    ///
    /// Called by [`Pipeline::update_parameters`] when runtime values change.
    /// Implementations should use `queue.write_buffer(...)` to push new uniforms.
    fn update_parameters(&mut self, queue: &wgpu::Queue);
}

/// Marker trait for components that process a single input texture and write to
/// an output texture (e.g., blur, brightness extract, color inversion).
pub trait CompositorComponent: PipelineComponent {}

/// Marker trait for components that composite two input textures into one output
/// texture (e.g., bloom composite, blend-mode compositors).
pub trait SimpleComponent: PipelineComponent {}

/// Lightweight descriptor for creating intermediate textures within a [`Pipeline`].
///
/// Passed to [`PipelineBuilder`] stage methods and forwarded to
/// [`Pipeline::setup_intermediate_texture`] during build. The format determines
/// available texture usages — HDR formats like `Rgba16Float` additionally receive
/// `STORAGE_BINDING`.
#[derive(Debug, Clone, Copy)]
pub struct TextureConfig {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}
