//! # Pipeline
//!
//! The main execution container for a sequence of [`PipelineComponent`] stages.
//! Manages intermediate textures, bind-group finalization, and per-frame encoding.
//!
//! ## Ownership Model
//!
//! A `Pipeline` owns its components and intermediate textures. It does **not** own
//! the input or output textures — those are supplied by the caller (typically
//! [`Nnpipe`]) at encode time.
//!
//! ## Encoding Patterns
//!
//! Three encoding methods are provided, ordered from most to least flexible:
//!
//! | Method | When to use |
//! |--------|-------------|
//! | [`encode_into_with_named_textures`] | Multi-pipeline workflows with named texture registry |
//! | [`encode_into`] | Single-encoder pattern with explicit input/output views |
//! | [`process`] | Standalone mode — creates and submits its own encoder |
//!
//! The `encode_into` family writes render passes into an existing
//! `CommandEncoder` without submitting, so multiple pipelines and renderers
//! can share a single submission per frame.
//!
//! ## Scene Replacement
//!
//! By calling [`set_scene_replacement_stage`], a pipeline can redirect the
//! "scene view" used by compositor stages later in the chain. This is how the
//! bloom effect works: the brightness-extracted, blurred result replaces the
//! original scene for the final composite.
//!
//! ## Construction
//!
//! Prefer [`PipelineBuilder`] over manual assembly:
//!
//! ```rust,ignore
//! let pipeline = PipelineBuilder::new()
//!     .name("My Effect")
//!     .brightness_extract(lo_config, 0.6)
//!     .gaussian_blur_passes(lo_config, 2, 2.0, 10.0)
//!     .bloom_composite_with_curve(hi_config, 3.0, 3.0)
//!     .build(device)?;
//! ```
//!
//! The builder handles intermediate texture allocation and component ordering.

use super::components::{
    ComponentType, CompositorComponent, PipelineComponent, SimpleComponent, TextureConfig,
};
use nannou::wgpu;

/// An ordered chain of [`PipelineComponent`] stages that are processed sequentially,
/// with automatic intermediate texture management.
///
/// Each stage reads from the previous stage's output (or the pipeline input for the
/// first stage) and writes to the next intermediate texture (or the pipeline output
/// for the last stage). Compositor stages additionally receive a "scene" view as a
/// second input.
///
/// # Example
///
/// ```rust,ignore
/// // Build via PipelineBuilder (recommended)
/// let mut pipeline = PipelineBuilder::new()
///     .name("Glow")
///     .brightness_extract(lo_config, 0.7)
///     .gaussian_blur_passes(lo_config, 2, 2.0, 5.0)
///     .bloom_composite_with_curve(hi_config, 2.0, 3.0)
///     .build(device)?;
///
/// // Encode into a shared command encoder each frame
/// pipeline.encode_into(device, &mut encoder, &scene_view, &output_view);
/// ```
pub struct Pipeline {
    name: String,
    components: Vec<Box<dyn PipelineComponent>>,
    intermediate_textures: Vec<nannou::wgpu::Texture>,
    intermediate_views: Vec<nannou::wgpu::TextureView>,
    is_finalized: bool,
    scene_replacement_stage: Option<usize>,
    // Named texture support for multi-pipeline coordination
    input_texture_name: Option<String>,
    input_texture_names: Vec<String>, // For multi-input support
    output_texture_name: Option<String>,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new_with_name("Default Pipeline")
    }
}

impl Pipeline {
    pub fn new_with_name(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            components: Vec::new(),
            intermediate_textures: Vec::new(),
            intermediate_views: Vec::new(),
            is_finalized: false,
            scene_replacement_stage: None,
            input_texture_name: None,
            input_texture_names: Vec::new(),
            output_texture_name: None,
        }
    }

    /// Add a single-input component (blur, brightness, inversion, etc.) to the pipeline.
    ///
    /// Functionally identical to [`add_component`](Pipeline::add_component), but
    /// accepts `Box<dyn SimpleComponent>` for type-safe construction.
    pub fn add_simple_stage(&mut self, stage: Box<dyn SimpleComponent>) {
        println!("Pipeline: adding {}", stage.name());
        self.components.push(stage);
    }

    /// Add a dual-input compositor (blend, bloom composite, etc.) to the pipeline.
    ///
    /// Functionally identical to [`add_component`](Pipeline::add_component), but
    /// accepts `Box<dyn CompositorComponent>` for type-safe construction.
    /// During finalization, compositors receive both a scene view and the previous
    /// stage's output as inputs.
    pub fn add_compositor(&mut self, compositor: Box<dyn CompositorComponent>) {
        println!("Pipeline: adding {}", compositor.name());
        self.components.push(compositor);
    }

    /// Add any [`PipelineComponent`] to the pipeline regardless of its subtype.
    ///
    /// This is the most general insertion method. Prefer [`add_simple_stage`](Pipeline::add_simple_stage)
    /// or [`add_compositor`](Pipeline::add_compositor) when the component type is known.
    pub fn add_component(&mut self, component: Box<dyn PipelineComponent>) {
        println!("Pipeline: adding {}", component.name());
        self.components.push(component);
    }

    /// Returns true if the Pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Returns the name of the Pipeline
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set which stage's output should replace the scene view for subsequent compositor stages.
    ///
    /// After `stage_index`, any compositor in the chain will receive this stage's
    /// intermediate output as its "scene" input instead of the original scene view.
    /// Pass `None` to disable scene replacement.
    ///
    /// This is used by the bloom effect: the brightness-extracted + blurred result
    /// becomes the "scene" for the final bloom composite.
    pub fn set_scene_replacement_stage(&mut self, stage_index: Option<usize>) {
        self.scene_replacement_stage = stage_index;
    }

    /// Set the named texture to use as this pipeline's primary input.
    ///
    /// When set, [`encode_into_with_named_textures`](Pipeline::encode_into_with_named_textures)
    /// resolves this name from the provided `HashMap<String, TextureView>` instead
    /// of using the default scene view. Pass `None` to use the default.
    pub fn set_input_texture_name(&mut self, name: Option<String>) {
        self.input_texture_name = name;
    }

    /// Set multiple named textures for multi-input compositor support.
    ///
    /// For compositor components, the first name becomes the "scene" input and
    /// the second becomes the "effect" input. This enables compositing textures
    /// from different rendering passes (e.g., combining two particle layers).
    pub fn set_input_texture_names(&mut self, names: Vec<String>) {
        self.input_texture_names = names;
    }

    /// Set the named texture to use as this pipeline's final output.
    ///
    /// When set, the last stage writes to the named texture instead of the
    /// default output view. This allows pipelines to feed results into other
    /// pipelines via the named texture registry.
    pub fn set_output_texture_name(&mut self, name: Option<String>) {
        self.output_texture_name = name;
    }

    /// Get the input texture name
    pub fn input_texture_name(&self) -> &Option<String> {
        &self.input_texture_name
    }

    /// Get the input texture names for multi-input support
    pub fn input_texture_names(&self) -> &Vec<String> {
        &self.input_texture_names
    }

    /// Get the output texture name
    pub fn output_texture_name(&self) -> &Option<String> {
        &self.output_texture_name
    }

    pub fn _setup_intermediate_textures(&mut self, device: &wgpu::Device, config: TextureConfig) {
        self.intermediate_textures.clear();
        self.intermediate_views.clear();

        for _i in 0..self.components.len().saturating_sub(1) {
            self.setup_intermediate_texture(device, config);
        }
    }

    /// Allocate a new intermediate texture and view for passing data between stages.
    ///
    /// Called by [`PipelineBuilder::build`] once per stage. The texture receives
    /// `RENDER_ATTACHMENT | TEXTURE_BINDING` usages, plus `STORAGE_BINDING` for
    /// formats that support it (e.g., `Rgba16Float`).
    pub fn setup_intermediate_texture(&mut self, device: &wgpu::Device, config: TextureConfig) {
        let mut usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

        // Only add STORAGE_BINDING for formats that support it
        if supports_storage_binding(config.format) {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }

        let texture = nannou::wgpu::TextureBuilder::new()
            .size([config.width, config.height])
            .format(config.format)
            .usage(usage)
            .build(device);

        let view = texture.view().build();

        self.intermediate_textures.push(texture);
        self.intermediate_views.push(view);
    }

    /// Finalize the pipeline by creating bind groups for all components.
    ///
    /// Must be called after all intermediate textures are allocated. This is
    /// called automatically by `encode_into` / `process` if not already finalized.
    /// Each component receives its resolved input view(s) and creates the GPU
    /// bind groups it needs for rendering.
    ///
    /// Shorthand for [`finalize_with_named_textures`](Pipeline::finalize_with_named_textures)
    /// with `named_views = None`.
    pub fn finalize(&mut self, device: &wgpu::Device, scene_view: &wgpu::TextureView) {
        self.finalize_with_named_textures(device, scene_view, None);
    }

    /// Finalize the pipeline, resolving named textures from the provided registry.
    ///
    /// For each component in order:
    /// - **Simple** stages receive the previous stage's output (or `scene_view` for the first).
    /// - **Compositor** stages receive two views: if `input_texture_names` are set and
    ///   present in `named_views`, those are used; otherwise the scene view and
    ///   previous output are used as defaults.
    ///
    /// Finalization is idempotent — calling it again after the pipeline is already
    /// finalized is a no-op.
    pub fn finalize_with_named_textures(
        &mut self,
        device: &wgpu::Device,
        scene_view: &wgpu::TextureView,
        named_views: Option<&std::collections::HashMap<String, wgpu::TextureView>>,
    ) {
        if self.is_finalized {
            return;
        }

        for (i, component) in self.components.iter_mut().enumerate() {
            // Establish the current input according to position in the chain
            let current_input = if i == 0 {
                scene_view
            } else {
                &self.intermediate_views[i - 1]
            };

            // Determine the effective scene view - use replacement if specified and we're past that stage
            let effective_scene_view = if let Some(replacement_stage) = self.scene_replacement_stage
            {
                if i > replacement_stage {
                    &self.intermediate_views[replacement_stage]
                } else {
                    scene_view
                }
            } else {
                scene_view
            };

            // Determine input and effect views based on component type and named textures
            let (input_view, effect_view) = match component.component_type() {
                ComponentType::Simple => (current_input, None),
                ComponentType::Compositor => {
                    // For compositors, check if we have named texture inputs
                    if let (Some(first), Some(views)) =
                        (self.input_texture_names.first(), named_views)
                    {
                        let input_view = views.get(first).unwrap_or(effective_scene_view);

                        let effect_view = self
                            .input_texture_names
                            .get(1) // safely get the second element if it exists
                            .and_then(|name| views.get(name))
                            .or(Some(current_input));

                        (input_view, effect_view)
                    } else {
                        // Default behavior: use scene view as input and previous stage as effect
                        (effective_scene_view, Some(current_input))
                    }
                }
            };

            component.finalize_bind_groups(device, input_view, effect_view);
        }

        self.is_finalized = true;
    }

    /// Encode all pipeline stages into an existing command encoder.
    ///
    /// This is the recommended per-frame entry point. The caller owns the encoder
    /// and can add other render passes (particle rendering, Nannou Draw, etc.)
    /// before or after this call, then submit once.
    ///
    /// Auto-finalizes on first call if the pipeline hasn't been finalized yet.
    pub fn encode_into(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) {
        if self.components.is_empty() {
            return;
        }

        // Finalize pipeline if not already done
        if !self.is_finalized {
            self.finalize(device, input_view);
        }

        let stage_count = self.components.len();

        // Encode all stages in order - no enum match needed in hot loop!
        for (i, component) in self.components.iter_mut().enumerate() {
            let current_output = if i < stage_count - 1 {
                // Not the last stage - use intermediate texture
                &self.intermediate_views[i]
            } else {
                // Last stage - use final output
                output_view
            };

            component.encode_pass(encoder, current_output);
        }
    }

    /// Encode all stages, resolving input/output from the named texture registry.
    ///
    /// Named texture names configured on this pipeline (via [`set_input_texture_name`],
    /// [`set_input_texture_names`], [`set_output_texture_name`]) are looked up in
    /// `named_views`. If a name is not found, the corresponding fallback view
    /// (`scene_view` or `output_view`) is used.
    ///
    /// Returns `Ok(())` on success. Currently infallible but returns `Result`
    /// for forward-compatibility with validation.
    pub fn encode_into_with_named_textures(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        named_views: &std::collections::HashMap<String, wgpu::TextureView>,
        scene_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) -> Result<(), String> {
        if self.components.is_empty() {
            return Ok(());
        }

        // Resolve input view from pipeline's named input texture, fallback to scene_view
        let resolved_input = if let Some(input_name) = &self.input_texture_name {
            named_views.get(input_name).unwrap_or(scene_view)
        } else {
            scene_view
        };

        // Resolve output view from pipeline's named output texture, fallback to output_view
        let resolved_output = if let Some(output_name) = &self.output_texture_name {
            named_views.get(output_name).unwrap_or(output_view)
        } else {
            output_view
        };

        // Finalize pipeline if not already done
        if !self.is_finalized {
            self.finalize_with_named_textures(device, resolved_input, Some(named_views));
        }

        let stage_count = self.components.len();

        // Encode all stages in order - no enum match needed in hot loop!
        for (i, component) in self.components.iter_mut().enumerate() {
            let current_output = if i < stage_count - 1 {
                // Not the last stage - use intermediate texture
                &self.intermediate_views[i]
            } else {
                // Last stage - use final output
                resolved_output
            };

            component.encode_pass(encoder, current_output);
        }

        Ok(())
    }

    /// Process the entire pipeline in standalone mode — creates, encodes, and submits
    /// its own command encoder.
    ///
    /// Convenient for simple setups but less efficient than [`encode_into`](Pipeline::encode_into)
    /// when multiple renderers and pipelines share a frame, because each `process` call
    /// results in a separate GPU submission. Prefer `encode_into` with a shared encoder
    /// for production use.
    pub fn process(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) {
        if self.components.is_empty() {
            return;
        }

        // Finalize pipeline if not already done
        if !self.is_finalized {
            self.finalize(device, input_view);
        }

        // Create a single command encoder for the entire pipeline
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{} Pipeline Encoder", self.name)),
        });

        // Use the new encode_into method
        self.encode_into(device, &mut encoder, input_view, output_view);

        // Submit the entire command encoder
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    /// Push updated runtime parameters to GPU buffers for all components.
    ///
    /// Iterates each component and calls [`PipelineComponent::update_parameters`],
    /// which writes new uniform values via `queue.write_buffer`. Call this when
    /// effect parameters (blur radius, blend intensity, etc.) change between frames.
    pub fn update_parameters(&mut self, queue: &wgpu::Queue) {
        for component in &mut self.components {
            component.update_parameters(queue);
        }
    }
}

/// Returns `true` if the given texture format supports `STORAGE_BINDING` usage.
///
/// Only float/int formats with power-of-two channel counts are eligible.
/// Common HDR formats like `Rgba16Float` return `true`; sRGB formats return `false`.
fn supports_storage_binding(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rgba16Float
            | wgpu::TextureFormat::Rgba32Float
            | wgpu::TextureFormat::R32Float
            | wgpu::TextureFormat::Rg32Float
            | wgpu::TextureFormat::R16Float
            | wgpu::TextureFormat::Rg16Float
            | wgpu::TextureFormat::R32Uint
            | wgpu::TextureFormat::Rg32Uint
            | wgpu::TextureFormat::Rgba32Uint
            | wgpu::TextureFormat::R32Sint
            | wgpu::TextureFormat::Rg32Sint
            | wgpu::TextureFormat::Rgba32Sint
    )
}
