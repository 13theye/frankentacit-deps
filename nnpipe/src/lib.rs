//! # Nnpipe - Nannou Pipeline Extension
//!
//! A modular GPU rendering and post-processing pipeline built on top of Nannou/wgpu.
//! Nnpipe provides particle rendering, texture compositing, and post-processing effects
//! with a focus on correct alpha handling for real-time graphics.
//!
//! ## Alpha Handling Architecture
//!
//! Nnpipe uses **straight alpha for rendering** and **premultiplied alpha for compositing**:
//!
//! - **Renderers** (particles, segments) output straight alpha. This allows multiple
//!   sources (particles, segments, Nannou Draw masks) to render to the same texture
//!   using standard alpha blending.
//!
//! - **Composite shaders** accept straight alpha inputs and output premultiplied alpha.
//!   The conversion happens internally in the shader, eliminating the need for a
//!   separate premultiply pass.
//!
//! ### Why Premultiplied Alpha for Compositing?
//!
//! Premultiplied alpha is the industry standard for compositing because:
//! - Mathematically correct blending when layers overlap
//! - No color fringing at alpha edges during texture filtering
//! - Simpler blend math for common operations (Over, Add)
//! - Clean transparent pixels: `(0, 0, 0, 0)`
//!
//! ### Blend Modes
//!
//! The `SimpleCompositeComponent` provides standard blend modes:
//!
//! - **Over**: Porter-Duff over (standard alpha compositing)
//! - **Add**: Additive blending (particles brighten when overlapping)
//! - **Lighten**: Takes the brighter pixel by luminance
//! - **Screen/Multiply/Overlay**: Standard Photoshop-style blending
//!
//! ## Typical Pipeline Structure
//!
//! ```text
//! [Particle/Segment Renderers]     [Nannou Draw (masks)]
//!            ↓                          ↓
//!      (straight α)               (straight α)
//!            ↓                          ↓
//!            └─────────┬────────────────┘
//!                      ↓
//!            [Shared Texture - straight α]
//!                      ↓
//!            [SimpleCompositeComponent]
//!            (accepts straight α, outputs premultiplied α)
//!                      ↓
//!            [Post-processing: bloom, blur, etc.]
//!                      ↓
//!            [Final composite]
//!                      ↓
//!            [Output to screen]
//! ```
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! // Create named textures
//! rendering.create_named_texture(device, "particles_voice_0", hi_config);
//! rendering.create_named_texture(device, "particles_voice_1", hi_config);
//! rendering.create_named_texture(device, "particles_combined", hi_config);
//!
//! // Render particles and masks to same texture (both use straight alpha)
//! particle_renderer.encode_only(&mut encoder, particle_count, particles_view);
//! mask.draw(&rendering.draw);
//! rendering.encode_draw_commands_into(device, &mut encoder, "particles_voice_0");
//!
//! // Composite straight alpha textures directly (no premultiply pass needed)
//! // The composite shader handles the conversion internally
//! let combine = PipelineBuilder::new()
//!     .input_textures(&["particles_voice_0", "particles_voice_1"])
//!     .output_texture("particles_combined")
//!     .simple_lighten_composite(hi_config, 1.0)
//!     .build(device)?;
//! ```

mod builder;
mod components;
mod effects;
mod nnpipe;
mod pipeline;
pub mod renderers;

pub use builder::*;
pub use components::*;
pub use effects::*;
pub use nnpipe::*;
pub use pipeline::*;
pub use renderers::*;
