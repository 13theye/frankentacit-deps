//! # Fragment Components
//!
//! Components that change texture resolution, pixel format, or temporal state.
//! Unlike the `post` components that perform image-processing math, these handle
//! structural transformations needed to connect pipeline stages at different
//! resolutions or alpha conventions.
//!
//! ## Components
//!
//! - [`DownsampleComponent`] — Reduces texture resolution by half (2x2 box filter).
//!   Used before blur passes to operate at lower resolution for performance,
//!   then followed by [`ResampleComponent`] to restore the original size.
//!
//! - [`ResampleComponent`] — Bilinear resample to an arbitrary resolution and/or
//!   texture format. Use this to upscale a downsampled texture back to full
//!   resolution, or to convert between formats (e.g., `Rgba8UnormSrgb` to
//!   `Rgba16Float`).
//!
//! - [`FeedbackComponent`] — Temporal feedback trail effect that blends the current
//!   frame with up to 8 previous frames. Each frame the history buffer shifts,
//!   creating motion trails and ghosting effects. Parameters:
//!   - `persistence` — how visible the oldest frames are (0.0 = invisible, 1.0 = full)
//!   - `frame_history` — how many history frames contribute (0.0 = none, 1.0 = all 8)
//!
//! - [`PremultiplyComponent`] — Converts straight alpha `(R, G, B, A)` to
//!   premultiplied alpha `(R*A, G*A, B*A, A)`. Uses `REPLACE` blend state for
//!   correct conversion without double-blending. See the crate-level docs on
//!   alpha handling for when this is needed.
//!
//! ## Typical Usage in a Bloom Pipeline
//!
//! ```text
//! [Scene at 1920x1080, Rgba16Float]
//!       ↓
//! [BrightnessComponent] — extract bright areas
//!       ↓
//! [DownsampleComponent] — reduce to 960x540
//!       ↓
//! [BlurComponent x4] — Gaussian blur at half res (cheap!)
//!       ↓
//! [BloomCompositeComponent] — add blurred glow back to scene
//!       ↓
//! [Output at 1920x1080]
//! ```

mod downsample;
mod feedback;
mod premultiply;
mod resample;

pub use downsample::*;
pub use feedback::*;
pub use premultiply::*;
pub use resample::*;
