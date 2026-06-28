//! FPS tracking module for apps that use Nannou

use nannou::prelude::*;
use std::time::Instant;

/// # FPS Tracking Manager
/// ## Description
/// Tracks the frames-per-second (FPS) of a Nannou app.
/// ## Usage
/// ### Initialization
/// ```rust
/// use nannou::prelude::*;
/// use fps::FpsManager;
/// ```
/// Initialize the FPSManager. Default values are `counting = false` and `should_draw = false`:
/// ```rust
/// let mut fps_manager = FpsManager::default();
/// ```
/// Initialze the FPSManager with values for `counting` and `should_draw`:
/// ```rust
/// let mut fps_manager = FpsManager::new_with(true, true);
/// ```
/// ### Control & Config:
/// Toggle the FPS counting state:
/// _(returns the current FPS counting state)_
/// Set the position where the FPS indicator will be drawn onscreen:
/// ``` rust
/// fps_manager.set_draw_position(vec2(10.0, 10.0));
/// ```
/// ``` rust
/// fps_manager.toggle();
/// ```
/// ### Update
/// Update the FPS and draw:
/// ``` rust
/// fps_manager.update();
/// fps_manager.draw(&draw);
/// ```
/// Convenience function to update and immediately draw if `should_draw` is true:
/// ``` rust
/// fps_manager.update_and_draw(&draw);
/// ```
/// Get the current FPS value:
/// ``` rust
/// let fps: f32 = fps_manager.fps();
/// ```
pub struct FpsManager {
    // Frames per second measurement
    fps: f32,
    // Minimum update interval, in seconds, used internally to control frequency of updates.
    // This helps smooth out the measurement so we get a more easy-to-read average.
    fps_update_interval: f32,
    // The last time the FPSManager.update() was called
    last_update: Instant,
    // The last time at FpsManager.fps was updated
    last_fps_update: Instant,
    // The total number of frames drawn since FpsManager began tracking.
    frame_count: usize,
    // The running total of frame durations (in seconds) collected within the fps_update_interval.
    // Smooths out the measurement so we get a more easy-to-read average.
    frame_time_accumulator: f32,

    // When `true`, the FPS is being counted
    counting: bool,

    // When `true`, the FPS is being drawn to screen in the `update_and_draw` function
    should_draw: bool,

    // The onscreen coordinate where the FPS will be drawn
    draw_position: Option<Vec2>,
}

impl Default for FpsManager {
    fn default() -> Self {
        Self::new_with(false, false)
    }
}

impl FpsManager {
    /// Create a new FpsManager with counting and drawing enabled or not
    pub fn new_with(counting: bool, should_draw: bool) -> Self {
        Self {
            fps: 0.0,
            fps_update_interval: 0.3,
            last_update: Instant::now(),
            last_fps_update: Instant::now(),
            frame_count: 0,
            frame_time_accumulator: 0.0,
            counting,
            should_draw,

            draw_position: None,
        }
    }

    /// Toggles the FPS counting state.
    /// - After toggling, returns the FPS counting state as a `bool`
    pub fn toggle(&mut self) -> bool {
        self.counting = !self.counting;
        if self.counting {
            self.initialize();
        }
        self.counting
    }

    /// Start counting frames
    fn initialize(&mut self) {
        self.fps = 0.0;
        self.frame_count = 0;
        self.frame_time_accumulator = 0.0;
        let now = Instant::now();
        self.last_update = now;
        self.last_fps_update = now;
    }

    /// Update the FPS. Call this every time the View function runs.
    pub fn update(&mut self) {
        if !self.counting {
            return;
        }

        self.calculate_fps();
    }

    /// Convenience function to update the FPS and draw if should_draw is true.
    pub fn update_and_draw(&mut self, draw: &Draw) {
        // Don't update if not counting
        if !self.counting {
            return;
        }

        self.calculate_fps();
        if self.should_draw {
            self.draw(draw);
        }
    }

    /// FPS math
    fn calculate_fps(&mut self) {
        let now = Instant::now();
        let dt_update = now - self.last_update;
        let elapsed_since_last_update = dt_update.as_secs_f32();

        let dt_fps = now - self.last_fps_update;
        let elapsed_since_last_fps = dt_fps.as_secs_f32();

        self.frame_count += 1;
        self.frame_time_accumulator += elapsed_since_last_update;
        self.last_update = now;

        if elapsed_since_last_fps >= self.fps_update_interval {
            if self.frame_count > 0 {
                let avg_frame_time = self.frame_time_accumulator / self.frame_count as f32;
                self.fps = if avg_frame_time > 0.0 {
                    1.0 / avg_frame_time
                } else {
                    0.0
                };
            }

            // Reset accumulators
            self.frame_count = 0;
            self.frame_time_accumulator = 0.0;
            self.last_fps_update = now;
        }
    }

    /// Returns the current FPS as a `f32`
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /***************** Drawing functionality ***************************/

    /// Sets the position as a `vec2` where FPS indicator will be drawn onscreen.
    pub fn set_draw_position(&mut self, position: Vec2) {
        self.draw_position = Some(position);
    }

    /// Draws to screen the FPS indicator at the stored `draw_position`.
    /// - Appears onscreen as: `FPS: {}`
    pub fn draw(&self, draw: &Draw) {
        // Don't draw if no position is set
        let Some(pos) = self.draw_position else {
            return;
        };

        draw.text(&format!("FPS: {:.1}", self.fps))
            .x_y(pos.x, pos.y)
            .color(RED)
            .font_size(10);
    }
}
