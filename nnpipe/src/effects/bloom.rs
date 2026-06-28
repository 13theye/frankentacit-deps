// src/effects/bloom.rs

// Bloom effect example implementation
// integration is currently incomplete: build the effect manually using EffectBuilder instead.

use crate::{
    builder::PipelineBuilder,
    pipeline::{Pipeline, PipelineComponent, SimpleStage, TextureConfig},
};
use nannou::wgpu;

/// Bloom Effect type. Currently incomplete:
/// - build the effect manually using EffectBuilder instead
pub struct BloomEffect {
    pipeline: Pipeline,

    // Bloom-specific parameters for easy access
    pub brightness_threshold: f32,
    pub blur_passes: u32,
    pub intensity: f32,
    pub adaptive_scaling: f32,
    pub max_radius: f32,
}

impl BloomEffect {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        brightness_threshold: f32,
        blur_passes: u32,
        intensity: f32,
        intensity_curve: f32,
    ) -> Result<Self, crate::builder::PipelineBuilderError> {
        let adaptive_scaling = 2.0;
        let max_radius = 10.0;

        let lo_config = TextureConfig {
            width: width / 2,
            height: height / 2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        };

        let hi_config = TextureConfig {
            width,
            height,
            format: wgpu::TextureFormat::Rgba16Float,
        };

        // Build the core bloom effect using the EffectBuilder
        let effect = PipelineBuilder::new()
            .name("Bloom")
            .brightness_extract(lo_config, brightness_threshold)
            .downsample(lo_config)
            .gaussian_blur_passes(lo_config, blur_passes, adaptive_scaling, max_radius)
            .bloom_composite_with_curve(hi_config, intensity, intensity_curve) // Bloom composite is always additive
            .build(device)?;

        Ok(Self {
            pipeline: effect,
            brightness_threshold,
            blur_passes,
            intensity,
            adaptive_scaling,
            max_radius,
        })
    }

    pub fn default(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<Self, crate::builder::PipelineBuilderError> {
        Self::new(device, width, height, 0.6, 2, 3.0, 3.0)
    }

    // Parameter update methods
    pub fn set_brightness_threshold(&mut self, _queue: &wgpu::Queue, threshold: f32) {
        self.brightness_threshold = threshold;
        // TODO: Update internal brightness component parameter
    }

    pub fn set_intensity(&mut self, _queue: &wgpu::Queue, intensity: f32) {
        self.intensity = intensity;
        // TODO: Update internal composite component parameter
    }

    pub fn set_adaptive_scaling(&mut self, _queue: &wgpu::Queue, scaling: f32) {
        self.adaptive_scaling = scaling;
        // TODO: Update internal blur component parameters
    }

    pub fn set_max_radius(&mut self, _queue: &wgpu::Queue, radius: f32) {
        self.max_radius = radius;
        // TODO: Update internal blur component parameters
    }

    // Get current parameter values
    pub fn get_brightness_threshold(&self) -> f32 {
        self.brightness_threshold
    }

    pub fn get_intensity(&self) -> f32 {
        self.intensity
    }

    pub fn get_adaptive_scaling(&self) -> f32 {
        self.adaptive_scaling
    }

    pub fn get_max_radius(&self) -> f32 {
        self.max_radius
    }
}

impl SimpleStage for BloomEffect {
    fn finalize_bind_groups(&mut self, device: &wgpu::Device, input_view: &wgpu::TextureView) {
        self.pipeline.finalize(device, input_view);
    }

    fn encode_pass(&mut self, encoder: &mut wgpu::CommandEncoder, output_view: &wgpu::TextureView) {
        // Forward to the pipeline - but we need to use a temporary encoder approach
        // since Pipeline::process expects its own encoder
        // For now, create a temporary encoder and merge operations
        // This is a limitation of the current architecture that could be improved
        let mut temp_encoder = encoder.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Bloom Effect Temp Encoder"),
        });
        
        // We need to refactor Pipeline to accept an encoder rather than create its own
        // For now, this is a placeholder that won't work correctly
        // TODO: Fix this architectural issue
        unimplemented!("BloomEffect needs Pipeline refactoring to accept encoder")
    }
}

impl PipelineComponent for BloomEffect {
    fn name(&self) -> &str {
        "Bloom Effect"
    }

    fn update_parameters(&mut self, _queue: &wgpu::Queue) {
        // Can't update parameters at the Pipeline level
    }
}

// Refactored BloomPipeline using the new component system
// Provides backward compatibility while using modular architecture internally

pub struct BloomPipeline {
    // Use the new BloomEffect internally
    bloom_effect: BloomEffect,

    // Keep public fields for backward compatibility
    pub brightness_threshold: f32,
    pub adaptive_blur_scaling: f32,
    pub max_blur_radius: f32,
}

impl BloomPipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // Default bloom parameters
        let brightness_threshold = 0.6;
        let adaptive_blur_scaling = 2.0;
        let max_blur_radius = 10.0;
        let blur_passes = 2;
        let intensity = 3.0;
        let intensity_curve = 3.0;

        // Create the internal BloomEffect using the new component system
        let bloom_effect = BloomEffect::new(
            device,
            width,
            height,
            brightness_threshold,
            blur_passes,
            intensity,
            intensity_curve,
        )
        .expect("Failed to create BloomEffect");

        Self {
            bloom_effect,
            brightness_threshold,
            adaptive_blur_scaling,
            max_blur_radius,
        }
    }

    pub fn set_brightness_threshold(&mut self, queue: &wgpu::Queue, threshold: f32) {
        self.brightness_threshold = threshold;
        self.bloom_effect.set_brightness_threshold(queue, threshold);
    }

    pub fn set_adaptive_blur_scaling(&mut self, queue: &wgpu::Queue, scaling: f32) {
        self.adaptive_blur_scaling = scaling;
        self.bloom_effect.set_adaptive_scaling(queue, scaling);
    }

    pub fn set_max_blur_radius(&mut self, queue: &wgpu::Queue, radius: f32) {
        self.max_blur_radius = radius;
        self.bloom_effect.set_max_radius(queue, radius);
    }

    pub fn set_intensity(&mut self, queue: &wgpu::Queue, intensity: f32) {
        self.bloom_effect.set_intensity(queue, intensity);
    }
}

impl SimpleStage for BloomPipeline {
    fn process(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) {
        // Delegate to the internal BloomEffect
        self.bloom_effect
            .process(device, queue, input_view, output_view);
    }
}

impl PipelineComponent for BloomPipeline {
    fn name(&self) -> &str {
        "Bloom Pipeline"
    }

    fn update_parameters(&mut self, queue: &wgpu::Queue) {
        self.bloom_effect.update_parameters(queue);
    }
}
