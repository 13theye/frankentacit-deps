// src/builder/validation.rs
// Basic validation rules for Pipeline Components evaluated during initialization.
// TODO: incorporate these into individual components

use crate::builder::PipelineBuilderError;
use nannou::wgpu;

pub trait PipelineValidation {
    fn validate_pipeline(&self) -> Result<(), PipelineBuilderError>;
    fn suggest_optimizations(&self) -> Vec<String>;
    fn estimate_performance_cost(&self) -> PerformanceCost;
}

#[derive(Debug, Clone, Copy)]
pub enum PerformanceCost {
    Low,
    Medium,
    High,
    VeryHigh,
}

impl PerformanceCost {
    pub fn description(&self) -> &'static str {
        match self {
            PerformanceCost::Low => "Minimal performance impact",
            PerformanceCost::Medium => "Moderate performance impact",
            PerformanceCost::High => "Significant performance impact",
            PerformanceCost::VeryHigh => "Very high performance impact - consider optimization",
        }
    }
}

// Validation rules for common effect patterns
pub struct PipelineValidator;

impl PipelineValidator {
    pub fn validate_blur_efficiency(
        blur_count: usize,
        max_radius: f32,
    ) -> Result<(), PipelineBuilderError> {
        if blur_count > 6 {
            return Err(PipelineBuilderError::InvalidConfiguration(format!(
                "Too many blur passes ({}). Consider using fewer passes with larger radius.",
                blur_count
            )));
        }

        if max_radius > 20.0 {
            return Err(PipelineBuilderError::InvalidConfiguration(format!(
                "Blur radius too large ({}). Values above 20.0 may cause performance issues.",
                max_radius
            )));
        }

        Ok(())
    }

    pub fn validate_texture_format_compatibility(
        input_format: wgpu::TextureFormat,
        output_format: wgpu::TextureFormat,
    ) -> Result<(), PipelineBuilderError> {
        // Check for potential precision loss
        match (input_format, output_format) {
            (wgpu::TextureFormat::Rgba16Float, wgpu::TextureFormat::Rgba8UnormSrgb) => {
                // This is actually common and acceptable for final output
                Ok(())
            }
            (high_precision, low_precision)
                if is_high_precision(high_precision) && !is_high_precision(low_precision) =>
            {
                Err(PipelineBuilderError::InvalidConfiguration(format!(
                    "Potential precision loss: {:?} -> {:?}",
                    high_precision, low_precision
                )))
            }
            _ => Ok(()),
        }
    }

    pub fn suggest_bloom_optimizations(threshold: f32, blur_passes: usize) -> Vec<String> {
        let mut suggestions = Vec::new();

        if threshold < 0.3 {
            suggestions.push(
                "Consider increasing brightness threshold to reduce bloom artifacts".to_string(),
            );
        }

        if blur_passes > 3 {
            suggestions.push(format!(
                "Consider reducing blur passes from {} to 2-3 for better performance",
                blur_passes
            ));
        }

        if threshold > 0.9 {
            suggestions.push(
                "Very high brightness threshold may result in minimal bloom effect".to_string(),
            );
        }

        suggestions
    }
}

fn is_high_precision(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rgba16Float
            | wgpu::TextureFormat::Rgba32Float
            | wgpu::TextureFormat::R32Float
            | wgpu::TextureFormat::Rg32Float
    )
}
