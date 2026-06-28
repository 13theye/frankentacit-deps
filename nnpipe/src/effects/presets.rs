// src/effects/presets.rs
//
// Predefined effect pipeline configurations for RenderWindow

use crate::builder::PipelineBuilder;
use crate::pipeline::{Pipeline, TextureConfig};
use nannou::wgpu;
use std::collections::HashMap;

/// Predefined effect configurations for easy use in MaskManager
pub struct EffectPresets {
    presets: HashMap<String, EffectPresetConfig>,
}

/// Configuration for a preset effect pipeline
#[derive(Debug, Clone)]
pub struct EffectPresetConfig {
    pub name: String,
    pub description: String,
    pub builder_fn: fn(TextureConfig) -> PipelineBuilder,
    pub parameters: Vec<ParameterDescriptor>,
}

/// Description of a tweakable parameter in an effect preset
#[derive(Debug, Clone)]
pub struct ParameterDescriptor {
    pub name: String,
    pub description: String,
    pub parameter_type: ParameterType,
    pub default_value: f32,
    pub min_value: f32,
    pub max_value: f32,
}

#[derive(Debug, Clone)]
pub enum ParameterType {
    Intensity,      // General intensity/strength parameter
    Threshold,      // Brightness/darkness threshold
    Radius,         // Blur radius or similar
    ColorChannel,   // Color component (0-1)
    Boolean,        // On/off parameter (0.0 = off, 1.0 = on)
    Custom(String), // Custom parameter type
}

impl EffectPresets {
    pub fn new() -> Self {
        let mut presets = HashMap::new();
        
        // Register built-in presets
        presets.insert("blur".to_string(), Self::create_blur_preset());
        presets.insert("glow".to_string(), Self::create_glow_preset());
        presets.insert("invert".to_string(), Self::create_invert_preset());
        presets.insert("feedback".to_string(), Self::create_feedback_preset());
        presets.insert("brightness_extract".to_string(), Self::create_brightness_extract_preset());
        presets.insert("color_key".to_string(), Self::create_color_key_preset());
        presets.insert("composite_add".to_string(), Self::create_additive_composite_preset());
        presets.insert("composite_screen".to_string(), Self::create_screen_composite_preset());
        
        Self { presets }
    }

    pub fn get_preset(&self, name: &str) -> Option<&EffectPresetConfig> {
        self.presets.get(name)
    }

    pub fn list_presets(&self) -> Vec<&String> {
        self.presets.keys().collect()
    }

    pub fn create_pipeline(&self, preset_name: &str, config: TextureConfig, device: &wgpu::Device) -> Result<Pipeline, String> {
        let preset = self.get_preset(preset_name)
            .ok_or_else(|| format!("Unknown preset: {}", preset_name))?;

        let builder = (preset.builder_fn)(config);
        builder.build(device)
            .map_err(|e| format!("Failed to build pipeline: {:?}", e))
    }

    pub fn create_pipeline_with_params(
        &self, 
        preset_name: &str, 
        config: TextureConfig, 
        parameters: &HashMap<String, f32>,
        device: &wgpu::Device
    ) -> Result<Pipeline, String> {
        match preset_name {
            "blur" => Self::create_blur_with_params(config, parameters, device),
            "glow" => Self::create_glow_with_params(config, parameters, device),
            "invert" => Self::create_invert_with_params(config, parameters, device),
            "feedback" => Self::create_feedback_with_params(config, parameters, device),
            "brightness_extract" => Self::create_brightness_extract_with_params(config, parameters, device),
            "color_key" => Self::create_color_key_with_params(config, parameters, device),
            "composite_add" => Self::create_additive_composite_with_params(config, parameters, device),
            "composite_screen" => Self::create_screen_composite_with_params(config, parameters, device),
            _ => Err(format!("Unknown preset or preset doesn't support parameters: {}", preset_name))
        }
    }

    // Preset definitions
    fn create_blur_preset() -> EffectPresetConfig {
        EffectPresetConfig {
            name: "blur".to_string(),
            description: "Gaussian blur effect".to_string(),
            builder_fn: |config| {
                PipelineBuilder::new()
                    .name("Blur Effect")
                    .gaussian_blur_passes(config, 2, 2.0, 5.0)
            },
            parameters: vec![
                ParameterDescriptor {
                    name: "intensity".to_string(),
                    description: "Blur intensity".to_string(),
                    parameter_type: ParameterType::Intensity,
                    default_value: 2.0,
                    min_value: 0.0,
                    max_value: 10.0,
                },
                ParameterDescriptor {
                    name: "radius".to_string(),
                    description: "Blur radius".to_string(),
                    parameter_type: ParameterType::Radius,
                    default_value: 5.0,
                    min_value: 1.0,
                    max_value: 20.0,
                },
            ],
        }
    }

    fn create_glow_preset() -> EffectPresetConfig {
        EffectPresetConfig {
            name: "glow".to_string(),
            description: "Bloom/glow effect with brightness extraction and blur".to_string(),
            builder_fn: |config| {
                let lo_config = TextureConfig {
                    width: config.width / 2,
                    height: config.height / 2,
                    format: config.format,
                };
                
                PipelineBuilder::new()
                    .name("Glow Effect")
                    .brightness_extract(config, 0.7)
                    .downsample(lo_config)
                    .gaussian_blur_passes(lo_config, 2, 2.0, 5.0)
                    .bloom_composite_with_curve(config, 2.0, 3.0)
            },
            parameters: vec![
                ParameterDescriptor {
                    name: "threshold".to_string(),
                    description: "Brightness extraction threshold".to_string(),
                    parameter_type: ParameterType::Threshold,
                    default_value: 0.7,
                    min_value: 0.0,
                    max_value: 1.0,
                },
                ParameterDescriptor {
                    name: "intensity".to_string(),
                    description: "Bloom intensity".to_string(),
                    parameter_type: ParameterType::Intensity,
                    default_value: 2.0,
                    min_value: 0.0,
                    max_value: 5.0,
                },
                ParameterDescriptor {
                    name: "curve".to_string(),
                    description: "Bloom curve".to_string(),
                    parameter_type: ParameterType::Custom("curve".to_string()),
                    default_value: 3.0,
                    min_value: 1.0,
                    max_value: 10.0,
                },
            ],
        }
    }

    fn create_invert_preset() -> EffectPresetConfig {
        EffectPresetConfig {
            name: "invert".to_string(),
            description: "Color inversion effect".to_string(),
            builder_fn: |config| {
                PipelineBuilder::new()
                    .name("Invert Effect")
                    .inversion(config, 1.0)
            },
            parameters: vec![
                ParameterDescriptor {
                    name: "darken_darks".to_string(),
                    description: "Amount to darken dark areas".to_string(),
                    parameter_type: ParameterType::Intensity,
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 2.0,
                },
            ],
        }
    }

    fn create_feedback_preset() -> EffectPresetConfig {
        EffectPresetConfig {
            name: "feedback".to_string(),
            description: "Temporal feedback effect".to_string(),
            builder_fn: |config| {
                PipelineBuilder::new()
                    .name("Feedback Effect")
                    .feedback(config, 0.95, 1.0)
            },
            parameters: vec![
                ParameterDescriptor {
                    name: "persistence".to_string(),
                    description: "How much of the previous frame to keep".to_string(),
                    parameter_type: ParameterType::Intensity,
                    default_value: 0.95,
                    min_value: 0.0,
                    max_value: 1.0,
                },
                ParameterDescriptor {
                    name: "frame_history".to_string(),
                    description: "Frame history factor".to_string(),
                    parameter_type: ParameterType::Custom("history".to_string()),
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 2.0,
                },
            ],
        }
    }

    fn create_brightness_extract_preset() -> EffectPresetConfig {
        EffectPresetConfig {
            name: "brightness_extract".to_string(),
            description: "Extract bright areas".to_string(),
            builder_fn: |config| {
                PipelineBuilder::new()
                    .name("Brightness Extract")
                    .brightness_extract(config, 0.5)
            },
            parameters: vec![
                ParameterDescriptor {
                    name: "threshold".to_string(),
                    description: "Brightness threshold".to_string(),
                    parameter_type: ParameterType::Threshold,
                    default_value: 0.5,
                    min_value: 0.0,
                    max_value: 1.0,
                },
            ],
        }
    }

    fn create_color_key_preset() -> EffectPresetConfig {
        EffectPresetConfig {
            name: "color_key".to_string(),
            description: "Extract specific color range".to_string(),
            builder_fn: |config| {
                PipelineBuilder::new()
                    .name("Color Key Extract")
                    .color_key_extract(config, [1.0, 1.0, 1.0], 0.1, 1.0)
            },
            parameters: vec![
                ParameterDescriptor {
                    name: "red".to_string(),
                    description: "Target red component".to_string(),
                    parameter_type: ParameterType::ColorChannel,
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 1.0,
                },
                ParameterDescriptor {
                    name: "green".to_string(),
                    description: "Target green component".to_string(),
                    parameter_type: ParameterType::ColorChannel,
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 1.0,
                },
                ParameterDescriptor {
                    name: "blue".to_string(),
                    description: "Target blue component".to_string(),
                    parameter_type: ParameterType::ColorChannel,
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 1.0,
                },
                ParameterDescriptor {
                    name: "threshold".to_string(),
                    description: "Color matching threshold".to_string(),
                    parameter_type: ParameterType::Threshold,
                    default_value: 0.1,
                    min_value: 0.0,
                    max_value: 1.0,
                },
                ParameterDescriptor {
                    name: "intensity".to_string(),
                    description: "Effect intensity".to_string(),
                    parameter_type: ParameterType::Intensity,
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 2.0,
                },
            ],
        }
    }

    fn create_additive_composite_preset() -> EffectPresetConfig {
        EffectPresetConfig {
            name: "composite_add".to_string(),
            description: "Additive composite with scene".to_string(),
            builder_fn: |config| {
                PipelineBuilder::new()
                    .name("Additive Composite")
                    .simple_additive_composite(config, 1.0)
            },
            parameters: vec![
                ParameterDescriptor {
                    name: "intensity".to_string(),
                    description: "Composite intensity".to_string(),
                    parameter_type: ParameterType::Intensity,
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 2.0,
                },
            ],
        }
    }

    fn create_screen_composite_preset() -> EffectPresetConfig {
        EffectPresetConfig {
            name: "composite_screen".to_string(),
            description: "Screen blend composite with scene".to_string(),
            builder_fn: |config| {
                PipelineBuilder::new()
                    .name("Screen Composite")
                    .simple_screen_composite(config, 1.0)
            },
            parameters: vec![
                ParameterDescriptor {
                    name: "intensity".to_string(),
                    description: "Composite intensity".to_string(),
                    parameter_type: ParameterType::Intensity,
                    default_value: 1.0,
                    min_value: 0.0,
                    max_value: 2.0,
                },
            ],
        }
    }

    // Parameter-driven pipeline creation methods
    fn create_blur_with_params(config: TextureConfig, params: &HashMap<String, f32>, device: &wgpu::Device) -> Result<Pipeline, String> {
        let intensity = params.get("intensity").unwrap_or(&2.0);
        let radius = params.get("radius").unwrap_or(&5.0);
        
        PipelineBuilder::new()
            .name("Parametric Blur Effect")
            .gaussian_blur_passes(config, 2, *intensity, *radius)
            .build(device)
            .map_err(|e| format!("Failed to build parametric blur: {:?}", e))
    }

    fn create_glow_with_params(config: TextureConfig, params: &HashMap<String, f32>, device: &wgpu::Device) -> Result<Pipeline, String> {
        let threshold = params.get("threshold").unwrap_or(&0.7);
        let intensity = params.get("intensity").unwrap_or(&2.0);
        let curve = params.get("curve").unwrap_or(&3.0);
        
        let lo_config = TextureConfig {
            width: config.width / 2,
            height: config.height / 2,
            format: config.format,
        };

        PipelineBuilder::new()
            .name("Parametric Glow Effect")
            .brightness_extract(config, *threshold)
            .downsample(lo_config)
            .gaussian_blur_passes(lo_config, 2, 2.0, 5.0)
            .bloom_composite_with_curve(config, *intensity, *curve)
            .build(device)
            .map_err(|e| format!("Failed to build parametric glow: {:?}", e))
    }

    fn create_invert_with_params(config: TextureConfig, params: &HashMap<String, f32>, device: &wgpu::Device) -> Result<Pipeline, String> {
        let darken_darks = params.get("darken_darks").unwrap_or(&1.0);
        
        PipelineBuilder::new()
            .name("Parametric Invert Effect")
            .inversion(config, *darken_darks)
            .build(device)
            .map_err(|e| format!("Failed to build parametric invert: {:?}", e))
    }

    fn create_feedback_with_params(config: TextureConfig, params: &HashMap<String, f32>, device: &wgpu::Device) -> Result<Pipeline, String> {
        let persistence = params.get("persistence").unwrap_or(&0.95);
        let frame_history = params.get("frame_history").unwrap_or(&1.0);
        
        PipelineBuilder::new()
            .name("Parametric Feedback Effect")
            .feedback(config, *persistence, *frame_history)
            .build(device)
            .map_err(|e| format!("Failed to build parametric feedback: {:?}", e))
    }

    fn create_brightness_extract_with_params(config: TextureConfig, params: &HashMap<String, f32>, device: &wgpu::Device) -> Result<Pipeline, String> {
        let threshold = params.get("threshold").unwrap_or(&0.5);
        
        PipelineBuilder::new()
            .name("Parametric Brightness Extract")
            .brightness_extract(config, *threshold)
            .build(device)
            .map_err(|e| format!("Failed to build parametric brightness extract: {:?}", e))
    }

    fn create_color_key_with_params(config: TextureConfig, params: &HashMap<String, f32>, device: &wgpu::Device) -> Result<Pipeline, String> {
        let red = params.get("red").unwrap_or(&1.0);
        let green = params.get("green").unwrap_or(&1.0);
        let blue = params.get("blue").unwrap_or(&1.0);
        let threshold = params.get("threshold").unwrap_or(&0.1);
        let intensity = params.get("intensity").unwrap_or(&1.0);
        
        PipelineBuilder::new()
            .name("Parametric Color Key Extract")
            .color_key_extract(config, [*red, *green, *blue], *threshold, *intensity)
            .build(device)
            .map_err(|e| format!("Failed to build parametric color key: {:?}", e))
    }

    fn create_additive_composite_with_params(config: TextureConfig, params: &HashMap<String, f32>, device: &wgpu::Device) -> Result<Pipeline, String> {
        let intensity = params.get("intensity").unwrap_or(&1.0);
        
        PipelineBuilder::new()
            .name("Parametric Additive Composite")
            .simple_additive_composite(config, *intensity)
            .build(device)
            .map_err(|e| format!("Failed to build parametric additive composite: {:?}", e))
    }

    fn create_screen_composite_with_params(config: TextureConfig, params: &HashMap<String, f32>, device: &wgpu::Device) -> Result<Pipeline, String> {
        let intensity = params.get("intensity").unwrap_or(&1.0);
        
        PipelineBuilder::new()
            .name("Parametric Screen Composite")
            .simple_screen_composite(config, *intensity)
            .build(device)
            .map_err(|e| format!("Failed to build parametric screen composite: {:?}", e))
    }
}

impl Default for EffectPresets {
    fn default() -> Self {
        Self::new()
    }
}