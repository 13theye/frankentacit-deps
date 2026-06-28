//! src/builder/pipeline_builder.rs
//!
//! A builder for constructing a Pipeline

use crate::{
    components::*,
    pipeline::{CompositorComponent, Pipeline, SimpleComponent, TextureConfig},
};
use nannou::wgpu;

#[derive(Debug)]
pub enum PipelineBuilderError {
    InvalidConfiguration(String),
    MissingRequiredComponent(String),
    IncompatibleComponents(String, String),
}

/// Represents any stage config in the pipeline
#[derive(Debug, Clone)]
enum ComponentConfig {
    Simple(SimpleComponentConfig),
    Compositor(CompositorComponentConfig),
}

pub struct PipelineBuilder {
    name: String,
    stages: Vec<ComponentConfig>,
    intermediate_textures_needed: usize,
    validation_enabled: bool,
    scene_replacement_stage: Option<usize>,
    // Default single-input texture handler
    input_texture_name: Option<String>,
    // For multi-input support
    input_texture_names: Vec<String>,
    // Output of this pipeline
    output_texture_name: Option<String>,
}

/// TODO: split this into a trait so that each component can manage its own config type
#[derive(Debug, Clone)]
enum SimpleComponentConfig {
    Brightness {
        output_config: TextureConfig,
        threshold: f32,
    },
    Blur {
        output_config: TextureConfig,
        direction: [f32; 2],
        adaptive_scaling: f32,
        max_radius: f32,
    },
    ColorKey {
        output_config: TextureConfig,
        target_color: [f32; 3],
        threshold: f32,
        intensity: f32,
    },
    Darkness {
        output_config: TextureConfig,
        threshold: f32,
    },
    Downsample {
        output_config: TextureConfig,
    },
    Resample {
        output_config: TextureConfig,
    },
    Inversion {
        output_config: TextureConfig,
        darken_darks: f32,
    },
    Feedback {
        output_config: TextureConfig,
        persistence: f32,
        frame_history: f32,
    },
    Premultiply {
        output_config: TextureConfig,
    },
}

/// TODO: split this into a trait so that each component can manage its own config type
#[derive(Debug, Clone)]
enum CompositorComponentConfig {
    BloomComposite {
        output_config: TextureConfig,
        intensity: f32,
        intensity_curve: f32,
    },
    SimpleComposite {
        blend_mode: BlendMode,
        output_config: TextureConfig,
        intensity: f32,
    },
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            name: "Custom Effect".to_string(),
            stages: Vec::new(),
            intermediate_textures_needed: 0,
            validation_enabled: true,
            scene_replacement_stage: None,
            input_texture_name: None,
            input_texture_names: Vec::new(),
            output_texture_name: None,
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set the input texture name for this pipeline (for named texture system)
    pub fn input_texture(mut self, texture_name: &str) -> Self {
        self.input_texture_name = Some(texture_name.to_string());
        self
    }

    /// Set multiple input texture names for this pipeline (for multi-input compositors)
    pub fn input_textures(mut self, texture_names: &[&str]) -> Self {
        self.input_texture_names = texture_names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set the output texture name for this pipeline (for named texture system)
    pub fn output_texture(mut self, texture_name: &str) -> Self {
        self.output_texture_name = Some(texture_name.to_string());
        self
    }

    pub fn disable_validation(mut self) -> Self {
        self.validation_enabled = false;
        self
    }

    /// Mark the current stage as the new scene view for subsequent pipeline stages.
    /// This allows replacing the original scene_view with an intermediate result,
    /// which affects compositing stages later in the pipeline.
    pub fn update_scene(mut self) -> Self {
        if !self.stages.is_empty() {
            self.scene_replacement_stage = Some(self.stages.len() - 1);
        }
        self
    }

    // Component addition methods
    pub fn brightness_extract(mut self, output_config: TextureConfig, threshold: f32) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::Brightness {
                output_config,
                threshold,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn color_key_extract(
        mut self,
        output_config: TextureConfig,
        target_color: [f32; 3],
        threshold: f32,
        intensity: f32,
    ) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::ColorKey {
                output_config,
                target_color,
                threshold,
                intensity,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn darkness_extract(mut self, output_config: TextureConfig, threshold: f32) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::Darkness {
                output_config,
                threshold,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn gaussian_blur_horizontal(
        mut self,
        output_config: TextureConfig,
        adaptive_scaling: f32,
        max_radius: f32,
    ) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::Blur {
                output_config,
                direction: [1.0, 0.0],
                adaptive_scaling,
                max_radius,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn gaussian_blur_vertical(
        mut self,
        output_config: TextureConfig,
        adaptive_scaling: f32,
        max_radius: f32,
    ) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::Blur {
                output_config,
                direction: [0.0, 0.7],
                adaptive_scaling,
                max_radius,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn gaussian_blur_passes(
        mut self,
        output_config: TextureConfig,
        passes: u32,
        adaptive_scaling: f32,
        max_radius: f32,
    ) -> Self {
        for _ in 0..passes {
            self = self.gaussian_blur_horizontal(output_config, adaptive_scaling, max_radius);
            self = self.gaussian_blur_vertical(output_config, adaptive_scaling, max_radius);
        }
        self
    }

    pub fn resample(mut self, output_config: TextureConfig) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::Resample {
                output_config,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn downsample(mut self, output_config: TextureConfig) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::Downsample {
                output_config,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn inversion(mut self, output_config: TextureConfig, darken_darks: f32) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::Inversion {
                output_config,
                darken_darks,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn feedback(
        mut self,
        output_config: TextureConfig,
        persistence: f32,
        frame_history: f32,
    ) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::Feedback {
                output_config,
                persistence,
                frame_history,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    /// Convert straight alpha to premultiplied alpha
    /// Use this when compositing content from sources that output straight alpha
    /// (like Nannou Draw) into a premultiplied alpha pipeline
    pub fn premultiply(mut self, output_config: TextureConfig) -> Self {
        self.stages
            .push(ComponentConfig::Simple(SimpleComponentConfig::Premultiply {
                output_config,
            }));
        self.intermediate_textures_needed += 1;
        self
    }

    /// Create a BloomComposite component with the given intensity and curve (outputs Rgba16Float)
    pub fn bloom_composite_with_curve(
        mut self,
        output_config: TextureConfig,
        intensity: f32,
        intensity_curve: f32,
    ) -> Self {
        self.stages.push(ComponentConfig::Compositor(
            CompositorComponentConfig::BloomComposite {
                intensity,
                intensity_curve,
                output_config,
            },
        ));
        self.intermediate_textures_needed += 1;
        self
    }

    /// Create a SimpleComposite component with Additive blend mode
    pub fn simple_additive_composite(
        mut self,
        output_config: TextureConfig,
        intensity: f32,
    ) -> Self {
        self.stages.push(ComponentConfig::Compositor(
            CompositorComponentConfig::SimpleComposite {
                blend_mode: BlendMode::Add,
                intensity,
                output_config,
            },
        ));
        self.intermediate_textures_needed += 1;
        self
    }

    /// Create a SimpleComposite component with Screen blend mode
    pub fn simple_screen_composite(mut self, output_config: TextureConfig, intensity: f32) -> Self {
        self.stages.push(ComponentConfig::Compositor(
            CompositorComponentConfig::SimpleComposite {
                blend_mode: BlendMode::Screen,
                intensity,
                output_config,
            },
        ));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn simple_multiply_composite(
        mut self,
        output_config: TextureConfig,
        intensity: f32,
    ) -> Self {
        self.stages.push(ComponentConfig::Compositor(
            CompositorComponentConfig::SimpleComposite {
                blend_mode: BlendMode::Multiply,
                intensity,
                output_config,
            },
        ));
        self.intermediate_textures_needed += 1;
        self
    }

    pub fn simple_overlay_composite(
        mut self,
        output_config: TextureConfig,
        intensity: f32,
    ) -> Self {
        self.stages.push(ComponentConfig::Compositor(
            CompositorComponentConfig::SimpleComposite {
                blend_mode: BlendMode::Overlay,
                intensity,
                output_config,
            },
        ));
        self.intermediate_textures_needed += 1;
        self
    }

    /// Create a SimpleComposite component with Over (normal alpha-over) blend mode
    pub fn simple_over_composite(mut self, output_config: TextureConfig, intensity: f32) -> Self {
        self.stages.push(ComponentConfig::Compositor(
            CompositorComponentConfig::SimpleComposite {
                blend_mode: BlendMode::Over,
                intensity,
                output_config,
            },
        ));
        self.intermediate_textures_needed += 1;
        self
    }

    /// Create a SimpleComposite component with Lighten blend mode
    /// Takes the brighter pixel (by luminance) - useful for particle compositing
    pub fn simple_lighten_composite(
        mut self,
        output_config: TextureConfig,
        intensity: f32,
    ) -> Self {
        self.stages.push(ComponentConfig::Compositor(
            CompositorComponentConfig::SimpleComposite {
                blend_mode: BlendMode::Lighten,
                intensity,
                output_config,
            },
        ));
        self.intermediate_textures_needed += 1;
        self
    }

    // Validation methods
    fn validate_configuration(&self) -> Result<(), PipelineBuilderError> {
        if !self.validation_enabled {
            return Ok(());
        }

        if self.stages.is_empty() {
            return Err(PipelineBuilderError::MissingRequiredComponent(
                "At least one component is required".to_string(),
            ));
        }

        // Check for common anti-patterns
        self.validate_blur_patterns()?;
        self.validate_composite_patterns()?;

        Ok(())
    }

    fn validate_blur_patterns(&self) -> Result<(), PipelineBuilderError> {
        let mut consecutive_same_direction = 0;
        let mut last_blur_direction: Option<[f32; 2]> = None;

        for stage in &self.stages {
            if let ComponentConfig::Simple(SimpleComponentConfig::Blur { direction, .. }) = stage {
                if Some(*direction) == last_blur_direction {
                    consecutive_same_direction += 1;
                    if consecutive_same_direction > 2 {
                        return Err(PipelineBuilderError::InvalidConfiguration(
                            "More than 2 consecutive blur passes in the same direction is inefficient".to_string(),
                        ));
                    }
                } else {
                    consecutive_same_direction = 1;
                    last_blur_direction = Some(*direction);
                }
            } else {
                consecutive_same_direction = 0;
                last_blur_direction = None;
            }
        }

        Ok(())
    }

    fn validate_composite_patterns(&self) -> Result<(), PipelineBuilderError> {
        let composite_count = self
            .stages
            .iter()
            .filter(|stage| matches!(stage, ComponentConfig::Compositor(_)))
            .count();

        // Note: We now allow multiple compositors since they can be anywhere in the pipeline
        // This validation could be enhanced to check for specific patterns if needed
        if composite_count > 1 {
            return Err(PipelineBuilderError::InvalidConfiguration(
                "Warning: Multiple Composite stages in pipeline".to_string(),
            ));
        }

        Ok(())
    }

    // Build methods
    pub fn build(self, device: &wgpu::Device) -> Result<Pipeline, PipelineBuilderError> {
        self.validate_configuration()?;

        let mut pipeline_chain = Pipeline::new_with_name(&self.name);
        pipeline_chain.set_scene_replacement_stage(self.scene_replacement_stage);

        // Set texture names for named texture coordination
        pipeline_chain.set_input_texture_name(self.input_texture_name.clone());
        pipeline_chain.set_input_texture_names(self.input_texture_names.clone());
        pipeline_chain.set_output_texture_name(self.output_texture_name.clone());

        // Create and add components in order
        for stage_config in &self.stages {
            match stage_config {
                ComponentConfig::Simple(simple_config) => {
                    let (component, output_config) = match simple_config {
                        SimpleComponentConfig::Brightness {
                            output_config,
                            threshold,
                        } => (
                            Box::new(BrightnessComponent::new(device, *output_config, *threshold))
                                as Box<dyn SimpleComponent>,
                            *output_config,
                        ),
                        SimpleComponentConfig::Blur {
                            output_config,
                            direction,
                            adaptive_scaling,
                            max_radius,
                        } => (
                            Box::new(BlurComponent::new(
                                device,
                                *output_config,
                                *direction,
                                *adaptive_scaling,
                                *max_radius,
                            )) as Box<dyn SimpleComponent>,
                            *output_config,
                        ),
                        SimpleComponentConfig::ColorKey {
                            output_config,
                            target_color,
                            threshold,
                            intensity,
                        } => (
                            Box::new(ColorKeyComponent::new(
                                device,
                                *output_config,
                                *target_color,
                                *threshold,
                                *intensity,
                            )) as Box<dyn SimpleComponent>,
                            *output_config,
                        ),
                        SimpleComponentConfig::Darkness {
                            output_config,
                            threshold,
                        } => (
                            Box::new(DarknessComponent::new(device, *output_config, *threshold))
                                as Box<dyn SimpleComponent>,
                            *output_config,
                        ),
                        SimpleComponentConfig::Downsample { output_config } => (
                            Box::new(DownsampleComponent::new(device, *output_config))
                                as Box<dyn SimpleComponent>,
                            *output_config,
                        ),
                        SimpleComponentConfig::Resample { output_config } => (
                            Box::new(ResampleComponent::new(device, *output_config))
                                as Box<dyn SimpleComponent>,
                            *output_config,
                        ),
                        SimpleComponentConfig::Inversion {
                            output_config,
                            darken_darks,
                        } => (
                            Box::new(InversionComponent::new(
                                device,
                                *output_config,
                                *darken_darks,
                            )) as Box<dyn SimpleComponent>,
                            *output_config,
                        ),
                        SimpleComponentConfig::Feedback {
                            output_config,
                            persistence,
                            frame_history,
                        } => (
                            Box::new(FeedbackComponent::new(
                                device,
                                *output_config,
                                *persistence,
                                *frame_history,
                            )) as Box<dyn SimpleComponent>,
                            *output_config,
                        ),
                        SimpleComponentConfig::Premultiply { output_config } => (
                            Box::new(PremultiplyComponent::new(device, *output_config))
                                as Box<dyn SimpleComponent>,
                            *output_config,
                        ),
                    };
                    pipeline_chain.add_component(component);
                    pipeline_chain.setup_intermediate_texture(device, output_config);
                }
                ComponentConfig::Compositor(compositor_config) => {
                    let component: Box<dyn CompositorComponent> = match compositor_config {
                        CompositorComponentConfig::BloomComposite {
                            intensity,
                            intensity_curve,
                            output_config,
                        } => Box::new(BloomCompositeComponent::new(
                            device,
                            *output_config,
                            *intensity,
                            *intensity_curve,
                        )),
                        CompositorComponentConfig::SimpleComposite {
                            blend_mode,
                            intensity,
                            output_config,
                        } => Box::new(SimpleCompositeComponent::new(
                            device,
                            *blend_mode,
                            *output_config,
                            *intensity,
                        )),
                    };
                    pipeline_chain.add_component(component);
                    // Compositors also need intermediate textures for their output
                    // The chain will automatically rig the scene_view and the previous stage's output as
                    // the two input textures of the comppositor.
                    pipeline_chain.setup_intermediate_texture(
                        device,
                        match compositor_config {
                            CompositorComponentConfig::BloomComposite { output_config, .. }
                            | CompositorComponentConfig::SimpleComposite {
                                output_config, ..
                            } => *output_config,
                        },
                    );
                }
            }
        }

        Ok(pipeline_chain)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
