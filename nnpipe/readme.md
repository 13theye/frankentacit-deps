
Nnpipe is a library meant to cover the middle ground between power of low-level libraries like WGPU and the ease-of-use of high-level APIs like Nannou. It's aimed at users who want to leverage the power of parallelism afforded by GPUs while sidestepping a lot of boilerplate and complexity.

Nnpipe works with and extends Nannou's Draw API. The Draw API can be slow when drawing a large number of objects to the screen, because Rust allows only one simultaneous mutable access to the Draw context. This means that Draw calls must be sequential -- a performance bottleneck. Nnpipe aims to provide a suite of CPU/GPU hybrid renderers that enable users to pass CPU-computed geometry to GPU renderers, which can be run in parallel. 

For example, this statement encodes instructions for the particle renderer to take an array of particle positions, with color and alpha, and draw the results to the texture named "particles":

```Rust
particle_renderer.encode_into(
    &mut encoder,
    queue,
    &particle_array,
    nnpipe.get_named_texture("particles").unwrap(),
);
```

This approach handles about 10x more particles than an implementation using Nannou::Draw, while retaining the ease of performing particle system updates on the CPU with high-level code.

Additionally, Nnpipe makes it possible to declare a rendering and post-processing pipeline with simple statements like:

```Rust

// Initialize an instance of the nnpipe renderer

let mut nnpipe = Nnpipe::new(
    device,
    config.rendering.texture_width,
    config.rendering.texture_height,
    config.rendering.texture_samples,
);

// Initialize a Particle Renderer with a capacity of 25000 particles
let particle_renderer: ParticleRenderer = ParticleRenderer::new(device, hi_config, 25000);

// Set up a target texture for the Particle Renderer
nnpipe.create_named_texture(device, "particles", hi_config);

// Define an bloom effect pipeline
let bloom_effect = PipelineBuilder::new()
    .name("Particle Effects Pipeline")
    .input_texture("particles")
    .brightness_extract(med_config, 0.7)
    .downsample(lo_config)
    .gaussian_blur_passes(lo_config, 2, 2.0, 5.0)
    .bloom_composite_with_curve(hi_config, 2.0, 3.0)
    .output_texture("effects_output")
    .build(device);

  
// Add the effect to Nnpipe
if let Ok(effect) = bloom_effect {
    nnpipe.add_multi_pipeline("bloom", effect);
}

```

Modules
- Pipeline: a series of shaders chained together. Functions as a single node that operates on one or more input textures, and draws to an output texture.
- PipelineBuilder: composer for pre-configuring and chaining together post-processing shaders.
- Components: WGSL shaders with Rust interfaces. Includes fragment  and post-processing shaders.
- Renderers: WGSL shaders that process arrays of points into particles, line segments, heatmaps, etc.
- Nnpipe: the main entry point for client apps.