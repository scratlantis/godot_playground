# Godot particle sim port

This is a Godot 4 port of the Vulkan compute particle simulation in `vka/src/0.1.04/applications/particle_sim`.

The simulation itself runs in `RenderingDevice` compute shaders:

- `shaders/particle_generate.glsl` initializes particles in the same centered sub-volume as the Vulkan generator.
- `shaders/particle_step.glsl` combines the VKA density, pressure, viscosity, random inter-particle gravity, cursor force, damping, and box collision logic into one Godot compute pass.
- `scripts/ParticleSim.gd` drives the compute buffers and displays positions through a `MultiMeshInstance3D`.

The current renderer reads particle positions back from the compute buffer each frame before updating the `MultiMesh`. That keeps the port compact and easy to debug, but it is not the fastest possible Godot architecture. Start around 1024 particles; very large counts will become expensive because this first Godot pass uses a direct neighbor loop.

Runtime controls:

- `R`: reset particle positions
- `G`: toggle earth gravity
- `B`: toggle box collision
- `I`: toggle random inter-particle gravity
- left mouse: attract near the cursor plane
- right mouse: repel near the cursor plane

Open the `godot` folder as a Godot 4 project and run `main.tscn`.
