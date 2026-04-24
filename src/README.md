# Godot 2D playground

This is the Godot 4 starting point for a standard 2D particle renderer.

The current scene keeps rendering in a plain `Node2D` canvas, while the simulation runs in two brute-force compute passes:

- `main.tscn` opens a `Node2D` scene.
- `scripts/Main2D.gd` seeds particles, dispatches the compute passes, reads positions back, and draws the points with CanvasItem draw calls.
- `shaders/particle_density.glsl` computes all-pairs density.
- `shaders/particle_forces.glsl` computes all-pairs pressure and viscosity forces, then integrates velocity and position.
- `config/simulation_params_2d.json` provides default runtime parameters. The in-app panel can save overrides to `user://simulation_params_2d.json`.

This is deliberately O(n^2). The next step is to replace the inner all-pairs loops with a hash grid, matching the structure of the VKA implementation.

Open `src` as a Godot project and run `main.tscn`.
