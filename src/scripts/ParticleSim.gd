extends Node3D

const WORKGROUP_SIZE := 64
const PARAM_BYTES := 128
const PARTICLE_BYTES := 32
const PREDICTED_BYTES := 16
const DENSITY_BYTES := 4

const PARTICLE_EARTH_GRAVITY_BIT := 0x1
const PARTICLE_GRAVITY_BIT := 0x2
const PARTICLE_BOX_BIT := 0x4

@export_range(1, 20000, 1) var particle_count := 1024
@export_range(0, 100, 1) var particle_seed := 39
@export_range(0.005, 0.2, 0.001) var particle_radius := 0.035
@export_range(1, 10, 1) var simulation_steps_per_frame := 2
@export_range(0.0, 10.0, 0.001) var gravity := 1.033
@export_range(0.0, 10.0, 0.001) var inter_particle_gravity := 1.5
@export_range(0.0, 1.0, 0.001) var damping := 0.995
@export_range(0.0, 1.0, 0.001) var border_damping := 0.694
@export_range(0.001, 1.0, 0.001) var density_coefficient := 1.0
@export_range(0.0, 0.5, 0.001) var pressure_force_coefficient := 0.076
@export_range(0.0, 10.0, 0.001) var viscosity_force_coefficient := 0.165
@export_range(0.0, 1.0, 0.001) var target_density := 1.0
@export_range(0.01, 1.0, 0.001) var mouse_influence_radius := 0.35
@export_range(0.0, 10.0, 0.001) var mouse_influence_strength := 5.0

var rd: RenderingDevice
var generate_shader: RID
var density_shader: RID
var step_shader: RID
var generate_pipeline: RID
var density_pipeline: RID
var step_pipeline: RID
var params_buffer: RID
var particle_buffers: Array[RID] = []
var predicted_buffers: Array[RID] = []
var density_buffer: RID
var read_index := 0
var write_index := 1
var frame_index := 0
var toggle_flags := PARTICLE_EARTH_GRAVITY_BIT | PARTICLE_BOX_BIT

var multimesh_instance: MultiMeshInstance3D
var multimesh: MultiMesh
var cursor_world := Vector3(0.5, 0.5, 0.5)

func _ready() -> void:
	_setup_rendering()
	if not _setup_compute():
		set_process(false)
		return
	_reset_simulation()

func _process(delta: float) -> void:
	_update_cursor()
	_step_simulation(delta)
	_update_multimesh_from_gpu()
	frame_index += 1

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_R:
			_reset_simulation()
		elif event.keycode == KEY_G:
			toggle_flags ^= PARTICLE_EARTH_GRAVITY_BIT
		elif event.keycode == KEY_B:
			toggle_flags ^= PARTICLE_BOX_BIT
		elif event.keycode == KEY_I:
			toggle_flags ^= PARTICLE_GRAVITY_BIT

func _exit_tree() -> void:
	if rd == null:
		return

	for buffer in particle_buffers:
		if buffer.is_valid():
			rd.free_rid(buffer)
	for buffer in predicted_buffers:
		if buffer.is_valid():
			rd.free_rid(buffer)
	for rid in [params_buffer, density_buffer, generate_pipeline, density_pipeline, step_pipeline, generate_shader, density_shader, step_shader]:
		if rid.is_valid():
			rd.free_rid(rid)

func _setup_rendering() -> void:
	multimesh = MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_3D
	multimesh.instance_count = particle_count
	multimesh.visible_instance_count = particle_count

	var sphere := SphereMesh.new()
	sphere.radius = particle_radius * 0.35
	sphere.height = particle_radius * 0.7
	sphere.radial_segments = 8
	sphere.rings = 4
	var particle_material := StandardMaterial3D.new()
	particle_material.albedo_color = Color(0.25, 0.78, 1.0)
	particle_material.emission_enabled = true
	particle_material.emission = Color(0.08, 0.28, 0.5)
	sphere.material = particle_material
	multimesh.mesh = sphere

	multimesh_instance = MultiMeshInstance3D.new()
	multimesh_instance.name = "Particles"
	multimesh_instance.multimesh = multimesh
	add_child(multimesh_instance)

	var box := MeshInstance3D.new()
	box.name = "SimulationBox"
	var box_mesh := BoxMesh.new()
	box_mesh.size = Vector3.ONE
	box.mesh = box_mesh
	box.position = Vector3(0.5, 0.5, 0.5)
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.18, 0.2, 0.24, 0.12)
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	box.material_override = material
	add_child(box)

func _setup_compute() -> bool:
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		push_warning("RenderingDevice is unavailable; particle compute simulation is disabled.")
		return false
	generate_shader = _load_shader("res://shaders/particle_generate.glsl")
	density_shader = _load_shader("res://shaders/particle_density.glsl")
	step_shader = _load_shader("res://shaders/particle_step.glsl")
	generate_pipeline = rd.compute_pipeline_create(generate_shader)
	density_pipeline = rd.compute_pipeline_create(density_shader)
	step_pipeline = rd.compute_pipeline_create(step_shader)

	params_buffer = rd.storage_buffer_create(PARAM_BYTES)
	var particle_size := particle_count * PARTICLE_BYTES
	var predicted_size := particle_count * PREDICTED_BYTES
	for i in 2:
		particle_buffers.append(rd.storage_buffer_create(particle_size))
		predicted_buffers.append(rd.storage_buffer_create(predicted_size))
	density_buffer = rd.storage_buffer_create(particle_count * DENSITY_BYTES)
	return true

func _load_shader(path: String) -> RID:
	var source_text := FileAccess.get_file_as_string(path)
	if source_text.begins_with("#[compute]\n"):
		source_text = source_text.substr(11)
	var source := RDShaderSource.new()
	source.language = RenderingDevice.SHADER_LANGUAGE_GLSL
	source.set_stage_source(RenderingDevice.SHADER_STAGE_COMPUTE, source_text)
	var spirv := rd.shader_compile_spirv_from_source(source)
	if spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE) != "":
		push_error(spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE))
	return rd.shader_create_from_spirv(spirv)

func _reset_simulation() -> void:
	read_index = 0
	write_index = 1
	frame_index = 0
	_upload_params(16.667)
	var set := rd.uniform_set_create([
		_storage_uniform(0, params_buffer),
		_storage_uniform(1, particle_buffers[read_index]),
		_storage_uniform(2, predicted_buffers[read_index]),
	], generate_shader, 0)
	_dispatch_single(generate_pipeline, set)
	rd.free_rid(set)
	_update_multimesh_from_gpu()

func _step_simulation(delta: float) -> void:
	var dt_ms: float = minf(delta * 1000.0, 20.0) / float(max(simulation_steps_per_frame, 1))
	for step in simulation_steps_per_frame:
		_upload_params(dt_ms, step)
		var density_set := rd.uniform_set_create([
			_storage_uniform(0, params_buffer),
			_storage_uniform(1, predicted_buffers[read_index]),
			_storage_uniform(2, density_buffer),
		], density_shader, 0)
		var step_set := rd.uniform_set_create([
			_storage_uniform(0, params_buffer),
			_storage_uniform(1, particle_buffers[read_index]),
			_storage_uniform(2, predicted_buffers[read_index]),
			_storage_uniform(3, density_buffer),
			_storage_uniform(4, particle_buffers[write_index]),
			_storage_uniform(5, predicted_buffers[write_index]),
		], step_shader, 0)
		_dispatch_density_and_step(density_set, step_set)
		rd.free_rid(density_set)
		rd.free_rid(step_set)
		var old_read := read_index
		read_index = write_index
		write_index = old_read

func _dispatch_single(pipeline: RID, uniform_set: RID) -> void:
	var groups := int(ceil(float(particle_count) / float(WORKGROUP_SIZE)))
	var list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(list, pipeline)
	rd.compute_list_bind_uniform_set(list, uniform_set, 0)
	rd.compute_list_dispatch(list, groups, 1, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()

func _dispatch_density_and_step(density_set: RID, step_set: RID) -> void:
	_dispatch_single(density_pipeline, density_set)
	_dispatch_single(step_pipeline, step_set)

func _upload_params(dt_ms: float, step_index: int = 0) -> void:
	var bytes := PackedByteArray()
	bytes.resize(PARAM_BYTES)
	_encode_vec4(bytes, 0, Vector4(0.0, 0.0, 0.0, 0.0))
	_encode_vec4(bytes, 16, Vector4(1.0, 1.0, 1.0, 0.0))
	_encode_vec4(bytes, 32, Vector4(cursor_world.x, cursor_world.y, cursor_world.z, mouse_influence_radius))
	_encode_uvec4(bytes, 48, Vector4i(
		1 if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT) else 0,
		1 if Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT) else 0,
		0,
		0
	))
	_encode_vec4(bytes, 64, Vector4(particle_radius, dt_ms, damping, border_damping))
	_encode_vec4(bytes, 80, Vector4(gravity, inter_particle_gravity, pressure_force_coefficient, viscosity_force_coefficient))
	_encode_vec4(bytes, 96, Vector4(density_coefficient, target_density, mouse_influence_strength, 0.0))
	_encode_uvec4(bytes, 112, Vector4i(particle_count, frame_index, toggle_flags, particle_seed + step_index))
	rd.buffer_update(params_buffer, 0, PARAM_BYTES, bytes)

func _update_multimesh_from_gpu() -> void:
	var data := rd.buffer_get_data(particle_buffers[read_index])
	var basis := Basis.IDENTITY.scaled(Vector3.ONE * particle_radius * 0.45)
	for i in particle_count:
		var offset := i * PARTICLE_BYTES
		var pos := Vector3(
			data.decode_float(offset),
			data.decode_float(offset + 4),
			data.decode_float(offset + 8)
		)
		multimesh.set_instance_transform(i, Transform3D(basis, pos))

func _update_cursor() -> void:
	var camera := get_viewport().get_camera_3d()
	if camera == null:
		return

	var mouse := get_viewport().get_mouse_position()
	var origin := camera.project_ray_origin(mouse)
	var dir := camera.project_ray_normal(mouse)
	if abs(dir.z) < 0.0001:
		return

	var t := (0.5 - origin.z) / dir.z
	if t <= 0.0:
		return

	cursor_world = origin + dir * t
	cursor_world = Vector3(
		clampf(cursor_world.x, 0.0, 1.0),
		clampf(cursor_world.y, 0.0, 1.0),
		clampf(cursor_world.z, 0.0, 1.0)
	)

func _storage_uniform(binding: int, rid: RID) -> RDUniform:
	var uniform := RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.binding = binding
	uniform.add_id(rid)
	return uniform

func _encode_vec4(bytes: PackedByteArray, offset: int, value: Vector4) -> void:
	bytes.encode_float(offset, value.x)
	bytes.encode_float(offset + 4, value.y)
	bytes.encode_float(offset + 8, value.z)
	bytes.encode_float(offset + 12, value.w)

func _encode_uvec4(bytes: PackedByteArray, offset: int, value: Vector4i) -> void:
	bytes.encode_u32(offset, value.x)
	bytes.encode_u32(offset + 4, value.y)
	bytes.encode_u32(offset + 8, value.z)
	bytes.encode_u32(offset + 12, value.w)
