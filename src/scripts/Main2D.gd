extends Node2D

const GRID_SPACING := 32.0
const WORKGROUP_SIZE := 64
const PARAM_BYTES := 80
const PARTICLE_BYTES := 16
const DENSITY_BYTES := 4

@export_range(16, 8192, 1) var particle_count := 512
@export_range(0.005, 0.15, 0.001) var particle_radius := 0.045
@export_range(0.0, 2.0, 0.001) var density_coefficient := 1.0
@export_range(0.0, 2.0, 0.001) var target_density := 0.15
@export_range(0.0, 2.0, 0.001) var pressure_force_coefficient := 0.18
@export_range(0.0, 10.0, 0.001) var viscosity_force_coefficient := 0.65
@export_range(0.0, 1.0, 0.001) var damping := 0.998
@export_range(0.0, 1.0, 0.001) var border_damping := 0.72
@export_range(-4.0, 4.0, 0.001) var gravity := 0.65

var rd: RenderingDevice
var density_shader: RID
var forces_shader: RID
var density_pipeline: RID
var forces_pipeline: RID
var params_buffer: RID
var particle_buffers: Array[RID] = []
var density_buffer: RID
var read_index := 0
var write_index := 1
var frame_index := 0
var gpu_enabled := false
var particles: PackedVector2Array = PackedVector2Array()

func _ready() -> void:
	DisplayServer.window_set_title("Particle Sim 2D")
	gpu_enabled = _setup_compute()
	_seed_particles()

func _process(delta: float) -> void:
	if gpu_enabled:
		_step_gpu(delta)
		_read_particles_from_gpu()
	queue_redraw()

func _draw() -> void:
	var rect := get_viewport_rect()
	draw_rect(rect, Color(0.035, 0.039, 0.047))
	_draw_grid(rect)
	_draw_bounds(rect)
	_draw_points(rect)

func _exit_tree() -> void:
	if rd == null:
		return

	for buffer in particle_buffers:
		if buffer.is_valid():
			rd.free_rid(buffer)
	for rid in [params_buffer, density_buffer, density_pipeline, forces_pipeline, density_shader, forces_shader]:
		if rid.is_valid():
			rd.free_rid(rid)

func _setup_compute() -> bool:
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		push_warning("RenderingDevice is unavailable; showing seeded particles without simulation.")
		return false

	density_shader = _load_shader("res://shaders/particle_density.glsl")
	forces_shader = _load_shader("res://shaders/particle_forces.glsl")
	density_pipeline = rd.compute_pipeline_create(density_shader)
	forces_pipeline = rd.compute_pipeline_create(forces_shader)
	params_buffer = rd.storage_buffer_create(PARAM_BYTES)
	for i in 2:
		particle_buffers.append(rd.storage_buffer_create(particle_count * PARTICLE_BYTES))
	density_buffer = rd.storage_buffer_create(particle_count * DENSITY_BYTES)
	return true

func _seed_particles() -> void:
	particles.clear()
	var rng := RandomNumberGenerator.new()
	rng.seed = 39
	var data := PackedByteArray()
	data.resize(particle_count * PARTICLE_BYTES)
	for i in particle_count:
		var pos := Vector2(
			0.25 + rng.randf() * 0.5,
			0.18 + rng.randf() * 0.42
		)
		var offset := i * PARTICLE_BYTES
		data.encode_float(offset, pos.x)
		data.encode_float(offset + 4, pos.y)
		data.encode_float(offset + 8, 0.0)
		data.encode_float(offset + 12, 0.0)
		particles.append(pos)

	if gpu_enabled:
		rd.buffer_update(particle_buffers[read_index], 0, data.size(), data)

func _load_shader(path: String) -> RID:
	var source_text := FileAccess.get_file_as_string(path)
	if source_text.begins_with("#[compute]\n"):
		source_text = source_text.substr(11)

	var source := RDShaderSource.new()
	source.language = RenderingDevice.SHADER_LANGUAGE_GLSL
	source.set_stage_source(RenderingDevice.SHADER_STAGE_COMPUTE, source_text)
	var spirv := rd.shader_compile_spirv_from_source(source)
	var compile_error := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if compile_error != "":
		push_error(compile_error)
	return rd.shader_create_from_spirv(spirv)

func _step_gpu(delta: float) -> void:
	var dt := minf(delta, 1.0 / 30.0)
	_upload_params(dt)

	var density_set := rd.uniform_set_create([
		_storage_uniform(0, params_buffer),
		_storage_uniform(1, particle_buffers[read_index]),
		_storage_uniform(2, density_buffer),
	], density_shader, 0)

	var forces_set := rd.uniform_set_create([
		_storage_uniform(0, params_buffer),
		_storage_uniform(1, particle_buffers[read_index]),
		_storage_uniform(2, density_buffer),
		_storage_uniform(3, particle_buffers[write_index]),
	], forces_shader, 0)

	_dispatch(density_pipeline, density_set)
	_dispatch(forces_pipeline, forces_set)
	rd.free_rid(density_set)
	rd.free_rid(forces_set)

	var old_read := read_index
	read_index = write_index
	write_index = old_read
	frame_index += 1

func _dispatch(pipeline: RID, uniform_set: RID) -> void:
	var groups := int(ceil(float(particle_count) / float(WORKGROUP_SIZE)))
	var list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(list, pipeline)
	rd.compute_list_bind_uniform_set(list, uniform_set, 0)
	rd.compute_list_dispatch(list, groups, 1, 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()

func _upload_params(dt: float) -> void:
	var bytes := PackedByteArray()
	bytes.resize(PARAM_BYTES)
	_encode_vec4(bytes, 0, Vector4(0.04, 0.04, 0.96, 0.96))
	_encode_vec4(bytes, 16, Vector4(particle_radius, density_coefficient, target_density, gravity))
	_encode_vec4(bytes, 32, Vector4(pressure_force_coefficient, viscosity_force_coefficient, damping, border_damping))
	_encode_vec4(bytes, 48, Vector4(dt, 0.0, 0.0, 0.0))
	_encode_uvec4(bytes, 64, Vector4i(particle_count, frame_index, 0, 0))
	rd.buffer_update(params_buffer, 0, PARAM_BYTES, bytes)

func _read_particles_from_gpu() -> void:
	var data := rd.buffer_get_data(particle_buffers[read_index])
	particles.resize(particle_count)
	for i in particle_count:
		var offset := i * PARTICLE_BYTES
		particles[i] = Vector2(data.decode_float(offset), data.decode_float(offset + 4))

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

func _draw_grid(rect: Rect2) -> void:
	var minor := Color(0.12, 0.14, 0.16, 0.62)
	var major := Color(0.20, 0.24, 0.27, 0.78)
	var x := 0.5
	var column := 0
	while x <= rect.size.x:
		var color := major if column % 4 == 0 else minor
		draw_line(Vector2(x, 0.0), Vector2(x, rect.size.y), color, 1.0)
		x += GRID_SPACING
		column += 1

	var y := 0.5
	var row := 0
	while y <= rect.size.y:
		var color := major if row % 4 == 0 else minor
		draw_line(Vector2(0.0, y), Vector2(rect.size.x, y), color, 1.0)
		y += GRID_SPACING
		row += 1

func _draw_bounds(rect: Rect2) -> void:
	var margin := 64.0
	var bounds := Rect2(Vector2(margin, margin), rect.size - Vector2.ONE * margin * 2.0)
	draw_rect(bounds, Color(0.45, 0.68, 0.78, 0.18), false, 2.0)

func _draw_points(rect: Rect2) -> void:
	var margin := 72.0
	var area := Rect2(Vector2(margin, margin), rect.size - Vector2.ONE * margin * 2.0)
	for unit in particles:
		var pos := area.position + unit * area.size
		var radius := 3.25
		var color := Color(0.30, 0.78, 0.94, 0.86)
		draw_circle(pos, radius, color)
