extends Node2D

const GRID_SPACING := 32.0
const WORKGROUP_SIZE := 64
const PARAM_BYTES := 80
const PARTICLE_BYTES := 16
const DENSITY_BYTES := 4
const DEFAULT_CONFIG_PATH := "res://config/simulation_params_2d.json"
const USER_CONFIG_PATH := "user://simulation_params_2d.json"

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
var predicted_buffers: Array[RID] = []
var density_buffer: RID
var read_index := 0
var write_index := 1
var frame_index := 0
var gpu_enabled := false
var particles: PackedVector2Array = PackedVector2Array()
var param_defs: Array[Dictionary] = []
var param_widgets: Dictionary = {}
var ui_root: Control

func _ready() -> void:
	DisplayServer.window_set_title("Particle Sim 2D")
	_init_param_defs()
	_load_params()
	gpu_enabled = _setup_compute()
	_seed_particles()
	_build_param_ui()

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
	for buffer in predicted_buffers:
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
	_create_sim_buffers()
	return true

func _create_sim_buffers() -> void:
	params_buffer = rd.storage_buffer_create(PARAM_BYTES)
	for i in 2:
		particle_buffers.append(rd.storage_buffer_create(particle_count * PARTICLE_BYTES))
		predicted_buffers.append(rd.storage_buffer_create(particle_count * PARTICLE_BYTES))
	density_buffer = rd.storage_buffer_create(particle_count * DENSITY_BYTES)

func _recreate_sim_buffers() -> void:
	if rd == null:
		_seed_particles()
		return

	for buffer in particle_buffers:
		if buffer.is_valid():
			rd.free_rid(buffer)
	particle_buffers.clear()
	for buffer in predicted_buffers:
		if buffer.is_valid():
			rd.free_rid(buffer)
	predicted_buffers.clear()

	for rid in [params_buffer, density_buffer]:
		if rid.is_valid():
			rd.free_rid(rid)

	read_index = 0
	write_index = 1
	frame_index = 0
	_create_sim_buffers()
	_seed_particles()

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
		rd.buffer_update(predicted_buffers[read_index], 0, data.size(), data)

func _init_param_defs() -> void:
	param_defs = [
		{"name": "particle_count", "label": "Particle Count", "type": "int", "min": 16.0, "max": 4096.0, "step": 16.0, "restart": true},
		{"name": "particle_radius", "label": "Particle Radius", "type": "float", "min": 0.005, "max": 0.15, "step": 0.001},
		{"name": "density_coefficient", "label": "Density Coef", "type": "float", "min": 0.0, "max": 2.0, "step": 0.001},
		{"name": "target_density", "label": "Target Density", "type": "float", "min": 0.0, "max": 2.0, "step": 0.001},
		{"name": "pressure_force_coefficient", "label": "Pressure Force", "type": "float", "min": 0.0, "max": 2.0, "step": 0.001},
		{"name": "viscosity_force_coefficient", "label": "Viscosity Force", "type": "float", "min": 0.0, "max": 10.0, "step": 0.001},
		{"name": "damping", "label": "Particle Damping", "type": "float", "min": 0.0, "max": 1.0, "step": 0.001},
		{"name": "border_damping", "label": "Border Damping", "type": "float", "min": 0.0, "max": 1.0, "step": 0.001},
		{"name": "gravity", "label": "Gravity", "type": "float", "min": -4.0, "max": 4.0, "step": 0.001},
	]

func _build_param_ui() -> void:
	var layer := CanvasLayer.new()
	layer.name = "ParameterLayer"
	add_child(layer)

	ui_root = PanelContainer.new()
	ui_root.name = "ParameterPanel"
	ui_root.position = Vector2(16.0, 16.0)
	ui_root.custom_minimum_size = Vector2(340.0, 0.0)
	layer.add_child(ui_root)

	var margin := MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 12)
	margin.add_theme_constant_override("margin_top", 10)
	margin.add_theme_constant_override("margin_right", 12)
	margin.add_theme_constant_override("margin_bottom", 10)
	ui_root.add_child(margin)

	var rows := VBoxContainer.new()
	rows.add_theme_constant_override("separation", 8)
	margin.add_child(rows)

	for def in param_defs:
		_add_param_row(rows, def)

	var buttons := HBoxContainer.new()
	buttons.add_theme_constant_override("separation", 8)
	rows.add_child(buttons)

	var save_button := Button.new()
	save_button.text = "Save"
	save_button.pressed.connect(_save_params)
	buttons.add_child(save_button)

	var load_button := Button.new()
	load_button.text = "Load"
	load_button.pressed.connect(func() -> void:
		_load_params()
		_sync_param_ui()
		_recreate_sim_buffers()
	)
	buttons.add_child(load_button)

	var reset_button := Button.new()
	reset_button.text = "Reset"
	reset_button.pressed.connect(func() -> void:
		_load_params(DEFAULT_CONFIG_PATH)
		_sync_param_ui()
		_recreate_sim_buffers()
	)
	buttons.add_child(reset_button)

func _add_param_row(parent: VBoxContainer, def: Dictionary) -> void:
	var key := String(def["name"])
	var row := VBoxContainer.new()
	row.add_theme_constant_override("separation", 3)
	parent.add_child(row)

	var title := HBoxContainer.new()
	row.add_child(title)

	var name_label := Label.new()
	name_label.text = String(def["label"])
	name_label.custom_minimum_size = Vector2(180.0, 0.0)
	title.add_child(name_label)

	var value_label := Label.new()
	value_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	value_label.custom_minimum_size = Vector2(110.0, 0.0)
	title.add_child(value_label)

	var control: Range
	if def.get("type", "float") == "int":
		var spin := SpinBox.new()
		spin.min_value = float(def["min"])
		spin.max_value = float(def["max"])
		spin.step = float(def["step"])
		spin.rounded = true
		control = spin
	else:
		var slider := HSlider.new()
		slider.min_value = float(def["min"])
		slider.max_value = float(def["max"])
		slider.step = float(def["step"])
		slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		control = slider

	control.value = float(get(key))
	control.value_changed.connect(func(value: float) -> void:
		_set_param_value(key, value, bool(def.get("restart", false)))
		_update_param_label(key)
	)
	row.add_child(control)

	param_widgets[key] = {
		"control": control,
		"value_label": value_label,
		"def": def,
	}
	_update_param_label(key)

func _set_param_value(key: String, value: float, restart: bool) -> void:
	var def := _get_param_def(key)
	if def.get("type", "float") == "int":
		set(key, int(round(value)))
	else:
		set(key, value)

	if restart:
		_recreate_sim_buffers()

func _get_param_def(key: String) -> Dictionary:
	for def in param_defs:
		if String(def["name"]) == key:
			return def
	return {}

func _sync_param_ui() -> void:
	for key in param_widgets.keys():
		var entry: Dictionary = param_widgets[key]
		var control: Range = entry["control"]
		control.set_value_no_signal(float(get(String(key))))
		_update_param_label(String(key))

func _update_param_label(key: String) -> void:
	if not param_widgets.has(key):
		return

	var entry: Dictionary = param_widgets[key]
	var def: Dictionary = entry["def"]
	var label: Label = entry["value_label"]
	if def.get("type", "float") == "int":
		label.text = "%d" % int(get(key))
	else:
		label.text = "%.3f" % float(get(key))

func _params_to_dict() -> Dictionary:
	var values := {}
	for def in param_defs:
		var key := String(def["name"])
		values[key] = get(key)
	return values

func _apply_params(values: Dictionary) -> void:
	for def in param_defs:
		var key := String(def["name"])
		if not values.has(key):
			continue
		if def.get("type", "float") == "int":
			set(key, int(values[key]))
		else:
			set(key, float(values[key]))

func _load_params(path: String = "") -> void:
	var load_path := path
	if load_path == "":
		load_path = USER_CONFIG_PATH if FileAccess.file_exists(USER_CONFIG_PATH) else DEFAULT_CONFIG_PATH
	if not FileAccess.file_exists(load_path):
		return

	var file := FileAccess.open(load_path, FileAccess.READ)
	if file == null:
		push_warning("Could not open parameter file: %s" % load_path)
		return

	var parsed = JSON.parse_string(file.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		push_warning("Parameter file is not a JSON object: %s" % load_path)
		return

	_apply_params(parsed)

func _save_params() -> void:
	var file := FileAccess.open(USER_CONFIG_PATH, FileAccess.WRITE)
	if file == null:
		push_warning("Could not write parameter file: %s" % USER_CONFIG_PATH)
		return
	file.store_string(JSON.stringify(_params_to_dict(), "\t"))

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
		_storage_uniform(1, predicted_buffers[read_index]),
		_storage_uniform(2, density_buffer),
	], density_shader, 0)

	var forces_set := rd.uniform_set_create([
		_storage_uniform(0, params_buffer),
		_storage_uniform(1, particle_buffers[read_index]),
		_storage_uniform(2, predicted_buffers[read_index]),
		_storage_uniform(3, density_buffer),
		_storage_uniform(4, particle_buffers[write_index]),
		_storage_uniform(5, predicted_buffers[write_index]),
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
