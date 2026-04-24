extends Node2D

const GRID_SPACING := 32.0
const DOT_COUNT := 96

var elapsed := 0.0
var points: Array[Vector2] = []

func _ready() -> void:
	DisplayServer.window_set_title("Particle Sim 2D")
	_seed_points()

func _process(delta: float) -> void:
	elapsed += delta
	queue_redraw()

func _draw() -> void:
	var rect := get_viewport_rect()
	draw_rect(rect, Color(0.035, 0.039, 0.047))
	_draw_grid(rect)
	_draw_bounds(rect)
	_draw_points(rect)

func _seed_points() -> void:
	points.clear()
	var rng := RandomNumberGenerator.new()
	rng.seed = 39
	for i in DOT_COUNT:
		points.append(Vector2(rng.randf(), rng.randf()))

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
	for i in points.size():
		var unit := points[i]
		var phase := elapsed * 0.9 + float(i) * 0.37
		var wobble := Vector2(cos(phase * 1.3), sin(phase * 1.7)) * 7.0
		var pos := area.position + unit * area.size + wobble
		var radius := 2.5 + 1.5 * sin(phase)
		var color := Color(0.28, 0.76, 0.92, 0.82)
		draw_circle(pos, radius, color)
