#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const float PI = 3.14159265358979323846;

layout(set = 0, binding = 0, std430) readonly buffer Params {
	vec4 bounds;
	vec4 sim0;
	vec4 sim1;
	vec4 sim2;
	uvec4 counts;
} params;

layout(set = 0, binding = 1, std430) readonly buffer Particles {
	vec4 particles[];
};

layout(set = 0, binding = 2, std430) writeonly buffer Densities {
	float densities[];
};

float quadratic_shape_r2(float dist2, float radius)
{
	float dist = sqrt(dist2);
	float q = max(radius - dist, 0.0);
	return q * q;
}

float density_norm_quadratic(float radius)
{
	return 6.0 / (PI * pow(radius, 4.0));
}

void main()
{
	uint id = gl_GlobalInvocationID.x;
	uint count = params.counts.x;
	if (id >= count) {
		return;
	}

	float radius = params.sim0.x;
	float radius2 = radius * radius;
	float density = 0.0;
	vec2 pos = particles[id].xy;

	for (uint other_id = 0u; other_id < count; ++other_id) {
		if (other_id == id) {
			continue;
		}

		vec2 delta = pos - particles[other_id].xy;
		float dist2 = dot(delta, delta);
		if (dist2 < radius2) {
			density += quadratic_shape_r2(dist2, radius);
		}
	}

	density += quadratic_shape_r2(0.0, radius);
	density *= density_norm_quadratic(radius) * params.sim0.y * 0.0001;
	densities[id] = density;
}
