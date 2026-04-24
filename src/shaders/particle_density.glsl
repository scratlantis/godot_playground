#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const float PI = 3.14159265358979323846;

layout(set = 0, binding = 0, std430) readonly buffer Params {
	vec4 min_range;
	vec4 max_range;
	vec4 cursor;
	uvec4 mouse;
	vec4 scalars0;
	vec4 scalars1;
	vec4 scalars2;
	uvec4 counts;
} params;

layout(set = 0, binding = 1, std430) readonly buffer PredictedIn {
	vec4 predicted_in[];
};

layout(set = 0, binding = 2, std430) writeonly buffer DensityOut {
	float density_out[];
};

float quadratic_shape_r2(float dist2, float radius)
{
	float dist = sqrt(dist2);
	float q = max(radius - dist, 0.0);
	return q * q;
}

float density_norm_quadratic(float radius)
{
	return 15.0 / (PI * pow(radius, 5.0));
}

void main()
{
	uint id = gl_GlobalInvocationID.x;
	uint count = params.counts.x;
	if (id >= count) {
		return;
	}

	float radius = params.scalars0.x;
	float radius2 = radius * radius;
	vec3 local_pred = predicted_in[id].xyz;
	float density = 0.0;

	for (uint other_id = 0u; other_id < count; ++other_id) {
		vec3 other_pos = predicted_in[other_id].xyz;
		float dist2 = dot(local_pred - other_pos, local_pred - other_pos);
		if (dist2 < radius2) {
			density += quadratic_shape_r2(dist2, radius);
		}
	}

	density += quadratic_shape_r2(0.0, radius);
	density *= density_norm_quadratic(radius) * params.scalars2.x * 0.0001;
	density_out[id] = density;
}
