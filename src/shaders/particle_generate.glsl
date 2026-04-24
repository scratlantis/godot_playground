#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Particle {
	vec4 pos_state;
	vec4 vel_pad;
};

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

layout(set = 0, binding = 1, std430) writeonly buffer Particles {
	Particle particles[];
};

layout(set = 0, binding = 2, std430) writeonly buffer PredictedPositions {
	vec4 predicted_positions[];
};

uint hash_u32(uint x)
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

float rand01(inout uint seed)
{
	seed = hash_u32(seed);
	return float(seed & 0x00ffffffu) / float(0x01000000u);
}

void main()
{
	uint id = gl_GlobalInvocationID.x;
	uint count = params.counts.x;
	if (id >= count) {
		return;
	}

	float radius = params.scalars0.x;
	uint seed = params.counts.w * count + id + 46732468u;
	vec3 rng = vec3(rand01(seed), rand01(seed), rand01(seed));
	vec3 scale = params.max_range.xyz - params.min_range.xyz - vec3(2.0 * radius);
	vec3 offset = params.min_range.xyz + vec3(radius);

	vec3 pos = rng * scale * 0.5 + scale * 0.25 + offset;
	particles[id].pos_state = vec4(pos, 0.0);
	particles[id].vel_pad = vec4(0.0);
	predicted_positions[id] = vec4(pos, 0.0);
}
