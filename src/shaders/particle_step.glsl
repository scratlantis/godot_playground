#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint PARTICLE_STATE_IN_BOX_BIT = 0x1u;
const uint PARTICLE_EARTH_GRAVITY_BIT = 0x1u;
const uint PARTICLE_GRAVITY_BIT = 0x2u;
const uint PARTICLE_BOX_BIT = 0x4u;
const float PI = 3.14159265358979323846;

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

layout(set = 0, binding = 1, std430) readonly buffer ParticlesIn {
	Particle particles_in[];
};

layout(set = 0, binding = 2, std430) readonly buffer PredictedIn {
	vec4 predicted_in[];
};

layout(set = 0, binding = 3, std430) readonly buffer DensityIn {
	float density_in[];
};

layout(set = 0, binding = 4, std430) writeonly buffer ParticlesOut {
	Particle particles_out[];
};

layout(set = 0, binding = 5, std430) writeonly buffer PredictedOut {
	vec4 predicted_out[];
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

float quadratic_shape_r2(float dist2, float radius)
{
	float dist = sqrt(dist2);
	float q = max(radius - dist, 0.0);
	return q * q;
}

float quadratic_deriv_shape(float dist, float radius)
{
	return max(radius - dist, 0.0);
}

float smooth_shape_r2(float dist2, float radius2)
{
	float q = max(radius2 - dist2, 0.0);
	return q * q * q;
}

float density_norm_quadratic(float radius)
{
	return 15.0 / (PI * pow(radius, 5.0));
}

float density_deriv_quadratic(float radius)
{
	return 30.0 / (PI * pow(radius, 5.0));
}

float density_norm_smooth(float radius)
{
	return 315.0 / (64.0 * PI * pow(radius, 9.0));
}

float pressure_from_density(float density, float target_density, float force_coef)
{
	return (density - target_density) * force_coef;
}

vec3 cursor_force(vec3 cursor_pos, vec3 pos, float radius)
{
	vec3 dir = cursor_pos - pos;
	float dist = length(dir);
	if (dist <= 0.00001 || dist >= radius) {
		return vec3(0.0);
	}

	dir /= dist;
	float q = radius * radius - dist * dist;
	float weight = q * q * q * density_norm_smooth(radius) * 0.001;
	return dir * weight * 100.0;
}

bool is_in_box(vec3 pos, vec3 min_range, vec3 max_range)
{
	return all(greaterThanEqual(pos, min_range)) && all(lessThanEqual(pos, max_range));
}

void resolve_border_collision(inout vec3 pos, inout vec3 vel, vec3 min_range, vec3 max_range, float damping)
{
	if (pos.x < min_range.x) {
		pos.x = min_range.x;
		vel.x *= -damping;
	} else if (pos.x > max_range.x) {
		pos.x = max_range.x;
		vel.x *= -damping;
	}

	if (pos.y < min_range.y) {
		pos.y = min_range.y;
		vel.y *= -damping;
	} else if (pos.y > max_range.y) {
		pos.y = max_range.y;
		vel.y *= -damping;
	}

	if (pos.z < min_range.z) {
		pos.z = min_range.z;
		vel.z *= -damping;
	} else if (pos.z > max_range.z) {
		pos.z = max_range.z;
		vel.z *= -damping;
	}
}

void main()
{
	uint id = gl_GlobalInvocationID.x;
	uint count = params.counts.x;
	if (id >= count) {
		return;
	}

	float radius = params.scalars0.x;
	float dt_ms = params.scalars0.y;
	float damping = params.scalars0.z;
	float damping_border = params.scalars0.w;
	float gravity = params.scalars1.x;
	float particle_gravity_coef = params.scalars1.y;
	float pressure_coef = params.scalars1.z;
	float viscosity_coef = params.scalars1.w;
	float target_density = params.scalars2.y;
	uint flags = params.counts.z;

	float radius2 = radius * radius;
	float density_deriv = density_deriv_quadratic(radius);
	float viscosity_norm = density_norm_smooth(radius);

	vec3 local_pred = predicted_in[id].xyz;
	float density = density_in[id];

	float local_pressure = pressure_from_density(density, target_density, pressure_coef);
	vec3 force = vec3(0.0);
	vec3 pressure_force = vec3(0.0);
	vec3 viscosity_force = vec3(0.0);
	vec3 local_vel = particles_in[id].vel_pad.xyz;

	for (uint other_id = 0u; other_id < count; ++other_id) {
		if (other_id == id) {
			continue;
		}

		vec3 other_pos = predicted_in[other_id].xyz;
		vec3 delta = other_pos - local_pred;
		float dist2 = dot(delta, delta);

		if (dist2 < radius2 && dist2 > 1e-12) {
			float inv_dist = inversesqrt(dist2);
			float dist = dist2 * inv_dist;
			float other_density = max(density_in[other_id], 1e-6);
			float other_pressure = pressure_from_density(other_density, target_density, pressure_coef);
			float shared_pressure = 0.5 * (local_pressure + other_pressure);
			float slope = density_deriv * quadratic_deriv_shape(dist, radius);
			vec3 grad_w = slope * inv_dist * delta;
			pressure_force += shared_pressure * grad_w / other_density;
		}

		if (dist2 < radius2) {
			float w = smooth_shape_r2(dist2, radius2);
			viscosity_force += (particles_in[other_id].vel_pad.xyz - local_vel) * w;
		}
	}

	force += pressure_force * pressure_coef * 0.01;
	force += viscosity_force * viscosity_norm * viscosity_coef * 0.0001;

	if ((flags & PARTICLE_EARTH_GRAVITY_BIT) != 0u) {
		force += vec3(0.0, -gravity, 0.0);
	}

	if ((flags & PARTICLE_GRAVITY_BIT) != 0u) {
		uint seed = id * 74775u + params.counts.y * 289363u + 100423u + 13789u;
		vec3 pos = particles_in[id].pos_state.xyz;
		for (uint i = 0u; i < 10u; ++i) {
			seed = hash_u32(seed);
			uint random_id = seed % count;
			vec3 dir = particles_in[random_id].pos_state.xyz - pos;
			float dist2 = dot(dir, dir);
			if (dist2 > 0.001) {
				force += 0.01 * normalize(dir) * particle_gravity_coef / dist2;
			}
		}
	}

	if (params.mouse.x == 1u) {
		force += cursor_force(params.cursor.xyz, particles_in[id].pos_state.xyz, params.cursor.w) * params.scalars2.z;
	}
	if (params.mouse.y == 1u) {
		force -= cursor_force(params.cursor.xyz, particles_in[id].pos_state.xyz, params.cursor.w) * params.scalars2.z;
	}

	Particle elem = particles_in[id];
	vec3 pos = elem.pos_state.xyz;
	vec3 vel = elem.vel_pad.xyz;
	float time = dt_ms * 0.001;

	vel += force * time;
	pos += vel * time;

	uint state = floatBitsToUint(elem.pos_state.w);
	if ((flags & PARTICLE_BOX_BIT) != 0u) {
		if ((state & PARTICLE_STATE_IN_BOX_BIT) != 0u || is_in_box(pos, params.min_range.xyz, params.max_range.xyz)) {
			state |= PARTICLE_STATE_IN_BOX_BIT;
			resolve_border_collision(pos, vel, params.min_range.xyz, params.max_range.xyz, damping_border);
		}
	} else {
		state &= ~PARTICLE_STATE_IN_BOX_BIT;
	}

	vel *= pow(damping, time * 60.0);

	vec3 pred_vel = vel;
	pred_vel.y -= gravity * time;
	vec3 pred_pos = pos + pred_vel * time;

	particles_out[id].pos_state = vec4(pos, uintBitsToFloat(state));
	particles_out[id].vel_pad = vec4(vel, 0.0);
	predicted_out[id] = vec4(pred_pos, 0.0);
}
