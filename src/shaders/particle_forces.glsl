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

layout(set = 0, binding = 1, std430) readonly buffer ParticlesIn {
	vec4 particles_in[];
};

layout(set = 0, binding = 2, std430) readonly buffer Densities {
	float densities[];
};

layout(set = 0, binding = 3, std430) writeonly buffer ParticlesOut {
	vec4 particles_out[];
};

float quadratic_deriv_shape(float dist, float radius)
{
	return max(radius - dist, 0.0);
}

float density_deriv_quadratic(float radius)
{
	return 12.0 / (PI * pow(radius, 4.0));
}

float smooth_shape_r2(float dist2, float radius2)
{
	float q = max(radius2 - dist2, 0.0);
	return q * q * q;
}

float density_norm_smooth(float radius)
{
	return 4.0 / (PI * pow(radius, 8.0));
}

float pressure_from_density(float density, float target_density, float force_coef)
{
	return (density - target_density) * force_coef;
}

void resolve_border_collision(inout vec2 pos, inout vec2 vel, vec4 bounds, float damping)
{
	if (pos.x < bounds.x) {
		pos.x = bounds.x;
		vel.x *= -damping;
	} else if (pos.x > bounds.z) {
		pos.x = bounds.z;
		vel.x *= -damping;
	}

	if (pos.y < bounds.y) {
		pos.y = bounds.y;
		vel.y *= -damping;
	} else if (pos.y > bounds.w) {
		pos.y = bounds.w;
		vel.y *= -damping;
	}
}

void main()
{
	uint id = gl_GlobalInvocationID.x;
	uint count = params.counts.x;
	if (id >= count) {
		return;
	}

	float radius = params.sim0.x;
	float target_density = params.sim0.z;
	float gravity = params.sim0.w;
	float pressure_coef = params.sim1.x;
	float viscosity_coef = params.sim1.y;
	float damping = params.sim1.z;
	float border_damping = params.sim1.w;
	float dt = params.sim2.x;

	float radius2 = radius * radius;
	float deriv_scale = density_deriv_quadratic(radius);
	float viscosity_norm = density_norm_smooth(radius);

	vec2 pos = particles_in[id].xy;
	vec2 vel = particles_in[id].zw;
	float density = densities[id];
	float local_pressure = pressure_from_density(density, target_density, pressure_coef);

	vec2 pressure_force = vec2(0.0);
	vec2 viscosity_force = vec2(0.0);

	for (uint other_id = 0u; other_id < count; ++other_id) {
		if (other_id == id) {
			continue;
		}

		vec2 other_pos = particles_in[other_id].xy;
		vec2 delta = other_pos - pos;
		float dist2 = dot(delta, delta);

		if (dist2 < radius2 && dist2 > 1e-12) {
			float inv_dist = inversesqrt(dist2);
			float dist = dist2 * inv_dist;
			float other_density = max(densities[other_id], 1e-6);
			float other_pressure = pressure_from_density(other_density, target_density, pressure_coef);
			float shared_pressure = 0.5 * (local_pressure + other_pressure);
			float slope = deriv_scale * quadratic_deriv_shape(dist, radius);
			vec2 grad_w = slope * inv_dist * delta;
			pressure_force += shared_pressure * grad_w / other_density;
		}

		if (dist2 < radius2) {
			float w = smooth_shape_r2(dist2, radius2);
			viscosity_force += (particles_in[other_id].zw - vel) * w;
		}
	}

	vec2 force = vec2(0.0, gravity);
	force += pressure_force * pressure_coef * 0.01;
	force += viscosity_force * viscosity_norm * viscosity_coef * 0.0001;

	vel += force * dt;
	pos += vel * dt;
	resolve_border_collision(pos, vel, params.bounds, border_damping);
	vel *= pow(damping, dt * 60.0);

	particles_out[id] = vec4(pos, vel);
}
