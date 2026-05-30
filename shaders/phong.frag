#version 330 core

// Bump this and the unrolled `if (i < numLights)` block in main() if you
// need more simultaneous lights. GLSL 330 does not allow indexing
// samplerCube arrays with a dynamic int, so the unroll is required.
#define MAX_LIGHTS 4

in vec3 frag_pos;
in vec3 frag_normal;

out vec4 fragColor;

struct PointLight {
    vec3 position;
    vec3 color;         // already pre-multiplied by intensity
    float range;        // attenuation falls to 0 at this distance
    float far_plane;    // shadow projection far plane (== range)
    float bias;         // shadow comparison bias (world units)
    int cast_shadows;   // 0/1
};

uniform PointLight pointLights[MAX_LIGHTS];
uniform samplerCube shadowMaps[MAX_LIGHTS];
uniform int numLights;

uniform vec3 cameraPos;
uniform vec3 objectColor;
uniform vec3 ambientColor;  // global ambient term (e.g. vec3(0.05))

// 9 offset directions for cubemap PCF. Combined with the disk-radius
// scaling below, this gives "3x3"-style soft shadow edges around the
// sample direction. Bump to 20 (LearnOpenGL set) if you want smoother
// penumbras at the cost of more texture fetches.
const vec3 SAMPLE_OFFSETS[9] = vec3[](
    vec3( 0,  0,  0),
    vec3( 1,  1,  0), vec3(-1, -1,  0), vec3( 1, -1,  0), vec3(-1,  1,  0),
    vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1)
);

float shadow_factor(samplerCube shadowMap, vec3 light_to_frag,
                    float dist, float far_plane, float bias) {
    // Disk radius scales with viewer distance so close objects keep crisp
    // edges and far objects get softer penumbras without obvious aliasing.
    float view_distance = length(cameraPos - frag_pos);
    float disk_radius = (1.0 + (view_distance / far_plane)) / 50.0;

    float shadow = 0.0;
    for (int i = 0; i < 9; ++i) {
        vec3 sample_dir = light_to_frag + SAMPLE_OFFSETS[i] * disk_radius;
        float closest = texture(shadowMap, sample_dir).r * far_plane;
        if (dist - bias > closest) shadow += 1.0;
    }
    return shadow / 9.0;
}

vec3 compute_point_light(PointLight light, samplerCube shadowMap,
                          vec3 normal, vec3 view_dir) {
    vec3 light_to_frag = frag_pos - light.position;
    float dist = length(light_to_frag);
    if (dist > light.range) return vec3(0.0);

    vec3 light_dir = -light_to_frag / dist;

    // Diffuse (Lambert)
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * light.color;

    // Specular (Blinn-Phong)
    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway), 0.0), 64.0);
    vec3 specular = 0.5 * spec * light.color;

    // Smooth quadratic falloff to zero at `range`.
    float t = clamp(dist / light.range, 0.0, 1.0);
    float attenuation = (1.0 - t) * (1.0 - t);

    float shadow = 0.0;
    if (light.cast_shadows == 1 && diff > 0.0) {
        shadow = shadow_factor(shadowMap, light_to_frag, dist,
                               light.far_plane, light.bias);
    }

    return attenuation * (1.0 - shadow) * (diffuse + specular);
}

void main() {
    vec3 normal = normalize(frag_normal);
    vec3 view_dir = normalize(cameraPos - frag_pos);

    vec3 lighting = ambientColor;

    // Manual unroll — samplerCube arrays can't be indexed by a dynamic int
    // in GLSL 330.
    if (0 < numLights) lighting += compute_point_light(pointLights[0], shadowMaps[0], normal, view_dir);
    if (1 < numLights) lighting += compute_point_light(pointLights[1], shadowMaps[1], normal, view_dir);
    if (2 < numLights) lighting += compute_point_light(pointLights[2], shadowMaps[2], normal, view_dir);
    if (3 < numLights) lighting += compute_point_light(pointLights[3], shadowMaps[3], normal, view_dir);

    fragColor = vec4(lighting * objectColor, 1.0);
}
