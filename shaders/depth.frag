#version 330 core
in vec3 world_pos;

uniform vec3 lightPos;
uniform float far_plane;

void main() {
    // Store normalized linear distance from the light. Sampled back in the
    // main shader as `texture(cube, dir).r * far_plane` to recover length.
    gl_FragDepth = length(world_pos - lightPos) / far_plane;
}
