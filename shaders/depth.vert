#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;  // unused, but matches scene VAO layout

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 world_pos;

void main() {
    world_pos = vec3(model * vec4(position, 1.0));
    gl_Position = projection * view * model * vec4(position, 1.0);
}
