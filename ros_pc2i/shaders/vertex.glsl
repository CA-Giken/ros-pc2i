#version 330

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_normal;
out vec4 v_position;

void main() {
    v_normal = mat3(model) * in_normal;
    vec4 world_pos = model * vec4(in_position, 1.0);
    v_position = world_pos;
    gl_Position = projection * view * world_pos;
}