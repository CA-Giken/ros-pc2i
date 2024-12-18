#version 330

uniform mat4 mvp;
uniform mat4 model_matrix;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_normal;
out vec4 v_position;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);

    v_position = model_matrix * vec4(in_position, 1.0);
    v_normal = mat3(model_matrix) * in_normal;
}