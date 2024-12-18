#version 330

struct Material {
    vec3 color;
    float ambient;
    float diffuse;
    float specular;
    float shininess;
    float metallic;
    float roughness;
};

uniform Material material;
uniform vec3 light_position;
uniform vec3 ambient_light;
uniform vec3 camera_position;

in vec3 v_normal;
in vec4 v_position;

out vec4 f_color;   

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(light_position - v_position.xyz);
    vec3 view_dir = normalize(camera_position - v_position.xyz);

    vec3 ambient = material.ambient * ambient_light;

    float diffuse = max(dot(normal, light_dir), 0.0);
    vec3 diffuseColor = mix(material.color, vec3(0.0), material.metallic);

    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);
    vec3 specularColor = material.specular * spec * vec3(1.0);

    vec3 rgbColor = diffuseColor * diffuse + ambient * material.color + specularColor;

    float depth = v_position.z / v_position.w;
    f_color = vec4(rgbColor, 1.0);
}