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
uniform vec3 camera_position;

in vec3 v_normal;
in vec4 v_position;

out vec4 f_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(camera_position - v_position.xyz);
    vec3 L = normalize(light_position - v_position.xyz);
    vec3 H = normalize(V + L);
    
    // Ambient
    vec3 ambient = material.ambient * material.color;
    
    // Diffuse
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = material.diffuse * diff * material.color;
    
    // Specular with metallic influence
    float spec = pow(max(dot(N, H), 0.0), material.shininess);
    vec3 specColor = mix(vec3(1.0), material.color, material.metallic);
    vec3 specular = material.specular * spec * specColor;
    
    // Metallic influence on diffuse
    float diffuseFactor = 1.0 - material.metallic * 0.5;
    diffuse *= diffuseFactor;
    
    // Additional metallic reflection
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0) * material.metallic;
    specular += fresnel * specColor * material.specular;
    
    // Combine all components
    vec3 color = ambient + diffuse + specular;
    
    // Simple tone mapping
    color = color / (color + vec3(1.0));
    
    f_color = vec4(color, 1.0);
}