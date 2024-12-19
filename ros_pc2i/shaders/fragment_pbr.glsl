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

const float PI = 3.14159265359;
const float REFLECTION_STRENGTH = 2.5; // 反射強度を上げる
const float FRESNEL_POWER = 7.0; // フレネル効果を強調

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), FRESNEL_POWER);
}

void main() {
    vec3 normal = normalize(v_normal);
    vec3 view_dir = normalize(camera_position - v_position.xyz);
    vec3 light_dir = normalize(light_position - v_position.xyz);
    vec3 H = normalize(view_dir + light_dir);

    // 金属のベースF0値
    vec3 F0 = mix(vec3(0.04), material.color, material.metallic);

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(normal, H, material.roughness);
    float G = GeometrySmith(normal, view_dir, light_dir, material.roughness);
    vec3 F = fresnelSchlick(max(dot(H, view_dir), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, view_dir), 0.0) * max(dot(normal, light_dir), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;
    specular *= REFLECTION_STRENGTH;

    // 光の寄与を計算
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - material.metallic;

    float NdotL = max(dot(normal, light_dir), 0.0);

    // 最終的な色を計算
    vec3 ambient = material.ambient * material.color;
    vec3 diffuse = material.diffuse * kD * material.color * NdotL;
    vec3 spec = material.specular * specular * NdotL;
    
    vec3 color = ambient + diffuse + spec;

    // HDRトーンマッピングとガンマ補正
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    float depth = v_position.z / v_position.w;
    f_color = vec4(color, 1.0);
}