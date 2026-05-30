#version 330 core

in vec3 frag_pos;
in vec3 frag_normal;

out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform vec3 cameraPos;

void main() {
    float ambientStrength = 0.15;
    vec3 ambient = ambientStrength * lightColor;

    vec3 normal = normalize(frag_normal);
    vec3 lightDir = normalize(lightPos - frag_pos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(cameraPos - frag_pos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    fragColor = vec4(result, 1.0);
}
