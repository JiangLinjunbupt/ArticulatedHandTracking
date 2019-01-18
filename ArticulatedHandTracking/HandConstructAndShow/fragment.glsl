#version 330 core

out vec4 FragColor;

struct Material {
	sampler2D Material_texture;
	float shininess;
};

struct DirLight {
	vec3 direction;

	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};


in vec3 FragPos;
in vec3 Normal;

uniform vec3 viewPos;
uniform DirLight dirLight;
uniform Material material;


void main()
{
	vec3 norm = normalize(Normal);
	vec3 viewDir = normalize(viewPos - FragPos);


	vec3 lightDir = normalize(-dirLight.direction);
	// diffuse shading
	float diff = max(dot(norm, lightDir), 0.0);
	// specular shading
	vec3 reflectDir = reflect(-lightDir, norm);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
	// combine results
	vec3 ambient = dirLight.ambient * vec3(texture(material.Material_texture, vec2(0.5,0.5)));
	vec3 diffuse = dirLight.diffuse * diff * vec3(texture(material.Material_texture, vec2(0.5, 0.5)));
	vec3 specular = dirLight.specular * spec * vec3(texture(material.Material_texture, vec2(0.5, 0.5)));

	vec3 color = ambient + diffuse + specular;

	FragColor = vec4(color,1.0);

}