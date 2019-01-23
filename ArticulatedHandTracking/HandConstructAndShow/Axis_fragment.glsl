#version 330 core

out vec4 FragColor;

in vec3 FragPos;

void main()
{
	vec3 color = vec3(1.0, 1.0, 1.0);
	float x = FragPos.x;
	float y = FragPos.y;
	float z = FragPos.z;

	if (x > 0.0)
	{
		color = vec3(1.0, 0.0, 0.0);
	}

	if (y > 0.0)
	{
		color = vec3(0.0, 1.0, 0.0);
	}

	if (z > 0.0)
	{
		color = vec3(0.0, 0.0, 1.0);
	}

	FragColor = vec4(color, 1.0);
}