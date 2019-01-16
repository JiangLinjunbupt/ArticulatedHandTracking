#version 330 core
uniform sampler1D fix_colormap;
uniform sampler2D tex_color;

uniform float zNear;
uniform float zFar;

uniform float enable_fix_colormap;
uniform float alpha;

in vec2 fragment_uv;
in float depth;


out vec4 color;

void main()
{
	if (depth<zNear || depth>zFar) discard;

	if (enable_fix_colormap > 0)
	{
		float range = zFar - zNear;
		float w = (depth - zNear) / range;
		color = vec4(texture(fix_colormap, w).rgb, alpha);
	}
	else
	{
		color = vec4(texture(tex_color, fragment_uv).rgb, alpha);
	}

}
