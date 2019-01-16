#version 330

//这里使用usampler很重要,
//因为使用usampler对应着TextureDepth16UC1中的的这个函数glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, data.data())中的GL_R16UI和后续的GL_UNSIGNED_SHORT
//由于深度图的的深度值，是unsigned short类型的整形，对应着16比特的无符号整形
//再由33行的texture(tex_depth, uv).r读出来的是一个uint的整形深度值，uint类型再glsl中可以通过隐式转换变成float类型的，但是float不能转换成uint
//具体可以参考：https://www.opengl.org/discussion_boards/showthread.php/199831-How-to-get-integer-textures 和 http://www.selfgleam.com/texture-type.html
uniform usampler2D tex_depth;  

uniform float CameraCenterX;
uniform float CameraCenterY;
uniform float focal_length_x;
uniform float focal_length_y;
uniform float zNear;
uniform float zFar;

uniform mat4 view;
uniform mat4 projection;

layout(location = 0) in vec2 vpoint;
layout(location = 1) in vec2 uv;


out vec2 fragment_uv;
out float depth;



void main()
{
	fragment_uv = uv;

	depth = float(texture(tex_depth, uv).r);


	float x = (vpoint[0] - CameraCenterX) / focal_length_x;
	float y = (vpoint[1] - CameraCenterY) / focal_length_y;

	vec3 p_world = vec3(x*depth, -y*depth, depth);  //加负号是因为转换得到的点云是上下颠倒的，得加负号转过来

	gl_Position = projection*view*vec4(p_world,1);


	gl_PointSize = 2.5;

}