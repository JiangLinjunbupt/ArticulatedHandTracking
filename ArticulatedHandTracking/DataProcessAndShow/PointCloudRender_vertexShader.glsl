#version 330

//����ʹ��usampler����Ҫ,
//��Ϊʹ��usampler��Ӧ��TextureDepth16UC1�еĵ��������glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, data.data())�е�GL_R16UI�ͺ�����GL_UNSIGNED_SHORT
//�������ͼ�ĵ����ֵ����unsigned short���͵����Σ���Ӧ��16���ص��޷�������
//����33�е�texture(tex_depth, uv).r����������һ��uint���������ֵ��uint������glsl�п���ͨ����ʽת�����float���͵ģ�����float����ת����uint
//������Բο���https://www.opengl.org/discussion_boards/showthread.php/199831-How-to-get-integer-textures �� http://www.selfgleam.com/texture-type.html
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

	vec3 p_world = vec3(x*depth, -y*depth, depth);  //�Ӹ�������Ϊת���õ��ĵ��������µߵ��ģ��üӸ���ת����

	gl_Position = projection*view*vec4(p_world,1);


	gl_PointSize = 2.5;

}