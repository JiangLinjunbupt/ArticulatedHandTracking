#include"PointCloudRender.h"
#include"Sensor.h"
#include"DataFrame.h"

#include <glm/glm.hpp>
#include"OpenGLShow.h"

int main()
{

	DisPlay_ReSult::init();

	DataFrame *dataframe = new DataFrame(0);
	Camera *camera = new Camera(RealsenseSR300, 60);

	HandFinder *handfinder = new HandFinder(camera);
	SensorRealSense *mysensor = new SensorRealSense(camera);

	DepthTexture16UC1* sensor_depth_texture = new DepthTexture16UC1(camera->width(), camera->height());
	ColorTexture8UC3* sensor_color_texture = new ColorTexture8UC3(camera->width(), camera->height());
	PointCloudRender* pointcloudshader = new PointCloudRender(camera,"PointCloudRender_vertexShader.glsl", "PointCloudRender_fragmentShader.glsl", sensor_depth_texture->texid(), sensor_color_texture->texid());


	DisPlay_ReSult::dataframe = dataframe;
	DisPlay_ReSult::handfinder = handfinder;
	DisPlay_ReSult::sensor = mysensor;
	DisPlay_ReSult::sensor->start();
	DisPlay_ReSult::sensor_depth_texture = sensor_depth_texture;
	DisPlay_ReSult::sensor_color_texture = sensor_color_texture;
	DisPlay_ReSult::pointcloudshader = pointcloudshader;

	DisPlay_ReSult::Display();

	return 0;
}

