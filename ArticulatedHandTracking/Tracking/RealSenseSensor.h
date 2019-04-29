#pragma once
#include"DataFrame.h"
#include"DistanceTransform.h"
#include"Camera.h"

class RealSenseSensor
{
public:
	const int BACK_BUFFER = 1;
	const int FRONT_BUFFER = 0;

	cv::Mat depth_array[2];
	cv::Mat hand_BinaryMap[2];

	int *idxs_image_BACK_BUFFER;
	int *idxs_image_FRONT_BUFFER;

	Eigen::RowVector3f palm_center[2];

	pcl::PointCloud<pcl::PointXYZ> handPointCloud[2];

	DistanceTransform distance_transform;

protected:
	bool initialized;
	Camera* camera;

public:
	RealSenseSensor(Camera* _camera);
	~RealSenseSensor();
	bool concurrent_fetch_streams(DataFrame& dataframe);
	bool run();
	bool start();

private:
	bool initialize();
};