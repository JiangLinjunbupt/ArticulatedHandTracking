#pragma once
#include"DataSet.h"
#include"HandModel.h"


class Evaluations
{
private:
	DistanceTransform distance_transform;
	Camera* camera;
	HandModel* handmodel;

public:
	Evaluations(Camera* _camera, HandModel* _handmodel);
	~Evaluations() {};

	void Compute_Metric(DataFrame* dataframe, int currentFrameNUM);
	void RecordData();

private:
	float Compute_3D_Metric(const cv::Mat& rendered_model, const cv::Mat& rendered_sihouette, 
		const cv::Mat& sensor_depth, const cv::Mat& sensor_sihouette);
	float Compute_2D_Metric(const cv::Mat& rendered_model, const cv::Mat& rendered_sihouette,
		const cv::Mat& sensor_depth, const cv::Mat& sensor_sihouette);
};