#pragma once
#include"Types.h"
#include "opencv2/core/core.hpp" /// cv::Mat

struct DataFrame
{
	int width;
	int height;

	cv::Mat original_DepthMap;
	cv::Mat hand_BinaryMap;
	cv::Mat handmodel_visibleMap;

	int *idxs_image;

	Eigen::RowVector3f palm_Center;

	pcl::PointCloud<pcl::PointXYZ> handPointCloud;

	DataFrame() {}
	~DataFrame()
	{
		delete idxs_image;
	}

	void Init(int W, int H)
	{
		width = W / 2;
		height = H / 2;

		original_DepthMap = cv::Mat(cv::Size(width, height), CV_16UC1, cv::Scalar(0));
		hand_BinaryMap = cv::Mat(cv::Size(width, height), CV_16UC1, cv::Scalar(0));
		handmodel_visibleMap = cv::Mat(cv::Size(width, width), CV_32SC1, cv::Scalar(10000));

		idxs_image = new int[width*height]();
		palm_Center = Eigen::RowVector3f::Zero();
		handPointCloud.points.clear();
	}
};