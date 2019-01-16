#pragma once
#include "Types.h"
#include "opencv_wrapper.h"
#define CHECK_NOTNULL(val) if(val==NULL){ std::cout << "!!!CHECK_NOTNULL: " << __FILE__ << " " << __LINE__ << std::endl; exit(0); }
#include"Camera.h"
#include <numeric> ///< std::iota
#include<iostream>
#include <fstream> ///< ifstream
using namespace std;

class HandFinder {
private:
	Camera* camera;
public:
	HandFinder(Camera * camera);
	~HandFinder() {
		delete[] sensor_indicator;
	}

	/// @{ Settings
public:
	struct Settings {
		bool show_hand = false;
		bool show_wband = false;
		float depth_range = 150;
		float wband_size = 30;

		//ºìÉ«Íó´ø
		cv::Scalar hsv_min1 = cv::Scalar(0, 80, 20); 
		cv::Scalar hsv_max1 = cv::Scalar(8, 255, 255);

		cv::Scalar hsv_min2 = cv::Scalar(120, 80, 20); 
		cv::Scalar hsv_max2 = cv::Scalar(180, 255, 255);

	} _settings;
	Settings*const settings = &_settings;
	/// @}

public:
	bool _has_useful_data = false;
	bool _wristband_found;
	Eigen::Vector3f _wband_center;
	Eigen::Vector3f _wband_dir;
public:
	cv::Mat sensor_silhouette; ///< created by binary_classifier
	cv::Mat mask_wristband; ///< created by binary_classifier, not used anywhere else
	cv::Mat mask_wristband2;
	int * sensor_indicator;
	int num_sensor_points;

public:
	bool has_useful_data() { return _has_useful_data; }
	bool wristband_found() { return _wristband_found; }
	Eigen::Vector3f wristband_center() { return _wband_center; }
	Eigen::Vector3f wristband_direction() { return _wband_dir; }
	void wristband_direction_flip() { _wband_dir = -_wband_dir; }
public:
	void binary_classification(cv::Mat& depth, cv::Mat& color);
};
