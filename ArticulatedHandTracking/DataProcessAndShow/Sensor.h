#pragma once
#include "HandFinder.h"

struct DataFrame;
class Camera;

class Sensor {

public:
	int Depth_width;
	int Depth_height;
	int Color_width;
	int Color_height;

	const int BACK_BUFFER = 1;
	const int FRONT_BUFFER = 0;

	cv::Mat color_array[2];
	cv::Mat depth_array[2];
	cv::Mat full_color_array[2];

	std::vector<int> sensor_indicator_array[2];
	int num_sensor_points_array[2];


	cv::Mat sensor_silhouette_buffer;
	bool wristband_found_buffer;
	Vector3 wristband_center_buffer;
	Vector3 wristband_direction_buffer;

protected:
	bool initialized;
	bool real_color;
	Camera * camera;
public:
	HandFinder * handfinder;

public:
	Sensor(Camera* camera) : initialized(false), camera(camera) {}
	Sensor(Camera* camera, bool real_color) : initialized(false), camera(camera) {}
	virtual ~Sensor() {}
	virtual bool concurrent_fetch_streams(DataFrame &frame, HandFinder & other_handfinder) = 0;
	virtual bool run() = 0;
	virtual void start() = 0;
	virtual void stop() = 0;
private:
	virtual int initialize() = 0;
};


class SensorKinect : public Sensor
{
public:
	SensorKinect(Camera* camera);
	SensorKinect(Camera* camera, bool real_color);
	virtual ~SensorKinect();
	bool concurrent_fetch_streams(DataFrame& frame, HandFinder& other_handfinder);
	bool run();
	bool run2();
	void start();
	void stop();
private:
	int initialize();

};

class SensorRealSense : public Sensor {
public:
	SensorRealSense(Camera* camera);
	SensorRealSense(Camera* camera, bool real_color);
	virtual ~SensorRealSense();
	bool concurrent_fetch_streams(DataFrame &frame, HandFinder & other_handfinder);
	bool run();
	bool run2();
	void start(); ///< calls initialize
	void stop();
private:
	int initialize();
};

