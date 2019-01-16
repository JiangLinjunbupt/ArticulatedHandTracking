#include"Sensor.h"
#include "pxcsensemanager.h"
#include "pxcsession.h"
#include "pxcprojection.h"
#include"DataFrame.h"

#include<vector>

#include<thread>
#include<mutex>
#include<condition_variable>


using namespace std;



PXCImage::ImageData depth_buffer;
PXCImage::ImageData color_buffer;
PXCImage * sync_color_pxc;
PXCImage * sync_depth_pxc;

PXCSenseManager *sense_manager;
PXCProjection *projection;

std::thread sensor_thread_realsense;
std::mutex swap_mutex_realsense;
std::condition_variable condition_realsense;
bool main_released_realsense = true;
bool thread_released_realsense = false;


int sensor_frame_realsense = 0;
int tracker_frame_realsense = 0;


SensorRealSense::SensorRealSense(Camera* camera) :Sensor(camera, false)
{
	Depth_width = 640;
	Depth_height = 480;
	Color_width = 640;
	Color_height = 480;

	this->handfinder = new HandFinder(camera);
	this->real_color = false;
}

SensorRealSense::SensorRealSense(Camera* camera, bool real_color) :Sensor(camera, real_color)
{
	Depth_width = 640;
	Depth_height = 480;
	Color_width = 640;
	Color_height = 480;

	this->handfinder = new HandFinder(camera);
	this->real_color = real_color;
}

SensorRealSense::~SensorRealSense()
{
	std::cout << "~SensorRealSense() function called  " << std::endl;
	if (!initialized) return;
	delete handfinder;
}

int SensorRealSense::initialize()
{
	{
		if (sensor_indicator_array[FRONT_BUFFER].empty())
			sensor_indicator_array[FRONT_BUFFER] = std::vector<int>(upper_bound_num_sensor_points_Realsense, 0);
		if (sensor_indicator_array[BACK_BUFFER].empty())
			sensor_indicator_array[BACK_BUFFER] = std::vector<int>(upper_bound_num_sensor_points_Realsense, 0);

		if (depth_array[FRONT_BUFFER].empty())
			depth_array[FRONT_BUFFER] = cv::Mat(cv::Size(Depth_width/2, Depth_height/2), CV_16UC1, cv::Scalar(0));
		if (depth_array[BACK_BUFFER].empty())
			depth_array[BACK_BUFFER] = cv::Mat(cv::Size(Depth_width/2, Depth_height/2), CV_16UC1, cv::Scalar(0));

		if (color_array[FRONT_BUFFER].empty())
			color_array[FRONT_BUFFER] = cv::Mat(cv::Size(Depth_width/2, Depth_height/2), CV_8UC3, cv::Scalar(0, 0, 0));
		if (color_array[BACK_BUFFER].empty())
			color_array[BACK_BUFFER] = cv::Mat(cv::Size(Depth_width/2, Depth_height/2), CV_8UC3, cv::Scalar(0, 0, 0));

		if (full_color_array[FRONT_BUFFER].empty())
			full_color_array[FRONT_BUFFER] = cv::Mat(cv::Size(Color_width, Color_height), CV_8UC3, cv::Scalar(0, 0, 0));
		if (full_color_array[BACK_BUFFER].empty())
			full_color_array[BACK_BUFFER] = cv::Mat(cv::Size(Color_width, Color_height), CV_8UC3, cv::Scalar(0, 0, 0));

	}
	

	std::cout << "SensorRealSense::initialize()" << std::endl;
	sense_manager = PXCSenseManager::CreateInstance();
	if (!sense_manager) {
		wprintf_s(L"Unable to create the PXCSenseManager\n");
		return -1;
	}

	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_COLOR, Depth_width, Depth_height, camera->FPS());
	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, Depth_width, Depth_height, camera->FPS());
	sense_manager->Init();

	PXCSession *session = PXCSession::CreateInstance();
	PXCSession::ImplDesc desc, desc1;
	memset(&desc, 0, sizeof(desc));
	desc.group = PXCSession::IMPL_GROUP_SENSOR;
	desc.subgroup = PXCSession::IMPL_SUBGROUP_VIDEO_CAPTURE;
	if (session->QueryImpl(&desc, 0, &desc1) < PXC_STATUS_NO_ERROR) return -1;

	PXCCapture * capture;
	pxcStatus status = session->CreateImpl<PXCCapture>(&desc1, &capture);
	if (status != PXC_STATUS_NO_ERROR) {
		std::cerr<<"FATAL ERROR", "Intel RealSense device not plugged?\n(CreateImpl<PXCCapture> failed)\n";
		exit(0);
	}

	PXCCapture::Device* device;
	device = capture->CreateDevice(0);
	projection = device->CreateProjection();


	sensor_thread_realsense = std::thread(&SensorRealSense::run, this);
	sensor_thread_realsense.detach();

	this->initialized = true;
	std::cout << "SensorRealSense Initialization Success ! " << std::endl;

	return 1;
}

bool SensorRealSense::concurrent_fetch_streams(DataFrame& frame, HandFinder& other_handfinder)
{
	std::unique_lock<std::mutex> lock(swap_mutex_realsense);
	condition_realsense.wait(lock, [] {return thread_released_realsense; });
	main_released_realsense = false;

	frame.id = tracker_frame_realsense;
	frame.color = color_array[FRONT_BUFFER].clone();
	frame.depth = depth_array[FRONT_BUFFER].clone();
	if (real_color) frame.full_color = full_color_array[FRONT_BUFFER].clone();

	other_handfinder.sensor_silhouette = sensor_silhouette_buffer.clone();
	other_handfinder._wristband_found = wristband_found_buffer;
	other_handfinder._wband_center = wristband_center_buffer;
	other_handfinder._wband_dir = wristband_direction_buffer;

	other_handfinder.num_sensor_points = num_sensor_points_array[FRONT_BUFFER];
	for (size_t i = 0; i < other_handfinder.num_sensor_points; i++)
		other_handfinder.sensor_indicator[i] = sensor_indicator_array[FRONT_BUFFER][i];

	main_released_realsense = true;
	lock.unlock();
	condition_realsense.notify_one();

	return  true;
}

bool SensorRealSense::run()
{
	PXCCapture::Sample *sample;
	std::cout << "sensorRealsense.run()" << std::endl;
	for (;;)
	{
		if (sense_manager->AcquireFrame(true) < PXC_STATUS_NO_ERROR) continue;

		sample = sense_manager->QuerySample();

		sample->depth->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_DEPTH, &depth_buffer);
		unsigned short* data = ((unsigned short *)depth_buffer.planes[0]);
		for (int y = 0, y_sub = 0; y_sub < camera->height(); y += 2, y_sub++) {
			for (int x = 0, x_sub = 0; x_sub < camera->width(); x += 2, x_sub++) {
				if (x == 0 || y == 0) {
					depth_array[BACK_BUFFER].at<unsigned short>(y_sub, x_sub) = data[y*Depth_width + (Depth_width - x - 1)];
					continue;
				}
				std::vector<int> neighbors = {
					data[(y - 1)* Depth_width + (Depth_width - (x - 1) - 1)],
					data[(y + 0)* Depth_width + (Depth_width - (x - 1) - 1)],
					data[(y + 1)* Depth_width + (Depth_width - (x - 1) - 1)],
					data[(y - 1)* Depth_width + (Depth_width - (x + 0) - 1)],
					data[(y + 0)* Depth_width + (Depth_width - (x + 0) - 1)],
					data[(y + 1)* Depth_width + (Depth_width - (x + 0) - 1)],
					data[(y - 1)* Depth_width + (Depth_width - (x + 1) - 1)],
					data[(y + 0)* Depth_width + (Depth_width - (x + 1) - 1)],
					data[(y + 1)* Depth_width + (Depth_width - (x + 1) - 1)],
				};
				std::sort(neighbors.begin(), neighbors.end());
				depth_array[BACK_BUFFER].at<unsigned short>(y_sub, x_sub) = neighbors[4];
			}
		}

		sample->depth->ReleaseAccess(&depth_buffer);

		if (real_color) {
			sample->color->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_RGB24, &color_buffer);
			for (int y = 0; y < Depth_height; y++) {
				for (int x = 0; x < Depth_width; x++) {
					unsigned char r = color_buffer.planes[0][y * Depth_width * 3 + (Depth_width - x - 1) * 3 + 0];
					unsigned char g = color_buffer.planes[0][y * Depth_width * 3 + (Depth_width - x - 1) * 3 + 1];
					unsigned char b = color_buffer.planes[0][y * Depth_width * 3 + (Depth_width - x - 1) * 3 + 2];
					full_color_array[BACK_BUFFER].at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
				}
			}
			sample->color->ReleaseAccess(&color_buffer);
		}

		
		sync_color_pxc = projection->CreateColorImageMappedToDepth(sample->depth, sample->color);
		sync_color_pxc->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_RGB24, &color_buffer);
		for (int y = 0, y_sub = 0; y_sub < camera->height(); y += 2, y_sub++) {
			for (int x = 0, x_sub = 0; x_sub < camera->width(); x += 2, x_sub++) {
				unsigned char b = color_buffer.planes[0][y * Depth_width * 3 + (Depth_width - x - 1) * 3 + 0];
				unsigned char g = color_buffer.planes[0][y * Depth_width * 3 + (Depth_width - x - 1) * 3 + 1];
				unsigned char r = color_buffer.planes[0][y * Depth_width * 3 + (Depth_width - x - 1) * 3 + 2];
				color_array[BACK_BUFFER].at<cv::Vec3b>(y_sub, x_sub) = cv::Vec3b(b, g, r); //因为Opencv中通道的排列是BGR, 因此，我们需要吧blue通道放在第一个上面，green通道放在第二个上面，red通道放在第三个上面，放错会导致颜色出错
			}
		}
		sync_color_pxc->ReleaseAccess(&color_buffer);
		sync_color_pxc->Release();

		sense_manager->ReleaseFrame();

		handfinder->binary_classification(depth_array[BACK_BUFFER], color_array[BACK_BUFFER]);
		num_sensor_points_array[BACK_BUFFER] = 0;
		int count = 0;
		for (int row = 0; row < handfinder->sensor_silhouette.rows; ++row) {
			for (int col = 0; col < handfinder->sensor_silhouette.cols; ++col) {
				if (handfinder->sensor_silhouette.at<uchar>(row, col) != 255) continue;
				if (count % 2 == 0) {
					sensor_indicator_array[BACK_BUFFER][num_sensor_points_array[BACK_BUFFER]] = row * handfinder->sensor_silhouette.cols + col;
					num_sensor_points_array[BACK_BUFFER]++;
				} 
				count++;
			}
		}

		// Lock the mutex and swap the buffers
		{
			std::unique_lock<std::mutex> lock(swap_mutex_realsense);
			condition_realsense.wait(lock, [] {return main_released_realsense; });
			thread_released_realsense = false;

			depth_array[FRONT_BUFFER] = depth_array[BACK_BUFFER].clone();
			color_array[FRONT_BUFFER] = color_array[BACK_BUFFER].clone();
			if (real_color) full_color_array[FRONT_BUFFER] = full_color_array[BACK_BUFFER].clone();


			sensor_silhouette_buffer = handfinder->sensor_silhouette.clone();
			wristband_found_buffer = handfinder->_wristband_found;
			wristband_center_buffer = handfinder->_wband_center;
			wristband_direction_buffer = handfinder->_wband_dir;

			std::copy(sensor_indicator_array[BACK_BUFFER].begin(),
				sensor_indicator_array[BACK_BUFFER].begin() + num_sensor_points_array[BACK_BUFFER], sensor_indicator_array[FRONT_BUFFER].begin());
			num_sensor_points_array[FRONT_BUFFER] = num_sensor_points_array[BACK_BUFFER];

			tracker_frame_realsense = sensor_frame_realsense;
			sensor_frame_realsense++;

			thread_released_realsense = true;
			lock.unlock();
			condition_realsense.notify_one();
		}
	}

	cout << "for no reason the run() breakdown!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
}

void SensorRealSense::start()
{
	if (!initialized) this->initialize();
}

void SensorRealSense::stop() { ; }