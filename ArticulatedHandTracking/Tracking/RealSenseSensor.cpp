/*
程序中所用的随机森林来自于github ：https://github.com/edoRemelli/hand-seg-rdf
没有详细的引用说明，这里直接给出url
*/

#include"RealSenseSensor.h"

#include <fertilized/fertilized.h>
#include "fertilized/ndarray.h"
#include "fertilized/global.h"


#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <fstream>
#include <algorithm>

// feature extraction params
# define N_FEAT 8.0
// how many points? n_features = (2*N_FEAT +1)*(2*N_FEAT +1)
# define N_FEAT_PER_NODE 100.0
// how far away should we shoot when computing features - should be proportional to N_FEAT 
# define DELTA 12000.0


// resolution on which process frame
# define SRC_COLS 80
# define SRC_ROWS 60
// htrack's resolution
# define OSRC_COLS 320
# define OSRC_ROWS 240
// depth of 0-depth pixels
# define BACKGROUND_DEPTH 3000.0
// rdf parameters 
# define THRESHOLD 0.7
# define N_THREADS 1
// post processing parameters
# define DILATION_SIZE 9
# define KERNEL_SIZE 3
# define GET_CLOSER_TO_SENSOR 700


using namespace fertilized;
using namespace std;
using namespace std::chrono;


#include "pxcsensemanager.h"
#include "pxcsession.h"
#include "pxcprojection.h"

#include<thread>
#include<mutex>
#include<condition_variable>

// sensor resolution
int D_width = 640;
int D_height = 480;

PXCSenseManager *sense_manager;

std::thread sensor_thread_realsense;
std::mutex swap_mutex_realsense;
std::condition_variable condition_realsense;
bool main_released_realsense = true;
bool thread_released_realsense = true;


int sensor_frame_realsense = 0;
int tracker_frame_realsense = 0;


int n_features = (2 * N_FEAT + 1)*(2 * N_FEAT + 1);

cv::Mat src_X;
cv::Mat src_Y;
cv::Mat mask;


auto soil = Soil<float, float, unsigned int, Result_Types::probabilities>();
auto forest = soil.ForestFromFile("E:\\githubProject\\hand-seg-rdf\\examples\\c++\\ff_handsegmentation.ff");


RealSenseSensor::RealSenseSensor(Camera* _camera)
{
	camera = _camera;
	initialized = false;

	distance_transform.init(OSRC_COLS, OSRC_ROWS);
}

RealSenseSensor::~RealSenseSensor()
{
	std::cout << "~RealSenseSensor() function called  " << std::endl;
	if (!initialized) return;
}

bool RealSenseSensor::initialize()
{
	{
		depth_array[FRONT_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_16UC1, cv::Scalar(0));
		depth_array[BACK_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_16UC1, cv::Scalar(0));

		hand_BinaryMap[FRONT_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_8UC1, cv::Scalar(0));
		hand_BinaryMap[BACK_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_8UC1, cv::Scalar(0));

		idxs_image_FRONT_BUFFER = new int[OSRC_COLS * OSRC_ROWS];
		idxs_image_BACK_BUFFER = new int[OSRC_COLS * OSRC_ROWS];

		palm_center[FRONT_BUFFER] = Eigen::RowVector3f::Zero();
		palm_center[BACK_BUFFER] = Eigen::RowVector3f::Zero();

		handPointCloud[FRONT_BUFFER].points.clear();
		handPointCloud[BACK_BUFFER].points.clear();
		handPointCloud[FRONT_BUFFER].points.reserve(OSRC_COLS*OSRC_ROWS);
		handPointCloud[BACK_BUFFER].points.reserve(OSRC_COLS*OSRC_ROWS);

	}

	std::cout << "RealSenseSensor::initialize()" << std::endl;
	sense_manager = PXCSenseManager::CreateInstance();
	if (!sense_manager) {
		wprintf_s(L"Unable to create the PXCSenseManager\n");
		return -1;
	}

	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_COLOR, D_width, D_height, 60);
	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, D_width, D_height, 60);
	sense_manager->Init();


	PXCSession *session = PXCSession::CreateInstance();
	PXCSession::ImplDesc desc, desc1;
	memset(&desc, 0, sizeof(desc));
	desc.group = PXCSession::IMPL_GROUP_SENSOR;
	desc.subgroup = PXCSession::IMPL_SUBGROUP_VIDEO_CAPTURE;
	if (session->QueryImpl(&desc, 0, &desc1) < PXC_STATUS_NO_ERROR) return false;

	PXCCapture * capture;
	pxcStatus status = session->CreateImpl<PXCCapture>(&desc1, &capture);
	if (status != PXC_STATUS_NO_ERROR) {
		std::cerr << "FATAL ERROR", "Intel RealSense device not plugged?\n(CreateImpl<PXCCapture> failed)\n";
		exit(0);
	}

	PXCCapture::Device* device;
	device = capture->CreateDevice(0);

	sensor_thread_realsense = std::thread(&RealSenseSensor::run, this);
	sensor_thread_realsense.detach();

	this->initialized = true;
	std::cout << "SensorRealSense Initialization Success ! " << std::endl;

	return true;
}


bool RealSenseSensor::concurrent_fetch_streams(DataFrame& dataframe)
{
	std::unique_lock<std::mutex> lock(swap_mutex_realsense);
	condition_realsense.wait(lock, [] {return thread_released_realsense; });
	main_released_realsense = false;

	dataframe.original_DepthMap = depth_array[FRONT_BUFFER].clone();
	dataframe.hand_BinaryMap = hand_BinaryMap[FRONT_BUFFER].clone();
	std::copy(idxs_image_FRONT_BUFFER, idxs_image_FRONT_BUFFER + OSRC_COLS * OSRC_ROWS, dataframe.idxs_image);

	dataframe.palm_Center = palm_center[FRONT_BUFFER];
	dataframe.handPointCloud.points.assign(handPointCloud[FRONT_BUFFER].points.begin(), handPointCloud[FRONT_BUFFER].points.end());

	main_released_realsense = true;
	lock.unlock();
	condition_realsense.notify_all();
	return  true;
}

bool RealSenseSensor::run()
{
	PXCImage::ImageData depth_buffer;
	PXCCapture::Sample *sample;

	// standard downscaling used by htrack
	int downsampling_factor = 2;
	// downscaling for processing
	int ds = 4;
	cv::Mat sensor_depth = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_16UC1, cv::Scalar(0));
	cv::Mat sensor_depth_ds = cv::Mat(cv::Size(D_width / (downsampling_factor*ds), D_height / (downsampling_factor * ds)), CV_16UC1, cv::Scalar(0));

	// TO DO: fix this crap at some point
	std::vector<cv::Point> locations;
	vector<vector< cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;

	std::cout << "sensorRealsense.run()" << std::endl;

	while (true)
	{
		if (sense_manager->AcquireFrame(true) < PXC_STATUS_NO_ERROR)
		{
			cout << "AcquireFrame fail -----> continue..." << endl;
			continue;
		}

		sample = sense_manager->QuerySample();

		sample->depth->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_DEPTH, &depth_buffer);
		unsigned short* data = ((unsigned short *)depth_buffer.planes[0]);

		/// we downsample the sensor image twice for hand-tracking system, feel free to experiment with the full image as well
		for (int x = 0, x_sub = 0; x_sub < D_width / downsampling_factor; x += downsampling_factor, x_sub++) {
			for (int y = 0, y_sub = 0; y_sub < D_height / downsampling_factor; y += downsampling_factor, y_sub++) {
				sensor_depth.at<unsigned short>(y_sub, x_sub) = data[y* D_width + x];
			}
		}
		sample->depth->ReleaseAccess(&depth_buffer);
		sense_manager->ReleaseFrame();

		cv::medianBlur(sensor_depth, sensor_depth, KERNEL_SIZE);
		cv::resize(sensor_depth, sensor_depth_ds, cv::Size(D_width / (downsampling_factor*ds), D_height / (downsampling_factor * ds)), 0, 0, cv::INTER_NEAREST);

		sensor_depth_ds.setTo(cv::Scalar(BACKGROUND_DEPTH), sensor_depth_ds == 0);  //这里的意思是，将sensor_depth_ds等于0的像素设置成最大深度，输入setTo的高级用法
		sensor_depth.setTo(cv::Scalar(BACKGROUND_DEPTH), sensor_depth == 0);

		// prepare vector for fast element acces
		sensor_depth_ds.convertTo(src_X, CV_32F);
		float* ptr = (float*)src_X.data;
		size_t elem_step = src_X.step / sizeof(float);

		// build feature vector 
		locations.clear();

		cv::findNonZero(src_X < GET_CLOSER_TO_SENSOR, locations);   //又是一个高级用法，找到所有深度值小与GET_CLOSER_TO_SENSOR的像素点
		int n_samples = locations.size();

		if (n_samples <= 0)
		{
			cout << "深度值小与GET_CLOSER_TO_SENSOR的像素点为零 -----> continue..." << endl;
			continue;
		}

		// allocat memory for new data
		Array<float, 2, 2> new_data = allocate(n_samples, n_features);
		{
			//这里的用法我没看懂，但是应该是这样的：new_data对应的是n_samples*n_feature这样一个特征输入，每一个sample都对应一个feature；接下来使用多线程填入这些feature
			// Extract the lines serially, since the Array class is not thread-safe (yet)
			std::vector<Array<float, 2, 2>::Reference> lines;
			for (int i = 0; i < n_samples; ++i)
			{
				lines.push_back(new_data[i]);
			}

			for (int j = 0; j < n_samples; j++)
			{
				// depth of current pixel
				//Array<float, 2, 2> line = allocate(1, n_features);
				std::vector<float> features;
				float d = (float)ptr[elem_step*locations[j].y + locations[j].x];
				for (int k = 0; k < (2 * N_FEAT + 1); k++)
				{
					int idx_x = locations[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);
					for (int l = 0; l < (2 * N_FEAT + 1); l++)
					{
						int idx_y = locations[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
						// read data
						if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
						{
							features.push_back(BACKGROUND_DEPTH - d);
							continue;
						}
						float d_idx = (float)ptr[elem_step*idx_y + idx_x];
						features.push_back(d_idx - d);
					}
				}
				std::copy(features.begin(), features.end(), lines[j].getData());
			}
		}

		// predict data
		Array<double, 2, 2> predictions = forest->predict(new_data, N_THREADS);

		// build probability maps for current frame
		// hand
		cv::Mat probabilityMap = cv::Mat::zeros(SRC_ROWS, SRC_COLS, CV_32F);
		for (size_t j = 0; j < locations.size(); j++)
		{
			probabilityMap.at<float>(locations[j]) = predictions[j][1];
		}

		// COMPUTE AVERAGE DEPTH OF HAND BLOB ON LOW RES IMAGE
		// threshold low res hand probability map to obtain hand mask
		cv::Mat mask_ds = probabilityMap > THRESHOLD;
		// find biggest blob, a.k.a. hand 
		cv::findContours(mask_ds, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		if (contours.size() <= 0)
		{
			cout << "contours  is zero   ---->  continue.." << endl;
			continue;
		}
		int idx = 0, largest_component = 0;
		double max_area = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			double area = fabs(cv::contourArea(cv::Mat(contours[idx])));
			if (area > max_area)
			{
				max_area = area;
				largest_component = idx;
			}
		}


		// draw biggest blob
		cv::Mat mask_ds_biggest_blob = cv::Mat::zeros(mask_ds.size(), CV_8U);
		cv::drawContours(mask_ds_biggest_blob, contours, largest_component, cv::Scalar(255), CV_FILLED, 8, hierarchy);   //这里mask_ds_biggest_blob的图像并不是连续的，属于有许多空洞的图，正因为如此，后续才会使用dilate对mask进行膨胀，再结合深度rangemask确定最终人手图像。

																														 // compute average depth
		std::pair<float, int> avg;
		for (int row = 0; row < mask_ds_biggest_blob.rows; ++row)
		{
			for (int col = 0; col < mask_ds_biggest_blob.cols; ++col)
			{
				float depth_wrist = sensor_depth_ds.at<ushort>(row, col);
				if (mask_ds_biggest_blob.at<uchar>(row, col) == 255)
				{
					avg.first += depth_wrist;
					avg.second++;
				}
			}
		}
		ushort depth_hand = (avg.second == 0) ? BACKGROUND_DEPTH : avg.first / avg.second;

		cv::Mat probabilityMap_us;

		// UPSAMPLE USING RESIZE: advantages of joint bilateral upsampling are already exploited 
		cv::resize(probabilityMap, probabilityMap_us, sensor_depth.size());
		// BUILD HIGH RESOLUTION MASKS FOR HAND AND WRIST
		cv::Mat mask = probabilityMap_us > THRESHOLD;


		// Extract pixels at depth range on hand only
		ushort depth_range = 100;
		cv::Mat range_mask;
		cv::inRange(sensor_depth, depth_hand - depth_range, depth_hand + depth_range, range_mask);

		// POSTPROCESSING: APPLY SOME DILATION and SELECT BIGGEST BLOB
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1));
		cv::dilate(mask, mask, element);

		// deep copy because find contours modifies original image
		cv::Mat pp;
		mask.copyTo(pp);

		cv::findContours(pp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		if (contours.size() <= 0) continue;
		idx = 0, largest_component = 0;
		max_area = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			double area = fabs(cv::contourArea(cv::Mat(contours[idx])));
			//std::cout << area << std::endl;
			if (area > max_area)
			{
				max_area = area;
				largest_component = idx;
			}
		}
		cv::Mat dst = cv::Mat::zeros(mask.size(), CV_8U);
		cv::drawContours(dst, contours, largest_component, cv::Scalar(255), CV_FILLED, 8, hierarchy);
		dst.setTo(cv::Scalar(0), range_mask == 0);
		mask.setTo(cv::Scalar(0), dst == 0);

		//先进行镜像变换，然后对进行求掌心、distancetransform、点云转换；
		cv::flip(sensor_depth, depth_array[BACK_BUFFER], -1);
		cv::flip(mask, hand_BinaryMap[BACK_BUFFER], -1);

		cv::Moments m = cv::moments(hand_BinaryMap[BACK_BUFFER], true);
		int center_x = m.m10 / m.m00;
		int center_y = m.m01 / m.m00;

		handPointCloud[BACK_BUFFER].points.clear();
		int temp = 0, R = 0, cx = 0, cy = 0;

		{
			cv::Mat dist_image;
			cv::distanceTransform(hand_BinaryMap[BACK_BUFFER], dist_image, CV_DIST_L2, 3);

			int search_area_min_col = center_x - 50 > 0 ? center_x - 50 : 0;
			int search_area_max_col = center_x + 50 > OSRC_COLS ? OSRC_COLS - 1 : center_x + 50;

			int search_area_min_row = center_y - 50 > 0 ? center_y - 50 : 0;
			int search_area_max_row = center_y + 50 > OSRC_ROWS ? OSRC_ROWS - 1 : center_y + 50;

			for (int row = search_area_min_row; row < search_area_max_row; row++)
			{
				for (int col = search_area_min_col; col < search_area_max_col; col++)
				{
					if (hand_BinaryMap[BACK_BUFFER].at<uchar>(row, col) != 0)
					{
						temp = (int)dist_image.ptr<float>(row)[col];
						if (temp > R)
						{
							R = temp;
							cy = row;
							cx = col;
						}

					}
				}
			}

			int num_palm = 0;
			Eigen::Vector3f palm_points = Eigen::Vector3f::Zero();
			int PointCloudDownSampleSize = 3;

			for (int row_ = 0; row_ < OSRC_ROWS; row_ += PointCloudDownSampleSize)
			{
				for (int col_ = 0; col_ < OSRC_COLS; col_ += PointCloudDownSampleSize)
				{
					if (hand_BinaryMap[BACK_BUFFER].at<uchar>(row_, col_) != 0)
					{
						//点云转换
						int  z = depth_array[BACK_BUFFER].at<unsigned short>(row_, col_);
						Eigen::Vector3f p = camera->depth_to_world(col_, row_, z);
						pcl::PointXYZ point(p(0), p(1), p(2));

						handPointCloud[BACK_BUFFER].points.push_back(point);

						float distance_to_palm = sqrt((cx - col_)*(cx - col_)
							+ (cy - row_)*(cy - row_));

						if (distance_to_palm < R)
						{
							palm_points += p;
							++num_palm;
						}
					}
				}
			}

			if (num_palm > 0)
			{
				palm_center[BACK_BUFFER](0) = palm_points(0) / num_palm;
				palm_center[BACK_BUFFER](1) = palm_points(1) / num_palm;
				palm_center[BACK_BUFFER](2) = palm_points(2) / num_palm;
			}
		}

		//最后进行distance_transfrom
		{
			distance_transform.exec(hand_BinaryMap[BACK_BUFFER].data, 125);
			std::copy(distance_transform.idxs_image_ptr(), distance_transform.idxs_image_ptr() + OSRC_COLS * OSRC_ROWS, idxs_image_BACK_BUFFER);
		}

		// Lock the mutex and swap the buffers
		{
			std::unique_lock<std::mutex> lock(swap_mutex_realsense);
			condition_realsense.wait(lock, [] {return main_released_realsense; });
			thread_released_realsense = false;

			depth_array[FRONT_BUFFER] = depth_array[BACK_BUFFER].clone();
			hand_BinaryMap[FRONT_BUFFER] = hand_BinaryMap[BACK_BUFFER].clone();
			std::copy(idxs_image_BACK_BUFFER, idxs_image_BACK_BUFFER + OSRC_COLS * OSRC_ROWS, idxs_image_FRONT_BUFFER);

			palm_center[FRONT_BUFFER] = palm_center[BACK_BUFFER];
			handPointCloud[FRONT_BUFFER].points.assign(handPointCloud[BACK_BUFFER].points.begin(), handPointCloud[BACK_BUFFER].points.end());

			thread_released_realsense = true;
			lock.unlock();
			condition_realsense.notify_all();
		}

	}

	cout << "Error Quit !!!\n";
	return true;
}

bool RealSenseSensor::start()
{
	if (!initialized) this->initialize();

	return true;
}