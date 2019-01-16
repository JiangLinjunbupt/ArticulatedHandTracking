#include"Sensor.h"
#include<Kinect.h>
#include"DataFrame.h"

#include<vector>

#include<thread>
#include<mutex>
#include<condition_variable>

using namespace std;

template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}

IKinectSensor       * mySensor = NULL;
IColorFrameReader   * mycolorReader = NULL;
IDepthFrameReader   * mydepthReader = NULL;
IBodyFrameReader	* myBodyReader = NULL;
IBodyFrameSource	* myBodySource = NULL;
ICoordinateMapper   * myMapper = NULL;

ColorSpacePoint     * m_pcolorcoordinate = new ColorSpacePoint[512 * 424];        //初始化很重要，不然再Map的时候会一直失败
CameraSpacePoint    * m_pcameracoordinate = new CameraSpacePoint[512 * 424];

std::thread sensor_thread_kinect;
std::mutex swap_mutex_kinect;
std::condition_variable condition_kinect;
bool main_released_kinect = true;
bool thread_released_kinect = false;

int sensor_frame_kinect = 0;
int tracker_frame_kinect = 0;


SensorKinect::SensorKinect(Camera* camera) :Sensor(camera, false)
{
	Depth_width = 512;
	Depth_height = 424;
	Color_width = 1920;
	Color_height = 1080;

	this->handfinder = new HandFinder(camera);
	this->real_color = false;
}

SensorKinect::SensorKinect(Camera* camera, bool real_color) :Sensor(camera, real_color)
{
	Depth_width = 512;
	Depth_height = 424;
	Color_width = 1920;
	Color_height = 1080;

	this->handfinder = new HandFinder(camera);
	this->real_color = real_color;
}

SensorKinect::~SensorKinect()
{
	std::cout << "~SensorKinect() function called  " << std::endl;
	if (!initialized) return;
	delete handfinder;
}

int SensorKinect::initialize()
{
	{
		if (sensor_indicator_array[FRONT_BUFFER].empty())
			sensor_indicator_array[FRONT_BUFFER] = std::vector<int>(upper_bound_num_sensor_points_Kinect, 0);
		if (sensor_indicator_array[BACK_BUFFER].empty())
			sensor_indicator_array[BACK_BUFFER] = std::vector<int>(upper_bound_num_sensor_points_Kinect, 0);

		if (depth_array[FRONT_BUFFER].empty())
			depth_array[FRONT_BUFFER] = cv::Mat(cv::Size(Depth_width/2, Depth_height/2), CV_16UC1, cv::Scalar(0));
		if (depth_array[BACK_BUFFER].empty())
			depth_array[BACK_BUFFER] = cv::Mat(cv::Size(Depth_width/2, Depth_height/2), CV_16UC1, cv::Scalar(0));

		if (color_array[FRONT_BUFFER].empty())
			color_array[FRONT_BUFFER] = cv::Mat(cv::Size(Depth_width/2, Depth_height/2), CV_8UC3, cv::Scalar(0, 0, 0));
		if (color_array[BACK_BUFFER].empty())
			color_array[BACK_BUFFER] = cv::Mat(cv::Size(Depth_width/2, Depth_height/2), CV_8UC3, cv::Scalar(0, 0, 0));

		if (full_color_array[FRONT_BUFFER].empty())
			full_color_array[FRONT_BUFFER] = cv::Mat(cv::Size(Color_width, Color_height), CV_8UC4, cv::Scalar(0, 0, 0, 0));
		if (full_color_array[BACK_BUFFER].empty())
			full_color_array[BACK_BUFFER] = cv::Mat(cv::Size(Color_width, Color_height), CV_8UC4, cv::Scalar(0, 0, 0, 0));

	}

	std::cout << "SensorKinect::initialize()" << std::endl;
	HRESULT hr;
	//搜索kinect
	hr = GetDefaultKinectSensor(&mySensor);
	if (FAILED(hr)) {
		return hr;
	}
	if (mySensor)
	{
		// Initialize the Kinect and get coordinate mapper and the body reader
		IColorFrameSource   * mycolorSource = nullptr;
		IDepthFrameSource   * mydepthSource = nullptr;   //取得深度数据

		hr = mySensor->Open();    //打开kinect 

								  //coordinatemapper
		if (SUCCEEDED(hr))
		{
			hr = mySensor->get_CoordinateMapper(&myMapper);
		}
		//color
		if (SUCCEEDED(hr))
		{
			hr = mySensor->get_ColorFrameSource(&mycolorSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = mycolorSource->OpenReader(&mycolorReader);
		}

		//depth
		if (SUCCEEDED(hr)) {
			hr = mySensor->get_DepthFrameSource(&mydepthSource);
		}

		if (SUCCEEDED(hr)) {
			hr = mydepthSource->OpenReader(&mydepthReader);
		}
		//body
		if (SUCCEEDED(hr)) {
			hr = mySensor->get_BodyFrameSource(&myBodySource);
		}

		if (SUCCEEDED(hr)) {
			hr = myBodySource->OpenReader(&myBodyReader);
		}


		SafeRelease(mycolorSource);
		SafeRelease(mydepthSource);
	}

	if (!mySensor || FAILED(hr))
	{
		std::cerr << "Kinect Initialization Failed!" << std::endl;
		return E_FAIL;
	}

	sensor_thread_kinect = std::thread(&SensorKinect::run, this);
	sensor_thread_kinect.detach();

	this->initialized = true;
	std::cout << "Kinect Initialization Success ! " << std::endl;
	return hr;
}

bool SensorKinect::concurrent_fetch_streams(DataFrame& frame, HandFinder& other_handfinder)
{
	std::unique_lock<std::mutex> lock(swap_mutex_kinect);
	condition_kinect.wait(lock, [] {return thread_released_kinect; });
	main_released_kinect = false;

	frame.id = tracker_frame_kinect;
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

	main_released_kinect = true;
	lock.unlock();
	condition_kinect.notify_one();

	return true;
}

bool SensorKinect::run()
{
	cout << "Kinect.run()" << endl;
	UINT16 *depthData = new UINT16[424 * 512];
	for (;;)
	{
		//如果丢失了kinect，则不继续操作
		if (!mydepthReader)
		{
			std::cout << "the depth reader is Missing, reboot the Kinect!!" << std::endl;
			return false;
		}

		IDepthFrame     * mydepthFrame = nullptr;
		IColorFrame     * mycolorFrame = nullptr;

		if (mydepthReader->AcquireLatestFrame(&mydepthFrame) == S_OK)
		{
			mydepthFrame->CopyFrameDataToArray(Depth_width*Depth_height, depthData); //先把数据存入16位的图像矩阵中

			for (int y = 0, y_sub = 0; y_sub < camera->height(); y += 2, y_sub++) {
				for (int x = 0, x_sub = 0; x_sub < camera->width(); x += 2, x_sub++) {
					if (x == 0 || y == 0) {
						depth_array[BACK_BUFFER].at<unsigned short>(y_sub, x_sub) = depthData[y*Depth_width + x ];
						continue;
					}
					std::vector<int> neighbors = {
						depthData[(y - 1)* Depth_width + (x - 1)],
						depthData[(y + 0)* Depth_width + (x - 1)],
						depthData[(y + 1)* Depth_width + (x - 1)],
						depthData[(y - 1)* Depth_width + (x + 0)],
						depthData[(y + 0)* Depth_width + (x + 0)],
						depthData[(y + 1)* Depth_width + (x + 0)],
						depthData[(y - 1)* Depth_width + (x + 1)],
						depthData[(y + 0)* Depth_width + (x + 1)],
						depthData[(y + 1)* Depth_width + (x + 1)],
					};
					std::sort(neighbors.begin(), neighbors.end());
					depth_array[BACK_BUFFER].at<unsigned short>(y_sub, x_sub) = neighbors[4];
				}
			}
		}

		if (mycolorReader->AcquireLatestFrame(&mycolorFrame) == S_OK)
		{
			mycolorFrame->CopyConvertedFrameDataToArray(Color_width*Color_height* 4, (BYTE *)full_color_array[BACK_BUFFER].data, ColorImageFormat_Bgra);
			while (myMapper->MapDepthFrameToColorSpace(Depth_width*Depth_height, depthData, Depth_width*Depth_height, m_pcolorcoordinate) != S_OK) { ; }


			for (int y = 0, y_sub = 0; y_sub < camera->height(); y += 2, y_sub++) {
				for (int x = 0, x_sub = 0; x_sub < camera->width(); x += 2, x_sub++) {
					int index_depth = y*Depth_width + x;
					ColorSpacePoint pp = m_pcolorcoordinate[index_depth];
					if (pp.X != -std::numeric_limits<float>::infinity() && pp.Y != -std::numeric_limits<float>::infinity())
					{
						int colorX = static_cast<int>(pp.X + 0.5f);   //上取整
						int colorY = static_cast<int>(pp.Y + 0.5f);
						if ((colorX >= 0 && colorX < 1920) && (colorY >= 0 && colorY < 1080))
						{
							unsigned char b = full_color_array[BACK_BUFFER].at<cv::Vec4b>(colorY, colorX)[0];
							unsigned char g = full_color_array[BACK_BUFFER].at<cv::Vec4b>(colorY, colorX)[1];
							unsigned char r = full_color_array[BACK_BUFFER].at<cv::Vec4b>(colorY, colorX)[2];
							color_array[BACK_BUFFER].at<cv::Vec3b>(y_sub, x_sub) = cv::Vec3b(b, g, r); //因为Opencv中通道的排列是BGR, 因此，我们需要吧blue通道放在第一个上面，green通道放在第二个上面，red通道放在第三个上面，放错会导致颜色出错
						}
					}
				}
			}
		}


		SafeRelease(mydepthFrame);
		SafeRelease(mycolorFrame);

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
			std::unique_lock<std::mutex> lock(swap_mutex_kinect);
			condition_kinect.wait(lock, [] {return main_released_kinect; });
			thread_released_kinect = false;

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

			tracker_frame_kinect = sensor_frame_kinect;
			sensor_frame_kinect++;

			thread_released_kinect = true;
			lock.unlock();
			condition_kinect.notify_one();
		}
	}
}

void SensorKinect::start()
{
	if (!initialized) this->initialize();
}

void SensorKinect::stop(){;}