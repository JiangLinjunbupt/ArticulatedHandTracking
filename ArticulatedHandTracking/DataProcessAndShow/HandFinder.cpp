#include "HandFinder.h"

HandFinder::HandFinder(Camera *camera) : camera(camera) {
	CHECK_NOTNULL(camera);

	if(camera->getmode() == CAMERAMODE(KinectV2))  sensor_indicator = new int[upper_bound_num_sensor_points_Kinect];
	if(camera->getmode() == CAMERAMODE(RealsenseSR300)) sensor_indicator = new int[upper_bound_num_sensor_points_Realsense];
}

Eigen::Vector3f point_at_depth_pixel(cv::Mat& depth, int x, int y, Camera* camera) {
	int z = depth.at<unsigned short>(y, x);
	return camera->depth_to_world(x, y, z);
}

void HandFinder::binary_classification(cv::Mat& depth, cv::Mat& color) {
	_wristband_found = false;


	///--- Fetch from settings
	float wband_size = _settings.wband_size;  //30
	float depth_range = _settings.depth_range;  //150

	///--- We look for wristband up to here...
	float depth_farplane = camera->zFar();

	float crop_radius = 150;

	///--- Allocated once
	static cv::Mat color_hsv;
	static cv::Mat in_z_range;
	static cv::Mat depth_copy;

	// TIMED_BLOCK(timer,"Worker_classify::(convert to HSV)")
	{
		cv::cvtColor(color, color_hsv, CV_BGR2HSV);

		cv::inRange(color_hsv, settings->hsv_min1, settings->hsv_max1, /*=*/ mask_wristband);
		cv::inRange(color_hsv, settings->hsv_min2, settings->hsv_max2, /*=*/ mask_wristband2);

		cv::bitwise_or(mask_wristband, mask_wristband2, mask_wristband);

		cv::inRange(depth, camera->zNear(), depth_farplane /*mm*/, /*=*/ in_z_range);
		cv::bitwise_and(mask_wristband, in_z_range, mask_wristband);
		//cv::imshow("mask_wristband (pre)", mask_wristband); cv::waitKey(1);
	}

	// TIMED_BLOCK(timer,"Worker_classify::(robust wrist)")
	{
		cv::Mat labels, stats, centroids;
		int num_components = cv::connectedComponentsWithStats(mask_wristband, labels, stats, centroids, 4 /*connectivity={4,8}*/);

		///--- Generate array to sort
		std::vector< int > to_sort(num_components);
		std::iota(to_sort.begin(), to_sort.end(), 0 /*start from*/);

		///--- Sort accoding to area
		auto lambda = [stats](int i1, int i2) {
			int area1 = stats.at<int>(i1, cv::CC_STAT_AREA);
			int area2 = stats.at<int>(i2, cv::CC_STAT_AREA);
			return area1>area2;
		};
		std::sort(to_sort.begin(), to_sort.end(), lambda);

		if (num_components<2 /*not found anything beyond background*/) {
			_has_useful_data = false;
		}
		else
		{
			if (_has_useful_data == false) {
				//std::cout << "NEW useful data => reinit" << std::endl;
				//trivial_detector->exec(frame, sensor_silhouette);
			}
			_has_useful_data = true;

			///--- Select 2nd biggest component
			mask_wristband = (labels == to_sort[1]);
			_wristband_found = true;
		}
	}

	if (_settings.show_wband) {
		cv::imshow("show_wband", mask_wristband);
		cv::waitKey(1);
	}
	else
		cv::destroyWindow("show_wband");

	// TIMED_BLOCK(timer,"Worker_classify::(crop at wrist depth)")
	{
		///--- Extract wristband average depth
		std::pair<float, int> avg;
		for (int row = 0; row < mask_wristband.rows; ++row) {
			for (int col = 0; col < mask_wristband.cols; ++col) {
				float depth_wrist = depth.at<ushort>(row, col);
				if (mask_wristband.at<uchar>(row, col) == 255) {
					if (camera->is_valid(depth_wrist)) {
						avg.first += depth_wrist;
						avg.second++;
					}
				}
			}
		}
		ushort depth_wrist = (avg.second == 0) ? camera->zNear() : avg.first / avg.second;

		///--- First just extract pixels at the depth range of the wrist
		cv::inRange(depth, depth_wrist - depth_range, /*mm*/
			depth_wrist + depth_range, /*mm*/
			sensor_silhouette /*=*/);
	}

	//cv::imshow("sensor_silhouette (before)", sensor_silhouette);

	_wband_center = Eigen::Vector3f(0, 0, 0);  //也就是wrist的3D点云，将腕带上的蓝色点深度点，反投影到世界坐标的点云中。
	_wband_dir = Eigen::Vector3f(0, 0, -1);
	// TIMED_BLOCK(timer,"Worker_classify::(PCA)")
	{
		///--- Compute MEAN
		int counter = 0;
		for (int row = 0; row < mask_wristband.rows; ++row) {
			for (int col = 0; col < mask_wristband.cols; ++col) {
				if (mask_wristband.at<uchar>(row, col) != 255) continue;
				_wband_center += point_at_depth_pixel(depth, col, row, camera);
				counter++;
			}
		}
		_wband_center /= counter;
		std::vector<Eigen::Vector3f> pts; pts.push_back(_wband_center);

		///--- Compute Covariance
		static std::vector<Eigen::Vector3f> points_pca;
		points_pca.reserve(100000);
		points_pca.clear();
		for (int row = 0; row < sensor_silhouette.rows; ++row) {
			for (int col = 0; col < sensor_silhouette.cols; ++col) {
				if (sensor_silhouette.at<uchar>(row, col) != 255) continue;
				Eigen::Vector3f p_pixel = point_at_depth_pixel(depth, col, row, camera);
				if ((p_pixel - _wband_center).norm()<100) {
					// sensor_silhouette.at<uchar>(row,col) = 255;
					points_pca.push_back(p_pixel);
				}
				else {
					// sensor_silhouette.at<uchar>(row,col) = 0;
				}
			}
		}
		if (points_pca.size() == 0) return;
		///--- Compute PCA
		Eigen::Map<Matrix_3xN> points_mat(points_pca[0].data(), 3, points_pca.size());
		for (int i : {0, 1, 2})
			points_mat.row(i).array() -= _wband_center(i);
		Matrix3 cov = points_mat*points_mat.adjoint();
		Eigen::SelfAdjointEigenSolver<Matrix3> eig(cov);
		_wband_dir = eig.eigenvectors().col(2);

		///--- Allow wrist to point downward
		if (_wband_dir.y()<0)
			_wband_dir = -_wband_dir;
	}
	// TIMED_BLOCK(timer,"Worker_classify::(in sphere)")
	{
		wband_size = 10;
		float crop_radius_sq = crop_radius*crop_radius;
		Eigen::Vector3f crop_center = _wband_center + _wband_dir*(crop_radius - wband_size /*mm*/);
		//Vector3 crop_center = _wband_center + _wband_dir*( crop_radius + wband_size /*mm*/);

		for (int row = 0; row < sensor_silhouette.rows; ++row) {
			for (int col = 0; col < sensor_silhouette.cols; ++col) {
				if (sensor_silhouette.at<uchar>(row, col) != 255) continue;

				Eigen::Vector3f p_pixel = point_at_depth_pixel(depth, col, row, camera);
				if ((p_pixel - crop_center).squaredNorm() < crop_radius_sq)
					sensor_silhouette.at<uchar>(row, col) = 255;
				else
					sensor_silhouette.at<uchar>(row, col) = 0;
			}
		}
	}


	if (_settings.show_hand) {
		cv::imshow("show_hand", sensor_silhouette);
	}
	else {
		cv::destroyWindow("show_hand");
	}
}

