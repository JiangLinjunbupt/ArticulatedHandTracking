#include"Evaluations.h"

Evaluations::Evaluations(Camera* _camera, HandModel* _handmodel) :camera(_camera), handmodel(_handmodel)
{
	distance_transform.init(_camera->width(), _camera->height());
}


void Evaluations::Compute_Metric(DataFrame* dataframe,int currentFrameNUM)
{
	handmodel->GenerateDepthMap();

	float E3D = Compute_3D_Metric(handmodel->HandModel_depthMap, handmodel->HandModel_binaryMap,
		dataframe->original_DepthMap, dataframe->hand_BinaryMap);

	float E2D = Compute_2D_Metric(handmodel->HandModel_depthMap, handmodel->HandModel_binaryMap,
		dataframe->original_DepthMap, dataframe->hand_BinaryMap);

	std::cout <<"Frame : "<< currentFrameNUM<< "\t\t E3D : " << E3D << '\t' << '\t' << "E2D : " << E2D << std::endl;
}


float Evaluations::Compute_3D_Metric(const cv::Mat& rendered_model, const cv::Mat& rendered_sihouette,
	const cv::Mat& sensor_depth, const cv::Mat& sensor_sihouette)
{
	//首先先将渲染得到的rendered_model深度图转换成点云，构造kdtree的输入集
	pcl::PointCloud<pcl::PointXYZ> handmodel_pointcloud;
	handmodel_pointcloud.clear();

	for (size_t row = 0; row < rendered_model.rows; ++row)
	{
		for (size_t col = 0; col < rendered_model.cols; ++col)
		{
			if (rendered_sihouette.at<uchar>(row, col) != 255) continue;
			float depth = rendered_model.at<ushort>(row, col);

			Eigen::Vector3f point = camera->depth_to_world(col, row, depth);
			handmodel_pointcloud.points.push_back(pcl::PointXYZ(point(0), point(1), point(2)));
		}
	}

	//构建kdtree
	pcl::KdTreeFLANN<pcl::PointXYZ> search_kdtree;
	search_kdtree.setInputCloud(handmodel_pointcloud.makeShared());  //这里注意PCL的flann会和opencv的flann冲突，注意解决

	const int k = 1;
	std::vector<int> k_indices(k);
	std::vector<float> k_squared_distances(k);

	//然后找深度图中的手部对应像素点转换为3D坐标后，在上述点云中的对应点
	float E = 0;
	int num_data_points = 0;
	float d, w, weight;

	for (size_t row = 0; row < sensor_depth.rows; ++row)
	{
		for (size_t col = 0; col < sensor_depth.cols; ++col)
		{
			if (sensor_sihouette.at<uchar>(row, col) != 255) continue;
			float depth = sensor_depth.at<ushort>(row, col);

			Eigen::Vector3f point = camera->depth_to_world(col, row, depth);
			pcl::PointXYZ p_search(point(0), point(1), point(2));

			search_kdtree.nearestKSearch(p_search, k, k_indices, k_squared_distances);

			d = sqrt(k_squared_distances[0]);
			w = 1 / (sqrt(d + 1e-3));
			weight = 1;
			if (d > 1e-3) weight = w*3.5;   //这个步骤是抑制异常点带来的大误差，3.5应该是一个经验值。可详细参见worker中的reweight least square
			//E += weight*d;

			if (d > 100) E += 100;
			else E += d;
			num_data_points++;
		}
	}

	float E3D = num_data_points > 0 ? E / num_data_points : 0;
	return E3D;
}


float Evaluations::Compute_2D_Metric(const cv::Mat& rendered_model, const cv::Mat& rendered_sihouette,
	const cv::Mat& sensor_depth, const cv::Mat& sensor_sihouette)
{
	this->distance_transform.exec(sensor_sihouette.data, 125);
	float E = 0;
	int num_data_points = 0;
	for (size_t col = 0; col < rendered_model.cols; ++col)
	{
		for (size_t row = 0; row < rendered_model.rows; ++row)
		{
			//只计算rendered_model深度图中落在sensor_depth深度图外的点，因为在内部距离认为是零
			if (rendered_sihouette.at<uchar>(row, col) == 255 && sensor_sihouette.at<uchar>(row, col) != 255)
			{
				//这里注意  distance_transform.idx_at 查询的是，第几行，第几列。
				int idx = distance_transform.idx_at(row, col);
				int cloest_row = idx / rendered_model.cols;
				int cloest_col = idx % rendered_model.cols;

				E += abs((float)cloest_row - (float)row);
				E += abs((float)cloest_col - (float)col);
				num_data_points++;
			}
		}
	}

	float E2D = num_data_points > 0 ? E / 2 / num_data_points : 0;

	return E2D;
}