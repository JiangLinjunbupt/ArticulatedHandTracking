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
	//�����Ƚ���Ⱦ�õ���rendered_model���ͼת���ɵ��ƣ�����kdtree�����뼯
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

	//����kdtree
	pcl::KdTreeFLANN<pcl::PointXYZ> search_kdtree;
	search_kdtree.setInputCloud(handmodel_pointcloud.makeShared());  //����ע��PCL��flann���opencv��flann��ͻ��ע����

	const int k = 1;
	std::vector<int> k_indices(k);
	std::vector<float> k_squared_distances(k);

	//Ȼ�������ͼ�е��ֲ���Ӧ���ص�ת��Ϊ3D����������������еĶ�Ӧ��
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
			if (d > 1e-3) weight = w*3.5;   //��������������쳣������Ĵ���3.5Ӧ����һ������ֵ������ϸ�μ�worker�е�reweight least square
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
			//ֻ����rendered_model���ͼ������sensor_depth���ͼ��ĵ㣬��Ϊ���ڲ�������Ϊ����
			if (rendered_sihouette.at<uchar>(row, col) == 255 && sensor_sihouette.at<uchar>(row, col) != 255)
			{
				//����ע��  distance_transform.idx_at ��ѯ���ǣ��ڼ��У��ڼ��С�
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