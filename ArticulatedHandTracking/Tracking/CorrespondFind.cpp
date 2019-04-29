#include"CorrespondFind.h"


void CorrespondFind::Find()
{
	correspond.clear();
	if (my_dataframe->handPointCloud.points.size() > 10)
	{
		correspond.resize(my_dataframe->handPointCloud.points.size());

		FindCore(0, my_dataframe->handPointCloud.points.size() - 1);

		num_matched_correspond = 0;
		for (int i = 0; i < my_dataframe->handPointCloud.points.size(); ++i)
		{
			if (correspond[i].is_match)
				++num_matched_correspond;
		}
		//std::cout << "matched correspond is :  " << num_matched_correspond << std::endl;

	}
}

void CorrespondFind::FindCore(int start, int end)
{
	for (int i = start; i <= end; ++i)
	{
		Eigen::Vector3f p = Eigen::Vector3f(this->my_dataframe->handPointCloud.points[i].x, this->my_dataframe->handPointCloud.points[i].y, this->my_dataframe->handPointCloud.points[i].z);
		correspond[i].pointcloud = p;
		correspond[i].pointcloud_idx = i;

		//然后将该点转换到图像平面中，确定搜索范围后，再在对应的手模可见点中找最近点
		Eigen::Vector2f p_2D = camera->world_to_image(p);

		int gap = 15;
		int search_min_x = (p_2D.x() - gap) > 0 ? (p_2D.x() - gap) : 0;
		int search_max_x = (p_2D.x() + gap) > my_dataframe->width ? my_dataframe->width : (p_2D.x() + gap);

		int search_min_y = (p_2D.y() - gap) > 0 ? (p_2D.y() - gap) : 0;
		int search_max_y = (p_2D.y() + gap) > my_dataframe->height ? my_dataframe->height : (p_2D.y() + gap);

		float Min_distance = 1000000;

		for (int search_row = search_min_y; search_row < search_max_y; ++search_row)
		{
			for (int search_col = search_min_x; search_col < search_max_x; ++search_col)
			{
				int idx_hand = my_dataframe->handmodel_visibleMap.at<int>(search_row, search_col);
				if (idx_hand < 10000)
				{
					Eigen::Vector3f p_hand = handmodel->V_Final.row(idx_hand).transpose();
					float distance = (p - p_hand).norm();

					if (distance < Min_distance)
					{
						Min_distance = distance;
						correspond[i].correspond = p_hand;
						correspond[i].correspond_idx = idx_hand;
					}
				}
			}
		}

		if (Min_distance > 50)
			correspond[i].is_match = false;
		else
			correspond[i].is_match = true;

	}
}

void CorrespondFind::Find_2()
{
	correspond.clear();
	//首先加载handmodel中的visible point

	pcl::PointCloud<pcl::PointXYZ> Handmodel_visible_cloud;
	std::vector<int> visible_idx;

	for (int i = 0; i < handmodel->Vertex_num; ++i)
	{
		if (handmodel->V_Normal_Final(i, 2) <= 0)
		{
			pcl::PointXYZ p;
			p.x = handmodel->V_Final(i, 0);
			p.y = handmodel->V_Final(i, 1);
			p.z = handmodel->V_Final(i, 2);

			Handmodel_visible_cloud.points.push_back(p);
			visible_idx.push_back(i);
		}
	}


	//然后再找对应点
	int NumVisible_ = Handmodel_visible_cloud.points.size();
	int NumPointCloud_sensor = my_dataframe->handPointCloud.points.size();

	if (NumVisible_ > 0 && NumPointCloud_sensor > 0)
	{
		correspond.resize(NumPointCloud_sensor);

		pcl::KdTreeFLANN<pcl::PointXYZ> search_kdtree;
		search_kdtree.setInputCloud(Handmodel_visible_cloud.makeShared());  //这里注意PCL的flann会和opencv的flann冲突，注意解决

		const int k = 1;
		std::vector<int> k_indices(k);
		std::vector<float> k_squared_distances(k);
		for (int i = 0; i < NumPointCloud_sensor; ++i)
		{
			search_kdtree.nearestKSearch(my_dataframe->handPointCloud, i, k, k_indices, k_squared_distances);

			Eigen::Vector3f p = Eigen::Vector3f(my_dataframe->handPointCloud.points[i].x, my_dataframe->handPointCloud.points[i].y, my_dataframe->handPointCloud.points[i].z);
			correspond[i].pointcloud = p;

			Eigen::Vector3f p_2 = Eigen::Vector3f(Handmodel_visible_cloud.points[k_indices[0]].x,
				Handmodel_visible_cloud.points[k_indices[0]].y,
				Handmodel_visible_cloud.points[k_indices[0]].z);

			correspond[i].correspond = p_2;
			correspond[i].correspond_idx = visible_idx[k_indices[0]];

			float distance = (p_2 - p).norm();

			if (distance > 50)
				correspond[i].is_match = false;
			else
				correspond[i].is_match = true;
		}
	}


	num_matched_correspond = 0;
	for (int i = 0; i < NumPointCloud_sensor; ++i)
	{
		if (correspond[i].is_match)
			++num_matched_correspond;
	}
}