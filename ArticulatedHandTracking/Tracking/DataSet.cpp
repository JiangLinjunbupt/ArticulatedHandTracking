#include"DataSet.h"

DataSet::DataSet(Camera* _camera) :camera(_camera)
{

	distance_transform.init(_camera->width(), _camera->height());

	this->LoadModel();

	//设置关节点之间的关系
	{
		//设置父节点关系
		this->Parent[0] = -1;
		for (int i = 1; i < this->Kintree_table.cols(); ++i) this->Parent[i] = this->Kintree_table(0, i);
		//设置子节点关系，为了后续计算局部坐标系
		{
			this->Child[0] = 4;

			this->Child[1] = 2;
			this->Child[2] = 3;
			this->Child[3] = -1;

			this->Child[4] = 5;
			this->Child[5] = 6;
			this->Child[6] = -1;

			this->Child[7] = 8;
			this->Child[8] = 9;
			this->Child[9] = -1;

			this->Child[10] = 11;
			this->Child[11] = 12;
			this->Child[12] = -1;

			this->Child[13] = 14;
			this->Child[14] = 15;
			this->Child[15] = -1;
		}

		//设置运动学链相关的节点关系，并且顺序按照：手腕->该节点
		this->joint_relation.resize(this->Joints_num);
		for (int i = 0; i < this->Joints_num; ++i)
		{
			std::vector<int> tmp;
			tmp.push_back(i);

			int parent = this->Parent[i];
			while (parent != -1)
			{
				tmp.push_back(parent);
				parent = this->Parent[parent];
			}
			std::sort(tmp.begin(), tmp.end());
			this->joint_relation[i] = tmp;
		}
		//for (int i = 0; i < this->Joints_num; ++i)
		//{
		//	for (auto j : this->joint_relation[i]) std::cout << j << "  ";
		//       std::cout << std::endl;
		//}
	}

	this->Local_Coordinate_Init();

	//控制手模位置、手型、状态的参数，初始都设置为0
	this->Shape_params = Eigen::VectorXf::Zero(this->Shape_params_num);
	this->Pose_params = Eigen::VectorXf::Zero(this->Pose_params_num);

	//初始化顶点矩阵
	this->V_shaped = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->V_posed = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->V_Final = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->V_Normal_Final = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	//初始化面法向量矩阵
	this->F_normal = Eigen::MatrixXf::Zero(this->Face_num, 3);
	//初始化关节点相关矩阵
	this->J_shaped = Eigen::MatrixXf::Zero(this->Joints_num, 3);
	this->J_Final = Eigen::MatrixXf::Zero(this->Joints_num, 3);

	//初始化jacobain计算相关的
	//Trans相关的
	{
		this->Trans_child_to_parent.resize(this->Joints_num);
		this->Trans_world_to_local.resize(this->Joints_num);
		for (int i = 0; i < this->Joints_num; ++i)
		{
			this->Trans_child_to_parent[i] = Eigen::Matrix4f::Zero();
			this->Trans_world_to_local[i] = Eigen::Matrix4f::Zero();
		}
	}


	//初始化中间变量
	this->result.resize(this->Joints_num);
	this->result2.resize(this->Joints_num);
	this->T.resize(this->Vertex_num);
	//对手模进行初始化变换

	Eigen::VectorXf pose_params_tmp = Eigen::VectorXf::Zero(Pose_params_num);
	Eigen::VectorXf shape_params_tmp = Eigen::VectorXf::Zero(Shape_params_num);

	this->set_Shape_Params(0);
	this->set_Pose_Params(Eigen::VectorXf::Zero(this->Pose_params_num));
	this->UpdataModel();

	std::cout << "Model Init Successed\n";
}


#pragma region LoadFunctions
void DataSet::LoadModel()
{
	std::string J_filename = ".\\model\\J.txt";
	std::string J_regressor_filename = ".\\model\\J_regressor.txt";
	std::string f_filename = ".\\model\\face.txt";
	std::string kintree_table_filename = ".\\model\\kintree_table.txt";
	std::string posedirs_filename = ".\\model\\posedirs.txt";
	std::string shapedirs_filename = ".\\model\\shapedirs.txt";
	std::string v_template_filename = ".\\model\\v_template.txt";
	std::string weights_filename = ".\\model\\weights.txt";

	std::string shape_params_filename = ".\\Data\\ShapeParams.txt";

	this->Load_J(J_filename.c_str());
	this->Load_J_regressor(J_regressor_filename.c_str());
	this->Load_F(f_filename.c_str());
	this->Load_Kintree_table(kintree_table_filename.c_str());
	this->Load_Posedirs(posedirs_filename.c_str());
	this->Load_Shapedirs(shapedirs_filename.c_str());
	this->Load_V_template(v_template_filename.c_str());
	this->Load_Weights(weights_filename.c_str());

	this->Load_ShapeParams(shape_params_filename.c_str());
	std::cout << "Load Model success " << std::endl;
}

void DataSet::Load_J(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  J error,  can not open this file !!! \n";

	f >> this->Joints_num;
	this->J = Eigen::MatrixXf::Zero(this->Joints_num, 3);
	for (int i = 0; i < this->Joints_num; ++i)
	{
		f >> this->J(i, 0) >> this->J(i, 1) >> this->J(i, 2);
	}
	f.close();

	this->J = this->J * 1000;
	std::cout << "Load J success\n";
}
void DataSet::Load_J_regressor(const char* filename)
{
	//这里注意加载的是稀疏矩阵，参考：
	//https://my.oschina.net/cvnote/blog/166980   或者  https://blog.csdn.net/xuezhisdc/article/details/54631490
	//Sparse Matrix转Dense Matrix ： MatrixXd dMat; dMat = MatrixXd(spMat);  可以方便观察
	//Dense Matrix转Sparse Matrix : SparseMatrix<double> spMat; spMat = dMat.sparseView(); 

	std::vector < Eigen::Triplet <float> > triplets;

	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  J_regressor  error,  can not open this file !!! \n";

	int rows, cols, NNZ_num;
	f >> rows >> cols >> NNZ_num;
	this->J_regressor.resize(rows, cols);

	float row, col, value;
	for (int i = 0; i < NNZ_num; ++i)
	{
		f >> row >> col >> value;
		triplets.emplace_back(row, col, value);
	}

	this->J_regressor.setFromTriplets(triplets.begin(), triplets.end());
	f.close();

	std::cout << "Load J_regressor success\n";
}
void DataSet::Load_F(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Face  error,  can not open this file !!! \n";

	f >> this->Face_num;
	this->F = Eigen::MatrixXi::Zero(this->Face_num, 3);

	for (int i = 0; i < this->Face_num; ++i)
	{
		int index1, index2, index3;
		f >> index1 >> index2 >> index3;
		this->F(i, 0) = index1;
		this->F(i, 1) = index2;
		this->F(i, 2) = index3;
	}
	f.close();

	std::cout << "Load face success\n";
}
void DataSet::Load_Kintree_table(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Kintree_table  error,  can not open this file !!! \n";

	int rows, cols;
	f >> rows >> cols;
	this->Kintree_table = Eigen::MatrixXf::Zero(rows, cols);
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			f >> this->Kintree_table(row, col);
		}
	}
	f.close();

	std::cout << "Load Kintree_table success\n";
}
void DataSet::Load_Posedirs(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Posedirs  error,  can not open this file !!! \n";

	int dim1, dim2, dim3;
	f >> dim1 >> dim2 >> dim3;
	this->Posedirs.resize(dim3);
	for (int d3 = 0; d3 < dim3; ++d3)
	{
		Eigen::MatrixXf tem = Eigen::MatrixXf::Zero(dim1, dim2);
		for (int d1 = 0; d1 < dim1; ++d1)
		{
			for (int d2 = 0; d2 < dim2; ++d2)
			{
				f >> tem(d1, d2);
			}
		}
		this->Posedirs[d3] = tem * 1000;
	}

	f.close();

	std::cout << "Load Posedirs success\n";
}
void DataSet::Load_Shapedirs(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Shapedirs  error,  can not open this file !!! \n";

	int dim1, dim2, dim3;
	f >> dim1 >> dim2 >> dim3;
	this->Shapedirs.resize(dim3);
	this->Joint_Shapedir.resize(dim3);

	for (int d3 = 0; d3 < dim3; ++d3)
	{
		Eigen::MatrixXf tem = Eigen::MatrixXf::Zero(dim1, dim2);
		for (int d1 = 0; d1 < dim1; ++d1)
		{
			for (int d2 = 0; d2 < dim2; ++d2)
			{
				f >> tem(d1, d2);
			}
		}
		this->Shapedirs[d3] = tem * 1000;

		//这里通过Shape_dir和J_regressor计算出Joint_shape_dir
		this->Joint_Shapedir[d3] = this->J_regressor * this->Shapedirs[d3];
	}

	f.close();

	std::cout << "Load Shapedirs success\n";
}
void DataSet::Load_V_template(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  V_template  error,  can not open this file !!! \n";

	int dim;
	f >> this->Vertex_num >> dim;
	this->V_template = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	for (int v = 0; v < this->Vertex_num; ++v)
	{
		f >> this->V_template(v, 0) >> this->V_template(v, 1) >> this->V_template(v, 2);
	}
	f.close();

	this->V_template = this->V_template * 1000;

	//这里为了保险，我重新计算一下V_template对应的Joint值；
	this->J = this->J_regressor * this->V_template;
	std::cout << "Load V_tempalte success\n";
}
void DataSet::Load_Weights(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  V_template  error,  can not open this file !!! \n";

	int rows, cols;
	f >> rows >> cols;
	this->Weights = Eigen::MatrixXf::Zero(rows, cols);
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			f >> this->Weights(row, col);
		}
	}
	f.close();

	std::cout << "Load Weights success\n";
}

void DataSet::Load_ShapeParams(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  V_template  error,  can not open this file !!! \n";

	int subjectsNum, paramsNum;
	f >> subjectsNum >> paramsNum;
	this->ShapeParams_Read = Eigen::MatrixXf::Zero(subjectsNum, paramsNum);

	for (int sub = 0; sub < subjectsNum; ++sub)
	{
		for (int params_idx = 0; params_idx < paramsNum; ++params_idx)
		{
			f >> ShapeParams_Read(sub, params_idx);
		}
	}
	f.close();
	std::cout << "Load ShapeParams_Read success\n";
}
#pragma endregion LoadFunctions


void DataSet::UpdataModel()
{
	this->Updata_V_rest();
	this->LBS_Updata();
	this->NormalUpdata();
}

void DataSet::Updata_V_rest()
{
	if (want_shapemodel && change_shape) this->ShapeSpaceBlend();
	this->PoseSpaceBlend();
}
void DataSet::ShapeSpaceBlend()
{
	this->V_shaped = this->V_template;
	this->J_shaped = this->J;

	for (int i = 0; i < this->Shape_params_num; ++i)
	{
		this->V_shaped += this->Shapedirs[i] * Shape_params[i];
		this->J_shaped += this->Joint_Shapedir[i] * Shape_params[i];
	}

	//为了后续的雅各比计算，这句话被我改编成了：在加载过程中先计算Joint_Shapedir，再通过Shape_params控制Joint的变化。
	//this->J_shaped = this->J_regressor*this->V_shaped;
	this->Local_Coordinate_Updata();
	this->Trans_Matrix_Updata();
	this->change_shape = false;
}

void DataSet::Local_Coordinate_Init()
{
	this->Local_Coordinate.resize(this->Joints_num);
	for (int i = 0; i < this->Joints_num; ++i) this->Local_Coordinate[i] = Eigen::Matrix4f::Zero();

	Eigen::Vector3f axis_x, axis_y, axis_z;

	for (int i = 0; i < this->Joints_num; ++i)
	{
		if (this->Child[i] != -1)
		{
			axis_x(0) = this->J(i, 0) - this->J(this->Child[i], 0);
			axis_x(1) = this->J(i, 1) - this->J(this->Child[i], 1);
			axis_x(2) = this->J(i, 2) - this->J(this->Child[i], 2);

			axis_z << 0.0f, 0.0f, 1.0f;

			//y = z*x
			axis_x.normalize();
			axis_y = axis_z.cross(axis_x);
			//z = x*y
			axis_y.normalize();
			axis_z = axis_x.cross(axis_y);
			axis_z.normalize();

			this->Local_Coordinate[i](0, 0) = axis_x(0); this->Local_Coordinate[i](0, 1) = axis_y(0); this->Local_Coordinate[i](0, 2) = axis_z(0); this->Local_Coordinate[i](0, 3) = this->J(i, 0);
			this->Local_Coordinate[i](1, 0) = axis_x(1); this->Local_Coordinate[i](1, 1) = axis_y(1); this->Local_Coordinate[i](1, 2) = axis_z(1); this->Local_Coordinate[i](1, 3) = this->J(i, 1);
			this->Local_Coordinate[i](2, 0) = axis_x(2); this->Local_Coordinate[i](2, 1) = axis_y(2); this->Local_Coordinate[i](2, 2) = axis_z(2); this->Local_Coordinate[i](2, 3) = this->J(i, 2);
			this->Local_Coordinate[i](3, 0) = 0.0f;      this->Local_Coordinate[i](3, 1) = 0.0f;      this->Local_Coordinate[i](3, 2) = 0.0f;      this->Local_Coordinate[i](3, 3) = 1.0f;
		}
		else
		{
			this->Local_Coordinate[i] = this->Local_Coordinate[this->Parent[i]];
			this->Local_Coordinate[i](0, 3) = this->J(i, 0);
			this->Local_Coordinate[i](1, 3) = this->J(i, 1);
			this->Local_Coordinate[i](2, 3) = this->J(i, 2);
		}
	}
}
void DataSet::Local_Coordinate_Updata()
{
	for (int i = 0; i < this->Joints_num; ++i)
	{
		this->Local_Coordinate[i](0, 3) = this->J_shaped(i, 0);
		this->Local_Coordinate[i](1, 3) = this->J_shaped(i, 1);
		this->Local_Coordinate[i](2, 3) = this->J_shaped(i, 2);
	}
}
void DataSet::Trans_Matrix_Updata()
{
	//因为0节点的父节点是世界坐标系，比较特殊，所以先设置0节点的值
	this->Trans_child_to_parent[0] = this->Local_Coordinate[0];

	Eigen::Matrix4f Trans_0_to_world_jacob = Eigen::Matrix4f::Zero();

	for (int i = 1; i < this->Joints_num; ++i)
	{
		this->Trans_child_to_parent[i] = this->Local_Coordinate[this->Parent[i]].inverse()*this->Local_Coordinate[i];
	}

	for (int i = 0; i < this->Joints_num; ++i)
	{
		this->Trans_world_to_local[i] = this->Local_Coordinate[i].inverse();
	}

}
void DataSet::PoseSpaceBlend()
{
	this->V_posed = this->V_shaped;

	std::vector<float> pose_vec = lortmin(this->Pose_params.tail(this->Finger_pose_num));

	for (int i = 0; i < pose_vec.size(); ++i)
	{
		this->V_posed += this->Posedirs[i] * pose_vec[i];
	}
}

void DataSet::LBS_Updata()
{
	//这一步的意思是，我先旋转，在将旋转后的结果转换到世界坐标系下，因此是：Trans*R
	Eigen::Matrix4f Rotate_0 = Eigen::Matrix4f::Identity();
	Rotate_0.block(0, 0, 3, 3) = EularToRotateMatrix(this->Pose_params[3], this->Pose_params[4], this->Pose_params[5]);
	result[0] = this->Trans_child_to_parent[0] * Rotate_0;

	//求剩下的result
	for (int i = 1; i < this->Joints_num; ++i)
	{
		Eigen::Matrix4f Rotate = Eigen::Matrix4f::Identity();
		Rotate.block(0, 0, 3, 3) = EularToRotateMatrix(this->Pose_params[(i + 1) * 3 + 0], this->Pose_params[(i + 1) * 3 + 1], this->Pose_params[(i + 1) * 3 + 2]);

		result[i] = result[this->Parent[i]] * this->Trans_child_to_parent[i] * Rotate;
	}

	//这一步的意思是，由于给出的顶点坐标和关节点坐标都是世界坐标系下的，因此我先要从世界坐标系转换到每个关节点下的局部坐标系，再做旋转变换，因此： R*Trans 
	for (int i = 0; i < result.size(); ++i)
	{
		result2[i] = result[i] * this->Trans_world_to_local[i];
	}

	//关节点变换，关节点没有权重分布，因此可以直接变换
	for (int i = 0; i < this->Joints_num; ++i)
	{
		Eigen::Vector4f temp(this->J_shaped.row(i)(0), this->J_shaped.row(i)(1), this->J_shaped.row(i)(2), 1);
		this->J_Final.row(i) = ((result2[i] * temp).head(3)).transpose() + (this->Pose_params.head(this->Global_position_num)).transpose();
	}

	//这个是考虑顶点权重累计之后的变换
	for (int i = 0; i < T.size(); ++i)
	{
		T[i].setZero();
		for (int j = 0; j < this->Joints_num; ++j)
		{
			T[i] += result2[j] * this->Weights(i, j);
		}
	}

	this->V_Final.setZero();

	for (int i = 0; i < this->Vertex_num; ++i)
	{
		Eigen::Vector4f temp(this->V_posed.row(i)(0), this->V_posed.row(i)(1), this->V_posed.row(i)(2), 1);
		this->V_Final.row(i) = ((T[i] * temp).head(3)).transpose() + (this->Pose_params.head(this->Global_position_num)).transpose();
	}

}
void DataSet::NormalUpdata()
{
	this->V_Normal_Final.setZero();
	this->F_normal.setZero();

	for (int i = 0; i < this->Face_num; ++i)
	{
		Eigen::Vector3f A, B, C, BA, BC;
		//这里我假设，如果假设错了，那么叉乘时候，就BC*BA变成BA*BC
		//            A
		//          /  \
										       //         B — C
		A << this->V_Final(this->F(i, 0), 0), this->V_Final(this->F(i, 0), 1), this->V_Final(this->F(i, 0), 2);
		B << this->V_Final(this->F(i, 1), 0), this->V_Final(this->F(i, 1), 1), this->V_Final(this->F(i, 1), 2);
		C << this->V_Final(this->F(i, 2), 0), this->V_Final(this->F(i, 2), 1), this->V_Final(this->F(i, 2), 2);


		BC << C - B;
		BA << A - B;

		Eigen::Vector3f nom(BC.cross(BA));

		nom.normalize();

		this->V_Normal_Final(this->F(i, 0), 0) += nom(0);
		this->V_Normal_Final(this->F(i, 0), 1) += nom(1);
		this->V_Normal_Final(this->F(i, 0), 2) += nom(2);

		this->V_Normal_Final(this->F(i, 1), 0) += nom(0);
		this->V_Normal_Final(this->F(i, 1), 1) += nom(1);
		this->V_Normal_Final(this->F(i, 1), 2) += nom(2);

		this->V_Normal_Final(this->F(i, 2), 0) += nom(0);
		this->V_Normal_Final(this->F(i, 2), 1) += nom(1);
		this->V_Normal_Final(this->F(i, 2), 2) += nom(2);


		this->F_normal(i, 0) = nom(0);
		this->F_normal(i, 1) = nom(1);
		this->F_normal(i, 2) = nom(2);
	}

	for (int i = 0; i < this->Vertex_num; ++i)
	{
		this->V_Normal_Final.row(i).normalize();
	}


	//这里统计一下可见点

	V_Visible.clear();

	for (int i = 0; i < this->Vertex_num; ++i)
	{
		if (this->V_Normal_Final(i, 2) < 0)
		{
			Eigen::Vector3f p(this->V_Final(i, 0), this->V_Final(i, 1), this->V_Final(i, 2));
			V_Visible.push_back(make_pair(p, i));
		}
	}
}

void DataSet::FetchDataFrame(DataFrame& dataframe)
{

	//需要准备的数据为：1.二值图 ；2.二值图进过distancetransfom之后的引索；3.可见点的pointcloud
	int width = camera->width();
	int heigth = camera->height();
	cv::Mat binaryMat = cv::Mat(cv::Size(width, heigth), CV_8UC1, cv::Scalar(0));

	for (int faceidx = 0; faceidx < this->Face_num; ++faceidx)
	{
		int a_idx = this->F(faceidx, 0);
		int b_idx = this->F(faceidx, 1);
		int c_idx = this->F(faceidx, 2);

		Eigen::Vector2f a_2D = camera->world_to_image(Eigen::Vector3f(this->V_Final(a_idx, 0), this->V_Final(a_idx, 1), this->V_Final(a_idx, 2)));
		Eigen::Vector2f b_2D = camera->world_to_image(Eigen::Vector3f(this->V_Final(b_idx, 0), this->V_Final(b_idx, 1), this->V_Final(b_idx, 2)));
		Eigen::Vector2f c_2D = camera->world_to_image(Eigen::Vector3f(this->V_Final(c_idx, 0), this->V_Final(c_idx, 1), this->V_Final(c_idx, 2)));

		int x_min = min(min(a_2D.x(), b_2D.x()), c_2D.x());
		int y_min = min(min(a_2D.y(), b_2D.y()), c_2D.y());
		int x_max = max(max(a_2D.x(), b_2D.x()), c_2D.x());
		int y_max = max(max(a_2D.y(), b_2D.y()), c_2D.y());

		for (int y = y_min; y <= y_max; ++y)
		{
			for (int x = x_min; x <= x_max; ++x)
			{
				if (x >= 0 && x < width && y >= 0 && y < heigth)
				{
					int a = (b_2D.x() - a_2D.x()) * (y - a_2D.y()) - (b_2D.y() - a_2D.y()) * (x - a_2D.x());
					int b = (c_2D.x() - b_2D.x()) * (y - b_2D.y()) - (c_2D.y() - b_2D.y()) * (x - b_2D.x());
					int c = (a_2D.x() - c_2D.x()) * (y - c_2D.y()) - (a_2D.y() - c_2D.y()) * (x - c_2D.x());

					if (a >= 0 && b >= 0 && c >= 0) binaryMat.at<uchar>(y, x) = 255;
					if (a <= 0 && b <= 0 && c <= 0) binaryMat.at<uchar>(y, x) = 255;
				}
			}
		}
	}

	dataframe.hand_BinaryMap = binaryMat.clone();

	//然后进行distance_transform
	this->distance_transform.exec(binaryMat.data, 125);
	std::copy(distance_transform.idxs_image_ptr(), distance_transform.idxs_image_ptr() + heigth * width, dataframe.idxs_image);

	//最后将可见点写入dataframe
	dataframe.handPointCloud.points.clear();
	for (int i = 0; i < this->Vertex_num; ++i)
	{
		if (this->V_Normal_Final(i, 2) < 0)
		{
			pcl::PointXYZ p(this->V_Final(i, 0), this->V_Final(i, 1), this->V_Final(i, 2));
			dataframe.handPointCloud.points.push_back(p);
		}
	}

}


//void HandModel::Change()
//{
//
//	this->Full_Hand_Pose.setZero();
//
//
//	Eigen::MatrixXf selected_hands_component = this->Hands_components.topRows(this->Num_Pose);
//	this->Full_Hand_Pose.segment(3, (this->Num_Params-3)) = selected_hands_component.transpose()*this->pose.segment(3, this->Num_Pose) + this->Hands_mean;
//
//
//	Eigen::MatrixXf v_shaped = this->V_template;
//	Eigen::MatrixXf J_shaped = Eigen::MatrixXf::Zero(this->Joints_num, 3);
//
//	Eigen::MatrixXf v_posed = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
//
//	if (want_shapemodel)
//	{
//		for (int i = 0; i < betas.size(); ++i)
//		{
//			v_shaped += this->Shapedirs[i] * betas[i];
//		}
//
//		J_shaped = this->J_regressor*v_shaped;
//
//	}
//
//
//	v_posed += v_shaped;
//
//	std::vector<float> pose_vec = lortmin(this->Full_Hand_Pose.segment(3, 45));
//
//	for (int i = 0; i < pose_vec.size(); ++i)
//	{
//		v_posed += this->Posedirs[i] * pose_vec[i];
//	}
//
//
//
//
//
//	std::vector<Eigen::Matrix4f> result(16,Eigen::Matrix4f::Identity());
//
//	result[0].block(0, 0, 3, 3) = rodrigues(this->Full_Hand_Pose[0], this->Full_Hand_Pose[1], this->Full_Hand_Pose[2]);
//	result[0].block(0, 3, 3, 1) = J_shaped.row(0).transpose();
//
//
//	for (int i = 1; i < this->Kintree_table.cols(); ++i)
//	{
//		Eigen::Matrix4f temp = Eigen::Matrix4f::Identity();
//		temp.block(0, 0, 3, 3) = rodrigues(this->Full_Hand_Pose[i * 3 + 0], this->Full_Hand_Pose[i * 3 + 1], this->Full_Hand_Pose[i * 3 + 2]);
//		temp.block(0, 3, 3, 1) = (J_shaped.row(i) - J_shaped.row(this->Parent[i])).transpose();
//
//		result[i] = result[this->Parent[i]] * temp;
//	}
//
//
//	std::vector<Eigen::Matrix4f> result2(result);
//	for (int i = 0; i < result.size(); ++i)
//	{
//		Eigen::MatrixXf tmp = result[i].block(0, 0, 3, 3)*(J_shaped.row(i).transpose());
//
//		result2[i].block(0, 3, 3, 1) -= tmp;
//	}
//
//
//	std::vector<Eigen::Matrix4f> T(this->Weights.rows(), Eigen::Matrix4f::Zero());
//
//	for (int i = 0; i < T.size(); ++i)
//	{
//		Eigen::Matrix4f temp = Eigen::Matrix4f::Zero();
//
//		for (int j = 0; j < result2.size(); ++j)
//		{
//			temp += result2[j]*this->Weights(i, j);
//		}
//		T[i] = temp;
//	}
//
//
//	Eigen::MatrixXf v_final = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
//
//	for (int i = 0; i < this->Vertex_num; ++i)
//	{
//
//		Eigen::Vector4f temp(v_posed.row(i)(0), v_posed.row(i)(1), v_posed.row(i)(2), 1);
//		v_final.row(i) = ((T[i] * temp).segment(0, 3)).transpose();
//	}
//
//
//
//}