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

void DataSet::FetchDataSet(DataFrame& dataframe, RuntimeType type, string& dataSetPath, int currframeIdx)
{
	int camera_width = camera->width();
	int camera_height = camera->height();
	cv::Mat depth_original = cv::Mat(cv::Size(camera_width, camera_height), CV_16UC1, cv::Scalar(0));
	cv::Mat hand_seg_binary = cv::Mat(cv::Size(camera_width, camera_height), CV_8UC1, cv::Scalar(0));
	//然后根据类型决定调用函数
	//需要准备的数据为：1.二值图 ；2.二值图进过distancetransfom之后的引索；3.可见点的pointcloud
	switch (type)
	{
	case Dataset_MSRA_14:
		fetch_DatasetMSRA_14(depth_original, hand_seg_binary,dataSetPath, currframeIdx);
		break;
	case Dataset_MSRA_15:
		fetch_DatasetMSRA_15(depth_original, hand_seg_binary, dataSetPath, currframeIdx);
		break;
	case Handy_teaser:
		fetch_Handy_teaser(depth_original, hand_seg_binary, dataSetPath, currframeIdx);
		break;
	default:
		cerr << "Invalid DataSetType , Cann't Fetch Input\n";
		return;
	}

	cv::flip(depth_original, dataframe.original_DepthMap, 0);
	cv::flip(hand_seg_binary, dataframe.hand_BinaryMap, 0);
	//然后进行distance_transform
	this->distance_transform.exec(dataframe.hand_BinaryMap.data, 125);
	std::copy(distance_transform.idxs_image_ptr(), distance_transform.idxs_image_ptr() + camera_width * camera_height, dataframe.idxs_image);


	cv::Moments m = cv::moments(hand_seg_binary, true);
	int center_x = m.m10 / m.m00;
	int center_y = m.m01 / m.m00;

	int temp = 0, R = 0, cx = 0, cy = 0;

	int COLS = hand_seg_binary.cols;
	int ROWS = hand_seg_binary.rows;

	float range = 80.0f;

	{
		cv::Mat dist_image;
		cv::distanceTransform(hand_seg_binary, dist_image, CV_DIST_L2, 3);

		int search_area_min_col = center_x - range > 0 ? center_x - range : 0;
		int search_area_max_col = center_x + range > COLS ? COLS - 1 : center_x + range;

		int search_area_min_row = center_y - range > 0 ? center_y - range : 0;
		int search_area_max_row = center_y + range > ROWS ? ROWS - 1 : center_y + range;

		for (int row = search_area_min_row; row < search_area_max_row; row++)
		{
			for (int col = search_area_min_col; col < search_area_max_col; col++)
			{
				if (hand_seg_binary.at<uchar>(row, col) != 0)
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
	}

	int DownSampleRate;
	int MaxPixelNUM = 192 * 2;
	int NonZero = cv::countNonZero(hand_seg_binary);
	if (NonZero > MaxPixelNUM)
		DownSampleRate = sqrt(NonZero / MaxPixelNUM);
	else
		DownSampleRate = 1;

	int num_palm = 0;

	dataframe.handPointCloud.points.clear();
	dataframe.palm_Center.setZero();
	for (int row = 0; row < hand_seg_binary.rows; row += DownSampleRate) {
		for (int col = 0; col < hand_seg_binary.cols; col += DownSampleRate) {
			if (hand_seg_binary.at<uchar>(row, col) != 255) continue;

			float depth_value = depth_original.at<unsigned short>(row, col);
			//Eigen::Vector3f p_pixel = camera->depth_to_world((camera->height() - row - 1), col, depth_value);
			Eigen::Vector3f p_pixel = camera->depth_to_world(col, (camera->height() - row - 1),depth_value);
			dataframe.handPointCloud.points.push_back(pcl::PointXYZ(p_pixel[0], p_pixel[1], p_pixel[2]));

			float distance_to_palm = sqrt((cx - col)*(cx - col)
				+ (cy - row)*(cy - row));

			if (distance_to_palm < R)
			{
				dataframe.palm_Center += p_pixel;
				++num_palm;
			}
		}
	}

	if (num_palm > 0)
	{
		dataframe.palm_Center /= num_palm;
	}
}

void DataSet::fetch_DatasetMSRA_14(cv::Mat& depth_original, cv::Mat& hand_seg_binary, string& dataSetPath, int currframeIdx)
{
	std::ostringstream stringstream;
	stringstream << std::setw(6) << std::setfill('0') << currframeIdx;
	string dataset_joint_filename = "joint.txt";
	string dataset_depth_path = dataSetPath + stringstream.str() + "_depth.bin";
	string dataset_joint_path = dataSetPath + dataset_joint_filename;

	//读取原始深度图，用数组保存
	int camera_width = camera->width();
	int camera_height = camera->height();
	float* pDepth = new float[camera_width *camera_height];
	std::ifstream fin(dataset_depth_path, std::ios::binary);
	if (!fin.is_open())
		cout << "can not Open" << dataset_depth_path << endl;
	fin.read((char*)pDepth, sizeof(float)*camera_height *camera_width);

	//将读入的原始深度数组保存，存入图片中
	for (int col = 0; col < camera_width; ++col)
	{
		for (int row = 0; row < camera_height; ++row)
		{
			if (pDepth[row*camera_width + col] != 0)
			{
				depth_original.at<ushort>(row, col) = pDepth[row*camera_width + col];

				hand_seg_binary.at<uchar>(row, col) = 255;
			}
		}
	}
	fin.close();

	//将数据集中的左手（至少看起来是左手，变成右手）
	cv::flip(depth_original, depth_original, 1);
	cv::flip(hand_seg_binary, hand_seg_binary, 1);

	////读取关节点
	//ifstream f_joint;
	//f_joint.open(dataset_joint_path, ios::in);
	//if (!f_joint.is_open())
	//	cout << "can not Open" << dataset_joint_path << endl;

	//int fileAmount;
	//f_joint >> fileAmount;

	//for (int i = 0; i <= index; ++i)
	//{
	//	for (int j = 0; j < JointNum; ++j)
	//	{
	//		f_joint >> joints_position_3D(j, 0)
	//			>> joints_position_3D(j, 1)
	//			>> joints_position_3D(j, 2);
	//		joints_position_3D(j, 0) = -joints_position_3D(j, 0);
	//		joints_position_3D(j, 2) = -joints_position_3D(j, 2);
	//	}
	//}
	//f_joint.close();

	delete[] pDepth;
}
void DataSet::fetch_DatasetMSRA_15(cv::Mat& depth_original, cv::Mat& hand_seg_binary, string& dataSetPath, int currframeIdx)
{
	std::ostringstream stringstream;
	stringstream << std::setw(6) << std::setfill('0') << currframeIdx;
	string dataset_joint_filename = "joint.txt";
	string dataset_depth_path = dataSetPath + stringstream.str() + "_depth.bin";
	string dataset_joint_path = dataSetPath + dataset_joint_filename;

	//读取原始深度图，用数组保存，注意DatasetMSRA_15的数据格式不是整幅图像，而是图像中的ROI的上下左右边界，所以要先把这些读出来
	FILE *pDepthFile = fopen(dataset_depth_path.c_str(), "rb");

	int img_width, img_height;
	int left, right, top, bottom;

	fread(&img_width, sizeof(int), 1, pDepthFile);
	fread(&img_height, sizeof(int), 1, pDepthFile);

	fread(&left, sizeof(int), 1, pDepthFile);
	fread(&top, sizeof(int), 1, pDepthFile);
	fread(&right, sizeof(int), 1, pDepthFile);
	fread(&bottom, sizeof(int), 1, pDepthFile);

	int bounding_box_width = right - left;
	int bounding_box_height = bottom - top;

	int cur_pixel_num = bounding_box_width * bounding_box_height;
	float* pDepth = new float[cur_pixel_num];
	fread(pDepth, sizeof(float), cur_pixel_num, pDepthFile);
	fclose(pDepthFile);

	//将读入的原始深度数组保存，存入图片中
	for (int row = top; row < bottom; ++row)
	{
		for (int col = left; col < right; ++col)
		{
			if (pDepth[(row - top)*bounding_box_width + (col - left)] != 0)
			{
				depth_original.at<ushort>(row, col) = pDepth[(row - top)*bounding_box_width + (col - left)];

				hand_seg_binary.at<uchar>(row, col) = 255;
			}

		}
	}
	//将数据集中的左手（至少看起来是左手，变成右手）
	cv::flip(depth_original, depth_original, 1);
	cv::flip(hand_seg_binary, hand_seg_binary, 1);

	////读取关节点
	//ifstream f_joint;
	//f_joint.open(dataset_joint_path, ios::in);
	//if (!f_joint.is_open())
	//	cout << "can not Open" << dataset_joint_path << endl;

	//int fileAmount;
	//f_joint >> fileAmount;

	//for (int i = 0; i <= index; ++i)
	//{
	//	for (int j = 0; j < JointNum; ++j)
	//	{
	//		f_joint >> joints_position_3D(j, 0)
	//			>> joints_position_3D(j, 1)
	//			>> joints_position_3D(j, 2);
	//		joints_position_3D(j, 0) = -joints_position_3D(j, 0);
	//		joints_position_3D(j, 2) = -joints_position_3D(j, 2);
	//	}
	//}
	//f_joint.close();

	delete[] pDepth;
}
void DataSet::fetch_Handy_teaser(cv::Mat& depth_original, cv::Mat& hand_seg_binary, string& dataSetPath, int currframeIdx)
{
	//先读入彩色图和深度图
	std::ostringstream stringstream;
	stringstream << std::setw(7) << std::setfill('0') << currframeIdx;
	depth_original = cv::imread(dataSetPath + "depth-" + stringstream.str() + ".png", cv::IMREAD_ANYDEPTH);
	cv::Mat color_original = cv::imread(dataSetPath + "color-" + stringstream.str() + ".png");

	//将数据集中的左手（至少看起来是左手，变成右手）
	cv::flip(depth_original, depth_original, 1);
	cv::flip(color_original, color_original, 1);

	//然后根据腕带进行分割
	float wband_size = 30.0f;
	float depth_range = 150.0f;

	//腕带的HSV颜色
	cv::Scalar hsv_min = cv::Scalar(94, 111, 37);
	cv::Scalar hsv_max = cv::Scalar(120, 255, 255);


	float depth_farplane = 500.0f;
	float depth_nearplane = 100.0f;
	float crop_radius = 150;

	cv::Mat color_hsv;
	cv::Mat in_z_range;
	cv::Mat depth_copy;
	cv::Mat mask_wristband;

	{
		cv::cvtColor(color_original, color_hsv, CV_RGB2HSV);
		cv::inRange(color_hsv, hsv_min, hsv_max, /*=*/ mask_wristband);

		cv::inRange(depth_original, depth_nearplane, depth_farplane /*mm*/, /*=*/ in_z_range);
		cv::bitwise_and(mask_wristband, in_z_range, mask_wristband);
	}

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
			//_has_useful_data = false;
		}
		else
		{
			//_has_useful_data = true;

			///--- Select 2nd biggest component
			mask_wristband = (labels == to_sort[1]);
			//_wristband_found = true;
		}
	}

	{
		///--- Extract wristband average depth
		std::pair<float, int> avg;
		for (int row = 0; row < mask_wristband.rows; ++row) {
			for (int col = 0; col < mask_wristband.cols; ++col) {
				float depth_wrist = depth_original.at<ushort>(row, col);
				if (mask_wristband.at<uchar>(row, col) == 255) {
					if ((depth_wrist>depth_nearplane) && (depth_wrist<depth_farplane)) {
						avg.first += depth_wrist;
						avg.second++;
					}
				}
			}
		}

		ushort depth_wrist = (avg.second == 0) ? depth_nearplane : avg.first / avg.second;

		///--- First just extract pixels at the depth range of the wrist
		cv::inRange(depth_original, depth_wrist - depth_range, /*mm*/
			depth_wrist + depth_range, /*mm*/
			hand_seg_binary /*=*/);
	}

	Eigen::Vector3f _wband_center = Eigen::Vector3f(0, 0, 0);  //也就是wrist的3D点云，将腕带上的蓝色点深度点，反投影到世界坐标的点云中。
	Eigen::Vector3f _wband_dir = Eigen::Vector3f(0, 0, -1);
	{
		///--- Compute MEAN
		int counter = 0;
		for (int row = 0; row < mask_wristband.rows; ++row) {
			for (int col = 0; col < mask_wristband.cols; ++col) {
				if (mask_wristband.at<uchar>(row, col) != 255) continue;
				float depth_value = depth_original.at<unsigned short>(row, col);
				_wband_center += camera->depth_to_world(col, (camera->height() - row - 1), depth_value);
				counter++;
			}
		}
		_wband_center /= counter;
		std::vector<Eigen::Vector3f> pts; pts.push_back(_wband_center);

		///--- Compute Covariance
		static std::vector<Eigen::Vector3f> points_pca;
		points_pca.reserve(100000);
		points_pca.clear();
		for (int row = 0; row < hand_seg_binary.rows; ++row) {
			for (int col = 0; col < hand_seg_binary.cols; ++col) {
				if (hand_seg_binary.at<uchar>(row, col) != 255) continue;
				float depth_value = depth_original.at<unsigned short>(row, col);
				Eigen::Vector3f p_pixel = camera->depth_to_world(col, (camera->height() - row - 1), depth_value);
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
		Eigen::Matrix3f cov = points_mat*points_mat.adjoint();
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
		_wband_dir = eig.eigenvectors().col(2);

		///--- Allow wrist to point downward
		if (_wband_dir.y()<0)
			_wband_dir = -_wband_dir;
	}

	{
		wband_size = 10;
		float crop_radius_sq = crop_radius*crop_radius;
		Eigen::Vector3f crop_center = _wband_center + _wband_dir*(crop_radius - wband_size /*mm*/);
		//Vector3 crop_center = _wband_center + _wband_dir*( crop_radius + wband_size /*mm*/);

		for (int row = 0; row < hand_seg_binary.rows; ++row) {
			for (int col = 0; col < hand_seg_binary.cols; ++col) {
				if (hand_seg_binary.at<uchar>(row, col) != 255) continue;
				float depth_value = depth_original.at<unsigned short>(row, col);
				Eigen::Vector3f p_pixel = camera->depth_to_world(col, (camera->height() - row - 1), depth_value);
				if ((p_pixel - crop_center).squaredNorm() < crop_radius_sq)
					hand_seg_binary.at<uchar>(row, col) = 255;
				else
					hand_seg_binary.at<uchar>(row, col) = 0;
			}
		}
	}

	int DILATION_SIZE = 9;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1));
	cv::dilate(mask_wristband, mask_wristband, element);
	hand_seg_binary.setTo(0, mask_wristband != 0);
}