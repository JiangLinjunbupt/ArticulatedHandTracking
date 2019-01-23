#include"HandModel.h"

HandModel::HandModel()
{
	this->LoadModel();
	std::cout << "Load Model success "<<std::endl;

	this->V_Final_array = new float[this->Vertex_num * 3]();
	this->Normal_Final_array = new float[this->Vertex_num * 3]();

	//设置一些初始值
	for (int i = 1; i < this->Kintree_table.cols(); ++i) this->Parent[i] = this->Kintree_table(0, i);

	//控制手模位置、手型、状态的参数，初始都设置为0
	this->trans = Eigen::Vector3f::Zero();
	this->betas = Eigen::VectorXf::Zero(this->Num_betas);
	this->pose = Eigen::VectorXf::Zero(this->Num_WristPose + this->Num_FingerPose);

	this->Full_Hand_Params = Eigen::VectorXf::Zero(this->Num_WristParams + this->Num_FingerParams);

	this->V_shaped = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->J_shaped = Eigen::MatrixXf::Zero(this->Joints_num, 3);
	this->V_posed = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->V_Final = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->Normal_Final = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->J_Final = Eigen::MatrixXf::Zero(this->Joints_num, 3);


	this->set_Shape_Params(Eigen::VectorXf::Zero(this->Num_betas));
	this->UpdataModel();

	std::cout << "Model Init Successed\n";

}


#pragma region LoadFunctions
void HandModel::LoadModel()
{
	std::string J_filename = ".\\model\\J.txt";
	std::string J_regressor_filename = ".\\model\\J_regressor.txt";
	std::string f_filename = ".\\model\\face.txt";
	std::string hands_coeffs_filename = ".\\model\\hands_coeffs.txt";
	std::string hands_components_filename = ".\\model\\hands_components.txt";
	std::string hands_mean_filename = ".\\model\\hands_mean.txt";
	std::string kintree_table_filename = ".\\model\\kintree_table.txt";
	std::string posedirs_filename = ".\\model\\posedirs.txt";
	std::string shapedirs_filename = ".\\model\\shapedirs.txt";
	std::string v_template_filename = ".\\model\\v_template.txt";
	std::string weights_filename = ".\\model\\weights.txt";


	this->Load_J(J_filename.c_str());
	this->Load_J_regressor(J_regressor_filename.c_str());
	this->Load_F(f_filename.c_str());
	//this->Load_Hands_coeffs(hands_coeffs_filename.c_str());   //这个好像没什么用，暂时不加载
	this->Load_Hands_components(hands_components_filename.c_str());
	this->Load_Hands_mean(hands_mean_filename.c_str());
	this->Load_Kintree_table(kintree_table_filename.c_str());
	this->Load_Posedirs(posedirs_filename.c_str());
	this->Load_Shapedirs(shapedirs_filename.c_str());
	this->Load_V_template(v_template_filename.c_str());
	this->Load_Weights(weights_filename.c_str());

}

void HandModel::Load_J(const char* filename)
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

	std::cout << "Load J success\n";
}
void HandModel::Load_J_regressor(const char* filename)
{
	//这里注意加载的是稀疏矩阵，参考：
	//https://my.oschina.net/cvnote/blog/166980   或者  https://blog.csdn.net/xuezhisdc/article/details/54631490
	//Sparse Matrix转Dense Matrix ： MatrixXd dMat; dMat = MatrixXd(spMat);  可以方便观察
	//Dense Matrix转Sparse Matrix : SparseMatrix<double> spMat; spMat = dMat.sparseView(); 

	std::vector < Eigen::Triplet <float> > triplets;

	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  J_regressor  error,  can not open this file !!! \n";

	int rows, cols,NNZ_num;
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
void HandModel::Load_F(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Face  error,  can not open this file !!! \n";

	f >> this->Face_num;
	this->F = Eigen::MatrixXi::Zero(this->Face_num, 3);
	this->F_array = new unsigned int[this->Face_num * 3]();

	for (int i = 0; i < this->Face_num; ++i)
	{
		int index1, index2, index3;
		f >> index1 >> index2 >> index3;
		this->F(i, 0) = index1;
		this->F(i, 1) = index2;
		this->F(i, 2) = index3;

		this->F_array[i * 3 + 0] = index1;
		this->F_array[i * 3 + 1] = index2;
		this->F_array[i * 3 + 2] = index3;
	}
	f.close();

	std::cout << "Load face success\n";
}
void HandModel::Load_Hands_coeffs(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Hands_coeffs  error,  can not open this file !!! \n";

	int rows, cols;
	f >> rows >> cols;
	this->Hands_coeffs = Eigen::MatrixXf::Zero(rows, cols);
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			f >> this->Hands_coeffs(row, col);
		}
	}
	f.close();

	std::cout << "Load Hands_coeffs success\n";
}
void HandModel::Load_Hands_components(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Hands_components  error,  can not open this file !!! \n";

	int rows, cols;
	f >> rows >> cols;
	this->Hands_components = Eigen::MatrixXf::Zero(rows, cols);
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			f >> this->Hands_components(row, col);
		}
	}
	f.close();

	std::cout << "Load Hands_components success\n";
}
void HandModel::Load_Hands_mean(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Hands_mean  error,  can not open this file !!! \n";

	int num;
	f >> num;
	this->Hands_mean = Eigen::VectorXf::Zero(num);
	for (int i = 0; i < num; ++i)
		f >> this->Hands_mean(i);
	f.close();

	std::cout << "Load Hands_mean success\n";
}
void HandModel::Load_Kintree_table(const char* filename)
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
void HandModel::Load_Posedirs(const char* filename)
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
		this->Posedirs[d3] = tem;
	}

	f.close();

	std::cout << "Load Posedirs success\n";
}
void HandModel::Load_Shapedirs(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Shapedirs  error,  can not open this file !!! \n";

	int dim1, dim2, dim3;
	f >> dim1 >> dim2 >> dim3;
	this->Shapedirs.resize(dim3);

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
		this->Shapedirs[d3] = tem;
	}

	f.close();

	std::cout << "Load Shapedirs success\n";
}
void HandModel::Load_V_template(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  V_template  error,  can not open this file !!! \n";

	int dim;
	f >> this->Vertex_num>>dim;
	this->V_template = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	for (int v = 0; v < this->Vertex_num; ++v)
	{
		f >> this->V_template(v, 0) >> this->V_template(v, 1) >> this->V_template(v, 2);
	}
	f.close();

	std::cout << "Load V_tempalte success\n";
}
void HandModel::Load_Weights(const char* filename)
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
#pragma endregion LoadFunctions


void HandModel::UpdataModel()
{
	this->Full_Hand_Params.setZero();

	//通过Pose构造Full_Hand_Pose
	Eigen::MatrixXf selected_hands_component = this->Hands_components.topRows(this->Num_FingerPose);
	this->Full_Hand_Params[0] = this->pose[0];
	this->Full_Hand_Params[1] = this->pose[1];
	this->Full_Hand_Params[2] = this->pose[2];
	this->Full_Hand_Params.segment(this->Num_WristParams, this->Num_FingerParams) = selected_hands_component.transpose()*this->pose.segment(this->Num_WristPose, this->Num_FingerPose) + this->Hands_mean;

	this->Updata_V_rest();
	this->LBS_Updata();
	this->NormalUpdata();
}


void HandModel::Updata_V_rest()
{

	if (want_shapemodel && change_shape) this->ShapeSpaceBlend();

	this->PoseSpaceBlend();
}
void HandModel::ShapeSpaceBlend()
{
	this->V_shaped = this->V_template;

	for (int i = 0; i < this->Num_betas; ++i)
	{
		this->V_shaped += this->Shapedirs[i] * betas[i];
	}

	this->J_shaped = this->J_regressor*this->V_shaped;

	this->change_shape = false;
}
void HandModel::PoseSpaceBlend()
{
	this->V_posed = this->V_shaped;
	
	std::vector<float> pose_vec = lortmin(this->Full_Hand_Params.segment(this->Num_WristParams,this->Num_FingerParams));

	for (int i = 0; i < pose_vec.size(); ++i)
	{
		this->V_posed += this->Posedirs[i] * pose_vec[i];
	}
}

void HandModel::LBS_Updata()
{

	std::vector<Eigen::Matrix4f> result(this->Joints_num, Eigen::Matrix4f::Identity());

	result[0].block(0, 0, 3, 3) = rodrigues(this->Full_Hand_Params[0], this->Full_Hand_Params[1], this->Full_Hand_Params[2]);
	result[0].block(0, 3, 3, 1) = J_shaped.row(0).transpose();


	for (int i = 1; i < this->Kintree_table.cols(); ++i)
	{
		Eigen::Matrix4f temp = Eigen::Matrix4f::Identity();
		temp.block(0, 0, 3, 3) = rodrigues(this->Full_Hand_Params[i * 3 + 0], this->Full_Hand_Params[i * 3 + 1], this->Full_Hand_Params[i * 3 + 2]);
		temp.block(0, 3, 3, 1) = (J_shaped.row(i) - J_shaped.row(this->Parent[i])).transpose();

		result[i] = result[this->Parent[i]] * temp;
	}


	std::vector<Eigen::Matrix4f> result2(result);
	for (int i = 0; i < result.size(); ++i)
	{
		Eigen::MatrixXf tmp = result[i].block(0, 0, 3, 3)*(J_shaped.row(i).transpose());

		result2[i].block(0, 3, 3, 1) -= tmp;
	}

	std::vector<Eigen::Matrix4f> T(this->Weights.rows(), Eigen::Matrix4f::Zero());

	for (int i = 0; i < T.size(); ++i)
	{
		Eigen::Matrix4f temp = Eigen::Matrix4f::Zero();

		for (int j = 0; j < result2.size(); ++j)
		{
			temp += result2[j] * this->Weights(i, j);
		}
		T[i] = temp;
	}

	this->V_Final.setZero();

	for (int i = 0; i < this->Vertex_num; ++i)
	{
		Eigen::Vector4f temp(this->V_posed.row(i)(0), this->V_posed.row(i)(1), this->V_posed.row(i)(2), 1);
		this->V_Final.row(i) = ((T[i] * temp).segment(0, 3)).transpose() + this->trans.transpose();
	}


}
void HandModel::NormalUpdata()
{
	this->Normal_Final.setZero();

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

		this->Normal_Final(this->F(i, 0), 0) += nom(0);
		this->Normal_Final(this->F(i, 0), 1) += nom(1);
		this->Normal_Final(this->F(i, 0), 2) += nom(2);

		this->Normal_Final(this->F(i, 1), 0) += nom(0);
		this->Normal_Final(this->F(i, 1), 1) += nom(1);
		this->Normal_Final(this->F(i, 1), 2) += nom(2);

		this->Normal_Final(this->F(i, 2), 0) += nom(0);
		this->Normal_Final(this->F(i, 2), 1) += nom(1);
		this->Normal_Final(this->F(i, 2), 2) += nom(2);

	}

	for (int i = 0; i < this->Vertex_num; ++i)
	{
		this->Normal_Final.row(i).normalize();
	}
}

void HandModel::Save_as_obj()
{
	std::ofstream f;
	f.open("MANO_HandModel.obj");
	if (!f.is_open())  std::cout << "Can not Save to .obj file, The file can not open ！！！\n";

	for (int i = 0; i < this->Vertex_num; ++i)
	{
		f << "v " << this->V_Final(i, 0) << " " << this->V_Final(i, 1) << " " << this->V_Final(i, 2) << std::endl;
	}

	for (int i = 0; i < this->Face_num; ++i)
	{
		f << "f " << (this->F(i, 0) + 1) << " " << (this->F(i, 1) + 1) << " " << (this->F(i, 2) + 1) << std::endl;
	}
	f.close();
	std::cout << "Save to MANO_HandModel.obj success\n";
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