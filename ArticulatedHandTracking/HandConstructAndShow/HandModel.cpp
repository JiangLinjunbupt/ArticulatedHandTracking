#include"HandModel.h"

HandModel::HandModel()
{
	this->LoadModel();
	std::cout << "Load Model success "<<std::endl;
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
	this->Load_Hands_coeffs(hands_coeffs_filename.c_str());
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
	this->F = Eigen::MatrixXf::Zero(this->Face_num, 3);
	for (int i = 0; i < this->Face_num; ++i)
	{
		f >> this->F(i, 0) >> this->F(i, 1) >> this->F(i, 2);
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
	this->Posedirs.resize(dim1);
	for (int d1 = 0; d1 < dim1; ++d1)
	{
		Eigen::MatrixXf tem = Eigen::MatrixXf::Zero(dim2, dim3);
		for (int d2 = 0; d2 < dim2; ++d2)
		{
			for (int d3 = 0; d3 < dim3; ++d3)
			{
				f >> tem(d2, d3);
			}
		}
		this->Posedirs[d1] = tem;
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
	this->Shapedirs.resize(dim1);
	for (int d1 = 0; d1 < dim1; ++d1)
	{
		Eigen::MatrixXf tem = Eigen::MatrixXf::Zero(dim2, dim3);
		for (int d2 = 0; d2 < dim2; ++d2)
		{
			for (int d3 = 0; d3 < dim3; ++d3)
			{
				f >> tem(d2, d3);
			}
		}
		this->Shapedirs[d1] = tem;
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
	std::cout << this->Weights << std::endl;
	std::cout << "Load Weights success\n";
}
#pragma endregion LoadFunctions