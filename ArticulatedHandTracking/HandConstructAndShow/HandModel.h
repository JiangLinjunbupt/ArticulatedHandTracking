#pragma once
#include <Eigen/Dense>   //包含所有普通的矩阵函数
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/Core>

#include <vector>
#include <iostream>
#include <string>
#include <fstream>

class HandModel
{
public:
	//这些都是从文件中读出来的, 只暴露face
	Eigen::MatrixXi F;
	unsigned int *F_array;

	int Joints_num;
	int Vertex_num;
	int Face_num;

	//计算出来的值
	Eigen::MatrixXf V_Final;   //经过pose和trans之后的某个姿态下的顶点
	float *V_Final_array;

	Eigen::MatrixXf J_Final;   //经过pose和trans之后某个姿态下的关节点

private:
	//控制手型的参数
	//这些是控制形状的值
	Eigen::Vector3f trans;
	Eigen::VectorXf pose;
	Eigen::VectorXf betas;


	const int Num_betas = 10;
	const int Num_WristPose = 3;
	const int Num_FingerPose = 6;   //控制手指状态的

	//这些是计算得到的输出
	Eigen::VectorXf Full_Hand_Params;
	std::map<int, int> Parent;
	Eigen::MatrixXf V_shaped;  //自然状态下，经过形状变换(shape blend)后的顶点
	Eigen::MatrixXf J_shaped;
	Eigen::MatrixXf V_posed;   //自然状态下，经过姿态变换(pose blend 或者 corrective blend shape)之后的顶点

	const int Num_WristParams = 3;
	const int Num_FingerParams = 45;
	bool want_shapemodel = true;
	bool change_shape = false;


	//这些都是从文件中读出来的
	Eigen::MatrixXf J;
	Eigen::SparseMatrix<float> J_regressor;
	Eigen::MatrixXf Hands_coeffs;
	Eigen::MatrixXf Hands_components;
	Eigen::VectorXf Hands_mean;
	Eigen::MatrixXf Kintree_table;
	std::vector<Eigen::MatrixXf> Posedirs;
	std::vector<Eigen::MatrixXf> Shapedirs;
	Eigen::MatrixXf V_template;
	Eigen::MatrixXf Weights;


public:
	HandModel();
	~HandModel() {};

	//更新顶点和关节点的函数
	void UpdataModel();
	void set_Shape_Params(const Eigen::VectorXf &shape_params)
	{
		for (int i = 0; i < Num_betas; ++i)
		{
			this->betas[i] = shape_params[i];
		}
		this->change_shape = true;
	}
	void set_Pose_Params(const Eigen::VectorXf &pose_params)
	{
		for (int i = 0; i < (Num_WristPose + Num_FingerPose); ++i)
		{
			this->pose[i] = pose_params[i];
		}
	}
	void set_trans_Params(float x, float y, float z) { this->trans[0] = x; this->trans[1] = y; this->trans[2] = z; }

	void serializeModel()
	{
		for (int i = 0; i < this->Vertex_num; ++i)
		{
			this->V_Final_array[i * 3 + 0] = this->V_Final(i, 0) * 10.0f;
			this->V_Final_array[i * 3 + 1] = this->V_Final(i, 1) * 10.0f;
			this->V_Final_array[i * 3 + 2] = this->V_Final(i, 2) * 10.0f;
		}
	}
	void Save_as_obj();

private:
	void LoadModel();
	void Load_J(const char* filename);
	void Load_J_regressor(const char* filename);
	void Load_F(const char* filename);
	void Load_Hands_coeffs(const char* filename);
	void Load_Hands_components(const char* filename);
	void Load_Hands_mean(const char* filename);
	void Load_Kintree_table(const char* filename);
	void Load_Posedirs(const char* filename);
	void Load_Shapedirs(const char* filename);
	void Load_V_template(const char* filename);
	void Load_Weights(const char* filename);


	//更新顶点和关节点的函数
	void Updata_V_rest();
	void ShapeSpaceBlend();
	void PoseSpaceBlend();

	void LBS_Updata();

private:

	//两个工具函数
	Eigen::Matrix3f rodrigues(float x, float y, float z)
	{
		Eigen::Vector3f rotate_vec(x, y, z);
		Eigen::Matrix3f rotate_mat;

		float angle = rotate_vec.norm();
		rotate_vec.normalize();

		rotate_mat = Eigen::AngleAxisf(angle, rotate_vec);
		return rotate_mat;
	}

	std::vector<float> lortmin(const Eigen::VectorXf &full_pose)
	{
		std::vector<float> result(3 * full_pose.size());  //实际计算是 9* (full_pose->size()/3)

		Eigen::Matrix3f rotate_mat;
		Eigen::Vector3f rotate_vec;

		for (int i = 0; i < full_pose.size() / 3; ++i)
		{
			rotate_vec<<full_pose[i * 3 + 0], full_pose[i * 3 + 1], full_pose[i * 3 + 2];

			float angle = rotate_vec.norm();
			rotate_vec.normalize();

			rotate_mat = Eigen::AngleAxisf(angle, rotate_vec);


			result[i * 9 + 0] = rotate_mat(0, 0) - 1; result[i * 9 + 1] = rotate_mat(0, 1);     result[i * 9 + 2] = rotate_mat(0, 2);
			result[i * 9 + 3] = rotate_mat(1, 0);     result[i * 9 + 4] = rotate_mat(1, 1) - 1; result[i * 9 + 5] = rotate_mat(1, 2);
			result[i * 9 + 6] = rotate_mat(2, 0);      result[i * 9 + 7] = rotate_mat(2, 1);    result[i * 9 + 8] = rotate_mat(2, 2) - 1;

		}

		return result;
	}
};