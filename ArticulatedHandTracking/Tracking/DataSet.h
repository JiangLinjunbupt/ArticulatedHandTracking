#pragma once
#include <assert.h>
#include <Eigen/Dense>   //包含所有普通的矩阵函数
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/Core>

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include"DataFrame.h"
#include"Camera.h"
#include<math.h>
#include<algorithm>

#include"DataFrame.h"
#include"DistanceTransform.h"
using namespace std;

class DataSet
{
public:
	DistanceTransform distance_transform;
	void FetchDataFrame(DataFrame& dataframe);

	int Joints_num;
	int Vertex_num;
	int Face_num;

	Eigen::MatrixXi F;
	Eigen::MatrixXf F_normal;

	//计算出来的值
	Eigen::MatrixXf V_Final;   //经过pose和trans之后的某个姿态下的顶点
	Eigen::MatrixXf V_Normal_Final;  //顶点的法向量
	Eigen::MatrixXf J_Final;   //经过pose和trans之后某个姿态下的关节点

	std::vector<std::pair<Eigen::Vector3f, int>> V_Visible;   //2D的可见点

	const int Shape_params_num = 10;
	const int Pose_params_num = 51;
	const int Global_position_num = 3;
	const int Wrist_pose_num = 3;
	const int Finger_pose_num = 45;

	std::map<int, int> Parent;
	std::map<int, int> Child;

	std::vector<Eigen::Matrix4f> Local_Coordinate;
private:
	Camera* camera;
	//控制手型的参数
	//这些是控制形状的值
	Eigen::VectorXf Shape_params;
	Eigen::VectorXf Pose_params;   //0-2控制手的全局位置，3-5控制手腕的旋转，6-50控制手势的状态；

	bool want_shapemodel = true;
	bool change_shape = false;


	//这些是计算得到的输出
	Eigen::MatrixXf V_shaped;  //自然状态下，经过形状变换(shape blend)后的顶点
	Eigen::MatrixXf J_shaped;
	Eigen::MatrixXf V_posed;   //自然状态下，经过姿态变换(pose blend 或者 corrective blend shape)之后的顶点


							   //这些都是从文件中读出来的
	Eigen::MatrixXf J;
	Eigen::SparseMatrix<float> J_regressor;
	Eigen::MatrixXf Kintree_table;
	std::vector<Eigen::MatrixXf> Posedirs;
	std::vector<Eigen::MatrixXf> Shapedirs;
	std::vector<Eigen::MatrixXf> Joint_Shapedir;  //这个是根据Joint_regressor和shape_dir算出来的
	Eigen::MatrixXf V_template;
	Eigen::MatrixXf Weights;

	Eigen::MatrixXf ShapeParams_Read;

	//与变换有关，存储雅各比矩阵
	std::vector<std::vector<int>> joint_relation;
public:
	DataSet(Camera *_camera);
	~DataSet() {};

	//更新顶点和关节点的函数
	void UpdataModel();
	void set_Shape_Params(int sub_idx) //sub_idx在0到30之间，一共31个数
	{
		sub_idx = abs(sub_idx);
		sub_idx = sub_idx % 31;

		this->Shape_params = (this->ShapeParams_Read.row(sub_idx)).transpose();
		this->change_shape = true;
	}
	void set_Pose_Params(const Eigen::VectorXf &pose_params)
	{
		assert(pose_params.size() == this->Pose_params.size());

		for (int i = 0; i < this->Pose_params_num; ++i) this->Pose_params[i] = pose_params[i];
	}
private:
	void LoadModel();
	void Load_J(const char* filename);
	void Load_J_regressor(const char* filename);
	void Load_F(const char* filename);
	void Load_Kintree_table(const char* filename);
	void Load_Posedirs(const char* filename);
	void Load_Shapedirs(const char* filename);
	void Load_V_template(const char* filename);
	void Load_Weights(const char* filename);
	void Load_ShapeParams(const char* filename);

	//更新顶点和关节点的函数
	void Updata_V_rest();
	void ShapeSpaceBlend();
	void PoseSpaceBlend();

	void LBS_Updata();
	void NormalUpdata();

	//存放LBSupdata中间值
	std::vector<Eigen::Matrix4f> result;

	std::vector<Eigen::Matrix4f> result2;
	std::vector<Eigen::Matrix4f> T;

	void Local_Coordinate_Init();
	void Local_Coordinate_Updata();

	void Trans_Matrix_Updata();  //与用于更新后续的四个中间变量，仅在形状改变时候更新
	std::vector<Eigen::Matrix4f> Trans_child_to_parent;
	std::vector<Eigen::Matrix4f> Trans_world_to_local;


private:

	//三个工具函数
	//旋转顺序，相对轴xyz的顺序
	Eigen::Matrix3f EularToRotateMatrix(float x, float y, float z)
	{
		Eigen::Matrix3f x_rotate = Eigen::Matrix3f::Identity();
		Eigen::Matrix3f y_rotate = Eigen::Matrix3f::Identity();
		Eigen::Matrix3f z_rotate = Eigen::Matrix3f::Identity();

		float sx = sin(x); float cx = cos(x);
		float sy = sin(y); float cy = cos(y);
		float sz = sin(z); float cz = cos(z);

		x_rotate(1, 1) = cx; x_rotate(1, 2) = -sx;
		x_rotate(2, 1) = sx; x_rotate(2, 2) = cx;

		y_rotate(0, 0) = cy; y_rotate(0, 2) = sy;
		y_rotate(2, 0) = -sy; y_rotate(2, 2) = cy;

		z_rotate(0, 0) = cz; z_rotate(0, 1) = -sz;
		z_rotate(1, 0) = sz; z_rotate(1, 1) = cz;


		return x_rotate*y_rotate*z_rotate;
	}
	std::vector<float> lortmin(const Eigen::VectorXf &finger_pose)
	{
		std::vector<float> result(3 * finger_pose.size());  //实际计算是 9* (full_pose->size()/3)
		Eigen::Matrix3f rotate_mat;

		for (int i = 1; i < this->Joints_num; ++i)
		{
			rotate_mat = EularToRotateMatrix(finger_pose[(i - 1) * 3 + 0], finger_pose[(i - 1) * 3 + 1], finger_pose[(i - 1) * 3 + 2]);
			Eigen::Matrix3f LocalCoordinateRotate = this->Local_Coordinate[i].block(0, 0, 3, 3);

			rotate_mat = LocalCoordinateRotate* rotate_mat*LocalCoordinateRotate.inverse();


			result[(i - 1) * 9 + 0] = rotate_mat(0, 0) - 1; result[(i - 1) * 9 + 1] = rotate_mat(0, 1);     result[(i - 1) * 9 + 2] = rotate_mat(0, 2);
			result[(i - 1) * 9 + 3] = rotate_mat(1, 0);     result[(i - 1) * 9 + 4] = rotate_mat(1, 1) - 1; result[(i - 1) * 9 + 5] = rotate_mat(1, 2);
			result[(i - 1) * 9 + 6] = rotate_mat(2, 0);      result[(i - 1) * 9 + 7] = rotate_mat(2, 1);    result[(i - 1) * 9 + 8] = rotate_mat(2, 2) - 1;

		}

		return result;
	}
};
