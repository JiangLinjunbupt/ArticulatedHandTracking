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
	Eigen::MatrixXf J;
	Eigen::SparseMatrix<float> J_regressor;
	Eigen::MatrixXf F;
	Eigen::MatrixXf Hands_coeffs;
	Eigen::MatrixXf Hands_components;
	Eigen::VectorXf Hands_mean;
	Eigen::MatrixXf Kintree_table;
	std::vector<Eigen::MatrixXf> Posedirs;
	std::vector<Eigen::MatrixXf> Shapedirs;
	Eigen::MatrixXf V_template;
	Eigen::MatrixXf Weights;

	int Joints_num;
	int Vertex_num;
	int Face_num;


public:
	HandModel();
	~HandModel() {};

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

};