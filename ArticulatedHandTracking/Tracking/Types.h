#pragma once
#include<pcl\point_types.h>
#include<pcl\filters\voxel_grid.h>
#include<pcl\filters\statistical_outlier_removal.h>
//必须放在最前面，不知道为什么放在opencv后面就会无法使用statistical_outlier_removal这个方法（幺蛾子事件）

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <limits>
#include <set>
#include <map>

#define M_PI 3.14159265358979323846

typedef Eigen::Matrix4f Matrix4;
typedef Eigen::Matrix<float, 3, 3> Matrix3;
typedef Eigen::Matrix<float, 2, 3> Matrix_2x3;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Matrix_3xN;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>  Matrix_MxN;
typedef Eigen::VectorXf VectorN;
typedef Eigen::Vector2f Vector2;
typedef Eigen::Vector3f Vector3;


/// Nan for the default type
inline float nan() { return (std::numeric_limits<float>::quiet_NaN)(); }
inline float inf() { return (std::numeric_limits<float>::max)(); }

/// Linear system lhs*x=rhs
struct LinearSystem {
	Matrix_MxN lhs; // J^T*J
	VectorN rhs; // J^T*r
	LinearSystem() {}
	LinearSystem(int n) {
		lhs = Matrix_MxN::Zero(n, n);
		rhs = VectorN::Zero(n);
	}
};

struct DataAndCorrespond
{
	Vector3 pointcloud;
	int pointcloud_idx;

	Vector3 correspond;
	int correspond_idx;

	bool is_match;
};

enum FingerType
{
	Index, Middle, Pinky, Ring, Thumb
};
struct Collision
{
	int id;
	bool root;

	Eigen::Vector3f Init_Center;
	Eigen::Vector3f Update_Center;

	float Radius;
	int joint_belong;
	FingerType fingerType;

};

enum RuntimeType
{
	REALTIME, SHAPE_VERIFY, Dataset_MSRA_14, Dataset_MSRA_15, Handy_teaser
};