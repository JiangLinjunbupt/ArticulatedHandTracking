#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <limits>

#define M_PI 3.14159265358979323846

const int num_fingers = 5;

const int d = 3;
const int upper_bound_num_rendered_outline_points = 5000;
const int num_thetas = 29;
const int upper_bound_num_sensor_points_Realsense = 76800;                   //320*240
const int upper_bound_num_sensor_points_Kinect = 54272;                   //256*212

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