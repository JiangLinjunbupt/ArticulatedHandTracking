#include"OpenGLShow.h"
#include<iostream>

using namespace std;
using namespace Eigen;

int main()
{
	DisPlay_ReSult::init();

	HandModel *handmodel = new HandModel();


	Eigen::Vector3f trans = Eigen::Vector3f::Constant(0.5);
	Eigen::VectorXf beta = Eigen::VectorXf::Ones(10);
	Eigen::VectorXf pose = Eigen::VectorXf::Zero(9);
	pose << 0, 0, 0, -0.42671473, -0.85829819, -0.50662164, +1.97374622, -0.84298473, -1.29958491;

	handmodel->set_trans_Params(trans[0], trans[1], trans[2]);
	handmodel->set_Shape_Params(beta);
	handmodel->set_Pose_Params(pose);
	handmodel->UpdataModel();
	handmodel->Save_as_obj();

	DisPlay_ReSult::handmodel = handmodel;

	DisPlay_ReSult::init_BUFFER();
	DisPlay_ReSult::Display();

	//handmodel->Save_as_obj();
	return 0;

}