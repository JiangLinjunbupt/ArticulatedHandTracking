#include"HandModel.h"
#include<iostream>

using namespace std;
using namespace Eigen;

int main()
{
	HandModel *handmodel = new HandModel();


	Eigen::VectorXf pose = Eigen::VectorXf::Zero(9);
	pose << 0, 0, 0, -0.42671473, -0.85829819, -0.50662164, +1.97374622, -0.84298473, -1.29958491;
	handmodel->set_Pose_Params(pose);
	handmodel->UpdataModel();
	handmodel->Save_as_obj();


	return 0;

}