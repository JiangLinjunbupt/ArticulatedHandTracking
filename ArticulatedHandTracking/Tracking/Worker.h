#pragma once
#include"Types.h"
#include"HandModel.h"
#include"CorrespondFind.h"
#include"Kalman.h"
#include<queue>

using namespace std;

class Worker
{
private:
	struct Settings {
		float Fitting_weight = 1.8;
		float Fitting_2D_weight = 1.0f;

		float Pose_Damping_weight_For_SmallRotate = 100000;
		float Pose_Damping_weight_For_BigRotate = 200;
		float Pose_MaxMinLimit_weight = 100000;
		float Pose_Pcalimit_weight = 200;

		float Shape_Damping_weight = 5000;

		float Pose_Difference_Var_weight = 1000;
		float Pose_Differnece_MaxMin_weight = 1000;

		float Temporal_FirstOrder_weight = 1;
		float Temporal_SecondOorder_weight = 1;

		float Temporal_finger_params_FirstOrder_weight = 100;
		float Temporal_finger_params_SecondOorder_weight = 100;

		float Collision_weight = 100.0f;

		int max_itr = 8;
		int frames_interval_between_measurements = 60;

		float track_fail_threshold = 8.0f;
	} _settings;
public:
	Settings*const settings = &_settings;

	float total_error = 0;
	bool track_success = false;

	Kalman* kalman;
private:
	Camera* camera;
	HandModel* handmodel;
	CorrespondFind* correspondfind;

	std::queue<Eigen::Matrix<float, 16, 3>> temporal_Joint_Position;
	std::queue<Eigen::VectorXf> temporal_finger_params;

	int itr = 0;
	int total_itr = 0;

	int Pose_params_num;
	int Shape_params_num;
	int Total_params_num;

	bool Has_Glove = false;

	Eigen::VectorXf Glove_params;
	Eigen::VectorXf PoseParams_previous;
	Eigen::VectorXf PoseParams_previous_Glove;

	Eigen::VectorXf Params;

public:
	Worker(HandModel* _handmodel, CorrespondFind* _dataset, Camera* _camera) :handmodel(_handmodel), correspondfind(_dataset), camera(_camera)
	{
		init_worker();
		kalman = new Kalman(handmodel);
	}
	~Worker() {
		delete camera;
		delete handmodel;
		delete correspondfind;
		delete kalman;
	}
	void ResetShapeParmas()
	{
		this->Params.head(Shape_params_num).setZero();
		this->kalman->ReSet();
	}
	void init_worker();
	void init_Params()
	{
		total_error = 0;
		itr = 0;
		total_itr = 0;
		track_success = false;

		Pose_params_num = handmodel->Pose_params_num;
		Shape_params_num = handmodel->Shape_params_num;
		Total_params_num = Pose_params_num + Shape_params_num;

		Glove_params = Eigen::VectorXf::Zero(Pose_params_num);
		PoseParams_previous = Eigen::VectorXf::Zero(Pose_params_num);
		PoseParams_previous_Glove = Eigen::VectorXf::Zero(Pose_params_num);

		Params = Eigen::VectorXf::Zero(Total_params_num);
	}
	void tracker();
	void SetGloveParams(Eigen::VectorXf pose, bool track);
	void SetTemporalJointPosition()
	{
		if (total_error < settings->track_fail_threshold)
		{
			track_success = true;
			this->PoseParams_previous = Params.tail(Pose_params_num);
			this->PoseParams_previous_Glove = Glove_params;

			if (temporal_Joint_Position.size() == 2)
			{
				temporal_Joint_Position.pop();
				temporal_Joint_Position.push(handmodel->J_Final);
			}
			else
			{
				temporal_Joint_Position.push(handmodel->J_Final);
			}

			if (temporal_finger_params.size() == 2)
			{
				temporal_finger_params.pop();
				temporal_finger_params.push(this->Params.tail(Pose_params_num - 6));
			}
			else
			{
				temporal_finger_params.push(this->Params.tail(Pose_params_num - 6));
			}
		}
		else
		{
			track_success = false;
			while (!temporal_Joint_Position.empty())
				temporal_Joint_Position.pop();

			while (!temporal_finger_params.empty())
				temporal_finger_params.pop();

			handmodel->set_Pose_Params(Glove_params);
			handmodel->UpdataModel();
		}
	}

private:

	void Fitting(LinearSystem& linear_system);
	void Fitting2D(LinearSystem& linear_system);
	void RigidOnly(LinearSystem& linear_system);
	void Damping(LinearSystem& linear_system);
	void MaxMinLimit(LinearSystem& linear_system);
	void PcaLimit(LinearSystem& linear_system);
	void GloveDifferenceMaxMinLimit(LinearSystem& linear_system);
	void GloveDifference_VarLimit(LinearSystem& linear_system);
	void TemporalLimit(LinearSystem& linear_system, bool first_order);
	void TemporalParamsLimit(LinearSystem& linear_system, bool first_order);
	void CollisionLimit(LinearSystem& linear_system);
	Eigen::VectorXf Solver(LinearSystem& linear_system);
};