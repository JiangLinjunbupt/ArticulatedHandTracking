#include"Worker.h"


void Worker::init_worker()
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

	//Params = handmodel->Hands_mean;
	handmodel->set_Shape_Params(Params.head(Shape_params_num));
	handmodel->set_Pose_Params(Params.tail(Pose_params_num));
	handmodel->UpdataModel();

	kalman = new Kalman(handmodel);
}

void Worker::tracker()
{
	//初始化求解相关
	LinearSystem linear_system;
	linear_system.lhs = Eigen::MatrixXf::Zero(Total_params_num, Total_params_num);
	linear_system.rhs = Eigen::VectorXf::Zero(Total_params_num);


	this->Fitting(linear_system);

	if (itr >= 0.5* settings->max_itr)
		this->Fitting2D(linear_system);

	//这里可以开始更新kalman的海森矩阵
	if (itr>(settings->max_itr - 2)
		&& total_error < settings->track_fail_threshold
		&& !kalman->judgeFitted())  //尽量在更新的迭代后几次，并且跟踪成功的时候更新
		kalman->Set_measured_hessian(linear_system);

	this->MaxMinLimit(linear_system);
	this->PcaLimit(linear_system);
	if (Has_Glove)
	{
		//this->GloveDifferenceMaxMinLimit(linear_system);
		//this->GloveDifference_VarLimit(linear_system);
	}
	this->TemporalLimit(linear_system, true);
	this->TemporalLimit(linear_system, false);
	kalman->track(linear_system, Params.head(Shape_params_num));
	this->CollisionLimit(linear_system);
	this->Damping(linear_system);
	if (itr < 2)
		this->RigidOnly(linear_system);  //这个一定要放最后


	Eigen::VectorXf solution = this->Solver(linear_system);

	for (int k = 0; k < Total_params_num; k++)
		Params[k] += solution(k);


	//这里可以通过total_itr迭代次数判断，通过kalman更新形状参数的间隔
	if (total_itr > 2 * settings->frames_interval_between_measurements
		&& total_itr % settings->frames_interval_between_measurements == 0
		&& total_error < settings->track_fail_threshold
		&& !kalman->judgeFitted())
		kalman->Update_estimate(Params.head(Shape_params_num));

	//std::cout << "第  " << itr << " 次迭代的参数为 : " << std::endl << this->Params << std::endl;
	handmodel->set_Shape_Params(Params.head(Shape_params_num));
	handmodel->set_Pose_Params(Params.tail(Pose_params_num));
	handmodel->UpdataModel();
	itr++;
	total_itr++;

	if (itr == settings->max_itr)
	{
		SetTemporalJointPosition();
		//kalman->ShowHessian();
	}
}

void Worker::Fitting(LinearSystem& linear_system)
{
	int NumofCorrespond = correspondfind->num_matched_correspond;
	Eigen::VectorXf e = Eigen::VectorXf::Zero(NumofCorrespond * 3);
	Eigen::MatrixXf J = Eigen::MatrixXf::Zero(NumofCorrespond * 3, Total_params_num);

	total_error = 0;
	//临时变量
	Eigen::MatrixXf shape_jacob, pose_jacob;
	int count = 0;

	for (int i = 0; i < correspondfind->correspond.size(); ++i)
	{
		if (correspondfind->correspond[i].is_match)
		{
			int v_id = correspondfind->correspond[i].correspond_idx;

			e(count * 3 + 0) = correspondfind->correspond[i].pointcloud(0) - correspondfind->correspond[i].correspond(0);
			e(count * 3 + 1) = correspondfind->correspond[i].pointcloud(1) - correspondfind->correspond[i].correspond(1);
			e(count * 3 + 2) = correspondfind->correspond[i].pointcloud(2) - correspondfind->correspond[i].correspond(2);

			float e_sqrt = sqrt(pow(e(count * 3 + 0), 2) + pow(e(count * 3 + 1), 2) + pow(e(count * 3 + 2), 2));
			total_error += e_sqrt;
			//这里使用的是Reweighted Least squard error
			//参考：
			//https://www.cs.bgu.ac.il/~mcv172/wiki.files/Lec5.pdf （主要）
			//https://blog.csdn.net/baidu_17640849/article/details/71155537  （辅助）
			//weight = s/(e^2 + s^2)


			//s越大，对异常值的容忍越大
			float s = 100;
			float weight = 1;
			weight = s / (e_sqrt + s);

			e(count * 3 + 0) *= weight;
			e(count * 3 + 1) *= weight;
			e(count * 3 + 2) *= weight;

			handmodel->Shape_jacobain(shape_jacob, v_id);
			handmodel->Pose_jacobain(pose_jacob, v_id);

			J.block(count * 3, 0, 3, Shape_params_num) = weight * shape_jacob;
			J.block(count * 3, Shape_params_num, 3, Pose_params_num) = weight*pose_jacob;

			++count;
		}
	}
	total_error = total_error / count;
	std::cout << "第  " << itr << "  次迭代的误差为  ： " << total_error << std::endl;

	Eigen::MatrixXf JtJ = J.transpose()*J;
	Eigen::VectorXf JTe = J.transpose()*e;

	linear_system.lhs += settings->Fitting_weight*JtJ;
	linear_system.rhs += settings->Fitting_weight*JTe;
}

void Worker::Fitting2D(LinearSystem& linear_system)
{
	vector<pair<Eigen::Matrix2Xf, Eigen::Vector2f>> JacobVector_2D;

	int width = camera->width();
	int height = camera->height();

	int visiblePointSize = handmodel->V_Visible_2D.size();

	Eigen::MatrixXf shape_jacob, pose_jacob;

	for (int i = 0; i < visiblePointSize; ++i)
	{
		int idx = handmodel->V_Visible_2D[i].second;

		Eigen::Vector3f pixel_3D_pos(handmodel->V_Final(idx, 0), handmodel->V_Final(idx, 1), handmodel->V_Final(idx, 2));
		Eigen::Vector2i pixel_2D_pos(handmodel->V_Visible_2D[i].first);

		int cloest_idx = correspondfind->my_dataframe->idxs_image[pixel_2D_pos(1) * width + pixel_2D_pos(0)];
		Eigen::Vector2i pixel_2D_cloest;
		pixel_2D_cloest << cloest_idx%width, cloest_idx / width;

		float closet_distance = (pixel_2D_cloest - pixel_2D_pos).norm();

		if (closet_distance > 0)
		{
			//计算 J 和 e
			pair<Eigen::Matrix2Xf, Eigen::Vector2f> J_and_e;

			//先算e
			Eigen::Vector2f e;
			e(0) = (float)pixel_2D_cloest(0) - (float)pixel_2D_pos(0);
			e(1) = (float)pixel_2D_cloest(1) - (float)pixel_2D_pos(1);

			J_and_e.second = e;

			//再计算J
			Eigen::Matrix<float, 2, 3> J_perspective = camera->projection_jacobian(pixel_3D_pos);
			Eigen::MatrixXf J_3D = Eigen::MatrixXf::Zero(3, Total_params_num);

			handmodel->Shape_jacobain(shape_jacob, idx);
			handmodel->Pose_jacobain(pose_jacob, idx);

			J_3D.block(0, 0, 3, Shape_params_num) = shape_jacob;
			J_3D.block(0, Shape_params_num, 3, Pose_params_num) = pose_jacob;

			J_and_e.first = J_perspective * J_3D;

			JacobVector_2D.push_back(J_and_e);
		}

	}

	int size = JacobVector_2D.size();

	if (size > 0)
	{
		Eigen::MatrixXf J_2D = Eigen::MatrixXf::Zero(2 * size, this->Total_params_num);
		Eigen::VectorXf e_2D = Eigen::VectorXf::Zero(2 * size);

		for (int i = 0; i < size; ++i)
		{
			J_2D.block(i * 2, 0, 2, Total_params_num) = JacobVector_2D[i].first;
			e_2D.segment(i * 2, 2) = JacobVector_2D[i].second;
		}

		linear_system.lhs += settings->Fitting_2D_weight * J_2D.transpose() * J_2D;
		linear_system.rhs += settings->Fitting_2D_weight * J_2D.transpose() * e_2D;
	}

}

void Worker::MaxMinLimit(LinearSystem& linear_system)
{
	Eigen::MatrixXf J_limit = Eigen::MatrixXf::Zero(this->Total_params_num, this->Total_params_num);
	Eigen::VectorXf e_limit = Eigen::VectorXf::Zero(this->Total_params_num);


	for (int i = 0; i < 45; ++i)
	{
		int index = this->Shape_params_num + 6 + i;

		float Params_Max = handmodel->Hands_Pose_Max[i];
		float Params_Min = handmodel->Hands_Pose_Min[i];

		if (this->Params[index] > Params_Max) {
			e_limit(index) = (Params_Max - this->Params[index]) - std::numeric_limits<float>::epsilon();
			J_limit(index, index) = 1;
		}
		else if (this->Params[index] < Params_Min) {
			e_limit(index) = (Params_Min - this->Params[index]) + std::numeric_limits<float>::epsilon();
			J_limit(index, index) = 1;
		}
		else {
			e_limit(index) = 0;
			J_limit(index, index) = 0;
		}
	}

	Eigen::MatrixXf JtJ = J_limit.transpose()*J_limit;
	Eigen::VectorXf JTe = J_limit.transpose()*e_limit;

	linear_system.lhs += settings->Pose_MaxMinLimit_weight*JtJ;
	linear_system.rhs += settings->Pose_MaxMinLimit_weight*JTe;
}

void Worker::PcaLimit(LinearSystem& linear_system)
{
	Eigen::MatrixXf JtJ = Eigen::MatrixXf::Zero(this->Total_params_num, this->Total_params_num);
	Eigen::VectorXf JTe = Eigen::VectorXf::Zero(this->Total_params_num);


	//{
	//	//这部分应该是随着kalman的整合变化的，无论是 均值 还是 方差
	//	Eigen::MatrixXf shape_Sigma = Eigen::MatrixXf::Identity(this->Shape_params_num, this->Shape_params_num);
	//	shape_Sigma.diagonal() = handmodel->Hand_Shape_var;
	//	Eigen::MatrixXf InvShape_Sigma = shape_Sigma.inverse();

	//	Eigen::MatrixXf J_shape_PCA = InvShape_Sigma;
	//	Eigen::MatrixXf e_shape_PCA = -1 * InvShape_Sigma * this->Params.head(this->Shape_params_num);

	//	JtJ.block(0, 0, this->Shape_params_num, this->Shape_params_num) = settings->Shape_PcaLimit_weight * J_shape_PCA.transpose() * J_shape_PCA;
	//	JTe.head(this->Shape_params_num) = settings->Shape_PcaLimit_weight * J_shape_PCA.transpose() * e_shape_PCA;
	//}


	{
		Eigen::VectorXf Params_fingers = this->Params.tail(this->Pose_params_num - 6);
		Eigen::VectorXf Params_fingers_Minus_Mean = Params_fingers - handmodel->Hands_mean;

		Eigen::MatrixXf P = handmodel->Hands_components.transpose();


		Eigen::MatrixXf pose_Sigma = Eigen::MatrixXf::Identity(this->Pose_params_num - 6, this->Pose_params_num - 6);
		pose_Sigma.diagonal() = handmodel->Hand_Pose_var;
		Eigen::MatrixXf Invpose_Sigma = pose_Sigma.inverse();


		Eigen::MatrixXf J_Pose_Pca = Invpose_Sigma * P;
		Eigen::VectorXf e_Pose_Pca = -1 * Invpose_Sigma*P*Params_fingers_Minus_Mean;

		JtJ.block(this->Shape_params_num + 6, this->Shape_params_num + 6, this->Pose_params_num - 6, this->Pose_params_num - 6) = settings->Pose_Pcalimit_weight*J_Pose_Pca.transpose()*J_Pose_Pca;
		JTe.tail(this->Pose_params_num - 6) = settings->Pose_Pcalimit_weight*J_Pose_Pca.transpose()*e_Pose_Pca;
	}

	linear_system.lhs += JtJ;
	linear_system.rhs += JTe;
}

void Worker::Damping(LinearSystem& linear_system)
{
	Eigen::MatrixXf D = Eigen::MatrixXf::Identity(Total_params_num, Total_params_num);

	D.block(0, 0, this->Shape_params_num, this->Shape_params_num) = settings->Shape_Damping_weight * Eigen::MatrixXf::Identity(this->Shape_params_num, this->Shape_params_num);

	for (int i = 0; i < 48; ++i)
	{
		if (i == 3 || i == 6 || i == 7 || i == 9 || i == 10 ||
			i == 12 || i == 15 || i == 16 || i == 18 || i == 19 ||
			i == 21 || i == 24 || i == 25 || i == 27 || i == 28 ||
			i == 30 || i == 33 || i == 34 || i == 36 || i == 37 ||
			i == 39 || i == 42 || i == 44 || i == 45 || i == 47)
		{
			int index = this->Shape_params_num + 3 + i;

			D(index, index) = settings->Pose_Damping_weight_For_BigRotate;
		}

		if (i == 4 || i == 5 || i == 8 || i == 11 ||
			i == 13 || i == 14 || i == 17 || i == 20 ||
			i == 22 || i == 23 || i == 26 || i == 29 ||
			i == 31 || i == 32 || i == 35 || i == 38 ||
			i == 40 || i == 41 || i == 43 || i == 46)
		{
			int index = this->Shape_params_num + 3 + i;

			D(index, index) = settings->Pose_Damping_weight_For_SmallRotate;
		}
	}
	linear_system.lhs += D;
}

void Worker::RigidOnly(LinearSystem& linear_system)
{
	for (int row = 0; row < this->Shape_params_num; ++row)
	{
		linear_system.lhs.row(row).setZero();
		linear_system.rhs.row(row).setZero();
	}
	for (int row = this->Shape_params_num + 6; row < this->Total_params_num; ++row)
	{
		linear_system.lhs.row(row).setZero();
		linear_system.rhs.row(row).setZero();
	}


	for (int col = 0; col < this->Shape_params_num; ++col) linear_system.lhs.col(col).setZero();
	for (int col = this->Shape_params_num + 6; col < this->Total_params_num; ++col) linear_system.lhs.col(col).setZero();

}

void Worker::GloveDifferenceMaxMinLimit(LinearSystem& linear_system)
{
	Eigen::MatrixXf J_limit = Eigen::MatrixXf::Zero(this->Total_params_num, this->Total_params_num);
	Eigen::VectorXf e_limit = Eigen::VectorXf::Zero(this->Total_params_num);

	for (int i = 0; i < 45; ++i)
	{
		int index = this->Shape_params_num + 6 + i;
		int gloveParams_idx = i + 6;

		float Params_Max = handmodel->Glove_Difference_Max[i] + this->Glove_params[gloveParams_idx];
		float Params_Min = handmodel->Glove_Difference_Min[i] + this->Glove_params[gloveParams_idx];


		if (this->Params[index] > Params_Max) {
			e_limit(index) = (Params_Max - this->Params[index]) - std::numeric_limits<float>::epsilon();
			J_limit(index, index) = 1;
		}
		else if (this->Params[index] < Params_Min) {
			e_limit(index) = (Params_Min - this->Params[index]) + std::numeric_limits<float>::epsilon();
			J_limit(index, index) = 1;
		}
		else {
			e_limit(index) = 0;
			J_limit(index, index) = 0;
		}
	}

	Eigen::MatrixXf JtJ = J_limit.transpose()*J_limit;
	Eigen::VectorXf JTe = J_limit.transpose()*e_limit;

	linear_system.lhs += settings->Pose_Differnece_MaxMin_weight*JtJ;
	linear_system.rhs += settings->Pose_Differnece_MaxMin_weight*JTe;
}

void Worker::GloveDifference_VarLimit(LinearSystem& linear_system)
{
	Eigen::MatrixXf JtJ = Eigen::MatrixXf::Zero(this->Total_params_num, this->Total_params_num);
	Eigen::VectorXf JTe = Eigen::VectorXf::Zero(this->Total_params_num);

	//这里我考虑对方差加一个测量噪声
	float Var_noise = 0.2f;

	{
		Eigen::VectorXf Params_fingers = this->Params.tail(this->Pose_params_num - 6);
		Eigen::VectorXf Params_glove_fingers = this->Glove_params.tail(this->Pose_params_num - 6);

		Eigen::VectorXf Params_Difference_Minus_Mean = Params_fingers - Params_glove_fingers - handmodel->Glove_Difference_mean;

		Eigen::MatrixXf poseDifference_Sigma = Eigen::MatrixXf::Identity(this->Pose_params_num - 6, this->Pose_params_num - 6);
		poseDifference_Sigma.diagonal() = handmodel->Glove_Difference_Var + Var_noise * Eigen::VectorXf::Ones(this->Pose_params_num - 6);

		{
			//这里观察到实际中小拇指的限制有些过大了，因此这里考虑降低小拇指的方差
			for (int i = 18; i < 20; ++i)
			{
				poseDifference_Sigma(i, i) = 500;
			}
		}
		Eigen::MatrixXf InvposeDifference_Sigma = poseDifference_Sigma.inverse();


		Eigen::MatrixXf J_PoseDifference_Pca = InvposeDifference_Sigma;
		Eigen::VectorXf e_PoseDifference_Pca = -1 * InvposeDifference_Sigma*Params_Difference_Minus_Mean;


		//这里根据手部骨架计算朝向，适当调整weight
		{
			/*
			参数说明：
			joint[0]  ----- wrist
			joint[1~3]  ---------食指
			joint[4~6]  -------中指
			joint[7~9]  -----小指
			joint[10~12]   ------无名指
			joint[13~15]   ------大拇指
			*/
			Eigen::RowVector3f LightDir(0, 0, 1);
			float fixed_weight = 0.8f;

			//食指
			{

				Eigen::RowVector3f Index_joint_direction_1 = handmodel->J_Final.row(2) - handmodel->J_Final.row(1);
				Eigen::RowVector3f Index_joint_direction_2 = handmodel->J_Final.row(3) - handmodel->J_Final.row(2);

				Index_joint_direction_1.normalize();
				Index_joint_direction_2.normalize();

				float weight1 = pow(Index_joint_direction_1.dot(LightDir), 2) + fixed_weight;
				float weight2 = pow(Index_joint_direction_2.dot(LightDir), 2) + fixed_weight;

				J_PoseDifference_Pca.block(0, 0, 3, this->Pose_params_num - 6) *= weight1;
				e_PoseDifference_Pca.segment(0, 3) *= weight1;

				J_PoseDifference_Pca.block(3, 0, 6, this->Pose_params_num - 6) *= weight2;
				e_PoseDifference_Pca.segment(3, 6) *= weight2;
			}

			//中指
			{
				Eigen::RowVector3f Middle_joint_direction_1 = handmodel->J_Final.row(5) - handmodel->J_Final.row(4);
				Eigen::RowVector3f Middle_joint_direction_2 = handmodel->J_Final.row(6) - handmodel->J_Final.row(5);

				Middle_joint_direction_1.normalize();
				Middle_joint_direction_2.normalize();


				float weight1 = pow(Middle_joint_direction_1.dot(LightDir), 2) + fixed_weight;
				float weight2 = pow(Middle_joint_direction_2.dot(LightDir), 2) + fixed_weight;

				J_PoseDifference_Pca.block(9, 0, 3, this->Pose_params_num - 6) *= weight1;
				e_PoseDifference_Pca.segment(9, 3) *= weight1;

				J_PoseDifference_Pca.block(12, 0, 6, this->Pose_params_num - 6) *= weight2;
				e_PoseDifference_Pca.segment(12, 6) *= weight2;
			}

			//小指
			{
				Eigen::RowVector3f Pinky_joint_direction_1 = handmodel->J_Final.row(8) - handmodel->J_Final.row(7);
				Eigen::RowVector3f Pinky_joint_direction_2 = handmodel->J_Final.row(9) - handmodel->J_Final.row(8);

				Pinky_joint_direction_1.normalize();
				Pinky_joint_direction_2.normalize();


				float weight1 = pow(Pinky_joint_direction_1.dot(LightDir), 2) + fixed_weight;
				float weight2 = pow(Pinky_joint_direction_2.dot(LightDir), 2) + fixed_weight;

				J_PoseDifference_Pca.block(18, 0, 3, this->Pose_params_num - 6) *= weight1;
				e_PoseDifference_Pca.segment(18, 3) *= weight1;

				J_PoseDifference_Pca.block(21, 0, 6, this->Pose_params_num - 6) *= weight2;
				e_PoseDifference_Pca.segment(21, 6) *= weight2;
			}

			//无名指
			{
				Eigen::RowVector3f Ring_joint_direction_1 = handmodel->J_Final.row(11) - handmodel->J_Final.row(10);
				Eigen::RowVector3f Ring_joint_direction_2 = handmodel->J_Final.row(12) - handmodel->J_Final.row(11);

				Ring_joint_direction_1.normalize();
				Ring_joint_direction_2.normalize();

				float weight1 = pow(Ring_joint_direction_1.dot(LightDir), 2) + fixed_weight;
				float weight2 = pow(Ring_joint_direction_2.dot(LightDir), 2) + fixed_weight;

				J_PoseDifference_Pca.block(27, 0, 3, this->Pose_params_num - 6) *= weight1;
				e_PoseDifference_Pca.segment(27, 3) *= weight1;

				J_PoseDifference_Pca.block(30, 0, 6, this->Pose_params_num - 6) *= weight2;
				e_PoseDifference_Pca.segment(30, 6) *= weight2;
			}

			//大拇指
			{
				Eigen::RowVector3f Thumb_joint_direction_1 = handmodel->J_Final.row(14) - handmodel->J_Final.row(13);
				Thumb_joint_direction_1.normalize();

				float weight = pow(Thumb_joint_direction_1.dot(LightDir), 2) + fixed_weight;

				J_PoseDifference_Pca.block(36, 0, 9, this->Pose_params_num - 6) *= weight;
				e_PoseDifference_Pca.segment(36, 9) *= weight;
			}
		}

		JtJ.block(this->Shape_params_num + 6, this->Shape_params_num + 6, this->Pose_params_num - 6, this->Pose_params_num - 6) = settings->Pose_Difference_Var_weight*J_PoseDifference_Pca.transpose()*J_PoseDifference_Pca;
		JTe.tail(this->Pose_params_num - 6) = settings->Pose_Difference_Var_weight*J_PoseDifference_Pca.transpose()*e_PoseDifference_Pca;
	}

	linear_system.lhs += JtJ;
	linear_system.rhs += JTe;
}

void Worker::TemporalLimit(LinearSystem& linear_system, bool first_order)
{
	Eigen::MatrixXf J_Tem = Eigen::MatrixXf::Zero(3 * handmodel->Joints_num, Pose_params_num);
	Eigen::VectorXf e_Tem = Eigen::VectorXf::Zero(3 * handmodel->Joints_num);

	Eigen::MatrixXf joint_jacob;

	if (temporal_Joint_Position.size() == 2)
	{
		for (int i = 0; i < handmodel->Joints_num; ++i)
		{
			if (first_order)
			{
				e_Tem(i * 3 + 0) = temporal_Joint_Position.back()(i, 0) - handmodel->J_Final(i, 0);
				e_Tem(i * 3 + 1) = temporal_Joint_Position.back()(i, 1) - handmodel->J_Final(i, 1);
				e_Tem(i * 3 + 2) = temporal_Joint_Position.back()(i, 2) - handmodel->J_Final(i, 2);
			}
			else
			{
				e_Tem(i * 3 + 0) = 2 * temporal_Joint_Position.back()(i, 0) - temporal_Joint_Position.front()(i, 0) - handmodel->J_Final(i, 0);
				e_Tem(i * 3 + 1) = 2 * temporal_Joint_Position.back()(i, 1) - temporal_Joint_Position.front()(i, 1) - handmodel->J_Final(i, 1);
				e_Tem(i * 3 + 2) = 2 * temporal_Joint_Position.back()(i, 2) - temporal_Joint_Position.front()(i, 2) - handmodel->J_Final(i, 2);
			}

			handmodel->Joint_Pose_jacobain(joint_jacob, i);
			J_Tem.block(i * 3, 0, 3, Pose_params_num) = joint_jacob;
		}
	}

	if (first_order)
	{
		linear_system.lhs.block(Shape_params_num, Shape_params_num, Pose_params_num, Pose_params_num) += settings->Temporal_FirstOrder_weight*J_Tem.transpose()*J_Tem;
		linear_system.rhs.tail(Pose_params_num) += settings->Temporal_FirstOrder_weight*J_Tem.transpose()*e_Tem;
	}
	else
	{
		linear_system.lhs.block(Shape_params_num, Shape_params_num, Pose_params_num, Pose_params_num) += settings->Temporal_SecondOorder_weight*J_Tem.transpose()*J_Tem;
		linear_system.rhs.tail(Pose_params_num) += settings->Temporal_SecondOorder_weight*J_Tem.transpose()*e_Tem;
	}
}

void Worker::CollisionLimit(LinearSystem& linear_system)
{
	int NumOfCollision = handmodel->NumOfCollision;
	int CollisionSphereNum = handmodel->Collision_sphere.size();

	float fraction = 0.1f;
	Eigen::MatrixXf jacob_tmp = Eigen::MatrixXf::Zero(3, Pose_params_num);

	if (NumOfCollision > 0)
	{
		Eigen::MatrixXf J_collision = Eigen::MatrixXf::Zero(3 * NumOfCollision, Pose_params_num);
		Eigen::VectorXf e_collision = Eigen::VectorXf::Zero(3 * NumOfCollision);

		int count = 0;

		for (int i = 0; i < CollisionSphereNum; ++i)
		{
			for (int j = 0; j < CollisionSphereNum; ++j)
			{
				if (handmodel->Collision_Judge_Matrix(i, j) == 1) //发生碰撞，规则为 i 和 j 碰撞，则i 需要移动（后面有 j 和 i 碰撞，j需要移动）
				{
					Eigen::Vector3f dir_i_to_j;

					dir_i_to_j << handmodel->Collision_sphere[j].Update_Center - handmodel->Collision_sphere[i].Update_Center;
					dir_i_to_j.normalize();

					Eigen::Vector3f now_point, target_point;

					now_point = handmodel->Collision_sphere[i].Update_Center + handmodel->Collision_sphere[i].Radius*dir_i_to_j;
					target_point = handmodel->Collision_sphere[j].Update_Center - handmodel->Collision_sphere[j].Radius*dir_i_to_j;

					e_collision(count * 3 + 0) = fraction * (target_point(0) - now_point(0));
					e_collision(count * 3 + 1) = fraction * (target_point(1) - now_point(1));
					e_collision(count * 3 + 2) = fraction * (target_point(2) - now_point(2));

					Eigen::Vector3f now_point_Initpos = now_point - handmodel->Collision_sphere[i].Update_Center + handmodel->Collision_sphere[i].Init_Center;

					handmodel->CollisionPoint_Jacobian(jacob_tmp, handmodel->Collision_sphere[i].joint_belong, now_point_Initpos);

					J_collision.block(count * 3, 0, 3, Pose_params_num) = jacob_tmp;
					count++;
				}
			}
		}

		linear_system.lhs.block(Shape_params_num, Shape_params_num, Pose_params_num, Pose_params_num) += settings->Collision_weight * J_collision.transpose() * J_collision;
		linear_system.rhs.tail(Pose_params_num) += settings->Collision_weight * J_collision.transpose() * e_collision;
	}
}

Eigen::VectorXf Worker::Solver(LinearSystem& linear_system)
{
	///--- Solve for update dt = (J^T * J + D)^-1 * J^T * r

	//http://eigen.tuxfamily.org/dox/group__LeastSquares.html  for linear least square problem
	Eigen::VectorXf solution = linear_system.lhs.colPivHouseholderQr().solve(linear_system.rhs);

	///--- Check for NaN
	for (int i = 0; i<solution.size(); i++) {
		if (isnan(solution[i])) {
			std::cout << "-------------------------------------------------------------\n";
			std::cout << "!!!WARNING: NaN DETECTED in the solution!!! (skipping update)\n";
			std::cout << "-------------------------------------------------------------\n";
			return Eigen::VectorXf::Zero(solution.size());
		}

		if (isinf(solution[i])) {
			std::cout << "-------------------------------------------------------------\n";
			std::cout << "!!!WARNING: INF DETECTED in the solution!!! (skipping update)\n";
			std::cout << "-------------------------------------------------------------\n";
			return Eigen::VectorXf::Zero(solution.size());
		}
	}

	return solution;
}


/////////
void Worker::SetGloveParams(Eigen::VectorXf pose, bool track)
{
	itr = 0;
	Glove_params = pose;

	if (track)
	{
		//float glove_params_difference = (Glove_params - Params_previous_Glove).norm();
		//cout << "glove_params_difference : "<<glove_params_difference << endl;

		/*if (track_success)  Params = Params_previous;
		else Params = Glove_params;*/

		if (track_success)
		{
			float glove_params_difference = (Glove_params - PoseParams_previous_Glove).norm();
			//cout << "glove_params_difference : "<<glove_params_difference << endl;

			if (glove_params_difference < 20)
				Params.tail(Pose_params_num) = PoseParams_previous;
		}
		else
		{
			Params.tail(Pose_params_num) = Glove_params;
		}
	}
	else
		Params.tail(Pose_params_num) = Glove_params;


	//handmodel->set_Shape_Params(Params.head(Shape_params_num));
	handmodel->set_Pose_Params(Params.tail(Pose_params_num));
	handmodel->UpdataModel();

	Has_Glove = true;
}
