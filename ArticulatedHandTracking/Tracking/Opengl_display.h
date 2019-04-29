#pragma once
#include <GL\freeglut.h>
#include"CorrespondFind.h"
#include <time.h>
#define ANGLE_TO_RADIUS 3.1415/180.0;
#include<queue>
#include<vector>
#include"DataFrame.h"
#include<chrono>
#include"RealSenseSensor.h"
#include"HandModel.h"
#include"Worker.h"
#include"DataSet.h"
#include<chrono>

using namespace std::chrono;
namespace DS {

	bool init_error = false;

	float *GetGloveData = new float[48];

	bool show_handmodel = true;
	bool show_Dataset = false;
	Camera* _camera;
	RealSenseSensor* _mysensor;
	Worker* _worker;
	CorrespondFind* _correspondfind;
	HandModel* _handmodel;
	DataSet* _dataset;

	RuntimeType runtimeType = REALTIME;

	bool pause = false;
	bool track = false;

	//10，12，13，16，18（显著），19，22，23（显著），25，26（显著）
	int ShapeVerify_SubIdx = 26;

	struct Control {
		int x;
		int y;
		bool mouse_click;
		GLfloat rotx;
		GLfloat roty;
		double gx;
		double gy;
		double gz;

		Control() :x(0), y(0), rotx(0.0), roty(0.0), mouse_click(false),
			gx(0), gy(0), gz(0) {

		}
	};
	Control control;


	void UpdataHandUsingGlove();
	void UpdataDataSetUsingGlove();
	void RealTimeFunc();
	void ShapeVerfyFunc();
	//定义光照
	void light() {

		GLfloat light_position[] = { 1.0,1.0,1.0,0.0 };//1.0表示光源为点坐标x,y,z
		GLfloat white_light[] = { 1.0,1.0,1.0,1.0 };   //光源的颜色
		GLfloat lmodel_ambient[] = { 0.2,0.2,0.2,1.0 };//微弱环境光，使物体可见
		glShadeModel(GL_SMOOTH);//GL_SMOOTH

		glLightfv(GL_LIGHT0, GL_POSITION, light_position);//光源编号-7，光源特性，参数数据
		glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
		glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient); //指定全局的环境光，物体才能可见//*/

		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_DEPTH_TEST);

		//glEnable(GL_LIGHTING);
		//glEnable(GL_NORMALIZE);
		//// 定义太阳光源，它是一种白色的光源  
		//GLfloat sun_light_position[] = { 0.0f, 0.0f, 0.0f, 1.0f };
		//GLfloat sun_light_ambient[] = { 0.25f, 0.25f, 0.15f, 1.0f };
		//GLfloat sun_light_diffuse[] = { 0.7f, 0.7f, 0.55f, 1.0f };
		//GLfloat sun_light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

		//glLightfv(GL_LIGHT0, GL_POSITION, sun_light_position); //指定第0号光源的位置   
		//glLightfv(GL_LIGHT0, GL_AMBIENT, sun_light_ambient); //GL_AMBIENT表示各种光线照射到该材质上，  
		//													 //经过很多次反射后最终遗留在环境中的光线强度（颜色）  
		//glLightfv(GL_LIGHT0, GL_DIFFUSE, sun_light_diffuse); //漫反射后~~  
		//glLightfv(GL_LIGHT0, GL_SPECULAR, sun_light_specular);//镜面反射后~~~  

		//glEnable(GL_LIGHT0); //使用第0号光照   
	}


	//定义窗口大小重新调整
	void reshape(int width, int height) {

		GLfloat fieldOfView = 90.0f;
		glViewport(0, 0, (GLsizei)width, (GLsizei)height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(fieldOfView, (GLfloat)width / (GLfloat)height, 0.1, 500.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	//键盘鼠标响应函数
	void keyboardDown(unsigned char key, int x, int y) {

		if (key == 'q') exit(0);
		if (key == 'h') show_handmodel = !show_handmodel;
		if (key == 'd') show_Dataset = !show_Dataset;
		if (key == 'p') pause = !pause;
		if (key == 't') track = !track;
		if (key == 'r')
		{
			_worker->ResetShapeParmas();
		}
	}
	void mouseClick(int button, int state, int x, int y) {
		control.mouse_click = 1;
		control.x = x;
		control.y = y;
	}
	void mouseMotion(int x, int y) {
		control.rotx = (x - control.x)*0.05f;
		control.roty = (y - control.y)*0.05f;

		if (control.roty > 1.57) control.roty = 1.57;
		if (control.roty < -1.57) control.roty = -1.57;
		//cout<< control.rotx <<" " << control.roty << endl;
		glutPostRedisplay();
	}


	//一系列绘制函数
	void draw_DataSetVisiblePoint()
	{
		if (_dataset->V_Visible.size() > 0)
		{
			glDisable(GL_LIGHT0);
			glDisable(GL_LIGHTING);
			glPointSize(2);
			glBegin(GL_POINTS);
			glColor3f(0.0f, 1.0f, 0.0f);
			for (int i = 0; i < _dataset->V_Visible.size(); i++) {
				glVertex3d(_dataset->V_Visible[i].first.x(), _dataset->V_Visible[i].first.y(), _dataset->V_Visible[i].first.z());
			}
			glEnd();
		}
	}
	void draw_HandPointCloud()
	{
		if (_correspondfind->my_dataframe->handPointCloud.points.size() > 0)
		{
			glDisable(GL_LIGHT0);
			glDisable(GL_LIGHTING);
			glPointSize(2);
			glBegin(GL_POINTS);
			glColor3f(0.0f, 1.0f, 0.0f);
			for (int i = 0; i < _correspondfind->my_dataframe->handPointCloud.points.size(); i++) {
				glVertex3d(_correspondfind->my_dataframe->handPointCloud.points[i].x, _correspondfind->my_dataframe->handPointCloud.points[i].y, _correspondfind->my_dataframe->handPointCloud.points[i].z);
			}
			glEnd();
		}
	}
	void draw_HandModel()
	{
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		{
			glPushMatrix();

			GLfloat mat_ambient[] = { 0.05, 0.0, 0.0, 1.0 };
			GLfloat mat_diffuse[] = { 0.5, 0.4,0.4, 1.0 };
			GLfloat mat_specular[] = { 0.7, 0.04, 0.04, 1.0 };
			GLfloat no_shininess[] = { 0.78125 };

			glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
			glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
			glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
			glMaterialfv(GL_FRONT, GL_SHININESS, no_shininess);

			glBegin(GL_TRIANGLES);
			for (int i = 0; i < _handmodel->Face_num; ++i)
			{

				glNormal3f(_handmodel->F_normal(i, 0), _handmodel->F_normal(i, 1), _handmodel->F_normal(i, 2));

				glVertex3f(_handmodel->V_Final(_handmodel->F(i, 0), 0), _handmodel->V_Final(_handmodel->F(i, 0), 1), _handmodel->V_Final(_handmodel->F(i, 0), 2));
				glVertex3f(_handmodel->V_Final(_handmodel->F(i, 1), 0), _handmodel->V_Final(_handmodel->F(i, 1), 1), _handmodel->V_Final(_handmodel->F(i, 1), 2));
				glVertex3f(_handmodel->V_Final(_handmodel->F(i, 2), 0), _handmodel->V_Final(_handmodel->F(i, 2), 1), _handmodel->V_Final(_handmodel->F(i, 2), 2));
			}
			glEnd();
			glPopMatrix();            //弹出矩阵。
		}
	}
	void draw_HandModel_wireFrame()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		{
			glColor3f(0.0f, 1.0f, 0.0f);
			glLineWidth(1);
			glBegin(GL_LINES);
			for (int i = 0; i < _handmodel->Face_num; ++i)
			{
				glVertex3f(_handmodel->V_Final(_handmodel->F(i, 0), 0), _handmodel->V_Final(_handmodel->F(i, 0), 1), _handmodel->V_Final(_handmodel->F(i, 0), 2));
				glVertex3f(_handmodel->V_Final(_handmodel->F(i, 1), 0), _handmodel->V_Final(_handmodel->F(i, 1), 1), _handmodel->V_Final(_handmodel->F(i, 1), 2));

				glVertex3f(_handmodel->V_Final(_handmodel->F(i, 1), 0), _handmodel->V_Final(_handmodel->F(i, 1), 1), _handmodel->V_Final(_handmodel->F(i, 1), 2));
				glVertex3f(_handmodel->V_Final(_handmodel->F(i, 2), 0), _handmodel->V_Final(_handmodel->F(i, 2), 1), _handmodel->V_Final(_handmodel->F(i, 2), 2));

				glVertex3f(_handmodel->V_Final(_handmodel->F(i, 2), 0), _handmodel->V_Final(_handmodel->F(i, 2), 1), _handmodel->V_Final(_handmodel->F(i, 2), 2));
				glVertex3f(_handmodel->V_Final(_handmodel->F(i, 0), 0), _handmodel->V_Final(_handmodel->F(i, 0), 1), _handmodel->V_Final(_handmodel->F(i, 0), 2));
			}
			glEnd();
		}
	}
	void draw_CollisionSphere()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		for (int i = 0; i < _handmodel->Collision_sphere.size(); ++i)
		{
			if (_handmodel->Collision_sphere[i].root)
			{
				glColor3f(1.0, 0.0, 1.0);
				glPushMatrix();
				glTranslatef(_handmodel->Collision_sphere[i].Update_Center.x(), _handmodel->Collision_sphere[i].Update_Center.y(), _handmodel->Collision_sphere[i].Update_Center.z());
				glutSolidSphere(_handmodel->Collision_sphere[i].Radius, 10, 10);
				glPopMatrix();
			}
			else
			{
				glColor3f(0.0, 1.0, 1.0);
				glPushMatrix();
				glTranslatef(_handmodel->Collision_sphere[i].Update_Center.x(), _handmodel->Collision_sphere[i].Update_Center.y(), _handmodel->Collision_sphere[i].Update_Center.z());
				glutSolidSphere(_handmodel->Collision_sphere[i].Radius, 10, 10);
				glPopMatrix();
			}
		}
	}
	void show_Collision()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		int NumOfCollisionSphere = _handmodel->Collision_sphere.size();

		for (int i = 0; i < NumOfCollisionSphere; ++i)
		{
			for (int j = 0; j < NumOfCollisionSphere; ++j)
			{
				if (_handmodel->Collision_Judge_Matrix(i, j) == 1)
				{
					glColor3f(1.0, 0.0, 0.0);
					glPushMatrix();
					glTranslatef(_handmodel->Collision_sphere[i].Update_Center.x(), _handmodel->Collision_sphere[i].Update_Center.y(), _handmodel->Collision_sphere[i].Update_Center.z());
					glutSolidSphere(_handmodel->Collision_sphere[i].Radius, 10, 10);
					glPopMatrix();

					glColor3f(1.0, 0.0, 0.0);
					glPushMatrix();
					glTranslatef(_handmodel->Collision_sphere[j].Update_Center.x(), _handmodel->Collision_sphere[j].Update_Center.y(), _handmodel->Collision_sphere[j].Update_Center.z());
					glutSolidSphere(_handmodel->Collision_sphere[j].Radius, 10, 10);
					glPopMatrix();
				}
			}
		}
	}
	void draw_DatasetModel()
	{
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		{
			glPushMatrix();

			GLfloat mat_ambient[] = { 0.0, 0.05,0.0, 1.0 };
			GLfloat mat_diffuse[] = { 0.4, 0.5, 0.4, 1.0 };
			GLfloat mat_specular[] = { 0.04, 0.7, 0.04, 1.0 };
			GLfloat no_shininess[] = { 0.78125 };

			glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
			glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
			glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
			glMaterialfv(GL_FRONT, GL_SHININESS, no_shininess);

			glBegin(GL_TRIANGLES);
			for (int i = 0; i < _dataset->Face_num; ++i)
			{

				glNormal3f(_dataset->F_normal(i, 0), _dataset->F_normal(i, 1), _dataset->F_normal(i, 2));

				glVertex3f(_dataset->V_Final(_dataset->F(i, 0), 0), _dataset->V_Final(_handmodel->F(i, 0), 1), _dataset->V_Final(_handmodel->F(i, 0), 2));
				glVertex3f(_dataset->V_Final(_dataset->F(i, 1), 0), _dataset->V_Final(_handmodel->F(i, 1), 1), _dataset->V_Final(_handmodel->F(i, 1), 2));
				glVertex3f(_dataset->V_Final(_dataset->F(i, 2), 0), _dataset->V_Final(_handmodel->F(i, 2), 1), _dataset->V_Final(_handmodel->F(i, 2), 2));
			}
			glEnd();
			glPopMatrix();            //弹出矩阵。
		}
	}
	void draw_skeleton()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		for (int i = 0; i < _handmodel->Joints_num; ++i) {
			//画点开始 
			glColor3f(1.0, 0.0, 0.0);
			glPushMatrix();
			glTranslatef(_handmodel->J_Final(i, 0), _handmodel->J_Final(i, 1), _handmodel->J_Final(i, 2));
			glutSolidSphere(5, 10, 10);
			glPopMatrix();
		}


		for (int i = 0; i < _handmodel->Joints_num; ++i)
		{
			int parent_id = _handmodel->Parent[i];
			if (parent_id == -1) continue;

			glLineWidth(5);
			glColor3f(1.0, 1.0, 0);
			glBegin(GL_LINES);
			glVertex3f(_handmodel->J_Final(i, 0), _handmodel->J_Final(i, 1), _handmodel->J_Final(i, 2));
			glVertex3f(_handmodel->J_Final(parent_id, 0), _handmodel->J_Final(parent_id, 1), _handmodel->J_Final(parent_id, 2));
			glEnd();
		}
	}
	void draw_correspond()
	{
		if (_correspondfind->correspond.size() > 0)
		{
			glDisable(GL_LIGHT0);
			glDisable(GL_LIGHTING);
			glLineWidth(2);
			glColor3f(1.0, 1.0, 1.0);
			glBegin(GL_LINES);
			for (int i = 0; i < _correspondfind->correspond.size(); i++)
			{
				if (_correspondfind->correspond[i].is_match)
				{
					glVertex3d(_correspondfind->correspond[i].pointcloud(0), _correspondfind->correspond[i].pointcloud(1), _correspondfind->correspond[i].pointcloud(2));
					glVertex3d(_correspondfind->correspond[i].correspond(0), _correspondfind->correspond[i].correspond(1), _correspondfind->correspond[i].correspond(2));
				}
			}
			glEnd();
		}
	}
	void draw_Coordinate()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		////x
		//glLineWidth(5);
		//glColor3f(1.0, 0.0, 0.0);
		//glBegin(GL_LINES);
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1), _dataframe->palm_Center(2));
		//glVertex3f(_dataframe->palm_Center(0) + 100, _dataframe->palm_Center(1), _dataframe->palm_Center(2));
		//glEnd();

		////y
		//glLineWidth(5);
		//glColor3f(0.0, 1.0, 0.0);
		//glBegin(GL_LINES);
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1), _dataframe->palm_Center(2));
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1) + 100, _dataframe->palm_Center(2));
		//glEnd();

		////z
		//glLineWidth(5);
		//glColor3f(0.0, 0.0, 1.0);
		//glBegin(GL_LINES);
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1), _dataframe->palm_Center(2));
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1), _dataframe->palm_Center(2) + 100);
		//glEnd();

		//x
		glLineWidth(5);
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
		glVertex3f(90, -50, 450);
		glVertex3f(90 + 100, -50, 450);
		glEnd();

		//y
		glLineWidth(5);
		glColor3f(0.0, 1.0, 0.0);
		glBegin(GL_LINES);
		glVertex3f(90, -50, 450);
		glVertex3f(90, -50 + 100, 450);
		glEnd();

		//z
		glLineWidth(5);
		glColor3f(0.0, 0.0, 1.0);
		glBegin(GL_LINES);
		glVertex3f(90, -50, 450);
		glVertex3f(90, -50, 450 + 100);
		glEnd();
	}
	void draw() {

		//glClearColor(1, 1, 0.8, 1);
		glClearColor(0.5, 0.5, 0.5, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glMatrixMode(GL_MODELVIEW);
		gluPerspective(180, 1.5, -1000, 1000);
		glLoadIdentity();
		/*	control.gx = _dataframe->palm_Center(0);
		control.gy = _dataframe->palm_Center(1);
		control.gz = _dataframe->palm_Center(2);*/

		//这个值是根据palm_Center设置的，因为如果使用palm_Center的话，跳动会变得非常明显
		control.gx = 90;
		control.gy = -50;
		control.gz = 450;

		double r = 300;
		double x = r*cos(control.roty)*sin(control.rotx);
		double y = r*sin(control.roty);
		double z = r*cos(control.roty)*cos(control.rotx);
		//cout<< x <<" "<< y <<" " << z<<endl;
		gluLookAt(x + control.gx, y + control.gy, z + control.gz, control.gx, control.gy, control.gz, 0.0, 1.0, 0.0);//个人理解最开始是看向-z的，之后的角度是在global中心上叠加的，所以要加

		if (show_handmodel) draw_HandModel();
		if (show_Dataset && runtimeType == SHAPE_VERIFY) draw_DatasetModel();
		if (runtimeType == REALTIME) draw_HandPointCloud();
		if (runtimeType == SHAPE_VERIFY) draw_DataSetVisiblePoint();

		draw_HandModel_wireFrame();
		draw_CollisionSphere();
		show_Collision();
		//draw_skeleton();
		draw_Coordinate();
		//draw_correspond();
		glFlush();
		glutSwapBuffers();
	}

	void idle() {

		//_mysensor->concurrent_fetch_streams(*_dataframe);
		//double min;
		//double max;
		//cv::minMaxIdx(_dataframe->original_DepthMap, &min, &max);
		//cv::Mat normalized_depth;
		//_dataframe->original_DepthMap.convertTo(normalized_depth, CV_8UC1, 255.0 / (max - min), 0);  //src.convertTo(dst,type,scale,shift)其中  dst(i) = src(i)* scale + shift;
		//cv::Mat color_map;
		//cv::applyColorMap(normalized_depth, color_map, cv::COLORMAP_COOL);       //灰度图转换成伪彩色，第二个参数是12种伪彩色中的一种
		//cv::flip(color_map, color_map, 0);
		//cv::imshow("depth", color_map);
		//cv::Mat hand;
		//cv::flip(_dataframe->hand_BinaryMap, hand, 0);
		//cv::imshow("Hand", hand);
		//cv::waitKey(20);

		if (runtimeType == REALTIME)
		{
			RealTimeFunc();
		}
		else if (runtimeType == SHAPE_VERIFY)
		{
			ShapeVerfyFunc();
		}
		glutPostRedisplay();
	}


	//GL初始化函数
	void InitializeGlutCallbacks()
	{
		glutKeyboardFunc(keyboardDown);
		glutMouseFunc(mouseClick);
		glutMotionFunc(mouseMotion);
		glutReshapeFunc(reshape);
		glutDisplayFunc(draw);
		glutIdleFunc(idle);
		glutIgnoreKeyRepeat(true); // ignore keys held down
	}
	void initScene(int width, int height) {
		reshape(width, height);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClearDepth(1.0f);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		light();
	}
	void init(int argc, char* argv[]) {
		// 初始化GLUT
		glutInit(&argc, argv);

		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
		glutInitWindowSize(800, 600);
		glutInitWindowPosition(100, 100);
		glutCreateWindow("SHOW RESULT");
		InitializeGlutCallbacks();
		initScene(800, 600);
	}

	void start() {
		// 通知开始GLUT的内部循环
		glutMainLoop();
	}

	void UpdataHandUsingGlove()
	{
		Eigen::VectorXf Rotate_pos = Eigen::VectorXf::Zero(_handmodel->Finger_pose_num + _handmodel->Wrist_pose_num);

		for (int i = 0; i < (_handmodel->Finger_pose_num + _handmodel->Wrist_pose_num); ++i)
		{
			Rotate_pos[i] = GetGloveData[i] * ANGLE_TO_RADIUS;
		}

		Eigen::VectorXf full_pos = Eigen::VectorXf::Zero(_handmodel->Pose_params_num);
		full_pos.tail(_handmodel->Finger_pose_num + _handmodel->Wrist_pose_num) = Rotate_pos;
		full_pos.head(3) = _handmodel->ComputePalmCenterPosition(_correspondfind->my_dataframe->palm_Center);

		_worker->SetGloveParams(full_pos, track);
		/*handmodel->set_Pose_Params(full_pos);
		handmodel->UpdataModel();*/
	}
	void UpdataDataSetUsingGlove()
	{
		Eigen::VectorXf Rotate_pos = Eigen::VectorXf::Zero(_handmodel->Finger_pose_num + _handmodel->Wrist_pose_num);

		for (int i = 0; i < (_handmodel->Finger_pose_num + _handmodel->Wrist_pose_num); ++i)
		{
			Rotate_pos[i] = GetGloveData[i] * ANGLE_TO_RADIUS;
		}

		Eigen::VectorXf full_pos = Eigen::VectorXf::Zero(_handmodel->Pose_params_num);
		full_pos.tail(_handmodel->Finger_pose_num + _handmodel->Wrist_pose_num) = Rotate_pos;
		full_pos.head(3) = Eigen::Vector3f(50, -50, 400);

		_dataset->set_Shape_Params(ShapeVerify_SubIdx);
		_dataset->set_Pose_Params(full_pos);
		_dataset->UpdataModel();

		_worker->SetGloveParams(full_pos, track);

		if (!init_error)
		{
			float total_error = 0;
			for (int i = 0; i < _handmodel->Vertex_num; ++i)
			{
				total_error += (_dataset->V_Final.row(i) - _handmodel->V_Final.row(i)).norm();
			}

			std::cout << "Init error is : " << total_error << std::endl;

			init_error = true;
		}
	}

	void RealTimeFunc()
	{
		if (!pause)
		{
			auto tp_1 = system_clock::now();

			_mysensor->concurrent_fetch_streams(*_correspondfind->my_dataframe);

			auto tp_2 = system_clock::now();

			UpdataHandUsingGlove();
			auto tp_3 = system_clock::now();

			_correspondfind->my_dataframe->handmodel_visibleMap = _handmodel->HandVisible_IndexMap.clone();
			//_correspondfind->Find();
			_correspondfind->Find_2();
			auto tp_4 = system_clock::now();

			if (track)
			{
				for (int i = 0; i < _worker->settings->max_itr; ++i)
				{
					_worker->tracker();
					_correspondfind->my_dataframe->handmodel_visibleMap = _handmodel->HandVisible_IndexMap.clone();
					_correspondfind->Find_2();
				}
			}

			auto tp_5 = system_clock::now();

			{
				double min;
				double max;
				cv::minMaxIdx(_correspondfind->my_dataframe->original_DepthMap, &min, &max);
				cv::Mat normalized_depth;
				_correspondfind->my_dataframe->original_DepthMap.convertTo(normalized_depth, CV_8UC1, 255.0 / (max - min), 0);  //src.convertTo(dst,type,scale,shift)其中  dst(i) = src(i)* scale + shift;
				cv::Mat color_map;
				cv::applyColorMap(normalized_depth, color_map, cv::COLORMAP_COOL);       //灰度图转换成伪彩色，第二个参数是12种伪彩色中的一种
				cv::flip(color_map, color_map, 0);
				cv::imshow("depth", color_map);
				cv::Mat hand;
				cv::flip(_correspondfind->my_dataframe->hand_BinaryMap, hand, 0);
				//cv::imshow("Hand", hand);

				cv::Mat handVisibleMap;
				cv::flip(_handmodel->HandVisible_Map, handVisibleMap, 0);
				cv::Mat HandVisiblePointMap = cv::Mat(handVisibleMap.size(), CV_8UC3, cv::Scalar(255, 255, 255));
				HandVisiblePointMap.setTo(cv::Scalar(0, 0, 0), handVisibleMap > 0);

				cv::Mat handColored = cv::Mat(hand.size(), CV_8UC3, cv::Scalar(255, 255, 255));
				handColored.setTo(cv::Scalar(0, 0, 255), hand > 0);

				cv::Mat MixShow;
				cv::addWeighted(handColored, 0.5, HandVisiblePointMap, 0.5, 0, MixShow);
				cv::imshow("mixShow", MixShow);
			}


			milliseconds dur_1 = duration_cast<milliseconds>(tp_2 - tp_1);  //获取数据时间
			milliseconds dur_2 = duration_cast<milliseconds>(tp_3 - tp_2);  //更新手模时间
			milliseconds dur_3 = duration_cast<milliseconds>(tp_4 - tp_3);   //寻找最近点时间
			milliseconds dur_4 = duration_cast<milliseconds>(tp_5 - tp_4);    //迭代时间

																			  /*cout << "获取数据时间 ： " << dur_1.count() << " ms\n";
																			  cout << "更新手模时间 ： " << dur_2.count() << " ms\n";
																			  cout << "寻找最近点时间 ： " << dur_3.count() << " ms\n";
																			  cout << "迭代时间  : " << dur_4.count() << " ms \n";
																			  cout << " 最终误差为： " << _worker->total_error << endl << endl;*/
		}
	}

	void ShapeVerfyFunc()
	{
		if (!pause)
		{
			UpdataDataSetUsingGlove();

			//UpdataHandUsingGlove();

			_dataset->FetchDataFrame(*_correspondfind->my_dataframe);
			_correspondfind->Find_2();

			if (track)
			{
				for (int i = 0; i < _worker->settings->max_itr; ++i)
				{
					_worker->tracker();
					_correspondfind->Find_2();
				}

				float total_error = 0;
				for (int i = 0; i < _handmodel->Vertex_num; ++i)
				{
					total_error += (_dataset->V_Final.row(i) - _handmodel->V_Final.row(i)).norm();
				}

				std::cout << "error is : " << total_error << std::endl;
			}

			{
				cv::Mat hand;
				cv::flip(_correspondfind->my_dataframe->hand_BinaryMap, hand, 0);
				//cv::imshow("Hand", hand);

				cv::Mat handVisibleMap;
				cv::flip(_handmodel->HandVisible_Map, handVisibleMap, 0);
				cv::Mat HandVisiblePointMap = cv::Mat(handVisibleMap.size(), CV_8UC3, cv::Scalar(255, 255, 255));
				HandVisiblePointMap.setTo(cv::Scalar(0, 0, 0), handVisibleMap > 0);

				cv::Mat handColored = cv::Mat(hand.size(), CV_8UC3, cv::Scalar(255, 255, 255));
				handColored.setTo(cv::Scalar(0, 0, 255), hand > 0);

				cv::Mat MixShow;
				cv::addWeighted(handColored, 0.5, HandVisiblePointMap, 0.5, 0, MixShow);
				cv::imshow("mixShow", MixShow);
			}
		}
	}
}