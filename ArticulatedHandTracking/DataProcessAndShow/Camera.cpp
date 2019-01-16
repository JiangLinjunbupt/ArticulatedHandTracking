#include "Camera.h"


Camera::Camera(CAMERAMODE mode,int fps) : _mode(mode){

	switch (_mode)
	{
		//�������в�������ʹ��MATLAB�Դ�������궨������в���
	case RealsenseSR300:
		_width = 320;
		_height = 240;
		_zFar = 500;
		_zNear = 100;
		_CameraCenterX = 160.0f;    //ͨ��������� _width/2
		_CameraCenterY = 120.0f;    //ͨ��������� _height/2
		_focal_length_x = 224.502f;
		_focal_length_y = 230.494f;
		_fps = fps;
		break;
	case KinectV2:
		_width = 256;
		_height = 212;
		_zFar = 3000;
		_zNear = 500;
		_CameraCenterX = 132.0f;
		_CameraCenterY = 108.5f;
		_focal_length_x = 190.5f;
		_focal_length_y = 191.0f;
		_fps = 30;
		//_width = 512;
		//_height = 424;
		//_zFar = 3000;
		//_zNear = 500;
		//_CameraCenterX = 264.0945f;
		//_CameraCenterY = 217.1487f;
		//_focal_length_x = 381.8452f;
		//_focal_length_y = 382.1713f;
		//_fps = 30;
		break;
	default:
		printf("!!!FATAL: Invalid Camera Mode !!!");
		exit(-1);
	}

	///--- Assemble projection matrix
	auto kinectproj = [=]() {
		Matrix33f cam_matrix = Matrix33f::Zero();
		cam_matrix(0, 0) = _focal_length_x; /// FocalLength X
		cam_matrix(1, 1) = _focal_length_y; /// FocalLength Y
		cam_matrix(0, 2) = _CameraCenterX;      /// CameraCenter X
		cam_matrix(1, 2) = _CameraCenterY;     /// CameraCenter Y
		cam_matrix(2, 2) = 1.0;
		return cam_matrix;
	};
	proj = kinectproj();
	iproj = proj.inverse();
}


//���������(height() - j - 1)�����ǿ��ǵ����¹�ϵ��
//camera����ϵ
//              |y
//              |
//              |________x              ��Kinect������ͷ��������ϵһ���������˳��ӽǣ�
//             /                       ����Ҫע����㣬Kinec���������ͼ�ǣ�   O----------x
//          z /                                                                |                  |y
//                                                                             |y                 |
//                                      ��ˣ�Ҫ�����ͼ����cv::filp(img,img,0)��x�ᷭת���õ�    O-------x����������֮���ٽ���ת��������ϵ
Eigen::Vector3f Camera::depth_to_world(float i, float j, float depth) {
	Eigen::Vector3f wrld = iproj * Eigen::Vector3f(i*depth, (height() - j - 1)*depth, depth);
	return wrld;
}


//����Ƿ�ͶӰ����������
Eigen::Vector3f Camera::unproject(int i, int j, float depth) {
	return pixel_to_image_plane(i, j) * depth;
}
Eigen::Vector3f Camera::pixel_to_image_plane(int i, int j) {
	float x = (i - proj(0, 2)) / proj(0, 0);
	float y = (j - proj(1, 2)) / proj(1, 1);
	return Eigen::Vector3f(x, y, 1);
}

//�����ͶӰ����������
Eigen::Vector2f Camera::world_to_image(const Eigen::Vector3f& wrld) {
	float x = wrld[0] / wrld[2];
	float y = wrld[1] / wrld[2];
	x = x*proj(0, 0) + proj(0, 2);
	y = y*proj(1, 1) + proj(1, 2);
	return Eigen::Vector2f(x, y);
}


Camera::Matrix44f Camera::view_projection_matrix() {
	///--- Intrinsics matrix
	Matrix33f& K = proj;
	int w = this->width();
	int h = this->height();

	Matrix44f mat = Matrix44f::Identity();
	mat(0, 0) = 2.0 / (float)w*K(0, 0); // use camera instrinsics and convert to GL [0,h] => [-1,1]
	mat(0, 2) = (2.0 / (float)w*(K(0, 2) + 0.5)) - 1.0; // 0.5 offset as GL pixel middle point is at 0.5,0.5
														// Y
	mat(1, 1) = 2.0 / (float)h*K(1, 1); // use camera instrinsics and convert to GL [0,h] => [-1,1]
	mat(1, 2) = (2.0 / (float)h*(K(1, 2) + 0.5)) - 1.0;
	// Z
	mat(2, 2) = (_zFar + _zNear) / (_zFar - _zNear);
	mat(2, 3) = -2.0*_zFar*_zNear / (_zFar - _zNear);
	// W
	mat(3, 2) = 1; // not as in GL where it would be -1
	mat(3, 3) = 0;

	return mat;
}

Matrix_2x3 Camera::projection_jacobian(const Eigen::Vector3f &p) {
	Matrix_2x3 M;
	M << _focal_length_x / p.z(), 0, -p.x() * _focal_length_x / (p.z()*p.z()),
		0, _focal_length_y / p.z(), -p.y() * _focal_length_y / (p.z()*p.z());
	return M;
}

