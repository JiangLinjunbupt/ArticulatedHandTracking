//��Ϊʹ���˹����ڴ棬��ʹ�ù���ԱȨ�޿���visual studio��Ȼ���ٴ򿪹���
//�������ȱ��fertilized.dll����ǰ��https://github.com/edoRemelli/hand-seg-rdf  ���б������Ĺ��̵õ�fertilized.dll
#include"Opengl_display.h"
#include <tchar.h>


using namespace std;
using namespace chrono;
//�����ڴ����ض���
HANDLE hMapFile;
LPCTSTR pBuf;
#define BUF_SIZE 1024
TCHAR szName[] = TEXT("Global\\MyFileMappingObject");    //ָ��ͬһ�鹲���ڴ������
float *GetSharedMemeryPtr;

void setSharedMemery();

int main(int argc, char** argv)
{
	setSharedMemery();
	std::cout << "Loadding The RDF...\n";

	int Height = 480;
	int Width = 640;
	string DatasetPath = "";
	Camera* camera = new Camera(REALTIME);
	switch (camera->_type)
	{
	case Dataset_MSRA_14:
		DatasetPath = "F:\\���ݼ�\\cvpr14_MSRAHandTrackingDB\\cvpr14_MSRAHandTrackingDB\\Subject1\\";
		break;
	case Dataset_MSRA_15:
		DatasetPath = "F:\\���ݼ�\\cvpr15_MSRAHandGestureDB\\cvpr15_MSRAHandGestureDB\\P0\\1\\";
		break;
	case Handy_teaser:
		DatasetPath = "F:\\���ݼ�\\teaser\\teaser\\";
		break; 
	default:
		break;
	}

	RealSenseSensor* sensor = new RealSenseSensor(camera);
	if(camera->_type == REALTIME || camera->_type == SHAPE_VERIFY) 	sensor->start();

	DataFrame* dataframe = new DataFrame();
	dataframe->Init(Width, Height);

	HandModel* handmodel = new HandModel(camera);
	DataSet* dataset = new DataSet(camera);

	CorrespondFind* correspondfind = new CorrespondFind(camera, handmodel, dataframe);

	Worker* worker = new Worker(handmodel, correspondfind, camera);
	Evaluations* evalutions = new Evaluations(camera, handmodel);

	DS::GetGloveData = GetSharedMemeryPtr;
	DS::_camera = camera;
	DS::_mysensor = sensor;
	DS::_correspondfind = correspondfind;
	DS::_handmodel = handmodel;
	DS::_worker = worker;
	DS::_dataset = dataset;
	DS::_evalutions = evalutions;
	DS::runtimeType = camera->_type;
	DS::datasetPath = DatasetPath;


	DS::init(argc, argv);
	DS::start();

	return 0;
}


void setSharedMemery()
{
#pragma region SharedMemery
	hMapFile = CreateFileMapping(
		INVALID_HANDLE_VALUE,    // use paging file
		NULL,                    // default security
		PAGE_READWRITE,          // read/write access
		0,                       // maximum object size (high-order DWORD)
		BUF_SIZE,                // maximum object size (low-order DWORD)
		szName);                 // name of mapping object

	if (hMapFile == NULL)
	{
		_tprintf(TEXT("Could not create file mapping object (%d).\n"),
			GetLastError());
		exit(0);
	}
	pBuf = (LPTSTR)MapViewOfFile(hMapFile,   // handle to map object
		FILE_MAP_ALL_ACCESS, // read/write permission
		0,
		0,
		BUF_SIZE);

	if (pBuf == NULL)
	{
		_tprintf(TEXT("Could not map view of file (%d).\n"),
			GetLastError());

		CloseHandle(hMapFile);
		exit(0);
	}

	GetSharedMemeryPtr = (float*)pBuf;
#pragma endregion SharedMemery
}