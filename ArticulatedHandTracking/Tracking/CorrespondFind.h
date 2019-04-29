#pragma once
#include"Types.h"
#include"Camera.h"
#include"DataFrame.h"
#include"HandModel.h"
#include<thread>

class CorrespondFind
{
public:
	std::vector<DataAndCorrespond> correspond;
	int num_matched_correspond;

	DataFrame* my_dataframe;
	HandModel *handmodel;

	CorrespondFind(Camera* _camera, HandModel* _handmodel, DataFrame* _dataframe) :camera(_camera), handmodel(_handmodel), my_dataframe(_dataframe)
	{};
	~CorrespondFind() {};

	void Find();
	void FindCore(int start, int end);

	void Find_2();
private:
	Camera* camera;
};