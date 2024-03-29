// NN.cpp : Defines the entry point for the console application.
//
#include "Net.h"

//#include "Aria.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>

using namespace std;


void train(Net &net, vector<int> &topology)
{
	srand(time(NULL));
	int input;
	cout << "Choose activation function 1 - Linear, 2 - Relu, 3 - Sigmoid, 4 - tanh:\n";
	cin >> input;

	net.setActivationFunction(input);
	net.build_net(topology);
	
	for (int i = 0; i < net.epochs; i++) {
		net.epochNum++;
		cout << "Epoch:   " << net.epochNum << "\n";
		net.train();
		net.clearData();
		net.test();
		net.clearData();
	}



	cin >> input;
}

void run(int argc, char **argv, Net &net, vector<int> &topology)
{
	/*
	Aria::init();
	ArRobot robot;
	ArPose pose;

	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();

	ArRobotConnector robotConnector(&argParser, &robot);

	if (robotConnector.connectRobot())
		cout << "Robot Connected!" << endl;

	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();

	ArSensorReading *sonarSensor[8];

	double sonarRange[8];
	*/
	net.set_best_weights();

	vector<double> outputs;

	while (true) {
		/*
		for (int i = 0; i < 8; i++)
		{
			sonarSensor[i] = robot.getSonarReading(i);
			sonarRange[i] = sonarSensor[i]->getRange();
		}
		cout << "Sonar 0: " << sonarRange[0] << "  Sonar 1: " << sonarRange[1] << "\n";
		*/
		double input1;
		cin >> input1;
		double input2;
		cin >> input2;

		outputs = net.net_run_cycle(input1 / 5000, input2 / 5000);

		cout << "Outputs 0: " << outputs.at(0) * 215 + 85 << "  Outputs 1: " << outputs.at(1) * 215 + 85 << "\n";


	}

	
}

int main(int argc, char **argv)
{

	Net net;
	net.load_data();

	vector<int> topology = { 2, 8, 2 };

	while(true) {
		int input;
		cout << "Type:\n1) For Training\n2) To Run The Most Successful Network\n";
		cin >> input;
		if ( input == 1)
		{
			train(net, topology);
		}else if (input == 2)
		{
			run(argc, argv, net, topology);
		}
	}
}


/* _____TO DO_____
	1.	LINEAR ACTIVATION FUNCTION
	2.	MOVE CLASSES INTO HEADER FILES
*/