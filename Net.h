#pragma once

#include "Neuron.h"

#include <vector>

using namespace std;

class Net {

public:

	void build_net(vector<int> &topology);

	void train();
	void test();

	const double lRate = 0.8;
	const double regParam = 0.2;
	const double epochs = 150;
	int epochNum = 0;

	void net_train_cycle();
	void net_test_cycle();
	vector<double> net_run_cycle(double x1, double x2);

	void load_data();

	void setInputPath(string path) { inputPath = path; }
	void setOutputPath(string path) { outputPath = path; }
	void clearData();
	void assignInputVals(int index);
	void assignOutputVals(int index) { this->expectedOut = { output1[index], output2[index] }; }
	vector<double> getInputVals(int index) { return { input1[index], input2[index] }; }
	vector<double> getOutputVals(int index) { return { output1[index], output2[index] }; }
	int getDataSize() { return input1.size(); }
	void feedForward();
	void errorCheck();

	void setActivationFunction(char func) { actFunc = func; };
	void outputGrads();
	void hiddenGrads();
	void outputGradsLinear();
	void hiddenGradsLinear();
	void outputGradsRelu();
	void hiddenGradsRelu();
	void outputGradsSigmoid();
	void hiddenGradsSigmoid();
	void outputGradsTanh();
	void hiddenGradsTanh();

	void updateweights();
	double calcRMSE();

	void set_best_weights();

	void saveWeights();
	vector<vector<double>> bestInputWeights;
	vector<vector<double>> bestHiddenWeights;

	void printErrors();
	void printNet();

private:

	string inputPath;
	string outputPath;

	vector<double> trainInput1;
	vector<double> trainInput2;
	vector<double> testInput1;
	vector<double> testInput2;

	vector<double> trainOutput1;
	vector<double> trainOutput2;
	vector<double> testOutput1;
	vector<double> testOutput2;

	vector<double> input1;
	vector<double> input2;
	vector<double> output1;
	vector<double> output2;

	int actFunc;

	vector<double> e1;
	vector<double> e2;
	double bestRMSE = 1;

	vector<Neuron> inputLayer;
	vector<Neuron> hiddenLayer;
	vector<Neuron> outputLayer;

	vector<double> expectedOut;

};




