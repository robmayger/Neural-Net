#include "Net.h"
#include "Neuron.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

void Net::load_data() {
	ifstream inputs;
	inputs.open("C:/Users/Rob/Documents/Essex/NNaDL/NN/NN/In.csv");
	if (!inputs.is_open())
	{
		cout << "***Error!***\n" << "Could Not Open Inputs File!\n";
	}
	ifstream outputs;
	outputs.open("C:/Users/Rob/Documents/Essex/NNaDL/NN/NN/Out.csv");
	if (!outputs.is_open())
	{
		cout << "***Error!***\n" << "Could Not Open Outputs File!\n";
	}
	string line;
	string delimiter = ",";
	while (getline(inputs, line))
	{
		trainInput1.push_back(stod(line.substr(0, line.find(delimiter))) / 5000);
		trainInput2.push_back(stod(line.substr(line.find(delimiter) + 1, line.size())) / 5000);
	}
	while (getline(outputs, line))
	{
		trainOutput1.push_back((stod(line.substr(0, line.find(delimiter))) - 85) / 215);
		trainOutput2.push_back((stod(line.substr(line.find(delimiter) + 1, line.size())) - 85) / 215);
	}
	ifstream testInputs;
	testInputs.open("C:/Users/Rob/Documents/Essex/NNaDL/NN/NN/TestIn.csv");
	if (!testInputs.is_open())
	{
		cout << "***Error!***\n" << "Could Not Open Inputs File!\n";
	}
	ifstream testOutputs;
	testOutputs.open("C:/Users/Rob/Documents/Essex/NNaDL/NN/NN/TestOut.csv");
	if (!testOutputs.is_open())
	{
		cout << "***Error!***\n" << "Could Not Open Outputs File!\n";
	}
	while (getline(testInputs, line))
	{
		testInput1.push_back(stod(line.substr(0, line.find(delimiter))) / 5000);
		testInput2.push_back(stod(line.substr(line.find(delimiter) + 1, line.size())) / 5000);
	}
	while (getline(testOutputs, line))
	{
		testOutput1.push_back((stod(line.substr(0, line.find(delimiter))) -85 ) / 215);
		testOutput2.push_back((stod(line.substr(line.find(delimiter) + 1, line.size())) - 85 ) / 215);
	}
}

void Net::build_net(vector<int> &topology) {
	for (int i = 0; i < topology.at(0) + 1; i++) {
		inputLayer.push_back(Neuron(topology.at(1)));
		inputLayer.back().setActivationFunction(actFunc);
	}
	for (int i = 0; i < topology.at(1) + 1; i++) {
		hiddenLayer.push_back(Neuron(topology.at(2)));
		hiddenLayer.back().setActivationFunction(actFunc);
	}
	for (int i = 0; i < topology.at(2); i++) {
		outputLayer.push_back(Neuron(0));
		outputLayer.back().setActivationFunction(actFunc);
	}

	inputLayer.at(0).val = 1;
	hiddenLayer.at(0).val = 1;

	printNet();
	int in;
	cin >> in;
}

void Net::assignInputVals(int index) {
	vector<double> inputVals = getInputVals(index);
	for (int i = 1; i < inputLayer.size(); i++) {
		inputLayer.at(i).val = inputVals.at(i - 1);
	}
}

void Net::feedForward() {
	int num = epochNum;
	double hi;
	for (int i = 1; i < hiddenLayer.size(); i++) {
		hi = 0;
		for (int j = 0; j < inputLayer.size(); j++) {
			hi += inputLayer.at(j).val * inputLayer.at(j).get_weight(i - 1);
		}
		hiddenLayer.at(i).activate(hi);
	}
	double yk;
	for (int k = 0; k < outputLayer.size(); k++) {
		yk = 0;
		for (int i = 0; i < hiddenLayer.size(); i++) {
			yk += hiddenLayer.at(i).val * hiddenLayer.at(i).get_weight(k);
		}
		outputLayer.at(k).activate(yk);
	}
}

void Net::errorCheck() {
	e1.push_back(expectedOut.at(0) - outputLayer.at(0).val);
	e2.push_back(expectedOut.at(1) - outputLayer.at(1).val);

	//cout << "============ERROR VALUES============\n";
	//cout << "Error 1: " << e1.back() << "      Error 2: " << e2.back() << "\n" << "\n";
}

void Net::outputGrads() {
	if (actFunc == 1) { outputGradsLinear(); }
	else if (actFunc == 2) { outputGradsRelu(); }
	else if (actFunc == 3) { outputGradsSigmoid(); }
	else if (actFunc == 4) { outputGradsTanh(); }
}

void Net::hiddenGrads() {
	if (actFunc == 1) { hiddenGradsLinear(); }
	else if (actFunc == 2) { hiddenGradsRelu(); }
	else if (actFunc == 3) { hiddenGradsSigmoid(); }
	else if (actFunc == 4) { hiddenGradsTanh(); }
}

void Net::outputGradsLinear() {
	outputLayer.at(0).set_grad(e1.back());
	outputLayer.at(1).set_grad(e2.back());
}

void Net::hiddenGradsLinear() {
	double sum;

	for (int i = 0; i < hiddenLayer.size(); i++) {
		sum = 0;
		for (int k = 0; k < outputLayer.size(); k++) {
			sum += outputLayer.at(k).get_grad() * hiddenLayer.at(i).get_weight(k);
		}
		hiddenLayer.at(i).set_grad(sum);
	}
}

void Net::outputGradsRelu() {

	if (outputLayer.at(0).val > 0) {
		outputLayer.at(0).set_grad(e1.back());
	} else {
		outputLayer.at(0).set_grad(0);
	}

	if (outputLayer.at(1).val > 0) {
		outputLayer.at(1).set_grad(e2.back());
	} else {
		outputLayer.at(1).set_grad(0);
	}
}

void Net::hiddenGradsRelu() {
	
	double sum;

	for (int i = 0; i < hiddenLayer.size(); i++) {
		sum = 0;
		if (hiddenLayer.at(i).val > 0) {
			for (int k = 0; k < outputLayer.size(); k++) {
				sum += outputLayer.at(k).get_grad() * hiddenLayer.at(i).get_weight(k);
			}
			hiddenLayer.at(i).set_grad(hiddenLayer.at(i).val * sum);
		} else {
			hiddenLayer.at(i).set_grad(0);
		}
	}
}

void Net::outputGradsSigmoid() {
	outputLayer.at(0).set_grad(regParam * outputLayer.at(0).val * (1 - outputLayer.at(0).val) * e1.back());
	outputLayer.at(1).set_grad(regParam * outputLayer.at(1).val * (1 - outputLayer.at(1).val) * e2.back());
}

void Net::hiddenGradsSigmoid() {
	double sum;

	for (int i = 0; i < hiddenLayer.size(); i++) {
		sum = 0;
		for (int k = 0; k < outputLayer.size(); k++) {
			sum += outputLayer.at(k).get_grad() * hiddenLayer.at(i).get_weight(k);
		}
		hiddenLayer.at(i).set_grad(regParam * hiddenLayer.at(i).val * (1 - hiddenLayer.at(i).val) * sum);
	}
}

void Net::outputGradsTanh() {
	outputLayer.at(0).set_grad((1 - pow(outputLayer.at(0).val, 2)) * e1.back());
	outputLayer.at(1).set_grad((1 - pow(outputLayer.at(1).val, 2)) * e2.back());
}

void Net::hiddenGradsTanh() {
	double sum;

	for (int i = 0; i < hiddenLayer.size(); i++) {
		sum = 0;
		for (int k = 0; k < outputLayer.size(); k++) {
			sum += outputLayer.at(k).get_grad() * hiddenLayer.at(i).get_weight(k);
		}
		hiddenLayer.at(i).set_grad((1 - pow(hiddenLayer.at(i).val, 2)) * sum);
	}
}

void Net::updateweights() {
	double change;
	for (int i = 0; i < hiddenLayer.size(); i++) {
		for (int k = 0; k < outputLayer.size(); k++) {
			change = lRate * outputLayer.at(k).get_grad() * hiddenLayer.at(i).val;
			hiddenLayer.at(i).change_weight(k, change);
		}
	}
	for (int j = 0; j < inputLayer.size(); j++) {
		for (int i = 0; i < hiddenLayer.size() - 1; i++) {
			change = lRate * hiddenLayer.at(i).get_grad() * inputLayer.at(j).val;
			inputLayer.at(j).change_weight(i, change);
		}
	}
}

void Net::printNet() {
	cout << "============CURRENT STATE============\n";
	for (int j = 0; j < inputLayer.size(); j++) {
		cout << "Neuron " << j << ":   Value:" << inputLayer.at(j).val
			<< "   Gradient:" << inputLayer.at(j).get_grad() << "\n";
		cout << "Weights:\n";
		for (int i = 0; i < hiddenLayer.size() - 1; i++) {
			cout << i << ": " << inputLayer.at(j).get_weight(i) << "\n";
		}
		cout << "\n";
	}
	for (int i = 0; i < hiddenLayer.size(); i++) {
		cout << "Neuron " << i + inputLayer.size() << ":   Value:"
			<< hiddenLayer.at(i).val << "   Gradient:"
			<< hiddenLayer.at(i).get_grad() << "\n";
		cout << "Weights:\n";
		for (int k = 0; k < outputLayer.size(); k++) {
			cout << k << ": " << hiddenLayer.at(i).get_weight(k) << "\n";
		}
		cout << "\n";
	}
	for (int k = 0; k < outputLayer.size(); k++) {
		cout << "Neuron " << k + inputLayer.size() + hiddenLayer.size()
			<< ":   Value:" << outputLayer.at(k).val << "   Gradient:"
			<< outputLayer.at(k).get_grad() << "\n";
		cout << "\n";
	}
}

void Net::printErrors() {

	for (int i = 0; i < e1.size(); i++) {
		cout << "Error1: " << e1.at(i) << "    Error2: " << e2.at(i) << "\n";
	}

}

double Net::calcRMSE() {
	double sumE1 = 0;
	double sumE2 = 0;
	double rmsE;
	for (int i = 0; i < e1.size(); i++) {
		sumE1 += pow(e1.at(i), 2);
		sumE2 += pow(e2.at(i), 2);
	}
	rmsE = (sqrt(sumE1 / e1.size()) + sqrt(sumE2 / e1.size())) / 2;

	return rmsE;
}

void Net::clearData() {
	//printNet();
	input1.clear();
	input2.clear();
	output1.clear();
	output2.clear();
	e1.clear();
	e2.clear();
}

void Net::net_train_cycle()
{
	//printNet();

	feedForward();

	//printNet();

	errorCheck();

	outputGrads();
	hiddenGrads();

	//printNet();

	updateweights();

	//printNet();
}

void Net::train() {

	input1 = trainInput1;
	input2 = trainInput2;
	output1 = trainOutput1;
	output2 = trainOutput2;

	for (int i = 0; i < getDataSize(); i++) {

		assignInputVals(i);

		assignOutputVals(i);

		net_train_cycle();
	}

	//printErrors();
	cout << "Train RMSE: " << calcRMSE() << "\n";
}

void Net::net_test_cycle()
{
	//printNet();

	feedForward();

	//printNet();

	errorCheck();
}

void Net::test() {

	input1 = testInput1;
	input2 = testInput2;
	output1 = testOutput1;
	output2 = testOutput2;

	for (int i = 0; i < getDataSize(); i++) {

		assignInputVals(i);

		assignOutputVals(i);

		net_test_cycle();
	}
	double rmse = calcRMSE();
	if (rmse < bestRMSE) 
	{
		bestRMSE = rmse;
		saveWeights();
	}
	cout << "Test RMSE: " << rmse << "\n";

}

vector <double> Net::net_run_cycle(double x1, double x2)
{
	inputLayer.at(1).val = x1;
	inputLayer.at(2).val = x2;

	feedForward();

	return{ outputLayer.at(0).val, outputLayer.at(1).val };
}

void Net::saveWeights()
{	
	bestInputWeights.clear();
	bestHiddenWeights.clear();
	vector<double> temp;
	for (int i = 0; i < inputLayer.size(); i++)
	{
		for (int j = 0; j < hiddenLayer.size() - 1; j++)
		{
			temp.push_back(inputLayer.at(i).get_weight(j));
		}
		bestInputWeights.push_back(temp);
		temp.clear();
	}
	for (int j = 0; j < hiddenLayer.size(); j++)
	{
		for (int k = 0; k < outputLayer.size(); k++)
		{
			temp.push_back(hiddenLayer.at(j).get_weight(k));
		}
		bestHiddenWeights.push_back(temp);
		temp.clear();
	}
}

void Net::set_best_weights()
{
	for (int i = 0; i < inputLayer.size(); i++)
	{
		for (int j = 0; j < hiddenLayer.size() - 1; j++)
		{
			inputLayer.at(i).set_weight(j, bestInputWeights.at(i).at(j));
		}
	}
	for (int j = 0; j < hiddenLayer.size(); j++)
	{
		for (int k = 0; k < outputLayer.size(); k++)
		{
			hiddenLayer.at(j).set_weight(k, bestHiddenWeights.at(j).at(k));
		}
	}
}


