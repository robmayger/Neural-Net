#pragma once

#include <vector>

using namespace std;

class Neuron {

public:

	Neuron(int nextLayerSize);

	double get_weight(int num);
	void set_weight(int index, double num);
	void change_weight(int num, double change);
	double get_grad();
	void set_grad(double val);

	void setActivationFunction(char func);
	void activate(double &v);
	void linearActivation(double &v);
	void reluActivation(double &v);
	void sigmoidActivation(double &v);
	void tanhActivation(double &v);

	double val;

private:

	const double regParam = 0.5;
	int actFunc;
	double grad;
	vector<double> weights;

};


