#include "Neuron.h"

#include <vector>


Neuron::Neuron(int nextLayerSize) {
	for (int i = 0; i < nextLayerSize; i++) {
		weights.push_back( ((rand() / double(RAND_MAX))  ));
	}
}


double Neuron::get_weight(int num) { 
	return weights.at(num); 
}

void Neuron::set_weight(int index, double num)
{
	weights.at(index) = num;
}

void Neuron::change_weight(int num, double change) { 
	weights.at(num) += change; 
}

double Neuron::get_grad() { return grad; }

void Neuron::set_grad(double val) { 
	grad = val; 
}

void Neuron::setActivationFunction(char func) { actFunc = func; }

void Neuron::activate(double &v) {
	if (actFunc == 1) { linearActivation(v); }
	else if (actFunc == 2) { reluActivation(v); }
	else if (actFunc == 3) { sigmoidActivation(v); }
	else if (actFunc == 4) { tanhActivation(v); }
}

void Neuron::linearActivation(double &v) { val = v; }

void Neuron::reluActivation(double &v) { 
	if (v > 0) { val = v; } else { val = 0; }; 
}

void Neuron::sigmoidActivation(double &v) { 
	val = 1 / (1 + exp(-regParam * v)); 
}

void Neuron::tanhActivation(double &v) {
	val = tanh(v);
}