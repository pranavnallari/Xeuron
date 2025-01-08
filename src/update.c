// update.c
/* This file contains the code for updating the parameters after calculating the gradients using stochastic gradient descent (SGD) */


#include "defs.h"

void update_parameters_sgd(Network *network, float learning_rate) {
    for (unsigned layer_idx = 1; layer_idx < network->num_layers; ++layer_idx) {
        Layer *layer = &network->layers[layer_idx];
        Layer *prev_layer = &network->layers[layer_idx - 1];

        for (unsigned i = 0; i < layer->num_neurons; ++i) {
            Neuron *neuron = &layer->neurons[i];
            for (unsigned j = 0; j < prev_layer->num_neurons; ++j) {
                neuron->weights[j] -= learning_rate * prev_layer->neurons[j].value * neuron->gradient;
            }
            neuron->bias -= learning_rate * neuron->gradient;
        }
    }
}