// forward_pass.c
/*This file contains functions for forward propagation*/

#include "defs.h"

static float weighted_sum(Layer *layer, int neuron_idx) {
    float z = 0.0f;

    for (int i = 0; i < layer->num_neurons; ++i) {
        z += layer->neurons[i].value * layer->neurons[neuron_idx].weight;
    }

    return z + layer->bias;
}

void forward_propagation(Network *network) {
    for (int layer_idx = 1; layer_idx < network->num_layers; ++layer_idx) {
        Layer *current_layer = &network->layers[layer_idx];
        Layer *prev_layer = &network->layers[layer_idx - 1];

        for (int neuron_idx = 0; neuron_idx < current_layer->num_neurons; ++neuron_idx) {
            float z = weighted_sum(prev_layer, neuron_idx);
            current_layer->neurons[neuron_idx].value = activation_function(current_layer->activation, z);
        }
    }
}


