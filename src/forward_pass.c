// forward_pass.c
/*This file contains functions for forward propagation*/

#include "defs.h"

static float weighted_sum(Layer *prev_layer, int neuron_idx) {
    float z = 0.0f;

    for (unsigned i = 0; i < prev_layer->num_neurons; ++i) {
        printf("z%d\n", i);
        z += prev_layer->neurons[i].value * prev_layer->neurons[i].weights[neuron_idx];
    }

    return z;
}


void forward_propagation(Network *network) {
    for (unsigned layer_idx = 1; layer_idx < network->num_layers; ++layer_idx) {
        printf("a%d\n", layer_idx);
        Layer *current_layer = &network->layers[layer_idx];
        Layer *prev_layer = &network->layers[layer_idx - 1];

        for (unsigned neuron_idx = 0; neuron_idx < current_layer->num_neurons; ++neuron_idx) {
            printf("aa%d.%d\n", layer_idx, neuron_idx);
            float z = weighted_sum(prev_layer, neuron_idx);
            z += current_layer->neurons[neuron_idx].bias;
            current_layer->neurons[neuron_idx].value = activation_function(current_layer->activation, z);
        }
    }
}

