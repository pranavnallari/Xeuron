// init.c
/*This file initializes the neural network based on the given training set*/

#include "defs.h"

static float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

static float xavier_initialization(unsigned num_in, unsigned num_out) {
    float stddev = sqrtf(2.0f / (num_in + num_out));
    return rand_float() * stddev;
}

static float he_initialization(unsigned num_in) {
    float stddev = sqrtf(2.0f / num_in);
    return rand_float() * stddev;
}

void one_hot_encode(unsigned char label, float actual_values[10]) {
    for (int i = 0; i < 10; ++i) {
        actual_values[i] = 0.0f;
    }
    actual_values[(int)label] = 1.0f;
}

void initialize_network(Network *network) {
    srand(time(NULL));
    
    for (unsigned layer_idx = 1; layer_idx < network->num_layers; ++layer_idx) {
        Layer *current_layer = &network->layers[layer_idx];
        Layer *prev_layer = &network->layers[layer_idx - 1];

        for (unsigned neuron_idx = 0; neuron_idx < current_layer->num_neurons; ++neuron_idx) {
            Neuron *neuron = &current_layer->neurons[neuron_idx];
            neuron->weights = malloc(prev_layer->num_neurons * sizeof(float));

            if (current_layer->activation == RELU) {
                for (unsigned i = 0; i < prev_layer->num_neurons; ++i) {
                    neuron->weights[i] = he_initialization(prev_layer->num_neurons);
                }
            } else {
                for (unsigned i = 0; i < prev_layer->num_neurons; ++i) {
                    neuron->weights[i] = xavier_initialization(prev_layer->num_neurons, current_layer->num_neurons);
                }
            }
            neuron->bias = 0.0001f;
        }
    }
}