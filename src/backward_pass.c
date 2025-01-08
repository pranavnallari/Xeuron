// backward_pass.c
/* This file contains the code for calculating the gradient of the loss function wrt network's parameters (weights and biases)*/

#include "defs.h"


static void gradient_output_layer_mse(Layer *output_layer, float actual_values[]) {
    for (unsigned i = 0; i < output_layer->num_neurons; ++i) {
        float pred = output_layer->neurons[i].value;
        float actual = actual_values[i];
        output_layer->neurons[i].gradient = 2.0f * (pred - actual) / (output_layer->num_neurons);
    }
}
static void gradient_output_layer_bce(Layer *output_layer, float actual_values[]) {
    const float epsilon = FLT_MIN;      // avoid log(0)
    for (unsigned i = 0; i < output_layer->num_neurons; ++i) {
        float pred = fmaxf(fminf(output_layer->neurons[i].value, 1.0f - epsilon), epsilon);
        float actual = actual_values[i];
        output_layer->neurons[i].gradient = -(actual / pred) + ((1.0f - actual) / (1.0f - pred));
    }
}
static void gradient_output_layer_cce(Layer *output_layer, float actual_values[]) {
    for (unsigned i = 0; i < output_layer->num_neurons; ++i) {
        float pred = output_layer->neurons[i].value;
        float actual = actual_values[i];
        output_layer->neurons[i].gradient = pred - actual;
    }
}

void calculate_output_layer_gradient(Layer *output_layer, float actual_values[], LossFunction type) {
    switch(type) {
        case BCE: gradient_output_layer_bce(output_layer, actual_values);
        break;
        case MSE: gradient_output_layer_mse(output_layer, actual_values);
        break;
        case CCE: gradient_output_layer_cce(output_layer, actual_values);
        break;
        default:
        fprintf(stderr, "Error: Unsupported loss function type.\n");
        exit(EXIT_FAILURE);
    }
}

void calculate_gradient_hidden_layer(Layer *curr_layer, Layer *next_layer) {
    for (unsigned i = 0; i < curr_layer->num_neurons; ++i) {
        float sum_gradient = 0.0f;

        for (unsigned j = 0; j < next_layer->num_neurons; ++j) {
            sum_gradient += next_layer->neurons[j].gradient * curr_layer->neurons[i].weights[j];
        }

        float value = curr_layer->neurons[i].value;
        float activation_derivative = (curr_layer->activation == RELU) 
                                      ? (value > 0 ? 1.0f : 0.0f) 
                                      : (curr_layer->activation == SIGMOID) 
                                        ? value * (1.0f - value)
                                        : 1.0f - value * value; // For TANH

        curr_layer->neurons[i].gradient = sum_gradient * activation_derivative;
    }
}