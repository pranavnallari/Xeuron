// loss_function.c
/*This function contains all the loss functions to calculate the cost*/

#include "defs.h"

static float mse(Layer *output_layer, float actual_values[]) {
    if (output_layer->num_neurons == 0) {
        fprintf(stderr, "Error: Output layer has no neurons.\n");
        exit(EXIT_FAILURE);
    }
    float result = 0.0f;
    for (unsigned i = 0; i < output_layer->num_neurons; ++i) {
        float error = output_layer->neurons[i].value - actual_values[i];
        result += error * error;
    }
    return result/output_layer->num_neurons;
}

static float bce(Layer *output_layer, float actual_values[]) {
    if (output_layer->num_neurons == 0) {
        fprintf(stderr, "Error: Output layer has no neurons.\n");
        exit(EXIT_FAILURE);
    }

    float result = 0.0f;
    const float epsilon = FLT_MIN; // Small constant to avoid log(0)

    for (unsigned i = 0; i < output_layer->num_neurons; ++i) {
        float predicted = output_layer->neurons[i].value;

        if (predicted < epsilon) predicted = epsilon;
        if (predicted > 1 - epsilon) predicted = 1 - epsilon;

        result += actual_values[i] * logf(predicted) + (1 - actual_values[i]) * logf(1 - predicted);
    }

    return -result / output_layer->num_neurons;
}


static float cce(Layer *output_layer, float actual_values[]) {
    if (output_layer->num_neurons == 0) {
        fprintf(stderr, "Error: Output layer has no neurons.\n");
        exit(EXIT_FAILURE);
    }

    float result = 0.0f;
    const float epsilon = FLT_MIN; // Small constant to avoid log(0)

    for (unsigned i = 0; i < output_layer->num_neurons; ++i) {
        float predicted = output_layer->neurons[i].value;

        if (predicted < epsilon) predicted = epsilon;
        if (predicted > 1 - epsilon) predicted = 1 - epsilon;

        if (actual_values[i] != 0 && actual_values[i] != 1) {
            fprintf(stderr, "Error: Actual values must be 0 or 1 for CCE.\n");
            exit(EXIT_FAILURE);
        }

        result += actual_values[i] * logf(predicted);
    }

    return -result;
}

float loss_function(LossFunction type, Layer *output_layer, float actual_values[]) {
    switch(type) {
        case MSE: return mse(output_layer, actual_values);
        case BCE: return bce(output_layer, actual_values);
        case CCE: return cce(output_layer, actual_values);
    }

    return NAN;
}