// activation.c
/*Contains all The Activation Functions used at each layer*/

#include "defs.h"

static float relu (float input) {
    return input > 0 ? input : 0;
}

static float sigmoid (float input) {
    if (input > 88.0f) {
        input = 88.0f;
    } else if (input < -88.0f) {
        input = -88.0f;
    }

    float result = 1.0f / (1.0f + expf(-input));
    return result;
}

float activation_function(ActivationFunction type, float input) {
    switch(type) {
        case RELU: return relu(input);
        case SIGMOID: return sigmoid(input);
        case TANH: return tanhf(input);
    }
    return NAN;
}


