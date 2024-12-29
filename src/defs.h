// defs.h
#ifndef DEFS_H
#define DEFS_H

// import stuff
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <string.h>

// LIMITS
#define MAX_INPUT 1024
#define MAX_OUTPUT 100
#define MAX_LAYERS 10
#define MAX_NEURONS_PER_LAYER 2048
#define MAX_BATCH_SIZE 64

// Enums
typedef enum {
    INPUT_LAYER = 0,
    HIDDEN_LAYER,
    OUTPUT_LAYER
} LayerType;

typedef enum {
    RELU,
    SIGMOID,
    TANH
} ActivationFunction;

typedef enum {
    MSE,      // Mean Squared Error
    BCE,      // Binary Cross-Entropy
    CCE       // Categorical Cross-Entropy
} LossFunction;


// Structs
typedef struct {
    float weight;    // weight
    float value;    // input feauture or output after activation
} Neuron;

typedef struct {
        LayerType type;      // type of layer (input, hidden, output)
        float bias;             
        unsigned int num_neurons;
        Neuron *neurons;
        ActivationFunction activation;
} Layer;

typedef struct {
    unsigned int num_layers;
    unsigned int num_input;
    unsigned int num_output;
    Layer *layers;
    LossFunction loss_type;
} Network;

//activation.c
extern float activation_function(ActivationFunction type, float input);
// forward_pass.c
extern void forward_propagation(Network *network);
// loss_function.c
extern float loss_function(LossFunction type, Layer *output_layer, float actual_values[]);
#endif