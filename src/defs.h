// defs.h
#ifndef DEFS_H
#define DEFS_H

// import stuff
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
    TANH,
    SOFTMAX
} ActivationFunction;

// Structs
typedef struct {
    float weight;    // weight
    float input;    // input feauture
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
} Network;

// main.c
// data.c
// init.c

#endif