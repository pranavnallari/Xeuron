// defs.h
#ifndef DEFS_H
#define DEFS_H

// import stuff
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <string.h>

// LIMITS
#define MAX_INPUT_NEURONS 1024
#define MAX_OUTPUT_NEURONS 100
#define MAX_LAYERS 10
#define MAX_NEURONS_PER_LAYER 2048
#define MAX_BATCH_SIZE 64

#define IMAGE_SIZE 28*28    // in bytess
#define LABEL_SIZE 1        // in bytes

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
    float *weights;    // weight
    float value;    // input feauture or output after activation
    float bias;
    float gradient; // hold the gradient (dLoss/dValue) at every neuron
} Neuron;

typedef struct {
        LayerType type;      // type of layer (input, hidden, output)            
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
// backward_pass.c
extern void calculate_output_layer_gradient(Layer *output_layer, float actual_values[], LossFunction type);
extern void calculate_gradient_hidden_layer(Layer *curr_layer, Layer *next_layer);
// update.c
extern void update_parameters_sgd(Network *network, float learning_rate);
// backpropogation.c
extern void backpropogation(Network *network, float actual_values[], float learning_rate);
// data.c
extern float* load_mnist_images(const char *filename, unsigned int *num_images);
extern unsigned char* load_mnist_labels(const char *filename, unsigned int *num_labels);
extern void set_input_value(Layer *input_layer, float *image);
// init.c
extern void initialize_network(Network *network);
extern void one_hot_encode(unsigned char label, float actual_values[10]);
#endif