// backpropogation.c

/*This code contains the main logic for backpropogation*/

#include "defs.h"

void backpropogation(Network *network, float actual_values[], float learning_rate) {
    printf("a\n");
    forward_propagation(network);
    printf("b\n");
    float loss = loss_function(network->loss_type, &network->layers[network->num_layers-1], actual_values);
    printf("Loss : %.5f\n", loss);

    calculate_output_layer_gradient(&network->layers[network->num_layers-1], actual_values, network->loss_type);
    printf("c\n");
    for (int i = network->num_layers - 2; i >= 0; --i) {
        calculate_gradient_hidden_layer(&network->layers[i], &network->layers[i+1]);
        printf("d\n");
    }

    update_parameters_sgd(network, learning_rate);
    printf("e\n");
}