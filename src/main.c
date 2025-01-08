// main.c

#include "defs.h"

int main() {
    unsigned int num_train_images, num_train_labels;

    // Load MNIST images and labels
    float *train_images = load_mnist_images("dataset/train-images.idx3-ubyte", &num_train_images);
    unsigned char *train_labels = load_mnist_labels("dataset/train-labels.idx1-ubyte", &num_train_labels);

    printf("Loaded %d training images and %d training labels\n", num_train_images, num_train_labels);

    // Define the network structure
    Network network;
    network.num_layers = 4;     // 4 layers: input + 2 hidden + output
    network.num_input = 784;    // 28x28 images
    network.num_output = 10;    // 10 classes (digits 0-9)
    network.loss_type = MSE;    // Using Mean Squared Error for loss

    // Memory allocation for layers
    network.layers = malloc(network.num_layers * sizeof(Layer));

    // Input Layer (784 neurons, sigmoid activation)
    network.layers[0].type = INPUT_LAYER;
    network.layers[0].num_neurons = 784;
    network.layers[0].neurons = malloc(784 * sizeof(Neuron));
    network.layers[0].activation = SIGMOID;

    // Hidden Layer 1 (128 neurons, ReLU activation)
    network.layers[1].type = HIDDEN_LAYER;
    network.layers[1].num_neurons = 128;
    network.layers[1].neurons = malloc(128 * sizeof(Neuron));
    network.layers[1].activation = RELU;

    // Hidden Layer 2 (128 neurons, ReLU activation)
    network.layers[2].type = HIDDEN_LAYER;
    network.layers[2].num_neurons = 128;
    network.layers[2].neurons = malloc(128 * sizeof(Neuron));
    network.layers[2].activation = SIGMOID;

    // Output Layer (10 neurons, sigmoid activation)
    network.layers[3].type = OUTPUT_LAYER;
    network.layers[3].num_neurons = 10;
    network.layers[3].neurons = malloc(10 * sizeof(Neuron));
    network.layers[3].activation = RELU;

    // Initialize network weights and biases
    initialize_network(&network);

    printf("Initialized Network....\n");

    int batch_size = 64;     // Batch size
    int epochs = 20;         // Number of epochs
    float learning_rate = 0.01f;

    // Training loop: Process each epoch and each batch
    for (int epoch = 0; epoch < epochs; ++epoch) {
        printf("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n\n\n");
        float epoch_loss = 0.0f;
        int num_batches = num_train_images / batch_size;

        for (int batch = 0; batch < num_batches; ++batch) {
            printf("EPOCH : %d\t\tBATCH : %d\n", epoch+1, batch+1);
            float batch_inputs[batch_size][784];
            float batch_actual_values[batch_size][10];

            for (int i = 0; i < batch_size; ++i) {
                int image_idx = batch * batch_size + i;
                
                for (int j = 0; j < 784; ++j) {
                    batch_inputs[i][j] = train_images[image_idx * 784 + j] / 255.0f;
                }
                
                one_hot_encode(train_labels[image_idx], batch_actual_values[i]);
            }

            // Training process for the current batch
            for (int i = 0; i < batch_size; ++i) {
                // Set the input values for the network
                set_input_value(&network.layers[0], batch_inputs[i]);
                
                // Perform backpropagation
                backpropogation(&network, batch_actual_values[i], learning_rate);
                printf("3\n");
            }

            // Calculate and accumulate the loss for monitoring
            epoch_loss += loss_function(network.loss_type, &network.layers[network.num_layers - 1], batch_actual_values[0]);
            printf("4\n");
        }

        // Print loss after each epoch
        printf("Epoch %d, Loss: %.5f\n", epoch + 1, epoch_loss / num_batches);
    }

    // Free the memory after training
    for (unsigned i = 0; i < network.num_layers; ++i) {
        free(network.layers[i].neurons);
    }
    free(network.layers);
    free(train_images);
    free(train_labels);

    return 0;
}
