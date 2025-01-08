
# Neural Network from Scratch in C

This project implements a simple feedforward neural network from scratch in C, with functionality to train and evaluate the network on the MNIST dataset. The network supports various activation functions, loss functions, and uses stochastic gradient descent (SGD) for optimization.

## Features
- **Feedforward Neural Network**: A basic implementation of a fully connected feedforward neural network.
- **Activation Functions**: Supports multiple activation functions including ReLU, Sigmoid, and Tanh.
- **Loss Functions**: Supports MSE (Mean Squared Error), BCE (Binary Cross-Entropy), and CCE (Categorical Cross-Entropy).
- **Optimization**: Implements SGD (Stochastic Gradient Descent) for weight updates.
- **Training**: Trains on the MNIST dataset using batches and multiple epochs.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/neural-network-c.git
   cd neural-network-c
   ```

2. **Install dependencies**:

   This project doesn't require any external libraries beyond the standard C libraries.

## How It Works

1. **Initialization**: The neural network is initialized with random weights and biases. The weights for hidden and output layers are initialized using Xavier or He initialization based on the activation function used.
2. **Forward Pass**: For each training example, the input data is passed through the network layer by layer, applying the activation function at each layer.
3. **Loss Calculation**: After the forward pass, the loss is calculated using the chosen loss function (MSE, BCE, or CCE).
4. **Backpropagation**: The gradients of the loss with respect to each weight are computed using backpropagation. These gradients are then used to update the weights using stochastic gradient descent (SGD).
5. **Training Loop**: The training loop iterates over the dataset for a set number of epochs. The dataset is processed in batches.


## Training Configuration

The following parameters can be adjusted in the `main.c` file:

- **`batch_size`**: The number of training samples per batch (default: 64).
- **`epochs`**: The number of times the entire training dataset is processed (default: 20).
- **`learning_rate`**: The learning rate for SGD (default: 0.01).
- **`loss_type`**: The loss function used during training. Options:
  - `MSE` (Mean Squared Error)
  - `BCE` (Binary Cross-Entropy)
  - `CCE` (Categorical Cross-Entropy)
- **`activation`**: The activation function used for hidden layers. Options:
  - `RELU`
  - `SIGMOID`
  - `TANH`

## Output

The program will print the training loss at the end of each epoch:

```bash
Loss: 0.34123
Loss: 0.32567
Loss: 0.31211
...
```

## Potential Improvements

- **Optimization**: Implement more advanced optimization techniques like Adam or RMSprop.
- **Regularization**: Add regularization techniques like L2 regularization or dropout to prevent overfitting.
- **GPU Support**: Extend the code to run on GPUs for faster computation.
- **Evaluation**: Implement a test set evaluation after training to measure the final accuracy.

## Contributions

Feel free to fork the repository, make improvements, or submit pull requests.

## Acknowledgments

- The MNIST dataset is provided by [Yann LeCun](http://yann.lecun.com/exdb/mnist/).
- The network design and algorithms are inspired by common neural network implementations used in machine learning research.