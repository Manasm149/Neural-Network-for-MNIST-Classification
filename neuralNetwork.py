import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        """
        Initialize the neural network.
        `sizes` is a list containing the number of neurons in each layer.
        For example, [2, 3, 1] represents a 3-layer network:
        - 2 neurons in the input layer
        - 3 in the hidden layer
        - 1 in the output layer
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases: no bias for input layer, only for layers 1 to L
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights randomly from normal distribution
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Return the output of the network for input `a` by applying
        weights and biases layer by layer and using the sigmoid activation function.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        - training_data: list of tuples (x, y)
        - epochs: number of epochs to train
        - mini_batch_size: number of training samples per mini-batch
        - eta: learning rate
        - test_data: optional, if provided, evaluates accuracy after each epoch
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)  # Shuffle the data each epoch
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation for a single mini-batch.
        """
        # Initialize gradient accumulators for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Accumulate gradients from each training example
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Update weights and biases using averaged gradients
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple (nabla_b, nabla_w) representing the gradient of the cost
        function with respect to biases and weights.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass
        activation = x
        activations = [x]  # Store activations layer by layer
        zs = []            # Store z vectors (weighted inputs) layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        # Compute the error at the output layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta  # Gradient for the output bias
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # Gradient for weights

        # Backpropagate the error
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs the correct result.
        Assumes the output is a classification with the highest activation representing the label.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives ∂C_x / ∂a for the output activations.
        This is the gradient of the cost with respect to the output.
        """
        return (output_activations - y)

# Sigmoid activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
