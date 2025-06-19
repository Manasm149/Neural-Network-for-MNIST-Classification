"""
network2.py
~~~~~~~~~~~~~~

An improved feedforward neural network implementation supporting:
- Cross-entropy and quadratic cost functions
- L2 regularization
- Better weight initialization
- Accuracy and cost monitoring during training
"""

# Standard library
import json
import random
import sys

# Third-party
import numpy as np

##########################################################
# Cost function classes
##########################################################

class QuadraticCost(object):
    """Standard quadratic (L2) cost: 0.5 * ||a - y||^2"""

    @staticmethod
    def fn(a, y):
        """Return the L2 loss between prediction a and target y"""
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta for the output layer"""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    """Cross-entropy loss, more stable and efficient for classification"""

    @staticmethod
    def fn(a, y):
        """Cross-entropy cost. Uses nan_to_num to prevent NaN from log(0)"""
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the delta for the output layer (z unused)"""
        return a - y

##########################################################
# Network class
##########################################################

class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Create a neural network.
        `sizes` is a list of integers specifying the number of neurons per layer.
        e.g. [784, 30, 10] = 3-layer network with 784 input, 30 hidden, 10 output.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """
        Initialize weights with N(0, 1/√n) and biases with N(0,1).
        Helps avoid vanishing/exploding gradients in deep networks.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        Initialize weights and biases with N(0,1) (not recommended).
        Included for legacy comparison with older implementations.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Compute network output from input `a`."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0, evaluation_data=None,
            monitor_evaluation_cost=False, monitor_evaluation_accuracy=False,
            monitor_training_cost=False, monitor_training_accuracy=False):
        """
        Train using mini-batch stochastic gradient descent.
        `eta`: learning rate
        `lmbda`: regularization parameter
        Optional evaluation_data enables accuracy/cost monitoring per epoch.
        Returns: 4 lists — evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)

        # Store monitoring data
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            print("Epoch {} training complete".format(j))

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {:.4f}".format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {:.4f}".format(cost))

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_data))

            print()

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Apply backpropagation and L2 regularization to update weights/biases."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Apply weight decay (L2) and gradient step
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Compute gradients via backpropagation for a single (x, y) pair."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate through layers l = L-1 to 1
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        Return number of correctly classified inputs.
        `convert=True` for training data (where y is vector), else y is label.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Compute total cost including L2 regularization."""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save network weights, biases, and structure to a fil
