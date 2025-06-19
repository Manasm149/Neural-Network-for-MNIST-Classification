"""network3.py

A Theano-based program for training and running simple neural
networks with support for fully connected, convolutional, max
pooling, and softmax layers. This program can utilize GPU for
faster computation.
"""

# Standard libraries
import cPickle  # For loading MNIST data
import gzip     # For decompressing MNIST data

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv  # Convolution
from theano.tensor.nnet import softmax  # Softmax for classification
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample  # Max pooling

# Activation functions
def linear(z): return z

def ReLU(z): return T.maximum(0.0, z)

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### GPU support configuration
GPU = True
if GPU:
    print "Trying to run under a GPU..."
    try:
        theano.config.device = 'gpu'
    except:
        pass  # Already configured
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU..."

#### Load the MNIST data and store in Theano shared variables
def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()

    def shared(data):
        # Store data on the GPU (if available) as shared variables
        shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")  # Return input and cast labels to int32

    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Network class implementing forward pass and SGD training
class Network(object):

    def __init__(self, layers, mini_batch_size):
        # Initialize the network with layers and batch size
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")  # Input placeholder
        self.y = T.ivector("y")  # Output labels placeholder

        # Set input to each layer
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)

        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        # Train the network using mini-batch stochastic gradient descent
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        num_training_batches = size(training_data) / mini_batch_size
        num_validation_batches = size(validation_data) / mini_batch_size
        num_test_batches = size(test_data) / mini_batch_size

        # Define cost function with L2 regularization
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)  # Compute gradients
        updates = [(param, param - eta*grad) for param, grad in zip(self.params, grads)]

        # Compile Theano functions for training, validation, and test
        i = T.lscalar()  # mini-batch index
        train_mb = theano.function([i], cost, updates=updates, givens={
            self.x: training_x[i*mini_batch_size: (i+1)*mini_batch_size],
            self.y: training_y[i*mini_batch_size: (i+1)*mini_batch_size]})

        validate_mb_accuracy = theano.function([i], self.layers[-1].accuracy(self.y), givens={
            self.x: validation_x[i*mini_batch_size: (i+1)*mini_batch_size],
            self.y: validation_y[i*mini_batch_size: (i+1)*mini_batch_size]})

        test_mb_accuracy = theano.function([i], self.layers[-1].accuracy(self.y), givens={
            self.x: test_x[i*mini_batch_size: (i+1)*mini_batch_size],
            self.y: test_y[i*mini_batch_size: (i+1)*mini_batch_size]})

        self.test_mb_predictions = theano.function([i], self.layers[-1].y_out, givens={
            self.x: test_x[i*mini_batch_size: (i+1)*mini_batch_size]})

        # Training loop
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)

                # Every epoch, compute validation accuracy
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean([validate_mb_accuracy(j)
                                                   for j in xrange(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))

                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean([test_mb_accuracy(j)
                                                     for j in xrange(num_test_batches)])
                            print("The corresponding test accuracy is {0:.2%}".format(test_accuracy))

        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Helper functions

def size(data):
    # Return number of examples in the dataset
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    # Apply dropout mask to the input layer
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)
