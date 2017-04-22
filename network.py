import numpy as np


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        return

    def feedforward(self, a):
        """Returns output of network if a is input."""

        for W, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(W, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

if __name__ == "__main__":

    net = Network([5, 3, 2])

    print(net.biases)
    print(net.weights)