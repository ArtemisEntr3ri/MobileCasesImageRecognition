import numpy as np
import random
import pickle

import matplotlib.pyplot as plt

class Network:

    def __init__(self, sizes=[]):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]

        self.parameter_num = np.dot(sizes[:-1], sizes[1:]) + sum(sizes[1:])
        return

    def feedforward(self, a):
        """Returns output of network if a is input."""

        for W, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(W, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, regularization_factor, test_data = None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if len(test_data) > 0:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, regularization_factor, n)

            if len(test_data) > 0:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
        return

    def update_mini_batch(self, mini_batch, learning_rate, regularization_factor, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "learning_rate"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Aproksimiramo parcijalnu derivaciju cost funkcije
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Dodajemo efekt regularizacije
        nabla_w = [nw + regularization_factor/n * w for w, nw in zip(self.weights, nabla_w)]

        # Gradient descend pravilo
        self.weights = [w - learning_rate/len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - learning_rate/len(mini_batch)* nb for b, nb in zip(self.biases, nabla_b)]

        return


    def backprop(self,x, y):
        """Applies backpropagation algorithm to a single input (x,y). Calculates partial derivatives for that input.
        Partial derivative of cost function for multiple examples is average over those examples."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x

        activations = [x]  # list to store all activations, layer by layer
        zs = [] # list to store all z vectors, layer by layer

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)

            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derived(zs[-1])

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for l in range(2, self.num_layers):
            z = zs[-l]

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_derived(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        #return sum(int(x == y) for (x, y) in test_results)
        return sum(int(y[x] == 1) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y

    def save(self, name='last_object'):
        """Saves network and its parameters to Objects folder"""

        with open('Objects\\' + name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        return

    def load(self, name='last_object'):
        """Returns loaded network. To load network create empty network called net and call
        net = net.load()"""

        with open('Objects\\' + name, 'rb') as f:
            return pickle.load(f)

# Miscellaneous functions

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_derived(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


if __name__ == "__main__":

    with open('Objects\\debug_dataset', 'rb') as f:
        data = pickle.load(f)

    data = np.array(data)

    print(data[1][0].shape)
    #plt.imshow(data[1][0].reshape(20, 10))
    #plt.show()
    # print(data[0][0].shape)
    # print(data[0][1].shape)
    # for item in data:
    #     print(item[0].shape)

    random.shuffle(data)

    train_data = data[0:3500]
    test_data = data[3501:]

    net = Network([1600, 17])
    print(net.parameter_num)

    net.SGD(train_data, epochs=30,
            mini_batch_size=20,
            learning_rate=0.3,
            regularization_factor=100000,
            test_data=test_data)

    print(1)