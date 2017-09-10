#%%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple


def relu(z):
    return np.maximum(0, z)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def relu_derivative(z):
    return np.where(z >= 0, 1.0, 0.0)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def sigmoid_backward(dA, z):
    s = sigmoid(z)
    return dA * s * (1 - s)


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


class GradDescentOptimizer:
    def __init__(self):
        self.alpha = 0.0075
        self.debug = True
        self.iters = 3000

    def optimize(self, cost_fn, params):
        costs = []
        for i in range(self.iters):
            cost, grad = cost_fn(params)
            params = params - self.alpha * grad
            if i % 100 == 0:
                costs.append(cost)
                if self.debug:
                    print('cost[{}]: {}'.format(i, cost))
        return params, costs


class NnClassifier:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.params = None
        self.lambdap = 1.0

    def predict(self, X):
        assert X.shape[0] == self.model.layers[0].dim
        A = X
        for l in range(1, len(self.model.layers)):
            W, b = self.model._get_layer_params(self.params, l)
            activation_f = self.model.layers[l].activation[0]
            Z = W.dot(A) + b
            A = activation_f(Z)
        Y_pred = A > 0.5
        assert Y_pred.shape == (1, X.shape[1])
        return Y_pred

    def train(self, X, Y):
        def J(params): return self.model.cost(params, X, Y, self.lambdap)
        guess = self.model.init_params()
        self.params, costs = self.optimizer.optimize(J, guess)
        return costs


NnLayer = namedtuple('NnLayer', ['dim', 'activation'])


class NnModel:
    def __init__(self):
        self.layers = []
        # Compiled
        self.layer_slices = []
        self.activations = {
            'relu': (relu, relu_backward),
            'sigmoid': (sigmoid, sigmoid_backward)
        }

    def add(self, units, activation, input_dim=0):
        if input_dim != 0 and not self.layers:
            self.layers.append(NnLayer(input_dim, None))
        self.layers.append(NnLayer(units, self.activations[activation]))

    def compile(self):
        offset = 0
        for l in range(1, len(self.layers)):
            w_slice = (offset, offset + self.layers[l - 1].dim * self.layers[l].dim)
            b_slice = (w_slice[1], w_slice[1] + self.layers[l].dim)
            self.layer_slices.append((w_slice, b_slice))
            offset += self.layers[l - 1].dim * self.layers[l].dim + self.layers[l].dim

    def init_params(self, epsilon=0.01):
        np.random.seed(1)
        dim = 0
        for l in range(1, len(self.layers)):
            dim += self.layers[l].dim * \
                self.layers[l - 1].dim + self.layers[l].dim
        params = np.zeros(dim, dtype=np.float32)
        for l in range(1, len(self.layers)):
            W = np.random.randn(
                self.layers[l].dim, self.layers[l - 1].dim) / np.sqrt(self.layers[l - 1].dim)  # * epsilon
            b = np.zeros((self.layers[l].dim, 1), dtype=np.float32)
            self._set_layer_params(params, l, W, b)
        return params

    def cost(self, params, X, Y, lambdap):
        A, Z = self._propagate_forward(params, X)
        cost = self._compute_cost(params, A[-1], Y, lambdap)
        grad = self._propagate_backward(params, A, Z, Y, lambdap)
        return cost, grad

    def _compute_cost(self, params, AL, Y, lambdap):
        m = AL.shape[1]
        L = Y * np.log(AL) + (1. - Y) * np.log(1. - AL)
        J = -(1. / m) * np.sum(L)
        for l in range(1, len(self.layers)):
            W, b = self._get_layer_params(params, l)
            J += (lambdap / (2 * m)) * np.sum(np.square(W))
        return np.squeeze(J)

    def _propagate_forward(self, params, X):
        A = [X]
        Z = [[]]
        for l in range(1, len(self.layers)):
            W, b = self._get_layer_params(params, l)
            activation_f = self.layers[l].activation[0]
            Z.append(W.dot(A[l - 1]) + b)
            A.append(activation_f(Z[l]))
            assert(A[l].shape == (self.layers[l].dim, X.shape[1]))
        return A, Z

    def _propagate_backward(self, params, A, Z, Y, lambdap):
        m = A[-1].shape[1]
        grad = np.zeros(params.shape, dtype=np.float32)
        dA = -(np.divide(Y, A[-1]) - np.divide(1 - Y, 1 - A[-1]))
        for l in reversed(range(1, len(self.layers))):
            W, b = self._get_layer_params(params, l)
            activation_df = self.layers[l].activation[1]
            dZ = activation_df(dA, Z[l])
            assert(dZ.shape == A[l].shape)
            dA_prev, dW, db = self._linear_backward(dZ, A[l - 1], W, b, lambdap)
            dA = dA_prev
            self._set_layer_params(grad, l, dW, db)
        return grad

    def _linear_backward(self, dZ, A_prev, W, b, lambdap):
        m = A_prev.shape[1]
        dW = (1. / m) * np.dot(dZ, A_prev.T)
        dW += ((lambdap / m) * W)
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def _get_layer_params(self, params, l):
        w_slice, b_slice = self.layer_slices[l - 1]
        W = params[w_slice[0]:w_slice[1], ...].reshape(
            (self.layers[l].dim, self.layers[l - 1].dim))
        b = params[b_slice[0]:b_slice[1], ...].reshape((self.layers[l].dim, 1))
        return W, b

    def _set_layer_params(self, params, l, W, b):
        assert(W.shape == (self.layers[l].dim, self.layers[l - 1].dim))
        assert(b.shape == (self.layers[l].dim, 1))
        w_slice, b_slice = self.layer_slices[l - 1]
        params = params.reshape((-1, 1))
        params[w_slice[0]:w_slice[1], ...] = W.reshape((-1, 1))
        params[b_slice[0]:b_slice[1], ...] = b.reshape((-1, 1))


Dataset = namedtuple('Dataset', ['X', 'Y'])


def load_dataset(file_name, prefix):
    model = h5py.File(file_name, 'r')
    X = np.array(model[prefix + '_x'][:], dtype=np.float32)
    X = X.reshape((X.shape[0], -1)).T
    X = X / 255
    Y = np.array(model[prefix + '_y'][:], dtype=np.float32)
    Y = Y.reshape((1, Y.shape[0]))
    return Dataset(X, Y)


def main():
    ds_train = load_dataset('datasets/images_train.h5', 'train_set')
    ds_test = load_dataset('datasets/images_test.h5', 'test_set')
    print('{} X{} Y{}'.format('train', ds_train.X.shape, ds_train.Y.shape))
    print('{} X{} Y{}'.format('test', ds_test.X.shape, ds_test.Y.shape))
    # Train model
    model = NnModel()
    model.add(20, 'relu', input_dim=ds_train.X.shape[0])
    model.add(7, 'relu')
    model.add(5, 'relu')
    model.add(1, 'sigmoid')
    model.compile()
    classifier = NnClassifier(model, GradDescentOptimizer())
    classifier.lambdap = 0.8
    start = time.time()
    costs = classifier.train(ds_train.X, ds_train.Y)
    end = time.time()
    # Compute predicitions
    Yp_train = classifier.predict(ds_train.X)
    Yp_test = classifier.predict(ds_test.X)
    print('train accuracy: {} %, took {}'.format(
        100 - np.mean(np.abs(Yp_train - ds_train.Y)) * 100, end - start))
    print('test accuracy: {} %'.format(
        100 - np.mean(np.abs(Yp_test - ds_test.Y)) * 100))
    # Plot cost
    plt.plot(np.squeeze(costs))
    plt.title('Learning rate = {}'.format(classifier.optimizer.alpha))
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('cost')
    plt.show()


main()
