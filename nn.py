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


def relu_derv(z):
    return np.where(z >= 0, 1.0, 0.0)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def sigmoid_backward(dA, z):
    s = sigmoid(z)
    return dA * s * (1 - s)


def sigmoid_derv(z):
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


NnLayer = namedtuple('NnLayer', ['dim', 'activation'])


class NnModel:
    def __init__(self):
        self.layers = []
        # Compiled
        self.layer_offsets = []
        self.activations = {
            'relu': (relu, relu_derv, relu_backward),
            'sigmoid': (sigmoid, sigmoid_derv, sigmoid_backward)
        }

    def add(self, units, activation, input_dim=0):
        if input_dim != 0 and not self.layers:
            self.layers.append(NnLayer(input_dim, None))
        self.layers.append(NnLayer(units, self.activations[activation]))

    def compile(self):
        offset = 0
        self.layer_offsets = [ (0, 0) ]
        for layer in range(1, len(self.layers)):
            prev_layer_dim = self.layers[layer - 1].dim
            layer_dim = self.layers[layer].dim
            w_offset = (offset, offset + prev_layer_dim * layer_dim)
            b_offset = (w_offset[1], w_offset[1] + layer_dim)
            self.layer_offsets.append((w_offset, b_offset))
            offset += prev_layer_dim * layer_dim + layer_dim

    def get_activation_fn(self, layer):
        return self.layers[layer].activation[0]

    def get_activation_backward(self, layer):
        return self.layers[layer].activation[2]

    def get_count(self):
        return len(self.layers)

    def get_dim(self, layer):
        return self.layers[layer].dim
       
    def get_params(self, layer, params):
        prev_layer_dim = self.layers[layer - 1].dim
        layer_dim = self.layers[layer].dim
        w_offset, b_offset = self.layer_offsets[layer]
        W = params[w_offset[0]:w_offset[1], ...].reshape((layer_dim, prev_layer_dim))
        b = params[b_offset[0]:b_offset[1], ...].reshape((layer_dim, 1))
        return W, b

    def set_params(self, layer, params, W, b):
        assert(W.shape == (self.layers[layer].dim, self.layers[layer - 1].dim))
        assert(b.shape == (self.layers[layer].dim, 1))
        w_offset, b_offset = self.layer_offsets[layer]
        params = params.reshape((-1, 1))
        params[w_offset[0]:w_offset[1], ...] = W.reshape((-1, 1))
        params[b_offset[0]:b_offset[1], ...] = b.reshape((-1, 1))


class NnClassifier:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.params = None
        self.lambd = 1.0

    def predict(self, X):
        assert X.shape[0] == self.model.get_dim(0)
        A = X
        for layer in range(1, self.model.get_count()):
            W, b = self.model.get_params(layer, self.params)
            activation_fn = self.model.get_activation_fn(layer)
            Z = np.dot(W, A) + b
            A = activation_fn(Z)
            assert A.shape == (self.model.get_dim(layer), X.shape[1])
        Y_pred = A > 0.5
        assert Y_pred.shape == (1, X.shape[1])
        return Y_pred

    def train(self, X, Y):
        def J(params): return self._cost(params, X, Y)
        guess = self._init_params()
        self.params, costs = self.optimizer.optimize(J, guess)
        return costs

    def _init_params(self, epsilon=0.01):
        np.random.seed(1)
        dim = 0
        for l in range(1, self.model.get_count()):
            dim += self.model.get_dim(l) * self.model.get_dim(l - 1) + self.model.get_dim(l)
        params = np.zeros(dim, dtype=np.float32)
        for l in range(1, self.model.get_count()):
            prev_layer_dim = self.model.get_dim(l - 1)
            layer_dim = self.model.get_dim(l)
            W = np.random.randn(layer_dim, prev_layer_dim) / np.sqrt(prev_layer_dim) 
            b = np.zeros((layer_dim, 1), dtype=np.float32)
            self.model.set_params(l, params, W, b)
        return params

    def _cost(self, params, X, Y):
        A, Z = self._propagate_forward(params, X)
        cost = self._compute_cost(params, A[-1], Y)
        grad = self._propagate_backward(params, A, Z, Y)
        return cost, grad

    def _compute_cost(self, params, AL, Y):
        m = AL.shape[1]
        # loss (1 x m)
        L = -(Y * np.log(AL) + (1 - Y) * np.log(1. - AL))
        # cost (scalar)
        J = (1. / m) * np.sum(L)
        for layer in range(1, self.model.get_count()):
            # weights (n[l] x n[l-1]), bias (n[l] x 1)
            W, b = self.model.get_params(layer, params)
            # regularized cost
            J += (self.lambd / (2 * m)) * np.sum(np.square(W))
        return J

    def _propagate_forward(self, params, X):
        A = [ X ]
        Z = [ None ]
        for layer in range(1, self.model.get_count()):
            # weights (n[l] x n[l-1]), bias (n[l] x 1)
            W, b = self.model.get_params(layer, params)
            activation_fn = self.model.get_activation_fn(layer)
            # activation[l] (n(l) x m)
            Z.append(np.dot(W, A[layer - 1]) + b)
            A.append(activation_fn(Z[layer]))
            assert A[layer].shape == (self.model.get_dim(layer), X.shape[1])
        return A, Z

    def _propagate_backward(self, params, A, Z, Y):
        m = A[-1].shape[1]
        grad = np.zeros(params.shape, dtype=np.float32)
        # dJ/dA (1 x m)
        dA = -(np.divide(Y, A[-1]) - np.divide(1 - Y, 1 - A[-1]))
        for layer in reversed(range(1, self.model.get_count())):
            # weights (n[l] x n[l-1]), bias (n[l] x 1)
            W, b = self.model.get_params(layer, params)
            # dJ/dZ = dJ/dA * dA/dZ (n[l] x m)
            dZ = self.model.get_activation_backward(layer)(dA, Z[layer])
            assert dZ.shape == A[layer].shape
            # dJ/dW = dJ/dZ * dZ/dW (dim TBD)
            dW, db, dA_prev = self._linear_backward(dZ, A[layer - 1], W, b)
            dA = dA_prev
            self.model.set_params(layer, grad, dW, db)
        return grad

    def _linear_backward(self, dZ, A_prev, W, b):
        m = A_prev.shape[1]
        dW = (1. / m) * np.dot(dZ, A_prev.T)
        dW += ((self.lambd / m) * W)
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dW, db, dA_prev


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
