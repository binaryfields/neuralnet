#%%
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple


def relu(z):
    return np.maximum(0, z)


def relu_backward(dA, Z):
    return np.multiply(dA, np.int64(Z > 0))


def relu_derv(z):
    return np.where(z >= 0, 1.0, 0.0)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def sigmoid_backward(dA, z):
    a = sigmoid(z)
    return dA * a * (1 - a)


def sigmoid_derv(z):
    a = sigmoid(z)
    return a * (1 - a)


class AdamOptimizer:
    def __init__(self, alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        # Runtime State
        self.grad_velocity = None
        self.grad_squares = None

    def init(self, params):
        self.grad_velocity = np.zeros(params.shape)
        self.grad_squares = np.zeros(params.shape)

    def optimize(self, cost_fn, params):
        cost, grad = cost_fn(params)
        self.grad_velocity = self.beta_1 * self.grad_velocity + (
            1. - self.beta_1) * grad
        self.grad_squares = self.beta_2 * self.grad_squares + (
            1. - self.beta_2) * np.square(grad)
        params = params - self.alpha * (self.grad_velocity / (
            np.sqrt(self.grad_squares) + self.epsilon))
        return params, cost


NnLayer = namedtuple('NnLayer', ['dim', 'activation', 'dropout'])


class NnModel:
    def __init__(self):
        self.layers = []
        # Compiled
        self.layer_offsets = []
        self.activations = {
            'relu': (relu, relu_derv, relu_backward),
            'sigmoid': (sigmoid, sigmoid_derv, sigmoid_backward)
        }

    def __getitem__(self, key):
        return self.layers[key]

    def __len__(self):
        return len(self.layers)

    def add(self, units, activation, dropout=0, input_dim=0):
        if input_dim != 0 and not self.layers:
            self.layers.append(NnLayer(input_dim, None, 0))
        self.layers.append(
            NnLayer(units, self.activations[activation], dropout))

    def compile(self):
        offset = 0
        self.layer_offsets = [(0, 0)]
        for layer in range(1, len(self.layers)):
            prev_layer_dim = self.layers[layer - 1].dim
            layer_dim = self.layers[layer].dim
            w_offset = (offset, offset + prev_layer_dim * layer_dim)
            b_offset = (w_offset[1], w_offset[1] + layer_dim)
            self.layer_offsets.append((w_offset, b_offset))
            offset += prev_layer_dim * layer_dim + layer_dim

    def pack_params(self, layer, params, W, b):
        assert (W.shape == (self.layers[layer].dim,
                            self.layers[layer - 1].dim))
        assert (b.shape == (self.layers[layer].dim, 1))
        w_offset, b_offset = self.layer_offsets[layer]
        params = params.reshape((-1, 1))
        params[w_offset[0]:w_offset[1], ...] = W.reshape((-1, 1))
        params[b_offset[0]:b_offset[1], ...] = b.reshape((-1, 1))

    def unpack_params(self, layer, params):
        prev_layer_dim = self.layers[layer - 1].dim
        layer_dim = self.layers[layer].dim
        w_offset, b_offset = self.layer_offsets[layer]
        W = params[w_offset[0]:w_offset[1], ...].reshape((layer_dim,
                                                          prev_layer_dim))
        b = params[b_offset[0]:b_offset[1], ...].reshape((layer_dim, 1))
        return W, b


class NnClassifier:
    def __init__(self,
                 model,
                 optimizer,
                 epochs,
                 batch_size=64,
                 lambd=0,
                 debug=False):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.debug = debug
        self.epochs = epochs
        self.lambd = lambd
        # Trained State
        self.params = None

    def predict(self, X):
        assert X.shape[0] == self.model[0].dim
        A = X
        for l in range(1, len(self.model)):
            layer = self.model[l]
            W, b = self.model.unpack_params(l, self.params)
            Z = np.dot(W, A) + b
            A = layer.activation[0](Z)
            assert A.shape == (layer.dim, X.shape[1])
        Y_pred = A > 0.5
        assert Y_pred.shape == (1, X.shape[1])
        return Y_pred

    def train(self, X, Y):
        costs = []
        self.params = self._init_params()
        self.optimizer.init(self.params)
        seed = 10
        for i in range(self.epochs):
            cost = 0
            seed += 1
            batches = self._partition_batches(X, Y, self.batch_size)
            for batch in batches:
                (batch_X, batch_Y) = batch
                cost_fn = lambda params: self._cost(params, batch_X, batch_Y)
                self.params, batch_cost = self.optimizer.optimize(
                    cost_fn, self.params)
                cost += batch_cost / len(batches)
            costs.append(cost)
            if self.debug and i % 100 == 0:
                print('cost[{}]: {}'.format(i, cost))
        return costs

    def _cost(self, params, X, Y):
        A, Z, D = self._propagate_forward(params, X)
        cost = self._compute_cost(params, A[-1], Y)
        grad = self._propagate_backward(params, A, Z, D, Y)
        return cost, grad

    def _compute_cost(self, params, AL, Y):
        m = AL.shape[1]
        # loss (1 x m)
        L = -(Y * np.log(AL) + (1 - Y) * np.log(1. - AL))
        # cross-entropy cost (scalar)
        J = (1. / m) * np.sum(L)
        for l in range(1, len(self.model)):
            # weights (n[l] x n[l-1]), bias (n[l] x 1)
            W, b = self.model.unpack_params(l, params)
            # regularized cost
            J += (self.lambd / (2 * m)) * np.sum(np.square(W))
        return J

    def _propagate_forward(self, params, X):
        A = [X]
        Z = [None]
        D = [None]
        for l in range(1, len(self.model)):
            layer = self.model[l]
            # weights (n[l] x n[l-1]), bias (n[l] x 1)
            W, b = self.model.unpack_params(l, params)
            # activation[l] (n(l) x m)
            Z.append(np.dot(W, A[l - 1]) + b)
            A.append(layer.activation[0](Z[l]))
            # dropout
            if layer.dropout > 0:
                D.append(
                    np.random.random_sample(A[l].shape) < (1. - layer.dropout))
                A[l] = A[l] * D[l]
                A[l] = A[l] / (1. - layer.dropout)
            else:
                D.append(None)
            assert A[l].shape == (layer.dim, X.shape[1])
        return A, Z, D

    def _propagate_backward(self, params, A, Z, D, Y):
        m = A[-1].shape[1]
        grad = np.zeros(params.shape, dtype=np.float)
        # dJ/dA (1 x m)
        dA = -np.divide(Y, A[-1]) + np.divide(1 - Y, np.maximum(
            1 - A[-1], 1e-8))
        for l in reversed(range(1, len(self.model))):
            layer = self.model[l]
            # weights (n[l] x n[l-1]), bias (n[l] x 1)
            W, b = self.model.unpack_params(l, params)
            # dropout
            if layer.dropout > 0:
                dA = dA * D[l]
                dA = dA / (1. - layer.dropout)
            # dJ/dZ = dJ/dA * dA/dZ (n[l] x m)
            dZ = layer.activation[2](dA, Z[l])
            assert dZ.shape == A[l].shape
            # dJ/dW = dJ/dZ * dZ/dW (dim TBD)
            dW, db, dA_prev = self._linear_backward(dZ, A[l - 1], W, b)
            dA = dA_prev
            self.model.pack_params(l, grad, dW, db)
        return grad

    def _linear_backward(self, dZ, A_prev, W, b):
        m = A_prev.shape[1]
        dW = (1. / m) * np.dot(dZ, A_prev.T)
        dW += (self.lambd / m) * W
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dW, db, dA_prev

    def _init_params(self, epsilon=0.01):
        np.random.seed(5)
        dim = 0
        for l in range(1, len(self.model)):
            dim += self.model[l].dim * self.model[l - 1].dim
            dim += self.model[l].dim
        params = np.zeros(dim, dtype=np.float)
        for l in range(1, len(self.model)):
            prev_layer_dim = self.model[l - 1].dim
            layer_dim = self.model[l].dim
            W = np.random.randn(layer_dim, prev_layer_dim) * np.sqrt(
                2. / prev_layer_dim)  #/ np.sqrt(prev_layer_dim)
            b = np.zeros((layer_dim, 1), dtype=np.float)
            self.model.pack_params(l, params, W, b)
        return params

    def _partition_batches(self, X, Y, batch_size):
        m = X.shape[1]
        batches = []
        n_batches = math.floor(m / batch_size)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))
        for k in range(n_batches):
            batch_X = shuffled_X[:, k * batch_size:(k + 1) * batch_size]
            batch_Y = shuffled_Y[:, k * batch_size:(k + 1) * batch_size]
            batches.append((batch_X, batch_Y))
        if m % batch_size != 0:
            batch_X = shuffled_X[:, n_batches * batch_size:m]
            batch_Y = shuffled_Y[:, n_batches * batch_size:m]
            batches.append((batch_X, batch_Y))
        return batches


def load_dataset(file_name, prefix):
    model = h5py.File(file_name, 'r')
    X = np.array(model[prefix + '_x'][:], dtype=np.float)
    X = X.reshape((X.shape[0], -1)).T
    X = X / 255
    Y = np.array(model[prefix + '_y'][:], dtype=np.int)
    Y = Y.reshape((1, Y.shape[0]))
    return (X, Y)


def main():
    (train_X, train_Y) = load_dataset('datasets/images_train.h5', 'train_set')
    (test_X, test_Y) = load_dataset('datasets/images_test.h5', 'test_set')
    print('{} X{} Y{}'.format('train', train_X.shape, train_Y.shape))
    print('{} X{} Y{}'.format('test', test_X.shape, test_Y.shape))
    # Train model
    model = NnModel()
    model.add(16, activation='relu', input_dim=train_X.shape[0], dropout=0)
    model.add(16, activation='relu', dropout=0)
    model.add(16, activation='relu', dropout=0)
    model.add(1, activation='sigmoid')
    model.compile()
    optimizer = AdamOptimizer(alpha=0.0001)
    classifier = NnClassifier(
        model, optimizer, epochs=1000, batch_size=256, lambd=1.8, debug=True)
    start = time.time()
    costs = classifier.train(train_X, train_Y)
    end = time.time()
    # Compute predicitions
    Yp_train = classifier.predict(train_X)
    Yp_test = classifier.predict(test_X)
    print('train accuracy: {} %, took {}'.format(
        100 - np.mean(np.abs(Yp_train - train_Y)) * 100, end - start))
    print('test accuracy: {} %'.format(
        100 - np.mean(np.abs(Yp_test - test_Y)) * 100))
    # Plot cost
    plt.plot(np.squeeze(costs))
    plt.title('Learning rate = {}'.format(classifier.optimizer.alpha))
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('cost')
    plt.show()


main()
