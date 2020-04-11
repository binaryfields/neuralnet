# %%
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple


NnLayer = namedtuple('NnLayer', ['units', 'activation', 'dropout'])


class NnModel:
    def __init__(self, lambd):
        self.lambd = lambd
        self.layers = []
        # Compiled State
        self.layer_offsets = []
        self.activations = {
            'relu': (relu_forward, None, relu_backward),
            'sigmoid': (sigmoid_forward, None, sigmoid_backward),
        }
        # Trained State
        self.params = None

    def add(self, units, activation, dropout=0, input_dim=0):
        if input_dim != 0 and not self.layers:
            self.layers.append(NnLayer(input_dim, None, 0))
        self.layers.append(NnLayer(units, self.activations[activation], dropout))

    def compile(self):
        offset = 0
        self.layer_offsets = [(0, 0)]
        for layer in range(1, len(self.layers)):
            prev_layer_dim = self.layers[layer - 1].units
            layer_dim = self.layers[layer].units
            w_offset = (offset, offset + prev_layer_dim * layer_dim)
            b_offset = (w_offset[1], w_offset[1] + layer_dim)
            self.layer_offsets.append((w_offset, b_offset))
            offset += prev_layer_dim * layer_dim + layer_dim

    def init_params(self):
        np.random.seed(5)
        dim = 0
        for l in range(1, len(self.layers)):
            dim += self.layers[l].units * self.layers[l - 1].units
            dim += self.layers[l].units
        self.params = np.zeros(dim, dtype=np.float)
        for l in range(1, len(self.layers)):
            prev_layer_dim = self.layers[l - 1].units
            layer_dim = self.layers[l].units
            W = np.random.randn(layer_dim, prev_layer_dim) * np.sqrt(2.0 / prev_layer_dim)
            b = np.zeros((layer_dim, 1), dtype=np.float)
            self._pack_params(l, self.params, W, b)

    def loss(self, params, x, y):
        a, cache = self._propagate_forward(params, x)
        cost = self._compute_cost(params, a, y)
        da = -np.divide(y, a) + np.divide(1 - y, np.maximum(1 - a, 1e-8))
        grad = self._propagate_backward(params, da, cache)
        return cost, grad

    def predict(self, x):
        assert x.shape[0] == self.layers[0].units
        a = x
        for l in range(1, len(self.layers)):
            layer = self.layers[l]
            weights, bias = self._unpack_params(l, self.params)
            z = np.dot(weights, a) + bias
            a, _ = layer.activation[0](z)
            assert a.shape == (layer.units, x.shape[1])
        y_pred = a > 0.5
        assert y_pred.shape == (1, x.shape[1])
        return y_pred

    def _compute_cost(self, params, a, y):
        m = a.shape[1]
        loss = -(y * np.log(a) + (1 - y) * np.log(1.0 - a))
        assert loss.shape == (1, m)
        # cross-entropy cost (scalar)
        cost = (1.0 / m) * np.sum(loss)
        # regularized cost
        for l in range(1, len(self.layers)):
            weights, _ = self._unpack_params(l, params)
            cost += (self.lambd / (2 * m)) * np.sum(np.square(weights))
        return cost

    def _propagate_forward(self, params, x):
        cache = [None]
        a = x
        for l in range(1, len(self.layers)):
            layer = self.layers[l]
            weights, bias = self._unpack_params(l, params)
            assert weights.shape == (layer.units, self.layers[l - 1].units)
            assert bias.shape == (layer.units, 1)
            z, z_cache = linear_forward(a, weights, bias)
            assert z.shape == (layer.units, x.shape[1])
            a, a_cache = layer.activation[0](z)
            assert a.shape == (layer.units, x.shape[1])
            cache.append((z_cache, a_cache))
        return a, cache

    def _propagate_backward(self, params, dout, cache):
        m = dout.shape[1]
        grad = np.zeros(params.shape, dtype=np.float)
        da = dout
        for l in reversed(range(1, len(self.layers))):
            layer = self.layers[l]
            weights, bias = self._unpack_params(l, params)
            z_cache, a_cache = cache[l]
            # dJ/dZ = dJ/dA * dA/dZ
            dz = layer.activation[2](da, a_cache)
            assert dz.shape == (layer.units, m)
            # dJ/dW = dJ/dZ * dZ/dW
            da_prev, dw, db = linear_backward(dz, z_cache)
            assert da_prev.shape == (self.layers[l - 1].units, m)
            assert dw.shape == weights.shape
            assert db.shape == bias.shape
            self._pack_params(l, grad, dw, db)
            da = da_prev
        return grad

    def _pack_params(self, layer, params, W, b):
        assert W.shape == (self.layers[layer].units, self.layers[layer - 1].units)
        assert b.shape == (self.layers[layer].units, 1)
        w_offset, b_offset = self.layer_offsets[layer]
        params = params.reshape((-1, 1))
        params[w_offset[0] : w_offset[1], ...] = W.reshape((-1, 1))
        params[b_offset[0] : b_offset[1], ...] = b.reshape((-1, 1))

    def _unpack_params(self, layer, params):
        prev_layer_dim = self.layers[layer - 1].units
        layer_dim = self.layers[layer].units
        w_offset, b_offset = self.layer_offsets[layer]
        W = params[w_offset[0] : w_offset[1], ...].reshape((layer_dim, prev_layer_dim))
        b = params[b_offset[0] : b_offset[1], ...].reshape((layer_dim, 1))
        return W, b


class NnTrainer:
    def __init__(self, model, optimizer, epochs, batch_size=64, debug=False):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.debug = debug
        self.epochs = epochs

    def train(self, x, y):
        costs = []
        self.model.init_params()
        self.optimizer.init(self.model.params)
        seed = 10
        for i in range(self.epochs):
            cost = 0
            seed += 1
            batches = self._partition_batches(x, y, self.batch_size)
            for batch in batches:
                batch_x, batch_y = batch

                def cost_fn(params):
                    return self.model.loss(params, batch_x, batch_y)

                self.model.params, batch_cost = self.optimizer.minimize(cost_fn, self.model.params)
                cost += batch_cost / len(batches)
            costs.append(cost)
            if self.debug and i % 100 == 0:
                print('cost[{}]: {}'.format(i, cost))
        return costs

    def _partition_batches(self, x, y, batch_size):
        m = x.shape[1]
        batches = []
        n_batches = math.floor(m / batch_size)
        permutation = list(np.random.permutation(m))
        shuffled_x = x[:, permutation]
        shuffled_y = y[:, permutation].reshape((1, m))
        for k in range(n_batches):
            batch_x = shuffled_x[:, k * batch_size : (k + 1) * batch_size]
            batch_y = shuffled_y[:, k * batch_size : (k + 1) * batch_size]
            batches.append((batch_x, batch_y))
        if m % batch_size != 0:
            batch_x = shuffled_x[:, n_batches * batch_size : m]
            batch_y = shuffled_y[:, n_batches * batch_size : m]
            batches.append((batch_x, batch_y))
        return batches


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

    def minimize(self, cost_fn, params):
        cost, grad = cost_fn(params)
        self.grad_velocity = self.beta_1 * self.grad_velocity + (1.0 - self.beta_1) * grad
        self.grad_squares = self.beta_2 * self.grad_squares + (1.0 - self.beta_2) * np.square(grad)
        params = params - (
            self.alpha * (self.grad_velocity / (np.sqrt(self.grad_squares) + self.epsilon))
        )
        return params, cost


def linear_forward(x, w, b):
    out = np.dot(w, x) + b
    cache = (x, w, b)
    return out, cache


def linear_backward(dout, cache):
    x, w, b = cache
    m = x.shape[1]
    dw = (1.0 / m) * np.dot(dout, x.T)
    # FIXME dw += (self.lambd / m) * w
    db = (1.0 / m) * np.sum(dout, axis=1, keepdims=True)
    dx = np.dot(w.T, dout)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = np.multiply(dout, np.int64(x > 0))
    return dx


def sigmoid_forward(x):
    out = 1.0 / (1.0 + np.exp(-x))
    cache = x
    return out, cache


def sigmoid_backward(dout, cache):
    x = cache
    a, _ = sigmoid_forward(x)
    dx = np.multiply(dout, a * (1 - a))
    return dx


def load_dataset(file_name, prefix):
    model = h5py.File(file_name, 'r')
    X = np.array(model[prefix + '_x'][:], dtype=np.float)
    X = X.reshape((X.shape[0], -1)).T
    X = X / 255
    Y = np.array(model[prefix + '_y'][:], dtype=np.int)
    Y = Y.reshape((1, Y.shape[0]))
    return (X, Y)


def main():
    # Dataset
    (train_x, train_y) = load_dataset('datasets/images_train.h5', 'train_set')
    (test_x, test_y) = load_dataset('datasets/images_test.h5', 'test_set')
    print('{} X{} Y{}'.format('train', train_x.shape, train_y.shape))
    print('{} X{} Y{}'.format('test', test_x.shape, test_y.shape))
    # Model
    model = NnModel(lambd=1.8)
    model.add(16, activation='relu', input_dim=train_x.shape[0], dropout=0)
    model.add(16, activation='relu', dropout=0)
    model.add(16, activation='relu', dropout=0)
    model.add(1, activation='sigmoid')
    model.compile()
    # Train
    optimizer = AdamOptimizer(alpha=0.0001)
    trainer = NnTrainer(model, optimizer, epochs=1000, batch_size=256, debug=True)
    start = time.time()
    costs = trainer.train(train_x, train_y)
    end = time.time()
    # Evaluate
    Yp_train = model.predict(train_x)
    Yp_test = model.predict(test_x)
    print(
        'train accuracy: {} %, took {}'.format(
            100 - np.mean(np.abs(Yp_train - train_y)) * 100, end - start
        )
    )
    print('test accuracy: {} %'.format(100 - np.mean(np.abs(Yp_test - test_y)) * 100))
    # Plot cost
    plt.plot(np.squeeze(costs))
    plt.title('Learning rate = {}'.format(optimizer.alpha))
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('cost')
    plt.show()


main()


# %%
