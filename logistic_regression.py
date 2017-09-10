#%%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple


class GradDescentOptimizer:
    def __init__(self):
        self.alpha = 0.005
        self.debug = True
        self.iters = 2000

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


class LrClassifier:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.params = None
        self.lambdap = 0.0

    def predict(self, X):
        w, b = self._params_unpack(self.params)
        assert X.shape[0] == w.shape[0]
        Z = np.dot(w.T, X) + b
        A = self._sigmoid(Z)
        Y_pred = A > 0.5
        assert Y_pred.shape == (1, X.shape[1])
        return Y_pred

    def train(self, X, Y):
        cost_fn = lambda params: self._propagate(params, X, Y)
        guess = self._params_init(X.shape[0])
        self.params, costs = self.optimizer.optimize(cost_fn, guess)
        return costs

    def _params_init(self, n):
        return np.zeros((n + 1, 1), dtype=np.float32)

    def _params_unpack(self, params):
        w = params[:-1]
        b = params[-1, 0]
        return w, b

    def _propagate(self, params, X, Y):
        # weights (n x 1), bias (scalar)
        w, b = self._params_unpack(params)
        assert w.shape == (X.shape[0], 1)
        # number of examples
        m = X.shape[1]
        # decision boundary (1 x m)
        Z = np.dot(w.T, X) + b
        # predictions (1 x m)
        A = self._sigmoid(Z)
        # loss (1 x m)
        L = -(Y * np.log(A) + (1 - Y) * np.log(1. - A))
        # cost (scalar)
        J = (1. / m) * np.sum(L)
        # regularized cost
        J += (self.lambdap / (2 * m)) * np.sum(np.square(w))
        # dJ/dz (1 x m)
        dZ = A - Y
        # dJ/dw (n x 1) = X (n x m) dot dZ.T (m x 1)
        dw = (1. / m) * np.dot(X, dZ.T)
        dw += (self.lambdap / m) * w
        assert dw.shape == w.shape
        # dJ/db
        db = (1. / m) * np.sum(dZ, axis=1).reshape(1, 1)
        # gradients ((n + 1) x 1)
        grad = np.concatenate([dw, db])
        assert grad.shape == params.shape
        return np.squeeze(J), grad

    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-z))


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
    classifier = LrClassifier(GradDescentOptimizer())
    classifier.lambdap = 0.0
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
