#%%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class GradDescentOptimizer:
    def __init__(self, iters, alpha, debug=False):
        self.alpha = alpha
        self.debug = debug
        self.iters = iters

    def optimize(self, cost_fn, params):
        costs = []
        for i in range(self.iters):
            cost, grad = cost_fn(params)
            params = params - self.alpha * grad
            costs.append(cost)
            if self.debug and i % 100 == 0:
                print('cost[{}]: {}'.format(i, cost))
        return params, costs


class LrClassifier:
    def __init__(self, optimizer, lambd = 0):
        self.optimizer = optimizer
        self.lambd = lambd
        self.params = None

    def init_params(self, n):
        self.params = np.zeros((n + 1, 1), dtype=np.float)

    def predict(self, X):
        w, b = self._unpack_params(self.params)
        assert X.shape[0] == w.shape[0]
        Z = np.dot(w.T, X) + b
        A = sigmoid(Z)
        Y_pred = A > 0.5
        assert Y_pred.shape == (1, X.shape[1])
        return Y_pred

    def train(self, X, Y):
        cost_fn = lambda params: self._cost(params, X, Y)
        self.params, costs = self.optimizer.optimize(cost_fn, self.params)
        return costs

    def _cost(self, params, X, Y):
        # weights (n x 1), bias (scalar)
        w, b = self._unpack_params(params)
        assert w.shape == (X.shape[0], 1)
        # number of examples
        m = X.shape[1]
        # decision boundary (1 x m)
        Z = np.dot(w.T, X) + b
        # predictions (1 x m)
        A = sigmoid(Z)
        # loss (1 x m)
        L = -(Y * np.log(A) + (1 - Y) * np.log(1. - A))
        # cross-entropy cost (scalar)
        J = (1. / m) * np.sum(L)
        # regularized cost
        J += (self.lambd / (2 * m)) * np.sum(np.square(w))
        # dJ/dz (1 x m)
        dZ = A - Y
        # dJ/dw (n x 1) = X (n x m) dot dZ.T (m x 1)
        dw = (1. / m) * np.dot(X, dZ.T)
        dw += (self.lambd / m) * w
        assert dw.shape == w.shape
        # dJ/db
        db = (1. / m) * np.sum(dZ, axis=1).reshape(1, 1)
        # gradients ((n + 1) x 1)
        grad = np.concatenate([dw, db])
        assert grad.shape == params.shape
        return J, grad

    def _unpack_params(self, params):
        w = params[:-1, ...]
        b = params[-1, 0]
        return w, b


Dataset = namedtuple('Dataset', ['X', 'Y'])


def load_dataset(file_name, prefix):
    model = h5py.File(file_name, 'r')
    X = np.array(model[prefix + '_x'][:], dtype=np.float)
    X = X.reshape((X.shape[0], -1)).T
    X = X / 255
    Y = np.array(model[prefix + '_y'][:], dtype=np.float)
    Y = Y.reshape((1, Y.shape[0]))
    return Dataset(X, Y)


def main():
    # Dataset
    ds_train = load_dataset('datasets/images_train.h5', 'train_set')
    ds_test = load_dataset('datasets/images_test.h5', 'test_set')
    print('{} X{} Y{}'.format('train', ds_train.X.shape, ds_train.Y.shape))
    print('{} X{} Y{}'.format('test', ds_test.X.shape, ds_test.Y.shape))
    # Train
    optimizer = GradDescentOptimizer(iters=2000, alpha=0.005, debug=True)
    classifier = LrClassifier(optimizer)
    classifier.init_params(ds_train.X.shape[0])
    start = time.time()
    costs = classifier.train(ds_train.X, ds_train.Y)
    end = time.time()
    # Evaluate
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
