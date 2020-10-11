# %%
# logistic_regression_raw.py
#
# - pure numpy impl
# - custom model
# - custom optimizer
# - custom training loop

import h5py
import matplotlib.pyplot as plt
import numpy as np
import time


# hyperparameters
learning_rate = 0.005
n_epochs = 2000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LrClassifier(object):
    def __init__(self, n_features, n_classes, lambd=0):
        self.lambd = lambd
        self.params = np.zeros((n_features + 1, n_classes), dtype=np.float32)

    def predict(self, x):
        w, b = self._unpack_params(self.params)
        assert x.shape[0] == w.shape[0]
        z = np.dot(w.T, x) + b
        a = sigmoid(z)
        predictions = a > 0.5
        assert predictions.shape == (1, x.shape[1])
        return predictions

    def loss(self, X, Y):
        # weights (n x 1), bias (scalar)
        w, b = self._unpack_params(self.params)
        assert w.shape == (X.shape[0], 1)
        # number of examples
        m = X.shape[1]
        # decision boundary (1 x m)
        Z = np.dot(w.T, X) + b
        # predictions (1 x m)
        A = sigmoid(Z)
        # loss (1 x m)
        L = -(Y * np.log(A) + (1 - Y) * np.log(1.0 - A))
        # cross-entropy cost (scalar)
        J = (1.0 / m) * np.sum(L)
        # regularized cost
        J += (self.lambd / (2 * m)) * np.sum(np.square(w))
        # dJ/dz (1 x m)
        dZ = A - Y
        # dJ/dw (n x 1) = X (n x m) dot dZ.T (m x 1)
        dw = (1.0 / m) * np.dot(X, dZ.T)
        dw += (self.lambd / m) * w
        assert dw.shape == w.shape
        # dJ/db
        db = (1.0 / m) * np.sum(dZ, axis=1).reshape(1, 1)
        # gradients ((n + 1) x 1)
        grad = np.concatenate([dw, db])
        assert grad.shape == self.params.shape
        return J, grad

    def _unpack_params(self, params):
        w = params[:-1, ...]
        b = params[-1, 0]
        return w, b


class GradDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, grads, params):
        return params - self.learning_rate * grads


def load_dataset():
    for (file_name, prefix) in [
        ('images_train.h5', 'train_set'),
        ('images_test.h5', 'test_set'),
    ]:
        model = h5py.File(f'data/{file_name}', 'r')
        x = np.array(model[prefix + '_x'][:], dtype=np.float32)
        x = x.reshape((x.shape[0], -1)).T
        x = x / 255
        y = np.array(model[prefix + '_y'][:], dtype=np.float32)
        y = y.reshape((1, y.shape[0]))
        yield (x, y)


def train_step(optimizer, model, x, y):
    loss, grads = model.loss(x, y)
    model.params = optimizer.apply_gradients(grads, model.params)
    return loss


def main():
    # data
    (train_ds, test_ds) = load_dataset()
    print('{} X{} Y{}'.format('train', train_ds[0].shape, train_ds[1].shape))
    print('{} X{} Y{}'.format('test', test_ds[0].shape, test_ds[1].shape))
    n_features = train_ds[0].shape[0]
    n_classes = train_ds[1].shape[0]

    # train
    optimizer = GradDescentOptimizer(learning_rate)
    model = LrClassifier(n_features, n_classes)
    losses = []

    start_time = time.time()
    for step in range(n_epochs):
        loss = train_step(optimizer, model, train_ds[0], train_ds[1])
        if (step + 1) % 100 == 0:
            print(f'cost[{step+1}]: {loss}')
        losses.append(loss)
    end_time = time.time()
    print(f'total time: {end_time - start_time}s')

    # evaluate
    Yp_train = model.predict(train_ds[0])
    Yp_test = model.predict(test_ds[0])
    print(
        'train accuracy: {} %'.format(100 - np.mean(np.abs(Yp_train - train_ds[1])) * 100)
    )
    print('test accuracy: {} %'.format(100 - np.mean(np.abs(Yp_test - test_ds[1])) * 100))

    # summary
    plt.plot(np.squeeze(losses))
    plt.title('Learning rate')
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    main()


# %%
