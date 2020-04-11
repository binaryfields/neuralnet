#%%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from collections import namedtuple
from tensorflow.python.framework import ops

Layer = namedtuple('Layer', ['kernel', 'weights', 'biases'])


class Model:
    def __init__(self, n_features, n_labels, lambd=0.0):
        self.inputs = tf.compat.v1.placeholder(tf.float32, shape=[n_features, None], name='inputs')
        self.labels = tf.compat.v1.placeholder(tf.float32, shape=[n_labels, None], name='labels')
        self.lambd = tf.constant(lambd, dtype=tf.float32)
        self.logits = None
        self.loss = None
        self.accuracy = None
        self._layers = []
        self._layers.append(Layer(self.inputs, None, None))

    def add_dense_layer(self, units, activation):
        layer_id = len(self._layers)
        inputs = self._layers[layer_id - 1].kernel
        weights = tf.compat.v1.get_variable(
            'W{}'.format(layer_id),
            shape=(units, inputs.shape[0]),
            initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform", seed=1
            ),
        )
        biases = tf.compat.v1.get_variable(
            'b{}'.format(layer_id), shape=(units, 1), initializer=tf.compat.v1.zeros_initializer()
        )
        z = tf.add(tf.matmul(weights, inputs), biases)
        kernel = activation(z) if activation else z
        self._layers.append(Layer(kernel, weights, biases))

    def compile(self):
        self.inputs = self._layers[0].kernel
        self.logits = self._layers[-1].kernel
        self.loss = self._compute_loss(self.logits, self.labels)
        self.accuracy = self._compute_accuracy(self.logits, self.labels)

    def evaluate(self, session, features, labels):
        return session.run(self.accuracy, feed_dict={self.inputs: features, self.labels: labels})

    def predict(self, session, features):
        return session.run(
            tf.greater(tf.nn.sigmoid(self.logits), tf.constant(0.5)),
            feed_dict={self.inputs: features},
        )

    def _compute_accuracy(self, logits, labels):
        predictions = tf.greater(tf.nn.sigmoid(logits), tf.constant(0.5))
        correct_prediction = tf.equal(tf.cast(predictions, tf.float32), labels)
        return tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

    def _compute_loss(self, logits, labels):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.transpose(a=logits), labels=tf.transpose(a=labels)
        )
        l2_losses = [tf.nn.l2_loss(layer.weights) for layer in self._layers[1:-1]]
        return tf.add(
            tf.reduce_mean(input_tensor=loss), tf.multiply(self.lambd, tf.add_n(l2_losses))
        )


class Trainer:
    def __init__(self, model, optimizer, debug=False):
        self._model = model
        self._optimizer = optimizer
        self._debug = debug
        self._train_op = self._optimizer.minimize(
            self._model.loss, global_step=tf.compat.v1.train.get_global_step()
        )

    def train(self, session, features, labels, steps):
        costs = []
        for s in range(steps):
            _, cost = session.run(
                [self._train_op, self._model.loss],
                feed_dict={self._model.inputs: features, self._model.labels: labels},
            )
            costs.append(cost)
            if self._debug and s % 100 == 0:
                print('cost[{}]: {}'.format(s, cost))
        return costs


def load_dataset(file_name, prefix):
    model = h5py.File(file_name, 'r')
    X = np.array(model[prefix + '_x'][:], dtype=np.float)
    X = X.reshape((X.shape[0], -1)).T
    X = X / 255
    Y = np.array(model[prefix + '_y'][:], dtype=np.float)
    Y = Y.reshape((1, Y.shape[0]))
    return (X, Y)


def main():
    ops.reset_default_graph()
    tf.compat.v1.set_random_seed(1)
    # Dataset
    (train_x, train_y) = load_dataset('datasets/images_train.h5', 'train_set')
    (test_x, test_y) = load_dataset('datasets/images_test.h5', 'test_set')
    print('{} X{} Y{}'.format('train', train_x.shape, train_y.shape))
    print('{} X{} Y{}'.format('test', test_x.shape, test_y.shape))
    # Model
    model = Model(train_x.shape[0], train_y.shape[0], lambd=(2.5 / train_x.shape[1]))
    model.add_dense_layer(50, activation=tf.nn.relu)
    model.add_dense_layer(30, activation=tf.nn.relu)
    model.add_dense_layer(10, activation=tf.nn.relu)
    model.add_dense_layer(1, activation=None)
    model.compile()
    # Train
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
    trainer = Trainer(model, optimizer, debug=True)
    init = tf.compat.v1.global_variables_initializer()
    costs = []
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        start = time.time()
        costs = trainer.train(sess, train_x, train_y, 500)
        end = time.time()
        train_acc = model.evaluate(sess, train_x, train_y)
        test_acc = model.evaluate(sess, test_x, test_y)
        print('train accuracy: {} %, took {}'.format(train_acc * 100, end - start))
        print('test accuracy: {} %'.format(test_acc * 100))
    # Plot cost
    plt.plot(np.squeeze(costs))
    plt.title('Learning rate = {}'.format(1))
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('cost')
    plt.show()


main()
