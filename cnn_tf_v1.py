#%%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import ops


def conv2d_layer(x,
                 filters,
                 ksize,
                 strides=(1, 1),
                 padding='SAME',
                 activation=None,
                 scope=None):
    with tf.variable_scope(scope):
        weights = tf.get_variable(
            'weights',
            shape=[ksize, ksize, x.shape[3], filters],
            initializer=tf.contrib.layers.xavier_initializer(seed=0))
        out = tf.nn.conv2d(
            x,
            weights,
            strides=[1, strides[0], strides[1], 1],
            padding=padding)
        return activation(out) if activation else out


def dense_layer(x, units, activation=None, scope=None):
    with tf.variable_scope(scope):
        weights = tf.get_variable(
            'weights',
            shape=[x.shape[1], units],
            initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias = tf.get_variable(
            'bias', shape=[1, units], initializer=tf.zeros_initializer())
        out = tf.add(tf.matmul(x, weights), bias)
        return activation(out) if activation else out


def flatten_layer(x):
    dim = np.prod(x.get_shape().as_list()[1:])
    return tf.reshape(x, [-1, dim])


def maxpool2d_layer(x, k):
    return tf.nn.max_pool(
        x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def model_fn(features, labels, mode):
    with tf.variable_scope('convnet'):
        conv_1 = conv2d_layer(
            features, 8, 4, activation=tf.nn.relu, scope='conv_1')
        conv_1 = maxpool2d_layer(conv_1, 8)
        conv_2 = conv2d_layer(
            conv_1, 16, 2, activation=tf.nn.relu, scope='conv_2')
        conv_2 = maxpool2d_layer(conv_2, 4)
        flatten_3 = flatten_layer(conv_2)
        out = dense_layer(flatten_3, labels.shape[1], scope='fc_3')
        return out


def load_dataset(file_name, prefix):
    model = h5py.File(file_name, 'r')
    X = np.array(model[prefix + '_x'][:], dtype=np.float)
    # X = X.reshape((X.shape[0], -1)).T
    X = X / 255
    Y = np.array(model[prefix + '_y'][:], dtype=np.int)
    Y = Y.reshape((Y.shape[0], 1))
    return (X, Y)


def main():
    ops.reset_default_graph()
    tf.set_random_seed(1)
    # Dataset
    (train_x, train_y) = load_dataset('datasets/images_train.h5', 'train_set')
    (test_x, test_y) = load_dataset('datasets/images_test.h5', 'test_set')
    print('{} X{} Y{}'.format('train', train_x.shape, train_y.shape))
    print('{} X{} Y{}'.format('test', test_x.shape, test_y.shape))
    # Input
    _, in_height, in_width, in_channels = train_x.shape
    n_classes = train_y.shape[1]
    features = tf.placeholder(tf.float32, [None, in_height, in_width, in_channels])
    labels = tf.placeholder(tf.float32, [None, n_classes])
    # Model
    logits = model_fn(features, labels, mode='train')
    prediction = tf.greater(tf.nn.sigmoid(logits), tf.constant(0.5))
    # Eval
    correct_prediction = tf.equal(tf.cast(prediction, tf.float32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Train
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = optimizer.minimize(loss)
    # Run
    init = tf.global_variables_initializer()
    costs = []
    with tf.Session() as sess:
        sess.run(init)
        start = time.time()
        for s in range(500):
            _, cost = sess.run(
                [train_op, loss],
                feed_dict={
                    features: train_x,
                    labels: train_y
                })
            costs.append(cost)
            if s % 100 == 0:
                print('cost[{}]: {}'.format(s, cost))
        end = time.time()
        train_acc = sess.run(accuracy, feed_dict={features: train_x, labels: train_y})
        test_acc = sess.run(accuracy, feed_dict={features: test_x, labels: test_y})
        print('train accuracy: {} %, took {}'.format(train_acc * 100,
                                                     end - start))
        print('test accuracy: {} %'.format(test_acc * 100))
    # Plot cost
    plt.plot(np.squeeze(costs))
    plt.title('Learning rate = {}'.format(1))
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('cost')
    plt.show()


main()
