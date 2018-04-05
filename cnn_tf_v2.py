#%%
import h5py
import numpy as np
import tensorflow as tf
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
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def maxpool2d_layer(x, k):
    return tf.nn.max_pool(
        x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def model_net(x, n_classes):
    with tf.variable_scope('convnet'):
        conv_1 = conv2d_layer(x, 8, 4, activation=tf.nn.relu, scope='conv_1')
        conv_1 = maxpool2d_layer(conv_1, 8)
        conv_2 = conv2d_layer(
            conv_1, 16, 2, activation=tf.nn.relu, scope='conv_2')
        conv_2 = maxpool2d_layer(conv_2, 4)
        flatten_3 = flatten_layer(conv_2)
        out = dense_layer(flatten_3, n_classes, scope='fc_3')
        return out


def model_fn(features, labels, mode):
    logits = model_net(features['images'], labels.shape[1])
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32)))
    predictions = tf.greater(tf.nn.sigmoid(logits), tf.constant(0.5))
    # Build estimator
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    elif mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops={'accuracy': accuracy})
    else:
        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def load_dataset(file_name, prefix):
    model = h5py.File(file_name, 'r')
    X = np.array(model[prefix + '_x'][:], dtype=np.float32)
    # X = X.reshape((X.shape[0], -1)).T
    X = X / 255
    Y = np.array(model[prefix + '_y'][:], dtype=np.int32)
    Y = Y.reshape((Y.shape[0], 1))
    return (X, Y)


def main():
    ops.reset_default_graph()
    tf.set_random_seed(1)
    # Dataset
    (train_x, train_y) = load_dataset('datasets/images_train.h5', 'train_set')
    (test_x, test_y) = load_dataset('datasets/images_test.h5', 'test_set')
    print('Train X{} Y{}'.format(train_x.shape, train_y.shape))
    print('Test  X{} Y{}'.format(test_x.shape, test_y.shape))
    # Train the model.
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {'images': train_x}, train_y, batch_size=128, shuffle=False, num_epochs=None)
    model = tf.estimator.Estimator(model_fn)
    model.train(input_fn, steps=500)
    # Evaluate the model.
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {'images': test_x}, test_y, batch_size=256, shuffle=False)
    metrics = model.evaluate(input_fn)
    print('Test accuracy: {accuracy:0.3f}'.format(**metrics))


main()
