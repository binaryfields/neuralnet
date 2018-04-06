#%%
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def model_net(x, n_classes):
    with tf.variable_scope('convnet'):
        conv_1 = tf.layers.conv2d(
            x,
            filters=8,
            kernel_size=4,
            activation=tf.nn.relu,
            padding='SAME',
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
            use_bias=False)
        conv_1 = tf.layers.max_pooling2d(conv_1, 8, 8, padding='SAME')
        conv_2 = tf.layers.conv2d(
            conv_1,
            filters=16,
            kernel_size=2,
            activation=tf.nn.relu,
            padding='SAME',
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
            use_bias=False)
        conv_2 = tf.layers.max_pooling2d(conv_2, 4, 4, padding='SAME')
        flatten_3 = tf.layers.flatten(conv_2)
        out = tf.layers.dense(
            flatten_3,
            units=n_classes,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
        return out


def model_fn(features, labels, mode):
    logits = model_net(features, labels.shape[1])
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, tf.float32)))
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
        train_x, train_y, batch_size=128, shuffle=False, num_epochs=None)
    model = tf.estimator.Estimator(model_fn)
    model.train(input_fn, steps=500)
    # Evaluate the model.
    input_fn = tf.estimator.inputs.numpy_input_fn(
        test_x, test_y, batch_size=256, shuffle=False)
    metrics = model.evaluate(input_fn)
    print('Test accuracy: {accuracy:0.3f}'.format(**metrics))


main()
