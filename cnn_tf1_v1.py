# %%
# cnn_tf1_v1
#
# - deferred execution
# - custom layers with tf.nn functions
# - custom training loop

import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf2

tf = tf2.compat.v1
tf.disable_v2_behavior()

# hyperparams
learning_rate = 0.001
n_epochs = 500


def conv2d_layer(
    x, filters, ksize, strides=(1, 1), padding='SAME', activation=None, scope=None
):
    with tf.variable_scope(scope):
        weights = tf.get_variable(
            'weights',
            shape=[ksize, ksize, x.shape[3], filters],
            dtype=tf.float32,
            initializer=tf2.keras.initializers.GlorotNormal(seed=0),
        )
        out = tf.nn.conv2d(
            x, weights, strides=[1, strides[0], strides[1], 1], padding=padding
        )
        return activation(out) if activation else out


def dense_layer(x, units, activation=None, scope=None):
    with tf.variable_scope(scope):
        weights = tf.get_variable(
            'weights',
            shape=[x.shape[1], units],
            dtype=tf.float32,
            initializer=tf2.keras.initializers.GlorotNormal(seed=0),
        )
        bias = tf.get_variable(
            'bias', shape=[1, units], dtype=tf.float32, initializer=tf.zeros_initializer()
        )
        out = tf.matmul(x, weights) + bias
        return activation(out) if activation else out


def flatten_layer(x):
    dim = np.prod(x.get_shape().as_list()[1:])
    return tf.reshape(x, [-1, dim])


def maxpool2d_layer(x, ksize):
    return tf.nn.max_pool(
        x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME'
    )


def model_fn(x, n_classes, mode):
    with tf.variable_scope('convnet'):
        x = conv2d_layer(x, 8, 4, activation=tf.nn.relu, scope='conv_1')
        x = maxpool2d_layer(x, 8)
        x = conv2d_layer(x, 16, 2, activation=tf.nn.relu, scope='conv_2')
        x = maxpool2d_layer(x, 4)
        x = flatten_layer(x)
        x = dense_layer(x, n_classes, scope='fc_3')
        return x


def load_dataset():
    for (file_name, prefix) in [
        ('images_train.h5', 'train_set'),
        ('images_test.h5', 'test_set'),
    ]:
        model = h5py.File(f'data/{file_name}', 'r')
        x = np.array(model[prefix + '_x'][:], dtype=np.float32)
        x = x / 255
        y = np.array(model[prefix + '_y'][:], dtype=np.int32)
        y = y.reshape((y.shape[0], 1))
        yield (x, y)


def main():
    # ops.reset_default_graph()
    tf.set_random_seed(1)
    # dataset
    (train_ds, test_ds) = load_dataset()
    print('Train X{} Y{}'.format(train_ds[0].shape, train_ds[1].shape))
    print('Test  X{} Y{}'.format(test_ds[0].shape, test_ds[1].shape))

    # inputs
    _, in_height, in_width, in_channels = train_ds[0].shape
    n_classes = train_ds[1].shape[1]
    features = tf.placeholder(
        tf.float32, [None, in_height, in_width, in_channels], name='features'
    )
    labels = tf.placeholder(tf.float32, [None, n_classes], name='labels')

    # graph
    optimizer = tf.train.AdamOptimizer(learning_rate)
    logits = model_fn(features, n_classes, mode='train')
    prediction = tf.greater(tf.nn.sigmoid(logits), tf.constant(0.5))
    correct_prediction = tf.equal(tf.cast(prediction, tf.float32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    train_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    # train
    costs = []
    with tf.Session() as sess:
        sess.run(init)
        start_time = time.time()
        for step in range(n_epochs):
            _, cost = sess.run(
                [train_op, loss], {features: train_ds[0], labels: train_ds[1]}
            )
            costs.append(cost)
            if (step + 1) % 100 == 0:
                print(f'cost[{step+1}]: {cost}')
        end_time = time.time()
        print(f'total time: {end_time - start_time}s')
        train_acc = sess.run(accuracy, {features: train_ds[0], labels: train_ds[1]})
        test_acc = sess.run(accuracy, {features: test_ds[0], labels: test_ds[1]})
        print(f'train accuracy: {train_acc * 100}%')
        print(f'test accuracy: {test_acc * 100}%')

    # summary
    plt.plot(np.squeeze(costs))
    plt.title('Learning rate')
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('cost')
    plt.show()


if __name__ == '__main__':
    main()


# %%
