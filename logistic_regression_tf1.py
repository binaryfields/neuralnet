# %%
# logistic_regression_tf1
#
# - deferred execution
# - input dataset
# - custom model layers
# - custom training loop

import time
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.disable_v2_behavior()

# hyperparameters
learning_rate = 0.01
batch_size = 128
n_classes = 10
n_epochs = 30
n_features = 28 * 28
n_test = 10000


class LrModel(object):
    def __init__(self):
        self.w = tf.get_variable(
            'weights',
            shape=[n_features, n_classes],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0, 0.01),
        )
        self.b = tf.get_variable(
            'bias', shape=[1, n_classes], dtype=tf.float32, initializer=tf.zeros_initializer()
        )

    def inference(self, x):
        logits = tf.matmul(x, self.w) + self.b
        return logits

    def loss(self, logits, labels):
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(entropy, name='loss')

    def evaluation(self, logits, labels):
        preds = tf.nn.softmax(logits)
        correct = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
        return accuracy


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255
    y_train = tf.one_hot(y_train, n_classes)
    y_test = tf.one_hot(y_test, n_classes)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return (train_data, test_data)


def main():
    # dataset
    (train_ds, test_ds) = load_dataset()
    (train_ds, test_ds) = (train_ds.batch(batch_size), test_ds.batch(batch_size))

    iterator = tf.data.Iterator.from_structure(
        tf.data.get_output_types(train_ds), tf.data.get_output_shapes(train_ds)
    )
    images, labels = iterator.get_next()
    print(f'images: {images.shape}, labels: {labels.shape}')

    # graph
    optimizer = tf.train.AdamOptimizer(learning_rate)
    model = LrModel()
    logits = model.inference(images)
    loss = model.loss(logits, labels)
    accuracy = model.evaluation(logits, labels)
    train_op = optimizer.minimize(loss)

    # train
    init = tf.global_variables_initializer()
    train_init = iterator.make_initializer(train_ds)
    test_init = iterator.make_initializer(test_ds)
    losses = []

    with tf.Session() as sess:
        sess.run(init)
        # train
        start_time = time.time()
        for step in range(n_epochs):
            print(f'epoch {step}')
            sess.run(train_init)
            total_loss = 0
            try:
                while True:
                    _, l = sess.run([train_op, loss])
                    total_loss += l
            except tf.errors.OutOfRangeError:
                pass
            losses.append(total_loss)
        end_time = time.time()
        print(f'total time: {end_time - start_time}s')
        # evaluate
        sess.run(test_init)
        correct = 0
        try:
            while True:
                correct += sess.run(accuracy)
        except tf.errors.OutOfRangeError:
            pass
        print(f'accuracy: {correct/n_test}')

    # summary
    plt.plot(np.squeeze(losses))
    plt.title('Learning rate')
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    main()


# %%
