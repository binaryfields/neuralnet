# %%
# logistic_regression_tf2
#
# - eager execution
# - input dataset
# - custom model layers
# - custom training loop

import time
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_classes = 10
n_features = 28 * 28
n_test = 10000

# hyperparameters
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# model
class LrModel(object):
    def __init__(self, n_features, n_classes):
        self.w = tf.Variable(tf.zeros([n_features, n_classes]), name='weights')
        self.b = tf.Variable(tf.zeros([1, n_classes]), name='bias')
        self.trainable_variables = [self.w, self.b]

    def inference(self, x):
        logits = tf.matmul(x, self.w) + self.b
        return logits

    def loss(self, logits, labels):
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(entropy, name='loss')

    def evaluate(self, images, labels):
        preds = tf.nn.softmax(self.inference(images))
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


@tf.function
def train_step(optimizer, model, images, labels):
    with tf.GradientTape() as tape:
        logits = model.inference(images)
        loss = model.loss(logits, labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def main():
    # dataset
    (train_ds, test_ds) = load_dataset()
    (train_ds, test_ds) = (train_ds.batch(batch_size), test_ds.batch(batch_size))

    # graph
    model = LrModel(n_features, n_classes)

    # train
    optimizer = tf.optimizers.Adam(learning_rate)
    writer = tf.summary.create_file_writer(
        'graphs/logistic_regression/lr' + str(optimizer.learning_rate.numpy())
    )
    start_time = time.time()
    for step in range(n_epochs):
        print(f'epoch {step}')
        total_loss = 0
        for images, labels in train_ds:
            total_loss += train_step(optimizer, model, images, labels)
        with writer.as_default():
            tf.summary.scalar('loss', total_loss, step=step)
        writer.flush()
    end_time = time.time()
    print(f'total time: {end_time - start_time}s')
    writer.close()

    # evaluate
    correct = 0
    for images, labels in test_ds:
        correct += model.evaluate(images, labels)
    print(f'accuracy: {correct/n_test}')


if __name__ == '__main__':
    main()

# %%
